"""Compiled TTA lookup equivalence, fallback routing and lazy worker loading."""
import builtins
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing

import numpy as np
import pandas as pd
import pytest

import pc_xpass_acceleration as motion
import physical_pass_model as physics
from test_pc_xpass_acceleration import table
from test_pc_xpass_blocking import bundle
from test_physical_xpass import make_graph


def adaptive_table(partial=False):
    # Different split axes and discontinuous leaf values expose wrong traversal
    # at adaptive faces. Internal nodes deliberately carry fallback=True.
    low = np.array([[0, 0, 0], [0, 0, 0], [0, math.pi/2, 0],
                    [0, 0, 0], [.5, 0, 0], [0, math.pi/2, 0], [0, math.pi/2, 2.]], float)
    high = np.array([[1, math.pi, 4], [1, math.pi/2, 4], [1, math.pi, 4],
                     [.5, math.pi/2, 4], [1, math.pi/2, 4], [1, math.pi, 2], [1, math.pi, 4]], float)
    children = np.array([[1, 2], [3, 4], [5, 6], [-1, -1], [-1, -1], [-1, -1], [-1, -1]])
    values = np.random.default_rng(17).normal(size=(7, 8))
    fallback = np.array([True, True, True, False, False, False, False])
    if partial:
        fallback[[3, 6]] = True
        values[[3, 6]] = np.nan
    return dict(low=low, high=high, children=children, axis=np.array([1, 0, 2, -1, -1, -1, -1]),
                values=values, fallback=fallback)


@pytest.mark.parametrize('partial', [False, True])
def test_random_queries_boundaries_corners_and_nonfinite(partial):
    arrays = adaptive_table(partial)
    rng = np.random.default_rng(123)
    random = rng.uniform([-.1, -.1, -.1], [1.1, math.pi+.1, 4.1], size=(2000, 3))
    corners = np.concatenate([lo+(hi-lo)*motion._CORNERS for lo, hi in zip(arrays['low'], arrays['high'])])
    splits = np.array([[.5, .5, 1.], [np.nextafter(.5, 0), .5, 1.],
                       [.5, math.pi/2, 2.], [.5, math.pi/2, np.nextafter(2., 0.)]])
    nonfinite = np.array([[np.nan, 0, 0], [0, np.inf, 0], [0, 0, -np.inf]])
    queries = np.vstack([random, corners, splits, nonfinite])
    expected, expected_mask = motion._lookup_arrival_corrections_numpy(queries, arrays)
    actual, actual_mask = motion._lookup_arrival_corrections(queries, arrays)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0.)
    assert not actual_mask[-3:].any()


def test_multilinear_interpolant_and_midpoint_side():
    arrays = adaptive_table()
    for i in range(3, 7):
        arrays['values'][i] = i
    queries = np.array([[.5, .5, 1.], [np.nextafter(.5, 0), .5, 1.],
                        [.5, math.pi/2, 2.], [.5, math.pi/2, np.nextafter(2., 0.)]])
    result, valid = motion._lookup_arrival_corrections(queries, arrays)
    assert valid.all()
    np.testing.assert_allclose(result, [4., 3., 6., 5.], atol=1e-14)
    def polynomial(p):
        x, y, z = p.T
        return 2+x+2*y+3*z+x*y-2*y*z+.5*x*z+3*x*y*z
    for i in range(3, 7):
        points = arrays['low'][i]+(arrays['high'][i]-arrays['low'][i])*motion._CORNERS
        arrays['values'][i] = polynomial(points)
    queries = np.random.default_rng(3).uniform([0, 0, 0], [1, math.pi, 4], size=(100, 3))
    result, valid = motion._lookup_arrival_corrections(queries, arrays)
    assert valid.all()
    np.testing.assert_allclose(result, polynomial(queries), rtol=1e-13, atol=1e-13)


def test_empty_lookups_do_not_load_compiler(monkeypatch):
    monkeypatch.setattr(motion, '_compiled_kernels', lambda: pytest.fail('Empty lookup initialized compiler'))
    for fn in (motion._lookup_arrival_corrections, motion._lookup_arrival_corrections_numpy):
        values, valid = fn(np.empty((0, 3)), adaptive_table())
        assert values.shape == valid.shape == (0,)


def test_kernel_and_native_signature_reused():
    arrays = adaptive_table()
    query = np.array([[.3, .2, 1.]])
    motion._lookup_arrival_corrections(query, arrays)
    kernels = motion._compiled_kernels()
    signatures = tuple(kernels.arrival_corrections.signatures)
    assert signatures
    motion._lookup_arrival_corrections(query, arrays)
    assert motion._compiled_kernels() is kernels
    assert tuple(kernels.arrival_corrections.signatures) == signatures


@pytest.mark.parametrize('missing', ['numba', 'llvmlite.binding'])
def test_missing_compiler_has_clear_error(monkeypatch, missing):
    original = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == 'pc_xpass_acceleration_numba':
            raise ModuleNotFoundError('missing compiler', name=missing)
        return original(name, *args, **kwargs)
    motion._compiled_kernels.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', unavailable)
        with pytest.raises(RuntimeError, match='TTA lookup requires Numba.*requirements'):
            motion._lookup_arrival_corrections(np.array([[.2, .3, 1.]]), adaptive_table())
    motion._compiled_kernels.cache_clear()


def mixed_table(root):
    np.savez_compressed(root/'table.npz', low=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2.]], float),
        high=np.array([[1, math.pi, 4], [1, math.pi, 2], [1, math.pi, 4]], float),
        children=np.array([[1, 2], [-1, -1], [-1, -1]]), axis=np.array([2, -1, -1]),
        values=np.array([[0.]*8, [0.]*8, [np.nan]*8]), fallback=np.array([True, False, True]))
    metadata = json.loads((root/'metadata.json').read_text())
    metadata['fingerprint'] = motion._digest(root/'table.npz')
    (root/'metadata.json').write_text(json.dumps(metadata))


def test_mixed_batch_solver_only_for_untrusted_queries(table, monkeypatch):
    mixed_table(table)
    config = motion.configuration(7, 'fixture')
    original = motion.solve_arrival
    calls = []
    def reference(*args, **kwargs):
        calls.append(np.asarray(args[2]).tolist())
        return original(*args, **kwargs)
    monkeypatch.setattr(motion, 'solve_arrival', reference)
    before = dict(motion.COUNTERS)
    result = motion.arrival_times(np.array([[0., 0., 5., 0.]]), np.array([1., 100., 1000.]),
                                  np.zeros(3), [5.], config)
    np.testing.assert_allclose(result, [[.2, 20., 200.]], atol=1e-12)
    assert calls == [[100., 0.], [1000., 0.]]
    assert motion.COUNTERS['queries']-before['queries'] == 3
    assert motion.COUNTERS['fallbacks']-before['fallbacks'] == 2
    def unresolved(*args, **kwargs):
        raise motion.UnresolvedArrival('test unresolved')
    monkeypatch.setattr(motion, 'solve_arrival', unresolved)
    with pytest.raises(motion.UnresolvedArrival, match='test unresolved'):
        motion.arrival_times(np.array([[0., 0., 5., 0.]]), np.array([100.]), np.zeros(1), [5.], config)


def test_analytic_masked_and_cache_only_paths_need_no_compiler(table, monkeypatch, tmp_path):
    config = motion.configuration(7, 'fixture')
    monkeypatch.setattr(motion, '_compiled_kernels', lambda: pytest.fail('Analytic/cache-only path initialized compiler'))
    result = motion.arrival_times(np.array([[0., 0., 0., 0.]]), np.array([0., 20.]), np.zeros(2), [5.], config)
    np.testing.assert_allclose(result, [[0., 4+5/14]])
    result = motion.arrival_times(np.array([[0., 0., 3., 4.]]), np.array([20.]), np.zeros(1), [5.], config, valid=np.array([False]))
    np.testing.assert_array_equal(result, [[0.]])
    point, record = motion.reference_point(([0., 0., math.log1p(2.)], .005, .001))
    assert 'unresolved' not in record  # Preparation's reference solver does not use lookup.
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                                        reaction_time=0., acceleration_config=config)
    cache = tmp_path/'predictions';cache.mkdir()
    (cache/'metadata.json').write_text(json.dumps(metadata))
    lookup = physics.physical_xpass_inference_lookup_config({'pc_xpass': True}, cache_dir=cache)
    assert lookup['acceleration_config'] == config


def test_inside_disk_needs_no_compiler(bundle, monkeypatch):
    config = motion.configuration(7, 'disk', max_speeds=(8., 5.), blocking_radius=1.)
    monkeypatch.setattr(motion, '_compiled_kernels', lambda: pytest.fail('Inside disk initialized compiler'))
    result = motion.arrival_times(np.array([[0., 0., 8., 0.]]), np.array([.5]), np.zeros(1), [8.], config)
    np.testing.assert_array_equal(result, [[0.]])


def _worker_lookup(queries, arrays):
    result = motion._lookup_arrival_corrections(queries, arrays)
    return result, bool(motion._compiled_kernels().arrival_corrections.signatures)


def test_spawn_compiles_and_preserves_partial_table():
    arrays = adaptive_table(True)
    query = np.array([[.2, .3, 1.], [.8, .3, 1.], [1., math.pi, 4.], [2., 0., 0.]])
    expected = motion._lookup_arrival_corrections_numpy(query, arrays)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        (values, valid), native = pool.submit(_worker_lookup, query, arrays).result()
    assert native
    np.testing.assert_allclose(values, expected[0], atol=1e-13)
    np.testing.assert_array_equal(valid, expected[1])


@pytest.mark.parametrize('radius', [0., 1.])
def test_complete_predictions_and_metadata_match(table, bundle, monkeypatch, radius):
    cap = 5. if not radius else 8.
    config = motion.configuration(7, 'disk' if radius else 'fixture', max_speeds=(cap, 5.), blocking_radius=radius)
    options = dict(reaction_time=0., acceleration_config=config, blocking_radius=radius,
                   max_player_speed_off=cap, max_player_speed_def=5., angle_step=90., radial_gridsize=25.,
                   min_speed=15., max_speed=15., lane_inflection_point=0., control_inflection_point=0.)
    before = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, **{
        k:v for k,v in options.items() if k != 'acceleration_config'}, acceleration_config=config)
    compiled = physics.compute_graph_pc_xpass_metrics(make_graph(), **options)
    monkeypatch.setattr(motion, '_lookup_arrival_corrections', motion._lookup_arrival_corrections_numpy)
    original = physics.compute_graph_pc_xpass_metrics(make_graph(), **options)
    pd.testing.assert_series_equal(compiled, original, check_exact=False, rtol=1e-12, atol=1e-12)
    after = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, **{
        k:v for k,v in options.items() if k != 'acceleration_config'}, acceleration_config=config)
    assert before == after
