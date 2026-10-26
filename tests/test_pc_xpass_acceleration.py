"""Physical constraints, first-arrival bounds, and acceleration cache contracts."""
from argparse import Namespace
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

import pc_xpass_acceleration as motion
import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate
from scripts import prepare_pc_xpass_acceleration as prepare
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph, make_label


@pytest.fixture
def table(tmp_path, monkeypatch):
    """Synthetic interpolation data for plumbing tests, not a physical reference."""
    monkeypatch.setattr(motion, "ROOT", tmp_path/"tables")
    root = motion.table_path("fixture")
    root.mkdir(parents=True)
    np.savez_compressed(root/"table.npz", low=np.array([[0., 0., 0.]]),
                        high=np.array([[1., math.pi, 8.]]), values=np.zeros((1, 8)),
                        children=np.array([[-1, -1]]), axis=np.array([-1]), fallback=np.array([False]))
    meta = dict(status="complete", movement_version=motion.VERSION,
                fingerprint=motion._digest(root/"table.npz"),
                time_error_normalized=.014, position_tolerance_normalized=.0028)
    (root/"metadata.json").write_text(json.dumps(meta))
    return root


def check_trajectory(result, initial, target, a, cap):
    assert 0 <= result.upper-result.lower <= .01+1e-9
    assert result.position_error <= .01
    np.testing.assert_allclose(result.velocities[0], initial, atol=1e-7)
    assert np.max(np.linalg.norm(result.velocities, axis=1)) <= cap+1e-6
    if len(result.times) > 1:
        dt = np.diff(result.times)
        acc = np.diff(result.velocities, axis=0)/dt[:, None]
        assert np.max(np.linalg.norm(acc, axis=1)) <= a+1e-5
        end = np.sum(dt[:, None]*(result.velocities[:-1]+result.velocities[1:])*.5, axis=0)
        np.testing.assert_allclose(end, target, atol=.01)


@pytest.mark.parametrize("v,q", [([0., 0.], [20., 0.]), ([-5., 0.], [20., 0.]),
                                 ([0., 5.], [20., 0.]), ([3., 4.], [50., 20.]),
                                 ([3., 4.], [.01, 0.]), ([3., 4.], [0., 0.])])
def test_reference_trajectory(v, q):
    result = motion.solve_arrival([0., 0.], v, q, 7., 5.)
    check_trajectory(result, v, q, 7., 5.)


def test_exact_standing_reversal_and_clipping():
    assert motion.solve_arrival([0, 0], [0, 0], [20, 0], 7, 5).time == pytest.approx(20/5+5/14)
    assert motion.solve_arrival([0, 0], [-5, 0], [20, 0], 7, 5).time == pytest.approx(20/5+10/7)
    left = motion.solve_arrival([0, 0], [30, 40], [20, 5], 7, 5)
    right = motion.solve_arrival([0, 0], [3, 4], [20, 5], 7, 5)
    assert left.time == pytest.approx(right.time, abs=.01)
    np.testing.assert_allclose(left.velocities[0], [3, 4])


def test_fixed_time_reachability_is_not_monotonic():
    at_first, _, _ = motion._fixed_time(np.array([1., 0.]), np.array([.1, 0.]), .1, 32)
    later, lower, _ = motion._fixed_time(np.array([1., 0.]), np.array([.1, 0.]), .2, 32)
    assert at_first < 1e-7
    assert later > .07
    assert 0 < lower <= later+1e-8


def test_rotation_translation_reflection_and_scaling():
    v, q = np.array([2., 3.]), np.array([20., -7.])
    original = motion.solve_arrival([0, 0], v, q, 7, 5)
    matrix = np.array([[0., -1.], [-1., 0.]])
    offset = np.array([10., 50.])
    transformed = motion.solve_arrival(offset, matrix@v, offset+matrix@q, 7, 5)
    assert transformed.time == pytest.approx(original.time, abs=.01)
    # Doubling both speed and acceleration preserves time and doubles distance.
    scaled = motion.solve_arrival([0, 0], 2*v, 2*q, 14, 10)
    assert scaled.time == pytest.approx(original.time, abs=.01)


def test_unresolved_is_explicit():
    with pytest.raises(motion.UnresolvedArrival):
        motion.solve_arrival([0, 0], [0, 5], [20, 0], 7, 5, max_evaluations=1)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "unknown"])
def test_invalid_acceleration(value):
    with pytest.raises(ValueError):
        motion.parse_accel(value)
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--accel", value])


@pytest.mark.parametrize("flags", [["--accel", "off"], ["--accel", "7"], ["--accel-table-id", "fixture"]])
def test_acceleration_flags_are_pc_only(flags):
    with pytest.raises(SystemExit):
        generate.parse_args(flags)


def test_table_compatibility_and_corruption(table):
    config = motion.configuration(7, "fixture", max_speeds=(5.,))
    assert config["initial_speed"] == "clip_preserve_direction"
    # A different physical setting may reuse normalized data if accuracy permits.
    motion.configuration(14, "fixture", max_speeds=(5.,))
    with pytest.raises(ValueError, match="too coarse"):
        motion.configuration(7, "fixture", max_speeds=(10.,))
    with pytest.raises(ValueError, match="fingerprint"):
        motion.configuration(7, "fixture", fingerprint="different")
    (table/"table.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        motion.load_table("fixture")


def test_cli_defaults_conflicts_and_location(table):
    args = generate.parse_args(["--pc-xpass", "--accel", "7", "--accel-table-id", "fixture"])
    assert args.reaction_time == 0 and args.reaction_time_mode == "fixed"
    assert args.acceleration_config["accel"] == 7
    assert location.parse_args(["--accel", "7", "--accel-table-id", "fixture"]).reaction_time == 0
    for extra in (["--reaction-time", ".2"], ["--reaction-time", "dist_pass"], ["--dist-pass-min", "0"]):
        with pytest.raises(SystemExit):
            generate.parse_args(["--pc-xpass", "--accel", "7", "--accel-table-id", "fixture", *extra])
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--accel", "7"])
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--accel-table-id", "fixture"])
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--accel", "7", "--accel-table-id", "fixture", "--margin", "reachability"])


def test_location_inherits_legacy_off_instead_of_table_override(table, tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path/"caches")
    original = location.parse_args([])
    original.pc_xpass_id = "legacy_location"
    versions.start_generation(original, location=True)
    with pytest.warns(UserWarning):
        loaded = location.parse_args(["--pc-xpass-id", "legacy_location", "--accel", "7",
                                      "--accel-table-id", "fixture"])
    assert loaded.accel is None and loaded.accel_table_id is None


def test_vectorized_arrivals_and_exact_zero(table):
    config = motion.configuration(7, "fixture")
    values = motion.arrival_times(np.array([[0., 0., 0., 0.], [0., 0., 30., 40.]]),
                                  np.array([[0., 20.]]), np.zeros((1, 2)), [5., 5.], config)
    assert values.shape == (2, 1, 2)
    assert values[0, 0, 0] == values[1, 0, 0] == 0.
    assert values[0, 0, 1] == pytest.approx(4+5/14)


def test_fallback_calls_reference_without_excluding_far_players(table, monkeypatch):
    meta, arrays = motion.load_table("fixture")
    monkeypatch.setitem(arrays, "fallback", np.array([True]))
    config = motion.configuration(7, "fixture")
    calls = []
    original = motion.solve_arrival
    def reference(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(motion, "solve_arrival", reference)
    values = motion.arrival_times(np.array([[0., 0., 5., 0.]]), np.array([60.]), np.array([0.]), [5.], config)
    assert values.item() == pytest.approx(12.)
    assert len(calls) == 1


def test_graph_batch_worker_cache_and_legacy_contract(table, tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path/"cache")
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "accelerated", "--accel", "7",
                               "--accel-table-id", "fixture", "--angle-step", "90", "--radial-gridsize", "40"])
    versions.start_generation(args)
    graph = make_graph()
    options = dict(angle_step=90, radial_gridsize=40, reaction_time=0, acceleration_config=args.acceleration_config)
    first = physics.compute_graph_pc_xpass_metrics(graph, **options)
    pd.testing.assert_series_equal(first, physics.compute_graphs_pc_xpass_metrics([graph], **options)[0])
    item = dict(match_id="fixture", graphs=[graph], labels=torch.stack([make_label()]))
    cache_dir = versions.cache_dir("hawkeye", args)
    initial = generate.prewarm_runtime_items([item], cache_dir=cache_dir, args=args)
    resumed = generate.prewarm_runtime_items([item], cache_dir=cache_dir, args=args)
    assert initial["cache_misses"] == 1 and resumed["cache_hits"] == 1
    assert initial["acceleration_queries"] > 0
    loaded = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "accelerated"])
    assert loaded.accel == 7 and loaded.accel_table_id == "fixture" and loaded.reaction_time == 0
    with pytest.raises(ValueError, match="accel"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "accelerated", "--accel", "off"])
    legacy = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "legacy"])
    versions.start_generation(legacy)
    assert "accel" not in versions.settings(legacy)
    with pytest.raises(ValueError, match="accel"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "legacy", "--accel", "7", "--accel-table-id", "fixture"])
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
    assert "acceleration_config" not in metadata
    accelerated = copy.deepcopy(metadata); accelerated["acceleration_config"] = args.acceleration_config
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(metadata) != physics.pc_xpass_lane_survival_metadata_fingerprint(accelerated)


def _process_graph(graph, config, root):
    from pathlib import Path
    motion.ROOT = Path(root)
    return physics.compute_graph_pc_xpass_metrics(graph, angle_step=90, radial_gridsize=40,
                                                   reaction_time=0, acceleration_config=config)


def test_spawned_process_matches_serial(table):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    graph = make_graph()
    config = motion.configuration(7, "fixture")
    expected = _process_graph(graph, config, table.parent)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = pool.submit(_process_graph, graph, config, table.parent).result()
    pd.testing.assert_series_equal(expected, actual)


def test_cache_only_metadata_does_not_load_table(table, tmp_path, monkeypatch):
    config = motion.configuration(7, "fixture")
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                                       reaction_time=0., acceleration_config=config)
    root = tmp_path/"predictions"; root.mkdir()
    (root/"metadata.json").write_text(json.dumps(metadata))
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache-only consumers must not load the movement table")
    monkeypatch.setattr(motion, "load_table", forbidden)
    read = physics.validate_physical_xpass_cache_metadata(root, expected_source=physics.PC_XPASS_SOURCE)
    assert read["acceleration_config"] == config
    lookup = physics.physical_xpass_inference_lookup_config({"pc_xpass": True}, cache_dir=root)
    assert lookup["reaction_time"] == 0 and lookup["acceleration_config"] == config


def test_prepare_publication_resume_and_id_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "ROOT", tmp_path/"tables")
    class InlinePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, fn, tasks, **kwargs):
            for point, _, _ in tasks:
                value = math.expm1(point[2])
                yield point, dict(lower=value, upper=value, residual=0.)
    monkeypatch.setattr(prepare, "ProcessPoolExecutor", InlinePool)
    args = prepare.parse_args(["--table-id", "prepared", "--accel", "7", "--workers", "1"])
    first = prepare.build(args)
    assert first["status"] == "complete" and first["max_distance"] == 150.
    assert first["fallback_leaves"] == 0
    assert prepare.build(args)["fingerprint"] == first["fingerprint"]
    clone = prepare.parse_args(["--table-id", "refined", "--source-table-id", "prepared", "--accel", "7", "--workers", "1"])
    cloned = prepare.build(clone)
    assert cloned["status"] == "complete"
    assert motion._digest(motion.table_path("prepared")/"table.npz") == first["fingerprint"]
    args.accel = 8.
    with pytest.raises(ValueError, match="different preparation settings"):
        prepare.build(args)


def test_holdout_failures_are_refined_or_remain_explicit_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(motion, "ROOT", tmp_path/"tables")
    lo = math.log1p(1e-4); hi = math.log1p(.28)
    def value(point):
        normalized = np.array([point[0], point[1]/math.pi, (point[2]-lo)/(hi-lo)])
        return math.expm1(point[2])+.1*float(np.prod(np.sin(4*math.pi*normalized)**2))
    class InlinePool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, fn, tasks, **kwargs):
            for point, _, _ in tasks:
                t = value(point)
                yield point, dict(lower=t, upper=t, residual=0.)
    monkeypatch.setattr(prepare, "ProcessPoolExecutor", InlinePool)
    args = prepare.parse_args(["--table-id", "bump", "--accel", "7", "--max-distance", "1", "--max-cells", "127", "--workers", "1"])
    meta = prepare.build(args)
    assert meta["status"] == "complete"
    state = json.loads((motion.table_path("bump")/"checkpoint.json").read_text())
    known = [p for node in state["nodes"] for p in node.get("failed_probes", [])]
    assert known, "The independent holdouts must detect a function hidden from regular probes"
    _, arrays = motion.load_table("bump")
    for point in known:
        index = 0
        while arrays["children"][index, 0] >= 0:
            axis = arrays["axis"][index]
            side = int(point[axis] >= .5*(arrays["low"][index, axis]+arrays["high"][index, axis]))
            index = arrays["children"][index, side]
        if not arrays["fallback"][index]:
            weight = (np.array(point)-arrays["low"][index])/(arrays["high"][index]-arrays["low"][index])
            estimated = float(motion._interpolate(arrays["values"][[index]], weight[None, :])[0]+math.expm1(point[2]))
            assert abs(estimated-value(point)) <= meta["time_error_normalized"]
