"""Compare point and disk TTA on real states, isolated workers and fresh caches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

import pc_xpass_acceleration as motion
from pc_xpass_versions import atomic_json
from scripts.diagnostics.benchmark_pc_xpass_acceleration import SETTINGS
from scripts.diagnostics.benchmark_pc_xpass_optimization import STATE_PAIRS
from scripts.diagnostics.compare_reachability_xpass import measure
from scripts.generate_physical_xpass import parse_args, prewarm_runtime_items
from datatools.benchmark import build_benchmark_state, load_benchmark_modification_data
import physical_pass_model as physics


@contextlib.contextmanager
def lookup_backend(use_numpy):
    """Diagnostic-only override; never persisted in model/cache settings."""
    original = motion._lookup_arrival_corrections
    if use_numpy:
        motion._lookup_arrival_corrections = motion._lookup_arrival_corrections_numpy
    try:
        yield
    finally:
        motion._lookup_arrival_corrections = original


def worker(graphs, settings, repeats, use_numpy=False):
    torch.set_num_threads(1)
    with lookup_backend(use_numpy):
        before = dict(motion.COUNTERS)
        start = time.perf_counter()
        physics.compute_graph_pc_xpass_metrics(graphs[0], **settings)
        first_state = time.perf_counter()-start
        rows, stats = measure(graphs, settings, repeats)
        stats['first_state_warmup_seconds'] = first_state
        stats['acceleration_stats_including_warmup'] = {k: motion.COUNTERS[k]-before[k] for k in before}
        return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-table-id', required=True)
    parser.add_argument('--radius-table-id', required=True)
    parser.add_argument('--blocking-radius', type=float, default=1.)
    parser.add_argument('--accel', type=float, default=7.)
    parser.add_argument('--max-player-speed', type=float, default=8.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-numpy', action='store_true', help='Compare compiled and original NumPy lookup for every variant, checking complete prediction equivalence')
    parser.add_argument('--hawkeye-graphs', type=Path, help='Optional graphs.pt from the lane investigation')
    parser.add_argument('--output', type=Path, default=ROOT/'out/pc_xpass_blocking_benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    torch.set_num_threads(1)
    items = []
    for modification, state_id in STATE_PAIRS:
        data = load_benchmark_modification_data(modification, ROOT/'benchmark')
        state, _, _ = build_benchmark_state(data[f'game_state_{state_id}'], modification, state_id, int(data['higher_state_id']))
        items.append(dict(match_id=state.match_id, graphs=state.graph_features_0[:1], labels=state.labels[:1]))
    graphs = [item['graphs'][0] for item in items]
    hawkeye = torch.load(args.hawkeye_graphs, weights_only=False) if args.hawkeye_graphs else {}
    graphs.extend(item['graph'] for item in hawkeye.values())
    base = {**SETTINGS, 'ball_dec': 'drag', 'reaction_time_mode': 'fixed', 'reaction_time': 0.,
            'max_player_speed': args.max_player_speed, 'max_player_speed_off': args.max_player_speed,
            'max_player_speed_def': args.max_player_speed}
    variants = {}
    for name, radius, offset, table_id in [('point_offset02', 0., .2, args.point_table_id),
                                         ('point_offset0', 0., 0., args.point_table_id),
                                         ('disk_offset0', args.blocking_radius, 0., args.radius_table_id)]:
        cfg = motion.configuration(args.accel, table_id, max_speeds=(args.max_player_speed,), blocking_radius=radius)
        variants[name] = {**base, 'blocking_radius': radius, 'lane_inflection_point': offset,
                          'control_inflection_point': offset, 'acceleration_config': cfg}
    report = dict(benchmark_states=len(items), numerical_states=len(graphs), variants={}, comparisons=[],
                  measurement='Numerical timings exclude graph loading and cache writes; fresh-cache timings include metadata/Parquet writes on 12 loaded benchmark states.')
    table_meta, _ = motion.load_table(args.radius_table_id)
    table_root = motion.table_path(args.radius_table_id)
    report['disk_table'] = dict(id=args.radius_table_id, fingerprint=table_meta['fingerprint'],
        preparation_seconds=table_meta.get('preparation_seconds'),
        cumulative_preparation_seconds=table_meta.get('cumulative_preparation_seconds'),
        lookup_bytes=sum(p.stat().st_size for p in table_root.rglob('table.npz')),
        independent_holdout_points=table_meta.get('independent_holdout_points'),
        independent_holdout_failed_leaves=table_meta.get('independent_holdout_failed_leaves'))
    outputs = {}
    experiments = [(name, settings, False) for name, settings in variants.items()]
    if args.include_numpy:
        experiments.extend((name+'_numpy', settings, True) for name, settings in variants.items())
    for name, settings, use_numpy in experiments:
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            outputs[name], stats = pool.submit(worker, graphs, settings, args.repeats, use_numpy).result()
        flags = ['--pc-xpass', '--accel', str(args.accel), '--accel-table-id', settings['acceleration_config']['table_id'],
                 '--blocking-radius', str(settings['blocking_radius']), '--reaction-time', '0',
                 '--max-player-speed', str(args.max_player_speed), '--ball-dec', 'drag', '--top-pass', '10', '--top-xt',
                 '--lane-power', '15', '--lane-inflection-point', str(settings['lane_inflection_point']),
                 '--control-power', '15', '--control-inflection-point', str(settings['control_inflection_point']),
                 '--endpoint-normalization', 'share', '--use-position-discount', 'false', '--min-speed', '3',
                 '--max-speed', '25', '--speed-step', '2', '--angle-step', '3', '--radial-gridsize', '1', '--num-workers', '1']
        cli = parse_args(flags)
        cold, warm = [], []
        with lookup_backend(use_numpy):
            start = time.perf_counter()
            physics.compute_graph_pc_xpass_metrics(graphs[0], **settings)
            stats['cache_process_first_state_warmup_seconds'] = time.perf_counter()-start
            for _ in range(args.repeats):
                with tempfile.TemporaryDirectory(prefix='pc_blocking_') as temporary:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        start = time.perf_counter()
                        result = prewarm_runtime_items(items, cache_dir=Path(temporary), args=cli)
                        cold.append(time.perf_counter()-start)
                        start = time.perf_counter()
                        reused = prewarm_runtime_items(items, cache_dir=Path(temporary), args=cli)
                        warm.append(time.perf_counter()-start)
                    assert result['cache_misses'] == len(items) and reused['cache_misses'] == 0
        stats.update(cache_seconds=cold, median_cache_seconds=float(np.median(cold)), warm_cache_seconds=warm,
                     settings=settings, lookup_backend='numpy' if use_numpy else 'compiled')
        report['variants'][name] = stats
        atomic_json(args.output, report)
        print(name, json.dumps(stats), flush=True)
    if args.include_numpy:
        report['compiled_numpy_comparisons'] = {}
        for name in variants:
            maximum, compared = 0., 0
            for compiled, original in zip(outputs[name], outputs[name+'_numpy']):
                pd.testing.assert_series_equal(compiled, original, check_exact=False, rtol=1e-12, atol=1e-12)
                finite = np.isfinite(compiled.to_numpy()) & np.isfinite(original.to_numpy())
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(compiled.to_numpy()[finite]-original.to_numpy()[finite]))))
                    compared += int(finite.sum())
            compiled, original = report['variants'][name], report['variants'][name+'_numpy']
            for counter in ('queries', 'fallbacks'):
                assert compiled['acceleration_stats_including_warmup'][counter] == original['acceleration_stats_including_warmup'][counter]
            report['compiled_numpy_comparisons'][name] = dict(predictions_match=True, maximum_difference=maximum,
                compared_finite_values=compared, numerical_runtime_reduction=1-compiled['median_seconds']/original['median_seconds'],
                fresh_cache_runtime_reduction=1-compiled['median_cache_seconds']/original['median_cache_seconds'])
    for i, sid in enumerate(hawkeye):
        node = 'away_250017233' if sid.startswith('5b9966a1') else 'home_1909098'
        report['comparisons'].append(dict(id=sid, receiver=node, scores={
            name: float(rows[len(items)+i][node+'__top_pass10_xpass']) for name, rows in outputs.items()}))
    # Independent real-state arrivals: target points need not coincide with table probes.
    rng = np.random.default_rng(20261026)
    checks = []
    for index in range(48):
        graph = graphs[index % len(graphs)]
        pos, _, _, _, _, _ = physics._full_as_default_simulation_inputs(graph, candidate_indices=physics._candidate_target_indices(graph))
        state = pos[int(rng.integers(len(pos)))]
        target = rng.uniform([-45., -28.], [45., 28.])
        reference = motion.solve_arrival(state[:2], state[2:], target, args.accel, args.max_player_speed,
                                         blocking_radius=args.blocking_radius, time_tol=motion.TIME_TOL*.1)
        lookup = float(motion.arrival_times(state[None, :], target[:1], target[1:], [args.max_player_speed],
                                            variants['disk_offset0']['acceleration_config'])[0, 0])
        error = max(reference.lower-lookup, lookup-reference.upper, 0.)
        checks.append(dict(player_state=state.tolist(), target=target.tolist(), lookup=lookup,
                           lower=reference.lower, upper=reference.upper, outside_bracket_error=error,
                           endpoint_error=max(abs(lookup-reference.lower), abs(lookup-reference.upper))))
    report['independent_arrival_checks'] = checks
    report['maximum_error_outside_reference_bracket'] = max(c['outside_bracket_error'] for c in checks)
    report['maximum_error_to_reference_endpoints'] = max(c['endpoint_error'] for c in checks)
    report['validation_reference_time_tolerance'] = motion.TIME_TOL*.1
    if report['maximum_error_to_reference_endpoints'] > motion.TIME_TOL:
        raise motion.UnresolvedArrival('Independent real-state disk lookup exceeded 10 ms error target')
    report['numerical_ratio_disk_to_point'] = report['variants']['disk_offset0']['median_seconds']/report['variants']['point_offset02']['median_seconds']
    report['cache_ratio_disk_to_point'] = report['variants']['disk_offset0']['median_cache_seconds']/report['variants']['point_offset02']['median_cache_seconds']
    atomic_json(args.output, report)
    print('RESULT', json.dumps({k: v for k, v in report.items() if k not in {'variants', 'independent_arrival_checks'}}), flush=True)


if __name__ == '__main__':
    main()
