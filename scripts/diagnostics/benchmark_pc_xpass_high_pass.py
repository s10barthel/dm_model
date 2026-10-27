"""Compare ground-only and aerial PC xPass on saved benchmark states.

Writes diagnostics and temporary prediction caches; no normal caches/latest
pointers are changed. Source graphs, imports and solver warmup are timed apart.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
import torch

from datatools.benchmark import build_benchmark_state, load_benchmark_modification_data
import pc_xpass_high_pass as hp
import pc_xpass_acceleration as acceleration
import physical_pass_model as physics
from scripts.diagnostics.benchmark_pc_xpass_optimization import STATE_PAIRS


def measured(call):
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        result = call()
        elapsed = time.perf_counter()-start
    finally:
        stop.set()
        sampler.join()
    return result, elapsed, max(peak[0], process.memory_info().rss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-repeats", type=int, default=3)
    parser.add_argument("--margin", choices=["tta", "distance"], default="tta")
    parser.add_argument("--accel-table-id", default="accel7_v8_radius1_refined")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"out/pc_xpass_high_pass")
    args = parser.parse_args()
    if not 1 <= args.states <= len(STATE_PAIRS) or min(args.repeats, args.cache_repeats) < 1:
        parser.error("Use available states and positive repeat counts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    graphs, items, names = [], [], []
    for modification, state_id in STATE_PAIRS[:args.states]:
        data = load_benchmark_modification_data(modification, ROOT/"benchmark")
        state, _, _ = build_benchmark_state(data[f"game_state_{state_id}"], modification,
                                           state_id, int(data["higher_state_id"]))
        graph = state.graph_features_0[0]
        graphs.append(graph)
        names.append(f"{modification}_{state_id}")
        items.append(dict(match_id=state.match_id, graphs=[graph], labels=state.labels[:1]))
    loading = time.perf_counter()-started
    radius = 1. if args.margin == "tta" else 0.
    if args.margin == "distance":
        import pc_xpass_distance as spatial
        movement = spatial.configuration(7, args.accel_table_id, max_speeds=(8.,))
    else:
        movement = acceleration.configuration(7, args.accel_table_id, max_speeds=(8.,), blocking_radius=radius)
    common = dict(min_speed=3., max_speed=25., speed_step=2., angle_step=3., radial_gridsize=1.,
                  reaction_time=0., ball_dec="drag", margin=args.margin, acceleration_config=movement,
                  blocking_radius=radius, lane_inflection_point=0., control_inflection_point=0.,
                  max_player_speed=8., max_player_speed_off=8., max_player_speed_def=8.,
                  use_position_discount=False, top_xt=True, top_n=None,
                  top_n_values=[10, 25], top_pass_values=[5, 10])
    report = dict(source_loading_seconds=loading, states=names, settings=common,
                  source_revision=hp.REVISION, variants={}, production_caches_changed=False,
                  memory_measurement="Absolute process working set sampled every 10 ms; shared process, not incremental memory")
    predictions = []
    for name, config in (("off", None), ("on", hp.configuration("on"))):
        opts = {**common, "high_pass_config": config}
        hp.profile.cache_clear()
        _, first, peak = measured(lambda: physics.compute_graph_pc_xpass_metrics(graphs[0], **opts))
        durations = []
        rows = None
        for _ in range(args.repeats):
            rows, seconds, rss = measured(lambda: [physics.compute_graph_pc_xpass_metrics(g, **opts) for g in graphs])
            durations.append(seconds)
            peak = max(peak, rss)
        caches, reuse, reuse_stats = [], [], []
        for _ in range(args.cache_repeats):
            with tempfile.TemporaryDirectory(prefix="pc_aerial_benchmark_") as directory:
                def warm():
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        return physics.prewarm_physical_xpass_runtime_cache(items, cache_dir=directory,
                            source=physics.PC_XPASS_SOURCE, num_workers=1, export_lane_control=True, **opts)
                _, seconds, rss = measured(warm)
                caches.append(seconds)
                peak = max(peak, rss)
                stats, seconds, _ = measured(warm)
                assert stats["cache_misses"] == 0
                reuse.append(seconds)
                reuse_stats.append({key: stats[key] for key in ("cache_hits", "cache_misses", "copied_from_reuse")})
        report["variants"][name] = dict(high_pass_config=config,
            aerial_profiles_per_azimuth=len(config["angles"])*len(config["speeds"]) if config else 0,
            top_pass_definition=hp.TOP_PASS_DEFINITION if config else physics.PC_XPASS_TOP_PASS_DEFINITION,
            first_state_seconds=first, computation_seconds=durations,
            median_computation_seconds=float(np.median(durations)), fresh_cache_seconds=caches,
            median_fresh_cache_seconds=float(np.median(caches)), cache_reuse_seconds=reuse,
            median_cache_reuse_seconds=float(np.median(reuse)), peak_working_set_bytes=peak,
            flight_profiles=hp.profile.cache_info()._asdict(), cache_reuse_stats=reuse_stats)
        for state_name, graph, row in zip(names, graphs, rows):
            for i in physics._candidate_target_indices(graph):
                player = graph.node_ids[i]
                predictions.append(dict(mode=name, state=state_name, receiver=player,
                    **{field: float(row.get(f"{player}__{field}", 0. if field in ("elevation", "target_z") else np.nan))
                       for field in ("max_xpass", "top_pass10_xpass", "lane_survival", "control_prob", "speed", "angle", "distance", "elevation", "target_z", "flight_time")}))
        print(name, json.dumps(report["variants"][name]), flush=True)
    report["warm_runtime_ratio"] = report["variants"]["on"]["median_computation_seconds"]/report["variants"]["off"]["median_computation_seconds"]
    (args.output_dir/"benchmark.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(predictions).to_csv(args.output_dir/"predictions.csv", index=False)


if __name__ == "__main__":
    main()
