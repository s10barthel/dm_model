"""Time PC xPass uncertainty on saved states without writing prediction caches."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from datatools.benchmark import build_benchmark_state, load_benchmark_modification_data
import pc_xpass_high_pass as hp
import pc_xpass_uncertainty as uncertainty
import physical_pass_model as physics
from scripts.diagnostics.benchmark_pc_xpass_optimization import STATE_PAIRS
from scripts.diagnostics.benchmark_pc_xpass_high_pass import measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--margin", choices=["tta", "distance"], default="tta")
    parser.add_argument("--accel-table-id", default="accel7_v8_radius1_refined")
    parser.add_argument("--high-pass", choices=["on", "off"], default="on")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/pc_xpass_uncertainty_independent")
    args = parser.parse_args()
    if not 1 <= args.states <= len(STATE_PAIRS) or args.repeats < 1:
        parser.error("Select available states and a positive repeat count")
    torch.set_num_threads(1)
    graphs = []
    for modification, state_id in STATE_PAIRS[:args.states]:
        data = load_benchmark_modification_data(modification, ROOT / "benchmark")
        state, _, _ = build_benchmark_state(data[f"game_state_{state_id}"], modification,
                                           state_id, int(data["higher_state_id"]))
        graphs.append(state.graph_features_0[0])
    radius = 1. if args.margin == "tta" else 0.
    if args.margin == "distance":
        import pc_xpass_distance as movement
        accel = movement.configuration(7, args.accel_table_id, max_speeds=(8.,))
    else:
        import pc_xpass_acceleration as movement
        accel = movement.configuration(7, args.accel_table_id, max_speeds=(8.,), blocking_radius=radius)
    opts = dict(min_speed=3., max_speed=25., speed_step=2., angle_step=3., radial_gridsize=1.,
                high_pass_config=hp.configuration(args.high_pass), margin=args.margin,
                acceleration_config=accel, blocking_radius=radius, reaction_time=0., ball_dec="drag",
                max_player_speed=8., max_player_speed_off=8., max_player_speed_def=8.,
                lane_inflection_point=0., control_inflection_point=0., use_position_discount=False,
                top_xt=True, top_n=None, top_pass_values=[1, 5, 10],
                enabled_metrics=["max_xpass", "top_pass1_xpass", "top_pass5_xpass", "top_pass10_xpass"])
    configs = {"off": None, "on": uncertainty.configuration(True)}
    timings = {key: [] for key in configs}
    warmup = {}
    warmup_profiles = {}
    warmup_rows = {}
    peak_rss = {key: 0 for key in configs}
    def compute(config):
        return [physics.compute_graph_pc_xpass_metrics(graph, uncertainty_config=config, **opts) for graph in graphs]
    for name, config in configs.items():
        start = time.perf_counter()
        warmup_rows[name] = compute(config)
        warmup[name] = time.perf_counter()-start
        warmup_profiles[name] = hp.profile.cache_info()._asdict()
    assert warmup_profiles["off"]["misses"] == warmup_profiles["on"]["misses"]
    for raw, added in zip(warmup_rows["off"], warmup_rows["on"]):
        pd.testing.assert_series_equal(raw, added.reindex(raw.index), check_exact=True)
    # Verify the mode adds no flight profiles, then alternate order across repeats.
    before = hp.profile.cache_info()._asdict()
    for repeat in range(args.repeats):
        for name in (list(configs) if repeat % 2 == 0 else list(reversed(configs))):
            _, elapsed, rss = measured(lambda: compute(configs[name]))
            timings[name].append(elapsed)
            peak_rss[name] = max(peak_rss[name], rss)
    after = hp.profile.cache_info()._asdict()
    assert before["misses"] == after["misses"]
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    report = dict(states=[list(pair) for pair in STATE_PAIRS[:args.states]], settings=opts,
                  uncertainty_config=configs["on"], warmup_seconds=warmup, computation_seconds=timings,
                  median_seconds=medians, runtime_ratio=medians["on"]/medians["off"],
                  added_runtime_percent=100*(medians["on"]/medians["off"]-1),
                  flight_profiles_before=before, flight_profiles_after=after,
                  warmup_flight_profiles=warmup_profiles, peak_working_set_bytes=peak_rss,
                  memory_measurement="Absolute shared-process RSS sampled every 10 ms; not incremental memory",
                  raw_exports_exact=True,
                  available_metrics_by_mode={name: [*opts["enabled_metrics"], *([uncertainty.METRIC] if config else [])]
                                             for name, config in configs.items()},
                  prediction_caches_changed=False, threads=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("median_seconds", "runtime_ratio", "added_runtime_percent")}))


if __name__ == "__main__":
    main()
