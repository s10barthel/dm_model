"""Compare feature optimizations with captured sources or a Git revision.

Uses isolated synthetic matches, real label/return kernels, and actual process
workers. This measures compute and staging, not production storage throughput.
"""
from pathlib import Path
import argparse
import copy
import importlib
import json
import math
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def measure(function):
    """Sample aggregate RSS, including children; shared pages count per process."""
    import psutil
    process, stop = psutil.Process(), threading.Event()
    peak = [process.memory_info().rss]
    def sample():
        while not stop.is_set():
            total = 0
            for member in [process, *process.children(recursive=True)]:
                try:
                    total += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak[0] = max(peak[0], total)
            stop.wait(.05)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        value = function()
        return value, time.perf_counter() - started, peak[0] / 2**20
    finally:
        stop.set()
        thread.join()


def original_function(args, module_name, name, class_name=None):
    module = importlib.import_module(module_name)
    path = module_name.replace(".", "/") + ".py"
    if args.baseline_sources:
        source = (Path(args.baseline_sources) / path).read_text()
    else:
        source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:{path}"], cwd=ROOT, text=True)
    namespace = dict(vars(module))
    exec(compile(source, f"baseline:{path}", "exec"), namespace)
    return getattr(namespace[class_name], name) if class_name else namespace[name]


def synthetic_match(count):
    import numpy as np
    import pandas as pd
    from datatools.match import Match
    from datatools.pass_trajectory import PASS_DURATION
    positions = np.arange(count)
    frames = positions * 4
    teams = np.where(positions % 4 < 2, "home", "away")
    events = pd.DataFrame(dict(action_id=positions, original_event_id=1000 + positions,
        object_id=[team + "_1" for team in teams], period_id=1 + (positions >= count // 2),
        spadl_type=np.where(positions % 11 == 0, "shot", "pass"),
        success=positions % 3 != 0, expected_goal=np.where(positions % 11 == 0, .1, np.nan),
        xT=np.linspace(.01, .8, count), scores_xT=0., concedes_xT=0.,
        goal_distance=np.linspace(.2, .9, count), epv=np.linspace(.05, .7, count),
        start_x=30., start_y=20., end_x=50., end_y=34., frame_id=frames))
    actions = events.assign(action_type=events.spadl_type, receive_frame_id=frames + 2,
        receiver_id=[team + "_2" for team in teams], intent_id=[team + "_2" for team in teams],
        blocked=False, pass_endpoint_definition=PASS_DURATION, max_pass_duration=np.nan,
        pass_duration=.08, success_known=positions % 17 != 0)
    actions.loc[positions % 29 == 1, "receiver_id"] = "untracked_player"
    tracking = pd.DataFrame(index=range(count * 4 + 3))
    for team, x in (("home", 20.), ("away", 70.)):
        for player in (1, 2):
            tracking[f"{team}_{player}_x"] = x + player
            tracking[f"{team}_{player}_y"] = 30. + player
    tracking["ball_z"] = 1.
    tracking["ball_x"], tracking["ball_y"] = 40., 34.
    tracking["period_id"] = 1 + (tracking.index >= count * 2)
    match = Match.__new__(Match)
    match.events, match.tracking = events, tracking
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.stop_condition = "base"
    return match, actions


def label_benchmark(args):
    import pandas as pd
    import torch
    from datatools.label_reuse import reuse_label_rows
    from datatools.match import Match
    old = original_function(args, "datatools.match", "construct_labels", "Match")
    results = []
    all_returns = ["next_3", "disc_0.9", "extended_possession_3", "in_1", "next_10", "disc_poly_max_0.1_1"]
    for count in (1, 3, 6):
        returns = all_returns[:count]
        samples, expected, final_actions = {}, None, None
        for variant in ("baseline", "optimized"):
            durations, memory = [], []
            for repeat in range(args.repeats + 1):
                match, actions = synthetic_match(args.actions)
                def calculate():
                    if variant == "baseline":
                        return [old(match, resolved_actions=actions, relabel_intended_receivers=False, return_type=ret)
                                for ret in returns]
                    with reuse_label_rows(match, enabled=len(returns) > 1):
                        return [match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                         return_type=ret) for ret in returns]
                labels, duration, peak = measure(calculate)
                durations.append(duration)
                memory.append(peak)
                if variant == "baseline":
                    expected, final_actions = labels, match.actions.copy()
                else:
                    for a, b in zip(expected, labels):
                        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
                    pd.testing.assert_frame_equal(final_actions, match.actions, check_exact=True)
            samples[variant] = dict(cold_seconds=durations[0], warm_median_seconds=statistics.median(durations[1:]),
                                    peak_aggregate_rss_mb=max(memory))
        results.append(dict(returns=returns, actions=args.actions, exact_data_equal=True, **samples,
            warm_speedup=samples["baseline"]["warm_median_seconds"] / samples["optimized"]["warm_median_seconds"]))
        print(json.dumps(results[-1]), flush=True)
    return results


def refresh_fixture(root, count, matches):
    import numpy as np
    import pandas as pd
    import torch
    import project_config as pc
    from datatools import config
    source, events_dir, annotations = [root / name for name in ("source", "events", "annotations")]
    for folder in (source, events_dir, annotations):
        folder.mkdir(parents=True)
    match, actions = synthetic_match(count)
    events = match.events
    returns = ["next_3", "disc_poly_max_0.1_1", "extended_possession_3"]
    metadata = dict(run_id="synthetic", return_types=returns,
                    intended_receiver_modes=["original", "angle_only", "model"],
                    carry_variant=dict(available=True, definition="fixture"),
                    split_manifest=dict(universe_count=matches))
    def labels(directory, mid, indices):
        directory.mkdir(parents=True, exist_ok=True)
        value = torch.arange(len(indices) * len(config.LABEL_COLUMNS), dtype=torch.float32).reshape(len(indices), -1)
        value[:, 0] = torch.tensor(indices)
        torch.save(value, directory / f"{mid}.pt")
    for number in range(matches):
        mid = f"match_{number:03d}"
        events.to_csv(events_dir / f"{mid}.csv", index=False)
        events[["action_id", "xT", "scores_xT", "concedes_xT", "goal_distance", "epv"]].to_csv(annotations / f"{mid}.csv", index=False)
        for folder in ("action_graphs", "post_action_graphs", "action_graphs_carries", "post_action_graphs_carries",
                       "action_graphs_intent_train", "action_graphs_success_intent", "augmented_graphs",
                       "augmented_graphs_angle_only", "augmented_graphs_model"):
            directory = source / folder
            directory.mkdir(exist_ok=True)
            (directory / f"{mid}.pt").write_bytes(b"inventory-only graph placeholder")
        for mode in metadata["intended_receiver_modes"]:
            for carry in (False, True):
                table = actions.copy()
                if carry:
                    extra = table.iloc[[0]].copy()
                    extra.index = [count + 9]
                    extra["original_event_id"], extra["action_id"] = "carry:fixture", count + 9
                    extra["return_event_index"], extra["action_type"] = 1, "dribble"
                    table = pd.concat([table, extra])
                    table["original_event_id"] = table.original_event_id.astype(str)
                directory = pc.get_resolved_action_dir(mode, root=source, use_carries=carry)
                directory.mkdir(exist_ok=True)
                table.to_parquet(directory / f"{mid}.parquet")
                for ret in returns:
                    labels(pc.get_action_label_dir(ret, mode, root=source, use_carries=carry), mid, list(table.index))
            for ret in returns:
                labels(pc.get_intent_train_label_dir(ret, mode, root=source), mid, [0, 0, *range(1, count)])
            labels(pc.get_augmented_label_dir(mode, root=source), mid, [0, 0, *range(1, count)])
        labels(pc.get_success_intent_label_dir(source), mid, list(actions.index))
    (source / "metadata.json").write_text(json.dumps(metadata))
    return source, events_dir, annotations, metadata


def compare_artifacts(a, b):
    import pandas as pd
    import torch
    files = {path.relative_to(a) for path in a.rglob("*") if path.suffix in {".pt", ".parquet"}}
    assert files == {path.relative_to(b) for path in b.rglob("*") if path.suffix in {".pt", ".parquet"}}
    for relative in files:
        left, right = a / relative, b / relative
        if "graphs" in relative.parent.name:
            assert left.read_bytes() == right.read_bytes()
        elif relative.suffix == ".pt":
            torch.testing.assert_close(torch.load(left, weights_only=True), torch.load(right, weights_only=True),
                                       rtol=0, atol=0, equal_nan=True)
        else:
            x, y = pd.read_parquet(left), pd.read_parquet(right)
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            assert x.attrs == y.attrs


def semantic_report(value, root):
    if isinstance(value, dict):
        return {semantic_report(k, root): semantic_report(v, root) for k, v in value.items()
                if k not in {"elapsed_seconds", "staging_seconds", "rewritten_bytes", "workers", "worker_pids"}}
    if isinstance(value, list):
        return [semantic_report(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root.resolve()), "{ROOT}")
    if isinstance(value, float) and math.isnan(value):
        return "{NaN}"
    return value


def refresh_benchmark(args, output):
    from datatools.artifact_storage import clone_feature_artifacts
    from datatools.xt_refresh import refresh_xt, fingerprint
    from datatools.goal_distance_refresh import refresh_goal_distance
    from datatools.stop_conditions import refresh_labels
    import project_config
    from scripts.feature_extension_diagnostics import ExtensionAttempt
    source, events, annotations, metadata = refresh_fixture(output / "inputs", args.refresh_actions, args.matches)
    originals = {path: fingerprint(path) for path in source.rglob("*") if path.is_file()}
    records = []
    project_config.GOAL_DISTANCE_MATCH_DIR = annotations
    project_config.EPV_MATCH_DIR = annotations
    cases = [("xt", refresh_xt, "datatools.xt_refresh", "refresh_xt"),
             ("goal_distance", refresh_goal_distance, "datatools.goal_distance_refresh", "refresh_goal_distance"),
             ("stop_conditions", refresh_labels, "datatools.stop_conditions", "refresh_labels")]
    for family, current, module, name in cases:
        if family not in args.families:
            continue
        baseline = original_function(args, module, name)
        reference, reference_report = None, None
        for workers in [None, *args.workers]:
            destination = output / f"{family}_{workers if workers is not None else 'baseline'}"
            clone_feature_artifacts(source, destination)
            options = dict(event_dir=events, attempt=ExtensionAttempt(destination, family, "derived", {}))
            if family == "xt":
                options["xt_match_dir"] = annotations
            elif family == "goal_distance":
                options.update(goal_distance_match_dir=annotations, target_families=["goal_distance", "progression"])
            else:
                options.update(xt_match_dir=annotations,
                               requests={ret: ["base"] for ret in metadata["return_types"]})
            if workers is not None:
                options.update(num_workers=workers, worker_thread_limit=1)
            report, elapsed, peak = measure(lambda: (baseline if workers is None else current)(
                destination, copy.deepcopy(metadata), **options))
            if workers is None:
                reference, reference_report = destination, semantic_report(report, destination)
            else:
                compare_artifacts(reference, destination)
                assert reference_report == semantic_report(report, destination)
            record = dict(family=family, workers=workers or "baseline", matches=args.matches,
                          actions_per_match=args.refresh_actions, seconds=elapsed,
                          peak_aggregate_rss_mb=peak, exact_data_equal=True)
            records.append(record)
            print(json.dumps(record), flush=True)
    assert all(fingerprint(path) == digest for path, digest in originals.items())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-sources")
    baseline.add_argument("--baseline-ref")
    parser.add_argument("--output", required=True)
    parser.add_argument("--actions", type=int, default=500)
    parser.add_argument("--refresh-actions", type=int, default=150)
    parser.add_argument("--matches", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--families", nargs="+", choices=["xt", "goal_distance", "stop_conditions"],
                        default=["xt", "goal_distance", "stop_conditions"])
    parser.add_argument("--labels-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = dict(workload="synthetic", labels=label_benchmark(args))
    if not args.labels_only:
        results["refresh"] = refresh_benchmark(args, output)
    (output / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {output / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
