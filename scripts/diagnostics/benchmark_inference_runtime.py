"""Compare runtime inference with captured working sources, without publishing runs."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import inference
from datatools import hawkeye
from datatools.inference_features import runtime_graph_feature_kwargs
from datatools.utils import filter_features_and_labels
from models.edge_schema import generation_kwargs
from models.utils import load_model, validate_model_graph_schemas


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-csv", type=Path, default=ROOT / "hawkeye_data/centroid_data_team.csv")
    parser.add_argument("--ball-csv", type=Path, default=ROOT / "hawkeye_data/ball_data_selected.csv")
    parser.add_argument("--sample-rows", type=int, default=20000, help="Zero reads the complete tracking CSV.")
    parser.add_argument("--situation-id")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repeats to observe baseline CUDA repeatability.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_graph = load_source(args.baseline_sources / "datatools/graph_feature.py", "baseline_runtime_graph")
    original_hawkeye = load_source(args.baseline_sources / "datatools/hawkeye.py", "baseline_runtime_hawkeye")
    original_hawkeye.construct_graph_for_frame = original_graph.construct_graph_for_frame
    original_inference = load_source(args.baseline_sources / "inference.py", "baseline_runtime_inference")
    identifiers, models = {}, {}
    for task in ("outcome_scoring", "outcome_conceding"):
        identifier = getattr(args, task + "_model_id")
        if not identifier:
            weights = max((ROOT / "saved" / task).glob("*/best_weights.pt"), key=lambda p: p.stat().st_mtime)
            identifier = task + "/" + weights.parent.name
        identifiers[task] = identifier
        models[task] = load_model(identifier, args.device)
    schema = validate_model_graph_schemas(models)
    options = dict(add_v_edge_features=bool(schema["add_v_edge_features"]),
        add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)),
        **generation_kwargs(schema))
    features = runtime_graph_feature_kwargs(models)
    report = dict(device=args.device, models=identifiers, computed_extended_features=features,
                  sample_rows=args.sample_rows, repeats=args.repeats, runs=[])

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)

    def run(mode):
        synchronize()
        started = time.perf_counter()
        tracking = hawkeye.clean_hawkeye_tracking(pd.read_csv(args.tracking_csv,
            nrows=args.sample_rows or None, low_memory=False))
        ball = hawkeye.clean_hawkeye_ball(hawkeye.load_hawkeye_ball(args.ball_csv))
        identifier = args.situation_id or str(tracking.id.iloc[0])
        selected = tracking.loc[tracking.id.eq(identifier)].copy()
        loaded = time.perf_counter()
        builder = original_hawkeye if mode == "baseline" else hawkeye
        with contextlib.redirect_stderr(io.StringIO()):
            state, _, _ = builder.build_hawkeye_situation(selected, ball,
                **options, **(features if mode == "optimized" else {}))
        built = time.perf_counter()
        prepared = {task: filter_features_and_labels(state.graph_features_0, state.labels, model.args)[0]
                    for task, model in models.items()}
        predictor = original_inference if mode == "baseline" else inference
        synchronize()
        infer_started = time.perf_counter()
        with contextlib.redirect_stderr(io.StringIO()):
            predictions = {task: predictor.inference_gnn(state, model, device=args.device,
                graph_override=state.graph_features_0, label_override=state.labels) for task, model in models.items()}
        synchronize()
        elapsed = time.perf_counter() - infer_started
        return dict(mode=mode, frames=len(state.graph_features_0), loading_seconds=loaded - started,
            graph_preparation_seconds=built - loaded, inference_seconds=elapsed,
            total_seconds=built - started + elapsed), prepared, predictions, state

    print("Warming both inference paths", flush=True)
    captured_logits = {}
    handles = [model.register_forward_hook(
        lambda module, inputs, output, task=task: captured_logits.__setitem__(task, output.detach().clone()))
        for task, model in models.items()]
    try:
        _, reference_inputs, reference_predictions, reference_state = run("baseline")
    finally:
        for handle in handles:
            handle.remove()
    run("optimized")
    # Replaying one actual forward output isolates decoding from nondeterministic
    # CUDA graph reductions. A maximum from three forward repeats is not a bound
    # on the next identical forward's rounding variation.
    for task, model in models.items():
        original_forward = model.forward
        model.forward = lambda graphs, task=task: captured_logits[task]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                decoded = [module.inference_gnn(reference_state, model, device=args.device,
                    graph_override=reference_state.graph_features_0, label_override=reference_state.labels)
                    for module in (original_inference, inference)]
            for before, after in zip(*decoded):
                pd.testing.assert_frame_equal(before, after, check_exact=True)
        finally:
            model.forward = original_forward
    report["fixed_logits_predictions_equal"] = True
    variations = dict(baseline=0., optimized=0.)
    for repeat in range(args.repeats):
        for mode in (("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")):
            timing, prepared, predictions, _ = run(mode)
            for task in models:
                assert len(prepared[task]) == len(reference_inputs[task])
                for before, after in zip(reference_inputs[task], prepared[task]):
                    for name in ("x", "edge_index", "edge_attr"):
                        assert torch.equal(getattr(before, name), getattr(after, name)), (task, name)
                for before, after in zip(reference_predictions[task], predictions[task]):
                    assert before.index.equals(after.index) and before.columns.equals(after.columns)
                    assert np.array_equal(before.isna().to_numpy(), after.isna().to_numpy())
                    difference = np.abs(before.to_numpy() - after.to_numpy())
                    variations[mode] = max(variations[mode], float(np.nanmax(difference)) if difference.size else 0.)
            report["runs"].append(timing)
            print(json.dumps(timing), flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.update(prepared_inputs_equal=True, max_output_difference=variations,
                  output_difference_within_baseline_variation=variations["optimized"] <= variations["baseline"])
    report["medians"] = {mode: {key: statistics.median(row[key] for row in report["runs"] if row["mode"] == mode)
        for key in ("loading_seconds", "graph_preparation_seconds", "inference_seconds", "total_seconds")}
        for mode in ("baseline", "optimized")}
    report["total_runtime_reduction_percent"] = 100 * (1 - report["medians"]["optimized"]["total_seconds"] / report["medians"]["baseline"]["total_seconds"])
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("medians", "total_runtime_reduction_percent", "max_output_difference")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
