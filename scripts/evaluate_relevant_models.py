from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.xpass_cli import add_top_pass_selector, resolve_top_pass_selector
from scripts import learning_curve
from models.utils import get_model_record, load_bundle_record, resolve_model_selection
from models.outcome_bootstrap import add_outcome_bootstrap_arguments, nonnegative_integer
import pc_xpass_versions as pc_versions

from project_config import EVALUATION_RUNS_DIR


SUPPORTED_EVALUATION_TASKS = (
    "action_intent",
    "pass_intent",
    "pass_success",
    "success_intent",
    "pass_height",
    "outcome_scoring",
    "outcome_conceding",
)


def parse_bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def probability_threshold(value: str) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be finite and between 0 and 1")
    return threshold


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_outcome_bootstrap_arguments(parser)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--learning-curve-model-id", action="append", default=[])
    parser.add_argument("--learning-curve-bootstrap-resamples", type=nonnegative_integer, default=2000)
    parser.add_argument("--learning-curve-bootstrap-seed", type=nonnegative_integer, default=42)
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--pass-height-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--diagnostic-feature-run-id")
    parser.add_argument("--evaluation-output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weighted-pass-success-metrics", action="store_true")
    parser.add_argument("--evaluate-xpass", action="store_true")
    parser.add_argument("--evaluate-combined-success", action="store_true")
    parser.add_argument("--xpass-version", default=None)
    parser.add_argument("--xpass-weight", choices=["v1", "v2", "v3", "v4", "v5"], default=None)
    parser.add_argument("--no-observed-pass-height-stratification", action="store_true")
    parser.add_argument("--classification-threshold", type=probability_threshold, default=0.5)
    parser.add_argument("--f1-outcome-threshold", type=probability_threshold, default=None)
    parser.add_argument("--pc-xpass-cache-dir", default=None)
    parser.add_argument("--discount", type=parse_bool_text, default=None)
    parser.add_argument("--v4-power", type=float, default=None)
    parser.add_argument("--v4-zero", type=float, default=None)
    parser.add_argument("--v5-intent-threshold", type=float, default=None)
    parser.add_argument("--v5-discount", type=parse_bool_text, default=None)
    add_top_pass_selector(parser)
    pc_versions.add_selection_argument(parser)
    args = parser.parse_args(argv)
    pc_versions.check_selectors(args)
    return resolve_top_pass_selector(parser, args, pc_only=True)


def validate_pass_success_predictor_args(args: argparse.Namespace) -> None:
    enabled = bool(args.evaluate_xpass or args.evaluate_combined_success)
    if enabled and not args.xpass_version:
        raise ValueError("--evaluate-xpass/--evaluate-combined-success require --xpass-version.")
    if args.xpass_weight and not args.evaluate_combined_success:
        raise ValueError("--xpass-weight requires --evaluate-combined-success.")
    if args.evaluate_combined_success and not args.xpass_weight:
        raise ValueError("--evaluate-combined-success requires --xpass-weight.")
    if args.evaluate_combined_success and args.xpass_weight == "v5":
        if not args.pass_intent_model_id:
            raise ValueError("Combined v5 evaluation requires explicit --pass-intent-model-id.")
        args.v5_intent_threshold = 0.01 if args.v5_intent_threshold is None else args.v5_intent_threshold
        args.v5_discount = True if args.v5_discount is None else args.v5_discount
        if not math.isfinite(float(args.v5_intent_threshold)) or float(args.v5_intent_threshold) <= 0.0:
            raise ValueError("--v5-intent-threshold must be a positive finite float.")
    elif args.v5_intent_threshold is not None or args.v5_discount is not None:
        raise ValueError("v5 options are only valid with combined --xpass-weight v5.")
    explicit_v4 = args.discount is not None or args.v4_power is not None or args.v4_zero is not None
    if args.evaluate_combined_success and args.xpass_weight == "v4":
        if args.discount is None or args.v4_power is None or args.v4_zero is None:
            raise ValueError("Combined v4 evaluation requires explicit --discount, --v4-power, and --v4-zero.")
        if not math.isfinite(float(args.v4_power)) or float(args.v4_power) <= 0.0:
            raise ValueError("--v4-power must be a positive finite float.")
        if not math.isfinite(float(args.v4_zero)) or float(args.v4_zero) <= 0.0:
            raise ValueError("--v4-zero must be a positive finite float.")
    elif args.evaluate_combined_success and explicit_v4:
        raise ValueError("--discount, --v4-power, and --v4-zero are only valid for combined --xpass-weight v4.")
    elif explicit_v4 and not args.weighted_pass_success_metrics:
        raise ValueError("v4 options require --xpass-weight v4 or --weighted-pass-success-metrics.")


def add_weighted_pass_success_options(command: list[str], args: argparse.Namespace) -> list[str]:
    """Append the evaluation-only v4 options to a pass-success test command."""
    command.extend(
        [
            "--weighted-pass-success-metrics",
            "--discount",
            str(True if args.discount is None else bool(args.discount)).lower(),
            "--v4-power",
            str(4.0 if args.v4_power is None else args.v4_power),
            "--v4-zero",
            str(0.7 if args.v4_zero is None else args.v4_zero),
        ]
    )
    if getattr(args, "pc_xpass_id", None):
        command.extend(["--pc-xpass-id", str(args.pc_xpass_id)])
    if args.pc_xpass_cache_dir:
        command.extend(["--pc-xpass-cache-dir", str(args.pc_xpass_cache_dir)])
    return command


def explicit_model_ids(args: argparse.Namespace) -> dict[str, str]:
    return {
        task: model_id
        for task in SUPPORTED_EVALUATION_TASKS
        if (model_id := getattr(args, f"{task}_model_id", None))
    }


def requested_evaluation_tasks(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str], dict | None]:
    explicit = explicit_model_ids(args)
    bundle = load_bundle_record(args.bundle_id) if args.bundle_id else None
    bundle_ids = dict((bundle or {}).get("model_ids") or {})
    requested = [
        task for task in SUPPORTED_EVALUATION_TASKS if explicit.get(task) or bundle_ids.get(task)
    ]
    if not requested:
        if bundle is not None:
            raise ValueError(f"Bundle {args.bundle_id!r} contains none of the supported model IDs to evaluate.")
        raise ValueError("At least one explicit --<task>-model-id or --bundle-id is required.")
    return requested, explicit, bundle


def pass_success_uses_diagnostic_labels(args: argparse.Namespace, selected_tasks: list[str] | set[str]) -> bool:
    return "pass_success" in set(selected_tasks) and bool(
        not args.no_observed_pass_height_stratification
        or args.weighted_pass_success_metrics
        or args.evaluate_xpass
        or args.evaluate_combined_success
    )


def task_uses_diagnostic_feature_run(args: argparse.Namespace, task: str) -> bool:
    return task in {"pass_height", "outcome_scoring", "outcome_conceding"} or pass_success_uses_diagnostic_labels(
        args, {task}
    )


def validate_selected_task_options(args: argparse.Namespace, requested_tasks: list[str]) -> None:
    selected = set(requested_tasks)
    pass_success_uses_diagnostics = pass_success_uses_diagnostic_labels(args, selected)
    if (args.evaluate_xpass or args.evaluate_combined_success or args.weighted_pass_success_metrics) and (
        "pass_success" not in selected
    ):
        raise ValueError("Pass-success evaluation options require pass_success to be selected.")
    if args.f1_outcome_threshold is not None and not selected.intersection(
        {"outcome_scoring", "outcome_conceding"}
    ):
        raise ValueError("--f1-outcome-threshold requires a selected outcome model.")
    if (
        args.diagnostic_feature_run_id
        and not selected.intersection({"pass_height", "outcome_scoring", "outcome_conceding"})
        and not pass_success_uses_diagnostics
    ):
        raise ValueError(
            "--diagnostic-feature-run-id requires a selected pass_height or outcome model, or a pass_success "
            "evaluation that uses pass-height diagnostics."
        )


def add_task_evaluation_options(command: list[str], args: argparse.Namespace, task: str) -> list[str]:
    """Append task-specific, evaluation-only CLI options."""
    if task in {"outcome_scoring", "outcome_conceding"}:
        command.extend(["--outcome-bootstrap-resamples", str(getattr(args, "outcome_bootstrap_resamples", 2000)),
                        "--outcome-bootstrap-seed", str(getattr(args, "outcome_bootstrap_seed", 42))])
    if task == "pass_success" and not args.no_observed_pass_height_stratification:
        command.append("--observed-pass-height-stratification")
    if task in {"pass_success", "pass_height"}:
        command.extend(["--classification-threshold", str(args.classification_threshold)])
    if task == "pass_success" and (args.evaluate_xpass or args.evaluate_combined_success):
        if args.evaluate_xpass:
            command.append("--evaluate-xpass")
        if args.evaluate_combined_success:
            command.append("--evaluate-combined-success")
        command.extend(["--xpass-version", str(args.xpass_version)])
        if getattr(args, "pc_xpass_id", None):
            command.extend(["--pc-xpass-id", str(args.pc_xpass_id)])
        if args.pc_xpass_cache_dir:
            command.extend(["--pc-xpass-cache-dir", str(args.pc_xpass_cache_dir)])
        if args.evaluate_combined_success:
            command.extend(["--xpass-weight", str(args.xpass_weight)])
            if args.xpass_weight == "v4":
                command.extend(
                    [
                        "--discount", str(bool(args.discount)).lower(),
                        "--v4-power", str(args.v4_power),
                        "--v4-zero", str(args.v4_zero),
                    ]
                )
            if args.xpass_weight == "v5":
                command.extend(
                    [
                        "--pass-intent-model-id", str(args.pass_intent_model_id),
                        "--v5-intent-threshold", str(args.v5_intent_threshold),
                        "--v5-discount", str(bool(args.v5_discount)).lower(),
                    ]
                )
    if task in {"outcome_scoring", "outcome_conceding"} and args.f1_outcome_threshold is not None:
        command.extend(["--f1-outcome-threshold", str(args.f1_outcome_threshold)])
    return command


def model_evaluation_output_dir(base_dir: Path, task: str, evaluation_timestamp: str) -> Path:
    return base_dir / f"{task}_{evaluation_timestamp}"


def model_evaluation_output_dirs(
    base_dir: Path,
    tasks: list[str],
    evaluation_timestamp: str,
) -> dict[str, Path]:
    output_dirs = {
        task: model_evaluation_output_dir(base_dir, task, evaluation_timestamp)
        for task in tasks
    }
    existing_dirs = [path for path in output_dirs.values() if path.exists()]
    if existing_dirs:
        raise FileExistsError(
            "Evaluation artifact directory already exists; choose a different output location or retry after the next second: "
            + ", ".join(str(path) for path in existing_dirs)
        )
    return output_dirs


def update_model_metadata(output_dir: Path, wrapper_context: dict) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["wrapper_context"] = wrapper_context
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")


def learning_curve_selection(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict[str, str]]:
    if args.evaluate_xpass or args.evaluate_combined_success or args.weighted_pass_success_metrics:
        raise ValueError("Learning-curve mode compares learned checkpoints only; xPass and weighted options are unavailable.")
    bundle = load_bundle_record(args.bundle_id) if args.bundle_id else None
    anchors = {task: str(model_id) for task, model_id in dict((bundle or {}).get("model_ids") or {}).items()
               if task in SUPPORTED_EVALUATION_TASKS}
    anchors.update(explicit_model_ids(args))
    direct: dict[str, list[str]] = defaultdict(list)
    for model_id in args.learning_curve_model_id:
        task, _ = learning_curve.parse_model_id(model_id)
        if task not in SUPPORTED_EVALUATION_TASKS:
            raise ValueError(f"Unsupported learning-curve task in {model_id!r}.")
        direct[task].append(model_id)
    if not anchors and not direct:
        raise ValueError("--learning-curve requires --bundle-id, a --<task>-model-id, or --learning-curve-model-id.")
    selected: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for task in SUPPORTED_EVALUATION_TASKS:
        if task in direct:
            selected[task] = direct[task]
        elif task in anchors:
            ids = learning_curve.discover(anchors[task])
            if len(ids) < 2:
                skipped[task] = f"Only {len(ids)} completed expanding checkpoints found."
            else:
                selected[task] = ids
    return selected, skipped


def run_learning_curve(args: argparse.Namespace) -> None:
    selected, skipped = learning_curve_selection(args)
    if not selected:
        raise ValueError(f"No comparable checkpoint series found; skipped: {skipped}")
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base_dir = Path(args.evaluation_output_dir) if args.evaluation_output_dir else EVALUATION_RUNS_DIR
    output_root = base_dir / f"learning_curve_{timestamp}"
    if output_root.exists():
        raise FileExistsError(f"Learning-curve output already exists: {output_root}")
    series = {}
    direct_tasks = {learning_curve.parse_model_id(value)[0] for value in args.learning_curve_model_id}
    for task, ids in selected.items():
        try:
            series[task] = learning_curve.validate_series(ids, expected_task=task)
        except ValueError as exc:
            if task in direct_tasks:
                raise
            skipped[task] = str(exc)
    if not series:
        raise ValueError(f"No comparable checkpoint series found; skipped: {skipped}")
    output_root.mkdir(parents=True)
    report = {"evaluation_timestamp": timestamp, "bundle_id": args.bundle_id,
              "bootstrap_resamples": args.learning_curve_bootstrap_resamples,
              "bootstrap_seed": args.learning_curve_bootstrap_seed,
              "classification_threshold": args.classification_threshold,
              "f1_outcome_threshold": args.f1_outcome_threshold,
              "series": series, "skipped": skipped}
    manifest_path = output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for task, records in series.items():
        task_dir = output_root / task
        for record in records:
            model_dir = task_dir / str(record["train_matches"])
            command = [sys.executable, "test.py", "--model_id", record["model_id"], "--device", args.device,
                       "--evaluation-output-dir", str(model_dir), "--evaluation-timestamp", timestamp,
                       "--learning-curve-rows"]
            if args.diagnostic_feature_run_id and task_uses_diagnostic_feature_run(args, task):
                command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
            if task in {"pass_success", "pass_height"}:
                command.extend(["--classification-threshold", str(args.classification_threshold)])
            if task in {"outcome_scoring", "outcome_conceding"}:
                command.extend(["--outcome-bootstrap-resamples", "0"])
                if args.f1_outcome_threshold is not None:
                    command.extend(["--f1-outcome-threshold", str(args.f1_outcome_threshold)])
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        frames = learning_curve.load_aligned_predictions(records, task_dir)
        identity_columns = [name for name in ("match_id", "source_index", "target", "soft_target", "execution_branch")
                            if name in frames[0].columns]
        aligned = frames[0][identity_columns].copy()
        for record, frame in zip(records, frames):
            for name in ("prediction", "target_probability", "reciprocal_rank"):
                if name in frame.columns:
                    aligned[f"{name}_{record['train_matches']}"] = frame[name].to_numpy()
        aligned.to_csv(task_dir / "aligned_predictions.csv", index=False)
        report.setdefault("evaluated_cohorts", {})[task] = {
            "example_count": len(frames[0]),
            "contributing_match_ids": sorted(frames[0]["match_id"].unique().tolist()),
        }
        summary, differences = learning_curve.compare(
            task, records, frames, resamples=args.learning_curve_bootstrap_resamples,
            seed=args.learning_curve_bootstrap_seed,
            threshold=(args.classification_threshold if task in {"pass_success", "pass_height"}
                       else args.f1_outcome_threshold if task in {"outcome_scoring", "outcome_conceding"} else None),
        )
        summary.to_csv(task_dir / "learning_curve_metrics.csv", index=False)
        differences.to_csv(task_dir / "learning_curve_differences.csv", index=False)
        learning_curve.save_plot(summary, task_dir / "learning_curve.png", task)
        manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Learning-curve report saved to {output_root}")
    for task, reason in skipped.items():
        print(f"Skipped {task}: {reason}")


def main() -> None:
    args = parse_args()
    if args.learning_curve:
        run_learning_curve(args)
        return
    if args.learning_curve_model_id:
        raise ValueError("--learning-curve-model-id requires --learning-curve.")
    if not args.pc_xpass_cache_dir and (args.pc_xpass_id or args.weighted_pass_success_metrics or args.evaluate_xpass or args.evaluate_combined_success):
        pc_versions.cache_dir("sportec", args)
    validate_pass_success_predictor_args(args)
    required_tasks, explicit_ids, selected_bundle = requested_evaluation_tasks(args)
    validate_selected_task_options(args, required_tasks)
    resolved_model_ids, _, bundle = resolve_model_selection(
        required_tasks=required_tasks,
        bundle_id=args.bundle_id,
        explicit_model_ids=explicit_ids,
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )
    if not args.pc_xpass_cache_dir and not getattr(args, "_pc_version_root", None):
        if any(get_model_record(model_id)["args"].get("lane_survival", False) for model_id in resolved_model_ids.values()):
            pc_versions.cache_dir("sportec", args)
    python = sys.executable
    models_to_evaluate = [(task, resolved_model_ids[task]) for task in required_tasks]

    evaluation_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    evaluation_base_dir = Path(args.evaluation_output_dir) if args.evaluation_output_dir else EVALUATION_RUNS_DIR
    model_output_dirs = model_evaluation_output_dirs(
        evaluation_base_dir,
        [task for task, _ in models_to_evaluate],
        evaluation_timestamp,
    )
    wrapper_context = {
        "evaluation_timestamp": evaluation_timestamp,
        "resolved_model_ids": resolved_model_ids,
        "requested_tasks": required_tasks,
        "explicit_model_ids": {task: model_id for task, model_id in explicit_ids.items() if model_id},
        "bundle_model_ids": {
            task: model_id
            for task, model_id in dict((selected_bundle or {}).get("model_ids") or {}).items()
            if task in SUPPORTED_EVALUATION_TASKS
        },
        "evaluated_models": {task: model_id for task, model_id in models_to_evaluate},
        "bundle_id": args.bundle_id,
        "diagnostic_feature_run_id": args.diagnostic_feature_run_id,
        "evaluation_options": {
            "weighted_pass_success_metrics": bool(args.weighted_pass_success_metrics),
            "evaluate_xpass": bool(args.evaluate_xpass),
            "evaluate_combined_success": bool(args.evaluate_combined_success),
            "xpass_version": args.xpass_version,
            "xpass_weight": args.xpass_weight,
            "v5_intent_threshold": args.v5_intent_threshold if args.xpass_weight == "v5" else None,
            "v5_discount": args.v5_discount if args.xpass_weight == "v5" else None,
            "pass_intent_model_id": args.pass_intent_model_id if args.xpass_weight == "v5" else None,
            "observed_pass_height_stratification": not bool(args.no_observed_pass_height_stratification),
            "classification_threshold": args.classification_threshold,
            "f1_outcome_threshold": args.f1_outcome_threshold,
            "discount": args.discount,
            "v4_power": args.v4_power,
            "v4_zero": args.v4_zero,
            "pc_xpass_id": getattr(args, "pc_xpass_id", None),
            "pc_xpass_cache_dir": args.pc_xpass_cache_dir,
        },
        "evaluation_base_dir": str(evaluation_base_dir.resolve()),
    }
    print(f"Evaluation artifacts will be saved below {evaluation_base_dir}")

    for task, model_id in models_to_evaluate:
        evaluation_output_dir = model_output_dirs[task]
        command = [python, "test.py", "--model_id", model_id, "--device", args.device]
        if args.diagnostic_feature_run_id and task_uses_diagnostic_feature_run(args, task):
            command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
        command.extend(
            [
                "--evaluation-output-dir",
                str(evaluation_output_dir),
                "--evaluation-timestamp",
                evaluation_timestamp,
            ]
        )
        if args.weighted_pass_success_metrics and task == "pass_success":
            add_weighted_pass_success_options(command, args)
        add_task_evaluation_options(command, args, task)
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
        update_model_metadata(evaluation_output_dir, wrapper_context)


if __name__ == "__main__":
    main()
