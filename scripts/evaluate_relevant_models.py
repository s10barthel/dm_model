from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.utils import load_bundle_record, resolve_model_selection
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
    parser.add_argument("--bundle-id", default=None)
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
    parser.add_argument("--xpass-weight", choices=["v1", "v2", "v3", "v4"], default=None)
    parser.add_argument("--no-observed-pass-height-stratification", action="store_true")
    parser.add_argument("--classification-threshold", type=probability_threshold, default=0.5)
    parser.add_argument("--f1-outcome-threshold", type=probability_threshold, default=None)
    parser.add_argument("--pc-xpass-cache-dir", default=None)
    parser.add_argument("--discount", type=parse_bool_text, default=None)
    parser.add_argument("--v4-power", type=float, default=None)
    parser.add_argument("--v4-zero", type=float, default=None)
    return parser.parse_args(argv)


def validate_pass_success_predictor_args(args: argparse.Namespace) -> None:
    enabled = bool(args.evaluate_xpass or args.evaluate_combined_success)
    if enabled and not args.xpass_version:
        raise ValueError("--evaluate-xpass/--evaluate-combined-success require --xpass-version.")
    if args.xpass_weight and not args.evaluate_combined_success:
        raise ValueError("--xpass-weight requires --evaluate-combined-success.")
    if args.evaluate_combined_success and not args.xpass_weight:
        raise ValueError("--evaluate-combined-success requires --xpass-weight.")
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


def main() -> None:
    args = parse_args()
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
            "observed_pass_height_stratification": not bool(args.no_observed_pass_height_stratification),
            "classification_threshold": args.classification_threshold,
            "f1_outcome_threshold": args.f1_outcome_threshold,
            "discount": args.discount,
            "v4_power": args.v4_power,
            "v4_zero": args.v4_zero,
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
