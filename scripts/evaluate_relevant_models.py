from __future__ import annotations

import argparse
import hashlib
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
from scripts import learning_curve, model_comparison
from scripts.comparison_warnings import ComparisonWarnings, ComparisonCancelled, add_warning_argument
from models.evaluation_context import add_evaluation_arguments, evaluation_selection, check_heldout
from models.utils import load_bundle_record
from models.outcome_bootstrap import add_outcome_bootstrap_arguments, nonnegative_integer
from models.inference_masks import (
    add_inference_mask_arguments, inference_mask_catalog,
    resolve_inference_mask_configurations, resolve_scoped_masks, resolve_model_masks,
)
import pc_xpass_versions as pc_versions
from pc_xpass_auxiliary import add_auxiliary_arguments, validate_auxiliary_selectors

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


class RepeatableModelId(argparse.Action):
    """Keep single-selector attributes compatible while accepting repeated IDs."""
    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        setattr(namespace, self.dest, values if current is None else model_id_list(current) + [values])


def model_id_list(value) -> list[str]:
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected checkpoints independently or compare models and inference masks.",
        epilog="For separate mask configurations, use --inference-mask-1 MASK ... --inference-mask-2 MASK ...; each positive number identifies a configuration.",
    )
    from models.pass_height import add_pass_height_argument
    add_pass_height_argument(parser)
    add_evaluation_arguments(parser)
    add_warning_argument(parser)
    add_outcome_bootstrap_arguments(parser)
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--compare", action="store_true", help="Compare at least two checkpoints of one task; masks compare within each checkpoint instead.")
    parser.add_argument("--comparison-bootstrap-resamples", type=nonnegative_integer, default=2000)
    parser.add_argument("--comparison-bootstrap-seed", type=nonnegative_integer, default=42)
    add_inference_mask_arguments(parser, numbered=True, argv=argv)
    parser.add_argument("--learning-curve", action="store_true")
    parser.add_argument("--learning-curve-model-id", action="append", default=[])
    parser.add_argument("--learning-curve-bootstrap-resamples", type=nonnegative_integer, default=2000)
    parser.add_argument("--learning-curve-bootstrap-seed", type=nonnegative_integer, default=42)
    for task in SUPPORTED_EVALUATION_TASKS:
        parser.add_argument(f"--{task.replace('_', '-')}-model-id", action=RepeatableModelId,
                            help="Task-qualified checkpoint ID; repeat to evaluate more checkpoints.")
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
    add_auxiliary_arguments(parser)
    args = parser.parse_args(argv)
    pc_versions.check_selectors(args)
    return resolve_top_pass_selector(parser, args, pc_only=True)


def validate_pass_success_predictor_args(args: argparse.Namespace) -> None:
    validate_auxiliary_selectors(args)
    enabled = bool(args.evaluate_xpass or args.evaluate_combined_success)
    if enabled and not args.xpass_version:
        raise ValueError("--evaluate-xpass/--evaluate-combined-success require --xpass-version.")
    if args.xpass_weight and not args.evaluate_combined_success:
        raise ValueError("--xpass-weight requires --evaluate-combined-success.")
    if args.evaluate_combined_success and not args.xpass_weight:
        raise ValueError("--evaluate-combined-success requires --xpass-weight.")
    if args.evaluate_combined_success and args.xpass_weight == "v5":
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
    if getattr(args, "diagnostic_return_type", None) is not None and not selected.intersection({"outcome_scoring", "outcome_conceding"}):
        raise ValueError("--diagnostic-return-type requires a selected outcome model.")
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
    if getattr(args, "evaluation_filter", None) is not None:
        command.extend(["--filter", args.evaluation_filter])
    for name in ("intended_receiver_mode", "diagnostic_return_type", "min_pass_duration", "season_half"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for season in getattr(args, "season", None) or []:
        command.extend(["--season", season])
    if getattr(args, "pass_height_threshold", None) is not None:
        command.extend(["--pass-height-threshold", str(args.pass_height_threshold)])
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
                        "--v5-intent-threshold", str(args.v5_intent_threshold),
                        "--v5-discount", str(bool(args.v5_discount)).lower(),
                    ]
                )
    if task == "pass_success":
        for role in ("pass_height", "pass_intent"):
            value = getattr(args, f"xpass_{role}_model_id", None)
            if value:
                command.extend(["--xpass-" + role.replace("_", "-") + "-model-id", value])
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
    if args.compare or resolve_inference_mask_configurations(args):
        raise ValueError("Learning-curve mode cannot be combined with --compare or inference masks.")
    if args.evaluate_xpass or args.evaluate_combined_success or args.weighted_pass_success_metrics:
        raise ValueError("Learning-curve mode compares learned checkpoints only; xPass and weighted options are unavailable.")
    bundle = load_bundle_record(args.bundle_id) if args.bundle_id else None
    anchors = {task: str(model_id) for task, model_id in dict((bundle or {}).get("model_ids") or {}).items()
               if task in SUPPORTED_EVALUATION_TASKS}
    explicit = explicit_model_ids(args)
    if any(len(model_id_list(value)) != 1 for value in explicit.values()):
        raise ValueError("Learning-curve task selectors accept one anchor per task; use --learning-curve-model-id for direct series.")
    anchors.update(explicit)
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
            stage_ids = ((bundle or {}).get("stage_model_ids") or {}).get(task)
            if stage_ids and anchors[task] == ((bundle or {}).get("model_ids") or {}).get(task):
                ids = list(stage_ids)
            else:
                ids = learning_curve.discover(anchors[task])
            if len(ids) < 2:
                skipped[task] = f"Only {len(ids)} completed learning-curve checkpoints found."
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
              "evaluation_pass_height_threshold_meters": args.pass_height_threshold,
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
            if args.pass_height_threshold is not None:
                command.extend(["--pass-height-threshold", str(args.pass_height_threshold)])
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


def resolve_evaluation_entries(args: argparse.Namespace) -> tuple[list[dict], dict, dict | None]:
    tasks, explicit, bundle = requested_evaluation_tasks(args)
    validate_selected_task_options(args, tasks)
    bundle_ids = dict((bundle or {}).get("model_ids") or {})
    entries = []
    seen = set()
    for task in tasks:
        ids = model_id_list(explicit.get(task) or bundle_ids.get(task))
        for model_id in ids:
            if model_id in seen:
                raise ValueError(f"Duplicate checkpoint selection: {model_id}.")
            seen.add(model_id)
            entry = model_comparison.checkpoint(model_id, expected_task=task)
            if args.pass_height_threshold is not None and "pass_height_definition" in entry["target_definitions"]:
                entry["target_definitions"]["pass_height_definition"] = {
                    "threshold_meters": args.pass_height_threshold, "operator": ">=", "target_source": "pass_max_ball_z",
                }
            entries.append(entry)
    return entries, explicit, bundle


def _entry_name(model_id: str) -> str:
    readable = "_".join(model_id.split("/")[1:])
    readable = "".join(value if value.isalnum() or value in "_-" else "_" for value in readable)[:80]
    return f"{readable}_{hashlib.sha256(model_id.encode()).hexdigest()[:8]}"


def evaluation_options(args: argparse.Namespace) -> dict:
    return {
        "filter": getattr(args, "evaluation_filter", None),
        "min_pass_duration": args.min_pass_duration,
        "weighted_pass_success_metrics": bool(args.weighted_pass_success_metrics),
        "evaluate_xpass": bool(args.evaluate_xpass),
        "evaluate_combined_success": bool(args.evaluate_combined_success),
        "xpass_version": args.xpass_version, "xpass_weight": args.xpass_weight,
        "v5_intent_threshold": args.v5_intent_threshold if args.xpass_weight == "v5" else None,
        "v5_discount": args.v5_discount if args.xpass_weight == "v5" else None,
        "xpass_pass_height_model_id": getattr(args, "xpass_pass_height_model_id", None),
        "xpass_pass_intent_model_id": getattr(args, "xpass_pass_intent_model_id", None),
        "observed_pass_height_stratification": not bool(args.no_observed_pass_height_stratification),
        "classification_threshold": args.classification_threshold,
        "f1_outcome_threshold": args.f1_outcome_threshold,
        "discount": args.discount, "v4_power": args.v4_power, "v4_zero": args.v4_zero,
        "pc_xpass_id": getattr(args, "pc_xpass_id", None), "pc_xpass_cache_dir": args.pc_xpass_cache_dir,
    }


def _comparison_threshold(args: argparse.Namespace, task: str) -> float | None:
    if task in {"pass_success", "pass_height"}:
        return args.classification_threshold
    if task in {"outcome_scoring", "outcome_conceding"}:
        return args.f1_outcome_threshold
    return None


def _participating_entry_tasks(args: argparse.Namespace, task: str) -> set[str]:
    tasks = {task}
    if task == "pass_success":
        tasks.update(validate_auxiliary_selectors(args))
    return tasks


def _write_manifest(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _main(args, policy) -> None:
    if args.list_inference_masks:
        print(json.dumps(inference_mask_catalog(), indent=2))
        return
    configurations = resolve_inference_mask_configurations(args)
    if args.learning_curve:
        if any(getattr(args, name, None) is not None for name in (
                "evaluation_filter", "intended_receiver_mode", "diagnostic_return_type", "min_pass_duration", "season", "season_half")):
            raise ValueError("Evaluation overrides cannot be combined with --learning-curve.")
        run_learning_curve(args)
        return
    if args.learning_curve_model_id:
        raise ValueError("--learning-curve-model-id requires --learning-curve.")
    validate_pass_success_predictor_args(args)
    entries, explicit, selected_bundle = resolve_evaluation_entries(args)
    all_entries = list(entries)
    mask_mode = bool(configurations)
    checkpoint_mode = bool(args.compare and not mask_mode)
    if checkpoint_mode:
        if args.bundle_id:
            raise ValueError("--compare without masks requires direct task-specific selections without --bundle-id.")
        subjects = entries
        if len(subjects) < 2 or len({entry["task"] for entry in subjects}) != 1:
            raise ValueError("--compare requires at least two distinct checkpoints of the same task.")
        entries = subjects
    selected_matches = evaluation_selection(args)
    exclusion_checks = []
    for entry in entries:
        population = selected_matches if selected_matches is not None else entry["test_match_ids"]
        if population is None:
            raise ValueError(f"No verified held-out population for {entry['model_id']}.")
        exclusion_checks.append(check_heldout(entry["checkpoint_record"], population))
        entry["test_match_ids"] = list(population)
        if args.intended_receiver_mode is not None:
            entry["target_definitions"]["intended_receiver_mode"] = args.intended_receiver_mode
    auxiliary_ids = [getattr(args, "xpass_" + role + "_model_id", None) for role in ("pass_height", "pass_intent")]
    for model_id in filter(None, auxiliary_ids):
        auxiliary = model_comparison.checkpoint(model_id)
        for entry in entries:
            exclusion_checks.append(check_heldout(auxiliary["checkpoint_record"], entry["test_match_ids"]))
    if checkpoint_mode:
        model_comparison.add_definition_warnings(policy, entries)
    participating_tasks = {entry["task"] for entry in entries}
    participating_tasks.update(validate_auxiliary_selectors(args))
    for configuration in configurations:
        configuration["scoped_masks"] = resolve_scoped_masks(configuration["masks"], participating_tasks)
        for entry in all_entries:
            names = configuration["scoped_masks"].get(entry["task"], [])
            if names:
                resolve_model_masks(entry["checkpoint_record"]["args"], names)
    if not args.pc_xpass_cache_dir and (args.pc_xpass_id or args.weighted_pass_success_metrics or args.evaluate_xpass or args.evaluate_combined_success):
        pc_versions.cache_dir("sportec", args)
    if not args.pc_xpass_cache_dir and not getattr(args, "_pc_version_root", None):
        if any(entry["checkpoint_record"]["args"].get("lane_survival", False) for entry in entries):
            pc_versions.cache_dir("sportec", args)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base_dir = Path(args.evaluation_output_dir) if args.evaluation_output_dir else EVALUATION_RUNS_DIR
    mode = "mask_comparison" if mask_mode else "model_comparison" if checkpoint_mode else "evaluation"
    repeated_tasks = len({entry["task"] for entry in entries}) < len(entries)
    grouped_output = mask_mode or checkpoint_mode or repeated_tasks
    output_root = base_dir / f"{mode}_{timestamp}"
    if grouped_output and output_root.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_root}")
    legacy_dirs = {} if grouped_output else model_evaluation_output_dirs(base_dir, [entry["task"] for entry in entries], timestamp)
    manifest_path = output_root / "experiment_manifest.json" if grouped_output else base_dir / f"evaluation_{timestamp}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Evaluation manifest already exists: {manifest_path}")
    selected_ids = defaultdict(list)
    for entry in entries:
        selected_ids[entry["task"]].append(entry["model_id"])
    resolved_ids = {task: values[0] if len(values) == 1 else values for task, values in selected_ids.items()}
    wrapper_context = {
        "mode": mode, "evaluation_timestamp": timestamp, "resolved_model_ids": resolved_ids,
        "requested_tasks": list(selected_ids), "explicit_model_ids": explicit,
        "bundle_model_ids": {task: model_id for task, model_id in dict((selected_bundle or {}).get("model_ids") or {}).items()
                             if task in SUPPORTED_EVALUATION_TASKS},
        "evaluated_models": resolved_ids, "bundle_id": args.bundle_id,
        "diagnostic_feature_run_id": args.diagnostic_feature_run_id,
        "pass_height_threshold_meters": args.pass_height_threshold, "evaluation_options": evaluation_options(args),
        "evaluation_base_dir": str(base_dir.resolve()),
        "mask_configurations": configurations,
        "comparison_bootstrap_resamples": args.comparison_bootstrap_resamples,
        "comparison_bootstrap_seed": args.comparison_bootstrap_seed,
        "comparison_interval_method": "paired_match_percentile",
        "comparison_confidence_level": 0.95,
        "comparison_interval_scope": "pointwise",
        "evaluation_context": {"intended_receiver_mode": args.intended_receiver_mode,
            "filter": args.evaluation_filter,
            "diagnostic_target_family": "goal", "diagnostic_return_type": args.diagnostic_return_type or "next_10",
            "min_pass_duration": args.min_pass_duration, "seasons": args.season,
            "season_half": args.season_half, "requested_match_ids": selected_matches,
            "exclusion_checks": exclusion_checks},
    }
    report = {**wrapper_context, **policy.metadata(), "entries": [], "comparisons": [], "skipped_configurations": []}
    args._comparison_manifest_path = manifest_path
    _write_manifest(manifest_path, report)
    policy.confirm()
    print(f"Evaluation artifacts will be saved below {base_dir}", flush=True)
    compare_records = []
    for entry in entries:
        task, model_id = entry["task"], entry["model_id"]
        model_dir = output_root / task / _entry_name(model_id)
        variants = [{"id": "unmasked", "masks": []}]
        if mask_mode:
            entry_tasks = _participating_entry_tasks(args, task)
            for configuration in configurations:
                scoped = configuration["scoped_masks"]
                values = [f"{scope}:{name}" for scope, names in scoped.items() if scope in entry_tasks for name in names]
                if not values:
                    report["skipped_configurations"].append({"model_id": model_id, "configuration_id": configuration["id"],
                                                             "reason": "Configuration affects neither this checkpoint nor its auxiliary dependencies."})
                    continue
                variants.append({"id": configuration["id"], "masks": values})
        records = []
        for variant in variants:
            output_dir = model_dir / variant["id"] if grouped_output else legacy_dirs[task]
            record = {**entry, "configuration_id": variant["id"], "masks": variant["masks"],
                      "output_dir": str(output_dir), "label": variant["id"] if mask_mode else model_id}
            command = [sys.executable, "test.py", "--model_id", model_id, "--device", args.device,
                       "--evaluation-output-dir", str(output_dir), "--evaluation-timestamp", timestamp]
            if grouped_output:
                command.extend(["--inference-configuration-id", variant["id"]])
            if mask_mode or checkpoint_mode:
                command.extend(["--learning-curve-rows", "--comparison-rows"])
            if variant["masks"]:
                command.extend(["--inference-mask", *variant["masks"]])
            if args.diagnostic_feature_run_id and task_uses_diagnostic_feature_run(args, task):
                command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
            if args.weighted_pass_success_metrics and task == "pass_success":
                add_weighted_pass_success_options(command, args)
            add_task_evaluation_options(command, args, task)
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            update_model_metadata(output_dir, {**wrapper_context, "configuration_id": variant["id"],
                                              "requested_inference_masks": variant["masks"]})
            artifact_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            record["inference_masks"] = artifact_metadata.get("inference_masks", {})
            record["evaluation_context"] = artifact_metadata.get("evaluation_context", {})
            records.append(record)
            report["entries"].append(record)
            _write_manifest(manifest_path, report)
        if mask_mode and len(records) >= 2:
            comparison = model_comparison.write_report(
                records, model_dir, resamples=args.comparison_bootstrap_resamples,
                seed=args.comparison_bootstrap_seed, threshold=_comparison_threshold(args, task),
                warning_policy=policy,
            )
            report["comparisons"].append({"model_id": model_id, "output_dir": str(model_dir),
                                          "warnings": comparison["warnings"], "example_count": comparison["example_count"]})
        compare_records.extend(records)
        _write_manifest(manifest_path, report)
    if checkpoint_mode:
        comparison = model_comparison.write_report(
            compare_records, output_root, resamples=args.comparison_bootstrap_resamples,
            seed=args.comparison_bootstrap_seed, threshold=_comparison_threshold(args, entries[0]["task"]),
            warning_policy=policy,
        )
        report["comparisons"].append({"task": entries[0]["task"], "output_dir": str(output_root),
                                      "warnings": comparison["warnings"], "example_count": comparison["example_count"]})
        _write_manifest(manifest_path, report)
    for skipped in report["skipped_configurations"]:
        print(f"Skipped {skipped['model_id']} {skipped['configuration_id']}: {skipped['reason']}", flush=True)
    print(f"Evaluation manifest saved to {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    policy = ComparisonWarnings(getattr(args, "accept_comparison_warnings", False))
    try:
        _main(args, policy)
    except Exception as exc:
        path = getattr(args, "_comparison_manifest_path", None)
        if path is not None:
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(policy.metadata(), status="cancelled" if isinstance(exc, ComparisonCancelled) else "failed", error=str(exc))
            _write_manifest(path, report)
        raise


if __name__ == "__main__":
    main()
