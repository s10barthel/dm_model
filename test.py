import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import ActionDataset, requires_goal_next10_diagnostics
from datatools import config
from models import utils
from models.dataset_config import build_action_dataset_kwargs
from models.utils import (
    calc_binary_metrics,
    calc_binary_diagnostic_rows,
    calc_continuous_target_metrics,
    calc_equal_frequency_bins,
    equal_frequency_slice_masks,
    get_losses_str,
    infer_feature_graph_schema,
    infer_training_edge_schema,
    load_splits,
    run_epoch,
)
from models.utils import validate_feature_graph_schema
from physical_pass_model import (
    PC_XPASS_SOURCE,
    PHYSICAL_XPASS_SOURCE,
    model_uses_physical_xpass,
    normalize_physical_xpass_speed_aggregation,
    pc_xpass_lane_survival_metadata_fingerprint,
    physical_xpass_metric_for_version,
    validate_physical_xpass_args,
    validate_physical_xpass_cache_metadata,
    validate_pc_xpass_lane_survival_mode_cache_metadata,
)
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    EVALUATION_RUNS_DIR,
    get_action_label_dir,
    get_pc_xpass_dir,
    get_physical_xpass_dir,
    load_feature_run_metadata,
    resolve_feature_root,
    resolve_feature_run_id,
)


def print_skipped_matches(name: str, dataset: ActionDataset, max_items: int = 10) -> None:
    skipped = getattr(dataset, "skipped_matches", {})
    if not skipped:
        return

    print(f"Skipped {len(skipped)} {name} matches due to unreadable or mismatched artifacts.")
    for match_id, reason in list(skipped.items())[:max_items]:
        print(f"  {match_id}: {reason}")
    if len(skipped) > max_items:
        print(f"  ... and {len(skipped) - max_items} more")


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


def resolve_weighted_pass_success_cache(args: argparse.Namespace, model_args: argparse.Namespace) -> str | None:
    """Validate and return the pc-xPass cache for weighted pass-success evaluation."""
    if not bool(args.weighted_pass_success_metrics):
        return None
    if str(getattr(model_args, "task", "")) != "pass_success":
        raise ValueError("--weighted-pass-success-metrics requires a pass_success checkpoint.")
    v4_power = 4.0 if args.v4_power is None else float(args.v4_power)
    v4_zero = 0.7 if args.v4_zero is None else float(args.v4_zero)
    if not math.isfinite(v4_power) or v4_power <= 0.0:
        raise ValueError("--v4-power must be a positive finite float.")
    if not math.isfinite(v4_zero) or v4_zero <= 0.0:
        raise ValueError("--v4-zero must be a positive finite float.")

    cache_dir = Path(args.pc_xpass_cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
    cached_model_id = metadata.get("pass_height_model_id")
    if not cached_model_id:
        raise ValueError("Weighted pass-success evaluation requires pass-height provenance in cache metadata pass_height_model_id.")
    return str(cache_dir)


def resolve_evaluation_xpass_cache(
    args: argparse.Namespace, model_args: argparse.Namespace,
) -> tuple[str | None, str | None, dict | None]:
    """Resolve and strictly validate the read-only pc-xPass evaluation cache."""
    enabled = bool(args.evaluate_xpass or args.evaluate_combined_success)
    if not enabled:
        return None, None, None
    if str(getattr(model_args, "task", "")) != "pass_success":
        raise ValueError("Physical xPass evaluation requires a pass_success checkpoint.")
    if not args.evaluation_output_dir:
        raise ValueError("Physical xPass evaluation requires --evaluation-output-dir.")
    if not args.xpass_version:
        raise ValueError("Physical xPass evaluation requires --xpass-version.")
    if args.xpass_weight and not args.evaluate_combined_success:
        raise ValueError("--xpass-weight requires --evaluate-combined-success.")
    if args.evaluate_combined_success and not args.xpass_weight:
        raise ValueError("--evaluate-combined-success requires --xpass-weight.")
    if args.evaluate_combined_success and args.xpass_weight == "v4":
        if args.discount is None or args.v4_power is None or args.v4_zero is None:
            raise ValueError("Combined v4 evaluation requires explicit --discount, --v4-power, and --v4-zero.")
        if not math.isfinite(float(args.v4_power)) or float(args.v4_power) <= 0.0:
            raise ValueError("--v4-power must be a positive finite float.")
        if not math.isfinite(float(args.v4_zero)) or float(args.v4_zero) <= 0.0:
            raise ValueError("--v4-zero must be a positive finite float.")
    elif args.evaluate_combined_success and any(
        value is not None for value in (args.discount, args.v4_power, args.v4_zero)
    ):
        raise ValueError("v4 options are only valid with combined --xpass-weight v4.")

    cache_dir = Path(args.pc_xpass_cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
    metric = physical_xpass_metric_for_version(args.xpass_version, pc_xpass=True)
    available = {str(value) for value in metadata.get("available_metrics", [])}
    if metric not in available:
        raise ValueError(
            f"Requested pc-xPass metric {metric!r} is not available in {cache_dir}; available={sorted(available)}."
        )
    if args.evaluate_combined_success and args.xpass_weight == "v4" and not metadata.get("pass_height_model_id"):
        raise ValueError("Combined v4 evaluation requires cache metadata pass_height_model_id provenance.")
    metadata_path = cache_dir / "metadata.json"
    metadata = dict(metadata)
    metadata["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    return str(cache_dir), metric, metadata


def resolve_goal_next10_diagnostic_context(
    cli_args: argparse.Namespace,
    model_args: argparse.Namespace,
    feature_root: Path,
) -> tuple[str | None, str | None]:
    if not requires_goal_next10_diagnostics(getattr(model_args, "task", None)):
        return None, None

    mode = getattr(model_args, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE
    if mode == "unknown":
        mode = DEFAULT_INTENDED_RECEIVER_MODE

    if cli_args.diagnostic_feature_run_id:
        diagnostic_feature_run_id = resolve_feature_run_id(
            cli_args.diagnostic_feature_run_id,
            required=True,
            allow_latest=False,
        )
        diagnostic_root = resolve_feature_root(diagnostic_feature_run_id)
        diagnostic_label_dir = get_action_label_dir(
            config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
            intended_receiver_mode=mode,
            root=diagnostic_root,
        )
        if not diagnostic_label_dir.exists():
            raise FileNotFoundError(f"Canonical goal-next10 diagnostic labels not found at {diagnostic_label_dir}.")
        return diagnostic_feature_run_id, str(diagnostic_label_dir)

    model_diagnostic_label_dir = getattr(model_args, "diagnostic_label_dir", None)
    if model_diagnostic_label_dir and Path(model_diagnostic_label_dir).exists():
        return getattr(model_args, "diagnostic_feature_run_id", None), str(model_diagnostic_label_dir)

    selected_label_dir = get_action_label_dir(
        config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
        intended_receiver_mode=mode,
        root=feature_root,
    )
    if selected_label_dir.exists():
        return getattr(model_args, "feature_run_id", None), str(selected_label_dir)
    return getattr(model_args, "feature_run_id", None), None


def validate_diagnostic_feature_run_ancestry(
    diagnostic_feature_run_id: str,
    selected_feature_run_id: str,
) -> list[str]:
    """Return the diagnostic lineage after proving it reaches the selected feature run."""
    current = str(diagnostic_feature_run_id)
    expected = str(selected_feature_run_id)
    lineage: list[str] = []
    seen: set[str] = set()
    while current:
        if current in seen:
            raise ValueError(f"Feature-run ancestry contains a cycle at {current!r}.")
        seen.add(current)
        lineage.append(current)
        if current == expected:
            return lineage
        metadata = load_feature_run_metadata(current, required=True) or {}
        current = str(metadata.get("derived_from_feature_run_id") or "")
    raise ValueError(
        f"Diagnostic feature run {diagnostic_feature_run_id!r} is not equal to or derived from "
        f"checkpoint feature run {selected_feature_run_id!r}."
    )


def resolve_pass_height_diagnostic_context(
    cli_args: argparse.Namespace,
    model_args: argparse.Namespace,
    selected_feature_run_id: str | None,
    *,
    required: bool,
) -> tuple[str | None, str | None, float | None, list[str] | None]:
    task = str(getattr(model_args, "task", ""))
    if not cli_args.diagnostic_feature_run_id or task not in {"pass_success", "pass_height"}:
        return None, None, None, None
    if not required:
        raise ValueError(
            "--diagnostic-feature-run-id is unused for pass_success when no pass-height evaluation is enabled."
        )
    if not selected_feature_run_id:
        raise ValueError("Pass-height diagnostic evaluation requires a checkpoint feature_run_id.")

    diagnostic_run_id = resolve_feature_run_id(
        cli_args.diagnostic_feature_run_id,
        required=True,
        allow_latest=False,
    )
    lineage = validate_diagnostic_feature_run_ancestry(diagnostic_run_id, str(selected_feature_run_id))
    metadata = load_feature_run_metadata(diagnostic_run_id, required=True) or {}
    threshold = metadata.get("pass_height_threshold_meters")
    if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError(
            f"Diagnostic feature run {diagnostic_run_id!r} must record a positive finite "
            "pass_height_threshold_meters value."
        )

    mode = getattr(model_args, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE
    if mode == "unknown":
        mode = DEFAULT_INTENDED_RECEIVER_MODE
    return_type = getattr(model_args, "return_type", None)
    if not return_type:
        raise ValueError(f"{task} checkpoint does not record return_type for diagnostic-label resolution.")
    diagnostic_root = resolve_feature_root(diagnostic_run_id)
    label_dir = get_action_label_dir(str(return_type), intended_receiver_mode=mode, root=diagnostic_root)
    if not label_dir.exists():
        raise FileNotFoundError(f"Pass-height diagnostic labels not found at {label_dir}.")
    return diagnostic_run_id, str(label_dir), float(threshold), lineage


def resolve_physical_xpass_context(
    model_args: argparse.Namespace,
    feature_root: Path,
    *,
    prefer_feature_root: bool = False,
) -> str | None:
    if not model_uses_physical_xpass(model_args):
        return None

    canonical_cache_dir = get_physical_xpass_dir(feature_root)
    recorded_cache_value = getattr(model_args, "physical_cache_dir", None)
    recorded_cache_dir = Path(recorded_cache_value) if recorded_cache_value else None
    if prefer_feature_root and canonical_cache_dir.exists():
        cache_dir = canonical_cache_dir
    elif recorded_cache_dir is not None and recorded_cache_dir.exists():
        cache_dir = recorded_cache_dir
    elif canonical_cache_dir.exists():
        cache_dir = canonical_cache_dir
    else:
        cache_dir = canonical_cache_dir

    model_args.physical_cache_dir = str(cache_dir)
    validate_physical_xpass_args(model_args)
    expected_source = getattr(model_args, "physical_xpass_source", PHYSICAL_XPASS_SOURCE)
    expected_speed_aggregation = getattr(model_args, "physical_xpass_speed_aggregation", None)
    metadata = validate_physical_xpass_cache_metadata(
        cache_dir,
        expected_source=expected_source,
        expected_speed_aggregation=expected_speed_aggregation,
    )
    recorded_teammate_policy = getattr(model_args, "physical_xpass_teammate_policy", None)
    actual_teammate_policy = metadata.get("teammate_policy")
    if recorded_teammate_policy is not None and actual_teammate_policy != recorded_teammate_policy:
        raise ValueError(
            "Physical xPass cache teammate_policy does not match the checkpoint: "
            f"cache={actual_teammate_policy!r}, checkpoint={recorded_teammate_policy!r}."
        )
    model_args.physical_xpass_source = str(metadata.get("source", expected_source))
    if actual_teammate_policy is not None:
        model_args.physical_xpass_teammate_policy = str(actual_teammate_policy)
    model_args.physical_xpass_speed_aggregation = normalize_physical_xpass_speed_aggregation(
        metadata.get("speed_aggregation")
    )
    return str(cache_dir)


def resolve_lane_survival_context(model_args: argparse.Namespace) -> str | None:
    if not bool(getattr(model_args, "lane_survival", False)):
        return None

    recorded_cache_value = getattr(model_args, "lane_survival_cache_dir", None)
    recorded_cache_dir = Path(recorded_cache_value) if recorded_cache_value else None
    canonical_cache_dir = get_pc_xpass_dir("sportec")
    cache_dir = canonical_cache_dir if canonical_cache_dir.exists() else recorded_cache_dir or canonical_cache_dir
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Lane-survival evaluation requires pc-xPass metadata at {metadata_path}. "
            "Run scripts/generate_physical_xpass.py --pc-xpass first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_args.lane_survival_mode = validate_pc_xpass_lane_survival_mode_cache_metadata(
        metadata,
        getattr(model_args, "lane_survival_mode", None),
    )
    actual_fingerprint = pc_xpass_lane_survival_metadata_fingerprint(metadata)
    expected_fingerprint = getattr(model_args, "lane_survival_cache_fingerprint", None)
    if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Lane-survival pc-xPass cache does not match the checkpoint metadata fingerprint: "
            f"cache={actual_fingerprint}, checkpoint={expected_fingerprint}."
        )
    model_args.lane_survival_cache_dir = str(cache_dir)
    return str(cache_dir)


def validate_test_dataset_dimensions(test_dataset: ActionDataset, model_args: argparse.Namespace) -> None:
    if not test_dataset.features:
        return
    graph = test_dataset.features[0]
    actual_node_dim = int(graph.x.shape[1])
    expected_node_dim = int(getattr(model_args, "node_in_dim", actual_node_dim))
    if actual_node_dim != expected_node_dim:
        raise ValueError(
            f"Test dataset node feature width {actual_node_dim} does not match checkpoint node_in_dim={expected_node_dim}."
        )
    edge_attr = getattr(graph, "edge_attr", None)
    actual_edge_dim = int(edge_attr.shape[1]) if edge_attr is not None else 0
    expected_edge_dim = int(getattr(model_args, "edge_in_dim", actual_edge_dim))
    if actual_edge_dim != expected_edge_dim:
        raise ValueError(
            f"Test dataset edge feature width {actual_edge_dim} does not match checkpoint edge_in_dim={expected_edge_dim}."
        )


def _outcome_strata(execution_branch: np.ndarray) -> list[tuple[str, np.ndarray]]:
    branches = np.asarray(execution_branch, dtype=int).reshape(-1)
    return [
        ("pooled_factual", np.ones(len(branches), dtype=bool)),
        ("observed_success", branches == 1),
        ("observed_failure", branches == 0),
    ]


def _rounded_axis_upper(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    scaled = float(value) * 1.1
    magnitude = 10.0 ** math.floor(math.log10(scaled))
    return math.ceil(scaled / magnitude) * magnitude


def binned_relationship_axis_limits(
    binned: pd.DataFrame,
    *,
    evaluation_target: str,
    show_identity: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return readable plot limits from all finite binned values for one target."""
    rows = binned.loc[binned["evaluation_target"] == evaluation_target]
    prediction = pd.to_numeric(rows.get("mean_prediction", pd.Series(dtype=float)), errors="coerce").to_numpy()
    observed = pd.to_numeric(rows.get("mean_observed", pd.Series(dtype=float)), errors="coerce").to_numpy()
    finite_prediction = prediction[np.isfinite(prediction)]
    finite_observed = observed[np.isfinite(observed)]
    if not len(finite_prediction) and not len(finite_observed):
        return (0.0, 1.0), (0.0, 1.0)
    if show_identity:
        upper = _rounded_axis_upper(float(np.max(np.concatenate([finite_prediction, finite_observed]))))
        return (0.0, upper), (0.0, upper)
    x_upper = _rounded_axis_upper(float(np.max(finite_prediction))) if len(finite_prediction) else 1.0
    y_upper = _rounded_axis_upper(float(np.max(finite_observed))) if len(finite_observed) else 1.0
    return (0.0, x_upper), (0.0, y_upper)


def _save_binned_relationship_plot(
    binned: pd.DataFrame,
    *,
    evaluation_target: str,
    title: str,
    y_label: str,
    output_path: Path,
    show_identity: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    strata = ["pooled_factual", "observed_success", "observed_failure"]
    x_limits, y_limits = binned_relationship_axis_limits(
        binned,
        evaluation_target=evaluation_target,
        show_identity=show_identity,
    )
    fig, axes = plt.subplots(1, len(strata), figsize=(14, 4.5), sharex=True, sharey=True)
    for axis, stratum in zip(axes, strata):
        rows = binned.loc[
            (binned["evaluation_target"] == evaluation_target) & (binned["stratum"] == stratum)
        ]
        axis.set_title(stratum.replace("_", " "))
        if rows.empty:
            axis.text(0.5, 0.5, "No samples", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(rows["mean_prediction"], rows["mean_observed"], marker="o")
        if show_identity:
            axis.plot(x_limits, y_limits, "--", color="0.5", linewidth=1, label="identity")
        axis.set_xlabel("Mean model prediction")
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outcome_evaluation_artifacts(
    output_root: str | Path,
    *,
    model_id: str,
    task: str,
    outcome_evaluation: dict[str, np.ndarray],
    f1_outcome_threshold: float | None = None,
    n_bins: int = 10,
) -> tuple[Path, pd.DataFrame]:
    """Write factual outcome-model target-fidelity and external-validity artifacts."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.asarray(outcome_evaluation["prediction"], dtype=float).reshape(-1)
    targets = np.asarray(outcome_evaluation["target"], dtype=float).reshape(-1)
    diagnostics = np.asarray(outcome_evaluation["diagnostic"], dtype=float).reshape(-1)
    branches = np.asarray(outcome_evaluation["execution_branch"], dtype=int).reshape(-1)
    if not (len(predictions) == len(targets) == len(diagnostics) == len(branches)):
        raise ValueError("Outcome evaluation arrays must have identical lengths.")

    metric_rows: list[dict] = []
    bin_frames: list[pd.DataFrame] = []
    targets_by_name = {
        "xt_training_target": targets,
        "goal_next10_diagnostic": diagnostics,
    }
    for stratum, mask in _outcome_strata(branches):
        stratum_predictions = predictions[mask]
        for evaluation_target, observed in targets_by_name.items():
            stratum_observed = observed[mask]
            row = {
                "model_id": model_id,
                "task": task,
                "evaluation_target": evaluation_target,
                "stratum": stratum,
                "sample_count": int(len(stratum_predictions)),
                "positive_count": int(np.sum(stratum_observed > 0)) if evaluation_target == "goal_next10_diagnostic" else np.nan,
            }
            if len(stratum_predictions):
                if evaluation_target == "xt_training_target":
                    row.update(calc_continuous_target_metrics(stratum_observed, stratum_predictions))
                else:
                    row.update(
                        calc_binary_metrics(
                            stratum_observed,
                            stratum_predictions,
                            threshold=f1_outcome_threshold,
                        )
                    )
                binned = calc_equal_frequency_bins(stratum_observed, stratum_predictions, n_bins=n_bins)
                binned.insert(0, "stratum", stratum)
                binned.insert(0, "evaluation_target", evaluation_target)
                binned.insert(0, "task", task)
                binned.insert(0, "model_id", model_id)
                bin_frames.append(binned)
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "outcome_metrics.csv", index=False)
    binned = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame(
        columns=[
            "model_id", "task", "evaluation_target", "stratum", "bin", "sample_count",
            "mean_prediction", "mean_observed", "prediction_minus_observed",
        ]
    )
    binned.to_csv(output_dir / "calibration_bins.csv", index=False)
    _save_binned_relationship_plot(
        binned,
        evaluation_target="xt_training_target",
        title=f"Direct held-out xT-target calibration: {model_id}",
        y_label="Mean held-out xT-derived target",
        output_path=output_dir / "xt_target_calibration.png",
        show_identity=True,
    )
    _save_binned_relationship_plot(
        binned,
        evaluation_target="goal_next10_diagnostic",
        title=f"External validity: next-10-action goal association: {model_id}",
        y_label="Mean next-10-action goal indicator",
        output_path=output_dir / "goal_next10_association.png",
        show_identity=False,
    )
    return output_dir, metrics


def write_pass_success_height_metrics(output_root: str | Path, model_id: str, rows: list[dict]) -> Path:
    """Write observed-pass-height pass-success metrics as a portable table."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id", "stratum", "sample_count", "positive_count", "success_prevalence", "roc_auc", "brier"
    ]
    records = [{"model_id": model_id, **row} for row in rows]
    pd.DataFrame(records, columns=columns).to_csv(output_dir / "pass_success_height_metrics.csv", index=False)
    return output_dir / "pass_success_height_metrics.csv"


def write_pass_success_predictor_metrics(output_root: str | Path, model_id: str, rows: list[dict]) -> Path:
    """Write comparable learning, physical, and combined pass-success metrics."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id", "predictor", "stratum", "sample_count", "positive_count",
        "success_prevalence", "classification_threshold", "roc_auc", "pr_auc", "brier", "log_loss",
        "calibration_intercept", "calibration_slope", "ece", "precision", "recall", "f1",
        "true_positive", "false_positive", "true_negative", "false_negative",
    ]
    pd.DataFrame([{"model_id": model_id, **row} for row in rows], columns=columns).to_csv(
        output_dir / "pass_success_predictor_metrics.csv", index=False
    )
    return output_dir / "pass_success_predictor_metrics.csv"


def _write_rows(output_dir: Path, filename: str, model_id: str, rows: list[dict]) -> Path:
    records = [{"model_id": model_id, **row} for row in rows]
    path = output_dir / filename
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def write_pass_success_predictor_diagnostics(
    output_root: str | Path,
    model_id: str,
    diagnostics: dict,
    *,
    threshold: float,
) -> list[dict]:
    """Write full comparable diagnostics for learning, physical, and combined pass-success scores."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(diagnostics["targets"])
    predictors = diagnostics["predictors"]
    heights = np.asarray(diagnostics["observed_pass_high"])
    strata = (
        ("pooled", np.ones(len(targets), dtype=bool)),
        ("observed_high", heights == 1.0),
        ("observed_non_high", heights == 0.0),
    )
    metric_rows, calibration_rows, curve_rows = calc_binary_diagnostic_rows(
        targets, predictors, strata, threshold=threshold
    )
    write_pass_success_predictor_metrics(output_dir, model_id, metric_rows)
    _write_rows(output_dir, "pass_success_predictor_calibration_bins.csv", model_id, calibration_rows)
    _write_rows(output_dir, "pass_success_predictor_threshold_curve.csv", model_id, curve_rows)

    distance_rows: list[dict] = []
    for name, mask, lower, upper in equal_frequency_slice_masks(diagnostics["pass_distance"], "distance"):
        rows, _, _ = calc_binary_diagnostic_rows(targets, predictors, ((name, mask),), threshold=threshold)
        distance_rows.extend({"distance_lower": lower, "distance_upper": upper, **row} for row in rows)
    _write_rows(output_dir, "pass_success_predictor_distance_slices.csv", model_id, distance_rows)

    weight = diagnostics.get("combined_learning_weight")
    combined_names = [name for name in predictors if name.startswith("combined_")]
    physical_names = [name for name in predictors if name.startswith("physical_xpass_")]
    weight_rows: list[dict] = []
    if weight is not None and combined_names and physical_names:
        weight = np.asarray(weight, dtype=float)
        bounds = np.linspace(0.0, 1.0, 6)
        for index, (lower, upper) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            mask = (weight >= lower) & ((weight <= upper) if index == 5 else (weight < upper))
            rows, _, _ = calc_binary_diagnostic_rows(
                targets, {combined_names[0]: predictors[combined_names[0]]}, ((f"weight_{index}", mask),), threshold=threshold
            )
            for row in rows:
                row.update({
                    "learning_weight_lower": float(lower), "learning_weight_upper": float(upper),
                    "mean_learning_weight": float(weight[mask].mean()) if mask.any() else np.nan,
                    "mean_learning_probability": float(np.asarray(predictors["learning"])[mask].mean()) if mask.any() else np.nan,
                    "mean_physical_xpass": float(np.asarray(predictors[physical_names[0]])[mask].mean()) if mask.any() else np.nan,
                })
                weight_rows.append(row)
    _write_rows(output_dir, "pass_success_combined_weight_slices.csv", model_id, weight_rows)
    return metric_rows


def write_pass_height_diagnostics(
    output_root: str | Path,
    model_id: str,
    diagnostics: dict,
    *,
    threshold: float,
) -> list[dict]:
    """Write the same binary-probability diagnostic suite for pass-height predictions."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(diagnostics["targets"])
    predictors = {"pass_height": np.asarray(diagnostics["predictions"])}
    metric_rows, calibration_rows, curve_rows = calc_binary_diagnostic_rows(targets, predictors, threshold=threshold)
    _write_rows(output_dir, "pass_height_metrics.csv", model_id, metric_rows)
    _write_rows(output_dir, "pass_height_calibration_bins.csv", model_id, calibration_rows)
    _write_rows(output_dir, "pass_height_threshold_curve.csv", model_id, curve_rows)
    heights = np.asarray(diagnostics["observed_pass_max_height"], dtype=float)
    bands = (
        ("max_height_le_1_5m", heights <= 1.5),
        ("max_height_1_5_to_2_0m", (heights > 1.5) & (heights <= 2.0)),
        ("max_height_2_0_to_2_5m", (heights > 2.0) & (heights <= 2.5)),
        ("max_height_gt_2_5m", heights > 2.5),
    )
    slice_rows, _, _ = calc_binary_diagnostic_rows(targets, predictors, bands, threshold=threshold)
    _write_rows(output_dir, "pass_height_observed_height_slices.csv", model_id, slice_rows)
    return metric_rows


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pooled_outcome_summary_metrics(metrics: pd.DataFrame) -> dict[str, float | int | None]:
    pooled = metrics.loc[metrics["stratum"] == "pooled_factual"]
    summary: dict[str, float | int | None] = {}
    for _, row in pooled.iterrows():
        prefix = f"outcome_{row['evaluation_target']}"
        for column, value in row.items():
            if column in {"model_id", "task", "evaluation_target", "stratum"} or pd.isna(value):
                continue
            summary[f"{prefix}_{column}"] = value.item() if isinstance(value, np.generic) else value
    return summary


def write_model_evaluation_artifacts(
    output_dir: str | Path,
    *,
    model_id: str,
    task: str,
    feature_run_id: str | None,
    diagnostic_feature_run_id: str | None,
    evaluation_timestamp: str | None,
    evaluation_options: dict,
    test_metrics: dict,
    outcome_metrics: pd.DataFrame | None = None,
    pass_height_diagnostic_feature_run_id: str | None = None,
    pass_height_diagnostic_label_dir: str | None = None,
    observed_pass_height_threshold_meters: float | None = None,
    pass_height_diagnostic_ancestry_validated: bool = False,
) -> None:
    """Write portable per-model results and update the cross-run comparison table."""
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = evaluation_timestamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    metadata = {
        "evaluation_timestamp": timestamp,
        "model_id": str(model_id),
        "task": task,
        "feature_run_id": feature_run_id,
        "diagnostic_feature_run_id": diagnostic_feature_run_id,
        "pass_height_diagnostic_feature_run_id": pass_height_diagnostic_feature_run_id,
        "pass_height_diagnostic_label_dir": pass_height_diagnostic_label_dir,
        "observed_pass_height_threshold_meters": observed_pass_height_threshold_meters,
        "pass_height_diagnostic_ancestry_validated": bool(pass_height_diagnostic_ancestry_validated),
        "evaluation_options": _json_compatible(evaluation_options),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    metric_record = {
        "evaluation_timestamp": timestamp,
        "model_id": str(model_id),
        "task": task,
        "metrics": _json_compatible(test_metrics),
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metric_record, indent=2, allow_nan=False), encoding="utf-8")

    flat_record = {
        "evaluation_timestamp": timestamp,
        "model_id": str(model_id),
        "task": task,
        "feature_run_id": feature_run_id,
        "diagnostic_feature_run_id": diagnostic_feature_run_id,
        "pass_height_diagnostic_feature_run_id": pass_height_diagnostic_feature_run_id,
        "pass_height_diagnostic_label_dir": pass_height_diagnostic_label_dir,
        "observed_pass_height_threshold_meters": observed_pass_height_threshold_meters,
        "pass_height_diagnostic_ancestry_validated": bool(pass_height_diagnostic_ancestry_validated),
        **_json_compatible(evaluation_options),
        **_json_compatible(test_metrics),
    }
    if outcome_metrics is not None:
        flat_record.update(_json_compatible(_pooled_outcome_summary_metrics(outcome_metrics)))
    pd.DataFrame([flat_record]).to_csv(artifact_dir / "metrics.csv", index=False)

    summary_path = EVALUATION_RUNS_DIR / "metrics_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    row = pd.DataFrame([flat_record])
    if not existing.empty:
        existing = existing.loc[
            ~(
                existing["evaluation_timestamp"].astype(str).eq(str(timestamp))
                & existing["model_id"].astype(str).eq(str(model_id))
            )
        ]
    columns = list(dict.fromkeys([*existing.columns, *row.columns]))
    pd.concat([existing.reindex(columns=columns), row.reindex(columns=columns)], ignore_index=True).to_csv(
        summary_path,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="task/trial, e.g., pass_success/01")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--feature_dir", type=str, required=False, default=None)
    parser.add_argument("--feature-run-id", type=str, required=False, default=None)
    parser.add_argument("--train-split", type=int, default=None)
    parser.add_argument("--diagnostic-feature-run-id", type=str, required=False, default=None)
    parser.add_argument("--evaluation-output-dir", type=str, required=False, default=None)
    parser.add_argument("--evaluation-timestamp", type=str, required=False, default=None)
    parser.add_argument("--weighted-pass-success-metrics", action="store_true")
    parser.add_argument("--evaluate-xpass", action="store_true")
    parser.add_argument("--evaluate-combined-success", action="store_true")
    parser.add_argument("--xpass-version", default=None)
    parser.add_argument("--xpass-weight", choices=["v1", "v2", "v3", "v4"], default=None)
    parser.add_argument("--observed-pass-height-stratification", action="store_true")
    parser.add_argument("--classification-threshold", type=probability_threshold, default=0.5)
    parser.add_argument("--f1-outcome-threshold", type=probability_threshold, default=None)
    parser.add_argument("--pass-height-model-id", type=str, default=None)
    parser.add_argument("--pc-xpass-cache-dir", type=str, default=str(get_pc_xpass_dir("sportec")))
    parser.add_argument("--discount", type=parse_bool_text, default=None)
    parser.add_argument("--v4-power", type=float, default=None)
    parser.add_argument("--v4-zero", type=float, default=None)
    args, _ = parser.parse_known_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)
    weighted_pc_xpass_cache_dir = resolve_weighted_pass_success_cache(args, model_args)
    evaluation_xpass_cache_dir, evaluation_xpass_metric, evaluation_xpass_metadata = (
        resolve_evaluation_xpass_cache(args, model_args)
    )
    model_args.weighted_pass_success_metrics = bool(args.weighted_pass_success_metrics)
    model_args.observed_pass_height_stratification = bool(args.observed_pass_height_stratification)
    model_args.return_pass_success_height_evaluation = bool(
        args.observed_pass_height_stratification or args.evaluate_xpass or args.evaluate_combined_success
    )
    model_args.evaluate_xpass = bool(args.evaluate_xpass)
    model_args.evaluate_combined_success = bool(args.evaluate_combined_success)
    model_args.xpass_metric = evaluation_xpass_metric
    model_args.xpass_weight = args.xpass_weight
    model_args.classification_threshold = args.classification_threshold
    model_args.f1_outcome_threshold = args.f1_outcome_threshold
    model_args.return_binary_diagnostics = bool(
        args.evaluation_output_dir and getattr(model_args, "task", None) == "pass_height"
    )
    model_args.discount = True if args.discount is None else bool(args.discount)
    model_args.v4_power = 4.0 if args.v4_power is None else float(args.v4_power)
    model_args.v4_zero = 0.7 if args.v4_zero is None else float(args.v4_zero)

    print("\nGenerating test datasets...")
    resolved_feature_run_id = args.feature_run_id or getattr(model_args, "feature_run_id", None)
    if resolved_feature_run_id:
        feature_root = resolve_feature_root(resolved_feature_run_id)
        feature_name = Path(getattr(model_args, "feature_dir", "data/features/action_graphs")).name
        label_name = Path(
            getattr(model_args, "label_dir", f"data/features/action_labels_{model_args.return_type}")
        ).name
        feature_dir = args.feature_dir or str(feature_root / feature_name)
        label_dir = str(feature_root / label_name)
    else:
        feature_dir = args.feature_dir or getattr(model_args, "feature_dir", "data/features/action_graphs")
        label_dir = getattr(model_args, "label_dir", f"data/features/action_labels_{model_args.return_type}")
        feature_root = Path(feature_dir).parent
    diagnostic_feature_run_id, diagnostic_label_dir = resolve_goal_next10_diagnostic_context(args, model_args, feature_root)
    evaluated_task = str(getattr(model_args, "task", ""))
    pass_height_diagnostics_required = bool(
        (evaluated_task == "pass_height" and args.diagnostic_feature_run_id)
        or (
            evaluated_task == "pass_success"
            and (
                args.observed_pass_height_stratification
                or args.weighted_pass_success_metrics
                or args.evaluate_xpass
                or args.evaluate_combined_success
            )
        )
    )
    (
        pass_height_diagnostic_feature_run_id,
        pass_height_diagnostic_label_dir,
        observed_pass_height_threshold_meters,
        pass_height_diagnostic_lineage,
    ) = resolve_pass_height_diagnostic_context(
        args,
        model_args,
        resolved_feature_run_id,
        required=pass_height_diagnostics_required,
    )
    physical_cache_dir = resolve_physical_xpass_context(
        model_args,
        feature_root,
        prefer_feature_root=bool(
            args.feature_run_id and args.feature_run_id != getattr(model_args, "feature_run_id", None)
        ),
    )
    lane_survival_cache_dir = resolve_lane_survival_context(model_args)
    feature_schema = infer_feature_graph_schema(feature_dir)
    model_schema = {
        "edge_in_dim": int(getattr(model_args, "edge_in_dim", 2)),
        "add_v_edge_features": bool(getattr(model_args, "add_v_edge_features", getattr(model_args, "edge_in_dim", 2) > 2)),
        "add_relative_speed_edge_features": bool(
            getattr(model_args, "add_relative_speed_edge_features", getattr(model_args, "edge_in_dim", 2) > 4)
        ),
    }
    reconstructed_schema = infer_training_edge_schema(
        feature_schema,
        v_edge_feature_mode=getattr(model_args, "v_edge_feature_mode", None),
        relative_speed_edge_feature_mode=getattr(model_args, "relative_speed_edge_feature_mode", None),
    )
    if reconstructed_schema != model_schema:
        raise ValueError(
            "Checkpoint edge-feature settings do not reconstruct its recorded graph schema: "
            f"reconstructed={reconstructed_schema}, checkpoint={model_schema}."
        )
    validate_feature_graph_schema(feature_schema, model_schema, context="Selected feature artifacts")

    checkpoint_train_split = int(getattr(model_args, "train_split", getattr(model_args, "train_split_percent", 50)))
    if args.train_split is not None and int(args.train_split) != checkpoint_train_split:
        parser.error(
            f"--train-split {args.train_split} does not match checkpoint split {checkpoint_train_split}."
        )
    if resolved_feature_run_id:
        feature_metadata = load_feature_run_metadata(resolved_feature_run_id, required=False) or {}
        checkpoint_split_id = getattr(model_args, "split_manifest_id", None)
        feature_split_id = feature_metadata.get("split_manifest_id")
        if checkpoint_split_id and feature_split_id and checkpoint_split_id != feature_split_id:
            raise ValueError(
                f"Checkpoint split {checkpoint_split_id} does not match feature-run split {feature_split_id}."
            )
    _, _, test_match_ids = load_splits(feature_dir=feature_dir, train_split=checkpoint_train_split)

    dataset_args = build_action_dataset_kwargs(
        model_args,
        train=False,
        diagnostic_label_dir=diagnostic_label_dir,
        pass_height_diagnostic_label_dir=pass_height_diagnostic_label_dir,
        physical_cache_dir=physical_cache_dir,
        lane_survival_cache_dir=lane_survival_cache_dir,
    )
    if weighted_pc_xpass_cache_dir is not None:
        # These attributes are evaluation sidecars; they do not modify node or edge features.
        dataset_args["pass_height_cache_dir"] = weighted_pc_xpass_cache_dir
        dataset_args["require_observed_pass_height"] = True
    if evaluation_xpass_cache_dir is not None:
        dataset_args["evaluation_xpass_cache_dir"] = evaluation_xpass_cache_dir
        dataset_args["evaluation_xpass_metric"] = evaluation_xpass_metric
        dataset_args["evaluation_xpass_require_nearest"] = bool(
            args.evaluate_combined_success and args.xpass_weight == "v2"
        )
        dataset_args["evaluation_xpass_require_height"] = bool(
            args.evaluate_combined_success and args.xpass_weight == "v4"
        )
        dataset_args["require_pass_height_labels"] = True
    if args.observed_pass_height_stratification:
        if getattr(model_args, "task", None) != "pass_success":
            parser.error("--observed-pass-height-stratification requires a pass_success checkpoint")
        if not args.evaluation_output_dir:
            parser.error("--observed-pass-height-stratification requires --evaluation-output-dir")
        dataset_args["require_pass_height_labels"] = True
    if args.f1_outcome_threshold is not None and getattr(model_args, "task", None) not in {
        "outcome_scoring", "outcome_conceding"
    }:
        parser.error("--f1-outcome-threshold requires an outcome_scoring or outcome_conceding checkpoint")
    test_dataset = ActionDataset(
        test_match_ids,
        feature_dir=feature_dir,
        label_dir=label_dir,
        **dataset_args,
    )
    print_skipped_matches("test", test_dataset)
    if len(test_dataset) == 0:
        raise ValueError("No usable test samples remained after loading graph and label artifacts.")
    validate_test_dataset_dimensions(test_dataset, model_args)
    
    #On Windows, num_workers > 0 can cause issues with PyTorch DataLoader, so we set it to 0 for better compatibility. 
    #Adjust as needed for your environment.
    #test_loader = DataLoader(test_dataset, model_args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        model_args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(getattr(model_args, "pin_memory", False)),
    )
    print(f"Evaluating {args.model_id} on {len(test_match_ids)} matches with {len(test_dataset)} samples")
    collect_outcome_evaluation = bool(
        args.evaluation_output_dir and getattr(model_args, "task", None) in {"outcome_scoring", "outcome_conceding"}
    )
    result = run_epoch(
        model_args,
        model,
        test_loader,
        device=device,
        train=False,
        return_outcome_evaluation=collect_outcome_evaluation,
    )
    outcome_metrics = None
    pass_success_height_rows = None
    if collect_outcome_evaluation:
        test_metrics, outcome_evaluation = result
        if outcome_evaluation is None:
            raise RuntimeError("Outcome evaluation artifacts were requested, but no outcome predictions were collected.")
        artifact_dir, outcome_metrics = write_outcome_evaluation_artifacts(
            args.evaluation_output_dir,
            model_id=args.model_id,
            task=model_args.task,
            outcome_evaluation=outcome_evaluation,
            f1_outcome_threshold=args.f1_outcome_threshold,
        )
        print(f"Saved outcome evaluation artifacts to {artifact_dir}")
    elif model_args.return_pass_success_height_evaluation or model_args.return_binary_diagnostics:
        test_metrics, binary_evaluation = result
        if model_args.return_pass_success_height_evaluation:
            pass_success_height_rows = binary_evaluation["height_rows"]
            if args.observed_pass_height_stratification:
                write_pass_success_height_metrics(args.evaluation_output_dir, args.model_id, pass_success_height_rows)
            if binary_evaluation.get("predictor_diagnostics"):
                write_pass_success_predictor_diagnostics(
                    args.evaluation_output_dir,
                    args.model_id,
                    binary_evaluation["predictor_diagnostics"],
                    threshold=args.classification_threshold,
                )
        elif model_args.return_binary_diagnostics:
            write_pass_height_diagnostics(
                args.evaluation_output_dir,
                args.model_id,
                binary_evaluation,
                threshold=args.classification_threshold,
            )
    else:
        test_metrics = result
    if args.evaluation_output_dir:
        write_model_evaluation_artifacts(
            args.evaluation_output_dir,
            model_id=args.model_id,
            task=model_args.task,
            feature_run_id=resolved_feature_run_id,
            diagnostic_feature_run_id=diagnostic_feature_run_id,
            evaluation_timestamp=getattr(args, "evaluation_timestamp", None),
            evaluation_options={
                "weighted_pass_success_metrics": bool(args.weighted_pass_success_metrics),
                "evaluate_xpass": bool(args.evaluate_xpass),
                "evaluate_combined_success": bool(args.evaluate_combined_success),
                "xpass_version": args.xpass_version,
                "xpass_metric": evaluation_xpass_metric,
                "xpass_weight": args.xpass_weight,
                "xpass_cache_metadata_sha256": (
                    evaluation_xpass_metadata.get("metadata_sha256") if evaluation_xpass_metadata else None
                ),
                "xpass_pass_height_model_id": (
                    evaluation_xpass_metadata.get("pass_height_model_id") if evaluation_xpass_metadata else None
                ),
                "xpass_missing_data_policy": "error" if evaluation_xpass_metadata else None,
                "xpass_blend_formula": (
                    "(1 - learning_weight) * physical_xpass + learning_weight * learning_probability"
                    if args.evaluate_combined_success else None
                ),
                "pass_success_f1_threshold": args.classification_threshold if evaluation_xpass_metadata else None,
                "classification_threshold": args.classification_threshold,
                "observed_pass_height_stratification": bool(args.observed_pass_height_stratification),
                "pass_height_diagnostic_lineage": pass_height_diagnostic_lineage,
                "f1_outcome_threshold": args.f1_outcome_threshold,
                "discount": model_args.discount if args.evaluate_combined_success and args.xpass_weight == "v4" else args.discount,
                "v4_power": model_args.v4_power if args.evaluate_combined_success and args.xpass_weight == "v4" else args.v4_power,
                "v4_zero": model_args.v4_zero if args.evaluate_combined_success and args.xpass_weight == "v4" else args.v4_zero,
                "pc_xpass_cache_dir": args.pc_xpass_cache_dir,
            },
            test_metrics=test_metrics,
            outcome_metrics=outcome_metrics,
            pass_height_diagnostic_feature_run_id=pass_height_diagnostic_feature_run_id,
            pass_height_diagnostic_label_dir=pass_height_diagnostic_label_dir,
            observed_pass_height_threshold_meters=observed_pass_height_threshold_meters,
            pass_height_diagnostic_ancestry_validated=bool(pass_height_diagnostic_lineage),
        )
    print("Test:\t" + get_losses_str(test_metrics))
