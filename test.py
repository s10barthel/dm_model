import argparse
import hashlib
import json
import math
import os
import sys
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
from models.outcome_bootstrap import add_outcome_bootstrap_arguments, outcome_bootstrap
from models.dataset_config import build_action_dataset_kwargs
from models.edge_schema import edge_names, schema_for_names, INTERACTION_GROUPS
from models.inference_masks import (
    add_inference_mask_arguments,
    configure_inference_masks,
    resolve_inference_mask_configurations,
)
from models.prediction_identity import ensure_dataset_node_identities
from models.evaluation_context import (add_evaluation_arguments, evaluation_selection,
                                       diagnostic_name)
from models.membership import recorded_manifest_id
from models.evaluation_membership import (collect_evaluation_membership, confirm_direct_evaluation_membership,
                                          EvaluationMembershipCancelled)
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
from scripts.xpass_cli import add_top_pass_selector, resolve_top_pass_selector
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
import pc_xpass_versions as pc_versions
from pc_xpass_auxiliary import add_auxiliary_arguments, configure_auxiliary_models, auxiliary_metadata

from project_config import (
    resolve_artifact_split,
    split_dataset_provenance,
    add_split_arguments,
    split_metadata,
    split_selector,
    DEFAULT_INTENDED_RECEIVER_MODE,
    EVALUATION_RUNS_DIR,
    get_action_label_dir,
    get_pc_xpass_dir,
    get_physical_xpass_dir,
    load_feature_run_metadata,
    resolve_feature_root,
    resolve_feature_run_id,
)
from models.pass_height import (add_pass_height_argument, evaluation_height,
                               height_export_context, definition)



def validate_checkpoint_edge_schema(feature_schema, model_args):
    settings = vars(model_args) if not isinstance(model_args, dict) else model_args
    model_schema = schema_for_names(edge_names(settings), explicit=settings.get("edge_feature_names") is not None)
    reconstructed = infer_training_edge_schema(
        feature_schema,
        v_edge_feature_mode=settings.get("v_edge_feature_mode"),
        relative_speed_edge_feature_mode=settings.get("relative_speed_edge_feature_mode"),
        **{f"{group}_edge_feature_mode": settings.get(f"{group}_edge_feature_mode", "none")
           for group in INTERACTION_GROUPS},
    )
    if edge_names(reconstructed) != edge_names(model_schema):
        raise ValueError(
            "Checkpoint edge-feature settings do not reconstruct its recorded graph schema: "
            f"reconstructed={reconstructed}, checkpoint={model_schema}."
        )
    validate_feature_graph_schema(feature_schema, model_schema, context="Selected feature artifacts")
    return model_schema


def resolve_evaluation_split(cli_args, checkpoint_record, selected_feature_run_id):
    """Resolve checkpoint membership independently of diagnostic label selection."""
    original_id = checkpoint_record.get("feature_run_id")
    feature_metadata = (load_feature_run_metadata(original_id, required=False) or {}) if original_id else {}
    checkpoint_provenance = {**checkpoint_record["args"], **checkpoint_record.get("metadata", {})}
    manifest_id = recorded_manifest_id(checkpoint_record, feature_loader=lambda *args, **kwargs: feature_metadata)
    if not manifest_id:
        if evaluation_selection(cli_args) is None:
            raise ValueError("No recorded default test cohort; select an evaluation population with --season.")
        if selected_feature_run_id != original_id:
            load_feature_run_metadata(selected_feature_run_id, required=True)
        return None, "explicit season selection; historical split unavailable"
    manifest, source = resolve_artifact_split(
        cli_args, feature_metadata, feature_run_id=original_id, checkpoint=checkpoint_provenance,
    )
    if selected_feature_run_id != original_id:
        resolve_artifact_split(
            cli_args, load_feature_run_metadata(selected_feature_run_id, required=True),
            feature_run_id=selected_feature_run_id, checkpoint={"split_manifest_id": manifest["manifest_id"]},
        )
    return manifest, source


def prepare_evaluation_population(args, checkpoint_record, selected_feature_run_id, auxiliary_models):
    """Select a cohort and verify all participants before constructing test datasets."""
    manifest, source = resolve_evaluation_split(args, checkpoint_record, selected_feature_run_id)
    selected = evaluation_selection(args)
    match_ids = np.asarray(selected if selected is not None else manifest["test"])
    record_cohorts = [(checkpoint_record, match_ids)]
    for auxiliary in auxiliary_models.values():
        auxiliary_id = auxiliary.args.get("model_id")
        if auxiliary_id is None:
            auxiliary_id = next((getattr(args, name, None) for name in (
                "xpass_pass_height_model_id", "xpass_pass_intent_model_id")
                if getattr(args, name, None) and getattr(args, name).split("/", 1)[0] == auxiliary.args.get("task")), None)
        if not auxiliary_id:
            raise ValueError("Cannot verify held-out membership of a participating auxiliary checkpoint.")
        record_cohorts.append((utils.get_model_record(auxiliary_id), match_ids))
    membership = collect_evaluation_membership(record_cohorts)
    confirm_direct_evaluation_membership(membership, args)
    return manifest, source, match_ids, membership


def evaluation_split_metadata(manifest):
    selector = split_selector(manifest or {}, default=None)
    return {"train_split_percent": selector["train_split"], "train_count": selector["train_count"],
            "split_manifest_id": (manifest or {}).get("manifest_id")}


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

    cache_dir = Path(args.pc_xpass_cache_dir) if args.pc_xpass_cache_dir else pc_versions.cache_dir("sportec", args)
    metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
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
    if args.evaluate_combined_success and args.xpass_weight == "v5":
        args.v5_intent_threshold = 0.01 if getattr(args, "v5_intent_threshold", None) is None else args.v5_intent_threshold
        args.v5_discount = True if getattr(args, "v5_discount", None) is None else args.v5_discount
        if not math.isfinite(float(args.v5_intent_threshold)) or float(args.v5_intent_threshold) <= 0.0:
            raise ValueError("--v5-intent-threshold must be a positive finite float.")
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
    if args.xpass_weight != "v5" and (
        getattr(args, "v5_intent_threshold", None) is not None or getattr(args, "v5_discount", None) is not None
    ):
        raise ValueError("v5 options are only valid with combined --xpass-weight v5.")

    cache_dir = Path(args.pc_xpass_cache_dir) if args.pc_xpass_cache_dir else pc_versions.cache_dir("sportec", args)
    metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
    metric = physical_xpass_metric_for_version(args.xpass_version, pc_xpass=True)
    available = {str(value) for value in metadata.get("available_metrics", [])}
    if metric not in available:
        raise ValueError(
            f"Requested pc-xPass metric {metric!r} is not available in {cache_dir}; available={sorted(available)}. "
            "Create a fresh cache with scripts/generate_physical_xpass.py --pc-xpass and the required metric selectors."
        )
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
    if getattr(cli_args, "diagnostic_return_type", None) is not None:
        return cli_args.diagnostic_feature_run_id or getattr(model_args, "feature_run_id", None), None

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


def resolve_lane_survival_context(model_args: argparse.Namespace, selection_args: argparse.Namespace | None = None) -> str | None:
    if not bool(getattr(model_args, "lane_survival", False)):
        return None

    selection_args = selection_args or argparse.Namespace(pc_xpass_id=None)
    explicit = getattr(selection_args, "pc_xpass_cache_dir", None)
    cache_dir = Path(explicit) if explicit else pc_versions.cache_dir("sportec", selection_args)
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
            f"cache={actual_fingerprint}, checkpoint={expected_fingerprint}. "
            "Select a compatible cache or regenerate PC-xPass with the checkpoint's physical settings."
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
    outcome_bootstrap_resamples: int = 2000,
    outcome_bootstrap_seed: int = 42,
) -> tuple[Path, pd.DataFrame]:
    """Write factual outcome-model target-fidelity and external-validity artifacts."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    ci, bootstrap_metadata = outcome_bootstrap(
        outcome_evaluation, model_id=model_id, task=task,
        resamples=outcome_bootstrap_resamples, seed=outcome_bootstrap_seed,
    )
    if outcome_bootstrap_resamples:
        ci.to_csv(output_dir / "outcome_bootstrap_ci.csv", index=False)
    else:
        (output_dir / "outcome_bootstrap_ci.csv").unlink(missing_ok=True)

    predictions = np.asarray(outcome_evaluation["prediction"], dtype=float).reshape(-1)
    targets = np.asarray(outcome_evaluation["target"], dtype=float).reshape(-1)
    diagnostics = np.asarray(outcome_evaluation["diagnostic"], dtype=float).reshape(-1)
    diagnostic_return = outcome_evaluation.get("diagnostic_return_type", "next_10")
    diagnostic_target = diagnostic_name(diagnostic_return)
    if not np.isfinite(diagnostics).all() or not np.isin(diagnostics, [0, 1]).all():
        raise ValueError("Common goal diagnostic targets must be finite binary values.")
    branches = np.asarray(outcome_evaluation["execution_branch"], dtype=int).reshape(-1)
    if not (len(predictions) == len(targets) == len(diagnostics) == len(branches)):
        raise ValueError("Outcome evaluation arrays must have identical lengths.")

    metric_rows: list[dict] = []
    bin_frames: list[pd.DataFrame] = []
    targets_by_name = {
        "xt_training_target": targets,
        diagnostic_target: diagnostics,
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
                "positive_count": int(np.sum(stratum_observed > 0)) if evaluation_target == diagnostic_target else np.nan,
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
    metrics.attrs["outcome_bootstrap"] = bootstrap_metadata
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
        title=f"Direct xT-target calibration: {model_id}",
        y_label="Mean xT-derived target",
        output_path=output_dir / "xt_target_calibration.png",
        show_identity=True,
    )
    _save_binned_relationship_plot(
        binned,
        evaluation_target=diagnostic_target,
        title=f"External validity: goal {diagnostic_return} association: {model_id}",
        y_label=f"Mean goal {diagnostic_return} indicator",
        output_path=output_dir / ("goal_next10_association.png" if diagnostic_return == "next_10" else f"goal_{diagnostic_return}_association.png"),
        show_identity=False,
    )
    return output_dir, metrics


def write_pass_success_height_metrics(output_root: str | Path, model_id: str, rows: list[dict], height_context=None) -> Path:
    """Write observed-pass-height pass-success metrics as a portable table."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id", "stratum", "sample_count", "positive_count", "success_prevalence", "roc_auc", "brier"
    ]
    columns.extend((height_context or {}).keys())
    records = [{"model_id": model_id, **row, **(height_context or {})} for row in rows]
    pd.DataFrame(records, columns=columns).to_csv(output_dir / "pass_success_height_metrics.csv", index=False)
    return output_dir / "pass_success_height_metrics.csv"


def write_pass_success_predictor_metrics(output_root: str | Path, model_id: str, rows: list[dict], height_context=None) -> Path:
    """Write comparable learning, physical, and combined pass-success metrics."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id", "predictor", "stratum", "sample_count", "positive_count",
        "success_prevalence", "classification_threshold", "roc_auc", "pr_auc", "brier", "log_loss",
        "calibration_intercept", "calibration_slope", "ece", "precision", "recall", "f1",
        "true_positive", "false_positive", "true_negative", "false_negative",
    ]
    columns.extend((height_context or {}).keys())
    pd.DataFrame([{"model_id": model_id, **row, **(height_context or {})} for row in rows], columns=columns).to_csv(
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
    height_context: dict | None = None,
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
    for row in metric_rows + calibration_rows + curve_rows:
        row.update(height_context or {})
    write_pass_success_predictor_metrics(output_dir, model_id, metric_rows, height_context)
    _write_rows(output_dir, "pass_success_predictor_calibration_bins.csv", model_id, calibration_rows)
    _write_rows(output_dir, "pass_success_predictor_threshold_curve.csv", model_id, curve_rows)

    distance_rows: list[dict] = []
    for name, mask, lower, upper in equal_frequency_slice_masks(diagnostics["pass_distance"], "distance"):
        rows, _, _ = calc_binary_diagnostic_rows(targets, predictors, ((name, mask),), threshold=threshold)
        distance_rows.extend({"distance_lower": lower, "distance_upper": upper, **row, **(height_context or {})} for row in rows)
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
                row.update(height_context or {})
                weight_rows.append(row)
    _write_rows(output_dir, "pass_success_combined_weight_slices.csv", model_id, weight_rows)
    return metric_rows


def write_pass_height_diagnostics(
    output_root: str | Path,
    model_id: str,
    diagnostics: dict,
    *,
    threshold: float,
    height_threshold: float | None = 2.0,
    height_context: dict | None = None,
) -> list[dict]:
    """Write the same binary-probability diagnostic suite for pass-height predictions."""
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(diagnostics["targets"])
    predictors = {"pass_height": np.asarray(diagnostics["predictions"])}
    metric_rows, calibration_rows, curve_rows = calc_binary_diagnostic_rows(targets, predictors, threshold=threshold)
    for row in metric_rows + calibration_rows + curve_rows:
        row.update(height_context or {})
    _write_rows(output_dir, "pass_height_metrics.csv", model_id, metric_rows)
    _write_rows(output_dir, "pass_height_calibration_bins.csv", model_id, calibration_rows)
    _write_rows(output_dir, "pass_height_threshold_curve.csv", model_id, curve_rows)
    heights = np.asarray(diagnostics["observed_pass_max_height"], dtype=float)
    if height_threshold is None:
        return metric_rows
    low, middle, high = height_threshold - 0.5, height_threshold, height_threshold + 0.5
    bands = (
        ("below_cutoff_neighborhood", heights < low),
        ("below_cutoff", (heights >= low) & (heights < middle)),
        ("above_cutoff", (heights >= middle) & (heights < high)),
        ("above_cutoff_neighborhood", heights >= high),
    )
    slice_rows, _, _ = calc_binary_diagnostic_rows(targets, predictors, bands, threshold=threshold)
    for row, (lower, upper) in zip(slice_rows, ((None, low), (low, middle), (middle, high), (high, None))):
        row.update(height_context or {})
        row.update(height_lower_meters=lower, height_upper_meters=upper,
                   lower_inclusive=True, upper_inclusive=False)
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
    split_provenance: dict | None = None,
    evaluation_membership: dict | None = None,
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
        "configuration_id": evaluation_options.get("configuration_id", "unmasked"),
        "inference_masks": _json_compatible(evaluation_options.get("inference_masks", {})),
    }
    if outcome_metrics is not None and "outcome_bootstrap" in outcome_metrics.attrs:
        metadata["outcome_bootstrap"] = outcome_metrics.attrs["outcome_bootstrap"]
    if split_provenance is not None:
        metadata["split_provenance"] = split_provenance
    if evaluation_membership is not None:
        metadata["evaluation_membership"] = evaluation_membership
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    metric_record = {
        "evaluation_timestamp": timestamp,
        "model_id": str(model_id),
        "task": task,
        "height_target_context": _json_compatible({key: value for key, value in evaluation_options.items()
                                                   if "pass_height" in key}),
        "metrics": _json_compatible(test_metrics),
        "configuration_id": evaluation_options.get("configuration_id", "unmasked"),
    }
    if evaluation_membership is not None:
        metric_record["evaluation_membership_status"] = evaluation_membership["status"]
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
                & existing.get("configuration_id", pd.Series("unmasked", index=existing.index))
                    .fillna("unmasked").astype(str).eq(str(evaluation_options.get("configuration_id", "unmasked")))
            )
        ]
    columns = list(dict.fromkeys([*existing.columns, *row.columns]))
    pd.concat([existing.reindex(columns=columns), row.reindex(columns=columns)], ignore_index=True).to_csv(
        summary_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    add_pass_height_argument(parser)
    add_evaluation_arguments(parser)
    add_outcome_bootstrap_arguments(parser)
    parser.add_argument("--model_id", type=str, required=True, help="task/trial, e.g., pass_success/01")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--feature_dir", type=str, required=False, default=None)
    parser.add_argument("--feature-run-id", type=str, required=False, default=None)
    add_split_arguments(parser)
    parser.add_argument("--diagnostic-feature-run-id", type=str, required=False, default=None)
    parser.add_argument("--evaluation-output-dir", type=str, required=False, default=None)
    parser.add_argument("--membership-acceptance-manifest", help=argparse.SUPPRESS)
    parser.add_argument("--learning-curve-rows", action="store_true")
    parser.add_argument("--comparison-rows", action="store_true", help="Export semantic prediction rows and retain diagnostic artifacts.")
    parser.add_argument("--inference-configuration-id", default=None, help="Artifact identity supplied by the evaluation wrapper.")
    add_inference_mask_arguments(parser)
    parser.add_argument("--evaluation-timestamp", type=str, required=False, default=None)
    parser.add_argument("--weighted-pass-success-metrics", action="store_true")
    parser.add_argument("--evaluate-xpass", action="store_true")
    parser.add_argument("--evaluate-combined-success", action="store_true")
    parser.add_argument("--xpass-version", default=None)
    parser.add_argument("--xpass-weight", choices=["v1", "v2", "v3", "v4", "v5"], default=None)
    parser.add_argument("--observed-pass-height-stratification", action="store_true")
    parser.add_argument("--classification-threshold", type=probability_threshold, default=0.5)
    parser.add_argument("--f1-outcome-threshold", type=probability_threshold, default=None)
    parser.add_argument("--pass-height-model-id", type=str, default=None)
    parser.add_argument("--pass-intent-model-id", type=str, default=None)
    parser.add_argument("--pc-xpass-cache-dir", type=str, default=None)
    parser.add_argument("--discount", type=parse_bool_text, default=None)
    parser.add_argument("--v4-power", type=float, default=None)
    parser.add_argument("--v4-zero", type=float, default=None)
    parser.add_argument("--v5-intent-threshold", type=float, default=None)
    parser.add_argument("--v5-discount", type=parse_bool_text, default=None)
    add_top_pass_selector(parser)
    pc_versions.add_selection_argument(parser)
    add_auxiliary_arguments(parser)
    args, unknown_args = parser.parse_known_args()
    if any(token.startswith("--inference-mask-") for token in unknown_args):
        parser.error("Numbered inference masks are supported by scripts/evaluate_relevant_models.py only.")
    if args.comparison_rows:
        args.learning_curve_rows = True
    pc_versions.check_selectors(args)
    resolve_top_pass_selector(parser, args, pc_only=True)
    # This entrypoint tolerates unrelated model flags, but not extra selector counts.
    for index, token in enumerate(sys.argv):
        if token.split("=", 1)[0] in {"--top-pass", "--top_pass"}:
            next_index = index + (1 if "=" in token else 2)
            if next_index < len(sys.argv) and sys.argv[next_index].lstrip("-").isdigit():
                parser.error("--top-pass selects exactly one integer.")

    device = args.device if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)
    training_evaluation_settings = {key: getattr(model_args, key, None) for key in
                                    ("intended_receiver_mode", "min_pass_dur", "return_type")}
    if args.intended_receiver_mode is not None:
        model_args.intended_receiver_mode = args.intended_receiver_mode
    model_args.min_pass_dur = (args.min_pass_duration if args.min_pass_duration is not None
                               else getattr(model_args, "min_pass_dur", 0.0))
    model_args.diagnostic_return_type = args.diagnostic_return_type or "next_10"
    model_args.pass_height_threshold, height_context = evaluation_height(model_args, args.pass_height_threshold)
    height_export = height_export_context(height_context)
    if args.pass_height_threshold is not None:
        print(f"Evaluation height target override: model={height_export['model_pass_height_threshold_meters']} m, "
              f"evaluation={args.pass_height_threshold} m; prediction meaning is unchanged.")
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
    model_args.v5_intent_threshold = float(0.01 if args.v5_intent_threshold is None else args.v5_intent_threshold)
    model_args.v5_discount = True if args.v5_discount is None else bool(args.v5_discount)
    participating_models = {model_args.task: model}
    auxiliary_models = configure_auxiliary_models(args, participating_models, device)
    for auxiliary in auxiliary_models.values():
        auxiliary.args["lane_survival_cache_dir"] = resolve_lane_survival_context(argparse.Namespace(**auxiliary.args), args)

    mask_configurations = resolve_inference_mask_configurations(args)
    inference_masks = mask_configurations[0]["masks"] if mask_configurations else []
    mask_metadata = configure_inference_masks(participating_models, inference_masks)
    configuration_id = args.inference_configuration_id or mask_metadata["configuration_id"]

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
    if args.intended_receiver_mode is not None:
        label_dir = str(get_action_label_dir(model_args.return_type, args.intended_receiver_mode,
            root=feature_root, use_carries=bool(getattr(model_args, "use_carries", False))))
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
    if model_args.pass_height_threshold is None and pass_height_diagnostic_label_dir:
        height_context["evaluation_pass_height_definition"] = definition(
            observed_pass_height_threshold_meters, "diagnostic_feature_run")
    height_export = height_export_context(height_context)
    observed_pass_height_threshold_meters = height_export["evaluation_pass_height_threshold_meters"]
    physical_cache_dir = resolve_physical_xpass_context(
        model_args,
        feature_root,
        prefer_feature_root=bool(
            args.feature_run_id and args.feature_run_id != getattr(model_args, "feature_run_id", None)
        ),
    )
    lane_survival_cache_dir = resolve_lane_survival_context(model_args, args)
    feature_schema = infer_feature_graph_schema(feature_dir)
    model_schema = validate_checkpoint_edge_schema(feature_schema, model_args)

    checkpoint_record = utils.get_model_record(args.model_id)
    evaluation_manifest, split_source, test_match_ids, evaluation_membership = prepare_evaluation_population(
        args, checkpoint_record, resolved_feature_run_id, auxiliary_models)
    exclusion_checks = evaluation_membership["checks"]
    evaluation_manifest = evaluation_manifest or {}
    print(f"Using {split_source} (manifest={evaluation_manifest.get('manifest_id')}): "
          f"{len(test_match_ids)} requested test matches.")

    dataset_args = build_action_dataset_kwargs(
        model_args,
        train=False,
        diagnostic_label_dir=diagnostic_label_dir,
        pass_height_diagnostic_label_dir=pass_height_diagnostic_label_dir,
        physical_cache_dir=physical_cache_dir,
        lane_survival_cache_dir=lane_survival_cache_dir,
        evaluation_filter=args.evaluation_filter,
    )
    if evaluated_task in {"outcome_scoring", "outcome_conceding"} and args.diagnostic_return_type is not None:
        diagnostic_root = resolve_feature_root(args.diagnostic_feature_run_id) if args.diagnostic_feature_run_id else feature_root
        dataset_args["evaluation_diagnostic_label_dir"] = str(get_action_label_dir(
            model_args.diagnostic_return_type, getattr(model_args, "intended_receiver_mode", None) or DEFAULT_INTENDED_RECEIVER_MODE,
            root=diagnostic_root, use_carries=bool(getattr(model_args, "use_carries", False))))
        dataset_args["require_goal_next10_diagnostics"] = False
    if weighted_pc_xpass_cache_dir is not None:
        dataset_args["pass_distance_cache_dir"] = weighted_pc_xpass_cache_dir
    if auxiliary_models:
        dataset_args["auxiliary_models_args"] = {role: dict(auxiliary.args) for role, auxiliary in auxiliary_models.items()}
        for role, auxiliary in auxiliary_models.items():
            validate_feature_graph_schema(feature_schema, utils.validate_model_graph_schemas({role: auxiliary}),
                                          context=f"Selected {role} auxiliary feature artifacts")
    if evaluation_xpass_cache_dir is not None:
        dataset_args["evaluation_xpass_cache_dir"] = evaluation_xpass_cache_dir
        dataset_args["evaluation_xpass_metric"] = evaluation_xpass_metric
        dataset_args["evaluation_xpass_require_nearest"] = bool(
            args.evaluate_combined_success and args.xpass_weight == "v2"
        )
        dataset_args["evaluation_xpass_require_height"] = False
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
    if test_dataset.skipped_matches:
        raise ValueError(f"Required evaluation artifacts could not be loaded: {test_dataset.skipped_matches}")
    if len(test_dataset) == 0:
        raise ValueError(f"No usable test samples remained after evaluation selection "
                         f"(filter={args.evaluation_filter!r}, min_pass_duration={model_args.min_pass_dur}): "
                         f"{test_dataset.selection_counts}.")
    validate_test_dataset_dimensions(test_dataset, model_args)
    if args.comparison_rows and model_args.gnn_task in {"node_selection", "node_binary"}:
        ensure_dataset_node_identities(test_dataset, feature_root, model_args)
    
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
    test_split_provenance = split_dataset_provenance(test_match_ids, test_dataset)
    test_split_provenance["selection_counts"] = test_dataset.selection_counts
    print(f"Evaluating {args.model_id}: {len(test_match_ids)} requested, "
          f"{test_split_provenance['loaded_match_count']} loaded, "
          f"{test_split_provenance['contributing_match_count']} contributing matches with {len(test_dataset)} samples")
    collect_outcome_evaluation = bool(
        args.evaluation_output_dir and (not args.learning_curve_rows or args.comparison_rows)
        and getattr(model_args, "task", None) in {"outcome_scoring", "outcome_conceding"}
    )
    result = run_epoch(
        model_args,
        model,
        test_loader,
        device=device,
        train=False,
        return_outcome_evaluation=collect_outcome_evaluation,
        return_learning_curve_rows=args.learning_curve_rows,
        auxiliary_models=auxiliary_models,
        return_semantic_prediction_rows=args.comparison_rows,
    )
    outcome_metrics = None
    pass_success_height_rows = None
    if args.learning_curve_rows:
        if not args.evaluation_output_dir:
            parser.error("--learning-curve-rows requires --evaluation-output-dir")
        if args.comparison_rows:
            test_metrics, learning_rows, comparison_diagnostics = result
        else:
            test_metrics, learning_rows = result
        if not learning_rows:
            raise ValueError("No learning-curve prediction rows were collected.")
        output_path = Path(args.evaluation_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for row in learning_rows:
            row.update(height_export)
            row.update(diagnostic_return_type=model_args.diagnostic_return_type,
                       evaluation_intended_receiver_mode=getattr(model_args, "intended_receiver_mode", None))
        pd.DataFrame(learning_rows).to_csv(output_path / "learning_curve_predictions.csv", index=False)
        if args.comparison_rows and collect_outcome_evaluation:
            outcome_evaluation = comparison_diagnostics["outcome"]
            if outcome_evaluation is None:
                raise RuntimeError("Comparison rows requested without outcome diagnostics.")
            artifact_dir, outcome_metrics = write_outcome_evaluation_artifacts(
                args.evaluation_output_dir, model_id=args.model_id, task=model_args.task,
                outcome_evaluation=outcome_evaluation, f1_outcome_threshold=args.f1_outcome_threshold,
                outcome_bootstrap_resamples=args.outcome_bootstrap_resamples,
                outcome_bootstrap_seed=args.outcome_bootstrap_seed)
        elif args.comparison_rows and comparison_diagnostics["binary"] is not None:
            binary_evaluation = comparison_diagnostics["binary"]
            if model_args.return_pass_success_height_evaluation:
                pass_success_height_rows = binary_evaluation["height_rows"]
                if args.observed_pass_height_stratification:
                    write_pass_success_height_metrics(args.evaluation_output_dir, args.model_id, pass_success_height_rows, height_export)
                if binary_evaluation.get("predictor_diagnostics"):
                    write_pass_success_predictor_diagnostics(
                        args.evaluation_output_dir, args.model_id, binary_evaluation["predictor_diagnostics"],
                        threshold=args.classification_threshold, height_context=height_export)
            else:
                write_pass_height_diagnostics(
                    args.evaluation_output_dir, args.model_id, binary_evaluation,
                    threshold=args.classification_threshold,
                    height_threshold=height_export["evaluation_pass_height_threshold_meters"], height_context=height_export)
    elif collect_outcome_evaluation:
        test_metrics, outcome_evaluation = result
        if outcome_evaluation is None:
            raise RuntimeError("Outcome evaluation artifacts were requested, but no outcome predictions were collected.")
        artifact_dir, outcome_metrics = write_outcome_evaluation_artifacts(
            args.evaluation_output_dir,
            model_id=args.model_id,
            task=model_args.task,
            outcome_evaluation=outcome_evaluation,
            f1_outcome_threshold=args.f1_outcome_threshold,
            outcome_bootstrap_resamples=args.outcome_bootstrap_resamples,
            outcome_bootstrap_seed=args.outcome_bootstrap_seed,
        )
        print(f"Saved outcome evaluation artifacts to {artifact_dir}")
    elif model_args.return_pass_success_height_evaluation or model_args.return_binary_diagnostics:
        test_metrics, binary_evaluation = result
        if model_args.return_pass_success_height_evaluation:
            pass_success_height_rows = binary_evaluation["height_rows"]
            if args.observed_pass_height_stratification:
                write_pass_success_height_metrics(args.evaluation_output_dir, args.model_id, pass_success_height_rows, height_export)
            if binary_evaluation.get("predictor_diagnostics"):
                write_pass_success_predictor_diagnostics(
                    args.evaluation_output_dir,
                    args.model_id,
                    binary_evaluation["predictor_diagnostics"],
                    threshold=args.classification_threshold,
                    height_context=height_export,
                )
        elif model_args.return_binary_diagnostics:
            write_pass_height_diagnostics(
                args.evaluation_output_dir,
                args.model_id,
                binary_evaluation,
                threshold=args.classification_threshold,
                height_threshold=height_export["evaluation_pass_height_threshold_meters"],
                height_context=height_export,
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
                "filter": args.evaluation_filter,
                "requested_min_pass_duration": args.min_pass_duration,
                "min_pass_duration": model_args.min_pass_dur,
                "selection_counts": test_dataset.selection_counts,
                "configuration_id": configuration_id,
                "inference_masks": mask_metadata,
                **height_context,
                **height_export,
                **evaluation_split_metadata(evaluation_manifest),
                "evaluation_membership_status": evaluation_membership["status"],
                "weighted_pass_success_metrics": bool(args.weighted_pass_success_metrics),
                "evaluate_xpass": bool(args.evaluate_xpass),
                "evaluate_combined_success": bool(args.evaluate_combined_success),
                "xpass_version": args.xpass_version,
                "xpass_metric": evaluation_xpass_metric,
                "xpass_weight": args.xpass_weight,
                "xpass_cache_metadata_sha256": (
                    evaluation_xpass_metadata.get("metadata_sha256") if evaluation_xpass_metadata else None
                ),
                "xpass_pass_height_model_id": args.xpass_pass_height_model_id,
                "xpass_pass_intent_model_id": args.xpass_pass_intent_model_id,
                "xpass_auxiliaries": auxiliary_metadata(participating_models),
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
                "v5_intent_threshold": model_args.v5_intent_threshold if args.evaluate_combined_success and args.xpass_weight == "v5" else None,
                "v5_discount": model_args.v5_discount if args.evaluate_combined_success and args.xpass_weight == "v5" else None,
                "pass_intent_model_id": args.xpass_pass_intent_model_id if args.evaluate_combined_success and args.xpass_weight == "v5" else None,
                "pc_xpass_id": getattr(args, "pc_xpass_id", None),
                "pc_xpass_cache_dir": args.pc_xpass_cache_dir,
            },
            test_metrics=test_metrics,
            outcome_metrics=outcome_metrics,
            evaluation_membership=evaluation_membership,
            split_provenance={"manifest_id": evaluation_manifest.get("manifest_id"),
                              "universe_fingerprint": (evaluation_manifest.get("metadata") or {}).get("universe_fingerprint"),
                              "resolution_source": split_source, "test": test_split_provenance},
            pass_height_diagnostic_feature_run_id=pass_height_diagnostic_feature_run_id,
            pass_height_diagnostic_label_dir=pass_height_diagnostic_label_dir,
            observed_pass_height_threshold_meters=observed_pass_height_threshold_meters,
            pass_height_diagnostic_ancestry_validated=bool(pass_height_diagnostic_lineage),
        )
    if args.evaluation_output_dir:
        metadata_path = Path(args.evaluation_output_dir) / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["evaluation_context"] = {"training_settings": training_evaluation_settings,
            "filter": args.evaluation_filter,
            "requested_min_pass_duration": args.min_pass_duration,
            "selection_counts": test_dataset.selection_counts,
            "intended_receiver_mode": getattr(model_args, "intended_receiver_mode", None),
            "diagnostic_target_family": "goal", "diagnostic_return_type": model_args.diagnostic_return_type,
            "min_pass_duration": getattr(model_args, "min_pass_dur", 0),
            "requested_match_ids": test_match_ids.tolist(), "seasons": args.season,
            "season_half": args.season_half, "exclusion_checks": exclusion_checks,
            "artifact_coverage": test_split_provenance}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Test:\t" + get_losses_str(test_metrics))


if __name__ == "__main__":
    try:
        main()
    except EvaluationMembershipCancelled as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
