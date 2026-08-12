import argparse
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
from models.utils import get_losses_str, infer_feature_graph_schema, infer_training_edge_schema, load_splits, run_epoch
from models.utils import validate_feature_graph_schema
from physical_pass_model import (
    PC_XPASS_SOURCE,
    PHYSICAL_XPASS_SOURCE,
    model_uses_physical_xpass,
    normalize_physical_xpass_speed_aggregation,
    pc_xpass_lane_survival_metadata_fingerprint,
    validate_physical_xpass_args,
    validate_physical_xpass_cache_metadata,
    validate_pc_xpass_lane_survival_mode_cache_metadata,
)
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    get_action_label_dir,
    get_pc_xpass_dir,
    get_physical_xpass_dir,
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


def resolve_weighted_pass_success_cache(args: argparse.Namespace, model_args: argparse.Namespace) -> str | None:
    """Validate and return the pc-xPass cache for weighted pass-success evaluation."""
    if not bool(args.weighted_pass_success_metrics):
        return None
    if str(getattr(model_args, "task", "")) != "pass_success":
        raise ValueError("--weighted-pass-success-metrics requires a pass_success checkpoint.")
    if not args.pass_height_model_id:
        raise ValueError("--weighted-pass-success-metrics requires --pass-height-model-id.")
    if not math.isfinite(float(args.v4_power)) or float(args.v4_power) <= 0.0:
        raise ValueError("--v4-power must be a positive finite float.")
    if not math.isfinite(float(args.v4_zero)) or float(args.v4_zero) <= 0.0:
        raise ValueError("--v4-zero must be a positive finite float.")

    cache_dir = Path(args.pc_xpass_cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_dir, expected_source=PC_XPASS_SOURCE)
    cached_model_id = metadata.get("pass_height_model_id")
    if str(cached_model_id) != str(args.pass_height_model_id):
        raise ValueError(
            "pc-xPass cache pass_height_model_id does not match --pass-height-model-id: "
            f"cache={cached_model_id!r}, requested={args.pass_height_model_id!r}."
        )
    return str(cache_dir)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="task/trial, e.g., pass_success/01")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--feature_dir", type=str, required=False, default=None)
    parser.add_argument("--feature-run-id", type=str, required=False, default=None)
    parser.add_argument("--diagnostic-feature-run-id", type=str, required=False, default=None)
    parser.add_argument("--weighted-pass-success-metrics", action="store_true")
    parser.add_argument("--pass-height-model-id", type=str, default=None)
    parser.add_argument("--pc-xpass-cache-dir", type=str, default=str(get_pc_xpass_dir("sportec")))
    parser.add_argument("--discount", type=parse_bool_text, default=True)
    parser.add_argument("--v4-power", type=float, default=4.0)
    parser.add_argument("--v4-zero", type=float, default=0.7)
    args, _ = parser.parse_known_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)
    weighted_pc_xpass_cache_dir = resolve_weighted_pass_success_cache(args, model_args)
    model_args.weighted_pass_success_metrics = bool(args.weighted_pass_success_metrics)
    model_args.discount = bool(args.discount)
    model_args.v4_power = float(args.v4_power)
    model_args.v4_zero = float(args.v4_zero)

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

    _, _, test_match_ids = load_splits(feature_dir=feature_dir)

    dataset_args = build_action_dataset_kwargs(
        model_args,
        train=False,
        diagnostic_label_dir=diagnostic_label_dir,
        physical_cache_dir=physical_cache_dir,
        lane_survival_cache_dir=lane_survival_cache_dir,
    )
    if weighted_pc_xpass_cache_dir is not None:
        # These attributes are evaluation sidecars; they do not modify node or edge features.
        dataset_args["pass_height_cache_dir"] = weighted_pc_xpass_cache_dir
        dataset_args["require_observed_pass_height"] = True
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
    test_metrics = run_epoch(model_args, model, test_loader, device=device, train=False)
    print("Test:\t" + get_losses_str(test_metrics))
