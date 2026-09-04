import argparse
import json
import math
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from datatools import config
from datatools.config import FIELD_SIZE, LABEL_INDEX
from models.gnn import GNN
from physical_pass_model import (
    EVALUATION_XPASS_DISTANCE_ATTR,
    EVALUATION_XPASS_NEAREST_OPPONENT_DISTANCE_ATTR,
    EVALUATION_XPASS_PASS_HEIGHT_ATTR,
    EVALUATION_XPASS_PROB_ATTR,
    PHYSICAL_XPASS_DISTANCE_ATTR,
    PHYSICAL_XPASS_NEAREST_OPPONENT_DISTANCE_ATTR,
    PHYSICAL_XPASS_PASS_HEIGHT_ATTR,
    PHYSICAL_XPASS_PROB_ATTR,
    blend_physical_xpass_predictions,
    physical_xpass_blend_weight_v4,
    physical_xpass_blend_weight,
    physical_xpass_blend_weight_v2,
    physical_xpass_blend_weight_v3,
    normalize_pc_xpass_lane_survival_mode,
    residual_distance_threshold,
    resolved_residual_regularization_lambdas,
)
from project_config import (
    FEATURE_RUNS_DIR,
    SAVED_DIR,
    get_action_graph_dir,
    get_model_bundle_root,
    get_feature_run_root,
    get_resolved_action_dir,
    get_task_saved_dir,
    infer_legacy_model_context,
    infer_target_family,
    load_feature_run_metadata,
    load_model_bundle_metadata,
    load_model_splits,
    resolve_feature_run_id,
)


FEATURE_SIGNATURE_KEYS = (
    "xy_only",
    "possessor_aware",
    "keeper_aware",
    "ball_z_aware",
    "poss_vel_aware",
    "poss_rel_vel_aware",
    "poss_geometry_aware",
    "goal_features_aware",
    "goal_nodes_aware",
    "accel_aware",
    "offside_aware",
    "extend_features",
    "lane_survival",
    "lane_survival_mode",
    "filter_blockers",
    "sparsify",
    "max_edge_dist",
    "v_edge_feature_mode",
    "add_v_edge_features",
    "relative_speed_edge_feature_mode",
    "add_relative_speed_edge_features",
    "node_in_dim",
    "edge_in_dim",
)

V_EDGE_FEATURE_MODE_ALL = "all"
V_EDGE_FEATURE_MODE_NONE = "none"
V_EDGE_FEATURE_MODE_NO_POSS = "no_poss"
V_EDGE_FEATURE_MODES = (
    V_EDGE_FEATURE_MODE_ALL,
    V_EDGE_FEATURE_MODE_NONE,
    V_EDGE_FEATURE_MODE_NO_POSS,
)
RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL = "all"
RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE = "none"
RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS = "no_poss"
RELATIVE_SPEED_EDGE_FEATURE_MODES = (
    RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL,
    RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE,
    RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS,
)
RUNTIME_INTENDED_RECEIVER_MODE_PREFERENCE = ("model", "original", "angle_only")
DEFAULT_RUNTIME_RETURN_TYPE = "disc_0.9"


def normalize_v_edge_feature_mode(
    v_edge_feature_mode: str | None = None,
    *,
    use_v_edge_features: bool | None = None,
    mask_possessor_v_edge_features: bool | None = None,
    add_v_edge_features: bool | None = None,
    edge_in_dim: int | None = None,
) -> str:
    if v_edge_feature_mode is not None:
        mode = str(v_edge_feature_mode).strip().replace("-", "_")
        if mode in V_EDGE_FEATURE_MODES:
            return mode
        raise ValueError(
            f"Invalid v_edge_feature_mode={v_edge_feature_mode!r}. "
            f"Expected one of: {', '.join(V_EDGE_FEATURE_MODES)}."
        )
    if bool(mask_possessor_v_edge_features):
        return V_EDGE_FEATURE_MODE_NO_POSS
    if use_v_edge_features is not None:
        return V_EDGE_FEATURE_MODE_ALL if bool(use_v_edge_features) else V_EDGE_FEATURE_MODE_NONE
    if add_v_edge_features is not None:
        return V_EDGE_FEATURE_MODE_ALL if bool(add_v_edge_features) else V_EDGE_FEATURE_MODE_NONE
    if edge_in_dim is not None:
        return V_EDGE_FEATURE_MODE_ALL if int(edge_in_dim) > 2 else V_EDGE_FEATURE_MODE_NONE
    return V_EDGE_FEATURE_MODE_NONE


def normalize_relative_speed_edge_feature_mode(
    relative_speed_edge_feature_mode: str | None = None,
    *,
    use_relative_speed_edge_features: bool | None = None,
    mask_possessor_relative_speed_edge_features: bool | None = None,
    add_relative_speed_edge_features: bool | None = None,
    edge_in_dim: int | None = None,
) -> str:
    if relative_speed_edge_feature_mode is not None:
        mode = str(relative_speed_edge_feature_mode).strip().replace("-", "_")
        if mode in RELATIVE_SPEED_EDGE_FEATURE_MODES:
            return mode
        raise ValueError(
            f"Invalid relative_speed_edge_feature_mode={relative_speed_edge_feature_mode!r}. "
            f"Expected one of: {', '.join(RELATIVE_SPEED_EDGE_FEATURE_MODES)}."
        )
    if bool(mask_possessor_relative_speed_edge_features):
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS
    if use_relative_speed_edge_features is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if bool(use_relative_speed_edge_features) else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    if add_relative_speed_edge_features is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if bool(add_relative_speed_edge_features) else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    if edge_in_dim is not None:
        return RELATIVE_SPEED_EDGE_FEATURE_MODE_ALL if int(edge_in_dim) > 4 else RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE
    return RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE


def use_v_edge_features_for_mode(v_edge_feature_mode: str | None) -> bool:
    return normalize_v_edge_feature_mode(v_edge_feature_mode) != V_EDGE_FEATURE_MODE_NONE


def mask_possessor_v_edge_features_for_mode(v_edge_feature_mode: str | None) -> bool:
    return normalize_v_edge_feature_mode(v_edge_feature_mode) == V_EDGE_FEATURE_MODE_NO_POSS


def use_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode: str | None) -> bool:
    return normalize_relative_speed_edge_feature_mode(relative_speed_edge_feature_mode) != RELATIVE_SPEED_EDGE_FEATURE_MODE_NONE


def mask_possessor_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode: str | None) -> bool:
    return normalize_relative_speed_edge_feature_mode(relative_speed_edge_feature_mode) == RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS


def validate_relative_speed_edge_feature_mode(
    v_edge_feature_mode: str | None,
    relative_speed_edge_feature_mode: str | None,
) -> None:
    if use_relative_speed_edge_features_for_mode(relative_speed_edge_feature_mode) and not use_v_edge_features_for_mode(v_edge_feature_mode):
        raise ValueError("Relative-speed edge features require velocity-angle edge features.")


def normalize_v_edge_feature_args(args: dict[str, Any]) -> dict[str, Any]:
    mode = normalize_v_edge_feature_mode(
        args.get("v_edge_feature_mode"),
        use_v_edge_features=args.get("use_v_edge_features"),
        mask_possessor_v_edge_features=args.get("mask_possessor_v_edge_features"),
        add_v_edge_features=args.get("add_v_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    relative_speed_mode = normalize_relative_speed_edge_feature_mode(
        args.get("relative_speed_edge_feature_mode"),
        use_relative_speed_edge_features=args.get("use_relative_speed_edge_features"),
        mask_possessor_relative_speed_edge_features=args.get("mask_possessor_relative_speed_edge_features"),
        add_relative_speed_edge_features=args.get("add_relative_speed_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    validate_relative_speed_edge_feature_mode(mode, relative_speed_mode)
    args["v_edge_feature_mode"] = mode
    args["use_v_edge_features"] = use_v_edge_features_for_mode(mode)
    args["mask_possessor_v_edge_features"] = mask_possessor_v_edge_features_for_mode(mode)
    args["relative_speed_edge_feature_mode"] = relative_speed_mode
    args["use_relative_speed_edge_features"] = use_relative_speed_edge_features_for_mode(relative_speed_mode)
    args["mask_possessor_relative_speed_edge_features"] = mask_possessor_relative_speed_edge_features_for_mode(relative_speed_mode)
    return args


def is_validation_loss_improved(current_loss: float, best_loss: float, min_delta: float) -> bool:
    if best_loss <= 0:
        return True
    return float(current_loss) < float(best_loss) - float(min_delta)


def should_stop_early(
    enabled: bool,
    epoch: int,
    min_epochs: int,
    epochs_since_loss_improvement: int,
    patience: int,
) -> bool:
    return bool(enabled) and int(epoch) >= int(min_epochs) and int(epochs_since_loss_improvement) >= int(patience)


def extract_model_feature_signature(args: dict[str, Any]) -> dict[str, Any]:
    v_edge_feature_mode = normalize_v_edge_feature_mode(
        args.get("v_edge_feature_mode"),
        use_v_edge_features=args.get("use_v_edge_features"),
        mask_possessor_v_edge_features=args.get("mask_possessor_v_edge_features"),
        add_v_edge_features=args.get("add_v_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    relative_speed_edge_feature_mode = normalize_relative_speed_edge_feature_mode(
        args.get("relative_speed_edge_feature_mode"),
        use_relative_speed_edge_features=args.get("use_relative_speed_edge_features"),
        mask_possessor_relative_speed_edge_features=args.get("mask_possessor_relative_speed_edge_features"),
        add_relative_speed_edge_features=args.get("add_relative_speed_edge_features"),
        edge_in_dim=args.get("edge_in_dim"),
    )
    validate_relative_speed_edge_feature_mode(v_edge_feature_mode, relative_speed_edge_feature_mode)
    signature = {
        "xy_only": bool(args.get("xy_only", False)),
        "possessor_aware": bool(args.get("possessor_aware", False)),
        "keeper_aware": bool(args.get("keeper_aware", False)),
        "ball_z_aware": bool(args.get("ball_z_aware", False)),
        "poss_vel_aware": bool(args.get("poss_vel_aware", False)),
        "poss_rel_vel_aware": bool(args.get("poss_rel_vel_aware", False)),
        "poss_geometry_aware": bool(args.get("poss_geometry_aware", True)),
        "goal_features_aware": bool(args.get("goal_features_aware", True)),
        "goal_nodes_aware": bool(args.get("goal_nodes_aware", True)),
        "accel_aware": True if args.get("accel_aware") is None else bool(args.get("accel_aware")),
        "offside_aware": True if args.get("offside_aware") is None else bool(args.get("offside_aware")),
        "extend_features": bool(args.get("extend_features", False)),
        "lane_survival": bool(args.get("lane_survival", False)),
        "lane_survival_mode": (
            normalize_pc_xpass_lane_survival_mode(args.get("lane_survival_mode"))
            if bool(args.get("lane_survival", False))
            else None
        ),
        "filter_blockers": bool(args.get("filter_blockers", False)),
        "sparsify": args.get("sparsify", "none"),
        "max_edge_dist": args.get("max_edge_dist", 10),
        "v_edge_feature_mode": v_edge_feature_mode,
        "relative_speed_edge_feature_mode": relative_speed_edge_feature_mode,
        "node_in_dim": int(args.get("node_in_dim", 0)),
        "edge_in_dim": int(args.get("edge_in_dim", 2)),
    }
    signature["add_v_edge_features"] = use_v_edge_features_for_mode(v_edge_feature_mode)
    signature["add_relative_speed_edge_features"] = use_relative_speed_edge_features_for_mode(
        relative_speed_edge_feature_mode
    )
    return signature


def parse_model_id(model_id: str) -> tuple[str, str]:
    try:
        task, run_id = str(model_id).split("/", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid model id format: {model_id!r}. Expected task/run_id.") from exc
    if not task or not run_id:
        raise ValueError(f"Invalid model id format: {model_id!r}. Expected task/run_id.")
    return task, run_id


def get_model_path(model_id: str) -> Path:
    task, run_id = parse_model_id(model_id)
    return get_task_saved_dir(task) / run_id


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_model_args_from_metadata(args: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    physical_metadata = (metadata or {}).get("physical_xpass")
    if isinstance(physical_metadata, dict):
        source = physical_metadata.get("source")
        teammate_policy = physical_metadata.get("teammate_policy")
        speed_aggregation = physical_metadata.get("speed_aggregation")
        floor = physical_metadata.get("physical_xpass_floor")
        if source:
            args["physical_xpass_source"] = str(source)
        if teammate_policy:
            args["physical_xpass_teammate_policy"] = str(teammate_policy)
        if speed_aggregation:
            args["physical_xpass_speed_aggregation"] = str(speed_aggregation)
        if floor is not None:
            args["physical_xpass_floor"] = float(floor)
    lane_metadata = (metadata or {}).get("lane_survival")
    if isinstance(lane_metadata, dict):
        mode = lane_metadata.get("mode")
        if mode and not args.get("lane_survival_mode"):
            args["lane_survival_mode"] = normalize_pc_xpass_lane_survival_mode(str(mode))
        fingerprint = lane_metadata.get("cache_fingerprint")
        if fingerprint and not args.get("lane_survival_cache_fingerprint"):
            args["lane_survival_cache_fingerprint"] = str(fingerprint)
    return args


def _iso_or_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def get_model_record(model_id: str) -> dict[str, Any]:
    model_path = get_model_path(model_id)
    args_path = model_path / "args.json"
    metadata_path = model_path / "metadata.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Model args.json not found at {args_path}.")

    args = _read_json_if_exists(args_path) or {}
    metadata = _read_json_if_exists(metadata_path) or {}
    task, run_id = parse_model_id(model_id)

    args.setdefault("edge_in_dim", 2)
    args.setdefault("add_v_edge_features", bool(args["edge_in_dim"] > 2))
    args.setdefault("add_relative_speed_edge_features", bool(args["edge_in_dim"] > 4))
    args.setdefault("accel_aware", True)
    args.setdefault("feature_run_id", None)
    enrich_model_args_from_metadata(args, metadata)
    normalize_v_edge_feature_args(args)

    legacy_context = infer_legacy_model_context(model_id)
    target_family = metadata.get("target_family")
    if target_family is None and task.startswith("outcome_"):
        target_family = infer_target_family(
            bool(args.get("use_xg", False)),
            bool(args.get("use_xt", False)),
            bool(args.get("use_goal_distance", False)),
            bool(args.get("use_epv", False)),
        )
    if target_family is None and legacy_context is not None:
        target_family = legacy_context.get("target_family")

    intended_receiver_mode = metadata.get("intended_receiver_mode")
    if intended_receiver_mode is None and legacy_context is not None:
        intended_receiver_mode = legacy_context.get("intended_receiver_mode")
    intended_receiver_mode = intended_receiver_mode or "unknown"
    return_type = metadata.get("return_type", args.get("return_type"))

    created_at = metadata.get("created_at") or _iso_or_mtime(metadata_path if metadata_path.exists() else args_path)
    feature_signature = extract_model_feature_signature(args)
    weights_path = model_path / "best_weights.pt"
    best_model_path = model_path / "best_model.json"
    has_weights = weights_path.exists() or best_model_path.exists()
    status = str(metadata.get("status") or ("completed" if has_weights else "unknown"))
    is_complete = bool(has_weights and status == "completed")

    return {
        "model_id": model_id,
        "task": str(metadata.get("task") or args.get("task") or task),
        "run_id": str(metadata.get("run_id") or run_id),
        "model_path": str(model_path),
        "created_at": created_at,
        "timestamp": created_at,
        "feature_run_id": metadata.get("feature_run_id", args.get("feature_run_id")),
        "train_split_percent": metadata.get("train_split_percent", args.get("train_split", 50)),
        "split_manifest_id": metadata.get("split_manifest_id", args.get("split_manifest_id")),
        "intended_receiver_mode": intended_receiver_mode,
        "target_family": target_family,
        "return_type": return_type,
        "model_name": str(args.get("model", metadata.get("model", "unknown"))),
        "feature_signature": feature_signature,
        "graph_schema": {
            "node_in_dim": feature_signature["node_in_dim"],
            "edge_in_dim": feature_signature["edge_in_dim"],
            "add_v_edge_features": feature_signature["add_v_edge_features"],
            "add_relative_speed_edge_features": feature_signature["add_relative_speed_edge_features"],
        },
        "status": status,
        "has_weights": has_weights,
        "is_complete": is_complete,
        "args": args,
        "metadata": metadata,
        "legacy": bool(metadata.get("legacy", False) or legacy_context is not None),
    }


def get_model_provenance(model_id: str) -> dict[str, Any]:
    record = get_model_record(model_id)
    return {
        key: record[key]
        for key in [
            "model_id",
            "task",
            "run_id",
            "model_path",
            "created_at",
            "feature_run_id",
            "intended_receiver_mode",
            "target_family",
            "return_type",
            "model_name",
            "feature_signature",
            "graph_schema",
            "status",
            "has_weights",
            "is_complete",
            "legacy",
        ]
    }


def iter_model_records(task: str | None = None) -> list[dict[str, Any]]:
    tasks = [str(task)] if task else [path.name for path in SAVED_DIR.iterdir() if path.is_dir() and path.name != "bundles"]
    records: list[dict[str, Any]] = []
    for task_name in tasks:
        task_dir = get_task_saved_dir(task_name)
        if not task_dir.exists():
            continue
        for model_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            model_id = f"{task_name}/{model_dir.name}"
            args_path = model_dir / "args.json"
            if not args_path.exists():
                continue
            try:
                records.append(get_model_record(model_id))
            except Exception:
                continue
    return records


def resolve_latest_model_id(
    task: str,
    intended_receiver_mode: str | None = None,
    target_family: str | None = None,
) -> str:
    candidates = [record for record in iter_model_records(task) if record["task"] == task]
    candidates = [record for record in candidates if record["is_complete"]]
    if intended_receiver_mode is not None:
        candidates = [record for record in candidates if record["intended_receiver_mode"] == intended_receiver_mode]
    if target_family is not None:
        candidates = [record for record in candidates if record["target_family"] == target_family]

    if not candidates:
        raise FileNotFoundError(
            "No compatible checkpoints were found for "
            f"task={task!r}, intended_receiver_mode={intended_receiver_mode!r}, target_family={target_family!r}."
        )

    feature_signatures = {
        json.dumps(
            {
                "model_name": record["model_name"],
                "feature_signature": record["feature_signature"],
            },
            sort_keys=True,
        )
        for record in candidates
    }
    candidates.sort(key=lambda record: (record["created_at"], record["model_id"]))
    if len(feature_signatures) > 1:
        candidate_ids = ", ".join(record["model_id"] for record in candidates)
        raise ValueError(
            "Multiple compatible checkpoints were found with different feature signatures. "
            f"Pass an explicit model id instead. Candidates: {candidate_ids}."
        )
    return candidates[-1]["model_id"]


def num_trainable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_arg_keys: List[str], args_dict: dict, parser: argparse.ArgumentParser):
    if parser is None:
        return args_dict

    for key in model_arg_keys:
        if key.startswith("n_") or key.endswith("_dim"):
            parser.add_argument("--" + key, type=int, required=True)
        elif key == "dropout":
            parser.add_argument("--" + key, type=float, default=0)
        else:
            parser.add_argument("--" + key, action="store_true", default=False)
    model_args, _ = parser.parse_known_args()

    for key in model_arg_keys:
        args_dict[key] = getattr(model_args, key)

    return args_dict


def get_args_str(keys, args_dict: dict) -> str:
    ret = ""
    for key in keys:
        if key in args_dict:
            ret += " {} {} |".format(key, args_dict[key])
    return ret[1:-2]


def get_losses_str(losses: dict) -> str:
    ret = ""
    for key, value in losses.items():
        if key == "count":
            continue
        ret += " {}: {:.4f} |".format(key, np.mean(value))
    # if len(losses) > 1:
    #     ret += " total_loss: {:.4f} |".format(sum(losses.values()))
    return ret[:-2]


def printlog(line: str, trial_path: str) -> None:
    print(line)
    with open(trial_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def l1_regularizer(model, lambda_l1=0.1):
    l1_loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            l1_loss += lambda_l1 * model_param_value.abs().sum()
    return l1_loss


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def encode_onehot(labels, classes=None):
    if classes:
        classes = [x for x in range(classes)]
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_splits(
    lineup_path="data/lineup/line_up.parquet",
    feature_dir: str = "data/features/action_graphs",
    train_split: int | None = None,
    validation_mode: str = "holdout_80_20",
    validation_fold: int | None = None,
    final_refit: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del lineup_path
    return load_model_splits(
        feature_dir,
        train_split=train_split,
        validation_mode=validation_mode,
        validation_fold=validation_fold,
        final_refit=final_refit,
    )


def load_model(model_id="pass_intent/01", device="cuda") -> GNN:
    if model_id is None:
        return None

    else:
        model_path = get_model_path(model_id)
        with open(model_path / "args.json", "r", encoding="utf-8") as f:
            args = json.load(f)
        metadata = _read_json_if_exists(model_path / "metadata.json") or {}
        args.setdefault("edge_in_dim", 2)
        args.setdefault("add_v_edge_features", bool(args["edge_in_dim"] > 2))
        args.setdefault("add_relative_speed_edge_features", bool(args["edge_in_dim"] > 4))
        args.setdefault("accel_aware", True)
        args.setdefault("feature_run_id", None)
        args.setdefault("model_id", str(model_id))
        enrich_model_args_from_metadata(args, metadata)
        normalize_v_edge_feature_args(args)

        if args["model"] in ["gcn", "gin", "gat"]:  # GNN models
            model = GNN(args).to(device)
            weights_path = model_path / "best_weights.pt"
            state_dict = torch.load(weights_path, weights_only=False, map_location=lambda storage, _: storage)

            # Backward compatibility for older checkpoints saved with encoder.gat_layers.*
            if any(key.startswith("encoder.gat_layers") for key in state_dict.keys()):
                remapped_state = OrderedDict()
                for key, value in state_dict.items():
                    if key.startswith("encoder.gat_layers"):
                        new_key = key.replace("encoder.gat_layers", "encoder.gnn_layers", 1)
                    else:
                        new_key = key
                    remapped_state[new_key] = value
                state_dict = remapped_state
            model.load_state_dict(state_dict)

        elif args["model"] in ["xgboost", "catboost"]:  # Gradient boosting models
            with open(model_path / "best_params.json", "r", encoding="utf-8") as f:
                params = json.load(f)
            model = XGBClassifier(**params) if args["model"] == "xgboost" else CatBoostClassifier(**params)
            model.load_model(str(model_path / "best_model.json"))

        return model


def resolve_relevant_model_ids(
    intended_receiver_mode: str,
    use_xg: bool = False,
    use_xt: bool = False,
    use_goal_distance: bool = False,
    use_epv: bool = False,
    explicit_model_ids: dict[str, str | None] | None = None,
    include_pass_intent: bool = False,
    include_success_intent: bool = False,
) -> dict[str, str]:
    explicit_model_ids = explicit_model_ids or {}
    target_family = infer_target_family(
        use_xg=bool(use_xg),
        use_xt=bool(use_xt),
        use_goal_distance=bool(use_goal_distance),
        use_epv=bool(use_epv),
    )

    resolved = {}
    tasks = ["action_intent", "pass_success", "outcome_scoring", "outcome_conceding"]
    if include_pass_intent:
        tasks.insert(1, "pass_intent")
    if include_success_intent:
        tasks.insert(3 if include_pass_intent else 2, "success_intent")

    for task in tasks:
        explicit_model_id = explicit_model_ids.get(task)
        if explicit_model_id:
            resolved[task] = str(explicit_model_id)
            continue

        resolved[task] = resolve_latest_model_id(
            task,
            intended_receiver_mode=intended_receiver_mode if task != "success_intent" else None,
            target_family=target_family if task.startswith("outcome_") else None,
        )

    return resolved


def load_bundle_record(bundle_id: str) -> dict[str, Any]:
    metadata = load_model_bundle_metadata(bundle_id, required=True)
    if metadata is None:
        raise FileNotFoundError(f"Model bundle metadata not found for {bundle_id}.")
    metadata.setdefault("bundle_id", str(bundle_id))
    metadata.setdefault("model_ids", {})
    metadata.setdefault("bundle_root", str(get_model_bundle_root(bundle_id)))
    return metadata


def resolve_bundle_model_ids(
    bundle_id: str,
    required_tasks: list[str],
    explicit_model_ids: dict[str, str | None] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    bundle = load_bundle_record(bundle_id)
    explicit_model_ids = explicit_model_ids or {}
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for task in required_tasks:
        if explicit_model_ids.get(task):
            resolved[task] = str(explicit_model_ids[task])
            continue
        model_id = bundle.get("model_ids", {}).get(task)
        if model_id:
            resolved[task] = str(model_id)
            continue
        missing.append(task)

    if missing:
        raise ValueError(
            f"Bundle {bundle_id!r} does not contain required model ids for: {', '.join(missing)}. "
            "Pass them explicitly."
        )

    return resolved, bundle


def require_explicit_model_ids(required_tasks: list[str], explicit_model_ids: dict[str, str | None]) -> dict[str, str]:
    missing = [task for task in required_tasks if not explicit_model_ids.get(task)]
    if missing:
        raise ValueError(
            "Missing explicit model ids for: "
            f"{', '.join(missing)}."
        )
    return {task: str(explicit_model_ids[task]) for task in required_tasks}


def is_feature_graph_schema_compatible(
    feature_schema: dict[str, int | bool],
    required_schema: dict[str, int | bool],
) -> bool:
    feature_node_dim = feature_schema.get("node_in_dim")
    required_node_dim = required_schema.get("node_in_dim")
    if required_node_dim is not None:
        if feature_node_dim is None:
            return False
        if int(feature_node_dim) < int(required_node_dim):
            return False
    feature_edge_dim = int(feature_schema.get("edge_in_dim", 0))
    required_edge_dim = int(required_schema.get("edge_in_dim", 0))
    return feature_edge_dim >= required_edge_dim


def validate_feature_graph_schema(
    feature_schema: dict[str, int | bool],
    required_schema: dict[str, int | bool],
    context: str = "Selected feature artifacts",
) -> None:
    if is_feature_graph_schema_compatible(feature_schema, required_schema):
        return
    raise ValueError(
        f"{context} are incompatible with the loaded model checkpoints: "
        f"features={feature_schema}, models={required_schema}."
    )


def infer_training_edge_schema(
    feature_schema: dict[str, int | bool],
    use_v_edge_features: bool | None = None,
    v_edge_feature_mode: str | None = None,
    use_relative_speed_edge_features: bool | None = None,
    relative_speed_edge_feature_mode: str | None = None,
) -> dict[str, int | bool]:
    mode = normalize_v_edge_feature_mode(v_edge_feature_mode, use_v_edge_features=use_v_edge_features)
    relative_speed_mode = normalize_relative_speed_edge_feature_mode(
        relative_speed_edge_feature_mode,
        use_relative_speed_edge_features=use_relative_speed_edge_features,
    )
    validate_relative_speed_edge_feature_mode(mode, relative_speed_mode)
    feature_edge_dim = int(feature_schema.get("edge_in_dim", 0))
    required_edge_dim = 5 if use_relative_speed_edge_features_for_mode(relative_speed_mode) else 4 if use_v_edge_features_for_mode(mode) else 2
    if feature_edge_dim < required_edge_dim:
        raise ValueError(
            "Selected feature artifacts do not provide the requested edge-feature schema: "
            f"features={feature_schema}, required_edge_dim={required_edge_dim}."
        )
    return {
        "edge_in_dim": required_edge_dim,
        "add_v_edge_features": use_v_edge_features_for_mode(mode),
        "add_relative_speed_edge_features": use_relative_speed_edge_features_for_mode(relative_speed_mode),
    }


def get_model_records(model_ids: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {task: get_model_record(model_id) for task, model_id in model_ids.items()}


def aggregate_graph_schemas(schemas: dict[str, dict[str, Any]]) -> dict[str, int | bool]:
    if not schemas:
        return {"edge_in_dim": 2, "add_v_edge_features": False, "add_relative_speed_edge_features": False}

    edge_dims = [int(schema.get("edge_in_dim", 2)) for schema in schemas.values()]
    node_dims = [int(schema["node_in_dim"]) for schema in schemas.values() if schema.get("node_in_dim") is not None]
    edge_in_dim = max(edge_dims) if edge_dims else 2
    result: dict[str, int | bool] = {
        "edge_in_dim": edge_in_dim,
        "add_v_edge_features": bool(edge_in_dim > 2 or any(schema.get("add_v_edge_features") for schema in schemas.values())),
        "add_relative_speed_edge_features": bool(
            edge_in_dim > 4 or any(schema.get("add_relative_speed_edge_features") for schema in schemas.values())
        ),
    }
    if node_dims:
        result["node_in_dim"] = max(node_dims)
    return result


def validate_model_record_consistency(
    model_records: dict[str, dict[str, Any]],
    require_feature_run_id: bool = True,
    require_intended_receiver_mode: bool = True,
    require_return_type: bool = True,
    require_target_family: bool = True,
    outcome_tasks: tuple[str, ...] = ("outcome_scoring", "outcome_conceding"),
) -> dict[str, Any]:
    if not model_records:
        raise ValueError("No model records were provided for consistency validation.")

    shared: dict[str, Any] = {}

    feature_run_ids = {record.get("feature_run_id") for record in model_records.values()}
    feature_run_ids.discard(None)
    if require_feature_run_id and len(feature_run_ids) != 1:
        details = ", ".join(f"{task}={record.get('feature_run_id')}" for task, record in model_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on feature_run_id: "
            f"{details}."
        )
    shared["feature_run_id"] = next(iter(feature_run_ids)) if len(feature_run_ids) == 1 else None
    shared["source_feature_run_ids"] = {
        task: record.get("feature_run_id")
        for task, record in model_records.items()
        if record.get("feature_run_id")
    }

    intended_modes = {record.get("intended_receiver_mode") for record in model_records.values()}
    intended_modes.discard(None)
    intended_modes.discard("unknown")
    if require_intended_receiver_mode and len(intended_modes) != 1:
        details = ", ".join(f"{task}={record.get('intended_receiver_mode')}" for task, record in model_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on intended_receiver_mode: "
            f"{details}."
        )
    shared["intended_receiver_mode"] = next(iter(intended_modes)) if len(intended_modes) == 1 else None
    shared["source_intended_receiver_modes"] = {
        task: record.get("intended_receiver_mode")
        for task, record in model_records.items()
        if record.get("intended_receiver_mode") not in (None, "unknown")
    }

    outcome_records = {task: model_records[task] for task in outcome_tasks if task in model_records}
    return_type_records = outcome_records if outcome_records else model_records
    return_types = {record.get("return_type") for record in return_type_records.values()}
    return_types.discard(None)
    if require_return_type and len(return_types) != 1:
        details = ", ".join(f"{task}={record.get('return_type')}" for task, record in return_type_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on return_type: "
            f"{details}."
        )
    shared["return_type"] = next(iter(return_types)) if len(return_types) == 1 else None
    shared["source_return_types"] = {
        task: record.get("return_type")
        for task, record in model_records.items()
        if record.get("return_type")
    }

    shared["graph_schema"] = aggregate_graph_schemas(
        {task: record["graph_schema"] for task, record in model_records.items()}
    )

    target_families = {record.get("target_family") for record in outcome_records.values()}
    target_families.discard(None)
    if require_target_family and len(target_families) > 1:
        details = ", ".join(f"{task}={record.get('target_family')}" for task, record in outcome_records.items())
        raise ValueError(
            "Selected outcome checkpoints do not agree on target_family: "
            f"{details}."
        )
    shared["target_family"] = next(iter(target_families)) if len(target_families) == 1 else None
    shared["source_target_families"] = {
        task: record.get("target_family")
        for task, record in model_records.items()
        if record.get("target_family")
    }

    return shared


def resolve_model_selection(
    required_tasks: list[str],
    bundle_id: str | None = None,
    explicit_model_ids: dict[str, str | None] | None = None,
    require_feature_run_id: bool = True,
    require_intended_receiver_mode: bool = True,
    require_return_type: bool = True,
    require_target_family: bool = True,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any] | None]:
    explicit_model_ids = explicit_model_ids or {}

    if bundle_id:
        resolved_model_ids, bundle = resolve_bundle_model_ids(
            bundle_id,
            required_tasks=required_tasks,
            explicit_model_ids=explicit_model_ids,
        )
    else:
        resolved_model_ids = require_explicit_model_ids(required_tasks, explicit_model_ids)
        bundle = None

    model_records = get_model_records(resolved_model_ids)
    shared = validate_model_record_consistency(
        model_records,
        require_feature_run_id=require_feature_run_id,
        require_intended_receiver_mode=require_intended_receiver_mode,
        require_return_type=require_return_type,
        require_target_family=require_target_family,
    )
    shared["model_records"] = model_records
    split_values = {
        int(record["train_split_percent"])
        for record in model_records.values()
        if record.get("train_split_percent") is not None
    }
    if len(split_values) > 1:
        raise ValueError(f"Selected model checkpoints do not agree on train_split_percent: {sorted(split_values)}.")
    if split_values:
        shared["train_split_percent"] = next(iter(split_values))
    split_manifest_ids = {
        str(record["split_manifest_id"])
        for record in model_records.values()
        if record.get("split_manifest_id")
    }
    if len(split_manifest_ids) > 1:
        raise ValueError(
            f"Selected model checkpoints do not agree on split_manifest_id: {sorted(split_manifest_ids)}."
        )
    if split_manifest_ids:
        shared["split_manifest_id"] = next(iter(split_manifest_ids))

    if bundle is not None:
        if require_feature_run_id and bundle.get("feature_run_id") and shared.get("feature_run_id") and bundle["feature_run_id"] != shared["feature_run_id"]:
            raise ValueError(
                f"Bundle {bundle_id!r} feature_run_id={bundle['feature_run_id']!r} does not match the selected model ids "
                f"(feature_run_id={shared['feature_run_id']!r})."
            )
        if require_intended_receiver_mode and bundle.get("intended_receiver_mode") and shared.get("intended_receiver_mode") and bundle["intended_receiver_mode"] != shared["intended_receiver_mode"]:
            raise ValueError(
                f"Bundle {bundle_id!r} intended_receiver_mode={bundle['intended_receiver_mode']!r} does not match the "
                f"selected model ids (intended_receiver_mode={shared['intended_receiver_mode']!r})."
            )
        if require_return_type and bundle.get("return_type") and shared.get("return_type") and bundle["return_type"] != shared["return_type"]:
            raise ValueError(
                f"Bundle {bundle_id!r} return_type={bundle['return_type']!r} does not match the selected model ids "
                f"(return_type={shared['return_type']!r})."
            )
        if require_target_family and bundle.get("target_family") and shared.get("target_family") and bundle["target_family"] != shared["target_family"]:
            raise ValueError(
                f"Bundle {bundle_id!r} target_family={bundle['target_family']!r} does not match the selected model ids "
                f"(target_family={shared['target_family']!r})."
            )

        if bundle.get("feature_run_id"):
            shared["bundle_feature_run_id"] = bundle.get("feature_run_id")
        shared["feature_run_id"] = shared.get("feature_run_id") or bundle.get("feature_run_id")
        shared["intended_receiver_mode"] = shared.get("intended_receiver_mode") or bundle.get("intended_receiver_mode")
        shared["return_type"] = shared.get("return_type") or bundle.get("return_type")
        shared["target_family"] = shared.get("target_family") or bundle.get("target_family")
        shared["train_split_percent"] = shared.get("train_split_percent") or bundle.get("train_split_percent")
        if bundle.get("split_manifest_id") and shared.get("split_manifest_id") and bundle["split_manifest_id"] != shared["split_manifest_id"]:
            raise ValueError(
                f"Bundle {bundle_id!r} split_manifest_id={bundle['split_manifest_id']!r} does not match selected "
                f"model checkpoints ({shared['split_manifest_id']!r})."
            )
        shared["split_manifest_id"] = shared.get("split_manifest_id") or bundle.get("split_manifest_id")

    return resolved_model_ids, shared, bundle


def get_model_graph_schema(model: GNN | None) -> dict[str, int | bool] | None:
    if model is None or not hasattr(model, "args"):
        return None
    edge_in_dim = int(model.args.get("edge_in_dim", 2))
    return {
        "node_in_dim": int(model.args.get("node_in_dim", 0)),
        "edge_in_dim": edge_in_dim,
        "add_v_edge_features": bool(model.args.get("add_v_edge_features", edge_in_dim > 2)),
        "add_relative_speed_edge_features": bool(model.args.get("add_relative_speed_edge_features", edge_in_dim > 4)),
    }


def validate_model_graph_schemas(models: dict[str, GNN | None]) -> dict[str, int | bool]:
    schemas = {
        name: schema
        for name, schema in ((name, get_model_graph_schema(model)) for name, model in models.items())
        if schema is not None
    }
    if not schemas:
        return {"edge_in_dim": 2, "add_v_edge_features": False, "add_relative_speed_edge_features": False}
    return aggregate_graph_schemas(schemas)


def infer_feature_graph_schema(feature_dir: str | Path) -> dict[str, int | bool]:
    feature_path = Path(feature_dir)
    for graph_file in sorted(feature_path.glob("*.pt")):
        try:
            graphs = torch.load(graph_file, weights_only=False)
            if not isinstance(graphs, list):
                continue
            first_graph = next((graph for graph in graphs if graph is not None), None)
            if first_graph is None:
                continue
            node_in_dim = int(first_graph.x.shape[1]) if getattr(first_graph, "x", None) is not None else 0
            edge_in_dim = int(first_graph.edge_attr.shape[1]) if getattr(first_graph, "edge_attr", None) is not None else 0
            return {
                "node_in_dim": node_in_dim,
                "edge_in_dim": edge_in_dim,
                "add_v_edge_features": bool(edge_in_dim > 2),
                "add_relative_speed_edge_features": bool(edge_in_dim > 4),
            }
        except Exception:
            continue
    raise FileNotFoundError(f"Could not infer graph schema from feature directory {feature_dir}.")


def runtime_feature_run_candidate_ids(
    shared_context: dict[str, Any],
    bundle: dict[str, Any] | None = None,
) -> list[str]:
    candidates: list[str] = []
    source_ids = shared_context.get("source_feature_run_ids", {})
    if isinstance(source_ids, dict):
        candidates.extend(str(feature_run_id) for feature_run_id in source_ids.values() if feature_run_id)
    for feature_run_id in (
        shared_context.get("feature_run_id"),
        shared_context.get("bundle_feature_run_id"),
        bundle.get("feature_run_id") if isinstance(bundle, dict) else None,
    ):
        if feature_run_id:
            candidates.append(str(feature_run_id))

    unique: list[str] = []
    seen: set[str] = set()
    for feature_run_id in candidates:
        if feature_run_id in seen:
            continue
        seen.add(feature_run_id)
        unique.append(feature_run_id)
    return unique


def _feature_run_sort_key(feature_run_id: str, metadata: dict[str, Any] | None) -> tuple[str, str]:
    return (str((metadata or {}).get("created_at") or ""), str(feature_run_id))


def all_runtime_feature_run_ids() -> list[str]:
    if not FEATURE_RUNS_DIR.exists():
        return []
    return sorted(path.name for path in FEATURE_RUNS_DIR.iterdir() if path.is_dir())


def _runtime_mode_candidates(intended_receiver_mode: str | None) -> list[str]:
    if intended_receiver_mode:
        return [str(intended_receiver_mode)]
    return list(RUNTIME_INTENDED_RECEIVER_MODE_PREFERENCE)


def validate_runtime_feature_run(
    feature_run_id: str,
    intended_receiver_mode: str,
    required_graph_schema: dict[str, int | bool],
    source_feature_metadata: dict[str, dict[str, Any]],
    *,
    context: str = "Selected feature artifacts",
) -> dict[str, Any]:
    del source_feature_metadata
    resolved_feature_run_id = resolve_feature_run_id(feature_run_id, required=True, allow_latest=False)
    feature_root = get_feature_run_root(str(resolved_feature_run_id))
    metadata = load_feature_run_metadata(str(resolved_feature_run_id), required=False) or {}

    modes = metadata.get("intended_receiver_modes")
    if isinstance(modes, list) and intended_receiver_mode not in {str(mode) for mode in modes}:
        raise ValueError(
            f"Feature run {resolved_feature_run_id!r} does not expose intended_receiver_mode={intended_receiver_mode!r}."
        )

    resolved_action_dir = get_resolved_action_dir(intended_receiver_mode, root=feature_root)
    if not resolved_action_dir.exists():
        raise FileNotFoundError(
            f"Resolved actions for intended_receiver_mode={intended_receiver_mode!r} not found at {resolved_action_dir}."
        )

    feature_schema = infer_feature_graph_schema(get_action_graph_dir(feature_root))
    validate_feature_graph_schema(feature_schema, required_graph_schema, context=context)

    return {
        "feature_run_id": str(resolved_feature_run_id),
        "feature_root": feature_root,
        "intended_receiver_mode": str(intended_receiver_mode),
        "feature_schema": feature_schema,
        "metadata": metadata,
    }


def _resolve_runtime_feature_run_candidate(
    feature_run_id: str,
    intended_receiver_mode: str | None,
    required_graph_schema: dict[str, int | bool],
    *,
    context: str,
) -> dict[str, Any]:
    rejected: list[str] = []
    for mode in _runtime_mode_candidates(intended_receiver_mode):
        try:
            return validate_runtime_feature_run(
                feature_run_id,
                mode,
                required_graph_schema,
                {},
                context=context,
            )
        except Exception as exc:
            rejected.append(f"{mode}: {type(exc).__name__}: {exc}")
    raise ValueError(
        f"Feature run {feature_run_id!r} is not compatible with any requested runtime mode. "
        f"Rejected modes: {'; '.join(rejected)}"
    )


def resolve_runtime_return_type(
    shared_context: dict[str, Any],
    explicit_return_type: str | None = None,
    *,
    preferred_task: str = "outcome_scoring",
    fallback: str = DEFAULT_RUNTIME_RETURN_TYPE,
) -> str:
    if explicit_return_type:
        return str(explicit_return_type)

    source_return_types = shared_context.get("source_return_types", {})
    if isinstance(source_return_types, dict) and source_return_types.get(preferred_task):
        return str(source_return_types[preferred_task])

    return_type = shared_context.get("return_type")
    return str(return_type) if return_type else fallback


def resolve_runtime_feature_run_context(
    explicit_feature_run_id: str | None,
    shared_context: dict[str, Any],
    bundle: dict[str, Any] | None,
    intended_receiver_mode: str | None,
    required_graph_schema: dict[str, int | bool],
    *,
    context: str = "Selected feature artifacts",
) -> dict[str, Any]:
    candidate_ids = runtime_feature_run_candidate_ids(shared_context, bundle)

    if explicit_feature_run_id:
        resolved = _resolve_runtime_feature_run_candidate(
            str(explicit_feature_run_id),
            intended_receiver_mode,
            required_graph_schema,
            context=context,
        )
        resolved["selection"] = "explicit"
        resolved["candidate_feature_run_ids"] = candidate_ids
        return resolved

    def select_compatible(feature_run_ids: list[str], selection: str) -> tuple[dict[str, Any] | None, list[str]]:
        compatible: list[dict[str, Any]] = []
        rejected: list[str] = []
        for feature_run_id in feature_run_ids:
            try:
                resolved = _resolve_runtime_feature_run_candidate(
                    feature_run_id,
                    intended_receiver_mode,
                    required_graph_schema,
                    context=context,
                )
            except Exception as exc:
                rejected.append(f"{feature_run_id}: {type(exc).__name__}: {exc}")
                continue
            compatible.append(resolved)
        if not compatible:
            return None, rejected
        compatible.sort(key=lambda item: _feature_run_sort_key(str(item["feature_run_id"]), item.get("metadata")))
        selected = compatible[-1]
        selected["selection"] = selection
        selected["candidate_feature_run_ids"] = candidate_ids
        return selected, rejected

    candidate_selection, rejected_candidates = select_compatible(candidate_ids, "newest_compatible")
    if candidate_selection is not None:
        return candidate_selection

    all_candidate_ids = [
        feature_run_id
        for feature_run_id in all_runtime_feature_run_ids()
        if feature_run_id not in set(candidate_ids)
    ]
    fallback_selection, rejected_fallbacks = select_compatible(
        all_candidate_ids,
        "newest_compatible_all_feature_runs",
    )
    if fallback_selection is not None:
        return fallback_selection

    rejected = rejected_candidates + rejected_fallbacks
    if not rejected:
        raise ValueError("No runtime feature runs are available. Pass --feature-run-id after generating compatible features.")
    raise ValueError(
        "No compatible runtime feature run could be selected. "
        f"Rejected candidates: {'; '.join(rejected)}"
    )


def adapt_batch_graphs_for_model(
    batch_graphs: Batch,
    model_args: dict[str, Any],
    *,
    context: str = "Loaded model",
) -> Batch:
    required_node_dim = int(model_args.get("node_in_dim", 0) or 0)
    actual_node_dim = int(batch_graphs.x.shape[1]) if getattr(batch_graphs, "x", None) is not None else 0
    if required_node_dim and actual_node_dim < required_node_dim:
        raise ValueError(
            f"{context} requires node_in_dim={required_node_dim}, "
            f"but runtime graphs provide node_in_dim={actual_node_dim}."
        )

    required_edge_dim = int(model_args.get("edge_in_dim", 0) or 0)
    actual_edge_dim = int(batch_graphs.edge_attr.shape[1]) if getattr(batch_graphs, "edge_attr", None) is not None else 0
    if required_edge_dim and actual_edge_dim < required_edge_dim:
        raise ValueError(
            f"{context} requires edge_in_dim={required_edge_dim}, "
            f"but runtime graphs provide edge_in_dim={actual_edge_dim}."
        )

    if (
        should_mask_possessor_v_edge_features_for_model(model_args)
        and actual_edge_dim >= 4
        and actual_node_dim > config.NODE_FEATURE_IS_POSSESSOR
    ):
        possessor_nodes = batch_graphs.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
        incident_edges = possessor_nodes[batch_graphs.edge_index[0]] | possessor_nodes[batch_graphs.edge_index[1]]
        batch_graphs.edge_attr[incident_edges, 2:4] = 0
    if (
        should_mask_possessor_relative_speed_edge_features_for_model(model_args)
        and actual_edge_dim >= 5
        and actual_node_dim > config.NODE_FEATURE_IS_POSSESSOR
    ):
        possessor_nodes = batch_graphs.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
        incident_edges = possessor_nodes[batch_graphs.edge_index[0]] | possessor_nodes[batch_graphs.edge_index[1]]
        batch_graphs.edge_attr[incident_edges, 4] = 0

    if required_edge_dim and actual_edge_dim > required_edge_dim:
        batch_graphs.edge_attr = batch_graphs.edge_attr[:, :required_edge_dim]
    if required_node_dim and actual_node_dim > required_node_dim:
        batch_graphs.x = batch_graphs.x[:, :required_node_dim]
    return batch_graphs


def should_mask_possessor_v_edge_features_for_model(model_args: dict[str, Any]) -> bool:
    mode = model_args.get("v_edge_feature_mode")
    if mode is not None:
        return normalize_v_edge_feature_mode(str(mode)) == V_EDGE_FEATURE_MODE_NO_POSS
    return bool(model_args.get("mask_possessor_v_edge_features", False))


def should_mask_possessor_relative_speed_edge_features_for_model(model_args: dict[str, Any]) -> bool:
    mode = model_args.get("relative_speed_edge_feature_mode")
    if mode is not None:
        return normalize_relative_speed_edge_feature_mode(str(mode)) == RELATIVE_SPEED_EDGE_FEATURE_MODE_NO_POSS
    return bool(model_args.get("mask_possessor_relative_speed_edge_features", False))


def estimate_propensity(dataset, model_id="pass_intent/00", device="cuda", min_clip=0.01, pin_memory: bool = True) -> torch.Tensor:
    model = load_model(model_id, device)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=pin_memory)
    likelihoods = []

    for batch_graphs, batch_labels, _ in tqdm(loader):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            batch_graphs = adapt_batch_graphs_for_model(batch_graphs, model.args, context=f"IPW model {model_id!r}")
            out: torch.Tensor = model(batch_graphs)
            for graph_index in range(batch_graphs.num_graphs):
                logits = out[
                    (batch_graphs.batch == graph_index)
                    & (batch_graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1)
                ]
                probs = nn.Softmax(dim=0)(logits).cpu().detach().numpy()
                likelihoods.append(probs[int(batch_labels[graph_index, 5].item())])

    return torch.Tensor(likelihoods).clip(min_clip)


def calc_pos_error(pred_xy, target_xy, aggfunc="mean"):
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_accuracy(y, y_hat, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(y_hat, dim=1) == y).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(y_hat, dim=1) == y).float().sum().item()


def calc_threshold_binary_metrics(y, y_hat, threshold: float) -> dict[str, float | int]:
    """Return threshold-dependent binary metrics using the documented strict comparison."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("Binary metric inputs must be non-empty and have identical lengths.")
    if not np.isfinite(y_score).all():
        raise ValueError("Binary prediction scores must be finite.")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Binary classification threshold must be finite and between 0 and 1.")
    y_pred = y_score > float(threshold)
    tp = int(np.sum((y_true == 1) & y_pred))
    fp = int(np.sum((y_true == 0) & y_pred))
    tn = int(np.sum((y_true == 0) & ~y_pred))
    fn = int(np.sum((y_true == 1) & ~y_pred))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "classification_threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def calc_binary_calibration_metrics(y, y_hat, n_bins: int = 10) -> dict[str, float]:
    """Return logistic calibration intercept/slope and equal-frequency ECE."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("Binary metric inputs must be non-empty and have identical lengths.")
    if not np.isfinite(y_score).all():
        raise ValueError("Binary prediction scores must be finite.")
    has_both_classes = bool(np.any(y_true == 0) and np.any(y_true == 1))
    has_score_variation = bool(np.ptp(y_score) > 0.0)
    intercept = np.nan
    slope = np.nan
    if has_both_classes and has_score_variation:
        eps = np.finfo(float).eps
        logits = np.log(np.clip(y_score, eps, 1.0 - eps) / np.clip(1.0 - y_score, eps, 1.0 - eps))
        try:
            calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logits.reshape(-1, 1), y_true)
            intercept = float(calibration.intercept_[0])
            slope = float(calibration.coef_[0, 0])
        except ValueError:
            pass
    bins = calc_equal_frequency_bins(y_true, y_score, n_bins=n_bins)
    if len(bins):
        ece = float(np.sum(
            bins["sample_count"].to_numpy(dtype=float) / len(y_true)
            * np.abs(bins["prediction_minus_observed"].to_numpy(dtype=float))
        ))
    else:
        ece = np.nan
    return {"calibration_intercept": intercept, "calibration_slope": slope, "ece": ece}


def calc_binary_metrics(y, y_hat, threshold: float | None = 0.5, *, include_calibration: bool = True):
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("Binary metric inputs must be non-empty and have identical lengths.")
    if not np.isfinite(y_score).all():
        raise ValueError("Binary prediction scores must be finite.")
    has_positive = np.sum(y_true) > 0
    has_negative = np.sum(y_true == 0) > 0

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_score) if has_positive and has_negative else np.nan,
        "pr_auc": average_precision_score(y_true, y_score) if has_positive else np.nan,
        "brier": brier_score_loss(y_true, y_score),
        "log_loss": log_loss(y_true, y_score, labels=[0, 1]) if has_positive else np.nan,
    }
    if include_calibration:
        metrics.update(calc_binary_calibration_metrics(y_true, y_score))
    if threshold is not None:
        metrics.update(calc_threshold_binary_metrics(y_true, y_score, float(threshold)))
    return metrics


def calc_binary_cohort_metrics(y) -> dict[str, float | int]:
    """Return cohort size and positive-label prevalence for a binary target."""
    y_true = np.asarray(y).reshape(-1)
    if not np.isfinite(y_true).all() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Binary cohort targets must contain finite 0/1 values.")
    sample_count = int(len(y_true))
    positive_count = int(y_true.sum())
    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "positive_prevalence": float(positive_count / sample_count) if sample_count else np.nan,
    }


def calc_pass_success_height_metrics(y, y_hat, observed_pass_high) -> tuple[dict[str, float | int], list[dict]]:
    """Calculate threshold-free pass-success metrics by observed pass-height class."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    pass_high = np.asarray(observed_pass_high, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_score) == len(pass_high)):
        raise ValueError("Pass-success height-stratification inputs must have identical lengths.")
    if not np.isfinite(pass_high).all() or not np.isin(pass_high, [0.0, 1.0]).all():
        raise ValueError("Observed pass-height labels must be finite binary pass_high values.")

    flat: dict[str, float | int] = {}
    rows: list[dict] = []
    for stratum, mask in (("observed_high", pass_high == 1.0), ("observed_non_high", pass_high == 0.0)):
        count = int(mask.sum())
        positive_count = int(y_true[mask].sum())
        prevalence = float(positive_count / count) if count else np.nan
        probability_metrics = (
            calc_binary_metrics(y_true[mask], y_score[mask], threshold=None)
            if count
            else {"roc_auc": np.nan, "brier": np.nan, "log_loss": np.nan}
        )
        row = {
            "stratum": stratum,
            "sample_count": count,
            "positive_count": positive_count,
            "success_prevalence": prevalence,
            "roc_auc": probability_metrics["roc_auc"],
            "brier": probability_metrics["brier"],
        }
        rows.append(row)
        prefix = f"pass_success_{stratum}"
        for key, value in row.items():
            if key != "stratum":
                flat[f"{prefix}_{key}"] = value
    return flat, rows


def calc_pass_success_predictor_metrics(
    y, observed_pass_high, predictors: dict[str, np.ndarray], threshold: float = 0.5,
) -> tuple[dict[str, float | int], list[dict]]:
    """Calculate pooled and observed-height metrics for aligned pass-success predictors."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    pass_high = np.asarray(observed_pass_high, dtype=float).reshape(-1)
    if len(y_true) != len(pass_high):
        raise ValueError("Pass-success targets and observed-height labels must have identical lengths.")
    flat: dict[str, float | int] = {}
    rows: list[dict] = []
    strata = (
        ("pooled", np.ones(len(y_true), dtype=bool)),
        ("observed_high", pass_high == 1.0),
        ("observed_non_high", pass_high == 0.0),
    )
    for predictor, values in predictors.items():
        scores = np.asarray(values, dtype=float).reshape(-1)
        if len(scores) != len(y_true) or not np.isfinite(scores).all():
            raise ValueError(f"Predictor {predictor!r} must contain one finite value per evaluated pass.")
        for stratum, mask in strata:
            count = int(mask.sum())
            positive_count = int(y_true[mask].sum())
            probability_metrics = (
                calc_binary_metrics(y_true[mask], scores[mask], threshold=threshold)
                if count
                else {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "log_loss": np.nan, "f1": np.nan}
            )
            row = {
                "predictor": predictor,
                "stratum": stratum,
                "sample_count": count,
                "positive_count": positive_count,
                "success_prevalence": float(positive_count / count) if count else np.nan,
                **probability_metrics,
            }
            rows.append(row)
            if stratum == "pooled":
                for key, value in row.items():
                    if key not in {"predictor", "stratum"}:
                        flat[f"pass_success_predictor_{predictor}_{key}"] = value
    return flat, rows


def calc_binary_diagnostic_rows(
    y,
    predictors: dict[str, np.ndarray],
    strata: tuple[tuple[str, np.ndarray], ...] | None = None,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
    curve_step: float = 0.01,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build comparable metric, calibration-bin, and threshold-curve rows for probabilities."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    if not len(y_true):
        raise ValueError("Binary diagnostics require at least one target.")
    if strata is None:
        strata = (("pooled", np.ones(len(y_true), dtype=bool)),)
    thresholds = np.round(np.arange(0.0, 1.0 + curve_step / 2.0, curve_step), 10)
    metric_rows: list[dict] = []
    calibration_rows: list[dict] = []
    curve_rows: list[dict] = []
    for predictor, values in predictors.items():
        scores = np.asarray(values, dtype=float).reshape(-1)
        if len(scores) != len(y_true) or not np.isfinite(scores).all():
            raise ValueError(f"Predictor {predictor!r} must contain one finite value per target.")
        for stratum, raw_mask in strata:
            mask = np.asarray(raw_mask, dtype=bool).reshape(-1)
            if len(mask) != len(y_true):
                raise ValueError(f"Stratum {stratum!r} must align with targets.")
            observed = y_true[mask]
            predicted = scores[mask]
            row = {
                "predictor": predictor,
                "stratum": stratum,
                "sample_count": int(len(observed)),
                "positive_count": int(observed.sum()),
                "success_prevalence": float(observed.mean()) if len(observed) else np.nan,
            }
            if len(observed):
                row.update(calc_binary_metrics(observed, predicted, threshold=threshold))
                bins = calc_equal_frequency_bins(observed, predicted, n_bins=n_bins)
                for bin_row in bins.to_dict("records"):
                    calibration_rows.append({"predictor": predictor, "stratum": stratum, **bin_row})
                for curve_threshold in thresholds:
                    curve_rows.append({
                        "predictor": predictor,
                        "stratum": stratum,
                        **calc_threshold_binary_metrics(observed, predicted, float(curve_threshold)),
                    })
            else:
                row.update({"classification_threshold": float(threshold)})
            metric_rows.append(row)
    return metric_rows, calibration_rows, curve_rows


def equal_frequency_slice_masks(values, prefix: str, n_slices: int = 5) -> list[tuple[str, np.ndarray, float, float]]:
    """Return deterministic equal-count slices plus their observed numeric bounds."""
    numeric = np.asarray(values, dtype=float).reshape(-1)
    if not len(numeric) or not np.isfinite(numeric).all():
        raise ValueError("Slice values must be a non-empty finite vector.")
    masks: list[tuple[str, np.ndarray, float, float]] = []
    for index, selected in enumerate(np.array_split(np.argsort(numeric, kind="stable"), min(n_slices, len(numeric))), start=1):
        mask = np.zeros(len(numeric), dtype=bool)
        mask[selected] = True
        masks.append((f"{prefix}_q{index}", mask, float(numeric[selected].min()), float(numeric[selected].max())))
    return masks


def calc_continuous_target_metrics(y, y_hat) -> dict[str, float]:
    """Summarize fidelity and linear calibration for a continuous [0, 1] target."""
    y_true = np.asarray(y, dtype=float).reshape(-1)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("Continuous metric inputs must be non-empty and have identical lengths.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_score).all():
        raise ValueError("Continuous metric inputs must be finite.")

    error = y_score - y_true
    has_variation = len(y_true) >= 2 and np.ptp(y_true) > 0 and np.ptp(y_score) > 0
    if has_variation:
        pearson_r = float(np.corrcoef(y_true, y_score)[0, 1])
        target_ranks = pd.Series(y_true).rank(method="average").to_numpy()
        prediction_ranks = pd.Series(y_score).rank(method="average").to_numpy()
        spearman_rho = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
        calibration_slope, calibration_intercept = np.polyfit(y_score, y_true, deg=1)
    else:
        pearson_r = np.nan
        spearman_rho = np.nan
        calibration_intercept = np.nan
        calibration_slope = np.nan

    clipped_scores = np.clip(y_score, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    soft_bce = -np.mean(y_true * np.log(clipped_scores) + (1.0 - y_true) * np.log(1.0 - clipped_scores))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "soft_bce": float(soft_bce),
        "mean_prediction": float(np.mean(y_score)),
        "mean_target": float(np.mean(y_true)),
        "mean_prediction_minus_target": float(np.mean(error)),
        "calibration_intercept": float(calibration_intercept),
        "calibration_slope": float(calibration_slope),
    }


def calc_equal_frequency_bins(y, y_hat, n_bins: int = 10) -> pd.DataFrame:
    """Return stable equal-frequency prediction bins for calibration-style plots."""
    y_true = np.asarray(y, dtype=float).reshape(-1)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    if len(y_true) != len(y_score):
        raise ValueError("Binned metric inputs must have identical lengths.")
    if n_bins < 1:
        raise ValueError("n_bins must be at least one.")
    columns = ["bin", "sample_count", "mean_prediction", "mean_observed", "prediction_minus_observed"]
    if len(y_true) == 0:
        return pd.DataFrame(columns=columns)

    order = np.argsort(y_score, kind="stable")
    rows = []
    for bin_index, indices in enumerate(np.array_split(order, min(int(n_bins), len(order))), start=1):
        predictions = y_score[indices]
        observations = y_true[indices]
        mean_prediction = float(np.mean(predictions))
        mean_observed = float(np.mean(observations))
        rows.append(
            {
                "bin": bin_index,
                "sample_count": int(len(indices)),
                "mean_prediction": mean_prediction,
                "mean_observed": mean_observed,
                "prediction_minus_observed": mean_prediction - mean_observed,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def calc_weighted_binary_probability_metrics(y, y_hat, sample_weight) -> dict[str, float]:
    """Calculate probability metrics for a non-empty, two-class weighted sample."""
    y_true = (np.asarray(y).reshape(-1) > 0).astype(int)
    y_score = np.asarray(y_hat, dtype=float).reshape(-1)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_score) == len(weights)):
        raise ValueError("Weighted metric inputs must have identical lengths.")
    if not np.isfinite(y_score).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weighted metric predictions and weights must be finite, with non-negative weights.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("Weighted pass-success evaluation has zero total effective weight.")
    if np.unique(y_true[weights > 0.0]).size < 2:
        raise ValueError("Weighted pass-success evaluation requires both successful and unsuccessful passes.")
    return {
        "high_pass_weighted_roc_auc": float(roc_auc_score(y_true, y_score, sample_weight=weights)),
        "high_pass_weighted_brier": float(brier_score_loss(y_true, y_score, sample_weight=weights)),
    }


def validate_target_flags(args) -> None:
    enabled_flags = sum(
        int(
            bool(getattr(args, name, False))
        )
        for name in ["use_xg", "use_xt", "use_goal_distance", "use_epv"]
    )
    if enabled_flags > 1:
        raise ValueError("--use_xg, --use_xt, --use_goal_distance, and --use_epv are mutually exclusive.")


def get_label_slice(labels: torch.Tensor, name: str) -> torch.Tensor:
    return labels[:, LABEL_INDEX[name]]


def get_outcome_targets(batch_labels: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    validate_target_flags(args)
    if getattr(args, "use_goal_distance", False):
        return get_label_slice(batch_labels, "scores_goal_distance"), get_label_slice(
            batch_labels, "concedes_goal_distance"
        )
    if getattr(args, "use_xt", False):
        return get_label_slice(batch_labels, "scores_xt"), get_label_slice(batch_labels, "concedes_xt")
    if getattr(args, "use_epv", False):
        return get_label_slice(batch_labels, "scores_epv"), get_label_slice(batch_labels, "concedes_epv")
    if getattr(args, "use_xg", False):
        return get_label_slice(batch_labels, "scores_xg"), get_label_slice(batch_labels, "concedes_xg")
    return get_label_slice(batch_labels, "scores"), get_label_slice(batch_labels, "concedes")


def get_outcome_diagnostic_targets(batch_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return get_label_slice(batch_labels, "scores_goal_next10"), get_label_slice(batch_labels, "concedes_goal_next10")


def adjust_dests(labels: torch.Tensor) -> torch.Tensor:
    start_xy = labels[:, LABEL_INDEX["start_x"] : LABEL_INDEX["start_y"] + 1]
    end_xy = labels[:, LABEL_INDEX["end_x"] : LABEL_INDEX["end_y"] + 1]
    intent_xy = labels[:, LABEL_INDEX["intent_x"] : LABEL_INDEX["intent_y"] + 1]

    # Masks for failed passes with valid coordinates
    is_failed_pass = (labels[:, 1] == 1) & (labels[:, LABEL_INDEX["success"]] == 0)
    has_valid_xy = (
        (start_xy[:, 0] >= 0)
        & (start_xy[:, 1] >= 0)
        & (end_xy[:, 0] >= 0)
        & (end_xy[:, 1] >= 0)
        & (intent_xy[:, 0] >= 0)
        & (intent_xy[:, 1] >= 0)
    )
    adjust_mask = is_failed_pass & has_valid_xy

    intended_len = torch.linalg.norm(start_xy - intent_xy, dim=1)
    actual_len = torch.linalg.norm(start_xy - end_xy, dim=1).clamp_min(1e-6)
    scale = (intended_len / actual_len).unsqueeze(1)
    end_xy_adj = start_xy + scale * (end_xy - start_xy)

    dests = torch.where(adjust_mask.unsqueeze(1), end_xy_adj, end_xy).clone()
    dests[:, 0] = dests[:, 0].clamp(0.0, FIELD_SIZE[0])
    dests[:, 1] = dests[:, 1].clamp(0.0, FIELD_SIZE[1])

    return dests


def build_dest_features(graphs: Batch, dest_xy: torch.Tensor, oppo_aware=True) -> torch.Tensor:
    B = graphs.num_graphs  # batch_size
    G = dest_xy.size(0)  # grid_size

    feat: torch.Tensor = graphs.x
    batch: torch.Tensor = graphs.batch

    poss_mask = feat[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
    team_mask = (feat[:, config.NODE_FEATURE_IS_TEAMMATE] == 1) & (feat[:, config.NODE_FEATURE_IS_GOAL] == 0)
    oppo_mask = (feat[:, config.NODE_FEATURE_IS_TEAMMATE] == 0) & (feat[:, config.NODE_FEATURE_IS_GOAL] == 0)

    dest_feat = []

    for graph_idx in range(B):
        graph_mask = batch == graph_idx
        xy_slice = slice(config.NODE_FEATURE_X, config.NODE_FEATURE_Y + 1)
        poss_xy = feat[graph_mask & poss_mask, xy_slice]  # [1, 2]
        team_xy = feat[graph_mask & team_mask, xy_slice]  # [team, 2]
        oppo_xy = feat[graph_mask & oppo_mask, xy_slice]  # [oppo, 2]

        poss_dx = dest_xy[:, 0] - poss_xy[0, 0]
        poss_dy = dest_xy[:, 1] - poss_xy[0, 1]

        # Distance to the nearest teammate
        team_dxy = dest_xy.unsqueeze(1) - team_xy.unsqueeze(0)  # [G, team, 2]
        team_nn_dist = (team_dxy**2).sum(-1).min(-1).values.sqrt()  # [G, team, 2] -> [G, team] -> [G]

        if oppo_aware:
            # Distance to the nearest opponent
            oppo_dxy = dest_xy.unsqueeze(1) - oppo_xy.unsqueeze(0)  # [G, oppo, 2]
            oppo_nn_dist = (oppo_dxy**2).sum(-1).min(-1).values.sqrt()  # [G, oppo, 2] -> [G, oppo] -> [G]

            # Distance from the possessor-cell line segment to its nearest opponent
            pass_xy = dest_xy - poss_xy  # [G, 2]
            pass_len = (pass_xy**2).sum(-1).clamp_min(1e-6)  # [G]
            dot = pass_xy @ (oppo_xy - poss_xy).t()  # [G, oppo]
            proj_ratio = (dot / pass_len.unsqueeze(1)).clamp(0.0, 1.0)  # [G, oppo]
            proj_point = poss_xy.unsqueeze(1) + proj_ratio.unsqueeze(-1) * pass_xy.unsqueeze(1)  # [G, oppo, 2]
            pass_oppo_dist = ((proj_point - oppo_xy.unsqueeze(0)) ** 2).sum(-1).sqrt()  # [G, oppo]
            pass_oppo_nn_dist, _ = pass_oppo_dist.min(dim=1)  # [G]

        else:
            oppo_nn_dist = torch.zeros(G, device=dest_xy.device)
            pass_oppo_nn_dist = torch.zeros(G, device=dest_xy.device)

        dest_feat_i = [dest_xy[:, 0], dest_xy[:, 1], poss_dx, poss_dy, team_nn_dist, oppo_nn_dist, pass_oppo_nn_dist]
        dest_feat.append(torch.stack(dest_feat_i, dim=1))  # [G, d]

    dest_feat = torch.cat(dest_feat, dim=0)  # [B*G, d]
    return dest_feat.reshape(B, G, -1).to(dest_xy.device)  # [B, G, d]


def get_grid_xy(grid_size: Tuple[int, int] = FIELD_SIZE, device="cuda") -> torch.Tensor:
    cell_x = FIELD_SIZE[0] / grid_size[0]
    cell_y = FIELD_SIZE[1] / grid_size[1]

    grid_size = (int(grid_size[0]), int(grid_size[1]))
    x_edges = torch.linspace(0, FIELD_SIZE[0], grid_size[0] + 1, device=device, dtype=torch.float32)
    y_edges = torch.linspace(FIELD_SIZE[1], 0, grid_size[1] + 1, device=device, dtype=torch.float32)
    x_centers = x_edges[:-1] + cell_x / 2
    y_centers = y_edges[1:] + cell_y / 2

    grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()]).T

    return grid_xy


def cartesian_to_polar(xy: torch.Tensor) -> torch.Tensor:
    goal_dx = FIELD_SIZE[0] - xy[:, 0]
    goal_dy = xy[:, 1] - FIELD_SIZE[1] / 2
    goal_dist = torch.sqrt(goal_dx**2 + goal_dy**2)
    goal_angle = torch.atan2(goal_dy, goal_dx)  # (-pi/2, pi/2) if x < 105
    return torch.stack([goal_dist, goal_angle], dim=1)


def find_cell_index(xy: torch.Tensor, grid_size: Tuple[int, int] = FIELD_SIZE) -> torch.Tensor:
    cell_x = FIELD_SIZE[0] / grid_size[0]
    cell_y = FIELD_SIZE[1] / grid_size[1]

    x_index = torch.floor(xy[:, 0] / cell_x).long().clamp(0, grid_size[0] - 1)
    y_index = (grid_size[1] - 1 - torch.floor(xy[:, 1] / cell_y)).long().clamp(0, grid_size[1] - 1)

    return y_index * grid_size[0] + x_index


def run_epoch(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Adam = None,
    device: str = "cuda",
    pos_weight: float = 1.0,
    train: bool = False,
    return_outcome_evaluation: bool = False,
):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)
    pos_weight = torch.tensor(pos_weight)

    if args.gnn_task in ["node_binary", "graph_binary"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0}
    elif args.gnn_task in ["node_selection", "graph_multiclass"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "accuracy": 0, "mrr": 0}
    elif args.gnn_task in ["node_regression", "graph_regression"]:
        metrics = {"count": 0, "mse_loss": 0, "l1_loss": 0}

    weighted_pass_success_eval = bool(getattr(args, "weighted_pass_success_metrics", False))
    if weighted_pass_success_eval and args.task != "pass_success":
        raise ValueError("weighted_pass_success_metrics is only supported for task='pass_success'.")
    weighted_predictions: list[np.ndarray] = []
    weighted_targets: list[np.ndarray] = []
    weighted_effective_weights: list[np.ndarray] = []
    observed_pass_height_labels: list[np.ndarray] = []
    physical_xpass_predictions: list[np.ndarray] = []
    combined_success_predictions: list[np.ndarray] = []
    combined_learning_weights: list[np.ndarray] = []
    evaluation_pass_distances: list[np.ndarray] = []
    observed_pass_max_heights: list[np.ndarray] = []
    binary_predictions: list[np.ndarray] = []
    binary_targets: list[np.ndarray] = []
    binary_threshold: float | None = None
    outcome_predictions: list[np.ndarray] = []
    outcome_targets: list[np.ndarray] = []
    outcome_diagnostics: list[np.ndarray] = []
    outcome_execution_branches: list[np.ndarray] = []

    for batch_index, (batch_graphs, batch_labels, batch_ipw) in enumerate(loader):
        batch_graphs: Batch = batch_graphs.to(device)
        batch_ipw: torch.Tensor = batch_ipw.to(device)
        index_range = torch.unique(batch_graphs.batch)

        metrics["count"] += batch_graphs.num_graphs
        outcome_scoring, outcome_conceding = get_outcome_targets(batch_labels := batch_labels.to(device), args)
        diagnostic_scoring, diagnostic_conceding = get_outcome_diagnostic_targets(batch_labels)

        if args.include_out:
            # One node per player and one ball-out node per graph instance
            batch = torch.cat([batch_graphs.batch, index_range])
        else:
            batch = batch_graphs.batch

        batch_labels[batch_labels[:, 6] == -1, 6] = batch_labels[batch_labels[:, 6] == -1, 4]  # -1 to n_players

        if "dest" in args.task:
            if getattr(args, "adjust_dest", False):
                batch_dests = adjust_dests(batch_labels)
            else:
                batch_dests = batch_labels[:, 10:12].clone()

            if getattr(args, "normalize_dest", False):
                assert not args.task.endswith("dest")
                batch_dests[:, 0] /= float(FIELD_SIZE[0])
                batch_dests[:, 1] /= float(FIELD_SIZE[1])

            elif getattr(args, "polar_dest", False):
                polar_dests = cartesian_to_polar(batch_dests)
                batch_dests = torch.cat([batch_dests, polar_dests], axis=1)

        else:
            batch_dests = None

        if "pass_dest" in args.task:
            grid_xy = get_grid_xy(device=device)  # [G, 2]

            if getattr(args, "more_dest_features", False):
                oppo_aware = "oppo_agn" not in args.task
                grid_features = build_dest_features(batch_graphs, grid_xy, oppo_aware)  # [B, G, d]
            else:
                grid_features = grid_xy.clone()  # [G, 2]

            if train:
                out: torch.Tensor = unwrap_model(model).forward_grid(batch_graphs, grid_features)
            else:
                with torch.no_grad():
                    out: torch.Tensor = unwrap_model(model).forward_grid(batch_graphs, grid_features)

        else:
            if train:
                out: torch.Tensor = model(batch_graphs, batch_dests)
            else:
                with torch.no_grad():
                    out: torch.Tensor = model(batch_graphs, batch_dests)

        if args.gnn_task == "node_selection":  # {pass/action}_intent, {success/failure}_receiver
            if args.task.split("_")[1] == "intent":
                target = batch_labels[:, 5].clone().long()
            elif args.task.split("_")[1] == "receiver":
                target = batch_labels[:, 6].clone().long()

            loss_fn = nn.CrossEntropyLoss()
            pred_loss = 0
            accuracy = 0

            for graph_index in index_range:
                if args.task in [
                    "pass_intent",
                    "success_intent",
                    "pass_intent_oppo_agn",
                    "action_intent",
                    "success_receiver",
                ]:
                    # Only take teammate nodes
                    assert not args.include_out
                    pred_i = out[
                        (batch == graph_index)
                        & (batch_graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1)
                    ]  # [N_i]
                    target_i = target[graph_index]

                elif args.task == "failure_receiver":
                    # Only take opponent nodes
                    if args.include_out:
                        ball_out_mask = torch.ones(batch_graphs.num_graphs).bool().to(device)
                        failure_mask = torch.cat(
                            [batch_graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 0, ball_out_mask]
                        )
                    else:
                        failure_mask = batch_graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 0
                    pred_i = out[(batch == graph_index) & failure_mask]
                    n_teammates = (
                        (batch_graphs.batch == graph_index)
                        & (batch_graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1)
                    ).sum()
                    target_i = target[graph_index] - n_teammates

                else:  # pass_receiver, dest_receiver
                    pred_i = out[batch == graph_index]
                    target_i = target[graph_index]

                pred_i = pred_i.reshape(-1)
                if target_i.numel() != 1:
                    raise ValueError(
                        f"Expected one node-selection target for graph {int(graph_index.item())}, "
                        f"got shape {tuple(target_i.shape)}."
                    )
                target_i = target_i.reshape(())
                pred_loss += loss_fn(pred_i.unsqueeze(0), target_i.unsqueeze(0))
                accuracy += (pred_i.argmax() == target_i).float()

                rank = (pred_i.argsort(descending=True) == target_i).nonzero(as_tuple=True)[0].item() + 1
                metrics["mrr"] += 1.0 / rank

            pred_loss /= index_range.shape[0]
            metrics["accuracy"] += accuracy.item()

        elif args.gnn_task == "node_binary":  # {pass/action}_success, outcome_{scoring/conceding}, intent_return
            intent = batch_labels[:, 5].clone().long()

            if args.task in ["pass_success", "pass_height", "action_success"]:
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index]])
                pred = torch.stack(pred)

                target_name = "pass_high" if args.task == "pass_height" else "success"
                target = get_label_slice(batch_labels, target_name)
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)
                short_residual_lambda, long_residual_lambda = resolved_residual_regularization_lambdas(args)
                if (
                    args.task == "pass_success"
                    and (short_residual_lambda > 0 or long_residual_lambda > 0)
                    and getattr(args, "use_physical_xpass", False)
                    and getattr(args, "model_variant", None)
                    in {"gat_phys_logit_offset", "gat_phys_logit_offset_regularized"}
                ):
                    delta_gat = unwrap_model(model).decoder.latest_delta_gat
                    if delta_gat is None:
                        raise ValueError("Residual regularization requested, but decoder did not expose latest_delta_gat.")
                    delta_observed = []
                    distance_observed = []
                    for graph_index in index_range:
                        graph_mask = batch == graph_index
                        target_index = intent[graph_index]
                        delta_observed.append(delta_gat[graph_mask][target_index])
                        distance_observed.append(batch_graphs.x[graph_mask][target_index, config.NODE_FEATURE_POSS_DIST])
                    delta_observed = torch.stack(delta_observed)
                    distance_observed = torch.stack(distance_observed).to(device=delta_observed.device, dtype=delta_observed.dtype)
                    residual_l2 = delta_observed.pow(2).mean()
                    threshold = float(residual_distance_threshold(args))
                    residual_lambdas = torch.where(
                        distance_observed <= threshold,
                        torch.full_like(distance_observed, float(short_residual_lambda)),
                        torch.full_like(distance_observed, float(long_residual_lambda)),
                    )
                    pred_loss = pred_loss + (residual_lambdas * delta_observed.pow(2)).mean()
                    metrics.setdefault("residual_l2", 0)
                    metrics["residual_l2"] += residual_l2.item() * batch_graphs.num_graphs

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = target.cpu().detach().numpy()
                evaluate_xpass = bool(getattr(args, "evaluate_xpass", False))
                evaluate_combined = bool(getattr(args, "evaluate_combined_success", False))
                if evaluate_xpass or evaluate_combined:
                    xpass_values = getattr(batch_graphs, EVALUATION_XPASS_PROB_ATTR, None)
                    distance_values = getattr(batch_graphs, EVALUATION_XPASS_DISTANCE_ATTR, None)
                    nearest_values = getattr(batch_graphs, EVALUATION_XPASS_NEAREST_OPPONENT_DISTANCE_ATTR, None)
                    height_values = getattr(batch_graphs, EVALUATION_XPASS_PASS_HEIGHT_ATTR, None)
                    if xpass_values is None or distance_values is None:
                        raise ValueError("Physical xPass evaluation requires cached xPass and pass-distance tensors.")
                    observed_xpass = []
                    observed_distance = []
                    observed_nearest = []
                    observed_height = []
                    for graph_index in index_range:
                        graph_mask = batch == graph_index
                        target_index = intent[graph_index]
                        observed_xpass.append(xpass_values[graph_mask][target_index])
                        observed_distance.append(distance_values[graph_mask][target_index])
                        if nearest_values is not None:
                            observed_nearest.append(nearest_values[graph_mask][target_index])
                        if height_values is not None:
                            observed_height.append(height_values[graph_mask][target_index])
                    xpass_array = torch.stack(observed_xpass).cpu().numpy().astype(float)
                    distance_array = torch.stack(observed_distance).cpu().numpy().astype(float)
                    if not np.isfinite(xpass_array).all() or not np.isfinite(distance_array).all():
                        raise ValueError("Physical xPass evaluation requires finite observed-target xPass and distance values.")
                    # The combined diagnostic needs the raw physical component as well.
                    if evaluate_xpass or evaluate_combined:
                        physical_xpass_predictions.append(xpass_array)
                    evaluation_pass_distances.append(distance_array)
                    if evaluate_combined:
                        blend_kwargs = {}
                        weight_version = str(getattr(args, "xpass_weight", "")).lower()
                        if weight_version == "v2":
                            if len(observed_nearest) != len(observed_xpass):
                                raise ValueError("Combined xPass weight v2 requires cached nearest-opponent distances.")
                            blend_kwargs["distance_to_nearest_opponent"] = torch.stack(observed_nearest).cpu().numpy()
                        if weight_version == "v4":
                            if len(observed_height) != len(observed_xpass):
                                raise ValueError("Combined xPass weight v4 requires cached pass-height probabilities.")
                            blend_kwargs.update(
                                pass_height=torch.stack(observed_height).cpu().numpy(),
                                v4_power=float(args.v4_power),
                                v4_zero=float(args.v4_zero),
                                v4_discount=bool(args.discount),
                            )
                        combined = blend_physical_xpass_predictions(
                            pass_success_model=np.asarray(y_hat, dtype=float),
                            xpass=xpass_array,
                            pass_distance=distance_array,
                            weight_version=weight_version,
                            **blend_kwargs,
                        )
                        combined_success_predictions.append(np.asarray(combined, dtype=float))
                        if weight_version == "v2":
                            learning_weight = physical_xpass_blend_weight_v2(
                                distance_array, blend_kwargs["distance_to_nearest_opponent"]
                            )
                        elif weight_version == "v3":
                            learning_weight = physical_xpass_blend_weight_v3(distance_array)
                        elif weight_version == "v4":
                            learning_weight = physical_xpass_blend_weight_v4(
                                distance_array,
                                blend_kwargs["pass_height"],
                                power=float(args.v4_power),
                                zero_point=float(args.v4_zero),
                                use_discount=bool(args.discount),
                            )
                        else:
                            learning_weight = physical_xpass_blend_weight(distance_array)
                        combined_learning_weights.append(np.asarray(learning_weight, dtype=float))
                if weighted_pass_success_eval:
                    pass_heights = getattr(batch_graphs, PHYSICAL_XPASS_PASS_HEIGHT_ATTR, None)
                    pass_distances = getattr(batch_graphs, PHYSICAL_XPASS_DISTANCE_ATTR, None)
                    if pass_heights is None or pass_distances is None:
                        raise ValueError(
                            "Weighted pass-success evaluation requires cached pass-height probabilities and pass distances."
                        )
                    observed_heights = []
                    observed_distances = []
                    for graph_index in index_range:
                        graph_mask = batch == graph_index
                        target_index = intent[graph_index]
                        observed_heights.append(pass_heights[graph_mask][target_index])
                        observed_distances.append(pass_distances[graph_mask][target_index])
                    effective_weights = physical_xpass_blend_weight_v4(
                        torch.stack(observed_distances),
                        torch.stack(observed_heights),
                        power=float(getattr(args, "v4_power", 4.0)),
                        zero_point=float(getattr(args, "v4_zero", 0.7)),
                        use_discount=bool(getattr(args, "discount", True)),
                    )
                    weighted_predictions.append(np.asarray(y_hat, dtype=float))
                    weighted_targets.append(np.asarray(y, dtype=float))
                    weighted_effective_weights.append(effective_weights.detach().cpu().numpy().astype(float))
                if bool(
                    getattr(args, "observed_pass_height_stratification", False)
                    or getattr(args, "evaluate_xpass", False)
                    or getattr(args, "evaluate_combined_success", False)
                ):
                    observed_pass_height_labels.append(
                        get_label_slice(batch_labels, "pass_high").cpu().detach().numpy().astype(float)
                    )
                if args.task == "pass_height":
                    observed_pass_max_heights.append(
                        get_label_slice(batch_labels, "pass_max_ball_z").cpu().detach().numpy().astype(float)
                    )
                threshold = (
                    float(getattr(args, "classification_threshold", 0.5))
                    if args.task.endswith("success") or args.task == "pass_height"
                    else 0.1
                )
                binary_predictions.append(np.asarray(y_hat, dtype=float))
                binary_targets.append(np.asarray(y, dtype=float))
                binary_threshold = threshold

            elif args.task in ["outcome_scoring", "outcome_conceding"]:
                outcome = get_label_slice(batch_labels, "success").clone().long()
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index], outcome[graph_index]])
                pred = torch.stack(pred)

                target = outcome_scoring if args.task.endswith("scoring") else outcome_conceding
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = diagnostic_scoring if args.task.endswith("scoring") else diagnostic_conceding
                y = y.cpu().detach().numpy()
                binary_predictions.append(np.asarray(y_hat, dtype=float))
                binary_targets.append(np.asarray(y, dtype=float))
                binary_threshold = getattr(args, "f1_outcome_threshold", None)
                outcome_predictions.append(np.asarray(y_hat, dtype=float))
                outcome_targets.append(target.cpu().detach().numpy().astype(float))
                outcome_diagnostics.append(np.asarray(y, dtype=float))
                outcome_execution_branches.append(outcome.cpu().detach().numpy().astype(int))

            elif args.task in ["intent_return", "intent_return_oppo_agn"]:
                pred_s = []
                pred_c = []
                for graph_index in index_range:
                    pred_s.append(out[batch == graph_index][intent[graph_index], 0])
                    pred_c.append(out[batch == graph_index][intent[graph_index], 1])
                pred_s = torch.stack(pred_s)
                pred_c = torch.stack(pred_c)

                pred_loss_s = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred_s, outcome_scoring)
                pred_loss_c = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred_c, outcome_conceding)
                pred_loss = pred_loss_s + pred_loss_c

                # Calculate performance metrics only for goal-scoring prediction for simplicity
                y_hat = torch.sigmoid(pred_s).cpu().detach().numpy()
                y = diagnostic_scoring.cpu().detach().numpy()
                binary_predictions.append(np.asarray(y_hat, dtype=float))
                binary_targets.append(np.asarray(y, dtype=float))
                binary_threshold = 0.1

        elif args.gnn_task == "node_regression":  # outcome_return
            intent = batch_labels[:, 5].clone().long()
            outcome = get_label_slice(batch_labels, "success").clone().long()

            pred = []
            for graph_index in index_range:
                pred.append(out[batch == graph_index][intent[graph_index], outcome[graph_index]])
            pred = torch.stack(pred) * 2 - 1  # Transform output to range from [0, 1] to [-1, 1]

            target = outcome_scoring - outcome_conceding
            pred_loss = nn.MSELoss()(pred, target)
            metrics["mse_loss"] += pred_loss.item() * batch_graphs.num_graphs

        elif args.gnn_task == "graph_binary":
            # overll_{scoring/conceding}, dest_{outcome/scoring/conceding}, shot_blocking
            if args.task in ["shot_blocking", "dest_success"]:
                target = (
                    get_label_slice(batch_labels, "success")
                    if args.task.endswith("success")
                    else get_label_slice(batch_labels, "blocked")
                )
                pred_loss = nn.BCEWithLogitsLoss()(out, target)

                y_hat = torch.sigmoid(out).cpu().detach().numpy()
                y = target.cpu().detach().numpy()
                binary_predictions.append(np.asarray(y_hat, dtype=float))
                binary_targets.append(np.asarray(y, dtype=float))
                binary_threshold = 0.5

            elif args.task.split("_")[0] in ["overall", "dest"]:  # {overall/dest}_{scoring/conceding}
                if args.task.startswith("dest"):
                    outcome = get_label_slice(batch_labels, "success").clone().long()
                    pred = out[tuple([list(range(batch_graphs.num_graphs)), outcome])]
                else:
                    pred = out

                target = outcome_scoring if args.task.endswith("scoring") else outcome_conceding
                pred_loss = nn.BCEWithLogitsLoss()(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = diagnostic_scoring if args.task.endswith("scoring") else diagnostic_conceding
                y = y.cpu().detach().numpy()
                binary_predictions.append(np.asarray(y_hat, dtype=float))
                binary_targets.append(np.asarray(y, dtype=float))
                binary_threshold = 0.1

        elif args.gnn_task == "graph_multiclass":  # pass_dest
            if args.task == "pass_dest":
                sigma = getattr(args, "dest_sigma", 3.0)
                dist_to_target = ((grid_xy.unsqueeze(0) - batch_dests.unsqueeze(1)) ** 2).sum(-1).float()  # [B, G]
                soft_target = torch.exp(-dist_to_target / (2 * sigma**2))
                soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)  # [B, G]

                log_probs = F.log_softmax(out, dim=-1)
                pred_loss = F.kl_div(log_probs, soft_target, reduction="batchmean")

                pred_xy = grid_xy[out.argmax(dim=1)]  # [B, G] to [B, 2]
                dist_error = torch.linalg.norm(pred_xy - batch_dests, dim=1)
                metrics["accuracy"] += (dist_error <= sigma).float().sum().item()

                target = find_cell_index(batch_dests)  # [B]
                pos = (out.argsort(dim=1, descending=True) == target.unsqueeze(1)).nonzero(as_tuple=False)  # [B, 2]
                rank = pos[:, 1].to(torch.float32) + 1.0
                metrics["mrr"] += (1.0 / rank).sum().item()

        elif args.gnn_task == "graph_regression":  # overall_return
            target = outcome_scoring - outcome_conceding

            pred = torch.sigmoid(out) * 2 - 1  # Transform output to range from [0, 1] to [-1, 1]
            pred_loss = nn.MSELoss()(pred, target)
            metrics["mse_loss"] += pred_loss.item() * batch_graphs.num_graphs

        if "ce_loss" in metrics:
            metrics["ce_loss"] += pred_loss.item() * batch_graphs.num_graphs

        l1_loss = l1_regularizer(model, lambda_l1=args.lambda_l1)
        metrics["l1_loss"] += l1_loss.item() * batch_graphs.num_graphs

        if train:
            optimizer.zero_grad()
            loss = pred_loss + l1_loss
            loss.backward()
            nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.clip)
            optimizer.step()

        if train and batch_index % args.print_freq == 0:
            interim_metrics = dict()
            for key, value in metrics.items():
                if key == "count":
                    continue
                interim_metrics[key] = value / metrics["count"]
            if binary_predictions:
                interim_metrics.update(
                calc_binary_metrics(
                    np.concatenate(binary_targets),
                    np.concatenate(binary_predictions),
                    threshold=binary_threshold,
                    include_calibration=False,
                )
                )
            print(f"[{batch_index:>{len(str(n_batches))}d}/{n_batches}]  {get_losses_str(interim_metrics)}")

    for key, value in metrics.items():
        if key == "count":
            continue
        metrics[key] = value / metrics["count"]

    if binary_predictions:
        pooled_binary_targets = np.concatenate(binary_targets)
        metrics.update(
            calc_binary_metrics(
                pooled_binary_targets,
                np.concatenate(binary_predictions),
                threshold=binary_threshold,
            )
        )
        if args.task == "pass_height":
            metrics.update(calc_binary_cohort_metrics(pooled_binary_targets))

    outcome_evaluation = None
    if outcome_predictions:
        outcome_evaluation = {
            "prediction": np.concatenate(outcome_predictions),
            "target": np.concatenate(outcome_targets),
            "diagnostic": np.concatenate(outcome_diagnostics),
            "execution_branch": np.concatenate(outcome_execution_branches),
        }
        target_metrics = calc_continuous_target_metrics(
            outcome_evaluation["target"], outcome_evaluation["prediction"]
        )
        metrics.update({f"xt_target_{key}": value for key, value in target_metrics.items()})

    if weighted_pass_success_eval:
        metrics.update(
            calc_weighted_binary_probability_metrics(
                np.concatenate(weighted_targets),
                np.concatenate(weighted_predictions),
                np.concatenate(weighted_effective_weights),
            )
        )

    pass_success_height_metrics = None
    if observed_pass_height_labels:
        pass_success_height_metrics, pass_success_height_rows = calc_pass_success_height_metrics(
            np.concatenate(binary_targets),
            np.concatenate(binary_predictions),
            np.concatenate(observed_pass_height_labels),
        )
        metrics.update(pass_success_height_metrics)

    pass_success_predictor_rows = None
    if physical_xpass_predictions or combined_success_predictions:
        predictors = {"learning": np.concatenate(binary_predictions)}
        metric_name = str(getattr(args, "xpass_metric", "xpass"))
        if physical_xpass_predictions:
            predictors[f"physical_xpass_{metric_name.removesuffix('_xpass')}"] = np.concatenate(
                physical_xpass_predictions
            )
        if combined_success_predictions:
            predictors[f"combined_{getattr(args, 'xpass_weight', 'unknown')}"] = np.concatenate(
                combined_success_predictions
            )
        predictor_metrics, pass_success_predictor_rows = calc_pass_success_predictor_metrics(
            np.concatenate(binary_targets),
            np.concatenate(observed_pass_height_labels),
            predictors,
            threshold=float(getattr(args, "classification_threshold", 0.5)),
        )
        metrics.update(predictor_metrics)
        if combined_learning_weights:
            learning_weights = np.concatenate(combined_learning_weights)
            metrics.update(
                {
                    "combined_learning_weight_mean": float(np.mean(learning_weights)),
                    "combined_learning_weight_min": float(np.min(learning_weights)),
                    "combined_learning_weight_max": float(np.max(learning_weights)),
                }
            )

    if return_outcome_evaluation:
        return metrics, outcome_evaluation
    if bool(getattr(args, "return_pass_success_height_evaluation", False)):
        evaluation = {
            "height_rows": pass_success_height_rows if observed_pass_height_labels else [],
            "predictor_rows": pass_success_predictor_rows or [],
        }
        if pass_success_predictor_rows:
            evaluation["predictor_diagnostics"] = {
                "targets": np.concatenate(binary_targets),
                "predictors": predictors,
                "observed_pass_high": np.concatenate(observed_pass_height_labels),
                "pass_distance": np.concatenate(evaluation_pass_distances),
                "combined_learning_weight": np.concatenate(combined_learning_weights) if combined_learning_weights else None,
            }
        return metrics, evaluation
    if bool(getattr(args, "return_binary_diagnostics", False)):
        return metrics, {
            "targets": np.concatenate(binary_targets),
            "predictions": np.concatenate(binary_predictions),
            "observed_pass_max_height": (
                np.concatenate(observed_pass_max_heights) if observed_pass_max_heights else None
            ),
        }
    return metrics
