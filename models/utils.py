import argparse
import json
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
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from datatools.config import FIELD_SIZE, LABEL_INDEX
from models.gnn import GNN
from project_config import (
    SAVED_DIR,
    get_model_bundle_root,
    get_task_saved_dir,
    infer_legacy_model_context,
    infer_target_family,
    load_model_bundle_metadata,
    load_model_splits,
)


FEATURE_SIGNATURE_KEYS = (
    "xy_only",
    "possessor_aware",
    "keeper_aware",
    "ball_z_aware",
    "poss_vel_aware",
    "extend_features",
    "filter_blockers",
    "sparsify",
    "max_edge_dist",
    "add_v_edge_features",
    "node_in_dim",
    "edge_in_dim",
)


def extract_model_feature_signature(args: dict[str, Any]) -> dict[str, Any]:
    signature = {
        "xy_only": bool(args.get("xy_only", False)),
        "possessor_aware": bool(args.get("possessor_aware", False)),
        "keeper_aware": bool(args.get("keeper_aware", False)),
        "ball_z_aware": bool(args.get("ball_z_aware", False)),
        "poss_vel_aware": bool(args.get("poss_vel_aware", False)),
        "extend_features": bool(args.get("extend_features", False)),
        "filter_blockers": bool(args.get("filter_blockers", False)),
        "sparsify": args.get("sparsify", "none"),
        "max_edge_dist": args.get("max_edge_dist", 10),
        "node_in_dim": int(args.get("node_in_dim", 0)),
        "edge_in_dim": int(args.get("edge_in_dim", 2)),
    }
    signature["add_v_edge_features"] = bool(args.get("add_v_edge_features", signature["edge_in_dim"] > 2))
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
    args.setdefault("feature_run_id", None)

    legacy_context = infer_legacy_model_context(model_id)
    target_family = metadata.get("target_family")
    if target_family is None and task.startswith("outcome_"):
        target_family = infer_target_family(
            bool(args.get("use_xg", False)),
            bool(args.get("use_xt", False)),
            bool(args.get("use_goal_distance", False)),
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
        "intended_receiver_mode": intended_receiver_mode,
        "target_family": target_family,
        "return_type": return_type,
        "model_name": str(args.get("model", metadata.get("model", "unknown"))),
        "feature_signature": feature_signature,
        "graph_schema": {
            "edge_in_dim": feature_signature["edge_in_dim"],
            "add_v_edge_features": feature_signature["add_v_edge_features"],
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


def encode_onehot(labels, classes=None):
    if classes:
        classes = [x for x in range(classes)]
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_splits(
    lineup_path="data/ajax/lineup/line_up.parquet",
    feature_dir: str = "data/ajax/features/action_graphs",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del lineup_path
    return load_model_splits(feature_dir)


def load_model(model_id="pass_intent/01", device="cuda") -> GNN:
    if model_id is None:
        return None

    else:
        model_path = get_model_path(model_id)
        with open(model_path / "args.json", "r", encoding="utf-8") as f:
            args = json.load(f)
        args.setdefault("edge_in_dim", 2)
        args.setdefault("add_v_edge_features", bool(args["edge_in_dim"] > 2))
        args.setdefault("feature_run_id", None)
        args.setdefault("model_id", str(model_id))

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
    explicit_model_ids: dict[str, str | None] | None = None,
    include_pass_intent: bool = False,
    include_success_intent: bool = False,
) -> dict[str, str]:
    explicit_model_ids = explicit_model_ids or {}
    target_family = infer_target_family(
        use_xg=bool(use_xg),
        use_xt=bool(use_xt),
        use_goal_distance=bool(use_goal_distance),
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
    use_v_edge_features: bool,
) -> dict[str, int | bool]:
    feature_edge_dim = int(feature_schema.get("edge_in_dim", 0))
    required_edge_dim = 4 if use_v_edge_features else 2
    if feature_edge_dim < required_edge_dim:
        raise ValueError(
            "Selected feature artifacts do not provide the requested edge-feature schema: "
            f"features={feature_schema}, required_edge_dim={required_edge_dim}."
        )
    return {
        "edge_in_dim": required_edge_dim,
        "add_v_edge_features": bool(use_v_edge_features),
    }


def get_model_records(model_ids: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {task: get_model_record(model_id) for task, model_id in model_ids.items()}


def validate_model_record_consistency(
    model_records: dict[str, dict[str, Any]],
    require_feature_run_id: bool = True,
    require_intended_receiver_mode: bool = True,
    require_return_type: bool = True,
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
    shared["feature_run_id"] = next(iter(feature_run_ids)) if feature_run_ids else None

    intended_modes = {record.get("intended_receiver_mode") for record in model_records.values()}
    intended_modes.discard(None)
    intended_modes.discard("unknown")
    if require_intended_receiver_mode and len(intended_modes) != 1:
        details = ", ".join(f"{task}={record.get('intended_receiver_mode')}" for task, record in model_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on intended_receiver_mode: "
            f"{details}."
        )
    shared["intended_receiver_mode"] = next(iter(intended_modes)) if intended_modes else None

    return_types = {record.get("return_type") for record in model_records.values()}
    return_types.discard(None)
    if require_return_type and len(return_types) != 1:
        details = ", ".join(f"{task}={record.get('return_type')}" for task, record in model_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on return_type: "
            f"{details}."
        )
    shared["return_type"] = next(iter(return_types)) if return_types else None

    graph_schemas = {json.dumps(record["graph_schema"], sort_keys=True) for record in model_records.values()}
    if len(graph_schemas) != 1:
        details = ", ".join(f"{task}={record.get('graph_schema')}" for task, record in model_records.items())
        raise ValueError(
            "Selected model checkpoints do not agree on graph schema: "
            f"{details}."
        )
    shared["graph_schema"] = next(iter(model_records.values()))["graph_schema"]

    outcome_records = {task: model_records[task] for task in outcome_tasks if task in model_records}
    target_families = {record.get("target_family") for record in outcome_records.values()}
    target_families.discard(None)
    if len(target_families) > 1:
        details = ", ".join(f"{task}={record.get('target_family')}" for task, record in outcome_records.items())
        raise ValueError(
            "Selected outcome checkpoints do not agree on target_family: "
            f"{details}."
        )
    shared["target_family"] = next(iter(target_families)) if target_families else None

    return shared


def resolve_model_selection(
    required_tasks: list[str],
    bundle_id: str | None = None,
    explicit_model_ids: dict[str, str | None] | None = None,
    require_feature_run_id: bool = True,
    require_intended_receiver_mode: bool = True,
    require_return_type: bool = True,
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
    )
    shared["model_records"] = model_records

    if bundle is not None:
        if bundle.get("feature_run_id") and shared.get("feature_run_id") and bundle["feature_run_id"] != shared["feature_run_id"]:
            raise ValueError(
                f"Bundle {bundle_id!r} feature_run_id={bundle['feature_run_id']!r} does not match the selected model ids "
                f"(feature_run_id={shared['feature_run_id']!r})."
            )
        if bundle.get("intended_receiver_mode") and shared.get("intended_receiver_mode") and bundle["intended_receiver_mode"] != shared["intended_receiver_mode"]:
            raise ValueError(
                f"Bundle {bundle_id!r} intended_receiver_mode={bundle['intended_receiver_mode']!r} does not match the "
                f"selected model ids (intended_receiver_mode={shared['intended_receiver_mode']!r})."
            )
        if bundle.get("return_type") and shared.get("return_type") and bundle["return_type"] != shared["return_type"]:
            raise ValueError(
                f"Bundle {bundle_id!r} return_type={bundle['return_type']!r} does not match the selected model ids "
                f"(return_type={shared['return_type']!r})."
            )
        if bundle.get("target_family") and shared.get("target_family") and bundle["target_family"] != shared["target_family"]:
            raise ValueError(
                f"Bundle {bundle_id!r} target_family={bundle['target_family']!r} does not match the selected model ids "
                f"(target_family={shared['target_family']!r})."
            )

        shared["feature_run_id"] = shared.get("feature_run_id") or bundle.get("feature_run_id")
        shared["intended_receiver_mode"] = shared.get("intended_receiver_mode") or bundle.get("intended_receiver_mode")
        shared["return_type"] = shared.get("return_type") or bundle.get("return_type")
        shared["target_family"] = shared.get("target_family") or bundle.get("target_family")

    return resolved_model_ids, shared, bundle


def get_model_graph_schema(model: GNN | None) -> dict[str, int | bool] | None:
    if model is None or not hasattr(model, "args"):
        return None
    edge_in_dim = int(model.args.get("edge_in_dim", 2))
    return {
        "edge_in_dim": edge_in_dim,
        "add_v_edge_features": bool(model.args.get("add_v_edge_features", edge_in_dim > 2)),
    }


def validate_model_graph_schemas(models: dict[str, GNN | None]) -> dict[str, int | bool]:
    schemas = {
        name: schema
        for name, schema in ((name, get_model_graph_schema(model)) for name, model in models.items())
        if schema is not None
    }
    if not schemas:
        return {"edge_in_dim": 2, "add_v_edge_features": False}

    reference_name, reference_schema = next(iter(schemas.items()))
    mismatches = []
    for name, schema in list(schemas.items())[1:]:
        if schema != reference_schema:
            mismatches.append(f"{name}={schema}")

    if mismatches:
        raise ValueError(
            "Loaded model checkpoints use incompatible graph edge schemas. "
            f"Reference {reference_name}={reference_schema}; mismatches: {', '.join(mismatches)}."
        )
    return reference_schema


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
            edge_in_dim = int(first_graph.edge_attr.shape[1]) if getattr(first_graph, "edge_attr", None) is not None else 0
            return {
                "edge_in_dim": edge_in_dim,
                "add_v_edge_features": bool(edge_in_dim > 2),
            }
        except Exception:
            continue
    raise FileNotFoundError(f"Could not infer graph schema from feature directory {feature_dir}.")


def estimate_propensity(dataset, model_id="pass_intent/00", device="cuda", min_clip=0.01) -> torch.Tensor:
    model = load_model(model_id, device)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True)
    likelihoods = []

    for batch_graphs, batch_labels, _ in tqdm(loader):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            batch_graphs.x = batch_graphs.x[:, : model.args["node_in_dim"]]
            out: torch.Tensor = model(batch_graphs)
            for graph_index in range(batch_graphs.num_graphs):
                logits = out[(batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)]
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


def calc_binary_metrics(y, y_hat, threshold=0.5):
    y_pred = y_hat > threshold
    precision = precision_score(y, y_pred) if np.sum(y_pred) > 0 else 0
    recall = recall_score(y, y_pred) if np.sum(y) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y, y_pred) if precision > 0 and recall > 0 else 0,
        "roc_auc": roc_auc_score(y, y_hat) if np.sum(y) > 0 else 0.5,
        "brier": brier_score_loss(y, y_hat),
        "log_loss": log_loss(y, y_hat) if np.sum(y) > 0 else np.nan,
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def validate_target_flags(args) -> None:
    enabled_flags = sum(
        int(
            bool(getattr(args, name, False))
        )
        for name in ["use_xg", "use_xt", "use_goal_distance"]
    )
    if enabled_flags > 1:
        raise ValueError("--use_xg, --use_xt, and --use_goal_distance are mutually exclusive.")


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
    if getattr(args, "use_xg", False):
        return get_label_slice(batch_labels, "scores_xg"), get_label_slice(batch_labels, "concedes_xg")
    return get_label_slice(batch_labels, "scores"), get_label_slice(batch_labels, "concedes")


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

    poss_mask = feat[:, 13] == 1
    team_mask = (feat[:, 0] == 1) & (feat[:, 2] == 0)
    oppo_mask = (feat[:, 0] == 0) & (feat[:, 2] == 0)

    dest_feat = []

    for graph_idx in range(B):
        graph_mask = batch == graph_idx
        poss_xy = feat[graph_mask & poss_mask, 3:5]  # [1, 2]
        team_xy = feat[graph_mask & team_mask, 3:5]  # [team, 2]
        oppo_xy = feat[graph_mask & oppo_mask, 3:5]  # [oppo, 2]

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
    model: nn.DataParallel,
    loader: DataLoader,
    optimizer: torch.optim.Adam = None,
    device: str = "cuda",
    pos_weight: float = 1.0,
    train: bool = False,
):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)
    pos_weight = torch.tensor(pos_weight)

    if args.gnn_task in ["node_binary", "graph_binary"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "f1": 0, "roc_auc": 0, "brier": 0}
    elif args.gnn_task in ["node_selection", "graph_multiclass"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "accuracy": 0, "mrr": 0}
    elif args.gnn_task in ["node_regression", "graph_regression"]:
        metrics = {"count": 0, "mse_loss": 0, "l1_loss": 0}

    for batch_index, (batch_graphs, batch_labels, batch_ipw) in enumerate(loader):
        batch_graphs: Batch = batch_graphs.to(device)
        batch_ipw: torch.Tensor = batch_ipw.to(device)
        index_range = torch.unique(batch_graphs.batch)

        metrics["count"] += batch_graphs.num_graphs
        outcome_scoring, outcome_conceding = get_outcome_targets(batch_labels := batch_labels.to(device), args)

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
                out: torch.Tensor = model.module.forward_grid(batch_graphs, grid_features)
            else:
                with torch.no_grad():
                    out: torch.Tensor = model.module.forward_grid(batch_graphs, grid_features)

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
                    pred_i = out[(batch == graph_index) & (batch_graphs.x[:, 0] == 1)]  # [N_i]
                    target_i = target[graph_index]

                elif args.task == "failure_receiver":
                    # Only take opponent nodes
                    if args.include_out:
                        ball_out_mask = torch.ones(batch_graphs.num_graphs).bool().to(device)
                        failure_mask = torch.cat([batch_graphs.x[:, 0] == 0, ball_out_mask])
                    else:
                        failure_mask = batch_graphs.x[:, 0] == 0
                    pred_i = out[(batch == graph_index) & failure_mask]
                    n_teammates = ((batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)).sum()
                    target_i = target[graph_index] - n_teammates

                else:  # pass_receiver, dest_receiver
                    pred_i = out[batch == graph_index]
                    target_i = target[graph_index]

                pred_loss += loss_fn(pred_i.unsqueeze(0), target_i.unsqueeze(0))
                accuracy += (pred_i.argmax() == target_i).float()

                rank = (pred_i.argsort(descending=True) == target_i).nonzero(as_tuple=True)[0].item() + 1
                metrics["mrr"] += 1.0 / rank

            pred_loss /= index_range.shape[0]
            metrics["accuracy"] += accuracy.item()

        elif args.gnn_task == "node_binary":  # {pass/action}_success, outcome_{scoring/conceding}, intent_return
            intent = batch_labels[:, 5].clone().long()

            if args.task in ["pass_success", "action_success"]:
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index]])
                pred = torch.stack(pred)

                target = get_label_slice(batch_labels, "success")
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = target.cpu().detach().numpy()
                threshold = 0.5 if args.task.endswith("success") else 0.1
                batch_metrics = calc_binary_metrics(y, y_hat, threshold)

            elif args.task in ["outcome_scoring", "outcome_conceding"]:
                outcome = get_label_slice(batch_labels, "success").clone().long()
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index], outcome[graph_index]])
                pred = torch.stack(pred)

                target = outcome_scoring if args.task.endswith("scoring") else outcome_conceding
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = get_label_slice(batch_labels, "scores") if args.task.endswith("scoring") else get_label_slice(batch_labels, "concedes")
                y = y.cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

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
                y = get_label_slice(batch_labels, "scores").cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

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
                batch_metrics = calc_binary_metrics(y, y_hat, 0.5)

            elif args.task.split("_")[0] in ["overall", "dest"]:  # {overall/dest}_{scoring/conceding}
                if args.task.startswith("dest"):
                    outcome = get_label_slice(batch_labels, "success").clone().long()
                    pred = out[tuple([list(range(batch_graphs.num_graphs)), outcome])]
                else:
                    pred = out

                target = outcome_scoring if args.task.endswith("scoring") else outcome_conceding
                pred_loss = nn.BCEWithLogitsLoss()(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = get_label_slice(batch_labels, "scores") if args.task.endswith("scoring") else get_label_slice(batch_labels, "concedes")
                y = y.cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

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
            nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
            optimizer.step()

        if train and batch_index % args.print_freq == 0:
            interim_metrics = dict()
            for key, value in metrics.items():
                if key == "count":
                    continue
                interim_metrics[key] = value / metrics["count"]
            print(f"[{batch_index:>{len(str(n_batches))}d}/{n_batches}]  {get_losses_str(interim_metrics)}")

    for key, value in metrics.items():
        if key == "count":
            continue
        metrics[key] = value / metrics["count"]

    return metrics
