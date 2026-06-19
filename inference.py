import re
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from xgboost import XGBClassifier

from datatools import config
from datatools.config import FIELD_SIZE, TASK_CONFIG
from datatools.match import Match
from datatools.utils import (
    filter_features_and_labels,
    find_active_players,
    player_sort_key,
)
from models.gnn import GNN
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    PHYSICAL_XPASS_DISTANCE_ATTR,
    PHYSICAL_XPASS_PROB_ATTR,
    attach_physical_xpass_cached_online_to_graphs,
    attach_physical_xpass_online_to_graphs,
    attach_physical_xpass_read_only_to_graphs,
    attach_physical_xpass_to_graphs,
    blend_physical_xpass_predictions,
    inference_uses_physical_xpass,
    load_physical_xpass_match,
    model_uses_physical_xpass,
    physical_xpass_floor,
    physical_xpass_inference_lookup_config,
    physical_xpass_metric,
    physical_xpass_metric_columns,
    physical_xpass_speed_aggregation,
    physical_xpass_source,
    physical_xpass_teammate_policy,
    requires_physical_xpass_for_inference,
    validate_physical_xpass_cache_metadata,
)
from project_config import get_physical_xpass_dir, get_runtime_physical_xpass_dir, get_success_intent_label_dir

PASS_ONLY_INTENT_TASKS = {"pass_intent", "pass_intent_oppo_agn", "success_intent"}


class PhysicalXPassNoUsableRowsError(ValueError):
    pass


def _exclude_possessor_from_pass_only_intent(
    task: str,
    player_indices: list[str],
    probs: np.ndarray,
    possessor_object_id: str,
) -> tuple[list[str], np.ndarray]:
    if task not in PASS_ONLY_INTENT_TASKS or possessor_object_id not in player_indices:
        return list(player_indices), probs

    keep_mask = np.array([player_id != possessor_object_id for player_id in player_indices], dtype=bool)
    filtered_players = [player_id for player_id in player_indices if player_id != possessor_object_id]
    filtered_probs = np.asarray(probs, dtype=float)[keep_mask]

    if filtered_probs.size > 0:
        total = float(filtered_probs.sum())
        if total > 0:
            filtered_probs = filtered_probs / total

    return filtered_players, filtered_probs


def _renormalize_probabilities(probs: np.ndarray) -> np.ndarray:
    total = float(np.asarray(probs, dtype=float).sum())
    if total > 0:
        return np.asarray(probs, dtype=float) / total
    return np.asarray(probs, dtype=float)


def _uses_offside_rule_mask(model: GNN, node_dim: int) -> bool:
    return (
        bool(model.args.get("offside_aware", True))
        and int(model.args.get("node_in_dim", node_dim) or node_dim) in config.NODE_FEATURE_OFFSIDE_DIMS
        and node_dim in config.NODE_FEATURE_OFFSIDE_DIMS
    )


def resolve_graph_feature_dir(model: GNN, post_action: bool = False) -> str:
    feature_dir = model.args.get("feature_dir", "data/features/action_graphs")
    feature_path = Path(feature_dir)
    feature_name = feature_path.name

    if not post_action:
        return str(feature_path)

    if feature_name == "action_graphs_temporal":
        return str(feature_path.with_name("post_action_graphs_temporal"))
    if feature_name == "action_graphs":
        return str(feature_path.with_name("post_action_graphs"))
    return str(feature_path)


def resolve_match_id(match: Match) -> str:
    match_id = getattr(match, "match_id", None)
    if match_id is not None:
        return str(match_id)

    lineup = getattr(match, "lineup", None)
    if lineup is not None and not lineup.empty and "stats_perform_match_id" in lineup.columns:
        return str(lineup["stats_perform_match_id"].iloc[0])

    events = getattr(match, "events", None)
    if events is not None and "game_id" in events.columns and not events.empty:
        return str(events["game_id"].iloc[0])

    raise ValueError("Could not determine match_id to resolve graph feature files.")


def _absolute_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _runtime_feature_root(match: Match) -> Path | None:
    feature_root = getattr(match, "runtime_feature_root", None)
    if feature_root is None:
        feature_root = getattr(match, "feature_root", None)
    return Path(feature_root) if feature_root is not None else None


def _allows_online_physical_xpass(match: Match) -> bool:
    if _runtime_feature_root(match) is not None:
        return False
    match_type = type(match)
    return (
        match_type.__module__ in {"datatools.benchmark", "datatools.hawkeye", "datatools.skillcorner"}
        and match_type.__name__ in {"BenchmarkState", "HawkeyeSituation", "SkillcornerPossession"}
    )


def _record_online_physical_xpass(
    match: Match,
    model: GNN,
    *,
    source: str,
    speed_aggregation: str,
    graph_count: int,
    extra_stats: dict[str, object] | None = None,
) -> None:
    stats = getattr(match, "physical_xpass_runtime_stats", None)
    if stats is None:
        stats = {}
        setattr(match, "physical_xpass_runtime_stats", stats)
    task = str(model.args.get("task", "unknown"))
    task_stats = stats.setdefault(
        task,
        {
            "source": source,
            "speed_aggregation": speed_aggregation,
            "model_id": model.args.get("model_id"),
            "cache_dir": None,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_written": 0,
            "copied_from_reuse": 0,
            "pass_distance_filled": 0,
            "hash_mismatch_recomputed": 0,
            "online_graphs": 0,
            "cache_disabled": False,
        },
    )
    task_stats["online_graphs"] = int(task_stats.get("online_graphs", 0)) + int(graph_count)
    if extra_stats:
        for key in [
            "cache_hits",
            "cache_misses",
            "cache_written",
            "copied_from_reuse",
            "pass_distance_filled",
            "hash_mismatch_recomputed",
            "online_graphs",
        ]:
            if key in extra_stats and key != "online_graphs":
                task_stats[key] = int(task_stats.get(key, 0)) + int(extra_stats[key])
        for key in ["cache_dir", "cache_disabled"]:
            if key in extra_stats:
                task_stats[key] = extra_stats[key]


def _runtime_physical_xpass_source_name(match: Match) -> str | None:
    match_type = type(match)
    if match_type.__module__ == "datatools.hawkeye" and match_type.__name__ == "HawkeyeSituation":
        return "hawkeye"
    if match_type.__module__ == "datatools.benchmark" and match_type.__name__ == "BenchmarkState":
        return "benchmark"
    if match_type.__module__ == "datatools.skillcorner" and match_type.__name__ == "SkillcornerPossession":
        return "skillcorner"
    return None


def attach_physical_xpass_for_inference(
    match: Match,
    graphs: list[Data],
    labels: torch.Tensor,
    model: GNN,
) -> list[Data]:
    eps = float(model.args.get("physical_eps", 1e-4))
    floor = physical_xpass_floor(model.args)
    feature_root = _runtime_feature_root(match)
    physical_cache_dir = model.args.get("physical_cache_dir")
    if not physical_cache_dir:
        if inference_uses_physical_xpass(model.args):
            runtime_source_name = _runtime_physical_xpass_source_name(match)
            physical_cache_dir = str(get_runtime_physical_xpass_dir(runtime_source_name or "sportec"))
        elif feature_root is None:
            runtime_source_name = _runtime_physical_xpass_source_name(match)
            if runtime_source_name is not None:
                physical_cache_dir = str(get_runtime_physical_xpass_dir(runtime_source_name))
            else:
                feature_root = Path(model.args.get("feature_dir", ".")).resolve().parent
                physical_cache_dir = str(get_physical_xpass_dir(feature_root))
        else:
            physical_cache_dir = str(get_physical_xpass_dir(feature_root))

    use_inference_blend = inference_uses_physical_xpass(model.args)
    lookup_config = physical_xpass_inference_lookup_config(model.args, cache_dir=physical_cache_dir)
    if use_inference_blend:
        setattr(model, "physical_xpass_lookup_policy", "dataset_event_frame_player_only")
    else:
        validate_physical_xpass_cache_metadata(
            physical_cache_dir,
            expected_source=physical_xpass_source(model.args),
            expected_speed_aggregation=physical_xpass_speed_aggregation(model.args),
        )
    return attach_physical_xpass_read_only_to_graphs(
        graphs,
        labels,
        cache_dir=physical_cache_dir,
        match_id=resolve_match_id(match),
        eps=eps,
        floor=floor,
        require_observed_target=False,
        metric=lookup_config["metric"] if use_inference_blend else physical_xpass_metric(model.args),
        missing_player_value=None if use_inference_blend else 0.5,
    )


def _physical_xpass_cache_dir_for_inference(match: Match, model: GNN) -> str:
    feature_root = _runtime_feature_root(match)
    physical_cache_dir = model.args.get("physical_cache_dir")
    if physical_cache_dir:
        return str(physical_cache_dir)

    if inference_uses_physical_xpass(model.args):
        runtime_source_name = _runtime_physical_xpass_source_name(match)
        return str(get_runtime_physical_xpass_dir(runtime_source_name or "sportec"))

    if feature_root is None:
        runtime_source_name = _runtime_physical_xpass_source_name(match)
        if runtime_source_name is not None:
            return str(get_runtime_physical_xpass_dir(runtime_source_name))
        feature_root = Path(model.args.get("feature_dir", ".")).resolve().parent

    return str(get_physical_xpass_dir(feature_root))


def _record_physical_xpass_skipped_actions(
    match: Match,
    model: GNN,
    *,
    cache_dir: str,
    requested_count: int,
    kept_count: int,
    skipped_action_indexes: list[int],
    reason: str | None = None,
) -> None:
    if not skipped_action_indexes:
        return

    stats = getattr(match, "physical_xpass_skipped_actions", None)
    if stats is None:
        stats = {}
        setattr(match, "physical_xpass_skipped_actions", stats)

    task = str(model.args.get("task", "unknown"))
    stats[task] = {
        "cache_dir": str(cache_dir),
        "requested_count": int(requested_count),
        "kept_count": int(kept_count),
        "skipped_count": int(len(skipped_action_indexes)),
        "sample_action_indexes": [int(action_index) for action_index in skipped_action_indexes[:10]],
    }
    if reason:
        stats[task]["reason"] = str(reason)


def _physical_xpass_row_skip_reason(
    row: pd.Series,
    graph: Data,
    *,
    match_id: str,
    action_index: int,
    metric: str | None = None,
) -> str | None:
    if metric is not None:
        node_ids = [str(node_id) for node_id in getattr(graph, "node_ids", [])]
        candidate_columns = [column for node_id in node_ids for column in physical_xpass_metric_columns(node_id, metric)]
        has_metric_column = any(column in row.index for column in candidate_columns)
        if not has_metric_column:
            return f"missing_{metric}_columns"
        has_finite_metric_value = False
        for column in candidate_columns:
            if column not in row.index:
                continue
            try:
                if np.isfinite(float(row.get(column))):
                    has_finite_metric_value = True
                    break
            except (TypeError, ValueError):
                continue
        if not has_finite_metric_value:
            return f"no_finite_{metric}_values"
    return None


def filter_missing_physical_xpass_rows_for_inference(
    match: Match,
    graphs: list[Data],
    labels: torch.Tensor,
    model: GNN,
) -> tuple[list[Data], torch.Tensor]:
    if not requires_physical_xpass_for_inference(model.args):
        return graphs, labels

    use_inference_blend = inference_uses_physical_xpass(model.args)
    cache_dir = _physical_xpass_cache_dir_for_inference(match, model)
    lookup_config = physical_xpass_inference_lookup_config(model.args, cache_dir=cache_dir)
    metric = lookup_config["metric"] if use_inference_blend else None
    setattr(match, "physical_xpass_hash_policy", PHYSICAL_XPASS_INFERENCE_HASH_POLICY)
    if use_inference_blend:
        setattr(match, "physical_xpass_lookup_policy", "dataset_event_frame_player_only")
    match_id = resolve_match_id(match)
    all_action_indexes = [int(label[config.LABEL_INDEX["action_index"]].item()) for label in labels]
    try:
        if not use_inference_blend:
            validate_physical_xpass_cache_metadata(
                cache_dir,
                expected_source=physical_xpass_source(model.args),
                expected_speed_aggregation=physical_xpass_speed_aggregation(model.args),
            )
        physical_rows = load_physical_xpass_match(cache_dir, match_id)
    except (FileNotFoundError, ValueError) as exc:
        _record_physical_xpass_skipped_actions(
            match,
            model,
            cache_dir=cache_dir,
            requested_count=len(graphs),
            kept_count=0,
            skipped_action_indexes=all_action_indexes,
            reason=f"{type(exc).__name__}: {exc}",
        )
        raise PhysicalXPassNoUsableRowsError(
            "No usable graph/label pairs remain because the physical xPass cache is missing or invalid."
        ) from exc
    available_action_indexes = set(physical_rows.index.astype(int).tolist())

    kept_graphs: list[Data] = []
    kept_labels: list[torch.Tensor] = []
    skipped_action_indexes: list[int] = []
    skip_reasons: dict[str, int] = {}
    for graph, label in zip(graphs, labels):
        action_index = int(label[config.LABEL_INDEX["action_index"]].item())
        if action_index not in available_action_indexes:
            skipped_action_indexes.append(action_index)
            skip_reasons["missing_row"] = skip_reasons.get("missing_row", 0) + 1
            continue
        reason = _physical_xpass_row_skip_reason(
            physical_rows.loc[action_index],
            graph,
            match_id=match_id,
            action_index=action_index,
            metric=metric,
        )
        if reason is not None:
            skipped_action_indexes.append(action_index)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        kept_graphs.append(graph)
        kept_labels.append(label)

    _record_physical_xpass_skipped_actions(
        match,
        model,
        cache_dir=cache_dir,
        requested_count=len(graphs),
        kept_count=len(kept_graphs),
        skipped_action_indexes=skipped_action_indexes,
        reason=", ".join(f"{key}={value}" for key, value in sorted(skip_reasons.items())) if skip_reasons else None,
    )
    if not kept_labels:
        raise PhysicalXPassNoUsableRowsError(
            "No usable graph/label pairs remain after filtering missing physical xPass sidecar rows."
        )

    return kept_graphs, torch.stack(kept_labels, axis=0)


def resolve_runtime_graph_feature_dir(match: Match, model: GNN, post_action: bool = False) -> Path:
    checkpoint_feature_path = Path(resolve_graph_feature_dir(model, post_action))
    feature_root = _runtime_feature_root(match)
    if feature_root is not None:
        return _absolute_path(Path(feature_root) / checkpoint_feature_path.name)
    return _absolute_path(checkpoint_feature_path)


def load_success_intent_labels(match: Match, feature_root: str | Path | None = None) -> torch.Tensor:
    resolved_feature_root = Path(feature_root) if feature_root is not None else _runtime_feature_root(match)
    label_dir = get_success_intent_label_dir(root=resolved_feature_root)
    label_path = label_dir / f"{resolve_match_id(match)}.pt"
    if not label_path.exists():
        raise FileNotFoundError(f"Success-intent labels not found at {label_path}.")
    labels = torch.load(label_path, weights_only=False)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"Success-intent labels at {label_path} have invalid shape.")
    if labels.numel() == 0:
        raise ValueError(f"Success-intent labels at {label_path} are empty.")
    return labels


def _get_graph_caches(match: Match) -> tuple[dict[str, object], dict[str, np.ndarray | None]]:
    if not hasattr(match, "graph_features_by_dir"):
        match.graph_features_by_dir = {}
    if not hasattr(match, "graph_feature_action_indices_by_dir"):
        match.graph_feature_action_indices_by_dir = {}
    return match.graph_features_by_dir, match.graph_feature_action_indices_by_dir


def _load_feature_action_indices(match: Match, feature_path: Path, model: GNN) -> np.ndarray | None:
    if model.args.get("task") != "success_intent":
        return None

    labels = load_success_intent_labels(match, feature_path.parent)
    return labels[:, 0].detach().cpu().numpy().astype(int)


def resolve_match_graphs(match: Match, model: GNN, post_action: bool = False) -> tuple[list[Data], np.ndarray | None]:
    feature_path = resolve_runtime_graph_feature_dir(match, model, post_action)
    cache_key = str(feature_path)
    graph_cache, action_index_cache = _get_graph_caches(match)

    if cache_key in graph_cache:
        return graph_cache[cache_key], action_index_cache.get(cache_key)

    feature_name = feature_path.name
    if not post_action and feature_name == "action_graphs" and getattr(match, "graph_features_0", None) is not None:
        graph_cache[cache_key] = match.graph_features_0
        action_index_cache[cache_key] = None
        return match.graph_features_0, None

    if post_action and feature_name == "post_action_graphs" and getattr(match, "graph_features_1", None) is not None:
        graph_cache[cache_key] = match.graph_features_1
        action_index_cache[cache_key] = None
        return match.graph_features_1, None

    match_id = resolve_match_id(match)
    graph_path = feature_path / f"{match_id}.pt"
    if graph_path.exists():
        graphs = torch.load(graph_path, weights_only=False)
        feature_action_indices = _load_feature_action_indices(match, feature_path, model)
        graph_cache[cache_key] = graphs
        action_index_cache[cache_key] = feature_action_indices
        return graphs, feature_action_indices

    if post_action and feature_name == "post_action_graphs_temporal":
        fallback_path = feature_path.with_name("post_action_graphs")
        fallback_key = str(fallback_path)
        if fallback_key in graph_cache:
            return graph_cache[fallback_key], action_index_cache.get(fallback_key)
        if getattr(match, "graph_features_1", None) is not None:
            graph_cache[fallback_key] = match.graph_features_1
            action_index_cache[fallback_key] = None
            return match.graph_features_1, None

    raise FileNotFoundError(f"Graph feature file not found at {graph_path}")


def inference_boost(
    match: Match,
    model: Union[XGBClassifier, CatBoostClassifier],
    post_action: bool = False,
    pad_own_half: bool = True,  # Zero-pad xG values for events occurring in the team's own half
    event_indices: pd.Index = None,
) -> pd.Series:
    features = match.tabular_features_0 if not post_action else match.tabular_features_1

    event_indices = event_indices if event_indices is not None else match.actions.index
    mask = match.actions.index.isin(event_indices)
    features = features.numpy()[mask, :20]

    probs = model.predict_proba(features)[:, 1]
    if pad_own_half:
        own_half_mask = features[:, 2] < FIELD_SIZE[0] / 2
        probs[own_half_mask] = 0.0

    return pd.Series(probs, index=event_indices, dtype=float)


def inference_gnn(
    match: Match,
    model: GNN,
    device: str = "cuda",
    post_action: bool = False,
    event_indices: pd.Index = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gnn_task = TASK_CONFIG.at[model.args["task"], "gnn_task"]
    include_goals = TASK_CONFIG.at[model.args["task"], "include_goals"]
    out_filter = TASK_CONFIG.at[model.args["task"], "out_filter"]
    match_graphs, feature_action_indices = resolve_match_graphs(match, model, post_action)
    graphs, labels = filter_features_and_labels(
        match_graphs,
        match.labels,
        model.args,
        event_indices,
        feature_action_indices=feature_action_indices,
    )
    if requires_physical_xpass_for_inference(model.args):
        graphs, labels = filter_missing_physical_xpass_rows_for_inference(match, graphs, labels, model)
    if requires_physical_xpass_for_inference(model.args):
        graphs = attach_physical_xpass_for_inference(match, graphs, labels, model)

    graphs = Batch.from_data_list(graphs).to(device)
    graphs.x = graphs.x[:, : model.args["node_in_dim"]]
    use_offside_rule_mask = model.args["task"] == "pass_success" and _uses_offside_rule_mask(model, int(graphs.x.shape[1]))
    offside_node_mask = (
        graphs.x[:, -1].bool()
        if use_offside_rule_mask
        else torch.zeros(graphs.x.shape[0], dtype=torch.bool, device=graphs.x.device)
    )

    two_case_tasks = ["outcome_scoring", "outcome_conceding", "outcome_return", "intent_return"]
    if model.args["task"] in two_case_tasks:
        probs_0 = []
        probs_1 = []
    else:
        probs = []

    with torch.no_grad():
        if model.args["task"] == "shot_blocking":
            out = torch.sigmoid(model(graphs)).cpu().detach().numpy()  # [B,]
            event_indices = labels[:, 0].cpu().detach().numpy().astype(int)
            return pd.Series(out, index=event_indices), None

        else:  # model.args["task"].startswith("node")
            batch = graphs.batch
            out = model(graphs)
            offside_out_mask = offside_node_mask
            physical_xpass_out = getattr(graphs, PHYSICAL_XPASS_PROB_ATTR, None)
            physical_distance_out = getattr(graphs, PHYSICAL_XPASS_DISTANCE_ATTR, None)

            if TASK_CONFIG.at[model.args["task"], "out_filter"] == "teammates":
                # Select components corresponding to teammates
                teammate_mask = graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
                batch = batch[teammate_mask]
                out = out[teammate_mask]  # [N',]
                offside_out_mask = offside_out_mask[teammate_mask]
                if physical_xpass_out is not None:
                    physical_xpass_out = physical_xpass_out[teammate_mask]
                if physical_distance_out is not None:
                    physical_distance_out = physical_distance_out[teammate_mask]

            if "receiver" in model.args["task"] and model.args["include_out"]:
                batch = torch.cat([batch, torch.unique(graphs.batch)])
                offside_out_mask = torch.cat(
                    [
                        offside_out_mask,
                        torch.zeros(graphs.num_graphs, dtype=torch.bool, device=offside_out_mask.device),
                    ]
                )
                if physical_xpass_out is not None:
                    physical_xpass_out = torch.cat(
                        [
                            physical_xpass_out,
                            torch.full((graphs.num_graphs,), np.nan, dtype=physical_xpass_out.dtype, device=physical_xpass_out.device),
                        ]
                    )
                if physical_distance_out is not None:
                    physical_distance_out = torch.cat(
                        [
                            physical_distance_out,
                            torch.full((graphs.num_graphs,), np.nan, dtype=physical_distance_out.dtype, device=physical_distance_out.device),
                        ]
                    )

    players = set()

    for i in tqdm(range(graphs.num_graphs), desc=model.args["task"]):
        event_index = int(labels[i, 0].item())

        if post_action:
            frame = int(match.actions.at[event_index, "end_frame_id"])
            team = match.actions.at[event_index, "end_player_id"][:4]
            possessor_object_id = match.actions.at[event_index, "end_player_id"]
        else:
            frame = int(match.actions.at[event_index, "frame_id"])
            team = match.actions.at[event_index, "object_id"][:4]
            possessor_object_id = match.actions.at[event_index, "object_id"]

        active_players = find_active_players(match.tracking, frame, team, include_goals=include_goals)

        if gnn_task == "node_selection":
            probs_i = torch.softmax(out[batch == i], dim=0).cpu().detach().numpy()
        elif gnn_task in ["node_binary", "graph_binary"]:
            probs_i = torch.sigmoid(out[batch == i]).cpu().detach().numpy()
        elif gnn_task == "node_regression":
            probs_i = torch.sigmoid(out[batch == i]).cpu().detach().numpy() * 2 - 1
        offside_i = offside_out_mask[batch == i].cpu().detach().numpy().astype(bool)

        if out_filter == "teammates":
            player_indices_i = list(active_players[0])
            if gnn_task == "node_selection":
                if model.args["task"] in PASS_ONLY_INTENT_TASKS and str(possessor_object_id) in player_indices_i:
                    keep_mask = np.array([player_id != str(possessor_object_id) for player_id in player_indices_i], dtype=bool)
                    player_indices_i = [player_id for player_id in player_indices_i if player_id != str(possessor_object_id)]
                    probs_i = _renormalize_probabilities(probs_i[keep_mask])
                    offside_i = offside_i[keep_mask]
        elif out_filter == "all":  # "receiver" in model.args["task"]
            player_indices_i = active_players[0] + active_players[1]
            if model.args["include_out"]:
                player_indices_i.append("out")

        players = players | set(player_indices_i)

        if model.args["task"] == "pass_success":
            probs_i = np.asarray(probs_i, dtype=float).copy()
            if probs_i.shape[0] == offside_i.shape[0]:
                probs_i[offside_i] = 0.0
            if inference_uses_physical_xpass(model.args):
                if physical_xpass_out is None or physical_distance_out is None:
                    raise ValueError("Inference physical xPass blending requires attached physical_xpass and pass-distance tensors.")
                xpass_i = physical_xpass_out[batch == i].cpu().detach().numpy().astype(float)
                distance_i = physical_distance_out[batch == i].cpu().detach().numpy().astype(float)
                if xpass_i.shape[0] != probs_i.shape[0] or distance_i.shape[0] != probs_i.shape[0]:
                    raise ValueError(
                        "Physical xPass blend tensors do not match pass-success probabilities: "
                        f"probs={probs_i.shape}, xpass={xpass_i.shape}, pass_distance={distance_i.shape}."
                    )
                finite_mask = np.isfinite(xpass_i) & np.isfinite(distance_i)
                missing_physical_mask = ~finite_mask
                blended_i = np.asarray(
                    blend_physical_xpass_predictions(
                        pass_success_model=probs_i[finite_mask],
                        xpass=xpass_i[finite_mask],
                        pass_distance=distance_i[finite_mask],
                    ),
                    dtype=float,
                )
                probs_i[finite_mask] = blended_i
                probs_i[missing_physical_mask] = np.nan

        if model.args["task"] in two_case_tasks:
            probs_i0 = dict(zip(player_indices_i, probs_i[:, 0].tolist()))
            probs_i1 = dict(zip(player_indices_i, probs_i[:, 1].tolist()))
            probs_0.append(dict(**probs_i0, **{"index": event_index}))
            probs_1.append(dict(**probs_i1, **{"index": event_index}))
        else:
            probs_i = dict(zip(player_indices_i, probs_i.tolist()))
            probs.append(dict(**probs_i, **{"index": event_index}))

    players = sorted(list(players), key=player_sort_key)

    if model.args["task"] in two_case_tasks:
        probs_0 = pd.DataFrame(probs_0).set_index("index")[players]
        probs_1 = pd.DataFrame(probs_1).set_index("index")[players]
        return probs_0, probs_1
    else:
        return pd.DataFrame(probs).set_index("index")[players], None


def inference_gnn_posterior(
    match: Match,
    model: GNN = None,
    device="cuda",
    event_indices: pd.Index = None,
    melt: bool = True,
) -> pd.DataFrame:
    match_graphs, feature_action_indices = resolve_match_graphs(match, model, post_action=False)
    graphs, labels = filter_features_and_labels(
        match_graphs,
        match.labels,
        model.args,
        event_indices,
        feature_action_indices=feature_action_indices,
    )
    include_goals = (graphs[0].x[:, config.NODE_FEATURE_IS_GOAL] == 1).any().item()
    posteriors = []

    for data_index in tqdm(range(len(graphs)), desc="failure_posterior"):
        graph_i = graphs[data_index].to(device)

        event_index = int(labels[data_index, 0].item())
        frame = int(match.actions.at[event_index, "frame_id"])
        team = match.actions.at[event_index, "object_id"][:4]
        active_players = find_active_players(match.tracking, frame, team, include_goals=include_goals)
        n_teammates = len(active_players[0])

        intended_graphs = []
        for intent_index in range(n_teammates):
            intent_onehot = torch.zeros(graph_i.x.shape[0]).to(device)
            intent_onehot[intent_index] = 1
            intended_nodes = torch.cat([graph_i.x, intent_onehot.unsqueeze(1)], -1)
            intended_graph = Data(x=intended_nodes, edge_index=graph_i.edge_index, edge_attr=graph_i.edge_attr)
            intended_graphs.append(intended_graph)
        intended_graphs = Batch.from_data_list(intended_graphs).to(device)

        with torch.no_grad():
            logits = model(intended_graphs)  # [12 * 24 + 12,] if include_out else [12 * 24,]

        if model.args["include_out"]:
            receive_logits = logits[:-n_teammates].reshape(n_teammates, -1)  # [12, 24]
            ballout_logits = logits[-n_teammates:].unsqueeze(1)  # [12, 1]
            logits = torch.cat([receive_logits, ballout_logits], 1)  # [12, 25]
            posteriors_i = torch.softmax(logits[:, n_teammates:], dim=1).cpu().detach().numpy()  # [12, 13]
            posteriors_i = pd.DataFrame(posteriors_i, index=active_players[0], columns=active_players[1] + ["out"])

        else:
            logits = logits.reshape(n_teammates, -1)  # [12, 24]
            posteriors_i = torch.softmax(logits[:, n_teammates:], dim=1).cpu().detach().numpy()  # [12, 12]
            posteriors_i = pd.DataFrame(posteriors_i, index=active_players[0], columns=active_players[1])

        posteriors_i["index"] = event_index
        posteriors_i.index.name = "option"
        posteriors.append(posteriors_i)

    valid_tracking = match.tracking.dropna(axis=1, how="all")
    home_players = [c[:-2] for c in valid_tracking.columns if re.match(r"home_\d+_x", c)]
    away_players = [c[:-2] for c in valid_tracking.columns if re.match(r"away_\d+_x", c)]

    if model.args["include_out"]:
        posteriors = pd.concat(posteriors)[["index"] + home_players + away_players + ["out"]].reset_index()
    else:
        posteriors = pd.concat(posteriors)[["index"] + home_players + away_players].reset_index()

    if melt:
        posteriors = posteriors.melt(id_vars=["index", "option"], var_name="defender", value_name="posterior")
        return posteriors.dropna(subset=["posterior"]).reset_index(drop=True).copy()
    else:
        return posteriors


def inference_gnn_grid(match: Match, model: GNN, device="cuda") -> Dict[int, torch.Tensor]:
    assert "dest" in model.args["task"]

    grid_size = (int(FIELD_SIZE[0]), int(FIELD_SIZE[1]))
    grid = np.mgrid[0 : grid_size[0], grid_size[1] - 1 : -1 : -1] + 0.5  # [2, 105, 68]
    grid = np.transpose(grid, (0, 2, 1)).reshape(2, -1)  # [2, 68 * 105]
    dest_tensor = torch.tensor(grid.T, dtype=torch.float32).to(device)  # [68 * 105, 2]
    n_cells = dest_tensor.shape[0]  # G = 68 * 105

    match_graphs, feature_action_indices = resolve_match_graphs(match, model, post_action=False)
    graphs, labels = filter_features_and_labels(
        match_graphs,
        match.labels,
        model.args,
        feature_action_indices=feature_action_indices,
    )
    receive_probs = dict()
    success_probs = dict()

    for data_index in tqdm(range(len(graphs)), desc="dest_receiver"):
        graph_i = graphs[data_index].to(device)

        with torch.no_grad():
            graph_i = Batch.from_data_list([graph_i]).to(device)
            node_emb, graph_emb = model.encoder(graph_i)  # [P, z], [1, z]

            node_feat_rep = graph_i.x.repeat(n_cells, 1)  # [G * P, x]
            node_emb_rep = node_emb.repeat(n_cells, 1)  # [G * P, z]
            graph_emb_rep = graph_emb.repeat(n_cells, 1)  # [G, z]
            batch_indices = torch.arange(n_cells, device=device).repeat_interleave(graph_i.num_nodes)

            logits_i = model.decoder(node_feat_rep, node_emb_rep, graph_emb_rep, batch_indices, dest_tensor)

        if model.args["include_out"]:
            node_logits = logits_i[:-n_cells].view(n_cells, -1)  # [G, P]
            out_logits = logits_i[-n_cells:].view(n_cells, 1)  # [G, 1]
            logits_i = torch.cat([node_logits, out_logits], dim=-1)  # [G, P + 1]
        else:
            logits_i = logits_i.view(n_cells, -1)  # [G, P]

        event_index = int(labels[data_index, 0].item())

        receive_probs_i = F.softmax(logits_i, dim=-1)  # [G, P(+1)]
        n_teammates = torch.sum(graph_i.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1).item()
        receive_probs[event_index] = receive_probs_i.reshape(grid_size[1], grid_size[0], -1)

        success_probs_i = receive_probs_i[:, :n_teammates].sum(axis=1)
        success_probs[event_index] = success_probs_i.reshape(grid_size[1], grid_size[0])

    return receive_probs, success_probs
