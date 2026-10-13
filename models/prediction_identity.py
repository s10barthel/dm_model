"""Stable candidate identities for paired evaluation of node-selection models."""

from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from datatools import config
from project_config import TRACKING_PROCESSED_DIR, get_resolved_action_path


def resolve_legacy_node_ids(graph, label, tracking, actions, *, post_action=False):
    """Recover IDs from original tracking, validating each node against its geometry.

    Source node positions may change when nodes are removed. Matching unchanged
    coordinates and team/goal roles avoids mistaking those positions for IDs.
    Ambiguous matches are rejected rather than assigning positional identities.
    """
    action_index = int(label[config.LABEL_INDEX["action_index"]])
    action = actions.loc[action_index]
    frame_key, player_key = ("end_frame_id", "end_player_id") if post_action else ("frame_id", "object_id")
    frame, possessor = int(action[frame_key]), str(action[player_key])
    snapshot = tracking.loc[frame]
    if isinstance(snapshot, pd.DataFrame):
        raise ValueError(f"Ambiguous tracking frame {frame} for action {action_index}.")
    players = [name[:-2] for name in tracking.columns if re.fullmatch(r"(?:home|away)_(?:\d+|goal)_x", name)]
    for goal in ("home_goal", "away_goal"):
        if goal not in players:
            players.append(goal)
    positions = []
    for player in players:
        if player.endswith("_goal"):
            xy = [config.FIELD_SIZE[0] if player.startswith("home") else 0.0, config.FIELD_SIZE[1] / 2]
        else:
            xy = [snapshot.get(f"{player}_x", np.nan), snapshot.get(f"{player}_y", np.nan)]
        xy = np.asarray(xy, dtype=float)
        if possessor.startswith("away"):
            xy = np.asarray(config.FIELD_SIZE) - xy
        positions.append(xy)
    positions = np.asarray(positions)
    ids = []
    features = graph.x.detach().cpu().numpy()
    for node in features:
        same_team = np.asarray([int(p[:4] == possessor[:4]) for p in players]) == node[config.NODE_FEATURE_IS_TEAMMATE]
        same_goal_role = np.asarray([int(p.endswith("_goal")) for p in players]) == node[config.NODE_FEATURE_IS_GOAL]
        matches = np.flatnonzero(same_team & same_goal_role & np.all(
            np.isclose(positions, node[config.NODE_FEATURE_X:config.NODE_FEATURE_Y + 1], atol=1e-3, rtol=0), axis=1))
        if len(matches) != 1:
            raise ValueError(f"Cannot establish a unique node identity for action {action_index}; "
                             "retain original tracking and resolved actions, or regenerate graphs with node IDs.")
        ids.append(players[int(matches[0])])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate recovered node identities for action {action_index}.")
    return ids


def ensure_dataset_node_identities(dataset, feature_root, model_args):
    """Backfill only in-memory legacy graphs, reading one source match at a time."""
    missing = defaultdict(list)
    for graph, label in zip(dataset.features, dataset.labels):
        ids = getattr(graph, "node_ids", None)
        if ids is not None:
            if len(ids) != graph.num_nodes or len(set(map(str, ids))) != len(ids):
                raise ValueError("Graph node IDs do not uniquely identify its nodes.")
            continue
        missing[str(graph.evaluation_match_id)].append((graph, label))
    for match_id, rows in missing.items():
        tracking_path = TRACKING_PROCESSED_DIR / f"{match_id}.parquet"
        actions_path = get_resolved_action_path(
            match_id, intended_receiver_mode=getattr(model_args, "intended_receiver_mode", None) or "original",
            root=feature_root, use_carries=bool(getattr(model_args, "use_carries", False)))
        if not tracking_path.is_file() or not actions_path.is_file():
            raise FileNotFoundError(f"Semantic comparison of legacy graphs requires {tracking_path} and {actions_path}.")
        tracking, actions = pd.read_parquet(tracking_path), pd.read_parquet(actions_path)
        for graph, label in rows:
            graph.node_ids = resolve_legacy_node_ids(
                graph, label, tracking, actions,
                post_action="post_action" in str(getattr(model_args, "feature_dir", "")))


def selection_prediction_identities(graphs, layout, targets, predictions):
    """Translate candidate offsets to IDs without changing legacy numeric exports."""
    graph_ids = getattr(graphs, "node_ids", None)
    if graph_ids is None or len(graph_ids) != graphs.num_graphs:
        raise ValueError("Semantic intent comparison requires node IDs for every graph.")
    ptr = graphs.ptr.detach().cpu().tolist()
    targets, predictions = targets.detach().cpu().tolist(), predictions.detach().cpu().tolist()
    rows = [None] * graphs.num_graphs
    for indices, candidates, _ in layout:
        for gi, positions in zip(indices.tolist(), candidates.tolist()):
            node_ids = list(map(str, graph_ids[gi]))
            if len(node_ids) != ptr[gi + 1] - ptr[gi] or len(node_ids) != len(set(node_ids)):
                raise ValueError("Invalid semantic node IDs in prediction export.")
            candidate_ids = []
            for position in positions:
                if ptr[gi] <= position < ptr[gi + 1]:
                    candidate_ids.append(node_ids[position - ptr[gi]])
                elif position == graphs.num_nodes + gi:
                    candidate_ids.append("__out__")
                else:
                    raise ValueError("Prediction candidate is outside its source graph.")
            rows[gi] = {"target_id": candidate_ids[int(targets[gi])],
                        "prediction_id": candidate_ids[int(predictions[gi])],
                        "candidate_ids": json.dumps(candidate_ids, separators=(",", ":"))}
    if any(row is None for row in rows):
        raise ValueError("Missing graph in semantic prediction export.")
    return rows


def observed_receiver_identities(graphs, labels):
    ids = getattr(graphs, "node_ids", None)
    if ids is None or len(ids) != graphs.num_graphs:
        raise ValueError("Observed receiver export requires semantic node IDs for every graph.")
    targets = labels[:, config.LABEL_INDEX["intent_index"]].detach().cpu().long().tolist()
    result = []
    for node_ids, index in zip(ids, targets):
        if not 0 <= index < len(node_ids):
            raise ValueError("Observed receiver identity is outside its source graph.")
        result.append(str(node_ids[index]))
    return result
