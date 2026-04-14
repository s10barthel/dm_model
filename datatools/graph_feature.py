import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

import datatools.preprocess as proc
from datatools import config, utils
from datatools.config import LABEL_INDEX
from datatools.match import Match
from project_config import (
    ACTION_GRAPH_DIR,
    ACTION_GRAPH_INTENT_TRAIN_DIR,
    DEFAULT_INTENDED_RECEIVER_MODEL_ID,
    POST_ACTION_GRAPH_DIR,
    INTENT_TRAIN_OFFSETS,
    SUCCESS_INTENT_GRAPH_DIR,
    get_action_label_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_intent_train_label_dir,
    get_resolved_action_path,
    load_base_splits,
    resolve_intended_receiver_mode,
)

BASE_NODE_FEATURE_DIM = 19
EXTENDED_NODE_FEATURE_DIM = 25
SUCCESS_INTENT_EXTRA_DIM = 4
SUCCESS_INTENT_WINDOW_SECONDS = 1.0


def calculate_event_features(
    match: Match,
    snapshot: pd.DataFrame,
    possessor: str,
    extend=False,
    sequential=False,
    rotate_to_ltr: bool = True,
    eps=1e-6,
) -> np.ndarray:
    seq_len = len(snapshot) if sequential else 1

    phase_id = snapshot["phase_id"].iloc[0]
    active_players = match.phases.at[phase_id, "active_players"]
    active_keepers = match.phases.at[phase_id, "active_keepers"]

    if not match.include_keepers:
        keeper_cols = [c for c in snapshot.columns if "_".join(c.split("_")[:2]) in active_keepers]
        snapshot = snapshot.drop(keeper_cols, axis=1).copy()

    home_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("home")]
    away_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("away")]
    if not match.include_goals:
        home_cols = [c for c in home_cols if not c.startswith("home_goal")]
        away_cols = [c for c in away_cols if not c.startswith("away_goal")]

    player_cols = home_cols + away_cols if possessor.startswith("home") else away_cols + home_cols
    players = [c[:-2] for c in player_cols if c.endswith("_x")]
    expected_players = set(active_players)
    actual_players = set(players) - {"home_goal", "away_goal"}
    if actual_players != expected_players:
        raise ValueError(
            f"Active-player mismatch for possessor {possessor}: expected {len(expected_players)} players, "
            f"found {len(actual_players)}."
        )

    is_teammate = np.tile([int(p[:4] == possessor[:4]) for p in players], (seq_len, 1))
    is_keeper = np.tile([int(p in active_keepers) for p in players], (seq_len, 1))
    is_goal = np.tile([int("goal" in p) for p in players], (seq_len, 1))

    if not sequential:
        snapshot = snapshot[-1:].copy()

    player_x = snapshot[player_cols[0::6]].values
    player_y = snapshot[player_cols[1::6]].values
    player_vx = snapshot[player_cols[2::6]].values
    player_vy = snapshot[player_cols[3::6]].values
    player_speeds = snapshot[player_cols[4::6]].values
    player_accels = snapshot[player_cols[5::6]].values

    if possessor.endswith("out"):
        poss_x = snapshot["ball_x"].values
        poss_y = snapshot["ball_y"].values
    poss_x = snapshot[f"{possessor}_x"].values[:, np.newaxis]
    poss_y = snapshot[f"{possessor}_y"].values[:, np.newaxis]
    poss_vx = snapshot[f"{possessor}_vx"].values[:, np.newaxis]
    poss_vy = snapshot[f"{possessor}_vy"].values[:, np.newaxis]

    # Make the attacking team plays from left to right
    if rotate_to_ltr and possessor[:4] == "away":
        player_x = config.FIELD_SIZE[0] - player_x
        player_y = config.FIELD_SIZE[1] - player_y
        player_vx = -player_vx
        player_vy = -player_vy

        poss_x = config.FIELD_SIZE[0] - poss_x
        poss_y = config.FIELD_SIZE[1] - poss_y
        poss_vx = -poss_vx
        poss_vy = -poss_vy

    goal_x = config.FIELD_SIZE[0]
    goal_y = config.FIELD_SIZE[1] / 2
    goal_dx, goal_dy, goal_dists = utils.calc_dist(player_x, player_y, goal_x, goal_y)

    if "ball_z" in snapshot.columns:
        ball_z = np.ones((seq_len, len(players))) * snapshot["ball_z"].iloc[-1]
    else:
        ball_z = np.zeros((seq_len, len(players)))

    is_possessor = np.tile((np.array(players) == possessor).astype(int), (seq_len, 1))
    poss_dx, poss_dy, poss_dists = utils.calc_dist(player_x, player_y, poss_x, poss_y)
    poss_vangles = utils.calc_angle(player_vx, player_vy, poss_vx, poss_vy, eps=eps)

    event_features = [
        # Binary features
        is_teammate,
        is_keeper,
        is_goal,
        # Possessor-independent features
        player_x,
        player_y,
        player_vx,
        player_vy,
        player_speeds,
        player_accels,
        goal_dists,
        goal_dx / (goal_dists + eps),  # Cosine between each player-goal line and the x-axis
        goal_dy / (goal_dists + eps),  # Sine between each player-goal line and the x-axis
        ball_z,
        # Possessor features
        is_possessor,
        poss_dists,
        poss_dx / (poss_dists + eps),  # Cosine between each player-possessor line and the x-axis
        poss_dy / (poss_dists + eps),  # Sine between each player-possessor line and the x-axis
        np.cos(poss_vangles),  # Cosine between each player's velocity and the possessor's velocity
        np.sin(poss_vangles),  # Sine between each player's velocity and the possessor's velocity
    ]

    if extend:
        player_xy = np.stack([player_x[-1], player_y[-1]], axis=-1)
        dist_mat = cdist(player_xy, player_xy)

        opponent_mask = is_teammate[-1] != is_teammate[-1][:, np.newaxis]
        neighbor_mask = (dist_mat < 3.0) & opponent_mask
        nearby_opponents_to_target: np.ndarray = neighbor_mask.sum(axis=1)

        opponent_dists = np.where(opponent_mask, dist_mat, np.inf)
        nearest_opponent_to_target: np.ndarray = opponent_dists.min(axis=1)

        closer_mask = (goal_dists[-1][np.newaxis, :] < goal_dists[-1][:, np.newaxis]) & opponent_mask
        closer_opponents_to_goal: np.ndarray = closer_mask.sum(axis=1)

        args = [poss_x[-1, 0], poss_y[-1, 0], player_x[-1], player_y[-1], is_teammate[-1]]
        nearest_opponent_to_pass = utils.find_nearest_opponent_to_pass(*args)
        potential_interceptors = utils.count_potential_interceptors(*args)
        potential_blockers = utils.count_potential_blockers(goal_x, goal_y, *args[2:])

        event_features.extend(
            [
                nearby_opponents_to_target[np.newaxis, :],
                nearest_opponent_to_target[np.newaxis, :],
                closer_opponents_to_goal[np.newaxis, :],
                nearest_opponent_to_pass[np.newaxis, :],
                potential_interceptors[np.newaxis, :],
                potential_blockers[np.newaxis, :],
            ]
        )

    return np.stack(event_features, axis=-1)  # [T, N, x]


def infer_node_feature_dim(extend: bool = True, feature_variant: str = "base") -> int:
    base_dim = EXTENDED_NODE_FEATURE_DIM if extend else BASE_NODE_FEATURE_DIM
    if feature_variant == "success_intent":
        return BASE_NODE_FEATURE_DIM + SUCCESS_INTENT_EXTRA_DIM
    return base_dim


def load_frame_snapshot(primary_tracking: pd.DataFrame, fallback_tracking: pd.DataFrame, frame: int) -> pd.DataFrame:
    snapshot = primary_tracking.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
    if snapshot.empty or "phase_id" not in snapshot.columns:
        snapshot = fallback_tracking.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
    return snapshot


def resolve_prior_frame(phase_tracking: pd.DataFrame, possessor: str, current_frame: int, offset: int) -> int | None:
    if offset <= 0:
        return int(current_frame)

    player_x = f"{possessor}_x"
    player_y = f"{possessor}_y"
    if player_x not in phase_tracking.columns or player_y not in phase_tracking.columns:
        return None

    earlier_tracking = phase_tracking.loc[phase_tracking.index <= current_frame - offset]
    if earlier_tracking.empty:
        return None

    earlier_tracking = earlier_tracking.dropna(subset=[player_x, player_y], how="any")
    if earlier_tracking.empty:
        return None

    return int(earlier_tracking.index[-1])


def resolve_snapshot_player_ids(match: Match, snapshot: pd.DataFrame, possessor: str) -> list[str]:
    phase_id = snapshot["phase_id"].iloc[0]
    active_players = match.phases.at[phase_id, "active_players"]
    active_keepers = match.phases.at[phase_id, "active_keepers"]

    if not match.include_keepers:
        keeper_cols = [c for c in snapshot.columns if "_".join(c.split("_")[:2]) in active_keepers]
        snapshot = snapshot.drop(keeper_cols, axis=1).copy()

    home_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("home")]
    away_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("away")]
    if not match.include_goals:
        home_cols = [c for c in home_cols if not c.startswith("home_goal")]
        away_cols = [c for c in away_cols if not c.startswith("away_goal")]

    player_cols = home_cols + away_cols if possessor.startswith("home") else away_cols + home_cols
    players = [c[:-2] for c in player_cols if c.endswith("_x")]
    expected_players = set(active_players)
    actual_players = set(players) - {"home_goal", "away_goal"}
    if actual_players != expected_players:
        raise ValueError(
            f"Active-player mismatch for possessor {possessor}: expected {len(expected_players)} players, "
            f"found {len(actual_players)}."
        )
    return players


def resolve_action_graph_context(
    match: Match,
    action_index: int,
    post_action: bool = False,
) -> tuple[int | float, str, pd.DataFrame]:
    period = int(match.actions.at[action_index, "period_id"])
    if post_action:
        frame = match.actions.at[action_index, "end_frame_id"]
        possessor = match.actions.at[action_index, "end_player_id"]
    else:
        frame = match.actions.at[action_index, "frame_id"]
        possessor = match.actions.at[action_index, "object_id"]
    period_tracking = match.tracking[match.tracking["period_id"] == period]
    return frame, possessor, period_tracking


def append_node_level_globals(graph: Data, global_features: np.ndarray | list[float] | None) -> Data:
    if graph is None or global_features is None:
        return graph

    global_features = np.asarray(global_features, dtype=float).reshape(1, -1)
    repeated = torch.tensor(global_features, dtype=torch.float32).repeat(graph.x.shape[0], 1)
    graph.x = torch.cat([graph.x, repeated], dim=1)
    return graph


def fallback_pass_trajectory_features(match: Match, action_index: int, rotate_to_ltr: bool = True) -> np.ndarray:
    action = match.actions.loc[action_index]
    start_x = float(action.get("start_x", np.nan))
    start_y = float(action.get("start_y", np.nan))
    end_x = float(action.get("end_x", np.nan))
    end_y = float(action.get("end_y", np.nan))
    start_z = float(action.get("start_z", 0.0))

    end_z = start_z
    receive_frame = action.get("receive_frame_id", np.nan)
    if not pd.isna(receive_frame) and int(receive_frame) in match.tracking.index:
        end_z = float(match.tracking.at[int(receive_frame), "ball_z"]) if "ball_z" in match.tracking.columns else start_z

    if any(pd.isna(v) for v in [start_x, start_y, end_x, end_y]):
        return np.zeros(SUCCESS_INTENT_EXTRA_DIM, dtype=float)

    dx = end_x - start_x
    dy = end_y - start_y
    dz = end_z - start_z
    if rotate_to_ltr and str(action["object_id"]).startswith("away"):
        dx = -dx
        dy = -dy

    disp = np.array([dx, dy, dz], dtype=float)
    disp_norm = float(np.linalg.norm(disp))
    direction = disp / disp_norm if disp_norm > 1e-6 else np.zeros(3, dtype=float)

    frame = action.get("frame_id", np.nan)
    end_frame = receive_frame if not pd.isna(receive_frame) else frame
    duration = 0.0 if pd.isna(frame) or pd.isna(end_frame) else max((float(end_frame) - float(frame)) / match.fps, 1 / match.fps)
    mean_speed = disp_norm / duration
    return np.array([mean_speed, *direction], dtype=float)


def summarize_ball_trajectory(
    match: Match,
    action_index: int,
    fps: int | None = None,
    rotate_to_ltr: bool = True,
) -> np.ndarray:
    fps = fps or match.fps
    action = match.actions.loc[action_index]
    frame = action.get("frame_id", np.nan)
    receive_frame = action.get("receive_frame_id", np.nan)

    if pd.isna(frame):
        return fallback_pass_trajectory_features(match, action_index, rotate_to_ltr=rotate_to_ltr)

    frame = int(frame)
    end_frame = frame + max(int(round(fps * SUCCESS_INTENT_WINDOW_SECONDS)), 1)
    if action.get("receiver_id") == "out":
        episode_end_frame = utils.resolve_episode_end_frame(match.tracking, frame)
        if episode_end_frame is not None:
            receive_frame = episode_end_frame
    if not pd.isna(receive_frame):
        end_frame = min(end_frame, int(receive_frame))
    if end_frame <= frame:
        return fallback_pass_trajectory_features(match, action_index, rotate_to_ltr=rotate_to_ltr)

    available_cols = [col for col in ["ball_x", "ball_y", "ball_z", "ball_speed", "ball_vz"] if col in match.tracking.columns]
    trajectory = match.tracking.loc[frame:end_frame, available_cols].copy()
    if trajectory.empty or not {"ball_x", "ball_y"}.issubset(set(trajectory.columns)):
        return fallback_pass_trajectory_features(match, action_index, rotate_to_ltr=rotate_to_ltr)

    trajectory["ball_z"] = trajectory.get("ball_z", 0.0)
    spatial = trajectory[["ball_x", "ball_y", "ball_z"]].dropna(subset=["ball_x", "ball_y"])
    if spatial.shape[0] < 2:
        return fallback_pass_trajectory_features(match, action_index, rotate_to_ltr=rotate_to_ltr)

    start_xyz = spatial.iloc[0].to_numpy(dtype=float)
    end_xyz = spatial.iloc[-1].to_numpy(dtype=float)
    disp = end_xyz - start_xyz
    if rotate_to_ltr and str(action["object_id"]).startswith("away"):
        disp[0] = -disp[0]
        disp[1] = -disp[1]

    disp_norm = float(np.linalg.norm(disp))
    direction = disp / disp_norm if disp_norm > 1e-6 else np.zeros(3, dtype=float)

    if {"ball_speed", "ball_vz"}.issubset(set(trajectory.columns)):
        velocity = trajectory[["ball_speed", "ball_vz"]].dropna()
        if not velocity.empty:
            mean_speed = float(np.sqrt(velocity["ball_speed"].astype(float) ** 2 + velocity["ball_vz"].astype(float) ** 2).mean())
        else:
            duration = max((spatial.index[-1] - spatial.index[0]) / fps, 1 / fps)
            mean_speed = disp_norm / duration
    else:
        duration = max((spatial.index[-1] - spatial.index[0]) / fps, 1 / fps)
        mean_speed = disp_norm / duration

    return np.array([mean_speed, *direction], dtype=float)


def construct_graph_for_frame(
    match: Match,
    frame: int,
    possessor: str,
    period_tracking: pd.DataFrame,
    feature_dim: int,
    extend: bool = True,
    rotate_to_ltr: bool = True,
    extra_node_features: np.ndarray | None = None,
) -> Data | None:
    if pd.isna(frame) or possessor.split("_")[0] not in ["home", "away"]:
        return None

    frame = int(frame)
    if frame not in match.tracking.index:
        return None

    snapshot = load_frame_snapshot(period_tracking, match.tracking, frame)
    if (
        snapshot.empty
        or "phase_id" not in snapshot.columns
        or f"{possessor}_x" not in snapshot.columns
        or f"{possessor}_y" not in snapshot.columns
    ):
        return None

    phase_id = snapshot["phase_id"].iloc[0]
    if pd.isna(phase_id) or int(phase_id) not in match.phases.index:
        return None

    try:
        player_ids = resolve_snapshot_player_ids(match, snapshot, possessor)
        event_features = torch.tensor(
            calculate_event_features(
                match,
                snapshot,
                possessor,
                extend,
                rotate_to_ltr=rotate_to_ltr,
            )[0],
            dtype=torch.float32,
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if extra_node_features is not None:
        extra = torch.tensor(np.asarray(extra_node_features, dtype=float), dtype=torch.float32).unsqueeze(0)
        event_features = torch.cat([event_features, extra.repeat(event_features.shape[0], 1)], dim=1)
    missing_players = match.max_players - event_features.shape[0]
    if missing_players > 0:
        padding_features = -torch.ones((missing_players, feature_dim))
        event_features = torch.cat([event_features, padding_features], 0)

    node_mask = event_features[:, 0] != -1
    node_attr = event_features[node_mask]
    distances = torch.cdist(node_attr[:, 3:5], node_attr[:, 3:5], p=2)
    teammates = (node_attr[:, 0].unsqueeze(-1) == node_attr[:, 0].unsqueeze(-2)).float()
    edge_index, _ = dense_to_sparse(torch.ones_like(distances))
    distances = distances[edge_index[0], edge_index[1]]
    teammates = teammates[edge_index[0], edge_index[1]]
    edge_attr = torch.stack([distances, teammates], dim=-1)
    graph = Data(x=node_attr, edge_index=edge_index.clone(), edge_attr=edge_attr)
    graph.node_ids = list(player_ids)
    return graph


def construct_graph_for_action(
    match: Match,
    action_index: int,
    feature_variant: str = "base",
    extend: bool = True,
    post_action: bool = False,
    rotate_to_ltr: bool = True,
) -> Data | None:
    if action_index not in match.actions.index:
        return None

    if "ball_accel" not in match.tracking.columns or (feature_variant == "success_intent" and "ball_vz" not in match.tracking.columns):
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    frame, possessor, period_tracking = resolve_action_graph_context(match, action_index, post_action=post_action)
    base_extend = extend if feature_variant == "base" else False
    feature_dim = infer_node_feature_dim(base_extend, feature_variant=feature_variant)
    extra_node_features = None
    if feature_variant == "success_intent":
        extra_node_features = summarize_ball_trajectory(match, action_index, fps=match.fps, rotate_to_ltr=rotate_to_ltr)

    return construct_graph_for_frame(
        match,
        frame,
        possessor,
        period_tracking,
        feature_dim,
        extend=base_extend,
        rotate_to_ltr=rotate_to_ltr,
        extra_node_features=extra_node_features,
    )


def construct_graph_features(
    match: Match,
    extend=True,
    post_action=False,
    feature_variant: str = "base",
    action_indices: np.ndarray | list[int] | None = None,
    verbose=True,
) -> List[Data | None]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    if post_action:
        match.actions = match.label_post_actions(match.actions)

    feature_graphs: List[Data | None] = []
    if action_indices is None:
        action_indices = match.labels[:, 0].long().numpy()
    action_indices = np.asarray(action_indices, dtype=int)
    iterator = tqdm(action_indices, desc=f"Constructing {feature_variant} graphs") if verbose else action_indices

    for action_index in iterator:
        graph = construct_graph_for_action(
            match,
            int(action_index),
            feature_variant=feature_variant,
            extend=extend,
            post_action=post_action,
        )
        feature_graphs.append(graph)

    return feature_graphs


def count_invalid_graphs(graphs: List[Data | None]) -> int:
    return sum(graph is None for graph in graphs)


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def construct_intent_training_samples(
    match: Match,
    offsets: Tuple[int, ...] = INTENT_TRAIN_OFFSETS,
    extend: bool = True,
    verbose: bool = True,
) -> Tuple[List[Data], torch.Tensor]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    feature_dim = infer_node_feature_dim(extend)
    offsets = tuple(int(offset) for offset in offsets if int(offset) > 0)
    augmented_graphs: List[Data] = []
    augmented_labels: List[torch.Tensor] = []
    action_indices = match.labels[:, 0].long().numpy()
    label_lookup = {int(label[0].item()): label for label in match.labels}
    iterator = tqdm(action_indices, desc="Augmenting intent samples") if verbose else action_indices

    for action_index in iterator:
        action_index = int(action_index)
        label_row = label_lookup[action_index]
        period = int(match.actions.at[action_index, "period_id"])
        current_frame = int(match.actions.at[action_index, "frame_id"])
        possessor = match.actions.at[action_index, "object_id"]
        period_tracking = match.tracking[match.tracking["period_id"] == period]

        base_graph = construct_graph_for_action(match, action_index, feature_variant="base", extend=extend)
        if base_graph is None:
            continue

        augmented_graphs.append(base_graph)
        augmented_labels.append(label_row.clone())

        snapshot = load_frame_snapshot(period_tracking, match.tracking, current_frame)
        phase_id = snapshot["phase_id"].iloc[0] if not snapshot.empty and "phase_id" in snapshot.columns else np.nan
        if pd.isna(phase_id) or int(phase_id) not in match.phases.index:
            continue

        phase_tracking = match.tracking[match.tracking["phase_id"] == int(phase_id)]
        for offset in offsets:
            prior_frame = resolve_prior_frame(phase_tracking, possessor, current_frame, offset)
            if prior_frame is None:
                continue

            graph = construct_graph_for_frame(match, prior_frame, possessor, period_tracking, feature_dim, extend=extend)
            if graph is None:
                continue

            augmented_graphs.append(graph)
            augmented_labels.append(label_row.clone())

    if augmented_labels:
        return augmented_graphs, torch.stack(augmented_labels, dim=0)
    return augmented_graphs, torch.empty((0, match.labels.shape[1]), dtype=match.labels.dtype)


def augment_blocked_actions(match: Match, max_block_dist=5, max_block_angle=15) -> Tuple[List[Data], torch.Tensor]:
    augmented_features = []
    augmented_labels = []

    action_indices = match.labels[:, 0].numpy().astype(int)
    tqdm_desc = "Augmenting features and labels"

    for i, action_index in enumerate(tqdm(action_indices, desc=tqdm_desc)):
        augmented_features.append(match.graph_features_0[i])
        augmented_labels.append(match.labels[i])

        if match.actions.at[action_index, "spadl_type"] in config.SET_PIECE:
            continue

        frame = match.actions.at[action_index, "frame_id"]
        possessor = match.actions.at[action_index, "object_id"]
        real_intent = match.actions.at[action_index, "intent_id"]
        snapshot: pd.Series = match.tracking.loc[frame]

        teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_.*_x", c)]
        team_x = snapshot[[f"{p}_x" for p in teammates]].values
        team_y = snapshot[[f"{p}_y" for p in teammates]].values

        team_dist_x = (team_x - snapshot[f"{possessor}_x"]).astype(float)
        team_dist_y = (team_y - snapshot[f"{possessor}_y"]).astype(float)
        team_dists = np.sqrt(team_dist_x**2 + team_dist_y**2)

        oppo_team = "away" if possessor[:4] == "home" else "home"
        opponents = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{oppo_team}_.*_x", c)]
        oppo_x = snapshot[[f"{p}_x" for p in opponents]].values
        oppo_y = snapshot[[f"{p}_y" for p in opponents]].values

        oppo_dist_x = (oppo_x - snapshot[f"{possessor}_x"]).astype(float)
        oppo_dist_y = (oppo_y - snapshot[f"{possessor}_y"]).astype(float)
        oppo_dists = np.sqrt(oppo_dist_x**2 + oppo_dist_y**2)
        blockers = np.array(opponents)[np.where(oppo_dists < max_block_dist)[0]][:3]

        if match.include_goals:
            goal_index = teammates.index(f"{possessor[:4]}_goal")

        for blocker in blockers:
            poss_x = snapshot[f"{possessor}_x"]
            poss_y = snapshot[f"{possessor}_y"]
            block_x = snapshot[f"{blocker}_x"]
            block_y = snapshot[f"{blocker}_y"]

            team_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, team_x, team_y)
            oppo_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, oppo_x, oppo_y)
            blocked_teammates = np.where(team_angles < max_block_angle / 180 * np.pi)[0].tolist()
            blocked_opponents = np.where(oppo_angles < max_block_angle / 180 * np.pi)[0].tolist()
            blocked_opponents = [k for k in blocked_opponents if k != opponents.index(blocker)]

            if not blocked_opponents:
                continue

            close_teammates = np.where(team_dists < oppo_dists[blocked_opponents].max() - 10)[0].tolist()
            if match.include_goals and team_dists[goal_index] < 40:
                close_teammates.append(goal_index)  # Assume that the shot was prevented if goal distance was < 30
            blocked_intent_indices = list(set(blocked_teammates) & set(close_teammates))
            blocked_intents = [p for p in np.array(teammates)[blocked_intent_indices] if p != real_intent]

            for blocked_intent in blocked_intents:
                augmented_labels_i = match.labels[i].clone()

                if blocked_intent.endswith("goal"):  # An augmented shot that would be blocked
                    augmented_labels_i[1] = 0
                    augmented_labels_i[3] = 1
                else:  # An augmented pass that would be blocked
                    augmented_labels_i[1] = 1
                    augmented_labels_i[3] = 0

                augmented_labels_i[5] = teammates.index(blocked_intent)
                augmented_labels_i[6] = (teammates + opponents).index(blocker)
                augmented_labels_i[LABEL_INDEX["is_real"]] = 0  # Indicating that this is an augmented event and not a real one
                augmented_labels_i[LABEL_INDEX["blocked"]] = 1  # Indicating that this is a blocked event
                augmented_labels_i[LABEL_INDEX["success"]] = 0  # Indicating that this is a failed event

                augmented_features.append(match.graph_features_0[i].clone())
                augmented_labels.append(augmented_labels_i)

    augmented_labels = torch.stack(augmented_labels, dim=0)
    return augmented_features, augmented_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, required=False, default="all", choices=["all", "shot_augment"])
    parser.add_argument("--split", type=str, required=False, default="train", choices=["train", "test"])
    parser.add_argument("--return_type", type=str, required=False, default="disc_0.9", help="way of defining future xG")
    parser.add_argument("--post_action", action="store_true", default=False, help="construct post-action features")
    parser.add_argument("--augment_blocks", action="store_true", default=False)
    parser.add_argument(
        "--feature_variant",
        type=str,
        default="base",
        choices=["base", "intent_train_augmented", "success_intent"],
    )
    parser.add_argument("--use-original-intended-receiver", action="store_true", default=False)
    parser.add_argument("--use-intended-receiver-model", action="store_true", default=False)
    parser.add_argument(
        "--intended-receiver-model-id",
        type=str,
        default=DEFAULT_INTENDED_RECEIVER_MODEL_ID,
    )
    args, _ = parser.parse_known_args()
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )

    if args.action_type.startswith("shot"):
        if args.feature_variant != "base":
            raise ValueError("Intent-training augmentation is only supported for action_type=all.")
        args.action_type = "shot_augment"
        feature_dir = "data/ajax/features/augmented_shot_graphs"
        label_dir = "data/ajax/features/augmented_shot_labels"
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

    else:  # args.actions_type == "all"
        if args.feature_variant == "intent_train_augmented":
            feature_dir = ACTION_GRAPH_INTENT_TRAIN_DIR
            label_dir = get_intent_train_label_dir(args.return_type, intended_receiver_mode=intended_receiver_mode)
        elif args.feature_variant == "success_intent":
            feature_dir = SUCCESS_INTENT_GRAPH_DIR
            label_dir = get_action_label_dir(args.return_type, intended_receiver_mode=intended_receiver_mode)
        else:
            feature_dir = ACTION_GRAPH_DIR
            label_dir = get_action_label_dir(args.return_type, intended_receiver_mode=intended_receiver_mode)
        Path(feature_dir).mkdir(parents=True, exist_ok=True)
        Path(label_dir).mkdir(parents=True, exist_ok=True)

        if args.post_action:
            if args.feature_variant != "base":
                raise ValueError("Post-action features are only supported for base graphs.")
            post_feature_dir = str(POST_ACTION_GRAPH_DIR)
            os.makedirs(post_feature_dir, exist_ok=True)

    if args.augment_blocks:
        augmented_feature_dir = str(get_augmented_feature_dir(intended_receiver_mode))
        augmented_label_dir = str(get_augmented_label_dir(intended_receiver_mode))
        os.makedirs(augmented_feature_dir, exist_ok=True)
        os.makedirs(augmented_label_dir, exist_ok=True)

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet")
    lineups["game_id"] = lineups["stats_perform_match_id"]
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    match_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]

    if args.split == "train":
        match_ids, _ = load_base_splits()
    else:  # split == "test"
        _, match_ids = load_base_splits()

    n_matches = len(match_ids)

    successful_matches = 0
    skipped_matches: list[dict[str, str]] = []
    aggregate_stats = {
        "failed_passes_total": 0,
        "failed_passes_labeled": 0,
        "failed_passes_missing_endpoint": 0,
        "model_failed_passes_total": 0,
        "model_failed_passes_scored": 0,
        "model_angle_only_fallbacks": 0,
    }

    for i, match_id in enumerate(match_ids):
        try:
            events = pd.read_csv(f"data/ajax/event_synced/{match_id}.csv", header=0, parse_dates=["utc_timestamp"])
            tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
            match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id]

            match = Match(events, tracking, match_lineup, args.action_type, include_goals=True)
            match_date = match_dates[match_id].date()
            match_name = " vs ".join(match_lineup["contestant_name"].unique())
            print(f"\n[{i+1}/{n_matches}] {match_id}: {match_name} on {match_date}")

            if args.return_type.startswith("disc"):
                gamma = float(args.return_type.split("_")[-1])
                match.labels = match.construct_labels(
                    discount_xg=True,
                    gamma=gamma,
                    intended_receiver_mode=intended_receiver_mode,
                    intended_receiver_model_id=args.intended_receiver_model_id,
                )
            elif args.return_type.startswith("next"):
                lookahead_len = int(args.return_type.split("_")[-1])
                match.labels = match.construct_labels(
                    discount_xg=False,
                    lookahead_len=lookahead_len,
                    intended_receiver_mode=intended_receiver_mode,
                    intended_receiver_model_id=args.intended_receiver_model_id,
                )
            if match.labels.numel() == 0:
                raise ValueError("No usable labels were constructed for this match.")

            match_stats = dict(match.intended_receiver_stats)
            for key in aggregate_stats:
                aggregate_stats[key] += int(match_stats.get(key, 0) or 0)
            if match_stats.get("model_failed_passes_total", 0):
                print(
                    "  Intended-receiver model fallback: "
                    f"{int(match_stats.get('model_angle_only_fallbacks', 0))} / "
                    f"{int(match_stats.get('model_failed_passes_total', 0))} failed passes used angle_only."
                )

            if args.action_type == "all":
                resolved_action_path = get_resolved_action_path(match_id, intended_receiver_mode=intended_receiver_mode)
                resolved_action_path.parent.mkdir(parents=True, exist_ok=True)
                match.actions.to_parquet(resolved_action_path)

            action_indices = match.labels[:, 0].numpy().astype(int)
            if not np.all(np.sort(action_indices) == action_indices):
                raise ValueError("Action labels are not sorted by action index.")

            if args.feature_variant == "intent_train_augmented":
                print("Constructing intent-training augmentation graphs...")
                augmented_graphs, augmented_labels = construct_intent_training_samples(match, extend=True)
                if not augmented_graphs or augmented_labels.numel() == 0:
                    raise ValueError("No usable intent-training samples were constructed for this match.")
                torch.save(augmented_labels, f"{label_dir}/{match_id}.pt")
                torch.save(augmented_graphs, f"{feature_dir}/{match_id}.pt")
                print(f"Successfully saved {len(augmented_graphs)} augmented intent samples.")
            elif args.feature_variant == "success_intent":
                print("Constructing success-intent graph features for actions...")
                success_graphs = construct_graph_features(
                    match,
                    extend=False,
                    post_action=False,
                    feature_variant="success_intent",
                )
                valid_graphs = len(success_graphs) - count_invalid_graphs(success_graphs)
                if valid_graphs == 0:
                    raise ValueError("No usable success-intent graphs were constructed for this match.")
                torch.save(match.labels, f"{label_dir}/{match_id}.pt")
                torch.save(success_graphs, f"{feature_dir}/{match_id}.pt")
                invalid_graphs = count_invalid_graphs(success_graphs)
                if invalid_graphs:
                    print(f"  Skipped {invalid_graphs} corrupted action frames while building success-intent graphs.")
                print(f"Successfully saved success-intent graphs for {match.labels.shape[0]} events.")
            else:
                print("Constructing base graph features for actions...")
                match.graph_features_0 = construct_graph_features(
                    match,
                    extend=True,
                    post_action=False,
                    feature_variant="base",
                )
                valid_graphs = len(match.graph_features_0) - count_invalid_graphs(match.graph_features_0)
                if valid_graphs == 0:
                    raise ValueError("No usable action graphs were constructed for this match.")
                torch.save(match.labels, f"{label_dir}/{match_id}.pt")
                torch.save(match.graph_features_0, f"{feature_dir}/{match_id}.pt")
                invalid_graphs = count_invalid_graphs(match.graph_features_0)
                if invalid_graphs:
                    print(f"  Skipped {invalid_graphs} corrupted action frames while building graphs.")

                if args.post_action:
                    print("Constructing base graph features for post-actions...")
                    match.graph_features_1 = construct_graph_features(
                        match,
                        extend=True,
                        post_action=True,
                        feature_variant="base",
                    )
                    torch.save(match.graph_features_1, f"{post_feature_dir}/{match_id}.pt")

                print(f"Successfully saved for {match.labels.shape[0]} events.")

            if args.augment_blocks:
                augmented_graph_features, augmented_labels = augment_blocked_actions(match)
                torch.save(augmented_graph_features, f"{augmented_feature_dir}/{match_id}.pt")
                torch.save(augmented_labels, f"{augmented_label_dir}/{match_id}.pt")
                print(f"Successfully saved for {augmented_labels.shape[0]} augmented events.")

            successful_matches += 1
        except Exception as exc:
            error_summary = summarize_exception(exc)
            skipped_matches.append({"match_id": match_id, "error": error_summary})
            print(f"  SKIP {match_id}: {error_summary}")

    if successful_matches == 0:
        raise RuntimeError(f"No usable matches were processed for split={args.split}, feature_variant={args.feature_variant}.")
    if skipped_matches:
        print(f"Skipped {len(skipped_matches)} matches while generating graph features.")
        for item in skipped_matches[:10]:
            print(f"  {item['match_id']}: {item['error']}")
        if len(skipped_matches) > 10:
            print(f"  ... and {len(skipped_matches) - 10} more")
    if intended_receiver_mode == "model" and aggregate_stats["model_failed_passes_total"] > 0:
        print(
            "Intended-receiver model fallback summary: "
            f"{aggregate_stats['model_angle_only_fallbacks']} / {aggregate_stats['model_failed_passes_total']} "
            "failed passes used angle_only."
        )
