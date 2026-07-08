import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from types import SimpleNamespace
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
from datatools.success_intent import build_success_intent_resolved_actions
from project_config import (
    ACTION_GRAPH_DIR,
    ACTION_GRAPH_INTENT_TRAIN_DIR,
    FEATURE_DIR,
    INTENDED_RECEIVER_MODE_MODEL,
    POST_ACTION_GRAPH_DIR,
    INTENT_TRAIN_OFFSETS,
    SUCCESS_INTENT_GRAPH_DIR,
    get_action_label_dir,
    get_action_graph_dir,
    get_action_graph_intent_train_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_feature_run_root,
    get_intent_train_label_dir,
    get_post_action_graph_dir,
    get_resolved_action_path,
    get_success_intent_graph_dir,
    get_success_intent_label_dir,
    load_base_splits,
    resolve_generation_intended_receiver_modes,
    resolve_requested_return_types,
    validate_intended_receiver_mode,
)

OFFSIDE_NODE_FEATURE_DIM = 1
BASE_NODE_FEATURE_DIM = config.NODE_FEATURE_CORE_DIM + OFFSIDE_NODE_FEATURE_DIM
EXTENDED_NODE_FEATURE_DIM = config.NODE_FEATURE_EXTENDED_END + OFFSIDE_NODE_FEATURE_DIM
BASE_EDGE_FEATURE_DIM = 2
VELOCITY_EDGE_FEATURE_EXTRA_DIM = 2
SUCCESS_INTENT_EXTRA_DIM = 4
SUCCESS_INTENT_WINDOW_SECONDS = 1.0


@dataclass(frozen=True)
class FeatureGenerationPaths:
    feature_dir: str
    label_dir: str | None = None
    post_feature_dir: str | None = None


@dataclass(frozen=True)
class MatchGenerationTask:
    match_id: str
    match_index: int
    n_matches: int
    match_lineup: pd.DataFrame
    args_dict: dict[str, object]
    feature_root: str
    paths: FeatureGenerationPaths
    show_progress: bool
    worker_thread_limit: int


@dataclass(frozen=True)
class MatchGenerationResult:
    match_id: str
    summary: str
    stats_by_mode: dict[str, dict[str, int]]


def resolve_num_workers(value: str | int) -> int:
    if isinstance(value, int):
        workers = value
    else:
        text = str(value).strip().lower()
        if text == "auto":
            cpu_count = os.cpu_count() or 1
            workers = max(1, min(6, cpu_count - 2))
        else:
            workers = int(text)
    if workers < 1:
        raise ValueError("--num-workers must be a positive integer or 'auto'.")
    return workers


def configure_worker_thread_limit(limit: int) -> None:
    value = str(int(limit))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def calculate_offside_flags(
    player_x: np.ndarray,
    ball_x: np.ndarray,
    is_teammate: np.ndarray,
    is_goal: np.ndarray,
) -> np.ndarray:
    player_x = np.asarray(player_x, dtype=float)
    ball_x = np.asarray(ball_x, dtype=float).reshape(player_x.shape[0])
    is_teammate = np.asarray(is_teammate).astype(bool)
    is_goal = np.asarray(is_goal).astype(bool)

    is_offside = np.zeros(player_x.shape, dtype=int)
    finite_player_x = np.isfinite(player_x)
    non_goal = ~is_goal
    half_line_x = config.FIELD_SIZE[0] / 2

    for team_is_teammate, attacks_right in [(True, True), (False, False)]:
        team_mask = (is_teammate == team_is_teammate) & non_goal & finite_player_x
        opponent_mask = (is_teammate != team_is_teammate) & non_goal & finite_player_x

        for t in range(player_x.shape[0]):
            if not np.isfinite(ball_x[t]):
                continue

            opponent_x = player_x[t, opponent_mask[t]]
            if opponent_x.size < 2:
                continue

            if attacks_right:
                second_last_opponent_x = np.partition(opponent_x, -2)[-2]
                offside_mask = (
                    team_mask[t]
                    & (player_x[t] > half_line_x)
                    & (player_x[t] > ball_x[t])
                    & (player_x[t] > second_last_opponent_x)
                )
            else:
                second_last_opponent_x = np.partition(opponent_x, 1)[1]
                offside_mask = (
                    team_mask[t]
                    & (player_x[t] < half_line_x)
                    & (player_x[t] < ball_x[t])
                    & (player_x[t] < second_last_opponent_x)
                )

            is_offside[t, offside_mask] = 1

    return is_offside


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
    ball_x = snapshot["ball_x"].values if "ball_x" in snapshot.columns else np.full(seq_len, np.nan)

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
        ball_x = config.FIELD_SIZE[0] - ball_x

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
    is_offside = calculate_offside_flags(player_x, ball_x, is_teammate, is_goal)

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

    event_features.append(is_offside)

    return np.stack(event_features, axis=-1)  # [T, N, x]


def infer_node_feature_dim(extend: bool = True, feature_variant: str = "base") -> int:
    base_dim = EXTENDED_NODE_FEATURE_DIM if extend else BASE_NODE_FEATURE_DIM
    if feature_variant == "success_intent":
        return BASE_NODE_FEATURE_DIM + SUCCESS_INTENT_EXTRA_DIM
    return base_dim


def infer_edge_feature_dim(add_v_edge_features: bool = False) -> int:
    return BASE_EDGE_FEATURE_DIM + (VELOCITY_EDGE_FEATURE_EXTRA_DIM if add_v_edge_features else 0)


def load_frame_snapshot(primary_tracking: pd.DataFrame, fallback_tracking: pd.DataFrame, frame: int) -> pd.DataFrame:
    snapshot = primary_tracking.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
    if snapshot.empty or "phase_id" not in snapshot.columns:
        snapshot = fallback_tracking.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
    if snapshot.empty or "phase_id" not in snapshot.columns or frame not in snapshot.index:
        return snapshot

    # Keep the snapshot phase-local so the first frame of a new phase can still
    # build a graph without inheriting an incompatible active-player set from
    # the immediately previous frame.
    current_phase = snapshot.at[frame, "phase_id"]
    if pd.notna(current_phase):
        same_phase_snapshot = snapshot.loc[snapshot["phase_id"].eq(current_phase)].copy()
        if not same_phase_snapshot.empty:
            snapshot = same_phase_snapshot
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
    add_v_edge_features: bool = False,
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
        offside_feature = event_features[:, -OFFSIDE_NODE_FEATURE_DIM:]
        event_features = torch.cat(
            [
                event_features[:, :-OFFSIDE_NODE_FEATURE_DIM],
                extra.repeat(event_features.shape[0], 1),
                offside_feature,
            ],
            dim=1,
        )
    missing_players = match.max_players - event_features.shape[0]
    if missing_players > 0:
        padding_features = -torch.ones((missing_players, feature_dim))
        event_features = torch.cat([event_features, padding_features], 0)

    node_mask = event_features[:, config.NODE_FEATURE_IS_TEAMMATE] != -1
    node_attr = event_features[node_mask]
    xy_slice = slice(config.NODE_FEATURE_X, config.NODE_FEATURE_Y + 1)
    distances = torch.cdist(node_attr[:, xy_slice], node_attr[:, xy_slice], p=2)
    teammates = (
        node_attr[:, config.NODE_FEATURE_IS_TEAMMATE].unsqueeze(-1)
        == node_attr[:, config.NODE_FEATURE_IS_TEAMMATE].unsqueeze(-2)
    ).float()
    edge_index, _ = dense_to_sparse(torch.ones_like(distances))
    distances = distances[edge_index[0], edge_index[1]]
    teammates = teammates[edge_index[0], edge_index[1]]
    edge_features = [distances, teammates]
    if add_v_edge_features:
        src_vx = node_attr[edge_index[0], config.NODE_FEATURE_VX]
        src_vy = node_attr[edge_index[0], config.NODE_FEATURE_VY]
        dst_vx = node_attr[edge_index[1], config.NODE_FEATURE_VX]
        dst_vy = node_attr[edge_index[1], config.NODE_FEATURE_VY]
        src_speed = torch.sqrt(src_vx.square() + src_vy.square()).clamp_min(1e-6)
        dst_speed = torch.sqrt(dst_vx.square() + dst_vy.square()).clamp_min(1e-6)
        vel_cos = torch.clamp((src_vx * dst_vx + src_vy * dst_vy) / (src_speed * dst_speed), -1.0, 1.0)
        vel_angle = torch.arccos(vel_cos)
        edge_features.extend([torch.cos(vel_angle), torch.sin(vel_angle)])
    edge_attr = torch.stack(edge_features, dim=-1)
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
    add_v_edge_features: bool = False,
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
        add_v_edge_features=add_v_edge_features,
    )


def construct_graph_features(
    match: Match,
    extend=True,
    post_action=False,
    feature_variant: str = "base",
    action_indices: np.ndarray | list[int] | None = None,
    verbose=True,
    add_v_edge_features: bool = False,
) -> List[Data | None]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    if post_action:
        match.actions = match.label_post_actions(match.actions)

    feature_graphs: List[Data | None] = []
    if action_indices is None:
        if match.labels is None:
            raise ValueError("construct_graph_features requires either action_indices or match.labels.")
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
            add_v_edge_features=add_v_edge_features,
        )
        feature_graphs.append(graph)

    return feature_graphs


def count_invalid_graphs(graphs: List[Data | None]) -> int:
    return sum(graph is None for graph in graphs)


def build_match_for_feature_generation(
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    match_lineup: pd.DataFrame,
    args: argparse.Namespace,
) -> Match:
    return Match(
        events,
        tracking,
        match_lineup,
        args.action_type,
        include_goals=True,
        next_action_conditions_enabled=args.next_action_conditions_enabled,
    )


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def extract_sorted_action_indices(labels: torch.Tensor, *, context: str) -> np.ndarray:
    action_indices = labels[:, 0].numpy().astype(int)
    if not np.all(np.sort(action_indices) == action_indices):
        raise ValueError(f"{context} are not sorted by action index.")
    return action_indices


def bind_canonical_graph_context(
    match: Match,
    resolved_actions_by_mode: dict[str, pd.DataFrame],
    labels_by_key: dict[tuple[str, str], torch.Tensor],
    primary_mode: str,
    primary_return_type: str,
) -> np.ndarray:
    canonical_key = (primary_mode, primary_return_type)
    if primary_mode not in resolved_actions_by_mode:
        raise ValueError(f"Missing resolved actions for intended_receiver_mode={primary_mode}.")
    if canonical_key not in labels_by_key:
        raise ValueError(
            "Missing canonical labels for "
            f"intended_receiver_mode={primary_mode}, return_type={primary_return_type}."
        )

    canonical_labels = labels_by_key[canonical_key]
    canonical_action_indices = extract_sorted_action_indices(
        canonical_labels,
        context=(
            "Canonical action labels for "
            f"intended_receiver_mode={primary_mode}, return_type={primary_return_type}"
        ),
    )
    for (mode, return_type), labels in labels_by_key.items():
        action_indices = extract_sorted_action_indices(
            labels,
            context=f"Action labels for intended_receiver_mode={mode}, return_type={return_type}",
        )
        if not np.array_equal(action_indices, canonical_action_indices):
            raise ValueError(
                "Shared base graph artifacts require identical action ordering across label variants, but "
                f"intended_receiver_mode={mode}, return_type={return_type} does not match the canonical ordering "
                f"from intended_receiver_mode={primary_mode}, return_type={primary_return_type}."
            )

    match.actions = resolved_actions_by_mode[primary_mode].copy()
    match.labels = canonical_labels.clone()
    return canonical_action_indices


def build_labels_by_mode_and_return(
    match: Match,
    intended_receiver_modes: list[str],
    return_types: list[str],
    intended_receiver_model_id: str | None,
    feature_root: Path | None = None,
    match_id: str | None = None,
    prefer_existing_resolved_actions: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], torch.Tensor], dict[str, dict[str, int]]]:
    if not return_types:
        raise ValueError("At least one return_type must be requested.")

    resolved_actions_by_mode: dict[str, pd.DataFrame] = {}
    labels_by_key: dict[tuple[str, str], torch.Tensor] = {}
    stats_by_mode: dict[str, dict[str, int]] = {}

    for mode in intended_receiver_modes:
        primary_return_type = return_types[0]
        resolved_actions: pd.DataFrame | None = None
        if prefer_existing_resolved_actions:
            if feature_root is None or match_id is None:
                raise ValueError("feature_root and match_id are required when prefer_existing_resolved_actions=True.")
            resolved_action_path = get_resolved_action_path(
                match_id,
                intended_receiver_mode=mode,
                root=feature_root,
            )
            if resolved_action_path.exists():
                resolved_actions = pd.read_parquet(resolved_action_path)

        if resolved_actions is not None:
            labels = match.construct_labels(
                intended_receiver_mode=mode,
                intended_receiver_model_id=intended_receiver_model_id,
                relabel_intended_receivers=False,
                resolved_actions=resolved_actions,
                return_type=primary_return_type,
            )
        else:
            labels = match.construct_labels(
                intended_receiver_mode=mode,
                intended_receiver_model_id=intended_receiver_model_id,
                return_type=primary_return_type,
            )
        if labels.numel() == 0:
            raise ValueError(f"No usable labels were constructed for intended_receiver_mode={mode}.")

        if resolved_actions is None:
            resolved_actions = match.actions.copy()
            resolved_actions.attrs["intended_receiver_stats"] = dict(match.intended_receiver_stats)
        resolved_actions_by_mode[mode] = resolved_actions
        stats_by_mode[mode] = {
            key: int(match.intended_receiver_stats.get(key, 0) or 0)
            for key in [
                "failed_passes_total",
                "failed_passes_labeled",
                "failed_passes_missing_endpoint",
                "model_failed_passes_total",
                "model_failed_passes_scored",
                "model_angle_only_fallbacks",
            ]
        }
        labels_by_key[(mode, primary_return_type)] = labels.clone()

        for return_type in return_types[1:]:
            labels_by_key[(mode, return_type)] = match.construct_labels(
                intended_receiver_mode=mode,
                intended_receiver_model_id=intended_receiver_model_id,
                relabel_intended_receivers=False,
                resolved_actions=resolved_actions,
                return_type=return_type,
            ).clone()

    return resolved_actions_by_mode, labels_by_key, stats_by_mode


def build_success_intent_labels_by_return(
    match: Match,
    return_types: list[str],
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    if not return_types:
        raise ValueError("At least one return_type must be requested.")

    resolved_actions = build_success_intent_resolved_actions(match.actions)
    labels_by_return: dict[str, torch.Tensor] = {}
    primary_return_type = return_types[0]
    labels = match.construct_labels(
        relabel_intended_receivers=False,
        resolved_actions=resolved_actions,
        return_type=primary_return_type,
    )
    if labels.numel() == 0:
        raise ValueError("No usable labels were constructed for success_intent.")
    labels_by_return[primary_return_type] = labels.clone()

    for return_type in return_types[1:]:
        labels_by_return[return_type] = match.construct_labels(
            relabel_intended_receivers=False,
            resolved_actions=resolved_actions,
            return_type=return_type,
        ).clone()

    return resolved_actions, labels_by_return


def build_intent_train_labels_from_source_actions(
    base_labels: torch.Tensor,
    source_intent_train_labels: torch.Tensor,
) -> torch.Tensor:
    if source_intent_train_labels.numel() == 0:
        raise ValueError("Source intent-train labels are empty.")

    label_lookup = {int(label[0].item()): label.clone() for label in base_labels}
    aligned_labels = []
    for action_index in source_intent_train_labels[:, 0].long().tolist():
        action_index = int(action_index)
        if action_index not in label_lookup:
            raise ValueError(f"Missing base label for intent-train action_index={action_index}.")
        aligned_labels.append(label_lookup[action_index].clone())
    return torch.stack(aligned_labels, dim=0)


def _validate_labels_only_base_artifacts(
    match_id: str,
    feature_root: Path,
    *,
    feature_variant: str = "base",
    intent_train_label_source_mode: str | None = None,
    intent_train_label_source_return_type: str | None = None,
) -> None:
    if feature_variant == "base":
        graph_path = get_action_graph_dir(feature_root) / f"{match_id}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Base action graphs not found at {graph_path}.")
        return

    if feature_variant == "intent_train_augmented":
        graph_path = get_action_graph_intent_train_dir(feature_root) / f"{match_id}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Base intent-train graphs not found at {graph_path}.")
        if intent_train_label_source_mode is None or intent_train_label_source_return_type is None:
            raise ValueError("Labels-only intent-train generation requires source mode and return_type.")
        source_label_path = (
            get_intent_train_label_dir(
                intent_train_label_source_return_type,
                intended_receiver_mode=intent_train_label_source_mode,
                root=feature_root,
            )
            / f"{match_id}.pt"
        )
        if not source_label_path.exists():
            raise FileNotFoundError(f"Source intent-train labels not found at {source_label_path}.")
        return

    raise ValueError(f"Unsupported labels-only feature_variant={feature_variant!r}.")


def save_resolved_actions_if_missing(
    resolved_actions_by_mode: dict[str, pd.DataFrame],
    match_id: str,
    feature_root: Path,
    overwrite: bool = False,
) -> None:
    for mode, resolved_actions in resolved_actions_by_mode.items():
        resolved_action_path = get_resolved_action_path(
            match_id,
            intended_receiver_mode=mode,
            root=feature_root,
        )
        if resolved_action_path.exists() and not overwrite:
            continue
        resolved_action_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_actions.to_parquet(resolved_action_path)


def save_action_labels_if_missing(
    labels_by_key: dict[tuple[str, str], torch.Tensor],
    match_id: str,
    feature_root: Path,
    overwrite: bool = False,
) -> None:
    for (mode, return_type), labels in labels_by_key.items():
        label_dir = get_action_label_dir(
            return_type,
            intended_receiver_mode=mode,
            root=feature_root,
        )
        label_path = label_dir / f"{match_id}.pt"
        if label_path.exists() and not overwrite:
            continue
        label_dir.mkdir(parents=True, exist_ok=True)
        torch.save(labels, label_path)


def save_intent_train_labels_from_existing_source(
    labels_by_key: dict[tuple[str, str], torch.Tensor],
    match_id: str,
    feature_root: Path,
    source_mode: str,
    source_return_type: str,
    overwrite: bool = False,
) -> None:
    source_label_path = (
        get_intent_train_label_dir(
            source_return_type,
            intended_receiver_mode=source_mode,
            root=feature_root,
        )
        / f"{match_id}.pt"
    )
    if not source_label_path.exists():
        raise FileNotFoundError(f"Source intent-train labels not found at {source_label_path}.")
    source_labels = torch.load(source_label_path, weights_only=False)

    for (mode, return_type), base_labels in labels_by_key.items():
        label_dir = get_intent_train_label_dir(
            return_type,
            intended_receiver_mode=mode,
            root=feature_root,
        )
        label_path = label_dir / f"{match_id}.pt"
        if label_path.exists() and not overwrite:
            continue
        label_dir.mkdir(parents=True, exist_ok=True)
        intent_train_labels = build_intent_train_labels_from_source_actions(base_labels, source_labels)
        torch.save(intent_train_labels, label_path)


def save_augmented_blocks_from_existing_graphs(
    match: Match,
    labels_by_key: dict[tuple[str, str], torch.Tensor],
    resolved_actions_by_mode: dict[str, pd.DataFrame],
    return_type: str,
    match_id: str,
    feature_root: Path,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    graph_path = get_action_graph_dir(feature_root) / f"{match_id}.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Base action graphs not found at {graph_path}.")
    base_graphs = torch.load(graph_path, weights_only=False)

    for mode, resolved_actions in resolved_actions_by_mode.items():
        augmented_feature_dir = get_augmented_feature_dir(mode, root=feature_root)
        augmented_label_dir = get_augmented_label_dir(mode, root=feature_root)
        augmented_feature_path = augmented_feature_dir / f"{match_id}.pt"
        augmented_label_path = augmented_label_dir / f"{match_id}.pt"
        if augmented_feature_path.exists() and augmented_label_path.exists() and not overwrite:
            continue

        augmented_feature_dir.mkdir(parents=True, exist_ok=True)
        augmented_label_dir.mkdir(parents=True, exist_ok=True)
        match.actions = resolved_actions.copy()
        match.labels = labels_by_key[(mode, return_type)]
        match.graph_features_0 = list(base_graphs)
        augmented_graph_features, augmented_labels = augment_blocked_actions(match, verbose=verbose)
        torch.save(augmented_graph_features, augmented_feature_path)
        torch.save(augmented_labels, augmented_label_path)
        if verbose:
            print(f"Successfully saved {augmented_labels.shape[0]} augmented events for mode={mode}.")


def save_labels_only_artifacts(
    match: Match,
    resolved_actions_by_mode: dict[str, pd.DataFrame],
    labels_by_key: dict[tuple[str, str], torch.Tensor],
    intended_receiver_modes: list[str],
    return_types: list[str],
    match_id: str,
    feature_root: Path,
    feature_variant: str = "base",
    intent_train_label_source_mode: str | None = None,
    intent_train_label_source_return_type: str | None = None,
    augment_blocks_from_existing_graphs: bool = False,
    overwrite_labels: bool = False,
    verbose: bool = True,
) -> None:
    _validate_labels_only_base_artifacts(
        match_id,
        feature_root,
        feature_variant=feature_variant,
        intent_train_label_source_mode=intent_train_label_source_mode,
        intent_train_label_source_return_type=intent_train_label_source_return_type,
    )
    primary_mode = intended_receiver_modes[0]
    primary_return_type = return_types[0]
    bind_canonical_graph_context(
        match,
        resolved_actions_by_mode,
        labels_by_key,
        primary_mode,
        primary_return_type,
    )
    save_resolved_actions_if_missing(resolved_actions_by_mode, match_id, feature_root)
    if feature_variant == "intent_train_augmented":
        save_intent_train_labels_from_existing_source(
            labels_by_key,
            match_id,
            feature_root,
            intent_train_label_source_mode,
            intent_train_label_source_return_type,
            overwrite=overwrite_labels,
        )
        if verbose:
            print("Successfully saved labels-only intent-training labels.")
    else:
        save_action_labels_if_missing(labels_by_key, match_id, feature_root, overwrite=overwrite_labels)
        if augment_blocks_from_existing_graphs:
            save_augmented_blocks_from_existing_graphs(
                match,
                labels_by_key,
                resolved_actions_by_mode,
                primary_return_type,
                match_id,
                feature_root,
                verbose=verbose,
            )
        if verbose:
            print("Successfully saved labels-only action labels.")


def construct_intent_training_samples(
    match: Match,
    offsets: Tuple[int, ...] = INTENT_TRAIN_OFFSETS,
    extend: bool = True,
    verbose: bool = True,
    add_v_edge_features: bool = False,
    return_action_indices: bool = False,
) -> Tuple[List[Data], torch.Tensor] | Tuple[List[Data], torch.Tensor, list[int]]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    feature_dim = infer_node_feature_dim(extend)
    offsets = tuple(int(offset) for offset in offsets if int(offset) > 0)
    augmented_graphs: List[Data] = []
    augmented_labels: List[torch.Tensor] = []
    augmented_action_indices: list[int] = []
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

        base_graph = construct_graph_for_action(
            match,
            action_index,
            feature_variant="base",
            extend=extend,
            add_v_edge_features=add_v_edge_features,
        )
        if base_graph is None:
            continue

        augmented_graphs.append(base_graph)
        augmented_labels.append(label_row.clone())
        augmented_action_indices.append(action_index)

        snapshot = load_frame_snapshot(period_tracking, match.tracking, current_frame)
        phase_id = snapshot["phase_id"].iloc[0] if not snapshot.empty and "phase_id" in snapshot.columns else np.nan
        if pd.isna(phase_id) or int(phase_id) not in match.phases.index:
            continue

        phase_tracking = match.tracking[match.tracking["phase_id"] == int(phase_id)]
        for offset in offsets:
            prior_frame = resolve_prior_frame(phase_tracking, possessor, current_frame, offset)
            if prior_frame is None:
                continue

            graph = construct_graph_for_frame(
                match,
                prior_frame,
                possessor,
                period_tracking,
                feature_dim,
                extend=extend,
                add_v_edge_features=add_v_edge_features,
            )
            if graph is None:
                continue

            augmented_graphs.append(graph)
            augmented_labels.append(label_row.clone())
            augmented_action_indices.append(action_index)

    if augmented_labels:
        labels = torch.stack(augmented_labels, dim=0)
    else:
        labels = torch.empty((0, match.labels.shape[1]), dtype=match.labels.dtype)

    if return_action_indices:
        return augmented_graphs, labels, augmented_action_indices
    return augmented_graphs, labels


def augment_blocked_actions(
    match: Match,
    max_block_dist=5,
    max_block_angle=15,
    verbose: bool = True,
) -> Tuple[List[Data], torch.Tensor]:
    augmented_features = []
    augmented_labels = []

    action_indices = match.labels[:, 0].numpy().astype(int)
    tqdm_desc = "Augmenting features and labels"

    iterator = tqdm(action_indices, desc=tqdm_desc) if verbose else action_indices

    for i, action_index in enumerate(iterator):
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


def args_to_worker_dict(args: argparse.Namespace) -> dict[str, object]:
    return {
        "action_type": args.action_type,
        "split": args.split,
        "return_types": list(args.return_types),
        "post_action": bool(args.post_action),
        "augment_blocks": bool(args.augment_blocks),
        "feature_variant": args.feature_variant,
        "intended_receiver_model_id": args.intended_receiver_model_id,
        "labels_only": bool(args.labels_only),
        "intended_receiver_modes": list(args.intended_receiver_modes),
        "intent_train_label_source_mode": args.intent_train_label_source_mode,
        "intent_train_label_source_return_type": args.intent_train_label_source_return_type,
        "augment_blocks_from_existing_graphs": bool(args.augment_blocks_from_existing_graphs),
        "overwrite_labels": bool(args.overwrite_labels),
        "next_action_conditions_enabled": bool(args.next_action_conditions_enabled),
    }


def ensure_output_dirs(args: argparse.Namespace, feature_root: Path) -> FeatureGenerationPaths:
    label_dir: str | None = None
    post_feature_dir: str | None = None

    if args.action_type.startswith("shot"):
        feature_dir = feature_root / "augmented_shot_graphs"
        label_dir = str(feature_root / "augmented_shot_labels")
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    else:
        if args.feature_variant == "intent_train_augmented":
            feature_dir = get_action_graph_intent_train_dir(feature_root)
            label_dir_builder = get_intent_train_label_dir
        elif args.feature_variant == "success_intent":
            feature_dir = get_success_intent_graph_dir(feature_root)
            label_dir_builder = None
        else:
            feature_dir = get_action_graph_dir(feature_root)
            label_dir_builder = get_action_label_dir
        Path(feature_dir).mkdir(parents=True, exist_ok=True)
        if args.feature_variant == "success_intent":
            Path(get_success_intent_label_dir(root=feature_root)).mkdir(parents=True, exist_ok=True)
        else:
            for intended_receiver_mode in args.intended_receiver_modes:
                for return_type in args.return_types:
                    Path(
                        label_dir_builder(
                            return_type,
                            intended_receiver_mode=intended_receiver_mode,
                            root=feature_root,
                        )
                    ).mkdir(parents=True, exist_ok=True)

        if args.post_action:
            if args.feature_variant != "base":
                raise ValueError("Post-action features are only supported for base graphs.")
            post_feature_dir = str(get_post_action_graph_dir(feature_root))
            os.makedirs(post_feature_dir, exist_ok=True)

    if args.augment_blocks or args.augment_blocks_from_existing_graphs:
        for intended_receiver_mode in args.intended_receiver_modes:
            augmented_feature_dir = str(get_augmented_feature_dir(intended_receiver_mode, root=feature_root))
            augmented_label_dir = str(get_augmented_label_dir(intended_receiver_mode, root=feature_root))
            os.makedirs(augmented_feature_dir, exist_ok=True)
            os.makedirs(augmented_label_dir, exist_ok=True)

    return FeatureGenerationPaths(
        feature_dir=str(feature_dir),
        label_dir=label_dir,
        post_feature_dir=post_feature_dir,
    )


def _match_heading(task: MatchGenerationTask, match_lineup: pd.DataFrame) -> str:
    if match_lineup.empty:
        return f"\n[{task.match_index + 1}/{task.n_matches}] {task.match_id}"
    game_date = pd.to_datetime(match_lineup["game_date"].iloc[0]).date()
    match_name = " vs ".join(match_lineup["contestant_name"].dropna().unique())
    return f"\n[{task.match_index + 1}/{task.n_matches}] {task.match_id}: {match_name} on {game_date}"


def _save_success_intent_match(
    match: Match,
    match_id: str,
    feature_root: Path,
    feature_dir: str,
    return_types: list[str],
    show_progress: bool,
) -> MatchGenerationResult:
    resolved_actions, success_intent_labels = build_success_intent_labels_by_return(match, return_types)
    for labels in success_intent_labels.values():
        extract_sorted_action_indices(labels, context="Action labels")

    primary_return_type = return_types[0]
    match.actions = resolved_actions.copy()
    match.labels = success_intent_labels[primary_return_type]
    if show_progress:
        print("Constructing success-intent graph features for actions...")
    success_graphs = construct_graph_features(
        match,
        extend=False,
        post_action=False,
        feature_variant="success_intent",
        add_v_edge_features=True,
        verbose=show_progress,
    )
    invalid_graphs = count_invalid_graphs(success_graphs)
    valid_graphs = len(success_graphs) - invalid_graphs
    if valid_graphs == 0:
        raise ValueError("No usable success-intent graphs were constructed for this match.")
    torch.save(success_graphs, f"{feature_dir}/{match_id}.pt")
    label_dir = get_success_intent_label_dir(root=feature_root)
    torch.save(success_intent_labels[primary_return_type], f"{label_dir}/{match_id}.pt")
    if invalid_graphs and show_progress:
        print(f"  Skipped {invalid_graphs} corrupted action frames while building success-intent graphs.")
    if show_progress:
        print(f"Successfully saved success-intent graphs for {len(success_graphs)} events.")
    summary = f"saved {len(success_graphs)} success-intent graphs"
    if invalid_graphs:
        summary += f" ({invalid_graphs} invalid skipped)"
    return MatchGenerationResult(match_id=match_id, summary=summary, stats_by_mode={})


def _save_action_type_all_match(
    match: Match,
    match_id: str,
    feature_root: Path,
    paths: FeatureGenerationPaths,
    args: argparse.Namespace,
    show_progress: bool,
) -> MatchGenerationResult:
    if args.labels_only:
        _validate_labels_only_base_artifacts(
            match_id,
            feature_root,
            feature_variant=args.feature_variant,
            intent_train_label_source_mode=args.intent_train_label_source_mode,
            intent_train_label_source_return_type=args.intent_train_label_source_return_type,
        )
    resolved_actions_by_mode, labels_by_key, stats_by_mode = build_labels_by_mode_and_return(
        match,
        intended_receiver_modes=args.intended_receiver_modes,
        return_types=args.return_types,
        intended_receiver_model_id=args.intended_receiver_model_id,
        feature_root=feature_root,
        match_id=match_id,
        prefer_existing_resolved_actions=args.labels_only,
    )

    if args.labels_only:
        save_labels_only_artifacts(
            match,
            resolved_actions_by_mode,
            labels_by_key,
            args.intended_receiver_modes,
            args.return_types,
            match_id,
            feature_root,
            feature_variant=args.feature_variant,
            intent_train_label_source_mode=args.intent_train_label_source_mode,
            intent_train_label_source_return_type=args.intent_train_label_source_return_type,
            augment_blocks_from_existing_graphs=args.augment_blocks_from_existing_graphs,
            overwrite_labels=args.overwrite_labels,
            verbose=show_progress,
        )
        primary_mode = args.intended_receiver_modes[0]
        primary_return_type = args.return_types[0]
        label_count = int(labels_by_key[(primary_mode, primary_return_type)].shape[0])
        return MatchGenerationResult(
            match_id=match_id,
            summary=f"saved labels-only artifacts for {label_count} actions",
            stats_by_mode=stats_by_mode,
        )

    for mode, resolved_actions in resolved_actions_by_mode.items():
        resolved_action_path = get_resolved_action_path(
            match_id,
            intended_receiver_mode=mode,
            root=feature_root,
        )
        resolved_action_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_actions.to_parquet(resolved_action_path)

    primary_mode = args.intended_receiver_modes[0]
    primary_return_type = args.return_types[0]
    base_action_indices = bind_canonical_graph_context(
        match,
        resolved_actions_by_mode,
        labels_by_key,
        primary_mode,
        primary_return_type,
    )

    if args.feature_variant == "intent_train_augmented":
        if show_progress:
            print("Constructing intent-training augmentation graphs...")
        augmented_graphs, _, augmented_action_indices = construct_intent_training_samples(
            match,
            extend=True,
            add_v_edge_features=True,
            return_action_indices=True,
            verbose=show_progress,
        )
        if not augmented_graphs:
            raise ValueError("No usable intent-training samples were constructed for this match.")
        torch.save(augmented_graphs, f"{paths.feature_dir}/{match_id}.pt")
        for mode in args.intended_receiver_modes:
            for return_type in args.return_types:
                label_lookup = {
                    int(label[0].item()): label.clone()
                    for label in labels_by_key[(mode, return_type)]
                }
                augmented_labels = torch.stack(
                    [label_lookup[action_index] for action_index in augmented_action_indices],
                    dim=0,
                )
                label_dir = get_intent_train_label_dir(
                    return_type,
                    intended_receiver_mode=mode,
                    root=feature_root,
                )
                torch.save(augmented_labels, f"{label_dir}/{match_id}.pt")
        if show_progress:
            print(f"Successfully saved {len(augmented_graphs)} augmented intent samples.")
        return MatchGenerationResult(
            match_id=match_id,
            summary=f"saved {len(augmented_graphs)} augmented intent samples",
            stats_by_mode=stats_by_mode,
        )

    if show_progress:
        print("Constructing base graph features for actions...")
    match.graph_features_0 = construct_graph_features(
        match,
        extend=True,
        post_action=False,
        feature_variant="base",
        action_indices=base_action_indices,
        add_v_edge_features=True,
        verbose=show_progress,
    )
    invalid_graphs = count_invalid_graphs(match.graph_features_0)
    valid_graphs = len(match.graph_features_0) - invalid_graphs
    if valid_graphs == 0:
        raise ValueError("No usable action graphs were constructed for this match.")
    torch.save(match.graph_features_0, f"{paths.feature_dir}/{match_id}.pt")
    if invalid_graphs and show_progress:
        print(f"  Skipped {invalid_graphs} corrupted action frames while building graphs.")

    if args.post_action:
        if show_progress:
            print("Constructing base graph features for post-actions...")
        match.graph_features_1 = construct_graph_features(
            match,
            extend=True,
            post_action=True,
            feature_variant="base",
            action_indices=base_action_indices,
            add_v_edge_features=True,
            verbose=show_progress,
        )
        torch.save(match.graph_features_1, f"{paths.post_feature_dir}/{match_id}.pt")

    for mode in args.intended_receiver_modes:
        for return_type in args.return_types:
            label_dir = get_action_label_dir(
                return_type,
                intended_receiver_mode=mode,
                root=feature_root,
            )
            torch.save(labels_by_key[(mode, return_type)], f"{label_dir}/{match_id}.pt")

    if args.augment_blocks:
        for mode in args.intended_receiver_modes:
            match.actions = resolved_actions_by_mode[mode].copy()
            match.labels = labels_by_key[(mode, args.return_types[0])]
            augmented_graph_features, augmented_labels = augment_blocked_actions(match, verbose=show_progress)
            augmented_feature_dir = str(get_augmented_feature_dir(mode, root=feature_root))
            augmented_label_dir = str(get_augmented_label_dir(mode, root=feature_root))
            torch.save(augmented_graph_features, f"{augmented_feature_dir}/{match_id}.pt")
            torch.save(augmented_labels, f"{augmented_label_dir}/{match_id}.pt")
            if show_progress:
                print(f"Successfully saved {augmented_labels.shape[0]} augmented events for mode={mode}.")

    if show_progress:
        print(f"Successfully saved for {len(match.graph_features_0)} events.")
    summary = f"saved {len(match.graph_features_0)} base graphs"
    if invalid_graphs:
        summary += f" ({invalid_graphs} invalid skipped)"
    return MatchGenerationResult(match_id=match_id, summary=summary, stats_by_mode=stats_by_mode)


def _save_non_all_match(
    match: Match,
    match_id: str,
    paths: FeatureGenerationPaths,
    return_types: list[str],
    show_progress: bool,
) -> MatchGenerationResult:
    primary_return_type = return_types[0]
    match.labels = match.construct_labels(return_type=primary_return_type)
    if match.labels.numel() == 0:
        raise ValueError("No usable labels were constructed for this match.")

    action_indices = extract_sorted_action_indices(match.labels, context="Action labels")

    if show_progress:
        print("Constructing base graph features for actions...")
    match.graph_features_0 = construct_graph_features(
        match,
        extend=True,
        post_action=False,
        feature_variant="base",
        action_indices=action_indices,
        add_v_edge_features=True,
        verbose=show_progress,
    )
    valid_graphs = len(match.graph_features_0) - count_invalid_graphs(match.graph_features_0)
    if valid_graphs == 0:
        raise ValueError("No usable action graphs were constructed for this match.")
    torch.save(match.labels, f"{paths.label_dir}/{match_id}.pt")
    torch.save(match.graph_features_0, f"{paths.feature_dir}/{match_id}.pt")
    return MatchGenerationResult(
        match_id=match_id,
        summary=f"saved {len(match.graph_features_0)} action graphs",
        stats_by_mode={},
    )


def process_match_generation_task(task: MatchGenerationTask) -> MatchGenerationResult:
    configure_worker_thread_limit(int(task.worker_thread_limit))
    args = SimpleNamespace(**task.args_dict)
    feature_root = Path(task.feature_root)
    events = pd.read_csv(f"data/event_synced/{task.match_id}.csv", header=0, parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(f"data/tracking_processed/{task.match_id}.parquet")
    match = build_match_for_feature_generation(events, tracking, task.match_lineup, args)
    if task.show_progress:
        print(_match_heading(task, task.match_lineup))

    if args.action_type == "all":
        if args.feature_variant == "success_intent":
            return _save_success_intent_match(
                match,
                task.match_id,
                feature_root,
                task.paths.feature_dir,
                args.return_types,
                task.show_progress,
            )
        return _save_action_type_all_match(
            match,
            task.match_id,
            feature_root,
            task.paths,
            args,
            task.show_progress,
        )

    return _save_non_all_match(
        match,
        task.match_id,
        task.paths,
        args.return_types,
        task.show_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, required=False, default="all", choices=["all", "shot_augment"])
    parser.add_argument("--split", type=str, required=False, default="train", choices=["train", "test"])
    parser.add_argument(
        "--return_type",
        type=str,
        action="append",
        default=None,
        help=(
            "Way of defining future returns. Repeat the flag to generate multiple return types in one run, "
            "including disc_<gamma>_skip1, disc_max_<gamma>[_skip1], next_<N>_skip1, "
            "and in_<N> for xt/goal_distance/epv training."
        ),
    )
    parser.add_argument("--post_action", action="store_true", default=False, help="construct post-action features")
    parser.add_argument("--augment_blocks", action="store_true", default=False)
    parser.add_argument(
        "--feature_variant",
        type=str,
        default="base",
        choices=["base", "intent_train_augmented", "success_intent"],
    )
    parser.add_argument(
        "--intended-receiver-model-id",
        type=str,
        default=None,
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        default=False,
        help="Generate labels/resolved-action artifacts without rebuilding graph tensors.",
    )
    parser.add_argument(
        "--only-intended-receiver-mode",
        action="append",
        default=None,
        help="Restrict generation to one intended-receiver mode. Repeat to include multiple modes.",
    )
    parser.add_argument(
        "--intent-train-label-source-mode",
        default=None,
        help="Existing intent-train label mode whose action-index sequence should be reused in --labels-only mode.",
    )
    parser.add_argument(
        "--intent-train-label-source-return-type",
        default=None,
        help="Existing intent-train label return_type whose action-index sequence should be reused in --labels-only mode.",
    )
    parser.add_argument(
        "--augment-blocks-from-existing-graphs",
        action="store_true",
        default=False,
        help="Build augmented blocked-action artifacts from copied base action graphs in --labels-only mode.",
    )
    parser.add_argument(
        "--overwrite-labels",
        action="store_true",
        default=False,
        help="Overwrite existing label tensors in --labels-only mode.",
    )
    next_action_group = parser.add_mutually_exclusive_group()
    next_action_group.add_argument(
        "--next-action-conditions-on",
        dest="next_action_conditions_enabled",
        action="store_true",
        default=True,
        help="Keep the current pass/cross next-action inclusion conditions enabled.",
    )
    next_action_group.add_argument(
        "--next-action-conditions-off",
        dest="next_action_conditions_enabled",
        action="store_false",
        help="Disable pass/cross next-action inclusion conditions while keeping frame requirements.",
    )
    parser.add_argument(
        "--num-workers",
        default="1",
        help="Number of match worker processes to use, or 'auto' to resolve automatically.",
    )
    parser.add_argument(
        "--worker-thread-limit",
        type=int,
        default=1,
        help="Thread limit propagated to each worker process for BLAS/OpenMP-backed libraries.",
    )
    args, _ = parser.parse_known_args()
    args.return_types = resolve_requested_return_types(args.return_type)
    if args.only_intended_receiver_mode:
        args.intended_receiver_modes = []
        seen_modes: set[str] = set()
        for requested_mode in args.only_intended_receiver_mode:
            mode = validate_intended_receiver_mode(requested_mode)
            if mode not in seen_modes:
                seen_modes.add(mode)
                args.intended_receiver_modes.append(mode)
    else:
        args.intended_receiver_modes = resolve_generation_intended_receiver_modes(args.intended_receiver_model_id)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.labels_only and args.post_action:
        raise ValueError("--labels-only cannot be combined with --post_action.")
    if args.labels_only and args.feature_variant == "success_intent":
        raise ValueError("--labels-only is not supported for feature_variant='success_intent'.")
    if args.augment_blocks_from_existing_graphs and not args.labels_only:
        raise ValueError("--augment-blocks-from-existing-graphs requires --labels-only.")
    if args.overwrite_labels and not args.labels_only:
        raise ValueError("--overwrite-labels requires --labels-only.")
    if args.augment_blocks_from_existing_graphs and args.feature_variant != "base":
        raise ValueError("--augment-blocks-from-existing-graphs is only supported for base labels.")
    if args.worker_thread_limit < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    if args.labels_only and args.feature_variant == "intent_train_augmented":
        if not args.intent_train_label_source_mode or not args.intent_train_label_source_return_type:
            raise ValueError(
                "--labels-only intent-train generation requires --intent-train-label-source-mode and "
                "--intent-train-label-source-return-type."
            )
        args.intent_train_label_source_mode = validate_intended_receiver_mode(args.intent_train_label_source_mode)
        args.intent_train_label_source_return_type = resolve_requested_return_types(
            [args.intent_train_label_source_return_type]
        )[0]


def aggregate_mode_stats(intended_receiver_modes: list[str]) -> dict[str, dict[str, int]]:
    return {
        mode: {
            "failed_passes_total": 0,
            "failed_passes_labeled": 0,
            "failed_passes_missing_endpoint": 0,
            "model_failed_passes_total": 0,
            "model_failed_passes_scored": 0,
            "model_angle_only_fallbacks": 0,
        }
        for mode in intended_receiver_modes
    }


def accumulate_mode_stats(
    aggregate_stats: dict[str, dict[str, int]],
    stats_by_mode: dict[str, dict[str, int]],
) -> None:
    for mode, mode_stats in stats_by_mode.items():
        if mode not in aggregate_stats:
            continue
        for key in aggregate_stats[mode]:
            aggregate_stats[mode][key] += int(mode_stats.get(key, 0) or 0)


def emit_model_fallback_message(
    stats_by_mode: dict[str, dict[str, int]],
    *,
    writer=print,
) -> None:
    mode_stats = stats_by_mode.get(INTENDED_RECEIVER_MODE_MODEL)
    if not mode_stats or not mode_stats.get("model_failed_passes_total", 0):
        return
    writer(
        "  Intended-receiver model fallback: "
        f"{int(mode_stats.get('model_angle_only_fallbacks', 0))} / "
        f"{int(mode_stats.get('model_failed_passes_total', 0))} failed passes used angle_only."
    )


def build_match_tasks(
    match_ids: list[str],
    lineups: pd.DataFrame,
    args: argparse.Namespace,
    feature_root: Path,
    paths: FeatureGenerationPaths,
    *,
    num_workers: int,
) -> list[MatchGenerationTask]:
    args_dict = args_to_worker_dict(args)
    tasks: list[MatchGenerationTask] = []
    for i, match_id in enumerate(match_ids):
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()
        tasks.append(
            MatchGenerationTask(
                match_id=str(match_id),
                match_index=i,
                n_matches=len(match_ids),
                match_lineup=match_lineup,
                args_dict=args_dict,
                feature_root=str(feature_root),
                paths=paths,
                show_progress=num_workers == 1,
                worker_thread_limit=int(args.worker_thread_limit),
            )
        )
    return tasks


def run_match_generation(
    tasks: list[MatchGenerationTask],
    *,
    num_workers: int,
    intended_receiver_modes: list[str],
) -> tuple[int, list[dict[str, str]], dict[str, dict[str, int]]]:
    successful_matches = 0
    skipped_matches: list[dict[str, str]] = []
    aggregate_stats = aggregate_mode_stats(intended_receiver_modes)

    if num_workers == 1:
        for task in tasks:
            try:
                result = process_match_generation_task(task)
                successful_matches += 1
                accumulate_mode_stats(aggregate_stats, result.stats_by_mode)
                emit_model_fallback_message(result.stats_by_mode)
            except Exception as exc:
                error_summary = summarize_exception(exc)
                skipped_matches.append({"match_id": task.match_id, "error": error_summary})
                print(f"  SKIP {task.match_id}: {error_summary}")
        return successful_matches, skipped_matches, aggregate_stats

    progress = tqdm(total=len(tasks), desc="compute matches")
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=configure_worker_thread_limit,
            initargs=(int(tasks[0].worker_thread_limit) if tasks else 1,),
        ) as executor:
            future_to_task = {executor.submit(process_match_generation_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    successful_matches += 1
                    accumulate_mode_stats(aggregate_stats, result.stats_by_mode)
                    progress.write(f"  DONE {result.match_id}: {result.summary}")
                    emit_model_fallback_message(result.stats_by_mode, writer=progress.write)
                except Exception as exc:
                    error_summary = summarize_exception(exc)
                    skipped_matches.append({"match_id": task.match_id, "error": error_summary})
                    progress.write(f"  SKIP {task.match_id}: {error_summary}")
                finally:
                    progress.update(1)
    finally:
        progress.close()

    return successful_matches, skipped_matches, aggregate_stats


def main() -> None:
    args = parse_args()
    validate_args(args)
    num_workers = resolve_num_workers(args.num_workers)
    configure_worker_thread_limit(int(args.worker_thread_limit))
    feature_root = get_feature_run_root(args.run_id) if args.run_id else FEATURE_DIR

    if args.action_type.startswith("shot"):
        if args.feature_variant != "base":
            raise ValueError("Intent-training augmentation is only supported for action_type=all.")
        args.action_type = "shot_augment"

    paths = ensure_output_dirs(args, feature_root)

    lineups = pd.read_parquet("data/lineup/line_up.parquet")
    lineups["game_id"] = lineups["stats_perform_match_id"]
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])

    if args.split == "train":
        match_ids, _ = load_base_splits()
    else:
        _, match_ids = load_base_splits()

    tasks = build_match_tasks(
        [str(match_id) for match_id in match_ids],
        lineups,
        args,
        feature_root,
        paths,
        num_workers=num_workers,
    )
    successful_matches, skipped_matches, aggregate_stats = run_match_generation(
        tasks,
        num_workers=num_workers,
        intended_receiver_modes=args.intended_receiver_modes,
    )

    if successful_matches == 0:
        raise RuntimeError(f"No usable matches were processed for split={args.split}, feature_variant={args.feature_variant}.")
    if skipped_matches:
        print(f"Skipped {len(skipped_matches)} matches while generating graph features.")
        for item in skipped_matches[:10]:
            print(f"  {item['match_id']}: {item['error']}")
        if len(skipped_matches) > 10:
            print(f"  ... and {len(skipped_matches) - 10} more")
    if INTENDED_RECEIVER_MODE_MODEL in aggregate_stats and aggregate_stats[INTENDED_RECEIVER_MODE_MODEL]["model_failed_passes_total"] > 0:
        print(
            "Intended-receiver model fallback summary: "
            f"{aggregate_stats[INTENDED_RECEIVER_MODE_MODEL]['model_angle_only_fallbacks']} / "
            f"{aggregate_stats[INTENDED_RECEIVER_MODE_MODEL]['model_failed_passes_total']} "
            "failed passes used angle_only."
        )


if __name__ == "__main__":
    freeze_support()
    main()
