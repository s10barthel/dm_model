import argparse
import os
import re
import sys
from datetime import datetime
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
    POST_ACTION_GRAPH_DIR,
    INTENT_TRAIN_OFFSETS,
    load_base_splits,
)


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
    assert set(players) - {"home_goal", "away_goal"} == set(active_players)

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


def infer_node_feature_dim(extend: bool = True) -> int:
    return 25 if extend else 19


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


def construct_graph_for_frame(
    match: Match,
    frame: int,
    possessor: str,
    period_tracking: pd.DataFrame,
    feature_dim: int,
    extend: bool = True,
    rotate_to_ltr: bool = True,
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

    return Data(x=node_attr, edge_index=edge_index.clone(), edge_attr=edge_attr)


def construct_graph_features(
    match: Match,
    extend=True,
    post_action=False,
    verbose=True,
) -> List[Data]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    if post_action:
        match.actions = match.label_post_actions(match.actions)

    feature_graphs: List[Data] = []
    feature_dim = infer_node_feature_dim(extend)

    for period in match.events["period_id"].unique():
        period_actions: pd.DataFrame = match.actions[match.actions["period_id"] == period]
        period_tracking: pd.DataFrame = match.tracking[match.tracking["period_id"] == period]
        action_indices = np.intersect1d(period_actions.index, match.labels[:, 0].long().numpy())
        iterator = tqdm(action_indices, desc=f"Period {period}") if verbose else action_indices

        for i in iterator:
            if post_action:
                frame = period_actions.at[i, "end_frame_id"]
                possessor = period_actions.at[i, "end_player_id"]
            else:
                frame = period_actions.at[i, "frame_id"]
                possessor = period_actions.at[i, "object_id"]

            graph = construct_graph_for_frame(match, frame, possessor, period_tracking, feature_dim, extend=extend)
            feature_graphs.append(graph)

    return feature_graphs


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

        base_graph = construct_graph_for_frame(match, current_frame, possessor, period_tracking, feature_dim, extend=extend)
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

    return augmented_graphs, torch.stack(augmented_labels, dim=0)


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
    parser.add_argument("--feature_variant", type=str, default="base", choices=["base", "intent_train_augmented"])
    args, _ = parser.parse_known_args()

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
            feature_dir = str(ACTION_GRAPH_INTENT_TRAIN_DIR)
            label_dir = f"data/ajax/features/action_labels_intent_train_{args.return_type}"
        else:
            feature_dir = str(ACTION_GRAPH_DIR)
            label_dir = f"data/ajax/features/action_labels_{args.return_type}"
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if args.post_action:
            if args.feature_variant != "base":
                raise ValueError("Post-action features are only supported for base graphs.")
            post_feature_dir = str(POST_ACTION_GRAPH_DIR)
            os.makedirs(post_feature_dir, exist_ok=True)

    if args.augment_blocks:
        augmented_feature_dir = "data/ajax/features/augmented_graphs"
        augmented_label_dir = "data/ajax/features/augmented_labels"
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

    for i, match_id in enumerate(match_ids):
        events = pd.read_csv(f"data/ajax/event_synced/{match_id}.csv", header=0, parse_dates=["utc_timestamp"])
        tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id]

        match = Match(events, tracking, match_lineup, args.action_type, include_goals=True)
        match_date = match_dates[match_id].date()
        match_name = " vs ".join(match_lineup["contestant_name"].unique())
        print(f"\n[{i+1}/{n_matches}] {match_id}: {match_name} on {match_date}")

        if args.return_type.startswith("disc"):
            gamma = float(args.return_type.split("_")[-1])
            match.labels = match.construct_labels(discount_xg=True, gamma=gamma)
        if args.return_type.startswith("next"):
            lookahead_len = int(args.return_type.split("_")[-1])
            match.labels = match.construct_labels(discount_xg=False, lookahead_len=lookahead_len)

        action_indices = match.labels[:, 0].numpy().astype(int)
        assert np.all(np.sort(action_indices) == action_indices)
        if args.feature_variant == "intent_train_augmented":
            print("Constructing intent-training augmentation graphs...")
            augmented_graphs, augmented_labels = construct_intent_training_samples(match, extend=True)
            torch.save(augmented_labels, f"{label_dir}/{match_id}.pt")
            torch.save(augmented_graphs, f"{feature_dir}/{match_id}.pt")
            print(f"Successfully saved {len(augmented_graphs)} augmented intent samples.")
        else:
            torch.save(match.labels, f"{label_dir}/{match_id}.pt")

            print("Constructing base graph features for actions...")
            match.graph_features_0 = construct_graph_features(
                match,
                extend=True,
                post_action=False,
            )
            torch.save(match.graph_features_0, f"{feature_dir}/{match_id}.pt")

            if args.post_action:
                print("Constructing base graph features for post-actions...")
                match.graph_features_1 = construct_graph_features(
                    match,
                    extend=True,
                    post_action=True,
                )
                torch.save(match.graph_features_1, f"{post_feature_dir}/{match_id}.pt")

            print(f"Successfully saved for {match.labels.shape[0]} events.")

        if args.augment_blocks:
            augmented_graph_features, augmented_labels = augment_blocked_actions(match)
            torch.save(augmented_graph_features, f"{augmented_feature_dir}/{match_id}.pt")
            torch.save(augmented_labels, f"{augmented_label_dir}/{match_id}.pt")
            print(f"Successfully saved for {augmented_labels.shape[0]} augmented events.")
