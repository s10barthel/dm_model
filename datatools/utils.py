import re
from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdp import rdp
from scipy.spatial import Delaunay
from shapely import vectorized
from shapely.geometry import Point, Polygon
from torch_geometric.data import Batch, Data

from datatools import config
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    INTENDED_RECEIVER_MODE_ANGLE_ONLY,
    INTENDED_RECEIVER_MODE_ORIGINAL,
)


def calc_dist(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray):
    dist_x = (x - ref_x).astype(float) if isinstance(x, np.ndarray) else x - ref_x
    dist_y = (y - ref_y).astype(float) if isinstance(y, np.ndarray) else y - ref_y
    return dist_x, dist_y, np.sqrt(dist_x**2 + dist_y**2)


def calc_angle(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray = None,
    cy: np.ndarray = None,
    eps: float = 1e-6,
) -> np.ndarray:
    if cx is None or cy is None:
        # Calculate angles between the vectors a and b
        a_len = np.sqrt(ax**2 + ay**2) + eps
        b_len = np.sqrt(bx**2 + by**2) + eps
        cos = np.clip((ax * bx + ay * by) / (a_len * b_len), -1, 1)

    else:
        # Calculate angles between the lines AB and AC
        ab_x = (bx - ax).astype(float)
        ab_y = (by - ay).astype(float)
        ab_len = np.sqrt(ab_x**2 + ab_y**2) + eps

        ac_x = (cx - ax).astype(float)
        ac_y = (cy - ay).astype(float)
        ac_len = np.sqrt(ac_x**2 + ac_y**2) + eps

        cos = np.clip((ab_x * ac_x + ab_y * ac_y) / (ab_len * ac_len), -1, 1)

    return np.arccos(cos)


def downscale_closer_candidates(dists: np.ndarray) -> np.ndarray:
    out = np.empty_like(dists)
    out[dists < -10] = 0.5
    out[(dists >= -10) & (dists < 0)] = dists[(dists >= -10) & (dists < 0)] / 20 + 1
    out[dists >= 0] = 1.0
    return out


def resolve_episode_end_frame(tracking: pd.DataFrame, frame: int) -> int | None:
    if frame not in tracking.index or "episode_id" not in tracking.columns:
        return None

    episode_id = tracking.at[frame, "episode_id"]
    if pd.isna(episode_id):
        return None

    episode_frames = tracking.index[(tracking["episode_id"] == episode_id) & (tracking["ball_state"] == "alive")]
    if len(episode_frames) == 0:
        return None
    return int(episode_frames.max())


def resolve_ball_endpoint(
    tracking: pd.DataFrame,
    preferred_frame: int,
    fallback_frame: int | None,
    fallback_xy: tuple[float, float],
) -> tuple[int | None, float, float]:
    frame_candidates = [preferred_frame]
    if fallback_frame is not None and fallback_frame != preferred_frame:
        frame_candidates.append(fallback_frame)

    for frame in frame_candidates:
        if frame is None or frame not in tracking.index:
            continue
        snapshot = tracking.loc[frame]
        ball_x = snapshot.get("ball_x", np.nan)
        ball_y = snapshot.get("ball_y", np.nan)
        if not pd.isna(ball_x) and not pd.isna(ball_y):
            return int(frame), float(ball_x), float(ball_y)

    return None, float(fallback_xy[0]), float(fallback_xy[1])


# Identify whether the shot trajectory is erroneous and the shot hits a post
def is_shot_anomaly(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    event_index: int,
    min_segment_len: float = 2.0,
    min_sim: float = 0.7,
    max_frames: int = 25,
) -> Tuple[bool, bool]:
    frame = events.at[event_index, "frame_id"]
    receive_frame = events.at[event_index, "receive_frame_id"]

    if pd.isna(frame) or pd.isna(receive_frame):
        return False, False

    frame = int(frame)
    receive_frame = int(receive_frame)
    if receive_frame <= frame:
        return False, False

    ball_xy = tracking.loc[frame:receive_frame, ["ball_x", "ball_y"]].astype(float).dropna()
    if len(ball_xy) < 2:
        return False, False

    try:
        simplified = np.asarray(rdp(ball_xy.to_numpy(), epsilon=0.5), dtype=float)
    except Exception:
        return False, False

    simplified = np.atleast_2d(simplified)
    if simplified.ndim != 2 or simplified.shape[0] < 2 or simplified.shape[1] != 2:
        return False, False

    dirs = np.diff(simplified, axis=0)
    if dirs.ndim != 2 or dirs.shape[0] == 0:
        return False, False

    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    valid_mask = np.array((norms > min_segment_len).flatten().tolist() + [True])
    simplified = simplified[valid_mask]
    if simplified.ndim != 2 or simplified.shape[0] < 2:
        return False, False

    dirs = np.diff(simplified, axis=0)
    if dirs.ndim != 2 or dirs.shape[0] == 0:
        return False, False

    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_norm = dirs / (norms + 1e-8)

    woodwork = False

    for i in range(len(dirs_norm) - 1):
        sim = np.dot(dirs_norm[i], dirs_norm[i + 1])

        if sim < min_sim:
            goalpost_dist = np.linalg.norm(config.GOAL_XY - simplified[[i + 1]], axis=1)

            if goalpost_dist.min() < 1:
                woodwork = True
            elif receive_frame - frame > max_frames:
                return True, woodwork

    return False, woodwork


# To make the attacking team always play from right to left (not needed for the current dataset)
def rotate_events_for_xg(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    for i in events["period_id"].unique():
        shots = events[(events["period_id"] == i) & events["spadl_type"].isin(config.SHOT)]

        if not shots.empty:
            home_shot_x = shots.loc[shots["object_id"].str.startswith("home"), "start_x"].mean()
            if home_shot_x > config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_shot_x = shots.loc[shots["object_id"].str.startswith("away"), "start_y"].mean()
            if away_shot_x > config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

        else:
            gk_events = events[(events["period_id"] == i) & (events["advanced_position"] == "goal_keeper")]

            home_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("home"), "start_x"].mean()
            if home_gk_x < config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("away"), "start_x"].mean()
            if away_gk_x < config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

    return events


def sanitize_expected_goal(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    if "expected_goal" not in events.columns:
        events["expected_goal"] = np.nan
        return events

    events["expected_goal"] = pd.to_numeric(events["expected_goal"], errors="coerce")
    if "spadl_type" in events.columns:
        shot_mask = events["spadl_type"].isin(config.SHOT)
        events.loc[~shot_mask, "expected_goal"] = np.nan

    return events


def label_future_max_value(
    events: pd.DataFrame,
    value_col: str,
    scores_col: str,
    concedes_col: str,
    lookahead_len: int = 5,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()
    eligible_types = eligible_types or tuple(config.XT_ACTION_TYPES)

    if value_col not in events.columns:
        events[value_col] = np.nan
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    events[scores_col] = 0.0
    events[concedes_col] = 0.0

    if events.empty or "spadl_type" not in events.columns or "object_id" not in events.columns:
        return events

    teams = events["object_id"].astype(str).str[:4]
    eligible_mask = events["spadl_type"].isin(eligible_types) & events[value_col].notna() & teams.isin(["home", "away"])
    eligible_positions = np.flatnonzero(eligible_mask.to_numpy())

    if len(eligible_positions) == 0:
        return events

    value_array = events[value_col].to_numpy(dtype=float)
    team_values = teams.to_numpy(dtype=object)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)

    for row_pos in range(len(events)):
        future_start = np.searchsorted(eligible_positions, row_pos + 1, side="left")
        future_positions = eligible_positions[future_start : future_start + lookahead_len]
        if len(future_positions) == 0:
            continue
        if skip_first and not shot_array[int(future_positions[0])]:
            future_positions = future_positions[1:]
        if len(future_positions) == 0:
            continue

        team_i = team_values[row_pos]
        future_teams = team_values[future_positions]
        future_values = value_array[future_positions]

        teammate_values = future_values[future_teams == team_i]
        opponent_values = future_values[future_teams != team_i]

        if teammate_values.size:
            events.iat[row_pos, events.columns.get_loc(scores_col)] = float(np.nanmax(teammate_values))
        if opponent_values.size:
            events.iat[row_pos, events.columns.get_loc(concedes_col)] = float(np.nanmax(opponent_values))

    return events


def label_nth_future_state_value(
    events: pd.DataFrame,
    value_col: str,
    scores_col: str,
    concedes_col: str,
    action_offset: int = 5,
    eligible_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    events = events.copy()
    eligible_types = eligible_types or tuple(config.XT_ACTION_TYPES)

    if value_col not in events.columns:
        events[value_col] = np.nan
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    events[scores_col] = 0.0
    events[concedes_col] = 0.0

    if events.empty or "spadl_type" not in events.columns or "object_id" not in events.columns:
        return events

    teams = events["object_id"].astype(str).str[:4]
    eligible_mask = events["spadl_type"].isin(eligible_types) & events[value_col].notna() & teams.isin(["home", "away"])
    eligible_positions = np.flatnonzero(eligible_mask.to_numpy())

    if len(eligible_positions) == 0:
        return events

    score_loc = events.columns.get_loc(scores_col)
    concede_loc = events.columns.get_loc(concedes_col)
    team_values = teams.to_numpy(dtype=object)
    value_array = events[value_col].to_numpy(dtype=float)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)

    for row_pos in range(len(events)):
        future_start = np.searchsorted(eligible_positions, row_pos + 1, side="left")
        future_positions = eligible_positions[future_start:]
        if len(future_positions) == 0:
            continue

        if len(future_positions) >= action_offset:
            search_positions = future_positions[:action_offset]
        else:
            search_positions = future_positions

        shot_positions = search_positions[shot_array[search_positions]]
        if len(shot_positions) > 0:
            selected_pos = int(shot_positions[0])
        elif len(future_positions) >= action_offset:
            selected_pos = int(future_positions[action_offset - 1])
        else:
            continue

        candidate = value_array[selected_pos]
        if not np.isfinite(candidate):
            continue

        if team_values[selected_pos] == team_values[row_pos]:
            events.iat[row_pos, score_loc] = float(candidate)
            events.iat[row_pos, concede_loc] = 0.0
        else:
            events.iat[row_pos, score_loc] = 0.0
            events.iat[row_pos, concede_loc] = float(candidate)

    return events


def _should_stop_discount_scan(events: pd.DataFrame, row_pos: int, period_i: Any, goal_array: np.ndarray) -> bool:
    if bool(goal_array[row_pos]):
        return True
    if row_pos + 1 < len(events) and events.iat[row_pos + 1, events.columns.get_loc("period_id")] != period_i:
        return True
    if row_pos + 1 < len(events) and events.iat[row_pos + 1, events.columns.get_loc("spadl_type")] == "goalkick":
        return True
    return False


def label_discounted_goal_returns(
    events: pd.DataFrame,
    gamma: float = 0.95,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()

    expected_goal_source = events["expected_goal"] if "expected_goal" in events.columns else pd.Series(0.0, index=events.index)
    expected_goals = pd.to_numeric(expected_goal_source, errors="coerce").fillna(0)
    teams = events["object_id"].astype(str).str[:4]
    goal_array = ((expected_goals > 0) & events["success"].fillna(False).astype(bool)).to_numpy(dtype=bool)
    team_values = teams.to_numpy(dtype=object)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)

    events["goal"] = goal_array
    events["scores"] = 0.0
    events["concedes"] = 0.0

    if events.empty:
        return events

    score_loc = events.columns.get_loc("scores")
    concede_loc = events.columns.get_loc("concedes")
    period_loc = events.columns.get_loc("period_id")

    for row_pos in range(len(events)):
        period_i = events.iat[row_pos, period_loc]
        team_i = team_values[row_pos]
        best_score = 0.0
        best_concede = 0.0
        first_future_seen = False

        for future_pos in range(row_pos, len(events)):
            skip_future = False
            if future_pos > row_pos and not first_future_seen:
                first_future_seen = True
                skip_future = skip_first and not shot_array[future_pos]

            if goal_array[future_pos] and not skip_future:
                weight = gamma ** (future_pos - row_pos)
                if team_values[future_pos] == team_i:
                    best_score = max(best_score, weight)
                else:
                    best_concede = max(best_concede, weight)

            if _should_stop_discount_scan(events, future_pos, period_i, goal_array):
                break

        events.iat[row_pos, score_loc] = float(best_score)
        events.iat[row_pos, concede_loc] = float(best_concede)

    return events


def label_discounted_future_sum_value(
    events: pd.DataFrame,
    value_col: str,
    scores_col: str,
    concedes_col: str,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()
    eligible_types = eligible_types or tuple(config.XT_ACTION_TYPES)

    if value_col not in events.columns:
        events[value_col] = np.nan
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    events[scores_col] = 0.0
    events[concedes_col] = 0.0

    if events.empty or "spadl_type" not in events.columns or "object_id" not in events.columns:
        return events

    expected_goal_source = events["expected_goal"] if "expected_goal" in events.columns else pd.Series(0.0, index=events.index)
    expected_goals = pd.to_numeric(expected_goal_source, errors="coerce").fillna(0)
    teams = events["object_id"].astype(str).str[:4]
    goal_array = ((expected_goals > 0) & events["success"].fillna(False).astype(bool)).to_numpy(dtype=bool)
    team_values = teams.to_numpy(dtype=object)
    value_array = events[value_col].to_numpy(dtype=float)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)
    eligible_array = (
        events["spadl_type"].isin(eligible_types).to_numpy(dtype=bool)
        & np.isfinite(value_array)
        & np.isin(team_values, ["home", "away"])
    )

    score_loc = events.columns.get_loc(scores_col)
    concede_loc = events.columns.get_loc(concedes_col)
    period_loc = events.columns.get_loc("period_id")

    for row_pos in range(len(events)):
        period_i = events.iat[row_pos, period_loc]
        team_i = team_values[row_pos]
        score_sum = 0.0
        concede_sum = 0.0
        eligible_rank = 0
        first_eligible_seen = False

        if _should_stop_discount_scan(events, row_pos, period_i, goal_array):
            events.iat[row_pos, score_loc] = float(score_sum)
            events.iat[row_pos, concede_loc] = float(concede_sum)
            continue

        for future_pos in range(row_pos + 1, len(events)):
            if eligible_array[future_pos]:
                skip_current = False
                if not first_eligible_seen:
                    first_eligible_seen = True
                    skip_current = skip_first and not shot_array[future_pos]

                if not skip_current:
                    weight = gamma ** eligible_rank
                    candidate = weight * value_array[future_pos]
                    if team_values[future_pos] == team_i:
                        score_sum += candidate
                    else:
                        concede_sum += candidate
                    eligible_rank += 1

            if _should_stop_discount_scan(events, future_pos, period_i, goal_array):
                break

        events.iat[row_pos, score_loc] = float(score_sum)
        events.iat[row_pos, concede_loc] = float(concede_sum)

    return events


def label_discounted_future_probability_value(
    events: pd.DataFrame,
    value_col: str,
    scores_col: str,
    concedes_col: str,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()
    eligible_types = eligible_types or tuple(config.XT_ACTION_TYPES)

    if value_col not in events.columns:
        events[value_col] = np.nan
    events[value_col] = pd.to_numeric(events[value_col], errors="coerce")
    events[scores_col] = 0.0
    events[concedes_col] = 0.0

    if events.empty or "spadl_type" not in events.columns or "object_id" not in events.columns:
        return events

    expected_goal_source = events["expected_goal"] if "expected_goal" in events.columns else pd.Series(0.0, index=events.index)
    expected_goals = pd.to_numeric(expected_goal_source, errors="coerce").fillna(0)
    teams = events["object_id"].astype(str).str[:4]
    goal_array = ((expected_goals > 0) & events["success"].fillna(False).astype(bool)).to_numpy(dtype=bool)
    team_values = teams.to_numpy(dtype=object)
    value_array = events[value_col].to_numpy(dtype=float)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)
    eligible_array = (
        events["spadl_type"].isin(eligible_types).to_numpy(dtype=bool)
        & np.isfinite(value_array)
        & np.isin(team_values, ["home", "away"])
    )

    score_loc = events.columns.get_loc(scores_col)
    concede_loc = events.columns.get_loc(concedes_col)
    period_loc = events.columns.get_loc("period_id")

    for row_pos in range(len(events)):
        period_i = events.iat[row_pos, period_loc]
        team_i = team_values[row_pos]
        prob_not_scoring = 1.0
        prob_not_conceding = 1.0
        first_eligible_seen = False

        if _should_stop_discount_scan(events, row_pos, period_i, goal_array):
            events.iat[row_pos, score_loc] = 0.0
            events.iat[row_pos, concede_loc] = 0.0
            continue

        for future_pos in range(row_pos + 1, len(events)):
            if eligible_array[future_pos]:
                skip_current = False
                if not first_eligible_seen:
                    first_eligible_seen = True
                    skip_current = skip_first and not shot_array[future_pos]

                if not skip_current:
                    value = float(np.clip(value_array[future_pos], 0.0, 1.0))
                    candidate = (gamma ** (future_pos - row_pos)) * value
                    if team_values[future_pos] == team_i:
                        prob_not_scoring *= 1.0 - candidate
                    else:
                        prob_not_conceding *= 1.0 - candidate

            if _should_stop_discount_scan(events, future_pos, period_i, goal_array):
                break

        events.iat[row_pos, score_loc] = float(1.0 - prob_not_scoring)
        events.iat[row_pos, concede_loc] = float(1.0 - prob_not_conceding)

    return events


def label_discounted_future_max_value(
    events: pd.DataFrame,
    value_col: str,
    scores_col: str,
    concedes_col: str,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_discounted_future_sum_value(
        events,
        value_col=value_col,
        scores_col=scores_col,
        concedes_col=concedes_col,
        gamma=gamma,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_xt_returns(
    events: pd.DataFrame,
    lookahead_len: int = 5,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_future_max_value(
        events,
        value_col="xT",
        scores_col="scores_xT",
        concedes_col="concedes_xT",
        lookahead_len=lookahead_len,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_discounted_xt_returns(
    events: pd.DataFrame,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_discounted_future_probability_value(
        events,
        value_col="xT",
        scores_col="scores_xT",
        concedes_col="concedes_xT",
        gamma=gamma,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_xt_in_state_returns(
    events: pd.DataFrame,
    action_offset: int = 5,
    eligible_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return label_nth_future_state_value(
        events,
        value_col="xT",
        scores_col="scores_xT",
        concedes_col="concedes_xT",
        action_offset=action_offset,
        eligible_types=eligible_types,
    )


def label_goal_distance_returns(
    events: pd.DataFrame,
    lookahead_len: int = 5,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_future_max_value(
        events,
        value_col="goal_distance",
        scores_col="scores_goal_distance",
        concedes_col="concedes_goal_distance",
        lookahead_len=lookahead_len,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_epv_returns(
    events: pd.DataFrame,
    lookahead_len: int = 5,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_future_max_value(
        events,
        value_col="epv",
        scores_col="scores_epv",
        concedes_col="concedes_epv",
        lookahead_len=lookahead_len,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_epv_in_state_returns(
    events: pd.DataFrame,
    action_offset: int = 5,
    eligible_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return label_nth_future_state_value(
        events,
        value_col="epv",
        scores_col="scores_epv",
        concedes_col="concedes_epv",
        action_offset=action_offset,
        eligible_types=eligible_types,
    )


def label_discounted_epv_returns(
    events: pd.DataFrame,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_discounted_future_probability_value(
        events,
        value_col="epv",
        scores_col="scores_epv",
        concedes_col="concedes_epv",
        gamma=gamma,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_goal_distance_in_state_returns(
    events: pd.DataFrame,
    action_offset: int = 5,
    eligible_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    return label_nth_future_state_value(
        events,
        value_col="goal_distance",
        scores_col="scores_goal_distance",
        concedes_col="concedes_goal_distance",
        action_offset=action_offset,
        eligible_types=eligible_types,
    )


def label_discounted_goal_distance_returns(
    events: pd.DataFrame,
    gamma: float = 0.95,
    eligible_types: tuple[str, ...] | None = None,
    skip_first: bool = False,
) -> pd.DataFrame:
    return label_discounted_future_probability_value(
        events,
        value_col="goal_distance",
        scores_col="scores_goal_distance",
        concedes_col="concedes_goal_distance",
        gamma=gamma,
        eligible_types=eligible_types,
        skip_first=skip_first,
    )


def label_intended_receivers(
    actions: pd.DataFrame,
    tracking: pd.DataFrame,
    action_type="shot",
    max_angle=45,
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
    fps: int = 25,
) -> pd.DataFrame:
    if intended_receiver_mode not in {INTENDED_RECEIVER_MODE_ORIGINAL, INTENDED_RECEIVER_MODE_ANGLE_ONLY}:
        raise ValueError(f"Unsupported heuristic intended receiver mode: {intended_receiver_mode}")

    actions = actions.copy()
    actions["intent_id"] = pd.Series(index=actions.index, dtype="object")
    stats = {
        "mode": intended_receiver_mode,
        "failed_passes_total": 0,
        "failed_passes_labeled": 0,
        "failed_passes_missing_endpoint": 0,
    }
    trajectory_window_frames = max(int(round(float(fps) * 2.0)), 1)

    for i in actions.index:
        event_frame = actions.at[i, "frame_id"]
        possessor = actions.at[i, "object_id"]
        if pd.isna(event_frame) or not isinstance(possessor, str):
            continue

        event_frame = int(event_frame)
        if event_frame not in tracking.index:
            continue

        snapshot: pd.Series = tracking.loc[event_frame]

        receive_frame = actions.at[i, "receive_frame_id"]
        receiver = actions.at[i, "receiver_id"]

        if actions.at[i, "action_type"] not in ["pass", "shot"]:  # Mostly for clearances and dribbles
            actions.at[i, "intent_id"] = possessor

        elif actions.at[i, "action_type"] == "shot" or action_type == "shot":
            actions.at[i, "intent_id"] = f"{possessor[:4]}_goal"

            if actions.at[i, "action_type"] == "shot":  # For real shots
                if actions.at[i, "success"]:
                    end_x = actions.at[i, "end_x"]
                    if possessor[:4] == "home" and end_x < 10:
                        actions.at[i, "spadl_type"] = "own_goal"
                        actions.at[i, "intent_id"] = possessor
                        actions.at[i, "receiver_id"] = "away_goal"
                    elif possessor[:4] == "away" and end_x > config.FIELD_SIZE[0] - 10:
                        actions.at[i, "spadl_type"] = "own_goal"
                        actions.at[i, "intent_id"] = possessor
                        actions.at[i, "receiver_id"] = "home_goal"
                    else:
                        actions.at[i, "receiver_id"] = f"{possessor[:4]}_goal"

                elif actions.at[i, "next_type"] in config.SET_PIECE_OOP:
                    actions.at[i, "receiver_id"] = "out"

                else:
                    # elif actions.at[i, "next_type"] in config.INCOMING + ["clearance"]:
                    actions.at[i, "receiver_id"] = actions.at[i, "next_player_id"]

            else:  # For passes that would be blocked if they were shots
                actions.at[i, "receiver_id"] = actions.at[i, "blocker_id"]

        elif actions.at[i, "success"]:  # For successful passes
            actions.at[i, "intent_id"] = actions.at[i, "receiver_id"]

        elif not pd.isna(receiver) and not pd.isna(receive_frame):  # For failed passes
            stats["failed_passes_total"] += 1
            receive_frame = int(receive_frame)
            if receive_frame not in tracking.index:
                continue

            effective_receive_frame = receive_frame
            if intended_receiver_mode == INTENDED_RECEIVER_MODE_ANGLE_ONLY and receiver == "out":
                episode_end_frame = resolve_episode_end_frame(tracking, event_frame)
                if episode_end_frame is not None:
                    effective_receive_frame = episode_end_frame

            if effective_receive_frame not in tracking.index:
                continue

            receive_snapshot: pd.Series = tracking.loc[effective_receive_frame]

            teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_\d+_x", c)]
            teammates = [p for p in teammates if p != possessor]
            teammates = [
                p
                for p in teammates
                if f"{p}_x" in receive_snapshot.index
                and f"{p}_y" in receive_snapshot.index
                and not pd.isna(receive_snapshot[f"{p}_x"])
                and not pd.isna(receive_snapshot[f"{p}_y"])
            ]
            if not teammates:
                continue

            start_x = snapshot.get(f"{possessor}_x", actions.at[i, "start_x"])
            start_y = snapshot.get(f"{possessor}_y", actions.at[i, "start_y"])
            if pd.isna(start_x) or pd.isna(start_y):
                start_x = actions.at[i, "start_x"]
                start_y = actions.at[i, "start_y"]

            if pd.isna(start_x) or pd.isna(start_y):
                continue

            if intended_receiver_mode == INTENDED_RECEIVER_MODE_ORIGINAL:
                if receiver != "out":
                    end_x = receive_snapshot.get(f"{receiver}_x", actions.at[i, "end_x"])
                    end_y = receive_snapshot.get(f"{receiver}_y", actions.at[i, "end_y"])
                else:
                    end_x = receive_snapshot.get("ball_x", actions.at[i, "end_x"])
                    end_y = receive_snapshot.get("ball_y", actions.at[i, "end_y"])
                trajectory_frame = effective_receive_frame
            else:
                preferred_frame = min(event_frame + trajectory_window_frames, effective_receive_frame)
                trajectory_frame, end_x, end_y = resolve_ball_endpoint(
                    tracking,
                    preferred_frame=preferred_frame,
                    fallback_frame=effective_receive_frame,
                    fallback_xy=(actions.at[i, "end_x"], actions.at[i, "end_y"]),
                )

            if pd.isna(end_x) or pd.isna(end_y):
                stats["failed_passes_missing_endpoint"] += 1
                continue

            player_x = receive_snapshot[[f"{p}_x" for p in teammates]].values
            player_y = receive_snapshot[[f"{p}_y" for p in teammates]].values

            pass_dist = calc_dist(start_x, start_y, end_x, end_y)[-1]
            origin_dist_diffs = calc_dist(player_x, player_y, start_x, start_y)[-1] - pass_dist
            weights = downscale_closer_candidates(origin_dist_diffs)
            dest_dists = np.clip(calc_dist(player_x, player_y, end_x, end_y)[-1], 1, None)
            angles = np.clip(calc_angle(start_x, start_y, end_x, end_y, player_x, player_y), 0.01, None)

            if intended_receiver_mode == INTENDED_RECEIVER_MODE_ORIGINAL:
                max_radian = max_angle / 180 * np.pi
                if np.min(angles) < max_radian:
                    scores = weights * (np.min(dest_dists) / dest_dists) * (np.min(angles) / angles)
                    scores = np.where(angles < max_radian, scores, 0)
                    actions.at[i, "intent_id"] = teammates[np.argmax(scores)]
                    stats["failed_passes_labeled"] += 1
            else:
                min_angle = float(np.min(angles))
                scores = weights * (min_angle / angles)
                best_score = float(np.max(scores))
                candidate_indices = np.flatnonzero(np.isclose(scores, best_score))
                if len(candidate_indices) > 1:
                    best_dest_idx = candidate_indices[np.argmin(dest_dists[candidate_indices])]
                else:
                    best_dest_idx = int(candidate_indices[0])
                actions.at[i, "intent_id"] = teammates[best_dest_idx]
                stats["failed_passes_labeled"] += 1

    actions.attrs["intended_receiver_stats"] = stats
    return actions


def label_returns(
    events: pd.DataFrame,
    lookahead_len: int = 10,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()

    events["team"] = events["object_id"].str[:4]
    events["goal"] = events["expected_goal"].notna() & events["success"]
    events["scores"] = 0.0
    events["concedes"] = 0.0
    events["scores_xg"] = 0.0
    events["concedes_xg"] = 0.0

    for period in events["period_id"].unique():
        period_events = events[events["period_id"] == period]
        labels = period_events[["team", "goal", "expected_goal"]].copy()
        first_future_types = period_events["spadl_type"].shift(-1)
        skip_mask = first_future_types.notna() & first_future_types.ne("shot")

        for i in range(lookahead_len):
            shifted_teams = labels["team"].shift(-i)
            shifted_goals = labels[["goal", "expected_goal"]].shift(-i).fillna(0)
            # shifted_returns = labels.shift(-i).fillna(0).infer_objects(copy=False)
            scoring_goals = shifted_goals["goal"] * (shifted_teams == labels["team"]).astype(int)
            conceding_goals = shifted_goals["goal"] * (shifted_teams != labels["team"]).astype(int)
            scoring_xg = shifted_goals["expected_goal"] * (shifted_teams == labels["team"]).astype(int)
            conceding_xg = shifted_goals["expected_goal"] * (shifted_teams != labels["team"]).astype(int)
            if skip_first and i == 1:
                scoring_goals = scoring_goals.where(~skip_mask, 0)
                conceding_goals = conceding_goals.where(~skip_mask, 0)
                scoring_xg = scoring_xg.where(~skip_mask, 0)
                conceding_xg = conceding_xg.where(~skip_mask, 0)
            labels[f"sg+{i}"] = scoring_goals
            labels[f"cg+{i}"] = conceding_goals
            labels[f"sxg+{i}"] = scoring_xg
            labels[f"cxg+{i}"] = conceding_xg

        scoring_cols = [c for c in labels.columns if c.startswith("sg+")]
        scoring_xg_cols = [c for c in labels.columns if c.startswith("sxg+")]
        conceding_cols = [c for c in labels.columns if c.startswith("cg+")]
        conceding_xg_cols = [c for c in labels.columns if c.startswith("cxg+")]

        events.loc[labels.index, "scores"] = labels[scoring_cols].sum(axis=1).clip(0, 1).astype(int)
        events.loc[labels.index, "scores_xg"] = 1 - (1 - labels[scoring_xg_cols]).prod(axis=1)
        events.loc[labels.index, "concedes"] = labels[conceding_cols].sum(axis=1).clip(0, 1).astype(int)
        events.loc[labels.index, "concedes_xg"] = 1 - (1 - labels[conceding_xg_cols]).prod(axis=1)

    return events


def label_discounted_returns(
    events: pd.DataFrame,
    gamma: float = 0.95,
    skip_first: bool = False,
) -> pd.DataFrame:
    events = events.copy()

    expected_goal_source = events["expected_goal"] if "expected_goal" in events.columns else pd.Series(0.0, index=events.index)
    expected_goals = pd.to_numeric(expected_goal_source, errors="coerce").fillna(0)
    goal_array = ((expected_goals > 0) & events["success"].fillna(False).astype(bool)).to_numpy(dtype=bool)
    shot_array = events["spadl_type"].eq("shot").to_numpy(dtype=bool)
    events["goal"] = goal_array
    events["scores_xg_disc"] = 0.0
    events["concedes_xg_disc"] = 0.0
    n_events = len(events)

    for i in range(n_events):
        period_i = events.iat[i, events.columns.get_loc("period_id")]
        team_i = str(events.iat[i, events.columns.get_loc("object_id")])[:4]

        prob_not_scoring = 1.0
        prob_not_conceding = 1.0
        first_future_seen = False

        for j in range(i, n_events):
            skip_future = False
            if j > i and not first_future_seen:
                first_future_seen = True
                skip_future = skip_first and not shot_array[j]

            if skip_future:
                if _should_stop_discount_scan(events, j, period_i, goal_array):
                    break
                continue

            if str(events.iat[j, events.columns.get_loc("object_id")])[:4] == team_i:  # future shot by a teammate
                prob_not_scoring *= 1 - gamma ** (j - i) * expected_goals.iat[j]
            else:  # future shot by an opponent
                prob_not_conceding *= 1 - gamma ** (j - i) * expected_goals.iat[j]

            if _should_stop_discount_scan(events, j, period_i, goal_array):
                break

        events.iat[i, events.columns.get_loc("scores_xg_disc")] = 1 - prob_not_scoring
        events.iat[i, events.columns.get_loc("concedes_xg_disc")] = 1 - prob_not_conceding

    return events


def count_potential_interceptors(
    poss_x: np.ndarray,
    poss_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    is_teammate: np.ndarray,
    corridor_width: float = 10.0,
) -> np.ndarray:
    opponent_x = player_x[is_teammate == 0]
    opponent_y = player_y[is_teammate == 0]
    potential_interceptors = np.zeros(len(player_x))

    for i in np.nonzero(is_teammate == 1)[0]:
        target_x = player_x[i]
        target_y = player_y[i]
        pass_dx = target_x - poss_x
        pass_dy = target_y - poss_y
        pass_len = np.hypot(pass_dx, pass_dy)

        if pass_len < 1e-6:  # Skip the possessor
            continue

        buffer_x = (corridor_width / 2) * (-pass_dy / pass_len)
        buffer_y = (corridor_width / 2) * (pass_dx / pass_len)

        p1 = (poss_x - buffer_x, poss_y - buffer_y)
        p2 = (poss_x + buffer_x, poss_y + buffer_y)
        p3 = (target_x + buffer_x, target_y + buffer_y)
        p4 = (target_x - buffer_x, target_y - buffer_y)

        corridor = Polygon([p1, p2, p3, p4])
        inside_mask = vectorized.contains(corridor, opponent_x, opponent_y)
        potential_interceptors[i] = np.count_nonzero(inside_mask)

    return potential_interceptors


def find_nearest_opponent_to_pass(
    poss_x: np.ndarray,
    poss_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    is_teammate: np.ndarray,
) -> np.ndarray:
    teammate_x = player_x[is_teammate == 1][np.newaxis, :]  # [1, teammates]
    teammate_y = player_y[is_teammate == 1][np.newaxis, :]  # [1, teammates]
    opponent_x = player_x[is_teammate == 0][:, np.newaxis]  # [opponents, 1]
    opponent_y = player_y[is_teammate == 0][:, np.newaxis]  # [opponents, 1]

    pass_x = teammate_x - poss_x  # [1, teammates]
    pass_y = teammate_y - poss_y  # [1, teammates]
    pass_len = np.hypot(pass_x, pass_y) + 1e-6  # [1, teammates]

    oppo_rel_x = opponent_x - poss_x
    oppo_rel_y = opponent_y - poss_y
    proj_coeffs = (oppo_rel_x * pass_x + oppo_rel_y * pass_y) / (pass_len**2)  # [opponents, teammates]
    proj_coeffs = np.clip(proj_coeffs, 0, 1)

    oppo_proj_x = poss_x + proj_coeffs * pass_x  # [opponents, teammates]
    oppo_proj_y = poss_y + proj_coeffs * pass_y  # [opponents, teammates]
    dists_to_pass = np.hypot(opponent_x - oppo_proj_x, opponent_y - oppo_proj_y)  # [opponents, teammates]
    min_dists = np.min(dists_to_pass, axis=0)  # [teammates,]

    return np.concatenate([min_dists, np.zeros(len(opponent_x))])  # [players,]


def count_potential_blockers(
    goal_x: float,
    goal_y: float,
    player_x: np.ndarray,
    player_y: np.ndarray,
    teammate_mask: np.ndarray,
    buffer: float = 1.0,
) -> np.ndarray:
    goal_left = (goal_x, goal_y - config.GOAL_SIZE / 2 + buffer)
    goal_right = (goal_x, goal_y + config.GOAL_SIZE / 2 - buffer)

    player_x = np.asarray(player_x, dtype=float)
    player_y = np.asarray(player_y, dtype=float)
    teammate_mask = np.asarray(teammate_mask).astype(bool)  # [players,]
    pairwise_opponent_mask = teammate_mask != teammate_mask[:, np.newaxis]  # [players, players]

    player_xy = np.stack([player_x, player_y], axis=-1)
    potential_blockers = np.zeros(len(player_xy))

    for i in np.nonzero(teammate_mask)[0]:
        tri = Polygon([goal_left, goal_right, tuple(player_xy[i])])
        if not tri.is_valid or tri.area < 1e-3:
            continue

        search_region = tri.buffer(buffer)
        opponent_idxs = np.where(pairwise_opponent_mask[i])[0]
        count = 0

        for idx in opponent_idxs:
            opponent_point = Point(player_x[idx], player_y[idx])
            if search_region.contains(opponent_point):
                count += 1

        potential_blockers[i] = count

    return potential_blockers


def find_nearest_blocker(event: pd.Series, tracking: pd.DataFrame, keepers: np.ndarray) -> int:
    snapshot: pd.Series = tracking.loc[event["frame_id"]]
    event_xy = event[["start_x", "start_y"]].values.tolist()

    goal_x = config.FIELD_SIZE[0] if event["object_id"].startswith("home") else 0
    goal_xy_lower = [goal_x, config.FIELD_SIZE[1] / 2 - 4]
    goal_xy_upper = [goal_x, config.FIELD_SIZE[1] / 2 + 4]
    goal_side_vertices = np.array([event_xy, goal_xy_lower, goal_xy_upper])
    goal_side = Polygon(goal_side_vertices).buffer(1)  # .intersection(Point(event_xy).buffer(10))

    oppo_team = "away" if event["object_id"].startswith("home") else "home"
    oppo_x_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_x") and c[:-2] not in keepers]
    oppo_y_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_y") and c[:-2] not in keepers]
    player_xy = np.stack([snapshot[oppo_x_cols].values, snapshot[oppo_y_cols].values]).T
    player_xy = pd.DataFrame(player_xy, index=[c[:-2] for c in oppo_x_cols], columns=["x", "y"])

    can_block = player_xy.apply(lambda p: goal_side.contains(Point(p["x"], p["y"])), axis=1)
    potential_blockers = player_xy.loc[can_block[can_block].index]

    if potential_blockers.empty:
        return np.nan
    else:
        potential_blockers["dist_x"] = potential_blockers["x"] - event_xy[0]
        potential_blockers["dist_y"] = potential_blockers["y"] - event_xy[1]
        blocker_dists = potential_blockers[["dist_x", "dist_y"]].apply(np.linalg.norm, axis=1)
        return blocker_dists.idxmin()


def drop_nodes(graph: Data, labels: torch.Tensor, node_mask: torch.BoolTensor) -> Tuple[Data, torch.Tensor]:
    node_mask_indices = torch.where(node_mask)[0]
    index_map = -torch.ones((graph.num_nodes,)).long()
    index_map[node_mask_indices] = torch.arange(len(node_mask_indices))
    index_map = torch.cat([index_map, torch.tensor([-1], dtype=torch.long)])  # To map -1 to -1

    node_attr = graph.x[node_mask]
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    edge_index = index_map[graph.edge_index[:, edge_mask]]
    edge_attr = graph.edge_attr[edge_mask]
    masked_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    if hasattr(graph, "node_ids"):
        masked_graph.node_ids = [graph.node_ids[idx] for idx in node_mask_indices.tolist()]

    masked_labels = labels.clone()
    masked_labels[4] = node_mask.long().sum()  # number of players
    masked_labels[5] = index_map[masked_labels[5].long()]  # intent index
    masked_labels[6] = index_map[masked_labels[6].long()]  # receiver index

    return masked_graph, masked_labels


def drop_opponent_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 0] == 1
    return drop_nodes(graph, labels, node_mask)


def drop_goal_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 2] == 0
    return drop_nodes(graph, labels, node_mask)


def drop_non_blocker_nodes(
    graph: Data, labels: torch.Tensor, poss_flag_index=13, buffer_x=5
) -> Tuple[Data, torch.Tensor]:
    poss_or_oppo = (graph.x[:, poss_flag_index] == 1) | (graph.x[:, 0] == 0)
    poss_x = graph.x[graph.x[:, poss_flag_index] == 1, 3].item()
    node_mask = poss_or_oppo & (graph.x[:, 3] > poss_x - buffer_x)
    return drop_nodes(graph, labels, node_mask)


def sparsify_edges(graph: Data, how="distance", possessor_index: int = None, max_dist=10) -> Data:
    if how == "distance":
        edge_index = graph.edge_index
        if possessor_index is not None:
            passer_edges = (edge_index[0] == possessor_index) | (edge_index[1] == possessor_index)
        close_edges = graph.edge_attr[:, 0] <= max_dist

        graph.edge_index = edge_index[:, passer_edges | close_edges]
        graph.edge_attr = graph.edge_attr[passer_edges | close_edges]

    elif how == "delaunay":
        # xy = graph.x[:, 1:3] if graph.x.shape[1] < 18 else graph.x[:, 3:5]
        xy = graph.x[:, 3:5]
        tri_pts = Delaunay(xy.cpu().detach().numpy()).simplices
        tri_edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        tri_edges = np.unique(tri_edges, axis=0).tolist()

        for [i, j] in tri_edges:
            if [j, i] not in tri_edges:
                tri_edges.append([j, i])

        complete_edges = graph.edge_index.cpu().detach().numpy().T
        complete_edge_dict = {tuple(e): i for i, e in enumerate(complete_edges)}
        tri_edge_index = np.sort([complete_edge_dict[tuple(e)] for e in tri_edges]).tolist()

        graph.edge_index = graph.edge_index[:, tri_edge_index]
        graph.edge_attr = graph.edge_attr[tri_edge_index]

    return graph


def adapt_graph_edge_features(graph: Data, edge_in_dim: int | None = None) -> Data:
    if graph is None or edge_in_dim is None or not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return graph

    edge_in_dim = int(edge_in_dim)
    actual_edge_dim = int(graph.edge_attr.shape[1])
    if actual_edge_dim < edge_in_dim:
        raise ValueError(
            f"Graph edge schema is incompatible with the requested model schema: "
            f"graph_edge_dim={actual_edge_dim}, required_edge_dim={edge_in_dim}."
        )
    if actual_edge_dim == edge_in_dim:
        return graph

    graph.edge_attr = graph.edge_attr[:, :edge_in_dim]
    return graph


def mask_possessor_velocity_edge_features(graph: Data, possessor_index: int) -> Data:
    if graph is None or not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return graph
    if int(graph.edge_attr.shape[1]) < 4:
        return graph

    edge_index = graph.edge_index
    incident_edges = (edge_index[0] == int(possessor_index)) | (edge_index[1] == int(possessor_index))
    graph.edge_attr[incident_edges, 2:4] = 0
    return graph


def should_mask_possessor_velocity_edge_features(args: Dict[str, Any]) -> bool:
    mode = args.get("v_edge_feature_mode")
    if mode is not None:
        return str(mode).strip().replace("-", "_") == "no_poss"
    return bool(args.get("mask_possessor_v_edge_features", False))


def filter_features_and_labels(
    features: List[Data],
    labels: torch.Tensor,
    args: Dict[str, Any],
    event_indices: np.ndarray = None,
    feature_action_indices: np.ndarray | torch.Tensor | list[int] | None = None,
) -> Tuple[List[Data], torch.Tensor]:
    filtered_features = []
    filtered_labels = []
    event_index_set = None if event_indices is None else {int(event_index) for event_index in event_indices}

    feature_lookup: dict[int, int] | None = None
    if feature_action_indices is not None:
        feature_action_indices = np.asarray(feature_action_indices, dtype=int)
        if len(feature_action_indices) != len(features):
            raise ValueError(
                "Feature action-index count does not match graph count: "
                f"feature_action_indices={len(feature_action_indices)}, features={len(features)}."
            )
        feature_lookup = {}
        duplicates = set()
        for feature_pos, action_index in enumerate(feature_action_indices.tolist()):
            action_index = int(action_index)
            if action_index in feature_lookup:
                duplicates.add(action_index)
            feature_lookup[action_index] = feature_pos
        if duplicates:
            sample = sorted(duplicates)[:5]
            raise ValueError(f"Feature action indexes contain duplicates, for example: {sample}.")
    elif len(features) != len(labels):
        raise ValueError(
            "Graph features and labels are not row-aligned and no feature action indexes were provided: "
            f"features={len(features)}, labels={len(labels)}."
        )

    for i in range(len(labels)):
        action_index = int(labels[i, 0].item())
        if event_index_set is not None and action_index not in event_index_set:
            continue

        feature_pos = feature_lookup.get(action_index) if feature_lookup is not None else i
        if feature_pos is None:
            continue

        graph: Data = features[feature_pos]
        graph_labels: torch.Tensor = labels[i]

        if graph is None:
            # filtered_features.append(graph)
            continue
        else:
            graph = graph.clone()
            graph = adapt_graph_edge_features(graph, args.get("edge_in_dim"))

        try:
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
        except RuntimeError:
            continue
        if should_mask_possessor_velocity_edge_features(args):
            graph = mask_possessor_velocity_edge_features(graph, int(possessor_index))

        if args["xy_only"]:
            graph.x[:, 7:12] = 0
            graph.x[:,13:19] = 0

        if not args["possessor_aware"]:
            assert not args["extend_features"]
            graph.x[:, 13:] = 0

        if not args["poss_vel_aware"]:
            if args["possessor_aware"]:
                graph.x[graph.x[:, 13] == 1, 5:9] = 0
            graph.x[:, 17:19] = 0

        if not args["keeper_aware"]:
            graph.x[:, 1] = 0

        if not args["ball_z_aware"]:
            graph.x[:, 12] = 0

        if not args.get("accel_aware", True):
            graph.x[:, 8] = 0

        if not args["extend_features"] and args.get("task") != "success_intent":
            graph.x[:, 19:] = 0

        if not config.TASK_CONFIG.at[args["task"], "include_goals"]:
            graph, graph_labels = drop_goal_nodes(graph, graph_labels)

        if args["task"].endswith("oppo_agn"):
            graph, graph_labels = drop_opponent_nodes(graph, graph_labels)

        if "filter_blockers" in args and args["filter_blockers"]:
            assert args["possessor_aware"]
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
            graph, graph_labels = drop_non_blocker_nodes(graph, graph_labels)

        if args["sparsify"] == "distance":
            assert args["possessor_aware"]
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
            graph = sparsify_edges(graph, "distance", possessor_index, args["max_edge_dist"])
        elif args["sparsify"] == "delaunay" and graph.x.shape[0] > 3:
            graph = sparsify_edges(graph, "delaunay")

        filtered_features.append(graph)
        filtered_labels.append(graph_labels)

    if not filtered_labels:
        raise ValueError("No usable graph/label pairs remain after filtering and alignment.")

    return filtered_features, torch.stack(filtered_labels, axis=0)


def find_active_players(tracking: pd.DataFrame, frame: int = None, team: str = None, include_goals=False) -> dict:
    if pd.isna(frame):
        snapshot = tracking.dropna(how="all", axis=1).copy()
    else:
        snapshot = tracking.loc[frame:frame].dropna(how="all", axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    if not pd.isna(frame):
        team = team or tracking.at[frame, "ball_owning_home_away"]
    else:
        team = team or "home"

    if team == "home":
        players = [home_players, away_players]
    else:
        players = [away_players, home_players]

    return players


def player_sort_key(s: str):
    if s == "home_goal":
        return (2, 0)
    elif s == "away_goal":
        return (2, 1)
    else:
        team, num = s.split("_", 1)
        return (0 if team == "home" else 1, int(num))


def abbr_position(position: str) -> str:
    if position == "striker":
        return "CF"
    else:
        tokens = ["back" if t == "defender" else t for t in position.split("_")]
        return "".join(t[0].upper() for t in tokens)


def insert_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    if "index" not in actions:
        actions = actions.copy().reset_index()

    dribble_actions = []

    for i in range(len(actions) - 1):
        cur_action = actions.iloc[i]
        next_action = actions.iloc[i + 1]

        if (
            (cur_action["period_id"] == next_action["period_id"])
            and not cur_action["offside"]
            and (next_action["spadl_type"] not in config.SET_PIECE_OOP)
            and (next_action["frame_id"] - cur_action["receive_frame_id"] >= 5)
            and (next_action["object_id"] == cur_action["receiver_id"])
        ):
            cur_action_dur = 0.04 * (cur_action["receive_frame_id"] - cur_action["frame_id"])
            action = next_action.copy().to_dict()
            action["index"] = -1
            action["frame_id"] = max(cur_action["receive_frame_id"], cur_action["frame_id"] + 1)
            action["seconds"] = round(cur_action["seconds"] + cur_action_dur, 2)
            action["spadl_type"] = "dribble"
            action["success"] = True
            action["offside"] = False
            action["expected_goal"] = np.nan
            action["next_player_id"] = next_action["object_id"]
            action["next_type"] = next_action["spadl_type"]
            action["receiver_id"] = next_action["object_id"]
            action["receive_frame_id"] = next_action["frame_id"]
            action["start_x"] = cur_action["end_x"]
            action["start_y"] = cur_action["end_y"]
            action["start_z"] = 0
            action["end_x"] = next_action["start_x"]
            action["end_y"] = next_action["start_y"]
            action["action_type"] = "dribble"
            action["blocked"] = False
            action["intent_id"] = next_action["object_id"]
            dribble_actions.append(action)

    return pd.concat([actions, pd.DataFrame(dribble_actions)]).sort_values("frame_id", ignore_index=True)
