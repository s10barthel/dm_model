from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from datatools import config, utils
from datatools.xt import build_xt_actions, infer_home_team_id, rotate_xt_actions, sort_events
from project_config import GOAL_DISTANCE_DIR, GOAL_DISTANCE_MATCH_DIR

GOAL_DISTANCE_GOAL_X = config.FIELD_SIZE[0]
GOAL_DISTANCE_GOAL_Y = config.FIELD_SIZE[1] / 2
GOAL_DISTANCE_MAX_RAW = float(np.hypot(config.FIELD_SIZE[0], config.FIELD_SIZE[1] / 2))
GOAL_DISTANCE_MAX_VALUE = 1.0


def goal_distance_from_xy(start_x: pd.Series, start_y: pd.Series) -> np.ndarray:
    x = pd.to_numeric(start_x, errors="coerce")
    y = pd.to_numeric(start_y, errors="coerce")

    values = np.full(len(x), np.nan, dtype=float)
    valid_mask = x.notna() & y.notna()
    if not valid_mask.any():
        return values

    goal_dx = GOAL_DISTANCE_GOAL_X - x[valid_mask].to_numpy(dtype=float)
    goal_dy = GOAL_DISTANCE_GOAL_Y - y[valid_mask].to_numpy(dtype=float)
    raw_distance = np.hypot(goal_dx, goal_dy)
    normalized = GOAL_DISTANCE_MAX_VALUE * (1.0 - raw_distance / GOAL_DISTANCE_MAX_RAW)
    values[valid_mask.to_numpy()] = np.clip(normalized, 0.0, GOAL_DISTANCE_MAX_VALUE)
    return values


def annotate_match_goal_distance(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_events = sort_events(events)
    sorted_events["goal_distance"] = np.nan

    goal_distance_actions = build_xt_actions(sorted_events)
    if not goal_distance_actions.empty:
        home_team_id = infer_home_team_id(sorted_events)
        rotated_actions = rotate_xt_actions(goal_distance_actions, home_team_id)
        rotated_actions["goal_distance"] = goal_distance_from_xy(
            rotated_actions["start_x"],
            rotated_actions["start_y"],
        )

        sorted_events = sorted_events.merge(
            rotated_actions[["action_id", "goal_distance"]],
            on="action_id",
            how="left",
            suffixes=("", "_goal_distance"),
        )
        sorted_events["goal_distance"] = sorted_events["goal_distance_goal_distance"].combine_first(
            sorted_events["goal_distance"]
        )
        sorted_events = sorted_events.drop(
            columns=[c for c in ["goal_distance_goal_distance"] if c in sorted_events.columns]
        )

    sorted_events = utils.label_goal_distance_returns(
        sorted_events,
        lookahead_len=5,
        eligible_types=tuple(config.XT_ACTION_TYPES),
    )

    user_export = sorted_events.loc[sorted_events["spadl_type"].isin(config.XT_ACTION_TYPES)].copy()
    export_cols = [
        col
        for col in [
            "game_id",
            "stats_perform_match_id",
            "action_id",
            "original_event_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "object_id",
            "spadl_type",
            "success",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "goal_distance",
            "scores_goal_distance",
            "concedes_goal_distance",
        ]
        if col in user_export.columns
    ]
    return sorted_events, user_export[export_cols].copy()


def merge_goal_distance_annotations(
    events: pd.DataFrame,
    match_id: str | None,
    goal_distance_match_dir: str | Path = GOAL_DISTANCE_MATCH_DIR,
) -> pd.DataFrame:
    events = events.copy()
    if match_id is None:
        return events

    sidecar_path = Path(goal_distance_match_dir) / f"{match_id}.csv"
    if not sidecar_path.exists():
        return events

    goal_distance_columns = ["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]
    sidecar = pd.read_csv(sidecar_path, usecols=lambda c: c in goal_distance_columns)
    if sidecar.empty:
        return events

    events = events.drop(
        columns=[c for c in ["goal_distance", "scores_goal_distance", "concedes_goal_distance"] if c in events.columns]
    )
    return events.merge(sidecar, on="action_id", how="left")


def save_goal_distance_outputs(
    all_events: Iterable[pd.DataFrame],
    metadata: dict,
    output_dir: str | Path = GOAL_DISTANCE_DIR,
    goal_distance_match_dir: str | Path = GOAL_DISTANCE_MATCH_DIR,
) -> None:
    output_dir = Path(output_dir)
    goal_distance_match_dir = Path(goal_distance_match_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    goal_distance_match_dir.mkdir(parents=True, exist_ok=True)

    exported_actions: list[pd.DataFrame] = []
    for events in all_events:
        if events.empty:
            continue

        annotated_events, exported_goal_distance = annotate_match_goal_distance(events)
        match_id = str(
            annotated_events["stats_perform_match_id"].iloc[0]
            if "stats_perform_match_id" in annotated_events.columns
            else annotated_events["game_id"].iloc[0]
        )
        sidecar = annotated_events[
            ["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]
        ].copy()
        sidecar.to_csv(goal_distance_match_dir / f"{match_id}.csv", index=False)
        exported_actions.append(exported_goal_distance)

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "goal_distance.csv", index=False)
    else:
        pd.DataFrame(
            columns=["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]
        ).to_csv(output_dir / "goal_distance.csv", index=False)

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
