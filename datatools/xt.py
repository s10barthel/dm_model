from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadlconfig
from socceraction.spadl import play_left_to_right
from socceraction.xthreat import ExpectedThreat

from datatools import config, utils
from project_config import XT_DIR, XT_MATCH_DIR

XT_GRID_L = 12
XT_GRID_W = 8
XT_GRID_COLUMNS = [f"X{i}" for i in range(XT_GRID_L)]


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    order_cols = [col for col in ["action_id", "period_id", "seconds", "original_event_id"] if col in events.columns]
    if order_cols:
        events = events.sort_values(order_cols).reset_index(drop=True)
    return events


def infer_home_team_id(events: pd.DataFrame) -> str:
    home_team_ids = events.loc[events["object_id"].astype(str).str.startswith("home"), "team_id"].dropna().unique().tolist()
    if len(home_team_ids) != 1:
        raise ValueError(f"Expected exactly one home team id, found {home_team_ids}.")
    return str(home_team_ids[0])


def spadl_result_id(spadl_type: str, success: bool, offside: bool) -> int:
    if offside:
        return spadlconfig.results.index("offside")
    if spadl_type == "shot" and not success:
        return spadlconfig.results.index("fail")
    return spadlconfig.results.index("success" if success else "fail")


def build_xt_actions(events: pd.DataFrame) -> pd.DataFrame:
    events = sort_events(utils.sanitize_expected_goal(events))
    xt_actions = events.loc[events["spadl_type"].isin(config.XT_ACTION_TYPES)].copy()

    if xt_actions.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "original_event_id",
                "action_id",
                "period_id",
                "seconds",
                "team_id",
                "player_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "type_id",
                "result_id",
                "bodypart_id",
                "spadl_type",
                "success",
                "offside",
                "xG",
            ]
        )

    xt_actions["game_id"] = xt_actions["stats_perform_match_id"] if "stats_perform_match_id" in xt_actions.columns else xt_actions["game_id"]
    xt_actions["team_id"] = xt_actions["team_id"].astype(str)
    xt_actions["type_id"] = xt_actions["spadl_type"].map(spadlconfig.actiontypes.index)
    xt_actions["result_id"] = xt_actions.apply(
        lambda row: spadl_result_id(row["spadl_type"], bool(row["success"]), bool(row.get("offside", False))),
        axis=1,
    )
    xt_actions["bodypart_id"] = spadlconfig.bodyparts.index("foot")
    xt_actions["xG"] = xt_actions["expected_goal"].astype(float)
    return xt_actions[
        [
            "game_id",
            "original_event_id",
            "action_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_id",
            "result_id",
            "bodypart_id",
            "spadl_type",
            "success",
            "offside",
            "xG",
        ]
    ].copy()


def rotate_xt_actions(xt_actions: pd.DataFrame, home_team_id: str) -> pd.DataFrame:
    if xt_actions.empty:
        return xt_actions.copy()

    rotated = play_left_to_right(
        xt_actions[
            [
                "game_id",
                "original_event_id",
                "action_id",
                "period_id",
                "seconds",
                "team_id",
                "player_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "type_id",
                "result_id",
                "bodypart_id",
            ]
        ].copy(),
        home_team_id=home_team_id,
    ).reset_index(drop=True)
    passthrough_cols = xt_actions.drop(
        columns=[
            "game_id",
            "original_event_id",
            "action_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_id",
            "result_id",
            "bodypart_id",
        ]
    )
    return rotated.join(passthrough_cols.reset_index(drop=True))


def symmetrize_grid(grid: np.ndarray) -> np.ndarray:
    symmetric = np.asarray(grid, dtype=float).copy()
    for upper, lower in [(0, 7), (1, 6), (2, 5), (3, 4)]:
        averaged = (symmetric[upper] + symmetric[lower]) / 2.0
        symmetric[upper] = averaged
        symmetric[lower] = averaged
    return symmetric


def fit_xt_surface(rotated_train_actions: pd.DataFrame) -> np.ndarray:
    model = ExpectedThreat(l=XT_GRID_L, w=XT_GRID_W)
    model.fit(rotated_train_actions)
    return symmetrize_grid(model.xT)


def zone_value_from_grid(start_x: pd.Series, start_y: pd.Series, grid: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(start_x, errors="coerce").clip(0, config.FIELD_SIZE[0] - 1e-9)
    y = pd.to_numeric(start_y, errors="coerce").clip(0, config.FIELD_SIZE[1] - 1e-9)

    x_index = np.floor((x / config.FIELD_SIZE[0]) * XT_GRID_L).astype("Int64").clip(0, XT_GRID_L - 1)
    y_index = np.floor((y / config.FIELD_SIZE[1]) * XT_GRID_W).astype("Int64").clip(0, XT_GRID_W - 1)
    row_index = XT_GRID_W - 1 - y_index

    values = np.full(len(x), np.nan, dtype=float)
    valid_mask = x.notna() & y.notna()
    values[valid_mask.to_numpy()] = grid[row_index[valid_mask].astype(int), x_index[valid_mask].astype(int)]
    return values


def annotate_match_xt(events: pd.DataFrame, grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_events = sort_events(events)
    sorted_events = utils.sanitize_expected_goal(sorted_events)
    sorted_events["xT"] = np.nan
    sorted_events["xG"] = sorted_events["expected_goal"].astype(float)

    xt_actions = build_xt_actions(sorted_events)
    if not xt_actions.empty:
        home_team_id = infer_home_team_id(sorted_events)
        rotated_actions = rotate_xt_actions(xt_actions, home_team_id)
        zone_values = zone_value_from_grid(rotated_actions["start_x"], rotated_actions["start_y"], grid)
        rotated_actions["xT"] = zone_values

        shot_mask = rotated_actions["spadl_type"] == "shot"
        rotated_actions.loc[shot_mask, "xT"] = np.maximum(
            rotated_actions.loc[shot_mask, "xT"].to_numpy(dtype=float),
            rotated_actions.loc[shot_mask, "xG"].fillna(0.0).to_numpy(dtype=float),
        )

        sorted_events = sorted_events.merge(
            rotated_actions[["action_id", "xG", "xT"]],
            on="action_id",
            how="left",
            suffixes=("", "_xt"),
        )
        sorted_events["xG"] = sorted_events["xG_xt"].combine_first(sorted_events["xG"])
        sorted_events["xT"] = sorted_events["xT_xt"].combine_first(sorted_events["xT"])
        sorted_events = sorted_events.drop(columns=[c for c in ["xG_xt", "xT_xt"] if c in sorted_events.columns])

    sorted_events = utils.label_xt_returns(sorted_events, lookahead_len=5, eligible_types=tuple(config.XT_ACTION_TYPES))

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
            "xG",
            "xT",
            "scores_xT",
            "concedes_xT",
        ]
        if col in user_export.columns
    ]
    return sorted_events, user_export[export_cols].copy()


def merge_xt_annotations(events: pd.DataFrame, match_id: str | None, xt_match_dir: str | Path = XT_MATCH_DIR) -> pd.DataFrame:
    events = events.copy()
    if match_id is None:
        return events

    sidecar_path = Path(xt_match_dir) / f"{match_id}.csv"
    if not sidecar_path.exists():
        return events

    xt_columns = ["action_id", "xT", "scores_xT", "concedes_xT"]
    xt_sidecar = pd.read_csv(sidecar_path, usecols=lambda c: c in xt_columns)
    if xt_sidecar.empty:
        return events

    events = events.drop(columns=[c for c in ["xT", "scores_xT", "concedes_xT"] if c in events.columns])
    return events.merge(xt_sidecar, on="action_id", how="left")


def save_xt_outputs(
    all_events: Iterable[pd.DataFrame],
    grid: np.ndarray,
    fit_metadata: dict,
    output_dir: str | Path = XT_DIR,
    xt_match_dir: str | Path = XT_MATCH_DIR,
) -> None:
    output_dir = Path(output_dir)
    xt_match_dir = Path(xt_match_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xt_match_dir.mkdir(parents=True, exist_ok=True)

    exported_actions: list[pd.DataFrame] = []
    for events in all_events:
        if events.empty:
            continue

        annotated_events, exported_xt = annotate_match_xt(events, grid)
        match_id = str(
            annotated_events["stats_perform_match_id"].iloc[0]
            if "stats_perform_match_id" in annotated_events.columns
            else annotated_events["game_id"].iloc[0]
        )
        sidecar = annotated_events[["action_id", "xG", "xT", "scores_xT", "concedes_xT"]].copy()
        sidecar.to_csv(xt_match_dir / f"{match_id}.csv", index=False)
        exported_actions.append(exported_xt)

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "xT.csv", index=False)
    else:
        pd.DataFrame(columns=["action_id", "xG", "xT", "scores_xT", "concedes_xT"]).to_csv(
            output_dir / "xT.csv", index=False
        )

    grid_frame = pd.DataFrame(grid, columns=XT_GRID_COLUMNS)
    grid_frame.to_csv(output_dir / "xT_grid.csv", index=False)
    (output_dir / "fit_metadata.json").write_text(json.dumps(fit_metadata, indent=2), encoding="utf-8")
