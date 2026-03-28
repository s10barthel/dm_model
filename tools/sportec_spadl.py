from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pandera.errors import SchemaError
from socceraction.spadl import add_names as add_spadl_names
from socceraction.spadl import config as spadlconfig
from socceraction.spadl.base import _add_dribbles, _fix_clearances
from socceraction.spadl.schema import SPADLSchema

FIELD_LENGTH = float(spadlconfig.field_length)
FIELD_WIDTH = float(spadlconfig.field_width)

PASS_LIKE_TYPES = {
    "pass",
    "cross",
    "throw_in",
    "goalkick",
    "corner_short",
    "corner_crossed",
    "freekick_short",
    "freekick_crossed",
}
SHOT_LIKE_TYPES = {"shot", "shot_freekick", "shot_penalty"}
STRICT_SPADL_TYPES = PASS_LIKE_TYPES | SHOT_LIKE_TYPES | {"foul"}


def align_sportec_event_orientations(lineup: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if events.empty:
        return events

    if "team_prefix" in lineup.columns:
        home_mask = lineup["team_prefix"].eq("home")
        away_mask = lineup["team_prefix"].eq("away")
    elif "object_id" in lineup.columns:
        home_mask = lineup["object_id"].astype("string").str.startswith("home_")
        away_mask = lineup["object_id"].astype("string").str.startswith("away_")
    else:
        raise ValueError("Lineup must contain either team_prefix or object_id to align event orientations.")

    goalkeeper_mask = lineup.get("advanced_position", pd.Series(index=lineup.index, dtype="string")).eq("goal_keeper")
    home_goalkeepers = lineup.loc[goalkeeper_mask & home_mask, "player_id"].astype("string").tolist()
    away_goalkeepers = lineup.loc[goalkeeper_mask & away_mask, "player_id"].astype("string").tolist()

    for period_id in sorted(pd.unique(events["period_id"])):
        if pd.isna(period_id) or int(period_id) <= 0:
            continue

        period_events = events.loc[events["period_id"] == period_id]
        home_gk_x = period_events.loc[period_events["player_id"].isin(home_goalkeepers), "coordinates_x"]
        away_gk_x = period_events.loc[period_events["player_id"].isin(away_goalkeepers), "coordinates_x"]
        if home_gk_x.empty or away_gk_x.empty:
            continue
        if home_gk_x.mean() <= away_gk_x.mean():
            continue

        idx = period_events.index
        coord_x = pd.to_numeric(period_events["coordinates_x"], errors="coerce")
        coord_y = pd.to_numeric(period_events["coordinates_y"], errors="coerce")
        events.loc[idx, "coordinates_x"] = (FIELD_LENGTH - coord_x).round(2)
        events.loc[idx, "coordinates_y"] = (FIELD_WIDTH - coord_y).round(2)
        if "end_coordinates_x" in events.columns:
            end_x = pd.to_numeric(period_events["end_coordinates_x"], errors="coerce")
            events.loc[idx, "end_coordinates_x"] = (FIELD_LENGTH - end_x).round(2)
        if "end_coordinates_y" in events.columns:
            end_y = pd.to_numeric(period_events["end_coordinates_y"], errors="coerce")
            events.loc[idx, "end_coordinates_y"] = (FIELD_WIDTH - end_y).round(2)

    return events


def compute_period_starts(raw_events: pd.DataFrame, kickoff_time: pd.Timestamp | None = None) -> dict[int, pd.Timestamp]:
    period_starts: dict[int, pd.Timestamp] = {}
    kickoff_events = raw_events.loc[raw_events["set_piece_type"] == "KickOff"].copy()

    for period_id in sorted(raw_events["period_id"].dropna().astype(int).unique()):
        if period_id <= 0:
            continue

        period_kickoffs = kickoff_events.loc[kickoff_events["period_id"] == period_id, "utc_timestamp"]
        if not period_kickoffs.empty:
            period_starts[period_id] = pd.Timestamp(period_kickoffs.iloc[0])
            continue

        period_events = raw_events.loc[raw_events["period_id"] == period_id, "utc_timestamp"]
        if not period_events.empty:
            period_starts[period_id] = pd.Timestamp(period_events.iloc[0])
        elif period_id == 1 and kickoff_time is not None:
            period_starts[period_id] = pd.Timestamp(kickoff_time)

    return period_starts


def classify_strict_spadl_type(events: pd.DataFrame) -> pd.Series:
    spadl_type = pd.Series(pd.NA, index=events.index, dtype="string")

    pass_mask = events["event_type"].eq("Pass")
    cross_mask = events["event_type"].eq("Cross")
    shot_mask = events["event_type"].eq("Shot")
    foul_mask = events["event_type"].eq("Foul")

    spadl_type.loc[pass_mask] = "pass"
    spadl_type.loc[pass_mask & events["set_piece_type"].eq("ThrowIn")] = "throw_in"
    spadl_type.loc[pass_mask & events["set_piece_type"].eq("GoalKick")] = "goalkick"
    spadl_type.loc[pass_mask & events["set_piece_type"].eq("CornerKick")] = "corner_short"
    spadl_type.loc[pass_mask & events["set_piece_type"].eq("FreeKick")] = "freekick_short"

    spadl_type.loc[cross_mask] = "cross"
    spadl_type.loc[cross_mask & events["set_piece_type"].eq("CornerKick")] = "corner_crossed"
    spadl_type.loc[cross_mask & events["set_piece_type"].eq("FreeKick")] = "freekick_crossed"

    spadl_type.loc[shot_mask] = "shot"
    spadl_type.loc[shot_mask & events["set_piece_type"].eq("FreeKick")] = "shot_freekick"
    spadl_type.loc[shot_mask & events["set_piece_type"].eq("Penalty")] = "shot_penalty"

    spadl_type.loc[foul_mask] = "foul"
    return spadl_type


def map_result_id(spadl_type: pd.Series, raw_result: pd.Series, success: pd.Series) -> pd.Series:
    result_ids = pd.Series(spadlconfig.results.index("fail"), index=spadl_type.index, dtype="int64")

    success_mask = success.eq(True)
    result_ids.loc[spadl_type.isin(PASS_LIKE_TYPES | SHOT_LIKE_TYPES) & success_mask] = spadlconfig.results.index("success")
    result_ids.loc[spadl_type.eq("foul")] = spadlconfig.results.index("fail")

    offside_mask = raw_result.astype("string").str.lower().eq("offside")
    result_ids.loc[offside_mask & spadl_type.isin(PASS_LIKE_TYPES)] = spadlconfig.results.index("offside")
    return result_ids


def map_bodypart_id(spadl_type: pd.Series, body_part_type: pd.Series) -> pd.Series:
    foot = spadlconfig.bodyparts.index("foot")
    other = spadlconfig.bodyparts.index("other")
    foot_left = spadlconfig.bodyparts.index("foot_left")
    foot_right = spadlconfig.bodyparts.index("foot_right")
    head = spadlconfig.bodyparts.index("head")

    bodypart_ids = pd.Series(other, index=spadl_type.index, dtype="int64")

    bodypart_ids.loc[spadl_type.isin(PASS_LIKE_TYPES)] = foot
    bodypart_ids.loc[spadl_type.eq("throw_in")] = other
    bodypart_ids.loc[spadl_type.isin(SHOT_LIKE_TYPES)] = foot
    bodypart_ids.loc[spadl_type.eq("foul")] = other

    body_series = body_part_type.astype("string")
    bodypart_ids.loc[spadl_type.isin(SHOT_LIKE_TYPES) & body_series.eq("Head")] = head
    bodypart_ids.loc[spadl_type.isin(SHOT_LIKE_TYPES) & body_series.eq("LeftFoot")] = foot_left
    bodypart_ids.loc[spadl_type.isin(SHOT_LIKE_TYPES) & body_series.eq("RightFoot")] = foot_right
    return bodypart_ids


def infer_receiver_end_locations(events: pd.DataFrame) -> pd.DataFrame:
    inferred = pd.DataFrame(index=events.index, columns=["end_x", "end_y"], dtype="float64")
    valid_starts = events["start_x"].notna() & events["start_y"].notna() & events["player_id"].notna()
    player_lookup: dict[tuple[int, str], np.ndarray] = {}

    for (period_id, player_id), index_values in events.loc[valid_starts].groupby(["period_id", "player_id"]).groups.items():
        player_lookup[(int(period_id), str(player_id))] = np.array(sorted(index_values), dtype=int)

    candidate_mask = (
        events["spadl_type"].isin(PASS_LIKE_TYPES)
        & events["success"].eq(True)
        & events["receiver_player_id"].notna()
    )

    for idx in events.index[candidate_mask]:
        key = (int(events.at[idx, "period_id"]), str(events.at[idx, "receiver_player_id"]))
        receiver_indices = player_lookup.get(key)
        if receiver_indices is None:
            continue

        next_pos = np.searchsorted(receiver_indices, idx + 1, side="left")
        if next_pos >= len(receiver_indices):
            continue

        receiver_idx = int(receiver_indices[next_pos])
        inferred.at[idx, "end_x"] = events.at[receiver_idx, "start_x"]
        inferred.at[idx, "end_y"] = events.at[receiver_idx, "start_y"]

    return inferred


def _period_seconds(events: pd.DataFrame, period_starts: dict[int, pd.Timestamp]) -> pd.Series:
    seconds = pd.Series(np.nan, index=events.index, dtype="float64")
    for period_id, start_ts in period_starts.items():
        period_mask = events["period_id"].eq(period_id)
        if not period_mask.any():
            continue
        period_seconds = (events.loc[period_mask, "utc_timestamp"] - start_ts).dt.total_seconds()
        seconds.loc[period_mask] = period_seconds.round(3)
    return seconds


def convert_sportec_events_to_spadl(
    match_id: str,
    raw_events: pd.DataFrame,
    lineup: pd.DataFrame,
    kickoff_time: pd.Timestamp | None = None,
    kloppy_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit: dict[str, Any] = {"match_id": match_id}

    events = raw_events.copy()
    events["event_id"] = events["event_id"].astype("string")
    events["player_id"] = events["player_id"].astype("string")
    events["receiver_player_id"] = events["receiver_player_id"].astype("string")
    events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
    events = align_sportec_event_orientations(lineup, events)
    events = events.sort_values(["period_id", "utc_timestamp", "event_id"], ignore_index=True)

    if kloppy_events is not None:
        kloppy_merge = kloppy_events.copy()
        kloppy_merge["event_id"] = kloppy_merge["event_id"].astype("string")
        kloppy_merge = kloppy_merge.rename(
            columns={
                "coordinates_x": "kloppy_start_x",
                "coordinates_y": "kloppy_start_y",
                "end_coordinates_x": "kloppy_end_x",
                "end_coordinates_y": "kloppy_end_y",
            }
        )
        keep_cols = ["event_id", "kloppy_start_x", "kloppy_start_y", "kloppy_end_x", "kloppy_end_y"]
        events = events.merge(kloppy_merge[keep_cols], on="event_id", how="left")
    else:
        events["kloppy_start_x"] = np.nan
        events["kloppy_start_y"] = np.nan
        events["kloppy_end_x"] = np.nan
        events["kloppy_end_y"] = np.nan

    events["spadl_type"] = classify_strict_spadl_type(events)
    audit["raw_event_count"] = int(len(events))
    audit["mapped_event_count"] = int(events["spadl_type"].notna().sum())
    audit["dropped_unmapped_events"] = int(events["spadl_type"].isna().sum())
    audit["mapped_type_counts"] = dict(Counter(events.loc[events["spadl_type"].notna(), "spadl_type"]))

    events = events.loc[
        events["spadl_type"].notna()
        & events["period_id"].fillna(0).astype(int).gt(0)
        & events["team_id"].notna()
        & events["player_id"].notna()
    ].copy()
    audit["dropped_missing_core_fields"] = int(audit["mapped_event_count"] - len(events))

    period_starts = compute_period_starts(raw_events, kickoff_time=kickoff_time)
    events["time_seconds"] = _period_seconds(events, period_starts)
    negative_time_mask = events["time_seconds"].isna() | events["time_seconds"].lt(0)
    audit["dropped_negative_time_actions"] = int(negative_time_mask.sum())
    events = events.loc[~negative_time_mask].copy()

    events["start_x"] = events["kloppy_start_x"].combine_first(events["coordinates_x"]).astype(float)
    events["start_y"] = events["kloppy_start_y"].combine_first(events["coordinates_y"]).astype(float)

    prev_start_x = events.groupby("period_id")["start_x"].transform(lambda s: s.ffill().shift())
    prev_start_y = events.groupby("period_id")["start_y"].transform(lambda s: s.ffill().shift())
    foul_missing_start = events["spadl_type"].eq("foul") & (events["start_x"].isna() | events["start_y"].isna())
    events.loc[foul_missing_start, "start_x"] = prev_start_x.loc[foul_missing_start]
    events.loc[foul_missing_start, "start_y"] = prev_start_y.loc[foul_missing_start]
    audit["repaired_foul_coordinates"] = int(
        (foul_missing_start & events["start_x"].notna() & events["start_y"].notna()).sum()
    )

    events["end_x"] = events["kloppy_end_x"].astype(float)
    events["end_y"] = events["kloppy_end_y"].astype(float)
    inferred_end = infer_receiver_end_locations(events)
    events["end_x"] = events["end_x"].combine_first(inferred_end["end_x"])
    events["end_y"] = events["end_y"].combine_first(inferred_end["end_y"])
    events["end_x"] = events["end_x"].combine_first(events["start_x"])
    events["end_y"] = events["end_y"].combine_first(events["start_y"])

    for col, upper in [("start_x", FIELD_LENGTH), ("end_x", FIELD_LENGTH), ("start_y", FIELD_WIDTH), ("end_y", FIELD_WIDTH)]:
        events[col] = events[col].clip(lower=0.0, upper=upper).round(2)

    coord_mask = events[["start_x", "start_y", "end_x", "end_y"]].notna().all(axis=1)
    audit["dropped_missing_coordinates"] = int((~coord_mask).sum())
    events = events.loc[coord_mask].copy()

    events["result_id"] = map_result_id(events["spadl_type"], events["result"], events["success"])
    events["bodypart_id"] = map_bodypart_id(events["spadl_type"], events["body_part_type"])
    events["type_id"] = events["spadl_type"].apply(spadlconfig.actiontypes.index).astype("int64")

    actions = pd.DataFrame(
        {
            "game_id": match_id,
            "original_event_id": events["event_id"].astype("string"),
            "period_id": events["period_id"].astype("int64"),
            "time_seconds": events["time_seconds"].astype(float).round(3),
            "team_id": events["team_id"].astype("string"),
            "player_id": events["player_id"].astype("string"),
            "start_x": events["start_x"].astype(float),
            "start_y": events["start_y"].astype(float),
            "end_x": events["end_x"].astype(float),
            "end_y": events["end_y"].astype(float),
            "type_id": events["type_id"].astype("int64"),
            "result_id": events["result_id"].astype("int64"),
            "bodypart_id": events["bodypart_id"].astype("int64"),
        }
    ).sort_values(["game_id", "period_id", "time_seconds", "original_event_id"], kind="mergesort")
    actions = actions.reset_index(drop=True)
    actions["action_id"] = range(len(actions))
    for column in ["game_id", "original_event_id", "team_id", "player_id"]:
        actions[column] = actions[column].astype(object)
    actions = _fix_clearances(actions)

    action_count_before_dribbles = len(actions)
    actions = _add_dribbles(actions)
    actions = add_spadl_names(actions)

    try:
        validated_actions = SPADLSchema.validate(actions)
    except SchemaError as exc:
        raise RuntimeError(f"Strict SPADL validation failed for {match_id}") from exc

    audit["auto_added_dribbles"] = int(len(validated_actions) - action_count_before_dribbles)
    audit["final_action_count"] = int(len(validated_actions))
    audit["final_type_counts"] = dict(Counter(validated_actions["type_name"]))
    return validated_actions, audit
