from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd

import datatools.preprocess as tracking_preprocess
from project_config import (
    DATA_ROOT,
    EVENT_PATH,
    EVENT_SYNCED_DIR,
    LINEUP_PATH,
    RAW_EVENT_DIR,
    RAW_META_DIR,
    RAW_TRACKING_DIR,
    TRACKING_DIR,
    TRACKING_PROCESSED_DIR,
    TRAIN_POOL_SIZE,
    ensure_project_dirs,
    save_split_manifest,
)
from sync import config as sync_config
from sync import elastic
from tools.match_data import MatchData
from tools.sportec_data import SportecData

try:
    from kloppy import sportec
    from kloppy.domain import Dimension, MetricPitchDimensions, Orientation
    from pandera.errors import SchemaError
    from socceraction.spadl import add_names as add_spadl_names
    from socceraction.spadl.kloppy import convert_to_actions
except ImportError as exc:  # pragma: no cover - validated at runtime once dependencies are installed
    raise SystemExit(
        "Missing preprocessing dependencies. Install kloppy, socceraction and pandera before running this script."
    ) from exc


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

SPORTTEC_TO_ADVANCED_POSITION = {
    None: "unknown",
    "TW": "goal_keeper",
    "IVR": "center_back",
    "IVL": "center_back",
    "IVZ": "center_back",
    "RV": "right_back",
    "LV": "left_back",
    "DMR": "defensive_midfielder",
    "DRM": "defensive_midfielder",
    "DML": "defensive_midfielder",
    "DLM": "defensive_midfielder",
    "DMZ": "defensive_midfielder",
    "HR": "central_midfielder",
    "HL": "central_midfielder",
    "MZ": "central_midfielder",
    "RM": "right_midfielder",
    "LM": "left_midfielder",
    "OHR": "attacking_midfielder",
    "OHL": "attacking_midfielder",
    "ORM": "attacking_midfielder",
    "OLM": "attacking_midfielder",
    "ZO": "attacking_midfielder",
    "RA": "right_winger",
    "LA": "left_winger",
    "STR": "striker",
    "STL": "striker",
    "STZ": "striker",
}


@dataclass(frozen=True)
class MatchFiles:
    match_id: str
    meta_path: Path
    event_path: Path
    tracking_path: Path


class SportecSpadlData(MatchData):
    def __init__(self, lineup: pd.DataFrame, events: pd.DataFrame, tracking: pd.DataFrame, fps: float = 25.0):
        super().__init__()
        self.lineup = lineup.copy()
        self.events = events.copy().sort_values(["period_id", "utc_timestamp", "original_event_id"]).reset_index(
            drop=True
        )
        self.tracking = tracking.copy().sort_values(["period_id", "timestamp"], ignore_index=True)
        self.fps = fps

    def format_events_for_syncer(self) -> pd.DataFrame:
        input_cols = ["period_id", "utc_timestamp", "object_id", "spadl_type", "start_x", "start_y", "success"]
        return self.events[input_cols].rename(columns={"object_id": "player_id"}).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", action="append", help="Process only the specified match id(s).")
    parser.add_argument("--limit", type=int, help="Process only the first N discovered matches.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing outputs.")
    parser.add_argument("--skip-sync", action="store_true", help="Skip ELASTIC synchronization.")
    return parser.parse_args()


def to_utc_naive(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def discover_match_files() -> list[MatchFiles]:
    meta_files = {path.stem: path for path in RAW_META_DIR.glob("*.xml")}
    event_files = {path.stem: path for path in RAW_EVENT_DIR.glob("*.xml")}
    tracking_files = {path.stem: path for path in RAW_TRACKING_DIR.glob("*.xml")}
    match_ids = sorted(meta_files.keys() & event_files.keys() & tracking_files.keys())
    return [MatchFiles(match_id, meta_files[match_id], event_files[match_id], tracking_files[match_id]) for match_id in match_ids]


def parse_match_information(match_files: MatchFiles) -> tuple[pd.DataFrame, dict[str, object]]:
    tree = ET.parse(match_files.meta_path)
    root = tree.getroot()
    general = root.find(".//General")
    if general is None:
        raise ValueError(f"Missing General section in {match_files.meta_path}")

    kickoff_time = to_utc_naive(general.attrib["KickoffTime"])
    match_title = general.attrib.get("MatchTitle") or f"{general.attrib.get('HomeTeamName')}:{general.attrib.get('GuestTeamName')}"
    other_info = root.find(".//OtherGameInformation")

    period_totals_ms = {
        1: int(other_info.attrib.get("TotalTimeFirstHalf", "0")) if other_info is not None else 0,
        2: int(other_info.attrib.get("TotalTimeSecondHalf", "0")) if other_info is not None else 0,
    }

    lineup_rows: list[dict[str, object]] = []

    for team in root.findall(".//Team"):
        team_name = team.attrib["TeamName"]
        team_role = team.attrib.get("Role")
        team_prefix = "away" if team_role == "guest" else "home"
        formation = team.attrib.get("LineUp")

        for player in team.findall("./Players/Player"):
            shirt_number = int(player.attrib["ShirtNumber"])
            shortname = player.attrib.get("Shortname") or f"{player.attrib.get('FirstName', '')} {player.attrib.get('LastName', '')}".strip()
            lineup_rows.append(
                {
                    "stats_perform_match_id": match_files.match_id,
                    "game_date": kickoff_time.date().isoformat(),
                    "contestant_name": team_name,
                    "player_id": player.attrib["PersonId"],
                    "object_id": f"{team_prefix}_{shirt_number}",
                    "shirt_number": shirt_number,
                    "match_name": shortname,
                    "formation": formation,
                    "advanced_position": SPORTTEC_TO_ADVANCED_POSITION.get(player.attrib.get("PlayingPosition"), "unknown"),
                    "mins_played": 0.0,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "starting": player.attrib.get("Starting") == "true",
                    "team_role": team_role,
                    "team_prefix": team_prefix,
                    "raw_playing_position": player.attrib.get("PlayingPosition"),
                }
            )

    lineup = pd.DataFrame(lineup_rows).sort_values(["contestant_name", "shirt_number"], ignore_index=True)
    metadata = {
        "match_id": match_files.match_id,
        "kickoff_time": kickoff_time,
        "match_title": match_title,
        "period_totals_ms": period_totals_ms,
    }
    return lineup, metadata


def extract_vendor_xg(event_path: Path) -> dict[str, float]:
    tree = ET.parse(event_path)
    root = tree.getroot()
    xg_map: dict[str, float] = {}

    for event in root.findall(".//Event"):
        shot = event.find("./ShotAtGoal")
        if shot is not None and shot.attrib.get("xG") is not None:
            xg_map[event.attrib["EventId"]] = float(shot.attrib["xG"])

    return xg_map


def compute_period_bounds(raw_events: pd.DataFrame, metadata: dict[str, object]) -> tuple[dict[int, pd.Timestamp], dict[int, float]]:
    kickoff_events = raw_events[raw_events["set_piece_type"] == "KickOff"].copy()
    final_events = raw_events[raw_events["event_type"] == "FinalWhistle"].copy()

    period_starts: dict[int, pd.Timestamp] = {}
    period_lengths: dict[int, float] = {}

    for period_id in [1, 2]:
        period_kickoffs = kickoff_events[kickoff_events["period_id"] == period_id]
        period_finals = final_events[final_events["period_id"] == period_id]

        if not period_kickoffs.empty:
            period_starts[period_id] = period_kickoffs["utc_timestamp"].iloc[0]
        elif period_id == 1:
            period_starts[period_id] = metadata["kickoff_time"]

        if period_id in period_starts and not period_finals.empty:
            period_lengths[period_id] = (
                period_finals["utc_timestamp"].iloc[-1] - period_starts[period_id]
            ).total_seconds()
        else:
            period_lengths[period_id] = metadata["period_totals_ms"].get(period_id, 0) / 1000

    return period_starts, period_lengths


def match_clock_seconds(timestamp: pd.Timestamp, period_id: int, period_starts: dict[int, pd.Timestamp], period_lengths: dict[int, float]) -> float:
    seconds_before = sum(period_lengths.get(prev_period, 0.0) for prev_period in sorted(period_lengths) if prev_period < period_id)
    if period_id not in period_starts:
        return seconds_before
    return seconds_before + max((timestamp - period_starts[period_id]).total_seconds(), 0.0)


def finalize_lineup(
    lineup: pd.DataFrame,
    raw_events: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    lineup = lineup.copy()
    period_starts, period_lengths = compute_period_bounds(raw_events, metadata)
    match_duration = float(sum(period_lengths.values()))

    starters = lineup["starting"].fillna(False)
    lineup.loc[starters, "start_time"] = 0.0
    lineup.loc[starters, "end_time"] = match_duration
    lineup.loc[~starters, ["start_time", "end_time"]] = 0.0

    substitutions = raw_events[raw_events["event_type"] == "Substitution"].sort_values("utc_timestamp")
    for _, substitution in substitutions.iterrows():
        event_time = match_clock_seconds(substitution["utc_timestamp"], int(substitution["period_id"]), period_starts, period_lengths)
        player_out = substitution["player_id"]
        player_in = substitution["receiver_player_id"]

        lineup.loc[lineup["player_id"] == player_out, "end_time"] = event_time
        lineup.loc[lineup["player_id"] == player_in, ["start_time", "end_time"]] = [event_time, match_duration]

    red_cards = raw_events[(raw_events["event_type"] == "Card") & (raw_events["card_type"] == "Red")].sort_values(
        "utc_timestamp"
    )
    for _, card in red_cards.iterrows():
        event_time = match_clock_seconds(card["utc_timestamp"], int(card["period_id"]), period_starts, period_lengths)
        lineup.loc[lineup["player_id"] == card["player_id"], "end_time"] = np.minimum(
            lineup.loc[lineup["player_id"] == card["player_id"], "end_time"].astype(float),
            event_time,
        )

    lineup["start_time"] = lineup["start_time"].fillna(0.0).astype(float)
    lineup["end_time"] = lineup["end_time"].fillna(0.0).astype(float)
    lineup["end_time"] = np.maximum(lineup["end_time"], lineup["start_time"])
    lineup["mins_played"] = ((lineup["end_time"] - lineup["start_time"]) / 60.0).round(2)

    ordered_cols = [
        "stats_perform_match_id",
        "game_date",
        "contestant_name",
        "player_id",
        "object_id",
        "shirt_number",
        "match_name",
        "formation",
        "advanced_position",
        "mins_played",
        "start_time",
        "end_time",
        "starting",
        "team_prefix",
    ]
    return lineup[ordered_cols].copy()


def build_pitch_dimensions() -> MetricPitchDimensions:
    return MetricPitchDimensions(
        standardized=True,
        x_dim=Dimension(0, FIELD_LENGTH),
        y_dim=Dimension(0, FIELD_WIDTH),
    )


def build_tracking_outputs(match_files: MatchFiles, lineup: pd.DataFrame, overwrite: bool) -> tuple[pd.DataFrame, float]:
    raw_output = TRACKING_DIR / f"{match_files.match_id}.parquet"
    processed_output = TRACKING_PROCESSED_DIR / f"{match_files.match_id}.parquet"

    if raw_output.exists() and processed_output.exists() and not overwrite:
        tracking = pd.read_parquet(raw_output)
        fps = 25.0
        return tracking, fps

    sync_lineup = lineup[["player_id", "object_id"]].copy()
    tracking_ds, tracking = SportecData.load_tracking_data(
        str(match_files.tracking_path),
        str(match_files.meta_path),
        sync_lineup,
    )
    fps = float(tracking_ds.frame_rate)
    tracking.to_parquet(raw_output, index=False)

    tracking_processed = tracking.copy()
    tracking_processed[["timestamp", "ball_x", "ball_y"]] = tracking_processed[["timestamp", "ball_x", "ball_y"]].round(2)
    if "ball_z" in tracking_processed:
        tracking_processed["ball_z"] = tracking_processed["ball_z"].astype(float).round(2)
    tracking_processed = tracking_preprocess.label_frames_and_episodes(tracking_processed, fps=int(round(fps)))
    tracking_processed = tracking_preprocess.calc_physical_features(tracking_processed, fps=int(round(fps)))
    tracking_processed.to_parquet(processed_output)
    return tracking, fps


def build_spadl_actions(match_files: MatchFiles) -> pd.DataFrame:
    pitch_dims = build_pitch_dimensions()
    event_ds = sportec.load_event(
        event_data=str(match_files.event_path),
        meta_data=str(match_files.meta_path),
        coordinates="sportec",
    )
    event_ds = event_ds.transform(
        to_orientation=Orientation.STATIC_HOME_AWAY,
        to_pitch_dimensions=pitch_dims,
    )

    actions = convert_to_actions(event_ds, game_id=match_files.match_id)
    actions = add_spadl_names(actions)
    return actions


def build_defcon_event_table(
    match_id: str,
    actions: pd.DataFrame,
    lineup: pd.DataFrame,
    raw_events: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    raw_events = raw_events.copy()
    raw_events["event_id"] = raw_events["event_id"].astype("string")

    period_starts, _ = compute_period_bounds(raw_events, metadata)
    lineup_lookup = lineup.set_index("player_id")

    actions = actions.copy()
    actions["original_event_id"] = actions["original_event_id"].astype("string")
    actions = actions.merge(
        raw_events[["event_id", "expected_goal"]].rename(columns={"event_id": "original_event_id"}),
        how="left",
        on="original_event_id",
    )

    actions["utc_timestamp"] = actions.apply(
        lambda row: period_starts[int(row["period_id"])] + pd.to_timedelta(float(row["time_seconds"]), unit="s"),
        axis=1,
    )
    actions["player_id"] = actions["player_id"].astype("string")
    actions["object_id"] = actions["player_id"].map(lineup_lookup["object_id"].to_dict())
    actions["player_name"] = actions["player_id"].map(lineup_lookup["match_name"].to_dict())
    actions["advanced_position"] = actions["player_id"].map(lineup_lookup["advanced_position"].to_dict())
    actions["spadl_type"] = actions["type_name"]
    actions["success"] = actions["result_name"].eq("success")
    actions["offside"] = actions["result_name"].eq("offside")
    actions["expected_goal"] = actions["expected_goal"].fillna(0.0).astype(float)

    supported_types = set(sync_config.SPADL_TYPES)
    actions = actions[actions["spadl_type"].isin(supported_types)].copy()
    actions = actions[actions["object_id"].notna()].copy()
    actions["utc_timestamp"] = pd.to_datetime(actions["utc_timestamp"]).dt.tz_localize(None)

    column_map = {
        "game_id": "stats_perform_match_id",
        "action_id": "action_id",
        "original_event_id": "original_event_id",
        "period_id": "period_id",
        "time_seconds": "seconds",
        "utc_timestamp": "utc_timestamp",
        "player_id": "player_id",
        "object_id": "object_id",
        "player_name": "player_name",
        "advanced_position": "advanced_position",
        "team_id": "team_id",
        "spadl_type": "spadl_type",
        "success": "success",
        "offside": "offside",
        "expected_goal": "expected_goal",
        "start_x": "start_x",
        "start_y": "start_y",
        "end_x": "end_x",
        "end_y": "end_y",
    }

    events = actions[column_map.keys()].rename(columns=column_map)
    events["stats_perform_match_id"] = match_id
    events = events.sort_values(["period_id", "seconds", "action_id"], ignore_index=True)
    return events


def sync_events(
    match_id: str,
    lineup: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: float,
    overwrite: bool,
) -> None:
    output_path = EVENT_SYNCED_DIR / f"{match_id}.csv"
    if output_path.exists() and not overwrite:
        return

    sync_input = SportecSpadlData(lineup, events, tracking, fps=fps)
    input_events = sync_input.format_events_for_syncer()
    input_tracking = sync_input.format_tracking_for_syncer()

    try:
        syncer = elastic.ELASTIC(input_events, input_tracking)
        syncer.run()
    except SchemaError as exc:
        raise RuntimeError(f"Synchronization schema validation failed for {match_id}") from exc

    sync_input.events[sync_config.EVENT_COLS[:4]] = syncer.events[sync_config.EVENT_COLS[:4]]
    sync_input.events[sync_config.NEXT_EVENT_COLS] = syncer.events[sync_config.NEXT_EVENT_COLS]
    output_events = sync_input.events[sync_config.EVENT_COLS + sync_config.NEXT_EVENT_COLS].copy()

    synced_events = sync_input.events[sync_input.events["frame_id"].notna()]
    if not synced_events.empty:
        last_synced_event = synced_events.iloc[-1]
        last_synced_episode = syncer.frames.at[last_synced_event["frame_id"], "episode_id"]
        if last_synced_episode >= syncer.frames["episode_id"].max() - 1:
            output_events = output_events.loc[: last_synced_event.name]

    output_events.to_csv(output_path, index=False, encoding="utf-8")


def build_split_manifest(metadata_records: pd.DataFrame) -> None:
    ordered = metadata_records.sort_values(["kickoff_time", "match_id"], ignore_index=True)
    if len(ordered) <= TRAIN_POOL_SIZE:
        raise ValueError(f"Need more than {TRAIN_POOL_SIZE} matches to create train/test splits.")

    train_ids = ordered["match_id"].iloc[:TRAIN_POOL_SIZE].tolist()
    test_ids = ordered["match_id"].iloc[TRAIN_POOL_SIZE:].tolist()
    save_split_manifest(
        train_ids,
        test_ids,
        metadata={
            "train_pool_size": TRAIN_POOL_SIZE,
            "test_size": len(test_ids),
            "split_rule": "sorted by kickoff_time, then match_id",
        },
    )


def filter_matches(matches: list[MatchFiles], requested_ids: Iterable[str] | None, limit: int | None) -> list[MatchFiles]:
    if requested_ids:
        requested = set(requested_ids)
        matches = [match for match in matches if match.match_id in requested]
    if limit is not None:
        matches = matches[:limit]
    return matches


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    all_matches = discover_match_files()
    if not all_matches:
        raise FileNotFoundError("No Sportec XML files found in the raw data directories.")

    metadata_records: list[dict[str, object]] = []
    processed_lineups: list[pd.DataFrame] = []
    processed_events: list[pd.DataFrame] = []

    for match_files in all_matches:
        _, metadata = parse_match_information(match_files)
        metadata_records.append({"match_id": match_files.match_id, "kickoff_time": metadata["kickoff_time"]})

    metadata_df = pd.DataFrame(metadata_records)
    build_split_manifest(metadata_df)

    selected_matches = filter_matches(all_matches, args.match_id, args.limit)
    if not selected_matches:
        raise ValueError("No matches selected for processing.")

    for index, match_files in enumerate(selected_matches, start=1):
        print(f"[{index}/{len(selected_matches)}] {match_files.match_id}")

        lineup, metadata = parse_match_information(match_files)
        raw_events = SportecData.load_event_data(str(match_files.event_path))
        raw_events["event_id"] = raw_events["event_id"].astype("string")
        raw_events["expected_goal"] = raw_events["event_id"].map(extract_vendor_xg(match_files.event_path)).astype(float)

        finalized_lineup = finalize_lineup(lineup, raw_events, metadata)
        tracking, fps = build_tracking_outputs(match_files, finalized_lineup, overwrite=args.overwrite)

        actions = build_spadl_actions(match_files)
        defcon_events = build_defcon_event_table(match_files.match_id, actions, finalized_lineup, raw_events, metadata)

        if not args.skip_sync:
            sync_events(match_files.match_id, finalized_lineup, defcon_events, tracking, fps=fps, overwrite=args.overwrite)

        processed_lineups.append(finalized_lineup.drop(columns=["starting", "team_prefix"]))
        processed_events.append(defcon_events)

    all_lineups = pd.concat(processed_lineups, ignore_index=True)
    all_lineups.to_parquet(LINEUP_PATH, index=False)

    all_events = pd.concat(processed_events, ignore_index=True)
    all_events.to_parquet(EVENT_PATH, index=False)

    print(f"Saved lineup parquet to {LINEUP_PATH}")
    print(f"Saved unsynced event parquet to {EVENT_PATH}")
    if not args.skip_sync:
        print(f"Saved synced per-match CSV files to {EVENT_SYNCED_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
