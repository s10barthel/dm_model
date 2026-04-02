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
    EVENT_PATH,
    EVENT_SYNCED_DIR,
    LINEUP_PATH,
    RAW_EVENT_DIR,
    RAW_KPI_DIR,
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
from sync import utils as sync_utils
from tools.match_data import MatchData
from tools.sportec_spadl import convert_sportec_events_to_spadl
from tools.sportec_data import SportecData

try:
    from kloppy import sportec
    from kloppy.domain import Dimension, MetricPitchDimensions, Orientation
    from pandera.errors import SchemaError
except ImportError as exc:  # pragma: no cover - validated at runtime once dependencies are installed
    raise SystemExit(
        "Missing preprocessing dependencies. Install kloppy, socceraction and pandera before running this script."
    ) from exc


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
SYNC_SOURCES = ("sportec_kpi", "elastic")
KPI_RECEIVE_TYPES = {
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
}

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

CANONICAL_LINEUP_COLS = [
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
]

INTERNAL_LINEUP_COLS = CANONICAL_LINEUP_COLS + ["starting", "team_prefix"]


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
    parser.add_argument("--skip-sync", action="store_true", help="Skip event-tracking synchronization.")
    parser.add_argument(
        "--sync-source",
        choices=SYNC_SOURCES,
        default="sportec_kpi",
        help="Synchronization source for canonical event outputs.",
    )
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
                    "team_prefix": team_prefix,
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


def collect_match_metadata(matches: Iterable[MatchFiles]) -> pd.DataFrame:
    metadata_records: list[dict[str, object]] = []
    for match_files in matches:
        _, metadata = parse_match_information(match_files)
        metadata_records.append({"match_id": match_files.match_id, "kickoff_time": metadata["kickoff_time"]})
    return pd.DataFrame(metadata_records)


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def sort_matches_by_kickoff(matches: list[MatchFiles]) -> tuple[list[MatchFiles], list[dict[str, str]]]:
    if not matches:
        return [], []

    metadata_records: list[dict[str, object]] = []
    skipped_matches: list[dict[str, str]] = []
    for match_files in matches:
        try:
            _, metadata = parse_match_information(match_files)
        except Exception as exc:
            skipped_matches.append({"match_id": match_files.match_id, "error": summarize_exception(exc)})
            continue
        metadata_records.append({"match_id": match_files.match_id, "kickoff_time": metadata["kickoff_time"]})

    if not metadata_records:
        return [], skipped_matches

    metadata = pd.DataFrame(metadata_records).sort_values(["kickoff_time", "match_id"], ignore_index=True)
    match_lookup = {match.match_id: match for match in matches}
    return [match_lookup[match_id] for match_id in metadata["match_id"].tolist()], skipped_matches


def extract_vendor_xg(event_path: Path) -> dict[str, float]:
    tree = ET.parse(event_path)
    root = tree.getroot()
    xg_map: dict[str, float] = {}

    for event in root.findall(".//Event"):
        shot = event.find("./ShotAtGoal")
        if shot is not None and shot.attrib.get("xG") is not None:
            xg_map[event.attrib["EventId"]] = float(shot.attrib["xG"])

    return xg_map


def load_match_raw_events(match_files: MatchFiles) -> pd.DataFrame:
    raw_events = SportecData.load_event_data(str(match_files.event_path))
    raw_events["event_id"] = raw_events["event_id"].astype("string")
    raw_events["expected_goal"] = raw_events["event_id"].map(extract_vendor_xg(match_files.event_path)).astype(float)
    return raw_events


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

    return lineup[INTERNAL_LINEUP_COLS].copy()


def export_lineup_table(lineup: pd.DataFrame) -> pd.DataFrame:
    return lineup.drop(columns=["starting", "team_prefix"], errors="ignore").copy()


def build_pitch_dimensions() -> MetricPitchDimensions:
    return MetricPitchDimensions(
        standardized=True,
        x_dim=Dimension(0, FIELD_LENGTH),
        y_dim=Dimension(0, FIELD_WIDTH),
    )


def load_kloppy_tracking(match_files: MatchFiles, lineup: pd.DataFrame) -> tuple[object, pd.DataFrame, float]:
    sync_lineup = lineup[["player_id", "object_id"]].copy()
    tracking_ds, tracking = SportecData.load_tracking_data(
        str(match_files.tracking_path),
        str(match_files.meta_path),
        sync_lineup,
    )
    fps = float(tracking_ds.frame_rate)
    return tracking_ds, tracking, fps


def build_kloppy_tracking_table(tracking: pd.DataFrame, raw_events: pd.DataFrame, fps: float) -> pd.DataFrame:
    tracking_export = MatchData.calculate_tracking_datetimes(raw_events, tracking, fps=int(round(fps))).copy()
    tracking_export["timestamp"] = tracking_export["timestamp"].astype(float).round(3)
    if "ball_z" in tracking_export.columns:
        tracking_export["ball_z"] = tracking_export["ball_z"].astype(float).round(3)

    lead_cols = [col for col in ["frame_id", "period_id", "timestamp", "utc_timestamp"] if col in tracking_export.columns]
    ordered_cols = lead_cols + [col for col in tracking_export.columns if col not in lead_cols]
    return tracking_export[ordered_cols].sort_values(["frame_id"], ignore_index=True)


def build_tracking_outputs(
    match_files: MatchFiles,
    lineup: pd.DataFrame,
    raw_events: pd.DataFrame,
    overwrite: bool,
) -> tuple[pd.DataFrame, float]:
    raw_output = TRACKING_DIR / f"{match_files.match_id}.parquet"
    processed_output = TRACKING_PROCESSED_DIR / f"{match_files.match_id}.parquet"

    if raw_output.exists() and processed_output.exists() and not overwrite:
        try:
            tracking = pd.read_parquet(raw_output)
            pd.read_parquet(processed_output)
            fps = 25.0
            return tracking, fps
        except Exception as exc:
            print(f"  Rebuilding cached tracking outputs after read failure: {summarize_exception(exc)}")

    _, tracking, fps = load_kloppy_tracking(match_files, lineup)
    tracking.to_parquet(raw_output, index=False)

    tracking_processed = tracking.copy()
    if "ball_owning_home_away" not in tracking_processed.columns and "ball_owning_team_id" in tracking_processed.columns:
        team_prefix_map = (
            raw_events[["team_id", "player_id"]]
            .dropna()
            .merge(lineup[["player_id", "team_prefix"]].drop_duplicates(), on="player_id", how="inner")
            .drop_duplicates(["team_id", "team_prefix"])
            .set_index("team_id")["team_prefix"]
            .to_dict()
        )
        tracking_processed["ball_owning_home_away"] = tracking_processed["ball_owning_team_id"].map(team_prefix_map)
    tracking_processed[["timestamp", "ball_x", "ball_y"]] = tracking_processed[["timestamp", "ball_x", "ball_y"]].round(2)
    if "ball_z" in tracking_processed:
        tracking_processed["ball_z"] = tracking_processed["ball_z"].astype(float).round(2)
    tracking_processed = tracking_preprocess.label_frames_and_episodes(tracking_processed, fps=int(round(fps)))
    tracking_processed = tracking_preprocess.calc_physical_features(tracking_processed, fps=int(round(fps)))
    tracking_processed.to_parquet(processed_output)
    return tracking, fps


def build_kloppy_event_dataset(match_files: MatchFiles) -> object:
    event_ds = sportec.load_event(
        event_data=str(match_files.event_path),
        meta_data=str(match_files.meta_path),
        coordinates="sportec",
    )
    return event_ds.transform(
        to_orientation=Orientation.STATIC_HOME_AWAY,
        to_pitch_dimensions=build_pitch_dimensions(),
    )


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds().round(3)


def build_kloppy_event_table(event_ds: object) -> pd.DataFrame:
    event_df = event_ds.to_df().reset_index(drop=True).copy()
    for column in ["event_id", "team_id", "player_id", "receiver_player_id"]:
        if column in event_df.columns:
            event_df[column] = event_df[column].astype("string")

    if "timestamp" in event_df.columns:
        event_df["seconds"] = _timedelta_to_seconds(event_df["timestamp"])
    if "end_timestamp" in event_df.columns:
        event_df["end_seconds"] = _timedelta_to_seconds(event_df["end_timestamp"])

    preferred_cols = [
        "event_id",
        "event_type",
        "period_id",
        "seconds",
        "end_seconds",
        "timestamp",
        "end_timestamp",
        "ball_state",
        "ball_owning_team",
        "team_id",
        "player_id",
        "receiver_player_id",
        "coordinates_x",
        "coordinates_y",
        "end_coordinates_x",
        "end_coordinates_y",
        "set_piece_type",
        "result",
        "success",
        "body_part_type",
        "card_type",
    ]
    ordered_cols = [col for col in preferred_cols if col in event_df.columns]
    ordered_cols.extend(col for col in event_df.columns if col not in ordered_cols)
    return event_df[ordered_cols]


def build_spadl_actions(
    match_files: MatchFiles,
    lineup: pd.DataFrame,
    raw_events: pd.DataFrame,
    metadata: dict[str, object],
    event_ds: object | None = None,
    kloppy_events: pd.DataFrame | None = None,
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    if event_ds is None:
        event_ds = build_kloppy_event_dataset(match_files)
    if kloppy_events is None:
        kloppy_events = build_kloppy_event_table(event_ds)

    actions, audit = convert_sportec_events_to_spadl(
        match_id=match_files.match_id,
        raw_events=raw_events,
        lineup=lineup,
        kickoff_time=pd.Timestamp(metadata["kickoff_time"]),
        kloppy_events=kloppy_events,
    )
    if return_audit:
        return actions, audit
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
    actions["expected_goal"] = actions["expected_goal"].astype(float)

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


def load_kpi_merged_table(match_id: str) -> pd.DataFrame:
    kpi_path = RAW_KPI_DIR / f"KPI_MGD_{match_id}.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"Missing KPI_Merged file for {match_id}: {kpi_path}")

    kpi = pd.read_csv(
        kpi_path,
        sep=";",
        decimal=",",
        encoding="cp1252",
        low_memory=False,
        dtype={"EVENT_ID": "string", "PUID2": "string"},
    )
    kpi["EVENT_ID"] = kpi["EVENT_ID"].astype("string")
    kpi["PUID2"] = kpi.get("PUID2", pd.Series(index=kpi.index, dtype="string")).astype("string")
    kpi["GDCP_EVENT_TIME"] = pd.to_datetime(kpi.get("GDCP_EVENT_TIME"), errors="coerce")
    kpi["TRACKING_TIME"] = pd.to_datetime(kpi.get("TRACKING_TIME"), errors="coerce")
    kpi["FRAME_NUMBER"] = pd.to_numeric(kpi.get("FRAME_NUMBER"), errors="coerce").astype("Int64")
    kpi["RECFRM"] = pd.to_numeric(kpi.get("RECFRM"), errors="coerce").astype("Int64")
    no_receiver = kpi.get("NORECEIVER", pd.Series(index=kpi.index, dtype="string")).astype("string").str.upper()
    kpi["NORECEIVER"] = no_receiver.map({"TRUE": True, "FALSE": False})
    return kpi.drop_duplicates("EVENT_ID", keep="first").copy()


def build_tracking_frame_table(events: pd.DataFrame, tracking: pd.DataFrame, fps: float) -> pd.DataFrame:
    tracking_frames = MatchData.calculate_tracking_datetimes(events, tracking, fps=int(round(fps))).copy()
    tracking_frames = tracking_frames[["frame_id", "period_id", "timestamp", "utc_timestamp"]].drop_duplicates("frame_id")
    tracking_frames["frame_id"] = pd.to_numeric(tracking_frames["frame_id"], errors="coerce").astype("Int64")
    tracking_frames["timestamp"] = tracking_frames["timestamp"].astype(float)
    tracking_frames["utc_timestamp"] = pd.to_datetime(tracking_frames["utc_timestamp"], errors="coerce")
    tracking_frames["synced_ts"] = tracking_frames["timestamp"].apply(sync_utils.seconds_to_timestamp)
    tracking_frames["local_tracking_time"] = (
        tracking_frames["utc_timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    )
    return tracking_frames.sort_values("frame_id", ignore_index=True)


def initialize_synced_output(events: pd.DataFrame) -> pd.DataFrame:
    output_events = events.copy().sort_values(["period_id", "seconds", "action_id"], ignore_index=True)
    output_events["frame_id"] = pd.Series(pd.NA, index=output_events.index, dtype="Int64")
    output_events["synced_ts"] = pd.Series(index=output_events.index, dtype="object")
    output_events["receiver_id"] = pd.Series(index=output_events.index, dtype="object")
    output_events["receive_frame_id"] = pd.Series(pd.NA, index=output_events.index, dtype="Int64")
    output_events["receive_ts"] = pd.Series(index=output_events.index, dtype="object")
    # Downstream DEFCON code expects next-player ids on the same normalized object-id axis as tracking.
    output_events["next_player_id"] = output_events.groupby("period_id")["object_id"].shift(-1)
    output_events["next_type"] = output_events.groupby("period_id")["spadl_type"].shift(-1)
    return output_events


def finalize_synced_output(output_events: pd.DataFrame, frame_table: pd.DataFrame) -> pd.DataFrame:
    output_events = output_events.copy().sort_values(["period_id", "seconds", "action_id"], ignore_index=True)
    frame_lookup = frame_table.set_index("frame_id")

    output_events["frame_id"] = pd.to_numeric(output_events["frame_id"], errors="coerce").astype("Int64")
    output_events["receive_frame_id"] = pd.to_numeric(output_events["receive_frame_id"], errors="coerce").astype("Int64")
    output_events["synced_ts"] = output_events["frame_id"].map(frame_lookup["synced_ts"])
    output_events["receive_ts"] = output_events["receive_frame_id"].map(frame_lookup["synced_ts"])

    synced_mask = output_events["frame_id"].notna()
    if synced_mask.any():
        last_synced_idx = output_events.index[synced_mask][-1]
        output_events = output_events.loc[:last_synced_idx].copy()

    # Recompute next-action references after any truncation so they remain aligned to canonical object ids.
    output_events["next_player_id"] = output_events.groupby("period_id")["object_id"].shift(-1)
    output_events["next_type"] = output_events.groupby("period_id")["spadl_type"].shift(-1)

    ordered_cols = [
        "stats_perform_match_id",
        "action_id",
        "original_event_id",
        "period_id",
        "seconds",
        "frame_id",
        "synced_ts",
        "utc_timestamp",
        "player_id",
        "object_id",
        "player_name",
        "advanced_position",
        "team_id",
        "spadl_type",
        "success",
        "offside",
        "expected_goal",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "next_player_id",
        "next_type",
        "receiver_id",
        "receive_frame_id",
        "receive_ts",
    ]
    return output_events[ordered_cols].reset_index(drop=True)


def run_elastic_synchronization(
    match_id: str,
    lineup: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    sync_input = SportecSpadlData(lineup, events, tracking, fps=fps)
    input_events = sync_input.format_events_for_syncer()
    input_tracking = sync_input.format_tracking_for_syncer()
    input_events["period_id"] = input_events["period_id"].astype("int64")
    input_tracking["period_id"] = input_tracking["period_id"].astype("int64")
    input_tracking["frame_id"] = input_tracking["frame_id"].astype("int64")

    try:
        syncer = elastic.ELASTIC(input_events, input_tracking)
        syncer.run()
    except SchemaError as exc:
        raise RuntimeError(f"Synchronization schema validation failed for {match_id}") from exc

    sync_input.events[sync_config.EVENT_COLS[:4]] = syncer.events[sync_config.EVENT_COLS[:4]]
    sync_input.events[sync_config.NEXT_EVENT_COLS] = syncer.events[sync_config.NEXT_EVENT_COLS]
    output_events = sync_input.events.copy()

    synced_events = sync_input.events[sync_input.events["frame_id"].notna()]
    if not synced_events.empty:
        last_synced_event = synced_events.iloc[-1]
        last_synced_episode = syncer.frames.at[last_synced_event["frame_id"], "episode_id"]
        if last_synced_episode >= syncer.frames["episode_id"].max() - 1:
            output_events = output_events.loc[: last_synced_event.name]

    frame_table = build_tracking_frame_table(events, tracking, fps)
    return finalize_synced_output(output_events, frame_table)


def run_kpi_synchronization(
    match_id: str,
    lineup: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_events = initialize_synced_output(events)
    frame_table = build_tracking_frame_table(events, tracking, fps)
    frame_lookup = frame_table.set_index("frame_id")
    valid_frames = set(frame_lookup.index.dropna().tolist())
    lineup_lookup = lineup[["player_id", "object_id"]].drop_duplicates().set_index("player_id")["object_id"].to_dict()
    kpi = load_kpi_merged_table(match_id)

    merged = output_events[["action_id", "original_event_id", "spadl_type"]].merge(
        kpi[["EVENT_ID", "FRAME_NUMBER", "RECFRM", "PUID2", "NORECEIVER"]],
        left_on="original_event_id",
        right_on="EVENT_ID",
        how="left",
    )

    frame_candidates = merged["FRAME_NUMBER"].where(merged["FRAME_NUMBER"].isin(valid_frames))
    output_events["frame_id"] = frame_candidates.astype("Int64")

    pass_like_mask = output_events["spadl_type"].isin(KPI_RECEIVE_TYPES)
    receive_candidates = merged["RECFRM"].where(merged["RECFRM"].isin(valid_frames))
    output_events.loc[pass_like_mask, "receive_frame_id"] = receive_candidates.loc[pass_like_mask].astype("Int64")

    kpi_receiver_ids = merged["PUID2"].map(lineup_lookup)
    output_events.loc[pass_like_mask & kpi_receiver_ids.notna(), "receiver_id"] = kpi_receiver_ids.loc[
        pass_like_mask & kpi_receiver_ids.notna()
    ]
    no_receiver_mask = pass_like_mask & output_events["receiver_id"].isna() & merged["NORECEIVER"].fillna(False)
    output_events.loc[no_receiver_mask, "receiver_id"] = "out"

    needs_elastic_frame = output_events["frame_id"].isna()
    needs_elastic_receive = pass_like_mask & (
        output_events["receive_frame_id"].isna() | output_events["receiver_id"].isna()
    )

    audit: dict[str, object] = {
        "sync_source": "sportec_kpi",
        "kpi_event_matches": int(merged["EVENT_ID"].notna().sum()),
        "kpi_frame_matches": int(output_events["frame_id"].notna().sum()),
        "kpi_receive_matches": int(output_events.loc[pass_like_mask, "receive_frame_id"].notna().sum()),
        "elastic_frame_fallbacks": 0,
        "elastic_receive_fallbacks": 0,
    }

    if needs_elastic_frame.any() or needs_elastic_receive.any():
        elastic_output = run_elastic_synchronization(match_id, lineup, events, tracking, fps).set_index("action_id")

        frame_fill = needs_elastic_frame & output_events["action_id"].isin(elastic_output.index)
        output_events.loc[frame_fill, "frame_id"] = output_events.loc[frame_fill, "action_id"].map(elastic_output["frame_id"])
        audit["elastic_frame_fallbacks"] = int(frame_fill.sum())

        receive_frame_fill = needs_elastic_receive & output_events["action_id"].isin(elastic_output.index)
        output_events.loc[receive_frame_fill, "receive_frame_id"] = output_events.loc[
            receive_frame_fill, "action_id"
        ].map(elastic_output["receive_frame_id"])

        receiver_fill = pass_like_mask & output_events["receiver_id"].isna() & output_events["action_id"].isin(elastic_output.index)
        output_events.loc[receiver_fill, "receiver_id"] = output_events.loc[receiver_fill, "action_id"].map(
            elastic_output["receiver_id"]
        )
        audit["elastic_receive_fallbacks"] = int(receive_frame_fill.sum())

    finalized = finalize_synced_output(output_events, frame_table)
    audit["final_synced_rows"] = int(finalized["frame_id"].notna().sum())
    audit["final_receive_rows"] = int(finalized["receive_frame_id"].notna().sum())
    return finalized, audit


def run_event_synchronization(
    match_id: str,
    lineup: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: float,
    sync_source: str = "sportec_kpi",
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    if sync_source == "elastic":
        output_events = run_elastic_synchronization(match_id, lineup, events, tracking, fps)
        audit = {"sync_source": "elastic", "final_synced_rows": int(output_events["frame_id"].notna().sum())}
    else:
        output_events, audit = run_kpi_synchronization(match_id, lineup, events, tracking, fps)

    if return_audit:
        return output_events, audit
    return output_events


def build_timestamp_comparison(
    match_id: str,
    elastic_events: pd.DataFrame,
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    frame_table = build_tracking_frame_table(events, tracking, fps).set_index("frame_id")
    kpi = load_kpi_merged_table(match_id)

    comparison = elastic_events[["stats_perform_match_id", "original_event_id", "period_id", "frame_id"]].copy()
    comparison = comparison.rename(
        columns={
            "stats_perform_match_id": "game_id",
            "original_event_id": "event_id",
            "period_id": "period",
        }
    )
    comparison["frame_id"] = pd.to_numeric(comparison["frame_id"], errors="coerce").astype("Int64")
    comparison["time_elastic"] = comparison["frame_id"].map(frame_table["local_tracking_time"])
    comparison = comparison.merge(
        kpi[["EVENT_ID", "TRACKING_TIME"]],
        left_on="event_id",
        right_on="EVENT_ID",
        how="left",
    )
    comparison["time_dif"] = (comparison["time_elastic"] - comparison["TRACKING_TIME"]).dt.total_seconds()
    return comparison[["game_id", "event_id", "period", "time_elastic", "TRACKING_TIME", "time_dif"]].copy()


def sync_events(
    match_id: str,
    lineup: pd.DataFrame,
    events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: float,
    overwrite: bool,
    sync_source: str = "sportec_kpi",
) -> pd.DataFrame:
    output_path = EVENT_SYNCED_DIR / f"{match_id}.csv"
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path)

    output_events = run_event_synchronization(match_id, lineup, events, tracking, fps=fps, sync_source=sync_source)
    output_events.to_csv(output_path, index=False, encoding="utf-8")
    return output_events


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

    discovered_matches = discover_match_files()
    if not discovered_matches:
        raise FileNotFoundError("No Sportec XML files found in the raw data directories.")
    all_matches, discovery_skips = sort_matches_by_kickoff(discovered_matches)
    if not all_matches:
        raise RuntimeError("No Sportec matches with readable metadata were found.")
    if discovery_skips:
        print(f"Skipping {len(discovery_skips)} matches during discovery due to metadata errors.")
        for item in discovery_skips[:10]:
            print(f"  DISCOVERY SKIP {item['match_id']}: {item['error']}")
        if len(discovery_skips) > 10:
            print(f"  ... and {len(discovery_skips) - 10} more")

    processed_lineups: list[pd.DataFrame] = []
    processed_events: list[pd.DataFrame] = []
    successful_metadata: list[dict[str, object]] = []
    skipped_matches: list[dict[str, str]] = []

    selected_matches = filter_matches(all_matches, args.match_id, args.limit)
    if not selected_matches:
        raise ValueError("No matches selected for processing.")

    for index, match_files in enumerate(selected_matches, start=1):
        print(f"[{index}/{len(selected_matches)}] {match_files.match_id}")
        try:
            lineup, metadata = parse_match_information(match_files)
            raw_events = load_match_raw_events(match_files)
            finalized_lineup = finalize_lineup(lineup, raw_events, metadata)
            tracking, fps = build_tracking_outputs(match_files, finalized_lineup, raw_events, overwrite=args.overwrite)

            event_ds = build_kloppy_event_dataset(match_files)
            kloppy_events = build_kloppy_event_table(event_ds)
            actions, spadl_audit = build_spadl_actions(
                match_files,
                finalized_lineup,
                raw_events,
                metadata,
                event_ds=event_ds,
                kloppy_events=kloppy_events,
                return_audit=True,
            )
            print(
                "  SPADL:",
                f"mapped={spadl_audit['mapped_event_count']}",
                f"dropped_unmapped={spadl_audit['dropped_unmapped_events']}",
                f"dropped_negative={spadl_audit['dropped_negative_time_actions']}",
                f"repaired_fouls={spadl_audit['repaired_foul_coordinates']}",
                f"dropped_missing_xy={spadl_audit['dropped_missing_coordinates']}",
                f"dribbles={spadl_audit['auto_added_dribbles']}",
                f"final_actions={spadl_audit['final_action_count']}",
            )
            defcon_events = build_defcon_event_table(match_files.match_id, actions, finalized_lineup, raw_events, metadata)

            if not args.skip_sync:
                synced_events, sync_audit = run_event_synchronization(
                    match_files.match_id,
                    finalized_lineup,
                    defcon_events,
                    tracking,
                    fps=fps,
                    sync_source=args.sync_source,
                    return_audit=True,
                )
                synced_events.to_csv(EVENT_SYNCED_DIR / f"{match_files.match_id}.csv", index=False, encoding="utf-8")
                print(
                    "  Sync:",
                    f"source={sync_audit['sync_source']}",
                    f"synced={sync_audit.get('final_synced_rows', 0)}",
                    f"receive={sync_audit.get('final_receive_rows', 0)}",
                    f"kpi_frames={sync_audit.get('kpi_frame_matches', 0)}",
                    f"kpi_receives={sync_audit.get('kpi_receive_matches', 0)}",
                    f"elastic_frame_fallbacks={sync_audit.get('elastic_frame_fallbacks', 0)}",
                    f"elastic_receive_fallbacks={sync_audit.get('elastic_receive_fallbacks', 0)}",
                )

            processed_lineups.append(export_lineup_table(finalized_lineup))
            processed_events.append(defcon_events)
            successful_metadata.append({"match_id": match_files.match_id, "kickoff_time": metadata["kickoff_time"]})
        except Exception as exc:
            error_summary = summarize_exception(exc)
            skipped_matches.append({"match_id": match_files.match_id, "error": error_summary})
            print(f"  SKIP {match_files.match_id}: {error_summary}")
            continue

    if not processed_lineups or not processed_events:
        raise RuntimeError("Preprocessing did not produce any usable matches.")

    successful_metadata_df = pd.DataFrame(successful_metadata)
    build_split_manifest(successful_metadata_df)

    all_lineups = pd.concat(processed_lineups, ignore_index=True)
    all_lineups.to_parquet(LINEUP_PATH, index=False)

    all_events = pd.concat(processed_events, ignore_index=True)
    all_events.to_parquet(EVENT_PATH, index=False)

    print(f"Saved lineup parquet to {LINEUP_PATH}")
    print(f"Saved unsynced event parquet to {EVENT_PATH}")
    if not args.skip_sync:
        print(f"Saved synced per-match CSV files to {EVENT_SYNCED_DIR}")
    if skipped_matches:
        print(f"Skipped {len(skipped_matches)} matches during preprocessing.")
        for item in skipped_matches[:10]:
            print(f"  {item['match_id']}: {item['error']}")
        if len(skipped_matches) > 10:
            print(f"  ... and {len(skipped_matches) - 10} more")
    print("Done.")


if __name__ == "__main__":
    main()
