from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CARRY_DEFINITION_VERSION = "sportec_fernandez_1s_v1"
RESTART_TAGS = {"KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "RefereeBall"}


def _utc_naive(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _player_map(lineup: pd.DataFrame) -> dict[str, str]:
    if "object_id" not in lineup.columns:
        return {}
    return (
        lineup.dropna(subset=["player_id", "object_id"])
        .drop_duplicates("player_id")
        .set_index("player_id")["object_id"]
        .astype(str)
        .to_dict()
    )


def _event_periods(records: list[dict[str, Any]]) -> None:
    boundaries: dict[int, dict[str, pd.Timestamp]] = {1: {}, 2: {}}
    for record in records:
        section = record.get("game_section")
        if section not in {"firstHalf", "secondHalf"}:
            continue
        period = 1 if section == "firstHalf" else 2
        if record["raw_tag"] == "KickOff":
            boundaries[period]["start"] = record["utc_timestamp"]
        elif record["raw_tag"] == "FinalWhistle":
            boundaries[period]["end"] = record["utc_timestamp"]

    for record in records:
        timestamp = record["utc_timestamp"]
        period = 0
        for candidate in (1, 2):
            start = boundaries[candidate].get("start")
            end = boundaries[candidate].get("end")
            if start is not None and timestamp >= start and (end is None or timestamp <= end):
                period = candidate
                break
        record["period_id"] = period


def _parse_raw_control_records(event_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(event_path).getroot()
    records: list[dict[str, Any]] = []
    for event in root.findall(".//Event"):
        child = next((node for node in event if node.tag != "Qualifier"), None)
        if child is None or child.tag == "Delete":
            continue
        tag = child.tag
        nested = child
        if tag in RESTART_TAGS:
            nested = child.find("Play")
            if nested is None:
                nested = child.find("ShotAtGoal")
            if nested is None:
                nested = child

        record: dict[str, Any] = {
            "event_id": str(event.attrib.get("EventId", "")),
            "utc_timestamp": _utc_naive(event.attrib["EventTime"]),
            "calculated_timestamp": (
                _utc_naive(event.attrib["CalculatedTimestamp"])
                if event.attrib.get("CalculatedTimestamp")
                else pd.NaT
            ),
            "calculated_frame": pd.to_numeric(event.attrib.get("CalculatedFrame"), errors="coerce"),
            "raw_tag": tag,
            "game_section": child.attrib.get("GameSection"),
            "event_kind": "administrative",
            "player_id": None,
            "team_id": None,
            "receiver_player_id": None,
            "winner_player_id": None,
            "winner_team_id": None,
            "loser_player_id": None,
            "loser_team_id": None,
            "winner_role": None,
            "loser_role": None,
            "winner_result": None,
            "possession_change": False,
            "fouler_player_id": None,
            "fouled_player_id": None,
            "clearance": False,
            "claim_type": None,
        }

        if tag in RESTART_TAGS:
            record["event_kind"] = "restart"
            record["team_id"] = child.attrib.get("Team") or nested.attrib.get("Team")
            record["player_id"] = nested.attrib.get("Player")
        elif tag == "FinalWhistle":
            record["event_kind"] = "period_end"
        elif tag == "Play":
            record["event_kind"] = "cross" if child.find("Cross") is not None else "pass"
            record["team_id"] = child.attrib.get("Team")
            record["player_id"] = child.attrib.get("Player")
            record["receiver_player_id"] = child.attrib.get("Recipient")
        elif tag == "ShotAtGoal":
            record["event_kind"] = "shot"
            record["team_id"] = child.attrib.get("Team")
            record["player_id"] = child.attrib.get("Player")
        elif tag == "TacklingGame":
            record["event_kind"] = "tackle"
            record["winner_player_id"] = child.attrib.get("Winner")
            record["winner_team_id"] = child.attrib.get("WinnerTeam")
            record["loser_player_id"] = child.attrib.get("Loser")
            record["loser_team_id"] = child.attrib.get("LoserTeam")
            record["winner_role"] = child.attrib.get("WinnerRole")
            record["loser_role"] = child.attrib.get("LoserRole")
            record["winner_result"] = child.attrib.get("WinnerResult")
            record["possession_change"] = child.attrib.get("PossessionChange", "").lower() == "true"
        elif tag == "BallClaiming":
            record["event_kind"] = "claim"
            record["team_id"] = child.attrib.get("Team")
            record["player_id"] = child.attrib.get("Player")
            record["claim_type"] = child.attrib.get("Type")
        elif tag == "OtherBallAction":
            record["event_kind"] = "other"
            record["team_id"] = child.attrib.get("Team")
            record["player_id"] = child.attrib.get("Player")
            record["clearance"] = child.attrib.get("DefensiveClearance", "").lower() == "true"
        elif tag == "Foul":
            record["event_kind"] = "foul"
            record["fouler_player_id"] = child.attrib.get("Fouler")
            record["fouled_player_id"] = child.attrib.get("Fouled")
            record["team_id"] = child.attrib.get("TeamFouler")
        else:
            continue
        records.append(record)

    _event_periods(records)
    return records


def _map_frames(
    control: pd.DataFrame,
    frame_table: pd.DataFrame,
    kpi: pd.DataFrame | None,
    canonical_events: pd.DataFrame | None = None,
    fps: float = 25.0,
) -> pd.DataFrame:
    control = control.copy()
    frame_table = frame_table.copy()
    frame_table["utc_timestamp"] = pd.to_datetime(frame_table["utc_timestamp"], errors="coerce")
    raw_lookup = (
        frame_table.dropna(subset=["period_id", "raw_frame_id"])
        .drop_duplicates(["period_id", "raw_frame_id"])
        .set_index(["period_id", "raw_frame_id"])["frame_id"]
        .to_dict()
    )
    kpi_frames: dict[str, object] = {}
    if kpi is not None and not kpi.empty:
        kpi_frames = kpi.drop_duplicates("EVENT_ID").set_index("EVENT_ID")["FRAME_NUMBER"].to_dict()

    canonical_frames: dict[str, object] = {}
    if canonical_events is not None and not canonical_events.empty:
        canonical = canonical_events.dropna(subset=["original_event_id", "frame_id"]).copy()
        canonical["_event_id_key"] = canonical["original_event_id"].astype(str)
        canonical_frames = (
            canonical.drop_duplicates("_event_id_key")
            .set_index("_event_id_key")["frame_id"]
            .to_dict()
        )

    period_frames = {
        int(period): np.sort(pd.to_numeric(rows["frame_id"], errors="coerce").dropna().astype(int).unique())
        for period, rows in frame_table.groupby("period_id")
        if pd.notna(period)
    }

    def nearest_tracking_frame(period: int, candidate: float) -> int | None:
        candidates = period_frames.get(period)
        if candidates is None or candidates.size == 0 or not np.isfinite(candidate):
            return None
        insertion = int(np.searchsorted(candidates, candidate))
        eligible = candidates[max(0, insertion - 1) : min(candidates.size, insertion + 1)]
        return int(eligible[np.argmin(np.abs(eligible - candidate))])

    mapped: list[object] = []
    sources: list[str] = []
    errors: list[float] = []
    for row in control.itertuples(index=False):
        period = int(row.period_id) if pd.notna(row.period_id) else 0
        frame = canonical_frames.get(str(row.event_id))
        source = "canonical"
        if pd.isna(frame):
            raw_frame = kpi_frames.get(str(row.event_id))
            frame = raw_lookup.get((period, int(raw_frame))) if period and pd.notna(raw_frame) else None
            source = "kpi"
        mapped.append(pd.NA if pd.isna(frame) else int(frame))
        sources.append("unmapped" if pd.isna(frame) else source)
        errors.append(np.nan)

    anchor_rows: dict[int, pd.DataFrame] = {}
    mapped_series = pd.Series(mapped, index=control.index, dtype="Int64")
    for period, rows in control.loc[mapped_series.notna()].groupby("period_id"):
        anchors = rows.assign(_frame=mapped_series.loc[rows.index]).dropna(subset=["utc_timestamp", "_frame"])
        if not anchors.empty:
            anchor_rows[int(period)] = anchors.sort_values("utc_timestamp")

    for position, row in enumerate(control.itertuples(index=False)):
        if pd.notna(mapped[position]):
            continue
        period = int(row.period_id) if pd.notna(row.period_id) else 0
        frame = None
        source = "unmapped"
        error = np.nan
        anchors = anchor_rows.get(period)
        if period and anchors is not None and pd.notna(row.utc_timestamp):
            deltas = (anchors["utc_timestamp"] - row.utc_timestamp).abs().dt.total_seconds()
            anchor_index = deltas.idxmin()
            anchor = anchors.loc[anchor_index]
            estimated_frame = int(anchor["_frame"]) + round(
                (row.utc_timestamp - anchor["utc_timestamp"]).total_seconds() * float(fps)
            )
            frame = nearest_tracking_frame(period, estimated_frame)
            if frame is not None:
                source = "event_time_interpolation"
                error = float(deltas.at[anchor_index])
        if frame is None and period and pd.notna(row.calculated_frame):
            frame = raw_lookup.get((period, int(row.calculated_frame)))
            if frame is not None:
                source = "calculated_frame"
        if frame is None and period:
            timestamp = row.utc_timestamp
            candidates = frame_table.loc[frame_table["period_id"] == period]
            if pd.notna(timestamp) and not candidates.empty:
                delta = (candidates["utc_timestamp"] - timestamp).abs().dt.total_seconds()
                nearest = delta.idxmin()
                error = float(delta.at[nearest])
                if error <= 2.0:
                    frame = int(candidates.at[nearest, "frame_id"])
                    source = "timestamp"
        mapped[position] = pd.NA if frame is None else int(frame)
        sources[position] = "unmapped" if frame is None else source
        errors[position] = error
    control["frame_id"] = pd.Series(mapped, dtype="Int64")
    control["frame_source"] = sources
    control["frame_error_seconds"] = errors
    return control


def build_synchronized_control_events(
    event_path: Path,
    lineup: pd.DataFrame,
    frame_table: pd.DataFrame,
    kpi: pd.DataFrame | None = None,
    canonical_events: pd.DataFrame | None = None,
    fps: float = 25.0,
) -> pd.DataFrame:
    records = _parse_raw_control_records(Path(event_path))
    control = pd.DataFrame.from_records(records)
    if control.empty:
        return control
    player_map = _player_map(lineup)
    for source, target in (
        ("player_id", "object_id"),
        ("receiver_player_id", "receiver_id"),
        ("winner_player_id", "winner_id"),
        ("loser_player_id", "loser_id"),
        ("fouler_player_id", "fouler_id"),
        ("fouled_player_id", "fouled_id"),
    ):
        control[target] = control[source].map(player_map)
    control = _map_frames(
        control,
        frame_table,
        kpi,
        canonical_events=canonical_events,
        fps=fps,
    )
    return control.sort_values(["period_id", "frame_id", "utc_timestamp", "event_id"], na_position="last").reset_index(drop=True)


def _team(object_id: object) -> str | None:
    value = str(object_id)
    if value.startswith("home_"):
        return "home"
    if value.startswith("away_"):
        return "away"
    return None


def _canonical_return_index(canonical: pd.DataFrame, period: int, terminal_frame: int, event_id: str) -> int | None:
    exact = canonical.index[canonical["original_event_id"].astype(str) == str(event_id)].tolist()
    if exact:
        return int(exact[0])
    candidates = canonical.loc[
        (pd.to_numeric(canonical["period_id"], errors="coerce") == period)
        & (pd.to_numeric(canonical["frame_id"], errors="coerce") >= terminal_frame)
    ]
    return int(candidates.index[0]) if not candidates.empty else None


def derive_carry_segments(
    control_events: pd.DataFrame,
    canonical_events: pd.DataFrame,
    tracking: pd.DataFrame,
    fps: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-second Fernandez-like carry segments and a spell-level audit table."""
    canonical = canonical_events.copy().reset_index(drop=True)
    canonical["frame_id"] = pd.to_numeric(canonical["frame_id"], errors="coerce").astype("Int64")
    timeline = control_events.loc[
        control_events["period_id"].isin([1, 2]) & control_events["frame_id"].notna()
    ].copy()
    timeline["timeline_kind"] = timeline["event_kind"]

    receptions: list[dict[str, Any]] = []
    pass_like = canonical["spadl_type"].isin(["pass", "cross"])
    for index, action in canonical.loc[pass_like].iterrows():
        receiver = action.get("receiver_id")
        if (
            pd.notna(action.get("receive_frame_id"))
            and _team(receiver) is not None
            and _team(receiver) == _team(action.get("object_id"))
            and not bool(action.get("offside", False))
        ):
            receptions.append(
                {
                    "event_id": f"receipt:{action.get('original_event_id')}:{index}",
                    "period_id": int(action["period_id"]),
                    "frame_id": int(action["receive_frame_id"]),
                    "utc_timestamp": pd.NaT,
                    "event_kind": "receipt",
                    "timeline_kind": "receipt",
                    "object_id": receiver,
                    "receiver_id": None,
                    "winner_id": None,
                    "loser_id": None,
                    "fouler_id": None,
                    "fouled_id": None,
                    "winner_role": None,
                    "loser_role": None,
                    "winner_result": None,
                    "possession_change": False,
                    "clearance": False,
                    "claim_type": None,
                }
            )
    if receptions:
        timeline = pd.concat([timeline, pd.DataFrame.from_records(receptions)], ignore_index=True, sort=False)
    priority = {"receipt": -1, "pass": 0, "cross": 0, "shot": 0, "foul": 0, "tackle": 0, "claim": 0, "other": 0, "restart": 0, "period_end": 0}
    timeline["_priority"] = timeline["timeline_kind"].map(priority).fillna(0)
    timeline = timeline.sort_values(["period_id", "frame_id", "_priority", "utc_timestamp", "event_id"], na_position="last")

    tracking_index = set(pd.to_numeric(pd.Index(tracking.index), errors="coerce").dropna().astype(int))
    spells: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    spell_counter = 0

    def start(row: pd.Series, carrier: object, source_kind: str) -> None:
        nonlocal current, spell_counter
        if _team(carrier) is None:
            current = None
            return
        spell_counter += 1
        current = {
            "carry_id": spell_counter,
            "period_id": int(row["period_id"]),
            "carrier_id": str(carrier),
            "start_frame": int(row["frame_id"]),
            "start_event_id": str(row["event_id"]),
            "start_kind": source_kind,
        }

    def finish(row: pd.Series, success: bool | None, reason: str, receiver: object = None) -> None:
        nonlocal current
        if current is None:
            return
        terminal_frame = int(row["frame_id"])
        duration_frames = terminal_frame - int(current["start_frame"])
        return_index = _canonical_return_index(
            canonical,
            int(current["period_id"]),
            terminal_frame,
            str(row["event_id"]),
        )
        retained = success is not None and duration_frames >= fps and return_index is not None
        audit = {
            **current,
            "terminal_frame": terminal_frame,
            "terminal_event_id": str(row["event_id"]),
            "terminal_kind": str(row["timeline_kind"]),
            "success": success,
            "duration_seconds": duration_frames / float(fps),
            "return_event_index": return_index,
            "retained": retained,
            "reason": reason if retained else (reason if success is None else "short_or_missing_return"),
        }
        spells.append(audit)
        if retained:
            count = duration_frames // fps
            for segment_id in range(int(count)):
                start_frame = int(current["start_frame"]) + segment_id * fps
                end_frame = terminal_frame if segment_id == count - 1 else start_frame + fps
                if start_frame not in tracking_index or end_frame not in tracking_index:
                    continue
                segment_success = bool(success) if segment_id == count - 1 else True
                segments.append(
                    {
                        "carry_definition": CARRY_DEFINITION_VERSION,
                        "carry_id": int(current["carry_id"]),
                        "segment_id": segment_id,
                        "period_id": int(current["period_id"]),
                        "frame_id": start_frame,
                        "receive_frame_id": end_frame,
                        "object_id": current["carrier_id"],
                        "receiver_id": str(receiver) if _team(receiver) is not None else current["carrier_id"],
                        "spadl_type": "ball_carry",
                        "action_type": "dribble",
                        "success": segment_success,
                        "duration": (end_frame - start_frame) / float(fps),
                        "start_x": tracking.at[start_frame, "ball_x"],
                        "start_y": tracking.at[start_frame, "ball_y"],
                        "end_x": tracking.at[end_frame, "ball_x"],
                        "end_y": tracking.at[end_frame, "ball_y"],
                        "start_event_id": current["start_event_id"],
                        "terminal_event_id": str(row["event_id"]),
                        "terminal_kind": str(row["timeline_kind"]),
                        "return_event_index": int(return_index),
                    }
                )
        current = None

    for _, row in timeline.iterrows():
        kind = str(row["timeline_kind"])
        actor = row.get("object_id")
        carrier = current["carrier_id"] if current is not None else None
        same_actor = current is not None and actor == carrier

        if kind == "receipt":
            if current is not None and actor == carrier and current["start_kind"] == "receipt":
                continue
            if current is not None:
                finish(row, None, "unexpected_receipt")
            start(row, actor, kind)
        elif kind in {"restart", "period_end"}:
            finish(row, None, f"direct_{kind}")
        elif kind in {"pass", "cross", "shot"}:
            if same_actor:
                finish(row, True, f"carrier_{kind}", receiver=actor)
            elif current is not None:
                finish(row, None, f"unexpected_opponent_{kind}")
        elif kind == "foul":
            if current is None:
                continue
            if row.get("fouler_id") == carrier:
                finish(row, False, "carrier_committed_foul", receiver=row.get("fouled_id"))
            elif row.get("fouled_id") == carrier:
                finish(row, True, "carrier_fouled", receiver=carrier)
            else:
                finish(row, None, "foul_without_carrier")
        elif kind == "tackle":
            winner = row.get("winner_id")
            loser = row.get("loser_id")
            changed = bool(row.get("possession_change", False))
            if current is None:
                if changed:
                    start(row, winner, "tackle_win")
                elif row.get("winner_role") == "withBallControl":
                    start(row, winner, "retained_tackle")
                elif row.get("loser_role") == "withBallControl":
                    start(row, loser, "retained_tackle")
                continue
            retained = (
                (winner == carrier and row.get("winner_role") == "withBallControl" and not changed)
                or (loser == carrier and row.get("loser_role") == "withBallControl" and not changed)
            )
            if retained:
                continue
            if changed and loser == carrier and _team(winner) not in {None, _team(carrier)}:
                finish(row, False, "possession_changing_tackle", receiver=winner)
                start(row, winner, "tackle_win")
            else:
                finish(row, None, "ambiguous_tackle")
                if changed:
                    start(row, winner, "tackle_win")
        elif kind == "claim":
            if _team(actor) is None:
                finish(row, None, "claim_without_player")
                continue
            if current is not None and actor != carrier:
                if _team(actor) != _team(carrier):
                    finish(row, False, "opponent_ball_claim", receiver=actor)
                else:
                    finish(row, None, "teammate_takeover")
            if current is None:
                start(row, actor, "ball_claim")
        elif kind == "other":
            if bool(row.get("clearance", False)):
                if same_actor:
                    finish(row, None, "carrier_clearance")
                elif current is not None and _team(actor) not in {None, _team(carrier)}:
                    finish(row, False, "opponent_clearance", receiver=actor)
                else:
                    finish(row, None, "ambiguous_clearance")
                continue
            if current is None:
                start(row, actor, "other_ball_action")
            elif actor != carrier:
                if _team(actor) != _team(carrier) and _team(actor) is not None:
                    finish(row, False, "corroborated_other_ball_loss", receiver=actor)
                else:
                    finish(row, None, "teammate_takeover")
                start(row, actor, "other_ball_action")

    if current is not None:
        dummy = pd.Series({"frame_id": current["start_frame"], "event_id": "end_of_data", "timeline_kind": "end_of_data"})
        finish(dummy, None, "end_of_data")

    segment_columns = [
        "action_key", "carry_definition", "carry_id", "segment_id", "period_id", "frame_id",
        "receive_frame_id", "object_id", "receiver_id", "spadl_type", "action_type", "success",
        "duration", "start_x", "start_y", "end_x", "end_y", "start_event_id",
        "terminal_event_id", "terminal_kind", "return_event_index",
    ]
    segment_table = pd.DataFrame.from_records(segments)
    if not segment_table.empty:
        segment_table.insert(0, "action_key", segment_table.apply(lambda row: f"carry:{int(row.carry_id)}:{int(row.segment_id)}", axis=1))
        segment_table = segment_table[segment_columns]
    else:
        segment_table = pd.DataFrame(columns=segment_columns)
    audit_columns = [
        "carry_id", "period_id", "carrier_id", "start_frame", "start_event_id", "start_kind",
        "terminal_frame", "terminal_event_id", "terminal_kind", "success", "duration_seconds",
        "return_event_index", "retained", "reason",
    ]
    audit_table = pd.DataFrame.from_records(spells)
    if audit_table.empty:
        audit_table = pd.DataFrame(columns=audit_columns)
    return segment_table, audit_table


def augment_match_actions_with_carries(match: Any, carries: pd.DataFrame) -> None:
    if carries.empty:
        return
    base_actions = match.actions.loc[match.actions.get("action_type", "") != "dribble"].copy()
    next_index = int(max(match.events.index.max(), base_actions.index.max())) + 1
    player_lookup = (
        match.lineup.drop_duplicates("object_id").set_index("object_id")["player_id"].to_dict()
        if "player_id" in match.lineup.columns
        else {}
    )
    action_metadata: dict[str, dict[str, Any]] = {}
    if "object_id" in base_actions.columns:
        metadata_columns = [
            column
            for column in ("player_name", "advanced_position", "team_id")
            if column in base_actions.columns
        ]
        if metadata_columns:
            action_metadata = (
                base_actions.dropna(subset=["object_id"])
                .drop_duplicates("object_id")
                .set_index("object_id")[metadata_columns]
                .to_dict(orient="index")
            )
    match_id = str(match.match_id) if match.match_id is not None else None
    records: list[dict[str, Any]] = []
    indices: list[int] = []
    for offset, carry in carries.reset_index(drop=True).iterrows():
        index = next_index + offset
        record = {column: np.nan for column in base_actions.columns}
        record.update(
            {
                "stats_perform_match_id": match_id,
                "game_id": match_id,
                "action_id": index,
                "original_event_id": carry["action_key"],
                "period_id": int(carry["period_id"]),
                "seconds": float(carry["frame_id"]) / float(match.fps),
                "frame_id": int(carry["frame_id"]),
                "receive_frame_id": int(carry["receive_frame_id"]),
                "player_id": player_lookup.get(carry["object_id"]),
                "object_id": carry["object_id"],
                "receiver_id": carry["receiver_id"],
                "next_player_id": carry["receiver_id"],
                "spadl_type": "ball_carry",
                "next_type": carry["terminal_kind"],
                "action_type": "dribble",
                "success": bool(carry["success"]),
                "offside": False,
                "expected_goal": 0.0,
                "start_x": carry["start_x"],
                "start_y": carry["start_y"],
                "end_x": carry["end_x"],
                "end_y": carry["end_y"],
                "blocked": False,
                "anomaly": False,
                "woodwork": False,
                "intent_id": carry["object_id"],
                "return_event_index": int(carry["return_event_index"]),
                "carry_definition": carry["carry_definition"],
                **action_metadata.get(carry["object_id"], {}),
            }
        )
        records.append(record)
        indices.append(index)
    carry_actions = pd.DataFrame.from_records(records, index=indices)
    carry_actions = carry_actions.dropna(axis=1, how="all")
    match.actions = pd.concat([base_actions, carry_actions], axis=0, sort=False).sort_index()
