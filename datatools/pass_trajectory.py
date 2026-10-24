"""Synchronized official pass endpoints shared by features, labels and visuals."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from bisect import bisect_right
from itertools import islice
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

PASS_DURATION_KEY = "pass_duration_definition"
PASS_DURATION = "canonical_official_pass_end_v3"
ANGLE_ONLY_KEY = "angle_only_definition"
ANGLE_ONLY = "canonical_official_ball_direction_1s_v3"
ORIGINAL_KEY = "original_receiver_definition"
ORIGINAL = "canonical_official_distance_angle_v3"
MAX_DURATION_KEY = "max_pass_duration"
TRAJECTORY_WINDOW_SECONDS = 1.0
ENDPOINT_COLUMNS = ("observed_end_frame_id", "trajectory_end_frame_id", "pass_endpoint_source",
    "pass_endpoint_event_id", "pass_endpoint_definition", "pass_duration", "pass_endpoint_reason",
    "candidate_end_frame_id", "candidate_pass_duration")
PHYSICAL_TAGS = {"Play", "ShotAtGoal", "OtherBallAction", "BallClaiming", "TacklingGame"}
STOPPAGE_TAGS = {"Foul", "Offside", "FinalWhistle"}
OUT_TAGS = {"BallOut", "BallOutOfPlay", "OutOfPlay"}
RESTART_TAGS = {"KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "RefereeBall"}
CONTACT_TAGS = PHYSICAL_TAGS | STOPPAGE_TAGS | OUT_TAGS | RESTART_TAGS


def event_key(value):
    return "" if pd.isna(value) else str(value).removesuffix(".0")


def positive_duration(value):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("maximum pass duration must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("maximum pass duration must be positive and finite")
    return result


@lru_cache(maxsize=1)
def _match_files():
    from scripts.preprocess_sportec import discover_match_files
    return {match.match_id: match for match in discover_match_files()}


def load_official_events(match_id):
    """Join active raw events to KPI frames, including events omitted by carry controls.

    No interpolated/calculated/raw-time frames are accepted. KPI frame numbers
    are converted to normalized tracking IDs using period-aware mapping.
    """
    if not match_id or not str(match_id).startswith("DFL-MAT-"):
        return pd.DataFrame()
    from scripts.preprocess_sportec import (load_kpi_merged_table, load_match_raw_events,
                                            map_period_raw_frames_to_tracking_ids)
    from project_config import DATA_ROOT
    files = _match_files().get(str(match_id))
    if files is None or files.kpi_path is None:
        return pd.DataFrame()
    kpi = load_kpi_merged_table(files).copy()
    kpi["_key"] = kpi.EVENT_ID.map(event_key)
    kpi = kpi.drop_duplicates("_key").set_index("_key")
    raw = load_match_raw_events(files)
    periods = raw.drop_duplicates("event_id").assign(
        _key=lambda x: x.event_id.map(event_key)).set_index("_key").period_id
    from pyarrow import ArrowInvalid
    tracking_path = DATA_ROOT / "tracking" / f"{match_id}.parquet"
    try:
        mapping = pd.read_parquet(tracking_path, columns=["period_id", "raw_frame_id", "frame_id"])
    except ArrowInvalid:
        # Preprocessing leaves IDs unchanged when periods do not overlap.
        mapping = pd.read_parquet(tracking_path, columns=["period_id", "frame_id"])
        if mapping.frame_id.duplicated().any():
            raise ValueError(f"Ambiguous tracking frame mapping for {match_id}")
        mapping["raw_frame_id"] = mapping.frame_id
    events = ET.parse(files.event_path).getroot().findall(".//Event")
    deleted = {event_key(node.attrib.get("EventId", event.attrib.get("EventId"))) for event in events
               for node in event if node.tag == "Delete"}
    records = []
    for event in events:
        node = next((node for node in event if node.tag != "Qualifier"), None)
        key = event_key(event.attrib.get("EventId"))
        if node is None or node.tag not in CONTACT_TAGS or key in deleted:
            continue
        row = kpi.loc[key] if key in kpi.index else pd.Series(dtype=object)
        records.append(dict(event_id=key, raw_tag=node.tag, frame_id=row.get("FRAME_NUMBER"),
                            reception_frame_id=row.get("RECFRM"), frame_source="kpi",
                            period_id=periods.get(key), reception_source=row.get("endpoint_source", "kpi_csv")))
    result = pd.DataFrame(records)
    if result.empty:
        return result
    for column in ("frame_id", "reception_frame_id"):
        result[column] = map_period_raw_frames_to_tracking_ids(mapping, result.period_id, result[column])
    result.loc[~result.reception_source.isin(["reception", "kpi_csv"]), "reception_frame_id"] = pd.NA
    return result


@dataclass(frozen=True)
class PassEndpoint:
    observed: int | None = None
    trajectory: int | None = None
    source: str = "unavailable"
    event_id: str = ""
    duration: float = np.nan
    reason: str = "missing_official_endpoint"
    candidate: int | None = None
    candidate_duration: float = np.nan


class PassEndpointResolver:
    def __init__(self, tracking, fps=25, events=None, max_pass_duration=None):
        self.tracking, self.fps = tracking, fps
        self.max_pass_duration = None if max_pass_duration is None else positive_duration(max_pass_duration)
        self.period = tracking.get("period_id", pd.Series(0, index=tracking.index))
        self.period_ends = {p: int(rows.index.max()) for p, rows in tracking.groupby(self.period)}
        self.episode_ends = {}
        if {"episode_id", "ball_state"}.issubset(tracking):
            live = tracking.loc[tracking.ball_state.eq("alive")]
            self.episode_ends = {(p, e): int(rows.index.max())
                for (p, e), rows in live.groupby([self.period.loc[live.index], "episode_id"])}
        self.events = (events if events is not None else pd.DataFrame()).copy()
        self.receptions, self.ordered, self.event_frames, self.restarts = {}, {}, {}, {}
        self.following = {}
        self.sequence = []
        if self.events.empty:
            return
        key_column = "event_id" if "event_id" in self.events else "original_event_id"
        self.events["_key"] = self.events[key_column].map(event_key)
        if "frame_source" in self.events:
            self.events = self.events.loc[self.events.frame_source.eq("kpi")].copy()
        if "reception_frame_id" in self.events:
            self.receptions = dict(zip(self.events._key, self.events.reception_frame_id))
        self.events["frame_id"] = pd.to_numeric(self.events.get("frame_id"), errors="coerce")
        sequence = self.sequence = self.events.to_dict("records")
        for position, row in enumerate(sequence):
            self.following[row["_key"]] = position + 1
        self.events = self.events.loc[self.events.frame_id.isin(tracking.index)].copy()
        self.events = self.events.loc[self.events.raw_tag.isin(CONTACT_TAGS)]
        self.events = self.events.drop_duplicates("_key").sort_values(["frame_id", "_key"])
        self.events["_period"] = self.events.frame_id.map(self.period)
        for period, rows in self.events.groupby("_period"):
            entries = list(rows[["frame_id", "raw_tag", "_key"]].itertuples(index=False, name=None))
            self.ordered[period] = entries
            self.event_frames[period] = [int(f) for f, _, _ in entries]
            self.restarts[period] = [(int(f), tag) for f, tag, _ in entries if tag in RESTART_TAGS]

    def resolve(self, action):
        if isinstance(action.get("pass_endpoint_definition"), str) and action.get("pass_endpoint_definition") == PASS_DURATION:
            stored_limit = action.get(MAX_DURATION_KEY)
            same_limit = (self.max_pass_duration is None if pd.isna(stored_limit) else stored_limit == self.max_pass_duration)
            if same_limit:
                end = action.get("receive_frame_id")
                if pd.isna(end):
                    return PassEndpoint(source=action.get("pass_endpoint_source", "unavailable"),
                        event_id=action.get("pass_endpoint_event_id", ""),
                        reason=action.get("pass_endpoint_reason") or "missing_official_endpoint",
                        candidate=action.get("candidate_end_frame_id"),
                        candidate_duration=action.get("candidate_pass_duration", np.nan))
                return PassEndpoint(int(end), int(action["trajectory_end_frame_id"]),
                    action["pass_endpoint_source"], action.get("pass_endpoint_event_id", ""),
                    float(action["pass_duration"]), "", int(end), float(action["pass_duration"]))
        frame = action.get("frame_id")
        if pd.isna(frame) or frame not in self.tracking.index:
            return PassEndpoint(reason="missing_launch_tracking")
        frame = int(frame)
        snapshot = self.tracking.loc[frame]
        period = self.period.at[frame]
        if pd.notna(action.get("period_id")) and action.get("period_id") != period:
            return PassEndpoint(reason="launch_period_mismatch")
        if "ball_state" in snapshot and pd.isna(snapshot.ball_state):
            return PassEndpoint(reason="missing_launch_state")
        if "ball_state" in snapshot and snapshot.ball_state != "alive":
            return PassEndpoint(reason="dead_launch")
        boundary = self.episode_ends.get((period, snapshot.get("episode_id")), self.period_ends[period])
        position = self.tracking.index.searchsorted(boundary, side="right")
        stop_boundary = boundary
        if position < len(self.tracking):
            following = self.tracking.index[position]
            if self.period.at[following] == period:
                stop_boundary = int(following)
        key = event_key(action.get("original_event_id"))
        candidates = []

        def valid(end, limit=boundary):
            return (pd.notna(end) and end in self.tracking.index and frame < end <= limit
                    and self.period.at[end] == period)

        reception = self.receptions.get(key)
        # Explicit official provenance also supports non-Sportec/injected actions.
        if pd.isna(reception) and action.get("endpoint_source") in {"kpi_csv", "reception", "official_reception"}:
            reception = action.get("receive_frame_id")
        if valid(reception):
            candidates.append((int(reception), 0, "official_reception", key))
        entries = self.ordered.get(period, ())
        begin = bisect_right(self.event_frames.get(period, ()), frame)
        restart_entry = next(((f, tag) for f, tag in self.restarts.get(period, ()) if f > frame), None)
        restart = restart_entry[0] if restart_entry else None
        if restart is not None:
            candidates = [c for c in candidates if c[0] < restart]
        for end, tag, eid in islice(entries, begin, None):
            if restart is not None and end >= restart:
                break
            limit = stop_boundary if tag in STOPPAGE_TAGS | OUT_TAGS else boundary
            if valid(end, limit):
                source = "official_stoppage" if tag in STOPPAGE_TAGS else "official_out" if tag in OUT_TAGS else "official_contact"
                candidates.append((end, 1, source, eid))
                break
        out_restart = restart_entry[1] if restart_entry else None
        known_out = (isinstance(action.get("receiver_id"), str) and action.get("receiver_id") == "out" and
            (out_restart in {"ThrowIn", "GoalKick", "CornerKick"}
             or action.get("next_type") in {"throw_in", "goalkick", "goal_kick", "corner_short", "corner_crossed"}))
        if known_out and boundary > frame and (restart is None or boundary < restart):
            candidates.append((boundary, 2, "known_out_boundary", ""))
        if not candidates:
            return PassEndpoint()
        end, _, source, event_id = min(candidates)
        for following in islice(self.sequence, self.following.get(key, len(self.sequence)), None):
            next_period = following.get("period_id")
            if pd.notna(next_period) and next_period != period:
                break
            next_frame = following.get("frame_id")
            if pd.isna(next_frame):
                return PassEndpoint(reason="missing_intervening_official_frame")
            if next_frame <= frame:
                continue
            if next_frame >= end:
                break
        duration = float((end - frame) / self.fps)
        if self.max_pass_duration is not None and duration > self.max_pass_duration:
            return PassEndpoint(source=source, event_id=event_id, reason="duration_limit_exceeded",
                                candidate=end, candidate_duration=duration)
        capped = min(end, frame + max(int(round(self.fps * TRAJECTORY_WINDOW_SECONDS)), 1))
        return PassEndpoint(end, capped, source, event_id, duration, "", end, duration)

    def annotate(self, actions):
        result = actions.copy()
        for column in ("receive_frame_id", "endpoint_source"):
            audit = "original_" + column
            if audit not in result:
                result[audit] = result.get(column)
        for column in ENDPOINT_COLUMNS:
            result[column] = np.nan if column.endswith("frame_id") or column.endswith("duration") else ""
        mask = result.get("action_type", pd.Series("", index=result.index)).eq("pass")
        mask |= result.get("spadl_type", pd.Series("", index=result.index)).isin(["pass", "cross"])
        for index, action in actions.loc[mask].iterrows():
            end = self.resolve(action)
            result.loc[index, list(ENDPOINT_COLUMNS)] = [end.observed, end.trajectory, end.source,
                end.event_id, PASS_DURATION, end.duration, end.reason, end.candidate, end.candidate_duration]
            result.loc[index, "receive_frame_id"] = end.observed if end.observed is not None else np.nan
            result.loc[index, "endpoint_source"] = end.source
            for axis in ("x", "y"):
                column = "ball_" + axis
                result.loc[index, "end_" + axis] = (self.tracking.at[end.observed, column]
                    if end.observed is not None and column in self.tracking else np.nan)
            if "receive_ts" in result:
                from sync.utils import seconds_to_timestamp
                timestamp = self.tracking.at[end.observed, "timestamp"] if end.observed is not None and "timestamp" in self.tracking else np.nan
                result.loc[index, "receive_ts"] = seconds_to_timestamp(timestamp) if pd.notna(timestamp) else None
            if "duration" in result:
                result.loc[index, "duration"] = end.duration
        result[MAX_DURATION_KEY] = self.max_pass_duration
        return result


def ensure_pass_endpoints(match, *, annotate=True):
    maximum = getattr(match, MAX_DURATION_KEY, None)
    definitions = match.actions.get("pass_endpoint_definition")
    limits = match.actions.get(MAX_DURATION_KEY)
    current_limit = limits is not None and (limits.isna().all() if maximum is None else limits.eq(maximum).all())
    pinned = definitions is not None and definitions.fillna("").isin(["", PASS_DURATION]).all() and current_limit
    resolver = getattr(match, "_pass_endpoint_resolver", None)
    if resolver is None or resolver.max_pass_duration != maximum or (not pinned and getattr(resolver, "pinned_only", False)):
        events = pd.DataFrame() if pinned else getattr(match, "official_pass_events", None)
        if events is None:
            events = load_official_events(getattr(match, "match_id", None))
        resolver = match._pass_endpoint_resolver = PassEndpointResolver(match.tracking, match.fps, events, maximum)
        resolver.pinned_only = pinned
    if annotate and not pinned:
        match.actions = resolver.annotate(match.actions)
    return resolver


def trajectory_samples(tracking, action):
    """Require finite segment endpoints; allow missing interior samples."""
    start, end = action.get("frame_id"), action.get("trajectory_end_frame_id")
    if pd.isna(start) or pd.isna(end) or end <= start or start not in tracking.index or end not in tracking.index:
        return None
    if not {"ball_x", "ball_y"}.issubset(tracking):
        return None
    segment = tracking.loc[int(start):int(end)].copy()
    if "ball_z" not in segment:
        segment["ball_z"] = 0.0
    finite = np.isfinite(segment[["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)).all(axis=1)
    if not finite[0] or not finite[-1]:
        return None
    segment = segment.loc[finite]
    return segment if len(segment) >= 2 else None


def prepare_pass_events(events, tracking, match_id, maximum=None):
    """Prepare all source events, including those absent from training samples."""
    table = tracking.set_index("frame_id") if "frame_id" in tracking.columns else tracking
    return PassEndpointResolver(table, 25, load_official_events(match_id), maximum).annotate(events)


def apply_pinned_endpoints(actions, root, match_id):
    """Overlay endpoint data by event identity, preserving action indices and membership."""
    from pathlib import Path
    if root is None or match_id is None or "original_event_id" not in actions:
        return actions.copy()
    path = Path(root) / "pass_endpoints" / f"{match_id}.parquet"
    if not path.exists():
        return actions.copy()
    pinned = pd.read_parquet(path)
    pinned = pinned.loc[pinned.pass_endpoint_definition.eq(PASS_DURATION)].copy()
    pinned["_key"] = pinned.original_event_id.map(event_key)
    pinned = pinned.drop_duplicates("_key").set_index("_key")
    result = actions.copy()
    keys = result.original_event_id.map(event_key)
    mask = keys.isin(pinned.index)
    for column in (*ENDPOINT_COLUMNS, MAX_DURATION_KEY, "receive_frame_id", "receive_ts", "endpoint_source", "end_x", "end_y"):
        if column in pinned:
            if column not in result:
                result[column] = np.nan if pd.api.types.is_numeric_dtype(pinned[column]) else None
            result.loc[mask, column] = keys[mask].map(pinned[column]).to_numpy()
    return result
