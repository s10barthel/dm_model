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
PASS_DURATION = "synchronized_official_pass_end_v2"
ANGLE_ONLY_KEY = "angle_only_definition"
ANGLE_ONLY = "synchronized_official_ball_direction_1s_v2"
ORIGINAL_KEY = "original_receiver_definition"
ORIGINAL = "synchronized_official_distance_angle_v2"
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
    are raw tracking frame IDs retained by processed tracking.
    """
    if not match_id or not str(match_id).startswith("DFL-MAT-"):
        return pd.DataFrame()
    from scripts.preprocess_sportec import load_kpi_merged_table
    files = _match_files().get(str(match_id))
    if files is None or files.kpi_path is None:
        return pd.DataFrame()
    kpi = load_kpi_merged_table(files).copy()
    kpi["_key"] = kpi.EVENT_ID.map(event_key)
    kpi = kpi.drop_duplicates("_key").set_index("_key")
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
                            reception_frame_id=row.get("RECFRM"), frame_source="kpi"))
    return pd.DataFrame(records)


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
        if self.events.empty:
            return
        key_column = "event_id" if "event_id" in self.events else "original_event_id"
        self.events["_key"] = self.events[key_column].map(event_key)
        if "frame_source" in self.events:
            self.events = self.events.loc[self.events.frame_source.eq("kpi")].copy()
        if "reception_frame_id" in self.events:
            self.receptions = dict(zip(self.events._key, self.events.reception_frame_id))
        self.events["frame_id"] = pd.to_numeric(self.events.get("frame_id"), errors="coerce")
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
        if pd.isna(reception) and action.get("endpoint_source") in {"kpi_csv", "reception", "kpi_xml"}:
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
        duration = float((end - frame) / self.fps)
        if self.max_pass_duration is not None and duration > self.max_pass_duration:
            return PassEndpoint(source=source, event_id=event_id, reason="duration_limit_exceeded",
                                candidate=end, candidate_duration=duration)
        capped = min(end, frame + max(int(round(self.fps * TRAJECTORY_WINDOW_SECONDS)), 1))
        return PassEndpoint(end, capped, source, event_id, duration, "", end, duration)

    def annotate(self, actions):
        result = actions.copy()
        for column in ENDPOINT_COLUMNS:
            result[column] = np.nan if column.endswith("frame_id") or column.endswith("duration") else ""
        mask = result.get("action_type", pd.Series("", index=result.index)).eq("pass")
        mask |= result.get("spadl_type", pd.Series("", index=result.index)).isin(["pass", "cross"])
        for index, action in result.loc[mask].iterrows():
            end = self.resolve(action)
            result.loc[index, list(ENDPOINT_COLUMNS)] = [end.observed, end.trajectory, end.source,
                end.event_id, PASS_DURATION, end.duration, end.reason, end.candidate, end.candidate_duration]
        result[MAX_DURATION_KEY] = self.max_pass_duration
        return result


def ensure_pass_endpoints(match, *, annotate=True):
    maximum = getattr(match, MAX_DURATION_KEY, None)
    resolver = getattr(match, "_pass_endpoint_resolver", None)
    if resolver is None or resolver.max_pass_duration != maximum:
        events = getattr(match, "official_pass_events", None)
        if events is None:
            events = load_official_events(getattr(match, "match_id", None))
        resolver = match._pass_endpoint_resolver = PassEndpointResolver(match.tracking, match.fps, events, maximum)
    definitions = match.actions.get("pass_endpoint_definition")
    limits = match.actions.get(MAX_DURATION_KEY)
    current_limit = limits is not None and (limits.isna().all() if maximum is None else limits.eq(maximum).all())
    if annotate and (definitions is None or not definitions.isin(["", PASS_DURATION]).all() or not current_limit):
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
