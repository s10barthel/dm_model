"""Official-event endpoints shared by pass labels, trajectories and visuals.

Contact timing uses raw event differences anchored to the synchronized launch.
It deliberately does not use synchronized contact frames or infer contacts from
ball motion. These endpoints do not replace possession/reception annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

PASS_DURATION_KEY = "pass_duration_definition"
PASS_DURATION = "official_event_observed_end_v1"
ANGLE_ONLY_KEY = "angle_only_definition"
ANGLE_ONLY = "official_event_ball_direction_1s_v1"
TRAJECTORY_WINDOW_SECONDS = 1.0
ENDPOINT_COLUMNS = ("observed_end_frame_id", "trajectory_end_frame_id", "pass_endpoint_source",
                    "pass_endpoint_event_id", "pass_endpoint_definition", "pass_duration")
CONTACT_TAGS = {"Play", "ShotAtGoal", "OtherBallAction", "BallClaiming", "TacklingGame",
                "Foul", "Offside", "FinalWhistle", "BallOut", "BallOutOfPlay", "OutOfPlay",
                "KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "RefereeBall"}


def event_key(value):
    if pd.isna(value):
        return ""
    return str(value).removesuffix(".0")


def utc(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def load_official_events(match_id):
    """Read raw contact events once per match; synced controls retain raw times."""
    from project_config import RAW_SEASON_ROOTS, CONTROL_EVENTS_SYNCED_DIR

    if not match_id or not str(match_id).startswith("DFL-MAT-"):
        return pd.DataFrame()
    for root in RAW_SEASON_ROOTS.values():
        path = root / "event_data" / f"{match_id}.xml"
        if not path.is_file():
            continue
        events = ET.parse(path).getroot().findall(".//Event")
        deleted = {event_key(node.attrib.get("EventId", event.attrib.get("EventId"))) for event in events
                   for node in event if node.tag == "Delete"}
        records = []
        for event in events:
            node = next((node for node in event if node.tag != "Qualifier"), None)
            key = event_key(event.attrib.get("EventId"))
            if node is None or node.tag not in CONTACT_TAGS or key in deleted:
                continue
            records.append(dict(event_id=key, utc_timestamp=utc(event.attrib.get("EventTime")),
                                raw_tag=node.tag))
        return pd.DataFrame(records)
    path = CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Raw Sportec events/control timestamps are required for pass endpoints: {match_id}")


@dataclass(frozen=True)
class PassEndpoint:
    observed: int | None = None
    trajectory: int | None = None
    source: str = "unavailable"
    event_id: str = ""
    duration: float = 0.0


class PassEndpointResolver:
    def __init__(self, tracking, fps=25, events=None):
        self.tracking, self.fps = tracking, fps
        self.period_ends = {}
        self.episode_ends = {}
        period = tracking.get("period_id", pd.Series(0, index=tracking.index))
        for key, rows in tracking.groupby(period):
            self.period_ends[key] = int(rows.index.max())
        if {"episode_id", "ball_state"}.issubset(tracking):
            live = tracking.loc[tracking.ball_state.eq("alive")]
            for (period_id, episode), rows in live.groupby([period.loc[live.index], "episode_id"]):
                self.episode_ends[(period_id, episode)] = int(rows.index.max())
        self.events = (events if events is not None else pd.DataFrame()).copy()
        self.event_times = {}
        self.following = {}
        if not self.events.empty:
            key_column = "event_id" if "event_id" in self.events else "original_event_id"
            self.events["_key"] = self.events[key_column].map(event_key)
            self.events["_time"] = utc(self.events.utc_timestamp)
            if "raw_tag" in self.events:
                self.events = self.events[self.events.raw_tag.isin(CONTACT_TAGS)]
            self.events = self.events.dropna(subset=["_time"]).drop_duplicates("_key").sort_values("_time")
            self.event_times = dict(zip(self.events._key, self.events._time))
            # Same-time annotations cannot end the flight. Each launch maps to
            # the first strictly later official event, regardless of KPI order.
            times = self.events._time.array
            for key, time in self.event_times.items():
                position = times.searchsorted(time, side="right")
                if position < len(times):
                    row = self.events.iloc[position]
                    self.following[key] = (row._time, row._key)

    def resolve(self, action):
        frame = action.get("frame_id")
        if pd.isna(frame) or frame not in self.tracking.index:
            return PassEndpoint()
        frame = int(frame)
        snapshot = self.tracking.loc[frame]
        period = snapshot.get("period_id", 0)
        if pd.notna(action.get("period_id")) and action.get("period_id") != period:
            return PassEndpoint()
        boundary = self.period_ends.get(period, frame)
        boundary_source = "period_end"
        episode_end = self.episode_ends.get((period, snapshot.get("episode_id")))
        if episode_end is not None and episode_end >= frame:
            boundary = min(boundary, episode_end)
            boundary_source = "episode_last_live"
        elif "ball_state" in self.tracking:
            later = self.tracking.loc[frame:boundary]
            dead = later.index[~later.ball_state.eq("alive")]
            if len(dead):
                live = later.index[(later.index < dead[0]) & later.ball_state.eq("alive")]
                boundary = int(live[-1]) if len(live) else frame
                boundary_source = "episode_last_live"

        end, source, contact_id = None, "unavailable", ""
        reception = action.get("receive_frame_id")
        success = pd.notna(action.get("success")) and bool(action.get("success"))
        provenance = action.get("endpoint_source")
        official = pd.isna(provenance) or provenance in {"kpi_csv", "reception", "kpi_xml"}
        if success and official and pd.notna(reception) and reception >= frame:
            if reception in self.tracking.index and self.tracking.loc[reception].get("period_id", 0) == period:
                end, source = int(reception), "kpi_reception"
        if end is None:
            key = event_key(action.get("original_event_id"))
            next_event = self.following.get(key)
            if next_event is not None:
                timestamp, contact_id = next_event
                delta = (timestamp - self.event_times[key]).total_seconds()
                end, source = frame + int(round(delta * self.fps)), "subsequent_event"
        if end is None:
            # An official episode/period boundary is itself an observed end.
            # Without live-state metadata a missing event remains unavailable.
            if "ball_state" not in self.tracking:
                return PassEndpoint()
            end, source = boundary, boundary_source
        if end > boundary:
            end, source = boundary, boundary_source
        capped = min(end, frame + max(int(round(self.fps * TRAJECTORY_WINDOW_SECONDS)), 1))
        duration = (end - frame) / self.fps
        return PassEndpoint(end, capped, source, contact_id, max(float(duration), 0.0))

    def annotate(self, actions):
        result = actions.copy()
        for column in ENDPOINT_COLUMNS:
            result[column] = np.nan if column.endswith("frame_id") or column == "pass_duration" else ""
        mask = result.get("action_type", pd.Series("", index=result.index)).eq("pass")
        mask |= result.get("spadl_type", pd.Series("", index=result.index)).isin(["pass", "cross"])
        for index, action in result.loc[mask].iterrows():
            endpoint = self.resolve(action)
            result.loc[index, list(ENDPOINT_COLUMNS)] = [endpoint.observed, endpoint.trajectory, endpoint.source,
                                                        endpoint.event_id, PASS_DURATION, endpoint.duration]
        return result


def ensure_pass_endpoints(match, *, annotate=True):
    resolver = getattr(match, "_pass_endpoint_resolver", None)
    if resolver is None:
        events = getattr(match, "official_pass_events", None)
        if events is None:
            events = load_official_events(getattr(match, "match_id", None))
        resolver = match._pass_endpoint_resolver = PassEndpointResolver(match.tracking, match.fps, events)
    definitions = match.actions.get("pass_endpoint_definition")
    if annotate and (definitions is None or not definitions.isin(["", PASS_DURATION]).all()):
        match.actions = resolver.annotate(match.actions)
    return resolver


def trajectory_samples(tracking, action):
    """Finite samples strictly inside the resolved segment; no coordinate fallback."""
    start, end = action.get("frame_id"), action.get("trajectory_end_frame_id")
    if pd.isna(start) or pd.isna(end) or end <= start or start not in tracking.index or end not in tracking.index:
        return None
    if not {"ball_x", "ball_y"}.issubset(tracking):
        return None
    segment = tracking.loc[int(start):int(end)].copy()
    if "ball_z" not in segment:
        segment["ball_z"] = 0.0
    finite = np.isfinite(segment[["ball_x", "ball_y", "ball_z"]].to_numpy(dtype=float)).all(axis=1)
    segment = segment.loc[finite]
    if len(segment) < 2:
        return None
    return segment
