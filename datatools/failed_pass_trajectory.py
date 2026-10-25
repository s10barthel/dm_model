"""Model-only observation cutoffs. Never modifies official pass endpoints."""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from datatools.pass_trajectory import event_key

KEY = "failed_pass_trajectory_policy"
VERSION = "proximity_xy_peak_v1"
SETTINGS = dict(radius=1., guard_seconds=.12, velocity_seconds=.08,
                angle_degrees=10., speed_change=5., minimum_incoming_speed=3.,
                jump_residual=1., jump_speed=60., jump_angle=120.)


def changes(incoming, outgoing):
    a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
    if not np.isfinite([a, b]).all() or a < SETTINGS["minimum_incoming_speed"]:
        return None
    angle = float(np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing) / (a*b), -1, 1)))) if b > 1e-9 else 0.
    delta = float(abs(b-a))
    return angle, delta, max(angle/SETTINGS["angle_degrees"], delta/SETTINGS["speed_change"])


def detect(action, tracking, cutoff, fps=25):
    """Return a deterministic diagnostic, including negative and unavailable results."""
    result = dict(status="unavailable", reason=None, original_cutoff=cutoff,
                  cutoff=cutoff, impact=None, defender=None, angle=None, speed_change=None,
                  score=None, policy=VERSION, tracking_jump_frames=[])
    if pd.isna(cutoff) or pd.isna(action.get("frame_id")):
        result["reason"] = "missing_official_observation"
        return result
    start, end = int(action.frame_id), int(cutoff)
    width = int(round(SETTINGS["velocity_seconds"]*fps))
    guard = int(np.ceil(SETTINGS["guard_seconds"]*fps))
    if width < 1 or end <= start:
        result["reason"] = "insufficient_prefix"
        return result
    frames = np.arange(start, min(end, start+round(fps))+width+1)
    rows = tracking.reindex(frames)
    xy = rows.reindex(columns=["ball_x", "ball_y"]).to_numpy(dtype=float)
    times = rows.timestamp.to_numpy(dtype=float) if "timestamp" in rows else frames/fps
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(times)
    if "period_id" in rows:
        valid &= rows.period_id.eq(rows.period_id.iloc[0]).to_numpy()
    bad = np.zeros(len(rows), dtype=bool)
    for i in range(1, len(rows)-1):
        if not valid[i-1:i+2].all():
            continue
        left, right = xy[i]-xy[i-1], xy[i+1]-xy[i]
        norms = np.linalg.norm(left)*np.linalg.norm(right)
        reversal = norms > 0 and np.dot(left, right)/norms <= np.cos(np.radians(120))
        bad[i] = (np.linalg.norm(xy[i]-(xy[i-1]+xy[i+1])/2)>1
                  and max(np.linalg.norm(left), np.linalg.norm(right))*fps>60 and reversal)
    result["tracking_jump_frames"] = frames[bad].tolist()

    def velocity(a, b):
        if a < 0 or b >= len(rows) or not valid[a:b+1].all() or bad[a:b+1].any():
            return None
        if not np.allclose(np.diff(times[a:b+1]), 1/fps, atol=1e-6, rtol=0):
            return None
        return (xy[b]-xy[a])/(times[b]-times[a])

    opponent = "away" if str(action.get("object_id", "")).startswith("home") else "home"
    players = sorted(c[:-2] for c in rows if re.fullmatch(opponent+r"_\d+_x", c) and c[:-2]+"_y" in rows)
    if not players or not valid[:min(end-start+1, len(valid))].all():
        result["reason"] = "missing_tracking"
        return result
    if not np.allclose(np.diff(times[:min(end-start+1, len(times))]), 1/fps, atol=1e-6, rtol=0):
        result["reason"] = "tracking_time_discontinuity"
        return result
    positions = np.stack([rows[[p+"_x", p+"_y"]].to_numpy(dtype=float) for p in players], axis=1)
    distances = np.linalg.norm(positions-xy[:, None, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    if not np.isfinite(distances[:end-start+1]).any(axis=1).all():
        result["reason"] = "missing_defender_tracking"
        return result
    nearest = distances.argmin(axis=1)
    near = distances.min(axis=1) <= SETTINGS["radius"]
    last = min(end-start, round(fps))
    peaks = []
    i = guard
    while i <= last:
        if not near[i]:
            i += 1
            continue
        entry, player = i, nearest[i]
        while i+1 <= last and near[i+1] and nearest[i+1] == player:
            i += 1
        departure = i
        incoming, outgoing = velocity(entry-width, entry), velocity(departure, departure+width)
        gate = changes(incoming, outgoing) if incoming is not None and outgoing is not None else None
        if gate is not None and gate[2] >= 1:
            cluster = []
            for j in range(entry, departure+2):
                a, b = velocity(j-width, j), velocity(j, j+width)
                local = changes(a, b) if j <= departure and a is not None and b is not None else None
                if local is not None and local[2] >= 1:
                    cluster.append((j, local))
                elif cluster:
                    peak, values = max(cluster, key=lambda item: item[1][2])
                    peaks.append((peak, values, players[player]))
                    cluster = []
        i += 1
    result.update(status="unchanged", reason="no_qualifying_contact")
    if peaks:
        peak, values, player = peaks[0]
        clean = start+peak-1
        if velocity(0, peak-1) is None:
            result.update(status="unavailable", reason="untrustworthy_prefix")
        elif clean < end:
            result.update(status="shortened", reason=None, cutoff=clean, impact=start+peak,
                          defender=player, angle=values[0], speed_change=values[1], score=values[2])
    elif bad[:last+1].any():
        result.update(status="unavailable", reason="tracking_jump")
    return result


def observation(match, action, original_cutoff):
    """Read pinned decisions, or compute them for a newly generated match."""
    if action.get("action_type") != "pass" or bool(action.get("success", False)):
        return original_cutoff
    root = getattr(match, "runtime_feature_root", None)
    if root is None:
        return original_cutoff
    root = Path(root)
    metadata_path = root / "metadata.json"
    stamp = (str(metadata_path), metadata_path.stat().st_mtime_ns if metadata_path.exists() else None)
    if getattr(match, "_failed_pass_metadata_stamp", None) != stamp:
        match._failed_pass_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
        match._failed_pass_metadata_stamp = stamp
    metadata = match._failed_pass_metadata
    policy = metadata.get(KEY) or getattr(match, "failed_pass_trajectory_policy", None)
    if not policy:
        return original_cutoff
    if policy != VERSION:
        raise ValueError(f"Unsupported failed-pass trajectory policy: {policy}")
    path = root / "failed_pass_trajectories" / f"{match.match_id}.parquet"
    cache_key = (str(path), path.stat().st_mtime_ns if path.exists() else None)
    if getattr(match, "_failed_pass_cache_key", None) != cache_key:
        if not path.exists():
            raise ValueError(f"Missing failed-pass refresh coverage: {path}. Run --refresh-failed-pass-trajectories.")
        table = pd.read_parquet(path)
        if table.event_key.duplicated().any():
            raise ValueError(f"Duplicate failed-pass trajectory identities: {path}")
        match._failed_pass_records = table.set_index("event_key").to_dict("index")
        match._failed_pass_cache_key = cache_key
    key = event_key(action.get("original_event_id"))
    if not key:
        return None  # A missing identity cannot safely borrow another pass's decision.
    record = match._failed_pass_records.get(key)
    if record is None:
        raise ValueError(f"Missing failed-pass trajectory coverage for event {key}")
    if record.get("policy") != VERSION or record["status"] not in {"shortened", "unchanged", "unavailable"}:
        raise ValueError(f"Invalid failed-pass trajectory decision for event {key}; refresh coverage.")
    if record["status"] == "unavailable":
        return None
    cutoff = record["cutoff"]
    original = record.get("original_cutoff")
    if (pd.isna(original_cutoff) or pd.isna(original) or original != original_cutoff
            or pd.isna(cutoff) or cutoff != int(cutoff)
            or not action.get("frame_id", cutoff) < cutoff <= original_cutoff):
        raise ValueError(f"Stale failed-pass trajectory cutoff for event {key}; run --refresh-failed-pass-trajectories.")
    return int(cutoff)


def prepare_new_match(match):
    """Create coverage once before any receiver relabeling in fresh generation."""
    from datatools.pass_trajectory import ensure_pass_endpoints
    from datatools.artifact_storage import atomic_parquet_save
    path = Path(match.runtime_feature_root) / "failed_pass_trajectories" / f"{match.match_id}.parquet"
    if path.exists():
        # Reading it also validates version/coverage when observations are used.
        return
    resolver = ensure_pass_endpoints(match, annotate=False)
    records = []
    for index, action in match.actions.loc[match.actions.action_type.eq("pass") & ~match.actions.success].iterrows():
        key = event_key(action.get("original_event_id"))
        record = detect(action, match.tracking, resolver.resolve(action).trajectory, match.fps)
        if not key:
            key = f"missing-event:{index}"
            record.update(status="unavailable", reason="missing_event_identity")
        record.update(event_key=key, action_index=int(index))
        records.append(record)
    table = pd.DataFrame(records, columns=None if records else ["event_key", "status", "cutoff"])
    if table.event_key.duplicated().any():
        raise ValueError("Duplicate failed-pass event identities")
    atomic_parquet_save(table, path)


def validate_graph_coverage(match, graphs, labels, action_indices=None, selected_indices=None):
    """Prevent graph overrides/caches from bypassing refreshed model inputs."""
    from datatools.pass_trajectory import ensure_pass_endpoints
    if labels is None or getattr(match, "runtime_feature_root", None) is None:
        return
    root = Path(match.runtime_feature_root)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    if not metadata.get(KEY):
        return
    if action_indices is not None:
        labels = [[index] for index in action_indices]
    selected = set(selected_indices) if selected_indices is not None else None
    for graph, label in zip(graphs, labels):
        if graph is None:
            continue
        index = int(label[0])
        if (selected is not None and index not in selected) or index not in match.actions.index:
            continue
        action = match.actions.loc[index]
        if action.get("action_type") != "pass" or bool(action.get("success", False)):
            continue
        expected = observation(match, action, ensure_pass_endpoints(match, annotate=False).resolve(action).trajectory)
        if expected is None or getattr(graph, KEY, None) != VERSION or getattr(graph, "trajectory_end_frame_id", None) != expected:
            raise ValueError(f"Stale/unavailable failed-pass graph for action {index}; refresh failed-pass trajectories.")
