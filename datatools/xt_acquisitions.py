"""Sparse acquisition locations without changing canonical action indices/counts."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from datatools.return_windows import ACQUISITION_TYPES

CONTEXT_KEY = "xt_acquisition_locations"


def _event_key(value):
    return str(value).removesuffix(".0") if pd.notna(value) else ""


def _identity(events):
    columns = [name for name in ("original_event_id", "period_id", "frame_id") if name in events]
    return sha256(events[columns].to_csv(index=True).encode()).hexdigest()


def acquisition_context(events, match_id, root=None, *, tracking=None):
    """Return context, an optional new cache, and input provenance for staging.

    Existing cache snapshots are authoritative. Only a missing cache reads ball
    coordinates; graph generation can supply its already loaded tracking table.
    """
    from project_config import CONTROL_EVENTS_SYNCED_DIR, DATA_ROOT
    from datatools.xt_refresh import fingerprint, file_signature
    if match_id is None or "original_event_id" not in events:
        return events, None, {}
    cache = Path(root) / "acquisition_locations" / f"{match_id}.parquet" if root is not None else None
    identity = _identity(events)
    if cache is not None and cache.is_file():
        locations = pd.read_parquet(cache)
        required = {"before_event_index", "period_id", "object_id", "spadl_type", "start_x", "start_y", "event_id"}
        if (locations.attrs.get("version") != 1 or not required <= set(locations)
                or locations.attrs.get("canonical_identity") != identity):
            raise ValueError(f"Acquisition locations do not match canonical events: {cache}")
        prepared = events.copy()
        prepared.attrs[CONTEXT_KEY] = locations.to_dict("records")
        return prepared, None, {"inputs": {cache: fingerprint(cache)}, "ball_files_loaded": 0}
    control_path = CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet"
    if root is not None:
        pinned = Path(root) / "carry_sidecars" / "control_events" / f"{match_id}.parquet"
        if pinned.is_file():
            control_path = pinned
    if not control_path.is_file():
        return events, None, {"canonical_only": True, "ball_files_loaded": 0}
    controls = pd.read_parquet(control_path)
    required = {"event_id", "event_kind", "frame_id", "period_id"}
    if not required <= set(controls) or controls.event_id.map(_event_key).duplicated().any():
        raise ValueError(f"Invalid synchronized acquisition controls: {control_path}")
    mapping = dict(zip(events.original_event_id.map(_event_key), events.index))
    # Control chronology supplies an insertion point before the next canonical
    # event. These extra rows will never be eligible xT actions.
    before = controls.event_id.map(_event_key).map(mapping).bfill()
    kinds, actors = [], []
    for row in controls.to_dict("records"):
        kind, actor = row["event_kind"], row.get("object_id")
        if kind == "tackle":
            actor = row.get("winner_id")
        elif kind == "claim":
            subtype = str(row.get("claim_type", "")).lower().replace("_", "")
            kind = {"interceptedball": "interception", "block": "shot_block",
                    "ballheld": "keeper_claim", "punch": "keeper_punch",
                    "badtouch": "bad_touch", "badballtouch": "bad_touch"}.get(subtype, "ball_recovery")
        elif kind == "other":
            kind = "clearance" if str(row.get("clearance", False)).lower() in {"true", "1"} else "other"
        kinds.append(kind)
        actors.append(actor)
    controls = controls.assign(spadl_type=kinds, object_id=actors, before_event_index=before)
    candidates = controls.loc[
        controls.spadl_type.isin(ACQUISITION_TYPES)
        & controls.object_id.astype(str).str[:4].isin(["home", "away"])
        & ~controls.event_id.map(_event_key).isin(mapping)
        & controls.before_event_index.notna() & controls.frame_id.notna()
        & controls.period_id.isin(events.period_id.dropna())
    ].copy()
    tracking_path = DATA_ROOT / "tracking_processed" / f"{match_id}.parquet"
    provenance = dict(inputs={control_path: fingerprint(control_path)}, signatures={}, ball_files_loaded=0)
    if tracking is None and not candidates.empty:
        if not tracking_path.is_file():
            return events, None, {**provenance, "canonical_only": True, "missing_ball_tracking": str(tracking_path)}
        provenance["signatures"][tracking_path] = file_signature(tracking_path)
        tracking = pd.read_parquet(tracking_path, columns=["ball_x", "ball_y"])
        provenance["ball_files_loaded"] = 1
    columns = ["before_event_index", "period_id", "object_id", "spadl_type", "start_x", "start_y", "event_id"]
    if candidates.empty:
        locations = pd.DataFrame(columns=columns)
    else:
        table = tracking.set_index("frame_id") if "frame_id" in tracking else tracking
        if not table.index.is_unique:
            raise ValueError("Ball tracking must have unique frame IDs")
        positions = table[["ball_x", "ball_y"]].reindex(candidates.frame_id.to_numpy())
        candidates["start_x"] = positions.ball_x.to_numpy()
        candidates["start_y"] = positions.ball_y.to_numpy()
        candidates = candidates.loc[np.isfinite(candidates.start_x) & np.isfinite(candidates.start_y)]
        locations = candidates[columns].reset_index(drop=True)
    locations.attrs = dict(version=1, canonical_identity=identity,
                           control_sha256=provenance["inputs"][control_path])
    prepared = events.copy()
    prepared.attrs[CONTEXT_KEY] = locations.to_dict("records")
    return prepared, (cache, locations) if cache is not None else None, provenance


def expand_acquisitions(events):
    """Build an xT-only timeline, retaining every canonical event index."""
    records = events.attrs.get(CONTEXT_KEY) or []
    if not records:
        return events
    positions = {index: position for position, index in enumerate(events.index)}
    canonical = events.copy()
    canonical.attrs = {}
    canonical["_xt_order"] = np.arange(len(canonical), dtype=float)
    additions = []
    for record in records:
        before = record["before_event_index"]
        if before not in positions or record["period_id"] != events.loc[before, "period_id"]:
            continue
        additions.append({**record, "_xt_order": positions[before] - .5,
                          "xT": np.nan, "expected_goal": 0., "success": False})
    if not additions:
        return events
    extra = pd.DataFrame(additions, index=-1 - np.arange(len(additions)))
    if extra.index.isin(events.index).any():
        raise ValueError("Canonical indices conflict with acquisition context indices")
    return pd.concat([canonical, extra]).sort_values("_xt_order", kind="stable").drop(columns="_xt_order")
