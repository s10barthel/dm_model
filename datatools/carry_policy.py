"""Shared, versioned Sportec control reconstruction and tracking validation.

The engine describes observed possession intervals, independently of training's
minimum duration and of inference's canonical action observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

DEFINITION = "sportec_fernandez_1s_v5"
ENDPOINT_COUPLED_DEFINITION = "sportec_fernandez_1s_v4"
PREVIOUS_DEFINITION = "sportec_fernandez_1s_v3"
LEGACY_DEFINITION = "sportec_fernandez_1s_v2"
PREVIOUS_STATE_CONTRACT = "sportec_possession_states_v4"
ENDPOINT_COUPLED_STATE_CONTRACT = "sportec_possession_states_v5"
STATE_CONTRACT = "sportec_possession_states_v6"


@dataclass(frozen=True)
class CarryPolicy:
    acquisition_distance: float = 3.0
    separation_distance: float = 5.0
    separation_seconds: float = 1.0
    terminal_grace_seconds: float = 0.2

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError("Carry policy parameters must be finite and nonnegative")
        if self.separation_seconds == 0:
            raise ValueError("Separation duration must be positive")


def player(value):
    return isinstance(value, str) and re.fullmatch(r"(?:home|away)_\d+", value) is not None


def truth(value):
    return str(value).lower() in {"true", "1", "1.0"}


def number(value):
    try:
        x = float(value)
        return int(x) if math.isfinite(x) and x.is_integer() else None
    except (ValueError, TypeError):
        return None


def event_key(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).removesuffix(".0")


def run_policy(root):
    """Missing metadata means legacy; never silently reinterpret old artifacts."""
    import json
    path = Path(root) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    carry = metadata.get("carry_variant") or {}
    definition = carry.get("definition", LEGACY_DEFINITION)
    if definition not in {DEFINITION, ENDPOINT_COUPLED_DEFINITION, PREVIOUS_DEFINITION, LEGACY_DEFINITION}:
        raise ValueError(f"Unsupported carry definition: {definition}")
    return definition, CarryPolicy(**carry.get("validation_parameters", {}))


def tracking_table(tracking):
    return tracking.set_index("frame_id") if "frame_id" in tracking.columns else tracking


def acquisition_reason(tracking, frame, period, actor, policy):
    if not player(actor) or frame not in tracking.index:
        return "missing_acquisition_tracking"
    row = tracking.loc[frame]
    columns = ["ball_x", "ball_y", f"{actor}_x", f"{actor}_y"]
    if any(not np.isfinite(pd.to_numeric(row.get(c), errors="coerce")) for c in columns):
        return "missing_acquisition_tracking"
    if row.get("period_id", period) != period:
        return "period_boundary"
    if np.hypot(row.ball_x - row[f"{actor}_x"], row.ball_y - row[f"{actor}_y"]) > policy.acquisition_distance:
        return "acquisition_distance"
    return None


def validate_spell(tracking, start, end, period, actor, fps, policy):
    """Return a reason plus diagnostics; loss endpoints need not be near the carrier."""
    diagnostics = {}
    if end is None or end < start:
        return "invalid_interval", diagnostics
    reason = acquisition_reason(tracking, start, period, actor, policy)
    if reason and reason != "acquisition_distance":
        return reason, diagnostics
    rows = tracking.loc[start:end]
    if len(rows) != end - start + 1 or end not in tracking.index:
        return "tracking_gap", diagnostics
    if "period_id" in rows and not rows.period_id.eq(period).all():
        return "period_boundary", diagnostics
    columns = ["ball_x", "ball_y", f"{actor}_x", f"{actor}_y"]
    if not np.isfinite(rows[columns].to_numpy(dtype=float)).all():
        return "missing_spell_tracking", diagnostics
    distances = np.hypot(rows.ball_x - rows[f"{actor}_x"], rows.ball_y - rows[f"{actor}_y"])
    diagnostics.update(acquisition_distance=float(distances.iloc[0]), terminal_distance=float(distances.iloc[-1]))
    if "ball_z" in rows:
        z = pd.to_numeric(rows.ball_z, errors="coerce")
        diagnostics["max_ball_z"] = float(z.max()) if z.notna().any() else None
    if reason:
        return reason, diagnostics
    if "ball_state" in rows and not rows.ball_state.eq("alive").all():
        return "dead_ball_interval", diagnostics
    grace = int(round(policy.terminal_grace_seconds * fps))
    interior = distances.iloc[:-grace] if grace else distances
    far = interior.gt(policy.separation_distance)
    longest = int(far.groupby((~far).cumsum()).sum().max()) if len(far) else 0
    diagnostics["separation_frames"] = longest
    if longest >= math.ceil(policy.separation_seconds * fps):
        return "sustained_separation", diagnostics
    return None, diagnostics


def reconstruct_spells(controls, canonical, tracking, fps=25, policy=None, *, independent_receipts=True):
    """Return audited spells. Only explicit control evidence can start an interval."""
    policy = policy or CarryPolicy()
    tracking = tracking_table(tracking)
    rows = controls.to_dict("records")
    canonical = canonical.reset_index(drop=True)
    canonical_rows = {event_key(r.original_event_id): r for r in canonical.itertuples()}
    for row in rows:
        action = canonical_rows.get(event_key(row.get("event_id")))
        if action is not None and number(action.frame_id) is not None:
            row.update(frame_id=action.frame_id, period_id=action.period_id,
                       object_id=action.object_id, frame_source="canonical")
    seen = {event_key(r.get("event_id")) for r in rows}
    for event, action in canonical_rows.items():
        if event not in seen and action.spadl_type in {"pass", "cross", "shot"}:
            rows.append(dict(event_id=event, event_kind=action.spadl_type, frame_id=action.frame_id,
                             period_id=action.period_id, object_id=action.object_id, frame_source="canonical"))
        if action.spadl_type not in {"pass", "cross"} or truth(getattr(action, "offside", False)):
            continue
        # An endpoint terminates a pass; only a confirmed reception establishes control.
        endpoint_kind = getattr(action, "pass_endpoint_source", None)
        success = independent_receipts and truth(getattr(action, "success", getattr(action, "source_event_success", False)))
        independent_receipt = getattr(action, "official_receipt_frame_id", None)
        if not success and endpoint_kind is not None and endpoint_kind != "official_reception":
            continue
        receiver = getattr(action, "receiver_id", None)
        if not player(receiver) or receiver[:4] != str(action.object_id)[:4]:
            continue
        if not success and not truth(getattr(action, "training_sample_eligible", True)):
            continue
        if success and hasattr(action, "official_receipt_frame_id"):
            frame = number(independent_receipt)
        elif success and endpoint_kind not in {None, "official_reception"}:
            original_source = getattr(action, "original_endpoint_source", None)
            if original_source not in {"kpi_csv", "reception", "official_reception"}:
                continue
            frame = number(getattr(action, "original_receive_frame_id", None))
        else:
            frame = number(getattr(action, "receive_frame_id", None))
        if frame is None:
            continue
        launch = number(action.frame_id)
        invalid = "invalid_receipt_interval" if (frame not in tracking.index
            or (launch is not None and frame < launch)
            or (not success and (launch is None or launch not in tracking.index
                or tracking.loc[launch].get("period_id", action.period_id) != action.period_id))
            or tracking.loc[frame].get("period_id", action.period_id) != action.period_id) else None
        rows.append(dict(event_id=f"receipt:{event}", parent_event_id=event,
                         event_kind="invalid_receipt" if invalid else "receipt", frame_id=frame,
                         period_id=action.period_id, object_id=receiver, invalid_reason=invalid,
                         frame_source=getattr(action, "endpoint_source", "unknown")))
    if independent_receipts:
        known_receipts = {r["parent_event_id"] for r in rows if r["event_kind"] == "receipt"}
        rows = [r for r in rows if number(r.get("frame_id")) is not None or event_key(r.get("event_id")) not in known_receipts]
    rows = [r for r in rows if number(r.get("period_id")) in (1, 2)]
    uncertainty_windows = []
    for index, row in enumerate(rows):
        row["event_id"] = event_key(row.get("event_id"))
        row["_order"] = index
        frame = number(row.get("frame_id"))
        row["_missing"] = frame is None
        if frame is None:
            # Position a barrier near its raw-time neighbours, never invent a physical endpoint.
            timestamp = pd.to_datetime(row.get("utc_timestamp"), utc=True, errors="coerce")
            candidates = [(pd.to_datetime(r.get("utc_timestamp"), utc=True, errors="coerce"), number(r.get("frame_id")))
                          for r in rows if r.get("period_id") == row.get("period_id") and number(r.get("frame_id")) is not None]
            prior = [(ts, f) for ts, f in candidates if pd.notna(ts) and pd.notna(timestamp) and ts <= timestamp]
            later = [(ts, f) for ts, f in candidates if pd.notna(ts) and pd.notna(timestamp) and ts > timestamp]
            frame = max(prior)[1] if prior else min((f for _, f in candidates), default=0)
            limit = min(later)[1] if later else max((f for _, f in candidates), default=frame)
            uncertainty_windows.append((row["period_id"], min(frame, limit), max(frame, limit)))
        row["_frame"] = frame
    groups = {}
    for row in rows:
        groups.setdefault((row["period_id"], row["_frame"]), []).append(row)
    priority = {"receipt": 0, "claim": 1, "tackle": 2, "other": 3, "pass": 4, "cross": 4, "shot": 4}
    immediate_parents = {(r["period_id"], r["_frame"], r.get("parent_event_id")) for r in rows
                         if r["event_kind"] == "receipt" and r.get("parent_event_id") in canonical_rows
                         and number(canonical_rows[r["parent_event_id"]].frame_id) == r["_frame"]}
    rows.sort(key=lambda r: (r["period_id"], r["_frame"],
        -1 if (r["period_id"], r["_frame"], r["event_id"]) in immediate_parents else priority.get(r["event_kind"], 5), r["_order"]))
    spells, current, held, previous_period = [], None, set(), None
    chain = 0

    def finish(row, success, reason, receiver=None):
        nonlocal current
        if current is None:
            return
        end = row["_frame"]
        error, diagnostic = validate_spell(tracking, current["start_frame"], end, current["period_id"],
                                            current["carrier_id"], fps, policy)
        error = current.get("invalid_reason") or error
        if any(p == current["period_id"] and left < end and right > current["start_frame"]
               for p, left, right in uncertainty_windows):
            error = "missing_control_frame"
        valid = success is not None and error is None
        spells.append({**current, **diagnostic, "terminal_frame": end, "terminal_event_id": row["event_id"],
                       "terminal_kind": row["event_kind"], "terminal_source": row.get("frame_source", "unknown"),
                       "success": success, "receiver_id": receiver, "valid": valid,
                       "duration_seconds": (end - current["start_frame"]) / fps,
                       "reason": error or reason, "terminal_reason": reason})
        current = None

    def start(row, actor, kind):
        nonlocal current
        if not player(actor) or actor in held:
            return
        if current and current["carrier_id"] == actor:
            return  # Companion evidence corroborates; it must not reset acquisition.
        finish(row, None, "unobserved_release")
        current = dict(carry_id=len(spells) + 1, carrier_id=actor, period_id=int(row["period_id"]), chain=chain,
                       start_frame=row["_frame"], start_event_id=row["event_id"], start_kind=kind,
                       start_source=row.get("frame_source", "unknown"),
                       invalid_reason=row.get("invalid_reason") or acquisition_reason(
                           tracking, row["_frame"], row["period_id"], actor, policy))

    for row in rows:
        kind, actor, frame, period = row["event_kind"], row.get("object_id"), row["_frame"], row["period_id"]
        if previous_period is not None and period != previous_period:
            finish(row, None, "period_boundary")
            held.clear()
            chain += 1
        previous_period = period
        if row["_missing"]:
            finish(row, None, "missing_control_frame")
            chain += 1
            continue
        companions = groups[(period, frame)]
        excluded = {r.get("object_id") for r in companions if r["event_kind"] == "claim"
                    and r.get("claim_type") in {"BallHeld", "Block"}}
        held.update(r.get("object_id") for r in companions if r["event_kind"] == "claim" and r.get("claim_type") == "BallHeld")
        carrier = current["carrier_id"] if current else None
        releases = {r.get("object_id") for r in companions if r["event_kind"] in {"pass", "cross", "shot"}}
        if kind == "receipt":
            if actor not in excluded:
                start(row, actor, "receipt")
        elif kind == "claim":
            subtype = row.get("claim_type")
            if subtype not in {"InterceptedBall", "BallClaimed"}:
                finish(row, None, "excluded_claim")
                chain += 1
                continue
            if actor not in excluded:
                if actor != carrier:
                    # An interception cannot directly dispose an uninterrupted carry.
                    success = False if carrier and str(actor)[:4] != carrier[:4] and subtype == "BallClaimed" else None
                    finish(row, success, "opponent_ball_claim" if success is False else "unobserved_release", actor)
                if subtype == "BallClaimed":
                    held.discard(actor)
                start(row, actor, "ball_claim")
        elif kind == "tackle":
            winner, loser = row.get("winner_id"), row.get("loser_id")
            changed = truth(row.get("possession_change"))
            retained = next((row.get(p) for p, role in [("winner_id", "winner_role"), ("loser_id", "loser_role")]
                             if row.get(role) == "withBallControl" and player(row.get(p))), None)
            if changed and player(winner):
                if winner == carrier:
                    continue  # Claim/tackle pair for an already established acquisition.
                finish(row, False if loser == carrier and winner[:4] != str(carrier)[:4] else None,
                       "possession_changing_tackle" if loser == carrier else "ambiguous_tackle", winner)
                if winner not in excluded:
                    held.discard(winner)
                    start(row, winner, "tackle_win")
            elif not changed and retained:
                if retained == carrier:
                    continue
                if retained in releases or retained in excluded:
                    continue
                finish(row, None, "inconsistent_tackle")
                start(row, retained, "retained_tackle")
            else:
                finish(row, None, "ambiguous_tackle")
                chain += 1
        elif kind in {"pass", "cross", "shot"}:
            finish(row, True if actor == carrier else None,
                   f"carrier_{kind}" if actor == carrier else "unexpected_carrier", actor)
            if actor != carrier:
                chain += 1
            held.discard(actor)
        elif kind == "foul":
            if carrier is not None and row.get("fouler_id") == carrier:
                finish(row, False, "carrier_committed_foul", row.get("fouled_id"))
            elif carrier is not None and row.get("fouled_id") == carrier:
                finish(row, True, "carrier_fouled", carrier)
            else:
                finish(row, None, "foul_without_carrier")
            chain += 1
        elif kind == "other":
            corroborated = any(r["event_kind"] in {"receipt", "claim", "tackle"}
                               and (r.get("object_id") == actor or (r["event_kind"] == "tackle"
                                    and truth(r.get("possession_change")) and r.get("winner_id") == actor))
                               and r.get("claim_type") not in {"BallHeld", "Block"} for r in companions)
            if truth(row.get("clearance")):
                finish(row, False if corroborated and carrier and player(actor) and actor[:4] != carrier[:4] else None,
                       "opponent_clearance" if carrier and player(actor) and actor[:4] != carrier[:4] else "carrier_clearance", actor)
                chain += 1
            elif not corroborated:
                finish(row, None, "uncorroborated_other")
                chain += 1
        else:
            finish(row, None, f"direct_{kind}")
            chain += 1
    if current:
        finish(dict(_frame=current["start_frame"], event_id="end_of_data", event_kind="end_of_data"), None, "end_of_data")
    columns = ["carry_id", "carrier_id", "period_id", "chain", "start_frame", "start_event_id", "start_kind",
               "start_source", "invalid_reason", "terminal_frame", "terminal_event_id", "terminal_kind",
               "terminal_source", "success", "receiver_id", "valid", "duration_seconds", "reason", "terminal_reason"]
    return pd.DataFrame(spells) if spells else pd.DataFrame(columns=columns)
