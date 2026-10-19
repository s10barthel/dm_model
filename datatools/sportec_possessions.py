"""Sportec inference states and spell chains, independent of carry training."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
import torch

from datatools import config, preprocess, utils
from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
from project_config import CONTROL_EVENTS_SYNCED_DIR
from datatools.carry_policy import DEFINITION, STATE_CONTRACT as VALIDATED_STATE_CONTRACT, run_policy
from pc_xpass_contract import PHYSICAL_INPUT_CONTRACT

STATE_CONTRACT = "sportec_possession_states_v3"
START_SCOPE = "spell_start"
PASSING_RESTARTS = {
    "throw_in": "ThrowIn", "goalkick": "GoalKick", "corner_short": "CornerKick",
    "corner_crossed": "CornerKick", "freekick_short": "FreeKick", "freekick_crossed": "FreeKick",
    "kick_off": "KickOff", "kickoff": "KickOff",
}
STATE_METADATA_COLUMNS = [
    "state_id", "state_partition_id", "possession_id", "possessor_id", "period_id",
    "state_frame_id", "frame_role", "pc_cache_match_id", "state_contract", "value_sign",
    "original_start_frame", "original_end_frame", "start_event_id",
    "start_action_id", "start_original_event_id", "end_action_id", "end_original_event_id",
]


def state_contract_for_run(feature_root):
    definition, _ = run_policy(feature_root)
    return VALIDATED_STATE_CONTRACT if definition == DEFINITION else STATE_CONTRACT


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _integer(value):
    try:
        numeric = float(value)
        return int(numeric) if numeric >= 0 and numeric.is_integer() else None
    except (ValueError, TypeError, OverflowError):
        return None


def _player(value):
    return isinstance(value, str) and re.fullmatch(r"(?:home|away)_\d+", value) is not None


def _truth(value):
    return str(value).lower() in {"true", "1", "1.0"}


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _event_id(value):
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def state_identity(match_id, period, frame, possessor, provenance):
    return _digest([provenance.get("state_contract", STATE_CONTRACT), str(match_id), int(period), int(frame), possessor, provenance])


def _table_fingerprint(table):
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=True).values.tobytes()
                          + repr(list(table.columns)).encode()).hexdigest()


def _action_fingerprint(actions):
    # Carry augmentation adds columns and changes numeric dtypes on canonical rows.
    # Only state-defining canonical fields participate in the inference contract.
    fields = ["action_id", "original_event_id", "period_id", "frame_id", "receive_frame_id", "index",
              "object_id", "receiver_id", "next_player_id", "next_type", "spadl_type"]
    table = actions.reindex(columns=fields).copy()
    for column in fields[:6]:
        table[column] = table[column].map(_event_id).astype("string")
    for column in fields[6:]:
        table[column] = table[column].astype("string")
    return _table_fingerprint(table)


def _canonical_fingerprint(canonical):
    flags = canonical.reindex(columns=["success", "offside", "source_has_reception"]).apply(lambda column: column.map(_truth))
    return _digest([_action_fingerprint(canonical), _table_fingerprint(flags)])


@dataclass
class SportecStatePlan:
    match_id: str
    provenance: dict
    observations: dict = field(default_factory=dict)
    spells: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    def observe(self, period, frame, possessor, role, event=None):
        period, frame = _integer(period), _integer(frame)
        if period not in (1, 2) or frame is None or not _player(possessor):
            self.diagnostics.append(dict(event_id=event, role=role, reason="unidentified_player_or_frame"))
            return None
        key = state_identity(self.match_id, period, frame, possessor, self.provenance)
        row = self.observations.setdefault(key, dict(state_id=key, period_id=period,
            state_frame_id=frame, possessor_id=possessor, roles=[], event_ids=[]))
        if role not in row["roles"]:
            row["roles"].append(role)
        if event is not None and event not in row["event_ids"]:
            row["event_ids"].append(event)
        return key

    def manifest(self):
        return dict(state_contract=self.provenance["state_contract"], match_id=self.match_id, provenance=self.provenance,
                    spells=self.spells, actions=self.actions,
                    observations=list(self.observations.values()), diagnostics=self.diagnostics)


def reconstruct_sportec_states(match, match_id, feature_root, *, control_events=None, graph_options=None):
    """Reconstruct all spells and canonical observations before sampling or graph validation."""
    definition, policy = run_policy(feature_root)
    if control_events is None:
        path = CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet"
        local = Path(feature_root) / "carry_sidecars" / "control_events" / f"{match_id}.parquet"
        if definition == DEFINITION:
            path = local
        if not path.exists():
            raise FileNotFoundError(f"Missing synchronized control events {path}; run Sportec preprocessing.")
        control_events = pd.read_parquet(path)
    required = {"event_id", "event_kind", "period_id", "frame_id"}
    if not required.issubset(control_events.columns):
        raise ValueError(f"Invalid synchronized control events: missing {sorted(required - set(control_events))}")
    if control_events.event_id.map(_event_id).duplicated().any():
        raise ValueError("Duplicate synchronized control event IDs")
    canonical = match.events.copy().reset_index(drop=True)
    actions = getattr(match, "actions", canonical).copy()
    if "spadl_type" in actions:
        actions = actions.loc[actions.spadl_type.ne("ball_carry")].copy()
    provenance = dict(state_contract=VALIDATED_STATE_CONTRACT if definition == DEFINITION else STATE_CONTRACT, feature_run_id=Path(feature_root).name,
        canonical_sha256=(_table_fingerprint(match.sportec_canonical_source) if hasattr(match, "sportec_canonical_source")
                          else _canonical_fingerprint(canonical)), control_sha256=_table_fingerprint(control_events),
        resolved_actions_sha256=_action_fingerprint(actions), tracking_sha256=_table_fingerprint(match.tracking),
        lineup_sha256=_table_fingerprint(getattr(match, "lineup", pd.DataFrame())),
        physical_input_contract=PHYSICAL_INPUT_CONTRACT)
    if definition == DEFINITION:
        provenance["carry_validation_parameters"] = asdict(policy)
    plan = SportecStatePlan(str(match_id), provenance)
    canonical["_event_key"] = canonical.original_event_id.map(_event_id)
    ambiguous_events = {event for event, rows in canonical.groupby("_event_key")
                        if len(rows[["period_id", "frame_id", "object_id", "spadl_type"]].drop_duplicates()) > 1}
    unique_events = canonical.drop_duplicates("_event_key", keep="first")
    event_order = {row["_event_key"]: int(i) for i, row in unique_events.iterrows()}
    event_rows = {row["_event_key"]: row for _, row in unique_events.iterrows() if row["_event_key"] not in ambiguous_events}
    control_rows = {_event_id(row.event_id): row for _, row in control_events.iterrows()}
    # Apply existing post-action policy on a copy; no training artifacts are changed.
    post_actions = match.label_post_actions(actions) if hasattr(match, "label_post_actions") else actions
    for index, action in actions.iterrows():
        event = _event_id(action.get("original_event_id"))
        post = post_actions.loc[index]
        pre_id = plan.observe(action.get("period_id"), action.get("frame_id"), action.get("object_id"), "canonical_pre", event)
        source_control = control_rows.get(event)
        if pre_id and (action.get("spadl_type") == "shot_penalty" or
                       (action.get("spadl_type") in config.SHOT and source_control is not None and source_control.event_kind == "restart")):
            plan.observations[pre_id]["roles"].append("shot_specific")
        post_id = plan.observe(action.get("period_id"), post.get("end_frame_id", post.get("receive_frame_id")),
                               post.get("end_player_id", post.get("receiver_id")), "canonical_post", event)
        plan.actions.append(dict(action_index=int(index), action_id=_integer(action.get("action_id")),
            original_event_id=event, period_id=_integer(action.get("period_id")),
            owner_id=action.get("object_id") if _player(action.get("object_id")) else None,
            pre_state_id=pre_id, post_state_id=post_id, spell_id=None,
            start_state_id=None, end_state_id=pre_id, next_state_id=None))
    if definition == DEFINITION:
        from datatools.carry_inference import populate
        return populate(plan, match, control_events, canonical, policy)
    timeline = []
    raw_orders = control_events.event_id.map(_event_id).map(event_order)
    for source_index, (_, raw) in enumerate(control_events.iterrows()):
        row = raw.to_dict()
        event = _event_id(row["event_id"])
        row["event_id"] = event
        canonical_row = event_rows.get(event)
        if canonical_row is not None:
            row.update(frame_id=canonical_row.frame_id, period_id=canonical_row.period_id,
                       object_id=canonical_row.object_id)
        if event in ambiguous_events:
            row["event_kind"] = "ambiguous_event"
        order = event_order.get(event)
        if order is None:
            prior, following = raw_orders.iloc[:source_index].dropna(), raw_orders.iloc[source_index + 1:].dropna()
            left = float(prior.iloc[-1]) if len(prior) else -1
            right = float(following.iloc[0]) if len(following) else len(canonical)
            order = (left + right) / 2
            # Canonical indexes can be out of timestamp order. Distinct frames
            # establish chronology; only a conflicting same-frame order is ambiguous.
            same_frame_orders = set(canonical.index[canonical.frame_id.eq(row.get("frame_id"))
                                     & canonical.period_id.eq(row.get("period_id"))])
            row["ordering_ambiguous"] = right <= left and left in same_frame_orders and right in same_frame_orders
        row["order"] = float(order)
        timeline.append(row)
    restart_points = {(int(r["period_id"]), _integer(r["frame_id"]), r.get("object_id")) for r in timeline
                      if r["event_kind"] == "restart" and _integer(r.get("period_id")) in (1, 2)}
    restart_points.update((_integer(r.period_id), _integer(r.frame_id), r.object_id)
                          for _, r in canonical.loc[canonical.spadl_type.isin(PASSING_RESTARTS)].iterrows())
    scheduled = set(control_rows)
    scheduled_receipts = set()
    for i, action in canonical.iterrows():
        event, kind = action["_event_key"], action.spadl_type
        source = control_rows.get(event)
        passing_restart = kind in PASSING_RESTARTS or (source is not None and source.event_kind == "restart"
                            and source.get("raw_tag") != "Penalty" and kind not in config.SHOT)
        if kind in ("pass", "cross", *config.SHOT, *PASSING_RESTARTS) and event not in scheduled:
            event_kind = ("ambiguous_event" if event in ambiguous_events else "restart" if passing_restart
                          else "shot" if kind in config.SHOT else kind)
            timeline.append(dict(event_id=event, event_kind=event_kind, raw_tag=PASSING_RESTARTS.get(kind),
                period_id=action.period_id, frame_id=action.frame_id, object_id=action.object_id, order=float(i)))
            scheduled.add(event)
        if (kind not in ("pass", "cross") and not passing_restart) or event in ambiguous_events or _truth(action.get("offside", False)):
            continue
        frame = _integer(action.get("receive_frame_id"))
        receiver = action.get("receiver_id")
        unidentified_receipt = pd.isna(receiver) and _truth(action.get("source_has_reception", False))
        if frame is None or (not _player(receiver) and not unidentified_receipt) or frame < (_integer(action.frame_id) or 0):
            continue
        receiver = receiver if _player(receiver) else None
        if (_integer(action.period_id), frame, receiver) in restart_points:
            continue
        receipt_key = (event, _integer(action.period_id), frame, receiver)
        if receipt_key in scheduled_receipts:
            continue
        scheduled_receipts.add(receipt_key)
        next_type = action.get("next_type")
        if next_type in ("bad_touch", "shot_block", "keeper_punch"):
            continue
        if next_type in ("interception", "keeper_save") and not unidentified_receipt:
            later = control_events.loc[(control_events.period_id == action.period_id) & (control_events.frame_id >= frame)]
            if later.empty:
                continue
            first = later.iloc[0]
            if not (first.event_kind == "claim" and first.get("object_id") == action.receiver_id):
                continue
        timeline.append(dict(event_id=f"receipt:{event}", parent_event_id=event, event_kind="receipt",
            period_id=action.period_id, frame_id=frame, object_id=receiver, order=float(i) + 0.25))
    timeline = [row for row in timeline if _integer(row.get("period_id")) in (1, 2)]
    for row in timeline:
        if _integer(row.get("frame_id")) is not None:
            continue
        # Anchor ordering only; this is not a substitute physical endpoint.
        earlier = [r for r in timeline if r["period_id"] == row["period_id"]
                   and r["order"] < row["order"] and _integer(r.get("frame_id")) is not None]
        later = [r for r in timeline if r["period_id"] == row["period_id"]
                 and r["order"] > row["order"] and _integer(r.get("frame_id")) is not None]
        neighbour = max(earlier, key=lambda r: r["order"]) if earlier else min(later, key=lambda r: r["order"]) if later else None
        row["frame_id"] = neighbour["frame_id"] if neighbour else 0
        row["event_kind"] = "missing_frame"
        plan.diagnostics.append(dict(event_id=row["event_id"], reason="missing_control_frame"))
    simultaneous = defaultdict(list)
    for row in timeline:
        simultaneous[(row["period_id"], row["frame_id"], row["order"])].append(row)
    def controller(row):
        if row["event_kind"] in ("claim", "other") and not _truth(row.get("clearance", False)):
            return row.get("object_id")
        if row["event_kind"] == "tackle":
            if _truth(row.get("possession_change", False)):
                return row.get("winner_id")
            return next((row.get(player) for player, role in (("winner_id", "winner_role"), ("loser_id", "loser_role"))
                         if row.get(role) == "withBallControl"), None)
        return None

    for rows in simultaneous.values():
        if len(rows) < 2:
            continue
        timestamps = [pd.to_datetime(row.get("utc_timestamp"), utc=True, errors="coerce") for row in rows]
        identity_fields = ("event_kind", "object_id", "raw_tag", "winner_id", "loser_id", "possession_change", "fouler_id", "fouled_id")
        equivalent = len({tuple(str(row.get(key)) for key in identity_fields) for row in rows}) == 1
        controllers = [controller(row) for row in rows]
        equivalent = equivalent or (all(_player(player) for player in controllers) and len(set(controllers)) == 1)
        if not equivalent and (any(pd.isna(t) for t in timestamps) or len(set(timestamps)) != len(rows)):
            for row in rows:
                row["ordering_ambiguous"] = True
        elif not equivalent:
            for rank, position in enumerate(sorted(range(len(rows)), key=lambda i: timestamps[i])):
                rows[position]["order"] += rank / (len(rows) + 1) / 1000
    timeline.sort(key=lambda row: (int(row["period_id"]), int(row["frame_id"]), row["order"]))
    current, previous_period = None, None
    terminal_to_spell = {}
    restart_spells = {}

    def start(row, actor, kind, *, known=True):
        nonlocal current
        period, frame, event = int(row["period_id"]), int(row["frame_id"]), row["event_id"]
        current = dict(spell_id=len(plan.spells) + 1, period_id=period, owner_id=actor if _player(actor) else None,
            start_frame=frame if known else None, end_frame=None, start_event_id=event if known else None,
            acquisition_order=row["order"],
            terminal_event_id=None, start_kind=kind, end_kind=None,
            start_state_id=plan.observe(period, frame, actor, "start", event) if known else None,
            end_state_id=None, supported_until=frame if known else None, break_after=False, reason=None)
        plan.spells.append(current)

    def finish(row, reason, *, endpoint=None, barrier=False):
        nonlocal current
        if current is None:
            if barrier and plan.spells:
                plan.spells[-1]["break_after"] = True
            return
        current.update(terminal_event_id=row["event_id"], end_kind=row["event_kind"], reason=reason, break_after=barrier)
        if endpoint is not None and _player(endpoint):
            frame = int(row["frame_id"])
            current.update(end_frame=frame, end_state_id=plan.observe(row["period_id"], frame, endpoint, "end", row["event_id"]))
            current["supported_until"] = frame if current["start_frame"] is not None else None
        terminal_to_spell[row["event_id"]] = current
        current = None

    for row in timeline:
        period, frame = int(row["period_id"]), int(row["frame_id"])
        kind, actor = row["event_kind"], row.get("object_id")
        if previous_period is not None and period != previous_period:
            if current is not None:
                finish(dict(row, event_id="period_boundary", event_kind="period_end"), "period_boundary", barrier=True)
            if plan.spells:
                plan.spells[-1]["break_after"] = True
        previous_period = period
        if row.get("ordering_ambiguous"):
            endpoint = None
            if current and kind == "tackle" and _truth(row.get("possession_change")) and row.get("loser_id") == current["owner_id"]:
                endpoint = row.get("winner_id")
            elif current and kind in ("claim", "other") and _player(actor) and actor[:4] != (current["owner_id"] or "")[:4]:
                endpoint = actor
            elif current and kind == "foul":
                if row.get("fouler_id") == current["owner_id"]:
                    endpoint = row.get("fouled_id")
                elif row.get("fouled_id") == current["owner_id"]:
                    endpoint = current["owner_id"]
            finish(row, "unresolved_event_order", endpoint=endpoint, barrier=True)
            plan.diagnostics.append(dict(event_id=row["event_id"], reason="unresolved_event_order"))
        carrier = current["owner_id"] if current else None
        if kind in ("missing_frame", "ambiguous_event"):
            reason = "missing_control_frame" if kind == "missing_frame" else "ambiguous_canonical_event"
            finish(row, reason, barrier=True)
            start(row, None, "unknown_acquisition", known=False)
            finish(row, reason, barrier=True)
            if kind == "ambiguous_event":
                plan.diagnostics.append(dict(event_id=row["event_id"], reason=reason))
        elif kind == "receipt":
            if current and carrier == actor:
                parent = row.get("parent_event_id")
                if (current["start_kind"] in ("claim", "tackle_win", "retained_tackle")
                        and parent in event_order and event_order[parent] <= current["acquisition_order"]):
                    old_start = plan.observations.get(current["start_state_id"])
                    if old_start and not any(s is not current and s["start_state_id"] == current["start_state_id"] for s in plan.spells):
                        old_start["roles"] = [r for r in old_start["roles"] if r != "start"]
                        if "control" not in old_start["roles"]:
                            old_start["roles"].append("control")
                    current.update(start_frame=frame, start_event_id=row["event_id"], start_kind="receipt",
                                   start_state_id=plan.observe(period, frame, actor, "start", row["event_id"]))
                current["supported_until"] = frame
                continue
            if current is not None:
                finish(row, "unobserved_release", barrier=True)
            start(row, actor, kind)
        elif kind == "restart":
            canonical_row = event_rows.get(row["event_id"])
            shot_restart = row.get("raw_tag") == "Penalty" or (canonical_row is not None and canonical_row.spadl_type in config.SHOT)
            tag = row.get("raw_tag") or (PASSING_RESTARTS.get(canonical_row.spadl_type) if canonical_row is not None else None)
            restart_key = (period, frame, actor, tag) if _player(actor) and tag is not None and not shot_restart else None
            duplicate = restart_spells.get(restart_key)
            if current is None and duplicate is not None and plan.spells[-1] is duplicate:
                terminal_to_spell[row["event_id"]] = duplicate
                plan.observe(period, frame, actor, "start", row["event_id"])
                plan.observe(period, frame, actor, "end", row["event_id"])
                plan.diagnostics.append(dict(event_id=row["event_id"], reason="duplicate_restart_representation"))
                continue
            finish(row, "release_before_restart")
            start(row, actor, "shot_restart" if shot_restart else "restart", known=not shot_restart)
            if restart_key is not None:
                restart_spells[restart_key] = current
            finish(row, "shot_specific_restart" if shot_restart else "restart", endpoint=None if shot_restart else actor,
                   barrier=bool(shot_restart and canonical_row is not None and _truth(canonical_row.get("success", False))))
        elif kind == "period_end":
            finish(row, "period_end", barrier=True)
            if plan.spells:
                plan.spells[-1]["break_after"] = True
        elif kind in ("pass", "cross", "shot"):
            if current is None or carrier != actor:
                finish(row, "unexpected_carrier", barrier=True)
                start(row, actor, "unknown_acquisition", known=False)
            canonical_row = event_rows.get(row["event_id"])
            goal = kind == "shot" and canonical_row is not None and _truth(canonical_row.get("success", False))
            finish(row, "release", endpoint=actor, barrier=goal)
        elif kind == "claim":
            if not _player(actor):
                finish(row, "unidentified_claim", barrier=True)
            elif current and actor == carrier:
                current["supported_until"] = frame
                plan.observe(period, frame, actor, "control", row["event_id"])
            else:
                finish(row, "ball_claim", endpoint=actor if carrier and actor[:4] != carrier[:4] else None,
                       barrier=bool(carrier and actor[:4] == carrier[:4]))
                start(row, actor, "claim")
        elif kind == "tackle":
            winner, loser = row.get("winner_id"), row.get("loser_id")
            changed = _truth(row.get("possession_change", False))
            retained = next((p for p, role in ((winner, row.get("winner_role")), (loser, row.get("loser_role")))
                             if _player(p) and role == "withBallControl"), None)
            if not changed and retained is not None:
                if current and retained == carrier:
                    current["supported_until"] = frame
                    plan.observe(period, frame, retained, "control", row["event_id"])
                elif current is None:
                    start(row, retained, "retained_tackle")
                else:
                    finish(row, "inconsistent_tackle", barrier=True)
                    start(row, retained, "retained_tackle")
            elif changed and _player(winner):
                finish(row, "tackle_loss" if loser == carrier else "ambiguous_tackle",
                       endpoint=winner if loser == carrier else None, barrier=bool(carrier and loser != carrier))
                start(row, winner, "tackle_win")
            else:
                finish(row, "ambiguous_tackle", barrier=True)
        elif kind == "foul":
            if current and row.get("fouler_id") == carrier:
                finish(row, "carrier_committed_foul", endpoint=row.get("fouled_id"))
            elif current and row.get("fouled_id") == carrier:
                finish(row, "carrier_fouled", endpoint=carrier)
            else:
                finish(row, "foul_without_carrier", barrier=True)
        elif kind == "other":
            canonical_row = event_rows.get(row["event_id"])
            deflection = canonical_row is not None and canonical_row.spadl_type in ("bad_touch", "shot_block", "keeper_punch")
            if deflection:
                # Retain the existing loss-state convention, without promoting a
                # canonical touch/block into an acquisition by the touching player.
                endpoint = actor if _player(actor) and carrier and actor[:4] != carrier[:4] else None
                finish(row, "deflection_without_control", endpoint=endpoint, barrier=endpoint is None)
                plan.diagnostics.append(dict(event_id=row["event_id"], reason="deflection_without_control"))
            elif _truth(row.get("clearance", False)):
                finish(row, "clearance", endpoint=actor if _player(actor) and carrier and actor[:4] != carrier[:4] else None)
            elif _player(actor):
                if current and carrier == actor:
                    current["supported_until"] = frame
                    plan.observe(period, frame, actor, "control", row["event_id"])
                else:
                    finish(row, "other_control", endpoint=actor if carrier and actor[:4] != carrier[:4] else None,
                           barrier=bool(carrier and actor[:4] == carrier[:4]))
                    start(row, actor, "other_control")
            else:
                finish(row, "unidentified_control", barrier=True)
    if current:
        current["reason"], current["break_after"] = "incomplete_end_of_data", True
    for i, spell in enumerate(plan.spells):
        following = plan.spells[i + 1] if i + 1 < len(plan.spells) else None
        linked = following is not None and following["period_id"] == spell["period_id"] and not spell["break_after"]
        spell["next_spell_id"] = following["spell_id"] if linked else None
        spell["next_state_id"] = following["start_state_id"] if linked else None
    for action in plan.actions:
        spell = terminal_to_spell.get(action["original_event_id"])
        if spell and spell["owner_id"] == action["owner_id"]:
            action.update(spell_id=spell["spell_id"], start_state_id=spell["start_state_id"],
                          end_state_id=spell["end_state_id"], next_state_id=spell["next_state_id"])
    return plan


def build_sportec_possessions(match, match_id: str, feature_root: Path, *, scope="actions", frames=1,
                              add_v_edge_features=False, add_relative_speed_edge_features=False, possession_ids=None,
                              add_position_direction_edge_features=True, add_relative_velocity_edge_features=True,
                              add_separation_rate_edge_features=True):
    """Yield physical partitions; semantic spell ownership is kept in the manifest."""
    if scope not in ("actions", "frames") or not isinstance(frames, int) or frames < 1:
        raise ValueError("Expected scope actions/frames and a positive frame stride")
    if "ball_accel" not in match.tracking.columns:
        match.tracking = preprocess.calc_physical_features(match.tracking, match.fps)
    options = dict(add_v_edge_features=add_v_edge_features, add_relative_speed_edge_features=add_relative_speed_edge_features,
                   add_position_direction_edge_features=add_position_direction_edge_features,
                   add_relative_velocity_edge_features=add_relative_velocity_edge_features,
                   add_separation_rate_edge_features=add_separation_rate_edge_features)
    plan = reconstruct_sportec_states(match, match_id, feature_root, graph_options=options)
    match.sportec_state_plan = plan
    report = dict(**plan.provenance, possessions={}, excluded_states=[], selected_states=0)
    match.possession_report, match._pc_match_cache = report, {}
    # Partition order comes from the complete plan, before frame selection.
    groups = sorted({(row["period_id"], row["possessor_id"]) for row in plan.observations.values()})
    exact_roles = {"start", "end", "canonical_pre", "canonical_post"}
    selected_ids = {key for key, row in plan.observations.items() if exact_roles.intersection(row["roles"])}
    if scope == "frames":
        period_frames = {int(period): rows.index.sort_values() for period, rows in match.tracking.groupby("period_id")}
        for spell in plan.spells:
            start, end = spell["start_frame"], spell["supported_until"]
            if start is None or end is None or not _player(spell["owner_id"]):
                continue
            available = period_frames.get(spell["period_id"], pd.Index([]))
            # The last corroborated frame is still an interior observation when
            # the release is unknown; a known loss/release keeps its exact role.
            right = "right" if spell["end_frame"] is None else "left"
            available = available[available.searchsorted(start, side="right"):available.searchsorted(end, side=right)]
            for frame in available[(available - start) % frames == 0]:
                selected_ids.add(plan.observe(spell["period_id"], frame, spell["owner_id"], "interior"))
    for partition_id, (period, carrier) in enumerate(groups, 1):
        if possession_ids is not None and partition_id not in possession_ids:
            continue
        tracking = match.tracking.loc[match.tracking.period_id == period]
        observations = sorted((row for key, row in plan.observations.items() if key in selected_ids
            and row["period_id"] == period and row["possessor_id"] == carrier), key=lambda row: row["state_frame_id"])
        cache_key = f"{plan.provenance['state_contract']}_{_digest([str(match_id), period, carrier, plan.provenance])}"
        actions, labels, graphs = [], [], []
        for observation in observations:
            frame = observation["state_frame_id"]
            required = ["ball_x", "ball_y", f"{carrier}_x", f"{carrier}_y"]
            reason = None
            if frame not in tracking.index:
                reason = "missing_tracking_frame_or_period"
            elif any(not _finite(tracking.loc[frame].get(column)) for column in required):
                reason = "missing_ball_or_possessor_tracking"
            graph = None if reason else construct_graph_for_frame(match, frame, carrier, tracking,
                        infer_node_feature_dim(extend=True), **options)
            if graph is None:
                reason = reason or "invalid_graph"
            if reason:
                observation["exclusion_reason"] = reason
                report["excluded_states"].append(dict(state_id=observation["state_id"], reason=reason))
                continue
            attackers, defenders = utils.find_active_players(match.tracking, frame, carrier[:4], include_goals=True)
            if carrier not in attackers:
                observation["exclusion_reason"] = "inactive_possessor"
                report["excluded_states"].append(dict(state_id=observation["state_id"], reason="inactive_possessor"))
                continue
            x, y = float(tracking.at[frame, f"{carrier}_x"]), float(tracking.at[frame, f"{carrier}_y"])
            action = dict(frame_id=frame, receive_frame_id=frame, object_id=carrier, period_id=period,
                action_type="pass", spadl_type="synthetic_frame", receiver_id=carrier, next_player_id=carrier,
                success=True, blocked=False, start_x=x, start_y=y, end_x=x, end_y=y,
                stats_perform_match_id=str(match_id), action_id=observation["state_id"], original_event_id=None,
                state_id=observation["state_id"], state_partition_id=partition_id, possessor_id=carrier,
                state_frame_id=frame, frame_role="|".join(observation["roles"]), state_contract=plan.provenance["state_contract"],
                pc_cache_match_id=cache_key)
            label = [0.0] * len(config.LABEL_COLUMNS)
            values = dict(action_index=frame, is_pass=1, n_players=len(attackers) + len(defenders),
                intent_index=attackers.index(carrier), receiver_index=attackers.index(carrier),
                start_x=x, start_y=y, end_x=x, end_y=y, intent_x=x, intent_y=y, is_real=1, success=1)
            for name, value in values.items():
                label[config.LABEL_INDEX[name]] = value
            actions.append(action)
            labels.append(label)
            graphs.append(graph)
        report["possessions"][str(partition_id)] = dict(period_id=period, possessor_id=carrier, selected_frames=len(actions))
        report["selected_states"] += len(actions)
        if not actions:
            continue
        state = copy.copy(match)
        state.match_id, state.event_index, state.pc_cache_match_id = str(match_id), partition_id, cache_key
        state.possession_provenance = report
        state.pc_generation_hint = f"Generate a new matching pc-xPass cache with --sportec-feature-run-id {Path(feature_root).name} --scope {scope}."
        state.actions = pd.DataFrame(actions).set_index("frame_id", drop=False)
        state.labels = torch.tensor(labels, dtype=torch.float64)
        state.graph_features_0, state.graph_features_1 = graphs, None
        state.graph_features_by_dir, state.graph_feature_action_indices_by_dir = {}, {}
        state.pc_state_descriptor = dict(match_id=str(match_id), possession_id=partition_id,
            state_partition_id=partition_id, possessor_id=carrier, period_id=period, **plan.provenance)
        yield state, report


def sportec_component_table(predictions: pd.DataFrame, state) -> pd.DataFrame:
    columns = ["stats_perform_match_id", "action_id", "original_event_id"] + [
        column for column in STATE_METADATA_COLUMNS if column in state.actions]
    return pd.concat([state.actions.loc[predictions.index, columns].reset_index(drop=True),
                      predictions.reset_index(drop=True)], axis=1)


def sportec_event_endpoint_view(component: pd.DataFrame, manifest=None) -> pd.DataFrame:
    """Resolve event roles; probabilities retain each physical state's team perspective."""
    if "state_id" not in component:
        return component
    if manifest is None or manifest.get("state_contract") not in {STATE_CONTRACT, VALIDATED_STATE_CONTRACT}:
        raise ValueError("Sportec state components require a matching possession manifest; regenerate the component run.")
    if component.state_id.duplicated().any():
        raise ValueError("Duplicate physical state predictions")
    if set(component.state_id) - {row["state_id"] for row in manifest["observations"]}:
        raise ValueError("Component state provenance differs from its possession manifest")
    lookup, rows = component.set_index("state_id", drop=False), []
    for action in manifest["actions"]:
        for reference, scope in (("start_state_id", START_SCOPE), ("end_state_id", "frame_id"), ("next_state_id", "receive_frame_id")):
            key = action[reference]
            if key is None or key not in lookup.index:
                continue
            row = lookup.loc[key].to_dict()
            actor = action["owner_id"]
            row.update(action_id=action["action_id"], original_event_id=action["original_event_id"], frame_scope=scope,
                       value_sign=1 if _player(actor) and actor[:4] == row["possessor_id"][:4] else -1)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(dict.fromkeys([*component.columns, "frame_scope", "value_sign"])))
