"""Read existing Sportec control spells and build hypothetical passing states.

No control spells or training features are generated here. Both the inference
runner and pc-xPass generator use this adapter.
"""
from __future__ import annotations

import copy
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch

from datatools import config, preprocess, utils
from datatools.ball_carries import CARRY_DEFINITION_VERSION
from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
from datatools.possession_frames import STATE_CONTRACT, possession_cache_identity, select_possession_frames
from project_config import CARRY_SEGMENTS_DIR, load_run_metadata


STATE_METADATA_COLUMNS = [
    "possession_id", "possessor_id", "period_id", "state_frame_id", "frame_role",
    "original_start_frame", "original_end_frame", "start_event_id", "pc_cache_match_id",
    "start_action_id", "start_original_event_id", "end_action_id", "end_original_event_id",
]


def _endpoint_event_link(events, frame, carrier, period, *, receipt):
    """Only link unambiguous canonical events; never infer a link by proximity."""
    frame_column, player_column = ("receive_frame_id", "receiver_id") if receipt else ("frame_id", "object_id")
    required = {frame_column, player_column, "period_id", "action_id", "original_event_id"}
    if events is None or not required.issubset(events.columns):
        return None, None
    candidates = events.loc[
        events[frame_column].eq(frame) & events[player_column].eq(carrier) & events.period_id.eq(period)
    ]
    if receipt and "spadl_type" in candidates.columns:
        candidates = candidates.loc[candidates.spadl_type.isin(["pass", "cross"])]
    if len(candidates) != 1:
        return None, None
    row = candidates.iloc[0]
    return row.action_id, row.original_event_id


def sportec_event_endpoint_view(component: pd.DataFrame) -> pd.DataFrame:
    """Compatibility view for event-based scoring, without averaging frame states.

    Start predictions belong to the preceding pass's receipt; end predictions
    belong to the carrier's terminal action. Claims/tackles with no unambiguous
    canonical event link remain in frame exports, not fabricated event rows.
    """
    if "possession_id" not in component.columns:
        return component
    parts = []
    for endpoint, scope in (("start", "receive_frame_id"), ("end", "frame_id")):
        action_column, event_column = f"{endpoint}_action_id", f"{endpoint}_original_event_id"
        selected = component.loc[component.frame_role.isin([endpoint, "start_end"])].copy()
        selected = selected.dropna(subset=[action_column, event_column])
        selected["action_id"] = selected[action_column]
        selected["original_event_id"] = selected[event_column]
        selected["frame_scope"] = scope
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    key = ["stats_perform_match_id", "action_id", "original_event_id", "frame_scope"]
    if result.duplicated(key).any():
        raise ValueError("Multiple possession endpoints map to the same canonical event; inspect spell artifacts.")
    return result


def load_control_spells(match_id: str, feature_root: Path, *, artifact_path: Path | None = None):
    path = artifact_path or CARRY_SEGMENTS_DIR / "audits" / f"{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing whole-spell artifact {path}. Run Sportec carry preprocessing; one-second segments are insufficient.")
    artifact_bytes = path.read_bytes()
    spells = pd.read_parquet(BytesIO(artifact_bytes))
    required = {"carry_id", "period_id", "carrier_id", "start_frame", "terminal_frame", "success", "reason"}
    missing = required.difference(spells.columns)
    if missing:
        raise ValueError(f"Incompatible control-spell artifact {path}: missing {sorted(missing)}. Rebuild Sportec preprocessing.")
    if spells.carry_id.duplicated().any():
        raise ValueError(f"Duplicate control-spell IDs in {path}.")
    metadata = load_run_metadata(Path(feature_root), required=False) or {}
    definition = (metadata.get("carry_variant") or {}).get("definition")
    if definition is not None and definition != CARRY_DEFINITION_VERSION:
        raise ValueError("Feature-run carry definition differs from the current control-spell definition; use matching artifacts.")
    provenance = {
        "state_contract": STATE_CONTRACT,
        "feature_run_id": Path(feature_root).name,
        "carry_definition": definition or CARRY_DEFINITION_VERSION,
        "spell_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    # 'retained' is a training decision: short spells and missing return labels
    # must not disappear from state inference. Null success marks ambiguity.
    numeric = spells[["start_frame", "terminal_frame", "period_id", "carry_id"]].apply(pd.to_numeric, errors="coerce")
    integral = numeric.notna().all(axis=1) & numeric.eq(numeric.round()).all(axis=1)
    valid = integral & spells.success.isin([True, False]) & (numeric.terminal_frame >= numeric.start_frame)
    valid &= numeric.start_frame.ge(0) & numeric.period_id.isin([1, 2])
    valid &= spells.carrier_id.astype(str).str.match(r"^(home|away)_")
    report = {"artifact": str(path), **provenance, "spells": len(spells), "excluded_spells": int((~valid).sum())}
    return spells.loc[valid].sort_values(["period_id", "start_frame", "carry_id"]), report


def build_sportec_possessions(match, match_id: str, feature_root: Path, *, scope="actions", frames=1,
                              add_v_edge_features=False, add_relative_speed_edge_features=False):
    spells, report = load_control_spells(match_id, feature_root)
    report["possessions"] = {}
    match.possession_report = report
    if "ball_accel" not in match.tracking.columns:
        match.tracking = preprocess.calc_physical_features(match.tracking, match.fps)
    for _, spell in spells.iterrows():
        spell_id = int(spell.carry_id)
        start, end, period = int(spell.start_frame), int(spell.terminal_frame), int(spell.period_id)
        carrier = str(spell.carrier_id)
        start_action, start_event = _endpoint_event_link(getattr(match, "events", None), start, carrier, period, receipt=True)
        end_action, end_event = _endpoint_event_link(getattr(match, "events", None), end, carrier, period, receipt=False)
        tracking = match.tracking.loc[match.tracking.period_id == period]
        available = tracking.index[(tracking.index >= start) & (tracking.index <= end)]

        def evaluate(frame):
            row = tracking.loc[frame]
            if any(pd.isna(row.get(column)) for column in ["ball_x", "ball_y", f"{carrier}_x", f"{carrier}_y"]):
                return None
            graph = construct_graph_for_frame(
                match, frame, carrier, tracking, infer_node_feature_dim(extend=True),
                add_v_edge_features=add_v_edge_features,
                add_relative_speed_edge_features=add_relative_speed_edge_features,
            )
            if graph is None:
                return None
            attackers, defenders = utils.find_active_players(match.tracking, frame, carrier[:4], include_goals=True)
            if carrier not in attackers:
                return None
            x, y = float(row[f"{carrier}_x"]), float(row[f"{carrier}_y"])
            action = {
                "frame_id": frame, "receive_frame_id": frame, "object_id": carrier,
                "period_id": period, "action_type": "pass", "spadl_type": "synthetic_frame",
                "receiver_id": carrier, "next_player_id": carrier, "success": True, "blocked": False,
                "start_x": x, "start_y": y, "end_x": x, "end_y": y,
                "stats_perform_match_id": str(match_id), "action_id": f"spell:{spell_id}:frame:{frame}",
                "original_event_id": spell.get("terminal_event_id"),
                "possession_id": spell_id, "possessor_id": carrier, "state_frame_id": frame,
                "original_start_frame": start, "original_end_frame": end,
                "start_event_id": spell.get("start_event_id"),
                "start_action_id": start_action, "start_original_event_id": start_event,
                "end_action_id": end_action, "end_original_event_id": end_event,
            }
            label = [0.0] * len(config.LABEL_COLUMNS)
            values = {"action_index": frame, "is_pass": 1, "n_players": len(attackers) + len(defenders),
                      "intent_index": attackers.index(carrier), "receiver_index": attackers.index(carrier),
                      "start_x": x, "start_y": y, "end_x": x, "end_y": y, "intent_x": x, "intent_y": y,
                      "is_real": 1, "success": 1}
            for name, value in values.items():
                label[config.LABEL_INDEX[name]] = value
            return action, label, graph

        selected, selection = select_possession_frames(available, start, end, evaluate, scope=scope, frames=frames)
        report["possessions"][str(spell_id)] = selection
        if not selected:
            continue
        state = copy.copy(match)
        state.match_id = str(match_id)
        state.event_index = spell_id
        state.pc_cache_match_id = possession_cache_identity("sportec", match_id, spell_id, carrier, {
            key: report[key] for key in ("feature_run_id", "carry_definition", "spell_artifact_sha256")
        })
        state.possession_provenance = report
        state.pc_generation_hint = (
            "Generate matching states with scripts/generate_physical_xpass.py --pc-xpass "
            f"--sportec-feature-run-id {Path(feature_root).name} --match-id {match_id} --scope {scope}"
            + (f" --frames {frames}" if scope == "frames" else "")
            + "; select the same pc-xPass version for generation and inference."
        )
        actions, labels, graphs = [], [], []
        for frame, (action, label, graph), role in selected:
            action["frame_role"] = role
            action["pc_cache_match_id"] = state.pc_cache_match_id
            actions.append(action)
            labels.append(label)
            graphs.append(graph)
        state.actions = pd.DataFrame(actions).set_index("frame_id", drop=False)
        state.labels = torch.tensor(labels, dtype=torch.float64)
        state.graph_features_0 = graphs
        state.graph_features_1 = None
        state.graph_features_by_dir = {}
        state.graph_feature_action_indices_by_dir = {}
        yield state, report


def sportec_component_table(predictions: pd.DataFrame, state) -> pd.DataFrame:
    columns = ["stats_perform_match_id", "action_id", "original_event_id", *STATE_METADATA_COLUMNS]
    return pd.concat([state.actions.loc[predictions.index, columns].reset_index(drop=True),
                      predictions.reset_index(drop=True)], axis=1)
