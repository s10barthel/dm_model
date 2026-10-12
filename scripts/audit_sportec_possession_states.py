"""Read-only comparison of reconstructed Sportec states with an old pc-xPass cache."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from datatools.ball_carries import derive_carry_segments
from datatools.sportec_possessions import build_sportec_possessions
from physical_pass_model import physical_state_hash
from project_config import (CARRY_SEGMENTS_DIR, CONTROL_EVENTS_SYNCED_DIR, DATA_ROOT,
                            resolve_feature_root)
from scripts.run_relevant_models import load_match


def audit_match(match_id, feature_root, old_root):
    match = load_match(match_id, feature_root=feature_root, load_endpoint_graphs=False)
    old = pd.read_parquet(old_root / "sportec" / "matches" / f"{match_id}.parquet",
                          columns=["action_index", "frame_scope", "state_frame_id", "physical_state_hash"])
    hashes = set()
    state_hashes = {}
    evaluated_state_ids = set()
    state_count = 0
    for state, _ in build_sportec_possessions(match, match_id, feature_root):
        state_hashes.update({key: physical_state_hash(graph) for key, graph in
                             zip(state.actions.state_id, state.graph_features_0)})
        hashes.update(state_hashes.values())
        evaluated_state_ids.update(state.actions.state_id)
        state_count += len(state.actions)
    plan = match.sportec_state_plan
    by_event = {a["action_index"]: a for a in plan.actions}
    association_matches = 0
    association_differences = []
    exclusion_by_hash = {}
    for scope in ("frame_id", "receive_frame_id"):
        for row in old.loc[old.frame_scope.eq(scope)].itertuples():
            action = by_event.get(row.action_index)
            reference = action.get("pre_state_id" if scope == "frame_id" else "post_state_id") if action else None
            observation = plan.observations.get(reference, {})
            if state_hashes.get(reference) == row.physical_state_hash:
                association_matches += 1
            else:
                association_differences.append(dict(action_index=int(row.action_index), frame_scope=scope,
                    canonical_state_id=reference, represented_elsewhere=row.physical_state_hash in hashes,
                    reason=observation.get("exclusion_reason", "canonical_graph_association_differs")))
            reason = observation.get("exclusion_reason")
            if reason:
                exclusion_by_hash[row.physical_state_hash] = reason
    unresolved = old.loc[~old.physical_state_hash.isin(hashes)].copy()
    unresolved["reason"] = unresolved.physical_state_hash.map(exclusion_by_hash).fillna("unexplained_state_difference")
    # Successors are compared separately from physical observation coverage.
    changed = 0
    reasons = Counter()
    successor_changes = []
    for action in plan.actions:
        if action["post_state_id"] != action["next_state_id"]:
            changed += 1
            spell = next((s for s in plan.spells if s["spell_id"] == action["spell_id"]), None)
            if action["next_state_id"] is not None:
                reason = "next_genuine_acquisition_differs_from_legacy_post_target"
            elif spell and spell["break_after"]:
                reason = "chain_boundary_or_uncertain_continuity"
            else:
                reason = "unknown_or_absent_successor_start"
            reasons[reason] += 1
            old_state = plan.observations.get(action["post_state_id"], {})
            new_state = plan.observations.get(action["next_state_id"], {})
            successor_changes.append(dict(match_id=match_id, action_index=action["action_index"],
                original_event_id=action["original_event_id"], reason=reason,
                old_frame=old_state.get("state_frame_id"), old_possessor=old_state.get("possessor_id"),
                next_frame=new_state.get("state_frame_id"), next_possessor=new_state.get("possessor_id")))
    canonical = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv")
    controls = pd.read_parquet(CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet")
    carries, _ = derive_carry_segments(controls, canonical, match.tracking, fps=match.fps)
    saved = pd.read_parquet(CARRY_SEGMENTS_DIR / f"{match_id}.parquet")
    pd.testing.assert_frame_equal(carries, saved, check_dtype=False, check_flags=False)
    terminal_segments = saved.sort_values("segment_id").groupby("carry_id").tail(1)
    losses = terminal_segments.loc[terminal_segments.success.eq(False)]
    spell_lookup = {(s["period_id"], s["terminal_event_id"], s["owner_id"]): s for s in plan.spells}
    loss_matches, loss_differences = 0, []
    for segment in losses.itertuples():
        spell = spell_lookup.get((segment.period_id, str(segment.terminal_event_id).removesuffix(".0"), segment.object_id))
        if spell is None:
            related = [s for s in plan.spells if s["period_id"] == segment.period_id
                       and s["start_frame"] is not None
                       and segment.frame_id <= s["start_frame"] <= segment.receive_frame_id]
            target = next((o for o in plan.observations.values()
                           if o["period_id"] == segment.period_id and o["state_frame_id"] == segment.receive_frame_id
                           and o["possessor_id"] == segment.receiver_id), {})
            loss_differences.append(dict(carry_id=int(segment.carry_id), terminal_event_id=str(segment.terminal_event_id),
                owner_id=segment.object_id, training_start_frame=int(segment.frame_id),
                training_endpoint_frame=int(segment.receive_frame_id), training_endpoint_player=segment.receiver_id,
                target_observation_id=target.get("state_id"), target_exclusion_reason=target.get("exclusion_reason"),
                target_evaluated=target.get("state_id") in evaluated_state_ids,
                inference_spells=[{k: s[k] for k in ("spell_id", "owner_id", "start_frame", "end_frame", "reason")}
                                  for s in related]))
            continue
        endpoint = plan.observations.get(spell["end_state_id"], {})
        if endpoint.get("state_frame_id") != segment.receive_frame_id or endpoint.get("possessor_id") != segment.receiver_id:
            raise AssertionError(f"Training/inference loss endpoint differs: {match_id}/{segment.terminal_event_id}")
        loss_matches += 1
    result = dict(match_id=match_id, old_rows=len(old), old_unique_states=old.physical_state_hash.nunique(),
        represented_old_rows=int(old.physical_state_hash.isin(hashes).sum()), new_states=state_count,
        canonical_association_graph_matches=association_matches,
        canonical_association_graph_differences=association_differences,
        spells=len(plan.spells), exclusions=dict(Counter(unresolved.reason)),
        changed_successor_associations=changed, successor_difference_reasons=dict(reasons),
        training_carry_segments_unchanged=len(carries), matching_training_loss_endpoints=loss_matches,
        training_losses_without_corresponding_inference_spell=len(loss_differences),
        training_loss_interval_differences=loss_differences)
    return result, unresolved, successor_changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--old-pc-xpass-id", required=True)
    parser.add_argument("--match-id", nargs="+", default=[f"DFL-MAT-J03WE{suffix}" for suffix in "LMNOPQ"])
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON; exclusion and successor-change CSVs are written alongside it.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    feature_root = resolve_feature_root(args.feature_run_id)
    old_root = DATA_ROOT / "pc_xpass" / args.old_pc_xpass_id
    report, exclusions, successors = [], [], []
    for match_id in args.match_id:
        result, missing, changes = audit_match(match_id, feature_root, old_root)
        report.append(result)
        missing["match_id"] = match_id
        exclusions.append(missing)
        successors.extend(changes)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.concat(exclusions, ignore_index=True).to_csv(args.output.with_suffix(".csv"), index=False)
    pd.DataFrame(successors).to_csv(args.output.with_name(args.output.stem + "_successors.csv"), index=False)
    if any(r["exclusions"].get("unexplained_state_difference") for r in report):
        raise SystemExit("Unexplained old-state coverage differences; inspect the exclusion CSV.")


if __name__ == "__main__":
    main()
