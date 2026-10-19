"""Inference semantics are deliberately independent of training carry eligibility."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from datatools import sportec_possessions as sp


def event(eid, frame, actor="home_1", receiver=None, receipt=None, **kwargs):
    return dict(original_event_id=eid, action_id=int(eid), period_id=1, frame_id=frame,
                receive_frame_id=receipt, object_id=actor, receiver_id=receiver,
                spadl_type="pass", next_type="pass", offside=False, success=True, **kwargs)


def control(eid, kind, frame, actor="home_1", **kwargs):
    return dict(event_id=str(eid), event_kind=kind, frame_id=frame, period_id=1, object_id=actor, **kwargs)


def fixture(monkeypatch, events=None, controls=None):
    events = events if events is not None else [event(1, 20, receiver="home_2", receipt=30),
        event(2, 40, "home_2"), event(3, 60, "home_3", "home_4", 70),
        event(4, 80, "home_4", "away_5", 90)]
    controls = controls if controls is not None else [control("claim", "claim", 10),
        control(1, "pass", 20), control(2, "pass", 40, "home_2"),
        control(3, "restart", 60, "home_3", raw_tag="CornerKick"), control(4, "pass", 80, "home_4")]
    tracking = pd.DataFrame(dict(period_id=1, ball_x=2., ball_y=3., ball_accel=0.), index=range(1, 151))
    for actor in ("home_1", "home_2", "home_3", "home_4", "away_5"):
        tracking[f"{actor}_x"], tracking[f"{actor}_y"] = 2., 3.
    match = SimpleNamespace(events=pd.DataFrame(events), actions=pd.DataFrame(events), tracking=tracking,
                            fps=25, lineup=pd.DataFrame())
    control_table = pd.DataFrame(controls)
    original = sp.reconstruct_sportec_states
    monkeypatch.setattr(sp, "reconstruct_sportec_states", lambda *a, **k: original(*a, control_events=control_table, **k))
    monkeypatch.setattr(sp, "construct_graph_for_frame", lambda *a, **k: SimpleNamespace(frame=a[1], carrier=a[2]))
    monkeypatch.setattr(sp.utils, "find_active_players", lambda tracking, frame, team, **k:
        ([p for p in ("home_1", "home_2", "home_3", "home_4", "away_5") if p.startswith(team)], []))
    return match, control_table


def reconstruct(match):
    return sp.reconstruct_sportec_states(match, "match", Path("feature"))


def test_model_graph_options_do_not_change_physical_ids_or_spell_associations(monkeypatch):
    match, _ = fixture(monkeypatch)
    basic = sp.reconstruct_sportec_states(match, "match", Path("feature"),
        graph_options=dict(add_v_edge_features=False, add_relative_speed_edge_features=False))
    extended = sp.reconstruct_sportec_states(match, "match", Path("feature"),
        graph_options=dict(add_v_edge_features=True, add_relative_speed_edge_features=True))
    assert basic.provenance == extended.provenance
    assert "graph_options" not in basic.provenance
    assert basic.provenance["physical_input_contract"] == "pc_xpass_physical_inputs_v1"
    assert basic.observations == extended.observations
    assert basic.spells == extended.spells
    assert basic.actions == extended.actions


def test_chain_restart_opponent_receipt_and_shared_roles(monkeypatch):
    match, _ = fixture(monkeypatch)
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(10, 20), (30, 40), (60, 60), (70, 80), (90, None)]
    assert plan.spells[1]["next_state_id"] == plan.spells[2]["start_state_id"]
    assert plan.spells[2]["start_state_id"] == plan.spells[2]["end_state_id"]
    assert plan.spells[3]["next_state_id"] == plan.spells[4]["start_state_id"]
    assert plan.spells[4]["owner_id"] == "away_5"
    assert plan.actions[2]["start_state_id"] == plan.actions[2]["end_state_id"]
    roles = plan.observations[plan.spells[2]["start_state_id"]]["roles"]
    assert {"start", "end", "canonical_pre"}.issubset(roles)


def test_orphan_pass_is_not_a_zero_length_carry_or_skipped_successor(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 10), event(2, 20), event(3, 40, "home_2")],
                       [control(1, "pass", 10), control(2, "pass", 20), control("claim", "claim", 30, "home_2"), control(3, "pass", 40, "home_2")])
    plan = reconstruct(match)
    assert plan.spells[0]["start_state_id"] is None
    assert plan.spells[0]["end_state_id"] is not None
    assert plan.spells[0]["next_state_id"] is None
    assert plan.spells[1]["next_state_id"] == plan.spells[2]["start_state_id"]


def test_ambiguous_ending_keeps_observations_without_inventing_interval(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 20, receiver="home_2", receipt=30)],
        [control("claim", "claim", 10), control("tackle", "tackle", 15)])
    plan = reconstruct(match)
    assert plan.spells[0]["start_frame"] == 10
    assert plan.spells[0]["end_state_id"] is None
    assert plan.spells[0]["supported_until"] == 10
    states = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames"))
    assert 11 not in {frame for state, _ in states for frame in state.actions.index}
    assert any(row["state_frame_id"] == 20 for row in plan.observations.values())


def test_unknown_release_samples_through_the_last_corroborated_control(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 40)], [control("claim", "claim", 10),
        control("touch", "other", 20), control("ambiguous", "tackle", 25), control(1, "pass", 40)])
    states = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames", frames=5))
    assert {frame for state, _ in states for frame in state.actions.index} == {10, 15, 20, 40}
    assert match.sportec_state_plan.spells[0]["end_state_id"] is None
    observation = next(o for o in match.sportec_state_plan.observations.values() if o["state_frame_id"] == 20)
    assert "interior" in observation["roles"] and "end" not in observation["roles"]


def test_loss_endpoint_is_winner_and_same_frame_players_are_distinct(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 30, "away_5")],
        [control("claim", "claim", 10), control("loss", "tackle", 20, winner_id="away_5", loser_id="home_1", possession_change=True), control(1, "pass", 30, "away_5")])
    plan = reconstruct(match)
    loss = plan.observations[plan.spells[0]["end_state_id"]]
    assert loss["possessor_id"] == "away_5"
    assert plan.spells[0]["owner_id"] == "home_1"
    assert plan.spells[0]["end_state_id"] == plan.spells[1]["start_state_id"]
    assert plan.observe(1, 20, "home_1", "canonical_pre") != loss["state_id"]


def test_distinct_frames_resolve_inverted_canonical_row_order(monkeypatch):
    # Sportec canonical row indexes need not follow synchronized frame order.
    # The unindexed claim/loss between these releases still has known chronology.
    events = [event(1, 50, "home_2"), event(2, 20)]
    controls = [control("initial", "claim", 10), control(2, "pass", 20),
        control("claim", "claim", 25),
        control("loss", "tackle", 30, winner_id="away_5", loser_id="home_1", possession_change=True),
        control(1, "pass", 50, "home_2")]
    match, _ = fixture(monkeypatch, events, controls)
    plan = reconstruct(match)
    spell = next(s for s in plan.spells if s["start_frame"] == 25)
    endpoint = plan.observations[spell["end_state_id"]]
    assert (endpoint["state_frame_id"], endpoint["possessor_id"]) == (30, "away_5")
    assert spell["next_state_id"] == spell["end_state_id"]
    assert not any(d["reason"] == "unresolved_event_order" for d in plan.diagnostics)


def test_receipt_then_claim_corroborates_without_reanchoring(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 20, receiver="away_5", receipt=30), event(2, 50, "away_5")],
        [control(1, "pass", 20), control("claim", "claim", 31, "away_5"), control(2, "pass", 50, "away_5")])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (30, 50)]


def test_period_and_goal_barriers(monkeypatch):
    events = [event(1, 20), event(2, 40, "home_2")]
    events[0]["spadl_type"] = "shot"
    match, _ = fixture(monkeypatch, events, [control(1, "shot", 20), control("restart", "restart", 30, "home_2"), control(2, "pass", 40, "home_2")])
    plan = reconstruct(match)
    assert plan.spells[0]["next_state_id"] is None
    match.events.loc[1, "period_id"] = 2
    match.actions.loc[1, "period_id"] = 2
    # The canonical row controls its raw event's period.
    plan = reconstruct(match)
    assert plan.spells[-2]["next_state_id"] is None


def test_exact_invalid_endpoint_remains_missing_and_scope_ids_are_stable(monkeypatch):
    match, _ = fixture(monkeypatch)
    match.tracking.loc[30, "home_2_x"] = float("nan")
    actions = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    missing = match.sportec_state_plan.spells[1]["start_state_id"]
    assert match.sportec_state_plan.observations[missing]["exclusion_reason"] == "missing_ball_or_possessor_tracking"
    assert match.sportec_state_plan.actions[0]["next_state_id"] == missing
    dense = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames", frames=3))
    keys = lambda states: {row.state_id for state, _ in states for _, row in state.actions.iterrows()}
    assert keys(actions).issubset(keys(dense))
    assert {s.pc_cache_match_id for s, _ in actions} == {s.pc_cache_match_id for s, _ in dense}
    assert missing not in keys(dense)


def test_cache_manifest_action_lookup_and_lane_identity(tmp_path, monkeypatch):
    import pc_xpass_match_cache as cache
    match, _ = fixture(monkeypatch)
    states = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    root = tmp_path / "sportec"
    cache.initialize(root)
    with cache.MatchAccumulator(root, "match") as transaction:
        for state, _ in states:
            transaction.register(state.pc_cache_match_id, state.pc_state_descriptor)
            frames = state.actions.index.tolist()
            rows = pd.DataFrame(dict(action_index=frames, state_frame_id=frames,
                physical_state_hash="hash", frame_scope="frame_id", home_1=.5))
            transaction.write(state.pc_cache_match_id, rows)
            transaction.write_lane(state.pc_cache_match_id, [], [dict(row, match_id=state.pc_cache_match_id) for row in rows.to_dict("records")])
        transaction.set_manifest(match.sportec_state_plan.manifest())
        transaction.set_provenance(match.possession_report)
        transaction.commit(export_lane=True)
    physical, _, record = cache.load_match(root, "match")
    assert set(physical["matches"].state_id) == {row.state_id for state, _ in states for _, row in state.actions.iterrows()}
    assert record["manifest"]["spells"][2]["start_frame"] == 60
    actions = cache.load_action_rows(root, "match", frame_scope="frame_id")
    assert actions.action_index.tolist() == [0, 1, 2, 3]
    assert actions.state_frame_id.tolist() == [20, 40, 60, 80]
    for state, _ in states:
        assert cache.load_possession(root, "match", cache.selection_for(state), lane=True,
                                     requested_frames=state.actions.index).empty


def test_match_without_valid_graphs_keeps_manifest_and_empty_canonical_view(tmp_path, monkeypatch):
    import pc_xpass_match_cache as cache
    match, _ = fixture(monkeypatch)
    monkeypatch.setattr(sp, "construct_graph_for_frame", lambda *a, **k: None)
    assert list(sp.build_sportec_possessions(match, "match", Path("feature"))) == []
    root = tmp_path / "sportec"
    cache.initialize(root)
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.set_manifest(match.sportec_state_plan.manifest())
        transaction.set_provenance(match.possession_report)
        transaction.commit(export_lane=True)
    frames, _, record = cache.load_match(root, "match")
    assert frames["matches"].empty
    assert record["manifest"]["spells"]
    assert cache.load_action_rows(root, "match").empty


def test_endpoint_export_uses_spell_links_and_team_perspective(monkeypatch):
    match, _ = fixture(monkeypatch)
    states = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames", frames=3))
    tables = [sp.sportec_component_table(pd.DataFrame({"home_1": .5}, index=s.actions.index), s) for s, _ in states]
    view = sp.sportec_event_endpoint_view(pd.concat(tables), match.sportec_state_plan.manifest())
    assert set(view.frame_scope) == {"frame_id", "receive_frame_id", "spell_start"}
    next_value = view.loc[view.action_id.eq(4) & view.frame_scope.eq("receive_frame_id")].iloc[0]
    assert next_value.value_sign == -1
    assert next_value.state_frame_id == 90
    corner = view.loc[view.action_id.eq(3)]
    assert corner.loc[corner.frame_scope.eq("spell_start"), "state_id"].iloc[0] == corner.loc[corner.frame_scope.eq("frame_id"), "state_id"].iloc[0]
    with pytest.raises(ValueError, match="manifest"):
        sp.sportec_event_endpoint_view(pd.concat(tables))


def test_reconstruction_does_not_change_training_carries(monkeypatch):
    from datatools.ball_carries import derive_carry_segments, CARRY_DEFINITION_VERSION
    match, controls = fixture(monkeypatch)
    for column in ("winner_id", "loser_id", "winner_role", "loser_role", "fouler_id", "fouled_id", "utc_timestamp"):
        if column not in controls:
            controls[column] = None
    controls["possession_change"], controls["clearance"] = False, False
    before = derive_carry_segments(controls, match.events, match.tracking, fps=5)
    original = copy.deepcopy(match.actions)
    reconstruct(match)
    after = derive_carry_segments(controls, match.events, match.tracking, fps=5)
    for left, right in zip(before, after):
        pd.testing.assert_frame_equal(left, right)
    pd.testing.assert_frame_equal(original, match.actions)
    assert CARRY_DEFINITION_VERSION == "sportec_fernandez_1s_v2"


def test_same_frame_receipt_precedes_its_release(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 20, receiver="home_2", receipt=30), event(2, 30, "home_2")],
                       [control(1, "pass", 20), control(2, "pass", 30, "home_2")])
    plan = reconstruct(match)
    assert plan.spells[1]["start_state_id"] == plan.spells[1]["end_state_id"]
    assert plan.spells[0]["next_state_id"] == plan.spells[1]["start_state_id"]


def test_receipt_at_restart_is_one_spell(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 20, receiver="home_2", receipt=40), event(2, 40, "home_2")],
                       [control(1, "pass", 20), control(2, "restart", 40, "home_2", raw_tag="CornerKick")])
    plan = reconstruct(match)
    assert len(plan.spells) == 2
    assert plan.spells[1]["start_frame"] == plan.spells[1]["end_frame"] == 40
    assert plan.spells[0]["next_state_id"] == plan.spells[1]["start_state_id"]


@pytest.mark.parametrize("kind", list(sp.PASSING_RESTARTS))
@pytest.mark.parametrize("with_control", [False, True])
def test_typed_passing_restart_keeps_its_following_receipt(monkeypatch, kind, with_control):
    events = [event(1, 20, receiver="home_2", receipt=40), event(2, 40, "home_2", "home_3", 50), event(3, 60, "home_3")]
    events[1]["spadl_type"] = kind
    controls = [control(1, "pass", 20), control(3, "pass", 60, "home_3")]
    if with_control:
        controls.insert(1, control(2, "restart", 40, "home_2", raw_tag=sp.PASSING_RESTARTS[kind]))
    match, _ = fixture(monkeypatch, events, controls)
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (40, 40), (50, 60)]
    assert plan.spells[0]["next_state_id"] == plan.spells[1]["start_state_id"]
    assert plan.spells[1]["next_state_id"] == plan.spells[2]["start_state_id"]


def test_duplicate_restart_representations_share_one_spell(monkeypatch):
    events = [event(1, 20, receiver="home_2", receipt=40), event(2, 40, "home_2", "home_3", 50), event(3, 60, "home_3")]
    events[1]["spadl_type"] = "corner_crossed"
    controls = [control(1, "pass", 20), control(2, "restart", 40, "home_2", raw_tag="CornerKick"),
                control("duplicate", "restart", 40, "home_2", raw_tag="CornerKick"), control(3, "pass", 60, "home_3")]
    match, _ = fixture(monkeypatch, events, controls)
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (40, 40), (50, 60)]
    assert any(d["reason"] == "duplicate_restart_representation" for d in plan.diagnostics)


def test_unrelated_simultaneous_claims_keep_states_and_break_uncertain_links(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 40, "home_2")], [control("initial", "claim", 10),
        control("a", "claim", 20, "home_2"), control("b", "claim", 20, "away_5"), control(1, "pass", 40, "home_2")])
    plan = reconstruct(match)
    assert {o["possessor_id"] for o in plan.observations.values() if o["state_frame_id"] == 20} == {"home_2", "away_5"}
    assert plan.spells[0]["next_state_id"] is None
    assert plan.actions[0]["start_state_id"] is None
    assert sum(d["reason"] == "unresolved_event_order" for d in plan.diagnostics) == 2


def test_simultaneous_touch_and_control_duel_corroborate_the_same_player(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 40, "away_5")], [control("initial", "claim", 10),
        control("touch", "other", 20, "away_5"),
        control("duel", "tackle", 20, winner_id="away_5", loser_id="home_1", winner_role="withBallControl", possession_change=False),
        control(1, "pass", 40, "away_5")])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(10, 20), (20, 40)]
    assert plan.spells[0]["end_state_id"] == plan.spells[1]["start_state_id"]
    assert plan.spells[0]["next_state_id"] == plan.spells[1]["start_state_id"]
    assert not any(d["reason"] == "unresolved_event_order" for d in plan.diagnostics)


def test_penalty_keeps_shot_specific_handling(monkeypatch):
    events = [event(1, 20)]
    events[0]["spadl_type"] = "shot_penalty"
    match, _ = fixture(monkeypatch, events, [control(1, "restart", 20, raw_tag="Penalty")])
    states = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    assert len(states) == 1  # Keep the canonical observation, without a passing valuation.
    assert "shot_specific" in states[0][0].actions.frame_role.iloc[0]
    assert match.sportec_state_plan.spells[0]["start_state_id"] is None
    assert match.sportec_state_plan.spells[0]["end_state_id"] is None
    assert match.sportec_state_plan.spells[0]["next_state_id"] is None


def test_linked_claim_before_receipt_keeps_canonical_acquisition(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 20, receiver="away_5", receipt=30), event(2, 50, "away_5")],
        [control(1, "pass", 20), control("claim", "claim", 29, "away_5"), control(2, "pass", 50, "away_5")])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (30, 50)]


def test_unknown_event_frame_blocks_successor_skipping(monkeypatch):
    events = [event(1, 10), event(2, None), event(3, 40, "home_2")]
    match, _ = fixture(monkeypatch, events, [control(1, "pass", 10), control(2, "pass", None),
        control("claim", "claim", 30, "home_2"), control(3, "pass", 40, "home_2")])
    plan = reconstruct(match)
    assert plan.spells[0]["next_state_id"] is None
    assert any(d["reason"] == "missing_control_frame" for d in plan.diagnostics)
    assert plan.actions[1]["pre_state_id"] is None


def test_training_only_columns_do_not_change_state_provenance(monkeypatch):
    match, _ = fixture(monkeypatch)
    original = reconstruct(match)
    match.actions["original_event_id"] = match.actions.original_event_id.astype("string")
    match.actions["carry_definition"] = "training_only"
    match.actions["return_event_index"] = 0
    match.events["scores"] = .9
    match.events["concedes"] = .1
    after = reconstruct(match)
    assert original.provenance == after.provenance
    assert original.observations.keys() == after.observations.keys()


def test_conflicting_canonical_identity_keeps_observations_without_links(monkeypatch):
    events = [event(1, 20), event(2, 20, "home_2")]
    events[1]["original_event_id"] = 1
    match, _ = fixture(monkeypatch, events, [control(1, "pass", 20)])
    plan = reconstruct(match)
    assert len(plan.observations) == 2
    assert all(a["start_state_id"] is None and a["next_state_id"] is None for a in plan.actions)
    assert plan.spells[0]["reason"] == "ambiguous_canonical_event"


def test_deflection_snapshot_does_not_create_a_possession(monkeypatch):
    events = [event(1, 20, receiver="away_5", receipt=30), event(2, 40, "home_2")]
    events[0]["next_type"] = "bad_touch"
    match, _ = fixture(monkeypatch, events, [control(1, "pass", 20), control(2, "restart", 40, "home_2", raw_tag="ThrowIn")])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (40, 40)]
    assert plan.actions[0]["post_state_id"] != plan.actions[0]["next_state_id"]
    assert any(r["state_frame_id"] == 30 for r in plan.observations.values())


@pytest.mark.parametrize("kind", ["bad_touch", "shot_block", "keeper_punch"])
def test_raw_deflection_event_is_an_observation_without_an_acquisition(monkeypatch, kind):
    events = [event(1, 20), event(2, 30, "away_5"), event(3, 40, "home_2")]
    events[1]["spadl_type"] = kind
    match, _ = fixture(monkeypatch, events, [control(1, "pass", 20),
        control(2, "other", 30, "away_5"), control(3, "restart", 40, "home_2", raw_tag="ThrowIn")])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(None, 20), (40, 40)]
    assert not any(s["owner_id"] == "away_5" for s in plan.spells)
    assert plan.actions[1]["pre_state_id"] in plan.observations


def test_derived_tracking_features_are_fingerprinted_before_selection(monkeypatch):
    match, _ = fixture(monkeypatch)
    match.tracking = match.tracking.drop(columns="ball_accel")
    monkeypatch.setattr(sp.preprocess, "calc_physical_features", lambda tracking, fps: tracking.assign(ball_accel=0.))
    list(sp.build_sportec_possessions(match, "match", Path("feature")))
    original = match.sportec_state_plan.manifest()
    list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames", frames=5))
    assert original["provenance"] == match.sportec_state_plan.provenance
    assert original["spells"] == match.sportec_state_plan.spells
    assert original["actions"] == match.sportec_state_plan.actions


def test_corroborating_control_observations_do_not_split_a_spell(monkeypatch):
    match, _ = fixture(monkeypatch, [event(1, 40)], [control("claim", "claim", 10),
        control("touch", "other", 20),
        control("duel", "tackle", 30, winner_id="home_1", winner_role="withBallControl", possession_change=False),
        control(1, "pass", 40)])
    plan = reconstruct(match)
    assert [(s["start_frame"], s["end_frame"]) for s in plan.spells] == [(10, 40)]
    assert {o["state_frame_id"] for o in plan.observations.values()} == {10, 20, 30, 40}
    assert all("control" in o["roles"] for o in plan.observations.values() if o["state_frame_id"] in (20, 30))
    states = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    assert {f for state, _ in states for f in state.actions.index} == {10, 40}


def test_scope_extension_and_height_refresh_keep_consistent_manifests(tmp_path, monkeypatch):
    import pc_xpass_match_cache as cache
    match, _ = fixture(monkeypatch)
    root = tmp_path / "sportec"
    cache.initialize(root)
    for scope in ("actions", "frames", "actions"):
        states = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope=scope, frames=5))
        with cache.MatchAccumulator(root, "match") as transaction:
            for state, _ in states:
                transaction.register(state.pc_cache_match_id, state.pc_state_descriptor)
                frames = state.actions.index.tolist()
                rows = pd.DataFrame(dict(action_index=frames, state_frame_id=frames,
                    physical_state_hash="hash", frame_scope="frame_id", home_1=.5))
                transaction.write(state.pc_cache_match_id, rows)
                transaction.write_lane(state.pc_cache_match_id, [], [dict(r, match_id=state.pc_cache_match_id) for r in rows.to_dict("records")])
            transaction.set_manifest(match.sportec_state_plan.manifest())
            transaction.set_provenance(match.possession_report)
            transaction.commit(export_lane=True)
        cache.load_match(root, "match")  # Both output headers must contain the same manifest.
    before = (root / "lane_control/match.parquet").read_bytes()
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.record_coverage = False
        transaction.set_manifest(match.sportec_state_plan.manifest())
        key = next(iter(transaction.parts))
        updated = transaction.read(key)
        updated["home_1__pass_height"] = .3
        transaction.write(key, updated)
        transaction.commit()
    cache.load_match(root, "match")
    assert (root / "lane_control/match.parquet").read_bytes() == before


def test_real_graphs_and_model_inference_use_the_selected_states(monkeypatch):
    from datatools import graph_feature, utils
    from inference import inference_gnn
    from scripts.generate_physical_xpass import runtime_cache_items_from_graphs
    from physical_pass_model import physical_state_hash
    from test_graph_feature_regressions import make_offside_match_and_snapshot, DummyNodeModel

    find_active_players = utils.find_active_players
    source, _ = fixture(monkeypatch, [event(1, 119, receiver="home_2", receipt=124)],
                          [control("claim", "claim", 102), control(1, "pass", 119)])
    match, snapshot = make_offside_match_and_snapshot()
    match.events, match.actions = source.events, source.actions
    match.max_players, match.fps = 24, 25
    match.tracking = pd.concat([snapshot] * 24, ignore_index=True)
    match.tracking.index = range(102, 126)
    match.tracking["period_id"], match.tracking["ball_accel"] = 1, 0.
    match.tracking["home_2_x"] = [65. + i / 10 for i in range(24)]
    monkeypatch.setattr(sp, "construct_graph_for_frame", graph_feature.construct_graph_for_frame)
    monkeypatch.setattr(utils, "find_active_players", find_active_players)
    state = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope="frames", frames=5))[0][0]
    assert state.actions.index.tolist() == [102, 107, 112, 117, 119]
    items = runtime_cache_items_from_graphs(state.pc_cache_match_id, state.graph_features_0, state.labels,
                                           frame_scope="frame_id", state_frame_ids=state.actions.index.tolist())
    assert items[0]["labels"].equal(state.labels)
    assert not torch.equal(state.graph_features_0[0].x, state.graph_features_0[1].x)
    model = DummyNodeModel("action_intent", [0.] * 1000, node_in_dim=graph_feature.infer_node_feature_dim(extend=True))
    model.args["extend_features"] = True
    predictions, _ = inference_gnn(state, model, device="cpu", graph_override=state.graph_features_0, label_override=state.labels)
    assert predictions.index.tolist() == state.actions.index.tolist()
    assert predictions.notna().any(axis=1).all()
    baseline = None
    for scope, stride in (("actions", 1), ("frames", 3), ("frames", 7)):
        states = list(sp.build_sportec_possessions(match, "match", Path("feature"), scope=scope, frames=stride))
        hashes = {row.state_id: physical_state_hash(graph) for partition, _ in states
                  for (_, row), graph in zip(partition.actions.iterrows(), partition.graph_features_0)}
        if baseline is None:
            baseline = hashes
            spells = copy.deepcopy(match.sportec_state_plan.spells)
        else:
            assert {key: hashes[key] for key in baseline} == baseline
            assert match.sportec_state_plan.spells == spells


def test_ranking_corner_credit_and_opponent_perspective(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation/bundesliga_ranking/code"))
    import bundesliga_ranking as ranking

    match, _ = fixture(monkeypatch)
    states = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    match_dir, events_dir = tmp_path / "match", tmp_path / "events"
    match_dir.mkdir()
    events_dir.mkdir()
    (match_dir / "possession_manifest.json").write_text(json.dumps(match.sportec_state_plan.manifest()), encoding="utf-8")
    actors = ["home_1", "home_2", "home_3", "home_4", "away_5"]
    for name in ranking.REQUIRED_SCOPED_COMPONENTS:
        tables = []
        for state, _ in states:
            rows = []
            for frame, row in state.actions.iterrows():
                owner = row.object_id
                if name == "pass_intent":
                    values = {p: float(p == owner) for p in actors}
                elif name == "pass_success":
                    values = dict.fromkeys(actors, 1.)
                elif name == "outcome_scoring_success":
                    values = {p: frame / 100 + (0 if p == owner else .1) for p in actors}
                else:
                    values = dict.fromkeys(actors, 0.)
                rows.append(values)
            tables.append(sp.sportec_component_table(pd.DataFrame(rows, index=state.actions.index), state))
        pd.concat(tables).to_parquet(match_dir / f"{name}.parquet", index=False)
    events = match.events.assign(stats_perform_match_id="match", seconds=match.events.frame_id / 25,
        player_id="player", player_name="name", advanced_position="midfielder", team_id="home")
    events.to_csv(events_dir / "match.csv", index=False)
    scored = ranking.add_scores_to_events(ranking.build_match_model_data(match_dir), events_dir).set_index("action_id")
    assert scored.loc[2, "game_state_value_next"] == pytest.approx(.6)
    assert scored.loc[3, "carry_epv"] == pytest.approx(0.)
    assert scored.loc[3, "game_state_value_next"] == pytest.approx(.7)
    assert scored.loc[4, "game_state_value_next"] == pytest.approx(-.9)
    assert scored.loc[4, "carry_epv"] == pytest.approx(.1)
    assert scored.loc[4, "pass_epv"] == pytest.approx(-1.7)
    assert scored.loc[4, "action_epv"] == pytest.approx(-1.6)


@pytest.mark.parametrize("use_carries", [False, True])
def test_visualization_and_generator_loaders_share_canonical_provenance(tmp_path, monkeypatch, use_carries):
    from scripts import run_relevant_models as runner, visualize_action_components as visualizer
    from project_config import get_action_graph_dir

    match, _ = fixture(monkeypatch)
    events = match.events.copy()
    lineup = pd.DataFrame({"stats_perform_match_id": ["match"]})
    resolved = match.actions.copy()
    resolved["xT"] = 0.9
    graph_dir = get_action_graph_dir(tmp_path, use_carries=use_carries)
    graph_dir.mkdir(parents=True)
    (graph_dir / "match.pt").touch()

    class LoadedMatch:
        def __init__(self, events, tracking, lineup, **kwargs):
            self.events, self.tracking, self.lineup = events.copy(), tracking.copy(), lineup.copy()

        def construct_labels(self, *, resolved_actions, **kwargs):
            # Both entrypoints must retain the unmodified source before constructing labels.
            assert self.sportec_canonical_source is events
            self.actions = resolved_actions.copy()
            return torch.empty((0, 0))

    def read_parquet(path):
        if "tracking_processed" in str(path):
            return match.tracking.copy()
        if Path(path).name == "line_up.parquet":
            return lineup.copy()
        return resolved.copy()

    for module in (runner, visualizer):
        monkeypatch.setattr(module, "Match", LoadedMatch)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: events)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr(torch, "load", lambda *a, **k: [])
    monkeypatch.setattr(runner, "get_resolved_action_path", lambda *a, **k: graph_dir / "match.pt")
    monkeypatch.setattr(visualizer, "get_resolved_action_path", lambda *a, **k: graph_dir / "match.pt")
    generated = runner.load_match("match", feature_root=tmp_path, use_carries=use_carries, load_endpoint_graphs=False)
    visualized = visualizer.load_match("match", feature_root=tmp_path, use_carries=use_carries)
    assert reconstruct(generated).provenance == reconstruct(visualized).provenance



def test_comparison_reuses_states_without_narrowing_the_cached_snapshot(monkeypatch):
    from scripts.visualize_action_components import possession_state_for_action

    match, _ = fixture(monkeypatch)
    match.match_id = "match"
    states = list(sp.build_sportec_possessions(match, "match", Path("feature")))
    original = {state.pc_cache_match_id: state.actions.copy(deep=True) for state, _ in states}
    monkeypatch.setattr(sp, "build_sportec_possessions", lambda *a, **k: pytest.fail("reconstructed twice"))
    for action_index in (0, 1, 0):
        state, frame = possession_state_for_action(match, action_index, Path("feature"), states)
        assert state.actions.index.tolist() == [frame]
        assert len(state.labels) == len(state.graph_features_0) == 1
        state.actions.loc[frame, "state_id"] = "local change"
    for state, _ in states:
        pd.testing.assert_frame_equal(state.actions, original[state.pc_cache_match_id])
