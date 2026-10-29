from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from datatools.ball_carries import derive_carry_segments
from datatools.carry_policy import CarryPolicy, DEFINITION, reconstruct_spells, run_policy
from datatools.sportec_possessions import reconstruct_sportec_states


def fixture():
    tracking = pd.DataFrame(dict(period_id=1, ball_x=0., ball_y=0., ball_z=0., ball_state="alive"), index=range(151))
    for actor in ["home_1", "home_2", "away_3"]:
        tracking[actor+"_x"] = 0.
        tracking[actor+"_y"] = 0.
    events = pd.DataFrame([
        dict(original_event_id="1", action_id=1, period_id=1, frame_id=10, receive_frame_id=20,
             object_id="home_1", receiver_id="home_2", spadl_type="pass", offside=False, success=True),
        dict(original_event_id="2", action_id=2, period_id=1, frame_id=83, receive_frame_id=np.nan,
             object_id="home_2", receiver_id=None, spadl_type="pass", offside=False, success=True)])
    controls = pd.DataFrame([
        dict(event_id="1", event_kind="pass", frame_id=10, period_id=1, object_id="home_1"),
        dict(event_id="2", event_kind="pass", frame_id=83, period_id=1, object_id="home_2")])
    return events, controls, tracking


def add(controls, kind, frame, actor="home_2", **values):
    return pd.concat([controls, pd.DataFrame([dict(event_id="extra", event_kind=kind, frame_id=frame,
                                                  period_id=1, object_id=actor, **values)])], ignore_index=True)


def test_segments_and_return_horizons():
    events, controls, tracking = fixture()
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.duration.tolist() == [1., 1.52]
    assert segments.success.tolist() == [True, True]
    assert segments.return_event_index.tolist() == [1, 1]
    assert audit.retained.sum() == 1
    failed = add(controls.iloc[:1], "tackle", 83, "away_3", winner_id="away_3", loser_id="home_2", possession_change=True)
    segments, _ = derive_carry_segments(failed, events.assign(frame_id=[10, 100], object_id=["home_1", "away_3"]), tracking)
    assert segments.success.tolist() == [True, False]


@pytest.mark.parametrize("subtype,keep", [("InterceptedBall", True), ("BallClaimed", True), ("BallHeld", False), ("Block", False)])
def test_claim_types_and_companion_receipt(subtype, keep):
    events, controls, tracking = fixture()
    controls = add(controls, "claim", 20, claim_type=subtype)
    segments, _ = derive_carry_segments(controls, events, tracking)
    assert bool(len(segments)) == keep


def test_hold_does_not_reappear_as_later_receipt_or_tackle():
    events, controls, tracking = fixture()
    controls = add(controls, "claim", 19, claim_type="BallHeld")
    controls = add(controls, "tackle", 21, winner_id="home_2", loser_id="away_3",
                   winner_role="withBallControl", possession_change=False)
    segments, _ = derive_carry_segments(controls, events, tracking)
    assert segments.empty


def test_same_frame_release_tackle_cannot_start_carry():
    events, controls, tracking = fixture()
    events.at[0, "receive_frame_id"] = np.nan
    controls = add(controls, "tackle", 10, "home_1", winner_id="home_1", loser_id="away_3",
                   winner_role="withBallControl", possession_change=False)
    audit = reconstruct_spells(controls, events, tracking)
    assert audit.empty


def test_claim_tackle_receipt_do_not_split_or_reset_acquisition():
    events, controls, tracking = fixture()
    controls = add(controls, "claim", 20, claim_type="BallClaimed")
    controls = add(controls, "tackle", 20, winner_id="home_2", loser_id="away_3", possession_change=True)
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert audit.retained.sum() == 1
    assert segments.frame_id.tolist() == [20, 45]


def test_generic_touch_invalidates_control_without_creating_acquisition():
    events, controls, tracking = fixture()
    controls = add(controls, "other", 40, "away_3")
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.empty
    assert "other_ball_action" not in audit.start_kind.tolist()
    assert "uncorroborated_other" in audit.reason.tolist()


@pytest.mark.parametrize("fault,reason", [("gap", "tracking_gap"), ("missing", "missing_spell_tracking"),
    ("dead", "dead_ball_interval"), ("period", "period_boundary"), ("distance", "sustained_separation")])
def test_tracking_guards_reject_complete_spell(fault, reason):
    events, controls, tracking = fixture()
    if fault == "gap":
        tracking = tracking.drop(40)
    elif fault == "missing":
        tracking.at[40, "home_2_x"] = np.nan
    elif fault == "dead":
        tracking.at[40, "ball_state"] = "dead"
    elif fault == "period":
        tracking.loc[40:, "period_id"] = 2
    else:
        tracking.loc[30:54, "home_2_x"] = 8.
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.empty
    assert reason in audit.reason.tolist()


def test_brief_separation_and_terminal_loss_distance_are_allowed():
    events, controls, tracking = fixture()
    tracking.loc[30:45, "home_2_x"] = 8.
    tracking.loc[79:83, "home_2_x"] = 20.
    segments, _ = derive_carry_segments(controls, events, tracking)
    assert len(segments) == 2


@pytest.mark.parametrize("source", ["kpi_csv", "elastic"])
def test_reception_validation_is_independent_of_source(source):
    events, controls, tracking = fixture()
    events["endpoint_source"] = source
    accepted, _ = derive_carry_segments(controls, events, tracking)
    assert len(accepted) == 2
    tracking.at[20, "home_2_x"] = 8.
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.empty
    assert "acquisition_distance" in audit.reason.tolist()


@pytest.mark.parametrize("committed,success", [(True, False), (False, True)])
def test_foul_terminal_semantics(committed, success):
    events, controls, tracking = fixture()
    events.at[1, "spadl_type"] = "foul"
    controls.at[1, "event_kind"] = "foul"
    controls.loc[1, "fouler_id"] = "home_2" if committed else "away_3"
    controls.loc[1, "fouled_id"] = "away_3" if committed else "home_2"
    segments, _ = derive_carry_segments(controls, events, tracking)
    assert segments.success.tolist() == [True, success]


def test_reversed_receipt_cannot_create_spell():
    events, controls, tracking = fixture()
    events.at[0, "receive_frame_id"] = 9
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.empty
    assert not audit.start_kind.eq("receipt").any()


def test_missing_control_frame_prevents_bridging():
    events, controls, tracking = fixture()
    controls["utc_timestamp"] = pd.to_datetime(["2020-01-01T00:00:10", "2020-01-01T00:00:20"])
    controls = add(controls, "other", np.nan, utc_timestamp=pd.Timestamp("2020-01-01T00:00:15"))
    segments, audit = derive_carry_segments(controls, events, tracking)
    assert segments.empty
    assert "missing_control_frame" in audit.reason.tolist()


def test_inference_rejects_interval_preserving_canonical_observations(tmp_path):
    events, controls, tracking = fixture()
    tracking.loc[30:54, "home_2_x"] = 10.
    (tmp_path / "metadata.json").write_text(json.dumps(dict(carry_variant=dict(
        definition=DEFINITION, validation_parameters=asdict(CarryPolicy())))))
    match = SimpleNamespace(events=events, actions=events, tracking=tracking, fps=25, lineup=pd.DataFrame())
    plan = reconstruct_sportec_states(match, "test", tmp_path, control_events=controls)
    from datatools.carry_policy import STATE_CONTRACT
    assert plan.provenance["state_contract"] == STATE_CONTRACT
    assert all(s["start_state_id"] is None and s["supported_until"] is None for s in plan.spells)
    assert plan.actions[-1]["pre_state_id"] in plan.observations
    assert plan.actions[-1]["start_state_id"] is None
    assert set(events.original_event_id) == {a["original_event_id"] for a in plan.actions}


def test_unknown_definition_fails_instead_of_silent_upgrade(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(dict(carry_variant=dict(definition="unknown"))))
    with pytest.raises(ValueError, match="Unsupported"):
        run_policy(tmp_path)


def test_unknown_release_between_intervals_breaks_successor_chain(tmp_path):
    events, controls, tracking = fixture()
    events = pd.concat([events, pd.DataFrame([dict(original_event_id="3", action_id=3, period_id=1,
        frame_id=120, receive_frame_id=np.nan, object_id="away_3", receiver_id=None,
        spadl_type="pass", offside=False, success=True)])], ignore_index=True)
    controls = add(controls, "pass", 90, "home_1")
    controls = add(controls, "claim", 95, "away_3", claim_type="BallClaimed")
    controls = pd.concat([controls, pd.DataFrame([dict(event_id="3", event_kind="pass", frame_id=120,
        period_id=1, object_id="away_3")])], ignore_index=True)
    (tmp_path / "metadata.json").write_text(json.dumps(dict(carry_variant=dict(definition=DEFINITION))))
    match = SimpleNamespace(events=events, actions=events, tracking=tracking, fps=25, lineup=pd.DataFrame())
    # Distinct IDs are required for the intentionally separate controls.
    controls.loc[controls.event_id.eq("extra"), "event_id"] = ["orphan", "claim"]
    plan = reconstruct_sportec_states(match, "test", tmp_path, control_events=controls)
    assert len(plan.spells) == 2
    assert plan.spells[0]["next_state_id"] is None
