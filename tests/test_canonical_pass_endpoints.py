from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from datatools.pass_trajectory import PASS_DURATION, PassEndpointResolver, load_official_events
from datatools.reception_policy import RETIRED, apply_policy
from datatools.carry_policy import reconstruct_spells


def test_period_mapping_regression_and_possession_end_provenance(tmp_path):
    xml = tmp_path / "events.xml"
    xml.write_text('<Root><Event EventId="launch"><Play /></Event>'
                   '<Event EventId="contact"><OtherBallAction /></Event></Root>')
    files = SimpleNamespace(event_path=xml, kpi_path=tmp_path / "kpi.csv")
    raw = pd.DataFrame(dict(event_id=["launch", "contact"], period_id=[2, 2]))
    kpi = pd.DataFrame(dict(EVENT_ID=["launch", "contact"], FRAME_NUMBER=[118358, 118990],
                            RECFRM=[118376, 119000], endpoint_source=["reception", "possession_end"]))
    mapping = pd.DataFrame(dict(period_id=[1, 2, 2, 2, 2],
        raw_frame_id=[118376, 118358, 118376, 118990, 119000],
        frame_id=[118376, 118496, 118514, 119128, 119138]))
    with patch("datatools.pass_trajectory._match_files", return_value={"DFL-MAT-test": files}), patch(
            "scripts.preprocess_sportec.load_match_raw_events", return_value=raw), patch(
            "scripts.preprocess_sportec.load_kpi_merged_table", return_value=kpi), patch(
            "pandas.read_parquet", return_value=mapping):
        events = load_official_events("DFL-MAT-test")
    assert events.loc[0, "frame_id"] == 118496
    assert events.loc[0, "reception_frame_id"] == 118514
    assert pd.isna(events.loc[1, "reception_frame_id"])
    tracking = pd.DataFrame(dict(period_id=2, ball_x=1., ball_y=2.), index=range(118496, 119140))
    endpoint = PassEndpointResolver(tracking, events=events).resolve(
        dict(frame_id=118496, period_id=2, original_event_id="launch"))
    assert endpoint.duration == .72
    assert endpoint.observed == 118514
    assert endpoint.trajectory == 118513  # Failed receipt/contact frame is contaminated.


def case():
    tracking = pd.DataFrame(dict(period_id=1, ball_x=np.arange(101), ball_y=0.), index=range(100, 201))
    actions = pd.DataFrame([dict(frame_id=100, receive_frame_id=180, endpoint_source="elastic",
                                action_type="pass", original_event_id="launch", period_id=1)])
    events = pd.DataFrame(dict(event_id=["launch", "contact"], frame_id=[100, 119],
                               raw_tag=["Play", "OtherBallAction"], period_id=[1, 1]))
    return actions, tracking, events


def test_authoritative_endpoint_and_retirement_are_idempotent():
    actions, tracking, events = case()
    resolver = PassEndpointResolver(tracking, events=events)
    retired = apply_policy(actions, RETIRED)
    first = resolver.annotate(retired)
    second = resolver.annotate(apply_policy(first, RETIRED))
    assert second.loc[0, "receive_frame_id"] == second.loc[0, "observed_end_frame_id"] == 119
    assert second.loc[0, "end_x"] == 19
    assert second.loc[0, "retired_receive_frame_id"] == 180
    pd.testing.assert_frame_equal(first, second)


def test_missing_intervening_contact_does_not_jump_to_later_event():
    actions, tracking, events = case()
    unknown = events.iloc[[1]].assign(event_id="unknown", frame_id=np.nan)
    events = pd.concat([events.iloc[:1], unknown, events.iloc[1:]], ignore_index=True)
    resolver = PassEndpointResolver(tracking, events=events)
    assert resolver.resolve(actions.iloc[0]).reason == "missing_intervening_official_frame"
    later = actions.iloc[0].copy()
    later["original_event_id"], later["frame_id"] = "unknown", 110
    assert resolver.resolve(later).observed == 119


@pytest.mark.parametrize("source,expected", [("official_reception", True), ("official_contact", False),
    ("official_stoppage", False), ("known_out_boundary", False)])
def test_only_reception_endpoint_can_create_receipt_evidence(source, expected):
    tracking = pd.DataFrame(dict(period_id=1, ball_x=0., ball_y=0., ball_z=0., ball_state="alive",
                                 home_1_x=0., home_1_y=0., home_2_x=0., home_2_y=0.), index=range(101))
    actions = pd.DataFrame([dict(original_event_id="launch", period_id=1, frame_id=10,
        receive_frame_id=20, object_id="home_1", receiver_id="home_2", spadl_type="pass",
        pass_endpoint_source=source)])
    controls = pd.DataFrame([dict(event_id="launch", event_kind="pass", frame_id=10, period_id=1,
        object_id="home_1"), dict(event_id="release", event_kind="pass", frame_id=80, period_id=1,
        object_id="home_2")])
    spells = reconstruct_spells(controls, actions, tracking)
    assert (not spells.empty) == expected


def test_refresh_disables_model_artifacts_until_checkpoint_extension(tmp_path, monkeypatch):
    import json
    from datatools import feature_refresh as refresh
    from datatools import reception_refresh
    from scripts import replace_carries
    from project_config import assert_feature_paths_usable
    source, output = tmp_path / "runs" / "source", tmp_path / "runs" / "derived"
    source.mkdir(parents=True)
    metadata = dict(status="completed", intended_receiver_modes=["original", "angle_only", "model"],
                    intended_receiver_model_id="old", return_types=["next_10"])
    (source / "metadata.json").write_text(json.dumps(metadata))
    for folder in ["resolved_actions_model", "action_labels_next_10_model", "resolved_actions_original"]:
        (source / folder).mkdir()
        (source / folder / "m.parquet").write_bytes(b"source")
    monkeypatch.setattr(reception_refresh, "resolve_source", lambda _: (source, metadata))
    monkeypatch.setattr(replace_carries, "match_ids", lambda *args: ["m"])
    monkeypatch.setattr(refresh, "get_feature_run_root", lambda _: output)
    monkeypatch.setattr(refresh, "write_latest_run", lambda *args: None)
    def worker(task):
        report = dict(match_id="m", skipped_actions=[], artifacts={"resolved_actions_original/m.parquet": 6})
        refresh.atomic_write_json(output / "feature_refresh" / "m.json", report)
        return report
    monkeypatch.setattr(refresh, "_refresh_match", worker)
    args = SimpleNamespace(extend_feature_run_id="source", run_id="derived", replace_carries=True,
        retire_elastic=True, artifact_copy_mode="hardlink", num_workers="1", worker_thread_limit=1)
    refresh.run_combined_refresh(args)
    result = json.loads((output / "metadata.json").read_text())
    assert result["intended_receiver_modes"] == ["original", "angle_only"]
    assert result["previous_intended_receiver_model_id"] == "old"
    assert result["model_mode_regeneration_required"]
    assert not (output / "resolved_actions_model").exists()
    with pytest.raises(ValueError, match="requires regeneration"):
        assert_feature_paths_usable(output / "action_labels_next_10_model")
    assert (source / "resolved_actions_model/m.parquet").read_bytes() == b"source"


def test_pinned_endpoints_do_not_require_provider_files():
    from datatools.pass_trajectory import ensure_pass_endpoints
    actions, tracking, events = case()
    actions = PassEndpointResolver(tracking, events=events).annotate(actions)
    match = SimpleNamespace(actions=actions, tracking=tracking, fps=25, match_id="DFL-MAT-test")
    with patch("datatools.pass_trajectory.load_official_events", side_effect=AssertionError("provider access")):
        resolver = ensure_pass_endpoints(match)
        assert resolver.resolve(actions.iloc[0]).observed == 119


def test_new_checkpoint_extension_restores_model_mode_without_replacement():
    from test_feature_run_extension import FeatureRunExtensionPlanTests, make_args, make_metadata
    from scripts import generate_relevant_features as generator
    metadata = make_metadata()
    metadata.update(model_mode_regeneration_required=True, previous_intended_receiver_model_id="old",
                    carry_variant=dict(available=True))
    args = make_args(model_id="success_intent/new")
    plan = FeatureRunExtensionPlanTests().build_plan(args, metadata)
    assert plan.added_intended_receiver_modes == ["model"]
    assert plan.intended_receiver_model_id == "success_intent/new"
    assert not plan.regenerate_model_mode
    final = generator.derived_metadata(args, plan, "completed")
    assert final["model_mode_regeneration_required"] is False
    assert final["previous_intended_receiver_model_id"] == "old"
    steps = generator.carry_extension_steps(args, dict(final, intended_receiver_modes=["model"]), "derived")
    assert len(steps) == 2
    for step in steps:
        assert "--use-carries" in step.command
        assert step.command[step.command.index("--intended-receiver-model-id") + 1] == "success_intent/new"


def test_duration_rejection_preserves_candidate_after_retirement_and_reannotation():
    actions, tracking, events = case()
    resolver = PassEndpointResolver(tracking, events=events, max_pass_duration=.5)
    first = resolver.annotate(apply_policy(actions, RETIRED))
    second = resolver.annotate(apply_policy(first, RETIRED))
    assert pd.isna(second.loc[0, "receive_frame_id"])
    assert second.loc[0, "candidate_end_frame_id"] == 119
    assert second.loc[0, "candidate_pass_duration"] == .76
    assert second.loc[0, "pass_endpoint_reason"] == "duration_limit_exceeded"


def test_pinned_overlay_uses_event_identity_and_preserves_nonpass_rows(tmp_path):
    from datatools.pass_trajectory import apply_pinned_endpoints
    actions, tracking, events = case()
    pinned = PassEndpointResolver(tracking, events=events).annotate(actions)
    (tmp_path / "pass_endpoints").mkdir()
    pinned.to_parquet(tmp_path / "pass_endpoints/m.parquet")
    source = pd.concat([actions, actions.assign(original_event_id="shot", action_type="shot")],ignore_index=True)
    source.index = [45, 89]
    output = apply_pinned_endpoints(source, tmp_path, "m")
    assert output.index.tolist() == [45, 89]
    assert output.at[45, "receive_frame_id"] == 119
    assert output.at[89, "receive_frame_id"] == 180
