"""Unsupported carry candidates stay missing through visualization and export."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from models import utils as model_utils
from scripts import inference_visualization_comparison as comparison
from scripts import run_and_visualize_hawkeye as hawkeye
from scripts import visualize_action_components as sportec
from scripts.comparison_warnings import ComparisonCancelled
from test_visualization_comparison import (
    TinyModel, _record, _subject, inference_fixture, outcome_pipeline,
)


def records_for(settings):
    records = {task: _record(f"{task}/test") for task in settings}
    for task, value in settings.items():
        records[task]["use_carries"] = value
        records[task]["args"]["use_carries"] = value
    return records


def test_pass_only_success_exception_preserves_sources_and_bundle(monkeypatch):
    settings = dict(pass_success=False, outcome_scoring=True, outcome_conceding=True, action_intent=True)
    records = records_for(settings)
    with pytest.raises(ValueError, match="do not agree on use_carries"):
        model_utils.validate_model_record_consistency(records)
    ids = {task: record["model_id"] for task, record in records.items()}
    bundle = dict(model_ids=ids, use_carries=True)
    monkeypatch.setattr(model_utils, "resolve_bundle_model_ids", lambda *a, **kw: (ids, bundle))
    monkeypatch.setattr(model_utils, "get_model_records", lambda *a: records)
    selected, context, _ = model_utils.resolve_model_selection(
        list(records), bundle_id="carry_bundle", allow_pass_only_success=True,
    )
    assert selected == ids
    assert context["use_carries"] is True
    assert context["source_use_carries"] == settings
    assert records["pass_success"]["use_carries"] is False


@pytest.mark.parametrize("settings", [
    dict(pass_success=True, outcome_scoring=False, outcome_conceding=True),
    dict(pass_success=False, action_intent=False, outcome_scoring=True, outcome_conceding=True),
    dict(pass_success=False, outcome_scoring=False, outcome_conceding=True),
])
def test_other_carry_mismatches_remain_strict(settings):
    with pytest.raises(ValueError, match="do not agree on use_carries"):
        model_utils.validate_model_record_consistency(records_for(settings), allow_pass_only_success=True)


@pytest.mark.parametrize("use_carries", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
def test_hawkeye_masks_each_frames_possessor_before_deriving(monkeypatch, use_carries, cache_enabled):
    subject = _subject()
    subject.frame_meta["possessor_object_id"] = ["home_1", "home_2"]
    # Both observations remain synthetic pass frames; carrier identity drives masking.
    subject.actions = pd.DataFrame(dict(object_id=["home_1", "home_2"], action_type=["pass", "pass"]), index=[10, 11])
    raw = pd.DataFrame({"home_2": [.8, .6], "away_3": [.1, .1], "home_1": [.6, .8]}, index=[11, 10])
    original = raw.copy(deep=True)
    models = {task: TinyModel(f"{task}/test") for task in ("pass_success", "outcome_scoring", "outcome_conceding")}
    models["pass_success"].args["use_carries"] = use_carries
    def predict(scene, model, **kwargs):
        if model.args["task"] == "pass_success":
            return raw, raw + .05
        return raw * .1, raw * .5
    predict = Mock(side_effect=predict)
    monkeypatch.setattr(hawkeye, "inference_gnn", predict)
    cache = {} if cache_enabled else None
    for _ in range(2):
        tables = hawkeye.collect_situation_components(
            subject, models, "cpu", [10, 11], ["pass_score", "model_pass_success"], prediction_cache=cache,
        )
        for frame, carrier, teammate in ((10, "home_1", "home_2"), (11, "home_2", "home_1")):
            for component in ("pass_success", "model_pass_success"):
                value = tables[component].at[frame, carrier]
                if use_carries:
                    assert value == pytest.approx(.8 + (.05 if component == "model_pass_success" else 0))
                else:
                    assert pd.isna(value)
                assert tables[component].at[frame, teammate] == pytest.approx(.6 + (.05 if component == "model_pass_success" else 0))
            assert tables["outcome_scoring_success"].at[frame, carrier] == pytest.approx(.4)
            assert tables["outcome_conceding_failure"].at[frame, carrier] == pytest.approx(.08)
            if use_carries:
                assert tables["pass_score"].at[frame, carrier] == 0
            else:
                assert pd.isna(tables["pass_score"].at[frame, carrier])
        pd.testing.assert_frame_equal(raw, original)
    assert predict.call_count == (3 if cache_enabled else 6)


@pytest.mark.parametrize("use_carries", [False, True])
@pytest.mark.parametrize("possession_input", [False, True])
def test_sportec_masks_only_carrier_with_recorded_or_possession_inputs(tmp_path, monkeypatch, use_carries, possession_input):
    subject = SimpleNamespace(actions=pd.DataFrame(dict(object_id=["home_2"], action_type=["pass"]), index=[7]))
    state = SimpleNamespace(graph_features_0=[1], labels=np.zeros((1, 1)))
    monkeypatch.setattr(sportec, "possession_state_for_action", lambda *a: (state, 77))
    models = {task: SimpleNamespace(args=dict(task=task, use_carries=True))
              for task in ("pass_success", "outcome_scoring", "outcome_conceding")}
    models["pass_success"].args.update(use_carries=use_carries, pc_xpass=possession_input)
    raw = pd.DataFrame({"home_1": [.6], "home_2": [.8]}, index=[7])
    def predict(scene, model, **kwargs):
        result = raw.copy()
        result.index = kwargs["event_indices"]
        if model.args["task"] == "pass_success":
            assert scene is (state if possession_input else subject)
            return result, result + .05
        return result * .1, result * .5
    monkeypatch.setattr(sportec, "inference_gnn", predict)
    rows = sportec.collect_action_components(subject, models, tmp_path, "cpu", 7, "event", tmp_path,
                                             rendered_components=["pass_score", "model_pass_success"])
    for name in ("pass_success", "model_pass_success", "pass_score"):
        assert pd.notna(rows[name]["home_1"])
        assert pd.notna(rows[name]["home_2"]) == use_carries
    assert rows["outcome_scoring_success"]["home_2"] == pytest.approx(.4)
    assert raw.at[7, "home_2"] == .8


def set_comparison_carries(monkeypatch, script, *, both_pass_only=False):
    def record(identifier):
        item = _record(identifier)
        value = not (identifier.startswith("pass_success/") and (both_pass_only or identifier.endswith("/a")))
        item["use_carries"] = item["args"]["use_carries"] = value
        return item
    monkeypatch.setattr(comparison, "get_model_record", record)
    def load(identifier, device):
        model = TinyModel(identifier)
        model.args["use_carries"] = record(identifier)["use_carries"]
        return model
    monkeypatch.setattr(comparison, "load_model", load)
    def predict(scene, model, **kwargs):
        values = pd.DataFrame({"home_1": [.8], "home_2": [.6]}, index=[10])
        if model.args["task"] == "pass_success":
            return values, values + .05 if kwargs.get("return_model_pass_success") else None
        factor = 1 if model.args["task"] == "outcome_scoring" else .2
        return values * .1 * factor, values * .5 * factor
    monkeypatch.setattr(script, "inference_gnn", predict)


def comparison_flags(required, tmp_path):
    return [*required, "--compare", "--pass-success-model-id", "pass_success/a",
            "--pass-success-model-id", "pass_success/b", "--outcome-scoring-model-id", "outcome_scoring/fixed",
            "--outcome-conceding-model-id", "outcome_conceding/fixed", "--only-pass-score",
            "--show-model-pass-success", "--output-dir", str(tmp_path), "--run-id", "carries"]


def test_comparison_preserves_unavailable_carry_values_and_provenance(tmp_path, monkeypatch, outcome_pipeline):
    script, required, _, _, build, render = outcome_pipeline
    set_comparison_carries(monkeypatch, script)
    args = script.parse_args([*comparison_flags(required, tmp_path), "--accept-comparison-warnings"])
    comparison.run_comparison(args, script)
    root = tmp_path / "carries"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert any("do not agree on use_carries" in message and "Ball-carrier" in message for message in metadata["warnings"])
    assert metadata["warning_acceptances"][0]["method"] == "command_line"
    assert metadata["variants"][0]["provenance"]["source_context"]["source_use_carries"]["pass_success"] is False
    if script is sportec:
        assert all(call.kwargs["use_carries"] for call in build.call_args_list)
    exported = pd.read_csv(root / "aligned_predictions.csv")
    unavailable = exported.component.isin(["pass_success", "model_pass_success", "pass_score"]) & exported.candidate_id.eq("home_1")
    assert unavailable.sum() == 3
    assert exported.loc[unavailable, "prediction:pass_success/a"].isna().all()
    assert exported.loc[unavailable, "prediction:pass_success/b"].notna().all()
    assert exported.loc[exported.candidate_id.eq("home_2"), "prediction:pass_success/a"].notna().all()
    outcomes = exported.component.str.startswith("outcome_") & exported.candidate_id.eq("home_1")
    assert exported.loc[outcomes, "prediction:pass_success/a"].notna().all()
    native = pd.read_parquet(metadata["variants"][0]["predictions"])
    masked_native = native.component.isin(["pass_success", "model_pass_success", "pass_score"]) & native.candidate_id.eq("home_1")
    assert not native.loc[masked_native, "available"].any()
    assert "N/A" in (root / "aligned_predictions.csv").read_text()


def test_within_pipeline_carry_warning_still_requires_acceptance(tmp_path, monkeypatch, outcome_pipeline):
    script, required, _, _, build, render = outcome_pipeline
    set_comparison_carries(monkeypatch, script, both_pass_only=True)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    args = script.parse_args(comparison_flags(required, tmp_path))
    with pytest.raises(ComparisonCancelled, match="explicit acceptance"):
        comparison.run_comparison(args, script)
    assert not build.call_count and not render.call_count
    metadata = json.loads((tmp_path / "carries/metadata.json").read_text())
    assert metadata["status"] == "cancelled"
    assert any("Within variant" in message and "pass_success=False" in message for message in metadata["warnings"])


@pytest.mark.parametrize("all_pass_only", [False, True])
def test_ordinary_visualizations_warn_and_record_carry_availability(tmp_path, monkeypatch, outcome_pipeline, all_pass_only):
    script, required, _, _, build, render = outcome_pipeline
    if script is sportec:
        monkeypatch.setattr(script, "resolve_runtime_feature_run_context", lambda *a, **kw: dict(
            feature_root=tmp_path, feature_run_id="feature", intended_receiver_mode="predict", selection="explicit"))
    settings = dict(pass_success=False, outcome_scoring=not all_pass_only, outcome_conceding=not all_pass_only)
    records = records_for(settings)
    identifiers = {task: record["model_id"] for task, record in records.items()}
    def resolve(required_tasks, **kwargs):
        assert set(required_tasks) == set(settings)
        assert kwargs["allow_pass_only_success"] is True
        return identifiers, model_utils.validate_model_record_consistency(
            records, allow_pass_only_success=True), None
    monkeypatch.setattr(script, "resolve_model_selection", resolve)
    def load(identifier, device):
        model = TinyModel(identifier)
        model.args["use_carries"] = settings[model.args["task"]]
        return model
    monkeypatch.setattr(script, "load_model", load)
    def predict(scene, model, **kwargs):
        values = pd.DataFrame({"home_1": [.8], "home_2": [.6]}, index=[10])
        return (values * .1, values * .5) if model.args["task"].startswith("outcome_") else (values, None)
    monkeypatch.setattr(script, "inference_gnn", predict)
    args = script.parse_args([*required, "--only-pass-score", "--output-dir", str(tmp_path), "--run-id", "ordinary"])
    for task, identifier in identifiers.items():
        setattr(args, f"{task}_model_id", identifier)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    with pytest.warns(UserWarning, match="Ball-carrier"):
        script.main()
    metadata = json.loads((tmp_path / "ordinary/metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["source_use_carries"] == settings
    assert any("Ball-carrier" in message for message in metadata["warnings"])
    if script is sportec:
        assert build.call_args.kwargs["use_carries"] == (not all_pass_only)
    assert render.called
    probabilities = render.call_args.kwargs["probs"] if script is sportec else render.call_args.args[3]
    assert pd.isna(probabilities["home_1"])
    assert pd.notna(probabilities["home_2"])


@pytest.mark.parametrize("component", ["pass_success", "model_pass_success", "pass_score"])
def test_all_unavailable_candidates_render_as_unavailable(tmp_path, component):
    from scripts.visualization_comparison import ComparisonExport, Variant, render_independent_groups
    args = SimpleNamespace(accept_comparison_warnings=True)
    report = ComparisonExport(tmp_path, [Variant("pass_only", "Pass only")], args, "checkpoints")
    render = Mock(side_effect=AssertionError("All-missing predictions should use an unavailable panel"))
    scene = dict(scene="s", frame=10, folder="s", suffix="", title="scene", render=render,
                 tables=[{component: pd.DataFrame({"home_1": [float("nan")]}, index=[10])}])
    render_independent_groups(report, [[scene]], [component])
    report.finish()
    assert not render.call_count
    with Image.open(tmp_path / "s" / f"{component}.png") as image:
        assert image.width > 0
    assert not pd.read_parquet(report.metadata["variants"][0]["predictions"])["available"].any()
