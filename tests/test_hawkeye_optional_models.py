"""Optional Hawkeye components and checkpoint-free physical pass success."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import inference
from datatools import config, hawkeye
from models import utils as model_utils
from pc_xpass_inference_policy import (
    PCXPassBlendingError, finalize_pc_xpass_attempt, inference_attempt, start_inference_attempt,
)
from scripts import run_hawkeye
from scripts.comparison_warnings import ComparisonCancelled
from scripts.hawkeye_model_warnings import HAWKEYE_MODEL_TASKS, confirm_missing_hawkeye_models
from test_hawkeye_target_clamping import make_ball, make_tracking
from test_pc_xpass_inference_policy import fixture as pc_fixture
from test_physical_xpass import DummyBaselineInferenceModel
from test_saved_visualization_xpass import write_cache


def omission_args(**changes):
    return SimpleNamespace(use_physical_xpass=False, xpass_weight="v0", pc_xpass=True, **changes)


@pytest.mark.parametrize("answer", ["no", "n", EOFError(), KeyboardInterrupt()])
def test_omission_cancel_records_decision(tmp_path, monkeypatch, answer):
    def respond(*args):
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr("builtins.input", respond)
    with pytest.raises((ComparisonCancelled, KeyboardInterrupt)):
        confirm_missing_hawkeye_models({}, omission_args(), tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "cancelled"
    assert metadata["model_selection"]["omitted_models"] == list(HAWKEYE_MODEL_TASKS)
    assert not metadata["model_selection"]["warning_acceptances"]
    assert not list(tmp_path.glob("*.csv"))


def test_omission_requires_explicit_answer_even_with_comparison_acceptance(tmp_path, monkeypatch, capsys):
    answers = iter(["", "maybe", "yes"])
    prompts = []
    def respond(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr("builtins.input", respond)
    selection = confirm_missing_hawkeye_models({}, omission_args(accept_comparison_warnings=True), tmp_path)
    assert len(prompts) == 3
    assert selection["warning_acceptances"][0]["method"] == "interactive"
    warning = capsys.readouterr().err
    assert all(task in warning for task in HAWKEYE_MODEL_TASKS)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["model_selection"] == selection


def test_complete_selection_does_not_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("unexpected prompt"))
    ids = {task: f"{task}/selected" for task in HAWKEYE_MODEL_TASKS}
    assert not confirm_missing_hawkeye_models(ids, omission_args(), tmp_path)["omitted_models"]


def test_optional_selection_resolves_bundle_and_explicit_override(monkeypatch):
    bundle = {"model_ids": {"action_intent": "action_intent/bundle", "pass_height": "pass_height/bundle"}}
    monkeypatch.setattr(model_utils, "load_bundle_record", lambda *a: bundle)
    monkeypatch.setattr(model_utils, "get_model_records", lambda ids: {task: {} for task in ids})
    monkeypatch.setattr(model_utils, "validate_model_record_consistency", lambda *a, **k: {})
    ids, _, selected_bundle = model_utils.resolve_model_selection(
        required_tasks=[], optional_tasks=list(HAWKEYE_MODEL_TASKS), bundle_id="bundle",
        explicit_model_ids={"action_intent": "action_intent/override"}, require_shared_split=False,
    )
    assert ids == {"action_intent": "action_intent/override", "pass_height": "pass_height/bundle"}
    assert selected_bundle is bundle


def test_empty_selection_is_opt_in():
    ids, shared, _ = model_utils.resolve_model_selection(required_tasks=[], optional_tasks=list(HAWKEYE_MODEL_TASKS))
    assert ids == {}
    assert shared["intended_receiver_mode"] is None
    with pytest.raises(ValueError, match="No model records"):
        model_utils.resolve_model_selection(required_tasks=[])


def test_loader_skips_only_unspecified_checkpoints(monkeypatch):
    load = Mock(return_value=object())
    monkeypatch.setattr(model_utils, "load_model", load)
    models = hawkeye.load_hawkeye_models(outcome_scoring_model_id="outcome_scoring/selected")
    assert list(models) == ["outcome_scoring"]
    load.assert_called_once_with("outcome_scoring/selected", "cpu")
    load.return_value = None
    with pytest.raises(FileNotFoundError, match="outcome_scoring"):
        hawkeye.load_hawkeye_models(outcome_scoring_model_id="outcome_scoring/missing")


@pytest.mark.parametrize("task", HAWKEYE_MODEL_TASKS)
def test_collector_evaluates_only_selected_component(pc_fixture, monkeypatch, task):
    subject, _, _, _, _ = pc_fixture
    selected = object()
    first = pd.DataFrame({"home_2": [.1, .2]}, index=[10, 11])
    second = first + .5
    predict = Mock(return_value=(first, second))
    monkeypatch.setattr(hawkeye, "inference_gnn", predict)
    result = hawkeye.infer_hawkeye_components(subject, {task: selected})
    predict.assert_called_once()
    assert predict.call_args.args[1] is selected
    assert result[task + "_failure" if task.startswith("outcome_") else task].equals(first)
    for column, frame in result.items():
        if column not in ({task + "_failure", task + "_success"} if task.startswith("outcome_")
                          else {task, "model_pass_success"} if task == "pass_success" else {task}):
            assert frame.empty


@pytest.mark.parametrize("values", [[0., 1.], [.123456789012345, .876543210987654]])
def test_checkpoint_free_v0_preserves_cache_values_and_inputs(pc_fixture, monkeypatch, values):
    subject, original_model, rows, folder, calls = pc_fixture
    runtime = SimpleNamespace(args={**original_model.args, "xpass_weight": "v0"})
    rows["home_2__top10_xpass"] = values
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    before = [graph.clone() for graph in subject.graph_features_0]
    monkeypatch.setattr(hawkeye, "inference_gnn", lambda *a, **k: pytest.fail("GNN evaluated"))
    result = hawkeye.infer_hawkeye_components(subject, {}, xpass_runtime=runtime)
    assert result["pass_success"]["home_2"].tolist() == values
    assert result["pass_success"]["home_1"].isna().all()
    assert result["model_pass_success"].empty
    assert not calls
    for graph, saved in zip(subject.graph_features_0, before):
        assert graph.keys() == saved.keys()
        assert torch.equal(graph.x, saved.x)


def test_checkpoint_free_v0_offside_and_carry_are_distinct(pc_fixture):
    subject, original_model, _, _, _ = pc_fixture
    runtime = SimpleNamespace(args={**original_model.args, "xpass_weight": "v0"})
    subject.graph_features_0[0].x[1, -1] = 1
    subject.labels[1, config.LABEL_INDEX["is_dribble"]] = 1
    result = inference.inference_xpass_v0(subject, runtime, graphs=subject.graph_features_0, labels=subject.labels)
    assert result.at[10, "home_2"] == 0.
    assert result.loc[11].isna().all()


def test_checkpoint_free_v0_partial_and_zero_coverage(pc_fixture, tmp_path):
    subject, model, rows, folder, _ = pc_fixture
    runtime = SimpleNamespace(args={**model.args, "xpass_weight": "v0"})
    @inference_attempt
    def run():
        start_inference_attempt(tmp_path / "run", SimpleNamespace(pc_xpass=True, use_physical_xpass=True))
        result = hawkeye.infer_hawkeye_components(subject, {}, xpass_runtime=runtime)
        metadata = finalize_pc_xpass_attempt()
        return result, metadata
    rows.loc[1, "home_2__top10_xpass"] = np.nan
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    result, metadata = run()
    assert result["pass_success"].at[10, "home_2"] == .2
    assert np.isnan(result["pass_success"].at[11, "home_2"])
    report, = metadata["pc_xpass_blending_coverage"].values()
    assert (report["eligible_count"], report["missing_count"]) == (2, 1)
    rows.loc[:, "home_2__top10_xpass"] = np.nan
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    with pytest.raises(PCXPassBlendingError, match="No usable"):
        run()


def test_v0_supplied_checkpoint_retains_model_and_carry_predictions(pc_fixture):
    subject, model, _, _, calls = pc_fixture
    model.args["xpass_weight"] = "v0"
    result = hawkeye.infer_hawkeye_components(subject, {"pass_success": model})
    assert result["pass_success"]["home_2"].tolist() == [.2, .8]
    assert result["model_pass_success"].at[10, "home_2"] == pytest.approx(.9)
    assert result["pass_success"].at[10, "home_1"] == pytest.approx(.9)
    assert len(calls) == 1


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
def test_blending_without_checkpoint_is_rejected_before_prompt(monkeypatch, version):
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", "--use-physical-xpass", "--pc-xpass", "--xpass-weight", version])
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("unexpected prompt"))
    with pytest.raises(ValueError, match="requires --pass-success-model-id"):
        run_hawkeye.main()


@pytest.mark.parametrize("version,extra,missing", [
    ("v4", [], "pass-height"),
    ("v5", ["--xpass-pass-height-model-id", "pass_height/aux"], "pass-intent"),
])
def test_blending_requires_auxiliaries_before_prompt(monkeypatch, version, extra, missing):
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", "--use-physical-xpass", "--pc-xpass",
        "--xpass-weight", version, "--pass-success-model-id", "pass_success/selected", *extra])
    monkeypatch.setattr(run_hawkeye, "resolve_model_selection", lambda **k: (
        {"pass_success": "pass_success/selected"}, {"intended_receiver_mode": None}, None))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("unexpected prompt"))
    with pytest.raises(ValueError, match=f"--xpass-{missing}-model-id is required"):
        run_hawkeye.main()


@pytest.fixture
def csv_inputs(tmp_path):
    tracking = make_tracking([(1., 10., 0.), (1.1, 10., 0.)])
    teammate = tracking.loc[tracking["team"] == "A"].copy()
    teammate["uefa_player_id"] = 3
    teammate["centroid_x"] = -10.
    tracking = pd.concat([tracking, teammate], ignore_index=True)
    tracking.to_csv(tmp_path / "tracking.csv", index=False)
    make_ball([1., 1.1]).to_csv(tmp_path / "ball.csv", index=False)
    return ["--tracking-csv", str(tmp_path / "tracking.csv"), "--ball-csv", str(tmp_path / "ball.csv"),
            "--output-dir", str(tmp_path / "runs"), "--run-id", "test", "--device", "cpu"]


def csv_pc_cache(tmp_path, values):
    folder = write_cache(tmp_path / "cache", cache_format=None)
    (folder / "matches").mkdir()
    pd.DataFrame({"match_id": ["s1", "s1"], "action_index": [0, 1], "frame_scope": ["frame_id"] * 2,
                  "home_3__top10_xpass": values}).to_parquet(folder / "matches/s1.parquet", index=False)
    return folder


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_runner_exports_without_any_checkpoint(tmp_path, monkeypatch, csv_inputs, cache_enabled):
    flags = list(csv_inputs)
    if cache_enabled:
        folder = csv_pc_cache(tmp_path, [.123456789012345, 1.])
        flags += ["--use-physical-xpass", "--pc-xpass", "--xpass-weight", "v0", "--physical-cache-dir", str(folder)]
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", *flags])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    monkeypatch.setattr(model_utils, "load_model", lambda *a: pytest.fail("checkpoint loaded"))
    run_hawkeye.main()
    output = tmp_path / "runs/test"
    table = pd.read_parquet(output / "hawkeye_data.parquet")
    csv = pd.read_csv(output / "hawkeye_data.csv")
    for column in hawkeye.COMPONENT_COLUMNS:
        if column == "pass_success" and cache_enabled:
            assert table.loc[table["uefa_player_id"] == 3, column].tolist() == [.123456789012345, 1.]
        else:
            assert table[column].isna().all()
            assert csv[column].isna().all()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["models"] == {}
    assert not metadata["model_pass_success_available"]
    assert metadata["model_selection"]["omitted_models"] == list(HAWKEYE_MODEL_TASKS)
    assert metadata["physical_xpass_cache_summary"]["physical_xpass_required"] == cache_enabled
    assert metadata["component_sources"]["pass_success"] == ("pc_xpass_v0" if cache_enabled else "unavailable")


def test_runner_exports_both_raw_outcomes_plus_pc_v0(tmp_path, monkeypatch, csv_inputs):
    folder = csv_pc_cache(tmp_path, [.2, .8])
    ids = {task: f"{task}/selected" for task in ("outcome_scoring", "outcome_conceding")}
    class OutcomeModel(DummyBaselineInferenceModel):
        def __init__(self, task):
            super().__init__(folder)
            self.args.update(task=task, model_id=ids[task], node_in_dim=26, inference_use_physical_xpass=False)
        def forward(self, graphs):
            return torch.logit(torch.tensor([.25, .75])).repeat(graphs.num_nodes, 1)
    load = Mock(side_effect=lambda identifier, device: OutcomeModel(identifier.split("/")[0]))
    monkeypatch.setattr(model_utils, "load_model", load)
    monkeypatch.setattr(model_utils, "get_model_records", lambda selected: {
        task: {"model_id": identifier, "graph_schema": {"node_in_dim": 26, "edge_in_dim": 2}}
        for task, identifier in selected.items()
    })
    monkeypatch.setattr(run_hawkeye, "get_model_provenance", lambda identifier: {"feature_signature": {}})
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", *csv_inputs, "--use-physical-xpass", "--pc-xpass",
        "--xpass-weight", "v0", "--physical-cache-dir", str(folder),
        "--outcome-scoring-model-id", ids["outcome_scoring"], "--outcome-conceding-model-id", ids["outcome_conceding"]])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    run_hawkeye.main()
    table = pd.read_parquet(tmp_path / "runs/test/hawkeye_data.parquet")
    for task in ids:
        assert table[task + "_failure"].tolist() == pytest.approx([.25] * len(table))
        assert table[task + "_success"].tolist() == pytest.approx([.75] * len(table))
    assert table.loc[table["uefa_player_id"] == 3, "pass_success"].tolist() == [.2, .8]
    assert table[["action_intent", "pass_intent", "pass_height", "model_pass_success"]].isna().all().all()
    assert load.call_count == 2
    metadata = json.loads((tmp_path / "runs/test/metadata.json").read_text())
    assert metadata["models"] == ids


@pytest.mark.parametrize("failure", ["invalid_cache", "zero_coverage"])
def test_checkpoint_free_runner_failure_does_not_publish(tmp_path, monkeypatch, csv_inputs, failure):
    folder = csv_pc_cache(tmp_path, [np.nan, np.nan])
    if failure == "invalid_cache":
        (folder / "metadata.json").write_text("invalid")
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", *csv_inputs, "--use-physical-xpass", "--pc-xpass",
        "--xpass-weight", "v0", "--physical-cache-dir", str(folder)])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    latest = Mock()
    monkeypatch.setattr(run_hawkeye, "write_latest_run", latest)
    with pytest.raises(PCXPassBlendingError):
        run_hawkeye.main()
    output = tmp_path / "runs/test"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["model_selection"]["warning_acceptances"]
    assert not list(output.glob("*.parquet"))
    assert not list(output.glob("*.csv"))
    latest.assert_not_called()


def test_runner_cancellation_prevents_data_loading_and_latest_update(tmp_path, monkeypatch, csv_inputs):
    monkeypatch.setattr("sys.argv", ["run_hawkeye.py", *csv_inputs, "--accept-comparison-warnings"])
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    monkeypatch.setattr(run_hawkeye, "load_hawkeye_tracking", lambda *a: pytest.fail("tracking loaded"))
    latest = Mock()
    monkeypatch.setattr(run_hawkeye, "write_latest_run", latest)
    with pytest.raises(ComparisonCancelled):
        run_hawkeye.main()
    assert not list((tmp_path / "runs/test").glob("*.parquet"))
    latest.assert_not_called()
