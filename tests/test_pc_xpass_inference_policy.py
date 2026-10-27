"""Required blending rejects incomplete inputs before models or later units run."""
from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import inference
from datatools import benchmark, hawkeye, skillcorner, config
from models.pass_height import definition
from pc_xpass_inference_policy import (PCXPassBlendingError, inference_attempt,
    start_inference_attempt, validate_pc_xpass_startup)
from test_physical_xpass import DummyBaselineInferenceModel, make_physical_inference_match
from test_saved_visualization_xpass import write_cache


@pytest.fixture
def fixture(tmp_path):
    folder = write_cache(tmp_path / "hawkeye", cache_format=None)
    (folder / "matches").mkdir()
    subject = make_physical_inference_match([10, 11], [1, 1])
    for graph in subject.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros((graph.num_nodes, 1))], dim=1)
    rows = pd.DataFrame({"match_id": ["match_1"] * 2, "action_index": [10, 11],
                         "frame_scope": ["frame_id"] * 2, "ball_z": [0., 0.],
                         "home_2__top10_xpass": [.2, .8],
                         "home_2__distance_to_nearest_opponent": [5., 5.]})
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    model = DummyBaselineInferenceModel(folder)
    model.args.update(pc_xpass=True, pc_xpass_id="fixture", x_pass_version="top10",
                      node_in_dim=26, physical_xpass_require_complete=False)
    calls = []
    model.register_forward_hook(lambda *a: calls.append(1))
    return subject, model, rows, folder, calls


@pytest.mark.parametrize("failure", ["missing", "corrupt", "source", "metric", "disabled"])
def test_startup_rejection_is_fatal(fixture, failure):
    subject, model, rows, folder, calls = fixture
    path = folder / "metadata.json"
    if failure == "missing":
        path.unlink()
    elif failure == "corrupt":
        path.write_text("invalid json")
    elif failure == "disabled":
        model.args["physical_runtime_cache_disabled"] = True
    else:
        metadata = json.loads(path.read_text())
        metadata.update({"source": "other"} if failure == "source" else {"available_metrics": ["max_xpass"]})
        path.write_text(json.dumps(metadata))
    with pytest.raises(PCXPassBlendingError) as error:
        inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert error.value.stage == "startup"
    assert error.value.__cause__ is not None
    assert not calls


@pytest.mark.parametrize("failure", ["row", "player", "nan", "inf", "corrupt_file", "nearest"])
def test_one_required_input_aborts_before_forward(fixture, failure):
    subject, model, rows, folder, calls = fixture
    if failure == "row":
        rows = rows.iloc[:1]
    elif failure == "player":
        rows = rows.drop(columns="home_2__top10_xpass")
    elif failure == "nearest":
        model.args["xpass_weight"] = "v2"
        rows.loc[1, "home_2__distance_to_nearest_opponent"] = np.nan
    elif failure in {"nan", "inf"}:
        rows.loc[1, "home_2__top10_xpass"] = np.nan if failure == "nan" else np.inf
    path = folder / "matches/match_1.parquet"
    rows.to_parquet(path, index=False)
    if failure == "corrupt_file":
        path.write_bytes(b"invalid parquet")
    with pytest.raises(PCXPassBlendingError):
        inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert not calls


@pytest.mark.parametrize("module", [benchmark, hawkeye, skillcorner])
def test_collector_preflight_runs_before_all_components(fixture, monkeypatch, module):
    subject, model, rows, folder, calls = fixture
    (folder / "matches/match_1.parquet").unlink()
    monkeypatch.setattr(module, "inference_gnn", lambda *a, **k: pytest.fail("component evaluated"))
    function = getattr(module, "infer_" + module.__name__.split(".")[-1] + "_components")
    with pytest.raises(PCXPassBlendingError):
        function(subject, {"pass_success": model}, device="cpu")


@pytest.mark.parametrize("role", ["pass_height", "pass_intent"])
def test_auxiliary_nonfinite_fails_before_primary_forward(fixture, role):
    subject, model, rows, folder, calls = fixture
    model.args.update(xpass_weight="v5", xpass_pass_height_definition=definition(2., "checkpoint"))
    height = pd.DataFrame({"home_1": [np.nan, np.nan], "home_2": [.5, .5]}, index=[10, 11])
    intent = pd.DataFrame({"home_1": [np.nan, np.nan], "home_2": [.5, .5]}, index=[10, 11])
    (height if role == "pass_height" else intent).loc[11, "home_2"] = np.nan
    with pytest.raises(PCXPassBlendingError) as error:
        inference.inference_gnn(subject, model, device="cpu", pass_height_probs=height, pass_intent_probs=intent)
    assert error.value.stage == "auxiliary_prediction"
    assert not calls


def test_self_opponents_and_offside_values_are_not_required(fixture):
    subject, model, rows, folder, calls = fixture
    # Cache deliberately contains neither carrier nor opponent predictions.
    rows.loc[1, "home_2__top10_xpass"] = np.nan
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    subject.graph_features_0[1].x[1, -1] = 1
    final, raw = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert final.at[11, "home_2"] == 0
    assert final.at[10, "home_1"] == raw.at[10, "home_1"]
    assert final.at[10, "home_2"] != raw.at[10, "home_2"]
    assert calls == [1]


def test_valid_model_only_weight_is_not_rejection(fixture):
    subject, model, rows, folder, calls = fixture
    model.args.update(xpass_weight="v5", xpass_pass_height_definition=definition(2., "checkpoint"))
    table = pd.DataFrame({"home_1": [np.nan, np.nan], "home_2": [1., 1.]}, index=[10, 11])
    final, raw = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True,
                                        pass_height_probs=table, pass_intent_probs=table)
    pd.testing.assert_frame_equal(final, raw)


def test_startup_memo_reuses_validation_but_detects_metadata_changes(fixture, monkeypatch):
    import physical_pass_model as physics
    from unittest.mock import Mock
    subject, model, rows, folder, calls = fixture
    check = Mock(wraps=physics.validate_x_pass_version_available)
    monkeypatch.setattr(physics, "validate_x_pass_version_available", check)
    memo = {}
    validate_pc_xpass_startup(model, memo=memo)
    validate_pc_xpass_startup(model, memo=memo)
    assert check.call_count == 1
    metadata = json.loads((folder / "metadata.json").read_text())
    metadata["available_metrics"] = ["max_xpass"]
    (folder / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(PCXPassBlendingError):
        validate_pc_xpass_startup(model, memo=memo)
    assert check.call_count == 2


def test_preflight_and_inference_reuse_lookup_settings(fixture, monkeypatch):
    from unittest.mock import Mock
    subject, model, rows, folder, calls = fixture
    lookup = Mock(wraps=inference.physical_xpass_inference_lookup_config)
    monkeypatch.setattr(inference, "physical_xpass_inference_lookup_config", lookup)
    inference.preflight_pc_xpass(subject, model)
    inference.inference_gnn(subject, model, device="cpu")
    assert lookup.call_count == 1


def test_capturing_model_predictions_cannot_swallow_rejected_rows(fixture, monkeypatch):
    subject, model, rows, folder, calls = fixture
    def reject(*a, **k):
        raise inference.PhysicalXPassNoUsableRowsError("no rows")
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference", reject)
    with pytest.raises(PCXPassBlendingError):
        inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert not calls


def test_invalid_attachment_fails_before_primary_forward(fixture, monkeypatch):
    import physical_pass_model as physics
    subject, model, rows, folder, calls = fixture
    attach = inference.attach_physical_xpass_for_inference
    def corrupt(*args, **kwargs):
        graphs = attach(*args, **kwargs)
        getattr(graphs[-1], physics.PHYSICAL_XPASS_DISTANCE_ATTR)[1] = np.nan
        return graphs
    monkeypatch.setattr(inference, "attach_physical_xpass_for_inference", corrupt)
    with pytest.raises(PCXPassBlendingError) as error:
        inference.inference_gnn(subject, model, device="cpu")
    assert error.value.stage == "attachment"
    assert not calls


@pytest.mark.parametrize("fail_at", [0, 1])
def test_attempt_stops_and_records_failed_partial_outputs(tmp_path, fixture, fail_at):
    subject, model, rows, folder, calls = fixture
    run = tmp_path / "attempt"
    latest = tmp_path / "latest.json"
    latest.write_text('{"run_id":"previous"}')
    visited = []
    @inference_attempt
    def execute():
        start_inference_attempt(run, Namespace(pc_xpass=True, use_physical_xpass=True))
        assert json.loads((run / "metadata.json").read_text())["status"] == "running"
        for index in range(3):
            visited.append(index)
            if index == fail_at:
                model.args["physical_cache_dir"] = str(tmp_path / "missing")
            validate_pc_xpass_startup(model)
            (run / f"unit_{index}.csv").write_text("complete")
        latest.write_text('{"run_id":"attempt"}')
    with pytest.raises(PCXPassBlendingError):
        execute()
    assert visited == list(range(fail_at + 1))
    assert latest.read_text() == '{"run_id":"previous"}'
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["failure"]["type"] == "PCXPassBlendingError"
    assert metadata["failure"]["stage"] == "startup"
    assert len(metadata["available_outputs"]) == fail_at


def test_failure_reporting_does_not_replace_original(tmp_path, monkeypatch):
    import project_config
    root = tmp_path / "attempt"
    error = PCXPassBlendingError("original", stage="blending")
    @inference_attempt
    def execute():
        start_inference_attempt(root, Namespace(pc_xpass=True, use_physical_xpass=True))
        monkeypatch.setattr(project_config, "write_run_metadata", lambda *a: (_ for _ in ()).throw(OSError("disk error")))
        raise error
    with pytest.raises(PCXPassBlendingError) as caught:
        execute()
    assert caught.value is error
    assert "disk error" in caught.value.__notes__[0]


@pytest.mark.parametrize("fail", [False, True])
def test_epv_attempt_preserves_canonical_outputs_on_blend_failure(tmp_path, fixture, monkeypatch, fail):
    from scripts import generate_epv as epv
    subject, model, rows, folder, calls = fixture
    output = tmp_path / "epv"
    match_dir = output / "matches"
    match_dir.mkdir(parents=True)
    (output / "epv.csv").write_text("previous aggregate")
    (output / "metadata.json").write_text('{"status":"completed","run_id":"previous"}')
    (match_dir / "one.csv").write_text("previous match")
    args = epv.parse_args(["--pc-xpass", "--use-physical-xpass", "--xpass-weight", "v1", "--overwrite"])
    tasks = ["pass_intent", "pass_success", "outcome_scoring", "outcome_conceding"]
    ids = {task: task + "/fake" for task in tasks}
    monkeypatch.setattr(epv, "parse_args", lambda: args)
    monkeypatch.setattr(epv, "EPV_DIR", output)
    monkeypatch.setattr(epv, "EPV_MATCH_DIR", match_dir)
    monkeypatch.setattr(epv, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr(epv, "generate_run_id", lambda *a: "attempt")
    monkeypatch.setattr(epv, "resolve_epv_model_selection", lambda *a: (ids, {}, None))
    monkeypatch.setattr(epv, "load_epv_models", lambda *a: {task: model for task in tasks})
    monkeypatch.setattr(epv, "configure_epv_physical_xpass", lambda *a: str(folder))
    monkeypatch.setattr(epv, "validate_model_graph_schemas", lambda *a: {"add_v_edge_features": False})
    monkeypatch.setattr(epv, "resolve_runtime_feature_run_context", lambda *a, **k: {
        "feature_run_id": "feature", "intended_receiver_mode": "model", "feature_schema": {},
        "runtime_feature_run_selection": "explicit", "selection": "explicit"})
    monkeypatch.setattr(epv, "resolve_runtime_return_type", lambda *a: "next_3")
    monkeypatch.setattr(epv, "get_model_provenance", lambda identifier: {"model_id": identifier})
    monkeypatch.setattr(epv, "resolve_feature_root", lambda *a: tmp_path / "feature")
    monkeypatch.setattr(epv, "resolve_match_ids", lambda *a: ["one", "two", "three"])
    monkeypatch.setattr(epv, "load_match", lambda *a, **k: subject)
    monkeypatch.setattr(epv, "load_events", lambda mid: pd.DataFrame({"stats_perform_match_id": [mid],
        "action_id": [10], "epv": [.2], "scores_epv": [.3], "concedes_epv": [.1]}))
    visited = []
    def predict(*a):
        visited.append(len(visited))
        if fail and len(visited) == 2:
            raise PCXPassBlendingError("missing candidate", stage="state_validation", context={"match_id": "two"})
        return pd.DataFrame()
    monkeypatch.setattr(epv, "infer_match_epv", predict)
    monkeypatch.setattr(epv, "annotate_match_epv", lambda events, values: (events, events))
    if fail:
        with pytest.raises(PCXPassBlendingError):
            epv.main()
        assert len(visited) == 2
        assert (output / "epv.csv").read_text() == "previous aggregate"
        assert json.loads((output / "metadata.json").read_text())["run_id"] == "previous"
        assert (match_dir / "one.csv").read_text() == "previous match"
        assert not (match_dir / "two.csv").exists()
    else:
        epv.main()
        assert len(visited) == 3
        assert len(pd.read_csv(output / "epv.csv")) == 3
        assert (match_dir / "three.csv").exists()
    attempt = output / "inference_runs/attempt"
    assert json.loads((attempt / "metadata.json").read_text())["status"] == ("failed" if fail else "completed")
    assert (attempt / "matches/one.csv").exists()


@pytest.mark.parametrize("provider", ["relevant_models", "skillcorner", "hawkeye", "benchmark"])
def test_runner_startup_writes_failed_metadata_before_unit_preparation(tmp_path, monkeypatch, provider):
    import importlib
    from unittest.mock import Mock
    script = importlib.import_module("scripts.run_" + provider)
    output = tmp_path / "runs"
    flags = ["--pc-xpass", "--use-physical-xpass", "--xpass-weight", "v1", "--run-id", "attempt",
             "--output-dir", str(output), "--physical-cache-dir", str(tmp_path / "missing")]
    monkeypatch.setattr("sys.argv", [script.__file__, *flags])
    args = script.parse_args()
    monkeypatch.setattr(script, "parse_args", lambda: args)
    tasks = ["action_intent", "pass_intent", "pass_success", "outcome_scoring", "outcome_conceding"]
    ids = {task: task + "/fake" for task in tasks}
    models = {task: DummyBaselineInferenceModel(tmp_path / "missing") for task in tasks}
    for task, model in models.items():
        model.args.update(task=task, node_in_dim=26)
    monkeypatch.setattr(script, "resolve_model_selection", lambda **k: (ids, {
        "intended_receiver_mode": "model", "feature_run_id": "feature", "return_type": "next_3"}, None))
    monkeypatch.setattr(script, "resolve_optional_model_id", lambda *a: None)
    monkeypatch.setattr(script, "validate_model_graph_schemas", lambda *a: {"node_in_dim": 26, "edge_in_dim": 2})
    if provider == "relevant_models":
        monkeypatch.setattr(script, "load_model", lambda identifier, device: models[identifier.split('/')[0]])
        monkeypatch.setattr(script, "resolve_component_feature_run_context", lambda *a: {
            "feature_run_id": "feature", "feature_root": tmp_path / "feature", "feature_schema": {},
            "intended_receiver_mode": "model", "selection": "explicit"})
        prepare = Mock()
        monkeypatch.setattr(script, "build_sportec_possessions", prepare)
    else:
        monkeypatch.setattr(script, "load_" + provider + "_models", lambda **k: models)
        if provider == "hawkeye":
            monkeypatch.setattr(script, "load_hawkeye_tracking", lambda *a: pd.DataFrame())
            monkeypatch.setattr(script, "clean_hawkeye_tracking", lambda data: data)
            monkeypatch.setattr(script, "load_hawkeye_ball", lambda *a: pd.DataFrame())
            monkeypatch.setattr(script, "clean_hawkeye_ball", lambda data: data)
            monkeypatch.setattr(script, "resolve_situation_ids", lambda *a, **k: ["s"])
            prepare = Mock()
            monkeypatch.setattr(script, "build_hawkeye_situation", prepare)
        elif provider == "skillcorner":
            monkeypatch.setattr(script, "discover_skillcorner_matches", lambda *a, **k: (["s"], {}))
            prepare = Mock()
            monkeypatch.setattr(script, "build_skillcorner_match_context", prepare)
        else:
            monkeypatch.setattr(script, "discover_benchmark_modifications", lambda *a, **k: (["s"], {}))
            prepare = Mock()
            monkeypatch.setattr(script, "build_benchmark_state", prepare)
    latest = Mock()
    monkeypatch.setattr(script, "write_latest_run", latest)
    with pytest.raises(PCXPassBlendingError):
        script.main()
    metadata = json.loads((output / "attempt/metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["failure"]["stage"] == "startup"
    prepare.assert_not_called()
    latest.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_location_attempt_keeps_fatal_error_and_finishes_cache_generation(tmp_path, monkeypatch, cleanup_fails):
    from scripts import run_hawkeye_loc as script
    from unittest.mock import Mock
    args = Namespace(run_id="attempt", output_dir=str(tmp_path), overwrite=False)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr(script.pc_versions, "start_generation", lambda *a, **k: None)
    finish = Mock(side_effect=OSError("cleanup") if cleanup_fails else None)
    monkeypatch.setattr(script.pc_versions, "finish_generation", finish)
    error = PCXPassBlendingError("missing target", stage="state_validation", context={"frame": 10})
    def reject(*a):
        raise error
    monkeypatch.setattr(script, "_run_location", reject)
    with pytest.raises(PCXPassBlendingError) as caught:
        script.main()
    assert caught.value is error
    finish.assert_called_once_with(args, success=False)
    metadata = json.loads((tmp_path / "attempt/metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["failure"]["context"]["frame"] == "10"


def test_cache_selection_errors_are_typed_startup_failures(tmp_path, monkeypatch):
    from scripts import run_relevant_models as script
    args = Namespace(pc_xpass=True, use_physical_xpass=True, physical_cache_dir=None)
    def reject(*a, **k):
        raise FileNotFoundError("selected cache missing")
    monkeypatch.setattr(script.pc_versions, "cache_dir", reject)
    with pytest.raises(PCXPassBlendingError) as error:
        script.resolve_component_physical_cache_dir(args)
    assert error.value.stage == "startup"
    cause = error.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, FileNotFoundError)
