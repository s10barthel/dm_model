"""Optional checkpoints and physical-only success across dataset entry points."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from datatools import benchmark, hawkeye, skillcorner
from datatools.epv import compute_epv_values
from models import utils as model_utils
from scripts import generate_epv, run_relevant_models
from scripts import run_skillcorner, run_benchmark, run_hawkeye_loc
from scripts import run_and_visualize_hawkeye as hawkeye_visualizer
from scripts import visualize_action_components as sportec_visualizer
from scripts.inference_components import (
    ComponentModels, component_availability, configure_xpass_runtime,
    confirm_component_selection, require_sportec_feature_context, validate_component_dependencies,
)
from scripts.visualization_availability import select_inference_components, saved_component_inputs
from scripts import inference_visualization_comparison as comparison
from test_pc_xpass_inference_policy import fixture as pc_fixture


def options(**changes):
    values = dict(use_physical_xpass=True, pc_xpass=True, xpass_weight="v0", x_pass_version="top10")
    values.update(changes)
    return SimpleNamespace(**values)


def outcome_ids():
    return {task: task + "/selected" for task in ("outcome_scoring", "outcome_conceding")}


def test_v0_supplies_pass_score_and_epv_without_checkpoint(tmp_path, monkeypatch, capsys):
    ids = {**outcome_ids(), "pass_intent": "pass_intent/selected"}
    report = component_availability(ids, options())
    assert report["available"]["pass_score"]
    assert report["available"]["epv"]
    assert not report["available"]["carry_score"]
    validate_component_dependencies(options(), ids, require_epv=True)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    decision = confirm_component_selection(ids, options(), tmp_path)
    warning = capsys.readouterr().err
    assert "carry_score: unavailable" in warning
    assert "epv: unavailable" not in warning
    assert "- pass_score: unavailable" not in warning
    assert decision["availability"] == report


def test_derived_warning_names_missing_dependencies(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    confirm_component_selection(outcome_ids(), options(), tmp_path)
    warning = capsys.readouterr().err
    assert "epv: unavailable; missing inputs: pass_intent" in warning
    assert "- pass_score: unavailable" not in warning


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
def test_blending_still_requires_learned_success(version):
    with pytest.raises(ValueError, match="requires --pass-success-model-id"):
        validate_component_dependencies(options(xpass_weight=version), outcome_ids())


@pytest.mark.parametrize("missing", ["pass_intent", "outcome_scoring", "outcome_conceding"])
def test_dedicated_epv_keeps_other_dependencies(missing):
    ids = {**outcome_ids(), "pass_intent": "pass_intent/selected"}
    del ids[missing]
    with pytest.raises(ValueError, match=missing):
        validate_component_dependencies(options(), ids, require_epv=True)


@pytest.mark.parametrize("module", [hawkeye, benchmark, skillcorner])
def test_dataset_collectors_use_v0_without_any_models(pc_fixture, module):
    subject, original, _, folder, calls = pc_fixture
    runtime = configure_xpass_runtime(options(physical_cache_dir=str(folder)), {}, folder)
    predict = getattr(module, "infer_" + module.__name__.split(".")[-1] + "_components")
    result = predict(subject, {}, device="cpu", xpass_runtime=runtime)
    assert result["pass_success"]["home_2"].tolist() == [.2, .8]
    assert result["pass_success"]["home_1"].isna().all()
    assert result["model_pass_success"].empty
    assert not calls


def test_skillcorner_blank_component_preserves_frames(pc_fixture):
    subject, _, _, _, _ = pc_fixture
    subject.frame_meta = pd.DataFrame({"match_id": ["match_1"] * 2, "index": [1, 1],
        "period": [1, 1], "player_id": [1, 1], "attacking_side": ["home"] * 2},
        index=pd.Index([10, 11], name="frame_id"))
    result = skillcorner.build_skillcorner_component_table(subject, pd.DataFrame(),
        pd.DataFrame({"player_id": [1, 2, 3]}), include_shot=False)
    assert result["frame"].tolist() == [10, 11]
    assert result[["1", "2", "3"]].isna().all().all()


def test_sportec_empty_selection_requires_feature_context():
    with pytest.raises(ValueError, match="--feature-run-id"):
        require_sportec_feature_context(SimpleNamespace(), {}, None)
    require_sportec_feature_context(SimpleNamespace(feature_run_id="selected"), {}, None)
    require_sportec_feature_context(SimpleNamespace(), {}, None, cache_selected=True)


def outcome_predictions(subject, model, **kwargs):
    task = model.args["task"]
    if task == "pass_intent":
        return pd.DataFrame({"home_2": [1., 1.]}, index=[10, 11]), None
    failure, success = ((.3, .6) if task == "outcome_scoring" else (.2, .1))
    return (pd.DataFrame({"home_1": [failure] * 2, "home_2": [failure] * 2}, index=[10, 11]),
            pd.DataFrame({"home_1": [success] * 2, "home_2": [success] * 2}, index=[10, 11]))


def test_epv_inference_uses_cached_v0_without_checkpoint(pc_fixture, monkeypatch):
    subject, _, _, folder, calls = pc_fixture
    models = {task: SimpleNamespace(args={"task": task}) for task in (*outcome_ids(), "pass_intent")}
    subject.xpass_runtime = configure_xpass_runtime(options(), models, folder)
    subject.actions["action_id"] = ["a", "b"]
    monkeypatch.setattr(generate_epv, "inference_gnn", outcome_predictions)
    result = generate_epv.infer_match_epv(subject, models, "cpu")
    assert result["epv"].tolist() == pytest.approx([.18, .42])
    assert not calls


def test_epv_command_publishes_v0_without_success_checkpoint(pc_fixture, monkeypatch, tmp_path):
    subject, _, _, folder, calls = pc_fixture
    ids = {**outcome_ids(), "pass_intent": "pass_intent/selected"}
    args = generate_epv.parse_args(["--feature-run-id", "fixture", "--physical-cache-dir", str(folder),
        "--use-physical-xpass", "--pc-xpass", "--xpass-weight", "v0",
        *[item for task, identifier in ids.items() for item in ("--" + task.replace("_", "-") + "-model-id", identifier)]])
    root = tmp_path / "epv"
    monkeypatch.setattr(generate_epv, "parse_args", lambda: args)
    monkeypatch.setattr(generate_epv, "EPV_DIR", root)
    monkeypatch.setattr(generate_epv, "EPV_MATCH_DIR", root / "matches")
    monkeypatch.setattr(generate_epv, "ensure_project_dirs", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    monkeypatch.setattr(model_utils, "get_model_records", lambda selected: {
        task: {"model_id": identifier, "graph_schema": {"node_in_dim": 26, "edge_in_dim": 2}}
        for task, identifier in selected.items()})
    monkeypatch.setattr(generate_epv, "load_epv_models", lambda selected, device: {
        task: SimpleNamespace(args={"task": task}) for task in selected})
    monkeypatch.setattr(generate_epv, "resolve_runtime_feature_run_context", lambda *a, **k: {
        "feature_run_id": "fixture", "intended_receiver_mode": "predict", "feature_schema": {}, "selection": "explicit"})
    monkeypatch.setattr(generate_epv, "resolve_runtime_return_type", lambda *a: "next_3")
    monkeypatch.setattr(generate_epv, "resolve_feature_root", lambda *a: tmp_path / "features")
    monkeypatch.setattr(generate_epv, "get_model_provenance", lambda identifier: {"feature_signature": {}})
    monkeypatch.setattr(generate_epv, "resolve_match_ids", lambda *a: ["match_1"])
    subject.actions["action_id"] = ["a", "b"]
    monkeypatch.setattr(generate_epv, "load_match", lambda *a, **k: subject)
    monkeypatch.setattr(generate_epv, "load_events", lambda *a: pd.DataFrame({
        "stats_perform_match_id": ["match_1"] * 2, "action_id": ["a", "b"]}))
    monkeypatch.setattr(generate_epv, "inference_gnn", outcome_predictions)
    def annotate(events, values):
        events = events.merge(values, on="action_id", how="left")
        events["scores_epv"] = events["concedes_epv"] = np.nan
        return events, events
    monkeypatch.setattr(generate_epv, "annotate_match_epv", annotate)
    generate_epv.main()
    assert pd.read_csv(root / "epv.csv")["epv"].tolist() == pytest.approx([.18, .42])
    metadata = json.loads((root / "metadata.json").read_text())
    assert "pass_success" not in metadata["models"]
    assert metadata["model_selection"]["availability"]["available"]["epv"]
    assert not metadata["model_selection"]["availability"]["available"]["carry_score"]
    assert not calls


def test_epv_rejects_partial_outcome_sum():
    intent = pd.DataFrame({"a": [.4], "b": [.6]})
    complete = pd.DataFrame({"a": [.5], "b": [.5]})
    missing = complete.copy()
    missing.at[0, "b"] = np.nan
    value = compute_epv_values(intent, complete, missing, complete, complete, complete)
    assert np.isnan(value.iloc[0])


@pytest.mark.parametrize("visualizer", [hawkeye_visualizer, sportec_visualizer])
def test_visualizer_pass_score_uses_v0_without_checkpoint(pc_fixture, monkeypatch, tmp_path, visualizer):
    subject, _, _, folder, calls = pc_fixture
    models = ComponentModels({task: SimpleNamespace(args={"task": task}) for task in outcome_ids()})
    configure_xpass_runtime(options(), models, folder)
    monkeypatch.setattr(visualizer, "inference_gnn", outcome_predictions)
    if visualizer is hawkeye_visualizer:
        result = visualizer.collect_situation_components(subject, models, "cpu", [10, 11], ["pass_score"])
        assert result["pass_score"]["home_2"].tolist() == pytest.approx([.18, .42])
        assert result["pass_score"]["home_1"].isna().all()
    else:
        monkeypatch.setattr(visualizer.match_cache, "require_format", lambda *a, **k: None)
        monkeypatch.setattr(visualizer, "possession_state_for_action", lambda *a: (subject, 10))
        result = visualizer.collect_action_components(subject, models, tmp_path, "cpu", 10, "a", tmp_path,
            rendered_components=["pass_score"], physical_cache_dir=folder)
        assert result["pass_score"]["home_2"] == pytest.approx(.18)
        assert np.isnan(result["pass_score"]["home_1"])
    assert not calls


def test_visualization_selection_accepts_v0_provider():
    selected = select_inference_components(options(only_pass_score=True), outcome_ids())
    assert selected.rendered_components == ["pass_score"]


def test_v0_cache_comparison_has_independent_checkpoint_free_runtimes(pc_fixture, tmp_path, monkeypatch):
    subject, _, _, folder, calls = pc_fixture
    args = hawkeye_visualizer.parse_args(["--only-pass-success", "--pc-xpass", "--use-physical-xpass",
        "--xpass-weight", "v0", "--pc-xpass-id", "cache_a", "--pc-xpass-id", "cache_b", "--compare",
        "--output-dir", str(tmp_path), "--run-id", "compare"])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    def cache_dir(dataset, selected):
        selected.pc_xpass_namespace = "normal"
        return folder
    monkeypatch.setattr(hawkeye_visualizer.pc_versions, "cache_dir", cache_dir)
    variants, selection, _, _, _ = comparison.resolve_variants(args, hawkeye_visualizer, False)
    comparison.configure_physical(args, hawkeye_visualizer, variants, "hawkeye")
    assert all(not variant.models for variant in variants)
    assert variants[0].models.xpass_runtime is not variants[1].models.xpass_runtime
    for variant in variants:
        with comparison.active_variant(variant, "fixture"):
            result = hawkeye_visualizer.collect_situation_components(subject, variant.models, "cpu", [10, 11],
                selection.rendered_components)
            assert result["pass_success"]["home_2"].tolist() == [.2, .8]
    assert not calls


def test_sportec_v0_visualizer_selects_cache_feature_context(pc_fixture, monkeypatch, tmp_path):
    _, _, _, folder, _ = pc_fixture
    path = folder / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["source_inputs"] = {"feature_run_id": "cached_features"}
    path.write_text(json.dumps(metadata))
    args = sportec_visualizer.parse_args(["--match-id", "match_1", "--only-pass-success",
        "--physical-cache-dir", str(folder), "--pc-xpass", "--use-physical-xpass", "--xpass-weight", "v0",
        "--output-dir", str(tmp_path), "--run-id", "visual"])
    monkeypatch.setattr(sportec_visualizer, "parse_args", lambda: args)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    class SelectedContext(Exception):
        pass
    def resolve(identifier, *a, **k):
        assert identifier == "cached_features"
        raise SelectedContext
    monkeypatch.setattr(sportec_visualizer, "resolve_runtime_feature_run_context", resolve)
    with pytest.raises(SelectedContext):
        sportec_visualizer.main()


def test_saved_nan_placeholders_are_unavailable():
    tables = {"pass_intent": pd.DataFrame({"frame": [1], "home_2": [np.nan]}),
              "pass_success": pd.DataFrame({"frame": [1], "home_2": [.5]})}
    metadata = {"model_selection": {"availability": component_availability({}, options())}}
    assert saved_component_inputs(tables, metadata) == {"pass_success"}


@pytest.mark.parametrize("mode", ["loc", "freeze"])
def test_location_configuration_needs_no_models(tmp_path, monkeypatch, mode):
    args = run_hawkeye_loc.parse_args(["--mode", mode, "--xpass-weight", "v0",
        "--pc-xpass-cache-dir", str(tmp_path / "cache")])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    ids, shared, models, schema, device = run_hawkeye_loc._configure_models(args, warning_root=tmp_path / "run")
    assert ids == models == {}
    assert args._xpass_runtime.args["xpass_weight"] == "v0"
    assert args._model_selection["availability"]["available"]["carry_score"] is False


def test_location_omission_cancellation_precedes_cache_generation(tmp_path, monkeypatch):
    from scripts.comparison_warnings import ComparisonCancelled
    args = run_hawkeye_loc.parse_args(["--xpass-weight", "v0", "--pc-xpass-cache-dir", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "runs"), "--run-id", "cancelled"])
    monkeypatch.setattr(run_hawkeye_loc, "parse_args", lambda: args)
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    monkeypatch.setattr(run_hawkeye_loc.pc_versions, "start_generation", lambda *a, **k: pytest.fail("cache generation started"))
    with pytest.raises(ComparisonCancelled):
        run_hawkeye_loc.main()
    assert not (tmp_path / "cache").exists()
    assert json.loads((tmp_path / "runs/cancelled/metadata.json").read_text())["status"] == "cancelled"


def test_benchmark_unweighted_score_does_not_need_intent():
    from validation.benchmark.benchmark_postprocessing import build_benchmark_summary
    raw = pd.DataFrame({"modification": [1, 1], "game_state": [1, 2], "higher_state_id": [1, 1],
        "pass_intent": [np.nan, np.nan], "pass_success": [.2, .8],
        "outcome_scoring_success": [.6, .6], "outcome_scoring_failure": [.3, .3],
        "outcome_conceding_success": [.1, .1], "outcome_conceding_failure": [.2, .2]})
    summary, _ = build_benchmark_summary(raw, pass_intent_available=False)
    assert summary["avg_pass_score_1"].iloc[0] == pytest.approx(.18)
    assert summary["avg_pass_score_2"].iloc[0] == pytest.approx(.42)
    assert summary[["game_state_value_1", "game_state_value_2"]].isna().all().all()


def test_skillcorner_incomplete_positive_intent_is_not_partial_value():
    from validation.skillcorner.code.skillcorner_postprocessing import compute_model_scores
    raw = pd.DataFrame({"match_id": [1, 1], "index": [1, 1], "frame": [10, 10],
        "pass_intent": [.5, .5], "pass_success": [.2, np.nan],
        "outcome_scoring_success": [.6, .6], "outcome_scoring_failure": [.3, .3],
        "outcome_conceding_success": [.1, .1], "outcome_conceding_failure": [.2, .2]})
    result = compute_model_scores(raw)
    assert result["pass_score"].iloc[0] == pytest.approx(.18)
    assert result["game_state_value"].isna().all()


def test_skillcorner_runner_exports_without_models(pc_fixture, tmp_path, monkeypatch):
    subject, _, _, folder, _ = pc_fixture
    subject.frame_meta = pd.DataFrame({"match_id": ["match_1"] * 2, "index": [1, 1],
        "period": [1, 1], "player_id": [1, 1], "attacking_side": ["home"] * 2},
        index=pd.Index([10, 11], name="frame_id"))
    args = run_skillcorner.parse_args(["--output-dir", str(tmp_path / "runs"), "--run-id", "test",
        "--physical-cache-dir", str(folder), "--use-physical-xpass", "--pc-xpass", "--xpass-weight", "v0"])
    monkeypatch.setattr(run_skillcorner, "parse_args", lambda: args)
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    monkeypatch.setattr(run_skillcorner, "discover_skillcorner_matches", lambda *a, **k: (["match_1"], {}))
    monkeypatch.setattr(run_skillcorner, "build_skillcorner_match_context", lambda *a: {
        "events": pd.DataFrame({"index": [1]}), "player_meta": pd.DataFrame({"player_id": [1, 2, 3]})})
    monkeypatch.setattr(run_skillcorner, "build_skillcorner_possession", lambda *a, **k: (
        subject, {"valid_frames": 2, "total_frames": 2}))
    monkeypatch.setattr(run_skillcorner, "run_skillcorner_postprocessing", lambda **k: (
        None, {"events_with_dm_score": 0, "event_rows": 1}, tmp_path / "summary.csv"))
    monkeypatch.setattr(run_skillcorner, "run_skillcorner_filter", lambda **k: (
        None, None, None, None, {"skillcorner_actions_rows": 0},
        {key: "unused" for key in ("actions_raw_path", "actions_path", "matches_path", "players_path")}))
    run_skillcorner.main()
    root = tmp_path / "runs/test"
    assert pd.read_parquet(root / "match_1/pass_success.parquet")["2"].tolist() == [.2, .8]
    assert pd.read_parquet(root / "match_1/pass_intent.parquet")["2"].isna().all()
    assert json.loads((root / "metadata.json").read_text())["models"] == {}


def test_benchmark_runner_exports_without_models(pc_fixture, tmp_path, monkeypatch):
    subject, _, _, folder, _ = pc_fixture
    monkeypatch.setattr("sys.argv", ["run_benchmark.py", "--output-dir", str(tmp_path / "runs"),
        "--run-id", "test", "--physical-cache-dir", str(folder), "--use-physical-xpass", "--pc-xpass", "--xpass-weight", "v0"])
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    monkeypatch.setattr(run_benchmark, "discover_benchmark_modifications", lambda *a, **k: ([1], {}))
    monkeypatch.setattr(run_benchmark, "load_benchmark_modification_data", lambda *a: {
        "game_state_1": None, "game_state_2": None, "higher_state_id": 1})
    def build(*args, **kwargs):
        rows = pd.DataFrame({"frame_id": [10, 11], "object_id": ["home_2"] * 2,
            "modification": [1, 1], "game_state": [kwargs["game_state_id"]] * 2,
            "higher_state_id": [1, 1], "team": [1, 1], "player": [2, 2]})
        for column in ("pos_x", "pos_y", "pos_z", "smooth_x_speed", "smooth_y_speed", "event_player"):
            rows[column] = 0
        return subject, rows, {"valid_frames": 2, "total_frames": 2}
    monkeypatch.setattr(run_benchmark, "build_benchmark_state", build)
    monkeypatch.setattr(run_benchmark, "run_benchmark_postprocessing", lambda *a, **k: (
        None, {}, tmp_path / "summary.csv", tmp_path / "summary.txt"))
    monkeypatch.setattr(run_benchmark, "update_benchmark_runs_ledger", lambda *a: tmp_path / "ledger.csv")
    run_benchmark.main()
    root = tmp_path / "runs/test"
    table = pd.read_parquet(root / "benchmark_data.parquet")
    assert table["pass_success"].tolist() == [.2, .8, .2, .8]
    assert table["model_pass_success"].isna().all()
    assert json.loads((root / "metadata.json").read_text())["models"] == {}


@pytest.mark.parametrize("physical", [False, True])
def test_sportec_runner_exports_states_with_no_models(pc_fixture, monkeypatch, tmp_path, physical):
    subject, _, _, folder, _ = pc_fixture
    subject.actions["stats_perform_match_id"] = "match_1"
    subject.actions["action_id"] = ["a", "b"]
    subject.actions["original_event_id"] = ["a", "b"]
    subject.event_index = 1
    subject.possession_report = {}
    subject.sportec_state_plan = SimpleNamespace(manifest=lambda: {})
    flags = ["--feature-run-id", "fixture", "--output-dir", str(tmp_path / "runs"), "--run-id", "test",
             "--physical-cache-dir", str(folder), "--match-id", "match_1"]
    if physical:
        flags += ["--use-physical-xpass", "--pc-xpass", "--xpass-weight", "v0"]
    parser = run_relevant_models.parse_args
    monkeypatch.setattr(run_relevant_models, "parse_args", lambda: parser(flags))
    monkeypatch.setattr("builtins.input", lambda *a: "yes")
    monkeypatch.setattr(run_relevant_models, "resolve_component_feature_run_context", lambda *a, **k: {
        "feature_run_id": "fixture", "feature_root": tmp_path, "feature_schema": {}, "selection": "explicit"})
    monkeypatch.setattr(run_relevant_models, "resolve_runtime_return_type", lambda *a: "next_3")
    monkeypatch.setattr(run_relevant_models, "load_feature_run_metadata", lambda *a, **k: {})
    monkeypatch.setattr(run_relevant_models, "resolve_match_ids", lambda *a, **k: ["match_1"])
    monkeypatch.setattr(run_relevant_models, "load_match", lambda *a, **k: subject)
    monkeypatch.setattr(run_relevant_models, "build_sportec_possessions", lambda *a, **k: iter([(subject, {})]))
    run_relevant_models.main()
    root = tmp_path / "runs/test"
    assert json.loads((root / "metadata.json").read_text())["processed_match_ids"] == ["match_1"]
    for name in skillcorner.COMPONENT_COLUMNS:
        table = pd.read_parquet(root / "match_1" / (name + ".parquet"))
        assert len(table) == 2
        if name == "pass_success" and physical:
            assert table["home_2"].tolist() == [.2, .8]
        else:
            assert table[["home_1", "home_2", "away_3"]].isna().all().all()


