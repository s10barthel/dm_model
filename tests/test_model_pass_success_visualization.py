"""Checkpoint predictions survive blending, export, selection, and rendering."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import inference
import physical_pass_model as physics
from datatools import config
from datatools.pass_success import require_model_pass_success
from scripts import run_and_visualize_hawkeye as direct
from scripts import visualize_action_components as sportec
from scripts import visualize_benchmark as benchmark
from scripts import visualize_hawkeye as hawkeye
from scripts import visualize_skillcorner as skillcorner
from scripts.visualization_comparison import cached_prediction
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection
from test_physical_xpass import (
    ConstantPassHeightModel, make_pass_success_args, make_physical_inference_match,
    DummyBaselineInferenceModel, write_physical_inference_cache,
)
from test_visualization_comparison import inference_fixture, saved_fixture, _raw_hawkeye, _subject
from test_auxiliary_prediction_reuse import pipeline, run_components


@pytest.fixture
def checkpoint(monkeypatch):
    subject = make_physical_inference_match([7, 8], [1, 1])
    for graph in subject.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros((graph.num_nodes, 1))], dim=1)
    model = ConstantPassHeightModel(probability=0.8, node_in_dim=26)
    model.args.update(make_pass_success_args(
        use_physical_xpass=False, inference_use_physical_xpass=True,
        node_in_dim=26, xpass_weight="v3",
        goal_context_version=2, goal_nodes_aware=False, include_goals=False,
    ))
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference",
                        lambda subject, graphs, labels, model, **kw: (graphs, labels))

    def attach(subject, graphs, labels, model, **kw):
        for graph in graphs:
            for name, value in (
                (physics.PHYSICAL_XPASS_PROB_ATTR, .2),
                (physics.PHYSICAL_XPASS_LOGIT_ATTR, -1.386294),
                (physics.PHYSICAL_XPASS_DISTANCE_ATTR, 20.),
                (physics.PHYSICAL_XPASS_NEAREST_OPPONENT_DISTANCE_ATTR, 10.),
                (physics.PHYSICAL_XPASS_PASS_HEIGHT_ATTR, .25),
                (physics.PHYSICAL_XPASS_BALL_Z_ATTR, .9),
            ):
                setattr(graph, name, graph.x.new_full((graph.num_nodes,), value))
        return graphs
    monkeypatch.setattr(inference, "attach_physical_xpass_for_inference", attach)
    calls = []
    model.register_forward_hook(lambda *args: calls.append(1))
    options = dict(device="cpu", graph_override=subject.graph_features_0, label_override=subject.labels,
                   pass_intent_probs=pd.DataFrame({"home_1": [.5, .5], "home_2": [.5, .5]}, index=[7, 8]),
                   pass_height_probs=pd.DataFrame({"home_1": [.25, .25], "home_2": [.25, .25]}, index=[7, 8]))
    return subject, model, options, calls


@pytest.mark.parametrize("version", [None, "v1", "v2", "v3", "v4", "v5"])
@pytest.mark.parametrize("high_ball", [False, True])
def test_model_and_final_probabilities_share_one_forward(checkpoint, version, high_ball):
    subject, model, options, calls = checkpoint
    model.args.update(inference_use_physical_xpass=version is not None, xpass_weight=version or "v3",
                      ball_z_limit=.5 if high_ball else "none")
    baseline, second = inference.inference_gnn(subject, model, **options)
    assert second is None
    calls.clear()
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    pd.testing.assert_frame_equal(final, baseline)
    assert calls == [1]
    assert list(learned.index) == [7, 8]
    assert learned["home_2"].tolist() == pytest.approx([.8, .8])
    expected = .8 if version is None else physics.blend_physical_xpass_predictions(
        pass_success_model=.8, xpass=.2, pass_distance=20., weight_version=version,
        distance_to_nearest_opponent=10., pass_height=.25, pass_intent=.5,
        ball_z=.9, ball_z_limit=.5 if high_ball else None,
    )
    assert final["home_2"].tolist() == pytest.approx([expected, expected])
    assert physics.PHYSICAL_XPASS_PROB_ATTR not in subject.graph_features_0[0]


@pytest.mark.parametrize("is_carry", [False, True])
def test_offside_and_carry_behavior(checkpoint, is_carry):
    subject, model, options, _ = checkpoint
    subject.graph_features_0[0].x[1, -1] = 1
    subject.labels[:, config.LABEL_INDEX["is_dribble"]] = int(is_carry)
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert learned.at[7, "home_2"] == final.at[7, "home_2"] == 0.
    assert learned.at[7, "home_1"] == pytest.approx(.8)
    if is_carry:
        pd.testing.assert_frame_equal(final, learned)


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_blending_states_keep_model_predictions(checkpoint, monkeypatch, all_missing):
    subject, model, options, calls = checkpoint
    def select(subject, graphs, labels, model, **kw):
        if all_missing:
            raise inference.PhysicalXPassNoUsableRowsError("missing cache")
        return graphs[1:], labels[1:]
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference", select)
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert list(final.index) == ([] if all_missing else [8])
    assert list(learned.index) == [7, 8]
    assert learned["home_2"].tolist() == pytest.approx([.8, .8])
    assert calls == [1]
    if all_missing:
        with pytest.raises(inference.PhysicalXPassNoUsableRowsError):
            inference.inference_gnn(subject, model, **options)


def test_missing_physical_candidate_keeps_model_probability(checkpoint, monkeypatch):
    subject, model, options, _ = checkpoint
    original = inference.attach_physical_xpass_for_inference
    def attach(*args, **kwargs):
        graphs = original(*args, **kwargs)
        for graph in graphs:
            getattr(graph, physics.PHYSICAL_XPASS_PROB_ATTR)[1] = float("nan")
        return graphs
    monkeypatch.setattr(inference, "attach_physical_xpass_for_inference", attach)
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert np.isnan(final.at[7, "home_2"])
    assert learned.at[7, "home_2"] == pytest.approx(.8)


def test_checkpoint_internal_physical_inputs_are_preserved(checkpoint, monkeypatch):
    subject, model, options, calls = checkpoint
    model.args.update(use_physical_xpass=True, model_variant="gat_phys_logit_offset")
    model.forward = lambda batch: getattr(batch, physics.PHYSICAL_XPASS_LOGIT_ATTR)
    _, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert learned.at[7, "home_2"] == pytest.approx(.2)
    assert calls == [1]
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference",
                        lambda *a, **k: (_ for _ in ()).throw(inference.PhysicalXPassNoUsableRowsError("missing")))
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert final.empty and learned.empty
    with pytest.raises(inference.PhysicalXPassNoUsableRowsError):
        inference.inference_gnn(subject, model, **options)


def test_other_tasks_keep_second_result_semantics(checkpoint):
    subject, model, options, _ = checkpoint
    model.args.update(task="pass_height", inference_use_physical_xpass=False)
    _, second = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert second is None


@pytest.mark.parametrize("module,required", [
    (sportec, ["--match-id", "m"]), (direct, []), (benchmark, []),
    (hawkeye, []), (skillcorner, ["--match-id", "m", "--index", "1"]),
])
def test_all_cli_selections(module, required):
    for physical in ([], ["--show-physical-xpass"]):
        args = module.parse_args([*required, "--only-model-pass-success", *physical])
        selected = resolve_component_selection(args)
        assert selected.rendered_components == ["model_pass_success", *(["physical_xpass"] if physical else [])]
        assert selected.required_component_groups == ["pass_success"]
        assert not getattr(module.parse_args(required), "show_model_pass_success")


def test_tracking_only_conflict():
    with pytest.raises(SystemExit):
        hawkeye.parse_args(["--only-tracking", "--show-model-pass-success"])


def test_cache_keeps_both_results_and_separates_opt_in(checkpoint):
    subject, model, options, calls = checkpoint
    cache = {}
    cached_prediction(cache, inference.inference_gnn, subject, model, **options)
    final, learned = cached_prediction(cache, inference.inference_gnn, subject, model,
                                      return_model_pass_success=True, **options)
    learned.iloc[0, 0] = -1
    cached_final, cached_model = cached_prediction(cache, inference.inference_gnn, subject, model,
                                                 return_model_pass_success=True, **options)
    assert calls == [1, 1]
    pd.testing.assert_frame_equal(cached_final, final)
    assert cached_model.iloc[0, 0] == pytest.approx(.8)


def test_saved_schema_distinguishes_absence_and_missing():
    require_model_pass_success({"model_pass_success": pd.DataFrame()})
    require_model_pass_success({}, requested=False)
    with pytest.raises(ValueError, match="Regenerate the source component run"):
        require_model_pass_success({})


@pytest.mark.parametrize("mode", ["standard", "loc", "freeze"])
def test_hawkeye_modes_accept_extra_component(mode):
    args = hawkeye.parse_args(["--mode", mode, "--show-model-pass-success"])
    assert "model_pass_success" in resolve_component_selection(args).rendered_components


@pytest.mark.parametrize("module", [benchmark, hawkeye])
def test_saved_model_row_preserves_missing_candidates(module):
    tables = {"model_pass_success": pd.DataFrame({"home_1": [.8], "home_2": [np.nan]}, index=[10])}
    row = module._probs_for_component_frame("model_pass_success", tables, 10)
    assert row["home_1"] == .8 and np.isnan(row["home_2"])
    assert module._probs_for_component_frame("model_pass_success", tables, 11).isna().all()


def test_saved_skillcorner_model_row_preserves_missing_candidates():
    tables = {"model_pass_success": pd.DataFrame({"index": [1], "1": [.8], "2": [np.nan]}, index=[10])}
    row = skillcorner._probs_for_component_frame("model_pass_success", tables, 10)
    assert row["home_1"] == .8 and np.isnan(row["home_2"])


@pytest.mark.parametrize("comparison", ["checkpoints", "masks"])
def test_direct_comparison_exports_model_result(tmp_path, monkeypatch, inference_fixture, comparison):
    subject, models, calls, _ = inference_fixture
    previous = direct.inference_gnn
    def predict(scene, model, **kwargs):
        final, second = previous(scene, model, **kwargs)
        return final, final + .1 if kwargs.get("return_model_pass_success") else second
    monkeypatch.setattr(direct, "inference_gnn", predict)
    flags = (["--compare", "--pass-success-model-id", "pass_success/b"] if comparison == "checkpoints"
             else ["--inference-mask-1", "accel"])
    args = direct.parse_args(["--accept-comparison-warnings", "--pass-success-model-id", "pass_success/a",
                             "--only-model-pass-success",
                             "--output-dir", str(tmp_path), "--run-id", "direct", *flags])
    monkeypatch.setattr(direct, "parse_args", lambda: args)
    direct.main()
    root = tmp_path / "direct"
    assert (root / "s/model_pass_success_0.png").exists()
    predictions = pd.read_csv(root / "aligned_predictions.csv")
    assert "model_pass_success" in set(predictions.component)
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["show_model_pass_success"] and metadata["model_pass_success_output_paths"]


@pytest.mark.parametrize("missing", [False, True])
def test_saved_comparison_extra_component(tmp_path, monkeypatch, saved_fixture, missing):
    def load(path):
        raw = _raw_hawkeye()
        if not missing:
            raw["model_pass_success"] = [.8, np.nan]
        return raw, {"status": "completed", "processed_situation_ids": ["s"], "freeze_ballreceipt": True}
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", load)
    args = hawkeye.parse_args(["--accept-comparison-warnings", "--compare", "--component-run-id", "a",
                              "--component-run-id", "b", "--only-model-pass-success", "--output-dir", str(tmp_path), "--run-id", "saved"])
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    if missing:
        with pytest.raises(ValueError, match="Regenerate the source component run"):
            hawkeye.main()
    else:
        hawkeye.main()
        assert (tmp_path / "saved/s/model_pass_success_0.png").exists()
        values = pd.read_csv(tmp_path / "saved/aligned_predictions.csv")
        assert np.isnan(values.loc[values.candidate_id.eq("home_2"), "prediction:a"].iloc[0])


def test_hawkeye_export_round_trip(tmp_path):
    from datatools.hawkeye import build_hawkeye_export, build_hawkeye_component_tables
    subject = _subject()
    rows = pd.DataFrame(dict(id=["s"] * 4, frame_id=[10, 10, 11, 11], abs_time=[0., 0., .1, .1],
                             team=["A"] * 4, possession_team=["A"] * 4, uefa_player_id=[1, 2, 1, 2],
                             object_id=["home_1", "home_2"] * 2))
    learned = pd.DataFrame({"home_1": [.8, .7], "home_2": [np.nan, .9]}, index=[10, 11])
    final = learned.fillna(.2) * .5
    export = build_hawkeye_export(rows, subject, {"pass_success": final, "model_pass_success": learned})
    path = tmp_path / "components.parquet"
    export.to_parquet(path, index=False)
    tables = build_hawkeye_component_tables(pd.read_parquet(path), subject)
    np.testing.assert_allclose(tables["model_pass_success"], learned, equal_nan=True)
    np.testing.assert_allclose(tables["pass_success"], final)
    # Legacy columns remain optional until the new view is explicitly requested.
    legacy = build_hawkeye_component_tables(export.drop(columns="model_pass_success"), subject)
    assert legacy["model_pass_success"].empty


def test_benchmark_export_round_trip(tmp_path):
    from datatools.benchmark import build_benchmark_state, build_benchmark_export, build_benchmark_component_tables
    source = pd.DataFrame(dict(team=[0, 1, 1, 2], player=[0, 1, 2, 3], pos_x=[0., 0., 10., 20.],
        pos_y=[0., 0., 5., 10.], pos_z=[0., np.nan, np.nan, np.nan], smooth_x_speed=[0.] * 4,
        smooth_y_speed=[0.] * 4, event_player=[1] * 4, playing_direction_event=[True] * 4))
    subject, rows, _ = build_benchmark_state(source, 1, 1, 2, build_graphs=False)
    learned = pd.DataFrame({"home_1": [.8], "home_2": [np.nan]}, index=[0])
    export = build_benchmark_export(rows, subject, {"model_pass_success": learned})
    path = tmp_path / "components.parquet"
    export.to_parquet(path, index=False)
    tables = build_benchmark_component_tables(pd.read_parquet(path), subject)
    assert tables["model_pass_success"].at[0, "home_1"] == .8
    assert np.isnan(tables["model_pass_success"].at[0, "home_2"])


def test_skillcorner_export_round_trip_and_legacy_loading(tmp_path):
    from datatools.skillcorner import build_skillcorner_component_table, load_skillcorner_component_tables, COMPONENT_COLUMNS
    subject = _subject()
    subject.frame_meta = subject.frame_meta.assign(match_id="s", index=1, period=1, player_id=1, attacking_side="home")
    subject.actions = pd.DataFrame(index=[10, 11])
    learned = pd.DataFrame({"home_1": [.8, .7], "home_2": [np.nan, .9]}, index=[10, 11])
    table = build_skillcorner_component_table(subject, learned, pd.DataFrame({"player_id": [1, 2]}), include_shot=False)
    folder = tmp_path / "s"
    folder.mkdir()
    for component in COMPONENT_COLUMNS:
        table.to_parquet(folder / f"{component}.parquet", index=False)
    tables = load_skillcorner_component_tables(tmp_path, "s")
    require_model_pass_success(tables)
    assert tables["model_pass_success"].loc[0, "1"] == .8
    assert np.isnan(tables["model_pass_success"].loc[0, "2"])
    (folder / "model_pass_success.parquet").unlink()
    legacy = load_skillcorner_component_tables(tmp_path, "s")
    assert "model_pass_success" not in legacy
    with pytest.raises(ValueError, match="Regenerate"):
        require_model_pass_success(legacy)


def test_sportec_export_keeps_model_only_state_identity(checkpoint, tmp_path):
    from datatools.sportec_possessions import sportec_component_table
    subject, model, options, _ = checkpoint
    subject.actions = subject.actions.assign(stats_perform_match_id="s", action_id=[1, 2], original_event_id=[3, 4])
    _, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    table = sportec_component_table(learned, subject)
    path = tmp_path / "model_pass_success.parquet"
    table.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    assert restored.action_id.tolist() == [1, 2]
    assert restored["home_2"].tolist() == pytest.approx([.8, .8])


@pytest.mark.parametrize("dataset", ["benchmark", "hawkeye", "skillcorner"])
def test_producers_retain_both_predictions(checkpoint, monkeypatch, dataset):
    from importlib import import_module
    subject, model, _, calls = checkpoint
    module = import_module("datatools." + dataset)
    models = {task: model if task == "pass_success" else type("Model", (), {"args": {"task": task}})()
              for task in ("action_intent", "pass_intent", "pass_success", "outcome_scoring", "outcome_conceding")}
    previous = inference.inference_gnn
    def predict(scene, selected, **kwargs):
        if selected.args["task"] == "pass_success":
            return previous(scene, selected, **kwargs)
        values = pd.DataFrame({"home_1": [.5, .5], "home_2": [.5, .5]}, index=[7, 8])
        return values, values if selected.args["task"].startswith("outcome_") else None
    monkeypatch.setattr(module, "inference_gnn", predict)
    result = getattr(module, f"infer_{dataset}_components")(subject, models, device="cpu")
    assert result["model_pass_success"].at[7, "home_2"] == pytest.approx(.8)
    assert result["pass_success"].at[7, "home_2"] != pytest.approx(.8)
    assert calls == [1]


@pytest.mark.parametrize("output", ["gif", "mp4"])
def test_direct_animation_includes_model_component(tmp_path, monkeypatch, inference_fixture, output):
    subject, models, _, _ = inference_fixture
    subject.frame_meta["frame_id"] = subject.frame_meta.index
    tracking = pd.DataFrame({"id": ["s"]})
    ball = pd.DataFrame()
    from test_visualization_comparison import TinyModel
    model = TinyModel("pass_success/a")
    model.eval()
    previous = direct.inference_gnn
    def predict(scene, model, **kwargs):
        values, _ = previous(scene, model, **kwargs)
        return values, values + .1
    monkeypatch.setattr(direct, "inference_gnn", predict)
    written = []
    def save(images, path, **kwargs):
        frames = list(images)
        assert len(frames) == 2
        assert all(image.width > 0 for image in frames)
        written.append(path)
    monkeypatch.setattr(direct, "save_animation", save)
    args = direct.parse_args(["--only-model-pass-success", "--output", output])
    direct.render_situation("s", tracking, ball, {"pass_success": model},
                            {"add_v_edge_features": False}, args, "cpu", tmp_path, ["model_pass_success"])
    assert [path.name for path in written] == [f"model_pass_success.{output}"]


def test_real_cache_subset_preserves_final_result(tmp_path):
    subject = make_physical_inference_match([7, 8], [1, 1])
    write_physical_inference_cache(tmp_path, [7])
    model = DummyBaselineInferenceModel(tmp_path)
    baseline, _ = inference.inference_gnn(subject, model, device="cpu")
    calls = []
    model.register_forward_hook(lambda *a: calls.append(1))
    final, learned = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    pd.testing.assert_frame_equal(final, baseline)
    assert list(learned.index) == [7, 8]
    assert learned["home_2"].tolist() == pytest.approx([.9, .9])
    assert calls == [1]


def test_missing_model_comparison_renders_unavailable_panel(tmp_path, monkeypatch, inference_fixture):
    monkeypatch.setattr(direct, "inference_gnn", lambda *a, **kw: (pd.DataFrame(), pd.DataFrame()))
    args = direct.parse_args(["--accept-comparison-warnings", "--compare",
                             "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b",
                             "--only-model-pass-success",
                             "--output-dir", str(tmp_path), "--run-id", "unavailable"])
    monkeypatch.setattr(direct, "parse_args", lambda: args)
    direct.main()
    assert (tmp_path / "unavailable/s/model_pass_success_0.png").exists()


@pytest.mark.parametrize("mode", ["loc", "freeze"])
@pytest.mark.parametrize("compare", [False, True])
def test_location_model_view_renders(tmp_path, monkeypatch, saved_fixture, mode, compare):
    def load(path):
        raw = _raw_hawkeye()
        raw["model_pass_success"] = [.8, np.nan]
        raw["selection_row_id"], raw["original_action_id"] = 1, "s"
        raw["PositionX"], raw["PositionY"] = 200., 300.
        raw["resolved_time_norm"], raw["loc_situation_id"] = 0., "cache_identity"
        return raw, {"status": "completed", "inference_mode": mode}
    monkeypatch.setattr(hawkeye, "get_hawkeye_loc_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(hawkeye, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", load)
    monkeypatch.setattr(hawkeye, "apply_hawkeye_possessor_offset", lambda tracking, **kw: (tracking, {}))
    monkeypatch.setattr(hawkeye, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    flags = ["--compare", "--component-run-id", "b"] if compare else []
    args = hawkeye.parse_args(["--accept-comparison-warnings", "--mode", mode, "--component-run-id", "a",
                              "--physical-cache-dir", str(tmp_path / "cache"), "--only-model-pass-success", "--output-dir", str(tmp_path), "--run-id", "location", *flags])
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    hawkeye.main()
    assert (tmp_path / "location/s_1/model_pass_success.png").exists()


@pytest.mark.parametrize("compare", [False, True])
def test_benchmark_model_view_renders_paired_states(tmp_path, monkeypatch, compare):
    from datatools.benchmark import build_benchmark_state
    source = pd.DataFrame(dict(team=[0, 1, 1, 2], player=[0, 1, 2, 3], pos_x=[0., 0., 10., 20.],
        pos_y=[0., 0., 5., 10.], pos_z=[0., np.nan, np.nan, np.nan], smooth_x_speed=[0.] * 4,
        smooth_y_speed=[0.] * 4, event_player=[1] * 4, playing_direction_event=[True] * 4))
    frames = []
    for state_id in (1, 2):
        _, export, _ = build_benchmark_state(source, 1, state_id, 2, build_graphs=False)
        export["model_pass_success"] = np.where(export.team.eq(1), .8, np.nan)
        frames.append(export)
    raw = pd.concat(frames, ignore_index=True)
    monkeypatch.setattr(benchmark, "get_benchmark_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(benchmark, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(benchmark, "load_benchmark_component_run", lambda path: (raw, {"status": "completed"}))
    monkeypatch.setattr(benchmark, "load_benchmark_modification_data", lambda *a: {
        "game_state_1": source, "game_state_2": source, "higher_state_id": 2})
    flags = ["--compare", "--component-run-id", "b"] if compare else []
    args = benchmark.parse_args(["--accept-comparison-warnings", "--component-run-id", "a", "--only-model-pass-success", "--output-dir", str(tmp_path),
                                "--run-id", "benchmark", *flags])
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    assert (tmp_path / "benchmark/modification_1/model_pass_success.png").exists()


@pytest.mark.parametrize("compare", [False, True])
def test_skillcorner_model_view_renders(tmp_path, monkeypatch, compare):
    subject = _subject()
    subject.event_index, subject.fps = 1, 25
    table = pd.DataFrame({"index": [1, 1], "frame": [10, 11], "1": [.8, .8], "2": [np.nan, .9]})
    monkeypatch.setattr(skillcorner, "get_skillcorner_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(skillcorner, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(skillcorner, "load_run_metadata", lambda *a, **kw: {"status": "completed"})
    monkeypatch.setattr(skillcorner, "load_skillcorner_component_tables", lambda *a, **kw: {"model_pass_success": table})
    monkeypatch.setattr(skillcorner, "build_skillcorner_match_context", lambda *a: object())
    monkeypatch.setattr(skillcorner, "build_skillcorner_possession", lambda *a: (subject, {}))
    flags = ["--compare", "--component-run-id", "b"] if compare else []
    args = skillcorner.parse_args(["--accept-comparison-warnings", "--match-id", "s", "--index", "1",
                                  "--component-run-id", "a", "--only-model-pass-success", "--output-dir", str(tmp_path), "--run-id", "sc", *flags])
    monkeypatch.setattr(skillcorner, "parse_args", lambda: args)
    skillcorner.main()
    assert (tmp_path / "sc/s/1/model_pass_success_first.png").exists()
    assert (tmp_path / "sc/s/1/model_pass_success_last.png").exists()


def test_sportec_model_comparison_renders(tmp_path, monkeypatch, inference_fixture):
    subject, _, _, _ = inference_fixture
    subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"], spadl_type=["pass"],
        action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
    monkeypatch.setattr(sportec, "resolve_runtime_feature_run_context", lambda *a, **kw: {
        "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
    monkeypatch.setattr(sportec, "load_match", lambda *a, **kw: subject)
    monkeypatch.setattr(sportec, "resolve_action_indices", lambda *a: [(10, "42")])
    monkeypatch.setattr(sportec, "configure_lane_survival_runtime_cache", lambda *a: None)
    previous = direct.inference_gnn
    def predict(scene, model, **kwargs):
        values, _ = previous(scene, model, **kwargs)
        return values, values + .1
    monkeypatch.setattr(sportec, "inference_gnn", predict)
    args = sportec.parse_args(["--accept-comparison-warnings", "--match-id", "s", "--compare",
                              "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b",
                              "--only-model-pass-success",
                              "--output-dir", str(tmp_path), "--run-id", "sp"])
    monkeypatch.setattr(sportec, "parse_args", lambda: args)
    sportec.main()
    assert (tmp_path / "sp/s/42/model_pass_success.png").exists()


def test_capture_reuses_pc_xpass_auxiliaries(pipeline):
    subject, models, calls = pipeline
    records = {}
    run_components(subject, models, records)
    final, learned = inference.inference_gnn(subject, models["pass_success"], device="cpu",
                                           return_model_pass_success=True, precomputed_auxiliaries=records)
    assert not final.empty and not learned.empty
    assert calls["pass_success"] == calls["pass_height"] == calls["pass_intent"] == 1
    assert calls["xpass_pass_height"] == calls["xpass_pass_intent"] == 0


def test_lane_survival_checkpoint_keeps_required_input_coverage(checkpoint, monkeypatch):
    subject, model, options, calls = checkpoint
    monkeypatch.setattr(inference, "model_requires_lane_survival", lambda model: True)
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference",
                        lambda subject, graphs, labels, model, **kw: (graphs[1:], labels[1:]))
    def attach(subject, graphs, labels, model):
        assert labels[:, 0].tolist() == [8]
        return graphs
    monkeypatch.setattr(inference, "attach_lane_survival_for_inference", attach)
    final, learned = inference.inference_gnn(subject, model, return_model_pass_success=True, **options)
    assert list(final.index) == list(learned.index) == [8]
    assert calls == [1]
