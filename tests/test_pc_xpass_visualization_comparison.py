"""Cache comparisons keep checkpoints/settings fixed and isolate runtime cache inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

import inference
import physical_pass_model as physics
import project_config
import pc_xpass_match_cache as match_cache
from scripts import inference_visualization_comparison as direct_compare
from scripts import saved_visualization_comparison as saved_compare
from scripts import run_and_visualize_hawkeye as direct
from scripts import visualize_action_components as sportec
from scripts import visualize_hawkeye as hawkeye
from scripts import visualize_benchmark as benchmark
from scripts import visualize_skillcorner as skillcorner
from scripts.visualization_comparison import Variant, cached_prediction
from test_saved_visualization_xpass import write_cache, component
from test_visualization_comparison import (
    TinyModel, _raw_hawkeye, inference_fixture, saved_fixture, outcome_pipeline,
)
from test_only_physical_xpass_visualization import setup_saved


COMMANDS = [(direct, []), (sportec, ["--match-id", "s"]), (hawkeye, []),
            (benchmark, []), (skillcorner, ["--match-id", "s", "--index", "1"])]
CACHE_FLAGS = ["--compare", "--pc-xpass-id", "a", "--pc-xpass-id", "b"]


@pytest.fixture
def caches(tmp_path, monkeypatch):
    root = tmp_path / "versions"
    monkeypatch.setattr(project_config, "PC_XPASS_DIR", root)
    for identifier, value in (("a", .2), ("b", .8)):
        write_cache(root / identifier)
        for dataset in ("hawkeye", "sportec", "skillcorner", "benchmark", "hawkeye_loc"):
            folder = (root / "hawkeye_loc" / identifier if dataset == "hawkeye_loc" else root / identifier / dataset)
            write_cache(folder, cache_format=(match_cache.SPORTEC_FORMAT if dataset == "sportec" else
                                             match_cache.FORMAT if dataset == "skillcorner" else None),
                        source_inputs={"feature_run_id": "feature"})
            (folder / "matches").mkdir()
            rows = []
            for frame in (10, 11):
                row = {"match_id": "s", "action_index": frame, "frame_scope": "frame_id", "ball_z": 0.}
                for player in ("home_1", "home_2", "away_3"):
                    row.update({f"{player}__{metric}": value for metric in
                                ("max_xpass", "top10_xpass", "top25_xpass", "top_pass10_xpass")})
                    row.update({f"{player}__distance": 20., f"{player}__distance_to_nearest_opponent": 5.,
                                f"{player}__lane_survival": value})
                rows.append(row)
            pd.DataFrame(rows).to_parquet(folder / "matches/s.parquet", index=False)
    # An invalid latest pointer must never affect explicit comparisons.
    (root / "latest.json").write_text('{"run_id":"missing"}')
    return root


@pytest.mark.parametrize("module,required", COMMANDS)
def test_cli_cache_comparison_and_single_id(module, required):
    single = module.parse_args([*required, "--pc-xpass-id", "a"])
    assert single.pc_xpass_id == "a" and single.visualization_comparison is None
    args = module.parse_args([*required, *CACHE_FLAGS, "--top-pass", "10"])
    assert args.pc_xpass_id == ["a", "b"] and args.pc_xpass
    assert args.visualization_comparison == "pc_xpass" and args.x_pass_version == "top-pass10"


@pytest.mark.parametrize("module,required", COMMANDS)
@pytest.mark.parametrize("extra", [
    ["--pc-xpass-id", "a", "--pc-xpass-id", "b"],
    ["--compare", "--pc-xpass-id", "a", "--pc-xpass-id", "a"],
    [*CACHE_FLAGS, "--physical-cache-dir", "cache"],
    [*CACHE_FLAGS, "--compare-pass-success"],
    [*CACHE_FLAGS, "--only-compare-pass-success"],
    ["--pc-xpass-ids", "a", "b"],
])
def test_cli_invalid_cache_comparisons(module, required, extra):
    with pytest.raises((SystemExit, ValueError)):
        module.parse_args([*required, *extra])


@pytest.mark.parametrize("module,required", COMMANDS[:2])
def test_inference_fixed_models_masks_and_conflicts(module, required):
    fixed = ["--pass-success-model-id", "pass_success/fixed", "--bundle-id", "fixed",
             "--inference-mask", "pass_success:accel", "--xpass-weight", "v5",
             "--xpass-pass-height-model-id", "pass_height/aux", "--xpass-pass-intent-model-id", "pass_intent/aux"]
    args = module.parse_args([*required, *CACHE_FLAGS, *fixed])
    assert args.xpass_weight == "v5" and args.pass_success_model_id == "pass_success/fixed"
    for extra in (["--pass-success-model-id", "pass_success/alternative"],
                  ["--inference-mask-1", "accel"], ["--no-physical-cache"],
                  ["--lane-survival-cache-dir", "cache"],
                  *([["--output", "gif"]] if module is direct else [])):
        with pytest.raises((SystemExit, ValueError)):
            module.parse_args([*required, *CACHE_FLAGS, *fixed, *extra])


@pytest.mark.parametrize("module,required", COMMANDS[2:])
def test_saved_fixed_source_and_conflicts(module, required):
    for source in (["--component-run-id", "fixed"], ["--component-dir", "folder"], []):
        args = module.parse_args([*required, *CACHE_FLAGS, *source])
        assert args.visualization_comparison == "pc_xpass"
    with pytest.raises(SystemExit):
        module.parse_args([*required, *CACHE_FLAGS, "--component-run-id", "fixed", "--component-run-id", "other"])
    if module is not benchmark:
        with pytest.raises(SystemExit):
            module.parse_args([*required, *CACHE_FLAGS, "--output", "gif"])
    if module is hawkeye:
        with pytest.raises(SystemExit):
            module.parse_args([*CACHE_FLAGS, "--only-tracking"])


@pytest.mark.parametrize("module,required", COMMANDS[2:])
def test_saved_physical_only_comparison_without_component_source(tmp_path, monkeypatch, saved_fixture, caches, module, required):
    setup_saved(module, tmp_path, monkeypatch, saved_fixture)
    def physical(folder, match, frames, **kwargs):
        value = .2 if Path(folder).parent.name == "a" else .8
        return pd.DataFrame({"home_1": [0.] * len(frames), "home_2": [value] * len(frames)}, index=frames)
    loader = Mock(side_effect=physical)
    monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", loader)
    monkeypatch.setattr(module, "render_state_image" if module is benchmark else "render_frame_image",
                        lambda *a, **kw: Image.new("RGB", (80, 60)))
    flags = [*required, *CACHE_FLAGS, "--only-physical-xpass", "--top-pass", "10",
             "--output-dir", str(tmp_path), "--run-id", "result"]
    if module is benchmark:
        flags += ["--input-dir", str(tmp_path / "input")]
    args = module.parse_args(flags)
    saved_compare.run_comparison(args, module)
    root = tmp_path / "result"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["comparison_mode"] == "pc_xpass" and metadata["status"] == "completed"
    assert [v["id"] for v in metadata["variants"]] == ["a", "b"]
    assert all(v["provenance"]["component_run_id"] is None for v in metadata["variants"])
    assert all(v["provenance"]["physical_xpass_metric"] == "top_pass10_xpass" for v in metadata["variants"])
    table = pd.read_csv(root / "aligned_predictions.csv")
    values = table.loc[table.candidate_id.eq("home_2")]
    assert values["prediction:a"].tolist() == pytest.approx([.2] * len(values))
    assert values["prediction:b"].tolist() == pytest.approx([.8] * len(values))
    assert len(metadata["output_paths"]) == (2 if module is skillcorner else 1)


@pytest.mark.parametrize("mode", ["standard", "loc", "freeze"])
@pytest.mark.parametrize("source", ["id", "directory", "default"])
def test_saved_source_loaded_once_with_shared_metric_and_unchanged_predictions(
        tmp_path, monkeypatch, saved_fixture, caches, mode, source):
    raw = _raw_hawkeye(.4)
    for name, value in (("pass_success", .7), ("outcome_scoring_success", .8),
                        ("outcome_scoring_failure", .1), ("outcome_conceding_success", .1),
                        ("outcome_conceding_failure", .02)):
        raw[name] = [0., value]
    if mode != "standard":
        raw["selection_row_id"], raw["original_action_id"] = 1, "s"
        raw["PositionX"], raw["PositionY"] = 200., 300.
        raw["resolved_time_norm"], raw["loc_situation_id"] = 0., "s"
        monkeypatch.setattr(hawkeye, "get_hawkeye_loc_component_run_root", lambda identifier: Path(identifier))
        monkeypatch.setattr(hawkeye, "apply_hawkeye_possessor_offset", lambda tracking, **kw: (tracking, {}))
    metadata = component(inference_mode=mode, processed_situation_ids=["s"], freeze_ballreceipt=True)
    load = Mock(return_value=(raw, metadata))
    monkeypatch.setattr(hawkeye, "resolve_named_component_run_id", lambda *a, **kw: "fixed")
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", load)
    monkeypatch.setattr(hawkeye, "render_frame_image", lambda *a, **kw: Image.new("RGB", (80, 60)))
    flags = {"id": ["--component-run-id", "fixed"], "directory": ["--component-dir", "fixed"], "default": []}[source]
    args = hawkeye.parse_args([*CACHE_FLAGS, *flags, "--mode", mode, "--only-pass-intent",
        "--only-pass-success", "--only-pass-score", "--show-physical-xpass",
        "--output-dir", str(tmp_path), "--run-id", "saved"])
    saved_compare.run_comparison(args, hawkeye)
    assert load.call_count == 1
    result = json.loads((tmp_path / "saved/metadata.json").read_text())
    namespace = "normal" if mode == "standard" else "hawkeye_loc"
    assert all(v["provenance"]["pc_xpass_namespace"] == namespace for v in result["variants"])
    assert all(v["provenance"]["x_pass_version"] == "top25" for v in result["variants"])
    aligned = pd.read_csv(tmp_path / "saved/aligned_predictions.csv")
    intent = aligned.loc[aligned.component.eq("pass_intent") & aligned.candidate_id.eq("home_2")].iloc[0]
    assert intent["prediction:a"] == intent["prediction:b"] == .4
    for name, expected in (("pass_success", .7), ("pass_score", .7 * .7 + .3 * .08)):
        row = aligned.loc[aligned.component.eq(name) & aligned.candidate_id.eq("home_2")].iloc[0]
        assert row["prediction:a"] == pytest.approx(expected)
        assert row["prediction:b"] == pytest.approx(expected)
    physical = aligned.loc[aligned.component.eq("physical_xpass") & aligned.candidate_id.eq("home_2")].iloc[0]
    assert physical["prediction:a"] == .2 and physical["prediction:b"] == .8


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_inference_blending_and_derived_score_are_cache_specific(
        tmp_path, monkeypatch, outcome_pipeline, caches, order):
    from scripts import visualization_comparison as comparison
    module, required, models, _, build, render = outcome_pipeline
    compose = Mock(wraps=comparison.compose_panels)
    monkeypatch.setattr(comparison, "compose_panels", compose)
    flags = ["--compare", *[token for identifier in order for token in ("--pc-xpass-id", identifier)]]
    args = module.parse_args([*required, *flags, "--only-pass-score", "--only-pass-success", "--show-physical-xpass",
        "--pass-success-model-id", "pass_success/fixed", "--outcome-scoring-model-id", "outcome_scoring/a",
        "--outcome-conceding-model-id", "outcome_conceding/x", "--use-physical-xpass", "--xpass-weight", "v3",
        "--inference-mask", "all:accel", "--output-dir", str(tmp_path), "--run-id", "inference"])
    calls = []
    def predict(scene, model, **kw):
        task = model.args["task"]
        calls.append((task, model.args.get("pc_xpass_id")))
        if task == "pass_success":
            cached = pd.read_parquet(Path(model.args["physical_cache_dir"]) / "matches/s.parquet")
            value = physics.blend_physical_xpass_predictions(pass_success_model=.6,
                xpass=float(cached.iloc[0]["home_2__top10_xpass"]), pass_distance=20., weight_version=model.args["xpass_weight"])
            return pd.DataFrame({"home_1": [0.], "home_2": [value]}, index=[10]), None
        values = (.1, .8) if task == "outcome_scoring" else (.02, .1)
        return tuple(pd.DataFrame({"home_1": [0.], "home_2": [value]}, index=[10]) for value in values)
    monkeypatch.setattr(module, "inference_gnn", predict)
    def physical(folder, match, frames, **kw):
        return physics.load_runtime_physical_xpass_visualization_table(folder, "s", frames,
            metric="top10_xpass", x_pass_version="top10")
    if module is direct:
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", physical)
    else:
        from datatools import sportec_possessions
        subject = build.return_value
        monkeypatch.setattr(sportec_possessions, "build_sportec_possessions", lambda *a, **kw: [])
        monkeypatch.setattr(module, "possession_state_for_action", lambda *a: (subject, 10))
        monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_component", lambda folder, *a, **kw:
            pd.Series({"home_1": 0., "home_2": float(pd.read_parquet(Path(folder) / "matches/s.parquet").iloc[0]["home_2__top10_xpass"])}))
    direct_compare.run_comparison(args, module)
    aligned = pd.read_csv(tmp_path / "inference/aligned_predictions.csv")
    for identifier, raw_value in (("a", .2), ("b", .8)):
        expected = physics.blend_physical_xpass_predictions(pass_success_model=.6, xpass=raw_value,
                                                           pass_distance=20., weight_version="v3")
        for name, value in (("physical_xpass", raw_value), ("pass_success", expected),
                            ("pass_score", expected * .7 + (1 - expected) * .08)):
            row = aligned.loc[aligned.component.eq(name) & aligned.candidate_id.eq("home_2")].iloc[0]
            assert row["prediction:" + identifier] == pytest.approx(value)
    assert calls.count(("outcome_scoring", None)) == calls.count(("outcome_conceding", None)) == 1
    assert [identifier for task, identifier in calls if task == "pass_success"] == list(order)
    assert len(models) == 3 and build.call_count == 1
    for model in models.values():
        assert "physical_cache_dir" not in model.args and "pc_xpass_id" not in model.args
        assert getattr(model, "inference_masks", None) is None
    metadata = json.loads((tmp_path / "inference/metadata.json").read_text())
    assert [v["id"] for v in metadata["variants"]] == list(order)
    assert compose.call_count == 3
    assert all([label.splitlines()[0] for label in call.args[1]] == list(order) for call in compose.call_args_list)
    assert all(v["provenance"]["effective_model_args"]["pass_success"]["xpass_weight"] == "v3" for v in metadata["variants"])


@pytest.mark.parametrize("failure", ["missing_cache", "wrong_source", "missing_metric", "missing_rows"])
def test_missing_cache_inputs_fail_with_cache_identity(tmp_path, monkeypatch, saved_fixture, caches, failure):
    folder = caches / "b/hawkeye"
    if failure == "missing_cache":
        (folder / "metadata.json").unlink()
    elif failure == "wrong_source":
        write_cache(folder, source="other")
    elif failure == "missing_metric":
        write_cache(folder, available_metrics=["max_xpass"])
    else:
        (folder / "matches/s.parquet").unlink()
    args = hawkeye.parse_args([*CACHE_FLAGS, "--only-physical-xpass", "--xpass-version", "top25",
                              "--output-dir", str(tmp_path), "--run-id", "failure"])
    with pytest.raises((FileNotFoundError, ValueError, KeyError), match="cache 'b'"):
        saved_compare.run_comparison(args, hawkeye)
    assert not list((tmp_path / "failure").rglob("*.png"))


def test_lane_survival_and_auxiliary_state_is_scoped_and_restored(tmp_path, monkeypatch, caches):
    primary = TinyModel("pass_success/fixed")
    height = TinyModel("pass_height/aux")
    intent = TinyModel("pass_intent/aux")
    for model in (primary, height, intent):
        model.args.update(lane_survival=True, lane_survival_mode="max", model_id=model.identifier)
    models = {"pass_success": primary, "xpass_pass_height": height, "xpass_pass_intent": intent}
    originals = {key: model.args for key, model in models.items()}
    variants = [Variant(identifier, identifier, models=models) for identifier in ("a", "b")]
    args = direct.parse_args([*CACHE_FLAGS, "--use-physical-xpass"])
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    scene, memo, calls = SimpleNamespace(), {}, []
    def predict(scene, model, **kw):
        calls.append(model.args["pc_xpass_id"])
        cached = pd.read_parquet(Path(model.args["lane_survival_runtime_cache_dir"]) / "matches/s.parquet")
        return pd.DataFrame({"home_2": [float(cached.iloc[0]["home_2__lane_survival"])]}), None
    for variant in [variants[0], variants[1], variants[0]]:
        with direct_compare.active_variant(variant, "test scene"):
            for model in models.values():
                assert model.args["pc_xpass_id"] == variant.identifier
                assert model.args["lane_survival_runtime_cache_dir"] == str(caches / variant.identifier / "hawkeye")
            assert primary._xpass_auxiliary_models == {"pass_height": height, "pass_intent": intent}
            value = cached_prediction(memo, predict, scene, primary)[0].iloc[0, 0]
            assert value == (.2 if variant.identifier == "a" else .8)
        assert all(model.args is originals[key] for key, model in models.items())
    assert calls == ["a", "b"]
    with pytest.raises(ValueError, match="cache 'b'.*test scene"):
        with direct_compare.active_variant(variants[1], "test scene"):
            raise ValueError("missing auxiliary column")
    assert all(model.args is originals[key] for key, model in models.items())
    assert not hasattr(primary, "_xpass_auxiliary_models")


@pytest.mark.parametrize("module,required", COMMANDS[:2])
def test_inference_physical_only_needs_no_models(tmp_path, monkeypatch, inference_fixture, caches, module, required):
    subject, _, _, build = inference_fixture
    monkeypatch.setattr(direct_compare, "load_model", lambda *a, **kw: pytest.fail("loaded a model"))
    monkeypatch.setattr(direct_compare, "configure_auxiliary_models", lambda *a, **kw: pytest.fail("loaded auxiliaries"))
    monkeypatch.setattr(module, "collect_situation_components" if module is direct else "inference_gnn",
                        lambda *a, **kw: pytest.fail("ran learned inference"))
    if module is sportec:
        from datatools import sportec_possessions
        subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"], spadl_type=["pass"],
            action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
        monkeypatch.setattr(module, "resolve_runtime_feature_run_context", lambda *a, **kw: {
            "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
        monkeypatch.setattr(module, "load_match", lambda *a, **kw: subject)
        monkeypatch.setattr(module, "resolve_action_indices", lambda *a: [(10, "42")])
        monkeypatch.setattr(sportec_possessions, "build_sportec_possessions", lambda *a, **kw: [])
        monkeypatch.setattr(module, "possession_state_for_action", lambda *a: (subject, 10))
        monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_component", lambda folder, *a, **kw:
            pd.Series({"home_1": 0., "home_2": float(pd.read_parquet(Path(folder) / "matches/s.parquet").iloc[0]["home_2__top10_xpass"])}))
    monkeypatch.setattr(module, "render_frame_image" if module is direct else "render_component",
                        lambda *a, **kw: Image.new("RGB", (80, 60)))
    args = module.parse_args([*required, *CACHE_FLAGS, "--only-physical-xpass",
                              "--output-dir", str(tmp_path), "--run-id", "raw"])
    direct_compare.run_comparison(args, module)
    table = pd.read_csv(tmp_path / "raw/aligned_predictions.csv")
    row = table.loc[table.candidate_id.eq("home_2")].iloc[0]
    assert row["prediction:a"] == .2 and row["prediction:b"] == .8
    metadata = json.loads((tmp_path / "raw/metadata.json").read_text())
    assert all(not v["provenance"]["model_ids"] for v in metadata["variants"])
    if module is direct:
        assert build.call_args.kwargs["build_graphs"] is False


@pytest.mark.parametrize("missing", [None, "row", "all_rows", "player", "nearest"])
def test_real_inference_matches_standalone_and_rejects_incomplete_cache(caches, monkeypatch, missing):
    from test_physical_xpass import DummyBaselineInferenceModel, make_physical_inference_match
    subject = make_physical_inference_match([10, 11], [1, 1])
    subject.match_id = "s"
    model = DummyBaselineInferenceModel(caches / "a/hawkeye")
    model.args.update(pc_xpass=True, x_pass_version="top10", xpass_weight="v2" if missing == "nearest" else "v3")
    variants = [Variant(identifier, identifier, models={"pass_success": model}) for identifier in ("a", "b")]
    args = direct.parse_args([*CACHE_FLAGS, "--use-physical-xpass", "--xpass-weight", model.args["xpass_weight"]])
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    if missing:
        path = caches / "b/hawkeye/matches/s.parquet"
        rows = pd.read_parquet(path)
        if missing in {"row", "all_rows"}:
            rows = rows.loc[rows.action_index.ne(11)] if missing == "row" else rows.iloc[:0]
        else:
            column = "home_2__top10_xpass" if missing == "player" else "home_2__distance_to_nearest_opponent"
            rows = rows.drop(columns=[column])
        rows.to_parquet(path, index=False)
    original_graphs = [graph.clone() for graph in subject.graph_features_0]
    memo = {}
    for variant in [variants[0], variants[1], variants[0]]:
        if missing and variant.identifier == "b":
            with pytest.raises(ValueError, match="cache 'b'.*situation s"):
                with direct_compare.active_variant(variant, "situation s"):
                    cached_prediction(memo, inference.inference_gnn, subject, model,
                                      device="cpu", return_model_pass_success=True)
            continue
        with direct_compare.active_variant(variant, "situation s"):
            compared, _ = cached_prediction(memo, inference.inference_gnn, subject, model,
                                           device="cpu", return_model_pass_success=True)
            # The same model with the same single-cache configuration is the reference.
            standalone, _ = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
        pd.testing.assert_frame_equal(compared, standalone)
        expected = physics.blend_physical_xpass_predictions(pass_success_model=.9,
            xpass=.2 if variant.identifier == "a" else .8, pass_distance=20.,
            weight_version=model.args["xpass_weight"], distance_to_nearest_opponent=5.)
        assert compared.at[10, "home_2"] == pytest.approx(expected)
    for original, graph in zip(original_graphs, subject.graph_features_0):
        assert original.to_dict().keys() == graph.to_dict().keys()
        assert original.x.equal(graph.x)


def test_sportec_missing_cache_scene_is_not_skipped(tmp_path, monkeypatch, inference_fixture, caches):
    from datatools import sportec_possessions
    subject, models, _, _ = inference_fixture
    subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"], spadl_type=["pass"],
        action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
    monkeypatch.setattr(sportec, "resolve_runtime_feature_run_context", lambda *a, **kw: {
        "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
    monkeypatch.setattr(sportec, "load_match", lambda *a, **kw: subject)
    monkeypatch.setattr(sportec, "resolve_action_indices", lambda *a: [(10, "42")])
    monkeypatch.setattr(sportec_possessions, "build_sportec_possessions", lambda *a, **kw: [])
    def collect(subject, models, *a, **kw):
        if models["pass_success"].args["pc_xpass_id"] == "b":
            raise ValueError("missing required receiver column")
        return {"pass_success": pd.Series({"home_1": 0., "home_2": .2})}
    monkeypatch.setattr(sportec, "collect_action_components", collect)
    args = sportec.parse_args(["--match-id", "s", *CACHE_FLAGS, "--only-pass-success",
        "--pass-success-model-id", "pass_success/fixed", "--use-physical-xpass",
        "--output-dir", str(tmp_path), "--run-id", "missing"])
    with pytest.raises(ValueError, match="cache 'b', match s, action 42.*missing required receiver"):
        direct_compare.run_comparison(args, sportec)
    metadata = json.loads((tmp_path / "missing/metadata.json").read_text())
    assert metadata["status"] == "failed" and not metadata["skipped_actions"]
    assert all("physical_cache_dir" not in model.args for model in models.values())


def test_auxiliary_prediction_reuse_includes_the_active_cache(caches, monkeypatch):
    from pc_xpass_auxiliary import predict_auxiliaries
    from test_visualization_comparison import _subject
    subject = _subject()
    primary, height, intent = (TinyModel(identifier) for identifier in
                               ("pass_success/fixed", "pass_height/h", "pass_intent/i"))
    for model in (height, intent):
        model.args.update(lane_survival=True, lane_survival_mode="max", model_id=model.identifier)
    models = {"pass_success": primary, "xpass_pass_height": height, "xpass_pass_intent": intent}
    variants = [Variant(identifier, identifier, models=models) for identifier in ("a", "b")]
    args = direct.parse_args([*CACHE_FLAGS, "--use-physical-xpass"])
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    calls = []
    def predict(scene, model, **kw):
        calls.append((model.args["task"], model.args["pc_xpass_id"]))
        value = .2 if model.args["pc_xpass_id"] == "a" else .8
        return pd.DataFrame({"home_2": [value]}, index=[10]), None
    monkeypatch.setattr(inference, "inference_gnn", predict)
    for variant in [variants[0], variants[1], variants[0]]:
        with direct_compare.active_variant(variant, "auxiliary test"):
            predictions = predict_auxiliaries(subject, primary, "cpu", graphs=subject.graph_features_0,
                                             labels=subject.labels, event_indices=[10])
            assert set(predictions) == {"pass_height", "pass_intent"}
            assert all(table.at[10, "home_2"] == (.2 if variant.identifier == "a" else .8)
                       for table in predictions.values())
    assert calls == [(role, identifier) for identifier in ("a", "b") for role in ("pass_height", "pass_intent")]
