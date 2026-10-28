"""PC-xPass comparisons vary one setting and isolate each runtime pipeline."""
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
import pc_xpass_uncertainty as uncertainty
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
def test_real_inference_matches_standalone_with_partial_cache(caches, monkeypatch, missing):
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
        with direct_compare.active_variant(variant, "situation s"):
            compared, _ = cached_prediction(memo, inference.inference_gnn, subject, model,
                                           device="cpu", return_model_pass_success=True)
            # The same model with the same single-cache configuration is the reference.
            standalone, _ = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
        pd.testing.assert_frame_equal(compared, standalone)
        expected = physics.blend_physical_xpass_predictions(pass_success_model=.9,
            xpass=.2 if variant.identifier == "a" else .8, pass_distance=20.,
            weight_version=model.args["xpass_weight"], distance_to_nearest_opponent=5.)
        if missing and variant.identifier == "b":
            assert pd.isna(compared.at[11, "home_2"])
            assert pd.isna(compared.at[10, "home_2"]) if missing != "row" else compared.at[10, "home_2"] == pytest.approx(expected)
        else:
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


@pytest.mark.parametrize("module,required", COMMANDS[:2])
def test_cli_versions_alias_order_and_defaults(module, required):
    default = module.parse_args(required)
    assert default.x_pass_version == "top10" and default.xpass_weight == "v3"
    assert default.comparison_dimension is None
    single = module.parse_args([*required, "--pc-xpass", "--xpass-version", "top_pass10", "--top_pass", "10"])
    assert single.x_pass_version == "top-pass10"
    args = module.parse_args([*required, "--compare", "--pc-xpass-id", "a", "--top_pass", "10",
        "--x-pass-version", "MAX_XPASS", "--top-pass", "5", "--x_pass_version", "uncertainty"])
    assert args.x_pass_version == ["top-pass10", "max", "top-pass5", "uncertainty"]
    assert args.pc_xpass and args.comparison_dimension == "version"
    weights = module.parse_args([*required, "--compare", "--pc-xpass-id", "a", "--use-physical-xpass",
        "--xpass-weight", "v5", "--xpass_weight", "v4", "--xpass-weight", "v3",
        "--v4-power", "3", "--v4-zero", "0.7", "--discount", "false", "--v5-discount", "false",
        "--v5-intent-threshold", "0.2", "--only-pass-success"])
    assert weights.xpass_weight == ["v5", "v4", "v3"]
    assert weights.x_pass_version == "top10" and weights.comparison_dimension == "weight" and weights.pc_xpass
    assert list(direct_compare.pc_xpass_specifications(weights)) == ["weight:v5", "weight:v4", "weight:v3"]


VERSION_FLAGS = ["--compare", "--pc-xpass-id", "a", "--xpass-version", "max", "--top-pass", "10"]
WEIGHT_FLAGS = ["--compare", "--pc-xpass-id", "a", "--xpass-weight", "v5", "--xpass-weight", "v4"]


@pytest.mark.parametrize("module,required", COMMANDS[:2])
@pytest.mark.parametrize("extra", [
    ["--xpass-version", "max", "--xpass-version", "top10"],
    ["--pc-xpass", "--top-pass", "10", "--top-pass", "5"],
    ["--xpass-weight", "v3", "--xpass-weight", "v4"],
    ["--pc-xpass", "--xpass-version", "max", "--top-pass", "10"],
    ["--compare", "--xpass-version", "max", "--top-pass", "10"],
    ["--compare", "--xpass-weight", "v3", "--xpass-weight", "v4", "--use-physical-xpass"],
    [*VERSION_FLAGS, "--pc-xpass-id", "b"],
    [*VERSION_FLAGS, "--xpass-weight", "v3", "--xpass-weight", "v4"],
    [*CACHE_FLAGS, "--xpass-weight", "v3", "--xpass-weight", "v4"],
    [*VERSION_FLAGS, "--xpass-version", "MAX_XPASS"],
    [*VERSION_FLAGS, "--xpass-version", "top_pass10"],
    [*VERSION_FLAGS, "--top-pass", "0"],
    [*VERSION_FLAGS, "--xpass-version", "noise-kernel"],
    [*VERSION_FLAGS, "--no-physical-cache"],
    [*VERSION_FLAGS, "--physical-cache-dir", "cache"],
    [*VERSION_FLAGS, "--lane-survival-cache-dir", "cache"],
    [*VERSION_FLAGS, "--compare-pass-success"],
    [*VERSION_FLAGS, "--only-compare-pass-success"],
    [*VERSION_FLAGS, "--inference-mask-1", "accel"],
    [*VERSION_FLAGS, "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b"],
    [*VERSION_FLAGS, "--uncertainty"],
    WEIGHT_FLAGS,
    [*WEIGHT_FLAGS, "--use-physical-xpass", "--only-physical-xpass"],
    [*WEIGHT_FLAGS, "--use-physical-xpass", "--only-pass-intent", "--show-model-pass-success"],
    [*WEIGHT_FLAGS, "--use-physical-xpass", "--only-outcome-success"],
    [*WEIGHT_FLAGS, "--use-physical-xpass", "--xpass-weight", "v5"],
    [*VERSION_FLAGS, "--v4-power", "3"],
    [*VERSION_FLAGS, "--v5-discount", "false"],
])
def test_cli_invalid_version_and_weight_comparisons(module, required, extra):
    with pytest.raises((SystemExit, ValueError)):
        module.parse_args([*required, *extra])


@pytest.fixture
def metric_caches(caches):
    values = {"max_xpass": .8, "top10_xpass": .2, "top_pass10_xpass": .5, "uncertainty_xpass": .3}
    for dataset in ("hawkeye", "sportec"):
        folder = caches / "a" / dataset
        metadata = json.loads((folder / "metadata.json").read_text())
        metadata.update(available_metrics=list(values), uncertainty_config=uncertainty.configuration(True),
                        uncertainty_metric_definition=uncertainty.DEFINITION)
        (folder / "metadata.json").write_text(json.dumps(metadata))
        path = folder / "matches/s.parquet"
        rows = pd.read_parquet(path)
        for player in ("home_1", "home_2", "away_3"):
            for metric, value in values.items():
                rows[f"{player}__{metric}"] = value
        rows.to_parquet(path, index=False)
    return values


@pytest.mark.parametrize("dimension", ["version", "weight"])
@pytest.mark.parametrize("reverse", [False, True])
def test_inference_versions_weights_and_scores(tmp_path, monkeypatch, outcome_pipeline, metric_caches, dimension, reverse):
    from models import utils
    from models.pass_height import definition
    module, required, models, _, build, _ = outcome_pipeline
    weights = ["v5", "v4", "v3"]
    versions = ["max", "top-pass10", "uncertainty"]
    values = versions if dimension == "version" else weights
    if reverse:
        values.reverse()
    flags = ["--compare", "--pc-xpass-id", "a", "--use-physical-xpass"]
    for value in values:
        flags.extend(["--top-pass", "10"] if value == "top-pass10" else
                     ["--xpass-version" if dimension == "version" else "--xpass-weight", value])
    auxiliary_calls = []
    def load_auxiliary(identifier, device):
        auxiliary_calls.append(identifier)
        model = TinyModel(identifier)
        if identifier.startswith("pass_height/"):
            model.args["pass_height_definition"] = definition(1.5, "checkpoint")
        return model.eval()
    monkeypatch.setattr(utils, "load_model", load_auxiliary)
    monkeypatch.setattr(utils, "get_model_provenance", lambda identifier: {"model_id": identifier})
    if dimension == "weight":
        flags.extend(["--xpass-pass-height-model-id", "pass_height/h", "--xpass-pass-intent-model-id", "pass_intent/i",
                      "--v4-power", "3", "--v5-intent-threshold", "2"])
    calls = []
    def predict(scene, model, **kw):
        task = model.args["task"]
        if task == "pass_success":
            settings = physics.physical_xpass_inference_lookup_config(model.args, cache_dir=model.args["physical_cache_dir"])
            weight = model.args["xpass_weight"]
            expected_roles = {"pass_height", "pass_intent"} if weight == "v5" else {"pass_height"} if weight == "v4" else set()
            assert set(model._xpass_auxiliary_models) == set(model.args["xpass_auxiliary_models"]) == expected_roles
            calls.append((model.args["x_pass_version"], weight))
            result = physics.blend_physical_xpass_predictions(pass_success_model=.6,
                xpass=metric_caches[settings["metric"]], pass_distance=20., weight_version=weight,
                pass_height=.25, pass_intent=1., v4_power=model.args.get("v4_power", 2.),
                v5_intent_threshold=model.args.get("v5_intent_threshold", .01))
            return pd.DataFrame({"home_1": [0.], "home_2": [result]}, index=[10]), None
        pair = (.1, .8) if task == "outcome_scoring" else (.02, .1)
        return tuple(pd.DataFrame({"home_1": [0.], "home_2": [value]}, index=[10]) for value in pair)
    monkeypatch.setattr(module, "inference_gnn", predict)
    physical_loader = None
    if module is direct:
        physical_loader = Mock(wraps=module.load_runtime_physical_xpass_visualization_table)
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", physical_loader)
    if module is sportec:
        from datatools import sportec_possessions
        subject = build.return_value
        monkeypatch.setattr(sportec_possessions, "build_sportec_possessions", lambda *a, **kw: [])
        monkeypatch.setattr(module, "possession_state_for_action", lambda *a: (subject, 10))
        monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
        def physical(folder, *a, metric, x_pass_version, **kw):
            assert metric == physics.physical_xpass_metric_for_version(x_pass_version, pc_xpass=True)
            return pd.Series({"home_1": 0., "home_2": metric_caches[metric]})
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_component", physical)
    args = module.parse_args([*required, *flags, "--only-pass-success", "--only-pass-score", "--show-physical-xpass",
        "--pass-success-model-id", "pass_success/fixed", "--outcome-scoring-model-id", "outcome_scoring/a",
        "--outcome-conceding-model-id", "outcome_conceding/x", "--output-dir", str(tmp_path), "--run-id", "settings"])
    direct_compare.run_comparison(args, module)
    aligned = pd.read_csv(tmp_path / "settings/aligned_predictions.csv")
    metadata = json.loads((tmp_path / "settings/metadata.json").read_text())
    assert metadata["comparison_dimension"] == dimension and metadata["comparison_mode"] == "pc_xpass"
    assert [v["id"] for v in metadata["variants"]] == [f"{dimension}:{value}" for value in values]
    assert build.call_count == 1 and len(calls) == 3
    if physical_loader is not None:
        assert physical_loader.call_count == (3 if dimension == "version" else 1)
    assert sorted(auxiliary_calls) == (["pass_height/h", "pass_intent/i"] if dimension == "weight" else [])
    for value in values:
        version, weight = (value, "v3") if dimension == "version" else ("top10", value)
        raw = metric_caches[physics.physical_xpass_metric_for_version(version, pc_xpass=True)]
        blended = physics.blend_physical_xpass_predictions(pass_success_model=.6, xpass=raw, pass_distance=20.,
            weight_version=weight, pass_height=.25, pass_intent=1., v4_power=3., v5_intent_threshold=2.)
        for name, expected in (("physical_xpass", raw), ("pass_success", blended),
                               ("pass_score", blended * .7 + (1 - blended) * .08)):
            row = aligned.loc[aligned.component.eq(name) & aligned.candidate_id.eq("home_2")].iloc[0]
            assert row[f"prediction:{dimension}:{value}"] == pytest.approx(expected)
    for model in models.values():
        assert "physical_cache_dir" not in model.args and not hasattr(model, "_xpass_auxiliary_models")


@pytest.mark.parametrize("reverse", [False, True])
def test_real_metric_comparison_matches_standalone(caches, metric_caches, reverse):
    from test_physical_xpass import DummyBaselineInferenceModel, make_physical_inference_match
    subject = make_physical_inference_match([10, 11], [1, 1])
    subject.match_id = "s"
    primary = DummyBaselineInferenceModel(caches / "a/hawkeye")
    original = primary.args
    versions = ["max", "top-pass10", "uncertainty"]
    if reverse:
        versions.reverse()
    args = direct.parse_args(["--compare", "--pc-xpass-id", "a", "--use-physical-xpass",
                             *[token for v in versions for token in ("--xpass-version", v)]])
    variants = [Variant(identifier, label, models={"pass_success": primary}, pc_xpass_settings=settings)
                for identifier, (label, settings) in direct_compare.pc_xpass_specifications(args).items()]
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    memo = {}
    for variant in [*variants, variants[0]]:
        with direct_compare.active_variant(variant, "situation s"):
            compared, _ = cached_prediction(memo, inference.inference_gnn, subject, primary, device="cpu")
            standalone, _ = inference.inference_gnn(subject, primary, device="cpu")
            pd.testing.assert_frame_equal(compared, standalone)
            expected = physics.blend_physical_xpass_predictions(pass_success_model=.9,
                xpass=metric_caches[variant.provenance["physical_xpass_metric"]], pass_distance=20., weight_version="v3")
            assert compared.at[10, "home_2"] == pytest.approx(expected)
        assert primary.args is original and not hasattr(primary, "_xpass_auxiliary_models")


@pytest.mark.parametrize("failure", ["metric", "uncertainty", "row", "player"])
def test_version_failure_has_complete_context(caches, metric_caches, failure):
    from test_physical_xpass import DummyBaselineInferenceModel, make_physical_inference_match
    folder = caches / "a/hawkeye"
    if failure in {"metric", "uncertainty"}:
        metadata = json.loads((folder / "metadata.json").read_text())
        if failure == "metric":
            metadata["available_metrics"].remove("uncertainty_xpass")
        else:
            metadata["uncertainty_config"]["schema_version"] = 1
        (folder / "metadata.json").write_text(json.dumps(metadata))
    else:
        path = folder / "matches/s.parquet"
        rows = pd.read_parquet(path)
        rows = rows.loc[rows.action_index.ne(11)] if failure == "row" else rows.drop(columns=["home_2__uncertainty_xpass"])
        rows.to_parquet(path, index=False)
    args = direct.parse_args([*VERSION_FLAGS, "--xpass-version", "uncertainty", "--use-physical-xpass"])
    primary = DummyBaselineInferenceModel(folder)
    primary.args["physical_xpass_require_complete"] = True
    variants = [Variant(identifier, label, models={"pass_success": primary}, pc_xpass_settings=settings)
                for identifier, (label, settings) in direct_compare.pc_xpass_specifications(args).items()]
    if failure in {"metric", "uncertainty"}:
        with pytest.raises(ValueError, match="cache 'a'.*version '.*'.*weight 'v3'"):
            direct_compare.configure_physical(args, direct, variants, "hawkeye")
    else:
        direct_compare.configure_physical(args, direct, variants, "hawkeye")
        subject = make_physical_inference_match([10, 11], [1, 1])
        subject.match_id = "s"
        with pytest.raises((inference.PCXPassBlendingError, ValueError),
                           match="cache 'a'.*situation s.*version 'uncertainty'.*weight 'v3'"):
            with direct_compare.active_variant(variants[-1], "situation s"):
                inference.inference_gnn(subject, primary, device="cpu")


@pytest.mark.parametrize("reverse", [False, True])
def test_weight_auxiliaries_union_graph_schema_and_state_restoration(monkeypatch, inference_fixture, caches, reverse):
    from models import utils
    from models.pass_height import definition
    from models.edge_schema import BASE_FEATURES, EDGE_GROUPS
    weights = ["v4", "v5", "v3"]
    if reverse:
        weights.reverse()
    loaded = {}
    def load(identifier, device):
        assert identifier not in loaded
        model = loaded[identifier] = TinyModel(identifier).eval()
        if identifier.startswith("pass_height/"):
            model.args.update(edge_in_dim=2, pass_height_definition=definition(1.5, "checkpoint"))
        else:
            names = BASE_FEATURES + EDGE_GROUPS["relative_velocity"]
            model.args.update(node_in_dim=30, edge_in_dim=len(names), edge_feature_names=list(names))
        return model
    monkeypatch.setattr(utils, "load_model", load)
    monkeypatch.setattr(utils, "get_model_provenance", lambda identifier: {"model_id": identifier})
    args = direct.parse_args(["--compare", "--pc-xpass-id", "a", "--use-physical-xpass", "--only-pass-success",
        "--pass-success-model-id", "pass_success/fixed", "--xpass-pass-height-model-id", "pass_height/h",
        "--xpass-pass-intent-model-id", "pass_intent/i", "--v4-power", "3", "--v5-intent-threshold", "2",
        *[token for weight in weights for token in ("--xpass-weight", weight)]])
    variants, _, _, _, schema = direct_compare.resolve_variants(args, direct, False)
    assert schema["node_in_dim"] == 30 and "relative_vx" in schema["edge_feature_names"]
    assert all(variant.provenance["graph_schema"] == schema for variant in variants)
    primary = variants[0].models["pass_success"]
    original_binding = {"original": object()}
    primary._xpass_auxiliary_models = original_binding
    originals = {id(model): model.args for variant in variants for model in variant.models.values()}
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    for variant in [*variants, variants[0]]:
        weight = variant.pc_xpass_settings["xpass_weight"]
        roles = {"pass_height", "pass_intent"} if weight == "v5" else {"pass_height"} if weight == "v4" else set()
        with pytest.raises(ValueError, match="deliberate failure"):
            with direct_compare.active_variant(variant, "situation s"):
                assert set(primary._xpass_auxiliary_models) == set(primary.args["xpass_auxiliary_models"]) == roles
                assert ("v4_power" in primary.args) == (weight == "v4")
                assert ("v5_intent_threshold" in primary.args) == (weight == "v5")
                raise ValueError("deliberate failure")
        assert primary._xpass_auxiliary_models is original_binding
        assert all(model.args is originals[id(model)] for model in variant.models.values())
    assert sorted(loaded) == ["pass_height/h", "pass_intent/i"]


@pytest.mark.parametrize("auxiliary_flags,weights", [
    ([], ["v4", "v5"]),
    (["--xpass-pass-height-model-id", "pass_height/h"], ["v4", "v5"]),
    (["--xpass-pass-intent-model-id", "pass_intent/i"], ["v4", "v5"]),
    (["--xpass-pass-height-model-id", "pass_height/h"], ["v2", "v3"]),
])
def test_weight_comparison_requires_exact_auxiliary_union(inference_fixture, auxiliary_flags, weights):
    args = direct.parse_args(["--compare", "--pc-xpass-id", "a", "--use-physical-xpass", "--only-pass-success",
        "--pass-success-model-id", "pass_success/fixed", *auxiliary_flags,
        *[token for weight in weights for token in ("--xpass-weight", weight)]])
    with pytest.raises((ValueError, inference.PCXPassBlendingError), match="required explicitly|unused"):
        direct_compare.resolve_variants(args, direct, False)


@pytest.mark.parametrize("reverse", [False, True])
def test_real_weight_comparison_matches_standalone_and_reuses_auxiliaries(caches, reverse):
    import torch
    from test_physical_xpass import DummyBaselineInferenceModel, make_physical_inference_match
    from test_pc_xpass_auxiliary import auxiliary_model
    from models.pass_height import definition
    subject = make_physical_inference_match([10, 11], [1, 1])
    subject.match_id = "s"
    for graph in subject.graph_features_0:
        graph.x = torch.cat([graph.x, graph.x.new_zeros((graph.num_nodes, 1))], dim=1)
    primary = DummyBaselineInferenceModel(caches / "a/hawkeye")
    primary.args["xpass_pass_height_definition"] = definition(1.5, "checkpoint")
    height = auxiliary_model(probability=.25)
    intent = auxiliary_model("pass_intent")
    weights = ["v5", "v4", "v3"]
    if reverse:
        weights.reverse()
    args = direct.parse_args(["--compare", "--pc-xpass-id", "a", "--use-physical-xpass",
        "--v5-intent-threshold", "2", "--v4-power", "3",
        *[token for weight in weights for token in ("--xpass-weight", weight)]])
    variants = []
    for identifier, (label, settings) in direct_compare.pc_xpass_specifications(args).items():
        weight = settings["xpass_weight"]
        models = {"pass_success": primary}
        if weight in {"v4", "v5"}:
            models["xpass_pass_height"] = height
        if weight == "v5":
            models["xpass_pass_intent"] = intent
        variants.append(Variant(identifier, label, models=models, pc_xpass_settings=settings))
    original_args = primary.args
    direct_compare.configure_physical(args, direct, variants, "hawkeye")
    memo, predictions = {}, {}
    for variant in [*variants, variants[0]]:
        weight = variant.pc_xpass_settings["xpass_weight"]
        with direct_compare.active_variant(variant, "situation s"):
            compared, _ = cached_prediction(memo, inference.inference_gnn, subject, primary, device="cpu")
            standalone, _ = inference.inference_gnn(subject, primary, device="cpu")
            pd.testing.assert_frame_equal(compared, standalone)
            expected = physics.blend_physical_xpass_predictions(pass_success_model=.9, xpass=.2,
                pass_distance=20., weight_version=weight, pass_height=.25, pass_intent=1.,
                v4_power=3., v5_intent_threshold=2.)
            assert compared.at[10, "home_2"] == pytest.approx(expected)
            predictions[weight] = compared.at[10, "home_2"]
        assert primary.args is original_args and not hasattr(primary, "_xpass_auxiliary_models")
    assert len(set(predictions.values())) == 3
    assert len(subject._xpass_auxiliary_predictions) == 2
