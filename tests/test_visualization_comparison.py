from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch_geometric.data import Data

from scripts import visualize_hawkeye as hv, visualize_skillcorner as sv, visualize_benchmark as bv
from scripts import visualize_action_components as av, run_and_visualize_hawkeye as dv
from scripts import inference_visualization_comparison as inference_compare
from scripts import saved_visualization_comparison as saved_compare
from scripts.visualization_comparison import (
    ComparisonExport, Variant, align_rows, compose_panels, require_row, validate_metadata_definitions,
)


@pytest.mark.parametrize("script,required", [(hv, []), (sv, ["--match-id", "m", "--index", "1"]), (bv, [])])
def test_saved_cli(script, required):
    single = script.parse_args(["--accept-comparison-warnings", *required, "--component-run-id", "a"])
    assert single.component_run_id == "a"
    assert single.visualization_comparison is None
    compared = script.parse_args(["--accept-comparison-warnings", *required, "--compare", "--component-run-id", "a", "--component-run-id", "b"])
    assert compared.component_run_id == ["a", "b"]
    assert compared.visualization_comparison == "components"
    for extra in (["--compare"], ["--compare", "--component-run-id", "a"],
                  ["--compare", "--component-run-id", "a", "--component-run-id", "a"],
                  ["--component-run-id", "a", "--component-run-id", "b"],
                  ["--compare", "--component-run-id", "a", "--component-run-id", "b", "--component-dir", "root"],
                  ["--inference-mask", "accel"], ["--inference-mask-1", "accel"], ["--comparison"]):
        with pytest.raises(SystemExit):
            script.parse_args(["--accept-comparison-warnings", *required, *extra])
    if script is not bv:
        with pytest.raises(SystemExit):
            script.parse_args(["--accept-comparison-warnings", *required, "--compare", "--component-run-id", "a", "--component-run-id", "b", "--output", "gif"])


@pytest.mark.parametrize("script,required", [(dv, []), (av, ["--match-id", "m"])])
def test_inference_cli(script, required):
    ids = ["--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b"]
    args = script.parse_args(["--accept-comparison-warnings", *required, "--compare", *ids, "--inference-mask", "accel"])
    assert args.visualization_comparison == "checkpoints"
    args = script.parse_args(["--accept-comparison-warnings", *required, "--inference-mask", "accel"])
    assert args.visualization_comparison is None
    args = script.parse_args(["--accept-comparison-warnings", *required, "--inference-mask-9", "accel", "--inference-mask-2", "goal-features"])
    assert args.visualization_comparison == "masks"
    for extra in (["--compare"], ["--compare", "--pass-success-model-id", "pass_success/a"], ids,
                  ["--compare", *ids, "--pass-success-model-id", "pass_success/a"],
                  ["--compare", *ids, "--pass-intent-model-id", "pass_intent/a", "--pass-intent-model-id", "pass_intent/b"],
                  ["--pass-success-model-id", "pass_intent/a"],
                  ["--compare", "--inference-mask-1", "accel"], [*ids, "--inference-mask-1", "accel"],
                  ["--inference-mask", "accel", "--inference-mask-1", "accel"],
                  ["--inference-mask-1", "accel", "--inference-mask-1", "goal-features"],
                  ["--inference-mask-0", "accel"], ["--inference-mask-1"], ["--comparison"]):
        with pytest.raises(SystemExit):
            script.parse_args(["--accept-comparison-warnings", *required, *extra])
    if script is dv:
        with pytest.raises(SystemExit):
            script.parse_args(["--accept-comparison-warnings", "--inference-mask-1", "accel", "--output", "mp4"])


def _outcome_pair_flags(pairs):
    return [flag for scoring, conceding in pairs for flag in (
        "--outcome-scoring-model-id", f"outcome_scoring/{scoring}",
        "--outcome-conceding-model-id", f"outcome_conceding/{conceding}")]


@pytest.mark.parametrize("script,required", [(dv, []), (av, ["--match-id", "m"])])
@pytest.mark.parametrize("pairs", [[("b", "x"), ("a", "y")],
                                   [("a", "x"), ("b", "x"), ("b", "y")]])
def test_outcome_pair_selection_order_and_identity(script, required, pairs, inference_fixture):
    flags = ["--compare", "--only-outcome-success", *_outcome_pair_flags(pairs)]
    args = script.parse_args([*required, *flags])
    assert args.visualization_comparison == "checkpoints"
    variants, _, _, _, _ = inference_compare.resolve_variants(args, script, script is av)
    for variant, (scoring, conceding) in zip(variants, pairs):
        expected = {"outcome_scoring": f"outcome_scoring/{scoring}",
                    "outcome_conceding": f"outcome_conceding/{conceding}"}
        assert variant.provenance["model_ids"] == expected
        assert json.loads(variant.identifier.removeprefix("outcome_pair:")) == list(expected.values())
        assert all(identifier in variant.label for identifier in expected.values())
        assert {task: model.identifier for task, model in variant.models.items()} == expected
    for task in ("outcome_scoring", "outcome_conceding"):
        selected = {}
        for variant in variants:
            identifier = variant.provenance["model_ids"][task]
            assert variant.models[task] is selected.setdefault(identifier, variant.models[task])
    reversed_args = script.parse_args([*required, "--compare", "--only-outcome-success",
                                       *_outcome_pair_flags(list(reversed(pairs)))])
    reversed_variants, _, _, _, _ = inference_compare.resolve_variants(reversed_args, script, script is av)
    assert [variant.identifier for variant in reversed_variants] == [variant.identifier for variant in reversed(variants)]


@pytest.mark.parametrize("script,required", [(dv, []), (av, ["--match-id", "m"])])
def test_outcome_pair_invalid_cli(script, required):
    pairs = _outcome_pair_flags([("a", "x"), ("b", "y")])
    invalid = [
        ["--compare", *_outcome_pair_flags([("a", "x"), ("a", "x")])],
        ["--compare", *pairs, "--outcome-conceding-model-id", "outcome_conceding/z"],
        ["--compare", *pairs, "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b"],
        ["--compare", *pairs, "--inference-mask-1", "all:accel"],
        [*pairs, "--inference-mask-1", "all:accel"],
        pairs,
        ["--compare", "--outcome-scoring-model-id", "outcome_scoring/a", "--outcome-scoring-model-id", "outcome_scoring/a"],
    ]
    for flags in invalid:
        with pytest.raises(SystemExit):
            script.parse_args([*required, *flags])
    if script is dv:
        with pytest.raises(SystemExit):
            script.parse_args(["--compare", *pairs, "--output", "mp4"])


@pytest.mark.parametrize("script,required", [(dv, []), (av, ["--match-id", "m"])])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
@pytest.mark.parametrize("output", ["outcome-success", "outcome-failure", "pass-score"])
@pytest.mark.parametrize("counterpart", ["missing", "explicit", "bundle"])
def test_derived_outcome_requires_pairs_before_loading(script, required, task, output, counterpart, monkeypatch):
    other = "outcome_conceding" if task == "outcome_scoring" else "outcome_scoring"
    selector = "--" + task.replace("_", "-") + "-model-id"
    flags = ["--compare", selector, f"{task}/a", selector, f"{task}/b", f"--only-{output}"]
    if counterpart == "explicit":
        flags += ["--" + other.replace("_", "-") + "-model-id", f"{other}/fixed"]
    elif counterpart == "bundle":
        flags += ["--bundle-id", "fixed"]
        monkeypatch.setattr(inference_compare, "load_bundle_record", lambda identifier: {
            "model_ids": {other: f"{other}/fixed", "pass_success": "pass_success/fixed"}})
    load = Mock()
    monkeypatch.setattr(inference_compare, "load_model", load)
    args = script.parse_args([*required, *flags])
    with pytest.raises(ValueError, match="Repeat --outcome-scoring-model-id and --outcome-conceding-model-id equally"):
        inference_compare.resolve_variants(args, script, script is av)
    load.assert_not_called()


@pytest.mark.parametrize("script,required", [(dv, []), (av, ["--match-id", "m"])])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_raw_outcome_single_task_comparison(script, required, task, inference_fixture):
    option = task.replace("_", "-")
    args = script.parse_args([*required, "--compare", f"--{option}-model-id", f"{task}/a",
                              f"--{option}-model-id", f"{task}/b", f"--only-{option}"])
    variants, _, _, _, _ = inference_compare.resolve_variants(args, script, script is av)
    assert [variant.identifier for variant in variants] == [f"{task}/a", f"{task}/b"]
    assert all(set(variant.models) == {task} for variant in variants)
    paired_args = script.parse_args([*required, "--compare", f"--only-{option}",
                                    *_outcome_pair_flags([("a", "x"), ("b", "y")])])
    with pytest.raises(ValueError, match="does not contribute"):
        inference_compare.resolve_variants(paired_args, script, script is av)


def test_alignment_candidates_and_strict_rows():
    a = pd.Series({"home_1": 0.0, "home_2": 0.8})
    b = pd.Series({"home_2": 0.3, "home_1": 0.0})
    aligned = align_rows([a, b], ["a", "b"], [], "scene")
    assert list(aligned[1].index) == ["home_1", "home_2"]
    messages = []
    with pytest.warns(UserWarning, match="membership"):
        aligned = align_rows([a, pd.Series({"home_3": 0.5})], ["a", "b"], messages, "scene")
    assert aligned[0]["home_1"] == 0
    assert pd.isna(aligned[0]["home_3"])
    assert messages
    with pytest.raises(ValueError, match="Duplicate"):
        align_rows([pd.Series([1, 2], index=["a", "a"])], ["a"], [], "scene")
    with pytest.raises(ValueError, match="Missing selected frame"):
        require_row(pd.DataFrame({"a": [1]}, index=[1]), 2, "pass_intent")
    assert validate_metadata_definitions([{"return_type": "next_5"}, {"return_type": "next_10"}])


def test_compositor_grid_and_benchmark_order():
    colors = ["red", "green", "blue", "yellow", "purple"]
    tiles = [Image.new("RGB", (100, 70), color) for color in colors]
    image = compose_panels(tiles, ["long qualified/model/identifier " * 3] * 5, "Scene | Component")
    # Every original tile is pasted without resizing; find its unique color block.
    pixels = np.asarray(image)
    boxes = []
    for tile in tiles:
        color = np.array(tile.getpixel((0, 0)))
        y, x = np.where(np.all(pixels == color, axis=-1))
        boxes.append((x.min(), y.min(), x.max(), y.max()))
        assert (x.max() - x.min() + 1, y.max() - y.min() + 1) == tile.size
    assert boxes[0][0] == boxes[2][0] == boxes[4][0]
    assert boxes[1][0] == boxes[3][0] > boxes[0][0]
    assert boxes[0][1] == boxes[1][1] < boxes[2][1] == boxes[3][1] < boxes[4][1]
    benchmark = compose_panels(tiles[:4], ["a", "b", "a", "b"], "Modification 1", columns=2,
                               row_labels=["Game state 1", "Game state 2"])
    assert benchmark.width > image.width
    assert compose_panels([tiles[0], Image.new("RGB", (90, 70))], ["a", "b"], "Title").width >= 200


def test_export_precision_availability_and_identity(tmp_path):
    variants = [Variant("run/a", "run/a"), Variant("run/b", "run/b")]
    report = ComparisonExport(tmp_path, variants, SimpleNamespace(accept_comparison_warnings=True), "components")
    value = 0.12345678912345678
    with pytest.warns(UserWarning):
        report.record("match/action", 3, "pass_intent", [pd.Series({"home_1": value}), pd.Series({"home_2": 0.0})])
    report.finish()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["status"] == "completed" and metadata["warnings"]
    a = pd.read_parquet(metadata["variants"][0]["predictions"])
    assert a.loc[a.candidate_id.eq("home_1"), "value"].iloc[0] == value
    assert not a.loc[a.candidate_id.eq("home_2"), "available"].iloc[0]
    text = (tmp_path / "aligned_predictions.csv").read_text()
    assert "N/A" in text and "prediction:run/a" in text
    assert len(list((tmp_path / "variants").glob("*/predictions.parquet"))) == 2


def _args(task="pass_success"):
    return dict(task=task, model="gat", gnn_task="node_binary", node_in_dim=26, edge_in_dim=5,
                node_emb_dim=8, graph_emb_dim=8, gnn_layers=1, gnn_heads=1, skip_conn=True,
                out_dim=1, include_out=False, use_carries=True, possessor_aware=True, poss_vel_aware=True,
                poss_rel_vel_aware=True, extend_features=True, keeper_aware=True, ball_z_aware=True)


class TinyModel(torch.nn.Module):
    def __init__(self, identifier):
        super().__init__()
        self.identifier = identifier
        self.args = _args(identifier.split("/")[0])


def _record(identifier):
    task = identifier.split("/")[0]
    return dict(task=task, model_id=identifier, is_complete=True, args=_args(task), metadata={},
                feature_run_id="feature", intended_receiver_mode="predict", use_carries=True,
                return_type="next_5", target_family="goal",
                graph_schema=dict(node_in_dim=26, edge_in_dim=5, add_v_edge_features=True,
                                  add_relative_speed_edge_features=True))


def _subject():
    frames = [10, 11]
    tracking = {}
    for i, player in enumerate(["home_1", "home_2", "away_3", "home_goal", "away_goal"]):
        for axis, value in [("x", 10 + i * 10), ("y", 20 + i * 2), ("vx", 1), ("vy", 0)]:
            tracking[f"{player}_{axis}"] = [value, value]
    tracking.update(ball_x=[10., 10.], ball_y=[20., 20.], ball_z=[0., 0.])
    return SimpleNamespace(situation_id="s", match_id="s", tracking=pd.DataFrame(tracking, index=frames),
        frame_meta=pd.DataFrame(dict(abs_time=[0., 0.1], possession_team=["A", "A"],
            possession_prefix=["home", "home"], possessor_object_id=["home_1", "home_1"]), index=pd.Index(frames, name="frame_id")),
        labels=torch.ones((2, 3)), graph_features_0=[Data(x=torch.zeros((1, 26)))], team_map={"A": "home", "B": "away"})


@pytest.fixture
def inference_fixture(monkeypatch):
    subject = _subject()
    models = {}
    calls = []
    monkeypatch.setattr(inference_compare, "get_model_record", _record)
    def load(identifier, device):
        models[identifier] = TinyModel(identifier)
        return models[identifier]
    monkeypatch.setattr(inference_compare, "load_model", load)
    monkeypatch.setattr(dv, "load_hawkeye_tracking", lambda path: pd.DataFrame({"id": ["s"]}))
    monkeypatch.setattr(dv, "clean_hawkeye_tracking", lambda frame: frame)
    monkeypatch.setattr(dv, "load_hawkeye_ball", lambda path: pd.DataFrame())
    monkeypatch.setattr(dv, "clean_hawkeye_ball", lambda frame: frame)
    monkeypatch.setattr(dv, "resolve_hawkeye_situation_ids", lambda *a, **k: ["s"])
    build = Mock(return_value=(subject, None, None))
    monkeypatch.setattr(dv, "build_hawkeye_situation", build)
    monkeypatch.setattr(dv, "resolve_ballreceipt", lambda frame: 0.)
    monkeypatch.setattr(dv, "resolve_hawkeye_png_frames", lambda *a: [{"frame_id": 10, "label": "0", "resolved_time_norm": 0.}])
    monkeypatch.setattr(dv, "load_overlay_data", lambda **kw: None)
    monkeypatch.setattr(dv, "build_situation_overlays", lambda *a, **kw: (pd.Series(dtype=float), pd.Series(dtype=str), {"selection_denominators": {}}))
    monkeypatch.setattr(dv, "configure_lane_survival_runtime_cache", lambda *a: None)
    def predict(scene, model, **kwargs):
        assert not model.training
        mask = bool(getattr(model, "inference_masks", None))
        intent = kwargs.get("pass_intent_probs")
        calls.append((model.identifier, mask, None if intent is None else intent.copy(deep=True)))
        value = (0.7 if model.identifier.endswith("/a") else 0.5) - (0.2 if mask else 0)
        if model.args.get("xpass_weight") == "v5" and intent is not None:
            value = float(intent.iloc[0]["home_2"])
        result = pd.DataFrame({"home_1": [0.0], "home_2": [value], "away_3": [0.1]}, index=[10])
        return (result * 0.1, result * 0.5) if model.args["task"].startswith("outcome_") else (result, None)
    monkeypatch.setattr(dv, "inference_gnn", predict)
    return subject, models, calls, build


def test_hawkeye_numbered_masks_synthetic_end_to_end(tmp_path, monkeypatch, inference_fixture):
    subject, models, calls, build = inference_fixture
    original = subject.tracking.copy(deep=True)
    args = dv.parse_args(["--accept-comparison-warnings", "--pass-success-model-id", "pass_success/a", "--only-pass-success",
        "--inference-mask-2", "pos-node-features", "--inference-mask-1", "accel",
        "--output-dir", str(tmp_path), "--run-id", "comparison"])
    monkeypatch.setattr(dv, "parse_args", lambda: args)
    dv.main()
    assert len(calls) == 3 and [item[1] for item in calls] == [False, True, True]
    assert build.call_count == 1
    pd.testing.assert_frame_equal(subject.tracking, original)
    assert models["pass_success/a"].inference_masks is None
    root = tmp_path / "comparison"
    assert (root / "s/pass_success_0.png").exists()
    metadata = json.loads((root / "metadata.json").read_text())
    assert [variant["id"] for variant in metadata["variants"]] == ["unmasked", "mask_1", "mask_2"]
    membership = metadata["model_membership"]["entries"]
    assert {item["variant_id"] for item in membership} == {"unmasked", "mask_1", "mask_2"}
    assert all(item["status"] == "unknown" and item["dataset"] == "hawkeye" for item in membership)
    aligned = pd.read_csv(root / "aligned_predictions.csv")
    assert aligned.loc[aligned.candidate_id.eq("home_2"), "prediction:unmasked"].iloc[0] == pytest.approx(0.7)
    assert aligned.loc[aligned.candidate_id.eq("home_2"), "prediction:mask_1"].iloc[0] == pytest.approx(0.5)


def test_hawkeye_checkpoints_common_mask_bundle_and_derived(tmp_path, monkeypatch, inference_fixture):
    subject, models, calls, build = inference_fixture
    def different_training_splits(identifier):
        item = _record(identifier)
        item.update(train_count=20 if identifier.endswith("/b") else 10,
                    split_manifest_id="split_b" if identifier.endswith("/b") else "split_a")
        return item
    monkeypatch.setattr(inference_compare, "get_model_record", different_training_splits)
    bundle = {"model_ids": {task: f"{task}/support" for task in ("pass_success", "outcome_scoring", "outcome_conceding")}}
    monkeypatch.setattr(inference_compare, "load_bundle_record", lambda identifier: bundle)
    args = dv.parse_args(["--accept-comparison-warnings", "--compare", "--bundle-id", "fixed", "--pass-success-model-id", "pass_success/a",
        "--pass-success-model-id", "pass_success/b", "--only-pass-score", "--inference-mask", "pass_success:accel",
        "--output-dir", str(tmp_path), "--run-id", "checkpoints"])
    monkeypatch.setattr(dv, "parse_args", lambda: args)
    dv.main()
    assert len([item for item in calls if item[0].startswith("pass_success/")]) == 2
    assert all(item[1] for item in calls if item[0].startswith("pass_success/"))
    assert len(calls) == 4  # Two pass-success checkpoints plus each fixed outcome once.
    assert build.call_count == 1
    assert not any(item[0] == "pass_success/support" for item in calls)
    metadata = json.loads((tmp_path / "checkpoints/metadata.json").read_text())
    assert [item["id"] for item in metadata["variants"]] == ["pass_success/a", "pass_success/b"]
    assert {item["role"] for item in metadata["model_membership"]["entries"]} == {
        "pass_success", "outcome_scoring", "outcome_conceding"}
    assert (tmp_path / "checkpoints/s/pass_score_0.png").exists()


@pytest.fixture(params=[dv, av], ids=["hawkeye", "sportec"])
def outcome_pipeline(request, tmp_path, monkeypatch, inference_fixture):
    script = request.param
    subject, models, _, build = inference_fixture
    required = []
    if script is av:
        required = ["--match-id", "match"]
        subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"],
            spadl_type=["pass"], action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
        monkeypatch.setattr(av, "resolve_runtime_feature_run_context", lambda *a, **kw: {
            "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
        build = Mock(return_value=subject)
        monkeypatch.setattr(av, "load_match", build)
        monkeypatch.setattr(av, "resolve_action_indices", lambda *a: [(10, "42")])
        monkeypatch.setattr(av, "configure_lane_survival_runtime_cache", lambda *a: None)
    values = {"outcome_scoring/a": (0.10, 0.80), "outcome_scoring/b": (0.25, 0.50),
              "outcome_conceding/x": (0.02, 0.10), "outcome_conceding/y": (0.30, 0.20),
              "pass_success/fixed": (0.60,)}
    calls = []
    def predict(scene, model, **kwargs):
        masked = bool(getattr(model, "inference_masks", None))
        calls.append((model.identifier, masked))
        frames = [pd.DataFrame({"home_1": [0.], "home_2": [value * (0.5 if masked else 1.0)],
                                "away_3": [0.05]}, index=[10]) for value in values[model.identifier]]
        return (frames[0], frames[1]) if len(frames) == 2 else (frames[0], None)
    monkeypatch.setattr(script, "inference_gnn", predict)
    render = Mock(side_effect=lambda *a, **kw: Image.new("RGB", (100, 70), "green"))
    monkeypatch.setattr(script, "render_component" if script is av else "render_frame_image", render)
    return script, required, models, calls, build, render


def test_paired_outcomes_end_to_end(tmp_path, monkeypatch, outcome_pipeline):
    from scripts import visualization_comparison as comparison
    script, required, models, calls, build, render = outcome_pipeline
    pairs = [("a", "x"), ("b", "x"), ("b", "y")]
    monkeypatch.setattr(inference_compare, "load_bundle_record", lambda identifier: {
        "model_ids": {"pass_success": "pass_success/fixed", "outcome_scoring": "outcome_scoring/unused",
                      "outcome_conceding": "outcome_conceding/unused"}})
    compose = Mock(wraps=compose_panels)
    monkeypatch.setattr(comparison, "compose_panels", compose)
    args = script.parse_args([*required, "--compare", *_outcome_pair_flags(pairs), "--bundle-id", "fixed",
        "--only-outcome-success", "--only-outcome-failure", "--only-pass-score", "--inference-mask", "all:accel",
        "--output-dir", str(tmp_path), "--run-id", "pairs"])
    monkeypatch.setattr(script, "parse_args", lambda: args)
    script.main()
    assert build.call_count == 1
    assert len(calls) == len(set(calls)) == 5
    assert all(masked for _, masked in calls)
    assert set(models) == {"pass_success/fixed", "outcome_scoring/a", "outcome_scoring/b",
                           "outcome_conceding/x", "outcome_conceding/y"}
    assert all(model.inference_masks is None for model in models.values())
    assert render.call_count == 9 and compose.call_count == 3
    root = tmp_path / "pairs"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == "completed" and metadata["comparison_mode"] == "checkpoints"
    assert len(metadata["output_paths"]) == 3
    assert all(Path(path).exists() for path in metadata["output_paths"])
    aligned = pd.read_csv(root / "aligned_predictions.csv")
    # Expected values use the common mask's 0.5 factor, including p_success=0.30.
    expected = [(0.35, 0.04, 0.133), (0.20, 0.115, 0.1405), (0.15, -0.025, 0.0275)]
    for index, (variant, pair, (success, failure, score)) in enumerate(zip(metadata["variants"], pairs, expected)):
        scoring, conceding = f"outcome_scoring/{pair[0]}", f"outcome_conceding/{pair[1]}"
        assert json.loads(variant["id"].removeprefix("outcome_pair:")) == [scoring, conceding]
        assert variant["provenance"]["model_ids"] == {
            "pass_success": "pass_success/fixed", "outcome_scoring": scoring, "outcome_conceding": conceding}
        assert variant["requested_masks"] == ["all:accel"]
        assert scoring in variant["label"] and conceding in variant["label"]
        column = "prediction:" + variant["id"]
        for component, value in (("outcome_success", success), ("outcome_failure", failure), ("pass_score", score)):
            row = aligned.loc[aligned.component.eq(component) & aligned.candidate_id.eq("home_2")]
            assert row[column].iloc[0] == pytest.approx(value)
        for invocation in compose.call_args_list:
            label = invocation.args[1][index]
            assert scoring in label and conceding in label and "Masked: all:accel" in label


@pytest.mark.parametrize("field", ["target_family", "return_type"])
def test_within_pair_warnings_require_acceptance(tmp_path, monkeypatch, outcome_pipeline, field):
    from scripts.comparison_warnings import ComparisonCancelled
    script, required, _, calls, build, render = outcome_pipeline
    alternative = "xt" if field == "target_family" else "next_10"
    def record(identifier):
        result = _record(identifier)
        # Each task is constant across variants, so only the within-pair check detects this.
        if identifier.startswith("outcome_conceding/"):
            result[field] = alternative
        return result
    monkeypatch.setattr(inference_compare, "get_model_record", record)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    flags = [*required, "--compare", *_outcome_pair_flags([("a", "x"), ("b", "y")]),
             "--only-outcome-success", "--output-dir", str(tmp_path), "--run-id", "warnings"]
    args = script.parse_args(flags)
    with pytest.raises(ComparisonCancelled, match="explicit acceptance"):
        inference_compare.run_comparison(args, script)
    assert not calls and not build.call_count and not render.call_count
    path = tmp_path / "warnings/metadata.json"
    metadata = json.loads(path.read_text())
    assert metadata["status"] == "cancelled"
    assert all(any("Within outcome pair" in message and field in message and variant["id"] in message
                   for message in metadata["warnings"]) for variant in metadata["variants"])
    args = script.parse_args([*flags, "--accept-comparison-warnings"])
    inference_compare.run_comparison(args, script)
    metadata = json.loads(path.read_text())
    assert metadata["status"] == "completed"
    assert metadata["warning_acceptances"][0]["method"] == "command_line"
    assert set(metadata["warning_acceptances"][0]["warnings"]) == set(metadata["warnings"])
    assert calls and render.call_count == 2


@pytest.mark.parametrize("definitions", ["different_between_pairs", "missing"])
def test_pair_definition_differences_and_missing_metadata(tmp_path, monkeypatch, outcome_pipeline, definitions):
    script, required, _, _, _, _ = outcome_pipeline
    def record(identifier):
        result = _record(identifier)
        if definitions == "missing" and identifier.startswith("outcome_conceding/"):
            result["target_family"] = result["return_type"] = None
        elif definitions == "different_between_pairs" and identifier.endswith(("/b", "/y")):
            result.update(target_family="xt", return_type="next_10")
        return result
    monkeypatch.setattr(inference_compare, "get_model_record", record)
    args = script.parse_args([*required, "--compare", *_outcome_pair_flags([("a", "x"), ("b", "y")]),
        "--only-outcome-success", "--accept-comparison-warnings", "--output-dir", str(tmp_path), "--run-id", "definitions"])
    inference_compare.run_comparison(args, script)
    metadata = json.loads((tmp_path / "definitions/metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert not any("Within outcome pair" in message for message in metadata["warnings"])
    if definitions == "different_between_pairs":
        assert any("target_family" in message for message in metadata["warnings"])
        assert any("return_type" in message for message in metadata["warnings"])
        assert metadata["warning_acceptances"]
    else:
        assert not metadata["warnings"]


def test_masked_auxiliary_changes_v5_prediction_cache_dependency():
    from scripts.visualization_comparison import cached_prediction
    from models.inference_masks import configure_inference_masks
    primary = TinyModel("pass_success/a")
    intent = TinyModel("pass_intent/auxiliary")
    height = TinyModel("pass_height/auxiliary")
    primary.args["xpass_weight"] = "v5"
    primary._xpass_auxiliary_models = {"pass_intent": intent, "pass_height": height}
    models = {"pass_success": primary, "xpass_pass_intent": intent, "xpass_pass_height": height}
    cache, calls, subject = {}, [], object()
    def predict(scene, model, **kwargs):
        calls.append(bool(getattr(intent, "inference_masks", None)))
        return pd.DataFrame({"home_2": [.5 if calls[-1] else .7]}, index=[10]), None
    configure_inference_masks(models, [])
    first = cached_prediction(cache, predict, subject, primary, event_indices=[10])[0]
    cached_prediction(cache, predict, subject, primary, event_indices=[10])
    assert len(calls) == 1
    configure_inference_masks(models, ["pass_intent:accel"])
    second = cached_prediction(cache, predict, subject, primary, event_indices=[10])[0]
    assert calls == [False, True]
    assert first.at[10, "home_2"] != second.at[10, "home_2"]



def test_prediction_reuse_requires_matching_original_state_inputs():
    from datatools import config
    from scripts.visualization_comparison import cached_prediction
    from torch_geometric.data import Data
    model = TinyModel("pass_height/auxiliary")
    source = Data(x=torch.zeros((2, 26)), edge_index=torch.tensor([[0, 1], [1, 0]]),
                  edge_attr=torch.zeros((2, 2)), node_ids=["home_1", "home_2"])
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    scene, calls, cache = object(), [], {}
    def predict(scene, model, **kwargs):
        calls.append(float(kwargs["graph_override"][0].x[1, 3]))
        return pd.DataFrame({"home_2": [calls[-1]]}, index=[10]), None
    options = dict(graph_override=[source], label_override=labels, event_indices=[10])
    cached_prediction(cache, predict, scene, model, **options)
    cached_prediction(cache, predict, scene, model, **options)
    assert len(calls) == 1
    source.x[1, 3] = 20
    result = cached_prediction(cache, predict, scene, model, **options)[0]
    assert calls == [0., 20.]
    assert result.at[10, "home_2"] == 20.


def _raw_hawkeye(value=0.4):
    return pd.DataFrame(dict(id=["s", "s"], abs_time=[0., 0.], possession_team=["A", "A"],
        team=["A", "A"], uefa_player_id=[1, 2], pass_intent=[0.0, value]))


@pytest.fixture
def saved_fixture(monkeypatch):
    subject = _subject()
    monkeypatch.setattr(hv, "get_hawkeye_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(hv, "load_hawkeye_component_run", lambda path: (_raw_hawkeye(0.4 if str(path) == "a" else 0.6),
        {"status": "completed", "processed_situation_ids": ["s"], "freeze_ballreceipt": True}))
    monkeypatch.setattr(hv, "load_hawkeye_tracking", lambda path: pd.DataFrame({"id": ["s"]}))
    monkeypatch.setattr(hv, "clean_hawkeye_tracking", lambda value: value)
    monkeypatch.setattr(hv, "load_hawkeye_ball", lambda path: pd.DataFrame())
    monkeypatch.setattr(hv, "clean_hawkeye_ball", lambda value: value)
    build = Mock(return_value=(subject, None, None))
    monkeypatch.setattr(hv, "build_hawkeye_situation", build)
    monkeypatch.setattr(hv, "resolve_ballreceipt", lambda value: 0.)
    monkeypatch.setattr(hv, "resolve_hawkeye_png_frames", lambda *a: [{"frame_id": 10, "label": "0", "resolved_time_norm": 0.}])
    monkeypatch.setattr(hv, "load_overlay_data", lambda **kw: None)
    monkeypatch.setattr(hv, "build_situation_overlays", lambda *a, **kw: (pd.Series(dtype=float), pd.Series(dtype=str), {"selection_denominators": {}}))
    return subject, build


def test_saved_hawkeye_synthetic_end_to_end(tmp_path, monkeypatch, saved_fixture):
    subject, build = saved_fixture
    args = hv.parse_args(["--accept-comparison-warnings", "--compare", "--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent",
                          "--output-dir", str(tmp_path), "--run-id", "saved"])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    hv.main()
    assert build.call_count == 2
    root = tmp_path / "saved"
    assert (root / "s/pass_intent_0.png").exists()
    table = pd.read_csv(root / "aligned_predictions.csv")
    assert table.loc[table.candidate_id.eq("home_2"), "prediction:a"].iloc[0] == 0.4
    assert table.loc[table.candidate_id.eq("home_2"), "prediction:b"].iloc[0] == pytest.approx(0.6)


@pytest.mark.parametrize("override", [False, True])
def test_saved_comparison_resolves_pc_xpass_per_variant(tmp_path, monkeypatch, saved_fixture, override):
    from test_saved_visualization_xpass import component, write_cache
    caches = {key: write_cache(tmp_path / key) for key in ("a", "b", "override")}
    def load(path):
        key = str(path)
        return _raw_hawkeye(), component(run_id=key, status="completed", physical_cache_dir=str(caches[key]),
                                        processed_situation_ids=["s"], freeze_ballreceipt=True,
                                        x_pass_version="top10" if key == "a" else "top25")
    monkeypatch.setattr(hv, "load_hawkeye_component_run", load)
    def physical(cache, *a, **kw):
        value = {caches["a"]: .2, caches["b"]: .8, caches["override"]: .5}[cache]
        return pd.DataFrame({"home_1": [0.], "home_2": [value]}, index=[10])
    loader = Mock(side_effect=physical)
    monkeypatch.setattr(hv, "load_runtime_physical_xpass_visualization_table", loader)
    args = hv.parse_args(["--compare", "--component-run-id", "a", "--component-run-id", "b",
                          "--only-pass-intent", "--show-physical-xpass", "--output-dir", str(tmp_path),
                          "--run-id", "comparison", *(["--physical-cache-dir", str(caches["override"]),
                                                       "--xpass-version", "max"] if override else [])])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    hv.main()
    table = pd.read_csv(tmp_path / "comparison/aligned_predictions.csv")
    row = table.loc[table.component.eq("physical_xpass") & table.candidate_id.eq("home_2")].iloc[0]
    assert row["prediction:a"] == pytest.approx(.5 if override else .2)
    assert row["prediction:b"] == pytest.approx(.5 if override else .8)
    assert loader.call_count == 2
    assert [call.kwargs["x_pass_version"] for call in loader.call_args_list] == (
        ["max", "max"] if override else ["top10", "top25"])
    metadata = json.loads((tmp_path / "comparison/metadata.json").read_text())
    for key, variant in zip(("a", "b"), metadata["variants"]):
        provenance = variant["provenance"]
        assert Path(provenance["physical_cache_dir"]) == caches["override" if override else key]
        assert provenance["physical_xpass_resolution"]["inherited_fields"] == ([] if override else ["cache", "metric"])
    # Settings remain CLI overrides only, rather than leaking the first variant into the second.
    assert args.pc_xpass_id is None
    assert args.physical_cache_dir == (str(caches["override"]) if override else None)


def test_saved_comparison_rejects_unresolved_variant_before_rendering(tmp_path, monkeypatch, saved_fixture):
    from test_saved_visualization_xpass import component, write_cache
    cache = write_cache(tmp_path / "cache")
    def load(path):
        metadata = component(physical_cache_dir=str(cache)) if str(path) == "a" else {}
        return _raw_hawkeye(), metadata
    monkeypatch.setattr(hv, "load_hawkeye_component_run", load)
    render = Mock()
    monkeypatch.setattr(hv, "render_frame_image", render)
    args = hv.parse_args(["--compare", "--component-run-id", "a", "--component-run-id", "b",
                          "--only-pass-intent", "--show-physical-xpass", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="Component run 'b'.*usage is unknown"):
        hv.main()
    render.assert_not_called()
    saved_fixture[1].assert_not_called()


@pytest.mark.parametrize("show_xpass", [False, True])
def test_standalone_hawkeye_uses_saved_pc_settings_only_for_requested_panel(tmp_path, monkeypatch, saved_fixture, show_xpass):
    from test_saved_visualization_xpass import component, write_cache
    cache = write_cache(tmp_path / "cache")
    monkeypatch.setattr(hv, "load_hawkeye_component_run", lambda path: (
        _raw_hawkeye(), component(physical_cache_dir=str(cache), processed_situation_ids=["s"])))
    monkeypatch.setattr(hv, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    loader = Mock(return_value=pd.DataFrame({"home_1": [0.], "home_2": [.8]}, index=[10]))
    monkeypatch.setattr(hv, "load_runtime_physical_xpass_visualization_table", loader)
    args = hv.parse_args(["--component-dir", "a", "--only-pass-intent", "--output-dir", str(tmp_path),
                          "--run-id", "standalone", *(["--show-physical-xpass"] if show_xpass else [])])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    hv.main()
    metadata = json.loads((tmp_path / "standalone/metadata.json").read_text())
    if show_xpass:
        loader.assert_called_once_with(cache, "s", [10], metric="top25_xpass", x_pass_version="top25")
        assert metadata["x_pass_version"] == "top25"
        assert (tmp_path / "standalone/s/physical_xpass_0.png").exists()
    else:
        loader.assert_not_called()
        assert metadata["physical_cache_dir"] is None


@pytest.mark.parametrize("problem", ["duplicate", "missing", "incomplete", "definition", "different_scene"])
def test_saved_comparison_fails_without_silent_cohort_changes(tmp_path, monkeypatch, saved_fixture, problem):
    def load(path):
        raw = _raw_hawkeye()
        metadata = {"status": "completed", "processed_situation_ids": ["s"], "freeze_ballreceipt": True}
        if str(path) == "b":
            if problem == "duplicate":
                raw = pd.concat([raw, raw.iloc[:1]], ignore_index=True)
            elif problem == "missing":
                raw["abs_time"] = 0.1
            elif problem == "incomplete":
                metadata["status"] = "running"
            elif problem == "definition":
                metadata["freeze_ballreceipt"] = False
            else:
                metadata["processed_situation_ids"] = ["other"]
        return raw, metadata
    monkeypatch.setattr(hv, "load_hawkeye_component_run", load)
    args = hv.parse_args(["--accept-comparison-warnings", "--compare", "--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent",
                          "--output-dir", str(tmp_path), "--run-id", "failed"])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    if problem == "different_scene":
        with pytest.raises((ValueError, KeyError)):
            hv.main()
        # The valid earlier scene remains available when a later scene fails.
        image = tmp_path / "failed/s/pass_intent_0.png"
        assert image.exists()
        metadata = json.loads((tmp_path / "failed/metadata.json").read_text())
        assert metadata["status"] == "failed"
        assert str(image.resolve()) in metadata["output_paths"]
    else:
        hv.main()
        assert list(tmp_path.rglob("*.png"))


def test_real_snapshot_na_and_identical_canvas(monkeypatch):
    subject = _subject()
    original = subject.tracking.copy(deep=True)
    texts = []
    import matplotlib.axes
    annotate = matplotlib.axes.Axes.annotate
    def record(ax, text, *a, **kw):
        texts.append(text)
        return annotate(ax, text, *a, **kw)
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", record)
    first = hv.render_frame_image(subject, 10, "pass_intent", pd.Series({"home_1": 0., "home_2": np.nan}), comparison=True)
    second = hv.render_frame_image(subject, 10, "pass_intent", pd.Series({"home_1": 0.3, "home_2": 0.8}), comparison=True)
    assert first.size == second.size
    assert "N/A" in texts and "0.000" in texts
    pd.testing.assert_frame_equal(subject.tracking, original)


def test_location_identity_uses_time_and_geometry_not_row_id():
    row = pd.Series(dict(_action_id="action", resolved_time_norm=0.1, PositionX=200, PositionY=300, selection_row_id=1))
    other = row.copy()
    other.selection_row_id = 9
    assert saved_compare.location_key(row, "loc") == saved_compare.location_key(other, "loc")
    other.PositionX = 300
    assert saved_compare.location_key(row, "loc") != saved_compare.location_key(other, "loc")


def test_incomplete_checkpoint_and_invalid_scope_fail_before_inference(tmp_path, monkeypatch, inference_fixture):
    _, _, calls, build = inference_fixture
    args = dv.parse_args(["--accept-comparison-warnings", "--pass-success-model-id", "pass_success/a", "--only-pass-success",
                          "--inference-mask-1", "pass_intent:accel", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="no participating"):
        inference_compare.run_comparison(args, dv)
    assert not calls and not build.call_count
    monkeypatch.setattr(inference_compare, "get_model_record", lambda identifier: {**_record(identifier), "is_complete": False})
    args = dv.parse_args(["--accept-comparison-warnings", "--pass-success-model-id", "pass_success/a", "--only-pass-success",
                          "--inference-mask-1", "accel", "--output-dir", str(tmp_path)])
    inference_compare.run_comparison(args, dv)


@pytest.mark.parametrize("mode", ["checkpoints", "masks"])
def test_sportec_inference_end_to_end(tmp_path, monkeypatch, inference_fixture, mode):
    subject, models, calls, _ = inference_fixture
    subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"], spadl_type=["pass"],
        action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
    monkeypatch.setattr(av, "resolve_runtime_feature_run_context", lambda *a, **kw: {
        "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
    load = Mock(return_value=subject)
    monkeypatch.setattr(av, "load_match", load)
    monkeypatch.setattr(av, "resolve_action_indices", lambda *a: [(10, "42")])
    monkeypatch.setattr(av, "inference_gnn", dv.inference_gnn)
    monkeypatch.setattr(av, "configure_lane_survival_runtime_cache", lambda *a: None)
    flags = ["--compare", "--pass-success-model-id", "pass_success/b", "--inference-mask", "accel"] if mode == "checkpoints" else [
        "--inference-mask-1", "accel", "--inference-mask-2", "pos-node-features"]
    args = av.parse_args(["--accept-comparison-warnings", "--match-id", "match", "--pass-success-model-id", "pass_success/a", "--only-pass-success",
                          "--output-dir", str(tmp_path), "--run-id", mode, *flags])
    monkeypatch.setattr(av, "parse_args", lambda: args)
    av.main()
    assert load.call_count == 1
    assert len(calls) == (2 if mode == "checkpoints" else 3)
    assert (tmp_path / mode / "match/42/pass_success.png").exists()
    metadata = json.loads((tmp_path / mode / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    if mode == "masks":
        requested = [item["provenance"]["inference_masks"]["requested"] for item in metadata["variants"]]
        assert requested == [[], ["accel"], ["pos-node-features"]]


@pytest.mark.parametrize("show_xpass", [False, True])
@pytest.mark.parametrize("compare", [False, True])
def test_skillcorner_saved_visualization_preserves_shot_export(tmp_path, monkeypatch, show_xpass, compare):
    from test_saved_visualization_xpass import component, write_cache
    cache = write_cache(tmp_path / "cache")
    metadata = component(physical_cache_dir=str(cache)) if show_xpass else {}
    subject = _subject()
    subject.event_index, subject.fps = 1, 25
    monkeypatch.setattr(sv, "get_skillcorner_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(sv, "load_run_metadata", lambda *a, **kw: {"status": "completed", **metadata})
    physical = Mock(return_value=pd.DataFrame({"home_1": [0., 0.], "home_2": [.2, .2]}, index=[10, 11]))
    monkeypatch.setattr(sv, "load_runtime_physical_xpass_visualization_table", physical)
    selection = Mock(return_value=object())
    monkeypatch.setattr(sv.match_cache, "selection_for", selection)
    def load(folder, match):
        value = 0.4 if str(folder) == "a" else 0.6
        return {"pass_success": pd.DataFrame(),
                "action_intent": pd.DataFrame({"index": [1, 1], "frame": [10, 11], "1": [0., 0.],
                                               "2": [value, value], "shot": [0.2, 0.2]})}
    monkeypatch.setattr(sv, "load_skillcorner_component_tables", load)
    monkeypatch.setattr(sv, "build_skillcorner_match_context", lambda *a: object())
    build = Mock(return_value=(subject, None))
    monkeypatch.setattr(sv, "build_skillcorner_possession", build)
    selection_flags = ["--compare", "--component-run-id", "a", "--component-run-id", "b"] if compare else ["--component-dir", "a"]
    args = sv.parse_args(["--accept-comparison-warnings", "--match-id", "m", "--index", "1", *selection_flags,
                          "--only-action-intent", "--output-dir", str(tmp_path), "--run-id", "sc",
                          *(["--show-physical-xpass"] if show_xpass else [])])
    monkeypatch.setattr(sv, "parse_args", lambda: args)
    sv.main()
    assert build.call_count == (2 if compare else 1)
    assert (tmp_path / "sc/m/1/action_intent_first.png").exists()
    assert (tmp_path / "sc/m/1/action_intent_last.png").exists()
    if compare:
        table = pd.read_csv(tmp_path / "sc/aligned_predictions.csv")
        assert "home_goal" in set(table.candidate_id)
        assert table.loc[table.candidate_id.eq("home_goal"), "prediction:a"].iloc[0] == pytest.approx(0.2)
    if show_xpass:
        assert physical.call_count == selection.call_count == (2 if compare else 1)
        assert physical.call_args.args[0] == cache
        assert physical.call_args.kwargs["metric"] == "top25_xpass"
        assert physical.call_args.kwargs["cache_selection"] is selection.return_value
        assert (tmp_path / "sc/m/1/physical_xpass_first.png").exists()
    else:
        physical.assert_not_called()
        selection.assert_not_called()


@pytest.mark.parametrize("show_xpass", [False, True])
@pytest.mark.parametrize("compare", [False, True])
def test_benchmark_saved_visualization_real_states(tmp_path, monkeypatch, show_xpass, compare):
    from datatools.benchmark import build_benchmark_state
    from test_saved_visualization_xpass import component, write_cache
    cache = write_cache(tmp_path / "cache")
    metadata = component(physical_cache_dir=str(cache)) if show_xpass else {}
    source = pd.DataFrame(dict(team=[0, 1, 1, 2], player=[0, 1, 2, 3], pos_x=[0., 0., 10., 20.],
        pos_y=[0., 0., 5., 10.], pos_z=[0., np.nan, np.nan, np.nan], smooth_x_speed=[0.] * 4,
        smooth_y_speed=[0.] * 4, event_player=[1] * 4, playing_direction_event=[True] * 4))
    raw = []
    for state_id in (1, 2):
        _, export, _ = build_benchmark_state(source, 1, state_id, 2, build_graphs=False)
        export["pass_intent"] = np.where(export.team.eq(1), 0.3, np.nan)
        raw.append(export)
    export = pd.concat(raw, ignore_index=True)
    monkeypatch.setattr(bv, "get_benchmark_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(bv, "load_benchmark_component_run", lambda folder: (export.copy(), {"status": "completed", **metadata}))
    physical = Mock(return_value=pd.DataFrame({"home_1": [0.], "home_2": [.2]}, index=[0]))
    monkeypatch.setattr(bv, "load_runtime_physical_xpass_visualization_table", physical)
    monkeypatch.setattr(bv, "load_benchmark_modification_data", lambda *a: {"game_state_1": source, "game_state_2": source, "higher_state_id": 2})
    build = Mock(wraps=build_benchmark_state)
    monkeypatch.setattr(bv, "build_benchmark_state", build)
    selection_flags = ["--compare", "--component-run-id", "a", "--component-run-id", "b", "--component-run-id", "c"] if compare else ["--component-dir", "a"]
    args = bv.parse_args(["--accept-comparison-warnings", *selection_flags,
                          "--only-pass-intent", "--output-dir", str(tmp_path), "--run-id", "benchmark",
                          *(["--show-physical-xpass"] if show_xpass else [])])
    monkeypatch.setattr(bv, "parse_args", lambda: args)
    bv.main()
    assert build.call_count == (6 if compare else 2)
    image = Image.open(tmp_path / "benchmark/modification_1/pass_intent.png")
    if compare:
        assert image.width > image.height
        table = pd.read_csv(tmp_path / "benchmark/aligned_predictions.csv")
        assert set(table.scene_id) == {"modification_1/game_state_1", "modification_1/game_state_2"}
        assert len([name for name in table if name.startswith("prediction:")]) == 3
    if show_xpass:
        assert physical.call_count == (6 if compare else 2)
        assert physical.call_args.args[0] == cache
        assert physical.call_args.kwargs["x_pass_version"] == "top25"
        assert (tmp_path / "benchmark/modification_1/physical_xpass.png").exists()
    else:
        physical.assert_not_called()


def test_hawkeye_geometry_conflict_and_explicit_unselected_data():
    first = _raw_hawkeye()
    first["centroid_x"] = [0., 10.]
    other = first.copy()
    other.loc[1, "centroid_x"] = 11.
    with pytest.raises(ValueError, match="different original scene geometry"):
        saved_compare.validate_hawkeye_geometry([first, other])


def test_mask_status_and_real_model_prediction_cache():
    from models.gnn import GNN
    from models.inference_masks import configure_inference_masks
    from torch_geometric.data import Batch, Data
    from scripts.visualization_comparison import cached_prediction
    torch.manual_seed(42)
    model = GNN(_args()).eval()
    x = torch.ones((3, 26))
    x[:, :3] = 0
    x[:2, 0] = 1
    x[:, 13] = torch.tensor([1., 0., 0.])
    graph = Data(x=x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), edge_attr=torch.ones((4, 5)))
    graph.inference_possessor_mask = x[:, 13].bool()
    batch = Batch.from_data_list([graph])
    original = batch.clone()
    calls = []
    def predict(scene, checkpoint, **kwargs):
        calls.append(1)
        with torch.no_grad():
            values = checkpoint(batch).reshape(-1).numpy()
        return pd.DataFrame([values]), None
    cache, subject = {}, object()
    configure_inference_masks({"selected": model}, [])
    unmasked, _ = cached_prediction(cache, predict, subject, model)
    cached, _ = cached_prediction(cache, predict, subject, model)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(unmasked, cached)
    metadata = configure_inference_masks({"selected": model}, ["accel"])
    masked, _ = cached_prediction(cache, predict, subject, model)
    assert len(calls) == 2 and not np.array_equal(unmasked.values, masked.values)
    assert metadata["models"]["selected"]["status"] == "applied"
    torch.testing.assert_close(batch.x, original.x)
    torch.testing.assert_close(batch.edge_attr, original.edge_attr)


def test_physical_table_stays_identical_across_masks(tmp_path, monkeypatch, inference_fixture):
    subject, models, calls, _ = inference_fixture
    table = pd.DataFrame({"home_1": [0.], "home_2": [0.9]}, index=[10])
    original = table.copy(deep=True)
    physical = Mock(return_value=table)
    monkeypatch.setattr(dv, "load_runtime_physical_xpass_visualization_table", physical)
    args = dv.parse_args(["--accept-comparison-warnings", "--pass-success-model-id", "pass_success/a", "--only-pass-success", "--show-physical-xpass",
        "--inference-mask-1", "pos-node-features", "--output-dir", str(tmp_path), "--run-id", "physical"])
    monkeypatch.setattr(dv, "parse_args", lambda: args)
    dv.main()
    assert physical.call_count == 1 and len(calls) == 2
    pd.testing.assert_frame_equal(table, original)
    aligned = pd.read_csv(tmp_path / "physical/aligned_predictions.csv")
    physical_rows = aligned.loc[aligned.component.eq("physical_xpass")]
    np.testing.assert_array_equal(physical_rows["prediction:unmasked"], physical_rows["prediction:mask_1"])


def test_checkpoint_selection_allows_training_size_architecture_and_feature_differences(monkeypatch, inference_fixture):
    def record(identifier):
        result = _record(identifier)
        result.update(train_count=10 if identifier.endswith("/a") else 20, split_manifest_id=identifier,
                      model_name="gat" if identifier.endswith("/a") else "gcn")
        result["args"].update(seed=1 if identifier.endswith("/a") else 2,
                              node_emb_dim=8 if identifier.endswith("/a") else 16,
                              pos_node_features_aware=identifier.endswith("/a"))
        return result
    monkeypatch.setattr(inference_compare, "get_model_record", record)
    args = dv.parse_args(["--accept-comparison-warnings", "--compare", "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b",
                          "--only-pass-success"])
    variants, _, _, _, _ = inference_compare.resolve_variants(args, dv, False)
    assert [item.identifier for item in variants] == ["pass_success/a", "pass_success/b"]


@pytest.mark.parametrize("problem", ["duplicate", "missing"])
def test_skillcorner_rejects_raw_selected_rows(tmp_path, monkeypatch, problem):
    subject = _subject()
    subject.event_index, subject.fps = 1, 25
    raw = pd.DataFrame({"index": [1, 1], "frame": [10, 11], "1": [0.1, 0.1]})
    broken = pd.concat([raw, raw.iloc[:1]]) if problem == "duplicate" else raw.iloc[:1]
    args = sv.parse_args(["--accept-comparison-warnings", "--match-id", "m", "--index", "1", "--compare", "--component-run-id", "a",
                          "--component-run-id", "b", "--only-pass-intent"])
    monkeypatch.setattr(sv, "build_skillcorner_match_context", lambda *a: object())
    monkeypatch.setattr(sv, "build_skillcorner_possession", lambda *a: (subject, None))
    groups = list(saved_compare.skillcorner_groups(args, sv, [{"pass_intent": raw}, {"pass_intent": broken}], [{}, {}], ["pass_intent"]))
    assert groups


def test_hawkeye_location_comparison_matches_different_selection_row_ids(tmp_path, monkeypatch, saved_fixture):
    def load(path):
        raw = _raw_hawkeye(0.4 if str(path) == "a" else 0.6)
        raw["selection_row_id"] = 1 if str(path) == "a" else 9
        raw["original_action_id"] = "s"
        raw["PositionX"], raw["PositionY"] = 200., 300.
        raw["resolved_time_norm"], raw["loc_situation_id"] = 0., "cache_identity"
        return raw, {"status": "completed", "inference_mode": "loc"}
    monkeypatch.setattr(hv, "get_hawkeye_loc_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(hv, "load_hawkeye_component_run", load)
    monkeypatch.setattr(hv, "apply_hawkeye_possessor_offset", lambda tracking, **kw: (tracking, {}))
    args = hv.parse_args(["--accept-comparison-warnings", "--mode", "loc", "--compare", "--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent",
                          "--output-dir", str(tmp_path), "--run-id", "loc"])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    hv.main()
    assert (tmp_path / "loc/s_1/pass_intent.png").exists()
    metadata = json.loads((tmp_path / "loc/metadata.json").read_text())
    assert metadata["scenes"][0]["location_key"] == ["s", 0., 2., -3.]


def test_candidate_membership_warning_is_exported_in_saved_comparison(tmp_path, monkeypatch, saved_fixture):
    def load(path):
        raw = _raw_hawkeye()
        if str(path) == "b":
            raw = raw.iloc[1:].copy()
        return raw, {"status": "completed"}
    monkeypatch.setattr(hv, "load_hawkeye_component_run", load)
    args = hv.parse_args(["--accept-comparison-warnings", "--compare", "--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent",
                          "--output-dir", str(tmp_path), "--run-id", "candidates"])
    monkeypatch.setattr(hv, "parse_args", lambda: args)
    with pytest.warns(UserWarning, match="membership"):
        hv.main()
    metadata = json.loads((tmp_path / "candidates/metadata.json").read_text())
    assert metadata["warnings"] and metadata["status"] == "completed"
    table = pd.read_csv(tmp_path / "candidates/aligned_predictions.csv", keep_default_na=False)
    assert table.loc[table.candidate_id.eq("home_1"), "prediction:b"].iloc[0] == "N/A"


@pytest.mark.parametrize("mode", ["checkpoints", "masks", "independent"])
@pytest.mark.parametrize("stop", [None, RuntimeError, KeyboardInterrupt])
def test_sportec_saves_before_predicting_next_action(tmp_path, monkeypatch, inference_fixture, mode, stop):
    subject, models, _, _ = inference_fixture
    subject.actions = pd.DataFrame(dict(frame_id=[10, 11], object_id=["home_1"] * 2,
        spadl_type=["pass"] * 2, action_type=["pass"] * 2, end_x=[40.] * 2, end_y=[25.] * 2), index=[10, 11])
    load = Mock(side_effect=lambda *a, **kw: copy.copy(subject))
    monkeypatch.setattr(av, "load_match", load)
    monkeypatch.setattr(av, "resolve_runtime_feature_run_context", lambda *a, **kw: {
        "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict"})
    monkeypatch.setattr(av, "resolve_action_indices", lambda *a: [(10, "42"), (11, "43")])
    monkeypatch.setattr(av, "configure_lane_survival_runtime_cache", lambda *a: None)
    if mode == "independent":
        monkeypatch.setattr(inference_compare, "get_model_record", lambda identifier: {
            **_record(identifier), "feature_run_id": identifier})
    root = tmp_path / "progressive"
    calls = []
    def collect(match, selected_models, feature_root, device, action_index, *a, **kw):
        calls.append(action_index)
        metadata = json.loads((root / "metadata.json").read_text())
        assert metadata["status"] == "running"
        if action_index == 11:
            path = root / "match/42/pass_success.png"
            with Image.open(path) as image:
                image.verify()
            assert str(path.resolve()) in metadata["output_paths"]
            assert not (root / "aligned_predictions.csv").exists()
            if stop:
                raise stop("later action failed")
        return {"pass_success": pd.Series({"home_1": 0., "home_2": .6})}
    monkeypatch.setattr(av, "collect_action_components", collect)
    monkeypatch.setattr(av, "render_component", lambda *a, **kw: Image.new("RGB", (80, 60)))
    flags = ["--inference-mask-1", "accel"] if mode == "masks" else [
        "--compare", "--pass-success-model-id", "pass_success/b"]
    args = av.parse_args(["--accept-comparison-warnings", "--match-id", "match", "--only-pass-success",
        "--pass-success-model-id", "pass_success/a", "--output-dir", str(tmp_path), "--run-id", "progressive", *flags])
    if stop is KeyboardInterrupt:
        with pytest.raises(stop, match="later action failed"):
            inference_compare.run_comparison(args, av)
    elif stop is RuntimeError:
        with pytest.warns(UserWarning, match="later action failed"):
            inference_compare.run_comparison(args, av)
    else:
        inference_compare.run_comparison(args, av)
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == ("cancelled" if stop is KeyboardInterrupt else "completed")
    if stop is RuntimeError:
        assert metadata["skipped_actions"][0]["action_id"] == "43"
    assert load.call_count == (2 if mode == "independent" else 1)
    assert (root / "match/42/pass_success.png").exists()
    assert (root / "match/43/pass_success.png").exists() == (stop is None)
    assert calls[:2] == [10, 10]
    assert all(model.inference_masks is None for model in models.values())


@pytest.mark.parametrize("saved", [False, True])
def test_hawkeye_saves_before_building_next_situation(tmp_path, monkeypatch, inference_fixture, saved_fixture, saved):
    subject, _, _, _ = inference_fixture
    script = hv if saved else dv
    root = tmp_path / "progressive"
    tracking = pd.DataFrame({"id": ["s", "t"]})
    monkeypatch.setattr(script, "load_hawkeye_tracking", lambda *a: tracking)
    def build(rows, *a, **kw):
        if rows.id.iloc[0] == "t":
            component = "pass_intent" if saved else "pass_success"
            assert (root / "s" / f"{component}_0.png").exists()
            assert json.loads((root / "metadata.json").read_text())["status"] == "running"
        return subject, None, None
    monkeypatch.setattr(script, "build_hawkeye_situation", build)
    monkeypatch.setattr(script, "render_frame_image", lambda *a, **kw: Image.new("RGB", (80, 60)))
    if saved:
        raw = pd.concat([_raw_hawkeye(), _raw_hawkeye().assign(id="t")], ignore_index=True)
        monkeypatch.setattr(script, "load_hawkeye_component_run", lambda *a: (
            raw, {"status": "completed", "processed_situation_ids": ["s", "t"]}))
        flags = ["--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent"]
    else:
        monkeypatch.setattr(script, "resolve_hawkeye_situation_ids", lambda *a, **kw: ["s", "t"])
        flags = ["--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b", "--only-pass-success"]
    args = script.parse_args(["--accept-comparison-warnings", "--compare", *flags,
                             "--output-dir", str(tmp_path), "--run-id", "progressive"])
    (saved_compare if saved else inference_compare).run_comparison(args, script)
    assert len(list(root.rglob("*.png"))) == 2


def test_skillcorner_saves_before_building_next_possession(tmp_path, monkeypatch):
    subject = _subject()
    subject.event_index, subject.fps = 1, 25
    root = tmp_path / "progressive"
    monkeypatch.setattr(sv, "get_skillcorner_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(sv, "load_run_metadata", lambda *a, **kw: {"status": "completed"})
    raw = pd.DataFrame({"index": [1, 1, 2, 2], "frame": [10, 11, 10, 11], "1": [.2] * 4})
    monkeypatch.setattr(sv, "load_skillcorner_component_tables", lambda *a: {"pass_intent": raw})
    monkeypatch.setattr(sv, "build_skillcorner_match_context", lambda *a: object())
    def build(context, index):
        if index == 2:
            assert (root / "m/1/pass_intent_first.png").exists()
            assert (root / "m/1/pass_intent_last.png").exists()
        return subject, None
    monkeypatch.setattr(sv, "build_skillcorner_possession", build)
    monkeypatch.setattr(sv, "render_frame_image", lambda *a, **kw: Image.new("RGB", (80, 60)))
    args = sv.parse_args(["--accept-comparison-warnings", "--match-id", "m", "--index", "1", "--index", "2",
        "--compare", "--component-run-id", "a", "--component-run-id", "b", "--only-pass-intent",
        "--output-dir", str(tmp_path), "--run-id", "progressive"])
    saved_compare.run_comparison(args, sv)
    assert len(list(root.rglob("*.png"))) == 4


def test_benchmark_saves_both_states_before_loading_next_modification(tmp_path, monkeypatch):
    root = tmp_path / "progressive"
    raw = pd.DataFrame({"modification": [1, 1, 2, 2], "game_state": [1, 2, 1, 2]})
    monkeypatch.setattr(bv, "get_benchmark_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(bv, "load_benchmark_component_run", lambda *a: (raw, {"status": "completed"}))
    def load(modification, *a):
        if modification == 2:
            assert (root / "modification_1/pass_intent.png").exists()
        return {"game_state_1": object(), "game_state_2": object(), "higher_state_id": 2}
    monkeypatch.setattr(bv, "load_benchmark_modification_data", load)
    monkeypatch.setattr(bv, "build_benchmark_state", lambda *a, **kw: (_subject(), None, None))
    monkeypatch.setattr(bv, "build_benchmark_component_tables", lambda *a: {
        "pass_intent": pd.DataFrame({"home_1": [.3]}, index=[10])})
    monkeypatch.setattr(bv, "render_state_image", lambda *a, **kw: Image.new("RGB", (80, 60)))
    args = bv.parse_args(["--accept-comparison-warnings", "--compare", "--component-run-id", "a",
        "--component-run-id", "b", "--only-pass-intent", "--output-dir", str(tmp_path), "--run-id", "progressive"])
    saved_compare.run_comparison(args, bv)
    table = pd.read_csv(root / "aligned_predictions.csv")
    assert len(set(table.scene_id)) == 4
    assert len(list(root.rglob("*.png"))) == 2
