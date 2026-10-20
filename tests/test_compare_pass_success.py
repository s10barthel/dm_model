"""Pass-success composition preserves existing images and prediction settings."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts import run_and_visualize_hawkeye as direct
from scripts import visualize_action_components as sportec
from scripts import visualize_benchmark as benchmark
from scripts import visualize_hawkeye as hawkeye
from scripts import visualize_skillcorner as skillcorner
from scripts.saved_visualization_xpass import resolve_saved_xpass
from scripts.visualization_pass_success import COMPARISON_COMPONENT, COMPARISON_INPUTS, save_component_pngs
from scripts.visualization_selection import resolve_component_selection
import project_config
from test_saved_visualization_xpass import write_cache, component
from test_visualization_comparison import TinyModel, _raw_hawkeye, _subject, inference_fixture, saved_fixture


COMMANDS = [(sportec, ["--match-id", "s"]), (direct, []), (hawkeye, []),
            (benchmark, []), (skillcorner, ["--match-id", "s", "--index", "1"])]


@pytest.mark.parametrize("module,required", COMMANDS)
@pytest.mark.parametrize("flag", ["--compare-pass-success", "--only-compare-pass-success"])
def test_cli_selection_and_conflicts(module, required, flag):
    args = module.parse_args([*required, flag])
    selection = resolve_component_selection(args)
    assert COMPARISON_COMPONENT in selection.rendered_components
    assert not args.show_physical_xpass and not args.show_model_pass_success
    if flag.startswith("--only"):
        assert selection.rendered_components == [COMPARISON_COMPONENT]
        assert selection.required_component_groups == ["pass_success"]
    conflicts = [["--compare"]]
    if module in (direct, sportec):
        conflicts += [["--inference-mask-1", "accel"]]
        assert module.parse_args([*required, flag, "--inference-mask", "accel"]).visualization_comparison is None
    if module not in (sportec, benchmark):
        conflicts += [["--output", "gif"], ["--output", "mp4"]]
    if module is hawkeye:
        conflicts += [["--only-tracking"]]
    for options in conflicts:
        with pytest.raises(SystemExit):
            module.parse_args([*required, flag, *options])


def test_selection_keeps_composite_inputs_when_standalone_is_hidden():
    args = direct.parse_args(["--only-compare-pass-success", "--only-pass-success", "--no-pass-success"])
    selection = resolve_component_selection(args)
    assert selection.rendered_components == [COMPARISON_COMPONENT]
    assert selection.required_component_groups == ["pass_success"]


@pytest.mark.parametrize("standalone", [False, True])
def test_pixels_order_padding_and_source_reuse(tmp_path, standalone):
    sizes = [(9, 5), (7, 3), (8, 4)]
    colors = ["red", "green", "blue"]
    calls = []
    def render(name):
        calls.append(name)
        index = COMPARISON_INPUTS.index(name)
        return Image.new("RGB", sizes[index], colors[index])
    names = [*COMPARISON_INPUTS, COMPARISON_COMPONENT] if standalone else [COMPARISON_COMPONENT]
    paths = save_component_pngs(names, render, tmp_path, "_first")
    assert len(calls) == len(set(calls)) == 3
    assert {Path(path).stem for path in paths} == {f"{name}_first" for name in names}
    assert len(list(tmp_path.glob("*.png"))) == len(names)
    with Image.open(tmp_path / "compare_pass_success_first.png") as combined:
        assert combined.size == (9, 12)
        offset = 0
        for size, color in zip(sizes, colors):
            expected = Image.new("RGB", (9, size[1]), "white")
            expected.paste(Image.new("RGB", size, color), (0, 0))
            np.testing.assert_array_equal(combined.crop((0, offset, 9, offset + size[1])), expected)
            offset += size[1]


@pytest.mark.parametrize("inherited", [False, True])
def test_cache_required_and_resolved_without_show_flags(tmp_path, inherited):
    cache = write_cache(tmp_path / "cache")
    options = [] if inherited else ["--physical-cache-dir", str(cache), "--xpass-version", "top25"]
    args = hawkeye.parse_args(["--only-compare-pass-success", *options])
    metadata = component(physical_cache_dir=str(cache)) if inherited else {}
    result = resolve_saved_xpass(args, metadata, "hawkeye")
    assert result.cache_dir == cache and result.metric == "top25_xpass"
    assert not args.show_physical_xpass
    with pytest.raises(ValueError, match="explicit"):
        resolve_saved_xpass(hawkeye.parse_args(["--only-compare-pass-success"]), {}, "hawkeye")


@pytest.mark.parametrize("inherited", [False, True])
def test_pinned_cache_id_never_uses_latest(tmp_path, monkeypatch, inherited):
    root = tmp_path / "pc_xpass"
    monkeypatch.setattr(project_config, "PC_XPASS_DIR", root)
    write_cache(root / "chosen")
    cache = write_cache(root / "chosen" / "hawkeye")
    (root / "latest.json").write_text('{"run_id": "wrong"}')
    options = [] if inherited else ["--pc-xpass-id", "chosen", "--xpass-version", "top25"]
    args = hawkeye.parse_args(["--only-compare-pass-success", *options])
    metadata = component(pc_xpass_id="chosen") if inherited else {}
    result = resolve_saved_xpass(args, metadata, "hawkeye")
    assert result.cache_dir == cache and result.pc_xpass_id == "chosen"
    with pytest.raises(ValueError, match="explicit"):
        resolve_saved_xpass(hawkeye.parse_args(["--only-compare-pass-success"]), {}, "hawkeye")


@pytest.mark.parametrize("module,required", COMMANDS[:2])
def test_missing_cache_fails_before_data_or_model_loading(module, required, monkeypatch):
    args = module.parse_args([*required, "--only-compare-pass-success"])
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: pytest.fail("Loaded a model before cache validation"))
    with pytest.raises(ValueError, match="explicit"):
        module.main()


def _capture_images(monkeypatch, module, function):
    original = getattr(module, function)
    images = {}
    def render(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        result = original(*args, **kwargs)
        images.setdefault(bound.arguments["component_name"], []).append(np.array(result))
        return result
    monkeypatch.setattr(module, function, render)
    return images


def _check_output(root, images, expected_count=1, paired=False):
    paths = list(root.rglob("*.png"))
    assert len(paths) == expected_count
    for path in paths:
        assert path.name.startswith("compare_pass_success")
        with Image.open(path) as image:
            # Every input keeps the exact pixels produced by the normal renderer.
            offset = 0
            for name in COMPARISON_INPUTS:
                frame_index = 1 if "last" in path.stem and len(images[name]) > 1 else 0
                sources = images[name] if paired else [images[name][frame_index]]
                for source in sources:
                    height, width = source.shape[:2]
                    np.testing.assert_array_equal(image.crop((0, offset, width, offset + height)), source)
                    offset += height
            assert image.height == offset
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["compare_pass_success"] and metadata["only_compare_pass_success"]
    assert set(metadata["compare_pass_success_output_paths"]) == {str(path.resolve()) for path in paths}


@pytest.mark.parametrize("module", [direct, sportec])
@pytest.mark.parametrize("blend", [False, True])
def test_inference_commands_render_once_without_changing_blending(
    tmp_path, monkeypatch, inference_fixture, module, blend,
):
    subject, _, _, _ = inference_fixture
    cache = write_cache(tmp_path / "cache")
    model = TinyModel("pass_success/a").eval()
    monkeypatch.setattr(module, "resolve_model_selection", lambda **kw: (
        {"pass_success": "pass_success/a"}, {"use_carries": True}, None))
    monkeypatch.setattr(module, "load_model", lambda *a, **kw: model)
    calls = []
    def predict(scene, selected, **kwargs):
        calls.append(kwargs)
        assert kwargs["return_model_pass_success"]
        learned = pd.DataFrame({"home_1": [.8], "home_2": [.7], "away_3": [.1]}, index=[10])
        return learned * .5 if blend else learned.copy(), learned
    monkeypatch.setattr(module, "inference_gnn", predict)
    physical_calls = []
    def physical(*args, **kwargs):
        physical_calls.append((args, kwargs))
        row = pd.Series({"home_1": .2, "home_2": .3})
        return row if module is sportec else pd.DataFrame([row], index=[10])
    loader = "load_runtime_physical_xpass_visualization_component" if module is sportec else "load_runtime_physical_xpass_visualization_table"
    monkeypatch.setattr(module, loader, physical)
    required = []
    if module is sportec:
        required = ["--match-id", "s"]
        subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"],
            spadl_type=["pass"], action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
        monkeypatch.setattr(module, "resolve_runtime_feature_run_context", lambda *a, **kw: {
            "feature_root": tmp_path, "feature_run_id": "feature", "intended_receiver_mode": "predict", "selection": {}})
        monkeypatch.setattr(module, "resolve_runtime_return_type", lambda *a: "next_5")
        monkeypatch.setattr(module, "load_match", lambda *a, **kw: subject)
        monkeypatch.setattr(module, "resolve_action_indices", lambda *a: [(10, "42")])
        monkeypatch.setattr(module, "configure_lane_survival_runtime_cache", lambda *a: None)
        monkeypatch.setattr(module, "possession_state_for_action", lambda *a: (subject, 10))
        monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
    original_renderer = module.render_component if module is sportec else module.render_frame_image
    images = _capture_images(monkeypatch, module, "render_component" if module is sportec else "render_frame_image")
    args = module.parse_args([*required, "--only-compare-pass-success", "--no-pass-success",
        "--pass-success-model-id", "pass_success/a", "--physical-cache-dir", str(cache),
        "--output-dir", str(tmp_path), "--run-id", "result", *(["--use-physical-xpass", "--pc-xpass"] if blend else [])])
    monkeypatch.setattr(module, "parse_args", lambda: args)
    module.main()
    assert len(calls) == 1 and len(physical_calls) == 1
    assert physical_calls[0][1]["metric"] == "top10_xpass"
    assert bool(model.args.get("inference_use_physical_xpass")) == blend
    assert bool(model.args.get("pc_xpass")) == blend
    assert all(len(values) == 1 for values in images.values())
    _check_output(tmp_path / "result", images)
    if module is sportec:
        original_path = tmp_path / "standalone.png"
        original_renderer(subject, 10, "42", "model_pass_success",
                          pd.Series({"home_1": .8, "home_2": .7, "away_3": .1}), original_path)
        with Image.open(original_path) as original:
            np.testing.assert_array_equal(original.convert("RGB"), images["model_pass_success"][0])


@pytest.mark.parametrize("mode", ["standard", "loc", "freeze"])
@pytest.mark.parametrize("missing_model", [False, True])
def test_saved_hawkeye_inherits_cache_and_preserves_views(tmp_path, monkeypatch, saved_fixture, mode, missing_model):
    cache = write_cache(tmp_path / "cache")
    raw = _raw_hawkeye()
    raw["pass_success"] = [.4, .35]
    monkeypatch.setattr(hawkeye, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    if not missing_model:
        raw["model_pass_success"] = [.8, .7]
    if mode != "standard":
        raw["selection_row_id"], raw["original_action_id"] = 1, "s"
        raw["PositionX"], raw["PositionY"] = 200., 300.
        raw["resolved_time_norm"], raw["loc_situation_id"] = 0., "cache_identity"
        monkeypatch.setattr(hawkeye, "get_hawkeye_loc_component_run_root", lambda identifier: Path(identifier))
        monkeypatch.setattr(hawkeye, "apply_hawkeye_possessor_offset", lambda tracking, **kw: (tracking, {}))
        monkeypatch.setattr(hawkeye, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    metadata = component(physical_cache_dir=str(cache), inference_mode=mode, status="completed", freeze_ballreceipt=True)
    monkeypatch.setattr(hawkeye, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", lambda *a: (raw, metadata))
    monkeypatch.setattr(hawkeye, "load_runtime_physical_xpass_visualization_table", lambda *a, **kw:
        pd.DataFrame({"home_1": [.2], "home_2": [.3]}, index=[10]))
    images = _capture_images(monkeypatch, hawkeye, "render_frame_image")
    args = hawkeye.parse_args(["--component-run-id", "a", "--mode", mode, "--only-compare-pass-success",
                              "--output-dir", str(tmp_path), "--run-id", "result"])
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    if missing_model:
        with pytest.raises(ValueError, match="Regenerate"):
            hawkeye.main()
    else:
        hawkeye.main()
        _check_output(tmp_path / "result", images)


def test_benchmark_stacks_existing_paired_state_images(tmp_path, monkeypatch):
    from datatools.benchmark import build_benchmark_state
    cache = write_cache(tmp_path / "cache")
    source = pd.DataFrame(dict(team=[0, 1, 1, 2], player=[0, 1, 2, 3], pos_x=[0., 0., 10., 20.],
        pos_y=[0., 0., 5., 10.], pos_z=[0., np.nan, np.nan, np.nan], smooth_x_speed=[0.] * 4,
        smooth_y_speed=[0.] * 4, event_player=[1] * 4, playing_direction_event=[True] * 4))
    frames = []
    for state_id in (1, 2):
        _, rows, _ = build_benchmark_state(source, 1, state_id, 2, build_graphs=False)
        rows["model_pass_success"], rows["pass_success"] = .8, .4
        frames.append(rows)
    monkeypatch.setattr(benchmark, "get_benchmark_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(benchmark, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(benchmark, "load_benchmark_component_run", lambda *a: (
        pd.concat(frames), component(physical_cache_dir=str(cache))))
    monkeypatch.setattr(benchmark, "load_benchmark_modification_data", lambda *a: {
        "game_state_1": source, "game_state_2": source, "higher_state_id": 2})
    monkeypatch.setattr(benchmark, "load_runtime_physical_xpass_visualization_table", lambda *a, **kw:
        pd.DataFrame({"home_1": [.2], "home_2": [.3]}, index=[0]))
    images = _capture_images(monkeypatch, benchmark, "render_state_image")
    args = benchmark.parse_args(["--component-run-id", "a", "--only-compare-pass-success",
                                "--output-dir", str(tmp_path), "--run-id", "result"])
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    _check_output(tmp_path / "result", images, paired=True)


@pytest.mark.parametrize("frame_flag,count", [([], 2), (["--only-first"], 1), (["--only-last"], 1)])
def test_skillcorner_preserves_frame_selection(tmp_path, monkeypatch, frame_flag, count):
    subject = _subject()
    subject.event_index = 1
    cache = write_cache(tmp_path / "cache")
    table = pd.DataFrame({"index": [1, 1], "frame": [10, 11], "1": [.8, .8], "2": [.7, .7]})
    monkeypatch.setattr(skillcorner, "get_skillcorner_component_run_root", lambda identifier: Path(identifier))
    monkeypatch.setattr(skillcorner, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(skillcorner, "load_run_metadata", lambda *a, **kw: component(physical_cache_dir=str(cache)))
    monkeypatch.setattr(skillcorner, "load_skillcorner_component_tables", lambda *a: {
        "model_pass_success": table, "pass_success": table.assign(**{"1": .4, "2": .35})})
    monkeypatch.setattr(skillcorner, "build_skillcorner_match_context", lambda *a: object())
    monkeypatch.setattr(skillcorner, "build_skillcorner_possession", lambda *a: (subject, {}))
    monkeypatch.setattr(skillcorner.match_cache, "selection_for", lambda *a: {})
    monkeypatch.setattr(skillcorner, "load_runtime_physical_xpass_visualization_table", lambda *a, **kw:
        pd.DataFrame({"home_1": [.2, .2], "home_2": [.3, .3]}, index=[10, 11]))
    images = _capture_images(monkeypatch, skillcorner, "render_frame_image")
    args = skillcorner.parse_args(["--match-id", "s", "--index", "1", "--component-run-id", "a",
        "--only-compare-pass-success", "--output-dir", str(tmp_path), "--run-id", "result", *frame_flag])
    monkeypatch.setattr(skillcorner, "parse_args", lambda: args)
    skillcorner.main()
    _check_output(tmp_path / "result", images, count)
