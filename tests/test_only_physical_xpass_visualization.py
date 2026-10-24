"""Physical-only exports use scene data and caches without learned predictions."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import project_config

from scripts import visualize_action_components as sportec
from scripts import run_and_visualize_hawkeye as direct
from scripts import visualize_hawkeye as hawkeye
from scripts import visualize_benchmark as benchmark
from scripts import visualize_skillcorner as skillcorner
from scripts.visualization_selection import resolve_component_selection
from scripts.visualization_pass_success import needs_physical_xpass
from test_saved_visualization_xpass import write_cache, component
from test_visualization_comparison import _subject, _raw_hawkeye, inference_fixture, saved_fixture


COMMANDS = [(sportec, ["--match-id", "s"]), (direct, []), (hawkeye, []),
            (benchmark, []), (skillcorner, ["--match-id", "s", "--index", "1"])]
VALUES = pd.Series({"home_1": .2, "home_2": .3})


def forbidden(*args, **kwargs):
    pytest.fail("Physical-only rendering accessed model or implicit component dependencies")


@pytest.mark.parametrize("module,required", COMMANDS)
def test_selection_is_additive_and_has_no_model_dependencies(module, required):
    args = module.parse_args([*required, "--only-physical-xpass"])
    selection = resolve_component_selection(args)
    assert selection.rendered_components == ["physical_xpass"]
    assert selection.requested_component_groups == ["physical_xpass"]
    assert selection.required_component_groups == []
    assert needs_physical_xpass(args) and not args.show_physical_xpass
    duplicate = resolve_component_selection(module.parse_args([
        *required, "--only-physical-xpass", "--show-physical-xpass"]))
    assert duplicate.rendered_components == ["physical_xpass"]
    mixed = resolve_component_selection(module.parse_args([
        *required, "--only-physical-xpass", "--only-pass-score", "--no-pass-success"]))
    assert mixed.rendered_components == ["pass_score", "physical_xpass"]
    assert set(mixed.required_component_groups) == {"pass_success", "outcome_scoring", "outcome_conceding", "pass_score"}
    model_view = resolve_component_selection(module.parse_args([
        *required, "--only-physical-xpass", "--show-model-pass-success"]))
    assert model_view.rendered_components == ["model_pass_success", "physical_xpass"]
    assert model_view.required_component_groups == ["pass_success"]
    composite = resolve_component_selection(module.parse_args([
        *required, "--only-physical-xpass", "--only-compare-pass-success"]))
    assert composite.rendered_components == ["compare_pass_success", "physical_xpass"]
    assert composite.required_component_groups == ["pass_success"]


@pytest.mark.parametrize("module,required", COMMANDS[:2])
@pytest.mark.parametrize("options", [["--no-physical-cache"], ["--inference-mask-1", "accel"],
    ["--compare", "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b"]])
def test_physical_only_rejects_disabled_cache_and_model_comparisons(module, required, options):
    with pytest.raises(SystemExit):
        module.parse_args([*required, "--only-physical-xpass", *options])


def test_tracking_only_conflict():
    with pytest.raises(SystemExit):
        hawkeye.parse_args(["--only-tracking", "--only-physical-xpass"])


def assert_outputs(root, count):
    paths = sorted(root.rglob("*.png"))
    assert len(paths) == count
    assert all(path.name.startswith("physical_xpass") for path in paths)
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["only_physical_xpass"]
    assert metadata["rendered_components"] == ["physical_xpass"]
    if "requested_component_groups" in metadata:
        assert metadata["requested_component_groups"] == ["physical_xpass"]
    assert set(metadata["physical_xpass_output_paths"]) == {str(path.resolve()) for path in paths}
    return paths, metadata


@pytest.mark.parametrize("module,required,pc_cache,explicit_feature", [
    (sportec, ["--match-id", "s"], False, False), (sportec, ["--match-id", "s"], True, False),
    (sportec, ["--match-id", "s"], "directory", False),
    (sportec, ["--match-id", "s"], True, True), (direct, [], False, False), (direct, [], True, False)])
def test_inference_commands_skip_all_model_dependencies(tmp_path, monkeypatch, inference_fixture, module, required, pc_cache, explicit_feature):
    subject, _, calls, build = inference_fixture
    cache = write_cache(tmp_path / "cache", source_inputs={"feature_run_id": "cache_feature"})
    if not pc_cache:
        (cache / "metadata.json").unlink()
    for name in ("resolve_model_selection", "load_model", "configure_auxiliary_models", "inference_gnn"):
        monkeypatch.setattr(module, name, forbidden)
    loader = Mock(side_effect=lambda *a, **kw: VALUES.copy() if module is sportec else
                  pd.DataFrame([VALUES for _ in a[2]], index=a[2]))
    if module is sportec:
        subject.actions = pd.DataFrame(dict(frame_id=[10], object_id=["home_1"],
            spadl_type=["pass"], action_type=["pass"], end_x=[40.], end_y=[25.]), index=[10])
        runtime = Mock(return_value={"feature_root": tmp_path, "feature_run_id": "feature",
                                    "intended_receiver_mode": "predict", "selection": "explicit"})
        monkeypatch.setattr(module, "resolve_runtime_feature_run_context", runtime)
        monkeypatch.setattr(module, "load_match", lambda *a, **kw: subject)
        monkeypatch.setattr(module, "resolve_action_indices", lambda *a: [(10, "42")])
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_component", loader)
        if pc_cache:
            state = Mock(return_value=(subject, 10))
            monkeypatch.setattr(module, "possession_state_for_action", state)
            monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
    else:
        monkeypatch.setattr(module, "collect_situation_components", forbidden)
        monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", loader)
    flags = [*required, "--only-physical-xpass", "--show-physical-xpass",
        "--physical-cache-dir", str(cache), "--output-dir", str(tmp_path), "--run-id", "result"]
    if pc_cache is True:
        flags += ["--pc-xpass"]
    if explicit_feature:
        flags += ["--feature-run-id", "explicit_feature"]
    args = module.parse_args(flags)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    module.main()
    paths, metadata = assert_outputs(tmp_path / "result", 1)
    assert metadata["model_ids"] == metadata["selected_model_ids"] == {}
    assert metadata["physical_xpass_runtime_source"] and metadata["x_pass_version"] == "top10"
    assert loader.call_count == 1 and not calls
    if module is sportec:
        assert runtime.call_args.args[0] == ("explicit_feature" if explicit_feature else "cache_feature" if pc_cache else None)
        if pc_cache:
            assert state.call_count == 1 and "cache_selection" in loader.call_args.kwargs
            assert metadata["physical_xpass_runtime_source"] == "pc_xpass"
            if not explicit_feature:
                assert metadata["runtime_feature_run_selection"] == "physical_xpass_cache"
    if module is direct:
        assert build.call_args.kwargs["build_graphs"] is False
        expected = module.render_frame_image(subject, 10, "physical_xpass", VALUES)
    else:
        expected_path = tmp_path / "expected.png"
        module.render_component(subject, 10, "42", "physical_xpass", VALUES, expected_path)
        expected = Image.open(expected_path).convert("RGB")
    with expected, Image.open(paths[0]) as result:
        np.testing.assert_array_equal(result.convert("RGB"), expected)


def setup_saved(module, tmp_path, monkeypatch, saved_fixture):
    subject, _ = saved_fixture
    subject.event_index = 1
    for name in ("resolve_named_component_run_id", "load_hawkeye_component_run" if module is hawkeye else
                 "load_benchmark_component_run" if module is benchmark else "load_skillcorner_component_tables"):
        monkeypatch.setattr(module, name, forbidden)
    if module is hawkeye:
        monkeypatch.setattr(module, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    elif module is skillcorner:
        monkeypatch.setattr(module, "load_run_metadata", forbidden)
        monkeypatch.setattr(module, "build_skillcorner_match_context", lambda *a: object())
        monkeypatch.setattr(module, "build_skillcorner_possession", lambda *a: (subject, {}))
        monkeypatch.setattr(module.match_cache, "selection_for", lambda *a: {})
    else:
        source = pd.DataFrame(dict(team=[0, 1, 1, 2], player=[0, 1, 2, 3], pos_x=[0., 0., 10., 20.],
            pos_y=[0., 0., 5., 10.], pos_z=[0., np.nan, np.nan, np.nan], smooth_x_speed=[0.] * 4,
            smooth_y_speed=[0.] * 4, event_player=[1] * 4, playing_direction_event=[True] * 4))
        directory = tmp_path / "input" / "modification_1"
        directory.mkdir(parents=True)
        source.to_csv(directory / "game_state_1.csv", index=False)
        source.to_csv(directory / "game_state_2.csv", index=False)
        pd.DataFrame({"higher_state_id": [2]}).to_csv(directory / "modification.csv", index=False)
    return subject


@pytest.mark.parametrize("module,required", COMMANDS[2:])
@pytest.mark.parametrize("by_id", [False, True])
def test_saved_commands_render_without_any_component_run(tmp_path, monkeypatch, saved_fixture, module, required, by_id):
    subject = setup_saved(module, tmp_path, monkeypatch, saved_fixture)
    cache = write_cache(tmp_path / "cache")
    if by_id:
        versions = tmp_path / "versions"
        write_cache(versions / "chosen")
        cache = write_cache(versions / "chosen" / module.__name__.split("visualize_")[-1])
        (versions / "latest.json").write_text('{"run_id":"missing"}')
        monkeypatch.setattr(project_config, "PC_XPASS_DIR", versions)
    loader = Mock(side_effect=lambda *a, **kw: pd.DataFrame([VALUES for _ in a[2]], index=a[2]))
    monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", loader)
    flags = [*required, "--only-physical-xpass", "--show-physical-xpass",
             "--xpass-version", "top25", "--output-dir", str(tmp_path), "--run-id", "result"]
    flags += ["--pc-xpass-id", "chosen"] if by_id else ["--physical-cache-dir", str(cache)]
    if module is benchmark:
        flags += ["--input-dir", str(tmp_path / "input"), "--modification", "1", "--modification", "2"]
    args = module.parse_args(flags)
    monkeypatch.setattr(module, "parse_args", lambda: args)
    module.main()
    paths, metadata = assert_outputs(tmp_path / "result", 2 if module is skillcorner else 1)
    assert metadata["component_run_id"] is metadata["component_dir"] is None
    assert metadata["source_models"] == {}
    assert metadata["physical_xpass_runtime_source"] == "pc_xpass"
    assert metadata["physical_xpass_metric"] == "top25_xpass"
    assert all(call.kwargs["metric"] == "top25_xpass" for call in loader.call_args_list)
    if module is benchmark:
        assert loader.call_count == 2
        assert metadata["skipped_modifications"] == [{"modification": 2, "reason": "missing_modification_dir"}]
        data = module.load_benchmark_modification_data(1, tmp_path / "input")
        images = []
        for state_id in (1, 2):
            state, _, _ = module.build_benchmark_state(data[f"game_state_{state_id}"], 1, state_id, 2, build_graphs=False)
            images.append(module.render_state_image(state, "physical_xpass", VALUES))
        with module.combine_state_images(*images) as expected, Image.open(paths[0]) as result:
            np.testing.assert_array_equal(result.convert("RGB"), expected)
        for image in images:
            image.close()
    else:
        for path in paths:
            frame = 11 if "last" in path.stem else 10
            with module.render_frame_image(subject, frame, "physical_xpass", VALUES) as expected, Image.open(path) as result:
                np.testing.assert_array_equal(result.convert("RGB"), expected)


@pytest.mark.parametrize("frame_flag,expected_frames", [([], [10, 11]), (["--only-first"], [10]), (["--only-last"], [11])])
def test_skillcorner_reads_selected_frames_without_pass_success(tmp_path, monkeypatch, saved_fixture, frame_flag, expected_frames):
    setup_saved(skillcorner, tmp_path, monkeypatch, saved_fixture)
    cache = write_cache(tmp_path / "cache")
    loader = Mock(side_effect=lambda *a, **kw: pd.DataFrame([VALUES for _ in a[2]], index=a[2]))
    monkeypatch.setattr(skillcorner, "load_runtime_physical_xpass_visualization_table", loader)
    args = skillcorner.parse_args(["--match-id", "s", "--index", "1", "--only-physical-xpass",
        "--physical-cache-dir", str(cache), "--top-pass", "10", "--output-dir", str(tmp_path), "--run-id", "result", *frame_flag])
    monkeypatch.setattr(skillcorner, "parse_args", lambda: args)
    skillcorner.main()
    assert loader.call_args.args[2] == expected_frames
    assert loader.call_args.kwargs["metric"] == "top_pass10_xpass"
    assert_outputs(tmp_path / "result", len(expected_frames))


@pytest.mark.parametrize("mode", ["standard", "loc", "freeze"])
@pytest.mark.parametrize("override", [False, True])
def test_explicit_hawkeye_run_preserves_geometry_and_cache(tmp_path, monkeypatch, saved_fixture, mode, override):
    cache = write_cache(tmp_path / "cache")
    other = write_cache(tmp_path / "other")
    raw = _raw_hawkeye()
    if mode != "standard":
        raw["selection_row_id"], raw["original_action_id"] = 1, "s"
        raw["PositionX"], raw["PositionY"] = 200., 300.
        raw["resolved_time_norm"], raw["loc_situation_id"] = 0., "cache_identity"
        monkeypatch.setattr(hawkeye, "get_hawkeye_loc_component_run_root", lambda identifier: Path(identifier))
        monkeypatch.setattr(hawkeye, "apply_hawkeye_possessor_offset", lambda tracking, **kw: (tracking, {}))
    metadata = component(physical_cache_dir=str(cache), inference_mode=mode, freeze_ballreceipt=True)
    monkeypatch.setattr(hawkeye, "load_overlay_data", lambda **kw: SimpleNamespace(metadata={}))
    monkeypatch.setattr(hawkeye, "resolve_named_component_run_id", lambda kind, identifier, **kw: identifier)
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", lambda *a: (raw, metadata))
    loader = Mock(return_value=pd.DataFrame([VALUES], index=[10]))
    monkeypatch.setattr(hawkeye, "load_runtime_physical_xpass_visualization_table", loader)
    flags = ["--component-run-id", "a", "--mode", mode, "--only-physical-xpass",
             "--output-dir", str(tmp_path), "--run-id", "result"]
    if override:
        flags += ["--physical-cache-dir", str(other), "--xpass-version", "max"]
    args = hawkeye.parse_args(flags)
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    hawkeye.main()
    _, result = assert_outputs(tmp_path / "result", 1)
    assert result["component_run_id"] == "a"
    assert loader.call_args.args[:2] == (other if override else cache, "s" if mode == "standard" else "cache_identity")
    assert loader.call_args.kwargs["metric"] == ("max_xpass" if override else "top25_xpass")


@pytest.mark.parametrize("module,required", COMMANDS[2:])
def test_independent_saved_mode_requires_explicit_cache_and_metric(tmp_path, monkeypatch, saved_fixture, module, required):
    setup_saved(module, tmp_path, monkeypatch, saved_fixture)
    cache = write_cache(tmp_path / "cache")
    parse = module.parse_args
    for options in ([], ["--physical-cache-dir", str(cache)], ["--xpass-version", "top25"]):
        flags = [*required, "--only-physical-xpass", "--output-dir", str(tmp_path), *options]
        if module is benchmark:
            flags += ["--input-dir", str(tmp_path / "input")]
        args = parse(flags)
        monkeypatch.setattr(module, "parse_args", lambda: args)
        with pytest.raises(ValueError, match="explicit"):
            module.main()


@pytest.mark.parametrize("module,required", [(hawkeye, []), (skillcorner, ["--match-id", "s", "--index", "1"]), (direct, [])])
@pytest.mark.parametrize("output", ["gif", "mp4"])
def test_animation_exports_only_physical_frames(tmp_path, monkeypatch, saved_fixture, inference_fixture, module, required, output):
    if module is direct:
        for name in ("resolve_model_selection", "load_model", "configure_auxiliary_models", "collect_situation_components", "inference_gnn"):
            monkeypatch.setattr(module, name, forbidden)
    else:
        setup_saved(module, tmp_path, monkeypatch, saved_fixture)
    cache = write_cache(tmp_path / "cache")
    requested = []
    def physical(cache_dir, match_id, frames, **kwargs):
        requested.extend(frames)
        return pd.DataFrame([VALUES for _ in frames], index=frames)
    monkeypatch.setattr(module, "load_runtime_physical_xpass_visualization_table", physical)
    frames = []
    def save(images, path, **kwargs):
        for image in images:
            frames.append(np.array(image))
            image.close()
        path.write_bytes(b"encoded")
    encoder = Mock(side_effect=save)
    monkeypatch.setattr(module, "save_animation", encoder)
    args = module.parse_args([*required, "--only-physical-xpass", "--output", output,
        "--physical-cache-dir", str(cache), "--xpass-version", "top25", "--output-dir", str(tmp_path), "--run-id", "result"])
    monkeypatch.setattr(module, "parse_args", lambda: args)
    module.main()
    assert requested == [10, 11] and len(frames) == 2 and encoder.call_count == 1
    path = encoder.call_args.args[1]
    assert path.name == f"physical_xpass.{output}" and path.exists()
    assert encoder.call_args.kwargs["gif"] == (output == "gif")
    metadata = json.loads((tmp_path / "result/metadata.json").read_text())
    assert metadata["physical_xpass_output_paths"] == [str(path.resolve())]
    assert not list((tmp_path / "result").rglob("*.png"))


def test_saved_comparison_supports_physical_only(tmp_path, monkeypatch, saved_fixture):
    caches = {name: write_cache(tmp_path / name) for name in ("a", "b")}
    def load(path):
        return _raw_hawkeye(), component(status="completed", processed_situation_ids=["s"],
                                        physical_cache_dir=str(caches[str(path)]))
    monkeypatch.setattr(hawkeye, "load_hawkeye_component_run", load)
    def physical(cache, match_id, frames, **kwargs):
        value = .2 if cache == caches["a"] else .3
        return pd.DataFrame({"home_2": [value] * len(frames)}, index=frames)
    monkeypatch.setattr(hawkeye, "load_runtime_physical_xpass_visualization_table", physical)
    args = hawkeye.parse_args(["--only-physical-xpass", "--compare", "--component-run-id", "a",
        "--component-run-id", "b", "--accept-comparison-warnings", "--output-dir", str(tmp_path), "--run-id", "result"])
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    hawkeye.main()
    paths, metadata = assert_outputs(tmp_path / "result", 1)
    values = pd.read_csv(tmp_path / "result/aligned_predictions.csv")
    assert set(values.component) == {"physical_xpass"}
    assert values["prediction:a"].iloc[0] == pytest.approx(.2)
    assert values["prediction:b"].iloc[0] == pytest.approx(.3)
    assert len(metadata["variants"]) == 2


@pytest.mark.parametrize("comparison", ["checkpoints", "masks"])
def test_inference_comparisons_allow_physical_with_model_output(tmp_path, monkeypatch, inference_fixture, comparison):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(direct, "load_runtime_physical_xpass_visualization_table", lambda *a, **kw:
        pd.DataFrame([VALUES], index=[10]))
    flags = ["--only-physical-xpass", "--only-pass-success", "--pass-success-model-id", "pass_success/a",
             "--physical-cache-dir", str(cache), "--output-dir", str(tmp_path), "--run-id", "result"]
    flags += ["--compare", "--pass-success-model-id", "pass_success/b"] if comparison == "checkpoints" else ["--inference-mask-1", "accel"]
    args = direct.parse_args(flags)
    monkeypatch.setattr(direct, "parse_args", lambda: args)
    direct.main()
    values = pd.read_csv(tmp_path / "result/aligned_predictions.csv")
    assert set(values.component) == {"physical_xpass", "pass_success"}
    column = "prediction:pass_success/a" if comparison == "checkpoints" else "prediction:unmasked"
    assert sorted(values.loc[values.component.eq("physical_xpass"), column]) == pytest.approx([.2, .3])
    assert len(list((tmp_path / "result").rglob("physical_xpass*.png"))) == 1


@pytest.mark.parametrize("failure", [None, "missing_directory", "missing_metric", "missing_row"])
def test_hawkeye_reads_real_cache_without_generating_data(tmp_path, monkeypatch, saved_fixture, failure):
    setup_saved(hawkeye, tmp_path, monkeypatch, saved_fixture)
    cache = write_cache(tmp_path / "cache", cache_format=None,
                        available_metrics=["max_xpass"] if failure == "missing_metric" else ["top25_xpass"])
    (cache / "matches").mkdir()
    frame = 11 if failure == "missing_row" else 10
    pd.DataFrame([{"match_id": "s", "action_index": frame, "home_1__top25_xpass": .2,
                   "home_2__top25_xpass": .3}]).to_parquet(cache / "matches/s.parquet", index=False)
    selected_cache = tmp_path / "missing" if failure == "missing_directory" else cache
    args = hawkeye.parse_args(["--only-physical-xpass", "--physical-cache-dir", str(selected_cache),
        "--xpass-version", "top25", "--output-dir", str(tmp_path), "--run-id", "result"])
    monkeypatch.setattr(hawkeye, "parse_args", lambda: args)
    if failure:
        with pytest.raises((FileNotFoundError, ValueError, KeyError)):
            hawkeye.main()
        assert not list((tmp_path / "result").rglob("*.png"))
        if failure == "missing_directory":
            assert not selected_cache.exists()
    else:
        hawkeye.main()
        assert_outputs(tmp_path / "result", 1)
