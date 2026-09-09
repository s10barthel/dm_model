import argparse
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from datatools.viz_helpers import compute_outcome
from datatools.viz_snapshot import SnapshotVisualizer
from scripts import visualize_hawkeye, visualize_benchmark, visualize_skillcorner
from scripts import visualize_action_components as sportec
from scripts import run_and_visualize_hawkeye as direct
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def selection(*flags):
    parser = argparse.ArgumentParser()
    add_component_selection_args(parser)
    return resolve_component_selection(parser.parse_args(flags))


def test_default_and_exclusion_selection():
    default = selection().rendered_components
    assert default[-2:] == ["outcome_failure", "outcome_success"]
    assert selection("--no-outcome-failure", "--no-outcome-success").rendered_components == default[:-2]
    assert "outcome_success" in selection("--no-outcome-failure").rendered_components
    assert "outcome_failure" in selection("--no-outcome-success").rendered_components
    assert selection("--only-pass-success").rendered_components == ["pass_success"]
    assert selection("--only-outcome-failure", "--only-outcome-success", "--no-outcome-failure").rendered_components == ["outcome_success"]
    for case in ("failure", "success"):
        with pytest.raises(SystemExit):
            selection(f"--show-outcome-{case}")


@pytest.mark.parametrize("component", ["outcome_failure", "outcome_success", "pass_score"])
def test_dependency_selection(component):
    selected = selection(f"--only-{component.replace('_', '-')}", "--no-outcome-scoring", "--no-outcome-conceding")
    assert selected.rendered_components == [component]
    assert {"outcome_scoring", "outcome_conceding"} <= set(selected.required_component_groups)
    assert ("pass_success" in selected.required_component_groups) == (component == "pass_score")


@pytest.mark.parametrize("as_frame", [False, True])
def test_subtraction_aligns_labels_and_preserves_missing(as_frame):
    scoring = pd.Series({"positive": 0.8, "negative": 0.1, "zero": 0.3, "missing": np.nan, "left": 0.5})
    conceding = pd.Series({"right": 0.2, "zero": 0.3, "negative": 0.6, "positive": 0.2, "missing": 0.2})
    if as_frame:
        scoring = pd.DataFrame([scoring, scoring], index=[10, 20])
        conceding = pd.DataFrame([conceding, conceding], index=[30, 10])
    result = compute_outcome(scoring, conceding)
    row = result.loc[10] if as_frame else result
    assert row["positive"] == pytest.approx(0.6)
    assert row["negative"] == pytest.approx(-0.5)
    assert row["zero"] == 0
    assert row[["missing", "left", "right"]].isna().all()
    if as_frame:
        assert result.loc[[20, 30]].isna().all().all()


def tables():
    return {name: pd.DataFrame({"home_1": [value]}, index=[10]) for name, value in {
        "outcome_scoring_failure": 0.1, "outcome_conceding_failure": 0.4,
        "outcome_scoring_success": 0.7, "outcome_conceding_success": 0.2,
        "pass_success": 0.25,
    }.items()}


@pytest.mark.parametrize("module", [visualize_hawkeye, visualize_benchmark, visualize_skillcorner])
@pytest.mark.parametrize("component,expected", [("outcome_failure", -0.3), ("outcome_success", 0.5), ("pass_score", -0.1)])
def test_saved_component_paths(module, component, expected):
    source = tables()
    result = module._probs_for_component_frame(component, source, 10)
    assert result.iloc[0] == pytest.approx(expected)
    assert module._probs_for_component_frame(component, source, 99).isna().all()
    del source["outcome_conceding_" + ("failure" if component == "outcome_failure" else "success")]
    with pytest.raises(ValueError, match="missing required source components: outcome_conceding"):
        module._probs_for_component_frame(component, source, 10)


@pytest.mark.parametrize("component,expected", [("outcome_failure", -0.3), ("outcome_success", 0.5), ("pass_score", -0.1)])
def test_sportec_derives_without_rendering_sources(tmp_path, component, expected):
    source = tables()
    def infer(match, model, **kwargs):
        if model == "pass_success":
            return source[model], None
        return source[f"{model}_failure"], source[f"{model}_success"]
    models = {name: name for name in ["pass_success", "outcome_scoring", "outcome_conceding"]}
    with patch.object(sportec, "inference_gnn", side_effect=infer), patch.object(sportec, "render_component") as render:
        sportec.render_action_components(object(), models, tmp_path, "cpu", 10, "10", tmp_path, rendered_components=[component])
    render.assert_called_once()
    assert render.call_args.kwargs["probs"].iloc[0] == pytest.approx(expected)
    assert render.call_args.kwargs["output_path"].name == f"{component}.png"


@pytest.mark.parametrize("output", ["png", "gif", "mp4"])
def test_direct_hawkeye_derives_outputs(tmp_path, output):
    source = tables()
    situation = SimpleNamespace(labels=SimpleNamespace(numel=lambda: 1), graph_features_0=[1], frame_meta=pd.DataFrame(index=[10]), match_id="m")
    args = SimpleNamespace(output=output, freeze_ballreceipt=True, show_trajectories=False, tracking_csv="tracking.csv")
    models = {name: name for name in ["pass_success", "outcome_scoring", "outcome_conceding"]}
    def infer(situation, model, **kwargs):
        if model == "pass_success":
            return source[model], None
        return source[f"{model}_failure"], source[f"{model}_success"]
    seen = {}
    def render(situation, frame_id, component, probs, **kwargs):
        seen[component] = probs.iloc[0]
        return Image.new("RGB", (2, 2))
    def save(images, path, **kwargs):
        list(images)
    with patch.object(direct, "build_hawkeye_situation", return_value=(situation, None, None)), patch.object(direct, "inference_gnn", side_effect=infer), patch.object(direct, "resolve_ballreceipt", return_value=0), patch.object(direct, "resolve_hawkeye_png_frames", return_value=[{"frame_id": 10, "label": "0"}]), patch.object(direct, "render_frame_image", side_effect=render), patch.object(direct, "save_animation", side_effect=save):
        _, _, info = direct.render_situation("s", pd.DataFrame({"id": ["s"]}), pd.DataFrame(), models, {"add_v_edge_features": False}, args, "cpu", tmp_path, ["outcome_failure", "outcome_success", "pass_score"])
    assert seen == pytest.approx({"outcome_failure": -0.3, "outcome_success": 0.5, "pass_score": -0.1})
    assert len(info["output_paths"]) == 3
    assert all(path.endswith("." + output) for path in info["output_paths"])


@pytest.mark.parametrize("component", ["outcome_failure", "outcome_success"])
def test_numeric_annotations_include_negative_and_zero(component):
    snapshot = pd.DataFrame({"home_1_x": [10], "home_1_y": [10], "home_2_x": [20], "home_2_y": [20]})
    visualizer = SnapshotVisualizer(snapshot, player_annots=pd.Series({"home_1": -0.125, "home_2": 0}), show_velocities=False, style="pitchcontrol", attacking_team_prefix="home")
    fig, ax = visualizer.plot(annot_type=component, show=False)
    try:
        assert {"-0.125", "0.000"} <= {text.get_text() for text in ax.texts}
        assert visualizer.colors is None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("module", [sportec, direct])
@pytest.mark.parametrize("component,weight", [("outcome_failure", "v3"), ("outcome_success", "v3"), ("pass_score", "v3"), ("pass_score", "v5")])
def test_inference_commands_load_hidden_dependencies(tmp_path, module, component, weight):
    flags = [f"--only-{component.replace('_', '-')}", "--no-outcome-scoring", "--no-outcome-conceding", "--output-dir", str(tmp_path)]
    if module is sportec:
        flags += ["--match-id", "m", "--action-id", "10"]
    args = module.parse_args(flags)
    args.xpass_weight = weight
    expected = {"outcome_scoring", "outcome_conceding"}
    if component == "pass_score":
        expected.add("pass_success")
        if weight == "v5":
            expected.add("pass_intent")
    resolved = {}
    loaded = []
    def resolve(required_tasks, **kwargs):
        assert set(required_tasks) == expected
        resolved.update({name: name for name in required_tasks})
        return resolved, {}, None
    class StopAfterLoading(Exception):
        pass
    def load(name, device):
        loaded.append(name)
        if len(loaded) == len(expected):
            raise StopAfterLoading
        return SimpleNamespace()
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "parse_args", return_value=args))
        stack.enter_context(patch.object(module, "resolve_model_selection", side_effect=resolve))
        stack.enter_context(patch.object(module, "load_model", side_effect=load))
        if module is direct:
            for name in ("load_hawkeye_tracking", "clean_hawkeye_tracking", "load_hawkeye_ball", "clean_hawkeye_ball"):
                stack.enter_context(patch.object(module, name, return_value=pd.DataFrame({"id": ["s"]})))
        with pytest.raises(StopAfterLoading):
            module.main()
    assert set(loaded) == expected
