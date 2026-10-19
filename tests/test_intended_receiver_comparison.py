import json
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from datatools import config
from datatools.viz_snapshot import SnapshotVisualizer
from scripts import visualize_action_components as viz
from scripts import visualization_receivers as receivers
from scripts import inference_visualization_comparison as comparisons
from scripts.visualization_selection import resolve_component_selection
from test_visualization_comparison import inference_fixture


@pytest.fixture
def receiver_match():
    tracking = {"ball_x": [10., 35., 58.], "ball_y": [20., 30., 35.], "ball_z": [0.] * 3}
    for player, (x, y) in {
        "home_1": (10., 20.), "home_2": (35., 32.), "home_3": (55., 35.),
        "home_4": (40., 50.), "away_5": (60., 30.),
        "home_goal": (config.FIELD_SIZE[0], config.FIELD_SIZE[1] / 2),
        "away_goal": (0., config.FIELD_SIZE[1] / 2),
    }.items():
        for axis, value in (("x", x), ("y", y), ("vx", 0.), ("vy", 0.)):
            tracking[f"{player}_{axis}"] = [value] * 3
    actions = pd.DataFrame([dict(
        action_id=123, action_type="pass", spadl_type="pass", frame_id=10, receive_frame_id=60,
        object_id="home_1", receiver_id="away_5", intent_id="home_4", success=False,
        start_x=10., start_y=20., end_x=58., end_y=35.,
    )], index=[7])
    return SimpleNamespace(match_id="match", actions=actions, tracking=pd.DataFrame(tracking, index=[10, 20, 60]), fps=25)


def probabilities(best="home_3"):
    result = pd.Series({"home_1": 0., "home_2": .1, "home_3": .1, "home_4": .1, "home_goal": 0., "away_5": 0.})
    result[best] = .8
    return result


@pytest.mark.parametrize("flag", ["--success-intent", "--compare-intended-receiver"])
def test_success_intent_flags_add_to_only_selection(flag):
    args = viz.parse_args(["--match-id", "match", "--only-pass-success", flag])
    selection = resolve_component_selection(args, include_success_intent=True)
    assert selection.rendered_components == ["pass_success", "success_intent"]
    assert args.pass_endpoint is True
    assert viz.parse_args(["--match-id", "match", "--pass-endpoint", "false"]).pass_endpoint is False


def test_success_intent_exclusion_and_invalid_options():
    args = viz.parse_args(["--match-id", "match", "--success-intent", "--no-success-intent"])
    assert "success_intent" not in resolve_component_selection(args, include_success_intent=True).rendered_components
    args = viz.parse_args(["--match-id", "match", "--only-success-intent"])
    assert resolve_component_selection(args, include_success_intent=True).rendered_components == ["success_intent"]
    for flags in (["--only-intended-recipient"], ["--no-intended-recipient"], ["--intended-recipient"],
                  ["--compare-intended-receiver", "--no-success-intent"], ["--pass-endpoint", "invalid"]):
        with pytest.raises(SystemExit):
            viz.parse_args(["--match-id", "match", *flags])


def test_endpoint_pair_fallback_disable_and_nonpass(receiver_match):
    assert receivers.resolve_pass_endpoint(receiver_match, 7) == (58., 35.)
    receiver_match.actions.at[7, "end_x"] = np.inf
    receiver_match.actions.at[7, "end_y"] = 999.
    assert receivers.resolve_pass_endpoint(receiver_match, 7) == (58., 35.)  # Use a complete tracking pair.
    receiver_match.tracking.at[60, "ball_x"] = np.nan
    with pytest.raises(ValueError, match="endpoint"):
        receivers.prepare_action_overlays(receiver_match, 7)
    assert receivers.prepare_action_overlays(receiver_match, 7, pass_endpoint=False).endpoint_xy is None
    receiver_match.actions.at[7, "action_type"] = "dribble"
    assert receivers.prepare_action_overlays(receiver_match, 7, compare_intended_receiver=True) == receivers.ActionOverlays()


def test_live_heuristics_are_cached_without_changing_runtime_labels(receiver_match, monkeypatch):
    before = receiver_match.actions.copy(deep=True)
    label = Mock(wraps=receivers.label_intended_receivers)
    monkeypatch.setattr(receivers, "label_intended_receivers", label)
    first = receivers.resolve_intended_receivers(receiver_match, 7, probabilities("home_2"))
    second = receivers.resolve_intended_receivers(receiver_match, 7, probabilities("home_3"))
    assert first["model"] == "home_2" and second["model"] == "home_3"
    assert first["original"] == second["original"]
    assert first["angle_only"] == second["angle_only"]
    assert label.call_count == 2
    assert all(call.args[0].index.tolist() == [7] for call in label.call_args_list)
    pd.testing.assert_frame_equal(receiver_match.actions, before)


def test_success_uses_actual_receiver_with_model_values_preserved(receiver_match, monkeypatch):
    receiver_match.actions.loc[7, ["success", "receiver_id"]] = [True, "home_2"]
    label = Mock(side_effect=AssertionError("Successful passes do not need heuristics"))
    monkeypatch.setattr(receivers, "label_intended_receivers", label)
    probs = probabilities("home_3")
    before = probs.copy()
    choices = receivers.resolve_intended_receivers(receiver_match, 7, probs)
    assert choices == dict.fromkeys(receivers.INTENDED_RECEIVER_STYLES, "home_2")
    pd.testing.assert_series_equal(probs, before)
    assert not label.called


@pytest.mark.parametrize("missing", ["prediction", "heuristic", "position", "actual_receiver"])
def test_missing_comparison_inputs_are_action_errors(receiver_match, monkeypatch, missing):
    probs = probabilities()
    if missing == "prediction":
        probs[:] = np.nan
    elif missing == "heuristic":
        monkeypatch.setattr(receivers, "_heuristic_receivers", lambda *a: {"original": None, "angle_only": "home_3"})
    elif missing == "position":
        monkeypatch.setattr(receivers, "_heuristic_receivers", lambda *a: {"original": "home_2", "angle_only": "home_3"})
        receiver_match.tracking.at[10, "home_2_x"] = np.nan
    else:
        receiver_match.actions.loc[7, ["success", "receiver_id"]] = [True, None]
    with pytest.raises(ValueError):
        receivers.prepare_action_overlays(receiver_match, 7, probs, compare_intended_receiver=True)


@pytest.mark.parametrize("choices", [
    ("home_2", "home_2", "home_2"),
    ("home_2", "home_3", "home_3"),
    ("home_2", "home_2", "home_3"),
    ("home_2", "home_3", "home_2"),
    ("home_2", "home_3", "home_4"),
])
def test_render_keeps_all_rings_and_runtime_probabilities(receiver_match, monkeypatch, tmp_path, choices):
    monkeypatch.setattr(receivers, "_heuristic_receivers", lambda *a: dict(zip(("original", "angle_only"), choices)))
    monkeypatch.setattr(viz, "resolve_ball_velocity_xy", lambda *a: None)
    figures = []
    def capture(fig, **kwargs):
        figures.append(fig)
        return Image.new("RGB", (40, 30))
    monkeypatch.setattr(viz, "figure_to_rgb_image", capture)
    image = viz.render_component(receiver_match, 7, "123", "success_intent", probabilities(choices[2]),
                                 tmp_path / "success_intent.png", comparison=True, compare_intended_receiver=True)
    image.close()
    ax = figures[0].axes[0]
    rings = [line for line in ax.lines if line.get_marker() == "o"]
    assert sorted(line.get_markersize() for line in rings) == [12., 18., 24.]
    for player, (color, diameter) in zip(choices, receivers.INTENDED_RECEIVER_STYLES.values()):
        line = next(line for line in rings if line.get_markeredgecolor() == color)
        assert line.get_markersize() == diameter
        assert line.get_markerfacecolor() == "none"
        assert line.get_markeredgecolor() == color
        assert line.get_markeredgewidth() == 1.5
        assert line.get_xdata()[0] == receiver_match.tracking.at[10, f"{player}_x"]
    assert "0.800" in [text.get_text() for text in ax.texts]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "original", "angle_only", "model", "Actual pass end"]
    # The runtime receiver's old green outline must not act as a fourth mode.
    highlights = viz.resolve_highlight_players(receiver_match.tracking, receiver_match.actions.loc[7], "success_intent", True)
    assert highlights == {"home_1": "#ffd400"}


def test_away_pitch_rotates_ring_and_endpoint_together(receiver_match):
    snapshot = receiver_match.tracking.loc[[10]]
    visualizer = SnapshotVisualizer(
        snapshot, player_annots=pd.Series({"home_2": .8}), show_velocities=False,
        style="pitchcontrol", player_rings={"home_2": [("#9467BD", 24.)]}, pass_endpoint_xy=(58., 35.),
    )
    fig, ax = visualizer.plot(rotate_pitch=True, annot_type="success_intent", show=False)
    try:
        ring = next(line for line in ax.lines if line.get_marker() == "o")
        endpoint = next(line for line in ax.lines if line.get_marker() == "x")
        np.testing.assert_allclose([ring.get_xdata()[0], ring.get_ydata()[0]], np.array(config.FIELD_SIZE) - [35., 32.])
        np.testing.assert_allclose([endpoint.get_xdata()[0], endpoint.get_ydata()[0]], np.array(config.FIELD_SIZE) - [58., 35.])
        text = next(text for text in ax.texts if text.get_text() == "0.800")
        assert text.get_position()[1] >= 18
    finally:
        plt.close(fig)


def test_endpoint_on_other_components_without_rings(receiver_match, monkeypatch, tmp_path):
    monkeypatch.setattr(viz, "resolve_ball_velocity_xy", lambda *a: None)
    observed = []
    original = viz.SnapshotVisualizer
    def visualizer(**kwargs):
        observed.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(viz, "SnapshotVisualizer", visualizer)
    image = viz.render_component(receiver_match, 7, "123", "pass_success", probabilities(), tmp_path / "plot.png",
                                 comparison=True, compare_intended_receiver=True)
    image.close()
    assert observed[0]["pass_endpoint_xy"] == (58., 35.)
    assert observed[0]["player_rings"] == {}


def configure_batch(monkeypatch, tmp_path, match, flags=()):
    args = viz.parse_args([
        "--match-id", "match", "--only-success-intent", "--success-intent-model-id", "success_intent/current",
        "--output-dir", str(tmp_path), "--run-id", "batch", *flags,
    ])
    monkeypatch.setattr(viz, "parse_args", lambda: args)
    monkeypatch.setattr(viz, "load_model", lambda *a: SimpleNamespace(args={"task": "success_intent"}))
    monkeypatch.setattr(viz, "get_model_record", lambda *a: {"task": "success_intent", "feature_run_id": "feature"})
    monkeypatch.setattr(viz, "validate_model_graph_schemas", lambda *a: {"add_v_edge_features": False})
    monkeypatch.setattr(viz, "resolve_runtime_feature_run_context", lambda *a, **kw: dict(
        feature_root=tmp_path, feature_run_id="feature", intended_receiver_mode="original", selection="explicit"))
    monkeypatch.setattr(viz, "resolve_runtime_return_type", lambda *a: "next_5")
    monkeypatch.setattr(viz, "load_match", lambda *a, **kw: match)
    monkeypatch.setattr(viz, "resolve_action_indices", lambda *a: [(int(i), str(i)) for i in match.actions.index])
    monkeypatch.setattr(viz, "collect_action_components", lambda *a, **kw: {"success_intent": probabilities()})
    monkeypatch.setattr(viz, "resolve_ball_velocity_xy", lambda *a: None)
    monkeypatch.setattr(viz, "attach_membership_report", lambda *a, **kw: None)
    return args


@pytest.mark.parametrize("all_missing", [False, True])
def test_batch_skips_missing_action_and_writes_metadata(receiver_match, monkeypatch, tmp_path, all_missing, capsys):
    receiver_match.actions.loc[12] = receiver_match.actions.loc[7]
    receiver_match.actions.loc[7, ["end_x", "end_y", "receive_frame_id"]] = [np.nan, np.nan, 999]
    if all_missing:
        receiver_match.actions.loc[12, ["end_x", "end_y", "receive_frame_id"]] = [np.nan, np.nan, 999]
    configure_batch(monkeypatch, tmp_path, receiver_match)
    if all_missing:
        with pytest.raises(RuntimeError, match="All selected actions"):
            viz.main()
    else:
        viz.main()
    assert "endpoint" in capsys.readouterr().err
    root = tmp_path / "batch"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == ("failed" if all_missing else "completed")
    assert metadata["model_ids"] == {"success_intent": "success_intent/current"}
    assert metadata["rendered_components"] == ["success_intent"]
    assert metadata["pass_endpoint"] is True
    assert len(metadata["skipped_actions"]) == (2 if all_missing else 1)
    assert not (root / "match/7/success_intent.png").exists()
    assert (root / "match/12/success_intent.png").exists() == (not all_missing)


@pytest.mark.parametrize("compare_receivers", [False, True])
def test_missing_checkpoint_is_configuration_error(tmp_path, monkeypatch, compare_receivers):
    flags = ["--only-success-intent"]
    if compare_receivers:
        flags.append("--compare-intended-receiver")
    args = viz.parse_args(["--match-id", "match", "--output-dir", str(tmp_path), *flags])
    monkeypatch.setattr(viz, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="checkpoint"):
        viz.main()


@pytest.mark.parametrize("mode", ["checkpoints", "masks"])
@pytest.mark.parametrize("all_missing", [False, True])
def test_comparison_panels_use_their_current_prediction_and_skip_missing_action(
    tmp_path, monkeypatch, inference_fixture, receiver_match, mode, all_missing,
):
    subject, models, _, _ = inference_fixture
    subject.actions = receiver_match.actions.copy()
    subject.actions.loc[12] = subject.actions.loc[7]
    subject.actions.loc[7, ["end_x", "end_y", "receive_frame_id"]] = [np.nan, np.nan, 999]
    if all_missing:
        subject.actions.loc[12, ["end_x", "end_y", "receive_frame_id"]] = [np.nan, np.nan, 999]
    subject.tracking, subject.fps = receiver_match.tracking, receiver_match.fps
    monkeypatch.setattr(viz, "load_match", lambda *a, **kw: subject)
    monkeypatch.setattr(viz, "resolve_action_indices", lambda *a: [(7, "7"), (12, "12")])
    monkeypatch.setattr(viz, "resolve_runtime_feature_run_context", lambda *a, **kw: dict(
        feature_root=tmp_path, feature_run_id="feature", intended_receiver_mode="original"))
    monkeypatch.setattr(viz, "configure_lane_survival_runtime_cache", lambda *a: None)
    monkeypatch.setattr(viz, "resolve_ball_velocity_xy", lambda *a: None)
    def collect(match, pipeline, *a, **kw):
        model = pipeline["success_intent"]
        best = "home_2" if model.identifier.endswith("/a") and not model.inference_masks else "home_3"
        return {"success_intent": probabilities(best)}
    monkeypatch.setattr(viz, "collect_action_components", collect)
    heuristic = Mock(wraps=receivers.label_intended_receivers)
    monkeypatch.setattr(receivers, "label_intended_receivers", heuristic)
    observed = []
    render = viz.render_component
    def capture(match, index, display, component, probs, *a, **kw):
        observed.append(receivers.resolve_intended_receivers(match, index, probs)["model"])
        return render(match, index, display, component, probs, *a, **kw)
    monkeypatch.setattr(viz, "render_component", capture)
    flags = ["--compare", "--success-intent-model-id", "success_intent/b"] if mode == "checkpoints" else [
        "--inference-mask-1", "accel"]
    args = viz.parse_args([
        "--accept-comparison-warnings", "--match-id", "match", "--only-success-intent",
        "--compare-intended-receiver", "--success-intent-model-id", "success_intent/a",
        "--output-dir", str(tmp_path), "--run-id", "panels", *flags,
    ])
    with pytest.warns(UserWarning, match="endpoint"):
        if all_missing:
            with pytest.raises(RuntimeError, match="All selected actions"):
                comparisons.run_comparison(args, viz)
        else:
            comparisons.run_comparison(args, viz)
    assert observed == ([] if all_missing else ["home_2", "home_3"])
    assert heuristic.call_count == (0 if all_missing else 2)
    metadata = json.loads((tmp_path / "panels/metadata.json").read_text())
    assert metadata["status"] == ("failed" if all_missing else "completed")
    assert metadata["compare_intended_receiver"] is True
    assert metadata["skipped_actions"][0]["action_id"] == "7"
    assert (tmp_path / "panels/match/12/success_intent.png").exists() == (not all_missing)
    assert not (tmp_path / "panels/match/7/success_intent.png").exists()
