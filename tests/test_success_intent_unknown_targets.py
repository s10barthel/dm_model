"""Unknown recipients through real preprocessing and prediction."""
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from datatools import config, utils
from datatools.match import Match
from datatools.trajectory import TRAJECTORY_SPEED
from inference import inference_gnn
from scripts import visualize_action_components as viz
from test_goal_context import DummyModel, settings, source


def prediction_context(nodes=True):
    args = settings("success_intent", nodes=nodes, include=False)
    args["trajectory_speed_definition"] = TRAJECTORY_SPEED
    graph, label = source()
    # A goal first exposes invalid zero placeholders when goal nodes are removed.
    order = torch.tensor([2, 0, 1, 3, 4, 5])
    graph.x = graph.x[order]
    graph.node_ids = [graph.node_ids[i] for i in order.tolist()]
    graph.trajectory_speed_definition = TRAJECTORY_SPEED
    label[0] = 12
    label[5:7] = -1
    label[config.LABEL_INDEX["success"]] = 0
    tracking = pd.DataFrame({f"{pid}_{axis}": [float(graph.x[i, 3+j])]
                             for i, pid in enumerate(graph.node_ids) for j, axis in enumerate(["x", "y"])})
    actions = pd.DataFrame([dict(action_type="pass", success=False, frame_id=0,
                                object_id="home_1", receive_frame_id=12,
                                receiver_id="away_2", intent_id="home_3")], index=[12])
    match = SimpleNamespace(actions=actions, tracking=tracking, labels=label[None].clone())
    return match, graph, label, DummyModel(args)


@pytest.mark.parametrize("nodes", [True, False])
@pytest.mark.parametrize("saved", [True, False])
def test_failed_pass_visualization_real_inference(tmp_path, nodes, saved):
    match, graph, label, model = prediction_context(nodes)
    original_actions, original_labels = match.actions.copy(deep=True), match.labels.clone()
    original_x = graph.x.clone()
    if saved:
        graph_dir = viz.get_success_intent_graph_dir(tmp_path)
        graph_dir.mkdir(parents=True)
        torch.save([graph], graph_dir / "match.pt")
    with (
        patch.object(viz, "load_success_intent_labels", side_effect=None if saved else FileNotFoundError(), return_value=label[None]),
        patch.object(viz, "resolve_match_id", return_value="match"),
        patch.object(viz, "construct_graph_for_action", side_effect=lambda *a, **kw: graph.clone()) as build,
        patch.object(viz, "validate_model_graph_schemas", return_value={"edge_in_dim": 2, "add_v_edge_features": False}),
    ):
        result = viz.run_success_intent_component(match, model, tmp_path, "cpu", 12)
    assert result.name == 12
    assert list(result.index) == ["home_3", "home_4"]
    assert torch.isfinite(torch.tensor(result.values)).all()
    assert result.sum() == pytest.approx(1)
    assert result["home_4"] > result["home_3"]
    assert build.call_count == (0 if saved else 1)
    pd.testing.assert_frame_equal(match.actions, original_actions)
    torch.testing.assert_close(match.labels, original_labels)
    torch.testing.assert_close(graph.x, original_x)


@pytest.mark.parametrize("nodes", [True, False])
def test_feature_generation_unknown_recipient(nodes):
    context, graph, _, model = prediction_context(nodes)
    match = Match.__new__(Match)
    match.actions = context.actions
    original = match.actions.copy(deep=True)
    actions = original.copy(deep=True)
    actions.loc[13] = actions.loc[12]
    actions.loc[14] = actions.loc[12]
    actions.at[14, "success"] = True
    with (
        patch.object(match, "_get_cached_intended_receiver_model", return_value=(model, {"edge_in_dim": 2, "add_v_edge_features": False})),
        patch("datatools.graph_feature.construct_graph_for_action", side_effect=[graph.clone(), None]),
    ):
        result = match._apply_intended_receiver_model(actions, "success_intent/test")
    assert result.at[12, "intent_id"] == "home_4"
    assert result.at[13, "intent_id"] == "home_3"
    assert result.at[14, "intent_id"] == "home_3"
    stats = result.attrs["intended_receiver_stats"]
    assert stats["model_failed_passes_total"] == 2
    assert stats["model_failed_passes_scored"] == 1
    assert stats["model_angle_only_fallbacks"] == 1
    pd.testing.assert_frame_equal(match.actions, original)


def test_predictions_do_not_depend_on_observed_target():
    match, graph, label, model = prediction_context()
    unknown, _ = inference_gnn(match, model, device="cpu", graph_override=[graph.clone()], label_override=label[None])
    observed_label = label.clone()
    observed_label[5:7] = 3
    observed_label[config.LABEL_INDEX["success"]] = 1
    observed, _ = inference_gnn(match, model, device="cpu", graph_override=[graph.clone()], label_override=observed_label[None])
    pd.testing.assert_frame_equal(unknown, observed)
    with pytest.raises(ValueError, match="not an eligible prediction candidate"):
        utils.filter_features_and_labels([graph.clone()], label[None], model.args)
    model.args["task"] = "pass_intent"
    with pytest.raises(ValueError, match="not an eligible prediction candidate"):
        inference_gnn(match, model, device="cpu", graph_override=[graph.clone()], label_override=label[None])


@pytest.mark.parametrize("failure", ["goals", "trajectory", "alignment"])
def test_unknown_target_does_not_bypass_input_validation(failure):
    match, graph, label, model = prediction_context()
    labels = label[None]
    if failure == "goals":
        graph.x[:, config.NODE_FEATURE_IS_GOAL] = 0
    elif failure == "trajectory":
        graph.trajectory_speed_definition = "smoothed_mean_3d_speed_v1"
    else:
        labels = labels.repeat(2, 1)
    with pytest.raises(ValueError, match={"goals": "source graph", "trajectory": "trajectory speed definition mismatch", "alignment": "row-aligned"}[failure]):
        inference_gnn(match, model, device="cpu", graph_override=[graph], label_override=labels)
