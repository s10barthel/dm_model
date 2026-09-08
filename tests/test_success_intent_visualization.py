from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

import inference
from datatools import config
from scripts import visualize_action_components as viz


@pytest.fixture
def context():
    match = SimpleNamespace(
        actions=pd.DataFrame({"action_type": ["pass", "pass"], "success": [True, False]}, index=[7, 12]),
        labels=torch.ones((2, len(config.LABEL_COLUMNS))),
        runtime_feature_root=Path("original"),
    )
    graph = SimpleNamespace(edge_attr=torch.zeros((2, 2)))
    return match, graph


@pytest.mark.parametrize("missing", [True, False])
def test_failed_pass_uses_runtime_graph_without_changing_match(context, missing):
    match, graph = context
    original_labels = match.labels
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, 0] = 7
    with (
        patch.object(viz, "load_success_intent_labels", side_effect=FileNotFoundError if missing else None, return_value=labels),
        patch.object(viz, "resolve_match_id", return_value="match"),
        patch.object(viz, "construct_graph_for_action", return_value=graph) as build,
        patch.object(viz, "validate_model_graph_schemas", return_value={"edge_in_dim": 2, "add_v_edge_features": False}),
        patch.object(viz, "inference_gnn", return_value=(pd.DataFrame({"home_2": [0.25], "home_3": [0.75]}, index=[12]), None)) as run,
    ):
        result = viz.run_success_intent_component(match, object(), Path("missing"), "cpu", 12)
    assert result.sum() == 1
    assert list(result.index) == ["home_2", "home_3"]
    assert build.call_args.kwargs["feature_variant"] == "success_intent"
    assert run.call_args.kwargs["graph_override"] == [graph]
    assert run.call_args.kwargs["label_override"][0, 0] == 12
    assert match.labels is original_labels
    assert match.runtime_feature_root == Path("original")


def test_saved_pass_reuses_graph_and_label(context, tmp_path):
    match, graph = context
    graph_dir = viz.get_success_intent_graph_dir(tmp_path)
    graph_dir.mkdir(parents=True)
    (graph_dir / "match.pt").touch()
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, 0] = 7
    with (
        patch.object(viz, "load_success_intent_labels", return_value=labels),
        patch.object(viz, "resolve_match_id", return_value="match"),
        patch.object(viz.torch, "load", return_value=[graph]),
        patch.object(viz, "construct_graph_for_action") as build,
        patch.object(viz, "inference_gnn", return_value=(pd.DataFrame({"home_2": [1.]}, index=[7]), None)) as run,
    ):
        viz.run_success_intent_component(match, object(), tmp_path, "cpu", 7)
    build.assert_not_called()
    assert run.call_args.kwargs["graph_override"] == [graph]
    assert torch.equal(run.call_args.kwargs["label_override"], labels)


@pytest.mark.parametrize("failure", ["graph", "schema", "inference", "labels"])
def test_failures_preserve_match_state(context, failure):
    match, graph = context
    original_labels = match.labels
    with (
        patch.object(viz, "load_success_intent_labels", side_effect=ValueError("bad labels") if failure == "labels" else FileNotFoundError()),
        patch.object(viz, "construct_graph_for_action", return_value=None if failure == "graph" else graph),
        patch.object(viz, "validate_model_graph_schemas", return_value={"edge_in_dim": 4 if failure == "schema" else 2, "add_v_edge_features": False}),
        patch.object(viz, "inference_gnn", side_effect=ValueError("inference failed")),
        pytest.raises(ValueError),
    ):
        viz.run_success_intent_component(match, object(), Path("missing"), "cpu", 12)
    assert match.labels is original_labels
    assert match.runtime_feature_root == Path("original")


@pytest.mark.parametrize("overrides", [{"graph_override": []}, {"label_override": torch.zeros((1, 1))}])
def test_inference_requires_paired_overrides(overrides):
    with pytest.raises(ValueError, match="supplied together"):
        inference.inference_gnn(None, None, **overrides)


def test_inference_override_bypasses_saved_graph_loading(context):
    match, graph = context
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, 0] = 12
    model = SimpleNamespace(args={"task": "success_intent"})
    with (
        patch.object(inference, "resolve_match_graphs") as resolve,
        patch.object(inference, "filter_features_and_labels", side_effect=ValueError("stop after alignment")) as filtered,
        pytest.raises(ValueError, match="stop after alignment"),
    ):
        inference.inference_gnn(match, model, event_indices=[12], graph_override=[graph], label_override=labels)
    resolve.assert_not_called()
    assert filtered.call_args.args[0] == [graph]
    assert filtered.call_args.args[1] is labels


def test_malformed_saved_graphs_do_not_trigger_fallback(context, tmp_path):
    match, graph = context
    graph_dir = viz.get_success_intent_graph_dir(tmp_path)
    graph_dir.mkdir(parents=True)
    (graph_dir / "match.pt").touch()
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, 0] = 7
    with (
        patch.object(viz, "load_success_intent_labels", return_value=labels),
        patch.object(viz, "resolve_match_id", return_value="match"),
        patch.object(viz.torch, "load", return_value=[graph, graph]),
        patch.object(viz, "construct_graph_for_action") as build,
        pytest.raises(ValueError, match="row-aligned"),
    ):
        viz.run_success_intent_component(match, object(), tmp_path, "cpu", 7)
    build.assert_not_called()
