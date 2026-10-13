"""Stored graph requirements exclude columns attached at inference time."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
import inference
from models import utils
import physical_pass_model as physics
import project_config


def model(task="pass_intent", node_dim=27, **settings):
    return SimpleNamespace(args={
        "task": task,
        "node_in_dim": node_dim,
        "edge_in_dim": 5,
        "add_v_edge_features": True,
        "add_relative_speed_edge_features": True,
        **settings,
    })


def graph():
    x = torch.zeros((3, 26))
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = torch.tensor([1, 1, 0])
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[:, config.NODE_FEATURE_X] = torch.tensor([10, 20, 60])
    x[:, config.NODE_FEATURE_Y] = 34
    return Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]),
                edge_attr=torch.ones((2, 5)), node_ids=["home_1", "home_2", "away_3"])


@pytest.mark.parametrize("task,width,settings,expected", [
    ("pass_intent", 27, {"lane_survival": True}, 26),
    ("pass_intent", 27, {}, 27),
    ("failure_receiver", 27, {}, 26),
    ("failure_receiver", 28, {"lane_survival": True}, 26),
    ("overall_scoring", 26, {"lane_survival": True}, 26),
    ("pass_intent", 27, {"lane_survival": True, "gnn_task": "graph_binary"}, 27),
    ("unknown_task", 27, {"lane_survival": True, "gnn_task": "node_selection"}, 26),
    ("unknown_task", 27, {"lane_survival": True}, 27),
])
def test_stored_schema_excludes_only_runtime_columns(task, width, settings, expected):
    checkpoint = model(task, width, **settings)
    original_args = deepcopy(checkpoint.args)
    full_schema = utils.get_model_graph_schema(checkpoint)

    feature_schema = utils.get_model_feature_graph_schema(checkpoint)

    assert feature_schema == {**full_schema, "node_in_dim": expected}
    assert utils.get_model_graph_schema(checkpoint) == full_schema
    assert full_schema["node_in_dim"] == width
    assert checkpoint.args == original_args


def test_absent_models_have_no_schema():
    assert utils.get_model_feature_graph_schema(None) is None
    assert utils.get_model_feature_graph_schema(SimpleNamespace()) is None


def mixed_models():
    return {
        "action_intent": model("action_intent", 26, edge_in_dim=2,
                               add_v_edge_features=False, add_relative_speed_edge_features=False),
        "pass_intent": model(lane_survival=True),
        "pass_success": model("pass_success", 26),
        "outcome_scoring": model("outcome_scoring", 26),
        "outcome_conceding": model("outcome_conceding", 26),
        "xpass_pass_height": model("pass_height", 26),
        "xpass_pass_intent": model(lane_survival=True),
    }


def test_mixed_checkpoints_aggregate_stored_width_and_edge_union():
    checkpoints = mixed_models()
    original_args = {name: deepcopy(checkpoint.args) for name, checkpoint in checkpoints.items()}
    assert utils.validate_model_graph_schemas(checkpoints) == {
        "node_in_dim": 26, "edge_in_dim": 5,
        "add_v_edge_features": True, "add_relative_speed_edge_features": True,
    }
    assert {name: checkpoint.args for name, checkpoint in checkpoints.items()} == original_args


@pytest.fixture
def feature_run(tmp_path, monkeypatch):
    monkeypatch.setattr(project_config, "FEATURE_RUNS_DIR", tmp_path)
    monkeypatch.setattr(utils, "FEATURE_RUNS_DIR", tmp_path)
    run_id = "feature_runtime"
    root = tmp_path / run_id
    graph_dir = project_config.get_action_graph_dir(root)
    graph_dir.mkdir(parents=True)
    torch.save([graph()], graph_dir / "match_1.pt")
    project_config.get_resolved_action_dir("model", root=root).mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"intended_receiver_modes": ["model"]}), encoding="utf-8")
    return run_id


def test_explicit_feature_run_accepts_mixed_checkpoint_requirements(feature_run):
    required = utils.validate_model_graph_schemas(mixed_models())
    selected = utils.resolve_runtime_feature_run_context(feature_run, {}, None, None, required)
    assert selected["feature_run_id"] == feature_run
    assert selected["selection"] == "explicit"
    assert selected["intended_receiver_mode"] == "model"
    assert selected["feature_schema"] == required


def test_real_missing_node_feature_still_rejected(feature_run):
    required = utils.validate_model_graph_schemas({"pass_intent": model()})
    with pytest.raises(ValueError, match="incompatible.*node_in_dim.*27"):
        utils.resolve_runtime_feature_run_context(feature_run, {}, None, "model", required)


def test_missing_edge_features_still_rejected():
    required = utils.validate_model_graph_schemas(mixed_models())
    with pytest.raises(ValueError, match="incompatible"):
        utils.validate_feature_graph_schema({"node_in_dim": 26, "edge_in_dim": 2}, required)


@pytest.fixture
def lane_cache(tmp_path):
    cache = tmp_path / "pc_xpass"
    (cache / "matches").mkdir(parents=True)
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame([{
        "match_id": "match_1", "action_index": 7,
        physics.pc_xpass_lane_survival_column_for_mode("home_2", "max"): 0.42,
    }]).to_parquet(cache / "matches" / "match_1.parquet", index=False)
    return cache


def attach_lane(cache, checkpoint=None):
    source = graph()
    match = SimpleNamespace(match_id="match_1", graph_features_0=[source])
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, config.LABEL_INDEX["action_index"]] = 7
    labels[0, config.LABEL_INDEX["is_pass"]] = 1
    labels[0, config.LABEL_INDEX["intent_index"]] = 1
    checkpoint = checkpoint or model(lane_survival=True)
    checkpoint.args["lane_survival_runtime_cache_dir"] = str(cache)
    # Feature filtering supplies a model-local clone before runtime attachment.
    attached = inference.attach_lane_survival_for_inference(match, [source.clone()], labels, checkpoint)
    return source, attached, checkpoint


def test_runtime_attachment_supplies_real_values_and_preserves_source(lane_cache):
    source, attached, checkpoint = attach_lane(lane_cache)
    batch = utils.adapt_batch_graphs_for_model(Batch.from_data_list(attached), checkpoint.args)
    assert batch.x.shape == (3, 27)
    torch.testing.assert_close(batch.x[:, -1], torch.tensor([0, 0.42, 0]))
    torch.testing.assert_close(batch.x[:, :26], source.x)
    torch.testing.assert_close(source.x, graph().x)


@pytest.mark.parametrize("failure", ["missing", "fingerprint", "mode"])
def test_lane_cache_errors_remain_actionable(lane_cache, failure):
    checkpoint = model(lane_survival=True)
    if failure == "missing":
        lane_cache = lane_cache / "missing"
    elif failure == "fingerprint":
        checkpoint.args["lane_survival_cache_fingerprint"] = "different-training-cache"
    else:
        checkpoint.args["lane_survival_mode"] = "top_999"
    with pytest.raises(inference.LaneSurvivalCacheError, match="generate_physical_xpass.py --pc-xpass"):
        attach_lane(lane_cache, checkpoint)


def test_full_input_validation_still_requires_runtime_column():
    checkpoint = model(lane_survival=True)
    with pytest.raises(ValueError, match="requires node_in_dim=27.*provide node_in_dim=26"):
        utils.adapt_batch_graphs_for_model(Batch.from_data_list([graph()]), checkpoint.args)
