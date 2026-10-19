"""Scalar node-speed ablation across training, saved models, and inference."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest import TestCase

import pytest
import torch
from torch_geometric.data import Batch

from dataset import ActionDataset
from datatools import config
from datatools.utils import filter_features_and_labels
from models import utils as model_utils
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.inference_masks import resolve_model_masks
from models.node_feature_config import resolve_speed_aware
from scripts import run_hawkeye_loc, train_relevant_models as wrapper
from split_fixture import install_wrapper_split
from test_benchmark_no_accel import make_graph, make_pass_labels, write_checkpoint
from test_success_intent_mode_independent import make_training_args
from test_training_controls import synthetic_training


@pytest.fixture
def wrapper_split(request):
    case = TestCase()
    install_wrapper_split(case)
    request.addfinalizer(case.doCleanups)


@pytest.mark.parametrize("flags,enabled", [([], True), (["--speed"], True), (["--no-speed"], False)])
def test_wrapper_parses_and_forwards_speed_to_every_task(wrapper_split, flags, enabled):
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original"]), \
         patch.object(wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]):
        parsed = wrapper.parse_args(["--feature-run-id", "feature_run", "--only-success-intent", *flags])
        assert wrapper.resolve_wrapper_feature_flags(parsed)["speed_aware"] == enabled
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--feature-run-id", "feature_run", "--speed", "--no-speed"])

    args = make_training_args("feature_run", pass_success_ipw=False)
    args.enabled_tasks.update(pass_height=True, failure_receiver=True)
    args.speed_aware = parsed.speed_aware
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        commands, _, _, _, feature_flags = wrapper.build_training_commands(args)
    assert len(commands) == 8
    assert feature_flags["speed_aware"] == enabled
    for command in commands:
        assert command.count("--speed" if enabled else "--no-speed") == 1
        assert ("--no-speed" if enabled else "--speed") not in command


@pytest.mark.parametrize("task,width", [("action_intent", 26), ("success_intent", 24)])
@pytest.mark.parametrize("other_masks", [
    {}, {"accel_aware": False}, {"vel_node_features_aware": False},
    {"poss_vel_aware": False}, {"xy_only": True},
])
def test_speed_only_mask_matches_dataset_and_runtime(tmp_path, task, width, other_masks):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    source = make_graph(node_dim=width, edge_dim=5)
    # Include a retained goal node and a distinct pass-trajectory speed slot.
    source.x = torch.cat([source.x, source.x[1:2].clone()])
    source.x[2, config.NODE_FEATURE_IS_GOAL] = 1
    source.x[:, 19] = 3
    original = source.clone()
    torch.save([source], features / "match.pt")
    torch.save(make_pass_labels(), labels / "match.pt")
    options = dict(task=task, possessor_aware=True, keeper_aware=True, ball_z_aware=True,
                   poss_vel_aware=True, poss_rel_vel_aware=True, extend_features=False,
                   goal_nodes_aware=True, edge_in_dim=5, v_edge_feature_mode="all",
                   relative_speed_edge_feature_mode="all", goal_poss_dist=False)
    options.update(other_masks)
    enabled = ActionDataset(["match"], feature_dir=features, label_dir=labels,
                            speed_aware=True, **options)[0][0]
    disabled = ActionDataset(["match"], feature_dir=features, label_dir=labels,
                             speed_aware=False, **options)[0][0]
    expected = enabled.x.clone()
    expected[:, config.NODE_FEATURE_SPEED] = 0
    torch.testing.assert_close(disabled.x, expected, rtol=0, atol=0)
    torch.testing.assert_close(disabled.edge_index, enabled.edge_index)
    torch.testing.assert_close(disabled.edge_attr, enabled.edge_attr)
    assert disabled.x.shape == enabled.x.shape
    assert disabled.x.shape[1] == width
    if task == "success_intent":
        assert torch.all(disabled.x[:, 19] == 3)

    runtime, _ = filter_features_and_labels([source], make_pass_labels(),
        dict(options, speed_aware=False, xy_only=other_masks.get("xy_only", False),
             sparsify="none", max_edge_dist=10))
    torch.testing.assert_close(runtime[0].x, expected, rtol=0, atol=0)
    torch.testing.assert_close(runtime[0].edge_attr, enabled.edge_attr)
    torch.testing.assert_close(source.x, original.x)
    stored = torch.load(features / "match.pt", weights_only=False)[0]
    torch.testing.assert_close(stored.x, original.x)


def test_speed_batch_adapter_masks_a_copy():
    source = Batch.from_data_list([make_graph()])
    expected = source.x.clone()
    expected[:, config.NODE_FEATURE_SPEED] = 0
    disabled = model_utils.adapt_batch_graphs_for_model(source, {"speed_aware": False})
    torch.testing.assert_close(disabled.x, expected)
    assert torch.all(source.x[:, config.NODE_FEATURE_SPEED] == 7)
    torch.testing.assert_close(disabled.edge_attr, source.edge_attr)


@pytest.mark.parametrize("saved", ["missing", None, False, True])
def test_speed_checkpoint_roundtrip_and_defaults(tmp_path, saved):
    directory = tmp_path / "action_intent" / "speed"
    write_checkpoint(directory)
    args_path = directory / "args.json"
    args = json.loads(args_path.read_text())
    if saved != "missing":
        args["speed_aware"] = saved
    args_path.write_text(json.dumps(args))
    with patch.object(model_utils, "get_model_path", return_value=directory):
        record = model_utils.get_model_record("action_intent/speed")
        model = model_utils.load_model("action_intent/speed", device="cpu")
    expected = saved is not False
    assert record["args"]["speed_aware"] == expected
    assert record["feature_signature"]["speed_aware"] == expected
    assert model.args["speed_aware"] == expected
    assert resolve_speed_aware(SimpleNamespace(**args)) == expected
    assert model_utils.extract_model_feature_signature(args)["speed_aware"] == expected


@pytest.mark.parametrize("target_enabled,checkpoint", [(False, True), (True, False), (False, None)])
def test_ipw_uses_its_own_speed_setting(target_enabled, checkpoint):
    target = build_action_dataset_kwargs(
        dict(task="pass_success", edge_in_dim=2, speed_aware=target_enabled),
        train=False, diagnostic_label_dir=None)
    dependency = build_ipw_dataset_kwargs(
        target, dict(edge_in_dim=2, speed_aware=checkpoint), None,
        diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert target["speed_aware"] == target_enabled
    assert dependency["speed_aware"] == (checkpoint is not False)
    assert build_action_dataset_kwargs(
        dict(task="pass_success", edge_in_dim=2), train=False,
        diagnostic_label_dir=None)["speed_aware"] is True


@pytest.mark.parametrize("enabled", [None, False, True])
def test_speed_runtime_alias_and_disabled_status(enabled):
    args = dict(task="action_intent", node_in_dim=25, edge_in_dim=2, speed_aware=enabled)
    alias = resolve_model_masks(args, ["speed"])
    semantic = resolve_model_masks(args, ["node.speed"])
    assert alias.operations == semantic.operations
    assert alias.operations[0].status == ("no additional effect" if enabled is False else "applied")
    source = make_graph()
    masked = alias.mask_nodes(source.x)
    expected = source.x.clone()
    if enabled is not False:
        expected[:, config.NODE_FEATURE_SPEED] = 0
    torch.testing.assert_close(masked, expected)


def test_location_inference_reports_speed_awareness():
    args = dict(poss_vel_aware=False, poss_rel_vel_aware=False, accel_aware=False)
    model = SimpleNamespace(args=dict(args, speed_aware=True))
    with pytest.warns(UserWarning, match="speed_aware"):
        warnings = run_hawkeye_loc.compatibility_warnings({"pass_intent": "fixture"}, {"pass_intent": model})
    assert warnings[0]["incompatible_features"] == ["speed_aware"]
    model.args["speed_aware"] = False
    assert run_hawkeye_loc.compatibility_warnings({}, {"pass_intent": model}) == []
    model.args.update(speed_aware=True, vel_node_features_aware=False)
    assert run_hawkeye_loc.compatibility_warnings({}, {"pass_intent": model}) == []


@pytest.mark.parametrize("flags,enabled", [([], True), (["--speed"], True), (["--no-speed"], False)])
def test_actual_training_saves_speed_setting(synthetic_training, flags, enabled):
    run, _, _, _, _ = synthetic_training
    checkpoint = run("speed", ["--max-epochs", "1", *flags])
    assert checkpoint["args"]["speed_aware"] == enabled
    assert checkpoint["metadata"]["feature_signature"]["speed_aware"] == enabled
    with pytest.raises(SystemExit):
        run("conflict", ["--speed", "--no-speed"])


@pytest.mark.parametrize("flag", ["--speed", "--no-speed"])
def test_resume_and_source_refit_reject_speed_overrides(flag):
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--resume-id", "pass_intent/source", flag])
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--refit-source-model-id", "pass_intent/source", flag])
