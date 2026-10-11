"""Player-only receiver outputs with independent goal context."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

import inference
from dataset import ActionDataset
from datatools import config
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.goal_context import goal_policy, resolve_training_goal_settings
from models.node_selection import selection_layout, selection_loss_metrics
from test_goal_context import DummyModel, settings, source


def receiver_source(prefix="home", extra_opponent=True):
    graph, label = source(prefix, task="failure_receiver")
    if extra_opponent:
        opponent = graph.x[1].clone()
        opponent[3] = 60
        graph.x = torch.cat([graph.x, opponent[None]])
        graph.node_ids.append(("away" if prefix == "home" else "home") + "_7")
        graph.edge_index = torch.cartesian_prod(torch.arange(graph.num_nodes), torch.arange(graph.num_nodes)).T
        graph.edge_attr = torch.ones(graph.edge_index.shape[1], 2)
        label[4] = graph.num_nodes
    return graph, label


def make_dataset(tmp_path, graphs, labels, args, **overrides):
    features, label_dir = tmp_path / "features", tmp_path / "labels"
    features.mkdir(exist_ok=True); label_dir.mkdir(exist_ok=True)
    torch.save(graphs, features / "m.pt")
    torch.save(torch.stack(labels), label_dir / "m.pt")
    kwargs = build_action_dataset_kwargs(args, train=False, diagnostic_label_dir=None,
                                        require_goal_next10_diagnostics=False)
    kwargs.update(overrides)
    return ActionDataset(["m"], feature_dir=features, label_dir=label_dir, **kwargs)


def match_for(graph, label, prefix="home"):
    tracking = pd.DataFrame({f"{pid}_{axis}": [float(graph.x[i, 3+j])]
                             for i, pid in enumerate(graph.node_ids) for j, axis in enumerate(["x", "y"])})
    return SimpleNamespace(tracking=tracking, labels=label[None],
                           actions=pd.DataFrame([dict(frame_id=0, object_id=prefix+"_1")]))


class ReceiverModel(DummyModel):
    def forward(self, graphs, dests=None):
        assert graphs.x.shape[1] == self.args["node_in_dim"]
        intent = graphs.x[:, -1]
        torch.testing.assert_close(torch.bincount(graphs.batch, weights=intent),
                                   intent.new_ones(graphs.num_graphs))
        return super().forward(graphs, dests)


@pytest.mark.parametrize("task", ["pass_height", "failure_receiver"])
@pytest.mark.parametrize("nodes", [True, False])
def test_new_defaults_and_legacy_resume(task, nodes):
    new = SimpleNamespace(task=task, goal_nodes_aware=nodes, include_goals=None)
    modern = resolve_training_goal_settings(new)
    assert modern.version == 2 and modern.input_goals == nodes and not modern.include_goals
    assert goal_policy(json.loads(json.dumps(vars(new)))) == modern
    old = dict(task=task, goal_nodes_aware=nodes)
    old_policy = goal_policy(old)
    assert old_policy.version == 1
    assert old_policy.include_goals == (task == "failure_receiver")
    assert old_policy.input_goals == (nodes and task == "failure_receiver")
    resumed = SimpleNamespace(goal_nodes_aware=None, include_goals=None)
    assert resolve_training_goal_settings(resumed, old) == old_policy
    with pytest.raises(ValueError, match="Cannot change"):
        resolve_training_goal_settings(SimpleNamespace(goal_nodes_aware=not nodes, include_goals=None), old)


@pytest.mark.parametrize("nodes", [True, False])
@pytest.mark.parametrize("include_out", [True, False])
def test_receiver_invalid_rows_are_counted(tmp_path, nodes, include_out):
    graph, label = receiver_source()
    args = settings("failure_receiver", nodes); args["include_out"] = include_out
    targets = [1, -1, float("nan"), float("inf"), 1.5, -2, 99, 4, 3]
    labels = []
    for i, target in enumerate(targets):
        row = label.clone(); row[0] = i; row[6] = target
        labels.append(row)
    dataset = make_dataset(tmp_path, [graph] * len(labels), labels, args)
    assert len(dataset) == (2 if include_out else 1)
    expected = {"nonfinite_index": 2, "noninteger_index": 1, "index_out_of_bounds": 2,
                "receiver_is_goal": 1, "receiver_not_opponent": 1}
    if not include_out:
        expected["missing_receiver"] = 1
    assert dataset.skipped_rows == {f"invalid_failure_receiver_target:{key}": count
                                    for key, count in expected.items()}
    assert dataset.features[0].node_ids[int(dataset.labels[0, 6])] == "away_2"
    assert dataset.features[0].node_ids[int(dataset.labels[0, 5])] == "home_3"
    assert dataset.features[0].x[int(dataset.labels[0, 5]), -1] == 1


@pytest.mark.parametrize("nodes", [True, False])
def test_shot_intentions_are_context_not_receiver_targets(tmp_path, nodes):
    graph, label = receiver_source()
    label[1] = 0; label[3] = 1; label[5] = 2; label[config.LABEL_INDEX["blocked"]] = 1
    dataset = make_dataset(tmp_path, [graph], [label], settings("failure_receiver", nodes))
    if nodes:
        assert len(dataset) == 1
        assert dataset.features[0].node_ids[int(dataset.labels[0, 5])] == "home_goal"
        assert dataset.features[0].x[2, -1] == 1
    else:
        assert len(dataset) == 0
        assert dataset.skipped_rows == {"invalid_failure_receiver_intent:intent_removed_with_goal": 1}


@pytest.mark.parametrize("intent,reason", [(float("nan"), "nonfinite_index"),
                                          (2.5, "noninteger_index"), (-1, "index_out_of_bounds"),
                                          (99, "index_out_of_bounds")])
def test_invalid_intentions_are_counted(tmp_path, intent, reason):
    graph, label = receiver_source(); label[5] = intent
    dataset = make_dataset(tmp_path, [graph], [label], settings("failure_receiver", False))
    assert len(dataset) == 0
    assert dataset.skipped_rows == {f"invalid_failure_receiver_intent:{reason}": 1}


@pytest.mark.parametrize("include_out", [True, False])
def test_variable_receiver_layout_and_goal_gradients(include_out):
    graph, label = receiver_source()
    second, other_label = receiver_source(extra_opponent=False)
    if include_out:
        other_label[6] = -1
    batch = Batch.from_data_list([graph, second])
    labels = torch.stack([label, other_label])
    args = settings("failure_receiver"); args["include_out"] = include_out
    layout = selection_layout(batch, labels, "failure_receiver", include_out, args)
    logits = torch.zeros(batch.num_nodes + (2 if include_out else 0), requires_grad=True)
    loss, _, _, _, probabilities = selection_loss_metrics(logits, layout)
    torch.testing.assert_close(probabilities, torch.tensor([1/3, 1/2] if include_out else [1/2, 1.]))
    loss.backward()
    assert torch.equal(logits.grad[[0, 2, 3, 4, 5, 7, 9, 10, 11, 12]], torch.zeros(10))
    labels[0, 6] = 4
    with pytest.raises(ValueError, match="not an eligible"):
        selection_layout(batch, labels, "failure_receiver", include_out, args)


@pytest.mark.parametrize("prefix", ["home", "away"])
@pytest.mark.parametrize("nodes", [True, False])
@pytest.mark.parametrize("include_out", [True, False])
@pytest.mark.parametrize("lane", [True, False])
def test_ordinary_and_posterior_outputs(monkeypatch, prefix, nodes, include_out, lane):
    graph, label = receiver_source(prefix)
    label[6] = float("nan")  # Hypothetical inference has no observed receiver.
    match = match_for(graph, label, prefix)
    args = settings("failure_receiver", nodes)
    args.update(include_out=include_out, node_in_dim=28 if lane else 27, lane_survival=lane)
    attached = []
    if lane:
        def attach_lane(match, graphs, labels, model):
            attached.append(True)
            for graph in graphs:
                graph.x = torch.cat([graph.x, graph.x.new_zeros((graph.num_nodes, 1))], dim=1)
            return graphs
        monkeypatch.setattr(inference, "attach_lane_survival_for_inference", attach_lane)
    model = ReceiverModel(args)
    probabilities, _ = inference.inference_gnn(match, model, device="cpu", graph_override=[graph], label_override=label[None])
    other = "away" if prefix == "home" else "home"
    expected_ids = {other+"_2", other+"_7"} | ({"out"} if include_out else set())
    assert set(probabilities.columns) == expected_ids
    assert probabilities.iloc[0].sum() == pytest.approx(1.)
    monkeypatch.setattr(inference, "resolve_match_graphs", lambda *a, **kw: ([graph], None))
    posterior = inference.inference_gnn_posterior(match, model, device="cpu")
    assert set(posterior.defender) == expected_ids
    assert (prefix+"_goal" in set(posterior.option)) == nodes
    assert posterior.groupby("option").posterior.sum().tolist() == pytest.approx([1.] * posterior.option.nunique())
    conditional = posterior[posterior.option == prefix+"_3"].set_index("defender").posterior
    for candidate in expected_ids:
        assert conditional[candidate] == pytest.approx(probabilities.iloc[0][candidate])
    assert len(attached) == (2 if lane else 0)


@pytest.mark.parametrize("task", ["pass_success", "pass_height"])
@pytest.mark.parametrize("target,reason", [(2, "target_is_goal"), (1, "target_not_teammate"),
                                          (float("nan"), "target_nonfinite_index"),
                                          (2.5, "target_noninteger_index"), (99, "target_index_out_of_bounds")])
def test_pass_models_skip_invalid_targets(tmp_path, task, target, reason):
    graph, label = source(); label[5] = target
    dataset = make_dataset(tmp_path, [graph], [label], settings(task, False))
    assert len(dataset) == 0
    assert dataset.skipped_rows == {f"invalid_{task}_target:{reason}": 1}


@pytest.mark.parametrize("main_version", [1, 2])
@pytest.mark.parametrize("aux_version", [1, 2])
def test_pass_height_ipw_membership_and_dependency_context(tmp_path, main_version, aux_version):
    from ipw_preparation import sample_records
    graph, label = source()
    main_args = settings("pass_height", version=main_version)
    aux_args = settings("pass_intent", version=aux_version)
    main = make_dataset(tmp_path, [graph], [label], main_args, auxiliary_model_args=aux_args)
    kwargs = build_action_dataset_kwargs(main_args, train=False, diagnostic_label_dir=None,
                                        require_goal_next10_diagnostics=False)
    ipw_kwargs = build_ipw_dataset_kwargs(kwargs, aux_args, {}, diagnostic_label_dir=None,
                                         require_goal_next10_diagnostics=False)
    aux = ActionDataset(["m"], feature_dir=tmp_path / "features", label_dir=tmp_path / "labels", **ipw_kwargs)
    assert sample_records(main) == sample_records(aux)
    assert main.features[0].num_nodes == (6 if main_version == 2 else 4)
    assert main.features[0].auxiliary_graph.num_nodes == aux.features[0].num_nodes == (6 if aux_version == 2 else 4)


def test_prepared_cache_separates_policy_versions(tmp_path):
    task = "pass_height"
    from prepared_dataset import PreparedActionDataset
    graph, label = source(task=task)
    make_dataset(tmp_path, [graph], [label], settings(task))
    cache_ids = []
    for version in [1, 2]:
        kwargs = build_action_dataset_kwargs(settings(task, version=version), train=False,
                                            diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
        disk = PreparedActionDataset(["m"], feature_dir=tmp_path / "features", label_dir=tmp_path / "labels",
                                     cache_dir=tmp_path / "cache", **kwargs)
        assert len(disk) == 1
        cache_ids.append(disk.cache_id)
    assert cache_ids[0] != cache_ids[1]


@pytest.mark.parametrize("nodes", [True, False])
def test_physical_cache_pass_height_respects_input_policy(nodes):
    from physical_pass_model import _pass_height_predictions_for_graphs
    graph, _ = source()
    args = settings("pass_height", nodes)
    class ContextModel(DummyModel):
        def forward(self, graphs, dests=None):
            assert graphs.num_nodes == (6 if nodes else 4)
            return super().forward(graphs, dests)
    predictions = _pass_height_predictions_for_graphs([graph], ContextModel(args), device="cpu")
    assert set(predictions[0]) == {"home_1", "home_3", "home_4"}


@pytest.mark.parametrize("nodes", [True, False])
@pytest.mark.parametrize("include_out", [True, False])
def test_real_receiver_model_loss_matches_inference(tmp_path, nodes, include_out):
    from models.gnn import GNN
    torch.manual_seed(19)
    graph, label = receiver_source()
    args = settings("failure_receiver", nodes)
    args.update(model="gat", gnn_task="node_selection", node_in_dim=27, node_emb_dim=8,
                graph_emb_dim=8, gnn_layers=1, gnn_heads=2, skip_conn=True,
                out_dim=1, mlp_h1_dim=8, mlp_h2_dim=4, include_out=include_out)
    dataset = make_dataset(tmp_path, [graph], [label], args)
    batch = Batch.from_data_list(dataset.features)
    model = GNN(args).eval()
    layout = selection_layout(batch, dataset.labels, "failure_receiver", include_out, args)
    loss, _, _, _, observed_probabilities = selection_loss_metrics(model(batch), layout)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    match = match_for(graph, label)
    probabilities, _ = inference.inference_gnn(match, model, device="cpu",
                                               graph_override=[graph], label_override=label[None])
    assert probabilities.iloc[0]["away_2"] == pytest.approx(float(observed_probabilities[0]), abs=1e-6)


@pytest.mark.parametrize("task", ["pass_height", "failure_receiver"])
@pytest.mark.parametrize("version", [None, 1, 2])
def test_loading_checkpoints_preserves_goal_policy(tmp_path, monkeypatch, task, version):
    import models.utils as utils
    from models.gnn import GNN
    args = settings(task, version=version)
    if version is None:
        args.pop("goal_context_version")
    args.update(model="gat", gnn_task=config.TASK_CONFIG.at[task, "gnn_task"],
                node_in_dim=27 if task == "failure_receiver" else 26, node_emb_dim=8,
                graph_emb_dim=8, gnn_layers=1, gnn_heads=2, skip_conn=True,
                out_dim=1, mlp_h1_dim=8, mlp_h2_dim=4)
    model = GNN(args)
    (tmp_path / "args.json").write_text(json.dumps(args))
    torch.save(model.state_dict(), tmp_path / "best_weights.pt")
    monkeypatch.setattr(utils, "get_model_path", lambda *a: tmp_path)
    restored = utils.load_model(f"{task}/fixture", device="cpu")
    assert goal_policy(restored.args) == goal_policy(args)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor)


def test_only_missing_receiver_sentinel_maps_to_out():
    graph, label = receiver_source()
    batch = Batch.from_data_list([graph])
    args = settings("failure_receiver"); args["include_out"] = True
    label[6] = graph.num_nodes
    with pytest.raises(ValueError, match="outside its graph"):
        selection_layout(batch, label[None], "failure_receiver", True, args)
    label[6] = -1
    label[4] = float("nan")  # Out is identified by the sentinel, not the saved node count.
    layout = selection_layout(batch, label[None], "failure_receiver", True, args)
    ids, indices, target = layout[0]
    assert indices[0, int(target[0])] == batch.num_nodes


@pytest.mark.parametrize("nodes", [True, False])
def test_physical_pass_height_lane_survival_after_goal_remapping(nodes):
    from physical_pass_model import _pass_height_predictions_for_graphs, pc_xpass_lane_survival_column_for_mode
    graph, label = source()
    args = settings("pass_height", nodes)
    args.update(lane_survival=True, lane_survival_mode="max", node_in_dim=27)
    row = {pc_xpass_lane_survival_column_for_mode(pid, "max"): .7
           for pid in ["home_1", "home_3", "home_4"]}
    class LaneModel(DummyModel):
        def forward(self, graphs, dests=None):
            assert graphs.num_nodes == (6 if nodes else 4)
            assert graphs.x.shape[1] == 27
            mask = (graphs.x[:, 0] == 1) & (graphs.x[:, 2] == 0) & (graphs.x[:, 13] == 0)
            torch.testing.assert_close(graphs.x[mask, -1], torch.full((2,), .7))
            assert graphs.x[graphs.x[:, 13] == 1, -1].item() == 0
            return super().forward(graphs, dests)
    predictions = _pass_height_predictions_for_graphs([graph], LaneModel(args), device="cpu",
                                                     labels=[label], pc_xpass_rows=[row])
    assert set(predictions[0]) == {"home_1", "home_3", "home_4"}
