"""Architecture integration and physical self-loop regression coverage."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax

from datatools import config
from datatools.utils import drop_nodes, filter_features_and_labels, sparsify_edges
from dataset import ActionDataset
from models.architecture import (ARCHITECTURES, add_architecture_arguments, architecture_identity,
                                 normalize_architecture)
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.gnn import GNN
from models.inference_masks import configure_inference_masks
from models.relationship import RelationshipConv
from models.self_loops import capture_self_attributes, explicit_self_edges, original_self_attributes
from models import utils
from prepared_dataset import preparation_code_paths
from scripts import train_relevant_models as wrapper
from test_benchmark_no_accel import make_model_args, make_pass_labels
from test_goal_poss_dist import graph, artifacts, FEATURE_OPTIONS
from test_success_intent_mode_independent import make_training_args
from test_training_controls import synthetic_training
from test_training_refit import sources
from training_refit import prepare_refit, load_refit_config

REAL_RUN_EPOCH = utils.run_epoch


def settings(name="gat", **overrides):
    return normalize_architecture(dict(make_model_args(), model=name, gnn_heads=2,
                                       **overrides), fresh=True)


def velocity_graph():
    g = graph()
    g.x[:, 5:7] = torch.tensor([[0., 0.], [0., 0.], [3., 4.],
                                [0., 0.], [1., 0.], [2., 0.]])
    g.edge_attr = torch.cat((g.edge_attr, torch.ones(g.num_edges, 3)), dim=1)
    capture_self_attributes(g)
    return g


def test_selection_defaults_aliases_and_conflicts():
    parser = argparse.ArgumentParser()
    add_architecture_arguments(parser)
    assert normalize_architecture(vars(parser.parse_args([])), fresh=True) == {
        "model": "gat", "gnn_architecture": "gat", "gnn_architecture_version": 1,
        "self_loop_policy": "explicit_v1"}
    for name in (*ARCHITECTURES, "gcn", "gin"):
        resolved = normalize_architecture(vars(parser.parse_args(["--model", name])), fresh=True)
        assert resolved["model"] == name
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_architecture(vars(parser.parse_args(["--model", "gat", "--gnn-architecture", "transformer"])))
    assert normalize_architecture({"model": "gat"})["self_loop_policy"] == "legacy"
    with pytest.raises(SystemExit):
        parser.parse_args(["--gnn-architecture", "gcn"])


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_wrapper_forwards_every_task(name):
    args = make_training_args("feature_run", pass_success_ipw=False, pass_height=True)
    args.enabled_tasks.update(pass_height=True, failure_receiver=True)
    args.gnn_architecture = name
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        commands = wrapper.build_training_commands(args)[0]
    assert len(commands) == 8
    for command in commands:
        assert wrapper.get_cli_value(command, "--gnn-architecture") == name
        assert "--model" not in command
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--resume-id", "pass_intent/x", "--gnn-architecture", name])


@pytest.mark.parametrize("name", ARCHITECTURES)
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("task,kind", [("action_intent", "node_selection"),
    ("pass_success", "node_binary"), ("overall_return", "graph_regression"),
    ("pass_dest", "graph_multiclass")])
def test_forward_backward_roundtrip_and_permutation(name, skip, task, kind, tmp_path):
    args = settings(name, skip_conn=skip, task=task, gnn_task=kind)
    batch = Batch.from_data_list([graph(), graph()])
    original = batch.clone()
    model = GNN(args)
    destinations = torch.ones(2, 2) if "dest" in task else None
    output = model(batch, destinations)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.encoder.gnn_layers.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.save(model.state_dict(), tmp_path / "best_weights.pt")
    (tmp_path / "args.json").write_text(json.dumps(args))
    with patch.object(utils, "get_model_path", return_value=tmp_path):
        loaded = utils.load_model("action_intent/fixture", "cpu")
        record = utils.get_model_record("action_intent/fixture")
    torch.testing.assert_close(output, loaded(batch, destinations))
    assert record["architecture_identity"] == architecture_identity(args)
    order = torch.tensor([2, 0, 5, 3, 1, 4])
    reordered = graph().subgraph(order)
    single = model.encoder(Batch.from_data_list([graph()]))
    permuted = model.encoder(Batch.from_data_list([reordered]))
    torch.testing.assert_close(permuted[0], single[0][order], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(permuted[1], single[1], atol=1e-5, rtol=1e-5)
    for key in ("x", "edge_attr", "edge_index"):
        torch.testing.assert_close(batch[key], original[key])


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_physical_model_and_empty_edges(name):
    from physical_pass_model import PHYSICAL_XPASS_LOGIT_ATTR
    model = GNN(settings(name, task="pass_success", gnn_task="node_binary",
                         use_physical_xpass=True, physical_xpass_model="gat_phys_logit_offset"))
    g = graph()
    g.edge_index = torch.empty(2, 0, dtype=torch.long)
    g.edge_attr = torch.empty(0, 2)
    setattr(g, PHYSICAL_XPASS_LOGIT_ATTR, torch.zeros(g.num_nodes))
    output = model(Batch.from_data_list([g]))
    assert output.numel() == g.num_nodes and torch.isfinite(output).all()
    output.sum().backward()


def test_relationship_aggregation_directions_and_empty_neighbours():
    mean = RelationshipConv(3, 4, 2, 2)
    attention = RelationshipConv(3, 4, 2, 2, attention=True)
    attention.load_state_dict(mean.state_dict(), strict=False)
    for scorer in (attention.in_attention, attention.out_attention):
        torch.nn.init.zeros_(scorer.weight)
        torch.nn.init.zeros_(scorer.bias)
    x = torch.randn(4, 3)
    edges = torch.tensor([[0, 2, 2], [1, 1, 0]])
    attrs = torch.randn(3, 2)
    torch.testing.assert_close(mean(x, edges, attrs), attention(x, edges, attrs))
    messages = torch.tensor([[1., 2., 3., 4.], [3., 4., 5., 6.], [9., 8., 7., 6.]])
    incoming = mean.aggregate(messages, edges[1], 4, None)
    outgoing = mean.aggregate(messages, edges[0], 4, None)
    torch.testing.assert_close(incoming[1], messages[:2].mean(0))
    torch.testing.assert_close(outgoing[2], messages[1:].mean(0))
    assert not incoming[3].any() and not outgoing[3].any()
    torch.nn.init.normal_(attention.in_attention.weight)
    weights = softmax(attention.in_attention(messages), edges[1], num_nodes=4)
    torch.testing.assert_close(weights[:2].sum(0), torch.ones(2))
    expected = (messages[:2].reshape(2, 2, 2) * weights[:2, :, None]).sum(0).flatten()
    torch.testing.assert_close(attention.aggregate(messages, edges[1], 4, attention.in_attention)[1], expected)
    assert torch.isfinite(attention(x, edges[:, :0], attrs[:0])).all()


@pytest.mark.parametrize("name", ARCHITECTURES[1:])
def test_edge_content_with_identical_nodes(name):
    layer = GNN(settings(name)).encoder.gnn_layers[0]
    if name == "transformer":
        with torch.no_grad():
            layer.lin_edge.weight.fill_(1.)
    else:
        with torch.no_grad():
            for param in layer.edge_mlp.parameters():
                param.fill_(0.1)
    nodes = torch.ones(2, 8)
    edges = torch.tensor([[0], [1]])
    a = layer(nodes, edges, torch.zeros(1, 2))
    b = layer(nodes, edges, torch.ones(1, 2))
    assert not torch.allclose(a[1], b[1])


def test_self_loop_restoration_optional_features_and_node_filtering():
    g = velocity_graph()
    raw = g.clone()
    g.x[:, 5:7] = 0  # A node ablation must not erase original edge geometry.
    labels = make_pass_labels(intent_index=4)[0]
    keep = torch.tensor([True, False, True, True, True, True])
    reduced, _ = drop_nodes(g, labels, keep)
    batch = Batch.from_data_list([reduced, reduced])
    index, attrs = explicit_self_edges(batch, settings(edge_in_dim=5))
    loop = index[0] == index[1]
    assert loop.sum() == batch.num_nodes
    torch.testing.assert_close(attrs[loop], torch.cat([raw.node_self_edge_attr[keep]] * 2))
    assert abs(attrs[loop][0, 2]) < 1e-6 and attrs[loop][0, 3] == 1
    assert attrs[loop][1, 2] > .999
    for mode in ("none", "no_poss"):
        _, masked = explicit_self_edges(batch, settings(edge_in_dim=5, v_edge_feature_mode=mode,
                                   relative_speed_edge_feature_mode=mode))
        if mode == "none":
            assert not masked[:, 2:].any()
        else:
            incident = (batch.x[index[0], 13] == 1) | (batch.x[index[1], 13] == 1)
            assert not masked[incident, 2:].any()
    torch.testing.assert_close(g.node_self_edge_attr, raw.node_self_edge_attr)


def test_existing_duplicate_loops_keep_first_and_invalid_schema():
    g = graph()
    g.edge_index = torch.cat([g.edge_index, torch.tensor([[2, 2], [2, 2]])], 1)
    g.edge_attr = torch.cat([g.edge_attr, torch.tensor([[0., 1.], [90., .2]])])
    index, attr = explicit_self_edges(g, settings())
    torch.testing.assert_close(attr[index[0] == index[1]], torch.tensor([[0., 1.]] * 6))
    assert g.num_edges == 32
    g.edge_attr = torch.ones(32, 3)
    with pytest.raises(ValueError, match="edge width"):
        original_self_attributes(g)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_runtime_masks_apply_to_restored_loops(name):
    model = GNN(settings(name, edge_in_dim=5)).eval()
    configure_inference_masks({"model": model}, ["v-edge-features", "edge.is_teammate"])
    batch = Batch.from_data_list([velocity_graph()])
    received = []
    handle = model.encoder.gnn_layers[0].register_forward_pre_hook(lambda _, inputs: received.append(inputs))
    model(batch)
    handle.remove()
    index, attr = received[0][1:]
    assert (index[0] == index[1]).sum() == batch.num_nodes
    assert not attr[:, 1:4].any()
    assert batch.node_self_edge_attr[:, 1].all()


def test_legacy_gat_exactly_matches_original_layer():
    args = make_model_args()
    model = GNN(args).eval()
    reference = GATConv(8, 8, heads=1, edge_dim=2)
    reference.load_state_dict(model.encoder.gnn_layers[0].state_dict())
    batch = Batch.from_data_list([graph()])
    x = model.encoder.node_in_fc(batch.x)
    torch.testing.assert_close(reference(x, batch.edge_index, batch.edge_attr),
                               model.encoder.gnn_layers[0](x, batch.edge_index, batch.edge_attr), atol=0, rtol=0)
    assert model.args["self_loop_policy"] == "legacy"
    corrected = GNN(settings())
    assert not corrected.encoder.gnn_layers[0].add_self_loops


def test_dataset_runtime_sparsification_and_prepared_sharing(artifacts):
    raw = velocity_graph()
    torch.save([raw], artifacts["feature_dir"] / "m.pt")
    common = dict(artifacts, edge_in_dim=5, sparsify="delaunay", vel_node_features_aware=False,
                  v_edge_feature_mode="all", relative_speed_edge_feature_mode="all")
    dataset = ActionDataset(["m"], **common)
    args = dict(FEATURE_OPTIONS, edge_in_dim=5, sparsify="delaunay", vel_node_features_aware=False,
                v_edge_feature_mode="all", relative_speed_edge_feature_mode="all")
    filtered, _ = filter_features_and_labels([raw], make_pass_labels(intent_index=4), args)
    for prepared in (dataset[0][0], filtered[0]):
        assert not (prepared.edge_index[0] == prepared.edge_index[1]).any()
        assert not prepared.x[:, 5:7].any()
        _, attr = explicit_self_edges(prepared, settings(edge_in_dim=5))
        torch.testing.assert_close(attr[-prepared.num_nodes:], raw.node_self_edge_attr)
    options = [build_action_dataset_kwargs(dict(FEATURE_OPTIONS, **settings(name)),
                 train=False, diagnostic_label_dir=None) for name in ARCHITECTURES]
    assert all(option == options[0] for option in options)
    assert any(path.name == "self_loops.py" for path in preparation_code_paths())


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_refit_inherits_architecture_and_policy(sources, name, tmp_path):
    create, _ = sources
    directory, args, _ = create("pass_intent/source")
    args.update(architecture_identity(settings(name)), model=name)
    (directory / "args.json").write_text(json.dumps(args))
    payload = prepare_refit("pass_intent/source", "new")
    path = tmp_path / "refit.json"
    path.write_text(json.dumps(payload))
    restored = load_refit_config(path)
    assert architecture_identity(restored) == architecture_identity(args)


@pytest.mark.parametrize("name", ARCHITECTURES)
def test_real_cpu_training_and_resume(synthetic_training, monkeypatch, name):
    run, _, _, module, root = synthetic_training
    def interrupted(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("architecture interruption")
        return REAL_RUN_EPOCH(args, *a, **kw)
    monkeypatch.setattr(module, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="architecture interruption"):
        run("architecture", ["--model", name, "--gnn-architecture", name, "--max-epochs", "2"])
    monkeypatch.setattr(module, "run_epoch", REAL_RUN_EPOCH)
    checkpoint = run("architecture", resume=True)
    assert checkpoint["finished"]
    for record in (checkpoint["args"], checkpoint["metadata"]):
        assert record["gnn_architecture"] == name
        assert record["self_loop_policy"] == "explicit_v1"
    assert checkpoint["metadata"]["last_epoch"] == 2


@pytest.mark.parametrize("target,dependency", [("gat", "relationship"), ("transformer", "relationship-attention")])
def test_auxiliary_models_keep_their_own_architecture_and_masks(artifacts, target, dependency):
    raw = velocity_graph()
    torch.save([raw], artifacts["feature_dir"] / "m.pt")
    auxiliary_args = dict(FEATURE_OPTIONS, **settings(dependency, edge_in_dim=5),
                          v_edge_feature_mode="all", relative_speed_edge_feature_mode="all")
    data = ActionDataset(["m"], **artifacts, edge_in_dim=5, v_edge_feature_mode="no_poss",
                         relative_speed_edge_feature_mode="no_poss", auxiliary_model_args=auxiliary_args)
    target_args = dict(FEATURE_OPTIONS, **settings(target, edge_in_dim=5),
                       v_edge_feature_mode="no_poss", relative_speed_edge_feature_mode="no_poss")
    target_batch = Batch.from_data_list([data[0][0]])
    auxiliary_batch = Batch.from_data_list([data[0][0].auxiliary_graph])
    target_input = utils.adapt_batch_graphs_for_model(target_batch, target_args)
    auxiliary_input = utils.adapt_batch_graphs_for_model(auxiliary_batch, auxiliary_args)
    target_index, target_attrs = explicit_self_edges(target_input, target_args)
    auxiliary_index, auxiliary_attrs = explicit_self_edges(auxiliary_input, auxiliary_args)
    torch.testing.assert_close(target_index, auxiliary_index)
    incident = (target_input.x[target_index[0], 13] == 1) | (target_input.x[target_index[1], 13] == 1)
    assert not target_attrs[incident, 2:].any()
    assert auxiliary_attrs[incident, 2:].any()
    assert torch.isfinite(GNN(target_args)(target_input)).all()
    assert torch.isfinite(GNN(auxiliary_args)(auxiliary_input)).all()
    options = build_action_dataset_kwargs(target_args, train=False, diagnostic_label_dir=None)
    ipw_options = build_ipw_dataset_kwargs(options, auxiliary_args, None,
                     diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert ipw_options["v_edge_feature_mode"] == "all"
    torch.testing.assert_close(raw.x, velocity_graph().x)


def test_identity_and_cache_fingerprints_distinguish_policy(monkeypatch):
    import ipw_preparation as ipw
    from prepared_dataset import _digest
    modern = architecture_identity(settings())
    legacy = architecture_identity({"model": "gat"})
    assert modern != legacy and _digest(modern) != _digest(legacy)
    records = [dict(task="pass_intent", is_complete=True, created_at=str(i), model_id=f"pass_intent/{i}",
                    model_name="gat", feature_signature={}, architecture_identity=identity)
               for i, identity in enumerate((modern, legacy))]
    with patch.object(utils, "iter_model_records", return_value=records):
        with pytest.raises(ValueError, match="explicit model id"):
            utils.resolve_latest_model_id("pass_intent")
    before = ipw.inference_fingerprint()
    original_digest = ipw._file_digest
    for filename in ("architecture.py", "relationship.py", "self_loops.py"):
        with patch.object(ipw, "_file_digest", side_effect=lambda path: "changed" if path.name == filename else original_digest(path)):
            assert ipw.inference_fingerprint() != before


@pytest.mark.parametrize("override", [{"gnn_layers": 0}, {"gnn_heads": 0}, {"node_emb_dim": 7}])
@pytest.mark.parametrize("name", ARCHITECTURES)
def test_invalid_dimensions_fail_clearly(name, override):
    args = settings(name)
    args.update(override)
    with pytest.raises(ValueError, match="positive.*divisible"):
        GNN(args)


def test_missing_velocity_data_fails_clearly():
    g = velocity_graph()
    del g.node_self_edge_attr
    g.x = g.x[:, :4]
    with pytest.raises(ValueError, match="vx/vy"):
        original_self_attributes(g)


def test_legacy_refit_resolves_legacy_policy(sources):
    create, _ = sources
    create("pass_intent/source")
    restored = prepare_refit("pass_intent/source", "new")["args"]
    assert restored["gnn_architecture"] == "gat"
    assert restored["self_loop_policy"] == "legacy"
