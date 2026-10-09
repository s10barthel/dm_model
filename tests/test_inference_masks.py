import argparse
import copy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
from models.gnn import GNN
from models.inference_masks import (
    add_inference_mask_arguments, configure_inference_masks, inference_mask_metadata,
    resolve_inference_mask_configurations, resolve_model_masks, resolve_scoped_masks,
)


def model_args(**overrides):
    values = dict(
        model="gat", task="pass_success", gnn_task="node_binary", node_in_dim=26,
        edge_in_dim=5, node_emb_dim=8, graph_emb_dim=8, gnn_layers=1, gnn_heads=1,
        skip_conn=True, out_dim=1, include_out=False, possessor_aware=True,
        poss_vel_aware=True, poss_rel_vel_aware=True, extend_features=True,
        keeper_aware=True, ball_z_aware=True,
    )
    values.update(overrides)
    return values


def source_batch(width=26, edge_width=5, graphs=1):
    graph_list = []
    for _ in range(graphs):
        x = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) / 7 + 1
        x[:, :3] = 0
        x[:2, 0] = 1
        x[:, 13] = torch.tensor([1, 0, 0])
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        attrs = torch.arange(4 * edge_width, dtype=torch.float32).reshape(4, edge_width) / 10 + 1
        graph = Data(x=x, edge_index=edges, edge_attr=attrs)
        graph.inference_possessor_mask = x[:, 13] == 1
        graph_list.append(graph)
    return Batch.from_data_list(graph_list)


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("architecture", ["gat", "gcn", "gin"])
def test_encoder_decoder_boundaries_match_manually_zeroed_features(skip, architecture):
    args = model_args()
    args.update(skip_conn=skip, model=architecture)
    unmasked = GNN(copy.deepcopy(args)).eval()
    masked = GNN(copy.deepcopy(args)).eval()
    masked.load_state_dict(unmasked.state_dict())
    original_args = copy.deepcopy(masked.args)
    masks = ["pos-node-features", "node.accel", "node.goal_dist", "v-edge-features"]
    metadata = configure_inference_masks({"selected": masked}, masks)
    source = source_batch(graphs=2)
    original = source.clone()
    manual = source.clone()
    manual.x[:, [3, 4, 8, 9]] = 0
    manual.edge_attr[:, 2:4] = 0
    with torch.no_grad():
        torch.testing.assert_close(masked(source), unmasked(manual))
        embeddings, pooled = masked.encoder(source)
        expected_embeddings, expected_pooled = unmasked.encoder(manual)
        torch.testing.assert_close(embeddings, expected_embeddings)
        torch.testing.assert_close(pooled, expected_pooled)
        torch.testing.assert_close(
            masked.decoder(source.x, embeddings, pooled, source.batch),
            unmasked.decoder(manual.x, expected_embeddings, expected_pooled, manual.batch),
        )
    assert masked.args == original_args
    torch.testing.assert_close(source.x, original.x)
    torch.testing.assert_close(source.edge_attr, original.edge_attr)
    torch.testing.assert_close(source.edge_index, original.edge_index)
    assert metadata["models"]["selected"]["status"] == "applied"


def test_possessor_subsets_and_composition_use_original_identity():
    runtime = resolve_model_masks(model_args(), ["possessor-aware", "poss-vel-aware", "mask-possessor-v-edge-features"])
    source = source_batch(graphs=2)
    result = runtime.mask_nodes(source.x, possessor_mask=source.inference_possessor_mask)
    expected = source.x.clone()
    expected[:, 13:19] = 0
    expected[source.inference_possessor_mask, 5:9] = 0
    torch.testing.assert_close(result, expected)
    result_edges = runtime.mask_edges(source.edge_attr, source.edge_index, source.x, possessor_mask=source.inference_possessor_mask)
    expected_edges = source.edge_attr.clone()
    incident = source.inference_possessor_mask[source.edge_index[0]] | source.inference_possessor_mask[source.edge_index[1]]
    expected_edges[incident, 2:4] = 0
    torch.testing.assert_close(result_edges, expected_edges)
    assert torch.equal(result_edges[~incident], source.edge_attr[~incident])


def test_sidecar_supports_identity_removed_during_checkpoint_preparation():
    args = model_args()
    args.update(possessor_aware=False)
    model = GNN(args).eval()
    runtime = resolve_model_masks(args, ["poss-vel-aware", "mask-possessor-relative-speed-edge-features"])
    source = source_batch()
    source.x[:, 13:19] = 0
    result = runtime.mask_nodes(source.x, possessor_mask=source.inference_possessor_mask)
    assert torch.count_nonzero(result[0, 5:9]) == 0
    torch.testing.assert_close(result[1:, 5:9], source.x[1:, 5:9])
    configure_inference_masks({"pass_success": model}, ["poss-vel-aware"])
    with torch.no_grad():
        model(source)
    with pytest.raises(ValueError, match="original possessor identity"):
        runtime.mask_nodes(source.x)


def test_empty_masks_reproduce_predictions_and_clear_runtime():
    model = GNN(model_args()).eval()
    source = source_batch()
    with torch.no_grad():
        expected = model(source)
        configure_inference_masks({"pass_success": model}, [])
        torch.testing.assert_close(model(source), expected)
        configure_inference_masks({"pass_success": model}, ["pos-node-features"])
        configure_inference_masks({"pass_success": model}, [])
        torch.testing.assert_close(model(source), expected)
    assert inference_mask_metadata(model)["status"] == "unmasked"
    assert configure_inference_masks({"mock": object()}, [])["models"]["mock"]["status"] == "unmasked"


@pytest.mark.parametrize("task,node_width,expected_offside", [
    ("pass_success", 20, 19), ("pass_success", 26, 25),
    ("success_intent", 24, 23), ("failure_receiver", 27, 25),
])
def test_optional_schema_offsets(task, node_width, expected_offside):
    args = model_args()
    args.update(task=task, node_in_dim=node_width)
    runtime = resolve_model_masks(args, ["offside"])
    assert runtime.operations[0].column == expected_offside


def test_lane_survival_and_failure_slots_follow_append_order():
    args = model_args()
    args.update(task="failure_receiver", lane_survival=True, node_in_dim=28)
    runtime = resolve_model_masks(args, ["offside", "lane-survival", "node.intended_target"])
    mapping = {op.feature: op.column for op in runtime.operations}
    assert mapping == {"node.offside": 25, "node.lane_survival": 26, "node.intended_target": 27}
    args.update(task="success_intent", node_in_dim=25)
    trajectory = resolve_model_masks(args, ["node.pass_mean_speed", "node.pass_direction_z", "offside", "lane-survival"])
    assert {op.feature: op.column for op in trajectory.operations} == {
        "node.pass_mean_speed": 19, "node.pass_direction_z": 22,
        "node.offside": 23, "node.lane_survival": 24,
    }


@pytest.mark.parametrize("task,width,lane,expected", [
    ("pass_success", 19, False, {"node.accel": 8}),
    ("pass_success", 25, False, {"node.potential_blockers": 24}),
    ("success_intent", 23, False, {"node.pass_direction_z": 22}),
    ("success_intent", 24, True, {"node.pass_direction_z": 22, "node.lane_survival": 23}),
    ("failure_receiver", 20, False, {"node.intended_target": 19}),
    ("failure_receiver", 22, True, {"node.offside": 19, "node.lane_survival": 20, "node.intended_target": 21}),
])
def test_legacy_schema_without_offside_or_extended_layout(task, width, lane, expected):
    args = model_args(task=task, node_in_dim=width, lane_survival=lane)
    resolved = resolve_model_masks(args, list(expected))
    assert {op.feature: op.column for op in resolved.operations} == expected


def test_effectiveness_matches_training_defaults_and_xy_only_ranges():
    args = model_args()
    for option in ("possessor_aware", "poss_vel_aware", "poss_rel_vel_aware", "extend_features", "keeper_aware", "ball_z_aware"):
        args.pop(option)
    masks = ["keeper-aware", "ball-z-aware", "possessor-aware", "extend-features"]
    assert all(op.status == "no additional effect" for op in resolve_model_masks(args, masks).operations)
    assert all(op.status == "applied" for op in resolve_model_masks(args, ["pos-node-features", "vel-node-features", "goal-features"]).operations)
    xy = model_args(xy_only=True)
    runtime = resolve_model_masks(xy, ["vel-node-features", "goal-features", "possessor-aware"])
    statuses = {op.feature: op.status for op in runtime.operations}
    assert statuses["node.vx"] == statuses["node.vy"] == "applied"
    assert all(status == "no additional effect" for feature, status in statuses.items() if feature not in ("node.vx", "node.vy"))
    pass_lane = model_args(extend_features=False, pass_lane_features=True)
    runtime = resolve_model_masks(pass_lane, ["extend-features"])
    active = {op.feature for op in runtime.operations if op.status == "applied"}
    assert active == {"node.nearest_opponent_to_pass", "node.potential_interceptors"}


@pytest.mark.parametrize("args_update,masks", [
    ({"node_in_dim": 19}, ["offside"]),
    ({"node_in_dim": 21}, ["offside"]),
    ({"task": "success_intent", "node_in_dim": 24}, ["extend-features"]),
    ({"edge_in_dim": 2}, ["v-edge-features"]),
    ({}, ["lane-survival"]),
])
def test_absent_and_ambiguous_features_fail(args_update, masks):
    args = model_args()
    args.update(args_update)
    with pytest.raises(ValueError, match="absent or ambiguous"):
        resolve_model_masks(args, masks)


def test_disabled_and_unused_inputs_have_explicit_status():
    args = model_args()
    args.update(model="gcn", accel_aware=False, pos_node_features_aware=False, poss_vel_aware=False)
    model = GNN(args).eval()
    metadata = configure_inference_masks({"pass_success": model}, ["accel", "pos-node-features", "v-edge-features", "poss-vel-aware"])
    entries = metadata["models"]["pass_success"]["resolved"]
    assert all(entry["status"] == "no additional effect" for entry in entries)
    assert all(entry["reason"] for entry in entries)


def test_scope_resolution_and_atomic_validation():
    assert resolve_scoped_masks(["accel"], ["pass_success"]) == {"pass_success": ["accel"]}
    resolved = resolve_scoped_masks([["all:accel"], ["pass_success:pos-node-features"]], ["pass_intent", "pass_success"])
    assert resolved == {"pass_intent": ["accel"], "pass_success": ["accel", "pos-node-features"]}
    with pytest.raises(ValueError, match="Unscoped"):
        resolve_scoped_masks(["accel"], ["pass_success", "pass_intent"])
    with pytest.raises(ValueError, match="no participating model"):
        resolve_scoped_masks(["pass_height:accel"], ["pass_success"])
    with pytest.raises(ValueError, match="Unknown inference mask"):
        resolve_scoped_masks(["vel-pos-features"], ["pass_success"])
    first = GNN(model_args()).eval()
    narrow = model_args()
    narrow.update(node_in_dim=19)
    second = GNN(narrow).eval()
    with pytest.raises(ValueError, match="absent"):
        configure_inference_masks({"first": first, "second": second}, ["offside"])
    assert getattr(first, "inference_masks", None) is None


def test_overlap_normalization_and_motion_expansion():
    velocity = resolve_model_masks(model_args(), ["vel-node-features", "accel", "poss-vel-aware"])
    assert len(velocity.operations) == 4
    assert all(operation.subset == "all" for operation in velocity.operations)
    motion = resolve_model_masks(model_args(), ["motion"])
    assert {op.feature for op in motion.operations} == {
        "node.vx", "node.vy", "node.speed", "node.accel", "node.poss_vangle_cos",
        "node.poss_vangle_sin", "edge.velocity_angle_cos", "edge.velocity_angle_sin", "edge.relative_speed",
    }


def test_configuration_identifiers_normalize_equivalent_masks():
    model = GNN(model_args()).eval()
    first = configure_inference_masks({"success": model}, ["vel-node-features"])
    second = configure_inference_masks({"success": model}, ["accel", "vel-node-features"])
    assert first["configuration_id"] == second["configuration_id"]
    third = configure_inference_masks({"success": model}, ["pos-node-features"])
    assert third["configuration_id"] != first["configuration_id"]
    edge = configure_inference_masks({"success": model}, ["v-edge-features-no-poss"])
    alias = configure_inference_masks({"success": model}, ["mask-possessor-v-edge-features"])
    assert edge["configuration_id"] == alias["configuration_id"]


def test_runtime_masks_reject_training():
    model = GNN(model_args())
    configure_inference_masks({"pass_success": model}, ["accel"])
    with pytest.raises(ValueError, match="during training"):
        model(source_batch())
    model.eval()
    with torch.no_grad():
        model(source_batch())
    model.train()
    with pytest.raises(ValueError, match="during training"):
        model(source_batch())


def test_grid_retains_destination_inputs_and_masks_skip_features():
    args = model_args()
    args.update(task="dest_receiver", gnn_task="node_selection")
    unmasked = GNN(copy.deepcopy(args)).eval()
    masked = GNN(copy.deepcopy(args)).eval()
    masked.load_state_dict(unmasked.state_dict())
    configure_inference_masks({"dest_receiver": masked}, ["poss-vel-aware", "pos-node-features"])
    source = source_batch()
    manual = source.clone()
    manual.x[:, 3:5] = 0
    manual.x[manual.inference_possessor_mask, 5:9] = 0
    grid = torch.tensor([[12., 17.], [45., 30.]])
    original_grid = grid.clone()
    with torch.no_grad():
        torch.testing.assert_close(masked.forward_grid(source, grid), unmasked.forward_grid(manual, grid))
    torch.testing.assert_close(grid, original_grid)


def test_physical_residual_control_uses_original_possessor_distances():
    args = model_args()
    args.update(use_physical_xpass=True, physical_xpass_model="gat_phys_logit_offset", physical_variant="gat_phys_logit_offset",
                short_residual_clip_value=1., long_residual_clip_value=3., residual_distance_threshold=20.)
    model = GNN(args).eval()
    # Use the instantiated decoder directly to isolate residual clipping from learned features.
    model.decoder.use_physical_xpass = True
    model.decoder.physical_variant = "gat_phys_logit_offset"
    model.decoder.short_residual_clip_value = 1.
    model.decoder.long_residual_clip_value = 3.
    model.decoder.residual_distance_threshold = 20.
    if not hasattr(model.decoder, "physical_beta0"):
        model.decoder.physical_beta0 = torch.nn.Parameter(torch.tensor(0.))
        model.decoder.physical_beta1 = torch.nn.Parameter(torch.tensor(1.))
    for layer in model.decoder.nodewise_mlp:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    model.decoder.nodewise_mlp[-1].bias.data.fill_(2.)
    configure_inference_masks({"pass_success": model}, ["poss-geometry"])
    source = source_batch()
    source.x[:, config.NODE_FEATURE_POSS_DIST] = torch.tensor([0., 10., 50.])
    logit = torch.tensor([-.2, .3, 1.])
    original = source.x.clone()
    with torch.no_grad():
        output = model.decoder(source.x, torch.zeros(3, 8), physical_xpass_logit=logit)
    expected_delta = torch.tensor([1., 1., 3.]) * torch.tanh(torch.tensor(2.) / torch.tensor([1., 1., 3.]))
    torch.testing.assert_close(output, logit + expected_delta)
    torch.testing.assert_close(source.x, original)


def test_cli_numbered_configs_and_early_listing(capsys):
    argv = ["--inference-mask-2", "accel", "pos-node-features", "--inference-mask-1", "vel-node-features"]
    parser = argparse.ArgumentParser()
    add_inference_mask_arguments(parser, numbered=True, argv=argv)
    configurations = resolve_inference_mask_configurations(parser.parse_args(argv))
    assert configurations == [{"id": "mask_1", "masks": ["vel-node-features"]}, {"id": "mask_2", "masks": ["accel", "pos-node-features"]}]
    with pytest.raises(ValueError, match="once"):
        resolve_inference_mask_configurations(parser.parse_args(["--inference-mask-1", "accel", "--inference-mask-1", "offside"]))
    with pytest.raises(ValueError, match="mix"):
        resolve_inference_mask_configurations(parser.parse_args(["--inference-mask", "accel", "--inference-mask-1", "offside"]))
    assert resolve_inference_mask_configurations(parser.parse_args(["--inference-mask", "accel", "--inference-mask", "offside"])) == [{"id": "mask", "masks": ["accel", "offside"]}]
    parser.add_argument("--required", required=True)
    with pytest.raises(SystemExit) as result:
        parser.parse_args(["--list-inference-masks"])
    assert result.value.code == 0
    assert "vel-node-features: node.vx" in capsys.readouterr().out


@pytest.mark.parametrize("route", ["ordinary", "posterior", "grid"])
def test_real_inference_routes_switch_to_eval_and_preserve_original_roles(monkeypatch, route):
    import inference

    task, node_width, gnn_task = {
        "ordinary": ("pass_success", 26, "node_binary"),
        "posterior": ("failure_receiver", 27, "node_selection"),
        "grid": ("dest_receiver", 26, "node_selection"),
    }[route]
    # Prepared checkpoint inputs have erased the possessor indicator. Masking
    # only the original possessor's velocity therefore requires the sidecar.
    args = model_args(
        task=task, node_in_dim=node_width, gnn_task=gnn_task,
        possessor_aware=False, extend_features=False, xy_only=False,
        sparsify="none", offside_aware=False,
    )
    model = GNN(args)  # Match load_model's train-mode default.
    configure_inference_masks({task: model}, ["poss-vel-aware"])
    source = source_batch().to_data_list()[0]
    source.node_ids = ["home_1", "home_2", "away_1"]
    del source.inference_possessor_mask
    original = source.clone()
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    for field, value in {"action_index": 7, "is_pass": 1, "intent_index": 1, "receiver_index": 2, "success": 1}.items():
        labels[0, config.LABEL_INDEX[field]] = value
    source_graphs = [source]
    actions = pd.DataFrame({"frame_id": [10], "object_id": ["home_1"]}, index=[7])
    if route == "posterior":
        # Posterior output covers defenders from both teams over the match.
        # Include one attacking action by each side, as a real match does.
        away_source = source.clone()
        away_source.node_ids = ["away_1", "home_1", "home_2"]
        away_source.x[:, config.NODE_FEATURE_IS_TEAMMATE] = torch.tensor([1., 0., 0.])
        source_graphs.append(away_source)
        labels = labels.repeat(2, 1)
        labels[1, config.LABEL_INDEX["action_index"]] = 8
        actions.loc[8] = [10, "away_1"]
    # A minimal in-memory match exercises real filtering and prediction without
    # filesystem feature or physical caches.
    match = SimpleNamespace(
        labels=labels,
        actions=actions,
        tracking=pd.DataFrame({"home_1_x": [10.], "home_2_x": [20.], "away_1_x": [30.]}, index=[10]),
    )
    monkeypatch.setattr(inference, "resolve_match_graphs", lambda *args, **kwargs: (source_graphs, None))
    learned_inputs = []
    handle = model.encoder.node_in_fc.register_forward_pre_hook(lambda _module, inputs: learned_inputs.append(inputs[0].clone()))
    try:
        if route == "ordinary":
            probabilities, second_case = inference.inference_gnn(match, model, device="cpu")
            assert second_case is None
            assert list(probabilities.columns) == ["home_1", "home_2"]
            assert not probabilities.isna().any().any()
        elif route == "posterior":
            probabilities = inference.inference_gnn_posterior(match, model, device="cpu", melt=False)
            assert set(probabilities["option"]) == {"home_1", "home_2", "away_1"}
            assert not probabilities.loc[probabilities["option"].str.startswith("home"), "away_1"].isna().any()
            assert not probabilities.loc[probabilities["option"] == "away_1", ["home_1", "home_2"]].isna().any().any()
        else:
            monkeypatch.setattr(inference, "FIELD_SIZE", (3., 2.))
            receive, success = inference.inference_gnn_grid(match, model, device="cpu")
            assert receive[7].shape == (2, 3, 3)
            assert success[7].shape == (2, 3)
            assert torch.isfinite(receive[7]).all()
    finally:
        handle.remove()
    assert model.training is False
    assert learned_inputs
    inputs = learned_inputs[0]
    assert torch.count_nonzero(inputs[::3, 5:9]) == 0
    torch.testing.assert_close(inputs[1, 5:9], original.x[1, 5:9])
    assert torch.count_nonzero(inputs[:, config.NODE_FEATURE_IS_POSSESSOR]) == 0
    torch.testing.assert_close(source.x, original.x)
    torch.testing.assert_close(source.edge_attr, original.edge_attr)
