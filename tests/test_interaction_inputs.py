"""Regression coverage for semantic interaction inputs and optional physical scaling."""
import argparse
import itertools
import json
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch

from datatools import config
from datatools.utils import filter_features_and_labels
from dataset import ActionDataset
from models.architecture import ARCHITECTURES, normalize_architecture
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.edge_schema import (BASE_FEATURES, CANONICAL_FEATURES, EDGE_GROUPS, INTERACTION_GROUPS,
    add_interaction_arguments, edge_names, generation_edge_names, group_mode, interaction_values,
    mask_edge_inputs, normalize_input_settings, project_graph_edges, schema_for_names)
from models.input_scaling import scale_nodes, scale_edges, scale_destinations
from models.gnn import GNN
from models.inference_masks import configure_inference_masks
from models.self_loops import explicit_self_edges
from models import utils
from prepared_dataset import PreparedActionDataset, preparation_code_paths
from scripts import generate_relevant_features as generation, train_relevant_models as wrapper
from test_benchmark_no_accel import make_model_args, make_pass_labels, write_checkpoint
from test_goal_poss_dist import graph, artifacts, FEATURE_OPTIONS
from test_training_controls import synthetic_training
from test_training_refit import sources
from training_refit import prepare_refit
from training_state import CHECKPOINT_NAME


def rich_graph(legacy=False):
    g = graph()
    g.x[:, 5:7] = torch.tensor([[2., 1.], [99., 99.], [-2., 0.], [99., 99.], [3., -1.], [0., 0.]])
    if legacy:
        g.edge_attr = torch.cat([g.edge_attr, torch.ones(g.num_edges, 3)], -1)
    return generation.append_interactions_to_graph(g)


def settings(**overrides):
    args = dict(make_model_args(), **FEATURE_OPTIONS)
    args.update(node_in_dim=25, goal_poss_dist=True, skip_conn=True)
    args.update(schema_for_names(generation_edge_names()))
    args.update({f"{group}_edge_feature_mode": "all" for group in INTERACTION_GROUPS})
    args.update(overrides)
    return normalize_architecture(normalize_input_settings(args), fresh=True)


def test_formulas_and_distance_derivative():
    x = torch.zeros(4, 19, dtype=torch.float64)
    x[:, 3:5] = torch.tensor([[0., 0.], [3., 4.], [3., 4.], [0., 0.]])
    x[:, 5:7] = torch.tensor([[0., 0.], [-3., -4.], [4., -3.], [1., 2.]])
    index = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 2, 2, 3]])
    values = interaction_values(x, index)
    torch.testing.assert_close(values["position_dx_norm"], x.new_tensor([.6, -.6, .6, 0., 0.]))
    torch.testing.assert_close(values["relative_vx"], x.new_tensor([-3., 3., 4., 7., 1.]))
    torch.testing.assert_close(values["separation_rate"], x.new_tensor([-5., -5., 0., 0., 0.]))
    step = 1e-5
    r = x[1, 3:5] - x[0, 3:5]
    u = x[1, 5:7] - x[0, 5:7]
    assert float(((r + step*u).norm() - r.norm()) / step) == pytest.approx(-5., abs=1e-7)
    x[1, 5:7] *= -1
    assert interaction_values(x, index)["separation_rate"][0] == 5


@pytest.mark.parametrize("modes", itertools.product(("all", "none", "no_poss"), repeat=3))
def test_independent_modes_and_goal_exclusion(modes):
    g = rich_graph(legacy=True)
    args = settings(**schema_for_names(edge_names(g)), v_edge_feature_mode="all", relative_speed_edge_feature_mode="all",
                    goal_edge_movement=False, **dict(zip((f"{group}_edge_feature_mode" for group in INTERACTION_GROUPS), modes)))
    before = g.edge_attr.clone()
    actual = mask_edge_inputs(g.edge_attr, g.edge_index, g.x, args)
    goals = (g.x[g.edge_index[0], 2] == 1) | (g.x[g.edge_index[1], 2] == 1)
    poss = (g.x[g.edge_index[0], 13] == 1) | (g.x[g.edge_index[1], 13] == 1)
    for group, names in EDGE_GROUPS.items():
        cols = [edge_names(g).index(name) for name in names]
        expected = before[:, cols].clone()
        mode = group_mode(args, group)
        if mode == "none": expected[:] = 0
        if mode == "no_poss": expected[poss] = 0
        if group != "position_direction": expected[goals] = 0
        torch.testing.assert_close(actual[:, cols], expected)
    torch.testing.assert_close(actual[:, :2], before[:, :2])
    torch.testing.assert_close(g.edge_attr, before)


def test_no_goals_and_reordered_batched_graphs():
    g = rich_graph()
    g.x[:, 2] = 0
    torch.testing.assert_close(mask_edge_inputs(g.edge_attr, g.edge_index, g.x,
        settings(goal_edge_movement=False)), g.edge_attr)
    g = rich_graph()
    order = torch.tensor([4, 3, 0, 5, 1, 2])
    inverse = order.argsort()
    reordered = g.clone()
    reordered.x = g.x[order]
    reordered.edge_index = inverse[g.edge_index]
    batch = Batch.from_data_list([g, reordered])
    args = settings(goal_edge_movement=False)
    masked = mask_edge_inputs(batch.edge_attr, batch.edge_index, batch.x, args)
    torch.testing.assert_close(masked[:g.num_edges], masked[g.num_edges:])
    index, attr = explicit_self_edges(batch, args)
    loops = index[0] == index[1]
    assert loops.sum() == batch.num_nodes
    assert not attr[loops, 2:].any()


def test_semantic_projection_union_and_ambiguous_width():
    g = rich_graph(legacy=True)
    old = project_graph_edges(g.clone(), CANONICAL_FEATURES[:5])
    assert old.edge_attr.shape[1] == 5
    torch.testing.assert_close(old.edge_attr, g.edge_attr[:, :5])
    union = utils.aggregate_graph_schemas({"new": schema_for_names(generation_edge_names()),
                                         "old": schema_for_names(CANONICAL_FEATURES[:5], explicit=False)})
    assert edge_names(union) == CANONICAL_FEATURES
    assert not utils.is_feature_graph_schema_compatible(schema_for_names(generation_edge_names()),
                                                       schema_for_names(CANONICAL_FEATURES[:5]))
    with pytest.raises(ValueError, match="ambiguous"):
        edge_names({"edge_in_dim": 7})
    with pytest.raises(ValueError, match="upgrade|extend-interaction"):
        project_graph_edges(graph(), generation_edge_names())
    batch = project_graph_edges(Batch.from_data_list([g, g]), generation_edge_names())
    assert edge_names(batch) == generation_edge_names()
    assert edge_names(batch.to_data_list()[0]) == generation_edge_names()


def test_fixed_scaling_optional_layouts_and_destinations():
    x = torch.ones(2, 26) * 110
    args = settings(node_in_dim=26, scaling=True)
    result = scale_nodes(x, args)
    assert result[0, 3] == pytest.approx(110/105)
    assert result[0, 4] == pytest.approx(110/68)
    assert result[0, 5] == 11 and result[0, 9] == pytest.approx(1.1)
    assert result[0, 19] == 10 and result[0, 20] == pytest.approx(1.1)
    assert result[0, 25] == 110
    assert result[0, 15] == 110
    assert torch.equal(x, torch.full_like(x, 110))
    success = scale_nodes(torch.ones(1, 24)*10, settings(task="success_intent", node_in_dim=24, scaling=True))
    assert success[0, 19] == 1 and success[0, 20] == 10
    dest = torch.tensor([[105., 68., 105., -68., 100., 100., 100.]])
    torch.testing.assert_close(scale_destinations(dest, dict(scaling=True, more_dest_features=True)),
                               torch.tensor([[1., 1., 1., -1., 1., 1., 1.]]))
    torch.testing.assert_close(scale_destinations(torch.tensor([[1., .5]]), dict(scaling=True, normalize_dest=True)),
                               torch.tensor([[1., .5]]))
    torch.testing.assert_close(scale_destinations(torch.tensor([[105., 68., 100., 1.]]), dict(scaling=True, polar_dest=True)),
                               torch.ones(1, 4))
    assert scale_nodes(x, dict(args, scaling=False)) is x


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_encoder_decoder_boundaries_and_backward(architecture):
    args = settings(gnn_architecture=architecture, model=architecture, scaling=True, goal_edge_movement=False)
    batch = Batch.from_data_list([rich_graph(), rich_graph()])
    original = batch.clone()
    model = GNN(args)
    received = {}
    hooks = [model.encoder.node_in_fc[0].register_forward_pre_hook(lambda _, ins: received.update(node=ins[0].detach().clone())),
             model.encoder.gnn_layers[0].register_forward_pre_hook(lambda _, ins: received.update(index=ins[1], edge=ins[2].detach().clone())),
             model.decoder.nodewise_mlp[0].register_forward_pre_hook(lambda _, ins: received.update(decoder=ins[0].detach().clone()))]
    output = model(batch)
    output.square().sum().backward()
    for hook in hooks: hook.remove()
    torch.testing.assert_close(received["node"], scale_nodes(batch.x, args))
    torch.testing.assert_close(received["decoder"][:, :25], scale_nodes(batch.x, args))
    index, raw_edges = explicit_self_edges(batch, args)
    torch.testing.assert_close(received["edge"], scale_edges(raw_edges, args))
    assert torch.isfinite(output).all()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())
    torch.testing.assert_close(batch.x, original.x)
    torch.testing.assert_close(batch.edge_attr, original.edge_attr)


def test_dataset_runtime_auxiliary_and_cache(artifacts, tmp_path):
    g = rich_graph(legacy=True)
    torch.save([g], artifacts["feature_dir"] / "m.pt")
    args = settings(goal_edge_movement=False, scaling=True, relative_velocity_edge_feature_mode="no_poss")
    kwargs = build_action_dataset_kwargs(args, train=False, diagnostic_label_dir=None)
    dataset = ActionDataset(["m"], **{**artifacts, **kwargs})
    assert len(dataset) == 1
    actual = dataset[0][0]
    labels = make_pass_labels(intent_index=4)
    filtered, _ = filter_features_and_labels([g.clone()], labels.clone(), args)
    torch.testing.assert_close(actual.edge_attr, filtered[0].edge_attr)
    source = Batch.from_data_list([g])
    target = utils.adapt_batch_graphs_for_model(source, args)
    torch.testing.assert_close(target.edge_attr, actual.edge_attr)
    auxiliary = dict(args, **schema_for_names(CANONICAL_FEATURES[:5]), scaling=False,
                     goal_edge_movement=True, v_edge_feature_mode="all", relative_speed_edge_feature_mode="all",
                     **{f"{group}_edge_feature_mode": "none" for group in INTERACTION_GROUPS})
    aux = utils.adapt_batch_graphs_for_model(source, auxiliary)
    torch.testing.assert_close(aux.edge_attr, g.edge_attr[:, :5])
    ipw = build_ipw_dataset_kwargs(kwargs, auxiliary, None, diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert not ipw["scaling"] and ipw["goal_edge_movement"] and ipw["edge_in_dim"] == 5
    torch.testing.assert_close(source.edge_attr, g.edge_attr)
    first = PreparedActionDataset(["m"], **{**artifacts, **kwargs}, cache_dir=tmp_path/"cache", progress=False)
    second = PreparedActionDataset(["m"], **{**artifacts, **kwargs, "scaling": False}, cache_dir=tmp_path/"cache", progress=False)
    assert first.cache_id != second.cache_id
    torch.testing.assert_close(next(iter(first))[0].x, next(iter(second))[0].x)
    assert any(path.name == "edge_schema.py" for path in preparation_code_paths())
    assert utils.extract_model_feature_signature(args) != utils.extract_model_feature_signature(dict(args, scaling=False))


def test_motion_mask_covers_new_features_without_changing_direction():
    args = settings()
    model = GNN(args).eval()
    configure_inference_masks({"m": model}, ["motion"])
    batch = Batch.from_data_list([rich_graph()])
    seen = []
    hook = model.encoder.gnn_layers[0].register_forward_pre_hook(lambda _, ins: seen.append(ins[2]))
    model(batch)
    hook.remove()
    names = edge_names(args)
    assert not seen[0][:, [names.index(name) for name in ("relative_vx", "relative_vy", "separation_rate")]].any()
    assert seen[0][:, names.index("position_dx_norm")].any()


def test_cli_defaults_forwarding_and_exclusivity():
    parser = argparse.ArgumentParser()
    add_interaction_arguments(parser)
    args = parser.parse_args([])
    assert not args.scaling and args.goal_edge_movement
    assert all(getattr(args, f"{group}_edge_feature_mode") == "all" for group in INTERACTION_GROUPS)
    flags = ["--scaling", "--no-goal-edge-movement", "--relative-velocity-edge-features-no-poss", "--no-separation-rate-edge-features"]
    args = parser.parse_args(flags)
    command = wrapper.append_low_level_feature_flags([], wrapper.resolve_wrapper_feature_flags(args))
    assert all(flag in command for flag in flags)
    with pytest.raises(SystemExit):
        parser.parse_args(["--position-direction-edge-features", "--no-position-direction-edge-features"])
    legacy = normalize_input_settings({})
    assert not legacy["scaling"] and legacy["goal_edge_movement"]
    assert all(legacy[f"{group}_edge_feature_mode"] == "none" for group in INTERACTION_GROUPS)
    for flag in flags[:2]:
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--resume-id", "pass_intent/old", flag])


def test_upgrade_stages_all_files_before_replacement(tmp_path):
    directory = tmp_path / "action_graphs"
    directory.mkdir()
    a, b = directory/"a.pt", directory/"b.pt"
    torch.save([graph(), None], a)
    invalid = graph()
    invalid.x[0, 5] = float("nan")
    torch.save([invalid], b)
    before = a.read_bytes()
    with patch.object(generation, "relative_speed_graph_dirs", return_value=[directory]):
        with pytest.raises(ValueError, match="finite"):
            generation.extend_interaction_edge_features(tmp_path, [])
        assert a.read_bytes() == before
        assert not list(directory.glob("*.tmp"))
        torch.save([graph()], b)
        generation.extend_interaction_edge_features(tmp_path, [])
    upgraded = torch.load(a, weights_only=False)
    assert upgraded[1] is None and edge_names(upgraded[0]) == generation_edge_names()
    torch.testing.assert_close(upgraded[0].x, graph().x)


def test_training_load_resume_and_refit(synthetic_training, sources, monkeypatch, tmp_path):
    run, _, epoch, model_utils, root = synthetic_training
    def interrupted(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("planned interruption")
        return epoch(args, *a, **kw)
    monkeypatch.setattr(model_utils, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="planned interruption"):
        run("interactions", ["--scaling", "--no-goal-edge-movement", "--relative-velocity-edge-features-no-poss", "--max-epochs", "2"])
    monkeypatch.setattr(model_utils, "run_epoch", epoch)
    resumed = run("interactions", resume=True)
    assert resumed["finished"] and resumed["args"]["scaling"]
    directory = root/"saved/pass_intent/interactions"
    with patch.object(model_utils, "get_model_path", return_value=directory):
        loaded = model_utils.load_model("pass_intent/interactions", device="cpu")
        assert loaded.args["scaling"] and loaded.args["relative_velocity_edge_feature_mode"] == "no_poss"
    checkpoint = torch.load(root/"saved/pass_intent/interactions"/CHECKPOINT_NAME, weights_only=False)
    assert checkpoint["args"]["scaling"] and not checkpoint["args"]["goal_edge_movement"]
    assert checkpoint["args"]["relative_velocity_edge_feature_mode"] == "no_poss"
    assert len(checkpoint["args"]["edge_feature_names"]) == 7
    # Source recipes preserve all settings rather than applying fresh defaults.
    create, _ = sources
    directory, saved, _ = create("outcome_scoring/interactions")
    saved.update(settings(scaling=True, goal_edge_movement=False))
    saved.update(task="outcome_scoring", model_id="outcome_scoring/interactions")
    (directory/"args.json").write_text(json.dumps(saved))
    payload = prepare_refit("outcome_scoring/interactions", "refitted")
    assert payload["args"]["scaling"] and not payload["args"]["goal_edge_movement"]
    assert payload["args"]["edge_feature_names"] == saved["edge_feature_names"]


def test_legacy_load_preserves_original_inputs(tmp_path):
    write_checkpoint(tmp_path)
    original = json.loads((tmp_path/"args.json").read_text())
    with patch.object(utils, "get_model_path", return_value=tmp_path):
        loaded = utils.load_model("action_intent/legacy", device="cpu").eval()
    assert not loaded.args["scaling"] and loaded.args["goal_edge_movement"]
    assert all(loaded.args[f"{group}_edge_feature_mode"] == "none" for group in INTERACTION_GROUPS)
    reference = GNN(original).eval()
    reference.load_state_dict(loaded.state_dict())
    batch = Batch.from_data_list([graph()])
    torch.testing.assert_close(loaded(batch), reference(batch), atol=0, rtol=0)


def test_extension_plan_and_legacy_schema_inheritance():
    from test_feature_run_extension import FeatureRunExtensionPlanTests, make_args
    args = make_args()
    args.extend_interaction_edge_features = True
    plan = FeatureRunExtensionPlanTests().build_plan(args)
    assert edge_names(plan.graph_schema) == generation_edge_names()
    assert plan.extend_interaction_edge_features and not plan.command_steps
    # Label-only extensions of older runs explicitly disable fresh feature defaults.
    plan = FeatureRunExtensionPlanTests().build_plan(make_args(["next_5"]))
    for step in plan.command_steps:
        assert "--no-position-direction-edge-features" in step.command
        assert "--no-relative-velocity-edge-features" in step.command
        assert "--no-separation-rate-edge-features" in step.command


@pytest.mark.parametrize("enabled", itertools.product((False, True), repeat=3))
def test_generation_optional_layouts_and_upgrade_equivalence(enabled):
    from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
    from test_graph_feature_regressions import make_offside_match_and_snapshot
    match, snapshot = make_offside_match_and_snapshot(include_goals=False)
    match.tracking = snapshot
    match.max_players = 6
    kwargs = dict(frame=10, possessor="home_1", period_tracking=snapshot,
                  feature_dim=infer_node_feature_dim(extend=False), extend=False)
    options = {f"add_{group}_edge_features": flag for group, flag in zip(INTERACTION_GROUPS, enabled)}
    fresh = construct_graph_for_frame(match, **kwargs, **options)
    assert edge_names(fresh) == generation_edge_names(False, False, *enabled)
    assert fresh.edge_attr.shape[1] == 2 + 2*enabled[0] + 2*enabled[1] + enabled[2]
    upgraded = generation.append_interactions_to_graph(project_graph_edges(fresh.clone(), BASE_FEATURES))
    complete = construct_graph_for_frame(match, **kwargs)
    torch.testing.assert_close(upgraded.edge_attr, complete.edge_attr)
    assert edge_names(upgraded) == edge_names(complete)
    # Even three/four/five-column new layouts use their saved semantics.
    args = settings(node_in_dim=fresh.x.shape[1], scaling=True, **schema_for_names(edge_names(fresh)),
                    **{f"{group}_edge_feature_mode": "all" if flag else "none" for group, flag in zip(INTERACTION_GROUPS, enabled)})
    assert torch.isfinite(GNN(args)(Batch.from_data_list([fresh]))).all()


def test_goal_velocities_are_stationary_in_new_features():
    g = rich_graph()
    index = ((g.edge_index[0] == 0) & (g.edge_index[1] == 1)).nonzero().item()
    names = edge_names(g)
    assert g.edge_attr[index, names.index("relative_vx")] == -2
    assert g.edge_attr[index, names.index("relative_vy")] == -1
    assert g.edge_attr[index, names.index("separation_rate")] == 2


def test_grid_destinations_and_physical_residuals_keep_raw_units():
    args = settings(task="pass_dest", gnn_task="graph_binary", more_dest_features=True,
                    scaling=True, graph_emb_dim=8)
    model = GNN(args).eval()
    batch = Batch.from_data_list([rich_graph(), rich_graph()])
    grid = torch.tensor([[105., 68., 105., -68., 100., 100., 100.],
                         [0., 0., 0., 0., 0., 0., 0.]])
    raw = grid.clone()
    seen = []
    hook = model.decoder.graphwise_mlp[0].register_forward_pre_hook(lambda _, ins: seen.append(ins[0]))
    model.forward_grid(batch, grid)
    hook.remove()
    torch.testing.assert_close(seen[0][:, -7:], scale_destinations(grid, args).repeat(2, 1))
    torch.testing.assert_close(grid, raw)
    from models.gnn import Decoder
    decoder = Decoder(settings(task="pass_success", gnn_task="node_binary", out_dim=1, scaling=True,
                               use_physical_xpass=True, model_variant="gat_phys_logit_offset",
                               residual_distance_threshold=30., short_residual_clip_value=1.,
                               long_residual_clip_value=3.))
    nodes = rich_graph().x
    nodes[0, 14], nodes[1, 14] = 10., 60.
    logits = torch.full((6,), .7)
    captured = []
    original_clip = decoder._clip_delta_gat
    def capture(delta, raw_nodes):
        captured.append(raw_nodes.clone())
        return original_clip(delta, raw_nodes)
    decoder._clip_delta_gat = capture
    decoder(nodes, torch.zeros(6, 8), batch_indices=torch.zeros(6, dtype=torch.long), physical_xpass_logit=logits)
    torch.testing.assert_close(captured[0], nodes)
    assert logits.eq(.7).all()
    clipped = original_clip(torch.ones(6)*2, nodes)
    assert clipped[0] < 1 and clipped[1] > 1


def test_dataset_auxiliary_preserves_its_own_edge_settings(artifacts):
    raw = rich_graph(legacy=True)
    torch.save([raw], artifacts["feature_dir"] / "m.pt")
    target = settings(goal_edge_movement=False)
    auxiliary = dict(target, **schema_for_names(CANONICAL_FEATURES[:5]), goal_edge_movement=True,
                     v_edge_feature_mode="all", relative_speed_edge_feature_mode="all",
                     **{f"{group}_edge_feature_mode": "none" for group in INTERACTION_GROUPS})
    kwargs = build_action_dataset_kwargs(target, train=False, diagnostic_label_dir=None)
    kwargs["auxiliary_models_args"] = {"intent": auxiliary}
    dataset = ActionDataset(["m"], **{**artifacts, **kwargs})
    prepared = dataset[0][0]
    aux_graph = prepared.xpass_intent_graph
    torch.testing.assert_close(aux_graph.edge_attr, raw.edge_attr[:, :5])
    assert prepared.edge_attr.shape[1] == 7 and aux_graph.edge_attr.shape[1] == 5


def test_upgrade_includes_available_carry_families(tmp_path):
    from project_config import get_action_graph_dir
    normal = get_action_graph_dir(tmp_path)
    carry = get_action_graph_dir(tmp_path, use_carries=True)
    normal.mkdir()
    carry.mkdir()
    assert generation.relative_speed_graph_dirs(tmp_path, []) == [normal, carry]


def test_relative_speed_extension_accepts_explicit_legacy_names():
    from test_feature_run_extension import FeatureRunExtensionPlanTests, make_args, make_metadata
    args = make_args(extend_relative_speed_edge_features=True)
    plan = FeatureRunExtensionPlanTests().build_plan(args, make_metadata(graph_schema=schema_for_names(CANONICAL_FEATURES[:4])))
    assert edge_names(plan.graph_schema) == CANONICAL_FEATURES[:5]
    args.extend_interaction_edge_features = True
    plan = FeatureRunExtensionPlanTests().build_plan(args, make_metadata(graph_schema=schema_for_names(CANONICAL_FEATURES[:4])))
    assert edge_names(plan.graph_schema) == CANONICAL_FEATURES


def test_direct_model_projects_shared_source_without_mutation():
    raw = Batch.from_data_list([rich_graph(legacy=True), rich_graph(legacy=True)])
    saved = raw.clone()
    args = settings(scaling=True, relative_velocity_edge_feature_mode="no_poss", goal_edge_movement=False)
    model = GNN(args).eval()
    projected = utils.adapt_batch_graphs_for_model(raw, args)
    torch.testing.assert_close(model(raw), model(projected))
    torch.testing.assert_close(raw.edge_attr, saved.edge_attr)
    assert edge_names(raw) == CANONICAL_FEATURES


@pytest.mark.parametrize("column", [3, 4, 5, 6])
def test_upgrade_rejects_nonfinite_raw_goal_inputs(column):
    raw = graph()
    raw.x[1, column] = float("nan")
    assert raw.x[1, config.NODE_FEATURE_IS_GOAL] == 1
    with pytest.raises(ValueError, match="finite"):
        generation.append_interactions_to_graph(raw)


def test_carry_extension_can_also_upgrade_interactions(tmp_path):
    from test_feature_run_extension import make_args, make_metadata
    from project_config import get_action_graph_dir
    normal = get_action_graph_dir(tmp_path)
    carry = get_action_graph_dir(tmp_path, use_carries=True)
    normal.mkdir()
    carry.mkdir()
    for directory in (normal, carry):
        torch.save([graph(), None], directory / "m.pt")
    args = make_args(run_id=None, in_place=True)
    args.extend_interaction_edge_features = True
    metadata = make_metadata(graph_schema=schema_for_names(BASE_FEATURES))
    written = []
    with patch.object(generation, "resolve_feature_run_id", return_value="base"), \
         patch.object(generation, "load_feature_run_metadata", return_value=metadata), \
         patch.object(generation, "get_feature_run_root", return_value=tmp_path), \
         patch.object(generation, "run_generation_steps"), \
         patch.object(generation, "write_run_metadata", side_effect=lambda _, value: written.append(dict(value))):
        generation.run_carry_extension_generation(args)
    assert edge_names(written[-1]["graph_schema"]) == generation_edge_names()
    assert written[-1]["extension_history"][-1]["interaction_edge_features"]
    for directory in (normal, carry):
        graphs = torch.load(directory / "m.pt", weights_only=False)
        assert edge_names(graphs[0]) == generation_edge_names() and graphs[1] is None


@pytest.mark.parametrize("legacy_width", [2, 4, 5])
def test_evaluation_accepts_legacy_checkpoint_on_explicit_superset(legacy_width):
    from test import validate_checkpoint_edge_schema
    from models.edge_feature_config import normalize_v_edge_feature_args
    saved = normalize_v_edge_feature_args({"edge_in_dim": legacy_width})
    resolved = validate_checkpoint_edge_schema(schema_for_names(CANONICAL_FEATURES), saved)
    assert len(edge_names(resolved)) == legacy_width


def test_evaluation_uses_saved_interaction_schema_and_rejects_missing_features():
    from test import validate_checkpoint_edge_schema
    saved = settings(relative_velocity_edge_feature_mode="no_poss")
    resolved = validate_checkpoint_edge_schema(schema_for_names(CANONICAL_FEATURES), saved)
    assert edge_names(resolved) == generation_edge_names()
    with pytest.raises(ValueError, match="missing|requested"):
        validate_checkpoint_edge_schema(schema_for_names(BASE_FEATURES), saved)
    with pytest.raises(ValueError, match="reconstruct"):
        validate_checkpoint_edge_schema(schema_for_names(CANONICAL_FEATURES),
                                        dict(saved, position_direction_edge_feature_mode="none"))



def test_upgrade_preserves_saved_self_attributes():
    raw = project_graph_edges(rich_graph(legacy=True), CANONICAL_FEATURES[:5])
    raw.node_self_edge_attr = raw.edge_attr.new_zeros((raw.num_nodes, 5))
    raw.node_self_edge_attr[:, 1:4] = torch.tensor([1., .2, .8])
    saved = raw.node_self_edge_attr.clone()
    upgraded = generation.append_interactions_to_graph(raw)
    torch.testing.assert_close(upgraded.node_self_edge_attr[:, :5], saved)
    assert not upgraded.node_self_edge_attr[:, 5:].any()
