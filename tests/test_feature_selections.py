"""Behavioral coverage for retained-input selections and legacy compatibility."""
import argparse
import copy
import json

import pytest
import torch
from torch_geometric.data import Batch, Data

from dataset import ActionDataset
from datatools.utils import filter_features_and_labels
from models.feature_masks import (
    NODE_MODES, add_feature_selection_arguments, feature_signature, mask_node_inputs,
    normalize_feature_settings,
)
from models.gnn import GNN
from models.inference_masks import configure_inference_masks, resolve_model_masks
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.utils import adapt_batch_graphs_for_model, get_model_record, build_dest_features
from prepared_dataset import PreparedActionDataset
from scripts import train_relevant_models as wrapper
from test_benchmark_no_accel import make_labels, write_checkpoint
from test_inference_masks import model_args
from test_training_controls import synthetic_training
from test_training_refit import sources


def graph():
    x = torch.arange(6 * 26, dtype=torch.float32).reshape(6, 26) + 1
    x[:, :3] = 0
    x[:2, 0] = 1
    x[4, 0] = 1
    x[[1, 3], 1] = 1  # Both goalkeepers are ordinary players for team selections.
    x[4:, 2] = 1
    x[:, 13] = 0
    x[0, 13] = 1
    edges = torch.tensor([(i, j) for i in range(6) for j in range(6) if i != j]).T
    result = Data(x=x, edge_index=edges, edge_attr=torch.ones((edges.shape[1], 5)))
    result.inference_possessor_mask = x[:, 13] == 1
    return result


def options(**kwargs):
    return dict(model_args(), goal_poss_dist=True, **kwargs)


@pytest.mark.parametrize("mode,retained", [
    ("all", [0, 1, 2, 3, 4, 5]), ("attacking", [0, 1]), ("defending", [2, 3]),
    ("no-poss", [1, 2, 3, 4, 5]), ("none", []),
])
@pytest.mark.parametrize("feature,column", [("vx", 5), ("vy", 6), ("speed", 7), ("accel", 8)])
def test_node_modes_retain_exact_values(feature, column, mode, retained):
    source = graph()
    original = source.x.clone()
    result = mask_node_inputs(source.x, options(feature_selections={feature: mode}))
    expected = original.clone()
    expected[:, column] = 0
    expected[retained, column] = original[retained, column]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(source.x, original)


@pytest.mark.parametrize("mode", NODE_MODES)
def test_inference_matches_training_selection_with_reordered_batched_nodes(mode):
    source = graph()
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    source.x = source.x[permutation]
    source.inference_possessor_mask = source.inference_possessor_mask[permutation]
    batch = Batch.from_data_list([source, source.clone()])
    runtime = resolve_model_masks(options(), ["vel-node-features=" + mode])
    actual = runtime.mask_nodes(batch.x, possessor_mask=batch.inference_possessor_mask)
    expected = mask_node_inputs(batch.x, options(feature_selections={"vel-node-features": mode}),
                                possessor_mask=batch.inference_possessor_mask)
    torch.testing.assert_close(actual, expected)


def test_team_selection_tracks_possession_not_home_or_node_order():
    source = graph()
    source.x[:, 0] = 1 - source.x[:, 0]
    source.x[:, 13] = 0
    source.x[2, 13] = 1
    result = mask_node_inputs(source.x, options(feature_selections={"vx": "defending"}))
    assert torch.equal(result[:2, 5], source.x[:2, 5])
    assert torch.count_nonzero(result[2:, 5]) == 0


@pytest.mark.parametrize("names", [
    ["vx", "vx=defending"], ["vx=defending", "vx"],
    ["node.vx", "vx=defending"], ["vx=defending", "node.vx"],
])
def test_explicit_inference_selector_overrides_shorthand(names):
    actual = resolve_model_masks(options(), names)
    expected = resolve_model_masks(options(), ["vx=defending"])
    assert actual.operations == expected.operations


def test_group_defaults_and_individual_overrides():
    source = graph()
    values = {"vel-node-features": "none", "vx": "defending", "vy": "defending"}
    actual = mask_node_inputs(source.x, options(feature_selections=values))
    assert torch.count_nonzero(actual[:, 5:9]) == 4
    torch.testing.assert_close(actual[2:4, 5:7], source.x[2:4, 5:7])
    runtime = resolve_model_masks(options(), ["vx=defending", "vel-node-features", "vy=defending"])
    torch.testing.assert_close(runtime.mask_nodes(source.x), actual)
    with pytest.raises(ValueError, match="Conflicting explicit"):
        resolve_model_masks(options(), ["vx=attacking", "node.vx=defending"])


def test_inference_only_restricts_checkpoint_inputs():
    source = graph()
    args = options(feature_selections={"vel-node-features": "attacking"})
    trained = mask_node_inputs(source.x, args)
    runtime = resolve_model_masks(args, ["vel-node-features=defending"])
    assert torch.count_nonzero(runtime.mask_nodes(trained)[:, 5:9]) == 0
    disabled = options(vel_node_features_aware=False)
    assert all(op.status == "no additional effect" for op in resolve_model_masks(disabled, ["vx=defending"]).operations)


@pytest.mark.parametrize("architecture", ["gat", "gcn", "gin", "transformer", "relationship"])
@pytest.mark.parametrize("skip", [False, True])
def test_model_encoder_decoder_and_grid_use_same_masks(architecture, skip):
    args = options()
    args.update(model=architecture, skip_conn=skip)
    baseline = GNN(copy.deepcopy(args)).eval()
    selected_args = dict(args, feature_selections={"vel-node-features": "none", "vx": "defending", "vy": "defending",
                                                  "possessor-id": "none"})
    selected = GNN(selected_args).eval()
    selected.load_state_dict(baseline.state_dict())
    batch = Batch.from_data_list([graph(), graph()])
    original = batch.clone()
    manual = batch.clone()
    manual.x = mask_node_inputs(batch.x, selected_args, possessor_mask=batch.inference_possessor_mask)
    with torch.no_grad():
        torch.testing.assert_close(selected(batch), baseline(manual))
        actual_emb, actual_pool = selected.encoder(batch)
        expected_emb, expected_pool = baseline.encoder(manual)
        torch.testing.assert_close(actual_emb, expected_emb)
        torch.testing.assert_close(actual_pool, expected_pool)
        configure_inference_masks({"selected": selected}, ["vx=no-poss"])
        selected(batch)
    torch.testing.assert_close(batch.x, original.x)
    torch.testing.assert_close(batch.edge_attr, original.edge_attr)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--speed", dest="speed_aware", action="store_true", default=True)
    p.add_argument("--lane-survival", dest="lane_survival_mode", nargs="?", const="max", default=None)
    add_feature_selection_arguments(p)
    return p


def test_parser_defaults_aliases_and_conflicts():
    p = parser()
    assert vars(p.parse_args([]))["speed_aware"] is True
    args = p.parse_args(["--vx", "defending", "--vy=attacking", "--speed", "false"])
    assert args.feature_selections == {"vx": "defending", "vy": "attacking", "speed": "none"}
    with pytest.raises(SystemExit):
        p.parse_args(["--vx", "defending", "--vx", "attacking"])
    with pytest.raises(SystemExit):
        p.parse_args(["--velocity-angle-edge-features", "defending"])
    assert p.parse_args(["--lane-survival", "top_10"]).lane_survival_mode == "top_10"
    for flags in (["--lane-survival", "none", "--lane-survival-mode", "top_10"],
                  ["--lane-survival-mode", "top_10", "--lane-survival", "none"]):
        args = normalize_feature_settings(vars(p.parse_args(flags)))
        assert args["lane_survival_mode"] is None


def test_wrapper_forwards_selections_and_preserves_defaults():
    baseline = wrapper.resolve_wrapper_feature_flags(argparse.Namespace())
    assert baseline["vel_node_features_aware"] is True
    args = argparse.Namespace(feature_selections={"vel-node-features": "none", "vx": "defending", "vy": "defending"})
    resolved = wrapper.resolve_wrapper_feature_flags(args)
    command = wrapper.append_low_level_feature_flags([], resolved)
    for name, mode in args.feature_selections.items():
        assert command[command.index("--" + name) + 1] == mode


def test_dataset_runtime_prepared_cache_and_auxiliary_configuration(tmp_path):
    source = graph()
    # No goal nodes for this task; preserve player goalkeepers.
    source.x = source.x[:4]
    source.edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    source.edge_attr = torch.ones((6, 5))
    source.inference_possessor_mask = source.inference_possessor_mask[:4]
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    torch.save([source], features / "match.pt")
    torch.save(make_labels(), labels / "match.pt")
    settings = dict(task="action_intent", xy_only=False, possessor_aware=True, keeper_aware=True,
                    ball_z_aware=True, poss_vel_aware=True, poss_rel_vel_aware=True, goal_poss_dist=True,
                    extend_features=True, sparsify="none", max_edge_dist=10, edge_in_dim=5,
                    v_edge_feature_mode="all", relative_speed_edge_feature_mode="all",
                    feature_selections={"vel-node-features": "none", "vx": "defending", "vy": "defending"})
    eager = ActionDataset(["match"], feature_dir=features, label_dir=labels, **settings)
    runtime, _ = filter_features_and_labels([source], make_labels(), settings)
    torch.testing.assert_close(eager[0][0].x, runtime[0].x)
    adapted = adapt_batch_graphs_for_model(Batch.from_data_list([source]), settings)
    torch.testing.assert_close(eager[0][0].x, adapted.x)
    disk = PreparedActionDataset(["match"], feature_dir=features, label_dir=labels, cache_dir=tmp_path / "cache",
                                 progress=False, **settings)
    torch.testing.assert_close(next(iter(disk))[0].x, eager[0][0].x)
    other = PreparedActionDataset(["match"], feature_dir=features, label_dir=labels, cache_dir=tmp_path / "cache",
                                  progress=False, **dict(settings, feature_selections={"vx": "attacking"}))
    assert disk.cache_id != other.cache_id
    target = build_action_dataset_kwargs(settings, train=False, diagnostic_label_dir=None)
    dependency = build_ipw_dataset_kwargs(target, dict(settings, feature_selections={"vx": "attacking"}), None,
                                          diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert target["feature_selections"]["vx"] == "defending"
    assert dependency["feature_selections"] == {"vx": "attacking"}


def test_real_training_saves_resolved_selections(synthetic_training):
    run, _, _, _, _ = synthetic_training
    checkpoint = run("selected", ["--max-epochs", "1", "--vel-node-features", "none", "--vx", "defending", "--vy", "defending"])
    assert checkpoint["args"]["feature_mask_version"] == 1
    masks = checkpoint["metadata"]["feature_signature"]["feature_masks"]
    assert masks["node"]["node.vx"] == ["defending"]
    assert masks["node"]["node.vy"] == ["defending"]
    assert masks["node"]["node.speed"] == []
    assert masks["node"]["node.accel"] == []


def test_old_and_new_checkpoint_masks_remain_usable(tmp_path, monkeypatch):
    from models import utils
    root = tmp_path / "pass_success" / "fixture"
    write_checkpoint(root)
    monkeypatch.setattr(utils, "get_model_path", lambda _: root)
    old = get_model_record("pass_success/fixture")
    assert resolve_model_masks(old["args"], ["vx=defending"]).operations[0].status == "applied"
    args_path = root / "args.json"
    args = json.loads(args_path.read_text())
    args.update(feature_mask_version=1, feature_selections={"vx": "none"})
    args_path.write_text(json.dumps(args))
    new = get_model_record("pass_success/fixture")
    assert resolve_model_masks(new["args"], ["vx=defending"]).operations[0].status == "no additional effect"
    assert old["feature_signature"]["feature_masks"] != new["feature_signature"]["feature_masks"]


def test_shorthand_and_explicit_metadata_identifiers_match():
    model = GNN(options()).eval()
    bare = configure_inference_masks({"model": model}, ["vel-node-features"])
    explicit = configure_inference_masks({"model": model}, ["vel-node-features=none"])
    assert bare["configuration_id"] == explicit["configuration_id"]
    assert bare["models"]["model"]["canonical"] == explicit["models"]["model"]["canonical"]


def test_missing_features_and_unknown_versions_fail_clearly():
    source = graph()
    with pytest.raises(ValueError, match="node.lane_survival"):
        mask_node_inputs(source.x, options(feature_selections={"lane-survival": "all"}))
    with pytest.raises(ValueError, match="Unsupported feature_mask_version"):
        feature_signature(dict(options(), feature_mask_version=99))


def test_direct_model_preserves_identity_before_training_masks():
    args = options(feature_selections={"possessor-id": "none"})
    model = GNN(args).eval()
    source = graph()
    del source.inference_possessor_mask
    batch = Batch.from_data_list([source])
    configure_inference_masks({"model": model}, ["vel-node-features=no-poss"])
    with torch.no_grad():
        result = model(batch)
    assert torch.isfinite(result).all()
    assert batch.x[0, 13] == 1


def test_goal_only_inference_controls_compose_with_node_and_edge_groups():
    source = graph()
    runtime = resolve_model_masks(options(), ["goal-poss-geometry", "goal-edge-movement", "vel-node-features=no-poss"])
    actual_nodes = runtime.mask_nodes(source.x)
    assert torch.count_nonzero(actual_nodes[4:, 14:17]) == 0
    torch.testing.assert_close(actual_nodes[:4, 14:17], source.x[:4, 14:17])
    actual_edges = runtime.mask_edges(source.edge_attr, source.edge_index, source.x)
    goals = source.x[:, 2] == 1
    incident = goals[source.edge_index[0]] | goals[source.edge_index[1]]
    assert torch.count_nonzero(actual_edges[incident, 2:5]) == 0
    torch.testing.assert_close(actual_edges[:, :2], source.edge_attr[:, :2])
    model = GNN(options()).eval()
    metadata = configure_inference_masks({"model": model}, ["goal-poss-geometry", "goal-edge-movement", "vx=defending"])
    canonical = metadata["models"]["model"]["canonical"]
    replayed = configure_inference_masks({"model": model}, canonical)
    assert metadata["configuration_id"] == replayed["configuration_id"]


def test_legacy_pass_lane_feature_validation_is_preserved():
    source = graph().x[:, :19]
    settings = options()
    settings.update(extend_features=False, pass_lane_features=True)
    with pytest.raises(ValueError, match="Pass-lane features require"):
        mask_node_inputs(source, settings, prepared=False)


def test_grid_decoding_and_geometry_use_original_identity():
    args = options()
    args.update(task="pass_dest", gnn_task="node_selection", skip_conn=True)
    baseline = GNN(copy.deepcopy(args)).eval()
    selected_args = dict(args, feature_selections={"vel-node-features": "defending", "possessor-id": "none"})
    selected = GNN(selected_args).eval()
    selected.load_state_dict(baseline.state_dict())
    source = Batch.from_data_list([graph(), graph()])
    grid = torch.tensor([[15., 15.], [40., 30.]])
    prepared = adapt_batch_graphs_for_model(source, selected_args)
    torch.testing.assert_close(build_dest_features(prepared, grid), build_dest_features(source, grid))
    manual = source.clone()
    manual.x = mask_node_inputs(source.x, selected_args, possessor_mask=source.inference_possessor_mask)
    with torch.no_grad():
        torch.testing.assert_close(selected.forward_grid(prepared, grid), baseline.forward_grid(manual, grid))
    assert prepared.x[0, 13] == 1
    assert mask_node_inputs(prepared.x, selected_args, possessor_mask=prepared.inference_possessor_mask)[0, 13] == 0


def test_source_refit_inherits_selections_without_modifying_source(sources):
    import training_refit
    create, _ = sources
    directory, settings, _ = create("outcome_scoring/selected")
    settings.update(feature_mask_version=1, feature_selections={"vx": "defending", "vy": "attacking"})
    (directory / "args.json").write_text(json.dumps(settings))
    payload = training_refit.prepare_refit("outcome_scoring/selected", "refitted")
    assert payload["args"]["feature_selections"] == settings["feature_selections"]
    assert json.loads((directory / "args.json").read_text()) == settings


def test_training_resume_inherits_selections(synthetic_training, monkeypatch):
    run, _, epoch, utils, _ = synthetic_training
    flags = ["--vel-node-features", "none", "--vx", "defending", "--vy", "defending"]
    baseline = run("selected_full", flags)

    def interrupted(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("interrupted selection training")
        return epoch(args, *a, **kw)

    monkeypatch.setattr(utils, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="interrupted selection training"):
        run("selected_resume", flags)
    monkeypatch.setattr(utils, "run_epoch", epoch)
    resumed = run("selected_resume", resume=True)
    assert resumed["args"]["feature_selections"] == baseline["args"]["feature_selections"]
    for name, value in baseline["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, rtol=0, atol=0)


def test_legacy_xy_only_retains_possessor_vx_vy_when_its_flag_was_erased():
    source = graph()
    settings = options()
    settings.update(xy_only=True, poss_vel_aware=False)
    actual = mask_node_inputs(source.x, settings)
    torch.testing.assert_close(actual[:, 5:7], source.x[:, 5:7])
    assert torch.count_nonzero(actual[:, 7:12]) == 0
    assert torch.count_nonzero(actual[:, 13:19]) == 0
    settings["feature_selections"] = {"vel-node-features": "no-poss"}
    assert torch.count_nonzero(mask_node_inputs(source.x, settings)[0, 5:9]) == 0
