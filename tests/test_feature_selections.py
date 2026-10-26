"""Behavioral coverage for retained-input selections and legacy compatibility."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from dataset import ActionDataset
from datatools.utils import filter_features_and_labels
from models.feature_masks import (
    BOOLEAN_FEATURES, EDGE_MODES, EDGE_OPTIONS, NODE_FEATURES, NODE_MODES,
    add_feature_selection_arguments, feature_signature, mask_node_inputs,
    normalize_feature_settings,
)
from models.edge_schema import group_mode
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


# Portable feature settings from the failed 2026-10-26 bundle command.
REPORTED_SELECTIONS = {
    "accel": "no-poss", "ball-z": "all", "extended-node-features": "none",
    "goal-features": "none", "keeper-id": "all", "lane-survival": "none",
    "offside": "all", "pass-lane-features": "none", "pos-node-features": "all",
    "position-direction-edge-features": "all", "poss-geometry": "all",
    "poss-velocity-angles": "none", "possessor-id": "all",
    "relative-speed-edge-features": "none", "relative-velocity-edge-features": "no-poss",
    "separation-rate-edge-features": "no-poss", "speed": "no-poss",
    "vel-node-features": "no-poss", "velocity-angle-edge-features": "none",
    "vx": "no-poss", "vy": "no-poss",
}
COMPETING_FLAGS = {
    **{name: ("--no-" + name,) for name in ("speed", "accel", "offside", "pass-lane-features", "lane-survival")},
    "velocity-angle-edge-features": ("--v-edge-features", "--no-v-edge-features", "--v-edge-features-no-poss"),
    **{name: ("--no-" + name, "--" + name + "-no-poss") for name in EDGE_OPTIONS
       if name != "velocity-angle-edge-features"},
}


@pytest.fixture(scope="module")
def training_parser():
    """Build the actual train.py parser; stop before artifact access or training."""
    class CapturedParser(Exception):
        pass

    original = argparse.ArgumentParser.parse_known_args

    def capture(parser, args=None, namespace=None):
        if "--task" in parser._option_string_actions:
            raise CapturedParser(parser)
        return original(parser, args, namespace)

    with patch.object(sys, "argv", ["train.py"]), \
         patch.object(argparse.ArgumentParser, "parse_known_args", capture), \
         pytest.raises(CapturedParser) as captured:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="mask_parser_test")
    return captured.value.args[0]


def wrapper_commands(selections=None, **overrides):
    from test_success_intent_mode_independent import make_training_args

    args = make_training_args("feature_run", pass_success_ipw=False,
                              feature_selections=selections, **overrides)
    args.enabled_tasks.update(pass_height=True, failure_receiver=True)
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        return wrapper.build_training_commands(args)[0]


def assert_single_selections(command, selections):
    for name, mode in selections.items():
        flag = "--" + name
        assert command.count(flag) == 1
        assert wrapper.get_cli_value(command, flag) == mode
        assert not set(COMPETING_FLAGS.get(name, ())).intersection(command)


@pytest.mark.parametrize("task", [spec[0] for spec in wrapper.MODEL_TOGGLE_SPECS])
def test_reported_masks_parse_for_every_wrapper_task(training_parser, task):
    commands = wrapper_commands(
        REPORTED_SELECTIONS, use_v_edge_features=False, relative_speed_edge_feature_mode="none",
        relative_velocity_edge_feature_mode="no_poss", separation_rate_edge_feature_mode="no_poss",
        scaling=True, sparsify="delaunay", gnn_architecture="relationship-attention",
    )
    assert len(commands) == 8
    command = next(command for command in commands if wrapper.get_cli_value(command, "--task") == task)
    assert_single_selections(command, REPORTED_SELECTIONS)
    args = normalize_feature_settings(vars(training_parser.parse_args(command)))
    assert args["feature_selections"] == REPORTED_SELECTIONS
    assert args["lane_survival_mode"] is None
    for name, group in EDGE_OPTIONS.items():
        assert group_mode(args, group) == REPORTED_SELECTIONS[name].replace("-", "_")


@pytest.mark.parametrize("name,mode", [
    (name, mode) for name in (*NODE_FEATURES, *EDGE_OPTIONS)
    for mode in (("all", "none") if name in BOOLEAN_FEATURES else EDGE_MODES if name in EDGE_OPTIONS else NODE_MODES)
])
def test_wrapper_selector_values_round_trip(training_parser, name, mode):
    selections = {name: mode}
    settings = dict(speed_aware=False, accel_aware=False, offside_aware=False, pass_lane_features=False)
    command = wrapper_commands(selections, **settings)[0]
    assert_single_selections(command, selections)
    args = normalize_feature_settings(vars(training_parser.parse_args(command)))
    assert args["feature_selections"] == selections
    if name in EDGE_OPTIONS:
        assert group_mode(args, EDGE_OPTIONS[name]) == mode.replace("-", "_")
    expected = wrapper.resolve_wrapper_feature_flags(argparse.Namespace(feature_selections=selections, **settings))
    normalize_feature_settings(expected)
    width = 26 + int(expected["lane_survival"])
    args.update(task="pass_intent", node_in_dim=width)
    expected.update(task="pass_intent", node_in_dim=width)
    assert feature_signature(args) == feature_signature(expected)


@pytest.mark.parametrize("mode", NODE_MODES)
@pytest.mark.parametrize("aggregation", ["max", "top_10", "top_pass_10"])
def test_wrapper_lane_selection_preserves_aggregation(training_parser, mode, aggregation):
    selections = {"lane-survival": mode}
    command = wrapper_commands(selections, lane_survival_mode=aggregation)[0]
    assert_single_selections(command, selections)
    args = normalize_feature_settings(vars(training_parser.parse_args(command)))
    expected = None if mode == "none" else aggregation
    assert args["lane_survival_mode"] == expected
    assert args["lane_survival"] == (mode != "none")
    assert command.count("--lane-survival-mode") == int(mode != "none")


def test_wrapper_omitted_selectors_preserve_legacy_defaults(training_parser):
    flags = wrapper.append_low_level_feature_flags([], wrapper.resolve_wrapper_feature_flags(argparse.Namespace()))
    assert flags == [
        "--possessor_aware", "--keeper_aware", "--ball_z_aware", "--poss_vel_aware",
        "--speed", "--accel", "--offside", "--no-pass-lane-features", "--no-lane-survival",
        "--position-direction-edge-features", "--relative-velocity-edge-features", "--separation-rate-edge-features",
    ]
    for command in wrapper_commands():
        args = normalize_feature_settings(vars(training_parser.parse_args(command)))
        assert not args.get("feature_selections")
        assert args["speed_aware"] and args["accel_aware"] and args["offside_aware"]
        assert args["lane_survival_mode"] is None


@pytest.mark.parametrize("aggregation", ["max", "top_25", "top_pass_10"])
def test_wrapper_legacy_only_flags_keep_their_behavior(training_parser, aggregation):
    for command in wrapper_commands(speed_aware=False, accel_aware=False, offside_aware=False,
                                    pass_lane_features=True, lane_survival_mode=aggregation,
                                    v_edge_feature_mode="no_poss", relative_speed_edge_feature_mode="no_poss",
                                    relative_velocity_edge_feature_mode="no_poss", separation_rate_edge_feature_mode="none"):
        args = normalize_feature_settings(vars(training_parser.parse_args(command)))
        assert not args.get("feature_selections")
        assert args["lane_survival_mode"] == aggregation
        assert args["pass_lane_features"]
        assert not any(args[name + "_aware"] for name in ("speed", "accel", "offside"))
        for group in ("v", "relative_speed", "relative_velocity"):
            assert group_mode(args, group) == "no_poss"
        assert group_mode(args, "separation_rate") == "none"
        assert command.count("--lane-survival") == 1
        assert "--lane-survival-mode" not in command
        assert "--no-lane-survival" not in command


@pytest.mark.parametrize("possessor_aware", [True, False])
def test_wrapper_round_trip_preserves_inherited_profile_and_column_overrides(training_parser, possessor_aware):
    selections = {"vel-node-features": "none", "vx": "defending", "vy": "defending", "possessor-id": "none"}
    settings = dict(possessor_aware=possessor_aware, poss_vel_aware=False)
    command = wrapper_commands(selections, **settings)[0]
    assert_single_selections(command, selections)
    assert ("--possessor_aware" in command) == possessor_aware
    actual = normalize_feature_settings(vars(training_parser.parse_args(command)))
    expected = wrapper.resolve_wrapper_feature_flags(argparse.Namespace(feature_selections=selections, **settings))
    # Include the optional columns to compare inherited masks as well as vx/vy.
    actual.update(task="pass_intent", node_in_dim=26)
    expected.update(task="pass_intent", node_in_dim=26)
    assert feature_signature(actual) == feature_signature(expected)
    masks = feature_signature(actual)["node"]
    assert masks["node.vx"] == masks["node.vy"] == ["defending"]
    assert masks["node.speed"] == masks["node.accel"] == []
    assert bool(masks["node.poss_dist"]) == possessor_aware


def test_actual_training_parser_still_rejects_conflicting_flags(training_parser):
    for flags in (["--lane-survival", "none", "--no-lane-survival"],
                  ["--relative-speed-edge-features", "none", "--no-relative-speed-edge-features"],
                  ["--vx", "attacking", "--vx", "defending"]):
        with pytest.raises(SystemExit):
            training_parser.parse_args(["--task", "pass_intent", *flags])


def test_dataset_runtime_prepared_cache_and_auxiliary_configuration(tmp_path):
    source = graph()
    # No goal nodes for this task; preserve player goalkeepers.
    source.x = source.x[:4]
    source.edge_index = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T.contiguous()
    source.edge_attr = torch.ones((16, 5))
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


def test_wrapper_generated_features_train_one_cpu_epoch(synthetic_training, monkeypatch):
    import pandas as pd
    import pc_xpass_versions
    from physical_pass_model import pc_xpass_metadata, pc_xpass_lane_survival_column_for_mode

    run, observations, _, _, root = synthetic_training
    cache = root / "pc_xpass"
    (cache / "matches").mkdir(parents=True)
    metadata = pc_xpass_metadata("consider_teammates", top_pass_values=[10])
    (cache / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Give the synthetic graphs aligned cache rows for real lane-survival loading.
    for path in (root / "features").glob("*.pt"):
        graphs = torch.load(path, weights_only=False)
        source = graphs[0]
        source.node_ids = [f"p{i}" for i in range(source.num_nodes)]
        torch.save(graphs, path)
        row = {"action_index": 0, **{
            pc_xpass_lane_survival_column_for_mode(player, "top_pass_10"): .42
            for player in source.node_ids
        }}
        pd.DataFrame([row]).to_parquet(cache / "matches" / (path.stem + ".parquet"), index=False)
    monkeypatch.setattr(pc_xpass_versions, "cache_dir", lambda *a, **kw: cache)
    selections = {
        "vel-node-features": "none", "vx": "defending", "vy": "defending",
        "lane-survival": "no-poss", "pass-lane-features": "none",
        "relative-speed-edge-features": "none", "relative-velocity-edge-features": "no-poss",
        "separation-rate-edge-features": "no-poss",
    }
    settings = wrapper.resolve_wrapper_feature_flags(argparse.Namespace(
        feature_selections=selections, lane_survival_mode="top_pass_10"))
    flags = wrapper.append_edge_feature_flag([], "none", "none", selections)
    flags = wrapper.append_low_level_feature_flags(flags, settings)
    checkpoint = run("wrapper_selected", ["--max-epochs", "1", *flags])
    assert len(observations) == 1
    assert checkpoint["args"]["device"] == "cpu"
    assert checkpoint["args"]["feature_selections"] == selections
    assert checkpoint["args"]["lane_survival_mode"] == "top_pass_10"
    assert checkpoint["metadata"]["lane_survival"]["mode"] == "top_pass_10"
    masks = checkpoint["metadata"]["feature_signature"]["feature_masks"]["node"]
    assert masks["node.vx"] == masks["node.vy"] == ["defending"]
    assert masks["node.speed"] == masks["node.accel"] == []
    assert masks["node.lane_survival"] == ["attacking", "defending", "goal"]


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
