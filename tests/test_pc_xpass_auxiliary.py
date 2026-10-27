"""Physical-only generation and explicitly aligned learned auxiliaries."""
from argparse import Namespace
import copy

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

import inference
import physical_pass_model as physics
import pc_xpass_versions as versions
from pc_xpass_auxiliary import (configure_auxiliary_models, validate_auxiliary_selectors,
                                predict_auxiliaries, observed_auxiliary_predictions)
from pc_xpass_contract import PHYSICAL_INPUT_CONTRACT
from datatools import config
from datatools.utils import filter_features_and_labels
from models.inference_masks import configure_inference_masks
from scripts import generate_physical_xpass as generate
from test_physical_xpass import (make_graph, make_label, make_pass_success_args,
                                 make_physical_inference_match, ConstantPassHeightModel)


def arguments(version="v5", **kwargs):
    return Namespace(pc_xpass=True, use_physical_xpass=True, xpass_weight=version,
                     xpass_pass_height_model_id=kwargs.pop("height", None),
                     xpass_pass_intent_model_id=kwargs.pop("intent", None), **kwargs)


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
def test_explicit_selector_matrix(version):
    args = arguments(version, height="pass_height/h" if version in {"v4", "v5"} else None,
                     intent="pass_intent/i" if version == "v5" else None, v5_discount=False)
    selected = validate_auxiliary_selectors(args)
    assert set(selected) == ({"pass_height", "pass_intent"} if version == "v5" else {"pass_height"} if version == "v4" else set())
    for role in selected:
        missing = copy.copy(args)
        setattr(missing, f"xpass_{role}_model_id", None)
        # Visible components and bundle contents do not satisfy the requirement.
        setattr(missing, f"{role}_model_id", selected[role])
        missing.bundle_id = "bundle"
        with pytest.raises(ValueError, match="required explicitly"):
            validate_auxiliary_selectors(missing)


@pytest.mark.parametrize("args", [arguments("v3", height="pass_height/h"),
    arguments("v4", height="pass_height/h", intent="pass_intent/i"),
    arguments("v4", height="pass_success/s")])
def test_unused_and_wrong_task_selectors(args):
    with pytest.raises(ValueError):
        validate_auxiliary_selectors(args)


def test_weighted_evaluation_requires_height_only():
    args = arguments("v3", height="pass_height/h", weighted_pass_success_metrics=True)
    args.use_physical_xpass = False
    assert validate_auxiliary_selectors(args) == {"pass_height": "pass_height/h"}


@pytest.mark.parametrize("flag,value", [("--pass-height-model-id", "pass_height/h"),
    ("--pass-height-device", "cpu"), ("--pass-height-threshold", "1.5")])
def test_generation_rejects_height_arguments(flag, value):
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", flag, value])


def test_generation_never_loads_predicts_or_expands_metrics(tmp_path, monkeypatch):
    args = generate.parse_args(["--pc-xpass", "--no-sportec", "--num-workers", "1",
        "--min-speed", "3", "--max-speed", "3", "--angle-step", "90", "--radial-gridsize", "5", "--top-pass", "10"])
    monkeypatch.setattr(generate, "load_model", lambda *a, **k: pytest.fail("generation loaded a model"))
    monkeypatch.setattr(physics, "_pass_height_predictions_for_graphs", lambda *a, **k: pytest.fail("generation predicted height"))
    generate.prepare_pass_height_context(args)
    generate.prepare_runtime_graph_schema(args)
    assert not generate.runtime_graph_schema_from_args(args)["add_v_edge_features"]
    item = dict(match_id="sample", graphs=[make_graph()], labels=torch.stack([make_label()]))
    generate.prewarm_runtime_items([item], cache_dir=tmp_path, args=args)
    rows = physics.load_physical_xpass_match(tmp_path, "sample")
    assert not any("pass_height" in name for name in rows.columns)
    assert rows.attrs["physical_xpass_cache_metadata"]["physical_input_contract"] == PHYSICAL_INPUT_CONTRACT
    before = (tmp_path / "matches/sample.parquet").read_bytes()
    stats = generate.prewarm_runtime_items([item], cache_dir=tmp_path, args=args)
    assert stats["cache_hits"] == 1
    assert not any("pass_height" in name for name in stats)
    assert (tmp_path / "matches/sample.parquet").read_bytes() == before


def auxiliary_model(task="pass_height", probability=.8, **kwargs):
    model = ConstantPassHeightModel(probability=probability, node_in_dim=26, edge_in_dim=kwargs.get("edge_in_dim", 2))
    options = dict(use_physical_xpass=False, task=task, node_in_dim=26,
                   goal_context_version=2, goal_nodes_aware=False, include_goals=False)
    options.update(kwargs)
    model.args.update(make_pass_success_args(**options))
    return model.eval()


def test_roles_load_independently_and_cutoff_is_checkpoint_owned(monkeypatch):
    from models import utils
    from models.pass_height import definition
    height = auxiliary_model(pass_height_definition=definition(1.5, "checkpoint"))
    intent = auxiliary_model("pass_intent")
    monkeypatch.setattr(utils, "load_model", lambda identifier, device: height if identifier.startswith("pass_height/") else intent)
    monkeypatch.setattr(utils, "get_model_provenance", lambda identifier: {"model_id": identifier})
    component = auxiliary_model(probability=.2)
    primary = auxiliary_model("pass_success")
    models = {"pass_height": component, "pass_success": primary}
    args = arguments(height="pass_height/h", intent="pass_intent/i", pass_height_threshold=1.5)
    selected = configure_auxiliary_models(args, models, "cpu")
    assert models["pass_height"] is component
    assert models["xpass_pass_height"] is height
    assert primary._xpass_auxiliary_models == selected
    args.pass_height_threshold = 2
    with pytest.raises(inference.PCXPassBlendingError, match="cutoff"):
        configure_auxiliary_models(args, models, "cpu")


def graph_with_goals():
    graph = make_graph(["home_1", "home_2", "away_3", "home_goal", "away_goal"], edge_dim=4)
    graph.x = torch.cat([graph.x, torch.zeros((5, 1))], dim=1)
    graph.x[3, config.NODE_FEATURE_IS_TEAMMATE] = 1
    graph.x[3:, config.NODE_FEATURE_IS_GOAL] = 1
    return graph


def test_private_graphs_goals_features_masks_and_original_target_alignment():
    source = graph_with_goals()
    label = make_label()
    label[config.LABEL_INDEX["pass_high"]] = float("nan")
    source.source_node_positions = torch.arange(source.num_nodes)
    before = source.x.clone()
    height = auxiliary_model(goal_nodes_aware=True, edge_in_dim=4)
    height.forward = lambda batch: batch.x[:, config.NODE_FEATURE_X] / 20.
    intent = auxiliary_model("pass_intent", xy_only=True, extend_features=False)
    models = {"pass_height": height, "pass_intent": intent}
    configure_inference_masks(models, ["pass_height:node.speed"])
    primary, _ = filter_features_and_labels([source], label[None], auxiliary_model("pass_success").args)
    primary = primary[0]
    primary.evaluation_match_id = "match"
    # Reorder auxiliary candidates, retaining immutable original node positions.
    for role, model in models.items():
        graphs, _ = filter_features_and_labels([source], label[None], model.args, validate_targets=False)
        graph = graphs[0]
        permutation = torch.arange(graph.num_nodes - 1, -1, -1)
        graph.x = graph.x[permutation]
        graph.source_node_positions = graph.source_node_positions[permutation]
        graph.inference_possessor_mask = graph.inference_possessor_mask[permutation]
        setattr(primary, f"xpass_{role}_graph", graph)
    values = observed_auxiliary_predictions(Batch.from_data_list([primary]), label[None], models, "cpu")
    assert values["pass_height"].item() == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
    assert values["pass_intent"].item() == pytest.approx(1.)
    torch.testing.assert_close(source.x, before)


def test_runtime_height_matches_enrichment_helper_without_cached_height():
    match = make_physical_inference_match([7], [1])
    graph = make_graph()
    graph.x = torch.cat([graph.x, torch.zeros((3, 1))], dim=1)
    height = auxiliary_model()
    primary = auxiliary_model("pass_success")
    primary._xpass_auxiliary_models = {"pass_height": height}
    old = physics._pass_height_predictions_for_graphs([graph], height, device="cpu")[0]
    predicted = predict_auxiliaries(match, primary, "cpu", graphs=[graph], labels=match.labels)["pass_height"]
    for player in ("home_1", "home_2"):
        assert predicted.at[7, player] == pytest.approx(old[player])
    # Matching calls reuse predictions; changed original inputs get a separate entry.
    predict_auxiliaries(match, primary, "cpu", graphs=[graph], labels=match.labels)
    assert len(match._xpass_auxiliary_predictions) == 1
    graph.x[1, config.NODE_FEATURE_X] += 1
    predict_auxiliaries(match, primary, "cpu", graphs=[graph], labels=match.labels)
    assert len(match._xpass_auxiliary_predictions) == 2


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_blending_ignores_wrong_cached_height_and_component_intent(monkeypatch, version):
    match = make_physical_inference_match([7], [1])
    graph = make_graph()
    graph.x = torch.cat([graph.x, torch.zeros((3, 1))], dim=1)
    model = auxiliary_model("pass_success")
    model.args.update(pc_xpass=True, inference_use_physical_xpass=True, xpass_weight=version)
    model._xpass_auxiliary_models = {"pass_height": auxiliary_model(probability=.25)}
    if version == "v5":
        model._xpass_auxiliary_models["pass_intent"] = auxiliary_model("pass_intent")
    monkeypatch.setattr(inference, "validate_pc_xpass_startup", lambda *a, **k: None)
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference", lambda match, graphs, labels, model, **kw: (graphs, labels))
    def attach(match, graphs, labels, model, **kwargs):
        for graph in graphs:
            for attribute, value in [(physics.PHYSICAL_XPASS_PROB_ATTR, .2),
                (physics.PHYSICAL_XPASS_DISTANCE_ATTR, 20), (physics.PHYSICAL_XPASS_PASS_HEIGHT_ATTR, float("nan"))]:
                setattr(graph, attribute, graph.x.new_full((graph.num_nodes,), value))
        return graphs
    monkeypatch.setattr(inference, "attach_physical_xpass_for_inference", attach)
    predicted, _ = inference.inference_gnn(match, model, device="cpu", graph_override=[graph], label_override=match.labels,
        pass_intent_probs=pd.DataFrame({"home_2": [float("nan")]}, index=[7]))
    expected = physics.blend_physical_xpass_predictions(pass_success_model=.8, xpass=.2, pass_distance=20.,
        weight_version=version, pass_height=.25, pass_intent=1.)
    assert predicted.at[7, "home_2"] == pytest.approx(float(expected))
    assert predicted.at[7, "home_1"] == pytest.approx(.8)


def test_missing_auxiliary_predictions_fail_with_role_and_state():
    with pytest.raises(ValueError, match="pass_height.*77"):
        inference._aligned_auxiliary_values(pd.DataFrame(), 77, ["home_2"], "pass_height")
    with pytest.raises(ValueError, match="ambiguous"):
        inference._aligned_auxiliary_values(pd.DataFrame([[.2], [.3]], index=[7, 7], columns=["home_2"]), 7, ["home_2"], "pass_height")


def test_dataset_keeps_primary_samples_without_height_targets_and_prepares_two_models(tmp_path):
    from dataset import ActionDataset
    source = graph_with_goals()
    label = make_label()
    label[config.LABEL_INDEX["pass_high"]] = float("nan")
    label[config.LABEL_INDEX["pass_max_ball_z"]] = float("nan")
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    torch.save([source], features / "m.pt")
    torch.save(label[None], labels / "m.pt")
    options = dict(feature_dir=features, label_dir=labels, task="pass_success", train=False,
        min_pass_dur=0, extend_features=True, possessor_aware=True, keeper_aware=True,
        ball_z_aware=True, poss_vel_aware=True, poss_rel_vel_aware=True,
        goal_context_version=2, goal_nodes_aware=False, include_goals=False)
    baseline = ActionDataset(["m"], **options)
    auxiliary = ActionDataset(["m"], auxiliary_models_args={
        "pass_height": auxiliary_model(goal_nodes_aware=True, edge_in_dim=4).args,
        "pass_intent": auxiliary_model("pass_intent", extend_features=False, xy_only=True).args}, **options)
    assert len(baseline) == len(auxiliary) == 1
    torch.testing.assert_close(baseline.labels, auxiliary.labels, equal_nan=True)
    torch.testing.assert_close(baseline.features[0].x, auxiliary.features[0].x)
    graph = auxiliary.features[0]
    assert graph.xpass_pass_height_graph.num_nodes == 5
    assert graph.xpass_pass_intent_graph.num_nodes == 3
    assert graph.xpass_pass_height_graph.edge_attr.shape[1] == 4
    assert graph.xpass_pass_intent_graph.edge_attr.shape[1] == 2


def test_lane_auxiliary_validates_original_physical_state_before_model_masks(monkeypatch, tmp_path):
    source = graph_with_goals()
    label = make_label()
    height = auxiliary_model(goal_nodes_aware=False, poss_vel_aware=False, node_in_dim=27,
                             lane_survival=True, lane_survival_mode="top_pass_10")
    source.x[:, config.NODE_FEATURE_VX] = 2
    prepared, prepared_labels = filter_features_and_labels([source], label[None], height.args, validate_targets=False)
    row = {"state_id": "physical", "physical_state_hash": physics.physical_state_hash(source),
           physics.pc_xpass_lane_survival_column_for_mode("home_1", "top_pass_10"): .4,
           physics.pc_xpass_lane_survival_column_for_mode("home_2", "top_pass_10"): .6}
    graph = physics.append_pc_xpass_lane_survival_row_to_graph(prepared[0], prepared_labels[0], row,
        match_id="m", require_observed_target=False, mode="top_pass_10")
    assert graph.num_nodes == 3
    assert graph.x[1, -1].item() == pytest.approx(.6)
    row["physical_state_hash"] = "changed"
    with pytest.raises(ValueError, match="state differs"):
        physics.append_pc_xpass_lane_survival_row_to_graph(prepared[0], prepared_labels[0], row, match_id="m")
