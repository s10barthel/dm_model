"""Component/xPass reuse saves forward passes without changing predictions."""
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest
import torch

import inference
import pc_xpass_auxiliary as auxiliary
import physical_pass_model as physics
from models.inference_masks import configure_inference_masks
from test_pc_xpass_auxiliary import auxiliary_model
from test_physical_xpass import make_physical_inference_match


@pytest.fixture
def pipeline(monkeypatch):
    match = make_physical_inference_match([7, 8], [1, 1])
    for graph in match.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros((graph.num_nodes, 1))], dim=1)
    tasks = ("action_intent", "pass_intent", "pass_height", "pass_success",
             "outcome_scoring", "outcome_conceding")
    models = {task: auxiliary_model(task, model_id=f"{task}/shared") for task in tasks}
    for task in ("outcome_scoring", "outcome_conceding"):
        models[task].forward = lambda batch: batch.x.new_zeros((batch.num_nodes, 2))
    primary = models["pass_success"]
    primary.args.update(pc_xpass=True, inference_use_physical_xpass=True, xpass_weight="v5")
    primary._xpass_auxiliary_models = {
        task: auxiliary_model(task, model_id=f"{task}/shared")
        for task in ("pass_intent", "pass_height")
    }
    calls = Counter()
    for role, model in {**models, **{f"xpass_{task}": model for task, model
                                    in primary._xpass_auxiliary_models.items()}}.items():
        def count(module, inputs, output, role=role):
            calls[role] += 1
        model.register_forward_hook(count)
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference",
                        lambda match, graphs, labels, model, **kw: (graphs, labels))

    def attach(match, graphs, labels, model, **kwargs):
        for graph in graphs:
            for attr, value in ((physics.PHYSICAL_XPASS_PROB_ATTR, .2),
                                (physics.PHYSICAL_XPASS_DISTANCE_ATTR, 20.)):
                setattr(graph, attr, graph.x.new_full((graph.num_nodes,), value))
        return graphs
    monkeypatch.setattr(inference, "attach_physical_xpass_for_inference", attach)
    return match, models, calls


def run_components(match, models, records):
    for task in ("pass_intent", "pass_height"):
        inference.inference_gnn(match, models[task], device="cpu", precomputed_auxiliaries=records)


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_reuse_skips_forward_and_fingerprinting_and_preserves_blending(pipeline, monkeypatch, version):
    match, models, calls = pipeline
    primary = models["pass_success"]
    primary.args["xpass_weight"] = version
    if version == "v4":
        del primary._xpass_auxiliary_models["pass_intent"]
    baseline, _ = inference.inference_gnn(match, primary, device="cpu")
    calls.clear()
    records = {}
    run_components(match, models, records)
    monkeypatch.setattr(auxiliary, "original_input_fingerprint",
                        lambda *args: pytest.fail("shared component inputs were hashed"))
    reused, _ = inference.inference_gnn(match, primary, device="cpu", precomputed_auxiliaries=records)
    pd.testing.assert_frame_equal(reused, baseline)
    assert calls["pass_intent"] == calls["pass_height"] == calls["pass_success"] == 1
    assert calls["xpass_pass_intent"] == calls["xpass_pass_height"] == 0


def test_subset_reordered_columns_and_feature_action_mapping(pipeline):
    match, models, calls = pipeline
    records = {}
    run_components(match, models, records)
    records = {task: replace(record, probabilities=record.probabilities.iloc[:, ::-1],
                             feature_action_indices=(7, 8)) for task, record in records.items()}
    predictions = auxiliary.predict_auxiliaries(
        match, models["pass_success"], "cpu", graphs=match.graph_features_0, labels=match.labels,
        feature_action_indices=[7, 8], event_indices=[8], precomputed_auxiliaries=records,
    )
    for task, probabilities in predictions.items():
        pd.testing.assert_frame_equal(probabilities, records[task].probabilities.loc[[8]])
        assert calls[f"xpass_{task}"] == 0


@pytest.mark.parametrize("difference", ["model_id", "settings", "mask", "graphs", "labels", "match",
                                        "post_action", "mapping", "coverage", "device"])
def test_incompatible_records_fall_back_to_auxiliary(pipeline, difference):
    match, models, calls = pipeline
    records = {}
    run_components(match, models, records)
    record = records["pass_height"]
    changes = {
        "graphs": {"graphs": list(record.graphs)}, "labels": {"labels": record.labels.clone()},
        "match": {"match": object()}, "post_action": {"post_action": True},
        "mapping": {"feature_action_indices": (8, 7)},
        "coverage": {"probabilities": record.probabilities.loc[[7]]}, "device": {"device": "cuda"},
    }
    if difference in changes:
        records["pass_height"] = replace(record, **changes[difference])
    else:
        height = models["pass_success"]._xpass_auxiliary_models["pass_height"]
        if difference == "mask":
            configure_inference_masks({"pass_height": height}, ["pass_height:node.speed"])
        else:
            height.args["model_id" if difference == "model_id" else "xy_only"] = (
                "pass_height/different" if difference == "model_id" else True)
    inference.inference_gnn(match, models["pass_success"], device="cpu", precomputed_auxiliaries=records)
    assert calls["xpass_pass_height"] == 1
    assert calls["xpass_pass_intent"] == 0


@pytest.mark.parametrize("collector", ["skillcorner", "hawkeye", "benchmark", "hawkeye_visualizer",
                                       "action_visualizer", "epv"])
def test_collectors_compute_shared_models_once(pipeline, monkeypatch, tmp_path, collector):
    match, models, calls = pipeline
    if collector in {"skillcorner", "hawkeye", "benchmark"}:
        import importlib
        module = importlib.import_module(f"datatools.{collector}")
        result = getattr(module, f"infer_{collector}_components")(match, models, device="cpu")
    elif collector == "hawkeye_visualizer":
        from scripts.run_and_visualize_hawkeye import collect_situation_components
        result = collect_situation_components(match, models, "cpu", [7, 8], [])
    elif collector == "action_visualizer":
        from scripts import visualize_action_components as visualizer
        monkeypatch.setattr(visualizer, "possession_state_for_action", lambda *a: (match, 7))
        # Pass-success can precede both components in user-selected order.
        selected = {task: models[task] for task in ("pass_success", "pass_height", "pass_intent")}
        result = visualizer.collect_action_components(
            match, selected, tmp_path, "cpu", 7, "action", tmp_path, rendered_components=list(selected))
    else:
        from scripts import generate_epv
        monkeypatch.setattr(generate_epv, "build_epv_action_values", lambda actions, values: values)
        result = generate_epv.infer_match_epv(match, models, "cpu")
    assert len(result) > 0
    assert calls["pass_intent"] == calls["pass_success"] == 1
    assert calls["xpass_pass_intent"] == 0
    assert calls["pass_height"] == (0 if collector == "epv" else 1)
    assert calls["xpass_pass_height"] == (1 if collector == "epv" else 0)


def test_visualizer_cache_hits_register_reusable_components(pipeline):
    from scripts.run_and_visualize_hawkeye import collect_situation_components
    match, models, calls = pipeline
    cache = {}
    components_only = {task: models[task] for task in ("pass_intent", "pass_height")}
    collect_situation_components(match, components_only, "cpu", [7, 8], [], prediction_cache=cache)
    result = collect_situation_components(match, models, "cpu", [7, 8], [], prediction_cache=cache)
    assert not result["pass_success"].empty
    assert calls["pass_intent"] == calls["pass_height"] == calls["pass_success"] == 1
    assert calls["xpass_pass_intent"] == calls["xpass_pass_height"] == 0


def test_action_visualizer_keeps_recorded_and_possession_inputs_separate(pipeline, monkeypatch, tmp_path):
    from scripts import visualize_action_components as visualizer
    match, models, calls = pipeline
    possession = make_physical_inference_match([7, 8], [1, 1])
    possession.graph_features_0 = [graph.clone() for graph in match.graph_features_0]
    monkeypatch.setattr(visualizer, "possession_state_for_action", lambda *a: (possession, 7))
    selected = {task: models[task] for task in ("pass_intent", "pass_height", "pass_success")}
    result = visualizer.collect_action_components(
        match, selected, tmp_path, "cpu", 7, "action", tmp_path, rendered_components=list(selected))
    assert "pass_success" in result
    assert calls["pass_intent"] == calls["pass_height"] == 1
    assert calls["xpass_pass_intent"] == calls["xpass_pass_height"] == 1


def test_auxiliary_only_and_new_possessions_do_not_reuse_previous_records(pipeline):
    from datatools.skillcorner import infer_skillcorner_components
    match, models, calls = pipeline
    models.pop("pass_height")
    second = make_physical_inference_match([7, 8], [1, 1])
    second.graph_features_0 = [graph.clone() for graph in match.graph_features_0]
    for state in (match, second):
        infer_skillcorner_components(state, models, device="cpu")
    assert calls["pass_intent"] == 2
    assert calls["xpass_pass_intent"] == 0
    assert calls["xpass_pass_height"] == 2


def test_matching_masks_reuse_and_filtered_physical_rows_align(pipeline, monkeypatch):
    match, models, calls = pipeline
    primary = models["pass_success"]
    participating = {**models, **{f"xpass_{task}": model for task, model
                                 in primary._xpass_auxiliary_models.items()}}
    configure_inference_masks(participating, ["pass_height:node.speed", "pass_intent:node.speed"])
    monkeypatch.setattr(inference, "filter_missing_physical_xpass_rows_for_inference",
                        lambda match, graphs, labels, model, **kw: (graphs[1:], labels[1:]))
    baseline, _ = inference.inference_gnn(match, primary, device="cpu")
    calls.clear()
    records = {}
    run_components(match, models, records)
    reused, _ = inference.inference_gnn(match, primary, device="cpu", precomputed_auxiliaries=records)
    pd.testing.assert_frame_equal(reused, baseline)
    assert reused.index.tolist() == [8]
    assert calls["xpass_pass_intent"] == calls["xpass_pass_height"] == 0
