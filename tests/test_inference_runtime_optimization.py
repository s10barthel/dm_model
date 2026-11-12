"""Temporary graph requirements and exact outcome decoding across inference paths."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import inference
from datatools import benchmark, hawkeye, skillcorner, sportec_possessions
from datatools.inference_features import EXTENDED_FEATURES, runtime_graph_feature_kwargs
from datatools.utils import filter_features_and_labels
from models.outcome_learning import expected_values, signed_contract
from test_hawkeye_target_clamping import make_ball, make_tracking
from test_physical_xpass import DummyBaselineInferenceModel
from test_sportec_possessions import fixture as sportec_fixture


def model(**changes):
    args = dict(task="outcome_scoring", node_in_dim=26, edge_in_dim=2,
                possessor_aware=True, extend_features=False, pass_lane_features=False, sparsify="none")
    args.update(changes)
    return SimpleNamespace(args=args)


def test_requirements_include_auxiliaries_and_role_selections():
    assert runtime_graph_feature_kwargs({"scoring": model()}) == {"computed_extended_features": ()}
    primary = model(feature_selections={"node.potential_blockers": "defending"})
    primary._xpass_auxiliary_models = {"height": model(task="pass_height", pass_lane_features=True)}
    assert runtime_graph_feature_kwargs({"scoring": primary}) == {
        "computed_extended_features": EXTENDED_FEATURES[3:],
    }
    assert runtime_graph_feature_kwargs({"scoring": primary, "conceding": model(extend_features=True)}) == {}


@pytest.mark.parametrize("changes", [
    {"lane_survival": True}, {"task": "success_intent"}, {"task": "failure_receiver"},
    {"task": "pass_success", "use_physical_xpass": True, "model_variant": "gat_phys_logit_offset"},
    {"node_in_dim": 27}, {"node_in_dim": None},
    {"node_feature_names": ["is_teammate", "is_keeper", "is_goal", *[f"custom_{i}" for i in range(23)]]},
])
def test_hash_consumers_and_unknown_layouts_keep_full_features(changes):
    assert runtime_graph_feature_kwargs({"selected": model(**changes)}) == {}


@pytest.fixture
def state():
    raw = make_tracking([(1., 10., 0.), (1.04, 10.1, 0.), (1.08, 10.2, 0.)])
    extra = []
    for team, identifier, x, y in [("A", 3, 20., 0.), ("B", 4, 15., 0.), ("B", 5, 25., 1.)]:
        rows = raw.loc[raw.team.eq(team)].copy()
        rows["uefa_player_id"], rows["centroid_x"], rows["centroid_y"] = identifier, x, y
        extra.append(rows)
    subject, _, _ = hawkeye.build_hawkeye_situation(pd.concat([raw, *extra]), make_ball([1., 1.04, 1.08]), build_graphs=False)
    subject.frame_meta["possessor_player"] = subject.frame_meta["PlayerID"]
    subject.frame_meta["player_id"] = subject.frame_meta["PlayerID"]
    subject.frame_meta["period"] = subject.frame_meta["period_id"]
    subject.original_start_frame = subject.original_end_frame = None
    return subject


@pytest.mark.parametrize("module", [hawkeye, benchmark, skillcorner])
@pytest.mark.parametrize("selected", [model(), model(pass_lane_features=True),
    model(feature_selections={"node.potential_blockers": "no-poss"}), model(extend_features=True)])
def test_runtime_builders_preserve_prepared_inputs_labels_and_selection(state, module, selected):
    kwargs = {"scope": "frames"} if module is skillcorner else {}
    actions, labels, graphs, stats = module._build_actions_and_labels(copy.copy(state), **kwargs)
    actual_actions, actual_labels, reduced, actual_stats = module._build_actions_and_labels(
        copy.copy(state), **kwargs, **runtime_graph_feature_kwargs({"selected": selected}))
    pd.testing.assert_frame_equal(actions, actual_actions)
    assert torch.equal(labels, actual_labels)
    assert stats == actual_stats
    before, _ = filter_features_and_labels(graphs, labels, selected.args)
    after, _ = filter_features_and_labels(reduced, actual_labels, selected.args)
    assert len(before) == len(after) == 3
    for a, b in zip(before, after):
        assert a.node_ids == b.node_ids
        assert a.edge_feature_names == b.edge_feature_names
        assert torch.equal(a.x, b.x)
        assert torch.equal(a.edge_index, b.edge_index)
        assert torch.equal(a.edge_attr, b.edge_attr)


def test_unused_geometry_is_not_evaluated_and_default_still_is(state, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("discarded geometry was computed")
    monkeypatch.setattr(hawkeye.utils, "find_nearest_opponent_to_pass", fail)
    monkeypatch.setattr(hawkeye.utils, "count_potential_interceptors", fail)
    monkeypatch.setattr(hawkeye.utils, "count_potential_blockers", fail)
    _, _, graphs, _ = hawkeye._build_actions_and_labels(state, **runtime_graph_feature_kwargs({"selected": model()}))
    assert all(graph.x.shape[1] == 26 for graph in graphs)
    assert all(torch.count_nonzero(graph.x[:, 19:25]) == 0 for graph in graphs)
    with pytest.raises(RuntimeError, match="discarded geometry"):
        hawkeye._build_actions_and_labels(state)


def test_sportec_runtime_policy_does_not_enter_state_provenance(monkeypatch):
    subject, _ = sportec_fixture(monkeypatch)
    original = sportec_possessions.construct_graph_for_frame
    requests = []
    def construct(*args, **kwargs):
        requests.append(kwargs.get("computed_extended_features"))
        return original(*args, **kwargs)
    monkeypatch.setattr(sportec_possessions, "construct_graph_for_frame", construct)
    before = list(sportec_possessions.build_sportec_possessions(subject, "match", Path("feature")))
    original_plan = subject.sportec_state_plan.manifest()
    assert requests and all(item is None for item in requests)
    requests.clear()
    after = list(sportec_possessions.build_sportec_possessions(subject, "match", Path("feature"),
        **runtime_graph_feature_kwargs({"selected": model()})))
    assert requests and all(item == () for item in requests)
    assert original_plan == subject.sportec_state_plan.manifest()
    for (a, _), (b, _) in zip(before, after):
        assert torch.equal(a.labels, b.labels)
        pd.testing.assert_frame_equal(a.actions, b.actions)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("signed", [False, True])
def test_bulk_outcomes_match_original_per_graph_activation_exactly(device, dtype, signed):
    counts = torch.tensor([1, 0, 3, 11, 12, 17, 2, 31], device=device)
    batch = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    logits = torch.linspace(-25., 25., 2 * len(batch), dtype=dtype, device=device).reshape(-1, 2)
    logits[0] = float("-inf")
    logits[2] = float("nan")
    offside = torch.zeros(len(batch), dtype=torch.bool, device=device)
    args = {"task": "outcome_scoring", "include_out": False}
    if signed:
        args["outcome_learning_contract"] = signed_contract(args)
    values, mask, offsets = inference._batched_outcome_predictions(logits, batch, offside, len(counts), args)
    for i in range(len(counts)):
        expected = expected_values(logits[batch == i], args).cpu().numpy()
        np.testing.assert_array_equal(values[offsets[i]:offsets[i + 1]], expected)
    assert mask.dtype == bool


@pytest.mark.parametrize("changes", [{"learning_mode": "distributional"}, {"learning_mode": "decomposition"},
                                      {"include_out": True}, {"task": "pass_success"}])
def test_specialized_decoders_retain_existing_path(changes):
    args = {"task": "outcome_scoring", **changes}
    assert inference._batched_outcome_predictions(None, None, None, 1, args) is None


def test_outcome_inference_uses_graph_candidate_ids_without_tracking_lookup(state, monkeypatch):
    actions, labels, graphs, _ = hawkeye._build_actions_and_labels(state)
    state.actions, state.labels, state.graph_features_0 = actions, labels, graphs
    class Outcome(DummyBaselineInferenceModel):
        def __init__(self):
            super().__init__(Path("unused"))
            self.args.update(task="outcome_scoring", node_in_dim=26, inference_use_physical_xpass=False)
        def forward(self, batch):
            return batch.x.new_tensor([-.3, .6]).repeat(batch.num_nodes, 1)
    selected = Outcome()
    before = inference.inference_gnn(state, selected, device="cpu")
    monkeypatch.setattr(inference, "find_active_players", lambda *a, **k: pytest.fail("unnecessary tracking lookup"))
    after = inference.inference_gnn(state, selected, device="cpu")
    for a, b in zip(before, after):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_reused_visualizer_match_does_not_keep_a_previous_feature_omission(tmp_path, monkeypatch):
    from scripts import visualize_action_components as visualizer
    subject = SimpleNamespace(runtime_graph_options={"computed_extended_features": (), "add_v_edge_features": False})
    monkeypatch.setattr(visualizer.match_cache, "read_json", lambda *a, **k: {"source": "pc_xpass"})
    monkeypatch.setattr(visualizer.match_cache, "require_format", lambda *a, **k: None)
    def resolve(match, *args):
        assert "computed_extended_features" not in match.runtime_graph_options
        assert match.runtime_graph_options["add_v_edge_features"] is False
        raise RuntimeError("stop before inference")
    monkeypatch.setattr(visualizer, "possession_state_for_action", resolve)
    with pytest.raises(RuntimeError, match="stop before inference"):
        visualizer.collect_action_components(subject, {"outcome_scoring": model(extend_features=True)},
            tmp_path, "cpu", 1, "1", tmp_path, physical_cache_dir=tmp_path, rendered_components=["outcome_scoring_success"])


def test_reused_visualizer_match_does_not_keep_a_previous_feature_omission(tmp_path, monkeypatch):
    from scripts import visualize_action_components as visualizer
    subject = SimpleNamespace(runtime_graph_options={"computed_extended_features": (), "add_v_edge_features": False})
    monkeypatch.setattr(visualizer.match_cache, "read_json", lambda *a, **k: {"source": "pc_xpass"})
    monkeypatch.setattr(visualizer.match_cache, "require_format", lambda *a, **k: None)
    def resolve(match, *args):
        assert "computed_extended_features" not in match.runtime_graph_options
        assert match.runtime_graph_options["add_v_edge_features"] is False
        raise RuntimeError("stop before inference")
    monkeypatch.setattr(visualizer, "possession_state_for_action", resolve)
    with pytest.raises(RuntimeError, match="stop before inference"):
        visualizer.collect_action_components(subject, {"outcome_scoring": model(extend_features=True)},
            tmp_path, "cpu", 1, "1", tmp_path, physical_cache_dir=tmp_path, rendered_components=["outcome_scoring_success"])
