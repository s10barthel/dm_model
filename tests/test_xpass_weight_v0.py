"""Pure physical pass-success values survive attachment and downstream scoring."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

import inference
import physical_pass_model as physics
from datatools import config, metadata_summary
from datatools.epv import compute_epv_values
from datatools.viz_helpers import compute_pass_score
from models.utils import run_epoch
from pc_xpass_inference_policy import PCXPassBlendingError
from test_physical_xpass import (
    DummyBaselineInferenceModel, DummyOffsetModel, make_graph, make_label,
    make_physical_inference_match, write_physical_inference_cache,
)
from test_saved_visualization_xpass import write_cache


@pytest.mark.parametrize("value", [0., 1., 1e-10, .123456789012345])
def test_scalar_v0_copies_without_using_model_or_distance(value):
    result = physics.blend_physical_xpass_predictions(
        pass_success_model=np.nan, xpass=value, pass_distance=None, weight_version="v0",
    )
    assert isinstance(result, float)
    assert result == value


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_v0_copies_arrays_and_preserves_precision(backend):
    values = np.array([0., 1., 1e-10, .123456789012345])
    xpass = torch.from_numpy(values) if backend == "torch" else values
    result = physics.blend_physical_xpass_predictions(
        pass_success_model=np.full(4, np.nan), xpass=xpass, pass_distance=None,
        distance_to_nearest_opponent=None, pass_height=None, pass_intent=None, weight_version="v0",
    )
    if backend == "torch":
        assert result.dtype == xpass.dtype and result.device == xpass.device
        assert torch.equal(result, xpass)
        assert result.data_ptr() != xpass.data_ptr()
    else:
        np.testing.assert_array_equal(result, values)
        assert not np.shares_memory(result, values)


def test_v0_mixed_tensor_inputs_preserve_physical_precision():
    values = np.array([0., 1., .123456789012345], dtype=np.float64)
    model = torch.full((3,), torch.nan, dtype=torch.float32)
    result = physics.blend_physical_xpass_predictions(
        pass_success_model=model, xpass=values, pass_distance=None, weight_version="v0",
    )
    assert result.device == model.device and result.dtype == torch.float64
    np.testing.assert_array_equal(result.numpy(), values)


def test_v0_finite_mask_needs_only_eligible_physical_values():
    finite, missing = inference._physical_xpass_blend_finite_mask(
        xpass=np.array([np.nan, 0., 1., np.nan, np.inf]),
        pass_distance=None, offside=np.array([False, False, False, True, False]),
        candidate_eligibility=np.array([False, True, True, True, True]), weight_version="v0",
    )
    assert finite.tolist() == [False, True, True, False, False]
    assert missing.tolist() == [False, False, False, False, True]


@pytest.fixture
def physical_run(tmp_path):
    folder = write_cache(tmp_path / "cache", cache_format=None)
    (folder / "matches").mkdir()
    values = [0., 1., 1e-10, .123456789012345]
    rows = pd.DataFrame({
        "match_id": ["match_1"] * 4, "action_index": [10, 11, 12, 13],
        "frame_scope": ["frame_id"] * 4, "home_2__top10_xpass": values,
        "home_2__top25_xpass": [.4, .5, .6, .7],
    })
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    subject = make_physical_inference_match([10, 11, 12, 13], [1] * 4)
    for graph in subject.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros((graph.num_nodes, 1))], dim=1)
    model = DummyBaselineInferenceModel(folder)
    model.args.update(pc_xpass=True, pc_xpass_id="fixture", x_pass_version="top10",
                      xpass_weight="v0", node_in_dim=26, physical_eps=.01, physical_xpass_floor=.2)
    return subject, model, rows, folder


@pytest.mark.parametrize("post_action", [False, True])
@pytest.mark.parametrize("version", ["top10", "top25"])
def test_v0_inference_uses_raw_selected_metric_and_keeps_model_inputs(physical_run, post_action, version):
    subject, model, rows, folder = physical_run
    model.args["x_pass_version"] = version
    if post_action:
        rows["frame_scope"] = "receive_frame_id"
        rows.to_parquet(folder / "matches/match_1.parquet", index=False)
        subject.graph_features_1 = [graph.clone() for graph in subject.graph_features_0]
    captured = []
    model.register_forward_pre_hook(lambda _, args: captured.append(getattr(args[0], physics.PHYSICAL_XPASS_PROB_ATTR).clone()))
    final, learned = inference.inference_gnn(subject, model, device="cpu", post_action=post_action,
                                            return_model_pass_success=True)
    np.testing.assert_array_equal(final.home_2, rows[f"home_2__{version}_xpass"])
    assert learned.home_2.tolist() == pytest.approx([.9] * 4)
    assert final.home_1.tolist() == pytest.approx([.9] * 4)
    assert len(captured) == 1
    finite_inputs = captured[0][torch.isfinite(captured[0])]
    assert torch.all(finite_inputs >= .2) and torch.all(finite_inputs <= .99)
    assert physics.PHYSICAL_XPASS_RAW_PROB_ATTR not in subject.graph_features_0[0]
    lookup = physics.physical_xpass_inference_lookup_config(model.args, cache_dir=folder)
    assert lookup["weight_version"] == "v0"
    assert metadata_summary._summary_xpass_weight({"physical_xpass_weight_version": "v0",
                                                 "physical_xpass_metric": "top10_xpass"}) == "v0"


def test_v0_keeps_offside_zeros_and_carry_predictions(physical_run):
    subject, model, _, _ = physical_run
    subject.graph_features_0[1].x[1, -1] = 1
    subject.labels[2, config.LABEL_INDEX["is_dribble"]] = 1
    final, learned = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert final.at[11, "home_2"] == learned.at[11, "home_2"] == 0.
    assert final.at[12, "home_2"] == learned.at[12, "home_2"] == pytest.approx(.9)
    assert final.home_1.tolist() == pytest.approx([.9] * 4)


def test_v0_supports_standard_physical_cache_without_pc_xpass(tmp_path):
    folder = tmp_path / "physical"
    write_physical_inference_cache(folder, [10, 11, 12])
    rows = pd.read_parquet(folder / "matches/match_1.parquet")
    values = [0., 1., .123456789012345]
    rows["home_2"] = rows["home_2__topmean_xpass"] = values
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    model = DummyBaselineInferenceModel(folder)
    model.args.update(xpass_weight="v0", x_pass_version="top10", physical_xpass_floor=.2)
    subject = make_physical_inference_match([10, 11, 12], [1] * 3)
    final, _ = inference.inference_gnn(subject, model, device="cpu")
    np.testing.assert_array_equal(final.home_2, values)


@pytest.mark.parametrize("failure", ["row", "column", "nan", "inf"])
def test_v0_missing_physical_values_remain_unavailable(physical_run, failure):
    subject, model, rows, folder = physical_run
    if failure == "row":
        rows = rows.iloc[1:]
    elif failure == "column":
        rows = rows.drop(columns="home_2__top10_xpass")
    else:
        rows.loc[0, "home_2__top10_xpass"] = np.nan if failure == "nan" else np.inf
    rows.to_parquet(folder / "matches/match_1.parquet", index=False)
    final, learned = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    assert np.isnan(final.at[10, "home_2"])
    assert np.isfinite(learned.at[10, "home_2"])
    assert final.attrs["pc_xpass_blending_coverage"]["missing_count"] == (4 if failure == "column" else 1)


def test_v0_pass_score_and_epv_use_physical_success(physical_run):
    subject, model, rows, _ = physical_run
    final, learned = inference.inference_gnn(subject, model, device="cpu", return_model_pass_success=True)
    success = final[["home_2"]]
    outcomes = [pd.DataFrame(value, index=success.index, columns=success.columns)
                for value in (.8, .2, .1, .3)]
    expected = rows.home_2__top10_xpass.to_numpy() * .7 + (1 - rows.home_2__top10_xpass.to_numpy()) * -.1
    score = compute_pass_score(success, *outcomes)
    np.testing.assert_allclose(score.home_2, expected)
    intent = pd.DataFrame(.25, index=success.index, columns=success.columns)
    np.testing.assert_allclose(compute_epv_values(intent, success, *outcomes), .25 * expected)
    assert not np.allclose(score, compute_pass_score(learned[["home_2"]], *outcomes))


def test_combined_v0_evaluation_preserves_raw_values_and_reports_zero_weight():
    samples = []
    for index, value in enumerate([0., 1., .123456789012345]):
        graph = make_graph(["home_1", "home_2"])
        rows = pd.DataFrame({"home_2__top10_xpass": [value]}, index=[index])
        label = make_label(action_index=index)
        physics.attach_evaluation_xpass_to_graph(graph, label, rows, match_id="m", metric="top10_xpass")
        # Distance is unnecessary for v0, including its observed-target validation.
        del graph[physics.EVALUATION_XPASS_DISTANCE_ATTR]
        samples.append((graph, label, torch.tensor(1.)))
    args = SimpleNamespace(
        gnn_task="node_binary", task="pass_success", include_out=False, lambda_l1=0.,
        residual_regularization_lambda=0., use_physical_xpass=False, model_variant="gat_baseline",
        use_xg=False, use_xt=False, use_goal_distance=False, use_epv=False, print_freq=99, clip=10,
        weighted_pass_success_metrics=False, evaluate_xpass=False, evaluate_combined_success=True,
        return_pass_success_height_evaluation=True, observed_pass_height_stratification=False,
        xpass_weight="v0", xpass_metric="top10_xpass", classification_threshold=.5,
    )
    metrics, evaluation = run_epoch(args, DummyOffsetModel(output_values=[0., 2.] * 3),
                                   DataLoader(samples, batch_size=3), device="cpu", train=False)
    diagnostic = evaluation["predictor_diagnostics"]
    np.testing.assert_array_equal(diagnostic["predictors"]["combined_v0"], [0., 1., .123456789012345])
    np.testing.assert_array_equal(diagnostic["combined_learning_weight"], np.zeros(3))
    assert metrics["combined_learning_weight_mean"] == metrics["combined_learning_weight_max"] == 0.


INFERENCE_COMMANDS = [
    ("run_relevant_models", []), ("run_hawkeye", []), ("run_skillcorner", []),
    ("run_benchmark", []), ("generate_epv", []), ("run_hawkeye_loc", []),
    ("run_and_visualize_hawkeye", []), ("visualize_action_components", ["--match-id", "m"]),
]


@pytest.mark.parametrize("command,required", INFERENCE_COMMANDS)
def test_inference_clis_accept_v0_and_reject_ball_z_limit(command, required, monkeypatch):
    module = importlib.import_module(f"scripts.{command}")
    def parse(extra):
        enable = [] if command == "run_hawkeye_loc" else ["--use-physical-xpass"]
        monkeypatch.setattr(sys, "argv", [command, *required, *enable, "--xpass-weight", "v0", *extra])
        return module.parse_args()
    assert parse([]).xpass_weight == "v0"
    assert parse(["--ball-z-limit", "none"]).xpass_weight == "v0"
    with pytest.raises(SystemExit):
        parse(["--ball-z-limit", "0"])


def test_v0_evaluation_wrapper_accepts_pure_physical_mode():
    from scripts import evaluate_relevant_models as wrapper
    args = wrapper.parse_args(["--evaluate-combined-success", "--xpass-version", "top10", "--xpass-weight", "v0"])
    wrapper.validate_pass_success_predictor_args(args)
    assert args.xpass_weight == "v0"


@pytest.mark.parametrize("command,required", INFERENCE_COMMANDS[-2:])
def test_v0_can_be_compared_with_existing_weights(command, required):
    module = importlib.import_module(f"scripts.{command}")
    flags = [*required, "--compare", "--pc-xpass-id", "cache", "--use-physical-xpass",
             "--xpass-weight", "v0", "--xpass-weight", "v3"]
    args = module.parse_args(flags)
    assert args.comparison_dimension == "weight" and args.xpass_weight == ["v0", "v3"]
    with pytest.raises(SystemExit):
        module.parse_args([*flags, "--ball-z-limit", "1"])


def test_v0_ball_z_limit_rejected_by_shared_apis():
    with pytest.raises(ValueError, match="ball-z-limit"):
        physics.physical_xpass_ball_z_limit({"xpass_weight": "v0", "ball_z_limit": 0.})
    with pytest.raises(ValueError, match="ball-z-limit"):
        physics.blend_physical_xpass_predictions(pass_success_model=.9, xpass=.2, pass_distance=None,
                                                weight_version="v0", ball_z_limit=0.)
