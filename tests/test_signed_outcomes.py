"""Signed gains preserve their units throughout training and evaluation."""
import copy
import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from models.outcome_learning import (
    configure_outcome_learning, normalize_learning_settings, validate_learning_metadata,
    fit_definition, encode_targets, expected_values, mean_mse, distribution_loss,
    categorical_probabilities, crps, mean_bce,
)


def metadata(clipped=False):
    return dict(xt_version="signed_fixture", xt_targets=dict(version=2, representation="gain", clip_negative_gains=clipped),
                stop_conditions={"version": 1, "available": {"next_10": ["base"]}})


def configuration(mode="scalar", task="outcome_scoring", values=(-1., -.2, 0., .1, 1.)):
    args = dict(task=task, use_xt=True, learning_mode=mode, return_type="next_10", distribution_atoms=7)
    configure_outcome_learning(args, metadata())
    if mode == "distributional":
        labels = torch.zeros(len(values), len(LABEL_COLUMNS))
        for column in ("scores_xt", "concedes_xt"):
            labels[:, LABEL_INDEX[column]] = torch.tensor(values)
        args["outcome_distribution_definition"] = fit_definition([labels], args, {"training_match_ids": ["train"]})
    return args


@pytest.mark.parametrize("tasks", [("outcome_scoring",), ("outcome_conceding",),
                                  ("outcome_scoring", "outcome_conceding")])
@pytest.mark.parametrize("mode", ["scalar", "distributional"])
@pytest.mark.parametrize("clipped", [False, True])
def test_wrapper_parses_xt_outcome_selections(monkeypatch, tasks, mode, clipped):
    from scripts import train_relevant_models as wrapper
    from models import outcome_learning
    saved = metadata(clipped)
    saved.update(carry_variant={"available": True},
                 stop_conditions={"version": 1, "available": {"time_10": ["base", "set-pieces"]}})
    manifest = dict(manifest_id="fixture", train=[f"m{i}" for i in range(10)], test=["held_out"])
    monkeypatch.setattr(wrapper, "resolve_feature_run_id", lambda *a, **kw: "feature_fixture")
    monkeypatch.setattr(wrapper, "load_feature_run_metadata", lambda *a, **kw: saved)
    monkeypatch.setattr(wrapper, "resolve_training_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(wrapper, "infer_feature_run_intended_receiver_modes", lambda *a: ["model"])
    monkeypatch.setattr(wrapper, "infer_feature_run_return_types", lambda *a: ["time_10"])
    configured = []
    def configure(args, feature_metadata):
        result = configure_outcome_learning(args, feature_metadata)
        configured.append((args["task"], result))
        return result
    monkeypatch.setattr(outcome_learning, "configure_outcome_learning", configure)
    args = wrapper.parse_args([
        "--feature-run-id", "feature_fixture", "--target-family", "xt", "--return_type", "time_10",
        "--stop-condition", "set-pieces", "--intended-receiver-mode", "model", "--use-carries",
        "--validation-mode", "holdout_80_20", "--gnn-architecture", "relationship",
        "--learning-mode", mode, *["--only-" + task.replace("_", "-") for task in tasks],
    ])
    assert set(args.trained_tasks) == set(tasks)
    assert args.stop_condition_definition == {"version": 1, "mode": "set-pieces"}
    assert len(configured) == len(tasks)
    assert {task for task, _ in configured} == set(tasks)
    assert all((contract is not None) == (not clipped) for _, contract in configured)


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_contract_is_metadata_driven_and_immutable(mode):
    args = configuration(mode, values=(0., .2))
    assert args["outcome_learning_contract"]["domain"] == "signed_gain"
    assert args["checkpoint_monitor"] == "mse_loss"
    validate_learning_metadata(args, copy.deepcopy(args))
    restored = dict(task=args["task"], use_xt=True, learning_mode=None, return_type="next_10")
    normalize_learning_settings(restored, args)
    configure_outcome_learning(restored, metadata(), args)
    assert restored["outcome_learning_contract"] == args["outcome_learning_contract"]
    with pytest.raises(ValueError, match="contract"):
        configure_outcome_learning(restored, metadata(clipped=True), args)
    missing = copy.deepcopy(args); missing.pop("outcome_learning_contract")
    with pytest.raises(ValueError, match="contract"):
        validate_learning_metadata(missing, {})
    corrupt = copy.deepcopy(args); corrupt["outcome_learning_contract"]["loss"] = "bce"
    with pytest.raises(ValueError, match="contract"):
        validate_learning_metadata(corrupt, {})
    with pytest.raises(ValueError, match="monitor"):
        validate_learning_metadata(copy.deepcopy(args), {"checkpoint_monitor": "ce_loss"})
    changed_surface = metadata(); changed_surface["xt_version"] = "another_surface"
    with pytest.raises(ValueError, match="surface or target definition"):
        configure_outcome_learning(copy.deepcopy(args), changed_surface, args)


@pytest.mark.parametrize("override", [{"task": "outcome_return"}, {"task": "intent_return"},
                                      {"model": "xgboost"}, {"model": "catboost"}, {"weight_bce": True}])
def test_unsupported_consumers_fail_before_training(override):
    with pytest.raises(ValueError):
        configure_outcome_learning({"task": "outcome_scoring", "use_xt": True, **override}, metadata())


def test_scalar_signed_weighted_loss_and_gradients():
    args = configuration()
    pred = torch.tensor([-.6, .2, 1.2], dtype=torch.double, requires_grad=True)
    target = torch.tensor([-.8, 0., .4], dtype=torch.double)
    weight = torch.tensor([1., .5, 2.], dtype=torch.double)
    loss = mean_mse(pred, target, args, weight)
    torch.testing.assert_close(loss, (weight * (pred - target).square()).mean())
    loss.backward()
    torch.testing.assert_close(pred.grad, 2 * weight * (pred.detach() - target) / 3)
    torch.testing.assert_close(expected_values(pred, args), pred)
    for value in (float("nan"), float("inf"), -1.01, 1.01):
        with pytest.raises(ValueError, match="finite"):
            mean_mse(pred[:1], torch.tensor([value]), args)


@pytest.mark.parametrize("values", [(-1., 0., 1.), (-.2, -.2), (0., 0.), (.1, .5), (-.999, -.001, .02)])
def test_signed_distribution_support_encoding_and_zero(values):
    rng = torch.get_rng_state().clone()
    args = configuration("distributional", values=values)
    assert torch.equal(rng, torch.get_rng_state())
    definition = args["outcome_distribution_definition"]
    assert definition == configuration("distributional", values=values)["outcome_distribution_definition"]
    atoms = torch.tensor(definition["atoms"], dtype=torch.double)
    assert atoms[0] == -1 and atoms[-1] == 1 and 0 in atoms
    target = torch.tensor([-1., -.987, -.003, 0., .03, .87, 1.], dtype=torch.double)
    encoded = encode_targets(target, atoms)
    torch.testing.assert_close(encoded.sum(-1), torch.ones_like(target))
    torch.testing.assert_close((encoded * atoms).sum(-1), target)
    logits = torch.randn(len(target), len(atoms), dtype=torch.double, requires_grad=True)
    distribution_loss(logits, target, args).backward()
    assert torch.isfinite(logits.grad).all()
    probs = categorical_probabilities(logits.detach(), args)
    explicit = (probs * (atoms - target[:, None]).abs()).sum(-1) - .5 * (
        probs[:, :, None] * probs[:, None, :] * (atoms[:, None] - atoms).abs()).sum((1, 2))
    torch.testing.assert_close(crps(logits.detach(), target, args), explicit)
    zero_logits = torch.full((2, len(atoms)), -float("inf"))
    torch.testing.assert_close(expected_values(zero_logits, args), torch.zeros(2))
    assert (categorical_probabilities(zero_logits, args)[:, (atoms == 0).nonzero().item()] == 1).all()
    assert torch.isnan(expected_values(torch.full_like(zero_logits, float("nan")), args)).all()
    with pytest.raises(ValueError, match="BCE"):
        mean_bce(logits, target, args)


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_epoch_selection_reports_and_exports(tmp_path, mode, task):
    from models.utils import run_epoch
    from test import write_outcome_evaluation_artifacts
    from scripts.comparison_utils import metrics, paired_bootstrap
    args = configuration(mode, task)
    args.update(gnn_task="node_binary", include_out=False, lambda_l1=0., clip=10., print_freq=99,
                export_outcome_distributions=True, use_xg=False, use_epv=False)
    samples = []
    targets = torch.tensor([-.8, -.1, 0., .2, -.3, .4])
    for i, target in enumerate(targets):
        x = torch.zeros(3, 26); x[:, 3] = torch.tensor([10., 40., 80.])
        graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 2),
                     node_ids=["home_1", "home_2", "home_3"], evaluation_match_id=f"m{i // 3}", evaluation_source_row=i)
        label = torch.zeros(len(LABEL_COLUMNS))
        label[LABEL_INDEX["action_index"]] = i
        label[LABEL_INDEX["intent_index"]] = i % 3
        label[LABEL_INDEX["success"]] = i % 2
        label[LABEL_INDEX["scores_xt"]] = target
        label[LABEL_INDEX["concedes_xt"]] = target
        label[LABEL_INDEX["scores_goal_next10"]] = i % 2
        label[LABEL_INDEX["concedes_goal_next10"]] = i % 2
        samples.append((graph, label, torch.tensor(1. + i / 10)))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shape = (3, 2) if mode == "scalar" else (3, 2, args["outcome_distribution_definition"]["effective_atoms"])
            self.weight = torch.nn.Parameter(torch.linspace(-.6, .3, int(np.prod(shape))).reshape(shape))
        def forward(self, graphs, destinations=None):
            return self.weight.repeat((graphs.num_graphs,) + (1,) * (self.weight.ndim - 1))
    model = Model()
    result, rows, diagnostics = run_epoch(SimpleNamespace(**args), model, DataLoader(samples, batch_size=3),
        device="cpu", return_outcome_evaluation=True, return_learning_curve_rows=True, return_semantic_prediction_rows=True)
    expected = expected_values(model.weight[torch.arange(6) % 3, torch.arange(6) % 2], args).detach()
    np.testing.assert_allclose(diagnostics["outcome"]["prediction"], expected.numpy(), atol=1e-7)
    assert result["mse_loss"] == pytest.approx(float(((expected - targets).square() * (1 + torch.arange(6) / 10)).mean()))
    assert "ce_loss" not in result and "brier" not in result and "xt_target_soft_bce" not in result
    assert all(row["outcome_domain"] == "signed_gain" for row in rows)
    frame = pd.DataFrame(rows)
    native = metrics(task, frame)
    assert "xt_rmse" in native and "brier" not in native and "xt_soft_bce" not in native
    paired_bootstrap(task, [frame, frame.copy()], resamples=3, seed=2)
    _, report = write_outcome_evaluation_artifacts(tmp_path, model_id=task + "/signed", task=task,
        outcome_evaluation=diagnostics["outcome"], outcome_bootstrap_resamples=3)
    assert "brier" not in report and "soft_bce" not in report
    assert (tmp_path / "outcome_learning_contract.json").is_file()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    run_epoch(SimpleNamespace(**args), model, DataLoader(samples, batch_size=3), optimizer, "cpu", train=True)
    assert torch.isfinite(model.weight).all()


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_real_cli_signed_training(tmp_path, monkeypatch, mode, task):
    import project_config as pc
    import test_all_task_training as fixture
    from training_state import load_checkpoint, CHECKPOINT_NAME
    original_source, original_run = fixture.source, runpy.run_path
    def signed_source(*args, **kwargs):
        graph, label = original_source(*args, **kwargs)
        label[LABEL_INDEX["scores_xt"]] = -.2
        label[LABEL_INDEX["concedes_xt"]] = -.1
        return graph, label
    def run_signed(*args, **kwargs):
        monkeypatch.setattr(pc, "load_feature_run_metadata", lambda *a, **kw: metadata())
        sys.argv.extend(["--use_xt", "--return_type", "next_10", "--learning-mode", mode, "--distribution-atoms", "7"])
        return original_run(*args, **kwargs)
    monkeypatch.setattr(fixture, "source", signed_source)
    monkeypatch.setattr(runpy, "run_path", run_signed)
    fixture.test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, task, "gat")
    directory = tmp_path / "saved" / task / "verification"
    checkpoint = load_checkpoint(directory / CHECKPOINT_NAME)
    assert checkpoint["args"]["outcome_learning_contract"]["domain"] == "signed_gain"
    assert checkpoint["metadata"]["checkpoint_monitor"] == "mse_loss"
    assert "mse_loss" in checkpoint["metadata"]["best_valid_metrics"]
    args = json.loads((directory / "args.json").read_text())
    saved = json.loads((directory / "metadata.json").read_text())
    validate_learning_metadata(args, saved)
    import models.utils as utils
    monkeypatch.setattr(utils, "get_model_path", lambda *a: directory)
    loaded = utils.load_model(task + "/verification", device="cpu")
    assert loaded.args["outcome_learning_contract"] == args["outcome_learning_contract"]
    for key, value in torch.load(directory / "best_weights.pt", weights_only=False).items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_signed_resume_equivalence(tmp_path, monkeypatch, mode):
    import faulthandler
    from pathlib import Path
    import models.utils as utils
    import project_config as pc
    import test_all_task_training as fixture
    from training_state import load_checkpoint, CHECKPOINT_NAME
    original_source, original_run, real_epoch = fixture.source, runpy.run_path, utils.run_epoch
    def signed_source(*args, **kwargs):
        graph, label = original_source(*args, **kwargs)
        label[LABEL_INDEX["scores_xt"]] = -.2
        label[LABEL_INDEX["concedes_xt"]] = -.1
        return graph, label
    def run_signed(*args, **kwargs):
        monkeypatch.setattr(pc, "load_feature_run_metadata", lambda *a, **kw: metadata())
        sys.argv.extend(["--use_xt", "--return_type", "next_10", "--learning-mode", mode,
                         "--distribution-atoms", "7", "--n_epochs", "2"])
        return original_run(*args, **kwargs)
    monkeypatch.setattr(fixture, "source", signed_source)
    monkeypatch.setattr(runpy, "run_path", run_signed)
    baseline_root, resumed_root = tmp_path / "baseline", tmp_path / "resumed"
    baseline_root.mkdir(); resumed_root.mkdir()
    fixture.test_fresh_cli_all_tasks_default_to_disk(baseline_root, monkeypatch, "outcome_conceding", "gat")
    baseline = load_checkpoint(baseline_root / "saved/outcome_conceding/verification" / CHECKPOINT_NAME)
    def interrupt(args, *pos, **kw):
        if args._epoch == 2:
            raise RuntimeError("signed interruption")
        return real_epoch(args, *pos, **kw)
    monkeypatch.setattr(utils, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="signed interruption"):
        fixture.test_fresh_cli_all_tasks_default_to_disk(resumed_root, monkeypatch, "outcome_conceding", "gat")
    path = resumed_root / "saved/outcome_conceding/verification" / CHECKPOINT_NAME
    interrupted_definition = load_checkpoint(path)["args"].get("outcome_distribution_definition")
    monkeypatch.setattr(utils, "run_epoch", real_epoch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume-checkpoint", str(path)])
    try:
        original_run(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        faulthandler.disable()
    resumed = load_checkpoint(path)
    assert resumed["finished"]
    assert resumed["args"]["outcome_learning_contract"] == baseline["args"]["outcome_learning_contract"]
    assert resumed["args"].get("outcome_distribution_definition") == interrupted_definition
    if mode == "distributional":
        assert resumed["args"]["outcome_distribution_definition"]["atoms"] == baseline["args"]["outcome_distribution_definition"]["atoms"]
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    for key, value in baseline["optimizer"]["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)


from test_training_refit import sources


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_refit_preserves_signed_contract(sources, mode):
    from training_refit import prepare_refit
    from models.outcome_learning import distribution_identity
    create, _ = sources
    directory, args, saved = create("outcome_scoring/source")
    args.update(configuration(mode))
    saved.update(distribution_identity(args))
    (directory / "args.json").write_text(json.dumps(args))
    (directory / "metadata.json").write_text(json.dumps(saved))
    refit = prepare_refit("outcome_scoring/source", "refit")
    assert refit["args"]["outcome_learning_contract"] == args["outcome_learning_contract"]
    assert refit["args"].get("outcome_distribution_definition") == args.get("outcome_distribution_definition")


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_signed_inference_and_component_values(tmp_path, mode, task):
    from test_pc_xpass_auxiliary import auxiliary_model
    from test_physical_xpass import make_physical_inference_match
    from models.outcome_learning import configure_distribution_exports, write_distribution_exports
    from inference import inference_gnn
    from datatools.viz_helpers import compute_outcome, compute_pass_score
    match = make_physical_inference_match([7, 8], [1, 1])
    for graph in match.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros(graph.num_nodes, 1)], 1)
    model = auxiliary_model(task, model_id=task + "/fixture")
    model.args.update(configuration(mode, task))
    k = model.args.get("outcome_distribution_definition", {}).get("effective_atoms")
    def forward(batch):
        if mode == "scalar":
            return batch.x.new_tensor([-.3, .2]).expand(batch.num_nodes, 2)
        out = batch.x.new_zeros((batch.num_nodes, 2, k))
        out[:, 0, 0] = 5.
        out[:, 1, -1] = 4.
        return out
    model.forward = forward
    configure_distribution_exports({task: model}, True)
    failure, success = inference_gnn(match, model, device="cpu")
    assert (failure.stack() < 0).all() and (success.stack() > 0).all()
    net = compute_outcome(failure, success)
    assert (net.stack() < 0).all()
    combined = compute_pass_score(success * 0, success, failure, success * 0, failure * 0)
    pd.testing.assert_frame_equal(combined, failure)
    if mode == "distributional":
        write_distribution_exports(tmp_path, {task: model})
        rows = pd.read_parquet(tmp_path / "outcome_distributions.parquet")
        atoms = np.array(model.args["outcome_distribution_definition"]["atoms"])
        for row in rows.to_dict("records"):
            assert np.dot(row["probabilities"], atoms) == pytest.approx(row["prediction"], abs=1e-7)
            table = success if row["execution_branch"] else failure
            assert table.at[row["action_index"], row["candidate_id"]] == pytest.approx(row["prediction"], abs=1e-7)


def test_mixed_comparison_keeps_individual_metrics_and_pairs_only_shared_metrics():
    from scripts.comparison_utils import paired_bootstrap
    signed = pd.DataFrame(dict(match_id=["a", "a", "b", "b"], action_index=[0, 1, 2, 3],
        target=[0, 1, 0, 1], soft_target=[-.2, .2, -.4, .4], prediction=[-.1, .1, -.3, .3],
        outcome_domain="signed_gain"))
    probability = signed.drop(columns="outcome_domain").copy()
    probability["soft_target"] = [0., .2, 0., .4]
    probability["prediction"] = [.1, .2, .3, .4]
    points, intervals, differences = paired_bootstrap("outcome_scoring", [signed, probability], resamples=3, seed=1)
    assert "brier" not in points[0] and "brier" in points[1]
    assert (1, "brier") in intervals and (0, 1, "brier") not in differences
    assert (0, 1, "xt_rmse") in differences


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_legacy_losses_predictions_and_optimizer_updates_are_identical(mode):
    import subprocess
    import types
    import models.utils as current
    from test_outcome_learning import configuration as probability_configuration
    source = subprocess.check_output(["git", "show", "HEAD:models/utils.py"], text=True)
    original = types.ModuleType("original_utils")
    exec(compile(source, "original_utils", "exec"), original.__dict__)
    args = probability_configuration()
    if mode == "scalar":
        args.pop("outcome_distribution_definition")
        args.update(learning_mode="scalar", checkpoint_monitor="ce_loss")
    args.update(gnn_task="node_binary", include_out=False, lambda_l1=.001, clip=10., print_freq=99,
                use_xt=False, use_xg=False, use_epv=False)
    configure_outcome_learning(args, metadata(clipped=True))
    k = args.get("outcome_distribution_definition", {}).get("effective_atoms")
    samples = []
    for i in range(4):
        graph = Data(x=torch.zeros(3, 26), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 2))
        label = torch.zeros(len(LABEL_COLUMNS))
        label[LABEL_INDEX["intent_index"]] = i % 3
        label[LABEL_INDEX["success"]] = i % 2
        label[LABEL_INDEX["concedes"]] = [0., .1, .6, 1.][i]
        label[LABEL_INDEX["concedes_goal_next10"]] = i % 2
        samples.append((graph, label, torch.tensor(.5 + i / 10)))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shape = (3, 2) if mode == "scalar" else (3, 2, k)
            self.weight = torch.nn.Parameter(torch.linspace(-.6, .3, int(np.prod(shape))).reshape(shape))
        def forward(self, graphs, destinations=None):
            return self.weight.repeat((graphs.num_graphs,) + (1,) * (self.weight.ndim - 1))
    old, new = Model(), Model()
    outputs = []
    for module, model in ((original, old), (current, new)):
        outputs.append(module.run_epoch(SimpleNamespace(**args), model, DataLoader(samples, batch_size=2),
            torch.optim.SGD(model.parameters(), lr=.03), "cpu", train=True, return_outcome_evaluation=True))
    torch.testing.assert_close(old.weight, new.weight, atol=0, rtol=0)
    assert set(outputs[0][0]) == set(outputs[1][0])
    for key, value in outputs[0][0].items():
        np.testing.assert_allclose(value, outputs[1][0][key], atol=0, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(outputs[0][1]["prediction"], outputs[1][1]["prediction"])
