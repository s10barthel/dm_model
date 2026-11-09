"""Distributional outcomes preserve value units and execution-branch semantics."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from models.outcome_learning import (normalize_learning_settings, fit_definition,
    validate_definition, encode_targets, expected_values, distribution_loss, mean_bce,
    crps, categorical_probabilities, validate_learning_metadata)


def configuration(task="outcome_conceding", values=(0., .001, .01, .15, 1.), count=7):
    args = dict(task=task, learning_mode="distributional", distribution_atoms=count)
    normalize_learning_settings(args)
    labels = torch.zeros(len(values), len(LABEL_COLUMNS))
    labels[:, LABEL_INDEX["concedes"]] = torch.tensor(values)
    labels[:, LABEL_INDEX["scores"]] = torch.tensor(values)
    args["outcome_distribution_definition"] = fit_definition([labels], args, {"training_match_ids": ["train"]})
    return args


def test_two_hot_preserves_values_and_probability_mass():
    atoms = torch.tensor([0., .00001, .001, .03, .5, 1.], dtype=torch.double)
    values = torch.tensor([0., 1e-10, .00001, .0002, .03, .2, .9999, 1.], dtype=torch.double)
    encoded = encode_targets(values, atoms)
    torch.testing.assert_close(encoded.sum(-1), torch.ones_like(values))
    torch.testing.assert_close((encoded * atoms).sum(-1), values, atol=1e-15, rtol=1e-12)
    assert (encoded > 0).sum(-1).max() <= 2


@pytest.mark.parametrize("values", [(0., 0.), (0., 1.), (.001, .001), (.000001, .01, .9)])
def test_deterministic_support_and_frozen_provenance(values):
    rng = torch.get_rng_state().clone()
    args = configuration(values=values)
    assert torch.equal(rng, torch.get_rng_state())
    definition = args["outcome_distribution_definition"]
    assert definition == configuration(values=values)["outcome_distribution_definition"]
    assert definition["atoms"][0] == 0 and definition["atoms"][-1] == 1
    assert len(set(definition["atoms"])) == len(definition["atoms"])
    assert len(definition["atoms"]) <= 7
    validate_definition(definition)
    corrupt = copy.deepcopy(definition)
    corrupt["atoms"][0] = .1
    with pytest.raises(ValueError):
        validate_definition(corrupt)


@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf")])
def test_invalid_labels_are_not_clipped(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        configuration(values=(value,))


def test_losses_gradients_and_extreme_logits():
    args = configuration()
    k = args["outcome_distribution_definition"]["effective_atoms"]
    logits = torch.randn(3, 2, k, dtype=torch.double, requires_grad=True)
    selected = logits[torch.arange(3), torch.tensor([0, 1, 0])]
    y = torch.tensor([0., .002, 1.], dtype=torch.double)
    loss = distribution_loss(selected, y, args, torch.tensor([1., .5, 2.]))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) == 0
    probs = categorical_probabilities(selected.detach())
    atoms = torch.tensor(args["outcome_distribution_definition"]["atoms"], dtype=torch.double)
    explicit = (probs * (atoms - y[:, None]).abs()).sum(-1) - .5 * (
        probs[:, :, None] * probs[:, None, :] * (atoms[:, None] - atoms).abs()).sum((1, 2))
    torch.testing.assert_close(crps(selected.detach(), y, args), explicit)
    extreme = torch.full((2, k), -10000., dtype=torch.double)
    extreme[0, 0] = 10000
    extreme[1, -1] = 10000
    assert torch.isfinite(mean_bce(extreme, torch.tensor([1., 0.]), args))
    zero = expected_values(torch.full((2, k), -float("inf")), args)
    torch.testing.assert_close(zero, torch.zeros_like(zero))


def test_binary_distribution_matches_scalar_bce():
    args = configuration(values=(0., 1.))
    scalar_logits = torch.tensor([-2., 0., 3.], dtype=torch.double)
    logits = torch.stack([torch.zeros_like(scalar_logits), scalar_logits], -1)
    y = torch.tensor([0., 1., 1.], dtype=torch.double)
    torch.testing.assert_close(expected_values(logits, args), scalar_logits.sigmoid())
    torch.testing.assert_close(mean_bce(logits, y, args),
                               torch.nn.functional.binary_cross_entropy_with_logits(scalar_logits, y))


def test_saved_mode_and_definition_cannot_be_reinterpreted():
    args = configuration()
    restored = dict(task=args["task"], learning_mode=None, distribution_atoms=None)
    normalize_learning_settings(restored, args)
    assert restored["outcome_distribution_definition"] == args["outcome_distribution_definition"]
    with pytest.raises(ValueError, match="learning mode"):
        normalize_learning_settings(dict(task=args["task"], learning_mode="scalar"), args)
    with pytest.raises(ValueError, match="atoms"):
        normalize_learning_settings(dict(task=args["task"], distribution_atoms=9), args)
    with pytest.raises(ValueError, match="metadata"):
        validate_learning_metadata(args, {"learning_mode": "scalar"})


@pytest.mark.parametrize("task", ["pass_success", "outcome_return", "pass_intent"])
def test_unsupported_tasks_rejected(task):
    with pytest.raises(ValueError, match="only"):
        normalize_learning_settings(dict(task=task, learning_mode="distributional"))


@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_real_cli_training_checkpoint_and_cache_lifecycle(tmp_path, monkeypatch, task):
    import runpy
    import sys
    from test_all_task_training import test_fresh_cli_all_tasks_default_to_disk
    from training_state import load_checkpoint, CHECKPOINT_NAME
    original = runpy.run_path
    def run_distributional(*args, **kwargs):
        sys.argv.extend(["--learning-mode", "distributional", "--distribution-atoms", "7"])
        return original(*args, **kwargs)
    monkeypatch.setattr(runpy, "run_path", run_distributional)
    test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, task, "relationship-attention")
    checkpoint = load_checkpoint(tmp_path / "saved" / task / "verification" / CHECKPOINT_NAME)
    assert checkpoint["args"]["learning_mode"] == "distributional"
    assert checkpoint["args"]["checkpoint_monitor"] == "mean_soft_bce"
    assert checkpoint["metadata"]["best_valid_metrics"]["distributional_ce"] >= 0
    assert checkpoint["metadata"]["best_valid_metrics"]["mean_soft_bce"] == checkpoint["metadata"]["best_valid_metrics"]["ce_loss"]


@pytest.mark.parametrize("architecture", ["gat", "transformer", "relationship", "relationship-attention"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_distributional_forward_shared_architectures(architecture, task):
    from models.gnn import GNN
    from test_benchmark_no_accel import make_model_args
    from test_goal_poss_dist import graph
    args = make_model_args()
    args.update(configuration(task))
    args.update(model=architecture, gnn_architecture=architecture, out_dim=2, gnn_task="node_binary", gnn_heads=2)
    model = GNN(args).eval()
    batch = Batch.from_data_list([graph(), graph()])
    out = model(batch)
    assert out.shape == (batch.num_nodes, 2, args["outcome_distribution_definition"]["effective_atoms"])
    mean = expected_values(out, args)
    assert mean.shape == (batch.num_nodes, 2)
    assert bool(((mean >= 0) & (mean <= 1)).all())
    out.sum().backward()
    assert model.decoder.nodewise_mlp[-1].weight.grad is not None


@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_inference_candidates_branches_and_sidecar_match(tmp_path, task):
    import numpy as np
    import pandas as pd
    from test_pc_xpass_auxiliary import auxiliary_model
    from test_physical_xpass import make_physical_inference_match
    from models.outcome_learning import configure_distribution_exports, write_distribution_exports
    from inference import inference_gnn
    match = make_physical_inference_match([7, 8], [1, 1])
    for graph in match.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros(graph.num_nodes, 1)], 1)
    model = auxiliary_model(task, model_id=task + "/fixture")
    model.args.update(configuration(task))
    k = model.args["outcome_distribution_definition"]["effective_atoms"]
    def forward(batch):
        out = batch.x.new_zeros((batch.num_nodes, 2, k))
        out[:, 0, 0] = 3.
        out[:, 1, -1] = 2.
        return out
    model.forward = forward
    configure_distribution_exports({task: model}, True)
    failure, success = inference_gnn(match, model, device="cpu")
    assert len(model.outcome_distribution_rows) == len(failure) * len(failure.columns) * 2
    write_distribution_exports(tmp_path, {task: model})
    rows = pd.read_parquet(tmp_path / "outcome_distributions.parquet")
    atoms = np.array(model.args["outcome_distribution_definition"]["atoms"])
    for row in rows.to_dict("records"):
        expected = float(np.dot(row["probabilities"], atoms))
        table = success if row["execution_branch"] else failure
        assert expected == pytest.approx(row["prediction"], abs=1e-7)
        assert table.at[row["action_index"], row["candidate_id"]] == pytest.approx(expected, abs=1e-7)


def test_resume_equivalence_with_fitted_support(tmp_path, monkeypatch):
    import runpy, sys, faulthandler
    import models.utils as utils
    from test_all_task_training import test_fresh_cli_all_tasks_default_to_disk
    from training_state import load_checkpoint, CHECKPOINT_NAME
    original_run = runpy.run_path
    real_epoch = utils.run_epoch
    def run_distributional(*args, **kwargs):
        sys.argv.extend(["--learning-mode", "distributional", "--distribution-atoms", "7", "--n_epochs", "2"])
        return original_run(*args, **kwargs)
    monkeypatch.setattr(runpy, "run_path", run_distributional)
    baseline_root, resumed_root = tmp_path / "baseline", tmp_path / "resumed"
    baseline_root.mkdir(); resumed_root.mkdir()
    test_fresh_cli_all_tasks_default_to_disk(baseline_root, monkeypatch, "outcome_conceding", "gat")
    baseline = load_checkpoint(baseline_root / "saved/outcome_conceding/verification" / CHECKPOINT_NAME)
    def interrupt(args, *pos, **kw):
        if args._epoch == 2:
            raise RuntimeError("distribution interruption")
        return real_epoch(args, *pos, **kw)
    monkeypatch.setattr(utils, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="distribution interruption"):
        test_fresh_cli_all_tasks_default_to_disk(resumed_root, monkeypatch, "outcome_conceding", "gat")
    faulthandler.disable()
    checkpoint_path = resumed_root / "saved/outcome_conceding/verification" / CHECKPOINT_NAME
    interrupted = load_checkpoint(checkpoint_path)
    assert not interrupted["finished"]
    support = interrupted["args"]["outcome_distribution_definition"]
    monkeypatch.setattr(utils, "run_epoch", real_epoch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume-checkpoint", str(checkpoint_path)])
    try:
        original_run(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        faulthandler.disable()
    resumed = load_checkpoint(checkpoint_path)
    assert resumed["finished"]
    assert resumed["args"]["outcome_distribution_definition"] == support
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    for key, value in baseline["optimizer"]["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, resumed["optimizer"]["state"][key][name], rtol=0, atol=0)


from test_training_refit import sources


def test_source_refit_inherits_fitted_support(sources):
    import json
    from training_refit import prepare_refit
    create, _ = sources
    directory, args, metadata = create("outcome_scoring/source")
    args.update(configuration("outcome_scoring"))
    metadata.update(learning_mode="distributional", outcome_distribution_definition=args["outcome_distribution_definition"])
    (directory / "args.json").write_text(json.dumps(args))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    refit = prepare_refit("outcome_scoring/source", "refit")
    assert refit["args"]["outcome_distribution_definition"] == args["outcome_distribution_definition"]
    assert refit["args"]["learning_mode"] == "distributional"


def test_evaluation_rows_and_distribution_sidecar(tmp_path):
    import numpy as np
    import pandas as pd
    from models.utils import run_epoch
    from test import write_outcome_evaluation_artifacts
    args = configuration()
    args.update(gnn_task="node_binary", include_out=False, lambda_l1=0., clip=10., print_freq=99,
                export_outcome_distributions=True, use_xt=False, use_xg=False, use_epv=False)
    k = args["outcome_distribution_definition"]["effective_atoms"]
    samples = []
    for i in range(6):
        x = torch.zeros(3, 26); x[:, 3] = torch.tensor([10., 40., 80.])
        graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 2),
                     node_ids=["home_1", "home_2", "home_3"], evaluation_match_id="m", evaluation_source_row=i)
        label = torch.zeros(len(LABEL_COLUMNS))
        label[LABEL_INDEX["intent_index"]] = i % 3
        label[LABEL_INDEX["success"]] = i % 2
        label[LABEL_INDEX["concedes"]] = [0., .001, .01, .15, 1., .02][i]
        label[LABEL_INDEX["concedes_goal_next10"]] = i % 2
        samples.append((graph, label, torch.tensor(1.)))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, graphs, destinations=None):
            return graphs.x.new_zeros((graphs.num_nodes, 2, k))
    metrics, evaluation = run_epoch(SimpleNamespace(**args), Model(), DataLoader(samples, batch_size=3),
                                   device="cpu", return_outcome_evaluation=True)
    assert metrics["mean_soft_bce"] == metrics["ce_loss"]
    assert len(evaluation["distribution_rows"]) == len(samples)
    _, report = write_outcome_evaluation_artifacts(tmp_path, model_id="outcome_conceding/fixture",
        task="outcome_conceding", outcome_evaluation=evaluation, outcome_bootstrap_resamples=0)
    assert "build_up_success" in set(report.stratum)
    rows = pd.read_parquet(tmp_path / "outcome_distributions.parquet")
    atoms = np.asarray(args["outcome_distribution_definition"]["atoms"])
    np.testing.assert_allclose([np.dot(p, atoms) for p in rows.probabilities], rows.prediction, atol=1e-7)


def test_scalar_decoder_matches_original_checkpoint_code():
    import subprocess, types
    from models.gnn import GNN
    from test_benchmark_no_accel import make_model_args
    from test_goal_poss_dist import graph
    # Execute the repository's original decoder in isolation; keep the user's working files intact.
    source = subprocess.check_output(["git", "show", "HEAD:models/gnn.py"], text=True)
    original = types.ModuleType("original_gnn")
    exec(compile(source, "original_gnn", "exec"), original.__dict__)
    args = make_model_args(); args.update(task="outcome_conceding", out_dim=2)
    torch.manual_seed(7); old = original.GNN(copy.deepcopy(args)).eval()
    torch.manual_seed(7); current = GNN(copy.deepcopy(args)).eval()
    assert list(old.state_dict()) == list(current.state_dict())
    batch = Batch.from_data_list([graph(), graph()])
    torch.testing.assert_close(old(batch), current(batch), rtol=0, atol=0)
