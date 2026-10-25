"""Training controls, including actual entry-point stopping and resume behavior."""
import argparse
import faulthandler
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from datatools import config
from training_state import (
    CHECKPOINT_NAME, add_training_control_arguments, load_checkpoint,
    lr_reduction_grace_complete, optimizer_snapshot, reduce_learning_rate,
    restore_training_control_defaults,
)


def control_parser(wrapper=False):
    parser = argparse.ArgumentParser()
    add_training_control_arguments(parser, wrapper=wrapper)
    return parser


@pytest.mark.parametrize("wrapper", [False, True])
def test_control_defaults_and_overrides(wrapper):
    parser = control_parser(wrapper)
    args = parser.parse_args([])
    assert args.lr_reduction_patience == 3 and args.lr_reduction_factor == .5
    assert args.lr_rollback and args.early_stopping_min_epochs_after_lr_reduction == 5
    assert args.seed == (100 if wrapper else 128)
    assert getattr(args, "max_epochs" if wrapper else "n_epochs") == (100 if wrapper else 200)
    args = parser.parse_args(["--lr-reduction-patience", "7", "--lr-reduction-factor", ".25",
        "--early-stopping-min-epochs-after-lr-reduction", "0", "--no-lr-rollback",
        "--max-epochs", "11", "--seed", str(2**32 - 1)])
    assert args.lr_reduction_patience == 7 and args.lr_reduction_factor == .25
    assert not args.lr_rollback and args.early_stopping_min_epochs_after_lr_reduction == 0
    assert args.seed == 2**32 - 1
    assert getattr(args, "max_epochs" if wrapper else "n_epochs") == 11
    assert parser.parse_args(["--seed", "0"]).seed == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--lr-rollback", "--no-lr-rollback"])


@pytest.mark.parametrize("flag,value", [
    ("--lr-reduction-patience", "0"), ("--lr-reduction-patience", "-1"),
    ("--lr-reduction-patience", "1.5"), ("--lr-reduction-factor", "0"),
    ("--lr-reduction-factor", "1"), ("--lr-reduction-factor", "1.1"),
    ("--lr-reduction-factor", "-0.1"), ("--lr-reduction-factor", "nan"),
    ("--lr-reduction-factor", "inf"),
    ("--early-stopping-min-epochs-after-lr-reduction", "-1"),
    ("--max-epochs", "0"), ("--max-epochs", "1.5"),
    ("--seed", "-1"), ("--seed", str(2**32)), ("--seed", "1.5"),
])
def test_invalid_controls(flag, value):
    for wrapper in (False, True):
        with pytest.raises(SystemExit):
            control_parser(wrapper).parse_args([flag, value])


def test_epoch_alias_and_legacy_defaults():
    parser = control_parser()
    assert parser.parse_args(["--n_epochs", "9"]).n_epochs == 9
    with pytest.raises(SystemExit):
        parser.parse_args(["--n_epochs", "9", "--max-epochs", "8"])
    legacy = SimpleNamespace(n_epochs=9, seed=128)
    restore_training_control_defaults(legacy)
    assert legacy.lr_reduction_patience == 3 and legacy.lr_reduction_factor == .5
    assert legacy.lr_rollback and legacy.early_stopping_min_epochs_after_lr_reduction == 0
    legacy.lr_rollback = False
    legacy.early_stopping_min_epochs_after_lr_reduction = 7
    restore_training_control_defaults(legacy)
    assert not legacy.lr_rollback and legacy.early_stopping_min_epochs_after_lr_reduction == 7


def test_wrapper_forwards_controls_and_rejects_resume_overrides():
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    args = make_training_args("feature_run", pass_success_ipw=False, pass_height=True)
    args.enabled_tasks.update(pass_height=True, failure_receiver=True)
    args.max_epochs, args.seed = 17, 42
    args.lr_reduction_patience, args.lr_reduction_factor = 7, .25
    args.lr_rollback, args.early_stopping_min_epochs_after_lr_reduction = False, 9
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        commands = wrapper.build_training_commands(args)[0]
    assert len(commands) == 8
    for base in commands:
        command = wrapper.append_training_control_flags(base, wrapper.get_training_control_settings(args))
        for flag, value in (("--n_epochs", "17"), ("--seed", "42"),
            ("--lr-reduction-patience", "7"), ("--lr-reduction-factor", "0.25"),
            ("--early-stopping-min-epochs-after-lr-reduction", "9")):
            assert wrapper.get_cli_value(command, flag) == value
        assert "--no-lr-rollback" in command
        # Existing refit selection still replaces only the epoch count.
        refit = wrapper._replace_cli_value(command, "--n_epochs", 3)
        assert wrapper.get_cli_value(refit, "--n_epochs") == "3"
        assert wrapper.get_cli_value(refit, "--seed") == "42"
    for flags in (["--lr-reduction-patience", "7"], ["--lr-reduction-factor", ".25"],
        ["--early-stopping-min-epochs-after-lr-reduction", "0"],
        ["--no-lr-rollback"], ["--max-epochs", "8"], ["--seed", "42"]):
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--resume-id", "pass_intent/a", *flags])


@pytest.mark.parametrize("rollback", [True, False])
def test_reduction_restores_or_preserves_adam_and_weights(rollback):
    torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    def update():
        optimizer.zero_grad()
        model(torch.ones(2, 2)).square().sum().backward()
        optimizer.step()
    update()
    best = optimizer_snapshot(model, optimizer)
    update()
    current = optimizer_snapshot(model, optimizer)
    lr = reduce_learning_rate(model, optimizer, best, .01, .001, .25, rollback)
    assert lr == .0025 and optimizer.param_groups[0]["lr"] == lr
    expected = best if rollback else current
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected["model"][name], atol=0, rtol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, expected["optimizer"]["state"][key][field], atol=0, rtol=0)
    assert reduce_learning_rate(model, optimizer, best, lr, .001, .25, rollback) == .001
    minimum_snapshot = optimizer_snapshot(model, optimizer)
    assert reduce_learning_rate(model, optimizer, None, .001, .001, .25, True) == .001
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, minimum_snapshot["model"][name], atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="Missing best"):
        reduce_learning_rate(model, optimizer, None, .01, .001, .5, True)


def test_grace_boundaries_and_renewal():
    assert lr_reduction_grace_complete(2, None, 5)
    assert lr_reduction_grace_complete(8, 8, 0)
    assert not lr_reduction_grace_complete(11, 8, 5)
    assert lr_reduction_grace_complete(12, 8, 5)
    assert not lr_reduction_grace_complete(12, 11, 5)
    assert lr_reduction_grace_complete(15, 11, 5)


@pytest.fixture
def synthetic_training(tmp_path, monkeypatch):
    """Actual CLI, datasets and checkpoint loop, with inexpensive Adam updates."""
    import project_config as pc
    import models.utils as utils
    from test_training_runtime import graph
    features, labels_dir = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels_dir.mkdir()
    ids = [f"m{i}" for i in range(5)]
    for match in ids:
        labels = torch.zeros(1, len(config.LABEL_COLUMNS))
        labels[:, config.LABEL_INDEX["is_pass"]] = 1
        labels[:, config.LABEL_INDEX["intent_index"]] = 1
        labels[:, 7] = 1
        from scripts.generate_relevant_features import append_interactions_to_graph
        raw = graph(5, 3)
        raw.edge_index = torch.cartesian_prod(torch.arange(5), torch.arange(5)).T.contiguous()
        raw.edge_attr = torch.ones((25, 2))
        torch.save([append_interactions_to_graph(raw)], features / f"{match}.pt")
        torch.save(labels, labels_dir / f"{match}.pt")
    manifest = {"manifest_id": "fixture", "train": ids, "test": [], "metadata": {"train_size": 5}}
    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: None)
    monkeypatch.setattr(pc, "resolve_feature_root", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(pc, "resolve_artifact_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(pc, "get_model_run_root", lambda task, run_id: tmp_path / "saved" / task / run_id)
    observations = []
    def epoch(args, model, loader, optimizer, device, pos_weight, train):
        if train:
            optimizer.zero_grad()
            loss = sum(p.square().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            observations.append((args._epoch, optimizer.param_groups[0]["lr"]))
        return {"ce_loss": 1.0}
    monkeypatch.setattr(utils, "run_epoch", epoch)
    base = ["train.py", "--task", "pass_intent", "--model", "gat", "--device", "cpu",
        "--feature_dir", str(features), "--label_dir", str(labels_dir), "--dataset-loading", "memory",
        "--node_emb_dim", "8", "--graph_emb_dim", "8", "--gnn_heads", "2",
        "--mlp_h1_dim", "8", "--mlp_h2_dim", "4", "--no-goal-nodes",
        "--start_lr", ".002", "--min_lr", ".001", "--early-stopping-min-epochs", "1",
        "--early-stopping-patience", "4", "--max-epochs", "15", "--seed", "42"]
    entry = Path(__file__).resolve().parents[1] / "train.py"
    def run(run_id, flags=(), resume=False):
        checkpoint = tmp_path / "saved/pass_intent" / run_id / CHECKPOINT_NAME
        argv = ["train.py", "--resume-checkpoint", str(checkpoint), *flags] if resume else [*base, "--run-id", run_id, *flags]
        monkeypatch.setattr(sys, "argv", argv)
        try:
            runpy.run_path(str(entry), run_name="__main__")
        finally:
            faulthandler.disable()
        return load_checkpoint(checkpoint)
    return run, observations, epoch, utils, tmp_path


@pytest.mark.parametrize("flags,last_epoch,reduction_epoch", [
    ([], 9, 5),
    (["--early-stopping-min-epochs-after-lr-reduction", "0"], 5, 5),
    (["--max-epochs", "6"], 6, 5),
    (["--start_lr", ".001"], 5, None),
    (["--early-stopping-patience", "3"], 4, None),
    (["--max-epochs", "2", "--early-stopping-min-epochs", "30"], 2, None),
    (["--no-early-stopping", "--max-epochs", "7"], 7, 5),
    (["--lr-reduction-patience", "1", "--lr-reduction-factor", ".75", "--max-epochs", "15"], 9, 5),
    (["--lr-reduction-patience", "2"], 8, 4),
])
def test_actual_stopping_and_caps(synthetic_training, flags, last_epoch, reduction_epoch):
    run, observations, _, _, _ = synthetic_training
    checkpoint = run("stopping", flags)
    metadata = checkpoint["metadata"]
    assert metadata["last_epoch"] == last_epoch
    assert metadata["last_lr_reduction_start_epoch"] == reduction_epoch
    assert metadata["epochs_since_loss_improvement"] == last_epoch - 1
    assert metadata["best_epoch"] == 1
    assert checkpoint["args"]["seed"] == 42
    if reduction_epoch:
        assert observations[reduction_epoch - 1][1] < .002
    if "--max-epochs" in flags:
        log = (synthetic_training[-1] / "saved/pass_intent/stopping/log.txt").read_text()
        cap = int(flags[flags.index("--max-epochs") + 1])
        assert ("Hard epoch cap reached" in log) == (last_epoch == cap)


@pytest.mark.parametrize("rollback", [True, False])
def test_resume_after_reduction_preserves_optimizer_and_grace(synthetic_training, monkeypatch, rollback):
    run, _, epoch, utils, tmp_path = synthetic_training
    flags = ["--lr-rollback" if rollback else "--no-lr-rollback", "--no-speed"]
    baseline = run("baseline", flags)
    def interrupted(args, *a, **kw):
        if args._epoch == 6:
            raise RuntimeError("interruption after reduction")
        return epoch(args, *a, **kw)
    monkeypatch.setattr(utils, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        run("resumed", flags)
    saved = load_checkpoint(tmp_path / "saved/pass_intent/resumed" / CHECKPOINT_NAME)
    assert saved["args"]["speed_aware"] is False
    assert saved["metadata"]["last_epoch"] == 5
    assert saved["metadata"]["last_lr_reduction_start_epoch"] == 5
    monkeypatch.setattr(utils, "run_epoch", epoch)
    resumed = run("resumed", resume=True)
    assert resumed["metadata"]["last_epoch"] == baseline["metadata"]["last_epoch"]
    assert resumed["metadata"]["last_lr_reduction_start_epoch"] == 5
    assert resumed["args"]["lr_rollback"] == rollback
    assert resumed["args"]["speed_aware"] is False
    assert resumed["metadata"]["feature_signature"]["speed_aware"] is False
    for name, tensor in baseline["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], atol=0, rtol=0)
    for key, state in baseline["optimizer"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][key][field], atol=0, rtol=0)
    assert all(int(state["step"]) == (6 if rollback else 9) for state in resumed["optimizer"]["state"].values())


def test_actual_final_refit_ignores_validation_controls(synthetic_training):
    run, observations, _, _, _ = synthetic_training
    checkpoint = run("refit", ["--final-refit", "--max-epochs", "3", "--lr-reduction-patience", "1"])
    assert checkpoint["metadata"]["train_match_count"] == 5
    assert checkpoint["metadata"]["validation_match_count"] == 0
    assert checkpoint["metadata"]["last_epoch"] == checkpoint["metadata"]["best_epoch"] == 3
    assert checkpoint["metadata"]["last_lr_reduction_start_epoch"] is None
    assert observations == [(1, .002), (2, .002), (3, .002)]


def test_actual_cli_rejects_epoch_alias_conflict_and_resume_controls(synthetic_training):
    run, _, _, _, _ = synthetic_training
    with pytest.raises(SystemExit):
        run("conflicting", ["--n_epochs", "8"])
    # Parsing forbids training overrides before it attempts to open a checkpoint.
    with pytest.raises(SystemExit):
        run("missing", ["--lr-reduction-factor", ".25"], resume=True)


@pytest.mark.parametrize("mode", [None, "none", "delaunay"])
@pytest.mark.parametrize("use_carries", [False, True])
def test_wrapper_forwards_sparsify_to_every_task(mode, use_carries):
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    args = make_training_args("feature_run", pass_success_ipw=False, use_carries=use_carries)
    args.enabled_tasks.update(pass_height=True, failure_receiver=True)
    if mode is not None:
        args.sparsify = mode
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        commands = wrapper.build_training_commands(args)[0]
    assert {wrapper.get_cli_value(command, "--task") for command in commands} == set(args.enabled_tasks)
    for command in commands:
        assert command.count("--sparsify") == 1
        assert wrapper.get_cli_value(command, "--sparsify") == (mode or "none")


@pytest.mark.parametrize("mode", [None, "off", "on"])
def test_resume_validation_uses_invocation_default(synthetic_training, monkeypatch, mode):
    run, _, epoch, utils, tmp_path = synthetic_training
    def interrupt(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("validation resume fixture")
        return epoch(args, *a, **kw)
    monkeypatch.setattr(utils, "run_epoch", interrupt)
    with pytest.raises(RuntimeError, match="validation resume fixture"):
        run("validation_policy", ["--graph-validation", "off", "--max-epochs", "3"])
    path = tmp_path / "saved/pass_intent/validation_policy" / CHECKPOINT_NAME
    saved = load_checkpoint(path)
    assert saved["args"]["graph_validation"] is False
    monkeypatch.setattr(utils, "run_epoch", epoch)
    flags = [] if mode is None else ["--graph-validation", mode]
    resumed = run("validation_policy", flags, resume=True)
    assert resumed["args"]["graph_validation"] is (mode != "off")
    stats = resumed["metadata"]["dataset_loading"]["train"]["graph_validation"]
    assert stats["enabled"] is (mode != "off")
    assert bool(stats["checked_files"]) is (mode != "off")
