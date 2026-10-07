import argparse
import copy
import faulthandler
import json
from pathlib import Path
import random
import runpy
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from datatools import config
from models.node_selection import selection_layout, selection_loss_metrics, validate_graph_batch
from training_state import (GradientAccumulator, atomic_torch_save, load_checkpoint, make_checkpoint,
    optimizer_snapshot, restore_optimizer_snapshot, capture_rng, restore_rng, resolve_resume_checkpoint,
    CHECKPOINT_NAME, dataset_signature, validate_learning_rates, add_training_runtime_arguments)
from training_monitor import TrainingMonitor


def test_accumulation_matches_large_batches_with_partial_group_and_regularization():
    torch.manual_seed(23)
    large = torch.nn.Linear(3, 2).double()
    small = copy.deepcopy(large)
    x, y = torch.randn(11, 3).double(), torch.randn(11, 2).double()
    opt_large = torch.optim.Adam(large.parameters(), lr=.003)
    opt_small = torch.optim.Adam(small.parameters(), lr=.003)
    for _ in range(2):
        for model, optimizer, batch, steps in ((large, opt_large, 6, 1), (small, opt_small, 2, 3)):
            accumulator = GradientAccumulator(optimizer, model.parameters(), steps, .2)
            for start in range(0, len(x), batch):
                end = min(start + batch, len(x))
                loss = (model(x[start:end]) - y[start:end]).square().mean()
                loss += .02 * sum(p.abs().sum() for p in model.parameters())
                accumulator.backward(loss, end - start)
            accumulator.flush()
    for a, b in zip(large.parameters(), small.parameters()):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert all(int(state["step"]) == 4 for state in opt_small.state.values())


def test_checkpoint_rng_rollback_and_atomic_failure(tmp_path):
    torch.manual_seed(21)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    model(torch.ones(2, 2)).sum().backward()
    optimizer.step()
    best = optimizer_snapshot(model, optimizer)
    state = make_checkpoint(model, optimizer, SimpleNamespace(seed=21), {"last_epoch": 1}, "data", best, None)
    path = tmp_path / CHECKPOINT_NAME
    atomic_torch_save(state, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    loaded = load_checkpoint(path)
    restore_rng(loaded["rng"])
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)
    optimizer.zero_grad()
    model(torch.zeros(2, 2)).sum().backward()
    optimizer.step()
    restore_optimizer_snapshot(best, model, optimizer, lr=.005)
    assert optimizer.param_groups[0]["lr"] == .005
    assert all(int(s["step"]) == 1 for s in optimizer.state.values())
    for k, value in model.state_dict().items():
        torch.testing.assert_close(value, best["model"][k])
    with patch("training_state.os.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            atomic_torch_save({"broken": True}, path)
    assert load_checkpoint(path)["metadata"]["last_epoch"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def graph(n, teammate_count):
    x = torch.zeros(n, config.NODE_FEATURE_CORE_DIM)
    x[:teammate_count, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[:, config.NODE_FEATURE_X] = torch.arange(n) + 20
    x[:, config.NODE_FEATURE_Y] = 34
    return Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 2))


@pytest.mark.parametrize("task,include_out", [("pass_intent", False), ("action_intent", False),
    ("success_intent", False), ("success_receiver", False), ("failure_receiver", False),
    ("failure_receiver", True), ("pass_receiver", True)])
@pytest.mark.parametrize("ties", [False, True])
def test_grouped_selection_matches_scalar_loss_gradients_and_ranking(task, include_out, ties):
    graphs = Batch.from_data_list([graph(5, 3), graph(6, 4), graph(5, 3)])
    labels = torch.zeros(3, len(config.LABEL_COLUMNS))
    labels[:, 4] = torch.tensor([5, 6, 5])
    labels[:, 5] = torch.tensor([1, 2, 0])
    labels[:, 6] = torch.tensor([3, 4, 3]) if task == "failure_receiver" else torch.tensor([1, 2, 0])
    if include_out:
        labels[1, 6] = -1
    layout = selection_layout(graphs, labels, task, include_out)
    torch.manual_seed(3)
    logits = torch.randn(graphs.num_nodes + (3 if include_out else 0), dtype=torch.double)
    if ties:
        logits.zero_()
    logits.requires_grad_()
    loss, predictions, targets, ranks, probabilities = selection_loss_metrics(logits, layout)
    scalar_losses = []
    for ids, indices, group_targets in layout:
        for gi, idx, target in zip(ids, indices, group_targets):
            values = logits[idx]
            scalar_losses.append(torch.nn.functional.cross_entropy(values[None], target[None]))
            assert predictions[gi] == values.argmax()
            rank = (values.argsort(descending=True) == target).nonzero()[0].item() + 1
            assert ranks[gi].item() == pytest.approx(1 / rank)
            torch.testing.assert_close(probabilities[gi], values.softmax(0)[target])
    expected = torch.stack(scalar_losses).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0])
    assert not ranks.requires_grad and not probabilities.requires_grad


def test_cpu_validation_rejects_invalid_samples():
    batch = Batch.from_data_list([graph(5, 3)])
    labels = torch.zeros(1, len(config.LABEL_COLUMNS))
    labels[0, 5] = 3
    with pytest.raises(ValueError, match="outside 3 candidates"):
        selection_layout(batch, labels, "pass_intent", False)
    labels[0, 5] = float("nan")
    with pytest.raises(ValueError, match="finite integers"):
        selection_layout(batch, labels, "pass_intent", False)
    batch.edge_index[0, 0] = 5
    with pytest.raises(ValueError, match="out of bounds"):
        validate_graph_batch(batch)


def test_runtime_options_and_resume_conflicts():
    from scripts.train_relevant_models import parse_args
    parsed = parse_args(["--resume-id", "pass_intent/a", "--monitoring", "on"])
    assert parsed.monitoring == "on" and parsed.accumulation_steps == 1
    for flag, value in (("--batch-size", "128"), ("--start_lr", ".001"), ("--accumulation-steps", "2")):
        with pytest.raises(SystemExit):
            parse_args(["--resume-id", "pass_intent/a", flag, value])
    parser = argparse.ArgumentParser()
    add_training_runtime_arguments(parser)
    assert parser.parse_args([]).monitoring == "off"
    for value in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--accumulation-steps", value])
    for start, minimum in ((0, .1), (.01, .1), (float("nan"), .01)):
        with pytest.raises(ValueError):
            validate_learning_rates(start, minimum)


def test_wrapper_lr_overrides_preserve_model_defaults():
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    args = make_training_args("feature_run", pass_success_ipw=False)
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        baseline = wrapper.build_training_commands(args)[0]
        args.start_lr, args.accumulation_steps, args.monitoring = .004, 2, "on"
        changed = wrapper.build_training_commands(args)[0]
        for before, after in zip(baseline, changed):
            assert wrapper.get_cli_value(after, "--start_lr") == "0.004"
            assert wrapper.get_cli_value(after, "--min_lr") == wrapper.get_cli_value(before, "--min_lr")
            assert wrapper.get_cli_value(after, "--accumulation-steps") == "2"
            assert wrapper.get_cli_value(after, "--monitoring") == "on"
        args.start_lr, args.min_lr = None, .000001
        changed = wrapper.build_training_commands(args)[0]
        for before, after in zip(baseline, changed):
            assert wrapper.get_cli_value(after, "--start_lr") == wrapper.get_cli_value(before, "--start_lr")
            assert float(wrapper.get_cli_value(after, "--min_lr")) == .000001


def test_resume_resolution_and_source_identity(tmp_path):
    directory = tmp_path / "pass_intent" / "a"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="Legacy"):
        resolve_resume_checkpoint("pass_intent/a", tmp_path)
    for value in ("../a", "bundles/a", "C:/outside"):
        with pytest.raises(ValueError):
            resolve_resume_checkpoint(value, tmp_path)
    (directory / CHECKPOINT_NAME).touch()
    assert resolve_resume_checkpoint("a", tmp_path) == directory / CHECKPOINT_NAME
    args = SimpleNamespace(feature_dir=str(directory), split_manifest={"train": ["a"]})
    signature = dataset_signature(args)
    (directory / "a.pt").write_bytes(b"changed")
    assert signature != dataset_signature(args)


def test_monitoring_is_optional_and_failures_are_nonfatal(tmp_path):
    with patch("training_monitor.subprocess.Popen", side_effect=OSError("unavailable")) as popen:
        off = TrainingMonitor(tmp_path, False)
        off.event("batch")
        popen.assert_not_called()
        assert not list(tmp_path.iterdir())
        on = TrainingMonitor(tmp_path, True)
        on.event("batch", epoch=1)
        assert json.loads((tmp_path / "training_monitor.jsonl").read_text())["epoch"] == 1
        with patch("training_monitor.append_event", side_effect=OSError("disk full")):
            on.event("batch")
        on.close()


def test_monitor_process_records_sample_and_exits(tmp_path):
    monitor = TrainingMonitor(tmp_path, True)
    process = monitor.process
    assert process is not None
    try:
        deadline = time.monotonic() + 10
        path = tmp_path / "gpu_monitor.jsonl"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        event = json.loads(path.read_text().splitlines()[0])
        assert "rows" in event or "error" in event  # unavailable NVIDIA telemetry is supported
    finally:
        monitor.close()
    assert process.poll() is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_train_epoch_checkpoint_resume_equivalence(tmp_path, monkeypatch, device):
    """Exercise the actual entry point, disk loader, optimizer loop and resume path."""
    import project_config as pc
    import models.utils as utils
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    features, labels_dir = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels_dir.mkdir()
    ids = [f"m{i}" for i in range(5)]
    for match in ids:
        labels = torch.zeros(3, len(config.LABEL_COLUMNS))
        labels[:, config.LABEL_INDEX["is_pass"]] = 1
        labels[:, config.LABEL_INDEX["intent_index"]] = 1
        labels[:, 7] = 1
        labels[:, 0] = torch.arange(3)
        torch.save([graph(5, 3) for _ in range(3)], features / f"{match}.pt")
        torch.save(labels, labels_dir / f"{match}.pt")
    manifest = {"manifest_id": "fixture", "train": ids, "test": [], "metadata": {"train_size": 5}}
    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: None)
    monkeypatch.setattr(pc, "resolve_feature_root", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(pc, "resolve_artifact_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(pc, "get_model_run_root", lambda task, run_id: tmp_path / "saved" / task / run_id)
    base = ["train.py", "--task", "pass_intent", "--model", "gat", "--device", device,
            "--feature_dir", str(features), "--label_dir", str(labels_dir), "--n_epochs", "5",
            "--start_lr", "0.002", "--min_lr", "0.00001",
            "--batch_size", "2", "--accumulation-steps", "2", "--dataset-loading", "disk",
            "--dataset-cache-dir", str(tmp_path / "cache"), "--node_emb_dim", "8", "--graph_emb_dim", "8",
            "--gnn_heads", "2", "--mlp_h1_dim", "8", "--mlp_h2_dim", "4", "--no-early-stopping"]
    entry = Path(__file__).resolve().parents[1] / "train.py"
    original_epoch = utils.run_epoch

    def plateau(*a, **kw):
        result = original_epoch(*a, **kw)
        if not kw.get("train", False):
            result["ce_loss"] = 1.0
        return result

    monkeypatch.setattr(utils, "run_epoch", plateau)
    monkeypatch.setattr(sys, "argv", base + ["--run-id", "baseline"])
    runpy.run_path(str(entry), run_name="__main__")
    baseline = load_checkpoint(tmp_path / "saved/pass_intent/baseline" / CHECKPOINT_NAME)
    original = utils.run_epoch

    def interrupt(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("simulated interruption")
        return original(args, *a, **kw)

    monkeypatch.setattr(utils, "run_epoch", interrupt)
    monkeypatch.setattr(sys, "argv", base + ["--run-id", "resumed"])
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runpy.run_path(str(entry), run_name="__main__")
    faulthandler.disable()
    checkpoint_path = tmp_path / "saved/pass_intent/resumed" / CHECKPOINT_NAME
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["metadata"]["last_epoch"] == 1 and not checkpoint["finished"]
    monkeypatch.setattr(utils, "run_epoch", original)
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume-checkpoint", str(checkpoint_path)])
    runpy.run_path(str(entry), run_name="__main__")
    resumed = load_checkpoint(checkpoint_path)
    assert resumed["finished"] and resumed["metadata"]["last_epoch"] == 5
    assert resumed["metadata"]["lr"] == .001  # plateau rolled back best model and optimizer
    assert resumed["metadata"]["resume_events"][0]["completed_epoch"] == 1
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], atol=0 if device == "cpu" else 1e-6, rtol=0 if device == "cpu" else 1e-5)
    for key, value in baseline["optimizer"]["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, resumed["optimizer"]["state"][key][field], atol=0 if device == "cpu" else 1e-6, rtol=0 if device == "cpu" else 1e-5)
    # Three updates per epoch, rollback to epoch 1 then one more epoch.
    assert all(int(value["step"]) == 6 for value in resumed["optimizer"]["state"].values())

    if device == "cuda":
        # Repeated real GNN batches check live memory, not the allocator cache.
        from models.gnn import GNN
        args = SimpleNamespace(**resumed["args"])
        model = GNN(vars(args)).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        sample_labels = torch.zeros(len(config.LABEL_COLUMNS))
        sample_labels[5] = 1
        samples = [(graph(22, 11), sample_labels.clone(), torch.tensor(1.)) for _ in range(128)]
        loader = DataLoader(samples, batch_size=32)
        allocated = []
        for _ in range(8):
            original_epoch(args, model, loader, optimizer, device, train=True)
            torch.cuda.synchronize()
            allocated.append(torch.cuda.memory_allocated())
        assert max(allocated[2:]) - min(allocated[2:]) < 1024 * 1024
        print("CUDA live memory after epochs:", allocated)
