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
from training_preflight import TrainingInputError, preflight_training_files, require_complete_dataset


@pytest.mark.parametrize("role", ["graph", "label"])
@pytest.mark.parametrize("failure", ["missing", "denied", "empty", "read"])
def test_training_preflight_reports_input_failures(tmp_path, monkeypatch, role, failure):
    import errno
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for directory in (features, labels):
        (directory / "match.pt").write_bytes(b"payload")
    target = (features if role == "graph" else labels) / "match.pt"
    original_open = Path.open
    cause = None
    if failure == "missing":
        target.unlink()
    elif failure == "empty":
        target.write_bytes(b"")
    else:
        cause = PermissionError(errno.EACCES, "access denied", str(target)) if failure == "denied" else OSError(errno.EIO, "read failed")
        if failure == "denied":
            cause.winerror = 5

        def failing_open(path, *args, **kwargs):
            if path == target:
                if failure == "denied":
                    raise cause
                stream = original_open(path, *args, **kwargs)

                class FailedRead:
                    def __enter__(self):
                        return self

                    def read(self, size):
                        assert size == 1
                        raise cause

                    def __exit__(self, *exc):
                        stream.close()

                return FailedRead()
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(TrainingInputError) as caught:
        preflight_training_files([("validation", ["match"], features, labels)])
    message = str(caught.value)
    for context in ("validation", "match", role, str(target.resolve())):
        assert context in message
    assert caught.value.__cause__ is not None
    if cause is not None:
        assert caught.value.__cause__ is cause
        assert f"errno={cause.errno}" in message
        if failure == "denied":
            assert "winerror=5" in message


def test_training_preflight_deduplicates_and_reads_only_one_byte(tmp_path, monkeypatch):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for directory in (features, labels):
        for mid in ("a", "b"):
            (directory / f"{mid}.pt").write_bytes(b"large payload")
    original_open, original_resolve = Path.open, Path.resolve
    opens, reads, resolutions, active = [], [], [], []

    def tracking_open(path, mode, buffering=-1):
        assert mode == "rb" and buffering == 0
        assert not active
        opens.append(path)
        stream = original_open(path, mode, buffering=buffering)
        active.append(stream)

        class TrackedRead:
            def __enter__(self):
                return self

            def read(self, size):
                reads.append(size)
                return stream.read(size)

            def __exit__(self, *exc):
                stream.close()
                active.remove(stream)

        return TrackedRead()

    def tracking_resolve(path, *args, **kwargs):
        resolutions.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(Path, "resolve", tracking_resolve)
    with patch("torch.load", side_effect=AssertionError("preflight deserialized inputs")), \
         patch("os.access", side_effect=AssertionError("preflight used os.access")), \
         patch.object(Path, "rglob", side_effect=AssertionError("preflight scanned directories")):
        preflight_training_files([
            ("training", ["a", "a", "b"], features, labels),
            ("validation", ["a"], features, labels),
        ])
    assert len(opens) == len(set(opens)) == 4
    assert reads == [1] * 4
    assert resolutions == [features, labels]
    assert not active


def test_training_preflight_stops_on_first_failure(tmp_path, monkeypatch):
    calls = []
    def deny(path, *args, **kwargs):
        calls.append(path)
        raise PermissionError("denied")
    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(TrainingInputError, match="first"):
        preflight_training_files([("training", ["first", "second"], tmp_path, tmp_path)])
    assert calls == [tmp_path / "first.pt"]


def test_complete_training_dataset_preserves_row_filtering():
    dataset = SimpleNamespace(loaded_match_ids=["filtered", "empty"], skipped_matches={},
                              contributing_match_ids=["filtered"], skipped_rows={"graph_none": 5})
    require_complete_dataset(dataset, ["filtered", "empty"], "training")
    with pytest.raises(TrainingInputError, match="validation.*absent"):
        require_complete_dataset(dataset, ["filtered", "absent"], "validation")
    dataset.skipped_matches["empty"] = "label_tensor_has_invalid_shape"
    with pytest.raises(TrainingInputError, match="empty.*label_tensor_has_invalid_shape"):
        require_complete_dataset(dataset, ["filtered", "empty"], "training")


@pytest.fixture
def preflight_inputs(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    ids = ["m0", "m1", "v0"]
    for mid in ids:
        rows = torch.zeros(3, len(config.LABEL_COLUMNS))
        rows[:, config.LABEL_INDEX["is_pass"]] = 1
        rows[:, config.LABEL_INDEX["intent_index"]] = 1
        rows[:, 7] = 1
        rows[:, 0] = torch.arange(3)
        torch.save([graph(5, 3) for _ in range(3)], features / f"{mid}.pt")
        torch.save(rows, labels / f"{mid}.pt")
    return dict(feature_dir=features, label_dir=labels, task="pass_intent", edge_in_dim=2,
                graph_validation=False, goal_nodes_aware=False)


@pytest.mark.parametrize("workers", [None, 1, 2], ids=["memory", "disk-serial", "disk-parallel"])
@pytest.mark.parametrize("failure", ["corrupt", "shape", "alignment"])
def test_training_rejects_file_skips_after_loading(tmp_path, preflight_inputs, workers, failure):
    from dataset import ActionDataset
    from prepared_dataset import PreparedActionDataset
    target = preflight_inputs["label_dir"] / "m0.pt"
    if failure == "corrupt":
        target.write_bytes(b"invalid pickle")
    else:
        torch.save(torch.zeros(3) if failure == "shape" else torch.zeros(1, len(config.LABEL_COLUMNS)), target)
    preflight_training_files([("training", ["m0", "m1"],
                               preflight_inputs["feature_dir"], preflight_inputs["label_dir"])])
    if workers is None:
        dataset = ActionDataset(["m0", "m1"], **preflight_inputs)
    else:
        dataset = PreparedActionDataset(["m0", "m1"], **preflight_inputs,
            cache_dir=tmp_path / "cache", preparation_workers=workers, progress=False)
    assert "m1" in dataset.loaded_match_ids
    with pytest.raises(TrainingInputError, match="training.*m0"):
        require_complete_dataset(dataset, ["m0", "m1"], "training")


def test_training_preflight_checks_warm_cache_without_rebuilding(tmp_path, preflight_inputs, monkeypatch):
    import prepared_dataset as prepared
    ids = ["m0", "m1"]
    split = ("training", ids, preflight_inputs["feature_dir"], preflight_inputs["label_dir"])
    preflight_training_files([split])
    original = prepared.PreparedActionDataset(ids, **preflight_inputs,
        cache_dir=tmp_path / "cache", progress=False)
    monkeypatch.setattr(prepared.PreparedActionDataset, "_build_entry",
                        lambda *a, **kw: pytest.fail("warm cache was rebuilt"))
    with patch("torch.load", side_effect=AssertionError("preflight loaded payloads")):
        preflight_training_files([split])
    reused = prepared.PreparedActionDataset(ids, **preflight_inputs,
        cache_dir=tmp_path / "cache", progress=False)
    require_complete_dataset(reused, ids, "training")
    assert reused.cache_id == original.cache_id
    assert reused.preparation["reused"] == 2
    target = preflight_inputs["feature_dir"] / "m0.pt"
    original_open = Path.open
    def deny(path, *a, **kw):
        if path == target:
            raise PermissionError("source access revoked")
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(TrainingInputError, match="source access revoked"):
        preflight_training_files([split])


@pytest.fixture
def preflight_training_entry(tmp_path, monkeypatch, preflight_inputs):
    import project_config as pc
    import models.utils as utils
    ids = ["m0", "m1", "v0"]
    manifest = {"manifest_id": "fixture", "train": ids, "test": ["reserved"],
                "metadata": {"train_size": 3}}
    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: None)
    monkeypatch.setattr(pc, "resolve_feature_root", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(pc, "resolve_artifact_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(pc, "get_model_run_root", lambda task, run_id: tmp_path / "saved" / task / run_id)
    monkeypatch.setattr(utils, "run_epoch", lambda *a, **kw: pytest.fail("optimization started"))
    argv = ["train.py", "--task", "pass_intent", "--model", "gat", "--device", "cpu",
            "--feature_dir", str(preflight_inputs["feature_dir"]),
            "--label_dir", str(preflight_inputs["label_dir"]), "--n_epochs", "1",
            "--batch_size", "2", "--node_emb_dim", "8", "--graph_emb_dim", "8",
            "--gnn_heads", "2", "--mlp_h1_dim", "8", "--mlp_h2_dim", "4",
            "--no-goal-nodes", "--no-position-direction-edge-features",
            "--no-relative-velocity-edge-features", "--no-separation-rate-edge-features",
            "--graph-validation", "off", "--run-id", "preflight",
            "--dataset-cache-dir", str(tmp_path / "cache")]
    entry = Path(__file__).resolve().parents[1] / "train.py"
    def invoke(*extra):
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        try:
            return runpy.run_path(str(entry), run_name="__main__")
        finally:
            faulthandler.disable()
    return invoke


@pytest.mark.parametrize("mode", ["memory", "disk"])
def test_unreadable_validation_stops_before_dataset_preparation(preflight_training_entry,
        preflight_inputs, monkeypatch, mode):
    import dataset
    import prepared_dataset
    import models.gnn
    import models.utils
    target = preflight_inputs["label_dir"] / "v0.pt"
    original_open = Path.open
    def deny(path, *a, **kw):
        if path == target:
            raise PermissionError("validation access denied")
        return original_open(path, *a, **kw)
    monkeypatch.setattr(Path, "open", deny)
    monkeypatch.setattr(dataset.ActionDataset, "__init__", lambda *a, **kw: pytest.fail("eager dataset started"))
    monkeypatch.setattr(prepared_dataset.PreparedActionDataset, "__init__",
                        lambda *a, **kw: pytest.fail("cache preparation started"))
    monkeypatch.setattr(models.gnn, "GNN", lambda *a, **kw: pytest.fail("model constructed"))
    with patch.object(models.utils, "load_splits", wraps=models.utils.load_splits) as splits:
        with pytest.raises(TrainingInputError, match="validation.*v0.*label.*validation access denied"):
            preflight_training_entry("--dataset-loading", mode, "--dataset-preparation-workers", "2")
    assert splits.call_count == 1


@pytest.mark.parametrize("failed_split,mid", [("training", "m0"), ("validation", "v0")])
def test_real_training_checks_completeness_before_proceeding(preflight_training_entry,
        preflight_inputs, monkeypatch, failed_split, mid):
    import dataset
    original_init = dataset.ActionDataset.__init__
    original_load = torch.load
    constructed = []
    target = preflight_inputs["feature_dir"] / f"{mid}.pt"
    def raced_load(path, *a, **kw):
        if Path(path) == target:
            raise PermissionError("access revoked after preflight")
        return original_load(path, *a, **kw)
    def init_with_race(self, ids, *a, **kw):
        constructed.append(list(ids))
        with patch("torch.load", side_effect=raced_load):
            original_init(self, ids, *a, **kw)
    monkeypatch.setattr(dataset.ActionDataset, "__init__", init_with_race)
    with pytest.raises(TrainingInputError, match=f"{failed_split}.*{mid}.*access revoked"):
        preflight_training_entry("--dataset-loading", "memory")
    assert len(constructed) == (1 if failed_split == "training" else 2)


def test_final_refit_preflights_only_selected_training_inputs(preflight_training_entry,
        preflight_inputs, monkeypatch, tmp_path):
    import training_preflight
    checked = []
    def check_and_stop(splits):
        checked.extend((name, list(ids)) for name, ids, _, _ in splits)
        preflight_training_files(splits)
        raise RuntimeError("preflight completed")
    monkeypatch.setattr(training_preflight, "preflight_training_files", check_and_stop)
    # Neither this unused validation directory nor the reserved test file exists.
    with pytest.raises(RuntimeError, match="preflight completed"):
        preflight_training_entry("--final-refit", "--valid_feature_dir", str(tmp_path / "unused"))
    assert checked == [("training", ["m0", "m1", "v0"])]


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
    parsed = parse_args(["--resume-id", "pass_intent/a", "--dataset-preparation-workers", "2",
                         "--dataset-worker-threads", "1", "--interim-metrics", "full"])
    assert parsed.dataset_preparation_workers == 2 and parsed.dataset_worker_threads == 1
    assert parsed.interim_metrics == "full"
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


def test_spawn_import_does_not_parse_or_load_training_checkpoint():
    import subprocess
    code = "import sys; sys.argv=['train.py', '--resume-checkpoint', 'missing.pt']; import train; assert not hasattr(train, 'args')"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_wrapper_lr_overrides_preserve_model_defaults():
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    args = make_training_args("feature_run", pass_success_ipw=False)
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
         patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        baseline = wrapper.build_training_commands(args)[0]
        args.start_lr, args.accumulation_steps, args.monitoring, args.batch_loader = .004, 2, "light", "pyg"
        changed = wrapper.build_training_commands(args)[0]
        for before, after in zip(baseline, changed):
            assert wrapper.get_cli_value(after, "--start_lr") == "0.004"
            assert wrapper.get_cli_value(after, "--min_lr") == wrapper.get_cli_value(before, "--min_lr")
            assert wrapper.get_cli_value(after, "--accumulation-steps") == "2"
            assert wrapper.get_cli_value(after, "--monitoring") == "light"
            assert wrapper.get_cli_value(after, "--batch-loader") == "pyg"
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
        while (not path.exists() or not path.stat().st_size) and time.monotonic() < deadline:
            time.sleep(.05)
        event = json.loads(path.read_text().splitlines()[0])
        assert "rows" in event or "error" in event  # unavailable NVIDIA telemetry is supported
    finally:
        monitor.close()
    assert process.poll() is not None


@pytest.mark.parametrize("monitoring", ["off", "light", "on"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_train_epoch_checkpoint_resume_equivalence(tmp_path, monkeypatch, device, monitoring):
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
            "--gnn_heads", "2", "--mlp_h1_dim", "8", "--mlp_h2_dim", "4", "--no-early-stopping",
            "--graph-validation", "off"]  # This numerical regression uses legacy sparse raw fixtures.
    entry = Path(__file__).resolve().parents[1] / "train.py"
    base.extend(["--no-goal-nodes", "--monitoring", monitoring,
                 "--no-position-direction-edge-features", "--no-relative-velocity-edge-features",
                 "--no-separation-rate-edge-features"])
    original_epoch = utils.run_epoch

    def plateau(*a, **kw):
        from models.graph_diagnostics import DiagnosticCollater
        assert isinstance(a[2].collate_fn, DiagnosticCollater) == (monitoring == "on")
        result = original_epoch(*a, **kw)
        if not kw.get("train", False):
            result["ce_loss"] = 1.0
        return result

    monkeypatch.setattr(utils, "run_epoch", plateau)
    monkeypatch.setattr(sys, "argv", base + ["--run-id", "baseline", "--batch-loader", "pyg"])
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
    # Exercise a legacy argument record and an explicit runtime override through
    # the real entry point, while preserving the saved optimizer/RNG state.
    if monitoring == "light":
        checkpoint["args"].pop("batch_loader")
        atomic_torch_save(checkpoint, checkpoint_path)
    override = ["--batch-loader", "pyg"] if monitoring == "off" else []
    monkeypatch.setattr(sys, "argv", ["train.py", "--resume-checkpoint", str(checkpoint_path),
                                    "--monitoring", monitoring, "--graph-validation", "off"] + override)
    runpy.run_path(str(entry), run_name="__main__")
    resumed = load_checkpoint(checkpoint_path)
    assert resumed["finished"] and resumed["metadata"]["last_epoch"] == 5
    assert resumed["args"]["batch_loader"] == ("auto" if monitoring == "on" else "pyg")
    assert all(split["effective"] == "pyg" for split in resumed["metadata"]["batch_loaders"].values())
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
