"""Exercise real forward/backward, validation, checkpointing and cache cleanup."""
import faulthandler
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from datatools import config
from dataset_loading import PREPARED_TASKS
from training_state import CHECKPOINT_NAME, load_checkpoint
from test_goal_context import source


@pytest.mark.parametrize("task", sorted(PREPARED_TASKS))
@pytest.mark.parametrize("architecture", ["gat", "relationship-attention"])
def test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, task, architecture, ipw_model_id=None, keep_cache=False):
    import project_config as pc
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(exist_ok=True); labels.mkdir(exist_ok=True)
    ids = [f"m{i}" for i in range(5)]
    for match in ids:
        graph, label = source(task=task)
        if task == "success_intent":
            graph.x = graph.x[:, :24]
        label[config.LABEL_INDEX["pass_max_ball_z"]] = 2
        label[config.LABEL_INDEX["pass_high"]] = 1
        torch.save([graph, graph.clone()], features / (match + ".pt"))
        torch.save(torch.stack([label, label.clone()]), labels / (match + ".pt"))
    manifest = dict(manifest_id="fixture", train=ids, test=[], metadata={"train_size": 5})
    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: None)
    monkeypatch.setattr(pc, "resolve_feature_root", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(pc, "resolve_artifact_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(pc, "get_model_run_root", lambda task, run_id: tmp_path / "saved" / task / run_id)
    command = ["train.py", "--task", task, "--model", architecture, "--device", "cpu",
        "--feature_dir", str(features), "--label_dir", str(labels), "--run-id", "verification",
        "--dataset-cache-dir", str(tmp_path / "cache"), "--n_epochs", "1", "--batch_size", "2",
        "--node_emb_dim", "8", "--graph_emb_dim", "8", "--gnn_heads", "2",
        "--mlp_h1_dim", "8", "--mlp_h2_dim", "4", "--no-goal-nodes",
        "--no-position-direction-edge-features", "--no-relative-velocity-edge-features",
        "--no-separation-rate-edge-features", "--scaling"]
    if task == "action_intent":
        command.append("--no-include-goals")
    if task == "pass_height":
        command.extend(["--pass-height-threshold", "1"])
    if keep_cache:
        command.append("--keep-cache")
    if ipw_model_id:
        command.extend(["--ipw_model_id", ipw_model_id, "--ipw-probability-cache-dir", str(tmp_path.parent / (task + "_ipw"))])
    monkeypatch.setattr(sys, "argv", command)
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
    finally:
        faulthandler.disable()
    directory = tmp_path / "saved" / task / "verification"
    checkpoint = load_checkpoint(directory / CHECKPOINT_NAME)
    assert checkpoint["finished"] and checkpoint["args"]["resolved_dataset_loading"] == "disk"
    assert checkpoint["metadata"]["last_epoch_train_metrics"]
    assert checkpoint["metadata"]["last_epoch_valid_metrics"]
    report = json.loads((directory / "metadata.json").read_text())["cache_lifetime"]
    assert report["status"] == "completed"
    assert report["roots"][0]["deleted_entries"] == (0 if keep_cache else 5)
    assert bool(list((tmp_path / "cache").rglob("*.pt"))) == keep_cache


@pytest.mark.parametrize("task", ["pass_success", "pass_height"])
def test_real_training_with_streamed_ipw(tmp_path, monkeypatch, task):
    import models.utils as utils
    import ipw_preparation as ipw
    import training_state
    monkeypatch.setattr(utils, "get_model_path", lambda model_id: tmp_path / "saved" / model_id)
    monkeypatch.setattr(ipw, "get_model_path", lambda model_id: tmp_path / "saved" / model_id)
    # The fixture's model directory is isolated from the repository saved/ root.
    monkeypatch.setattr(training_state, "dataset_signature", lambda args: "synthetic_sources")
    test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, "pass_intent", "gat")
    test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, task, "relationship-attention",
                                           ipw_model_id="pass_intent/verification")
    checkpoint = load_checkpoint(tmp_path / "saved" / task / "verification" / CHECKPOINT_NAME)
    assert checkpoint["metadata"]["ipw_preparation"]["training"]["misses"] == 4
    assert checkpoint["metadata"]["ipw_preparation"]["validation"]["misses"] == 1
    assert list((tmp_path.parent / (task + "_ipw")).rglob("*.json"))


def test_real_cli_keep_cache(tmp_path, monkeypatch):
    test_fresh_cli_all_tasks_default_to_disk(tmp_path, monkeypatch, "pass_intent", "gat", keep_cache=True)
