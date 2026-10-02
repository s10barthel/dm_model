from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset
from models.pass_height import (definition, evaluation_height, positive_height, relabel_height,
                                resolve_training_height, check_height_probability)
from prepared_dataset import PreparedActionDataset
from test import write_pass_height_diagnostics


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_cutoffs(value):
    with pytest.raises(ArgumentTypeError):
        positive_height(value)


def test_targets_boundaries_and_private_storage():
    labels = torch.zeros((5, len(LABEL_COLUMNS)))
    labels[:, LABEL_INDEX["pass_max_ball_z"]] = torch.tensor([0.9, 1., 1.1, float("nan"), float("inf")])
    result = relabel_height(labels, 1)
    torch.testing.assert_close(result[:, LABEL_INDEX["pass_high"]], torch.tensor([0., 1., 1., float("nan"), float("nan")]), equal_nan=True)
    assert not labels[:, LABEL_INDEX["pass_high"]].any()


def test_resolution_override_legacy_and_resume():
    args = SimpleNamespace(task="pass_height", pass_height_threshold=None)
    resolve_training_height(args, {"pass_height_threshold_meters": 2.})
    assert args.pass_height_definition == definition(2, "feature_run")
    cutoff, context = evaluation_height(args)
    assert cutoff == 2 and not context["pass_height_target_override"]
    cutoff, context = evaluation_height(args, 1.)
    assert cutoff == 1 and context["model_pass_height_definition"]["threshold_meters"] == 2
    args.pass_height_threshold = 1
    with pytest.raises(ValueError, match="Resume"):
        resolve_training_height(args, {}, resume=True)
    with pytest.raises(ValueError, match="requires"):
        resolve_training_height(SimpleNamespace(task="pass_height", pass_height_threshold=None), {})
    with patch("project_config.load_feature_run_metadata", return_value={"pass_height_threshold_meters": 2}):
        cutoff, context = evaluation_height({"feature_run_id": "legacy"})
        assert cutoff is None and context["pass_height_label_mode"] == "legacy_stored"
        assert context["model_pass_height_definition"]["threshold_meters"] == 2
    assert evaluation_height({})[1]["model_pass_height_definition"]["threshold_meters"] is None


@pytest.fixture
def height_artifacts(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM))
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones((2, 2)))
    rows = torch.zeros((5, len(LABEL_COLUMNS)))
    rows[:, LABEL_INDEX["is_pass"]] = 1
    rows[:, LABEL_INDEX["intent_index"]] = 1
    rows[:, 7] = 1
    rows[:, LABEL_INDEX["pass_max_ball_z"]] = torch.tensor([0.5, 1., 1.5, 2., float("nan")])
    rows[:, LABEL_INDEX["pass_high"]] = float("nan")
    rows[:, LABEL_INDEX["action_index"]] = torch.arange(5)
    torch.save([graph.clone() for _ in range(5)], features / "match.pt")
    torch.save(rows, labels / "match.pt")
    return dict(feature_dir=features, label_dir=labels, task="pass_height", min_pass_dur=0,
                extend_features=False, possessor_aware=True, keeper_aware=True,
                ball_z_aware=True, poss_vel_aware=True, poss_rel_vel_aware=True,
                goal_nodes_aware=True, accel_aware=True, train=False)


def test_memory_disk_equivalence_shared_cache_and_missing_binary(height_artifacts, tmp_path):
    source = height_artifacts["label_dir"] / "match.pt"
    before = source.read_bytes()
    eager = ActionDataset(["match"], pass_height_threshold=1., **height_artifacts)
    assert len(eager) == 4
    assert eager.labels[:, LABEL_INDEX["pass_high"]].tolist() == [0, 1, 1, 1]
    disk = PreparedActionDataset(["match"], cache_dir=tmp_path / "cache", progress=False,
                                 pass_height_threshold=1., **height_artifacts)
    disk2 = PreparedActionDataset(["match"], cache_dir=tmp_path / "cache", progress=False,
                                  pass_height_threshold=2., **height_artifacts)
    assert disk.cache_id == disk2.cache_id
    assert disk2.preparation["built"] == 0 and disk2.preparation["reused"] == 1
    assert len(disk) == len(disk2) == 4
    for i, (_, label, _) in enumerate(disk):
        torch.testing.assert_close(label, eager.labels[i])
    assert [float(label[LABEL_INDEX["pass_high"]]) for _, label, _ in disk2] == [0, 0, 0, 1]
    assert [float(label[LABEL_INDEX["pass_high"]]) for _, label, _ in disk] == [0, 1, 1, 1]
    assert source.read_bytes() == before


@pytest.mark.parametrize("cutoff", [0.2, 1., 2.])
def test_diagnostic_bands_and_empty_single_class(tmp_path, cutoff):
    heights = np.array([cutoff - .5, cutoff, cutoff + .5])
    write_pass_height_diagnostics(tmp_path, "height/test", {
        "targets": heights >= cutoff, "predictions": np.array([.1, .6, .9]),
        "observed_pass_max_height": heights}, threshold=.5, height_threshold=cutoff)
    slices = pd.read_csv(tmp_path / "pass_height_observed_height_slices.csv")
    assert slices.sample_count.tolist() == [0, 1, 1, 1]
    assert slices.positive_count.tolist() == [0, 0, 1, 1]
    assert slices.roc_auc.isna().all()


def test_probability_compatibility():
    check_height_probability(None, None)
    check_height_probability(definition(1, "checkpoint"), 1)
    for cached in [None, definition(2, "checkpoint")]:
        with pytest.raises(ValueError, match="incompatible"):
            check_height_probability(cached, 1)


def test_wrapper_forwards_height_cutoff():
    from scripts import evaluate_relevant_models as wrapper
    args = wrapper.parse_args(["--pass-height-model-id", "pass_height/test", "--pass-height-threshold", "1"])
    command = wrapper.add_task_evaluation_options([], args, "pass_height")
    assert command[command.index("--pass-height-threshold") + 1] == "1.0"


def test_train_and_evaluate_two_thresholds_one_feature_source(height_artifacts, tmp_path, monkeypatch):
    """Real CPU training and evaluation entrypoints, with only artifact roots/splits isolated."""
    import faulthandler
    import json
    import runpy
    import sys
    from pathlib import Path
    import project_config as pc
    import models.utils as utils

    features, labels = height_artifacts["feature_dir"], height_artifacts["label_dir"]
    for match in ["m0", "m1", "m2", "m3", "m4", "heldout"]:
        torch.save(torch.load(features / "match.pt", weights_only=False), features / f"{match}.pt")
        torch.save(torch.load(labels / "match.pt", weights_only=False), labels / f"{match}.pt")
    before = {p: p.read_bytes() for root in (features, labels) for p in root.glob("*.pt")}
    manifest = {"manifest_id": "fixture", "train": ["m0", "m1", "m2", "m3", "m4"],
                "test": ["heldout"], "metadata": {"train_size": 5, "universe_fingerprint": "fixture"}}
    monkeypatch.setattr(pc, "resolve_feature_run_id", lambda *a, **kw: None)
    monkeypatch.setattr(pc, "resolve_feature_root", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(pc, "resolve_artifact_split", lambda *a, **kw: (manifest, "fixture"))
    monkeypatch.setattr(pc, "get_model_run_root", lambda task, run_id: tmp_path / "saved" / task / run_id)
    monkeypatch.setattr(utils, "get_model_path", lambda model_id: tmp_path / "saved" / model_id)
    monkeypatch.setattr(pc, "EVALUATION_RUNS_DIR", tmp_path / "evaluations")
    root = Path(__file__).resolve().parents[1]
    base = ["train.py", "--task", "pass_height", "--model", "gat", "--device", "cpu",
            "--feature_dir", str(features), "--label_dir", str(labels), "--n_epochs", "1",
            "--start_lr", "0.002", "--min_lr", "0.00001", "--batch_size", "4",
            "--dataset-loading", "disk", "--dataset-cache-dir", str(tmp_path / "cache"),
            "--node_emb_dim", "8", "--graph_emb_dim", "8", "--gnn_heads", "2",
            "--mlp_h1_dim", "8", "--mlp_h2_dim", "4", "--no-early-stopping"]
    cache_ids = []
    for cutoff in (1, 2):
        run_id = f"height_{cutoff}"
        monkeypatch.setattr(sys, "argv", base + ["--run-id", run_id, "--pass-height-threshold", str(cutoff)])
        try:
            runpy.run_path(str(root / "train.py"), run_name="__main__")
        finally:
            faulthandler.disable()
        metadata = json.loads((tmp_path / "saved/pass_height" / run_id / "metadata.json").read_text())
        assert metadata["pass_height_definition"]["threshold_meters"] == cutoff
        checkpoint = tmp_path / "saved/pass_height" / run_id / "training_checkpoint.pt"
        monkeypatch.setattr(sys, "argv", ["train.py", "--resume-checkpoint", str(checkpoint),
                                         "--pass-height-threshold", "1.5"])
        with pytest.raises(SystemExit) as conflict:
            runpy.run_path(str(root / "train.py"), run_name="__main__")
        assert conflict.value.code == 2
        cache_ids.append(metadata["dataset_loading"]["train"]["cache_id"])
        for override in (None, 1.5):
            output = tmp_path / f"evaluation_{cutoff}_{override}"
            argv = ["test.py", "--model_id", f"pass_height/{run_id}", "--device", "cpu",
                    "--evaluation-output-dir", str(output)]
            if override is not None:
                argv += ["--pass-height-threshold", str(override)]
            monkeypatch.setattr(sys, "argv", argv)
            runpy.run_path(str(root / "test.py"), run_name="__main__")
            evaluation = json.loads((output / "metadata.json").read_text())["evaluation_options"]
            assert evaluation["model_pass_height_threshold_meters"] == cutoff
            assert evaluation["evaluation_pass_height_threshold_meters"] == (override or cutoff)
            assert evaluation["pass_height_target_override"] == (override is not None)
            metrics = json.loads((output / "metrics.json").read_text())["metrics"]
            assert metrics["sample_count"] == 4
            assert metrics["positive_count"] == (2 if override else (3 if cutoff == 1 else 1))
    assert cache_ids[0] == cache_ids[1]
    assert all(path.read_bytes() == data for path, data in before.items())


def test_legacy_labels_preserved_and_sidecar_override(height_artifacts, tmp_path):
    path = height_artifacts["label_dir"] / "match.pt"
    rows = torch.load(path, weights_only=False)
    rows[:, LABEL_INDEX["pass_high"]] = 0
    torch.save(rows, path)
    legacy = ActionDataset(["match"], **height_artifacts)
    assert len(legacy) == 4
    assert legacy.labels[:, LABEL_INDEX["pass_high"]].tolist() == [0, 0, 0, 0]
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    rows[:, LABEL_INDEX["pass_high"]] = float("nan")
    torch.save(rows, diagnostics / "match.pt")
    updated = ActionDataset(["match"], pass_height_diagnostic_label_dir=diagnostics,
                            pass_height_threshold=1, **height_artifacts)
    assert len(updated) == 4
    assert updated.labels[:, LABEL_INDEX["pass_high"]].tolist() == [0, 1, 1, 1]
    rows[-1, LABEL_INDEX["pass_max_ball_z"]] = float("inf")
    torch.save(rows, diagnostics / "match.pt")
    nonfinite = ActionDataset(["match"], pass_height_diagnostic_label_dir=diagnostics,
                              pass_height_threshold=1, **height_artifacts)
    assert len(nonfinite) == 4


def test_training_wrapper_forwards_to_each_stage():
    from scripts import train_relevant_models as wrapper
    from test_success_intent_mode_independent import make_training_args
    args = make_training_args("feature_run", pass_success_ipw=False)
    args.pass_height_threshold = 1.0
    with patch.object(wrapper, "resolve_feature_run_id", return_value="feature_run"), \
            patch.object(wrapper, "resolve_feature_root", return_value=Path("fixture_features")):
        commands = wrapper.build_training_commands(args)[0]
    assert commands
    assert all(command[command.index("--pass-height-threshold") + 1] == "1.0" for command in commands)


def test_inference_probability_cutoff_check(tmp_path):
    import json
    from physical_pass_model import physical_xpass_inference_lookup_config
    (tmp_path / "metadata.json").write_text(json.dumps({"pass_height_definition": definition(2, "checkpoint")}))
    args = {"task": "pass_success", "xpass_weight": "v4", "pass_height_threshold": 1}
    with pytest.raises(ValueError, match="incompatible"):
        physical_xpass_inference_lookup_config(args, cache_dir=tmp_path)
    args["pass_height_threshold"] = 2
    assert physical_xpass_inference_lookup_config(args, cache_dir=tmp_path)["pass_height_definition"]["threshold_meters"] == 2
