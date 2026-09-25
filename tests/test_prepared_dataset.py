import argparse
import gc
import inspect
import json
import sys
from types import SimpleNamespace
from pathlib import Path
import weakref
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datatools import config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset
from dataset_loading import add_dataset_loading_arguments, dataset_loading_flags, resolve_dataset_loading
import prepared_dataset as prepared
from project_config import split_dataset_provenance


@pytest.fixture
def artifacts(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    for match in range(5):
        graphs = []
        rows = torch.zeros((7, len(LABEL_COLUMNS)))
        for i in range(7):
            x = torch.zeros((4, config.NODE_FEATURE_CORE_DIM))
            x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
            x[0, config.NODE_FEATURE_IS_POSSESSOR] = int(i != 5)
            x[:, config.NODE_FEATURE_X] = torch.arange(4) + i * 5 + match * 50
            x[:, config.NODE_FEATURE_Y] = 34
            x[:, config.NODE_FEATURE_VX] = 3
            graph = Data(x=x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                         edge_attr=torch.ones((4, 5)))
            graphs.append(None if i == 6 else graph)
            rows[i, LABEL_INDEX["action_index"]] = i // 2  # repeated IDs represent augmentation
            rows[i, LABEL_INDEX["is_pass"]] = int(i != 2)
            rows[i, LABEL_INDEX["is_shot"]] = int(i == 2)
            rows[i, LABEL_INDEX["intent_index"]] = -1 if i == 3 else 1
            rows[i, 7] = 0.01 if i == 4 else 1
        torch.save(graphs, features / f"m{match}.pt")
        torch.save(rows, labels / f"m{match}.pt")
    return dict(feature_dir=features, label_dir=labels, task="pass_intent", min_pass_dur=0.1,
                edge_in_dim=5, v_edge_feature_mode="no_poss", relative_speed_edge_feature_mode="no_poss",
                mask_possessor_v_edge_features=True, mask_possessor_relative_speed_edge_features=True,
                vel_node_features_aware=False, extend_features=False)


def make_disk(tmp_path, artifacts, ids=None, **kwargs):
    return prepared.PreparedActionDataset([f"m{i}" for i in range(5)] if ids is None else ids,
                                         cache_dir=tmp_path / "cache", progress=False, **artifacts, **kwargs)


def assert_sample_equal(a, b):
    ga, la, wa = a
    gb, lb, wb = b
    assert set(ga.keys()) == set(gb.keys())
    for k in ga.keys():
        if isinstance(ga[k], torch.Tensor):
            torch.testing.assert_close(ga[k], gb[k], equal_nan=True)
        else:
            assert ga[k] == gb[k]
    torch.testing.assert_close(la, lb, equal_nan=True)
    torch.testing.assert_close(wa, wb)


@pytest.mark.parametrize("task", ["pass_intent", "action_intent"])
def test_equivalence_and_reuse(tmp_path, artifacts, task):
    artifacts["task"] = task
    ids = ["m0", "m1", "missing"]
    eager = ActionDataset(ids, **artifacts)
    disk = make_disk(tmp_path, artifacts, ids)
    assert len(disk) == len(eager)
    for i, sample in enumerate(disk):
        assert_sample_equal(sample, eager[i])
    assert split_dataset_provenance(ids, disk) == split_dataset_provenance(ids, eager)
    with patch.object(prepared.PreparedActionDataset, "_build_entry", side_effect=AssertionError("rebuilt")):
        reused = make_disk(tmp_path, artifacts, ["m0", "m1"])
    assert reused.preparation["reused"] == 2


def identity(sample):
    graph, label, _ = sample
    return graph.evaluation_match_id, float(graph.x[0, config.NODE_FEATURE_X]), tuple(label.tolist())


def test_shuffle_coverage_epoch_and_batch_boundaries(tmp_path, artifacts):
    disk = make_disk(tmp_path, artifacts, shuffle=True, buffer_matches=2, seed=42)
    baseline = sorted(identity(s) for s in disk)
    disk.set_epoch(3)
    order = [identity(s) for s in disk]
    assert order == [identity(s) for s in disk]
    disk.set_epoch(4)
    changed = [identity(s) for s in disk]
    assert order != changed
    assert sorted(changed) == baseline
    loader = DataLoader(disk, batch_size=3, num_workers=0)
    sizes = [batch.num_graphs for batch, _, _ in loader]
    assert sizes == [3, 3, 3, 1]
    assert len(loader) == len(sizes)
    validation = make_disk(tmp_path, artifacts, buffer_matches=9)
    assert validation.buffer_matches == 1
    validation.set_epoch(5)
    assert [identity(s) for s in validation] == baseline


def test_invalidation_and_corrupt_repair(tmp_path, artifacts):
    disk = make_disk(tmp_path, artifacts, ["m0"])
    entry = disk.entries[0]
    Path(entry["path"]).write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="changed"):
        next(iter(disk))
    repaired = make_disk(tmp_path, artifacts, ["m0"])
    assert repaired.preparation["built"] == 1
    assert len(list(repaired)) == 2
    changed_options = {**artifacts, "min_pass_dur": 0.0}
    assert make_disk(tmp_path, changed_options, ["m0"]).cache_id != disk.cache_id
    with patch.object(prepared, "preprocessing_fingerprint", return_value="new-code"), \
            patch.object(prepared, "loaded_preprocessing_fingerprint", return_value="new-code"):
        assert make_disk(tmp_path, artifacts, ["m0"]).cache_id != disk.cache_id
    label_path = artifacts["label_dir"] / "m0.pt"
    x = torch.load(label_path, weights_only=False)
    x[0, LABEL_INDEX["intent_index"]] = -1
    torch.save(x, label_path)
    with pytest.raises(RuntimeError, match="changed"):
        next(iter(repaired))
    refreshed = make_disk(tmp_path, artifacts, ["m0"])
    assert len(refreshed) == 1


def test_space_and_interrupted_write(tmp_path, artifacts):
    with patch.object(prepared.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
        with pytest.raises(OSError, match="10 GiB reserve"):
            make_disk(tmp_path, artifacts, ["m0"])
    with patch.object(prepared.os, "replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            make_disk(tmp_path, artifacts, ["m0"])
    assert not list((tmp_path / "cache").rglob("*.tmp"))
    assert len(make_disk(tmp_path, artifacts, ["m0"])) == 2


def test_buffer_release_and_cancellation(tmp_path, artifacts):
    disk = make_disk(tmp_path, artifacts, shuffle=True, buffer_matches=2)
    refs = []
    original = disk._load_entry
    def load(entry):
        gc.collect()
        # The caller may still own one previous graph, but no previous match list.
        live = sum(ref() is not None for ref in refs)
        assert live <= 4
        payload = original(entry)
        refs.extend(weakref.ref(g) for g in payload["graphs"])
        return payload
    disk._load_entry = load
    for sample in disk:
        pass
    del sample
    gc.collect()
    assert not any(ref() is not None for ref in refs)
    iterator = iter(disk)
    sample = next(iterator)
    iterator.close()
    del sample
    gc.collect()
    assert not any(ref() is not None for ref in refs)


def test_cli_and_unsupported_modes():
    parser = argparse.ArgumentParser()
    add_dataset_loading_arguments(parser)
    args = parser.parse_args([])
    assert resolve_dataset_loading(args.dataset_loading, "pass_intent") == "disk"
    assert resolve_dataset_loading(args.dataset_loading, "outcome_scoring") == "memory"
    assert dataset_loading_flags(args)[3] == "4"
    for task, ipw in [("outcome_scoring", "none"), ("pass_intent", "some/model")]:
        with pytest.raises(ValueError):
            resolve_dataset_loading("disk", task, ipw)
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-buffer-matches", "0"])


def test_empty_skipped_and_worker_guard(tmp_path, artifacts):
    assert len(make_disk(tmp_path, artifacts, [])) == 0
    absent = make_disk(tmp_path, artifacts, ["absent"])
    assert len(absent) == 0 and "absent" in absent.skipped_matches
    empty_options = {**artifacts, "min_pass_dur": 10}
    empty = make_disk(tmp_path, empty_options, ["m0"])
    assert len(empty) == 0 and empty.loaded_match_ids == ["m0"]
    assert empty.contributing_match_ids == []
    assert list(empty) == []
    disk = make_disk(tmp_path, artifacts, ["m0"])
    with patch.object(prepared, "get_worker_info", return_value=object()):
        with pytest.raises(RuntimeError, match="num_workers=0"):
            next(iter(disk))


def test_source_edits_after_import_cannot_relabel_cached_graphs(tmp_path, artifacts):
    with patch.object(prepared, "preprocessing_fingerprint", return_value="edited-on-disk"):
        with pytest.raises(RuntimeError, match="changed since import"):
            make_disk(tmp_path, artifacts, ["m0"])


def test_preparation_fingerprint_has_narrow_code_boundary():
    standard = {path.relative_to(prepared._ROOT).as_posix()
                for path in prepared.preparation_code_paths({"task": "pass_intent"})}
    assert "models/edge_feature_config.py" in standard
    assert "models/dataset_config.py" in standard
    assert "dataset.py" in standard
    assert "datatools/config.py" in standard
    assert "datatools/utils.py" in standard
    for training_only in (
        "models/utils.py", "models/gnn.py", "models/node_selection.py",
        "train.py", "training_state.py", "training_monitor.py",
    ):
        assert training_only not in standard
    assert "physical_pass_model.py" not in standard
    assert "reachability.py" not in standard
    assert prepared.preprocessing_fingerprint({"task": "pass_intent"}) != prepared.preprocessing_fingerprint(
        {"task": "pass_intent", "use_physical_xpass": True}
    )


def test_edge_feature_helpers_remain_compatible_reexports():
    from models import edge_feature_config
    from models import utils

    for name in (
        "normalize_v_edge_feature_mode", "normalize_relative_speed_edge_feature_mode",
        "validate_relative_speed_edge_feature_mode", "normalize_v_edge_feature_args",
        "use_v_edge_features_for_mode", "mask_possessor_v_edge_features_for_mode",
    ):
        assert getattr(utils, name) is getattr(edge_feature_config, name)


@pytest.mark.parametrize("option", [
    "use_physical_xpass", "require_observed_pass_height", "pass_height_cache_dir",
    "evaluation_xpass_cache_dir", "lane_survival", "lane_survival_cache_dir",
])
def test_physical_preparation_code_is_conditional(option):
    paths = {path.name for path in prepared.preparation_code_paths({option: True})}
    assert {"physical_pass_model.py", "reachability.py"}.issubset(paths)


def test_cache_identity_uses_all_action_dataset_options_but_not_runtime_controls(tmp_path, artifacts):
    first = make_disk(tmp_path, artifacts, ["m0"], seed=1, shuffle=True, buffer_matches=1)
    second = make_disk(tmp_path, artifacts, ["m0"], seed=999, shuffle=False, buffer_matches=9)
    assert first.cache_id == second.cache_id
    assert second.preparation["reused"] == 1

    bound = inspect.signature(ActionDataset.__init__).bind(
        None, [], feature_dir=str(Path(artifacts["feature_dir"]).resolve()),
        label_dir=str(Path(artifacts["label_dir"]).resolve()),
        **{key: value for key, value in artifacts.items() if key not in {"feature_dir", "label_dir"}},
    )
    bound.apply_defaults()
    expected = set(bound.arguments) - {"self", "match_ids", "feature_dir", "label_dir"}
    assert set(first.options) == expected
    for training_only in (
        "batch_size", "accumulation_steps", "start_lr", "min_lr", "n_epochs",
        "optimizer", "early_stopping", "monitoring", "device", "pin_memory",
        "seed", "shuffle", "buffer_matches",
    ):
        assert training_only not in first.identity["options"]


def test_interrupted_manifest_publish_and_iterator_error(tmp_path, artifacts):
    original = prepared.os.replace
    calls = []
    def interrupted(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("manifest interrupted")
        original(source, destination)
    with patch.object(prepared.os, "replace", side_effect=interrupted):
        with pytest.raises(OSError, match="manifest interrupted"):
            make_disk(tmp_path, artifacts, ["m0"])
    disk = make_disk(tmp_path, artifacts, ["m0", "m1"], shuffle=True, buffer_matches=2)
    assert disk.preparation["built"] == 2
    refs = []
    load_entry = disk._load_entry
    def load(entry):
        if refs:
            raise OSError("read failure")
        payload = load_entry(entry)
        refs.extend(weakref.ref(g) for g in payload["graphs"])
        return payload
    disk._load_entry = load
    with pytest.raises(OSError, match="read failure"):
        next(iter(disk))
    gc.collect()
    assert not any(ref() is not None for ref in refs)


def test_pipeline_forwards_loading_controls(monkeypatch, tmp_path):
    from scripts import main
    cache = str(tmp_path / "prepared cache")
    monkeypatch.setattr(sys, "argv", ["main.py", "--train-count", "765",
                                     "--target-family", "xt", "--return_type", "disc_0.9",
                                     "--intended-receiver-mode", "original",
                                     "--dataset-loading", "memory", "--dataset-buffer-matches", "7",
                                     "--dataset-cache-dir", cache])
    commands = main.build_commands(main.parse_args())
    command = next(c for c in commands if c[1] == "scripts/train_relevant_models.py")
    for flag, value in [("--dataset-loading", "memory"), ("--dataset-buffer-matches", "7"),
                        ("--dataset-cache-dir", cache)]:
        assert command[command.index(flag) + 1] == value


def test_progress_report_error_does_not_kill_verification(monkeypatch, tmp_path):
    from scripts import verify_intent_loader as verifier
    process = SimpleNamespace(pid=123, wait=lambda: 0)
    polls = iter([None, None, 0, 0])
    process.poll = lambda: next(polls)
    process.terminate = lambda: pytest.fail("Training was terminated by a progress-report error")
    watched = SimpleNamespace(children=lambda **kw: [],
                              memory_info=lambda: SimpleNamespace(rss=100, private=200, vms=300))
    monkeypatch.setattr(verifier.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(verifier.psutil, "Process", lambda pid: watched)
    monkeypatch.setattr(verifier.time, "sleep", lambda _: None)
    monkeypatch.setattr(verifier, "mark_verification_run", lambda *a: None)
    original_write = Path.write_text
    writes = []
    def transient_failure(path, *args, **kwargs):
        writes.append(path)
        if len(writes) == 1:
            raise OSError(22, "Invalid argument")
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", transient_failure)
    result = verifier.verify_epoch({"command": "train.py --task pass_intent", "task": "pass_intent"},
                                   str(tmp_path / "cache"), tmp_path)
    assert result["status"] == "completed"
    assert result["report_write_errors"] == 1
    assert json.loads((tmp_path / "epoch_report.json").read_text())["peak_private"] == 200
