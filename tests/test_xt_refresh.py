from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.artifact_storage import atomic_parquet_save, atomic_torch_save, clone_feature_artifacts, staged_path
from datatools.match import Match
from datatools.xt_refresh import (TARGET_COLUMNS, action_return_pairs, fingerprint, patch_labels,
                                 read_annotated_events, refresh_inventory, refresh_xt)
from datatools.xt_returns import label_xt_for_return_type
from project_config import get_action_label_dir, get_intent_train_label_dir, get_resolved_action_dir
from scripts.feature_extension_diagnostics import ExtensionAttempt

RETURNS = ["next_3", "next_3_skip1", "in_1", "disc_0.9", "disc_0.9_skip1",
           "disc_max_0.9", "disc_max_0.9_skip1", "disc_poly_max_0.1_1", "disc_poly_max_0.1_1_spstop"]


@pytest.fixture
def sample(tmp_path):
    root, events_dir, sidecars = [tmp_path / name for name in ("source", "events", "sidecars")]
    for directory in (root, events_dir, sidecars):
        directory.mkdir()
    events = pd.DataFrame(dict(action_id=[0, 1, 2], original_event_id=[100, 101, 102],
                               object_id=["home_1", "away_1", "home_1"], period_id=[1, 1, 2],
                               spadl_type=["pass", "pass", "pass"], success=[True, False, True],
                               expected_goal=[np.nan] * 3, xT=[0.1, 0.4, 0.8]))
    events.drop(columns="xT").to_csv(events_dir / "m.csv", index=False)
    annotations = events[["action_id", "xT"]].assign(scores_xT=0.0, concedes_xT=0.0)
    annotations.to_csv(sidecars / "m.csv", index=False)
    actions = events.drop(columns="xT").assign(scores_xt=0.0, concedes_xt=0.0,
                                              frame_id=0, receive_frame_id=0, receiver_id="out",
                                              intent_id="home_1", action_type="pass", blocked=False,
                                              start_x=10.0, start_y=20.0, end_x=30.0, end_y=40.0)
    carry = actions.iloc[[0]].copy()
    carry.index = [9]
    carry["action_id"] = 9
    carry["original_event_id"] = "carry:1:0"
    carry["return_event_index"] = 1
    carry["action_type"] = "dribble"
    carry_actions = pd.concat([actions, carry])
    carry_actions["original_event_id"] = carry_actions.original_event_id.astype(str)
    metadata = dict(run_id="source", status="completed", return_types=RETURNS, intended_receiver_modes=["original", "angle_only", "model"],
                    carry_variant=dict(available=True, definition="fixture"))
    def write_label(directory, indices):
        directory.mkdir(exist_ok=True)
        labels = torch.arange(len(indices) * len(config.LABEL_COLUMNS), dtype=torch.float32).reshape(len(indices), -1)
        labels[:, 0] = torch.tensor(indices)
        labels[:, TARGET_COLUMNS] = 0
        torch.save(labels, directory / "m.pt")
    for name in ("action_graphs", "post_action_graphs", "action_graphs_carries", "post_action_graphs_carries",
                 "action_graphs_intent_train", "action_graphs_success_intent", "augmented_graphs",
                 "augmented_graphs_angle_only", "augmented_graphs_model"):
        (root / name).mkdir()
        (root / name / "m.pt").write_bytes(b"graph payload must never be read by refresh")
    for mode in metadata["intended_receiver_modes"]:
        for use_carries, table in ((False, actions), (True, carry_actions)):
            directory = get_resolved_action_dir(mode, root=root, use_carries=use_carries)
            directory.mkdir()
            table.to_parquet(directory / "m.parquet")
            for ret in RETURNS:
                write_label(get_action_label_dir(ret, mode, root=root, use_carries=use_carries), list(table.index))
        for ret in RETURNS:
            write_label(get_intent_train_label_dir(ret, mode, root=root), [0, 0, 2])
        write_label(root / ("augmented_labels" + ("" if mode == "original" else "_" + mode)), [0, 0, 1, 2])
    write_label(root / "success_intent_labels", [0, 1, 2])
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root, events_dir, sidecars, metadata, events, actions, carry_actions


@pytest.mark.parametrize("return_type", RETURNS)
def test_refresh_matches_normal_label_construction(sample, return_type):
    _, _, _, _, events, _, actions = sample
    match = Match.__new__(Match)
    match.events = events.copy()
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.tracking = pd.DataFrame({"home_1_x": [1.0], "home_1_y": [2.0], "away_1_x": [3.0], "away_1_y": [4.0]})
    normal = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=return_type)
    pairs = action_return_pairs(actions, label_xt_for_return_type(events, return_type))
    patched, _, _ = patch_labels(torch.zeros_like(normal).index_copy(1, torch.tensor([0]), normal[:, [0]]), pairs)
    torch.testing.assert_close(patched[:, TARGET_COLUMNS], normal[:, TARGET_COLUMNS], rtol=0, atol=0)


def test_all_variants_preserved_source_safe_and_noop(sample, tmp_path):
    root, event_dir, sidecars, metadata, _, _, _ = sample
    output = tmp_path / "derived"
    diagnostics = {}
    with patch("datatools.artifact_storage.shutil.copy2", side_effect=AssertionError("payload copy")):
        clone_feature_artifacts(root, output, diagnostics=diagnostics)
    originals = {p.relative_to(root): fingerprint(p) for p in root.rglob("*") if p.is_file()}
    graphs = [p for p in root.rglob("*.pt") if "graphs" in p.parent.name]
    assert diagnostics["artifact_clone"]["copied_bytes"] == 0
    assert all(os.path.samefile(p, output / p.relative_to(root)) for p in graphs)
    report = refresh_xt(output, metadata, xt_match_dir=sidecars, event_dir=event_dir)
    assert report["rewritten_files"] > 0 and report["graph_files_loaded"] == 0
    for relative, before in originals.items():
        assert fingerprint(root / relative) == before
        if relative.suffix == ".pt" and "graphs" not in relative.parent.name:
            original, refreshed = [torch.load(base / relative, weights_only=True) for base in (root, output)]
            keep = [i for i in range(original.shape[1]) if i not in TARGET_COLUMNS]
            assert torch.equal(original[:, keep], refreshed[:, keep])
        elif relative.suffix == ".parquet":
            original, refreshed = [pd.read_parquet(base / relative) for base in (root, output)]
            keep = [c for c in original if c not in {"xT", "scores_xT", "concedes_xT", "scores_xt", "concedes_xt"}]
            pd.testing.assert_frame_equal(original[keep], refreshed[keep])
    assert all(os.path.samefile(p, output / p.relative_to(root)) for p in graphs)
    signatures = {p: p.stat().st_ino for p in output.rglob("*.pt")}
    second = refresh_xt(output, metadata, xt_match_dir=sidecars, event_dir=event_dir)
    assert second["rewritten_files"] == 0
    assert signatures == {p: p.stat().st_ino for p in signatures}


@pytest.mark.parametrize("problem", ["missing", "empty", "duplicate", "incomplete", "invalid", "bad_id", "bad_anchor", "identity", "metadata"])
def test_invalid_inventory_does_not_replace_any_artifact(sample, problem):
    root, events, sidecars, metadata, _, _, _ = sample
    if problem == "missing":
        (sidecars / "m.csv").unlink()
    elif problem in {"empty", "duplicate", "incomplete", "invalid"}:
        annotation = pd.read_csv(sidecars / "m.csv")
        if problem == "empty":
            annotation = annotation.iloc[:0]
        elif problem == "duplicate":
            annotation = pd.concat([annotation, annotation.iloc[:1]])
        elif problem == "incomplete":
            annotation = annotation.iloc[:2]
        else:
            annotation.loc[0, "xT"] = np.nan
        annotation.to_csv(sidecars / "m.csv", index=False)
    elif problem == "metadata":
        metadata = {**metadata, "carry_variant": {"available": False}}
    elif problem == "bad_id":
        path = root / "success_intent_labels/m.pt"
        labels = torch.load(path, weights_only=True)
        labels[0, 0] = 400
        torch.save(labels, path)
    else:
        path = root / "resolved_actions_model_carries/m.parquet"
        actions = pd.read_parquet(path)
        if problem == "bad_anchor":
            actions.loc[9, "return_event_index"] = 300
        else:
            actions.loc[0, "original_event_id"] = "999"
        actions.to_parquet(path)
    originals = {p: fingerprint(p) for p in root.rglob("*") if p.suffix in {".pt", ".parquet"}}
    with pytest.raises((ValueError, FileNotFoundError)):
        refresh_xt(root, metadata, xt_match_dir=sidecars, event_dir=events)
    assert all(fingerprint(p) == value for p, value in originals.items())
    assert not list(root.rglob("*.tmp"))


def test_valid_zero_annotations(sample):
    root, events, sidecars, metadata, _, _, _ = sample
    frame = pd.read_csv(sidecars / "m.csv")
    frame["xT"] = 0.0
    frame.to_csv(sidecars / "m.csv", index=False)
    assert refresh_xt(root, metadata, xt_match_dir=sidecars, event_dir=events)["rewritten_files"] == 0


def test_input_changes_during_staging_are_rejected(sample):
    root, events, sidecars, metadata, _, _, _ = sample
    original = read_annotated_events
    def change_input(*args):
        result = original(*args)
        with (sidecars / "m.csv").open("a") as stream:
            stream.write("\n")
        return result
    before = fingerprint(root / "success_intent_labels/m.pt")
    with patch("datatools.xt_refresh.read_annotated_events", side_effect=change_input):
        with pytest.raises(RuntimeError, match="input changed"):
            refresh_xt(root, metadata, xt_match_dir=sidecars, event_dir=events)
    assert fingerprint(root / "success_intent_labels/m.pt") == before
    assert not list(root.rglob("*.tmp"))


def test_atomic_writes_isolate_both_directions_and_deletion(tmp_path):
    source, derived = tmp_path / "source", tmp_path / "derived"
    source.mkdir()
    atomic_torch_save(torch.tensor([1]), source / "graph.pt")
    atomic_parquet_save(pd.DataFrame({"x": [1]}), source / "actions.parquet")
    clone_feature_artifacts(source, derived)
    atomic_torch_save(torch.tensor([2]), source / "graph.pt")
    atomic_parquet_save(pd.DataFrame({"x": [2]}), derived / "actions.parquet")
    assert torch.load(derived / "graph.pt", weights_only=True).item() == 1
    assert pd.read_parquet(source / "actions.parquet").x.item() == 1
    (source / "graph.pt").unlink()
    assert (derived / "graph.pt").exists()


def test_copy_fallback_is_explicit_and_independent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "graph.pt").write_bytes(b"payload")
    with patch("datatools.artifact_storage.os.link", side_effect=OSError("unsupported")):
        with pytest.raises(OSError, match="--artifact-copy-mode copy"):
            clone_feature_artifacts(source, tmp_path / "failed")
        stats = clone_feature_artifacts(source, tmp_path / "copied", mode="copy")
    assert stats["copied_bytes"] == 7 and stats["linked_files"] == 0
    assert not os.path.samefile(source / "graph.pt", tmp_path / "copied/graph.pt")


def test_atomic_write_interruption_cleans_staging_and_preserves_links(tmp_path):
    path, alias = tmp_path / "artifact.pt", tmp_path / "alias.pt"
    atomic_torch_save(torch.tensor([1]), path)
    os.link(path, alias)
    def fail(value, stream):
        stream.write(b"partial")
        raise KeyboardInterrupt()
    with patch("datatools.artifact_storage.torch.save", side_effect=fail):
        with pytest.raises(KeyboardInterrupt):
            atomic_torch_save(torch.tensor([2]), path)
    assert torch.load(path, weights_only=True).item() == 1 and os.path.samefile(path, alias)
    assert not list(tmp_path.glob("*.tmp"))


def test_cleanup_failure_preserves_primary_error_and_records_leftover(tmp_path):
    attempt = ExtensionAttempt(tmp_path, "xt_refresh", "derived", {})
    with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
        with pytest.raises(RuntimeError, match="primary") as error:
            with staged_path(tmp_path / "labels.pt", attempt=attempt) as temporary:
                raise RuntimeError("primary")
    assert "cleanup failed" in error.value.__notes__[0]
    assert attempt.data["leftover_temporary_paths"] == [str(temporary)]
    temporary.unlink()


def test_prepared_cache_identity_changes_for_shared_paths(sample, tmp_path):
    from prepared_dataset import PreparedActionDataset
    root = sample[0]
    derived = tmp_path / "derived"
    clone_feature_artifacts(root, derived)
    datasets = [PreparedActionDataset([], feature_dir=base / "action_graphs",
                                     label_dir=base / "action_labels_next_3", cache_dir=tmp_path / "cache",
                                     task="pass_intent", progress=False) for base in (root, derived)]
    assert datasets[0].cache_id != datasets[1].cache_id


@pytest.mark.parametrize("failure", [None, "validation", "replacement", "publication"])
@pytest.mark.parametrize("destination_flags", [(), ("--in-place",), ("--overwrite-feature-run",),
                                                ("--in-place", "--overwrite-feature-run")],
                         ids=["derived", "in_place", "legacy_alias", "both_aliases"])
def test_refresh_execution_publication_and_source_integrity(sample, tmp_path, monkeypatch, failure, destination_flags):
    from scripts import generate_relevant_features as generator
    root, events, sidecars, metadata, _, _, _ = sample
    overwrite = bool(destination_flags)
    historical_entry = dict(mode="overwrite_feature_run", status="completed", command="historical command")
    metadata.update(graph_schema=generator.EXPECTED_GRAPH_SCHEMA.copy(),
                    next_action_conditions_enabled=True, intended_receiver_model_id="success_intent/fixture",
                    pass_height_threshold_meters=3.5, extension_history=[historical_entry])
    protected = tmp_path / "older_derived" if overwrite else root
    if overwrite:
        clone_feature_artifacts(root, protected)
    target = root if overwrite else tmp_path / "derived"
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "source")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: metadata.copy())
    monkeypatch.setattr(generator, "get_feature_run_root", lambda run_id: tmp_path / run_id)
    argv = ["generate_relevant_features.py", "--extend-feature-run-id", "source",
            "--refresh-target-family", "xt", "--xt-match-dir", str(sidecars), *destination_flags]
    if not overwrite:
        argv.extend(["--run-id", "derived"])
    monkeypatch.setattr(sys, "argv", argv)
    args = generator.parse_args()
    clone_calls = []
    clone = generator.copy_base_feature_run
    def clone_spy(*a, **kw):
        clone_calls.append(a)
        return clone(*a, **kw)
    monkeypatch.setattr(generator, "copy_base_feature_run", clone_spy)
    latest = []
    monkeypatch.setattr(generator, "write_latest_run", lambda *a: latest.append(a))
    def writer(directory, payload):
        if failure == "publication" and payload["status"] == "completed":
            raise OSError("publication failed")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(json.dumps(payload))
    monkeypatch.setattr(generator, "write_run_metadata", writer)
    refresh = generator.refresh_xt
    monkeypatch.setattr(generator, "refresh_xt", lambda *a, **kw: refresh(*a, event_dir=events, **kw))
    monkeypatch.setattr(generator, "run_generation_steps", lambda *a, **kw: pytest.fail("Full generation invoked"))
    original_replace = Path.replace
    def replace(path, destination):
        if failure == "replacement" and path.name.endswith(".tmp") and Path(destination).suffix == ".pt":
            raise OSError("replacement failed")
        return original_replace(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    before = fingerprint(protected / "success_intent_labels/m.pt")
    if failure == "validation":
        (sidecars / "m.csv").unlink()
    if failure:
        with pytest.raises(OSError, match="missing" if failure == "validation" else failure):
            generator.run_extension_generation(args)
        assert not latest
        output = json.loads((target / "metadata.json").read_text())
        assert output["status"] == ("completed" if overwrite and failure == "validation" else "failed")
        if overwrite and failure == "validation":
            assert output["extension_attempt"]["artifact_state"] == "unchanged"
            assert fingerprint(root / "success_intent_labels/m.pt") == before
    else:
        generator.run_extension_generation(args)
        assert latest == ([] if overwrite else [("feature", "derived")])
        output = json.loads((target / "metadata.json").read_text())
        assert output["status"] == "completed" and output["carry_variant"] == metadata["carry_variant"]
        assert output["return_types"] == RETURNS
        assert output["pass_height_threshold_meters"] == 3.5
        assert output["intended_receiver_model_id"] == metadata["intended_receiver_model_id"]
        if not overwrite:
            assert output["extension_attempt"]["artifact_clone"]["linked_files"] > 0
    assert fingerprint(protected / "success_intent_labels/m.pt") == before
    assert not list(target.rglob("*.tmp"))
    if overwrite:
        assert not clone_calls
        assert output["last_extension_mode"] == "in_place"
        assert output["extension_history"][0] == historical_entry
        assert output["extension_history"][-1]["mode"] == "in_place"
        assert output["extension_history"][-1]["status"] == ("failed" if failure else "completed")
        assert output["command"] == generator.subprocess.list2cmdline(argv)
