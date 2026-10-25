"""Fault-injection tests for staged extensions and honest artifact publication."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

import project_config as project
from dataset import ActionDataset
from prepared_dataset import PreparedActionDataset
from scripts import generate_relevant_features as generator
from scripts import feature_extension_diagnostics as diagnostics
from test_feature_run_extension import FeatureRunExtensionPlanTests, make_args, make_metadata
from test_goal_poss_dist import graph
from models.edge_schema import BASE_FEATURES, CANONICAL_FEATURES, edge_names, schema_for_names


@pytest.fixture
def environment(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = runs / "base"
    graphs = root / "action_graphs"
    graphs.mkdir(parents=True)
    labels = root / "action_labels_disc_0.9"
    labels.mkdir()
    metadata = make_metadata(graph_schema={"node_in_dim": 25, **schema_for_names(BASE_FEATURES)})
    metadata[generator.PASS_DURATION_KEY] = generator.PASS_DURATION
    (root / "metadata.json").write_text(json.dumps(metadata))
    for name in ("a", "b"):
        torch.save([graph(), None], graphs / f"{name}.pt")
    monkeypatch.setattr(generator, "get_feature_run_root", lambda name: runs / name)
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "base")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: copy.deepcopy(metadata))
    monkeypatch.setattr(generator, "write_latest_run", lambda *a: None)

    def run(*, derived=False, generation=False, relative=False):
        args = make_args(["next_5"] if generation else None,
                         run_id="derived" if derived else None, in_place=not derived, extend_relative_speed_edge_features=relative)
        args.extend_interaction_edge_features = not generation
        plan = FeatureRunExtensionPlanTests().build_plan(args, metadata)
        monkeypatch.setattr(generator, "build_extension_plan", lambda _: plan)
        generator.run_extension_generation(args)

    return root, graphs, labels, metadata, run


def read(root):
    return json.loads((root / "metadata.json").read_text())


def test_save_failure_closes_handle_and_preserves_entire_chain(environment, monkeypatch):
    root, graphs, _, _, run = environment
    before = {p: p.read_bytes() for p in graphs.glob("*.pt")}
    handles = []
    initial = OSError("disk full")
    finalization = RuntimeError("unexpected archive position")

    def fail_save(_, handle):
        handles.append(handle)
        handle.write(b"incomplete")
        try:
            raise initial
        except OSError:
            raise finalization

    monkeypatch.setattr(diagnostics.torch, "save", fail_save)
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is finalization and handles[0].closed
    meta = read(root)
    assert meta["status"] == "completed"
    attempt = meta["extension_attempt"]
    assert attempt["artifact_state"] == "unchanged" and attempt["phase"] == "staging"
    assert [item["message"] for item in attempt["primary_error_chain"]] == ["disk full", "unexpected archive position"]
    assert "disk full" in attempt["primary_traceback"]
    assert meta["graph_schema"]["edge_in_dim"] == 2
    assert not list(graphs.glob("*.tmp"))
    assert all(p.read_bytes() == data for p, data in before.items())


def test_relative_speed_tracks_failing_file_before_save(environment, monkeypatch):
    root, graphs, _, metadata, run = environment
    metadata["graph_schema"] = {"node_in_dim": 25, **schema_for_names(CANONICAL_FEATURES[:4])}
    for path in graphs.glob("*.pt"):
        raw = graph()
        raw.edge_attr = torch.cat((raw.edge_attr, raw.edge_attr.new_ones((raw.num_edges, 2))), dim=-1)
        torch.save([raw], path)
    handles = []
    def fail(_, handle):
        handles.append(handle)
        handle.write(b"partial")
        raise OSError("write failed")
    monkeypatch.setattr(diagnostics.torch, "save", fail)
    with pytest.raises(OSError, match="write failed"):
        run(relative=True)
    assert handles[0].closed and not list(graphs.glob("*.tmp"))
    assert read(root)["extension_attempt"]["artifact_state"] == "unchanged"


def test_cleanup_continues_and_never_hides_primary(environment, monkeypatch):
    root, graphs, _, _, run = environment
    save, unlink = torch.save, Path.unlink
    failure = OSError("primary serialization failure")
    calls = []
    def fail_second(value, handle):
        if handle.name.endswith("b.pt.interactions.tmp"):
            handle.write(b"partial")
            raise failure
        return save(value, handle)
    def lock_first(path, *a, **kw):
        if path.suffix == ".tmp" and path.name.endswith("interactions.tmp"):
            calls.append(path.name)
            if path.name.startswith("a."):
                raise PermissionError("locked temporary file")
        return unlink(path, *a, **kw)
    monkeypatch.setattr(diagnostics.torch, "save", fail_second)
    monkeypatch.setattr(Path, "unlink", lock_first)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is failure
    assert calls == ["a.pt.interactions.tmp", "b.pt.interactions.tmp"]
    attempt = read(root)["extension_attempt"]
    assert attempt["leftover_temporary_paths"] == [str(graphs / "a.pt.interactions.tmp")]
    assert "locked temporary file" in attempt["secondary_errors"][0]
    assert not (graphs / "b.pt.interactions.tmp").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_partial_replacement_is_failed_and_old_schema_retained(environment, monkeypatch, interrupt):
    root, graphs, _, _, run = environment
    replace = Path.replace
    failure = KeyboardInterrupt() if interrupt else PermissionError("replacement locked")
    def fail_second(path, target):
        if path.name == "b.pt.interactions.tmp":
            raise failure
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", fail_second)
    with pytest.raises(type(failure)) as caught:
        run()
    assert caught.value is failure
    meta = read(root)
    assert meta["status"] == "failed" and meta["graph_schema"]["edge_in_dim"] == 2
    attempt = meta["extension_attempt"]
    assert attempt["artifact_state"] == "partial" and attempt["completed_replacements"] == 1
    assert attempt["current_file"] == str(graphs / "b.pt")
    assert attempt["phase"] == "replacing"
    assert torch.load(graphs / "a.pt", weights_only=False)[0].edge_attr.shape[1] == 7
    assert torch.load(graphs / "b.pt", weights_only=False)[0].edge_attr.shape[1] == 2
    with pytest.raises(ValueError, match="replacing"):
        project.assert_feature_run_usable(root)


def test_interruption_after_rename_before_counter_is_unknown(environment, monkeypatch):
    root, graphs, _, _, run = environment
    replace = Path.replace
    def interrupt(path, target):
        result = replace(path, target)
        if path.name.endswith(".interactions.tmp"):
            raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run()
    attempt = read(root)["extension_attempt"]
    assert attempt["artifact_state"] == "unknown" and read(root)["status"] == "failed"
    assert torch.load(graphs / "a.pt", weights_only=False)[0].edge_attr.shape[1] == 7


def test_combined_extensions_keep_evidence_of_first_mutation(environment, monkeypatch):
    root, graphs, _, metadata, run = environment
    metadata["graph_schema"] = {"node_in_dim": 25, **schema_for_names(CANONICAL_FEATURES[:4])}
    for path in graphs.glob("*.pt"):
        raw = graph()
        raw.edge_attr = torch.cat((raw.edge_attr, raw.edge_attr.new_ones((raw.num_edges, 2))), dim=-1)
        torch.save([raw], path)
    def fail(graph):
        raise ValueError("interaction validation failed")
    monkeypatch.setattr(generator, "append_interactions_to_graph", fail)
    with pytest.raises(ValueError, match="validation failed"):
        run(relative=True)
    attempt = read(root)["extension_attempt"]
    assert attempt["phase"] == "staging" and attempt["artifact_state"] == "partial"
    assert attempt["completed_replacements"] == 2
    assert read(root)["status"] == "failed"
    assert torch.load(graphs / "a.pt", weights_only=False)[0].edge_attr.shape[1] == 5


def test_failure_metadata_error_does_not_mask_original(environment, monkeypatch, capsys):
    root, _, _, _, run = environment
    writer = generator.write_run_metadata
    initial = OSError("disk writer failed")
    def fail_save(*a):
        raise initial
    def fail_error_metadata(path, payload):
        if payload["extension_attempt"]["primary_error_chain"]:
            raise PermissionError("metadata locked")
        return writer(path, payload)
    monkeypatch.setattr(diagnostics.torch, "save", fail_save)
    monkeypatch.setattr(generator, "write_run_metadata", fail_error_metadata)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is initial
    assert read(root)["status"] == "in_progress"
    assert "metadata locked" in capsys.readouterr().err
    with pytest.raises(ValueError, match="staging"):
        project.assert_feature_run_usable(root)


def test_metadata_publication_failure_records_complete_artifacts_without_advertising_new_schema(environment, monkeypatch):
    root, graphs, _, _, run = environment
    writer = generator.write_run_metadata
    def fail_completion(path, payload):
        if payload["status"] == "completed" and payload["extension_attempt"]["artifact_state"] == "complete":
            raise OSError("metadata publication failed")
        return writer(path, payload)
    monkeypatch.setattr(generator, "write_run_metadata", fail_completion)
    with pytest.raises(OSError, match="publication failed"):
        run()
    meta = read(root)
    assert meta["status"] == "failed" and meta["graph_schema"]["edge_in_dim"] == 2
    assert meta["extension_attempt"]["phase"] == "publishing_metadata"
    assert meta["extension_attempt"]["artifact_state"] == "complete"
    assert all(torch.load(p, weights_only=False)[0].edge_attr.shape[1] == 7 for p in graphs.glob("*.pt"))


def test_abrupt_stop_leaves_hazardous_phase_and_unknown_count(environment, monkeypatch):
    root, _, _, _, run = environment
    writer = generator.write_run_metadata
    def observe_replace(path, payload):
        if payload["extension_attempt"]["phase"] == "replacing" and payload["status"] == "in_progress":
            assert payload["extension_attempt"]["artifact_state"] == "unknown"
            assert payload["extension_attempt"]["completed_replacements"] is None
        return writer(path, payload)
    monkeypatch.setattr(generator, "write_run_metadata", observe_replace)
    run()
    assert read(root)["status"] == "completed"


@pytest.mark.parametrize("derived", [False, True])
def test_generation_failure_is_unknown_and_does_not_advertise_new_labels(environment, monkeypatch, derived):
    root, _, _, metadata, run = environment
    source_bytes = (root / "metadata.json").read_bytes()
    def fail(_):
        raise RuntimeError("subprocess failed")
    monkeypatch.setattr(generator, "run_command", fail)
    with pytest.raises(RuntimeError, match="subprocess failed"):
        run(generation=True, derived=derived)
    target = root.parent / "derived" if derived else root
    meta = read(target)
    assert meta["status"] == "failed" and meta["extension_attempt"]["artifact_state"] == "unknown"
    assert meta["extension_attempt"]["current_command"]
    assert meta["return_types"] == metadata["return_types"]
    if derived:
        assert (root / "metadata.json").read_bytes() == source_bytes


def test_derived_copy_failure_is_diagnosed_and_source_unchanged(environment, monkeypatch):
    root, _, _, _, run = environment
    before = (root / "metadata.json").read_bytes()
    def fail_copy(*a, **kw):
        raise OSError("copy failed")
    monkeypatch.setattr(generator, "clone_feature_artifacts", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        run(derived=True)
    meta = read(root.parent / "derived")
    assert meta["status"] == "failed" and meta["extension_attempt"]["phase"] == "copying"
    assert meta["extension_attempt"]["artifact_state"] == "unknown"
    assert (root / "metadata.json").read_bytes() == before


def test_refresh_removal_failure_is_unknown(environment, monkeypatch):
    root, _, _, metadata, _ = environment
    args = make_args(model_id="success_intent/new", run_id=None, replace_model=True, in_place=True)
    metadata["intended_receiver_model_id"] = "success_intent/old"
    metadata["intended_receiver_modes"] = ["original", "model"]
    plan = FeatureRunExtensionPlanTests().build_plan(args, metadata)
    monkeypatch.setattr(generator, "build_extension_plan", lambda _: plan)
    def fail(*a):
        raise PermissionError("removal failed")
    monkeypatch.setattr(generator, "remove_model_mode_artifacts", fail)
    with pytest.raises(PermissionError, match="removal failed"):
        generator.run_extension_generation(args)
    attempt = read(root)["extension_attempt"]
    assert attempt["phase"] == "removing_artifacts" and attempt["artifact_state"] == "unknown"


def test_carry_generation_failure_does_not_claim_completion(environment, monkeypatch):
    root, _, _, _, _ = environment
    args = make_args(run_id=None, in_place=True)
    def fail(*a):
        raise RuntimeError("carry generation failed")
    monkeypatch.setattr(generator, "run_command", fail)
    with pytest.raises(RuntimeError, match="carry generation failed"):
        generator.run_carry_extension_generation(args)
    meta = read(root)
    assert meta["status"] == "failed" and not (meta.get("carry_variant") or {}).get("available")
    assert meta["extension_attempt"]["operation"] == "carry_extension"
    assert meta["extension_attempt"]["artifact_state"] == "unknown"


def test_atomic_metadata_failure_leaves_old_json_and_no_temporary(tmp_path, monkeypatch):
    original = {"status": "completed", "version": "old"}
    project.atomic_write_json(tmp_path / "metadata.json", original)
    replace = Path.replace
    def fail(path, target):
        if Path(target).name == "metadata.json":
            raise OSError("metadata rename failed")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="rename failed"):
        project.write_run_metadata(tmp_path, {"status": "failed"})
    assert read(tmp_path) == original and not list(tmp_path.glob("*.tmp"))


def test_summary_failure_is_secondary_and_completion_stays_published(environment, monkeypatch):
    root, _, _, _, run = environment
    from datatools import metadata_summary
    def fail(_):
        raise RuntimeError("summary index unavailable")
    monkeypatch.setattr(metadata_summary, "update_summary_for_run", fail)
    run()
    meta = read(root)
    assert meta["status"] == "completed" and meta["graph_schema"]["edge_in_dim"] == 7
    assert meta["extension_attempt"]["artifact_state"] == "complete"
    assert any("summary index unavailable" in item for item in meta["extension_attempt"]["secondary_errors"])


@pytest.mark.parametrize("status,state", [("failed", "partial"), ("in_progress", "unchanged"),
                                          ("completed", "unknown"), ("failed", "complete")])
def test_consumers_reject_incomplete_runs_but_inspection_remains_possible(environment, monkeypatch, status, state):
    root, graphs, labels, metadata, _ = environment
    metadata.update(status=status, extension_attempt={"artifact_state": state, "phase": "replacing"})
    (root / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(project, "get_feature_run_root", lambda _: root)
    assert project.load_feature_run_metadata("base")["status"] == status
    with pytest.raises(ValueError, match="replacing"):
        project.resolve_feature_run_id("base")
    with pytest.raises(ValueError, match="replacing"):
        ActionDataset([], feature_dir=graphs, label_dir=labels)
    with pytest.raises(ValueError, match="replacing"):
        PreparedActionDataset([], feature_dir=graphs, label_dir=labels, task="pass_intent", progress=False)


def test_legacy_runs_without_status_and_completed_runs_are_usable(environment, monkeypatch):
    root, _, _, metadata, _ = environment
    monkeypatch.setattr(project, "get_feature_run_root", lambda _: root)
    for status in (None, "completed"):
        metadata.pop("status", None)
        if status:
            metadata["status"] = status
        (root / "metadata.json").write_text(json.dumps(metadata))
        assert project.resolve_feature_run_id("base") == "base"
    (root / "metadata.json").unlink()
    assert project.resolve_feature_run_id("base") == "base"


def test_prepared_iteration_rechecks_run_state(environment):
    root, graphs, labels, metadata, _ = environment
    dataset = PreparedActionDataset([], feature_dir=graphs, label_dir=labels,
                                    task="pass_intent", cache_dir=root.parent / "cache", progress=False)
    metadata.update(status="failed", extension_attempt={"artifact_state": "partial", "phase": "replacing"})
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="replacing"):
        next(iter(dataset))


def test_success_preserves_graphs_nulls_and_metadata(environment):
    root, graphs, _, _, run = environment
    original = torch.load(graphs / "a.pt", weights_only=False)
    run()
    upgraded = torch.load(graphs / "a.pt", weights_only=False)
    torch.testing.assert_close(upgraded[0].x, original[0].x)
    torch.testing.assert_close(upgraded[0].edge_index, original[0].edge_index)
    torch.testing.assert_close(upgraded[0].edge_attr[:, :2], original[0].edge_attr)
    assert upgraded[1] is None and edge_names(upgraded[0])[:2] == BASE_FEATURES
    assert read(root)["extension_attempt"]["completed_replacements"] == 2
    assert not list(graphs.glob("*.tmp"))



def test_first_replacement_failure_can_prove_originals_unchanged(environment, monkeypatch):
    root, graphs, _, _, run = environment
    before = {p: p.read_bytes() for p in graphs.glob("*.pt")}
    replace = Path.replace
    def fail(path, target):
        if path.name.endswith(".interactions.tmp"):
            raise PermissionError("first replacement blocked")
        return replace(path, target)
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(PermissionError, match="blocked"):
        run()
    assert read(root)["status"] == "completed"
    assert read(root)["extension_attempt"]["artifact_state"] == "unchanged"
    assert all(path.read_bytes() == data for path, data in before.items())


def test_write_failure_before_metadata_rename_keeps_existing_json(tmp_path, monkeypatch):
    original = {"status": "completed"}
    project.atomic_write_json(tmp_path / "metadata.json", original)
    def fail(_):
        raise OSError("flush failed")
    monkeypatch.setattr(project.os, "fsync", fail)
    with pytest.raises(OSError, match="flush failed"):
        project.write_run_metadata(tmp_path, {"status": "in_progress"})
    assert read(tmp_path) == original and not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("flags", [("--in-place",), ("--overwrite-feature-run",),
                                   ("--in-place", "--overwrite-feature-run")])
def test_in_place_aliases_mark_uncertain_subprocess_outcomes_failed(environment, monkeypatch, flags):
    root, _, _, _, _ = environment
    monkeypatch.setattr(sys, "argv", ["generate_relevant_features.py", "--extend-feature-run-id", "base",
                                     "--refresh-target-family", "epv", *flags])
    monkeypatch.setattr(generator, "copy_base_feature_run", lambda *a, **kw: pytest.fail("In-place clone"))
    monkeypatch.setattr(generator, "write_latest_run", lambda *a: pytest.fail("In-place latest update"))
    def fail(_):
        raise RuntimeError("subprocess failed")
    monkeypatch.setattr(generator, "run_command", fail)
    with pytest.raises(RuntimeError, match="subprocess failed"):
        generator.run_extension_generation(generator.parse_args())
    meta = read(root)
    assert meta["status"] == "failed" and meta["extension_attempt"]["artifact_state"] == "unknown"
    assert meta["last_extension_mode"] == meta["extension_history"][-1]["mode"] == "in_place"
    assert meta["extension_history"][-1]["status"] == "failed"


def test_initial_metadata_failure_prevents_any_artifact_changes(environment, monkeypatch):
    root, graphs, _, _, run = environment
    before = {p: p.read_bytes() for p in graphs.glob("*.pt")}
    def fail(*a):
        raise OSError("cannot record start")
    monkeypatch.setattr(generator, "write_run_metadata", fail)
    with pytest.raises(OSError, match="cannot record start"):
        run()
    assert read(root)["status"] == "completed"
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(graphs.glob("*.tmp"))


@pytest.mark.parametrize("flags", [("--in-place",), ("--overwrite-feature-run",),
                                   ("--in-place", "--overwrite-feature-run")])
def test_successful_carry_extension_records_complete_attempt(environment, monkeypatch, flags):
    root, _, _, metadata, _ = environment
    monkeypatch.setattr(sys, "argv", ["generate_relevant_features.py", "--extend-feature-run-id", "base",
                                     "--use-carries", *flags])
    commands = []
    monkeypatch.setattr(generator, "run_command", commands.append)
    monkeypatch.setattr(generator, "copy_base_feature_run", lambda *a, **kw: pytest.fail("In-place clone"))
    monkeypatch.setattr(generator, "write_latest_run", lambda *a: pytest.fail("In-place latest update"))
    args = generator.parse_args()
    generator.run_carry_extension_generation(args)
    meta = read(root)
    assert meta["status"] == "completed" and meta["carry_variant"]["available"]
    assert meta["extension_attempt"]["artifact_state"] == "complete"
    assert meta["run_id"] == "base" and meta["last_extension_mode"] == "in_place"
    assert meta["extension_history"][-1]["mode"] == "in_place"
    assert meta["command"] == generator.subprocess.list2cmdline(sys.argv)
    assert meta["extension_history"][-1]["command"] == meta["command"]
    assert len(commands) == 2 and all("--use-carries" in command for command in commands)
    metadata["carry_variant"] = meta["carry_variant"]
    before = (root / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="already contains"):
        generator.run_carry_extension_generation(args)
    assert (root / "metadata.json").read_bytes() == before


def test_refresh_failure_retains_source_height_threshold(environment, monkeypatch):
    root, _, _, metadata, _ = environment
    metadata["pass_height_threshold_meters"] = 2.0
    args = make_args(run_id=None, in_place=True, pass_height=True, pass_height_threshold=1.8)
    plan = FeatureRunExtensionPlanTests().build_plan(args, metadata)
    monkeypatch.setattr(generator, "build_extension_plan", lambda _: plan)
    def fail(_):
        raise RuntimeError("height refresh failed")
    monkeypatch.setattr(generator, "run_command", fail)
    with pytest.raises(RuntimeError, match="refresh failed"):
        generator.run_extension_generation(args)
    meta = read(root)
    assert meta["pass_height_threshold_meters"] == 2.0
    assert meta["extension_attempt"]["target_settings"]["pass_height_threshold_meters"] == 1.8



def test_summary_interruption_after_metadata_commit_preserves_completed_schema(environment, monkeypatch):
    root, _, _, _, run = environment
    from datatools import metadata_summary
    def interrupt_after_publication(path):
        if read(path)["status"] == "completed" and read(path)["extension_attempt"]["artifact_state"] == "complete":
            raise KeyboardInterrupt()
    monkeypatch.setattr(metadata_summary, "update_summary_for_run", interrupt_after_publication)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert read(root)["status"] == "completed"
    assert read(root)["graph_schema"]["edge_in_dim"] == 7
    assert read(root)["extension_attempt"]["artifact_state"] == "complete"
    project.assert_feature_run_usable(root)


def test_failure_metadata_and_cleanup_errors_together_keep_initial_error(environment, monkeypatch):
    root, _, _, _, run = environment
    failure = OSError("original write error")
    writer, unlink = generator.write_run_metadata, Path.unlink
    def fail_save(*a):
        raise failure
    def fail_cleanup(path, *a, **kw):
        if path.name.endswith(".interactions.tmp"):
            raise PermissionError("cleanup error")
        return unlink(path, *a, **kw)
    def fail_metadata(path, payload):
        if payload["extension_attempt"]["primary_error_chain"]:
            raise PermissionError("failure metadata error")
        return writer(path, payload)
    monkeypatch.setattr(diagnostics.torch, "save", fail_save)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    monkeypatch.setattr(generator, "write_run_metadata", fail_metadata)
    with pytest.raises(OSError) as caught:
        run()
    assert caught.value is failure
    assert any("cleanup error" in note for note in failure.__notes__)
    assert any("failure metadata error" in note for note in failure.__notes__)
    assert read(root)["status"] == "in_progress"


def test_close_failure_is_secondary_when_serialization_already_failed(tmp_path, monkeypatch):
    class BrokenStream:
        def close(self):
            raise OSError("close failed")
    initial = OSError("serialization failed")
    monkeypatch.setattr(Path, "open", lambda *a, **kw: BrokenStream())
    def fail_save(*a):
        raise initial
    monkeypatch.setattr(diagnostics.torch, "save", fail_save)
    attempt = diagnostics.ExtensionAttempt(tmp_path, "interaction", "in_place", {})
    with pytest.raises(OSError) as caught:
        diagnostics.save_staged_graphs([], tmp_path / "staged.tmp", attempt)
    assert caught.value is initial
    assert "close failed" in attempt.data["secondary_errors"][0]


def test_metadata_cleanup_failure_does_not_hide_metadata_write_error(tmp_path, monkeypatch):
    original = {"status": "completed"}
    project.atomic_write_json(tmp_path / "metadata.json", original)
    unlink = Path.unlink
    initial = OSError("metadata write failed")
    def fail_sync(_):
        raise initial
    def fail_cleanup(path, *a, **kw):
        if path.suffix == ".tmp":
            raise PermissionError("metadata cleanup locked")
        return unlink(path, *a, **kw)
    monkeypatch.setattr(project.os, "fsync", fail_sync)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(OSError) as caught:
        project.write_run_metadata(tmp_path, {"status": "failed"})
    assert caught.value is initial
    assert "metadata cleanup locked" in initial.__notes__[0]
    assert read(tmp_path) == original


def test_validation_failure_in_later_file_leaves_all_originals_untouched(environment):
    root, graphs, _, _, run = environment
    raw = graph()
    raw.x[0, 5] = float("nan")
    torch.save([raw], graphs / "b.pt")
    before = {path: path.read_bytes() for path in graphs.glob("*.pt")}
    with pytest.raises(ValueError, match="finite"):
        run()
    assert read(root)["status"] == "completed"
    assert read(root)["extension_attempt"]["current_file"] == str(graphs / "b.pt")
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(graphs.glob("*.tmp"))



def test_replacement_outcome_probe_failure_preserves_original_error(environment, monkeypatch):
    root, _, _, _, run = environment
    replace, exists = Path.replace, Path.exists
    failure = PermissionError("original replacement failure")
    def fail_replace(path, target):
        if path.name.endswith(".interactions.tmp"):
            raise failure
        return replace(path, target)
    def fail_probe(path):
        if path.name.endswith(".interactions.tmp"):
            raise OSError("cannot inspect replacement outcome")
        return exists(path)
    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "exists", fail_probe)
    with pytest.raises(PermissionError) as caught:
        run()
    assert caught.value is failure
    assert read(root)["status"] == "failed"
    attempt = read(root)["extension_attempt"]
    assert attempt["artifact_state"] == "unknown"
    assert "cannot inspect replacement outcome" in attempt["secondary_errors"][0]
