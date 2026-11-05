"""Return additions preserve carry membership and cannot publish incomplete labels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from datatools import config, graph_feature as gf
from datatools.stop_conditions import label_path
from scripts import generate_relevant_features as generator
import test_feature_run_extension as extension_tests
from test_feature_run_extension import make_args, make_metadata


def values(command, flag):
    return [command[i + 1] for i, value in enumerate(command) if value == flag]


@pytest.mark.parametrize("return_type", ["next_5", "next_5+", "next_5+_skip1", "extended_possession"])
@pytest.mark.parametrize("model", [False, True])
def test_return_addition_schedules_every_carry_mode(return_type, model):
    modes = ["original", "angle_only", *(["model"] if model else [])]
    metadata = make_metadata(["next_10"], modes, "success_intent/saved" if model else None)
    metadata.update(carry_variant=dict(available=True), pass_height_threshold_meters=1.7)
    args = make_args([return_type], num_workers="6")
    plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(args, metadata)
    carry_steps = [step for step in plan.command_steps if "--use-carries" in step.command]
    assert len(plan.command_steps) == 5 and len(carry_steps) == 2
    assert [values(step.command, "--split") for step in carry_steps] == [["train"], ["test"]]
    for step in carry_steps:
        assert values(step.command, "--return_type") == [return_type]
        assert values(step.command, "--only-intended-receiver-mode") == modes
        assert values(step.command, "--num-workers") == ["6"]
        assert values(step.command, "--pass-height-threshold") == ["1.7"]
        assert "--labels-only" in step.command and "--post_action" not in step.command
        assert "--receiver-extension-attempt" not in step.command


@pytest.mark.parametrize("replace", [False, True])
def test_combined_receiver_and_return_addition_has_no_duplicate_carry_jobs(replace):
    metadata = make_metadata(["next_10"], ["original", "angle_only", *(["model"] if replace else [])],
                             "success_intent/old" if replace else None)
    metadata["carry_variant"] = dict(available=True)
    plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(
        make_args(["next_5"], model_id="success_intent/new", replace_model=replace), metadata)
    jobs = []
    for step in plan.command_steps:
        if "--use-carries" not in step.command:
            continue
        for split in values(step.command, "--split"):
            for mode in values(step.command, "--only-intended-receiver-mode"):
                for ret in values(step.command, "--return_type"):
                    jobs.append((split, mode, ret))
    expected = {(split, mode, "next_5") for split in ("train", "test")
                for mode in ("original", "angle_only", "model")}
    expected.update((split, "model", "next_10") for split in ("train", "test"))
    assert len(jobs) == len(set(jobs)) and set(jobs) == expected


def test_return_addition_without_carries_keeps_existing_workload():
    plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(make_args(["next_5"]))
    assert len(plan.command_steps) == 3
    assert all("--use-carries" not in step.command for step in plan.command_steps)


@pytest.fixture
def label_run(tmp_path, monkeypatch):
    root = tmp_path / "base"
    root.mkdir()
    metadata = make_metadata(["next_10"], ["original", "angle_only", "model"], "success_intent/saved")
    metadata.update(label_layout_version=1, carry_variant=dict(available=True, definition="fixture"),
                    split_manifest=dict(universe_count=2))
    metadata["stop_conditions"]["available"]["next_5"] = ["base"]
    (root / "metadata.json").write_text(json.dumps(metadata))
    for directory in ("action_graphs", "post_action_graphs", "action_graphs_carries", "post_action_graphs_carries",
                      "action_graphs_intent_train"):
        folder = root / directory
        folder.mkdir()
        for mid in (["train"] if directory == "action_graphs_intent_train" else ["train", "test"]):
            (folder / f"{mid}.pt").write_bytes(b"retained graph payload")
    for mode in metadata["intended_receiver_modes"]:
        for carry in (False, True):
            ids = [1, 3, 8] if carry else [1, 3]
            for mid in ("train", "test"):
                actions = gf.get_resolved_action_path(mid, intended_receiver_mode=mode, root=root, use_carries=carry)
                actions.parent.mkdir(exist_ok=True)
                pd.DataFrame(dict(saved=[True] * len(ids)), index=ids).to_parquet(actions)
                for ret in ("next_10", "next_5"):
                    path = gf.get_action_label_dir(ret, mode, root=root, use_carries=carry) / f"{mid}.pt"
                    write_labels(path, ids)
        for ret in ("next_10", "next_5"):
            path = gf.get_intent_train_label_dir(ret, mode, root=root) / "train.pt"
            write_labels(path, [1, 1, 3])
    plan = SimpleNamespace(target_run_id="base", base_metadata=metadata,
        base_return_types=["next_10"], final_return_types=["next_10", "next_5"],
        added_return_types=["next_5"], final_intended_receiver_modes=metadata["intended_receiver_modes"])
    monkeypatch.setattr(generator, "get_feature_run_root", lambda run: tmp_path / run)
    monkeypatch.setattr(generator, "resolve_split_manifest", lambda **_: dict(train=["train"], test=["test"]))
    return root, metadata, plan


def write_labels(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = torch.zeros((len(ids), len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.tensor(ids)
    torch.save(labels, path)


def test_audit_preserves_graphs_and_validates_all_modes(label_run):
    root, metadata, plan = label_run
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    generator.audit_return_type_extension(plan, metadata)
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("problem", ["missing", "reordered", "wrong_ids", "missing_set_pieces", "wrong_split"])
def test_audit_rejects_incomplete_or_misaligned_carry_labels(label_run, monkeypatch, problem):
    root, metadata, plan = label_run
    path = gf.get_action_label_dir("next_5", "angle_only", root=root, use_carries=True) / "test.pt"
    kwargs = {}
    if problem == "missing":
        path.unlink()
    elif problem == "reordered":
        write_labels(path, [8, 3, 1])
    elif problem == "wrong_ids":
        write_labels(path, [1, 3, 9])
    elif problem == "missing_set_pieces":
        kwargs["set_piece_return_types"] = ["next_5"]
    else:
        monkeypatch.setattr(generator, "resolve_split_manifest", lambda **_: dict(train=["train"], test=["other"]))
    with pytest.raises((ValueError, FileNotFoundError)):
        generator.audit_return_type_extension(plan, metadata, **kwargs)


def test_audit_checks_selected_set_piece_variants(label_run):
    root, metadata, plan = label_run
    for path in list((root / "labels" / "base").rglob("*.pt")):
        if "next_5" in path.parent.name:
            target = label_path(path, root, "set-pieces")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    metadata["stop_conditions"]["available"]["next_5"].append("set-pieces")
    generator.audit_return_type_extension(plan, metadata, set_piece_return_types=["next_5"])
    path = label_path(gf.get_action_label_dir("next_5", "model", root=root, use_carries=True) / "test.pt",
                      root, "set-pieces")
    write_labels(path, [1, 8, 3])
    with pytest.raises(ValueError, match="alignment"):
        generator.audit_return_type_extension(plan, metadata, set_piece_return_types=["next_5"])


def test_labels_only_reuses_saved_carry_actions_for_every_mode(label_run):
    root, metadata, _ = label_run
    called = []
    def construct_labels(**kwargs):
        assert kwargs["relabel_intended_receivers"] is False
        table = kwargs["resolved_actions"]
        assert table.index.tolist() == [1, 3, 8]
        called.append((kwargs["intended_receiver_mode"], kwargs["return_type"]))
        match.actions = table
        labels = torch.zeros((len(table), len(config.LABEL_COLUMNS)))
        labels[:, 0] = torch.tensor(table.index)
        return labels
    match = SimpleNamespace(construct_labels=construct_labels, intended_receiver_stats={})
    gf.build_labels_by_mode_and_return(match, metadata["intended_receiver_modes"], ["next_5", "next_5+"],
        "success_intent/saved", root, "train", prefer_existing_resolved_actions=True, use_carries=True)
    assert len(called) == 6


@pytest.mark.parametrize("omit_carry", [False, True])
def test_extension_publication_requires_complete_carry_outputs(label_run, monkeypatch, omit_carry):
    root, metadata, _ = label_run
    # The source has only the old return; the simulated jobs materialize the new labels.
    for path in list((root / "labels" / "base").rglob("*next_5*")):
        for file in path.iterdir():
            file.unlink()
        path.rmdir()
    metadata["stop_conditions"]["available"].pop("next_5")
    (root / "metadata.json").write_text(json.dumps(metadata))
    source_before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = dict(manifest_id="fixture", train=["train"], test=["test"], metadata=dict(universe_count=2))
    monkeypatch.setattr(generator, "resolve_split_manifest", lambda **_: manifest)
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "base")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: copy.deepcopy(metadata))
    latest = Mock()
    monkeypatch.setattr(generator, "write_latest_run", latest)
    def generate(command):
        variant = values(command, "--feature_variant")
        builder = gf.get_intent_train_label_dir if variant == ["intent_train_augmented"] else gf.get_action_label_dir
        carry = "--use-carries" in command
        kwargs = dict(use_carries=carry) if builder == gf.get_action_label_dir else {}
        for mid in manifest[values(command, "--split")[0]]:
            for mode in values(command, "--only-intended-receiver-mode"):
                if omit_carry and carry and mode == "angle_only" and mid == "test":
                    continue
                reference = builder("next_10", mode, root=root, **kwargs) / f"{mid}.pt"
                target = builder("next_5", mode, root=root.parent / "derived", **kwargs) / f"{mid}.pt"
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(reference.read_bytes())
    monkeypatch.setattr(generator, "run_command", generate)
    def refreshed(root, metadata, requests, **_):
        metadata["stop_conditions"]["available"].update(requests)
    monkeypatch.setattr("datatools.stop_conditions.refresh_labels", refreshed)
    args = make_args(["next_5"])
    if omit_carry:
        with pytest.raises(ValueError, match="inventory mismatch"):
            generator.run_extension_generation(args)
        latest.assert_not_called()
    else:
        generator.run_extension_generation(args)
        latest.assert_called_once_with("feature", "derived")
        assert not (root.parent / "derived" / "labels" / "set-pieces").exists()
    result = json.loads((root.parent / "derived" / "metadata.json").read_text())
    assert result["status"] == ("failed" if omit_carry else "completed")
    assert ("next_5" in result["return_types"]) is not omit_carry
    assert all(p.read_bytes() == data for p, data in source_before.items())
