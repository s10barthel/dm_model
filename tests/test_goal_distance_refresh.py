from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.goal_distance_refresh import (PAIR_COLUMNS, preflight_goal_distance, read_goal_distance_events,
                                              refresh_goal_distance)
from datatools.goal_distance_returns import label_goal_distance_for_return_type
from datatools.match import Match
from datatools.xt_refresh import action_return_pairs, fingerprint, refresh_inventory
from project_config import get_action_label_dir, get_resolved_action_dir
from scripts.feature_extension_diagnostics import ExtensionAttempt
from test_progression import prepare_progression_fixture
from test_xt_refresh import sample, RETURNS
import test_feature_run_extension as extension_tests


def refresh(sample, families=("goal_distance", "progression")):
    root, event_dir, annotations, metadata, *_ = sample
    return refresh_goal_distance(root, metadata, target_families=families,
        event_dir=event_dir, goal_distance_match_dir=annotations,
        attempt=ExtensionAttempt(root, "targets", "in_place", {}))


def tensors(root):
    return {p: torch.load(p, weights_only=True) for p in root.rglob("*.pt") if "labels" in p.parent.name}


def test_combined_repairs_stale_zeros_and_carry_tables_once(sample):
    root, metadata = prepare_progression_fixture(sample)
    for path, labels in tensors(root).items():
        labels[:, PAIR_COLUMNS] = 0
        torch.save(labels, path)
    before = tensors(root)
    real_save = torch.save
    saves = []
    def record(value, stream, *args, **kwargs):
        saves.append(stream.name)
        return real_save(value, stream, *args, **kwargs)
    with patch("torch.save", side_effect=record):
        report = refresh(sample)
    assert len(saves) == len(before) and len(saves) == len(set(saves))
    for path, old in before.items():
        updated = torch.load(path, weights_only=True)
        keep = [i for i in range(old.shape[1]) if i not in PAIR_COLUMNS]
        assert torch.equal(updated[:, keep], old[:, keep])
    actions = pd.read_parquet(root / "resolved_actions_original_carries/m.parquet")
    assert np.isfinite(actions.loc[9, list(("scores_goal_distance", "concedes_goal_distance"))].astype(float)).all()
    assert report["graph_files_loaded"] == 0 and report["tracking_files_loaded"] == 0
    assert report["progression_present"]
    assert refresh(sample)["rewritten_files"] == 0


def test_missing_season_annotation_preflight_and_recovery(sample):
    root, metadata = prepare_progression_fixture(sample)
    annotation_path = sample[2] / "m.csv"
    contents = annotation_path.read_bytes()
    annotation_path.unlink()
    with pytest.raises(FileNotFoundError, match=r"Missing annotation sidecars \(1\): m"):
        preflight_goal_distance(root, metadata, event_dir=sample[1], goal_distance_match_dir=sample[2])
    annotation_path.write_bytes(contents)
    assert refresh(sample)["matches"] == 1


def test_standalone_progression_rejects_stale_labels_without_writes(sample):
    root, _ = prepare_progression_fixture(sample)
    path = root / "success_intent_labels/m.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, PAIR_COLUMNS[0]] = 0.123
    torch.save(labels, path)
    before = {p: fingerprint(p) for p in root.rglob("*") if p.suffix in {".pt", ".parquet"}}
    with pytest.raises(ValueError, match="goal_distance --refresh-target-family progression") as error:
        refresh(sample, ("progression",))
    assert any("match_id" in note and "m.pt" in note for note in error.value.__notes__)
    assert all(fingerprint(p) == digest for p, digest in before.items())
    assert not list(root.rglob("*.tmp"))


def test_goal_distance_alone_does_not_add_progression_but_keeps_existing_it_fresh(sample):
    root, _ = prepare_progression_fixture(sample)
    refresh(sample, ("goal_distance",))
    assert all(labels.shape[1] == config.LABEL_INDEX["scores_progression"] for labels in tensors(root).values())
    refresh(sample)
    sidecar = pd.read_csv(sample[2] / "m.csv")
    sidecar["goal_distance"] = 0.95
    sidecar.to_csv(sample[2] / "m.csv", index=False)
    report = refresh(sample, ("goal_distance",))
    assert report["progression_present"]
    # Goal-only refresh has already recomputed progression; standalone progression is now a no-op.
    assert refresh(sample, ("progression",))["rewritten_files"] == 0


@pytest.mark.parametrize("ret", RETURNS)
def test_refresh_equals_normal_generation_for_every_return_and_team_swapped_carry(sample, ret):
    root, _ = prepare_progression_fixture(sample)
    refresh(sample)
    events = sample[4].copy()
    events["goal_distance"] = events.xT
    actions = pd.read_parquet(root / "resolved_actions_original_carries/m.parquet")
    match = Match.__new__(Match)
    match.events = events
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.tracking = pd.DataFrame({"home_1_x": [1.0], "home_1_y": [2.0], "away_1_x": [3.0], "away_1_y": [4.0]})
    normal = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=ret)
    saved = torch.load(get_action_label_dir(ret, "original", root=root, use_carries=True) / "m.pt", weights_only=True)
    columns = [*PAIR_COLUMNS, config.LABEL_INDEX["scores_progression"]]
    torch.testing.assert_close(saved[:, columns], normal[:, columns], rtol=1e-6, atol=1e-7, equal_nan=True)


def test_unlabeled_action_is_unavailable_not_zero_and_default_sidecar_returns_ignored(sample):
    root, _ = prepare_progression_fixture(sample)
    path = root / "resolved_actions_original/m.parquet"
    # Canonical event exists but no corresponding primary training row.
    primary_path = get_action_label_dir("next_3", "original", root=root) / "m.pt"
    # The fixture primary return is next_3, not the sidecar's default next_5.
    labels = torch.load(primary_path, weights_only=True)
    torch.save(labels[1:], primary_path)
    sidecar = pd.read_csv(sample[2] / "m.csv")
    sidecar["scores_goal_distance"] = 0.999
    sidecar["concedes_goal_distance"] = 0.999
    sidecar.to_csv(sample[2] / "m.csv", index=False)
    refresh(sample)
    updated = pd.read_parquet(path)
    assert np.isnan(updated.loc[0, "scores_progression"])
    assert not np.isclose(updated.scores_goal_distance, 0.999).any()


@pytest.mark.parametrize("problem", ["duplicate", "mismatch", "nonfinite", "bad_anchor", "bad_coordinate"])
def test_invalid_input_rejected_before_replacement(sample, problem):
    root, _ = prepare_progression_fixture(sample)
    if problem in {"duplicate", "mismatch", "nonfinite"}:
        path = sample[2] / "m.csv"
        frame = pd.read_csv(path)
        if problem == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]])
        elif problem == "mismatch":
            frame.loc[0, "action_id"] = 999
        else:
            frame.loc[0, "goal_distance"] = np.nan
        frame.to_csv(path, index=False)
    else:
        path = root / "resolved_actions_original_carries/m.parquet"
        frame = pd.read_parquet(path)
        frame.loc[9, "return_event_index" if problem == "bad_anchor" else "start_x"] = 999 if problem == "bad_anchor" else np.nan
        frame.to_parquet(path)
    before = {p: fingerprint(p) for p in root.rglob("*") if p.suffix in {".pt", ".parquet"}}
    with pytest.raises(ValueError):
        refresh(sample)
    assert all(fingerprint(p) == digest for p, digest in before.items())


def test_pinned_chronology_and_input_changes_during_staging(sample):
    root, _ = prepare_progression_fixture(sample)
    pinned = root / "pass_endpoints/m.parquet"
    pinned.parent.mkdir()
    events = pd.read_csv(sample[1] / "m.csv")
    events.to_parquet(pinned)
    # Raw CSV is not the run's event input when pinned chronology exists.
    (sample[1] / "m.csv").unlink()
    original = read_goal_distance_events
    def mutate(*args):
        values = original(*args)
        sidecar = pd.read_csv(args[1])
        sidecar.loc[0, "goal_distance"] = 0.11
        sidecar.to_csv(args[1], index=False)
        return values
    before = {p: fingerprint(p) for p in root.rglob("*") if p.suffix in {".pt", ".parquet"}}
    with patch("datatools.goal_distance_refresh.read_goal_distance_events", side_effect=mutate):
        with pytest.raises(RuntimeError, match="input changed"):
            refresh(sample)
    assert all(fingerprint(p) == digest for p, digest in before.items())


def test_combined_plan_accepts_either_flag_order_and_has_no_generation_steps():
    for families in [["goal_distance", "progression"], ["progression", "goal_distance"], ["goal_distance"]]:
        plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(
            extension_tests.make_args(refresh_target_families=families),
            extension_tests.make_metadata(return_types=["next_3"]))
        assert plan.command_steps == []


@pytest.mark.parametrize("missing", [False, True])
def test_extension_preflight_before_clone_and_completed_metadata(sample, tmp_path, monkeypatch, missing):
    from dataclasses import replace
    from types import SimpleNamespace
    from scripts import generate_relevant_features as generator
    from project_config import atomic_write_json
    root, metadata = prepare_progression_fixture(sample)
    base = extension_tests.make_metadata(return_types=metadata["return_types"],
        modes=metadata["intended_receiver_modes"], model_id="success_intent/01")
    plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(
        extension_tests.make_args(refresh_target_families=["progression", "goal_distance"]), base)
    plan = replace(plan, base_metadata={**metadata, **base})
    output = tmp_path / "extension"
    monkeypatch.setattr(generator, "build_extension_plan", lambda args: plan)
    monkeypatch.setattr(generator, "get_feature_run_root", lambda name: root if name == plan.base_run_id else output)
    actual_preflight = preflight_goal_distance
    monkeypatch.setattr(generator, "preflight_goal_distance", lambda *a, **kw:
        actual_preflight(*a, **kw, event_dir=sample[1], goal_distance_match_dir=sample[2]))
    monkeypatch.setattr(generator, "derived_metadata", lambda args, plan, status, error=None:
        dict(metadata, run_id=plan.target_run_id, status=status))
    monkeypatch.setattr(generator, "write_run_metadata", lambda path, values:
        atomic_write_json(Path(path) / "metadata.json", values))
    published = []
    monkeypatch.setattr(generator, "write_latest_run", lambda *args: published.append(args))
    if missing:
        (sample[2] / "m.csv").unlink()
        with pytest.raises(FileNotFoundError):
            generator.run_extension_generation(SimpleNamespace())
        assert not output.exists() and not published
    else:
        generator.run_extension_generation(SimpleNamespace())
        import json
        saved = json.loads((output / "metadata.json").read_text())
        assert saved["status"] == "completed"
        assert saved["progression_targets"]["return_types"]
        assert saved["goal_distance_targets"]["return_types"] == metadata["return_types"]
        assert saved["extension_attempt"]["clone_complete"]
        assert saved["extension_attempt"]["target_replacements"] > 0
        assert len(published) == 1


def test_valid_zero_annotations_are_not_treated_as_missing(sample):
    root, _ = prepare_progression_fixture(sample)
    path = sample[2] / "m.csv"
    values = pd.read_csv(path)
    values["goal_distance"] = 0
    values.to_csv(path, index=False)
    refresh(sample)
    assert all(torch.count_nonzero(labels[:, PAIR_COLUMNS]) == 0 for labels in tensors(root).values())
    assert refresh(sample, ("progression",))["rewritten_files"] == 0
