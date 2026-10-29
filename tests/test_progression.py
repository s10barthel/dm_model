from types import SimpleNamespace
from unittest.mock import patch
import os

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config, utils
from datatools.artifact_storage import clone_feature_artifacts
from datatools.goal_distance import GOAL_DISTANCE_MAX_RAW
from datatools.match import Match
from datatools.progression import action_start_closeness, relative_progression, action_progression
from datatools.progression_refresh import patch_progression, refresh_progression
from datatools.xt_refresh import fingerprint, refresh_inventory
from dataset import _normalize_label_width
from models.utils import get_outcome_targets
from project_config import validate_return_type_for_target_family, infer_target_family
from scripts.feature_extension_diagnostics import ExtensionAttempt
from test_xt_refresh import sample  # Shared inventory includes carries and all receiver/augmentation modes.
import test_feature_run_extension as extension_tests


def closeness(distance):
    return 1 - np.asarray(distance) / GOAL_DISTANCE_MAX_RAW


def test_examples_and_edge_cases():
    np.testing.assert_allclose(relative_progression(closeness([80, 20, 50, 20, 0, 20]),
                                                    closeness([40, 10, 60, 20, 0, 0])),
                               [0.5, 0.5, 0, 0, 0, 1])
    assert relative_progression(closeness(20), 0) == 0
    with pytest.raises(ValueError, match="finite"):
        relative_progression(np.nan, 0.5)


@pytest.mark.parametrize("ret", ["next_3", "next_3_skip1", "in_3"])
def test_existing_selection_and_conceding_preserved(ret):
    events = pd.DataFrame(dict(spadl_type=["pass", "pass", "pass", "shot"],
        object_id=["home_1", "home_1", "away_1", "home_1"],
        goal_distance=closeness([80, 10, 5, 40])))
    if ret == "in_3":
        selected = utils.label_goal_distance_in_state_returns(events, action_offset=3)
    else:
        selected = utils.label_goal_distance_returns(events, lookahead_len=3, skip_first=ret.endswith("skip1"))
    labels = torch.zeros(1, len(config.LABEL_COLUMNS) - 1)
    labels[0, config.LABEL_INDEX["scores_goal_distance"]] = selected.scores_goal_distance.iloc[0]
    labels[0, config.LABEL_INDEX["concedes_goal_distance"]] = selected.concedes_goal_distance.iloc[0]
    updated, changed = patch_progression(labels, pd.Series([closeness(80)]), ret)
    scoring, conceding = get_outcome_targets(updated, SimpleNamespace(use_progression=True))
    assert changed
    assert scoring.item() == pytest.approx(0.875 if ret == "next_3" else 0.5)
    assert torch.equal(conceding, labels[:, config.LABEL_INDEX["concedes_goal_distance"]])
    assert torch.equal(updated[:, :-1], labels)


def test_orientation_and_carry_own_baseline():
    actions = pd.DataFrame(dict(object_id=["home_1", "away_1", "home_1"],
        start_x=[25, 80, 85], start_y=[34, 34, 34],
        return_event_index=[0, 0, 0], scores_goal_distance=closeness([40, 40, 10])))
    np.testing.assert_allclose(action_progression(actions, "next_3"), [0.5, 0.5, 0.5])
    actions.loc[0, "start_x"] = np.nan
    with pytest.raises(ValueError, match="coordinates"):
        action_start_closeness(actions)


def test_old_runs_do_not_acquire_fake_zero_progression():
    old = torch.zeros(2, len(config.LABEL_COLUMNS) - 1)
    normalized = _normalize_label_width(old)
    with pytest.raises(ValueError, match="--refresh-target-family progression"):
        get_outcome_targets(normalized, SimpleNamespace(use_progression=True))
    scoring, _ = get_outcome_targets(normalized, SimpleNamespace(use_goal_distance=True))
    assert torch.equal(scoring, torch.zeros(2))


@pytest.mark.parametrize("return_type", ["next_3", "next_3_skip1", "in_1"])
def test_fresh_generation_matches_extension_including_carry(sample, return_type):
    _, _, _, _, events, _, actions = sample
    events["goal_distance"] = closeness([80, 40, 20])
    match = Match.__new__(Match)
    match.events = events
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.tracking = pd.DataFrame({"home_1_x": [1.0], "home_1_y": [2.0],
                                   "away_1_x": [3.0], "away_1_y": [4.0]})
    normal = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                    return_type=return_type)
    patched, _ = patch_progression(normal[:, :-1], action_start_closeness(match.actions), return_type)
    torch.testing.assert_close(patched, normal, rtol=1e-6, atol=1e-7, equal_nan=True)


def test_wrapper_routes_progression_flag():
    from scripts.train_relevant_models import outcome_command, resolve_wrapper_feature_flags
    command = outcome_command("outcome_scoring", "outcome_scoring/01", "graphs", "labels",
                              "progression", "next_3", "original", 32, "none", "none",
                              resolve_wrapper_feature_flags(SimpleNamespace()))
    assert "--use_progression" in command
    assert "--use_goal_distance" not in command


def test_return_validation_and_extension_plan():
    assert infer_target_family(use_progression=True) == "progression"
    for ret in ("next_3", "next_3_skip1", "in_3"):
        assert validate_return_type_for_target_family(ret, "progression") == ret
    for ret in ("disc_0.9", "disc_max_0.9", "disc_poly_max_0.05_2"):
        with pytest.raises(ValueError, match="progression supports"):
            validate_return_type_for_target_family(ret, "progression")
    plan = extension_tests.FeatureRunExtensionPlanTests().build_plan(
        extension_tests.make_args(refresh_target_families=["progression"]), extension_tests.make_metadata(return_types=["next_3"]))
    assert plan.command_steps == []
    assert plan.final_return_types == ["next_3"]


def prepare_progression_fixture(sample):
    root, _, _, metadata, *_ = sample
    inventory, _, _, _ = refresh_inventory(root, metadata)
    for path in root.glob("resolved_actions*/*.parquet"):
        actions = pd.read_parquet(path)
        actions["scores_goal_distance"] = 0.8
        actions.to_parquet(path)
    for artifact in inventory["m"]:
        labels = torch.load(artifact.path, weights_only=True)[:, :-1]
        labels[:, config.LABEL_INDEX["scores_goal_distance"]] = 0.8
        torch.save(labels, artifact.path)
    return root, metadata


def test_refresh_all_variants_source_preservation_and_noop(sample, tmp_path):
    root, metadata = prepare_progression_fixture(sample)
    before = {p.relative_to(root): fingerprint(p) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "derived"
    clone_feature_artifacts(root, output)
    report = refresh_progression(output, metadata,
        attempt=ExtensionAttempt(output, "progression", "derived", {}))
    assert report["rewritten_files"] > 0
    assert report["graph_files_loaded"] == 0
    for relative, digest in before.items():
        assert fingerprint(root / relative) == digest
        if "graphs" in relative.parent.name:
            assert os.path.samefile(root / relative, output / relative)
        elif relative.suffix == ".pt":
            old = torch.load(root / relative, weights_only=True)
            new = torch.load(output / relative, weights_only=True)
            assert torch.equal(old, new[:, :-1])
    second = refresh_progression(output, metadata,
        attempt=ExtensionAttempt(output, "progression", "derived", {}))
    assert second["rewritten_files"] == 0


def test_invalid_last_variant_leaves_all_artifacts_unchanged(sample):
    root, metadata = prepare_progression_fixture(sample)
    path = root / "success_intent_labels/m.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, 0] = 999
    torch.save(labels, path)
    before = {p: fingerprint(p) for p in root.rglob("*") if p.suffix in {".pt", ".parquet"}}
    with pytest.raises(ValueError, match="action IDs"):
        refresh_progression(root, metadata, attempt=ExtensionAttempt(root, "progression", "in_place", {}))
    assert all(fingerprint(p) == digest for p, digest in before.items())
    assert not list(root.rglob("*.tmp"))
