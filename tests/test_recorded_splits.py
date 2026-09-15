"""Regression coverage for historical membership after universe expansion."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import project_config as config
from models.utils import load_splits
from scripts.train_relevant_models import resolve_training_split


@pytest.fixture
def universe(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MATCH_UNIVERSE_PATH", tmp_path / "universe.json")
    monkeypatch.setattr(config, "SPLIT_MANIFESTS_DIR", tmp_path / "manifests")
    return tmp_path


def metadata(manifest):
    return {"split_manifest_id": manifest["manifest_id"], **config.split_metadata(manifest),
            "split_manifest": manifest["metadata"]}


def create_manifest(n, **selector):
    config.save_match_universe([f"m{i:04d}" for i in range(n)])
    return config.resolve_split_manifest(**selector)


@pytest.mark.parametrize("run_id", list(config.LEGACY_FEATURE_SPLIT_MANIFESTS))
def test_legacy_membership_survives_added_earlier_season(universe, monkeypatch, run_id):
    old = create_manifest(612, train_split=50)
    monkeypatch.setitem(config.LEGACY_FEATURE_SPLIT_MANIFESTS, run_id, old["manifest_id"])
    config.save_match_universe([f"a{i:04d}" for i in range(306)] + old["train"] + old["test"])
    for checkpoint in ({}, {"feature_run_id": run_id}):
        resolved, source = config.resolve_artifact_split({}, {}, feature_run_id=run_id, checkpoint=checkpoint)
        train, valid, test = load_splits(feature_dir=None, manifest=resolved)
        assert (len(train), len(valid), len(test)) == (244, 62, 306)
        assert list(test) == old["test"]
        assert set(test).isdisjoint(old["train"])
        assert source == "explicit legacy mapping"
    args = SimpleNamespace(feature_run_id=run_id, train_split=None, train_count=None)
    assert resolve_training_split(args, {})[0] == old


def test_modern_feature_and_checkpoint_survive_expansion_read_only(universe):
    old = create_manifest(918, train_count=765)
    feature = metadata(old)
    # Checkpoints record the selector and complete manifest through args.json.
    checkpoint = {**feature, "split_manifest": old, "feature_run_id": "modern"}
    create_manifest(1224, train_count=765)
    config.MATCH_UNIVERSE_PATH.unlink()
    before = {p.name: p.read_bytes() for p in config.SPLIT_MANIFESTS_DIR.iterdir()}
    with patch.object(config, "resolve_split_manifest", side_effect=AssertionError("must not regenerate")):
        for provenance in (None, checkpoint):
            resolved, _ = config.resolve_artifact_split({}, feature, feature_run_id="modern", checkpoint=provenance)
            train, valid, test = load_splits(feature_dir=None, manifest=resolved)
            assert (len(train), len(valid), len(test)) == (612, 153, 153)
            assert list(test) == old["test"]
        args = SimpleNamespace(feature_run_id="modern", train_split=None, train_count=None)
        assert resolve_training_split(args, feature)[0] == old
    assert before == {p.name: p.read_bytes() for p in config.SPLIT_MANIFESTS_DIR.iterdir()}


def test_filtered_files_do_not_move_boundary_or_validation(universe):
    manifest = create_manifest(918, train_count=765)
    features = universe / "features"
    features.mkdir()
    for match_id in (manifest["train"][764], manifest["test"][0]):
        (features / f"{match_id}.pt").touch()
    train, test = config.load_base_splits(features, manifest=manifest)
    assert list(train) == [manifest["train"][764]]
    assert list(test) == [manifest["test"][0]]
    train, valid, test = load_splits(feature_dir=features, manifest=manifest)
    assert len(train) == 0
    assert list(valid) == [manifest["train"][764]]
    assert list(test) == [manifest["test"][0]]
    for fold in (1, 2, 3):
        train, valid, test = load_splits(feature_dir=None, manifest=manifest, validation_mode="expanding", validation_fold=fold)
        assert set(train).isdisjoint(valid) and set(valid).issubset(manifest["train"])
        assert list(test) == manifest["test"]
    train, valid, test = load_splits(feature_dir=None, manifest=manifest, final_refit=True)
    assert list(train) == manifest["train"] and len(valid) == 0


@pytest.mark.parametrize("damage", ["identity", "duplicate", "overlap", "count", "fingerprint", "selector", "order", "empty"])
def test_corrupt_manifest_is_rejected(universe, damage):
    manifest = create_manifest(10, train_split=50)
    if damage == "identity": manifest["manifest_id"] = "other"
    if damage == "duplicate": manifest["train"][1] = manifest["train"][0]
    if damage == "overlap": manifest["test"][0] = manifest["train"][0]
    if damage == "count": manifest["metadata"]["test_size"] = 99
    if damage == "fingerprint": manifest["metadata"]["universe_fingerprint"] = "bad"
    if damage == "selector": manifest["train_split_percent"] = 70
    if damage == "order": manifest["train"].reverse()
    if damage == "empty": manifest["test"] = []
    path = Path(manifest["path"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        config.load_recorded_split_manifest(path.stem)


def test_missing_unknown_and_conflicting_provenance(universe):
    first = create_manifest(10, train_split=50)
    second = config.resolve_split_manifest(train_count=7)
    with pytest.raises(ValueError, match="unmapped legacy"):
        config.resolve_artifact_split({}, {}, feature_run_id="unknown")
    with pytest.raises(FileNotFoundError):
        config.resolve_artifact_split({}, {"split_manifest_id": "missing"}, feature_run_id="modern")
    with pytest.raises(ValueError, match="does not match"):
        config.resolve_artifact_split({}, metadata(first), feature_run_id="modern", checkpoint=metadata(second))
    with pytest.raises(ValueError, match="does not match"):
        config.resolve_artifact_split({"train_split": 60}, metadata(first), feature_run_id="modern")
    with pytest.raises(ValueError, match="does not match"):
        config.resolve_artifact_split({}, metadata(first), feature_run_id="modern", checkpoint={"train_count": 7})


def test_real_saved_manifests_and_available_features():
    # Read-only integration check. Skip where the repository's data are unavailable.
    if not (config.SPLIT_MANIFESTS_DIR / "train_50pct_910f288a826f.json").exists():
        pytest.skip("local historical manifests not installed")
    for run_id in config.LEGACY_FEATURE_SPLIT_MANIFESTS:
        resolved, _ = config.resolve_artifact_split({}, {}, feature_run_id=run_id)
        assert len(resolved["train"]) == len(resolved["test"]) == 306
        root = config.resolve_feature_root(run_id) / "action_graphs"
        if root.exists():
            train, test = config.load_base_splits(root, manifest=resolved)
            assert (len(train), len(test)) == (306, 306)
    modern = config.load_recorded_split_manifest("train_count_765_8018228599b2")
    assert (len(modern["train"]), len(modern["test"])) == (765, 153)


def test_evaluation_entrypoint_uses_original_checkpoint_run(universe, monkeypatch):
    import test as evaluation
    old = create_manifest(612, train_split=50)
    modern = create_manifest(918, train_count=765)
    run_id = next(iter(config.LEGACY_FEATURE_SPLIT_MANIFESTS))
    monkeypatch.setitem(config.LEGACY_FEATURE_SPLIT_MANIFESTS, run_id, old["manifest_id"])
    monkeypatch.setattr(evaluation, "load_feature_run_metadata", lambda run, **kwargs: {} if run == run_id else metadata(modern))
    record = {"feature_run_id": run_id, "args": {"feature_run_id": run_id}, "metadata": {}}
    args = SimpleNamespace(train_split=None, train_count=None, diagnostic_feature_run_id="modern")
    resolved, _ = evaluation.resolve_evaluation_split(args, record, run_id)
    assert resolved["test"] == old["test"]
    with pytest.raises(ValueError, match="does not match"):
        evaluation.resolve_evaluation_split(args, record, "modern")


def test_requested_loaded_and_contributing_metadata():
    dataset = SimpleNamespace(loaded_match_ids=["a", "b"],
                              features=[SimpleNamespace(evaluation_match_id="a")],
                              skipped_matches={"c": "missing_feature"}, skipped_rows={"graph_none": 2})
    provenance = config.split_dataset_provenance(["a", "b", "c"], dataset)
    assert provenance["requested_match_count"] == 3
    assert provenance["loaded_match_count"] == 2
    assert provenance["contributing_match_count"] == 1
    assert provenance["skipped_matches"] == {"c": "missing_feature"}


def test_direct_training_does_not_default_to_current_fifty_percent(universe, monkeypatch):
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["train.py", "--task", "pass_success", "--model", "gat"])
    module = runpy.run_path(str(config.PROJECT_ROOT / "train.py"), run_name="train_parser_test")
    args = module["args"]
    assert args.train_split is None and args.train_count is None
    manifest = create_manifest(918, train_count=765)
    resolved, _ = module["resolve_artifact_split"](args, metadata(manifest), feature_run_id="modern")
    assert resolved["manifest_id"] == manifest["manifest_id"]


def test_actual_outcome_checkpoint_resolution_is_read_only():
    import test as evaluation
    from models.utils import get_model_record
    runs = {"outcome_scoring": "outcome_scoring_20260708T164345_726883_b85b512c",
            "outcome_conceding": "outcome_conceding_20260708T164345_726883_a0bd5e71"}
    for task, run in runs.items():
        root = config.SAVED_DIR / task / run
        if not (root / "args.json").exists():
            pytest.skip("historical checkpoints not installed")
        record = get_model_record(f"{task}/{run}")
        manifest, source = evaluation.resolve_evaluation_split({}, record, record["feature_run_id"])
        assert manifest["manifest_id"] == "train_50pct_910f288a826f"
        assert (len(manifest["train"]), len(manifest["test"])) == (306, 306)
        assert source == "explicit legacy mapping"
