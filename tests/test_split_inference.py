import argparse

import pytest

import project_config
from scripts import generate_physical_xpass as physical
from scripts import train_relevant_models as training


@pytest.fixture
def universe(monkeypatch, tmp_path):
    monkeypatch.setattr(project_config, "MATCH_UNIVERSE_PATH", tmp_path / "universe.json")
    monkeypatch.setattr(project_config, "SPLIT_MANIFESTS_DIR", tmp_path / "manifests")
    monkeypatch.setattr(training, "SPLIT_MANIFESTS_DIR", tmp_path / "manifests")
    project_config.save_match_universe([f"m{i:02}" for i in range(10)])
    return tmp_path


def args(**kwargs):
    return argparse.Namespace(**{"train_split": None, "train_count": None, **kwargs})


@pytest.mark.parametrize("selector", [{"train_count": 7}, {"train_split": 70}])
def test_inference_and_assertion(universe, selector):
    manifest = project_config.resolve_split_manifest(**selector)
    metadata = {**project_config.split_metadata(selector), "split_manifest_id": manifest["manifest_id"], "split_manifest": manifest["metadata"]}
    for requested in (args(), args(**selector)):
        resolved, source = training.resolve_training_split(requested, metadata)
        assert resolved == manifest
        assert project_config.split_metadata(requested) == project_config.split_metadata(selector)
        assert project_config.split_cli_args(requested) == project_config.split_cli_args(selector)
    recovered, source = training.resolve_training_split(args(), {"split_manifest_id": manifest["manifest_id"]})
    assert recovered == manifest
    assert source == "feature-run manifest"
    opposite = args(train_split=70) if "train_count" in selector else args(train_count=7)
    with pytest.raises(ValueError, match="does not match"):
        training.resolve_training_split(opposite, metadata)
    project_config.save_match_universe([f"m{i:02}" for i in range(11)])
    assert training.resolve_training_split(args(), metadata)[0] == manifest


def test_legacy_and_malformed(universe):
    with pytest.raises(ValueError, match="unmapped legacy"):
        training.resolve_training_split(args(), {})
    with pytest.raises(ValueError):
        training.resolve_training_split(args(train_count=5), {})
    for metadata in ({"train_count": 7}, {"split_manifest": {}}, {"split_manifest_id": "missing"}):
        with pytest.raises((ValueError, FileNotFoundError)):
            training.resolve_training_split(args(), metadata)
    manifest = project_config.resolve_split_manifest(train_count=7)
    with pytest.raises(ValueError, match="does not match"):
        training.resolve_training_split(args(), {"split_manifest_id": manifest["manifest_id"], "train_count": 6})
    with pytest.raises(ValueError, match="conflicts"):
        training.resolve_training_split(args(), {"split_manifest_id": manifest["manifest_id"], "split_manifest": {}})


def test_physical_enumeration_without_split_manifest(universe):
    selected = physical.parse_args([])
    assert physical.resolve_match_ids(selected, universe / "missing_graphs") == [f"m{i:02}" for i in range(10)]
    assert not (universe / "manifests").exists()
    selected = physical.parse_args(["--match-id", "m08", "--match-id", "m02"])
    assert physical.resolve_match_ids(selected, universe) == ["m08", "m02"]


@pytest.mark.parametrize("flag,value", [("--split", "all"), ("--train-count", "7"), ("--train-split", "70")])
def test_physical_removed_flags(flag, value):
    with pytest.raises(SystemExit):
        physical.parse_args([flag, value])
