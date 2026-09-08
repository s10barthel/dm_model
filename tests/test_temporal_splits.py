from __future__ import annotations

import json
import argparse
from types import SimpleNamespace

import numpy as np
import pytest

import project_config
from scripts import train_relevant_models


def configure_split_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    split_dir = tmp_path / "splits"
    monkeypatch.setattr(project_config, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(project_config, "SPLIT_PATH", split_dir / "match_splits.json")
    monkeypatch.setattr(project_config, "MATCH_UNIVERSE_PATH", split_dir / "match_universe.json")
    monkeypatch.setattr(project_config, "SPLIT_MANIFESTS_DIR", split_dir / "manifests")


def test_percentage_splits_and_manifest_identity(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    configure_split_paths(monkeypatch, tmp_path)
    ids = [f"match_{index:04d}" for index in range(612, 0, -1)]
    universe = project_config.save_match_universe(ids)

    split_50 = project_config.resolve_split_manifest(50)
    split_75 = project_config.resolve_split_manifest(75)

    assert len(split_50["train"]) == 306
    assert len(split_50["test"]) == 306
    assert len(split_75["train"]) == 459
    assert len(split_75["test"]) == 153
    assert split_75["train"] == sorted(ids)[:459]
    assert universe["fingerprint"] in json.dumps(split_75)
    assert split_50["manifest_id"] != split_75["manifest_id"]


def test_exact_count_manifest_and_model_splits(monkeypatch, tmp_path):
    configure_split_paths(monkeypatch, tmp_path)
    ids = [f"match_{index:04d}" for index in range(918)]
    universe = project_config.save_match_universe(ids)
    manifest = project_config.resolve_split_manifest(train_count=765)
    assert manifest["train"] == ids[:765]
    assert manifest["test"] == ids[765:]
    assert manifest["train_split_percent"] is None
    assert manifest["train_count"] == 765
    assert manifest["metadata"]["rounding"] == "exact"
    assert manifest["manifest_id"] == f"train_count_765_{universe['fingerprint'][:12]}"
    assert project_config.resolve_split_manifest(train_count=765) == manifest
    train, valid, test = project_config.load_model_splits(train_count=765)
    assert (len(train), len(valid), len(test)) == (612, 153, 153)
    for fold, sizes in enumerate([(382, 128), (510, 127), (637, 128)], 1):
        train, valid, test = project_config.load_model_splits(train_count=765, validation_mode="expanding", validation_fold=fold)
        assert (len(train), len(valid)) == sizes
        assert list(test) == ids[765:]
    train, valid, test = project_config.load_model_splits(train_count=765, final_refit=True)
    assert list(train) == ids[:765] and len(valid) == 0
    # Missing features must not shift the original test boundary.
    features = tmp_path / "features"
    features.mkdir()
    for match_id in (ids[764], ids[765]):
        (features / f"{match_id}.pt").touch()
    train, test = project_config.load_base_splits(features, train_count=765)
    assert list(train) == [ids[764]] and list(test) == [ids[765]]
    project_config.save_match_universe(ids + ["match_0918"])
    assert project_config.resolve_split_manifest(train_count=765)["manifest_id"] != manifest["manifest_id"]


@pytest.mark.parametrize("value", [0, -1, 918, 919, True, 765.0, 765.5, "765"])
def test_invalid_train_count(monkeypatch, tmp_path, value):
    configure_split_paths(monkeypatch, tmp_path)
    project_config.save_match_universe([f"match_{i:04d}" for i in range(918)])
    with pytest.raises(ValueError, match="--train-count must be an integer"):
        project_config.resolve_split_manifest(train_count=value)


def test_count_boundaries_and_percentage_compatibility(monkeypatch, tmp_path):
    configure_split_paths(monkeypatch, tmp_path)
    project_config.save_match_universe([f"match_{i:04d}" for i in range(918)])
    for count in (1, 917):
        assert len(project_config.resolve_split_manifest(train_count=count)["train"]) == count
    percent = project_config.resolve_split_manifest(50)
    assert "train_count" not in percent
    assert percent["metadata"]["rounding"] == "floor"
    assert project_config.resolve_split_manifest() == percent
    count = project_config.resolve_split_manifest(train_count=459)
    assert percent["train"] == count["train"]
    assert percent["manifest_id"] != count["manifest_id"]
    assert len(project_config.resolve_split_manifest(83)["train"]) == 761
    with pytest.raises(ValueError, match="mutually exclusive"):
        project_config.resolve_split_manifest(50, train_count=765)
    # An existing immutable percentage manifest reloads without alteration.
    from pathlib import Path
    path = Path(percent["path"])
    original = path.read_bytes()
    project_config.resolve_split_manifest(50)
    assert path.read_bytes() == original
    payload = json.loads(Path(count["path"]).read_text())
    payload["train_count"] = 458
    Path(count["path"]).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="collision"):
        project_config.resolve_split_manifest(train_count=459)


def test_split_cli_and_provenance():
    parser = argparse.ArgumentParser()
    project_config.add_split_arguments(parser)
    empty = parser.parse_args([])
    assert vars(empty) == {"train_split": None, "train_count": None}
    assert project_config.split_selector(empty) == {"train_split": 50, "train_count": None}
    args = parser.parse_args(["--train-count", "765"])
    metadata = project_config.split_metadata(args)
    assert metadata == {"train_split_percent": None, "train_count": 765}
    assert project_config.split_cli_args(metadata) == ["--train-count", "765"]
    assert project_config.checked_split_selector(empty, metadata) == vars(args)
    assert project_config.checked_split_selector(args, metadata) == vars(args)
    assert project_config.checked_split_selector(empty, {})["train_split"] == 50
    for argv in (["--train-count", "765", "--train-split", "83"], ["--train-count", "765.0"], ["--train-split", "83.33"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    for recorded in ({"train_count": 764}, {"train_split_percent": 83}, {}):
        with pytest.raises(ValueError, match="does not match"):
            project_config.checked_split_selector(args, recorded)


def test_count_forwarded_to_carry_extension():
    from scripts import generate_relevant_features as generator
    metadata = {"train_count": 765, "train_split_percent": None,
                "graph_schema": generator.EXPECTED_GRAPH_SCHEMA.copy(),
                "return_types": ["disc_0.9"], "intended_receiver_modes": ["original"]}
    steps = generator.carry_extension_steps(SimpleNamespace(), metadata, "count-run")
    assert steps
    for step in steps:
        assert "--train-split" not in step.command
        assert step.command[step.command.index("--train-count") + 1] == "765"


def test_count_forwarded_by_pipeline_and_feature_generation(monkeypatch):
    import sys
    from scripts import main, generate_relevant_features as generator
    monkeypatch.setattr(sys, "argv", ["main.py", "--train-count", "765",
                                     "--target-family", "xt", "--return_type", "disc_0.9",
                                     "--intended-receiver-mode", "original"])
    commands = main.build_commands(main.parse_args())
    split_scripts = {"scripts/generate_xt.py", "scripts/generate_relevant_features.py",
                     "scripts/train_relevant_models.py", "scripts/run_relevant_models.py"}
    selected = [command for command in commands if command[1] in split_scripts]
    assert {command[1] for command in selected} == split_scripts
    for command in selected:
        assert "--train-split" not in command
        assert command[command.index("--train-count") + 1] == "765"
    args = SimpleNamespace(train_count=765, train_split=None, return_types=["disc_0.9"],
                           intended_receiver_model_id=None, run_id="count-run",
                           next_action_conditions_enabled=True)
    for step in generator.full_generation_commands("python", use_carries=True):
        command = generator.with_mode_flags(step.command, args)
        assert "--train-split" not in command
        assert command[command.index("--train-count") + 1] == "765"


@pytest.mark.parametrize("metadata_only", [False, True])
def test_count_model_record_round_trip(monkeypatch, tmp_path, metadata_only):
    from models import utils
    selector = {"train_split": None, "train_count": 765}
    args = {"task": "pass_intent", **({} if metadata_only else selector)}
    (tmp_path / "args.json").write_text(json.dumps(args))
    if metadata_only:
        (tmp_path / "metadata.json").write_text(json.dumps(project_config.split_metadata(selector)))
    monkeypatch.setattr(utils, "get_model_path", lambda model_id: tmp_path)
    record = utils.get_model_record("pass_intent/count-run")
    assert record["train_count"] == 765
    assert record["train_split_percent"] is None


@pytest.mark.parametrize("mismatch", [None, "count", "mode", "manifest", "bundle"])
def test_model_selection_count_consistency(monkeypatch, mismatch):
    from models import utils
    base = {"train_count": 765, "train_split_percent": None, "split_manifest_id": "count-manifest"}
    records = {"a": dict(base), "b": dict(base)}
    bundle = dict(base)
    if mismatch == "count":
        records["b"]["train_count"] = 764
    elif mismatch == "mode":
        records["b"].update(train_count=None, train_split_percent=83)
    elif mismatch == "manifest":
        records["b"]["split_manifest_id"] = "other-universe"
    elif mismatch == "bundle":
        bundle["train_count"] = 764
    monkeypatch.setattr(utils, "resolve_bundle_model_ids", lambda *a, **k: ({"a": "a/1", "b": "b/1"}, bundle))
    monkeypatch.setattr(utils, "get_model_records", lambda *a: records)
    monkeypatch.setattr(utils, "validate_model_record_consistency", lambda *a, **k: {})
    if mismatch:
        with pytest.raises(ValueError, match="split"):
            utils.resolve_model_selection(["a", "b"], bundle_id="bundle")
    else:
        _, shared, _ = utils.resolve_model_selection(["a", "b"], bundle_id="bundle")
        assert shared["train_count"] == 765 and shared["train_split_percent"] is None


def test_expanding_folds_for_recommended_development_size() -> None:
    ids = np.array([f"match_{index:03d}" for index in range(459)])
    folds = project_config.derive_expanding_folds(ids)

    assert [(len(train), len(valid)) for train, valid in folds] == [(229, 77), (306, 76), (382, 77)]
    for train, valid in folds:
        assert set(train).isdisjoint(valid)
        assert list(ids[: len(train)]) == list(train)
        assert list(ids[len(train) : len(train) + len(valid)]) == list(valid)


def test_expanding_folds_enforce_minimum_sizes() -> None:
    with pytest.raises(ValueError, match="100 first-fold training"):
        project_config.derive_expanding_folds([f"match_{index}" for index in range(180)])


@pytest.mark.parametrize("value", [0, 100, -1, 50.5, True])
def test_invalid_train_split_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path, value) -> None:
    configure_split_paths(monkeypatch, tmp_path)
    project_config.save_match_universe([f"match_{index:03d}" for index in range(200)])
    with pytest.raises(ValueError, match="integer percentage"):
        project_config.resolve_split_manifest(value)


def test_third_season_sized_universe_needs_no_season_logic(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    configure_split_paths(monkeypatch, tmp_path)
    project_config.save_match_universe([f"match_{index:04d}" for index in range(918)])
    manifest = project_config.resolve_split_manifest(75)
    folds = project_config.derive_expanding_folds(manifest["train"])

    assert (len(manifest["train"]), len(manifest["test"])) == (688, 230)
    assert [(len(train), len(valid)) for train, valid in folds] == [(344, 114), (458, 115), (573, 115)]


def test_fold_metrics_are_weighted_by_validation_sample_count() -> None:
    summaries = {
        "pass_success": [
            {"validation_matches": 10, "metrics": {"count": 100, "log_loss": 0.4, "brier": 0.2}},
            {"validation_matches": 10, "metrics": {"count": 300, "log_loss": 0.2, "brier": 0.1}},
        ]
    }
    metrics = train_relevant_models._aggregate_fold_metrics(summaries)["pass_success"]
    assert metrics["log_loss"] == pytest.approx(0.25)
    assert metrics["brier"] == pytest.approx(0.125)


def test_learning_curve_has_three_points_and_two_panels(tmp_path) -> None:
    rows = [
        {
            "fold": fold,
            "train_matches": train_matches,
            "validation_matches": 77,
            "best_epoch": 10 + fold,
            "metrics": {"count": 100, "log_loss": 0.5 - fold / 20, "brier": 0.2 - fold / 50},
        }
        for fold, train_matches in ((1, 229), (2, 306), (3, 382))
    ]
    outputs = train_relevant_models._write_learning_curves(tmp_path, {"pass_success": rows})
    csv_text = (tmp_path / "learning_curves" / "pass_success.csv").read_text(encoding="utf-8")
    assert outputs["pass_success"]
    assert (tmp_path / "learning_curves" / "pass_success.png").exists()
    assert csv_text.count("\n") == 4
    assert "459" not in csv_text
