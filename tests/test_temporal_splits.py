from __future__ import annotations

import json

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
