from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import project_config
from datatools import metadata_summary


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summary_path(parent_dir: Path) -> Path:
    return parent_dir / metadata_summary.summary_filename_for_parent(parent_dir)


def patch_summary_roots(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    saved = tmp_path / "saved"
    component_runs = tmp_path / "data" / "component_runs"
    visualizations = tmp_path / "data" / "visualizations"
    monkeypatch.setattr(metadata_summary, "SAVED_DIR", saved)
    monkeypatch.setattr(metadata_summary, "COMPONENT_RUNS_DIR", component_runs)
    monkeypatch.setattr(metadata_summary, "VISUALIZATION_DIR", visualizations)
    return saved, component_runs, visualizations


def model_metadata(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_id": "pass_success/pass_success_1",
        "task": "pass_success",
        "run_id": "pass_success_1",
        "created_at": "2026-06-01T10:00:00",
        "feature_run_id": "feature_1",
        "intended_receiver_mode": "model",
        "target_family": "goal",
        "return_type": "disc_0.7",
        "feature_signature": {
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": False,
            "poss_rel_vel_aware": False,
            "poss_geometry_aware": True,
            "goal_features_aware": False,
            "goal_nodes_aware": True,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": False,
            "v_edge_feature_mode": "all",
        },
        "training_args": {
            "model": "gat",
            "ipw_model_id": "pass_intent/pass_intent_1",
            "batch_size": 256,
        },
        "status": "completed",
        "last_epoch": 12,
        "best_loss": 0.3,
        "best_acc": 0.8,
    }
    payload.update(overrides)
    return payload


def test_saved_model_summary_includes_training_metrics_and_feature_flags(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    run_root = saved / "pass_success" / "pass_success_1"
    write_json(run_root / "metadata.json", model_metadata())

    summary_path = metadata_summary.refresh_summary_for_parent(saved / "pass_success")

    assert summary_path == saved / "pass_success" / "pass_success_metadata_summary.csv"
    rows = read_summary(summary_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["summary_scope"] == "saved_model"
    assert row["model_id"] == "pass_success/pass_success_1"
    assert row["model_role"] == "pass_success"
    assert row["feature_run_id"] == "feature_1"
    assert row["possessor_aware"] == "true"
    assert row["poss_geometry_aware"] == "true"
    assert row["goal_features_aware"] == "false"
    assert row["goal_nodes_aware"] == "true"
    assert row["v_edge_feature_mode"] == "all"
    assert row["model_name"] == "gat"
    assert row["ipw_model_id"] == "pass_intent/pass_intent_1"
    assert row["last_epoch"] == "12"
    assert row["best_loss"] == "0.3"
    assert row["best_acc"] == "0.8"
    assert row["batch_size"] == "256"


def test_bundle_summary_emits_one_row_per_model_with_bundle_id(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    write_json(
        saved / "pass_intent" / "pass_intent_1" / "metadata.json",
        model_metadata(
            model_id="pass_intent/pass_intent_1",
            task="pass_intent",
            run_id="pass_intent_1",
            training_args={"model": "gat", "batch_size": 128},
        ),
    )
    write_json(
        saved / "bundles" / "bundle_1" / "metadata.json",
        {
            "bundle_id": "bundle_1",
            "model_ids": {
                "pass_success": "pass_success/pass_success_1",
                "pass_intent": "pass_intent/pass_intent_1",
            },
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(saved / "bundles")

    rows = read_summary(summary_path)
    assert [row["model_role"] for row in rows] == ["pass_intent", "pass_success"]
    assert {row["bundle_id"] for row in rows} == {"bundle_1"}
    assert {row["run_id"] for row in rows} == {"bundle_1"}
    assert {row["summary_scope"] for row in rows} == {"saved_bundle"}


def test_component_summary_uses_model_records(tmp_path, monkeypatch) -> None:
    saved, component_runs, _ = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    run_root = component_runs / "hawkeye" / "hawkeye_component_1"
    write_json(
        run_root / "metadata.json",
        {
            "run_id": "hawkeye_component_1",
            "created_at": "2026-06-02T10:00:00",
            "model_records": {
                "pass_success": {
                    "model_id": "pass_success/pass_success_1",
                    "feature_run_id": "feature_from_record",
                    "feature_signature": {
                        "accel_aware": False,
                        "poss_geometry_aware": False,
                        "goal_features_aware": True,
                        "goal_nodes_aware": False,
                        "v_edge_feature_mode": "no_poss",
                    },
                    "status": "running",
                }
            },
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(component_runs / "hawkeye")

    rows = read_summary(summary_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["summary_scope"] == "component_run"
    assert row["run_id"] == "hawkeye_component_1"
    assert row["created_at"] == "2026-06-02T10:00:00"
    assert row["feature_run_id"] == "feature_from_record"
    assert row["accel_aware"] == "false"
    assert row["poss_geometry_aware"] == "false"
    assert row["goal_features_aware"] == "true"
    assert row["goal_nodes_aware"] == "false"
    assert row["v_edge_feature_mode"] == "no_poss"
    assert row["status"] == "running"
    assert row["xpass_metric"] == "None"
    assert row["xpass_weight"] == ""


def test_component_summary_records_default_physical_xpass_original_weight(tmp_path, monkeypatch) -> None:
    saved, component_runs, _ = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    run_root = component_runs / "hawkeye" / "hawkeye_component_1"
    write_json(
        run_root / "metadata.json",
        {
            "run_id": "hawkeye_component_1",
            "created_at": "2026-06-02T10:00:00",
            "physical_xpass_requested": True,
            "physical_xpass_metric": "noise_kernel_xpass",
            "physical_xpass_weight_version": "v1",
            "model_records": {"pass_success": {"model_id": "pass_success/pass_success_1"}},
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(component_runs / "hawkeye")

    row = read_summary(summary_path)[0]
    assert row["xpass_metric"] == "noise_kernel"
    assert row["xpass_weight"] == "v1"


def test_component_summary_records_topmean_physical_xpass_v2_weight(tmp_path, monkeypatch) -> None:
    saved, component_runs, _ = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    run_root = component_runs / "benchmark" / "benchmark_component_1"
    write_json(
        run_root / "metadata.json",
        {
            "run_id": "benchmark_component_1",
            "physical_xpass_requested": True,
            "physical_xpass_metric": "topmean_xpass",
            "physical_xpass_weight_version": "v2",
            "models": {"pass_success": "pass_success/pass_success_1"},
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(component_runs / "benchmark")

    row = read_summary(summary_path)[0]
    assert row["xpass_metric"] == "topmean_xpass"
    assert row["xpass_weight"] == "v2"


def test_component_summary_records_max_physical_xpass_from_cache_summary(tmp_path, monkeypatch) -> None:
    saved, component_runs, _ = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    run_root = component_runs / "sportec" / "component_1"
    write_json(
        run_root / "metadata.json",
        {
            "run_id": "component_1",
            "physical_xpass_cache_summary": {"physical_xpass_required": True},
            "physical_xpass_metric": "max_xpass",
            "models": {"pass_success": "pass_success/pass_success_1"},
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(component_runs / "sportec")

    row = read_summary(summary_path)[0]
    assert row["xpass_metric"] == "max_xpass"
    assert row["xpass_weight"] == "v3"


def test_visualization_summary_prefers_selected_model_ids_and_falls_back_to_source_models(tmp_path, monkeypatch) -> None:
    saved, _, visualizations = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "selected" / "metadata.json", model_metadata(model_id="pass_success/selected", run_id="selected"))
    write_json(saved / "pass_success" / "fallback" / "metadata.json", model_metadata(model_id="pass_success/fallback", run_id="fallback"))
    write_json(
        visualizations / "hawkeye" / "viz_selected" / "metadata.json",
        {
            "run_id": "viz_selected",
            "created_at": "2026-06-03T10:00:00",
            "selected_model_ids": {"pass_success": "pass_success/selected"},
            "model_ids": {"pass_success": "pass_success/ignored"},
        },
    )
    write_json(
        visualizations / "hawkeye" / "viz_fallback" / "metadata.json",
        {
            "run_id": "viz_fallback",
            "source_models": {"pass_success": "pass_success/fallback"},
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(visualizations / "hawkeye")

    rows = read_summary(summary_path)
    assert [row["model_id"] for row in rows] == ["pass_success/fallback", "pass_success/selected"]
    assert {row["summary_scope"] for row in rows} == {"visualization"}
    assert {row["xpass_metric"] for row in rows} == {"None"}
    assert {row["xpass_weight"] for row in rows} == {""}


def test_visualization_summary_records_topmean_physical_xpass_v2_weight(tmp_path, monkeypatch) -> None:
    saved, _, visualizations = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    write_json(
        visualizations / "hawkeye" / "viz_1" / "metadata.json",
        {
            "run_id": "viz_1",
            "source_models": {"pass_success": "pass_success/pass_success_1"},
            "physical_xpass_requested": True,
            "physical_xpass_metric": "topmean_xpass",
            "physical_xpass_weight_version": "v2",
            "show_physical_xpass": True,
            "physical_cache_dir": "ignored",
        },
    )

    summary_path = metadata_summary.refresh_summary_for_parent(visualizations / "hawkeye")

    row = read_summary(summary_path)[0]
    assert row["xpass_metric"] == "topmean_xpass"
    assert row["xpass_weight"] == "v2"
    assert "show_physical_xpass" not in row
    assert "physical_cache_dir" not in row


def test_write_run_metadata_refreshes_only_matching_parent_summary(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    run_root = saved / "action_intent" / "action_intent_1"

    project_config.write_run_metadata(
        run_root,
        model_metadata(
            model_id="action_intent/action_intent_1",
            task="action_intent",
            run_id="action_intent_1",
        ),
    )

    assert summary_path(saved / "action_intent").exists()
    assert not summary_path(saved / "pass_success").exists()


def test_refresh_summary_removes_legacy_metadata_summary_csv(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    parent_dir = saved / "pass_success"
    run_root = parent_dir / "pass_success_1"
    write_json(run_root / "metadata.json", model_metadata())
    legacy_path = parent_dir / metadata_summary.LEGACY_SUMMARY_FILENAME
    legacy_path.write_text("stale", encoding="utf-8")

    new_path = metadata_summary.refresh_summary_for_parent(parent_dir)

    assert new_path == summary_path(parent_dir)
    assert new_path.exists()
    assert not legacy_path.exists()


def test_refresh_summary_warns_but_writes_when_legacy_summary_csv_is_locked(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    parent_dir = saved / "pass_success"
    run_root = parent_dir / "pass_success_1"
    write_json(run_root / "metadata.json", model_metadata())
    legacy_path = parent_dir / metadata_summary.LEGACY_SUMMARY_FILENAME
    legacy_path.write_text("stale", encoding="utf-8")
    original_unlink = Path.unlink

    def raise_for_legacy_csv(path: Path, *args: object, **kwargs: object):
        if path == legacy_path:
            raise PermissionError("file is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", raise_for_legacy_csv)

    with pytest.warns(RuntimeWarning, match="legacy metadata summary"):
        new_path = metadata_summary.refresh_summary_for_parent(parent_dir)

    assert new_path == summary_path(parent_dir)
    assert new_path.exists()
    assert legacy_path.exists()


def test_refresh_summary_warns_and_skips_locked_summary_csv(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    run_root = saved / "pass_success" / "pass_success_1"
    write_json(run_root / "metadata.json", model_metadata())
    original_open = Path.open

    def raise_for_summary_csv(path: Path, *args: object, **kwargs: object):
        if path.name == metadata_summary.summary_filename_for_parent(saved / "pass_success"):
            raise PermissionError("file is locked")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raise_for_summary_csv)

    with pytest.warns(RuntimeWarning, match="pass_success_metadata_summary\\.csv"):
        summary_path = metadata_summary.refresh_summary_for_parent(saved / "pass_success")

    assert summary_path is None


def test_write_run_metadata_keeps_metadata_when_summary_csv_is_locked(tmp_path, monkeypatch) -> None:
    saved, _, _ = patch_summary_roots(monkeypatch, tmp_path)
    run_root = saved / "action_intent" / "action_intent_1"
    original_open = Path.open

    def raise_for_summary_csv(path: Path, *args: object, **kwargs: object):
        if path.name == metadata_summary.summary_filename_for_parent(saved / "action_intent"):
            raise PermissionError("file is locked")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raise_for_summary_csv)

    with pytest.warns(RuntimeWarning, match="Skipping metadata summary refresh"):
        metadata_path = project_config.write_run_metadata(
            run_root,
            model_metadata(
                model_id="action_intent/action_intent_1",
                task="action_intent",
                run_id="action_intent_1",
            ),
        )

    assert json.loads(metadata_path.read_text(encoding="utf-8"))["run_id"] == "action_intent_1"


def test_backfill_rebuilds_deterministically_and_skips_malformed_metadata(tmp_path, monkeypatch) -> None:
    saved, component_runs, visualizations = patch_summary_roots(monkeypatch, tmp_path)
    write_json(saved / "pass_success" / "pass_success_1" / "metadata.json", model_metadata())
    write_json(component_runs / "hawkeye" / "hawkeye_component_1" / "metadata.json", {"run_id": "hawkeye_component_1"})
    write_json(visualizations / "hawkeye" / "viz_1" / "metadata.json", {"run_id": "viz_1", "source_models": {"pass_success": "pass_success/pass_success_1"}})
    malformed = saved / "pass_success" / "bad" / "metadata.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")

    paths = metadata_summary.refresh_all_summaries()

    assert paths == [
        summary_path(saved / "pass_success"),
        summary_path(component_runs / "hawkeye"),
        summary_path(visualizations / "hawkeye"),
    ]
    rows = read_summary(summary_path(saved / "pass_success"))
    assert [row["run_id"] for row in rows] == ["pass_success_1"]
