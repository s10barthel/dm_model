from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_DIR = PROJECT_ROOT / "saved"
COMPONENT_RUNS_DIR = PROJECT_ROOT / "data" / "component_runs"
VISUALIZATION_DIR = PROJECT_ROOT / "data" / "visualizations"

LEGACY_SUMMARY_FILENAME = "metadata_summary.csv"
SUMMARY_FILENAME = LEGACY_SUMMARY_FILENAME
SUMMARY_COLUMNS = [
    "summary_scope",
    "parent_group",
    "run_id",
    "model_role",
    "model_id",
    "bundle_id",
    "created_at",
    "feature_run_id",
    "intended_receiver_mode",
    "target_family",
    "return_type",
    "xy_only",
    "possessor_aware",
    "keeper_aware",
    "ball_z_aware",
    "poss_vel_aware",
    "poss_rel_vel_aware",
    "poss_geometry_aware",
    "goal_features_aware",
    "goal_nodes_aware",
    "accel_aware",
    "offside_aware",
    "extend_features",
    "v_edge_feature_mode",
    "relative_speed_edge_feature_mode",
    "status",
    "model_name",
    "ipw_model_id",
    "last_epoch",
    "best_loss",
    "best_acc",
    "batch_size",
    "xpass_metric",
    "xpass_weight",
]

FEATURE_COLUMNS = [
    "xy_only",
    "possessor_aware",
    "keeper_aware",
    "ball_z_aware",
    "poss_vel_aware",
    "poss_rel_vel_aware",
    "poss_geometry_aware",
    "goal_features_aware",
    "goal_nodes_aware",
    "accel_aware",
    "offside_aware",
    "extend_features",
    "v_edge_feature_mode",
    "relative_speed_edge_feature_mode",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _blank_row(summary_scope: str, parent_group: str) -> dict[str, Any]:
    row = {column: "" for column in SUMMARY_COLUMNS}
    row["summary_scope"] = summary_scope
    row["parent_group"] = parent_group
    return row


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _model_path(model_id: str) -> Path | None:
    parts = str(model_id).split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return SAVED_DIR / parts[0] / parts[1]


def _load_model_metadata(model_id: str | None) -> dict[str, Any]:
    if not model_id:
        return {}
    path = _model_path(str(model_id))
    if path is None:
        return {}
    return _read_json(path / "metadata.json") or {}


def _metadata_args(metadata: dict[str, Any]) -> dict[str, Any]:
    args = metadata.get("training_args")
    return args if isinstance(args, dict) else {}


def _metadata_feature_signature(metadata: dict[str, Any], record: dict[str, Any] | None = None) -> dict[str, Any]:
    if record:
        signature = record.get("feature_signature")
        if isinstance(signature, dict):
            return signature
    signature = metadata.get("feature_signature")
    if isinstance(signature, dict):
        return signature
    args = _metadata_args(metadata)
    return {key: args.get(key) for key in FEATURE_COLUMNS if key in args}


def _model_name(metadata: dict[str, Any], record: dict[str, Any] | None = None) -> Any:
    if record and record.get("model_name") is not None:
        return record.get("model_name")
    args = _metadata_args(metadata)
    return _first_value(metadata.get("model_name"), metadata.get("model"), args.get("model"))


def _ipw_model_id(metadata: dict[str, Any], record: dict[str, Any] | None = None) -> Any:
    if record and record.get("ipw_model_id") is not None:
        return record.get("ipw_model_id")
    args = _metadata_args(metadata)
    return _first_value(metadata.get("ipw_model_id"), args.get("ipw_model_id"))


def _batch_size(metadata: dict[str, Any], record: dict[str, Any] | None = None) -> Any:
    if record and record.get("batch_size") is not None:
        return record.get("batch_size")
    args = _metadata_args(metadata)
    return _first_value(metadata.get("batch_size"), args.get("batch_size"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _physical_xpass_used(metadata: dict[str, Any]) -> bool:
    if "physical_xpass_requested" in metadata:
        return _truthy(metadata.get("physical_xpass_requested"))
    summary = metadata.get("physical_xpass_cache_summary")
    if isinstance(summary, dict) and "physical_xpass_required" in summary:
        return _truthy(summary.get("physical_xpass_required"))
    if metadata.get("physical_xpass_metric") is not None:
        return True
    if metadata.get("physical_xpass_weight_version") is not None:
        return True
    return False


def _summary_xpass_metric(metadata: dict[str, Any]) -> str:
    if not _physical_xpass_used(metadata):
        return "None"
    version = str(metadata.get("x_pass_version") or "").strip().lower().replace("_", "-")
    if version:
        if version == "noise-kernel":
            return "noise_kernel"
        if version == "max":
            return "max_xpass"
        if version.startswith("top") and version[3:].isdigit():
            return version
    metric = str(metadata.get("physical_xpass_metric") or "").strip().lower()
    if metric in {"", "noise_kernel", "noise_kernel_xpass"}:
        return "noise_kernel"
    if metric in {"max", "max_xpass"}:
        return "max_xpass"
    if metric in {"top10", "top10_xpass"}:
        return "top10_xpass"
    if metric in {"top25", "top25_xpass"}:
        return "top25_xpass"
    if metric in {"topmean", "topmean_xpass", "top10mean", "top10mean_xpass"}:
        return "topmean_xpass"
    return metric


def _summary_xpass_weight(metadata: dict[str, Any]) -> str:
    if _summary_xpass_metric(metadata) == "None":
        return ""
    weight_version = str(metadata.get("physical_xpass_weight_version") or "").strip().lower()
    if weight_version in {"v1", "v2", "v3"}:
        return weight_version
    if weight_version == "":
        return "v3"
    if weight_version == "v2":
        return "v2"
    return weight_version


def _fill_xpass_summary_fields(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    row["xpass_metric"] = _summary_xpass_metric(metadata)
    row["xpass_weight"] = _summary_xpass_weight(metadata)
    return row


def _fill_model_fields(
    row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model_id: str | None = None,
    model_role: str | None = None,
    record: dict[str, Any] | None = None,
    prefer_run_created_at: str | None = None,
) -> dict[str, Any]:
    args = _metadata_args(metadata)
    signature = _metadata_feature_signature(metadata, record)
    row["model_role"] = _first_value(model_role, (record or {}).get("task"), metadata.get("task"), args.get("task"), "")
    row["model_id"] = _first_value(model_id, (record or {}).get("model_id"), metadata.get("model_id"), args.get("model_id"), "")
    row["created_at"] = _first_value(prefer_run_created_at, (record or {}).get("created_at"), metadata.get("created_at"))
    row["feature_run_id"] = _first_value(
        (record or {}).get("feature_run_id"),
        metadata.get("feature_run_id"),
        args.get("feature_run_id"),
    )
    row["intended_receiver_mode"] = _first_value(
        (record or {}).get("intended_receiver_mode"),
        metadata.get("intended_receiver_mode"),
        args.get("intended_receiver_mode"),
    )
    row["target_family"] = _first_value(
        (record or {}).get("target_family"),
        metadata.get("target_family"),
        args.get("target_family"),
    )
    row["return_type"] = _first_value(
        (record or {}).get("return_type"),
        metadata.get("return_type"),
        args.get("return_type"),
    )
    for key in FEATURE_COLUMNS:
        row[key] = signature.get(key, args.get(key, ""))
    row["status"] = _first_value((record or {}).get("status"), metadata.get("status"))
    row["model_name"] = _model_name(metadata, record)
    row["ipw_model_id"] = _ipw_model_id(metadata, record)
    row["last_epoch"] = _first_value(metadata.get("last_epoch"), args.get("last_epoch"))
    row["best_loss"] = _first_value(metadata.get("best_loss"), args.get("best_loss"))
    row["best_acc"] = _first_value(metadata.get("best_acc"), args.get("best_acc"))
    row["batch_size"] = _batch_size(metadata, record)
    return row


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def summary_filename_for_parent(parent_dir: Path) -> str:
    return f"{Path(parent_dir).name}_metadata_summary.csv"


def summary_path_for_parent(parent_dir: Path) -> Path:
    parent_dir = Path(parent_dir)
    return parent_dir / summary_filename_for_parent(parent_dir)


def _remove_legacy_summary(parent_dir: Path, summary_path: Path) -> None:
    legacy_path = Path(parent_dir) / LEGACY_SUMMARY_FILENAME
    if legacy_path == summary_path or not legacy_path.exists():
        return
    try:
        legacy_path.unlink()
    except PermissionError as exc:
        warnings.warn(
            f"Could not remove legacy metadata summary {legacy_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _write_rows(parent_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    summary_path = summary_path_for_parent(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    try:
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: _csv_value(row.get(column)) for column in SUMMARY_COLUMNS})
    except PermissionError as exc:
        warnings.warn(
            f"Skipping metadata summary refresh for {summary_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    _remove_legacy_summary(parent_dir, summary_path)
    return summary_path


def _summary_scope_for_parent(parent_dir: Path) -> str | None:
    parent_dir = Path(parent_dir).resolve()
    saved = SAVED_DIR.resolve()
    components = COMPONENT_RUNS_DIR.resolve()
    visualizations = VISUALIZATION_DIR.resolve()
    if parent_dir.parent == saved:
        return "saved_bundle" if parent_dir.name == "bundles" else "saved_model"
    if parent_dir.parent == components:
        return "component_run"
    if parent_dir.parent == visualizations:
        return "visualization"
    return None


def _iter_run_metadata(parent_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for child in sorted(path for path in Path(parent_dir).iterdir() if path.is_dir()):
        metadata = _read_json(child / "metadata.json")
        if metadata is not None:
            yield child, metadata


def extract_model_rows(run_root: Path, metadata: dict[str, Any], parent_group: str) -> list[dict[str, Any]]:
    row = _blank_row("saved_model", parent_group)
    row["run_id"] = metadata.get("run_id") or run_root.name
    model_id = metadata.get("model_id") or f"{parent_group}/{run_root.name}"
    _fill_model_fields(row, metadata, model_id=str(model_id))
    return [row]


def extract_bundle_rows(run_root: Path, metadata: dict[str, Any], parent_group: str) -> list[dict[str, Any]]:
    model_ids = metadata.get("model_ids")
    if not isinstance(model_ids, dict):
        return []
    bundle_id = str(metadata.get("bundle_id") or run_root.name)
    rows: list[dict[str, Any]] = []
    for role, model_id in sorted(model_ids.items()):
        if not model_id:
            continue
        model_metadata = _load_model_metadata(str(model_id))
        row = _blank_row("saved_bundle", parent_group)
        row["run_id"] = bundle_id
        row["bundle_id"] = bundle_id
        _fill_model_fields(row, model_metadata, model_id=str(model_id), model_role=str(role))
        rows.append(row)
    return rows


def extract_component_rows(run_root: Path, metadata: dict[str, Any], parent_group: str) -> list[dict[str, Any]]:
    run_id = str(metadata.get("run_id") or run_root.name)
    records = metadata.get("model_records")
    if isinstance(records, dict) and records:
        items = [(str(role), str(record.get("model_id") or ""), record) for role, record in records.items() if isinstance(record, dict)]
    else:
        models = metadata.get("models")
        items = [(str(role), str(model_id), {}) for role, model_id in models.items()] if isinstance(models, dict) else []

    rows: list[dict[str, Any]] = []
    for role, model_id, record in sorted(items):
        if not model_id:
            continue
        model_metadata = _load_model_metadata(model_id)
        row = _blank_row("component_run", parent_group)
        row["run_id"] = run_id
        row["bundle_id"] = metadata.get("bundle_id", "")
        _fill_model_fields(
            row,
            model_metadata,
            model_id=model_id,
            model_role=role,
            record=record,
            prefer_run_created_at=metadata.get("created_at"),
        )
        _fill_xpass_summary_fields(row, metadata)
        rows.append(row)
    return rows


def _visualization_model_ids(metadata: dict[str, Any]) -> dict[str, Any]:
    for key in ("selected_model_ids", "model_ids", "source_models"):
        values = metadata.get(key)
        if isinstance(values, dict) and values:
            return values
    return {}


def extract_visualization_rows(run_root: Path, metadata: dict[str, Any], parent_group: str) -> list[dict[str, Any]]:
    run_id = str(metadata.get("run_id") or run_root.name)
    rows: list[dict[str, Any]] = []
    for role, model_id in sorted(_visualization_model_ids(metadata).items()):
        if not model_id:
            continue
        model_metadata = _load_model_metadata(str(model_id))
        row = _blank_row("visualization", parent_group)
        row["run_id"] = run_id
        row["bundle_id"] = metadata.get("bundle_id", "")
        _fill_model_fields(
            row,
            model_metadata,
            model_id=str(model_id),
            model_role=str(role),
            prefer_run_created_at=metadata.get("created_at"),
        )
        _fill_xpass_summary_fields(row, metadata)
        rows.append(row)
    return rows


def rows_for_run(run_root: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    parent_dir = Path(run_root).parent
    parent_group = parent_dir.name
    scope = _summary_scope_for_parent(parent_dir)
    if scope == "saved_model":
        return extract_model_rows(Path(run_root), metadata, parent_group)
    if scope == "saved_bundle":
        return extract_bundle_rows(Path(run_root), metadata, parent_group)
    if scope == "component_run":
        return extract_component_rows(Path(run_root), metadata, parent_group)
    if scope == "visualization":
        return extract_visualization_rows(Path(run_root), metadata, parent_group)
    return []


def refresh_summary_for_parent(parent_dir: Path) -> Path | None:
    parent_dir = Path(parent_dir)
    if _summary_scope_for_parent(parent_dir) is None:
        return None
    rows: list[dict[str, Any]] = []
    for run_root, metadata in _iter_run_metadata(parent_dir):
        rows.extend(rows_for_run(run_root, metadata))
    rows.sort(key=lambda row: (str(row.get("run_id", "")), str(row.get("model_role", "")), str(row.get("model_id", ""))))
    return _write_rows(parent_dir, rows)


def update_summary_for_run(run_root: Path) -> Path | None:
    return refresh_summary_for_parent(Path(run_root).parent)


def refresh_all_summaries() -> list[Path]:
    summary_paths: list[Path] = []
    for root in (SAVED_DIR, COMPONENT_RUNS_DIR, VISUALIZATION_DIR):
        if not root.exists():
            continue
        for parent_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            summary_path = refresh_summary_for_parent(parent_dir)
            if summary_path is not None:
                summary_paths.append(summary_path)
    return summary_paths
