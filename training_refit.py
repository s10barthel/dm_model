"""Validation-derived refit recipes. No source weights or optimizer state are loaded."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from project_config import (SAVED_DIR, VALIDATION_PROTOCOL_SHARED, load_feature_run_metadata,
                            resolve_artifact_split, split_selector, load_model_splits)
from dataset_loading import resolve_dataset_loading, saved_dataset_loading
from models.node_feature_config import resolve_goal_poss_dist, resolve_speed_aware

HISTORY_VERSION = 1
OPERATIONAL_OPTIONS = {
    "--device": "device", "--monitoring": "monitoring", "--pin-memory": "pin_memory",
    "--no-pin-memory": "pin_memory", "--dataset-loading": "dataset_loading",
    "--dataset-buffer-matches": "dataset_buffer_matches", "--dataset-cache-dir": "dataset_cache_dir",
    "--keep-cache": "keep_cache", "--batch-loader": "batch_loader",
    "--crash-dump": "crash_dump", "--procdump-path": "procdump_path",
}


def validate_rate_history(history, through_epoch):
    if isinstance(through_epoch, bool) or not isinstance(through_epoch, int) or through_epoch < 1:
        raise ValueError("Refit source requires a positive validation-selected best_epoch.")
    if not isinstance(history, list) or len(history) < through_epoch:
        raise ValueError("Incomplete structured learning-rate history.")
    result = []
    for epoch, row in enumerate(history[:through_epoch], 1):
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row.get("epoch") != epoch:
            raise ValueError("Learning-rate history must contain consecutive, unique epochs starting at 1.")
        rate = row.get("lr")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Learning-rate history must contain positive finite rates.")
        result.append({"epoch": epoch, "lr": float(rate)})
    return result


def append_rate_history(metadata, epoch, rate):
    """Legacy resumed runs cannot claim complete history for their missing prefix."""
    history = metadata.setdefault("learning_rate_history", [])
    if metadata.get("learning_rate_history_version") != HISTORY_VERSION:
        return
    validate_rate_history([*history, {"epoch": epoch, "lr": rate}], epoch)
    if len(history) != epoch - 1:
        raise ValueError("Cannot append a duplicate or out-of-order history epoch.")
    history.append({"epoch": epoch, "lr": float(rate)})


def apply_refit_rate(optimizer, recipe, epoch):
    rate = recipe["learning_rate_history"][epoch - 1]["lr"]
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate


class ValidationExampleIdentity:
    """Fingerprint examples and labels during the existing validation pass."""
    def __init__(self):
        self.digest = hashlib.sha256()
        self.count = 0

    def update(self, graphs, labels):
        matches = list(getattr(graphs, "evaluation_match_id", []))
        rows = getattr(graphs, "evaluation_source_row", None)
        if rows is None or len(matches) != len(labels) or len(rows) != len(labels):
            raise ValueError("Validation examples lack stable match/source-row identities.")
        for match, row, label in zip(matches, rows.tolist(), labels):
            self.digest.update(json.dumps([match, row, label.tolist()], separators=(",", ":")).encode("utf-8"))
            self.digest.update(b"\n")
            self.count += 1

    def result(self):
        return {"count": self.count, "sha256": self.digest.hexdigest()}


def source_path(model_id, saved_root=None):
    parts = str(model_id).split("/")
    if len(parts) not in (2, 3) or any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
        raise ValueError("Refit sources must be TASK/RUN[/STAGE].")
    if parts[0] in ("bundles", "refit_batches"):
        raise ValueError("Refit sources must be individual validation models.")
    return Path(SAVED_DIR if saved_root is None else saved_root).joinpath(*parts)


def prepare_refit(model_id, run_id, *, operational=None, saved_root=None):
    """Preflight a source and reconstruct its exact saved configuration."""
    directory = source_path(model_id, saved_root)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    settings = json.loads((directory / "args.json").read_text(encoding="utf-8"))
    task = model_id.split("/", 1)[0]
    if metadata.get("status") != "completed" or metadata.get("task") != task or settings.get("task") != task:
        raise ValueError(f"Refit source {model_id} must be a completed matching validation model.")
    if metadata.get("final_refit") or settings.get("final_refit") or not metadata.get("validation_match_ids"):
        raise ValueError(f"Refit source {model_id} must have validation data and cannot be a refit.")
    if not (directory / "best_weights.pt").is_file():
        raise ValueError(f"Refit source {model_id} is missing its completed checkpoint weights.")
    if metadata.get("learning_rate_history_version") != HISTORY_VERSION:
        raise ValueError(f"Refit source {model_id} lacks supported structured learning-rate history.")
    best_epoch = metadata.get("best_epoch")
    history = validate_rate_history(metadata.get("learning_rate_history"), metadata.get("last_epoch"))
    if len(history) != len(metadata["learning_rate_history"]):
        raise ValueError("Source history length disagrees with its completed epochs.")
    history = validate_rate_history(history, best_epoch)
    feature_id = settings.get("feature_run_id")
    if not feature_id:
        raise ValueError("Refit source requires a pinned feature run.")
    feature_metadata = load_feature_run_metadata(feature_id, required=True)
    manifest, _ = resolve_artifact_split({}, feature_metadata,
        feature_run_id=feature_id, checkpoint=settings)
    recorded_manifest, _ = resolve_artifact_split({}, feature_metadata,
        feature_run_id=feature_id, checkpoint=metadata)
    if recorded_manifest["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Source configuration and metadata split provenance disagree.")
    train_ids = metadata.get("train_match_ids") or []
    if not train_ids or len(train_ids) != len(set(train_ids)) or not set(train_ids).issubset(manifest["train"]):
        raise ValueError("Refit source training membership conflicts with its saved development pool.")
    if len(metadata["validation_match_ids"]) != len(set(metadata["validation_match_ids"])) or not set(metadata["validation_match_ids"]).issubset(manifest["train"]):
        raise ValueError("Refit source validation membership conflicts with its saved development pool.")
    if set(metadata.get("train_match_ids") or []) & set(metadata["validation_match_ids"]):
        raise ValueError("Refit source training and validation memberships overlap.")
    expected_train, expected_validation, _ = load_model_splits(manifest=manifest,
        validation_mode=settings.get("validation_mode", "holdout_80_20"),
        validation_protocol=settings.get("validation_protocol"),
        validation_fold=settings.get("validation_fold"), training_count=settings.get("training_count"),
        validation_count=settings.get("validation_count"), validation_fraction=settings.get("validation_fraction"))
    if train_ids != expected_train.tolist() or metadata["validation_match_ids"] != expected_validation.tolist():
        raise ValueError("Refit source membership disagrees with its saved split configuration.")
    for key, value in settings.items():
        if value and isinstance(value, str) and (key.endswith("feature_dir") or key.endswith("label_dir")
                or key in ("physical_cache_dir", "lane_survival_cache_dir")):
            if not Path(value).is_dir():
                raise ValueError(f"Refit source artifact directory is unavailable: {key}={value}")
    dependency_id = settings.get("ipw_model_id", "none")
    if dependency_id != "none":
        dependency_dir = source_path(dependency_id, saved_root)
        dependency_meta = json.loads((dependency_dir / "metadata.json").read_text(encoding="utf-8"))
        if dependency_meta.get("status") != "completed" or dependency_meta.get("split_manifest_id") != manifest["manifest_id"]:
            raise ValueError("Refit source propensity dependency is incomplete or has incompatible split provenance.")
        weights = dependency_dir / "best_weights.pt"
        fingerprint = hashlib.sha256(weights.read_bytes()).hexdigest()
        recorded = metadata.get("ipw_dependency_weights_sha256")
        if not recorded or fingerprint != recorded:
            raise ValueError("Refit source propensity dependency weights do not match recorded provenance.")
        dependency_stage = dependency_meta.get("artifact_stage")
        if len(dependency_id.split("/")) == 2 and dependency_stage:
            immutable_id = f"{dependency_id}/{dependency_stage}"
            immutable_dir = source_path(immutable_id, saved_root)
            immutable_meta = json.loads((immutable_dir / "metadata.json").read_text(encoding="utf-8"))
            if (immutable_meta.get("status") != "completed"
                    or immutable_meta.get("split_manifest_id") != manifest["manifest_id"]
                    or hashlib.sha256((immutable_dir / "best_weights.pt").read_bytes()).hexdigest() != recorded):
                raise ValueError("Immutable propensity stage differs from recorded source dependency.")
            dependency_id = immutable_id
    # Read the immutable stage's complete configuration when a published root is supplied.
    stage = settings.get("artifact_stage")
    if len(model_id.split("/")) == 2 and stage and (directory / stage / "metadata.json").is_file():
        canonical_id = f"{model_id}/{stage}"
        stage_meta = json.loads((directory / stage / "metadata.json").read_text(encoding="utf-8"))
        if (stage_meta.get("learning_rate_history") != metadata.get("learning_rate_history")
                or stage_meta.get("best_epoch") != best_epoch):
            raise ValueError("Published source and immutable stage recipes disagree.")
        return prepare_refit(canonical_id, run_id, operational=operational, saved_root=saved_root)
    canonical_id = model_id
    recipe = {"version": 1, "source_model_id": canonical_id, "best_epoch": best_epoch,
              "learning_rate_history": history, "rollback_replayed": False,
              "source_optimizer_policy": metadata.get("optimizer_policy"),
              "split_manifest_id": manifest["manifest_id"],
              "ipw_dependency_weights_sha256": metadata.get("ipw_dependency_weights_sha256")}
    args = copy.deepcopy(settings)
    from models.feature_masks import normalize_feature_settings
    normalize_feature_settings(args)
    from models.architecture import normalize_architecture
    normalize_architecture(args)
    from models.edge_schema import normalize_input_settings
    normalize_input_settings(args)
    args["goal_poss_dist"] = resolve_goal_poss_dist(settings)
    args["speed_aware"] = resolve_speed_aware(settings)
    args.update(split_selector(manifest))
    args.update(run_id=run_id, model_id=f"{task}/{run_id}", ipw_model_id=dependency_id, trial=None, cont=False,
                resume_run_id=None, resume_checkpoint=None, final_refit=True, artifact_stage="final_refit",
                n_epochs=best_epoch, early_stopping=False, lr_rollback=False, best_loss=None, best_acc=None,
                validation_protocol=VALIDATION_PROTOCOL_SHARED, validation_fold=None,
                training_count=None, validation_count=None, validation_fraction=None, refit_recipe=recipe,
                split_manifest=manifest, split_manifest_id=manifest["manifest_id"],
                training_step_index=None, training_step_total=None)
    source_loading = saved_dataset_loading(settings)
    args["dataset_loading"] = source_loading
    for key in ("refit_config", "resolved_dataset_loading", "source_identity", "cache_session"):
        args.pop(key, None)
    overrides = operational or {}
    if set(overrides) - set(OPERATIONAL_OPTIONS.values()):
        raise ValueError("Refit permits operational overrides only.")
    args.update(overrides)
    target_loading = resolve_dataset_loading(args.get("dataset_loading", "auto"), task, dependency_id)
    if target_loading != source_loading:
        raise ValueError("Refit data-loading overrides cannot change memory/disk sampling semantics.")
    if source_loading == "disk" and args.get("dataset_buffer_matches", 4) != settings.get("dataset_buffer_matches", 4):
        raise ValueError("Changing prepared dataset buffer size changes sample order; retain the source setting.")
    return {"args": args, "recipe": recipe}


def load_refit_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    args, recipe = payload["args"], payload["recipe"]
    if recipe.get("version") != 1 or recipe.get("rollback_replayed") is not False:
        raise ValueError("Unsupported refit recipe.")
    history = validate_rate_history(recipe.get("learning_rate_history"), recipe.get("best_epoch"))
    if len(history) != len(recipe["learning_rate_history"]) or args.get("refit_recipe") != recipe:
        raise ValueError("Refit configuration and recipe disagree.")
    if not args.get("final_refit") or args.get("n_epochs") != recipe["best_epoch"] or args.get("cont") or args.get("resume_run_id"):
        raise ValueError("Refit configuration must initialize a fresh full-development fit.")
    return args
