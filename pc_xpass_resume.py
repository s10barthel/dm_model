"""Model-independent source contracts for versioned runtime caches."""
from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import torch

import pc_xpass_versions as versions
import pc_xpass_match_cache as match_cache
from datatools.possession_frames import STATE_CONTRACT
from pc_xpass_contract import PHYSICAL_INPUT_CONTRACT

MODEL_COLUMN = "pass_height_model_fingerprint"
INPUT_COLUMN = "pass_height_input_sha256"
DATASETS = ("sportec", "skillcorner", "benchmark", "hawkeye")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def model_identity(args):
    record = args._pass_height_model_record or {}
    model_path = Path(record["model_path"])
    weights = model_path / "best_weights.pt"
    if not weights.exists():
        weights = model_path / "best_model.json"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"model_id": args.pass_height_model_id, "artifact_sha256": digest.hexdigest(),
            "definition": record.get("pass_height_definition"),
            "graph_schema": record.get("graph_schema")}


def input_digest(graph, label, row, *, omit_missing=False):
    digest = hashlib.sha256()
    for key, value in sorted(graph.to_dict().items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    digest.update(label.detach().cpu().contiguous().numpy().tobytes())
    # Height models may consume cached lane-survival metrics as features.
    digest.update(json.dumps({k: v.item() if hasattr(v, "item") else v for k, v in row.items()
                              if k not in (MODEL_COLUMN, INPUT_COLUMN)
                              and not k.endswith("__pass_height")
                              and not (omit_missing and pd.api.types.is_scalar(v) and pd.isna(v))},
                             sort_keys=True, default=str).encode())
    return digest.hexdigest()


def root(args):
    return Path(args._pc_version_root)


def enabled(args):
    return bool(getattr(args, "pc_xpass", False) and getattr(args, "_pc_version_root", None)
                and not getattr(args, "_pc_location", False))


class MatchInventory(Mapping):
    """Read one match at a time, including interrupted publications."""
    def __init__(self, directory):
        self.directory = directory
        self.paths = {path.stem: path for path in sorted((directory / "matches").glob("*.parquet"))
                      if not path.name.startswith(".")}
        for record in (directory / "completion").glob("*.json"):
            self.paths.setdefault(record.stem, directory / "matches" / f"{record.stem}.parquet")

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def __getitem__(self, key):
        record = match_cache.read_json(self.directory / "completion" / f"{key}.json")
        if record.get("status") == "pending":
            frame = pd.DataFrame()
            frame.attrs["pending_match_commit"] = True
            return frame
        return pd.read_parquet(self.paths[key])


def inventory(cache_root, *, datasets=DATASETS, lazy_possessions=False):
    return {dataset: (MatchInventory(cache_root / dataset)
                      if lazy_possessions and dataset in match_cache.DATASETS else
                      {path.stem: pd.read_parquet(path)
                       for path in sorted((cache_root / dataset / "matches").glob("*.parquet"))
                       if not path.name.startswith(".")})
            for dataset in datasets}


def dataset_metadata(cache_root, dataset):
    path = cache_root / dataset / "metadata.json"
    return versions.read_metadata(path.parent) if path.exists() else {}


def prepare_contracts(args, selected):
    """Validate physical source contracts without writes."""
    if not getattr(args, "pc_xpass", False) or getattr(args, "_pc_location", False):
        return
    cache_root = root(args) if getattr(args, "_pc_version_root", None) else versions.config.PC_XPASS_DIR / ".new-generation"
    metadata = versions.read_metadata(cache_root) if (cache_root / "metadata.json").exists() else {}
    import physical_pass_model as physics
    kwargs = {key: getattr(args, key) for key in inspect.signature(physics.pc_xpass_metadata).parameters
              if key != "teammate_policy" and hasattr(args, key)}
    effective = physics.pc_xpass_metadata(
        physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER if args.consider_teammates else physics.PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
        **kwargs)
    dependencies = {key: effective.get(key) for key in (
        "physics_version", "ranking_mode", "lane_function", "control_function", "lane_survival_aggregation",
        "top_pass_definition", "position_discount_function", "xt_surface", "reachability_config")}
    dependencies["physical_input_contract"] = PHYSICAL_INPUT_CONTRACT
    if effective.get("high_pass_config"):
        dependencies["high_pass_config"] = effective["high_pass_config"]
    if effective.get("uncertainty_config"):
        dependencies["uncertainty_config"] = effective["uncertainty_config"]
    if effective["ball_dec"] == "drag":
        dependencies["ball_drag"] = effective["ball_drag"]
    if dependencies["xt_surface"]:
        dependencies["xt_surface"] = {key: value for key, value in dependencies["xt_surface"].items() if key != "path"}
    saved_dependencies = metadata.get("computation_dependencies")
    if saved_dependencies is not None and saved_dependencies != dependencies:
        raise ValueError("pc-xPass computation dependencies changed; create a new version.")
    for dataset in DATASETS:
        previous = dataset_metadata(cache_root, dataset)
        for key, value in dependencies.items():
            if key not in previous:
                continue
            cached = previous[key]
            if key == "xt_surface" and cached:
                cached = {k: v for k, v in cached.items() if k != "path"}
            if cached != value:
                raise ValueError(f"Legacy pc-xPass computation dependency changed: {dataset}/{key}")
    args._pc_computation_dependencies = dependencies
    contracts = copy.deepcopy(metadata.get("dataset_contracts", {}))
    explicit = args._pc_explicit
    needed = set(selected)
    history = metadata.get("invocations", [])
    for dataset in needed:
        if dataset in match_cache.DATASETS:
            match_cache.require_format(cache_root / dataset, allow_new=True)
        old = contracts.get(dataset, {})
        source = dataset_metadata(cache_root, dataset).get("source_inputs", {})
        if dataset in {"sportec", "skillcorner"}:
            saved_state_contract = old.get("state_contract") or source.get("state_contract")
            if saved_state_contract is not None and saved_state_contract != match_cache.state_contract(cache_root / dataset):
                raise ValueError(f"Incompatible {dataset} state contract; create a new version.")
        if dataset == "sportec":
            saved = old.get("feature_run_id") or source.get("feature_run_id")
            prior = {entry["arguments"].get("sportec_feature_run_id") for entry in history
                     if entry.get("arguments", {}).get("sportec_feature_run_id")}
            if not saved and len(prior) == 1:
                saved = prior.pop()
            requested = args.sportec_feature_run_id
            if saved and "sportec_feature_run_id" in explicit and requested != saved:
                raise ValueError("Sportec feature run differs from this version; create a new version.")
            if not saved and any((cache_root / "sportec" / "matches").glob("*.parquet")) and not requested:
                raise ValueError("Original Sportec feature run is unknown; provide --sportec-feature-run-id explicitly.")
            from project_config import resolve_feature_run_id
            saved = saved or resolve_feature_run_id(requested, required=True, allow_latest=True)
            args.sportec_feature_run_id = saved
            contracts[dataset] = {**old, "feature_run_id": saved, "state_contract": match_cache.state_contract(cache_root / dataset)}
        elif dataset == "hawkeye":
            saved = old.get("freeze_ballreceipt")
            if saved is None:
                prior = {entry["arguments"].get("freeze_ballreceipt", True) for entry in history
                         if not entry.get("arguments", {}).get("no_hawkeye", False)}
                if len(prior) > 1:
                    raise ValueError("Legacy Hawkeye cache has conflicting freeze_ballreceipt settings.")
                saved = next(iter(prior)) if prior else None
            if saved is not None and "freeze_ballreceipt" in explicit and args.freeze_ballreceipt != saved:
                raise ValueError("Hawkeye freeze_ballreceipt differs from this version.")
            args.freeze_ballreceipt = args.freeze_ballreceipt if saved is None else saved
            contracts[dataset] = {**old, "freeze_ballreceipt": args.freeze_ballreceipt}
        else:
            contracts.setdefault(dataset, {"state_contract": STATE_CONTRACT} if dataset == "skillcorner" else {})
    args._pc_dataset_contracts = contracts
    for contract in contracts.values():
        contract["physical_input_contract"] = PHYSICAL_INPUT_CONTRACT


def persist_contracts(args):
    if not enabled(args) or args.dry_run:
        return
    metadata = versions.read_metadata(root(args))
    metadata["dataset_contracts"] = args._pc_dataset_contracts
    metadata["computation_dependencies"] = args._pc_computation_dependencies
    metadata["resume_schema_version"] = 1
    versions.atomic_json(root(args) / "metadata.json", metadata)


def record_source(args, dataset, match_id, descriptor):
    transaction = match_cache.active(root(args) / dataset) if enabled(args) else None
    if transaction is not None:
        transaction.register(str(match_id), descriptor)
        return
    if not enabled(args):
        return
    metadata = versions.read_metadata(root(args)) if (root(args) / "metadata.json").exists() else {}
    previous = metadata.get("reconstruction", {}).get(dataset, {}).get(str(match_id))
    logical_source = None
    if dataset == "skillcorner":
        logical_source = fingerprint([descriptor["match_id"], descriptor["possession_id"]])
        definition = {key: descriptor.get(key) for key in
                      ("possessor_id", "original_start_frame", "original_end_frame")}
        saved_definition = metadata.get("skillcorner_possessions", {}).get(logical_source)
        if saved_definition is not None and saved_definition != definition:
            raise ValueError(f"Changed SkillCorner possession definition for {descriptor['match_id']}:{descriptor['possession_id']}")
    if dataset == "sportec":
        artifact = {k: descriptor[k] for k in ("feature_run_id", "state_contract", "canonical_sha256",
                    "control_sha256", "resolved_actions_sha256", "tracking_sha256", "lineup_sha256", "physical_input_contract") if k in descriptor}
        old_artifact = metadata.get("sportec_artifacts", {}).get(descriptor["match_id"])
        if old_artifact and old_artifact != artifact:
            raise ValueError(f"Changed Sportec source provenance for {descriptor['match_id']}")
    if previous:
        # Locators can move, but state-defining provenance cannot change.
        for key in ("feature_run_id", "spell_artifact_sha256", "carry_definition", "possessor_id"):
            if key in previous and descriptor.get(key) != previous[key]:
                raise ValueError(f"Changed {dataset} provenance for {match_id}: {key}")
    if args.dry_run:
        return
    metadata.setdefault("reconstruction", {}).setdefault(dataset, {})[str(match_id)] = descriptor
    if dataset == "sportec":
        metadata.setdefault("sportec_artifacts", {})[descriptor["match_id"]] = artifact
    if logical_source is not None:
        metadata.setdefault("skillcorner_possessions", {})[logical_source] = definition
    versions.atomic_json(root(args) / "metadata.json", metadata)


def assert_height_readable(metadata):
    if metadata.get("pass_height_refresh_pending"):
        raise ValueError("Pass-height refresh is pending; resume cache generation before reading height predictions.")
