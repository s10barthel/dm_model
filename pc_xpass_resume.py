"""Input contracts and transactional enrichment for versioned runtime caches.

Dataset runners reconstruct states; refresh consumes only exact saved identities.
No physical calculation or lane-control write is performed by a refresh.
"""
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
    """Validate existing contracts without writes, including legacy recovery."""
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
    if args.pass_height_model_id:
        needed.update(d for d in DATASETS if any((cache_root / d / "matches").glob("*.parquet")))
    history = metadata.get("invocations", [])
    for dataset in needed:
        if dataset in match_cache.DATASETS:
            match_cache.require_format(cache_root / dataset, allow_new=True)
        old = contracts.get(dataset, {})
        source = dataset_metadata(cache_root, dataset).get("source_inputs", {})
        if dataset in {"sportec", "skillcorner"}:
            saved_state_contract = old.get("state_contract") or source.get("state_contract")
            if saved_state_contract is not None and saved_state_contract != STATE_CONTRACT:
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
            contracts[dataset] = {**old, "feature_run_id": saved, "state_contract": STATE_CONTRACT}
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
        refresh = getattr(args, "_pc_refresh", None)
        if refresh is not None and str(match_id) not in refresh.frames:
            return
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
        artifact = {k: descriptor[k] for k in ("feature_run_id", "spell_artifact_sha256", "carry_definition")}
        old_artifact = metadata.get("sportec_artifacts", {}).get(descriptor["match_id"])
        if old_artifact and old_artifact != artifact:
            raise ValueError(f"Changed Sportec spell artifact for {descriptor['match_id']}")
    if previous:
        # Locators can move, but state-defining provenance cannot change.
        for key in ("feature_run_id", "spell_artifact_sha256", "carry_definition", "possessor_id"):
            if key in previous and descriptor.get(key) != previous[key]:
                raise ValueError(f"Changed {dataset} provenance for {match_id}: {key}")
    if args.dry_run:
        return
    if getattr(args, "_pc_refresh", None) is not None:
        if str(match_id) not in args._pc_refresh.frames:
            return
        if args._pc_refresh.identity is None:
            args._pc_recovered_sources[str(match_id)] = descriptor
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


class HeightRefresh:
    def __init__(self, args, dataset, frames, identity):
        self.args, self.dataset, self.frames, self.identity = args, dataset, frames, identity
        if dataset in match_cache.DATASETS:
            match_cache.require_format(root(args) / dataset)
            self.frames = {str(key): match_cache.MatchAccumulator._internal(part, str(key))
                           for frame in frames.values()
                           for key, part in frame.groupby("possession_provenance_id")}
        self.verified = set()

    @staticmethod
    def key(row):
        import physical_pass_model as physics
        scope = row.get("frame_scope")
        scope = physics.PHYSICAL_XPASS_FRAME_SCOPE_ACTION if scope is None or pd.isna(scope) else str(scope)
        return int(row["action_index"]), scope

    def consume(self, items):
        import physical_pass_model as physics
        stats = physics._runtime_physical_xpass_stats(root(self.args) / self.dataset)
        target = fingerprint(self.identity) if self.identity is not None else None
        for item in items:
            match_id = str(item["match_id"])
            frame = self.frames.get(match_id)
            if frame is None:
                continue
            lookup = {self.key(row): row for row in frame.to_dict("records")}
            updates = []
            confirmed = set()
            for graph, label in zip(item["graphs"], item["labels"]):
                key = self.key({"action_index": int(label[physics.LABEL_INDEX["action_index"]]),
                                "frame_scope": item.get("frame_scope")})
                if key not in lookup:
                    continue
                row = lookup[key]
                if row.get("physical_state_hash") != physics.physical_state_hash(graph):
                    raise ValueError(f"Changed input state: {self.dataset}/{match_id}/{key}")
                if self.identity is None:
                    self.verified.add((match_id, key))
                    continue
                digest = input_digest(graph, label, row, omit_missing=self.dataset in match_cache.DATASETS)
                if (row.get(MODEL_COLUMN) != target or row.get(INPUT_COLUMN) != digest
                        or not physics._has_finite_pass_height_predictions(pd.Series(row), graph)):
                    prediction = physics._pass_height_predictions_for_graphs(
                        [graph], self.args._pass_height_model, device=str(self.args.pass_height_device),
                        labels=[label], pc_xpass_rows=[row], match_id=match_id)[0]
                    # Remove old probabilities for players no longer in the model output.
                    for column in list(row):
                        if column.endswith("__pass_height"):
                            row[column] = float("nan")
                    physics._apply_pass_height_predictions(row, prediction)
                    if not physics._has_finite_pass_height_predictions(pd.Series(row), graph):
                        raise ValueError(f"Incomplete height prediction: {self.dataset}/{match_id}/{key}")
                    row[MODEL_COLUMN], row[INPUT_COLUMN] = target, digest
                    updates.append(row)
                    stats["pass_height_refreshed"] += 1
                confirmed.add((match_id, key))
            if updates:
                physics._write_runtime_physical_xpass_rows(root(self.args) / self.dataset, match_id, pd.DataFrame(updates))
                transaction = match_cache.active(root(self.args) / self.dataset)
                self.frames[match_id] = (transaction.read(match_id) if transaction is not None else
                    pd.read_parquet(root(self.args) / self.dataset / "matches" / f"{match_id}.parquet"))
            self.verified.update(confirmed)
        return stats

    def verify(self):
        expected = {(match_id, self.key(row)) for match_id, frame in self.frames.items()
                    for row in frame.to_dict("records")}
        missing = expected - self.verified
        if missing:
            raise ValueError(f"Unresolved pass-height states in {self.dataset}: {len(missing)}; examples: {sorted(missing)[:5]}")


def replay_args(args, dataset, metadata):
    replay = copy.copy(args)
    replay.scope, replay.frames = "frames", 1
    replay.season = None
    replay.limit = None
    replay.skillcorner_limit = replay.benchmark_limit = replay.hawkeye_limit = None
    replay.time_norm = None
    replay._preflight_season_match_ids = None
    saved = dataset_metadata(root(args), dataset).get("source_inputs", {})
    descriptors = list(metadata.get("reconstruction", {}).get(dataset, {}).values())
    history = [entry.get("arguments", {}) for entry in metadata.get("invocations", [])]
    if dataset == "sportec":
        matches = {d["match_id"] for d in descriptors if "match_id" in d}
        matches.update(saved.get("selected_match_ids", []))
        for invocation in history:
            matches.update(invocation.get("selected_sportec_match_ids") or invocation.get("match_id") or [])
        replay.match_id = sorted(matches) or None
    elif dataset == "skillcorner":
        replay.skillcorner_match_id = sorted({d["match_id"] for d in descriptors if "match_id" in d}
                                            | set(saved.get("match_ids", []))
                                            | {str(value) for entry in history for value in (entry.get("skillcorner_match_id") or [])}) or None
    elif dataset == "benchmark":
        replay.benchmark_modification = sorted({d["modification_id"] for d in descriptors if "modification_id" in d}
                                              | set(saved.get("modifications", []))
                                              | {int(value) for entry in history for value in (entry.get("benchmark_modification") or [])}) or None
    else:
        replay.hawkeye_situation_id = sorted({d["situation_id"] for d in descriptors if "situation_id" in d}
                                            | {str(value) for value in saved.get("situation_ids", [])}
                                            | {str(value) for entry in history for value in (entry.get("hawkeye_situation_id") or [])}) or None
    locator_keys = {"skillcorner": ("skillcorner_input_dir",), "benchmark": ("benchmark_input_dir",),
                    "hawkeye": ("hawkeye_tracking_csv", "hawkeye_ball_csv")}.get(dataset, ())
    for key in locator_keys:
        if key in args._pc_explicit:
            continue
        candidates = [d[key] for d in descriptors if d.get(key)]
        candidates += [inv[key] for inv in history if inv.get(key)]
        if candidates:
            setattr(replay, key, candidates[-1])
    return replay


def refresh_all(args, runners):
    if not enabled(args) or not getattr(args, "_pass_height_model", None):
        return
    cache_root = root(args)
    frames = inventory(cache_root, lazy_possessions=True)
    for dataset in match_cache.DATASETS:
        if frames.get(dataset):
            match_cache.require_format(cache_root / dataset)
    identity = model_identity(args)
    args._pass_height_identity = identity
    target = fingerprint(identity)
    metadata = versions.read_metadata(cache_root) if (cache_root / "metadata.json").exists() else {}
    enrichment = metadata.get("pass_height_enrichment", {})
    if enrichment.get("pending") and fingerprint(enrichment["pending"]) != target:
        raise ValueError("Pending pass-height model artifact changed; restore it before resuming.")
    stale = any(frame.attrs.get("pending_match_commit") or (not frame.empty and (
                MODEL_COLUMN not in frame or INPUT_COLUMN not in frame
                or not frame[MODEL_COLUMN].eq(target).all() or frame[INPUT_COLUMN].isna().any()))
                for dataset in frames.values() for frame in dataset.values())
    if args.dry_run:
        print(f"Pass-height full refresh required: {bool(stale or enrichment.get('pending'))}")
        return
    if stale or enrichment.get("pending"):
        from models.pass_height import cached_height_definition, check_height_probability
        for dataset, matches in frames.items():
            old = dataset_metadata(cache_root, dataset)
            if matches and old.get("pass_height_model_id"):
                check_height_probability(cached_height_definition(old), (identity.get("definition") or {}).get("threshold_meters"))
        metadata.setdefault("pass_height_enrichment", {})["pending"] = identity
        versions.atomic_json(cache_root / "metadata.json", metadata)
        # Mark every dataset before refreshing any rows, so readers never see a mixture.
        for dataset, matches in frames.items():
            if matches:
                old = dataset_metadata(cache_root, dataset)
                old["pass_height_refresh_pending"] = identity
                versions.atomic_json(cache_root / dataset / "metadata.json", old)
        for dataset, matches in frames.items():
            if not matches:
                continue
            if dataset in match_cache.DATASETS:
                for match_id, frame in matches.items():
                    replay = replay_args(args, dataset, metadata)
                    setattr(replay, "match_id" if dataset == "sportec" else "skillcorner_match_id", [match_id])
                    pending = bool(frame.attrs.get("pending_match_commit"))
                    refresh = None if pending else HeightRefresh(replay, dataset, {match_id: frame}, identity)
                    if refresh is not None:
                        replay._pc_refresh = refresh
                    print(f"Refreshing saved pass-height states for {dataset}/{match_id}...")
                    result = runners[dataset](replay)
                    if _has_failures((result or {}).get("skipped")):
                        raise ValueError(f"Pass-height refresh failed for {dataset}/{match_id}; resume generation")
                    if refresh is not None:
                        refresh.verify()
                continue
            replay = replay_args(args, dataset, metadata)
            refresh = HeightRefresh(replay, dataset, matches, identity)
            replay._pc_refresh = refresh
            print(f"Refreshing all saved pass-height states for {dataset}...")
            result = runners[dataset](replay)
            if _has_failures((result or {}).get("skipped")):
                raise ValueError(f"Pass-height refresh failed for {dataset}; resume generation")
            refresh.verify()
        # Root pending remains the barrier until every dataset is committed.
        for dataset, matches in frames.items():
            if matches:
                old = dataset_metadata(cache_root, dataset)
                old.pop("pass_height_refresh_pending", None)
                old.update(pass_height_model_id=identity["model_id"], pass_height_definition=identity["definition"],
                           pass_height_model_record=args._pass_height_model_record,
                           runtime_graph_schema=getattr(args, "_runtime_graph_schema", old.get("runtime_graph_schema")))
                versions.atomic_json(cache_root / dataset / "metadata.json", old)
        args._pc_height_refresh_completed = True
    metadata = versions.read_metadata(cache_root)
    metadata["pass_height_enrichment"] = {**identity, "status": "completed"}
    versions.atomic_json(cache_root / "metadata.json", metadata)


def _has_failures(value):
    if isinstance(value, dict):
        return any(_has_failures(item) for item in value.values())
    return bool(value)


def recover_sources(args, runners):
    """Audit legacy states before allowing a guessed input source to add partitions."""
    if not enabled(args):
        return
    metadata = versions.read_metadata(root(args)) if (root(args) / "metadata.json").exists() else {}
    for dataset in match_cache.DATASETS:
        if (args.pass_height_model_id or not getattr(args, "no_" + dataset, False)) and MatchInventory(root(args) / dataset):
            match_cache.require_format(root(args) / dataset)
    for dataset, matches in inventory(root(args), datasets=[d for d in DATASETS if d not in match_cache.DATASETS]).items():
        if not args.pass_height_model_id and getattr(args, "no_" + dataset, False):
            continue
        known = metadata.get("reconstruction", {}).get(dataset, {})
        missing = {key: frame for key, frame in matches.items() if key not in known}
        if not missing:
            continue
        replay = replay_args(args, dataset, metadata)
        audit = HeightRefresh(replay, dataset, missing, None)
        replay._pc_refresh = audit
        replay._pc_recovered_sources = {}
        print(f"Recovering saved state provenance for {dataset}...")
        runners[dataset](replay)
        audit.verify()
        for match_id, descriptor in replay._pc_recovered_sources.items():
            record_source(args, dataset, match_id, descriptor)
