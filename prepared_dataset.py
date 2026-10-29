"""Prepared per-match graph cache with bounded, reproducible epoch iteration."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import math
import os
import pickle
from pathlib import Path
import random
import shutil
import time
import uuid

import torch
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm

from dataset import ActionDataset
from packed_graphs import FORMAT, VERSION, pack_graphs, decode_payload, validate_payload
from project_config import assert_feature_paths_usable
from dataset_loading import DEFAULT_CACHE_DIR, PREPARED_TASKS
from models.pass_height import positive_height, relabel_height
from cache_lifetime import CacheWorkflow
from training_state import capture_rng, restore_rng

CACHE_VERSION = VERSION
DISK_RESERVE_BYTES = 10 * 1024**3
BASE_PREPARATION_FILES = (
    "prepared_dataset.py",
    "packed_graphs.py",
    "dataset.py",
    "models/dataset_config.py",
    "models/node_feature_config.py",
    "models/feature_masks.py",
    "models/input_scaling.py",
    "models/goal_context.py",
    "models/edge_feature_config.py",
    "models/edge_schema.py",
    "models/self_loops.py",
    "models/pass_height.py",
    "datatools/config.py",
    "datatools/graph_validation.py",
    "datatools/utils.py",
    "datatools/trajectory.py",
    "datatools/pass_trajectory.py",
    "datatools/pass_eligibility.py",
    "datatools/endpoint_policy.py",
    "datatools/failed_pass_trajectory.py",
)
PHYSICAL_PREPARATION_FILES = (
    "physical_pass_model.py",
    "reachability.py",
)
PHYSICAL_PREPARATION_OPTIONS = (
    "use_physical_xpass",
    "require_observed_pass_height",
    "pass_height_cache_dir",
    "evaluation_xpass_cache_dir",
    "lane_survival",
    "lane_survival_cache_dir",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _signature(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]
    except FileNotFoundError:
        return [str(path), None, None]


def _uses_physical_preparation(options=None):
    options = options or {}
    return any(bool(options.get(name)) for name in PHYSICAL_PREPARATION_OPTIONS)


def preparation_code_paths(options=None):
    root = Path(__file__).resolve().parent
    names = list(BASE_PREPARATION_FILES)
    if _uses_physical_preparation(options):
        names.extend(PHYSICAL_PREPARATION_FILES)
    return tuple(root / name for name in names)


def _fingerprint_from_digests(paths, digests):
    root = Path(__file__).resolve().parent
    return _digest([(str(path.relative_to(root)), digests[str(path.relative_to(root))]) for path in paths])


def preprocessing_fingerprint(options=None):
    paths = preparation_code_paths(options)
    root = Path(__file__).resolve().parent
    digests = {str(path.relative_to(root)): _file_digest(path) for path in paths}
    return _fingerprint_from_digests(paths, digests)


# Freeze every possible preparation dependency when imported. A dataset selects
# the base subset or the physical-xPass subset from these frozen digests.
_ROOT = Path(__file__).resolve().parent
_ALL_PREPARATION_PATHS = preparation_code_paths(dict.fromkeys(PHYSICAL_PREPARATION_OPTIONS, True))
LOADED_PREPARATION_FILE_DIGESTS = {
    str(path.relative_to(_ROOT)): _file_digest(path) for path in _ALL_PREPARATION_PATHS
}


def loaded_preprocessing_fingerprint(options=None):
    paths = preparation_code_paths(options)
    return _fingerprint_from_digests(paths, LOADED_PREPARATION_FILE_DIGESTS)


# Compatibility export for provenance that records the standard preparation
# code. Dataset cache identities use the option-sensitive value below.
LOADED_PREPROCESSING_FINGERPRINT = loaded_preprocessing_fingerprint()


class PreparedActionDataset(IterableDataset):
    """Only metadata survives preparation; at most one group is loaded per iterator."""

    def __init__(self, match_ids, *, feature_dir, label_dir, cache_dir=DEFAULT_CACHE_DIR,
                 buffer_matches=4, shuffle=False, seed=100, progress=True, cache_session=None, **options):
        super().__init__()
        assert_feature_paths_usable(feature_dir, label_dir, options.get("diagnostic_label_dir"),
                                    options.get("pass_height_diagnostic_label_dir"))
        if options.get("task") not in PREPARED_TASKS:
            raise ValueError(f"Prepared datasets do not support task {options.get('task')!r}.")
        if int(buffer_matches) < 1:
            raise ValueError("buffer_matches must be positive")
        self.buffer_matches = int(buffer_matches) if shuffle else 1
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self.requested_match_ids = [str(m) for m in match_ids]
        if len(set(self.requested_match_ids)) != len(self.requested_match_ids):
            raise ValueError("Duplicate match IDs in dataset split")
        self.feature_dir, self.label_dir = Path(feature_dir).resolve(), Path(label_dir).resolve()
        bound = inspect.signature(ActionDataset.__init__).bind(
            None, [], feature_dir=str(self.feature_dir), label_dir=str(self.label_dir), **options)
        bound.apply_defaults()
        self.options = {k: v for k, v in bound.arguments.items()
                        if k not in {"self", "match_ids", "feature_dir", "label_dir"}}
        from datatools.graph_validation import graph_validation_enabled, validation_stats
        self.options["graph_validation"] = graph_validation_enabled(self.options["graph_validation"])
        if self.options.get("task") == "success_intent":
            from datatools.trajectory import directory_speed_definition, require_speed_definition
            require_speed_definition(directory_speed_definition(self.feature_dir),
                                     self.options.get("trajectory_speed_definition"), context="Prepared success_intent inputs")
        self.pass_height_threshold = self.options.get("pass_height_threshold")
        self.options["pass_height_threshold"] = None
        if self.pass_height_threshold is not None:
            self.pass_height_threshold = positive_height(self.pass_height_threshold)
            self.options["defer_pass_height_relabel"] = True
        # Resolve path options before recording identity or reading their signatures.
        for key, value in self.options.items():
            if key.endswith("_dir") and value is not None:
                self.options[key] = str(Path(value).resolve())
        if self.options.get("lane_survival") and not self.options.get("lane_survival_cache_dir"):
            from project_config import get_pc_xpass_dir
            self.options["lane_survival_cache_dir"] = str(get_pc_xpass_dir("sportec").resolve())
        loaded_code = loaded_preprocessing_fingerprint(self.options)
        if preprocessing_fingerprint(self.options) != loaded_code:
            raise RuntimeError("Preprocessing code changed since import; restart cache preparation with stable code.")
        self.identity = {"version": CACHE_VERSION, "code": loaded_code,
                         "feature_dir": str(self.feature_dir), "label_dir": str(self.label_dir),
                         "options": self.options}
        self.cache_id = _digest(self.identity)
        # Library callers retain their caches; CLI training supplies workflow ownership.
        self.cache_session = cache_session or CacheWorkflow(cache_dir, keep=True, kind="library").session_for(cache_dir)
        self.cache_root = Path(cache_dir).resolve() / self.cache_id[:32]
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.loaded_match_ids, self.contributing_match_ids = [], []
        self.skipped_matches, self.skipped_rows = {}, {}
        self.entries = []
        self.preparation = {"built": 0, "reused": 0, "bytes": 0, "packing_seconds": 0.0,
                            "transform_seconds": 0.0, "save_and_hash_seconds": 0.0,
                            "graph_validation": validation_stats(self.options["graph_validation"])}
        self.last_iteration_seconds = 0.0
        self.last_load_seconds = 0.0
        self._reconstruction_timing = {"seconds": 0.0}
        started = time.perf_counter()
        pending = []
        for match_id in self.requested_match_ids:
            signatures = self._source_signatures(match_id)
            key = _digest([match_id, signatures])
            path = self.cache_root / f"{key[:32]}.pt"
            pending.append((match_id, signatures, path, path.is_file()))
        missing_bytes = sum(s[0][1] or 0 for _, s, _, exists in pending if not exists)
        if progress:
            print(f"Prepared cache: {sum(exists for _, _, _, exists in pending)} reusable, "
                  f"{sum(not exists for _, _, _, exists in pending)} to build; "
                  f"estimated additional disk {missing_bytes * 1.15 / 1024**3:.2f} GiB.", flush=True)
        self._check_space(math.ceil(missing_bytes * 1.15))
        for match_id, signatures, path, _ in tqdm(pending, desc="Preparing match cache", disable=not progress):
            with self.cache_session.entry(path):
                entry = self._read_entry(path, match_id, signatures)
                built = entry is None
                if built:
                    rng = capture_rng()
                    try:
                        entry = self._build_entry(match_id, signatures, path)
                    finally:
                        restore_rng(rng)
                    self.preparation["built"] += int(entry is not None)
                else:
                    self.preparation["reused"] += 1
                if entry is not None:
                    self.cache_session.recorded(path, built=built, size=entry["bytes"])
            if entry is None:
                continue
            self.loaded_match_ids.append(match_id)
            for reason, count in entry["skipped_rows"].items():
                self.skipped_rows[reason] = self.skipped_rows.get(reason, 0) + count
            self.preparation["bytes"] += entry["bytes"]
            if entry["count"]:
                self.contributing_match_ids.append(match_id)
                self.entries.append(entry)
        self.sample_count = 0
        for entry in self.entries:
            entry["offset"] = self.sample_count
            self.sample_count += entry["count"]
        self.ip_weights = None
        self.preparation["seconds"] = time.perf_counter() - started
        if progress:
            print(f"Prepared {self.sample_count:,} samples; cache {self.preparation['bytes'] / 1024**3:.2f} GiB; "
                  f"preparation {self.preparation['seconds']:.1f}s.", flush=True)

    def _source_signatures(self, match_id):
        paths = [self.feature_dir / f"{match_id}.pt", self.label_dir / f"{match_id}.pt"]
        paths.append(self.feature_dir.parent / "failed_pass_trajectories" / f"{match_id}.parquet")
        metadata_path = self.feature_dir.parent / "metadata.json"
        if metadata_path.is_file() or self.options.get("task") == "success_intent":
            paths.append(metadata_path)
        for key, directory in self.options.items():
            if not key.endswith("_dir") or directory is None:
                continue
            root = Path(directory)
            if "label" in key:
                paths.append(root / f"{match_id}.pt")
            else:
                paths.extend([root / "metadata.json", root / "matches" / f"{match_id}.parquet",
                              root / f"{match_id}.parquet"])
        return [_signature(p) for p in paths]

    def _check_space(self, additional=0):
        free = shutil.disk_usage(self.cache_root).free
        if free < DISK_RESERVE_BYTES + additional:
            raise OSError(f"Insufficient cache disk space: {free / 1024**3:.2f} GiB free; "
                          f"need {additional / 1024**3:.2f} GiB plus a 10 GiB reserve. "
                          "Completed cache entries are retained. Choose --dataset-cache-dir on a larger drive "
                          "or remove unused caches manually.")

    def _read_entry(self, path, match_id, signatures):
        try:
            entry = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (entry["cache_id"] != self.cache_id or entry["match_id"] != match_id
                    or entry["sources"] != signatures or entry["count"] < 0
                    or entry.get("format") != FORMAT or entry.get("version") != CACHE_VERSION
                    or not isinstance(entry["skipped_rows"], dict)
                    or path.stat().st_size != entry["bytes"]
                    or _file_digest(path) != entry["sha256"]):
                return None
            # A checksum alone cannot validate a well-formed but incompatible payload.
            payload = torch.load(path, weights_only=False, map_location="cpu", mmap=False)
            validate_payload(payload, entry["count"])
            del payload
            entry["path"] = str(path)
            entry["cache_signature"] = _signature(path)
            entry["manifest_signature"] = _signature(path.with_suffix(".json"))
            return entry
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
            return None

    def _build_entry(self, match_id, signatures, path):
        # The existing eager path is bounded here to one match, including its clones.
        transform_started = time.perf_counter()
        with contextlib.redirect_stderr(io.StringIO()):
            dataset = ActionDataset([match_id], feature_dir=self.feature_dir,
                                    label_dir=self.label_dir, **self.options)
        self.preparation["transform_seconds"] += time.perf_counter() - transform_started
        from datatools.graph_validation import merge_validation_stats
        self.preparation["graph_validation"] = merge_validation_stats(
            [self.preparation.get("graph_validation"), dataset.graph_validation_stats],
            self.options.get("graph_validation", True))
        self.skipped_matches.update(dataset.skipped_matches)
        if match_id not in dataset.loaded_match_ids:
            return None  # Do not persist unreadable/missing matches; retry next run.
        if self._source_signatures(match_id) != signatures:
            raise RuntimeError(f"Source changed while preparing {match_id}; retry with stable artifacts.")
        for graph in dataset.features:
            graph.cpu()
        if dataset.features:
            del graph
        packing_started = time.perf_counter()
        payload = pack_graphs(dataset.features, dataset.labels)
        self.preparation["packing_seconds"] += time.perf_counter() - packing_started
        count, skipped_rows = len(dataset), dict(dataset.skipped_rows)
        del dataset

        temporary = path.with_suffix(f".{uuid.uuid4().hex[:12]}.tmp")
        manifest_tmp = temporary.with_suffix(".json.tmp")
        try:
            save_started = time.perf_counter()
            self._check_space(math.ceil((signatures[0][1] or 0) * 1.15))
            # Use Python's file API, including for non-ASCII cache directories.
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
            self._check_space()
            entry = {"cache_id": self.cache_id, "match_id": match_id, "sources": signatures,
                     "format": FORMAT, "version": CACHE_VERSION,
                     "count": count, "skipped_rows": skipped_rows,
                     "bytes": temporary.stat().st_size, "sha256": _file_digest(temporary)}
            manifest_tmp.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            os.replace(manifest_tmp, path.with_suffix(".json"))
            self.preparation["save_and_hash_seconds"] += time.perf_counter() - save_started
        finally:
            temporary.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
        entry["path"] = str(path)
        entry["cache_signature"] = _signature(path)
        entry["manifest_signature"] = _signature(path.with_suffix(".json"))
        return entry

    def __len__(self):
        return self.sample_count

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _load_entry(self, entry):
        path = Path(entry["path"])
        if (self._source_signatures(entry["match_id"]) != entry["sources"]
                or _signature(path) != entry["cache_signature"]
                or _signature(path.with_suffix(".json")) != entry["manifest_signature"]):
            raise RuntimeError(f"Source or prepared cache changed for {entry['match_id']}; restart preparation.")
        started = time.perf_counter()
        try:
            packed = torch.load(path, weights_only=False, map_location="cpu", mmap=False)
            self.last_load_seconds += time.perf_counter() - started
            payload = decode_payload(packed, expected_count=entry["count"], timing=self._reconstruction_timing)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            raise RuntimeError(f"Invalid prepared cache for {entry['match_id']}: {exc}") from exc
        if self.pass_height_threshold is not None:
            payload["labels"] = relabel_height(payload["labels"], self.pass_height_threshold)
        if len(payload["graphs"]) != entry["count"] or len(payload["labels"]) != entry["count"]:
            raise RuntimeError(f"Prepared sample count changed for {entry['match_id']}")
        return payload

    def iter_matches(self):
        """Source-order one-match views for IPW and other preparation passes."""
        for entry in self.entries:
            payload = self._load_entry(entry)
            try:
                yield entry["match_id"], entry["offset"], payload
            finally:
                del payload

    def set_inverse_propensity_weights(self, weights):
        if weights.ndim != 1 or len(weights) != len(self):
            raise ValueError("IPW weight count differs from the prepared dataset.")
        self.ip_weights = weights.detach().cpu()

    def __iter__(self):
        assert_feature_paths_usable(self.feature_dir, self.label_dir, self.options.get("diagnostic_label_dir"),
                                    self.options.get("pass_height_diagnostic_label_dir"))
        if get_worker_info() is not None:
            raise RuntimeError("PreparedActionDataset requires num_workers=0")
        rng = random.Random(f"{self.seed}:{self.epoch}")
        entries = list(self.entries)
        if self.shuffle:
            rng.shuffle(entries)
        self.last_load_seconds = 0.0
        self._reconstruction_timing["seconds"] = 0.0
        started = time.perf_counter()
        group = []
        try:
            for offset in range(0, len(entries), self.buffer_matches):
                for entry in entries[offset:offset + self.buffer_matches]:
                    group.append(self._load_entry(entry))
                rows = [(m, r) for m, payload in enumerate(group) for r in range(len(payload["graphs"]))]
                if self.shuffle:
                    rng.shuffle(rows)
                for m, r in rows:
                    weight = (torch.tensor(1.0) if self.ip_weights is None else
                              self.ip_weights[entries[offset + m]["offset"] + r])
                    yield group[m]["graphs"][r], group[m]["labels"][r], weight
                group.clear()
        finally:
            group.clear()
            self.last_iteration_seconds = time.perf_counter() - started

    @property
    def last_reconstruct_seconds(self):
        return self._reconstruction_timing["seconds"]

    def metadata(self):
        return {"mode": "disk", "format": FORMAT, "version": CACHE_VERSION, "cache_id": self.cache_id, "cache_root": str(self.cache_root),
                "pass_height_threshold_meters": self.pass_height_threshold,
                "buffer_matches": self.buffer_matches, "shuffle": "match_groups" if self.shuffle else "sequential",
                "sample_count": len(self), "preparation": dict(self.preparation),
                "last_load_seconds": self.last_load_seconds,
                "last_reconstruct_seconds": self.last_reconstruct_seconds,
                "last_load_and_reconstruct_seconds": self.last_load_seconds + self.last_reconstruct_seconds}
