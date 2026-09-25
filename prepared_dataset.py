"""Prepared per-match graph cache with bounded, reproducible epoch iteration."""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import torch
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm

from dataset import ActionDataset
from dataset_loading import DEFAULT_CACHE_DIR, INTENT_TASKS

CACHE_VERSION = 1
DISK_RESERVE_BYTES = 10 * 1024**3
BASE_PREPARATION_FILES = (
    "prepared_dataset.py",
    "dataset.py",
    "models/dataset_config.py",
    "models/edge_feature_config.py",
    "datatools/config.py",
    "datatools/utils.py",
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
                 buffer_matches=4, shuffle=False, seed=100, progress=True, **options):
        super().__init__()
        if options.get("task") not in INTENT_TASKS:
            raise ValueError("Prepared datasets support pass_intent and action_intent only.")
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
        self.cache_root = Path(cache_dir).resolve() / self.cache_id[:32]
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.loaded_match_ids, self.contributing_match_ids = [], []
        self.skipped_matches, self.skipped_rows = {}, {}
        self.entries = []
        self.preparation = {"built": 0, "reused": 0, "bytes": 0}
        self.last_iteration_seconds = 0.0
        self.last_load_seconds = 0.0
        started = time.perf_counter()
        pending = []
        for match_id in self.requested_match_ids:
            signatures = self._source_signatures(match_id)
            key = _digest([match_id, signatures])
            path = self.cache_root / f"{key[:32]}.pt"
            entry = self._read_entry(path, match_id, signatures)
            pending.append((match_id, signatures, path, entry))
        missing_bytes = sum(s[0][1] or 0 for _, s, _, entry in pending if entry is None)
        if progress:
            print(f"Prepared cache: {sum(e is not None for _, _, _, e in pending)} reusable, "
                  f"{sum(e is None for _, _, _, e in pending)} to build; "
                  f"estimated additional disk {missing_bytes * 1.15 / 1024**3:.2f} GiB.", flush=True)
        self._check_space(math.ceil(missing_bytes * 1.15))
        for match_id, signatures, path, entry in tqdm(pending, desc="Preparing match cache", disable=not progress):
            if entry is None:
                entry = self._build_entry(match_id, signatures, path)
                self.preparation["built"] += int(entry is not None)
            else:
                self.preparation["reused"] += 1
            if entry is None:
                continue
            self.loaded_match_ids.append(match_id)
            for reason, count in entry["skipped_rows"].items():
                self.skipped_rows[reason] = self.skipped_rows.get(reason, 0) + count
            self.preparation["bytes"] += entry["bytes"]
            if entry["count"]:
                self.contributing_match_ids.append(match_id)
                self.entries.append(entry)
        self.sample_count = sum(e["count"] for e in self.entries)
        self.preparation["seconds"] = time.perf_counter() - started
        if progress:
            print(f"Prepared {self.sample_count:,} samples; cache {self.preparation['bytes'] / 1024**3:.2f} GiB; "
                  f"preparation {self.preparation['seconds']:.1f}s.", flush=True)

    def _source_signatures(self, match_id):
        paths = [self.feature_dir / f"{match_id}.pt", self.label_dir / f"{match_id}.pt"]
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
                    or not isinstance(entry["skipped_rows"], dict)
                    or path.stat().st_size != entry["bytes"]
                    or _file_digest(path) != entry["sha256"]):
                return None
            entry["path"] = str(path)
            entry["cache_signature"] = _signature(path)
            entry["manifest_signature"] = _signature(path.with_suffix(".json"))
            return entry
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _build_entry(self, match_id, signatures, path):
        # The existing eager path is bounded here to one match, including its clones.
        with contextlib.redirect_stderr(io.StringIO()):
            dataset = ActionDataset([match_id], feature_dir=self.feature_dir,
                                    label_dir=self.label_dir, **self.options)
        self.skipped_matches.update(dataset.skipped_matches)
        if match_id not in dataset.loaded_match_ids:
            return None  # Do not persist unreadable/missing matches; retry next run.
        if self._source_signatures(match_id) != signatures:
            raise RuntimeError(f"Source changed while preparing {match_id}; retry with stable artifacts.")
        for graph in dataset.features:
            graph.cpu()
        payload = {"graphs": dataset.features, "labels": dataset.labels.cpu()}
        temporary = path.with_suffix(f".{uuid.uuid4().hex[:12]}.tmp")
        manifest_tmp = temporary.with_suffix(".json.tmp")
        try:
            self._check_space(math.ceil((signatures[0][1] or 0) * 1.15))
            # Use Python's file API, including for non-ASCII cache directories.
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
            self._check_space()
            entry = {"cache_id": self.cache_id, "match_id": match_id, "sources": signatures,
                     "count": len(dataset), "skipped_rows": dict(dataset.skipped_rows),
                     "bytes": temporary.stat().st_size, "sha256": _file_digest(temporary)}
            manifest_tmp.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            os.replace(temporary, path)
            os.replace(manifest_tmp, path.with_suffix(".json"))
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
        payload = torch.load(path, weights_only=False, map_location="cpu")
        self.last_load_seconds += time.perf_counter() - started
        if len(payload["graphs"]) != entry["count"] or len(payload["labels"]) != entry["count"]:
            raise RuntimeError(f"Prepared sample count changed for {entry['match_id']}")
        return payload

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("PreparedActionDataset requires num_workers=0")
        rng = random.Random(f"{self.seed}:{self.epoch}")
        entries = list(self.entries)
        if self.shuffle:
            rng.shuffle(entries)
        self.last_load_seconds = 0.0
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
                    yield group[m]["graphs"][r], group[m]["labels"][r], torch.tensor(1.0)
                group.clear()
        finally:
            group.clear()
            self.last_iteration_seconds = time.perf_counter() - started

    def metadata(self):
        return {"mode": "disk", "cache_id": self.cache_id, "cache_root": str(self.cache_root),
                "buffer_matches": self.buffer_matches, "shuffle": "match_groups" if self.shuffle else "sequential",
                "sample_count": len(self), "preparation": dict(self.preparation),
                "last_load_seconds": self.last_load_seconds}
