"""One-match IPW inference and atomic, validated probability sidecars."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import ActionDataset
from datatools import config
from ipw_options import DEFAULT_IPW_CACHE_DIR
from models.utils import adapt_batch_graphs_for_model, get_model_path, load_model
from prepared_dataset import (_digest, _file_digest, _signature, PreparedActionDataset,
                              loaded_preprocessing_fingerprint, preprocessing_fingerprint)
from training_state import capture_rng, restore_rng

CACHE_VERSION = 1
_ROOT = Path(__file__).resolve().parent


def inference_fingerprint():
    # Track only the reachable inference helpers in the mixed training utility
    # module. Edits to run_epoch, metrics and optimizer code must not invalidate.
    tree = ast.parse((_ROOT / "models/utils.py").read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    pending = ["load_model", "adapt_batch_graphs_for_model"]
    selected = {}
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        node = definitions[name]
        selected[name] = ast.dump(node, include_attributes=False)
        pending.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in definitions)
    return _digest({"helpers": selected, "files": {
        name: _file_digest(_ROOT / name) for name in ("ipw_preparation.py", "models/gnn.py", "models/goal_context.py")}})


LOADED_INFERENCE_FINGERPRINT = inference_fingerprint()


def sample_records(dataset, indices=None):
    """Identity plus labels whose correspondence is essential for weighting."""
    occurrences = Counter()
    records = []
    if len(dataset.features) != len(dataset.labels):
        raise ValueError("IPW dataset graph/label counts differ.")
    for index in range(len(dataset)) if indices is None else indices:
        graph, label = dataset.features[index], dataset.labels[index]
        identity = (str(graph.evaluation_match_id), int(graph.evaluation_source_row))
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        target = float(label[config.LABEL_INDEX["intent_index"]])
        if target.is_integer() and 0 <= target < graph.num_nodes and hasattr(graph, "source_node_positions"):
            target = float(graph.source_node_positions[int(target)])
        records.append([*identity, occurrence, target,
                        float(label[config.LABEL_INDEX["is_dribble"]])])
    return records


def predict_probabilities(dataset, model, *, device, batch_size, pin_memory):
    """Raw probabilities with the historical IPW teammate/target semantics."""
    from models.goal_context import goal_policy, candidate_mask, target_candidate_position
    probabilities = torch.empty(len(dataset), dtype=torch.float32)
    offset = 0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=pin_memory, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        for graphs, labels, _ in loader:
            graphs = adapt_batch_graphs_for_model(graphs.to(device), model.args, context="IPW model")
            out = model(graphs)
            for i in range(graphs.num_graphs):
                if bool(labels[i, config.LABEL_INDEX["is_dribble"]]):
                    probability = 1.0
                else:
                    indices = torch.where((graphs.batch == i) & candidate_mask(graphs, model.args))[0]
                    logits = out[indices]
                    target = int(labels[i, config.LABEL_INDEX["intent_index"]])
                    if goal_policy(model.args).version == 2:
                        target = target_candidate_position(indices, int(graphs.ptr[i]) + target)
                    if target < 0 or target >= len(logits):
                        raise ValueError("IPW target is outside the teammate candidates.")
                    probability = torch.softmax(logits, dim=0)[target].item()
                probabilities[offset] = probability
                offset += 1
            del graphs, labels, out
    return probabilities


def normalized_ipw(probabilities, labels):
    inverse = probabilities.clamp_min(0.01).reciprocal()
    carries = labels[:, config.LABEL_INDEX["is_dribble"]] == 1
    normalizer = inverse[~carries].mean() if bool((~carries).any()) else 1.0
    weights = inverse / normalizer
    weights[carries] = 1.0
    return weights


class IPWPreparer:
    def __init__(self, *, model_id, model_args, feature_dir, label_dir, options,
                 device="cpu", batch_size=256, pin_memory=False, cache="on",
                 cache_dir=DEFAULT_IPW_CACHE_DIR, monitor=None):
        if batch_size < 1:
            raise ValueError("IPW batch size must be positive.")
        self.model_id, self.device = model_id, device
        self.batch_size, self.pin_memory = batch_size, pin_memory
        self.enabled, self.monitor, self.model = cache == "on", monitor, None
        self.feature_dir, self.label_dir = Path(feature_dir).resolve(), Path(label_dir).resolve()
        bound = inspect.signature(ActionDataset.__init__).bind(
            None, [], feature_dir=str(self.feature_dir), label_dir=str(self.label_dir), **options)
        bound.apply_defaults()
        self.options = {k: v for k, v in bound.arguments.items()
                        if k not in {"self", "match_ids", "feature_dir", "label_dir"}}
        for key, value in self.options.items():
            if key.endswith("_dir") and value is not None:
                self.options[key] = str(Path(value).resolve())
        if self.options.get("lane_survival") and not self.options.get("lane_survival_cache_dir"):
            from project_config import get_pc_xpass_dir
            self.options["lane_survival_cache_dir"] = str(get_pc_xpass_dir("sportec").resolve())
        model_root = get_model_path(model_id)
        self.model_files = [model_root / "args.json", model_root / "best_weights.pt", model_root / "metadata.json"]
        self.model_signatures = [_signature(p) for p in self.model_files]
        self.model_digests = [_file_digest(p) if p.exists() else None for p in self.model_files]
        if any(digest is None for digest in self.model_digests[:2]):
            raise FileNotFoundError(f"IPW model {model_id!r} requires args.json and best_weights.pt.")
        self.identity = {"version": CACHE_VERSION, "model": self.model_digests,
                         "model_args": model_args, "options": self.options,
                         "feature_dir": str(self.feature_dir), "label_dir": str(self.label_dir),
                         "preparation": loaded_preprocessing_fingerprint(self.options),
                         "inference": LOADED_INFERENCE_FINGERPRINT}
        self.cache_id = _digest(self.identity)
        self.cache_root = Path(cache_dir).resolve() / self.cache_id
        self._check_stable(code=True)

    # Reuse the graph cache's complete source/sidecar signature policy without
    # constructing a PreparedActionDataset or writing any graph cache.
    _source_signatures = PreparedActionDataset._source_signatures

    def _check_stable(self, *, code=False):
        if [_signature(p) for p in self.model_files] != self.model_signatures:
            raise RuntimeError("IPW checkpoint changed during preparation; restart with stable inputs.")
        if code and (preprocessing_fingerprint(self.options) != self.identity["preparation"]
                     or inference_fingerprint() != self.identity["inference"]):
            raise RuntimeError("IPW preparation/inference code changed since import; restart with stable code.")

    @staticmethod
    def _valid_probabilities(values, count):
        return (isinstance(values, list) and len(values) == count
                and all(isinstance(p, (float, int)) and 0 <= p <= 1 for p in values))

    def _read(self, path, match_id, sources, expected):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if (envelope["sha256"] != _digest(payload) or payload["cache_id"] != self.cache_id
                    or payload["match_id"] != match_id or payload["sources"] != sources
                    or payload["records"] != expected
                    or not self._valid_probabilities(payload["probabilities"], len(expected))):
                return None
            return payload["probabilities"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"sha256": _digest(payload), "payload": payload}, stream,
                          separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def prepare(self, dataset, match_ids, *, split="training"):
        if str(self.device).startswith("cuda"):
            torch.cuda.init()  # Capture CUDA RNG even when this is its first use.
        rng = capture_rng()
        try:
            return self._prepare(dataset, match_ids, split)
        finally:
            restore_rng(rng)

    def _prepare(self, dataset, match_ids, split):
        started = time.perf_counter()
        ids = [str(m) for m in match_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate match IDs in IPW split.")
        groups = defaultdict(list)
        for index, graph in enumerate(dataset.features):
            groups[str(graph.evaluation_match_id)].append(index)
        if set(groups) - set(ids):
            raise ValueError("Main dataset contains samples outside the IPW split.")
        result = torch.empty(len(dataset), dtype=torch.float32)
        stats = {"hits": 0, "misses": 0, "samples": len(dataset), "cache_id": self.cache_id,
                 "cache": "on" if self.enabled else "off"}
        if self.monitor is not None:
            self.monitor.event("ipw_preparation_" + split, self.device)
        for match_id in tqdm(ids, desc=f"IPW {split}"):
            expected = sample_records(dataset, groups[match_id])
            sources = self._source_signatures(match_id)
            path = self.cache_root / (_digest(match_id) + ".json")
            values = self._read(path, match_id, sources, expected) if self.enabled else None
            if values is None:
                stats["misses"] += 1
                with contextlib.redirect_stderr(io.StringIO()):
                    match_dataset = ActionDataset([match_id], feature_dir=self.feature_dir,
                                                  label_dir=self.label_dir, **self.options)
                try:
                    actual = sample_records(match_dataset)
                    # Reorder by stable identity; never align by incidental position.
                    actual_map = {tuple(record[:3]): (i, record) for i, record in enumerate(actual)}
                    if (len(actual) != len(expected) or any(
                            tuple(r[:3]) not in actual_map or actual_map[tuple(r[:3])][1] != r
                            for r in expected)):
                        raise ValueError(f"IPW sample/target alignment mismatch for match {match_id}.")
                    if actual:
                        if self.model is None:
                            self.model = load_model(self.model_id, self.device)
                        probabilities = predict_probabilities(match_dataset, self.model, device=self.device,
                                                              batch_size=self.batch_size, pin_memory=self.pin_memory)
                        values = [float(probabilities[actual_map[tuple(r[:3])][0]]) for r in expected]
                        del probabilities
                    else:
                        values = []
                finally:
                    del match_dataset
                if not self._valid_probabilities(values, len(expected)):
                    raise ValueError(f"Invalid IPW probabilities for match {match_id}.")
                self._check_stable(code=True)
                if self._source_signatures(match_id) != sources:
                    raise RuntimeError(f"IPW sources changed during preparation of {match_id}.")
                # Missing matches are retried on the next run, even when both views are empty.
                if self.enabled and all(s[1] is not None for s in sources[:2]):
                    self._write(path, {"cache_id": self.cache_id, "match_id": match_id,
                                      "sources": sources, "records": expected, "probabilities": values})
            else:
                stats["hits"] += 1
            self._check_stable()
            if self._source_signatures(match_id) != sources:
                raise RuntimeError(f"IPW sources changed during preparation of {match_id}.")
            for index, value in zip(groups[match_id], values):
                result[index] = value
        self._check_stable(code=True)
        stats["seconds"] = time.perf_counter() - started
        print(f"IPW {split}: {stats['hits']} cache hits, {stats['misses']} misses; "
              f"{len(dataset):,} samples in {stats['seconds']:.1f}s.", flush=True)
        if self.monitor is not None:
            self.monitor.event("ipw_complete_" + split, self.device)
        return normalized_ipw(result, dataset.labels), stats

    def close(self):
        self.model = None
