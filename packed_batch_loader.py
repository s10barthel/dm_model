"""Runtime batching of existing prepared caches without per-sample Data objects.

Keep this reader outside the preprocessing fingerprint: it changes neither the
cache format nor the samples. Unsupported PyG schemas use the ordinary collater.
"""
from __future__ import annotations

import copy
from itertools import accumulate
import random
import time

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater

from project_config import assert_feature_paths_usable
from packed_graphs import PackedGraphs, pack_graphs


class UnsupportedPackedSchema(Exception):
    """A valid cache needs PyG's general collation rules."""


def _plain_category(value):
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, dict):
        raise UnsupportedPackedSchema("nested mapping")
    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (str, bytes)):
        raise UnsupportedPackedSchema("nested sequence")
    return "object"


def direct_packed_batch(packed, rows, weights=None):
    """Collate (payload index, row) selections; raise only for unsupported rules.

    Payloads must already have passed PackedGraphs validation. Errors in their
    tensors are deliberately not caught and reinterpreted as schema fallbacks.
    """
    descriptors = {m: {f[0]: f for f in packed[m]["graphs"].fields} for m in {m for m, _ in rows}}
    fields = descriptors[rows[0][0]]
    if "x" not in fields or any(set(d) != set(fields) for d in descriptors.values()):
        raise UnsupportedPackedSchema("missing x or differing attributes")
    for key in fields:
        first = fields[key]
        _, kind, value, mapping, dim, offsets, stacked = first
        if key in {"num_nodes", "ptr", "face"} or "batch" in key or ("index" in key and key != "edge_index"):
            raise UnsupportedPackedSchema(f"custom increment/storage rule: {key}")
        if key == "edge_index" and (kind != "tensor" or stacked or dim != 1):
            raise UnsupportedPackedSchema("edge_index layout")
        if key == "x" and (kind != "tensor" or stacked or dim != 0 or value.ndim != 2):
            raise UnsupportedPackedSchema("node layout")
        for d in descriptors.values():
            _, other_kind, other, optional, other_dim, _, other_stacked = d[key]
            if optional is not None:
                raise UnsupportedPackedSchema(f"optional attribute: {key}")
            if kind == "tensor":
                if (other_kind != kind or type(other) is not torch.Tensor or (dim, stacked) != (other_dim, other_stacked)
                        or other.dtype != value.dtype or other.ndim != value.ndim
                        or tuple(other.shape[:dim]) + tuple(other.shape[dim+1:]) !=
                           tuple(value.shape[:dim]) + tuple(value.shape[dim+1:])
                        or (stacked and value.ndim != 1)):
                    raise UnsupportedPackedSchema(f"tensor layout: {key}")
            elif other_kind == "tensor":
                raise UnsupportedPackedSchema(f"mixed attribute types: {key}")

    node_counts = [descriptors[m]["x"][5][r+1] - descriptors[m]["x"][5][r] for m, r in rows]
    ptr = torch.tensor([0, *accumulate(node_counts)], dtype=torch.long)
    batch = Batch()
    slices, increments = {}, {}
    for key, (_, kind, _, _, dim, _, _) in fields.items():
        values, lengths = [], []
        for m, r in rows:
            _, row_kind, value, _, row_dim, offsets, _ = descriptors[m][key]
            if row_kind == "tensor":
                start, end = offsets[r:r+2]
                values.append(value.narrow(row_dim, start, end-start))
                lengths.append(end-start)
            else:
                values.append(value if row_kind == "constant" else value[r])
        if kind == "tensor":
            batch[key] = torch.cat(values, dim=dim)
            if key == "edge_index":
                batch[key] += torch.repeat_interleave(ptr[:-1], torch.tensor(lengths)).unsqueeze(0)
            slices[key] = torch.tensor([0, *accumulate(lengths)], dtype=torch.long)
            increments[key] = ptr[:-1] if key == "edge_index" else torch.zeros(len(rows), dtype=torch.long)
        else:
            categories = [_plain_category(v) for v in values]
            if len(set(categories)) != 1:
                raise UnsupportedPackedSchema(f"mixed metadata: {key}")
            numeric = categories[0] == "number"
            batch[key] = torch.tensor(values) if numeric else copy.deepcopy(values)
            slices[key] = torch.arange(len(rows)+1)
            increments[key] = torch.zeros(len(rows), dtype=torch.long) if numeric else None
    batch.ptr = ptr
    batch.batch = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor(node_counts))
    batch._num_graphs = len(rows)
    batch._slice_dict, batch._inc_dict = slices, increments
    labels = torch.stack([packed[m]["labels"][r] for m, r in rows])
    return batch, labels, torch.ones(len(rows)) if weights is None else torch.stack(weights)


class PackedBatchLoader:
    """Same stream and RNG behavior as DataLoader(PreparedActionDataset)."""
    collate_fn = None
    num_workers = 0
    effective_batch_loader = "packed"

    def __init__(self, dataset, batch_size, pin_memory=False):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.dataset, self.batch_size, self.pin_memory = dataset, batch_size, pin_memory
        self.collate_fn = Collater(dataset)
        self.direct_batches = self.fallback_batches = 0
        self.fallback_reasons = {}
        self.last_batching_seconds = 0.0

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def metadata(self):
        return dict(effective=self.effective_batch_loader, direct_batches=self.direct_batches,
                    fallback_batches=self.fallback_batches, fallback_reasons=dict(self.fallback_reasons),
                    last_batching_seconds=self.last_batching_seconds)

    def _finish(self, group, rows, weights):
        started = time.perf_counter()
        try:
            result = direct_packed_batch(group, rows, weights)
            self.direct_batches += 1
        except UnsupportedPackedSchema as exc:
            reason = str(exc)
            self.fallback_batches += 1
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            result = self.collate_fn([(group[m]["graphs"][r], group[m]["labels"][r], w)
                                      for (m, r), w in zip(rows, weights)])
        if self.pin_memory and torch.cuda.is_available():
            result = tuple(value.pin_memory() for value in result)
        self.last_batching_seconds += time.perf_counter() - started
        return result

    def __iter__(self):
        dataset = self.dataset
        assert_feature_paths_usable(dataset.feature_dir, dataset.label_dir,
                                    dataset.options.get("diagnostic_label_dir"),
                                    dataset.options.get("pass_height_diagnostic_label_dir"))
        # DataLoader's iterator consumes a base seed even with num_workers=0.
        torch.empty((), dtype=torch.int64).random_().item()
        rng = random.Random(f"{dataset.seed}:{dataset.epoch}")
        entries = list(dataset.entries)
        if dataset.shuffle:
            rng.shuffle(entries)
        dataset.last_load_seconds = dataset._reconstruction_timing["seconds"] = 0.0
        self.last_batching_seconds = 0.0
        started = time.perf_counter()
        group, pending, weights = [], [], []
        try:
            for offset in range(0, len(entries), dataset.buffer_matches):
                base = len(group)  # an owned partial batch may precede this buffer
                selected = entries[offset:offset + dataset.buffer_matches]
                group.extend(dataset._load_entry(entry) for entry in selected)
                rows = [(m, r) for m in range(base, len(group)) for r in range(len(group[m]["graphs"]))]
                if dataset.shuffle:
                    rng.shuffle(rows)
                for m, r in rows:
                    pending.append((m, r))
                    weights.append(torch.tensor(1.0) if dataset.ip_weights is None else
                                   dataset.ip_weights[selected[m-base]["offset"] + r])
                    if len(pending) == self.batch_size:
                        yield self._finish(group, pending, weights)
                        pending, weights = [], []
                if pending and offset + dataset.buffer_matches < len(entries):
                    # Only the partial batch survives the buffer boundary. Packing
                    # copies its tensor views, releasing all previous match storage.
                    copy_started = time.perf_counter()
                    raw = pack_graphs([group[m]["graphs"][r] for m, r in pending],
                                      torch.stack([group[m]["labels"][r] for m, r in pending]))
                    group = [dict(graphs=PackedGraphs(raw), labels=raw["labels"])]
                    del raw
                    pending = [(0, r) for r in range(len(pending))]
                    weights = [w.clone() for w in weights]
                    self.last_batching_seconds += time.perf_counter() - copy_started
                elif not pending:
                    group.clear()
            if pending:
                yield self._finish(group, pending, weights)
        finally:
            group.clear()
            dataset.last_iteration_seconds = time.perf_counter() - started


def make_batch_loader(dataset, *, batch_size, shuffle=False, pin_memory=False,
                      batch_loader="auto", monitoring="off"):
    from prepared_dataset import PreparedActionDataset
    if batch_loader == "auto" and monitoring != "on" and isinstance(dataset, PreparedActionDataset):
        return PackedBatchLoader(dataset, batch_size, pin_memory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, pin_memory=pin_memory)
    loader.effective_batch_loader = "pyg"
    return loader


def batch_loader_metadata(loader):
    return loader.metadata() if isinstance(loader, PackedBatchLoader) else dict(effective="pyg")
