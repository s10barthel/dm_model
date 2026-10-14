"""Lossless, versioned CPU graph packing with lazy, non-memoized reconstruction."""
from __future__ import annotations

from collections.abc import Sequence
import copy
from itertools import accumulate
import operator
import time

import torch
from torch_geometric.data import Data

FORMAT = "packed_tensors"
VERSION = 2


def _plain(value, depth=0):
    if depth > 64:
        return False
    if type(value) in (type(None), bool, int, float, str, bytes):
        return True
    if type(value) in (list, tuple):
        return all(_plain(v, depth + 1) for v in value)
    if type(value) is dict:
        return all(type(k) in (str, int, float, bool, bytes, type(None)) and _plain(v, depth + 1)
                   for k, v in value.items())
    return False


def _same(a, b):
    # No implicit coercion of bool/int, tuple/list, or precision.
    if type(a) is not type(b):
        return False
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) is dict:
        return len(a) == len(b) and all(_same(ka, kb) and _same(va, vb)
                                       for (ka, va), (kb, vb) in zip(a.items(), b.items()))
    return a == b


def pack_graphs(graphs, labels):
    """Pack one match, keeping indices local (no PyG batching increments)."""
    if not isinstance(labels, torch.Tensor) or labels.ndim < 1 or len(labels) != len(graphs):
        raise ValueError("Packed graph/label counts differ")
    if any(type(g) is not Data for g in graphs):
        raise ValueError("Packed caches support plain PyG Data graphs only")
    count = len(graphs)
    fields = {}
    for key in sorted(set().union(*(set(g.keys()) for g in graphs))):
        rows = [i for i, graph in enumerate(graphs) if key in graph]
        values = [graphs[i][key] for i in rows]
        first = values[0]
        field = {}
        if len(rows) != count:
            field["rows"] = rows
        if isinstance(first, torch.Tensor):
            if any(not isinstance(v, torch.Tensor) or v.layout != torch.strided or v.is_quantized
                   or v.dtype != first.dtype or v.ndim != first.ndim for v in values):
                raise ValueError(f"Unsupported/inconsistent tensor attribute {key!r}")
            dims = [graphs[i].__cat_dim__(key, v) for i, v in zip(rows, values)]
            dim = dims[0]
            if any(d != dim for d in dims) or (dim is not None and type(dim) is not int):
                raise ValueError(f"Unsupported concatenation dimension for {key!r}")
            stacked = first.ndim == 0 or dim is None
            if stacked:
                if any(v.shape != first.shape for v in values):
                    raise ValueError(f"Inconsistent stacked tensor shape for {key!r}")
                packed = torch.stack([v.detach().cpu() for v in values])
                offsets, dim = list(range(len(values) + 1)), 0
            else:
                dim %= first.ndim
                if any(tuple(v.shape[:dim]) + tuple(v.shape[dim+1:]) !=
                       tuple(first.shape[:dim]) + tuple(first.shape[dim+1:]) for v in values):
                    raise ValueError(f"Inconsistent tensor shape for {key!r}")
                offsets = [0, *accumulate(v.shape[dim] for v in values)]
                packed = torch.cat([v.detach().cpu() for v in values], dim=dim)
            field.update(kind="tensor", value=packed, offsets=offsets, dim=dim, stacked=stacked)
        else:
            if any(not _plain(v) for v in values):
                raise ValueError(f"Unsupported graph attribute {key!r}; expected tensors or plain Python values")
            constant = all(_same(first, v) for v in values)
            field.update(kind="constant" if constant else "objects",
                         value=copy.deepcopy(first if constant else values))
        fields[key] = field
    return dict(format=FORMAT, version=VERSION, count=count, fields=fields,
                labels=labels.detach().cpu().clone())


def validate_payload(payload, expected_count=None):
    """Validate serialized descriptors without constructing any graph objects."""
    def invalid(detail):
        raise ValueError("Invalid packed graph cache: " + detail)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or type(payload.get("version")) is not int or payload.get("version") != VERSION:
        invalid("unsupported format/version")
    count, fields, labels = payload.get("count"), payload.get("fields"), payload.get("labels")
    if type(count) is not int or count < 0 or (expected_count is not None and count != expected_count):
        invalid("sample count")
    if (not isinstance(labels, torch.Tensor) or labels.device.type != "cpu"
            or labels.layout != torch.strided or labels.ndim < 1 or len(labels) != count):
        invalid("label tensor/count")
    if not isinstance(fields, dict) or (count == 0 and fields):
        invalid("attribute descriptors")
    for key, field in fields.items():
        if type(key) is not str or not isinstance(field, dict):
            invalid("attribute descriptor")
        rows = field.get("rows")
        n = count
        if rows is not None:
            if (type(rows) is not list or not rows or any(type(r) is not int for r in rows)
                    or rows != sorted(set(rows)) or rows[0] < 0 or rows[-1] >= count):
                invalid(f"optional rows for {key!r}")
            n = len(rows)
        kind, value = field.get("kind"), field.get("value")
        if kind == "tensor":
            dim, offsets, stacked = field.get("dim"), field.get("offsets"), field.get("stacked")
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.layout != torch.strided or value.is_quantized or value.ndim < 1
                    or type(dim) is not int or not 0 <= dim < value.ndim or type(stacked) is not bool):
                invalid(f"tensor dimensions for {key!r}")
            if (type(offsets) is not list or len(offsets) != n + 1
                    or any(type(x) is not int for x in offsets) or offsets[0] != 0
                    or offsets[-1] != value.shape[dim] or any(a > b for a, b in zip(offsets, offsets[1:]))):
                invalid(f"offsets for {key!r}")
            if stacked and (dim != 0 or offsets != list(range(n + 1))):
                invalid(f"stacked tensor offsets for {key!r}")
        elif kind == "constant":
            if not _plain(value):
                invalid(f"constant attribute {key!r}")
        elif kind == "objects":
            if type(value) is not list or len(value) != n or not all(_plain(v) for v in value):
                invalid(f"object attribute {key!r}")
        else:
            invalid(f"unknown attribute kind for {key!r}")


class PackedGraphs(Sequence):
    """One match's tensor storage; access creates a fresh Data with tensor views.

    Graph tensors are views intended for read-only input processing. Different
    samples occupy disjoint slices. PyG collation owns the resulting batch data.
    No graph, reconstruction callback, or dataset reference is retained here.
    """
    def __init__(self, payload, *, expected_count=None, timing=None):
        validate_payload(payload, expected_count)
        self.count = payload["count"]
        self.timing = timing
        self.fields = []
        for key, field in payload["fields"].items():
            rows = field.get("rows")
            mapping = None if rows is None else {row: i for i, row in enumerate(rows)}
            kind = field["kind"]
            self.fields.append((key, kind, field["value"], mapping,
                                field.get("dim"), field.get("offsets"), field.get("stacked")))

    def __len__(self):
        return self.count

    def __getitem__(self, row):
        row = operator.index(row)
        if row < 0:
            row += self.count
        if not 0 <= row < self.count:
            raise IndexError(row)
        started = time.perf_counter()
        graph = Data()
        for key, kind, value, mapping, dim, offsets, stacked in self.fields:
            index = row if mapping is None else mapping.get(row)
            if index is None:
                continue
            if kind == "tensor":
                start, end = offsets[index], offsets[index + 1]
                view = value.narrow(dim, start, end - start)
                graph[key] = view.squeeze(0) if stacked else view
            else:
                item = value if kind == "constant" else value[index]
                graph[key] = copy.deepcopy(item) if type(item) in (dict, list, tuple) else item
        if self.timing is not None:
            self.timing["seconds"] += time.perf_counter() - started
        return graph


def decode_payload(payload, *, expected_count=None, timing=None):
    return dict(graphs=PackedGraphs(payload, expected_count=expected_count, timing=timing),
                labels=payload["labels"])
