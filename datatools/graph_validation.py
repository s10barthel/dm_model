"""Raw graph contracts and immutable preservation evidence; CLI imports stay light."""
from __future__ import annotations

import argparse
import hashlib
import json
import time


class GraphValidationError(ValueError):
    """A graph artifact violates its declared structural contract."""


def graph_validation_enabled(value=True):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if value in ("on", "off"):
        return value == "on"
    raise argparse.ArgumentTypeError("graph validation must be 'on' or 'off'")


def add_graph_validation_argument(parser):
    parser.add_argument("--graph-validation", type=graph_validation_enabled, default=True,
                        metavar="{on,off}", help="Validate raw graphs and verify saved graph outputs (default: on).")


def graph_validation_flags(args):
    value = args.get("graph_validation", True) if isinstance(args, dict) else getattr(args, "graph_validation", True)
    return ["--graph-validation", "on" if graph_validation_enabled(value) else "off"]


def validation_stats(enabled=True):
    return dict(enabled=graph_validation_enabled(enabled), checked_files=0, checked_graphs=0,
                verified_output_files=0, validation_seconds=0.0, fingerprint_seconds=0.0,
                readback_seconds=0.0)


def merge_validation_stats(records, enabled=True):
    result = validation_stats(enabled)
    for record in records:
        if record:
            for key in result:
                if key != "enabled":
                    result[key] += record.get(key, 0)
    return result


def _location(context, row=None, labels=None):
    text = str(context)
    if row is not None:
        text += f"; source row {row}"
        if labels is not None and row < len(labels) and labels.ndim == 2 and labels.shape[1]:
            text += f"; action index {labels[row, 0].item():g}"
    return text


def validate_graphs(graphs, *, context, labels=None, dense=True, graph_validation=True, stats=None):
    if not graph_validation_enabled(graph_validation):
        return
    import numpy as np
    import torch
    from models.edge_schema import edge_names
    started = time.perf_counter()
    checked = 0
    try:
        if not isinstance(graphs, (list, tuple)):
            raise GraphValidationError(f"{context}: expected a graph list")
        if labels is not None and len(graphs) != len(labels):
            raise GraphValidationError(f"{context}: graph/label row count mismatch")
        for row, graph in enumerate(graphs):
            if graph is None:
                continue
            where = _location(context, row, labels)
            def fail(message):
                raise GraphValidationError(f"{where}: {message}")
            x, edges, attrs = (getattr(graph, name, None) for name in ("x", "edge_index", "edge_attr"))
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1:
                fail("node features must be a nonempty [N,F] tensor")
            n = int(x.shape[0])
            try:
                declared_nodes = int(graph.num_nodes)
            except (AttributeError, TypeError, ValueError):
                fail("invalid declared node count")
            if declared_nodes != n:
                fail("node count disagrees with node features")
            if not isinstance(edges, torch.Tensor) or edges.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or edges.ndim != 2 or edges.shape[0] != 2:
                fail("edge_index must be an integer [2,E] tensor")
            if attrs is None:
                fail("Graph artifact is missing edge_attr required for connectivity validation")
            if not isinstance(attrs, torch.Tensor) or attrs.ndim != 2 or attrs.shape[0] != edges.shape[1]:
                fail("edge attribute row count disagrees with edge_index")
            try:
                names = edge_names(graph)
            except (ValueError, TypeError, AttributeError) as error:
                fail(f"invalid edge schema: {error}")
            if len(names) != attrs.shape[1] or len(set(names)) != len(names):
                fail("edge attribute schema width/names disagree")
            ids = getattr(graph, "node_ids", None)
            if ids is not None:
                try:
                    id_count = len(ids)
                except TypeError:
                    fail("node IDs must be an ordered sequence")
                if id_count != n:
                    fail("node-ID count disagrees with node features")
            array = edges.detach().cpu().numpy().astype(np.int64, copy=False)
            if array.size and (array.min() < 0 or array.max() >= n):
                fail(f"edge index outside [0,{n})")
            counts = np.bincount(array[0] * n + array[1], minlength=n * n)
            if np.any(counts > 1) or (dense and (edges.shape[1] != n * n or np.any(counts != 1))):
                missing = [(int(v // n), int(v % n)) for v in np.flatnonzero(counts == 0)[:5]] if dense else []
                duplicates = [(int(v // n), int(v % n), int(counts[v])) for v in np.flatnonzero(counts > 1)[:5]]
                fail(f"invalid {'dense ' if dense else ''}connectivity: {edges.shape[1]} entries, "
                     f"{np.count_nonzero(counts)} distinct; missing pairs {missing}; duplicate pairs/counts {duplicates}")
            checked += 1
    finally:
        if stats is not None:
            stats["validation_seconds"] += time.perf_counter() - started
            stats["checked_graphs"] += checked
            stats["checked_files"] += 1


def _tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def graph_fingerprints(graphs, *, graph_validation=True, stats=None):
    """Snapshots own bytes/digests, never views into mutable graph tensors."""
    if not graph_validation_enabled(graph_validation):
        return None
    from models.edge_schema import edge_names
    started = time.perf_counter()
    result = []
    for graph in graphs:
        if graph is None:
            result.append(None)
            continue
        names = tuple(edge_names(graph))
        result.append(dict(nodes=int(graph.num_nodes), node_ids=json.dumps(getattr(graph, "node_ids", None), default=str),
                           edges=_tensor_digest(graph.edge_index), names=names,
                           attributes=_tensor_digest(graph.edge_attr),
                           columns={name: _tensor_digest(graph.edge_attr[:, i]) for i, name in enumerate(names)}))
    if stats is not None:
        stats["fingerprint_seconds"] += time.perf_counter() - started
    return tuple(result)


def assert_preserved(before, graphs, *, context, labels=None, allow_availability=False,
                     allow_edge_extension=False, graph_validation=True, stats=None):
    if not graph_validation_enabled(graph_validation):
        return
    after = graph_fingerprints(graphs, stats=stats)
    if len(before) != len(after):
        raise GraphValidationError(f"{context}: graph row count changed")
    for row, (old, new) in enumerate(zip(before, after)):
        if allow_availability and (old is None or new is None):
            continue
        if allow_edge_extension and old is not None and new is not None:
            valid = all(old[key] == new[key] for key in ("nodes", "node_ids", "edges"))
            valid = valid and all(new["columns"].get(name) == value for name, value in old["columns"].items())
        else:
            valid = old == new
        if not valid:
            raise GraphValidationError(f"{_location(context, row, labels)}: retained graph connectivity, "
                                       "node order, schema or edge attributes changed")


def verify_graph_file(path, expected, *, context, labels=None, graph_validation=True, stats=None):
    if not graph_validation_enabled(graph_validation):
        return
    import torch
    started = time.perf_counter()
    try:
        value = torch.load(path, weights_only=False, map_location="cpu")
    finally:
        if stats is not None:
            stats["readback_seconds"] += time.perf_counter() - started
    validate_graphs(value, context=context, labels=labels, stats=stats)
    assert_preserved(expected, value, context=context, labels=labels, stats=stats)
    if stats is not None:
        stats["verified_output_files"] += 1
