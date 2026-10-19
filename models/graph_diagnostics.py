"""Failure-only CPU evidence for training batches; outside cache dependencies."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import torch


def sample_description(graphs, index):
    result = {"graph": int(index)}
    for attr, name in (("evaluation_match_id", "match_id"), ("evaluation_source_row", "source_row")):
        values = getattr(graphs, attr, None)
        if values is not None and 0 <= index < len(values):
            value = values[index]
            result[name] = value.item() if isinstance(value, torch.Tensor) else value
    return result


def validate_boundaries(graphs):
    """Check CPU collation bookkeeping before using it to validate connectivity."""
    ptr, batch = graphs.ptr, graphs.batch
    if graphs.x is not None and graphs.x.shape[0] != graphs.num_nodes:
        raise ValueError("Node feature count disagrees with num_nodes.")
    if ptr.dtype != torch.long or ptr.ndim != 1 or len(ptr) != graphs.num_graphs + 1:
        raise ValueError("Invalid graph boundary pointer shape or dtype.")
    if int(ptr[0]) != 0 or int(ptr[-1]) != graphs.num_nodes or bool((ptr[1:] < ptr[:-1]).any()):
        raise ValueError(f"Invalid graph boundary pointers: {ptr.tolist()}")
    expected = torch.repeat_interleave(torch.arange(graphs.num_graphs, device=ptr.device), ptr.diff())
    if batch.dtype != torch.long or batch.shape != expected.shape:
        raise ValueError("Invalid node-to-graph assignment shape or dtype.")
    bad = torch.where(batch != expected)[0]
    if bad.numel():
        node = int(bad[0])
        raise ValueError(f"Node-to-graph assignment disagrees with ptr at node {node}: "
                         f"actual={int(batch[node])}, expected={sample_description(graphs, int(expected[node]))}")


def edge_description(graphs, edge):
    endpoints = graphs.edge_index[:, edge].tolist()
    result = {"edge": int(edge), "endpoints": endpoints}
    for name, node in zip(("source", "destination"), endpoints):
        if 0 <= node < graphs.num_nodes:
            gi = int(graphs.batch[node])
            result[name] = {**sample_description(graphs, gi), "local_node": node - int(graphs.ptr[gi])}
    slices = getattr(graphs, "_slice_dict", {}).get("edge_index")
    if slices is not None:
        owner = int(torch.searchsorted(slices[1:], torch.tensor(edge, device=slices.device), right=True))
        if owner < graphs.num_graphs:
            result["collated_edge_owner"] = sample_description(graphs, owner)
    return result


def _save_failure_snapshot(error, graphs, labels, weights, *, directory=None, context=None,
                           input_graphs=None):
    if directory is None:
        return
    # No CUDA operations: all callers run before device transfer.
    try:
        root = Path(directory) / "batch_failures"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = root / f"batch_{stamp}_{uuid.uuid4().hex[:8]}.pt"
        temporary = path.with_suffix(".tmp")
        payload = {"graphs": graphs, "labels": labels, "weights": weights,
                   "context": context or {}, "error": str(error)}
        if input_graphs is not None:
            payload["input_graphs"] = input_graphs
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        error.add_note(f"Failing CPU batch saved to {path}")
        path.with_suffix(".json").write_text(json.dumps({"error": str(error), "context": context or {},
                                                        "snapshot": str(path)}, indent=2), encoding="utf-8")
    except Exception as save_error:
        error.add_note(f"Could not finish batch snapshot: {save_error}")


def validate_with_snapshot(graphs, labels, weights, *, directory=None, context=None,
                           input_graphs=None):
    from models.node_selection import validate_graph_batch
    try:
        validate_graph_batch(graphs)
    except Exception as error:
        _save_failure_snapshot(error, graphs, labels, weights, directory=directory,
                               context=context, input_graphs=input_graphs)
        raise


def _input_sample_description(graph, index):
    result = {"graph": index}
    for attr, name in (("evaluation_match_id", "match_id"), ("evaluation_source_row", "source_row")):
        value = getattr(graph, attr, None)
        if value is not None:
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.tolist()
            result[name] = value
    return result


def validate_graph_connectivity(graph, index):
    """Validate local edge indices only; finite features are checked after collation."""
    edges = getattr(graph, "edge_index", None)
    if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"Graph edge_index must be an int64 [2, E] tensor: {_input_sample_description(graph, index)}")
    nodes = graph.num_nodes
    if nodes is None:
        raise ValueError(f"Graph node count is unavailable: {_input_sample_description(graph, index)}")
    if edges.numel() and (edges.min() < 0 or edges.max() >= nodes):
        edge = int(torch.where(((edges < 0) | (edges >= nodes)).any(dim=0))[0][0])
        details = dict(_input_sample_description(graph, index), edge=edge,
                       endpoints=edges[:, edge].tolist(), num_nodes=nodes)
        raise ValueError(f"Graph edge index out of bounds before collation: {details}")


class DiagnosticCollater:
    """Failure-only diagnostics around the existing collator (num_workers=0)."""

    def __init__(self, collate_fn, directory=None):
        self.collate_fn = collate_fn
        self.directory = directory
        self.start_epoch(None, "train")

    def start_epoch(self, epoch, phase):
        self.epoch, self.phase, self.batch_index = epoch, phase, 0

    def __call__(self, samples):
        context = {"epoch": self.epoch, "phase": self.phase, "batch": self.batch_index}
        self.batch_index += 1
        graphs, labels, weights = (list(values) for values in zip(*samples))
        for index, graph in enumerate(graphs):
            try:
                validate_graph_connectivity(graph, index)
            except Exception as error:
                _save_failure_snapshot(error, graphs, labels, weights, directory=self.directory,
                                       context={**context, "stage": "before_collation",
                                                "sample": _input_sample_description(graph, index)})
                raise
        result = self.collate_fn(samples)
        validate_with_snapshot(*result, directory=self.directory,
                               context={**context, "stage": "after_collation"}, input_graphs=graphs)
        return result


def install_collation_diagnostics(loader, *, enabled, directory=None):
    """Wrap after construction: PyG discards constructor-supplied collate_fn."""
    if enabled:
        if loader.num_workers != 0:
            raise ValueError("Graph collation diagnostics require num_workers=0.")
        loader.collate_fn = DiagnosticCollater(loader.collate_fn, directory)
