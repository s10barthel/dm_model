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


def validate_with_snapshot(graphs, labels, weights, *, directory=None, context=None):
    from models.node_selection import validate_graph_batch
    try:
        validate_graph_batch(graphs)
    except ValueError as error:
        if directory is None:
            raise
        # No CUDA operations: this runs before the batch is transferred to the GPU.
        try:
            root = Path(directory) / "batch_failures"
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            path = root / f"batch_{stamp}_{uuid.uuid4().hex[:8]}.pt"
            temporary = path.with_suffix(".tmp")
            try:
                torch.save({"graphs": graphs, "labels": labels, "weights": weights,
                            "context": context or {}, "error": str(error)}, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            error.add_note(f"Failing CPU batch saved to {path}")
            path.with_suffix(".json").write_text(json.dumps({"error": str(error), "context": context or {},
                                                            "snapshot": str(path)}, indent=2), encoding="utf-8")
        except Exception as save_error:
            error.add_note(f"Could not finish batch snapshot: {save_error}")
        raise
