"""CPU-validated candidate layouts and grouped node-selection loss/metrics."""
import torch
import torch.nn.functional as F

from datatools import config

TEAMMATE_TASKS = {"pass_intent", "success_intent", "pass_intent_oppo_agn", "action_intent", "success_receiver"}


def batch_identifiers(graphs):
    return {"match_ids": list(getattr(graphs, "evaluation_match_id", [])),
            "source_rows": getattr(graphs, "evaluation_source_row", torch.empty(0)).tolist()}


def validate_graph_batch(graphs):
    edges = graphs.edge_index
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Graph edge_index must be an int64 [2, E] tensor.")
    if edges.numel() and (edges.min() < 0 or edges.max() >= graphs.num_nodes):
        raise ValueError(f"Graph edge index out of bounds: {batch_identifiers(graphs)}")
    if edges.numel() and not torch.equal(graphs.batch[edges[0]], graphs.batch[edges[1]]):
        raise ValueError(f"Edge crosses graph boundaries: {batch_identifiers(graphs)}")
    for name in ("x", "edge_attr"):
        value = getattr(graphs, name, None)
        if value is not None and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite graph {name}: {batch_identifiers(graphs)}")


def selection_layout(graphs, labels, task, include_out):
    """Group equal candidate counts to preserve the existing argsort tie behavior."""
    graph_count = graphs.num_graphs
    batch = graphs.batch
    teammate = graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
    target = labels[:, 5 if task.split("_")[1] == "intent" else 6].clone()
    if task.split("_")[1] != "intent":
        target[target == -1] = labels[target == -1, 4]
    if not bool(torch.isfinite(target).all()) or not torch.equal(target, target.trunc()):
        raise ValueError(f"Node-selection targets must be finite integers: {batch_identifiers(graphs)}")
    target = target.long()
    if task in TEAMMATE_TASKS:
        if include_out:
            raise ValueError(f"{task} does not support include_out.")
        candidate_mask = teammate
    elif task == "failure_receiver":
        candidate_mask = ~teammate
        target -= torch.bincount(batch[teammate], minlength=graph_count)
    else:
        candidate_mask = torch.ones_like(batch, dtype=torch.bool)
    if include_out:
        batch = torch.cat((batch, torch.arange(graph_count)))
        candidate_mask = torch.cat((candidate_mask, torch.ones(graph_count, dtype=torch.bool)))
    groups = {}
    for gi in range(graph_count):
        indices = torch.where((batch == gi) & candidate_mask)[0]
        count = indices.numel()
        if not 0 <= int(target[gi]) < count:
            raise ValueError(f"Node-selection target {int(target[gi])} outside {count} candidates "
                             f"in graph {gi}: {batch_identifiers(graphs)}")
        groups.setdefault(count, []).append((gi, indices))
    return [(torch.tensor([gi for gi, _ in rows]), torch.stack([idx for _, idx in rows]),
             target[[gi for gi, _ in rows]]) for rows in groups.values()]


def selection_loss_metrics(out, layout):
    device = out.device
    count = sum(ids.numel() for ids, _, _ in layout)
    loss = out.new_zeros(())
    # These tensors never retain the autograd graph.
    predictions = torch.empty(count, dtype=torch.long, device=device)
    targets = torch.empty_like(predictions)
    reciprocal_ranks = out.new_empty(count)
    probabilities = out.new_empty(count)
    for ids, indices, target in layout:
        ids, indices, target = ids.to(device), indices.to(device), target.to(device)
        logits = out.reshape(-1)[indices]
        loss = loss + F.cross_entropy(logits, target, reduction="sum")
        with torch.no_grad():
            predictions[ids] = logits.argmax(dim=1)
            targets[ids] = target
            ranks = (logits.argsort(dim=1, descending=True) == target[:, None]).long().argmax(dim=1) + 1
            reciprocal_ranks[ids] = ranks.to(logits.dtype).reciprocal()
            probabilities[ids] = logits.softmax(dim=1).gather(1, target[:, None]).squeeze(1)
    return loss / count, predictions, targets, reciprocal_ranks, probabilities
