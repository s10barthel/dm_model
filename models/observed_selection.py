"""CPU-validated positions for observed node and execution-branch outputs."""
import torch
from datatools.config import LABEL_INDEX


def observed_layout(graphs, labels, include_out=False, branches=False):
    target = labels[:, LABEL_INDEX['intent_index']]
    counts = graphs.ptr[1:] - graphs.ptr[:-1]
    valid = torch.isfinite(target) & (target == target.trunc())
    valid &= (target >= 0) & (target < counts + int(include_out))
    branch = labels[:, LABEL_INDEX['success']] if branches else None
    if branch is not None:
        valid &= torch.isfinite(branch) & ((branch == 0) | (branch == 1))
    if not bool(valid.all()):
        i = int(torch.where(~valid)[0][0])
        matches = getattr(graphs, 'evaluation_match_id', ['unknown'] * len(target))
        rows = getattr(graphs, 'evaluation_source_row', ['unknown'] * len(target))
        raise ValueError(f"Invalid observed target/branch in graph {i}, match {matches[i]}, source row {rows[i]}.")
    target = target.long()
    positions = graphs.ptr[:-1] + target
    if include_out:
        positions = torch.where(target == counts, graphs.num_nodes + torch.arange(len(target)), positions)
    return positions, branch.long() if branch is not None else None


def select_observed(out, positions, branches=None):
    return out[positions] if branches is None else out[positions, branches]
