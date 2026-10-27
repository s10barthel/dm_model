"""Pinned membership and validation for extensions that retain carry graphs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datatools.graph_validation import GraphValidationError, validate_graphs
from datatools.input_diagnostics import read_json, read_parquet
from models.edge_schema import edge_names
from project_config import get_action_graph_dir, get_post_action_graph_dir, get_action_label_dir, get_resolved_action_path


def load_carry_reference(root, match_id):
    root = Path(root)
    metadata = read_json(root / "metadata.json")
    if not (metadata.get("carry_variant") or {}).get("available"):
        raise ValueError("Carry labels-only generation requires an existing completed carry variant")
    return_types = metadata.get("return_types") or [metadata.get("return_type")]
    for mode in ("original", "angle_only"):
        path = get_resolved_action_path(match_id, intended_receiver_mode=mode, root=root, use_carries=True)
        if not path.exists():
            continue
        label_path = get_action_label_dir(return_types[0], intended_receiver_mode=mode, root=root, use_carries=True) / f"{match_id}.pt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing reference carry labels: {label_path}")
        labels = torch.load(label_path, weights_only=False, map_location="cpu")
        actions = read_parquet(path)
        if not actions.index.is_unique or labels.ndim != 2 or labels.shape[1] < 1:
            raise ValueError(f"Malformed pinned carry actions/labels: {match_id}")
        indices = labels[:, 0].numpy()
        if (not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all()
                or not np.array_equal(indices, actions.index.to_numpy())):
            raise ValueError(f"Pinned carry action ordering disagrees with labels: {match_id}")
        for directory in (get_action_graph_dir(root, use_carries=True), get_post_action_graph_dir(root, use_carries=True)):
            if not (directory / f"{match_id}.pt").exists():
                raise FileNotFoundError(f"Missing retained carry graph: {directory / (match_id + '.pt')}")
        carry_rows = actions.loc[actions.spadl_type.eq("ball_carry")]
        if not carry_rows.empty and "carry_definition" not in carry_rows:
            raise ValueError(f"Missing pinned carry definition: {match_id}")
        definitions = carry_rows.get("carry_definition", pd.Series(dtype=object)).dropna().unique()
        if len(definitions) > 1 or (len(definitions) == 1 and definitions[0] != metadata["carry_variant"]["definition"]):
            raise ValueError(f"Pinned carry definition disagrees with metadata: {match_id}")
        return actions, labels, metadata
    raise FileNotFoundError(f"Missing original/angle_only pinned carry actions: {match_id}")


def validate_retained_carry_graphs(root, match_id, labels, metadata, *, graph_validation=True, stats=None):
    if not graph_validation:
        return
    schema = metadata.get("graph_schema") or {}
    for directory in (get_action_graph_dir(root, use_carries=True), get_post_action_graph_dir(root, use_carries=True)):
        path = directory / f"{match_id}.pt"
        graphs = torch.load(path, weights_only=False, map_location="cpu")
        validate_graphs(graphs, context=path, labels=labels, stats=stats)
        for graph in graphs:
            if graph is None:
                continue
            if ((schema.get("node_in_dim") is not None and graph.x.shape[1] != schema["node_in_dim"])
                    or list(edge_names(graph)) != list(edge_names(schema))):
                raise GraphValidationError(f"Retained carry graph schema disagrees with metadata: {path}")
