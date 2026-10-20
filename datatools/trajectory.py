"""Semantic contract for success-intent early-pass trajectory inputs.

Missing identifiers always mean legacy; never upgrade a saved checkpoint by
looking at its (possibly subsequently mutated) source feature run.
"""
from __future__ import annotations

import json
from pathlib import Path

LEGACY_TRAJECTORY_SPEED = "smoothed_mean_3d_speed_v1"
PREVIOUS_TRAJECTORY_SPEED = "net_displacement_3d_speed_v2"
OFFICIAL_RAW_TRAJECTORY_SPEED = "official_event_net_displacement_3d_1s_v3"
TRAJECTORY_SPEED = "synchronized_official_net_displacement_3d_1s_v4"
TRAJECTORY_SPEED_KEY = "trajectory_speed_definition"


def speed_definition(value=None):
    if isinstance(value, dict):
        value = value.get(TRAJECTORY_SPEED_KEY)
    value = value or LEGACY_TRAJECTORY_SPEED
    if value not in (LEGACY_TRAJECTORY_SPEED, PREVIOUS_TRAJECTORY_SPEED, OFFICIAL_RAW_TRAJECTORY_SPEED, TRAJECTORY_SPEED):
        raise ValueError(f"Unknown trajectory speed definition: {value!r}")
    return value


def require_speed_definition(actual, expected, *, context):
    actual, expected = speed_definition(actual), speed_definition(expected)
    if actual != expected:
        raise ValueError(
            f"{context}: trajectory speed definition mismatch: inputs={actual}, checkpoint/expected={expected}. "
            "Refresh with --refresh-trajectory and retrain success_intent, or select matching stored legacy graphs."
        )
    return actual


def validate_trajectory_graphs(graphs, expected, *, context="success_intent"):
    for index, graph in enumerate(graphs):
        if graph is not None:
            require_speed_definition(getattr(graph, TRAJECTORY_SPEED_KEY, None), expected,
                                     context=f"{context}, graph {index}")


def directory_speed_definition(directory):
    """Read the run contract and verify a representative graph.

    Every graph is additionally checked by dataset/inference consumers when
    loaded, avoiding a second full read of large graph corpora at startup.
    """
    import torch
    from project_config import assert_feature_paths_usable

    directory = Path(directory)
    assert_feature_paths_usable(directory)
    recorded = None
    for parent in (directory, *directory.parents):
        metadata_path = parent / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if TRAJECTORY_SPEED_KEY in metadata:
                recorded = speed_definition(metadata)
            break
    for file in sorted(directory.glob("*.pt")):
        graphs = torch.load(file, map_location="cpu", weights_only=False)
        if not isinstance(graphs, list):
            raise ValueError(f"Invalid success-intent graph artifact: {file}")
        first = next((graph for graph in graphs if graph is not None), None)
        if first is not None:
            definition = speed_definition(getattr(first, TRAJECTORY_SPEED_KEY, None))
            validate_trajectory_graphs(graphs, recorded or definition, context=str(file))
            return recorded or definition
    raise ValueError(f"No usable success-intent graphs in {directory}")


def resolve_training_speed(args, *, checkpoint=None):
    """Resolve fresh training inputs; pin resume/refit to saved semantics."""
    if args.task != "success_intent":
        return
    definition = directory_speed_definition(args.train_feature_dir)
    require_speed_definition(directory_speed_definition(args.valid_feature_dir), definition,
                             context="success_intent training/validation")
    if checkpoint is not None:
        require_speed_definition(definition, speed_definition(checkpoint), context="success_intent resume/refit")
    args.trajectory_speed_definition = definition
