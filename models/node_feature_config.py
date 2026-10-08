"""Shared configuration for positional node input ablation."""

from collections.abc import Mapping


def resolve_pos_node_features(args) -> bool:
    """Resolve legacy defaults and reject contradictory xy-only inputs."""
    def get(name, default=None):
        return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)

    value = get("pos_node_features_aware")
    enabled = True if value is None else bool(value)
    if get("xy_only", False) and not enabled:
        raise ValueError("--no-pos-node-features cannot be combined with --xy-only (--xy_only in train.py).")
    return enabled
