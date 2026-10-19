"""Shared configuration and masks for node inputs."""

from collections.abc import Mapping

from datatools import config


def resolve_speed_aware(args) -> bool:
    """Keep scalar node speed enabled for saved settings predating the flag."""
    value = args.get("speed_aware") if isinstance(args, Mapping) else getattr(args, "speed_aware", None)
    return True if value is None else bool(value)


def resolve_goal_poss_dist(args) -> bool:
    """Preserve goal geometry for saved configurations predating the flag."""
    value = args.get("goal_poss_dist") if isinstance(args, Mapping) else getattr(args, "goal_poss_dist", None)
    return True if value is None else bool(value)


def mask_goal_poss_features(node_features, args):
    """Mask only goal-node possessor geometry on a copy of learned inputs."""
    if node_features is None or resolve_goal_poss_dist(args):
        return node_features
    masked = node_features.clone()
    goals = node_features[:, config.NODE_FEATURE_IS_GOAL] == 1
    masked[goals, config.NODE_FEATURE_POSS_DIST : config.NODE_FEATURE_POSS_DY_NORM + 1] = 0
    return masked


def resolve_pos_node_features(args) -> bool:
    """Resolve legacy defaults and reject contradictory xy-only inputs."""
    def get(name, default=None):
        return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)

    value = get("pos_node_features_aware")
    enabled = True if value is None else bool(value)
    if get("xy_only", False) and not enabled:
        raise ValueError("--no-pos-node-features cannot be combined with --xy-only (--xy_only in train.py).")
    return enabled
