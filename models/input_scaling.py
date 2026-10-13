"""Fixed, zero-preserving scaling of copies at learned-input boundaries."""
from __future__ import annotations
from typing import Any, Mapping
from datatools import config
from models.edge_schema import edge_names

NODE_FEATURE_NAMES = {
    "is_teammate": 0, "is_keeper": 1, "is_goal": 2,
    "x": 3, "y": 4, "vx": 5, "vy": 6, "speed": 7, "accel": 8,
    "goal_dist": 9, "goal_dx_norm": 10, "goal_dy_norm": 11, "ball_z": 12,
    "is_possessor": 13, "poss_dist": 14, "poss_dx_norm": 15, "poss_dy_norm": 16,
    "poss_vangle_cos": 17, "poss_vangle_sin": 18,
}
OPTIONAL_NODE_FEATURE_NAMES = (
    "nearby_opponents_to_target", "nearest_opponent_to_target", "closer_opponents_to_goal",
    "nearest_opponent_to_pass", "potential_interceptors", "potential_blockers",
    "offside", "lane_survival", "pass_mean_speed", "pass_direction_x", "pass_direction_y",
    "pass_direction_z", "intended_target",
)
def node_schema(args: Mapping[str, Any]) -> dict[str, int]:
    width = int(args.get("node_in_dim", 0))
    schema = {f"node.{name}": index for name, index in NODE_FEATURE_NAMES.items() if index < width}
    explicit_names = args.get("node_feature_names")
    if explicit_names is not None:
        if len(explicit_names) != width or len(set(explicit_names)) != width:
            raise ValueError("Ambiguous node_feature_names schema: names must uniquely identify every input column.")
        return {name if str(name).startswith("node.") else f"node.{name}": index for index, name in enumerate(explicit_names)}
    task = str(args.get("task", ""))
    failure = int(task == "failure_receiver")
    node_task = str(args.get("gnn_task", "node")).startswith("node")
    lane = int(bool(args.get("lane_survival", False)) and node_task)
    base_width = width - failure - lane
    if failure:
        schema["node.intended_target"] = width - 1
    if lane:
        schema["node.lane_survival"] = width - failure - 1
    success_variant = task == "success_intent" or args.get("feature_variant") == "success_intent"
    if success_variant:
        if base_width in (23, 24):
            for index, name in enumerate(OPTIONAL_NODE_FEATURE_NAMES[8:12], start=19):
                schema[f"node.{name}"] = index
            if base_width == 24:
                schema["node.offside"] = 23
    elif base_width in (25, 26):
        for index, name in enumerate(OPTIONAL_NODE_FEATURE_NAMES[:6], start=19):
            schema[f"node.{name}"] = index
        if base_width == 26:
            schema["node.offside"] = 25
    elif base_width == 20:
        schema["node.offside"] = 19
    # Unknown optional layouts deliberately remain unresolved. Core columns are
    # stable, but an optional slot cannot safely be inferred from an arbitrary width.
    return schema


COUNT_FEATURES = {"nearby_opponents_to_target", "closer_opponents_to_goal", "potential_interceptors", "potential_blockers"}
DISTANCE_FEATURES = {"goal_dist", "poss_dist", "nearest_opponent_to_target", "nearest_opponent_to_pass", "distance"}
SPEED_FEATURES = {"vx", "vy", "speed", "pass_mean_speed", "relative_speed", "relative_vx", "relative_vy", "separation_rate"}


def feature_divisor(name):
    """Return the physical_v1 divisor for a semantic input name."""
    name = name.split(".")[-1]
    if name == "x":
        return float(config.FIELD_SIZE[0])
    if name == "y":
        return float(config.FIELD_SIZE[1])
    if name in DISTANCE_FEATURES:
        return 100.0
    if name in SPEED_FEATURES or name in ("accel", "ball_z"):
        return 10.0
    if name in COUNT_FEATURES:
        return 11.0
    return 1.0


def scale_nodes(features, args):
    if features is None or not args.get("scaling", False):
        return features
    schema = node_schema(args)
    if set(schema.values()) != set(range(features.shape[1])):
        raise ValueError("Scaling requires an unambiguous semantic node schema; supply node_feature_names.")
    divisors = features.new_ones(features.shape[1])
    for name, column in schema.items():
        divisors[column] = feature_divisor(name)
    return features / divisors


def scale_edges(features, args):
    if not args.get("scaling", False):
        return features
    return features / features.new_tensor([feature_divisor(name) for name in edge_names(args, features.shape[1])])


def scale_destinations(features, args):
    if features is None or not args.get("scaling", False):
        return features
    # Existing layouts: xy; xy+polar radius/angle; xy+dx/dy+three distances.
    expected_width = 2 + 2 * bool(args.get("polar_dest", False)) + 5 * bool(args.get("more_dest_features", False))
    if features.shape[-1] != expected_width:
        raise ValueError("Destination inputs do not match the saved scaling layout.")
    divisors = features.new_ones(features.shape[-1])
    if not args.get("normalize_dest", False):
        divisors[:2] = features.new_tensor(config.FIELD_SIZE)
    offset = 2
    if args.get("polar_dest", False):
        divisors[offset] = 100.0
        offset += 2
    if args.get("more_dest_features", False):
        divisors[offset:offset + 2] = features.new_tensor(config.FIELD_SIZE)
        divisors[offset + 2:offset + 5] = 100.0
        offset += 5
    return features / divisors
