"""Semantic schemas and physical input preparation for directed graph edges."""
from __future__ import annotations
from collections.abc import Mapping
import torch
from datatools import config

BASE_FEATURES = ("distance", "is_teammate")
EDGE_GROUPS = {
    "v": ("velocity_angle_cos", "velocity_angle_sin"),
    "relative_speed": ("relative_speed",),
    "position_direction": ("position_dx_norm", "position_dy_norm"),
    "relative_velocity": ("relative_vx", "relative_vy"),
    "separation_rate": ("separation_rate",),
}
INTERACTION_GROUPS = ("position_direction", "relative_velocity", "separation_rate")
CANONICAL_FEATURES = BASE_FEATURES + tuple(name for group in EDGE_GROUPS.values() for name in group)
MOVEMENT_FEATURES = frozenset(name for group, names in EDGE_GROUPS.items() if group != "position_direction" for name in names)
EDGE_SCHEMA_VERSION = 1


def legacy_edge_names(width):
    layouts = {0: (), 2: BASE_FEATURES, 4: BASE_FEATURES + EDGE_GROUPS["v"],
               5: BASE_FEATURES + EDGE_GROUPS["v"] + EDGE_GROUPS["relative_speed"]}
    if int(width) not in layouts:
        raise ValueError(f"Unnamed edge width {width} is ambiguous; supply edge_feature_names or upgrade/regenerate the feature run.")
    return layouts[int(width)]


def edge_names(source, width=None):
    if isinstance(source, Mapping):
        names = source.get("edge_feature_names")
        width = source.get("edge_in_dim", 2) if width is None else width
    else:
        names = getattr(source, "edge_feature_names", None)
        width = source.edge_attr.shape[1] if width is None else width
    if names is None:
        return legacy_edge_names(width)
    version = source.get("edge_schema_version", 1) if isinstance(source, Mapping) else getattr(source, "edge_schema_version", 1)
    if version is not None and not bool(torch.as_tensor(version).eq(EDGE_SCHEMA_VERSION).all()):
        raise ValueError(f"Unsupported edge schema version: {version}.")
    # PyG collates per-graph string lists into a list of lists.
    if names and isinstance(names[0], (list, tuple)):
        if any(list(item) != list(names[0]) for item in names):
            raise ValueError("A batch contains different edge schemas; project graphs before batching.")
        names = names[0]
    names = tuple(str(name).removeprefix("edge.") for name in names)
    if len(names) != int(width) or len(set(names)) != len(names):
        raise ValueError("edge_feature_names must uniquely identify every edge input column.")
    if names and (names[:2] != BASE_FEATURES or any(name not in CANONICAL_FEATURES for name in names)):
        raise ValueError(f"Unsupported edge_feature_names: {names}.")
    for group in EDGE_GROUPS.values():
        present = set(group).intersection(names)
        if present and len(present) != len(group):
            raise ValueError(f"Incomplete edge feature group: {group}.")
    return names


def schema_for_names(names, *, explicit=True):
    names = tuple(names)
    result = {"edge_in_dim": len(names), "add_v_edge_features": "velocity_angle_cos" in names,
              "add_relative_speed_edge_features": "relative_speed" in names}
    if explicit:
        result.update(edge_schema_version=EDGE_SCHEMA_VERSION, edge_feature_names=list(names))
        for group in INTERACTION_GROUPS:
            result[f"add_{group}_edge_features"] = EDGE_GROUPS[group][0] in names
    return result


def group_mode(args, group, *, fresh=False):
    mode = args.get(f"{group}_edge_feature_mode")
    if mode is not None:
        mode = str(mode).strip().replace("-", "_")
        if mode not in ("all", "none", "no_poss"):
            raise ValueError(f"Invalid {group}_edge_feature_mode={mode!r}.")
        return mode
    if args.get(f"mask_possessor_{group}_edge_features", False):
        return "no_poss"
    for prefix in ("use", "add"):
        value = args.get(f"{prefix}_{group}_edge_features")
        if value is not None:
            return "all" if value else "none"
    if args.get("edge_feature_names") is not None:
        return "all" if EDGE_GROUPS[group][0] in edge_names(args) else "none"
    if group not in INTERACTION_GROUPS:
        return "all" if EDGE_GROUPS[group][0] in legacy_edge_names(args.get("edge_in_dim", 2)) else "none"
    return "all" if fresh else "none"


def normalize_input_settings(args, *, fresh=False):
    args["scaling"] = bool(args.get("scaling", False))
    args["goal_edge_movement"] = True if args.get("goal_edge_movement") is None else bool(args["goal_edge_movement"])
    if args.get("scaling_profile") not in (None, "physical_v1"):
        raise ValueError("Unsupported scaling profile.")
    args["scaling_profile"] = "physical_v1" if args["scaling"] else None
    for group in INTERACTION_GROUPS:
        args[f"{group}_edge_feature_mode"] = group_mode(args, group, fresh=fresh)
    return args


def selected_edge_names(args, *, fresh=False):
    return BASE_FEATURES + tuple(name for group, names in EDGE_GROUPS.items()
                                 if group_mode(args, group, fresh=fresh) != "none" for name in names)


def generation_edge_names(add_v_edge_features=False, add_relative_speed_edge_features=False,
                          add_position_direction_edge_features=True, add_relative_velocity_edge_features=True,
                          add_separation_rate_edge_features=True):
    if add_relative_speed_edge_features and not add_v_edge_features:
        raise ValueError("Relative-speed edge features require velocity-angle edge features.")
    enabled = (add_v_edge_features, add_relative_speed_edge_features, add_position_direction_edge_features,
               add_relative_velocity_edge_features, add_separation_rate_edge_features)
    return BASE_FEATURES + tuple(name for flag, names in zip(enabled, EDGE_GROUPS.values()) if flag for name in names)


def generation_kwargs(schema):
    names = edge_names(schema)
    return {f"add_{group}_edge_features": EDGE_GROUPS[group][0] in names for group in INTERACTION_GROUPS}


def add_interaction_arguments(parser, *, training=True, default=True):
    for group in INTERACTION_GROUPS:
        option = group.replace("_", "-") + "-edge-features"
        dest = f"{group}_edge_feature_mode" if training else f"add_{group}_edge_features"
        choices = parser.add_mutually_exclusive_group()
        choices.add_argument("--" + option, dest=dest, action="store_const", const="all" if training else True,
                             help=f"Enable {group.replace('_', ' ')} edge inputs (default on for fresh runs).")
        choices.add_argument("--no-" + option, dest=dest, action="store_const", const="none" if training else False)
        if training:
            choices.add_argument("--" + option + "-no-poss", dest=dest, action="store_const", const="no_poss",
                                 help="Zero this group on incoming and outgoing possessor edges.")
        parser.set_defaults(**{dest: ("all" if default else "none") if training else default})
    if training:
        parser.add_argument("--scaling", action="store_true", help="Scale learned inputs with the saved physical_v1 profile.")
        parser.add_argument("--no-goal-edge-movement", dest="goal_edge_movement", action="store_false", default=True,
                            help="Zero movement attributes on every edge involving a goal; retain distance and positional direction.")


def input_setting_flags(args, *, excluded_groups=()):
    """Serialize settings, omitting edge groups emitted by another caller."""
    flags = []
    for group in INTERACTION_GROUPS:
        if group in excluded_groups:
            continue
        mode = group_mode(args, group, fresh=True)
        name = group.replace("_", "-") + "-edge-features"
        flags.append("--no-" + name if mode == "none" else "--" + name + ("-no-poss" if mode == "no_poss" else ""))
    if args.get("scaling", False):
        flags.append("--scaling")
    if args.get("goal_edge_movement", True) is False:
        flags.append("--no-goal-edge-movement")
    return flags


def interaction_values(node_features, edge_index):
    if node_features is None or node_features.shape[1] <= config.NODE_FEATURE_VY:
        raise ValueError("Interaction edge features require preserved raw x/y/vx/vy; regenerate the feature run.")
    xy = node_features[:, config.NODE_FEATURE_X:config.NODE_FEATURE_Y + 1]
    velocity = node_features[:, config.NODE_FEATURE_VX:config.NODE_FEATURE_VY + 1].clone()
    if not torch.isfinite(xy).all() or not torch.isfinite(velocity).all():
        raise ValueError("Interaction edge features require finite raw coordinates and velocities; regenerate the feature run.")
    velocity[node_features[:, config.NODE_FEATURE_IS_GOAL] == 1] = 0
    src, dst = edge_index
    displacement = xy[dst] - xy[src]
    distance = displacement.square().sum(-1).sqrt()
    direction = displacement / distance.clamp_min(1e-6).unsqueeze(-1)
    direction = torch.where((distance > 1e-6).unsqueeze(-1), direction, torch.zeros_like(direction))
    relative = velocity[dst] - velocity[src]
    return dict(zip((*EDGE_GROUPS["position_direction"], *EDGE_GROUPS["relative_velocity"], "separation_rate"),
                    (direction[:, 0], direction[:, 1], relative[:, 0], relative[:, 1], (direction * relative).sum(-1))))


def project_graph_edges(graph, names):
    source = edge_names(graph)
    names = tuple(names)
    missing = set(names).difference(source)
    if missing:
        raise ValueError(f"Graph requires edge_in_dim={len(names)} and is missing requested edge features {sorted(missing)}; use --extend-interaction-edge-features or regenerate the feature run.")
    columns = [source.index(name) for name in names]
    graph.edge_attr = graph.edge_attr[:, columns].clone()
    saved = getattr(graph, "node_self_edge_attr", None)
    if saved is not None:
        graph.node_self_edge_attr = saved[:, columns].clone()
    # Do not add metadata to unnamed legacy artifacts unless projection needs it.
    if getattr(graph, "edge_feature_names", None) is not None or names not in (legacy_edge_names(2), legacy_edge_names(4), legacy_edge_names(5)):
        graph.edge_feature_names = ([list(names) for _ in range(graph.num_graphs)]
                                    if hasattr(graph, "num_graphs") else list(names))
        graph.edge_schema_version = (torch.full((graph.num_graphs,), EDGE_SCHEMA_VERSION, dtype=torch.long,
                                              device=graph.edge_attr.device)
                                     if hasattr(graph, "num_graphs") else EDGE_SCHEMA_VERSION)
    return graph


def mask_edge_inputs(attributes, edge_index, node_features, args, *, names=None, possessor_mask=None, goal_mask=None):
    names = edge_names(args, attributes.shape[1]) if names is None else tuple(names)
    result = attributes.clone()
    src, dst = edge_index
    for group, features in EDGE_GROUPS.items():
        columns = [names.index(name) for name in features if name in names]
        if not columns:
            continue
        mode = group_mode(args, group)
        if mode == "none":
            result[:, columns] = 0
        elif mode == "no_poss":
            identity = possessor_mask if possessor_mask is not None else node_features[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
            incident = identity[src] | identity[dst]
            result[incident.unsqueeze(-1).expand(-1, result.shape[1]) & torch.tensor([name in features for name in names], device=result.device)] = 0
    if args.get("goal_edge_movement", True) is False:
        identity = goal_mask if goal_mask is not None else node_features[:, config.NODE_FEATURE_IS_GOAL] == 1
        incident = identity[src] | identity[dst]
        columns = torch.tensor([name in MOVEMENT_FEATURES for name in names], device=result.device)
        result[incident.unsqueeze(-1) & columns.unsqueeze(0)] = 0
    return result
