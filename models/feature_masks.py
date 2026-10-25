"""Feature selections shared by training, checkpoint reconstruction and inference.

Public selections describe retained inputs. Roles are captured from the original
graph, so masking identity inputs never changes subsequent selections.
"""
from __future__ import annotations

import argparse
import torch

from models.input_scaling import NODE_FEATURE_NAMES, OPTIONAL_NODE_FEATURE_NAMES, node_schema

VERSION = 1
NODE_MODES = ("all", "attacking", "defending", "no-poss", "none")
EDGE_MODES = ("all", "no-poss", "none")
ROLES = ("possessor", "attacking", "defending", "goal")
MODE_ROLES = {
    "all": ROLES, "none": (), "attacking": ("possessor", "attacking"),
    "defending": ("defending",), "no-poss": ("attacking", "defending", "goal"),
}
NODE_GROUPS = {
    "pos-node-features": ("x", "y"),
    "vel-node-features": ("vx", "vy", "speed", "accel"),
    "goal-features": ("goal_dist", "goal_dx_norm", "goal_dy_norm"),
    "poss-geometry": ("poss_dist", "poss_dx_norm", "poss_dy_norm"),
    "poss-velocity-angles": ("poss_vangle_cos", "poss_vangle_sin"),
    "extended-node-features": OPTIONAL_NODE_FEATURE_NAMES[:6],
    "pass-lane-features": OPTIONAL_NODE_FEATURE_NAMES[3:5],
}
BOOLEAN_FEATURES = {"possessor-id": "is_possessor", "keeper-id": "is_keeper", "ball-z": "ball_z"}
NODE_FEATURES = {**{name: (name,) for name in ("vx", "vy", "speed", "accel", "offside", "lane-survival")},
                 **NODE_GROUPS, **{name: (feature,) for name, feature in BOOLEAN_FEATURES.items()}}
NODE_FEATURES["lane-survival"] = ("lane_survival",)
EDGE_OPTIONS = {
    "velocity-angle-edge-features": "v", "relative-speed-edge-features": "relative_speed",
    "position-direction-edge-features": "position_direction",
    "relative-velocity-edge-features": "relative_velocity", "separation-rate-edge-features": "separation_rate",
}
ALIASES = {"v-edge-features": "velocity-angle-edge-features", "extend-features": "extended-node-features",
           "poss-rel-vel-aware": "poss-velocity-angles", "keeper-aware": "keeper-id", "ball-z-aware": "ball-z"}


def selection(value, *, edge=False, boolean=False):
    value = str(value).strip().lower().replace("_", "-")
    value = {"true": "all", "false": "none"}.get(value, value)
    choices = ("all", "none") if boolean else EDGE_MODES if edge else NODE_MODES
    if value not in choices:
        raise ValueError(f"Invalid feature selection {value!r}; expected {', '.join(choices)}.")
    return value


def feature_members(name):
    """Return semantic members, retaining support for schema-qualified leaves."""
    from models.edge_schema import EDGE_GROUPS, CANONICAL_FEATURES
    name = ALIASES.get(name, name)
    if name in NODE_FEATURES:
        return tuple("node." + member for member in NODE_FEATURES[name])
    if name in EDGE_OPTIONS:
        return tuple("edge." + member for member in EDGE_GROUPS[EDGE_OPTIONS[name]])
    if name.startswith("node.") and name[5:] in {*NODE_FEATURE_NAMES, *OPTIONAL_NODE_FEATURE_NAMES}:
        return (name,)
    if name.startswith("edge.") and name[5:] in CANONICAL_FEATURES:
        return (name,)
    raise ValueError(f"Unknown feature {name!r}. Use --list-inference-masks for supported names.")


def expand_selections(values):
    """Broad groups first, leaves last, independent of CLI ordering."""
    result = {}
    for name in sorted(values, key=lambda name: (-len(feature_members(name)), name)):
        for member in feature_members(name):
            result[member] = selection(values[name], edge=member.startswith("edge."), boolean=name in BOOLEAN_FEATURES)
    return result


def normalize_feature_settings(args):
    """Keep legacy settings untouched; validate and version explicit selections."""
    version = args.get("feature_mask_version", VERSION)
    if version != VERSION:
        raise ValueError(f"Unsupported feature_mask_version={version!r}.")
    values = args.get("feature_selections") or {}
    normalized = {}
    for name, value in values.items():
        name = ALIASES.get(name, name)
        feature_members(name)
        normalized[name] = selection(value, edge=name in EDGE_OPTIONS, boolean=name in BOOLEAN_FEATURES)
    if normalized:
        args["feature_selections"] = normalized
        args["feature_mask_version"] = VERSION
    if "lane-survival" in normalized:
        enabled = normalized["lane-survival"] != "none"
        args["lane_survival"] = enabled
        args["lane_survival_mode"] = (args.get("lane_survival_mode") or "max") if enabled else None
    for name, group in EDGE_OPTIONS.items():
        if name in normalized:
            args[group + "_edge_feature_mode"] = normalized[name].replace("-", "_")
    return args


def node_permissions(args, schema):
    """Reconstruct historical masks, then override explicitly selected features."""
    masks = {name: set(ROLES) for name in schema}

    def disable(names, roles=ROLES):
        for name in names:
            if "node." + name in masks:
                masks["node." + name].difference_update(roles)

    if args.get("xy_only", False):
        disable([name for name, index in NODE_FEATURE_NAMES.items() if 7 <= index < 12 or 13 <= index < 19])
    if not args.get("possessor_aware", False):
        disable([name for name, index in NODE_FEATURE_NAMES.items() if 13 <= index < 19])
    elif not args.get("poss_vel_aware", False) and not args.get("xy_only", False):
        # Legacy xy_only erases the possessor flag before the old possessor-only
        # velocity mask runs; preserve its retained vx/vy on saved checkpoints.
        disable(NODE_GROUPS["vel-node-features"], ("possessor",))
    for key, default, names in (
        ("poss_rel_vel_aware", False, NODE_GROUPS["poss-velocity-angles"]),
        ("poss_geometry_aware", True, NODE_GROUPS["poss-geometry"]),
        ("keeper_aware", False, ("is_keeper",)), ("ball_z_aware", False, ("ball_z",)),
        ("goal_features_aware", True, NODE_GROUPS["goal-features"]),
        ("vel_node_features_aware", True, NODE_GROUPS["vel-node-features"]),
        ("pos_node_features_aware", True, NODE_GROUPS["pos-node-features"]),
        ("speed_aware", True, ("speed",)), ("accel_aware", True, ("accel",)),
        ("offside_aware", True, ("offside",)),
    ):
        if args.get(key, default) is False:
            disable(names)
    if args.get("task") != "success_intent" and not args.get("extend_features", False):
        names = set(NODE_GROUPS["extended-node-features"])
        if args.get("pass_lane_features", False):
            names.difference_update(NODE_GROUPS["pass-lane-features"])
        disable(names)
    for name, mode in expand_selections(args.get("feature_selections") or {}).items():
        if name in masks:
            masks[name] = set(MODE_ROLES[mode])
    if args.get("goal_poss_dist", True) is False:
        disable(NODE_GROUPS["poss-geometry"], ("goal",))
    return {name: tuple(role for role in ROLES if role in allowed) for name, allowed in masks.items()}


def node_roles(source, possessor_mask=None):
    goals = source[:, NODE_FEATURE_NAMES["is_goal"]] == 1
    teammates = source[:, NODE_FEATURE_NAMES["is_teammate"]] == 1
    possessor = (source[:, NODE_FEATURE_NAMES["is_possessor"]] == 1 if possessor_mask is None
                 else possessor_mask.to(device=source.device, dtype=torch.bool).reshape(-1))
    if possessor.shape[0] != source.shape[0]:
        raise ValueError("inference_possessor_mask does not match the learned node inputs.")
    return {"goal": goals, "possessor": possessor & ~goals,
            "attacking": teammates & ~possessor & ~goals, "defending": ~teammates & ~goals}


def role_mask(roles, allowed):
    result = torch.zeros_like(roles["goal"])
    for role in allowed:
        result |= roles[role]
    return result


def mask_node_inputs(source, args, *, possessor_mask=None, prepared=True, positions=True, identities=True):
    if source is None:
        return source
    if not args.get("feature_selections") and not args.get("possessor_aware", False):
        assert not args.get("extend_features", False) and not args.get("pass_lane_features", False)
    settings = dict(args, node_in_dim=source.shape[1])
    if not prepared:
        settings["lane_survival"] = False
        if settings.get("task") == "failure_receiver":
            settings["task"] = "pass_success"
    schema = node_schema(settings)
    if (args.get("task") != "success_intent" and args.get("pass_lane_features", False)
            and not args.get("extend_features", False)
            and not all("node." + name in schema for name in NODE_GROUPS["pass-lane-features"])):
        raise ValueError("Pass-lane features require nearest-opponent-to-pass and potential-interceptor node columns.")
    if prepared:
        for member, mode in expand_selections(args.get("feature_selections") or {}).items():
            if member.startswith("node.") and member not in schema and mode != "none":
                if args.get("task") == "success_intent" and member[5:] in NODE_GROUPS["extended-node-features"]:
                    continue  # This task has a separate trajectory block, as in legacy training.
                raise ValueError(f"Feature selection requires {member}, absent or ambiguous in task={args.get('task')!r} node schema.")
    permissions = node_permissions(settings, schema)
    roles = node_roles(source, possessor_mask)
    masked = source.clone()
    for name, column in schema.items():
        if not positions and name in ("node.x", "node.y"):
            continue
        if not identities and name == "node.is_possessor":
            continue
        if permissions[name] != ROLES:
            masked[~role_mask(roles, permissions[name]), column] = 0
    return masked


class _FeatureSelection(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        name = self.feature_name
        # Bare options that already existed retain their historical meaning.
        if value is None and self.legacy_dest:
            setattr(namespace, self.legacy_dest, self.legacy_value)
            return
        if name == "lane-survival" and value not in (*NODE_MODES, "true", "false"):
            from physical_pass_model import normalize_pc_xpass_lane_survival_mode
            try:
                namespace.lane_survival_mode = normalize_pc_xpass_lane_survival_mode(value)
            except ValueError as exc:
                raise argparse.ArgumentError(self, str(exc)) from exc
            namespace.lane_survival = True
            return
        try:
            mode = selection(value, edge=name in EDGE_OPTIONS, boolean=name in BOOLEAN_FEATURES)
        except ValueError as exc:
            raise argparse.ArgumentError(self, str(exc)) from exc
        values = dict(getattr(namespace, "feature_selections", None) or {})
        if name in values and values[name] != mode:
            raise argparse.ArgumentError(self, f"Conflicting explicit selections for {name}.")
        values[name] = mode
        namespace.feature_selections = values
        if name in EDGE_OPTIONS:
            setattr(namespace, EDGE_OPTIONS[name] + "_edge_feature_mode", mode.replace("-", "_"))
        if name in ("extended-node-features", "pass-lane-features", "lane-survival"):
            dest = {"extended-node-features": "extend_features", "pass-lane-features": "pass_lane_features",
                    "lane-survival": "lane_survival"}[name]
            setattr(namespace, dest, mode != "none")
            if name == "lane-survival" and mode != "none" and getattr(namespace, "lane_survival_mode", None) is None:
                namespace.lane_survival_mode = "max"


class _FeatureHelpFormatter(argparse.HelpFormatter):
    def _format_actions_usage(self, actions, groups):
        # Python 3.11 leaves extra separators for entirely hidden legacy groups,
        # which makes its wrapped-usage assertion fail on otherwise valid parsers.
        visible_groups = [group for group in groups
                          if any(action.help != argparse.SUPPRESS for action in group._group_actions)]
        return super()._format_actions_usage(actions, visible_groups)


def add_feature_selection_arguments(parser):
    """Upgrade overlapping options without changing parser-specific defaults."""
    parser.formatter_class = _FeatureHelpFormatter
    def detach(option):
        old = parser._option_string_actions.get(option)
        if old is None:
            return None
        parser.set_defaults(**{old.dest: parser._defaults.get(old.dest, old.default)})
        old.feature_action_position = parser._actions.index(old)
        old.feature_exclusive_groups = [(group, group._group_actions.index(old))
                                        for group in parser._mutually_exclusive_groups if old in group._group_actions]
        old.option_strings.remove(option)
        del parser._option_string_actions[option]
        if not old.option_strings:
            parser._actions.remove(old)
            for group in (*parser._action_groups, *parser._mutually_exclusive_groups):
                if old in group._group_actions:
                    group._group_actions.remove(old)
        return old

    def replace_action(action, old):
        # argparse's usage formatter requires mutually exclusive actions to be
        # contiguous and ordered like the parser's main action list.
        parser._actions.remove(action)
        parser._actions.insert(old.feature_action_position + bool(old.option_strings), action)
        for group, position in old.feature_exclusive_groups:
            group._group_actions.insert(position + bool(old.option_strings), action)

    for name in (*NODE_FEATURES, *EDGE_OPTIONS):
        option = "--" + name
        old = detach(option)
        legacy_dest, legacy_value = None, True
        if old is not None:
            legacy_dest = old.dest
            legacy_value = getattr(old, "const", True)
            if name == "lane-survival":
                legacy_value = "max"
        action = parser.add_argument(option, action=_FeatureSelection, nargs="?" if old else None,
                                     default=argparse.SUPPRESS, metavar="SELECTION",
                                     help="Retain inputs: " + ", ".join(("true", "false") if name in BOOLEAN_FEATURES else EDGE_MODES if name in EDGE_OPTIONS else NODE_MODES) + ".")
        action.feature_name, action.legacy_dest, action.legacy_value = name, legacy_dest, legacy_value
        if old is not None:
            replace_action(action, old)
    for option, dest in (("--goal-poss-geometry", "goal_poss_dist"), ("--goal-nodes", "goal_nodes_aware"),
                         ("--goal-edge-movement", "goal_edge_movement")):
        action = parser.add_argument(option, dest=dest, type=parse_boolean, default=argparse.SUPPRESS)
        for group in parser._mutually_exclusive_groups:
            if any(member.dest == dest for member in group._group_actions):
                position = max(parser._actions.index(member) for member in group._group_actions)
                parser._actions.remove(action)
                parser._actions.insert(position + 1, action)
                group._group_actions.append(action)
    # Existing candidate flags also accept an explicit boolean.
    for option in ("--action-intent-include-goals", "--include-goals"):
        old = detach(option)
        if old is not None:
            action = parser.add_argument(option, dest=old.dest, nargs="?", const=True, type=parse_boolean,
                                         default=argparse.SUPPRESS)
            replace_action(action, old)
    parser.add_argument("--lane-survival-mode", default=argparse.SUPPRESS,
                        help="Lane-survival computation: max, top_N or top_pass_N.")
    legacy = {
        "possessor-aware", "possessor_aware", "keeper-aware", "keeper_aware", "ball-z-aware", "ball_z_aware",
        "poss-vel-aware", "poss_vel_aware", "poss-rel-vel-aware", "poss_rel_vel_aware", "goal-poss-dist",
        "goal-nodes-aware", "extend-features", "extend_features", "v-edge-features",
        "v-edge-features-no-poss", "relative-speed-edge-features-no-poss",
        "position-direction-edge-features-no-poss", "relative-velocity-edge-features-no-poss",
        "separation-rate-edge-features-no-poss",
    }
    negative = {*NODE_FEATURES, *EDGE_OPTIONS, "v-edge-features", "possessor-aware", "keeper-aware", "ball-z-aware",
                "poss-vel-aware", "poss-rel-vel-aware", "extend-features", "goal-nodes", "include-goals",
                "action-intent-include-goals", "goal-edge-movement"}
    hidden = {"--" + name for name in legacy} | {"--no-" + name for name in negative}
    for action in parser._actions:
        if set(action.option_strings).intersection(hidden):
            action.help = argparse.SUPPRESS


def parse_boolean(value):
    try:
        return selection(value, boolean=True) == "all"
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def feature_selection_flags(args):
    values = args.get("feature_selections") or {}
    return [token for name in sorted(values) for token in ("--" + name, values[name])]


def feature_signature(args):
    normalize_feature_settings(args)
    schema = node_schema(args)
    return {"version": VERSION, "node": {name: list(roles) for name, roles in node_permissions(args, schema).items()}}
