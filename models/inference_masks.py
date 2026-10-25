"""Feature interventions at learned-input boundaries, independent of training flags.

Coordinates, graph construction, physical features and candidate identities retain
their original values. A runtime mask zeros copies consumed by the learned model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from datatools import config
from models.edge_schema import EDGE_GROUPS, edge_names, group_mode
from models.input_scaling import node_schema, NODE_FEATURE_NAMES, OPTIONAL_NODE_FEATURE_NAMES
from models.node_feature_config import resolve_speed_aware
from models.feature_masks import (
    ALIASES, BOOLEAN_FEATURES, NODE_FEATURES, EDGE_OPTIONS, NODE_MODES, EDGE_MODES,
    MODE_ROLES, ROLES, feature_members, selection, node_permissions, node_roles, role_mask,
)
from models.edge_feature_config import (
    normalize_relative_speed_edge_feature_mode,
    normalize_v_edge_feature_mode,
)


EDGE_FEATURE_NAMES = {
    "distance": 0, "is_teammate": 1, "velocity_angle_cos": 2,
    "velocity_angle_sin": 3, "relative_speed": 4,
}

EDGE_FEATURE_NAMES.update({name: None for group in ("position_direction", "relative_velocity", "separation_rate") for name in EDGE_GROUPS[group]})

_VELOCITY = tuple(f"node.{name}" for name in ("vx", "vy", "speed", "accel"))
_GOAL = tuple(f"node.{name}" for name in ("goal_dist", "goal_dx_norm", "goal_dy_norm"))
_POSS_GEOMETRY = tuple(f"node.{name}" for name in ("poss_dist", "poss_dx_norm", "poss_dy_norm"))
_POSS_ANGLES = ("node.poss_vangle_cos", "node.poss_vangle_sin")
_EXTENDED = tuple(f"node.{name}" for name in OPTIONAL_NODE_FEATURE_NAMES[:6])
_V_EDGES = ("edge.velocity_angle_cos", "edge.velocity_angle_sin")

# Each member is (semantic feature, affected node/edge subset).
INFERENCE_MASK_GROUPS = {
    "pos-node-features": tuple((name, "all") for name in ("node.x", "node.y")),
    "vel-node-features": tuple((name, "all") for name in _VELOCITY),
    "goal-features": tuple((name, "all") for name in _GOAL),
    "poss-geometry": tuple((name, "all") for name in _POSS_GEOMETRY),
    "poss-rel-vel-aware": tuple((name, "all") for name in _POSS_ANGLES),
    "speed": (("node.speed", "all"),),
    "accel": (("node.accel", "all"),),
    "ball-z-aware": (("node.ball_z", "all"),),
    "keeper-aware": (("node.is_keeper", "all"),),
    "offside": (("node.offside", "all"),),
    "possessor-aware": tuple((f"node.{name}", "all") for name, index in NODE_FEATURE_NAMES.items() if 13 <= index <= 18),
    "poss-vel-aware": tuple((name, "possessor") for name in _VELOCITY),
    "extend-features": tuple((name, "all") for name in _EXTENDED),
    "pass-lane-features": tuple((name, "all") for name in _EXTENDED[3:5]),
    "lane-survival": (("node.lane_survival", "all"),),
    "v-edge-features": tuple((name, "all") for name in _V_EDGES),
    "relative-speed-edge-features": (("edge.relative_speed", "all"),),
    "mask-possessor-v-edge-features": tuple((name, "possessor_incident") for name in _V_EDGES),
    "mask-possessor-relative-speed-edge-features": (("edge.relative_speed", "possessor_incident"),),
    "motion": tuple((name, "all") for name in (*_VELOCITY, *_POSS_ANGLES, *_V_EDGES, "edge.relative_speed", "edge.relative_vx", "edge.relative_vy", "edge.separation_rate")),
}
INFERENCE_MASK_GROUPS["v-edge-features-no-poss"] = INFERENCE_MASK_GROUPS["mask-possessor-v-edge-features"]
INFERENCE_MASK_GROUPS["relative-speed-edge-features-no-poss"] = INFERENCE_MASK_GROUPS["mask-possessor-relative-speed-edge-features"]
INFERENCE_MASK_GROUPS["goal-poss-geometry"] = tuple((name, "goal") for name in _POSS_GEOMETRY)
INFERENCE_MASK_GROUPS["goal-edge-movement"] = tuple(("edge." + name, "goal_incident")
    for group, members in EDGE_GROUPS.items() if group != "position_direction" for name in members)
_GOAL_CONTROLS = {"goal-poss-geometry", "goal-edge-movement"}

_SPECIAL_ALIASES = {
    "poss-vel-aware": ("vel-node-features", "no-poss"),
    "mask-possessor-v-edge-features": ("velocity-angle-edge-features", "no-poss"),
    "v-edge-features-no-poss": ("velocity-angle-edge-features", "no-poss"),
    "mask-possessor-relative-speed-edge-features": ("relative-speed-edge-features", "no-poss"),
    "relative-speed-edge-features-no-poss": ("relative-speed-edge-features", "no-poss"),
}


def _assignment(token):
    name, separator, value = token.partition("=")
    name, default = _SPECIAL_ALIASES.get(name, (ALIASES.get(name, name), "none"))
    if name.startswith("node.") and name[5:] in NODE_FEATURES:
        name = name[5:]
    edge = name in EDGE_OPTIONS or name.startswith("edge.") or name == "motion"
    mode = selection(value if separator else default, edge=edge, boolean=name in BOOLEAN_FEATURES or name in _GOAL_CONTROLS)
    return name, mode, bool(separator)


def _members(name):
    if name in INFERENCE_MASK_GROUPS:
        return tuple(feature for feature, _ in INFERENCE_MASK_GROUPS[name])
    try:
        return feature_members(name)
    except ValueError as exc:
        raise ValueError(f"Unknown inference mask {name!r}. Use --list-inference-masks for supported names.") from exc


def inference_mask_catalog() -> list[dict[str, Any]]:
    entries = [
        {"name": name, "expansion": [{"feature": feature, "subset": subset} for feature, subset in members]}
        for name, members in INFERENCE_MASK_GROUPS.items()
    ]
    for name in [*(f"node.{key}" for key in NODE_FEATURE_NAMES), *(f"node.{key}" for key in OPTIONAL_NODE_FEATURE_NAMES), *(f"edge.{key}" for key in EDGE_FEATURE_NAMES)]:
        entries.append({"name": name, "expansion": [{"feature": name, "subset": "all"}]})
    known = {entry["name"] for entry in entries}
    for name in (*NODE_FEATURES, *EDGE_OPTIONS):
        if name not in known:
            entries.append({"name": name, "expansion": [{"feature": member, "subset": "all"} for member in feature_members(name)]})
    for entry in entries:
        canonical, _, _ = _assignment(entry["name"])
        entry["selections"] = list(("all", "none") if canonical in BOOLEAN_FEATURES or canonical in _GOAL_CONTROLS else EDGE_MODES
                                    if canonical in EDGE_OPTIONS or canonical.startswith("edge.") or canonical == "motion" else NODE_MODES)
        entry["default"] = _assignment(entry["name"])[1]
    return entries


class _ListMasksAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for entry in inference_mask_catalog():
            expanded = ", ".join(
                member["feature"] + (f" [{member['subset']}]" if member["subset"] != "all" else "")
                for member in entry["expansion"]
            )
            parser._print_message(f"{entry['name']}: {expanded}; ={{{','.join(entry['selections'])}}} (bare: ={entry['default']})\n", sys.stdout)
        parser.exit()


def add_inference_mask_arguments(parser: argparse.ArgumentParser, numbered: bool = False, argv=None) -> None:
    parser.add_argument(
        "--inference-mask", nargs="+", action="append", default=[], metavar="MASK",
        help="Restrict learned inputs with FEATURE[=SELECTION]; bare names mean =none. Optional task: or all: scope.",
    )
    parser.add_argument(
        "--list-inference-masks", nargs=0, action=_ListMasksAction,
        help="List mask names and exact expansions, then exit.",
    )
    if numbered:
        requested = {
            value.split("=", 1)[0]
            for value in (sys.argv[1:] if argv is None else argv)
            if re.fullmatch(r"--inference-mask-\d+(?:=.*)?", value)
        }
        for option in sorted(requested, key=lambda value: (int(value.rsplit("-", 1)[1]), value)):
            identifier = int(option.rsplit("-", 1)[1])
            parser.add_argument(
                option, dest=f"inference_mask_{identifier}", nargs="+", action="append", default=[],
                metavar="MASK", help="A separate combined evaluation mask configuration.",
            )


def flatten_mask_values(values: Iterable[Any] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    flattened = []
    for value in values:
        if isinstance(value, str):
            flattened.append(value)
        else:
            flattened.extend(flatten_mask_values(value))
    return flattened


def resolve_inference_mask_configurations(args) -> list[dict[str, Any]]:
    values = args if isinstance(args, dict) else vars(args)
    combined = flatten_mask_values(values.get("inference_mask"))
    numbered = []
    for key, masks in values.items():
        match = re.fullmatch(r"inference_mask_(\d+)", key)
        if not match or not masks:
            continue
        identifier = int(match.group(1))
        if identifier < 1:
            raise ValueError("Numbered inference masks require a positive configuration number.")
        if len(masks) != 1:
            raise ValueError(f"--inference-mask-{identifier} may only be specified once.")
        flattened = flatten_mask_values(masks)
        if not flattened:
            raise ValueError(f"--inference-mask-{identifier} requires at least one mask.")
        numbered.append({"id": f"mask_{identifier}", "masks": flattened})
    if combined and numbered:
        raise ValueError("Do not mix --inference-mask with numbered inference mask configurations.")
    if numbered:
        return sorted(numbered, key=lambda entry: int(entry["id"].split("_")[-1]))
    return [{"id": "mask", "masks": combined}] if combined else []


def _expand_name(name: str) -> tuple[tuple[str, str], ...]:
    canonical, mode, _ = _assignment(name)
    members = _members(canonical)
    if canonical in _GOAL_CONTROLS:
        return tuple((feature, subset if mode == "none" else "none")
                     for feature, subset in INFERENCE_MASK_GROUPS[canonical])
    subset = {"none": "all", "all": "none", "no-poss": "possessor",
              "attacking": "except_attacking", "defending": "except_defending"}[mode]
    return tuple((feature, "possessor_incident" if subset == "possessor" and feature.startswith("edge.") else subset)
                 for feature in members)


def resolve_scoped_masks(values, participating_tasks: Iterable[str]) -> dict[str, list[str]]:
    tasks = set(participating_tasks)
    flattened = flatten_mask_values(values)
    if flattened and not tasks:
        raise ValueError("Inference masks require at least one participating model task.")
    resolved = {task: [] for task in sorted(tasks)}
    for value in flattened:
        if ":" in value:
            scope, name = value.split(":", 1)
            if scope != "all" and scope not in tasks:
                raise ValueError(f"Inference mask scope {scope!r} has no participating model.")
            targets = tasks if scope == "all" else {scope}
        else:
            if len(tasks) != 1:
                raise ValueError("Unscoped inference masks require one participating task, including auxiliaries; use task: or all:.")
            name = value
            targets = tasks
        _expand_name(name)
        for task in targets:
            if name not in resolved[task]:
                resolved[task].append(name)
    return resolved


_node_schema = node_schema


def _edge_mode(args, relative=False):
    prefix = "relative_speed" if relative else "v"
    normalizer = normalize_relative_speed_edge_feature_mode if relative else normalize_v_edge_feature_mode
    return normalizer(
        args.get(f"{prefix}_edge_feature_mode"),
        **{f"use_{prefix}_edge_features": args.get(f"use_{prefix}_edge_features"),
           f"mask_possessor_{prefix}_edge_features": args.get(f"mask_possessor_{prefix}_edge_features"),
           f"add_{prefix}_edge_features": args.get(f"add_{prefix}_edge_features"),
           "edge_in_dim": args.get("edge_in_dim")},
    )


def _disabled(args, feature: str, column: int, subset: str) -> str | None:
    if feature.startswith("edge."):
        from models.architecture import consumes_edge_attributes
        if not consumes_edge_attributes(args):
            return "architecture does not consume edge attributes"
        name = feature.removeprefix("edge.")
        group = next((group for group, members in EDGE_GROUPS.items() if name in members), None)
        if group:
            mode = group_mode(args, group)
            if mode == "none" or (mode == "no_poss" and subset == "possessor_incident"):
                return "disabled in checkpoint training configuration"
        return None
    if args.get("xy_only", False) and (7 <= column < 12 or 13 <= column < 19):
        return "disabled by xy_only"
    if column in (3, 4) and args.get("pos_node_features_aware") is False:
        return "disabled in checkpoint training configuration"
    if 5 <= column <= 8:
        if args.get("vel_node_features_aware") is False:
            return "disabled in checkpoint training configuration"
        if column == config.NODE_FEATURE_SPEED and not resolve_speed_aware(args):
            return "disabled in checkpoint training configuration"
        if column == 8 and args.get("accel_aware", True) is False:
            return "disabled in checkpoint training configuration"
        if subset == "possessor" and args.get("possessor_aware", False) and not args.get("poss_vel_aware", False):
            return "possessor velocity disabled in checkpoint training configuration"
    if column == 1 and not args.get("keeper_aware", False):
        return "disabled in checkpoint training configuration"
    if column == 12 and not args.get("ball_z_aware", False):
        return "disabled in checkpoint training configuration"
    if 9 <= column <= 11 and args.get("goal_features_aware", True) is False:
        return "disabled in checkpoint training configuration"
    if 13 <= column <= 18 and not args.get("possessor_aware", False):
        return "disabled in checkpoint training configuration"
    if 14 <= column <= 16 and args.get("poss_geometry_aware", True) is False:
        return "disabled in checkpoint training configuration"
    if column in (17, 18) and not args.get("poss_rel_vel_aware", False):
        return "disabled in checkpoint training configuration"
    if feature == "node.offside" and args.get("offside_aware", True) is False:
        return "disabled in checkpoint training configuration"
    if feature in _EXTENDED and not args.get("extend_features", False):
        if column not in (22, 23) or not args.get("pass_lane_features", False):
            return "disabled in checkpoint training configuration"
    return None


@dataclass(frozen=True)
class ResolvedInferenceMask:
    feature: str
    column: int
    subset: str
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class RuntimeInferenceMasks:
    requested: tuple[str, ...]
    operations: tuple[ResolvedInferenceMask, ...]
    node_width: int
    edge_width: int

    def _ensure_inference(self, training: bool):
        if training and self.requested:
            raise ValueError("Runtime inference masks cannot be used during training; call model.eval().")

    def _possessor(self, source: torch.Tensor, identity: torch.Tensor | None) -> torch.Tensor:
        if identity is None:
            identity = source[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
            if not bool(identity.any().item()):
                raise ValueError("Possessor-subset inference mask requires original possessor identity; preserve inference_possessor_mask before preprocessing.")
        identity = identity.to(device=source.device, dtype=torch.bool).reshape(-1)
        if identity.shape[0] != source.shape[0]:
            raise ValueError("inference_possessor_mask does not match the learned node inputs.")
        return identity

    def mask_nodes(self, source: torch.Tensor | None, *, training=False, possessor_mask=None):
        self._ensure_inference(training)
        operations = [op for op in self.operations if op.feature.startswith("node.") and op.status == "applied"]
        if source is None or not operations:
            return source
        if source.ndim != 2 or source.shape[1] != self.node_width:
            raise ValueError("Learned node tensor does not match the resolved inference mask schema.")
        masked = source.clone()
        identity = None
        for op in operations:
            if op.subset == "all":
                masked[:, op.column] = 0
            elif op.subset == "possessor":
                if identity is None:
                    identity = self._possessor(source, possessor_mask)
                masked[identity, op.column] = 0
            elif op.subset.startswith("except_"):
                roles = node_roles(source, possessor_mask)
                retained = MODE_ROLES[op.subset.removeprefix("except_")]
                masked[~role_mask(roles, retained), op.column] = 0
            elif op.subset == "goal":
                masked[source[:, config.NODE_FEATURE_IS_GOAL] == 1, op.column] = 0
        return masked

    def mask_edges(self, source, edge_index, node_features, *, training=False, possessor_mask=None):
        self._ensure_inference(training)
        operations = [op for op in self.operations if op.feature.startswith("edge.") and op.status == "applied"]
        if not operations:
            return source
        if source is None or source.ndim != 2 or source.shape[1] != self.edge_width:
            raise ValueError("Learned edge tensor does not match the resolved inference mask schema.")
        masked = source.clone()
        incident = None
        for op in operations:
            if op.subset == "all":
                masked[:, op.column] = 0
            elif op.subset == "goal_incident":
                goals = node_features[:, config.NODE_FEATURE_IS_GOAL] == 1
                masked[goals[edge_index[0]] | goals[edge_index[1]], op.column] = 0
            elif op.subset == "possessor_incident":
                if incident is None:
                    identity = self._possessor(node_features, possessor_mask)
                    incident = identity[edge_index[0]] | identity[edge_index[1]]
                masked[incident, op.column] = 0
        return masked


def _resolve_runtime(args: Mapping[str, Any], names: list[str]) -> RuntimeInferenceMasks:
    node_schema = _node_schema(args)
    edge_width = int(args.get("edge_in_dim", 0))
    edge_schema = {f"edge.{name}": index for index, name in enumerate(edge_names(args, edge_width))}
    assignments = {}
    for token in names:
        name, mode, explicit = _assignment(token)
        _members(name)
        previous = assignments.get(name)
        if previous is not None:
            old_mode, old_explicit = previous
            if explicit and old_explicit and mode != old_mode:
                raise ValueError(f"Conflicting explicit inference selections for {name}.")
            if old_explicit and not explicit:
                continue
            if not explicit and not old_explicit and old_mode == "none":
                continue
        assignments[name] = (mode, explicit)
    requested = {}
    goal_requests = {}
    for name in sorted(assignments, key=lambda name: (-len(_members(name)), name)):
        mode, _ = assignments[name]
        for feature, subset in _expand_name(name + "=" + mode):
            schema = node_schema if feature.startswith("node.") else edge_schema
            if feature not in schema and name in ("motion", "goal-edge-movement"):
                continue
            if feature not in schema:
                raise ValueError(f"Inference mask {name!r} requires {feature}, absent or ambiguous in checkpoint task={args.get('task')!r} schema (node={args.get('node_in_dim')}, edge={edge_width}).")
            if name in _GOAL_CONTROLS:
                goal_requests[feature] = (subset, schema[feature])
            else:
                requested[feature] = (subset, schema[feature])
    operations = []
    permissions = node_permissions(args, node_schema)
    for feature, (subset, column) in sorted(requested.items()):
        reason = "selection retains the checkpoint's normal inputs" if subset == "none" else None
        if reason is None and feature.startswith("node."):
            removed = {"all": set(ROLES), "possessor": {"possessor"},
                       "except_attacking": {"defending", "goal"},
                       "except_defending": {"possessor", "attacking", "goal"}}[subset]
            if not removed.intersection(permissions[feature]):
                reason = "disabled in checkpoint training configuration"
        elif reason is None:
            reason = _disabled(args, feature, column, subset)
        operations.append(ResolvedInferenceMask(feature, column, subset, "no additional effect" if reason else "applied", reason))
    for feature, (subset, column) in sorted(goal_requests.items()):
        if feature in requested and requested[feature][0] == "all":
            continue
        reason = "selection retains the checkpoint's normal inputs" if subset == "none" else None
        if reason is None and feature.startswith("node."):
            if "goal" not in permissions[feature]:
                reason = "disabled in checkpoint training configuration"
        elif reason is None:
            reason = _disabled(args, feature, column, subset)
            if args.get("goal_edge_movement", True) is False:
                reason = "disabled in checkpoint training configuration"
        operations.append(ResolvedInferenceMask(feature, column, subset, "no additional effect" if reason else "applied", reason))
    return RuntimeInferenceMasks(tuple(names), tuple(operations), int(args.get("node_in_dim", 0)), edge_width)


def resolve_model_masks(args: Mapping[str, Any], names) -> RuntimeInferenceMasks:
    """Validate task-resolved names against checkpoint args without loading weights."""
    return _resolve_runtime(args, flatten_mask_values(names))


def inference_mask_metadata(model) -> dict[str, Any]:
    runtime = getattr(model, "inference_masks", None)
    if runtime is None:
        return {"requested": [], "resolved": [], "status": "unmasked", "configuration_id": "unmasked"}
    signature = sorted({(op.feature, op.subset) for op in runtime.operations})
    identifier = "mask_" + hashlib.sha256(json.dumps(signature).encode("utf-8")).hexdigest()[:12]
    selections = {"all": "none", "none": "all", "possessor": "no-poss", "possessor_incident": "no-poss",
                  "except_attacking": "attacking", "except_defending": "defending"}
    canonical = set()
    for op in runtime.operations:
        if op.subset in ("goal", "goal_incident"):
            canonical.add(("goal-poss-geometry" if op.subset == "goal" else "goal-edge-movement") + "=none")
        else:
            canonical.add(op.feature + "=" + selections[op.subset])
    return {
        "version": 1,
        "requested": list(runtime.requested),
        "canonical": sorted(canonical),
        "resolved": [dict(feature=op.feature, column=op.column, subset=op.subset, status=op.status, reason=op.reason) for op in runtime.operations],
        "status": "applied" if any(op.status == "applied" for op in runtime.operations) else "no additional effect",
        "configuration_id": identifier,
    }


def configure_inference_masks(models: Mapping[Any, Any], masks) -> dict[str, Any]:
    values = flatten_mask_values(masks)
    participating = {str(model.args["task"]) for model in models.values() if model is not None and hasattr(model, "args") and "task" in model.args}
    scoped = resolve_scoped_masks(values, participating) if values else {}
    # Resolve everything before attaching anything, so validation is atomic.
    resolved = {}
    for key, model in models.items():
        if model is None:
            continue
        args = getattr(model, "args", {})
        names = scoped.get(str(args.get("task", "")), [])
        # Loading checkpoints creates train-mode modules; inference entry points
        # switch them to eval immediately before prediction. Attachment itself
        # performs no intervention. Encoder/decoder application enforces eval.
        resolved[key] = _resolve_runtime(args, names) if names else None
    metadata = {"requested": values, "models": {}, "physical_xpass_policy": {
        "physical_values_unchanged": True, "geometry_and_cache_inputs_unchanged": True,
        "masked_pass_intent_may_affect_v5_weights": True,
    }}
    for key, model in models.items():
        if model is None:
            continue
        runtime = resolved[key]
        for component in (model, getattr(model, "encoder", None), getattr(model, "decoder", None)):
            if component is not None and hasattr(component, "__dict__"):
                component.inference_masks = runtime
        metadata["models"][str(key)] = {"task": getattr(model, "args", {}).get("task"), **inference_mask_metadata(model)}
    signature = sorted({
        (entry["task"], entry["configuration_id"])
        for entry in metadata["models"].values() if entry["configuration_id"] != "unmasked"
    })
    metadata["configuration_id"] = (
        "mask_" + hashlib.sha256(json.dumps(signature).encode("utf-8")).hexdigest()[:12]
        if signature else "unmasked"
    )
    metadata["config_id"] = metadata["configuration_id"]
    return metadata
