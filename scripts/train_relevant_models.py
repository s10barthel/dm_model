from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datatools import config
from physical_pass_model import PHYSICAL_XPASS_SOURCE
from models.utils import (
    infer_feature_graph_schema,
    infer_training_edge_schema,
    get_model_record,
    get_model_records,
    mask_possessor_v_edge_features_for_mode,
    normalize_v_edge_feature_mode,
    parse_model_id,
    use_v_edge_features_for_mode,
    validate_model_record_consistency,
)
from datatools.success_intent import SUCCESS_INTENT_LABEL_SOURCE, SUCCESS_INTENT_TRAINING_FILTER
from project_config import (
    generate_model_run_id,
    generate_run_id,
    get_action_label_dir,
    get_action_graph_dir,
    get_action_graph_intent_train_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_intent_train_label_dir,
    get_model_bundle_root,
    get_model_run_root,
    get_success_intent_graph_dir,
    get_success_intent_label_dir,
    infer_feature_run_intended_receiver_modes,
    infer_feature_run_return_types,
    load_feature_run_metadata,
    load_model_bundle_metadata,
    resolve_feature_run_id,
    resolve_feature_root,
    validate_intended_receiver_mode,
    validate_return_type,
    validate_return_type_for_target_family,
    write_run_metadata,
)

WRAPPER_FEATURE_DEFAULTS = {
    "xy_only": False,
    "possessor_aware": True,
    "keeper_aware": True,
    "ball_z_aware": True,
    "poss_vel_aware": True,
    "poss_rel_vel_aware": False,
    "poss_geometry_aware": True,
    "goal_features_aware": True,
    "goal_nodes_aware": True,
    "accel_aware": True,
    "offside_aware": True,
    "extend_features": False,
}

LOW_LEVEL_FEATURE_FLAGS = {
    "xy_only": "--xy_only",
    "possessor_aware": "--possessor_aware",
    "keeper_aware": "--keeper_aware",
    "ball_z_aware": "--ball_z_aware",
    "poss_vel_aware": "--poss_vel_aware",
    "poss_rel_vel_aware": "--poss_rel_vel_aware",
    "extend_features": "--extend_features",
}

LOW_LEVEL_FALSE_FLAGS = {
    "poss_geometry_aware": "--no-poss-geometry",
    "goal_features_aware": "--no-goal-features",
    "goal_nodes_aware": "--no-goal-nodes",
}

LOW_LEVEL_BOOL_OVERRIDE_FLAGS = {
    "accel_aware": ("--accel", "--no-accel"),
    "offside_aware": ("--offside", "--no-offside"),
}

MODEL_TOGGLE_SPECS = (
    (
        "action_intent",
        "action-intent",
        "Train the action_intent checkpoint.",
        "Skip training the action_intent checkpoint.",
    ),
    (
        "pass_intent",
        "pass-intent",
        "Train the pass_intent checkpoint.",
        "Skip training the pass_intent checkpoint.",
    ),
    (
        "success_intent",
        "success-intent",
        "Train the mode-independent success_intent checkpoint.",
        "Skip training the success_intent checkpoint.",
    ),
    (
        "pass_success",
        "pass-success",
        "Train the pass_success checkpoint.",
        "Skip training the pass_success checkpoint.",
    ),
    (
        "pass_height",
        "pass-height",
        "Train the pass_height checkpoint.",
        "Skip training the pass_height checkpoint.",
    ),
    (
        "outcome_scoring",
        "outcome-scoring",
        "Train the outcome_scoring checkpoint.",
        "Skip training the outcome_scoring checkpoint.",
    ),
    (
        "outcome_conceding",
        "outcome-conceding",
        "Train the outcome_conceding checkpoint.",
        "Skip training the outcome_conceding checkpoint.",
    ),
    (
        "failure_receiver",
        "failure-receiver",
        "Train the failure_receiver checkpoint.",
        "Skip training the failure_receiver checkpoint.",
    ),
)
MODEL_TOGGLE_DEFAULTS = {
    "action_intent": True,
    "pass_intent": True,
    "success_intent": True,
    "pass_success": True,
    "pass_height": False,
    "outcome_scoring": True,
    "outcome_conceding": True,
    "failure_receiver": False,
}
BATCH_SIZE_DEFAULTS = {
    "action_intent": 256,
    "pass_intent": 256,
    "success_intent": 256,
    "pass_success": 512,
    "pass_height": 512,
    "outcome_scoring": 512,
    "outcome_conceding": 512,
    "failure_receiver": 256,
}
MODE_DEPENDENT_TASKS = {
    "action_intent",
    "pass_intent",
    "pass_success",
    "pass_height",
    "outcome_scoring",
    "outcome_conceding",
    "failure_receiver",
}
OUTCOME_TASKS = {"outcome_scoring", "outcome_conceding"}
RETAINED_BUNDLE_TASKS = (
    "action_intent",
    "pass_intent",
    "pass_success",
    "pass_height",
    "outcome_scoring",
    "outcome_conceding",
)


def add_bool_override(
    parser: argparse.ArgumentParser,
    option_name: str,
    dest: str,
    enable_help: str,
    disable_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{option_name}", dest=dest, action="store_true", help=enable_help)
    group.add_argument(f"--no-{option_name}", dest=dest, action="store_false", help=disable_help)
    parser.set_defaults(**{dest: None})


def resolve_wrapper_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    resolved_flags = {
        name: WRAPPER_FEATURE_DEFAULTS[name] if getattr(args, name, None) is None else bool(getattr(args, name))
        for name in WRAPPER_FEATURE_DEFAULTS
    }
    if not resolved_flags["possessor_aware"] and resolved_flags["extend_features"]:
        raise ValueError(
            "--extend-features requires possessor-aware features; remove --extend-features or enable --possessor-aware."
        )
    return resolved_flags


def append_low_level_feature_flags(command: list[str], feature_flags: dict[str, bool]) -> list[str]:
    command = list(command)
    for name, cli_flag in LOW_LEVEL_FEATURE_FLAGS.items():
        if feature_flags[name]:
            command.append(cli_flag)
    for name, cli_flag in LOW_LEVEL_FALSE_FLAGS.items():
        if not feature_flags[name]:
            command.append(cli_flag)
    for name, (enabled_flag, disabled_flag) in LOW_LEVEL_BOOL_OVERRIDE_FLAGS.items():
        command.append(enabled_flag if feature_flags[name] else disabled_flag)
    return command


def append_physical_xpass_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    if bool(getattr(args, "use_physical_xpass", False)):
        command.append("--use_physical_xpass")
    command.extend(["--model-variant", str(getattr(args, "model_variant", "gat_phys_logit_offset"))])
    if getattr(args, "physical_cache_dir", None):
        command.extend(["--physical-cache-dir", str(args.physical_cache_dir)])
    command.extend(["--physical-eps", str(getattr(args, "physical_eps", 1e-4))])
    physical_xpass_floor = getattr(args, "physical_xpass_floor", None)
    if physical_xpass_floor is not None:
        command.extend(["--physical-xpass-floor", str(physical_xpass_floor)])
    freeze_beta0 = bool(getattr(args, "freeze_beta0", False))
    freeze_beta1 = getattr(args, "freeze_beta1", None)
    if freeze_beta1 is None:
        freeze_beta1 = not bool(getattr(args, "learn_physical_scale", True))
    if freeze_beta0:
        command.append("--freeze-beta0")
    if bool(freeze_beta1):
        command.append("--freeze-beta1")
    command.extend(["--residual-distance-threshold", str(getattr(args, "residual_distance_threshold", 30.0))])
    residual_lambda = float(getattr(args, "residual_regularization_lambda", 0.0) or 0.0)
    if residual_lambda:
        command.extend(["--residual-regularization-lambda", str(residual_lambda)])
    residual_clip_value = getattr(args, "residual_clip_value", None)
    if residual_clip_value is not None:
        command.extend(["--residual-clip-value", str(residual_clip_value)])
    short_residual_lambda = getattr(args, "short_residual_regularization_lambda", None)
    if short_residual_lambda is not None:
        command.extend(["--short-residual-regularization-lambda", str(short_residual_lambda)])
    long_residual_lambda = getattr(args, "long_residual_regularization_lambda", None)
    if long_residual_lambda is not None:
        command.extend(["--long-residual-regularization-lambda", str(long_residual_lambda)])
    short_residual_clip_value = getattr(args, "short_residual_clip_value", None)
    if short_residual_clip_value is not None:
        command.extend(["--short-residual-clip-value", str(short_residual_clip_value)])
    long_residual_clip_value = getattr(args, "long_residual_clip_value", None)
    if long_residual_clip_value is not None:
        command.extend(["--long-residual-clip-value", str(long_residual_clip_value)])
    return command


def physical_xpass_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "use_physical_xpass": bool(getattr(args, "use_physical_xpass", False)),
        "model_variant": str(getattr(args, "model_variant", "gat_phys_logit_offset")),
        "source": PHYSICAL_XPASS_SOURCE,
        "physical_cache_dir": getattr(args, "physical_cache_dir", None),
        "physical_eps": float(getattr(args, "physical_eps", 1e-4)),
        "physical_xpass_floor": getattr(args, "physical_xpass_floor", None),
        "freeze_beta0": bool(getattr(args, "freeze_beta0", False)),
        "freeze_beta1": bool(getattr(args, "freeze_beta1", not bool(getattr(args, "learn_physical_scale", True)))),
        "learn_physical_scale": not bool(getattr(args, "freeze_beta1", not bool(getattr(args, "learn_physical_scale", True)))),
        "residual_regularization_lambda": float(getattr(args, "residual_regularization_lambda", 0.0) or 0.0),
        "residual_clip_value": getattr(args, "residual_clip_value", None),
        "residual_distance_threshold": float(getattr(args, "residual_distance_threshold", 30.0)),
        "short_residual_regularization_lambda": getattr(args, "short_residual_regularization_lambda", None),
        "long_residual_regularization_lambda": getattr(args, "long_residual_regularization_lambda", None),
        "short_residual_clip_value": getattr(args, "short_residual_clip_value", None),
        "long_residual_clip_value": getattr(args, "long_residual_clip_value", None),
    }


def cli_v_edge_feature_mode(args: argparse.Namespace) -> str:
    return normalize_v_edge_feature_mode(
        getattr(args, "v_edge_feature_mode", None),
        use_v_edge_features=getattr(args, "use_v_edge_features", None),
        mask_possessor_v_edge_features=getattr(args, "mask_possessor_v_edge_features", None),
    )


def edge_feature_flag_for_mode(v_edge_feature_mode: str) -> str:
    mode = normalize_v_edge_feature_mode(v_edge_feature_mode)
    if mode == "none":
        return "--no-v-edge-features"
    if mode == "no_poss":
        return "--v-edge-features-no-poss"
    return "--v-edge-features"


def append_edge_feature_flag(command: list[str], v_edge_feature_mode: str) -> list[str]:
    command = list(command)
    command.append(edge_feature_flag_for_mode(v_edge_feature_mode))
    return command


def append_runtime_flags(command: list[str], device: str | None = None, pin_memory: bool | None = False) -> list[str]:
    command = list(command)
    if device:
        command.extend(["--device", str(device)])
    command.append("--pin-memory" if bool(pin_memory) else "--no-pin-memory")
    return command


def get_training_control_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "early_stopping": bool(getattr(args, "early_stopping", True)),
        "early_stopping_patience": int(getattr(args, "early_stopping_patience", 10)),
        "early_stopping_min_epochs": int(getattr(args, "early_stopping_min_epochs", 30)),
        "early_stopping_min_delta": float(getattr(args, "early_stopping_min_delta", 1e-5)),
    }


def resolve_batch_sizes(args: argparse.Namespace, enabled_tasks: dict[str, bool]) -> dict[str, int]:
    general_batch_size = getattr(args, "batch_size", None)
    batch_sizes = {}
    for task, enabled in enabled_tasks.items():
        if not enabled:
            continue
        task_batch_size = getattr(args, f"{task}_batch_size", None)
        batch_sizes[task] = int(
            task_batch_size
            if task_batch_size is not None
            else general_batch_size
            if general_batch_size is not None
            else BATCH_SIZE_DEFAULTS[task]
        )
    return batch_sizes


def validate_batch_size_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    batch_size_values = [("--batch-size", getattr(args, "batch_size", None))]
    for _, option_name, _, _ in MODEL_TOGGLE_SPECS:
        batch_size_values.append((f"--{option_name}-batch-size", getattr(args, f"{option_name.replace('-', '_')}_batch_size", None)))
    for option_name, value in batch_size_values:
        if value is not None and int(value) < 1:
            parser.error(f"{option_name} must be at least 1.")


def append_training_control_flags(command: list[str], settings: dict[str, object]) -> list[str]:
    command = list(command)
    command.extend(
        [
            "--early-stopping-patience",
            str(settings["early_stopping_patience"]),
            "--early-stopping-min-epochs",
            str(settings["early_stopping_min_epochs"]),
            "--early-stopping-min-delta",
            str(settings["early_stopping_min_delta"]),
        ]
    )
    if not bool(settings["early_stopping"]):
        command.append("--no-early-stopping")
    return command


def get_cli_value(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return str(command[value_index])


def describe_returncode(returncode: int) -> str:
    unsigned_code = int(returncode) & 0xFFFFFFFF
    hex_code = f"0x{unsigned_code:08X}"
    if unsigned_code == 0xC0000005:
        return f"{hex_code} Windows access violation from native code, commonly CUDA/PyTorch/driver-side."
    return f"{hex_code} process exit code"


def resolve_enabled_tasks(args: argparse.Namespace) -> OrderedDict[str, bool]:
    explicit_toggles = [
        (
            task,
            f"--{option_name}" if getattr(args, task) else f"--no-{option_name}",
        )
        for task, option_name, _, _ in MODEL_TOGGLE_SPECS
        if getattr(args, task) is not None
    ]
    if args.success_intent_only or getattr(args, "only_pass_height", False):
        only_flag = "--success-intent-only" if args.success_intent_only else "--only-pass-height"
        only_task = "success_intent" if args.success_intent_only else "pass_height"
        if args.success_intent_only and getattr(args, "pass_intent_model_id", None):
            raise ValueError("--pass-intent-model-id requires --pass-success and cannot be combined with --success-intent-only.")
        if args.success_intent_only and getattr(args, "pass_height_ipw_model_id", None):
            raise ValueError("--pass-height-ipw-model-id requires --pass-height and cannot be combined with --success-intent-only.")
        if explicit_toggles:
            toggles = ", ".join(flag for _, flag in explicit_toggles)
            raise ValueError(f"{only_flag} cannot be combined with explicit per-model toggles: {toggles}.")
        if args.success_intent_only and getattr(args, "only_pass_height", False):
            raise ValueError("--success-intent-only cannot be combined with --only-pass-height.")
        if getattr(args, "only_pass_height", False):
            if getattr(args, "pass_intent_model_id", None):
                raise ValueError("--pass-intent-model-id requires --pass-success and cannot be combined with --only-pass-height.")
            pass_height_ipw = bool(getattr(args, "pass_height_ipw", False))
            if getattr(args, "pass_height_ipw_model_id", None) and not pass_height_ipw:
                raise ValueError("--pass-height-ipw-model-id requires --pass-height-ipw.")
            if pass_height_ipw and not getattr(args, "pass_height_ipw_model_id", None):
                raise ValueError("--only-pass-height --pass-height-ipw requires --pass-height-ipw-model-id.")
        return OrderedDict((task, task == only_task) for task, _, _, _ in MODEL_TOGGLE_SPECS)

    enabled_tasks = OrderedDict(
        (task, MODEL_TOGGLE_DEFAULTS[task] if getattr(args, task) is None else bool(getattr(args, task)))
        for task, _, _, _ in MODEL_TOGGLE_SPECS
    )
    if not any(enabled_tasks.values()):
        raise ValueError("At least one model must be enabled.")
    pass_success_ipw = bool(getattr(args, "pass_success_ipw", True))
    pass_height_ipw = bool(getattr(args, "pass_height_ipw", False))
    pass_intent_model_id = getattr(args, "pass_intent_model_id", None)
    pass_height_ipw_model_id = getattr(args, "pass_height_ipw_model_id", None)
    if pass_intent_model_id and not pass_success_ipw:
        raise ValueError("--pass-intent-model-id requires --pass-success-ipw.")
    if pass_intent_model_id and not enabled_tasks["pass_success"]:
        raise ValueError("--pass-intent-model-id requires --pass-success.")
    if pass_intent_model_id and enabled_tasks["pass_intent"]:
        raise ValueError("--pass-intent-model-id requires --no-pass-intent.")
    if (
        pass_success_ipw
        and enabled_tasks["pass_success"]
        and not enabled_tasks["pass_intent"]
        and not pass_intent_model_id
    ):
        raise ValueError("--pass-success requires --pass-intent or --pass-intent-model-id.")
    if pass_height_ipw_model_id and not pass_height_ipw:
        raise ValueError("--pass-height-ipw-model-id requires --pass-height-ipw.")
    if pass_height_ipw_model_id and not enabled_tasks["pass_height"]:
        raise ValueError("--pass-height-ipw-model-id requires --pass-height.")
    if pass_height_ipw_model_id and enabled_tasks["pass_intent"]:
        raise ValueError("--pass-height-ipw-model-id requires --no-pass-intent.")
    if (
        pass_height_ipw
        and enabled_tasks["pass_height"]
        and not enabled_tasks["pass_intent"]
        and not pass_height_ipw_model_id
    ):
        raise ValueError("--pass-height-ipw requires --pass-intent or --pass-height-ipw-model-id.")
    if pass_intent_model_id and pass_height_ipw_model_id and str(pass_intent_model_id) != str(pass_height_ipw_model_id):
        raise ValueError("--pass-intent-model-id and --pass-height-ipw-model-id must reference the same checkpoint.")
    if args.outcome_scoring_trial is not None and not enabled_tasks["outcome_scoring"]:
        raise ValueError("--outcome-scoring-trial requires --outcome-scoring.")
    if args.outcome_conceding_trial is not None and not enabled_tasks["outcome_conceding"]:
        raise ValueError("--outcome-conceding-trial requires --outcome-conceding.")
    return enabled_tasks


def source_model_ids(args: argparse.Namespace) -> dict[str, str]:
    pass_intent_model_id = getattr(args, "pass_intent_model_id", None)
    pass_height_ipw_model_id = getattr(args, "pass_height_ipw_model_id", None)
    model_id = pass_intent_model_id or pass_height_ipw_model_id
    return {"pass_intent": str(model_id)} if model_id else {}


def resolve_diagnostic_label_dir(
    feature_run_id: str | None,
    intended_receiver_mode: str | None,
) -> Path | None:
    if not feature_run_id or not intended_receiver_mode:
        return None
    try:
        feature_root = resolve_feature_root(feature_run_id)
    except FileNotFoundError:
        return None
    label_dir = get_action_label_dir(
        config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
        intended_receiver_mode=intended_receiver_mode,
        root=feature_root,
    )
    return label_dir if label_dir.exists() else None


def validate_diagnostic_feature_run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    requires_outcome_config: bool,
) -> None:
    if not args.diagnostic_feature_run_id:
        return
    if not requires_outcome_config:
        parser.error("--diagnostic-feature-run-id requires --outcome-scoring or --outcome-conceding.")
    try:
        args.diagnostic_feature_run_id = resolve_feature_run_id(
            args.diagnostic_feature_run_id,
            required=True,
            allow_latest=False,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    label_dir = resolve_diagnostic_label_dir(args.diagnostic_feature_run_id, args.intended_receiver_mode)
    if label_dir is None:
        parser.error(
            f"Diagnostic feature run {args.diagnostic_feature_run_id!r} does not expose "
            f"return_type={config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE!r} for "
            f"intended_receiver_mode={args.intended_receiver_mode!r}."
        )


def diagnostic_metadata(
    args: argparse.Namespace,
    intended_receiver_mode: str | None,
    *,
    outcome_enabled: bool | None = None,
    feature_run_id_fallback: str | None = None,
) -> dict[str, object]:
    if outcome_enabled is None:
        outcome_enabled = any(getattr(args, "enabled_tasks", {}).get(task, False) for task in OUTCOME_TASKS)
    if not outcome_enabled:
        return {
            "diagnostic_target": None,
            "diagnostic_return_type": None,
            "diagnostic_feature_run_id": None,
            "diagnostic_label_dir": None,
        }

    diagnostic_feature_run_id = (
        getattr(args, "diagnostic_feature_run_id", None)
        or getattr(args, "feature_run_id", None)
        or feature_run_id_fallback
    )
    label_dir = resolve_diagnostic_label_dir(diagnostic_feature_run_id, intended_receiver_mode)
    return {
        "diagnostic_target": config.GOAL_NEXT10_DIAGNOSTIC_TARGET,
        "diagnostic_return_type": config.GOAL_NEXT10_DIAGNOSTIC_RETURN_TYPE,
        "diagnostic_feature_run_id": diagnostic_feature_run_id,
        "diagnostic_label_dir": str(label_dir) if label_dir is not None else None,
    }


def validate_external_pass_intent_model_id(
    args: argparse.Namespace,
    feature_flags: dict[str, bool],
    resolved_feature_run_id: str,
    runtime_schema: dict[str, int | bool] | None = None,
    *,
    model_id_attr: str = "pass_intent_model_id",
    option_name: str = "--pass-intent-model-id",
    runtime_context: str = "pass_success",
) -> str | None:
    del feature_flags
    pass_intent_model_id = getattr(args, model_id_attr, None)
    if not pass_intent_model_id:
        return None

    task, _ = parse_model_id(str(pass_intent_model_id))
    if task != "pass_intent":
        raise ValueError(f"{option_name} must reference a pass_intent checkpoint, got {pass_intent_model_id!r}.")

    record = get_model_record(str(pass_intent_model_id))
    mismatches: list[str] = []
    if record.get("task") != "pass_intent":
        mismatches.append(f"task={record.get('task')!r}")
    if not record.get("has_weights"):
        mismatches.append("missing best_weights.pt or best_model.json")

    if runtime_schema is None:
        feature_root = resolve_feature_root(resolved_feature_run_id)
        runtime_schema = resolve_pass_success_runtime_schema(feature_root, cli_v_edge_feature_mode(args))

    required_schema = record.get("graph_schema", {})
    runtime_node_dim = runtime_schema.get("node_in_dim")
    required_node_dim = required_schema.get("node_in_dim")
    if required_node_dim is not None and runtime_node_dim is not None:
        if int(runtime_node_dim) < int(required_node_dim):
            mismatches.append(
                f"requires node_in_dim={int(required_node_dim)}, "
                f"but {runtime_context} runtime provides node_in_dim={int(runtime_node_dim)}"
            )
    elif required_node_dim is not None:
        mismatches.append(f"requires node_in_dim={int(required_node_dim)}, but {runtime_context} runtime node schema is unknown")

    runtime_edge_dim = int(runtime_schema.get("edge_in_dim", 0) or 0)
    required_edge_dim = int(required_schema.get("edge_in_dim", 0) or 0)
    if required_edge_dim and runtime_edge_dim < required_edge_dim:
        suggestion = ""
        if required_edge_dim >= 4 and runtime_edge_dim <= 2:
            suggestion = " Use --v-edge-features or a feature run with velocity edge features."
        mismatches.append(
            f"requires edge_in_dim={required_edge_dim}, "
            f"but {runtime_context} runtime provides edge_in_dim={runtime_edge_dim}.{suggestion}"
        )

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"External pass_intent checkpoint {pass_intent_model_id!r} is invalid: {details}.")

    return str(pass_intent_model_id)


def resolve_pass_success_runtime_schema(feature_root: Path, v_edge_feature_mode: str) -> dict[str, int | bool]:
    action_graph_dir = get_action_graph_dir(feature_root)
    try:
        feature_schema = infer_feature_graph_schema(action_graph_dir)
    except FileNotFoundError:
        metadata = load_feature_run_metadata(Path(feature_root).name, required=False) or {}
        metadata_schema = dict(metadata.get("graph_schema") or {})
        metadata_edge_dim = int(metadata_schema.get("edge_in_dim", 4) or 4)
        feature_schema = {
            "node_in_dim": int(metadata_schema.get("node_in_dim", 25) or 25),
            "edge_in_dim": metadata_edge_dim,
            "add_v_edge_features": bool(metadata_schema.get("add_v_edge_features", metadata_edge_dim > 2)),
        }
    training_edge_schema = infer_training_edge_schema(feature_schema, v_edge_feature_mode=v_edge_feature_mode)
    return {
        "node_in_dim": int(feature_schema["node_in_dim"]) if feature_schema.get("node_in_dim") is not None else None,
        "edge_in_dim": int(training_edge_schema["edge_in_dim"]),
        "add_v_edge_features": bool(training_edge_schema["add_v_edge_features"]),
    }


def retained_bundle_model_ids(model_ids: dict[str, str]) -> dict[str, str]:
    return {task: model_ids[task] for task in RETAINED_BUNDLE_TASKS if task in model_ids}


def derive_bundle_shared_context(
    final_model_ids: dict[str, str],
    cli_args: argparse.Namespace,
    resolved_feature_run_id: str | None,
) -> dict[str, object]:
    retained_model_ids = retained_bundle_model_ids(final_model_ids)
    if retained_model_ids:
        model_records = get_model_records(retained_model_ids)
        shared = validate_model_record_consistency(
            model_records,
            require_feature_run_id=False,
            require_intended_receiver_mode=False,
            require_return_type=False,
            require_target_family=False,
        )
        graph_schema = dict(shared["graph_schema"])
        retained_modes = {
            normalize_v_edge_feature_mode(
                record.get("feature_signature", {}).get("v_edge_feature_mode"),
                add_v_edge_features=record.get("graph_schema", {}).get("add_v_edge_features"),
                edge_in_dim=record.get("graph_schema", {}).get("edge_in_dim"),
            )
            for record in model_records.values()
        }
        v_edge_feature_mode = next(iter(retained_modes)) if len(retained_modes) == 1 else "mixed"
        return {
            "feature_run_id": resolved_feature_run_id or shared.get("feature_run_id"),
            "intended_receiver_mode": getattr(cli_args, "intended_receiver_mode", None),
            "return_type": cli_args.return_type,
            "target_family": cli_args.target_family,
            "graph_schema": graph_schema,
            "use_v_edge_features": bool(graph_schema.get("add_v_edge_features", False)),
            "v_edge_feature_mode": v_edge_feature_mode,
            "source_feature_run_ids": shared.get("source_feature_run_ids", {}),
            "source_intended_receiver_modes": shared.get("source_intended_receiver_modes", {}),
            "source_return_types": shared.get("source_return_types", {}),
            "source_target_families": shared.get("source_target_families", {}),
        }

    v_edge_feature_mode = cli_v_edge_feature_mode(cli_args)
    use_v_edge_features = use_v_edge_features_for_mode(v_edge_feature_mode)
    return {
        "feature_run_id": resolved_feature_run_id,
        "intended_receiver_mode": None,
        "return_type": cli_args.return_type,
        "target_family": cli_args.target_family,
        "graph_schema": {
            "edge_in_dim": 4 if use_v_edge_features else 2,
            "add_v_edge_features": use_v_edge_features,
        },
        "use_v_edge_features": use_v_edge_features,
        "v_edge_feature_mode": v_edge_feature_mode,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-family",
        choices=["goal", "xg", "xt", "goal_distance", "epv"],
        default=None,
        help="Outcome target family for the retained outcome models.",
    )
    parser.add_argument(
        "--return_type",
        default=None,
        help=(
            "Resolved outcome return type to use for label generation: disc_<gamma>, disc_<gamma>_skip1, "
            "next_<N>, next_<N>_skip1, or in_<N> (xt/goal_distance/epv only)."
        ),
    )
    parser.add_argument("--feature-run-id", default=None, help="Pinned feature-artifact run id.")
    parser.add_argument(
        "--diagnostic-feature-run-id",
        default=None,
        help="Feature run containing canonical goal-next10 labels for comparable outcome diagnostics.",
    )
    parser.add_argument("--device", default=None, help="Training device passed to train.py.")
    parser.add_argument(
        "--intended-receiver-mode",
        choices=["original", "angle_only", "model"],
        default=None,
        help="Intended-receiver variant to train against. Not used with --success-intent-only.",
    )
    parser.add_argument(
        "--success-intent-only",
        action="store_true",
        help="Only train the mode-independent success_intent model from observed successful-pass receivers.",
    )
    parser.add_argument(
        "--only-pass-height",
        action="store_true",
        help="Only train the pass_height checkpoint.",
    )
    for task, option_name, enable_help, disable_help in MODEL_TOGGLE_SPECS:
        add_bool_override(parser, option_name, task, enable_help, disable_help)
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override the wrapper batch size for every low-level model training command.",
    )
    for task, option_name, _, _ in MODEL_TOGGLE_SPECS:
        parser.add_argument(
            f"--{option_name}-batch-size",
            dest=f"{task}_batch_size",
            type=int,
            default=None,
            help=f"Override the wrapper batch size for {task}.",
        )
    parser.add_argument(
        "--pass-intent-model-id",
        default=None,
        help="Existing pass_intent checkpoint to use as the pass_success IPW model when --no-pass-intent is set.",
    )
    parser.add_argument(
        "--pass-height-ipw-model-id",
        default=None,
        help="Existing pass_intent checkpoint to use as the pass_height IPW model when --no-pass-intent is set.",
    )
    pass_success_ipw_group = parser.add_mutually_exclusive_group()
    pass_success_ipw_group.add_argument(
        "--pass-success-ipw",
        dest="pass_success_ipw",
        action="store_true",
        help="Use a pass_intent checkpoint for pass_success inverse-propensity weighting.",
    )
    pass_success_ipw_group.add_argument(
        "--no-pass-success-ipw",
        dest="pass_success_ipw",
        action="store_false",
        help="Train pass_success without inverse-propensity weighting.",
    )
    parser.set_defaults(pass_success_ipw=True)
    pass_height_ipw_group = parser.add_mutually_exclusive_group()
    pass_height_ipw_group.add_argument(
        "--pass-height-ipw",
        dest="pass_height_ipw",
        action="store_true",
        help="Use a pass_intent checkpoint for pass_height inverse-propensity weighting.",
    )
    pass_height_ipw_group.add_argument(
        "--no-pass-height-ipw",
        dest="pass_height_ipw",
        action="store_false",
        help="Train pass_height without inverse-propensity weighting.",
    )
    parser.set_defaults(pass_height_ipw=False)
    parser.add_argument(
        "--outcome-scoring-trial",
        type=int,
        default=None,
        help="Optional override for the outcome_scoring checkpoint trial.",
    )
    parser.add_argument(
        "--outcome-conceding-trial",
        type=int,
        default=None,
        help="Optional override for the outcome_conceding checkpoint trial.",
    )
    parser.add_argument("--bundle-id", default=None, help="Optional manifest id for the produced model bundle.")
    edge_feature_group = parser.add_mutually_exclusive_group()
    edge_feature_group.add_argument(
        "--v-edge-features",
        dest="v_edge_feature_mode",
        action="store_const",
        const="all",
        help="Use the stored velocity-angle edge features during training.",
    )
    edge_feature_group.add_argument(
        "--no-v-edge-features",
        dest="v_edge_feature_mode",
        action="store_const",
        const="none",
        help="Ignore the stored velocity-angle edge features during training.",
    )
    edge_feature_group.add_argument(
        "--v-edge-features-no-poss",
        dest="v_edge_feature_mode",
        action="store_const",
        const="no_poss",
        help="Use velocity-angle edge features except on edges incident to the ball possessor.",
    )
    parser.set_defaults(v_edge_feature_mode="all")
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned host memory in low-level DataLoaders.",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory in low-level DataLoaders.",
    )
    parser.set_defaults(pin_memory=False)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop each low-level training run after this many validation-loss misses.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=30,
        help="Minimum epoch before early stopping can terminate a low-level training run.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-5,
        help="Minimum validation-loss improvement passed to train.py.",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Disable validation-loss early stopping in low-level training runs.",
    )
    parser.set_defaults(early_stopping=True)
    add_bool_override(
        parser,
        "xy-only",
        "xy_only",
        "Train with xy-only node features instead of the wrapper default profile.",
        "Disable xy-only node features and use the wrapper default profile instead.",
    )
    add_bool_override(
        parser,
        "possessor-aware",
        "possessor_aware",
        "Include possessor-awareness features during training.",
        "Disable possessor-awareness features during training.",
    )
    add_bool_override(
        parser,
        "keeper-aware",
        "keeper_aware",
        "Include keeper/goal-node awareness features during training.",
        "Disable keeper/goal-node awareness features during training.",
    )
    add_bool_override(
        parser,
        "ball-z-aware",
        "ball_z_aware",
        "Include ball-height features during training.",
        "Disable ball-height features during training.",
    )
    add_bool_override(
        parser,
        "poss-vel-aware",
        "poss_vel_aware",
        "Include the ball possessor's own velocity features during training.",
        "Disable the ball possessor's own velocity features during training.",
    )
    add_bool_override(
        parser,
        "poss-rel-vel-aware",
        "poss_rel_vel_aware",
        "Include player velocity relative to the ball possessor's velocity during training.",
        "Disable player velocity relative to the ball possessor's velocity during training.",
    )
    parser.add_argument(
        "--no-poss-geometry",
        dest="poss_geometry_aware",
        action="store_false",
        default=None,
        help="Disable possessor-relative geometry features while keeping is_possessor.",
    )
    parser.add_argument(
        "--no-goal-features",
        dest="goal_features_aware",
        action="store_false",
        default=None,
        help="Disable goal-relative geometry node features.",
    )
    parser.add_argument(
        "--no-goal-nodes",
        dest="goal_nodes_aware",
        action="store_false",
        default=None,
        help="Remove goal nodes and their incident edges regardless of task defaults.",
    )
    add_bool_override(
        parser,
        "accel",
        "accel_aware",
        "Include player-acceleration node features during training.",
        "Disable player-acceleration node features during training.",
    )
    add_bool_override(
        parser,
        "offside",
        "offside_aware",
        "Include the is_offside node feature during training.",
        "Disable the is_offside node feature during training.",
    )
    add_bool_override(
        parser,
        "extend-features",
        "extend_features",
        "Enable the extended handcrafted node features during training.",
        "Disable the extended handcrafted node features during training.",
    )
    parser.add_argument(
        "--use_physical_xpass",
        "--use-physical-xpass",
        dest="use_physical_xpass",
        action="store_true",
        default=False,
        help="Use precomputed AS-default max player_cum_prob physical xPass for pass_success only.",
    )
    parser.add_argument(
        "--model-variant",
        choices=["gat_baseline", "gat_plus_phys_feature", "gat_phys_logit_offset", "gat_phys_logit_offset_regularized"],
        default="gat_phys_logit_offset",
        help="Physical xPass pass_success architecture variant.",
    )
    parser.add_argument("--physical-cache-dir", default=None, help="Override physical xPass sidecar directory.")
    parser.add_argument("--physical-eps", type=float, default=1e-4, help="Physical probability clamp epsilon.")
    parser.add_argument(
        "--physical-xpass-floor",
        "--physical_xpass_floor",
        dest="physical_xpass_floor",
        type=float,
        default=None,
        help="Optional lower probability floor applied before physical xPass logit conversion.",
    )
    physical_beta1_group = parser.add_mutually_exclusive_group()
    physical_beta1_group.add_argument(
        "--freeze-beta1",
        dest="freeze_beta1",
        action="store_true",
        help="Freeze beta1 at 1.0 for the physical logit offset.",
    )
    physical_beta1_group.add_argument(
        "--learn-physical-scale",
        dest="freeze_beta1",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    physical_beta1_group.add_argument(
        "--fixed-physical-scale",
        dest="freeze_beta1",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(freeze_beta1=False)
    parser.add_argument(
        "--freeze-beta0",
        dest="freeze_beta0",
        action="store_true",
        default=False,
        help="Freeze beta0 at 0.0 for the physical logit offset.",
    )
    parser.add_argument(
        "--residual-regularization-lambda",
        type=float,
        default=0.0,
        help="Optional L2 penalty on the observed-target GAT residual in pass_success.",
    )
    parser.add_argument(
        "--residual-clip-value",
        type=float,
        default=None,
        help="Optional tanh bound for the pass_success residual.",
    )
    parser.add_argument(
        "--residual-distance-threshold",
        type=float,
        default=30.0,
        help="Passer-target distance threshold separating short and long residual controls.",
    )
    parser.add_argument(
        "--short-residual-regularization-lambda",
        type=float,
        default=None,
        help="Optional short-pass override for pass_success residual L2.",
    )
    parser.add_argument(
        "--long-residual-regularization-lambda",
        type=float,
        default=None,
        help="Optional long-pass override for pass_success residual L2.",
    )
    parser.add_argument(
        "--short-residual-clip-value",
        type=float,
        default=None,
        help="Optional short-pass override for pass_success residual clipping.",
    )
    parser.add_argument(
        "--long-residual-clip-value",
        type=float,
        default=None,
        help="Optional long-pass override for pass_success residual clipping.",
    )
    args = parser.parse_args(argv)
    args.learn_physical_scale = not bool(args.freeze_beta1)
    args.v_edge_feature_mode = cli_v_edge_feature_mode(args)
    args.use_v_edge_features = use_v_edge_features_for_mode(args.v_edge_feature_mode)
    args.mask_possessor_v_edge_features = mask_possessor_v_edge_features_for_mode(args.v_edge_feature_mode)
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1.")
    if args.early_stopping_min_epochs < 1:
        parser.error("--early-stopping-min-epochs must be at least 1.")
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta must be non-negative.")
    if not (0.0 < args.physical_eps < 0.5):
        parser.error("--physical-eps must be between 0 and 0.5.")
    if args.physical_xpass_floor is not None and not (0.0 <= args.physical_xpass_floor < 1.0):
        parser.error("--physical-xpass-floor must be in [0.0, 1.0) when provided.")
    if args.residual_distance_threshold <= 0:
        parser.error("--residual-distance-threshold must be positive.")
    if args.residual_regularization_lambda < 0:
        parser.error("--residual-regularization-lambda must be non-negative.")
    for name in ("short_residual_regularization_lambda", "long_residual_regularization_lambda"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    if args.residual_clip_value is not None and args.residual_clip_value <= 0:
        parser.error("--residual-clip-value must be positive when provided.")
    for name in ("short_residual_clip_value", "long_residual_clip_value"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive when provided.")
    validate_batch_size_args(args, parser)
    try:
        resolve_wrapper_feature_flags(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    available_modes = infer_feature_run_intended_receiver_modes(args.feature_run_id)
    available_return_types = infer_feature_run_return_types(args.feature_run_id)

    try:
        args.enabled_tasks = resolve_enabled_tasks(args)
    except ValueError as exc:
        parser.error(str(exc))

    requires_mode = any(args.enabled_tasks.get(task, False) for task in MODE_DEPENDENT_TASKS)
    requires_outcome_config = any(args.enabled_tasks.get(task, False) for task in OUTCOME_TASKS)

    if args.success_intent_only:
        if args.intended_receiver_mode:
            parser.error("--success-intent-only is mode-independent and does not accept --intended-receiver-mode.")
        if args.target_family is not None:
            parser.error("--success-intent-only does not accept --target-family.")
        args.intended_receiver_mode = None
        args.target_family = None
    elif requires_mode:
        if not args.intended_receiver_mode:
            parser.error("--intended-receiver-mode is required when any retained model is enabled.")
        args.intended_receiver_mode = validate_intended_receiver_mode(args.intended_receiver_mode)
    else:
        args.intended_receiver_mode = None

    if requires_outcome_config:
        if not args.target_family:
            parser.error("--target-family is required when outcome models are enabled.")
        if not args.return_type:
            parser.error("--return_type is required when outcome models are enabled.")
        try:
            args.return_type = validate_return_type_for_target_family(args.return_type, target_family=args.target_family)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.target_family = None
        if args.return_type is not None:
            args.return_type = validate_return_type(args.return_type)
        elif available_return_types:
            args.return_type = available_return_types[0]
        else:
            parser.error(f"Feature run {args.feature_run_id} does not expose any return types.")

    if args.intended_receiver_mode is not None and args.intended_receiver_mode not in available_modes:
        parser.error(
            f"Feature run {args.feature_run_id} does not expose intended_receiver_mode={args.intended_receiver_mode!r}. "
            f"Available: {', '.join(available_modes) or 'none'}."
        )
    if args.return_type not in available_return_types:
        parser.error(
            f"Feature run {args.feature_run_id} does not expose return_type={args.return_type!r}. "
            f"Available: {', '.join(available_return_types) or 'none'}."
        )
    validate_diagnostic_feature_run(args, parser, requires_outcome_config=requires_outcome_config)

    args.available_return_types = available_return_types
    args.available_intended_receiver_modes = available_modes
    args.trained_tasks = [task for task, enabled in args.enabled_tasks.items() if enabled]
    return args


def base_gnn_args(
    feature_dir: str,
    label_dir: str,
    model_id: str,
    intended_receiver_mode: str | None,
    return_type: str,
    batch_size: int,
    v_edge_feature_mode: str,
) -> list[str]:
    _, run_id = str(model_id).split("/", 1)
    command = [
        "--run-id",
        run_id,
        "--model",
        "gat",
        "--sparsify",
        "none",
        "--node_emb_dim",
        "128",
        "--graph_emb_dim",
        "128",
        "--mlp_h1_dim",
        "64",
        "--mlp_h2_dim",
        "16",
        "--gnn_layers",
        "2",
        "--gnn_heads",
        "4",
        "--skip_conn",
        "--n_epochs",
        "100",
        "--batch_size",
        str(batch_size),
        "--print_freq",
        "50",
        "--seed",
        "100",
        "--feature_dir",
        feature_dir,
        "--label_dir",
        label_dir,
        "--return_type",
        return_type,
    ]
    if intended_receiver_mode:
        command.extend(["--intended-receiver-mode", intended_receiver_mode])
    return append_edge_feature_flag(command, v_edge_feature_mode)


def intent_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    train_feature_dir: str,
    train_label_dir: str,
    intended_receiver_mode: str,
    return_type: str,
    batch_size: int,
    v_edge_feature_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, batch_size, v_edge_feature_mode),
        "--min_pass_dur",
        "0.5",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
        "--train_feature_dir",
        train_feature_dir,
        "--train_label_dir",
        train_label_dir,
    ]
    return append_low_level_feature_flags(command, feature_flags)


def success_intent_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    return_type: str,
    batch_size: int,
    v_edge_feature_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, None, return_type, batch_size, v_edge_feature_mode),
        "--min_pass_dur",
        "0.5",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
        "--label-source",
        SUCCESS_INTENT_LABEL_SOURCE,
        "--training-filter",
        SUCCESS_INTENT_TRAINING_FILTER,
    ]
    return append_low_level_feature_flags(command, feature_flags)


def pass_success_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    ipw_model_id: str | None,
    intended_receiver_mode: str,
    return_type: str,
    batch_size: int,
    v_edge_feature_mode: str,
    feature_flags: dict[str, bool],
    physical_args: argparse.Namespace | None = None,
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, batch_size, v_edge_feature_mode),
        "--min_pass_dur",
        "0.5",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]
    if ipw_model_id:
        command.extend(["--ipw_model_id", ipw_model_id])
    if physical_args is not None:
        command = append_physical_xpass_flags(command, physical_args)
    return append_low_level_feature_flags(command, feature_flags)


def outcome_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    target_family: str,
    return_type: str,
    intended_receiver_mode: str,
    batch_size: int,
    v_edge_feature_mode: str,
    feature_flags: dict[str, bool],
    diagnostic_feature_run_id: str | None = None,
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, batch_size, v_edge_feature_mode),
        "--lambda_l1",
        "1e-6",
        "--start_lr",
        "0.0002",
        "--min_lr",
        "1e-5",
    ]
    if target_family == "goal_distance":
        command.append("--use_goal_distance")
    elif target_family == "epv":
        command.append("--use_epv")
    elif target_family == "xt":
        command.append("--use_xt")
    elif target_family == "xg":
        command.append("--use_xg")
    if diagnostic_feature_run_id:
        command.extend(["--diagnostic-feature-run-id", str(diagnostic_feature_run_id)])
    return append_low_level_feature_flags(command, feature_flags)


def failure_receiver_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    intended_receiver_mode: str,
    return_type: str,
    batch_size: int,
    v_edge_feature_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, batch_size, v_edge_feature_mode),
        "--augment_blocks",
        "--shot_success",
        "unblocked",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]
    return append_low_level_feature_flags(command, feature_flags)


def build_model_ids(args: argparse.Namespace, enabled_tasks: dict[str, bool]) -> dict[str, str]:
    model_ids = {
        task: f"{task}/{generate_model_run_id(task)}"
        for task, enabled in enabled_tasks.items()
        if enabled
    }
    if enabled_tasks.get("outcome_scoring") and args.outcome_scoring_trial is not None:
        model_ids["outcome_scoring"] = f"outcome_scoring/{int(args.outcome_scoring_trial):02d}"
    if enabled_tasks.get("outcome_conceding") and args.outcome_conceding_trial is not None:
        model_ids["outcome_conceding"] = f"outcome_conceding/{int(args.outcome_conceding_trial):02d}"
    return model_ids


def build_training_commands(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, str], str | None, str | None, dict[str, bool]]:
    mode = args.intended_receiver_mode if any(args.enabled_tasks.get(task, False) for task in MODE_DEPENDENT_TASKS) else None
    target_family = args.target_family if any(args.enabled_tasks.get(task, False) for task in OUTCOME_TASKS) else None
    effective_return_type = args.return_type
    feature_flags = resolve_wrapper_feature_flags(args)
    batch_sizes = resolve_batch_sizes(args, args.enabled_tasks)
    resolved_feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(resolved_feature_run_id)
    model_ids = build_model_ids(args, args.enabled_tasks)
    v_edge_feature_mode = cli_v_edge_feature_mode(args)
    success_intent_feature_dir = str(get_success_intent_graph_dir(feature_root))
    success_intent_label_dir = str(get_success_intent_label_dir(root=feature_root))
    commands = []
    trained_model_ids: dict[str, str] = {}
    pass_success_ipw = bool(getattr(args, "pass_success_ipw", True))
    pass_height_ipw = bool(getattr(args, "pass_height_ipw", False))
    external_pass_intent_model_id = (
        validate_external_pass_intent_model_id(
            args,
            feature_flags,
            resolved_feature_run_id,
        )
        if pass_success_ipw
        else None
    )
    external_pass_height_ipw_model_id = (
        validate_external_pass_intent_model_id(
            args,
            feature_flags,
            resolved_feature_run_id,
            model_id_attr="pass_height_ipw_model_id",
            option_name="--pass-height-ipw-model-id",
            runtime_context="pass_height",
        )
        if pass_height_ipw
        else None
    )

    if args.enabled_tasks.get("success_intent", False):
        commands.append(
            success_intent_command(
                "success_intent",
                model_ids["success_intent"],
                success_intent_feature_dir,
                success_intent_label_dir,
                effective_return_type,
                batch_sizes["success_intent"],
                v_edge_feature_mode,
                feature_flags,
            )
        )
        trained_model_ids["success_intent"] = model_ids["success_intent"]

    if any(args.enabled_tasks.get(task, False) for task in MODE_DEPENDENT_TASKS):
        base_feature_dir = str(get_action_graph_dir(feature_root))
        base_label_dir = str(
            get_action_label_dir(
                effective_return_type,
                intended_receiver_mode=mode,
                root=feature_root,
            )
        )
        intent_train_feature_dir = str(get_action_graph_intent_train_dir(feature_root))
        intent_train_label_dir = str(
            get_intent_train_label_dir(effective_return_type, intended_receiver_mode=mode, root=feature_root)
        )
        augmented_feature_dir = str(get_augmented_feature_dir(intended_receiver_mode=mode, root=feature_root))
        augmented_label_dir = str(get_augmented_label_dir(intended_receiver_mode=mode, root=feature_root))

        if args.enabled_tasks.get("pass_intent", False):
            commands.append(
                intent_command(
                    "pass_intent",
                    model_ids["pass_intent"],
                    base_feature_dir,
                    base_label_dir,
                    intent_train_feature_dir,
                    intent_train_label_dir,
                    mode,
                    effective_return_type,
                    batch_sizes["pass_intent"],
                    v_edge_feature_mode,
                    feature_flags,
                )
            )
            trained_model_ids["pass_intent"] = model_ids["pass_intent"]

        if args.enabled_tasks.get("action_intent", False):
            commands.append(
                intent_command(
                    "action_intent",
                    model_ids["action_intent"],
                    base_feature_dir,
                    base_label_dir,
                    intent_train_feature_dir,
                    intent_train_label_dir,
                    mode,
                    effective_return_type,
                    batch_sizes["action_intent"],
                    v_edge_feature_mode,
                    feature_flags,
                )
            )
            trained_model_ids["action_intent"] = model_ids["action_intent"]

        if args.enabled_tasks.get("pass_success", False):
            ipw_model_id = (
                model_ids["pass_intent"] if args.enabled_tasks.get("pass_intent", False) else external_pass_intent_model_id
            ) if pass_success_ipw else None
            if pass_success_ipw and ipw_model_id is None:
                raise ValueError("--pass-success requires --pass-intent or --pass-intent-model-id.")
            commands.append(
                pass_success_command(
                    "pass_success",
                    model_ids["pass_success"],
                    base_feature_dir,
                    base_label_dir,
                    ipw_model_id,
                    mode,
                    effective_return_type,
                    batch_sizes["pass_success"],
                    v_edge_feature_mode,
                    feature_flags,
                    args,
                )
            )
            trained_model_ids["pass_success"] = model_ids["pass_success"]

        if args.enabled_tasks.get("pass_height", False):
            pass_height_ipw_model_id = (
                model_ids["pass_intent"] if args.enabled_tasks.get("pass_intent", False) else external_pass_height_ipw_model_id
            ) if pass_height_ipw else None
            if pass_height_ipw and pass_height_ipw_model_id is None:
                raise ValueError("--pass-height-ipw requires --pass-intent or --pass-height-ipw-model-id.")
            commands.append(
                pass_success_command(
                    "pass_height",
                    model_ids["pass_height"],
                    base_feature_dir,
                    base_label_dir,
                    pass_height_ipw_model_id,
                    mode,
                    effective_return_type,
                    batch_sizes["pass_height"],
                    v_edge_feature_mode,
                    feature_flags,
                )
            )
            trained_model_ids["pass_height"] = model_ids["pass_height"]

        if args.enabled_tasks.get("outcome_scoring", False):
            commands.append(
                outcome_command(
                    "outcome_scoring",
                    model_ids["outcome_scoring"],
                    base_feature_dir,
                    base_label_dir,
                    target_family,
                    effective_return_type,
                    mode,
                    batch_sizes["outcome_scoring"],
                    v_edge_feature_mode,
                    feature_flags,
                    getattr(args, "diagnostic_feature_run_id", None),
                )
            )
            trained_model_ids["outcome_scoring"] = model_ids["outcome_scoring"]

        if args.enabled_tasks.get("outcome_conceding", False):
            commands.append(
                outcome_command(
                    "outcome_conceding",
                    model_ids["outcome_conceding"],
                    base_feature_dir,
                    base_label_dir,
                    target_family,
                    effective_return_type,
                    mode,
                    batch_sizes["outcome_conceding"],
                    v_edge_feature_mode,
                    feature_flags,
                    getattr(args, "diagnostic_feature_run_id", None),
                )
            )
            trained_model_ids["outcome_conceding"] = model_ids["outcome_conceding"]

        if args.enabled_tasks.get("failure_receiver", False):
            commands.append(
                failure_receiver_command(
                    "failure_receiver",
                    model_ids["failure_receiver"],
                    augmented_feature_dir,
                    augmented_label_dir,
                    mode,
                    effective_return_type,
                    batch_sizes["failure_receiver"],
                    v_edge_feature_mode,
                    feature_flags,
                )
            )
            trained_model_ids["failure_receiver"] = model_ids["failure_receiver"]

    return (
        commands,
        trained_model_ids,
        mode,
        resolved_feature_run_id,
        feature_flags,
    )


def main() -> None:
    cli_args = parse_args()
    v_edge_feature_mode = cli_v_edge_feature_mode(cli_args)
    python = sys.executable
    bundle_id = cli_args.bundle_id or generate_run_id("model_bundle")
    bundle_root = get_model_bundle_root(bundle_id)
    commands, trained_model_ids, intended_receiver_mode, resolved_feature_run_id, feature_flags = build_training_commands(
        cli_args
    )
    executed_commands: list[list[str]] = []
    completed_model_ids: dict[str, str] = {}
    existing_bundle = load_model_bundle_metadata(bundle_id, required=False) or {}
    training_control_settings = get_training_control_settings(cli_args)
    explicit_source_model_ids = source_model_ids(cli_args)
    wrapper_diagnostic_metadata = diagnostic_metadata(
        cli_args,
        intended_receiver_mode,
        outcome_enabled=any(task in trained_model_ids for task in OUTCOME_TASKS),
        feature_run_id_fallback=resolved_feature_run_id,
    )
    trained_batch_sizes = resolve_batch_sizes(
        cli_args,
        {task: task in trained_model_ids for task in MODEL_TOGGLE_DEFAULTS},
    )

    total_commands = len(commands)
    for index, args in enumerate(commands, start=1):
        command = [python, "train.py"]
        if resolved_feature_run_id:
            command.extend(["--feature-run-id", str(resolved_feature_run_id)])
        command.extend(args)
        command = append_runtime_flags(
            command,
            device=getattr(cli_args, "device", None),
            pin_memory=getattr(cli_args, "pin_memory", None),
        )
        command = append_training_control_flags(command, training_control_settings)
        command.extend(["--training-step-index", str(index), "--training-step-total", str(total_commands)])
        print("Running:", " ".join(command))
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            failed_task = get_cli_value(command, "--task")
            failed_run_id = get_cli_value(command, "--run-id")
            failed_log = str(get_model_run_root(failed_task, failed_run_id) / "log.txt") if failed_task and failed_run_id else None
            failed_crash_log = (
                str(get_model_run_root(failed_task, failed_run_id) / "crash.log") if failed_task and failed_run_id else None
            )
            timestamp = datetime.now().isoformat(timespec="seconds")
            failure_metadata = {
                "bundle_id": bundle_id,
                "created_at": existing_bundle.get("created_at", timestamp),
                "updated_at": timestamp,
                "command": subprocess.list2cmdline(sys.argv),
                "feature_run_id": resolved_feature_run_id,
                "intended_receiver_mode": intended_receiver_mode,
                "return_type": cli_args.return_type,
                "target_family": cli_args.target_family,
                **wrapper_diagnostic_metadata,
                "use_v_edge_features": use_v_edge_features_for_mode(v_edge_feature_mode),
                "v_edge_feature_mode": v_edge_feature_mode,
                "device": getattr(cli_args, "device", None),
                "pin_memory": bool(getattr(cli_args, "pin_memory", False)),
                **training_control_settings,
                "physical_xpass": physical_xpass_settings(cli_args),
                "pass_success_ipw": bool(getattr(cli_args, "pass_success_ipw", True)),
                "pass_height_ipw": bool(getattr(cli_args, "pass_height_ipw", False)),
                "training_feature_flags": feature_flags,
                "batch_sizes": trained_batch_sizes,
                "success_intent_only": bool(cli_args.success_intent_only),
                "only_pass_height": bool(getattr(cli_args, "only_pass_height", False)),
                "planned_tasks": list(trained_model_ids.keys()),
                "completed_tasks": list(completed_model_ids.keys()),
                "completed_model_ids": completed_model_ids,
                "model_ids": {
                    **dict(existing_bundle.get("model_ids", {})),
                    **explicit_source_model_ids,
                    **completed_model_ids,
                },
                "commands": executed_commands,
                "failed_command": command,
                "failed_task": failed_task,
                "failed_model_id": f"{failed_task}/{failed_run_id}" if failed_task and failed_run_id else None,
                "failed_log": failed_log,
                "failed_crash_log": failed_crash_log,
                "returncode": int(exc.returncode),
                "returncode_description": describe_returncode(int(exc.returncode)),
                "error": str(exc),
                "status": "failed",
            }
            if explicit_source_model_ids:
                failure_metadata["source_model_ids"] = explicit_source_model_ids
            write_run_metadata(bundle_root, failure_metadata)
            print(f"Training failed for {failure_metadata['failed_model_id'] or failed_task or 'unknown task'}.")
            print(f"Return code: {failure_metadata['returncode_description']}")
            if failed_log:
                print(f"Model log: {failed_log}")
            if failed_crash_log:
                print(f"Crash log: {failed_crash_log}")
            print(f"Model bundle manifest: {bundle_root / 'metadata.json'}")
            raise
        executed_commands.append(command)
        task = get_cli_value(command, "--task")
        if task and task in trained_model_ids:
            completed_model_ids[task] = trained_model_ids[task]

    final_model_ids = dict(existing_bundle.get("model_ids", {}))
    final_model_ids.update(explicit_source_model_ids)
    final_model_ids.update(trained_model_ids)

    bundle_shared = derive_bundle_shared_context(final_model_ids, cli_args, resolved_feature_run_id)
    effective_feature_run_id = str(bundle_shared.get("feature_run_id")) if bundle_shared.get("feature_run_id") else None
    feature_run_metadata = load_feature_run_metadata(effective_feature_run_id, required=False) or {} if effective_feature_run_id else {}
    timestamp = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "bundle_id": bundle_id,
        "created_at": existing_bundle.get("created_at", timestamp),
        "updated_at": timestamp,
        "command": subprocess.list2cmdline(sys.argv),
        "feature_run_id": effective_feature_run_id,
        "intended_receiver_mode": bundle_shared.get("intended_receiver_mode"),
        "feature_run_intended_receiver_modes": feature_run_metadata.get(
            "intended_receiver_modes",
            cli_args.available_intended_receiver_modes,
        ),
        "feature_run_return_types": feature_run_metadata.get("return_types", cli_args.available_return_types),
        "feature_run_intended_receiver_model_id": feature_run_metadata.get("intended_receiver_model_id"),
        "training_feature_flags": feature_flags,
        "batch_sizes": trained_batch_sizes,
        "physical_xpass": physical_xpass_settings(cli_args),
        "pass_success_ipw": bool(getattr(cli_args, "pass_success_ipw", True)),
        "pass_height_ipw": bool(getattr(cli_args, "pass_height_ipw", False)),
        "device": getattr(cli_args, "device", None),
        "pin_memory": bool(getattr(cli_args, "pin_memory", False)),
        **training_control_settings,
        "target_family": bundle_shared.get("target_family"),
        "return_type": bundle_shared.get("return_type"),
        "source_feature_run_ids": bundle_shared.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": bundle_shared.get("source_intended_receiver_modes", {}),
        "source_return_types": bundle_shared.get("source_return_types", {}),
        "source_target_families": bundle_shared.get("source_target_families", {}),
        **wrapper_diagnostic_metadata,
        "use_v_edge_features": bool(bundle_shared.get("use_v_edge_features", use_v_edge_features_for_mode(v_edge_feature_mode))),
        "v_edge_feature_mode": bundle_shared.get("v_edge_feature_mode", v_edge_feature_mode),
        "graph_schema": dict(bundle_shared.get("graph_schema", {})),
        "success_intent_only": bool(cli_args.success_intent_only),
        "only_pass_height": bool(getattr(cli_args, "only_pass_height", False)),
        "trained_tasks": list(trained_model_ids.keys()),
        "success_intent_label_source": (
            SUCCESS_INTENT_LABEL_SOURCE
            if "success_intent" in trained_model_ids
            else existing_bundle.get("success_intent_label_source")
        ),
        "success_intent_training_filter": (
            SUCCESS_INTENT_TRAINING_FILTER
            if "success_intent" in trained_model_ids
            else existing_bundle.get("success_intent_training_filter")
        ),
        "model_ids": final_model_ids,
        "commands": executed_commands,
        "status": "completed",
    }
    if explicit_source_model_ids:
        metadata["source_model_ids"] = explicit_source_model_ids
    write_run_metadata(bundle_root, metadata)
    print(f"Model bundle id: {bundle_id}")
    print(f"Model bundle manifest: {bundle_root / 'metadata.json'}")
    for task, model_id in final_model_ids.items():
        print(f"{task}: {model_id}")


if __name__ == "__main__":
    main()
