from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datatools import config
from datatools.graph_feature import infer_node_feature_dim
from project_config import (
    INTENDED_RECEIVER_MODE_MODEL,
    generate_run_id,
    get_action_label_dir,
    get_action_graph_dir,
    get_action_graph_intent_train_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_feature_run_root,
    get_intent_train_label_dir,
    get_post_action_graph_dir,
    get_resolved_action_dir,
    get_success_intent_graph_dir,
    load_feature_run_metadata,
    resolve_generation_intended_receiver_modes,
    resolve_requested_return_types,
    resolve_feature_run_id,
    validate_intended_receiver_mode,
    write_latest_run,
    write_run_metadata,
)

EXPECTED_GRAPH_SCHEMA = {
    "node_in_dim": infer_node_feature_dim(extend=True),
    "edge_in_dim": 2,
    "add_v_edge_features": False,
    "add_relative_speed_edge_features": False,
}
EXPECTED_EDGE_GRAPH_SCHEMA = {"edge_in_dim": 2, "add_v_edge_features": False, "add_relative_speed_edge_features": False}
BASE_EDGE_GRAPH_SCHEMA = {"edge_in_dim": 2, "add_v_edge_features": False, "add_relative_speed_edge_features": False}
ALIGNMENT_EDGE_GRAPH_SCHEMA = {"edge_in_dim": 4, "add_v_edge_features": True, "add_relative_speed_edge_features": False}
RELATIVE_SPEED_EDGE_GRAPH_SCHEMA = {"edge_in_dim": 5, "add_v_edge_features": True, "add_relative_speed_edge_features": True}
SUPPORTED_EDGE_GRAPH_SCHEMAS = (BASE_EDGE_GRAPH_SCHEMA, ALIGNMENT_EDGE_GRAPH_SCHEMA, RELATIVE_SPEED_EDGE_GRAPH_SCHEMA)
REFRESH_TARGET_FAMILIES = ("xt", "goal_distance", "epv")
EXTENSION_MODE_DERIVED = "derived"
EXTENSION_MODE_IN_PLACE = "in_place"
EXTENSION_MODE_OVERWRITE_FEATURE_RUN = "overwrite_feature_run"


@dataclass(frozen=True)
class FeatureGenerationStep:
    description: str
    command: list[str]


@dataclass(frozen=True)
class FeatureExtensionPlan:
    base_run_id: str
    output_run_id: str
    target_run_id: str
    extension_mode: str
    base_metadata: dict[str, Any]
    base_return_types: list[str]
    final_return_types: list[str]
    added_return_types: list[str]
    base_intended_receiver_modes: list[str]
    final_intended_receiver_modes: list[str]
    added_intended_receiver_modes: list[str]
    refresh_target_families: list[str]
    refresh_pass_height_labels: bool
    extend_relative_speed_edge_features: bool
    refreshed_return_types: list[str]
    refreshed_intended_receiver_modes: list[str]
    intended_receiver_model_id: str | None
    regenerate_model_mode: bool
    replaced_intended_receiver_model_id: str | None
    replaced_intended_receiver_modes: list[str]
    next_action_conditions_enabled: bool
    graph_schema: dict[str, Any]
    commands: list[list[str]]
    command_steps: list[FeatureGenerationStep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--return_type",
        action="append",
        default=None,
        help=(
            "Resolved return type for generated action labels. Repeat the flag to include multiple return types, "
            "including disc_<gamma>_skip1 and next_<N>_skip1, plus in_<N> for xt/goal_distance/epv training."
        ),
    )
    parser.add_argument(
        "--intended-receiver-model-id",
        default=None,
        help="Optional success_intent checkpoint used to add the model-backed intended-receiver variant.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--extend-feature-run-id",
        default=None,
        help=(
            "Create a new derived feature run from this completed feature run, copying existing artifacts and "
            "generating only newly requested return types, refreshed target labels, or the model intended-receiver "
            "variant."
        ),
    )
    parser.add_argument(
        "--refresh-target-family",
        action="append",
        choices=REFRESH_TARGET_FAMILIES,
        default=None,
        help=(
            "With --extend-feature-run-id, rebuild copied label tensors from current sidecar target artifacts "
            "without rebuilding graph tensors. Repeat to record multiple refreshed target families."
        ),
    )
    parser.add_argument(
        "--pass-height",
        action="store_true",
        help=(
            "Enable pass-height label generation/configuration. With --extend-feature-run-id, rebuild copied "
            "label tensors so pass-height labels are present. "
            f"High passes use ball_z >= {config.PASS_HEIGHT_THRESHOLD_METERS:g}m between pass and receipt."
        ),
    )
    parser.add_argument(
        "--pass-height-threshold",
        type=float,
        default=None,
        help=(
            "Maximum-ball-height cutoff in metres used to classify a pass as high. Requires --pass-height; "
            f"defaults to {config.PASS_HEIGHT_THRESHOLD_METERS:g}."
        ),
    )
    parser.add_argument(
        "--extend-relative-speed-edge-features",
        action="store_true",
        default=False,
        help=(
            "With --extend-feature-run-id, append raw relative-speed edge features to existing 4-column "
            "velocity-angle graph tensors."
        ),
    )
    parser.add_argument(
        "--replace-intended-receiver-model",
        action="store_true",
        default=False,
        help=(
            "When extending a feature run that already contains the model intended-receiver variant, create a "
            "new derived run and regenerate only model-mode artifacts with --intended-receiver-model-id."
        ),
    )
    extension_mode_group = parser.add_mutually_exclusive_group()
    extension_mode_group.add_argument(
        "--in-place",
        action="store_true",
        default=False,
        help=(
            "With --extend-feature-run-id, add only missing return-type or intended-receiver-mode artifacts "
            "directly to the existing feature run instead of copying it."
        ),
    )
    extension_mode_group.add_argument(
        "--overwrite-feature-run",
        action="store_true",
        default=False,
        help=(
            "With --extend-feature-run-id, mutate the existing feature run and allow copied labels/model-mode "
            "artifacts to be overwritten or regenerated."
        ),
    )
    edge_feature_group = parser.add_mutually_exclusive_group()
    edge_feature_group.add_argument(
        "--v-edge-features",
        dest="add_v_edge_features",
        action="store_true",
        help="Generate velocity-angle edge features.",
    )
    edge_feature_group.add_argument(
        "--no-v-edge-features",
        dest="add_v_edge_features",
        action="store_false",
        help="Generate only base edge features.",
    )
    parser.set_defaults(add_v_edge_features=False)
    relative_speed_edge_feature_group = parser.add_mutually_exclusive_group()
    relative_speed_edge_feature_group.add_argument(
        "--relative-speed-edge-features",
        dest="add_relative_speed_edge_features",
        action="store_true",
        help="Generate raw relative-speed edge features after velocity-angle edge features.",
    )
    relative_speed_edge_feature_group.add_argument(
        "--no-relative-speed-edge-features",
        dest="add_relative_speed_edge_features",
        action="store_false",
        help="Do not generate raw relative-speed edge features.",
    )
    parser.set_defaults(add_relative_speed_edge_features=False)
    next_action_group = parser.add_mutually_exclusive_group()
    next_action_group.add_argument(
        "--next-action-conditions-on",
        dest="next_action_conditions_enabled",
        action="store_true",
        default=True,
        help="Keep the current pass/cross next-action inclusion conditions enabled.",
    )
    next_action_group.add_argument(
        "--next-action-conditions-off",
        dest="next_action_conditions_enabled",
        action="store_false",
        help="Disable pass/cross next-action inclusion conditions while keeping frame requirements.",
    )
    parser.add_argument(
        "--num-workers",
        default="1",
        help="Number of match worker processes to use inside each graph_feature step, or 'auto'.",
    )
    parser.add_argument(
        "--worker-thread-limit",
        type=int,
        default=1,
        help="Thread limit passed to each graph_feature worker process.",
    )
    args = parser.parse_args()
    args.requested_return_types = resolve_requested_return_types(args.return_type) if args.return_type else []
    args.return_types = args.requested_return_types or resolve_requested_return_types(None)
    args.intended_receiver_modes = resolve_generation_intended_receiver_modes(args.intended_receiver_model_id)
    args.refresh_target_families = normalize_refresh_target_families(args.refresh_target_family)
    if args.refresh_target_families and not args.extend_feature_run_id:
        parser.error("--refresh-target-family requires --extend-feature-run-id.")
    if args.pass_height_threshold is not None and not args.pass_height:
        parser.error("--pass-height-threshold requires --pass-height.")
    if args.extend_relative_speed_edge_features and not args.extend_feature_run_id:
        parser.error("--extend-relative-speed-edge-features requires --extend-feature-run-id.")
    if (args.in_place or args.overwrite_feature_run) and not args.extend_feature_run_id:
        parser.error("--in-place and --overwrite-feature-run require --extend-feature-run-id.")
    if (args.in_place or args.overwrite_feature_run) and args.run_id:
        parser.error("--run-id cannot be used with --in-place or --overwrite-feature-run.")
    if args.in_place and args.refresh_target_families:
        parser.error("--in-place only supports additive extensions; use --overwrite-feature-run for target refreshes.")
    if args.in_place and args.pass_height:
        parser.error("--in-place only supports additive extensions; use --overwrite-feature-run for pass-height refreshes.")
    if args.in_place and args.replace_intended_receiver_model:
        parser.error(
            "--in-place only supports additive extensions; use --overwrite-feature-run to replace model-mode artifacts."
        )
    if args.add_relative_speed_edge_features and not args.add_v_edge_features:
        parser.error("--relative-speed-edge-features requires --v-edge-features.")
    if args.worker_thread_limit < 1:
        parser.error("--worker-thread-limit must be a positive integer.")
    return args


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def run_generation_steps(steps: list[FeatureGenerationStep]) -> None:
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f"Feature generation step {index}/{total}: {step.description}")
        run_command(step.command)


def with_mode_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    for return_type in args.return_types:
        command.extend(["--return_type", return_type])
    if args.intended_receiver_model_id:
        command.extend(["--intended-receiver-model-id", args.intended_receiver_model_id])
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    command.extend(["--num-workers", str(getattr(args, "num_workers", "1"))])
    command.extend(["--worker-thread-limit", str(getattr(args, "worker_thread_limit", 1))])
    command.append(next_action_conditions_flag(args.next_action_conditions_enabled))
    command.append("--v-edge-features" if bool(getattr(args, "add_v_edge_features", False)) else "--no-v-edge-features")
    command.append(
        "--relative-speed-edge-features"
        if bool(getattr(args, "add_relative_speed_edge_features", False))
        else "--no-relative-speed-edge-features"
    )
    if bool(getattr(args, "pass_height", False)):
        command.extend(["--pass-height-threshold", str(pass_height_threshold_meters(args))])
    return command


def pass_height_threshold_meters(args: argparse.Namespace) -> float:
    value = getattr(args, "pass_height_threshold", None)
    return config.PASS_HEIGHT_THRESHOLD_METERS if value is None else float(value)


def next_action_conditions_flag(enabled: bool) -> str:
    return "--next-action-conditions-on" if enabled else "--next-action-conditions-off"


def full_generation_commands(python: str) -> list[FeatureGenerationStep]:
    return [
        FeatureGenerationStep(
            "train split with post_action + augment_blocks",
            [
                python,
                "datatools/graph_feature.py",
                "--action_type",
                "all",
                "--split",
                "train",
                "--post_action",
                "--augment_blocks",
            ],
        ),
        FeatureGenerationStep(
            "test split with post_action",
            [
                python,
                "datatools/graph_feature.py",
                "--action_type",
                "all",
                "--split",
                "test",
                "--post_action",
            ],
        ),
        FeatureGenerationStep(
            "train split with intent_train_augmented",
            [
                python,
                "datatools/graph_feature.py",
                "--action_type",
                "all",
                "--split",
                "train",
                "--feature_variant",
                "intent_train_augmented",
            ],
        ),
        FeatureGenerationStep(
            "train split with success_intent",
            [
                python,
                "datatools/graph_feature.py",
                "--action_type",
                "all",
                "--split",
                "train",
                "--feature_variant",
                "success_intent",
            ],
        ),
        FeatureGenerationStep(
            "test split with success_intent",
            [
                python,
                "datatools/graph_feature.py",
                "--action_type",
                "all",
                "--split",
                "test",
                "--feature_variant",
                "success_intent",
            ],
        ),
    ]


def normalize_graph_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    normalized = {
        "edge_in_dim": int(schema.get("edge_in_dim", -1)),
        "add_v_edge_features": bool(schema.get("add_v_edge_features", False)),
        "add_relative_speed_edge_features": bool(schema.get("add_relative_speed_edge_features", False)),
    }
    if schema.get("node_in_dim") is not None:
        normalized["node_in_dim"] = int(schema["node_in_dim"])
    return normalized


def normalize_edge_graph_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_graph_schema(schema)
    return {
        "edge_in_dim": int(normalized.get("edge_in_dim", -1)),
        "add_v_edge_features": bool(normalized.get("add_v_edge_features", False)),
        "add_relative_speed_edge_features": bool(normalized.get("add_relative_speed_edge_features", False)),
    }


def is_supported_edge_graph_schema(schema: dict[str, Any]) -> bool:
    normalized = normalize_edge_graph_schema(schema)
    return any(normalized == expected for expected in SUPPORTED_EDGE_GRAPH_SCHEMAS)


def graph_schema_from_args(args: argparse.Namespace) -> dict[str, Any]:
    add_v_edge_features = bool(getattr(args, "add_v_edge_features", False))
    add_relative_speed_edge_features = bool(getattr(args, "add_relative_speed_edge_features", False))
    edge_in_dim = 5 if add_relative_speed_edge_features else 4 if add_v_edge_features else 2
    return {
        "node_in_dim": infer_node_feature_dim(extend=True),
        "edge_in_dim": edge_in_dim,
        "add_v_edge_features": add_v_edge_features,
        "add_relative_speed_edge_features": add_relative_speed_edge_features,
    }


def edge_feature_flags_for_schema(schema: dict[str, Any]) -> list[str]:
    normalized = normalize_edge_graph_schema(schema)
    flags = ["--v-edge-features" if normalized["add_v_edge_features"] else "--no-v-edge-features"]
    flags.append(
        "--relative-speed-edge-features"
        if normalized["add_relative_speed_edge_features"]
        else "--no-relative-speed-edge-features"
    )
    return flags


def with_edge_schema_flags(command: list[str], schema: dict[str, Any]) -> list[str]:
    return [*command, *edge_feature_flags_for_schema(schema)]


def metadata_return_types(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("return_types")
    if isinstance(values, list) and values:
        return resolve_requested_return_types(values)
    legacy_value = metadata.get("return_type")
    if legacy_value:
        return resolve_requested_return_types([str(legacy_value)])
    raise ValueError("Base feature run metadata does not record any return_types.")


def metadata_next_action_conditions_enabled(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("next_action_conditions_enabled", True))


def metadata_intended_receiver_modes(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("intended_receiver_modes")
    if not isinstance(values, list) or not values:
        raise ValueError("Base feature run metadata does not record intended_receiver_modes.")

    modes: list[str] = []
    seen: set[str] = set()
    for value in values:
        mode = validate_intended_receiver_mode(str(value))
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes


def normalize_refresh_target_families(values: list[str] | None) -> list[str]:
    if not values:
        return []

    families: list[str] = []
    seen: set[str] = set()
    for value in values:
        family = str(value)
        if family not in REFRESH_TARGET_FAMILIES:
            raise ValueError(
                f"Unsupported refresh target family {family!r}. Expected one of: "
                f"{', '.join(REFRESH_TARGET_FAMILIES)}."
            )
        if family in seen:
            continue
        seen.add(family)
        families.append(family)
    return families


def args_refresh_target_families(args: argparse.Namespace) -> list[str]:
    values = getattr(args, "refresh_target_families", None)
    if values is None:
        values = getattr(args, "refresh_target_family", None)
    return normalize_refresh_target_families(values)


def union_preserving_order(base_values: list[str], requested_values: list[str]) -> tuple[list[str], list[str]]:
    final_values = list(base_values)
    seen = set(final_values)
    added_values: list[str] = []
    for value in requested_values:
        if value in seen:
            continue
        seen.add(value)
        final_values.append(value)
        added_values.append(value)
    return final_values, added_values


def extension_graph_feature_command(
    python: str,
    output_run_id: str,
    *,
    split: str,
    return_types: list[str],
    intended_receiver_modes: list[str],
    intended_receiver_model_id: str | None,
    feature_variant: str | None = None,
    intent_train_label_source_mode: str | None = None,
    intent_train_label_source_return_type: str | None = None,
    augment_blocks_from_existing_graphs: bool = False,
    overwrite_labels: bool = False,
    next_action_conditions_enabled: bool = True,
    num_workers: str | int = "1",
    worker_thread_limit: int = 1,
) -> list[str]:
    command = [
        python,
        "datatools/graph_feature.py",
        "--action_type",
        "all",
        "--split",
        split,
        "--labels-only",
        "--run-id",
        output_run_id,
        "--num-workers",
        str(num_workers),
        "--worker-thread-limit",
        str(worker_thread_limit),
        next_action_conditions_flag(next_action_conditions_enabled),
    ]
    if feature_variant:
        command.extend(["--feature_variant", feature_variant])
    for return_type in return_types:
        command.extend(["--return_type", return_type])
    for mode in intended_receiver_modes:
        command.extend(["--only-intended-receiver-mode", mode])
    if intended_receiver_model_id and INTENDED_RECEIVER_MODE_MODEL in intended_receiver_modes:
        command.extend(["--intended-receiver-model-id", intended_receiver_model_id])
    if intent_train_label_source_mode:
        command.extend(["--intent-train-label-source-mode", intent_train_label_source_mode])
    if intent_train_label_source_return_type:
        command.extend(["--intent-train-label-source-return-type", intent_train_label_source_return_type])
    if augment_blocks_from_existing_graphs:
        command.append("--augment-blocks-from-existing-graphs")
    if overwrite_labels:
        command.append("--overwrite-labels")
    return command


def extension_commands_for_plan(
    *,
    python: str,
    output_run_id: str,
    base_return_types: list[str],
    final_return_types: list[str],
    added_return_types: list[str],
    base_modes: list[str],
    added_modes: list[str],
    intended_receiver_model_id: str | None,
    regenerate_model_mode: bool = False,
    refresh_existing_labels: bool = False,
    final_modes: list[str] | None = None,
    next_action_conditions_enabled: bool = True,
    num_workers: str | int = "1",
    worker_thread_limit: int = 1,
) -> list[FeatureGenerationStep]:
    steps: list[FeatureGenerationStep] = []
    final_modes = final_modes if final_modes is not None else [*base_modes, *added_modes]
    non_model_base_modes = [mode for mode in base_modes if mode != INTENDED_RECEIVER_MODE_MODEL]
    refreshes_model_mode = refresh_existing_labels and INTENDED_RECEIVER_MODE_MODEL in final_modes
    if (regenerate_model_mode or refreshes_model_mode) and not non_model_base_modes:
        raise ValueError("Regenerating model-mode artifacts requires at least one non-model base mode.")

    source_mode = non_model_base_modes[0] if non_model_base_modes else base_modes[0]
    source_return_type = base_return_types[0]

    if refresh_existing_labels:
        non_model_final_modes = [mode for mode in final_modes if mode != INTENDED_RECEIVER_MODE_MODEL]
        if non_model_final_modes:
            for split in ["train", "test"]:
                steps.append(
                    FeatureGenerationStep(
                        f"{split} split target-label refresh (labels-only)",
                        extension_graph_feature_command(
                            python,
                            output_run_id,
                            split=split,
                            return_types=final_return_types,
                            intended_receiver_modes=non_model_final_modes,
                            intended_receiver_model_id=intended_receiver_model_id,
                            overwrite_labels=True,
                            next_action_conditions_enabled=next_action_conditions_enabled,
                            num_workers=num_workers,
                            worker_thread_limit=worker_thread_limit,
                        ),
                    )
                )
            steps.append(
                FeatureGenerationStep(
                    "train split target-label refresh with intent_train_augmented (labels-only)",
                    extension_graph_feature_command(
                        python,
                        output_run_id,
                        split="train",
                        return_types=final_return_types,
                        intended_receiver_modes=non_model_final_modes,
                        intended_receiver_model_id=intended_receiver_model_id,
                        feature_variant="intent_train_augmented",
                        intent_train_label_source_mode=source_mode,
                        intent_train_label_source_return_type=source_return_type,
                        overwrite_labels=True,
                        next_action_conditions_enabled=next_action_conditions_enabled,
                        num_workers=num_workers,
                        worker_thread_limit=worker_thread_limit,
                    ),
                )
            )

        if INTENDED_RECEIVER_MODE_MODEL in final_modes:
            validation_modes = [source_mode, INTENDED_RECEIVER_MODE_MODEL]
            should_generate_model_augmented = INTENDED_RECEIVER_MODE_MODEL in added_modes or regenerate_model_mode
            for split in ["train", "test"]:
                steps.append(
                    FeatureGenerationStep(
                        f"{split} split target-label refresh with model mode (labels-only)",
                        extension_graph_feature_command(
                            python,
                            output_run_id,
                            split=split,
                            return_types=final_return_types,
                            intended_receiver_modes=validation_modes,
                            intended_receiver_model_id=intended_receiver_model_id,
                            augment_blocks_from_existing_graphs=should_generate_model_augmented and split == "train",
                            overwrite_labels=True,
                            next_action_conditions_enabled=next_action_conditions_enabled,
                            num_workers=num_workers,
                            worker_thread_limit=worker_thread_limit,
                        ),
                    )
                )
            steps.append(
                FeatureGenerationStep(
                    "train split target-label refresh with model mode intent_train_augmented (labels-only)",
                    extension_graph_feature_command(
                        python,
                        output_run_id,
                        split="train",
                        return_types=final_return_types,
                        intended_receiver_modes=[INTENDED_RECEIVER_MODE_MODEL],
                        intended_receiver_model_id=intended_receiver_model_id,
                        feature_variant="intent_train_augmented",
                        intent_train_label_source_mode=source_mode,
                        intent_train_label_source_return_type=source_return_type,
                        overwrite_labels=True,
                        next_action_conditions_enabled=next_action_conditions_enabled,
                        num_workers=num_workers,
                        worker_thread_limit=worker_thread_limit,
                    ),
                )
            )
        return steps

    if added_return_types:
        added_return_modes = non_model_base_modes if regenerate_model_mode else base_modes
        for split in ["train", "test"]:
            steps.append(
                FeatureGenerationStep(
                    f"{split} split (labels-only)",
                    extension_graph_feature_command(
                        python,
                        output_run_id,
                        split=split,
                        return_types=added_return_types,
                        intended_receiver_modes=added_return_modes,
                        intended_receiver_model_id=intended_receiver_model_id,
                        next_action_conditions_enabled=next_action_conditions_enabled,
                        num_workers=num_workers,
                        worker_thread_limit=worker_thread_limit,
                    ),
                )
            )
        steps.append(
            FeatureGenerationStep(
                "train split with intent_train_augmented (labels-only)",
                extension_graph_feature_command(
                    python,
                    output_run_id,
                    split="train",
                    return_types=added_return_types,
                    intended_receiver_modes=added_return_modes,
                    intended_receiver_model_id=intended_receiver_model_id,
                    feature_variant="intent_train_augmented",
                    intent_train_label_source_mode=source_mode,
                    intent_train_label_source_return_type=source_return_type,
                    next_action_conditions_enabled=next_action_conditions_enabled,
                    num_workers=num_workers,
                    worker_thread_limit=worker_thread_limit,
                ),
            )
        )

    if INTENDED_RECEIVER_MODE_MODEL in added_modes or regenerate_model_mode:
        validation_modes = [source_mode, INTENDED_RECEIVER_MODE_MODEL]
        for split in ["train", "test"]:
            steps.append(
                FeatureGenerationStep(
                    f"{split} split with model mode (labels-only)",
                    extension_graph_feature_command(
                        python,
                        output_run_id,
                        split=split,
                        return_types=final_return_types,
                        intended_receiver_modes=validation_modes,
                        intended_receiver_model_id=intended_receiver_model_id,
                        augment_blocks_from_existing_graphs=(split == "train"),
                        next_action_conditions_enabled=next_action_conditions_enabled,
                        num_workers=num_workers,
                        worker_thread_limit=worker_thread_limit,
                    ),
                )
            )
        steps.append(
            FeatureGenerationStep(
                "train split with model mode intent_train_augmented (labels-only)",
                extension_graph_feature_command(
                    python,
                    output_run_id,
                    split="train",
                    return_types=final_return_types,
                    intended_receiver_modes=[INTENDED_RECEIVER_MODE_MODEL],
                    intended_receiver_model_id=intended_receiver_model_id,
                    feature_variant="intent_train_augmented",
                    intent_train_label_source_mode=source_mode,
                    intent_train_label_source_return_type=source_return_type,
                    next_action_conditions_enabled=next_action_conditions_enabled,
                    num_workers=num_workers,
                    worker_thread_limit=worker_thread_limit,
                ),
            )
        )
    return steps


def build_extension_plan(args: argparse.Namespace, python: str | None = None) -> FeatureExtensionPlan:
    python = python or sys.executable
    base_run_id = resolve_feature_run_id(args.extend_feature_run_id, required=True, allow_latest=False)
    if base_run_id is None:
        raise FileNotFoundError("No base feature run id was provided.")

    base_metadata = load_feature_run_metadata(base_run_id, required=True)
    if base_metadata is None:
        raise FileNotFoundError(f"Feature run {base_run_id} does not have metadata.json.")
    if base_metadata.get("status") != "completed":
        raise ValueError(f"Feature run {base_run_id} is not completed; status={base_metadata.get('status')!r}.")

    graph_schema = normalize_graph_schema(base_metadata.get("graph_schema"))
    if not is_supported_edge_graph_schema(graph_schema):
        raise ValueError(
            f"Feature run {base_run_id} has unsupported graph_schema={graph_schema!r}; expected one of "
            f"{SUPPORTED_EDGE_GRAPH_SCHEMAS!r}."
        )

    base_return_types = metadata_return_types(base_metadata)
    base_modes = metadata_intended_receiver_modes(base_metadata)
    base_next_action_conditions_enabled = metadata_next_action_conditions_enabled(base_metadata)
    if args.next_action_conditions_enabled != base_next_action_conditions_enabled:
        raise ValueError(
            "Feature run extensions must use the same next-action condition setting as the base run; "
            f"base next_action_conditions_enabled={base_next_action_conditions_enabled!r}, "
            f"requested={args.next_action_conditions_enabled!r}."
        )
    final_return_types, added_return_types = union_preserving_order(base_return_types, args.requested_return_types)
    refresh_target_families = args_refresh_target_families(args)
    refresh_pass_height_labels = bool(getattr(args, "pass_height", False))
    extend_relative_speed_edge_features = bool(getattr(args, "extend_relative_speed_edge_features", False))
    if extend_relative_speed_edge_features:
        edge_schema = normalize_edge_graph_schema(graph_schema)
        if edge_schema != ALIGNMENT_EDGE_GRAPH_SCHEMA:
            raise ValueError(
                "--extend-relative-speed-edge-features requires a completed 4-column velocity-angle feature run "
                f"with graph_schema={ALIGNMENT_EDGE_GRAPH_SCHEMA!r}; got {edge_schema!r}."
            )

    base_model_id = base_metadata.get("intended_receiver_model_id")
    if base_model_id is not None:
        base_model_id = str(base_model_id)
    requested_model_id = str(args.intended_receiver_model_id) if args.intended_receiver_model_id else None
    replace_model = bool(getattr(args, "replace_intended_receiver_model", False))
    if replace_model and not requested_model_id:
        raise ValueError("--replace-intended-receiver-model requires --intended-receiver-model-id.")

    final_modes = list(base_modes)
    added_modes: list[str] = []
    regenerate_model_mode = False
    replaced_model_id: str | None = None
    replaced_modes: list[str] = []
    if INTENDED_RECEIVER_MODE_MODEL in base_modes:
        if not base_model_id:
            raise ValueError(
                f"Feature run {base_run_id} exposes intended_receiver_mode='model' but does not record "
                "intended_receiver_model_id."
            )
        if requested_model_id and requested_model_id != base_model_id:
            if not replace_model:
                raise ValueError(
                    f"Feature run {base_run_id} already uses intended_receiver_model_id={base_model_id!r}; "
                    f"cannot derive it with {requested_model_id!r}."
                )
            final_model_id = requested_model_id
            regenerate_model_mode = True
            replaced_model_id = base_model_id
            replaced_modes = [INTENDED_RECEIVER_MODE_MODEL]
        else:
            final_model_id = requested_model_id if replace_model and requested_model_id else base_model_id
            if replace_model and requested_model_id:
                regenerate_model_mode = True
                replaced_model_id = base_model_id
                replaced_modes = [INTENDED_RECEIVER_MODE_MODEL]
    elif requested_model_id:
        final_modes.append(INTENDED_RECEIVER_MODE_MODEL)
        added_modes.append(INTENDED_RECEIVER_MODE_MODEL)
        final_model_id = requested_model_id
    else:
        final_model_id = base_model_id

    if (
        not added_return_types
        and not added_modes
        and not regenerate_model_mode
        and not refresh_target_families
        and not refresh_pass_height_labels
        and not extend_relative_speed_edge_features
    ):
        raise ValueError(
            "Extension would not add any return types, intended-receiver modes, or target-label refreshes. "
            "Use a fresh full run or request a new --return_type / --intended-receiver-model-id / "
            "--refresh-target-family / --pass-height / --extend-relative-speed-edge-features."
        )

    in_place = bool(getattr(args, "in_place", False))
    overwrite_feature_run = bool(getattr(args, "overwrite_feature_run", False))
    if in_place and overwrite_feature_run:
        raise ValueError("--in-place and --overwrite-feature-run are mutually exclusive.")
    if (in_place or overwrite_feature_run) and getattr(args, "run_id", None):
        raise ValueError("--run-id cannot be used with --in-place or --overwrite-feature-run.")

    extension_mode = EXTENSION_MODE_DERIVED
    if in_place:
        extension_mode = EXTENSION_MODE_IN_PLACE
    elif overwrite_feature_run:
        extension_mode = EXTENSION_MODE_OVERWRITE_FEATURE_RUN

    is_regenerative = bool(
        refresh_target_families or refresh_pass_height_labels or regenerate_model_mode or replace_model
    )
    if extension_mode == EXTENSION_MODE_IN_PLACE and is_regenerative:
        raise ValueError(
            "--in-place only supports additive extensions. Use --overwrite-feature-run for target refreshes, "
            "pass-height refreshes, or intended-receiver model replacement."
        )

    if extension_mode == EXTENSION_MODE_DERIVED:
        output_run_id = str(args.run_id) if args.run_id else generate_run_id("feature")
        if output_run_id == base_run_id:
            raise ValueError("--run-id for an extension must name a new feature run, not the base run.")
        output_root = get_feature_run_root(output_run_id)
        if output_root.exists():
            raise FileExistsError(f"Derived feature run {output_run_id} already exists at {output_root}.")
    else:
        output_run_id = base_run_id

    target_run_id = output_run_id

    command_steps = extension_commands_for_plan(
        python=python,
        output_run_id=target_run_id,
        base_return_types=base_return_types,
        final_return_types=final_return_types,
        added_return_types=added_return_types,
        base_modes=base_modes,
        final_modes=final_modes,
        added_modes=added_modes,
        intended_receiver_model_id=final_model_id,
        regenerate_model_mode=regenerate_model_mode,
        refresh_existing_labels=bool(refresh_target_families or refresh_pass_height_labels),
        next_action_conditions_enabled=base_next_action_conditions_enabled,
        num_workers=getattr(args, "num_workers", "1"),
        worker_thread_limit=int(getattr(args, "worker_thread_limit", 1)),
    )
    command_steps = [
        FeatureGenerationStep(step.description, with_edge_schema_flags(step.command, graph_schema))
        for step in command_steps
    ]
    if refresh_pass_height_labels:
        threshold = str(pass_height_threshold_meters(args))
        command_steps = [
            FeatureGenerationStep(step.description, [*step.command, "--pass-height-threshold", threshold])
            for step in command_steps
        ]
    commands = [step.command for step in command_steps]
    final_graph_schema = copy.deepcopy(graph_schema)
    if extend_relative_speed_edge_features:
        final_graph_schema["edge_in_dim"] = 5
        final_graph_schema["add_v_edge_features"] = True
        final_graph_schema["add_relative_speed_edge_features"] = True
    return FeatureExtensionPlan(
        base_run_id=base_run_id,
        output_run_id=output_run_id,
        target_run_id=target_run_id,
        extension_mode=extension_mode,
        base_metadata=copy.deepcopy(base_metadata),
        base_return_types=base_return_types,
        final_return_types=final_return_types,
        added_return_types=added_return_types,
        base_intended_receiver_modes=base_modes,
        final_intended_receiver_modes=final_modes,
        added_intended_receiver_modes=added_modes,
        refresh_target_families=refresh_target_families,
        refresh_pass_height_labels=refresh_pass_height_labels,
        extend_relative_speed_edge_features=extend_relative_speed_edge_features,
        refreshed_return_types=final_return_types if refresh_target_families or refresh_pass_height_labels else [],
        refreshed_intended_receiver_modes=final_modes if refresh_target_families or refresh_pass_height_labels else [],
        intended_receiver_model_id=final_model_id,
        regenerate_model_mode=regenerate_model_mode,
        replaced_intended_receiver_model_id=replaced_model_id,
        replaced_intended_receiver_modes=replaced_modes,
        next_action_conditions_enabled=base_next_action_conditions_enabled,
        graph_schema=copy.deepcopy(final_graph_schema),
        commands=commands,
        command_steps=command_steps,
    )


def derived_metadata(args: argparse.Namespace, plan: FeatureExtensionPlan, status: str, error: str | None = None) -> dict[str, Any]:
    metadata = {
        "run_id": plan.output_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "derived_from_feature_run_id": plan.base_run_id,
        "extension_requested_return_types": args.requested_return_types,
        "extension_added_return_types": plan.added_return_types,
        "extension_added_intended_receiver_modes": plan.added_intended_receiver_modes,
        "extension_refresh_target_families": plan.refresh_target_families,
        "extension_refresh_pass_height_labels": plan.refresh_pass_height_labels,
        "extension_relative_speed_edge_features": plan.extend_relative_speed_edge_features,
        "pass_height_threshold_meters": pass_height_threshold_meters(args),
        "extension_refreshed_return_types": plan.refreshed_return_types,
        "extension_refreshed_intended_receiver_modes": plan.refreshed_intended_receiver_modes,
        "extension_replaced_intended_receiver_model_id": plan.replaced_intended_receiver_model_id,
        "extension_replaced_intended_receiver_modes": plan.replaced_intended_receiver_modes,
        "next_action_conditions_enabled": plan.next_action_conditions_enabled,
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "extension_commands": plan.commands,
        "intended_receiver_modes": plan.final_intended_receiver_modes,
        "intended_receiver_model_id": plan.intended_receiver_model_id,
        "graph_schema": copy.deepcopy(plan.graph_schema),
        "splits": ["train", "test"],
        "return_types": plan.final_return_types,
        "return_type": plan.final_return_types[0] if len(plan.final_return_types) == 1 else None,
        "base_feature_run_metadata": plan.base_metadata,
        "status": status,
    }
    if error:
        metadata["error"] = error
    return metadata


def extension_history_entry(
    args: argparse.Namespace,
    plan: FeatureExtensionPlan,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": plan.extension_mode,
        "status": status,
        "command": subprocess.list2cmdline(sys.argv),
        "requested_return_types": args.requested_return_types,
        "added_return_types": plan.added_return_types,
        "added_intended_receiver_modes": plan.added_intended_receiver_modes,
        "refresh_target_families": plan.refresh_target_families,
        "refresh_pass_height_labels": plan.refresh_pass_height_labels,
        "relative_speed_edge_features": plan.extend_relative_speed_edge_features,
        "refreshed_return_types": plan.refreshed_return_types,
        "refreshed_intended_receiver_modes": plan.refreshed_intended_receiver_modes,
        "replaced_intended_receiver_model_id": plan.replaced_intended_receiver_model_id,
        "replaced_intended_receiver_modes": plan.replaced_intended_receiver_modes,
        "intended_receiver_model_id": plan.intended_receiver_model_id,
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "commands": plan.commands,
        "pass_height_threshold_meters": pass_height_threshold_meters(args),
    }
    if error:
        entry["error"] = error
    return entry


def mutating_metadata(
    args: argparse.Namespace,
    plan: FeatureExtensionPlan,
    status: str,
    error: str | None = None,
    *,
    advertise_final_state: bool = True,
    history_status: str | None = None,
) -> dict[str, Any]:
    metadata = copy.deepcopy(plan.base_metadata)
    history = metadata.get("extension_history")
    if not isinstance(history, list):
        history = []
    history.append(extension_history_entry(args, plan, history_status or status, error=error))

    return_types = plan.final_return_types if advertise_final_state else plan.base_return_types
    intended_receiver_modes = (
        plan.final_intended_receiver_modes if advertise_final_state else plan.base_intended_receiver_modes
    )
    intended_receiver_model_id = (
        plan.intended_receiver_model_id
        if advertise_final_state
        else plan.base_metadata.get("intended_receiver_model_id")
    )
    graph_schema = copy.deepcopy(plan.graph_schema if advertise_final_state else plan.base_metadata.get("graph_schema"))

    metadata.update(
        {
            "run_id": plan.base_run_id,
            "command": subprocess.list2cmdline(sys.argv),
            "last_extension_mode": plan.extension_mode,
            "extension_requested_return_types": args.requested_return_types,
            "extension_added_return_types": plan.added_return_types if advertise_final_state else [],
            "extension_added_intended_receiver_modes": (
                plan.added_intended_receiver_modes if advertise_final_state else []
            ),
            "extension_refresh_target_families": plan.refresh_target_families,
            "extension_refresh_pass_height_labels": plan.refresh_pass_height_labels,
            "extension_relative_speed_edge_features": (
                plan.extend_relative_speed_edge_features if advertise_final_state else False
            ),
            "pass_height_threshold_meters": pass_height_threshold_meters(args),
            "extension_refreshed_return_types": plan.refreshed_return_types if advertise_final_state else [],
            "extension_refreshed_intended_receiver_modes": (
                plan.refreshed_intended_receiver_modes if advertise_final_state else []
            ),
            "extension_replaced_intended_receiver_model_id": (
                plan.replaced_intended_receiver_model_id if advertise_final_state else None
            ),
            "extension_replaced_intended_receiver_modes": (
                plan.replaced_intended_receiver_modes if advertise_final_state else []
            ),
            "next_action_conditions_enabled": plan.next_action_conditions_enabled,
            "num_workers": str(getattr(args, "num_workers", "1")),
            "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
            "extension_commands": plan.commands,
            "intended_receiver_modes": intended_receiver_modes,
            "intended_receiver_model_id": intended_receiver_model_id,
            "graph_schema": graph_schema,
            "splits": ["train", "test"],
            "return_types": return_types,
            "return_type": return_types[0] if len(return_types) == 1 else None,
            "extension_history": history,
            "status": status,
        }
    )
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    return metadata


def copy_base_feature_run(base_run_id: str, output_run_id: str) -> Path:
    base_root = get_feature_run_root(base_run_id)
    output_root = get_feature_run_root(output_run_id)

    def ignore_root_metadata(current_dir: str, names: list[str]) -> set[str]:
        if Path(current_dir).resolve() == base_root.resolve() and "metadata.json" in names:
            return {"metadata.json"}
        return set()

    shutil.copytree(base_root, output_root, ignore=ignore_root_metadata)
    return output_root


def remove_model_mode_artifacts(output_root: Path, return_types: list[str]) -> None:
    paths = [
        get_resolved_action_dir(INTENDED_RECEIVER_MODE_MODEL, root=output_root),
        get_augmented_feature_dir(INTENDED_RECEIVER_MODE_MODEL, root=output_root),
        get_augmented_label_dir(INTENDED_RECEIVER_MODE_MODEL, root=output_root),
    ]
    for return_type in return_types:
        paths.extend(
            [
                get_action_label_dir(return_type, intended_receiver_mode=INTENDED_RECEIVER_MODE_MODEL, root=output_root),
                get_intent_train_label_dir(
                    return_type,
                    intended_receiver_mode=INTENDED_RECEIVER_MODE_MODEL,
                    root=output_root,
                ),
            ]
        )

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def relative_speed_graph_dirs(feature_root: Path, intended_receiver_modes: list[str]) -> list[Path]:
    dirs = [
        get_action_graph_dir(feature_root),
        get_post_action_graph_dir(feature_root),
        get_action_graph_intent_train_dir(feature_root),
        get_success_intent_graph_dir(feature_root),
    ]
    dirs.extend(get_augmented_feature_dir(mode, root=feature_root) for mode in intended_receiver_modes)
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = Path(directory)
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def append_relative_speed_to_graph(graph: object) -> object:
    if graph is None:
        return None
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        raise ValueError("Graph artifact is missing edge_attr required for relative-speed extension.")
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("Graph artifact is missing edge_index required for relative-speed extension.")
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph artifact is missing node features required for relative-speed extension.")
    edge_dim = int(graph.edge_attr.shape[1])
    if edge_dim != 4:
        raise ValueError(f"Expected 4-column edge_attr before relative-speed extension, got edge_dim={edge_dim}.")
    if graph.x.shape[1] <= config.NODE_FEATURE_VY:
        raise ValueError("Graph node features do not contain vx/vy columns required for relative-speed extension.")
    src_vx = graph.x[graph.edge_index[0], config.NODE_FEATURE_VX]
    src_vy = graph.x[graph.edge_index[0], config.NODE_FEATURE_VY]
    dst_vx = graph.x[graph.edge_index[1], config.NODE_FEATURE_VX]
    dst_vy = graph.x[graph.edge_index[1], config.NODE_FEATURE_VY]
    relative_speed = torch.sqrt((src_vx - dst_vx).square() + (src_vy - dst_vy).square()).unsqueeze(-1)
    graph.edge_attr = torch.cat([graph.edge_attr, relative_speed.to(dtype=graph.edge_attr.dtype)], dim=1)
    return graph


def extend_relative_speed_edge_features(feature_root: Path, intended_receiver_modes: list[str]) -> list[Path]:
    graph_files = [
        graph_file
        for directory in relative_speed_graph_dirs(feature_root, intended_receiver_modes)
        for graph_file in sorted(directory.glob("*.pt"))
    ]
    if not graph_files:
        raise FileNotFoundError(f"No graph artifact files found under {feature_root}.")

    temp_files: list[tuple[Path, Path]] = []
    graphs_extended = 0
    null_graphs_preserved = 0
    try:
        for graph_file in graph_files:
            graphs = torch.load(graph_file, weights_only=False)
            if not isinstance(graphs, list):
                raise TypeError(f"Graph artifact {graph_file} is not a list.")
            null_graphs_preserved += sum(graph is None for graph in graphs)
            graphs_extended += sum(graph is not None for graph in graphs)
            extended_graphs = [append_relative_speed_to_graph(graph) for graph in graphs]
            temp_file = graph_file.with_name(f"{graph_file.name}.relative_speed.tmp")
            torch.save(extended_graphs, temp_file)
            temp_files.append((graph_file, temp_file))

        for graph_file, temp_file in temp_files:
            temp_file.replace(graph_file)
    except Exception:
        for _, temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        raise
    print(
        "Extended relative-speed edge features for "
        f"{len(temp_files)} graph files, {graphs_extended} graphs; "
        f"preserved {null_graphs_preserved} null graph placeholders."
    )
    return [graph_file for graph_file, _ in temp_files]


def run_extension_generation(args: argparse.Namespace) -> None:
    plan = build_extension_plan(args)
    if plan.extension_mode == EXTENSION_MODE_DERIVED:
        output_root = copy_base_feature_run(plan.base_run_id, plan.output_run_id)
        write_run_metadata(output_root, derived_metadata(args, plan, "in_progress"))
        try:
            if plan.regenerate_model_mode:
                remove_model_mode_artifacts(output_root, plan.final_return_types)
            run_generation_steps(plan.command_steps)
            if plan.extend_relative_speed_edge_features:
                extend_relative_speed_edge_features(output_root, plan.final_intended_receiver_modes)
        except Exception as exc:
            write_run_metadata(output_root, derived_metadata(args, plan, "failed", error=str(exc)))
            raise

        write_run_metadata(output_root, derived_metadata(args, plan, "completed"))
        write_latest_run("feature", plan.output_run_id)
        print(f"Derived feature run id: {plan.output_run_id}")
        print(f"Derived from feature run id: {plan.base_run_id}")
        return

    target_root = get_feature_run_root(plan.target_run_id)
    if plan.extension_mode == EXTENSION_MODE_OVERWRITE_FEATURE_RUN:
        write_run_metadata(target_root, mutating_metadata(args, plan, "in_progress"))

    try:
        if plan.regenerate_model_mode:
            remove_model_mode_artifacts(target_root, plan.final_return_types)
        run_generation_steps(plan.command_steps)
        if plan.extend_relative_speed_edge_features:
            extend_relative_speed_edge_features(target_root, plan.final_intended_receiver_modes)
    except Exception as exc:
        if plan.extension_mode == EXTENSION_MODE_IN_PLACE:
            write_run_metadata(
                target_root,
                mutating_metadata(
                    args,
                    plan,
                    "completed",
                    error=str(exc),
                    advertise_final_state=False,
                    history_status="failed",
                ),
            )
        else:
            write_run_metadata(target_root, mutating_metadata(args, plan, "failed", error=str(exc)))
        raise

    write_run_metadata(target_root, mutating_metadata(args, plan, "completed"))
    print(f"Updated feature run id: {plan.target_run_id}")
    print(f"Extension mode: {plan.extension_mode}")


def run_full_generation(args: argparse.Namespace) -> None:
    args.run_id = args.run_id or generate_run_id("feature")
    python = sys.executable
    command_steps = [
        FeatureGenerationStep(step.description, with_mode_flags(step.command, args))
        for step in full_generation_commands(python)
    ]
    run_generation_steps(command_steps)

    run_root = get_feature_run_root(args.run_id)
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "intended_receiver_modes": args.intended_receiver_modes,
        "intended_receiver_model_id": args.intended_receiver_model_id,
        "graph_schema": graph_schema_from_args(args),
        "next_action_conditions_enabled": args.next_action_conditions_enabled,
        "pass_height_threshold_meters": pass_height_threshold_meters(args),
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "splits": ["train", "test"],
        "return_types": args.return_types,
        "return_type": args.return_types[0] if len(args.return_types) == 1 else None,
        "commands": [step.command for step in command_steps],
        "status": "completed",
    }
    write_run_metadata(run_root, metadata)
    write_latest_run("feature", args.run_id)
    print(f"Feature run id: {args.run_id}")


def main() -> None:
    args = parse_args()
    if args.extend_feature_run_id:
        run_extension_generation(args)
    else:
        run_full_generation(args)


if __name__ == "__main__":
    main()
