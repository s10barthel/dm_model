from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.edge_schema import (edge_names, schema_for_names, generation_edge_names, add_interaction_arguments,
                                generation_kwargs, interaction_values, CANONICAL_FEATURES, INTERACTION_GROUPS, EDGE_GROUPS)
from scripts.feature_extension_diagnostics import ExtensionAttempt, extend_graph_files
from datatools.graph_validation import add_graph_validation_argument, graph_validation_flags, graph_validation_enabled
from datatools import config
from datatools.artifact_storage import clone_feature_artifacts
from datatools.xt_refresh import refresh_xt
from datatools.trajectory import TRAJECTORY_SPEED
from datatools.trajectory_refresh import run_trajectory_refresh
from datatools.pass_trajectory import PASS_DURATION_KEY, PASS_DURATION, ANGLE_ONLY_KEY, ANGLE_ONLY, ORIGINAL_KEY, ORIGINAL, MAX_DURATION_KEY, positive_duration
from models.pass_height import positive_height
from datatools.ball_carries import CARRY_DEFINITION_VERSION
from datatools.carry_policy import CarryPolicy
from datatools.endpoint_policy import ENDPOINT_POLICY_VERSION
from datatools.graph_feature import infer_node_feature_dim
from project_config import (
    add_split_arguments,
    split_selector,
    split_metadata,
    split_cli_args,
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
    resolve_split_manifest,
    resolve_generation_intended_receiver_modes,
    resolve_requested_return_types,
    resolve_feature_run_id,
    validate_intended_receiver_mode,
    write_latest_run,
    write_run_metadata,
    XT_MATCH_DIR,
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
    extend_interaction_edge_features: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_graph_validation_argument(parser)
    add_split_arguments(parser)
    parser.add_argument(
        "--return_type",
        action="append",
        default=None,
        help=(
            "Resolved return type for generated action labels. Repeat the flag to include multiple return types, "
            "including disc_<gamma>_skip1, disc_max_<gamma>[_skip1], disc_poly_max_<b>_<z>[_spstop], "
            "and next_<N>_skip1, plus in_<N> for xt/goal_distance/epv training."
        ),
    )
    parser.add_argument(
        "--intended-receiver-model-id",
        default=None,
        help="Optional success_intent checkpoint used to add the model-backed intended-receiver variant.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-copy-mode", choices=("hardlink", "copy"), default="hardlink",
                        help="Storage for unchanged derived-run artifacts; copying requires explicit opt-in.")
    parser.add_argument("--xt-match-dir", type=Path, default=XT_MATCH_DIR,
                        help="xT annotation directory used by the targeted xT refresh.")
    parser.add_argument(
        "--extend-feature-run-id",
        default=None,
        help=(
            "Extend this completed feature run. By default create a derived run sharing unchanged artifacts and "
            "generating only newly requested return types, refreshed target labels, or the model intended-receiver "
            "variant; use --in-place to update the existing run."
        ),
    )
    parser.add_argument("--replace-carries", action="store_true",
                        help="Create a derived run reusing canonical artifacts and completely rebuilding carries; supports --extend-feature-run-id latest.")
    parser.add_argument("--max-pass-duration", type=positive_duration, default=None,
                        help="Reject pass endpoints exceeding this many seconds; disabled by default.")
    parser.add_argument("--refresh-failed-pass-trajectories", action="store_true",
                        help="Refresh model-only failed-pass observations using proximity-gated contacts; invalidate model mode for regeneration.")
    parser.add_argument("--failed-pass-contact-detection", action="store_true",
                        help="Enable model-only contact cutoffs for fresh generation. Existing runs use the dedicated refresh.")
    parser.add_argument("--retire-elastic", action="store_true",
                        help="Retire Elastic reception fallback (not launch timing); refresh height/receiver targets and post-action states. Combine with --replace-carries for carry-enabled sources.")
    for name, value in asdict(CarryPolicy()).items():
        parser.add_argument("--carry-" + name.replace("_", "-"), type=float, default=value)
    parser.add_argument(
        "--refresh-trajectory", "--refresh-trajectory-speed", dest="refresh_trajectory", action="store_true",
        help="Refresh canonical pass endpoints, height, trajectory, receiver dependencies and retire stale model mode; requires --extend-feature-run-id. "
             "--refresh-trajectory-speed is a deprecated alias. Original and angle_only receiver labels and their dependencies are regenerated.",
    )
    parser.add_argument(
        "--refresh-target-family",
        action="append",
        choices=REFRESH_TARGET_FAMILIES,
        default=None,
        help=(
            "With --extend-feature-run-id, refresh label targets from current sidecar artifacts without "
            "rebuilding graphs. xT refresh runs separately from other refresh families."
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
        "--use-carries",
        action="store_true",
        help=(
            "Generate additive Sportec carry-augmented base artifacts. With --extend-feature-run-id, "
            "this can extend an existing completed run without changing its canonical artifacts."
        ),
    )
    parser.add_argument(
        "--pass-height-threshold",
        type=positive_height,
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
            "When extending a feature run that already contains the model intended-receiver variant, "
            "regenerate only model-mode artifacts with --intended-receiver-model-id."
        ),
    )
    parser.add_argument(
        "--in-place",
        "--overwrite-feature-run",
        dest="in_place",
        action="store_true",
        default=False,
        help=(
            "With --extend-feature-run-id, apply the explicitly requested extension or refresh to the "
            "existing run. Otherwise create a derived run sharing unchanged artifacts. "
            "--overwrite-feature-run is a compatibility alias."
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
    add_interaction_arguments(parser, training=False)
    parser.add_argument("--extend-interaction-edge-features", action="store_true",
                        help="Add positional direction, relative velocity and separation rate to an existing feature run using its raw node inputs.")
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
    args.explicit_carry_parameters = {key for key in asdict(CarryPolicy())
        if any(token.split("=", 1)[0] == "--carry-" + key.replace("_", "-") for token in sys.argv[1:])}
    vars(args).update(split_selector(args))
    args.requested_return_types = resolve_requested_return_types(args.return_type) if args.return_type else []
    args.return_types = args.requested_return_types or resolve_requested_return_types(None)
    args.intended_receiver_modes = resolve_generation_intended_receiver_modes(args.intended_receiver_model_id)
    args.refresh_target_families = normalize_refresh_target_families(args.refresh_target_family)
    args.refresh_trajectory_speed = args.refresh_trajectory  # Deprecated programmatic alias.
    if args.failed_pass_contact_detection and args.extend_feature_run_id:
        parser.error("Existing runs inherit detector policy; use --refresh-failed-pass-trajectories to enable it.")
    if args.refresh_failed_pass_trajectories:
        if not args.extend_feature_run_id:
            parser.error("--refresh-failed-pass-trajectories requires --extend-feature-run-id.")
        supplied = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        allowed = {"--refresh-failed-pass-trajectories", "--extend-feature-run-id", "--run-id",
                   "--artifact-copy-mode", "--num-workers", "--worker-thread-limit", "--in-place", "--overwrite-feature-run"}
        allowed.add("--graph-validation")
        if supplied - allowed:
            parser.error("Failed-pass refresh is dedicated; unsupported options: " + ", ".join(sorted(supplied - allowed)))
    if args.retire_elastic and args.extend_feature_run_id:
        if args.in_place:
            parser.error("--retire-elastic creates a derived run; --in-place is unsupported.")
        supplied = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        allowed = {"--retire-elastic", "--replace-carries", "--extend-feature-run-id", "--run-id",
                   "--artifact-copy-mode", "--num-workers", "--worker-thread-limit",
                   "--refresh-trajectory", "--refresh-trajectory-speed", "--max-pass-duration"}
        allowed.update("--carry-" + name.replace("_", "-") for name in asdict(CarryPolicy()))
        allowed.add("--graph-validation")
        if supplied - allowed:
            parser.error("Reception retirement inherits source settings; unsupported options: " + ", ".join(sorted(supplied - allowed)))
    if args.replace_carries:
        if not args.extend_feature_run_id or args.in_place:
            parser.error("--replace-carries requires --extend-feature-run-id and creates a derived run; --in-place is unsupported.")
        supplied = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        allowed = {"--replace-carries", "--retire-elastic", "--extend-feature-run-id", "--run-id", "--artifact-copy-mode",
                   "--num-workers", "--worker-thread-limit", "--refresh-trajectory", "--refresh-trajectory-speed", "--max-pass-duration"}
        allowed.update("--carry-" + name.replace("_", "-") for name in asdict(CarryPolicy()))
        allowed.add("--graph-validation")
        if supplied - allowed:
            parser.error("--replace-carries inherits source settings; unsupported options: " + ", ".join(sorted(supplied - allowed)))
    if args.refresh_trajectory:
        if not args.extend_feature_run_id:
            parser.error("--refresh-trajectory requires --extend-feature-run-id.")
        supplied = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
        allowed = {"--refresh-trajectory", "--refresh-trajectory-speed", "--extend-feature-run-id", "--in-place", "--overwrite-feature-run",
                   "--run-id", "--artifact-copy-mode", "--num-workers", "--worker-thread-limit",
                   "--replace-carries", "--retire-elastic", "--max-pass-duration"}
        allowed.update("--carry-" + name.replace("_", "-") for name in asdict(CarryPolicy()))
        allowed.add("--graph-validation")
        if supplied - allowed:
            parser.error("--refresh-trajectory is dedicated; unsupported combined options: "
                         + ", ".join(sorted(supplied - allowed)))
        if not (args.retire_elastic or args.replace_carries):
            base = load_feature_run_metadata(args.extend_feature_run_id, required=True)
            vars(args).update(split_selector(base))
            args.next_action_conditions_enabled = metadata_next_action_conditions_enabled(base)
    if args.extend_feature_run_id and args.max_pass_duration is not None and not args.refresh_trajectory:
        parser.error("Changing --max-pass-duration on an existing run requires --refresh-trajectory.")
    if "xt" in args.refresh_target_families:
        if args.use_carries or len(args.refresh_target_families) != 1:
            parser.error("xT refresh must run separately from carry generation and other target refreshes.")
        # A target-only operation inherits the source split/selection settings unless
        # the caller explicitly supplies them (build_extension_plan checks equality).
        if args.extend_feature_run_id:
            base = load_feature_run_metadata(args.extend_feature_run_id, required=True)
            supplied = {token.split("=", 1)[0] for token in sys.argv[1:]}
            if not supplied.intersection({"--train-count", "--train-split"}):
                vars(args).update(split_selector(base))
            if not supplied.intersection({"--next-action-conditions-on", "--next-action-conditions-off"}):
                args.next_action_conditions_enabled = metadata_next_action_conditions_enabled(base)
    if args.refresh_target_families and not args.extend_feature_run_id:
        parser.error("--refresh-target-family requires --extend-feature-run-id.")
    if args.pass_height_threshold is not None and not args.pass_height:
        parser.error("--pass-height-threshold requires --pass-height.")
    if args.extend_interaction_edge_features and not args.extend_feature_run_id:
        parser.error("--extend-interaction-edge-features requires --extend-feature-run-id.")
    if args.extend_relative_speed_edge_features and not args.extend_feature_run_id:
        parser.error("--extend-relative-speed-edge-features requires --extend-feature-run-id.")
    if args.in_place and not args.extend_feature_run_id:
        parser.error("--in-place (alias --overwrite-feature-run) requires --extend-feature-run-id.")
    if args.in_place and args.run_id:
        parser.error("--run-id cannot be used with --in-place (alias --overwrite-feature-run).")
    if args.add_relative_speed_edge_features and not args.add_v_edge_features:
        parser.error("--relative-speed-edge-features requires --v-edge-features.")
    if args.worker_thread_limit < 1:
        parser.error("--worker-thread-limit must be a positive integer.")
    return args


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def run_generation_steps(steps: list[FeatureGenerationStep], attempt: ExtensionAttempt | None = None) -> None:
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        print(f"Feature generation step {index}/{total}: {step.description}")
        if attempt is not None:
            attempt.begin_mutation("generating", current_command=step.command)
        command = list(step.command)
        if attempt is not None:
            value = "on" if attempt.data["target_settings"].get("graph_validation", True) else "off"
            if "--graph-validation" in command:
                command[command.index("--graph-validation") + 1] = value
            else:
                command.extend(["--graph-validation", value])
        run_command(command)
        if attempt is not None:
            attempt.mutation_finished()


def with_mode_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    if getattr(args, "failed_pass_contact_detection", False):
        command = [*command, "--failed-pass-contact-detection"]
    command = list(command)
    for return_type in args.return_types:
        command.extend(["--return_type", return_type])
    if args.intended_receiver_model_id:
        command.extend(["--intended-receiver-model-id", args.intended_receiver_model_id])
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    command.extend(split_cli_args(args))
    command.extend(["--num-workers", str(getattr(args, "num_workers", "1"))])
    command.extend(["--worker-thread-limit", str(getattr(args, "worker_thread_limit", 1))])
    command.extend(graph_validation_flags(args))
    command.append(next_action_conditions_flag(args.next_action_conditions_enabled))
    command.append("--v-edge-features" if bool(getattr(args, "add_v_edge_features", False)) else "--no-v-edge-features")
    command.append(
        "--relative-speed-edge-features"
        if bool(getattr(args, "add_relative_speed_edge_features", False))
        else "--no-relative-speed-edge-features"
    )
    for group in INTERACTION_GROUPS:
        option = group.replace("_", "-") + "-edge-features"
        command.append("--" + ("" if getattr(args, f"add_{group}_edge_features", True) else "no-") + option)
    if getattr(args, "max_pass_duration", None) is not None:
        command.extend(["--max-pass-duration", str(args.max_pass_duration)])
    if bool(getattr(args, "pass_height", False)):
        command.extend(["--pass-height-threshold", str(pass_height_threshold_meters(args))])
    return command


def pass_height_threshold_meters(args: argparse.Namespace) -> float:
    value = getattr(args, "pass_height_threshold", None)
    return config.PASS_HEIGHT_THRESHOLD_METERS if value is None else float(value)


def next_action_conditions_flag(enabled: bool) -> str:
    return "--next-action-conditions-on" if enabled else "--next-action-conditions-off"


def full_generation_commands(python: str, use_carries: bool = False) -> list[FeatureGenerationStep]:
    steps = [
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
    if use_carries:
        steps.extend(
            [
                FeatureGenerationStep(
                    "train split with carry-augmented base and post_action",
                    [python, "datatools/graph_feature.py", "--action_type", "all", "--split", "train", "--post_action", "--use-carries"],
                ),
                FeatureGenerationStep(
                    "test split with carry-augmented base and post_action",
                    [python, "datatools/graph_feature.py", "--action_type", "all", "--split", "test", "--post_action", "--use-carries"],
                ),
            ]
        )
    return steps


def normalize_graph_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    normalized = dict(schema)
    normalized.update(edge_in_dim=int(schema.get("edge_in_dim", -1)),
                      add_v_edge_features=bool(schema.get("add_v_edge_features", False)),
                      add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)))
    if schema.get("node_in_dim") is not None:
        normalized["node_in_dim"] = int(schema["node_in_dim"])
    return normalized


def normalize_edge_graph_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_graph_schema(schema)
    return {key: value for key, value in normalized.items() if key != "node_in_dim"}


def is_supported_edge_graph_schema(schema: dict[str, Any]) -> bool:
    try:
        names = edge_names(schema)
        return bool(names) and ("relative_speed" not in names or "velocity_angle_cos" in names)
    except (ValueError, TypeError):
        return False


def graph_schema_from_args(args: argparse.Namespace) -> dict[str, Any]:
    names = generation_edge_names(bool(getattr(args, "add_v_edge_features", False)),
                                  bool(getattr(args, "add_relative_speed_edge_features", False)),
                                  *(getattr(args, f"add_{group}_edge_features", True) for group in INTERACTION_GROUPS))
    return {"node_in_dim": infer_node_feature_dim(extend=True), **schema_for_names(names)}


def edge_feature_flags_for_schema(schema: dict[str, Any]) -> list[str]:
    normalized = normalize_edge_graph_schema(schema)
    flags = ["--v-edge-features" if normalized["add_v_edge_features"] else "--no-v-edge-features"]
    flags.append(
        "--relative-speed-edge-features"
        if normalized["add_relative_speed_edge_features"]
        else "--no-relative-speed-edge-features"
    )
    for group, enabled in generation_kwargs(normalized).items():
        option = group.removeprefix("add_").replace("_", "-")
        flags.append("--" + ("" if enabled else "no-") + option)
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
    requested_split = resolve_split_manifest(**split_selector(args))
    base_split_id = base_metadata.get("split_manifest_id")
    if not base_split_id and split_selector(args) != split_selector({}):
        raise ValueError("Legacy feature runs without split metadata can only be used with --train-split 50.")
    if base_split_id and base_split_id != requested_split["manifest_id"]:
        raise ValueError(
            f"Feature run {base_run_id} uses split {base_split_id}, but requested split {split_cli_args(args)} "
            f"resolves to {requested_split['manifest_id']}. Create a new full feature run for a different split."
        )
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
    extend_interaction_edge_features = bool(getattr(args, "extend_interaction_edge_features", False))
    if extend_relative_speed_edge_features:
        edge_schema = normalize_edge_graph_schema(graph_schema)
        if edge_names(edge_schema) != tuple(CANONICAL_FEATURES[:4]):
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

    if "xt" in refresh_target_families:
        if (len(refresh_target_families) != 1 or added_return_types or added_modes or replace_model
                or regenerate_model_mode or refresh_pass_height_labels or extend_relative_speed_edge_features
                or extend_interaction_edge_features or bool(getattr(args, "use_carries", False))):
            raise ValueError("xT refresh must run separately from other refreshes, new return types, "
                             "receiver replacement/addition, graph extensions, and carry generation.")

    if (
        not added_return_types
        and not added_modes
        and not regenerate_model_mode
        and not refresh_target_families
        and not refresh_pass_height_labels
        and not extend_relative_speed_edge_features
        and not extend_interaction_edge_features
    ):
        raise ValueError(
            "Extension would not add any return types, intended-receiver modes, or target-label refreshes. "
            "Use a fresh full run or request a new --return_type / --intended-receiver-model-id / "
            "--refresh-target-family / --pass-height / --extend-relative-speed-edge-features / --extend-interaction-edge-features."
        )

    in_place = bool(getattr(args, "in_place", False))
    if in_place and getattr(args, "run_id", None):
        raise ValueError("--run-id cannot be used with --in-place (alias --overwrite-feature-run).")

    extension_mode = EXTENSION_MODE_IN_PLACE if in_place else EXTENSION_MODE_DERIVED

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
    command_steps = [
        FeatureGenerationStep(
            step.description,
            [*step.command, *split_cli_args(args), *graph_validation_flags(args)],
        )
        for step in command_steps
    ]
    if refresh_pass_height_labels:
        threshold = str(pass_height_threshold_meters(args))
        command_steps = [
            FeatureGenerationStep(step.description, [*step.command, "--pass-height-threshold", threshold])
            for step in command_steps
        ]
    commands = [step.command for step in command_steps]
    if refresh_target_families == ["xt"]:
        command_steps, commands = [], []
    final_graph_schema = copy.deepcopy(graph_schema)
    if extend_relative_speed_edge_features:
        final_graph_schema.update(schema_for_names(CANONICAL_FEATURES[:5],
                                  explicit=graph_schema.get("edge_feature_names") is not None))
    if extend_interaction_edge_features:
        union = set(edge_names(final_graph_schema)) | {name for group in INTERACTION_GROUPS for name in EDGE_GROUPS[group]}
        final_graph_schema.update(schema_for_names([name for name in CANONICAL_FEATURES if name in union]))
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
        extend_interaction_edge_features=extend_interaction_edge_features,
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


def extension_pass_height_threshold(args, plan):
    if plan.refresh_target_families == ["xt"]:
        return plan.base_metadata.get("pass_height_threshold_meters", config.PASS_HEIGHT_THRESHOLD_METERS)
    return pass_height_threshold_meters(args)


def derived_metadata(args: argparse.Namespace, plan: FeatureExtensionPlan, status: str, error: str | None = None) -> dict[str, Any]:
    split_manifest = resolve_split_manifest(**split_selector(args))
    metadata = {
        **copy.deepcopy(plan.base_metadata),
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
        "extension_interaction_edge_features": plan.extend_interaction_edge_features,
        "pass_height_threshold_meters": extension_pass_height_threshold(args, plan),
        "extension_refreshed_return_types": plan.refreshed_return_types,
        "extension_refreshed_intended_receiver_modes": plan.refreshed_intended_receiver_modes,
        "extension_replaced_intended_receiver_model_id": plan.replaced_intended_receiver_model_id,
        "extension_replaced_intended_receiver_modes": plan.replaced_intended_receiver_modes,
        "next_action_conditions_enabled": plan.next_action_conditions_enabled,
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "graph_validation": graph_validation_enabled(getattr(args, "graph_validation", True)),
        "extension_commands": plan.commands,
        "intended_receiver_modes": plan.final_intended_receiver_modes,
        "intended_receiver_model_id": plan.intended_receiver_model_id,
        "graph_schema": copy.deepcopy(plan.graph_schema),
        "splits": ["train", "test"],
        **split_metadata(args),
        "split_manifest_id": split_manifest["manifest_id"],
        "split_manifest": split_manifest["metadata"],
        "return_types": plan.final_return_types,
        "return_type": plan.final_return_types[0] if len(plan.final_return_types) == 1 else None,
        "base_feature_run_metadata": plan.base_metadata,
        "status": status,
    }
    if status == "completed" and "model" in metadata.get("intended_receiver_modes", []):
        metadata["model_mode_regeneration_required"] = False
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
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
        "interaction_edge_features": plan.extend_interaction_edge_features,
        "refreshed_return_types": plan.refreshed_return_types,
        "refreshed_intended_receiver_modes": plan.refreshed_intended_receiver_modes,
        "replaced_intended_receiver_model_id": plan.replaced_intended_receiver_model_id,
        "replaced_intended_receiver_modes": plan.replaced_intended_receiver_modes,
        "intended_receiver_model_id": plan.intended_receiver_model_id,
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "graph_validation": graph_validation_enabled(getattr(args, "graph_validation", True)),
        "commands": plan.commands,
        "pass_height_threshold_meters": extension_pass_height_threshold(args, plan),
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
    split_manifest = resolve_split_manifest(**split_selector(args))
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
            **split_metadata(args),
            "split_manifest_id": split_manifest["manifest_id"],
            "split_manifest": split_manifest["metadata"],
            "command": subprocess.list2cmdline(sys.argv),
            "last_extension_mode": plan.extension_mode,
            "extension_requested_return_types": args.requested_return_types,
            "extension_added_return_types": plan.added_return_types if advertise_final_state else [],
            "extension_added_intended_receiver_modes": (
                plan.added_intended_receiver_modes if advertise_final_state else []
            ),
            "extension_refresh_target_families": plan.refresh_target_families,
            "extension_refresh_pass_height_labels": plan.refresh_pass_height_labels,
            "extension_interaction_edge_features": plan.extend_interaction_edge_features if advertise_final_state else False,
            "extension_relative_speed_edge_features": (
                plan.extend_relative_speed_edge_features if advertise_final_state else False
            ),
            "pass_height_threshold_meters": (extension_pass_height_threshold(args, plan) if advertise_final_state
                                             else plan.base_metadata.get("pass_height_threshold_meters", config.PASS_HEIGHT_THRESHOLD_METERS)),
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
            "graph_validation": graph_validation_enabled(getattr(args, "graph_validation", True)),
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
    if status == "completed" and "model" in metadata.get("intended_receiver_modes", []):
        metadata["model_mode_regeneration_required"] = False
    if error:
        metadata["error"] = error
    else:
        metadata.pop("error", None)
    return metadata


def copy_base_feature_run(base_run_id: str, output_run_id: str, *, mode="hardlink", diagnostics=None) -> Path:
    base_root = get_feature_run_root(base_run_id)
    output_root = get_feature_run_root(output_run_id)

    clone_feature_artifacts(base_root, output_root, mode=mode, diagnostics=diagnostics)
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

    paths = [*paths, *(Path(str(path) + "_carries") for path in paths)]
    resolved_root = output_root.resolve()
    for path in paths:
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"Model artifact path escapes the target run: {path}")
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
    for directory in [*dirs, *(Path(str(directory) + "_carries") for directory in dirs)]:
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
    if getattr(graph, "edge_feature_names", None) is not None:
        graph.edge_feature_names = list(CANONICAL_FEATURES[:5])
    saved_self = getattr(graph, "node_self_edge_attr", None)
    if saved_self is not None:
        graph.node_self_edge_attr = torch.cat([saved_self, saved_self.new_zeros((graph.num_nodes, 1))], dim=1)
    return graph


def extend_relative_speed_edge_features(feature_root: Path, intended_receiver_modes: list[str],
                                        attempt: ExtensionAttempt | None = None, *, graph_validation=True) -> list[Path]:
    graph_files = [path for directory in relative_speed_graph_dirs(feature_root, intended_receiver_modes)
                   for path in sorted(directory.glob("*.pt"))]
    if not graph_files:
        raise FileNotFoundError(f"No graph artifact files found under {feature_root}.")
    attempt = attempt or ExtensionAttempt(feature_root, "relative_speed", EXTENSION_MODE_IN_PLACE, {})
    result = extend_graph_files(graph_files, append_relative_speed_to_graph, ".relative_speed.tmp", attempt,
                                graph_validation=graph_validation)
    print(f"Extended relative-speed edge features in {len(result)} graph files.")
    return result


def append_interactions_to_graph(graph):
    if graph is None:
        return None
    source = edge_names(graph)
    interactions = interaction_values(graph.x, graph.edge_index)
    names = tuple(name for name in CANONICAL_FEATURES if name in source or name in interactions)
    columns = [graph.edge_attr[:, source.index(name)] if name in source else interactions[name].to(graph.edge_attr)
               for name in names]
    graph.edge_attr = torch.stack(columns, dim=-1)
    graph.edge_feature_names = list(names)
    graph.edge_schema_version = 1
    # Preserve saved physical loop attributes and append zero-valued new self inputs.
    saved_self = getattr(graph, "node_self_edge_attr", None)
    if saved_self is not None:
        if saved_self.shape != (graph.num_nodes, len(source)):
            raise ValueError("Saved self-edge metadata does not match the source schema.")
        graph.node_self_edge_attr = torch.stack([
            saved_self[:, source.index(name)] if name in source else saved_self.new_zeros(graph.num_nodes)
            for name in names
        ], dim=-1)
    return graph


def extend_interaction_edge_features(feature_root: Path, intended_receiver_modes: list[str],
                                     attempt: ExtensionAttempt | None = None, *, graph_validation=True) -> list[Path]:
    graph_files = [path for directory in relative_speed_graph_dirs(feature_root, intended_receiver_modes)
                   for path in sorted(directory.glob("*.pt"))]
    if not graph_files:
        raise FileNotFoundError(f"No graph artifact files found under {feature_root}.")
    attempt = attempt or ExtensionAttempt(feature_root, "interaction", EXTENSION_MODE_IN_PLACE, {})
    result = extend_graph_files(graph_files, append_interactions_to_graph, ".interactions.tmp", attempt,
                                graph_validation=graph_validation)
    print(f"Extended interaction edge features in {len(result)} graph files.")
    return result


def run_extension_generation(args: argparse.Namespace) -> None:
    plan = build_extension_plan(args)
    derived = plan.extension_mode == EXTENSION_MODE_DERIVED
    target_root = get_feature_run_root(plan.target_run_id)

    def metadata_factory(status, error, final):
        if derived:
            payload = derived_metadata(args, plan, status, error=error)
            if not final:
                payload.update(graph_schema=copy.deepcopy(plan.base_metadata.get("graph_schema")),
                               return_types=plan.base_return_types,
                               return_type=plan.base_return_types[0] if len(plan.base_return_types) == 1 else None,
                               intended_receiver_modes=plan.base_intended_receiver_modes,
                               intended_receiver_model_id=plan.base_metadata.get("intended_receiver_model_id"),
                               pass_height_threshold_meters=plan.base_metadata.get(
                                   "pass_height_threshold_meters", config.PASS_HEIGHT_THRESHOLD_METERS))
            return payload
        return mutating_metadata(args, plan, status, error=error, advertise_final_state=final,
                                 history_status="failed" if error is not None else status)

    target = dict(graph_validation=graph_validation_enabled(getattr(args, "graph_validation", True)), graph_schema=plan.graph_schema, return_types=plan.final_return_types,
                  intended_receiver_modes=plan.final_intended_receiver_modes,
                  intended_receiver_model_id=plan.intended_receiver_model_id,
                  relative_speed_edge_features=plan.extend_relative_speed_edge_features,
                  interaction_edge_features=plan.extend_interaction_edge_features,
                  refresh_target_families=plan.refresh_target_families,
                  pass_height_threshold_meters=extension_pass_height_threshold(args, plan),
                  next_action_conditions_enabled=plan.next_action_conditions_enabled)
    attempt = ExtensionAttempt(target_root, "feature_extension", plan.extension_mode, target,
                               metadata_factory, lambda root, payload: write_run_metadata(root, payload))
    try:
        if derived:
            attempt.begin_mutation("copying", current_file=target_root)
            copy_base_feature_run(plan.base_run_id, plan.output_run_id,
                                  mode=getattr(args, "artifact_copy_mode", "hardlink"), diagnostics=attempt.data)
            attempt.mutation_finished()
        else:
            attempt.enter("staging")
        if plan.regenerate_model_mode:
            attempt.begin_mutation("removing_artifacts", current_file=target_root)
            remove_model_mode_artifacts(target_root, plan.final_return_types)
            attempt.mutation_finished()
        if plan.refresh_target_families == ["xt"]:
            refresh_xt(target_root, plan.base_metadata, xt_match_dir=getattr(args, "xt_match_dir", XT_MATCH_DIR),
                       attempt=attempt)
        else:
            run_generation_steps(plan.command_steps, attempt=attempt)
            if ((plan.base_metadata.get("carry_variant") or {}).get("available")
                    and ("model" in plan.added_intended_receiver_modes or plan.regenerate_model_mode)):
                carry_args = copy.copy(args)
                carry_metadata = dict(plan.base_metadata, intended_receiver_modes=["model"],
                                      intended_receiver_model_id=plan.intended_receiver_model_id)
                run_generation_steps(carry_extension_steps(carry_args, carry_metadata, plan.target_run_id), attempt=attempt)
        if plan.extend_relative_speed_edge_features:
            extend_relative_speed_edge_features(target_root, plan.final_intended_receiver_modes, attempt=attempt,
                                                graph_validation=getattr(args, "graph_validation", True))
        if plan.extend_interaction_edge_features:
            extend_interaction_edge_features(target_root, plan.final_intended_receiver_modes, attempt=attempt,
                                                graph_validation=getattr(args, "graph_validation", True))
        attempt.finish()
    except BaseException as exc:
        attempt.fail(exc)
        raise
    if derived:
        write_latest_run("feature", plan.output_run_id)
        print(f"Derived feature run id: {plan.output_run_id}")
        print(f"Derived from feature run id: {plan.base_run_id}")
    else:
        print(f"Updated feature run id: {plan.target_run_id}")
        print(f"Extension mode: {plan.extension_mode}")


def carry_extension_steps(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    output_run_id: str,
) -> list[FeatureGenerationStep]:
    python = sys.executable
    schema = normalize_graph_schema(metadata.get("graph_schema"))
    common = [
        "--run-id",
        output_run_id,
        *split_cli_args(metadata),
        "--num-workers",
        str(getattr(args, "num_workers", "1")),
        "--worker-thread-limit",
        str(getattr(args, "worker_thread_limit", 1)),
        next_action_conditions_flag(metadata_next_action_conditions_enabled(metadata)),
        *edge_feature_flags_for_schema(schema),
    ]
    for return_type in metadata_return_types(metadata):
        common.extend(["--return_type", return_type])
    for mode in metadata_intended_receiver_modes(metadata):
        common.extend(["--only-intended-receiver-mode", mode])
    model_id = metadata.get("intended_receiver_model_id")
    if model_id:
        common.extend(["--intended-receiver-model-id", str(model_id)])
    threshold = metadata.get("pass_height_threshold_meters")
    if threshold is not None:
        common.extend(["--pass-height-threshold", str(float(threshold))])

    steps: list[FeatureGenerationStep] = []
    for split in ("train", "test"):
        command = [
            python,
            "datatools/graph_feature.py",
            "--action_type",
            "all",
            "--split",
            split,
            "--post_action",
            "--use-carries",
            *common,
        ]
        steps.append(FeatureGenerationStep(f"{split} split carry-augmented base", command))
    return steps


def run_carry_extension_generation(args: argparse.Namespace) -> None:
    base_run_id = resolve_feature_run_id(args.extend_feature_run_id, required=True, allow_latest=False)
    if base_run_id is None:
        raise FileNotFoundError("No base feature run id was provided.")
    metadata = load_feature_run_metadata(base_run_id, required=True)
    if metadata is None or metadata.get("status") != "completed":
        raise ValueError(f"Feature run {base_run_id} must be completed before adding carries.")
    if metadata.get(PASS_DURATION_KEY) != PASS_DURATION:
        raise ValueError("Adding carries to an existing run requires --refresh-trajectory first.")
    if bool((metadata.get("carry_variant") or {}).get("available", False)):
        raise ValueError(f"Feature run {base_run_id} already contains carry-augmented artifacts.")

    mode = EXTENSION_MODE_IN_PLACE if args.in_place else EXTENSION_MODE_DERIVED
    output_run_id = base_run_id if args.in_place else (args.run_id or generate_run_id("feature"))
    output_root = get_feature_run_root(output_run_id)
    if not args.in_place and output_root.exists():
        raise FileExistsError(f"Derived feature run {output_run_id} already exists at {output_root}.")
    steps = carry_extension_steps(args, metadata, output_run_id)
    extend_interactions = bool(getattr(args, "extend_interaction_edge_features", False))
    final_schema = copy.deepcopy(metadata.get("graph_schema"))
    if extend_interactions:
        source_names = edge_names(normalize_graph_schema(final_schema))
        final_schema = {**(final_schema or {}), **schema_for_names(tuple(
            name for name in CANONICAL_FEATURES
            if name in source_names or any(name in EDGE_GROUPS[group] for group in INTERACTION_GROUPS)
        ))}

    def metadata_factory(status, error, final):
        updated = copy.deepcopy(metadata)
        updated.update(run_id=output_run_id, status=status, last_extension_mode=mode,
                       graph_validation=graph_validation_enabled(getattr(args, "graph_validation", True)),
                       command=subprocess.list2cmdline(sys.argv))
        if not args.in_place:
            updated["base_feature_run_id"] = base_run_id
        entry = dict(created_at=datetime.now().isoformat(timespec="seconds"), type="carry_variant", mode=mode,
                     command=subprocess.list2cmdline(sys.argv),
                     interaction_edge_features=extend_interactions, status="failed" if error is not None else status,
                     base_feature_run_id=base_run_id, commands=[step.command for step in steps])
        if error is not None:
            entry["error"] = error
            updated["carry_variant_error"] = error
        else:
            updated.pop("carry_variant_error", None)
        updated["extension_history"] = list(updated.get("extension_history") or []) + [entry]
        if final:
            updated["graph_schema"] = final_schema
            updated["carry_variant"] = dict(available=True, definition=CARRY_DEFINITION_VERSION,
                                            artifact_suffix="_carries",
                                            validation_parameters=asdict(CarryPolicy()),
                                            sidecar_root="carry_sidecars",
                                            generated_at=datetime.now().isoformat(timespec="seconds"))
        return updated

    attempt = ExtensionAttempt(output_root, "carry_extension", mode,
                               dict(graph_validation=graph_validation_enabled(getattr(args, "graph_validation", True)),
                                    graph_schema=final_schema, carry_variant=True,
                                    interaction_edge_features=extend_interactions),
                               metadata_factory, lambda root, payload: write_run_metadata(root, payload))
    try:
        if not args.in_place:
            attempt.begin_mutation("copying", current_file=output_root)
            copy_base_feature_run(base_run_id, output_run_id,
                                  mode=getattr(args, "artifact_copy_mode", "hardlink"), diagnostics=attempt.data)
            attempt.mutation_finished()
        else:
            attempt.enter("staging")
        run_generation_steps(steps, attempt=attempt)
        if extend_interactions:
            extend_interaction_edge_features(output_root, metadata_intended_receiver_modes(metadata), attempt=attempt,
                                              graph_validation=getattr(args, "graph_validation", True))
        attempt.finish()
    except BaseException as exc:
        attempt.fail(exc)
        raise
    if not args.in_place:
        write_latest_run("feature", output_run_id)
    print(f"Feature run id with carry variant: {output_run_id}")


def run_full_generation(args: argparse.Namespace) -> None:
    args.run_id = args.run_id or generate_run_id("feature")
    python = sys.executable
    command_steps = [
        FeatureGenerationStep(step.description, with_mode_flags(step.command, args))
        for step in full_generation_commands(python, use_carries=bool(getattr(args, "use_carries", False)))
    ]
    run_generation_steps(command_steps)

    run_root = get_feature_run_root(args.run_id)
    split_manifest = resolve_split_manifest(**split_selector(args))
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "intended_receiver_modes": args.intended_receiver_modes,
        "intended_receiver_model_id": args.intended_receiver_model_id,
        "graph_schema": graph_schema_from_args(args),
        "trajectory_speed_definition": TRAJECTORY_SPEED,
        PASS_DURATION_KEY: PASS_DURATION,
        ANGLE_ONLY_KEY: ANGLE_ONLY,
        ORIGINAL_KEY: ORIGINAL,
        MAX_DURATION_KEY: getattr(args, "max_pass_duration", None),
        "next_action_conditions_enabled": args.next_action_conditions_enabled,
        "endpoint_consumer_version": ENDPOINT_POLICY_VERSION,
        "pass_height_threshold_meters": pass_height_threshold_meters(args),
        "num_workers": str(getattr(args, "num_workers", "1")),
        "worker_thread_limit": int(getattr(args, "worker_thread_limit", 1)),
        "graph_validation": graph_validation_enabled(getattr(args, "graph_validation", True)),
        "splits": ["train", "test"],
        **split_metadata(args),
        "split_manifest_id": split_manifest["manifest_id"],
        "split_manifest": split_manifest["metadata"],
        "return_types": args.return_types,
        "return_type": args.return_types[0] if len(args.return_types) == 1 else None,
        "commands": [step.command for step in command_steps],
        "status": "completed",
    }
    if bool(getattr(args, "use_carries", False)):
        metadata["carry_variant"] = {
            "available": True,
            "definition": CARRY_DEFINITION_VERSION,
            "validation_parameters": asdict(CarryPolicy()),
            "sidecar_root": "carry_sidecars",
            "artifact_suffix": "_carries",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    from datatools.reception_policy import KEY, RETIRED
    metadata[KEY] = RETIRED
    if getattr(args, "failed_pass_contact_detection", False):
        from datatools.failed_pass_trajectory import KEY as detector_key, VERSION, SETTINGS
        metadata[detector_key] = VERSION
        metadata["failed_pass_detector_settings"] = SETTINGS
    write_run_metadata(run_root, metadata)
    write_latest_run("feature", args.run_id)
    print(f"Feature run id: {args.run_id}")


def main() -> None:
    args = parse_args()
    if args.refresh_failed_pass_trajectories:
        from datatools.failed_pass_refresh import run_failed_pass_refresh
        run_failed_pass_refresh(args)
    elif args.refresh_trajectory and (args.retire_elastic or args.replace_carries):
        from datatools.feature_refresh import run_combined_refresh
        run_combined_refresh(args)
    elif args.retire_elastic and args.extend_feature_run_id:
        from datatools.reception_refresh import run_retirement
        run_retirement(args)
    elif args.replace_carries:
        from scripts.replace_carries import replace_carries
        replace_carries(args)
    elif args.refresh_trajectory:
        run_trajectory_refresh(args)
    elif args.extend_feature_run_id:
        if args.use_carries:
            run_carry_extension_generation(args)
        else:
            run_extension_generation(args)
    else:
        run_full_generation(args)


if __name__ == "__main__":
    main()
