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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (
    INTENDED_RECEIVER_MODE_MODEL,
    generate_run_id,
    get_action_label_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_feature_run_root,
    get_intent_train_label_dir,
    get_resolved_action_dir,
    load_feature_run_metadata,
    resolve_generation_intended_receiver_modes,
    resolve_requested_return_types,
    resolve_feature_run_id,
    validate_intended_receiver_mode,
    write_latest_run,
    write_run_metadata,
)

EXPECTED_GRAPH_SCHEMA = {"edge_in_dim": 4, "add_v_edge_features": True}


@dataclass(frozen=True)
class FeatureExtensionPlan:
    base_run_id: str
    output_run_id: str
    base_metadata: dict[str, Any]
    base_return_types: list[str]
    final_return_types: list[str]
    added_return_types: list[str]
    base_intended_receiver_modes: list[str]
    final_intended_receiver_modes: list[str]
    added_intended_receiver_modes: list[str]
    intended_receiver_model_id: str | None
    regenerate_model_mode: bool
    replaced_intended_receiver_model_id: str | None
    replaced_intended_receiver_modes: list[str]
    commands: list[list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--return_type",
        action="append",
        default=None,
        help=(
            "Resolved return type for generated action labels. Repeat the flag to include multiple return types, "
            "including disc_<gamma>_skip1 and next_<N>_skip1, plus in_<N> for xt/goal_distance training."
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
            "generating only newly requested return types or the model intended-receiver variant."
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
    args = parser.parse_args()
    args.requested_return_types = resolve_requested_return_types(args.return_type) if args.return_type else []
    args.return_types = args.requested_return_types or resolve_requested_return_types(None)
    args.intended_receiver_modes = resolve_generation_intended_receiver_modes(args.intended_receiver_model_id)
    return args


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def with_mode_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    for return_type in args.return_types:
        command.extend(["--return_type", return_type])
    if args.intended_receiver_model_id:
        command.extend(["--intended-receiver-model-id", args.intended_receiver_model_id])
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    return command


def full_generation_commands(python: str) -> list[list[str]]:
    return [
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
        [
            python,
            "datatools/graph_feature.py",
            "--action_type",
            "all",
            "--split",
            "test",
            "--post_action",
        ],
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
    ]


def normalize_graph_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    return {
        "edge_in_dim": int(schema.get("edge_in_dim", -1)),
        "add_v_edge_features": bool(schema.get("add_v_edge_features", False)),
    }


def metadata_return_types(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("return_types")
    if isinstance(values, list) and values:
        return resolve_requested_return_types(values)
    legacy_value = metadata.get("return_type")
    if legacy_value:
        return resolve_requested_return_types([str(legacy_value)])
    raise ValueError("Base feature run metadata does not record any return_types.")


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
) -> list[list[str]]:
    commands: list[list[str]] = []
    non_model_base_modes = [mode for mode in base_modes if mode != INTENDED_RECEIVER_MODE_MODEL]
    if regenerate_model_mode and not non_model_base_modes:
        raise ValueError("Regenerating model-mode artifacts requires at least one non-model base mode.")

    source_mode = non_model_base_modes[0] if non_model_base_modes else base_modes[0]
    source_return_type = base_return_types[0]

    if added_return_types:
        added_return_modes = non_model_base_modes if regenerate_model_mode else base_modes
        for split in ["train", "test"]:
            commands.append(
                extension_graph_feature_command(
                    python,
                    output_run_id,
                    split=split,
                    return_types=added_return_types,
                    intended_receiver_modes=added_return_modes,
                    intended_receiver_model_id=intended_receiver_model_id,
                )
            )
        commands.append(
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
            )
        )

    if INTENDED_RECEIVER_MODE_MODEL in added_modes or regenerate_model_mode:
        validation_modes = [source_mode, INTENDED_RECEIVER_MODE_MODEL]
        for split in ["train", "test"]:
            commands.append(
                extension_graph_feature_command(
                    python,
                    output_run_id,
                    split=split,
                    return_types=final_return_types,
                    intended_receiver_modes=validation_modes,
                    intended_receiver_model_id=intended_receiver_model_id,
                    augment_blocks_from_existing_graphs=(split == "train"),
                )
            )
        commands.append(
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
            )
        )
    return commands


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
    if graph_schema != EXPECTED_GRAPH_SCHEMA:
        raise ValueError(
            f"Feature run {base_run_id} has graph_schema={graph_schema!r}; expected {EXPECTED_GRAPH_SCHEMA!r}."
        )

    base_return_types = metadata_return_types(base_metadata)
    base_modes = metadata_intended_receiver_modes(base_metadata)
    final_return_types, added_return_types = union_preserving_order(base_return_types, args.requested_return_types)

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

    if not added_return_types and not added_modes and not regenerate_model_mode:
        raise ValueError(
            "Extension would not add any return types or intended-receiver modes. "
            "Use a fresh full run or request a new --return_type / --intended-receiver-model-id."
        )

    output_run_id = str(args.run_id) if args.run_id else generate_run_id("feature")
    if output_run_id == base_run_id:
        raise ValueError("--run-id for an extension must name a new feature run, not the base run.")
    output_root = get_feature_run_root(output_run_id)
    if output_root.exists():
        raise FileExistsError(f"Derived feature run {output_run_id} already exists at {output_root}.")

    commands = extension_commands_for_plan(
        python=python,
        output_run_id=output_run_id,
        base_return_types=base_return_types,
        final_return_types=final_return_types,
        added_return_types=added_return_types,
        base_modes=base_modes,
        added_modes=added_modes,
        intended_receiver_model_id=final_model_id,
        regenerate_model_mode=regenerate_model_mode,
    )
    return FeatureExtensionPlan(
        base_run_id=base_run_id,
        output_run_id=output_run_id,
        base_metadata=copy.deepcopy(base_metadata),
        base_return_types=base_return_types,
        final_return_types=final_return_types,
        added_return_types=added_return_types,
        base_intended_receiver_modes=base_modes,
        final_intended_receiver_modes=final_modes,
        added_intended_receiver_modes=added_modes,
        intended_receiver_model_id=final_model_id,
        regenerate_model_mode=regenerate_model_mode,
        replaced_intended_receiver_model_id=replaced_model_id,
        replaced_intended_receiver_modes=replaced_modes,
        commands=commands,
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
        "extension_replaced_intended_receiver_model_id": plan.replaced_intended_receiver_model_id,
        "extension_replaced_intended_receiver_modes": plan.replaced_intended_receiver_modes,
        "extension_commands": plan.commands,
        "intended_receiver_modes": plan.final_intended_receiver_modes,
        "intended_receiver_model_id": plan.intended_receiver_model_id,
        "graph_schema": EXPECTED_GRAPH_SCHEMA.copy(),
        "splits": ["train", "test"],
        "return_types": plan.final_return_types,
        "return_type": plan.final_return_types[0] if len(plan.final_return_types) == 1 else None,
        "base_feature_run_metadata": plan.base_metadata,
        "status": status,
    }
    if error:
        metadata["error"] = error
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


def run_extension_generation(args: argparse.Namespace) -> None:
    plan = build_extension_plan(args)
    output_root = copy_base_feature_run(plan.base_run_id, plan.output_run_id)
    write_run_metadata(output_root, derived_metadata(args, plan, "in_progress"))
    try:
        if plan.regenerate_model_mode:
            remove_model_mode_artifacts(output_root, plan.final_return_types)
        for command in plan.commands:
            run_command(command)
    except Exception as exc:
        write_run_metadata(output_root, derived_metadata(args, plan, "failed", error=str(exc)))
        raise

    write_run_metadata(output_root, derived_metadata(args, plan, "completed"))
    write_latest_run("feature", plan.output_run_id)
    print(f"Derived feature run id: {plan.output_run_id}")
    print(f"Derived from feature run id: {plan.base_run_id}")


def run_full_generation(args: argparse.Namespace) -> None:
    args.run_id = args.run_id or generate_run_id("feature")
    python = sys.executable
    commands = full_generation_commands(python)
    for command in commands:
        run_command(with_mode_flags(command, args))

    run_root = get_feature_run_root(args.run_id)
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "intended_receiver_modes": args.intended_receiver_modes,
        "intended_receiver_model_id": args.intended_receiver_model_id,
        "graph_schema": {"edge_in_dim": 4, "add_v_edge_features": True},
        "splits": ["train", "test"],
        "return_types": args.return_types,
        "return_type": args.return_types[0] if len(args.return_types) == 1 else None,
        "commands": [with_mode_flags(command, args) for command in commands],
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
