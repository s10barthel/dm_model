from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (
    EVENT_SYNCED_DIR,
    GOAL_DISTANCE_DIR,
    GOAL_DISTANCE_MATCH_DIR,
    INTENDED_RECEIVER_MODE_MODEL,
    XT_DIR,
    XT_MATCH_DIR,
    generate_run_id,
    infer_target_family,
    resolve_effective_return_type,
    resolve_intended_receiver_mode,
)

TRAINING_WRAPPER_FEATURE_DEFAULTS = {
    "xy_only": False,
    "possessor_aware": True,
    "keeper_aware": True,
    "ball_z_aware": True,
    "poss_vel_aware": True,
    "extend_features": False,
}

WRAPPER_OVERRIDE_FLAGS = {
    "xy_only": ("--xy-only", "--no-xy-only"),
    "possessor_aware": ("--possessor-aware", "--no-possessor-aware"),
    "keeper_aware": ("--keeper-aware", "--no-keeper-aware"),
    "ball_z_aware": ("--ball-z-aware", "--no-ball-z-aware"),
    "poss_vel_aware": ("--poss-vel-aware", "--no-poss-vel-aware"),
    "extend_features": ("--extend-features", "--no-extend-features"),
}


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


def resolve_training_feature_overrides(args: argparse.Namespace) -> dict[str, bool]:
    resolved_flags = {
        name: TRAINING_WRAPPER_FEATURE_DEFAULTS[name] if getattr(args, name) is None else bool(getattr(args, name))
        for name in TRAINING_WRAPPER_FEATURE_DEFAULTS
    }
    if not resolved_flags["possessor_aware"] and resolved_flags["extend_features"]:
        raise ValueError(
            "--extend-features requires possessor-aware features; remove --extend-features or enable --possessor-aware."
        )
    return resolved_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scoped DEFCON pipeline described in README.md without visualization steps."
    )
    parser.add_argument("--use_xg", action="store_true", help="Use xG for the outcome models instead of binary goals.")
    parser.add_argument("--use_xt", action="store_true", help="Use xT for the outcome models instead of binary goals.")
    parser.add_argument(
        "--use_goal_distance",
        action="store_true",
        help="Use goal-distance labels for the outcome models instead of binary goals.",
    )
    parser.add_argument("--return_type", default=None, help="Resolved return type for generated labels and outcome training.")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip scripts/preprocess_sportec.py.")
    parser.add_argument("--skip-xt", action="store_true", help="Skip scripts/generate_xt.py.")
    parser.add_argument("--skip-goal-distance", action="store_true", help="Skip scripts/generate_goal_distance.py.")
    parser.add_argument("--skip-features", action="store_true", help="Skip scripts/generate_relevant_features.py.")
    parser.add_argument("--skip-train", action="store_true", help="Skip scripts/train_relevant_models.py.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip scripts/evaluate_relevant_models.py.")
    parser.add_argument("--skip-run-relevant", action="store_true", help="Skip scripts/run_relevant_models.py.")
    parser.add_argument("--skip-hawkeye", action="store_true", help="Skip scripts/run_hawkeye.py.")
    parser.add_argument("--skip-skillcorner", action="store_true", help="Skip scripts/run_skillcorner.py.")
    parser.add_argument("--add_v_edge_features", action="store_true", help="Append velocity-angle edge features during feature generation.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite supported preprocessing and target-artifact outputs.",
    )
    parser.add_argument(
        "--relevant-split",
        choices=["train", "test", "all"],
        default="test",
        help="Split passed to scripts/run_relevant_models.py.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device passed to evaluation and inference scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    add_bool_override(
        parser,
        "xy-only",
        "xy_only",
        "Train downstream models with xy-only node features.",
        "Disable xy-only node features for downstream training.",
    )
    add_bool_override(
        parser,
        "possessor-aware",
        "possessor_aware",
        "Train downstream models with possessor-awareness features.",
        "Disable possessor-awareness features for downstream training.",
    )
    add_bool_override(
        parser,
        "keeper-aware",
        "keeper_aware",
        "Train downstream models with keeper/goal awareness features.",
        "Disable keeper/goal awareness features for downstream training.",
    )
    add_bool_override(
        parser,
        "ball-z-aware",
        "ball_z_aware",
        "Train downstream models with ball-height features.",
        "Disable ball-height features for downstream training.",
    )
    add_bool_override(
        parser,
        "poss-vel-aware",
        "poss_vel_aware",
        "Train downstream models with possessor-velocity relation features.",
        "Disable possessor-velocity relation features for downstream training.",
    )
    add_bool_override(
        parser,
        "extend-features",
        "extend_features",
        "Enable the extended handcrafted node features for downstream training.",
        "Disable the extended handcrafted node features for downstream training.",
    )
    args = parser.parse_args()
    try:
        args.target_family = infer_target_family(
            use_xg=bool(args.use_xg),
            use_xt=bool(args.use_xt),
            use_goal_distance=bool(args.use_goal_distance),
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.return_type = resolve_effective_return_type(args.target_family, args.return_type)
    if not args.skip_train:
        try:
            resolve_training_feature_overrides(args)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: list[str], dry_run: bool) -> None:
    print("Running:" if not dry_run else "Dry-run:", format_command(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def validate_xt_artifacts(require_match_sidecars: bool) -> None:
    missing_paths = [
        path
        for path in [XT_DIR / "xT.csv", XT_DIR / "xT_grid.csv", XT_DIR / "fit_metadata.json"]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(ROOT)) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing required xT artifacts: {missing}. Run scripts/generate_xt.py or rerun without --skip-xt."
        )

    if not require_match_sidecars:
        return

    synced_match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    missing_sidecars = [match_id for match_id in synced_match_ids if not (XT_MATCH_DIR / f"{match_id}.csv").exists()]
    if missing_sidecars:
        preview = ", ".join(missing_sidecars[:5])
        if len(missing_sidecars) > 5:
            preview += f", ... ({len(missing_sidecars)} total)"
        raise FileNotFoundError(
            "Missing per-match xT sidecars for synced events: "
            f"{preview}. Run scripts/generate_xt.py or rerun without --skip-xt."
        )


def maybe_validate_xt_skip(args: argparse.Namespace) -> None:
    if not args.use_xt or not args.skip_xt or args.dry_run:
        return

    needs_xt_artifacts = any(
        not skipped
        for skipped in [
            args.skip_features,
            args.skip_train,
            args.skip_evaluate,
            args.skip_run_relevant,
            args.skip_hawkeye,
            args.skip_skillcorner,
        ]
    )
    if not needs_xt_artifacts:
        return

    require_match_sidecars = not args.skip_features or not args.skip_run_relevant
    validate_xt_artifacts(require_match_sidecars=require_match_sidecars)


def validate_goal_distance_artifacts(require_match_sidecars: bool) -> None:
    missing_paths = [path for path in [GOAL_DISTANCE_DIR / "goal_distance.csv", GOAL_DISTANCE_DIR / "metadata.json"] if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(ROOT)) for path in missing_paths)
        raise FileNotFoundError(
            "Missing required goal-distance artifacts: "
            f"{missing}. Run scripts/generate_goal_distance.py or rerun without --skip-goal-distance."
        )

    if not require_match_sidecars:
        return

    synced_match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    missing_sidecars = [match_id for match_id in synced_match_ids if not (GOAL_DISTANCE_MATCH_DIR / f"{match_id}.csv").exists()]
    if missing_sidecars:
        preview = ", ".join(missing_sidecars[:5])
        if len(missing_sidecars) > 5:
            preview += f", ... ({len(missing_sidecars)} total)"
        raise FileNotFoundError(
            "Missing per-match goal-distance sidecars for synced events: "
            f"{preview}. Run scripts/generate_goal_distance.py or rerun without --skip-goal-distance."
        )


def maybe_validate_goal_distance_skip(args: argparse.Namespace) -> None:
    if not args.use_goal_distance or not args.skip_goal_distance or args.dry_run:
        return

    needs_goal_distance_artifacts = any(
        not skipped
        for skipped in [
            args.skip_features,
            args.skip_train,
            args.skip_evaluate,
            args.skip_run_relevant,
            args.skip_hawkeye,
            args.skip_skillcorner,
        ]
    )
    if not needs_goal_distance_artifacts:
        return

    require_match_sidecars = not args.skip_features or not args.skip_run_relevant
    validate_goal_distance_artifacts(require_match_sidecars=require_match_sidecars)


def append_mode_flags(command: list[str], args: argparse.Namespace, include_target: bool = False) -> list[str]:
    command = list(command)
    if include_target and args.use_xg:
        command.append("--use_xg")
    if include_target and args.use_xt:
        command.append("--use_xt")
    if include_target and args.use_goal_distance:
        command.append("--use_goal_distance")
    if args.use_original_intended_receiver:
        command.append("--use-original-intended-receiver")
    if args.use_intended_receiver_model:
        command.append("--use-intended-receiver-model")
    return command


def append_return_type_flag(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    if args.return_type:
        command.extend(["--return_type", args.return_type])
    return command


def append_feature_run_flag(command: list[str], feature_run_id: str | None) -> list[str]:
    command = list(command)
    if feature_run_id:
        command.extend(["--feature-run-id", feature_run_id])
    return command


def append_training_feature_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    for name, (enabled_flag, disabled_flag) in WRAPPER_OVERRIDE_FLAGS.items():
        value = getattr(args, name)
        if value is True:
            command.append(enabled_flag)
        elif value is False:
            command.append(disabled_flag)
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    commands: list[list[str]] = []
    preparatory_feature_run_id: str | None = None
    main_feature_run_id: str | None = None

    if not args.skip_preprocess:
        preprocess_command = [python, "scripts/preprocess_sportec.py"]
        if args.overwrite:
            preprocess_command.append("--overwrite")
        commands.append(preprocess_command)

    if args.use_xt and not args.skip_xt:
        xt_command = [python, "scripts/generate_xt.py"]
        if args.overwrite:
            xt_command.append("--overwrite")
        commands.append(xt_command)
    if args.use_goal_distance and not args.skip_goal_distance:
        goal_distance_command = [python, "scripts/generate_goal_distance.py"]
        if args.overwrite:
            goal_distance_command.append("--overwrite")
        commands.append(goal_distance_command)

    if not args.skip_features:
        if intended_receiver_mode == INTENDED_RECEIVER_MODE_MODEL and not args.skip_train:
            preparatory_feature_run_id = generate_run_id("feature")
            feature_command = append_return_type_flag(
                [python, "scripts/generate_relevant_features.py", "--run-id", preparatory_feature_run_id],
                args,
            )
            if args.add_v_edge_features:
                feature_command.append("--add_v_edge_features")
            commands.append(feature_command)
        else:
            main_feature_run_id = generate_run_id("feature")
            feature_command = append_return_type_flag(
                append_mode_flags(
                    [python, "scripts/generate_relevant_features.py", "--run-id", main_feature_run_id],
                    args,
                ),
                args,
            )
            if args.add_v_edge_features:
                feature_command.append("--add_v_edge_features")
            commands.append(feature_command)

    if not args.skip_train:
        if intended_receiver_mode == INTENDED_RECEIVER_MODE_MODEL:
            commands.append(
                append_training_feature_flags(
                    append_feature_run_flag(
                        append_return_type_flag([python, "scripts/train_relevant_models.py", "--success-intent-only"], args),
                        preparatory_feature_run_id,
                    ),
                    args,
                )
            )
            if not args.skip_features:
                main_feature_run_id = generate_run_id("feature")
                feature_command = append_return_type_flag(
                    append_mode_flags(
                        [python, "scripts/generate_relevant_features.py", "--run-id", main_feature_run_id],
                        args,
                    ),
                    args,
                )
                if args.add_v_edge_features:
                    feature_command.append("--add_v_edge_features")
                commands.append(feature_command)
        commands.append(
            append_training_feature_flags(
                append_feature_run_flag(
                    append_return_type_flag(
                        append_mode_flags([python, "scripts/train_relevant_models.py"], args, include_target=True),
                        args,
                    ),
                    main_feature_run_id,
                ),
                args,
            )
        )

    if not args.skip_evaluate:
        commands.append(
            append_feature_run_flag(
                append_mode_flags(
                    [
                    python,
                    "scripts/evaluate_relevant_models.py",
                    "--device",
                    args.device,
                    ],
                    args,
                    include_target=True,
                ),
                main_feature_run_id,
            )
        )

    if not args.skip_run_relevant:
        commands.append(
            append_feature_run_flag(
                append_mode_flags(
                    [
                    python,
                    "scripts/run_relevant_models.py",
                    "--split",
                    args.relevant_split,
                    "--device",
                    args.device,
                    ],
                    args,
                    include_target=True,
                ),
                main_feature_run_id,
            )
        )

    if not args.skip_hawkeye:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/run_hawkeye.py",
                "--device",
                args.device,
                ],
                args,
                include_target=True,
            )
        )

    if not args.skip_skillcorner:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/run_skillcorner.py",
                "--device",
                args.device,
                ],
                args,
                include_target=True,
            )
        )

    return commands


def main() -> None:
    args = parse_args()
    maybe_validate_xt_skip(args)
    maybe_validate_goal_distance_skip(args)
    commands = build_commands(args)

    if not commands:
        print("No pipeline stages selected.")
        return

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
