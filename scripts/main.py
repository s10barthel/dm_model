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
    EPV_DIR,
    EPV_MATCH_DIR,
    EVENT_SYNCED_DIR,
    GOAL_DISTANCE_DIR,
    GOAL_DISTANCE_MATCH_DIR,
    PROJECT_ROOT,
    XT_DIR,
    XT_MATCH_DIR,
    generate_run_id,
    validate_return_type_for_target_family,
)
from models.utils import normalize_v_edge_feature_mode, use_v_edge_features_for_mode

TRAINING_WRAPPER_FEATURE_DEFAULTS = {
    "xy_only": False,
    "possessor_aware": True,
    "keeper_aware": True,
    "ball_z_aware": True,
    "poss_vel_aware": True,
    "poss_rel_vel_aware": False,
    "offside_aware": True,
    "extend_features": False,
}

WRAPPER_OVERRIDE_FLAGS = {
    "xy_only": ("--xy-only", "--no-xy-only"),
    "possessor_aware": ("--possessor-aware", "--no-possessor-aware"),
    "keeper_aware": ("--keeper-aware", "--no-keeper-aware"),
    "ball_z_aware": ("--ball-z-aware", "--no-ball-z-aware"),
    "poss_vel_aware": ("--poss-vel-aware", "--no-poss-vel-aware"),
    "poss_rel_vel_aware": ("--poss-rel-vel-aware", "--no-poss-rel-vel-aware"),
    "offside_aware": ("--offside", "--no-offside"),
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
        name: TRAINING_WRAPPER_FEATURE_DEFAULTS[name] if getattr(args, name, None) is None else bool(getattr(args, name))
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
    parser.add_argument(
        "--target-family",
        choices=["goal", "xg", "xt", "goal_distance", "epv"],
        default=None,
        help="Outcome target family for retained outcome models.",
    )
    parser.add_argument(
        "--return_type",
        default=None,
        help=(
            "Resolved return type to generate and train against: disc_<gamma>, disc_<gamma>_skip1, "
            "next_<N>, next_<N>_skip1, or in_<N> (xt/goal_distance/epv only)."
        ),
    )
    parser.add_argument(
        "--intended-receiver-mode",
        choices=["original", "angle_only", "model"],
        default=None,
        help="Intended-receiver variant for retained-model training.",
    )
    parser.add_argument(
        "--intended-receiver-model-id",
        default=None,
        help="Pinned success_intent checkpoint used to add the model-backed intended-receiver variant during feature generation.",
    )
    parser.add_argument("--feature-run-id", default=None, help="Explicit feature-run id to reuse or assign.")
    parser.add_argument("--bundle-id", default=None, help="Explicit model bundle id to reuse or assign.")
    parser.add_argument(
        "--success-intent-model-id",
        default=None,
        help="Optional success_intent checkpoint for evaluation; defaults to --intended-receiver-model-id when present.",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip scripts/preprocess_sportec.py.")
    parser.add_argument("--skip-xt", action="store_true", help="Skip scripts/generate_xt.py.")
    parser.add_argument("--skip-goal-distance", action="store_true", help="Skip scripts/generate_goal_distance.py.")
    parser.add_argument("--skip-epv", action="store_true", help="Skip scripts/generate_epv.py.")
    parser.add_argument("--skip-features", action="store_true", help="Skip scripts/generate_relevant_features.py.")
    parser.add_argument("--skip-train", action="store_true", help="Skip scripts/train_relevant_models.py.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip scripts/evaluate_relevant_models.py.")
    parser.add_argument("--skip-run-relevant", action="store_true", help="Skip scripts/run_relevant_models.py.")
    parser.add_argument("--skip-hawkeye", action="store_true", help="Skip scripts/run_hawkeye.py.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip scripts/run_benchmark.py.")
    parser.add_argument("--skip-skillcorner", action="store_true", help="Skip scripts/run_skillcorner.py.")
    parser.add_argument(
        "--benchmark-input-dir",
        default=str(PROJECT_ROOT / "benchmark"),
        help="Benchmark data root passed to scripts/run_benchmark.py.",
    )
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
        "Train downstream models with the ball possessor's own velocity features.",
        "Disable the ball possessor's own velocity features for downstream training.",
    )
    add_bool_override(
        parser,
        "poss-rel-vel-aware",
        "poss_rel_vel_aware",
        "Train downstream models with player velocity relative to the ball possessor's velocity.",
        "Disable player velocity relative to the ball possessor's velocity for downstream training.",
    )
    add_bool_override(
        parser,
        "offside",
        "offside_aware",
        "Train downstream models with the is_offside node feature.",
        "Disable the is_offside node feature for downstream training.",
    )
    add_bool_override(
        parser,
        "extend-features",
        "extend_features",
        "Enable the extended handcrafted node features for downstream training.",
        "Disable the extended handcrafted node features for downstream training.",
    )
    parser.set_defaults(v_edge_feature_mode="all")
    args = parser.parse_args()
    args.v_edge_feature_mode = normalize_v_edge_feature_mode(args.v_edge_feature_mode)
    args.use_v_edge_features = use_v_edge_features_for_mode(args.v_edge_feature_mode)
    if not args.skip_train:
        try:
            resolve_training_feature_overrides(args)
        except ValueError as exc:
            parser.error(str(exc))

    args.success_intent_model_id = args.success_intent_model_id or args.intended_receiver_model_id
    needs_training_config = not args.skip_train
    needs_feature_generation = not args.skip_features
    needs_bundle = args.skip_train and any(
        not skipped
        for skipped in [
            args.skip_evaluate,
            args.skip_run_relevant,
            args.skip_hawkeye,
            args.skip_benchmark,
            args.skip_skillcorner,
        ]
    )

    if needs_training_config:
        if not args.target_family:
            parser.error("--target-family is required unless --skip-train is set.")
        if not args.return_type:
            parser.error("--return_type is required unless --skip-train is set.")
        if not args.intended_receiver_mode:
            parser.error("--intended-receiver-mode is required unless --skip-train is set.")

    if needs_feature_generation and not args.return_type:
        parser.error("--return_type is required when scripts/main.py generates feature artifacts.")

    if args.target_family and args.return_type:
        try:
            args.return_type = validate_return_type_for_target_family(args.return_type, target_family=args.target_family)
        except ValueError as exc:
            parser.error(str(exc))

    if args.target_family == "epv" and not args.skip_epv and not args.bundle_id:
        parser.error("--bundle-id is required when scripts/main.py generates EPV artifacts.")

    if args.skip_features and not args.skip_train and not args.feature_run_id:
        parser.error("--feature-run-id is required when --skip-features is set and training is still enabled.")

    if args.intended_receiver_mode == "model" and needs_feature_generation and not args.intended_receiver_model_id:
        parser.error(
            "--intended-receiver-model-id is required to generate a feature run with intended_receiver_mode=model."
        )

    if needs_bundle and not args.bundle_id:
        parser.error(
            "--bundle-id is required when --skip-train is set but downstream evaluation or inference stages are enabled."
        )

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
        for path in [
            XT_DIR / "xT.csv",
            XT_DIR / "xT_grid.csv",
            XT_DIR / "xT_source_grid.csv",
            XT_DIR / "xT_xy_surface.csv",
            XT_DIR / "xT_glm_fit_sample.csv",
            XT_DIR / "xT_xy_surface_3d.png",
            XT_DIR / "fit_metadata.json",
        ]
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
    if args.target_family != "xt" or not args.skip_xt or args.dry_run:
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
    if args.target_family != "goal_distance" or not args.skip_goal_distance or args.dry_run:
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


def validate_epv_artifacts(require_match_sidecars: bool) -> None:
    missing_paths = [path for path in [EPV_DIR / "epv.csv", EPV_DIR / "metadata.json"] if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(ROOT)) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing required EPV artifacts: {missing}. Run scripts/generate_epv.py or rerun without --skip-epv."
        )

    if not require_match_sidecars:
        return

    synced_match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    missing_sidecars = [match_id for match_id in synced_match_ids if not (EPV_MATCH_DIR / f"{match_id}.csv").exists()]
    if missing_sidecars:
        preview = ", ".join(missing_sidecars[:5])
        if len(missing_sidecars) > 5:
            preview += f", ... ({len(missing_sidecars)} total)"
        raise FileNotFoundError(
            "Missing per-match EPV sidecars for synced events: "
            f"{preview}. Run scripts/generate_epv.py or rerun without --skip-epv."
        )


def maybe_validate_epv_skip(args: argparse.Namespace) -> None:
    if args.target_family != "epv" or not args.skip_epv or args.dry_run:
        return

    needs_epv_artifacts = any(
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
    if not needs_epv_artifacts:
        return

    require_match_sidecars = not args.skip_features or not args.skip_run_relevant
    validate_epv_artifacts(require_match_sidecars=require_match_sidecars)


def append_training_target_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    if args.target_family:
        command.extend(["--target-family", args.target_family])
    if args.intended_receiver_mode:
        command.extend(["--intended-receiver-mode", args.intended_receiver_mode])
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


def append_bundle_flag(command: list[str], bundle_id: str | None) -> list[str]:
    command = list(command)
    if bundle_id:
        command.extend(["--bundle-id", bundle_id])
    return command


def append_training_feature_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    for name, (enabled_flag, disabled_flag) in WRAPPER_OVERRIDE_FLAGS.items():
        value = getattr(args, name, None)
        if value is True:
            command.append(enabled_flag)
        elif value is False:
            command.append(disabled_flag)
    return command


def append_edge_feature_flag(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    mode = normalize_v_edge_feature_mode(
        getattr(args, "v_edge_feature_mode", None),
        use_v_edge_features=getattr(args, "use_v_edge_features", None),
    )
    if mode == "none":
        command.append("--no-v-edge-features")
    elif mode == "no_poss":
        command.append("--v-edge-features-no-poss")
    else:
        command.append("--v-edge-features")
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    commands: list[list[str]] = []
    feature_run_id = args.feature_run_id if args.skip_features else (args.feature_run_id or generate_run_id("feature"))
    bundle_id = args.bundle_id if args.skip_train else (args.bundle_id or generate_run_id("model_bundle"))

    if not args.skip_preprocess:
        preprocess_command = [python, "scripts/preprocess_sportec.py"]
        if args.overwrite:
            preprocess_command.append("--overwrite")
        commands.append(preprocess_command)

    if args.target_family == "xt" and not args.skip_xt:
        xt_command = [python, "scripts/generate_xt.py"]
        if args.overwrite:
            xt_command.append("--overwrite")
        commands.append(xt_command)
    if args.target_family == "goal_distance" and not args.skip_goal_distance:
        goal_distance_command = [python, "scripts/generate_goal_distance.py"]
        if args.overwrite:
            goal_distance_command.append("--overwrite")
        commands.append(goal_distance_command)
    if args.target_family == "epv" and not args.skip_epv:
        epv_command = [python, "scripts/generate_epv.py", "--bundle-id", args.bundle_id]
        if args.overwrite:
            epv_command.append("--overwrite")
        commands.append(epv_command)

    if not args.skip_features:
        feature_command = append_return_type_flag(
            [python, "scripts/generate_relevant_features.py", "--run-id", feature_run_id],
            args,
        )
        if args.intended_receiver_model_id:
            feature_command.extend(["--intended-receiver-model-id", args.intended_receiver_model_id])
        commands.append(feature_command)

    if not args.skip_train:
        train_command = [
            python,
            "scripts/train_relevant_models.py",
            "--bundle-id",
            bundle_id,
            "--feature-run-id",
            feature_run_id,
        ]
        train_command = append_training_target_flags(train_command, args)
        train_command = append_return_type_flag(train_command, args)
        train_command = append_edge_feature_flag(train_command, args)
        train_command = append_training_feature_flags(train_command, args)
        commands.append(train_command)

    if not args.skip_evaluate:
        evaluate_command = append_bundle_flag(
            [
                python,
                "scripts/evaluate_relevant_models.py",
                "--device",
                args.device,
            ],
            bundle_id,
        )
        if args.success_intent_model_id:
            evaluate_command.extend(["--success-intent-model-id", args.success_intent_model_id])
        commands.append(evaluate_command)

    if not args.skip_run_relevant:
        commands.append(
            append_bundle_flag(
                [
                    python,
                    "scripts/run_relevant_models.py",
                    "--split",
                    args.relevant_split,
                    "--device",
                    args.device,
                ],
                bundle_id,
            )
        )

    if not args.skip_hawkeye:
        commands.append(
            append_bundle_flag(
                [
                    python,
                    "scripts/run_hawkeye.py",
                    "--device",
                    args.device,
                ],
                bundle_id,
            )
        )

    if not args.skip_benchmark:
        benchmark_command = append_bundle_flag(
            [
                python,
                "scripts/run_benchmark.py",
                "--input-dir",
                args.benchmark_input_dir,
                "--device",
                args.device,
            ],
            bundle_id,
        )
        commands.append(benchmark_command)

    if not args.skip_skillcorner:
        commands.append(
            append_bundle_flag(
                [
                    python,
                    "scripts/run_skillcorner.py",
                    "--device",
                    args.device,
                ],
                bundle_id,
            )
        )

    return commands


def main() -> None:
    args = parse_args()
    maybe_validate_xt_skip(args)
    maybe_validate_goal_distance_skip(args)
    maybe_validate_epv_skip(args)
    commands = build_commands(args)

    if not commands:
        print("No pipeline stages selected.")
        return

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
