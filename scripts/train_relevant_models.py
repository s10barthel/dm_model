from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (
    INTENDED_RECEIVER_MODE_ANGLE_ONLY,
    INTENDED_RECEIVER_MODE_MODEL,
    generate_model_run_id,
    generate_run_id,
    get_action_label_dir,
    get_action_graph_dir,
    get_action_graph_intent_train_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_intent_train_label_dir,
    get_model_bundle_root,
    get_success_intent_graph_dir,
    infer_target_family,
    resolve_feature_run_id,
    resolve_feature_root,
    resolve_intended_receiver_mode,
    write_run_metadata,
)

WRAPPER_FEATURE_DEFAULTS = {
    "xy_only": False,
    "possessor_aware": True,
    "keeper_aware": True,
    "ball_z_aware": True,
    "poss_vel_aware": True,
    "extend_features": False,
}

LOW_LEVEL_FEATURE_FLAGS = {
    "xy_only": "--xy_only",
    "possessor_aware": "--possessor_aware",
    "keeper_aware": "--keeper_aware",
    "ball_z_aware": "--ball_z_aware",
    "poss_vel_aware": "--poss_vel_aware",
    "extend_features": "--extend_features",
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


def resolve_wrapper_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    resolved_flags = {
        name: WRAPPER_FEATURE_DEFAULTS[name] if getattr(args, name) is None else bool(getattr(args, name))
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
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_xt", action="store_true", help="Train outcome models with xT targets instead of xG.")
    parser.add_argument(
        "--use_goal_distance",
        action="store_true",
        help="Train outcome models with goal-distance targets instead of xG.",
    )
    parser.add_argument("--feature-run-id", default=None, help="Pinned feature-artifact run id.")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument(
        "--success-intent-only",
        action="store_true",
        help="Only train the success_intent model. Useful for the first stage of learned intended-receiver mode.",
    )
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
        "Include possessor-velocity relation features during training.",
        "Disable possessor-velocity relation features during training.",
    )
    add_bool_override(
        parser,
        "extend-features",
        "extend_features",
        "Enable the extended handcrafted node features during training.",
        "Disable the extended handcrafted node features during training.",
    )
    args = parser.parse_args()
    if args.use_xt and args.use_goal_distance:
        parser.error("--use_xt and --use_goal_distance are mutually exclusive.")
    try:
        resolve_wrapper_feature_flags(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def base_gnn_args(feature_dir: str, label_dir: str, model_id: str, intended_receiver_mode: str) -> list[str]:
    _, run_id = str(model_id).split("/", 1)
    return [
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
        "512",
        "--print_freq",
        "50",
        "--seed",
        "100",
        "--intended-receiver-mode",
        intended_receiver_mode,
        "--feature_dir",
        feature_dir,
        "--label_dir",
        label_dir,
    ]


def intent_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    train_feature_dir: str,
    train_label_dir: str,
    intended_receiver_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode),
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
    intended_receiver_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode),
        "--min_pass_dur",
        "0.5",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]
    return append_low_level_feature_flags(command, feature_flags)


def pass_success_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    ipw_model_id: str,
    intended_receiver_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode),
        "--ipw_model_id",
        ipw_model_id,
        "--min_pass_dur",
        "0.5",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]
    return append_low_level_feature_flags(command, feature_flags)


def outcome_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    use_xt: bool,
    use_goal_distance: bool,
    intended_receiver_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    if use_xt and use_goal_distance:
        raise ValueError("use_xt and use_goal_distance are mutually exclusive.")
    target_flag = "--use_goal_distance" if use_goal_distance else ("--use_xt" if use_xt else "--use_xg")
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode),
        "--return_type",
        "disc_0.9",
        "--lambda_l1",
        "1e-6",
        "--start_lr",
        "0.0002",
        "--min_lr",
        "1e-5",
        target_flag,
    ]
    return append_low_level_feature_flags(command, feature_flags)


def failure_receiver_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    intended_receiver_mode: str,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode),
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


def build_model_ids(args: argparse.Namespace, mode: str) -> dict[str, str]:
    model_ids = {
        "success_intent": f"success_intent/{generate_model_run_id('success_intent')}",
        "pass_intent": f"pass_intent/{generate_model_run_id('pass_intent')}",
        "action_intent": f"action_intent/{generate_model_run_id('action_intent')}",
        "pass_success": f"pass_success/{generate_model_run_id('pass_success')}",
        "outcome_scoring": f"outcome_scoring/{generate_model_run_id('outcome_scoring')}",
        "outcome_conceding": f"outcome_conceding/{generate_model_run_id('outcome_conceding')}",
        "failure_receiver": f"failure_receiver/{generate_model_run_id('failure_receiver')}",
    }
    if args.outcome_scoring_trial is not None:
        model_ids["outcome_scoring"] = f"outcome_scoring/{int(args.outcome_scoring_trial):02d}"
    if args.outcome_conceding_trial is not None:
        model_ids["outcome_conceding"] = f"outcome_conceding/{int(args.outcome_conceding_trial):02d}"
    if args.success_intent_only and mode != INTENDED_RECEIVER_MODE_MODEL:
        model_ids["success_intent"] = f"success_intent/{generate_model_run_id('success_intent')}"
    return model_ids


def build_training_commands(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, str], str, str | None, dict[str, bool]]:
    mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    feature_flags = resolve_wrapper_feature_flags(args)
    resolved_feature_run_id = resolve_feature_run_id(args.feature_run_id, required=False)
    feature_root = resolve_feature_root(resolved_feature_run_id)
    model_ids = build_model_ids(args, mode)

    base_feature_dir = str(get_action_graph_dir(feature_root))
    success_intent_label_mode = (
        INTENDED_RECEIVER_MODE_ANGLE_ONLY
        if args.success_intent_only and mode == INTENDED_RECEIVER_MODE_MODEL
        else mode
    )
    base_label_dir = str(
        get_action_label_dir("disc_0.9", intended_receiver_mode=success_intent_label_mode, root=feature_root)
    )
    intent_train_feature_dir = str(get_action_graph_intent_train_dir(feature_root))
    intent_train_label_dir = str(get_intent_train_label_dir("disc_0.9", intended_receiver_mode=mode, root=feature_root))
    success_intent_feature_dir = str(get_success_intent_graph_dir(feature_root))
    augmented_feature_dir = str(get_augmented_feature_dir(intended_receiver_mode=mode, root=feature_root))
    augmented_label_dir = str(get_augmented_label_dir(intended_receiver_mode=mode, root=feature_root))

    commands = []
    if args.success_intent_only:
        return (
            [
                success_intent_command(
                    "success_intent",
                    model_ids["success_intent"],
                    success_intent_feature_dir,
                    base_label_dir,
                    success_intent_label_mode,
                    feature_flags,
                )
            ],
            {"success_intent": model_ids["success_intent"]},
            mode,
            resolved_feature_run_id,
            feature_flags,
        )

    commands.extend(
        [
            intent_command(
                "pass_intent",
                model_ids["pass_intent"],
                base_feature_dir,
                base_label_dir,
                intent_train_feature_dir,
                intent_train_label_dir,
                mode,
                feature_flags,
            ),
            intent_command(
                "action_intent",
                model_ids["action_intent"],
                base_feature_dir,
                base_label_dir,
                intent_train_feature_dir,
                intent_train_label_dir,
                mode,
                feature_flags,
            ),
            pass_success_command(
                "pass_success",
                model_ids["pass_success"],
                base_feature_dir,
                base_label_dir,
                model_ids["pass_intent"],
                mode,
                feature_flags,
            ),
            outcome_command(
                "outcome_scoring",
                model_ids["outcome_scoring"],
                base_feature_dir,
                base_label_dir,
                args.use_xt,
                args.use_goal_distance,
                mode,
                feature_flags,
            ),
            outcome_command(
                "outcome_conceding",
                model_ids["outcome_conceding"],
                base_feature_dir,
                base_label_dir,
                args.use_xt,
                args.use_goal_distance,
                mode,
                feature_flags,
            ),
            failure_receiver_command(
                "failure_receiver",
                model_ids["failure_receiver"],
                augmented_feature_dir,
                augmented_label_dir,
                mode,
                feature_flags,
            ),
        ]
    )
    return (
        commands,
        {
            key: model_ids[key]
            for key in [
                "pass_intent",
                "action_intent",
                "pass_success",
                "outcome_scoring",
                "outcome_conceding",
                "failure_receiver",
            ]
        },
        mode,
        resolved_feature_run_id,
        feature_flags,
    )


def main() -> None:
    cli_args = parse_args()
    python = sys.executable
    bundle_id = cli_args.bundle_id or generate_run_id("model_bundle")
    bundle_root = get_model_bundle_root(bundle_id)
    commands, bundle_model_ids, intended_receiver_mode, resolved_feature_run_id, feature_flags = build_training_commands(
        cli_args
    )
    executed_commands: list[list[str]] = []

    for args in commands:
        command = [python, "train.py"]
        if resolved_feature_run_id:
            command.extend(["--feature-run-id", str(resolved_feature_run_id)])
        command.extend(args)
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
        executed_commands.append(command)

    metadata = {
        "bundle_id": bundle_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "feature_run_id": resolved_feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "training_feature_flags": feature_flags,
        "use_xt": bool(cli_args.use_xt),
        "use_goal_distance": bool(cli_args.use_goal_distance),
        "target_family": infer_target_family(
            use_xg=not cli_args.use_xt and not cli_args.use_goal_distance,
            use_xt=bool(cli_args.use_xt),
            use_goal_distance=bool(cli_args.use_goal_distance),
        ),
        "success_intent_only": bool(cli_args.success_intent_only),
        "model_ids": bundle_model_ids,
        "commands": executed_commands,
        "status": "completed",
    }
    write_run_metadata(bundle_root, metadata)
    print(f"Model bundle id: {bundle_id}")
    print(f"Model bundle manifest: {bundle_root / 'metadata.json'}")
    for task, model_id in bundle_model_ids.items():
        print(f"{task}: {model_id}")


if __name__ == "__main__":
    main()
