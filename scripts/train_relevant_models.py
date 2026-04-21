from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

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
    get_success_intent_graph_dir,
    get_success_intent_label_dir,
    infer_feature_run_intended_receiver_modes,
    infer_feature_run_return_types,
    load_feature_run_metadata,
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
    "accel_aware": True,
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

LOW_LEVEL_BOOL_OVERRIDE_FLAGS = {
    "accel_aware": ("--accel", "--no-accel"),
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
    for name, (enabled_flag, disabled_flag) in LOW_LEVEL_BOOL_OVERRIDE_FLAGS.items():
        command.append(enabled_flag if feature_flags[name] else disabled_flag)
    return command


def append_edge_feature_flag(command: list[str], use_v_edge_features: bool) -> list[str]:
    command = list(command)
    command.append("--v-edge-features" if use_v_edge_features else "--no-v-edge-features")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-family",
        choices=["goal", "xg", "xt", "goal_distance"],
        default=None,
        help="Outcome target family for the retained outcome models.",
    )
    parser.add_argument(
        "--return_type",
        default=None,
        help="Resolved outcome return type to use for label generation: disc_<gamma>, next_<N>, or in_<N> (xt/goal_distance only).",
    )
    parser.add_argument("--feature-run-id", default=None, help="Pinned feature-artifact run id.")
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
        dest="use_v_edge_features",
        action="store_true",
        help="Use the stored velocity-angle edge features during training.",
    )
    edge_feature_group.add_argument(
        "--no-v-edge-features",
        dest="use_v_edge_features",
        action="store_false",
        help="Ignore the stored velocity-angle edge features during training.",
    )
    parser.set_defaults(use_v_edge_features=True)
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
        "accel",
        "accel_aware",
        "Include player-acceleration node features during training.",
        "Disable player-acceleration node features during training.",
    )
    add_bool_override(
        parser,
        "extend-features",
        "extend_features",
        "Enable the extended handcrafted node features during training.",
        "Disable the extended handcrafted node features during training.",
    )
    args = parser.parse_args(argv)
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

    if args.success_intent_only:
        if args.intended_receiver_mode:
            parser.error("--success-intent-only is mode-independent and does not accept --intended-receiver-mode.")
        args.intended_receiver_mode = None
        if args.target_family is not None:
            parser.error("--success-intent-only does not accept --target-family.")
        if args.return_type is not None:
            args.return_type = validate_return_type(args.return_type)
        elif available_return_types:
            args.return_type = available_return_types[0]
        else:
            parser.error(f"Feature run {args.feature_run_id} does not expose any return types.")
    else:
        if not args.intended_receiver_mode:
            parser.error("--intended-receiver-mode is required.")
        args.intended_receiver_mode = validate_intended_receiver_mode(args.intended_receiver_mode)
        if not args.target_family:
            parser.error("--target-family is required unless --success-intent-only is set.")
        if not args.return_type:
            parser.error("--return_type is required unless --success-intent-only is set.")
        try:
            args.return_type = validate_return_type_for_target_family(args.return_type, target_family=args.target_family)
        except ValueError as exc:
            parser.error(str(exc))

    if not args.success_intent_only and args.intended_receiver_mode not in available_modes:
        parser.error(
            f"Feature run {args.feature_run_id} does not expose intended_receiver_mode={args.intended_receiver_mode!r}. "
            f"Available: {', '.join(available_modes) or 'none'}."
        )
    if args.return_type not in available_return_types:
        parser.error(
            f"Feature run {args.feature_run_id} does not expose return_type={args.return_type!r}. "
            f"Available: {', '.join(available_return_types) or 'none'}."
        )

    args.available_return_types = available_return_types
    args.available_intended_receiver_modes = available_modes
    return args


def base_gnn_args(
    feature_dir: str,
    label_dir: str,
    model_id: str,
    intended_receiver_mode: str | None,
    return_type: str,
    use_v_edge_features: bool,
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
        "256", #originally 512, reduced due to memory constraints on GPU; adjust as needed
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
    return append_edge_feature_flag(command, use_v_edge_features)


def intent_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    train_feature_dir: str,
    train_label_dir: str,
    intended_receiver_mode: str,
    return_type: str,
    use_v_edge_features: bool,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, use_v_edge_features),
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
    use_v_edge_features: bool,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, None, return_type, use_v_edge_features),
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
    ipw_model_id: str,
    intended_receiver_mode: str,
    return_type: str,
    use_v_edge_features: bool,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, use_v_edge_features),
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
    target_family: str,
    return_type: str,
    intended_receiver_mode: str,
    use_v_edge_features: bool,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, use_v_edge_features),
        "--lambda_l1",
        "1e-6",
        "--start_lr",
        "0.0002",
        "--min_lr",
        "1e-5",
    ]
    if target_family == "goal_distance":
        command.append("--use_goal_distance")
    elif target_family == "xt":
        command.append("--use_xt")
    elif target_family == "xg":
        command.append("--use_xg")
    return append_low_level_feature_flags(command, feature_flags)


def failure_receiver_command(
    task: str,
    model_id: str,
    feature_dir: str,
    label_dir: str,
    intended_receiver_mode: str,
    return_type: str,
    use_v_edge_features: bool,
    feature_flags: dict[str, bool],
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, model_id, intended_receiver_mode, return_type, use_v_edge_features),
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


def build_model_ids(args: argparse.Namespace, mode: str | None) -> dict[str, str]:
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
    return model_ids


def build_training_commands(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, str], str | None, str | None, dict[str, bool]]:
    mode = args.intended_receiver_mode
    target_family = args.target_family
    effective_return_type = args.return_type
    feature_flags = resolve_wrapper_feature_flags(args)
    resolved_feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(resolved_feature_run_id)
    model_ids = build_model_ids(args, mode)
    success_intent_feature_dir = str(get_success_intent_graph_dir(feature_root))
    success_intent_label_dir = str(get_success_intent_label_dir(root=feature_root))

    if args.success_intent_only:
        return (
            [
                success_intent_command(
                    "success_intent",
                    model_ids["success_intent"],
                    success_intent_feature_dir,
                    success_intent_label_dir,
                    effective_return_type,
                    bool(args.use_v_edge_features),
                    feature_flags,
                )
            ],
            {"success_intent": model_ids["success_intent"]},
            None,
            resolved_feature_run_id,
            feature_flags,
        )

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

    commands = []
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
                effective_return_type,
                bool(args.use_v_edge_features),
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
                effective_return_type,
                bool(args.use_v_edge_features),
                feature_flags,
            ),
            pass_success_command(
                "pass_success",
                model_ids["pass_success"],
                base_feature_dir,
                base_label_dir,
                model_ids["pass_intent"],
                mode,
                effective_return_type,
                bool(args.use_v_edge_features),
                feature_flags,
            ),
            outcome_command(
                "outcome_scoring",
                model_ids["outcome_scoring"],
                base_feature_dir,
                base_label_dir,
                target_family,
                effective_return_type,
                mode,
                bool(args.use_v_edge_features),
                feature_flags,
            ),
            outcome_command(
                "outcome_conceding",
                model_ids["outcome_conceding"],
                base_feature_dir,
                base_label_dir,
                target_family,
                effective_return_type,
                mode,
                bool(args.use_v_edge_features),
                feature_flags,
            ),
            failure_receiver_command(
                "failure_receiver",
                model_ids["failure_receiver"],
                augmented_feature_dir,
                augmented_label_dir,
                mode,
                effective_return_type,
                bool(args.use_v_edge_features),
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

    feature_run_metadata = load_feature_run_metadata(resolved_feature_run_id, required=False) or {}
    metadata = {
        "bundle_id": bundle_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "feature_run_id": resolved_feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "feature_run_intended_receiver_modes": cli_args.available_intended_receiver_modes,
        "feature_run_return_types": cli_args.available_return_types,
        "feature_run_intended_receiver_model_id": feature_run_metadata.get("intended_receiver_model_id"),
        "training_feature_flags": feature_flags,
        "target_family": cli_args.target_family,
        "return_type": cli_args.return_type,
        "use_v_edge_features": bool(cli_args.use_v_edge_features),
        "graph_schema": {
            "edge_in_dim": 4 if cli_args.use_v_edge_features else 2,
            "add_v_edge_features": bool(cli_args.use_v_edge_features),
        },
        "success_intent_only": bool(cli_args.success_intent_only),
        "success_intent_label_source": SUCCESS_INTENT_LABEL_SOURCE if cli_args.success_intent_only else None,
        "success_intent_training_filter": SUCCESS_INTENT_TRAINING_FILTER if cli_args.success_intent_only else None,
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
