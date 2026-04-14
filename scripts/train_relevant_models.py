from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_config import (
    ACTION_GRAPH_DIR,
    ACTION_GRAPH_INTENT_TRAIN_DIR,
    INTENDED_RECEIVER_MODE_ANGLE_ONLY,
    INTENDED_RECEIVER_MODE_MODEL,
    SUCCESS_INTENT_GRAPH_DIR,
    get_action_label_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_intent_train_label_dir,
    get_relevant_model_ids,
    resolve_intended_receiver_mode,
)

ROOT = Path(__file__).resolve().parents[1]


def trial_from_model_id(model_id: str) -> int:
    try:
        return int(str(model_id).rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid model id format: {model_id}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_xt", action="store_true", help="Train outcome models with xT targets instead of xG.")
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
    return parser.parse_args()


def base_gnn_args(feature_dir: str, label_dir: str, trial: int) -> list[str]:
    return [
        "--trial",
        str(trial),
        "--model",
        "gat",
        "--sparsify",
        "none",
        "--edge_in_dim",
        "2",
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
        "--feature_dir",
        feature_dir,
        "--label_dir",
        label_dir,
    ]


def intent_command(
    task: str,
    trial: int,
    feature_dir: str,
    label_dir: str,
    train_feature_dir: str,
    train_label_dir: str,
) -> list[str]:
    command = [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, trial),
        "--min_pass_dur",
        "0.5",
        "--possessor_aware",
        "--keeper_aware",
        "--ball_z_aware",
        "--poss_vel_aware",
        "--extend_features",
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
    return command


def success_intent_command(task: str, trial: int, feature_dir: str, label_dir: str) -> list[str]:
    return [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, trial),
        "--min_pass_dur",
        "0.5",
        "--possessor_aware",
        "--keeper_aware",
        "--ball_z_aware",
        "--poss_vel_aware",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]


def pass_success_command(task: str, trial: int, feature_dir: str, label_dir: str, ipw_model_id: str) -> list[str]:
    return [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, trial),
        "--ipw_model_id",
        ipw_model_id,
        "--min_pass_dur",
        "0.5",
        "--possessor_aware",
        "--keeper_aware",
        "--ball_z_aware",
        "--poss_vel_aware",
        "--extend_features",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]


def outcome_command(task: str, trial: int, feature_dir: str, label_dir: str, use_xt: bool) -> list[str]:
    return [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, trial),
        "--keeper_aware",
        "--return_type",
        "disc_0.9",
        "--lambda_l1",
        "1e-6",
        "--start_lr",
        "0.0002",
        "--min_lr",
        "1e-5",
        "--use_xt" if use_xt else "--use_xg",
    ]


def failure_receiver_command(task: str, trial: int, feature_dir: str, label_dir: str) -> list[str]:
    return [
        "--task",
        task,
        *base_gnn_args(feature_dir, label_dir, trial),
        "--augment_blocks",
        "--shot_success",
        "unblocked",
        "--possessor_aware",
        "--keeper_aware",
        "--ball_z_aware",
        "--poss_vel_aware",
        "--extend_features",
        "--lambda_l1",
        "0.0001",
        "--start_lr",
        "0.002",
        "--min_lr",
        "1e-5",
    ]


def build_training_commands(args: argparse.Namespace) -> list[list[str]]:
    mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    model_ids = get_relevant_model_ids(intended_receiver_mode=mode, use_xt=args.use_xt)
    if args.outcome_scoring_trial is not None:
        model_ids["outcome_scoring"] = f"outcome_scoring/{int(args.outcome_scoring_trial):02d}"
    if args.outcome_conceding_trial is not None:
        model_ids["outcome_conceding"] = f"outcome_conceding/{int(args.outcome_conceding_trial):02d}"

    base_feature_dir = str(ACTION_GRAPH_DIR)
    success_intent_label_mode = (
        INTENDED_RECEIVER_MODE_ANGLE_ONLY
        if args.success_intent_only and mode == INTENDED_RECEIVER_MODE_MODEL
        else mode
    )
    base_label_dir = str(get_action_label_dir("disc_0.9", intended_receiver_mode=success_intent_label_mode))
    intent_train_feature_dir = str(ACTION_GRAPH_INTENT_TRAIN_DIR)
    intent_train_label_dir = str(get_intent_train_label_dir("disc_0.9", intended_receiver_mode=mode))
    success_intent_feature_dir = str(SUCCESS_INTENT_GRAPH_DIR)
    augmented_feature_dir = str(get_augmented_feature_dir(intended_receiver_mode=mode))
    augmented_label_dir = str(get_augmented_label_dir(intended_receiver_mode=mode))

    commands = []
    if args.success_intent_only:
        return [
            success_intent_command(
                "success_intent",
                trial_from_model_id(model_ids["success_intent"]),
                success_intent_feature_dir,
                base_label_dir,
            )
        ]

    commands.extend(
        [
            intent_command(
                "pass_intent",
                trial_from_model_id(model_ids["pass_intent"]),
                base_feature_dir,
                base_label_dir,
                intent_train_feature_dir,
                intent_train_label_dir,
            ),
            intent_command(
                "action_intent",
                trial_from_model_id(model_ids["action_intent"]),
                base_feature_dir,
                base_label_dir,
                intent_train_feature_dir,
                intent_train_label_dir,
            ),
            pass_success_command(
                "pass_success",
                trial_from_model_id(model_ids["pass_success"]),
                base_feature_dir,
                base_label_dir,
                model_ids["pass_intent"],
            ),
            outcome_command(
                "outcome_scoring",
                trial_from_model_id(model_ids["outcome_scoring"]),
                base_feature_dir,
                base_label_dir,
                args.use_xt,
            ),
            outcome_command(
                "outcome_conceding",
                trial_from_model_id(model_ids["outcome_conceding"]),
                base_feature_dir,
                base_label_dir,
                args.use_xt,
            ),
            failure_receiver_command(
                "failure_receiver",
                trial_from_model_id(model_ids["failure_receiver"]),
                augmented_feature_dir,
                augmented_label_dir,
            ),
        ]
    )
    return commands


def main() -> None:
    cli_args = parse_args()
    python = sys.executable

    for args in build_training_commands(cli_args):
        command = [python, "train.py", *args]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
