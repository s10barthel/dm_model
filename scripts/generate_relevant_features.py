from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--intended-receiver-model-id", default="success_intent/00")
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def with_mode_flags(command: list[str], args: argparse.Namespace) -> list[str]:
    command = list(command)
    if args.use_original_intended_receiver:
        command.append("--use-original-intended-receiver")
    if args.use_intended_receiver_model:
        command.append("--use-intended-receiver-model")
    if args.intended_receiver_model_id:
        command.extend(["--intended-receiver-model-id", args.intended_receiver_model_id])
    return command


def main() -> None:
    args = parse_args()
    python = sys.executable
    commands = [
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
    for command in commands:
        run_command(with_mode_flags(command, args))


if __name__ == "__main__":
    main()
