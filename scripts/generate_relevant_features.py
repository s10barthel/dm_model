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
    generate_run_id,
    get_feature_run_root,
    resolve_intended_receiver_mode,
    write_latest_run,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--intended-receiver-model-id", default="success_intent/00")
    parser.add_argument("--add_v_edge_features", action="store_true")
    parser.add_argument("--run-id", default=None)
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
    if args.add_v_edge_features:
        command.append("--add_v_edge_features")
    if args.run_id:
        command.extend(["--run-id", args.run_id])
    return command


def main() -> None:
    args = parse_args()
    args.run_id = args.run_id or generate_run_id("feature")
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

    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    run_root = get_feature_run_root(args.run_id)
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "intended_receiver_mode": intended_receiver_mode,
        "intended_receiver_model_id": args.intended_receiver_model_id,
        "add_v_edge_features": bool(args.add_v_edge_features),
        "splits": ["train", "test"],
        "return_type": "disc_0.9",
        "commands": [with_mode_flags(command, args) for command in commands],
        "status": "completed",
    }
    write_run_metadata(run_root, metadata)
    write_latest_run("feature", args.run_id)
    print(f"Feature run id: {args.run_id}")


if __name__ == "__main__":
    main()
