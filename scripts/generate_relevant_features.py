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
    resolve_generation_intended_receiver_modes,
    resolve_requested_return_types,
    write_latest_run,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--return_type",
        action="append",
        default=None,
        help="Resolved return type for generated action labels. Repeat the flag to include multiple return types.",
    )
    parser.add_argument(
        "--intended-receiver-model-id",
        default=None,
        help="Optional success_intent checkpoint used to add the model-backed intended-receiver variant.",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    args.return_types = resolve_requested_return_types(args.return_type)
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


if __name__ == "__main__":
    main()
