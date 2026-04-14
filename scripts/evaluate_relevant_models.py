from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from project_config import get_relevant_model_ids, resolve_intended_receiver_mode

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_xt", action="store_true")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    default_model_ids = get_relevant_model_ids(intended_receiver_mode=intended_receiver_mode, use_xt=args.use_xt)
    python = sys.executable
    model_ids = [
        args.action_intent_model_id or default_model_ids["action_intent"],
        args.pass_success_model_id or default_model_ids["pass_success"],
        args.outcome_scoring_model_id or default_model_ids["outcome_scoring"],
        args.outcome_conceding_model_id or default_model_ids["outcome_conceding"],
    ]
    for model_id in model_ids:
        command = [python, "test.py", "--model_id", model_id, "--device", args.device]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
