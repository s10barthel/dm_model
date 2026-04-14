from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.utils import resolve_relevant_model_ids
from project_config import resolve_intended_receiver_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_xt", action="store_true")
    parser.add_argument("--use_goal_distance", action="store_true")
    parser.add_argument("--feature-run-id", default=None)
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.use_xt and args.use_goal_distance:
        parser.error("--use_xt and --use_goal_distance are mutually exclusive.")
    return args


def main() -> None:
    args = parse_args()
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    default_model_ids = resolve_relevant_model_ids(
        intended_receiver_mode=intended_receiver_mode,
        use_xt=args.use_xt,
        use_goal_distance=args.use_goal_distance,
        explicit_model_ids={
            "action_intent": args.action_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
    )
    python = sys.executable
    model_ids = [
        default_model_ids["action_intent"],
        default_model_ids["pass_success"],
        default_model_ids["outcome_scoring"],
        default_model_ids["outcome_conceding"],
    ]
    for model_id in model_ids:
        command = [python, "test.py", "--model_id", model_id, "--device", args.device]
        if args.feature_run_id:
            command.extend(["--feature-run-id", str(args.feature_run_id)])
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
