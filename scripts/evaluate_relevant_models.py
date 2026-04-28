from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.utils import resolve_model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--diagnostic-feature-run-id")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_tasks = [
        "action_intent",
        "pass_intent",
        "pass_success",
        "outcome_scoring",
        "outcome_conceding",
    ]
    resolved_model_ids, _, bundle = resolve_model_selection(
        required_tasks=required_tasks,
        bundle_id=args.bundle_id,
        explicit_model_ids={
            "action_intent": args.action_intent_model_id,
            "pass_intent": args.pass_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        require_feature_run_id=False,
    )
    python = sys.executable
    model_ids = [
        resolved_model_ids["action_intent"],
        resolved_model_ids["pass_intent"],
        resolved_model_ids["pass_success"],
        resolved_model_ids["outcome_scoring"],
        resolved_model_ids["outcome_conceding"],
    ]
    success_intent_model_id = args.success_intent_model_id
    if not success_intent_model_id and bundle is not None:
        success_intent_model_id = bundle.get("model_ids", {}).get("success_intent")
    if success_intent_model_id:
        model_ids.insert(3, str(success_intent_model_id))

    for model_id in model_ids:
        command = [python, "test.py", "--model_id", model_id, "--device", args.device]
        if args.diagnostic_feature_run_id:
            command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
