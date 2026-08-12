from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.utils import resolve_model_selection


def parse_bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--pass-height-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--diagnostic-feature-run-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weighted-pass-success-metrics", action="store_true")
    parser.add_argument("--pc-xpass-cache-dir", default=None)
    parser.add_argument("--discount", type=parse_bool_text, default=True)
    parser.add_argument("--v4-power", type=float, default=4.0)
    parser.add_argument("--v4-zero", type=float, default=0.7)
    return parser.parse_args(argv)


def add_weighted_pass_success_options(
    command: list[str], args: argparse.Namespace, pass_height_model_id: str,
) -> list[str]:
    """Append the evaluation-only v4 options to a pass-success test command."""
    command.extend(
        [
            "--weighted-pass-success-metrics",
            "--pass-height-model-id",
            str(pass_height_model_id),
            "--discount",
            str(bool(args.discount)).lower(),
            "--v4-power",
            str(args.v4_power),
            "--v4-zero",
            str(args.v4_zero),
        ]
    )
    if args.pc_xpass_cache_dir:
        command.extend(["--pc-xpass-cache-dir", str(args.pc_xpass_cache_dir)])
    return command


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
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
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
    pass_height_model_id = getattr(args, "pass_height_model_id", None)
    if not pass_height_model_id and bundle is not None:
        pass_height_model_id = bundle.get("model_ids", {}).get("pass_height")
    if pass_height_model_id:
        model_ids.insert(4 if success_intent_model_id else 3, str(pass_height_model_id))
    if args.weighted_pass_success_metrics and not pass_height_model_id:
        raise ValueError(
            "--weighted-pass-success-metrics requires --pass-height-model-id or a bundle containing pass_height."
        )

    for model_id in model_ids:
        command = [python, "test.py", "--model_id", model_id, "--device", args.device]
        if args.diagnostic_feature_run_id:
            command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
        if args.weighted_pass_success_metrics and model_id == resolved_model_ids["pass_success"]:
            add_weighted_pass_success_options(command, args, str(pass_height_model_id))
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
