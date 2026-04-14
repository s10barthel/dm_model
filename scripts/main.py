from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (
    EVENT_SYNCED_DIR,
    INTENDED_RECEIVER_MODE_MODEL,
    XT_DIR,
    XT_MATCH_DIR,
    get_component_dir,
    get_relevant_model_ids,
    resolve_intended_receiver_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scoped DEFCON pipeline described in README.md without visualization steps."
    )
    parser.add_argument("--use_xt", action="store_true", help="Use xT for the outcome models instead of xG.")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip scripts/preprocess_sportec.py.")
    parser.add_argument("--skip-xt", action="store_true", help="Skip scripts/generate_xt.py.")
    parser.add_argument("--skip-features", action="store_true", help="Skip scripts/generate_relevant_features.py.")
    parser.add_argument("--skip-train", action="store_true", help="Skip scripts/train_relevant_models.py.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip scripts/evaluate_relevant_models.py.")
    parser.add_argument("--skip-run-relevant", action="store_true", help="Skip scripts/run_relevant_models.py.")
    parser.add_argument("--skip-hawkeye", action="store_true", help="Skip scripts/run_hawkeye.py.")
    parser.add_argument("--skip-skillcorner", action="store_true", help="Skip scripts/run_skillcorner.py.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite supported preprocessing and xT artifacts.")
    parser.add_argument(
        "--relevant-split",
        choices=["train", "test", "all"],
        default="test",
        help="Split passed to scripts/run_relevant_models.py.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device passed to evaluation and inference scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: list[str], dry_run: bool) -> None:
    print("Running:" if not dry_run else "Dry-run:", format_command(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def validate_xt_artifacts(require_match_sidecars: bool) -> None:
    missing_paths = [
        path
        for path in [XT_DIR / "xT.csv", XT_DIR / "xT_grid.csv", XT_DIR / "fit_metadata.json"]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(ROOT)) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing required xT artifacts: {missing}. Run scripts/generate_xt.py or rerun without --skip-xt."
        )

    if not require_match_sidecars:
        return

    synced_match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    missing_sidecars = [match_id for match_id in synced_match_ids if not (XT_MATCH_DIR / f"{match_id}.csv").exists()]
    if missing_sidecars:
        preview = ", ".join(missing_sidecars[:5])
        if len(missing_sidecars) > 5:
            preview += f", ... ({len(missing_sidecars)} total)"
        raise FileNotFoundError(
            "Missing per-match xT sidecars for synced events: "
            f"{preview}. Run scripts/generate_xt.py or rerun without --skip-xt."
        )


def maybe_validate_xt_skip(args: argparse.Namespace) -> None:
    if not args.use_xt or not args.skip_xt or args.dry_run:
        return

    needs_xt_artifacts = any(
        not skipped
        for skipped in [
            args.skip_features,
            args.skip_train,
            args.skip_evaluate,
            args.skip_run_relevant,
            args.skip_hawkeye,
            args.skip_skillcorner,
        ]
    )
    if not needs_xt_artifacts:
        return

    require_match_sidecars = not args.skip_features or not args.skip_run_relevant
    validate_xt_artifacts(require_match_sidecars=require_match_sidecars)


def append_mode_flags(command: list[str], args: argparse.Namespace, include_xt: bool = False) -> list[str]:
    command = list(command)
    if include_xt and args.use_xt:
        command.append("--use_xt")
    if args.use_original_intended_receiver:
        command.append("--use-original-intended-receiver")
    if args.use_intended_receiver_model:
        command.append("--use-intended-receiver-model")
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    model_ids = get_relevant_model_ids(intended_receiver_mode=intended_receiver_mode, use_xt=args.use_xt)
    relevant_output_dir = get_component_dir(args.use_xt, intended_receiver_mode)
    skillcorner_output_dir = relevant_output_dir / "skillcorner"
    commands: list[list[str]] = []

    if not args.skip_preprocess:
        preprocess_command = [python, "scripts/preprocess_sportec.py"]
        if args.overwrite:
            preprocess_command.append("--overwrite")
        commands.append(preprocess_command)

    if args.use_xt and not args.skip_xt:
        xt_command = [python, "scripts/generate_xt.py"]
        if args.overwrite:
            xt_command.append("--overwrite")
        commands.append(xt_command)

    if not args.skip_features:
        if intended_receiver_mode == INTENDED_RECEIVER_MODE_MODEL and not args.skip_train:
            commands.append([python, "scripts/generate_relevant_features.py"])
        else:
            commands.append(append_mode_flags([python, "scripts/generate_relevant_features.py"], args))

    if not args.skip_train:
        if intended_receiver_mode == INTENDED_RECEIVER_MODE_MODEL:
            commands.append([python, "scripts/train_relevant_models.py", "--success-intent-only"])
            if not args.skip_features:
                commands.append(append_mode_flags([python, "scripts/generate_relevant_features.py"], args))
        commands.append(append_mode_flags([python, "scripts/train_relevant_models.py"], args, include_xt=True))

    if not args.skip_evaluate:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/evaluate_relevant_models.py",
                "--action-intent-model-id",
                model_ids["action_intent"],
                "--pass-success-model-id",
                model_ids["pass_success"],
                "--outcome-scoring-model-id",
                model_ids["outcome_scoring"],
                "--outcome-conceding-model-id",
                model_ids["outcome_conceding"],
                "--device",
                args.device,
                ],
                args,
                include_xt=True,
            )
        )

    if not args.skip_run_relevant:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/run_relevant_models.py",
                "--split",
                args.relevant_split,
                "--device",
                args.device,
                "--action-intent-model-id",
                model_ids["action_intent"],
                "--pass-success-model-id",
                model_ids["pass_success"],
                "--outcome-scoring-model-id",
                model_ids["outcome_scoring"],
                "--outcome-conceding-model-id",
                model_ids["outcome_conceding"],
                "--output-dir",
                str(relevant_output_dir),
                ],
                args,
                include_xt=True,
            )
        )

    if not args.skip_hawkeye:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/run_hawkeye.py",
                "--device",
                args.device,
                "--action-intent-model-id",
                model_ids["action_intent"],
                "--pass-success-model-id",
                model_ids["pass_success"],
                "--outcome-scoring-model-id",
                model_ids["outcome_scoring"],
                "--outcome-conceding-model-id",
                model_ids["outcome_conceding"],
                "--output-dir",
                str(relevant_output_dir),
                ],
                args,
                include_xt=True,
            )
        )

    if not args.skip_skillcorner:
        commands.append(
            append_mode_flags(
                [
                python,
                "scripts/run_skillcorner.py",
                "--device",
                args.device,
                "--action-intent-model-id",
                model_ids["action_intent"],
                "--pass-success-model-id",
                model_ids["pass_success"],
                "--outcome-scoring-model-id",
                model_ids["outcome_scoring"],
                "--outcome-conceding-model-id",
                model_ids["outcome_conceding"],
                "--output-dir",
                str(skillcorner_output_dir),
                ],
                args,
                include_xt=True,
            )
        )

    return commands


def main() -> None:
    args = parse_args()
    maybe_validate_xt_skip(args)
    commands = build_commands(args)

    if not commands:
        print("No pipeline stages selected.")
        return

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
