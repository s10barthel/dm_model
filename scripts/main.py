from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import COMPONENT_DIR, EVENT_SYNCED_DIR, XT_DIR, XT_MATCH_DIR

XG_MODEL_IDS = {
    "action_intent": "action_intent/00",
    "pass_success": "pass_success/20",
    "outcome_scoring": "outcome_scoring/20",
    "outcome_conceding": "outcome_conceding/20",
}
XT_MODEL_IDS = {
    "action_intent": "action_intent/00",
    "pass_success": "pass_success/20",
    "outcome_scoring": "outcome_scoring/21",
    "outcome_conceding": "outcome_conceding/21",
}
XT_COMPONENT_DIR = COMPONENT_DIR.with_name(f"{COMPONENT_DIR.name}_xt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scoped DEFCON pipeline described in README.md without visualization steps."
    )
    parser.add_argument("--use_xt", action="store_true", help="Use xT for the outcome models instead of xG.")
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


def get_model_ids(use_xt: bool) -> dict[str, str]:
    return XT_MODEL_IDS if use_xt else XG_MODEL_IDS


def get_component_dirs(use_xt: bool) -> tuple[Path, Path]:
    component_dir = XT_COMPONENT_DIR if use_xt else COMPONENT_DIR
    return component_dir, component_dir / "skillcorner"


def trial_from_model_id(model_id: str) -> int:
    try:
        return int(model_id.rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid model id format: {model_id}") from exc


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


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    model_ids = get_model_ids(args.use_xt)
    outcome_scoring_trial = trial_from_model_id(model_ids["outcome_scoring"])
    outcome_conceding_trial = trial_from_model_id(model_ids["outcome_conceding"])
    relevant_output_dir, skillcorner_output_dir = get_component_dirs(args.use_xt)
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
        commands.append([python, "scripts/generate_relevant_features.py"])

    if not args.skip_train:
        train_command = [
            python,
            "scripts/train_relevant_models.py",
            "--outcome-scoring-trial",
            str(outcome_scoring_trial),
            "--outcome-conceding-trial",
            str(outcome_conceding_trial),
        ]
        if args.use_xt:
            train_command.append("--use_xt")
        commands.append(train_command)

    if not args.skip_evaluate:
        commands.append(
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
            ]
        )

    if not args.skip_run_relevant:
        commands.append(
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
            ]
        )

    if not args.skip_hawkeye:
        commands.append(
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
            ]
        )

    if not args.skip_skillcorner:
        commands.append(
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
            ]
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
