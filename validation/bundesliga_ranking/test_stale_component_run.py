from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import bundesliga_ranking


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPONENT_RUNS_DIR = PROJECT_ROOT / "data" / "component_runs"
DEFAULT_EVENT_SYNCED_DIR = PROJECT_ROOT / "data" / "event_synced"
DEFAULT_TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the current stale component parquets to the modern identifier schema "
            "in a temporary directory, then run bundesliga_ranking.py against them."
        )
    )
    parser.add_argument("--source-component-run-id", default="1")
    parser.add_argument("--source-component-run-root", type=Path)
    parser.add_argument("--component-runs-dir", type=Path, default=DEFAULT_COMPONENT_RUNS_DIR)
    parser.add_argument("--event-synced-dir", type=Path, default=DEFAULT_EVENT_SYNCED_DIR)
    parser.add_argument("--season-prefix", default="DFL-MAT-J03")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEST_OUTPUT_DIR)
    parser.add_argument(
        "--converted-run-root",
        type=Path,
        help="Optional persistent output directory for converted parquets. Defaults to a temporary directory.",
    )
    return parser.parse_args(argv)


def resolve_source_root(args: argparse.Namespace) -> Path:
    if args.source_component_run_root is not None:
        source_root = args.source_component_run_root
    else:
        source_root = args.component_runs_dir / str(args.source_component_run_id)
    if not source_root.exists():
        raise FileNotFoundError(f"Source component run does not exist: {source_root}")
    return source_root


def read_events(event_synced_dir: Path, match_id: str) -> pd.DataFrame:
    event_path = event_synced_dir / f"{match_id}.csv"
    if not event_path.exists():
        raise FileNotFoundError(f"Missing event CSV for stale component conversion: {event_path}")
    events = pd.read_csv(event_path)
    bundesliga_ranking.validate_required_columns(
        events,
        bundesliga_ranking.COMPONENT_IDENTIFIER_COLUMNS,
        str(event_path),
    )
    return events


def modernize_component_table(table: pd.DataFrame, events: pd.DataFrame, label: str) -> pd.DataFrame:
    if all(column in table.columns for column in bundesliga_ranking.COMPONENT_IDENTIFIER_COLUMNS):
        return table.copy()
    if "action_id" not in table.columns:
        raise ValueError(f"{label} is missing both modern identifiers and legacy action_id.")

    legacy_indexes = pd.to_numeric(table["action_id"], errors="raise").astype(int)
    out_of_range = legacy_indexes[(legacy_indexes < 0) | (legacy_indexes >= len(events))]
    if not out_of_range.empty:
        sample = out_of_range.head(5).tolist()
        raise ValueError(f"{label} has legacy action indexes outside event row range: {sample}")

    identifiers = events.iloc[legacy_indexes][
        bundesliga_ranking.COMPONENT_IDENTIFIER_COLUMNS
    ].reset_index(drop=True)
    value_columns = table.drop(
        columns=[
            column
            for column in bundesliga_ranking.COMPONENT_IDENTIFIER_COLUMNS
            if column in table.columns
        ],
        errors="ignore",
    ).drop(columns=["action_id"], errors="ignore")
    return pd.concat([identifiers, value_columns.reset_index(drop=True)], axis=1)


def convert_match_dir(
    source_match_dir: Path,
    target_match_dir: Path,
    events: pd.DataFrame,
) -> None:
    target_match_dir.mkdir(parents=True, exist_ok=True)
    for component_name in bundesliga_ranking.REQUIRED_COMPONENTS:
        source_path = source_match_dir / f"{component_name}.parquet"
        if not source_path.exists() and component_name in {"pass_intent", "success_intent"}:
            source_path = source_match_dir / "pass_success.parquet"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source component for test conversion: {source_path}")

        table = pd.read_parquet(source_path)
        modern_table = modernize_component_table(table, events, str(source_path))
        modern_table.to_parquet(target_match_dir / f"{component_name}.parquet", index=False)


def convert_component_run(
    source_root: Path,
    target_root: Path,
    event_synced_dir: Path,
    season_prefix: str,
) -> list[str]:
    target_root.mkdir(parents=True, exist_ok=True)
    converted_match_ids: list[str] = []
    for source_match_dir in sorted(source_root.iterdir()):
        if not source_match_dir.is_dir() or not source_match_dir.name.startswith(season_prefix):
            continue
        match_id = source_match_dir.name
        events = read_events(event_synced_dir, match_id)
        convert_match_dir(source_match_dir, target_root / match_id, events)
        converted_match_ids.append(match_id)

    if not converted_match_ids:
        raise FileNotFoundError(
            f"No source match directories under {source_root} match {season_prefix!r}."
        )
    return converted_match_ids


def run_with_converted_root(args: argparse.Namespace, converted_root: Path) -> int:
    return bundesliga_ranking.main(
        [
            "--season-prefix",
            args.season_prefix,
            "--component-run-root",
            str(converted_root),
            "--event-synced-dir",
            str(args.event_synced_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = resolve_source_root(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.converted_run_root is not None:
        converted_root = args.converted_run_root
        if converted_root.exists():
            shutil.rmtree(converted_root)
        converted = convert_component_run(
            source_root,
            converted_root,
            args.event_synced_dir,
            args.season_prefix,
        )
        print(f"Converted {len(converted)} match(es) to {converted_root}")
        return run_with_converted_root(args, converted_root)

    with tempfile.TemporaryDirectory(prefix="bundesliga_ranking_components_") as tmpdir:
        converted_root = Path(tmpdir) / "component_run"
        converted = convert_component_run(
            source_root,
            converted_root,
            args.event_synced_dir,
            args.season_prefix,
        )
        print(f"Converted {len(converted)} match(es) to temporary component run {converted_root}")
        return run_with_converted_root(args, converted_root)


if __name__ == "__main__":
    raise SystemExit(main())
