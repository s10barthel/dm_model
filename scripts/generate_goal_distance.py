from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from datatools.goal_distance import annotate_match_goal_distance
from project_config import (
    EVENT_SYNCED_DIR,
    GOAL_DISTANCE_DIR,
    GOAL_DISTANCE_MATCH_DIR,
    ensure_project_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-id",
        action="append",
        help="Restrict export generation to one or more match ids.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N available matches.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing goal-distance outputs.")
    return parser.parse_args()


def load_events(match_id: str) -> pd.DataFrame:
    return pd.read_csv(EVENT_SYNCED_DIR / f"{match_id}.csv", parse_dates=["utc_timestamp"])


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def resolve_match_ids(requested_match_ids: list[str] | None, limit: int | None) -> list[str]:
    match_ids = sorted(path.stem for path in EVENT_SYNCED_DIR.glob("*.csv"))
    if requested_match_ids:
        requested = set(requested_match_ids)
        match_ids = [match_id for match_id in match_ids if match_id in requested]
    if limit is not None:
        match_ids = match_ids[:limit]
    if not match_ids:
        raise ValueError("No synced event files were selected for goal-distance generation.")
    return match_ids


def collect_export_matches(match_ids: list[str]) -> tuple[list[pd.DataFrame], list[str], list[dict[str, str]]]:
    all_events: list[pd.DataFrame] = []
    export_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []

    for match_id in match_ids:
        try:
            all_events.append(load_events(match_id))
            export_match_ids.append(match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not all_events:
        raise ValueError("No usable synced event files remained for goal-distance export generation.")

    return all_events, export_match_ids, skipped_matches


def save_export_outputs(
    all_events: list[pd.DataFrame],
    output_dir: Path,
    match_dir: Path,
) -> tuple[list[str], list[dict[str, str]], list[pd.DataFrame]]:
    processed_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []
    exported_actions: list[pd.DataFrame] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    for events in all_events:
        try:
            annotated_events, exported_goal_distance = annotate_match_goal_distance(events)
            match_id = str(
                annotated_events["stats_perform_match_id"].iloc[0]
                if "stats_perform_match_id" in annotated_events.columns
                else annotated_events["game_id"].iloc[0]
            )
            sidecar = annotated_events[
                ["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]
            ].copy()
            sidecar.to_csv(match_dir / f"{match_id}.csv", index=False)
            exported_actions.append(exported_goal_distance)
            processed_match_ids.append(match_id)
        except Exception as exc:
            match_id = "<unknown>"
            if "stats_perform_match_id" in events.columns and not events.empty:
                match_id = str(events["stats_perform_match_id"].iloc[0])
            elif "game_id" in events.columns and not events.empty:
                match_id = str(events["game_id"].iloc[0])
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not processed_match_ids:
        raise ValueError("No usable synced event files remained for goal-distance export writing.")

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "goal_distance.csv", index=False)
    else:
        pd.DataFrame(
            columns=["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]
        ).to_csv(output_dir / "goal_distance.csv", index=False)

    return processed_match_ids, skipped_matches, exported_actions


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_csv = GOAL_DISTANCE_DIR / "goal_distance.csv"
    output_metadata = GOAL_DISTANCE_DIR / "metadata.json"
    if not args.overwrite and output_csv.exists() and output_metadata.exists():
        print(f"Goal-distance outputs already exist in {GOAL_DISTANCE_DIR}. Use --overwrite to rebuild them.")
        return

    match_ids = resolve_match_ids(args.match_id, args.limit)
    all_events, _, skipped_export_matches = collect_export_matches(match_ids)
    processed_export_ids, skipped_export_write_matches, _ = save_export_outputs(
        all_events,
        GOAL_DISTANCE_DIR,
        GOAL_DISTANCE_MATCH_DIR,
    )

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "export_match_ids": processed_export_ids,
        "eligible_action_types": ["pass", "cross", "shot"],
        "target_name": "goal_distance",
        "target_range": [0.0, 100.0],
        "goal_center_xy": [105.0, 34.0],
        "max_raw_distance": float((105.0**2 + 34.0**2) ** 0.5),
        "skipped_export_matches": skipped_export_matches + skipped_export_write_matches,
    }
    (GOAL_DISTANCE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_skipped_export = len(skipped_export_matches) + len(skipped_export_write_matches)
    if total_skipped_export:
        print(f"Skipped {total_skipped_export} export matches while generating goal-distance outputs.")
    print(f"Saved goal-distance outputs to {GOAL_DISTANCE_DIR}")


if __name__ == "__main__":
    main()
