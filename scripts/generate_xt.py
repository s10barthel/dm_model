from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from datatools.xt import (
    XT_GRID_L,
    XT_GRID_W,
    annotate_match_xt,
    build_xt_actions,
    fit_xt_surface,
    infer_home_team_id,
    rotate_xt_actions,
)
from project_config import EVENT_SYNCED_DIR, XT_DIR, XT_MATCH_DIR, ensure_project_dirs, load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", action="append", help="Restrict export generation to one or more match ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N available matches.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing xT outputs.")
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
        raise ValueError("No synced event files were selected for xT generation.")
    return match_ids


def fit_actions_for_train_split(train_ids: list[str]) -> tuple[pd.DataFrame, list[str], list[dict[str, str]]]:
    rotated_train_actions = []
    fit_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []

    for match_id in train_ids:
        event_path = EVENT_SYNCED_DIR / f"{match_id}.csv"
        if not event_path.exists():
            skipped_matches.append({"match_id": match_id, "error": "missing_synced_events"})
            continue

        try:
            events = load_events(match_id)
            xt_actions = build_xt_actions(events)
            if xt_actions.empty:
                skipped_matches.append({"match_id": match_id, "error": "no_eligible_actions"})
                continue
            rotated_train_actions.append(rotate_xt_actions(xt_actions, infer_home_team_id(events)))
            fit_match_ids.append(match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})
            continue

    if not rotated_train_actions:
        raise ValueError("No eligible pass/cross/shot actions were found in the training split for xT fitting.")

    return pd.concat(rotated_train_actions, ignore_index=True), fit_match_ids, skipped_matches


def collect_export_matches(match_ids: list[str]) -> tuple[list[pd.DataFrame], list[str], list[dict[str, str]]]:
    all_events: list[pd.DataFrame] = []
    export_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []

    for match_id in match_ids:
        try:
            events = load_events(match_id)
            xt_actions = build_xt_actions(events)
            if not xt_actions.empty:
                infer_home_team_id(events)
            all_events.append(events)
            export_match_ids.append(match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not all_events:
        raise ValueError("No usable synced event files remained for xT export generation.")

    return all_events, export_match_ids, skipped_matches


def save_export_outputs(
    all_events: list[pd.DataFrame],
    xt_grid,
    output_dir: Path,
    xt_match_dir: Path,
) -> tuple[list[str], list[dict[str, str]], list[pd.DataFrame]]:
    processed_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []
    exported_actions: list[pd.DataFrame] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    xt_match_dir.mkdir(parents=True, exist_ok=True)

    for events in all_events:
        try:
            annotated_events, exported_xt = annotate_match_xt(events, xt_grid)
            match_id = str(
                annotated_events["stats_perform_match_id"].iloc[0]
                if "stats_perform_match_id" in annotated_events.columns
                else annotated_events["game_id"].iloc[0]
            )
            sidecar = annotated_events[["action_id", "xG", "xT", "scores_xT", "concedes_xT"]].copy()
            sidecar.to_csv(xt_match_dir / f"{match_id}.csv", index=False)
            exported_actions.append(exported_xt)
            processed_match_ids.append(match_id)
        except Exception as exc:
            match_id = "<unknown>"
            if "stats_perform_match_id" in events.columns and not events.empty:
                match_id = str(events["stats_perform_match_id"].iloc[0])
            elif "game_id" in events.columns and not events.empty:
                match_id = str(events["game_id"].iloc[0])
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})

    if not processed_match_ids:
        raise ValueError("No usable synced event files remained for xT export writing.")

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "xT.csv", index=False)
    else:
        pd.DataFrame(columns=["action_id", "xG", "xT", "scores_xT", "concedes_xT"]).to_csv(output_dir / "xT.csv", index=False)

    pd.DataFrame(xt_grid, columns=[f"X{i}" for i in range(XT_GRID_L)]).to_csv(output_dir / "xT_grid.csv", index=False)
    return processed_match_ids, skipped_matches, exported_actions


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_csv = XT_DIR / "xT.csv"
    output_grid = XT_DIR / "xT_grid.csv"
    output_metadata = XT_DIR / "fit_metadata.json"
    if not args.overwrite and output_csv.exists() and output_grid.exists() and output_metadata.exists():
        print(f"xT outputs already exist in {XT_DIR}. Use --overwrite to rebuild them.")
        return

    manifest = load_split_manifest()
    train_ids = [match_id for match_id in manifest["train"] if (EVENT_SYNCED_DIR / f"{match_id}.csv").exists()]
    match_ids = resolve_match_ids(args.match_id, args.limit)
    all_events, _, skipped_export_matches = collect_export_matches(match_ids)

    rotated_train_actions, fit_match_ids, skipped_fit_matches = fit_actions_for_train_split(train_ids)
    xt_grid = fit_xt_surface(rotated_train_actions)

    processed_export_ids, skipped_export_write_matches, _ = save_export_outputs(all_events, xt_grid, XT_DIR, XT_MATCH_DIR)

    fit_metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "grid_l": XT_GRID_L,
        "grid_w": XT_GRID_W,
        "fit_match_ids": fit_match_ids,
        "export_match_ids": processed_export_ids,
        "eligible_action_types": ["pass", "cross", "shot"],
        "symmetry_pairs": [[1, 8], [2, 7], [3, 6], [4, 5]],
        "fit_samples": int(len(rotated_train_actions)),
        "skipped_fit_matches": skipped_fit_matches,
        "skipped_export_matches": skipped_export_matches + skipped_export_write_matches,
    }
    (XT_DIR / "fit_metadata.json").write_text(json.dumps(fit_metadata, indent=2), encoding="utf-8")
    if skipped_fit_matches:
        print(f"Skipped {len(skipped_fit_matches)} training matches while fitting xT.")
    total_skipped_export = len(skipped_export_matches) + len(skipped_export_write_matches)
    if total_skipped_export:
        print(f"Skipped {total_skipped_export} export matches while generating xT outputs.")
    print(f"Saved xT outputs to {XT_DIR}")


if __name__ == "__main__":
    main()
