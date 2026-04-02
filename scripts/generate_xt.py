from __future__ import annotations

import argparse
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
    build_xt_actions,
    fit_xt_surface,
    infer_home_team_id,
    rotate_xt_actions,
    save_xt_outputs,
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


def fit_actions_for_train_split(train_ids: list[str]) -> pd.DataFrame:
    rotated_train_actions = []

    for match_id in train_ids:
        event_path = EVENT_SYNCED_DIR / f"{match_id}.csv"
        if not event_path.exists():
            continue

        events = load_events(match_id)
        xt_actions = build_xt_actions(events)
        if xt_actions.empty:
            continue

        rotated_train_actions.append(rotate_xt_actions(xt_actions, infer_home_team_id(events)))

    if not rotated_train_actions:
        raise ValueError("No eligible pass/cross/shot actions were found in the training split for xT fitting.")

    return pd.concat(rotated_train_actions, ignore_index=True)


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
    all_events = [load_events(match_id) for match_id in match_ids]

    rotated_train_actions = fit_actions_for_train_split(train_ids)
    xt_grid = fit_xt_surface(rotated_train_actions)

    fit_metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "grid_l": XT_GRID_L,
        "grid_w": XT_GRID_W,
        "fit_match_ids": train_ids,
        "export_match_ids": match_ids,
        "eligible_action_types": ["pass", "cross", "shot"],
        "symmetry_pairs": [[1, 8], [2, 7], [3, 6], [4, 5]],
        "fit_samples": int(len(rotated_train_actions)),
    }
    save_xt_outputs(all_events, xt_grid, fit_metadata, output_dir=XT_DIR, xt_match_dir=XT_MATCH_DIR)
    print(f"Saved xT outputs to {XT_DIR}")


if __name__ == "__main__":
    main()
