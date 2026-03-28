from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from scripts.preprocess_sportec import (
    build_timestamp_comparison,
    build_defcon_event_table,
    build_kloppy_event_dataset,
    build_kloppy_event_table,
    build_kloppy_tracking_table,
    build_spadl_actions,
    discover_match_files,
    export_lineup_table,
    finalize_lineup,
    load_kloppy_tracking,
    load_match_raw_events,
    parse_match_information,
    run_elastic_synchronization,
    run_event_synchronization,
    sort_matches_by_kickoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run steps 1-4 for one Sportec match and export inspection CSVs.")
    parser.add_argument("--match-id", help="Optional match id to process. Defaults to the first kickoff-sorted match.")
    parser.add_argument("--output-dir", default="test", help="Directory where the smoke-test artifacts will be written.")
    return parser.parse_args()


def select_match(match_id: str | None):
    matches = sort_matches_by_kickoff(discover_match_files())
    if not matches:
        raise FileNotFoundError("No Sportec XML files found in the raw data directories.")

    if match_id is None:
        return matches[0]

    for match_files in matches:
        if match_files.match_id == match_id:
            return match_files

    raise ValueError(f"Unknown match id: {match_id}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    match_files = select_match(args.match_id)
    lineup, metadata = parse_match_information(match_files)
    raw_events = load_match_raw_events(match_files)
    finalized_lineup = finalize_lineup(lineup, raw_events, metadata)

    _, tracking, fps = load_kloppy_tracking(match_files, finalized_lineup)
    kloppy_tracking = build_kloppy_tracking_table(tracking, raw_events, fps)

    event_ds = build_kloppy_event_dataset(match_files)
    kloppy_events = build_kloppy_event_table(event_ds)
    spadl_events, spadl_audit = build_spadl_actions(
        match_files,
        finalized_lineup,
        raw_events,
        metadata,
        event_ds=event_ds,
        kloppy_events=kloppy_events,
        return_audit=True,
    )

    defcon_events = build_defcon_event_table(
        match_files.match_id,
        spadl_events,
        finalized_lineup,
        raw_events,
        metadata,
    )
    elastic_only_events = run_elastic_synchronization(
        match_files.match_id,
        finalized_lineup,
        defcon_events,
        tracking,
        fps=fps,
    )
    synced_events, sync_audit = run_event_synchronization(
        match_files.match_id,
        finalized_lineup,
        defcon_events,
        tracking,
        fps=fps,
        sync_source="sportec_kpi",
        return_audit=True,
    )
    timestamps_comp = build_timestamp_comparison(
        match_files.match_id,
        elastic_only_events,
        tracking,
        defcon_events,
        fps=fps,
    )
    lineup_export = export_lineup_table(finalized_lineup)

    exports = {
        "kloppy_tracking.csv": kloppy_tracking,
        "kloppy_events.csv": kloppy_events,
        "spadl_events.csv": spadl_events,
        "elastic_events.csv": synced_events,
        "line_up.csv": lineup_export,
        "spadl_audit.csv": pd.DataFrame([spadl_audit]),
        "timestamps_comp.csv": timestamps_comp,
    }

    for filename, df in exports.items():
        if df.empty:
            raise ValueError(f"{filename} is empty for match {match_files.match_id}")
        save_csv(df, output_dir / filename)

    print(f"Match id: {match_files.match_id}")
    for filename, df in exports.items():
        print(f"{filename}: {len(df)} rows")
    print(f"Artifacts written to: {output_dir}")
    print("SPADL audit:", spadl_audit)
    overlap = timestamps_comp["time_dif"].notna().sum()
    if overlap:
        abs_diff = timestamps_comp.loc[timestamps_comp["time_dif"].notna(), "time_dif"].abs()
        print(
            "Sync audit:",
            sync_audit,
            f"elastic_overlap={overlap}",
            f"abs_time_diff_median={abs_diff.median():.3f}",
            f"abs_time_diff_p95={abs_diff.quantile(0.95):.3f}",
        )
    else:
        print("Sync audit:", sync_audit, "elastic_overlap=0")


if __name__ == "__main__":
    main()
