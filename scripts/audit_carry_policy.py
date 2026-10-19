"""Bounded carry audit; never starts a complete feature-run update."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from datatools.artifact_storage import atomic_parquet_save
from datatools.ball_carries import derive_carry_segments
from datatools.carry_policy import CarryPolicy, DEFINITION
from project_config import DATA_ROOT, CONTROL_EVENTS_SYNCED_DIR, atomic_write_json
from scripts.preprocess_sportec import discover_match_files
from scripts.replace_carries import resolve_source, rebuild_match


def run(source, output, per_season=2, benchmark=1):
    if per_season < 1 or per_season > 3 or benchmark < 0 or benchmark > 3:
        raise ValueError("Bounded audit permits 1-3 matches per season and 0-3 rebuild benchmarks")
    output = Path(output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    available = {p.stem for p in (source / "action_graphs").glob("*.pt")}
    by_season = {}
    for match in discover_match_files():
        if match.match_id in available:
            by_season.setdefault(match.season, []).append(match.match_id)
    selected = {"DFL-MAT-J03YDU": "23_24"}
    for season, ids in sorted(by_season.items()):
        for match_id in sorted(ids)[:per_season]:
            selected[match_id] = season
    reports = []
    for match_id, season in selected.items():
        started = time.perf_counter()
        controls = pd.read_parquet(CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet")
        events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv")
        tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
        segments, audit = derive_carry_segments(controls, events, tracking)
        atomic_parquet_save(audit, output / "audits" / f"{match_id}.parquet")
        original = pd.read_parquet(DATA_ROOT / "carry_segments" / f"{match_id}.parquet")
        groups = lambda columns: [{**dict(zip(columns, key if isinstance(key, tuple) else (key,))), "count": int(n)}
                                  for key, n in audit.groupby(columns, dropna=False).size().items()]
        report = dict(match_id=match_id, season=season, old_segments=len(original), new_segments=len(segments),
            spells=len(audit), retained_spells=int(audit.retained.sum()),
            exclusions=audit.loc[~audit.retained, "reason"].value_counts().to_dict(),
            by_start=groups(["start_kind", "retained"]), by_end=groups(["terminal_kind", "retained"]),
            by_source=groups(["start_source", "terminal_source", "retained"]),
            by_reason_start=groups(["reason", "start_kind", "retained"]),
            by_reason_end=groups(["reason", "terminal_kind", "retained"]),
            by_reason_source=groups(["reason", "start_source", "terminal_source", "retained"]),
            diagnostic_examples=audit.loc[audit.reason.isin(["acquisition_distance", "sustained_separation", "dead_ball_interval"])]
                .head(12).to_dict("records"), elapsed_seconds=time.perf_counter()-started)
        reports.append(report)
        print(f"{season} {match_id}: {len(original)} -> {len(segments)} segments; {report['exclusions']}", flush=True)
    rebuilds = []
    for match_id in list(selected)[:benchmark]:
        rebuilds.append(rebuild_match((str(source), str(output / "benchmark"), metadata, match_id, asdict(CarryPolicy()), 1)))
    result = dict(definition=DEFINITION, source_run=source.name, policy=asdict(CarryPolicy()), matches=reports,
                  rebuilds=rebuilds, total_source_matches=len(available))
    if rebuilds:
        seconds = sum(r["elapsed_seconds"] for r in rebuilds)/len(rebuilds)
        result["estimated_serial_rebuild_hours"] = seconds * len(available)/3600
    atomic_write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default="latest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-season", type=int, default=2)
    parser.add_argument("--benchmark", type=int, default=1)
    args = parser.parse_args()
    run(resolve_source(args.source_run_id), args.output, args.per_season, args.benchmark)
