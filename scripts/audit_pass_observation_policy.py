"""Compare receipt/duration and clean-prefix decisions without rewriting feature runs.

Example:
python -m scripts.audit_pass_observation_policy --feature-run-id FEATURE_ID \
    --match-id DFL-MAT-J03YDU --output-dir tmp/pass_policy_audit

Omit --match-id to audit every match in the source run. Receiver predictions in
that run are reported as historical evidence, never certified for the new inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
from pathlib import Path

import pandas as pd

from datatools.failed_pass_trajectory import detect, VERSION
from datatools.pass_trajectory import PassEndpointResolver, load_official_events, event_key, PASS_DURATION
from project_config import DATA_ROOT, get_feature_run_root, load_feature_run_metadata, atomic_write_json


def audit_match(task):
    root, match_id, folder = task
    source = pd.read_parquet(Path(root) / folder / f"{match_id}.parquet")
    source = source.loc[source.action_type.eq("pass")].copy()
    import pyarrow.parquet as pq
    tracking_path = DATA_ROOT / "tracking_processed" / f"{match_id}.parquet"
    fields = set(pq.read_schema(tracking_path).names)
    base = {"frame_id", "period_id", "episode_id", "ball_state", "timestamp", "ball_x", "ball_y", "ball_z"}
    columns = sorted(name for name in fields if name in base or re.fullmatch(r"(?:home|away)_\d+_[xy]", name))
    tracking = pd.read_parquet(tracking_path, columns=columns)
    if "frame_id" in tracking:
        tracking = tracking.set_index("frame_id")
    resolver = PassEndpointResolver(tracking, events=load_official_events(match_id))
    rows = []
    for index, action in source.iterrows():
        endpoint = resolver.resolve(action)
        receipt = resolver.receipt(action)
        valid_receipt = pd.notna(receipt) and receipt in tracking.index and (
            pd.isna(action.get("period_id")) or resolver.period.at[receipt] == action.period_id)
        failed = not bool(action.success)
        observed_action = action.copy()
        observed_action["observation_reason"] = endpoint.observation_reason
        quality = detect(observed_action, tracking, endpoint.trajectory) if failed else None
        rows.append(dict(match_id=match_id, event_id=event_key(action.original_event_id), action_index=index,
            success=bool(action.success), launch_frame=action.frame_id,
            previous_duration=action.get("pass_duration"), duration=endpoint.duration,
            previous_endpoint=action.get("receive_frame_id"), official_receipt=receipt,
            receipt_status="valid" if valid_receipt else "unknown" if pd.isna(receipt) else "invalid",
            duration_status="valid" if pd.notna(endpoint.duration) else "invalid" if endpoint.reason == "invalid_official_receipt_interval" else "unknown",
            duration_reason=endpoint.reason, endpoint_source=endpoint.source,
            observation_cutoff=quality["cutoff"] if failed else endpoint.trajectory,
            observation_status=quality["status"] if failed else "not_required_for_downstream_success",
            observation_reason=quality["reason"] if failed else endpoint.observation_reason,
            clean_intervals=quality["clean_intervals"] if failed else endpoint.trajectory - action.frame_id if endpoint.trajectory is not None else None,
            historical_receiver_available=pd.notna(action.get("intent_id"))))
    return match_id, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--match-id", action="append", dest="matches")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = get_feature_run_root(args.feature_run_id)
    load_feature_run_metadata(args.feature_run_id, required=True)
    folder = next((name for name in ("resolved_actions_model", "resolved_actions_original", "resolved_actions_angle_only")
                   if (root / name).is_dir()), None)
    if folder is None:
        raise ValueError(f"No resolved action tables in {root}")
    matches = sorted(set(args.matches or [p.stem for p in (root / folder).glob("*.parquet")]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(root), match_id, folder) for match_id in matches]
    tables = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = pool.map(audit_match, tasks) if pool else map(audit_match, tasks)
        for match_id, table in results:
            table.to_csv(args.output_dir / f"{match_id}.csv", index=False)
            tables.append(table)
            print(f"Audited {match_id}: {len(table)} passes", flush=True)
    finally:
        if pool:
            pool.shutdown()
    all_rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if all_rows.empty:
        raise ValueError("No passes to audit")
    successful = all_rows.loc[all_rows.success]
    summary = dict(feature_run_id=args.feature_run_id, matches=matches, passes=len(all_rows),
        pass_duration_definition=PASS_DURATION, failed_pass_trajectory_policy=VERSION,
        successful_passes=len(successful), successful_missing_duration=int(successful.duration.isna().sum()),
        successful_short_duration=int(successful.duration.lt(.1).sum()),
        successful_duration_changed=int(successful.previous_duration.fillna(-1).ne(successful.duration.fillna(-1)).sum()),
        failed_observation_status=all_rows.loc[~all_rows.success, "observation_status"].value_counts().to_dict(),
        failed_observation_reasons=all_rows.loc[~all_rows.success, "observation_reason"].value_counts().to_dict(),
        receiver_predictions_require_regeneration=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
