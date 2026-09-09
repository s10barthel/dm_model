"""Re-synchronize a small sample and audit endpoint consumers in isolated outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from datatools.ball_carries import build_synchronized_control_events, derive_carry_segments
from datatools.endpoint_policy import ENDPOINT_POLICY_VERSION, valid_interval
from datatools.graph_feature import summarize_ball_trajectory
from datatools.match import Match
from scripts import preprocess_sportec as pre


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are rejected.")
    parser.add_argument("--match-id", action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ids = args.match_id or ["DFL-MAT-J03WEL", "DFL-MAT-J03YDU", "DFL-MAT-J04034"]
    sources = {m.match_id: m for m in pre.discover_match_files()}
    lineup = pd.read_parquet(ROOT / "data/lineup/line_up.parquet")
    reports = []
    for mid in ids:
        print(f"Synchronizing {mid}", flush=True)
        old = pd.read_csv(ROOT / f"data/event_synced/{mid}.csv", dtype={"original_event_id": "string"})
        old["utc_timestamp"] = pd.to_datetime(old["utc_timestamp"])
        raw = pd.read_parquet(ROOT / f"data/tracking/{mid}.parquet")
        tracking = pd.read_parquet(ROOT / f"data/tracking_processed/{mid}.parquet")
        players = lineup[lineup.stats_perform_match_id.eq(mid)].copy()
        updated, synchronization = pre.run_kpi_synchronization(sources[mid], players, old, raw, 25.)
        assert updated.original_event_id.tolist() == old.original_event_id.tolist()
        assert updated.success.tolist() == old.success.tolist()
        updated.to_csv(args.output_dir / f"{mid}.csv", index=False)

        frame_table = pre.build_tracking_frame_table(pre.load_match_raw_events(sources[mid]), raw, 25.)
        control = build_synchronized_control_events(sources[mid].event_path, players, frame_table,
                    kpi=pre.load_kpi_merged_table(sources[mid]), canonical_events=updated, fps=25.)
        segments, spells = derive_carry_segments(control, updated, raw, fps=25)
        control.to_parquet(args.output_dir / f"{mid}_control.parquet", index=False)
        segments.to_parquet(args.output_dir / f"{mid}_carries.parquet", index=False)
        spells.to_csv(args.output_dir / f"{mid}_carry_audit.csv", index=False)

        match = Match(updated, tracking, players, action_type="pass", include_goals=True,
                      next_action_conditions_enabled=False)
        actions = match.actions.copy()
        observations = []
        for index, action in actions.iterrows():
            start, end = action.frame_id, action.receive_frame_id
            max_z, high = match.pass_height_labels(start, end)
            trajectory = summarize_ball_trajectory(match, index)
            observations.append(dict(action_id=action.action_id, success=bool(action.success),
                                     duration=(end-start)/25., max_ball_z=max_z, pass_high=high,
                                     trajectory_available=trajectory is not None))
        samples = pd.DataFrame(observations)
        samples.to_csv(args.output_dir / f"{mid}_samples.csv", index=False)
        old_candidates = old[old.spadl_type.isin(["pass", "cross"])
                             & old[["frame_id", "receive_frame_id"]].notna().all(axis=1)]
        rejected_receipts = 0
        for _, action in old_candidates.iterrows():
            if (str(action.receiver_id).startswith(str(action.object_id)[:4]) and not action.offside
                    and not valid_interval(tracking, action.frame_id, action.receive_frame_id,
                                           allow_equal=True, period=action.period_id)):
                rejected_receipts += 1
        old_context = SimpleNamespace(actions=old_candidates, tracking=tracking, fps=25)
        invalid_old_trajectories = sum(summarize_ball_trajectory(old_context, i) is None for i in old_candidates.index)
        previous_carries = pd.read_parquet(ROOT / f"data/carry_segments/{mid}.parquet")
        report = dict(match_id=mid, canonical_events=len(updated), synchronization=synchronization,
                      excluded=int((~updated.training_sample_eligible).sum()),
                      repaired=int(updated.endpoint_source.eq("tracking_episode_end").sum()),
                      pass_candidates_before=len(old_candidates), pass_candidates_after=len(samples),
                      invalid_old_trajectory_intervals=int(invalid_old_trajectories),
                      unavailable_trajectories_after=int((~samples.trajectory_available).sum()),
                      unavailable_height_after=int(samples.pass_high.isna().sum()),
                      rejected_old_synthetic_receipts=rejected_receipts,
                      carry_segments_before=len(previous_carries), carry_segments_after=len(segments),
                      thresholds={str(d): dict(total=int(samples.duration.ge(d).sum()),
                          successful=int((samples.duration.ge(d) & samples.success).sum()),
                          unsuccessful=int((samples.duration.ge(d) & ~samples.success).sum())) for d in [.1, .2, .5]})
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    summary = dict(endpoint_policy_version=ENDPOINT_POLICY_VERSION, next_action_conditions_enabled=False,
                   count_stage="valid start snapshots and pass eligibility, before task-specific label/graph filtering",
                   source="Existing canonical source events; freshly recomputed KPI/ELASTIC synchronization and carry sidecars",
                   matches=reports)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
