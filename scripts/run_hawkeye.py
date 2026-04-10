from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch

from datatools.hawkeye import (
    build_hawkeye_export,
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    infer_hawkeye_components,
    load_hawkeye_ball,
    load_hawkeye_models,
    load_hawkeye_tracking,
    resolve_situation_ids,
    summarize_hawkeye_stats,
)
from project_config import COMPONENT_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "centroid_data_team.csv"),
    )
    parser.add_argument(
        "--ball-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "ball_data_selected.csv"),
    )
    parser.add_argument("--situation-id", action="append", help="Restrict inference to one or more Hawkeye situation ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N Hawkeye situations after sorting.")
    parser.add_argument("--freeze-ballreceipt", dest="freeze_ballreceipt", action="store_true")
    parser.add_argument("--no-freeze-ballreceipt", dest="freeze_ballreceipt", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(COMPONENT_DIR))
    parser.set_defaults(freeze_ballreceipt=True)
    return parser.parse_args()


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    situation_ids = resolve_situation_ids(tracking, requested_ids=args.situation_id, limit=args.limit)

    model_specs = load_hawkeye_models(
        action_intent_model_id=args.action_intent_model_id,
        pass_success_model_id=args.pass_success_model_id,
        outcome_scoring_model_id=args.outcome_scoring_model_id,
        outcome_conceding_model_id=args.outcome_conceding_model_id,
        device=device,
    )

    export_tables: list[pd.DataFrame] = []
    stats_by_situation: dict[str, dict[str, int]] = {}
    processed_situation_ids: list[str] = []
    skipped_situations: list[dict[str, str]] = []

    for index, situation_id in enumerate(situation_ids, start=1):
        print(f"[{index}/{len(situation_ids)}] {situation_id}")
        try:
            situation_tracking = tracking.loc[tracking["id"] == situation_id].copy()
            situation, attacking_rows, stats = build_hawkeye_situation(
                situation_tracking,
                ball,
                freeze_ballreceipt=args.freeze_ballreceipt,
            )
            components = infer_hawkeye_components(situation, model_specs, device=device)
            export_tables.append(build_hawkeye_export(attacking_rows, situation, components))
            stats_by_situation[situation_id] = stats
            processed_situation_ids.append(str(situation_id))
        except Exception as exc:
            error_summary = summarize_exception(exc)
            skipped_situations.append({"situation_id": str(situation_id), "error": error_summary})
            print(f"  SKIP {situation_id}: {error_summary}")

    totals = summarize_hawkeye_stats(stats_by_situation)
    metadata = {
        "output_dir": str(output_dir.resolve()),
        "processed_situation_ids": processed_situation_ids,
        "skipped_situations": skipped_situations,
        "totals": totals,
        "models": {
            "action_intent": args.action_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not processed_situation_ids:
        raise RuntimeError("No usable Hawkeye situations were processed.")

    hawkeye_table = pd.concat(export_tables, ignore_index=True) if export_tables else pd.DataFrame()
    parquet_path = output_dir / "hawkeye_data.parquet"
    csv_path = output_dir / "hawkeye_data.csv"
    hawkeye_table.to_parquet(parquet_path, index=False)
    hawkeye_table.to_csv(csv_path, index=False)
    print(f"Saved Hawkeye components to {parquet_path} and {csv_path}")
    if skipped_situations:
        print(f"Skipped {len(skipped_situations)} Hawkeye situations.")
    print(
        "Processed {situations} situations, {valid_frames}/{total_frames} valid frames, "
        "skipped {skipped_missing_ball} missing-ball frames, "
        "{skipped_missing_possessor} missing-possessor frames, "
        "{skipped_missing_graph} missing-graph frames.".format(**totals)
    )


if __name__ == "__main__":
    main()
