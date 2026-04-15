from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
from models.utils import get_model_provenance, resolve_relevant_model_ids, validate_model_graph_schemas
from project_config import (
    HAWKEYE_COMPONENT_RUNS_DIR,
    PROJECT_ROOT,
    generate_run_id,
    write_latest_run,
    write_run_metadata,
    resolve_intended_receiver_mode,
)


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
    parser.add_argument("--use_xg", action="store_true")
    parser.add_argument("--use_xt", action="store_true")
    parser.add_argument("--use_goal_distance", action="store_true")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.set_defaults(freeze_ballreceipt=True)
    args = parser.parse_args()
    enabled_flags = int(bool(args.use_xg)) + int(bool(args.use_xt)) + int(bool(args.use_goal_distance))
    if enabled_flags > 1:
        parser.error("--use_xg, --use_xt, and --use_goal_distance are mutually exclusive.")
    return args


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    resolved_model_ids = resolve_relevant_model_ids(
        intended_receiver_mode=intended_receiver_mode,
        use_xg=args.use_xg,
        use_xt=args.use_xt,
        use_goal_distance=args.use_goal_distance,
        explicit_model_ids={
            "action_intent": args.action_intent_model_id,
            "pass_intent": args.pass_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        include_pass_intent=True,
    )
    component_run_id = args.run_id or generate_run_id("hawkeye_component")
    output_parent = Path(args.output_dir) if args.output_dir else HAWKEYE_COMPONENT_RUNS_DIR
    output_dir = output_parent / component_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    situation_ids = resolve_situation_ids(tracking, requested_ids=args.situation_id, limit=args.limit)

    model_specs = load_hawkeye_models(
        action_intent_model_id=resolved_model_ids["action_intent"],
        pass_intent_model_id=resolved_model_ids["pass_intent"],
        pass_success_model_id=resolved_model_ids["pass_success"],
        outcome_scoring_model_id=resolved_model_ids["outcome_scoring"],
        outcome_conceding_model_id=resolved_model_ids["outcome_conceding"],
        device=device,
    )
    graph_schema = validate_model_graph_schemas(model_specs)
    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_model_ids.items()}

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
                add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
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
        "run_id": component_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_parent": str(output_parent),
        "output_dir": str(output_dir.resolve()),
        "intended_receiver_mode": intended_receiver_mode,
        "use_xg": bool(args.use_xg),
        "use_xt": bool(args.use_xt),
        "use_goal_distance": bool(args.use_goal_distance),
        "target_family": "goal_distance" if args.use_goal_distance else ("xt" if args.use_xt else ("xg" if args.use_xg else "goal")),
        "freeze_ballreceipt": bool(args.freeze_ballreceipt),
        "requested_situation_ids": args.situation_id or [],
        "limit": args.limit,
        "processed_situation_ids": processed_situation_ids,
        "skipped_situations": skipped_situations,
        "totals": totals,
        "models": resolved_model_ids,
        "model_records": model_records,
        "model_feature_signatures": {task: record["feature_signature"] for task, record in model_records.items()},
        "graph_schema": graph_schema,
        "status": "completed",
    }
    if not processed_situation_ids:
        raise RuntimeError("No usable Hawkeye situations were processed.")

    hawkeye_table = pd.concat(export_tables, ignore_index=True) if export_tables else pd.DataFrame()
    parquet_path = output_dir / "hawkeye_data.parquet"
    csv_path = output_dir / "hawkeye_data.csv"
    hawkeye_table.to_parquet(parquet_path, index=False)
    hawkeye_table.to_csv(csv_path, index=False)
    write_run_metadata(output_dir, metadata)
    if output_parent.resolve() == HAWKEYE_COMPONENT_RUNS_DIR.resolve():
        write_latest_run("hawkeye_component", component_run_id)
    print(f"Hawkeye component run id: {component_run_id}")
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
