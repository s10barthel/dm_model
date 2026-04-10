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

from datatools.skillcorner import (
    COMPONENT_COLUMNS,
    build_skillcorner_component_table,
    build_skillcorner_match_context,
    build_skillcorner_possession,
    discover_skillcorner_matches,
    infer_skillcorner_components,
    load_skillcorner_models,
    summarize_skillcorner_stats,
)
from project_config import COMPONENT_DIR, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "skillcorner_data"))
    parser.add_argument("--match-id", action="append", help="Restrict inference to one or more SkillCorner match ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N selected SkillCorner matches.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(COMPONENT_DIR / "skillcorner"))
    return parser.parse_args()


def _save_component_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_match_ids, skipped_matches = discover_skillcorner_matches(
        args.input_dir,
        requested_match_ids=args.match_id,
        limit=args.limit,
    )
    skipped_matches.setdefault("no_player_possession", [])
    skipped_matches.setdefault("processing_error", [])

    model_specs = load_skillcorner_models(
        action_intent_model_id=args.action_intent_model_id,
        pass_success_model_id=args.pass_success_model_id,
        outcome_scoring_model_id=args.outcome_scoring_model_id,
        outcome_conceding_model_id=args.outcome_conceding_model_id,
        device=device,
    )

    stats_by_match: dict[str, dict[str, int]] = {}
    processed_matches: list[str] = []
    skipped_match_errors: list[dict[str, str]] = []
    skipped_possessions: dict[str, list[dict[str, str]]] = {}

    for index, match_id in enumerate(selected_match_ids, start=1):
        print(f"[{index}/{len(selected_match_ids)}] {match_id}")
        try:
            context = build_skillcorner_match_context(match_id, args.input_dir)
            if context["events"].empty:
                skipped_matches["no_player_possession"].append(str(match_id))
                continue

            tables_by_component: dict[str, list[pd.DataFrame]] = {component: [] for component in COMPONENT_COLUMNS}
            match_stats = {
                "possessions": 0,
                "processed_possessions": 0,
                "skipped_possessions": 0,
                "total_frames": 0,
                "valid_frames": 0,
                "skipped_missing_ball": 0,
                "skipped_missing_possessor": 0,
                "skipped_missing_graph": 0,
            }

            for event_index in context["events"]["index"].tolist():
                try:
                    possession, possession_stats = build_skillcorner_possession(context, int(event_index))
                    match_stats["possessions"] += 1
                    for key in [
                        "total_frames",
                        "valid_frames",
                        "skipped_missing_ball",
                        "skipped_missing_possessor",
                        "skipped_missing_graph",
                    ]:
                        match_stats[key] += int(possession_stats.get(key, 0))

                    components = infer_skillcorner_components(possession, model_specs, device=device)
                    player_meta = context["player_meta"]
                    for component_name in COMPONENT_COLUMNS:
                        table = build_skillcorner_component_table(
                            possession,
                            components.get(component_name),
                            player_meta,
                            include_shot=component_name == "action_intent",
                        )
                        tables_by_component[component_name].append(table)
                    match_stats["processed_possessions"] += 1
                except Exception as exc:
                    skipped_possessions.setdefault(str(match_id), []).append(
                        {"event_index": str(event_index), "error": summarize_exception(exc)}
                    )
                    match_stats["skipped_possessions"] += 1
                    print(f"  SKIP possession {event_index}: {summarize_exception(exc)}")

            if match_stats["processed_possessions"] == 0:
                skipped_matches["processing_error"].append(str(match_id))
                skipped_match_errors.append({"match_id": str(match_id), "error": "no_usable_possessions"})
                continue

            match_output_dir = output_dir / str(match_id)
            for component_name, frames in tables_by_component.items():
                component_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                _save_component_table(component_table, match_output_dir / f"{component_name}.parquet")

            stats_by_match[str(match_id)] = match_stats
            processed_matches.append(str(match_id))
        except Exception as exc:
            skipped_matches["processing_error"].append(str(match_id))
            skipped_match_errors.append({"match_id": str(match_id), "error": summarize_exception(exc)})
            print(f"  SKIP {match_id}: {summarize_exception(exc)}")

    summary = summarize_skillcorner_stats(stats_by_match, skipped_matches)
    metadata = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "processed_matches": processed_matches,
        "skipped_match_errors": skipped_match_errors,
        "skipped_possessions": skipped_possessions,
        "models": {
            "action_intent": args.action_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        **summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not processed_matches:
        raise RuntimeError("No usable SkillCorner matches were processed.")

    totals = summary["totals"]
    print(f"Saved SkillCorner components to {output_dir}")
    if skipped_match_errors or skipped_possessions:
        print(
            f"Skipped {len(skipped_match_errors)} matches and "
            f"{sum(len(items) for items in skipped_possessions.values())} possessions during SkillCorner inference."
        )
    print(
        "Processed {matches} matches, {possessions} possessions, {valid_frames}/{total_frames} valid frames, "
        "skipped {skipped_missing_ball} missing-ball frames, "
        "{skipped_missing_possessor} missing-possessor frames, "
        "{skipped_missing_graph} missing-graph frames.".format(**totals)
    )


if __name__ == "__main__":
    main()
