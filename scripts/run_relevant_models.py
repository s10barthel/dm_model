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

from datatools.graph_feature import construct_graph_features
from datatools.match import Match
from inference import inference_gnn
from models.utils import load_model
from project_config import COMPONENT_DIR, DATA_ROOT, FEATURE_DIR, load_base_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--match-id", action="append", help="Restrict inference to one or more match ids.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(COMPONENT_DIR))
    return parser.parse_args()


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def count_valid_graphs(graphs: list[object]) -> int:
    if not isinstance(graphs, list):
        raise TypeError("Cached graph artifact is not a list of graphs.")
    return sum(graph is not None for graph in graphs)


def load_match(match_id: str) -> Match:
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineups = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()

    match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
    match.labels = match.construct_labels(discount_xg=True)

    graph_path = FEATURE_DIR / "action_graphs" / f"{match_id}.pt"
    if graph_path.exists():
        try:
            match.graph_features_0 = torch.load(graph_path, weights_only=False)
        except Exception as exc:
            print(f"  Rebuilding cached action graphs after read failure: {summarize_exception(exc)}")
            match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)
        else:
            if count_valid_graphs(match.graph_features_0) == 0:
                print("  Rebuilding cached action graphs because the cached file contains no usable graphs.")
                match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)
    else:
        match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)
    if count_valid_graphs(match.graph_features_0) == 0:
        raise ValueError("No usable action graphs are available for this match.")

    return match


def save_component_table(frame: pd.DataFrame, output_path: Path) -> None:
    table = frame.copy()
    table.index.name = "action_id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_parquet(output_path, index=False)


def resolve_match_ids(split: str, requested_match_ids: list[str] | None) -> list[str]:
    feature_dir = FEATURE_DIR / "action_graphs"
    train_ids, test_ids = load_base_splits(feature_dir)

    if split == "train":
        match_ids = train_ids.tolist()
    elif split == "test":
        match_ids = test_ids.tolist()
    else:
        match_ids = train_ids.tolist() + [match_id for match_id in test_ids.tolist() if match_id not in set(train_ids)]

    if requested_match_ids:
        requested = set(requested_match_ids)
        match_ids = [match_id for match_id in match_ids if match_id in requested]

    if not match_ids:
        raise ValueError("No matches selected for component inference.")

    return match_ids


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)

    model_specs = {
        "action_intent": load_model(args.action_intent_model_id, device),
        "pass_success": load_model(args.pass_success_model_id, device),
        "outcome_scoring": load_model(args.outcome_scoring_model_id, device),
        "outcome_conceding": load_model(args.outcome_conceding_model_id, device),
    }
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")

    match_ids = resolve_match_ids(args.split, args.match_id)

    metadata = {
        "split": args.split,
        "requested_match_ids": match_ids,
        "models": {
            "action_intent": args.action_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        "processed_match_ids": [],
        "skipped_matches": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, match_id in enumerate(match_ids, start=1):
        print(f"[{index}/{len(match_ids)}] {match_id}")
        try:
            match = load_match(match_id)
            match_output_dir = output_dir / match_id

            action_intent, _ = inference_gnn(match, model_specs["action_intent"], device=device, post_action=False)
            pass_success, _ = inference_gnn(match, model_specs["pass_success"], device=device, post_action=False)
            scoring_failure, scoring_success = inference_gnn(
                match,
                model_specs["outcome_scoring"],
                device=device,
                post_action=False,
            )
            conceding_failure, conceding_success = inference_gnn(
                match,
                model_specs["outcome_conceding"],
                device=device,
                post_action=False,
            )
            if action_intent.empty:
                raise ValueError("No usable inference rows were produced for this match.")

            save_component_table(action_intent, match_output_dir / "action_intent.parquet")
            save_component_table(pass_success, match_output_dir / "pass_success.parquet")
            save_component_table(scoring_success, match_output_dir / "outcome_scoring_success.parquet")
            save_component_table(scoring_failure, match_output_dir / "outcome_scoring_failure.parquet")
            save_component_table(conceding_success, match_output_dir / "outcome_conceding_success.parquet")
            save_component_table(conceding_failure, match_output_dir / "outcome_conceding_failure.parquet")
            metadata["processed_match_ids"].append(match_id)
        except Exception as exc:
            error_summary = summarize_exception(exc)
            metadata["skipped_matches"].append({"match_id": match_id, "error": error_summary})
            print(f"  SKIP {match_id}: {error_summary}")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["processed_match_ids"]:
        raise RuntimeError("No usable matches were available for component inference.")
    if metadata["skipped_matches"]:
        print(f"Skipped {len(metadata['skipped_matches'])} matches during component inference.")

    print(f"Saved component predictions to {output_dir}")


if __name__ == "__main__":
    main()
