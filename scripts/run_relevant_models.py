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

from datatools.graph_feature import construct_graph_features
from datatools.match import Match
from datatools.success_intent import build_success_intent_resolved_actions
from inference import inference_gnn
from models.utils import (
    get_model_provenance,
    infer_feature_graph_schema,
    load_model,
    resolve_model_selection,
    validate_feature_graph_schema,
    validate_model_graph_schemas,
)
from project_config import (
    COMPONENT_RUNS_DIR,
    DATA_ROOT,
    DEFAULT_INTENDED_RECEIVER_MODE,
    generate_run_id,
    get_action_graph_dir,
    get_resolved_action_path,
    get_success_intent_graph_dir,
    load_base_splits,
    resolve_feature_root,
    write_latest_run,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--match-id", action="append", help="Restrict inference to one or more match ids.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-id")
    parser.add_argument("--run-id")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def count_valid_graphs(graphs: list[object]) -> int:
    if not isinstance(graphs, list):
        raise TypeError("Cached graph artifact is not a list of graphs.")
    return sum(graph is not None for graph in graphs)


def load_match(
    match_id: str,
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
    return_type: str | None = None,
    feature_root: Path | None = None,
    add_v_edge_features: bool = False,
) -> Match:
    feature_root = Path(feature_root) if feature_root is not None else None
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineups = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()

    match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
    resolved_action_path = get_resolved_action_path(
        match_id,
        intended_receiver_mode=intended_receiver_mode,
        root=feature_root,
    )
    if not resolved_action_path.exists():
        raise FileNotFoundError(
            f"Resolved actions not found at {resolved_action_path}. Run scripts/generate_relevant_features.py for this mode first."
        )
    resolved_actions = pd.read_parquet(resolved_action_path)
    match.labels = match.construct_labels(
        discount_xg=True,
        intended_receiver_mode=intended_receiver_mode,
        relabel_intended_receivers=False,
        resolved_actions=resolved_actions,
        return_type=return_type,
    )

    graph_path = get_action_graph_dir(feature_root) / f"{match_id}.pt"
    if graph_path.exists():
        try:
            match.graph_features_0 = torch.load(graph_path, weights_only=False)
        except Exception as exc:
            print(f"  Rebuilding cached action graphs after read failure: {summarize_exception(exc)}")
            match.graph_features_0 = construct_graph_features(
                match,
                extend=True,
                post_action=False,
                add_v_edge_features=add_v_edge_features,
            )
        else:
            if count_valid_graphs(match.graph_features_0) == 0:
                print("  Rebuilding cached action graphs because the cached file contains no usable graphs.")
                match.graph_features_0 = construct_graph_features(
                    match,
                    extend=True,
                    post_action=False,
                    add_v_edge_features=add_v_edge_features,
                )
    else:
        match.graph_features_0 = construct_graph_features(
            match,
            extend=True,
            post_action=False,
            add_v_edge_features=add_v_edge_features,
        )
    if count_valid_graphs(match.graph_features_0) == 0:
        raise ValueError("No usable action graphs are available for this match.")

    return match


COMPONENT_IDENTIFIER_COLUMNS = ["stats_perform_match_id", "action_id", "original_event_id"]


def save_component_table(frame: pd.DataFrame, actions: pd.DataFrame, output_path: Path) -> None:
    missing_columns = [column for column in COMPONENT_IDENTIFIER_COLUMNS if column not in actions.columns]
    if missing_columns:
        raise ValueError(
            "Cannot export component table because match.actions is missing: "
            f"{', '.join(missing_columns)}"
        )
    if not frame.index.is_unique:
        raise ValueError("Cannot export component table because prediction rows have duplicate action indexes.")
    if not actions.index.is_unique:
        raise ValueError("Cannot export component table because match.actions has duplicate indexes.")

    missing_action_indexes = frame.index.difference(actions.index)
    if len(missing_action_indexes) > 0:
        sample = missing_action_indexes[:5].tolist()
        raise ValueError(
            "Cannot export component table because prediction action indexes are missing from match.actions: "
            f"{sample}"
        )

    identifiers = actions.loc[frame.index, COMPONENT_IDENTIFIER_COLUMNS].reset_index(drop=True)
    table = frame.reset_index(drop=True)
    duplicate_columns = [column for column in COMPONENT_IDENTIFIER_COLUMNS if column in table.columns]
    if duplicate_columns:
        raise ValueError(
            "Cannot export component table because prediction columns duplicate identifiers: "
            f"{', '.join(duplicate_columns)}"
        )

    table = pd.concat([identifiers, table], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)


def resolve_optional_success_intent_model_id(
    args: argparse.Namespace,
    bundle: dict[str, object] | None,
) -> str | None:
    if args.success_intent_model_id:
        return str(args.success_intent_model_id)
    if bundle is None:
        return None
    bundle_model_ids = bundle.get("model_ids", {})
    if not isinstance(bundle_model_ids, dict):
        return None
    success_intent_model_id = bundle_model_ids.get("success_intent")
    return str(success_intent_model_id) if success_intent_model_id else None


def load_optional_success_intent_model(
    success_intent_model_id: str | None,
    device: str,
    feature_root: Path,
) -> tuple[object | None, dict[str, object] | None, dict[str, int | bool] | None, dict[str, int | bool] | None]:
    if not success_intent_model_id:
        return None, None, None, None

    success_intent_model = load_model(success_intent_model_id, device)
    if success_intent_model is None:
        raise FileNotFoundError(f"Missing success_intent model checkpoint: {success_intent_model_id}")

    success_intent_graph_schema = validate_model_graph_schemas({"success_intent": success_intent_model})
    success_intent_feature_schema = infer_feature_graph_schema(get_success_intent_graph_dir(feature_root))
    validate_feature_graph_schema(
        success_intent_feature_schema,
        success_intent_graph_schema,
        context="Selected success_intent feature artifacts",
    )
    success_intent_model_record = get_model_provenance(success_intent_model_id)
    return (
        success_intent_model,
        success_intent_model_record,
        success_intent_graph_schema,
        success_intent_feature_schema,
    )


def run_success_intent_inference(
    match: Match,
    model: object,
    return_type: str | None,
    device: str,
) -> pd.DataFrame:
    original_actions = match.actions.copy()
    original_labels = match.labels.clone() if isinstance(match.labels, torch.Tensor) else match.labels
    original_stats = dict(getattr(match, "intended_receiver_stats", {}))

    try:
        resolved_actions = build_success_intent_resolved_actions(original_actions)
        labels = match.construct_labels(
            discount_xg=True,
            relabel_intended_receivers=False,
            resolved_actions=resolved_actions,
            return_type=return_type,
        )
        if labels.numel() == 0:
            raise ValueError("No usable success_intent labels were produced for this match.")
        match.labels = labels
        success_intent, _ = inference_gnn(match, model, device=device, post_action=False)
        if success_intent.empty:
            raise ValueError("No usable success_intent inference rows were produced for this match.")
        return success_intent
    finally:
        match.actions = original_actions
        match.labels = original_labels
        match.intended_receiver_stats = original_stats


def resolve_match_ids(split: str, requested_match_ids: list[str] | None, feature_dir: Path) -> list[str]:
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
    resolved_model_ids, shared_context, bundle = resolve_model_selection(
        required_tasks=[
            "action_intent",
            "pass_intent",
            "pass_success",
            "outcome_scoring",
            "outcome_conceding",
        ],
        bundle_id=args.bundle_id,
        explicit_model_ids={
            "action_intent": args.action_intent_model_id,
            "pass_intent": args.pass_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
    )
    feature_run_id = shared_context["feature_run_id"]
    intended_receiver_mode = shared_context["intended_receiver_mode"]
    return_type = shared_context["return_type"]
    feature_root = resolve_feature_root(feature_run_id)
    component_run_id = args.run_id or generate_run_id("component")
    output_parent = Path(args.output_dir) if args.output_dir else COMPONENT_RUNS_DIR
    output_dir = output_parent / component_run_id

    model_specs = {
        "action_intent": load_model(resolved_model_ids["action_intent"], device),
        "pass_intent": load_model(resolved_model_ids["pass_intent"], device),
        "pass_success": load_model(resolved_model_ids["pass_success"], device),
        "outcome_scoring": load_model(resolved_model_ids["outcome_scoring"], device),
        "outcome_conceding": load_model(
            resolved_model_ids["outcome_conceding"],
            device,
        ),
    }
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    graph_schema = validate_model_graph_schemas(model_specs)
    feature_schema = infer_feature_graph_schema(get_action_graph_dir(feature_root))
    validate_feature_graph_schema(feature_schema, graph_schema, context="Selected feature artifacts")
    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_model_ids.items()}
    success_intent_model_id = resolve_optional_success_intent_model_id(args, bundle)
    (
        success_intent_model,
        success_intent_model_record,
        success_intent_graph_schema,
        success_intent_feature_schema,
    ) = load_optional_success_intent_model(success_intent_model_id, device, feature_root)

    match_ids = resolve_match_ids(args.split, args.match_id, get_action_graph_dir(feature_root))

    metadata = {
        "run_id": component_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_parent": str(output_parent),
        "split": args.split,
        "requested_match_ids": match_ids,
        "feature_run_id": feature_run_id,
        "feature_root": str(feature_root),
        "feature_schema": feature_schema,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": shared_context["target_family"],
        "graph_schema": graph_schema,
        "models": {
            "action_intent": resolved_model_ids["action_intent"],
            "pass_intent": resolved_model_ids["pass_intent"],
            "pass_success": resolved_model_ids["pass_success"],
            "outcome_scoring": resolved_model_ids["outcome_scoring"],
            "outcome_conceding": resolved_model_ids["outcome_conceding"],
        },
        "model_records": model_records,
        "model_feature_signatures": {task: record["feature_signature"] for task, record in model_records.items()},
        "processed_match_ids": [],
        "skipped_matches": [],
        "status": "completed",
    }
    if success_intent_model_id and success_intent_model_record is not None:
        metadata["models"]["success_intent"] = success_intent_model_id
        metadata["model_records"]["success_intent"] = success_intent_model_record
        metadata["model_feature_signatures"]["success_intent"] = success_intent_model_record["feature_signature"]
        metadata["success_intent_graph_schema"] = success_intent_graph_schema
        metadata["success_intent_feature_schema"] = success_intent_feature_schema
        metadata["success_intent_skipped_matches"] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, match_id in enumerate(match_ids, start=1):
        print(f"[{index}/{len(match_ids)}] {match_id}")
        try:
            match = load_match(
                match_id,
                intended_receiver_mode=intended_receiver_mode,
                return_type=return_type,
                feature_root=feature_root,
                add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
            )
            match_output_dir = output_dir / match_id

            action_intent, _ = inference_gnn(match, model_specs["action_intent"], device=device, post_action=False)
            pass_intent, _ = inference_gnn(match, model_specs["pass_intent"], device=device, post_action=False)
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

            save_component_table(action_intent, match.actions, match_output_dir / "action_intent.parquet")
            save_component_table(pass_intent, match.actions, match_output_dir / "pass_intent.parquet")
            save_component_table(pass_success, match.actions, match_output_dir / "pass_success.parquet")
            save_component_table(scoring_success, match.actions, match_output_dir / "outcome_scoring_success.parquet")
            save_component_table(scoring_failure, match.actions, match_output_dir / "outcome_scoring_failure.parquet")
            save_component_table(conceding_success, match.actions, match_output_dir / "outcome_conceding_success.parquet")
            save_component_table(conceding_failure, match.actions, match_output_dir / "outcome_conceding_failure.parquet")

            if success_intent_model is not None:
                try:
                    success_intent = run_success_intent_inference(
                        match,
                        success_intent_model,
                        return_type=return_type,
                        device=device,
                    )
                    save_component_table(success_intent, match.actions, match_output_dir / "success_intent.parquet")
                except Exception as exc:
                    error_summary = summarize_exception(exc)
                    metadata["success_intent_skipped_matches"].append({"match_id": match_id, "error": error_summary})
                    print(f"  WARN {match_id}: success_intent export skipped: {error_summary}")
            metadata["processed_match_ids"].append(match_id)
        except Exception as exc:
            error_summary = summarize_exception(exc)
            metadata["skipped_matches"].append({"match_id": match_id, "error": error_summary})
            print(f"  SKIP {match_id}: {error_summary}")

    write_run_metadata(output_dir, metadata)
    if not metadata["processed_match_ids"]:
        raise RuntimeError("No usable matches were available for component inference.")
    if metadata["skipped_matches"]:
        print(f"Skipped {len(metadata['skipped_matches'])} matches during component inference.")

    if output_parent.resolve() == COMPONENT_RUNS_DIR.resolve():
        write_latest_run("component", component_run_id)
    print(f"Component run id: {component_run_id}")
    print(f"Saved component predictions to {output_dir}")


if __name__ == "__main__":
    main()
