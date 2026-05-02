from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch

from datatools.epv import annotate_match_epv, build_epv_action_values, compute_epv_values
from inference import inference_gnn
from models.utils import (
    get_model_provenance,
    load_model,
    resolve_model_selection,
    resolve_runtime_return_type,
    resolve_runtime_feature_run_context,
    validate_model_graph_schemas,
)
from project_config import (
    EPV_DIR,
    EPV_MATCH_DIR,
    EVENT_SYNCED_DIR,
    INTENDED_RECEIVER_MODES,
    ensure_project_dirs,
    resolve_feature_root,
)
from scripts.run_relevant_models import load_match


REQUIRED_EPV_MODEL_TASKS = [
    "pass_intent",
    "pass_success",
    "outcome_scoring",
    "outcome_conceding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--feature-run-id", help="Runtime feature run used to load graphs/resolved actions.")
    parser.add_argument("--intended-receiver-mode", choices=INTENDED_RECEIVER_MODES, help="Runtime resolved-action mode.")
    parser.add_argument("--return-type", "--return_type", dest="return_type", help="Runtime return type used for label construction.")
    parser.add_argument("--match-id", action="append", help="Restrict export generation to one or more match ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N available matches.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing EPV outputs.")
    return parser.parse_args()


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


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
        raise ValueError("No synced event files were selected for EPV generation.")
    return match_ids


def resolve_epv_model_selection(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, object], dict[str, object] | None]:
    return resolve_model_selection(
        required_tasks=REQUIRED_EPV_MODEL_TASKS,
        bundle_id=args.bundle_id,
        explicit_model_ids={
            "pass_intent": args.pass_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )


def load_epv_models(resolved_model_ids: dict[str, str], device: str) -> dict[str, object]:
    model_specs = {task: load_model(model_id, device) for task, model_id in resolved_model_ids.items()}
    missing = [task for task, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    return model_specs


def infer_match_epv(match, model_specs: dict[str, object], device: str) -> pd.DataFrame:
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

    if pass_intent.empty:
        return pd.DataFrame(columns=["action_id", "epv"])

    epv_values = compute_epv_values(
        pass_intent,
        pass_success,
        scoring_success,
        scoring_failure,
        conceding_success,
        conceding_failure,
    )
    return build_epv_action_values(match.actions, epv_values.dropna())


def save_export_outputs(
    match_ids: list[str],
    model_specs: dict[str, object],
    shared_context: dict[str, object],
    graph_schema: dict[str, int | bool],
    device: str,
    output_dir: Path,
    match_dir: Path,
) -> tuple[list[str], list[dict[str, str]], list[pd.DataFrame]]:
    processed_match_ids: list[str] = []
    skipped_matches: list[dict[str, str]] = []
    exported_actions: list[pd.DataFrame] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    feature_root = resolve_feature_root(str(shared_context["feature_run_id"]))
    intended_receiver_mode = str(shared_context["intended_receiver_mode"])
    return_type = str(shared_context["return_type"])

    for index, match_id in enumerate(match_ids, start=1):
        print(f"[{index}/{len(match_ids)}] {match_id}")
        try:
            events = load_events(match_id)
            match = load_match(
                match_id,
                intended_receiver_mode=intended_receiver_mode,
                return_type=return_type,
                feature_root=feature_root,
                add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
            )
            epv_action_values = infer_match_epv(match, model_specs, device)
            annotated_events, exported_epv = annotate_match_epv(events, epv_action_values)
            resolved_match_id = str(
                annotated_events["stats_perform_match_id"].iloc[0]
                if "stats_perform_match_id" in annotated_events.columns
                else annotated_events["game_id"].iloc[0]
            )
            sidecar = annotated_events[["action_id", "epv", "scores_epv", "concedes_epv"]].copy()
            sidecar.to_csv(match_dir / f"{resolved_match_id}.csv", index=False)
            exported_actions.append(exported_epv)
            processed_match_ids.append(resolved_match_id)
        except Exception as exc:
            skipped_matches.append({"match_id": match_id, "error": summarize_exception(exc)})
            print(f"  SKIP {match_id}: {summarize_exception(exc)}")

    if not processed_match_ids:
        raise ValueError("No usable synced event files remained for EPV export writing.")

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "epv.csv", index=False)
    else:
        pd.DataFrame(columns=["action_id", "epv", "scores_epv", "concedes_epv"]).to_csv(
            output_dir / "epv.csv",
            index=False,
        )

    return processed_match_ids, skipped_matches, exported_actions


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    output_csv = EPV_DIR / "epv.csv"
    output_metadata = EPV_DIR / "metadata.json"
    if not args.overwrite and output_csv.exists() and output_metadata.exists():
        print(f"EPV outputs already exist in {EPV_DIR}. Use --overwrite to rebuild them.")
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    resolved_model_ids, shared_context, bundle = resolve_epv_model_selection(args)
    model_specs = load_epv_models(resolved_model_ids, device)
    graph_schema = validate_model_graph_schemas(model_specs)
    runtime_feature_context = resolve_runtime_feature_run_context(
        args.feature_run_id,
        shared_context,
        bundle,
        args.intended_receiver_mode,
        graph_schema,
        context="Selected EPV feature artifacts",
    )
    runtime_return_type = resolve_runtime_return_type(shared_context, args.return_type)
    shared_context = dict(shared_context)
    shared_context["feature_run_id"] = runtime_feature_context["feature_run_id"]
    shared_context["runtime_feature_run_id"] = runtime_feature_context["feature_run_id"]
    shared_context["intended_receiver_mode"] = runtime_feature_context["intended_receiver_mode"]
    shared_context["runtime_intended_receiver_mode"] = runtime_feature_context["intended_receiver_mode"]
    shared_context["return_type"] = runtime_return_type
    shared_context["runtime_return_type"] = runtime_return_type
    shared_context["runtime_feature_run_selection"] = runtime_feature_context["selection"]
    feature_schema = runtime_feature_context["feature_schema"]
    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_model_ids.items()}

    match_ids = resolve_match_ids(args.match_id, args.limit)
    processed_export_ids, skipped_export_matches, _ = save_export_outputs(
        match_ids,
        model_specs,
        shared_context,
        graph_schema,
        device,
        EPV_DIR,
        EPV_MATCH_DIR,
    )

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "export_match_ids": processed_export_ids,
        "eligible_action_types": ["pass", "cross", "shot"],
        "target_name": "epv",
        "formula": (
            "sum(pass_intent * (pass_success * (outcome_scoring_success - outcome_conceding_success) "
            "+ (1 - pass_success) * (outcome_scoring_failure - outcome_conceding_failure)))"
        ),
        "shot_value": "max(epv, xG)",
        "bundle_id": args.bundle_id,
        "bundle": bundle,
        "models": resolved_model_ids,
        "model_records": model_records,
        "feature_run_id": shared_context["feature_run_id"],
        "runtime_feature_run_id": shared_context["runtime_feature_run_id"],
        "runtime_intended_receiver_mode": shared_context["runtime_intended_receiver_mode"],
        "runtime_return_type": shared_context["runtime_return_type"],
        "runtime_feature_run_selection": shared_context["runtime_feature_run_selection"],
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
        "intended_receiver_mode": shared_context["intended_receiver_mode"],
        "return_type": shared_context["return_type"],
        "source_target_family": shared_context.get("target_family"),
        "graph_schema": graph_schema,
        "feature_schema": feature_schema,
        "skipped_export_matches": skipped_export_matches,
    }
    (EPV_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if skipped_export_matches:
        print(f"Skipped {len(skipped_export_matches)} export matches while generating EPV outputs.")
    print(f"Saved EPV outputs to {EPV_DIR}")


if __name__ == "__main__":
    main()
