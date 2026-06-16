from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from datatools.benchmark import (
    build_benchmark_export,
    build_benchmark_state,
    discover_benchmark_modifications,
    infer_benchmark_components,
    load_benchmark_models,
    load_benchmark_modification_data,
    summarize_benchmark_stats,
)
from models.utils import get_model_provenance, resolve_model_selection, validate_model_graph_schemas
from physical_pass_model import (
    format_physical_xpass_cache_summary,
    inference_uses_physical_xpass,
    model_uses_physical_xpass,
    physical_xpass_source,
    physical_xpass_speed_aggregation,
    physical_xpass_teammate_policy,
    prepare_runtime_physical_xpass_prewarm_items,
    prewarm_physical_xpass_runtime_cache,
    resolve_physical_num_workers,
    summarize_physical_xpass_cache_usage,
)
from project_config import (
    BENCHMARK_COMPONENT_RUNS_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_runtime_physical_xpass_dir,
    write_latest_run,
    write_run_metadata,
)
from validation.benchmark.benchmark_postprocessing import run_benchmark_postprocessing


BENCHMARK_RUNS_LEDGER_FILENAME = "benchmark_runs.csv"
BENCHMARK_RUNS_LEDGER_COLUMNS = [
    "run_id",
    "agreements",
    "disagreements",
    "performance",
    "pass_score_agreements",
    "pass_score_disagreements",
    "pass_score_performance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "benchmark"))
    parser.add_argument("--modification", action="append", type=int, help="Restrict inference to one or more benchmark modifications.")
    parser.add_argument("--limit", type=int, help="Only process the first N selected benchmark modifications.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-id")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass sidecar directory override.")
    parser.add_argument("--use-physical-xpass", "--use_physical_xpass", dest="use_physical_xpass", action="store_true", help="Blend pass-success inference with physical xPass.")
    parser.add_argument("--no-physical-cache", action="store_true", help="Disable runtime physical xPass cache.")
    parser.add_argument("--refresh-physical-cache", action="store_true", help="Deprecated during inference; run scripts/generate_physical_xpass.py to refresh/fill caches.")
    parser.add_argument("--physical-num-workers", "--num-workers", dest="physical_num_workers", default="auto")
    parser.add_argument("--physical-worker-thread-limit", "--worker-thread-limit", dest="physical_worker_thread_limit", type=int, default=1)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        resolve_physical_num_workers(args.physical_num_workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.physical_worker_thread_limit < 1:
        parser.error("--physical-worker-thread-limit must be positive.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    return args


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def format_benchmark_state_progress(index: int, total: int, modification_id: int, game_state_id: int) -> str:
    remaining = max(int(total) - int(index), 0)
    unit = "state" if remaining == 1 else "states"
    return (
        f"benchmark state {int(index)}/{int(total)} | "
        f"modification_{int(modification_id)} game_state_{int(game_state_id)} | "
        f"{remaining} {unit} left"
    )


def _pass_success_uses_physical_xpass(model_specs: dict[str, object]) -> bool:
    model = model_specs.get("pass_success")
    return model is not None and (model_uses_physical_xpass(model.args) or inference_uses_physical_xpass(model.args))


def _prewarm_benchmark_physical_xpass(
    states: list[object],
    model_specs: dict[str, object],
    *,
    cache_dir: str,
    num_workers: str | int,
    worker_thread_limit: int,
    physical_batch_size: int,
) -> dict[str, object] | None:
    model = model_specs.get("pass_success")
    if model is None or not (model_uses_physical_xpass(model.args) or inference_uses_physical_xpass(model.args)):
        return None
    items = prepare_runtime_physical_xpass_prewarm_items(states, model)
    if not items:
        return None
    source = physical_xpass_source(model.args)
    return prewarm_physical_xpass_runtime_cache(
        items,
        cache_dir=cache_dir,
        source=source,
        eps=float(model.args.get("physical_eps", 1e-4)),
        teammate_policy=physical_xpass_teammate_policy(model.args, source=source),
        speed_aggregation=physical_xpass_speed_aggregation(model.args),
        refresh=bool(model.args.get("physical_runtime_cache_refresh", False)),
        num_workers=num_workers,
        worker_thread_limit=int(worker_thread_limit),
        physical_batch_size=int(physical_batch_size),
    )


def _compact_skips(skipped: dict[str, list[int]]) -> dict[str, list[int]]:
    return {key: sorted(int(value) for value in values) for key, values in skipped.items() if values}


def _coerce_benchmark_identifier_columns(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for column in ["team", "player", "event_player"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    return normalized


def update_benchmark_runs_ledger(
    run_id: str,
    stats: dict[str, int],
    ledger_path: str | Path | None = None,
) -> Path:
    path = Path(ledger_path) if ledger_path is not None else BENCHMARK_COMPONENT_RUNS_DIR / BENCHMARK_RUNS_LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    agreements = int(stats["agreements"])
    disagreements = int(stats["disagreements"])
    comparable_total = agreements + disagreements
    performance = math.nan if comparable_total == 0 else agreements / comparable_total
    pass_score_agreements = int(stats.get("pass_score_agreements", 0))
    pass_score_disagreements = int(stats.get("pass_score_disagreements", 0))
    pass_score_comparable_total = pass_score_agreements + pass_score_disagreements
    pass_score_performance = (
        math.nan
        if pass_score_comparable_total == 0
        else pass_score_agreements / pass_score_comparable_total
    )
    row = pd.DataFrame(
        [
            {
                "run_id": str(run_id),
                "agreements": agreements,
                "disagreements": disagreements,
                "performance": performance,
                "pass_score_agreements": pass_score_agreements,
                "pass_score_disagreements": pass_score_disagreements,
                "pass_score_performance": pass_score_performance,
            }
        ],
        columns=BENCHMARK_RUNS_LEDGER_COLUMNS,
    )

    if path.exists():
        ledger = pd.read_csv(path)
        for column in BENCHMARK_RUNS_LEDGER_COLUMNS:
            if column not in ledger.columns:
                ledger[column] = pd.NA
        ledger = ledger[BENCHMARK_RUNS_LEDGER_COLUMNS].copy()
        ledger = ledger.loc[ledger["run_id"].astype(str).ne(str(run_id))]
        ledger = pd.concat([ledger, row], ignore_index=True)
    else:
        ledger = row

    ledger.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    resolved_model_ids, shared_context, _ = resolve_model_selection(
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
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )
    component_run_id = args.run_id or generate_run_id("benchmark_component")
    output_parent = Path(args.output_dir) if args.output_dir else BENCHMARK_COMPONENT_RUNS_DIR
    output_dir = output_parent / component_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_modifications, skipped_modifications = discover_benchmark_modifications(
        args.input_dir,
        requested_modifications=args.modification,
        limit=args.limit,
    )

    model_specs = load_benchmark_models(
        action_intent_model_id=resolved_model_ids["action_intent"],
        pass_intent_model_id=resolved_model_ids["pass_intent"],
        pass_success_model_id=resolved_model_ids["pass_success"],
        outcome_scoring_model_id=resolved_model_ids["outcome_scoring"],
        outcome_conceding_model_id=resolved_model_ids["outcome_conceding"],
        device=device,
    )
    no_physical_cache = bool(getattr(args, "no_physical_cache", False))
    refresh_physical_cache = bool(getattr(args, "refresh_physical_cache", False))
    physical_cache_dir = getattr(args, "physical_cache_dir", None) or str(get_runtime_physical_xpass_dir("benchmark"))
    physical_num_workers = getattr(args, "physical_num_workers", "auto")
    physical_worker_thread_limit = int(getattr(args, "physical_worker_thread_limit", 1))
    physical_batch_size = int(getattr(args, "physical_batch_size", 16))
    pass_success_model = model_specs.get("pass_success")
    if pass_success_model is not None and bool(getattr(args, "use_physical_xpass", False)):
        pass_success_model.args["inference_use_physical_xpass"] = True
    if pass_success_model is not None and (model_uses_physical_xpass(pass_success_model.args) or inference_uses_physical_xpass(pass_success_model.args)):
        pass_success_model.args["physical_runtime_cache_disabled"] = no_physical_cache
        pass_success_model.args["physical_runtime_cache_refresh"] = False
        pass_success_model.args["physical_num_workers"] = physical_num_workers
        pass_success_model.args["physical_worker_thread_limit"] = physical_worker_thread_limit
        pass_success_model.args["physical_batch_size"] = physical_batch_size
        pass_success_model.args["physical_runtime_cache_read_only"] = True
        if not no_physical_cache:
            pass_success_model.args["physical_cache_dir"] = physical_cache_dir
    graph_schema = validate_model_graph_schemas(model_specs)
    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_model_ids.items()}

    export_tables: list[pd.DataFrame] = []
    stats_by_state: dict[str, dict[str, int]] = {}
    processed_modifications: list[int] = []
    processed_states: list[dict[str, object]] = []
    physical_xpass_runtime_stats: dict[str, dict[str, object]] = {}
    physical_xpass_prewarm_stats: dict[str, dict[str, object]] = {}
    skipped_state_errors: list[dict[str, str | int]] = []
    skipped_modification_errors: list[dict[str, str | int]] = []
    built_states: list[tuple[int, int, object, pd.DataFrame, dict[str, int]]] = []

    for index, modification_id in enumerate(selected_modifications, start=1):
        print(f"[{index}/{len(selected_modifications)}] modification_{modification_id}")

        try:
            modification_data = load_benchmark_modification_data(modification_id, args.input_dir)
        except Exception as exc:
            error_summary = summarize_exception(exc)
            skipped_modification_errors.append({"modification": int(modification_id), "error": error_summary})
            print(f"  SKIP modification_{modification_id}: {error_summary}")
            continue

        for game_state_id in (1, 2):
            try:
                state, state_rows, stats = build_benchmark_state(
                    modification_data[f"game_state_{game_state_id}"],
                    modification_id=int(modification_id),
                    game_state_id=int(game_state_id),
                    higher_state_id=int(modification_data["higher_state_id"]),
                    add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
                )
                built_states.append((int(modification_id), int(game_state_id), state, state_rows, stats))
            except Exception as exc:
                error_summary = summarize_exception(exc)
                skipped_state_errors.append(
                    {
                        "modification": int(modification_id),
                        "game_state": int(game_state_id),
                        "error": error_summary,
                    }
                )
                print(f"  SKIP game_state_{game_state_id}: {error_summary}")

    if refresh_physical_cache and _pass_success_uses_physical_xpass(model_specs):
        print("  WARN --refresh-physical-cache is ignored during inference; run scripts/generate_physical_xpass.py to fill runtime caches.")

    processed_state_counts: dict[int, int] = {}
    with tqdm(built_states, total=len(built_states), desc="benchmark states") as progress:
        for state_number, (modification_id, game_state_id, state, state_rows, stats) in enumerate(progress, start=1):
            progress.set_postfix(modification=int(modification_id), game_state=int(game_state_id))
            try:
                progress.write(
                    format_benchmark_state_progress(
                        state_number,
                        len(built_states),
                        modification_id,
                        game_state_id,
                    )
                )
                components = infer_benchmark_components(state, model_specs, device=device)
                export_tables.append(build_benchmark_export(state_rows, state, components))
                state_key = f"{modification_id}:{game_state_id}"
                stats_by_state[state_key] = stats
                state_record = {"modification": int(modification_id), "game_state": int(game_state_id)}
                runtime_physical_stats = getattr(state, "physical_xpass_runtime_stats", None)
                if runtime_physical_stats:
                    physical_xpass_runtime_stats[state_key] = runtime_physical_stats
                    state_record["physical_xpass_runtime_stats"] = runtime_physical_stats
                processed_states.append(state_record)
                processed_state_counts[int(modification_id)] = processed_state_counts.get(int(modification_id), 0) + 1
            except Exception as exc:
                error_summary = summarize_exception(exc)
                skipped_state_errors.append(
                    {
                        "modification": int(modification_id),
                        "game_state": int(game_state_id),
                        "error": error_summary,
                    }
                )
                progress.write(f"  SKIP game_state_{game_state_id}: {error_summary}")

    for modification_id in selected_modifications:
        if processed_state_counts.get(int(modification_id), 0) > 0:
            processed_modifications.append(int(modification_id))
        elif not any(error["modification"] == int(modification_id) for error in skipped_modification_errors):
            skipped_modification_errors.append({"modification": int(modification_id), "error": "no_usable_states"})

    if not processed_states:
        raise RuntimeError("No usable benchmark states were processed.")

    benchmark_table = pd.concat(export_tables, ignore_index=True) if export_tables else pd.DataFrame()
    benchmark_table = _coerce_benchmark_identifier_columns(benchmark_table)
    parquet_path = output_dir / "benchmark_data.parquet"
    csv_path = output_dir / "benchmark_data.csv"
    benchmark_table.to_parquet(parquet_path, index=False)
    benchmark_table.to_csv(csv_path, index=False)

    totals = summarize_benchmark_stats(stats_by_state)
    synthetic_shot_rows = int(benchmark_table["team"].isna().sum()) if "team" in benchmark_table.columns else 0
    physical_xpass_required = _pass_success_uses_physical_xpass(model_specs)
    physical_xpass_cache_summary = summarize_physical_xpass_cache_usage(
        physical_xpass_required=physical_xpass_required,
        cache_disabled=no_physical_cache,
        refresh_requested=refresh_physical_cache,
        cache_dir=None if no_physical_cache else physical_cache_dir,
        prewarm_stats=physical_xpass_prewarm_stats,
        runtime_stats=physical_xpass_runtime_stats,
    )
    metadata = {
        "run_id": component_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_parent": str(output_parent),
        "output_dir": str(output_dir.resolve()),
        "intended_receiver_mode": shared_context["intended_receiver_mode"],
        "return_type": shared_context["return_type"],
        "target_family": shared_context["target_family"],
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
        "physical_cache_dir": None if no_physical_cache else physical_cache_dir,
        "physical_cache_disabled": no_physical_cache,
        "refresh_physical_cache": refresh_physical_cache,
        "physical_num_workers": physical_num_workers,
        "physical_worker_thread_limit": physical_worker_thread_limit,
        "physical_batch_size": physical_batch_size,
        "requested_modifications": [int(value) for value in (args.modification or [])],
        "limit": args.limit,
        "selected_modifications": [int(value) for value in selected_modifications],
        "processed_modifications": processed_modifications,
        "processed_states": processed_states,
        "physical_xpass_runtime_stats": physical_xpass_runtime_stats,
        "physical_xpass_prewarm_stats": physical_xpass_prewarm_stats,
        "physical_xpass_cache_summary": physical_xpass_cache_summary,
        "skipped_modifications": _compact_skips(skipped_modifications),
        "skipped_modification_errors": skipped_modification_errors,
        "skipped_states": skipped_state_errors,
        "synthetic_shot_rows": synthetic_shot_rows,
        "totals": totals,
        "models": resolved_model_ids,
        "model_records": model_records,
        "model_feature_signatures": {task: record["feature_signature"] for task, record in model_records.items()},
        "graph_schema": graph_schema,
        "status": "completed",
    }
    write_run_metadata(output_dir, metadata)
    if output_parent.resolve() == BENCHMARK_COMPONENT_RUNS_DIR.resolve():
        write_latest_run("benchmark_component", component_run_id)
    _, postprocessing_stats, summary_path, text_summary_path = run_benchmark_postprocessing(
        component_run_id,
        output_dir=output_dir,
    )
    ledger_path = update_benchmark_runs_ledger(component_run_id, postprocessing_stats)

    print(f"Benchmark component run id: {component_run_id}")
    print(f"Saved benchmark components to {parquet_path} and {csv_path}")
    print(f"Saved benchmark summary to {summary_path} and {text_summary_path}")
    print(f"Updated benchmark runs ledger at {ledger_path}")
    if skipped_modification_errors or skipped_state_errors:
        print(
            f"Skipped {len(skipped_modification_errors)} modifications and "
            f"{len(skipped_state_errors)} game states during benchmark inference."
        )
    print(
        "Processed {states} states, {valid_frames}/{total_frames} valid frames, "
        "skipped {skipped_missing_ball} missing-ball states, "
        "{skipped_missing_possessor} missing-possessor states, "
        "{skipped_missing_graph} missing-graph states.".format(**totals)
    )
    print(format_physical_xpass_cache_summary(physical_xpass_cache_summary))


if __name__ == "__main__":
    main()
