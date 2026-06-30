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
from models.utils import get_model_provenance, resolve_model_selection, validate_model_graph_schemas
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    format_physical_xpass_cache_summary,
    inference_uses_physical_xpass,
    model_uses_physical_xpass,
    physical_xpass_inference_lookup_config,
    physical_xpass_source,
    physical_xpass_speed_aggregation,
    physical_xpass_teammate_policy,
    prepare_runtime_physical_xpass_prewarm_items,
    prewarm_physical_xpass_runtime_cache,
    resolve_physical_num_workers,
    summarize_physical_xpass_cache_usage,
)
from project_config import (
    HAWKEYE_COMPONENT_RUNS_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_pc_xpass_dir,
    get_runtime_physical_xpass_dir,
    write_latest_run,
    write_run_metadata,
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
    parser.add_argument("--bundle-id")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--pass-height-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass sidecar directory override.")
    parser.add_argument("--use-physical-xpass", "--use_physical_xpass", dest="use_physical_xpass", action="store_true", help="Blend pass-success inference with physical xPass.")
    parser.add_argument("--pc-xpass", "--pc_xpass", dest="pc_xpass", action="store_true", help="Use pc-xPass cache values for physical xPass inference blending.")
    parser.add_argument("--xpass-version", "--x-pass-version", "--x_pass_version", dest="x_pass_version", default="top10", help="Cached xPass version to use: max, noise-kernel, or top<N> such as top10/top25/top50.")
    parser.add_argument("--xpass-weight", "--xpass_weight", dest="xpass_weight", choices=["v1", "v2", "v3", "v4"], default="v3", help="Physical xPass/model blend weighting version.")
    parser.add_argument("--v4-power", dest="v4_power", type=float, default=None, help="Power for --xpass-weight v4. Default: 2.0.")
    parser.add_argument("--ball-z-limit", dest="ball_z_limit", default="none", help="If set to a float, use 100%% pass-success model weight when cached ball_z exceeds this value. Use 'none' to disable.")
    parser.add_argument("--no-physical-cache", action="store_true", help="Disable runtime physical xPass cache.")
    parser.add_argument("--refresh-physical-cache", action="store_true", help="Deprecated during inference; run scripts/generate_physical_xpass.py to refresh/fill caches.")
    parser.add_argument("--physical-num-workers", "--num-workers", dest="physical_num_workers", default="auto")
    parser.add_argument("--physical-worker-thread-limit", "--worker-thread-limit", dest="physical_worker_thread_limit", type=int, default=1)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.set_defaults(freeze_ballreceipt=True)
    args = parser.parse_args()
    try:
        resolve_physical_num_workers(args.physical_num_workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.physical_worker_thread_limit < 1:
        parser.error("--physical-worker-thread-limit must be positive.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    if args.v4_power is not None:
        if not math.isfinite(args.v4_power) or args.v4_power <= 0:
            parser.error("--v4-power must be positive.")
        if args.xpass_weight != "v4":
            parser.error("--v4-power is only valid with --xpass-weight v4.")
    return args


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _pass_success_uses_physical_xpass(model_specs: dict[str, object]) -> bool:
    model = model_specs.get("pass_success")
    return model is not None and (model_uses_physical_xpass(model.args) or inference_uses_physical_xpass(model.args))


def resolve_optional_model_id(task: str, explicit_model_id: str | None, bundle: dict[str, object] | None) -> str | None:
    if explicit_model_id:
        return str(explicit_model_id)
    if bundle is None:
        return None
    model_ids = bundle.get("model_ids", {})
    if not isinstance(model_ids, dict):
        return None
    model_id = model_ids.get(task)
    return str(model_id) if model_id else None


def _prewarm_hawkeye_physical_xpass(
    situation,
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
    items = prepare_runtime_physical_xpass_prewarm_items([situation], model)
    if not items:
        return None
    lookup_config = physical_xpass_inference_lookup_config(model.args, cache_dir=cache_dir)
    source = lookup_config["source"] if inference_uses_physical_xpass(model.args) else physical_xpass_source(model.args)
    return prewarm_physical_xpass_runtime_cache(
        items,
        cache_dir=cache_dir,
        source=source,
        eps=float(model.args.get("physical_eps", 1e-4)),
        teammate_policy=physical_xpass_teammate_policy(model.args, source=source),
        speed_aggregation=lookup_config["speed_aggregation"] if inference_uses_physical_xpass(model.args) else physical_xpass_speed_aggregation(model.args),
        refresh=bool(model.args.get("physical_runtime_cache_refresh", False)),
        num_workers=num_workers,
        worker_thread_limit=int(worker_thread_limit),
        physical_batch_size=int(physical_batch_size),
    )


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
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )
    pass_height_model_id = resolve_optional_model_id("pass_height", getattr(args, "pass_height_model_id", None), bundle)
    if pass_height_model_id:
        resolved_model_ids["pass_height"] = pass_height_model_id
    intended_receiver_mode = shared_context["intended_receiver_mode"]
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
        pass_height_model_id=resolved_model_ids.get("pass_height"),
        outcome_scoring_model_id=resolved_model_ids["outcome_scoring"],
        outcome_conceding_model_id=resolved_model_ids["outcome_conceding"],
        device=device,
    )
    no_physical_cache = bool(getattr(args, "no_physical_cache", False))
    refresh_physical_cache = bool(getattr(args, "refresh_physical_cache", False))
    physical_cache_dir = getattr(args, "physical_cache_dir", None) or str(
        get_pc_xpass_dir("hawkeye") if bool(getattr(args, "pc_xpass", False)) else get_runtime_physical_xpass_dir("hawkeye")
    )
    physical_num_workers = getattr(args, "physical_num_workers", "auto")
    physical_worker_thread_limit = int(getattr(args, "physical_worker_thread_limit", 1))
    physical_batch_size = int(getattr(args, "physical_batch_size", 16))
    pass_success_model = model_specs.get("pass_success")
    if pass_success_model is not None and bool(getattr(args, "use_physical_xpass", False)):
        pass_success_model.args["inference_use_physical_xpass"] = True
        pass_success_model.args["pc_xpass"] = bool(getattr(args, "pc_xpass", False))
        pass_success_model.args["x_pass_version"] = str(getattr(args, "x_pass_version", "top10"))
        pass_success_model.args["xpass_weight"] = str(getattr(args, "xpass_weight", "v3"))
        if getattr(args, "v4_power", None) is not None:
            pass_success_model.args["v4_power"] = float(args.v4_power)
        pass_success_model.args["ball_z_limit"] = getattr(args, "ball_z_limit", "none")
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
    stats_by_situation: dict[str, dict[str, int]] = {}
    processed_situation_ids: list[str] = []
    physical_xpass_runtime_stats: dict[str, dict[str, object]] = {}
    physical_xpass_skipped_actions: dict[str, dict[str, object]] = {}
    physical_xpass_prewarm_stats: dict[str, dict[str, object]] = {}
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
            runtime_physical_stats = getattr(situation, "physical_xpass_runtime_stats", None)
            if runtime_physical_stats:
                physical_xpass_runtime_stats[str(situation_id)] = runtime_physical_stats
            physical_skip_stats = getattr(situation, "physical_xpass_skipped_actions", None)
            if physical_skip_stats:
                physical_xpass_skipped_actions[str(situation_id)] = physical_skip_stats
            processed_situation_ids.append(str(situation_id))
        except Exception as exc:
            error_summary = summarize_exception(exc)
            skipped_situations.append({"situation_id": str(situation_id), "error": error_summary})
            print(f"  SKIP {situation_id}: {error_summary}")

    if refresh_physical_cache and _pass_success_uses_physical_xpass(model_specs):
        print("  WARN --refresh-physical-cache is ignored during inference; run scripts/generate_physical_xpass.py to fill runtime caches.")

    totals = summarize_hawkeye_stats(stats_by_situation)
    physical_xpass_required = _pass_success_uses_physical_xpass(model_specs)
    physical_xpass_cache_summary = summarize_physical_xpass_cache_usage(
        physical_xpass_required=physical_xpass_required,
        cache_disabled=no_physical_cache,
        refresh_requested=refresh_physical_cache,
        cache_dir=None if no_physical_cache else physical_cache_dir,
        prewarm_stats=physical_xpass_prewarm_stats,
        runtime_stats=physical_xpass_runtime_stats,
        skipped_stats=physical_xpass_skipped_actions,
    )
    physical_lookup_config = (
        physical_xpass_inference_lookup_config(pass_success_model.args, cache_dir=physical_cache_dir)
        if pass_success_model is not None
        else {}
    )
    metadata = {
        "run_id": component_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_parent": str(output_parent),
        "output_dir": str(output_dir.resolve()),
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": shared_context["return_type"],
        "target_family": shared_context["target_family"],
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
        "freeze_ballreceipt": bool(args.freeze_ballreceipt),
        "physical_cache_dir": None if no_physical_cache else physical_cache_dir,
        "physical_xpass_hash_policy": PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
        "physical_xpass_lookup_policy": "dataset_event_frame_player_only",
        "physical_xpass_checkpoint_source": physical_xpass_source(pass_success_model.args) if pass_success_model is not None else None,
        "physical_xpass_runtime_source": physical_lookup_config.get("source"),
        "physical_xpass_metric": physical_lookup_config.get("metric"),
        "x_pass_version": physical_lookup_config.get("x_pass_version"),
        "physical_xpass_weight_version": physical_lookup_config.get("weight_version"),
        "physical_xpass_v4_power": physical_lookup_config.get("v4_power") if physical_lookup_config.get("weight_version") == "v4" else None,
        "physical_xpass_ball_z_limit": physical_lookup_config.get("ball_z_limit"),
        "physical_cache_disabled": no_physical_cache,
        "refresh_physical_cache": refresh_physical_cache,
        "physical_num_workers": physical_num_workers,
        "physical_worker_thread_limit": physical_worker_thread_limit,
        "physical_batch_size": physical_batch_size,
        "requested_situation_ids": args.situation_id or [],
        "limit": args.limit,
        "processed_situation_ids": processed_situation_ids,
        "physical_xpass_runtime_stats": physical_xpass_runtime_stats,
        "physical_xpass_skipped_actions": physical_xpass_skipped_actions,
        "physical_xpass_prewarm_stats": physical_xpass_prewarm_stats,
        "physical_xpass_cache_summary": physical_xpass_cache_summary,
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
    print(format_physical_xpass_cache_summary(physical_xpass_cache_summary))


if __name__ == "__main__":
    main()
