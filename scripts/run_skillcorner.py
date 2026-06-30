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
    PROJECT_ROOT,
    SKILLCORNER_COMPONENT_RUNS_DIR,
    generate_run_id,
    get_pc_xpass_dir,
    get_runtime_physical_xpass_dir,
    write_latest_run,
    write_run_metadata,
)
from validation.skillcorner.code.skillcorner_filter import run_skillcorner_filter
from validation.skillcorner.code.skillcorner_postprocessing import run_skillcorner_postprocessing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "skillcorner_data"))
    parser.add_argument("--match-id", action="append", help="Restrict inference to one or more SkillCorner match ids.")
    parser.add_argument("--limit", type=int, help="Only process the first N selected SkillCorner matches.")
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
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--frames-first-and-last",
        dest="frames_mode",
        action="store_const",
        const="first_and_last",
        default="first_and_last",
        help="Process only the first and last valid frame per possession.",
    )
    frame_group.add_argument(
        "--frames-all",
        dest="frames_mode",
        action="store_const",
        const="all",
        help="Process every valid frame per possession.",
    )
    args = parser.parse_args(argv)
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


def _save_component_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


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


def _prewarm_skillcorner_physical_xpass(
    possessions: list[object],
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
    items = prepare_runtime_physical_xpass_prewarm_items(possessions, model)
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


def format_remaining_games(count: int) -> str:
    unit = "game" if int(count) == 1 else "games"
    return f"{int(count)} {unit} left"


def format_match_progress(index: int, total: int, match_id: str) -> str:
    remaining = max(int(total) - int(index), 0)
    return f"[{index}/{total}] match_id={match_id} | {format_remaining_games(remaining)}"


def format_possession_progress(match_id: str, possession_number: int, total_possessions: int) -> str:
    return f"  match {match_id} possession {possession_number}/{total_possessions}"


def format_skillcorner_inference_progress(match_id: str, index: int, total: int, event_index: int | str) -> str:
    remaining = max(int(total) - int(index), 0)
    unit = "possession" if remaining == 1 else "possessions"
    return (
        f"match {match_id} inference {int(index)}/{int(total)} | "
        f"event_index={event_index} | {remaining} {unit} left"
    )


def format_possession_skip(
    match_id: str,
    possession_number: int,
    total_possessions: int,
    event_index: int | str,
    error: str,
) -> str:
    return (
        f"  SKIP match {match_id} possession {possession_number}/{total_possessions} "
        f"event_index={event_index}: {error}"
    )


def format_match_completion(match_id: str, match_stats: dict[str, int]) -> str:
    return (
        f"  DONE match {match_id}: "
        f"{int(match_stats.get('processed_possessions', 0))}/{int(match_stats.get('possessions', 0))} possessions, "
        f"{int(match_stats.get('skipped_possessions', 0))} skipped, "
        f"{int(match_stats.get('selected_frames', 0))} selected frames, "
        f"{int(match_stats.get('evaluated_frames', 0))} evaluated frames, "
        f"{int(match_stats.get('valid_frames', 0))}/{int(match_stats.get('total_frames', 0))} valid frames"
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
    component_run_id = args.run_id or generate_run_id("skillcorner_component")
    output_parent = Path(args.output_dir) if args.output_dir else SKILLCORNER_COMPONENT_RUNS_DIR
    output_dir = output_parent / component_run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_match_ids, skipped_matches = discover_skillcorner_matches(
        args.input_dir,
        requested_match_ids=args.match_id,
        limit=args.limit,
    )
    skipped_matches.setdefault("no_player_possession", [])
    skipped_matches.setdefault("processing_error", [])

    model_specs = load_skillcorner_models(
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
        get_pc_xpass_dir("skillcorner") if bool(getattr(args, "pc_xpass", False)) else get_runtime_physical_xpass_dir("skillcorner")
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

    stats_by_match: dict[str, dict[str, int]] = {}
    processed_matches: list[str] = []
    skipped_match_errors: list[dict[str, str]] = []
    skipped_possessions: dict[str, list[dict[str, str]]] = {}
    physical_xpass_runtime_stats: dict[str, dict[str, object]] = {}
    physical_xpass_skipped_actions: dict[str, dict[str, object]] = {}
    physical_xpass_prewarm_stats: dict[str, dict[str, object]] = {}

    for index, match_id in enumerate(selected_match_ids, start=1):
        print(format_match_progress(index, len(selected_match_ids), str(match_id)))
        try:
            context = build_skillcorner_match_context(match_id, args.input_dir)
            if context["events"].empty:
                skipped_matches["no_player_possession"].append(str(match_id))
                print(f"  SKIP match {match_id}: no eligible possessions")
                continue

            event_indices = context["events"]["index"].tolist()
            total_possessions = len(event_indices)
            print(f"  match {match_id}: {total_possessions} eligible possessions")
            tables_by_component: dict[str, list[pd.DataFrame]] = {component: [] for component in COMPONENT_COLUMNS}
            match_stats = {
                "possessions": 0,
                "processed_possessions": 0,
                "skipped_possessions": 0,
                "total_frames": 0,
                "valid_frames": 0,
                "evaluated_frames": 0,
                "selected_frames": 0,
                "skipped_missing_ball": 0,
                "skipped_missing_possessor": 0,
                "skipped_missing_graph": 0,
            }

            built_possessions: list[tuple[int, int, object]] = []
            for possession_number, event_index in enumerate(event_indices, start=1):
                print(format_possession_progress(str(match_id), possession_number, total_possessions))
                try:
                    possession, possession_stats = build_skillcorner_possession(
                        context,
                        int(event_index),
                        add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
                        frames_mode=args.frames_mode,
                    )
                    match_stats["possessions"] += 1
                    for key in [
                        "total_frames",
                        "valid_frames",
                        "skipped_missing_ball",
                        "skipped_missing_possessor",
                        "skipped_missing_graph",
                        "evaluated_frames",
                        "selected_frames",
                    ]:
                        match_stats[key] += int(possession_stats.get(key, 0))

                    if int(possession_stats.get("valid_frames", 0)) == 0:
                        error_summary = "ValueError: no valid frames were available after SkillCorner graph construction."
                        skipped_possessions.setdefault(str(match_id), []).append(
                            {
                                "event_index": str(event_index),
                                "error": error_summary,
                            }
                        )
                        match_stats["skipped_possessions"] += 1
                        print(
                            format_possession_skip(
                                str(match_id),
                                possession_number,
                                total_possessions,
                                event_index,
                                error_summary,
                            )
                        )
                        continue

                    built_possessions.append((possession_number, int(event_index), possession))
                except Exception as exc:
                    error_summary = summarize_exception(exc)
                    skipped_possessions.setdefault(str(match_id), []).append(
                        {"event_index": str(event_index), "error": error_summary}
                    )
                    match_stats["skipped_possessions"] += 1
                    print(
                        format_possession_skip(
                            str(match_id),
                            possession_number,
                            total_possessions,
                            event_index,
                            error_summary,
                        )
                    )

            if refresh_physical_cache and built_possessions and _pass_success_uses_physical_xpass(model_specs):
                print("  WARN --refresh-physical-cache is ignored during inference; run scripts/generate_physical_xpass.py to fill runtime caches.")

            with tqdm(
                built_possessions,
                total=len(built_possessions),
                desc=f"match {match_id} possessions",
            ) as progress:
                for inference_number, (possession_number, event_index, possession) in enumerate(progress, start=1):
                    progress.set_postfix(event_index=int(event_index))
                    try:
                        progress.write(
                            format_skillcorner_inference_progress(
                                str(match_id),
                                inference_number,
                                len(built_possessions),
                                event_index,
                            )
                        )
                        components = infer_skillcorner_components(possession, model_specs, device=device)
                        runtime_physical_stats = getattr(possession, "physical_xpass_runtime_stats", None)
                        if runtime_physical_stats:
                            physical_xpass_runtime_stats.setdefault(str(match_id), {})[str(event_index)] = runtime_physical_stats
                        physical_skip_stats = getattr(possession, "physical_xpass_skipped_actions", None)
                        if physical_skip_stats:
                            physical_xpass_skipped_actions.setdefault(str(match_id), {})[str(event_index)] = physical_skip_stats
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
                        error_summary = summarize_exception(exc)
                        skipped_possessions.setdefault(str(match_id), []).append(
                            {"event_index": str(event_index), "error": error_summary}
                        )
                        match_stats["skipped_possessions"] += 1
                        progress.write(
                            format_possession_skip(
                                str(match_id),
                                possession_number,
                                total_possessions,
                                event_index,
                                error_summary,
                            )
                        )

            if match_stats["processed_possessions"] == 0:
                skipped_matches["processing_error"].append(str(match_id))
                skipped_match_errors.append({"match_id": str(match_id), "error": "no_usable_possessions"})
                print(f"  SKIP match {match_id}: no usable possessions")
                continue

            match_output_dir = output_dir / str(match_id)
            for component_name, frames in tables_by_component.items():
                component_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                _save_component_table(component_table, match_output_dir / f"{component_name}.parquet")

            stats_by_match[str(match_id)] = match_stats
            processed_matches.append(str(match_id))
            print(format_match_completion(str(match_id), match_stats))
        except Exception as exc:
            skipped_matches["processing_error"].append(str(match_id))
            skipped_match_errors.append({"match_id": str(match_id), "error": summarize_exception(exc)})
            print(f"  SKIP match {match_id}: {summarize_exception(exc)}")

    summary = summarize_skillcorner_stats(stats_by_match, skipped_matches)
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
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_parent": str(output_parent),
        "output_dir": str(output_dir.resolve()),
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": shared_context["return_type"],
        "target_family": shared_context["target_family"],
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
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
        "requested_match_ids": args.match_id or [],
        "limit": args.limit,
        "frames_mode": args.frames_mode,
        "processed_matches": processed_matches,
        "skipped_match_errors": skipped_match_errors,
        "skipped_possessions": skipped_possessions,
        "physical_xpass_runtime_stats": physical_xpass_runtime_stats,
        "physical_xpass_skipped_actions": physical_xpass_skipped_actions,
        "physical_xpass_prewarm_stats": physical_xpass_prewarm_stats,
        "physical_xpass_cache_summary": physical_xpass_cache_summary,
        "models": resolved_model_ids,
        "model_records": model_records,
        "model_feature_signatures": {task: record["feature_signature"] for task, record in model_records.items()},
        "graph_schema": graph_schema,
        "status": "completed",
        **summary,
    }
    if not processed_matches:
        raise RuntimeError("No usable SkillCorner matches were processed.")
    write_run_metadata(output_dir, metadata)
    if output_parent.resolve() == SKILLCORNER_COMPONENT_RUNS_DIR.resolve():
        write_latest_run("skillcorner_component", component_run_id)
    _, postprocessing_summary, summary_path = run_skillcorner_postprocessing(
        component_run_root=output_dir,
        event_data_dir=Path(args.input_dir),
        output_file=output_dir / "skillcorner_summary.csv",
    )
    _, _, _, _, filter_summary, filter_paths = run_skillcorner_filter(
        skillcorner_data_path=summary_path,
        output_dir=output_dir,
    )
    print(f"SkillCorner component run id: {component_run_id}")

    totals = summary["totals"]
    print(f"Saved SkillCorner components to {output_dir}")
    print(f"Saved SkillCorner summary to {summary_path}")
    print(
        "Saved SkillCorner filtered outputs to {actions_raw_path}, {actions_path}, {matches_path}, "
        "and {players_path}".format(
            **filter_paths
        )
    )
    print(
        "SkillCorner postprocessing scored {events_with_dm_score}/{event_rows} events; "
        "filtered actions rows: {skillcorner_actions_rows}.".format(
            **postprocessing_summary,
            **filter_summary,
        )
    )
    if skipped_match_errors or skipped_possessions:
        print(
            f"Skipped {len(skipped_match_errors)} matches and "
            f"{sum(len(items) for items in skipped_possessions.values())} possessions during SkillCorner inference."
        )
    print(
        "Processed {matches} matches, {possessions} possessions, {valid_frames}/{total_frames} valid frames, "
        "{selected_frames} selected frames, {evaluated_frames} evaluated frames, "
        "skipped {skipped_missing_ball} missing-ball frames, "
        "{skipped_missing_possessor} missing-possessor frames, "
        "{skipped_missing_graph} missing-graph frames.".format(**totals)
    )
    print(format_physical_xpass_cache_summary(physical_xpass_cache_summary))


if __name__ == "__main__":
    main()
