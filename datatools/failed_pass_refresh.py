"""Derived, resumable refresh of model-only failed-pass observation inputs."""
from __future__ import annotations

import copy
import json
from datatools.graph_validation import (validate_graphs, graph_fingerprints, assert_preserved,
    graph_validation_enabled, validation_stats, merge_validation_stats)
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import torch

from datatools.failed_pass_trajectory import KEY, VERSION, SETTINGS, detect, observation
from datatools.pass_trajectory import ensure_pass_endpoints, event_key, PASS_DURATION, PASS_DURATION_KEY
from datatools.artifact_storage import clone_feature_artifacts, atomic_parquet_save, atomic_graph_save
from project_config import (DATA_ROOT, get_feature_run_root, load_feature_run_metadata,
                            write_run_metadata, generate_run_id, write_latest_run, atomic_write_json)


def refresh_match(task):
    root, match_id, metadata, thread_limit = task
    root = Path(root)
    enabled = graph_validation_enabled(metadata.get("graph_validation", True))
    stats = validation_stats(enabled)
    from datatools.graph_feature import (build_match_for_feature_generation, configure_worker_thread_limit,
                                        summarize_ball_trajectory, construct_graph_for_action)
    from models.edge_schema import generation_kwargs
    from datatools import config
    configure_worker_thread_limit(thread_limit)
    events = pd.read_parquet(root / "pass_endpoints" / f"{match_id}.parquet")
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineup = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    lineup = lineup.loc[lineup.stats_perform_match_id.astype(str).eq(match_id)]
    match = build_match_for_feature_generation(events, tracking, lineup, SimpleNamespace(
        action_type="all", next_action_conditions_enabled=metadata.get("next_action_conditions_enabled", True),
        reception_policy=metadata.get("reception_policy", "legacy_elastic_v1"),
        max_pass_duration=metadata.get("max_pass_duration")))
    for folder in ("resolved_actions_original", "resolved_actions_angle_only"):
        path = root / folder / f"{match_id}.parquet"
        if path.exists():
            match.actions = pd.read_parquet(path)
            break
    resolver = ensure_pass_endpoints(match, annotate=False)
    tracking = match.tracking
    records = []
    for index, action in match.actions.loc[match.actions.action_type.eq("pass") & ~match.actions.success].iterrows():
        key = event_key(action.get("original_event_id"))
        record = detect(action, tracking, resolver.resolve(action).trajectory, match.fps)
        if not key:
            key = f"missing-event:{index}"
            record.update(status="unavailable", reason="missing_event_identity")
        record.update(event_key=key, action_index=int(index))
        original = summarize_ball_trajectory(match, index)
        record["original_features"] = original.tolist() if original is not None else None
        records.append(record)
    table = pd.DataFrame(records, columns=None if records else ["event_key", "status", "cutoff"])
    if table.event_key.duplicated().any():
        raise ValueError(f"{match_id}: duplicate pass identities")
    atomic_parquet_save(table, root / "failed_pass_trajectories" / f"{match_id}.parquet")
    match.runtime_feature_root = root
    for record in records:
        summary = summarize_ball_trajectory(match, record["action_index"])
        record["updated_features"] = summary.tolist() if summary is not None else None
        if summary is None:
            record.update(status="unavailable", reason=record["reason"] or "unusable_trajectory")
    if records:
        atomic_parquet_save(pd.DataFrame(records), root / "failed_pass_trajectories" / f"{match_id}.parquet")
    by_index = {record["action_index"]: record for record in records}
    # Commit graph files by replacement, never mutate a source run's hard links.
    for folder in root.glob("action_graphs_success_intent*"):
        path = folder / f"{match_id}.pt"
        if not path.exists():
            continue
        labels_path = root / folder.name.replace("action_graphs_success_intent", "success_intent_labels") / path.name
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        labels = torch.load(labels_path, map_location="cpu", weights_only=False)
        if len(graphs) != len(labels):
            raise ValueError(f"Graph/label alignment mismatch: {path}")
        validate_graphs(graphs, context=path, labels=labels, graph_validation=enabled, stats=stats)
        before = graph_fingerprints(graphs, graph_validation=enabled, stats=stats)
        for pos, label in enumerate(labels):
            index = int(label[0])
            if index not in match.actions.index:
                raise ValueError(f"Missing action {index} for {path}")
            action = match.actions.loc[index]
            if action.action_type != "pass" or bool(action.success):
                continue
            summary = summarize_ball_trajectory(match, index)
            if summary is None:
                graphs[pos] = None
            else:
                if graphs[pos] is None:
                    schema = metadata.get("graph_schema") or {}
                    graphs[pos] = construct_graph_for_action(
                        match, index, feature_variant="success_intent", extend=False,
                        add_v_edge_features=bool(schema.get("add_v_edge_features", False)),
                        add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)),
                        **generation_kwargs(schema))
                    if graphs[pos] is None:
                        continue
                graph = graphs[pos]
                col = config.NODE_FEATURE_CORE_DIM
                if (graph.x.ndim != 2 or graph.x.shape[1] < col+4
                        or not graph.x.is_floating_point()):
                    raise ValueError(f"Malformed success-intent graph for action {index}: {path}")
                if by_index[index]["status"] == "shortened":
                    graph.x[:, col:col+4] = graph.x.new_tensor(summary)
                graph.trajectory_end_frame_id = observation(match, action, resolver.resolve(action).trajectory)
                setattr(graph, KEY, VERSION)
        validate_graphs(graphs, context=path, labels=labels, graph_validation=enabled, stats=stats)
        assert_preserved(before, graphs, context=path, labels=labels, allow_availability=True, graph_validation=enabled, stats=stats)
        atomic_graph_save(graphs, path, labels=labels, graph_validation=enabled, stats=stats)
    report = dict(graph_validation=stats, match_id=match_id, total=len(records),
                  counts=pd.Series([r["status"] for r in records], dtype=object).value_counts().to_dict(),
                  unavailable=[dict(action_index=r["action_index"], reason=r["reason"]) for r in records if r["status"] == "unavailable"])
    atomic_write_json(root / "failed_pass_trajectory_refresh" / f"{match_id}.json", report)
    return report


def run_failed_pass_refresh(args):
    base_id = args.extend_feature_run_id
    base = get_feature_run_root(base_id)
    metadata = copy.deepcopy(load_feature_run_metadata(base_id, required=True))
    metadata["graph_validation"] = graph_validation_enabled(getattr(args, "graph_validation", True))
    resume = metadata.get("status") in {"failed", "in_progress"} and metadata.get("refresh_operation") == KEY
    if metadata.get("status") != "completed" and not resume:
        raise ValueError("Source run must be completed, or a failed failed-pass refresh to resume.")
    if args.in_place and args.run_id:
        raise ValueError("--run-id cannot be combined with --in-place.")
    if args.in_place and not resume:
        raise ValueError("This refresh preserves its source. --in-place is only supported to resume a failed refresh.")
    if resume and (not args.in_place or metadata.get("refresh_phase") == "copying"):
        raise ValueError("Resume using --in-place; interrupted cloning requires a new refresh from the original source.")
    files = sorted((base / "pass_endpoints").glob("*.parquet"))
    if not files:
        raise ValueError("Pinned pass endpoints are required. Run the endpoint refresh first.")
    if metadata.get(PASS_DURATION_KEY) != PASS_DURATION:
        raise ValueError("Current canonical pass endpoints are required. Run --refresh-trajectory first.")
    match_ids = [path.stem for path in files]
    if resume and metadata.get("failed_pass_refresh_matches") != match_ids:
        raise ValueError("Pinned match coverage changed during refresh; restart from the original source.")
    output_id = base_id if resume else (args.run_id or generate_run_id("feature"))
    root = get_feature_run_root(output_id)
    if not resume and root.exists():
        raise FileExistsError(root)
    metadata.update(run_id=output_id, status="in_progress", refresh_operation=KEY, refresh_phase="copying",
                    **{KEY: VERSION}, failed_pass_detector_settings=SETTINGS,
                    failed_pass_refresh_matches=match_ids)
    if not resume:
        metadata["derived_from_feature_run_id"] = base_id
    metadata["previous_intended_receiver_model_id"] = metadata.pop("intended_receiver_model_id", metadata.get("previous_intended_receiver_model_id"))
    metadata["model_mode_regeneration_required"] = True
    metadata["intended_receiver_modes"] = [m for m in metadata.get("intended_receiver_modes", []) if m != "model"]
    try:
        write_run_metadata(root, metadata)
        if not resume:
            clone_feature_artifacts(base, root, mode=args.artifact_copy_mode,
                exclude=lambda p: any(part.endswith(("_model", "_model_carries")) for part in p.parts))
        metadata["refresh_phase"] = "processing"
        write_run_metadata(root, metadata)
        tasks = [(str(root), p.stem, metadata, args.worker_thread_limit) for p in files]
        from datatools.graph_feature import resolve_num_workers
        workers = resolve_num_workers(args.num_workers)
        if workers == 1:
            reports = [refresh_match(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                reports = list(pool.map(refresh_match, tasks))
        metadata["failed_pass_refresh_reports"] = reports
        metadata["graph_validation_summary"] = merge_validation_stats(
            [r.get("graph_validation") for r in reports], metadata["graph_validation"])
        if sum(r["total"] for r in reports) and all(r["counts"].get("unavailable", 0) == r["total"] for r in reports):
            raise ValueError("All failed passes unavailable; see failed_pass_trajectory_refresh diagnostics.")
        metadata.update(status="completed", refresh_phase="completed")
        metadata.pop("error", None)
        write_run_metadata(root, metadata)
    except BaseException as exc:
        metadata["failed_pass_refresh_reports"] = [json.loads(path.read_text())
            for path in sorted((root / "failed_pass_trajectory_refresh").glob("*.json"))]
        metadata.update(status="failed", error=str(exc))
        write_run_metadata(root, metadata)
        raise
    write_latest_run("feature", output_id)
    print(f"Refreshed failed-pass trajectories: {output_id}. Extend with --intended-receiver-model-id to regenerate model mode.")
