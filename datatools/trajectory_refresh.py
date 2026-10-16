"""Targeted, recoverable replacement of success-intent trajectory speed."""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.artifact_storage import clone_feature_artifacts
from datatools.graph_feature import (build_match_for_feature_generation, configure_worker_thread_limit,
                                    infer_node_feature_dim, resolve_num_workers, summarize_ball_trajectory)
from datatools.success_intent import build_success_intent_resolved_actions
from datatools.trajectory import (TRAJECTORY_SPEED, TRAJECTORY_SPEED_KEY, speed_definition)
from project_config import (DATA_ROOT, get_feature_run_root, get_success_intent_graph_dir,
                            get_success_intent_label_dir, generate_run_id, load_feature_run_metadata,
                            write_latest_run, write_run_metadata)
from scripts.feature_extension_diagnostics import (ExtensionAttempt, cleanup_temporary_files,
                                                  replace_staged_graphs, save_staged_graphs)


def load_refresh_match(match_id, metadata):
    """Reconstruct action filtering, but do not relabel returns or infer receivers."""
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineup = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    lineup = lineup.loc[lineup.stats_perform_match_id.astype(str).eq(str(match_id))].copy()
    match = build_match_for_feature_generation(events, tracking, lineup, SimpleNamespace(
        action_type="all", next_action_conditions_enabled=metadata.get("next_action_conditions_enabled", True)))
    match.actions = build_success_intent_resolved_actions(match.actions)
    return match


def refresh_match_graphs(graphs, labels, match, *, context):
    """Mutate only the speed scalar and definition tag, aligned to saved labels."""
    if not isinstance(graphs, list) or not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"{context}: malformed graph/label artifacts")
    if len(graphs) != len(labels) or labels.shape[1] < 1:
        raise ValueError(f"{context}: graph/label length mismatch")
    indices = labels[:, 0].cpu().numpy()
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or len(set(indices)) != len(indices):
        raise ValueError(f"{context}: invalid/duplicate action indices")
    column = config.NODE_FEATURE_CORE_DIM
    for row, graph in zip(labels, graphs):
        if graph is None:
            continue
        action_index = int(row[0].item())
        speed_definition(getattr(graph, TRAJECTORY_SPEED_KEY, None))  # Reject unknown versions.
        if action_index not in match.actions.index:
            raise ValueError(f"{context}: action {action_index} is missing from reconstructed actions")
        if (not hasattr(graph, "x") or graph.x is None or graph.x.ndim != 2 or graph.x.shape[0] == 0
                or graph.x.shape[1] != infer_node_feature_dim(False, "success_intent")
                or not graph.x.is_floating_point()):
            raise ValueError(f"{context}: malformed success-intent graph for action {action_index}")
        summary = summarize_ball_trajectory(match, action_index)
        if summary is None or not np.isfinite(summary).all():
            raise ValueError(f"{context}: invalid trajectory for action {action_index}")
        direction = graph.x.new_tensor(summary[1:]).expand(graph.x.shape[0], -1)
        if not torch.allclose(graph.x[:, column + 1:column + 4], direction, atol=1e-6, rtol=1e-5):
            raise ValueError(f"{context}: stored direction disagrees for action {action_index}; source/alignment changed")
        graph.x[:, column] = float(summary[0])
        setattr(graph, TRAJECTORY_SPEED_KEY, TRAJECTORY_SPEED)
    return graphs


def stage_refresh_match(task):
    file, temp, label_path, metadata, thread_limit = task
    configure_worker_thread_limit(thread_limit)
    graphs = torch.load(file, map_location="cpu", weights_only=False)
    labels = torch.load(label_path, map_location="cpu", weights_only=False)
    match = load_refresh_match(file.stem, metadata)
    graphs = refresh_match_graphs(graphs, labels, match, context=str(file))
    save_staged_graphs(graphs, temp, ExtensionAttempt(file.parent, "trajectory_speed_staging", "in_place", {}))


def run_trajectory_speed_refresh(args):
    base_id = str(args.extend_feature_run_id)
    base_root = get_feature_run_root(base_id)
    metadata = load_feature_run_metadata(base_id, required=True)
    previous = metadata.get("extension_attempt") or {}
    recovery = (metadata.get("status") in ("failed", "in_progress")
                and previous.get("operation") == "refresh_trajectory_speed")
    if recovery and previous.get("phase") == "copying":
        raise ValueError("The derived-run clone was interrupted. Create a new derived refresh from the original run; "
                         "an incomplete clone cannot be published as a complete feature run.")
    if metadata.get("status") != "completed" and not recovery:
        raise ValueError(f"Feature run {base_id} is not completed and is not a recoverable trajectory refresh")
    if metadata.get("status") == "completed" and speed_definition(metadata) == TRAJECTORY_SPEED:
        print(f"Trajectory speed already uses {TRAJECTORY_SPEED}; no-op for {base_id}.")
        return
    in_place = bool(getattr(args, "in_place", False))
    if in_place and getattr(args, "run_id", None):
        raise ValueError("--run-id cannot be used with --in-place (alias --overwrite-feature-run).")
    workers = resolve_num_workers(getattr(args, "num_workers", "1"))
    if recovery and not in_place:
        raise ValueError("Resume an interrupted trajectory refresh using --in-place on its target run.")
    output_id = base_id if in_place else (args.run_id or generate_run_id("feature"))
    output_root = get_feature_run_root(output_id)
    if not in_place and (output_id == base_id or output_root.exists()):
        raise FileExistsError(f"Derived trajectory refresh requires a new output run: {output_root}")
    files = sorted(get_success_intent_graph_dir(base_root).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No success-intent graph artifacts in {base_root}")
    expected_files = (previous.get("target_settings") or {}).get("graph_files") if recovery else None
    if expected_files is not None and [file.name for file in files] != expected_files:
        raise ValueError("Success-intent graph file set changed during refresh recovery; inspect the target before repair.")
    for file in files:
        label_path = get_success_intent_label_dir(base_root) / file.name
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
    history = copy.deepcopy(metadata.get("extension_history", []))

    def metadata_factory(status, error, final):
        if recovery and status == "completed" and not final:
            status = "failed"  # A failed repair must not bless the original partial run.
        result = copy.deepcopy(metadata)
        result.update(run_id=output_id, status=status, last_extension_mode="in_place" if in_place else "derived")
        if not in_place:
            result.update(derived_from_feature_run_id=base_id)
        entry = dict(operation="refresh_trajectory_speed", timestamp=datetime.now().isoformat(timespec="seconds"),
                     mode="in_place" if in_place else "derived", status=status,
                     trajectory_speed_definition=TRAJECTORY_SPEED)
        result["extension_history"] = [*history, entry]
        if final:
            result[TRAJECTORY_SPEED_KEY] = TRAJECTORY_SPEED
        if error:
            result["error"] = error
        else:
            result.pop("error", None)
        return result

    attempt = ExtensionAttempt(output_root, "refresh_trajectory_speed", "in_place" if in_place else "derived",
                               {TRAJECTORY_SPEED_KEY: TRAJECTORY_SPEED, "graph_files": [file.name for file in files]},
                               metadata_factory, write_run_metadata)
    temporary, primary = [], None
    try:
        if not in_place:
            attempt.begin_mutation("copying", current_file=output_root)
            clone_feature_artifacts(base_root, output_root, mode=getattr(args, "artifact_copy_mode", "hardlink"),
                                    diagnostics=attempt.data)
            attempt.mutation_finished()
        output_files = [get_success_intent_graph_dir(output_root) / file.name for file in files]
        temporary = [(file, file.with_name(file.name + ".trajectory_speed.tmp")) for file in output_files]
        tasks = [(file, temp, get_success_intent_label_dir(output_root) / file.name, metadata,
                  int(getattr(args, "worker_thread_limit", 1))) for file, temp in temporary]
        attempt.enter("staging", current_file=output_files[0])
        if workers == 1:
            for task in tasks:
                attempt.data["current_file"] = str(task[0])
                stage_refresh_match(task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(stage_refresh_match, tasks))
        replace_staged_graphs(temporary, attempt)
        attempt.finish()
    except BaseException as error:
        primary = error
        cleanup_temporary_files(temporary, attempt, primary)
        attempt.fail(error)
        raise
    finally:
        if primary is None:
            cleanup_temporary_files(temporary, attempt)
    if not in_place:
        write_latest_run("feature", output_id)
    print(f"Refreshed trajectory speed: {output_id} ({TRAJECTORY_SPEED}). Retrain success_intent before model replacement.")
