"""Targeted, recoverable trajectory and pass-duration refresh."""
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
                                    infer_node_feature_dim, resolve_num_workers, summarize_ball_trajectory,
                                    construct_graph_for_action)
from datatools.pass_trajectory import PASS_DURATION, PASS_DURATION_KEY, ENDPOINT_COLUMNS, ensure_pass_endpoints
from models.edge_schema import edge_names, schema_for_names, generation_kwargs
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
        action_type="all", next_action_conditions_enabled=metadata.get("next_action_conditions_enabled", True),
        reception_policy=metadata.get("reception_policy", "legacy_elastic_v1")))
    match.actions = build_success_intent_resolved_actions(match.actions)
    return match


def refresh_match_graphs(graphs, labels, match, *, context, schema=None):
    """Replace trajectory inputs, preserving every other graph feature."""
    if not isinstance(graphs, list) or not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"{context}: malformed graph/label artifacts")
    if len(graphs) != len(labels) or labels.shape[1] < 1:
        raise ValueError(f"{context}: graph/label length mismatch")
    indices = labels[:, 0].cpu().numpy()
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or len(set(indices)) != len(indices):
        raise ValueError(f"{context}: invalid/duplicate action indices")
    column = config.NODE_FEATURE_CORE_DIM
    template = next((graph for graph in graphs if graph is not None), None)
    schema = schema or (schema_for_names(edge_names(template)) if template is not None else {})
    schema = schema_for_names(edge_names(schema))
    for position, (row, graph) in enumerate(zip(labels, graphs)):
        action_index = int(row[0].item())
        if action_index not in match.actions.index:
            if graph is None:
                continue  # An already unavailable action can remain unavailable.
            raise ValueError(f"{context}: action {action_index} is missing from reconstructed actions")
        if graph is not None:
            speed_definition(getattr(graph, TRAJECTORY_SPEED_KEY, None))
        if graph is not None and (not hasattr(graph, "x") or graph.x is None or graph.x.ndim != 2 or graph.x.shape[0] == 0
                or graph.x.shape[1] != infer_node_feature_dim(False, "success_intent")
                or not graph.x.is_floating_point()):
            raise ValueError(f"{context}: malformed success-intent graph for action {action_index}")
        summary = summarize_ball_trajectory(match, action_index)
        if summary is None or not np.isfinite(summary).all():
            graphs[position] = None
            continue
        if graph is None:
            graphs[position] = construct_graph_for_action(
                match, action_index, feature_variant="success_intent",
                add_v_edge_features=bool(schema.get("add_v_edge_features", False)),
                add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)),
                **generation_kwargs(schema))
            continue
        graph.x[:, column:column + 4] = graph.x.new_tensor(summary)
        setattr(graph, TRAJECTORY_SPEED_KEY, TRAJECTORY_SPEED)
    return graphs


def refresh_duration_labels(labels, match, *, context):
    """Patch only duration, including repeated source indices in augmented rows."""
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] <= config.LABEL_INDEX["duration"]:
        raise ValueError(f"{context}: malformed label artifact")
    result = labels.clone()
    indices = labels[:, 0].cpu().numpy()
    valid = np.isfinite(indices) & np.equal(indices, np.floor(indices)) & np.isin(indices, match.actions.index)
    invalid_pass = labels[:, config.LABEL_INDEX["is_pass"]].eq(1).cpu().numpy() & ~valid
    if invalid_pass.any():
        raise ValueError(f"{context}: missing/invalid source action {indices[invalid_pass][0]}")
    durations = match.actions.loc[match.actions.action_type.eq("pass"), "pass_duration"]
    values = pd.Series(indices).map(durations).to_numpy(dtype=float)
    # Source passes can also produce synthetic shot rows. Match by source index,
    # independent of synthetic type flags, while retaining genuine shot/carry durations.
    selected = np.isin(indices, durations.index)
    if not np.isfinite(values[selected]).all():
        raise ValueError(f"{context}: invalid resolved pass duration")
    result[torch.as_tensor(selected, device=result.device), config.LABEL_INDEX["duration"]] = result.new_tensor(values[selected])
    return result


def stage_refresh_match(task):
    file, temp, label_path, metadata, thread_limit, artifacts = task
    configure_worker_thread_limit(thread_limit)
    match = load_refresh_match(file.stem, metadata)
    ensure_pass_endpoints(match)
    staging = ExtensionAttempt(file.parent, "trajectory_staging", "in_place", {})
    if file.is_file():
        graphs = torch.load(file, map_location="cpu", weights_only=False)
        labels = torch.load(label_path, map_location="cpu", weights_only=False)
        graphs = refresh_match_graphs(graphs, labels, match, context=str(file), schema=metadata.get("graph_schema"))
        save_staged_graphs(graphs, temp, staging)
    for path, staged in artifacts:
        if path.suffix == ".pt":
            labels = torch.load(path, map_location="cpu", weights_only=False)
            save_staged_graphs(refresh_duration_labels(labels, match, context=str(path)), staged, staging)
        else:
            actions = pd.read_parquet(path)
            # Carries retain their existing duration/receiver metadata. Source
            # pass indices are stable across receiver modes and carry variants.
            for index in actions.index.intersection(match.actions.index):
                source = match.actions.loc[index]
                if source.get("action_type") == "pass":
                    for column in ENDPOINT_COLUMNS:
                        if column not in actions:
                            actions[column] = np.nan if column.endswith("frame_id") or column == "pass_duration" else ""
                        actions.at[index, column] = source[column]
            if "duration" in actions:
                mask = actions.action_type.eq("pass")
                actions.loc[mask, "duration"] = actions.loc[mask, "pass_duration"]
            actions.to_parquet(staged)


def run_trajectory_refresh(args):
    base_id = str(args.extend_feature_run_id)
    base_root = get_feature_run_root(base_id)
    metadata = load_feature_run_metadata(base_id, required=True)
    previous = metadata.get("extension_attempt") or {}
    recovery = (metadata.get("status") in ("failed", "in_progress")
                and previous.get("operation") in {"refresh_trajectory", "refresh_trajectory_speed"})
    if recovery and previous.get("phase") == "copying":
        raise ValueError("The derived-run clone was interrupted. Create a new derived refresh from the original run; "
                         "an incomplete clone cannot be published as a complete feature run.")
    if metadata.get("status") != "completed" and not recovery:
        raise ValueError(f"Feature run {base_id} is not completed and is not a recoverable trajectory refresh")
    if (metadata.get("status") == "completed" and speed_definition(metadata) == TRAJECTORY_SPEED
            and metadata.get(PASS_DURATION_KEY) == PASS_DURATION):
        print(f"Trajectory and pass duration already current; no-op for {base_id}.")
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
    related = sorted(path for path in base_root.rglob("*") if path.is_file() and (
        (path.suffix == ".pt" and (path.parent.name.startswith(("action_labels", "augmented_labels"))
                                   or path.parent.name == "success_intent_labels"))
        or (path.suffix == ".parquet" and path.parent.name.startswith("resolved_actions"))))
    manifest = sorted(str(path.relative_to(base_root)) for path in [*files, *related])
    expected_manifest = (previous.get("target_settings") or {}).get("artifacts") if recovery else None
    if expected_manifest is not None and manifest != expected_manifest:
        raise ValueError("Trajectory refresh artifact set changed during recovery")
    history = copy.deepcopy(metadata.get("extension_history", []))

    def metadata_factory(status, error, final):
        if recovery and status == "completed" and not final:
            status = "failed"  # A failed repair must not bless the original partial run.
        result = copy.deepcopy(metadata)
        result.update(run_id=output_id, status=status, last_extension_mode="in_place" if in_place else "derived")
        if not in_place:
            result.update(derived_from_feature_run_id=base_id)
        entry = dict(operation="refresh_trajectory", timestamp=datetime.now().isoformat(timespec="seconds"),
                     mode="in_place" if in_place else "derived", status=status,
                     trajectory_speed_definition=TRAJECTORY_SPEED, pass_duration_definition=PASS_DURATION)
        result["extension_history"] = [*history, entry]
        if final:
            result[TRAJECTORY_SPEED_KEY] = TRAJECTORY_SPEED
            result[PASS_DURATION_KEY] = PASS_DURATION
            # Receiver definitions deliberately remain those of the source run.
            result.setdefault("angle_only_definition", "legacy_2s_reception_positions_v1")
        if error:
            result["error"] = error
        else:
            result.pop("error", None)
        return result

    attempt = ExtensionAttempt(output_root, "refresh_trajectory", "in_place" if in_place else "derived",
                               {TRAJECTORY_SPEED_KEY: TRAJECTORY_SPEED, PASS_DURATION_KEY: PASS_DURATION,
                                "graph_files": [file.name for file in files], "artifacts": manifest},
                               metadata_factory, write_run_metadata)
    temporary, primary = [], None
    try:
        if not in_place:
            attempt.begin_mutation("copying", current_file=output_root)
            clone_feature_artifacts(base_root, output_root, mode=getattr(args, "artifact_copy_mode", "hardlink"),
                                    diagnostics=attempt.data)
            attempt.mutation_finished()
        output_files = [get_success_intent_graph_dir(output_root) / file.name for file in files]
        pairs = [(output_root / name, (output_root / name).with_name(Path(name).name + ".trajectory.tmp"))
                 for name in manifest]
        temporary = pairs
        by_match = {}
        for path, staged in pairs:
            if path.parent != get_success_intent_graph_dir(output_root):
                by_match.setdefault(path.stem, []).append((path, staged))
        match_ids = sorted(set(by_match) | {file.stem for file in files})
        tasks = []
        for match_id in match_ids:
            file = get_success_intent_graph_dir(output_root) / f"{match_id}.pt"
            tasks.append((file, file.with_name(file.name + ".trajectory.tmp"),
                          get_success_intent_label_dir(output_root) / file.name, metadata,
                          int(getattr(args, "worker_thread_limit", 1)), by_match.get(match_id, [])))
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
    print(f"Refreshed trajectory and pass duration: {output_id}. Retrain success_intent before model replacement.")


# Python entrypoint compatibility; the CLI also retains the deprecated spelling.
run_trajectory_speed_refresh = run_trajectory_refresh
