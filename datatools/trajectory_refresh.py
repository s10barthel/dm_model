"""Targeted, recoverable trajectory and pass-duration refresh."""
from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from datatools.graph_validation import (validate_graphs, graph_fingerprints, assert_preserved,
    validation_stats, graph_validation_enabled, merge_validation_stats)

from datatools import config
from datatools.artifact_storage import clone_feature_artifacts
from datatools.graph_feature import (build_match_for_feature_generation, configure_worker_thread_limit,
                                    infer_node_feature_dim, resolve_num_workers, summarize_ball_trajectory,
                                    construct_graph_for_action)
from datatools.pass_trajectory import (PASS_DURATION, PASS_DURATION_KEY, ENDPOINT_COLUMNS, ensure_pass_endpoints,
                                      ANGLE_ONLY_KEY, ANGLE_ONLY, ORIGINAL_KEY, ORIGINAL, MAX_DURATION_KEY)
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
    config.PASS_HEIGHT_THRESHOLD_METERS = metadata.get("pass_height_threshold_meters", config.PASS_HEIGHT_THRESHOLD_METERS)
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineup = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    lineup = lineup.loc[lineup.stats_perform_match_id.astype(str).eq(str(match_id))].copy()
    from datatools.reception_policy import apply_policy
    from datatools.pass_trajectory import prepare_pass_events
    events = apply_policy(events, metadata.get("reception_policy", "legacy_elastic_v1"))
    events = prepare_pass_events(events, tracking, match_id, metadata.get(MAX_DURATION_KEY))
    if metadata.get("_endpoint_root"):
        path = Path(metadata["_endpoint_root"]) / "pass_endpoints" / f"{match_id}.parquet.trajectory.tmp"
        path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(path)
    match = build_match_for_feature_generation(events, tracking, lineup, SimpleNamespace(
        action_type="all", next_action_conditions_enabled=metadata.get("next_action_conditions_enabled", True),
        reception_policy=metadata.get("reception_policy", "legacy_elastic_v1"),
        max_pass_duration=metadata.get(MAX_DURATION_KEY)))
    match.actions = build_success_intent_resolved_actions(match.actions)
    return match


def refresh_match_graphs(graphs, labels, match, *, context, schema=None, graph_validation=True, stats=None):
    """Replace trajectory inputs, preserving every other graph feature."""
    if not isinstance(graphs, list) or not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"{context}: malformed graph/label artifacts")
    if len(graphs) != len(labels) or labels.shape[1] < 1:
        raise ValueError(f"{context}: graph/label length mismatch")
    indices = labels[:, 0].cpu().numpy()
    if not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or len(set(indices)) != len(indices):
        raise ValueError(f"{context}: invalid/duplicate action indices")
    validate_graphs(graphs, context=context, labels=labels, graph_validation=graph_validation, stats=stats)
    before = graph_fingerprints(graphs, graph_validation=graph_validation, stats=stats)
    column = config.NODE_FEATURE_CORE_DIM
    template = next((graph for graph in graphs if graph is not None), None)
    schema = schema or (schema_for_names(edge_names(template)) if template is not None else {})
    schema = schema_for_names(edge_names(schema))
    match._trajectory_refresh_skips = []
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
            if match.actions.at[action_index, "action_type"] == "pass":
                reason = match.actions.loc[action_index].get("pass_endpoint_reason") or "unusable_trajectory"
                match._trajectory_refresh_skips.append(dict(action_index=action_index, reason=reason))
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
        endpoint = ensure_pass_endpoints(match, annotate=False).resolve(match.actions.loc[action_index])
        graph.trajectory_start_frame_id = int(match.actions.at[action_index, "frame_id"])
        from datatools.failed_pass_trajectory import observation, KEY, VERSION
        graph.trajectory_end_frame_id = observation(match, match.actions.loc[action_index], endpoint.trajectory)
        setattr(graph, KEY, VERSION)
        graph.pass_endpoint_definition = PASS_DURATION
    validate_graphs(graphs, context=context, labels=labels, graph_validation=graph_validation, stats=stats)
    assert_preserved(before, graphs, context=context, labels=labels, allow_availability=True,
                     graph_validation=graph_validation, stats=stats)
    return graphs


def refresh_duration_labels(labels, match, *, context):
    """Refresh endpoint-derived targets, including repeated source indices."""
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
    result[torch.as_tensor(selected, device=result.device), config.LABEL_INDEX["duration"]] = result.new_tensor(values[selected])
    for position in np.flatnonzero(selected):
        action = match.actions.loc[int(indices[position])]
        if hasattr(match, "pass_height_labels"):
            height, high = match.pass_height_labels(action.frame_id, action.receive_frame_id)
            for name, value in (("pass_max_ball_z", height), ("pass_high", high)):
                result[position, config.LABEL_INDEX[name]] = float(value)
        for axis, extent in (("x", config.FIELD_SIZE[0]), ("y", config.FIELD_SIZE[1])):
            name = "end_" + axis
            if name in action and name in config.LABEL_INDEX:
                value = action[name]
                if str(action.get("object_id", "")).startswith("away"):
                    value = extent - value
                result[position, config.LABEL_INDEX[name]] = float(value)
    return result


def artifact_mode(name):
    name = name.removesuffix("_carries")
    return "model" if name.endswith("_model") else "angle_only" if name.endswith("_angle_only") else "original"


def patch_receiver_targets(labels, actions, tracking):
    """Patch real/temporal heuristic targets, retaining returns and other supervision."""
    result = labels.clone()
    lookup = {}
    for index, action in actions.loc[actions.action_type.eq("pass")].iterrows():
        player, frame = action.get("intent_id"), action.get("frame_id")
        intent_index, x, y = -1, -config.FIELD_SIZE[0], -config.FIELD_SIZE[1]
        if pd.notna(frame) and frame in tracking.index and isinstance(player, str):
            prefix = str(action.object_id)[:4]
            snapshot = tracking.loc[frame].dropna()
            players = [c[:-2] for c in snapshot.index if re.fullmatch(rf"{prefix}_(?:\d+|goal)_x", c)]
            if player in players:
                intent_index = players.index(player)
            end = action.get("receive_frame_id")
            if pd.notna(end) and end in tracking.index:
                values = tracking.loc[end].reindex([player + "_x", player + "_y"]).to_numpy(dtype=float)
                if np.isfinite(values).all():
                    x, y = values
                    if prefix == "away":
                        x, y = config.FIELD_SIZE[0] - x, config.FIELD_SIZE[1] - y
        lookup[int(index)] = (intent_index, x, y)
    for position, row in enumerate(labels):
        values = lookup.get(int(row[0].item()))
        if values is not None:
            for column, value in zip(("intent_index", "intent_x", "intent_y"), values):
                result[position, config.LABEL_INDEX[column]] = value
    return result


def refresh_artifact_paths(root, match_id=None):
    """All rewritten files, including graph partners of regenerated augmentations."""
    paths = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        selected = (name == "action_graphs_success_intent" or name == "success_intent_labels"
            or name.startswith(("action_labels", "augmented_labels", "resolved_actions"))
            or name in {"augmented_graphs", "augmented_graphs_angle_only", "post_action_graphs"})
        if selected:
            paths.extend(p for p in folder.glob("*") if p.suffix in {".pt", ".parquet"}
                         and (match_id is None or p.stem == match_id))
    return sorted(paths)


def stage_refresh_match(task):
    file, temp, label_path, metadata, thread_limit, artifacts = task
    configure_worker_thread_limit(thread_limit)
    enabled = graph_validation_enabled(metadata.get("graph_validation", True))
    stats = validation_stats(enabled)
    match = load_refresh_match(file.stem, dict(metadata, _endpoint_root=str(file.parent.parent)))
    match.max_pass_duration = metadata.get(MAX_DURATION_KEY)
    # Use persisted canonical identity/membership, even if current filtering differs.
    root = file.parent.parent
    resolved_paths = [p for p, _ in artifacts if p.parent.name.startswith("resolved_actions")
                      and not p.parent.name.endswith("_carries")]
    if resolved_paths:
        match.actions = pd.read_parquet(resolved_paths[0])
    resolver = ensure_pass_endpoints(match, annotate=False)
    match.actions = resolver.annotate(match.actions)
    from datatools.failed_pass_trajectory import prepare_new_match, VERSION
    match.runtime_feature_root = root
    match.match_id = file.stem
    match.failed_pass_trajectory_policy = VERSION
    prepare_new_match(match, staged_path=root / "failed_pass_trajectories" / f"{file.stem}.parquet.trajectory.tmp")
    endpoint_stage = root / "pass_endpoints" / f"{file.stem}.parquet.trajectory.tmp"
    if not endpoint_stage.exists():
        endpoint_stage.parent.mkdir(parents=True, exist_ok=True)
        match.actions.to_parquet(endpoint_stage)
    canonical = match.actions.copy()
    staging = ExtensionAttempt(root, "trajectory_staging", "in_place", {})
    if file.is_file():
        graphs = torch.load(file, map_location="cpu", weights_only=False)
        labels = torch.load(label_path, map_location="cpu", weights_only=False)
        graphs = refresh_match_graphs(graphs, labels, match, context=str(file), schema=metadata.get("graph_schema"), graph_validation=enabled, stats=stats)
        save_staged_graphs(graphs, temp, staging, graph_validation=enabled, labels=labels, context=file)
    from datatools.utils import label_intended_receivers
    frames, tensors, graph_labels = {}, {}, {}
    changed = {}
    for path, staged in artifacts:
        if path.suffix != ".parquet":
            continue
        actions = resolver.annotate(pd.read_parquet(path))
        mode = artifact_mode(path.parent.name)
        if mode in {"original", "angle_only"}:
            labeled = label_intended_receivers(actions, match.tracking, action_type="all",
                                               intended_receiver_mode=mode, fps=match.fps)
            mask = actions.action_type.eq("pass")
            previous = actions.get("intent_id", pd.Series(None, index=actions.index, dtype=object))
            changed[path.parent.name] = int((previous.fillna("") != labeled.intent_id.fillna(""))[mask].sum())
            actions.loc[mask, "intent_id"] = labeled.loc[mask, "intent_id"]
        for index, row in actions.loc[actions.action_type.eq("pass")].iterrows():
            if hasattr(match, "pass_height_labels"):
                height, high = match.pass_height_labels(row.frame_id, row.receive_frame_id)
                actions.loc[index, ["pass_max_ball_z", "pass_high"]] = [height, high]
            for axis in ("x", "y"):
                column = "intent_" + axis
                player_column = str(row.get("intent_id")) + "_" + axis
                if column in actions:
                    actions.loc[index, column] = (match.tracking.at[row.receive_frame_id, player_column]
                        if pd.notna(row.receive_frame_id) and row.receive_frame_id in match.tracking.index
                        and player_column in match.tracking else np.nan)
        if "duration" in actions:
            mask = actions.action_type.eq("pass")
            actions.loc[mask, "duration"] = actions.loc[mask, "pass_duration"]
        frames[(mode, path.parent.name.endswith("_carries"))] = actions
        actions.to_parquet(staged)
    for path, staged in artifacts:
        if path.suffix != ".pt" or "labels" not in path.parent.name:
            continue
        labels = torch.load(path, map_location="cpu", weights_only=False)
        labels = refresh_duration_labels(labels, match, context=str(path))
        mode = artifact_mode(path.parent.name)
        actions = frames.get((mode, path.parent.name.endswith("_carries")))
        if actions is not None and mode in {"original", "angle_only"} and path.parent.name != "success_intent_labels" and not path.parent.name.startswith("augmented_labels"):
            labels = patch_receiver_targets(labels, actions, match.tracking)
        tensors[path] = labels
    # Blocked-sample membership depends on the real intended receiver.
    from datatools.graph_feature import augment_blocked_actions
    from project_config import get_action_label_dir, get_augmented_label_dir, get_augmented_feature_dir
    for mode in ("original", "angle_only"):
        graph_path = get_augmented_feature_dir(mode, root=root) / file.name
        augmented_path = get_augmented_label_dir(mode, root=root) / file.name
        if not graph_path.exists() and not augmented_path.exists():
            continue
        if not graph_path.exists() or not augmented_path.exists():
            raise ValueError(f"Missing augmented graph/label partner for {file.stem}/{mode}")
        actions = frames.get((mode, False))
        if actions is None:
            raise ValueError(f"Missing resolved {mode} actions for augmentation refresh")
        return_types = metadata.get("return_types") or ["disc_0.9"]
        base_label = get_action_label_dir(return_types[0], intended_receiver_mode=mode, root=root) / file.name
        base_graph = root / "action_graphs" / file.name
        if base_label not in tensors or not base_graph.exists():
            raise ValueError(f"Missing canonical graph/label inputs for {file.stem}/{mode}")
        match.labels = tensors[base_label]
        indices = match.labels[:, 0].long().tolist()
        if len(indices) != len(set(indices)) or not set(indices).issubset(actions.index):
            raise ValueError(f"Canonical action/label alignment mismatch: {base_label}")
        match.actions = actions.loc[indices].copy()
        match.graph_features_0 = torch.load(base_graph, map_location="cpu", weights_only=False)
        validate_graphs(match.graph_features_0, context=base_graph, labels=match.labels,
                        graph_validation=enabled, stats=stats)
        if len(match.graph_features_0) != len(indices):
            raise ValueError(f"Canonical graph/label alignment mismatch: {base_graph}")
        graphs, labels = augment_blocked_actions(match, verbose=False)
        tensors[graph_path], tensors[augmented_path] = graphs, labels
        graph_labels[graph_path] = labels
    # Rebuild post-action observations, whose initial frame is the pass endpoint.
    for path, _ in artifacts:
        if path.parent.name != "post_action_graphs":
            continue
        from datatools.graph_feature import construct_graph_features
        actions = frames.get(("original", False), canonical)
        match.actions = actions.copy()
        label_path = get_action_label_dir((metadata.get("return_types") or ["disc_0.9"])[0],
                                         intended_receiver_mode="original", root=root) / file.name
        labels = tensors.get(label_path)
        if labels is None:
            labels = torch.load(label_path, map_location="cpu", weights_only=False)
        schema = metadata.get("graph_schema") or {}
        graph_labels[path] = labels
        tensors[path] = construct_graph_features(match, action_indices=labels[:, 0].long().tolist(),
            post_action=True, verbose=False, add_v_edge_features=bool(schema.get("add_v_edge_features", False)),
            add_relative_speed_edge_features=bool(schema.get("add_relative_speed_edge_features", False)),
            **generation_kwargs(schema))
    for path, staged in artifacts:
        if path.suffix == ".pt":
            if path not in tensors:
                raise ValueError(f"Unstaged dependent artifact: {path}")
            save_staged_graphs(tensors[path], staged, staging, graph_payload=isinstance(tensors[path], list),
                               graph_validation=enabled, labels=graph_labels.get(path), context=path)
    failed = canonical.loc[canonical.action_type.eq("pass") & canonical.observed_end_frame_id.isna()]
    stats = merge_validation_stats([stats, staging.data.get("graph_validation")], enabled)
    report = dict(graph_validation=stats, match_id=file.stem, trajectory_skips=getattr(match, "_trajectory_refresh_skips", []), skipped_actions=[dict(action_index=int(i),
        action_id=str(a.get("action_id", i)), reason=a.pass_endpoint_reason) for i, a in failed.iterrows()],
        pass_count=int(canonical.action_type.eq("pass").sum()),
        usable_pass_count=int((canonical.action_type.eq("pass") & canonical.receive_frame_id.notna()).sum()),
        endpoint_sources=canonical.loc[canonical.action_type.eq("pass"), "pass_endpoint_source"].value_counts().to_dict(),
        changed_endpoint_count=int((canonical.loc[canonical.action_type.eq("pass"), "receive_frame_id"].fillna(-1)
            != canonical.loc[canonical.action_type.eq("pass"), "original_receive_frame_id"].fillna(-1)).sum()),
        heuristic_changes=changed, pass_endpoint_definition=PASS_DURATION,
        max_pass_duration=metadata.get(MAX_DURATION_KEY))
    return report


def refresh_directory_match(root, metadata, match_id, thread_limit=1, *, graph_validation=True):
    """Stage one match completely before replacing its dependent files."""
    metadata = dict(metadata, graph_validation=graph_validation_enabled(graph_validation))
    root = Path(root)
    file = get_success_intent_graph_dir(root) / (match_id + ".pt")
    paths = refresh_artifact_paths(root, match_id)
    paths += [root / folder / f"{match_id}.parquet" for folder in ("pass_endpoints", "failed_pass_trajectories")]
    pairs = [(p, p.with_name(p.name + ".trajectory.tmp")) for p in paths]
    artifacts = [(p, t) for p, t in pairs if p != file and p.parent.name not in {"pass_endpoints", "failed_pass_trajectories"}]
    try:
        report = stage_refresh_match((file, file.with_name(file.name + ".trajectory.tmp"),
            get_success_intent_label_dir(root) / file.name, metadata, thread_limit, artifacts))
        for path, temporary in pairs:
            temporary.replace(path)
        from project_config import atomic_write_json
        atomic_write_json(root / "trajectory_refresh" / (match_id + ".json"), report)
        return report
    finally:
        for _, temporary in pairs:
            temporary.unlink(missing_ok=True)


def run_trajectory_refresh(args):
    base_id = str(args.extend_feature_run_id)
    base_root = get_feature_run_root(base_id)
    metadata = load_feature_run_metadata(base_id, required=True)
    metadata["graph_validation"] = graph_validation_enabled(getattr(args, "graph_validation", True))
    previous = metadata.get("extension_attempt") or {}
    recovery = (metadata.get("status") in ("failed", "in_progress")
                and previous.get("operation") in {"refresh_trajectory", "refresh_trajectory_speed"})
    if recovery and previous.get("phase") == "copying":
        raise ValueError("The derived-run clone was interrupted. Create a new derived refresh from the original run; "
                         "an incomplete clone cannot be published as a complete feature run.")
    if metadata.get("status") != "completed" and not recovery:
        raise ValueError(f"Feature run {base_id} is not completed and is not a recoverable trajectory refresh")
    if (metadata.get("status") == "completed" and speed_definition(metadata) == TRAJECTORY_SPEED
            and metadata.get(PASS_DURATION_KEY) == PASS_DURATION
            and metadata.get(ANGLE_ONLY_KEY) == ANGLE_ONLY and metadata.get(ORIGINAL_KEY) == ORIGINAL
            and metadata.get(MAX_DURATION_KEY) == getattr(args, MAX_DURATION_KEY, None)):
        print(f"Trajectory and pass duration already current; no-op for {base_id}.")
        return
    if (metadata.get("carry_variant") or {}).get("available"):
        raise ValueError("Refreshing pass endpoints in a carry-enabled run requires --replace-carries.")
    if "model" in metadata.get("intended_receiver_modes", []):
        metadata["model_mode_regeneration_required"] = True
        metadata["previous_intended_receiver_model_id"] = metadata.pop("intended_receiver_model_id", None)
        metadata["intended_receiver_modes"] = [m for m in metadata["intended_receiver_modes"] if m != "model"]
    metadata[MAX_DURATION_KEY] = getattr(args, MAX_DURATION_KEY, None)
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
    related = [p for p in refresh_artifact_paths(base_root) if p not in files
               and not (metadata.get("model_mode_regeneration_required") and p.parent.name.endswith(("_model", "_model_carries")))]
    manifest = sorted({str(path.relative_to(base_root)) for path in [*files, *related]} |
        {f"{folder}/{file.stem}.parquet" for file in files for folder in ("pass_endpoints", "failed_pass_trajectories")})
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
            from datatools.pass_eligibility import policy_metadata
            result.update(policy_metadata())
            result[TRAJECTORY_SPEED_KEY] = TRAJECTORY_SPEED
            result[PASS_DURATION_KEY] = PASS_DURATION
            result[ANGLE_ONLY_KEY] = ANGLE_ONLY
            result[ORIGINAL_KEY] = ORIGINAL
            result[MAX_DURATION_KEY] = metadata.get(MAX_DURATION_KEY)
            reports = [json.loads(p.read_text()) for p in (output_root / "trajectory_refresh").glob("*.json")]
            result["skipped_actions"] = {r["match_id"]: r["skipped_actions"] for r in reports if r["skipped_actions"]}
            result["skipped_action_count"] = sum(len(r["skipped_actions"]) for r in reports)
            result["graph_validation_summary"] = merge_validation_stats(
                [r.get("graph_validation") for r in reports], metadata["graph_validation"])
        if error:
            reports = [json.loads(p.read_text()) for p in (output_root / "trajectory_refresh").glob("*.json")]
            result["skipped_actions"] = {r["match_id"]: r["skipped_actions"] for r in reports if r["skipped_actions"]}
            result["skipped_action_count"] = sum(len(r["skipped_actions"]) for r in reports)
            result["error"] = error
        else:
            result.pop("error", None)
        return result

    attempt = ExtensionAttempt(output_root, "refresh_trajectory", "in_place" if in_place else "derived",
                               {TRAJECTORY_SPEED_KEY: TRAJECTORY_SPEED, PASS_DURATION_KEY: PASS_DURATION,
                                ANGLE_ONLY_KEY: ANGLE_ONLY, ORIGINAL_KEY: ORIGINAL, MAX_DURATION_KEY: metadata.get(MAX_DURATION_KEY),
                                "graph_files": [file.name for file in files], "artifacts": manifest},
                               metadata_factory, write_run_metadata)
    temporary, primary = [], None
    try:
        if not in_place:
            attempt.begin_mutation("copying", current_file=output_root)
            clone_feature_artifacts(base_root, output_root, mode=getattr(args, "artifact_copy_mode", "hardlink"),
                                    diagnostics=attempt.data,
                                    exclude=lambda p: metadata.get("model_mode_regeneration_required", False)
                                    and any(part.endswith(("_model", "_model_carries")) for part in p.parts))
            attempt.mutation_finished()
        output_files = [get_success_intent_graph_dir(output_root) / file.name for file in files]
        pairs = [(output_root / name, (output_root / name).with_name(Path(name).name + ".trajectory.tmp"))
                 for name in manifest]
        temporary = pairs
        by_match = {}
        for path, staged in pairs:
            if path.parent != get_success_intent_graph_dir(output_root) and path.parent.name not in {"pass_endpoints", "failed_pass_trajectories"}:
                by_match.setdefault(path.stem, []).append((path, staged))
        match_ids = sorted(set(by_match) | {file.stem for file in files})
        tasks = []
        for match_id in match_ids:
            file = get_success_intent_graph_dir(output_root) / f"{match_id}.pt"
            tasks.append((file, file.with_name(file.name + ".trajectory.tmp"),
                          get_success_intent_label_dir(output_root) / file.name, metadata,
                          int(getattr(args, "worker_thread_limit", 1)), by_match.get(match_id, [])))
        attempt.enter("staging", current_file=output_files[0])
        reports = []
        if workers == 1:
            for task in tasks:
                attempt.data["current_file"] = str(task[0])
                reports.append(stage_refresh_match(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                reports = list(pool.map(stage_refresh_match, tasks))
        replace_staged_graphs(temporary, attempt)
        from project_config import atomic_write_json
        for report in reports:
            atomic_write_json(output_root / "trajectory_refresh" / (report["match_id"] + ".json"), report)
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
