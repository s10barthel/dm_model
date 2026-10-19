"""Derived-only carry replacement, reusing all canonical training samples.

Each match commits a completion marker last. A failed derived run can be resumed
with its explicit run ID; uncommitted match artifacts are safely replaced.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pandas as pd
import torch

from datatools.artifact_storage import atomic_parquet_save, atomic_torch_save, clone_feature_artifacts
from datatools.ball_carries import augment_match_actions_with_carries, derive_carry_segments
from datatools.carry_policy import DEFINITION, CarryPolicy
from datatools.graph_feature import (build_match_for_feature_generation, configure_worker_thread_limit,
                                     construct_graph_features)
from models.edge_schema import generation_kwargs
from project_config import (DATA_ROOT, CONTROL_EVENTS_SYNCED_DIR, get_feature_run_root,
    get_action_graph_dir, get_post_action_graph_dir, get_action_label_dir, get_resolved_action_path,
    atomic_write_json, generate_run_id, load_feature_run_metadata, write_latest_run, write_run_metadata)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible(root):
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
        return metadata.get("status") == "completed" and bool((metadata.get("carry_variant") or {}).get("available"))
    except (OSError, ValueError):
        return False


def resolve_source(value):
    if value != "latest":
        root = get_feature_run_root(value)
        if not compatible(root):
            raise ValueError(f"Source must be a completed carry-enabled feature run: {value}")
        return root
    roots = sorted(get_feature_run_root("placeholder").parent.glob("feature_*"), reverse=True)
    for root in roots:
        if compatible(root):
            try:
                metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
                match_ids(root, metadata)
            except (KeyError, ValueError, FileNotFoundError):
                continue
            return root
    return _no_source()


def _no_source():
    raise FileNotFoundError("No completed carry-enabled source feature run found")


def excluded_artifact(relative):
    return any(part.endswith("_carries") or part in {"carry_sidecars", "carry_replacement"} for part in relative.parts)


def match_ids(source, metadata):
    modes = metadata["intended_receiver_modes"]
    ids = sorted(p.stem for p in Path(get_action_graph_dir(source)).glob("*.pt"))
    if not ids:
        raise ValueError("Source has no canonical action graphs")
    for match_id in ids:
        needed = [Path(get_post_action_graph_dir(source)) / f"{match_id}.pt"]
        for mode in modes:
            needed.append(get_resolved_action_path(match_id, intended_receiver_mode=mode, root=source))
            for ret in metadata["return_types"]:
                needed.append(Path(get_action_label_dir(ret, intended_receiver_mode=mode, root=source)) / f"{match_id}.pt")
        if any(not p.exists() for p in needed):
            raise FileNotFoundError(f"Incomplete canonical source artifacts for {match_id}")
    return ids


def rebuild_match(task):
    source, output, metadata, match_id, policy_values, thread_limit = task
    source, output = Path(source), Path(output)
    configure_worker_thread_limit(thread_limit)
    start = time.perf_counter()
    policy = CarryPolicy(**policy_values)
    event_path = DATA_ROOT / "event_synced" / f"{match_id}.csv"
    control_path = CONTROL_EVENTS_SYNCED_DIR / f"{match_id}.parquet"
    # A new derived version can itself be a source; prefer its pinned controls.
    pinned = source / "carry_sidecars" / "control_events" / f"{match_id}.parquet"
    if pinned.exists():
        control_path = pinned
    events = pd.read_csv(event_path, parse_dates=["utc_timestamp"])
    controls = pd.read_parquet(control_path)
    tracking_path = DATA_ROOT / "tracking_processed" / f"{match_id}.parquet"
    tracking = pd.read_parquet(tracking_path)
    segments, audit = derive_carry_segments(controls, events, tracking, policy=policy)
    lineup = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    lineup = lineup.loc[lineup.stats_perform_match_id.eq(match_id)].copy()
    match = build_match_for_feature_generation(events, tracking, lineup, SimpleNamespace(
        action_type="all", next_action_conditions_enabled=metadata.get("next_action_conditions_enabled", True)))
    primary = metadata["intended_receiver_modes"][0]
    canonical_actions = pd.read_parquet(get_resolved_action_path(match_id, intended_receiver_mode=primary, root=source))
    match.actions = canonical_actions.copy()
    augment_match_actions_with_carries(match, segments)
    carry_actions = match.actions.loc[match.actions.spadl_type.eq("ball_carry")].copy()
    if not segments.empty and len(carry_actions) != len(segments):
        raise ValueError(f"Carry/action count mismatch for {match_id}")
    carry_ids = carry_actions.index.tolist()
    schema = metadata.get("graph_schema") or {}
    options = dict(add_v_edge_features=schema.get("add_v_edge_features", False),
                   add_relative_speed_edge_features=schema.get("add_relative_speed_edge_features", False),
                   **generation_kwargs(schema))
    match.actions = carry_actions.copy()
    generated = {}
    for post in [False, True]:
        graphs = construct_graph_features(match, extend=True, post_action=post, action_indices=carry_ids,
                                          verbose=False, **options) if carry_ids else []
        base_dir = get_post_action_graph_dir(source) if post else get_action_graph_dir(source)
        target_dir = get_post_action_graph_dir(output, use_carries=True) if post else get_action_graph_dir(output, use_carries=True)
        base_graphs = torch.load(Path(base_dir) / f"{match_id}.pt", map_location="cpu", weights_only=False)
        if len(graphs) != len(carry_ids):
            raise ValueError("Carry graph alignment failed")
        generated[Path(target_dir) / f"{match_id}.pt"] = base_graphs + graphs
    expected_rows = len(base_graphs) + len(carry_ids)
    for mode in metadata["intended_receiver_modes"]:
        base_actions = pd.read_parquet(get_resolved_action_path(match_id, intended_receiver_mode=mode, root=source))
        resolved = pd.concat([base_actions, carry_actions], sort=False)
        # Sportec IDs can be numeric in base artifacts; synthetic IDs are strings.
        resolved["original_event_id"] = resolved.original_event_id.astype("string")
        if not resolved.index.is_unique:
            raise ValueError("Duplicate canonical/carry action identities")
        generated[get_resolved_action_path(match_id, intended_receiver_mode=mode, root=output, use_carries=True)] = resolved
        for ret in metadata["return_types"]:
            from datatools import config
            threshold = metadata.get("pass_height_threshold_meters")
            if threshold is not None:
                config.PASS_HEIGHT_THRESHOLD_METERS = float(threshold)
            base_path = Path(get_action_label_dir(ret, intended_receiver_mode=mode, root=source)) / f"{match_id}.pt"
            base_labels = torch.load(base_path, map_location="cpu", weights_only=False)
            match.actions = carry_actions.copy()
            extra = match.construct_labels(resolved_actions=carry_actions.copy(), relabel_intended_receivers=False,
                                           intended_receiver_mode=mode, return_type=ret) if carry_ids else base_labels[:0]
            if len(extra) != len(carry_ids) or len(base_labels) != expected_rows - len(carry_ids):
                raise ValueError("Carry label alignment failed")
            if carry_ids and extra[:, 0].long().tolist() != carry_ids:
                raise ValueError("Carry label identity mismatch")
            if base_labels[:, 0].long().tolist() != base_actions.index.tolist():
                raise ValueError("Canonical action/label alignment failed")
            target = Path(get_action_label_dir(ret, intended_receiver_mode=mode, root=output, use_carries=True)) / f"{match_id}.pt"
            generated[target] = torch.cat([base_labels, extra])
    sidecars = output / "carry_sidecars"
    generated[sidecars / "control_events" / f"{match_id}.parquet"] = controls
    generated[sidecars / "segments" / f"{match_id}.parquet"] = segments
    generated[sidecars / "audits" / f"{match_id}.parquet"] = audit
    # Every writer replaces complete files, including hard-linked source artifacts.
    for path, value in generated.items():
        if path.suffix == ".pt":
            atomic_torch_save(value, path)
        else:
            atomic_parquet_save(value, path)
    summary = dict(match_id=match_id, definition=DEFINITION, parameters=policy_values,
        segments=len(segments), spells=len(audit), retained_spells=int(audit.retained.sum()),
        exclusions=audit.loc[~audit.retained, "reason"].value_counts().to_dict() if len(audit) else {},
        starts=audit.start_kind.value_counts().to_dict() if len(audit) else {},
        ends=audit.terminal_kind.value_counts().to_dict() if len(audit) else {},
        sources=controls.frame_source.value_counts().to_dict(),
        source_events_sha256=digest(event_path), source_controls_sha256=digest(control_path),
        tracking_stat=dict(size=tracking_path.stat().st_size, mtime_ns=tracking_path.stat().st_mtime_ns),
        elapsed_seconds=time.perf_counter() - start,
        artifacts={str(path.relative_to(output)): path.stat().st_size for path in generated})
    excluded = audit.loc[~audit.retained]
    for category in ["start_kind", "terminal_kind", "start_source", "terminal_source"]:
        summary["excluded_by_" + category] = [dict(category=key[0], reason=key[1], count=int(n))
            for key, n in excluded.groupby([category, "reason"], dropna=False).size().items()]
    atomic_write_json(output / "carry_replacement" / f"{match_id}.json", summary)
    return summary


def marker_complete(output, match_id):
    try:
        marker = json.loads((output / "carry_replacement" / f"{match_id}.json").read_text())
        return marker.get("definition") == DEFINITION and all(
            (output / path).is_file() and (output / path).stat().st_size == size
            for path, size in marker["artifacts"].items())
    except (OSError, ValueError, KeyError):
        return False


def replace_carries(args):
    source = resolve_source(args.extend_feature_run_id)
    original = load_feature_run_metadata(source.name, required=True)
    ids = match_ids(source, original)
    output = get_feature_run_root(args.run_id or generate_run_id("feature"))
    policy = CarryPolicy(**{key: getattr(args, "carry_" + key, value) for key, value in asdict(CarryPolicy()).items()})
    fingerprint = digest(source / "metadata.json")
    print(f"Carry replacement source: {source.name}\nDerived run: {output.name}", flush=True)
    settings = dict(definition=DEFINITION, validation_parameters=asdict(policy), artifact_suffix="_carries",
                    available=False, sidecar_root="carry_sidecars")
    metadata = dict(original, run_id=output.name, base_feature_run_id=source.name, status="in_progress",
                    carry_variant=settings, command=" ".join(sys.argv),
                    carry_replacement=dict(source_metadata_sha256=fingerprint, matches=ids))
    cloning_complete = False
    if output.exists():
        existing = json.loads((output / "metadata.json").read_text())
        if (existing.get("status") not in {"failed", "in_progress"}
                or existing.get("base_feature_run_id") != source.name
                or existing.get("carry_replacement", {}).get("source_metadata_sha256") != fingerprint
                or existing.get("carry_variant") != settings):
            raise ValueError("Resume requires a failed/in-progress replacement with identical source and policy")
        cloning_complete = existing.get("carry_replacement", {}).get("cloning_complete", False)
    metadata["carry_replacement"]["cloning_complete"] = cloning_complete
    write_run_metadata(output, metadata)
    try:
        if not cloning_complete:
            clone_feature_artifacts(source, output, mode=args.artifact_copy_mode, exclude=excluded_artifact, resume=True)
            metadata["carry_replacement"]["cloning_complete"] = True
            write_run_metadata(output, metadata)
        todo = [m for m in ids if not marker_complete(output, m)]
        tasks = [(str(source), str(output), original, m, asdict(policy), args.worker_thread_limit) for m in todo]
        workers = int(args.num_workers)
        if workers < 1:
            raise ValueError("Carry replacement requires a positive integer --num-workers")
        if workers == 1:
            for task in tasks:
                result = rebuild_match(task)
                print(f"Rebuilt {result['match_id']}: {result['segments']} carry segments", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(rebuild_match, tasks):
                    print(f"Rebuilt {result['match_id']}: {result['segments']} carry segments", flush=True)
        if not all(marker_complete(output, m) for m in ids):
            raise ValueError("Carry replacement did not produce every required match")
        reports = [json.loads((output / "carry_replacement" / f"{m}.json").read_text()) for m in ids]
        counts = Counter()
        for report in reports:
            counts.update(report["exclusions"])
        metadata["carry_replacement"].update(exclusions=dict(counts), segments=sum(r["segments"] for r in reports),
                                               elapsed_match_seconds=sum(r["elapsed_seconds"] for r in reports))
        metadata["carry_variant"]["available"] = True
        metadata["carry_variant"]["generated_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["status"] = "completed"
        write_run_metadata(output, metadata)
        write_latest_run("feature", output.name)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["carry_replacement"]["error"] = str(error)
        write_run_metadata(output, metadata)
        print(f"Resume with --replace-carries --extend-feature-run-id {source.name} --run-id {output.name} "
              "and the same validation parameters.", file=sys.stderr)
        raise
    print(f"Derived feature run: {output.name}")
