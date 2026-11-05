"""Mask saved targets in bounded match workers, without recomputing returns."""
from contextlib import ExitStack, closing
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datatools import config, utils
from datatools.artifact_storage import staged_path
from datatools.empty_windows import DEFINITION, KEY, REASON_COLUMN, TARGETS, WindowAvailability, mask_targets, enabled
from datatools.worker_runtime import map_shared_tasks, resolve_num_workers
from datatools.xt_refresh import refresh_inventory, fingerprint, file_signature
from project_config import atomic_write_json, get_resolved_action_dir, parse_return_type


def _stage_match(root, mid, artifacts, options, temporary_paths):
    staged, signatures, inputs = [], {}, {}
    report = dict(match_id=mid, coverage=[], changed_rows=0, rewritten_files=0, worker_pid=os.getpid())

    def stage(path, value):
        temporary = temporary_paths[path]
        if isinstance(value, torch.Tensor):
            torch.save(value, temporary)
        elif isinstance(value, dict):
            temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        else:
            value.to_parquet(temporary)
        staged.append((path, temporary))
        report["rewritten_files"] += 1

    pinned = root / "pass_endpoints" / f"{mid}.parquet"
    event_path = pinned if pinned.is_file() else options["event_dir"] / f"{mid}.csv"
    inputs[event_path] = fingerprint(event_path)
    events = utils.sanitize_expected_goal(pd.read_parquet(event_path) if event_path.suffix == ".parquet"
                                         else pd.read_csv(event_path))
    if any(parse_return_type(ret)[0] == "next_plus" and parse_return_type(ret)[2] for ret in options["returns"]):
        cache_path = root / "acquisition_locations" / f"{mid}.parquet"
        if cache_path.is_file():
            from datatools.xt_acquisitions import acquisition_context
            events, _, provenance = acquisition_context(events, mid, root)
            inputs.update(provenance.get("inputs", {}))
    availability, actions_cache, target_cache = {}, {}, {}
    from datatools.epv_coverage import DIRECTORY, label_fingerprint
    coverage_path = root / DIRECTORY / f"{mid}.json"
    coverage = None
    if coverage_path.exists():
        signatures[coverage_path] = file_signature(coverage_path)
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    previous_coverage = copy.deepcopy(coverage)
    for artifact in artifacts:
        action_path = get_resolved_action_dir(artifact.mode, root=root, use_carries=artifact.carries) / f"{mid}.parquet"
        if action_path not in actions_cache:
            signatures[action_path] = file_signature(action_path)
            actions_cache[action_path] = pd.read_parquet(action_path)
        actions = actions_cache[action_path]
        for condition in options["requests"][artifact.return_type]:
            from datatools.stop_conditions import label_path
            path = label_path(artifact.path, root, condition)
            if not path.is_file():
                raise FileNotFoundError(f"Missing advertised label variant: {path}")
            signatures[path] = file_signature(path)
            labels = torch.load(path, weights_only=True, map_location="cpu")
            if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or not labels.is_floating_point()
                    or not config.LABEL_INDEX["scores_progression"] < labels.shape[1] <= len(config.LABEL_COLUMNS)):
                raise ValueError(f"Invalid label tensor: {path}")
            ids = labels[:, config.LABEL_INDEX["action_index"]].numpy()
            if not np.isfinite(ids).all() or (ids % 1).any() or not np.isin(ids, actions.index).all():
                raise ValueError(f"Invalid action identities in labels: {path}")
            key = (action_path, artifact.return_type, condition)
            if key not in target_cache:
                if condition not in availability:
                    availability[condition] = WindowAvailability(events, condition)
                reasons = availability[condition].reasons(actions, artifact.return_type)
                target_cache[key] = pd.Series(reasons, index=actions.index)
                from collections import Counter
                from datatools.empty_windows import REASONS
                report["coverage"].append(dict(return_type=artifact.return_type, condition=condition,
                    mode=artifact.mode, carries=artifact.carries, actions=len(actions),
                    empty_windows=int((reasons != 0).sum()),
                    reasons={REASONS[int(k)]: v for k, v in sorted(Counter(reasons[reasons != 0]).items())}))
            reasons = target_cache[key].loc[ids.astype(int)].to_numpy()
            width = config.LABEL_INDEX[REASON_COLUMN] + 1
            if labels.shape[1] < width:
                updated = torch.cat([labels, labels.new_zeros((len(labels), width - labels.shape[1]))], dim=1)
            else:
                updated = labels.clone()
            updated[:, config.LABEL_INDEX[REASON_COLUMN]] = torch.as_tensor(reasons, dtype=updated.dtype)
            rows = torch.as_tensor(reasons != 0)
            for name in TARGETS:
                updated[rows, config.LABEL_INDEX[name]] = float("nan")
            same = labels.shape == updated.shape and ((labels == updated) | (labels.isnan() & updated.isnan())).all()
            if not same:
                report["changed_rows"] += (int(reasons.astype(bool).sum()) if labels.shape != updated.shape else
                    int((~((labels == updated) | (labels.isnan() & updated.isnan())).all(dim=1)).sum()))
                stage(path, updated)
            if coverage is not None:
                relative = path.relative_to(root).as_posix()
                # Keep existing certifications only; never certify unavailable annotations.
                verified = coverage.get("verified_labels", {})
                if relative in verified:
                    if verified[relative] != label_fingerprint(labels):
                        raise ValueError(f"EPV labels changed since certification: {path}")
                    verified[relative] = label_fingerprint(updated)
        if (action_path, "primary") not in target_cache:
            primary = options["returns"][0]
            if artifact.return_type == primary:
                saved = actions.copy()
                mask_targets(saved, actions, events, primary, "base", availability=availability["base"])
                if not saved.equals(actions):
                    stage(action_path, saved)
                target_cache[action_path, "primary"] = True
    if coverage is not None and coverage != previous_coverage:
        stage(coverage_path, coverage)
    return dict(staged=staged, signatures=signatures, inputs=inputs, report=report)


def refresh_empty_windows(root, metadata, *, attempt, event_dir="data/event_synced", num_workers=1,
                          worker_thread_limit=1):
    """All advertised variants in one pass; publish only after every worker succeeds."""
    from datatools.stop_conditions import VERSION, label_path
    started = time.perf_counter()
    root = Path(root)
    enabled(metadata)  # Reject unknown definitions instead of silently downgrading a future policy.
    stops = metadata.get("stop_conditions") or {}
    if stops.get("version") != VERSION or metadata.get("label_layout_version") != 1:
        raise ValueError("Run --refresh-stop-conditions before --refresh-empty-window-policy")
    inventory, returns, modes, carries = refresh_inventory(root, metadata)
    requests = stops.get("available", {})
    if set(returns) != set(requests) or any("base" not in requests[r] or set(requests[r]) - {"base", "set-pieces"} for r in returns):
        raise ValueError("Incomplete or invalid recorded stop-condition variants")
    workers = min(resolve_num_workers(num_workers), len(inventory))
    if worker_thread_limit < 1:
        raise ValueError("--worker-thread-limit must be positive")
    report = dict(definition=DEFINITION, workers=workers, total_matches=len(inventory), completed_matches=0,
                  rewritten_files=0, changed_rows=0, coverage=[], worker_pids=[], graph_files_loaded=0,
                  graph_files_written=0, tracking_files_loaded=0, target_annotation_files_loaded=0)
    options = dict(requests=requests, returns=returns, event_dir=Path(event_dir))
    staged, signatures, inputs = [], {}, {}
    with ExitStack() as stack:
        def tasks():
            from datatools.epv_coverage import DIRECTORY
            for mid, artifacts in inventory.items():
                paths = {root / DIRECTORY / f"{mid}.json"}
                for artifact in artifacts:
                    paths.add(get_resolved_action_dir(artifact.mode, root=root, use_carries=artifact.carries) / f"{mid}.parquet")
                    paths.update(label_path(artifact.path, root, condition) for condition in requests[artifact.return_type])
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                temporary = {p: stack.enter_context(staged_path(p, attempt=attempt)) for p in sorted(paths)}
                yield root, mid, artifacts, options, temporary
        attempt.enter("staging_empty_window_policy")
        with closing(map_shared_tasks(_stage_match, tasks(), workers, worker_thread_limit,
                                      shared_indices=(0, 3), unpack=True)) as results:
            for result in results:
                staged.extend(result["staged"])
                signatures.update(result["signatures"])
                inputs.update(result["inputs"])
                entry = result["report"]
                report["coverage"].extend(dict(match_id=entry["match_id"], **row) for row in entry["coverage"])
                for name in ("changed_rows", "rewritten_files"):
                    report[name] += entry[name]
                report["worker_pids"] = sorted(set(report["worker_pids"]) | {entry["worker_pid"]})
                report["completed_matches"] += 1
                attempt.data.update(completed_matches=report["completed_matches"], total_matches=len(inventory),
                                    workers=workers, worker_pids=report["worker_pids"])
                print(f"Staged empty-window policy: {report['completed_matches']}/{len(inventory)} "
                      f"({workers} workers; {entry['match_id']})", flush=True)
        for path, signature in signatures.items():
            if file_signature(path) != signature:
                raise RuntimeError(f"Artifact changed during empty-window refresh: {path}")
        for path, digest in inputs.items():
            if fingerprint(path) != digest:
                raise RuntimeError(f"Input changed during empty-window refresh: {path}")
        report["staging_seconds"] = time.perf_counter() - started
        if staged:
            attempt.enter("replacing_empty_window_targets", hazardous=True)
        for path, temporary in staged:
            attempt.data["current_file"] = str(path)
            try:
                temporary.replace(path)
            except BaseException:
                attempt.uncertain = not temporary.exists()
                raise
            attempt.changed = True
            attempt.data["completed_replacements"] += 1
    for path, digest in inputs.items():
        if fingerprint(path) != digest:
            raise RuntimeError(f"Input changed during empty-window refresh replacement: {path}")
    metadata[KEY] = copy.deepcopy(DEFINITION)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["input_sha256"] = {str(p.resolve()): digest for p, digest in inputs.items()}
    atomic_write_json(root / "empty_window_refresh_report.json", report)
    attempt.data["empty_window_refresh"] = report
    return report
