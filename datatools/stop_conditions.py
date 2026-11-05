"""Independent stop-condition label variants and staged target-only migration."""
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.artifact_storage import staged_path
from datatools.return_windows import bounded_returns
from datatools.xt_targets import target_settings, surface_from_metadata, transform_action_targets, unavailable_xt_returns
from project_config import parse_return_type, get_resolved_action_dir, atomic_write_json
from datatools.worker_runtime import map_shared_tasks
from datatools.epv_coverage import (ATTRIBUTE, DIRECTORY, load_annotations, unavailable, mask_targets,
                                   verified_report, label_fingerprint)

VERSION = 1
PAIR_SOURCES = {
    "scores": ("scores", "concedes"),
    "scores_xg": ("scores_xg", "concedes_xg"),
    "scores_xt": ("scores_xT", "concedes_xT"),
    "scores_goal_distance": ("scores_goal_distance", "concedes_goal_distance"),
    "scores_epv": ("scores_epv", "concedes_epv"),
}
FAMILY_SCORES = {"goal": "scores", "xg": "scores_xg", "xt": "scores_xt",
                 "goal_distance": "scores_goal_distance", "epv": "scores_epv"}


def label_path(path, root, condition="base"):
    path, root = Path(path), Path(root)
    # All artifact builders preserve their historical basename.
    relative = Path(path.parent.name) / path.name if path.suffix == ".pt" else Path(path.name)
    return root / "labels" / condition / relative


def definition(metadata, return_type, condition="base", *, legacy=False):
    saved = metadata.get("stop_conditions")
    if not saved:
        if legacy and condition in {"legacy", "base"}:
            return {"version": 0, "mode": "legacy"}
        raise ValueError("Feature run requires --refresh-stop-conditions before new outcome training.")
    if saved.get("version") != VERSION or condition not in saved.get("available", {}).get(return_type, []):
        raise ValueError(f"No {condition} labels for {return_type}; extend with --stop-condition {condition} --return_type {return_type}.")
    return {"version": VERSION, "mode": condition}


def event_returns(events, return_type, condition, surface_metadata=None):
    from datatools.match import Match
    kind, value, skip = parse_return_type(return_type)
    def calculate(part):
        match = SimpleNamespace(events=part, xt_surface_metadata=surface_metadata)
        return Match._get_events_for_return_type(match, return_type, kind, value, skip)
    result = bounded_returns(events, condition, calculate)
    if ATTRIBUTE in events.attrs:
        result.attrs[ATTRIBUTE] = events.attrs[ATTRIBUTE]
    return mask_targets(result, events)


def action_targets(actions, events, return_type, condition, settings, surface, surface_metadata,
                   xt_only=False, computed=None, target_families=None):
    from datatools.xt_refresh import action_return_pairs
    from datatools.progression import action_progression
    if computed is None:
        if xt_only:
            from datatools.xt_returns import label_xt_for_return_type
            computed = bounded_returns(events, condition, lambda part: label_xt_for_return_type(part, return_type))
        else:
            computed = event_returns(events, return_type, condition, surface_metadata)
    result = actions.copy()
    kind, max_actions, _ = parse_return_type(return_type)
    families = {"xt"} if xt_only else set(target_families or [*FAMILY_SCORES, "progression"])
    selected_scores = {score for family, score in FAMILY_SCORES.items() if family in families}
    if "progression" in families:
        selected_scores.add("scores_goal_distance")
    possession_targets = None
    if kind == "time":
        from datatools.time_returns import time_targets
        from datatools.xt_refresh import _prepare_return_mapping
        _prepare_return_mapping(actions, events)
        possession_targets = time_targets(actions, events, max_actions, condition,
            {family for family, score in FAMILY_SCORES.items() if score in selected_scores})
    if kind == "extended_possession":
        from datatools.possession_returns import extended_possession_targets
        possession_targets = extended_possession_targets(actions, events, condition,
            {family for family, score in FAMILY_SCORES.items() if score in selected_scores}, max_actions=max_actions)
    mapping = {}
    for score, source in PAIR_SOURCES.items():
        if score not in selected_scores:
            continue
        concede = score.replace("scores", "concedes", 1)
        if kind == "next_plus" and score != "scores_xt":
            result[[score, concede]] = np.nan
            continue
        if score == "scores_epv" and unavailable(events):
            result[[score, concede]] = np.nan
            continue
        if score == "scores_xt" and unavailable_xt_returns(surface_metadata, [return_type]):
            result[[score, concede]] = np.nan
            continue
        if score == "scores_xg" and kind in {"disc", "disc_max"}:
            source = ("scores_xg_disc", "concedes_xg_disc")
        if kind == "time":
            pairs = possession_targets[[score, concede]]
        else:
            pairs = action_return_pairs(actions, computed, source_columns=source, target_columns=(score, concede),
                                        _mapping=mapping)
            if possession_targets is not None:
                pairs = possession_targets[[score, concede]]
        if score == "scores_xt":
            pairs = transform_action_targets(actions, pairs, settings, surface,
                events=events, return_type=return_type, stop_condition=condition)
        result[[score, concede]] = pairs
    if "progression" in families:
        result["scores_progression"] = action_progression(result, return_type)
    if kind == "time":
        from datatools.time_returns import timing_report
        result.attrs["time_return_coverage"] = timing_report(actions, possession_targets)
    return result


def _stage_match(root, mid, artifacts, options, temporary_paths):
    """Compute one match; only the parent may replace published artifacts."""
    from datatools.xt_refresh import read_annotated_events, fingerprint, file_signature
    requests, returns = options["requests"], options["returns"]
    xt_only, families = options["xt_only"], options["families"]
    target_families, names = options["target_families"], options["names"]
    settings, surface_metadata = options["settings"], options["surface_metadata"]
    surface = surface_from_metadata(surface_metadata) if settings["representation"] == "gain" else None
    event_dir, xt_dir = options["event_dir"], options["xt_dir"]
    report = dict(changed_rows=0, ball_coordinate_files_loaded=0,
                  canonical_only_acquisition_matches=[], absent_optional_epv_matches=[], epv_coverage=[], time_return_coverage=[])
    staged, signatures, inputs = [], {}, {}
    epv_label_hashes = {}

    def stage(path, value, previous=None):
        path = Path(path)
        if path.exists():
            signatures[path] = file_signature(path)
        if isinstance(value, dict) and path.exists() and value == json.loads(path.read_text(encoding="utf-8")):
            return
        if isinstance(value, pd.DataFrame) and path.exists() and value.equals(pd.read_parquet(path)):
            return
        if isinstance(value, torch.Tensor) and previous is not None:
            if value.shape == previous.shape:
                equal = (value == previous) | (value.isnan() & previous.isnan())
                if equal.all():
                    return
                report["changed_rows"] += int((~equal.all(dim=1)).sum())
            else:
                report["changed_rows"] += len(value)
        if isinstance(value, torch.Tensor) and previous is None:
            report["changed_rows"] += len(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = temporary_paths[path]
        if isinstance(value, torch.Tensor):
            torch.save(value, temporary)
        elif isinstance(value, dict):
            temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        else:
            value.to_parquet(temporary)
        staged.append((path, temporary))

    pinned = root / "pass_endpoints" / f"{mid}.parquet"
    event_path = pinned if pinned.is_file() else Path(event_dir) / f"{mid}.csv"
    sidecar = xt_dir / f"{mid}.csv"
    for path in (event_path, sidecar):
        inputs[path] = fingerprint(path)
    events = read_annotated_events(event_path, sidecar)
    from datatools.xt_acquisitions import acquisition_context
    events, acquisition_cache, acquisition_inputs = acquisition_context(events, mid, root)
    inputs.update(acquisition_inputs.get("inputs", {}))
    signatures.update(acquisition_inputs.get("signatures", {}))
    report["ball_coordinate_files_loaded"] += acquisition_inputs.get("ball_files_loaded", 0)
    if acquisition_inputs.get("canonical_only"):
        report["canonical_only_acquisition_matches"].append(mid)
    if acquisition_cache is not None:
        stage(acquisition_cache[0], acquisition_cache[1])
    if not xt_only:
        folders = ([("goal_distance", options["goal_distance_dir"])] if {"goal_distance", "progression"} & families else [])
        if "epv" in families:
            events["epv"], coverage = load_annotations(events, mid, options["epv_dir"])
            events.attrs[ATTRIBUTE] = coverage
            if coverage["annotation_sha256"] is not None:
                inputs[Path(coverage["annotation"])] = coverage["annotation_sha256"]
            if "missing_file" in coverage["reasons"]:
                report["absent_optional_epv_matches"].append(mid)
            report["epv_coverage"].append(coverage)
        for name, folder in folders:
            annotation = Path(folder) / f"{mid}.csv"
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing stop-refresh annotation: {annotation}")
            inputs[annotation] = fingerprint(annotation)
            side = pd.read_csv(annotation)
            if (not {"action_id", name} <= set(side) or not side.action_id.is_unique
                    or set(side.action_id) != set(events.action_id)):
                raise ValueError(f"Incomplete or mismatched {name} annotation coverage: {annotation}")
            values = pd.to_numeric(side.set_index("action_id").loc[events.action_id, name], errors="coerce").to_numpy()
            if name == "goal_distance":
                eligible = events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
                if (not np.isfinite(values[eligible]).all()
                        or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any()):
                    raise ValueError(f"Invalid goal-distance annotations: {annotation}")
            events[name] = values
    cache = {}
    for artifact in artifacts:
        old_ret = artifact.return_type
        ret = old_ret.removesuffix("_spstop")
        if ret not in requests:
            continue
        if old_ret != ret and ret in returns:
            continue  # Prefer the canonical artifact when migration deduplicates legacy names.
        action_path = get_resolved_action_dir(artifact.mode, root=root, use_carries=artifact.carries) / f"{mid}.parquet"
        if action_path not in cache:
            signatures[action_path] = file_signature(action_path)
            cache[action_path] = pd.read_parquet(action_path)
        actions = cache[action_path]
        signatures[artifact.path] = file_signature(artifact.path)
        template = torch.load(artifact.path, weights_only=True, map_location="cpu")
        if (not isinstance(template, torch.Tensor) or template.ndim != 2
                or not template.is_floating_point() or template.shape[1] <= config.LABEL_INDEX["action_index"]):
            raise ValueError("Invalid stop-condition label tensor")
        ids = template[:, config.LABEL_INDEX["action_index"]].numpy()
        if not np.isfinite(ids).all() or (ids % 1).any() or not np.isin(ids, actions.index).all():
            raise ValueError("Invalid action identities in stop-condition labels")
        for condition in requests[ret]:
            key = (action_path, ret, condition)
            event_key = (ret, condition)
            if event_key not in cache:
                if xt_only:
                    from datatools.xt_returns import label_xt_for_return_type
                    cache[event_key] = bounded_returns(events, condition, lambda part: label_xt_for_return_type(part, ret))
                else:
                    cache[event_key] = event_returns(events, ret, condition, surface_metadata)
            if key not in cache:
                cache[key] = action_targets(actions, events, ret, condition, settings, surface, surface_metadata,
                                            xt_only=xt_only, computed=cache[event_key], target_families=families)
                if "time_return_coverage" in cache[key].attrs:
                    report["time_return_coverage"].append(dict(match_id=mid, return_type=ret,
                        condition=condition, mode=artifact.mode, carries=artifact.carries,
                        **cache[key].attrs["time_return_coverage"]))
            targets = cache[key]
            destination = label_path(artifact.path, root, condition)
            destination = destination.parent.with_name(destination.parent.name.replace(old_ret, ret)) / destination.name
            if (xt_only or target_families is not None) and not destination.is_file():
                raise FileNotFoundError(f"Missing advertised label variant: {destination}")
            previous = (template if destination == artifact.path else
                        torch.load(destination, weights_only=True, map_location="cpu") if destination.exists() else None)
            if previous is not None and (previous.shape != template.shape
                    or not torch.equal(previous[:, config.LABEL_INDEX["action_index"]], template[:, config.LABEL_INDEX["action_index"]])):
                raise ValueError(f"Stop variants have different action identities: {destination}")
            updated = (previous if (xt_only or target_families is not None) and previous is not None else template).clone()
            if families == {"progression"}:
                columns = [config.LABEL_INDEX[name] for name in ("scores_goal_distance", "concedes_goal_distance")]
                expected = torch.as_tensor(targets.loc[ids.astype(int), ["scores_goal_distance", "concedes_goal_distance"]].to_numpy(), dtype=updated.dtype)
                actual = updated[:, columns]
                if not ((actual == expected) | (actual.isnan() & expected.isnan())).all():
                    raise ValueError("Saved goal-distance labels are stale; refresh goal_distance and progression together")
            width = max(config.LABEL_INDEX[name] for name in names) + 1
            if width > updated.shape[1]:
                updated = torch.cat([updated, updated.new_full((len(updated), width - updated.shape[1]), float("nan"))], dim=1)
            for name in names:
                col = config.LABEL_INDEX[name]
                updated[:, col] = torch.as_tensor(targets.loc[ids.astype(int), name].to_numpy(), dtype=updated.dtype)
            stage(destination, updated, previous)
            if "epv" in families:
                epv_label_hashes[destination.relative_to(root).as_posix()] = label_fingerprint(updated)
            if condition == "base" and old_ret == returns[0] and (action_path, "staged") not in cache:
                saved = actions.copy()
                for name in names:
                    saved[name] = targets[name]
                canonical = actions.index.isin(events.index)
                raw_names = (["xT", "scores_xT", "concedes_xT"] if "xt" in families else [])
                if "goal_distance" in families:
                    raw_names.append("goal_distance")
                for name in raw_names:
                    if name in saved and name in events:
                        saved.loc[canonical, name] = events.loc[actions.index[canonical], name].to_numpy()
                stage(action_path, saved)
                cache[(action_path, "staged")] = True
    if "epv" in families:
        stage(root / DIRECTORY / f"{mid}.json", verified_report(root, coverage, requests, epv_label_hashes))
    return dict(match_id=mid, staged=staged, signatures=signatures, inputs=inputs,
                report=report, worker_pid=os.getpid())


def _match_tasks(root, inventory, options, stack, attempt):
    """Register every possible temporary in the parent, including failed workers' files."""
    for mid, artifacts in inventory.items():
        destinations = {root / "acquisition_locations" / f"{mid}.parquet"}
        if "epv" in options["families"]:
            destinations.add(root / DIRECTORY / f"{mid}.json")
        for artifact in artifacts:
            old_ret = artifact.return_type
            ret = old_ret.removesuffix("_spstop")
            if ret not in options["requests"] or (old_ret != ret and ret in options["returns"]):
                continue
            destinations.add(get_resolved_action_dir(artifact.mode, root=root,
                             use_carries=artifact.carries) / f"{mid}.parquet")
            for condition in options["requests"][ret]:
                destination = label_path(artifact.path, root, condition)
                destinations.add(destination.parent.with_name(destination.parent.name.replace(old_ret, ret)) / destination.name)
        temporary_paths = {path: stack.enter_context(staged_path(path, attempt=attempt))
                           for path in sorted(destinations)}
        yield root, mid, artifacts, options, temporary_paths


def _stage_matches(tasks, workers, thread_limit):
    match_ids = []
    def tracked():
        for task in tasks:
            match_ids.append(task[1])
            yield task
    try:
        with closing(map_shared_tasks(_stage_match, tracked(), workers, thread_limit,
                                      shared_indices=(0, 3), unpack=True)) as results:
            yield from results
    except BaseException as error:
        index = getattr(error, "feature_task_index", None)
        if index is not None and index < len(match_ids):
            error.add_note(f"Stop-condition refresh match: {match_ids[index]}")
        raise


def refresh_labels(root, metadata, requests, *, attempt=None, xt_only=False, target_families=None,
                   event_dir="data/event_synced", xt_match_dir=None, surface_metadata=None,
                   num_workers=1, worker_thread_limit=1):
    """Requests map return names to modes. Graphs and unrelated label columns are untouched."""
    from datatools.xt_refresh import refresh_inventory, fingerprint, file_signature
    from xt_versions import resolve_xt_version
    from scripts.feature_extension_diagnostics import ExtensionAttempt
    from project_config import XT_MATCH_DIR, GOAL_DISTANCE_MATCH_DIR, EPV_MATCH_DIR
    from datatools.graph_feature import resolve_num_workers
    workers = resolve_num_workers(num_workers)
    if worker_thread_limit < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    root = Path(root)
    attempt = attempt or ExtensionAttempt(root, "stop_conditions", "derived", {})
    inventory, returns, modes, carries = refresh_inventory(root, metadata)
    if not requests or set(requests) - {ret.removesuffix("_spstop") for ret in returns}:
        raise ValueError("Stop refresh requires existing, explicitly selected return types")
    if any(not modes or set(modes) - {"base", "set-pieces"} for modes in requests.values()):
        raise ValueError("Invalid stop-condition request")
    requests = {ret: list(dict.fromkeys(modes)) for ret, modes in requests.items()}
    selected = resolve_xt_version(metadata["xt_version"]) if metadata.get("xt_version") else None
    if selected and metadata.get("xt_fit_metadata_sha256") not in (None, selected.metadata_sha256):
        raise ValueError("The pinned xT surface changed after feature generation")
    xt_dir = Path(xt_match_dir) if xt_match_dir else selected.match_dir if selected else XT_MATCH_DIR
    surface_metadata = surface_metadata or (selected.metadata if selected else None)
    settings = target_settings(metadata)
    surface = surface_from_metadata(surface_metadata) if settings["representation"] == "gain" and surface_metadata else None
    if settings["representation"] == "gain" and surface is None:
        raise ValueError("Gain target refresh requires a pinned xT surface")
    families = {"xt"} if xt_only else set(target_families or [*FAMILY_SCORES, "progression"])
    if families - {*FAMILY_SCORES, "progression"}:
        raise ValueError("Unsupported target family for stop refresh")
    names = [name for family, score in FAMILY_SCORES.items() if family in families
             for name in (score, score.replace("scores", "concedes", 1))]
    if "progression" in families:
        names.append("scores_progression")
    report = dict(version=VERSION, requests=requests, xt_only=xt_only, rewritten_files=0, changed_rows=0,
                  ball_coordinate_files_loaded=0, canonical_only_acquisition_matches=[],
                  target_families=sorted(families),
                  source_run_id=metadata.get("run_id"), xt_version=metadata.get("xt_version"),
                  xt_targets=settings, return_types=list(dict.fromkeys(ret.removesuffix("_spstop") for ret in returns)),
                  intended_receiver_modes=modes, carries=carries,
                  xt_unavailable_return_types=unavailable_xt_returns(surface_metadata,
                      list(dict.fromkeys(ret.removesuffix("_spstop") for ret in returns))),
                  graph_files_loaded=0, graph_files_written=0, absent_optional_epv_matches=[], epv_coverage=[], time_return_coverage=[])
    workers = min(workers, len(inventory))
    report.update(workers=workers, total_matches=len(inventory), completed_matches=0)
    options = dict(requests=requests, returns=returns, xt_only=xt_only, families=families,
                   target_families=target_families, names=names, settings=settings,
                   surface_metadata=surface_metadata, event_dir=Path(event_dir), xt_dir=xt_dir,
                   goal_distance_dir=GOAL_DISTANCE_MATCH_DIR, epv_dir=EPV_MATCH_DIR)
    staged, signatures, inputs = [], {}, {}
    worker_pids = set()
    attempt.data.update(total_matches=len(inventory), completed_matches=0, workers=workers)
    attempt.enter("staging_stop_targets")
    if selected:
        path = selected.root / "fit_metadata.json"
        inputs[path] = fingerprint(path)
    with ExitStack() as stack:
        tasks = _match_tasks(root, inventory, options, stack, attempt)
        with closing(_stage_matches(tasks, workers, worker_thread_limit)) as results:
            for result in results:
                staged.extend(result["staged"])
                signatures.update(result["signatures"])
                inputs.update(result["inputs"])
                for name, value in result["report"].items():
                    if isinstance(value, list):
                        report[name].extend(value)
                    else:
                        report[name] += value
                report["completed_matches"] += 1
                worker_pids.add(result["worker_pid"])
                attempt.data.update(current_match=result["match_id"],
                                    completed_matches=report["completed_matches"], worker_pids=sorted(worker_pids))
                print(f"Staged stop-condition targets: {report['completed_matches']}/{len(inventory)} "
                      f"({workers} workers; {result['match_id']})", flush=True)
        for name in ("canonical_only_acquisition_matches", "absent_optional_epv_matches"):
            report[name].sort()
        report["epv_coverage"].sort(key=lambda item: item["match_id"])
        for path, signature in signatures.items():
            if file_signature(path) != signature:
                raise RuntimeError(f"Artifact changed during stop refresh: {path}")
        for path, digest in inputs.items():
            if fingerprint(path) != digest:
                raise RuntimeError(f"Input changed during stop refresh: {path}")
        report["input_sha256"] = {str(path.resolve()): digest for path, digest in inputs.items()}
        if staged:
            attempt.enter("replacing_stop_targets", hazardous=True)
        for path, temporary in staged:
            attempt.data["current_file"] = str(path)
            try:
                temporary.replace(path)
            except BaseException:
                attempt.uncertain = not temporary.exists()
                raise
            attempt.changed = True
            attempt.data["completed_replacements"] += 1
            report["rewritten_files"] += 1
    updated = copy.deepcopy(metadata.get("stop_conditions") or {"version": VERSION, "available": {}})
    for ret, modes in requests.items():
        updated["available"][ret] = sorted(set(updated["available"].get(ret, [])) | set(modes))
    metadata["stop_conditions"] = updated
    if "epv" in families:
        from datatools.epv_coverage import VERSION as EPV_VERSION, POLICY
        complete = sum(item["complete"] for item in report["epv_coverage"])
        metadata[ATTRIBUTE] = dict(version=EPV_VERSION, policy=POLICY, directory=DIRECTORY,
                                   complete_matches=complete, unavailable_matches=len(inventory) - complete)
        print(f"EPV coverage: {complete}/{len(inventory)} complete matches; "
              f"{len(inventory) - complete} unavailable (other targets retained).", flush=True)
    metadata["label_layout_version"] = 1
    metadata["return_types"] = list(dict.fromkeys(r.removesuffix("_spstop") for r in returns))
    metadata["xt_unavailable_return_types"] = list(dict.fromkeys(r.removesuffix("_spstop") for r in metadata.get("xt_unavailable_return_types", [])))
    metadata["return_type"] = metadata["return_types"][0] if len(metadata["return_types"]) == 1 else None
    from datatools.time_returns import record_timing_coverage
    record_timing_coverage(metadata, report)
    for path, digest in inputs.items():
        if fingerprint(path) != digest:
            raise RuntimeError(f"Input changed during stop refresh replacement: {path}")
    atomic_write_json(root / ("xt_refresh_report.json" if xt_only else "stop_conditions_refresh_report.json"), report)
    attempt.data["xt_refresh" if xt_only else "stop_conditions_refresh"] = report
    return report
