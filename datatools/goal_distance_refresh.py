"""Targeted goal-distance/progression refresh using saved identities and labels."""
from contextlib import ExitStack, closing
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from datatools import config, utils
from datatools.worker_runtime import map_tasks, resolve_num_workers
from datatools.refresh_staging import target_refresh_tasks
from datatools.goal_distance import GOAL_DISTANCE_MAX_RAW
from datatools.goal_distance_returns import label_goal_distance_for_return_type
from datatools.progression import DEFINITION, action_start_closeness, supported
from datatools.progression_refresh import patch_progression
from datatools.xt_refresh import (action_return_pairs, file_signature, fingerprint, refresh_inventory)
from project_config import (GOAL_DISTANCE_MATCH_DIR, atomic_write_json, get_action_label_dir,
                            get_resolved_action_dir)

PAIR_NAMES = ("scores_goal_distance", "concedes_goal_distance")
PAIR_COLUMNS = [config.LABEL_INDEX[name] for name in PAIR_NAMES]
PROGRESSION_COLUMN = config.LABEL_INDEX["scores_progression"]
GOAL_DISTANCE_DEFINITION = dict(version=1, value="1 - distance_to_opponent_goal / max_raw_distance",
    goal_center_xy=[config.FIELD_SIZE[0], config.FIELD_SIZE[1] / 2],
    max_raw_distance=GOAL_DISTANCE_MAX_RAW, eligible_action_types=list(config.XT_ACTION_TYPES))


def targeted_goal_distance(families):
    return bool(families) and set(families) <= {"goal_distance", "progression"}


def preflight_goal_distance(root, metadata, *, goal_distance_match_dir=GOAL_DISTANCE_MATCH_DIR,
                            event_dir="data/event_synced"):
    """Check complete input coverage before creating a derived run."""
    root, event_dir, annotation_dir = Path(root), Path(event_dir), Path(goal_distance_match_dir)
    inventory, _, _, _ = refresh_inventory(root, metadata)
    paths, missing_annotations, missing_events = {}, [], []
    for mid in inventory:
        pinned = root / "pass_endpoints" / f"{mid}.parquet"
        events = pinned if pinned.is_file() else event_dir / f"{mid}.csv"
        annotation = annotation_dir / f"{mid}.csv"
        if not annotation.is_file():
            missing_annotations.append(mid)
        if not events.is_file():
            missing_events.append(mid)
        paths[mid] = (events, annotation)
    if missing_annotations or missing_events:
        raise FileNotFoundError(
            f"Goal-distance refresh input coverage is incomplete for {len(inventory)} matches. "
            f"Missing annotation sidecars ({len(missing_annotations)}): {', '.join(missing_annotations) or 'none'}. "
            f"Missing event inputs ({len(missing_events)}): {', '.join(missing_events) or 'none'}. "
            "Run python scripts/generate_goal_distance.py --overwrite and verify full source-run coverage.")
    hashes = {str(path.resolve()): fingerprint(path) for pair in paths.values() for path in pair}
    return dict(paths=paths, input_sha256=hashes)


def read_goal_distance_events(event_path, annotation_path):
    event_path = Path(event_path)
    events = utils.sanitize_expected_goal(pd.read_parquet(event_path) if event_path.suffix == ".parquet"
                                         else pd.read_csv(event_path))
    annotations = pd.read_csv(annotation_path)
    for frame, description in ((events, "events"), (annotations, "annotations")):
        if ("action_id" not in frame or frame.empty or frame.action_id.isna().any()
                or frame.action_id.duplicated().any()):
            raise ValueError(f"Empty, missing or duplicate {description} action IDs: {annotation_path}")
    if "goal_distance" not in annotations:
        raise ValueError(f"Missing goal_distance annotations: {annotation_path}")
    event_indices = pd.to_numeric(pd.Series(events.index), errors="coerce").to_numpy()
    if (not events.index.is_unique or not np.isfinite(event_indices).all()
            or (event_indices % 1 != 0).any()):
        raise ValueError(f"Invalid saved event indices: {event_path}")
    if set(events.action_id) != set(annotations.action_id):
        raise ValueError(f"Mismatched annotation/event identities: {annotation_path}")
    values = pd.to_numeric(annotations.set_index("action_id").loc[events.action_id, "goal_distance"],
                           errors="coerce").to_numpy(dtype=float)
    eligible = events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
    invalid = (eligible & ~np.isfinite(values)) | (np.isfinite(values) & ((values < 0) | (values > 1)))
    if invalid.any():
        raise ValueError(f"Invalid goal_distance annotations: {annotation_path}; "
                         f"action IDs={events.loc[invalid, 'action_id'].head(8).tolist()}")
    # Preserve the saved chronology/index; sidecar return columns are default next_5, not run targets.
    events["goal_distance"] = values
    return events


def label_action_ids(labels, actions):
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or not labels.is_floating_point()
            or labels.shape[1] <= max(PAIR_COLUMNS) or labels.shape[1] > len(config.LABEL_COLUMNS)):
        raise ValueError("Invalid goal-distance label tensor shape/dtype.")
    ids = labels[:, config.LABEL_INDEX["action_index"]].cpu().numpy()
    invalid = ~np.isfinite(ids) | (ids % 1 != 0) | ~np.isin(ids, actions.index)
    if invalid.any():
        raise ValueError(f"Label action IDs do not resolve to saved actions: {ids[invalid][:8].tolist()}")
    return ids.astype(int)


def changed_tensor_rows(before, after):
    if before.shape != after.shape:
        return len(after), True
    equal = (before == after) | (before.isnan() & after.isnan())
    changed = int((~equal.all(dim=1)).sum())
    return changed, bool(changed)


def _stage_goal_distance_match(task, options):
    root, mid, artifacts, temporary_paths = task
    returns, modes, carries = options["returns"], options["modes"], options["carries"]
    repair, add_progression = options["repair"], options["add_progression"]
    inputs = options["inputs"]
    report = dict(artifacts=[], changed_rows=0, rewritten_bytes=0,
                  progression_present=add_progression, progression_return_types=[], time_return_coverage=[])
    staged, context = [], {}

    def locate(mid, path, mode=None, ret=None, carry=None):
        context.update(match_id=mid, path=str(path), receiver_mode=mode, return_type=ret, carries=carry)

    def stage(path, value, changed_rows, changed, stats=None):
        entry = dict(path=str(path.relative_to(root)), changed_rows=changed_rows,
                     rewritten=changed, **(stats or {}))
        report["artifacts"].append(entry)
        report["changed_rows"] += changed_rows
        if not changed:
            return
        temporary = temporary_paths[path]
        if path.suffix == ".pt":
            with temporary.open("wb") as stream:
                torch.save(value, stream)
        else:
            value.to_parquet(temporary)
        staged.append((path, temporary, dict(context)))
        report["rewritten_bytes"] += temporary.stat().st_size

    try:
        event_path, annotation_path = inputs["paths"][mid]
        locate(mid, annotation_path)
        events = read_goal_distance_events(event_path, annotation_path)
        event_returns = {ret: label_goal_distance_for_return_type(events, ret) for ret in returns}
        actions_by_variant, pairs, primary_progression = {}, {}, {}
        for carry in ([False, True] if carries else [False]):
            for mode in modes:
                path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                locate(mid, path, mode, returns[0], carry)
                actions = pd.read_parquet(path)
                actions_by_variant[mode, carry] = actions
                for ret, values in event_returns.items():
                    locate(mid, path, mode, ret, carry)
                    pairs[mode, carry, ret] = action_return_pairs(
                        actions, values, source_columns=PAIR_NAMES, target_columns=PAIR_NAMES, return_type=ret)
                    from project_config import parse_return_type
                    if parse_return_type(ret)[0] == "time":
                        from datatools.time_returns import timing_report
                        report["time_return_coverage"].append(dict(match_id=mid, return_type=ret, condition="base",
                            mode=mode, carries=carry, **timing_report(actions, pairs[mode, carry, ret])))
        for artifact in artifacts:
            mode, carry, ret = artifact.mode, artifact.carries, artifact.return_type
            locate(mid, artifact.path, mode, ret, carry)
            actions = actions_by_variant[mode, carry]
            labels = torch.load(artifact.path, map_location="cpu", weights_only=True)
            ids = label_action_ids(labels, actions)
            expected = torch.as_tensor(pairs[mode, carry, ret].loc[ids].to_numpy(), dtype=labels.dtype)
            old_pairs = labels[:, PAIR_COLUMNS]
            equal = (old_pairs == expected) | (old_pairs.isnan() & expected.isnan())
            if not repair and not equal.all():
                bad = (~equal.all(dim=1)).numpy()
                raise ValueError(
                    f"Saved goal-distance labels are stale or invalid; action IDs={ids[bad][:8].tolist()}. "
                    "Use --refresh-target-family goal_distance --refresh-target-family progression together.")
            updated = labels.clone()
            if repair:
                updated[:, PAIR_COLUMNS] = expected
            primary_path = get_action_label_dir(ret, mode, root=root, use_carries=carry) / f"{mid}.pt"
            primary = ret == returns[0] and artifact.path == primary_path
            needs_progression = add_progression or labels.shape[1] > PROGRESSION_COLUMN
            table_progression = primary and (add_progression or "scores_progression" in actions)
            if needs_progression or table_progression:
                starts = (action_start_closeness(actions.loc[np.unique(ids)]) if supported(ret)
                          else pd.Series(np.zeros(len(actions)), index=actions.index))
                derived, _ = patch_progression(updated, starts, ret)
                if needs_progression:
                    updated = derived
                report["progression_present"] = True
                if supported(ret) and ret not in report["progression_return_types"]:
                    report["progression_return_types"].append(ret)
                if primary:
                    if len(np.unique(ids)) != len(ids):
                        raise ValueError("Duplicate primary label action IDs.")
                    primary_progression[mode, carry] = pd.Series(
                        derived[:, PROGRESSION_COLUMN].numpy(), index=ids)
            changed_rows, changed = changed_tensor_rows(labels, updated)
            values = updated[:, PAIR_COLUMNS]
            available = torch.isfinite(values).all(dim=1)
            stats = dict(rows=len(labels), goal_distance_mean=values[available].double().mean(dim=0).tolist()
                         if available.any() else [None, None], unavailable_rows=int((~available).sum()))
            if needs_progression:
                values = updated[:, PROGRESSION_COLUMN]
                stats.update(progression_available=int(torch.isfinite(values).sum()),
                             progression_nonzero=int((torch.isfinite(values) & (values > 0)).sum()))
            stage(artifact.path, updated, changed_rows, changed, stats)
        for (mode, carry), actions in actions_by_variant.items():
            path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
            locate(mid, path, mode, returns[0], carry)
            updated = actions.copy()
            if repair:
                updated[list(PAIR_NAMES)] = pairs[mode, carry, returns[0]]
                if "goal_distance" in actions:
                    canonical = actions.index.isin(events.index)
                    updated.loc[canonical, "goal_distance"] = events.loc[
                        actions.index[canonical], "goal_distance"].to_numpy()
            if (mode, carry) in primary_progression:
                values = primary_progression[mode, carry]
                updated["scores_progression"] = values.reindex(actions.index)
            before = actions.reindex(columns=updated.columns)
            equal = (before == updated) | (before.isna() & updated.isna())
            changed_rows = int((~equal.all(axis=1)).sum())
            changed = changed_rows > 0 or not actions.columns.equals(updated.columns)
            stage(path, updated, changed_rows, changed)
    except Exception as error:
        error.add_note(f"Target refresh context: {context}")
        raise
    return dict(staged=staged, report=report)


def refresh_goal_distance(root, metadata, *, target_families=("goal_distance",), attempt,
                          preflight=None, goal_distance_match_dir=GOAL_DISTANCE_MATCH_DIR,
                          event_dir="data/event_synced", num_workers=1, worker_thread_limit=1):
    """Validate and stage every artifact before replacing any target file."""
    workers = resolve_num_workers(num_workers)
    if int(worker_thread_limit) < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    root = Path(root)
    families = set(target_families)
    if not targeted_goal_distance(families):
        raise ValueError("Targeted refresh accepts goal_distance, progression, or both.")
    repair = "goal_distance" in families
    add_progression = "progression" in families
    started = time.perf_counter()
    attempt.data["target_replacements"] = 0
    attempt.enter("validating_goal_distance_inventory")
    inventory, returns, modes, carries = refresh_inventory(root, metadata)
    if add_progression and not any(supported(ret) for ret in returns):
        raise ValueError("Progression requires a saved next_N, time_<seconds>, extended_possession[_N], or in_N return type.")
    inputs = preflight or preflight_goal_distance(root, metadata,
        goal_distance_match_dir=goal_distance_match_dir, event_dir=event_dir)
    if set(inputs["paths"]) != set(inventory):
        raise ValueError("Preflight match inventory differs from refresh inventory.")
    report = dict(source_run_id=metadata.get("run_id"), target_families=sorted(families),
                  matches=len(inventory), return_types=returns, carries=carries,
                  input_sha256=inputs["input_sha256"], definition=DEFINITION,
                  goal_distance_definition=GOAL_DISTANCE_DEFINITION,
                  graph_files_loaded=0, graph_files_written=0, tracking_files_loaded=0,
                  rewritten_files=0, rewritten_bytes=0, changed_rows=0, artifacts=[],
                  progression_present=add_progression, progression_return_types=[], time_return_coverage=[])
    signatures = {artifact.path: file_signature(artifact.path)
                  for artifacts in inventory.values() for artifact in artifacts}
    for mid in inventory:
        for carry in ([False, True] if carries else [False]):
            for mode in modes:
                path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                signatures[path] = file_signature(path)
    attempt.enter("staging_goal_distance_targets")
    staged = []
    context = {}
    try:
        with ExitStack() as stack:
            options = dict(returns=returns, modes=modes, carries=carries, repair=repair,
                           add_progression=add_progression, inputs=inputs)
            tasks = target_refresh_tasks(root, inventory, modes, carries, stack, attempt)
            with closing(map_tasks(_stage_goal_distance_match, tasks, workers, worker_thread_limit,
                                   shared_args=(options,))) as results:
                for mid, result in zip(inventory, results):
                    attempt.data.update(current_match=mid)
                    staged.extend(result["staged"])
                    for key, value in result["report"].items():
                        if key == "progression_present":
                            report[key] |= value
                        elif key == "progression_return_types":
                            report[key] = list(dict.fromkeys([*report[key], *value]))
                        elif isinstance(value, list):
                            report[key].extend(value)
                        else:
                            report[key] += value
            # A staging error above leaves every destination data file untouched.
            for path, digest in inputs["input_sha256"].items():
                if fingerprint(path) != digest:
                    raise RuntimeError(f"Goal-distance input changed during staging: {path}")
            for path, signature in signatures.items():
                if file_signature(path) != signature:
                    raise RuntimeError(f"Feature artifact changed during staging: {path}")
            report["staging_seconds"] = time.perf_counter() - started
            if staged:
                attempt.enter("replacing_goal_distance_targets", hazardous=True)
            for path, temporary, target_context in staged:
                context.update(target_context)
                attempt.data.update(current_file=str(path), current_match=context["match_id"],
                                    target_context=dict(context))
                try:
                    temporary.replace(path)
                except BaseException:
                    attempt.uncertain = not temporary.exists()
                    raise
                attempt.changed = True
                attempt.data["completed_replacements"] += 1
                attempt.data["target_replacements"] += 1
                report["rewritten_files"] += 1
        for path, digest in inputs["input_sha256"].items():
            if fingerprint(path) != digest:
                raise RuntimeError(f"Goal-distance input changed during replacement: {path}")
    except Exception as error:
        error.add_note(f"Target refresh context: {context}; target replacements="
                       f"{attempt.data['target_replacements']}; clone complete="
                       f"{attempt.data.get('clone_complete', False)}")
        raise
    report["elapsed_seconds"] = time.perf_counter() - started
    from datatools.time_returns import record_timing_coverage
    record_timing_coverage(metadata, report)
    atomic_write_json(root / "goal_distance_refresh_report.json", report)
    summary = {key: value for key, value in report.items() if key not in {"artifacts", "input_sha256"}}
    summary["report"] = "goal_distance_refresh_report.json"
    attempt.data["goal_distance_refresh"] = summary
    return report
