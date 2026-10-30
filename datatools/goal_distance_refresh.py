"""Targeted goal-distance/progression refresh using saved identities and labels."""
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from datatools import config, utils
from datatools.artifact_storage import staged_path
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


def refresh_goal_distance(root, metadata, *, target_families=("goal_distance",), attempt,
                          preflight=None, goal_distance_match_dir=GOAL_DISTANCE_MATCH_DIR,
                          event_dir="data/event_synced"):
    """Validate and stage every artifact before replacing any target file."""
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
        raise ValueError("Progression requires a saved next_N or in_N return type.")
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
                  progression_present=add_progression, progression_return_types=[])
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
            def locate(mid, path, mode=None, ret=None, carry=None):
                context.update(match_id=mid, path=str(path), receiver_mode=mode, return_type=ret, carries=carry)
                attempt.data.update(current_match=mid, current_file=str(path), target_context=dict(context))

            def stage(path, value, changed_rows, changed, stats=None):
                entry = dict(path=str(path.relative_to(root)), changed_rows=changed_rows,
                             rewritten=changed, **(stats or {}))
                report["artifacts"].append(entry)
                report["changed_rows"] += changed_rows
                if not changed:
                    return
                temporary = stack.enter_context(staged_path(path, attempt=attempt))
                if path.suffix == ".pt":
                    with temporary.open("wb") as stream:
                        torch.save(value, stream)
                else:
                    value.to_parquet(temporary)
                staged.append((path, temporary, dict(context)))
                report["rewritten_bytes"] += temporary.stat().st_size

            for mid, artifacts in inventory.items():
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
                                actions, values, source_columns=PAIR_NAMES, target_columns=PAIR_NAMES)
                for artifact in artifacts:
                    mode, carry, ret = artifact.mode, artifact.carries, artifact.return_type
                    locate(mid, artifact.path, mode, ret, carry)
                    actions = actions_by_variant[mode, carry]
                    labels = torch.load(artifact.path, map_location="cpu", weights_only=True)
                    ids = label_action_ids(labels, actions)
                    expected = torch.as_tensor(pairs[mode, carry, ret].loc[ids].to_numpy(), dtype=labels.dtype)
                    old_pairs = labels[:, PAIR_COLUMNS]
                    if not repair and not torch.equal(old_pairs, expected):
                        bad = (~(old_pairs == expected).all(dim=1)).numpy()
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
                    stats = dict(rows=len(labels), goal_distance_mean=updated[:, PAIR_COLUMNS].double().mean(dim=0).tolist()
                                 if len(labels) else [0, 0])
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
    atomic_write_json(root / "goal_distance_refresh_report.json", report)
    summary = {key: value for key, value in report.items() if key not in {"artifacts", "input_sha256"}}
    summary["report"] = "goal_distance_refresh_report.json"
    attempt.data["goal_distance_refresh"] = summary
    return report
