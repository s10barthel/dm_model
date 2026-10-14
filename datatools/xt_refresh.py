"""Refresh xT targets from saved action identities, without loading graph/tracking data."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from datatools import config, utils
from datatools.artifact_storage import staged_path
from datatools.xt_returns import label_xt_for_return_type
from project_config import (XT_MATCH_DIR, atomic_write_json, get_action_label_dir,
                            get_intent_train_label_dir, get_augmented_label_dir,
                            get_resolved_action_dir)
from scripts.feature_extension_diagnostics import ExtensionAttempt

TARGET_COLUMNS = [config.LABEL_INDEX[name] for name in ("scores_xt", "concedes_xt")]


@dataclass(frozen=True)
class LabelArtifact:
    path: Path
    mode: str
    return_type: str
    carries: bool = False


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _ids(directory, suffix=".pt"):
    return {p.stem for p in directory.glob(f"*{suffix}")}


def refresh_inventory(root, metadata):
    """Validate complete directory membership using filenames, never graph payloads."""
    root = Path(root)
    returns = list(metadata.get("return_types") or [metadata.get("return_type")])
    modes = list(metadata.get("intended_receiver_modes") or ["original", "angle_only"])
    if not returns or any(r is None for r in returns) or not modes:
        raise ValueError("Refresh requires recorded return types and receiver modes.")
    if len(set(returns)) != len(returns) or len(set(modes)) != len(modes):
        raise ValueError("Duplicate return types or receiver modes in metadata.")
    match_ids = _ids(root / "action_graphs")
    if not match_ids:
        raise ValueError("No canonical action graphs found.")
    expected_count = (metadata.get("split_manifest") or {}).get("universe_count")
    if expected_count is not None and len(match_ids) != expected_count:
        raise ValueError(f"Canonical graph count {len(match_ids)} differs from split universe {expected_count}.")
    carries = bool((metadata.get("carry_variant") or {}).get("available"))
    has_carries = any(p.is_dir() and p.name.endswith("_carries") for p in root.iterdir())
    if carries != has_carries:
        raise ValueError("Carry metadata and artifact directories disagree.")
    inventory = {mid: [] for mid in sorted(match_ids)}
    known = set()

    def require(directory, expected, suffix=".pt"):
        known.add(directory.name)
        actual = _ids(directory, suffix)
        if actual != expected:
            raise ValueError(f"Artifact inventory mismatch in {directory}: "
                             f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")

    def labels(directory, expected, mode, return_type, carry=False):
        require(directory, expected)
        for mid in sorted(expected):
            inventory[mid].append(LabelArtifact(directory / f"{mid}.pt", mode, return_type, carry))

    require(root / "post_action_graphs", match_ids)
    intent_ids = _ids(root / "action_graphs_intent_train")
    success_ids = _ids(root / "action_graphs_success_intent")
    if not intent_ids <= match_ids or not success_ids <= match_ids:
        raise ValueError("Auxiliary graph match IDs are outside canonical matches.")
    for carry in ([False, True] if carries else [False]):
        if carry:
            require(root / "action_graphs_carries", match_ids)
            require(root / "post_action_graphs_carries", match_ids)
        for mode in modes:
            require(get_resolved_action_dir(mode, root=root, use_carries=carry), match_ids, ".parquet")
            for return_type in returns:
                labels(get_action_label_dir(return_type, mode, root=root, use_carries=carry),
                       match_ids, mode, return_type, carry)
    for mode in modes:
        for return_type in returns:
            directory = get_intent_train_label_dir(return_type, mode, root=root)
            if intent_ids or directory.exists():
                labels(directory, intent_ids, mode, return_type)
        directory = get_augmented_label_dir(mode, root=root)
        graph_dir = root / directory.name.replace("labels", "graphs")
        if graph_dir.exists() or directory.exists():
            augmented_ids = _ids(graph_dir)
            if not augmented_ids <= match_ids:
                raise ValueError(f"Unexpected augmented match IDs in {graph_dir}")
            labels(directory, augmented_ids, mode, returns[0])
    if success_ids or (root / "success_intent_labels").exists():
        # Successful observed passes retain the canonical action team/identity.
        labels(root / "success_intent_labels", success_ids, modes[0], returns[0])
    for directory in root.iterdir():
        if directory.is_dir() and directory.name.startswith(
                ("action_labels", "augmented_labels", "success_intent_labels", "resolved_actions")):
            if directory.name not in known:
                raise ValueError(f"Unrecorded label/action variant: {directory}")
    return inventory, returns, modes, carries


def read_annotated_events(event_path, sidecar_path):
    events = utils.sanitize_expected_goal(pd.read_csv(event_path))
    sidecar = pd.read_csv(sidecar_path)
    required = {"action_id", "xT", "scores_xT", "concedes_xT"}
    if not required <= set(sidecar):
        raise ValueError(f"Missing annotation columns in {sidecar_path}: {required - set(sidecar)}")
    for frame, description in ((events, "events"), (sidecar, "annotations")):
        if frame.empty or frame.action_id.isna().any() or frame.action_id.duplicated().any():
            raise ValueError(f"Empty or non-unique {description} action IDs: {sidecar_path}")
    # Generator exports one row per synchronized event, including non-xT events.
    if set(events.action_id) != set(sidecar.action_id):
        raise ValueError(f"Incomplete or mismatched annotation action coverage: {sidecar_path}")
    sidecar = sidecar.set_index("action_id").loc[events.action_id]
    for column in ("xT", "scores_xT", "concedes_xT"):
        values = pd.to_numeric(sidecar[column], errors="coerce").to_numpy(dtype=float)
        expected = events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy() if column == "xT" else np.ones(len(events), bool)
        if not np.isfinite(values[expected]).all() or ((values[expected] < 0) | (values[expected] > 1)).any():
            raise ValueError(f"Invalid {column} annotations in {sidecar_path}")
        events[column] = values
    return events


def action_return_pairs(actions, events):
    if not actions.index.is_unique:
        raise ValueError("Resolved action indices are not unique.")
    indices = pd.to_numeric(pd.Series(actions.index, index=actions.index), errors="coerce")
    raw_anchors = actions.get("return_event_index", indices)
    anchors = pd.to_numeric(raw_anchors, errors="coerce")
    if (raw_anchors.notna() & anchors.isna()).any():
        raise ValueError("Invalid return-event anchors in resolved actions.")
    anchors = anchors.fillna(indices)
    if (anchors.isna().any() or (anchors % 1 != 0).any()
            or not anchors.isin(events.index).all()):
        raise ValueError("Invalid return-event anchors in resolved actions.")
    if indices.isna().any() or (indices % 1 != 0).any():
        raise ValueError("Resolved action indices must be integers.")
    canonical = indices.isin(events.index)
    for column in ("action_id", "original_event_id", "object_id", "period_id"):
        if column not in actions or column not in events:
            raise ValueError(f"Missing action identity column: {column}")
        actual = actions.loc[canonical, column].reset_index(drop=True)
        expected = events.loc[indices[canonical].astype(int), column].reset_index(drop=True)
        # Carry concatenation can convert numeric event IDs to strings.
        if not (actual.astype(str) == expected.astype(str)).all():
            numeric_equal = pd.to_numeric(actual, errors="coerce") == pd.to_numeric(expected, errors="coerce")
            if not ((actual.astype(str) == expected.astype(str)) | numeric_equal).all():
                raise ValueError(f"Saved action identity differs from synchronized events: {column}")
    if (~canonical).any():
        synthetic = actions.loc[~canonical]
        if ("return_event_index" not in synthetic or synthetic.return_event_index.isna().any()
                or not synthetic.original_event_id.astype(str).str.startswith("carry:").all()):
            raise ValueError("Unknown synthetic action or missing carry anchor.")
    selected = events.loc[anchors.astype(int).to_numpy()]
    pairs = selected[["scores_xT", "concedes_xT"]].to_numpy(copy=True)
    teams = actions.object_id.astype(str).str[:4].to_numpy()
    if not np.isin(teams, ["home", "away"]).all():
        raise ValueError("Unknown action team perspective.")
    swap = selected.object_id.astype(str).str[:4].to_numpy() != teams
    pairs[swap] = pairs[swap, ::-1]
    if not np.isfinite(pairs).all():
        raise ValueError("Non-finite computed xT returns.")
    return pd.DataFrame(pairs, index=actions.index, columns=["scores_xt", "concedes_xt"])


def patch_labels(labels, pairs):
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2 or not labels.is_floating_point()
            or labels.shape[1] <= max(TARGET_COLUMNS)):
        raise ValueError("Invalid label tensor shape.")
    action_ids = labels[:, config.LABEL_INDEX["action_index"]].cpu().numpy()
    if not np.isfinite(action_ids).all() or (action_ids % 1 != 0).any() or not np.isin(action_ids, pairs.index).all():
        raise ValueError("Label action IDs do not resolve to saved actions.")
    values = torch.as_tensor(pairs.loc[action_ids.astype(int)].to_numpy(), dtype=labels.dtype, device=labels.device)
    changed = (labels[:, TARGET_COLUMNS] != values).any(dim=1)
    updated = labels.clone()
    updated[:, TARGET_COLUMNS] = values
    return updated, int(changed.sum()), values


def refresh_xt(root, metadata, *, xt_match_dir=XT_MATCH_DIR, event_dir="data/event_synced", attempt=None):
    """Validate and stage the entire refresh before replacing any data artifact."""
    root, xt_match_dir, event_dir = Path(root), Path(xt_match_dir), Path(event_dir)
    attempt = attempt or ExtensionAttempt(root, "xt_refresh", "derived", {})
    started = time.perf_counter()
    attempt.enter("validating_xt_inventory")
    inventory, returns, modes, carries = refresh_inventory(root, metadata)
    inputs = {}
    for mid in inventory:
        for path in (event_dir / f"{mid}.csv", xt_match_dir / f"{mid}.csv"):
            if not path.is_file():
                raise FileNotFoundError(f"Required refresh input is missing: {path}")
            inputs[str(path.resolve())] = fingerprint(path)
    report = dict(source_run_id=metadata.get("run_id"), matches=len(inventory), return_types=returns,
                  carries=carries, input_sha256=inputs, artifacts=[], changed_rows=0,
                  rewritten_files=0, rewritten_bytes=0, graph_files_loaded=0, graph_files_written=0)
    attempt.enter("staging_xt_targets")
    staged = []
    signatures = {artifact.path: file_signature(artifact.path)
                  for artifacts in inventory.values() for artifact in artifacts}
    for mid in inventory:
        for carry in ([False, True] if carries else [False]):
            for mode in modes:
                path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                signatures[path] = file_signature(path)
    with ExitStack() as stack:
        def stage(path, value, changed_rows, stats=None):
            entry = dict(path=str(path.relative_to(root)), changed_rows=changed_rows)
            if stats is not None:
                entry.update(stats)
            report["artifacts"].append(entry)
            report["changed_rows"] += changed_rows
            if not changed_rows:
                return
            temporary = stack.enter_context(staged_path(path, attempt=attempt))
            if path.suffix == ".pt":
                with temporary.open("wb") as stream:
                    torch.save(value, stream)
            else:
                value.to_parquet(temporary)
            staged.append((path, temporary))
            report["rewritten_bytes"] += temporary.stat().st_size

        for mid, artifacts in inventory.items():
            events = read_annotated_events(event_dir / f"{mid}.csv", xt_match_dir / f"{mid}.csv")
            event_returns = {ret: label_xt_for_return_type(events, ret) for ret in returns}
            pairs = {}
            for carry in ([False, True] if carries else [False]):
                for mode in modes:
                    path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                    attempt.data["current_file"] = str(path)
                    actions = pd.read_parquet(path)
                    pairs.update({(mode, carry, ret): action_return_pairs(actions, values)
                                  for ret, values in event_returns.items()})
                    updated = actions.copy()
                    updated[["scores_xt", "concedes_xt"]] = pairs[(mode, carry, returns[0])]
                    # Refresh the saved raw annotation columns for canonical rows;
                    # synthetic carries have no independent xT event annotation.
                    canonical = actions.index.isin(events.index)
                    for column in ("xT", "scores_xT", "concedes_xT"):
                        if column in actions:
                            updated.loc[canonical, column] = events.loc[actions.index[canonical], column].to_numpy()
                    target_names = [c for c in ("xT", "scores_xT", "concedes_xT", "scores_xt", "concedes_xt") if c in updated]
                    before = actions.reindex(columns=target_names)
                    after = updated[target_names]
                    equal = (before == after) | (before.isna() & after.isna())
                    stage(path, updated, int((~equal.all(axis=1)).sum()))
            for artifact in artifacts:
                attempt.data["current_file"] = str(artifact.path)
                labels = torch.load(artifact.path, map_location="cpu", weights_only=True)
                updated, changed, values = patch_labels(labels, pairs[(artifact.mode, artifact.carries, artifact.return_type)])
                stats = dict(rows=len(labels), target_mean=values.double().mean(dim=0).tolist() if len(labels) else [0, 0],
                             nonzero_rows=int((values != 0).any(dim=1).sum()))
                stage(artifact.path, updated, changed, stats)
        # No destination artifact has been replaced yet. Validate all inputs again.
        for path, digest in inputs.items():
            if fingerprint(path) != digest:
                raise RuntimeError(f"Refresh input changed during staging: {path}")
        for path, signature in signatures.items():
            if file_signature(path) != signature:
                raise RuntimeError(f"Feature artifact changed during staging: {path}")
        report["staging_seconds"] = time.perf_counter() - started
        attempt.data["xt_refresh"] = {k: v for k, v in report.items() if k not in {"artifacts", "input_sha256"}}
        if staged:
            attempt.enter("replacing_xt_targets", hazardous=True)
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
    report["elapsed_seconds"] = time.perf_counter() - started
    for path, digest in inputs.items():
        if fingerprint(path) != digest:
            raise RuntimeError(f"Refresh input changed during replacement: {path}")
    atomic_write_json(root / "xt_refresh_report.json", report)
    attempt.data["xt_refresh"] = {k: v for k, v in report.items() if k not in {"artifacts", "input_sha256"}}
    attempt.data["xt_refresh"]["report"] = "xt_refresh_report.json"
    return report
