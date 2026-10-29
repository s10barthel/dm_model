"""Add progression to saved labels without regenerating graphs or returns."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.artifact_storage import staged_path
from datatools.progression import DEFINITION, action_start_closeness, relative_progression, supported
from datatools.xt_refresh import refresh_inventory, file_signature
from project_config import atomic_write_json, get_resolved_action_dir


def patch_progression(labels, starts, return_type):
    column = config.LABEL_INDEX["scores_progression"]
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 2
            or not labels.is_floating_point() or labels.shape[1] not in {column, column + 1}):
        raise ValueError("Progression extension requires the current pre-progression label schema.")
    ids = labels[:, config.LABEL_INDEX["action_index"]].cpu().numpy()
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or not np.isin(ids, starts.index).all():
        raise ValueError("Progression label action IDs do not resolve to saved actions.")
    values = np.full(len(labels), np.nan)
    if supported(return_type):
        values = relative_progression(starts.loc[ids.astype(int)].to_numpy(),
                                      labels[:, config.LABEL_INDEX["scores_goal_distance"]].cpu().numpy())
    values = torch.as_tensor(values, dtype=labels.dtype, device=labels.device)
    if labels.shape[1] == column:
        return torch.cat([labels, values[:, None]], dim=1), True
    updated = labels.clone()
    updated[:, column] = values
    equal = (labels[:, column] == values) | (labels[:, column].isnan() & values.isnan())
    return updated, not bool(equal.all())


def refresh_progression(root, metadata, *, attempt):
    root = Path(root)
    attempt.enter("validating_progression_inventory")
    inventory, returns, modes, carries = refresh_inventory(root, metadata)
    supported_returns = [ret for ret in returns if supported(ret)]
    if not supported_returns:
        raise ValueError("Progression requires a saved next_N or in_N return type.")
    report = dict(definition=DEFINITION, return_types=supported_returns, matches=len(inventory),
                  graph_files_loaded=0, rewritten_files=0, rewritten_bytes=0)
    signatures, staged = {}, []
    attempt.enter("staging_progression_targets")
    with ExitStack() as stack:
        def stage(path, value):
            temporary = stack.enter_context(staged_path(path, attempt=attempt))
            if path.suffix == ".pt":
                with temporary.open("wb") as stream:
                    torch.save(value, stream)
            else:
                value.to_parquet(temporary)
            staged.append((path, temporary))
            report["rewritten_bytes"] += temporary.stat().st_size

        for mid, artifacts in inventory.items():
            starts = {}
            for carry in ([False, True] if carries else [False]):
                for mode in modes:
                    path = get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                    signatures[path] = file_signature(path)
                    actions = pd.read_parquet(path)
                    if not actions.index.is_unique:
                        raise ValueError(f"Duplicate resolved action indices: {path}")
                    starts[mode, carry] = action_start_closeness(actions)
                    values = np.full(len(actions), np.nan)
                    if supported(returns[0]):
                        values = relative_progression(starts[mode, carry], actions.scores_goal_distance)
                    previous = actions.get("scores_progression")
                    if previous is None or not np.array_equal(previous.to_numpy(), values, equal_nan=True):
                        actions["scores_progression"] = values
                        stage(path, actions)
            for artifact in artifacts:
                signatures[artifact.path] = file_signature(artifact.path)
                labels = torch.load(artifact.path, map_location="cpu", weights_only=True)
                updated, changed = patch_progression(labels, starts[artifact.mode, artifact.carries], artifact.return_type)
                if changed:
                    stage(artifact.path, updated)
        for path, signature in signatures.items():
            if file_signature(path) != signature:
                raise RuntimeError(f"Feature artifact changed during progression staging: {path}")
        if staged:
            attempt.enter("replacing_progression_targets", hazardous=True)
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
    atomic_write_json(root / "progression_refresh_report.json", report)
    attempt.data["progression_refresh"] = report
    return report
