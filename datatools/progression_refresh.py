"""Add progression to saved labels without regenerating graphs or returns."""

import numpy as np
import torch

from datatools import config
from datatools.progression import relative_progression, supported


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


def refresh_progression(root, metadata, *, attempt, **kwargs):
    """Validate saved goal-distance targets and derive progression from label tensors."""
    from datatools.goal_distance_refresh import refresh_goal_distance
    return refresh_goal_distance(root, metadata, target_families=("progression",), attempt=attempt, **kwargs)
