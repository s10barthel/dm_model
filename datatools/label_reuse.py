"""Short-lived reuse of label rows across returns for one resolved variant."""
from contextlib import contextmanager

import torch

from datatools import config


TARGET_NAMES = tuple(name for name in config.LABEL_COLUMNS
                     if name.startswith(("scores", "concedes"))
                     and name not in {"scores_goal_next10", "concedes_goal_next10"})
TARGET_COLUMNS = tuple(config.LABEL_INDEX[name] for name in TARGET_NAMES)


@contextmanager
def reuse_label_rows(match, *, enabled=True):
    if not enabled:
        yield {}
        return
    previous = getattr(match, "_label_row_cache", None)
    cache = {}
    match._label_row_cache = cache
    try:
        yield cache
    finally:
        if previous is None:
            del match._label_row_cache
        else:
            match._label_row_cache = previous


def _context(match):
    return (id(match.tracking), match.fps, match.include_goals, match.action_type,
            config.PASS_HEIGHT_THRESHOLD_METERS)


def _common_actions(match):
    return match.actions.drop(columns=list(TARGET_NAMES), errors="ignore")


def reused_labels(match):
    cache = getattr(match, "_label_row_cache", None)
    if not cache or cache.get("context") != _context(match):
        return None
    if not cache["actions"].equals(_common_actions(match)):
        return None
    values = match.actions.loc[cache["indices"], list(TARGET_NAMES)]
    if any(getattr(dtype, "kind", None) not in {"b", "i", "u", "f"}
           or getattr(dtype, "itemsize", 8) > 8 for dtype in values.dtypes):
        return None
    try:
        targets = torch.tensor(values.to_numpy(dtype=float), dtype=cache["labels"].dtype)
    except (TypeError, ValueError):
        # Retain the original row construction and its errors for unusual dtypes.
        return None
    if cache.get("validate_only"):
        return cache["labels"]
    labels = cache["labels"].clone()
    labels[:, TARGET_COLUMNS] = targets
    return labels


def remember_label_rows(match, labels, indices):
    cache = getattr(match, "_label_row_cache", None)
    if cache is not None and labels.ndim == 2:
        cache.update(context=_context(match), actions=_common_actions(match).copy(),
                     labels=labels, indices=indices)
