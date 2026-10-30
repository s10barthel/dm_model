"""Relative advancement using the existing goal-distance return selection."""
import numpy as np
import pandas as pd

from datatools import config
from datatools.goal_distance import GOAL_DISTANCE_MAX_RAW, goal_distance_from_xy
from project_config import parse_return_type

DEFINITION = {"version": 1, "scoring": "relative_goal_distance_reduction",
              "conceding": "concedes_goal_distance", "return_kinds": ["next", "in"]}


def supported(return_type):
    return parse_return_type(return_type)[0] in {"next", "in"}


def relative_progression(start_closeness, future_closeness):
    start, future = np.broadcast_arrays(np.asarray(start_closeness, dtype=float),
                                       np.asarray(future_closeness, dtype=float))
    if not np.isfinite(start).all() or not np.isfinite(future).all():
        raise ValueError("Progression requires finite starting positions and goal-distance returns.")
    if ((start < 0) | (start > 1) | (future < 0) | (future > 1)).any():
        raise ValueError("Goal-distance closeness must be in [0, 1].")
    remaining = 1.0 - start
    result = np.zeros_like(start)
    np.divide(future - start, remaining, out=result,
              where=remaining > 1e-6 / GOAL_DISTANCE_MAX_RAW)
    return np.clip(result, 0.0, 1.0)


def action_start_closeness(actions):
    """Resolved actions use home-oriented coordinates; rotate away actions."""
    teams = actions.object_id.astype(str).str[:4]
    if not teams.isin(["home", "away"]).all():
        raise ValueError("Unknown progression action team perspective.")
    x = pd.to_numeric(actions.start_x, errors="coerce").copy()
    y = pd.to_numeric(actions.start_y, errors="coerce").copy()
    invalid = (~np.isfinite(x) | ~np.isfinite(y)
               | (x == -config.FIELD_SIZE[0]) | (y == -config.FIELD_SIZE[1]))
    if invalid.any():
        raise ValueError("Progression requires valid action starting coordinates; "
                         f"action indices={actions.index[invalid].tolist()[:8]}.")
    away = teams.eq("away")
    x.loc[away] = config.FIELD_SIZE[0] - x.loc[away]
    y.loc[away] = config.FIELD_SIZE[1] - y.loc[away]
    values = goal_distance_from_xy(x, y)
    if not np.isfinite(values).all():
        raise ValueError("Progression requires valid action starting coordinates.")
    return pd.Series(values, index=actions.index)


def action_progression(actions, return_type):
    if not supported(return_type):
        return np.full(len(actions), np.nan)
    return relative_progression(action_start_closeness(actions), actions.scores_goal_distance)
