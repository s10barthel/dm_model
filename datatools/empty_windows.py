"""Versioned availability of future-window targets, independent of target values."""
from collections import Counter

import numpy as np
import pandas as pd

from datatools import config
from datatools.return_windows import boundary_segments
from project_config import parse_return_type

KEY = "empty_window_policy"
DEFINITION = dict(version=1, policy="exclude_empty_future_windows", families=["xt", "goal_distance", "epv", "progression"])
REASON_COLUMN = "empty_window_reason"
REASONS = {0: "available", 1: "set_piece", 2: "goal", 3: "period_end", 4: "match_end",
           5: "time_limit", 6: "skip1"}
KINDS = {"next", "next_plus", "time", "extended_possession", "disc", "disc_max", "disc_poly_max"}
TARGETS = tuple(name for family in ("xt", "goal_distance", "epv")
                for name in ("scores_" + family, "concedes_" + family)) + ("scores_progression",)


def enabled(metadata):
    saved = (metadata or {}).get(KEY)
    if saved is not None and saved != DEFINITION:
        raise ValueError("Unsupported empty-window policy definition")
    return saved is not None


class WindowAvailability:
    """Prepare one canonical eligible timeline per match/stop condition."""

    def __init__(self, events, condition="base"):
        self.events, self.condition = events, condition or "base"
        self.teams = events.object_id.astype(str).str[:4].to_numpy()
        self.eligible = np.flatnonzero(events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
                                      & np.isin(self.teams, ["home", "away"]))
        self.ends = np.empty(len(events), dtype=int)
        self.boundary_reasons = np.empty(len(events), dtype=int)
        goals = (pd.to_numeric(events.get("expected_goal", pd.Series(0., index=events.index)), errors="coerce")
                 .fillna(0).gt(0) & events.success.fillna(False).astype(bool)).to_numpy()
        for start, end in boundary_segments(events, self.condition):
            if end == len(events):
                reason = 4
            elif events.period_id.iloc[end] != events.period_id.iloc[end - 1]:
                reason = 3
            elif goals[end - 1]:
                reason = 2
            else:
                reason = 1
            self.ends[start:end], self.boundary_reasons[start:end] = end, reason
        self.time_windows = None
        self.turnover_windows = None
        self.action_anchors = {}

    def reasons(self, actions, return_type):
        kind, value, skip = parse_return_type(return_type)
        result = np.zeros(len(actions), dtype=int)
        if kind not in KINDS or not len(actions):
            return result
        from datatools.xt_refresh import _prepare_return_mapping
        key = id(actions)
        if key not in self.action_anchors:
            selected = _prepare_return_mapping(actions, self.events)
            self.action_anchors[key] = (actions, self.events.index.get_indexer(selected.index))
        _, anchors = self.action_anchors[key]
        if (anchors < 0).any():
            raise ValueError("Unknown action return anchor")
        starts = np.searchsorted(self.eligible, anchors + 1)
        limits = np.searchsorted(self.eligible, self.ends[anchors])
        count = limits - starts
        if kind == "time":
            from datatools.time_returns import TimeWindows, action_times
            if self.time_windows is None:
                self.time_windows = TimeWindows(self.events, self.condition)
            times = action_times(actions)
            synthetic = ~actions.index.isin(self.events.index)
            for row, anchor in enumerate(anchors):
                selected = self.time_windows.select(anchor, times[row], value, synthetic=synthetic[row])
                # Unknown timing is a distinct existing exclusion, never an empty window.
                if selected is not None and not len(selected):
                    segment = self.time_windows.segment_anchor(anchor, times[row], synthetic[row])
                    lower = self.time_windows.starts[segment] if synthetic[row] else anchor + 1
                    candidates = np.arange(lower, self.ends[segment])
                    candidates = candidates[self.time_windows.eligible[candidates]]
                    future = candidates[self.time_windows.times[candidates] >= times[row]]
                    result[row] = 5 if len(future) else self.boundary_reasons[segment]
            return result
        result[count == 0] = self.boundary_reasons[anchors[count == 0]]
        if kind == "next_plus" and skip:
            from datatools.return_windows import TurnoverWindows
            if self.turnover_windows is None:
                # Membership must never depend on xT annotation availability.
                self.turnover_windows = TurnoverWindows(self.events.assign(xT=0.), self.condition)
            window = self.turnover_windows
            indices = pd.Series(actions.index, index=actions.index)
            mapped = actions.get("return_event_index", indices).fillna(indices)
            for row in np.flatnonzero(count):
                anchor = window.events.index.get_indexer([mapped.iloc[row]])[0]
                selected, _ = window.select(anchor, str(actions.object_id.iloc[row])[:4], int(value), True, True)
                if not len(selected):
                    result[row] = 6
        elif skip:
            selected_count = np.minimum(count, int(value)) if kind == "next" else count
            rows = np.flatnonzero(selected_count == 1)
            first = self.eligible[starts[rows]]
            result[rows[self.events.spadl_type.to_numpy()[first] != "shot"]] = 6
        return result


def mask_targets(targets, actions, events, return_type, condition="base", *, availability=None):
    availability = availability or WindowAvailability(events, condition)
    reasons = availability.reasons(actions, return_type)
    targets[REASON_COLUMN] = reasons
    columns = [name for name in TARGETS if name in targets]
    targets.loc[reasons != 0, columns] = np.nan
    return dict(actions=len(actions), empty_windows=int((reasons != 0).sum()),
                reasons={REASONS[int(key)]: value for key, value in sorted(Counter(reasons[reasons != 0]).items())})


def validate_labels(labels):
    """Require an explicit, valid reason; never interpret arbitrary NaNs as exclusions."""
    import torch
    column = config.LABEL_INDEX[REASON_COLUMN]
    if labels.ndim != 2 or labels.shape[1] <= column:
        raise ValueError("Missing empty-window availability labels; refresh the feature run")
    reasons = labels[:, column]
    if not torch.isfinite(reasons).all() or not torch.isin(reasons, reasons.new_tensor(list(REASONS))).all():
        raise ValueError("Invalid empty-window availability labels")
    return reasons != 0
