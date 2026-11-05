"""Event boundaries and turnover windows shared by generation and refresh."""
import numpy as np
import pandas as pd

from datatools import config

ACQUISITION_TYPES = frozenset({
    "interception", "ball_recovery", "recovery", "tackle", "keeper_claim",
    "keeper_pick_up", "keeper_save", "keeper_sweeper", "shot_block", "block",
    "keeper_punch", "punch", "clearance",
})


def boundary_segments(events, condition="base"):
    """Half-open segments: goals belong to the preceding segment, restarts to the next."""
    if condition not in {"base", "set-pieces"}:
        raise ValueError(f"Unknown stop condition: {condition}")
    types = events.spadl_type.to_numpy()
    periods = events.period_id.to_numpy()
    goals = (pd.to_numeric(events.get("expected_goal", pd.Series(0., index=events.index)),
                           errors="coerce").fillna(0).gt(0)
             & events.success.fillna(False).astype(bool)).to_numpy()
    restarts = set(config.SET_PIECE) if condition == "set-pieces" else {"goalkick"}
    starts = [0]
    for i in range(1, len(events)):
        if periods[i] != periods[i - 1] or goals[i - 1] or types[i] in restarts:
            starts.append(i)
    return list(zip(starts, starts[1:] + [len(events)])) if len(events) else []


def bounded_returns(events, condition, labeler):
    """Apply an existing return definition without allowing it to cross a boundary."""
    if condition is None:
        return labeler(events)
    parts = [labeler(events.iloc[start:end].copy()) for start, end in boundary_segments(events, condition)]
    return pd.concat(parts).reindex(events.index) if parts else labeler(events)


class PossessionWindows:
    """Current team sequence plus the next opponent sequence, within stop boundaries.

    Eligibility depends only on action type and team, never on target availability.
    Bounds index the eligible timeline. Per-run suffix aggregates keep long
    possession windows linear to prepare and constant-time to reduce.
    """

    def __init__(self, events, condition=None):
        self.teams = events.object_id.astype(str).str[:4].to_numpy()
        self.eligible = np.flatnonzero(events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
                                      & np.isin(self.teams, ["home", "away"]))
        segment = np.zeros(len(events), dtype=int)
        self.ends = np.empty(len(events), dtype=int)
        for number, (start, end) in enumerate(boundary_segments(events, condition or "base")):
            segment[start:end] = number
            self.ends[start:end] = end
        positions = self.eligible
        self.run_ends = np.empty(len(positions), dtype=int)
        if len(positions):
            changes = ((self.teams[positions[1:]] != self.teams[positions[:-1]])
                       | (segment[positions[1:]] != segment[positions[:-1]]))
            starts = np.r_[0, np.flatnonzero(changes) + 1]
            ends = np.r_[starts[1:], len(positions)]
            self.run_ends[:] = np.repeat(ends, ends - starts)

    def bounds(self, anchors, teams):
        """Future-only bounds, seeded by the labeled action's team (also for carries)."""
        anchors = np.asarray(anchors, dtype=int)
        teams = np.asarray(teams)
        starts = np.searchsorted(self.eligible, anchors + 1)
        limits = np.searchsorted(self.eligible, self.ends[anchors])
        ends = starts.copy()
        valid = starts < limits
        ends[valid] = self.run_ends[starts[valid]]
        same = np.zeros(len(anchors), dtype=bool)
        same[valid] = self.teams[self.eligible[starts[valid]]] == teams[valid]
        second = same & (ends < limits)
        ends[second] = self.run_ends[ends[second]]
        return starts, np.minimum(ends, limits)

    def select(self, anchor, team):
        starts, ends = self.bounds([anchor], [team])
        return self.eligible[starts[0]:ends[0]]

    def aggregate(self, values, anchors, teams, *, aggregation="max", include_anchor=False):
        """Team-perspective maxima or independent-event probabilities in each window."""
        if aggregation not in {"max", "probability"}:
            raise ValueError(f"Unknown possession aggregation: {aggregation}")
        anchors = np.asarray(anchors, dtype=int)
        teams = np.asarray(teams)
        values = np.asarray(values, dtype=float)
        starts, ends = self.bounds(anchors, teams)
        suffix = np.zeros(len(self.eligible))
        selected = values[self.eligible]
        for i in range(len(selected) - 1, -1, -1):
            value = selected[i] if np.isfinite(selected[i]) else (-np.inf if aggregation == "max" else 0.)
            if i + 1 < self.run_ends[i]:
                value = (max(value, suffix[i + 1]) if aggregation == "max"
                         else 1 - (1 - value) * (1 - suffix[i + 1]))
            suffix[i] = value
        pairs = np.zeros((len(anchors), 2))
        rows = np.flatnonzero(starts < ends)
        for _ in range(2):
            sides = (self.teams[self.eligible[starts[rows]]] != teams[rows]).astype(int)
            reduced = suffix[starts[rows]]
            pairs[rows, sides] = np.where(np.isfinite(reduced), reduced, 0.)
            starts[rows] = self.run_ends[starts[rows]]
            rows = rows[starts[rows] < ends[rows]]
        if include_anchor:
            eligible_anchor = np.isin(anchors, self.eligible)
            rows = np.flatnonzero(eligible_anchor & np.isfinite(values[anchors]))
            sides = (self.teams[anchors[rows]] != teams[rows]).astype(int)
            current = values[anchors[rows]]
            previous = pairs[rows, sides]
            pairs[rows, sides] = (np.maximum(previous, current) if aggregation == "max"
                                  else 1 - (1 - previous) * (1 - current))
        return pairs


class TurnoverWindows:
    def __init__(self, events, condition=None):
        from datatools.xt_acquisitions import expand_acquisitions
        events = expand_acquisitions(events)
        self.events = events
        self.teams = events.object_id.astype(str).str[:4].to_numpy()
        self.types = events.spadl_type.to_numpy()
        self.eligible = np.flatnonzero(events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
                                      & np.isfinite(pd.to_numeric(events.xT, errors="coerce"))
                                      & np.isin(self.teams, ["home", "away"]))
        # Legacy maxima remain unbounded; acquisition searches still cannot cross periods.
        self.ends = np.full(len(events), len(events), dtype=int)
        self.starts = np.zeros(len(events), dtype=int)
        segments = boundary_segments(events, condition) if condition else boundary_segments(events, "base")
        for start, end in segments:
            self.starts[start:end] = start
            if condition:
                self.ends[start:end] = end
        self.x = pd.to_numeric(events.start_x, errors="coerce").to_numpy()
        self.y = pd.to_numeric(events.start_y, errors="coerce").to_numpy()

    def acquisition(self, anchor, first, previous):
        lower = max(anchor, previous, int(self.starts[first]) - 1)
        for pos in range(lower + 1, first):
            if (self.teams[pos] == self.teams[first] and self.types[pos] in ACQUISITION_TYPES
                    and np.isfinite(self.x[pos]) and np.isfinite(self.y[pos])
                    and self.x[pos] != -config.FIELD_SIZE[0] and self.y[pos] != -config.FIELD_SIZE[1]):
                return pos
        return first

    def select(self, anchor, team, n, extended=False, skip_first=False):
        start = np.searchsorted(self.eligible, anchor + 1)
        positions = self.eligible[start:start + n]
        positions = positions[positions < self.ends[anchor]]
        opponents = positions[self.teams[positions] != team]
        acquisition = None
        if len(opponents):
            first = int(opponents[0])
            own = positions[(positions < first) & (self.teams[positions] == team)]
            acquisition = self.acquisition(anchor, first, int(own[-1]) if len(own) else anchor)
            if extended:
                after = np.searchsorted(self.eligible, acquisition + 1)
                positions = self.eligible[start:after + n]
                positions = positions[positions < self.ends[anchor]]
        if skip_first and len(positions) and self.types[positions[0]] != "shot":
            positions = positions[1:]
        return positions, acquisition


def _acquisition_baseline(windows, acquisition, surface):
    x, y = windows.x[acquisition], windows.y[acquisition]
    if (not np.isfinite(x) or not np.isfinite(y)
            or x == -config.FIELD_SIZE[0] or y == -config.FIELD_SIZE[1]):
        raise ValueError("Opponent acquisition baseline requires valid starting coordinates")
    if windows.teams[acquisition] == "away":
        x, y = config.FIELD_SIZE[0] - x, config.FIELD_SIZE[1] - y
    return float(surface.predict_xy(x, y))


def _possession_xt_windows(actions, windows, surface, condition):
    possessions = PossessionWindows(windows.events, condition)
    indices = pd.Series(actions.index, index=actions.index)
    anchors = windows.events.index.get_indexer(actions.get("return_event_index", indices).fillna(indices))
    if (anchors < 0).any():
        raise ValueError("Unknown action return anchor")
    teams = actions.object_id.astype(str).str[:4].to_numpy()
    values = pd.to_numeric(windows.events.xT, errors="coerce").to_numpy()
    pairs = possessions.aggregate(values, anchors, teams)
    baselines = np.zeros(len(actions))
    if surface is not None:
        starts, ends = possessions.bounds(anchors, teams)
        eligible = possessions.eligible
        counts = np.r_[0, np.cumsum(np.isfinite(values[eligible]))]
        for row in np.flatnonzero(starts < ends):
            first = starts[row]
            if windows.teams[eligible[first]] == teams[row]:
                first = possessions.run_ends[first]
            if first >= ends[row] or counts[ends[row]] == counts[first]:
                continue
            previous = int(eligible[first - 1]) if first > starts[row] else int(anchors[row])
            acquisition = windows.acquisition(int(anchors[row]), int(eligible[first]), previous)
            baselines[row] = _acquisition_baseline(windows, acquisition, surface)
    return pairs, baselines


def action_xt_windows(actions, events, return_type, surface=None, condition=None):
    """Action-perspective maxima and acquisition baselines, including carry anchors."""
    from project_config import parse_return_type
    kind, n, skip = parse_return_type(return_type)
    possession = kind == "extended_possession"
    windows = TurnoverWindows(events, (condition or "base") if possession else condition)
    if possession:
        return _possession_xt_windows(actions, windows, surface, condition)
    values = pd.to_numeric(windows.events.xT, errors="coerce").to_numpy()
    pairs = np.zeros((len(actions), 2))
    baselines = np.zeros(len(actions))
    anchors = actions.get("return_event_index", pd.Series(actions.index, index=actions.index))
    anchors = anchors.fillna(pd.Series(actions.index, index=actions.index))
    for row, (index, action) in enumerate(actions.iterrows()):
        anchor = windows.events.index.get_indexer([anchors.loc[index]])[0]
        if anchor < 0:
            raise ValueError("Unknown action return anchor")
        team = str(action.object_id)[:4]
        positions, acquisition = windows.select(anchor, team, int(n), kind == "next_plus", skip)
        for side, same in enumerate((True, False)):
            selected = positions[(windows.teams[positions] == team) == same]
            if len(selected):
                pairs[row, side] = values[selected].max()
                if side == 1 and surface is not None and acquisition is not None:
                    baselines[row] = _acquisition_baseline(windows, acquisition, surface)
    return pairs, baselines
