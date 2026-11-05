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

    def select(self, anchor, team, n, extended=False, skip_first=False):
        start = np.searchsorted(self.eligible, anchor + 1)
        positions = self.eligible[start:start + n]
        positions = positions[positions < self.ends[anchor]]
        opponents = positions[self.teams[positions] != team]
        acquisition = None
        if len(opponents):
            first = int(opponents[0])
            own = positions[(positions < first) & (self.teams[positions] == team)]
            lower = max(anchor, int(own[-1]) if len(own) else anchor, int(self.starts[first]) - 1)
            acquisition = first
            for pos in range(lower + 1, first):
                if (self.teams[pos] == self.teams[first] and self.types[pos] in ACQUISITION_TYPES
                        and np.isfinite(self.x[pos]) and np.isfinite(self.y[pos])
                        and self.x[pos] != -config.FIELD_SIZE[0] and self.y[pos] != -config.FIELD_SIZE[1]):
                    acquisition = pos
                    break
            if extended:
                after = np.searchsorted(self.eligible, acquisition + 1)
                positions = self.eligible[start:after + n]
                positions = positions[positions < self.ends[anchor]]
        if skip_first and len(positions) and self.types[positions[0]] != "shot":
            positions = positions[1:]
        return positions, acquisition


def action_xt_windows(actions, events, return_type, surface=None, condition=None):
    """Action-perspective maxima and acquisition baselines, including carry anchors."""
    from project_config import parse_return_type
    kind, n, skip = parse_return_type(return_type)
    windows = TurnoverWindows(events, condition)
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
                    x, y = windows.x[acquisition], windows.y[acquisition]
                    if (not np.isfinite(x) or not np.isfinite(y)
                            or x == -config.FIELD_SIZE[0] or y == -config.FIELD_SIZE[1]):
                        raise ValueError("Opponent acquisition baseline requires valid starting coordinates")
                    if windows.teams[acquisition] == "away":
                        x, y = config.FIELD_SIZE[0] - x, config.FIELD_SIZE[1] - y
                    baselines[row] = float(surface.predict_xy(x, y))
    return pairs, baselines
