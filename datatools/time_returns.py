"""Elapsed-time returns on the synchronized action clock, shared by all writers."""
import numpy as np
import pandas as pd

from datatools import config
from datatools.return_windows import boundary_segments


DEFINITION = dict(version=1, clock="frame_id / 25", fps=25, cutoff="inclusive_action_start",
                  initial_action="goal_and_xg_only", missing_timing="nan",
                  ordering="canonical_forward_with_nonnegative_elapsed_time")
FAMILIES = ("goal", "xg", "xt", "goal_distance", "epv")


def action_times(actions):
    frames = pd.to_numeric(actions.get("frame_id", pd.Series(np.nan, index=actions.index)), errors="coerce")
    frames = frames.to_numpy(dtype=float, na_value=np.nan)
    return np.where(np.isfinite(frames) & (frames >= 0) & (frames == np.floor(frames)), frames / 25., np.nan)


class TimeWindows:
    """Keep canonical boundaries and identities even when synchronized times decrease.

    An untimed eligible candidate makes preceding windows in its stop segment
    uncertain. Never silently drop it or substitute a different clock.
    """

    def __init__(self, events, condition=None):
        self.events = events
        self.times = action_times(events)
        self.teams = events.object_id.astype(str).str[:4].to_numpy()
        self.eligible = (events.spadl_type.isin(config.XT_ACTION_TYPES).to_numpy()
                         & np.isin(self.teams, ["home", "away"]))
        self.starts = np.zeros(len(events), dtype=int)
        self.ends = np.zeros(len(events), dtype=int)
        self.periods = events.period_id.to_numpy()
        for start, end in boundary_segments(events, condition or "base"):
            self.starts[start:end] = start
            self.ends[start:end] = end

    def segment_anchor(self, anchor, start_time, synthetic):
        if not synthetic:
            return anchor
        # Carries can map to a terminal event after a restart. Choose the play
        # containing their start instead of borrowing the terminal event's play.
        before = np.flatnonzero((self.periods == self.periods[anchor]) & (self.times <= start_time))
        if not len(before):
            return int(np.flatnonzero(self.periods == self.periods[anchor])[0])
        previous = int(before[np.argsort(self.times[before], kind="stable")[-1]])
        if (previous + 1 == self.ends[previous] and previous + 1 < len(self.events)
                and self.periods[previous + 1] == self.periods[previous]
                and self.times[previous] < start_time):
            # A goal ends the preceding play immediately; a subsequent restart
            # starts the next play only when its own time is reached.
            xg = pd.to_numeric(pd.Series([self.events.iloc[previous].get("expected_goal")]), errors="coerce").iloc[0]
            if xg > 0 and bool(self.events.iloc[previous].success):
                return previous + 1
        return previous

    def select(self, anchor, start_time, seconds, *, synthetic=False, include_anchor=False):
        if not np.isfinite(start_time):
            return None
        anchor = self.segment_anchor(anchor, start_time, synthetic)
        lower = self.starts[anchor] if synthetic else anchor + int(not include_anchor)
        candidates = np.arange(lower, self.ends[anchor])
        # Goal/xG use event rows, other families use pass/cross/shot candidates.
        if not include_anchor:
            candidates = candidates[self.eligible[candidates]]
        if (~np.isfinite(self.times[candidates])).any():
            return None
        elapsed = self.times[candidates] - start_time
        candidates = candidates[(elapsed >= 0) & (elapsed <= seconds + 1e-9)]
        return candidates[np.argsort(self.times[candidates], kind="stable")]


def time_targets(actions, events, seconds, condition=None, families=None):
    """Raw returns in the labelled action's perspective, using its actual start."""
    families = tuple(families if families is not None else FAMILIES)
    windows = TimeWindows(events, condition)
    indices = pd.Series(actions.index, index=actions.index)
    anchors = events.index.get_indexer(actions.get("return_event_index", indices).fillna(indices))
    if (anchors < 0).any():
        raise ValueError("Unknown action return anchor")
    teams = actions.object_id.astype(str).str[:4].to_numpy()
    if not np.isin(teams, ["home", "away"]).all():
        raise ValueError("Unknown action team perspective")
    times = action_times(actions)
    synthetic = ~actions.index.isin(events.index)
    result = pd.DataFrame(index=actions.index)
    for family in families:
        value_col = "xT" if family == "xt" else "expected_goal" if family in {"goal", "xg"} else family
        values = pd.to_numeric(events.get(value_col, pd.Series(np.nan, index=events.index)), errors="coerce").to_numpy(dtype=float)
        if family == "goal":
            values = (np.isfinite(values) & events.success.fillna(False).astype(bool).to_numpy()).astype(float)
        pairs = np.zeros((len(actions), 2))
        for row, anchor in enumerate(anchors):
            selected = windows.select(anchor, times[row], seconds, synthetic=synthetic[row],
                                      include_anchor=family in {"goal", "xg"})
            if selected is None:
                pairs[row] = np.nan
                continue
            for side, same in enumerate((True, False)):
                candidates = values[selected[(windows.teams[selected] == teams[row]) == same]]
                candidates = candidates[np.isfinite(candidates)]
                if len(candidates):
                    pairs[row, side] = (1 - np.prod(1 - candidates) if family == "xg" else candidates.max())
        suffix = "" if family == "goal" else "_" + family
        result[["scores" + suffix, "concedes" + suffix]] = pairs
    return result


def label_time_events(events, seconds, families=None):
    result = events.copy()
    known = events.object_id.astype(str).str[:4].isin(["home", "away"])
    targets = time_targets(events.loc[known], events, seconds, families=families)
    targets = targets.reindex(events.index, fill_value=0.)
    result[targets.rename(columns={"scores_xt": "scores_xT", "concedes_xt": "concedes_xT"}).columns] = targets.to_numpy()
    return result


def timing_report(actions, targets):
    """Counts are per resolved action, never per target cell."""
    return dict(actions=len(actions), missing_timing_actions=int((~np.isfinite(action_times(actions))).sum()),
                missing_return_actions=int(targets.isna().any(axis=1).sum()))


def record_timing_coverage(metadata, report):
    entries = report.get("time_return_coverage", [])
    if not entries:
        return
    metadata["time_returns"] = dict(DEFINITION)
    totals = {}
    for entry in entries:
        key = (entry["return_type"], entry["condition"], entry["mode"], entry["carries"])
        counts = totals.setdefault(key, dict(actions=0, missing_timing_actions=0, missing_return_actions=0))
        for name in counts:
            counts[name] += entry[name]
    report["time_return_totals"] = []
    for (ret, condition, mode, carries), counts in sorted(totals.items()):
        report["time_return_totals"].append(dict(return_type=ret, condition=condition, mode=mode, carries=carries, **counts))
        print(f"Time returns {ret}/{condition}/{mode}/carries={carries}: "
              f"{counts['missing_timing_actions']}/{counts['actions']} actions missing timing; "
              f"{counts['missing_return_actions']} actions have NaN returns.", flush=True)


def action_time_xt_windows(actions, events, seconds, surface=None, condition=None):
    """Time maxima with the same opponent-acquisition baseline as next_N gains."""
    from datatools.return_windows import TurnoverWindows, _acquisition_baseline
    turnover = TurnoverWindows(events, condition or "base")
    events = turnover.events
    windows = TimeWindows(events, condition)
    indices = pd.Series(actions.index, index=actions.index)
    anchors = events.index.get_indexer(actions.get("return_event_index", indices).fillna(indices))
    if (anchors < 0).any():
        raise ValueError("Unknown action return anchor")
    times = action_times(actions)
    teams = actions.object_id.astype(str).str[:4].to_numpy()
    synthetic = ~actions.index.isin(events.index)
    values = pd.to_numeric(events.xT, errors="coerce").to_numpy(dtype=float)
    pairs, baselines = np.zeros((len(actions), 2)), np.zeros(len(actions))
    for row, anchor in enumerate(anchors):
        selected = windows.select(anchor, times[row], seconds, synthetic=synthetic[row])
        if selected is None:
            pairs[row], baselines[row] = np.nan, np.nan
            continue
        selected = selected[np.isfinite(values[selected])]
        for side, same in enumerate((True, False)):
            candidates = selected[(windows.teams[selected] == teams[row]) == same]
            if len(candidates):
                pairs[row, side] = values[candidates].max()
                if side == 1 and surface is not None:
                    first = int(candidates[0])
                    prior = selected[(selected < first) & (windows.teams[selected] == teams[row])]
                    lower = int(anchor)
                    if synthetic[row]:
                        segment_anchor = windows.segment_anchor(anchor, times[row], True)
                        before = np.arange(windows.starts[segment_anchor], first)
                        before = before[windows.times[before] < times[row]]
                        lower = int(before[-1]) if len(before) else int(windows.starts[segment_anchor]) - 1
                    acquisition = turnover.acquisition(lower, first, int(prior[-1]) if len(prior) else lower)
                    baselines[row] = _acquisition_baseline(turnover, acquisition, surface)
    return pairs, baselines
