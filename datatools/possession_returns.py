"""Extended-possession targets shared by generation and label refresh."""
import numpy as np
import pandas as pd

from datatools.return_windows import PossessionWindows


def extended_possession_targets(actions, events, condition=None, families=None, *, max_actions=None):
    """Raw action-perspective returns; the anchor establishes time, the action its team."""
    families = set(families if families is not None else ("goal", "xg", "xt", "goal_distance", "epv"))
    windows = PossessionWindows(events, condition, max_actions)
    indices = pd.Series(actions.index, index=actions.index)
    anchors = actions.get("return_event_index", indices).fillna(indices)
    positions = events.index.get_indexer(anchors)
    if (positions < 0).any():
        raise ValueError("Unknown action return anchor")
    teams = actions.object_id.astype(str).str[:4].to_numpy()
    if not np.isin(teams, ["home", "away"]).all():
        raise ValueError("Unknown action team perspective")
    result = pd.DataFrame(index=actions.index)
    for family in families:
        value_col = "xT" if family == "xt" else "expected_goal" if family in {"goal", "xg"} else family
        values = pd.to_numeric(events.get(value_col, pd.Series(np.nan, index=events.index)), errors="coerce")
        if family == "goal":
            values = (values.notna() & events.success.fillna(False).astype(bool)).astype(float)
        suffix = "" if family == "goal" else "_" + family
        result[["scores" + suffix, "concedes" + suffix]] = windows.aggregate(
            values.to_numpy(), positions, teams,
            aggregation="probability" if family == "xg" else "max",
            include_anchor=family in {"goal", "xg"})
    return result


def label_extended_possession_events(events, families=None, *, max_actions=None):
    result = events.copy()
    known_team = events.object_id.astype(str).str[:4].isin(["home", "away"])
    targets = extended_possession_targets(events.loc[known_team], events, families=families, max_actions=max_actions)
    targets = targets.reindex(events.index, fill_value=0.)
    targets = targets.rename(columns={"scores_xt": "scores_xT", "concedes_xt": "concedes_xT"})
    result[targets.columns] = targets
    return result
