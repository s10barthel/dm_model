"""Pass endpoints and receiver comparisons independent of runtime action labels."""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd

from datatools.utils import label_intended_receivers


# Ring diameters are in display points, independent of pitch size or rotation.
INTENDED_RECEIVER_STYLES = {
    "original": ("#009E73", 12.0),
    "angle_only": ("#E69F00", 18.0),
    "model": ("#9467BD", 24.0),
}


@dataclass(frozen=True)
class ActionOverlays:
    endpoint_xy: tuple[float, float] | None = None
    receivers: dict[str, str] = field(default_factory=dict)


def _finite_xy(x, y) -> tuple[float, float] | None:
    try:
        coordinates = (float(x), float(y))
    except (TypeError, ValueError):
        return None
    return coordinates if np.isfinite(coordinates).all() else None


def resolve_pass_endpoint(match, action_index: int) -> tuple[float, float]:
    action = match.actions.loc[action_index]
    endpoint = _finite_xy(action.get("end_x"), action.get("end_y"))
    if endpoint is not None:
        return endpoint
    receive_frame = action.get("receive_frame_id")
    if pd.notna(receive_frame) and receive_frame in match.tracking.index:
        snapshot = match.tracking.loc[receive_frame]
        endpoint = _finite_xy(snapshot.get("ball_x"), snapshot.get("ball_y"))
        if endpoint is not None:
            return endpoint
    raise ValueError(f"Missing actual pass endpoint for action {action_index}.")


def _heuristic_receivers(match, action_index: int) -> dict[str, str]:
    cache = getattr(match, "_visualization_heuristic_receivers", None)
    if cache is None:
        cache = match._visualization_heuristic_receivers = {}
    if action_index not in cache:
        receivers = {}
        for mode in ("original", "angle_only"):
            # Restrict work to this action; the utility copies the rows before relabeling.
            labeled = label_intended_receivers(
                match.actions.loc[[action_index]], match.tracking, action_type="pass",
                intended_receiver_mode=mode, fps=match.fps,
            )
            receivers[mode] = labeled.at[action_index, "intent_id"]
        cache[action_index] = receivers
    return dict(cache[action_index])


def resolve_intended_receivers(match, action_index: int, probs: pd.Series | None) -> dict[str, str]:
    action = match.actions.loc[action_index]
    possessor = action.get("object_id")
    if not isinstance(possessor, str) or not possessor.startswith(("home_", "away_")):
        raise ValueError(f"Missing pass possessor for action {action_index}.")
    prefix = possessor[:4]
    if pd.isna(action.get("success")):
        raise ValueError(f"Missing pass success value for action {action_index}.")
    if probs is None or not probs.index.is_unique:
        raise ValueError(f"Missing or ambiguous success-intent predictions for action {action_index}.")
    candidates = probs.loc[[
        player for player in probs.index
        if isinstance(player, str) and re.fullmatch(rf"{prefix}_\d+", player) and player != possessor
    ]]
    candidates = pd.to_numeric(candidates, errors="coerce")
    candidates = candidates[np.isfinite(candidates) & (candidates >= 0)]
    if candidates.empty or candidates.sum() <= 0:
        raise ValueError(f"Unusable success-intent predictions for action {action_index}.")

    if bool(action["success"]):
        receivers = dict.fromkeys(INTENDED_RECEIVER_STYLES, action.get("receiver_id"))
    else:
        receivers = _heuristic_receivers(match, action_index)
        receivers["model"] = candidates.idxmax()

    frame = action.get("frame_id")
    if pd.isna(frame) or frame not in match.tracking.index:
        raise ValueError(f"Missing pass-start tracking frame for action {action_index}.")
    snapshot = match.tracking.loc[frame]
    for mode, player in receivers.items():
        if not isinstance(player, str) or not re.fullmatch(rf"{prefix}_\d+", player) or player == possessor:
            raise ValueError(f"Missing {mode} intended receiver for action {action_index}.")
        if _finite_xy(snapshot.get(f"{player}_x"), snapshot.get(f"{player}_y")) is None:
            raise ValueError(f"Missing pass-start position for {mode} receiver {player} in action {action_index}.")
    return receivers


def prepare_action_overlays(
    match, action_index: int, probs: pd.Series | None = None, *,
    pass_endpoint: bool = True, compare_intended_receiver: bool = False,
) -> ActionOverlays:
    """Validate every required overlay before any image of an action is published."""
    if match.actions.at[action_index, "action_type"] != "pass":
        return ActionOverlays()
    endpoint = resolve_pass_endpoint(match, action_index) if pass_endpoint else None
    receivers = resolve_intended_receivers(match, action_index, probs) if compare_intended_receiver else {}
    return ActionOverlays(endpoint, receivers)
