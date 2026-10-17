"""Carry availability for visualizations, independent of raw model predictions."""
from __future__ import annotations

import pandas as pd


def carry_visualization_warnings(source_use_carries: dict[str, bool]) -> list[str]:
    if source_use_carries.get("pass_success") is not False:
        return []
    message = "The pass-success checkpoint has use_carries=False."
    if len(set(source_use_carries.values())) > 1:
        details = ", ".join(f"{task}={value}" for task, value in source_use_carries.items())
        message = f"Selected carry-sensitive checkpoints do not agree on use_carries: {details}."
    message += (
        " Ball-carrier pass-success, model pass-success, and pass-score values are unavailable"
        " in every frame; teammate pass predictions remain available."
    )
    if any(source_use_carries.get(task) for task in ("outcome_scoring", "outcome_conceding")):
        message += " Carry outcome values from carry-trained checkpoints remain available."
    return [message]


def mask_unsupported_carry_success(
    components: dict[str, pd.Series | pd.DataFrame], possessors: str | pd.Series,
) -> None:
    """Mask the self option on copies, preserving shared prediction caches.

    Table indexes are inference action/state IDs, including synthetic Hawkeye frame
    IDs. A real pass label still has a carry candidate at the possessor's node.
    """
    for name in ("pass_success", "model_pass_success"):
        values = components.get(name)
        if values is None or values.empty:
            continue
        masked = values.copy(deep=True)
        if isinstance(masked, pd.Series):
            if possessors in masked.index:
                masked.loc[possessors] = float("nan")
        else:
            for index in masked.index:
                player = possessors.loc[index]
                if player in masked.columns:
                    masked.at[index, player] = float("nan")
        components[name] = masked
