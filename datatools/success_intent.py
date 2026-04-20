from __future__ import annotations

import pandas as pd

SUCCESS_INTENT_LABEL_SOURCE = "receiver_id"
SUCCESS_INTENT_TRAINING_FILTER = "successful_pass_actions"


def build_success_intent_resolved_actions(actions: pd.DataFrame) -> pd.DataFrame:
    resolved_actions = actions.copy()
    resolved_actions["intent_id"] = pd.Series(index=resolved_actions.index, dtype="object")

    object_ids = resolved_actions["object_id"].astype("string")
    receiver_ids = resolved_actions["receiver_id"].astype("string")
    valid_receiver_mask = (
        receiver_ids.notna()
        & receiver_ids.ne("out")
        & ~receiver_ids.str.endswith("goal", na=False)
        & receiver_ids.str.slice(0, 4).eq(object_ids.str.slice(0, 4))
    )
    success_pass_mask = (
        resolved_actions["action_type"].eq("pass")
        & resolved_actions["success"].eq(True)
        & valid_receiver_mask
    )
    resolved_actions.loc[success_pass_mask, "intent_id"] = resolved_actions.loc[success_pass_mask, "receiver_id"]
    resolved_actions.attrs["success_intent_stats"] = {
        "successful_pass_actions": int((resolved_actions["action_type"].eq("pass") & resolved_actions["success"].eq(True)).sum()),
        "labeled_successful_pass_actions": int(success_pass_mask.sum()),
    }
    return resolved_actions
