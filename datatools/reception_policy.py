"""Run-scoped retirement of Elastic reception timing (not launch timing)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

LEGACY = "legacy_elastic_v1"
RETIRED = "provider_receptions_only_v1"
KEY = "reception_policy"


def event_key(value):
    return "" if pd.isna(value) else str(value).removesuffix(".0")


def policy_version(metadata):
    value = metadata.get(KEY, LEGACY)
    if value not in {LEGACY, RETIRED}:
        raise ValueError(f"Unsupported reception policy: {value}")
    return value


def run_version(root):
    path = Path(root) / "metadata.json" if root is not None else None
    return policy_version(json.loads(path.read_text(encoding="utf-8-sig"))) if path and path.exists() else LEGACY


def retired_mask(actions):
    source = actions.get("endpoint_source", pd.Series("", index=actions.index))
    previous = actions.get("reception_retired", pd.Series(False, index=actions.index))
    return source.eq("elastic") | previous.map(lambda x: str(x).lower() in {"true", "1", "1.0"})


def apply_policy(actions, version=LEGACY, *, mode=None, records=None):
    """Keep identity/outcome data; remove only retired timing and its derivatives."""
    if version not in {LEGACY, RETIRED}:
        raise ValueError(f"Unsupported reception policy: {version}")
    if version == LEGACY:
        return actions.copy()
    result = actions.copy()
    mask = retired_mask(result)
    if records is not None and "original_event_id" in result:
        keys = {event_key(r["event_id"]) for r in records}
        mask |= result.original_event_id.map(event_key).isin(keys)
    # Synthetic carries use their own validated segment endpoints.
    mask &= result.get("spadl_type", pd.Series("", index=result.index)).ne("ball_carry")
    result["reception_retired"] = mask
    result["reception_policy_version"] = RETIRED
    for column in ("receive_frame_id", "receive_ts"):
        if column in result:
            audit = "retired_" + column
            if audit not in result:
                result[audit] = result[column]
            result.loc[mask, column] = np.nan
    for column in ("intent_x", "intent_y", "pass_max_ball_z", "pass_high"):
        if column in result:
            result.loc[mask, column] = np.nan
    if mode == "original" and "intent_id" in result:
        failure = result.get("success", result.get("source_event_success", pd.Series(None, index=result.index))).eq(False)
        result.loc[mask & failure, "intent_id"] = None
    return result


def apply_run_policy(actions, root, match_id=None, *, mode=None):
    version = run_version(root)
    records = None
    if version == RETIRED and match_id is not None:
        path = Path(root) / "reception_retirement" / f"{match_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing pinned reception policy records: {path}")
        records = json.loads(path.read_text(encoding="utf-8"))["receptions"]
    return apply_policy(actions, version, mode=mode, records=records)


def retirement_records(events):
    records = []
    for _, row in events.loc[retired_mask(events)].iterrows():
        frame = row.get("retired_receive_frame_id", row.get("receive_frame_id"))
        records.append(dict(event_id=event_key(row.original_event_id),
                            frame=None if pd.isna(frame) else int(frame), source="elastic"))
    return records
