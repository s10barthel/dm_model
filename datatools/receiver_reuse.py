"""Receiver predictions may be reused only within one offline extension attempt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from numbers import Real

import pandas as pd

from datatools.pass_trajectory import event_key
from datatools.input_diagnostics import read_json
from project_config import SAVED_DIR

PROVENANCE = "receiver_prediction_provenance"
POLICY_KEYS = ("graph_schema", "reception_policy", "failed_pass_trajectory_policy",
               "failed_pass_detector_settings", "trajectory_speed_definition",
               "pass_duration_definition", "max_pass_duration")
IDENTITY_COLUMNS = ("period_id", "frame_id", "object_id", "observed_end_frame_id",
                    "trajectory_end_frame_id", "pass_endpoint_definition", "pass_duration")


def prediction_provenance(root, match_id, model_id, attempt_id):
    if not attempt_id:
        return None
    root = Path(root)
    metadata = read_json(root / "metadata.json")
    inputs = {key: metadata.get(key) for key in POLICY_KEYS}
    for name, path in (("checkpoint_args", SAVED_DIR / model_id / "args.json"),
                       ("endpoints", root / "pass_endpoints" / f"{match_id}.parquet"),
                       ("observations", root / "failed_pass_trajectories" / f"{match_id}.parquet")):
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(version=1, attempt_id=attempt_id, match_id=str(match_id), model_id=model_id, signature=signature)


def _failed_passes(actions):
    return actions.loc[actions.action_type.eq("pass") & actions.success.eq(False)]


def _keys(actions):
    keys = actions.get("original_event_id", pd.Series(None, index=actions.index, dtype=object)).map(event_key)
    if keys[keys.ne("")].duplicated().any():
        raise ValueError("Duplicate failed-pass event identities in receiver predictions/actions")
    return keys


def _value(value):
    if pd.isna(value):
        return None
    if isinstance(value, Real):
        return float(value)
    return str(value)


def reusable_predictions(actions, predictions, provenance):
    """Return action-index keyed decisions, including explicitly unavailable ones."""
    failed = _failed_passes(actions)
    target_keys = _keys(failed)
    if predictions is None:
        return {}
    source = _failed_passes(predictions)
    source_keys = _keys(source)
    if not provenance or predictions.attrs.get(PROVENANCE) != provenance:
        return {}
    if not {"intent_id", "model_receiver_reason"}.issubset(source.columns):
        return {}
    lookup = {key: index for index, key in source_keys.items() if key}
    result = {}
    for index, key in target_keys.items():
        if key not in lookup:
            continue
        row = source.loc[lookup[key]]
        target = failed.loc[index]
        if any(column not in row or column not in target or _value(row[column]) != _value(target[column])
               for column in IDENTITY_COLUMNS):
            continue
        receiver, reason = row.intent_id, row.model_receiver_reason
        if pd.isna(receiver):
            if pd.isna(reason) or not str(reason):
                continue
            receiver = None
        else:
            receiver = str(receiver)
            if not receiver.startswith(("home_", "away_")):
                continue
        result[index] = (receiver, None if pd.isna(reason) else str(reason))
    return result
