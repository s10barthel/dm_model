"""Training gates independent of receipt duration; historical runs retain their policy."""
from pathlib import Path
import math
import copy

from datatools.input_diagnostics import read_parquet
from datatools.pass_trajectory import PASS_DURATION, PASS_DURATION_KEY
from datatools.failed_pass_trajectory import VERSION

VERSION_KEY = "pass_eligibility_policy"
VERSION_ID = "independent_receipt_clean_prefix_v1"


def policy_metadata():
    from datatools.failed_pass_trajectory import KEY, SETTINGS
    return {VERSION_KEY: VERSION_ID, KEY: VERSION, "failed_pass_detector_settings": dict(SETTINGS)}


def independent_policy(metadata):
    return metadata.get(PASS_DURATION_KEY) == PASS_DURATION


def training_policy(metadata):
    """Snapshot the input/selection contract instead of consulting it after training."""
    keys = (PASS_DURATION_KEY, VERSION_KEY, "failed_pass_trajectory_policy", "failed_pass_detector_settings",
            "max_pass_duration", "reception_policy", "trajectory_speed_definition")
    result = {key: metadata[key] for key in keys if key in metadata}
    if "carry_variant" in metadata:
        result["carry_definition"] = (metadata.get("carry_variant") or {}).get("definition")
    return copy.deepcopy(result)


def resolve_training_policy(args, metadata, checkpoint=None):
    actual = training_policy(metadata)
    expected = checkpoint.get("pass_input_policy") if checkpoint is not None else None
    if checkpoint is not None and ((expected is not None and expected != actual)
                                   or (expected is None and independent_policy(metadata))):
        raise ValueError("Resume/refit pass-input policy differs from the checkpoint; train a new compatible model.")
    if independent_policy(metadata) and getattr(args, "min_pass_dur", 0.) > 0:
        raise ValueError("--min_pass_dur must be 0 for independent receipt/observation inputs.")
    args.pass_input_policy = actual


def failed_pass_reasons(root, match_id, labels):
    """Return aligned reasons without removing events or changing return targets."""
    from datatools.config import LABEL_INDEX
    mask = (labels[:, LABEL_INDEX["is_pass"]] == 1) & (labels[:, LABEL_INDEX["success"]] == 0)
    reasons = [""] * len(labels)
    if not mask.any():
        return reasons
    path = Path(root) / "failed_pass_trajectories" / f"{match_id}.parquet"
    if not path.exists():
        raise ValueError(f"Missing failed-pass quality coverage: {path}; refresh feature inputs.")
    table = read_parquet(path)
    if table.action_index.duplicated().any() or table.event_key.duplicated().any():
        raise ValueError(f"Ambiguous failed-pass quality identities: {path}")
    records = table.set_index("action_index").to_dict("index")
    for pos in mask.nonzero().flatten().tolist():
        index = labels[pos, 0].item()
        record = records.get(index)
        if record is None or record.get("policy") != VERSION:
            raise ValueError(f"Missing/stale failed-pass quality for action {index}: {path}")
        if record["status"] == "unavailable":
            reasons[pos] = record.get("reason") or "unusable_trajectory"
        elif (record["status"] not in {"unchanged", "shortened"}
              or not math.isfinite(record.get("clean_intervals", float("nan")))
              or record["clean_intervals"] < 3):
            raise ValueError(f"Invalid failed-pass quality for action {index}: {path}")
    return reasons
