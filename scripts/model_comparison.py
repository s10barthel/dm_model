"""Compare independent checkpoints or runtime masks on identical prediction rows."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.utils import get_model_record, parse_model_id
from project_config import load_recorded_split_manifest
from scripts.comparison_utils import INTENT_TASKS, paired_bootstrap, comparison_identity_columns
from scripts.comparison_warnings import ComparisonWarnings, metadata_differences
from models.evaluation_context import recorded_manifest_id
from models.evaluation_membership import (combine_evaluation_membership, collect_evaluation_membership,
                                          accept_recorded_membership)


TARGET_DEFINITION_KEYS = ("target_family", "return_type", "intended_receiver_mode", "label_source",
                          "pass_height_definition")
IDENTITY_COLUMNS = ("soft_target", "execution_branch",
                    "evaluation_pass_height_threshold_meters", "pass_height_label_mode")
SOFT_TARGET_RTOL = 1e-6
SOFT_TARGET_ATOL = 1e-8


def reference_differences(left: pd.Series, right: pd.Series, field: str) -> pd.Series:
    if field == "soft_target":
        return pd.Series(~np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                    rtol=SOFT_TARGET_RTOL, atol=SOFT_TARGET_ATOL), index=left.index)
    return left != right


def checkpoint(model_id: str, *, expected_task: str | None = None) -> dict:
    """Validate completion/provenance without learning-curve training-size restrictions."""
    task, _ = parse_model_id(model_id)
    if expected_task is not None and task != expected_task:
        raise ValueError(f"Checkpoint {model_id!r} does not match selector task {expected_task!r}.")
    record = get_model_record(model_id)
    if record.get("task") != task or not record.get("is_complete"):
        raise ValueError(f"Checkpoint {model_id!r} is incomplete or has a different task.")
    metadata = record.get("metadata") or {}
    train_data = ((metadata.get("split_datasets") or {}).get("train") or {})
    train_ids = train_data.get("contributing_match_ids")
    if train_ids is None:
        train_ids = metadata.get("train_match_ids") or []
    train_ids = [str(value) for value in train_ids]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError(f"Checkpoint {model_id!r} has duplicate recorded training match IDs.")
    manifest_id = recorded_manifest_id(record)
    test_ids = None
    if manifest_id:
        manifest = load_recorded_split_manifest(manifest_id)
        test_ids = [str(value) for value in manifest["test"]]
        if set(train_ids).intersection(test_ids):
            raise ValueError(f"Checkpoint {model_id!r} has training matches in its test set.")
    configuration = {}
    for name in TARGET_DEFINITION_KEYS:
        value = record.get(name, metadata.get(name, (record.get("args") or {}).get(name)))
        if name == "pass_height_definition" and value:
            value = {key: item for key, item in value.items() if key != "resolution_source"}
        configuration[name] = value
    if task not in {"pass_height", "pass_success", "outcome_scoring", "outcome_conceding"}:
        configuration.pop("pass_height_definition", None)
    return {
        "model_id": model_id, "task": task, "train_matches": len(train_ids),
        "train_match_ids": train_ids, "test_match_ids": test_ids,
        "split_manifest_id": manifest_id, "target_definitions": configuration,
        "checkpoint_record": record,
    }


def validate_target_definitions(records: list[dict]) -> list[str]:
    if not records or len({record["task"] for record in records}) != 1:
        raise ValueError("Checkpoint comparisons require models of the same task.")
    definitions = [{**(record.get("target_definitions") or {}), "model_id": record["model_id"]} for record in records]
    task = records[0]["task"]
    keys = ("target_family", "return_type", "intended_receiver_mode", "label_source") if task.startswith("outcome_") else (
        "intended_receiver_mode", "label_source", "pass_height_definition")
    messages = metadata_differences(definitions, keys, prefix="Evaluation reference ")
    if task.startswith("outcome_"):
        messages = [message + " Native-target metrics use different references; binary goal metrics retain the common diagnostic definition."
                    if any(f"{key} differs" in message for key in ("target_family", "return_type", "label_source")) else message
                    for message in messages]
    return messages


def definition_warning_key(message, task):
    if "intended_receiver_mode differs" in message:
        return "statistical:receiver_reference"
    if "pass_height_definition differs" in message:
        return "statistical:height_reference"
    if task.startswith("outcome_"):
        return "statistical:native_reference"
    return "statistical:label_reference"


def add_definition_warnings(policy, records):
    for message in validate_target_definitions(records):
        policy.add(message, key=definition_warning_key(message, records[0]["task"]))


def reference_qualification(task, metric, records, frames):
    native = task in {"outcome_scoring", "outcome_conceding"} and metric.startswith("xt_")
    fields = ["soft_target", "execution_branch", "receiver_id"] if native else [
        "target_id" if task in INTENT_TASKS else "target", "execution_branch", "receiver_id",
        "diagnostic_return_type", "evaluation_pass_height_threshold_meters"]
    for field in fields:
        if any(field in frame for frame in frames):
            if any(field not in frame for frame in frames) or any(
                    reference_differences(frame[field], frames[0][field], field).any() for frame in frames[1:]):
                return "different_references"
    keys = ("target_family", "return_type", "label_source") if native else (
        ("label_source", "pass_height_definition") if task == "pass_height" else (() if task.startswith("outcome_") else ("label_source",)))
    definitions = [record.get("target_definitions") or {} for record in records]
    if any(any(definition.get(key) != definitions[0].get(key) for definition in definitions[1:]) for key in keys):
        return "different_references"
    return "common_references"


def _candidate_set(value: str) -> frozenset[str]:
    try:
        candidates = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Intent candidate_ids must contain JSON arrays.") from exc
    if not isinstance(candidates, list):
        raise ValueError("Intent candidate_ids must contain JSON arrays.")
    return frozenset(str(item) for item in candidates)


def load_aligned_predictions(records: list[dict]) -> tuple[list[pd.DataFrame], list[dict]]:
    """Align all rows strictly; candidate membership differences are warnings only."""
    frames = []
    messages = []
    reference = None
    reference_candidates = None
    raw_frames = [pd.read_csv(Path(record["output_dir"]) / "learning_curve_predictions.csv",
                              dtype={"match_id": str, "target_id": str, "prediction_id": str},
                              keep_default_na=False) for record in records]
    identity_columns = comparison_identity_columns(raw_frames)
    for record, frame in zip(records, raw_frames):
        path = Path(record["output_dir"]) / "learning_curve_predictions.csv"
        required = {"match_id", "source_index", "target", "prediction", *identity_columns}
        intent = record["task"] in INTENT_TASKS
        if intent:
            required |= {"target_id", "prediction_id", "candidate_ids", "target_probability", "reciprocal_rank"}
        if frame.empty or not required.issubset(frame):
            raise ValueError(f"Missing prediction columns or rows in {path}.")
        if "action_index" in frame:
            action_indices = pd.to_numeric(frame["action_index"], errors="coerce")
            if not (np.isfinite(action_indices) & (action_indices % 1 == 0)).all():
                raise ValueError(f"Missing or invalid event identities in {path}.")
            frame["action_index"] = action_indices.astype("int64")
        if frame.duplicated(identity_columns).any():
            raise ValueError(f"Duplicate example identities in {path}.")
        if frame["match_id"].eq("").any() or frame["source_index"].eq("").any():
            raise ValueError(f"Missing example identities in {path}.")
        frame = frame.sort_values(identity_columns).reset_index(drop=True)
        if record.get("test_match_ids") is not None and not set(frame["match_id"]).issubset(record["test_match_ids"]):
            raise ValueError(f"Predictions outside the test manifest in {path}.")
        if set(frame["match_id"]).intersection(record.get("train_match_ids") or []):
            raise ValueError(f"Predictions include recorded training matches in {path}.")
        if record.get("checkpoint_record", {}).get("model_id"):
            cohort = record.get("test_match_ids")
            if cohort is None:
                cohort = frame["match_id"].unique()
            checked = collect_evaluation_membership([(record["checkpoint_record"], cohort)])
            if checked["status"] == "unverified":
                accept_recorded_membership(checked, record.get("evaluation_membership"))
        target_name = "target_id" if intent else "target"
        if intent and (frame["target_id"].eq("").any() or frame["prediction_id"].eq("").any()):
            raise ValueError(f"Missing semantic intent identities in {path}.")
        identity = frame[identity_columns].copy()
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            expected = set(map(tuple, reference.to_numpy()))
            actual = set(map(tuple, identity.to_numpy()))
            raise ValueError(f"Evaluated action populations differ for {record['model_id']}: "
                             f"reference={len(expected)}, variant={len(actual)}, missing={len(expected-actual)}, "
                             f"extra={len(actual-expected)}; examples={list(expected ^ actual)[:5]}. No intersection is performed.")
        if frames:
            for name in (target_name, *IDENTITY_COLUMNS, "receiver_id", "diagnostic_return_type"):
                if name in frame or name in frames[0]:
                    if name not in frame or name not in frames[0]:
                        different = pd.Series(True, index=frame.index)
                    else:
                        different = reference_differences(frame[name], frames[0][name], name)
                    if different.any():
                        messages.append({"kind": "evaluation_reference_mismatch", "column": name,
                                         "model_id": record["model_id"], "affected_row_count": int(different.sum()),
                                         "examples": frame.loc[different, identity_columns].head(5).to_dict("records")})
        if intent:
            candidates = frame["candidate_ids"].map(_candidate_set)
            if reference_candidates is None:
                reference_candidates = candidates
            else:
                different = candidates != reference_candidates
                if different.any():
                    sample = frame.loc[different, identity_columns].head(5).to_dict("records")
                    message = {
                        "kind": "candidate_membership_mismatch", "model_id": record["model_id"],
                        "configuration_id": record.get("configuration_id", "unmasked"),
                        "reference_model_id": records[0]["model_id"],
                        "reference_configuration_id": records[0].get("configuration_id", "unmasked"),
                        "affected_row_count": int(different.sum()), "examples": sample,
                    }
                    messages.append(message)
                    warnings.warn(f"Candidate membership differs for {record['model_id']} on {int(different.sum())} rows; "
                                  "paired comparison continues.", RuntimeWarning, stacklevel=2)
        frames.append(frame)
    return frames, messages


def evaluation_membership_status(records):
    statuses = [(record.get("evaluation_membership") or {}).get("status", "unknown") for record in records]
    return "unverified" if "unverified" in statuses else "verified" if all(s == "verified" for s in statuses) else "unknown"


def compare(task: str, records: list[dict], frames: list[pd.DataFrame], *, resamples: int, seed: int,
            threshold: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, intervals, pair_intervals = paired_bootstrap(
        task, frames, resamples=resamples, seed=seed, threshold=threshold, strict_threshold=True,
    )
    rows = []
    differences = []
    for index, record in enumerate(records):
        for metric, estimate in points[index].items():
            low, high, valid, status = intervals[index, metric]
            rows.append({"task": task, "model_id": record["model_id"],
                         "configuration_id": record.get("configuration_id", "unmasked"),
                         "label": record.get("label", record["model_id"]), "metric": metric,
                         "estimate": estimate, "lower": low, "upper": high,
                         "evaluation_membership_status": evaluation_membership_status([record]),
                         "valid_resamples": valid, "status": status})
    for (a, b, metric), (low, high, valid, status) in pair_intervals.items():
        differences.append({
            "task": task, "model_a": records[a]["model_id"], "model_b": records[b]["model_id"],
            "configuration_a": records[a].get("configuration_id", "unmasked"),
            "configuration_b": records[b].get("configuration_id", "unmasked"),
            "direction": f"{records[b].get('label', records[b]['model_id'])} - {records[a].get('label', records[a]['model_id'])}",
            "metric": metric, "difference": points[b][metric] - points[a][metric],
            "evaluation_membership_status": evaluation_membership_status([records[a], records[b]]),
            "lower": low, "upper": high, "valid_resamples": valid, "status": status,
        })
    return pd.DataFrame(rows), pd.DataFrame(differences)


def save_plot(summary: pd.DataFrame, output_path: Path, task: str) -> None:
    metric_names = summary["metric"].unique()
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(9, max(3, 2.5 * len(metric_names))), squeeze=False)
    for axis, name in zip(axes[:, 0], metric_names):
        data = summary.loc[summary.metric == name]
        x = np.arange(len(data))
        y = data.estimate.to_numpy(dtype=float)
        axis.plot(x, y, "o")
        low, high = data.lower.to_numpy(dtype=float), data.upper.to_numpy(dtype=float)
        finite = np.isfinite(low) & np.isfinite(high)
        axis.vlines(x[finite], low[finite], high[finite])
        axis.set_xticks(x, data.label.tolist(), rotation=20, ha="right")
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    qualification = ("\nEvaluation membership unverified" if
                     "evaluation_membership_status" in summary and
                     summary.evaluation_membership_status.eq("unverified").any() else "")
    fig.suptitle(f"{task}: identical evaluated rows{qualification}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_report(records: list[dict], output_dir: Path, *, resamples: int, seed: int,
                 threshold: float | None = None, warning_policy=None) -> dict:
    if len(records) < 2:
        raise ValueError("A comparison report requires at least two configurations or checkpoints.")
    policy = warning_policy or ComparisonWarnings()
    add_definition_warnings(policy, records)
    policy.confirm()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, messages = load_aligned_predictions(records)
    for message in messages:
        column = message.get("column")
        key = {"receiver_id": "statistical:receiver_reference", "target_id": "statistical:receiver_reference",
               "soft_target": "statistical:native_reference", "evaluation_pass_height_threshold_meters": "statistical:height_reference",
               "pass_height_label_mode": "statistical:height_reference"}.get(column)
        if column == "target" and records[0]["task"] == "pass_height":
            key = "statistical:height_reference"
        policy.add(f"{message['kind']}: {message}. Metrics and paired differences use each variant's reference.", key=key)
    policy.confirm()
    task = records[0]["task"]
    summary, differences = compare(task, records, frames, resamples=resamples, seed=seed, threshold=threshold)
    for table in (summary, differences):
        table["reference_qualification"] = table["metric"].map(
            lambda metric: reference_qualification(task, metric, records, frames))
    summary.to_csv(output_dir / "comparison_metrics.csv", index=False)
    differences.to_csv(output_dir / "comparison_differences.csv", index=False)
    identity_names = [name for name in ("match_id", "action_index", "source_index", "target", "target_id", *IDENTITY_COLUMNS)
                      if name in frames[0]]
    aligned = frames[0][identity_names].copy()
    for index, frame in enumerate(frames):
        for name in ("source_index", "target", "target_id", *IDENTITY_COLUMNS, "receiver_id", "diagnostic_return_type",
                     "prediction", "prediction_id", "target_probability", "reciprocal_rank", "candidate_ids"):
            if name in frame:
                aligned[f"{name}_{index}"] = frame[name].to_numpy()
    aligned.to_csv(output_dir / "aligned_predictions.csv", index=False)
    save_plot(summary, output_dir / "comparison.png", task)
    membership = combine_evaluation_membership(record["evaluation_membership"] for record in records
                                              if record.get("evaluation_membership"))
    membership["status"] = evaluation_membership_status(records)
    report = {"task": task, "variants": records, **policy.metadata(), "reference_differences": messages,
              "evaluation_membership": membership,
              "identity_columns": comparison_identity_columns(frames),
              "soft_target_tolerance": {"rtol": SOFT_TARGET_RTOL, "atol": SOFT_TARGET_ATOL},
              "example_count": len(frames[0]), "contributing_match_ids": sorted(frames[0].match_id.unique().tolist()),
              "bootstrap_resamples": resamples, "bootstrap_seed": seed, "confidence_level": 0.95,
              "interval_method": "paired_match_percentile", "interval_scope": "pointwise",
              "classification_threshold": threshold}
    (output_dir / "experiment_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
