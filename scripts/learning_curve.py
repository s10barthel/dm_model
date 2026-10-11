"""Compare saved model checkpoints on one aligned test cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.utils import get_model_path, parse_model_id
from scripts.comparison_utils import metrics, paired_bootstrap
from project_config import load_recorded_split_manifest, VALIDATION_PROTOCOL_SHARED


OUTCOME_TASKS = {"outcome_scoring", "outcome_conceding"}
INTENT_TASKS = {"action_intent", "pass_intent", "success_intent"}
BINARY_TASKS = {"pass_success", "pass_height"}


def checkpoint(model_id: str) -> dict:
    path = get_model_path(model_id)
    metadata_path = path / "metadata.json"
    weights_path = path / "best_weights.pt"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Completed checkpoint files missing for {model_id}: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task, _ = parse_model_id(model_id)
    if metadata.get("status") != "completed" or metadata.get("task") != task:
        raise ValueError(f"Checkpoint {model_id} is incomplete or has a different task.")
    requested_train_ids = [str(value) for value in metadata.get("train_match_ids") or []]
    contributing = ((metadata.get("split_datasets") or {}).get("train") or {}).get("contributing_match_ids")
    train_ids = [str(value) for value in (contributing if contributing is not None else requested_train_ids)]
    if not train_ids or len(train_ids) != len(set(train_ids)):
        raise ValueError(f"Checkpoint {model_id} lacks unique recorded training match IDs.")
    manifest_id = metadata.get("split_manifest_id")
    if not manifest_id:
        raise ValueError(f"Checkpoint {model_id} lacks a split manifest ID.")
    manifest = load_recorded_split_manifest(manifest_id)
    height_definition = metadata.get("pass_height_definition")
    if height_definition:
        height_definition = {key: value for key, value in height_definition.items() if key != "resolution_source"}
    test_ids = [str(value) for value in manifest["test"]]
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Checkpoint {model_id} has training matches in its test set.")
    return {
        "model_id": model_id, "task": task, "train_match_ids": train_ids,
        "requested_train_match_ids": requested_train_ids,
        "train_matches": len(train_ids), "test_match_ids": test_ids,
        "split_manifest_id": manifest_id, "seed": metadata.get("training_args", {}).get("seed"),
        "validation_protocol": metadata.get("validation_protocol"),
        "final_refit": bool(metadata.get("final_refit")),
        "validation_match_ids": metadata.get("validation_match_ids", []),
        "validation_provenance": (metadata.get("split_datasets") or {}).get("validation"),
        "validation_example_identity": metadata.get("validation_example_identity"),
        "comparison_config": {
            key: height_definition if key == "pass_height_definition" else metadata.get(key) for key in (
                "feature_run_id", "use_carries", "min_pass_dur", "target_family",
                "intended_receiver_mode", "training_filter", "label_source", "feature_signature",
                "pass_height_definition",
            )
        },
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    }


def discover(anchor_id: str) -> list[str]:
    """Discover fold checkpoints; the promoted root represents final_refit once."""
    task, run_id = parse_model_id(anchor_id)
    run_id = run_id.split("/", 1)[0]
    root_id = f"{task}/{run_id}"
    root = get_model_path(root_id)
    ids = []
    folds = sorted((p.name for p in root.glob("fold_*") if p.is_dir() and p.name[5:].isdigit()),
                   key=lambda name: int(name[5:]))
    if (root / "holdout").is_dir():
        folds.append("holdout")
    for fold in folds:
        path = root / fold
        if (path / "metadata.json").is_file() and (path / "best_weights.pt").is_file():
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                ids.append(f"{root_id}/{fold}")
    root_promoted = False
    if (root / "metadata.json").is_file():
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") == "completed" and metadata.get("final_refit"):
            ids.append(root_id)
            root_promoted = True
    if not root_promoted and (root / "final_refit" / "metadata.json").is_file():
        metadata = json.loads((root / "final_refit" / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") == "completed":
            ids.append(f"{root_id}/final_refit")
    return ids


def validate_series(model_ids: list[str], *, expected_task: str) -> list[dict]:
    records = [checkpoint(model_id) for model_id in model_ids]
    if any(record["task"] != expected_task for record in records):
        raise ValueError(f"Learning-curve IDs for {expected_task} include another task.")
    records.sort(key=lambda record: record["train_matches"])
    sizes = [record["train_matches"] for record in records]
    if len(sizes) < 2 or len(sizes) != len(set(sizes)):
        raise ValueError(f"{expected_task} requires at least two distinct training sizes; found {sizes}.")
    reference_test = records[0]["test_match_ids"]
    reference_manifest = records[0].get("split_manifest_id")
    reference_seed = records[0]["seed"]
    reference_config = records[0].get("comparison_config")
    for smaller, larger in zip(records, records[1:]):
        if not set(smaller["train_match_ids"]).issubset(larger["train_match_ids"]):
            raise ValueError(f"Training matches are not nested: {smaller['model_id']} and {larger['model_id']}.")
    for record in records:
        if record["test_match_ids"] != reference_test:
            raise ValueError(f"Test membership differs for {record['model_id']}.")
        if record.get("split_manifest_id") != reference_manifest:
            raise ValueError(f"Split manifest differs for {record['model_id']}.")
        if record["seed"] != reference_seed:
            raise ValueError(f"Training seed differs for {record['model_id']}.")
        if record.get("comparison_config") != reference_config:
            raise ValueError(f"Feature or target configuration differs for {record['model_id']}.")
    shared = [r for r in records if r.get("validation_protocol") == VALIDATION_PROTOCOL_SHARED and not r.get("final_refit")]
    if shared and any(r.get("validation_match_ids") != shared[0].get("validation_match_ids") for r in shared):
        raise ValueError("Shared validation membership differs between checkpoints.")
    for record in shared:
        if not record.get("validation_match_ids") or set(record["validation_match_ids"]) & set(record["train_match_ids"]):
            raise ValueError("Shared validation membership is missing or overlaps training.")
        provenance = record.get("validation_provenance") or {}
        reference_provenance = shared[0].get("validation_provenance") or {}
        if (provenance.get("contributing_match_ids") != reference_provenance.get("contributing_match_ids")
                or record.get("validation_example_identity") != shared[0].get("validation_example_identity")):
            raise ValueError("Usable validation membership differs between checkpoints.")
    return records


def load_aligned_predictions(records: list[dict], output_dir: Path) -> list[pd.DataFrame]:
    aligned = []
    reference = None
    for record in records:
        path = output_dir / str(record["train_matches"]) / "learning_curve_predictions.csv"
        frame = pd.read_csv(path)
        required = {"match_id", "source_index", "target", "prediction"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"Missing prediction columns or rows in {path}.")
        if frame.duplicated(["match_id", "source_index"]).any():
            raise ValueError(f"Duplicate example identities in {path}.")
        frame = frame.sort_values(["match_id", "source_index"]).reset_index(drop=True)
        if not set(frame["match_id"]).issubset(record["test_match_ids"]):
            raise ValueError(f"Predictions outside the test manifest in {path}.")
        identity = frame[["match_id", "source_index", "target"]].copy()
        for column in ("soft_target", "execution_branch", "model_pass_height_threshold_meters",
                       "evaluation_pass_height_threshold_meters", "pass_height_label_mode"):
            if column in frame:
                identity[column] = frame[column]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError(f"Evaluated examples or targets differ for {record['model_id']}.")
        aligned.append(frame)
    return aligned


def compare(task: str, records: list[dict], frames: list[pd.DataFrame], *, resamples: int, seed: int,
            threshold: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    points, intervals, pair_intervals = paired_bootstrap(
        task, frames, resamples=resamples, seed=seed, threshold=threshold,
    )
    rows = []
    diff_rows = []
    for index, record in enumerate(records):
        for name, estimate in points[index].items():
            low, high, valid, status = intervals[index, name]
            rows.append({"task": task, "model_id": record["model_id"], "train_matches": record["train_matches"],
                         "metric": name, "estimate": estimate, "lower": low, "upper": high,
                         "valid_resamples": valid, "status": status})
    for (a, b, name), (low, high, valid, status) in pair_intervals.items():
        if a != 0 and b != a + 1:
            continue
        diff_rows.append({"task": task, "from_matches": records[a]["train_matches"],
                          "to_matches": records[b]["train_matches"], "metric": name,
                          "difference": points[b][name] - points[a][name], "lower": low, "upper": high,
                          "valid_resamples": valid, "status": status})
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)

def save_plot(summary: pd.DataFrame, output_path: Path, task: str) -> None:
    metric_names = summary["metric"].unique()
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(7, max(3, 2.5 * len(metric_names))), squeeze=False)
    for axis, name in zip(axes[:, 0], metric_names):
        data = summary.loc[summary.metric == name].sort_values("train_matches")
        x = data.train_matches.to_numpy()
        y = data.estimate.to_numpy(dtype=float)
        axis.plot(x, y, marker="o")
        axis.fill_between(x, data.lower.to_numpy(dtype=float), data.upper.to_numpy(dtype=float), alpha=0.2)
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Training matches")
    fig.suptitle(f"{task}: common test set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
