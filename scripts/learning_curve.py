"""Compare saved model checkpoints on one aligned test cohort."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from models.utils import get_model_path, parse_model_id
from project_config import load_recorded_split_manifest


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
    test_ids = [str(value) for value in manifest["test"]]
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Checkpoint {model_id} has training matches in its test set.")
    return {
        "model_id": model_id, "task": task, "train_match_ids": train_ids,
        "requested_train_match_ids": requested_train_ids,
        "train_matches": len(train_ids), "test_match_ids": test_ids,
        "split_manifest_id": manifest_id, "seed": metadata.get("training_args", {}).get("seed"),
        "comparison_config": {
            key: metadata.get(key) for key in (
                "feature_run_id", "use_carries", "min_pass_dur", "target_family",
                "intended_receiver_mode", "training_filter", "label_source", "feature_signature",
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
    for fold in ("fold_1", "fold_2", "fold_3"):
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
        for column in ("soft_target", "execution_branch"):
            if column in frame:
                identity[column] = frame[column]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError(f"Evaluated examples or targets differ for {record['model_id']}.")
        aligned.append(frame)
    return aligned


def metrics(task: str, frame: pd.DataFrame, indices: np.ndarray | None = None,
            *, threshold: float | None = None) -> dict[str, float]:
    data = frame if indices is None else frame.iloc[indices]
    prediction = data["prediction"].to_numpy(dtype=float)
    target = data["target"].to_numpy(dtype=float)
    if task in INTENT_TASKS:
        probability = np.clip(data["target_probability"].to_numpy(dtype=float), 1e-12, 1)
        return {
            "accuracy": float(np.mean(prediction == target)),
            "mrr": float(data["reciprocal_rank"].mean()),
            "target_log_loss": float(-np.log(probability).mean()),
        }
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Learning-curve predictions and targets must be finite.")
    result = {
        "roc_auc": float(roc_auc_score(target, prediction)) if np.unique(target).size == 2 else math.nan,
        "pr_auc": float(average_precision_score(target, prediction)) if np.any(target > 0) else math.nan,
        "brier": float(brier_score_loss(target, prediction)),
        "log_loss": float(log_loss(target, prediction, labels=[0, 1])),
        "calibration_bias": float(np.mean(prediction - target)),
    }
    ordered = np.argsort(prediction, kind="stable")
    result["ece"] = float(sum(
        len(group) * abs(float(np.mean(prediction[group] - target[group])))
        for group in np.array_split(ordered, min(10, len(ordered)))
    ) / len(ordered))
    if threshold is not None:
        classified = prediction >= threshold
        positive = target > 0
        tp = int(np.sum(classified & positive))
        fp = int(np.sum(classified & ~positive))
        fn = int(np.sum(~classified & positive))
        result["accuracy"] = float(np.mean(classified == positive))
        result["precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
        result["recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
        result["f1"] = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    if task in OUTCOME_TASKS:
        soft = data["soft_target"].to_numpy(dtype=float)
        clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
        result.update({
            "xt_soft_bce": float(np.mean(-soft * np.log(clipped) - (1 - soft) * np.log1p(-clipped))),
            "xt_rmse": float(np.sqrt(np.mean((soft - prediction) ** 2))),
            "xt_spearman": float(pd.Series(soft).corr(pd.Series(prediction), method="spearman")),
            "xt_mean_bias": float(np.mean(prediction - soft)),
        })
    return result


def compare(task: str, records: list[dict], frames: list[pd.DataFrame], *, resamples: int, seed: int,
            threshold: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resamples < 0 or seed < 0:
        raise ValueError("Bootstrap resamples and seed must be nonnegative.")
    groups = frames[0].groupby("match_id", sort=True).indices
    match_ids = sorted(groups)
    if len(match_ids) < 2:
        raise ValueError("Paired bootstrap requires at least two contributing test matches.")
    names = list(metrics(task, frames[0], threshold=threshold))
    points = [metrics(task, frame, threshold=threshold) for frame in frames]
    draws = {name: [[] for _ in records] for name in names}
    differences = {(a, b, name): [] for a in range(len(records)) for b in range(a + 1, len(records)) for name in names}
    rng = np.random.default_rng(seed)
    last_report = time.perf_counter()
    for repeat in range(resamples):
        sampled = rng.integers(0, len(match_ids), size=len(match_ids))
        indices = np.concatenate([groups[match_ids[index]] for index in sampled])
        batch = [metrics(task, frame, indices, threshold=threshold) for frame in frames]
        for name in names:
            for index, item in enumerate(batch):
                draws[name][index].append(item[name])
            for a in range(len(records)):
                for b in range(a + 1, len(records)):
                    differences[(a, b, name)].append(batch[b][name] - batch[a][name])
        now = time.perf_counter()
        if now - last_report >= 30 or repeat + 1 == resamples:
            print(f"{task} paired bootstrap: {repeat + 1}/{resamples}", flush=True)
            last_report = now
    def interval(values):
        if resamples == 0:
            return math.nan, math.nan, 0, "disabled"
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < math.ceil(0.95 * resamples) or not len(finite):
            return math.nan, math.nan, len(finite), "insufficient_valid_resamples"
        lower, upper = np.quantile(finite, [0.025, 0.975])
        return float(lower), float(upper), len(finite), "ok"
    rows = []
    diff_rows = []
    for index, record in enumerate(records):
        for name in names:
            low, high, valid, status = interval(draws[name][index])
            rows.append({"task": task, "model_id": record["model_id"], "train_matches": record["train_matches"],
                         "metric": name, "estimate": points[index][name], "lower": low, "upper": high,
                         "valid_resamples": valid, "status": status})
    for (a, b, name), values in differences.items():
        if a != 0 and b != a + 1:
            continue
        low, high, valid, status = interval(values)
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
