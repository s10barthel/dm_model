"""Shared task metrics and paired whole-match bootstrap for evaluation comparisons."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

OUTCOME_TASKS = {"outcome_scoring", "outcome_conceding"}
INTENT_TASKS = {"action_intent", "pass_intent", "success_intent"}
BINARY_TASKS = {"pass_success", "pass_height"}


def metrics(task: str, frame: pd.DataFrame, indices: np.ndarray | None = None,
            *, threshold: float | None = None, strict_threshold: bool = False) -> dict[str, float]:
    data = frame if indices is None else frame.iloc[indices]
    if task in INTENT_TASKS:
        prediction = data["prediction_id"].astype(str).to_numpy() if "prediction_id" in data else data["prediction"].to_numpy()
        target = data["target_id"].astype(str).to_numpy() if "target_id" in data else data["target"].to_numpy()
        probability = np.clip(data["target_probability"].to_numpy(dtype=float), 1e-12, 1)
        if not np.isfinite(probability).all() or not np.isfinite(data["reciprocal_rank"].to_numpy(dtype=float)).all():
            raise ValueError("Comparison intent probabilities and ranks must be finite.")
        return {
            "accuracy": float(np.mean(prediction == target)),
            "mrr": float(data["reciprocal_rank"].mean()),
            "target_log_loss": float(-np.log(probability).mean()),
        }
    prediction = data["prediction"].to_numpy(dtype=float)
    target = data["target"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Comparison predictions and targets must be finite.")
    if not np.isin(target, [0, 1]).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Binary comparison requires binary targets and predictions in [0, 1].")
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
        classified = prediction > threshold if strict_threshold else prediction >= threshold
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
        if not np.isfinite(soft).all():
            raise ValueError("Native outcome targets must be finite.")
        clipped = np.clip(prediction, 1e-12, 1 - 1e-12)
        result.update({
            "xt_soft_bce": float(np.mean(-soft * np.log(clipped) - (1 - soft) * np.log1p(-clipped))),
            "xt_rmse": float(np.sqrt(np.mean((soft - prediction) ** 2))),
            "xt_spearman": float(pd.Series(soft).corr(pd.Series(prediction), method="spearman")),
            "xt_mean_bias": float(np.mean(prediction - soft)),
        })
    return result


def paired_bootstrap(task: str, frames: list[pd.DataFrame], *, resamples: int, seed: int,
                     threshold: float | None = None, strict_threshold: bool = False) -> tuple[list[dict], dict, dict]:
    """Return point metrics and intervals using identical match draws for every variant."""
    if resamples < 0 or seed < 0:
        raise ValueError("Bootstrap resamples and seed must be nonnegative.")
    if not frames or any(frame.empty for frame in frames):
        raise ValueError("Comparison requires nonempty aligned prediction frames.")
    reference_rows = frames[0][["match_id", "source_index"]].reset_index(drop=True)
    if any(not frame[["match_id", "source_index"]].reset_index(drop=True).equals(reference_rows)
           for frame in frames[1:]):
        raise ValueError("Paired bootstrap requires aligned row identities.")
    groups = frames[0].groupby("match_id", sort=True).indices
    match_ids = sorted(groups)
    if resamples and len(match_ids) < 2:
        raise ValueError("Paired bootstrap requires at least two contributing test matches.")
    points = [metrics(task, frame, threshold=threshold, strict_threshold=strict_threshold) for frame in frames]
    names = list(points[0])
    draws = {(index, name): [] for index in range(len(frames)) for name in names}
    differences = {(a, b, name): [] for a in range(len(frames))
                   for b in range(a + 1, len(frames)) for name in names}
    rng = np.random.default_rng(seed)
    last_report = time.perf_counter()
    for repeat in range(resamples):
        sampled = rng.integers(0, len(match_ids), size=len(match_ids))
        indices = np.concatenate([groups[match_ids[index]] for index in sampled])
        batch = [metrics(task, frame, indices, threshold=threshold, strict_threshold=strict_threshold) for frame in frames]
        for name in names:
            for index, item in enumerate(batch):
                draws[index, name].append(item[name])
            for a, b, metric in differences:
                if metric == name:
                    differences[a, b, name].append(batch[b][name] - batch[a][name])
        now = time.perf_counter()
        if now - last_report >= 30 or repeat + 1 == resamples:
            print(f"{task} paired bootstrap: {repeat + 1}/{resamples}", flush=True)
            last_report = now

    def interval(values, point):
        if resamples == 0:
            return math.nan, math.nan, 0, "disabled"
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not math.isfinite(point):
            return math.nan, math.nan, len(finite), "undefined_point_estimate"
        if len(finite) < math.ceil(0.95 * resamples) or not len(finite):
            return math.nan, math.nan, len(finite), "insufficient_valid_resamples"
        low, high = np.quantile(finite, [0.025, 0.975])
        return float(low), float(high), len(finite), "ok"

    intervals = {key: interval(values, points[key[0]][key[1]]) for key, values in draws.items()}
    pair_intervals = {key: interval(values, points[key[1]][key[2]] - points[key[0]][key[2]])
                      for key, values in differences.items()}
    return points, intervals, pair_intervals


