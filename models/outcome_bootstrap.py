"""Conditional-on-model uncertainty from resampling whole evaluation matches."""

import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, roc_auc_score

from models.utils import calc_binary_calibration_metrics, calc_continuous_target_metrics


CONTINUOUS = ("mae", "rmse", "pearson_r", "spearman_rho", "mean_prediction_minus_target",
              "calibration_intercept", "calibration_slope")
BINARY = ("roc_auc", "brier", "calibration_intercept", "calibration_slope")
TARGET_METRICS = {"xt_training_target": CONTINUOUS, "goal_next10_diagnostic": BINARY}


def nonnegative_integer(value):
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return result


def add_outcome_bootstrap_arguments(parser):
    parser.add_argument("--outcome-bootstrap-resamples", type=nonnegative_integer, default=2000)
    parser.add_argument("--outcome-bootstrap-seed", type=nonnegative_integer, default=42)


def selected_metrics(prediction, target, diagnostic):
    continuous = calc_continuous_target_metrics(target, prediction, include_soft_bce=False)
    binary_y = (diagnostic > 0).astype(int)
    binary = {
        "roc_auc": roc_auc_score(binary_y, prediction) if np.unique(binary_y).size == 2 else np.nan,
        "brier": brier_score_loss(binary_y, prediction),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            calibration = calc_binary_calibration_metrics(binary_y, prediction, include_ece=False)
    except (ConvergenceWarning, ValueError, FloatingPointError, np.linalg.LinAlgError):
        calibration = {"calibration_intercept": np.nan, "calibration_slope": np.nan}
    binary.update(calibration)
    return {
        "xt_training_target": {key: continuous[key] for key in CONTINUOUS},
        "goal_next10_diagnostic": binary,
    }


def outcome_bootstrap(evaluation, *, model_id, task, resamples=2000, seed=42):
    """Return a long-form CI table and JSON-compatible metadata; no model inference."""
    if not isinstance(resamples, (int, np.integer)) or resamples < 0:
        raise ValueError("resamples must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    metadata = dict(enabled=bool(resamples), method="match_percentile", confidence_level=0.95,
                    quantile_method="linear", stratum="pooled_factual", resamples=int(resamples),
                    seed=int(seed), minimum_valid_fraction=0.95)
    if not resamples:
        return pd.DataFrame(), metadata
    arrays = {key: np.asarray(evaluation[key]).reshape(-1)
              for key in ("prediction", "target", "diagnostic", "execution_branch")}
    n = len(arrays["prediction"])
    ids = np.asarray(evaluation.get("match_id", []), dtype=object).reshape(-1)
    if not n or len(ids) != n or any(len(value) != n for value in arrays.values()):
        raise ValueError("Bootstrap requires aligned, non-empty outcome arrays and match IDs.")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Bootstrap match IDs must be non-empty strings.")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Bootstrap outcome arrays must be finite.")
    match_ids = sorted(set(ids))
    groups = [np.flatnonzero(ids == match_id) for match_id in match_ids]
    metadata.update(match_ids=match_ids, match_count=len(groups))
    prediction, target, diagnostic = (arrays[key].astype(float) for key in ("prediction", "target", "diagnostic"))
    point = selected_metrics(prediction, target, diagnostic)
    values = {(target_name, metric): [] for target_name, metrics in TARGET_METRICS.items() for metric in metrics}
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    last_report = started
    if len(groups) >= 2:
        for repeat in range(resamples):
            chosen = rng.integers(0, len(groups), size=len(groups))
            indices = np.concatenate([groups[index] for index in chosen])
            metrics = selected_metrics(prediction[indices], target[indices], diagnostic[indices])
            for key, estimates in values.items():
                value = metrics[key[0]][key[1]]
                if np.isfinite(value):
                    estimates.append(float(value))
            now = time.perf_counter()
            if now - last_report >= 30 or repeat + 1 == resamples:
                print(f"Outcome bootstrap: {repeat + 1}/{resamples} resamples in {now - started:.1f}s", flush=True)
                last_report = now
    rows = []
    validity = {}
    for (target_name, metric), estimates in values.items():
        estimate = point[target_name][metric]
        status = ("insufficient_matches" if len(groups) < 2 else
                  "undefined_point_estimate" if not np.isfinite(estimate) else
                  "insufficient_valid_resamples" if len(estimates) < math.ceil(0.95 * resamples) else "ok")
        lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear") if status == "ok" else (np.nan, np.nan)
        rows.append(dict(model_id=model_id, task=task, evaluation_target=target_name, stratum="pooled_factual",
                         metric=metric, point_estimate=estimate, lower=lower, upper=upper, confidence_level=0.95,
                         match_count=len(groups), requested_resamples=resamples, valid_resamples=len(estimates),
                         seed=seed, status=status))
        validity.setdefault(target_name, {})[metric] = dict(valid_resamples=len(estimates), status=status)
    metadata["metrics"] = validity
    metadata["elapsed_seconds"] = time.perf_counter() - started
    unavailable = [f"{row['evaluation_target']}/{row['metric']}: {row['status']}" for row in rows if row["status"] != "ok"]
    if unavailable:
        warnings.warn("Unavailable outcome bootstrap intervals: " + "; ".join(unavailable), RuntimeWarning)
    return pd.DataFrame(rows), metadata
