from __future__ import annotations

import hashlib
import math
import json
import os
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from datatools import config
from datatools.config import FIELD_SIZE, LABEL_INDEX
from datatools.utils import filter_features_and_labels
from project_config import get_physical_xpass_match_path

PHYSICAL_XPASS_SOURCE = "accessible_space_max_player_cum_prob_as_defaults"
PHYSICAL_XPASS_LEGACY_SOURCE = "accessible_space_player_cum_prob"
PHYSICAL_XPASS_SOURCES = {PHYSICAL_XPASS_SOURCE, PHYSICAL_XPASS_LEGACY_SOURCE}
PHYSICAL_XPASS_NEUTRAL_PROB = 0.5
PHYSICAL_XPASS_LOGIT_ATTR = "physical_xpass_logit"
PHYSICAL_XPASS_PROB_ATTR = "physical_xpass"
PHYSICAL_XPASS_DISTANCE_ATTR = "physical_xpass_pass_distance"
DEFAULT_RESIDUAL_DISTANCE_THRESHOLD = 30.0
PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE = "ignore_teammates"
PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER = "consider_teammates"
PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX = "package_max"
PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED = "exact_separate_speed"
PHYSICAL_XPASS_SPEED_AGGREGATIONS = {
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
}
PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION = PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX
PHYSICAL_MODEL_VARIANTS = {
    "gat_baseline",
    "gat_plus_phys_feature",
    "gat_phys_logit_offset",
    "gat_phys_logit_offset_regularized",
}
PHYSICAL_XPASS_VARIANTS = {
    "gat_plus_phys_feature",
    "gat_phys_logit_offset",
    "gat_phys_logit_offset_regularized",
}
PHYSICAL_XPASS_PASS_DISTANCE_COLUMN = "pass_distance"
PHYSICAL_XPASS_METRIC_NOISE_KERNEL = "noise_kernel_xpass"
PHYSICAL_XPASS_METRIC_MAX = "max_xpass"
PHYSICAL_XPASS_METRIC_TOP10MEAN = "top10mean_xpass"
PHYSICAL_XPASS_DEFAULT_METRIC = PHYSICAL_XPASS_METRIC_NOISE_KERNEL
PHYSICAL_XPASS_AVAILABLE_METRICS = [
    PHYSICAL_XPASS_METRIC_NOISE_KERNEL,
    PHYSICAL_XPASS_METRIC_MAX,
    PHYSICAL_XPASS_METRIC_TOP10MEAN,
]
PHYSICAL_XPASS_METRIC_SCHEMA_VERSION = 2
PHYSICAL_XPASS_METRIC_SUFFIXES = {
    PHYSICAL_XPASS_METRIC_NOISE_KERNEL: "",
    PHYSICAL_XPASS_METRIC_MAX: "__max_xpass",
    PHYSICAL_XPASS_METRIC_TOP10MEAN: "__top10mean_xpass",
}
PHYSICAL_XPASS_ID_COLUMNS = {
    "match_id",
    "action_index",
    "action_id",
    "physical_state_hash",
    PHYSICAL_XPASS_PASS_DISTANCE_COLUMN,
}
DEFAULT_V0_MIN = 8.886015553615485
DEFAULT_V0_MAX = 42.18118275402132
DEFAULT_N_V0 = 14
AS_DEFAULT_N_ANGLES = 30
AS_DEFAULT_PHI_OFFSET = 0.0
AS_DEFAULT_N_V0 = 20
AS_DEFAULT_V0_MIN = 3.0
AS_DEFAULT_V0_MAX = 22.0
AS_DEFAULT_SPEED_STEP = 1.0
AS_DEFAULT_COARSE_N_ANGLES = 36
AS_DEFAULT_REFINE_TOP_K_ANGLES = 2
AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG = 10.0
AS_DEFAULT_ANGLE_STEP_DEG = 2.5
AS_DEFAULT_PASS_START_LOCATION_OFFSET = 0.0
AS_DEFAULT_TIME_OFFSET_BALL = 0.0
AS_DEFAULT_RADIAL_GRIDSIZE = 3.0
AS_DEFAULT_B0 = -4.565680899844368
AS_DEFAULT_B1 = -2000.0
AS_DEFAULT_PLAYER_VELOCITY = 9.0
AS_DEFAULT_KEEP_INERTIAL_VELOCITY = True
AS_DEFAULT_USE_MAX = False
AS_DEFAULT_V_MAX = 19.85563874348074
AS_DEFAULT_A_MAX = 10.659091365334193
AS_DEFAULT_INERTIAL_SECONDS = 0.17
AS_DEFAULT_TOL_DISTANCE = 5.0
AS_DEFAULT_USE_APPROX_TWO_POINT = True
AS_DEFAULT_V0_PROB_AGGREGATION_MODE = "mean"
AS_DEFAULT_NORMALIZE = True
AS_DEFAULT_USE_EFFICIENT_SIGMOID = True
AS_DEFAULT_FACTOR = 5.077423030272923
AS_DEFAULT_FACTOR2 = 1.0063028450754512
AS_DEFAULT_RESPECT_OFFSIDE = True
AS_DEFAULT_EXCLUDE_PASSER = False


def physical_xpass_as_default_metadata(
    teammate_policy: str = PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    *,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    metric_schema_version: int = PHYSICAL_XPASS_METRIC_SCHEMA_VERSION,
    default_metric: str = PHYSICAL_XPASS_DEFAULT_METRIC,
    available_metrics: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if teammate_policy not in {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE, PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER}:
        raise ValueError(
            f"Unsupported teammate_policy={teammate_policy!r}. "
            f"Expected {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE!r} or {PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER!r}."
        )
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    metric_schema_version = int(metric_schema_version)
    if available_metrics is None:
        available_metrics = list(PHYSICAL_XPASS_AVAILABLE_METRICS)
    default_metric = normalize_physical_xpass_metric(default_metric)
    available_metrics = [normalize_physical_xpass_metric(metric) for metric in available_metrics]
    v0_values = as_default_v0_values(max_speed=max_speed, speed_step=speed_step)
    return {
        "metric": default_metric if metric_schema_version >= PHYSICAL_XPASS_METRIC_SCHEMA_VERSION else "max_player_cum_prob",
        "source": PHYSICAL_XPASS_SOURCE,
        "teammate_policy": teammate_policy,
        "speed_aggregation": speed_aggregation,
        "metric_schema_version": metric_schema_version,
        "default_metric": default_metric,
        "available_metrics": available_metrics,
        "max_speed": float(v0_values[-1]),
        "speed_step": float(speed_step if speed_step is not None else AS_DEFAULT_SPEED_STEP),
        "coordinate_system": "centered_pitch",
        "n_angles": int(coarse_n_angles),
        "coarse_n_angles": int(coarse_n_angles),
        "refine_top_k_angles": int(refine_top_k_angles),
        "refine_angle_radius_deg": float(refine_angle_radius),
        "angle_step_deg": float(angle_step),
        "phi_offset": AS_DEFAULT_PHI_OFFSET,
        "n_v0": int(v0_values.shape[0]),
        "v0_min": AS_DEFAULT_V0_MIN,
        "v0_max": float(v0_values[-1]),
        "pass_start_location_offset": AS_DEFAULT_PASS_START_LOCATION_OFFSET,
        "time_offset_ball": AS_DEFAULT_TIME_OFFSET_BALL,
        "radial_gridsize": AS_DEFAULT_RADIAL_GRIDSIZE,
        "b0": AS_DEFAULT_B0,
        "b1": AS_DEFAULT_B1,
        "player_velocity": AS_DEFAULT_PLAYER_VELOCITY,
        "keep_inertial_velocity": AS_DEFAULT_KEEP_INERTIAL_VELOCITY,
        "use_max": AS_DEFAULT_USE_MAX,
        "v_max": AS_DEFAULT_V_MAX,
        "a_max": AS_DEFAULT_A_MAX,
        "inertial_seconds": AS_DEFAULT_INERTIAL_SECONDS,
        "tol_distance": AS_DEFAULT_TOL_DISTANCE,
        "use_approx_two_point": AS_DEFAULT_USE_APPROX_TWO_POINT,
        "v0_prob_aggregation_mode": (
            "max"
            if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX
            else AS_DEFAULT_V0_PROB_AGGREGATION_MODE
        ),
        "normalize": AS_DEFAULT_NORMALIZE,
        "use_efficient_sigmoid": AS_DEFAULT_USE_EFFICIENT_SIGMOID,
        "factor": AS_DEFAULT_FACTOR,
        "factor2": AS_DEFAULT_FACTOR2,
        "respect_offside": AS_DEFAULT_RESPECT_OFFSIDE,
        "exclude_passer": teammate_policy == PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    }


def as_default_v0_values(
    max_speed: float | None = None,
    *,
    speed_step: float | None = None,
    min_speed: float = AS_DEFAULT_V0_MIN,
) -> np.ndarray:
    max_speed = AS_DEFAULT_V0_MAX if max_speed is None else float(max_speed)
    speed_step = AS_DEFAULT_SPEED_STEP if speed_step is None else float(speed_step)
    if speed_step <= 0:
        raise ValueError("--speed-step must be positive.")
    values = np.arange(float(min_speed), max_speed + (speed_step * 0.5), speed_step, dtype=float)
    values = values[values <= max_speed + 1e-9]
    if values.size == 0:
        raise ValueError(f"--max-speed must be at least {float(min_speed):g} m/s.")
    return values


def normalize_physical_xpass_metric(value: str | None) -> str:
    metric = PHYSICAL_XPASS_DEFAULT_METRIC if value is None else str(value).replace("-", "_")
    if metric == "top10mean":
        metric = PHYSICAL_XPASS_METRIC_TOP10MEAN
    if metric not in PHYSICAL_XPASS_AVAILABLE_METRICS:
        raise ValueError(
            f"Unsupported physical_xpass_metric={value!r}. "
            f"Expected one of {PHYSICAL_XPASS_AVAILABLE_METRICS}."
        )
    return metric


def physical_xpass_metric_column(player_id: str, metric: str | None = None) -> str:
    metric = normalize_physical_xpass_metric(metric)
    return f"{player_id}{PHYSICAL_XPASS_METRIC_SUFFIXES[metric]}"


def normalize_physical_xpass_speed_aggregation(value: str | None) -> str:
    if value is None:
        return PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED
    value = str(value)
    if value not in PHYSICAL_XPASS_SPEED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported physical_xpass_speed_aggregation={value!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SPEED_AGGREGATIONS)}."
        )
    return value


def validate_physical_xpass_cache_metadata(
    cache_dir: str | Path,
    *,
    expected_source: str | None = None,
    expected_speed_aggregation: str | None = None,
    expected_metric_schema_version: int | None = None,
    expected_default_metric: str | None = None,
    expected_max_speed: float | None = None,
    expected_speed_step: float | None = None,
    expected_coarse_n_angles: int | None = None,
    expected_refine_top_k_angles: int | None = None,
    expected_refine_angle_radius: float | None = None,
    expected_angle_step: float | None = None,
) -> dict[str, Any]:
    metadata_path = Path(cache_dir) / "metadata.json"
    expected_source = expected_source or PHYSICAL_XPASS_SOURCE
    if expected_source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={expected_source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    rerun_message = "Run scripts/generate_physical_xpass.py to regenerate compatible physical xPass sidecars/runtime caches."
    if not metadata_path.exists():
        raise FileNotFoundError(f"Physical xPass metadata not found at {metadata_path}. {rerun_message}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    source = metadata.get("source")
    if source != expected_source:
        raise ValueError(
            f"Physical xPass sidecars at {cache_dir} use incompatible source {source!r}; "
            f"expected {expected_source!r}. {rerun_message}"
        )
    if expected_speed_aggregation is not None and source == PHYSICAL_XPASS_SOURCE:
        actual_speed_aggregation = normalize_physical_xpass_speed_aggregation(metadata.get("speed_aggregation"))
        expected_speed_aggregation = normalize_physical_xpass_speed_aggregation(expected_speed_aggregation)
        if actual_speed_aggregation != expected_speed_aggregation:
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} use incompatible speed_aggregation "
                f"{actual_speed_aggregation!r}; expected {expected_speed_aggregation!r}. {rerun_message}"
            )
    if expected_metric_schema_version is not None and source == PHYSICAL_XPASS_SOURCE:
        actual_schema = metadata.get("metric_schema_version")
        if actual_schema is None or int(actual_schema) != int(expected_metric_schema_version):
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} use incompatible metric schema "
                f"{actual_schema!r}; expected {int(expected_metric_schema_version)!r}. {rerun_message}"
            )
    if expected_default_metric is not None and source == PHYSICAL_XPASS_SOURCE:
        actual_metric = normalize_physical_xpass_metric(metadata.get("default_metric"))
        expected_metric = normalize_physical_xpass_metric(expected_default_metric)
        if actual_metric != expected_metric:
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} use incompatible default_metric "
                f"{actual_metric!r}; expected {expected_metric!r}. {rerun_message}"
            )
    expected_numeric = {
        "max_speed": expected_max_speed,
        "speed_step": expected_speed_step,
        "coarse_n_angles": expected_coarse_n_angles,
        "refine_top_k_angles": expected_refine_top_k_angles,
        "refine_angle_radius_deg": expected_refine_angle_radius,
        "angle_step_deg": expected_angle_step,
    }
    for key, expected_value in expected_numeric.items():
        if expected_value is None or source != PHYSICAL_XPASS_SOURCE:
            continue
        actual_value = metadata.get(key)
        if actual_value is None:
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} are missing {key}; "
                f"expected {expected_value!r}. {rerun_message}"
            )
        if isinstance(expected_value, int):
            matches = int(actual_value) == int(expected_value)
        else:
            matches = abs(float(actual_value) - float(expected_value)) <= 1e-9
        if not matches:
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} use incompatible {key} "
                f"{actual_value!r}; expected {expected_value!r}. {rerun_message}"
            )
    return metadata


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def physical_xpass_enabled(args: Any) -> bool:
    return bool(_get_arg(args, "use_physical_xpass", False))


def physical_xpass_model_variant(args: Any) -> str:
    variant = str(_get_arg(args, "model_variant", "gat_baseline") or "gat_baseline")
    if variant not in PHYSICAL_MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant={variant!r}. Expected one of {sorted(PHYSICAL_MODEL_VARIANTS)}.")
    return variant


def physical_xpass_source(args: Any) -> str:
    source = _get_arg(args, "physical_xpass_source", None) or PHYSICAL_XPASS_SOURCE
    source = str(source)
    if source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    return source


def physical_xpass_teammate_policy(args: Any, *, source: str | None = None) -> str:
    policy = _get_arg(args, "physical_xpass_teammate_policy", None)
    if policy is None:
        resolved_source = source or physical_xpass_source(args)
        return (
            PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
            if resolved_source == PHYSICAL_XPASS_LEGACY_SOURCE
            else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
        )
    policy = str(policy)
    if policy not in {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE, PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER}:
        raise ValueError(
            f"Unsupported physical_xpass_teammate_policy={policy!r}. "
            f"Expected {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE!r} or {PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER!r}."
        )
    return policy


def physical_xpass_speed_aggregation(args: Any) -> str:
    value = _get_arg(args, "physical_xpass_speed_aggregation", None)
    if value is None:
        value = _get_arg(args, "speed_aggregation", None)
    return normalize_physical_xpass_speed_aggregation(value)


def runtime_physical_xpass_speed_aggregation(args: Any) -> str:
    value = _get_arg(args, "physical_xpass_speed_aggregation", None)
    if value is None:
        value = _get_arg(args, "speed_aggregation", None)
    if value is not None:
        return normalize_physical_xpass_speed_aggregation(value)
    if inference_uses_physical_xpass(args) and not model_uses_physical_xpass(args):
        return PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION
    return normalize_physical_xpass_speed_aggregation(None)


def physical_xpass_metric(args: Any) -> str:
    if bool(_get_arg(args, "max_xpass", False)) or bool(_get_arg(args, "use_max_xpass", False)):
        return PHYSICAL_XPASS_METRIC_MAX
    if bool(_get_arg(args, "top10mean_xpass", False)) or bool(_get_arg(args, "use_top10mean_xpass", False)):
        return PHYSICAL_XPASS_METRIC_TOP10MEAN
    return normalize_physical_xpass_metric(_get_arg(args, "physical_xpass_metric", None))


def model_uses_physical_xpass(args: Any) -> bool:
    task = _get_arg(args, "task", None)
    return task == "pass_success" and physical_xpass_enabled(args) and physical_xpass_model_variant(args) in PHYSICAL_XPASS_VARIANTS


def inference_uses_physical_xpass(args: Any) -> bool:
    task = _get_arg(args, "task", None)
    return task == "pass_success" and bool(
        _get_arg(args, "inference_use_physical_xpass", False)
        or _get_arg(args, "use_physical_xpass_at_inference", False)
        or _get_arg(args, "blend_physical_xpass", False)
    )


def requires_physical_xpass_for_inference(args: Any) -> bool:
    return model_uses_physical_xpass(args) or inference_uses_physical_xpass(args)


def residual_distance_threshold(args: Any) -> float:
    return float(_get_arg(args, "residual_distance_threshold", DEFAULT_RESIDUAL_DISTANCE_THRESHOLD))


def physical_xpass_floor(args: Any) -> float | None:
    value = _get_arg(args, "physical_xpass_floor", None)
    return None if value is None else float(value)


def resolved_residual_regularization_lambdas(args: Any) -> tuple[float, float]:
    base = float(_get_arg(args, "residual_regularization_lambda", 0.0) or 0.0)
    short_value = _get_arg(args, "short_residual_regularization_lambda", None)
    long_value = _get_arg(args, "long_residual_regularization_lambda", None)
    short_lambda = base if short_value is None else float(short_value)
    long_lambda = base if long_value is None else float(long_value)
    return short_lambda, long_lambda


def resolved_residual_clip_values(args: Any) -> tuple[float | None, float | None]:
    base = _get_arg(args, "residual_clip_value", None)
    short_value = _get_arg(args, "short_residual_clip_value", None)
    long_value = _get_arg(args, "long_residual_clip_value", None)
    short_clip = base if short_value is None else short_value
    long_clip = base if long_value is None else long_value
    return (
        None if short_clip is None else float(short_clip),
        None if long_clip is None else float(long_clip),
    )


def validate_physical_xpass_args(args: Any) -> None:
    variant = physical_xpass_model_variant(args)
    eps = float(_get_arg(args, "physical_eps", 1e-4))
    floor = physical_xpass_floor(args)
    residual_clip_value = _get_arg(args, "residual_clip_value", None)
    residual_regularization_lambda = float(_get_arg(args, "residual_regularization_lambda", 0.0) or 0.0)
    distance_threshold = residual_distance_threshold(args)

    if not (0.0 < eps < 0.5):
        raise ValueError(f"--physical-eps must be between 0 and 0.5, got {eps}.")
    if floor is not None and not (0.0 <= floor < 1.0):
        raise ValueError("--physical-xpass-floor must be in [0.0, 1.0) when provided.")
    if distance_threshold <= 0:
        raise ValueError("--residual-distance-threshold must be positive.")
    if residual_regularization_lambda < 0:
        raise ValueError("--residual-regularization-lambda must be non-negative.")
    for name in ("short_residual_regularization_lambda", "long_residual_regularization_lambda"):
        value = _get_arg(args, name, None)
        if value is not None and float(value) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if residual_clip_value is not None and float(residual_clip_value) <= 0:
        raise ValueError("--residual-clip-value must be positive when provided.")
    for name in ("short_residual_clip_value", "long_residual_clip_value"):
        value = _get_arg(args, name, None)
        if value is not None and float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided.")
    if _get_arg(args, "physical_xpass_speed_aggregation", None) is not None or _get_arg(args, "speed_aggregation", None) is not None:
        physical_xpass_speed_aggregation(args)
    if physical_xpass_enabled(args) and _get_arg(args, "task", None) != "pass_success":
        raise ValueError("--use_physical_xpass is only supported for task=pass_success.")
    if physical_xpass_enabled(args) and variant == "gat_baseline":
        raise ValueError("--use_physical_xpass requires a physical model variant, not gat_baseline.")
    if model_uses_physical_xpass(args) and bool(_get_arg(args, "include_out", False)):
        raise ValueError("--include_out is not supported with physical xPass because player_cum_prob has no out node.")


def probability_to_logit(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    prob = torch.as_tensor(prob, dtype=torch.float32)
    prob = torch.clamp(prob, float(eps), 1.0 - float(eps))
    return torch.log(prob / (1.0 - prob))


def probability_to_logit_numpy(prob: np.ndarray | pd.Series | list[float], eps: float = 1e-4) -> np.ndarray:
    values = np.asarray(prob, dtype=float)
    values = np.clip(values, float(eps), 1.0 - float(eps))
    return np.log(values / (1.0 - values))


def physical_xpass_blend_weight(pass_distance: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
    if isinstance(pass_distance, torch.Tensor):
        return torch.clamp(pass_distance.to(dtype=torch.float32) / 100.0, 0.0, 1.0)
    values = np.asarray(pass_distance, dtype=float)
    weights = np.clip(values / 100.0, 0.0, 1.0)
    if np.isscalar(pass_distance):
        return float(weights)
    return weights


def blend_physical_xpass_predictions(
    *,
    pass_success_model: np.ndarray | torch.Tensor | float,
    xpass: np.ndarray | torch.Tensor | float,
    pass_distance: np.ndarray | torch.Tensor | float,
) -> np.ndarray | torch.Tensor | float:
    if isinstance(pass_success_model, torch.Tensor) or isinstance(xpass, torch.Tensor) or isinstance(pass_distance, torch.Tensor):
        model_tensor = torch.as_tensor(pass_success_model, dtype=torch.float32)
        xpass_tensor = torch.as_tensor(xpass, dtype=torch.float32, device=model_tensor.device)
        distance_tensor = torch.as_tensor(pass_distance, dtype=torch.float32, device=model_tensor.device)
        weight = physical_xpass_blend_weight(distance_tensor)
        return torch.clamp((1.0 - weight) * xpass_tensor + weight * model_tensor, 0.0, 1.0)
    model_values = np.asarray(pass_success_model, dtype=float)
    xpass_values = np.asarray(xpass, dtype=float)
    distance_values = np.asarray(pass_distance, dtype=float)
    weight = physical_xpass_blend_weight(distance_values)
    blended = np.clip((1.0 - weight) * xpass_values + weight * model_values, 0.0, 1.0)
    if np.isscalar(pass_success_model) and np.isscalar(xpass) and np.isscalar(pass_distance):
        return float(blended)
    return blended


def _physical_xpass_lower_bound(eps: float, floor: float | None = None) -> float:
    if floor is not None and not (0.0 <= float(floor) < 1.0):
        raise ValueError("--physical-xpass-floor must be in [0.0, 1.0) when provided.")
    lower_bound = max(float(eps), float(floor)) if floor is not None else float(eps)
    return min(lower_bound, 1.0 - float(eps))


def _node_ids(graph: Data) -> list[str]:
    node_ids = getattr(graph, "node_ids", None)
    if node_ids is None:
        raise ValueError("Graph is missing node_ids; physical xPass requires player-id aligned graph nodes.")
    node_ids = [str(node_id) for node_id in node_ids]
    if len(node_ids) != int(graph.x.shape[0]):
        raise ValueError(f"Graph node_ids length {len(node_ids)} does not match node count {int(graph.x.shape[0])}.")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Graph node_ids contain duplicates; physical xPass alignment would be ambiguous.")
    return node_ids


def _candidate_target_indices(graph: Data) -> list[int]:
    teammate = graph.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
    possessor = graph.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
    goal = (
        graph.x[:, config.NODE_FEATURE_IS_GOAL] == 1
        if graph.x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.zeros(graph.x.shape[0], dtype=torch.bool)
    )
    finite_xy = torch.isfinite(graph.x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1]).all(dim=1)
    mask = teammate & (~possessor) & (~goal) & finite_xy
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def physical_state_hash(graph: Data) -> str:
    node_ids = _node_ids(graph)
    x = graph.x.detach().cpu().to(torch.float32)
    feature_indices = [
        config.NODE_FEATURE_IS_TEAMMATE,
        config.NODE_FEATURE_X,
        config.NODE_FEATURE_Y,
        config.NODE_FEATURE_VX,
        config.NODE_FEATURE_VY,
        config.NODE_FEATURE_IS_GOAL,
        config.NODE_FEATURE_IS_POSSESSOR,
    ]
    rows: list[dict[str, Any]] = []
    for node_index, node_id in enumerate(node_ids):
        values: dict[str, Any] = {"node_id": node_id}
        for feature_index in feature_indices:
            if feature_index >= x.shape[1]:
                values[str(feature_index)] = None
                continue
            value = float(x[node_index, feature_index].item())
            values[str(feature_index)] = None if not math.isfinite(value) else round(value, 6)
        rows.append(values)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_pass_distances(graph: Data) -> torch.Tensor:
    x = graph.x.detach().cpu().to(torch.float32)
    if x.shape[1] <= config.NODE_FEATURE_Y:
        raise ValueError("Graph node features do not include x/y coordinates for pass-distance calculation.")
    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node for pass-distance calculation, found {len(possessor_indices)}.")
    possessor_xy = x[int(possessor_indices[0]), config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1]
    node_xy = x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1]
    finite = torch.isfinite(node_xy).all(dim=1) & torch.isfinite(possessor_xy).all()
    distances = torch.linalg.norm(node_xy - possessor_xy.unsqueeze(0), dim=1)
    distances = torch.where(finite, distances, torch.full_like(distances, float("nan")))
    return distances


def observed_pass_distance(graph: Data, labels: torch.Tensor) -> float:
    target_index = int(labels[LABEL_INDEX["intent_index"]].item())
    distances = graph_pass_distances(graph)
    if target_index < 0 or target_index >= int(distances.shape[0]):
        return float("nan")
    return float(distances[target_index].item())


def prepare_runtime_physical_xpass_prewarm_items(
    runtime_objects: list[Any],
    model: Any,
    *,
    match_id_getter: Callable[[Any], str] | None = None,
) -> list[dict[str, Any]]:
    """Prepare runtime cache prewarm inputs exactly as pass-success inference sees them."""
    if model is None or not requires_physical_xpass_for_inference(model.args):
        return []
    match_id_getter = match_id_getter or (lambda runtime_object: str(runtime_object.match_id))
    items: list[dict[str, Any]] = []
    for runtime_object in runtime_objects:
        graphs = list(getattr(runtime_object, "graph_features_0", None) or [])
        labels = getattr(runtime_object, "labels", None)
        if labels is None or len(graphs) == 0:
            continue
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.float32)
        if labels.numel() == 0:
            continue
        filtered_graphs, filtered_labels = filter_features_and_labels(
            graphs,
            labels,
            model.args,
        )
        if filtered_labels.numel() == 0 or not filtered_graphs:
            continue
        items.append(
            {
                "match_id": str(match_id_getter(runtime_object)),
                "graphs": filtered_graphs,
                "labels": filtered_labels,
            }
        )
    return items


def _extract_simulation_probability(
    simulation_result: Any,
    *,
    frame_index: int,
    target_player_index: int,
    angle_index: int,
    target_distance: float,
) -> tuple[float, dict[str, Any]]:
    player_cum_prob = getattr(simulation_result, "player_cum_prob", None)
    if player_cum_prob is None:
        raise ValueError("accessible-space simulation_result did not return player_cum_prob.")

    arr = np.asarray(player_cum_prob, dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"player_cum_prob must be 4D [frame, player, angle, distance], got shape={arr.shape}.")
    if frame_index < 0 or frame_index >= arr.shape[0]:
        raise ValueError(f"frame_index={frame_index} out of bounds for player_cum_prob shape={arr.shape}.")
    if target_player_index < 0 or target_player_index >= arr.shape[1]:
        raise ValueError(f"target_player_index={target_player_index} out of bounds for player_cum_prob shape={arr.shape}.")
    if angle_index < 0 or angle_index >= arr.shape[2]:
        raise ValueError(f"angle_index={angle_index} out of bounds for player_cum_prob shape={arr.shape}.")

    r_grid = np.asarray(getattr(simulation_result, "r_grid", None), dtype=float)
    if r_grid.ndim != 1 or r_grid.shape[0] != arr.shape[3]:
        raise ValueError(
            f"simulation_result.r_grid must be 1D with length {arr.shape[3]}, got shape={r_grid.shape}."
        )
    distance_index = int(np.nanargmin(np.abs(r_grid - float(target_distance))))
    value = float(arr[frame_index, target_player_index, angle_index, distance_index])
    if not math.isfinite(value):
        raise ValueError(
            "Extracted non-finite player_cum_prob "
            f"for frame={frame_index}, player={target_player_index}, angle={angle_index}, distance={distance_index}."
        )
    return value, {
        "distance": float(target_distance),
        "distance_index": distance_index,
        "player_index": int(target_player_index),
        "frame_index": int(frame_index),
        "angle_index": int(angle_index),
    }


def _resolve_simulate_passes_fn() -> Callable[..., Any]:
    try:
        import accessible_space  # noqa: F401
        from accessible_space.core import simulate_passes_chunked
    except ImportError as exc:
        raise ImportError(
            "accessible-space is required to generate physical xPass sidecars. "
            "Install project dependencies, then rerun scripts/generate_physical_xpass.py."
        ) from exc
    return simulate_passes_chunked


def _validate_simulation_contract(
    *,
    players: np.ndarray,
    passers: np.ndarray,
    exclude_passer: bool,
) -> None:
    if not exclude_passer:
        return
    missing_passers = [str(passer) for passer in np.asarray(passers, dtype=object).tolist() if passer not in players.tolist()]
    if missing_passers:
        raise ValueError(
            "accessible-space requires every passer id to be present in the simulation players array when "
            f"exclude_passer=True. Missing passers: {missing_passers}."
        )


def _build_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
    consider_teammates: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    x = graph.x.detach().cpu().to(torch.float32)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    if consider_teammates:
        sim_player_indices = player_indices
        player_index_to_sim_index = {node_index: sim_index for sim_index, node_index in enumerate(sim_player_indices)}
        missing_targets = [idx for idx in candidate_indices if idx not in player_index_to_sim_index]
        if missing_targets:
            missing_ids = [str(graph.node_ids[idx]) for idx in missing_targets]
            raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

        player_pos = x[sim_player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy()
        player_teams = np.where(
            (x[sim_player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1),
            "attack",
            "defense",
        ).astype(object)
        players = np.array([str(graph.node_ids[idx]) for idx in sim_player_indices], dtype=object)
        target_index_lookup = {graph_target_index: player_index_to_sim_index[graph_target_index] for graph_target_index in candidate_indices}
        return (
            np.repeat(player_pos[np.newaxis, :, :], len(candidate_indices), axis=0),
            player_teams,
            players,
            x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy(),
            np.repeat("attack", len(candidate_indices)).astype(object),
            target_index_lookup,
        )

    defender_indices = [
        node_index
        for node_index in player_indices
        if node_index != possessor_index and int(x[node_index, config.NODE_FEATURE_IS_TEAMMATE].item()) == 0
    ]
    frame_player_indices = [[possessor_index, graph_target_index] + defender_indices for graph_target_index in candidate_indices]
    player_counts = {len(indices) for indices in frame_player_indices}
    if len(player_counts) != 1:
        raise ValueError(f"Physical xPass reduced simulation player count must be constant across candidates, got {sorted(player_counts)}.")

    player_pos = np.stack(
        [x[indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy() for indices in frame_player_indices],
        axis=0,
    )
    player_teams = np.array(["attack", "attack"] + ["defense"] * len(defender_indices), dtype=object)
    players = np.array(
        [str(graph.node_ids[possessor_index]), "target_player"] + [str(graph.node_ids[idx]) for idx in defender_indices],
        dtype=object,
    )
    target_index_lookup = {graph_target_index: 1 for graph_target_index in candidate_indices}
    return (
        player_pos,
        player_teams,
        players,
        x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy(),
        np.repeat("attack", len(candidate_indices)).astype(object),
        target_index_lookup,
    )


def _infer_playing_direction_from_centered_players(player_pos: np.ndarray, player_teams: np.ndarray) -> float:
    finite_x = np.isfinite(player_pos[:, 0])
    attack_x = player_pos[(player_teams == "attack") & finite_x, 0]
    defense_x = player_pos[(player_teams == "defense") & finite_x, 0]
    if attack_x.size == 0 or defense_x.size == 0:
        return 1.0
    return 1.0 if float(np.nanmean(attack_x)) <= float(np.nanmean(defense_x)) else -1.0


def _full_as_default_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], int]:
    x = graph.x.detach().cpu().to(torch.float32)
    node_ids = _node_ids(graph)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    player_index_to_sim_index = {node_index: sim_index for sim_index, node_index in enumerate(player_indices)}
    missing_targets = [node_index for node_index in candidate_indices if node_index not in player_index_to_sim_index]
    if missing_targets:
        missing_ids = [node_ids[node_index] for node_index in missing_targets]
        raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

    player_pos = x[player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy()
    player_pos = np.asarray(player_pos, dtype=float)
    player_pos[:, 0] -= FIELD_SIZE[0] / 2.0
    player_pos[:, 1] -= FIELD_SIZE[1] / 2.0
    player_teams = np.where((x[player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1), "attack", "defense").astype(object)
    players = np.array([node_ids[node_index] for node_index in player_indices], dtype=object)
    ball_pos = x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy().astype(float)
    ball_pos = np.array([ball_pos[0] - FIELD_SIZE[0] / 2.0, ball_pos[1] - FIELD_SIZE[1] / 2.0], dtype=float)
    target_index_lookup = {node_index: player_index_to_sim_index[node_index] for node_index in candidate_indices}
    return player_pos, player_teams, players, ball_pos, target_index_lookup, possessor_index


def _reduced_as_default_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[int, int]], int, float]:
    x = graph.x.detach().cpu().to(torch.float32)
    node_ids = _node_ids(graph)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    missing_targets = [node_index for node_index in candidate_indices if node_index not in player_indices]
    if missing_targets:
        missing_ids = [node_ids[node_index] for node_index in missing_targets]
        raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

    full_player_pos = x[player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy().astype(float)
    full_player_pos[:, 0] -= FIELD_SIZE[0] / 2.0
    full_player_pos[:, 1] -= FIELD_SIZE[1] / 2.0
    full_player_teams = np.where(
        (x[player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1),
        "attack",
        "defense",
    ).astype(object)
    playing_direction = _infer_playing_direction_from_centered_players(full_player_pos, full_player_teams)

    defender_indices = [
        node_index
        for node_index in player_indices
        if node_index != possessor_index and int(x[node_index, config.NODE_FEATURE_IS_TEAMMATE].item()) == 0
    ]
    frame_player_indices = [[possessor_index, graph_target_index] + defender_indices for graph_target_index in candidate_indices]
    player_counts = {len(indices) for indices in frame_player_indices}
    if len(player_counts) != 1:
        raise ValueError(
            f"Physical xPass reduced simulation player count must be constant across candidates, got {sorted(player_counts)}."
        )

    player_pos = np.stack(
        [x[indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy() for indices in frame_player_indices],
        axis=0,
    ).astype(float)
    player_pos[:, :, 0] -= FIELD_SIZE[0] / 2.0
    player_pos[:, :, 1] -= FIELD_SIZE[1] / 2.0
    player_teams = np.array(["attack", "attack"] + ["defense"] * len(defender_indices), dtype=object)
    players = np.array(
        [node_ids[possessor_index], "target_player"] + [node_ids[node_index] for node_index in defender_indices],
        dtype=object,
    )
    ball_pos = x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy().astype(float)
    ball_pos = np.array([ball_pos[0] - FIELD_SIZE[0] / 2.0, ball_pos[1] - FIELD_SIZE[1] / 2.0], dtype=float)
    target_index_lookup = {node_index: (frame_index, 1) for frame_index, node_index in enumerate(candidate_indices)}
    return player_pos, player_teams, players, ball_pos, target_index_lookup, possessor_index, playing_direction


def _on_pitch_mask(simulation_result: Any, frame_index: int = 0) -> np.ndarray:
    x_grid = getattr(simulation_result, "x_grid", None)
    y_grid = getattr(simulation_result, "y_grid", None)
    player_cum_prob = np.asarray(getattr(simulation_result, "player_cum_prob", None), dtype=float)
    if x_grid is None or y_grid is None:
        return np.ones(player_cum_prob.shape[2:], dtype=bool)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if x_grid.ndim != 3 or y_grid.ndim != 3:
        raise ValueError(f"simulation_result x_grid/y_grid must be 3D [frame, angle, distance], got {x_grid.shape}/{y_grid.shape}.")
    return (
        np.isfinite(x_grid[frame_index])
        & np.isfinite(y_grid[frame_index])
        & (x_grid[frame_index] >= -FIELD_SIZE[0] / 2.0)
        & (x_grid[frame_index] <= FIELD_SIZE[0] / 2.0)
        & (y_grid[frame_index] >= -FIELD_SIZE[1] / 2.0)
        & (y_grid[frame_index] <= FIELD_SIZE[1] / 2.0)
    )


def _update_as_default_maxima_from_simulation_result(
    *,
    maxima: dict[int, float],
    simulation_result: Any,
    candidate_indices: list[int],
    target_index_lookup: dict[int, tuple[int, int]],
    frame_count: int,
    player_count: int,
) -> None:
    player_cum_prob = np.asarray(getattr(simulation_result, "player_cum_prob", None), dtype=float)
    if player_cum_prob.ndim != 4:
        raise ValueError(
            "accessible-space simulation_result.player_cum_prob must be 4D "
            f"[frame, player, angle, distance], got shape={player_cum_prob.shape}."
        )
    if (
        player_cum_prob.shape[0] != frame_count
        or player_cum_prob.shape[1] != player_count
        or player_cum_prob.shape[2] != AS_DEFAULT_N_ANGLES
    ):
        raise ValueError(
            "Unexpected player_cum_prob shape for AS-default physical xPass: "
            f"{player_cum_prob.shape}, frames={frame_count}, players={player_count}, "
            f"n_angles={AS_DEFAULT_N_ANGLES}."
        )
    for graph_target_index in candidate_indices:
        frame_index, target_sim_index = target_index_lookup[graph_target_index]
        on_pitch = _on_pitch_mask(simulation_result, frame_index=frame_index)
        if on_pitch.shape != player_cum_prob.shape[2:]:
            raise ValueError(
                f"On-pitch mask shape {on_pitch.shape} does not match "
                f"player_cum_prob grid {player_cum_prob.shape[2:]}."
            )
        values = np.where(on_pitch, player_cum_prob[frame_index, target_sim_index, :, :], np.nan)
        if np.isfinite(values).any():
            maxima[graph_target_index] = max(maxima[graph_target_index], float(np.nanmax(values)))


def _as_default_simulation_kwargs(
    *,
    PLAYER_POS: np.ndarray,
    BALL_POS: np.ndarray,
    phi_grid: np.ndarray,
    v0_grid: np.ndarray,
    passer_teams: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    passers: np.ndarray,
    exclude_passer: bool,
    playing_direction: np.ndarray,
    use_progress_bar: bool,
    chunk_size: int,
    v0_prob_aggregation_mode: str,
) -> dict[str, Any]:
    return {
        "PLAYER_POS": PLAYER_POS,
        "BALL_POS": BALL_POS,
        "phi_grid": phi_grid,
        "v0_grid": v0_grid,
        "passer_teams": passer_teams,
        "player_teams": player_teams,
        "players": players,
        "passers": passers,
        "exclude_passer": bool(exclude_passer),
        "respect_offside": AS_DEFAULT_RESPECT_OFFSIDE,
        "playing_direction": playing_direction,
        "x_pitch_min": -FIELD_SIZE[0] / 2.0,
        "x_pitch_max": FIELD_SIZE[0] / 2.0,
        "y_pitch_min": -FIELD_SIZE[1] / 2.0,
        "y_pitch_max": FIELD_SIZE[1] / 2.0,
        "use_progress_bar": use_progress_bar,
        "chunk_size": chunk_size,
        "fields_to_return": ("player_cum_prob",),
        "normalize": AS_DEFAULT_NORMALIZE,
        "pass_start_location_offset": AS_DEFAULT_PASS_START_LOCATION_OFFSET,
        "time_offset_ball": AS_DEFAULT_TIME_OFFSET_BALL,
        "radial_gridsize": AS_DEFAULT_RADIAL_GRIDSIZE,
        "b0": AS_DEFAULT_B0,
        "b1": AS_DEFAULT_B1,
        "player_velocity": AS_DEFAULT_PLAYER_VELOCITY,
        "keep_inertial_velocity": AS_DEFAULT_KEEP_INERTIAL_VELOCITY,
        "use_max": AS_DEFAULT_USE_MAX,
        "v_max": AS_DEFAULT_V_MAX,
        "a_max": AS_DEFAULT_A_MAX,
        "inertial_seconds": AS_DEFAULT_INERTIAL_SECONDS,
        "tol_distance": AS_DEFAULT_TOL_DISTANCE,
        "use_approx_two_point": AS_DEFAULT_USE_APPROX_TWO_POINT,
        "v0_prob_aggregation_mode": v0_prob_aggregation_mode,
        "use_efficient_sigmoid": AS_DEFAULT_USE_EFFICIENT_SIGMOID,
        "factor": AS_DEFAULT_FACTOR,
        "factor2": AS_DEFAULT_FACTOR2,
    }


def _compute_as_default_max_from_simulation_inputs(
    *,
    node_ids: list[str],
    candidate_indices: list[int],
    player_pos: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    ball_pos: np.ndarray,
    target_index_lookup: dict[int, tuple[int, int]],
    possessor_index: int,
    playing_direction: float,
    exclude_passer: bool,
    eps: float,
    simulate_passes_fn: Callable[..., Any] | None,
    use_progress_bar: bool,
    chunk_size: int,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    max_speed: int | None = None,
    speed_step: float | None = None,
) -> pd.Series:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    frame_count = int(player_pos.shape[0])
    PLAYER_POS = np.asarray(player_pos, dtype=float)
    BALL_POS = np.repeat(np.asarray(ball_pos, dtype=float)[np.newaxis, :], frame_count, axis=0)
    phi_grid = np.repeat(
        np.linspace(
            AS_DEFAULT_PHI_OFFSET,
            2.0 * np.pi + AS_DEFAULT_PHI_OFFSET,
            AS_DEFAULT_N_ANGLES,
            endpoint=False,
            dtype=float,
        )[np.newaxis, :],
        frame_count,
        axis=0,
    )
    v0_values = as_default_v0_values(max_speed=max_speed, speed_step=speed_step)
    passer_teams = np.repeat("attack", frame_count).astype(object)
    passers = np.repeat(node_ids[possessor_index], frame_count).astype(object)
    _validate_simulation_contract(players=players, passers=passers, exclude_passer=exclude_passer)

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    maxima = {node_index: -np.inf for node_index in candidate_indices}
    if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX:
        simulation_result = simulate_passes_fn(
            **_as_default_simulation_kwargs(
                PLAYER_POS=PLAYER_POS,
                BALL_POS=BALL_POS,
                phi_grid=phi_grid,
                v0_grid=np.repeat(v0_values[np.newaxis, :], frame_count, axis=0),
                passer_teams=passer_teams,
                player_teams=player_teams,
                players=players,
                passers=passers,
                exclude_passer=exclude_passer,
                playing_direction=np.repeat(float(playing_direction), frame_count),
                use_progress_bar=use_progress_bar,
                chunk_size=chunk_size,
                v0_prob_aggregation_mode="max",
            )
        )
        _update_as_default_maxima_from_simulation_result(
            maxima=maxima,
            simulation_result=simulation_result,
            candidate_indices=candidate_indices,
            target_index_lookup=target_index_lookup,
            frame_count=frame_count,
            player_count=len(players),
        )
    else:
        for speed in v0_values:
            simulation_result = simulate_passes_fn(
                **_as_default_simulation_kwargs(
                    PLAYER_POS=PLAYER_POS,
                    BALL_POS=BALL_POS,
                    phi_grid=phi_grid,
                    v0_grid=np.full((frame_count, 1), float(speed), dtype=float),
                    passer_teams=passer_teams,
                    player_teams=player_teams,
                    players=players,
                    passers=passers,
                    exclude_passer=exclude_passer,
                    playing_direction=np.repeat(float(playing_direction), frame_count),
                    use_progress_bar=use_progress_bar,
                    chunk_size=chunk_size,
                    v0_prob_aggregation_mode=AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
                )
            )
            _update_as_default_maxima_from_simulation_result(
                maxima=maxima,
                simulation_result=simulation_result,
                candidate_indices=candidate_indices,
                target_index_lookup=target_index_lookup,
                frame_count=frame_count,
                player_count=len(players),
            )

    for graph_target_index, value in maxima.items():
        if math.isfinite(value):
            result.loc[node_ids[graph_target_index]] = float(np.clip(value, float(eps), 1.0 - float(eps)))
    return result


def _angle_grid(n_angles: int, *, offset: float = AS_DEFAULT_PHI_OFFSET) -> np.ndarray:
    return np.linspace(offset, 2.0 * np.pi + offset, int(n_angles), endpoint=False, dtype=float)


def _circular_angle_error(values: np.ndarray, center: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(center) + np.pi) % (2.0 * np.pi) - np.pi


def refined_angle_grid_from_coarse_angles(
    coarse_angles: np.ndarray,
    selected_angle_indices: list[int],
    *,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
) -> np.ndarray:
    if angle_step <= 0:
        raise ValueError("--angle-step must be positive.")
    if refine_angle_radius < 0:
        raise ValueError("--refine-angle-radius must be non-negative.")
    coarse_angles = np.asarray(coarse_angles, dtype=float)
    if coarse_angles.size == 0:
        return coarse_angles
    selected = sorted({int(index) % int(coarse_angles.size) for index in selected_angle_indices})
    radius = math.radians(float(refine_angle_radius))
    step = math.radians(float(angle_step))
    offsets = np.arange(-radius, radius + (step * 0.5), step, dtype=float)
    angles: list[float] = []
    for index in selected:
        base = float(coarse_angles[index])
        angles.extend(((base + offsets) % (2.0 * np.pi)).tolist())
    if not angles:
        return coarse_angles
    rounded = {round(float(angle), 12): float(angle) for angle in angles}
    return np.asarray(sorted(rounded.values()), dtype=float)


def physical_xpass_kernel_sigmas(speed: float, distance: float) -> tuple[float, float, float]:
    speed = float(speed)
    distance = float(distance)
    sigma_angle = math.radians((5.0 + max(0.0, speed - 15.0)) / 2.0)
    sigma_speed = 0.1 * speed
    sigma_distance = 0.1 * distance
    return sigma_angle, sigma_speed, sigma_distance


def _simulation_player_cum_prob(simulation_result: Any, *, frame_count: int, player_count: int, n_angles: int) -> np.ndarray:
    player_cum_prob = np.asarray(getattr(simulation_result, "player_cum_prob", None), dtype=float)
    if player_cum_prob.ndim != 4:
        raise ValueError(
            "accessible-space simulation_result.player_cum_prob must be 4D "
            f"[frame, player, angle, distance], got shape={player_cum_prob.shape}."
        )
    if (
        player_cum_prob.shape[0] != int(frame_count)
        or player_cum_prob.shape[1] != int(player_count)
        or player_cum_prob.shape[2] != int(n_angles)
    ):
        raise ValueError(
            "Unexpected player_cum_prob shape for AS-default physical xPass: "
            f"{player_cum_prob.shape}, frames={frame_count}, players={player_count}, "
            f"n_angles={n_angles}."
        )
    return player_cum_prob


def _simulation_r_grid(simulation_result: Any, n_distances: int) -> np.ndarray:
    r_grid = getattr(simulation_result, "r_grid", None)
    if r_grid is None:
        return np.arange(int(n_distances), dtype=float) * float(AS_DEFAULT_RADIAL_GRIDSIZE)
    r_grid = np.asarray(r_grid, dtype=float)
    if r_grid.ndim != 1 or r_grid.shape[0] != int(n_distances):
        raise ValueError(f"simulation_result.r_grid must be 1D with length {n_distances}, got shape={r_grid.shape}.")
    return r_grid


def _simulate_as_default_per_speed(
    *,
    simulate_passes_fn: Callable[..., Any],
    PLAYER_POS: np.ndarray,
    BALL_POS: np.ndarray,
    phi_values: np.ndarray,
    speeds: np.ndarray,
    passer_teams: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    passers: np.ndarray,
    exclude_passer: bool,
    playing_direction: np.ndarray,
    use_progress_bar: bool,
    chunk_size: int,
) -> list[Any]:
    frame_count = int(PLAYER_POS.shape[0])
    phi_grid = np.repeat(np.asarray(phi_values, dtype=float)[np.newaxis, :], frame_count, axis=0)
    return [
        simulate_passes_fn(
            **_as_default_simulation_kwargs(
                PLAYER_POS=PLAYER_POS,
                BALL_POS=BALL_POS,
                phi_grid=phi_grid,
                v0_grid=np.full((frame_count, 1), float(speed), dtype=float),
                passer_teams=passer_teams,
                player_teams=player_teams,
                players=players,
                passers=passers,
                exclude_passer=exclude_passer,
                playing_direction=playing_direction,
                use_progress_bar=use_progress_bar,
                chunk_size=chunk_size,
                v0_prob_aggregation_mode=AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
            )
        )
        for speed in speeds
    ]


def _target_grid_values(
    simulation_results: list[Any],
    *,
    speeds: np.ndarray,
    angles: np.ndarray,
    frame_index: int,
    target_sim_index: int,
    frame_count: int,
    player_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values_by_speed: list[np.ndarray] = []
    r_grid: np.ndarray | None = None
    for simulation_result in simulation_results:
        arr = _simulation_player_cum_prob(
            simulation_result,
            frame_count=frame_count,
            player_count=player_count,
            n_angles=len(angles),
        )
        on_pitch = _on_pitch_mask(simulation_result, frame_index=frame_index)
        if on_pitch.shape != arr.shape[2:]:
            raise ValueError(f"On-pitch mask shape {on_pitch.shape} does not match player_cum_prob grid {arr.shape[2:]}.")
        speed_values = np.where(on_pitch, arr[frame_index, target_sim_index, :, :], np.nan)
        values_by_speed.append(speed_values)
        if r_grid is None:
            r_grid = _simulation_r_grid(simulation_result, speed_values.shape[1])
    if r_grid is None:
        r_grid = np.asarray([], dtype=float)
    values = np.stack(values_by_speed, axis=0) if values_by_speed else np.empty((0, len(angles), 0), dtype=float)
    if values.shape[0] != len(speeds):
        raise ValueError(f"Expected {len(speeds)} speed surfaces, got {values.shape[0]}.")
    return values, r_grid


def _robust_xpass_metrics_from_values(values: np.ndarray, speeds: np.ndarray, angles: np.ndarray, distances: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return {
            PHYSICAL_XPASS_METRIC_NOISE_KERNEL: float("nan"),
            PHYSICAL_XPASS_METRIC_MAX: float("nan"),
            PHYSICAL_XPASS_METRIC_TOP10MEAN: float("nan"),
        }

    max_flat_index = int(np.nanargmax(np.where(finite, values, np.nan)))
    speed_index, angle_index, distance_index = np.unravel_index(max_flat_index, values.shape)
    best_value = float(values[speed_index, angle_index, distance_index])
    finite_values = values[finite]
    top_count = max(1, int(math.ceil(float(finite_values.size) * 0.10)))
    top_values = np.partition(finite_values, finite_values.size - top_count)[finite_values.size - top_count :]
    top10_value = float(np.mean(top_values))

    best_speed = float(speeds[speed_index])
    best_angle = float(angles[angle_index])
    best_distance = float(distances[distance_index])
    sigma_angle, sigma_speed, sigma_distance = physical_xpass_kernel_sigmas(best_speed, best_distance)
    sigma_angle = max(float(sigma_angle), 1e-12)
    sigma_speed = max(float(sigma_speed), 1e-12)
    sigma_distance = max(float(sigma_distance), 1e-12)

    speed_error = speeds[:, np.newaxis, np.newaxis] - best_speed
    angle_error = _circular_angle_error(angles[np.newaxis, :, np.newaxis], best_angle)
    distance_error = distances[np.newaxis, np.newaxis, :] - best_distance
    weights = (
        np.exp(-0.5 * (angle_error / sigma_angle) ** 2)
        * np.exp(-0.5 * (speed_error / sigma_speed) ** 2)
        * np.exp(-0.5 * (distance_error / sigma_distance) ** 2)
    )
    weights = np.where(finite, weights, 0.0)
    weight_sum = float(weights.sum())
    noise_value = best_value if weight_sum <= 0.0 else float(np.sum(weights * np.where(finite, values, 0.0)) / weight_sum)
    return {
        PHYSICAL_XPASS_METRIC_NOISE_KERNEL: noise_value,
        PHYSICAL_XPASS_METRIC_MAX: best_value,
        PHYSICAL_XPASS_METRIC_TOP10MEAN: top10_value,
    }


def _compute_as_default_robust_metrics_from_simulation_inputs(
    *,
    node_ids: list[str],
    candidate_indices: list[int],
    player_pos: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    ball_pos: np.ndarray,
    target_index_lookup: dict[int, tuple[int, int]],
    possessor_index: int,
    playing_direction: float,
    exclude_passer: bool,
    eps: float,
    simulate_passes_fn: Callable[..., Any] | None,
    use_progress_bar: bool,
    chunk_size: int,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
) -> pd.Series:
    frame_count = int(player_pos.shape[0])
    PLAYER_POS = np.asarray(player_pos, dtype=float)
    BALL_POS = np.repeat(np.asarray(ball_pos, dtype=float)[np.newaxis, :], frame_count, axis=0)
    speeds = as_default_v0_values(max_speed=max_speed, speed_step=speed_step)
    coarse_angles = _angle_grid(coarse_n_angles)
    passer_teams = np.repeat("attack", frame_count).astype(object)
    passers = np.repeat(node_ids[int(possessor_index)], frame_count).astype(object)
    _validate_simulation_contract(players=players, passers=passers, exclude_passer=exclude_passer)

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    coarse_results = _simulate_as_default_per_speed(
        simulate_passes_fn=simulate_passes_fn,
        PLAYER_POS=PLAYER_POS,
        BALL_POS=BALL_POS,
        phi_values=coarse_angles,
        speeds=speeds,
        passer_teams=passer_teams,
        player_teams=player_teams,
        players=players,
        passers=passers,
        exclude_passer=exclude_passer,
        playing_direction=np.repeat(float(playing_direction), frame_count),
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )

    selected_angle_indices: list[int] = []
    for graph_target_index in candidate_indices:
        frame_index, target_sim_index = target_index_lookup[graph_target_index]
        values, _distances = _target_grid_values(
            coarse_results,
            speeds=speeds,
            angles=coarse_angles,
            frame_index=frame_index,
            target_sim_index=target_sim_index,
            frame_count=frame_count,
            player_count=len(players),
        )
        finite_values = np.isfinite(values)
        if not bool(finite_values.any()):
            continue
        per_angle = np.full(values.shape[1], np.nan, dtype=float)
        for angle_index in range(values.shape[1]):
            angle_values = values[:, angle_index, :]
            if np.isfinite(angle_values).any():
                per_angle[angle_index] = float(np.nanmax(angle_values))
        finite_indices = np.flatnonzero(np.isfinite(per_angle))
        if finite_indices.size == 0:
            continue
        top_count = min(max(1, int(refine_top_k_angles)), int(finite_indices.size))
        top_local = finite_indices[np.argsort(per_angle[finite_indices])[-top_count:]]
        selected_angle_indices.extend(int(index) for index in top_local.tolist())

    refined_angles = refined_angle_grid_from_coarse_angles(
        coarse_angles,
        selected_angle_indices,
        refine_angle_radius=refine_angle_radius,
        angle_step=angle_step,
    )
    if refined_angles.size == 0:
        refined_angles = coarse_angles
    refined_results = _simulate_as_default_per_speed(
        simulate_passes_fn=simulate_passes_fn,
        PLAYER_POS=PLAYER_POS,
        BALL_POS=BALL_POS,
        phi_values=refined_angles,
        speeds=speeds,
        passer_teams=passer_teams,
        player_teams=player_teams,
        players=players,
        passers=passers,
        exclude_passer=exclude_passer,
        playing_direction=np.repeat(float(playing_direction), frame_count),
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )

    result_columns: list[str] = []
    for node_id in node_ids:
        result_columns.append(str(node_id))
        result_columns.append(physical_xpass_metric_column(str(node_id), PHYSICAL_XPASS_METRIC_MAX))
        result_columns.append(physical_xpass_metric_column(str(node_id), PHYSICAL_XPASS_METRIC_TOP10MEAN))
    result = pd.Series(np.nan, index=result_columns, dtype=float)
    for graph_target_index in candidate_indices:
        frame_index, target_sim_index = target_index_lookup[graph_target_index]
        values, distances = _target_grid_values(
            refined_results,
            speeds=speeds,
            angles=refined_angles,
            frame_index=frame_index,
            target_sim_index=target_sim_index,
            frame_count=frame_count,
            player_count=len(players),
        )
        metrics = _robust_xpass_metrics_from_values(values, speeds, refined_angles, distances)
        node_id = str(node_ids[graph_target_index])
        for metric, value in metrics.items():
            if math.isfinite(value):
                result.loc[physical_xpass_metric_column(node_id, metric)] = float(np.clip(value, float(eps), 1.0 - float(eps)))
    return result


def compute_graph_max_player_cum_prob_as_defaults(
    graph: Data,
    *,
    eps: float = 1e-4,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    max_speed: int | None = None,
    speed_step: float | None = None,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    node_ids = _node_ids(graph)
    candidate_indices = _candidate_target_indices(graph)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    if not candidate_indices:
        return result

    if consider_teammates:
        player_pos, player_teams, players, ball_pos, raw_target_lookup, possessor_index = _full_as_default_simulation_inputs(
            graph,
            candidate_indices=candidate_indices,
        )
        player_pos = player_pos[np.newaxis, :, :]
        target_index_lookup = {node_index: (0, sim_index) for node_index, sim_index in raw_target_lookup.items()}
        playing_direction = _infer_playing_direction_from_centered_players(player_pos[0], player_teams)
        exclude_passer = False
    else:
        (
            player_pos,
            player_teams,
            players,
            ball_pos,
            target_index_lookup,
            possessor_index,
            playing_direction,
        ) = _reduced_as_default_simulation_inputs(graph, candidate_indices=candidate_indices)
        exclude_passer = True

    return _compute_as_default_max_from_simulation_inputs(
        node_ids=node_ids,
        candidate_indices=candidate_indices,
        player_pos=player_pos,
        player_teams=player_teams,
        players=players,
        ball_pos=ball_pos,
        target_index_lookup=target_index_lookup,
        possessor_index=possessor_index,
        playing_direction=playing_direction,
        exclude_passer=exclude_passer,
        eps=eps,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
        speed_aggregation=speed_aggregation,
        max_speed=max_speed,
        speed_step=speed_step,
    )


def compute_graph_physical_xpass_metrics_as_defaults(
    graph: Data,
    *,
    eps: float = 1e-4,
    consider_teammates: bool = True,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    node_ids = _node_ids(graph)
    candidate_indices = _candidate_target_indices(graph)
    result_columns: list[str] = []
    for node_id in node_ids:
        result_columns.append(str(node_id))
        result_columns.append(physical_xpass_metric_column(str(node_id), PHYSICAL_XPASS_METRIC_MAX))
        result_columns.append(physical_xpass_metric_column(str(node_id), PHYSICAL_XPASS_METRIC_TOP10MEAN))
    result = pd.Series(np.nan, index=result_columns, dtype=float)
    if not candidate_indices:
        return result

    if consider_teammates:
        player_pos, player_teams, players, ball_pos, raw_target_lookup, possessor_index = _full_as_default_simulation_inputs(
            graph,
            candidate_indices=candidate_indices,
        )
        player_pos = player_pos[np.newaxis, :, :]
        target_index_lookup = {node_index: (0, sim_index) for node_index, sim_index in raw_target_lookup.items()}
        playing_direction = _infer_playing_direction_from_centered_players(player_pos[0], player_teams)
        exclude_passer = False
    else:
        (
            player_pos,
            player_teams,
            players,
            ball_pos,
            target_index_lookup,
            possessor_index,
            playing_direction,
        ) = _reduced_as_default_simulation_inputs(graph, candidate_indices=candidate_indices)
        exclude_passer = True

    return _compute_as_default_robust_metrics_from_simulation_inputs(
        node_ids=node_ids,
        candidate_indices=candidate_indices,
        player_pos=player_pos,
        player_teams=player_teams,
        players=players,
        ball_pos=ball_pos,
        target_index_lookup=target_index_lookup,
        possessor_index=possessor_index,
        playing_direction=playing_direction,
        exclude_passer=exclude_passer,
        eps=eps,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
        max_speed=max_speed,
        speed_step=speed_step,
        coarse_n_angles=coarse_n_angles,
        refine_top_k_angles=refine_top_k_angles,
        refine_angle_radius=refine_angle_radius,
        angle_step=angle_step,
    )


def compute_graphs_physical_xpass_metrics_as_defaults(
    graphs: list[Data],
    *,
    eps: float = 1e-4,
    consider_teammates: bool = True,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
    batch_size: int = 16,
) -> list[pd.Series]:
    del batch_size
    return [
        compute_graph_physical_xpass_metrics_as_defaults(
            graph,
            eps=eps,
            consider_teammates=consider_teammates,
            max_speed=max_speed,
            speed_step=speed_step,
            coarse_n_angles=coarse_n_angles,
            refine_top_k_angles=refine_top_k_angles,
            refine_angle_radius=refine_angle_radius,
            angle_step=angle_step,
            simulate_passes_fn=simulate_passes_fn,
            use_progress_bar=use_progress_bar,
            chunk_size=chunk_size,
        )
        for graph in graphs
    ]


def compute_graphs_max_player_cum_prob_as_defaults(
    graphs: list[Data],
    *,
    eps: float = 1e-4,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    max_speed: int | None = None,
    speed_step: float | None = None,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
    batch_size: int = 16,
) -> list[pd.Series]:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    batch_size = max(1, int(batch_size))
    results: list[pd.Series | None] = [None] * len(graphs)
    prepared_groups: dict[tuple[bool, tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    for graph_index, graph in enumerate(graphs):
        node_ids = _node_ids(graph)
        candidate_indices = _candidate_target_indices(graph)
        result = pd.Series(np.nan, index=node_ids, dtype=float)
        if not candidate_indices:
            results[graph_index] = result
            continue

        if consider_teammates:
            player_pos, player_teams, players, ball_pos, raw_target_lookup, possessor_index = _full_as_default_simulation_inputs(
                graph,
                candidate_indices=candidate_indices,
            )
            player_pos = player_pos[np.newaxis, :, :]
            frame_count = 1
            target_lookup = {node_index: (0, int(sim_index)) for node_index, sim_index in raw_target_lookup.items()}
            passers = np.asarray([node_ids[possessor_index]], dtype=object)
            playing_direction = np.asarray(
                [_infer_playing_direction_from_centered_players(player_pos[0], player_teams)],
                dtype=float,
            )
            exclude_passer = False
        else:
            (
                player_pos,
                player_teams,
                _players,
                ball_pos,
                target_lookup,
                _possessor_index,
                playing_direction_value,
            ) = _reduced_as_default_simulation_inputs(graph, candidate_indices=candidate_indices)
            frame_count = int(player_pos.shape[0])
            players = np.asarray(
                ["passer", "target_player"] + [f"defender_{index}" for index in range(int(player_pos.shape[1]) - 2)],
                dtype=object,
            )
            passers = np.repeat("passer", frame_count).astype(object)
            playing_direction = np.repeat(float(playing_direction_value), frame_count).astype(float)
            exclude_passer = True

        key = (
            bool(exclude_passer),
            tuple(str(player) for player in players.tolist()),
            tuple(str(team) for team in player_teams.tolist()),
        )
        prepared_groups.setdefault(key, []).append(
            {
                "graph_index": graph_index,
                "node_ids": node_ids,
                "candidate_indices": candidate_indices,
                "player_pos": player_pos,
                "player_teams": player_teams,
                "players": players,
                "ball_pos": np.repeat(np.asarray(ball_pos, dtype=float)[np.newaxis, :], frame_count, axis=0),
                "target_lookup": target_lookup,
                "passers": passers,
                "playing_direction": playing_direction,
                "exclude_passer": exclude_passer,
                "frame_count": frame_count,
            }
        )

    if not prepared_groups:
        return [result if result is not None else pd.Series(dtype=float) for result in results]

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    for group_frames in prepared_groups.values():
        for batch_start in range(0, len(group_frames), batch_size):
            batch = group_frames[batch_start : batch_start + batch_size]
            players = np.asarray(batch[0]["players"], dtype=object)
            player_teams = np.asarray(batch[0]["player_teams"], dtype=object)
            exclude_passer = bool(batch[0]["exclude_passer"])
            player_pos = np.concatenate([np.asarray(item["player_pos"], dtype=float) for item in batch], axis=0)
            ball_pos = np.concatenate([np.asarray(item["ball_pos"], dtype=float) for item in batch], axis=0)
            frame_count = int(player_pos.shape[0])
            phi_grid = np.repeat(
                np.linspace(
                    AS_DEFAULT_PHI_OFFSET,
                    2.0 * np.pi + AS_DEFAULT_PHI_OFFSET,
                    AS_DEFAULT_N_ANGLES,
                    endpoint=False,
                    dtype=float,
                )[np.newaxis, :],
                frame_count,
                axis=0,
            )
            passers = np.concatenate([np.asarray(item["passers"], dtype=object) for item in batch], axis=0)
            passer_teams = np.repeat("attack", frame_count).astype(object)
            playing_direction = np.concatenate(
                [np.asarray(item["playing_direction"], dtype=float) for item in batch],
                axis=0,
            )
            _validate_simulation_contract(players=players, passers=passers, exclude_passer=exclude_passer)
            maxima: dict[int, float] = {}
            target_index_lookup: dict[int, tuple[int, int]] = {}
            target_keys: dict[int, tuple[int, int]] = {}
            next_target_key = 0
            frame_offset = 0
            for item in batch:
                for graph_target_index, (local_frame_index, sim_index) in item["target_lookup"].items():
                    target_key = next_target_key
                    next_target_key += 1
                    maxima[target_key] = -np.inf
                    target_index_lookup[target_key] = (frame_offset + int(local_frame_index), int(sim_index))
                    target_keys[target_key] = (int(item["graph_index"]), int(graph_target_index))
                frame_offset += int(item["frame_count"])

            v0_values = as_default_v0_values(max_speed=max_speed, speed_step=speed_step)
            if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX:
                simulation_results = [
                    simulate_passes_fn(
                        **_as_default_simulation_kwargs(
                            PLAYER_POS=player_pos,
                            BALL_POS=ball_pos,
                            phi_grid=phi_grid,
                            v0_grid=np.repeat(v0_values[np.newaxis, :], frame_count, axis=0),
                            passer_teams=passer_teams,
                            player_teams=player_teams,
                            players=players,
                            passers=passers,
                            exclude_passer=exclude_passer,
                            playing_direction=playing_direction,
                            use_progress_bar=use_progress_bar,
                            chunk_size=chunk_size,
                            v0_prob_aggregation_mode="max",
                        )
                    )
                ]
            else:
                simulation_results = [
                    simulate_passes_fn(
                        **_as_default_simulation_kwargs(
                            PLAYER_POS=player_pos,
                            BALL_POS=ball_pos,
                            phi_grid=phi_grid,
                            v0_grid=np.full((frame_count, 1), float(speed), dtype=float),
                            passer_teams=passer_teams,
                            player_teams=player_teams,
                            players=players,
                            passers=passers,
                            exclude_passer=exclude_passer,
                            playing_direction=playing_direction,
                            use_progress_bar=use_progress_bar,
                            chunk_size=chunk_size,
                            v0_prob_aggregation_mode=AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
                        )
                    )
                    for speed in v0_values
                ]

            for simulation_result in simulation_results:
                _update_as_default_maxima_from_simulation_result(
                    maxima=maxima,
                    simulation_result=simulation_result,
                    candidate_indices=list(maxima.keys()),
                    target_index_lookup=target_index_lookup,
                    frame_count=frame_count,
                    player_count=len(players),
                )

            for item in batch:
                graph_index = int(item["graph_index"])
                if results[graph_index] is None:
                    results[graph_index] = pd.Series(np.nan, index=item["node_ids"], dtype=float)
            for target_key, value in maxima.items():
                graph_index, graph_target_index = target_keys[target_key]
                if math.isfinite(value):
                    node_id = graphs[graph_index].node_ids[graph_target_index]
                    results[graph_index].loc[str(node_id)] = float(np.clip(value, float(eps), 1.0 - float(eps)))

    return [result if result is not None else pd.Series(dtype=float) for result in results]


def compute_graph_player_cum_prob(
    graph: Data,
    *,
    eps: float = 1e-4,
    normalize: bool = True,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    if not normalize:
        warnings.warn("--no-normalize is ignored for AS-default max physical xPass; normalize=True is required.")
    return compute_graph_max_player_cum_prob_as_defaults(
        graph,
        eps=eps,
        consider_teammates=consider_teammates,
        speed_aggregation=speed_aggregation,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )


def compute_graph_player_cum_prob_at_target_location(
    graph: Data,
    *,
    eps: float = 1e-4,
    normalize: bool = True,
    consider_teammates: bool = False,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    node_ids = _node_ids(graph)
    candidate_indices = _candidate_target_indices(graph)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    if not candidate_indices:
        return result

    x = graph.x.detach().cpu().to(torch.float32)
    target_xy = x[candidate_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy()
    player_pos, player_teams, players, ball_pos, passer_teams, target_index_lookup = _build_simulation_inputs(
        graph,
        candidate_indices=candidate_indices,
        consider_teammates=consider_teammates,
    )
    deltas = target_xy - ball_pos[np.newaxis, :]
    distances = np.linalg.norm(deltas, axis=1)
    phis = np.arctan2(deltas[:, 1], deltas[:, 0])

    finite_targets = np.isfinite(distances) & (distances > 1e-6) & np.isfinite(phis)
    if not finite_targets.all():
        bad_ids = [node_ids[candidate_indices[i]] for i, ok in enumerate(finite_targets.tolist()) if not ok]
        raise ValueError(f"Physical xPass candidate targets have invalid distance/angle: {bad_ids}.")

    frame_count = len(candidate_indices)
    PLAYER_POS = np.asarray(player_pos, dtype=float)
    BALL_POS = np.repeat(ball_pos[np.newaxis, :], frame_count, axis=0)
    phi_grid = phis.reshape(frame_count, 1)
    v0_grid = np.repeat(
        np.linspace(DEFAULT_V0_MIN, DEFAULT_V0_MAX, DEFAULT_N_V0, dtype=float)[np.newaxis, :],
        frame_count,
        axis=0,
    )
    possessor_index = int(torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).item())
    passers = np.repeat(node_ids[possessor_index], frame_count).astype(object)
    _validate_simulation_contract(players=players, passers=passers, exclude_passer=True)

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    simulation_result = simulate_passes_fn(
        PLAYER_POS=PLAYER_POS,
        BALL_POS=BALL_POS,
        phi_grid=phi_grid,
        v0_grid=v0_grid,
        passer_teams=passer_teams,
        player_teams=player_teams,
        players=players,
        passers=passers,
        exclude_passer=True,
        x_pitch_min=0.0,
        x_pitch_max=float(FIELD_SIZE[0]),
        y_pitch_min=0.0,
        y_pitch_max=float(FIELD_SIZE[1]),
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
        fields_to_return=("player_cum_prob",),
        normalize=normalize,
    )

    for frame_index, graph_target_index in enumerate(candidate_indices):
        target_sim_index = target_index_lookup[graph_target_index]
        value, _ = _extract_simulation_probability(
            simulation_result,
            frame_index=frame_index,
            target_player_index=target_sim_index,
            angle_index=0,
            target_distance=float(distances[frame_index]),
        )
        value = float(np.clip(value, float(eps), 1.0 - float(eps)))
        result.loc[node_ids[graph_target_index]] = value

    return result


def compute_graph_physical_xpass_for_source(
    graph: Data,
    *,
    source: str,
    eps: float = 1e-4,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    source = str(source)
    if source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    policy = teammate_policy or (
        PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
        if source == PHYSICAL_XPASS_LEGACY_SOURCE
        else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    )
    consider_teammates = policy == PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    if source == PHYSICAL_XPASS_LEGACY_SOURCE:
        return compute_graph_player_cum_prob_at_target_location(
            graph,
            eps=eps,
            normalize=True,
            consider_teammates=consider_teammates,
            simulate_passes_fn=simulate_passes_fn,
            use_progress_bar=use_progress_bar,
            chunk_size=chunk_size,
        )
    return compute_graph_max_player_cum_prob_as_defaults(
        graph,
        eps=eps,
        consider_teammates=consider_teammates,
        speed_aggregation=speed_aggregation,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )


def load_physical_xpass_match(cache_dir: str | Path, match_id: str) -> pd.DataFrame:
    cache_root = Path(cache_dir)
    direct_path = cache_root / "matches" / f"{match_id}.parquet"
    path = direct_path if cache_root.name == "physical_xpass" or direct_path.exists() else get_physical_xpass_match_path(str(match_id), root=cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Physical xPass sidecar not found at {path}. "
            "Run scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> before training with --use_physical_xpass."
        )
    frame = pd.read_parquet(path)
    if "action_index" not in frame.columns:
        raise ValueError(f"Physical xPass sidecar {path} is missing required column 'action_index'.")
    frame = frame.copy()
    frame["action_index"] = frame["action_index"].astype(int)
    if frame["action_index"].duplicated().any():
        duplicates = frame.loc[frame["action_index"].duplicated(), "action_index"].head(5).tolist()
        raise ValueError(f"Physical xPass sidecar {path} contains duplicate action_index rows, e.g. {duplicates}.")
    return frame.set_index("action_index", drop=False)


def load_physical_xpass_component(
    cache_dir: str | Path,
    match_id: str,
    action_index: int,
    *,
    metric: str | None = None,
) -> pd.Series:
    frame = load_physical_xpass_match(cache_dir, match_id)
    if int(action_index) not in frame.index:
        raise KeyError(f"Physical xPass sidecar has no row for match_id={match_id}, action_index={int(action_index)}.")
    row = frame.loc[int(action_index)]
    selected_metric = normalize_physical_xpass_metric(metric)
    suffix = PHYSICAL_XPASS_METRIC_SUFFIXES[selected_metric]
    player_columns = []
    output_index = []
    for column in row.index:
        if column in PHYSICAL_XPASS_ID_COLUMNS:
            continue
        column = str(column)
        if "__" in column and not suffix:
            continue
        if suffix and not column.endswith(suffix):
            continue
        player_columns.append(column)
        output_index.append(column[: -len(suffix)] if suffix else column)
    series = pd.to_numeric(row[player_columns], errors="coerce").astype(float)
    series.index = output_index
    series.name = selected_metric
    return series


def validate_runtime_physical_xpass_visualization_cache(cache_dir: str | Path) -> dict[str, Any]:
    return validate_physical_xpass_cache_metadata(
        cache_dir,
        expected_source=PHYSICAL_XPASS_SOURCE,
        expected_speed_aggregation=PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
        expected_metric_schema_version=PHYSICAL_XPASS_METRIC_SCHEMA_VERSION,
        expected_default_metric=PHYSICAL_XPASS_DEFAULT_METRIC,
        expected_max_speed=AS_DEFAULT_V0_MAX,
        expected_speed_step=AS_DEFAULT_SPEED_STEP,
        expected_coarse_n_angles=AS_DEFAULT_COARSE_N_ANGLES,
        expected_refine_top_k_angles=AS_DEFAULT_REFINE_TOP_K_ANGLES,
        expected_refine_angle_radius=AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
        expected_angle_step=AS_DEFAULT_ANGLE_STEP_DEG,
    )


def load_runtime_physical_xpass_visualization_component(
    cache_dir: str | Path,
    match_id: str,
    action_index: int,
    *,
    metric: str | None = None,
) -> pd.Series:
    selected_metric = normalize_physical_xpass_metric(metric)
    validate_runtime_physical_xpass_visualization_cache(cache_dir)
    try:
        series = load_physical_xpass_component(cache_dir, match_id, action_index, metric=selected_metric)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise type(exc)(
            f"Could not load runtime physical xPass visualization row for "
            f"match_id={match_id}, action_index={int(action_index)}, metric={selected_metric!r} "
            f"from {Path(cache_dir)}. Run scripts/generate_physical_xpass.py first. {exc}"
        ) from exc
    if series.empty:
        raise ValueError(
            f"Runtime physical xPass cache row for match_id={match_id}, action_index={int(action_index)} "
            f"does not contain columns for metric {selected_metric!r} in {Path(cache_dir)}."
        )
    series.name = int(action_index)
    return series


def load_runtime_physical_xpass_visualization_table(
    cache_dir: str | Path,
    match_id: str,
    action_indices: list[int] | tuple[int, ...],
    *,
    metric: str | None = None,
) -> pd.DataFrame:
    rows = [
        load_runtime_physical_xpass_visualization_component(
            cache_dir,
            match_id,
            int(action_index),
            metric=metric,
        )
        for action_index in action_indices
    ]
    if not rows:
        return pd.DataFrame(dtype=float)
    table = pd.DataFrame(rows)
    table.index = [int(action_index) for action_index in action_indices]
    table.index.name = "action_index"
    return table.sort_index()


def resolve_physical_num_workers(value: str | int) -> int:
    if isinstance(value, int):
        workers = value
    else:
        text = str(value).strip().lower()
        if text == "auto":
            cpu_count = os.cpu_count() or 1
            workers = max(1, min(6, cpu_count - 2))
        else:
            workers = int(text)
    if workers < 1:
        raise ValueError("--physical-num-workers must be a positive integer or 'auto'.")
    return workers


def configure_physical_worker_thread_limit(limit: int) -> None:
    value = str(int(limit))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def _runtime_cache_metadata(
    *,
    source: str,
    teammate_policy: str,
    speed_aggregation: str,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
) -> dict[str, Any]:
    if source == PHYSICAL_XPASS_SOURCE:
        return physical_xpass_as_default_metadata(
            teammate_policy,
            speed_aggregation=speed_aggregation,
            max_speed=max_speed,
            speed_step=speed_step,
            coarse_n_angles=coarse_n_angles,
            refine_top_k_angles=refine_top_k_angles,
            refine_angle_radius=refine_angle_radius,
            angle_step=angle_step,
        )
    if source == PHYSICAL_XPASS_LEGACY_SOURCE:
        return {
            "metric": "player_cum_prob",
            "source": PHYSICAL_XPASS_LEGACY_SOURCE,
            "teammate_policy": teammate_policy,
            "speed_aggregation": normalize_physical_xpass_speed_aggregation(speed_aggregation),
            "storage": "wide_parquet_one_row_per_action_player_id_columns",
        }
    raise ValueError(f"Unsupported physical_xpass_source={source!r}.")


def _ensure_runtime_physical_xpass_cache(
    cache_dir: str | Path,
    *,
    source: str,
    teammate_policy: str,
    speed_aggregation: str,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    cache_root = Path(cache_dir)
    metadata_path = cache_root / "metadata.json"
    expected_metadata = _runtime_cache_metadata(
        source=source,
        teammate_policy=teammate_policy,
        speed_aggregation=speed_aggregation,
        max_speed=max_speed,
        speed_step=speed_step,
        coarse_n_angles=coarse_n_angles,
        refine_top_k_angles=refine_top_k_angles,
        refine_angle_radius=refine_angle_radius,
        angle_step=angle_step,
    )
    if metadata_path.exists():
        metadata = validate_physical_xpass_cache_metadata(
            cache_root,
            expected_source=source,
            expected_speed_aggregation=speed_aggregation,
            expected_metric_schema_version=PHYSICAL_XPASS_METRIC_SCHEMA_VERSION if source == PHYSICAL_XPASS_SOURCE else None,
            expected_default_metric=PHYSICAL_XPASS_DEFAULT_METRIC if source == PHYSICAL_XPASS_SOURCE else None,
            expected_max_speed=float(as_default_v0_values(max_speed=max_speed, speed_step=speed_step)[-1])
            if source == PHYSICAL_XPASS_SOURCE
            else None,
            expected_speed_step=float(speed_step if speed_step is not None else AS_DEFAULT_SPEED_STEP)
            if source == PHYSICAL_XPASS_SOURCE
            else None,
            expected_coarse_n_angles=int(coarse_n_angles) if source == PHYSICAL_XPASS_SOURCE else None,
            expected_refine_top_k_angles=int(refine_top_k_angles) if source == PHYSICAL_XPASS_SOURCE else None,
            expected_refine_angle_radius=float(refine_angle_radius) if source == PHYSICAL_XPASS_SOURCE else None,
            expected_angle_step=float(angle_step) if source == PHYSICAL_XPASS_SOURCE else None,
        )
        actual_policy = metadata.get("teammate_policy")
        if actual_policy is not None and actual_policy != teammate_policy:
            raise ValueError(
                f"Physical xPass sidecars at {cache_root} use incompatible teammate_policy "
                f"{actual_policy!r}; expected {teammate_policy!r}."
            )
        return metadata

    metadata = {
        **expected_metadata,
        "created_for": "runtime_physical_xpass_cache",
        "storage": "wide_parquet_one_row_per_action_player_id_columns",
    }
    if create_if_missing:
        (cache_root / "matches").mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _write_runtime_physical_xpass_rows(
    cache_dir: str | Path,
    match_id: str,
    rows: pd.DataFrame,
) -> Path:
    cache_root = Path(cache_dir)
    output_path = cache_root / "matches" / f"{match_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if "action_index" in existing.columns:
            rows = pd.concat(
                [
                    existing.loc[~existing["action_index"].astype(int).isin(rows["action_index"].astype(int))],
                    rows,
                ],
                ignore_index=True,
                sort=False,
            )
    rows = rows.sort_values("action_index").reset_index(drop=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp.parquet")
    rows.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    return output_path


def _runtime_physical_xpass_stats(cache_dir: str | Path) -> dict[str, object]:
    return {
        "cache_dir": str(Path(cache_dir)),
        "rows_scanned": 0,
        "pass_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_written": 0,
        "copied_from_reuse": 0,
        "pass_distance_filled": 0,
        "hash_mismatch_recomputed": 0,
        "online_graphs": 0,
        "compute_chunks": 0,
        "skipped_all_nan": 0,
        "cache_disabled": False,
        "dry_run": False,
        "num_workers": 1,
        "worker_thread_limit": None,
        "physical_batch_size": 16,
        "matches": {},
    }


def _merge_runtime_physical_xpass_stats(target: dict[str, object], source: dict[str, object]) -> dict[str, object]:
    for key in [
        "rows_scanned",
        "pass_rows",
        "cache_hits",
        "cache_misses",
        "cache_written",
        "copied_from_reuse",
        "pass_distance_filled",
        "hash_mismatch_recomputed",
        "online_graphs",
        "compute_chunks",
        "skipped_all_nan",
    ]:
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))
    target["cache_dir"] = source.get("cache_dir", target.get("cache_dir"))
    target["cache_disabled"] = bool(target.get("cache_disabled", False)) or bool(source.get("cache_disabled", False))
    target["dry_run"] = bool(target.get("dry_run", False)) or bool(source.get("dry_run", False))
    target["num_workers"] = source.get("num_workers", target.get("num_workers"))
    target["worker_thread_limit"] = source.get("worker_thread_limit", target.get("worker_thread_limit"))
    target["physical_batch_size"] = source.get("physical_batch_size", target.get("physical_batch_size"))
    target_matches = target.setdefault("matches", {})
    source_matches = source.get("matches", {})
    if isinstance(target_matches, dict) and isinstance(source_matches, dict):
        for match_id, match_stats in source_matches.items():
            current = target_matches.setdefault(str(match_id), {})
            if isinstance(current, dict) and isinstance(match_stats, dict):
                for key in [
                    "rows_scanned",
                    "pass_rows",
                    "cache_hits",
                    "cache_misses",
                    "cache_written",
                    "copied_from_reuse",
                    "pass_distance_filled",
                    "hash_mismatch_recomputed",
                    "online_graphs",
                    "compute_chunks",
                    "skipped_all_nan",
                ]:
                    current[key] = int(current.get(key, 0)) + int(match_stats.get(key, 0))
    return target


def _aggregate_physical_xpass_stat_tree(stats: dict[str, object] | None) -> dict[str, int]:
    totals = {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_written": 0,
        "copied_from_reuse": 0,
        "pass_distance_filled": 0,
        "hash_mismatch_recomputed": 0,
        "online_graphs": 0,
        "compute_chunks": 0,
        "skipped_all_nan": 0,
    }
    if not isinstance(stats, dict):
        return totals

    def visit(value: object) -> None:
        if not isinstance(value, dict):
            return
        if any(key in value for key in totals):
            for key in totals:
                totals[key] += int(value.get(key, 0) or 0)
            return
        for child in value.values():
            visit(child)

    visit(stats)
    return totals


def summarize_physical_xpass_cache_usage(
    *,
    physical_xpass_required: bool,
    cache_disabled: bool,
    refresh_requested: bool,
    cache_dir: str | Path | None,
    prewarm_stats: dict[str, object] | None = None,
    runtime_stats: dict[str, object] | None = None,
) -> dict[str, object]:
    prewarm_totals = _aggregate_physical_xpass_stat_tree(prewarm_stats)
    runtime_totals = _aggregate_physical_xpass_stat_tree(runtime_stats)
    has_prewarm_work = any(prewarm_totals.values())
    totals = prewarm_totals if has_prewarm_work else runtime_totals

    cache_hits = int(totals["cache_hits"])
    cache_misses = int(totals["cache_misses"])
    cache_written = int(totals["cache_written"])
    copied_from_reuse = int(totals["copied_from_reuse"])
    pass_distance_filled = int(totals["pass_distance_filled"])
    hash_mismatch_recomputed = int(totals["hash_mismatch_recomputed"])
    online_graphs = int(totals["online_graphs"])
    requested_rows = cache_hits + cache_misses

    if not physical_xpass_required:
        reason = "physical_xpass_not_required"
    elif cache_disabled:
        reason = "cache_disabled"
    elif refresh_requested:
        reason = "refresh_requested"
    elif hash_mismatch_recomputed > 0:
        reason = "hash_mismatch"
    elif cache_misses > 0 or cache_written > 0:
        reason = "missing_or_cold_cache_rows"
    elif cache_hits > 0:
        reason = "cache_hit"
    else:
        reason = "no_runtime_physical_xpass_work"

    return {
        "cache_enabled": bool(not cache_disabled),
        "physical_xpass_required": bool(physical_xpass_required),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_reused": cache_hits > 0,
        "cache_fully_reused": requested_rows > 0 and cache_hits == requested_rows,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_written": cache_written,
        "copied_from_reuse": copied_from_reuse,
        "pass_distance_filled": pass_distance_filled,
        "hash_mismatch_recomputed": hash_mismatch_recomputed,
        "online_graphs": online_graphs,
        "requested_rows": requested_rows,
        "reason": reason,
    }


def format_physical_xpass_cache_summary(summary: dict[str, object]) -> str:
    reason = str(summary.get("reason", "unknown"))
    cache_dir = summary.get("cache_dir")
    hits = int(summary.get("cache_hits", 0) or 0)
    misses = int(summary.get("cache_misses", 0) or 0)
    written = int(summary.get("cache_written", 0) or 0)
    copied = int(summary.get("copied_from_reuse", 0) or 0)
    distance_filled = int(summary.get("pass_distance_filled", 0) or 0)
    requested = int(summary.get("requested_rows", hits + misses) or 0)
    online_graphs = int(summary.get("online_graphs", 0) or 0)

    if reason == "physical_xpass_not_required":
        return "Physical xPass cache: not used; pass-success model does not require physical xPass."
    if reason == "cache_disabled":
        return f"Physical xPass cache: disabled by --no-physical-cache; computed {online_graphs} rows online during inference."
    if reason == "cache_hit":
        return (
            f"Physical xPass cache: reused {hits}/{requested} rows from {cache_dir}; "
            f"copied={copied} pass_distance_filled={distance_filled}."
        )
    if reason in {"hash_mismatch", "missing_or_cold_cache_rows", "refresh_requested"}:
        return (
            f"Physical xPass cache: recomputed {written} rows; reason={reason}; "
            f"hits={hits} misses={misses} written={written} copied={copied} "
            f"pass_distance_filled={distance_filled}."
        )
    return f"Physical xPass cache: no runtime physical xPass work; reason={reason}."


def _chunk_items(items: list[dict[str, Any]], chunks: int) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    chunks = max(1, min(int(chunks), len(items)))
    chunk_size = int(math.ceil(len(items) / chunks))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _chunk_items_by_size(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    chunk_size = max(1, int(chunk_size))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _physical_xpass_row_has_finite_metric(row: dict[str, Any]) -> bool:
    metadata_columns = {"match_id", "action_index", "physical_state_hash", PHYSICAL_XPASS_PASS_DISTANCE_COLUMN}
    for key, value in row.items():
        if key in metadata_columns:
            continue
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _compute_runtime_physical_xpass_chunk(task: dict[str, Any]) -> dict[str, object]:
    misses = list(task["misses"])
    source = str(task["source"])
    eps = float(task["eps"])
    teammate_policy = str(task["teammate_policy"])
    speed_aggregation = normalize_physical_xpass_speed_aggregation(task.get("speed_aggregation"))
    physical_batch_size = int(task.get("physical_batch_size", 16))
    consider_teammates = teammate_policy == PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    max_speed = task.get("max_speed", None)
    speed_step = task.get("speed_step", None)
    coarse_n_angles = int(task.get("coarse_n_angles", AS_DEFAULT_COARSE_N_ANGLES))
    refine_top_k_angles = int(task.get("refine_top_k_angles", AS_DEFAULT_REFINE_TOP_K_ANGLES))
    refine_angle_radius = float(task.get("refine_angle_radius", AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG))
    angle_step = float(task.get("angle_step", AS_DEFAULT_ANGLE_STEP_DEG))

    if source == PHYSICAL_XPASS_SOURCE:
        computed_probs = compute_graphs_physical_xpass_metrics_as_defaults(
            [item["graph"] for item in misses],
            eps=eps,
            consider_teammates=consider_teammates,
            max_speed=max_speed,
            speed_step=speed_step,
            coarse_n_angles=coarse_n_angles,
            refine_top_k_angles=refine_top_k_angles,
            refine_angle_radius=refine_angle_radius,
            angle_step=angle_step,
            batch_size=physical_batch_size,
        )
    else:
        computed_probs = [
            compute_graph_physical_xpass_for_source(
                item["graph"],
                source=source,
                eps=eps,
                teammate_policy=teammate_policy,
                speed_aggregation=speed_aggregation,
            )
            for item in misses
        ]

    if len(computed_probs) != len(misses):
        raise ValueError(f"Physical xPass runtime worker returned {len(computed_probs)} rows for {len(misses)} misses.")

    rows = []
    for item, probs in zip(misses, computed_probs):
        row = {
            "match_id": str(item["match_id"]),
            "action_index": int(item["action_index"]),
            "physical_state_hash": str(item["physical_state_hash"]),
            PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: float(item.get(PHYSICAL_XPASS_PASS_DISTANCE_COLUMN, float("nan"))),
        }
        row.update({str(player_id): float(value) for player_id, value in probs.items()})
        rows.append(row)
    skipped_all_nan = sum(1 for row in rows if not _physical_xpass_row_has_finite_metric(row))
    return {"rows": rows, "computed": len(rows), "skipped_all_nan": int(skipped_all_nan)}


def _has_finite_pass_distance(row: pd.Series) -> bool:
    if PHYSICAL_XPASS_PASS_DISTANCE_COLUMN not in row.index:
        return False
    value = row.get(PHYSICAL_XPASS_PASS_DISTANCE_COLUMN, np.nan)
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _physical_row_hash_matches_or_missing(row: pd.Series, state_hash: str) -> tuple[bool, bool]:
    cached_hash = row.get("physical_state_hash", None)
    if pd.isna(cached_hash) or cached_hash is None:
        return True, True
    return str(cached_hash) == state_hash, False


def _runtime_match_stats_template() -> dict[str, int]:
    return {
        "rows_scanned": 0,
        "pass_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_written": 0,
        "copied_from_reuse": 0,
        "pass_distance_filled": 0,
        "hash_mismatch_recomputed": 0,
        "online_graphs": 0,
        "compute_chunks": 0,
        "skipped_all_nan": 0,
    }


def prewarm_physical_xpass_runtime_cache(
    items: list[dict[str, Any]],
    *,
    cache_dir: str | Path,
    source: str,
    eps: float = 1e-4,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    refresh: bool = False,
    num_workers: str | int = "auto",
    worker_thread_limit: int = 1,
    physical_batch_size: int = 16,
    reuse_cache_dir: str | Path | None = None,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    dry_run: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
    verbose_status: bool = False,
) -> dict[str, object]:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    teammate_policy = teammate_policy or (
        PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
        if source == PHYSICAL_XPASS_LEGACY_SOURCE
        else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    )
    if int(worker_thread_limit) < 1:
        raise ValueError("--physical-worker-thread-limit must be positive.")
    if int(physical_batch_size) < 1:
        raise ValueError("--physical-batch-size must be positive.")

    resolved_workers = resolve_physical_num_workers(num_workers)
    _ensure_runtime_physical_xpass_cache(
        cache_dir,
        source=source,
        teammate_policy=teammate_policy,
        speed_aggregation=speed_aggregation,
        max_speed=max_speed,
        speed_step=speed_step,
        coarse_n_angles=coarse_n_angles,
        refine_top_k_angles=refine_top_k_angles,
        refine_angle_radius=refine_angle_radius,
        angle_step=angle_step,
        create_if_missing=not bool(dry_run),
    )

    stats = _runtime_physical_xpass_stats(cache_dir)
    stats["num_workers"] = int(resolved_workers)
    stats["worker_thread_limit"] = int(worker_thread_limit)
    stats["physical_batch_size"] = int(physical_batch_size)
    stats["dry_run"] = bool(dry_run)
    match_stats_by_id: dict[str, dict[str, int]] = {}
    misses: list[dict[str, Any]] = []
    copied_rows: list[dict[str, Any]] = []
    seen_miss_keys: set[tuple[str, int]] = set()
    cache_by_match: dict[str, pd.DataFrame | None] = {}
    reuse_by_match: dict[str, pd.DataFrame | None] = {}

    for item in items:
        match_id = str(item["match_id"])
        graphs = list(item.get("graphs") or [])
        labels = item.get("labels")
        if labels is None or len(graphs) == 0:
            continue
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.float32)
        if len(graphs) != int(labels.shape[0]):
            raise ValueError(f"Runtime physical xPass item for {match_id} has {len(graphs)} graphs and {int(labels.shape[0])} labels.")

        if match_id not in cache_by_match:
            try:
                cache_by_match[match_id] = load_physical_xpass_match(cache_dir, match_id)
            except FileNotFoundError:
                cache_by_match[match_id] = None
        cached_rows = cache_by_match[match_id]
        if reuse_cache_dir is not None and match_id not in reuse_by_match:
            try:
                reuse_by_match[match_id] = load_physical_xpass_match(reuse_cache_dir, match_id)
            except FileNotFoundError:
                reuse_by_match[match_id] = None
        reuse_rows = reuse_by_match.get(match_id)
        match_stats = match_stats_by_id.setdefault(match_id, _runtime_match_stats_template())
        row_count = int(labels.shape[0])
        try:
            pass_count = int((labels[:, LABEL_INDEX["is_pass"]] == 1).sum().item())
        except Exception:
            pass_count = row_count
        stats["rows_scanned"] = int(stats["rows_scanned"]) + row_count
        stats["pass_rows"] = int(stats["pass_rows"]) + pass_count
        match_stats["rows_scanned"] += row_count
        match_stats["pass_rows"] += pass_count

        for graph, label in zip(graphs, labels):
            action_index = int(label[LABEL_INDEX["action_index"]].item())
            state_hash = physical_state_hash(graph)
            pass_distance = observed_pass_distance(graph, label)
            if not refresh and cached_rows is not None and action_index in cached_rows.index:
                cached_row = cached_rows.loc[action_index]
                hash_matches, missing_hash = _physical_row_hash_matches_or_missing(cached_row, state_hash)
                if hash_matches and not missing_hash and _has_finite_pass_distance(cached_row):
                    stats["cache_hits"] = int(stats["cache_hits"]) + 1
                    match_stats["cache_hits"] += 1
                    continue
                if not hash_matches:
                    stats["hash_mismatch_recomputed"] = int(stats["hash_mismatch_recomputed"]) + 1
                    match_stats["hash_mismatch_recomputed"] += 1
                if hash_matches:
                    copied_row = cached_row.to_dict()
                    copied_row["match_id"] = match_id
                    copied_row["action_index"] = action_index
                    copied_row["physical_state_hash"] = state_hash
                    copied_row[PHYSICAL_XPASS_PASS_DISTANCE_COLUMN] = pass_distance
                    copied_rows.append(copied_row)
                    stats["copied_from_reuse"] = int(stats["copied_from_reuse"]) + 1
                    stats["pass_distance_filled"] = int(stats["pass_distance_filled"]) + int(
                        not _has_finite_pass_distance(cached_row)
                    )
                    match_stats["copied_from_reuse"] += 1
                    match_stats["pass_distance_filled"] += int(not _has_finite_pass_distance(cached_row))
                    continue

            if not refresh and reuse_rows is not None and action_index in reuse_rows.index:
                reuse_row = reuse_rows.loc[action_index]
                hash_matches, _missing_hash = _physical_row_hash_matches_or_missing(reuse_row, state_hash)
                if hash_matches:
                    copied_row = reuse_row.to_dict()
                    copied_row["match_id"] = match_id
                    copied_row["action_index"] = action_index
                    copied_row["physical_state_hash"] = state_hash
                    copied_row[PHYSICAL_XPASS_PASS_DISTANCE_COLUMN] = pass_distance
                    copied_rows.append(copied_row)
                    stats["copied_from_reuse"] = int(stats["copied_from_reuse"]) + 1
                    stats["pass_distance_filled"] = int(stats["pass_distance_filled"]) + int(
                        not _has_finite_pass_distance(reuse_row)
                    )
                    match_stats["copied_from_reuse"] += 1
                    match_stats["pass_distance_filled"] += int(not _has_finite_pass_distance(reuse_row))
                    continue
                stats["hash_mismatch_recomputed"] = int(stats["hash_mismatch_recomputed"]) + 1
                match_stats["hash_mismatch_recomputed"] += 1

            stats["cache_misses"] = int(stats["cache_misses"]) + 1
            match_stats["cache_misses"] += 1
            miss_key = (match_id, action_index)
            if miss_key in seen_miss_keys:
                continue
            seen_miss_keys.add(miss_key)
            misses.append(
                {
                    "match_id": match_id,
                    "action_index": action_index,
                    "physical_state_hash": state_hash,
                    PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: pass_distance,
                    "graph": graph,
                }
            )

    if verbose_status:
        for match_id, match_stats in sorted(match_stats_by_id.items()):
            print(
                "physical xPass "
                f"{match_id}: rows={int(match_stats.get('rows_scanned', 0))} "
                f"passes={int(match_stats.get('pass_rows', 0))} "
                f"hits={int(match_stats.get('cache_hits', 0))} "
                f"misses={int(match_stats.get('cache_misses', 0))} "
                f"to_compute={int(match_stats.get('cache_misses', 0))} "
                f"copied={int(match_stats.get('copied_from_reuse', 0))}"
            )

    if dry_run:
        stats["matches"] = {match_id: dict(match_stats) for match_id, match_stats in sorted(match_stats_by_id.items())}
        return stats

    if copied_rows:
        copied_by_match: dict[str, list[dict[str, Any]]] = {}
        for row in copied_rows:
            copied_by_match.setdefault(str(row["match_id"]), []).append(row)
        for match_id, match_rows in copied_by_match.items():
            frame = pd.DataFrame(match_rows)
            frame = frame.drop_duplicates(subset=["action_index"], keep="last")
            _write_runtime_physical_xpass_rows(cache_dir, match_id, frame)
            written = int(len(frame))
            stats["cache_written"] = int(stats["cache_written"]) + written
            match_stats = match_stats_by_id.setdefault(match_id, _runtime_match_stats_template())
            match_stats["cache_written"] += written

    if misses:
        worker_count = min(int(resolved_workers), len(misses))
        chunks = _chunk_items_by_size(misses, int(physical_batch_size))
        task_template = {
            "source": source,
            "eps": float(eps),
            "teammate_policy": teammate_policy,
            "speed_aggregation": speed_aggregation,
            "physical_batch_size": int(physical_batch_size),
            "max_speed": max_speed,
            "speed_step": speed_step,
            "coarse_n_angles": int(coarse_n_angles),
            "refine_top_k_angles": int(refine_top_k_angles),
            "refine_angle_radius": float(refine_angle_radius),
            "angle_step": float(angle_step),
        }
        progress = tqdm(
            total=len(misses),
            desc=progress_desc or "physical xPass runtime",
            leave=False,
            disable=not show_progress,
        )

        def handle_chunk_result(result: dict[str, object]) -> None:
            rows = list(result.get("rows") or [])
            progress.update(int(result.get("computed", len(rows)) or 0))
            if not rows:
                return
            skipped_all_nan = int(result.get("skipped_all_nan", 0) or 0)
            rows_by_match: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                rows_by_match.setdefault(str(row["match_id"]), []).append(row)
            for match_id, match_rows in rows_by_match.items():
                frame = pd.DataFrame(match_rows)
                frame = frame.drop_duplicates(subset=["action_index"], keep="last")
                _write_runtime_physical_xpass_rows(cache_dir, match_id, frame)
                written = int(len(frame))
                stats["cache_written"] = int(stats["cache_written"]) + written
                stats["online_graphs"] = int(stats["online_graphs"]) + written
                match_stats = match_stats_by_id.setdefault(match_id, _runtime_match_stats_template())
                match_stats["cache_written"] += written
                match_stats["online_graphs"] += written
            stats["compute_chunks"] = int(stats["compute_chunks"]) + 1
            stats["skipped_all_nan"] = int(stats["skipped_all_nan"]) + skipped_all_nan
            for match_id, match_rows in rows_by_match.items():
                match_stats = match_stats_by_id.setdefault(str(match_id), _runtime_match_stats_template())
                match_stats["compute_chunks"] += 1
                match_stats["skipped_all_nan"] += sum(
                    1 for row in match_rows if not _physical_xpass_row_has_finite_metric(row)
                )

        if worker_count == 1:
            for chunk in chunks:
                result = _compute_runtime_physical_xpass_chunk({**task_template, "misses": chunk})
                handle_chunk_result(result)
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=configure_physical_worker_thread_limit,
                initargs=(int(worker_thread_limit),),
            ) as executor:
                futures = [
                    executor.submit(_compute_runtime_physical_xpass_chunk, {**task_template, "misses": chunk})
                    for chunk in chunks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    handle_chunk_result(result)
        progress.close()

    stats["matches"] = {match_id: dict(match_stats) for match_id, match_stats in sorted(match_stats_by_id.items())}
    return stats


def attach_physical_xpass_cached_online_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    cache_dir: str | Path,
    match_id: str,
    source: str,
    eps: float = 1e-4,
    floor: float | None = None,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    refresh: bool = False,
    num_workers: str | int = "auto",
    worker_thread_limit: int = 1,
    physical_batch_size: int = 16,
    require_observed_target: bool = False,
    reuse_cache_dir: str | Path | None = None,
    max_speed: float | None = None,
    speed_step: float | None = None,
    coarse_n_angles: int = AS_DEFAULT_COARSE_N_ANGLES,
    refine_top_k_angles: int = AS_DEFAULT_REFINE_TOP_K_ANGLES,
    refine_angle_radius: float = AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    angle_step: float = AS_DEFAULT_ANGLE_STEP_DEG,
    metric: str | None = None,
) -> tuple[list[Data], dict[str, object]]:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    teammate_policy = teammate_policy or (
        PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
        if source == PHYSICAL_XPASS_LEGACY_SOURCE
        else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    )
    stats = prewarm_physical_xpass_runtime_cache(
        [{"match_id": match_id, "graphs": graphs, "labels": labels}],
        cache_dir=cache_dir,
        source=source,
        eps=eps,
        teammate_policy=teammate_policy,
        speed_aggregation=speed_aggregation,
        refresh=refresh,
        num_workers=num_workers,
        worker_thread_limit=worker_thread_limit,
        physical_batch_size=physical_batch_size,
        reuse_cache_dir=reuse_cache_dir,
        max_speed=max_speed,
        speed_step=speed_step,
        coarse_n_angles=coarse_n_angles,
        refine_top_k_angles=refine_top_k_angles,
        refine_angle_radius=refine_angle_radius,
        angle_step=angle_step,
    )
    physical_rows = load_physical_xpass_match(cache_dir, match_id)
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                physical_rows,
                match_id=match_id,
                eps=eps,
                floor=floor,
                require_observed_target=require_observed_target,
                metric=metric,
            )
        )
    return attached, stats


def attach_physical_xpass_to_graph(
    graph: Data,
    labels: torch.Tensor,
    physical_rows: pd.DataFrame,
    *,
    match_id: str,
    eps: float = 1e-4,
    floor: float | None = None,
    require_observed_target: bool = True,
    metric: str | None = None,
) -> Data:
    if physical_rows is None:
        raise ValueError("physical_rows must be provided when attaching physical xPass.")
    action_index = int(labels[LABEL_INDEX["action_index"]].item())
    if action_index not in physical_rows.index:
        raise FileNotFoundError(
            f"Physical xPass sidecar for match {match_id} has no row for action_index={action_index}. "
            "Run scripts/generate_physical_xpass.py for the full feature run."
        )

    row = physical_rows.loc[action_index]
    node_ids = _node_ids(graph)
    selected_metric = normalize_physical_xpass_metric(metric)
    probs = torch.full((len(node_ids),), PHYSICAL_XPASS_NEUTRAL_PROB, dtype=torch.float32)
    missing_columns = []
    for node_index, node_id in enumerate(node_ids):
        value_column = physical_xpass_metric_column(str(node_id), selected_metric)
        if value_column not in row.index:
            missing_columns.append(node_id)
            continue
        value = row[value_column]
        if pd.isna(value):
            continue
        probs[node_index] = float(value)

    if missing_columns and require_observed_target:
        target_index = int(labels[LABEL_INDEX["intent_index"]].item())
        if 0 <= target_index < len(node_ids) and node_ids[target_index] in missing_columns:
            raise ValueError(
                f"Physical xPass sidecar for match {match_id}, action_index={action_index} is missing "
                f"observed target player column {node_ids[target_index]!r}."
            )

    lower_bound = _physical_xpass_lower_bound(eps, floor)
    probs = torch.clamp(probs, lower_bound, 1.0 - float(eps))
    logits = probability_to_logit(probs, eps=eps)
    pass_distances = graph_pass_distances(graph)
    if not torch.isfinite(probs).all() or not torch.isfinite(logits).all():
        raise ValueError(f"Physical xPass values are non-finite after clipping for match {match_id}, action_index={action_index}.")

    if require_observed_target:
        target_index = int(labels[LABEL_INDEX["intent_index"]].item())
        if 0 <= target_index < len(node_ids):
            target_column = physical_xpass_metric_column(str(node_ids[target_index]), selected_metric)
            target_value = row[target_column] if target_column in row.index else np.nan
            if pd.isna(target_value):
                raise ValueError(
                    f"Physical xPass sidecar has no finite probability for observed target {node_ids[target_index]!r} "
                    f"in match {match_id}, action_index={action_index}."
                )

    setattr(graph, PHYSICAL_XPASS_PROB_ATTR, probs)
    setattr(graph, PHYSICAL_XPASS_LOGIT_ATTR, logits)
    setattr(graph, PHYSICAL_XPASS_DISTANCE_ATTR, pass_distances)
    return graph


def attach_physical_xpass_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    cache_dir: str | Path,
    match_id: str,
    eps: float = 1e-4,
    floor: float | None = None,
    require_observed_target: bool = True,
    metric: str | None = None,
) -> list[Data]:
    rows = load_physical_xpass_match(cache_dir, match_id)
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                rows,
                match_id=match_id,
                eps=eps,
                floor=floor,
                require_observed_target=require_observed_target,
                metric=metric,
            )
        )
    return attached


def attach_physical_xpass_read_only_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    cache_dir: str | Path,
    match_id: str,
    eps: float = 1e-4,
    floor: float | None = None,
    require_observed_target: bool = True,
    metric: str | None = None,
) -> list[Data]:
    rows = load_physical_xpass_match(cache_dir, match_id)
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        action_index = int(label[LABEL_INDEX["action_index"]].item())
        if action_index not in rows.index:
            raise FileNotFoundError(
                f"Physical xPass runtime cache for match {match_id} has no row for action_index={action_index}. "
                "Runtime prewarm should have written this row before inference."
            )
        row = rows.loc[action_index]
        cached_hash = row.get("physical_state_hash", None)
        if pd.isna(cached_hash):
            raise ValueError(
                f"Physical xPass runtime cache for match {match_id}, action_index={action_index} "
                "is missing physical_state_hash."
            )
        state_hash = physical_state_hash(graph)
        if str(cached_hash) != state_hash:
            raise ValueError(
                f"Physical xPass runtime cache hash mismatch for match {match_id}, action_index={action_index}; "
                "runtime prewarm and inference graph filtering are not aligned."
            )
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                rows,
                match_id=match_id,
                eps=eps,
                floor=floor,
                require_observed_target=require_observed_target,
                metric=metric,
            )
        )
    return attached


def attach_physical_xpass_online_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    source: str,
    eps: float = 1e-4,
    floor: float | None = None,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    require_observed_target: bool = False,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
    metric: str | None = None,
) -> list[Data]:
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        action_index = int(label[LABEL_INDEX["action_index"]].item())
        probs = compute_graph_physical_xpass_for_source(
            graph,
            source=source,
            eps=eps,
            teammate_policy=teammate_policy,
            speed_aggregation=speed_aggregation,
            simulate_passes_fn=simulate_passes_fn,
            use_progress_bar=use_progress_bar,
            chunk_size=chunk_size,
        )
        row = {"action_index": action_index}
        row[PHYSICAL_XPASS_PASS_DISTANCE_COLUMN] = observed_pass_distance(graph, label)
        row.update({str(player_id): float(value) for player_id, value in probs.items()})
        physical_rows = pd.DataFrame([row]).set_index("action_index", drop=False)
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                physical_rows,
                match_id="<runtime>",
                eps=eps,
                floor=floor,
                require_observed_target=require_observed_target,
                metric=metric,
            )
        )
    return attached
