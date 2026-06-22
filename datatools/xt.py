from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit
from socceraction.spadl import config as spadlconfig
from socceraction.spadl import play_left_to_right
from socceraction.xthreat import ExpectedThreat

from datatools import config, utils
from project_config import XT_DIR, XT_MATCH_DIR

XT_GRID_L = 12
XT_GRID_W = 8
XT_GRID_COLUMNS = [f"X{i}" for i in range(XT_GRID_L)]
XT_SOURCE_GRID_FILENAME = "xT_source_grid.csv"
XT_GLM_TARGET_EPS = 1e-6
XT_GLM_X_BINS = 12
XT_GLM_CENTRALITY_BINS = 8
XT_GLM_MONOTONIC_GRID_SIZE = 101
XT_GLM_MONOTONIC_TOL = 1e-10
XT_GLM_OPTIMIZER_MAXITER = 1000
XT_GLM_OPTIMIZER_FTOL = 1e-10
XT_SURFACE_X_POINTS = 105
XT_SURFACE_Y_POINTS = 68
XT_GLM_BASE_TERMS = ("normalized_x", "normalized_centrality")
XT_GLM_INTERACTION_TERM = "normalized_x_centrality"
XT_GLM_NONLINEAR_TERMS = {
    "x": ("normalized_x_squared", "normalized_x_cubed"),
    "y": ("normalized_centrality_squared", "normalized_centrality_cubed"),
}


@dataclass(frozen=True)
class XTXYLogitModel:
    coefficients: dict[str, float]
    terms: tuple[str, ...] = XT_GLM_BASE_TERMS

    def predict_features(self, normalized_x: np.ndarray, normalized_centrality: np.ndarray) -> np.ndarray:
        features = build_xt_glm_design_frame(
            normalized_x,
            normalized_centrality,
            terms=self.terms,
        )
        linear = np.full(len(features), float(self.coefficients.get("intercept", 0.0)), dtype=float)
        for term in self.terms:
            linear += float(self.coefficients.get(term, 0.0)) * features[term].to_numpy(dtype=float)
        values = expit(linear)
        return np.clip(values, 0.0, 1.0)

    def predict_xy(self, start_x: pd.Series | np.ndarray, start_y: pd.Series | np.ndarray) -> np.ndarray:
        normalized_x, normalized_centrality = normalized_xt_xy_features(start_x, start_y)
        return self.predict_features(normalized_x, normalized_centrality)

    def to_metadata(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.coefficients.items()}


@dataclass(frozen=True)
class XTXYGLMFit:
    source_grid: np.ndarray
    projection_grid: np.ndarray
    model: XTXYLogitModel
    fit_sample: pd.DataFrame
    diagnostics: dict[str, object]


def sort_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    order_cols = [col for col in ["action_id", "period_id", "seconds", "original_event_id"] if col in events.columns]
    if order_cols:
        events = events.sort_values(order_cols).reset_index(drop=True)
    return events


def infer_home_team_id(events: pd.DataFrame) -> str:
    home_team_ids = events.loc[events["object_id"].astype(str).str.startswith("home"), "team_id"].dropna().unique().tolist()
    if len(home_team_ids) != 1:
        raise ValueError(f"Expected exactly one home team id, found {home_team_ids}.")
    return str(home_team_ids[0])


def spadl_result_id(spadl_type: str, success: bool, offside: bool) -> int:
    if offside:
        return spadlconfig.results.index("offside")
    if spadl_type == "shot" and not success:
        return spadlconfig.results.index("fail")
    return spadlconfig.results.index("success" if success else "fail")


def build_xt_actions(events: pd.DataFrame) -> pd.DataFrame:
    events = sort_events(utils.sanitize_expected_goal(events))
    xt_actions = events.loc[events["spadl_type"].isin(config.XT_ACTION_TYPES)].copy()

    if xt_actions.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "original_event_id",
                "action_id",
                "period_id",
                "seconds",
                "team_id",
                "player_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "type_id",
                "result_id",
                "bodypart_id",
                "spadl_type",
                "success",
                "offside",
                "xG",
            ]
        )

    xt_actions["game_id"] = xt_actions["stats_perform_match_id"] if "stats_perform_match_id" in xt_actions.columns else xt_actions["game_id"]
    xt_actions["team_id"] = xt_actions["team_id"].astype(str)
    xt_actions["type_id"] = xt_actions["spadl_type"].map(spadlconfig.actiontypes.index)
    xt_actions["result_id"] = xt_actions.apply(
        lambda row: spadl_result_id(row["spadl_type"], bool(row["success"]), bool(row.get("offside", False))),
        axis=1,
    )
    xt_actions["bodypart_id"] = spadlconfig.bodyparts.index("foot")
    xt_actions["xG"] = xt_actions["expected_goal"].astype(float)
    return xt_actions[
        [
            "game_id",
            "original_event_id",
            "action_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_id",
            "result_id",
            "bodypart_id",
            "spadl_type",
            "success",
            "offside",
            "xG",
        ]
    ].copy()


def rotate_xt_actions(xt_actions: pd.DataFrame, home_team_id: str) -> pd.DataFrame:
    if xt_actions.empty:
        return xt_actions.copy()

    rotated = play_left_to_right(
        xt_actions[
            [
                "game_id",
                "original_event_id",
                "action_id",
                "period_id",
                "seconds",
                "team_id",
                "player_id",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "type_id",
                "result_id",
                "bodypart_id",
            ]
        ].copy(),
        home_team_id=home_team_id,
    ).reset_index(drop=True)
    passthrough_cols = xt_actions.drop(
        columns=[
            "game_id",
            "original_event_id",
            "action_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_id",
            "result_id",
            "bodypart_id",
        ]
    )
    return rotated.join(passthrough_cols.reset_index(drop=True))


def symmetrize_grid(grid: np.ndarray) -> np.ndarray:
    symmetric = np.asarray(grid, dtype=float).copy()
    for upper in range(symmetric.shape[0] // 2):
        lower = symmetric.shape[0] - 1 - upper
        averaged = (symmetric[upper] + symmetric[lower]) / 2.0
        symmetric[upper] = averaged
        symmetric[lower] = averaged
    return symmetric


def validate_xt_grid_shape(grid: np.ndarray, *, context: str = "xT grid") -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(f"{context} must be a non-empty 2D matrix, found shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite values.")
    return values


def fit_xt_surface(rotated_train_actions: pd.DataFrame, grid_l: int = XT_GRID_L, grid_w: int = XT_GRID_W) -> np.ndarray:
    if grid_l < 1 or grid_w < 1:
        raise ValueError(f"Source xT grid dimensions must be positive, found l={grid_l}, w={grid_w}.")
    model = ExpectedThreat(l=int(grid_l), w=int(grid_w))
    model.fit(rotated_train_actions)
    return symmetrize_grid(model.xT)


def normalized_xt_xy_features(
    start_x: pd.Series | np.ndarray,
    start_y: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(pd.Series(start_x), errors="coerce").clip(0.0, config.FIELD_SIZE[0])
    y = pd.to_numeric(pd.Series(start_y), errors="coerce").clip(0.0, config.FIELD_SIZE[1])

    normalized_x = (x / config.FIELD_SIZE[0]).to_numpy(dtype=float)
    normalized_y = (y / config.FIELD_SIZE[1]).to_numpy(dtype=float)
    normalized_centrality = np.sin(np.pi * normalized_y)
    return normalized_x, np.clip(normalized_centrality, 0.0, 1.0)


def resolve_xt_glm_terms(
    *,
    use_interaction: bool = False,
    nonlinear_axes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    terms = list(XT_GLM_BASE_TERMS)
    if use_interaction:
        terms.append(XT_GLM_INTERACTION_TERM)

    seen_axes: set[str] = set()
    for axis in nonlinear_axes or []:
        normalized_axis = str(axis).strip().lower()
        if normalized_axis not in XT_GLM_NONLINEAR_TERMS:
            raise ValueError(f"Unsupported xT GLM nonlinear axis {axis!r}. Expected one of: x, y.")
        if normalized_axis in seen_axes:
            continue
        seen_axes.add(normalized_axis)
        terms.extend(XT_GLM_NONLINEAR_TERMS[normalized_axis])
    return tuple(terms)


def build_xt_glm_design_frame(
    normalized_x: pd.Series | np.ndarray,
    normalized_centrality: pd.Series | np.ndarray,
    *,
    terms: tuple[str, ...],
) -> pd.DataFrame:
    x = pd.to_numeric(pd.Series(normalized_x), errors="coerce").to_numpy(dtype=float)
    centrality = pd.to_numeric(pd.Series(normalized_centrality), errors="coerce").to_numpy(dtype=float)
    frame = pd.DataFrame(
        {
            "normalized_x": x,
            "normalized_centrality": centrality,
            XT_GLM_INTERACTION_TERM: x * centrality,
            "normalized_x_squared": x**2,
            "normalized_x_cubed": x**3,
            "normalized_centrality_squared": centrality**2,
            "normalized_centrality_cubed": centrality**3,
        }
    )
    missing_terms = [term for term in terms if term not in frame.columns]
    if missing_terms:
        raise ValueError(f"Unsupported xT GLM terms: {missing_terms}.")
    return frame.loc[:, list(terms)].copy()


def xt_glm_formula(terms: tuple[str, ...]) -> str:
    return "logit(xT) = intercept + " + " + ".join(terms)


def check_xt_glm_monotonicity(
    model: XTXYLogitModel,
    *,
    grid_size: int = XT_GLM_MONOTONIC_GRID_SIZE,
    tolerance: float = XT_GLM_MONOTONIC_TOL,
) -> dict[str, object]:
    grid = np.linspace(0.0, 1.0, int(grid_size))
    xx, cc = np.meshgrid(grid, grid)
    predictions = model.predict_features(xx.ravel(), cc.ravel()).reshape(int(grid_size), int(grid_size))
    x_diffs = np.diff(predictions, axis=1)
    centrality_diffs = np.diff(predictions, axis=0)
    x_violations = x_diffs < -float(tolerance)
    centrality_violations = centrality_diffs < -float(tolerance)
    result = {
        "passed": not bool(x_violations.any() or centrality_violations.any()),
        "grid_size": int(grid_size),
        "tolerance": float(tolerance),
        "x_violation_count": int(x_violations.sum()),
        "centrality_violation_count": int(centrality_violations.sum()),
        "min_x_diff": float(x_diffs.min()) if x_diffs.size else 0.0,
        "min_centrality_diff": float(centrality_diffs.min()) if centrality_diffs.size else 0.0,
    }
    return result


def validate_xt_glm_monotonicity(model: XTXYLogitModel) -> dict[str, object]:
    result = check_xt_glm_monotonicity(model)
    if not result["passed"]:
        raise ValueError(
            "x/y xT GLM violated monotonicity: "
            f"x_violations={result['x_violation_count']}, "
            f"centrality_violations={result['centrality_violation_count']}, "
            f"min_x_diff={result['min_x_diff']:.6g}, "
            f"min_centrality_diff={result['min_centrality_diff']:.6g}."
        )
    return result


def build_xt_glm_monotonicity_constraint_matrix(
    terms: tuple[str, ...],
    *,
    grid_size: int = XT_GLM_MONOTONIC_GRID_SIZE,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, int(grid_size))
    xx, cc = np.meshgrid(grid, grid)
    x_values = xx.ravel()
    centrality_values = cc.ravel()
    param_names = ("intercept",) + tuple(terms)
    rows: list[np.ndarray] = []

    for derivative_axis in ("x", "centrality"):
        for x, centrality in zip(x_values, centrality_values, strict=True):
            row = np.zeros(len(param_names), dtype=float)
            for index, term in enumerate(param_names):
                if derivative_axis == "x":
                    if term == "normalized_x":
                        row[index] = 1.0
                    elif term == XT_GLM_INTERACTION_TERM:
                        row[index] = centrality
                    elif term == "normalized_x_squared":
                        row[index] = 2.0 * x
                    elif term == "normalized_x_cubed":
                        row[index] = 3.0 * x**2
                else:
                    if term == "normalized_centrality":
                        row[index] = 1.0
                    elif term == XT_GLM_INTERACTION_TERM:
                        row[index] = x
                    elif term == "normalized_centrality_squared":
                        row[index] = 2.0 * centrality
                    elif term == "normalized_centrality_cubed":
                        row[index] = 3.0 * centrality**2
            rows.append(row)

    return np.vstack(rows)


def compute_spatial_bin_weights(
    normalized_x: np.ndarray,
    normalized_centrality: np.ndarray,
    *,
    x_bins: int = XT_GLM_X_BINS,
    centrality_bins: int = XT_GLM_CENTRALITY_BINS,
) -> np.ndarray:
    x_values = np.asarray(normalized_x, dtype=float)
    centrality_values = np.asarray(normalized_centrality, dtype=float)
    valid_mask = np.isfinite(x_values) & np.isfinite(centrality_values)
    weights = np.zeros(len(x_values), dtype=float)
    if not valid_mask.any():
        return weights

    x_bin = np.floor(np.clip(x_values[valid_mask], 0.0, 1.0 - np.finfo(float).eps) * x_bins).astype(int)
    centrality_bin = np.floor(
        np.clip(centrality_values[valid_mask], 0.0, 1.0 - np.finfo(float).eps) * centrality_bins
    ).astype(int)
    keys = pd.MultiIndex.from_arrays([x_bin, centrality_bin])
    counts = pd.Series(1, index=keys).groupby(level=[0, 1]).transform("sum").to_numpy(dtype=float)
    populated_bins = int(keys.nunique())
    valid_count = int(valid_mask.sum())
    weights[valid_mask] = valid_count / (populated_bins * counts)
    return weights


def _weighted_logit_intercept(y: np.ndarray, weights: np.ndarray) -> float:
    mean_y = float(np.average(y, weights=weights))
    clipped_mean = float(np.clip(mean_y, XT_GLM_TARGET_EPS, 1.0 - XT_GLM_TARGET_EPS))
    return float(np.log(clipped_mean / (1.0 - clipped_mean)))


def _fallback_xt_glm_initial_params(y: np.ndarray, weights: np.ndarray, terms: tuple[str, ...]) -> np.ndarray:
    params = np.zeros(len(terms) + 1, dtype=float)
    params[0] = _weighted_logit_intercept(y, weights)
    for index, term in enumerate(terms, start=1):
        if term in XT_GLM_BASE_TERMS:
            params[index] = 1e-3
    return params


def _unconstrained_xt_glm_initial_params(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    terms: tuple[str, ...],
    constraint_matrix: np.ndarray,
) -> np.ndarray:
    fallback = _fallback_xt_glm_initial_params(y, weights, terms)
    try:
        result = sm.GLM(
            y,
            design,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
            freq_weights=weights,
        ).fit()
    except Exception:
        return fallback

    params = np.asarray(result.params, dtype=float)
    if params.shape != fallback.shape or not np.isfinite(params).all():
        return fallback
    if (constraint_matrix @ params < -XT_GLM_MONOTONIC_TOL).any():
        return fallback
    return params


def fit_constrained_xt_glm_params(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    terms: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    constraint_matrix = build_xt_glm_monotonicity_constraint_matrix(terms)
    initial_params = _unconstrained_xt_glm_initial_params(design, y, weights, terms, constraint_matrix)

    def objective(params: np.ndarray) -> float:
        linear = design @ params
        return float(np.sum(weights * (np.logaddexp(0.0, linear) - y * linear)))

    def gradient(params: np.ndarray) -> np.ndarray:
        linear = design @ params
        probabilities = expit(linear)
        return design.T @ (weights * (probabilities - y))

    constraint = LinearConstraint(constraint_matrix, lb=0.0, ub=np.inf)
    result = minimize(
        objective,
        initial_params,
        method="SLSQP",
        jac=gradient,
        constraints=[constraint],
        options={"maxiter": XT_GLM_OPTIMIZER_MAXITER, "ftol": XT_GLM_OPTIMIZER_FTOL, "disp": False},
    )

    diagnostics = {
        "method": "SLSQP",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "objective": float(result.fun) if np.isfinite(result.fun) else None,
        "maxiter": XT_GLM_OPTIMIZER_MAXITER,
        "ftol": XT_GLM_OPTIMIZER_FTOL,
        "constraint_grid_size": XT_GLM_MONOTONIC_GRID_SIZE,
        "constraint_tolerance": XT_GLM_MONOTONIC_TOL,
        "constraint_count": int(constraint_matrix.shape[0]),
    }
    if not result.success:
        raise ValueError(f"x/y xT GLM constrained optimizer failed: {result.message}")
    return np.asarray(result.x, dtype=float), diagnostics


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    valid = pd.concat([a, b], axis=1).dropna()
    if len(valid) < 2:
        return None
    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
    return None if pd.isna(corr) else float(corr)


def _summarize_residuals(residuals: pd.Series) -> dict[str, float]:
    finite = pd.to_numeric(residuals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "p05": float(finite.quantile(0.05)),
        "median": float(finite.median()),
        "p95": float(finite.quantile(0.95)),
        "max": float(finite.max()),
    }


def build_xt_glm_fit_sample(
    rotated_train_actions: pd.DataFrame,
    source_grid: np.ndarray,
    *,
    terms: tuple[str, ...] = XT_GLM_BASE_TERMS,
) -> pd.DataFrame:
    sample = rotated_train_actions.copy()
    sample["source_grid_xT"] = zone_value_from_grid(sample["start_x"], sample["start_y"], source_grid)
    normalized_x, normalized_centrality = normalized_xt_xy_features(sample["start_x"], sample["start_y"])
    sample["normalized_x"] = normalized_x
    sample["normalized_centrality"] = normalized_centrality
    design_features = build_xt_glm_design_frame(normalized_x, normalized_centrality, terms=terms)
    for term in terms:
        if term not in sample.columns:
            sample[term] = design_features[term].to_numpy(dtype=float)
    valid_mask = (
        sample["source_grid_xT"].notna()
        & np.isfinite(sample["normalized_x"].to_numpy(dtype=float))
        & np.isfinite(sample["normalized_centrality"].to_numpy(dtype=float))
    )
    sample = sample.loc[valid_mask].copy().reset_index(drop=True)
    if sample.empty:
        raise ValueError("No valid xT GLM calibration rows were available after assigning source grid values.")

    sample["weight"] = compute_spatial_bin_weights(
        sample["normalized_x"].to_numpy(dtype=float),
        sample["normalized_centrality"].to_numpy(dtype=float),
    )
    return sample


def fit_xt_xy_glm(
    rotated_train_actions: pd.DataFrame,
    source_grid: np.ndarray,
    *,
    use_interaction: bool = False,
    nonlinear_axes: Iterable[str] | None = None,
) -> XTXYGLMFit:
    terms = resolve_xt_glm_terms(use_interaction=use_interaction, nonlinear_axes=nonlinear_axes)
    fit_sample = build_xt_glm_fit_sample(rotated_train_actions, source_grid, terms=terms)
    y = fit_sample["source_grid_xT"].clip(XT_GLM_TARGET_EPS, 1.0 - XT_GLM_TARGET_EPS).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(fit_sample), dtype=float), fit_sample[list(terms)].to_numpy(dtype=float)])
    weights = fit_sample["weight"].to_numpy(dtype=float)
    params, optimizer_diagnostics = fit_constrained_xt_glm_params(design, y, weights, terms)
    coefficients = {"intercept": float(params[0])}
    coefficients.update({term: float(params[index + 1]) for index, term in enumerate(terms)})
    model = XTXYLogitModel(coefficients=coefficients, terms=terms)
    monotonicity_check = check_xt_glm_monotonicity(model)
    monotonicity_error = None
    if not monotonicity_check["passed"]:
        monotonicity_error = (
            "x/y xT GLM violated monotonicity: "
            f"x_violations={monotonicity_check['x_violation_count']}, "
            f"centrality_violations={monotonicity_check['centrality_violation_count']}, "
            f"min_x_diff={monotonicity_check['min_x_diff']:.6g}, "
            f"min_centrality_diff={monotonicity_check['min_centrality_diff']:.6g}."
        )

    fit_sample["glm_xT"] = model.predict_features(
        fit_sample["normalized_x"].to_numpy(dtype=float),
        fit_sample["normalized_centrality"].to_numpy(dtype=float),
    )
    fit_sample["residual"] = fit_sample["glm_xT"] - fit_sample["source_grid_xT"]
    diagnostics = build_xt_glm_diagnostics(fit_sample, optimizer_diagnostics)
    diagnostics["monotonicity_check"] = monotonicity_check
    diagnostics["monotonicity_error"] = monotonicity_error
    diagnostics["monotonicity_error_raised"] = monotonicity_error is not None
    return XTXYGLMFit(
        source_grid=np.asarray(source_grid, dtype=float),
        projection_grid=project_xt_glm_to_grid(model),
        model=model,
        fit_sample=fit_sample,
        diagnostics=diagnostics,
    )


def build_xt_glm_diagnostics(fit_sample: pd.DataFrame, optimizer_diagnostics: dict[str, object]) -> dict[str, object]:
    residual = fit_sample["residual"]
    rmse = float(np.sqrt(np.mean(np.square(residual.to_numpy(dtype=float)))))
    diagnostics: dict[str, object] = {
        "fit_samples": int(len(fit_sample)),
        "weighted_fit_samples": float(fit_sample["weight"].sum()),
        "rmse_vs_source_grid_xt": rmse,
        "correlation_vs_source_grid_xt": _safe_corr(fit_sample["source_grid_xT"], fit_sample["glm_xT"]),
        "residual_summary": _summarize_residuals(residual),
        "optimizer": optimizer_diagnostics,
    }

    if "spadl_type" in fit_sample.columns:
        by_type: dict[str, dict[str, object]] = {}
        for spadl_type, group in fit_sample.groupby("spadl_type"):
            group_residual = group["glm_xT"] - group["source_grid_xT"]
            by_type[str(spadl_type)] = {
                "count": int(len(group)),
                "rmse_vs_source_grid_xt": float(np.sqrt(np.mean(np.square(group_residual.to_numpy(dtype=float))))),
                "correlation_vs_source_grid_xt": _safe_corr(group["source_grid_xT"], group["glm_xT"]),
                "residual_summary": _summarize_residuals(group_residual),
            }
        diagnostics["by_spadl_type"] = by_type

    shot_sample = fit_sample.loc[fit_sample.get("spadl_type", pd.Series(dtype=object)).eq("shot")].copy()
    if not shot_sample.empty and "xG" in shot_sample.columns:
        shot_xg = pd.to_numeric(shot_sample["xG"], errors="coerce")
        valid_shots = shot_sample.loc[shot_xg.notna()].copy()
        if not valid_shots.empty:
            valid_xg = pd.to_numeric(valid_shots["xG"], errors="coerce")
            diagnostics["shot_xg_diagnostic"] = {
                "count": int(len(valid_shots)),
                "rmse_glm_xt_vs_xg": float(
                    np.sqrt(np.mean(np.square((valid_shots["glm_xT"] - valid_xg).to_numpy(dtype=float))))
                ),
                "correlation_glm_xt_vs_xg": _safe_corr(valid_shots["glm_xT"], valid_xg),
            }
    return diagnostics


def project_xt_glm_to_grid(model: XTXYLogitModel, grid_l: int = XT_GRID_L, grid_w: int = XT_GRID_W) -> np.ndarray:
    if grid_l < 1 or grid_w < 1:
        raise ValueError(f"Projection xT grid dimensions must be positive, found l={grid_l}, w={grid_w}.")
    projected = np.zeros((int(grid_w), int(grid_l)), dtype=float)
    cell_length = config.FIELD_SIZE[0] / int(grid_l)
    cell_width = config.FIELD_SIZE[1] / int(grid_w)
    for row_index in range(int(grid_w)):
        y_index = int(grid_w) - 1 - row_index
        y = (y_index + 0.5) * cell_width
        for x_index in range(int(grid_l)):
            x = (x_index + 0.5) * cell_length
            projected[row_index, x_index] = model.predict_xy(np.array([x]), np.array([y]))[0]
    return projected


def build_xt_xy_surface(model: XTXYLogitModel) -> pd.DataFrame:
    xs = np.arange(XT_SURFACE_X_POINTS, dtype=float) + 0.5
    ys = np.arange(XT_SURFACE_Y_POINTS, dtype=float) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    normalized_x, normalized_centrality = normalized_xt_xy_features(xx.ravel(), yy.ravel())
    surface = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "normalized_x": normalized_x,
            "normalized_centrality": normalized_centrality,
        }
    )
    surface["xT"] = model.predict_features(normalized_x, normalized_centrality)
    return surface


def save_xt_xy_surface_plot(surface: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    output_path = Path(output_path)
    pivot = surface.pivot(index="y", columns="x", values="xT")
    x_values = pivot.columns.to_numpy(dtype=float)
    y_values = pivot.index.to_numpy(dtype=float)
    xx, yy = np.meshgrid(x_values, y_values)
    zz = pivot.to_numpy(dtype=float)

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap=cm.magma, linewidth=0.1, edgecolor="0.35", antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("xT")
    ax.set_title("x/y GLM xT surface")
    ax.view_init(elev=28, azim=-132)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def zone_value_from_grid(start_x: pd.Series, start_y: pd.Series, grid: np.ndarray) -> np.ndarray:
    grid = validate_xt_grid_shape(grid)
    grid_w, grid_l = grid.shape
    x = pd.to_numeric(start_x, errors="coerce").clip(0, config.FIELD_SIZE[0] - 1e-9)
    y = pd.to_numeric(start_y, errors="coerce").clip(0, config.FIELD_SIZE[1] - 1e-9)

    x_index = np.floor((x / config.FIELD_SIZE[0]) * grid_l).astype("Int64").clip(0, grid_l - 1)
    y_index = np.floor((y / config.FIELD_SIZE[1]) * grid_w).astype("Int64").clip(0, grid_w - 1)
    row_index = grid_w - 1 - y_index

    values = np.full(len(x), np.nan, dtype=float)
    valid_mask = x.notna() & y.notna()
    values[valid_mask.to_numpy()] = grid[row_index[valid_mask].astype(int), x_index[valid_mask].astype(int)]
    return values


def annotate_match_xt(events: pd.DataFrame, xt_fit: XTXYGLMFit | XTXYLogitModel | np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_events = sort_events(events)
    sorted_events = utils.sanitize_expected_goal(sorted_events)
    sorted_events["xT"] = np.nan
    sorted_events["xG"] = sorted_events["expected_goal"].astype(float)
    glm_model = xt_fit.model if isinstance(xt_fit, XTXYGLMFit) else xt_fit

    xt_actions = build_xt_actions(sorted_events)
    if not xt_actions.empty:
        home_team_id = infer_home_team_id(sorted_events)
        rotated_actions = rotate_xt_actions(xt_actions, home_team_id)
        if isinstance(glm_model, XTXYLogitModel):
            rotated_actions["xT"] = glm_model.predict_xy(rotated_actions["start_x"], rotated_actions["start_y"])
        else:
            # Backward-compatible fallback for tests or callers that still pass a legacy grid.
            rotated_actions["xT"] = zone_value_from_grid(rotated_actions["start_x"], rotated_actions["start_y"], glm_model)
        shot_mask = rotated_actions["spadl_type"].eq("shot")
        if shot_mask.any():
            rotated_actions.loc[shot_mask, "xT"] = np.maximum(
                rotated_actions.loc[shot_mask, "xT"].to_numpy(dtype=float),
                rotated_actions.loc[shot_mask, "xG"].fillna(0.0).to_numpy(dtype=float),
            )

        sorted_events = sorted_events.merge(
            rotated_actions[["action_id", "xG", "xT"]],
            on="action_id",
            how="left",
            suffixes=("", "_xt"),
        )
        sorted_events["xG"] = sorted_events["xG_xt"].combine_first(sorted_events["xG"])
        sorted_events["xT"] = sorted_events["xT_xt"].combine_first(sorted_events["xT"])
        sorted_events = sorted_events.drop(columns=[c for c in ["xG_xt", "xT_xt"] if c in sorted_events.columns])

    sorted_events = utils.label_xt_returns(sorted_events, lookahead_len=5, eligible_types=tuple(config.XT_ACTION_TYPES))

    user_export = sorted_events.loc[sorted_events["spadl_type"].isin(config.XT_ACTION_TYPES)].copy()
    export_cols = [
        col
        for col in [
            "game_id",
            "stats_perform_match_id",
            "action_id",
            "original_event_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "object_id",
            "spadl_type",
            "success",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "xG",
            "xT",
            "scores_xT",
            "concedes_xT",
        ]
        if col in user_export.columns
    ]
    return sorted_events, user_export[export_cols].copy()


def merge_xt_annotations(events: pd.DataFrame, match_id: str | None, xt_match_dir: str | Path = XT_MATCH_DIR) -> pd.DataFrame:
    events = events.copy()
    if match_id is None:
        return events

    sidecar_path = Path(xt_match_dir) / f"{match_id}.csv"
    if not sidecar_path.exists():
        return events

    xt_columns = ["action_id", "xT", "scores_xT", "concedes_xT"]
    xt_sidecar = pd.read_csv(sidecar_path, usecols=lambda c: c in xt_columns)
    if xt_sidecar.empty:
        return events

    events = events.drop(columns=[c for c in ["xT", "scores_xT", "concedes_xT"] if c in events.columns])
    return events.merge(xt_sidecar, on="action_id", how="left")


def save_xt_outputs(
    all_events: Iterable[pd.DataFrame],
    xt_fit: XTXYGLMFit | np.ndarray,
    fit_metadata: dict,
    output_dir: str | Path = XT_DIR,
    xt_match_dir: str | Path = XT_MATCH_DIR,
) -> None:
    output_dir = Path(output_dir)
    xt_match_dir = Path(xt_match_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xt_match_dir.mkdir(parents=True, exist_ok=True)

    exported_actions: list[pd.DataFrame] = []
    for events in all_events:
        if events.empty:
            continue

        annotated_events, exported_xt = annotate_match_xt(events, xt_fit)
        match_id = str(
            annotated_events["stats_perform_match_id"].iloc[0]
            if "stats_perform_match_id" in annotated_events.columns
            else annotated_events["game_id"].iloc[0]
        )
        sidecar = annotated_events[["action_id", "xG", "xT", "scores_xT", "concedes_xT"]].copy()
        sidecar.to_csv(xt_match_dir / f"{match_id}.csv", index=False)
        exported_actions.append(exported_xt)

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "xT.csv", index=False)
    else:
        pd.DataFrame(columns=["action_id", "xG", "xT", "scores_xT", "concedes_xT"]).to_csv(
            output_dir / "xT.csv", index=False
        )

    grid = xt_fit.projection_grid if isinstance(xt_fit, XTXYGLMFit) else xt_fit
    grid_frame = pd.DataFrame(grid, columns=XT_GRID_COLUMNS)
    grid_frame.to_csv(output_dir / "xT_grid.csv", index=False)

    if isinstance(xt_fit, XTXYGLMFit):
        source_grid_frame = pd.DataFrame(xt_fit.source_grid, columns=[f"X{i}" for i in range(xt_fit.source_grid.shape[1])])
        source_grid_frame.to_csv(output_dir / XT_SOURCE_GRID_FILENAME, index=False)
        surface = build_xt_xy_surface(xt_fit.model)
        surface.to_csv(output_dir / "xT_xy_surface.csv", index=False)
        export_fit_sample = xt_fit.fit_sample[
            [
                col
                for col in [
                    "game_id",
                    "stats_perform_match_id",
                    "original_event_id",
                    "action_id",
                    "period_id",
                    "seconds",
                    "team_id",
                    "player_id",
                    "spadl_type",
                    "success",
                    "start_x",
                    "start_y",
                    "xG",
                    "source_grid_xT",
                    "normalized_x",
                    "normalized_centrality",
                    XT_GLM_INTERACTION_TERM,
                    *XT_GLM_NONLINEAR_TERMS["x"],
                    *XT_GLM_NONLINEAR_TERMS["y"],
                    "glm_xT",
                    "residual",
                    "weight",
                ]
                if col in xt_fit.fit_sample.columns
            ]
        ].copy()
        export_fit_sample.to_csv(output_dir / "xT_glm_fit_sample.csv", index=False)
        save_xt_xy_surface_plot(surface, output_dir / "xT_xy_surface_3d.png")

    (output_dir / "fit_metadata.json").write_text(json.dumps(fit_metadata, indent=2), encoding="utf-8")
