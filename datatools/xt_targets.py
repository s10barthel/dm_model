"""Versioned xT surface selection and action-level target transformations."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import numpy as np
import pandas as pd

from datatools import config
from project_config import parse_return_type


@dataclass(frozen=True)
class LogProgressionSurface:
    k: float = 2.0
    min_goal_distance: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError("Log progression k must be finite and positive.")
        if not np.isfinite(self.min_goal_distance) or not 0 < self.min_goal_distance < self.max_distance:
            raise ValueError("Minimum goal distance must be finite, positive and below maximum goal distance.")

    @property
    def max_distance(self):
        length, width = config.FIELD_SIZE
        return float(np.hypot(length, np.sqrt(self.k) * width / 2))

    def predict_xy(self, start_x, start_y):
        length, width = config.FIELD_SIZE
        x, y = np.broadcast_arrays(np.asarray(start_x, dtype=float), np.asarray(start_y, dtype=float))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Log progression requires finite coordinates.")
        distance = np.hypot(length - np.clip(x, 0, length),
                            np.sqrt(self.k) * (np.clip(y, 0, width) - width / 2))
        distance = np.clip(distance, self.min_goal_distance, self.max_distance)
        return np.clip(np.log(self.max_distance / distance)
                       / np.log(self.max_distance / self.min_goal_distance), 0, 1)


def surface_from_metadata(metadata):
    kind = metadata.get("surface_type", metadata.get("xt_model_type", "xy_logit_glm"))
    if kind == "log_progression":
        if metadata.get("surface_definition_version", 1) != 1:
            raise ValueError("Unsupported log progression surface definition version.")
        return LogProgressionSurface(float(metadata["k"]), float(metadata["min_goal_distance"]))
    if kind != "xy_logit_glm":
        raise ValueError(f"Unsupported xT surface type: {kind!r}.")
    from datatools.xt import XTXYLogitModel, XT_GLM_BASE_TERMS
    return XTXYLogitModel(metadata["coefficients"], tuple(metadata.get("terms", XT_GLM_BASE_TERMS)))


def target_settings(metadata=None):
    saved = (metadata or {}).get("xt_targets") or {}
    representation = saved.get("representation", "absolute")
    clipping = bool(saved.get("clip_negative_gains", False))
    if representation not in {"absolute", "gain"}:
        raise ValueError(f"Unknown xT target representation: {representation!r}.")
    if clipping and representation != "gain":
        raise ValueError("--xt-clip-negative-gains requires --xt-target-representation gain.")
    version = int(saved.get("version", 1))
    if version not in (1, 2):
        raise ValueError("Unsupported xT target definition version")
    return dict(version=version, representation=representation, clip_negative_gains=clipping,
                scoring_baseline="action_start_surface" if representation == "gain" else "zero",
                conceding_baseline=("turnover_start_surface_next" if version == 2 and representation == "gain" else "zero"))


def add_target_arguments(parser):
    parser.add_argument("--xt-target-representation", choices=("absolute", "gain"), default=None,
                        help="Saved xT target representation; extensions inherit the source setting.")
    parser.add_argument("--xt-clip-negative-gains", action=argparse.BooleanOptionalAction, default=None,
                        help="Clip scoring and next-window conceding gains below zero; requires gain representation.")


def configure_targets(args, metadata=None, *, allow_change=False):
    """Resolve unspecified extension options from the source run, then validate."""
    previous = target_settings(metadata)
    representation = getattr(args, "xt_target_representation", None)
    clipping = getattr(args, "xt_clip_negative_gains", None)
    settings = target_settings({"xt_targets": dict(
        version=2 if metadata is None or allow_change else previous["version"],
        representation=representation if representation is not None else previous["representation"],
        clip_negative_gains=(clipping if clipping is not None else
                             previous["clip_negative_gains"] if representation in (None, previous["representation"]) else False))})
    if metadata is not None and settings != previous and not allow_change:
        raise ValueError("Changing xT target settings requires --refresh-target-family xt.")
    args.xt_target_representation = settings["representation"]
    args.xt_clip_negative_gains = settings["clip_negative_gains"]
    args.xt_targets = settings
    return settings


def target_cli(settings):
    settings = target_settings({"xt_targets": settings})
    return ["--xt-target-representation", settings["representation"],
            *(["--xt-clip-negative-gains"] if settings["clip_negative_gains"] else [])]


def unavailable_xt_returns(surface_metadata, return_types):
    if (surface_metadata or {}).get("surface_type", (surface_metadata or {}).get("xt_model_type")) == "log_progression":
        return [r for r in return_types if parse_return_type(r)[0] not in {"next", "next_plus", "in", "extended_possession"}]
    return []


def validate_surface_returns(surface_metadata, return_types):
    unsupported = unavailable_xt_returns(surface_metadata, return_types)
    if unsupported:
        raise ValueError(f"Log progression supports next_N[+][_skip1], extended_possession, and in_N returns only: {unsupported}.")


def transform_action_targets(actions, pairs, settings=None, surface=None, *, events=None, return_type=None, stop_condition=None):
    settings = target_settings({"xt_targets": settings or {}})
    if settings["version"] == 2 and settings["representation"] == "gain" and return_type is None:
        raise ValueError("Version-2 xT gains require an explicit return type")
    conceding_baselines = None
    kind = parse_return_type(return_type)[0] if return_type else None
    if kind == "next_plus" or (settings["version"] == 2 and settings["representation"] == "gain"
                               and kind in {"next", "extended_possession"}):
        if events is None:
            raise ValueError("Turnover targets require event context")
        from datatools.return_windows import action_xt_windows
        raw, conceding_baselines = action_xt_windows(actions, events, return_type, surface, stop_condition)
        pairs = pd.DataFrame(raw, index=actions.index, columns=pairs.columns)
    if settings["representation"] == "absolute":
        return pairs
    if surface is None:
        raise ValueError("Gain targets require a pinned xT surface version.")
    teams = actions.object_id.astype(str).str[:4]
    x = pd.to_numeric(actions.start_x, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(actions.start_y, errors="coerce").to_numpy(dtype=float)
    if (not teams.isin(["home", "away"]).all() or not np.isfinite(x).all()
            or not np.isfinite(y).all() or (x == -config.FIELD_SIZE[0]).any()
            or (y == -config.FIELD_SIZE[1]).any()):
        raise ValueError("xT gains require valid action starting coordinates and team perspective.")
    away = teams.eq("away").to_numpy()
    x = np.where(away, config.FIELD_SIZE[0] - x, x)
    y = np.where(away, config.FIELD_SIZE[1] - y, y)
    baseline = surface.predict_xy(x, y)
    updated = pairs.copy()
    gains = updated.iloc[:, 0].to_numpy(dtype=float) - baseline
    updated.iloc[:, 0] = np.maximum(gains, 0) if settings["clip_negative_gains"] else gains
    if conceding_baselines is not None and settings["version"] == 2:
        gains = updated.iloc[:, 1].to_numpy(dtype=float) - conceding_baselines
        updated.iloc[:, 1] = np.maximum(gains, 0) if settings["clip_negative_gains"] else gains
    if not np.isfinite(updated.to_numpy()).all():
        raise ValueError("Non-finite xT gain targets.")
    return updated


def training_target_definition(metadata):
    """Comparable definition for training records and model bundles."""
    return dict(xt_version=(metadata or {}).get("xt_version"),
                xt_fit_metadata_sha256=(metadata or {}).get("xt_fit_metadata_sha256"),
                xt_targets=target_settings(metadata))


def validate_training_targets(metadata, return_type=None):
    settings = target_settings(metadata)
    unavailable = (metadata or {}).get("xt_unavailable_return_types") or []
    if return_type is not None and return_type in unavailable:
        raise ValueError(f"xT targets for return_type={return_type!r} are unavailable in this feature run.")
    if settings["representation"] == "gain" and not settings["clip_negative_gains"]:
        raise ValueError("Existing xT outcome training requires --xt-clip-negative-gains; "
                         "refresh the feature targets with clipping before training.")
