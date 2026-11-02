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
    return dict(version=1, representation=representation, clip_negative_gains=clipping,
                scoring_baseline="action_start_surface" if representation == "gain" else "zero",
                conceding_baseline="zero")


def add_target_arguments(parser):
    parser.add_argument("--xt-target-representation", choices=("absolute", "gain"), default=None,
                        help="Saved xT target representation; extensions inherit the source setting.")
    parser.add_argument("--xt-clip-negative-gains", action=argparse.BooleanOptionalAction, default=None,
                        help="Clip scoring gains below zero; requires gain representation.")


def configure_targets(args, metadata=None, *, allow_change=False):
    """Resolve unspecified extension options from the source run, then validate."""
    previous = target_settings(metadata)
    representation = getattr(args, "xt_target_representation", None)
    clipping = getattr(args, "xt_clip_negative_gains", None)
    settings = target_settings({"xt_targets": dict(
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


def validate_surface_returns(surface_metadata, return_types):
    if (surface_metadata or {}).get("surface_type", (surface_metadata or {}).get("xt_model_type")) == "log_progression":
        unsupported = [r for r in return_types if parse_return_type(r)[0] not in {"next", "in"}]
        if unsupported:
            raise ValueError(f"Log progression supports next_N[_skip1] and in_N returns only: {unsupported}.")


def transform_action_targets(actions, pairs, settings=None, surface=None):
    settings = target_settings({"xt_targets": settings or {}})
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
    if not np.isfinite(updated.to_numpy()).all():
        raise ValueError("Non-finite xT gain targets.")
    return updated


def training_target_definition(metadata):
    """Comparable definition for training records and model bundles."""
    return dict(xt_version=(metadata or {}).get("xt_version"),
                xt_fit_metadata_sha256=(metadata or {}).get("xt_fit_metadata_sha256"),
                xt_targets=target_settings(metadata))


def validate_training_targets(metadata):
    settings = target_settings(metadata)
    if settings["representation"] == "gain" and not settings["clip_negative_gains"]:
        raise ValueError("Existing xT outcome training requires --xt-clip-negative-gains; "
                         "refresh the feature targets with clipping before training.")
