"""Distance sigmoid settings and prediction-cache contract (no movement lookup)."""
from __future__ import annotations

from collections.abc import Mapping
import math

VERSION = "split_lane_control_v1"
OVERRIDE_KEYS = (
    "distance_lane_power", "distance_control_power",
    "distance_lane_allowance", "distance_control_allowance",
)
PARAMETER_KEYS = ("distance_power", "distance_allowance", *OVERRIDE_KEYS)
SETTING_KEYS = (*PARAMETER_KEYS, "distance_control_version")


def configuration(
    margin: str = "distance", *,
    distance_power: float = 3.,
    distance_allowance: float = 1.,
    distance_lane_power: float | None = None,
    distance_control_power: float | None = None,
    distance_lane_allowance: float | None = None,
    distance_control_allowance: float | None = None,
) -> dict:
    """Resolve independent overrides, preserving explicit zero allowances."""
    overrides = dict(zip(OVERRIDE_KEYS, (
        distance_lane_power, distance_control_power,
        distance_lane_allowance, distance_control_allowance,
    )))
    if margin != "distance":
        if any(value is not None for value in overrides.values()):
            raise ValueError("Distance control overrides require --margin distance")
        return {}
    values = {"distance_power": distance_power, "distance_allowance": distance_allowance}
    values.update({key: (distance_power if key.endswith("power") else distance_allowance)
                   if value is None else value for key, value in overrides.items()})
    for key, value in values.items():
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"--{key.replace('_', '-')} must be a finite number") from exc
        positive = key.endswith("power")
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            requirement = "positive" if positive else "nonnegative"
            raise ValueError(f"--{key.replace('_', '-')} must be {requirement} and finite")
        values[key] = value
    return {**values, "distance_control_version": VERSION}


def validate_metadata(metadata: Mapping) -> None:
    """Reject old distance prediction caches, including version-root settings.

    Movement tables deliberately do not use this scoring contract. Validation
    only reads metadata, so cache consumers never initialize numerical solvers.
    """
    records = [metadata]
    saved = metadata.get("generation_settings")
    if isinstance(saved, Mapping):
        records.append(saved)
    for record in records:
        if record.get("margin", "tta") != "distance":
            continue
        message = ("Incompatible distance control cache; regenerate predictions under a new "
                   "--pc-xpass-id using the existing distance movement table")
        if record.get("distance_control_version") != VERSION:
            raise ValueError(message + f" (required distance_control_version={VERSION!r})")
        if any(record.get(key) is None for key in PARAMETER_KEYS):
            raise ValueError(message + " (missing resolved distance parameters)")
        try:
            configuration(**{key: record[key] for key in PARAMETER_KEYS})
        except ValueError as exc:
            raise ValueError(message + f" ({exc})") from exc
