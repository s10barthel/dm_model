"""Discrete launch uncertainty using only already computed PC xPass outcomes.

The Gaussian shape and default scales follow Dick, Link and Brefeld (2022),
section 3.4/5, doi:10.1007/s10618-022-00827-2. The one-scale rectangular
cutoff and normalization over existing grid points are our local approximation.
"""
from functools import lru_cache
import argparse
import math

import numpy as np
from scipy.sparse import csr_matrix

DEFAULTS = dict(uncertainty=False, speed_uncertainty=2., angle_uncertainty=3.,
                elevation_uncertainty=6.)
KEYS = set(DEFAULTS)
METRIC = "uncertainty_xpass"
VERSION = "uncertainty"
DEFINITION = "expected_best_existing_launch_independent_v1"
GEOMETRY = ("speed", "angle", "elevation", "distance", "target_x", "target_y", "target_z", "flight_time")
COMPONENTS = ("xpass", "lane_survival", "control_prob", "position_discount", "ranking")
DETAIL_FIELDS = [*GEOMETRY, *(f"{kind}_{field}" for kind in ("nominal", "expected") for field in COMPONENTS)]
DETAIL_SUFFIXES = [f"__uncertainty_{field}" for field in DETAIL_FIELDS]


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    raise argparse.ArgumentTypeError("--uncertainty must be true or false")


def add_arguments(parser):
    parser.add_argument("--uncertainty", type=boolean, default=False,
                        help="Add a separate uncertainty_xpass export using existing launches (PC xPass only).")
    for name in ("speed_uncertainty", "angle_uncertainty", "elevation_uncertainty"):
        units = "m/s" if name == "speed_uncertainty" else "degrees"
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name],
                            help=f"Fixed Gaussian scale and inclusive neighbourhood radius in {units}; zero freezes this dimension.")


def configuration(uncertainty=False, speed_uncertainty=2., angle_uncertainty=3.,
                  elevation_uncertainty=6.):
    enabled = boolean(uncertainty)
    scales = tuple(float(v) for v in (speed_uncertainty, angle_uncertainty, elevation_uncertainty))
    if any(not math.isfinite(v) or v < 0 for v in scales):
        raise ValueError("Uncertainty scales must be finite and non-negative")
    if not enabled:
        return None
    return dict(schema_version=2, algorithm="existing_launch_gaussian_v1", metric=METRIC, definition=DEFINITION,
                speed_uncertainty=scales[0], angle_uncertainty=scales[1], elevation_uncertainty=scales[2],
                cutoff="inclusive_one_scale_per_dimension", weights="normalized_gaussian_point_weights",
                horizontal_difference="shortest_circular_degrees", distance="best_endpoint_per_executed_launch",
                missing_profiles="excluded", unsuccessful_outcomes="zero_with_weight_retained",
                selection="expectation_before_independent_max", xt="expected_xpass_times_outcome_endpoint_xt",
                endpoint_details="uncertainty_prefixed_nominal_selected_intention", lane_sidecar="raw_max")


def validate(config):
    if config is None:
        return None
    if isinstance(config, dict) and config.get("schema_version") == 1:
        raise ValueError("Legacy schema-1 uncertainty caches store adjusted scores as max_xpass; "
                         "regenerate under a new cache ID")
    try:
        expected = configuration(True, *(config[k] for k in
            ("speed_uncertainty", "angle_uncertainty", "elevation_uncertainty")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid uncertainty configuration; create a new pc-xPass version") from exc
    if config != expected:
        raise ValueError("Incompatible uncertainty configuration; create a new pc-xPass version")
    return config


def cli_settings(config):
    if not config:
        return dict(DEFAULTS)
    validate(config)
    return {"uncertainty": True, **{key: config[key] for key in KEYS - {"uncertainty"}}}


def configure_args(parser, args):
    explicit = getattr(args, "_pc_explicit", set())
    try:
        selected = {key: getattr(args, key, default) for key, default in DEFAULTS.items()}
        if explicit & KEYS and not getattr(args, "pc_xpass", False):
            raise ValueError("Uncertainty options require --pc-xpass")
        args.uncertainty_config = configuration(**selected)
        if getattr(args, "x_pass_version", None) == VERSION and not args.uncertainty_config:
            raise ValueError("Generating --xpass-version uncertainty requires --uncertainty true")
        if not args.uncertainty_config:
            if explicit & (KEYS - {"uncertainty"}):
                raise ValueError("Explicit uncertainty scales require --uncertainty true")
            return
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def metrics(config, values):
    validate(config)
    values = list(values)
    if not config and METRIC in values:
        raise ValueError("The uncertainty metric requires --uncertainty true during generation")
    if config and METRIC not in values:
        values.append(METRIC)
    return values


def _weights(difference, scale):
    tolerance = 1e-10
    if scale == 0:
        return (np.abs(difference) <= tolerance).astype(float)
    return np.where(np.abs(difference) <= scale + tolerance,
                    np.exp(-.5 * np.square(difference / scale)), 0.)


@lru_cache(maxsize=32)
def _neighbourhood(speeds, elevations, angles, speed_scale, angle_scale, elevation_scale):
    speeds, elevations, angles = map(np.asarray, (speeds, elevations, angles))
    profile = (_weights(speeds[:, None] - speeds[None, :], speed_scale)
               * _weights(elevations[:, None] - elevations[None, :], elevation_scale))
    delta = (angles[:, None] - angles[None, :] + 180.) % 360. - 180.
    horizontal = _weights(delta, angle_scale)
    # The intended launch always supplies positive self weight.
    profile /= profile.sum(axis=1, keepdims=True)
    horizontal /= horizontal.sum(axis=1, keepdims=True)
    return csr_matrix(profile), csr_matrix(horizontal)


def neighbourhood(speeds, elevations, angles_radians, config):
    validate(config)
    speeds, elevations, angles = map(lambda x: np.asarray(x, dtype=float),
                                     (speeds, elevations, angles_radians))
    if speeds.ndim != 1 or elevations.shape != speeds.shape or angles.ndim != 1 or not len(speeds) or not len(angles):
        raise ValueError("One speed/elevation per profile and a nonempty horizontal-angle grid are required")
    if not all(np.isfinite(x).all() for x in (speeds, elevations, angles)):
        raise ValueError("Uncertainty grid must be finite")
    return _neighbourhood(tuple(speeds), tuple(elevations), tuple(np.degrees(angles)),
                          config["speed_uncertainty"], config["angle_uncertainty"], config["elevation_uncertainty"])


def average(values, weights):
    profile, horizontal = weights
    # Failed computed outcomes keep their weight; only absent profiles are excluded.
    values = np.where(np.isfinite(values), values, 0.)
    return horizontal.dot(profile.dot(values).T).T


def outcomes(score, ranking, lane, control, discount, weights):
    """Reduce each executed trajectory to its best endpoint, then average launches."""
    finite = np.isfinite(score) & np.isfinite(ranking)
    valid = finite.any(axis=-1)
    endpoint = np.argmax(np.where(finite, ranking, -np.inf), axis=-1)
    index = np.indices(endpoint.shape)
    selected = (index[0], index[1], endpoint)
    data = dict(endpoint=endpoint, valid=valid)
    for name, values in (("xpass", score), ("ranking", ranking), ("lane_survival", lane),
                         ("control_prob", control), ("position_discount", np.broadcast_to(discount, score.shape))):
        nominal = np.where(valid, values[selected], 0.)
        data["nominal_" + name] = nominal
        data["expected_" + name] = average(nominal, weights)
    return data
