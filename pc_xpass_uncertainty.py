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
TOP_PASS_DEFINITION = "expected_launch_then_best_elevation_per_speed_azimuth_v1"
DETAIL_SUFFIXES = ["__nominal_xpass", "__expected_xpass", "__expected_lane_survival",
                   "__expected_control_prob", "__expected_position_discount",
                   "__expected_ranking"]
TOP_DETAILS = ["nominal_xpass", "nominal_lane_survival", "nominal_control_prob"]


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    raise argparse.ArgumentTypeError("--uncertainty must be true or false")


def add_arguments(parser):
    parser.add_argument("--uncertainty", type=boolean, default=False,
                        help="Average existing launch outcomes before selection (PC xPass only).")
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
    return dict(schema_version=1, algorithm="existing_launch_gaussian_v1",
                speed_uncertainty=scales[0], angle_uncertainty=scales[1], elevation_uncertainty=scales[2],
                cutoff="inclusive_one_scale_per_dimension", weights="normalized_gaussian_point_weights",
                horizontal_difference="shortest_circular_degrees", distance="best_endpoint_per_executed_launch",
                missing_profiles="excluded", unsuccessful_outcomes="zero_with_weight_retained",
                selection="expectation_before_max_or_top_pass", xt="expected_xpass_times_outcome_endpoint_xt",
                endpoint_details="nominal_selected_intention", lane_sidecar="nominal_selected_intention")


def validate(config):
    if config is None:
        return None
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
        if not args.uncertainty_config:
            if explicit & (KEYS - {"uncertainty"}):
                raise ValueError("Explicit uncertainty scales require --uncertainty true")
            return
        if explicit & {"top_n", "top_n_values"}:
            raise ValueError("Uncertainty supports max and top-pass<N>; --top-n/--top-n-values are endpoint metrics")
        version = getattr(args, "x_pass_version", None)
        if version and version != "max" and not version.startswith("top-pass"):
            raise ValueError("Uncertainty supports --xpass-version max or top-pass<N>")
        # Ordinary endpoint metrics are not generated implicitly in this mode.
        args.top_n = None
        args.top_n_values = None
        args.export_topmean = False
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def metrics(config, values):
    validate(config)
    if config and any(v != "max_xpass" and not v.startswith("top_pass") for v in values):
        raise ValueError("Uncertainty supports only max and top-pass<N> metrics")
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
