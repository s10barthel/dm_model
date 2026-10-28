"""Optional aerial grids, CLI configuration and reusable 25 Hz flight profiles."""
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import warnings
import numpy as np

DEFAULTS = dict(high_pass="off", high_pass_angle=[20., 25., 30.],
                high_pass_speed=[21., 23., 25., 27., 29.], high_pass_cutoff=None,
                high_pass_lane_cutoff=2., high_pass_control_cutoff=.5, high_pass_bounce="on")
KEYS = set(DEFAULTS)
LEGACY_TOP_PASS_DEFINITION = "best_endpoint_per_speed_azimuth_elevation_v2"
TOP_PASS_DEFINITION = "best_endpoint_per_speed_azimuth_over_elevation_v3"
DETAIL_SUFFIXES = ["__elevation", "__target_z", "__flight_time"]
LANE_DETAILS = ["elevation", "target_z", "flight_time"]
REVISION = "cc6af7b00769fc780d890eca49f530c2dce97e4b"


def add_arguments(parser):
    parser.add_argument("--high-pass", choices=["on", "off"], default="off")
    parser.add_argument("--high-pass-bounce", choices=["on", "off"], default="on",
                        help="Continue aerial trajectories through bouncing and rolling (default on); "
                             "off stops at first ground contact, inclusive.")
    parser.add_argument("--high-pass-angle", nargs="+", type=float, default=DEFAULTS["high_pass_angle"].copy())
    parser.add_argument("--high-pass-speed", nargs="+", type=float, default=DEFAULTS["high_pass_speed"].copy(),
                        help="Total 3D release speeds in m/s; independent of --speed-step.")
    parser.add_argument("--high-pass-lane-cutoff", type=float, default=2.,
                        help="Maximum lane interception height in metres, inclusive; default 2.")
    parser.add_argument("--high-pass-control-cutoff", type=float, default=.5,
                        help="Maximum endpoint control height in metres, inclusive; default 0.5.")
    parser.add_argument("--high-pass-cutoff", type=float, default=None,
                        help="Deprecated: explicitly set both height cutoffs to this value.")


def configuration(high_pass="off", high_pass_angle=None, *, high_pass_speed=None, high_pass_cutoff=None,
                  high_pass_lane_cutoff=None, high_pass_control_cutoff=None, high_pass_bounce="on"):
    lane_cutoff, control_cutoff = _cutoffs(high_pass_cutoff, high_pass_lane_cutoff, high_pass_control_cutoff)
    if high_pass not in ("on", "off"):
        raise ValueError("--high-pass must be on or off")
    if high_pass_bounce not in ("on", "off"):
        raise ValueError("--high-pass-bounce must be on or off")
    if high_pass == "off":
        return None
    angles = sorted(set(float(x) for x in (DEFAULTS["high_pass_angle"] if high_pass_angle is None else high_pass_angle)))
    if not angles or any(not math.isfinite(a) or not 0 < a < 90 for a in angles):
        raise ValueError("--high-pass-angle requires finite angles strictly between 0 and 90 degrees")
    speeds = sorted(set(float(x) for x in (DEFAULTS["high_pass_speed"] if high_pass_speed is None else high_pass_speed)))
    if not speeds or any(not math.isfinite(v) or v <= 0 for v in (*speeds, lane_cutoff, control_cutoff)):
        raise ValueError("High-pass speeds and cutoffs must be positive and finite; speed list must not be empty")
    result = dict(schema_version=3, angles=angles,
                  lane_cutoff=lane_cutoff, control_cutoff=control_cutoff,
                  speeds=speeds, physics=physics_metadata(high_pass_bounce))
    # Omit the default to preserve existing bounce-on cache identities.
    if high_pass_bounce == "off":
        result["bounce"] = "off"
    return result


def _cutoffs(alias, lane, control):
    if alias is not None:
        if lane is not None or control is not None:
            raise ValueError("--high-pass-cutoff cannot be combined with --high-pass-lane-cutoff or --high-pass-control-cutoff")
        warnings.warn("--high-pass-cutoff is deprecated; use --high-pass-lane-cutoff and --high-pass-control-cutoff. "
                      "The alias sets both cutoffs.", FutureWarning, stacklevel=3)
        return float(alias), float(alias)
    return (float(DEFAULTS["high_pass_lane_cutoff"] if lane is None else lane),
            float(DEFAULTS["high_pass_control_cutoff"] if control is None else control))


def normalize_cutoff_args(parser, args, explicit):
    """Resolve the alias before cached settings are compared or inherited."""
    explicit = set(explicit)
    if "high_pass_cutoff" in explicit:
        try:
            lane, control = _cutoffs(args.high_pass_cutoff,
                args.high_pass_lane_cutoff if "high_pass_lane_cutoff" in explicit else None,
                args.high_pass_control_cutoff if "high_pass_control_cutoff" in explicit else None)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        args.high_pass_lane_cutoff, args.high_pass_control_cutoff = lane, control
        args.high_pass_cutoff = None
        explicit.remove("high_pass_cutoff")
        explicit.update({"high_pass_lane_cutoff", "high_pass_control_cutoff"})
    return explicit


def physics_metadata(bounce="on"):
    # Keep cache-only readers independent of the simulator import and profile cache.
    return dict(model="kang_zero_spin_v1", source_revision=REVISION, source_sha256=_source_digest(), timestep=.04,
                gravity=9.81, air_resistance=.00648, restitution=.109,
                ground_friction=0., rolling_friction=.02731, rolling_z_thresh=.054,
                rolling_vertical_speed_threshold=1., launch_height=0., spin=0.,
                interpolation="linear_horizontal_range_v1", max_range=150.,
                termination=("first_ground_contact_inclusive_v1" if bounce == "off"
                             else "air_only_remaining_range_bound_v1"), remaining_range_tolerance=.001,
                max_steps=100000)


@lru_cache(maxsize=1)
def _source_digest():
    path = Path(__file__).parent / "_vendor/montecarlo_pass_search/ball_motion.py"
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def is_legacy(config):
    """The original single-cutoff format also used a different ranking definition."""
    return config is not None and "schema_version" not in config and "cutoff" in config


def _range_configuration(angles, low, high, step, lane, control, *, legacy=False):
    """Reconstruct historical range formats for validation, never generation."""
    low, high, step = float(low), float(high), float(step)
    if any(not math.isfinite(v) or v <= 0 for v in (low, high, step)) or low > high:
        raise ValueError("Invalid historical high-pass speed range")
    speeds = np.arange(low, high + step*.5, step)
    speeds = speeds[speeds <= high + 1e-9]
    result = configuration("on", angles, high_pass_speed=speeds,
                           high_pass_lane_cutoff=lane, high_pass_control_cutoff=control)
    result.update(schema_version=2, min_speed=low, max_speed=high, speed_step=step)
    if legacy:
        result.pop("schema_version")
        result["cutoff"] = result.pop("lane_cutoff")
        result.pop("control_cutoff")
    return result


def configuration_from_settings(settings, *, schema_version=3):
    if schema_version in (None, 2):
        # Historical defaults must not follow changes to today's CLI defaults.
        if settings.get("high_pass", "off") == "off":
            return None
        if settings.get("high_pass") != "on":
            raise ValueError("Invalid historical high-pass mode")
        step = settings.get("high_pass_speed_step")
        if step is None:
            step = settings.get("speed_step", 2.)
        legacy = schema_version is None
        lane = settings.get("high_pass_cutoff", 2.) if legacy else settings.get("high_pass_lane_cutoff", 2.)
        control = lane if legacy else settings.get("high_pass_control_cutoff", .5)
        return _range_configuration(settings.get("high_pass_angle", [15., 20., 25., 30.]),
            settings.get("high_pass_min_speed", 20.), settings.get("high_pass_max_speed", 30.),
            step, lane, control, legacy=legacy)
    if schema_version != 3:
        raise ValueError("Unsupported high-pass configuration schema; create a new pc-xPass version")
    selected = {key: settings.get(key, default) for key, default in DEFAULTS.items()}
    return configuration(**selected)


def validate(config, *, allow_legacy=False):
    if config is None:
        return None
    try:
        legacy = is_legacy(config)
        version = config.get("schema_version")
        if legacy or version == 2:
            expected = _range_configuration(config["angles"], config["min_speed"], config["max_speed"],
                config["speed_step"], config["cutoff"] if legacy else config["lane_cutoff"],
                config["cutoff"] if legacy else config["control_cutoff"], legacy=legacy)
        elif version == 3:
            expected = configuration("on", config["angles"], high_pass_speed=config["speeds"],
                high_pass_lane_cutoff=config["lane_cutoff"], high_pass_control_cutoff=config["control_cutoff"],
                high_pass_bounce=config.get("bounce", "on"))
        else:
            raise ValueError("Unsupported high-pass configuration schema")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid high-pass configuration; create a new pc-xPass version") from exc
    if config != expected:
        raise ValueError("Incompatible high-pass configuration or physics; create a new pc-xPass version")
    if version != 3 and not allow_legacy:
        raise ValueError("Range-based high-pass caches are read-only; create a new pc-xPass version")
    return config


def kwargs(args):
    return {key: getattr(args, key, default) for key, default in DEFAULTS.items()}


def cli_settings(config):
    if not config:
        return dict(DEFAULTS)
    validate(config)
    return dict(high_pass="on", high_pass_angle=config["angles"],
                high_pass_speed=config["speeds"], high_pass_cutoff=None,
                high_pass_lane_cutoff=config["lane_cutoff"], high_pass_control_cutoff=config["control_cutoff"],
                high_pass_bounce=config.get("bounce", "on"))


def configure_args(parser, args):
    explicit = normalize_cutoff_args(parser, args, getattr(args, "_pc_explicit", set()))
    selected = kwargs(args)
    try:
        if explicit & KEYS and not getattr(args, "pc_xpass", False):
            raise ValueError("High-pass options require --pc-xpass")
        if selected["high_pass"] == "off" and explicit & (KEYS - {"high_pass"}):
            raise ValueError("Explicit high-pass tuning requires --high-pass on")
        args.high_pass_config = configuration(**selected)
        if args.high_pass_config:
            args.high_pass_angle = args.high_pass_config["angles"]
            args.high_pass_speed = args.high_pass_config["speeds"]
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


@lru_cache(maxsize=512)
def profile(speed, elevation, bounce="on"):
    """Immutable range, time, height; optionally end at first ground contact."""
    from _vendor.montecarlo_pass_search import ball_motion as motion
    if not math.isfinite(speed) or speed <= 0 or not math.isfinite(elevation) or not 0 < elevation < 90:
        raise ValueError("Flight profiles require positive finite speed and elevation strictly between 0 and 90")
    if bounce not in ("on", "off"):
        raise ValueError("--high-pass-bounce must be on or off")
    beta = math.radians(elevation)
    initial = np.array([speed*math.cos(beta), 0., speed*math.sin(beta)])
    rows = []
    for index, (pos, vel) in enumerate(motion.states(np.zeros(3), initial)):
        rows.append((float(pos[0]), index/motion.FPS, float(pos[2])))
        if bounce == "off" and index > 0 and pos[2] == 0.:
            break
        bound = math.hypot(*vel[:2])*(1-motion.AIR_RESISTANCE)/(motion.FPS*motion.AIR_RESISTANCE)
        if pos[0] >= 150. or bound <= .001:
            break
        if index >= 100000:
            raise RuntimeError("High-pass trajectory failed to reach its stopping tolerance")
    result = np.asarray(rows, dtype=float)
    if np.any(np.diff(result[:, 0]) <= 0):
        raise RuntimeError("High-pass horizontal trajectory is not strictly increasing")
    result.setflags(write=False)
    return result


def candidates(distances, ground_speeds, ground_times, ground_reachable, config):
    validate(config)
    times, heights, reachable = [ground_times], [np.zeros_like(ground_times)], [ground_reachable]
    speeds, elevations = list(ground_speeds), [0.]*len(ground_speeds)
    for elevation in config["angles"]:
        for speed in config["speeds"]:
            path = profile(speed, elevation, config.get("bounce", "on"))
            valid = distances <= path[-1, 0]
            times.append(np.where(valid, np.interp(distances, path[:, 0], path[:, 1]), 0.)[None, None, :])
            heights.append(np.interp(distances, path[:, 0], path[:, 2])[None, None, :])
            reachable.append(valid[None, None, :])
            speeds.append(speed)
            elevations.append(elevation)
    return (np.asarray(speeds), np.asarray(elevations), np.concatenate(times),
            np.concatenate(heights), np.concatenate(reachable))


def top_pass_definition(metadata, legacy):
    config = metadata.get("high_pass_config")
    if not config:
        return legacy
    return LEGACY_TOP_PASS_DEFINITION if is_legacy(config) else TOP_PASS_DEFINITION
