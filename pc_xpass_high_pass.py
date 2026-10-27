"""Optional aerial grids, CLI configuration and reusable 25 Hz flight profiles."""
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import numpy as np

DEFAULTS = dict(high_pass="off", high_pass_angle=[15., 20., 25., 30.],
                high_pass_max_speed=30., high_pass_min_speed=20.,
                high_pass_speed_step=None, high_pass_cutoff=2.)
KEYS = set(DEFAULTS)
TOP_PASS_DEFINITION = "best_endpoint_per_speed_azimuth_elevation_v2"
DETAIL_SUFFIXES = ["__elevation", "__target_z", "__flight_time"]
LANE_DETAILS = ["elevation", "target_z", "flight_time"]
REVISION = "cc6af7b00769fc780d890eca49f530c2dce97e4b"


def add_arguments(parser):
    parser.add_argument("--high-pass", choices=["on", "off"], default="off")
    parser.add_argument("--high-pass-angle", nargs="+", type=float, default=DEFAULTS["high_pass_angle"].copy())
    parser.add_argument("--high-pass-max-speed", type=float, default=30.)
    parser.add_argument("--high-pass-min-speed", type=float, default=20.)
    parser.add_argument("--high-pass-speed-step", type=float, default=None,
                        help="Aerial total release-speed spacing; inherits --speed-step.")
    parser.add_argument("--high-pass-cutoff", type=float, default=2.,
                        help="Maximum contact height in metres, inclusive; default 2.")


def configuration(high_pass="off", high_pass_angle=None, high_pass_max_speed=30.,
                  high_pass_min_speed=20., high_pass_speed_step=None,
                  high_pass_cutoff=2., *, speed_step=2.):
    if high_pass not in ("on", "off"):
        raise ValueError("--high-pass must be on or off")
    if high_pass == "off":
        return None
    angles = sorted(set(float(x) for x in (DEFAULTS["high_pass_angle"] if high_pass_angle is None else high_pass_angle)))
    if not angles or any(not math.isfinite(a) or not 0 < a < 90 for a in angles):
        raise ValueError("--high-pass-angle requires finite angles strictly between 0 and 90 degrees")
    low, high = float(high_pass_min_speed), float(high_pass_max_speed)
    step = float(speed_step if high_pass_speed_step is None else high_pass_speed_step)
    cutoff = float(high_pass_cutoff)
    if any(not math.isfinite(v) or v <= 0 for v in (low, high, step, cutoff)) or low > high:
        raise ValueError("High-pass speeds, step and cutoff must be positive and finite; min speed must not exceed max")
    speeds = np.arange(low, high + step*.5, step)
    speeds = speeds[speeds <= high + 1e-9]
    return dict(angles=angles, min_speed=low, max_speed=high, speed_step=step,
                cutoff=cutoff, speeds=speeds.tolist(), physics=physics_metadata())


def physics_metadata():
    # Keep cache-only readers independent of the simulator import and profile cache.
    return dict(model="kang_zero_spin_v1", source_revision=REVISION, source_sha256=_source_digest(), timestep=.04,
                gravity=9.81, air_resistance=.00648, restitution=.109,
                ground_friction=0., rolling_friction=.02731, rolling_z_thresh=.054,
                rolling_vertical_speed_threshold=1., launch_height=0., spin=0.,
                interpolation="linear_horizontal_range_v1", max_range=150.,
                termination="air_only_remaining_range_bound_v1", remaining_range_tolerance=.001,
                max_steps=100000)


@lru_cache(maxsize=1)
def _source_digest():
    path = Path(__file__).parent / "_vendor/montecarlo_pass_search/ball_motion.py"
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def validate(config):
    if config is None:
        return None
    try:
        expected = configuration("on", config["angles"], config["max_speed"], config["min_speed"],
                                 config["speed_step"], config["cutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid high-pass configuration; create a new pc-xPass version") from exc
    if config != expected:
        raise ValueError("Incompatible high-pass configuration or physics; create a new pc-xPass version")
    return config


def kwargs(args):
    return {key: getattr(args, key, default) for key, default in DEFAULTS.items()}


def cli_settings(config):
    if not config:
        return dict(DEFAULTS)
    validate(config)
    return dict(high_pass="on", high_pass_angle=config["angles"],
                high_pass_max_speed=config["max_speed"], high_pass_min_speed=config["min_speed"],
                high_pass_speed_step=config["speed_step"], high_pass_cutoff=config["cutoff"])


def configure_args(parser, args):
    explicit = getattr(args, "_pc_explicit", set())
    selected = kwargs(args)
    try:
        if explicit & KEYS and not getattr(args, "pc_xpass", False):
            raise ValueError("High-pass options require --pc-xpass")
        if selected["high_pass"] == "off" and explicit & (KEYS - {"high_pass"}):
            raise ValueError("Explicit high-pass tuning requires --high-pass on")
        args.high_pass_config = configuration(**selected, speed_step=args.speed_step)
        if args.high_pass_config:
            args.high_pass_angle = args.high_pass_config["angles"]
            args.high_pass_speed_step = args.high_pass_config["speed_step"]
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


@lru_cache(maxsize=512)
def profile(speed, elevation):
    """Immutable range, time, height; remaining unrepresented travel <=1 mm."""
    from _vendor.montecarlo_pass_search import ball_motion as motion
    if not math.isfinite(speed) or speed <= 0 or not math.isfinite(elevation) or not 0 < elevation < 90:
        raise ValueError("Flight profiles require positive finite speed and elevation strictly between 0 and 90")
    beta = math.radians(elevation)
    initial = np.array([speed*math.cos(beta), 0., speed*math.sin(beta)])
    rows = []
    for index, (pos, vel) in enumerate(motion.states(np.zeros(3), initial)):
        rows.append((float(pos[0]), index/motion.FPS, float(pos[2])))
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
            path = profile(speed, elevation)
            valid = distances <= path[-1, 0]
            times.append(np.where(valid, np.interp(distances, path[:, 0], path[:, 1]), 0.)[None, None, :])
            heights.append(np.interp(distances, path[:, 0], path[:, 2])[None, None, :])
            reachable.append(valid[None, None, :])
            speeds.append(speed)
            elevations.append(elevation)
    return (np.asarray(speeds), np.asarray(elevations), np.concatenate(times),
            np.concatenate(heights), np.concatenate(reachable))


def top_pass_definition(metadata, legacy):
    return TOP_PASS_DEFINITION if metadata.get("high_pass_config") else legacy
