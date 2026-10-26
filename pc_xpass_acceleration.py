"""Bounded vector acceleration for pc-xPass; independent of ball physics.

The reference solver brackets the first arrival using continuous-time support
bounds. Mesh convergence alone is deliberately not a convergence certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from itertools import product

import numpy as np

VERSION = "vector_acceleration_v1"
TIME_TOL = 0.01
POSITION_TOL = 0.01
ROOT = Path(__file__).resolve().parent / "data" / "pc_xpass" / "acceleration"
COUNTERS = dict(queries=0, fallbacks=0, fallback_seconds=0.)


class UnresolvedArrival(RuntimeError):
    """A reference solve did not establish the requested numerical accuracy."""


@dataclass
class Arrival:
    lower: float
    upper: float
    times: np.ndarray
    velocities: np.ndarray
    position_error: float
    intervals: int
    evaluations: int = 0

    @property
    def time(self):
        return self.upper


def parse_accel(value):
    if value is None or str(value).lower() == "off":
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("--accel must be 'off' or a positive finite acceleration in m/s^2") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("--accel must be 'off' or a positive finite acceleration in m/s^2")
    return result


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def _straight(distance, initial, accel, cap):
    """Earliest 1-D arrival toward positive distance, including signed velocity."""
    if distance == 0:
        return 0.
    ramp = (cap - initial) / accel
    ramp_distance = (cap * cap - initial * initial) / (2 * accel)
    if distance <= ramp_distance:
        return (-initial + math.sqrt(initial * initial + 2 * accel * distance)) / accel
    return ramp + (distance - ramp_distance) / cap


def _stop_path(v0, target):
    speed = float(np.linalg.norm(v0))
    stop = .5 * speed * v0
    delta = target - stop
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance else np.zeros(2)
    run = _straight(distance, 0., 1., 1.)
    ramp = min(run, 1.)
    durations = [speed, ramp, max(0., run - ramp)]
    velocities = [v0, np.zeros(2), direction * ramp, direction * ramp]
    keep = [0] + [i + 1 for i, d in enumerate(durations) if d > 0]
    return np.r_[0., np.cumsum(durations)][keep], np.asarray(velocities)[keep]


def _fixed_time(v0, target, duration, intervals):
    """Feasible polygonal velocity path and continuous reachable-distance bound.

    For any p, |p|<=1 and piecewise linear costate l with l(T)=0,
      p.integral(v) <= l(0).v0 + integral(|p+l'|) + integral(|l|).
    Thus p.target minus this expression bounds distance to the *continuous*
    reachable set, regardless of discretization. Convex trapezoidal integration
    overestimates integral(|l|), preserving the lower bound.
    """
    try:
        import clarabel
    except ImportError as exc:
        raise RuntimeError("Acceleration reference solving requires clarabel; install project requirements") from exc
    from scipy import sparse

    n = intervals
    h = duration / n
    variables = 2 * (n + 1) + 1
    rows, cols, vals, rhs, cones = [], [], [], [], []

    def block(entries, b, cone):
        offset = len(rhs)
        for row, col, value in entries:
            rows.append(offset + row); cols.append(col); vals.append(value)
        rhs.extend(b); cones.append(cone)

    block([(0, 0, 1.), (1, 1, 1.)], v0, clarabel.ZeroConeT(2))
    for k in range(1, n + 1):
        block([(1, 2*k, -1.), (2, 2*k+1, -1.)], [1., 0., 0.], clarabel.SecondOrderConeT(3))
    accel_start = len(rhs)
    for k in range(n):
        block([(axis+1, 2*(k+1)+axis, -1.) for axis in range(2)] +
              [(axis+1, 2*k+axis, 1.) for axis in range(2)],
              [h, 0., 0.], clarabel.SecondOrderConeT(3))
    weights = np.full(n + 1, h); weights[[0, -1]] *= .5
    terminal_start = len(rhs)
    entries = [(0, variables-1, -1.)]
    entries.extend((axis+1, 2*k+axis, w) for k, w in enumerate(weights) for axis in range(2))
    block(entries, [0., *target], clarabel.SecondOrderConeT(3))
    matrix = sparse.csc_matrix((vals, (rows, cols)), shape=(len(rhs), variables))
    objective = np.zeros(variables); objective[-1] = 1.
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 150
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    solver = clarabel.DefaultSolver(sparse.csc_matrix((variables, variables)), objective,
                                   matrix, np.asarray(rhs), cones, settings)
    sol = solver.solve()
    if str(sol.status) not in {"Solved", "AlmostSolved"}:
        raise UnresolvedArrival("Fixed-time acceleration solve: " + str(sol.status))
    velocity = np.asarray(sol.x[:-1]).reshape(n+1, 2)
    position = weights @ velocity
    residual = float(np.linalg.norm(position - target))
    if (np.max(np.linalg.norm(velocity, axis=1)) > 1 + 1e-7 or
            np.max(np.linalg.norm(np.diff(velocity, axis=0), axis=1)) > h * (1 + 1e-7) or
            np.linalg.norm(velocity[0] - v0) > 1e-7):
        raise UnresolvedArrival("Trajectory violates speed, acceleration, or initial velocity")
    dual = np.asarray(sol.z)
    p = -dual[terminal_start+1:terminal_start+3]
    p /= max(1., float(np.linalg.norm(p)))
    costate = -dual[accel_start:terminal_start].reshape(n, 3)[:, 1:]
    costate = np.vstack([costate[0] + .5*h*p, costate, np.zeros(2)])
    knots = np.r_[0., (np.arange(n)+.5)*h, duration]
    widths = np.diff(knots)
    slopes = np.diff(costate, axis=0) / widths[:, None]
    # Refine the integral bound without enlarging the trajectory SOCP. Every
    # chord still overestimates the convex norm, so this remains conservative.
    fractions = np.linspace(0., 1., 17)
    samples = costate[:-1, None, :] + fractions[None, :, None] * np.diff(costate, axis=0)[:, None, :]
    norms = np.linalg.norm(samples, axis=2)
    norm_integral = widths @ ((.5*(norms[:, 0]+norms[:, -1])+norms[:, 1:-1].sum(axis=1))/16.)
    support = float(costate[0] @ v0 + widths @ np.linalg.norm(p + slopes, axis=1) + norm_integral)
    # Explicit floating-point guard; dual feasibility is not needed by this identity.
    guard = 1e-8 * (1 + abs(support) + float(np.linalg.norm(target)))
    distance_lower = max(0., float(p @ target) - support - guard)
    return residual, distance_lower, velocity


def solve_arrival(position, velocity, target, accel, max_speed, *,
                  time_tol=TIME_TOL, position_tol=POSITION_TOL,
                  max_intervals=4096, max_evaluations=4000):
    """Bracket the first arrival, not merely the first root found by bisection."""
    a = _positive(accel, "accel"); cap = _positive(max_speed, "max_speed")
    time_tol = _positive(time_tol, "time_tol"); position_tol = _positive(position_tol, "position_tol")
    origin, initial, destination = [np.asarray(x, dtype=float) for x in (position, velocity, target)]
    if any(x.shape != (2,) or not np.isfinite(x).all() for x in (origin, initial, destination)):
        raise ValueError("Movement position, velocity, and target must be finite 2-vectors")
    initial = initial * min(1., cap / max(float(np.linalg.norm(initial)), 1e-300))
    length_scale, time_scale = cap*cap/a, cap/a
    q, v0 = (destination-origin)/length_scale, initial/cap
    distance, speed = float(np.linalg.norm(q)), float(np.linalg.norm(v0))
    if distance == 0:
        return Arrival(0., 0., np.array([0.]), initial[None, :], 0., 0)
    direction = q/distance
    lateral = v0 - (v0 @ direction)*direction
    if np.linalg.norm(lateral) < 1e-12:
        t = _straight(distance, float(v0 @ direction), 1., 1.)
        ramp = min(t, 1.-float(v0 @ direction))
        knots = np.unique([0., ramp, t])
        velocities = np.minimum(float(v0 @ direction)+knots, 1.)[:, None]*direction
        return Arrival(t*time_scale, t*time_scale, knots*time_scale, velocities*cap, 0., len(knots)-1)
    # Without the speed cap, the reachable position set is exactly the disk
    # centred at v0*t with radius t^2/2. Its first boundary crossing is a global
    # lower bound and, if final speed respects the cap, an exact solution.
    roots = np.roots([.25, 0., -speed*speed, 2.*float(q @ v0), -distance*distance])
    positive = sorted(float(root.real) for root in roots if root.real > 0 and abs(root.imag) < 1e-7)
    unbounded = positive[0] if positive else 0.
    if unbounded:
        control = 2*(q-v0*unbounded)/(unbounded*unbounded)
        final = v0+control*unbounded
        if np.linalg.norm(final) <= 1.+1e-10 and np.linalg.norm(control) <= 1.+1e-8:
            t = unbounded*time_scale
            return Arrival(t, t, np.array([0., t]), np.array([v0, final])*cap, 0., 1)
    upper_times, upper_velocity = _stop_path(v0, q)
    upper = float(upper_times[-1])
    lower = max(distance, unbounded, _straight(distance, float(v0 @ direction), 1., 1.))
    # A constant-acceleration arc followed by cruising supplies a tighter *feasible
    # upper bound*. It does not restrict the optimizer's allowed trajectories.
    from scipy.optimize import brentq
    def arc(duration):
        offset = q-v0*duration
        norm = float(np.linalg.norm(offset))
        u = offset/max(norm, 1e-300)
        projection = float(v0 @ u)
        ramp = -projection+math.sqrt(max(0., projection**2+1-speed**2))
        return ramp*duration-.5*ramp*ramp-norm, ramp, u
    if arc(lower)[0] <= 0 and arc(upper)[0] >= 0:
        candidate = brentq(lambda t: arc(t)[0], lower, upper, xtol=1e-12)
        _, ramp, u = arc(candidate)
        if 0 < ramp <= candidate and np.linalg.norm(v0+u*ramp) <= 1+1e-8:
            upper = candidate
            upper_times = np.array([0., ramp, candidate])
            upper_velocity = np.array([v0, v0+u*ramp, v0+u*ramp])
    queue = [(-(upper-lower), lower, upper)]
    evaluations, largest_mesh = 0, 32
    residual_upper = 0.
    t_eps, r_eps = time_tol/time_scale, position_tol/length_scale
    # Terminal residual is a feasibility tolerance, not an interception radius.
    residual_goal = min(r_eps, 1e-8)
    while queue:
        earliest = min(upper, min(left for _, left, _ in queue))
        if upper-earliest <= t_eps:
            break
        _, left, right = heapq.heappop(queue)
        right = min(right, upper)
        if left >= upper:
            continue
        midpoint = .5*(left+right)
        n = 32
        while True:
            evaluations += 1
            if evaluations > max_evaluations:
                raise UnresolvedArrival("Arrival-time search exceeded evaluation budget")
            residual, bound, trial_velocity = _fixed_time(v0, q, midpoint, n)
            largest_mesh = max(largest_mesh, n)
            if residual <= residual_goal:
                upper = midpoint
                upper_times = np.linspace(0., midpoint, n+1)
                upper_velocity = trial_velocity
                residual_upper = residual
                break
            if bound > .5*(right-left):
                break
            if n >= max_intervals or right-left > 4*t_eps or residual-bound <= t_eps/16:
                break
            n *= 2
        if bound > .5*(right-left):
            continue
        if right-left <= np.finfo(float).eps*max(1., upper):
            raise UnresolvedArrival("Could not certify earliest arrival on refined mesh")
        if left < min(midpoint, upper):
            end = min(midpoint, upper)
            heapq.heappush(queue, (-(end-left), left, end))
        if midpoint < min(right, upper):
            end = min(right, upper)
            heapq.heappush(queue, (-(end-midpoint), midpoint, end))
    earliest = min(upper, min((left for _, left, _ in queue), default=upper))
    if upper-earliest > t_eps*(1+1e-8):
        raise UnresolvedArrival("Arrival bracket exceeds requested tolerance")
    return Arrival(earliest*time_scale, upper*time_scale, upper_times*time_scale,
                   upper_velocity*cap, residual_upper*length_scale, largest_mesh, evaluations)


def table_path(table_id):
    from pc_xpass_versions import validate_id
    return ROOT / validate_id(table_id)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=8)
def _load(path, meta_stamp, array_stamp):
    root = Path(path)
    meta = json.loads((root/"metadata.json").read_text())
    if meta.get("status") != "complete" or meta.get("movement_version") != VERSION:
        raise ValueError("Incomplete or incompatible acceleration table: " + path)
    if _digest(root/"table.npz") != meta.get("fingerprint"):
        raise ValueError("Corrupted acceleration table: " + path)
    with np.load(root/"table.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    n = len(arrays["low"])
    if (arrays["low"].shape != (n, 3) or arrays["high"].shape != (n, 3) or
            arrays["values"].shape != (n, 8) or arrays["children"].shape != (n, 2) or
            arrays["fallback"].shape != (n,) or not n or
            not np.isfinite(arrays["low"]).all() or not np.isfinite(arrays["high"]).all() or
            np.any(arrays["high"] <= arrays["low"]) or
            np.any(arrays["children"] >= n)):
        raise ValueError("Invalid acceleration table arrays")
    for i, child in enumerate(arrays["children"]):
        if child[0] >= 0 and (np.any(child <= i) or arrays["axis"][i] not in (0, 1, 2)):
            raise ValueError("Invalid acceleration table tree")
        if child[0] < 0 and not arrays["fallback"][i] and not np.isfinite(arrays["values"][i]).all():
            raise ValueError("Invalid acceleration interpolation leaf")
    return meta, arrays


def load_table(table_id):
    root = table_path(table_id)
    try:
        return _load(str(root), (root/"metadata.json").stat().st_mtime_ns,
                     (root/"table.npz").stat().st_mtime_ns)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing acceleration table {table_id!r}; run scripts/prepare_pc_xpass_acceleration.py "
                         f"--table-id {table_id} --accel <a> --max-player-speed <V> first") from exc


def configuration(accel=None, table_id=None, *, max_speeds=(5.,), fingerprint=None):
    a = parse_accel(accel)
    if a is None:
        if table_id is not None:
            raise ValueError("--accel-table-id requires acceleration to be enabled")
        return {}
    if not table_id:
        raise ValueError("--accel-table-id is required with --accel <number>")
    caps = [_positive(v, "max_player_speed") for v in max_speeds]
    meta, _ = load_table(table_id)
    if fingerprint is not None and fingerprint != meta["fingerprint"]:
        raise ValueError("Acceleration table fingerprint differs from recorded cache")
    if (meta["time_error_normalized"]*max(caps)/a > TIME_TOL*(1+1e-10) or
            meta["position_tolerance_normalized"]*max(caps)**2/a > POSITION_TOL*(1+1e-10)):
        raise ValueError("Acceleration table is too coarse for requested accel/speed; prepare a compatible table")
    return dict(accel=a, table_id=table_id, fingerprint=meta["fingerprint"], movement_version=VERSION,
                initial_speed="clip_preserve_direction", reaction_time=0., time_tolerance=TIME_TOL,
                position_tolerance=POSITION_TOL)


def add_arguments(parser):
    parser.add_argument("--accel", default="off", help="pc-xPass only: 'off' or maximum vector acceleration in m/s^2")
    parser.add_argument("--accel-table-id", help="Prepared acceleration lookup table; required when --accel is enabled")


def configure_args(parser, args):
    explicit = getattr(args, "_pc_explicit", set())
    # The ordinary physical path does not run pc_xpass_versions preparation.
    if not getattr(args, "pc_xpass", False):
        if getattr(args, "accel", "off") != "off" or getattr(args, "accel_table_id", None):
            parser.error("--accel and --accel-table-id require --pc-xpass")
        args.accel, args.acceleration_config = None, {}
        return
    try:
        a = parse_accel(getattr(args, "accel", None))
        if a is not None:
            if getattr(args, "margin", "tta") != "tta":
                raise ValueError("--accel requires --margin tta; cannot combine with empirical reachability")
            if any(k in explicit for k in ("dist_pass_div", "dist_pass_min", "dist_pass_max")):
                raise ValueError("--dist-pass-* cannot be combined with --accel")
            reaction = getattr(args, "reaction_time", 0.)
            recorded = getattr(args, "_pc_existing_metadata", {}).get("generation_settings", {})
            if "reaction_time" in explicit or recorded:
                if getattr(args, "reaction_time_mode", "fixed") != "fixed" or float(reaction) != 0.:
                    raise ValueError("Acceleration requires --reaction-time 0; remove distance/nonzero reaction settings")
            args.reaction_time, args.reaction_time_mode = 0., "fixed"
        common = getattr(args, "max_player_speed", 5.)
        caps = [getattr(args, "max_player_speed_off", None) or common,
                getattr(args, "max_player_speed_def", None) or common]
        args.acceleration_config = configuration(a, getattr(args, "accel_table_id", None), max_speeds=caps,
                                                fingerprint=getattr(args, "accel_table_fingerprint", None))
        args.accel = a
        if a is not None:
            args.accel_table_fingerprint = args.acceleration_config["fingerprint"]
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


_CORNERS = np.asarray(list(product((0., 1.), repeat=3)))


def _interpolate(values, weights):
    factors = np.where(_CORNERS[None, :, :] > 0, weights[:, None, :], 1.-weights[:, None, :])
    return np.sum(np.prod(factors, axis=2)*values, axis=1)


def arrival_times(players, target_x, target_y, max_speeds, config, valid=None):
    """Player x target TTA, evaluated once independently of ball launch speed."""
    meta, arrays = load_table(config["table_id"])
    if meta["fingerprint"] != config["fingerprint"] or config["movement_version"] != VERSION:
        raise ValueError("Acceleration movement/table definition changed")
    caps = np.asarray(max_speeds, float)
    a = _positive(config["accel"], "accel")
    if caps.shape != (len(players),) or not np.isfinite(caps).all() or np.any(caps <= 0):
        raise ValueError("Acceleration needs one positive finite speed cap per player")
    if (meta["time_error_normalized"]*caps.max()/a > TIME_TOL*(1+1e-10) or
            meta["position_tolerance_normalized"]*caps.max()**2/a > POSITION_TOL*(1+1e-10)):
        raise ValueError("Acceleration table is too coarse for requested player speeds")
    targets = np.column_stack([np.asarray(target_x).ravel(), np.asarray(target_y).ravel()])
    selected = np.ones(len(targets), bool) if valid is None else np.asarray(valid, bool).ravel()
    output = np.zeros((len(players), len(targets)))
    for i, (state, cap) in enumerate(zip(players, caps)):
        position, velocity = np.asarray(state[:2], float), np.asarray(state[2:4], float)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite player input for acceleration")
        a = config["accel"]
        delta = targets[selected]-position
        distance = np.linalg.norm(delta, axis=1)
        speed = min(float(np.linalg.norm(velocity)), cap)
        indices = np.flatnonzero(selected)
        COUNTERS["queries"] += len(indices)
        if speed == 0:
            ramp_distance = cap*cap/(2*a)
            output[i, indices] = np.where(distance <= ramp_distance, np.sqrt(2*distance/a),
                                           cap/a+(distance-ramp_distance)/cap)
            continue
        direction = velocity/np.linalg.norm(velocity)
        theta = np.arccos(np.clip((delta @ direction)/np.maximum(distance, 1e-300), -1., 1.))
        radius = a*distance/cap**2
        queries = np.column_stack([np.full(len(indices), speed/cap), theta, np.log1p(radius)])
        node = np.zeros(len(indices), dtype=int)
        inside = np.all((queries >= arrays["low"][0]) & (queries <= arrays["high"][0]), axis=1)
        while True:
            active = inside & (arrays["children"][node, 0] >= 0)
            if not active.any():
                break
            idx = np.flatnonzero(active)
            axis = arrays["axis"][node[idx]]
            midpoint = .5*(arrays["low"][node[idx], axis]+arrays["high"][node[idx], axis])
            branch = (queries[idx, axis] >= midpoint).astype(int)
            node[idx] = arrays["children"][node[idx], branch]
        interpolated = inside & ~arrays["fallback"][node]
        if interpolated.any():
            ids = np.flatnonzero(interpolated); leaves = node[ids]
            weights = (queries[ids]-arrays["low"][leaves])/(arrays["high"][leaves]-arrays["low"][leaves])
            delay = _interpolate(arrays["values"][leaves], weights)
            output[i, indices[ids]] = (radius[ids]+delay)*cap/a
        for j in np.flatnonzero(~interpolated):
            start = time.perf_counter()
            result = solve_arrival(position, velocity, targets[indices[j]], a, cap)
            output[i, indices[j]] = result.time
            COUNTERS["fallbacks"] += 1
            COUNTERS["fallback_seconds"] += time.perf_counter()-start
        output[i, indices[distance == 0]] = 0.
    return output.reshape((len(players), *np.asarray(target_x).shape))


def reference_point(task):
    """Picklable preparation worker, solving in normalized units."""
    point, time_tol, position_tol = task
    speed, theta, log_radius = point
    radius = math.expm1(log_radius)
    target = radius*np.array([math.cos(theta), math.sin(theta)])
    try:
        result = solve_arrival([0., 0.], [speed, 0.], target, 1., 1.,
                               time_tol=time_tol, position_tol=position_tol)
        return point, dict(lower=result.lower, upper=result.upper, residual=result.position_error,
                           intervals=result.intervals, evaluations=result.evaluations)
    except UnresolvedArrival as exc:
        return point, dict(unresolved=str(exc))
