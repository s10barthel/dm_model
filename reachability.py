"""Versioned empirical circles and vectorized spatial margins (metres)."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / 'data' / 'pc_xpass' / 'reachability'
DEFAULTS = dict(model_id=None, envelope='0.999', lane_power=3.0,
                lane_inflection_point=1.0, control_power=3.0,
                control_inflection_point=1.0, extrapolation_start=5.0,
                extrapolation_speed=8.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_path(model_id):
    from pc_xpass_versions import validate_id
    return ROOT / validate_id(model_id)


@lru_cache(maxsize=8)
def _load(path, metadata_stamp, array_stamp):
    root = Path(path)
    meta = json.loads((root / 'metadata.json').read_text())
    if meta.get('status') != 'complete' or digest(root / 'circles.npz') != meta['fingerprint']:
        raise ValueError('Incomplete or corrupted reachability artifact: ' + path)
    with np.load(root / 'circles.npz', allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    speeds, times = arrays['speeds'], arrays['times']
    if (speeds.ndim != 1 or times.ndim != 1 or not len(speeds) or len(times) < 2
            or not np.isfinite(speeds).all() or not np.isfinite(times).all()
            or np.any(np.diff(speeds) <= 0) or np.any(np.diff(times) <= 0) or times[0] != 0):
        raise ValueError('Invalid reachability grids')
    for envelope in meta['envelopes']:
        c, r = arrays['c_' + envelope], arrays['r_' + envelope]
        if c.shape != (len(speeds), len(times)) or r.shape != c.shape or not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(r < 0):
            raise ValueError('Invalid reachability circles')
        if np.any(c[:, 0] != 0) or np.any(r[:, 0] != 0):
            raise ValueError('Reachability time-zero anchor must be zero')
    return meta, arrays


def load_model(model_id):
    path = model_path(model_id)
    return _load(str(path), (path / 'metadata.json').stat().st_mtime_ns,
                 (path / 'circles.npz').stat().st_mtime_ns)


def configuration(margin='tta', config=None):
    if margin == 'tta':
        return {}
    if margin != 'reachability':
        raise ValueError('margin must be tta or reachability')
    values = {**DEFAULTS, **(config or {})}
    if not values['model_id']:
        raise ValueError('--reachability-model-id is required')
    for key in DEFAULTS:
        if key in ('model_id', 'envelope'):
            continue
        value = float(values[key])
        if not math.isfinite(value) or (key.endswith('power') or key == 'extrapolation_start') and value <= 0 or key == 'extrapolation_speed' and value < 0:
            raise ValueError('Invalid reachability ' + key)
        values[key] = value
    meta, arrays = load_model(values['model_id'])
    if values['envelope'] not in meta['envelopes']:
        raise ValueError('Envelope not fitted: ' + str(values['envelope']))
    if values['extrapolation_start'] > arrays['times'][-1]:
        raise ValueError('Extrapolation start exceeds fitted horizon')
    if values.get('fingerprint', meta['fingerprint']) != meta['fingerprint']:
        raise ValueError('Reachability artifact fingerprint differs from cached configuration')
    values['fingerprint'] = meta['fingerprint']
    return values


def circle(arrays, speed, times, envelope, transition=5., expansion=8.):
    grid = arrays['speeds']
    speed = float(np.clip(speed, grid[0], grid[-1]))
    hi = min(int(np.searchsorted(grid, speed, side='right')), len(grid) - 1)
    lo = max(0, hi - 1)
    w = 0. if hi == lo else (speed - grid[lo]) / (grid[hi] - grid[lo])
    t = np.asarray(times)
    query = np.clip(t, 0, transition)
    result = []
    for prefix in ('c_', 'r_'):
        weight = w
        if prefix == 'c_' and lo == 0 and grid[0] < .5 / 3.6 and hi != lo:
            # Keep the entire standing bin symmetric and join continuously to bin 1.
            weight = np.clip((speed - .5 / 3.6) / (grid[hi] - .5 / 3.6), 0., 1.)
        row = (1 - weight) * arrays[prefix + envelope][lo] + weight * arrays[prefix + envelope][hi]
        result.append(np.interp(query, arrays['times'], row))
    result[1] += expansion * np.maximum(t - transition, 0)
    if speed < .5 / 3.6:
        result[0] = np.zeros_like(result[0])
    return result


def spatial_margins(players, target_x, target_y, t_ball, config):
    _, arrays = load_model(config['model_id'])
    # Stopped-ball cells may carry inf; they are masked by the caller.
    times = np.where(np.isfinite(t_ball), t_ball, 0.)
    result = []
    for x, y, vx, vy in players:
        speed = float(np.hypot(vx, vy))
        c, r = circle(arrays, speed, times, config['envelope'],
                      config['extrapolation_start'], config['extrapolation_speed'])
        ux, uy = (vx / speed, vy / speed) if speed > 0 else (0., 0.)
        result.append(r - np.hypot(target_x - x - c * ux, target_y - y - c * uy))
    return np.stack(result)


def add_arguments(parser):
    parser.add_argument('--margin', choices=['tta', 'reachability'], default='tta')
    for key, default in DEFAULTS.items():
        parser.add_argument('--reachability-' + key.replace('_', '-'),
                            default=default, type=str if key in ('model_id', 'envelope') else float)


def configure_args(parser, args):
    explicit = getattr(args, '_pc_explicit', set())
    if not getattr(args, 'pc_xpass', False) and ('margin' in explicit or any(k.startswith('reachability_') for k in explicit)):
        parser.error('Reachability options require --pc-xpass')
    config = {key: getattr(args, 'reachability_' + key, default) for key, default in DEFAULTS.items()}
    if hasattr(args, 'reachability_fingerprint'):
        config['fingerprint'] = args.reachability_fingerprint
    try:
        args.reachability_config = configuration(args.margin, config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.margin == 'reachability':
        args.reachability_fingerprint = args.reachability_config['fingerprint']
        inactive = explicit & {'reaction_time', 'dist_pass_div', 'dist_pass_min', 'dist_pass_max',
                               'max_player_speed', 'max_player_speed_off', 'max_player_speed_def',
                               'lane_power', 'lane_inflection_point', 'control_power', 'control_inflection_point'}
        if inactive:
            import warnings
            warnings.warn('Inactive in reachability mode: ' + ', '.join(sorted(inactive)), stacklevel=2)
