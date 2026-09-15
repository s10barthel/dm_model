"""Prepare reusable tracking shards, fit extreme circles, and diagnose holdout coverage."""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

import reachability as reach
from pc_xpass_versions import atomic_json
from project_config import resolve_split_manifest

ALGORITHM = 'aligned_directional_circle_v1'
HORIZONS = np.r_[np.arange(1, 26), np.arange(30, 151, 5)]
ENVELOPES = ['0.99', '0.995', '0.999', 'max']


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def split_pool(ids, seed=42):
    if len(ids) < 2:
        raise ValueError('At least two training-pool matches required')
    shuffled = np.random.default_rng(seed).permutation(sorted(ids)).tolist()
    n = max(1, math.ceil(.1 * len(ids)))
    return sorted(shuffled[n:]), sorted(shuffled[:n])


def indexed(frame):
    if 'frame_id' in frame.columns:
        frame = frame.set_index('frame_id')
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Tracking frame IDs must be unique and increasing')
    return frame


def eligible_players(lineup, columns):
    players = sorted(c[:-2] for c in columns if c.startswith(('home_', 'away_')) and c.endswith('_x'))
    roles = lineup.groupby('object_id')['advanced_position'].agg(lambda s: set(s.dropna().astype(str)))
    accepted, excluded = [], {'goalkeeper': 0, 'unresolved_role': 0}
    for player in players:
        role = roles.get(player, set())
        if 'goal_keeper' in role:
            excluded['goalkeeper'] += 1
        elif not role or role & {'', 'unknown', 'Unknown'}:
            excluded['unresolved_role'] += 1
        else:
            accepted.append(player)
    return accepted, excluded


def extract_player(raw, frame, player, args, player_index=0):
    n = len(frame)
    xy = frame[[player + '_x', player + '_y']].to_numpy(float)
    v = frame[[player + '_vx', player + '_vy']].to_numpy(float)
    rawxy = raw.reindex(frame.index)[[player + '_x', player + '_y']].to_numpy(float)
    accel = frame[player + '_accel'].to_numpy(float)
    speed = np.linalg.norm(v, axis=1)
    missing = ~(np.isfinite(xy).all(1) & np.isfinite(v).all(1) & np.isfinite(rawxy).all(1) & np.isfinite(accel))
    fast = speed > args.max_speed
    acceleration = np.abs(accel) > args.max_acceleration
    jumps = np.r_[False, np.linalg.norm(np.diff(rawxy, axis=0), axis=1) > args.max_speed / 25 + args.jump_tolerance]
    state = frame['ball_state'].astype(str).str.lower()
    in_play = state.isin(['alive', 'in_play', 'in play', '1', 'true']).to_numpy()
    episode = frame['episode_id'].to_numpy()
    period = frame['period_id'].to_numpy()
    boundary = np.r_[True, (np.diff(frame.index.to_numpy()) != 1) | (episode[1:] != episode[:-1]) | (period[1:] != period[:-1])]
    bad = missing | fast | acceleration | jumps | ~in_play | pd.isna(episode) | pd.isna(period)
    # Match the +/-7 samples used by the existing 15-sample velocity filter.
    guard = np.convolve((bad | boundary).astype(int), np.ones(15, int), mode='full')[7:7+n] > 0
    prefix = np.r_[0, np.cumsum(bad)]
    breaks = np.r_[0, np.cumsum(boundary)]
    records = []
    for horizon in HORIZONS:
        starts = np.arange(0, max(0, n - horizon), args.frame_stride)
        ends = starts + horizon
        valid = (~guard[starts]) & (prefix[ends + 1] == prefix[starts]) & (breaks[ends + 1] == breaks[starts + 1])
        starts, ends = starts[valid], ends[valid]
        if not len(starts):
            continue
        angle = np.arctan2(v[starts, 1], v[starts, 0])
        delta = xy[ends] - xy[starts]
        x = delta[:, 0] * np.cos(angle) + delta[:, 1] * np.sin(angle)
        y = -delta[:, 0] * np.sin(angle) + delta[:, 1] * np.cos(angle)
        records.append(pd.DataFrame(dict(bin=np.floor(speed[starts] * 3.6 / .5).astype('int16'),
            horizon=np.full(len(starts), horizon, dtype='int16'), x=x.astype('float32'), y=y.astype('float32'),
            episode=episode[starts], player=np.full(len(starts), player_index, dtype='int16'))))
    counts = dict(missing=int(missing.sum()), speed=int(fast.sum()), acceleration=int(acceleration.sum()),
                  jump=int(jumps.sum()), not_in_play=int((~in_play).sum()), smoothing_guard=int(guard.sum()))
    return records, counts


def prepare(args):
    path = reach.model_path(args.model_id)
    if (path / 'circles.npz').exists():
        raise ValueError('Completed model is immutable; use a new model ID')
    manifest = resolve_split_manifest(args.train_split, train_count=args.train_count)
    fit_ids, holdout = split_pool(manifest['train'])
    lineup_path = ROOT / 'data/lineup/line_up.parquet'
    sources = [lineup_path] + [ROOT / 'data' / folder / (m + '.parquet')
        for m in manifest['train'] for folder in ('tracking', 'tracking_processed')]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing))
    fingerprints = {str(p.relative_to(ROOT)): reach.digest(p) for p in sources}
    meta = dict(algorithm=ALGORITHM, status='preparing', source='Bundesliga', manifest=manifest,
        fit_matches=fit_ids, holdout_matches=holdout, seed=42, fps=25,
        preprocessing='existing processed diff(position)*25; Savitzky-Golay velocity window=15 order=2; acceleration window=9 order=2',
        horizons=HORIZONS.tolist(), speed_bin_kmh=.5, frame_stride=args.frame_stride,
        max_speed=args.max_speed, max_acceleration=args.max_acceleration, jump_tolerance=args.jump_tolerance,
        sources=fingerprints)
    prep_path = path / 'preparation.json'
    if prep_path.exists():
        previous = json.loads(prep_path.read_text())
        if previous != meta:
            raise ValueError('Preparation sources/configuration changed; use a new model ID')
    else:
        atomic_json(prep_path, meta)
    lineup = pd.read_parquet(lineup_path)
    for mid in fit_ids + holdout:
        shard = path / 'shards' / (mid + '.parquet')
        report_path = shard.with_suffix('.json')
        if shard.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get('sha256') == reach.digest(shard):
                print('Resume: ' + mid, flush=True)
                continue
            raise ValueError('Corrupt preparation shard: ' + mid)
        raw = indexed(pd.read_parquet(ROOT / 'data/tracking' / (mid + '.parquet')))
        frame = indexed(pd.read_parquet(ROOT / 'data/tracking_processed' / (mid + '.parquet')))
        players, exclusions = eligible_players(lineup[lineup.stats_perform_match_id.astype(str) == mid], frame.columns)
        if not players:
            raise ValueError('No resolved outfield players: ' + mid)
        import pyarrow as pa
        import pyarrow.parquet as pq
        reports = {}
        shard.parent.mkdir(parents=True, exist_ok=True)
        temp = shard.with_suffix('.tmp.parquet')
        writer = None
        try:
            for i, player in enumerate(players):
                rows, reports[player] = extract_player(raw, frame, player, args, i)
                for rows_at_horizon in rows:
                    table = pa.Table.from_pandas(rows_at_horizon, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
                    writer.write_table(table)
            if writer is None:
                raise ValueError('No valid trajectories: ' + mid)
        finally:
            if writer is not None:
                writer.close()
        temp.replace(shard)
        atomic_json(report_path, dict(players=players, exclusions=exclusions, quality=reports, sha256=reach.digest(shard)))
        print('Prepared ' + mid, flush=True)


def frontier_circle(points, envelope, standing=False):
    angles = np.deg2rad(np.arange(0, 360, 5))
    # One direction at a time avoids an observations x 72 allocation.
    frontier = np.array([np.max(z) if envelope == 'max' else np.quantile(z, float(envelope))
                        for a in angles for z in [points[:, 0] * np.cos(a) + points[:, 1] * np.sin(a)]])
    if standing:
        return 0., max(0., float(frontier.mean())), frontier
    design = np.column_stack([np.cos(angles), np.ones(len(angles))])
    c, r = np.linalg.lstsq(design, frontier, rcond=None)[0]
    return float(c), max(0., float(r)), frontier


def pooled(points, speed_bin, minimum):
    low, high = int(points.bin.min()), int(points.bin.max())
    width = 0
    while True:
        chosen = points[(points.bin >= speed_bin - width) & (points.bin <= speed_bin + width)]
        if len(chosen) >= minimum or speed_bin - width <= low and speed_bin + width >= high:
            return chosen, width
        width += 1


def read_horizon(path, matches, horizon):
    pieces = []
    for mid in matches:
        frame = pd.read_parquet(path / 'shards' / (mid + '.parquet'), filters=[('horizon', '=', int(horizon))])
        frame['match'] = mid
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


@contextmanager
def partition_horizon(path, matches, horizon):
    """Stage one horizon on disk; never concatenate the whole training pool in RAM."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory(prefix='reachability_horizon_') as directory:
        root = Path(directory)
        writers, counts = {}, {}
        try:
            for mid in matches:
                frame = pd.read_parquet(path / 'shards' / (mid + '.parquet'), filters=[('horizon', '=', int(horizon))])
                frame['match'] = mid
                for b, group in frame.groupby('bin'):
                    table = pa.Table.from_pandas(group, preserve_index=False)
                    if b not in writers:
                        writers[b] = pq.ParquetWriter(root / f'{b}.parquet', table.schema, compression='zstd')
                    writers[b].write_table(table)
                    counts[int(b)] = counts.get(int(b), 0) + len(group)
        finally:
            for writer in writers.values():
                writer.close()
        if not counts:
            raise ValueError('No trajectories at horizon ' + str(horizon))
        yield root, counts


def read_pooled_partition(root, counts, b, minimum):
    width = 0
    while sum(n for k, n in counts.items() if abs(k-b) <= width) < minimum:
        if b-width <= min(counts) and b+width >= max(counts):
            break
        width += 1
    frames = [pd.read_parquet(root / f'{k}.parquet') for k in sorted(counts) if abs(k-b) <= width]
    return pd.concat(frames, ignore_index=True), width


def fit(args):
    path = reach.model_path(args.model_id)
    if (path / 'circles.npz').exists() or (path / 'metadata.json').exists():
        raise ValueError('Model ID is immutable; use --source-model-id with a new ID for a changed fit')
    source = reach.model_path(args.source_model_id or args.model_id)
    meta = json.loads((source / 'preparation.json').read_text())
    for mid in meta['fit_matches'] + meta['holdout_matches']:
        shard = source / 'shards' / (mid + '.parquet')
        report = json.loads(shard.with_suffix('.json').read_text())
        if report['sha256'] != reach.digest(shard):
            raise ValueError('Corrupt shard: ' + mid)
    bins = sorted(set(int(b) for mid in meta['fit_matches']
                      for b in pd.read_parquet(source / 'shards' / (mid + '.parquet'), columns=['bin']).bin.unique()))
    bins = np.arange(0, max(bins) + 1)
    arrays = dict(speeds=(bins + .5) * .5 / 3.6, times=np.r_[0., HORIZONS / 25.])
    for e in args.envelopes:
        arrays['c_' + e] = np.zeros((len(bins), len(HORIZONS) + 1))
        arrays['r_' + e] = np.zeros((len(bins), len(HORIZONS) + 1))
    reports = []
    for j, h in enumerate(HORIZONS, 1):
        with partition_horizon(source, meta['fit_matches'], h) as (partition, counts):
            for i, b in enumerate(bins):
                chosen, width = read_pooled_partition(partition, counts, b, args.min_samples)
                xy = chosen[['x', 'y']].to_numpy()
                for e in args.envelopes:
                    c, r, frontier = frontier_circle(xy, e, standing=b == 0)
                    arrays['c_' + e][i, j], arrays['r_' + e][i, j] = c, r
                    errors = c * np.cos(np.deg2rad(np.arange(0, 360, 5))) + r - frontier
                    reports.append(dict(bin=int(b), time=h / 25, envelope=e, count=len(chosen), pooling_half_width=int(width),
                        episodes=len(chosen[['match', 'player', 'episode']].drop_duplicates()), low_support=len(chosen) < args.min_samples,
                        centre=c, radius=r, directional_rmse=float(np.sqrt(np.mean(errors ** 2))),
                        directional_min_error=float(errors.min()), directional_max_error=float(errors.max())))
        print('Fitted horizon ' + str(h / 25), flush=True)
    atomic_npz(path / 'circles.npz', **arrays)
    pd.DataFrame(reports).to_csv(path / 'fit_report.csv', index=False)
    meta.update(status='complete', envelopes=args.envelopes, min_samples=args.min_samples,
                preparation_model_id=args.source_model_id or args.model_id, fingerprint=reach.digest(path / 'circles.npz'),
                standing_policy='zero-centre least-squares directional support fit', interpolation='linear_speed_time')
    atomic_json(path / 'metadata.json', meta)


def diagnose(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta, arrays = reach.load_model(args.model_id)
    path = reach.model_path(args.model_id)
    source = reach.model_path(meta['preparation_model_id'])
    out = path / 'diagnostics'
    out.mkdir(exist_ok=True)
    rows, panel, bootstrap = [], [], []
    rng = np.random.default_rng(42)
    for h in HORIZONS:
        with partition_horizon(source, meta['holdout_matches'], h) as (partition, counts):
            for b in sorted(counts):
                points = pd.read_parquet(partition / f'{b}.parquet')
                speed = (b + .5) * .5 / 3.6
                xy = points[['x', 'y']].to_numpy()
                direction = (np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0])) % 360 // 30).astype(int)
                for e in meta['envelopes']:
                    c, r = reach.circle(arrays, speed, np.array(h / 25), e, 6., 0.)
                    inside = np.hypot(xy[:, 0] - c, xy[:, 1]) <= r
                    for sector in [-1] + list(range(12)):
                        mask = np.ones(len(points), bool) if sector == -1 else direction == sector
                        if mask.any():
                            rows.append(dict(bin=int(b), time=h / 25, envelope=e, sector=sector, count=int(mask.sum()),
                                covered=int(inside[mask].sum()), coverage=float(inside[mask].mean()),
                                clamped=bool(speed < arrays['speeds'][0] or speed > arrays['speeds'][-1])))
                if h in (25, 75, 125, 150) and b in (0, 10, 20, 40):
                    fig, (ax, support_ax) = plt.subplots(1, 2, figsize=(10, 4))
                    sample = xy[::max(1, len(xy)//3000)]
                    ax.scatter(sample[:, 0], sample[:, 1], s=1, alpha=.2)
                    theta = np.linspace(0, 2*np.pi, 200)
                    for e in meta['envelopes']:
                        c, r = reach.circle(arrays, speed, np.array(h / 25), e, 6., 0.)
                        ax.plot(c + r*np.cos(theta), r*np.sin(theta), label=e)
                        _, _, frontier = frontier_circle(xy, e, b == 0)
                        degrees = np.arange(0, 360, 5)
                        line, = support_ax.plot(degrees, frontier, label=e)
                        support_ax.plot(degrees, c*np.cos(np.deg2rad(degrees))+r,
                                        linestyle='--', color=line.get_color())
                    ax.set_aspect('equal'); ax.legend(); ax.set_title(f'Holdout bin {b}, {h/25}s')
                    support_ax.set_title('Directional support: holdout / fitted circle (dashed)')
                    support_ax.set_xlabel('Direction (degrees)'); support_ax.set_ylabel('Extent (m)')
                    fig.tight_layout()
                    fig.savefig(out / f'circle_{b}_{h}.png'); plt.close(fig)
                    panel.append((int(b), int(h)))
    pd.DataFrame(rows).to_csv(out / 'coverage.csv', index=False)
    coverage = pd.DataFrame(rows)
    total = coverage[coverage.sector == -1].groupby('envelope')[['covered', 'count']].sum()
    total['coverage'] = total.covered / total['count']
    total.to_csv(out / 'overall_coverage.csv')
    for b, h in panel:
        with partition_horizon(source, meta['fit_matches'], h) as (partition, counts):
            for repeat in range(args.bootstrap):
                sampled = rng.choice(meta['fit_matches'], len(meta['fit_matches']), replace=True)
                width = 0
                while True:
                    frames = [pd.read_parquet(partition / f'{k}.parquet') for k in sorted(counts) if abs(k-b) <= width]
                    if frames:
                        training = pd.concat(frames, ignore_index=True)
                        chosen = pd.concat([training[training['match'] == m] for m in sampled], ignore_index=True)
                    else:
                        chosen = pd.DataFrame()
                    if len(chosen) >= meta['min_samples'] or b-width <= min(counts) and b+width >= max(counts):
                        break
                    width += 1
                if chosen.empty:
                    continue
                for e in meta['envelopes']:
                    c, r, _ = frontier_circle(chosen[['x','y']].to_numpy(), e, b == 0)
                    bootstrap.append(dict(bin=b, horizon=h/25, envelope=e, repeat=repeat, centre=c, radius=r, pooling_half_width=width))
    pd.DataFrame(bootstrap).to_csv(out / 'bootstrap.csv', index=False)
    extensions = []
    for e in meta['envelopes']:
        fig, axes = plt.subplots(1, 2)
        for i in range(0, len(arrays['speeds']), max(1, len(arrays['speeds']) // 8)):
            s = arrays['speeds'][i]
            for ax, prefix in zip(axes, ['c_', 'r_']):
                ax.plot(arrays['times'], arrays[prefix + e][i], label=f'{s*3.6:.1f} km/h')
            for t in np.arange(5., 6.01, .2):
                c, r = reach.circle(arrays, s, np.array(t), e, 5., 8.)
                ec, er = reach.circle(arrays, s, np.array(t), e, 6., 0.)
                extensions.append(dict(envelope=e, speed=s, time=t, empirical_c=float(ec), empirical_r=float(er), extrapolated_c=float(c), extrapolated_r=float(r)))
        axes[0].set_title('Centre (m)'); axes[1].set_title('Radius (m)'); axes[1].legend(fontsize=6)
        fig.savefig(out / ('curves_' + e + '.png')); plt.close(fig)
    pd.DataFrame(extensions).to_csv(out / 'extrapolation.csv', index=False)
    atomic_json(out / 'metadata.json', dict(fingerprint=meta['fingerprint'], bootstrap=args.bootstrap, seed=42,
        note='Directional projection quantiles are not joint coverage guarantees. Quality counts are in shard reports.', panel=panel))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--model-id', required=True)
    split = p.add_mutually_exclusive_group(required=True)
    split.add_argument('--train-split', type=int); split.add_argument('--train-count', type=int)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--max-speed', type=float, default=15.)
    p.add_argument('--max-acceleration', type=float, default=30.)
    p.add_argument('--jump-tolerance', type=float, default=.5)
    p = sub.add_parser('fit'); p.add_argument('--model-id', required=True)
    p.add_argument('--source-model-id'); p.add_argument('--envelopes', nargs='+', choices=ENVELOPES, default=ENVELOPES)
    p.add_argument('--min-samples', type=int, default=10000)
    p = sub.add_parser('diagnose'); p.add_argument('--model-id', required=True)
    p.add_argument('--bootstrap', type=int, default=20)
    args = parser.parse_args(argv)
    for key in ('frame_stride', 'max_speed', 'max_acceleration', 'min_samples'):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
            parser.error(key + ' must be positive and finite')
    if hasattr(args, 'jump_tolerance') and (not math.isfinite(args.jump_tolerance) or args.jump_tolerance < 0):
        parser.error('jump tolerance must be nonnegative and finite')
    if hasattr(args, 'bootstrap') and args.bootstrap < 0:
        parser.error('bootstrap must be nonnegative')
    return args


if __name__ == '__main__':
    args = parse_args()
    globals()[args.command](args)
