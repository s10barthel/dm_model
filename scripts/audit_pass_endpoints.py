ÃƒÂ¯Ã‚Â»Ã‚Â¿"""Read-only audit of official pass endpoints and an optional duration safeguard."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from datatools.pass_trajectory import PassEndpointResolver, load_official_events, positive_duration, PASS_DURATION, _match_files
from project_config import DATA_ROOT, get_feature_run_root, load_feature_run_metadata


def audit_match(task):
    path, limit, count_labels = task
    actions = pd.read_parquet(path)
    actions = actions.loc[actions.action_type.eq('pass')]
    tracking = pd.read_parquet(DATA_ROOT / 'tracking_processed' / path.name,
        columns=['period_id', 'episode_id', 'ball_state'])
    events = load_official_events(path.stem)
    resolver = PassEndpointResolver(tracking, 25, events)
    files = _match_files().get(path.stem)
    rows, affected = [], set()
    for index, action in actions.iterrows():
        endpoint = resolver.resolve(action)
        exceeds = bool(endpoint.observed is not None and limit is not None and endpoint.duration > limit)
        if exceeds:
            affected.add(int(index))
        rows.append(dict(match_id=path.stem, season=files.season if files else 'unknown',
            action_index=int(index), action_id=str(action.get('action_id', index)),
            event_id=str(action.get('original_event_id', '')), success=bool(action.success),
            launch_frame=int(action.frame_id), endpoint_frame=endpoint.observed,
            endpoint_event_id=endpoint.event_id, source=endpoint.source, duration=endpoint.duration,
            unavailable_reason=endpoint.reason, affected=exceeds))
    label_counts = {}
    if affected and count_labels:
        import torch
        for folder in path.parent.parent.iterdir():
            label_path = folder / (path.stem + '.pt')
            if folder.is_dir() and 'labels' in folder.name and label_path.exists():
                labels = torch.load(label_path, map_location='cpu', weights_only=False)
                label_counts[folder.name] = sum(int(i) in affected for i in labels[:, 0].tolist())
    return rows, label_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-run-id', required=True)
    parser.add_argument('--max-pass-duration', type=positive_duration, default=None)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--match-id', action='append')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--count-label-rows', action='store_true')
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error('--num-workers must be positive')
    metadata = load_feature_run_metadata(args.feature_run_id, required=True)
    root = get_feature_run_root(args.feature_run_id)
    mode = 'original' if 'original' in metadata.get('intended_receiver_modes', ['original']) else metadata['intended_receiver_modes'][0]
    files = sorted((root / ('resolved_actions_' + mode)).glob('*.parquet'))
    if args.match_id:
        files = [p for p in files if p.stem in args.match_id]
    if not files:
        raise ValueError('No selected canonical action files')
    output = args.output_dir or ROOT / 'out' / ('pass_endpoint_audit_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(p, args.max_pass_duration, args.count_label_rows) for p in files]
    rows, counts = [], Counter()
    def collect(results):
        for number, (entries, labels) in enumerate(results, 1):
            rows.extend(entries); counts.update(labels)
            if number % 25 == 0 or number == len(tasks):
                print(f'Audited {number}/{len(tasks)} matches', flush=True)
    if args.num_workers == 1:
        collect(map(audit_match, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            collect(pool.map(audit_match, tasks))
    frame = pd.DataFrame(rows)
    from project_config import resolve_artifact_split
    split, _ = resolve_artifact_split({}, metadata, feature_run_id=args.feature_run_id)
    split_lookup = {mid: name for name in ("train", "test") for mid in split.get(name, [])}
    frame["split"] = frame.match_id.map(split_lookup).fillna("unknown")
    frame.to_csv(output / 'actions.csv', index=False)
    selected = frame.loc[frame.affected]
    selected.to_csv(output / 'affected_actions.csv', index=False)
    unavailable = frame.endpoint_frame.isna()
    report = dict(feature_run_id=args.feature_run_id, pass_endpoint_definition=PASS_DURATION,
        max_pass_duration=args.max_pass_duration, safeguard_applied=False,
        matches=len(files), unique_passes=len(frame), resolved_endpoints=int((~unavailable).sum()),
        already_unavailable=int(unavailable.sum()), affected_unique_passes=len(selected),
        affected_failed=int((~selected.success).sum()), affected_successful=int(selected.success.sum()),
        affected_label_rows=dict(counts),
        by_season=selected.groupby('season').size().to_dict(),
        by_source=selected.groupby('source').size().to_dict(),
        by_split=selected.groupby('split').size().to_dict(),
        unavailable_reasons=frame.loc[unavailable].unavailable_reason.value_counts().to_dict())
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2)); print(f'Report: {output}', flush=True)


if __name__ == '__main__':
    main()
