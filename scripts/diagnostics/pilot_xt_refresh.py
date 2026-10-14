"""Isolated real-data validation. Never publishes a feature run or fits an xT model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from datetime import datetime
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import psutil
import torch
from datatools.artifact_storage import clone_feature_artifacts
from datatools.match import Match
from datatools.xt import XTXYLogitModel, annotate_match_xt
from datatools.xt_refresh import (TARGET_COLUMNS, file_signature, fingerprint, read_annotated_events,
                                 refresh_inventory, refresh_xt)
from project_config import atomic_write_json, get_resolved_action_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('data/features/runs/feature_20260910T110759_053479_8b508137'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.source.parent.parent / 'xt_refresh_pilots' / datetime.now().strftime('pilot_%Y%m%dT%H%M%S')
    if output.exists():
        raise FileExistsError(output)
    seed, derived, sidecars = [output / name for name in ('source_subset', 'refreshed', 'annotations')]
    seed.mkdir(parents=True)
    sidecars.mkdir()
    metadata = json.loads((args.source / 'metadata.json').read_text())
    match_ids = ['DFL-MAT-J03WEL', 'DFL-MAT-J03YDU', 'DFL-MAT-J04034']
    metadata['split_manifest'] = {**metadata.get('split_manifest', {}), 'universe_count': len(match_ids)}
    source_files = []
    for directory in args.source.iterdir():
        if not directory.is_dir():
            continue
        (seed / directory.name).mkdir()
        for mid in match_ids:
            for suffix in ('.pt', '.parquet'):
                path = directory / (mid + suffix)
                if path.exists():
                    os.link(path, seed / directory.name / path.name)
                    source_files.append(path)
    atomic_write_json(seed / 'metadata.json', metadata)
    original_signatures = {p: file_signature(p) for p in source_files}
    original_hashes = {p: fingerprint(p) for p in source_files if 'graphs' not in p.parent.name}
    production_annotations = {mid: fingerprint(Path('data/xT/matches') / f'{mid}.csv') for mid in match_ids}
    for mid in match_ids[1:]:
        shutil.copy2(Path('data/xT/matches') / f'{mid}.csv', sidecars / f'{mid}.csv')
    atomic_write_json(output / 'source_baseline.json', {str(p): {'signature': original_signatures[p], 'sha256': original_hashes.get(p)} for p in source_files})
    clone_stats = clone_feature_artifacts(seed, derived)
    try:
        refresh_xt(derived, metadata, xt_match_dir=sidecars)
    except FileNotFoundError as error:
        missing_annotation_error = str(error)
    else:
        raise AssertionError('Missing annotation was accepted')
    assert all(file_signature(p) == original_signatures[p] for p in source_files)
    fit = json.loads(Path('data/xT/fit_metadata.json').read_text())
    model = XTXYLogitModel(coefficients=fit['coefficients'], terms=tuple(fit['terms']))
    events = pd.read_csv(Path('data/event_synced') / f'{match_ids[0]}.csv')
    annotated, _ = annotate_match_xt(events, model)
    annotated[['action_id', 'xT', 'scores_xT', 'concedes_xT']].to_csv(sidecars / f'{match_ids[0]}.csv', index=False)
    process = psutil.Process()
    memory = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(0.05):
            memory.append(process.memory_info().rss)
    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    original_load, original_parquet = torch.load, pd.read_parquet
    def guarded_load(path, *a, **kw):
        if 'graphs' in Path(path).parent.name:
            raise AssertionError(f'Refresh loaded graph: {path}')
        return original_load(path, *a, **kw)
    def guarded_parquet(path, *a, **kw):
        if 'tracking' in str(path):
            raise AssertionError(f'Refresh loaded tracking: {path}')
        return original_parquet(path, *a, **kw)
    try:
        with patch('torch.load', side_effect=guarded_load), patch('pandas.read_parquet', side_effect=guarded_parquet):
            report = refresh_xt(derived, metadata, xt_match_dir=sidecars)
    finally:
        stop.set()
        sampler.join()
    refresh_peak_rss, refresh_baseline_rss = max(memory), memory[0]
    print(f"Refresh completed in {report['elapsed_seconds']:.2f}s; validating normal label construction...", flush=True)
    inventory, returns, modes, carries = refresh_inventory(derived, metadata)
    compared_rows, compared_files, maximum_difference = 0, 0, 0.0
    validation_started = time.perf_counter()
    for mid, artifacts in inventory.items():
        events = read_annotated_events(Path('data/event_synced') / f'{mid}.csv', sidecars / f'{mid}.csv')
        tracking = pd.read_parquet(Path('data/tracking_processed') / f'{mid}.parquet')
        normal_labels = {}
        for mode in modes:
            for carry in ([False, True] if carries else [False]):
                actions = pd.read_parquet(get_resolved_action_dir(mode, root=seed, use_carries=carry) / f'{mid}.parquet')
                # Skip __init__ so the reference cannot consult production sidecars.
                match = Match.__new__(Match)
                match.events = events.copy()
                match.tracking, match.action_type, match.fps, match.include_goals = tracking, 'all', 25, False
                for ret in returns:
                    labels = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                    intended_receiver_mode=mode, return_type=ret)
                    normal_labels[(mode, carry, ret)] = pd.DataFrame(labels[:, TARGET_COLUMNS].numpy(),
                                                                    index=labels[:, 0].numpy().astype(int))
        for artifact in artifacts:
            source = seed / artifact.path.relative_to(derived)
            before, after = [torch.load(path, weights_only=True) for path in (source, artifact.path)]
            reference = normal_labels[(artifact.mode, artifact.carries, artifact.return_type)]
            expected = torch.as_tensor(reference.loc[after[:, 0].numpy().astype(int)].to_numpy(), dtype=after.dtype)
            difference = float((after[:, TARGET_COLUMNS] - expected).abs().max())
            maximum_difference = max(maximum_difference, difference)
            torch.testing.assert_close(after[:, TARGET_COLUMNS], expected, rtol=0, atol=0)
            keep = [i for i in range(before.shape[1]) if i not in TARGET_COLUMNS]
            torch.testing.assert_close(after[:, keep], before[:, keep], rtol=0, atol=0, equal_nan=True)
            compared_rows += len(after)
            compared_files += 1
        print(f'Validated {mid}', flush=True)
    changed_signatures = [str(p) for p, signature in original_signatures.items() if file_signature(p) != signature]
    assert not changed_signatures, changed_signatures
    changed_source = [str(p) for p, digest in original_hashes.items() if fingerprint(p) != digest]
    assert not changed_source, changed_source
    changed_annotations = [mid for mid, digest in production_annotations.items() if fingerprint(Path('data/xT/matches') / f'{mid}.csv') != digest]
    assert not changed_annotations, changed_annotations
    graphs = [p for p in source_files if 'graphs' in p.parent.name]
    assert all(os.path.samefile(p, derived / p.relative_to(args.source)) for p in graphs)
    second = refresh_xt(derived, metadata, xt_match_dir=sidecars)
    assert second['rewritten_files'] == 0
    result = dict(matches=match_ids, missing_annotation_rejected=missing_annotation_error,
                  clone=clone_stats, refresh={k: v for k, v in report.items() if k not in {'artifacts', 'input_sha256'}},
                  compared_label_files=compared_files, compared_label_rows=compared_rows,
                  maximum_target_difference=maximum_difference, non_xt_columns_unchanged=True,
                  source_data_hashes_unchanged=True, source_graph_signatures_unchanged=True,
                  production_annotations_unchanged=True, shared_graph_files=len(graphs),
                  shared_graph_bytes=sum(p.stat().st_size for p in graphs),
                  refresh_peak_rss_bytes=refresh_peak_rss, refresh_baseline_rss_bytes=refresh_baseline_rss,
                  validation_seconds=time.perf_counter() - validation_started,
                  second_refresh_rewritten_files=second['rewritten_files'])
    atomic_write_json(output / 'pilot_report.json', result)
    print(json.dumps(result, indent=2))
    print(f"Pilot report: {output / 'pilot_report.json'}")


if __name__ == '__main__':
    main()
