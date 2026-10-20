"""Bounded integration audit; never publishes a feature run or latest pointer."""
import argparse
from dataclasses import asdict
import json
import os
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from datatools.carry_policy import CarryPolicy, DEFINITION
from datatools.reception_policy import KEY, RETIRED
from datatools.reception_refresh import refresh_match, resolve_source
from project_config import atomic_write_json
from scripts.preprocess_sportec import discover_match_files
from scripts.replace_carries import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id', default='latest')
    parser.add_argument('--output', default='tmp/reception_retirement_audit')
    parser.add_argument('--graph-cache', type=Path, help='Reuse carry graphs from a completed bounded audit after verifying carry geometry.')
    args = parser.parse_args()
    source, metadata = resolve_source(args.source_run_id)
    available = discover_match_files()
    chosen = ['DFL-MAT-J03YDU']
    for season in ['22_23', '23_24', '24_25']:
        chosen.append(next(f.match_id for f in sorted(available, key=lambda x:x.match_id)
            if f.season == season and f.match_id not in chosen and
            (source / 'action_graphs' / f'{f.match_id}.pt').exists()))
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new temporary audit directory: {output}')
    output.mkdir(parents=True)
    audit_metadata = dict(metadata, **{KEY: RETIRED}, status='audit_only', run_id=output.name,
                          carry_variant=dict(definition=DEFINITION, available=True,
                                             validation_parameters=asdict(CarryPolicy())))
    atomic_write_json(output / 'metadata.json', audit_metadata)
    reports = []
    print(f'Source: {source.name}; bounded matches: {chosen}', flush=True)
    for match_id in chosen:
        if args.graph_cache:
            from scripts import replace_carries as rc
            from project_config import get_resolved_action_path
            def cached_graphs(match, *, post_action=False, **kwargs):
                saved = pd.read_parquet(get_resolved_action_path(match_id,
                    intended_receiver_mode=metadata['intended_receiver_modes'][0],
                    root=args.graph_cache, use_carries=True))
                saved = saved.loc[saved.spadl_type.eq('ball_carry')]
                columns = ['frame_id', 'receive_frame_id', 'object_id', 'receiver_id']
                pd.testing.assert_frame_equal(match.actions[columns], saved[columns], check_dtype=False)
                name = 'post_action_graphs_carries' if post_action else 'action_graphs_carries'
                graphs = torch.load(args.graph_cache / name / f'{match_id}.pt',
                                    map_location='cpu', weights_only=False)
                return graphs[len(graphs)-len(saved):] if len(saved) else []
            rc.construct_graph_features = cached_graphs
        original = source / 'action_graphs' / f'{match_id}.pt'
        checksum = digest(original)
        destination = output / 'action_graphs' / original.name
        destination.parent.mkdir(exist_ok=True)
        try:
            os.link(original, destination)
        except OSError:
            shutil.copy2(original, destination)  # Bounded audit only; no full-run copy.
        report = refresh_match((str(source), str(output), audit_metadata, match_id,
                                True, asdict(CarryPolicy()), 1))
        assert digest(original) == checksum, 'Source graph changed'
        for mode in metadata['intended_receiver_modes']:
            old = pd.read_parquet(source / f'resolved_actions_{mode}' / f'{match_id}.parquet')
            new = pd.read_parquet(output / f'resolved_actions_{mode}' / f'{match_id}.parquet')
            assert old.index.equals(new.index)
            for column in ['frame_id', 'object_id', 'receiver_id', 'success']:
                pd.testing.assert_series_equal(old[column], new[column])
        for folder in source.glob('*labels*'):
            if folder.name.endswith('_carries') or not folder.is_dir():
                continue
            old_path, new_path = folder / original.name, output / folder.name / original.name
            if old_path.exists() and new_path.exists() and folder.name.endswith(('_model', '_angle_only')):
                from datatools import config
                old_labels = torch.load(old_path, map_location='cpu', weights_only=False)
                new_labels = torch.load(new_path, map_location='cpu', weights_only=False)
                torch.testing.assert_close(old_labels[:, config.LABEL_INDEX['intent_index']],
                                           new_labels[:, config.LABEL_INDEX['intent_index']], equal_nan=True)
        post = torch.load(output / 'post_action_graphs' / original.name, map_location='cpu', weights_only=False)
        combined = torch.load(output / 'post_action_graphs_carries' / original.name, map_location='cpu', weights_only=False)
        assert len(combined) == len(post) + report['carry_segments']
        assert [x is None for x in combined[:len(post)]] == [x is None for x in post]
        reports.append({k:v for k,v in report.items() if k not in ['artifacts','receptions']})
        print(json.dumps(reports[-1]), flush=True)
    atomic_write_json(output / 'audit_report.json', dict(source=source.name, reports=reports))
    print(f'Audit complete: {output / "audit_report.json"}', flush=True)


if __name__ == '__main__':
    main()
