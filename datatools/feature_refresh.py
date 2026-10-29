"""One derived-run transaction for pass refresh, reception retirement and carries."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import copy
import json
from pathlib import Path
import subprocess
import sys

from datatools.artifact_storage import clone_feature_artifacts
from datatools.graph_validation import graph_validation_enabled, merge_validation_stats
from datatools.carry_policy import CarryPolicy, DEFINITION
from datatools.graph_feature import configure_worker_thread_limit, resolve_num_workers
from datatools.pass_trajectory import (PASS_DURATION_KEY, PASS_DURATION, ANGLE_ONLY_KEY, ANGLE_ONLY,
                                      ORIGINAL_KEY, ORIGINAL, MAX_DURATION_KEY)
from datatools.reception_policy import KEY, RETIRED
from datatools.trajectory import TRAJECTORY_SPEED_KEY, TRAJECTORY_SPEED
from project_config import atomic_write_json, generate_run_id, get_feature_run_root, write_run_metadata, write_latest_run


def _refresh_match(task):
    source, output, metadata, match_id, settings, threads = task
    from datatools.reception_refresh import refresh_match as retire_match
    from datatools.trajectory_refresh import refresh_directory_match
    from scripts.replace_carries import rebuild_match
    configure_worker_thread_limit(threads)
    source, output = Path(source), Path(output)
    retirement = None
    if settings['retire_elastic']:
        retirement = retire_match((str(source), str(output), metadata, match_id, False, settings['carry_parameters'], threads))
    report = refresh_directory_match(output, metadata, match_id, threads,
                                     graph_validation=metadata.get("graph_validation", True))
    report["graph_validation"] = merge_validation_stats(
        [report.get("graph_validation"), (retirement or {}).get("graph_validation")],
        metadata.get("graph_validation", True))
    if settings['replace_carries']:
        carry = rebuild_match((str(source), str(output), dict(metadata, _canonical_root=str(output)),
                               match_id, settings['carry_parameters'], threads))
        report['carry_segments'] = carry['segments']
        report['graph_validation'] = merge_validation_stats([report.get('graph_validation'), carry.get('graph_validation')], metadata.get('graph_validation', True))
    carry_artifacts = carry.get('artifacts', {}) if settings['replace_carries'] else {}
    report['artifacts'] = {**carry_artifacts, **{str(p.relative_to(output)): p.stat().st_size
        for folder in output.iterdir() if folder.is_dir()
        for p in folder.glob(match_id + '.*') if p.suffix in {'.pt', '.parquet'}}}
    atomic_write_json(output / 'feature_refresh' / (match_id + '.json'), report)
    return report


def _complete(output, match_id):
    try:
        report = json.loads((output / 'feature_refresh' / (match_id + '.json')).read_text())
        return all((output / p).is_file() and (output / p).stat().st_size == size
                   for p, size in report['artifacts'].items())
    except (OSError, ValueError, KeyError):
        return False


def run_combined_refresh(args):
    from scripts import replace_carries as rc
    from datatools.reception_refresh import resolve_source
    source, original = resolve_source(args.extend_feature_run_id)
    if getattr(args, 'in_place', False):
        raise ValueError('Combined refresh creates a derived run; --in-place is unsupported.')
    replace, retire = bool(args.replace_carries), bool(args.retire_elastic)
    if (original.get('carry_variant') or {}).get('available') and not replace:
        raise ValueError('Refreshing pass endpoints in a carry-enabled run requires --replace-carries')
    ids = rc.match_ids(source, original)
    defaults = {**asdict(CarryPolicy()), **(original.get('carry_variant') or {}).get('validation_parameters', {})}
    explicit = getattr(args, 'explicit_carry_parameters', set())
    parameters = {key: getattr(args, 'carry_' + key, value) if key in explicit else value
                  for key, value in defaults.items()}
    CarryPolicy(**parameters)
    settings = dict(source_sha256=rc.digest(source / 'metadata.json'), matches=ids,
        retire_elastic=retire, replace_carries=replace, carry_definition=DEFINITION, carry_parameters=parameters,
        artifact_copy_mode=args.artifact_copy_mode, **{PASS_DURATION_KEY: PASS_DURATION,
        TRAJECTORY_SPEED_KEY: TRAJECTORY_SPEED, ANGLE_ONLY_KEY: ANGLE_ONLY, ORIGINAL_KEY: ORIGINAL,
        MAX_DURATION_KEY: getattr(args, MAX_DURATION_KEY, None)})
    output = get_feature_run_root(args.run_id or generate_run_id('feature'))
    if output == source:
        raise ValueError('Combined refresh requires a new derived run')
    metadata = copy.deepcopy(original)
    from datatools.pass_eligibility import policy_metadata
    metadata.update(policy_metadata())
    metadata["graph_validation"] = graph_validation_enabled(getattr(args, "graph_validation", True))
    metadata.update(run_id=output.name, base_feature_run_id=source.name, status='in_progress',
        created_at=datetime.now().isoformat(timespec='seconds'), command=subprocess.list2cmdline(sys.argv),
        feature_refresh=settings, **{PASS_DURATION_KEY: PASS_DURATION, TRAJECTORY_SPEED_KEY: TRAJECTORY_SPEED,
        ANGLE_ONLY_KEY: ANGLE_ONLY, ORIGINAL_KEY: ORIGINAL, MAX_DURATION_KEY: settings[MAX_DURATION_KEY]})
    if "model" in metadata.get("intended_receiver_modes", []):
        metadata["model_mode_regeneration_required"] = True
        metadata["previous_intended_receiver_model_id"] = metadata.pop("intended_receiver_model_id", None)
        metadata["intended_receiver_modes"] = [m for m in metadata["intended_receiver_modes"] if m != "model"]
    if retire:
        metadata[KEY] = RETIRED
    if replace:
        metadata['carry_variant'] = dict(definition=DEFINITION, available=False,
            validation_parameters=parameters, sidecar_root='carry_sidecars', artifact_suffix='_carries')
    cloned = False
    if output.exists():
        previous = json.loads((output / 'metadata.json').read_text(encoding='utf-8-sig'))
        if previous.get('status') not in {'in_progress', 'failed'} or previous.get('feature_refresh') != settings:
            raise ValueError('Resume requires identical source, output, operations and endpoint policy')
        cloned = previous.get('feature_refresh_clone_complete', False)
        metadata['created_at'] = previous.get('created_at', metadata['created_at'])
    metadata['feature_refresh_clone_complete'] = cloned
    print(f'Feature refresh source: {source.name}\nDerived run: {output.name}', flush=True)
    write_run_metadata(output, metadata)
    try:
        if not cloned:
            clone_feature_artifacts(source, output, mode=args.artifact_copy_mode, resume=True,
                exclude=lambda p: (replace and rc.excluded_artifact(p))
                    or any(part in {'feature_refresh', 'trajectory_refresh'} or part.endswith(('_model', '_model_carries')) for part in p.parts))
            metadata['feature_refresh_clone_complete'] = True
            write_run_metadata(output, metadata)
        tasks = [(str(source), str(output), metadata, mid, settings, args.worker_thread_limit)
                 for mid in ids if not _complete(output, mid)]
        workers = resolve_num_workers(args.num_workers)
        if workers == 1:
            for task in tasks:
                report = _refresh_match(task)
                print(f"Refreshed {report['match_id']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for report in pool.map(_refresh_match, tasks):
                    print(f"Refreshed {report['match_id']}", flush=True)
        if not all(_complete(output, mid) for mid in ids):
            raise ValueError('Incomplete combined feature refresh')
        reports = [json.loads((output / 'feature_refresh' / (mid + '.json')).read_text()) for mid in ids]
        metadata['skipped_actions'] = {r['match_id']: r['skipped_actions'] for r in reports if r['skipped_actions']}
        metadata['skipped_action_count'] = sum(len(r['skipped_actions']) for r in reports)
        metadata['graph_validation_summary'] = merge_validation_stats(
            [r.get('graph_validation') for r in reports], metadata['graph_validation'])
        if replace:
            metadata['carry_variant'].update(available=True, generated_at=datetime.now().isoformat(timespec='seconds'))
        metadata['status'] = 'completed'
        write_run_metadata(output, metadata)
        write_latest_run('feature', output.name)
    except BaseException as exc:
        metadata.update(status='failed', feature_refresh_error=str(exc))
        write_run_metadata(output, metadata)
        print(f'Resume with the same command and --run-id {output.name}', file=sys.stderr)
        raise
