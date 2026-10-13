"""Bounded, isolated real-generation comparison of Sportec windowing and prefetch.

Run from the repository with .venv/Scripts/python.exe. Generated versions and
latest pointers live only under the requested output directory (default: tmp).
"""
from __future__ import annotations

import time
PROGRAM_STARTED = time.perf_counter()

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from threading import Event, Thread
import traceback
import _thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import psutil
import torch
import pc_xpass_match_cache as match_cache
import pc_xpass_resume as resume
import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate


def ready_worker(directory):
    directory = Path(directory)
    (directory / f'{os.getpid()}.ready').write_text('ready')
    started = time.perf_counter()
    while len(list(directory.glob('*.ready'))) < 8:
        if time.perf_counter() - started > 45:
            raise RuntimeError('Eight-worker readiness barrier timed out')
        time.sleep(.05)
    return os.getpid()


class MemoryMonitor:
    def __enter__(self):
        self.done, self.peak_parent, self.peak_workers, self.peak_total = Event(), 0, 0, 0
        self.thread = Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        parent = psutil.Process()
        while not self.done.is_set():
            try:
                parent_rss = parent.memory_info().rss
                workers = 0
                for child in parent.children(recursive=True):
                    try:
                        workers += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self.peak_parent = max(self.peak_parent, parent_rss)
                self.peak_workers = max(self.peak_workers, workers)
                self.peak_total = max(self.peak_total, parent_rss + workers)
            except psutil.Error:
                pass
            self.done.wait(1)

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()

    def summary(self):
        return dict(peak_parent_rss_bytes=self.peak_parent, peak_workers_rss_bytes=self.peak_workers,
                    peak_total_rss_bytes=self.peak_total, sample_interval_seconds=1)


def arguments(options, case, window, prefetch):
    argv = [
        '--pc-xpass', '--pc-xpass-id', case, '--sportec-feature-run-id', options.feature_run_id,
        '--match-id', options.match_id, '--no-skillcorner', '--no-hawkeye', '--no-benchmark',
        '--sportec-runtime-row-window', str(window), '--num-workers', '8',
        '--physical-batch-size', '16', '--worker-thread-limit', '1', '--scope', 'actions',
        '--top-pass', '5', '10', '--top-n-values', '10', '25', '--top-xt', '--export-lane-control',
        '--reaction-time', 'dist_pass', '--dist-pass-div', '50',
        '--max-player-speed', '5', '--max-player-speed-off', '5', '--max-player-speed-def', '5',
    ]
    return generate.parse_args(argv + (['--sportec-runtime-prefetch'] if prefetch else []))


def compare(expected, actual):
    left, details_left, a = expected
    right, details_right, b = actual
    assert left.keys() == right.keys()
    for kind in left:
        keys = ['state_id'] + (['receiver_id', 'opponent_id'] if kind == 'lane_control' else [])
        pd.testing.assert_frame_equal(left[kind].sort_values(keys).reset_index(drop=True),
            right[kind].sort_values(keys).reset_index(drop=True), check_exact=True, check_like=True)
    assert details_left == details_right
    for key in ('manifest', 'descriptors', 'provenance', 'selection_report', 'coverage'):
        assert a[key] == b[key], key
    lane_sort = lambda rows: sorted(rows, key=lambda row: (row['possession_provenance_id'], row['state_frame_id']))
    assert lane_sort(a['lane_states']) == lane_sort(b['lane_states'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget-seconds', type=int, default=900)
    parser.add_argument('--match-id', default='DFL-MAT-J03WEL')
    parser.add_argument('--feature-run-id', default='feature_20260910T110759_053479_8b508137')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'tmp' / ('pc_prefetch_' + time.strftime('%Y%m%dT%H%M%S')))
    options = parser.parse_args()
    if options.budget_seconds < 120:
        parser.error('Budget must allow at least 120 seconds for warm-up and safe shutdown')
    output = options.output_dir.resolve()
    # Explicitly prevent this diagnostic from writing into the production cache tree.
    production = versions.config.PC_XPASS_DIR.resolve()
    if output == production or production in output.parents or output in production.parents:
        parser.error('Measurement output must be outside the production pc-xPass tree')
    output.mkdir(parents=True, exist_ok=False)
    deadline = PROGRAM_STARTED + options.budget_seconds
    result = dict(budget_seconds=options.budget_seconds, output=str(output),
                  feature_run_id=options.feature_run_id, match_id=options.match_id, cases=[])
    latest = production / 'latest.json'
    digest = lambda: hashlib.sha256(latest.read_bytes()).hexdigest() if latest.exists() else None
    original_latest = digest()
    original_root = versions.config.PC_XPASS_DIR
    versions.config.PC_XPASS_DIR = output / 'pc_xpass'
    stop_watchdog = Event()
    def watchdog():
        if not stop_watchdog.wait(max(0, deadline - time.perf_counter() - 45)):
            _thread.interrupt_main()
    timer = Thread(target=watchdog, daemon=True)
    timer.start()
    original_prewarm = generate.prewarm_runtime_items
    def bounded_prewarm(*args, **kwargs):
        if time.perf_counter() > deadline - 60:
            raise KeyboardInterrupt('Measurement budget reached; settling work before cleanup')
        return original_prewarm(*args, **kwargs)
    generate.prewarm_runtime_items = bounded_prewarm
    physics.configure_physical_worker_thread_limit(1)
    def persist():
        result['total_seconds_including_imports'] = time.perf_counter() - PROGRAM_STARTED
        (output / 'results.json').write_text(json.dumps(result, indent=2, default=str))
    expected = None
    chosen_window = 256
    args = None
    try:
        warm_args = arguments(options, 'warmup_inputs', 128, False)
        generate.prepare_runtime_graph_schema(warm_args)
        feature_id = generate.resolve_feature_run_id(options.feature_run_id, required=True, allow_latest=False)
        feature_root = generate.resolve_feature_root(feature_id)
        _, return_type, receiver_mode = generate.resolve_reference_label_context(feature_id, feature_root, warm_args)
        preparation = generate.prepare_sportec_windows(warm_args, options.match_id, feature_root, return_type, receiver_mode)
        try:
            first_window = next(preparation)
            warmup = [item for unit in first_window['units'] for item in unit['items']]
        finally:
            preparation.close()
        # Retain exactly the first 128 real states, independent of group boundaries;
        # this slice is only warm-up, never a timed generation comparison.
        warm_items, remaining = [], 128
        for item in warmup:
            count = min(remaining, len(item['graphs']))
            if count:
                warm_items.append({**item, 'graphs': item['graphs'][:count], 'labels': item['labels'][:count],
                                   'state_frame_id': item['state_frame_id'][:count]})
                remaining -= count
        assert remaining == 0
        for index in range(3):
            if index == 2:
                first, second = result['cases'][:2]
                if second['elapsed_seconds'] <= .95 * first['elapsed_seconds']:
                    chosen_window = 512
            window = (256, 512, chosen_window)[index]
            case = f'case_{index}_{window}' + ('_prefetch' if index == 2 else '_sequential')
            # Do not start a case that would consume the verification/shutdown reserve.
            estimate = max([row['elapsed_seconds'] for row in result['cases']] or [220]) + 55
            if time.perf_counter() + estimate > deadline:
                result['budget_limited'] = True
                break
            args = arguments(options, case, window, index == 2)
            generate.prepare_pass_height_context(args)
            generate.prepare_runtime_graph_schema(args)
            resume.prepare_contracts(args, ['sportec'])
            versions.start_generation(args)
            resume.persist_contracts(args)
            print(f'Starting {case}; {deadline - time.perf_counter():.1f}s remain.', flush=True)
            with physics.RuntimeComputeSession(num_workers=8, worker_thread_limit=1, row_window=window,
                                              process_start_method='spawn') as session:
                ready = output / f'ready_{index}'
                ready.mkdir()
                started = time.perf_counter()
                session._executor = ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn'),
                    initializer=physics.configure_physical_worker_thread_limit, initargs=(1,))
                session.pool_creations += 1
                pids = {future.result(timeout=60) for future in
                        [session._executor.submit(ready_worker, str(ready)) for _ in range(8)]}
                assert len(pids) == 8
                original_prewarm(warm_items, cache_dir=output / f'warmup_{index}', args=args, compute_session=session)
                assert session.windows[-1]['worker_processes_used'] == 8
                warmup_seconds = time.perf_counter() - started
                offset = len(session.windows)
                session._idle_since = time.perf_counter()
                with MemoryMonitor() as memory:
                    started = time.perf_counter()
                    summary = generate.run_runtime_sportec_possessions(args, compute_session=session)
                    elapsed = time.perf_counter() - started
                assert not summary['skipped'], summary['skipped']
                assert not summary['stats']['skipped_all_nan']
                versions.finish_generation(args, success=True, coverage={'sportec': summary})
                root = Path(args._pc_version_root) / 'sportec'
                verification_started = time.perf_counter()
                actual = match_cache.load_match(root, options.match_id, include_lane=True)
                assert actual[2]['status'] == 'completed'
                if expected is None:
                    expected = actual
                else:
                    compare(expected, actual)
                row = dict(case=case, row_window=window, prefetch=index == 2,
                    elapsed_seconds=elapsed, warmup_seconds=warmup_seconds, ready_pids=sorted(pids),
                    verification_seconds=time.perf_counter() - verification_started,
                    states=len(actual[0]['matches']), lane_rows=len(actual[0]['lane_control']),
                    exact_equal=True, memory=memory.summary(), windows=session.windows[offset:],
                    preparation=session.preparation_reports, publication_seconds=session.publication_seconds,
                    stats={k: v for k, v in summary['stats'].items() if k != 'scheduling'})
            row['session'] = session.summary()
            result['cases'].append(row)
            persist()
            print(f'{case}: {row["states"]} states in {elapsed:.2f}s; exact checks passed.', flush=True)
    except BaseException as exc:
        result['incomplete'] = True
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        if args is not None:
            versions.finish_generation(args, success=False)
        print(result['traceback'], flush=True)
    finally:
        stop_watchdog.set()
        timer.join()
        generate.prewarm_runtime_items = original_prewarm
        versions.config.PC_XPASS_DIR = original_root
        result['production_latest_unchanged'] = digest() == original_latest
        result['chosen_window'] = chosen_window
        persist()
        print(f'Results: {output}; total {result["total_seconds_including_imports"]:.2f}s.', flush=True)

    return 0 if len(result['cases']) == 3 and not result.get('incomplete') else 1


if __name__ == '__main__':
    sys.exit(main())
