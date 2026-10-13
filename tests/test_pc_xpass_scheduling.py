"""Scheduling changes must preserve numerical rows and parent publication boundaries."""
from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import physical_pass_model as physics
import pc_xpass_versions as versions
from scripts import generate_physical_xpass as generate
from test_physical_xpass import make_graph, make_label


@pytest.fixture
def controlled_pool(monkeypatch):
    pools = []

    class Pool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.futures = []
            self.closed = False
            self.rows_at_submit = []
            pools.append(self)

        def submit(self, function, task):
            future = Future()
            future.work = (function, task)
            self.futures.append(future)
            self.rows_at_submit.append(sum(len(f.work[1]['misses']) for f in self.futures if not f.done()))
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures
            for future in self.futures:
                if not future.done():
                    if not future.cancel():
                        complete(future)
            assert all(f.done() for f in self.futures)
            self.closed = True

    def complete(future):
        function, task = future.work
        if not future.running():
            assert future.set_running_or_notify_cancel()
        try:
            future.set_result(function(task))
        except BaseException as exc:
            future.set_exception(exc)

    def wait(futures, **kwargs):
        # Deliberately finish the newest task first: completion order is not input order.
        future = list(futures)[-1]
        complete(future)
        return {future}, set(futures) - {future}

    monkeypatch.setattr(physics, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(physics, 'wait', wait)
    return pools


def session(*, rows=4, workers=2):
    return physics.RuntimeComputeSession(num_workers=workers, worker_thread_limit=1, row_window=rows)


def items(key='one', start=0, count=5):
    return [{'match_id': key, 'graphs': [make_graph() for _ in range(count)],
             'labels': torch.stack([make_label(action_index=i) for i in range(start, start + count)]),
             'frame_scope': 'frame_id', 'state_frame_id': list(range(start, start + count))}]


def kwargs(root, *, workers=2):
    return dict(cache_dir=root, source=physics.PC_XPASS_SOURCE, num_workers=workers,
                physical_batch_size=2, max_speed=3, min_speed=3, speed_step=2,
                angle_step=90, radial_gridsize=10, top_pass_values=[2],
                available_metrics=['max_xpass', 'top_pass2_xpass'], export_lane_control=True,
                show_progress=False)


def assert_cache_equal(left, right):
    for subdir in ('matches', 'lane_control'):
        expected_paths = list((left / subdir).glob('*.parquet'))
        assert {p.name for p in expected_paths} == {p.name for p in (right / subdir).glob('*.parquet')}
        for path in expected_paths:
            expected = pd.read_parquet(path)
            actual = pd.read_parquet(right / subdir / path.name)
            keys = ['action_index', 'frame_scope'] if subdir == 'matches' else physics.LANE_CONTROL_ID_COLUMNS + ['receiver_id', 'opponent_id']
            pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
                                          actual.sort_values(keys).reset_index(drop=True), check_exact=True, check_like=True)
    for path in (left / 'lane_control/coverage').glob('*.json'):
        expected = json.loads(path.read_text())
        actual = json.loads((right / 'lane_control/coverage' / path.name).read_text())
        sort = lambda rows: sorted(rows, key=lambda row: (row['action_index'], row['frame_scope']))
        expected['states'] = sort(expected['states'])
        actual['states'] = sort(actual['states'])
        assert expected == actual


@pytest.mark.parametrize('row_budget,batch', [(10, 4), (3, 4), (1, 4), (16, 16)])
def test_incremental_submission_preserves_all_inputs_and_budget(controlled_pool, monkeypatch, row_budget, batch):
    monkeypatch.setattr(physics, '_compute_runtime_physical_xpass_chunk', lambda task: {'rows': task['misses']})
    result = []
    with session(rows=row_budget) as compute:
        diagnostic = compute.compute([{'index': i} for i in range(25)], {'physical_batch_size': batch},
                                     lambda chunk: result.extend(chunk['rows']))
    assert sorted(r['index'] for r in result) == list(range(25))
    assert diagnostic['peak_outstanding_rows'] <= row_budget
    assert max(map(int, diagnostic['task_sizes'])) <= min(row_budget, batch)
    assert all(max(pool.rows_at_submit) <= row_budget and pool.closed for pool in controlled_pool)
    assert diagnostic['worker_busy_seconds'] >= 0 and diagnostic['tail_unused_worker_seconds'] >= 0


def test_lazy_pool_reuse_single_tasks_and_close(controlled_pool, monkeypatch):
    monkeypatch.setattr(physics, '_compute_runtime_physical_xpass_chunk', lambda task: {'rows': task['misses']})
    with session(rows=32, workers=8) as compute:
        compute.compute([{}], {'physical_batch_size': 16}, lambda result: None)
        assert not controlled_pool
        compute.compute([{}] * 32, {'physical_batch_size': 16}, lambda result: None)
        compute.compute([{}], {'physical_batch_size': 16}, lambda result: None)
        assert len(controlled_pool) == 1
        assert controlled_pool[0].kwargs['max_workers'] == 8
    assert controlled_pool[0].closed
    assert compute.summary()['pool_creations'] == 1
    with pytest.raises(RuntimeError, match='closed'):
        compute.compute([{}], {'physical_batch_size': 16}, lambda result: None)


@pytest.mark.parametrize('window', [1, 3, 8])
def test_exact_numerical_and_lane_outputs_across_windows(tmp_path, controlled_pool, window):
    reference, optimized = tmp_path / 'reference', tmp_path / 'optimized'
    data = items() + items('two', count=2)
    # Another physical possessor at the same frame must remain a separate state.
    data[1]['graphs'][0].x[:, 13] = torch.tensor([0., 1., 0.])
    physics.prewarm_physical_xpass_runtime_cache(data, **kwargs(reference, workers=1))
    with session(rows=window) as compute:
        stats = physics.prewarm_physical_xpass_runtime_cache(data, compute_session=compute, **kwargs(optimized))
    assert stats['cache_written'] == 7
    assert_cache_equal(reference, optimized)
    assert all('runtime' not in column for column in pd.read_parquet(optimized / 'matches/one.parquet'))


@pytest.mark.parametrize('dry_run', [False, True])
def test_hits_and_dry_run_do_not_create_pool(tmp_path, controlled_pool, dry_run):
    if not dry_run:
        physics.prewarm_physical_xpass_runtime_cache(items(), **kwargs(tmp_path, workers=1))
    with session() as compute:
        stats = physics.prewarm_physical_xpass_runtime_cache(items(), compute_session=compute, dry_run=dry_run, **kwargs(tmp_path))
    assert not controlled_pool
    assert stats['cache_misses'] == (5 if dry_run else 0)
    if dry_run:
        assert not list(tmp_path.rglob('*.parquet'))


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_failure_settles_work_before_cleanup_without_partial_publication(tmp_path, controlled_pool, monkeypatch, error):
    original = physics._compute_runtime_physical_xpass_chunk
    def fail(task):
        # Put one other future into a running state; shutdown must settle it.
        for future in controlled_pool[0].futures:
            if not future.running() and not future.done():
                future.set_running_or_notify_cancel()
                break
        raise error('worker failed')
    monkeypatch.setattr(physics, '_compute_runtime_physical_xpass_chunk', fail)
    with session() as compute:
        with pytest.raises(error, match='worker failed'):
            physics.prewarm_physical_xpass_runtime_cache(items(), compute_session=compute, **kwargs(tmp_path))
        assert controlled_pool[0].closed
        assert all(f.done() for f in controlled_pool[0].futures)
        assert not list((tmp_path / 'matches').glob('*.parquet'))
        assert not list((tmp_path / 'lane_control').glob('*.parquet'))
        monkeypatch.setattr(physics, '_compute_runtime_physical_xpass_chunk', original)
        physics.prewarm_physical_xpass_runtime_cache(items(), compute_session=compute, **kwargs(tmp_path))
        assert len(controlled_pool) == 2


def test_batch_failure_retries_original_units_serially(tmp_path, controlled_pool, monkeypatch):
    args = generate.parse_args(['--pc-xpass', '--num-workers', '2', '--physical-batch-size', '2',
        '--min-speed', '3', '--max-speed', '3', '--angle-step', '90', '--radial-gridsize', '10'])
    original = physics._compute_runtime_physical_xpass_chunk
    calls = 0
    def fail_once(task):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('worker failed')
        return original(task)
    monkeypatch.setattr(physics, '_compute_runtime_physical_xpass_chunk', fail_once)
    units = [generate.make_runtime_buffer_unit(display_key=key, skip_keys=[key], items=items(key, count=3))
             for key in ('one', 'two')]
    stats, skipped = generate.empty_runtime_stats(tmp_path), {}
    with session() as compute:
        failed = generate.flush_runtime_buffer(units, cache_dir=tmp_path, args=args, stats=stats,
            skipped=skipped, progress_desc='test', compute_session=compute)
    assert failed == 0 and not skipped and not units
    assert stats['serial_retry_recovered'] == 2
    assert stats['serial_retry_attempts'] == 2
    assert stats['cache_written'] == 6
    assert len(controlled_pool) == 1  # Serial retry and verification must bypass the pool.


@pytest.mark.parametrize('dataset', ['sportec', 'hawkeye'])
def test_pc_only_row_window_validation_and_cache_settings(dataset):
    flag = f'--{dataset}-runtime-row-window'
    with pytest.raises(SystemExit):
        generate.parse_args([flag, '128'])
    for value in ('0', '-1'):
        with pytest.raises(SystemExit):
            generate.parse_args(['--pc-xpass', flag, value])
    default = generate.parse_args(['--pc-xpass', '--num-workers', '8'])
    selected = generate.parse_args(['--pc-xpass', '--num-workers', '8', flag, '128'])
    assert generate.resolve_runtime_row_window(default, f'{dataset}_runtime_row_window') == 256
    assert versions.settings(default) == versions.settings(selected)
    with pytest.raises(SystemExit):
        generate.parse_args(['--pc-xpass', '--sportec-runtime-match-window=4'])
    assert generate.parse_args(['--sportec-runtime-match-window', '2']).sportec_runtime_match_window == 2


@pytest.mark.parametrize('dataset', ['sportec', 'skillcorner', 'benchmark', 'hawkeye'])
def test_runner_session_ownership_is_shared_and_never_serialized(dataset):
    args = generate.parse_args(['--pc-xpass', '--num-workers', '2'])
    received = []
    @generate.runtime_compute_runner(dataset)
    def runner(args, *, compute_session):
        received.append(compute_session)
        assert all(not isinstance(value, physics.RuntimeComputeSession) for value in vars(args).values())
        return {'stats': {}}
    result = runner(args)
    assert received[0]._closed and result['stats']['scheduling']['pool_creations'] == 0


@pytest.mark.parametrize('pc', [True, False])
def test_hawkeye_buffers_pc_situations_and_preserves_ordinary_behavior(tmp_path, monkeypatch, pc):
    args = generate.parse_args((['--pc-xpass', '--hawkeye-runtime-row-window', '2'] if pc else []) + ['--num-workers', '1'])
    monkeypatch.setattr(versions, 'cache_dir', lambda *a: tmp_path)
    monkeypatch.setattr(generate, 'get_runtime_physical_xpass_dir', lambda *a: tmp_path)
    tracking = pd.DataFrame({'id': ['a', 'b', 'c']})
    monkeypatch.setattr(generate, 'load_hawkeye_tracking', lambda *a: tracking)
    monkeypatch.setattr(generate, 'clean_hawkeye_tracking', lambda frame: frame)
    monkeypatch.setattr(generate, 'load_hawkeye_ball', lambda *a: pd.DataFrame())
    monkeypatch.setattr(generate, 'clean_hawkeye_ball', lambda frame: frame)
    monkeypatch.setattr(generate, 'resolve_situation_ids', lambda *a, **k: ['a', 'b', 'c'])
    def build(frame, *a, **k):
        return (SimpleNamespace(match_id='hawkeye_' + frame.id.iloc[0], graph_features_0=[make_graph()],
                                labels=torch.stack([make_label()])),)
    monkeypatch.setattr(generate, 'build_hawkeye_situation', build)
    monkeypatch.setattr(generate, 'write_runtime_dataset_metadata', lambda *a, **k: None)
    calls = []
    def prewarm(data, **kw):
        calls.append((data, kw.get('compute_session')))
        return {'cache_misses': len(data), 'cache_written': len(data)}
    monkeypatch.setattr(generate, 'prewarm_runtime_items', prewarm)
    result = generate.run_runtime_hawkeye(args)
    assert [len(data) for data, _ in calls] == ([2, 1] if pc else [1, 1, 1])
    assert [row['match_id'] for data, _ in calls for row in data] == ['hawkeye_a', 'hawkeye_b', 'hawkeye_c']
    assert not result['skipped']
    if pc:
        assert calls[0][1] is calls[1][1] and calls[0][1]._closed
    else:
        assert all(compute is None for _, compute in calls)

@pytest.mark.parametrize('dataset', ['sportec', 'skillcorner', 'benchmark'])
def test_dataset_runners_keep_session_and_publication_boundaries(tmp_path, monkeypatch, dataset):
    import scripts.run_relevant_models as inference_runner
    args = generate.parse_args(['--pc-xpass', '--num-workers', '2', f'--{dataset}-runtime-row-window', '2'])
    monkeypatch.setattr(versions, 'cache_dir', lambda *a: tmp_path)
    monkeypatch.setattr(generate, 'write_runtime_dataset_metadata', lambda *a, **k: None)
    calls, committed, ended = [], [], []
    def prewarm(data, **kw):
        calls.append((data, kw['compute_session']))
        return {'cache_misses': generate.runtime_items_row_count(data), 'cache_written': generate.runtime_items_row_count(data)}
    monkeypatch.setattr(generate, 'prewarm_runtime_items', prewarm)
    class Transaction:
        def __init__(self, key):
            self.key, self.failed = key, False
            self.provenance = self.manifest = None
        def set_provenance(self, value):
            self.provenance = value
        def set_manifest(self, value):
            self.manifest = value
    monkeypatch.setattr(generate.match_cache, 'begin', lambda args, root, key: Transaction(key))
    monkeypatch.setattr(generate.match_cache, 'finish', lambda tx, args: committed.append((tx, len(calls))))
    monkeypatch.setattr(generate.match_cache, 'end', lambda tx: ended.append(tx.key))
    def state(key, event):
        frames = [event * 10 + i for i in range(3)]
        actions = pd.DataFrame({'object_id': ['home_1'] * 3, 'player_id': ['home_1'] * 3,
                                'period_id': [1] * 3}, index=frames)
        return SimpleNamespace(match_id=key, pc_cache_match_id=f'{key}_{event}', event_index=event,
            graph_features_0=[make_graph() for _ in frames], labels=torch.stack([make_label(action_index=i) for i in frames]),
            actions=actions, original_start_frame=frames[0], original_end_frame=frames[-1], frame_selection={'frames': frames},
            pc_state_descriptor={'match_id': key, 'possession_id': event, 'possessor_id': 'home_1'})
    if dataset == 'skillcorner':
        monkeypatch.setattr(generate, 'discover_skillcorner_matches', lambda *a, **k: (['m1', 'm2'], {}))
        monkeypatch.setattr(generate, 'build_skillcorner_match_context',
                            lambda key, *a: {'key': key, 'events': pd.DataFrame({'index': [1, 2]})})
        monkeypatch.setattr(generate, 'build_skillcorner_possession', lambda context, event, **kw: (state(context['key'], event), {}))
        result = generate.run_runtime_skillcorner(args)
    elif dataset == 'sportec':
        monkeypatch.setattr(generate, 'resolve_feature_run_id', lambda *a, **k: 'features')
        monkeypatch.setattr(generate, 'resolve_feature_root', lambda *a: tmp_path)
        monkeypatch.setattr(generate, 'resolve_reference_label_context', lambda *a: (tmp_path, 'return', 'angle_only'))
        monkeypatch.setattr(generate, 'resolve_match_ids', lambda *a: ['m1', 'm2'])
        def match(key, **kwargs):
            return SimpleNamespace(key=key, possession_report={'match': key},
                sportec_state_plan=SimpleNamespace(manifest=lambda: {'match': key, 'spells': [1, 2]}))
        monkeypatch.setattr(inference_runner, 'load_match', match)
        monkeypatch.setattr(generate, 'build_sportec_possessions',
                            lambda match, key, *a, **k: iter((state(key, event), match.possession_report) for event in [1, 2]))
        result = generate.run_runtime_sportec(args)
        assert [tx.manifest for tx, _ in committed] == [{'match': key, 'spells': [1, 2]} for key in ['m1', 'm2']]
    else:
        monkeypatch.setattr(generate, 'discover_benchmark_modifications', lambda *a, **k: ([1, 2, 3], {}))
        monkeypatch.setattr(generate, 'load_benchmark_modification_data',
                            lambda *a: {'game_state_1': {}, 'game_state_2': {}, 'higher_state_id': 1})
        def build(raw, modification, index, *a, **kw):
            value = state(f'b{modification}_{index}', 1)
            value.graph_features_0, value.labels = value.graph_features_0[:1], value.labels[:1]
            return (value,)
        monkeypatch.setattr(generate, 'build_benchmark_state', build)
        result = generate.run_runtime_benchmark(args)
    assert not result['skipped']
    assert all(compute is calls[0][1] for _, compute in calls)
    assert calls[0][1]._closed
    if dataset != 'benchmark':
        assert [(tx.key, position) for tx, position in committed] == [('m1', 2), ('m2', 4)]
        assert ended == ['m1', 'm2']
        assert [generate.runtime_items_row_count(data) for data, _ in calls] == [3, 3, 3, 3]
    else:
        assert [len(data) for data, _ in calls] == [2, 2, 2]


def test_single_worker_never_creates_pool_and_empty_inputs_are_lazy(tmp_path, controlled_pool):
    with session(workers=1) as compute:
        stats = physics.prewarm_physical_xpass_runtime_cache(items(), compute_session=compute,
                                                           **kwargs(tmp_path, workers=1))
        assert stats['cache_written'] == 5
    assert not controlled_pool
    with session() as compute:
        args = generate.parse_args(['--pc-xpass', '--num-workers', '2'])
        assert generate.prewarm_runtime_items([], cache_dir=tmp_path, args=args, compute_session=compute) is None
        assert not compute.windows
    assert not controlled_pool
