"""Exercise the real report writer and real worker-process error transport."""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datatools import graph_feature as gf
from datatools.graph_validation import GraphValidationError
from datatools.input_diagnostics import WorkerFailure, failure_report, failure_summary, read_json, read_parquet

INVALID_JSON = '{\n  "run_id": "bad\x00value"\n}'


def test_json_reader_retains_type_position_and_input_path(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text(INVALID_JSON, encoding='utf-8')
    with pytest.raises(json.JSONDecodeError) as caught:
        read_json(path)
    error = caught.value
    assert error.input_path == str(path.absolute())
    assert error.input_format == 'json'
    report = failure_report(error, 'match')
    assert report['json_error'] == dict(line=error.lineno, column=error.colno, position=error.pos)
    assert str(path) in report['traceback']
    assert 'JSONDecodeError' in report['traceback']
    assert '\n' not in failure_summary(report)


def test_parquet_reader_identifies_malformed_embedded_json(tmp_path):
    path = tmp_path / 'tracking.parquet'
    table = pa.table({'value': [1]}).replace_schema_metadata({b'PANDAS_ATTRS': INVALID_JSON.encode()})
    pq.write_table(table, path)
    with pytest.raises(json.JSONDecodeError) as caught:
        read_parquet(path)
    assert caught.value.input_path == str(path.absolute())
    assert caught.value.input_format == 'parquet'
    assert failure_report(caught.value, 'match')['json_error']['line'] == 2


def test_worker_failure_survives_pickle_with_original_traceback(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text(INVALID_JSON, encoding='utf-8')
    try:
        read_json(path)
    except json.JSONDecodeError as error:
        original = failure_report(error, 'match')
    restored = pickle.loads(pickle.dumps(WorkerFailure(original)))
    assert failure_report(restored, 'match') == original
    assert str(restored).startswith('JSONDecodeError:')


def test_chained_failure_keeps_input_context(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_text(INVALID_JSON, encoding='utf-8')
    try:
        try:
            read_json(path)
        except json.JSONDecodeError as error:
            raise RuntimeError('input preparation failed') from error
    except RuntimeError as error:
        report = failure_report(error, 'match')
    assert report['exception_type'] == 'RuntimeError'
    assert report['input_path'] == str(path.absolute())
    assert report['json_error']['line'] == 2
    assert 'JSONDecodeError' in report['traceback'] and 'RuntimeError' in report['traceback']


@pytest.mark.parametrize('input_format', ['json', 'parquet'])
@pytest.mark.parametrize('num_workers', [1, 2])
def test_entrypoint_records_input_path_and_worker_traceback(tmp_path, monkeypatch, capsys, input_format, num_workers):
    monkeypatch.chdir(tmp_path)
    event_dir = tmp_path / 'data' / 'event_synced'
    event_dir.mkdir(parents=True)
    (event_dir / 'match.csv').write_text('utc_timestamp\n', encoding='utf-8')
    feature_root = tmp_path / 'features'
    feature_root.mkdir()
    if input_format == 'json':
        input_path = feature_root / 'metadata.json'
        input_path.write_text(INVALID_JSON, encoding='utf-8')
    else:
        (feature_root / 'metadata.json').write_text('{}', encoding='utf-8')
        tracking_dir = tmp_path / 'data' / 'tracking_processed'
        tracking_dir.mkdir()
        input_path = tracking_dir / 'match.parquet'
        table = pa.table({'value': [1]}).replace_schema_metadata({b'PANDAS_ATTRS': INVALID_JSON.encode()})
        pq.write_table(table, input_path)
    with pytest.raises(json.JSONDecodeError) as expected:
        json.loads(INVALID_JSON)
    monkeypatch.setattr(sys, 'argv', ['graph_feature.py', '--run-id', 'diagnostics-test',
        '--labels-only', '--train-count', '1', '--return_type', 'next_10',
        '--only-intended-receiver-mode', 'original', '--receiver-extension-attempt', 'diagnostic-attempt',
        '--num-workers', str(num_workers)])
    monkeypatch.setattr(gf, 'get_feature_run_root', lambda _: feature_root)
    monkeypatch.setattr(gf, 'load_base_splits', lambda **_: (np.array(['match']), np.array([], dtype=str)))
    original_read = pd.read_parquet

    def read_lineups_or_input(path, *args, **kwargs):
        if str(path).replace('\\', '/') == 'data/lineup/line_up.parquet':
            return pd.DataFrame({'stats_perform_match_id': ['match'], 'game_date': ['2026-01-01']})
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_parquet', read_lineups_or_input)
    with pytest.raises(RuntimeError, match='No usable matches'):
        gf.main()
    report = json.loads((feature_root / 'generation_reports' / 'train_base.json').read_text())
    assert report['successful_matches'] == 0
    assert report['expected_matches'] == ['match']
    assert len(report['skipped_matches']) == 1
    failure = report['skipped_matches'][0]
    assert failure['match_id'] == 'match'
    assert failure['exception_type'] == 'JSONDecodeError'
    assert failure['error'].startswith('JSONDecodeError: Invalid control character')
    assert failure['input_path'] == str(input_path.absolute())
    assert failure['input_format'] == input_format
    assert failure['json_error'] == dict(line=expected.value.lineno, column=expected.value.colno,
                                       position=expected.value.pos)
    assert '_process_match_generation_task' in failure['traceback']
    assert 'JSONDecodeError' in failure['traceback']
    assert str(input_path.absolute()) in failure['traceback']
    output = capsys.readouterr()
    console = output.out + output.err
    assert 'SKIP match: JSONDecodeError:' in console
    assert str(input_path.absolute()) in console
    assert 'Traceback (most recent call last)' not in console


def test_graph_validation_failure_remains_fatal(monkeypatch):
    def fail(_):
        raise GraphValidationError('invalid connectivity')
    monkeypatch.setattr(gf, 'process_match_generation_task', fail)
    with pytest.raises(GraphValidationError, match='invalid connectivity'):
        gf._process_match_with_diagnostics(object())
