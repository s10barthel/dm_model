"""Partial blend inputs stay unavailable without hiding rejected caches."""
from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

import inference
from datatools.epv import compute_epv_values
from pc_xpass_inference_policy import (PCXPassBlendingError, inference_attempt, start_inference_attempt,
    finalize_pc_xpass_attempt, pc_xpass_variant, pc_xpass_coverage_unit)
from test_pc_xpass_inference_policy import fixture


@pytest.mark.parametrize('failure', ['row', 'column', 'nan', 'inf', 'nearest', 'ball_z'])
def test_partial_values_preserve_other_predictions(fixture, failure):
    subject, model, rows, folder, calls = fixture
    if failure == 'row':
        rows = rows.iloc[:1]
    elif failure == 'column':
        rows = rows.drop(columns='home_2__top10_xpass')
    elif failure in {'nan', 'inf'}:
        rows.loc[1, 'home_2__top10_xpass'] = np.nan if failure == 'nan' else np.inf
    elif failure == 'nearest':
        model.args['xpass_weight'] = 'v2'
        rows.loc[1, 'home_2__distance_to_nearest_opponent'] = np.nan
    else:
        model.args['ball_z_limit'] = 1.
        rows.loc[1, 'ball_z'] = np.inf
    rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
    inference.preflight_pc_xpass(subject, model)
    assert not calls
    final, learned = inference.inference_gnn(subject, model, device='cpu', return_model_pass_success=True)
    assert list(final.index) == [10, 11]
    assert np.isnan(final.at[11, 'home_2'])
    assert learned.at[11, 'home_2'] == pytest.approx(.9)
    assert np.isfinite(final.at[11, 'home_1'])
    coverage = final.attrs['pc_xpass_blending_coverage']
    assert coverage['eligible_count'] == 2
    assert coverage['missing_count'] == (2 if failure == 'column' else 1)
    assert len(calls) == 1


@pytest.mark.parametrize('missing', ['height_nan', 'height_row', 'intent_inf', 'intent_row'])
def test_partial_auxiliary_values(fixture, missing):
    from models.pass_height import definition
    subject, model, _, _, calls = fixture
    model.args.update(xpass_weight='v5', xpass_pass_height_definition=definition(2., 'checkpoint'))
    height = pd.DataFrame({'home_2': [.5, .5]}, index=[10, 11])
    intent = height.copy()
    table = height if missing.startswith('height') else intent
    if missing.endswith('row'):
        table.drop(index=11, inplace=True)
    else:
        table.at[11, 'home_2'] = np.nan if missing.endswith('nan') else np.inf
    final, _ = inference.inference_gnn(subject, model, device='cpu', pass_height_probs=height, pass_intent_probs=intent)
    assert np.isfinite(final.at[10, 'home_2']) and np.isnan(final.at[11, 'home_2'])
    assert final.attrs['pc_xpass_blending_coverage']['missing_count'] == 1
    assert calls == [1]


@pytest.mark.parametrize('recover', [False, True])
def test_attempt_finalizes_after_all_units(fixture, tmp_path, recover):
    subject, model, rows, folder, _ = fixture
    run = tmp_path / 'attempt'
    latest = tmp_path / 'latest.json'
    latest.write_text('previous')
    @inference_attempt
    def execute():
        start_inference_attempt(run, Namespace(pc_xpass=True, use_physical_xpass=True))
        rows['home_2__top10_xpass'] = np.nan
        rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
        final, _ = inference.inference_gnn(subject, model, device='cpu')
        final.to_csv(run / 'first.csv')
        if recover:
            rows['home_2__top10_xpass'] = .2
            rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
            inference.inference_gnn(subject, model, device='cpu')
        finalize_pc_xpass_attempt()
        latest.write_text('attempt')
    if recover:
        execute()
    else:
        with pytest.raises(PCXPassBlendingError, match='coverage_finalization'):
            execute()
    metadata = json.loads((run / 'metadata.json').read_text())
    assert metadata['status'] == ('completed' if recover else 'failed')
    assert latest.read_text() == ('attempt' if recover else 'previous')
    assert (run / 'first.csv').exists()
    assert metadata['pc_xpass_blending_policy'] == 'cache_required'
    coverage, = metadata['pc_xpass_blending_coverage'].values()
    assert coverage['missing_count'] == 2
    assert coverage['usable_count'] == (2 if recover else 0)


def test_variants_cannot_mask_each_others_zero_coverage(fixture, tmp_path):
    subject, model, rows, folder, _ = fixture
    @inference_attempt
    def execute():
        start_inference_attempt(tmp_path / 'run', Namespace(pc_xpass=True, use_physical_xpass=True))
        with pc_xpass_variant('valid'):
            inference.inference_gnn(subject, model, device='cpu')
        rows['home_2__top10_xpass'] = np.nan
        rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
        with pc_xpass_variant('missing'):
            inference.inference_gnn(subject, model, device='cpu')
        finalize_pc_xpass_attempt()
    with pytest.raises(PCXPassBlendingError, match='variant=missing'):
        execute()


def test_all_offside_is_not_zero_coverage_failure(fixture, tmp_path):
    subject, model, rows, folder, _ = fixture
    for graph in subject.graph_features_0:
        graph.x[1, -1] = 1
    rows['home_2__top10_xpass'] = np.nan
    rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
    @inference_attempt
    def execute():
        start_inference_attempt(tmp_path / 'run', Namespace(pc_xpass=True, use_physical_xpass=True))
        final, _ = inference.inference_gnn(subject, model, device='cpu')
        assert final.home_2.eq(0).all()
        assert final.attrs['pc_xpass_blending_coverage']['eligible_count'] == 0
        finalize_pc_xpass_attempt()
    execute()


def test_preflight_does_not_prepare_graphs_or_scan_candidates(fixture, monkeypatch):
    subject, model, _, _, _ = fixture
    monkeypatch.setattr(inference, 'resolve_match_graphs', lambda *a: pytest.fail('graph preparation'))
    monkeypatch.setattr(inference, '_pc_pass_candidate_mask', lambda *a: pytest.fail('candidate scan'))
    monkeypatch.setattr(inference, 'filter_missing_physical_xpass_rows_for_inference', lambda *a, **kw: pytest.fail('state scan'))
    inference.preflight_pc_xpass(subject, model)


def test_checkpoint_internal_physical_inputs_remain_required(fixture):
    subject, model, rows, folder, calls = fixture
    model.args.update(use_physical_xpass=True, model_variant='gat_phys_logit_offset')
    rows.loc[1, 'home_2__top10_xpass'] = np.nan
    rows.to_parquet(folder / 'matches/match_1.parquet', index=False)
    with pytest.raises(PCXPassBlendingError):
        inference.inference_gnn(subject, model, device='cpu')
    assert not calls


def test_cached_predictions_replay_coverage_for_each_variant(fixture, tmp_path):
    from scripts.visualization_comparison import cached_prediction
    subject, model, _, _, calls = fixture
    @inference_attempt
    def execute():
        start_inference_attempt(tmp_path / 'run', Namespace(pc_xpass=True, use_physical_xpass=True))
        memo = {}
        for key in ('one', 'two'):
            with pc_xpass_variant(key):
                cached_prediction(memo, inference.inference_gnn, subject, model, device='cpu')
        report = finalize_pc_xpass_attempt()['pc_xpass_blending_coverage']
        assert set(report) == {'one', 'two'}
        assert all(item['usable_count'] == 2 for item in report.values())
    execute()
    assert calls == [1]


def test_failed_collector_does_not_count_predictions(fixture, tmp_path):
    subject, model, _, _, _ = fixture
    @pc_xpass_coverage_unit
    def collect():
        inference.inference_gnn(subject, model, device='cpu')
        raise ValueError('unrelated collector failure')
    @inference_attempt
    def execute():
        start_inference_attempt(tmp_path / 'run', Namespace(pc_xpass=True, use_physical_xpass=True))
        with pytest.raises(ValueError):
            collect()
        report = finalize_pc_xpass_attempt()
        assert all(value['eligible_count'] == 0 for value in report['pc_xpass_blending_coverage'].values())
    execute()


@pytest.mark.parametrize('intent', [0., .2])
def test_epv_does_not_present_incomplete_positive_intent_sum(intent):
    table = pd.DataFrame([[.4, .6]], columns=['home_2', 'home_3'])
    success = table.copy()
    success.iloc[0, 1] = np.nan
    probabilities = pd.DataFrame([[.8, intent]], columns=table.columns)
    result = compute_epv_values(probabilities, success, table, table * 0, table * 0, table * 0)
    assert np.isnan(result.iloc[0]) if intent else np.isfinite(result.iloc[0])
