import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.match import Match
from datatools.reception_policy import (KEY, RETIRED, LEGACY, apply_policy, apply_run_policy,
                                        retirement_records)
from datatools.reception_refresh import patch_labels, run_retirement
from datatools.carry_policy import DEFINITION, CarryPolicy
from datatools.ball_carries import derive_carry_segments
from datatools.sportec_possessions import reconstruct_sportec_states
from test_carry_policy import fixture, add


def test_policy_preserves_identity_outcome_launch_and_provider_receptions():
    events, _, _ = fixture()
    events['endpoint_source'] = ['elastic', 'reception']
    events.at[1, 'receive_frame_id'] = 100
    source = events.copy()
    result = apply_policy(events, RETIRED)
    assert pd.isna(result.at[0, 'receive_frame_id'])
    assert result.at[0, 'retired_receive_frame_id'] == 20
    for col in ['frame_id', 'object_id', 'receiver_id', 'success']:
        pd.testing.assert_series_equal(source[col], result[col])
    assert result.at[1, 'receive_frame_id'] == 100
    pd.testing.assert_frame_equal(result, apply_policy(result, RETIRED))
    pd.testing.assert_frame_equal(source, apply_policy(source, LEGACY))


def test_pass_filter_keeps_missing_reception_and_source_success():
    events, _, _ = fixture()
    events['endpoint_source'] = 'elastic'
    events['source_event_success'] = [True, False]
    events['receiver_id'] = ['home_2', None]
    m = Match.__new__(Match)
    m.events = apply_policy(events, RETIRED)
    m.reception_policy = RETIRED
    m.next_action_conditions_enabled = False
    p = m.filter_passes()
    assert len(p) == 2
    assert p.success.tolist() == [True, False]
    assert p.receive_frame_id.isna().all()


def test_retirement_removes_validated_carry_but_keeps_later_claim():
    events, controls, tracking = fixture()
    events['endpoint_source'] = 'elastic'
    before, _ = derive_carry_segments(controls, events, tracking)
    assert len(before) == 2
    controls = add(controls, 'claim', 30, claim_type='BallClaimed')
    after, audit = derive_carry_segments(controls, apply_policy(events, RETIRED), tracking)
    assert not audit.start_kind.eq('receipt').any()
    assert after.frame_id.min() == 30


def test_post_action_reception_removed_but_restart_preserved():
    events, _, _ = fixture()
    events['endpoint_source'] = 'elastic'
    events['next_type'] = 'pass'
    events['next_player_id'] = 'home_2'
    m = SimpleNamespace(events=events, reception_policy=RETIRED)
    result = Match.label_post_actions(m, events)
    assert pd.isna(result.at[0, 'end_frame_id'])
    events.at[0, 'next_type'] = 'throw_in'
    result = Match.label_post_actions(m, events)
    assert result.at[0, 'end_frame_id'] == 83


def test_runtime_pinned_records_prevent_reintroduction(tmp_path):
    events, controls, tracking = fixture()
    events['endpoint_source'] = 'elastic'
    (tmp_path / 'metadata.json').write_text(json.dumps({KEY: RETIRED, 'carry_variant':
        dict(definition=DEFINITION, validation_parameters=asdict(CarryPolicy()))}))
    (tmp_path / 'reception_retirement').mkdir()
    (tmp_path / 'reception_retirement/test.json').write_text(json.dumps(dict(receptions=retirement_records(events))))
    # Even a later source provenance change cannot restore the pinned reception.
    events.endpoint_source = 'reception'
    filtered = apply_run_policy(events, tmp_path, 'test')
    assert filtered.receive_frame_id.isna().all()
    m = SimpleNamespace(events=events, actions=events, tracking=tracking, fps=25, lineup=pd.DataFrame())
    plan = reconstruct_sportec_states(m, 'test', tmp_path, control_events=controls)
    assert all(a['post_state_id'] is None for a in plan.actions)
    assert all(a['pre_state_id'] in plan.observations for a in plan.actions)
    assert not any(s['start_kind'] == 'receipt' for s in plan.spells)
    assert plan.provenance[KEY] == RETIRED


def test_label_refresh_preserves_predictions_and_invalidates_original():
    labels = torch.zeros((3, len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.tensor([4, 5, 4])
    labels[:, config.LABEL_INDEX['intent_index']] = 2
    labels[:, config.LABEL_INDEX['pass_high']] = 1
    labels[:, config.LABEL_INDEX['pass_max_ball_z']] = 3
    result, _, _ = patch_labels(labels, {4}, {4}, original=True)
    assert result[:, config.LABEL_INDEX['intent_index']].tolist() == [-1, 2, -1]
    assert torch.isnan(result[[0, 2], config.LABEL_INDEX['pass_high']]).all()
    model, _, _ = patch_labels(labels, {4}, {4}, original=False)
    assert model[:, config.LABEL_INDEX['intent_index']].tolist() == [2, 2, 2]


def test_carry_source_requires_combination_and_corrected_noop(tmp_path, monkeypatch):
    import datatools.reception_refresh as rf
    source = tmp_path / 'source'
    metadata = dict(status='completed', carry_variant=dict(available=True))
    monkeypatch.setattr(rf, 'resolve_source', lambda _: (source, metadata))
    args = SimpleNamespace(extend_feature_run_id='source', replace_carries=False)
    with pytest.raises(ValueError, match='requires --replace-carries'):
        run_retirement(args)
    metadata[KEY] = RETIRED
    run_retirement(args)


def test_missing_original_target_is_skipped_before_dataset_indexing(tmp_path):
    from dataset import ActionDataset
    from torch_geometric.data import Data
    features, label_dir = tmp_path / 'graphs', tmp_path / 'labels'
    features.mkdir()
    label_dir.mkdir()
    x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM))
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros((2, 2)))
    labels = torch.zeros((2, len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.arange(2)
    labels[:, config.LABEL_INDEX['is_pass']] = 1
    labels[:, config.LABEL_INDEX['duration']] = 1
    labels[:, config.LABEL_INDEX['intent_index']] = torch.tensor([-1, 1])
    torch.save([graph, graph.clone()], features / 'm.pt')
    torch.save(labels, label_dir / 'm.pt')
    dataset = ActionDataset(['m'], feature_dir=features, label_dir=label_dir,
                            task='action_intent', min_pass_dur=0, edge_in_dim=2)
    assert dataset.labels[:, 0].tolist() == [1]


def test_refresh_resume_policy_and_latest_publication(tmp_path, monkeypatch):
    import datatools.reception_refresh as rf
    from scripts import replace_carries as rc
    source, output = tmp_path / 'source', tmp_path / 'derived'
    source.mkdir()
    original = dict(status='completed', intended_receiver_modes=['original'], return_types=['next_10'])
    (source / 'metadata.json').write_text(json.dumps(original))
    (source / 'canonical.pt').write_bytes(b'source')
    monkeypatch.setattr(rf, 'resolve_source', lambda _: (source, original))
    monkeypatch.setattr(rc, 'match_ids', lambda *a: ['one', 'two'])
    monkeypatch.setattr(rf, 'get_feature_run_root', lambda _: output)
    def write(root, data):
        root.mkdir(exist_ok=True)
        (root / 'metadata.json').write_text(json.dumps(data))
    monkeypatch.setattr(rf, 'write_run_metadata', write)
    latest = []
    monkeypatch.setattr(rf, 'write_latest_run', lambda *a: latest.append(a))
    calls, fail = [], [True]
    def refresh(task):
        match_id = task[3]
        calls.append(match_id)
        if match_id == 'two' and fail[0]:
            raise RuntimeError('interrupted')
        path = output / (match_id + '.pt')
        path.write_bytes(b'done')
        report = dict(policy=RETIRED, counts={'receptions': 1}, artifacts={path.name: 4})
        dest = output / 'reception_retirement/completed'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / (match_id + '.json')).write_text(json.dumps(report))
        return dict(report, match_id=match_id)
    monkeypatch.setattr(rf, 'refresh_match', refresh)
    args = SimpleNamespace(extend_feature_run_id='source', run_id='derived', replace_carries=False,
                           num_workers='1', worker_thread_limit=1, artifact_copy_mode='hardlink')
    with pytest.raises(RuntimeError, match='interrupted'):
        run_retirement(args)
    assert not latest
    assert json.loads((output / 'metadata.json').read_text())['status'] == 'failed'
    args.carry_acquisition_distance = 2.5
    with pytest.raises(ValueError, match='identical'):
        run_retirement(args)
    args.carry_acquisition_distance = 3
    fail[0] = False
    run_retirement(args)
    assert calls == ['one', 'two', 'two']
    assert len(latest) == 1
    assert (source / 'canonical.pt').read_bytes() == b'source'


def test_cli_allows_combination_but_rejects_unrelated_flags(monkeypatch):
    import sys
    from scripts.generate_relevant_features import parse_args
    base = ['generate_relevant_features.py', '--retire-elastic', '--replace-carries',
            '--extend-feature-run-id', 'source']
    monkeypatch.setattr(sys, 'argv', base)
    assert parse_args().retire_elastic
    monkeypatch.setattr(sys, 'argv', base + ['--in-place'])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, 'argv', base + ['--refresh-trajectory'])
    with pytest.raises(SystemExit):
        parse_args()


def test_unknown_outcome_preserves_launch_without_manufacturing_failure():
    events, _, _ = fixture()
    events['endpoint_source'] = 'elastic'
    events['source_event_success'] = None
    events['receiver_id'] = None
    m = Match.__new__(Match)
    m.events = apply_policy(events, RETIRED)
    m.reception_policy = RETIRED
    m.next_action_conditions_enabled = False
    result = m.filter_passes()
    assert len(result) == 2
    assert not result.success_known.any()
    assert m.filter_passes(failure_only=True).empty


def test_real_refresh_preserves_temporal_model_targets_and_combination_order(tmp_path, monkeypatch):
    import datatools.reception_refresh as rf
    from scripts import replace_carries as rc
    source, output, data = tmp_path / 'source', tmp_path / 'output', tmp_path / 'data'
    events, _, _ = fixture()
    events['endpoint_source'] = ['elastic', 'reception']
    events['next_type'] = 'pass'
    events['next_player_id'] = 'home_2'
    events.at[0, 'success'] = False
    raw = data / 'event_synced/m.csv'
    raw.parent.mkdir(parents=True)
    events.to_csv(raw, index=False)
    actions = events.assign(action_type='pass', intent_id='home_2')
    resolved = source / 'resolved_actions_original/m.parquet'
    resolved.parent.mkdir(parents=True)
    actions.to_parquet(resolved)
    labels = torch.zeros((2, len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.arange(2)
    labels[:, config.LABEL_INDEX['intent_index']] = 1
    for folder in ['action_labels_next_10', 'action_labels_intent_train_next_10_model',
                   'action_labels_intent_train_next_10_angle_only']:
        path = source / folder / 'm.pt'
        path.parent.mkdir()
        torch.save(labels, path)
    post = source / 'post_action_graphs/m.pt'
    post.parent.mkdir()
    torch.save([1, 2], post)
    monkeypatch.setattr(rf, 'DATA_ROOT', data)
    def rebuild(task):
        assert task[2][KEY] == RETIRED
        assert Path(task[2]['_canonical_root']) == output
        changed = pd.read_parquet(output / resolved.relative_to(source))
        assert pd.isna(changed.at[0, 'receive_frame_id'])
        assert pd.isna(changed.at[0, 'intent_id'])
        assert torch.load(output / 'action_labels_next_10/m.pt', weights_only=False)[0, 5] == -1
        assert torch.load(output / 'post_action_graphs/m.pt', weights_only=False)[0] is None
        return dict(segments=0, artifacts={})
    monkeypatch.setattr(rc, 'rebuild_match', rebuild)
    rf.refresh_match((str(source), str(output), dict(intended_receiver_modes=['original']),
                      'm', True, asdict(CarryPolicy()), 1))
    for mode in ['model', 'angle_only']:
        result = torch.load(output / f'action_labels_intent_train_next_10_{mode}/m.pt', weights_only=False)
        assert result[:, 5].tolist() == [1, 1]
    assert torch.load(post, weights_only=False) == [1, 2]


def test_unknown_policy_and_missing_manifest_fail_explicitly(tmp_path):
    events, _, _ = fixture()
    with pytest.raises(ValueError, match='Unsupported'):
        apply_policy(events, 'unknown')
    (tmp_path / 'metadata.json').write_text(json.dumps({KEY: RETIRED}))
    with pytest.raises(FileNotFoundError, match='pinned'):
        apply_run_policy(events, tmp_path, 'm')
