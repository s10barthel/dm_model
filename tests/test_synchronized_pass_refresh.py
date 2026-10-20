"""Regression coverage for synchronized endpoints and the composed refresh."""
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from datatools import config
from datatools.pass_trajectory import PassEndpointResolver
from datatools import trajectory_refresh as refresh
from datatools import feature_refresh as combined
from test_pass_trajectory import fixture
from test_trajectory_speed_refresh import feature_runs


def test_synchronized_order_preserves_intermediate_events():
    match = fixture(contact=3)
    events = match.official_pass_events
    events.loc[3] = ['earlier', pd.Timestamp('2023-08-18T18:00:20Z'), 'TacklingGame', 110]
    end = PassEndpointResolver(match.tracking, events=events).resolve(match.actions.loc[7])
    assert (end.observed, end.event_id) == (110, 'earlier')


def test_failed_pass_linked_reception_can_precede_next_event():
    match = fixture(contact=3)
    events = match.official_pass_events
    events['reception_frame_id'] = [108, None, None]
    end = PassEndpointResolver(match.tracking, events=events).resolve(match.actions.loc[7])
    assert (end.observed, end.source) == (108, 'official_reception')


def test_interpolated_events_and_unknown_reception_are_unavailable():
    match = fixture()
    events = match.official_pass_events.assign(frame_source='event_time_interpolation')
    end = PassEndpointResolver(match.tracking, events=events).resolve(match.actions.loc[7])
    assert end.observed is None and np.isnan(end.duration)


def test_official_stoppage_at_first_dead_frame_is_accepted():
    match = fixture(contact=1.6, tag='Offside')
    match.tracking.loc[140:, ['ball_state', 'episode_id']] = ['dead', 0]
    end = PassEndpointResolver(match.tracking, events=match.official_pass_events).resolve(match.actions.loc[7])
    assert (end.observed, end.source) == (140, 'official_stoppage')


def test_restart_is_boundary_not_replacement_contact():
    match = fixture(contact=2, tag='FreeKick')
    assert PassEndpointResolver(match.tracking, events=match.official_pass_events).resolve(match.actions.loc[7]).observed is None
    match.actions.loc[7, ['receiver_id', 'next_type']] = ['out', 'throw_in']
    match.tracking.loc[140:, ['ball_state', 'episode_id']] = ['dead', 0]
    assert PassEndpointResolver(match.tracking, events=match.official_pass_events).resolve(match.actions.loc[7]).observed == 139


def test_duration_guard_disabled_default_equality_and_diagnostics():
    match = fixture(contact=2)
    assert PassEndpointResolver(match.tracking, events=match.official_pass_events).resolve(match.actions.loc[7]).duration == 2
    resolver = PassEndpointResolver(match.tracking, events=match.official_pass_events, max_pass_duration=2)
    assert resolver.resolve(match.actions.loc[7]).observed == 150
    resolver = PassEndpointResolver(match.tracking, events=match.official_pass_events, max_pass_duration=1)
    end = resolver.resolve(match.actions.loc[7])
    assert end.observed is None and end.reason == 'duration_limit_exceeded'
    assert (end.candidate, end.candidate_duration) == (150, 2)
    assert np.isnan(end.duration)


def test_safeguard_changes_invalidate_refresh_noop_and_record_skips(feature_runs):
    runs, base, args, latest = feature_runs
    refresh.run_trajectory_refresh(args)
    args.extend_feature_run_id, args.run_id, args.max_pass_duration = 'derived', 'limited', .5
    refresh.run_trajectory_refresh(args)
    metadata = json.loads((runs / 'limited/metadata.json').read_text())
    assert metadata['max_pass_duration'] == .5
    assert metadata['skipped_action_count'] == 2
    assert metadata['skipped_actions']['a'][0]['reason'] == 'duration_limit_exceeded'
    assert torch.load(runs / 'limited/action_graphs_success_intent/a.pt', weights_only=False) == [None, None]
    assert json.loads((base / 'metadata.json').read_text()).get('max_pass_duration') is None


def test_receiver_refresh_rebuilds_synthetic_membership(tmp_path):
    tracking = pd.DataFrame(index=np.arange(100, 151))
    tracking['ball_x'] = (tracking.index - 100) * .8
    tracking['ball_y'] = (tracking.index - 100) * .2
    tracking['ball_z'], tracking['period_id'] = 0., 1
    for player, xy in {'home_1': (0, 0), 'home_2': (20, 0), 'home_3': (25, 5),
                       'away_1': (2, 0), 'away_2': (40, 0)}.items():
        tracking[player + '_x'], tracking[player + '_y'] = xy
    actions = pd.DataFrame([dict(action_id=12, original_event_id='launch', action_type='pass', spadl_type='pass',
        frame_id=100, receive_frame_id=145, object_id='home_1', receiver_id='away_1', intent_id='home_2',
        period_id=1, success=False, start_x=0., start_y=0., end_x=99., end_y=99.)], index=[7])
    events = pd.DataFrame(dict(event_id=['launch', 'contact'], frame_id=[100, 150], raw_tag=['Play', 'OtherBallAction']))
    match = SimpleNamespace(actions=actions.copy(), tracking=tracking, official_pass_events=events,
                            fps=25, include_goals=False, max_pass_duration=None)
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    for key, value in dict(is_pass=1, is_real=1, n_players=5, intent_index=1).items():
        labels[0, config.LABEL_INDEX[key]] = value
    labels[0, 0] = 7
    graph = Data(x=torch.zeros((5, 26)), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros((2, 2)))
    for folder in ['resolved_actions_original', 'action_labels_next_10', 'action_graphs', 'augmented_graphs', 'augmented_labels']:
        (tmp_path / folder).mkdir()
    actions.to_parquet(tmp_path / 'resolved_actions_original/m.parquet')
    torch.save(labels, tmp_path / 'action_labels_next_10/m.pt')
    torch.save([graph], tmp_path / 'action_graphs/m.pt')
    from datatools.graph_feature import augment_blocked_actions
    match.labels, match.graph_features_0 = labels, [graph]
    old_graphs, old_labels = augment_blocked_actions(match, verbose=False)
    torch.save(old_graphs, tmp_path / 'augmented_graphs/m.pt')
    torch.save(old_labels, tmp_path / 'augmented_labels/m.pt')
    with patch.object(refresh, 'load_refresh_match', return_value=match):
        report = refresh.refresh_directory_match(tmp_path, {'return_types': ['next_10']}, 'm')
    result = pd.read_parquet(tmp_path / 'resolved_actions_original/m.parquet')
    new_labels = torch.load(tmp_path / 'augmented_labels/m.pt', weights_only=False)
    assert result.at[7, 'intent_id'] == 'home_3'
    assert result.at[7, 'observed_end_frame_id'] == 150
    assert old_labels[:, config.LABEL_INDEX['intent_index']].tolist() == [1, 2]
    assert new_labels[:, config.LABEL_INDEX['intent_index']].tolist() == [2, 1]
    assert report['heuristic_changes']['resolved_actions_original'] == 1


def test_combined_worker_orders_retirement_refresh_and_carries(tmp_path, monkeypatch):
    import datatools.reception_refresh as retirement
    import scripts.replace_carries as carries
    order = []
    monkeypatch.setattr(retirement, 'refresh_match', lambda *a: order.append('retire'))
    monkeypatch.setattr(refresh, 'refresh_directory_match', lambda *a: (order.append('refresh') or dict(match_id='m', skipped_actions=[])))
    monkeypatch.setattr(carries, 'rebuild_match', lambda *a: (order.append('carries') or dict(segments=3)))
    result = combined._refresh_match((str(tmp_path), str(tmp_path), {}, 'm',
        dict(retire_elastic=True, replace_carries=True, carry_parameters={}), 1))
    assert order == ['retire', 'refresh', 'carries']
    assert result['carry_segments'] == 3


def test_combined_failure_resume_and_source_preservation(tmp_path, monkeypatch):
    import datatools.reception_refresh as retirement
    import scripts.replace_carries as carries
    source, output = tmp_path / 'source', tmp_path / 'derived'
    source.mkdir()
    original = dict(status='completed', intended_receiver_modes=['original'], return_types=['next_10'])
    (source / 'metadata.json').write_text(json.dumps(original))
    (source / 'canonical.pt').write_bytes(b'original')
    monkeypatch.setattr(retirement, 'resolve_source', lambda _: (source, original))
    monkeypatch.setattr(carries, 'match_ids', lambda *a: ['a', 'b'])
    monkeypatch.setattr(combined, 'get_feature_run_root', lambda _: output)
    latest, calls = [], []
    monkeypatch.setattr(combined, 'write_latest_run', lambda *a: latest.append(a))
    fail = [True]
    def worker(task):
        mid = task[3]; calls.append(mid)
        if mid == 'b' and fail[0]:
            raise RuntimeError('interrupted')
        report = dict(match_id=mid, skipped_actions=[], artifacts={'canonical.pt': 8})
        combined.atomic_write_json(output / 'feature_refresh' / (mid + '.json'), report)
        return report
    monkeypatch.setattr(combined, '_refresh_match', worker)
    args = SimpleNamespace(extend_feature_run_id='source', run_id='derived', in_place=False,
        replace_carries=True, retire_elastic=True, artifact_copy_mode='hardlink', num_workers='1', worker_thread_limit=1)
    with pytest.raises(RuntimeError):
        combined.run_combined_refresh(args)
    assert json.loads((output / 'metadata.json').read_text())['status'] == 'failed' and not latest
    fail[0] = False
    combined.run_combined_refresh(args)
    assert calls == ['a', 'b', 'b'] and len(latest) == 1
    assert (source / 'canonical.pt').read_bytes() == b'original'
    assert json.loads((output / 'metadata.json').read_text())['status'] == 'completed'

def test_optional_guard_cli_defaults_and_invalid_values(monkeypatch):
    import sys
    from scripts.generate_relevant_features import parse_args
    base = ['generate', '--refresh-trajectory', '--retire-elastic', '--replace-carries', '--extend-feature-run-id', 'source']
    monkeypatch.setattr(sys, 'argv', base)
    assert parse_args().max_pass_duration is None
    monkeypatch.setattr(sys, 'argv', base + ['--max-pass-duration', '6'])
    assert parse_args().max_pass_duration == 6
    for value in ['0', '-1', 'nan', 'inf']:
        monkeypatch.setattr(sys, 'argv', base + ['--max-pass-duration', value])
        with pytest.raises(SystemExit):
            parse_args()


def test_arrow_uses_displayed_graph_trajectory_and_away_rotation():
    from scripts.visualize_action_components import resolve_ball_velocity_xy
    match = SimpleNamespace(actions=pd.DataFrame([dict(action_type='pass', object_id='away_1')], index=[7]))
    row = pd.Series({'away_2': .8})
    row.attrs['model_trajectory'] = [10., .6, .8, 0.]
    with patch('scripts.visualize_action_components.summarize_ball_trajectory', side_effect=AssertionError('live recomputation')):
        assert resolve_ball_velocity_xy(match, 7, 'success_intent', row) == (-6., -8.)


def test_guard_filters_cached_model_targets_and_synthetic_rows(tmp_path):
    from dataset import ActionDataset
    graphs, labels_dir = tmp_path / 'graphs', tmp_path / 'labels'
    graphs.mkdir(); labels_dir.mkdir()
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(max_pass_duration=6)))
    x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM))
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros((2, 2)))
    labels = torch.zeros((3, len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.arange(3)
    labels[:, config.LABEL_INDEX['is_pass']] = 1
    labels[:, config.LABEL_INDEX['is_real']] = 1
    labels[:, config.LABEL_INDEX['intent_index']] = 1
    labels[:, config.LABEL_INDEX['duration']] = torch.tensor([2., 8., float('nan')])
    torch.save([graph, graph.clone(), graph.clone()], graphs / 'm.pt')
    torch.save(labels, labels_dir / 'm.pt')
    data = ActionDataset(['m'], feature_dir=graphs, label_dir=labels_dir, task='action_intent', min_pass_dur=0, edge_in_dim=2)
    assert data.labels[:, 0].tolist() == [0]

def test_unknown_success_and_launch_state_skip_instead_of_aborting():
    from datatools.utils import label_intended_receivers
    match = fixture()
    match.actions['success'] = pd.Series([pd.NA], index=match.actions.index, dtype='boolean')
    result = label_intended_receivers(match.actions, match.tracking, action_type='pass', intended_receiver_mode='original')
    assert pd.isna(result.at[7, 'intent_id'])
    match.tracking.at[100, 'ball_state'] = pd.NA
    endpoint = PassEndpointResolver(match.tracking, events=match.official_pass_events).resolve(match.actions.loc[7])
    assert endpoint.observed is None and endpoint.reason == 'missing_launch_state'
