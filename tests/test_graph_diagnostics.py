import pytest
import torch
from torch_geometric.data import Batch, Data

from models.graph_diagnostics import (DiagnosticCollater, install_collation_diagnostics,
                                      validate_with_snapshot)
from models.node_selection import validate_graph_batch


def batch():
    return Batch.from_data_list([
        Data(x=torch.ones(n, 2), edge_index=torch.tensor([[0, 1], [1, 0]]),
             edge_attr=torch.ones(2, 1), evaluation_match_id=f'm{i}', evaluation_source_row=10 + i)
        for i, n in enumerate((2, 3))])


def test_valid_batch_has_no_snapshot(tmp_path):
    graph = batch()
    validate_with_snapshot(graph, torch.zeros(2, 8), torch.ones(2), directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_crossing_edge_reports_samples_and_preserves_snapshot(tmp_path):
    graph = batch()
    graph.edge_index[1, 0] = 2
    labels, weights = torch.zeros(2, 8), torch.ones(2)
    for _ in range(2):
        with pytest.raises(ValueError, match='Edge crosses graph boundaries') as caught:
            validate_with_snapshot(graph, labels, weights, directory=tmp_path, context={'batch': 421})
        assert "'edge': 0" in str(caught.value)
        assert 'm0' in str(caught.value) and 'm1' in str(caught.value)
        assert 'collated_edge_owner' in str(caught.value)
        assert 'saved to' in caught.value.__notes__[0]
    files = list((tmp_path / 'batch_failures').glob('*.pt'))
    assert len(files) == 2
    saved = torch.load(files[0], weights_only=False, map_location='cpu')
    torch.testing.assert_close(saved['graphs'].edge_index, graph.edge_index)
    torch.testing.assert_close(saved['labels'], labels)
    torch.testing.assert_close(saved['weights'], weights)
    assert saved['context']['batch'] == 421
    with pytest.raises(ValueError, match='Edge crosses graph boundaries'):
        validate_graph_batch(saved['graphs'])


@pytest.mark.parametrize('kind', ['pointer', 'assignment', 'negative', 'overflow', 'nan'])
def test_invalid_batch(kind):
    graph = batch()
    if kind == 'pointer': graph.ptr[-1] += 1
    if kind == 'assignment': graph.batch[0] = 1
    if kind == 'negative': graph.edge_index[0, 0] = -1
    if kind == 'overflow': graph.edge_index[0, 0] = graph.num_nodes
    if kind == 'nan': graph.x[0, 0] = float('nan')
    with pytest.raises(ValueError): validate_graph_batch(graph)


def test_snapshot_failure_preserves_validation_error(tmp_path, monkeypatch):
    graph = batch()
    graph.edge_index[1, 0] = 2
    def fail(*args, **kwargs): raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(ValueError, match='Edge crosses graph boundaries') as caught:
        validate_with_snapshot(graph, None, None, directory=tmp_path)
    assert 'disk full' in caught.value.__notes__[0]
    assert not list((tmp_path / 'batch_failures').iterdir())


def test_run_epoch_saves_before_forward_with_monitoring_off(tmp_path):
    from types import SimpleNamespace
    from torch_geometric.loader import DataLoader
    from models.utils import run_epoch
    from datatools.config import LABEL_COLUMNS
    samples = batch().to_data_list()
    # Invalid local endpoint becomes a cross-graph edge after collation.
    samples[0].edge_index[1, 0] = samples[0].num_nodes
    loader = DataLoader([(graph, torch.zeros(len(LABEL_COLUMNS)), torch.tensor(1.))
                         for graph in samples], batch_size=2)
    class NoForward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            pytest.fail('Invalid graph reached model forward')
    args = SimpleNamespace(task='pass_success', gnn_task='node_binary',
                           _batch_failure_dir=tmp_path, _epoch=1)
    with pytest.raises(ValueError, match='Edge crosses graph boundaries'):
        run_epoch(args, NoForward(), loader, None, device='cpu')
    saved = torch.load(next((tmp_path / 'batch_failures').glob('*.pt')), weights_only=False)
    assert saved['context'] == {'epoch': 1, 'batch': 0, 'phase': 'validation'}


def samples():
    return [(graph, torch.full((8,), float(i)), torch.tensor(float(i + 1)))
            for i, graph in enumerate(batch().to_data_list())]


def snapshot(directory):
    return torch.load(next((directory / 'batch_failures').glob('*.pt')),
                      weights_only=False, map_location='cpu')


@pytest.mark.parametrize('enabled', [False, True])
def test_collation_preserves_data_order_rng_and_partial_batches(tmp_path, enabled):
    from torch_geometric.loader import DataLoader
    rows = samples() + samples()[:1]
    loader = DataLoader(rows, batch_size=2, shuffle=True)
    original = loader.collate_fn
    install_collation_diagnostics(loader, enabled=enabled, directory=tmp_path)
    assert isinstance(loader.collate_fn, DiagnosticCollater) == enabled
    if not enabled:
        assert loader.collate_fn is original
    state = torch.random.get_rng_state()
    actual = list(loader)
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(state)
    expected = list(DataLoader(rows, batch_size=2, shuffle=True))
    assert torch.equal(after, torch.random.get_rng_state())
    assert [g.num_graphs for g, _, _ in actual] == [2, 1]
    for (a, al, aw), (b, bl, bw) in zip(actual, expected):
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            else:
                assert a[key] == b[key]
        torch.testing.assert_close(al, bl)
        torch.testing.assert_close(aw, bw)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('kind', ['negative', 'overflow', 'missing', 'float', 'int32', 'rank', 'shape'])
@pytest.mark.parametrize('identifiers', [False, True])
def test_input_failure_saved_before_original_collator(tmp_path, kind, identifiers):
    rows = samples()
    graph = rows[1][0]
    if not identifiers:
        del graph.evaluation_match_id
        del graph.evaluation_source_row
    if kind == 'negative': graph.edge_index[0, 0] = -1
    if kind == 'overflow': graph.edge_index[1, 0] = graph.num_nodes
    if kind == 'missing': graph.edge_index = None
    if kind == 'float': graph.edge_index = graph.edge_index.float()
    if kind == 'int32': graph.edge_index = graph.edge_index.int()
    if kind == 'rank': graph.edge_index = graph.edge_index.flatten()
    if kind == 'shape': graph.edge_index = graph.edge_index[:1]
    def forbidden(_): pytest.fail('Invalid input reached original collator')
    collate = DiagnosticCollater(forbidden, tmp_path)
    collate.start_epoch(25, 'validation')
    with pytest.raises(ValueError) as caught:
        collate(rows)
    saved = snapshot(tmp_path)
    assert saved['context'] == {'epoch': 25, 'phase': 'validation', 'batch': 0,
                                'stage': 'before_collation',
                                'sample': {'graph': 1, **({'match_id': 'm1', 'source_row': 11} if identifiers else {})}}
    assert isinstance(saved['graphs'], list) and len(saved['graphs']) == 2
    torch.testing.assert_close(saved['labels'][1], rows[1][1])
    torch.testing.assert_close(saved['weights'][1], rows[1][2])
    assert "'graph': 1" in str(caught.value)
    if kind in ('negative', 'overflow'):
        assert "'edge': 0" in str(caught.value) and 'endpoints' in str(caught.value)
        torch.testing.assert_close(saved['graphs'][1].edge_index, graph.edge_index)
    assert 'saved to' in caught.value.__notes__[0]


def test_empty_edges_accepted_and_nonfinite_features_checked_after_collation(tmp_path):
    from torch_geometric.loader import DataLoader
    rows = samples()
    for g, _, _ in rows:
        g.edge_index = torch.empty((2, 0), dtype=torch.long)
        g.edge_attr = torch.empty((0, 1))
    loader = DataLoader(rows, batch_size=2)
    install_collation_diagnostics(loader, enabled=True, directory=tmp_path)
    assert next(iter(loader))[0].num_edges == 0
    rows[1][0].x[0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite graph x'):
        next(iter(loader))
    assert snapshot(tmp_path)['context']['stage'] == 'after_collation'


def test_collator_corruption_saves_collated_batch_and_inputs(tmp_path):
    from torch_geometric.loader import DataLoader
    rows = samples()
    original = DataLoader(rows).collate_fn
    def corrupt(rows):
        result = original(rows)
        result[0].edge_index[1, 0] = 2
        return result
    collate = DiagnosticCollater(corrupt, tmp_path)
    collate.start_epoch(25, 'validation')
    with pytest.raises(ValueError, match='Edge crosses graph boundaries'):
        collate(rows)
    saved = snapshot(tmp_path)
    assert saved['context'] == {'epoch': 25, 'phase': 'validation', 'batch': 0, 'stage': 'after_collation'}
    assert isinstance(saved['graphs'], Batch)
    assert len(saved['input_graphs']) == 2
    assert saved['input_graphs'][0].edge_index[1, 0] == 1
    assert saved['graphs'].edge_index[1, 0] == 2


def test_pre_collation_snapshot_write_failure_preserves_error(tmp_path, monkeypatch):
    rows = samples()
    rows[0][0].edge_index[0, 0] = -1
    def fail(*args, **kwargs): raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(ValueError, match='out of bounds') as caught:
        DiagnosticCollater(None, tmp_path)(rows)
    assert 'disk full' in caught.value.__notes__[0]
    assert not list((tmp_path / 'batch_failures').iterdir())


@pytest.mark.parametrize('epoch,train', [(25, False), (26, True)])
def test_run_epoch_resets_diagnostic_context_despite_telemetry_failure(tmp_path, monkeypatch, epoch, train):
    from types import SimpleNamespace
    from torch_geometric.loader import DataLoader
    from models.utils import run_epoch
    import training_monitor
    monitor = training_monitor.TrainingMonitor(tmp_path, enabled=False)
    monitor.enabled = True  # Exercise logging without starting a sampler process.
    def fail(*args, **kwargs): raise OSError('telemetry unavailable')
    monkeypatch.setattr(training_monitor, 'append_event', fail)
    monitor.event('test')
    rows = samples()
    loader = DataLoader(rows, batch_size=2)
    install_collation_diagnostics(loader, enabled=True, directory=tmp_path)
    loader.collate_fn.start_epoch(24, 'train')
    list(loader)  # Prior iteration must not leak its batch count into the next epoch.
    rows[0][0].edge_index[0, 0] = -1
    args = SimpleNamespace(task='pass_success', gnn_task='node_binary', clip=1.,
                           monitoring='on', _epoch=epoch, _monitor=monitor, _batch_failure_dir=tmp_path)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters()) if train else None
    with pytest.raises(ValueError, match='before collation'):
        run_epoch(args, model, loader, optimizer, device='cpu', train=train)
    context = snapshot(tmp_path)['context']
    assert context['epoch'] == epoch and context['batch'] == 0
    assert context['phase'] == ('train' if train else 'validation')
    assert context['stage'] == 'before_collation'


def test_retained_check_catches_corruption_after_collation(tmp_path):
    from types import SimpleNamespace
    from torch_geometric.loader import DataLoader
    from models.utils import run_epoch
    loader = DataLoader(samples(), batch_size=2)
    install_collation_diagnostics(loader, enabled=True, directory=tmp_path)
    class CorruptAfterCollationLoader(DataLoader):
        def __iter__(self):
            for result in super().__iter__():
                result[0].edge_index[1, 0] = 2
                yield result
    loader.__class__ = CorruptAfterCollationLoader
    args = SimpleNamespace(task='pass_success', gnn_task='node_binary', monitoring='on',
                           _epoch=25, _batch_failure_dir=tmp_path)
    with pytest.raises(ValueError, match='Edge crosses graph boundaries'):
        run_epoch(args, torch.nn.Linear(2, 1), loader, device='cpu')
    assert snapshot(tmp_path)['context']['stage'] == 'before_device_transfer'


def test_batch_counter_advances_until_failure(tmp_path):
    from torch_geometric.loader import DataLoader
    rows = samples() + samples()
    rows[-1][0].edge_index[0, 0] = -1
    loader = DataLoader(rows, batch_size=2)
    install_collation_diagnostics(loader, enabled=True, directory=tmp_path)
    loader.collate_fn.start_epoch(25, 'validation')
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(ValueError): next(iterator)
    assert snapshot(tmp_path)['context']['batch'] == 1
