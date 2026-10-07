import pytest
import torch
from torch_geometric.data import Batch, Data

from models.graph_diagnostics import validate_with_snapshot
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
