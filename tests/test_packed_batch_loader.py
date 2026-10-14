import argparse
import copy
import gc
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater

from packed_graphs import PackedGraphs, pack_graphs
from packed_batch_loader import (PackedBatchLoader, direct_packed_batch, make_batch_loader,
                                 UnsupportedPackedSchema)
from training_monitor import TrainingMonitor
from training_state import add_training_runtime_arguments, restored_batch_loader, dataset_signature
from test_prepared_dataset import artifacts, make_disk


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def equal_batch(left, right):
    equal(left[0].to_dict(), right[0].to_dict())
    equal(left[0]._slice_dict, right[0]._slice_dict)
    equal(left[0]._inc_dict, right[0]._inc_dict)
    equal(left[1:], right[1:])
    for a, b in zip(left[0].to_data_list(), right[0].to_data_list()):
        equal(a.to_dict(), b.to_dict())


def packed_samples(extra=None):
    graphs = []
    for i, n in enumerate((2, 5, 3)):
        edge = torch.tensor([[0, 1], [1, 0]]) if i != 1 else torch.empty((2, 0), dtype=torch.long)
        graph = Data(x=torch.arange(n*2).reshape(n, 2).float(), edge_index=edge,
                     edge_attr=torch.ones(edge.shape[1], 2), score=torch.tensor(float(i)),
                     evaluation_match_id='match', evaluation_source_row=i,
                     node_ids=[str(j) for j in range(n)], edge_feature_names=['a', 'b'],
                     mask=torch.ones(n, dtype=torch.bool))
        if extra:
            extra(graph, i)
        graphs.append(graph)
    labels = torch.arange(9).reshape(3, 3).float()
    raw = pack_graphs(graphs, labels)
    return dict(graphs=PackedGraphs(raw), labels=labels)


def test_direct_batch_exact_roundtrip_and_no_reconstruction():
    payload = packed_samples()
    rows = [(0, r) for r in (2, 0, 1)]
    weights = [torch.tensor(float(r+1)) for _, r in rows]
    reference = Collater(None)([(payload['graphs'][r], payload['labels'][r], w)
                               for (_, r), w in zip(rows, weights)])
    with patch.object(PackedGraphs, '__getitem__', side_effect=AssertionError('reconstructed')):
        actual = direct_packed_batch([payload], rows, weights)
    equal_batch(tuple(reference), actual)
    actual[0].node_ids[0][0] = 'changed'
    assert payload['graphs'][2].node_ids[0] == '0'


@pytest.mark.parametrize('extra', [
    lambda g, i: setattr(g, 'nested', {'items': [i, i+1]}),
    lambda g, i: setattr(g, 'other_index', torch.tensor([0])),
    lambda g, i: setattr(g, 'num_nodes', len(g.x)),
    lambda g, i: setattr(g, 'optional', torch.ones(1)) if i == 1 else None,
])
def test_unsupported_schema_uses_pyg(extra):
    payload = packed_samples(extra)
    rows = [(0, r) for r in range(3)]
    with pytest.raises(UnsupportedPackedSchema):
        direct_packed_batch([payload], rows)
    loader = PackedBatchLoader(SimpleNamespace(), 3)
    # Optional attributes absent on the first graph are ignored by PyG too.
    expected = Collater(None)([(payload['graphs'][r], payload['labels'][r], torch.tensor(1.)) for _, r in rows])
    equal_batch(tuple(expected), tuple(loader._finish([payload], rows, [torch.tensor(1.)]*3)))
    assert loader.fallback_batches == 1 and loader.direct_batches == 0


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('batch_size', [1, 3, 7, 64])
def test_stream_order_ipw_partial_batches_rng_and_cache_reuse(tmp_path, artifacts, shuffle, batch_size):
    dataset = make_disk(tmp_path, artifacts, shuffle=shuffle, buffer_matches=2, seed=9)
    dataset.set_inverse_propensity_weights(torch.arange(len(dataset)).float()+.5)
    cache_id = dataset.cache_id
    fast = make_batch_loader(dataset, batch_size=batch_size)
    assert isinstance(fast, PackedBatchLoader)
    for epoch in (0, 3, 4):
        dataset.set_epoch(epoch)
        torch.manual_seed(19)
        expected = list(DataLoader(dataset, batch_size=batch_size))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(19)
        actual = list(fast)
        equal(torch.get_rng_state(), expected_rng)
        assert len(actual) == len(expected) == len(fast)
        for left, right in zip(expected, actual):
            equal_batch(tuple(left), tuple(right))
    with patch('prepared_dataset.PreparedActionDataset._build_entry', side_effect=AssertionError('rebuilt')):
        assert make_disk(tmp_path, artifacts).cache_id == cache_id


def test_memory_bound_at_partial_buffer_boundary(tmp_path, artifacts, monkeypatch):
    dataset = make_disk(tmp_path, artifacts, shuffle=False)
    original = dataset._load_entry
    prior = []
    def load(entry):
        gc.collect()
        assert all(ref() is None for ref in prior), 'previous match storage survived'
        payload = original(entry)
        prior.extend([weakref.ref(payload['graphs']), weakref.ref(payload['graphs'].fields[-1][2])])
        return payload
    monkeypatch.setattr(dataset, '_load_entry', load)
    iterator = iter(PackedBatchLoader(dataset, 3))
    first = next(iterator)  # first match has two rows; this crosses the boundary
    iterator.close()
    gc.collect()
    assert all(ref() is None for ref in prior)
    assert first[0].num_graphs == 3


def test_integrity_errors_and_fallback_errors_propagate(tmp_path, artifacts):
    dataset = make_disk(tmp_path, artifacts)
    from pathlib import Path
    Path(dataset.entries[0]['path']).write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='changed'):
        next(iter(PackedBatchLoader(dataset, 3)))
    payload = packed_samples(lambda g, i: setattr(g, 'optional', torch.ones(1)) if i == 0 else None)
    loader = PackedBatchLoader(SimpleNamespace(), 3)
    with pytest.raises(KeyError):
        loader._finish([payload], [(0, r) for r in range(3)], [torch.tensor(1.)]*3)


def test_defaults_resume_overrides_identity_and_factory(tmp_path, artifacts):
    parser = argparse.ArgumentParser()
    add_training_runtime_arguments(parser)
    assert vars(parser.parse_args([]))['batch_loader'] == 'auto'
    assert parser.parse_args([]).monitoring == 'off'
    assert parser.parse_args(['--monitoring', 'light']).monitoring == 'light'
    assert restored_batch_loader({}) == 'pyg'
    assert restored_batch_loader({'batch_loader': 'auto'}) == 'auto'
    assert restored_batch_loader({}, 'auto') == 'auto'
    assert restored_batch_loader({'batch_loader': 'auto'}, 'pyg') == 'pyg'
    dataset = make_disk(tmp_path, artifacts)
    assert isinstance(make_batch_loader(dataset, batch_size=3, monitoring='on'), DataLoader)
    assert isinstance(make_batch_loader(dataset, batch_size=3, batch_loader='pyg'), DataLoader)
    assert isinstance(make_batch_loader([], batch_size=3), DataLoader)
    args = SimpleNamespace(feature_dir=str(artifacts['feature_dir']), batch_loader='auto', monitoring='off')
    original = dataset_signature(args)
    args.batch_loader, args.monitoring = 'pyg', 'light'
    assert dataset_signature(args) == original
    from scripts.train_relevant_models import parse_args
    assert parse_args(['--resume-id', 'pass_intent/a']).batch_loader is None
    assert parse_args(['--resume-id', 'pass_intent/a', '--batch-loader', 'auto']).batch_loader == 'auto'


@pytest.mark.parametrize('mode,expected', [('off', []), ('light', [0, 50, 100]), ('on', list(range(101)))])
def test_monitoring_sampling_and_failure_tolerance(tmp_path, mode, expected):
    with patch('training_monitor.subprocess.Popen', side_effect=OSError('unavailable')):
        monitor = TrainingMonitor(tmp_path, mode=mode)
        assert [i for i in range(101) if monitor.should_sample_batch(i)] == expected
        with patch('training_monitor.append_event', side_effect=OSError('full')):
            monitor.event('failure', error='example')
        monitor.close()


def test_cpu_optimizer_and_dropout_replay(tmp_path, artifacts):
    dataset = make_disk(tmp_path, artifacts, shuffle=True, buffer_matches=2)
    dataset.set_inverse_propensity_weights(torch.linspace(.5, 1.5, len(dataset)))
    initial = torch.nn.Sequential(torch.nn.Linear(dataset._load_entry(dataset.entries[0])['graphs'][0].x.shape[1], 2),
                                  torch.nn.Dropout(.4))
    results = []
    for loader in (DataLoader(dataset, batch_size=3), PackedBatchLoader(dataset, 3)):
        torch.manual_seed(7)
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        predictions = []
        for graph, labels, weights in loader:
            pred = model(graph.x)
            loss = pred.square().mean() * weights.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions.append(pred.detach())
        results.append((predictions, model.state_dict(), optimizer.state_dict()))
    equal(results[0], results[1])


@pytest.mark.parametrize('override', [None, 'auto', 'pyg'])
def test_wrapper_resume_forwards_only_explicit_loader(tmp_path, monkeypatch, override):
    from scripts import train_relevant_models as wrapper
    flags = ['--resume-id', 'pass_intent/a', '--monitoring', 'light']
    if override is not None:
        flags += ['--batch-loader', override]
    args = wrapper.parse_args(flags)
    monkeypatch.setattr(wrapper, 'resolve_resume_checkpoint', lambda *a: tmp_path/'checkpoint.pt')
    monkeypatch.setattr(wrapper, 'load_checkpoint', lambda *a: {'args': {'task':'pass_intent'}, 'finished':False})
    monkeypatch.setattr(wrapper, 'publish_checkpoint_artifacts', lambda *a: None)
    monkeypatch.setattr(wrapper, 'resolve_training_goal_settings', lambda *a: None)
    commands = []
    monkeypatch.setattr(wrapper, 'run_training', lambda command, **kwargs: commands.append(command))
    wrapper._main(args)
    assert wrapper.get_cli_value(commands[0], '--batch-loader') == override
    assert wrapper.get_cli_value(commands[0], '--monitoring') == 'light'


@pytest.mark.parametrize('monitoring', ['off', 'light'])
def test_unsampled_monitor_skips_identifiers_but_keeps_validation(tmp_path, monkeypatch, monitoring):
    from models.utils import run_epoch
    from datatools.config import LABEL_COLUMNS
    payload = packed_samples()
    # A local overflow becomes a cross-graph edge in the direct batch.
    payload['graphs'][0].edge_index[1, 0] = 2
    graph, _, _ = direct_packed_batch([payload], [(0, 0), (0, 1)])
    class OneBatch:
        collate_fn = None
        def __len__(self): return 1
        def __iter__(self):
            yield graph, torch.zeros((2,len(LABEL_COLUMNS))), torch.ones(2)
    class NoForward(torch.nn.Module):
        def forward(self, *args): pytest.fail('invalid batch reached forward')
    monitor = TrainingMonitor(tmp_path, enabled=False)
    monitor.mode = monitoring
    monitor.enabled = monitoring == 'light'
    # Simulate an unsampled batch; integration must ask before building metadata.
    monkeypatch.setattr(monitor, 'should_sample_batch', lambda index: False)
    def forbidden(*args): pytest.fail('identifiers collected for unsampled batch')
    monkeypatch.setattr('models.node_selection.batch_identifiers', forbidden)
    args = SimpleNamespace(task='pass_success',gnn_task='node_binary',_epoch=1,
                           _monitor=monitor,monitoring=monitoring,_batch_failure_dir=tmp_path)
    with pytest.raises(ValueError, match='Edge crosses graph boundaries'):
        run_epoch(args, NoForward(), OneBatch(), device='cpu')
    assert len(list((tmp_path/'batch_failures').glob('*.pt'))) == 1


def test_refit_allows_runtime_loader_and_light_monitoring():
    from scripts.train_relevant_models import parse_args
    args = parse_args(['--refit-source-model-id','pass_intent/a','--batch-loader','pyg','--monitoring','light'])
    assert args.refit_operational['batch_loader'] == 'pyg'
    assert args.refit_operational['monitoring'] == 'light'


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match='positive integer'):
        PackedBatchLoader(SimpleNamespace(), batch_size)
