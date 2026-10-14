import copy
import gc
import weakref

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from packed_graphs import pack_graphs, decode_payload, PackedGraphs, validate_payload


def assert_graph(a, b):
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        if isinstance(a[key], torch.Tensor):
            assert a[key].dtype == b[key].dtype
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
        else:
            assert type(a[key]) is type(b[key]) and a[key] == b[key]


def examples():
    graphs = []
    for row, n in enumerate([2, 5, 3]):
        edge = torch.tensor([[0, n-1], [n-1, 0]]) if row != 1 else torch.empty((2, 0), dtype=torch.long)
        graph = Data(x=torch.arange(n*3, dtype=torch.float64).reshape(n, 3), edge_index=edge,
                     edge_attr=torch.zeros((edge.shape[1], 2)), num_nodes=n,
                     scalar=torch.tensor(row, dtype=torch.int16), empty=torch.empty((0, 4)),
                     diagnostic=torch.tensor([float('nan'), row]),
                     evaluation_match_id='match', evaluation_source_row=row,
                     metadata={'nested': [1, {'height': 2.0}], 'tuple': (True, b'bytes')})
        if row != 1:
            graph.optional = torch.ones((n, 2), dtype=torch.bool)
        if row == 2:
            graph.optional_text = 'late'
        graphs.append(graph)
    return graphs, torch.arange(12, dtype=torch.float32).reshape(3, 4)


def test_round_trip_optional_scalar_empty_and_local_indices(tmp_path):
    graphs, labels = examples()
    payload = pack_graphs(graphs, labels)
    path = tmp_path / 'packed.pt'
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=True)
    decoded = decode_payload(loaded, expected_count=3)
    assert isinstance(decoded['graphs'], PackedGraphs)
    torch.testing.assert_close(decoded['labels'], labels, rtol=0, atol=0)
    for a, b in zip(graphs, decoded['graphs'], strict=True):
        assert_graph(a, b)
    assert_graph(decoded['graphs'][-1], graphs[-1])
    with pytest.raises(IndexError):
        decoded['graphs'][3]
    with pytest.raises(TypeError):
        decoded['graphs'][1.2]
    # Batching homogeneous keys still applies PyG's increments exactly once.
    for graph in graphs:
        graph.pop('optional', None); graph.pop('optional_text', None)
    packed = PackedGraphs(pack_graphs(graphs, labels))
    a = next(iter(DataLoader(graphs, batch_size=3)))
    b = next(iter(DataLoader(packed, batch_size=3)))
    assert_graph(a, b)


def test_empty_match_and_no_attributes():
    assert len(PackedGraphs(pack_graphs([], torch.empty((0, 4))))) == 0
    assert PackedGraphs(pack_graphs([Data()], torch.zeros((1, 4))))[0].keys() == []


def test_views_disjoint_samples_and_fresh_mutable_metadata():
    graphs, labels = examples()
    payload = pack_graphs(graphs, labels)
    lazy = PackedGraphs(payload)
    first, again, second = lazy[0], lazy[0], lazy[1]
    assert first is not again
    assert first.x.untyped_storage().data_ptr() == payload['fields']['x']['value'].untyped_storage().data_ptr()
    first.x.fill_(999)
    torch.testing.assert_close(second.x, graphs[1].x, rtol=0, atol=0)
    first.metadata['nested'][1]['height'] = 7
    assert again.metadata['nested'][1]['height'] == 2
    assert lazy[0].metadata['nested'][1]['height'] == 2
    # Preserve nested types rather than treating int and float values as constants.
    mixed = [Data(meta={'a': [1]}), Data(meta={'a': [1.]})]
    decoded = PackedGraphs(pack_graphs(mixed, torch.zeros((2, 1))))
    assert type(decoded[0].meta['a'][0]) is int
    assert type(decoded[1].meta['a'][0]) is float


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(version=1), lambda p: p.update(format='graphs'),
    lambda p: p.update(count=4), lambda p: p.update(labels=torch.zeros((2, 4))),
    lambda p: p['fields']['x'].update(dim=3),
    lambda p: p['fields']['x'].update(offsets=[0, 2, 1, 10]),
    lambda p: p['fields']['x'].update(offsets=[0, 2, 5]),
    lambda p: p['fields']['x'].update(offsets=[1, 2, 7, 10]),
    lambda p: p['fields']['scalar'].update(offsets=[0, 0, 2, 3]),
    lambda p: p['fields']['optional'].update(rows=[2, 0]),
    lambda p: p['fields']['optional'].update(rows=[0, 3]),
    lambda p: p['fields']['x'].update(kind='unknown'),
    lambda p: p['fields']['metadata'].update(value=object()),
])
def test_invalid_payloads(mutation):
    payload = pack_graphs(*examples())
    mutation(payload)
    with pytest.raises(ValueError, match='Invalid packed'):
        validate_payload(payload)


@pytest.mark.parametrize('value', [object(), {'nested': torch.ones(2)}, {1, 2}, Data(x=torch.ones(2))])
def test_unsupported_attributes_are_explicit(value):
    with pytest.raises(ValueError, match='Unsupported graph attribute'):
        pack_graphs([Data(attribute=value)], torch.zeros((1, 1)))


def test_incompatible_tensor_schema_and_counts():
    with pytest.raises(ValueError, match='counts differ'):
        pack_graphs([Data()], torch.zeros((0, 1)))
    for tensors in [(torch.ones(2, 3), torch.ones(2, 4)),
                    (torch.ones(2), torch.ones(2, dtype=torch.int64))]:
        with pytest.raises(ValueError, match='[Ii]nconsistent'):
            pack_graphs([Data(x=tensors[0]), Data(x=tensors[1])], torch.zeros((2, 1)))


def test_graph_access_is_lazy_and_storage_lifetime_is_bounded(monkeypatch):
    import packed_graphs
    graphs, labels = examples()
    payload = pack_graphs(graphs, labels)
    ref = weakref.ref(payload['fields']['x']['value'])
    lazy = PackedGraphs(payload)
    with monkeypatch.context() as context:
        context.setattr(packed_graphs, 'Data', lambda: pytest.fail('Eager graph reconstruction'))
        assert len(lazy) == 3
    view = lazy[0]
    graph_ref = weakref.ref(view)
    del payload, lazy
    gc.collect()
    assert ref() is not None  # The caller's input view may outlive its match.
    del view
    gc.collect()
    assert graph_ref() is None and ref() is None
