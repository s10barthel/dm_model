import pytest
import torch
from torch_geometric.data import Batch, Data
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from models.observed_selection import observed_layout, select_observed


@pytest.mark.parametrize('task', ['pass_success', 'pass_height', 'action_success', 'outcome_scoring',
                                 'outcome_conceding', 'outcome_return', 'intent_return', 'intent_return_oppo_agn'])
@pytest.mark.parametrize('include_out', [False, True])
def test_selection_loss_and_gradients(task, include_out):
    torch.manual_seed(17)
    graphs = Batch.from_data_list([Data(x=torch.randn(n, 2)) for n in (2, 5, 3)])
    labels = torch.zeros(3, len(LABEL_COLUMNS))
    labels[:, LABEL_INDEX['intent_index']] = torch.tensor([0, 5 if include_out else 4, 1])
    labels[:, LABEL_INDEX['success']] = torch.tensor([1, 0, 1])
    labels[0, LABEL_INDEX['is_dribble']] = 1
    branches = task.startswith('outcome_')
    multi = branches or task.startswith('intent_return')
    positions, branch = observed_layout(graphs, labels, include_out, branches)
    size = graphs.num_nodes + (3 if include_out else 0)
    out = torch.randn((size, 2) if multi else (size,), dtype=torch.double, requires_grad=True)
    batch = torch.cat((graphs.batch, torch.arange(3))) if include_out else graphs.batch
    old = torch.stack([out[batch == i][int(labels[i, 5])] for i in range(3)])
    if branches:
        old = old[torch.arange(3), labels[:, LABEL_INDEX['success']].long()]
    new = select_observed(out, positions, branch)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    weights = torch.tensor([1., .3, 1.7], dtype=torch.double)
    targets = torch.tensor([0., .2, 1.], dtype=torch.double)
    def loss(pred):
        if task == 'outcome_return':
            return torch.nn.functional.mse_loss(pred * 2 - 1, targets)
        if task.startswith('intent_return'):
            return sum(torch.nn.functional.binary_cross_entropy_with_logits(pred[:, i], targets,
                       weight=weights, pos_weight=torch.tensor(1.2)) for i in range(2))
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, targets, weight=weights,
                                                                     pos_weight=torch.tensor(1.2))
    torch.testing.assert_close(loss(new), loss(old))
    torch.testing.assert_close(torch.autograd.grad(loss(new), out)[0], torch.autograd.grad(loss(old), out)[0])


@pytest.mark.parametrize('target,branch', [(float('nan'), 0), (float('inf'), 0), (.5, 0), (-1, 0), (3, 0),
                                        (0, .5), (0, 2), (0, float('nan'))])
def test_invalid_targets(target, branch):
    graph = Batch.from_data_list([Data(x=torch.zeros(3, 2), evaluation_match_id='match', evaluation_source_row=7)])
    labels = torch.zeros(1, len(LABEL_COLUMNS))
    labels[0, 5], labels[0, LABEL_INDEX['success']] = target, branch
    with pytest.raises(ValueError, match='match match, source row 7'):
        observed_layout(graph, labels, branches=True)


def test_single_graph_and_physical_sidecar_selection():
    graph = Batch.from_data_list([Data(x=torch.arange(12).reshape(3, 4).float())])
    labels = torch.zeros(1, len(LABEL_COLUMNS))
    labels[0, 5] = 2
    positions, _ = observed_layout(graph, labels)
    sidecar = torch.tensor([.1, .2, .3])
    torch.testing.assert_close(sidecar[positions], sidecar[graph.batch == 0][2:3])
    mask = labels[:, LABEL_INDEX['is_pass']].bool()
    assert sidecar[positions[mask]].numel() == 0


@pytest.mark.parametrize('task', ['pass_success', 'pass_height', 'action_success', 'outcome_scoring',
                                 'outcome_conceding', 'outcome_return', 'intent_return', 'intent_return_oppo_agn'])
def test_run_epoch_metrics_rows_and_updates(task, monkeypatch):
    import copy
    from types import SimpleNamespace
    import numpy as np
    from torch_geometric.loader import DataLoader
    from datatools import config
    from models.utils import run_epoch
    import models.observed_selection as selection
    torch.manual_seed(3)
    samples = []
    for i in range(5):
        n = i + 2
        graph = Data(x=torch.randn(n, config.NODE_FEATURE_CORE_DIM),
                     edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.ones(2, 2),
                     evaluation_match_id='m', evaluation_source_row=i)
        label = torch.zeros(len(LABEL_COLUMNS))
        for field, value in {'intent_index': i % n, 'success': i % 2, 'is_pass': i != 0,
                             'is_dribble': i == 0, 'scores': i % 2, 'concedes': (i + 1) % 2,
                             'pass_high': i % 2, 'pass_max_ball_z': 2}.items():
            label[LABEL_INDEX[field]] = value
        samples.append((graph, label, torch.tensor(1. if i == 0 else .7)))
    multi = task.startswith(('outcome_', 'intent_return'))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(config.NODE_FEATURE_CORE_DIM, 2 if multi else 1)
        def forward(self, graphs, destinations=None):
            values = self.linear(graphs.x)
            return values if multi else values.squeeze(-1)
    first = Model()
    second = copy.deepcopy(first)
    args = SimpleNamespace(task=task, gnn_task='node_regression' if task == 'outcome_return' else 'node_binary',
                           include_out=False, lambda_l1=.001, clip=10, print_freq=99,
                           use_xg=False, use_xt=False, use_goal_distance=False, use_epv=False)
    def evaluate(model):
        return run_epoch(args, model, DataLoader(samples, batch_size=3),
                         torch.optim.SGD(model.parameters(), lr=.01), device='cpu', train=True,
                         return_learning_curve_rows=True)
    actual = evaluate(first)
    # Scalar indexing reference builds the separate selection autograd operations.
    def scalar(out, positions, branches=None):
        return torch.stack([out[p] if branches is None else out[p, branches[i]]
                            for i, p in enumerate(positions)])
    monkeypatch.setattr(selection, 'select_observed', scalar)
    expected = evaluate(second)
    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        elif isinstance(a, str):
            assert a == b
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    compare(actual, expected)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b)
