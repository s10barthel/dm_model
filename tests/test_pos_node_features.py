import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
from datatools.utils import filter_features_and_labels
from dataset import ActionDataset
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.gnn import GNN, mask_pos_node_features
from models.node_feature_config import resolve_pos_node_features
from models import utils as model_utils
from physical_pass_model import physical_state_hash, graph_pass_distances
from scripts import train_relevant_models as wrapper
from test_benchmark_no_accel import make_model_args, make_labels, write_checkpoint


def graph():
    x = torch.arange(4 * 25, dtype=torch.float32).reshape(4, 25) / 10
    x[:, :3] = 0
    x[:2, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[:, config.NODE_FEATURE_IS_POSSESSOR] = 0
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[:, config.NODE_FEATURE_X:config.NODE_FEATURE_Y + 1] = torch.tensor(
        [[10., 10.], [20., 10.], [20., 30.], [10., 30.]])
    edges = torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j]).T
    result = Data(x=x, edge_index=edges, edge_attr=torch.ones((12, 2)))
    result.node_ids = ['possessor', 'target', 'opponent1', 'opponent2']
    return result


@pytest.mark.parametrize('skip_conn', [False, True])
@pytest.mark.parametrize('model_type', ['gat', 'gcn', 'gin'])
def test_model_boundaries_and_direct_calls(skip_conn, model_type):
    args = dict(make_model_args(), skip_conn=skip_conn, model=model_type)
    enabled = GNN(args).eval()
    disabled = GNN(dict(args, pos_node_features_aware=False)).eval()
    disabled.load_state_dict(enabled.state_dict())
    source = Batch.from_data_list([graph()])
    source.x[3, config.NODE_FEATURE_IS_GOAL] = 1
    original = source.clone()
    masked = source.clone()
    masked.x = mask_pos_node_features(source.x, disabled.args)
    expected = source.x.clone()
    expected[:, config.NODE_FEATURE_X:config.NODE_FEATURE_Y + 1] = 0
    torch.testing.assert_close(masked.x, expected)
    with torch.no_grad():
        torch.testing.assert_close(disabled(source), enabled(masked))
        embeddings, pooled = disabled.encoder(source)
        expected_embeddings, expected_pooled = enabled.encoder(masked)
        torch.testing.assert_close(embeddings, expected_embeddings)
        torch.testing.assert_close(pooled, expected_pooled)
        torch.testing.assert_close(
            disabled.decoder(source.x, embeddings, pooled, source.batch),
            enabled.decoder(masked.x, embeddings, pooled, source.batch))
    torch.testing.assert_close(source.x, original.x)
    torch.testing.assert_close(source.edge_index, original.edge_index)
    torch.testing.assert_close(source.edge_attr, original.edge_attr)
    # Model calls must leave operational geometry intact.
    assert physical_state_hash(source.to_data_list()[0]) == physical_state_hash(original.to_data_list()[0])
    torch.testing.assert_close(graph_pass_distances(source), graph_pass_distances(original))


def test_grid_and_destination_geometry():
    args = dict(make_model_args(), task='pass_dest', skip_conn=True)
    enabled = GNN(args).eval()
    disabled = GNN(dict(args, pos_node_features_aware=False)).eval()
    disabled.load_state_dict(enabled.state_dict())
    source = Batch.from_data_list([graph()])
    masked = source.clone()
    masked.x = mask_pos_node_features(source.x, disabled.args)
    grid = torch.tensor([[15., 15.], [40., 30.]])
    features = model_utils.build_dest_features(source, grid)
    with torch.no_grad():
        torch.testing.assert_close(disabled.forward_grid(source, grid), enabled.forward_grid(masked, grid))
    torch.testing.assert_close(model_utils.build_dest_features(source, grid), features)


@pytest.mark.parametrize('value', [None, True, False])
def test_checkpoint_defaults_and_signatures(tmp_path, value):
    write_checkpoint(tmp_path)
    args_path = tmp_path / 'args.json'
    args = json.loads(args_path.read_text())
    if value is not None:
        args['pos_node_features_aware'] = value
    args_path.write_text(json.dumps(args))
    with patch.object(model_utils, 'get_model_path', return_value=tmp_path):
        record = model_utils.get_model_record('action_intent/positions')
        loaded = model_utils.load_model('action_intent/positions', device='cpu')
    assert record['feature_signature']['pos_node_features_aware'] is (value is not False)
    assert loaded.args['pos_node_features_aware'] is (value is not False)
    assert resolve_pos_node_features({'pos_node_features_aware': None})


def test_wrapper_forwarding_and_conflicts():
    with (
        patch.object(wrapper, 'resolve_feature_run_id', return_value='features'),
        patch.object(wrapper, 'resolve_training_split', return_value=({'train': [], 'test': []}, 'test')),
        patch.object(wrapper, 'infer_feature_run_intended_receiver_modes', return_value=['original']),
        patch.object(wrapper, 'infer_feature_run_return_types', return_value=['disc_0.9']),
    ):
        for flags in [[], ['--no-pos-node-features']]:
            args = wrapper.parse_args(['--feature-run-id', 'features', '--success-intent-only', *flags])
            resolved = wrapper.resolve_wrapper_feature_flags(args)
            assert resolved['pos_node_features_aware'] is (not flags)
            command = wrapper.append_low_level_feature_flags([], resolved)
            assert ('--no-pos-node-features' in command) is bool(flags)
        for flags in [['--xy-only', '--no-pos-node-features'], ['--no-pos-node-features', '--xy-only']]:
            with pytest.raises(SystemExit):
                wrapper.parse_args(['--feature-run-id', 'features', '--success-intent-only', *flags])
    with pytest.raises(ValueError, match='--xy-only'):
        GNN(dict(make_model_args(), xy_only=True, pos_node_features_aware=False))
    with pytest.raises(ValueError, match='--xy-only'):
        model_utils.extract_model_feature_signature(dict(xy_only=True, pos_node_features_aware=False))


@pytest.mark.parametrize('sparsify', ['none', 'distance', 'delaunay'])
@pytest.mark.parametrize('blockers', [False, True])
def test_dataset_and_runtime_keep_geometry(tmp_path, sparsify, blockers):
    (tmp_path / 'features').mkdir()
    (tmp_path / 'labels').mkdir()
    source = graph()
    torch.save([source], tmp_path / 'features/m.pt')
    torch.save(make_labels(), tmp_path / 'labels/m.pt')
    options = dict(task='action_intent', possessor_aware=True, sparsify=sparsify,
                   max_edge_dist=10, drop_non_blockers=blockers)
    datasets = [ActionDataset(['m'], feature_dir=str(tmp_path / 'features'),
                 label_dir=str(tmp_path / 'labels'), pos_node_features_aware=value, **options)
                for value in [True, False]]
    for key in ['x', 'edge_index', 'edge_attr']:
        torch.testing.assert_close(datasets[0][0][0][key], datasets[1][0][0][key])
    torch.testing.assert_close(datasets[0].labels, datasets[1].labels)
    runtime_options = dict(options, xy_only=False, extend_features=False, keeper_aware=False,
        ball_z_aware=False, poss_vel_aware=False, filter_blockers=blockers)
    results = [filter_features_and_labels([source], make_labels(),
                   dict(runtime_options, pos_node_features_aware=value)) for value in [True, False]]
    for key in ['x', 'edge_index', 'edge_attr']:
        torch.testing.assert_close(results[0][0][0][key], results[1][0][0][key])


@pytest.mark.parametrize('target_value,checkpoint_value', [(True, False), (False, True)])
def test_ipw_uses_own_setting(target_value, checkpoint_value):
    target = build_action_dataset_kwargs(dict(task='pass_success', pos_node_features_aware=target_value),
                                         train=False, diagnostic_label_dir=None)
    dependency = build_ipw_dataset_kwargs(target, dict(pos_node_features_aware=checkpoint_value), None,
        diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert target['pos_node_features_aware'] is target_value
    assert dependency['pos_node_features_aware'] is checkpoint_value


def test_low_level_cli_conflict():
    # Execute only argparse setup and the actual position-validation block.
    # Importing train.py would start training and touch artifacts.
    import ast
    tree = ast.parse(Path('train.py').read_text())
    namespace = {'argparse': argparse, 'resolve_pos_node_features': resolve_pos_node_features}
    nodes = [node for node in tree.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'parser'
        and any(isinstance(arg, ast.Constant) and arg.value in {'--xy_only', '--no-pos-node-features'}
                for arg in node.value.args)]
    namespace['parser'] = argparse.ArgumentParser()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'train.py', 'exec'), namespace)
    validation = next(node for node in tree.body if isinstance(node, ast.Try)
        and 'resolve_pos_node_features' in ast.unparse(node))
    for flags in [['--xy_only', '--no-pos-node-features'], ['--no-pos-node-features', '--xy_only']]:
        namespace['args'] = namespace['parser'].parse_args(flags)
        with pytest.raises(SystemExit):
            exec(compile(ast.Module(body=[validation], type_ignores=[]), 'train.py', 'exec'), namespace)
