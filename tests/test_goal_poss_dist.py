import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
from datatools.utils import filter_features_and_labels
from dataset import ActionDataset
from models import utils as model_utils
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.gnn import GNN
from models.node_feature_config import mask_goal_poss_features, resolve_goal_poss_dist
from prepared_dataset import PreparedActionDataset, preparation_code_paths
from scripts import train_relevant_models as wrapper
from test_benchmark_no_accel import make_model_args, make_pass_labels, write_checkpoint
from test_training_refit import sources
from test_training_controls import synthetic_training
from training_refit import prepare_refit, load_refit_config

GEOMETRY = slice(config.NODE_FEATURE_POSS_DIST, config.NODE_FEATURE_POSS_DY_NORM + 1)
FEATURE_OPTIONS = dict(
    task="action_intent", xy_only=False, possessor_aware=True, keeper_aware=True,
    ball_z_aware=True, poss_vel_aware=True, poss_rel_vel_aware=True,
    extend_features=True, goal_context_version=2, include_goals=True, sparsify="none",
)


def graph():
    # Interleave defending and attacking goal nodes with players.
    x = torch.arange(6 * 25, dtype=torch.float32).reshape(6, 25) / 10
    x[:, :3] = 0
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = torch.tensor([1, 0, 0, 1, 1, 0])
    x[[1, 3], config.NODE_FEATURE_IS_GOAL] = 1
    x[5, config.NODE_FEATURE_IS_KEEPER] = 1
    x[:, config.NODE_FEATURE_IS_POSSESSOR] = 0
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[:, 3:5] = torch.tensor([[40., 34.], [0., 34.], [50., 20.],
                              [105., 34.], [60., 40.], [5., 34.]])
    delta = x[:, 3:5] - x[0, 3:5]
    distance = delta.norm(dim=1)
    x[:, GEOMETRY] = torch.cat([distance[:, None], delta / (distance[:, None] + 1e-6)], dim=1)
    edges = torch.tensor([(i, j) for i in range(6) for j in range(6) if i != j]).T
    return Data(x=x, edge_index=edges,
                edge_attr=torch.stack([(x[edges[0], 3:5] - x[edges[1], 3:5]).norm(dim=1),
                                       (x[edges[0], 0] == x[edges[1], 0]).float()], dim=1),
                node_ids=["possessor", "away_goal", "opponent", "home_goal", "target", "keeper"])


def expected_mask(x):
    expected = x.clone()
    expected[x[:, config.NODE_FEATURE_IS_GOAL] == 1, GEOMETRY] = 0
    return expected


@pytest.fixture
def artifacts(tmp_path):
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    torch.save([graph()], features / "m.pt")
    torch.save(make_pass_labels(intent_index=4), labels / "m.pt")
    return dict(feature_dir=features, label_dir=labels, **FEATURE_OPTIONS)


@pytest.mark.parametrize("goals", [True, False])
def test_mask_exact_columns_and_no_mutation(goals):
    source = graph()
    if not goals:
        source.x[:, config.NODE_FEATURE_IS_GOAL] = 0
    original = source.clone()
    torch.testing.assert_close(mask_goal_poss_features(source.x, {"goal_poss_dist": False}),
                               expected_mask(source.x))
    for args in [{}, {"goal_poss_dist": True}, {"goal_poss_dist": None}]:
        torch.testing.assert_close(mask_goal_poss_features(source.x, args), source.x)
    torch.testing.assert_close(source.x, original.x)
    assert mask_goal_poss_features(None, {"goal_poss_dist": False}) is None


@pytest.mark.parametrize("model_type", ["gat", "gcn", "gin"])
@pytest.mark.parametrize("skip_conn", [True, False])
def test_batched_encoder_decoder_and_direct_calls(model_type, skip_conn):
    args = dict(make_model_args(), model=model_type, skip_conn=skip_conn)
    enabled = GNN(dict(args, goal_poss_dist=True)).eval()
    disabled = GNN(dict(args, goal_poss_dist=False)).eval()
    disabled.load_state_dict(enabled.state_dict())
    second = graph()
    second.x = second.x.flip(0)
    source = Batch.from_data_list([graph(), second])
    original = source.clone()
    masked = source.clone()
    masked.x = expected_mask(source.x)
    with torch.no_grad():
        torch.testing.assert_close(disabled(source), enabled(masked))
        embeddings, pooled = disabled.encoder(source)
        expected_embeddings, expected_pooled = enabled.encoder(masked)
        torch.testing.assert_close(embeddings, expected_embeddings)
        torch.testing.assert_close(pooled, expected_pooled)
        torch.testing.assert_close(disabled.decoder(source.x, embeddings, pooled, source.batch),
                                   enabled.decoder(masked.x, embeddings, pooled, source.batch))
    for key in ["x", "edge_attr", "edge_index"]:
        torch.testing.assert_close(source[key], original[key])


@pytest.mark.parametrize("override", [{}, {"poss_geometry_aware": False},
                                      {"possessor_aware": False, "extend_features": False},
                                      {"xy_only": True}])
def test_dataset_runtime_and_mask_precedence(artifacts, override):
    options = dict(artifacts, **override)
    enabled = ActionDataset(["m"], goal_poss_dist=True, **options)
    disabled = ActionDataset(["m"], **options)
    raw = graph()
    original = raw.clone()
    args = dict(FEATURE_OPTIONS, **override)
    runtime_enabled, labels_on = filter_features_and_labels(
        [raw], make_pass_labels(intent_index=4), dict(args, goal_poss_dist=True))
    runtime_disabled, labels_off = filter_features_and_labels(
        [raw], make_pass_labels(intent_index=4), dict(args, goal_poss_dist=False))
    for on, off in [(enabled[0][0], disabled[0][0]), (runtime_enabled[0], runtime_disabled[0])]:
        torch.testing.assert_close(off.x, expected_mask(on.x))
        torch.testing.assert_close(on.edge_attr, off.edge_attr)
        torch.testing.assert_close(on.edge_index, off.edge_index)
        assert on.node_ids == off.node_ids
        if override:
            assert torch.count_nonzero(on.x[:, GEOMETRY]) == 0
    torch.testing.assert_close(disabled[0][0].x, runtime_disabled[0].x)
    torch.testing.assert_close(enabled.labels, disabled.labels)
    torch.testing.assert_close(labels_on, labels_off)
    torch.testing.assert_close(raw.x, original.x)
    torch.testing.assert_close(torch.load(artifacts["feature_dir"] / "m.pt", weights_only=False)[0].x,
                               original.x)


@pytest.mark.parametrize("target,auxiliary", [(False, True), (True, False)])
def test_auxiliary_and_ipw_use_own_setting(artifacts, target, auxiliary):
    kwargs = build_action_dataset_kwargs(dict(FEATURE_OPTIONS, goal_poss_dist=target),
                                        train=False, diagnostic_label_dir=None)
    dependency = build_ipw_dataset_kwargs(kwargs, dict(FEATURE_OPTIONS, goal_poss_dist=auxiliary), None,
        diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert dependency["goal_poss_dist"] is auxiliary
    dataset = ActionDataset(["m"], **artifacts, goal_poss_dist=target,
                            auxiliary_model_args=dict(FEATURE_OPTIONS, goal_poss_dist=auxiliary))
    result = dataset[0][0]
    torch.testing.assert_close(result.x, graph().x if target else expected_mask(graph().x))
    torch.testing.assert_close(result.auxiliary_graph.x,
                               graph().x if auxiliary else expected_mask(graph().x))
    legacy_dependency = build_ipw_dataset_kwargs(kwargs, FEATURE_OPTIONS, None,
        diagnostic_label_dir=None, require_goal_next10_diagnostics=False)
    assert legacy_dependency["goal_poss_dist"] is True


@pytest.mark.parametrize("value", [None, True, False])
def test_checkpoint_loading_defaults_and_signatures(tmp_path, value):
    write_checkpoint(tmp_path)
    path = tmp_path / "args.json"
    args = json.loads(path.read_text())
    if value is not None:
        args["goal_poss_dist"] = value
    path.write_text(json.dumps(args))
    with patch.object(model_utils, "get_model_path", return_value=tmp_path):
        record = model_utils.get_model_record("action_intent/geometry")
        loaded = model_utils.load_model("action_intent/geometry", device="cpu")
    expected = value is not False
    assert record["feature_signature"]["goal_poss_dist"] is expected
    assert loaded.args["goal_poss_dist"] is expected
    assert build_action_dataset_kwargs(args, train=False,
                                      diagnostic_label_dir=None)["goal_poss_dist"] is expected
    source = Batch.from_data_list([graph()])
    reference = GNN(dict(args, goal_poss_dist=True)).eval()
    reference.load_state_dict(loaded.state_dict())
    loaded.eval()
    if not expected:
        source.x = expected_mask(source.x)
    with torch.no_grad():
        torch.testing.assert_close(loaded(Batch.from_data_list([graph()])), reference(source))


def test_both_training_cli_defaults_forwarding_and_saved_restoration():
    # Execute actual low-level argparse and saved-setting normalization without starting training.
    tree = ast.parse(Path("train.py").read_text())
    declaration = next(node for node in tree.body if isinstance(node, ast.Expr)
                       and isinstance(node.value, ast.Call)
                       and any(isinstance(arg, ast.Constant) and arg.value == "--goal-poss-dist"
                               for arg in node.value.args))
    restoration = next(node for node in tree.body if isinstance(node, ast.Assign)
                       and ast.unparse(node).startswith("args.goal_poss_dist ="))
    namespace = {"parser": argparse.ArgumentParser(), "resolve_goal_poss_dist": resolve_goal_poss_dist}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), "train.py", "exec"), namespace)
    for flags, expected in [([], False), (["--goal-poss-dist"], True)]:
        args = namespace["parser"].parse_args(flags)
        assert args.goal_poss_dist is expected
        with (patch.object(wrapper, "resolve_feature_run_id", return_value="features"),
              patch.object(wrapper, "resolve_training_split", return_value=({"train": ["m0", "m1", "m2", "m3", "m4"], "test": []}, "test")),
              patch.object(wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original"]),
              patch.object(wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"])):
            args = wrapper.parse_args(["--feature-run-id", "features", "--success-intent-only", *flags])
        resolved = wrapper.resolve_wrapper_feature_flags(args)
        assert resolved["goal_poss_dist"] is expected
        assert ("--goal-poss-dist" in wrapper.append_low_level_feature_flags([], resolved)) is expected
    for saved, expected in [({}, True), ({"goal_poss_dist": False}, False), ({"goal_poss_dist": True}, True)]:
        namespace["args"] = argparse.Namespace(**saved)
        exec(compile(ast.Module(body=[restoration], type_ignores=[]), "train.py", "exec"), namespace)
        assert namespace["args"].goal_poss_dist is expected
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--resume-id", "action_intent/run", "--goal-poss-dist"])


@pytest.mark.parametrize("value", [None, True, False])
def test_refit_preserves_source_setting(sources, tmp_path, value):
    create, _ = sources
    directory, settings, _ = create("outcome_scoring/geometry")
    if value is not None:
        settings["goal_poss_dist"] = value
        (directory / "args.json").write_text(json.dumps(settings))
    payload = prepare_refit("outcome_scoring/geometry", "new")
    assert payload["args"]["goal_poss_dist"] is (value is not False)
    path = tmp_path / "refit.json"
    path.write_text(json.dumps(payload))
    assert load_refit_config(path)["goal_poss_dist"] is (value is not False)
    assert json.loads((directory / "args.json").read_text()) == settings


def test_prepared_cache_separates_settings(artifacts, tmp_path):
    cache = tmp_path / "cache"
    disabled = PreparedActionDataset(["m"], cache_dir=cache, progress=False, **artifacts)
    enabled = PreparedActionDataset(["m"], cache_dir=cache, progress=False,
                                    goal_poss_dist=True, **artifacts)
    assert disabled.cache_id != enabled.cache_id
    torch.testing.assert_close(next(iter(disabled))[0].x, expected_mask(next(iter(enabled))[0].x))
    assert any(path.name == "node_feature_config.py" for path in
               preparation_code_paths())


@pytest.mark.parametrize("flags,expected", [([], False), (["--goal-poss-dist"], True)])
def test_actual_training_saves_and_resumes_setting(synthetic_training, monkeypatch, flags, expected):
    run, _, epoch, utils, root = synthetic_training

    def interrupted(args, *a, **kw):
        if args._epoch == 2:
            raise RuntimeError("interrupt before second epoch")
        return epoch(args, *a, **kw)

    monkeypatch.setattr(utils, "run_epoch", interrupted)
    with pytest.raises(RuntimeError, match="interrupt before second epoch"):
        run("geometry", [*flags, "--max-epochs", "2"])
    directory = root / "saved" / "pass_intent" / "geometry"
    assert json.loads((directory / "args.json").read_text())["goal_poss_dist"] is expected
    monkeypatch.setattr(utils, "run_epoch", epoch)
    restored = run("geometry", resume=True)
    assert restored["args"]["goal_poss_dist"] is expected
    assert restored["metadata"]["goal_poss_dist"] is expected
    assert restored["metadata"]["last_epoch"] == 2
