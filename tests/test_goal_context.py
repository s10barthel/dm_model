import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from datatools import config
from datatools.utils import filter_features_and_labels
from dataset import ActionDataset
from models.goal_context import (GOAL_CONTEXT_TASKS, candidate_mask, goal_policy,
                                 resolve_training_goal_settings)
from models.node_selection import selection_layout, selection_loss_metrics
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from inference import inference_gnn


def settings(task="pass_intent", nodes=True, include=None, version=2):
    return dict(task=task, goal_context_version=version, goal_nodes_aware=nodes,
                include_goals=include, xy_only=False, possessor_aware=True,
                keeper_aware=True, ball_z_aware=True, poss_vel_aware=True,
                poss_rel_vel_aware=True, poss_geometry_aware=True,
                goal_features_aware=False, vel_node_features_aware=True,
                accel_aware=True, offside_aware=False, extend_features=False,
                sparsify="none", edge_in_dim=2, node_in_dim=26,
                include_out=False, lane_survival=False, use_physical_xpass=False)


def source(prefix="home"):
    other = "away" if prefix == "home" else "home"
    x = torch.zeros(6, 26)
    x[:, 0] = torch.tensor([1, 0, 1, 1, 0, 1])
    x[[2, 4], 2] = 1
    x[0, 13] = 1
    x[:, 3] = torch.tensor([10, 25, 105, 30, 0, 50])
    x[:, 4] = 34
    edges = torch.cartesian_prod(torch.arange(6), torch.arange(6)).T
    graph = Data(x=x, edge_index=edges, edge_attr=torch.ones(36, 2))
    graph.node_ids = [prefix+"_1", other+"_2", prefix+"_goal", prefix+"_3", other+"_goal", prefix+"_4"]
    label = torch.zeros(len(config.LABEL_COLUMNS))
    label[1] = 1
    label[4] = 6
    label[5:7] = 3
    label[7] = 1
    label[config.LABEL_INDEX["success"]] = 1
    return graph, label


@pytest.mark.parametrize("task", sorted(GOAL_CONTEXT_TASKS))
@pytest.mark.parametrize("nodes", [True, False])
def test_dataset_and_inference_context_agree(tmp_path, task, nodes):
    args = settings(task, nodes, include=nodes if task == "action_intent" else False)
    graph, label = source()
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); labels.mkdir()
    torch.save([graph], features / "match.pt")
    torch.save(label[None], labels / "match.pt")
    kwargs = build_action_dataset_kwargs(args, train=False, diagnostic_label_dir=None,
                                        require_goal_next10_diagnostics=False)
    dataset = ActionDataset(["match"], feature_dir=features, label_dir=labels, **kwargs)
    runtime, runtime_labels = filter_features_and_labels([graph], label[None], args)
    assert len(dataset) == 1
    torch.testing.assert_close(dataset.features[0].x, runtime[0].x)
    torch.testing.assert_close(dataset.labels, runtime_labels)
    assert runtime[0].num_nodes == (6 if nodes else 4)
    assert runtime[0].edge_index.shape[1] == (36 if nodes else 16)
    assert runtime[0].node_ids[int(runtime_labels[0, 5])] == "home_3"
    assert not runtime[0].x[:, 9:12].any()


@pytest.mark.parametrize("nodes,include,valid", [(True, True, True), (True, False, True),
                                                (False, False, True), (False, True, False)])
def test_action_combinations(nodes, include, valid):
    if valid:
        p = goal_policy(settings("action_intent", nodes, include))
        assert p.input_goals == nodes and p.include_goals == include
    else:
        with pytest.raises(ValueError, match="requires goal_nodes"):
            goal_policy(settings("action_intent", nodes, include))


@pytest.mark.parametrize("task", sorted(GOAL_CONTEXT_TASKS - {"action_intent"}))
def test_predictions_fixed_false(task):
    with pytest.raises(ValueError, match="requires include_goals=False"):
        goal_policy(settings(task, include=True))


def test_training_defaults_and_resume():
    new = SimpleNamespace(task="pass_intent", goal_nodes_aware=None, include_goals=None)
    assert resolve_training_goal_settings(new).input_goals
    old = dict(task="pass_intent", goal_nodes_aware=True)
    resumed = SimpleNamespace(goal_nodes_aware=None, include_goals=None)
    assert not resolve_training_goal_settings(resumed, old).input_goals
    assert resumed.goal_context_version == 1
    with pytest.raises(ValueError, match="Cannot change"):
        resolve_training_goal_settings(SimpleNamespace(goal_nodes_aware=False, include_goals=None), old)
    assert goal_policy(json.loads(json.dumps(vars(new)))) == goal_policy(new)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("override,conflict", [(None, False), (True, False), (False, True)])
def test_wrapper_resume_goal_overrides(monkeypatch, tmp_path, version, override, conflict):
    from scripts import train_relevant_models as wrapper
    flags = ["--resume-id", "action_intent/test"]
    if override is not None:
        flags.append("--action-intent-include-goals" if override else "--no-action-intent-include-goals")
    parsed = wrapper.parse_args(flags)
    assert parsed.action_intent_include_goals is override
    assert parsed.goal_nodes_aware is None
    checkpoint = dict(args=settings("action_intent", include=True, version=version), finished=True)
    monkeypatch.setattr(wrapper, "parse_args", lambda: parsed)
    monkeypatch.setattr(wrapper, "resolve_resume_checkpoint", lambda *a: tmp_path / "checkpoint.pt")
    monkeypatch.setattr(wrapper, "load_checkpoint", lambda *a: checkpoint)
    published = []
    monkeypatch.setattr(wrapper, "publish_checkpoint_artifacts", lambda *a: published.append(True))
    if conflict:
        with pytest.raises(ValueError, match="Cannot change include_goals"):
            wrapper.main()
        assert not published
    else:
        wrapper.main()
        assert published


def test_wrapper_goal_flags_are_mutually_exclusive():
    from scripts import train_relevant_models as wrapper
    for flags in [("--goal-nodes-aware", "--no-goal-nodes"),
                  ("--action-intent-include-goals", "--no-action-intent-include-goals")]:
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--resume-id", "action_intent/test", *flags])


@pytest.mark.parametrize("include", [None, True, False])
def test_wrapper_forwards_candidates_only_to_action(monkeypatch, tmp_path, include):
    from scripts import train_relevant_models as wrapper
    args = wrapper.parse_args(["--resume-id", "action_intent/test"])
    args.enabled_tasks = {"action_intent": True, "pass_intent": True}
    args.intended_receiver_mode = "original"
    args.target_family = None
    args.return_type = "disc_0.9"
    args.feature_run_id = "fixture"
    args.action_intent_include_goals = include
    monkeypatch.setattr(wrapper, "resolve_feature_run_id", lambda *a, **kw: "fixture")
    monkeypatch.setattr(wrapper, "resolve_feature_root", lambda *a: tmp_path)
    commands, *_ = wrapper.build_training_commands(args)
    for command in commands:
        task = wrapper.get_cli_value(command, "--task")
        assert ("--include-goals" in command) == (task == "action_intent" and include is not False)
        assert ("--no-include-goals" in command) == (task == "action_intent" and include is False)


def test_shot_filtering_and_missing_source(tmp_path):
    graph, label = source()
    shot = label.clone(); shot[1] = 0; shot[3] = 1; shot[5:7] = 2; shot[0] = 1
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); labels.mkdir()
    torch.save([graph, graph], features / "m.pt")
    torch.save(torch.stack([label, shot]), labels / "m.pt")
    args = settings("action_intent", include=False)
    kwargs = build_action_dataset_kwargs(args, train=True, diagnostic_label_dir=None)
    ds = ActionDataset(["m"], feature_dir=features, label_dir=labels, **kwargs)
    assert len(ds) == 1 and ds.skipped_rows["shot_candidate_disabled"] == 1
    with pytest.raises(ValueError, match="No usable"):
        filter_features_and_labels([graph], shot[None], args)
    broken = graph.clone(); broken.x[:, 2] = 0
    with pytest.raises(ValueError, match="source graph"):
        filter_features_and_labels([broken], label[None], args)


def test_noncontiguous_candidates_and_gradients():
    graph, label = source()
    second = graph.clone()
    second.x = second.x[:-1]; second.edge_index = torch.empty(2, 0, dtype=torch.long)
    second.edge_attr = torch.empty(0, 2)
    batch = Batch.from_data_list([graph, second])
    labels = torch.stack([label, label]); labels[1, 4] = 5
    args = settings()
    layout = selection_layout(batch, labels, "pass_intent", False, args)
    logits = torch.zeros(batch.num_nodes, requires_grad=True)
    loss, _, _, _, probabilities = selection_loss_metrics(logits, layout)
    torch.testing.assert_close(probabilities, torch.tensor([1/3, 1/2]))
    loss.backward()
    assert logits.grad[2] == 0 and logits.grad[8] == 0
    labels[0, 5] = 2
    with pytest.raises(ValueError, match="not an eligible"):
        selection_layout(batch, labels, "pass_intent", False, args)


class DummyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__(); self.args = args
        self.weight = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, graphs, dests=None):
        logits = graphs.x[:, 3] / 100 + self.weight
        if self.args["task"].startswith("outcome_"):
            return torch.stack([logits, logits], dim=1)
        return logits


@pytest.mark.parametrize("task", sorted(GOAL_CONTEXT_TASKS))
@pytest.mark.parametrize("prefix", ["home", "away"])
@pytest.mark.parametrize("nodes", [True, False])
def test_exports_match_actual_candidates(task, prefix, nodes):
    args = settings(task, nodes, nodes if task == "action_intent" else False)
    graph, label = source(prefix)
    tracking = pd.DataFrame({f"{pid}_{axis}": [float(graph.x[i, 3+j])]
                             for i, pid in enumerate(graph.node_ids) for j, axis in enumerate(["x", "y"])})
    match = SimpleNamespace(tracking=tracking, actions=pd.DataFrame([dict(frame_id=0, object_id=prefix+"_1")]))
    probs, _ = inference_gnn(match, DummyModel(args), device="cpu", graph_override=[graph], label_override=label[None])
    assert (prefix+"_goal" in probs.columns) == (task == "action_intent" and nodes)
    if task in {"pass_intent", "success_intent", "action_intent"}:
        assert float(probs.iloc[0].sum()) == pytest.approx(1)
    if task in {"pass_intent", "success_intent"}:
        assert prefix+"_1" not in probs.columns
        assert set(probs.columns) == {prefix+"_3", prefix+"_4"}


@pytest.mark.parametrize("aux_version", [1, 2])
@pytest.mark.parametrize("target_version", [1, 2])
def test_ipw_input_policy_belongs_to_dependency(aux_version, target_version):
    target = build_action_dataset_kwargs(settings("pass_success", version=target_version),
                                        train=True, diagnostic_label_dir=None)
    aux = settings("pass_intent", version=aux_version)
    result = build_ipw_dataset_kwargs(target, aux, {}, diagnostic_label_dir=None,
                                     require_goal_next10_diagnostics=False)
    assert result["goal_input_nodes"] == (aux_version == 2)
    assert result["goal_context_version"] == target_version


def test_reference_checkpoints_remain_legacy():
    root = Path(__file__).resolve().parents[1]
    meta = root / "data/visualizations/hawkeye/hawkeye_visualization_20261007T233931_573925_14c351b7/metadata.json"
    if not meta.exists():
        pytest.skip("Local reference checkpoints not available")
    for task, model_id in json.loads(meta.read_text())["model_ids"].items():
        args = json.loads((root / "saved" / model_id / "args.json").read_text())
        policy = goal_policy(args)
        assert policy.version == 1
        assert policy.input_goals == (bool(args["goal_nodes_aware"]) and bool(config.TASK_CONFIG.at[task, "include_goals"]))


@pytest.mark.parametrize("main_version,aux_version", [(1, 2), (2, 1), (2, 2)])
def test_separate_auxiliary_graphs_and_ipw_identities(tmp_path, monkeypatch, main_version, aux_version):
    from ipw_preparation import sample_records, predict_probabilities
    from torch_geometric.loader import DataLoader
    graph, label = source()
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); labels.mkdir()
    torch.save([graph], features / "m.pt")
    torch.save(label[None], labels / "m.pt")
    main_args = settings("pass_success", version=main_version)
    aux_args = settings("pass_intent", version=aux_version)
    kwargs = build_action_dataset_kwargs(main_args, train=False, diagnostic_label_dir=None,
                                        require_goal_next10_diagnostics=False)
    main = ActionDataset(["m"], feature_dir=features, label_dir=labels,
                         **dict(kwargs, auxiliary_model_args=aux_args))
    aux_kwargs = build_ipw_dataset_kwargs(kwargs, aux_args, {}, diagnostic_label_dir=None,
                                         require_goal_next10_diagnostics=False)
    aux = ActionDataset(["m"], feature_dir=features, label_dir=labels, **aux_kwargs)
    assert sample_records(main) == sample_records(aux)
    batch, _, _ = next(iter(DataLoader(main, batch_size=1)))
    assert batch.num_nodes == (6 if main_version == 2 else 4)
    rebuilt = Batch.from_data_list(batch.auxiliary_graph)
    assert rebuilt.num_nodes == (6 if aux_version == 2 else 4)
    assert rebuilt.node_ids[0][int(batch.auxiliary_label[0, 5])] == "home_3"
    # Legacy node-selection historically relies on teammates-first ordering;
    # the v2 path must handle our deliberately interleaved graph.
    if aux_version == 2:
        values = predict_probabilities(aux, DummyModel(aux_args), device="cpu", batch_size=1, pin_memory=False)
        expected = torch.tensor([.1, .3, .5]).softmax(0)[1]
        torch.testing.assert_close(values[0], expected)

    # Exercise the combined evaluator, including a legacy/new policy mismatch.
    from models import utils as model_utils
    from physical_pass_model import (EVALUATION_XPASS_PROB_ATTR,
        EVALUATION_XPASS_DISTANCE_ATTR, EVALUATION_XPASS_PASS_HEIGHT_ATTR)
    for name, value in [(EVALUATION_XPASS_PROB_ATTR, .2),
                        (EVALUATION_XPASS_DISTANCE_ATTR, 30.),
                        (EVALUATION_XPASS_PASS_HEIGHT_ATTR, .8)]:
        setattr(main.features[0], name, torch.full((main.features[0].num_nodes,), value))
    observed_probabilities = []
    original_blend = model_utils.blend_physical_xpass_predictions
    def capture_blend(**kwargs):
        observed_probabilities.extend(kwargs["pass_intent"])
        return original_blend(**kwargs)
    monkeypatch.setattr(model_utils, "blend_physical_xpass_predictions", capture_blend)
    epoch_args = SimpleNamespace(**dict(main_args, gnn_task="node_binary", lambda_l1=0.,
        residual_regularization_lambda=0., model_variant="gat_baseline", use_xg=False,
        use_xt=False, use_goal_distance=False, use_epv=False, print_freq=99, clip=10,
        evaluate_combined_success=True, return_pass_success_height_evaluation=True,
        xpass_weight="v5", xpass_metric="top10_xpass", v5_intent_threshold=.01,
        v5_discount=True, classification_threshold=.5))
    model_utils.run_epoch(epoch_args, DummyModel(main_args), DataLoader(main, batch_size=1),
                          device="cpu", train=False, pass_intent_model=DummyModel(aux_args))
    assert observed_probabilities == pytest.approx([torch.tensor([.3, .5]).softmax(0)[0].item()])


def test_sidecars_need_no_goal_columns():
    from physical_pass_model import (attach_physical_xpass_to_graph,
        append_pc_xpass_lane_survival_to_graph, pc_xpass_lane_survival_column_for_mode,
        PHYSICAL_XPASS_LOGIT_ATTR)
    graph, label = source()
    row = {pid: .6 for pid in graph.node_ids if not pid.endswith("_goal")}
    rows = pd.DataFrame([row])
    attached = attach_physical_xpass_to_graph(graph.clone(), label, rows, match_id="m")
    assert torch.isfinite(getattr(attached, PHYSICAL_XPASS_LOGIT_ATTR)).all()
    assert torch.equal(getattr(attached, PHYSICAL_XPASS_LOGIT_ATTR)[[2, 4]], torch.zeros(2))
    row.update({pc_xpass_lane_survival_column_for_mode(pid, "max"): .7 for pid in ["home_3", "home_4"]})
    lane = append_pc_xpass_lane_survival_to_graph(graph.clone(), label, pd.DataFrame([row]),
                                                 match_id="m", possessor_index=0, mode="max")
    assert torch.equal(lane.x[[2, 4], -1], torch.zeros(2))


def test_cache_identity_changes_with_policy(tmp_path):
    from prepared_dataset import PreparedActionDataset
    graph, label = source()
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); labels.mkdir()
    torch.save([graph], features / "m.pt")
    torch.save(label[None], labels / "m.pt")
    cache_ids = []
    for version in [1, 2]:
        kwargs = build_action_dataset_kwargs(settings(version=version), train=True, diagnostic_label_dir=None)
        dataset = PreparedActionDataset(["m"], feature_dir=features, label_dir=labels,
                                        cache_dir=tmp_path / "cache", **kwargs)
        cache_ids.append(dataset.cache_id)
    assert cache_ids[0] != cache_ids[1]
