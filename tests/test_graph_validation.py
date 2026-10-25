"""Regression coverage for the six-index corruption and publication boundary."""
import argparse
import copy
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from datatools.artifact_storage import atomic_graph_save
from datatools import graph_validation as validation
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset
from prepared_dataset import PreparedActionDataset


def graph(n=24):
    x = torch.zeros((n, 25))
    x[:, 0] = 1
    x[0, 13] = 1
    x[:, 3:5] = torch.randn(n, 2, generator=torch.Generator().manual_seed(17)) * 10 + 30
    edges = torch.cartesian_prod(torch.arange(n), torch.arange(n)).T.contiguous()
    return Data(x=x, edge_index=edges, edge_attr=torch.ones((n*n, 2)),
                node_ids=[f"home_{i}" for i in range(n)])


def corrupt(g):
    g = g.clone()
    g.edge_index[1, [132, 133, 194, 195, 574, 575]] = 0
    return g


def labels():
    value = torch.zeros((1, len(LABEL_COLUMNS)))
    for name, v in dict(action_index=469, is_pass=1, success=1, is_real=1,
                        intent_index=1, receiver_index=1, duration=1).items():
        value[0, LABEL_INDEX[name]] = v
    return value


def test_cli_default_on_off_and_invalid():
    parser = argparse.ArgumentParser()
    validation.add_graph_validation_argument(parser)
    assert parser.parse_args([]).graph_validation is True
    assert parser.parse_args(["--graph-validation", "on"]).graph_validation is True
    off = parser.parse_args(["--graph-validation", "off"])
    assert off.graph_validation is False
    assert validation.graph_validation_flags(off) == ["--graph-validation", "off"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--graph-validation", "false"])


def test_exact_six_zeroes_are_rejected_with_identity():
    g = corrupt(graph())
    assert g.num_edges == 576
    assert torch.unique(g.edge_index.T, dim=0).shape[0] == 570
    with pytest.raises(validation.GraphValidationError, match=r"run/match.*source row 0.*action index 469.*missing pairs"):
        validation.validate_graphs([g], context="run/match", labels=labels())
    validation.validate_graphs([g], context="off", graph_validation=False)


@pytest.mark.parametrize("fault", ["dtype", "shape", "bounds", "attrs", "schema", "ids", "empty_nodes"])
def test_structural_failures(fault):
    g = graph()
    if fault == "dtype": g.edge_index = g.edge_index.float()
    elif fault == "shape": g.edge_index = g.edge_index.T
    elif fault == "bounds": g.edge_index[0, 0] = 24
    elif fault == "attrs": g.edge_attr = g.edge_attr[:-1]
    elif fault == "schema": g.edge_feature_names = ["distance"]
    elif fault == "ids": g.node_ids = ["a"]
    else: g.x = g.x[:0]
    with pytest.raises(validation.GraphValidationError):
        validation.validate_graphs([g], context=fault)


def test_permutation_legacy_empty_and_sparse_contracts():
    g = graph()
    permutation = torch.randperm(g.num_edges)
    g.edge_index, g.edge_attr = g.edge_index[:, permutation], g.edge_attr[permutation]
    del g.node_ids
    validation.validate_graphs([None, g], context="legacy")
    validation.validate_graphs([], context="empty")
    sparse = g.clone()
    sparse.edge_index, sparse.edge_attr = sparse.edge_index[:, :5], sparse.edge_attr[:5]
    validation.validate_graphs([sparse], context="sparse", dense=False)
    with pytest.raises(validation.GraphValidationError):
        validation.validate_graphs([sparse], context="raw")


def test_snapshots_do_not_alias_and_named_extensions_preserve_columns():
    g = graph()
    before = validation.graph_fingerprints([g])
    g.edge_attr = torch.cat([g.edge_attr, torch.zeros((g.num_edges, 1))], dim=1)
    g.edge_feature_names = ["distance", "is_teammate", "relative_speed"]
    validation.assert_preserved(before, [g], context="extension", allow_edge_extension=True)
    g.edge_attr[0, 0] = 99
    with pytest.raises(validation.GraphValidationError, match="retained graph"):
        validation.assert_preserved(before, [g], context="extension", allow_edge_extension=True)
    validation.assert_preserved(before, [None], context="unavailable", allow_availability=True)


def test_off_does_not_inspect_fingerprint_or_read_back(tmp_path):
    p = tmp_path / "off.pt"
    with patch.object(validation, "_tensor_digest", side_effect=AssertionError("fingerprinted")), \
         patch("torch.load", side_effect=AssertionError("read back")):
        atomic_graph_save([corrupt(graph())], p, graph_validation=False)
    assert p.exists()


def test_corrupt_staged_output_is_never_published(tmp_path, monkeypatch):
    from datatools import artifact_storage
    target = tmp_path / "match.pt"
    target.write_bytes(b"original")
    original_save = torch.save
    def damaged_save(value, stream):
        value = copy.deepcopy(value)
        value[0].edge_attr[0, 0] = 77
        original_save(value, stream)
    monkeypatch.setattr(artifact_storage.torch, "save", damaged_save)
    with pytest.raises(validation.GraphValidationError, match="retained graph"):
        atomic_graph_save([graph()], target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_graph_save_breaks_target_hardlink_only(tmp_path):
    import os
    parent, target = tmp_path / "parent.pt", tmp_path / "target.pt"
    torch.save([graph()], parent)
    original = parent.read_bytes()
    os.link(parent, target)
    stats = validation.validation_stats()
    atomic_graph_save([graph()], target, stats=stats)
    assert parent.read_bytes() == original
    assert parent.stat().st_ino != target.stat().st_ino
    assert stats["verified_output_files"] == 1


@pytest.mark.parametrize("sparsify", ["none", "delaunay"])
def test_loading_rejects_raw_corruption_before_filtering(tmp_path, sparsify):
    features, target_labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); target_labels.mkdir()
    torch.save([corrupt(graph())], features / "m.pt")
    torch.save(labels(), target_labels / "m.pt")
    with pytest.raises(validation.GraphValidationError, match="action index 469"):
        ActionDataset(["m"], feature_dir=features, label_dir=target_labels,
                      task="pass_intent", extend_features=False, sparsify=sparsify)
    if sparsify == "none":
        data = ActionDataset(["m"], feature_dir=features, label_dir=target_labels,
                             task="pass_intent", extend_features=False, graph_validation=False)
        assert len(data) == 1
        assert data.graph_validation_stats["checked_files"] == 0


def test_cache_modes_are_separate_and_reused_within_mode(tmp_path):
    features, target_labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir(); target_labels.mkdir()
    torch.save([graph()], features / "m.pt"); torch.save(labels(), target_labels / "m.pt")
    kwargs = dict(feature_dir=features, label_dir=target_labels, task="pass_intent",
                  extend_features=False, cache_dir=tmp_path / "cache", progress=False)
    with patch.object(PreparedActionDataset, "_check_space"):
        on = PreparedActionDataset(["m"], **kwargs)
        off = PreparedActionDataset(["m"], **kwargs, graph_validation=False)
        again = PreparedActionDataset(["m"], **kwargs)
        off_again = PreparedActionDataset(["m"], **kwargs, graph_validation=False)
    assert on.cache_id != off.cache_id
    assert again.preparation["reused"] == off_again.preparation["reused"] == 1
    assert again.preparation["graph_validation"]["enabled"] is True
    assert off_again.preparation["graph_validation"] == validation.validation_stats(False)


@pytest.mark.parametrize("mode,expected", [("on", True), ("off", False)])
def test_generation_and_training_cli_forwarding(monkeypatch, mode, expected):
    import sys
    from scripts import generate_relevant_features as generate
    from scripts import train_relevant_models as train
    from datatools import graph_feature
    monkeypatch.setattr(sys, "argv", ["generate", "--graph-validation", mode])
    args = generate.parse_args()
    assert args.graph_validation is expected
    command = generate.with_mode_flags(["python", "graph_feature.py"], args)
    assert command[command.index("--graph-validation") + 1] == mode
    monkeypatch.setattr(sys, "argv", ["graph_feature", "--graph-validation", mode])
    worker = graph_feature.parse_args()
    assert graph_feature.args_to_worker_dict(worker)["graph_validation"] is expected
    assert train.parse_args(["--refit-source-model-id", "pass_intent/fixture", "--graph-validation", mode]).graph_validation is expected
    assert train.parse_args(["--resume-id", "pass_intent/run", "--graph-validation", mode]).graph_validation is expected


@pytest.mark.parametrize("operation", ["--refresh-failed-pass-trajectories", "--refresh-trajectory",
                                       "--retire-elastic", "--replace-carries"])
def test_dedicated_refresh_allows_operational_flag(monkeypatch, operation):
    import sys
    from scripts import generate_relevant_features as generate
    monkeypatch.setattr(generate, "load_feature_run_metadata", lambda *a, **k: {"train_count": 765})
    monkeypatch.setattr(sys, "argv", ["generate", operation, "--extend-feature-run-id", "base", "--graph-validation", "off"])
    assert generate.parse_args().graph_validation is False


def test_extension_step_forwards_off_to_subprocess(tmp_path, monkeypatch):
    from scripts import generate_relevant_features as generate
    from scripts.feature_extension_diagnostics import ExtensionAttempt
    commands = []
    monkeypatch.setattr(generate, "run_command", lambda command: commands.append(command))
    attempt = ExtensionAttempt(tmp_path, "extension", "derived", {"graph_validation": False})
    generate.run_generation_steps([generate.FeatureGenerationStep("fixture", ["python", "graph_feature.py"])], attempt)
    assert commands[0][-2:] == ["--graph-validation", "off"]


def test_inference_runtime_default_does_not_inherit_saved_off():
    from datatools.utils import filter_features_and_labels
    with pytest.raises(validation.GraphValidationError, match="connectivity"):
        filter_features_and_labels([corrupt(graph())], labels(), {"task": "pass_intent", "graph_validation": False})


def test_delaunay_lookup_failure_has_context_without_validation_scan():
    from scipy.spatial import Delaunay
    from datatools.utils import sparsify_edges
    g = graph()
    pair = Delaunay(g.x[:, 3:5].numpy()).simplices[0, :2]
    absent = (g.edge_index[0] == int(pair[0])) & (g.edge_index[1] == int(pair[1]))
    g.edge_index, g.edge_attr = g.edge_index[:, ~absent], g.edge_attr[~absent]
    with pytest.raises(validation.GraphValidationError, match="match m; source row 7.*missing directed edge"):
        sparsify_edges(g, "delaunay", context="match m; source row 7")


@pytest.mark.parametrize("mode", ["on", "off"])
def test_evaluation_and_pipeline_forwarding(monkeypatch, mode):
    import sys
    from scripts import evaluate_relevant_models as evaluation
    from scripts import main as pipeline
    args = evaluation.parse_args(["--pass-intent-model-id", "pass_intent/fixture", "--graph-validation", mode])
    command = evaluation.add_task_evaluation_options(["python", "test.py"], args, "pass_intent")
    assert command[command.index("--graph-validation") + 1] == mode
    monkeypatch.setattr(sys, "argv", ["main", "--graph-validation", mode, "--only-success-intent", "--return_type", "next_10"])
    args = pipeline.parse_args()
    commands = pipeline.build_commands(args)
    for script in ("scripts/generate_relevant_features.py", "scripts/train_relevant_models.py",
                   "scripts/evaluate_relevant_models.py"):
        command = next(command for command in commands if command[1] == script)
        assert command[command.index("--graph-validation") + 1] == mode


@pytest.mark.parametrize("mutation", ["edges", "attributes", "node_order"])
def test_refresh_detects_mutation_from_valid_input(monkeypatch, mutation):
    from test_trajectory_speed_refresh import stored_graph, trajectory_match
    from datatools import trajectory_refresh as refresh
    match = trajectory_match()
    g = stored_graph(match)
    summarize = refresh.summarize_ball_trajectory
    def mutate(*args):
        if mutation == "edges":
            g.edge_index = g.edge_index.flip(1)  # Still dense, but changed order.
        elif mutation == "attributes":
            g.edge_attr[0, 0] += 1
        else:
            g.node_ids.reverse()
        return summarize(*args)
    monkeypatch.setattr(refresh, "summarize_ball_trajectory", mutate)
    with pytest.raises(validation.GraphValidationError, match="retained graph"):
        refresh.refresh_match_graphs([g], torch.tensor([[17.]]), match, context="refresh match m")


def test_disabled_refresh_skips_scans_and_fingerprints(monkeypatch):
    from test_trajectory_speed_refresh import stored_graph, trajectory_match
    from datatools import trajectory_refresh as refresh
    match = trajectory_match()
    g = stored_graph(match)
    g.edge_index[1, 2] = 0
    stats = validation.validation_stats(False)
    with patch.object(validation, "_tensor_digest", side_effect=AssertionError("fingerprinted")):
        refresh.refresh_match_graphs([g], torch.tensor([[17.]]), match, context="unchecked", graph_validation=False, stats=stats)
    assert stats == validation.validation_stats(False)


def test_named_extension_accepts_reordered_groups_and_checks_old_values():
    from models.edge_schema import EDGE_GROUPS
    g = graph(3)
    g.edge_attr = torch.cat([g.edge_attr, torch.ones((g.num_edges, 2))], dim=1)
    g.edge_feature_names = ["distance", "is_teammate", *EDGE_GROUPS["v"]]
    before = validation.graph_fingerprints([g])
    g.edge_attr = torch.cat([g.edge_attr[:, :2], torch.zeros((g.num_edges, 1)), g.edge_attr[:, 2:]], dim=1)
    g.edge_feature_names = ["distance", "is_teammate", "relative_speed", *EDGE_GROUPS["v"]]
    validation.validate_graphs([g], context="reordered extension")
    validation.assert_preserved(before, [g], context="extension", allow_edge_extension=True)


def test_integer_indices_and_legacy_metadata():
    g = graph(3)
    g.edge_index = g.edge_index.int()
    g.edge_feature_names = ["edge.distance", "edge.is_teammate"]
    g.edge_schema_version = None
    validation.validate_graphs([g], context="legacy int32")


def test_failed_atomic_replacement_cleans_temp_and_retains_original(tmp_path, monkeypatch):
    from pathlib import Path
    target = tmp_path / "match.pt"
    target.write_bytes(b"original")
    replace = Path.replace
    def blocked(source, destination):
        if Path(destination) == target:
            raise PermissionError("blocked publication")
        return replace(source, destination)
    monkeypatch.setattr(Path, "replace", blocked)
    with pytest.raises(PermissionError, match="blocked publication"):
        atomic_graph_save([graph()], target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]
