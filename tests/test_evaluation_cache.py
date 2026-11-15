"""Evaluation cache parity, compatibility, process lifetime and summary schema."""
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

import test as evaluation
import prepared_dataset as prepared
from cache_lifetime import evaluation_context, workflow_context
from datatools.config import LABEL_INDEX
from models import prediction_identity
from project_config import split_dataset_provenance
from scripts import evaluate_relevant_models as wrapper
from test_prepared_dataset import artifacts, assert_sample_equal


@pytest.fixture
def outcome_artifacts(artifacts):
    for path in artifacts["feature_dir"].glob("*.pt"):
        graphs = torch.load(path, weights_only=False)
        for graph in graphs:
            if graph is not None:
                graph.node_ids = [f"home_{i}" for i in range(graph.num_nodes)]
        torch.save(graphs, path)
        label_path = artifacts["label_dir"] / path.name
        labels = torch.load(label_path, weights_only=False)
        labels[2, LABEL_INDEX["is_shot"]] = 0
        labels[2, LABEL_INDEX["is_dribble"]] = 1
        labels[:, LABEL_INDEX["success"]] = torch.arange(len(labels)) % 2
        for name in ("scores_xt", "concedes_xt"):
            labels[:, LABEL_INDEX[name]] = torch.linspace(0, 1, len(labels))
        for name in ("scores_goal_next10", "concedes_goal_next10"):
            labels[:, LABEL_INDEX[name]] = torch.arange(len(labels)) % 2
        torch.save(labels, label_path)
    return dict(artifacts, task="outcome_scoring", train=False,
                require_goal_next10_diagnostics=False)


def cli(tmp_path, mode="auto", **extra):
    return SimpleNamespace(dataset_loading=mode, dataset_cache_dir=str(tmp_path / "cache"),
                           dataset_buffer_matches=4, dataset_preparation_workers=1,
                           dataset_worker_threads=1, keep_cache=True, comparison_rows=True,
                           export_outcome_distributions=False, export_outcome_decomposition=False,
                           evaluation_output_dir=None, **extra)


def build(args, options, model_args=None, ids=("m1", "m0"), session=None):
    options = dict(options)
    features, labels = options.pop("feature_dir"), options.pop("label_dir")
    model_args = model_args or SimpleNamespace(task=options["task"], gnn_task="node_binary")
    return evaluation.build_test_dataset(
        args, model_args, ids, feature_dir=features, label_dir=labels,
        feature_root=features.parent, dataset_args=options, cache_session=session)[0]


def test_summary_handles_empty_and_all_na_columns_without_future_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "EVALUATION_RUNS_DIR", tmp_path)
    pd.DataFrame(columns=["old_empty_column", "evaluation_timestamp", "model_id"]).to_csv(
        tmp_path / "metrics_summary.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for model, config, value, extra in (
            ("a", "unmasked", .2, None), ("b", "unmasked", .4, 3.),
            ("a", "unmasked", .3, None), ("a", "masked", .5, None),
        ):
            evaluation.write_model_evaluation_artifacts(
                tmp_path / model / config, model_id=f"outcome_scoring/{model}", task="outcome_scoring",
                feature_run_id=None, diagnostic_feature_run_id=None, evaluation_timestamp="now",
                evaluation_options={"configuration_id": config, "optional_flag": True if extra else None},
                test_metrics={"ce_loss": value, "all_missing": None, "optional_metric": extra})
    summary = pd.read_csv(tmp_path / "metrics_summary.csv")
    assert len(summary) == 3
    assert summary.old_empty_column.isna().all() and summary.all_missing.isna().all()
    assert summary.iloc[0].optional_metric == 3.
    assert summary.loc[(summary.model_id == "outcome_scoring/a") &
                       (summary.configuration_id == "unmasked"), "ce_loss"].item() == .3
    assert list(summary.columns[:3]) == ["old_empty_column", "evaluation_timestamp", "model_id"]


def test_eight_outcome_checkpoints_share_preparation_and_preserve_batches(tmp_path, outcome_artifacts, monkeypatch):
    args = cli(tmp_path)
    modes = [("decomposition", "distributional"), ("scalar", None),
             ("decomposition", "scalar"), ("distributional", None)]
    reference = None
    for task in ("outcome_scoring", "outcome_conceding"):
        options = dict(outcome_artifacts, task=task, evaluation_filter="pass")
        memory = build(cli(tmp_path, "memory"), options)
        for mode, severity in modes:
            model = SimpleNamespace(task=task, gnn_task="node_binary", learning_mode=mode, severity_mode=severity)
            if reference is not None:
                monkeypatch.setattr(prepared.PreparedActionDataset, "_build_entry",
                                    lambda *a: pytest.fail("compatible checkpoint rebuilt its dataset"))
            disk = build(args, options, model)
            assert disk.selection_counts == memory.selection_counts
            for field in ("enabled", "checked_files", "checked_graphs"):
                assert disk.graph_validation_stats[field] == memory.graph_validation_stats[field]
            assert split_dataset_provenance(["m1", "m0"], disk) == split_dataset_provenance(["m1", "m0"], memory)
            memory_batches = list(DataLoader(memory, batch_size=3, shuffle=False))
            disk_batches = list(DataLoader(disk, batch_size=3))
            assert len(memory_batches) == len(disk_batches)
            for a, b in zip(memory_batches, disk_batches):
                assert_sample_equal(a, b)
            first_graph = next(iter(disk))[0]
            evaluation.validate_test_dataset_dimensions(disk, SimpleNamespace(
                node_in_dim=first_graph.x.shape[1], edge_in_dim=first_graph.edge_attr.shape[1]))
            with pytest.raises(ValueError, match="node feature width"):
                evaluation.validate_test_dataset_dimensions(disk, SimpleNamespace(node_in_dim=1000))
            if reference is None:
                reference = disk
                assert disk.preparation["built"] == 2
            else:
                assert disk.cache_id == reference.cache_id and disk.preparation["reused"] == 2
                assert disk.graph_validation_stats == reference.graph_validation_stats


def test_filter_labels_diagnostics_and_feature_options_invalidate_cache(tmp_path, outcome_artifacts):
    args = cli(tmp_path)
    original = build(args, outcome_artifacts)
    passes = build(args, dict(outcome_artifacts, evaluation_filter="pass"))
    assert passes.cache_id != original.cache_id and len(passes) < len(original)
    changed_features = build(args, dict(outcome_artifacts, vel_node_features_aware=True))
    assert changed_features.cache_id != original.cache_id
    labels = outcome_artifacts["label_dir"] / "m0.pt"
    values = torch.load(labels, weights_only=False)
    values[:, LABEL_INDEX["scores_xt"]] = .25
    torch.save(values, labels)
    changed_labels = build(args, outcome_artifacts)
    assert changed_labels.preparation["built"] == 1 and changed_labels.preparation["reused"] == 1
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    options = dict(outcome_artifacts, diagnostic_label_dir=str(diagnostics))
    first = build(args, options)
    torch.save(values, diagnostics / "m0.pt")
    second = build(args, options)
    assert second.preparation["built"] == 1 and second.preparation["reused"] == 1


def test_cached_legacy_node_ids_track_recovery_sources(tmp_path, outcome_artifacts, monkeypatch):
    for path in outcome_artifacts["feature_dir"].glob("*.pt"):
        graphs = torch.load(path, weights_only=False)
        for graph in graphs:
            if graph is not None:
                del graph.node_ids
        torch.save(graphs, path)
    tracking = tmp_path / "tracking"
    tracking.mkdir()
    monkeypatch.setattr(prediction_identity, "TRACKING_PROCESSED_DIR", tracking)
    calls = []
    def recover(graph, *a, **kw):
        calls.append(graph)
        return [f"player_{i}" for i in range(graph.num_nodes)]
    monkeypatch.setattr(prediction_identity, "resolve_legacy_node_ids", recover)
    for match in ("m0", "m1"):
        paths = prediction_identity.legacy_identity_paths(match, tmp_path, SimpleNamespace())
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"value": [1]}).to_parquet(path)
    original = build(cli(tmp_path), outcome_artifacts)
    assert calls
    calls.clear()
    reused = build(cli(tmp_path), outcome_artifacts)
    assert not calls and reused.preparation["reused"] == 2
    assert next(iter(reused))[0].node_ids == next(iter(original))[0].node_ids
    pd.DataFrame({"value": [2, 3]}).to_parquet(tracking / "m0.parquet")
    changed = build(cli(tmp_path), outcome_artifacts)
    assert calls and changed.preparation["built"] == 1 and changed.preparation["reused"] == 1


@pytest.mark.parametrize("keep", [False, True])
def test_evaluation_children_share_owner_and_retention(tmp_path, outcome_artifacts, keep):
    args = cli(tmp_path)
    args.keep_cache = keep
    with workflow_context(args, kind="evaluation_wrapper") as owner:
        with evaluation_context(args) as session:
            first = build(args, outcome_artifacts, session=session)
        assert all(Path(e["path"]).is_file() for e in first.entries)
        with evaluation_context(args) as session:
            second = build(args, dict(outcome_artifacts, task="outcome_conceding"), session=session)
        assert second.preparation["reused"] == 2
    assert all(Path(e["path"]).is_file() == keep for e in first.entries)
    assert owner.report["status"] == "completed"


def test_direct_evaluation_failure_preserves_cache(tmp_path, outcome_artifacts):
    args = cli(tmp_path)
    args.keep_cache = False
    with pytest.raises(RuntimeError, match="interrupted"):
        with evaluation_context(args) as session:
            dataset = build(args, outcome_artifacts, session=session)
            raise RuntimeError("interrupted")
    assert all(Path(e["path"]).is_file() for e in dataset.entries)


def test_cache_reused_by_separate_evaluation_processes(tmp_path, outcome_artifacts):
    code = '''import json,sys
from types import SimpleNamespace
from cache_lifetime import evaluation_context
from prepared_dataset import PreparedActionDataset
args=SimpleNamespace(dataset_cache_dir=sys.argv[1],keep_cache=True,evaluation_output_dir=None)
with evaluation_context(args) as session:
 dataset=PreparedActionDataset(['m0'],cache_dir=args.dataset_cache_dir,cache_session=session,
     progress=False,**json.loads(sys.argv[2]))
 print(json.dumps(dict(built=dataset.preparation['built'],reused=dataset.preparation['reused'])))
'''
    options = {k: str(v) if isinstance(v, Path) else v for k, v in outcome_artifacts.items()}
    results = []
    for task in ("outcome_scoring", "outcome_conceding"):
        options["task"] = task
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path / "cache"), json.dumps(options)],
                               capture_output=True, text=True, timeout=60, check=True)
        results.append(json.loads(child.stdout.strip().splitlines()[-1]))
    assert results == [{"built": 1, "reused": 0}, {"built": 0, "reused": 1}]


def test_wrapper_forwards_dataset_options_to_each_task(tmp_path):
    args = wrapper.parse_args(["--dataset-loading", "disk", "--dataset-cache-dir", str(tmp_path),
                               "--dataset-preparation-workers", "2", "--keep-cache"])
    for task in ("outcome_scoring", "outcome_conceding"):
        command = wrapper.add_task_evaluation_options(["test.py"], args, task)
        assert command[command.index("--dataset-loading") + 1] == "disk"
        assert command[command.index("--dataset-cache-dir") + 1] == str(tmp_path)
        assert command[command.index("--dataset-preparation-workers") + 1] == "2"
        assert "--keep-cache" in command


@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
@pytest.mark.parametrize("mode,severity", [("scalar", None), ("distributional", None),
                                         ("decomposition", "scalar"), ("decomposition", "distributional")])
def test_real_inference_and_comparison_rows_match_memory(tmp_path, outcome_artifacts, task, mode, severity):
    import numpy as np
    from models.gnn import GNN
    from models.utils import run_epoch
    from test_masked_evaluation_smoke import model_args
    from test_outcome_decomposition import configuration as decomposition
    from test_outcome_learning import configuration as distributional
    options = dict(outcome_artifacts, task=task, evaluation_filter="pass")
    memory, disk = (build(cli(tmp_path, loading), options) for loading in ("memory", "disk"))
    graph = memory.features[0]
    args = model_args(task)
    args.update(node_in_dim=graph.x.shape[1], edge_in_dim=graph.edge_attr.shape[1],
                use_xt=True, use_goal_distance=False, learning_mode=mode)
    if mode == "decomposition":
        args.update(decomposition(severity, task))
    elif mode == "distributional":
        args.update(distributional(task))
    torch.manual_seed(7)
    model = GNN(args).eval()
    results = [run_epoch(SimpleNamespace(**args), model, DataLoader(dataset, batch_size=3),
                         device="cpu", return_learning_curve_rows=True, return_semantic_prediction_rows=True)
               for dataset in (memory, disk)]
    for memory_result, disk_result in zip(*results):
        np.testing.assert_equal(memory_result, disk_result)


def test_direct_cli_writes_dataset_metadata_and_finishes_cache(tmp_path, outcome_artifacts, monkeypatch):
    import numpy as np
    from models.gnn import GNN
    from test_masked_evaluation_smoke import model_args
    options = dict(outcome_artifacts)
    sample = build(cli(tmp_path, "memory"), options).features[0]
    saved = model_args("outcome_scoring")
    saved.update({k: v for k, v in options.items() if k not in {"feature_dir", "label_dir", "train"}})
    saved.update(node_in_dim=sample.x.shape[1], edge_in_dim=sample.edge_attr.shape[1],
                 use_xt=True, return_type="time_15", batch_size=3)
    model = GNN(saved).eval()
    monkeypatch.setattr(evaluation.utils, "load_model", lambda *a: model)
    monkeypatch.setattr(evaluation.utils, "get_model_record", lambda *a: {})
    context = dict(feature_run_id="fixture", feature_root=str(tmp_path),
                   feature_dir=str(options["feature_dir"]), label_dir=str(options["label_dir"]),
                   diagnostic_feature_run_id=None, diagnostic_label_dir=None,
                   pass_height_diagnostic_feature_run_id=None, pass_height_diagnostic_label_dir=None,
                   observed_pass_height_threshold_meters=None, pass_height_diagnostic_lineage=None,
                   evaluation_diagnostic_label_dir=None, feature_schema={}, model_schema={},
                   training_feature_run_id="fixture", requested_feature_run_id=None)
    monkeypatch.setattr(evaluation, "resolve_evaluation_data", lambda *a: dict(context))
    monkeypatch.setattr(evaluation, "prepare_evaluation_population", lambda *a, **kw: (
        {"manifest_id": "fixture"}, "fixture", np.asarray(["m0"]), {"checks": [], "status": "verified"}))
    monkeypatch.setattr(evaluation, "EVALUATION_RUNS_DIR", tmp_path / "summary")
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["test.py", "--model_id", "outcome_scoring/fixture", "--device", "cpu",
        "--evaluation-output-dir", str(output), "--dataset-cache-dir", str(tmp_path / "cache"),
        "--comparison-rows", "--outcome-bootstrap-resamples", "0"])
    evaluation.main()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["dataset_loading"]["resolved"] == "disk"
    assert metadata["dataset_loading"]["test"]["preparation"]["built"] == 1
    assert metadata["evaluation_context"]["selection_counts"]["evaluated_samples"] > 0
    assert metadata["cache_lifetime"]["status"] == "completed"
    assert metadata["cache_lifetime"]["roots"][0]["deleted_entries"] == 1
    assert not list((tmp_path / "cache").rglob("*.pt"))
    assert (output / "learning_curve_predictions.csv").is_file()


def test_auxiliary_graph_evaluation_keeps_memory_path(tmp_path, outcome_artifacts, monkeypatch):
    args = cli(tmp_path)
    args.comparison_rows = False
    memory = object()
    captured = {}
    def make_memory(*a, **kw):
        captured.update(kw)
        return memory
    monkeypatch.setattr(evaluation, "ActionDataset", make_memory)
    options = dict(outcome_artifacts, auxiliary_models_args={"pass_height": {"task": "pass_height"}})
    assert build(args, options) is memory
    assert captured["auxiliary_models_args"] == options["auxiliary_models_args"]
    args.dataset_loading = "disk"
    with pytest.raises(ValueError, match="nested xPass auxiliary graphs"):
        build(args, options)
