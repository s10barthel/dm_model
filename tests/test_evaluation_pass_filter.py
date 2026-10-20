"""Pass selection and event alignment across carry-inclusive feature runs."""
import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

import test as evaluation
import project_config
from datatools import config
from dataset import ActionDataset
from models.gnn import GNN
from models import evaluation_context, utils
from models.utils import run_epoch
from scripts import evaluate_relevant_models as wrapper, model_comparison
from scripts.comparison_warnings import ComparisonWarnings
from test_masked_evaluation_smoke import model_args, samples


@pytest.fixture
def action_artifacts(tmp_path):
    features, labels = tmp_path / "graphs", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    source = samples()
    graphs, rows = [], []
    for i, (graph, label, _) in enumerate(source):
        label[config.LABEL_INDEX["action_index"]] = 10 + i
        label[7] = [.05, .1, .15, .2][i]
        label[config.LABEL_INDEX["scores_xt"]] = .02 * (i + 1)
        graphs.append(graph)
        rows.append(label)
    carry = rows[-1].clone()
    for name, value in dict(action_index=100, is_pass=0, is_dribble=1, intent_index=0).items():
        carry[config.LABEL_INDEX[name]] = value
    # Interleave a carry so artifact row positions differ from pass event IDs.
    graphs.insert(1, graphs[-1].clone())
    rows.insert(1, carry)
    torch.save(graphs, features / "match.pt")
    torch.save(torch.stack(rows), labels / "match.pt")
    return features, labels


def dataset(artifacts, **kwargs):
    features, labels = artifacts
    return ActionDataset(["match"], feature_dir=features, label_dir=labels,
                         task="outcome_scoring", train=False, **kwargs)


def test_cli_filter_and_duration_forwarding():
    args = wrapper.parse_args(["--filter", "pass", "--min-pass-duration", "0.1"])
    for task in ("outcome_scoring", "outcome_conceding", "pass_success"):
        command = wrapper.add_task_evaluation_options([], args, task)
        assert command[command.index("--filter") + 1] == "pass"
        assert command[command.index("--min-pass-duration") + 1] == "0.1"
    assert wrapper.evaluation_options(args)["filter"] == "pass"
    assert wrapper.evaluation_options(args)["min_pass_duration"] == .1
    assert wrapper.parse_args([]).evaluation_filter is None
    assert wrapper.parse_args([]).min_pass_duration is None
    command = wrapper.add_task_evaluation_options(
        [], wrapper.parse_args(["--filter", "pass", "--min-pass-duration", "0"]), "outcome_scoring")
    assert command[command.index("--min-pass-duration") + 1] == "0.0"
    for value in ("-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            wrapper.parse_args(["--min-pass-duration", value])
    with pytest.raises(SystemExit):
        wrapper.parse_args(["--filter", "carry"])
    with pytest.raises(ValueError, match="learning-curve"):
        wrapper._main(wrapper.parse_args(["--filter", "pass", "--learning-curve"]), ComparisonWarnings())


def test_pass_selection_duration_boundary_and_counts(action_artifacts):
    assert len(dataset(action_artifacts)) == 5
    selected = dataset(action_artifacts, evaluation_filter="pass", min_pass_dur=.1)
    assert selected.labels[:, config.LABEL_INDEX["action_index"]].tolist() == [11, 12, 13]
    assert selected.labels[:, config.LABEL_INDEX["is_pass"]].tolist() == [1, 1, 1]
    assert [g.evaluation_source_row for g in selected.features] == [2, 3, 4]
    assert selected.selection_counts == dict(loaded_samples=5, task_eligible_samples=5,
                                            excluded_by_filter=1, excluded_by_duration=1,
                                            selected_samples=3, evaluated_samples=3)
    assert selected.skipped_rows == {}
    assert len(dataset(action_artifacts, evaluation_filter="pass", min_pass_dur=0)) == 4
    # Duration alone continues to leave carry rows in the population.
    assert len(dataset(action_artifacts, min_pass_dur=.1)) == 4
    empty = dataset(action_artifacts, evaluation_filter="pass", min_pass_dur=10)
    assert len(empty) == 0 and empty.selection_counts["evaluated_samples"] == 0
    features, labels = action_artifacts
    with pytest.raises(ValueError, match="train=False"):
        ActionDataset([], feature_dir=features, label_dir=labels, task="outcome_scoring", evaluation_filter="pass")


def test_filter_preserves_predictions_targets_and_filtered_artifacts(action_artifacts, tmp_path):
    torch.manual_seed(12)
    arguments = model_args("outcome_scoring")
    arguments["use_xt"] = True
    model = GNN(arguments).eval()
    full, selected = [dataset(action_artifacts, **options) for options in (
        {}, dict(evaluation_filter="pass", min_pass_dur=.1))]
    results = []
    for population in (full, selected):
        results.append(run_epoch(argparse.Namespace(**arguments), model,
                                 DataLoader(population, batch_size=2), device="cpu",
                                 return_learning_curve_rows=True, return_semantic_prediction_rows=True,
                                 return_outcome_evaluation=True))
    _, all_rows, _ = results[0]
    metrics, selected_rows, diagnostics = results[1]
    assert metrics["count"] == 3
    expected = {row["action_index"]: row for row in all_rows}
    for row in selected_rows:
        assert row["prediction"] == pytest.approx(expected[row["action_index"]]["prediction"], abs=1e-7)
        assert row["soft_target"] == expected[row["action_index"]]["soft_target"]
        assert row["target"] == expected[row["action_index"]]["target"]
    _, table = evaluation.write_outcome_evaluation_artifacts(
        tmp_path / "report", model_id="outcome_scoring/test", task="outcome_scoring",
        outcome_evaluation=diagnostics["outcome"], outcome_bootstrap_resamples=0)
    assert (table.loc[table.stratum == "pooled_factual", "sample_count"] == 3).all()
    bins = pd.read_csv(tmp_path / "report/calibration_bins.csv")
    pooled = bins.loc[bins.stratum == "pooled_factual"]
    assert (pooled.groupby("evaluation_target").sample_count.sum() == 3).all()


@pytest.mark.parametrize("threshold,expected_count", [(None, 1), ("0", 4), ("0.1", 3), ("10", 0)])
@pytest.mark.parametrize("explicit_source", [False, True])
def test_evaluator_cli_selection_and_metadata(action_artifacts, tmp_path, monkeypatch, threshold, expected_count, explicit_source):
    features, labels = action_artifacts
    arguments = model_args("outcome_scoring")
    arguments.update(batch_size=2, min_pass_dur=.2, return_type="next_3", use_xt=True,
                     feature_dir=str(features), label_dir=str(labels))
    if explicit_source:
        arguments["feature_run_id"] = "missing_original"
        monkeypatch.setattr(project_config, "FEATURE_RUNS_DIR", tmp_path / "runs")
        root = tmp_path / "runs" / "selected"
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({"status": "completed"}))
        shutil.copytree(features, root / features.name)
        mode = arguments.get("intended_receiver_mode") or project_config.DEFAULT_INTENDED_RECEIVER_MODE
        for return_type in ("next_3", "next_10"):
            shutil.copytree(labels, project_config.get_action_label_dir(return_type, intended_receiver_mode=mode, root=root))
    model = GNN(arguments).eval()
    monkeypatch.setattr(utils, "load_model", lambda *args: model)
    monkeypatch.setattr(utils, "get_model_record", lambda *args: {
        "model_id": "outcome_scoring/test", "args": arguments,
        "metadata": {"split_manifest_id": "fixture", "train_match_ids": ["train"], "validation_match_ids": ["valid"]}})
    monkeypatch.setattr(evaluation_context, "load_recorded_split_manifest",
                        lambda _: {"train": ["train", "valid"], "test": ["match"]})
    monkeypatch.setattr(project_config, "resolve_artifact_split", lambda *args, **kwargs: (
        dict(manifest_id="fixture", test=["match"], metadata={"universe_fingerprint": "fixture"}), "fixture"))
    monkeypatch.setattr(project_config, "EVALUATION_RUNS_DIR", tmp_path / "summary")
    output = tmp_path / "evaluation"
    command = ["test.py", "--model_id", "outcome_scoring/test", "--filter", "pass",
               "--device", "cpu", "--evaluation-output-dir", str(output),
               "--outcome-bootstrap-resamples", "0", "--comparison-rows"]
    if threshold is not None:
        command.extend(["--min-pass-duration", threshold])
    if explicit_source:
        command.extend(["--feature-run-id", "selected"])
    monkeypatch.setattr(sys, "argv", command)
    if expected_count == 0:
        with pytest.raises(ValueError, match="No usable test samples.*filter='pass'"):
            runpy.run_path(str(Path(evaluation.__file__)), run_name="__main__")
        return
    runpy.run_path(str(Path(evaluation.__file__)), run_name="__main__")
    metadata = json.loads((output / "metadata.json").read_text())
    if explicit_source:
        assert metadata["feature_run_id"] == "selected"
        assert metadata["evaluation_data"]["training_feature_run_id"] == "missing_original"
        assert metadata["evaluation_data"]["diagnostic_feature_run_id"] == "selected"
    context = metadata["evaluation_context"]
    assert context["filter"] == "pass"
    assert context["requested_min_pass_duration"] == (None if threshold is None else float(threshold))
    assert context["min_pass_duration"] == (.2 if threshold is None else float(threshold))
    assert context["training_settings"]["min_pass_dur"] == .2
    assert context["selection_counts"]["evaluated_samples"] == expected_count
    assert metadata["evaluation_options"]["selection_counts"] == context["selection_counts"]
    exported = pd.read_csv(output / "learning_curve_predictions.csv")
    assert len(exported) == expected_count and exported.action_index.between(10, 13).all()
    assert arguments["min_pass_dur"] == .2


def test_wrapper_applies_selection_to_all_checkpoints_and_saves_effective_context(tmp_path, monkeypatch):
    entries = [dict(model_id=f"outcome_scoring/{name}", task="outcome_scoring", test_match_ids=["test"],
                    target_definitions={}, checkpoint_record={"args": {}, "metadata": {
                        "train_match_ids": ["train"], "validation_match_ids": ["valid"]}})
               for name in ("july", "october")]
    args = wrapper.parse_args(["--filter", "pass", "--min-pass-duration", "0.1",
                               "--evaluation-output-dir", str(tmp_path)])
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "resolve_evaluation_entries", lambda args: (entries, {}, None))
    monkeypatch.setattr(wrapper, "preflight_evaluation_data", lambda requests: {})
    context = dict(filter="pass", requested_min_pass_duration=.1, min_pass_duration=.1,
                   selection_counts={"evaluated_samples": 3})
    commands = []

    def evaluate(command, **kwargs):
        commands.append(command)
        assert command[command.index("--filter") + 1] == "pass"
        assert command[command.index("--min-pass-duration") + 1] == "0.1"
        output = Path(command[command.index("--evaluation-output-dir") + 1])
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(json.dumps({"evaluation_context": context}))

    monkeypatch.setattr(wrapper.subprocess, "run", evaluate)
    wrapper.main()
    assert len(commands) == 2
    manifest = json.loads(next(tmp_path.rglob("experiment_manifest.json")).read_text())
    assert manifest["evaluation_context"]["filter"] == "pass"
    assert manifest["evaluation_options"]["min_pass_duration"] == .1
    assert [entry["evaluation_context"] for entry in manifest["entries"]] == [context, context]


def save_frames(root, frames, task="pass_success"):
    records = []
    for i, frame in enumerate(frames):
        folder = root / str(i)
        folder.mkdir()
        frame.to_csv(folder / "learning_curve_predictions.csv", index=False)
        records.append(dict(model_id=f"{task}/{i}", task=task, output_dir=str(folder)))
    return records


def event_frame():
    return pd.DataFrame(dict(match_id=["a", "a", "b", "b"], action_index=[10, 20, 30, 40],
                             source_index=[0, 1, 0, 1], target=[0, 1, 0, 1], prediction=[.1, .8, .2, .9]))


def test_comparison_and_bootstrap_use_events_despite_different_artifact_rows(tmp_path):
    first = event_frame()
    second = first.iloc[::-1].copy()
    second["source_index"] = [9, 7, 5, 3]
    records = save_frames(tmp_path, [first, second])
    report = model_comparison.write_report(records, tmp_path / "report", resamples=10, seed=42)
    assert report["identity_columns"] == ["match_id", "action_index"]
    assert report["reference_differences"] == []
    differences = pd.read_csv(tmp_path / "report/comparison_differences.csv")
    assert np.allclose(differences.difference, 0)
    assert np.allclose(differences.lower, 0)
    assert np.allclose(differences.upper, 0)
    aligned = pd.read_csv(tmp_path / "report/aligned_predictions.csv")
    assert aligned.action_index.tolist() == [10, 20, 30, 40]
    assert aligned.source_index_1.tolist() == [3, 5, 7, 9]
    saved = json.loads((tmp_path / "report/experiment_manifest.json").read_text())
    assert saved["example_count"] == 4


@pytest.mark.parametrize("problem", ["missing", "duplicate", "blank", "fractional", "mixed"])
def test_comparison_rejects_bad_or_mixed_event_identities(tmp_path, problem):
    first, second = event_frame(), event_frame()
    if problem == "missing":
        second = second.iloc[1:]
    elif problem == "duplicate":
        second.loc[1, "action_index"] = 10
    elif problem == "blank":
        second["action_index"] = second.action_index.astype(object)
        second.loc[1, "action_index"] = ""
    elif problem == "fractional":
        second["action_index"] = second.action_index.astype(float)
        second.loc[1, "action_index"] = 20.5
    elif problem == "mixed":
        second = second.drop(columns="action_index")
    records = save_frames(tmp_path, [first, second])
    with pytest.raises(ValueError, match="action populations|Duplicate|invalid event|Cannot mix"):
        model_comparison.load_aligned_predictions(records)


def test_targets_use_float_tolerance_but_diagnostic_and_branch_checks_are_exact(tmp_path):
    first = event_frame().assign(soft_target=[.02, .03, .04, .05], execution_branch=[0, 1, 0, 1])
    second = first.copy()
    second["soft_target"] += 1e-9
    records = save_frames(tmp_path, [first, second], "outcome_scoring")
    frames, messages = model_comparison.load_aligned_predictions(records)
    assert messages == []
    assert model_comparison.reference_qualification("outcome_scoring", "xt_rmse", records, frames) == "common_references"
    second.loc[0, "soft_target"] += .01
    second.loc[1, "execution_branch"] = 0
    second.loc[2, "target"] = 1
    second.to_csv(tmp_path / "1/learning_curve_predictions.csv", index=False)
    frames, messages = model_comparison.load_aligned_predictions(records)
    assert {row["column"] for row in messages} == {"soft_target", "execution_branch", "target"}
    assert all(row["affected_row_count"] == 1 for row in messages)
    assert model_comparison.reference_qualification("outcome_scoring", "xt_rmse", records, frames) == "different_references"
