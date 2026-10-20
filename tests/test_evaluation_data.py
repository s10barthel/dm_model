"""Evaluation sources are explicit, compatible, and complete before inference."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import project_config
import test as evaluator
from datatools.config import LABEL_COLUMNS
from models import evaluation_data as sources
from scripts import evaluate_relevant_models as wrapper, model_comparison


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(project_config, "FEATURE_RUNS_DIR", runs)
    roots = {}
    for identifier, parent in (("selected", None), ("diagnostic", "selected")):
        root = runs / identifier
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({
            "status": "completed", "derived_from_feature_run_id": parent,
            "pass_height_threshold_meters": 2.0, "split_manifest_id": "artifact_split",
        }))
        graph_dir = root / "action_graphs"
        graph_dir.mkdir()
        for match in ("m1", "m2"):
            graph = Data(x=torch.zeros((2, 26)), edge_index=torch.tensor([[0, 1], [1, 0]]),
                         edge_attr=torch.zeros((2, 5)))
            torch.save([graph, graph.clone()], graph_dir / f"{match}.pt")
        for variant in ("disc_0.7", "next_10", "next_3"):
            labels = root / f"action_labels_{variant}_model"
            labels.mkdir()
            for match in ("m1", "m2"):
                torch.save(torch.zeros((2, len(LABEL_COLUMNS))), labels / f"{match}.pt")
        roots[identifier] = root
    return roots


def settings(*, newer=False, task="pass_intent", training="missing_original"):
    return {"task": task, "feature_run_id": training,
            "feature_dir": f"old/{training}/action_graphs",
            "label_dir": f"old/{training}/action_labels_{'next_10' if newer else 'disc_0.7'}_model",
            "return_type": "next_10" if newer else "disc_0.7", "intended_receiver_mode": "model",
            "node_in_dim": 27 if newer else 26, "edge_in_dim": 5 if newer else 4,
            "v_edge_feature_mode": "no_poss", "relative_speed_edge_feature_mode": "no_poss" if newer else "none",
            "lane_survival": newer}


def request(model_id, *, newer=False, cli=None, model=None):
    return {"model_id": model_id, "model_args": model or settings(newer=newer),
            "cli_args": cli or SimpleNamespace(feature_run_id="selected"), "match_ids": ["m1", "m2"]}


def entries():
    return [{"model_id": f"pass_intent/{name}", "task": "pass_intent", "train_matches": index + 1,
             "test_match_ids": ["m1", "m2"], "train_match_ids": ["train"], "target_definitions": {},
             "checkpoint_record": {"model_id": f"pass_intent/{name}", "feature_run_id": "missing_original",
                                   "args": settings(newer=bool(index)),
                                   "metadata": {"train_match_ids": ["train"], "validation_match_ids": ["valid"]}}}
            for index, name in enumerate(("old", "new"))]


def write_predictions(command):
    output = Path(command[command.index("--evaluation-output-dir") + 1])
    output.mkdir(parents=True)
    (output / "metadata.json").write_text("{}")
    pd.DataFrame({"match_id": ["m1", "m1", "m2", "m2"], "source_index": [0, 1, 0, 1],
                  "target": [0, 1, 0, 1], "prediction": [0, 1, 0, 1],
                  "target_id": ["p0", "p1"] * 2, "prediction_id": ["p0", "p1"] * 2,
                  "candidate_ids": ['["p0","p1"]'] * 4, "target_probability": [.8, .9] * 2,
                  "reciprocal_rank": [1., 1.] * 2}).to_csv(output / "learning_curve_predictions.csv", index=False)


def test_explicit_run_works_without_original_inputs_and_preserves_definitions(artifacts):
    originals = [settings(), settings(newer=True)]
    before = deepcopy(originals)
    resolved = sources.preflight_evaluation_data([
        request("pass_intent/old", model=originals[0]), request("pass_intent/new", model=originals[1]),
    ])
    for identifier, variant in (("pass_intent/old", "disc_0.7"), ("pass_intent/new", "next_10")):
        context = resolved[identifier]
        assert context["training_feature_run_id"] == "missing_original"
        assert context["feature_run_id"] == "selected"
        assert context["feature_dir"] == str(artifacts["selected"] / "action_graphs")
        assert context["label_dir"] == str(artifacts["selected"] / f"action_labels_{variant}_model")
        assert context["label_definitions"]["return_type"] == variant
    assert resolved["pass_intent/new"]["model_schema"]["node_in_dim"] == 26
    assert originals == before


def test_diagnostics_default_to_explicit_primary_and_override_only_labels(artifacts):
    model = settings(task="outcome_scoring")
    # An archived diagnostic path must not override the explicit input run.
    model["diagnostic_label_dir"] = str(artifacts["diagnostic"] / "action_labels_next_10_model")
    default = sources.resolve_evaluation_data(SimpleNamespace(feature_run_id="selected"), model)
    override = sources.resolve_evaluation_data(
        SimpleNamespace(feature_run_id="selected", diagnostic_feature_run_id="diagnostic"), model)
    assert default["diagnostic_feature_run_id"] == "selected"
    assert default["diagnostic_label_dir"] == str(artifacts["selected"] / "action_labels_next_10_model")
    assert override["diagnostic_feature_run_id"] == "diagnostic"
    assert override["feature_dir"] == default["feature_dir"]
    assert override["label_dir"] == default["label_dir"]
    assert override["diagnostic_label_dir"] == model["diagnostic_label_dir"]
    (artifacts["selected"] / "action_labels_next_10_model").rename(artifacts["selected"] / "archived_next_10")
    with pytest.raises(FileNotFoundError, match="diagnostic labels"):
        sources.resolve_evaluation_data(SimpleNamespace(feature_run_id="selected"), model)


def test_no_flags_preserve_checkpoint_sources_and_diagnostic_override_cannot_supply_graphs(artifacts):
    native = settings(task="outcome_scoring", training="selected")
    native["diagnostic_label_dir"] = str(artifacts["diagnostic"] / "action_labels_next_10_model")
    resolved = sources.resolve_evaluation_data(SimpleNamespace(), native)
    assert resolved["feature_run_id"] == "selected"
    assert resolved["requested_feature_run_id"] is None
    assert resolved["diagnostic_label_dir"] == native["diagnostic_label_dir"]
    with pytest.raises(FileNotFoundError, match="missing_original"):
        sources.resolve_evaluation_data(SimpleNamespace(diagnostic_feature_run_id="diagnostic"), settings())


def test_explicit_goal_horizon_uses_effective_diagnostic_source(artifacts):
    context = sources.resolve_evaluation_data(
        SimpleNamespace(feature_run_id="selected", diagnostic_return_type="next_3"), settings(task="outcome_scoring"))
    assert context["evaluation_diagnostic_label_dir"] == str(artifacts["selected"] / "action_labels_next_3_model")
    assert context["label_definitions"]["return_type"] == "disc_0.7"


def test_height_diagnostic_ancestry_is_relative_to_selected_inputs(artifacts):
    primary = sources.resolve_evaluation_data(SimpleNamespace(feature_run_id="selected"), settings(task="pass_height"))
    assert primary["pass_height_diagnostic_feature_run_id"] == "selected"
    assert primary["pass_height_diagnostic_lineage"] == ["selected"]
    derived = sources.resolve_evaluation_data(
        SimpleNamespace(feature_run_id="selected", diagnostic_feature_run_id="diagnostic"), settings(task="pass_height"))
    assert derived["pass_height_diagnostic_lineage"] == ["diagnostic", "selected"]
    metadata_path = artifacts["diagnostic"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["derived_from_feature_run_id"] = None
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="not equal to or derived"):
        sources.resolve_evaluation_data(
            SimpleNamespace(feature_run_id="selected", diagnostic_feature_run_id="diagnostic"), settings(task="pass_height"))


def test_graph_directory_override_is_preserved_and_recorded(artifacts):
    context = sources.resolve_evaluation_data(SimpleNamespace(feature_run_id="selected",
        feature_dir=str(artifacts["diagnostic"] / "action_graphs")), settings())
    assert context["feature_dir"] == str(artifacts["diagnostic"] / "action_graphs")
    assert context["label_dir"] == str(artifacts["selected"] / "action_labels_disc_0.7_model")


@pytest.mark.parametrize("auxiliary", [False, True])
def test_incompatible_graph_requirements_are_blocking(artifacts, auxiliary):
    incompatible = settings(newer=True)
    incompatible["node_in_dim"] = 40
    candidate = request("pass_intent/old")
    if auxiliary:
        candidate["auxiliaries"] = [{"model_id": "pass_height/aux", "args": incompatible}]
    else:
        candidate["model_args"] = incompatible
    with pytest.raises(sources.EvaluationDataError, match="incompatible"):
        sources.preflight_evaluation_data([candidate])


def test_missing_files_are_consolidated_without_changing_cohort(artifacts):
    (artifacts["selected"] / "action_graphs/m1.pt").unlink()
    (artifacts["selected"] / "action_labels_disc_0.7_model/m2.pt").unlink()
    with pytest.raises(sources.EvaluationDataError) as failure:
        sources.preflight_evaluation_data([request("pass_intent/old"), request("pass_intent/new", newer=True)])
    message = str(failure.value)
    assert all(text in message for text in ("pass_intent/old", "pass_intent/new", "feature_dir", "label_dir", "m1", "m2"))


@pytest.mark.parametrize("mode", ["ordinary", "comparison", "masks"])
def test_wrapper_forwards_primary_run_and_saves_preflight_provenance(artifacts, tmp_path, monkeypatch, mode):
    flags = ["--compare"] if mode == "comparison" else ["--inference-mask", "accel"] if mode == "masks" else []
    args = wrapper.parse_args([*flags, "--feature-run-id", "selected", "--evaluation-output-dir", str(tmp_path / "results"),
                               "--comparison-bootstrap-resamples", "0", "--accept-comparison-warnings"])
    selected = entries()
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "resolve_evaluation_entries", lambda args: (selected, {}, None))
    # Lane-survival cache resolution is independent of evaluation artifact selection.
    monkeypatch.setattr(wrapper, "resolve_pc_xpass_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_comparison, "save_plot", Mock())
    run = Mock(side_effect=lambda command, **kwargs: write_predictions(command))
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    wrapper.main()
    assert run.call_count == (4 if mode == "masks" else 2)
    for call in run.call_args_list:
        command = call.args[0]
        assert command.count("--feature-run-id") == 1
        assert command[command.index("--feature-run-id") + 1] == "selected"
    manifest = json.loads(next((tmp_path / "results").glob("*/experiment_manifest.json")).read_text())
    assert manifest["evaluation_data"]["pass_intent/old"]["feature_run_id"] == "selected"
    assert manifest["evaluation_data"]["pass_intent/old"]["training_feature_run_id"] == "missing_original"
    assert manifest["evaluation_data"]["pass_intent/new"]["label_definitions"]["return_type"] == "next_10"


def test_incomplete_primary_prevents_all_wrapper_subprocesses(artifacts, tmp_path, monkeypatch):
    (artifacts["selected"] / "action_labels_next_10_model/m2.pt").unlink()
    args = wrapper.parse_args(["--compare", "--feature-run-id", "selected", "--evaluation-output-dir", str(tmp_path / "results")])
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(wrapper, "resolve_evaluation_entries", lambda args: (entries(), {}, None))
    run = Mock()
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    with pytest.raises(sources.EvaluationDataError, match="m2"):
        wrapper.main()
    run.assert_not_called()


def test_learning_curve_forwards_shared_evaluation_source(artifacts, tmp_path, monkeypatch):
    args = wrapper.parse_args(["--learning-curve", "--pass-intent-model-id", "pass_intent/old",
                               "--feature-run-id", "selected", "--evaluation-output-dir", str(tmp_path / "results"),
                               "--learning-curve-bootstrap-resamples", "0"])
    selected = entries()
    monkeypatch.setattr(wrapper.learning_curve, "discover", lambda model: [entry["model_id"] for entry in selected])
    monkeypatch.setattr(wrapper.learning_curve, "validate_series", lambda *args, **kwargs: selected)
    monkeypatch.setattr(wrapper.learning_curve, "save_plot", Mock())
    run = Mock(side_effect=lambda command, **kwargs: write_predictions(command))
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    wrapper.run_learning_curve(args)
    assert run.call_count == 2
    for call in run.call_args_list:
        command = call.args[0]
        assert command[command.index("--feature-run-id") + 1] == "selected"


def test_per_model_metadata_keeps_evaluation_and_training_sources(artifacts, tmp_path, monkeypatch):
    context = sources.preflight_evaluation_data([request("pass_intent/old")])["pass_intent/old"]
    monkeypatch.setattr(evaluator, "EVALUATION_RUNS_DIR", tmp_path / "summary")
    evaluator.write_model_evaluation_artifacts(
        tmp_path / "results", model_id="pass_intent/old", task="pass_intent", feature_run_id="selected",
        diagnostic_feature_run_id=None, evaluation_timestamp="fixed", evaluation_options={},
        test_metrics={"accuracy": 1.}, evaluation_data=context)
    metadata = json.loads((tmp_path / "results/metadata.json").read_text())
    assert metadata["feature_run_id"] == "selected"
    assert metadata["evaluation_data"]["training_feature_run_id"] == "missing_original"
