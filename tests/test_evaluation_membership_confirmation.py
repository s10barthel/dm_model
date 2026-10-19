"""Missing historical membership can be accepted only for a specific evaluation cohort."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import test as evaluation
from models import evaluation_context, membership
from models.evaluation_membership import (
    EvaluationMembershipCancelled, collect_evaluation_membership, confirm_evaluation_membership,
    confirm_direct_evaluation_membership,
)
from scripts import evaluate_relevant_models as wrapper, model_comparison


def record(model_id="pass_intent/legacy", *, verified=False, **metadata):
    if verified:
        metadata = {"train_match_ids": ["train"], "validation_match_ids": ["valid"], **metadata}
    return {"model_id": model_id, "args": {}, "metadata": metadata}


def interactive(monkeypatch, answers):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    responses = iter(answers)
    prompt = Mock(side_effect=lambda text: next(responses))
    monkeypatch.setattr("builtins.input", prompt)
    return prompt


def test_missing_diagnostic_and_partial_positive_evidence():
    item = record(train_match_ids=["train"])
    with pytest.raises(membership.MissingMembershipError) as error:
        evaluation_context.check_heldout(item, ["eval"])
    assert error.value.check["status"] == "unverified"
    assert error.value.check["overlap_count"] is None
    assert error.value.check["exclusion_sources"] == {"training": 1}
    with pytest.raises(ValueError, match="overlaps") as overlap:
        evaluation_context.check_heldout(item, ["train", "eval"])
    assert not isinstance(overlap.value, membership.MissingMembershipError)


@pytest.mark.parametrize("metadata", [
    {"train_match_ids": "invalid"}, {"train_match_ids": ["train", "train"]},
    {"train_match_ids": ["train"], "split_datasets": {"train": {"requested_match_ids": ["other"]}}},
    {"train_match_ids": ["train"], "split_datasets": {"train": {"contributing_match_ids": ["other"]}}},
])
def test_invalid_membership_is_not_confirmable(metadata):
    with pytest.raises(ValueError) as error:
        collect_evaluation_membership([(record(**metadata), ["eval"])])
    assert not isinstance(error.value, membership.MissingMembershipError)


@pytest.mark.parametrize("failure", [FileNotFoundError("missing manifest"), ValueError("corrupt manifest")])
def test_broken_recorded_manifest_is_not_confirmable(monkeypatch, failure):
    monkeypatch.setattr(evaluation_context, "load_recorded_split_manifest", Mock(side_effect=failure))
    with pytest.raises(type(failure), match=str(failure)):
        collect_evaluation_membership([(record(split_manifest_id="recorded"), ["eval"])])


def test_conflicting_manifests_are_not_confirmable(monkeypatch):
    item = record(split_manifest_id="one")
    item["args"]["split_manifest_id"] = "two"
    monkeypatch.setattr(evaluation_context, "load_recorded_split_manifest", lambda _: {"train": []})
    with pytest.raises(ValueError, match="Conflicting"):
        collect_evaluation_membership([(item, ["eval"])])


def test_one_confirmation_for_all_models_and_auxiliaries(monkeypatch):
    report = collect_evaluation_membership([
        (record(), ["eval"]), (record("pass_height/aux"), ["eval"]),
        (record(), ["eval"]), (record("pass_intent/modern", verified=True), ["eval"]),
    ])
    prompt = interactive(monkeypatch, ["", "perhaps", "YeS"])
    confirm_evaluation_membership(report)
    assert prompt.call_count == 3
    assert report["affected_model_ids"] == ["pass_height/aux", "pass_intent/legacy"]
    assert len(report["checks"]) == 3
    assert report["confirmation"] == {"decision": "accepted", "method": "interactive"}
    confirm_evaluation_membership(collect_evaluation_membership([(record(verified=True), ["eval"])]))
    assert prompt.call_count == 3


@pytest.mark.parametrize("answer", ["n", "no", "eof", "interrupt", "noninteractive"])
def test_direct_cancellation_is_persisted_without_metrics(monkeypatch, tmp_path, answer):
    report = collect_evaluation_membership([(record(), ["eval"])])
    prompt = interactive(monkeypatch, [answer])
    if answer == "eof":
        prompt.side_effect = EOFError
    elif answer == "interrupt":
        prompt.side_effect = KeyboardInterrupt
    elif answer == "noninteractive":
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    args = SimpleNamespace(model_id="pass_intent/legacy", evaluation_output_dir=str(tmp_path))
    with pytest.raises(EvaluationMembershipCancelled):
        confirm_direct_evaluation_membership(report, args)
    saved = json.loads((tmp_path / "membership_cancellation.json").read_text())
    assert saved["status"] == "cancelled"
    assert saved["evaluation_membership"]["confirmation"]["decision"] == "cancelled"
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "metrics.csv").exists()
    if answer == "noninteractive":
        prompt.assert_not_called()


@pytest.mark.parametrize("change", [None, "model", "cohort", "decision", "method", "absent"])
def test_child_approval_requires_exact_interactive_acceptance(monkeypatch, tmp_path, change):
    source = collect_evaluation_membership([(record(), ["eval"])])
    source["confirmation"] = {"decision": "accepted", "method": "interactive"}
    child = deepcopy(source)
    child["confirmation"] = None
    if change == "model":
        source["checks"][0]["model_id"] = "pass_intent/different"
    elif change == "cohort":
        source["checks"][0]["requested_match_ids"] = ["other"]
    elif change in {"decision", "method"}:
        source["confirmation"][change] = "unapproved"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({} if change == "absent" else {"evaluation_membership": source}))
    prompt = Mock(side_effect=AssertionError("A child must not prompt"))
    monkeypatch.setattr("builtins.input", prompt)
    if change is None:
        confirm_evaluation_membership(child, acceptance_manifest=path)
        assert child["confirmation"]["source"] == "wrapper_manifest"
    else:
        with pytest.raises(ValueError, match="[Mm]embership"):
            confirm_evaluation_membership(child, acceptance_manifest=path)
    prompt.assert_not_called()


def test_direct_explicit_cohort_without_historical_split(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "evaluation_selection", lambda args: ["eval"])
    prompt = interactive(monkeypatch, ["yes"])
    args = SimpleNamespace(model_id="pass_intent/legacy", evaluation_output_dir=str(tmp_path))
    manifest, source, ids, report = evaluation.prepare_evaluation_population(args, record(), None, {})
    assert manifest is None
    assert source == "explicit season selection; historical split unavailable"
    assert ids.tolist() == ["eval"]
    assert report["status"] == "unverified"
    assert prompt.call_count == 1
    assert evaluation.evaluation_split_metadata(manifest) == {
        "train_split_percent": None, "train_count": None, "split_manifest_id": None}


def test_no_default_or_explicit_cohort_is_blocking(monkeypatch):
    monkeypatch.setattr(evaluation, "evaluation_selection", lambda args: None)
    with pytest.raises(ValueError, match="--season"):
        evaluation.prepare_evaluation_population(SimpleNamespace(), record(), None, {})


def test_direct_checks_auxiliaries_before_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "evaluation_selection", lambda args: ["eval"])
    monkeypatch.setattr(evaluation.utils, "get_model_record", lambda _: record("pass_height/aux", train_match_ids=["eval"]))
    prompt = interactive(monkeypatch, ["yes"])
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.prepare_evaluation_population(
            SimpleNamespace(model_id="pass_intent/legacy", evaluation_output_dir=str(tmp_path)), record(), None,
            {"pass_height": SimpleNamespace(args={"model_id": "pass_height/aux"})})
    prompt.assert_not_called()


def wrapper_checkpoint(model_id, **kwargs):
    item = record(model_id, verified=model_id.endswith("/modern"))
    item["args"] = {"task": "pass_intent", "node_in_dim": 19, "edge_in_dim": 2}
    return {"model_id": model_id, "task": "pass_intent", "train_matches": 0,
            "train_match_ids": [], "test_match_ids": None, "target_definitions": {},
            "checkpoint_record": item}


@pytest.mark.parametrize("mode", ["ordinary", "comparison", "masks"])
def test_wrapper_confirms_once_and_preserves_export_qualification(monkeypatch, tmp_path, mode):
    flags = ["--compare"] if mode == "comparison" else ["--inference-mask", "accel"] if mode == "masks" else []
    args = wrapper.parse_args([
        *flags, "--pass-intent-model-id", "pass_intent/legacy", "--pass-intent-model-id", "pass_intent/modern",
        "--season", "24_25", "--season-half", "second", "--filter", "pass", "--min-pass-duration", "0.5",
        "--evaluation-output-dir", str(tmp_path), "--comparison-bootstrap-resamples", "0", "--accept-comparison-warnings",
    ])
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(model_comparison, "checkpoint", wrapper_checkpoint)
    monkeypatch.setattr(wrapper, "evaluation_selection", lambda args: ["eval"])
    monkeypatch.setattr(evaluation, "evaluation_selection", lambda args: ["eval"])
    prompt = interactive(monkeypatch, ["yes"])
    plot = Mock()
    monkeypatch.setattr(model_comparison, "save_plot", plot)

    def child(command, **kwargs):
        model_id = command[command.index("--model_id") + 1]
        output = Path(command[command.index("--evaluation-output-dir") + 1])
        approval = (command[command.index("--membership-acceptance-manifest") + 1]
                    if "--membership-acceptance-manifest" in command else None)
        _, _, _, report = evaluation.prepare_evaluation_population(
            SimpleNamespace(model_id=model_id, evaluation_output_dir=str(output), membership_acceptance_manifest=approval),
            wrapper_checkpoint(model_id)["checkpoint_record"], None, {})
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(json.dumps({"evaluation_membership": report}))
        frame = pd.DataFrame({"match_id": ["eval", "eval"], "source_index": [0, 1], "target": [0, 1],
                              "prediction": [0, 1], "target_id": ["p0", "p1"], "prediction_id": ["p0", "p1"],
                              "candidate_ids": ['["p0","p1"]'] * 2, "target_probability": [.8, .9],
                              "reciprocal_rank": [1., 1.]})
        frame.to_csv(output / "learning_curve_predictions.csv", index=False)

    run = Mock(side_effect=child)
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    wrapper.main()
    assert prompt.call_count == 1
    assert run.call_count == (4 if mode == "masks" else 2)
    manifest_path = next(tmp_path.glob("*/experiment_manifest.json"))
    report = json.loads(manifest_path.read_text())
    assert report["evaluation_membership"]["status"] == "unverified"
    assert report["evaluation_membership"]["confirmation"]["decision"] == "accepted"
    assert report["evaluation_membership"]["affected_model_ids"] == ["pass_intent/legacy"]
    for path in tmp_path.rglob("comparison_metrics.csv"):
        table = pd.read_csv(path)
        assert table.loc[table.model_id.eq("pass_intent/legacy"), "evaluation_membership_status"].eq("unverified").all()
        assert table.loc[table.model_id.eq("pass_intent/modern"), "evaluation_membership_status"].eq("verified").all()
    if mode == "comparison":
        assert pd.read_csv(manifest_path.parent / "comparison_differences.csv").evaluation_membership_status.eq("unverified").all()
    if mode != "ordinary":
        assert plot.called


def test_wrapper_refusal_and_general_acceptance_flag_do_not_launch_children(monkeypatch, tmp_path):
    args = wrapper.parse_args(["--pass-intent-model-id", "pass_intent/legacy", "--season", "24_25",
                               "--evaluation-output-dir", str(tmp_path), "--accept-comparison-warnings"])
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(model_comparison, "checkpoint", wrapper_checkpoint)
    monkeypatch.setattr(wrapper, "evaluation_selection", lambda args: ["eval"])
    prompt = interactive(monkeypatch, ["no"])
    run = Mock()
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    with pytest.raises(SystemExit) as cancelled:
        wrapper.main()
    assert cancelled.value.code == 1
    assert prompt.call_count == 1
    run.assert_not_called()
    path = next(tmp_path.glob("evaluation_*_manifest.json"))
    report = json.loads(path.read_text())
    assert report["status"] == "cancelled"
    assert report["evaluation_membership"]["confirmation"]["decision"] == "cancelled"


def test_exports_and_plot_preserve_unverified_membership(monkeypatch, tmp_path):
    prompt = interactive(monkeypatch, ["yes"])
    unverified = collect_evaluation_membership([(record(), ["eval"])])
    confirm_evaluation_membership(unverified)
    verified = collect_evaluation_membership([(record("pass_intent/modern", verified=True), ["eval"])])
    monkeypatch.setattr(evaluation, "EVALUATION_RUNS_DIR", tmp_path / "summary")
    evaluation.write_model_evaluation_artifacts(
        tmp_path / "single", model_id="pass_intent/legacy", task="pass_intent", feature_run_id=None,
        diagnostic_feature_run_id=None, evaluation_timestamp="fixed", test_metrics={"accuracy": 1.},
        evaluation_options={**evaluation.evaluation_split_metadata(None), "evaluation_membership_status": "unverified"},
        evaluation_membership=unverified,
    )
    metadata = json.loads((tmp_path / "single" / "metadata.json").read_text())
    assert metadata["evaluation_membership"] == unverified
    assert metadata["evaluation_options"]["train_split_percent"] is None
    assert json.loads((tmp_path / "single" / "metrics.json").read_text())["evaluation_membership_status"] == "unverified"
    assert pd.read_csv(tmp_path / "single" / "metrics.csv").evaluation_membership_status.tolist() == ["unverified"]

    records = []
    frame = pd.DataFrame({"match_id": ["eval", "eval"], "source_index": [0, 1], "target": [0, 1],
                          "prediction": [.1, .9]})
    for index, report in enumerate((unverified, verified, verified)):
        output = tmp_path / str(index)
        output.mkdir()
        frame.to_csv(output / "learning_curve_predictions.csv", index=False)
        records.append({"task": "pass_success", "model_id": f"pass_success/{index}", "output_dir": str(output),
                        "evaluation_membership": report})
    titles = []
    original = model_comparison.plt.Figure.suptitle

    def title(figure, text, *args, **kwargs):
        titles.append(text)
        return original(figure, text, *args, **kwargs)

    monkeypatch.setattr(model_comparison.plt.Figure, "suptitle", title)
    exported = model_comparison.write_report(records, tmp_path / "comparison", resamples=0, seed=42)
    assert exported["evaluation_membership"]["status"] == "unverified"
    assert exported["evaluation_membership"]["confirmations"][0]["decision"] == "accepted"
    assert "Evaluation membership unverified" in titles[0]
    assert (tmp_path / "comparison" / "comparison.png").is_file()
    differences = pd.read_csv(tmp_path / "comparison" / "comparison_differences.csv")
    assert differences.loc[differences.model_a.eq("pass_success/0"), "evaluation_membership_status"].eq("unverified").all()
    assert differences.loc[differences.model_a.eq("pass_success/1"), "evaluation_membership_status"].eq("verified").all()
    assert prompt.call_count == 1
