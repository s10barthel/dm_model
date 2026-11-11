"""Mixed outcome targets require explicit, recorded inference acceptance."""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datatools.xt_targets import training_target_definition
from models import utils
from models.outcome_learning import signed_contract
from scripts.comparison_warnings import ComparisonCancelled, ComparisonWarnings
from scripts.outcome_warnings import confirm_outcome_warnings, outcome_warning_metadata
from scripts import inference_visualization_comparison as comparison
from scripts import run_and_visualize_hawkeye as hawkeye
from scripts import visualize_action_components as sportec
from test_visualization_comparison import _record, inference_fixture, outcome_pipeline


def metadata(signed=False, surface="surface"):
    return dict(xt_version=surface, xt_fit_metadata_sha256="fit",
                xt_targets=dict(version=2, representation="gain" if signed else "absolute",
                                clip_negative_gains=False))


def records(mismatch="both"):
    result = {task: _record(task + "/fixture") for task in ("outcome_scoring", "outcome_conceding")}
    for record in result.values():
        record.update(target_family="xt", xt_target_definition=training_target_definition(metadata()))
    if mismatch in {"representation", "both"}:
        result["outcome_scoring"]["xt_target_definition"] = training_target_definition(metadata(True))
    if mismatch in {"contract", "both"}:
        result["outcome_scoring"]["outcome_learning_contract"] = signed_contract({})
    if mismatch == "surface":
        result["outcome_scoring"]["xt_target_definition"] = training_target_definition(metadata(surface="other"))
    return result


@pytest.mark.parametrize("mismatch", ["same", "representation", "contract", "surface", "both"])
def test_strict_validation_and_optional_collector(mismatch):
    selected = records(mismatch)
    original = copy.deepcopy(selected)
    policy = ComparisonWarnings(True)
    shared = utils.validate_model_record_consistency(selected, warning_collector=policy)
    assert len(policy.messages) == (mismatch != "same")
    assert selected == original
    assert shared["source_xt_target_definitions"] == {
        task: record["xt_target_definition"] for task, record in selected.items()}
    assert shared["source_outcome_learning_contracts"]["outcome_conceding"] is None
    if mismatch != "same":
        with pytest.raises(ValueError, match="Selected xT outcome checkpoints"):
            utils.validate_model_record_consistency(selected)
        assert all(record["model_id"] in policy.messages[0] for record in selected.values())
    if mismatch in {"both", "representation", "surface"}:
        assert shared["xt_target_definition"] is None
    if mismatch in {"both", "contract"}:
        assert shared["outcome_learning_contract"] is None
    if mismatch == "both":
        assert "xt_targets.representation" in policy.messages[0]
        assert "outcome_learning_contract" in policy.messages[0]
        assert "surface and fit fingerprint agree" in policy.messages[0]
    if mismatch == "surface":
        assert "surface and fit fingerprint agree" not in policy.messages[0]


@pytest.mark.parametrize("answers,accepted", [(["", "maybe", "YES"], True), (["no"], False),
                                            (["n"], False), ([EOFError], False)])
def test_explicit_confirmation_and_recorded_cancellation(tmp_path, monkeypatch, answers, accepted):
    policy = ComparisonWarnings()
    shared = utils.validate_model_record_consistency(records(), warning_collector=policy)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    responses = iter(answers)
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        response = next(responses)
        if response is EOFError:
            raise EOFError
        return response
    monkeypatch.setattr("builtins.input", answer)
    if accepted:
        confirm_outcome_warnings(policy, tmp_path, shared)
        confirm_outcome_warnings(policy, tmp_path, shared)
        assert len(prompts) == len(answers)
    else:
        with pytest.raises(ComparisonCancelled):
            confirm_outcome_warnings(policy, tmp_path, shared)
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["status"] == ("running" if accepted else "cancelled")
    details = saved["outcome_compatibility"]
    assert details["comparison_cancelled"] == (not accepted)
    assert bool(details["warning_acceptances"]) == accepted
    assert details["source_xt_target_definitions"] == shared["source_xt_target_definitions"]


@pytest.mark.parametrize("accept", [False, True])
def test_noninteractive_requires_acceptance_flag(tmp_path, monkeypatch, accept):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    policy = ComparisonWarnings(accept)
    shared = utils.validate_model_record_consistency(records(), warning_collector=policy)
    if accept:
        confirm_outcome_warnings(policy, tmp_path, shared)
        assert policy.acceptances[0]["method"] == "command_line"
    else:
        with pytest.raises(ComparisonCancelled, match="accept-comparison-warnings"):
            confirm_outcome_warnings(policy, tmp_path, shared)


def test_same_definitions_need_no_prompt_or_startup_artifact(tmp_path, monkeypatch):
    policy = ComparisonWarnings()
    shared = utils.validate_model_record_consistency(records("same"), warning_collector=policy)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected prompt"))
    confirm_outcome_warnings(policy, tmp_path, shared)
    assert not (tmp_path / "metadata.json").exists()


@pytest.fixture
def runtime(monkeypatch):
    policy = ComparisonWarnings(True)
    shared = utils.validate_model_record_consistency(records(), warning_collector=policy)
    shared.pop("stop_condition_definition")
    runs = {"a": metadata(), "b": metadata(True), "z": metadata(surface="new")}
    def resolve(identifier, *args, **kwargs):
        return dict(feature_run_id=identifier, metadata=runs[identifier], feature_root=identifier)
    monkeypatch.setattr(utils, "_resolve_runtime_feature_run_candidate", resolve)
    monkeypatch.setattr(utils, "runtime_feature_run_candidate_ids", lambda *a: list(runs))
    monkeypatch.setattr(utils, "all_runtime_feature_run_ids", lambda: list(runs))
    monkeypatch.setattr(utils, "_feature_run_sort_key", lambda identifier, metadata: identifier)
    return policy, shared, runs


def test_runtime_prefers_known_definition_and_records_label_matches(runtime):
    policy, shared, _ = runtime
    chosen = utils.resolve_runtime_feature_run_context(None, shared, None, None, {}, warning_collector=policy)
    assert chosen["feature_run_id"] == "b"  # z is newer, but matches neither checkpoint.
    assert chosen["xt_target_reference"]["matching_tasks"] == ["outcome_scoring"]
    assert shared["runtime_xt_reference"]["mismatching_tasks"] == ["outcome_conceding"]
    assert len(policy.messages) == 1


def test_runtime_new_definition_requires_new_acceptance(runtime):
    policy, shared, _ = runtime
    policy.confirm()
    chosen = utils.resolve_runtime_feature_run_context("z", shared, None, None, {}, warning_collector=policy)
    assert chosen["xt_target_reference"]["matching_tasks"] == []
    assert len(policy.messages) == 2
    assert "native-target evaluation reference" in policy.messages[1]
    policy.confirm()
    assert len(policy.acceptances) == 2
    assert outcome_warning_metadata(policy, shared)["outcome_compatibility"]["runtime_xt_reference"]


def test_runtime_default_still_rejects_mixed_labels(runtime):
    _, shared, _ = runtime
    with pytest.raises(ValueError, match="different xT surface or target definition"):
        utils.resolve_runtime_feature_run_context("b", shared, None, None, {})


def test_legacy_unknown_definitions_do_not_invent_runtime_mismatch(runtime):
    policy, shared, _ = runtime
    policy.messages.clear()
    shared["source_xt_target_definitions"] = {task: None for task in records()}
    shared["xt_target_definition"] = None
    result = utils.resolve_runtime_feature_run_context("b", shared, None, None, {}, warning_collector=policy)
    assert "xt_target_reference" not in result
    assert not policy.messages


def test_warning_identity_deduplicates_repeated_variant_and_model_order(tmp_path, monkeypatch):
    policy = ComparisonWarnings()
    selected = records()
    shared = utils.validate_model_record_consistency(selected, warning_collector=policy)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "yes")
    confirm_outcome_warnings(policy, tmp_path, shared)
    utils.validate_model_record_consistency(dict(reversed(list(selected.items()))), warning_collector=policy)
    confirm_outcome_warnings(policy, tmp_path, shared)
    assert len(prompts) == 1


def test_runtime_searches_fallback_for_matching_definition(runtime, monkeypatch):
    policy, shared, _ = runtime
    monkeypatch.setattr(utils, "runtime_feature_run_candidate_ids", lambda *a: ["z"])
    result = utils.resolve_runtime_feature_run_context(None, shared, None, None, {}, warning_collector=policy)
    assert result["feature_run_id"] == "b"
    assert result["selection"] == "newest_compatible_all_feature_runs"
    assert len(policy.messages) == 1


def test_runtime_unavailable_targets_remain_errors(runtime):
    policy, shared, runs = runtime
    runs["b"]["xt_unavailable_return_types"] = ["next_5"]
    with pytest.raises(ValueError, match="unavailable"):
        utils.resolve_runtime_feature_run_context("b", shared, None, None, {}, warning_collector=policy)


@pytest.mark.parametrize("accept", [False, True])
def test_ordinary_visualizer_accepts_or_cancels_before_prediction(tmp_path, monkeypatch, outcome_pipeline, accept):
    script, required, _, calls, build, render = outcome_pipeline
    selected = records()
    selected["pass_success"] = _record("pass_success/fixed")
    ids = {task: record["model_id"] for task, record in selected.items()}
    ids["outcome_scoring"] = "outcome_scoring/a"
    ids["outcome_conceding"] = "outcome_conceding/x"
    for task, identifier in ids.items():
        selected[task]["model_id"] = identifier
    def resolve(required_tasks, **kwargs):
        return ids, utils.validate_model_record_consistency(selected, warning_collector=kwargs["warning_collector"]), None
    monkeypatch.setattr(script, "resolve_model_selection", resolve)
    monkeypatch.setattr(script, "load_model", lambda identifier, device: comparison.load_model(identifier, device))
    if script is sportec:
        monkeypatch.setattr(script, "resolve_runtime_feature_run_context", lambda *a, **kw: dict(
            feature_root=tmp_path, feature_run_id="feature", intended_receiver_mode="predict", selection="explicit"))
    flags = [*required, "--only-pass-score", "--output-dir", str(tmp_path), "--run-id", "ordinary"]
    for task, identifier in ids.items():
        flags += [f"--{task.replace('_', '-')}-model-id", identifier]
    args = script.parse_args(flags)
    monkeypatch.setattr(script, "parse_args", lambda: args)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or ("yes" if accept else "no"))
    if accept:
        script.main()
        assert calls and render.called
    else:
        with pytest.raises(ComparisonCancelled):
            script.main()
        assert not calls and not render.called
    assert len(prompts) == 1
    saved = json.loads((tmp_path / "ordinary/metadata.json").read_text())
    assert saved["status"] == ("completed" if accept else "cancelled")
    assert bool(saved["outcome_compatibility"]["warning_acceptances"]) == accept


@pytest.mark.parametrize("accept", [False, True])
def test_comparison_reuses_one_policy(tmp_path, monkeypatch, outcome_pipeline, accept):
    script, required, _, calls, build, render = outcome_pipeline
    def record(identifier):
        item = records().get(identifier.split("/")[0], _record(identifier))
        item["model_id"] = identifier
        return item
    monkeypatch.setattr(comparison, "get_model_record", record)
    flags = [*required, "--only-outcome-success", "--outcome-scoring-model-id", "outcome_scoring/a",
             "--outcome-conceding-model-id", "outcome_conceding/x", "--inference-mask-1", "all:accel",
             "--output-dir", str(tmp_path), "--run-id", "comparison"]
    args = script.parse_args(flags)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or ("yes" if accept else "no"))
    if accept:
        comparison.run_comparison(args, script)
        assert calls and render.called
    else:
        with pytest.raises(ComparisonCancelled):
            comparison.run_comparison(args, script)
        assert not calls and not build.called
    assert len(prompts) == 1
    saved = json.loads((tmp_path / "comparison/metadata.json").read_text())
    assert saved["status"] == ("completed" if accept else "cancelled")
    assert bool(saved["warning_acceptances"]) == accept


@pytest.mark.parametrize("module", ["run_relevant_models", "run_hawkeye", "run_hawkeye_loc",
                                   "run_skillcorner", "run_benchmark", "generate_epv",
                                   "run_and_visualize_hawkeye", "visualize_action_components"])
def test_inference_commands_expose_existing_acceptance_flag(module, capsys, monkeypatch):
    from importlib import import_module
    command = import_module("scripts." + module)
    monkeypatch.setattr("sys.argv", [module, "--help"])
    with pytest.raises(SystemExit) as error:
        command.parse_args()
    assert error.value.code == 0
    assert "--accept-comparison-warnings" in capsys.readouterr().out


def test_pc_attempt_preserves_cancelled_status(tmp_path, monkeypatch):
    from pc_xpass_inference_policy import inference_attempt, start_inference_attempt
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "no")
    @inference_attempt
    def run():
        start_inference_attempt(tmp_path, SimpleNamespace(pc_xpass=True, use_physical_xpass=True))
        policy = ComparisonWarnings()
        shared = utils.validate_model_record_consistency(records(), warning_collector=policy)
        confirm_outcome_warnings(policy, tmp_path, shared)
    with pytest.raises(ComparisonCancelled):
        run()
    saved = json.loads((tmp_path / "metadata.json").read_text())
    assert saved["status"] == "cancelled"
    assert saved["outcome_compatibility"]["comparison_cancelled"] is True


def test_acceptance_preserves_signed_and_absolute_inference(tmp_path):
    import torch
    from inference import inference_gnn
    from datatools.viz_helpers import compute_pass_score
    from test_pc_xpass_auxiliary import auxiliary_model
    from test_physical_xpass import make_physical_inference_match
    from test_signed_outcomes import configuration
    match = make_physical_inference_match([7, 8], [1, 1])
    for graph in match.graph_features_0:
        graph.x = torch.cat([graph.x, torch.zeros(graph.num_nodes, 1)], 1)
    scoring = auxiliary_model("outcome_scoring", model_id="outcome_scoring/fixture")
    scoring.args.update(configuration("scalar", "outcome_scoring"))
    conceding = auxiliary_model("outcome_conceding", model_id="outcome_conceding/fixture")
    scoring.forward = lambda batch: batch.x.new_tensor([-.3, .2]).expand(batch.num_nodes, 2)
    conceding.forward = lambda batch: torch.logit(batch.x.new_tensor([.7, .1])).expand(batch.num_nodes, 2)
    before = {"scoring": inference_gnn(match, scoring, device="cpu"),
              "conceding": inference_gnn(match, conceding, device="cpu")}
    policy = ComparisonWarnings(True)
    shared = utils.validate_model_record_consistency(records(), warning_collector=policy)
    confirm_outcome_warnings(policy, tmp_path, shared)
    after = {"scoring": inference_gnn(match, scoring, device="cpu"),
             "conceding": inference_gnn(match, conceding, device="cpu")}
    for task in before:
        for old, new in zip(before[task], after[task]):
            pd.testing.assert_frame_equal(old, new)
    sf, ss = after["scoring"]
    cf, cs = after["conceding"]
    assert (sf.stack() < 0).all() and (cf.stack() > 0).all()
    actual = compute_pass_score(ss * 0 + .8, ss, sf, cs, cf)
    expected = .8 * (ss - cs) + .2 * (sf - cf)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.stack().iloc[0] == pytest.approx(-.12)
