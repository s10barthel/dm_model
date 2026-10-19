"""Visualization provenance must never turn absent evidence into a holdout claim."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from models import membership, utils
from models.evaluation_context import check_heldout


def record(identifier="pass_success/a", **metadata):
    return {"model_id": identifier, "metadata": {
        "train_match_ids": ["train"], "validation_match_ids": ["valid"], **metadata}}


@pytest.mark.parametrize("match,status", [("train", "training"), ("valid", "validation"), ("test", "held_out")])
def test_recorded_membership_and_actual_contribution(match, status):
    item = record(split_datasets={"train": {"requested_match_ids": ["train"], "contributing_match_ids": []}})
    evidence = membership.resolve_membership_evidence(item)
    result = membership.describe_match_membership(evidence, "sportec", match)
    assert result["status"] == status
    assert result["contributing"]["training"] is False
    assert result["contributing"]["validation"] is None


def test_development_membership_does_not_invent_training_or_validation():
    item = {"model_id": "m", "split_manifest_id": "saved"}
    loader = Mock(return_value={"train": ["development"], "test": ["test"]})
    evidence = membership.resolve_membership_evidence(item, manifest_loader=loader)
    result = membership.describe_match_membership(evidence, "sportec", "development")
    assert result["status"] == "development"
    assert result["membership"] == {"training": None, "validation": None, "development": True}
    assert membership.describe_match_membership(evidence, "sportec", "test")["status"] == "held_out"


@pytest.mark.parametrize("dataset", ["hawkeye", "benchmark", "skillcorner", "unknown"])
def test_native_provider_ids_are_not_canonical(dataset):
    evidence = membership.resolve_membership_evidence(record())
    # Even a coincidentally equal string has no verified correspondence.
    for identifier in ("train", "test"):
        result = membership.describe_match_membership(evidence, dataset, identifier)
        assert result["status"] == "not_checked"
        assert result["check_status"] == "not_checked"
        assert result["canonical_match_id"] is None
        assert all(value is None for value in result["membership"].values())


def test_partial_membership_preserves_known_positive_evidence():
    evidence = membership.resolve_membership_evidence({"metadata": {"train_match_ids": ["train"]}})
    assert membership.describe_match_membership(evidence, "sportec", "train")["status"] == "training"
    assert membership.describe_match_membership(evidence, "sportec", "test")["status"] == "unknown"
    with pytest.raises(ValueError, match="missing recorded membership"):
        check_heldout({"metadata": {"train_match_ids": ["train"]}}, ["test"])


@pytest.mark.parametrize("metadata", [["invalid"], {"split_datasets": ["invalid"]},
                                      {"split_datasets": {"train": ["invalid"]}},
                                      {"train_match_ids": "invalid"}])
def test_malformed_membership_is_informational_for_visualization(metadata):
    item = {"model_id": "m", "metadata": metadata}
    evidence = membership.resolve_membership_evidence(item)
    assert membership.describe_match_membership(evidence, "sportec", "test")["status"] == "unknown"
    with pytest.raises(ValueError):
        membership.resolve_membership_evidence(item, strict=True)


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), ValueError("invalid fingerprint")])
def test_missing_or_corrupt_manifest_prevents_holdout_claim(error):
    item = record(split_manifest_id="saved")
    loader = Mock(side_effect=error)
    evidence = membership.resolve_membership_evidence(item, manifest_loader=loader)
    assert membership.describe_match_membership(evidence, "sportec", "test")["status"] == "unknown"
    assert membership.describe_match_membership(evidence, "sportec", "train")["status"] == "training"
    with pytest.raises(type(error)):
        membership.resolve_membership_evidence(item, strict=True, manifest_loader=loader)


@pytest.mark.parametrize("change", [
    {"split_datasets": {"train": {"requested_match_ids": ["other"]}}},
    {"split_datasets": {"train": {"contributing_match_ids": ["other"]}}},
    {"split_manifest_id": "saved"},
])
def test_conflicting_evidence_is_not_reported_as_held_out(change):
    item = record(**change)
    if "split_manifest_id" in change:
        item["split_manifest_id"] = "different"
    evidence = membership.resolve_membership_evidence(item, manifest_loader=lambda _: {"train": ["train", "valid"]})
    assert membership.describe_match_membership(evidence, "sportec", "test")["status"] == "unknown"
    assert evidence["errors"]


def test_refit_uses_recorded_full_training_membership():
    item = record(train_match_ids=["train", "valid"], validation_match_ids=[])
    result = membership.describe_match_membership(membership.resolve_membership_evidence(item), "sportec", "valid")
    assert result["status"] == "training"
    assert result["membership"]["validation"] is False


def test_common_holdout_accepts_different_training_memberships():
    first = record("pass_success/a", train_match_ids=["1", "2"], validation_match_ids=["3"])
    second = record("pass_success/b", train_match_ids=["1", "2", "3"], validation_match_ids=["4"])
    for item in (first, second):
        assert check_heldout(item, ["5", "6"])["overlap_count"] == 0
    with pytest.raises(ValueError, match="overlaps"):
        check_heldout(second, ["3", "5"])


@pytest.mark.parametrize("difference", ["count", "selector", "manifest", "missing", "bundle"])
def test_relaxed_selection_preserves_per_model_splits(monkeypatch, difference):
    first = {"train_count": 10, "split_manifest_id": "first"}
    second = dict(first)
    bundle = dict(first)
    if difference == "count":
        second["train_count"] = 20
    elif difference == "selector":
        second = {"train_split_percent": 50, "split_manifest_id": "first"}
    elif difference == "manifest":
        second["split_manifest_id"] = "second"
    elif difference == "missing":
        second = {"args": {}, "metadata": {}, "train_split_percent": 50}
    elif difference == "bundle":
        bundle = {"train_count": 99, "split_manifest_id": "bundle"}
    records = {"a": first, "b": second}
    original = deepcopy(records)
    monkeypatch.setattr(utils, "resolve_bundle_model_ids", lambda *a, **kw: ({"a": "a/1", "b": "b/1"}, bundle))
    monkeypatch.setattr(utils, "get_model_records", lambda *_: records)
    monkeypatch.setattr(utils, "validate_model_record_consistency", lambda *a, **kw: {})
    _, context, _ = utils.resolve_model_selection(["a", "b"], bundle_id="bundle", require_shared_split=False)
    assert context["source_splits"]["a"]["train_count"] == 10
    if difference != "bundle":
        assert not set(context).intersection({"train_count", "train_split_percent", "split_manifest_id"})
    else:
        assert context["train_count"] == 10 and context["split_manifest_id"] == "first"
    assert records == original
    if difference != "missing":
        with pytest.raises(ValueError, match="split"):
            utils.resolve_model_selection(["a", "b"], bundle_id="bundle")


def test_legacy_default_is_not_explicit_provenance():
    assert membership.explicit_split_provenance({"args": {}, "metadata": {}, "train_split_percent": 50}) == {
        "train_split_percent": None, "train_count": None, "split_manifest_id": None}


def test_report_caches_evidence_and_prints_once_per_match(monkeypatch, capsys):
    load = Mock(return_value=record())
    monkeypatch.setattr(utils, "get_model_record", load)
    reporter = membership.MembershipReport()
    source = {"models": {"pass_success": "pass_success/a"}}
    report = reporter.build("sportec", [("action_1", "train"), ("action_2", "train")], source)
    reporter.print(report)
    reporter.print(report)
    assert load.call_count == 1
    assert capsys.readouterr().out.count("Model membership |") == 1
    assert len(report["entries"]) == 2
    assert all(item["status"] == "training" for item in report["entries"])


def test_manifest_evidence_is_cached_across_models(monkeypatch):
    loader = Mock(return_value={"train": ["train", "valid"], "test": ["test"]})
    monkeypatch.setattr(membership, "load_recorded_split_manifest", loader)
    records = {role: record(role + "/a", split_manifest_id="shared")
               for role in ("pass_success", "outcome_scoring")}
    source = {"models": {role: item["model_id"] for role, item in records.items()}, "model_records": records}
    result = membership.MembershipReport().build("sportec", [("test", "test")], source)
    assert len(result["entries"]) == 2
    assert all(item["status"] == "held_out" for item in result["entries"])
    loader.assert_called_once_with("shared")


@pytest.mark.parametrize("version", [1, 2])
def test_archived_snapshot_is_reused_without_checkpoint(monkeypatch, version):
    reporter = membership.MembershipReport()
    source = {"models": {"pass_success": "pass_success/a"}, "model_records": {"pass_success": record()}}
    snapshot = reporter.build("sportec", [("original", "test")], source)
    snapshot["version"] = version
    if version == 1:
        snapshot.pop("check_status")
        snapshot.pop("dataset")
        for entry in snapshot["entries"]:
            entry.pop("check_status")
    unavailable = Mock(side_effect=AssertionError("No checkpoint lookup for a saved snapshot"))
    monkeypatch.setattr(utils, "get_model_record", unavailable)
    archived = {"models": source["models"], "model_membership": snapshot}
    result = membership.MembershipReport().build("sportec", [("new_action", "test")], archived)
    assert result["entries"][0]["status"] == "held_out"
    assert result["entries"][0]["scene_id"] == "new_action"
    assert result["version"] == 2 and result["check_status"] == "checked"
    unavailable.assert_not_called()


def test_archived_run_without_membership_or_checkpoint_is_unknown(monkeypatch):
    monkeypatch.setattr(utils, "get_model_record", Mock(side_effect=FileNotFoundError("archived")))
    metadata = {"match_id": "test", "rendered_components": ["pass_success"]}
    membership.attach_membership_report(metadata, "sportec", source={"models": {"pass_success": "pass_success/a"}})
    entry = metadata["model_membership"]["entries"][0]
    assert entry["status"] == "unknown"
    assert any("Checkpoint unavailable" in reason for reason in entry["reasons"])


def test_derived_components_recipient_and_auxiliary_roles(monkeypatch):
    ids = {task: task + "/a" for task in ("pass_success", "outcome_scoring", "outcome_conceding", "success_intent", "pass_height")}
    monkeypatch.setattr(utils, "get_model_record", lambda identifier: record(identifier))
    source = {"models": ids, "xpass_auxiliaries": {"pass_height": {"model_id": "pass_height/aux"}}}
    metadata = {"match_id": "test", "rendered_components": ["pass_score", "intended_recipient", "physical_xpass"]}
    report = membership.attach_membership_report(metadata, "sportec", source=source)
    assert {item["role"] for item in report["models"]} == {
        "pass_success", "outcome_scoring", "outcome_conceding", "success_intent", "xpass_pass_height"}
    assert all(item["status"] == "held_out" for item in report["entries"])


def test_tracking_only_report_is_empty(capsys):
    metadata = {"rendered_situation_ids": ["s"], "source_models": {}, "rendered_components": []}
    report = membership.attach_membership_report(metadata, "hawkeye")
    assert report == {"version": 2, "dataset": "hawkeye", "check_status": "not_checked",
                      "reason": "no_verified_sportec_match_id", "models": [], "entries": []}
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("dataset,metadata,expected", [
    ("sportec", {"match_id": "m"}, [("m", "m")]),
    ("skillcorner", {"processed_matches": ["1"]}, [("1", "1")]),
    ("benchmark", {"processed_states": [{"modification": 1, "game_state": 2}]}, [("modification_1_game_state_2", None)]),
    ("benchmark", {"rendered_modifications": [{"modification": 1, "game_states": [1, 2]}]},
     [("modification_1_game_state_1", None), ("modification_1_game_state_2", None)]),
    ("hawkeye", {"processed_situation_ids": ["s"]}, [("s", None)]),
])
def test_producer_and_visualization_identities(dataset, metadata, expected):
    assert membership.displayed_identities(metadata, dataset) == expected


def test_comparison_reports_each_variant_and_keeps_sources_unchanged(monkeypatch):
    source = {"models": {"pass_success": "pass_success/a"}, "model_records": {"pass_success": record()}}
    original = deepcopy(source)
    metadata = {"script": "visualize_action_components.py", "arguments": {"match_id": "test"},
                "rendered_components": ["pass_success"], "variants": [
                    {"id": "a", "provenance": {"metadata": source}},
                    {"id": "b", "provenance": {"metadata": source}}],
                "scenes": [{"scene_id": "action", "variant_id": variant} for variant in ("a", "b")]}
    report = membership.attach_comparison_membership(metadata)
    assert {item["variant_id"] for item in report["entries"]} == {"a", "b"}
    assert all(item["status"] == "held_out" for item in report["entries"])
    assert source == original


@pytest.mark.parametrize("dataset", ["hawkeye", "skillcorner", "benchmark", "unknown"])
@pytest.mark.parametrize("snapshot_version", [None, 1, 2])
def test_native_reports_skip_evidence_and_normalize_saved_reports(monkeypatch, capsys, dataset, snapshot_version):
    unavailable = Mock(side_effect=AssertionError("No membership lookup for native IDs"))
    monkeypatch.setattr(utils, "get_model_record", unavailable)
    monkeypatch.setattr(membership, "load_recorded_split_manifest", unavailable)
    monkeypatch.setattr(membership, "load_feature_run_metadata", unavailable)
    source = {"models": {"pass_success": "pass_success/a", "pass_intent": "pass_intent/b"},
              "xpass_auxiliaries": {"pass_height": {"model_id": "pass_height/c"}}}
    if snapshot_version is not None:
        source["model_membership"] = {
            "version": snapshot_version, "models": [{"role": "pass_success", "model_id": "pass_success/a"}],
            "entries": [{"dataset": dataset, "scene_id": "train", "match_id": "train",
                         "role": "pass_success", "model_id": "pass_success/a", "status": "unknown"}]}
    original = deepcopy(source)
    reporter = membership.MembershipReport()
    report = reporter.build(dataset, [("train", "train"), ("another", None)], source)
    assert report["version"] == 2
    assert report["check_status"] == "not_checked"
    assert report["reason"] == "no_verified_sportec_match_id"
    assert report["entries"] == []
    assert {item["role"] for item in report["models"]} == {"pass_success", "pass_intent", "xpass_pass_height"}
    reporter.print(report)
    reporter.print(report)
    assert capsys.readouterr().out.strip() == membership.SKIPPED_MESSAGE
    unavailable.assert_not_called()
    assert source == original


@pytest.mark.parametrize("identifier", [None, "", " "])
def test_sportec_missing_identity_is_unchecked_without_evidence_lookup(monkeypatch, identifier):
    lookup = Mock(side_effect=AssertionError("Missing identity needs no checkpoint"))
    monkeypatch.setattr(utils, "get_model_record", lookup)
    result = membership.MembershipReport().build("sportec", [("scene", identifier)],
                                                {"models": {"pass_success": "pass_success/a"}})
    assert result["check_status"] == "not_checked"
    assert result["entries"][0]["status"] == "not_checked"
    assert result["entries"][0]["canonical_match_id"] is None
    lookup.assert_not_called()


def test_sportec_mixed_identities_preserve_checked_entries_and_skip_notice(monkeypatch, capsys):
    lookup = Mock(return_value=record())
    monkeypatch.setattr(utils, "get_model_record", lookup)
    reporter = membership.MembershipReport()
    result = reporter.build("sportec", [("first", None), ("second", "test")],
                            {"models": {"pass_success": "pass_success/a"}})
    assert result["check_status"] == "partially_checked"
    assert [item["status"] for item in result["entries"]] == ["not_checked", "held_out"]
    reporter.print(result)
    reporter.print(result)
    output = capsys.readouterr().out
    assert output.count(membership.SKIPPED_MESSAGE) == 1
    assert output.count("Model membership |") == 1
    lookup.assert_called_once_with("pass_success/a")


def test_missing_evidence_prints_once_per_model_across_scenes_roles_and_variants(monkeypatch, capsys):
    lookup = Mock(side_effect=FileNotFoundError("archived"))
    monkeypatch.setattr(utils, "get_model_record", lookup)
    reporter = membership.MembershipReport()
    source = {"models": {"pass_success": "pass_success/a", "pass_height": "pass_success/a",
                         "outcome_scoring": "outcome_scoring/b"},
              "xpass_auxiliaries": {"pass_height": {"model_id": "pass_success/a"}}}
    reports = [reporter.build("sportec", [(name, name) for name in ("one", "two", "three")], source,
                              variant_id=variant) for variant in ("a", "b")]
    for report in reports + reports:
        reporter.print(report)
    output = capsys.readouterr().out
    assert output.count("Model membership |") == 2
    assert output.count("missing recorded membership") == 2
    assert output.count("Checkpoint unavailable") == 2
    assert lookup.call_count == 2
    assert all(item["status"] == "unknown" and len(item["reasons"]) == 2
               for report in reports for item in report["entries"])


def test_saved_report_only_preserves_model_roles_for_native_dataset(monkeypatch):
    lookup = Mock(side_effect=AssertionError("Archived report must not load weights"))
    monkeypatch.setattr(utils, "get_model_record", lookup)
    source = {"model_membership": {"version": 1,
              "models": [{"role": "pass_success", "model_id": "pass_success/a"}], "entries": []}}
    report = membership.MembershipReport().build("hawkeye", [("native", None)], source)
    assert report["models"] == source["model_membership"]["models"]
    assert report["entries"] == [] and report["check_status"] == "not_checked"
    lookup.assert_not_called()


@pytest.mark.parametrize("explicit_auxiliaries", [False, True])
def test_archived_auxiliary_roles_are_preserved_unless_current_settings_override(monkeypatch, explicit_auxiliaries):
    lookup = Mock(side_effect=AssertionError("Native membership must not load checkpoints"))
    monkeypatch.setattr(utils, "get_model_record", lookup)
    saved_models = [{"role": "pass_success", "model_id": "pass_success/a"},
                    {"role": "xpass_pass_intent", "model_id": "pass_intent/old"}]
    source = {"source_models": {"pass_success": "pass_success/a"},
              "model_membership": {"version": 1, "models": saved_models, "entries": []}}
    if explicit_auxiliaries:
        source["xpass_auxiliaries"] = {"pass_intent": {"model_id": "pass_intent/current"}}
    report = membership.MembershipReport().build("hawkeye", [("native", None)], source, roles={"pass_success"})
    expected = "pass_intent/current" if explicit_auxiliaries else "pass_intent/old"
    assert {item["role"]: item["model_id"] for item in report["models"]} == {
        "pass_success": "pass_success/a", "xpass_pass_intent": expected}
    lookup.assert_not_called()
