"""Incomplete EPV is optional for features and an error for selected EPV jobs."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.epv import merge_epv_annotations
from datatools.epv_coverage import (ATTRIBUTE, DIRECTORY, POLICY, VERSION, TARGETS,
                                   load_annotations, validate_labels, verified_report, label_fingerprint)
from datatools.match import Match
from datatools.stop_conditions import action_targets, refresh_labels
from datatools.xt_refresh import fingerprint
from models.utils import get_outcome_targets
from project_config import get_action_label_dir, parse_return_type
from tests.test_xt_refresh import sample
from tests.test_training_controls import synthetic_training


def events():
    return pd.DataFrame(dict(action_id=[10, 11, 12, 13], object_id=["home_1", "away_1", "away_1", "home_1"],
        spadl_type=["pass", "goalkick", "shot", "cross"], period_id=1, expected_goal=[np.nan, np.nan, .3, np.nan],
        success=[True, True, False, True]), index=[2, 4, 6, 8])


@pytest.mark.parametrize("problem, reason", [
    ("none", None), ("missing_file", "missing_file"), ("missing_row", "missing_required_actions"),
    ("nan", "nonfinite_required_values"), ("infinity", "nonfinite_required_values"),
    ("duplicate", "invalid_or_duplicate_action_ids"), ("unknown", "unknown_action_ids"),
    ("missing_column", "missing_columns"), ("empty_file", "malformed_file"),
])
def test_shared_coverage_only_requires_candidate_rows(tmp_path, problem, reason):
    source = events()
    side = pd.DataFrame(dict(action_id=[10, 12, 13], epv=[.2, .3, .5]))
    if problem == "missing_row":
        side = side.iloc[1:]
    elif problem in {"nan", "infinity"}:
        side.loc[0, "epv"] = np.nan if problem == "nan" else np.inf
    elif problem == "duplicate":
        side = pd.concat([side, side.iloc[:1]])
    elif problem == "unknown":
        side.loc[0, "action_id"] = 999
    elif problem == "missing_column":
        side = side.drop(columns="epv")
    if problem != "missing_file":
        side.to_csv(tmp_path / "m.csv", index=False)
    if problem == "empty_file":
        (tmp_path / "m.csv").write_text("")
    values, report = load_annotations(source, "m", tmp_path)
    assert report["complete"] is (problem == "none")
    assert values.index.equals(source.index)
    assert report["required_actions"] == 3
    if reason:
        assert reason in report["reasons"] and values.isna().all()
    else:
        np.testing.assert_array_equal(values.loc[[2, 6, 8]], [.2, .3, .5])
        assert np.isnan(values.loc[4]) and report["ignored_missing_events"] == 1
        assert report["finite_actions"] == 3 and report["annotation_sha256"]
    merged = merge_epv_annotations(source.assign(scores_epv=99., concedes_epv=99.), "m", tmp_path)
    pd.testing.assert_frame_equal(merged[source.columns], source)
    assert merged.attrs[ATTRIBUTE] == report
    assert "scores_epv" not in merged  # Returns must be computed for the requested definition.


@pytest.mark.parametrize("ret", ["next_3", "in_1", "disc_0.9", "disc_max_0.9", "extended_possession", "next_3+"])
@pytest.mark.parametrize("condition", [None, "base", "set-pieces"])
def test_normal_generation_and_refresh_mask_only_epv_for_carries(sample, tmp_path, ret, condition):
    _, _, _, _, source, _, actions = sample
    source = source.assign(goal_distance=[.2, .4, .8], start_x=10., start_y=20.)
    merged = merge_epv_annotations(source, "missing", tmp_path)
    complete = source.copy()
    complete["epv"] = [.1, .3, .5]
    settings = dict(surface_type="learned")
    outputs = []
    for e in (merged, complete):
        match = Match.__new__(Match)
        match.events, match.stop_condition, match.xt_surface_metadata = e, condition, settings
        match.action_type, match.fps, match.include_goals = "all", 25, False
        match.tracking = pd.DataFrame(dict(home_1_x=[1.], home_1_y=[2.], away_1_x=[3.], away_1_y=[4.]))
        outputs.append(match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=ret))
    columns = [config.LABEL_INDEX[name] for name in TARGETS]
    assert outputs[0][:, columns].isnan().all()
    other = [i for i in range(len(config.LABEL_COLUMNS)) if i not in columns]
    torch.testing.assert_close(outputs[0][:, other], outputs[1][:, other], rtol=0, atol=0, equal_nan=True)
    refreshed = action_targets(actions, merged, ret, condition, {}, None, settings)
    assert refreshed[list(TARGETS)].isna().all().all()


@pytest.mark.parametrize("missing", [True, False])
def test_migration_replaces_stale_epv_with_nan_and_reports_coverage(sample, tmp_path, monkeypatch, missing):
    import project_config
    root, event_dir, sidecars, metadata, source, _, _ = sample
    goal_dir, epv_dir = tmp_path / "distance", tmp_path / "epv"
    goal_dir.mkdir(); epv_dir.mkdir()
    source[["action_id"]].assign(goal_distance=[.1, .4, .8]).to_csv(goal_dir / "m.csv", index=False)
    if not missing:
        source[["action_id"]].assign(epv=[.1, np.nan, .8]).to_csv(epv_dir / "m.csv", index=False)
    monkeypatch.setattr(project_config, "GOAL_DISTANCE_MATCH_DIR", goal_dir)
    monkeypatch.setattr(project_config, "EPV_MATCH_DIR", epv_dir)
    graphs = {p: fingerprint(p) for p in root.rglob("*.pt") if "graphs" in p.parent.name}
    requests = {ret.removesuffix("_spstop"): ["base", "set-pieces"] for ret in metadata["return_types"]}
    report = refresh_labels(root, metadata, requests, event_dir=event_dir, xt_match_dir=sidecars)
    assert metadata[ATTRIBUTE]["unavailable_matches"] == 1
    assert len(report[ATTRIBUTE]) == 1 and not report[ATTRIBUTE][0]["complete"]
    record = json.loads((root / DIRECTORY / "m.json").read_text())
    assert record["verified_variants"]["next_3"] == ["base", "set-pieces"]
    for path in (root / "labels").rglob("*.pt"):
        labels = torch.load(path, weights_only=True)
        assert labels[:, [config.LABEL_INDEX[name] for name in TARGETS]].isnan().all()
    assert all(fingerprint(path) == before for path, before in graphs.items())
    (root / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="m:.*missing_file|m:.*nonfinite_required_values"):
        validate_labels(root, metadata, ["m"], get_action_label_dir("next_3", root=root), "next_3")
    repeated = refresh_labels(root, metadata, requests, event_dir=event_dir, xt_match_dir=sidecars)
    assert repeated["rewritten_files"] == 0


def certified_run(tmp_path):
    root = tmp_path / "run"
    records = root / DIRECTORY
    records.mkdir(parents=True)
    metadata = {ATTRIBUTE: dict(version=VERSION, policy=POLICY)}
    (root / "metadata.json").write_text(json.dumps(metadata))
    directory = get_action_label_dir("next_3", "original", root=root)
    directory.mkdir()
    side = tmp_path / "annotations"
    side.mkdir()
    source = events()
    pd.DataFrame(dict(action_id=[10, 12, 13], epv=[0., .3, .5])).to_csv(side / "good.csv", index=False)
    for mid in ("good", "bad"):
        _, report = load_annotations(source, mid, side)
        path = directory / f"{mid}.pt"
        torch.save(torch.zeros((2, len(config.LABEL_COLUMNS))), path)
        (records / f"{mid}.json").write_text(json.dumps(verified_report(root, report, {"next_3": ["base"]},
            {path.relative_to(root).as_posix(): label_fingerprint(torch.load(path, weights_only=True))})))
    return root, directory, metadata, side


def test_training_checks_only_selected_matches_and_certified_variants(tmp_path):
    root, directory, metadata, _ = certified_run(tmp_path)
    assert validate_labels(root, metadata, ["good"], directory, "next_3")["selected_matches"] == 1
    with pytest.raises(ValueError, match="bad: missing_file"):
        validate_labels(root, metadata, ["good", "bad"], directory, "next_3")
    with pytest.raises(ValueError, match="unverified EPV labels"):
        validate_labels(root, metadata, ["good"], directory, "next_5")
    with pytest.raises(ValueError, match="unverified"):
        validate_labels(root, {}, ["good"], directory, "next_3")
    path = directory / "good.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, config.LABEL_INDEX["scores_xt"]] = .9
    torch.save(labels, path)
    validate_labels(root, metadata, ["good"], directory, "next_3")
    labels[0, config.LABEL_INDEX["scores_epv"]] = .9
    torch.save(labels, path)
    with pytest.raises(ValueError, match="unverified or changed"):
        validate_labels(root, metadata, ["good"], directory, "next_3")
    labels[0, config.LABEL_INDEX["scores_epv"]] = np.nan
    torch.save(labels, path)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_labels(root, metadata, ["good"], directory, "next_3")


def test_changed_annotations_clear_certifications_for_older_variants(tmp_path):
    root, _, _, side = certified_run(tmp_path)
    source = events()
    _, report = load_annotations(source, "good", side)
    unchanged = verified_report(root, report, {"next_5": ["base"]})
    assert set(unchanged["verified_variants"]) == {"next_3", "next_5"}
    pd.DataFrame(dict(action_id=[10, 12, 13], epv=[.2, .3, .5])).to_csv(side / "good.csv", index=False)
    _, changed = load_annotations(source, "good", side)
    assert set(verified_report(root, changed, {"next_5": ["base"]})["verified_variants"]) == {"next_5"}


@pytest.mark.parametrize("width, value", [(2, 0.), (len(config.LABEL_COLUMNS), np.nan), (len(config.LABEL_COLUMNS), np.inf)])
def test_epv_batch_guard_rejects_missing_or_invalid_targets(width, value):
    labels = torch.full((2, width), value)
    with pytest.raises(ValueError, match="EPV targets"):
        get_outcome_targets(labels, SimpleNamespace(use_epv=True))


@pytest.mark.parametrize("loading", ["memory", "disk"])
def test_direct_training_rejects_unverified_epv_before_dataset_loading(synthetic_training, loading):
    run, observations, _, _, _ = synthetic_training
    with pytest.raises(ValueError, match="EPV coverage is unverified"):
        run("unverified_epv", ["--use_epv", "--dataset-loading", loading])
    assert not observations


def test_wrapper_checks_requested_training_stages_without_unused_matches(tmp_path, monkeypatch):
    from scripts import train_relevant_models as wrapper
    root, directory, metadata, side = certified_run(tmp_path)
    (side / "valid.csv").write_bytes((side / "good.csv").read_bytes())
    _, record = load_annotations(events(), "valid", side)
    (root / DIRECTORY / "valid.json").write_text(json.dumps(verified_report(root, record, {"next_3": ["base"]})))
    target = root / "labels" / "base" / directory.name
    target.mkdir(parents=True)
    for mid in ("good", "valid", "bad"):
        (target / f"{mid}.pt").write_bytes((directory / ("good.pt" if mid == "valid" else f"{mid}.pt")).read_bytes())
        path = root / DIRECTORY / f"{mid}.json"
        record = json.loads(path.read_text())
        label = target / f"{mid}.pt"
        record["verified_labels"][label.relative_to(root).as_posix()] = label_fingerprint(torch.load(label, weights_only=True))
        path.write_text(json.dumps(record))
    metadata.update(label_layout_version=1, stop_conditions=dict(version=1, available={"next_3": ["base"]}))
    (root / "metadata.json").write_text(json.dumps(metadata))
    manifest = dict(manifest_id="fixture", train=["good", "bad", "valid"], test=["unselected"])
    monkeypatch.setattr(wrapper, "resolve_feature_run_id", lambda *a, **kw: "run")
    monkeypatch.setattr(wrapper, "resolve_feature_root", lambda *a, **kw: root)
    monkeypatch.setattr(wrapper, "load_feature_run_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(wrapper, "resolve_training_split", lambda *a: (manifest, "fixture"))
    monkeypatch.setattr(wrapper, "infer_feature_run_intended_receiver_modes", lambda _: ["original"])
    monkeypatch.setattr(wrapper, "infer_feature_run_return_types", lambda _: ["next_3"])
    flags = ["--feature-run-id", "run", "--only-outcome-scoring", "--target-family", "epv", "--return_type", "next_3",
             "--intended-receiver-mode", "original", "--validation-count", "1", "--training-counts", "1"]
    args = wrapper.parse_args(flags)
    assert args.validation_stages[0]["train_match_ids"] == ["good"]
    with pytest.raises(SystemExit):
        wrapper.parse_args([*flags[:-1], "2"])


def test_evaluation_checks_its_selected_cohort_before_graph_loading(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from models import evaluation_data as evaluation
    root, directory, _, _ = certified_run(tmp_path)
    context = dict(feature_root=str(root), feature_dir=str(directory), label_dir=str(directory),
        diagnostic_label_dir=None, evaluation_diagnostic_label_dir=None, pass_height_diagnostic_label_dir=None,
        label_definitions=dict(stop_condition_definition=dict(mode="base")))
    monkeypatch.setattr(evaluation, "resolve_evaluation_data", lambda *a: copy.deepcopy(context))
    schema = Mock(return_value={})
    monkeypatch.setattr(evaluation, "infer_feature_graph_schema", schema)
    monkeypatch.setattr(evaluation, "validate_checkpoint_edge_schema", lambda *a: {})
    request = dict(model_id="outcome_scoring/test", model_args=dict(task="outcome_scoring", use_epv=True, return_type="next_3"),
                   cli_args=SimpleNamespace(), match_ids=["good"])
    evaluation.preflight_evaluation_data([request])
    schema.reset_mock()
    with pytest.raises(evaluation.EvaluationDataError, match="bad: missing_file"):
        evaluation.preflight_evaluation_data([dict(request, match_ids=["bad"])])
    schema.assert_not_called()


def test_adding_carries_finalizes_available_stop_variants_and_coverage(tmp_path, monkeypatch):
    from scripts import generate_relevant_features as generator
    from tests.test_feature_run_extension import make_args, make_metadata
    metadata = make_metadata(["next_3"])
    metadata[generator.PASS_DURATION_KEY] = generator.PASS_DURATION
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "base")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(generator, "get_feature_run_root", lambda name: tmp_path / name)
    monkeypatch.setattr(generator, "run_generation_steps", lambda *a, **kw: None)
    monkeypatch.setattr(generator, "write_latest_run", lambda *a: None)
    calls = []
    def refresh(root, refreshed, requests, **kwargs):
        assert refreshed["carry_variant"]["available"]
        assert kwargs["num_workers"] == "2"
        calls.append(requests)
        refreshed[ATTRIBUTE] = dict(version=VERSION, policy=POLICY)
    monkeypatch.setattr("datatools.stop_conditions.refresh_labels", refresh)
    generator.run_carry_extension_generation(make_args(num_workers="2"))
    result = json.loads((tmp_path / "derived" / "metadata.json").read_text())
    assert result[ATTRIBUTE]["policy"] == POLICY and result["carry_variant"]["available"]
    assert calls == [metadata["stop_conditions"]["available"]]
