import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.empty_windows import DEFINITION, KEY, REASON_COLUMN, TARGETS, WindowAvailability
from datatools.empty_window_refresh import refresh_empty_windows
from datatools.stop_conditions import action_targets, definition, label_path
from datatools.xt_refresh import fingerprint
from project_config import get_action_label_dir, get_resolved_action_dir
from scripts.feature_extension_diagnostics import ExtensionAttempt
from tests.test_prepared_dataset import artifacts
from tests.test_time_returns import events_at


@pytest.mark.parametrize("return_type", ["next_2", "next_2_skip1", "next_2+", "extended_possession",
    "extended_possession_1", "time_3", "disc_0.9", "disc_max_0.9", "disc_poly_max_0.1_1"])
def test_policy_masks_both_sides_without_changing_goal_or_xg(return_type):
    if "+" in return_type:
        families = {"xt"}
    else:
        families = None
    source = events_at([0, 2, 3], types=["pass", "throw_in", "shot"])
    source.loc[2, "expected_goal"] = .3
    kwargs = dict(target_families=families)
    raw = action_targets(source, source, return_type, "set-pieces", {}, None, None, **kwargs)
    masked = action_targets(source, source, return_type, "set-pieces", {}, None, None,
                            exclude_empty_windows=True, **kwargs)
    for name in TARGETS:
        if name in masked:
            assert pd.isna(masked.loc[0, name])
            assert pd.isna(masked.loc[2, name])
    for name in ("scores", "concedes", "scores_xg", "concedes_xg"):
        if name in raw:
            pd.testing.assert_series_equal(masked[name], raw[name])
    assert masked.loc[0, REASON_COLUMN] == 1
    assert masked.loc[2, REASON_COLUMN] == 4


def test_structural_membership_keeps_zero_one_team_and_missing_annotations():
    source = events_at([0, 1, 2])
    source[["xT", "goal_distance", "epv"]] = 0.
    source.loc[1, ["xT", "goal_distance", "epv"]] = np.nan
    for ret in ("next_1", "next_100", "extended_possession", "time_1"):
        reasons = WindowAvailability(source).reasons(source, ret)
        assert reasons.tolist() == [0, 0, 4]
    source["xT"] = 0.
    result = action_targets(source, source, "extended_possession", "base", {}, None, None,
                            exclude_empty_windows=True)
    assert result.loc[0, "scores_xt"] == result.loc[0, "concedes_xt"] == 0


def test_skip_and_time_limits_are_distinct_from_boundaries_or_unknown_timing():
    source = events_at([0, 2, 4])
    w = WindowAvailability(source)
    assert w.reasons(source, "next_1_skip1").tolist() == [6, 6, 4]
    source.loc[1, "spadl_type"] = "shot"
    assert WindowAvailability(source).reasons(source, "next_1_skip1").tolist() == [0, 6, 4]
    assert w.reasons(source, "time_1").tolist() == [5, 5, 4]
    source.loc[1, "frame_id"] = np.nan
    assert WindowAvailability(source).reasons(source, "time_1").tolist() == [0, 0, 4]


def test_xt_gains_mask_after_baseline_subtraction_and_clipping():
    from datatools.xt_targets import LogProgressionSurface
    source = events_at([0, 1], types=["pass", "throw_in"])
    for clipping in (True, False):
        result = action_targets(source, source, "extended_possession", "set-pieces",
            dict(version=2, representation="gain", clip_negative_gains=clipping), LogProgressionSurface(), None,
            exclude_empty_windows=True)
        assert result[["scores_xt", "concedes_xt"]].isna().all().all()


def test_carries_use_saved_team_anchor_and_actual_time():
    source = events_at([0, 2, 4], teams=["home_1", "away_1", "away_1"])
    carry = source.iloc[[1]].copy()
    carry.index = [9]
    carry["original_event_id"] = "carry:1:0"
    carry["action_id"] = 9
    carry["object_id"] = "home_1"
    carry["return_event_index"] = 1
    carry["frame_id"] = 25
    w = WindowAvailability(source)
    assert w.reasons(carry, "time_1").tolist() == [0]
    assert w.reasons(carry, "extended_possession").tolist() == [0]


def test_extended_next_skip_uses_acquisition_context_for_membership():
    from datatools.xt_acquisitions import CONTEXT_KEY
    source = events_at([0, 1, 2], teams=["home_1", "away_1", "away_1"])
    assert WindowAvailability(source).reasons(source, "next_1+_skip1")[0] == 0
    source.attrs[CONTEXT_KEY] = [dict(before_event_index=1, period_id=1, object_id="away_1",
        spadl_type="interception", start_x=20., start_y=30., event_id="control")]
    assert WindowAvailability(source).reasons(source, "next_1+_skip1")[0] == 6


@pytest.fixture
def migration_source(tmp_path):
    root, event_dir = tmp_path / "source", tmp_path / "events"
    root.mkdir()
    event_dir.mkdir()
    returns = ["next_2", "next_1_skip1", "extended_possession", "time_1"]
    metadata = dict(run_id="source", status="completed", return_types=returns,
                    intended_receiver_modes=["original", "angle_only"], label_layout_version=1,
                    stop_conditions=dict(version=1, available={ret: ["base", "set-pieces"] for ret in returns}),
                    carry_variant=dict(available=True), split_manifest=dict(universe_count=4))
    (root / "metadata.json").write_text(json.dumps(metadata))
    source = events_at([0, 1, 2, 4], types=["pass", "throw_in", "pass", "pass"])
    source.loc[3, "period_id"] = 2
    for mid in ("m", "n", "o", "p"):
        source.to_csv(event_dir / f"{mid}.csv", index=False)
        for name in ("action_graphs", "post_action_graphs", "action_graphs_carries", "post_action_graphs_carries"):
            (root / name).mkdir(exist_ok=True)
            (root / name / f"{mid}.pt").write_bytes(b"graphs must not be loaded")
        for mode in metadata["intended_receiver_modes"]:
            for carries in (False, True):
                actions = source.copy()
                if carries:
                    carry = source.iloc[[2]].copy()
                    carry.index = [9]
                    carry["original_event_id"] = "carry:1:0"
                    carry["action_id"], carry["return_event_index"], carry["frame_id"] = 9, 2, 37
                    actions = pd.concat([actions, carry])
                    actions["original_event_id"] = actions.original_event_id.astype(str)
                action_dir = get_resolved_action_dir(mode, root=root, use_carries=carries)
                action_dir.mkdir(exist_ok=True)
                actions.to_parquet(action_dir / f"{mid}.parquet")
                for ret in returns:
                    for condition in ("base", "set-pieces"):
                        directory = get_action_label_dir(ret, mode, root=root, use_carries=carries)
                        path = label_path(directory / f"{mid}.pt", root, condition)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        labels = torch.full((len(actions), len(config.LABEL_COLUMNS) - 1), .3)
                        labels[:, 0] = torch.as_tensor(actions.index.to_numpy())
                        torch.save(labels, path)
    return root, event_dir, metadata


def test_parallel_migration_matches_serial_preserves_sources_and_is_idempotent(migration_source, tmp_path):
    from datatools.artifact_storage import clone_feature_artifacts
    source, event_dir, metadata = migration_source
    original = {p: fingerprint(p) for p in source.rglob("*") if p.is_file()}
    roots, metas, reports = [], [], []
    for workers in (1, 2):
        root = tmp_path / f"derived{workers}"
        clone_feature_artifacts(source, root)
        meta = copy.deepcopy(metadata)
        (root / "metadata.json").write_text(json.dumps(meta))
        reports.append(refresh_empty_windows(root, meta, event_dir=event_dir, num_workers=workers,
            attempt=ExtensionAttempt(root, "empty_window_policy", "derived", {})))
        roots.append(root)
        metas.append(meta)
        assert meta[KEY] == DEFINITION
        assert reports[-1]["completed_matches"] == 4
        assert reports[-1]["workers"] == workers
        assert reports[-1]["graph_files_loaded"] == reports[-1]["target_annotation_files_loaded"] == 0
        if workers == 2:
            assert len(reports[-1]["worker_pids"]) == 2
            assert os.getpid() not in reports[-1]["worker_pids"]
    assert metas[0] == metas[1]
    for path in roots[0].rglob("*.pt"):
        if "graphs" in path.parent.name:
            assert fingerprint(path) == fingerprint(roots[1] / path.relative_to(roots[0]))
            continue
        first, second = [torch.load(p, weights_only=True) for p in (path, roots[1] / path.relative_to(roots[0]))]
        torch.testing.assert_close(first, second, equal_nan=True, atol=0, rtol=0)
        old = torch.load(source / path.relative_to(roots[0]), weights_only=True)
        for name in ("scores", "scores_xg", "concedes", "concedes_xg", "scores_goal_next10"):
            assert torch.equal(first[:, config.LABEL_INDEX[name]], old[:, config.LABEL_INDEX[name]])
    repeated = refresh_empty_windows(roots[1], metas[1], event_dir=event_dir, num_workers=2,
        attempt=ExtensionAttempt(roots[1], "empty_window_policy", "derived", {}))
    assert repeated["rewritten_files"] == repeated["changed_rows"] == 0
    assert all(fingerprint(p) == digest for p, digest in original.items())
    assert not list(tmp_path.rglob("*.tmp"))


def test_worker_failure_publishes_nothing_and_cleans_staging(migration_source):
    root, event_dir, metadata = migration_source
    path = get_action_label_dir("time_1", root=root) / "p.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, 0] = -100
    torch.save(labels, path)
    original = {p: fingerprint(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Invalid action identities"):
        refresh_empty_windows(root, metadata, event_dir=event_dir, num_workers=2,
            attempt=ExtensionAttempt(root, "empty_window_policy", "derived", {}))
    assert KEY not in metadata
    assert all(fingerprint(p) == digest for p, digest in original.items())
    assert not list(root.rglob("*.tmp"))


def test_regular_refresh_preserves_policy_and_epv_certification(migration_source, tmp_path, monkeypatch):
    import project_config
    from datatools.stop_conditions import refresh_labels
    from datatools.epv_coverage import validate_labels
    root, event_dir, metadata = migration_source
    metadata[KEY] = DEFINITION
    (root / "metadata.json").write_text(json.dumps(metadata))
    sidecars = tmp_path / "xt"
    sidecars.mkdir()
    for name in ("goal_distance", "epv"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(project_config, name.upper() + "_MATCH_DIR", directory)
    for mid in ("m", "n", "o", "p"):
        source = pd.read_csv(event_dir / f"{mid}.csv")
        source[["action_id", "xT"]].assign(scores_xT=0., concedes_xT=0.).to_csv(sidecars / f"{mid}.csv", index=False)
        for name in ("goal_distance", "epv"):
            source[["action_id", name]].to_csv(tmp_path / name / f"{mid}.csv", index=False)
    refresh_labels(root, metadata, metadata["stop_conditions"]["available"], event_dir=event_dir, xt_match_dir=sidecars)
    (root / "metadata.json").write_text(json.dumps(metadata))
    labels_path = label_path(get_action_label_dir("extended_possession", root=root) / "m.pt", root, "set-pieces").parent
    for mid in ("m", "n", "o", "p"):
        labels = torch.load(labels_path / f"{mid}.pt", weights_only=True)
        assert labels[0, config.LABEL_INDEX[REASON_COLUMN]] == 1
        assert labels[0, config.LABEL_INDEX["scores_xt"]].isnan()
    validate_labels(root, metadata, ["m"], labels_path, "extended_possession", "set-pieces")
    repeated = refresh_empty_windows(root, metadata, event_dir=event_dir,
        attempt=ExtensionAttempt(root, "empty_window_policy", "derived", {}))
    assert repeated["rewritten_files"] == 0
    validate_labels(root, metadata, ["m"], labels_path, "extended_possession", "set-pieces")


def test_dataset_filters_only_selected_family_for_eager_and_prepared(artifacts, tmp_path):
    from dataset import ActionDataset
    from prepared_dataset import PreparedActionDataset
    (tmp_path / "metadata.json").write_text(json.dumps({KEY: DEFINITION}))
    options = dict(artifacts, task="outcome_scoring", require_goal_next10_diagnostics=False)
    path = options["label_dir"] / "m0.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, config.LABEL_INDEX[REASON_COLUMN]] = 1
    for name in TARGETS:
        labels[0, config.LABEL_INDEX[name]] = np.nan
    torch.save(labels, path)
    goal = ActionDataset(["m0"], **options, empty_window_target_family="goal")
    xt_options = dict(options, empty_window_target_family="xt")
    xt = ActionDataset(["m0"], **xt_options)
    prepared = PreparedActionDataset(["m0"], cache_dir=tmp_path / "cache", progress=False, **xt_options)
    assert len(xt) == len(goal) - 1 == len(prepared)
    assert xt.selection_counts["excluded_by_empty_window"] == 1
    assert xt.selection_counts["excluded_by_empty_window:set_piece"] == 1
    assert torch.isfinite(xt.labels[:, config.LABEL_INDEX["scores_xt"]]).all()


def test_dataset_rejects_unrecorded_nan_and_preserves_legacy_progression(artifacts, tmp_path):
    from dataset import ActionDataset, _normalize_label_width
    legacy = torch.zeros(1, config.LABEL_INDEX[REASON_COLUMN])
    legacy[0, config.LABEL_INDEX["scores_progression"]] = .7
    assert _normalize_label_width(legacy)[0, config.LABEL_INDEX["scores_progression"]] == .7
    (tmp_path / "metadata.json").write_text(json.dumps({KEY: DEFINITION}))
    path = artifacts["label_dir"] / "m0.pt"
    labels = torch.load(path, weights_only=True)
    labels[0, config.LABEL_INDEX["scores_goal_distance"]] = np.nan
    torch.save(labels, path)
    with pytest.raises(ValueError, match="outside recorded empty windows"):
        ActionDataset(["m0"], **dict(artifacts, task="outcome_scoring", require_goal_next10_diagnostics=False,
                                   empty_window_target_family="goal_distance"))


def test_definition_distinguishes_legacy_and_new_labels():
    meta = dict(stop_conditions=dict(version=1, available={"next_2": ["base"]}))
    assert KEY not in definition(meta, "next_2")
    meta[KEY] = DEFINITION
    assert definition(meta, "next_2")[KEY] == DEFINITION


def test_generation_matches_refresh_and_reuses_rows_without_losing_reasons():
    from datatools.label_reuse import reuse_label_rows
    from tests.test_time_returns import make_match
    source = events_at([0, 2, 3], types=["pass", "throw_in", "shot"])
    source.loc[2, "expected_goal"] = .3
    actions = source.assign(frame_id=0, receive_frame_id=0, receiver_id="out", intent_id="home_1",
                            action_type="pass", blocked=False, end_x=30., end_y=40.)
    match = make_match(source, actions)
    match.stop_condition = "set-pieces"
    match.exclude_empty_windows = True
    with reuse_label_rows(match):
        for ret in ("extended_possession", "time_3", "next_2"):
            generated = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=ret)
            expected = action_targets(actions, source, ret, "set-pieces", {}, None, None, exclude_empty_windows=True)
            for name in (*TARGETS, REASON_COLUMN, "scores", "scores_xg"):
                np.testing.assert_allclose(generated[:, config.LABEL_INDEX[name]], expected[name], equal_nan=True, rtol=1e-6)


def test_entrypoint_forwards_workers_and_publishes_policy_only_on_success(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from scripts import generate_relevant_features as generator
    from tests.test_feature_run_extension import make_args, make_metadata
    meta = make_metadata()
    meta.update(label_layout_version=1, stop_conditions=dict(version=1,
                available={ret: ["base"] for ret in meta["return_types"]}))
    args = make_args(num_workers="8", worker_thread_limit=2)
    args.refresh_empty_window_policy = True
    (tmp_path / "base").mkdir()
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "base")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: meta)
    monkeypatch.setattr(generator, "get_feature_run_root", lambda run_id: tmp_path / run_id)
    monkeypatch.setattr(generator, "write_latest_run", Mock())
    def refresh(root, metadata, **kwargs):
        assert KEY not in json.loads((root / "metadata.json").read_text())
        assert kwargs["num_workers"] == "8" and kwargs["worker_thread_limit"] == 2
        metadata[KEY] = DEFINITION
    monkeypatch.setattr("datatools.empty_window_refresh.refresh_empty_windows", refresh)
    generator.run_extension_generation(args)
    saved = json.loads((tmp_path / "derived" / "metadata.json").read_text())
    assert saved[KEY] == DEFINITION and saved["status"] == "completed"
    assert KEY not in meta


def test_cli_and_plan_dedicate_derived_refresh_and_forward_workers(monkeypatch):
    from scripts import generate_relevant_features as generator
    from tests.test_feature_run_extension import make_args, make_metadata, FeatureRunExtensionPlanTests
    args = make_args(num_workers="2", worker_thread_limit=3)
    args.refresh_empty_window_policy = True
    meta = make_metadata()
    meta.update(label_layout_version=1, stop_conditions=dict(version=1, available={"next_10": ["base"]}))
    plan = FeatureRunExtensionPlanTests().build_plan(args, meta)
    assert not plan.command_steps
    assert plan.extension_mode == "derived"
    args.in_place = True
    with pytest.raises(ValueError, match="derived run"):
        FeatureRunExtensionPlanTests().build_plan(args, meta)
    monkeypatch.setattr("sys.argv", ["generate", "--refresh-empty-window-policy"])
    with pytest.raises(SystemExit):
        generator.parse_args()
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: meta)
    monkeypatch.setattr("sys.argv", ["generate", "--extend-feature-run-id", "base",
                                    "--refresh-empty-window-policy", "--num-workers", "auto", "--worker-thread-limit", "2"])
    parsed = generator.parse_args()
    assert parsed.refresh_empty_window_policy and parsed.num_workers == "auto" and parsed.worker_thread_limit == 2
    monkeypatch.setattr("sys.argv", ["generate", "--extend-feature-run-id", "base",
                                    "--refresh-empty-window-policy", "--in-place"])
    with pytest.raises(SystemExit):
        generator.parse_args()
