from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.match import Match
from datatools.time_returns import action_time_xt_windows, label_time_events, time_targets, timing_report
from datatools.stop_conditions import action_targets
from datatools.xt_targets import LogProgressionSurface, transform_action_targets, unavailable_xt_returns
from project_config import parse_return_type, resolve_requested_return_types, validate_return_type_for_target_family
from tests.test_prepared_dataset import artifacts


def events_at(times, teams=None, types=None):
    n = len(times)
    return pd.DataFrame(dict(action_id=np.arange(n), original_event_id=100 + np.arange(n),
        object_id=teams or ["home_1"] * n, period_id=[1] * n,
        frame_id=np.asarray(times, dtype=float) * 25, spadl_type=types or ["pass"] * n,
        success=[True] * n, expected_goal=[np.nan] * n,
        xT=np.linspace(.1, .9, n), goal_distance=np.linspace(.1, .9, n), epv=np.linspace(.1, .9, n),
        start_x=[10.] * n, start_y=[20.] * n))


@pytest.mark.parametrize("value,expected", [("time_10", 10.), ("time_2.500", 2.5), (" time_01.0 ", 1.), ("time_0.04", .04)])
def test_parse(value, expected):
    assert parse_return_type(value) == ("time", expected, False)
    for family in ("goal", "xg", "xt", "goal_distance", "epv", "progression"):
        assert validate_return_type_for_target_family(value, family).startswith("time_")
    assert unavailable_xt_returns(dict(surface_type="log_progression"), [value]) == []


@pytest.mark.parametrize("value", ["time_0", "time_-1", "time_nan", "time_inf", "time_", "time_2_skip1", "time_2+", "time_2_spstop", "time_1_0", "time_1e-999"])
def test_invalid_durations(value):
    with pytest.raises(ValueError):
        parse_return_type(value)


def test_equivalent_names_deduplicate():
    assert resolve_requested_return_types(["time_10", "time_10.0", "time_010.000"]) == ["time_10"]


def test_inclusive_cutoff_future_maxima_and_initial_xg():
    events = events_at([0, 2, 5, 5.04], ["home_1", "home_1", "away_1", "home_1"])
    events.expected_goal = [.2, .3, .4, .9]
    events.success = False  # Avoid goal boundaries for this probability example.
    result = time_targets(events.iloc[:1], events, 5)
    assert result.iloc[0].scores_xg == pytest.approx(1 - .8 * .7)
    assert result.iloc[0].concedes_xg == pytest.approx(.4)
    for family in ("xt", "goal_distance", "epv"):
        assert result.iloc[0]["scores_" + family] == pytest.approx(events.iloc[1]["xT"])
        assert result.iloc[0]["concedes_" + family] == pytest.approx(events.iloc[2]["xT"])
    assert time_targets(events.iloc[:1], events, .04).iloc[0].scores_xt == 0


def test_ties_nonmonotonic_frames_and_canonical_forward_order():
    events = events_at([0, 6, 2, 2, 2.04, 20], ["home_1", "home_1", "home_1", "away_1", "home_1", "home_1"])
    result = time_targets(events.iloc[[2]], events, .04)
    assert result.iloc[0].scores_xt == pytest.approx(events.xT.iloc[4])
    assert result.iloc[0].concedes_xt == pytest.approx(events.xT.iloc[3])
    assert time_targets(events.iloc[:1], events, 2).iloc[0].scores_xt == pytest.approx(events.xT.iloc[2])


@pytest.mark.parametrize("boundary", ["goal", "goalkick", "period", "set-pieces"])
def test_boundaries_and_restart_anchor(boundary):
    events = events_at([0, 1, 2, 3])
    condition = "base"
    if boundary == "goal":
        events.loc[1, ["spadl_type", "expected_goal"]] = ["shot", .5]
    elif boundary == "period":
        events.loc[2:, "period_id"] = 2
    else:
        events.loc[2, "spadl_type"] = "goalkick" if boundary == "goalkick" else "throw_in"
        condition = "set-pieces" if boundary == "set-pieces" else "base"
    result = time_targets(events.iloc[:1], events, 30, condition)
    assert result.iloc[0].scores_xt == pytest.approx(events.xT.iloc[1])
    assert result.iloc[0].scores == (1 if boundary == "goal" else 0)
    if boundary in {"goalkick", "set-pieces"}:
        assert time_targets(events.iloc[[2]], events, 30, condition).iloc[0].scores_xt == pytest.approx(events.xT.iloc[3])


def test_missing_candidate_propagates_nan_within_boundary_and_recovers_afterward():
    events = events_at([0, np.nan, 2, 3, 4])
    events.loc[3, "spadl_type"] = "goalkick"
    result = time_targets(events, events, 10)
    assert result.iloc[:2].isna().all().all()
    assert result.iloc[2:].notna().all().all()
    assert timing_report(events, result) == dict(actions=5, missing_timing_actions=1, missing_return_actions=2)
    refreshed = action_targets(events, events, "time_10", "base", {}, None, None)
    assert refreshed.iloc[:2].scores_progression.isna().all()
    assert refreshed.attrs["time_return_coverage"]["missing_return_actions"] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 2.5, "invalid"])
def test_invalid_anchor_time_is_nan(bad):
    events = events_at([0, 1])
    actions = events.iloc[:1].copy()
    actions["frame_id"] = bad
    assert time_targets(actions, events, 5).isna().all().all()


def test_missing_clock_never_falls_back_to_recorded_time():
    events = events_at([0, 1]).drop(columns="frame_id").assign(seconds=[0, 1], utc_timestamp="2026-01-01")
    assert time_targets(events, events, 5).isna().all().all()


def test_carry_clock_starts_before_its_terminal_return_anchor():
    events = events_at([0, 3, 10], ["home_1", "away_1", "home_1"])
    carry = events.iloc[[0]].copy()
    carry.index = [9]
    carry["original_event_id"], carry["return_event_index"], carry["frame_id"] = "carry:1:0", 2, 50
    result = time_targets(carry, events, 2)
    assert result.iloc[0].concedes_xt == pytest.approx(events.xT.iloc[1])
    assert result.iloc[0].scores_xt == 0
    pairs, _ = action_time_xt_windows(carry, events, 2)
    np.testing.assert_allclose(pairs, result[["scores_xt", "concedes_xt"]])


def test_time_gain_uses_opponent_acquisition_and_keeps_nan():
    events = events_at([0, .5, 1, 2], ["home_1", "away_1", "away_1", "home_1"],
                       ["pass", "interception", "pass", "pass"])
    events.loc[1, "start_x"] = 80
    surface = LogProgressionSurface()
    settings = dict(version=2, representation="gain", clip_negative_gains=True)
    raw, baselines = action_time_xt_windows(events.iloc[:1], events, 1, surface)
    assert baselines[0] == pytest.approx(surface.predict_xy(config.FIELD_SIZE[0] - 80, config.FIELD_SIZE[1] - 20))
    result = transform_action_targets(events.iloc[:1], pd.DataFrame(raw, columns=["scores_xt", "concedes_xt"]),
                                     settings, surface, events=events, return_type="time_1")
    assert result.iloc[0].concedes_xt == pytest.approx(max(0, raw[0, 1] - baselines[0]))
    events.loc[2, "frame_id"] = np.nan
    result = transform_action_targets(events.iloc[:1], result, settings, surface, events=events, return_type="time_1")
    assert result.isna().all().all()


def test_carry_does_not_borrow_terminal_restarts_segment():
    events = events_at([0, 3, 5, 6], types=["pass", "pass", "goalkick", "pass"])
    carry = events.iloc[[0]].copy()
    carry.index = [9]
    carry["return_event_index"], carry["frame_id"] = 2, 50
    result = time_targets(carry, events, 20)
    assert result.iloc[0].scores_xt == pytest.approx(events.xT.iloc[1])


def test_empty_timeline():
    events = events_at([])
    assert label_time_events(events, 5).empty


def make_match(events, actions):
    match = Match.__new__(Match)
    match.events = events.copy()
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.stop_condition = "base"
    match.tracking = pd.DataFrame({"home_1_x": [1.], "home_1_y": [2.], "away_1_x": [3.], "away_1_y": [4.]}, index=[0])
    return match


def test_full_generation_refresh_parity_and_nan_diagnostics(tmp_path, monkeypatch):
    from datatools.stop_conditions import refresh_labels
    from project_config import get_action_label_dir, get_resolved_action_dir
    import project_config
    events = events_at([0, 1, np.nan, 4, 5])
    # Only the first/last actions need snapshots; intermediate events remain candidates.
    actions = events.iloc[[0, 4]].assign(receive_frame_id=0, receiver_id="out", intent_id="home_1",
        action_type="pass", blocked=False, end_x=30., end_y=40.)
    actions.loc[4, "frame_id"] = 0  # Available snapshot, independent of the candidate clock.
    match = make_match(events, actions)
    generated = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type="time_3")
    expected = action_targets(actions, events, "time_3", "base", {}, None, None)
    names = [name for name in expected if name.startswith(("scores", "concedes"))]
    columns = [config.LABEL_INDEX[name] for name in names]
    np.testing.assert_allclose(generated[:, columns], expected[names], rtol=1e-6, equal_nan=True)
    assert torch.isnan(generated[0, config.LABEL_INDEX["scores_xt"]])
    assert torch.isfinite(generated[:, config.LABEL_INDEX["scores_goal_next10"]]).all()

    root, event_dir, xt_dir = [tmp_path / name for name in ("features", "events", "xt")]
    for path in (root, event_dir, xt_dir):
        path.mkdir()
    events.drop(columns=["xT", "goal_distance", "epv"]).to_csv(event_dir / "m.csv", index=False)
    events[["action_id", "xT"]].assign(scores_xT=0., concedes_xT=0.).to_csv(xt_dir / "m.csv", index=False)
    for family in ("goal_distance", "epv"):
        directory = tmp_path / family
        directory.mkdir()
        events[["action_id", family]].to_csv(directory / "m.csv", index=False)
        monkeypatch.setattr(project_config, family.upper() + "_MATCH_DIR", directory)
    for name in ("action_graphs", "post_action_graphs"):
        directory = root / name
        directory.mkdir()
        (directory / "m.pt").write_bytes(b"graph payload is never loaded")
    action_dir = get_resolved_action_dir("original", root=root)
    action_dir.mkdir()
    actions.to_parquet(action_dir / "m.parquet")
    label_dir = get_action_label_dir("time_3", "original", root=root)
    label_dir.mkdir()
    torch.save(generated, label_dir / "m.pt")
    metadata = dict(return_types=["time_3"], intended_receiver_modes=["original"])
    report = refresh_labels(root, metadata, {"time_3": ["base", "set-pieces"]}, event_dir=event_dir, xt_match_dir=xt_dir)
    assert report["time_return_totals"][0]["missing_return_actions"] == 1
    assert metadata["time_returns"]["clock"] == "frame_id / 25"
    actual = torch.load(get_action_label_dir("time_3", "original", root=root) / "m.pt", weights_only=True)
    torch.testing.assert_close(actual[:, columns], generated[:, columns], equal_nan=True)
    repeated = refresh_labels(root, metadata, {"time_3": ["base", "set-pieces"]}, event_dir=event_dir, xt_match_dir=xt_dir)
    assert repeated["changed_rows"] == 0
    xt_report = refresh_labels(root, metadata, {"time_3": ["base", "set-pieces"]},
        event_dir=event_dir, xt_match_dir=xt_dir, xt_only=True)
    assert xt_report["time_return_totals"][0]["missing_return_actions"] == 1
    assert xt_report["changed_rows"] == 0
    from datatools.xt_refresh import refresh_xt
    xt_report = refresh_xt(root, metadata, event_dir=event_dir, xt_match_dir=xt_dir)
    assert xt_report["time_return_totals"][0]["missing_return_actions"] == 1
    from datatools.goal_distance_refresh import refresh_goal_distance
    from scripts.feature_extension_diagnostics import ExtensionAttempt
    gd_report = refresh_goal_distance(root, metadata, event_dir=event_dir,
        goal_distance_match_dir=tmp_path / "goal_distance", target_families=("goal_distance", "progression"),
        attempt=ExtensionAttempt(root, "targets", "in_place", {}))
    assert gd_report["time_return_totals"][0]["missing_return_actions"] == 1


def test_dataset_configuration_filters_only_timed_outcome_targets():
    from models.dataset_config import build_action_dataset_kwargs
    args = SimpleNamespace(task="outcome_scoring", return_type="time_5", use_xt=True)
    assert build_action_dataset_kwargs(args, train=True, diagnostic_label_dir=None)["finite_outcome_target_family"] == "xt"
    args.return_type = "next_5"
    assert build_action_dataset_kwargs(args, train=True, diagnostic_label_dir=None)["finite_outcome_target_family"] is None


def test_return_extension_schedules_time_labels_for_all_carry_modes():
    from tests.test_feature_run_extension import FeatureRunExtensionPlanTests, make_args, make_metadata
    metadata = make_metadata(["next_10"], ["original", "angle_only"])
    metadata["carry_variant"] = dict(available=True)
    plan = FeatureRunExtensionPlanTests().build_plan(make_args(["time_2.5"]), metadata)
    assert plan.final_return_types == ["next_10", "time_2.5"]
    assert len(plan.command_steps) == 5
    assert all("--labels-only" in step.command for step in plan.command_steps)
    assert sum("--use-carries" in step.command for step in plan.command_steps) == 2


def test_training_eager_and_prepared_exclude_only_missing_selected_pair(artifacts, tmp_path):
    from dataset import ActionDataset
    from prepared_dataset import PreparedActionDataset
    artifacts = dict(artifacts, task="outcome_scoring", require_goal_next10_diagnostics=False)
    label_path = artifacts["label_dir"] / "m0.pt"
    labels = torch.load(label_path, weights_only=True)
    labels[0, config.LABEL_INDEX["scores_xt"]] = np.nan
    labels[1, config.LABEL_INDEX["scores_goal_distance"]] = np.nan
    torch.save(labels, label_path)
    reference = ActionDataset(["m0"], **artifacts)
    options = dict(artifacts, finite_outcome_target_family="xt")
    eager = ActionDataset(["m0"], **options)
    prepared = PreparedActionDataset(["m0"], cache_dir=tmp_path / "cache", progress=False, **options)
    assert len(eager) == len(reference) - 1
    assert len(prepared) == len(eager)
    assert eager.selection_counts["excluded_by_missing_time_return"] == 1
    assert torch.isfinite(eager.labels[:, [config.LABEL_INDEX["scores_xt"], config.LABEL_INDEX["concedes_xt"]]]).all()
