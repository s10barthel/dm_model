"""Possession windows and consistent generation/refresh across target families."""
import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.match import Match
from datatools.possession_returns import extended_possession_targets, label_extended_possession_events
from datatools.return_windows import PossessionWindows, action_xt_windows
from datatools.stop_conditions import action_targets, event_returns
from datatools.xt_targets import LogProgressionSurface, unavailable_xt_returns
from project_config import (get_action_label_dir, infer_feature_run_return_types, parse_return_type,
                            validate_return_type, validate_return_type_for_target_family)
from test_xt_refresh import sample


RETURN = "extended_possession"
FAMILIES = ("goal", "xg", "xt", "goal_distance", "epv", "progression")


def events(teams, types=None):
    n = len(teams)
    return pd.DataFrame(dict(action_id=range(n), original_event_id=range(n),
        object_id=[team + "_1" for team in teams], spadl_type=types or ["pass"] * n,
        period_id=[1] * n, success=[False] * n, expected_goal=[np.nan] * n,
        xT=np.linspace(.1, .9, n), goal_distance=np.linspace(.1, .9, n), epv=np.linspace(.1, .9, n),
        start_x=[20.] * n, start_y=[34.] * n))


@pytest.mark.parametrize("teams,anchor,expected", [
    (["home"] * 6 + ["away", "home"], 0, [1, 2, 3, 4, 5, 6]),
    (["home"] + ["away"] * 4 + ["home"], 0, [1, 2, 3, 4]),
    (["home"] * 6 + ["away", "home"], 3, [4, 5, 6]),
    (["home", "away", "home", "away"], 0, [1]),
    (["home", "away", "home", "away"], 1, [2]),
    (["home"] * 4, 0, [1, 2, 3]),
    (["home", "away", "away"], 0, [1, 2]),
    (["home"], 0, []),
])
def test_window_endpoints_and_failed_initial_action(teams, anchor, expected):
    source = events(teams)  # All passes fail; failure never changes a possession boundary.
    assert PossessionWindows(source).select(anchor, teams[anchor]).tolist() == expected


def test_noneligible_events_do_not_interrupt_and_missing_values_do_not_hide_a_turnover():
    source = events(["home", "away", "home", "away", "away", "home", "away"],
                    ["pass", "interception", "cross", "shot", "ball_carry", "pass", "pass"])
    source.loc[3, ["xT", "goal_distance", "epv"]] = np.nan
    assert PossessionWindows(source).select(0, "home").tolist() == [2, 3]
    result = label_extended_possession_events(source)
    for suffix in ("xT", "goal_distance", "epv"):
        assert result.loc[0, "scores_" + suffix] == pytest.approx(source.loc[2, suffix])
        assert result.loc[0, "concedes_" + suffix] == 0


def test_each_family_keeps_its_aggregation_and_initial_action_rules():
    source = events(["home", "home", "away", "away", "home"],
                    ["shot", "shot", "shot", "shot", "shot"])
    source["expected_goal"] = [.2, .3, .4, .5, .99]
    source.loc[3, "success"] = True
    source[["xT", "goal_distance", "epv"]] = np.array([.95, .25, .45, .55, .99])[:, None] * np.ones((1, 3))
    result = label_extended_possession_events(source)
    assert result.loc[0, "scores_xg"] == pytest.approx(1 - .8 * .7)
    assert result.loc[0, "concedes_xg"] == pytest.approx(1 - .6 * .5)
    assert result.loc[0, "scores"] == 0
    assert result.loc[0, "concedes"] == 1
    for suffix in ("xT", "goal_distance", "epv"):
        assert result.loc[0, "scores_" + suffix] == .25
        assert result.loc[0, "concedes_" + suffix] == .55
    assert result.loc[3, "scores"] == 1  # Initial goal is counted despite ending the segment.
    assert result.loc[3, "scores_xg"] == .5
    assert result.loc[3, "scores_xT"] == 0


@pytest.mark.parametrize("boundary", ["goal", "period", "goalkick", "throw_in"])
def test_stop_conditions_shorten_the_window_and_restart_can_start_its_own_window(boundary):
    source = events(["home", "home", "home", "away", "home"])
    if boundary == "goal":
        source.loc[1, ["spadl_type", "expected_goal", "success"]] = ["shot", .2, True]
    elif boundary == "period":
        source.loc[2:, "period_id"] = 2
    else:
        source.loc[2, "spadl_type"] = boundary
    condition = "set-pieces" if boundary == "throw_in" else "base"
    assert PossessionWindows(source, condition).select(0, "home").tolist() == [1]
    if boundary == "throw_in":
        assert PossessionWindows(source, "base").select(0, "home").tolist() == [1, 3]
    if boundary in {"throw_in", "goalkick"}:
        assert PossessionWindows(source, condition).select(2, "home").tolist() == [3]
    result = event_returns(source, RETURN, condition)
    assert result.loc[0, "scores_xT"] == pytest.approx(source.loc[1, "xT"])
    assert result.loc[0, "concedes_xT"] == 0


def test_empty_and_no_eligible_events_give_zero_returns():
    source = events(["home", "away"], ["interception", "tackle"])
    assert not label_extended_possession_events(source).filter(regex="^(scores|concedes)").to_numpy().any()
    empty = label_extended_possession_events(source.iloc[:0])
    assert empty.empty and "scores_xT" in empty


def test_events_without_an_actor_do_not_interrupt_a_possession():
    source = events(["home", "away", "home", "away", "home"],
                    ["pass", "other", "cross", "shot", "pass"])
    source.loc[1, "object_id"] = None
    labeled = label_extended_possession_events(source)
    assert labeled.loc[0, "scores_xT"] == source.loc[2, "xT"]
    assert labeled.loc[0, "concedes_xT"] == source.loc[3, "xT"]
    assert not labeled.loc[1].filter(regex="^(scores|concedes)").to_numpy().any()


@pytest.mark.parametrize("family", FAMILIES)
def test_validation_and_training_families(family):
    assert parse_return_type(RETURN) == (RETURN, None, False)
    assert validate_return_type_for_target_family(RETURN, family) == RETURN
    assert unavailable_xt_returns({"surface_type": "log_progression"}, [RETURN]) == []


@pytest.mark.parametrize("name", [RETURN + "_skip1", RETURN + "_spstop", RETURN + "_3"])
def test_no_unrequested_modifiers(name):
    with pytest.raises(ValueError):
        validate_return_type(name)


def test_paths_and_legacy_directory_discovery(tmp_path, monkeypatch):
    import project_config
    directory = get_action_label_dir(RETURN, root=tmp_path)
    directory.mkdir()
    monkeypatch.setattr(project_config, "get_feature_run_root", lambda *_: tmp_path)
    monkeypatch.setattr(project_config, "load_feature_run_metadata", lambda *a, **kw: {})
    assert infer_feature_run_return_types("fixture") == [RETURN]


def test_carry_uses_its_own_team_and_saved_anchor_for_all_families():
    source = events(["away", "away", "home", "away", "home"], ["pass", "shot", "shot", "shot", "shot"])
    source["expected_goal"] = [np.nan, .2, .8, .9, .99]
    carry = source.iloc[[0]].copy()
    carry.index = [20]
    carry["object_id"] = "home_1"
    carry["return_event_index"] = 0
    carry["original_event_id"] = "carry:1:0"
    carry["action_id"] = 20
    carry["action_type"] = "dribble"
    result = action_targets(carry, source, RETURN, "base", {}, None, None)
    for suffix, value in (("xt", "xT"), ("goal_distance", "goal_distance"), ("epv", "epv")):
        assert result.loc[20, "scores_" + suffix] == 0
        assert result.loc[20, "concedes_" + suffix] == pytest.approx(source.loc[1, value])
    assert result.loc[20, "scores_xg"] == 0
    assert result.loc[20, "concedes_xg"] == pytest.approx(.2)
    assert result.loc[20, "scores_progression"] == 0


@pytest.mark.parametrize("clipping", [False, True])
def test_xt_gains_use_acquisition_location_and_preserve_the_selected_window(clipping):
    from datatools.xt_targets import transform_action_targets
    source = events(["home", "home", "away", "away", "away", "home"],
                    ["pass", "pass", "interception", "pass", "pass", "pass"])
    source["xT"] = [.8, .4, np.nan, .2, .3, .99]
    source.loc[2, "start_x"] = 30.
    surface = LogProgressionSurface()
    pairs, baselines = action_xt_windows(source.iloc[:1], source, RETURN, surface, "base")
    np.testing.assert_allclose(pairs, [[.4, .3]])
    assert baselines[0] == pytest.approx(surface.predict_xy(75., 34.))
    settings = dict(version=2, representation="gain", clip_negative_gains=clipping)
    result = transform_action_targets(source.iloc[:1], pd.DataFrame(pairs), settings, surface,
                                      events=source, return_type=RETURN, stop_condition="base")
    expected = [.4 - surface.predict_xy(20., 34.), .3 - surface.predict_xy(75., 34.)]
    np.testing.assert_allclose(result.iloc[0], np.maximum(expected, 0) if clipping else expected)


@pytest.mark.parametrize("condition", ["base", "set-pieces"])
@pytest.mark.parametrize("representation", ["absolute", "gain"])
def test_full_label_generation_and_refresh_agree_for_every_target(condition, representation):
    source = events(["home", "home", "away", "away", "home", "away", "home"],
                    ["shot", "pass", "shot", "shot", "throw_in", "shot", "pass"])
    source["expected_goal"] = [.2, np.nan, .3, .4, np.nan, .5, np.nan]
    actions = source.copy().assign(frame_id=0, receive_frame_id=0, receiver_id="out", intent_id="home_1",
                                   action_type="shot", blocked=False, end_x=30., end_y=34.)
    carry = actions.iloc[[1]].copy()
    carry.index = [20]
    carry["action_id"], carry["original_event_id"] = 20, "carry:1:0"
    carry["return_event_index"], carry["object_id"] = 1, "away_1"
    carry["action_type"] = "dribble"
    actions = pd.concat([actions, carry])
    settings = dict(version=2, representation=representation, clip_negative_gains=representation == "gain")
    surface = LogProgressionSurface()
    metadata = dict(surface_type="log_progression", k=2, min_goal_distance=1)
    match = Match.__new__(Match)
    match.events, match.stop_condition, match.xt_surface_metadata = source, condition, metadata
    match.xt_targets, match.xt_surface = settings, surface
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.tracking = pd.DataFrame(dict(home_1_x=[1.], home_1_y=[2.], away_1_x=[3.], away_1_y=[4.]))
    labels = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False, return_type=RETURN)
    refreshed = action_targets(actions, source, RETURN, condition, settings, surface, metadata)
    columns = [name for name in config.LABEL_COLUMNS
               if name in refreshed and name.startswith(("scores", "concedes"))]
    tensor_columns = [config.LABEL_INDEX[name] for name in columns]
    torch.testing.assert_close(labels[:, tensor_columns],
        torch.tensor(refreshed[columns].to_numpy(), dtype=labels.dtype))


def test_long_possession_aggregates_without_rescanning_each_window():
    source = events(["home"] * 20000 + ["away"] * 20000 + ["home"])
    result = extended_possession_targets(source, source, families={"xt"})
    assert result.loc[0, "scores_xt"] == source.loc[19999, "xT"]
    assert result.loc[0, "concedes_xt"] == source.loc[39999, "xT"]
    assert result.loc[39999, "scores_xt"] == 0
    assert result.loc[39999, "concedes_xt"] == source.loc[40000, "xT"]


def test_saved_xt_and_goal_distance_refreshes_use_carry_team_across_all_modes(sample):
    from datatools.goal_distance_refresh import refresh_goal_distance
    from datatools.xt_refresh import fingerprint, refresh_xt
    from project_config import get_resolved_action_dir
    from scripts.feature_extension_diagnostics import ExtensionAttempt
    root, event_dir, sidecars, metadata, *_ = sample
    source = events(["home", "away", "home", "away", "home"])
    source["original_event_id"] = np.arange(100, 105)
    source.drop(columns="xT").to_csv(event_dir / "m.csv", index=False)
    source[["action_id", "xT"]].assign(scores_xT=0., concedes_xT=0.).to_csv(sidecars / "m.csv", index=False)
    for mode in metadata["intended_receiver_modes"]:
        for carry in (False, True):
            path = get_resolved_action_dir(mode, root=root, use_carries=carry) / "m.parquet"
            actions = pd.read_parquet(path).assign(period_id=1)
            actions.to_parquet(path)
    graphs = {p: fingerprint(p) for p in root.rglob("*.pt") if "graphs" in p.parent.name}
    refresh_xt(root, metadata, event_dir=event_dir, xt_match_dir=sidecars)
    source[["action_id", "goal_distance"]].to_csv(sidecars / "m.csv", index=False)
    refresh_goal_distance(root, metadata, target_families=("goal_distance", "progression"),
        event_dir=event_dir, goal_distance_match_dir=sidecars,
        attempt=ExtensionAttempt(root, "targets", "in_place", {}))
    for mode in metadata["intended_receiver_modes"]:
        path = get_action_label_dir(RETURN, mode, root=root, use_carries=True) / "m.pt"
        labels = torch.load(path, weights_only=True)
        carry = labels[labels[:, config.LABEL_INDEX["action_index"]] == 9]
        for suffix in ("xt", "goal_distance"):
            np.testing.assert_allclose(carry[:, [config.LABEL_INDEX["scores_" + suffix],
                                                config.LABEL_INDEX["concedes_" + suffix]]], [[.5, .7]])
    assert all(fingerprint(path) == digest for path, digest in graphs.items())
