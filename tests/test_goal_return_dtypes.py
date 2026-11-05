"""Goal/xG labels keep their values without pandas object downcasting."""
import numpy as np
import pandas as pd
import pytest

from datatools.return_windows import bounded_returns
from datatools.utils import label_returns


TARGETS = ["scores", "concedes", "scores_xg", "concedes_xg"]
pytestmark = pytest.mark.filterwarnings("error::FutureWarning")


@pytest.fixture(autouse=True, params=[False, True], ids=["current-pandas", "future-pandas"])
def downcasting_behavior(request):
    with pd.option_context("future.no_silent_downcasting", request.param):
        yield


def make_events(teams, types, xg, success, periods=None):
    n = len(teams)
    return pd.DataFrame(dict(
        object_id=[team + "_1" for team in teams], spadl_type=types,
        expected_goal=xg, success=success, period_id=periods or [1] * n,
    ), index=pd.Index(range(10, 10 + 2 * n, 2), name="event")).astype(dict(object_id=object, spadl_type=object))


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("xg_dtype", ["float32", "float64", "object"])
def test_short_segments_with_missing_xg_remain_zero_and_preserve_inputs(length, xg_dtype):
    events = make_events(["home"] * length, ["pass"] * length,
                         [np.nan] * length, [True] * length)
    events["expected_goal"] = events.expected_goal.astype(xg_dtype)
    original = events.copy(deep=True)
    result = label_returns(events, lookahead_len=10)
    pd.testing.assert_frame_equal(events, original, check_exact=True)
    pd.testing.assert_frame_equal(result[original.columns], original, check_exact=True)
    assert not result[TARGETS].to_numpy().any()
    assert all(dtype == np.dtype("float64") for dtype in result[TARGETS].dtypes)
    assert result.goal.dtype == np.dtype("bool") and not result.goal.any()


def test_empty_input_preserves_schema_without_warnings():
    events = make_events([], [], [], []).astype(dict(success=bool, expected_goal=float, period_id=int))
    result = label_returns(events, lookahead_len=10)
    assert result.empty and set(TARGETS) <= set(result)
    assert result.goal.dtype == np.dtype("bool")
    pd.testing.assert_frame_equal(result[events.columns], events)


def test_goal_and_xg_aggregation_preserves_team_perspective_and_period_boundary():
    events = make_events(
        ["home", "away", "home", "home", "away"],
        ["pass", "shot", "shot", "shot", "shot"],
        [np.nan, .4, .3, .2, .9], [True, True, False, True, True],
        periods=[1, 1, 1, 1, 2],
    )
    result = label_returns(events, lookahead_len=10)
    home_xg = 1 - (1 - .3) * (1 - .2)
    expected = np.array([
        [1, 1, home_xg, .4],
        [1, 1, .4, home_xg],
        [1, 0, home_xg, 0],
        [1, 0, .2, 0],
        [1, 0, .9, 0],
    ])
    np.testing.assert_allclose(result[TARGETS], expected, rtol=0, atol=1e-15)
    assert result.goal.tolist() == [False, True, False, True, True]
    assert result.index.equals(events.index)


@pytest.mark.parametrize("xg", [0., .25])
def test_current_action_goal_uses_existing_success_and_nonmissing_xg_rule(xg):
    events = make_events(["home"], ["shot"], [xg], [True])
    result = label_returns(events, lookahead_len=1)
    np.testing.assert_array_equal(result[TARGETS], [[1., 0., xg, 0.]])
    assert result.goal.dtype == np.dtype("bool") and bool(result.goal.iloc[0])


def test_multiple_goals_clip_to_one_while_xg_combines_probabilities():
    events = make_events(["home", "home"], ["shot", "shot"], [.2, .3], [True, True])
    result = label_returns(events, lookahead_len=10)
    np.testing.assert_allclose(result[TARGETS], [[1, 0, 1 - .8 * .7, 0], [1, 0, .3, 0]],
                               rtol=0, atol=1e-15)


@pytest.mark.parametrize("first_future_type", ["shot", "shot_freekick"])
@pytest.mark.parametrize("skip_first", [False, True])
def test_skip1_preserves_direct_shot_exception(first_future_type, skip_first):
    events = make_events(["home", "away", "home"], ["pass", first_future_type, "shot"],
                         [np.nan, .4, .2], [True, True, True])
    result = label_returns(events, lookahead_len=3, skip_first=skip_first)
    include_opponent = not skip_first or first_future_type == "shot"
    expected = [[1, int(include_opponent), .2, .4 if include_opponent else 0],
                [1, 1, .4, .2], [1, 0, .2, 0]]
    np.testing.assert_allclose(result[TARGETS], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("condition", ["base", "set-pieces"])
def test_short_stop_segments_respect_the_selected_boundary(condition):
    events = make_events(["home", "away", "away"], ["pass", "throw_in", "shot"],
                         [np.nan, np.nan, .4], [True, True, True])
    result = bounded_returns(events, condition, lambda part: label_returns(part, lookahead_len=10))
    include_opponent = condition == "base"
    np.testing.assert_allclose(result[TARGETS],
        [[0, int(include_opponent), 0, .4 if include_opponent else 0], [1, 0, .4, 0], [1, 0, .4, 0]],
        rtol=0, atol=1e-15)
    assert result.goal.dtype == np.dtype("bool")
