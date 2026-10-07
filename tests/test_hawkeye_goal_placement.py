import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from datatools.hawkeye import _add_goal_nodes, build_hawkeye_situation
from datatools.viz_snapshot import SnapshotVisualizer
from physical_pass_model import physical_state_hash


def test_goal_positions_follow_each_rows_possession():
    tracking = _add_goal_nodes(pd.DataFrame({"ball_owning_home_away": ["home", "away"]}))
    assert tracking.home_goal_x.tolist() == [105, 0]
    assert tracking.away_goal_x.tolist() == [0, 105]
    for prefix in ["home", "away"]:
        assert tracking[f"{prefix}_goal_y"].tolist() == [34, 34]
        for feature in ["vx", "vy", "speed", "accel"]:
            assert (tracking[f"{prefix}_goal_{feature}"] == 0).all()


@pytest.mark.parametrize("value", [None, "", "unknown"])
def test_invalid_possession_is_rejected(value):
    with pytest.raises(ValueError, match="possession prefixes"):
        _add_goal_nodes(pd.DataFrame({"ball_owning_home_away": [value]}))


def test_missing_possession_is_rejected():
    with pytest.raises(ValueError, match="ball_owning_home_away"):
        _add_goal_nodes(pd.DataFrame({"ball_x": [0]}))


@pytest.mark.parametrize("team,prefix", [("A", "home"), ("B", "away")])
def test_situation_graph_has_attacking_goal_on_right(team, prefix):
    raw = pd.DataFrame([
        {"game_id": 1, "half": 1, "abs_time": time, "uefa_player_id": player,
         "role": 1, "centroid_x": x, "centroid_y": 0., "PlayerID": 1 if team == "A" else 2,
         "id": "s1", "team": name, "possession_team": team, "GameID": "G1"}
        for time in [0., .04, .08]
        for name, player, x in [("A", 1, -40.), ("B", 2, 40.)]
    ])
    ball = pd.DataFrame([{"game_id": 1, "half": 1, "abs_time": time,
                          "ball_x": 0., "ball_y": 0., "ball_z": 0.}
                         for time in [0., .04, .08]])
    situation, _, _ = build_hawkeye_situation(raw, ball, freeze_ballreceipt=False)
    graph = situation.graph_features_0[0]
    attacking = graph.x[(graph.x[:, 2] == 1) & (graph.x[:, 0] == 1)]
    defending = graph.x[(graph.x[:, 2] == 1) & (graph.x[:, 0] == 0)]
    assert attacking[:, 3:5].tolist() == [[105., 34.]]
    assert defending[:, 3:5].tolist() == [[0., 34.]]
    assert situation.tracking.at[0, f"{prefix}_goal_x"] == 105
    # Existing caches must distinguish the formerly incorrect away-goal geometry.
    if prefix == "away":
        legacy = graph.clone()
        goal_mask = legacy.x[:, 2] == 1
        legacy.x[goal_mask, 3] = 105 - legacy.x[goal_mask, 3]
        assert physical_state_hash(legacy) != physical_state_hash(graph)


@pytest.mark.parametrize("goal", ["home_goal", "away_goal"])
@pytest.mark.parametrize("x", [0., 105.])
@pytest.mark.parametrize("rotate", [False, True])
def test_goal_annotation_is_inside_displayed_pitch(goal, x, rotate):
    snapshot = pd.DataFrame({f"{goal}_x": [x], f"{goal}_y": [34.]})
    visualizer = SnapshotVisualizer(snapshot, player_annots=pd.Series({goal: .417}),
                                    show_velocities=False, style="pitchcontrol")
    fig, ax = visualizer.plot(rotate_pitch=rotate, annot_type="action_intent", show=False)
    try:
        fig.canvas.draw()
        annotation = next(text for text in ax.texts if text.get_text() == "0.417")
        displayed_x = 105 - x if rotate else x
        assert annotation.xy == (3 if displayed_x == 0 else 102, 34)
        assert ax.get_xlim()[0] < annotation.xy[0] < ax.get_xlim()[1]
        assert annotation.get_window_extent().width > 0
    finally:
        plt.close(fig)
