import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from datatools.failed_pass_trajectory import detect, observation, KEY, VERSION


def sample(impact=12):
    frames = np.arange(28)
    xy = np.column_stack([np.minimum(frames, impact)*.4,
                          np.maximum(frames-impact, 0)*.4])
    tracking = pd.DataFrame(dict(timestamp=frames/25, period_id=1,
        ball_x=xy[:, 0], ball_y=xy[:, 1], ball_z=0.,
        away_1_x=impact*.4, away_1_y=0.), index=frames)
    action = pd.Series(dict(frame_id=0, original_event_id="a", action_type="pass",
                            object_id="home_1", success=False))
    return action, tracking


def test_peak_is_impact_not_proximity_entry():
    action, tracking = sample()
    result = detect(action, tracking, 25)
    assert result["status"] == "shortened"
    assert result["impact"] == 12
    assert result["cutoff"] == 11
    assert result["angle"] == pytest.approx(90)
    # Vertical tracking cannot change the XY detector.
    tracking.ball_z = np.nan
    assert detect(action, tracking, 25) == result


def test_guard_and_no_contact():
    action, tracking = sample(1)
    assert detect(action, tracking, 25)["status"] == "unavailable"
    assert detect(action, tracking, 25)["reason"] == "insufficient_clean_intervals"
    action, tracking = sample()
    tracking.away_1_x = 100
    assert detect(action, tracking, 25)["status"] == "unchanged"


@pytest.mark.parametrize("defect", ["gap", "time", "jump", "defender"])
def test_bad_data_does_not_create_confident_contact(defect):
    action, tracking = sample()
    if defect == "gap":
        tracking = tracking.drop(11)
    elif defect == "time":
        tracking.loc[11, "timestamp"] = 99
    elif defect == "jump":
        tracking.loc[11, "ball_y"] += 5
    else:
        tracking.away_1_x = np.nan
    assert detect(action, tracking, 25)["status"] == "unavailable"


def test_pinned_coverage_and_success_bypass(tmp_path):
    action, tracking = sample()
    match = SimpleNamespace(runtime_feature_root=tmp_path, match_id="m")
    (tmp_path / "metadata.json").write_text(json.dumps({KEY: VERSION}))
    with pytest.raises(ValueError, match="coverage"):
        observation(match, action, 25)
    folder = tmp_path / "failed_pass_trajectories"
    folder.mkdir()
    pd.DataFrame([dict(event_key="a", status="shortened", cutoff=11, original_cutoff=25,
                       policy=VERSION)]).to_parquet(folder / "m.parquet")
    assert observation(match, action, 25) == 11
    with pytest.raises(ValueError, match="Stale"):
        observation(match, action, 24)
    action.success = True
    assert observation(match, action, 25) == 25


def test_cli_requires_extension(monkeypatch):
    from scripts.generate_relevant_features import parse_args
    monkeypatch.setattr("sys.argv", ["generate", "--refresh-failed-pass-trajectories"])
    with pytest.raises(SystemExit):
        parse_args()


def test_contact_at_one_second_uses_confirmation_lookahead():
    action, tracking = sample(25)
    result = detect(action, tracking, 25)
    assert result["status"] == "shortened"
    assert result["impact"] == 25
    assert result["cutoff"] == 24
    # A period transition cannot supply confirmation for the previous period.
    tracking.loc[26:, "period_id"] = 2
    assert detect(action, tracking, 25)["status"] != "shortened"


def test_earliest_accepted_cluster_wins():
    action, tracking = sample(8)
    # A second, much larger reversal follows the first impact.
    tracking.loc[18:, "ball_y"] = 4. - (tracking.index[18:] - 18)*.8
    tracking["away_2_x"], tracking["away_2_y"] = 3.2, 4.
    assert detect(action, tracking, 25)["impact"] == 8


def test_away_pass_and_xyz_observation_and_stale_override(tmp_path, monkeypatch):
    from datatools.failed_pass_trajectory import prepare_new_match, validate_graph_coverage
    from datatools.graph_feature import summarize_ball_trajectory
    from torch_geometric.data import Data
    import torch
    action, tracking = sample()
    action.object_id = "away_1"
    tracking = tracking.rename(columns={"away_1_x": "home_1_x", "away_1_y": "home_1_y"})
    tracking.ball_z = tracking.index*.1
    actions = pd.DataFrame([action], index=[7])
    match = SimpleNamespace(runtime_feature_root=tmp_path, match_id="m", tracking=tracking,
                            actions=actions, fps=25, failed_pass_trajectory_policy=VERSION)
    resolver = SimpleNamespace(resolve=lambda a: SimpleNamespace(trajectory=25))
    monkeypatch.setattr("datatools.pass_trajectory.ensure_pass_endpoints", lambda *a, **k: resolver)
    monkeypatch.setattr("datatools.graph_feature.ensure_pass_endpoints", lambda *a, **k: resolver)
    prepare_new_match(match)
    assert observation(match, action, 25) == 11
    result = summarize_ball_trajectory(match, 7)
    assert result[1] < 0 and result[3] > 0  # Rotation retains positive Z.
    (tmp_path / "metadata.json").write_text(json.dumps({KEY: VERSION}))
    stale = Data(x=torch.ones((2, 24)))
    stale.trajectory_end_frame_id = 25
    with pytest.raises(ValueError, match="Stale"):
        validate_graph_coverage(match, [stale], [[7]])
    setattr(stale, KEY, VERSION)
    stale.trajectory_end_frame_id = 11
    validate_graph_coverage(match, [stale], [[7]])
