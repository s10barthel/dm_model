from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import config
from datatools.graph_feature import summarize_ball_trajectory
from datatools.pass_trajectory import (PassEndpointResolver, PASS_DURATION, ensure_pass_endpoints,
                                     load_official_events)
from datatools.trajectory_refresh import refresh_duration_labels
from datatools.utils import label_intended_receivers
from scripts.visualize_action_components import resolve_ball_velocity_xy
from scripts.visualization_receivers import resolve_pass_endpoint
from datatools.trajectory import TRAJECTORY_SPEED
from dataset import ActionDataset


def fixture(*, contact=.76, tag="OtherBallAction", success=False, reception=170, source="elastic"):
    frames = np.arange(100, 201)
    elapsed = (frames - 100) / 25
    tracking = pd.DataFrame(dict(period_id=1, episode_id=1, ball_state="alive", timestamp=elapsed,
                                 ball_x=elapsed * 10, ball_y=elapsed * 2, ball_z=elapsed,
                                 home_1_x=0., home_1_y=0., home_2_x=10., home_2_y=2.,
                                 home_3_x=10., home_3_y=20.), index=frames)
    actions = pd.DataFrame([dict(frame_id=100, receive_frame_id=reception, action_type="pass", spadl_type="pass",
                                 original_event_id="launch", period_id=1, object_id="home_1", receiver_id="away_1",
                                 success=success, endpoint_source=source, start_x=0., start_y=0.,
                                 end_x=999., end_y=999.)], index=[7])
    time = pd.Timestamp("2023-08-18T18:00:00Z")
    events = pd.DataFrame(dict(event_id=["launch", "same_time_annotation", "contact"],
                               utc_timestamp=[time, time, time + pd.Timedelta(seconds=contact)],
                               raw_tag=["Play", "Play", tag], frame_id=[100, 100, 100 + round(contact * 25)]))
    return SimpleNamespace(actions=actions, tracking=tracking, fps=25, official_pass_events=events)


@pytest.mark.parametrize("tag", ["OtherBallAction", "BallClaiming", "TacklingGame", "Play", "ShotAtGoal", "Foul", "Offside"])
def test_failed_pass_uses_synchronized_contact_instead_of_raw_delta(tag):
    match = fixture(tag=tag)
    match.official_pass_events.loc[2, "utc_timestamp"] += pd.Timedelta(seconds=15)
    ensure_pass_endpoints(match)
    action = match.actions.loc[7]
    assert action.observed_end_frame_id == action.trajectory_end_frame_id == 119
    assert action.pass_endpoint_event_id == "contact"
    assert action.pass_endpoint_source == ("official_stoppage" if tag in {"Foul", "Offside"} else "official_contact")
    assert action.pass_duration == .76
    assert action.receive_frame_id == 119
    assert action.original_receive_frame_id == 170
    assert action.pass_endpoint_definition == PASS_DURATION


def test_success_uses_earliest_official_endpoint():
    match = fixture(success=True, reception=150, source="kpi_csv", contact=3)
    ensure_pass_endpoints(match)
    assert match.actions.at[7, "observed_end_frame_id"] == 150
    assert match.actions.at[7, "trajectory_end_frame_id"] == 125
    assert match.actions.at[7, "pass_duration"] == 2
    assert match.actions.at[7, "pass_endpoint_source"] == "official_reception"
    match.actions.at[7, "endpoint_source"] = "elastic"
    assert match._pass_endpoint_resolver.resolve(match.actions.loc[7]).observed == 150  # Pinned endpoint wins.


@pytest.mark.parametrize("period_boundary", [False, True])
def test_invalid_endpoint_does_not_fall_back_to_arbitrary_boundary(period_boundary):
    match = fixture(contact=3, success=period_boundary, source="kpi_csv")
    if period_boundary:
        match.tracking.loc[140:, "period_id"] = 2
    else:
        match.tracking.loc[140:, ["ball_state", "episode_id"]] = ["dead", 0]
    ensure_pass_endpoints(match)
    assert pd.isna(match.actions.at[7, "observed_end_frame_id"])
    assert pd.isna(match.actions.at[7, "trajectory_end_frame_id"])
    assert pd.isna(match.actions.at[7, "pass_duration"])


def test_one_frame_segment_arrow_and_endpoint_share_tracking_inputs():
    match = fixture(contact=.04)
    summary = summarize_ball_trajectory(match, 7)
    np.testing.assert_allclose(summary, [np.sqrt(105), 10 / np.sqrt(105), 2 / np.sqrt(105), 1 / np.sqrt(105)])
    np.testing.assert_allclose(resolve_ball_velocity_xy(match, 7, "success_intent"), [10, 2])
    np.testing.assert_allclose(resolve_pass_endpoint(match, 7), [.4, .08])
    ensure_pass_endpoints(match)
    assert match.actions.at[7, "pass_duration"] == .04
    match.tracking.loc[101, ["ball_x", "ball_y"]] = np.nan
    assert summarize_ball_trajectory(match, 7) is None
    assert resolve_ball_velocity_xy(match, 7, "success_intent") is None
    assert resolve_pass_endpoint(match, 7) is None


def test_cap_and_candidate_positions_follow_same_trajectory_end():
    match = fixture(contact=2)
    ensure_pass_endpoints(match)
    match.tracking.loc[126:, "home_2_y"] = 100
    labels = label_intended_receivers(match.actions, match.tracking, action_type="pass", intended_receiver_mode="angle_only")
    assert labels.at[7, "intent_id"] == "home_2"
    assert match.actions.at[7, "pass_duration"] == 2
    np.testing.assert_allclose(resolve_pass_endpoint(match, 7), [20, 4])


def test_no_samples_and_zero_displacement_do_not_use_event_coordinates():
    match = fixture()
    match.tracking[["ball_x", "ball_y", "ball_z"]] = 0
    np.testing.assert_equal(summarize_ball_trajectory(match, 7), np.zeros(4))
    match.tracking[["ball_x", "ball_y"]] = np.nan
    assert summarize_ball_trajectory(match, 7) is None


def test_duplicate_duration_rows_change_only_duration():
    match = fixture(contact=.04)
    ensure_pass_endpoints(match)
    labels = torch.randn(5, len(config.LABEL_COLUMNS))
    labels[:, 0] = 7
    labels[:, config.LABEL_INDEX["is_pass"]] = torch.tensor([1, 1, 1, 0, 0])
    labels[3, config.LABEL_INDEX["is_shot"]] = 1  # Synthetic shot from the same pass.
    match.actions.loc[8] = match.actions.loc[7]
    match.actions.at[8, "action_type"] = "dribble"
    labels[4, 0] = 8
    labels[1, config.LABEL_INDEX["is_real"]] = 0
    labels[1, config.LABEL_INDEX["blocked"]] = 1
    before = labels.clone()
    updated = refresh_duration_labels(labels, match, context="duplicated labels")
    torch.testing.assert_close(updated[:4, 7], torch.full((4,), .04))
    for name in ("duration", "end_x", "end_y"):
        updated[:4, config.LABEL_INDEX[name]] = before[:4, config.LABEL_INDEX[name]]
    torch.testing.assert_close(updated, before, rtol=0, atol=0)
    torch.testing.assert_close(labels, before, rtol=0, atol=0)


def test_raw_xml_filters_administration_deletions_and_same_time_annotations(tmp_path):
    directory = tmp_path / "event_data"
    directory.mkdir()
    (directory / "DFL-MAT-test.xml").write_text('''<Root>
      <Event EventId="launch" EventTime="2023-01-01T00:00:00Z"><Play /></Event>
      <Event EventId="annotation" EventTime="2023-01-01T00:00:00Z"><Play /></Event>
      <Event EventId="admin" EventTime="2023-01-01T00:00:00.01Z"><Substitution /></Event>
      <Event EventId="deleted" EventTime="2023-01-01T00:00:00.02Z"><OtherBallAction /></Event>
      <Event EventId="deletion" EventTime="2023-01-01T00:00:01Z"><Delete EventId="deleted" /></Event>
      <Event EventId="contact" EventTime="2023-01-01T00:00:00.04Z"><OtherBallAction /></Event>
      </Root>''')
    files = SimpleNamespace(event_path=directory / "DFL-MAT-test.xml", kpi_path=directory / "kpi.csv")
    kpi = pd.DataFrame(dict(EVENT_ID=["launch", "annotation", "admin", "deleted", "contact"],
                            FRAME_NUMBER=[100, 100, 100, 100, 101], RECFRM=[None]*5))
    with patch("datatools.pass_trajectory._match_files", return_value={"DFL-MAT-test": files}), patch("scripts.preprocess_sportec.load_kpi_merged_table", return_value=kpi), patch(
        "scripts.preprocess_sportec.load_match_raw_events", return_value=pd.DataFrame(
            dict(event_id=kpi.EVENT_ID, period_id=1))), patch("pandas.read_parquet", return_value=pd.DataFrame(
                dict(period_id=[1, 1], raw_frame_id=[100, 101], frame_id=[100, 101]))):
        events = load_official_events("DFL-MAT-test")
    match = fixture()
    endpoint = PassEndpointResolver(match.tracking, 25, events).resolve(match.actions.loc[7])
    assert endpoint.observed == 101
    assert endpoint.event_id == "contact"


@pytest.mark.parametrize("task", ["action_intent", "pass_intent", "success_intent", "pass_success",
                                  "outcome_scoring", "outcome_conceding", "failure_receiver"])
@pytest.mark.parametrize("minimum,expected", [(0., 2), (.1, 1)])
def test_refreshed_duration_filters_every_component_model(tmp_path, task, minimum, expected):
    match = fixture(success=True, reception=101, source="kpi_csv", contact=3)
    match.actions.loc[8] = match.actions.loc[7]
    match.actions.at[8, "receive_frame_id"] = 103
    ensure_pass_endpoints(match)
    labels = torch.zeros((2, len(config.LABEL_COLUMNS)))
    labels[:, 0] = torch.tensor([7, 8])
    for name, value in dict(is_pass=1, n_players=3, intent_index=1, receiver_index=2, is_real=1).items():
        labels[:, config.LABEL_INDEX[name]] = value
    if task == "success_intent":
        labels[:, config.LABEL_INDEX["success"]] = 1
    labels = refresh_duration_labels(labels, match, context=task)
    x = torch.zeros((3, 24 if task == "success_intent" else config.NODE_FEATURE_CORE_DIM))
    x[:2, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros((2, 2)))
    graph.trajectory_speed_definition = TRAJECTORY_SPEED
    features, label_dir = tmp_path / "graphs", tmp_path / "labels"
    features.mkdir()
    label_dir.mkdir()
    torch.save([graph, graph.clone()], features / "match.pt")
    torch.save(labels, label_dir / "match.pt")
    dataset = ActionDataset(["match"], feature_dir=features, label_dir=label_dir, task=task,
                            min_pass_dur=minimum, trajectory_speed_definition=TRAJECTORY_SPEED,
                            extend_features=False, edge_in_dim=2)
    assert len(dataset) == expected
