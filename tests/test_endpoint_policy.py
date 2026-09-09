from types import SimpleNamespace
from unittest.mock import patch

import argparse
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools.endpoint_policy import nonnegative_duration, valid_interval
from datatools import graph_feature
from datatools import config
from dataset import ActionDataset, _copy_pass_height_diagnostics
from datatools.match import Match
from datatools.ball_carries import derive_carry_segments, validate_carry_artifact_version
from models.dataset_config import build_action_dataset_kwargs
from scripts import preprocess_sportec as pre
from scripts import train_relevant_models as wrapper
from test_graph_feature_regressions import BallCarryRegressionTests
from test_success_intent_mode_independent import make_training_args, make_enabled_tasks


def tracking():
    return pd.DataFrame({"period_id": [1] * 5 + [2], "episode_id": [1] * 4 + [0, 2],
                         "ball_state": ["alive"] * 4 + ["dead", "alive"],
                         "ball_z": [0., 1., 2., 3., 0., 0.], "ball_x": range(6),
                         "ball_y": [0.] * 6, "ball_accel": [0.] * 6, "ball_vz": [0.] * 6},
                        index=pd.Index(range(10, 16), name="frame_id"))


def events(restart="throw_in"):
    return pd.DataFrame({"action_id": [0, 1], "original_event_id": ["pass", "restart"],
                         "seconds": [0., 1.], "period_id": [1, 1],
                         "frame_id": pd.array([10, 14], dtype="Int64"),
                         "receive_frame_id": pd.array([9, None], dtype="Int64"),
                         "spadl_type": ["pass", restart], "receiver_id": ["out", None],
                         "source_has_reception": [False, None],
                         "source_has_receiver_identifier": [False, None],
                         "source_xml_success": [False, None], "source_event_success": [False, None],
                         "success": [False, False], "object_id": ["home_1", "away_1"],
                         "offside": [False, False]})


@pytest.mark.parametrize("restart", ["throw_in", "goalkick", "corner_short", "corner_crossed"])
def test_only_failed_restart_passes_are_repaired_and_history_preserved(restart):
    source = events(restart)
    result = pre.apply_missing_reception_policy(source, tracking(), 25)
    assert result.loc[0, "receive_frame_id"] == 13
    assert result.loc[0, "receiver_id"] == "out"
    assert result.loc[0, "endpoint_source"] == "tracking_episode_end"
    assert result.training_sample_eligible.all()
    pd.testing.assert_frame_equal(result[source.columns.difference(["receive_frame_id", "receiver_id"])],
                                  source[source.columns.difference(["receive_frame_id", "receiver_id"])])


@pytest.mark.parametrize("changes,reason", [
    ({"source_xml_success": True}, "source_successful"),
    ({"source_event_success": True}, "source_successful"),
    ({"source_event_success": None, "source_xml_success": None}, "unknown_source_success"),
    ({"frame_id": None}, "invalid_episode_termination"),
    ({"frame_id": 13}, "invalid_episode_termination"),
    ({"frame_id": 14}, "invalid_episode_termination"),
    ({"frame_id": 15}, "invalid_episode_termination"),
])
def test_unresolved_cases_never_receive_a_placeholder(changes, reason):
    source = events()
    for key, value in changes.items():
        source.at[0, key] = value
    result = pre.apply_missing_reception_policy(source, tracking(), 25)
    assert not result.at[0, "training_sample_eligible"]
    assert pd.isna(result.at[0, "receive_frame_id"])
    assert result.at[0, "sample_exclusion_reason"] == reason
    assert len(result) == len(source)


@pytest.mark.parametrize("restart", ["foul", "freekick_short", "pass", None])
def test_other_next_events_are_excluded(restart):
    result = pre.apply_missing_reception_policy(events(restart), tracking(), 25)
    assert not result.at[0, "training_sample_eligible"]


def test_period_boundary_restart_overlap_and_known_receiver():
    source = events()
    source.at[1, "period_id"] = 2
    assert not pre.apply_missing_reception_policy(source, tracking(), 25).at[0, "training_sample_eligible"]
    source = events()
    source.at[1, "frame_id"] = 13
    assert not pre.apply_missing_reception_policy(source, tracking(), 25).at[0, "training_sample_eligible"]
    source.at[0, "source_has_receiver_identifier"] = True
    result = pre.apply_missing_reception_policy(source, tracking(), 25)
    assert result.at[0, "receive_frame_id"] == 9  # Existing fallback is outside this policy.
    assert result.at[0, "training_sample_eligible"]
    legacy = source.drop(columns=["source_has_reception", "source_has_receiver_identifier"])
    assert pre.apply_missing_reception_policy(legacy, tracking(), 25).training_sample_eligible.all()


@pytest.mark.parametrize("start,end", [(10, 10), (12, 10), (None, 12), (10, None), (9, 12), (10, 15)])
def test_invalid_height_and_trajectory_do_not_fabricate_supervision(start, end):
    match = Match.__new__(Match)
    match.tracking = tracking()
    match.fps = 25
    match.actions = pd.DataFrame([dict(frame_id=start, receive_frame_id=end, period_id=1,
                                     receiver_id="out", object_id="home_1", start_x=0., start_y=0., end_x=5., end_y=0.)])
    assert all(np.isnan(v) for v in match.pass_height_labels(start, end))
    assert graph_feature.summarize_ball_trajectory(match, 0) is None
    assert graph_feature.fallback_pass_trajectory_features(match, 0) is None
    with patch.object(graph_feature, "resolve_action_graph_context", return_value=(10, "home_1", match.tracking)):
        graphs = graph_feature.construct_graph_features(match, feature_variant="success_intent", action_indices=[0], verbose=False)
    assert graphs == [None]


def test_valid_height_and_trajectory_unchanged():
    match = Match.__new__(Match)
    match.tracking = tracking()
    match.fps = 25
    match.actions = pd.DataFrame([dict(frame_id=10, receive_frame_id=13, period_id=1, receiver_id="home_2", object_id="home_1")])
    assert match.pass_height_labels(10, 13) == (3., 1.)
    assert np.isfinite(graph_feature.summarize_ball_trajectory(match, 0)).all()


@pytest.mark.parametrize("start,end,eligible", [(110, 100, True), (None, 100, True), (80, 999, True), (80, 100, False)])
def test_invalid_receipts_do_not_start_carries(start, end, eligible):
    source = BallCarryRegressionTests.canonical()
    source.at[0, "frame_id"] = start
    source.at[0, "receive_frame_id"] = end
    source["training_sample_eligible"] = eligible
    segments, audit = derive_carry_segments(BallCarryRegressionTests.control(163), source, BallCarryRegressionTests.tracking())
    assert segments.empty
    assert not audit.start_kind.eq("receipt").any()


def test_equal_receipts_remain_valid_and_raw_frame_column_supported():
    source = BallCarryRegressionTests.canonical()
    source.at[0, "frame_id"] = 100
    t = BallCarryRegressionTests.tracking().reset_index()
    segments, _ = derive_carry_segments(BallCarryRegressionTests.control(163), source, t)
    assert len(segments) == 2
    t = t.set_index("frame_id")
    t["period_id"] = np.where(t.index < 90, 1, 2)
    assert not valid_interval(t, 80, 100, allow_equal=True, period=1)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1"])
def test_invalid_duration_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_duration(value)


@pytest.mark.parametrize("duration", [0., 0.1, 0.2, 0.5])
@pytest.mark.parametrize("carries", [False, True])
def test_duration_forwarded_to_every_component(tmp_path, duration, carries):
    args = make_training_args("features", min_pass_dur=duration, use_carries=carries,
                              enabled_tasks=make_enabled_tasks(pass_height=True))
    with patch.object(wrapper, "resolve_feature_run_id", return_value="features"), patch.object(wrapper, "resolve_feature_root", return_value=tmp_path):
        commands, _, _, _, _ = wrapper.build_training_commands(args)
    assert commands
    for command in commands:
        assert command.count("--min_pass_dur") == 1
        assert float(command[command.index("--min_pass_dur") + 1]) == duration


def test_legacy_dataset_configuration_keeps_historical_default():
    for train in [False, True]:
        kwargs = build_action_dataset_kwargs(SimpleNamespace(task="pass_success", edge_in_dim=2), train=train, diagnostic_label_dir=None)
        assert kwargs["min_pass_dur"] == 0.


def test_excluded_pass_cannot_be_sampled_even_if_endpoint_is_restored():
    match = Match.__new__(Match)
    match.events = events()
    match.events["training_sample_eligible"] = [False, True]
    match.events["next_player_id"] = "away_1"
    match.events["next_type"] = "throw_in"
    match.next_action_conditions_enabled = False
    assert match.filter_passes().empty
    assert len(match.events) == 2


def test_xml_provenance_and_sync_policy_survive_elastic_and_csv(tmp_path, monkeypatch):
    xml = tmp_path / "advanced.xml"
    xml.write_text('''<Root>
      <Event><Play EventId="pass" SyncedFrameId="10" Evaluation="unsuccessful" SyncSuccessful="true" /></Event>
      <Event><Play EventId="restart" SyncedFrameId="14" ReceiverId="p2" /></Event>
      <Event><TeamPossession EndSyncedFrameId="9"><PossessionEvent EventId="pass" /></TeamPossession></Event>
      </Root>''')
    match_files = SimpleNamespace(kpi_path=xml, match_id="fixture", kpi_format=pre.KpiFormat.ADVANCED_EVENTS_XML)
    kpi = pre._parse_kpi_xml_table(match_files)
    assert not kpi.at[0, "source_has_reception"]
    assert not kpi.at[0, "source_has_receiver_identifier"]
    assert not kpi.at[0, "source_xml_success"]
    assert kpi.at[1, "source_has_receiver_identifier"]
    source = events().drop(columns=[c for c in events() if c.startswith("source_")])
    for column in ["stats_perform_match_id", "utc_timestamp", "player_id", "player_name", "advanced_position",
                   "team_id", "expected_goal", "start_x", "start_y", "end_x", "end_y"]:
        source[column] = None
    frames = pd.DataFrame({"frame_id": range(10, 16), "raw_frame_id": range(10, 16), "period_id": [1]*5+[2], "synced_ts": ["time"]*6})
    monkeypatch.setattr(pre, "build_tracking_frame_table", lambda *a, **k: frames)
    fallback = source.copy()
    fallback["receive_frame_id"] = [11, 15]
    fallback["receiver_id"] = "home_2"
    monkeypatch.setattr(pre, "run_elastic_synchronization", lambda *a, **k: fallback)
    players = pd.DataFrame({"player_id": ["p2"], "object_id": ["home_2"]})
    result, _ = pre.run_kpi_synchronization(match_files, players, source, tracking(), 25.)
    assert result.at[0, "receive_frame_id"] == 13  # Neither possession 9 nor ELASTIC 11.
    assert result.at[0, "receiver_id"] == "out"
    assert result.at[1, "receive_frame_id"] == 15  # Known receiver still uses ELASTIC fallback.
    assert result.at[1, "receiver_id"] == "home_2"
    source.at[0, "success"] = True
    result, _ = pre.run_kpi_synchronization(match_files, players, source, tracking(), 25.)
    assert pd.isna(result.at[0, "receive_frame_id"])
    assert not result.at[0, "training_sample_eligible"]
    path = tmp_path / "synced.csv"
    result.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert not reloaded.at[0, "training_sample_eligible"]
    assert reloaded.at[0, "source_event_success"]
    assert reloaded.original_event_id.tolist() == source.original_event_id.tolist()
    # Missing final release does not remove the final canonical event.
    result.at[1, "frame_id"] = pd.NA
    assert len(pre.finalize_synced_output(result, frames)) == 2


@pytest.mark.parametrize("duration,expected", [(0., 4), (.1, 3), (.2, 2), (.5, 2)])
def test_dataset_duration_filters_passes_but_keeps_short_carries_and_shots(tmp_path, duration, expected):
    features = tmp_path / "features"
    labels_dir = tmp_path / "labels"
    features.mkdir()
    labels_dir.mkdir()
    x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM))
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.zeros((2, 2)))
    labels = torch.zeros((4, len(config.LABEL_COLUMNS)))
    labels[:, config.LABEL_INDEX["action_index"]] = torch.arange(4)
    labels[:, config.LABEL_INDEX["intent_index"]] = 1
    labels[:2, config.LABEL_INDEX["is_pass"]] = 1
    labels[2, config.LABEL_INDEX["is_dribble"]] = 1
    labels[3, config.LABEL_INDEX["is_shot"]] = 1
    labels[:, config.LABEL_INDEX["duration"]] = torch.tensor([.08, .12, .04, .04])
    torch.save([graph.clone() for _ in range(4)], features / "sample.pt")
    torch.save(labels, labels_dir / "sample.pt")
    dataset = ActionDataset(["sample"], feature_dir=features, label_dir=labels_dir,
                            task="action_intent", min_pass_dur=duration, edge_in_dim=2)
    assert len(dataset) == expected
    assert {2, 3}.issubset(set(dataset.labels[:, 0].tolist()))


def test_unavailable_height_diagnostics_propagate_as_nan():
    labels = torch.zeros((1, len(config.LABEL_COLUMNS)))
    labels[0, config.LABEL_INDEX["is_pass"]] = 1
    diagnostic = labels.clone()
    diagnostic[0, config.LABEL_INDEX["pass_high"]] = float("nan")
    diagnostic[0, config.LABEL_INDEX["pass_max_ball_z"]] = float("nan")
    copied = _copy_pass_height_diagnostics("sample", labels, diagnostic)
    assert torch.isnan(copied[0, config.LABEL_INDEX["pass_high"]])


def test_carry_artifact_version_roundtrip_and_legacy_rejection(tmp_path):
    segments, _ = derive_carry_segments(BallCarryRegressionTests.control(163),
                                        BallCarryRegressionTests.canonical(), BallCarryRegressionTests.tracking())
    for value in [segments, segments.iloc[:0]]:
        path = tmp_path / "carry.parquet"
        value.to_parquet(path, index=False)
        validate_carry_artifact_version(pd.read_parquet(path))
    segments["carry_definition"] = "sportec_fernandez_1s_v1"
    with pytest.raises(ValueError, match="Stale carry"):
        validate_carry_artifact_version(segments)
