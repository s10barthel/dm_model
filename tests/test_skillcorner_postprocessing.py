from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import pandas as pd
import torch
from torch_geometric.data import Data

import scripts.run_skillcorner as run_skillcorner
from datatools import config
from datatools.graph_feature import load_frame_snapshot
from datatools import skillcorner
from datatools.skillcorner import SkillcornerPossession, load_skillcorner_events
from validation.skillcorner.code import skillcorner_postprocessing as post


def test_load_frame_snapshot_keeps_current_phase_only() -> None:
    tracking = pd.DataFrame(
        [
            {"frame_id": 9, "phase_id": 1, "home_1_x": 1.0, "home_1_y": 2.0},
            {"frame_id": 10, "phase_id": 2, "home_1_x": 1.5, "home_1_y": 2.5},
        ]
    ).set_index("frame_id", drop=False)

    snapshot = load_frame_snapshot(tracking, tracking, 10)

    assert snapshot.index.tolist() == [10]
    assert snapshot["phase_id"].tolist() == [2]


def make_skillcorner_possession(
    frame_ids: list[int],
    *,
    has_ball: dict[int, bool] | None = None,
    missing_possessor_frames: set[int] | None = None,
) -> SkillcornerPossession:
    has_ball = has_ball or {}
    missing_possessor_frames = missing_possessor_frames or set()
    tracking_rows = []
    frame_meta_rows = []
    for frame_id in frame_ids:
        possessor_x = pd.NA if frame_id in missing_possessor_frames else 30.0 + frame_id
        possessor_y = pd.NA if frame_id in missing_possessor_frames else 20.0
        ball_present = has_ball.get(frame_id, True)
        tracking_rows.append(
            {
                "frame_id": frame_id,
                "period_id": 1,
                "ball_x": 40.0 if ball_present else pd.NA,
                "ball_y": 30.0 if ball_present else pd.NA,
                "ball_owning_home_away": "home",
                "home_1_x": possessor_x,
                "home_1_y": possessor_y,
                "away_2_x": 60.0,
                "away_2_y": 30.0,
                "home_goal_x": 105.0,
                "home_goal_y": 34.0,
                "away_goal_x": 0.0,
                "away_goal_y": 34.0,
            }
        )
        frame_meta_rows.append(
            {
                "frame_id": frame_id,
                "match_id": "match-1",
                "index": 7,
                "period": 1,
                "player_id": 1,
                "attacking_side": "left_to_right",
                "possessor_object_id": "home_1",
                "has_ball": ball_present,
            }
        )

    tracking = pd.DataFrame(tracking_rows).set_index("frame_id", drop=False)
    frame_meta = pd.DataFrame(frame_meta_rows).set_index("frame_id", drop=True)
    return SkillcornerPossession(
        match_id="match-1",
        event_index=7,
        fps=25.0,
        tracking=tracking,
        phases=pd.DataFrame(),
        frame_meta=frame_meta,
        actions=pd.DataFrame(),
        labels=torch.empty((0, len(config.LABEL_COLUMNS))),
        graph_features_0=[],
    )


def patch_skillcorner_graph_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_graph_frames: set[int] | None = None,
) -> list[int]:
    missing_graph_frames = missing_graph_frames or set()
    calls: list[int] = []

    def fake_construct_graph_for_frame(possession, frame_id, *args, **kwargs):
        calls.append(int(frame_id))
        if int(frame_id) in missing_graph_frames:
            return None
        return Data()

    monkeypatch.setattr(skillcorner, "construct_graph_for_frame", fake_construct_graph_for_frame)
    return calls


def test_skillcorner_default_frame_mode_selects_only_boundary_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12, 13, 14])

    actions, labels, graphs, stats = skillcorner._build_actions_and_labels(possession)

    assert actions.index.tolist() == [10, 14]
    assert labels[:, 0].tolist() == [10, 14]
    assert len(graphs) == 2
    assert calls == [10, 14]
    assert stats["total_frames"] == 5
    assert stats["evaluated_frames"] == 2
    assert stats["selected_frames"] == 2
    assert stats["valid_frames"] == 2


def test_skillcorner_all_frame_mode_preserves_all_valid_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12, 13, 14])

    actions, labels, graphs, stats = skillcorner._build_actions_and_labels(possession, frames_mode="all")

    assert actions.index.tolist() == [10, 11, 12, 13, 14]
    assert labels[:, 0].tolist() == [10, 11, 12, 13, 14]
    assert len(graphs) == 5
    assert stats["evaluated_frames"] == 5
    assert stats["selected_frames"] == 5
    assert stats["valid_frames"] == 5


def test_skillcorner_first_last_frame_mode_dedupes_single_valid_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch)
    possession = make_skillcorner_possession([10, 11, 12], has_ball={10: False, 12: False})

    actions, _labels, graphs, stats = skillcorner._build_actions_and_labels(possession, frames_mode="first_and_last")

    assert actions.index.tolist() == [11]
    assert len(graphs) == 1
    assert stats["evaluated_frames"] == 3
    assert stats["selected_frames"] == 1
    assert stats["valid_frames"] == 1
    assert stats["skipped_missing_ball"] == 2


def test_skillcorner_first_last_frame_mode_falls_back_from_invalid_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_skillcorner_graph_builder(monkeypatch, missing_graph_frames={11})
    possession = make_skillcorner_possession(
        [10, 11, 12, 13, 14, 15],
        has_ball={10: False},
        missing_possessor_frames={15},
    )

    actions, _labels, _graphs, stats = skillcorner._build_actions_and_labels(
        possession,
        frames_mode="first_and_last",
    )

    assert actions.index.tolist() == [12, 14]
    assert stats["evaluated_frames"] == 5
    assert stats["selected_frames"] == 2
    assert stats["valid_frames"] == 2
    assert stats["skipped_missing_ball"] == 1
    assert stats["skipped_missing_graph"] == 1
    assert stats["skipped_missing_possessor"] == 1


def test_run_skillcorner_frame_mode_cli_defaults_and_flags() -> None:
    assert run_skillcorner.parse_args([]).frames_mode == "first_and_last"
    assert run_skillcorner.parse_args(["--frames-first-and-last"]).frames_mode == "first_and_last"
    assert run_skillcorner.parse_args(["--frames-all"]).frames_mode == "all"

    with pytest.raises(SystemExit):
        run_skillcorner.parse_args(["--frames-first-and-last", "--frames-all"])


def test_filter_event_rows_and_drop_empty_columns() -> None:
    event_data = pd.DataFrame(
        [
            {"match_id": 117670, "index": 1, "event_type_id": 8, "start_type_id": 1, "frame_start": 100, "frame_end": 105, "player_targeted_id": 10, "always_empty": pd.NA},
            {"match_id": 117670, "index": 2, "event_type_id": 8, "start_type_id": 2, "frame_start": 110, "frame_end": 115, "player_targeted_id": 11, "always_empty": pd.NA},
            {"match_id": 117670, "index": 3, "event_type_id": 7, "start_type_id": 1, "frame_start": 120, "frame_end": 125, "player_targeted_id": 12, "always_empty": pd.NA},
        ]
    )
    event_data = post.normalize_event_identifiers(event_data)

    filtered = post.filter_event_rows(event_data)
    filtered = post.drop_all_empty_columns(filtered)

    assert filtered["index"].tolist() == [1]
    assert "always_empty" not in filtered.columns


def test_load_skillcorner_events_filters_start_type_id() -> None:
    raw_events = pd.DataFrame(
        [
            {
                "match_id": 117670,
                "index": 1,
                "event_type": "player_possession",
                "frame_start": 100,
                "frame_end": 105,
                "period": 1,
                "start_type_id": 1,
                "attacking_side": "left_to_right",
                "player_id": 10,
            },
            {
                "match_id": 117670,
                "index": 2,
                "event_type": "player_possession",
                "frame_start": 110,
                "frame_end": 115,
                "period": 1,
                "start_type_id": 2,
                "attacking_side": "left_to_right",
                "player_id": 11,
            },
        ]
    )
    with patch("datatools.skillcorner.pd.read_csv", return_value=raw_events):
        filtered = load_skillcorner_events("117670", ".")

    assert filtered["index"].tolist() == [1]


def test_select_nearest_frame_match_prefers_earlier_or_later_on_ties() -> None:
    candidate_rows = pd.DataFrame(
        [
            {"event_row_id": 0, "frame_start": 12, "frame": 10, "game_state_value": 1.0},
            {"event_row_id": 0, "frame_start": 12, "frame": 14, "game_state_value": 2.0},
            {"event_row_id": 1, "frame_end": 12, "frame": 10, "pass_score": 3.0},
            {"event_row_id": 1, "frame_end": 12, "frame": 14, "pass_score": 4.0},
        ]
    )

    earlier = post.select_nearest_frame_match(
        candidate_rows.loc[candidate_rows["event_row_id"].eq(0)].copy(),
        event_id_column="event_row_id",
        event_frame_column="frame_start",
        candidate_frame_column="frame",
        tie_break="earlier",
        value_columns=["game_state_value"],
    )
    later = post.select_nearest_frame_match(
        candidate_rows.loc[candidate_rows["event_row_id"].eq(1)].copy(),
        event_id_column="event_row_id",
        event_frame_column="frame_end",
        candidate_frame_column="frame",
        tie_break="later",
        value_columns=["pass_score"],
    )

    assert earlier["game_state_value"].tolist() == [1.0]
    assert later["pass_score"].tolist() == [4.0]


def test_prune_component_frames_keeps_first_and_last_only() -> None:
    component = pd.DataFrame(
        [
            {"match_id": "117670", "frame": 100, "index": 1, "player_id": 9, "111": 0.1},
            {"match_id": "117670", "frame": 101, "index": 1, "player_id": 9, "111": 0.2},
            {"match_id": "117670", "frame": 105, "index": 1, "player_id": 9, "111": 0.3},
            {"match_id": "117670", "frame": 200, "index": 2, "player_id": 10, "111": 0.4},
        ]
    )

    pruned = post.prune_component_frames(component)

    assert pruned[["index", "frame"]].to_dict("records") == [
        {"index": 1, "frame": 100},
        {"index": 1, "frame": 105},
        {"index": 2, "frame": 200},
    ]


def test_melt_component_frame_preserves_possessor_and_renames_receiver() -> None:
    component = pd.DataFrame(
        [
            {
                "match_id": 117670,
                "frame": 337,
                "index": 0,
                "period": 1,
                "player_id": 63637,
                "attacking_side": "right_to_left",
                "63801": 0.4,
                "69889": 0.6,
                "shot": 0.01,
            }
        ]
    )

    melted = post.melt_component_frame(component, "action_intent", "test_component")

    assert list(melted.columns) == ["match_id", "frame", "index", "player_id", "receiver_id", "action_intent"]
    assert melted["match_id"].tolist() == ["117670", "117670"]
    assert melted["player_id"].tolist() == [63637, 63637]
    assert melted["receiver_id"].tolist() == [63801, 69889]
    assert melted["action_intent"].tolist() == [0.4, 0.6]


def test_compute_model_scores_calculates_pass_score_risk_reward_and_game_state_value() -> None:
    model_data = pd.DataFrame(
        [
            {
                "match_id": "117670",
                "frame": 337,
                "index": 0,
                "player_id": 63637,
                "receiver_id": 63801,
                "action_intent": 0.2,
                "pass_intent": 0.7,
                "pass_success": 0.8,
                "outcome_scoring_success": 0.5,
                "outcome_scoring_failure": 0.2,
                "outcome_conceding_success": 0.1,
                "outcome_conceding_failure": 0.3,
            },
            {
                "match_id": "117670",
                "frame": 337,
                "index": 0,
                "player_id": 63637,
                "receiver_id": 69889,
                "action_intent": 0.8,
                "pass_intent": 0.3,
                "pass_success": 0.2,
                "outcome_scoring_success": 0.6,
                "outcome_scoring_failure": 0.1,
                "outcome_conceding_success": 0.1,
                "outcome_conceding_failure": 0.05,
            },
        ]
    )

    scored = post.compute_model_scores(model_data)

    pass_scores = scored.sort_values("receiver_id")["pass_score"].tolist()
    risks = scored.sort_values("receiver_id")["risk"].tolist()
    rewards = scored.sort_values("receiver_id")["reward"].tolist()
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(pass_scores, [0.3, 0.14]))
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(risks, [0.14, 0.06]))
    assert all(math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9) for left, right in zip(rewards, [0.44, 0.2]))
    assert all(
        math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
        for left, right in zip(scored["game_state_value"].tolist(), [0.252, 0.252])
    )


def test_add_scores_to_event_data_uses_nearest_available_frames_and_one_frame_events() -> None:
    model_data = pd.DataFrame(
        [
            {"match_id": "117670", "index": 0, "frame": 337, "player_id": 63637, "receiver_id": 63801, "pass_score": 0.10, "risk": 0.01, "reward": 0.11, "game_state_value": 0.25},
            {"match_id": "117670", "index": 0, "frame": 337, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.20, "risk": 0.02, "reward": 0.22, "game_state_value": 0.25},
            {"match_id": "117670", "index": 0, "frame": 346, "player_id": 63637, "receiver_id": 63801, "pass_score": 0.30, "risk": 0.03, "reward": 0.33, "game_state_value": 0.25},
            {"match_id": "117670", "index": 0, "frame": 346, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.40, "risk": 0.04, "reward": 0.44, "game_state_value": 0.25},
            {"match_id": "117670", "index": 1, "frame": 400, "player_id": 63637, "receiver_id": 69889, "pass_score": 0.50, "risk": 0.05, "reward": 0.55, "game_state_value": 0.35},
        ]
    )
    event_data = pd.DataFrame(
        [
            {"match_id": "117670", "index": 0, "frame_start": 335, "frame_end": 347, "player_id": 63637, "player_targeted_id": 69889},
            {"match_id": "117670", "index": 0, "frame_start": 335, "frame_end": 347, "player_id": 63637, "player_targeted_id": 12345},
            {"match_id": "117670", "index": 1, "frame_start": 400, "frame_end": 400, "player_id": 63637, "player_targeted_id": 69889},
        ]
    )
    event_data = post.normalize_event_identifiers(event_data)

    scored = post.add_scores_to_event_data(model_data, event_data)

    assert scored.columns.tolist()[-5:] == ["pass_score", "risk", "reward", "game_state_value", "dm_score"]
    assert math.isclose(scored.loc[0, "pass_score"], 0.40, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "risk"], 0.04, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "reward"], 0.44, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "game_state_value"], 0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[0, "dm_score"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[1, "pass_score"])
    assert pd.isna(scored.loc[1, "risk"])
    assert pd.isna(scored.loc[1, "reward"])
    assert math.isclose(scored.loc[1, "game_state_value"], 0.25, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(scored.loc[1, "dm_score"])
    assert math.isclose(scored.loc[2, "pass_score"], 0.50, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "risk"], 0.05, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "reward"], 0.55, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "game_state_value"], 0.35, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(scored.loc[2, "dm_score"], 0.15, rel_tol=1e-9, abs_tol=1e-9)
