from __future__ import annotations

import math

import pandas as pd
import pytest

from validation.skillcorner.code import skillcorner_filter as filt


def test_filter_skillcorner_ids_removes_empty_participants() -> None:
    skillcorner_ids = pd.DataFrame(
        [
            {"player_id": 1, "participant": "DM1"},
            {"player_id": 2, "participant": pd.NA},
            {"player_id": 3, "participant": "   "},
            {"player_id": 4, "participant": "DM4"},
        ]
    )
    skillcorner_ids["player_id"] = filt.coerce_nullable_integer(skillcorner_ids["player_id"])
    skillcorner_ids["participant"] = filt.normalize_participant(skillcorner_ids["participant"])

    filtered = filt.filter_skillcorner_ids(skillcorner_ids, expected_rows=2)

    assert filtered["player_id"].tolist() == [1, 4]
    assert filtered["participant"].tolist() == ["DM1", "DM4"]


def test_filter_skillcorner_ids_rejects_duplicate_player_id() -> None:
    skillcorner_ids = pd.DataFrame(
        [
            {"player_id": 1, "participant": "DM1"},
            {"player_id": 1, "participant": "DM2"},
        ]
    )
    skillcorner_ids["player_id"] = filt.coerce_nullable_integer(skillcorner_ids["player_id"])
    skillcorner_ids["participant"] = filt.normalize_participant(skillcorner_ids["participant"])

    with pytest.raises(ValueError, match="duplicate player_id"):
        filt.filter_skillcorner_ids(skillcorner_ids, expected_rows=2)


def test_filter_skillcorner_actions_raw_keeps_only_known_players() -> None:
    skillcorner_data = pd.DataFrame(
        [
            {
                "player_id": 10,
                "match_id": "m1",
                "pass_score": 0.1,
                "game_state_value_start": 0.2,
                "game_state_value_end": 0.25,
                "game_state_value_next": 0.3,
                "action_epv": 0.1,
                "end_type": "pass",
                "dm_score": 0.3,
                "extra": "keep",
            },
            {
                "player_id": 20,
                "match_id": "m2",
                "pass_score": 0.4,
                "game_state_value_start": 0.5,
                "game_state_value_end": 0.55,
                "game_state_value_next": pd.NA,
                "action_epv": pd.NA,
                "end_type": "pass",
                "dm_score": 0.6,
                "extra": "drop",
            },
        ]
    )
    filtered_ids = pd.DataFrame(
        [
            {"player_id": 10, "participant": "DM10"},
        ]
    )
    skillcorner_data["player_id"] = filt.coerce_nullable_integer(skillcorner_data["player_id"])
    filtered_ids["player_id"] = filt.coerce_nullable_integer(filtered_ids["player_id"])

    filtered = filt.filter_skillcorner_actions_raw(skillcorner_data, filtered_ids)

    assert filtered["player_id"].tolist() == [10]
    assert filtered["extra"].tolist() == ["keep"]


def test_build_skillcorner_actions_keeps_requested_columns_and_drops_empty_dm_score() -> None:
    skillcorner_actions_raw = pd.DataFrame(
        [
            {
                "player_id": 10,
                "match_id": "m1",
                "pass_score": 0.1,
                "risk": 0.01,
                "reward": 0.11,
                "game_state_value_start": 0.2,
                "game_state_value_end": 0.25,
                "game_state_value_next": 0.4,
                "action_epv": 0.2,
                "end_type": "pass",
                "dm_score": 0.3,
            },
            {
                "player_id": 20,
                "match_id": "m2",
                "pass_score": 0.4,
                "risk": 0.04,
                "reward": 0.44,
                "game_state_value_start": 0.5,
                "game_state_value_end": 0.55,
                "game_state_value_next": pd.NA,
                "action_epv": pd.NA,
                "end_type": "pass",
                "dm_score": pd.NA,
            },
        ]
    )
    filtered_ids = pd.DataFrame(
        [
            {"player_id": 10, "participant": "DM10"},
            {"player_id": 20, "participant": "DM20"},
        ]
    )
    skillcorner_actions_raw["player_id"] = filt.coerce_nullable_integer(skillcorner_actions_raw["player_id"])
    filtered_ids["player_id"] = filt.coerce_nullable_integer(filtered_ids["player_id"])

    actions = filt.build_skillcorner_actions(skillcorner_actions_raw, filtered_ids)

    assert actions.columns.tolist() == filt.ACTIONS_COLUMNS
    assert actions["participant"].tolist() == ["DM10"]
    assert actions["match_id"].tolist() == ["m1"]
    assert actions["risk"].tolist() == [0.01]
    assert actions["reward"].tolist() == [0.11]
    assert actions["game_state_value_start"].tolist() == [0.2]
    assert actions["game_state_value_end"].tolist() == [0.25]
    assert actions["game_state_value_next"].tolist() == [0.4]
    assert actions["action_epv"].tolist() == [0.2]
    assert actions["end_type"].tolist() == ["pass"]


def test_aggregate_skillcorner_players_computes_counts_sums_means_and_medians() -> None:
    skillcorner_actions = pd.DataFrame(
        [
            {"participant": "DM1", "pass_score": 1.0, "risk": 2.0, "reward": 10.0, "game_state_value_start": 4.0, "action_epv": 5.0, "dm_score": 7.0, "match_id": "m1"},
            {"participant": "DM1", "pass_score": 3.0, "risk": 4.0, "reward": 20.0, "game_state_value_start": 6.0, "action_epv": 7.0, "dm_score": 9.0, "match_id": "m2"},
            {"participant": "DM2", "pass_score": 5.0, "risk": 6.0, "reward": 30.0, "game_state_value_start": 8.0, "action_epv": 13.0, "dm_score": 11.0, "match_id": "m3"},
        ]
    )

    players = filt.aggregate_skillcorner_players(skillcorner_actions)

    assert players["participant"].tolist() == ["DM1", "DM2"]
    dm1 = players.loc[players["participant"].eq("DM1")].iloc[0]
    assert int(dm1["actions"]) == 2
    assert math.isclose(dm1["pass_score_sum"], 4.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_score_avg"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_score_median"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["risk_sum"], 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["risk_avg"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["risk_median"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["reward_sum"], 30.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["reward_avg"], 15.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["reward_median"], 15.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["game_state_value_start_sum"], 10.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["game_state_value_start_avg"], 5.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["game_state_value_start_median"], 5.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["action_epv_sum"], 12.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["action_epv_avg"], 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["action_epv_median"], 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["dm_score_sum"], 16.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["dm_score_avg"], 8.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["dm_score_median"], 8.0, rel_tol=1e-9, abs_tol=1e-9)
