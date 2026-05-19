from __future__ import annotations

import math

import pandas as pd
import pytest

from validation.skillcorner.code import skillcorner_filter as filt


ACTION_EXTRA_COLUMNS = {
    "minute_start": 12,
    "duration": 1.5,
    "period": 1,
    "player_position": "Midfielder",
    "game_state": "open_play",
    "team_score": 1,
    "opponent_team_score": 0,
    "team_in_possession_phase_type": "build_up",
    "team_out_of_possession_phase_type": "mid_block",
    "distance_covered": 8.0,
    "speed_avg": 5.0,
    "speed_avg_band": "medium",
    "separation_start": 3.0,
    "separation_end": 4.0,
    "separation_gain": 1.0,
    "one_touch": False,
    "quick_pass": True,
    "carry": False,
    "pass_outcome": "complete",
    "high_pass": False,
    "player_targeted_xpass_completion": 0.7,
    "player_targeted_xthreat": 0.03,
}


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
                **ACTION_EXTRA_COLUMNS,
                "pass_score": 0.1,
                "risk": 0.01,
                "reward": 0.11,
                "game_state_value_start": 0.2,
                "game_state_value_end": 0.25,
                "game_state_value_next": 0.3,
                "action_epv": 0.1,
                "pass_dm_score": -0.15,
                "carry_epv": 0.05,
                "pass_epv": 0.05,
                "z_dm_score": -1.0,
                "z_pass_dm_score": -1.5,
                "rank": 2.0,
                "end_type": "pass",
                "dm_score": 0.3,
                "extra": "keep",
            },
            {
                "player_id": 20,
                "match_id": "m2",
                **ACTION_EXTRA_COLUMNS,
                "pass_score": 0.4,
                "risk": 0.04,
                "reward": 0.44,
                "game_state_value_start": 0.5,
                "game_state_value_end": 0.55,
                "game_state_value_next": pd.NA,
                "action_epv": pd.NA,
                "pass_dm_score": -0.15,
                "carry_epv": 0.05,
                "pass_epv": pd.NA,
                "z_dm_score": -1.0,
                "z_pass_dm_score": -1.5,
                "rank": 1.0,
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


def test_build_skillcorner_actions_keeps_requested_columns_and_drops_only_all_empty_metrics() -> None:
    skillcorner_actions_raw = pd.DataFrame(
        [
            {
                "player_id": 10,
                "match_id": "m1",
                **ACTION_EXTRA_COLUMNS,
                "pass_score": 0.1,
                "risk": 0.01,
                "reward": 0.11,
                "game_state_value_start": 0.2,
                "game_state_value_end": 0.25,
                "game_state_value_next": 0.4,
                "action_epv": 0.2,
                "end_type": "pass",
                "dm_score": 0.3,
                "pass_dm_score": -0.15,
                "carry_epv": 0.05,
                "pass_epv": 0.15,
                "z_dm_score": -1.0,
                "z_pass_dm_score": -1.5,
                "rank": 2.0,
            },
            {
                "player_id": 20,
                "match_id": "m2",
                **ACTION_EXTRA_COLUMNS,
                "pass_score": 0.4,
                "risk": 0.04,
                "reward": 0.44,
                "game_state_value_start": 0.5,
                "game_state_value_end": 0.55,
                "game_state_value_next": pd.NA,
                "action_epv": pd.NA,
                "end_type": "pass",
                "dm_score": pd.NA,
                "pass_dm_score": -0.15,
                "carry_epv": 0.05,
                "pass_epv": pd.NA,
                "z_dm_score": -1.0,
                "z_pass_dm_score": -1.5,
                "rank": 1.0,
            },
            {
                "player_id": 30,
                "match_id": "m3",
                **ACTION_EXTRA_COLUMNS,
                "pass_score": pd.NA,
                "risk": pd.NA,
                "reward": pd.NA,
                "game_state_value_start": pd.NA,
                "game_state_value_end": pd.NA,
                "game_state_value_next": pd.NA,
                "action_epv": pd.NA,
                "end_type": "pass",
                "dm_score": pd.NA,
                "pass_dm_score": pd.NA,
                "carry_epv": pd.NA,
                "pass_epv": pd.NA,
                "z_dm_score": pd.NA,
                "z_pass_dm_score": pd.NA,
                "rank": pd.NA,
            },
        ]
    )
    filtered_ids = pd.DataFrame(
        [
            {"player_id": 10, "participant": "DM10"},
            {"player_id": 20, "participant": "DM20"},
            {"player_id": 30, "participant": "DM30"},
        ]
    )
    skillcorner_actions_raw["player_id"] = filt.coerce_nullable_integer(skillcorner_actions_raw["player_id"])
    filtered_ids["player_id"] = filt.coerce_nullable_integer(filtered_ids["player_id"])

    actions = filt.build_skillcorner_actions(skillcorner_actions_raw, filtered_ids)

    assert actions.columns.tolist() == filt.ACTIONS_COLUMNS
    assert actions["participant"].tolist() == ["DM10", "DM20"]
    assert actions["match_id"].tolist() == ["m1", "m2"]
    assert actions["risk"].tolist() == [0.01, 0.04]
    assert actions["reward"].tolist() == [0.11, 0.44]
    assert actions["game_state_value_start"].tolist() == [0.2, 0.5]
    assert actions["game_state_value_end"].tolist() == [0.25, 0.55]
    assert actions["game_state_value_next"].tolist() == [0.4, pd.NA]
    assert actions["action_epv"].tolist() == [0.2, pd.NA]
    assert actions["dm_score"].tolist() == [0.3, pd.NA]
    assert actions["pass_dm_score"].tolist() == [-0.15, -0.15]
    assert actions["carry_epv"].tolist() == [0.05, 0.05]
    assert actions["pass_epv"].tolist() == [0.15, pd.NA]
    assert actions["z_dm_score"].tolist() == [-1.0, -1.0]
    assert actions["z_pass_dm_score"].tolist() == [-1.5, -1.5]
    assert actions["rank"].tolist() == [2.0, 1.0]
    assert actions["minute_start"].tolist() == [12, 12]
    assert actions["opponent_team_score"].tolist() == [0, 0]
    assert actions["player_targeted_xthreat"].tolist() == [0.03, 0.03]
    assert actions["end_type"].tolist() == ["pass", "pass"]


def test_aggregate_skillcorner_players_computes_counts_sums_means_and_medians() -> None:
    skillcorner_actions = pd.DataFrame(
        [
            {"participant": "DM1", "pass_score": 1.0, "risk": 2.0, "reward": 10.0, "game_state_value_start": 4.0, "action_epv": 5.0, "dm_score": 7.0, "pass_dm_score": 1.0, "carry_epv": 2.0, "pass_epv": 3.0, "z_dm_score": 0.5, "z_pass_dm_score": 0.25, "rank": 2.0, "match_id": "m1"},
            {"participant": "DM1", "pass_score": 3.0, "risk": 4.0, "reward": 20.0, "game_state_value_start": 6.0, "action_epv": 7.0, "dm_score": 9.0, "pass_dm_score": 3.0, "carry_epv": 4.0, "pass_epv": 5.0, "z_dm_score": 1.5, "z_pass_dm_score": 0.75, "rank": 1.0, "match_id": "m2"},
            {"participant": "DM2", "pass_score": 5.0, "risk": 6.0, "reward": 30.0, "game_state_value_start": 8.0, "action_epv": 13.0, "dm_score": 11.0, "pass_dm_score": 5.0, "carry_epv": 6.0, "pass_epv": 7.0, "z_dm_score": 2.5, "z_pass_dm_score": 1.25, "rank": 1.0, "match_id": "m3"},
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
    assert math.isclose(dm1["pass_dm_score_sum"], 4.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_dm_score_avg"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_dm_score_median"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["carry_epv_sum"], 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["carry_epv_avg"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["carry_epv_median"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_epv_sum"], 8.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_epv_avg"], 4.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["pass_epv_median"], 4.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_dm_score_sum"], 2.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_dm_score_avg"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_dm_score_median"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_pass_dm_score_sum"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_pass_dm_score_avg"], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["z_pass_dm_score_median"], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["rank_avg"], 1.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(dm1["rank_median"], 1.5, rel_tol=1e-9, abs_tol=1e-9)
