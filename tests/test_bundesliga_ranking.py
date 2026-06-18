from __future__ import annotations

import math
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDESLIGA_RANKING_CODE = PROJECT_ROOT / "validation" / "bundesliga_ranking" / "code"
if str(BUNDESLIGA_RANKING_CODE) not in sys.path:
    sys.path.insert(0, str(BUNDESLIGA_RANKING_CODE))

import bundesliga_ranking as ranking  # noqa: E402


MATCH_ID = "DFL-MAT-TEST"
OBJECT_COLUMNS = ["home_1", "away_2"]
ACTION_IDENTIFIERS = [
    {"stats_perform_match_id": MATCH_ID, "action_id": 0, "original_event_id": 100},
    {"stats_perform_match_id": MATCH_ID, "action_id": 1, "original_event_id": 101},
]
STATE_FRAMES = {
    (0, ranking.FRAME_ID_SCOPE): 10,
    (0, ranking.RECEIVE_FRAME_ID_SCOPE): 20,
    (1, ranking.FRAME_ID_SCOPE): 30,
    (1, ranking.RECEIVE_FRAME_ID_SCOPE): 40,
}


def test_default_component_runs_dir_points_to_sportec_subfolder() -> None:
    assert ranking.COMPONENT_RUNS_DIR == PROJECT_ROOT / "data" / "component_runs" / "sportec"
    assert "sportec" in ranking.SPECIAL_COMPONENT_DIRS


def test_component_run_discovery_ignores_dataset_subfolders() -> None:
    with workspace_temp_dir("component_roots") as root:
        (root / "sportec").mkdir()
        (root / "hawkeye").mkdir()
        run_root = root / "component_1"
        run_root.mkdir()
        args = SimpleNamespace(
            component_run_root=None,
            component_run_id=None,
            component_runs_dir=root,
        )

        assert ranking.resolve_component_run_root(args) == run_root


@contextmanager
def workspace_temp_dir(prefix: str):
    tmp_root = PROJECT_ROOT / ".pycache_tmp" / "br"
    tmp_root.mkdir(parents=True, exist_ok=True)
    path = tmp_root / f"{prefix}_{uuid.uuid4().hex[:8]}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def scoped_component_frame(values_by_action_scope: dict[tuple[int, str], dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for identifiers in ACTION_IDENTIFIERS:
        action_id = int(identifiers["action_id"])
        for scope in [ranking.FRAME_ID_SCOPE, ranking.RECEIVE_FRAME_ID_SCOPE]:
            row = {
                **identifiers,
                ranking.FRAME_SCOPE_COLUMN: scope,
                ranking.STATE_FRAME_ID_COLUMN: STATE_FRAMES[(action_id, scope)],
            }
            row.update(values_by_action_scope[(action_id, scope)])
            rows.append(row)
    return pd.DataFrame(rows)


def write_scoped_component_run(match_dir: Path) -> None:
    match_dir.mkdir(parents=True)
    pass_score_inputs = {
        (0, ranking.FRAME_ID_SCOPE): {
            "home_1": {"pass_score": 0.20, "risk": 0.05, "reward": 0.25},
            "away_2": {"pass_score": 0.40, "risk": 0.10, "reward": 0.50},
        },
        (0, ranking.RECEIVE_FRAME_ID_SCOPE): {
            "home_1": {"pass_score": 0.60, "risk": 0.10, "reward": 0.70},
            "away_2": {"pass_score": 0.80, "risk": 0.10, "reward": 0.90},
        },
        (1, ranking.FRAME_ID_SCOPE): {
            "home_1": {"pass_score": 0.50, "risk": 0.12, "reward": 0.62},
            "away_2": {"pass_score": 0.30, "risk": 0.05, "reward": 0.35},
        },
        (1, ranking.RECEIVE_FRAME_ID_SCOPE): {
            "home_1": {"pass_score": 0.05, "risk": 0.01, "reward": 0.06},
            "away_2": {"pass_score": 0.90, "risk": 0.20, "reward": 1.10},
        },
    }
    pass_intent = {
        (0, ranking.FRAME_ID_SCOPE): {"home_1": 0.25, "away_2": 0.75},
        (0, ranking.RECEIVE_FRAME_ID_SCOPE): {"home_1": 0.25, "away_2": 0.75},
        (1, ranking.FRAME_ID_SCOPE): {"home_1": 0.80, "away_2": 0.20},
        (1, ranking.RECEIVE_FRAME_ID_SCOPE): {"home_1": 0.80, "away_2": 0.20},
    }
    pass_success = {
        key: {object_id: 1.0 for object_id in OBJECT_COLUMNS}
        for key in pass_score_inputs
    }
    scoring_success = {
        key: {object_id: values[object_id]["reward"] for object_id in OBJECT_COLUMNS}
        for key, values in pass_score_inputs.items()
    }
    conceding_success = {
        key: {object_id: values[object_id]["risk"] for object_id in OBJECT_COLUMNS}
        for key, values in pass_score_inputs.items()
    }
    zero_component = {
        key: {object_id: 0.0 for object_id in OBJECT_COLUMNS}
        for key in pass_score_inputs
    }
    components = {
        "pass_intent": pass_intent,
        "pass_success": pass_success,
        "outcome_scoring_success": scoring_success,
        "outcome_scoring_failure": zero_component,
        "outcome_conceding_success": conceding_success,
        "outcome_conceding_failure": zero_component,
    }
    for component_name, values in components.items():
        scoped_component_frame(values).to_parquet(match_dir / f"{component_name}.parquet", index=False)


def write_synced_events(event_dir: Path) -> None:
    event_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stats_perform_match_id": MATCH_ID,
                "action_id": 0,
                "original_event_id": 100,
                "period_id": 1,
                "seconds": 1.0,
                "frame_id": 10,
                "receive_frame_id": 20,
                "spadl_type": "pass",
                "success": True,
                "receiver_id": "away_2",
                "player_id": "DFL-P1",
                "object_id": "home_1",
                "player_name": "Player One",
                "advanced_position": "CM",
                "team_id": "home",
            },
            {
                "stats_perform_match_id": MATCH_ID,
                "action_id": 1,
                "original_event_id": 101,
                "period_id": 1,
                "seconds": 2.0,
                "frame_id": 30,
                "receive_frame_id": 40,
                "spadl_type": "pass",
                "success": True,
                "receiver_id": "home_1",
                "player_id": "DFL-P2",
                "object_id": "away_2",
                "player_name": "Player Two",
                "advanced_position": "DM",
                "team_id": "away",
            },
        ]
    ).to_csv(event_dir / f"{MATCH_ID}.csv", index=False)


def test_bundesliga_scores_use_frame_scope_for_target_and_receive_scope_for_next_state() -> None:
    with workspace_temp_dir("scoring") as tmp_path:
        match_dir = tmp_path / "component_run" / MATCH_ID
        event_dir = tmp_path / "event_synced"
        write_scoped_component_run(match_dir)
        write_synced_events(event_dir)

        model_data = ranking.build_match_model_data(match_dir)
        scored = ranking.add_scores_to_events(model_data, event_dir)

        first = scored.loc[scored["action_id"].eq(0)].iloc[0]
        assert math.isclose(first["pass_score"], 0.40, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(first["game_state_value_end"], 0.35, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(first["game_state_value_next"], 0.75, rel_tol=1e-9, abs_tol=1e-9)
        assert pd.isna(first["game_state_value_start"])

        second = scored.loc[scored["action_id"].eq(1)].iloc[0]
        assert math.isclose(second["pass_score"], 0.50, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["risk"], 0.12, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["reward"], 0.62, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["rank"], 1.0, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["game_state_value_start"], 0.75, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["game_state_value_end"], 0.46, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["game_state_value_next"], 0.22, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["action_epv"], -0.53, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["dm_score"], -0.25, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["pass_dm_score"], 0.04, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["carry_epv"], -0.29, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(second["pass_epv"], -0.24, rel_tol=1e-9, abs_tol=1e-9)


def test_scoped_component_reader_rejects_missing_receive_scope() -> None:
    with workspace_temp_dir("missing_scope") as tmp_path:
        match_dir = tmp_path / MATCH_ID
        match_dir.mkdir()
        frame_only = scoped_component_frame(
            {
                (0, ranking.FRAME_ID_SCOPE): {"home_1": 1.0, "away_2": 0.0},
                (0, ranking.RECEIVE_FRAME_ID_SCOPE): {"home_1": 0.0, "away_2": 1.0},
                (1, ranking.FRAME_ID_SCOPE): {"home_1": 1.0, "away_2": 0.0},
                (1, ranking.RECEIVE_FRAME_ID_SCOPE): {"home_1": 0.0, "away_2": 1.0},
            }
        )
        frame_only = frame_only.loc[frame_only[ranking.FRAME_SCOPE_COLUMN].eq(ranking.FRAME_ID_SCOPE)]
        frame_only.to_parquet(match_dir / "pass_intent.parquet", index=False)

        try:
            ranking.read_scoped_component_long(match_dir, "pass_intent")
        except ValueError as exc:
            assert "receive_frame_id" in str(exc)
        else:
            raise AssertionError("missing receive_frame_id scope should fail")


def write_raw_bundesliga_match(season_dir: Path) -> None:
    (season_dir / "match_information").mkdir(parents=True)
    (season_dir / "event_data").mkdir(parents=True)
    (season_dir / "match_information" / f"{MATCH_ID}.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
  <MatchInformation>
    <Teams>
      <Team TeamId="TEAM-1">
        <Players>
          <Player PersonId="P1" Shortname="Starter" Starting="true" PlayingPosition="DM" />
          <Player PersonId="P2" Shortname="Sub" Starting="false" PlayingPosition="CM" />
          <Player PersonId="P3" Shortname="Unused" Starting="false" PlayingPosition="ST" />
        </Players>
      </Team>
    </Teams>
  </MatchInformation>
</PutDataRequest>
""",
        encoding="utf-8",
    )
    (season_dir / "event_data" / f"{MATCH_ID}.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
  <Event EventTime="2025-01-01T15:00:00+00:00"><KickOff GameSection="firstHalf" /></Event>
  <Event EventTime="2025-01-01T15:45:00+00:00"><FinalWhistle GameSection="firstHalf" /></Event>
  <Event EventTime="2025-01-01T16:00:00+00:00"><KickOff GameSection="secondHalf" /></Event>
  <Event EventTime="2025-01-01T16:15:00+00:00">
    <Substitution Team="TEAM-1" PlayerOut="P1" PlayerIn="P2" PlayingPosition="DM" />
  </Event>
  <Event EventTime="2025-01-01T16:45:00+00:00"><FinalWhistle GameSection="secondHalf" /></Event>
</PutDataRequest>
""",
        encoding="utf-8",
    )


def test_minutes_played_derivation_and_cache() -> None:
    with workspace_temp_dir("minutes") as tmp_path:
        season_dir = tmp_path / "Bundesliga_season"
        cache_dir = tmp_path / "minutes_played"
        write_raw_bundesliga_match(season_dir)

        minutes, hits, writes = ranking.read_or_build_minutes_played_cache(
            [MATCH_ID],
            [season_dir],
            cache_dir,
        )
        assert hits == 0
        assert writes == 1
        by_player = minutes.set_index("player_id")["minutes_played"].to_dict()
        assert math.isclose(by_player["P1"], 60.0, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(by_player["P2"], 30.0, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(by_player["P3"], 0.0, rel_tol=1e-9, abs_tol=1e-9)

        cached_minutes, cached_hits, cached_writes = ranking.read_or_build_minutes_played_cache(
            [MATCH_ID],
            [season_dir],
            cache_dir,
        )
        assert cached_hits == 1
        assert cached_writes == 0
        assert cached_minutes[["match_id", "player_id", "minutes_played"]].equals(
            minutes[["match_id", "player_id", "minutes_played"]]
        )


def test_parse_args_accepts_raw_data_and_minutes_cache_options() -> None:
    args = ranking.parse_args(
        [
            "--bundesliga-data-dir",
            "season_a",
            "--bundesliga-data-dir",
            "season_b",
            "--minutes-played-cache-dir",
            "minutes_cache",
            "--refresh-minutes-played-cache",
        ]
    )

    assert [str(path) for path in args.bundesliga_data_dirs] == ["season_a", "season_b"]
    assert str(args.minutes_played_cache_dir) == "minutes_cache"
    assert args.refresh_minutes_played_cache is True


def bundesliga_action_row(
    player_id: str,
    player_name: str,
    match_id: str,
    position: str,
    minutes_played: float,
    metric_value: float,
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "match_id": match_id,
        "advanced_position": position,
        "minutes_played": minutes_played,
        **{column: metric_value for column in ranking.ACTION_METRIC_COLUMNS},
    }


def test_match_and_player_aggregates_include_sums_per90_without_double_counting_minutes() -> None:
    actions = pd.DataFrame(
        [
            bundesliga_action_row("P1", "Player One", "M1", "CM", 45.0, 1.0),
            bundesliga_action_row("P1", "Player One", "M1", "AM", 45.0, 3.0),
            bundesliga_action_row("P1", "Player One", "M2", "DM", 90.0, 5.0),
            bundesliga_action_row("P2", "Player Two", "M3", "CB", 0.0, 7.0),
        ]
    )

    matches = ranking.aggregate_bundesliga_matches(actions)
    players = ranking.aggregate_bundesliga_players(matches, actions)

    assert matches.columns.tolist() == ranking.MATCH_SUMMARY_COLUMNS
    p1_m1 = matches.loc[matches["player_id"].eq("P1") & matches["match_id"].eq("M1")].iloc[0]
    assert p1_m1["advanced_position"] == "AM"
    assert math.isclose(p1_m1["minutes_played"], 45.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1_m1["pass_score_sum"], 4.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1_m1["pass_score_per90"], 8.0, rel_tol=1e-9, abs_tol=1e-9)

    p2_m3 = matches.loc[matches["player_id"].eq("P2") & matches["match_id"].eq("M3")].iloc[0]
    assert math.isclose(p2_m3["pass_score_sum"], 7.0, rel_tol=1e-9, abs_tol=1e-9)
    assert pd.isna(p2_m3["pass_score_per90"])

    p1 = players.loc[players["player_id"].eq("P1")].iloc[0]
    assert int(p1["actions"]) == 3
    assert math.isclose(p1["minutes_played"], 135.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1["pass_score_sum"], 9.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1["pass_score_per90"], 6.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1["pass_score_avg"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1["pass_score_median"], 3.0, rel_tol=1e-9, abs_tol=1e-9)
