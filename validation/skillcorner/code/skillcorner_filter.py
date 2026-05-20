from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = VALIDATION_ROOT / "output"
DEFAULT_SKILLCORNER_DATA_PATH = OUTPUT_DIR / "skillcorner_summary.csv"
DEFAULT_SKILLCORNER_IDS_PATH = OUTPUT_DIR / "skillcorner_id.csv"
PLAYING_TIME_COLUMNS = ["minutes_tip", "minutes_otip", "minutes_played"]
DEFAULT_PLAYING_TIME_COLUMN = "minutes_played"

SKILLCORNER_DATA_REQUIRED_COLUMNS = [
    "player_id",
    "pass_score",
    "risk",
    "reward",
    "game_state_value_start",
    "game_state_value_end",
    "game_state_value_next",
    "action_epv",
    "dm_score",
    "pass_dm_score",
    "carry_epv",
    "pass_epv",
    "z_dm_score",
    "z_pass_dm_score",
    "rank",
    "minute_start",
    "duration",
    "period",
    "player_position",
    "game_state",
    "team_score",
    "opponent_team_score",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "distance_covered",
    "speed_avg",
    "speed_avg_band",
    "separation_start",
    "separation_end",
    "separation_gain",
    "one_touch",
    "quick_pass",
    "carry",
    "pass_outcome",
    "high_pass",
    "player_targeted_xpass_completion",
    "player_targeted_xthreat",
    "end_type",
    "match_id",
    *PLAYING_TIME_COLUMNS,
]
SKILLCORNER_IDS_REQUIRED_COLUMNS = ["player_id", "participant"]
RAW_ACTIONS_FILENAME = "skillcorner_actions_raw.csv"
ACTIONS_FILENAME = "skillcorner_actions.csv"
MATCHES_FILENAME = "skillcorner_matches.csv"
PLAYERS_FILENAME = "skillcorner_players.csv"
ACTIONS_COLUMNS = [
    "participant",
    "pass_score",
    "risk",
    "reward",
    "game_state_value_start",
    "game_state_value_end",
    "game_state_value_next",
    "action_epv",
    "dm_score",
    "pass_dm_score",
    "carry_epv",
    "pass_epv",
    "z_dm_score",
    "z_pass_dm_score",
    "rank",
    "minute_start",
    "duration",
    "period",
    "player_position",
    "game_state",
    "team_score",
    "opponent_team_score",
    "team_in_possession_phase_type",
    "team_out_of_possession_phase_type",
    "distance_covered",
    "speed_avg",
    "speed_avg_band",
    "separation_start",
    "separation_end",
    "separation_gain",
    "one_touch",
    "quick_pass",
    "carry",
    "pass_outcome",
    "high_pass",
    "player_targeted_xpass_completion",
    "player_targeted_xthreat",
    "end_type",
    "match_id",
    *PLAYING_TIME_COLUMNS,
]
ACTION_METRIC_COLUMNS = [
    "pass_score",
    "risk",
    "reward",
    "game_state_value_start",
    "game_state_value_end",
    "game_state_value_next",
    "action_epv",
    "dm_score",
    "pass_dm_score",
    "carry_epv",
    "pass_epv",
    "z_dm_score",
    "z_pass_dm_score",
    "rank",
]
METRIC_SUM_COLUMNS = [f"{column}_sum" for column in ACTION_METRIC_COLUMNS]
METRIC_PER90_COLUMNS = [f"{column}_per90" for column in ACTION_METRIC_COLUMNS]
PLAYER_AVG_MEDIAN_METRIC_COLUMNS = [
    "pass_score",
    "risk",
    "reward",
    "game_state_value_start",
    "action_epv",
    "dm_score",
    "pass_dm_score",
    "carry_epv",
    "pass_epv",
    "z_dm_score",
    "z_pass_dm_score",
    "rank",
]
PLAYER_AVG_MEDIAN_COLUMNS = [
    f"{column}_{suffix}"
    for column in PLAYER_AVG_MEDIAN_METRIC_COLUMNS
    for suffix in ["avg", "median"]
]
MATCH_SUMMARY_COLUMNS = [
    "participant",
    "match_id",
    "player_position",
    *PLAYING_TIME_COLUMNS,
    *[
        output_column
        for metric in ACTION_METRIC_COLUMNS
        for output_column in [f"{metric}_sum", f"{metric}_per90"]
    ],
]
PLAYER_SUMMARY_COLUMNS = [
    "participant",
    "actions",
    *PLAYING_TIME_COLUMNS,
    *[
        output_column
        for metric in ACTION_METRIC_COLUMNS
        for output_column in [f"{metric}_sum", f"{metric}_per90"]
    ],
    *PLAYER_AVG_MEDIAN_COLUMNS,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillcorner-data-path", type=Path, default=DEFAULT_SKILLCORNER_DATA_PATH)
    parser.add_argument("--skillcorner-ids-path", type=Path, default=DEFAULT_SKILLCORNER_IDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--playing-time", choices=PLAYING_TIME_COLUMNS, default=DEFAULT_PLAYING_TIME_COLUMN)
    return parser.parse_args(argv)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing_columns)}")


def coerce_nullable_integer(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_participant(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""))


def read_skillcorner_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SkillCorner summary CSV not found at {path}")
    skillcorner_data = pd.read_csv(path, low_memory=False)
    validate_required_columns(skillcorner_data, SKILLCORNER_DATA_REQUIRED_COLUMNS, "SkillCorner summary")
    skillcorner_data = skillcorner_data.copy()
    skillcorner_data["player_id"] = coerce_nullable_integer(skillcorner_data["player_id"])
    return skillcorner_data


def read_skillcorner_ids(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"SkillCorner ids CSV not found at {path}")
    skillcorner_ids = pd.read_csv(path, low_memory=False)
    validate_required_columns(skillcorner_ids, SKILLCORNER_IDS_REQUIRED_COLUMNS, "SkillCorner ids")
    skillcorner_ids = skillcorner_ids.copy()
    skillcorner_ids["player_id"] = coerce_nullable_integer(skillcorner_ids["player_id"])
    skillcorner_ids["participant"] = normalize_participant(skillcorner_ids["participant"])
    return skillcorner_ids


def filter_skillcorner_ids(skillcorner_ids: pd.DataFrame, expected_rows: int = 63) -> pd.DataFrame:
    filtered = skillcorner_ids.loc[skillcorner_ids["participant"].notna()].copy()
    filtered = filtered.dropna(subset=["player_id"]).copy()
    filtered = filtered.reset_index(drop=True)

    if len(filtered) != expected_rows:
        raise ValueError(
            f"Filtered SkillCorner ids must contain exactly {expected_rows} rows, found {len(filtered)}."
        )
    if filtered["player_id"].duplicated().any():
        raise ValueError("Filtered SkillCorner ids contain duplicate player_id values.")
    if filtered["participant"].duplicated().any():
        raise ValueError("Filtered SkillCorner ids contain duplicate participant values.")
    return filtered


def filter_skillcorner_actions_raw(
    skillcorner_data: pd.DataFrame,
    filtered_skillcorner_ids: pd.DataFrame,
) -> pd.DataFrame:
    valid_player_ids = set(filtered_skillcorner_ids["player_id"].dropna().astype(int).tolist())
    filtered = skillcorner_data.loc[
        skillcorner_data["player_id"].notna() & skillcorner_data["player_id"].astype(int).isin(valid_player_ids)
    ].copy()
    return filtered.reset_index(drop=True)


def build_skillcorner_actions(
    skillcorner_actions_raw: pd.DataFrame,
    filtered_skillcorner_ids: pd.DataFrame,
) -> pd.DataFrame:
    participants = filtered_skillcorner_ids[["player_id", "participant"]].copy()
    merged = skillcorner_actions_raw.merge(participants, on="player_id", how="left")
    merged = merged[ACTIONS_COLUMNS].copy()
    merged = merged.dropna(subset=ACTION_METRIC_COLUMNS, how="all").reset_index(drop=True)
    return merged


def select_dominant_value(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    counts = values.astype("string").value_counts(sort=False)
    max_count = counts.max()
    return sorted(counts[counts.eq(max_count)].index.astype(str).tolist())[0]


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def add_per90_columns(df: pd.DataFrame, playing_time_column: str) -> pd.DataFrame:
    with_per90 = df.copy()
    denominator = pd.to_numeric(with_per90[playing_time_column], errors="coerce")
    valid_denominator = denominator.notna() & denominator.ne(0)
    for metric in ACTION_METRIC_COLUMNS:
        per90_column = f"{metric}_per90"
        with_per90[per90_column] = pd.NA
        with_per90.loc[valid_denominator, per90_column] = (
            with_per90.loc[valid_denominator, f"{metric}_sum"] * (90 / denominator.loc[valid_denominator])
        )
    return with_per90


def aggregate_skillcorner_matches(
    skillcorner_actions: pd.DataFrame,
    playing_time_column: str = DEFAULT_PLAYING_TIME_COLUMN,
) -> pd.DataFrame:
    if playing_time_column not in PLAYING_TIME_COLUMNS:
        raise ValueError(f"Unsupported playing time column: {playing_time_column!r}")
    if skillcorner_actions.empty:
        return pd.DataFrame(columns=MATCH_SUMMARY_COLUMNS)

    grouped = skillcorner_actions.groupby(["participant", "match_id"], dropna=False, as_index=False)
    match_rows = grouped.agg(
        player_position=("player_position", select_dominant_value),
        **{column: (column, first_non_null) for column in PLAYING_TIME_COLUMNS},
        **{f"{column}_sum": (column, "sum") for column in ACTION_METRIC_COLUMNS},
    )
    match_rows = add_per90_columns(match_rows, playing_time_column)
    match_rows = match_rows.sort_values(["participant", "match_id"]).reset_index(drop=True)
    return match_rows[MATCH_SUMMARY_COLUMNS].copy()


def aggregate_skillcorner_players(
    skillcorner_matches: pd.DataFrame,
    skillcorner_actions: pd.DataFrame,
) -> pd.DataFrame:
    if skillcorner_actions.empty:
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)

    match_aggregates = (
        skillcorner_matches.groupby("participant", dropna=False, as_index=False)
        .agg(
            **{column: (column, "sum") for column in PLAYING_TIME_COLUMNS},
            **{column: (column, "sum") for column in METRIC_SUM_COLUMNS},
            **{column: (column, "mean") for column in METRIC_PER90_COLUMNS},
        )
    )
    action_aggregates = (
        skillcorner_actions.groupby("participant", dropna=False, as_index=False)
        .agg(
            actions=("dm_score", "size"),
            **{
                f"{column}_avg": (column, "mean")
                for column in PLAYER_AVG_MEDIAN_METRIC_COLUMNS
            },
            **{
                f"{column}_median": (column, "median")
                for column in PLAYER_AVG_MEDIAN_METRIC_COLUMNS
            },
        )
    )
    aggregated = action_aggregates.merge(match_aggregates, on="participant", how="left")
    aggregated = aggregated.sort_values("participant").reset_index(drop=True)
    return aggregated[PLAYER_SUMMARY_COLUMNS].copy()


def print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def run_skillcorner_filter(
    *,
    skillcorner_data_path: Path = DEFAULT_SKILLCORNER_DATA_PATH,
    skillcorner_ids_path: Path = DEFAULT_SKILLCORNER_IDS_PATH,
    output_dir: Path = OUTPUT_DIR,
    playing_time_column: str = DEFAULT_PLAYING_TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, Path]]:
    skillcorner_data = read_skillcorner_data(skillcorner_data_path)
    skillcorner_ids = read_skillcorner_ids(skillcorner_ids_path)
    filtered_skillcorner_ids = filter_skillcorner_ids(skillcorner_ids)

    skillcorner_actions_raw = filter_skillcorner_actions_raw(skillcorner_data, filtered_skillcorner_ids)
    skillcorner_actions = build_skillcorner_actions(skillcorner_actions_raw, filtered_skillcorner_ids)
    skillcorner_matches = aggregate_skillcorner_matches(skillcorner_actions, playing_time_column)
    skillcorner_players = aggregate_skillcorner_players(skillcorner_matches, skillcorner_actions)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_raw_path = output_dir / RAW_ACTIONS_FILENAME
    actions_path = output_dir / ACTIONS_FILENAME
    matches_path = output_dir / MATCHES_FILENAME
    players_path = output_dir / PLAYERS_FILENAME

    skillcorner_actions_raw.to_csv(actions_raw_path, index=False)
    skillcorner_actions.to_csv(actions_path, index=False)
    skillcorner_matches.to_csv(matches_path, index=False)
    skillcorner_players.to_csv(players_path, index=False)

    summary = {
        "skillcorner_data_rows": len(skillcorner_data),
        "filtered_skillcorner_ids_rows": len(filtered_skillcorner_ids),
        "skillcorner_actions_raw_rows": len(skillcorner_actions_raw),
        "skillcorner_actions_rows": len(skillcorner_actions),
        "skillcorner_matches_rows": len(skillcorner_matches),
        "skillcorner_players_rows": len(skillcorner_players),
        "playing_time_column": playing_time_column,
        "actions_raw_path": actions_raw_path,
        "actions_path": actions_path,
        "matches_path": matches_path,
        "players_path": players_path,
    }
    paths = {
        "actions_raw_path": actions_raw_path,
        "actions_path": actions_path,
        "matches_path": matches_path,
        "players_path": players_path,
    }
    return skillcorner_actions_raw, skillcorner_actions, skillcorner_matches, skillcorner_players, summary, paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, _, _, _, summary, _ = run_skillcorner_filter(
        skillcorner_data_path=args.skillcorner_data_path,
        skillcorner_ids_path=args.skillcorner_ids_path,
        output_dir=args.output_dir,
        playing_time_column=args.playing_time,
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
