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

SKILLCORNER_DATA_REQUIRED_COLUMNS = [
    "player_id",
    "pass_score",
    "risk",
    "reward",
    "game_state_value",
    "dm_score",
    "match_id",
]
SKILLCORNER_IDS_REQUIRED_COLUMNS = ["player_id", "participant"]
RAW_ACTIONS_FILENAME = "skillcorner_actions_raw.csv"
ACTIONS_FILENAME = "skillcorner_actions.csv"
PLAYERS_FILENAME = "skillcorner_players.csv"
ACTIONS_COLUMNS = [
    "participant",
    "pass_score",
    "risk",
    "reward",
    "game_state_value",
    "dm_score",
    "match_id",
]
PLAYER_SUMMARY_COLUMNS = [
    "participant",
    "actions",
    "pass_score_sum",
    "pass_score_avg",
    "pass_score_median",
    "risk_sum",
    "risk_avg",
    "risk_median",
    "reward_sum",
    "reward_avg",
    "reward_median",
    "game_state_value_sum",
    "game_state_value_avg",
    "game_state_value_median",
    "dm_score_sum",
    "dm_score_avg",
    "dm_score_median",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillcorner-data-path", type=Path, default=DEFAULT_SKILLCORNER_DATA_PATH)
    parser.add_argument("--skillcorner-ids-path", type=Path, default=DEFAULT_SKILLCORNER_IDS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
    merged = merged.dropna(subset=["dm_score"]).reset_index(drop=True)
    return merged


def aggregate_skillcorner_players(skillcorner_actions: pd.DataFrame) -> pd.DataFrame:
    if skillcorner_actions.empty:
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)

    aggregated = (
        skillcorner_actions.groupby("participant", dropna=False, as_index=False)
        .agg(
            actions=("dm_score", "size"),
            pass_score_sum=("pass_score", "sum"),
            pass_score_avg=("pass_score", "mean"),
            pass_score_median=("pass_score", "median"),
            risk_sum=("risk", "sum"),
            risk_avg=("risk", "mean"),
            risk_median=("risk", "median"),
            reward_sum=("reward", "sum"),
            reward_avg=("reward", "mean"),
            reward_median=("reward", "median"),
            game_state_value_sum=("game_state_value", "sum"),
            game_state_value_avg=("game_state_value", "mean"),
            game_state_value_median=("game_state_value", "median"),
            dm_score_sum=("dm_score", "sum"),
            dm_score_avg=("dm_score", "mean"),
            dm_score_median=("dm_score", "median"),
        )
        .sort_values("participant")
        .reset_index(drop=True)
    )
    return aggregated[PLAYER_SUMMARY_COLUMNS].copy()


def print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skillcorner_data = read_skillcorner_data(args.skillcorner_data_path)
    skillcorner_ids = read_skillcorner_ids(args.skillcorner_ids_path)
    filtered_skillcorner_ids = filter_skillcorner_ids(skillcorner_ids)

    skillcorner_actions_raw = filter_skillcorner_actions_raw(skillcorner_data, filtered_skillcorner_ids)
    skillcorner_actions = build_skillcorner_actions(skillcorner_actions_raw, filtered_skillcorner_ids)
    skillcorner_players = aggregate_skillcorner_players(skillcorner_actions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions_raw_path = args.output_dir / RAW_ACTIONS_FILENAME
    actions_path = args.output_dir / ACTIONS_FILENAME
    players_path = args.output_dir / PLAYERS_FILENAME

    skillcorner_actions_raw.to_csv(actions_raw_path, index=False)
    skillcorner_actions.to_csv(actions_path, index=False)
    skillcorner_players.to_csv(players_path, index=False)

    summary = {
        "skillcorner_data_rows": len(skillcorner_data),
        "filtered_skillcorner_ids_rows": len(filtered_skillcorner_ids),
        "skillcorner_actions_raw_rows": len(skillcorner_actions_raw),
        "skillcorner_actions_rows": len(skillcorner_actions),
        "skillcorner_players_rows": len(skillcorner_players),
        "actions_raw_path": actions_raw_path,
        "actions_path": actions_path,
        "players_path": players_path,
    }
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
