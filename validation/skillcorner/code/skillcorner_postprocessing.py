from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import (
    PROJECT_ROOT,
    SKILLCORNER_COMPONENT_RUNS_DIR,
    resolve_named_component_run_id,
)


VALIDATION_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = VALIDATION_ROOT / "output"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "skillcorner_summary.csv"
DEFAULT_EVENT_DATA_DIR = PROJECT_ROOT / "skillcorner_data"

REQUIRED_COMPONENTS = [
    "action_intent",
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]
REQUIRED_COMPONENT_IDENTIFIER_COLUMNS = ["match_id", "frame", "index", "player_id"]
IGNORED_COMPONENT_COLUMNS = {"period", "attacking_side", "shot"}
MODEL_KEY_COLUMNS = ["match_id", "frame", "index", "player_id", "receiver_id"]
GAME_STATE_KEY_COLUMNS = ["match_id", "index"]
GAME_STATE_FRAME_KEY_COLUMNS = ["match_id", "index", "frame"]
PASS_SCORE_LOOKUP_COLUMNS = ["match_id", "index", "frame", "receiver_id", "pass_score", "risk", "reward"]
PASS_SCORE_STATS_LOOKUP_COLUMNS = ["match_id", "index", "frame", "pass_score_std"]
GAME_STATE_LOOKUP_COLUMNS = ["match_id", "index", "frame", "game_state_value"]
REQUIRED_EVENT_COLUMNS = [
    "match_id",
    "index",
    "frame_start",
    "frame_end",
    "player_targeted_id",
    "event_type_id",
    "start_type_id",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-run-root", type=Path)
    parser.add_argument("--component-run-id")
    parser.add_argument("--component-runs-dir", type=Path, default=SKILLCORNER_COMPONENT_RUNS_DIR)
    parser.add_argument("--event-data-dir", type=Path, default=DEFAULT_EVENT_DATA_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args(argv)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{label} is missing required columns: {missing_text}")


def normalize_match_id(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        normalized.loc[numeric_mask] = numeric.loc[numeric_mask].astype("Int64").astype("string")
    return normalized.mask(normalized.eq(""))


def coerce_nullable_integer(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_component_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["match_id"] = normalize_match_id(normalized["match_id"])
    for column in ["frame", "index", "player_id"]:
        normalized[column] = coerce_nullable_integer(normalized[column])
    return normalized


def normalize_event_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "match_id" in normalized.columns:
        normalized["match_id"] = normalize_match_id(normalized["match_id"])
    for column in ["index", "frame_start", "frame_end", "player_id", "player_targeted_id", "event_type_id", "start_type_id"]:
        if column in normalized.columns:
            normalized[column] = coerce_nullable_integer(normalized[column])
    if "team_id" in normalized.columns:
        normalized["team_id"] = coerce_nullable_integer(normalized["team_id"])
    return normalized


def ensure_event_matching_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "match_id" not in normalized.columns:
        normalized["match_id"] = pd.Series(pd.NA, index=normalized.index, dtype="string")
    for column in ["index", "frame_start", "frame_end", "player_targeted_id"]:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    if "team_id" not in normalized.columns:
        normalized["team_id"] = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    if "end_type" not in normalized.columns:
        normalized["end_type"] = pd.Series(pd.NA, index=normalized.index, dtype="string")
    return normalize_event_identifiers(normalized)


def prune_component_frames(component: pd.DataFrame) -> pd.DataFrame:
    if component.empty:
        return component.copy()

    sorted_component = component.sort_values(["match_id", "index", "frame"]).reset_index(drop=True)
    grouped_frames = sorted_component.groupby(["match_id", "index"], dropna=False)["frame"]
    keep_mask = sorted_component["frame"].eq(grouped_frames.transform("min")) | sorted_component["frame"].eq(
        grouped_frames.transform("max")
    )
    pruned = sorted_component.loc[keep_mask].copy()
    pruned = pruned.drop_duplicates(subset=REQUIRED_COMPONENT_IDENTIFIER_COLUMNS, keep="first")

    duplicate_mask = pruned.duplicated(subset=REQUIRED_COMPONENT_IDENTIFIER_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise ValueError("Pruned SkillCorner component data still contains duplicate identifier rows.")
    return pruned.reset_index(drop=True)


def melt_component_frame(
    component: pd.DataFrame,
    component_name: str,
    label: str,
) -> pd.DataFrame:
    validate_required_columns(component, REQUIRED_COMPONENT_IDENTIFIER_COLUMNS, label)
    normalized = normalize_component_identifiers(component)
    pruned = prune_component_frames(normalized)

    value_columns = [
        column
        for column in pruned.columns
        if column not in REQUIRED_COMPONENT_IDENTIFIER_COLUMNS and column not in IGNORED_COMPONENT_COLUMNS
    ]
    if not value_columns:
        raise ValueError(f"{label} has no receiver probability columns.")

    melted = pruned.melt(
        id_vars=REQUIRED_COMPONENT_IDENTIFIER_COLUMNS,
        value_vars=value_columns,
        var_name="receiver_id",
        value_name=component_name,
    )
    melted = melted.dropna(subset=[component_name]).copy()
    receiver_labels = melted["receiver_id"].astype("string")
    melted["receiver_id"] = coerce_nullable_integer(melted["receiver_id"])
    invalid_mask = melted["receiver_id"].isna()
    if invalid_mask.any():
        invalid_columns = sorted({str(value) for value in receiver_labels.loc[invalid_mask].tolist()})
        raise ValueError(f"{label} contains non-player receiver columns after filtering: {', '.join(invalid_columns)}")

    duplicate_mask = melted.duplicated(subset=MODEL_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{label} produces duplicate match/frame/index/player/receiver rows.")
    return melted.reset_index(drop=True)


def read_component_long(match_dir: Path, component_name: str) -> pd.DataFrame:
    path = match_dir / f"{component_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing required component file: {path}")
    return melt_component_frame(pd.read_parquet(path), component_name, str(path))


def compute_model_scores(model_data: pd.DataFrame) -> pd.DataFrame:
    scored = model_data.copy()
    if scored.empty:
        scored["pass_score"] = pd.Series(dtype=float)
        scored["risk"] = pd.Series(dtype=float)
        scored["reward"] = pd.Series(dtype=float)
        scored["game_state_value"] = pd.Series(dtype=float)
        return scored

    scored["pass_score"] = (
        scored["pass_success"]
        * (scored["outcome_scoring_success"] - scored["outcome_conceding_success"])
        + (1 - scored["pass_success"])
        * (scored["outcome_scoring_failure"] - scored["outcome_conceding_failure"])
    )
    scored["reward"] = (
        scored["pass_success"] * scored["outcome_scoring_success"]
        + (1 - scored["pass_success"]) * scored["outcome_scoring_failure"]
    )
    scored["risk"] = (
        scored["pass_success"] * scored["outcome_conceding_success"]
        + (1 - scored["pass_success"]) * scored["outcome_conceding_failure"]
    )
    game_state_values = (
        (scored["pass_intent"] * scored["pass_score"])
        .groupby([scored["match_id"], scored["index"], scored["frame"]], dropna=False)
        .sum(min_count=1)
        .rename("game_state_value")
        .reset_index()
    )
    return scored.merge(game_state_values, on=GAME_STATE_FRAME_KEY_COLUMNS, how="left")


def build_match_model_data(match_dir: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component_name in REQUIRED_COMPONENTS:
        component_long = read_component_long(match_dir, component_name)
        if merged is None:
            merged = component_long
        else:
            merged = merged.merge(component_long, on=MODEL_KEY_COLUMNS, how="outer")

    if merged is None:
        raise ValueError(f"No SkillCorner components were loaded for {match_dir}")
    merged = merged.sort_values(MODEL_KEY_COLUMNS).reset_index(drop=True)
    return compute_model_scores(merged)


def build_model_data(match_dirs: list[Path]) -> pd.DataFrame:
    if not match_dirs:
        return pd.DataFrame(
            columns=MODEL_KEY_COLUMNS
            + REQUIRED_COMPONENTS
            + ["pass_score", "risk", "reward", "game_state_value"]
        )
    frames = [build_match_model_data(match_dir) for match_dir in match_dirs]
    return pd.concat(frames, ignore_index=True)


def validate_unique_lookup(df: pd.DataFrame, key_columns: list[str], label: str) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        raise ValueError(f"{label} contains duplicate rows for keys: {', '.join(key_columns)}")


def build_pass_score_candidates(model_data: pd.DataFrame) -> pd.DataFrame:
    if model_data.empty:
        return pd.DataFrame(columns=PASS_SCORE_LOOKUP_COLUMNS + ["rank"])

    max_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("max")
    candidates = model_data.loc[model_data["frame"].eq(max_frames), PASS_SCORE_LOOKUP_COLUMNS].copy()
    candidates = candidates.dropna(subset=["pass_score"]).reset_index(drop=True)
    candidates["rank"] = candidates.groupby(["match_id", "index", "frame"], dropna=False)["pass_score"].rank(
        method="dense",
        ascending=False,
    )
    validate_unique_lookup(candidates, ["match_id", "index", "frame", "receiver_id"], "pass_score_candidates")
    return candidates


def build_pass_score_std_candidates(model_data: pd.DataFrame, frame_position: str) -> pd.DataFrame:
    if model_data.empty:
        return pd.DataFrame(columns=PASS_SCORE_STATS_LOOKUP_COLUMNS)

    if frame_position == "start":
        selected_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("min")
    elif frame_position == "end":
        selected_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("max")
    else:
        raise ValueError(f"Unsupported pass score stats frame position: {frame_position!r}")

    candidates = model_data.loc[model_data["frame"].eq(selected_frames), PASS_SCORE_LOOKUP_COLUMNS].copy()
    candidates = candidates.dropna(subset=["pass_score"]).reset_index(drop=True)
    pass_score_std = (
        candidates.groupby(["match_id", "index", "frame"], dropna=False)["pass_score"]
        .std()
        .rename("pass_score_std")
        .reset_index()
    )
    return pass_score_std.reset_index(drop=True)


def build_game_state_candidates(model_data: pd.DataFrame, frame_position: str) -> pd.DataFrame:
    if model_data.empty:
        return pd.DataFrame(columns=["match_id", "index", "frame", "game_state_value"])

    if frame_position == "start":
        selected_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("min")
    elif frame_position == "end":
        selected_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("max")
    else:
        raise ValueError(f"Unsupported game state frame position: {frame_position!r}")

    candidates = model_data.loc[model_data["frame"].eq(selected_frames), GAME_STATE_LOOKUP_COLUMNS].copy()
    candidates = candidates.dropna(subset=["game_state_value"]).reset_index(drop=True)

    value_counts = (
        candidates.groupby(["match_id", "index", "frame"], dropna=False)["game_state_value"]
        .nunique(dropna=False)
        .rename("distinct_values")
    )
    conflicting = value_counts[value_counts > 1]
    if not conflicting.empty:
        raise ValueError("game_state_candidates contain conflicting game_state_value rows for the same event key.")

    candidates = candidates.drop_duplicates(subset=["match_id", "index", "frame"], keep="first")
    return candidates.reset_index(drop=True)


def rename_value_column(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    renamed = df.copy()
    if source in renamed.columns:
        renamed = renamed.rename(columns={source: target})
    elif target not in renamed.columns:
        renamed[target] = pd.Series(dtype=float)
    return renamed


def read_event_data(event_path: Path) -> pd.DataFrame:
    event_data = pd.read_csv(event_path, low_memory=False)
    validate_required_columns(event_data, REQUIRED_EVENT_COLUMNS, str(event_path))
    return normalize_event_identifiers(event_data)


def filter_event_rows(event_data: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(event_data, ["event_type_id", "start_type_id"], "SkillCorner event data")
    filtered = event_data.loc[event_data["event_type_id"].eq(8) & event_data["start_type_id"].eq(1)].copy()
    return filtered.reset_index(drop=True)


def drop_all_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all").copy()


def load_event_data(event_data_dir: Path) -> pd.DataFrame:
    event_paths = sorted(event_data_dir.glob("*_dynamic_events.csv"))
    if not event_paths:
        raise FileNotFoundError(f"No SkillCorner event CSV files found in {event_data_dir}")

    frames = [read_event_data(event_path) for event_path in event_paths]
    event_data = pd.concat(frames, ignore_index=True)
    event_data = filter_event_rows(event_data)
    event_data = drop_all_empty_columns(event_data)
    return event_data.reset_index(drop=True)


def select_nearest_frame_match(
    candidate_rows: pd.DataFrame,
    *,
    event_id_column: str,
    event_frame_column: str,
    candidate_frame_column: str,
    tie_break: str,
    value_columns: list[str],
) -> pd.DataFrame:
    if candidate_rows.empty:
        return pd.DataFrame(columns=[event_id_column] + value_columns)

    ranked = candidate_rows.dropna(subset=[candidate_frame_column]).copy()
    if ranked.empty:
        return pd.DataFrame(columns=[event_id_column] + value_columns)

    ranked["__frame_distance"] = (ranked[candidate_frame_column] - ranked[event_frame_column]).abs()
    frame_ascending = tie_break == "earlier"
    ranked = ranked.sort_values(
        by=[event_id_column, "__frame_distance", candidate_frame_column],
        ascending=[True, True, frame_ascending],
        kind="mergesort",
    )
    selected = ranked.drop_duplicates(subset=[event_id_column], keep="first")
    return selected[[event_id_column] + value_columns].reset_index(drop=True)


def add_scores_to_event_data(model_data: pd.DataFrame, event_data: pd.DataFrame) -> pd.DataFrame:
    base = event_data.reset_index(drop=True).copy()
    original_columns = base.columns.tolist()
    base["__event_row_id"] = pd.RangeIndex(len(base))
    working = ensure_event_matching_columns(base.copy())
    for column in ["team_id", "end_type", "player_targeted_id"]:
        if column not in base.columns:
            base[column] = working[column]

    game_state_start_candidates = rename_value_column(
        build_game_state_candidates(model_data, "start"),
        "game_state_value",
        "game_state_value_start",
    )
    game_state_matches = working[["__event_row_id", "match_id", "index", "frame_start"]].merge(
        game_state_start_candidates,
        on=["match_id", "index"],
        how="left",
    )
    selected_game_state = select_nearest_frame_match(
        game_state_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_start",
        candidate_frame_column="frame",
        tie_break="earlier",
        value_columns=["game_state_value_start"],
    )

    pass_score_start_std_candidates = rename_value_column(
        build_pass_score_std_candidates(model_data, "start"),
        "pass_score_std",
        "pass_score_std_start",
    )
    pass_score_start_std_matches = working[["__event_row_id", "match_id", "index", "frame_start"]].merge(
        pass_score_start_std_candidates,
        on=["match_id", "index"],
        how="left",
    )
    selected_pass_score_start_std = select_nearest_frame_match(
        pass_score_start_std_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_start",
        candidate_frame_column="frame",
        tie_break="earlier",
        value_columns=["pass_score_std_start"],
    )

    game_state_end_candidates = rename_value_column(
        build_game_state_candidates(model_data, "end"),
        "game_state_value",
        "game_state_value_end",
    )
    game_state_end_matches = working[["__event_row_id", "match_id", "index", "frame_end"]].merge(
        game_state_end_candidates,
        on=["match_id", "index"],
        how="left",
    )
    selected_game_state_end = select_nearest_frame_match(
        game_state_end_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_end",
        candidate_frame_column="frame",
        tie_break="later",
        value_columns=["game_state_value_end"],
    )

    pass_score_end_std_candidates = rename_value_column(
        build_pass_score_std_candidates(model_data, "end"),
        "pass_score_std",
        "pass_score_std_end",
    )
    pass_score_end_std_matches = working[["__event_row_id", "match_id", "index", "frame_end"]].merge(
        pass_score_end_std_candidates,
        on=["match_id", "index"],
        how="left",
    )
    selected_pass_score_end_std = select_nearest_frame_match(
        pass_score_end_std_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_end",
        candidate_frame_column="frame",
        tie_break="later",
        value_columns=["pass_score_std_end"],
    )

    pass_score_candidates = build_pass_score_candidates(model_data)
    pass_score_matches = working[["__event_row_id", "match_id", "index", "frame_end", "player_targeted_id"]].merge(
        pass_score_candidates,
        left_on=["match_id", "index", "player_targeted_id"],
        right_on=["match_id", "index", "receiver_id"],
        how="left",
    )
    selected_pass_score = select_nearest_frame_match(
        pass_score_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_end",
        candidate_frame_column="frame",
        tie_break="later",
        value_columns=["pass_score", "risk", "reward", "rank"],
    )

    scored = base.merge(selected_game_state, on="__event_row_id", how="left")
    scored = scored.merge(selected_pass_score_start_std, on="__event_row_id", how="left")
    scored = scored.merge(selected_game_state_end, on="__event_row_id", how="left")
    scored = scored.merge(selected_pass_score_end_std, on="__event_row_id", how="left")
    scored = scored.merge(selected_pass_score, on="__event_row_id", how="left")

    scored["__game_state_value_next_raw"] = scored.groupby("match_id", dropna=False)[
        "game_state_value_start"
    ].shift(-1)
    scored["__team_id_next"] = scored.groupby("match_id", dropna=False)["team_id"].shift(-1)
    has_signed_next_inputs = (
        scored["__game_state_value_next_raw"].notna()
        & scored["team_id"].notna()
        & scored["__team_id_next"].notna()
    )
    different_next_team_mask = has_signed_next_inputs & scored["team_id"].ne(scored["__team_id_next"]).fillna(False)
    scored["game_state_value_next"] = scored["__game_state_value_next_raw"].where(has_signed_next_inputs)
    scored.loc[different_next_team_mask, "game_state_value_next"] = -scored.loc[
        different_next_team_mask,
        "__game_state_value_next_raw",
    ]
    scored["action_epv"] = scored["game_state_value_next"] - scored["game_state_value_start"]
    scored["dm_score"] = scored["pass_score"] - scored["game_state_value_start"]
    scored["pass_dm_score"] = scored["pass_score"] - scored["game_state_value_end"]
    scored["carry_epv"] = scored["game_state_value_end"] - scored["game_state_value_start"]
    scored["pass_epv"] = scored["game_state_value_next"] - scored["game_state_value_end"]

    pass_score_std_start_values = scored["pass_score_std_start"].dropna()
    stabilizer = pass_score_std_start_values.quantile(0.01) if not pass_score_std_start_values.empty else pd.NA
    scored["z_dm_score"] = pd.NA
    scored["z_pass_dm_score"] = pd.NA
    if pd.notna(stabilizer):
        z_dm_denominator = (scored["pass_score_std_start"].pow(2) + stabilizer**2).pow(0.5)
        z_pass_dm_denominator = (scored["pass_score_std_end"].pow(2) + stabilizer**2).pow(0.5)
        z_dm_mask = z_dm_denominator.notna() & z_dm_denominator.ne(0)
        z_pass_dm_mask = z_pass_dm_denominator.notna() & z_pass_dm_denominator.ne(0)
        scored.loc[z_dm_mask, "z_dm_score"] = (
            scored.loc[z_dm_mask, "dm_score"] / z_dm_denominator.loc[z_dm_mask]
        )
        scored.loc[z_pass_dm_mask, "z_pass_dm_score"] = (
            scored.loc[z_pass_dm_mask, "pass_dm_score"] / z_pass_dm_denominator.loc[z_pass_dm_mask]
        )

    empty_target = scored["player_targeted_id"].isna() & scored["pass_score"].isna()
    has_start_end = scored["game_state_value_start"].notna() & scored["game_state_value_end"].notna()
    foul_mask = empty_target & scored["end_type"].eq("foul_suffered") & has_start_end
    scored.loc[foul_mask, "dm_score"] = (
        scored.loc[foul_mask, "game_state_value_end"] - scored.loc[foul_mask, "game_state_value_start"]
    )

    has_next_inputs = (
        scored["game_state_value_start"].notna()
        & scored["game_state_value_next"].notna()
        & scored["team_id"].notna()
        & scored["__team_id_next"].notna()
    )
    possession_loss_mask = empty_target & scored["end_type"].eq("possession_loss") & has_next_inputs
    scored.loc[possession_loss_mask, "dm_score"] = scored.loc[possession_loss_mask, "action_epv"]

    return scored[
        original_columns
        + [
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
    ]


def resolve_component_run_root(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.component_run_root is not None:
        component_run_root = Path(args.component_run_root)
        resolved_run_id = component_run_root.name
    elif args.component_run_id:
        resolved_run_id = str(args.component_run_id)
        component_run_root = args.component_runs_dir / resolved_run_id
    else:
        resolved_run_id: str | None = None
        if args.component_runs_dir.resolve() == SKILLCORNER_COMPONENT_RUNS_DIR.resolve():
            resolved_run_id = resolve_named_component_run_id("skillcorner_component", required=False)
        else:
            latest_path = args.component_runs_dir / "latest.json"
            if latest_path.exists():
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
                latest_run_id = payload.get("run_id")
                resolved_run_id = str(latest_run_id) if latest_run_id else None

        if resolved_run_id is not None:
            component_run_root = args.component_runs_dir / resolved_run_id
        else:
            candidates = [path for path in sorted(args.component_runs_dir.iterdir()) if path.is_dir()]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "No SkillCorner component run was provided, no latest run could be resolved, "
                    "and the component-runs directory does not contain exactly one candidate run."
                )
            component_run_root = candidates[0]
            resolved_run_id = component_run_root.name

    if not component_run_root.exists():
        raise FileNotFoundError(f"SkillCorner component run root does not exist: {component_run_root}")
    return component_run_root, resolved_run_id


def discover_match_dirs(component_run_root: Path) -> list[Path]:
    match_dirs = [path for path in sorted(component_run_root.iterdir()) if path.is_dir()]
    if not match_dirs:
        raise FileNotFoundError(f"No match directories were found under {component_run_root}")
    return match_dirs


def print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def summarize_scored_events(
    event_data: pd.DataFrame,
    *,
    resolved_run_id: str | None,
    component_run_root: Path,
    match_dirs: list[Path],
    model_data: pd.DataFrame,
    output_file: Path,
) -> dict[str, object]:
    return {
        "component_run_id": resolved_run_id,
        "component_run_root": component_run_root,
        "match_dirs": len(match_dirs),
        "model_rows": len(model_data),
        "event_rows": len(event_data),
        "events_with_pass_score": int(event_data["pass_score"].notna().sum()),
        "events_with_risk": int(event_data["risk"].notna().sum()),
        "events_with_reward": int(event_data["reward"].notna().sum()),
        "events_with_game_state_value_start": int(event_data["game_state_value_start"].notna().sum()),
        "events_with_game_state_value_end": int(event_data["game_state_value_end"].notna().sum()),
        "events_with_game_state_value_next": int(event_data["game_state_value_next"].notna().sum()),
        "events_with_action_epv": int(event_data["action_epv"].notna().sum()),
        "events_with_dm_score": int(event_data["dm_score"].notna().sum()),
        "events_with_pass_dm_score": int(event_data["pass_dm_score"].notna().sum()),
        "events_with_carry_epv": int(event_data["carry_epv"].notna().sum()),
        "events_with_pass_epv": int(event_data["pass_epv"].notna().sum()),
        "events_with_z_dm_score": int(event_data["z_dm_score"].notna().sum()),
        "events_with_z_pass_dm_score": int(event_data["z_pass_dm_score"].notna().sum()),
        "events_with_rank": int(event_data["rank"].notna().sum()),
        "output_file": output_file,
    }


def run_skillcorner_postprocessing(
    *,
    component_run_root: Path | None = None,
    component_run_id: str | None = None,
    component_runs_dir: Path = SKILLCORNER_COMPONENT_RUNS_DIR,
    event_data_dir: Path = DEFAULT_EVENT_DATA_DIR,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    args = argparse.Namespace(
        component_run_root=component_run_root,
        component_run_id=component_run_id,
        component_runs_dir=component_runs_dir,
        event_data_dir=event_data_dir,
        output_file=output_file,
    )
    resolved_component_run_root, resolved_run_id = resolve_component_run_root(args)
    match_dirs = discover_match_dirs(resolved_component_run_root)
    model_data = build_model_data(match_dirs)
    event_data = load_event_data(args.event_data_dir)
    event_data = add_scores_to_event_data(model_data, event_data)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_data.to_csv(output_path, index=False)

    summary = summarize_scored_events(
        event_data,
        resolved_run_id=resolved_run_id,
        component_run_root=resolved_component_run_root,
        match_dirs=match_dirs,
        model_data=model_data,
        output_file=output_path,
    )
    return event_data, summary, output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event_data, summary, _ = run_skillcorner_postprocessing(
        component_run_root=args.component_run_root,
        component_run_id=args.component_run_id,
        component_runs_dir=args.component_runs_dir,
        event_data_dir=args.event_data_dir,
        output_file=args.output_file,
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
