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
PASS_SCORE_LOOKUP_COLUMNS = ["match_id", "index", "frame", "receiver_id", "pass_score", "risk", "reward"]
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
    return normalized


def ensure_event_matching_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "match_id" not in normalized.columns:
        normalized["match_id"] = pd.Series(pd.NA, index=normalized.index, dtype="string")
    for column in ["index", "frame_start", "frame_end", "player_targeted_id"]:
        if column not in normalized.columns:
            normalized[column] = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
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
        .groupby([scored["match_id"], scored["index"]], dropna=False)
        .sum(min_count=1)
        .rename("game_state_value")
        .reset_index()
    )
    return scored.merge(game_state_values, on=GAME_STATE_KEY_COLUMNS, how="left")


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
        return pd.DataFrame(columns=PASS_SCORE_LOOKUP_COLUMNS)

    max_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("max")
    candidates = model_data.loc[model_data["frame"].eq(max_frames), PASS_SCORE_LOOKUP_COLUMNS].copy()
    candidates = candidates.dropna(subset=["pass_score"]).reset_index(drop=True)
    validate_unique_lookup(candidates, ["match_id", "index", "frame", "receiver_id"], "pass_score_candidates")
    return candidates


def build_game_state_candidates(model_data: pd.DataFrame) -> pd.DataFrame:
    if model_data.empty:
        return pd.DataFrame(columns=["match_id", "index", "frame", "game_state_value"])

    min_frames = model_data.groupby(GAME_STATE_KEY_COLUMNS, dropna=False)["frame"].transform("min")
    candidates = model_data.loc[model_data["frame"].eq(min_frames), GAME_STATE_LOOKUP_COLUMNS].copy()
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

    game_state_candidates = build_game_state_candidates(model_data)
    game_state_matches = working[["__event_row_id", "match_id", "index", "frame_start"]].merge(
        game_state_candidates,
        on=["match_id", "index"],
        how="left",
    )
    selected_game_state = select_nearest_frame_match(
        game_state_matches,
        event_id_column="__event_row_id",
        event_frame_column="frame_start",
        candidate_frame_column="frame",
        tie_break="earlier",
        value_columns=["game_state_value"],
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
        value_columns=["pass_score", "risk", "reward"],
    )

    scored = base.merge(selected_game_state, on="__event_row_id", how="left")
    scored = scored.merge(selected_pass_score, on="__event_row_id", how="left")
    scored["dm_score"] = scored["pass_score"] - scored["game_state_value"]
    return scored[original_columns + ["pass_score", "risk", "reward", "game_state_value", "dm_score"]]


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    component_run_root, resolved_run_id = resolve_component_run_root(args)
    match_dirs = discover_match_dirs(component_run_root)
    model_data = build_model_data(match_dirs)
    event_data = load_event_data(args.event_data_dir)
    event_data = add_scores_to_event_data(model_data, event_data)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    event_data.to_csv(args.output_file, index=False)

    summary = {
        "component_run_id": resolved_run_id,
        "component_run_root": component_run_root,
        "match_dirs": len(match_dirs),
        "model_rows": len(model_data),
        "event_rows": len(event_data),
        "events_with_pass_score": int(event_data["pass_score"].notna().sum()),
        "events_with_risk": int(event_data["risk"].notna().sum()),
        "events_with_reward": int(event_data["reward"].notna().sum()),
        "events_with_game_state_value": int(event_data["game_state_value"].notna().sum()),
        "events_with_dm_score": int(event_data["dm_score"].notna().sum()),
        "output_file": args.output_file,
    }
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
