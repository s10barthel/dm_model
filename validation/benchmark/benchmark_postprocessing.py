from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import get_benchmark_component_run_root, resolve_named_component_run_id


VALIDATION_ROOT = Path(__file__).resolve().parent
INPUT_FILENAME = "benchmark_data.csv"
OUTPUT_PATH = VALIDATION_ROOT / "benchmark_summary.csv"

REQUIRED_COLUMNS = [
    "modification",
    "game_state",
    "higher_state_id",
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_conceding_success",
    "outcome_scoring_failure",
    "outcome_conceding_failure",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS.copy()
OUTPUT_COLUMNS = [
    "modification",
    "game_state_value_1",
    "game_state_value_2",
    "higher_state_id",
    "model_rating",
    "agreement",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component-run-id",
        default=None,
        help="Optional benchmark component run id. Defaults to the latest benchmark component run.",
    )
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


def load_benchmark_df(component_run_id: str | None) -> tuple[pd.DataFrame, Path, str]:
    resolved_run_id = resolve_named_component_run_id(
        "benchmark_component",
        component_run_id,
        required=True,
    )
    if resolved_run_id is None:
        raise FileNotFoundError("No benchmark component run id could be resolved.")

    input_path = get_benchmark_component_run_root(resolved_run_id) / INPUT_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Benchmark input CSV not found at {input_path}.")

    print(f"Loading benchmark input: {input_path}")
    return pd.read_csv(input_path), input_path, resolved_run_id


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    coerced_df = df.copy()
    for column in columns:
        coerced_df[column] = pd.to_numeric(coerced_df[column], errors="coerce")
    return coerced_df


def normalize_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in ["modification", "game_state", "higher_state_id"]:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
        if normalized_df[column].isna().any():
            raise ValueError(f"Benchmark input contains missing or invalid values in {column!r}.")
        normalized_df[column] = normalized_df[column].astype("Int64")

    game_states = set(normalized_df["game_state"].dropna().astype(int).unique().tolist())
    if not game_states.issubset({1, 2}):
        raise ValueError(f"Benchmark input contains unsupported game_state values: {sorted(game_states)}")

    higher_state_ids = set(normalized_df["higher_state_id"].dropna().astype(int).unique().tolist())
    if not higher_state_ids.issubset({1, 2}):
        raise ValueError(
            "Benchmark input contains unsupported higher_state_id values: "
            f"{sorted(higher_state_ids)}"
        )

    return normalized_df


def collect_modification_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_pairs = df[["modification", "higher_state_id"]].drop_duplicates().copy()
    label_counts = (
        label_pairs.groupby("modification")["higher_state_id"]
        .nunique(dropna=True)
        .rename("higher_state_count")
    )
    inconsistent = label_counts[label_counts > 1]
    if not inconsistent.empty:
        inconsistent_modifications = ", ".join(str(index) for index in inconsistent.index.tolist())
        raise ValueError(
            "Benchmark input contains inconsistent higher_state_id values for modifications: "
            f"{inconsistent_modifications}"
        )

    return (
        label_pairs.drop_duplicates(subset=["modification"], keep="first")
        .sort_values("modification")
        .reset_index(drop=True)
    )


def filter_pass_intent_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["pass_intent"]).copy()


def compute_game_state_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["modification", "game_state", "game_state_value"])

    scored_df = df.copy()
    scored_df["pass_score"] = (
        scored_df["pass_success"]
        * (
            scored_df["outcome_scoring_success"]
            - scored_df["outcome_conceding_success"]
        )
        + (1 - scored_df["pass_success"])
        * (
            scored_df["outcome_scoring_failure"]
            - scored_df["outcome_conceding_failure"]
        )
    )
    scored_df["weighted_pass_score"] = scored_df["pass_intent"] * scored_df["pass_score"]

    return (
        scored_df.groupby(["modification", "game_state"])["weighted_pass_score"]
        .sum(min_count=1)
        .rename("game_state_value")
        .reset_index()
    )


def pivot_game_state_values(
    modification_labels: pd.DataFrame,
    game_state_values: pd.DataFrame,
) -> pd.DataFrame:
    if game_state_values.empty:
        summary_df = modification_labels.copy()
        summary_df["game_state_value_1"] = pd.NA
        summary_df["game_state_value_2"] = pd.NA
        return summary_df

    pivoted_values = game_state_values.pivot(
        index="modification",
        columns="game_state",
        values="game_state_value",
    ).rename(
        columns={
            1: "game_state_value_1",
            2: "game_state_value_2",
        }
    )
    pivoted_values.columns.name = None
    pivoted_values = pivoted_values.reset_index()

    summary_df = modification_labels.merge(pivoted_values, on="modification", how="left")
    for column in ["game_state_value_1", "game_state_value_2"]:
        if column not in summary_df.columns:
            summary_df[column] = pd.NA
    return summary_df


def add_ratings(summary_df: pd.DataFrame) -> pd.DataFrame:
    rated_df = summary_df.copy()

    game_state_value_1 = rated_df["game_state_value_1"]
    game_state_value_2 = rated_df["game_state_value_2"]
    both_present = game_state_value_1.notna() & game_state_value_2.notna()

    model_rating = pd.Series(pd.NA, index=rated_df.index, dtype="Int64")
    model_rating.loc[both_present & (game_state_value_1 > game_state_value_2)] = 1
    model_rating.loc[both_present & (game_state_value_1 < game_state_value_2)] = 2
    model_rating.loc[both_present & (game_state_value_1 == game_state_value_2)] = 0
    rated_df["model_rating"] = model_rating

    agreement = pd.Series(pd.NA, index=rated_df.index, dtype="Int64")
    comparable = rated_df["model_rating"].notna() & rated_df["model_rating"].ne(0)
    agreement.loc[comparable & rated_df["higher_state_id"].eq(rated_df["model_rating"])] = 1
    agreement.loc[comparable & rated_df["higher_state_id"].ne(rated_df["model_rating"])] = 0
    rated_df["agreement"] = agreement

    return rated_df


def build_benchmark_summary(benchmark_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    benchmark_df = normalize_identifier_columns(benchmark_df)
    modification_labels = collect_modification_labels(benchmark_df)

    filtered_df = filter_pass_intent_rows(benchmark_df)
    game_state_values = compute_game_state_values(filtered_df)
    summary_df = pivot_game_state_values(modification_labels, game_state_values)
    summary_df = add_ratings(summary_df)
    summary_df = summary_df.sort_values("modification").reset_index(drop=True)

    for column in OUTPUT_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = pd.NA
    summary_df = summary_df[OUTPUT_COLUMNS].copy()

    stats = {
        "benchmark_rows_total": len(benchmark_df),
        "benchmark_rows_with_pass_intent": len(filtered_df),
        "benchmark_rows_filtered_missing_pass_intent": len(benchmark_df) - len(filtered_df),
        "modifications_total": len(modification_labels),
        "modifications_with_game_state_value_1": int(summary_df["game_state_value_1"].notna().sum()),
        "modifications_with_game_state_value_2": int(summary_df["game_state_value_2"].notna().sum()),
        "modifications_with_both_game_states": int(
            (summary_df["game_state_value_1"].notna() & summary_df["game_state_value_2"].notna()).sum()
        ),
        "draws": int(summary_df["model_rating"].eq(0).sum()),
        "agreements": int(summary_df["agreement"].eq(1).sum()),
        "disagreements": int(summary_df["agreement"].eq(0).sum()),
        "output_rows": len(summary_df),
    }
    return summary_df, stats


def print_summary(stats: dict[str, int]) -> None:
    print("\nSummary")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    benchmark_df, input_path, resolved_run_id = load_benchmark_df(args.component_run_id)
    validate_required_columns(benchmark_df, REQUIRED_COLUMNS, "Benchmark input")
    benchmark_df = coerce_numeric_columns(benchmark_df, NUMERIC_COLUMNS)

    summary_df, stats = build_benchmark_summary(benchmark_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Resolved benchmark component run id: {resolved_run_id}")
    print(f"Source benchmark CSV: {input_path}")
    print(f"Saved benchmark summary to: {OUTPUT_PATH}")
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
