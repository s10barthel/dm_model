from pathlib import Path

import pandas as pd


COACH_RATINGS_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\dm_model\validation\coach_ratings\preprocessed_coach_ratings.csv"
)
PASS_SCORES_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\dm_model\data\defcon_components\hawkeye_data.csv"
)
OUTPUT_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\dm_model\validation\coach_ratings\coach_ratings.csv"
)

KEY_COLUMNS = ["id", "uefa_player_id"]
HAWKEYE_OUTPUT_COLUMNS = [
    "pass_score_max",
    "pass_score_avg",
    "pass_score_med",
    "pass_score_br",
    "abs_time",
    "action_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]
RANKING_OUTPUT_COLUMNS = [
    "coach_ranking",
    "model_ranking_max",
    "model_ranking_avg",
    "model_ranking_med",
    "model_ranking_br",
]
HAWKEYE_NUMERIC_COLUMNS = [
    "abs_time",
    "BallReceipt",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{label} is missing required columns: {missing_text}")


def normalize_text_key(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned == "")


def normalize_int_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_key_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    normalized_df = df.copy()
    validate_required_columns(normalized_df, KEY_COLUMNS, label)
    normalized_df["id"] = normalize_text_key(normalized_df["id"])
    normalized_df["uefa_player_id"] = normalize_int_key(normalized_df["uefa_player_id"])
    return normalized_df


def normalize_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
    return normalized_df


def read_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"Loading {label}: {path}")
    return pd.read_csv(path)


def get_scored_row_mask(df: pd.DataFrame) -> tuple[pd.Series, int]:
    validate_required_columns(df, ["Scores"], "Coach ratings")
    scores_text = df["Scores"].astype("string").str.strip()
    scored_mask = df["Scores"].notna() & scores_text.ne("")
    ignored_rows = int((~scored_mask).sum())
    return scored_mask, ignored_rows


def prepare_pass_score_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_required_columns(
        df,
        KEY_COLUMNS + ["action_intent"] + HAWKEYE_NUMERIC_COLUMNS,
        "Pass scores",
    )
    pass_scores_df = normalize_key_columns(df, "Pass scores")
    pass_scores_df = normalize_numeric_columns(pass_scores_df, HAWKEYE_NUMERIC_COLUMNS)
    pass_scores_df = pass_scores_df.dropna(subset=KEY_COLUMNS).copy()

    pass_scores_df["time_norm"] = pass_scores_df["abs_time"] - pass_scores_df["BallReceipt"]
    pass_scores_df["pass_score"] = (
        pass_scores_df["pass_success"]
        * (
            pass_scores_df["outcome_scoring_success"]
            - pass_scores_df["outcome_conceding_success"]
        )
        + (1 - pass_scores_df["pass_success"])
        * (
            pass_scores_df["outcome_scoring_failure"]
            - pass_scores_df["outcome_conceding_failure"]
        )
    )

    filtered_df = pass_scores_df.loc[
        pass_scores_df["time_norm"].between(0, 1, inclusive="both")
    ].copy()
    filtered_df = filtered_df.dropna(subset=["pass_score"]).copy()

    aggregated_scores = (
        filtered_df.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            pass_score_max=("pass_score", "max"),
            pass_score_avg=("pass_score", "mean"),
            pass_score_med=("pass_score", "median"),
        )
    )

    sorted_df = filtered_df.sort_values(
        by=KEY_COLUMNS + ["pass_score"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    best_rows = sorted_df.drop_duplicates(subset=KEY_COLUMNS, keep="first").copy()
    best_rows = best_rows[
        KEY_COLUMNS
        + [
            "abs_time",
            "action_intent",
            "pass_success",
            "outcome_scoring_success",
            "outcome_scoring_failure",
            "outcome_conceding_success",
            "outcome_conceding_failure",
        ]
    ]

    ball_receipt_scores = pass_scores_df.loc[
        pass_scores_df["abs_time"].eq(pass_scores_df["BallReceipt"])
    ].dropna(subset=["pass_score"])
    pass_score_br = (
        ball_receipt_scores.groupby(KEY_COLUMNS, as_index=False)
        .agg(pass_score_br=("pass_score", "max"))
    )

    pass_features_df = aggregated_scores.merge(best_rows, on=KEY_COLUMNS, how="left")
    pass_features_df = pass_features_df.merge(pass_score_br, on=KEY_COLUMNS, how="left")

    stats = {
        "pass_rows_total": len(pass_scores_df),
        "pass_rows_in_time_window": len(filtered_df),
        "pass_key_matches_available": len(pass_features_df),
        "pass_rows_at_ball_receipt": len(ball_receipt_scores),
        "pass_key_matches_ball_receipt": len(pass_score_br),
    }
    return pass_features_df, stats


def merge_pass_scores(
    coach_ratings_df: pd.DataFrame,
    pass_features_df: pd.DataFrame,
    scored_mask: pd.Series,
) -> pd.DataFrame:
    coach_ratings_with_row_id = coach_ratings_df.copy()
    coach_ratings_with_row_id["__row_id"] = range(len(coach_ratings_with_row_id))
    eligible_rows_df = coach_ratings_with_row_id.loc[scored_mask].copy()
    coach_base_columns = coach_ratings_df.columns.tolist()
    pass_columns_to_merge = KEY_COLUMNS + [
        "pass_score_max",
        "pass_score_avg",
        "pass_score_med",
        "pass_score_br",
        "abs_time",
        "action_intent",
        "pass_success",
        "outcome_scoring_success",
        "outcome_scoring_failure",
        "outcome_conceding_success",
        "outcome_conceding_failure",
    ]
    pass_merge_df = pass_features_df[pass_columns_to_merge].rename(
        columns={"abs_time": "pass_abs_time"}
    )
    eligible_rows_df = eligible_rows_df.merge(pass_merge_df, on=KEY_COLUMNS, how="left")
    eligible_rows_df["abs_time"] = eligible_rows_df["pass_abs_time"].combine_first(
        eligible_rows_df["abs_time"]
    )
    eligible_rows_df = eligible_rows_df.drop(columns=["pass_abs_time"])

    merged_df = coach_ratings_with_row_id.merge(
        eligible_rows_df[["__row_id"] + HAWKEYE_OUTPUT_COLUMNS],
        on="__row_id",
        how="left",
        suffixes=("", "_enriched"),
    )
    merged_df = merged_df.drop(columns=["__row_id"])

    ordered_columns = coach_base_columns.copy()
    for column in HAWKEYE_OUTPUT_COLUMNS:
        if column not in ordered_columns:
            ordered_columns.append(column)
    return merged_df[ordered_columns].copy()


def add_rankings(df: pd.DataFrame) -> pd.DataFrame:
    ranked_df = df.copy()
    ranked_df["Scores"] = pd.to_numeric(ranked_df["Scores"], errors="coerce")
    for column in [
        "pass_score_max",
        "pass_score_avg",
        "pass_score_med",
        "pass_score_br",
    ]:
        ranked_df[column] = pd.to_numeric(ranked_df[column], errors="coerce")

    ranked_df["coach_ranking"] = (
        ranked_df.groupby("id")["Scores"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ranked_df["model_ranking_max"] = (
        ranked_df.groupby("id")["pass_score_max"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ranked_df["model_ranking_avg"] = (
        ranked_df.groupby("id")["pass_score_avg"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ranked_df["model_ranking_med"] = (
        ranked_df.groupby("id")["pass_score_med"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ranked_df["model_ranking_br"] = (
        ranked_df.groupby("id")["pass_score_br"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    ordered_columns = df.columns.tolist()
    for column in RANKING_OUTPUT_COLUMNS:
        if column not in ordered_columns:
            ordered_columns.append(column)
    return ranked_df[ordered_columns].copy()


def print_summary(summary: dict[str, int]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> int:
    coach_ratings_df = read_csv(COACH_RATINGS_PATH, "Coach ratings")
    coach_ratings_df = normalize_key_columns(coach_ratings_df, "Coach ratings")
    coach_ratings_df = coach_ratings_df.dropna(subset=KEY_COLUMNS).copy()
    scored_mask, ignored_unscored_rows = get_scored_row_mask(coach_ratings_df)

    pass_scores_df = read_csv(PASS_SCORES_PATH, "Pass scores")
    pass_features_df, pass_stats = prepare_pass_score_features(pass_scores_df)

    output_df = merge_pass_scores(coach_ratings_df, pass_features_df, scored_mask)
    output_df = add_rankings(output_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "coach_rows_total": len(coach_ratings_df),
        "coach_rows_ignored_missing_scores": ignored_unscored_rows,
        "pass_rows_total": pass_stats["pass_rows_total"],
        "pass_rows_in_time_window": pass_stats["pass_rows_in_time_window"],
        "pass_key_matches_available": pass_stats["pass_key_matches_available"],
        "pass_rows_at_ball_receipt": pass_stats["pass_rows_at_ball_receipt"],
        "pass_key_matches_ball_receipt": pass_stats["pass_key_matches_ball_receipt"],
        "output_rows": len(output_df),
        "output_rows_with_pass_score_max": int(output_df["pass_score_max"].notna().sum()),
        "output_rows_with_pass_score_avg": int(output_df["pass_score_avg"].notna().sum()),
        "output_rows_with_pass_score_med": int(output_df["pass_score_med"].notna().sum()),
        "output_rows_with_pass_score_br": int(output_df["pass_score_br"].notna().sum()),
        "output_rows_with_coach_ranking": int(output_df["coach_ranking"].notna().sum()),
        "output_rows_with_model_ranking_max": int(
            output_df["model_ranking_max"].notna().sum()
        ),
        "output_rows_with_model_ranking_avg": int(
            output_df["model_ranking_avg"].notna().sum()
        ),
        "output_rows_with_model_ranking_med": int(
            output_df["model_ranking_med"].notna().sum()
        ),
        "output_rows_with_model_ranking_br": int(
            output_df["model_ranking_br"].notna().sum()
        ),
    }
    print(f"Saved output to: {OUTPUT_PATH}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
