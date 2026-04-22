from pathlib import Path

import pandas as pd


PATRICK_CSV_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\sequence_files\files_patrick\coach_ratings\xy_data_patrick.csv"
)
JELLE_CSV_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VS Code\.vscode\sequence_files\files_jelle\coach_ratings\xy_data_jelle.csv"
)
PATRICK_XLSX_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VU\mrp\Patrick\coach_ratings.xlsx"
)
JELLE_METADATA_XLSX_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VU\mrp\Jelle\coach_ratings_metadata.xlsx"
)
JELLE_RATINGS_CSV_PATH = Path(
    r"C:\Users\steffen.barthel\OneDrive - TSG 1899 Hoffenheim Fußball-Spielbetriebs GmbH\Dokumente\VU\mrp\Jelle\coach_ratings.csv"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "preprocessed_coach_ratings.csv"

KEY_COLUMNS = ["uefa_player_id", "id"]
FINAL_EXTRA_COLUMNS = [
    "QNr",
    "PNr",
    "Scores",
    "Coach1",
    "Coach2",
    "Coach3",
    "Coach4",
    "Coach5",
    "Coach6",
    "mrp",
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{label} is missing required columns: {missing_text}")


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def normalize_text_key(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned == "")


def normalize_int_key(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    validate_required_columns(normalized_df, KEY_COLUMNS, "DataFrame")
    normalized_df["uefa_player_id"] = normalize_int_key(normalized_df["uefa_player_id"])
    normalized_df["id"] = normalize_text_key(normalized_df["id"])
    return normalized_df


def normalize_nullable_int_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns:
            normalized_df[column] = normalize_int_key(normalized_df[column])
    return normalized_df


def first_non_null(series: pd.Series):
    non_null_values = series.dropna()
    if not non_null_values.empty:
        return non_null_values.iloc[0]
    return pd.NA


def collapse_duplicate_rows(
    df: pd.DataFrame,
    key_columns: list[str],
    label: str,
) -> tuple[pd.DataFrame, int]:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows == 0:
        return df.copy(), 0

    aggregated_df = (
        df.groupby(key_columns, dropna=False, as_index=False)
        .agg({column: first_non_null for column in df.columns if column not in key_columns})
    )
    resolved_groups = int(df.loc[duplicate_mask, key_columns].drop_duplicates().shape[0])
    print(
        f"{label}: resolved {duplicate_rows} duplicate rows across "
        f"{resolved_groups} duplicate keys."
    )
    return aggregated_df, duplicate_rows


def read_base_csv(path: Path, label: str) -> pd.DataFrame:
    print(f"Loading {label}: {path}")
    df = pd.read_csv(path, sep=";")
    validate_required_columns(df, KEY_COLUMNS, label)
    return normalize_key_columns(df)


def read_excel_first_sheet(path: Path, label: str) -> pd.DataFrame:
    print(f"Loading {label}: {path}")
    df = pd.read_excel(path, sheet_name=0)
    return drop_unnamed_columns(df)


def read_delimited_csv(path: Path, label: str, delimiter: str = ";") -> pd.DataFrame:
    print(f"Loading {label}: {path}")
    df = pd.read_csv(path, sep=delimiter)
    return drop_unnamed_columns(df)


def prepare_patrick_enrichment() -> tuple[pd.DataFrame, dict[str, int]]:
    patrick_xlsx = read_excel_first_sheet(PATRICK_XLSX_PATH, "Patrick workbook")
    patrick_xlsx = patrick_xlsx.rename(columns={"Qnr": "QNr", "Pnr": "PNr"})
    validate_required_columns(
        patrick_xlsx,
        ["uefa_player_id", "id", "QNr", "PNr", "Scores"],
        "Patrick workbook",
    )

    patrick_xlsx = patrick_xlsx[["uefa_player_id", "id", "QNr", "PNr", "Scores"]].copy()
    patrick_xlsx = normalize_key_columns(patrick_xlsx)
    patrick_xlsx = normalize_nullable_int_columns(patrick_xlsx, ["QNr", "PNr"])

    original_rows = len(patrick_xlsx)
    patrick_xlsx = patrick_xlsx.dropna(subset=KEY_COLUMNS).copy()
    dropped_invalid_rows = original_rows - len(patrick_xlsx)

    patrick_xlsx, duplicate_rows_resolved = collapse_duplicate_rows(
        patrick_xlsx,
        KEY_COLUMNS,
        "Patrick workbook",
    )

    stats = {
        "patrick_workbook_rows": original_rows,
        "patrick_invalid_rows_dropped": dropped_invalid_rows,
        "patrick_duplicate_rows_resolved": duplicate_rows_resolved,
    }
    return patrick_xlsx, stats


def prepare_jelle_helper() -> tuple[pd.DataFrame, dict[str, int]]:
    jelle_metadata = read_excel_first_sheet(
        JELLE_METADATA_XLSX_PATH,
        "Jelle metadata workbook",
    )
    validate_required_columns(
        jelle_metadata,
        ["uefa_player_id", "id", "QNr", "PNr"],
        "Jelle metadata workbook",
    )
    jelle_metadata = jelle_metadata[["uefa_player_id", "id", "QNr", "PNr"]].copy()
    jelle_metadata = normalize_key_columns(jelle_metadata)
    jelle_metadata = normalize_nullable_int_columns(jelle_metadata, ["QNr", "PNr"])
    jelle_metadata = jelle_metadata.dropna(subset=KEY_COLUMNS).copy()

    jelle_metadata, metadata_duplicate_rows_resolved = collapse_duplicate_rows(
        jelle_metadata,
        KEY_COLUMNS,
        "Jelle metadata workbook",
    )

    jelle_ratings = read_delimited_csv(
        JELLE_RATINGS_CSV_PATH,
        "Jelle ratings workbook",
    )
    validate_required_columns(
        jelle_ratings,
        [
            "QuestionNr",
            "Player",
            "Average_rating",
            "Coach1",
            "Coach2",
            "Coach3",
            "Coach4",
            "Coach5",
            "Coach6",
        ],
        "Jelle ratings workbook",
    )
    jelle_ratings = jelle_ratings[
        [
            "QuestionNr",
            "Player",
            "Average_rating",
            "Coach1",
            "Coach2",
            "Coach3",
            "Coach4",
            "Coach5",
            "Coach6",
        ]
    ].copy()
    jelle_ratings = normalize_nullable_int_columns(jelle_ratings, ["QuestionNr", "Player"])
    jelle_ratings = jelle_ratings.dropna(subset=["QuestionNr", "Player"]).copy()
    jelle_ratings, ratings_duplicate_rows_resolved = collapse_duplicate_rows(
        jelle_ratings,
        ["QuestionNr", "Player"],
        "Jelle ratings workbook",
    )

    df_helper = jelle_metadata.merge(
        jelle_ratings,
        left_on=["QNr", "PNr"],
        right_on=["QuestionNr", "Player"],
        how="left",
    )
    df_helper = df_helper.drop(columns=["QuestionNr", "Player"])

    stats = {
        "jelle_metadata_rows": len(jelle_metadata),
        "jelle_metadata_duplicate_rows_resolved": metadata_duplicate_rows_resolved,
        "jelle_ratings_rows": len(jelle_ratings),
        "jelle_ratings_duplicate_rows_resolved": ratings_duplicate_rows_resolved,
        "df_helper_rows": len(df_helper),
        "df_helper_rows_with_rating": int(df_helper["Average_rating"].notna().sum()),
    }
    return df_helper, stats


def enrich_patrick_rows(base_df: pd.DataFrame, enrichment_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    merged_df = base_df.merge(enrichment_df, on=KEY_COLUMNS, how="left")
    for column in ["Coach1", "Coach2", "Coach3", "Coach4", "Coach5", "Coach6"]:
        merged_df[column] = pd.NA
    merged_df["mrp"] = "patrick"

    stats = {
        "patrick_base_rows": len(base_df),
        "patrick_rows_with_qnr": int(merged_df["QNr"].notna().sum()),
        "patrick_rows_with_scores": int(merged_df["Scores"].notna().sum()),
    }
    return merged_df, stats


def enrich_jelle_rows(base_df: pd.DataFrame, df_helper: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    helper_subset = df_helper[
        [
            "uefa_player_id",
            "id",
            "QNr",
            "PNr",
            "Average_rating",
            "Coach1",
            "Coach2",
            "Coach3",
            "Coach4",
            "Coach5",
            "Coach6",
        ]
    ].copy()

    merged_df = base_df.merge(helper_subset, on=KEY_COLUMNS, how="left")
    merged_df["Scores"] = merged_df.pop("Average_rating")
    merged_df["mrp"] = "jelle"

    stats = {
        "jelle_base_rows": len(base_df),
        "jelle_rows_with_qnr": int(merged_df["QNr"].notna().sum()),
        "jelle_rows_with_scores": int(merged_df["Scores"].notna().sum()),
        "jelle_rows_with_any_coach_rating": int(
            merged_df[["Coach1", "Coach2", "Coach3", "Coach4", "Coach5", "Coach6"]]
            .notna()
            .any(axis=1)
            .sum()
        ),
    }
    return merged_df, stats


def order_final_columns(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    ordered_columns = base_columns.copy()
    for column in FINAL_EXTRA_COLUMNS:
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)
    return df[ordered_columns].copy()


def concatenate_rows(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    combined_records: list[dict] = []
    for df in dataframes:
        combined_records.extend(df.to_dict(orient="records"))
    return pd.DataFrame.from_records(combined_records)


def finalize_export_types(df: pd.DataFrame) -> pd.DataFrame:
    finalized_df = df.copy()
    for column in ["QNr", "PNr"]:
        if column in finalized_df.columns:
            finalized_df[column] = normalize_int_key(finalized_df[column])
    return finalized_df


def print_summary(summary: dict[str, int]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> int:
    patrick_base = read_base_csv(PATRICK_CSV_PATH, "Patrick base CSV")
    jelle_base = read_base_csv(JELLE_CSV_PATH, "Jelle base CSV")
    base_columns = patrick_base.columns.tolist()

    patrick_enrichment, patrick_enrichment_stats = prepare_patrick_enrichment()
    df_helper, jelle_helper_stats = prepare_jelle_helper()

    patrick_df, patrick_stats = enrich_patrick_rows(patrick_base, patrick_enrichment)
    jelle_df, jelle_stats = enrich_jelle_rows(jelle_base, df_helper)

    df = concatenate_rows([patrick_df, jelle_df])
    df = order_final_columns(df, base_columns)
    df = finalize_export_types(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "patrick_base_rows": patrick_stats["patrick_base_rows"],
        "jelle_base_rows": jelle_stats["jelle_base_rows"],
        "patrick_invalid_rows_dropped": patrick_enrichment_stats["patrick_invalid_rows_dropped"],
        "patrick_duplicate_rows_resolved": patrick_enrichment_stats[
            "patrick_duplicate_rows_resolved"
        ],
        "patrick_rows_with_qnr": patrick_stats["patrick_rows_with_qnr"],
        "patrick_rows_with_scores": patrick_stats["patrick_rows_with_scores"],
        "df_helper_rows": jelle_helper_stats["df_helper_rows"],
        "df_helper_rows_with_rating": jelle_helper_stats["df_helper_rows_with_rating"],
        "jelle_rows_with_qnr": jelle_stats["jelle_rows_with_qnr"],
        "jelle_rows_with_scores": jelle_stats["jelle_rows_with_scores"],
        "jelle_rows_with_any_coach_rating": jelle_stats["jelle_rows_with_any_coach_rating"],
    }
    print(f"Saved output to: {OUTPUT_PATH}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
