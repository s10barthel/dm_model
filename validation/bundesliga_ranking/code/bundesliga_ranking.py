from __future__ import annotations

import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd


RANKING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = RANKING_ROOT / "output"

LINEUP_PATH = PROJECT_ROOT / "data" / "lineup" / "line_up.parquet"
EVENT_SYNCED_DIR = PROJECT_ROOT / "data" / "event_synced"
COMPONENT_RUNS_DIR = PROJECT_ROOT / "data" / "component_runs" / "sportec"
COMPONENT_LATEST_PATH = COMPONENT_RUNS_DIR / "latest.json"
FC25_RATINGS_PATH = OUTPUT_DIR / "fc25_ratings.csv"
DEFAULT_MINUTES_PLAYED_CACHE_DIR = RANKING_ROOT / "minutes_played"
DEFAULT_BUNDESLIGA_DATA_DIRS = [
    PROJECT_ROOT / "Bundesliga_season_23_24",
    PROJECT_ROOT / "Bundesliga_season_24_25",
]

COMPONENT_BASE_IDENTIFIER_COLUMNS = [
    "stats_perform_match_id",
    "action_id",
    "original_event_id",
]
COMPONENT_IDENTIFIER_COLUMNS = COMPONENT_BASE_IDENTIFIER_COLUMNS
FRAME_SCOPE_COLUMN = "frame_scope"
STATE_FRAME_ID_COLUMN = "state_frame_id"
FRAME_ID_SCOPE = "frame_id"
RECEIVE_FRAME_ID_SCOPE = "receive_frame_id"
SCOPED_COMPONENT_IDENTIFIER_COLUMNS = COMPONENT_BASE_IDENTIFIER_COLUMNS + [FRAME_SCOPE_COLUMN]
ACTION_JOIN_COLUMNS = COMPONENT_BASE_IDENTIFIER_COLUMNS
REQUIRED_SCOPED_COMPONENTS = [
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]
OPTIONAL_COMPONENTS = ["success_intent"]
IGNORED_COMPONENT_COLUMNS = {"home_goal", "away_goal"}
PASS_ACTION_TYPES = {"pass", "cross"}
SPECIAL_COMPONENT_DIRS = {"benchmark", "hawkeye", "skillcorner", "sportec"}
MATCHES_FILENAME = "bundesliga_matches.csv"
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
    "player_id",
    "player_name",
    "match_id",
    "advanced_position",
    "minutes_played",
    *[
        output_column
        for metric in ACTION_METRIC_COLUMNS
        for output_column in [f"{metric}_sum", f"{metric}_per90"]
    ],
]
PLAYER_SUMMARY_COLUMNS = [
    "player_id",
    "player_name",
    "actions",
    "minutes_played",
    *[
        output_column
        for metric in ACTION_METRIC_COLUMNS
        for output_column in [f"{metric}_sum", f"{metric}_per90"]
    ],
    *PLAYER_AVG_MEDIAN_COLUMNS,
]
POSITION_SUMMARY_COLUMNS = [
    "player_id",
    "advanced_position",
    "player_name",
    "actions",
    *METRIC_SUM_COLUMNS,
    *PLAYER_AVG_MEDIAN_COLUMNS,
]
TEAM_NAME_ALIASES = {
    "sport club freiburg": "sc freiburg",
    "tsg hoffenheim": "tsg hoffenheim",
    "tsg 1899 hoffenheim": "tsg hoffenheim",
}
NAME_PARTICLES = {"al", "el", "de", "del", "der", "den", "van", "von", "da", "di", "la", "le", "du", "dos", "das", "ter", "te"}
MATCH_METHOD_PRIORITY = {
    "exact_name": 500,
    "token_subset": 400,
    "initial_surname": 300,
    "token_prefix_subset": 200,
}

# FC25 stores some player names in non-Latin scripts. Keep aliases local and
# explicit so weak fuzzy matches are not forced.
FC25_PLAYER_ALIASES = {
    237633: "Budu Zivzivadze",
    277799: "Andrej Ilic",
    239138: "Woo Yeong Jeong",
    73078: "Kaishu Sano",
    221671: "Jae Sung Lee",
    244108: "Hyun Seok Hong",
    74749: "Artem Stepanov",
    255223: "Amine Adli",
    229476: "Waldemar Anton",
    76107: "Elias Benkara",
    224196: "Ramy Bensebaini",
    233152: "Ko Itakura",
    70149: "Shio Fukuda",
    75810: "Amil Siljevic",
    225126: "Ellyes Skhiri",
    270670: "Fares Chaibi",
    218339: "Mahmoud Dahoud",
    236457: "Dimitrios Giannoulis",
    234205: "Hiroki Ito",
    237086: "Min Jae Kim",
    222542: "Manolis Saliakas",
    274000: "Elias Saad",
    276167: "Marko Ivezic",
    245622: "Shuto Machino",
    74207: "Kosta Nedeljkovic",
    232639: "Ritsu Doan",
    221354: "Milos Veljkovic",
    269951: "Thomas Kastanaras",
    247360: "Leonidas Stergiou",
    263573: "Ameen Al Dakhil",
    70752: "Anrie Chase",
    246708: "Georgios Masouras",
    238473: "Erhan Masovic",
    224811: "Ivan Ordets",
    233151: "Koji Miyoshi",
    270077: "Konstantinos Koulierakis",
    264697: "Mohamed Amoura",
    204638: "Willi Orban",
}

TRANSLITERATION_MAP = str.maketrans(
    {
        "\u00f8": "o",
        "\u00d8": "O",
        "\u0111": "d",
        "\u0110": "D",
        "\u00f0": "d",
        "\u00d0": "D",
        "\u00fe": "th",
        "\u00de": "Th",
        "\u0142": "l",
        "\u0141": "L",
        "\u00df": "ss",
        "\u00e6": "ae",
        "\u00c6": "Ae",
        "\u0153": "oe",
        "\u0152": "Oe",
        "\u0131": "i",
        "\u0130": "I",
        "\u011f": "g",
        "\u011e": "G",
        "\u015f": "s",
        "\u015e": "S",
        "\u0107": "c",
        "\u0106": "C",
        "\u010d": "c",
        "\u010c": "C",
        "\u00f1": "n",
        "\u00d1": "N",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season-prefix", default="DFL-MAT-J04")
    parser.add_argument("--component-run-id")
    parser.add_argument("--component-run-root", type=Path)
    parser.add_argument("--component-runs-dir", type=Path, default=COMPONENT_RUNS_DIR)
    parser.add_argument("--event-synced-dir", type=Path, default=EVENT_SYNCED_DIR)
    parser.add_argument("--lineup-path", type=Path, default=LINEUP_PATH)
    parser.add_argument("--fc25-ratings-path", type=Path, default=FC25_RATINGS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--bundesliga-data-dir",
        action="append",
        type=Path,
        dest="bundesliga_data_dirs",
        help="Raw Bundesliga season directory used for minutes-played derivation. Can be passed more than once.",
    )
    parser.add_argument("--minutes-played-cache-dir", type=Path, default=DEFAULT_MINUTES_PLAYED_CACHE_DIR)
    parser.add_argument("--refresh-minutes-played-cache", action="store_true")
    return parser.parse_args(argv)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    label: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing_columns)}")


def normalize_identifier(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_original_event_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype("string")
    return normalize_identifier(series)


def normalize_object_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def normalize_frame_scope(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def coerce_nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def select_dominant_value(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    counts = values.astype("string").value_counts(sort=False)
    max_count = counts.max()
    return sorted(counts[counts.eq(max_count)].index.astype(str).tolist())[0]


def false_success_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return ~series.fillna(False)

    text = series.astype("string").str.strip().str.casefold()
    return series.isna() | text.isin({"", "false", "0", "no", "nan", "none"})


def fold_to_ascii(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).translate(TRANSLITERATION_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def team_key(value: object) -> str:
    text = fold_to_ascii(value)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(ev|e v)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_NAME_ALIASES.get(text, text)


def tokens(value: object) -> list[str]:
    return fold_to_ascii(value).split()


def fc25_alias_names(row: pd.Series) -> list[str]:
    names = [str(row["fc_player_name"])]
    alias = FC25_PLAYER_ALIASES.get(int(row["fc_player_id"]))
    if alias:
        names.append(alias)
    return names


def meaningful_tokens(value: object) -> list[str]:
    return [token for token in tokens(value) if token not in NAME_PARTICLES]


def surname_token(value: object) -> str:
    for token in reversed(tokens(value)):
        if len(token) > 1 and token not in NAME_PARTICLES:
            return token
    value_tokens = tokens(value)
    return value_tokens[-1] if value_tokens else ""


def fc25_search_tokens(names: list[str]) -> set[str]:
    search_tokens: set[str] = set()
    for name in names:
        search_tokens.update(tokens(name))
    return search_tokens


def fc25_surnames(names: list[str]) -> set[str]:
    surnames: set[str] = set()
    for name in names:
        surnames.update(meaningful_tokens(name))
    return surnames


def token_is_represented(token: str, right_tokens: set[str], allow_prefix: bool) -> bool:
    if len(token) == 1:
        return any(right_token.startswith(token) for right_token in right_tokens)
    if token in right_tokens:
        return True
    return allow_prefix and len(token) >= 3 and any(right_token.startswith(token) for right_token in right_tokens)


def classify_name_match(left_name: object, right_names: list[str]) -> tuple[str | None, int]:
    left_name_key = fold_to_ascii(left_name)
    left_tokens = meaningful_tokens(left_name)
    left_surname = surname_token(left_name)
    if not left_name_key or not left_tokens or not left_surname:
        return None, 0

    right_tokens = fc25_search_tokens(right_names)
    right_surnames = fc25_surnames(right_names)
    right_name_keys = {fold_to_ascii(name) for name in right_names}
    if left_name_key in right_name_keys:
        return "exact_name", len(left_tokens)

    if left_surname not in right_tokens:
        return None, 0

    subset_tokens = [token for token in left_tokens if len(token) > 1]
    if subset_tokens and all(token in right_tokens for token in subset_tokens):
        return "token_subset", len(subset_tokens)

    if len(left_tokens[0]) == 1 and any(token.startswith(left_tokens[0]) for token in right_tokens):
        return "initial_surname", len(left_tokens)

    if all(token_is_represented(token, right_tokens, allow_prefix=True) for token in left_tokens):
        return "token_prefix_subset", len(left_tokens)

    return None, 0


def read_bundesliga_ids(lineup_path: Path, season_prefix: str) -> pd.DataFrame:
    columns = [
        "stats_perform_match_id",
        "game_date",
        "player_id",
        "shirt_number",
        "match_name",
        "contestant_name",
    ]
    bundesliga_ids = pd.read_parquet(lineup_path, columns=columns)
    validate_required_columns(bundesliga_ids, columns, "Bundesliga lineup")
    bundesliga_ids = bundesliga_ids.loc[
        bundesliga_ids["stats_perform_match_id"].astype(str).str.startswith(season_prefix)
    ].copy()
    if bundesliga_ids.empty:
        raise ValueError(f"No lineup rows found for season prefix {season_prefix!r}.")

    bundesliga_ids["game_date"] = pd.to_datetime(bundesliga_ids["game_date"], errors="coerce")
    bundesliga_ids = bundesliga_ids.sort_values(["player_id", "game_date"], kind="mergesort")
    bundesliga_ids = bundesliga_ids.drop_duplicates(subset=["player_id"], keep="last")
    bundesliga_ids = bundesliga_ids.rename(
        columns={
            "match_name": "player_name",
            "contestant_name": "team_name",
        }
    )
    return bundesliga_ids[["player_id", "shirt_number", "player_name", "team_name"]].reset_index(drop=True)


def read_fc25_ratings(fc25_ratings_path: Path) -> pd.DataFrame:
    fc25_ratings = pd.read_csv(fc25_ratings_path)
    validate_required_columns(
        fc25_ratings,
        ["player_id", "player", "team", "overallrating"],
        "FC25 ratings",
    )
    return fc25_ratings.rename(
        columns={
            "player_id": "fc_player_id",
            "player": "fc_player_name",
            "team": "fc_team_name",
        }
    )


def add_fc25_matches(
    bundesliga_ids: pd.DataFrame,
    fc25_ratings: pd.DataFrame,
) -> pd.DataFrame:
    bundesliga_ids = bundesliga_ids.copy().reset_index(drop=True)
    fc25_ratings = fc25_ratings.copy().reset_index(drop=True)
    fc25_ratings["_alias_names"] = fc25_ratings.apply(fc25_alias_names, axis=1)
    fc25_ratings["_team_key"] = fc25_ratings["fc_team_name"].map(team_key)
    fc25_ratings["_search_tokens"] = fc25_ratings["_alias_names"].map(fc25_search_tokens)
    fc25_ratings["_surnames"] = fc25_ratings["_alias_names"].map(fc25_surnames)

    bundesliga_ids["_team_key"] = bundesliga_ids["team_name"].map(team_key)
    bundesliga_ids["_surname"] = bundesliga_ids["player_name"].map(surname_token)

    candidates: list[tuple[int, int, int, float, int, int, str]] = []
    for left_index, bundesliga_row in bundesliga_ids.iterrows():
        same_team = fc25_ratings.index[
            fc25_ratings["_team_key"].eq(bundesliga_row["_team_key"])
        ].tolist()
        for right_index in same_team:
            method, represented_tokens = classify_name_match(
                bundesliga_row["player_name"],
                fc25_ratings.at[right_index, "_alias_names"],
            )
            if method is None:
                continue
            extra_tokens = max(len(fc25_ratings.at[right_index, "_search_tokens"]) - represented_tokens, 0)
            rating = pd.to_numeric(fc25_ratings.at[right_index, "overallrating"], errors="coerce")
            rating_value = float(rating) if pd.notna(rating) else 0.0
            candidates.append(
                (
                    MATCH_METHOD_PRIORITY[method],
                    represented_tokens,
                    -extra_tokens,
                    rating_value,
                    int(left_index),
                    int(right_index),
                    method,
                )
            )
    candidates.sort(reverse=True)

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    selected: dict[int, tuple[int, str, float]] = {}
    for priority, represented_tokens, _extra_tokens, _rating, left_index, right_index, method in candidates:
        if left_index in matched_left or right_index in matched_right:
            continue
        selected[left_index] = (right_index, method, priority + represented_tokens / 10)
        matched_left.add(left_index)
        matched_right.add(right_index)

    unmatched_left_by_key: dict[tuple[str, str], list[int]] = {}
    for left_index, bundesliga_row in bundesliga_ids.iterrows():
        if left_index in matched_left:
            continue
        surname = bundesliga_row["_surname"]
        if not surname:
            continue
        key = (str(bundesliga_row["_team_key"]), str(surname))
        unmatched_left_by_key.setdefault(key, []).append(int(left_index))

    unmatched_right_by_key: dict[tuple[str, str], list[int]] = {}
    for right_index, fc25_row in fc25_ratings.iterrows():
        if right_index in matched_right:
            continue
        for surname in fc25_row["_surnames"]:
            if not surname:
                continue
            key = (str(fc25_row["_team_key"]), str(surname))
            unmatched_right_by_key.setdefault(key, []).append(int(right_index))

    for key, left_indices in sorted(unmatched_left_by_key.items()):
        right_indices = unmatched_right_by_key.get(key, [])
        if len(left_indices) != 1 or len(right_indices) != 1:
            continue
        left_index = left_indices[0]
        right_index = right_indices[0]
        if left_index in matched_left or right_index in matched_right:
            continue
        selected[left_index] = (right_index, "unique_surname", 100.0)
        matched_left.add(left_index)
        matched_right.add(right_index)

    rows: list[dict[str, object]] = []
    helper_columns = {"_alias_names", "_team_key", "_search_tokens", "_surnames"}
    fc_columns = [column for column in fc25_ratings.columns if column not in helper_columns]
    for left_index, bundesliga_row in bundesliga_ids.iterrows():
        row = {
            column: bundesliga_row[column]
            for column in bundesliga_ids.columns
            if not column.startswith("_")
        }
        match = selected.get(left_index)
        if match is None:
            for column in fc_columns:
                row[column] = pd.NA
            row["fc_match_player_similarity"] = pd.NA
            row["fc_match_team_similarity"] = pd.NA
            row["fc_match_score"] = pd.NA
            row["fc_match_method"] = pd.NA
            row["fc_match_status"] = "unmatched"
        else:
            right_index, method, match_score = match
            for column in fc_columns:
                row[column] = fc25_ratings.at[right_index, column]
            row["fc_match_player_similarity"] = 1.0
            row["fc_match_team_similarity"] = 1.0
            row["fc_match_score"] = match_score
            row["fc_match_method"] = method
            row["fc_match_status"] = "matched"
        rows.append(row)

    return pd.DataFrame(rows)


def resolve_component_run_root(args: argparse.Namespace) -> Path:
    if args.component_run_root is not None:
        component_run_root = args.component_run_root
    elif args.component_run_id:
        component_run_root = args.component_runs_dir / str(args.component_run_id)
    else:
        latest_run_id = None
        latest_path = args.component_runs_dir / COMPONENT_LATEST_PATH.name
        if latest_path.exists():
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_run_id = payload.get("run_id")
        if latest_run_id:
            component_run_root = args.component_runs_dir / str(latest_run_id)
        else:
            candidates = [
                path
                for path in args.component_runs_dir.iterdir()
                if path.is_dir() and path.name not in SPECIAL_COMPONENT_DIRS
            ]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "No component run was provided and a single component run could not be inferred."
                )
            component_run_root = candidates[0]

    if not component_run_root.exists():
        raise FileNotFoundError(f"Component run root does not exist: {component_run_root}")
    return component_run_root


def discover_match_dirs(
    component_run_root: Path,
    event_synced_dir: Path,
    season_prefix: str,
) -> list[Path]:
    match_dirs = [
        path
        for path in sorted(component_run_root.iterdir())
        if path.is_dir()
        and path.name.startswith(season_prefix)
        and (event_synced_dir / f"{path.name}.csv").exists()
    ]
    if not match_dirs:
        raise FileNotFoundError(
            f"No component match directories under {component_run_root} match {season_prefix!r} "
            f"and have corresponding event CSV files."
        )
    return match_dirs


def normalize_component_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["stats_perform_match_id"] = normalize_identifier(normalized["stats_perform_match_id"])
    normalized["original_event_id"] = normalize_original_event_id(normalized["original_event_id"])
    normalized["action_id"] = pd.to_numeric(normalized["action_id"], errors="coerce").astype("Int64")
    if FRAME_SCOPE_COLUMN in normalized.columns:
        normalized[FRAME_SCOPE_COLUMN] = normalize_frame_scope(normalized[FRAME_SCOPE_COLUMN])
    return normalized


def component_value_columns(component: pd.DataFrame, id_columns: list[str]) -> list[str]:
    ignored_columns = set(id_columns) | IGNORED_COMPONENT_COLUMNS | {STATE_FRAME_ID_COLUMN}
    return [column for column in component.columns if column not in ignored_columns]


def read_scoped_component_long(match_dir: Path, component_name: str) -> pd.DataFrame:
    path = match_dir / f"{component_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing required component file: {path}")

    component = pd.read_parquet(path)
    required_columns = COMPONENT_BASE_IDENTIFIER_COLUMNS + [FRAME_SCOPE_COLUMN]
    validate_required_columns(component, required_columns, str(path))
    component = normalize_component_identifiers(component)
    observed_scopes = set(component[FRAME_SCOPE_COLUMN].dropna().astype(str).tolist())
    required_scopes = {FRAME_ID_SCOPE, RECEIVE_FRAME_ID_SCOPE}
    missing_scopes = sorted(required_scopes - observed_scopes)
    if missing_scopes:
        raise ValueError(
            f"{path} is missing required frame_scope values: {', '.join(missing_scopes)}. "
            "Regenerate the component run with scoped frame_id/receive_frame_id predictions."
        )

    id_columns = COMPONENT_BASE_IDENTIFIER_COLUMNS + [FRAME_SCOPE_COLUMN]
    if STATE_FRAME_ID_COLUMN in component.columns:
        id_columns.append(STATE_FRAME_ID_COLUMN)
    value_columns = component_value_columns(component, id_columns)
    if not value_columns:
        raise ValueError(f"{path} has no object/player probability columns.")

    long_component = component.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="object_id",
        value_name=component_name,
    )
    long_component["object_id"] = normalize_object_id(long_component["object_id"])
    long_component = long_component.dropna(subset=["object_id", component_name]).copy()
    duplicate_mask = long_component.duplicated(
        subset=SCOPED_COMPONENT_IDENTIFIER_COLUMNS + ["object_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(f"{path} produces duplicate identifier/scope/object rows.")
    return long_component[SCOPED_COMPONENT_IDENTIFIER_COLUMNS + ["object_id", component_name]]


def read_frame_component_long(match_dir: Path, component_name: str) -> pd.DataFrame:
    path = match_dir / f"{component_name}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=COMPONENT_BASE_IDENTIFIER_COLUMNS + ["object_id", component_name])

    component = pd.read_parquet(path)
    validate_required_columns(component, COMPONENT_BASE_IDENTIFIER_COLUMNS, str(path))
    component = normalize_component_identifiers(component)
    if FRAME_SCOPE_COLUMN in component.columns:
        component = component.loc[component[FRAME_SCOPE_COLUMN].eq(FRAME_ID_SCOPE)].copy()

    id_columns = COMPONENT_BASE_IDENTIFIER_COLUMNS.copy()
    if FRAME_SCOPE_COLUMN in component.columns:
        id_columns.append(FRAME_SCOPE_COLUMN)
    if STATE_FRAME_ID_COLUMN in component.columns:
        id_columns.append(STATE_FRAME_ID_COLUMN)

    value_columns = component_value_columns(component, id_columns)
    if not value_columns:
        raise ValueError(f"{path} has no object/player probability columns.")

    long_component = component.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="object_id",
        value_name=component_name,
    )
    long_component["object_id"] = normalize_object_id(long_component["object_id"])
    long_component = long_component.dropna(subset=["object_id", component_name]).copy()
    duplicate_mask = long_component.duplicated(
        subset=COMPONENT_BASE_IDENTIFIER_COLUMNS + ["object_id"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(f"{path} produces duplicate identifier/object rows.")
    return long_component[COMPONENT_BASE_IDENTIFIER_COLUMNS + ["object_id", component_name]]


def build_match_model_data(match_dir: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for component_name in REQUIRED_SCOPED_COMPONENTS:
        component = read_scoped_component_long(match_dir, component_name)
        if merged is None:
            merged = component
        else:
            merged = merged.merge(
                component,
                on=SCOPED_COMPONENT_IDENTIFIER_COLUMNS + ["object_id"],
                how="outer",
            )
    if merged is None:
        raise ValueError(f"No component data loaded for {match_dir}")

    merged["pass_score"] = (
        merged["pass_success"]
        * (merged["outcome_scoring_success"] - merged["outcome_conceding_success"])
        + (1 - merged["pass_success"])
        * (merged["outcome_scoring_failure"] - merged["outcome_conceding_failure"])
    )
    merged["reward"] = (
        merged["pass_success"] * merged["outcome_scoring_success"]
        + (1 - merged["pass_success"]) * merged["outcome_scoring_failure"]
    )
    merged["risk"] = (
        merged["pass_success"] * merged["outcome_conceding_success"]
        + (1 - merged["pass_success"]) * merged["outcome_conceding_failure"]
    )
    game_state_values = (
        (merged["pass_intent"] * merged["pass_score"])
        .groupby(
            [merged[column] for column in SCOPED_COMPONENT_IDENTIFIER_COLUMNS],
            dropna=False,
        )
        .sum(min_count=1)
        .rename("game_state_value")
        .reset_index()
    )
    pass_score_std = (
        merged.groupby(SCOPED_COMPONENT_IDENTIFIER_COLUMNS, dropna=False)["pass_score"]
        .std()
        .rename("pass_score_std")
        .reset_index()
    )
    merged = merged.merge(game_state_values, on=SCOPED_COMPONENT_IDENTIFIER_COLUMNS, how="left")
    merged = merged.merge(pass_score_std, on=SCOPED_COMPONENT_IDENTIFIER_COLUMNS, how="left")

    frame_mask = merged[FRAME_SCOPE_COLUMN].eq(FRAME_ID_SCOPE)
    merged["rank"] = pd.NA
    merged.loc[frame_mask, "rank"] = merged.loc[frame_mask].groupby(
        SCOPED_COMPONENT_IDENTIFIER_COLUMNS,
        dropna=False,
    )["pass_score"].rank(method="dense", ascending=False)

    success_intent = read_frame_component_long(match_dir, OPTIONAL_COMPONENTS[0])
    if not success_intent.empty:
        merged = merged.merge(
            success_intent,
            on=COMPONENT_BASE_IDENTIFIER_COLUMNS + ["object_id"],
            how="left",
        )
    return merged.reset_index(drop=True)


def build_model_data(match_dirs: list[Path]) -> pd.DataFrame:
    model_frames = [build_match_model_data(match_dir) for match_dir in match_dirs]
    return pd.concat(model_frames, ignore_index=True)


def read_pass_cross_events(event_path: Path) -> pd.DataFrame:
    events = pd.read_csv(event_path)
    validate_required_columns(
        events,
        [
            "stats_perform_match_id",
            "action_id",
            "original_event_id",
            "period_id",
            "seconds",
            "frame_id",
            "receive_frame_id",
            "spadl_type",
            "success",
            "receiver_id",
            "player_id",
            "object_id",
            "player_name",
            "advanced_position",
            "team_id",
        ],
        str(event_path),
    )
    events = events.loc[events["spadl_type"].isin(PASS_ACTION_TYPES)].copy()
    events["stats_perform_match_id"] = normalize_identifier(events["stats_perform_match_id"])
    events["original_event_id"] = normalize_original_event_id(events["original_event_id"])
    events["action_id"] = pd.to_numeric(events["action_id"], errors="coerce").astype("Int64")
    events["receiver_id"] = normalize_object_id(events["receiver_id"])
    events["object_id"] = normalize_object_id(events["object_id"])
    events["player_id"] = normalize_identifier(events["player_id"])
    events["receiver_id_original"] = events["receiver_id"]
    return events


def add_scores_to_events(model_data: pd.DataFrame, event_synced_dir: Path) -> pd.DataFrame:
    match_ids = sorted(model_data["stats_perform_match_id"].dropna().unique().tolist())
    event_frames = [
        read_pass_cross_events(event_synced_dir / f"{match_id}.csv")
        for match_id in match_ids
        if (event_synced_dir / f"{match_id}.csv").exists()
    ]
    if not event_frames:
        raise ValueError("No event CSV files were available for model data matches.")
    bundesliga_actions = pd.concat(event_frames, ignore_index=True)

    frame_model = model_data.loc[model_data[FRAME_SCOPE_COLUMN].eq(FRAME_ID_SCOPE)].copy()
    receive_model = model_data.loc[model_data[FRAME_SCOPE_COLUMN].eq(RECEIVE_FRAME_ID_SCOPE)].copy()
    if frame_model.empty or receive_model.empty:
        raise ValueError("Scoped model data must contain both frame_id and receive_frame_id rows.")

    if "success_intent" in frame_model.columns:
        best_intent_receivers = (
            frame_model.dropna(subset=["success_intent"])
            .sort_values(
                by=ACTION_JOIN_COLUMNS + ["success_intent"],
                ascending=[True] * len(ACTION_JOIN_COLUMNS) + [False],
                kind="mergesort",
            )
            .drop_duplicates(subset=ACTION_JOIN_COLUMNS, keep="first")
            [ACTION_JOIN_COLUMNS + ["object_id"]]
            .rename(columns={"object_id": "model_receiver_id"})
        )
        bundesliga_actions = bundesliga_actions.merge(
            best_intent_receivers,
            on=ACTION_JOIN_COLUMNS,
            how="left",
        )
    else:
        print(
            "INFO: success_intent component is unavailable; failed or receiver-missing "
            "passes will keep blank model_receiver_id values and may remain unscored."
        )
        bundesliga_actions["model_receiver_id"] = pd.NA

    receiver_missing = bundesliga_actions["receiver_id"].isna()
    success_false = false_success_mask(bundesliga_actions["success"])
    replace_receiver = (receiver_missing | success_false) & bundesliga_actions["model_receiver_id"].notna()
    bundesliga_actions.loc[replace_receiver, "receiver_id"] = bundesliga_actions.loc[
        replace_receiver,
        "model_receiver_id",
    ]
    bundesliga_actions["receiver_id"] = normalize_object_id(bundesliga_actions["receiver_id"])

    target_lookup = frame_model[
        ACTION_JOIN_COLUMNS + ["object_id", "pass_score", "risk", "reward", "rank"]
    ].rename(columns={"object_id": "receiver_id"})
    target_lookup["receiver_id"] = normalize_object_id(target_lookup["receiver_id"])
    bundesliga_actions = bundesliga_actions.merge(
        target_lookup,
        on=ACTION_JOIN_COLUMNS + ["receiver_id"],
        how="left",
    )

    frame_state_lookup = frame_model[
        ACTION_JOIN_COLUMNS + ["game_state_value", "pass_score_std"]
    ].drop_duplicates(subset=ACTION_JOIN_COLUMNS)
    frame_state_lookup = frame_state_lookup.rename(
        columns={
            "game_state_value": "game_state_value_end",
            "pass_score_std": "pass_score_std_end",
        }
    )
    receive_state_lookup = receive_model[
        ACTION_JOIN_COLUMNS + ["game_state_value", "pass_score_std"]
    ].drop_duplicates(subset=ACTION_JOIN_COLUMNS)
    receive_state_lookup = receive_state_lookup.rename(
        columns={
            "game_state_value": "game_state_value_next",
            "pass_score_std": "pass_score_std_next",
        }
    )
    bundesliga_actions = bundesliga_actions.merge(frame_state_lookup, on=ACTION_JOIN_COLUMNS, how="left")
    bundesliga_actions = bundesliga_actions.merge(receive_state_lookup, on=ACTION_JOIN_COLUMNS, how="left")

    original_order = pd.Series(range(len(bundesliga_actions)), index=bundesliga_actions.index)
    bundesliga_actions["__original_order"] = original_order
    bundesliga_actions = bundesliga_actions.sort_values(
        ["stats_perform_match_id", "action_id", "__original_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = bundesliga_actions.groupby("stats_perform_match_id", dropna=False)
    bundesliga_actions["__previous_receiver_id"] = grouped["receiver_id"].shift(1)
    bundesliga_actions["__previous_game_state_value_next"] = grouped["game_state_value_next"].shift(1)
    bundesliga_actions["__previous_pass_score_std_next"] = grouped["pass_score_std_next"].shift(1)
    same_possessor = bundesliga_actions["__previous_receiver_id"].eq(bundesliga_actions["object_id"]).fillna(False)
    bundesliga_actions["game_state_value_start"] = bundesliga_actions["__previous_game_state_value_next"].where(
        same_possessor
    )
    bundesliga_actions["pass_score_std_start"] = bundesliga_actions["__previous_pass_score_std_next"].where(
        same_possessor
    )

    bundesliga_actions["action_epv"] = (
        bundesliga_actions["game_state_value_next"] - bundesliga_actions["game_state_value_start"]
    )
    bundesliga_actions["dm_score"] = bundesliga_actions["pass_score"] - bundesliga_actions["game_state_value_start"]
    bundesliga_actions["pass_dm_score"] = (
        bundesliga_actions["pass_score"] - bundesliga_actions["game_state_value_end"]
    )
    bundesliga_actions["carry_epv"] = (
        bundesliga_actions["game_state_value_end"] - bundesliga_actions["game_state_value_start"]
    )
    bundesliga_actions["pass_epv"] = (
        bundesliga_actions["game_state_value_next"] - bundesliga_actions["game_state_value_end"]
    )

    pass_score_std_start_values = bundesliga_actions["pass_score_std_start"].dropna()
    stabilizer = pass_score_std_start_values.quantile(0.01) if not pass_score_std_start_values.empty else pd.NA
    bundesliga_actions["z_dm_score"] = pd.NA
    bundesliga_actions["z_pass_dm_score"] = pd.NA
    if pd.notna(stabilizer):
        z_dm_denominator = (bundesliga_actions["pass_score_std_start"].pow(2) + stabilizer**2).pow(0.5)
        z_pass_dm_denominator = (bundesliga_actions["pass_score_std_end"].pow(2) + stabilizer**2).pow(0.5)
        z_dm_mask = z_dm_denominator.notna() & z_dm_denominator.ne(0)
        z_pass_dm_mask = z_pass_dm_denominator.notna() & z_pass_dm_denominator.ne(0)
        bundesliga_actions.loc[z_dm_mask, "z_dm_score"] = (
            bundesliga_actions.loc[z_dm_mask, "dm_score"] / z_dm_denominator.loc[z_dm_mask]
        )
        bundesliga_actions.loc[z_pass_dm_mask, "z_pass_dm_score"] = (
            bundesliga_actions.loc[z_pass_dm_mask, "pass_dm_score"] / z_pass_dm_denominator.loc[z_pass_dm_mask]
        )

    bundesliga_actions = bundesliga_actions.sort_values("__original_order", kind="mergesort").reset_index(drop=True)
    helper_columns = [
        "__original_order",
        "__previous_receiver_id",
        "__previous_game_state_value_next",
        "__previous_pass_score_std_next",
        "pass_score_std_start",
        "pass_score_std_end",
        "pass_score_std_next",
    ]
    return bundesliga_actions.drop(columns=[column for column in helper_columns if column in bundesliga_actions.columns])


def resolve_raw_match_path(match_id: str, bundesliga_data_dirs: list[Path], *relative_candidates: str) -> Path | None:
    for data_dir in bundesliga_data_dirs:
        for relative_candidate in relative_candidates:
            path = data_dir / relative_candidate.format(match_id=match_id)
            if path.exists():
                return path
    return None


def parse_event_time(value: str | None) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True)


def parse_match_lineup(match_info_path: Path, match_id: str) -> pd.DataFrame:
    root = ET.parse(match_info_path).getroot()
    rows: list[dict[str, object]] = []
    for team in root.iter("Team"):
        team_id = team.attrib.get("TeamId")
        for player in team.findall(".//Player"):
            player_id = player.attrib.get("PersonId")
            if not player_id:
                continue
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": player_id,
                    "team_id": team_id,
                    "player_name": player.attrib.get("Shortname")
                    or " ".join(
                        part
                        for part in [player.attrib.get("FirstName"), player.attrib.get("LastName")]
                        if part
                    ),
                    "player_position": player.attrib.get("PlayingPosition"),
                    "starting": str(player.attrib.get("Starting", "")).casefold() == "true",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "player_id",
            "team_id",
            "player_name",
            "player_position",
            "starting",
        ],
    )


def parse_match_timing_and_substitutions(event_path: Path) -> tuple[dict[str, tuple[pd.Timestamp, pd.Timestamp]], list[dict[str, object]]]:
    root = ET.parse(event_path).getroot()
    kickoffs: dict[str, pd.Timestamp] = {}
    final_whistles: dict[str, pd.Timestamp] = {}
    substitutions: list[dict[str, object]] = []

    for event in root.iter("Event"):
        event_time = parse_event_time(event.attrib.get("EventTime"))
        if pd.isna(event_time):
            continue
        for child in list(event):
            if child.tag == "KickOff":
                game_section = child.attrib.get("GameSection")
                if game_section in {"firstHalf", "secondHalf"} and game_section not in kickoffs:
                    kickoffs[game_section] = event_time
            elif child.tag == "FinalWhistle":
                game_section = child.attrib.get("GameSection")
                if game_section in {"firstHalf", "secondHalf"}:
                    final_whistles[game_section] = event_time
            elif child.tag == "Substitution":
                substitutions.append(
                    {
                        "event_time": event_time,
                        "team_id": child.attrib.get("Team"),
                        "player_out": child.attrib.get("PlayerOut"),
                        "player_in": child.attrib.get("PlayerIn"),
                        "player_position": child.attrib.get("PlayingPosition"),
                    }
                )

    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for section in ["firstHalf", "secondHalf"]:
        if section not in kickoffs or section not in final_whistles:
            raise ValueError(f"{event_path} is missing kickoff/final-whistle timing for {section}.")
        periods[section] = (kickoffs[section], final_whistles[section])
    return periods, substitutions


def timestamp_to_match_seconds(timestamp: pd.Timestamp, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> float | None:
    first_start, first_end = periods["firstHalf"]
    second_start, second_end = periods["secondHalf"]
    first_duration = (first_end - first_start).total_seconds()

    if first_start <= timestamp <= first_end:
        return max((timestamp - first_start).total_seconds(), 0.0)
    if second_start <= timestamp <= second_end:
        return first_duration + max((timestamp - second_start).total_seconds(), 0.0)
    return None


def derive_match_minutes_played(match_id: str, bundesliga_data_dirs: list[Path]) -> pd.DataFrame:
    match_info_path = resolve_raw_match_path(
        match_id,
        bundesliga_data_dirs,
        "match_information/{match_id}.xml",
        "match_information/starting_players/{match_id}",
        "match_information/starting_players/{match_id}.xml",
    )
    event_path = resolve_raw_match_path(
        match_id,
        bundesliga_data_dirs,
        "event_data/{match_id}.xml",
        "event_data/{match_id}",
    )
    if match_info_path is None:
        raise FileNotFoundError(f"Could not find raw match-information file for {match_id}.")
    if event_path is None:
        raise FileNotFoundError(f"Could not find raw event-data file for {match_id}.")

    lineup = parse_match_lineup(match_info_path, match_id)
    periods, substitutions = parse_match_timing_and_substitutions(event_path)
    total_seconds = sum((end - start).total_seconds() for start, end in periods.values())

    players: dict[str, dict[str, object]] = {}
    on_field: dict[str, bool] = {}
    open_start: dict[str, float | None] = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    for row in lineup.to_dict("records"):
        player_id = str(row["player_id"])
        players[player_id] = row
        on_field[player_id] = bool(row["starting"])
        open_start[player_id] = 0.0 if on_field[player_id] else None
        intervals[player_id] = []

    substitutions = sorted(substitutions, key=lambda row: row["event_time"])
    for substitution in substitutions:
        elapsed = timestamp_to_match_seconds(substitution["event_time"], periods)
        if elapsed is None:
            continue
        elapsed = min(max(float(elapsed), 0.0), total_seconds)
        player_out = substitution.get("player_out")
        player_in = substitution.get("player_in")

        if player_out:
            player_out = str(player_out)
            players.setdefault(
                player_out,
                {
                    "match_id": match_id,
                    "player_id": player_out,
                    "team_id": substitution.get("team_id"),
                    "player_name": pd.NA,
                    "player_position": substitution.get("player_position"),
                    "starting": pd.NA,
                },
            )
            intervals.setdefault(player_out, [])
            if on_field.get(player_out) and open_start.get(player_out) is not None:
                intervals[player_out].append((float(open_start[player_out]), elapsed))
            on_field[player_out] = False
            open_start[player_out] = None

        if player_in:
            player_in = str(player_in)
            player = players.setdefault(
                player_in,
                {
                    "match_id": match_id,
                    "player_id": player_in,
                    "team_id": substitution.get("team_id"),
                    "player_name": pd.NA,
                    "player_position": substitution.get("player_position"),
                    "starting": False,
                },
            )
            if pd.isna(player.get("player_position")) and substitution.get("player_position"):
                player["player_position"] = substitution.get("player_position")
            intervals.setdefault(player_in, [])
            if not on_field.get(player_in, False):
                open_start[player_in] = elapsed
            on_field[player_in] = True

    for player_id, is_on_field in list(on_field.items()):
        if is_on_field and open_start.get(player_id) is not None:
            intervals.setdefault(player_id, []).append((float(open_start[player_id]), total_seconds))

    rows: list[dict[str, object]] = []
    for player_id, player in players.items():
        played_seconds = sum(max(end - start, 0.0) for start, end in intervals.get(player_id, []))
        player_intervals = intervals.get(player_id, [])
        rows.append(
            {
                **player,
                "start_time": min((start for start, _end in player_intervals), default=pd.NA),
                "end_time": max((end for _start, end in player_intervals), default=pd.NA),
                "minutes_played": played_seconds / 60.0,
            }
        )
    return pd.DataFrame(rows)


def read_or_build_minutes_played_cache(
    match_ids: list[str],
    bundesliga_data_dirs: list[Path],
    cache_dir: Path,
    *,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    cache_hits = 0
    cache_writes = 0
    for match_id in sorted(set(str(match_id) for match_id in match_ids)):
        cache_path = cache_dir / f"{match_id}.csv"
        if cache_path.exists() and not refresh_cache:
            frames.append(pd.read_csv(cache_path))
            cache_hits += 1
            continue

        minutes = derive_match_minutes_played(match_id, bundesliga_data_dirs)
        minutes.to_csv(cache_path, index=False)
        frames.append(minutes)
        cache_writes += 1

    if not frames:
        return pd.DataFrame(columns=["match_id", "player_id", "minutes_played"]), cache_hits, cache_writes
    minutes_played = pd.concat(frames, ignore_index=True)
    minutes_played["match_id"] = normalize_identifier(minutes_played["match_id"])
    minutes_played["player_id"] = normalize_identifier(minutes_played["player_id"])
    minutes_played["minutes_played"] = coerce_nullable_float(minutes_played["minutes_played"])
    return minutes_played, cache_hits, cache_writes


def add_minutes_played_to_actions(
    bundesliga_actions: pd.DataFrame,
    minutes_played: pd.DataFrame,
) -> pd.DataFrame:
    enriched = bundesliga_actions.copy()
    enriched["match_id"] = normalize_identifier(enriched["stats_perform_match_id"])
    if minutes_played.empty:
        enriched["minutes_played"] = pd.NA
        return enriched

    lookup = minutes_played[["match_id", "player_id", "minutes_played"]].drop_duplicates(
        subset=["match_id", "player_id"],
        keep="first",
    )
    enriched["player_id"] = normalize_identifier(enriched["player_id"])
    return enriched.merge(lookup, on=["match_id", "player_id"], how="left")


def add_per90_columns(df: pd.DataFrame) -> pd.DataFrame:
    with_per90 = df.copy()
    denominator = pd.to_numeric(with_per90["minutes_played"], errors="coerce")
    valid_denominator = denominator.notna() & denominator.ne(0)
    for metric in ACTION_METRIC_COLUMNS:
        per90_column = f"{metric}_per90"
        with_per90[per90_column] = pd.NA
        with_per90.loc[valid_denominator, per90_column] = (
            with_per90.loc[valid_denominator, f"{metric}_sum"] * (90 / denominator.loc[valid_denominator])
        )
    return with_per90


def aggregate_bundesliga_matches(bundesliga_actions: pd.DataFrame) -> pd.DataFrame:
    if bundesliga_actions.empty:
        return pd.DataFrame(columns=MATCH_SUMMARY_COLUMNS)

    grouped = bundesliga_actions.groupby(["player_id", "match_id"], dropna=False, as_index=False)
    match_rows = grouped.agg(
        player_name=("player_name", first_non_null),
        advanced_position=("advanced_position", select_dominant_value),
        minutes_played=("minutes_played", first_non_null),
        **{f"{column}_sum": (column, "sum") for column in ACTION_METRIC_COLUMNS},
    )
    match_rows = add_per90_columns(match_rows)
    match_rows = match_rows.sort_values(["player_id", "match_id"]).reset_index(drop=True)
    return match_rows[MATCH_SUMMARY_COLUMNS].copy()


def aggregate_bundesliga_players(
    bundesliga_matches: pd.DataFrame,
    bundesliga_actions: pd.DataFrame,
) -> pd.DataFrame:
    if bundesliga_actions.empty:
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)

    match_aggregates = (
        bundesliga_matches.groupby("player_id", dropna=False, as_index=False)
        .agg(
            minutes_played=("minutes_played", "sum"),
            **{column: (column, "sum") for column in METRIC_SUM_COLUMNS},
            **{column: (column, "mean") for column in METRIC_PER90_COLUMNS},
        )
    )
    action_aggregates = (
        bundesliga_actions.groupby("player_id", dropna=False, as_index=False)
        .agg(
            player_name=("player_name", first_non_null),
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
    aggregated = action_aggregates.merge(match_aggregates, on="player_id", how="left")
    aggregated = aggregated.sort_values("player_id").reset_index(drop=True)
    return aggregated[PLAYER_SUMMARY_COLUMNS].copy()


def aggregate_scores(
    bundesliga_actions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if bundesliga_actions.empty:
        return pd.DataFrame(columns=POSITION_SUMMARY_COLUMNS)

    aggregated = (
        bundesliga_actions.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            player_name=("player_name", first_non_null),
            actions=("dm_score", "size"),
            **{f"{column}_sum": (column, "sum") for column in ACTION_METRIC_COLUMNS},
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
    ordered_columns = group_columns + [
        "player_name",
        "actions",
        *METRIC_SUM_COLUMNS,
        *PLAYER_AVG_MEDIAN_COLUMNS,
    ]
    return aggregated[ordered_columns]


def add_overallrating(
    aggregate: pd.DataFrame,
    fc25_ratings_ids: pd.DataFrame,
) -> pd.DataFrame:
    ratings = fc25_ratings_ids[["player_id", "team_name", "overallrating"]].copy()
    return aggregate.merge(ratings, on="player_id", how="left")


def print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundesliga_data_dirs = (
        [Path(path) for path in args.bundesliga_data_dirs]
        if args.bundesliga_data_dirs
        else DEFAULT_BUNDESLIGA_DATA_DIRS
    )

    bundesliga_ids = read_bundesliga_ids(args.lineup_path, args.season_prefix)
    fc25_ratings = read_fc25_ratings(args.fc25_ratings_path)
    fc25_ratings_ids = add_fc25_matches(bundesliga_ids, fc25_ratings)

    component_run_root = resolve_component_run_root(args)
    match_dirs = discover_match_dirs(component_run_root, args.event_synced_dir, args.season_prefix)
    model_data = build_model_data(match_dirs)
    bundesliga_actions = add_scores_to_events(model_data, args.event_synced_dir)
    match_ids = sorted(bundesliga_actions["stats_perform_match_id"].dropna().astype(str).unique().tolist())
    minutes_played, minutes_cache_hits, minutes_cache_writes = read_or_build_minutes_played_cache(
        match_ids,
        bundesliga_data_dirs,
        args.minutes_played_cache_dir,
        refresh_cache=args.refresh_minutes_played_cache,
    )
    bundesliga_actions = add_minutes_played_to_actions(bundesliga_actions, minutes_played)

    bundesliga_matches = aggregate_bundesliga_matches(bundesliga_actions)
    bundesliga_players = aggregate_bundesliga_players(bundesliga_matches, bundesliga_actions)
    bundesliga_positions = aggregate_scores(
        bundesliga_actions,
        ["player_id", "advanced_position"],
    )
    bundesliga_players = add_overallrating(bundesliga_players, fc25_ratings_ids)
    bundesliga_positions = add_overallrating(bundesliga_positions, fc25_ratings_ids)

    fc25_ratings_ids_path = args.output_dir / "fc25_ratings_ids.csv"
    bundesliga_actions_path = args.output_dir / "bundesliga_actions.csv"
    bundesliga_matches_path = args.output_dir / MATCHES_FILENAME
    bundesliga_players_path = args.output_dir / "bundesliga_players.csv"
    bundesliga_positions_path = args.output_dir / "bundesliga_positions.csv"

    fc25_ratings_ids.to_csv(fc25_ratings_ids_path, index=False)
    bundesliga_actions.to_csv(bundesliga_actions_path, index=False)
    bundesliga_matches.to_csv(bundesliga_matches_path, index=False)
    bundesliga_players.to_csv(bundesliga_players_path, index=False)
    bundesliga_positions.to_csv(bundesliga_positions_path, index=False)

    summary = {
        "season_prefix": args.season_prefix,
        "component_run_root": component_run_root,
        "component_matches": len(match_dirs),
        "fc25_players": len(fc25_ratings_ids),
        "fc25_matched_players": int(fc25_ratings_ids["overallrating"].notna().sum()),
        "model_rows": len(model_data),
        "bundesliga_actions_rows": len(bundesliga_actions),
        "bundesliga_actions_scored": int(bundesliga_actions["dm_score"].notna().sum()),
        "bundesliga_matches_rows": len(bundesliga_matches),
        "bundesliga_players_rows": len(bundesliga_players),
        "bundesliga_positions_rows": len(bundesliga_positions),
        "minutes_played_cache_dir": args.minutes_played_cache_dir,
        "minutes_played_cache_hits": minutes_cache_hits,
        "minutes_played_cache_writes": minutes_cache_writes,
        "fc25_ratings_ids_path": fc25_ratings_ids_path,
        "bundesliga_actions_path": bundesliga_actions_path,
        "bundesliga_matches_path": bundesliga_matches_path,
        "bundesliga_players_path": bundesliga_players_path,
        "bundesliga_positions_path": bundesliga_positions_path,
    }
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
