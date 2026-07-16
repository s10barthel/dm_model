from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from project_config import PROJECT_ROOT


COACH_RATINGS_PATH = PROJECT_ROOT / "validation" / "coach_ratings" / "output" / "coach_ratings.csv"
SELECTIONS_PATH = PROJECT_ROOT / "validation" / "selections" / "per_action_option_counts.csv"
SELECTION_SETTINGS = ("CAVE", "HMD")


@dataclass(frozen=True)
class OverlayData:
    coach_ratings: pd.DataFrame
    selections: pd.DataFrame
    metadata: dict[str, object]


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalize_text_id(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized == "")


def _normalize_player_id(series: pd.Series, *, label: str) -> pd.Series:
    raw = series.astype("string").str.strip()
    normalized = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & raw.ne("") & normalized.isna()
    if invalid.any():
        values = ", ".join(raw.loc[invalid].head(5).tolist())
        raise ValueError(f"{label} contains non-numeric player ids: {values}")
    return normalized.astype("Int64")


def _normalize_proportions(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    normalized = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & raw.ne("") & normalized.isna()
    if invalid.any():
        values = ", ".join(raw.loc[invalid].head(5).tolist())
        raise ValueError(f"Selection proportions contain non-numeric values: {values}")
    out_of_range = normalized.notna() & ~normalized.between(0.0, 1.0)
    if out_of_range.any():
        values = ", ".join(raw.loc[out_of_range].head(5).tolist())
        raise ValueError(f"Selection proportions must be between 0 and 1: {values}")
    return normalized.astype(float)


def load_coach_ratings(path: str | Path = COACH_RATINGS_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_path = Path(path).expanduser().resolve()
    coach_ratings = pd.read_csv(resolved_path, low_memory=False)
    _validate_columns(coach_ratings, ["id", "uefa_player_id", "Scores", "team"], "Coach ratings")

    normalized = coach_ratings.copy()
    normalized["id"] = _normalize_text_id(normalized["id"])
    normalized["uefa_player_id"] = _normalize_player_id(
        normalized["uefa_player_id"], label="Coach ratings"
    )
    normalized["team"] = _normalize_text_id(normalized["team"])
    normalized["Scores"] = pd.to_numeric(normalized["Scores"], errors="coerce")
    normalized = normalized.dropna(subset=["id", "uefa_player_id", "team", "Scores"]).copy()
    normalized["uefa_player_id"] = normalized["uefa_player_id"].astype(int)
    return normalized, {
        "path": str(resolved_path),
        "rows_read": len(coach_ratings),
        "scored_rows": len(normalized),
    }


def load_selection_proportions(path: str | Path = SELECTIONS_PATH) -> tuple[pd.DataFrame, dict[str, object]]:
    resolved_path = Path(path).expanduser().resolve()
    selections = pd.read_csv(resolved_path, low_memory=False)
    _validate_columns(selections, ["action_id", "setting", "SelectedPlayer", "proportion"], "Selections")

    normalized = selections[["action_id", "setting", "SelectedPlayer", "proportion"]].copy()
    normalized["action_id"] = _normalize_text_id(normalized["action_id"])
    normalized["SelectedPlayer"] = _normalize_player_id(
        normalized["SelectedPlayer"], label="Selections"
    )
    normalized["setting"] = _normalize_text_id(normalized["setting"]).str.upper()
    normalized["proportion"] = _normalize_proportions(normalized["proportion"])
    if normalized[["action_id", "SelectedPlayer", "setting"]].isna().any().any():
        raise ValueError("Selections contain an empty action_id, SelectedPlayer, or setting.")

    unexpected_settings = sorted(set(normalized["setting"].unique()) - set(SELECTION_SETTINGS))
    if unexpected_settings:
        raise ValueError("Selections contain unsupported settings: " + ", ".join(unexpected_settings))
    duplicate_mask = normalized.duplicated(["action_id", "SelectedPlayer", "setting"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = normalized.loc[duplicate_mask, ["action_id", "SelectedPlayer", "setting"]].head(5)
        raise ValueError(
            "Selections contain duplicate (action_id, SelectedPlayer, setting) rows: "
            + duplicate_rows.to_dict("records").__repr__()
        )

    normalized["SelectedPlayer"] = normalized["SelectedPlayer"].astype(int)
    return normalized, {
        "path": str(resolved_path),
        "rows_read": len(selections),
        "action_player_pairs": int(normalized[["action_id", "SelectedPlayer"]].drop_duplicates().shape[0]),
        "actions": int(normalized["action_id"].nunique()),
    }


def load_overlay_data(*, include_coach_ratings: bool, include_selections: bool) -> OverlayData:
    coach_ratings = pd.DataFrame(columns=["id", "uefa_player_id", "team", "Scores"])
    selections = pd.DataFrame(columns=["action_id", "SelectedPlayer", "setting", "proportion"])
    metadata: dict[str, object] = {
        "coach_ratings_enabled": bool(include_coach_ratings),
        "selections_enabled": bool(include_selections),
    }
    if include_coach_ratings:
        coach_ratings, coach_metadata = load_coach_ratings()
        metadata["coach_ratings"] = coach_metadata
    if include_selections:
        selections, selection_metadata = load_selection_proportions()
        metadata["selections"] = selection_metadata
    return OverlayData(coach_ratings=coach_ratings, selections=selections, metadata=metadata)


def filter_coach_rated_situation_ids(
    candidate_situation_ids: list[str], coach_ratings: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Return stable, de-duplicated coach-rated and skipped Hawkeye situation IDs."""
    candidates = list(dict.fromkeys(str(situation_id) for situation_id in candidate_situation_ids))
    scored_ids = set(coach_ratings["id"].dropna().astype(str).tolist())
    eligible_ids = [situation_id for situation_id in candidates if situation_id in scored_ids]
    skipped_ids = [situation_id for situation_id in candidates if situation_id not in scored_ids]
    return eligible_ids, skipped_ids


def format_coach_score(value: float) -> str:
    text = f"{float(value):.2f}"
    return text.rstrip("0").rstrip(".")


def format_selection_label(cave: float | None, hmd: float | None) -> str | None:
    if pd.isna(cave) and pd.isna(hmd):
        return None
    cave_percent = 0.0 if pd.isna(cave) else float(cave) * 100.0
    hmd_percent = 0.0 if pd.isna(hmd) else float(hmd) * 100.0
    return f"{cave_percent:.0f} | {hmd_percent:.0f}"


def add_overlay_annotations(
    ax,
    snapshot: pd.DataFrame,
    attacking_prefix: str,
    *,
    coach_scores: pd.Series | None,
    selection_labels: pd.Series | None,
) -> None:
    coach_scores = pd.Series(dtype=float) if coach_scores is None else coach_scores.dropna()
    selection_labels = pd.Series(dtype="string") if selection_labels is None else selection_labels.dropna()
    object_ids = sorted(set(coach_scores.index.astype(str)).union(selection_labels.index.astype(str)))
    for object_id in object_ids:
        if not object_id.startswith(attacking_prefix):
            continue
        x_column = f"{object_id}_x"
        y_column = f"{object_id}_y"
        if x_column not in snapshot.columns or y_column not in snapshot.columns:
            continue
        x_value = snapshot[x_column].iloc[-1]
        y_value = snapshot[y_column].iloc[-1]
        has_coach_score = object_id in coach_scores.index
        if has_coach_score:
            ax.annotate(
                format_coach_score(float(coach_scores.at[object_id])),
                xy=(x_value, y_value),
                xytext=(0, -3.5),
                textcoords="offset points",
                ha="center",
                va="top",
                color="#1f4e79",
                fontsize=11,
                fontweight="normal",
                zorder=7,
            )
        if object_id in selection_labels.index:
            ax.annotate(
                str(selection_labels.at[object_id]),
                xy=(x_value, y_value),
                xytext=(0, -15 if has_coach_score else -3.5),
                textcoords="offset points",
                ha="center",
                va="top",
                color="#5b3c00",
                fontsize=10,
                fontweight="normal",
                zorder=7,
            )


def _object_ids_for_players(situation_tracking: pd.DataFrame, situation, player_ids: set[int]) -> dict[int, list[str]]:
    if not player_ids:
        return {}
    players = situation_tracking.loc[
        situation_tracking["uefa_player_id"].isin(player_ids), ["team", "uefa_player_id"]
    ].drop_duplicates()
    object_ids: dict[int, list[str]] = {}
    for row in players.itertuples(index=False):
        team = str(row.team)
        if team not in situation.team_map:
            continue
        player_id = int(row.uefa_player_id)
        object_id = f"{situation.team_map[team]}_{player_id}"
        if f"{object_id}_x" not in situation.tracking.columns:
            continue
        object_ids.setdefault(player_id, []).append(object_id)
    return object_ids


def build_situation_overlays(
    overlay_data: OverlayData,
    situation_id: str,
    situation_tracking: pd.DataFrame,
    situation,
) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    situation_id = str(situation_id)
    coach_rows = overlay_data.coach_ratings.loc[
        overlay_data.coach_ratings.get("id", pd.Series(dtype="string")).eq(situation_id)
    ].copy()
    selection_rows = overlay_data.selections.loc[
        overlay_data.selections.get("action_id", pd.Series(dtype="string")).eq(situation_id)
    ].copy()
    player_ids = set(coach_rows.get("uefa_player_id", pd.Series(dtype=int)).dropna().astype(int).tolist())
    player_ids.update(selection_rows.get("SelectedPlayer", pd.Series(dtype=int)).dropna().astype(int).tolist())
    object_ids = _object_ids_for_players(situation_tracking, situation, player_ids)

    coach_scores: dict[str, float] = {}
    for row in coach_rows.itertuples(index=False):
        team = str(row.team)
        if team not in situation.team_map:
            continue
        object_id = f"{situation.team_map[team]}_{int(row.uefa_player_id)}"
        if f"{object_id}_x" in situation.tracking.columns:
            coach_scores.setdefault(object_id, float(row.Scores))

    selection_labels: dict[str, str] = {}
    for player_id, player_rows in selection_rows.groupby("SelectedPlayer", sort=False):
        by_setting = player_rows.set_index("setting")["proportion"]
        label = format_selection_label(by_setting.get("CAVE"), by_setting.get("HMD"))
        if label is None:
            continue
        for object_id in object_ids.get(int(player_id), []):
            selection_labels[object_id] = label

    return (
        pd.Series(coach_scores, dtype=float).sort_index(),
        pd.Series(selection_labels, dtype="string").sort_index(),
        {
            "coach_rows_for_situation": len(coach_rows),
            "coach_annotations": len(coach_scores),
            "selection_rows_for_situation": len(selection_rows),
            "selection_annotations": len(selection_labels),
        },
    )
