from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from datatools import config, utils
from datatools.xt import sort_events
from project_config import EPV_DIR, EPV_MATCH_DIR


EPV_COLUMNS = ["action_id", "epv", "scores_epv", "concedes_epv"]


def _align_component_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if not frames:
        raise ValueError("Expected at least one component frame.")

    index = frames[0].index
    columns = frames[0].columns
    for frame in frames[1:]:
        index = index.union(frame.index)
        columns = columns.union(frame.columns)
    return [frame.reindex(index=index, columns=columns) for frame in frames]


def compute_epv_values(
    pass_intent: pd.DataFrame,
    pass_success: pd.DataFrame,
    outcome_scoring_success: pd.DataFrame,
    outcome_scoring_failure: pd.DataFrame,
    outcome_conceding_success: pd.DataFrame,
    outcome_conceding_failure: pd.DataFrame,
) -> pd.Series:
    aligned = _align_component_frames(
        [
            pass_intent,
            pass_success,
            outcome_scoring_success,
            outcome_scoring_failure,
            outcome_conceding_success,
            outcome_conceding_failure,
        ]
    )
    (
        aligned_pass_intent,
        aligned_pass_success,
        scoring_success,
        scoring_failure,
        conceding_success,
        conceding_failure,
    ) = aligned

    pass_score = aligned_pass_success * (scoring_success - conceding_success) + (1.0 - aligned_pass_success) * (
        scoring_failure - conceding_failure
    )
    weighted_pass_score = aligned_pass_intent * pass_score
    if len(weighted_pass_score.columns) == 0:
        return pd.Series(np.nan, index=weighted_pass_score.index, name="epv", dtype=float)
    epv = weighted_pass_score.sum(axis=1, min_count=1)
    epv.name = "epv"
    return epv.astype(float)


def build_epv_action_values(actions: pd.DataFrame, epv_values: pd.Series) -> pd.DataFrame:
    if "action_id" not in actions.columns:
        raise ValueError("Cannot build EPV action values because actions are missing action_id.")
    missing_action_indexes = epv_values.index.difference(actions.index)
    if len(missing_action_indexes) > 0:
        sample = missing_action_indexes[:5].tolist()
        raise ValueError(f"EPV predictions reference action indexes missing from actions: {sample}")

    return pd.DataFrame(
        {
            "action_id": actions.loc[epv_values.index, "action_id"].to_numpy(),
            "epv": epv_values.to_numpy(dtype=float),
        }
    )


def annotate_match_epv(events: pd.DataFrame, epv_action_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_events = sort_events(utils.sanitize_expected_goal(events))
    sorted_events["epv"] = np.nan

    if not epv_action_values.empty:
        missing = [column for column in ["action_id", "epv"] if column not in epv_action_values.columns]
        if missing:
            raise ValueError(f"EPV action values are missing required columns: {missing}")
        sorted_events = sorted_events.merge(
            epv_action_values[["action_id", "epv"]],
            on="action_id",
            how="left",
            suffixes=("", "_pred"),
        )
        sorted_events["epv"] = sorted_events["epv_pred"].combine_first(sorted_events["epv"])
        sorted_events = sorted_events.drop(columns=["epv_pred"])

    expected_goal_source = (
        sorted_events["expected_goal"]
        if "expected_goal" in sorted_events.columns
        else pd.Series(0.0, index=sorted_events.index)
    )
    expected_goal = pd.to_numeric(expected_goal_source, errors="coerce").fillna(0.0)
    shot_mask = sorted_events["spadl_type"].eq("shot")
    sorted_events.loc[shot_mask, "epv"] = np.fmax(
        pd.to_numeric(sorted_events.loc[shot_mask, "epv"], errors="coerce").to_numpy(dtype=float),
        expected_goal.loc[shot_mask].to_numpy(dtype=float),
    )

    sorted_events = utils.label_epv_returns(
        sorted_events,
        lookahead_len=5,
        eligible_types=tuple(config.XT_ACTION_TYPES),
    )

    user_export = sorted_events.loc[sorted_events["spadl_type"].isin(config.XT_ACTION_TYPES)].copy()
    export_cols = [
        col
        for col in [
            "game_id",
            "stats_perform_match_id",
            "action_id",
            "original_event_id",
            "period_id",
            "seconds",
            "team_id",
            "player_id",
            "object_id",
            "spadl_type",
            "success",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "epv",
            "scores_epv",
            "concedes_epv",
        ]
        if col in user_export.columns
    ]
    return sorted_events, user_export[export_cols].copy()


def merge_epv_annotations(
    events: pd.DataFrame,
    match_id: str | None,
    epv_match_dir: str | Path = EPV_MATCH_DIR,
) -> pd.DataFrame:
    events = events.copy()
    if match_id is None:
        return events

    sidecar_path = Path(epv_match_dir) / f"{match_id}.csv"
    if not sidecar_path.exists():
        return events

    sidecar = pd.read_csv(sidecar_path, usecols=lambda c: c in EPV_COLUMNS)
    if sidecar.empty:
        return events

    events = events.drop(columns=[c for c in ["epv", "scores_epv", "concedes_epv"] if c in events.columns])
    return events.merge(sidecar, on="action_id", how="left")


def save_epv_outputs(
    all_events: Iterable[pd.DataFrame],
    epv_values_by_match: dict[str, pd.DataFrame],
    metadata: dict,
    output_dir: str | Path = EPV_DIR,
    epv_match_dir: str | Path = EPV_MATCH_DIR,
) -> None:
    output_dir = Path(output_dir)
    epv_match_dir = Path(epv_match_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epv_match_dir.mkdir(parents=True, exist_ok=True)

    exported_actions: list[pd.DataFrame] = []
    for events in all_events:
        if events.empty:
            continue
        match_id = str(
            events["stats_perform_match_id"].iloc[0]
            if "stats_perform_match_id" in events.columns
            else events["game_id"].iloc[0]
        )
        annotated_events, exported_epv = annotate_match_epv(events, epv_values_by_match.get(match_id, pd.DataFrame()))
        sidecar = annotated_events[EPV_COLUMNS].copy()
        sidecar.to_csv(epv_match_dir / f"{match_id}.csv", index=False)
        exported_actions.append(exported_epv)

    if exported_actions:
        pd.concat(exported_actions, ignore_index=True).to_csv(output_dir / "epv.csv", index=False)
    else:
        pd.DataFrame(columns=EPV_COLUMNS).to_csv(output_dir / "epv.csv", index=False)

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
