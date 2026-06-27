from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from kloppy import skillcorner as kloppy_skillcorner
from torch_geometric.data import Data

from datatools import config, utils
import datatools.preprocess as proc
from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
from inference import PhysicalXPassNoUsableRowsError, inference_gnn

REQUIRED_EVENT_COLUMNS = [
    "match_id",
    "index",
    "event_type",
    "frame_start",
    "frame_end",
    "period",
    "start_type_id",
    "attacking_side",
    "player_id",
]
PLAYER_COL_RE = re.compile(r"^(?P<player_id>\d+)_(?P<feature>x|y)$")
COMPONENT_COLUMNS = [
    "action_intent",
    "pass_intent",
    "pass_success",
    "pass_height",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]
METADATA_COLUMNS = ["match_id", "frame", "index", "period", "player_id", "attacking_side"]
SKILLCORNER_FRAME_MODES = {"all", "first_and_last"}


@dataclass
class SkillcornerPossession:
    match_id: str
    event_index: int
    fps: float
    tracking: pd.DataFrame
    phases: pd.DataFrame
    frame_meta: pd.DataFrame
    actions: pd.DataFrame
    labels: torch.Tensor
    graph_features_0: list[Data]
    graph_features_by_dir: dict[str, object] = field(default_factory=dict)
    include_keepers: bool = True
    include_goals: bool = True
    max_players: int = 24
    graph_features_1: list[Data] | None = None
    lineup: pd.DataFrame = field(default_factory=pd.DataFrame)
    tabular_features_0: None = None
    tabular_features_1: None = None


def _match_paths(input_dir: str | Path, match_id: str) -> dict[str, Path]:
    root = Path(input_dir)
    return {
        "tracking": root / f"{match_id}_tracking.jsonl",
        "meta": root / f"{match_id}_match.json",
        "events": root / f"{match_id}_dynamic_events.csv",
    }


def discover_skillcorner_matches(
    input_dir: str | Path,
    requested_match_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    root = Path(input_dir)
    requested = [str(match_id) for match_id in requested_match_ids] if requested_match_ids else None
    skipped: dict[str, list[str]] = {"missing_tracking": [], "missing_meta": [], "missing_events": []}

    if requested:
        candidate_ids = requested
    else:
        candidate_ids = sorted(path.stem.replace("_tracking", "") for path in root.glob("*_tracking.jsonl"))

    valid_ids: list[str] = []
    for match_id in candidate_ids:
        paths = _match_paths(root, match_id)
        if not paths["tracking"].exists():
            skipped["missing_tracking"].append(match_id)
            continue
        if not paths["meta"].exists():
            skipped["missing_meta"].append(match_id)
            continue
        if not paths["events"].exists():
            skipped["missing_events"].append(match_id)
            continue
        valid_ids.append(match_id)

    if limit is not None:
        valid_ids = valid_ids[:limit]
    if not valid_ids:
        raise ValueError("No complete SkillCorner match trios were selected.")

    return valid_ids, skipped


def load_skillcorner_match_data(match_id: str, input_dir: str | Path) -> dict[str, Any]:
    return json.loads(_match_paths(input_dir, match_id)["meta"].read_text(encoding="utf-8"))


def build_skillcorner_player_meta(match_data: dict[str, Any]) -> pd.DataFrame:
    home_team_id = int(match_data["home_team"]["id"])
    away_team_id = int(match_data["away_team"]["id"])
    rows: list[dict[str, Any]] = []

    for player in match_data.get("players", []):
        player_role = player.get("player_role") or {}
        acronym = str(player_role.get("acronym", "")).upper()
        name = str(player_role.get("name", "")).lower()
        rows.append(
            {
                "player_id": int(player["id"]),
                "team_id": int(player["team_id"]),
                "actual_prefix": "home" if int(player["team_id"]) == home_team_id else "away",
                "is_goalkeeper": acronym == "GK" or "goalkeeper" in name,
            }
        )

    player_meta = pd.DataFrame(rows).drop_duplicates(subset=["player_id"]).sort_values("player_id").reset_index(drop=True)
    if player_meta.empty:
        raise ValueError(f"Match {match_data.get('id')} does not contain player metadata.")
    return player_meta


def load_skillcorner_events(match_id: str, input_dir: str | Path) -> pd.DataFrame:
    events = pd.read_csv(_match_paths(input_dir, match_id)["events"], low_memory=False)
    missing = [column for column in REQUIRED_EVENT_COLUMNS if column not in events.columns]
    if missing:
        raise KeyError(f"SkillCorner event file for {match_id} is missing required columns: {missing}")

    events = events[events["event_type"] == "player_possession"].copy()
    if events.empty:
        return events

    numeric_cols = ["match_id", "index", "frame_start", "frame_end", "period", "player_id", "start_type_id"]
    for column in numeric_cols:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    events = events.dropna(subset=numeric_cols + ["attacking_side"]).copy()
    if events.empty:
        return events.reset_index(drop=True)

    events = events.loc[events["start_type_id"].eq(1)].copy()
    if events.empty:
        return events.reset_index(drop=True)

    events[numeric_cols] = events[numeric_cols].astype(int)
    events["attacking_side"] = events["attacking_side"].astype(str)
    events = events.sort_values(["frame_start", "frame_end", "index"]).reset_index(drop=True)

    previous_end: int | None = None
    previous_index: int | None = None
    for row in events.itertuples(index=False):
        if previous_end is not None and int(row.frame_start) <= previous_end:
            raise ValueError(
                f"Overlapping player_possession intervals in match {match_id}: "
                f"event {previous_index} overlaps event {row.index}."
            )
        previous_end = int(row.frame_end)
        previous_index = int(row.index)

    return events


def load_skillcorner_tracking(match_id: str, input_dir: str | Path) -> tuple[pd.DataFrame, float]:
    paths = _match_paths(input_dir, match_id)
    dataset = kloppy_skillcorner.load(
        meta_data=paths["meta"],
        raw_data=paths["tracking"],
        coordinates="skillcorner",
        include_empty_frames=True,
    )
    tracking = dataset.to_df().copy()
    if tracking.empty:
        raise ValueError(f"Kloppy returned an empty tracking table for match {match_id}.")

    if pd.api.types.is_timedelta64_dtype(tracking["timestamp"]):
        tracking["timestamp"] = tracking["timestamp"].dt.total_seconds()
    else:
        tracking["timestamp"] = pd.to_numeric(tracking["timestamp"], errors="coerce")

    tracking["frame_id"] = pd.to_numeric(tracking["frame_id"], errors="coerce").astype("Int64")
    tracking["period_id"] = pd.to_numeric(tracking["period_id"], errors="coerce").astype("Int64")
    tracking["ball_x"] = pd.to_numeric(tracking["ball_x"], errors="coerce")
    tracking["ball_y"] = pd.to_numeric(tracking["ball_y"], errors="coerce")
    tracking["ball_z"] = pd.to_numeric(tracking["ball_z"], errors="coerce")
    tracking["ball_state"] = tracking["ball_state"].fillna("alive")
    tracking = tracking.dropna(subset=["frame_id"]).copy()
    tracking["frame_id"] = tracking["frame_id"].astype(int)
    tracking["period_id"] = tracking["period_id"].ffill().bfill().astype(int)
    tracking = tracking.sort_values("frame_id").set_index("frame_id", drop=False)

    fps = float(getattr(dataset.metadata, "frame_rate", 10) or 10)
    return tracking, fps


def build_skillcorner_match_context(match_id: str, input_dir: str | Path) -> dict[str, Any]:
    match_data = load_skillcorner_match_data(match_id, input_dir)
    player_meta = build_skillcorner_player_meta(match_data)
    tracking, fps = load_skillcorner_tracking(match_id, input_dir)
    events = load_skillcorner_events(match_id, input_dir)
    return {
        "match_id": str(match_id),
        "match_data": match_data,
        "player_meta": player_meta,
        "tracking": tracking,
        "events": events,
        "fps": fps,
    }


def _field_x_from_centered(x: float) -> float:
    return float(x) + config.FIELD_SIZE[0] / 2


def _field_y_from_centered(y: float) -> float:
    return float(y) + config.FIELD_SIZE[1] / 2


def _determine_context_start(
    full_tracking: pd.DataFrame,
    period_id: int,
    frame_start: int,
) -> int:
    period_frames = full_tracking.loc[full_tracking["period_id"] == period_id, "frame_id"]
    if period_frames.empty:
        return frame_start
    period_start = int(period_frames.min())
    candidate = frame_start - 1
    return candidate if candidate >= period_start else frame_start


def _build_normalized_tracking_subset(
    full_tracking: pd.DataFrame,
    player_meta: pd.DataFrame,
    event_row: pd.Series,
) -> pd.DataFrame:
    frame_start = int(event_row["frame_start"])
    frame_end = int(event_row["frame_end"])
    period_id = int(event_row["period"])
    possession_player_id = int(event_row["player_id"])
    attacking_side = str(event_row["attacking_side"])

    possession_team_row = player_meta.loc[player_meta["player_id"] == possession_player_id]
    if possession_team_row.empty:
        raise ValueError(f"Possession player {possession_player_id} is missing from the SkillCorner match metadata.")
    possession_team_id = int(possession_team_row["team_id"].iloc[0])

    context_start = _determine_context_start(full_tracking, period_id, frame_start)
    subset = full_tracking.loc[(full_tracking["frame_id"] >= context_start) & (full_tracking["frame_id"] <= frame_end)].copy()
    subset = subset.loc[subset["period_id"] == period_id].copy()
    if subset.empty:
        raise ValueError(
            f"Could not find any tracking frames for match {event_row['match_id']} possession {event_row['index']}."
        )

    rotate = attacking_side == "right_to_left"
    rows: list[dict[str, Any]] = []

    for _, raw_row in subset.iterrows():
        normalized_row: dict[str, Any] = {
            "frame_id": int(raw_row["frame_id"]),
            "period_id": int(raw_row["period_id"]),
            "timestamp": float(raw_row["timestamp"]) if not pd.isna(raw_row["timestamp"]) else np.nan,
            "ball_state": raw_row["ball_state"] if pd.notna(raw_row["ball_state"]) else "alive",
            "ball_owning_home_away": "home",
        }

        ball_x = raw_row.get("ball_x")
        ball_y = raw_row.get("ball_y")
        if pd.notna(ball_x) and pd.notna(ball_y):
            bx = -float(ball_x) if rotate else float(ball_x)
            by = -float(ball_y) if rotate else float(ball_y)
            normalized_row["ball_x"] = _field_x_from_centered(bx)
            normalized_row["ball_y"] = _field_y_from_centered(by)
        else:
            normalized_row["ball_x"] = np.nan
            normalized_row["ball_y"] = np.nan
        normalized_row["ball_z"] = float(raw_row["ball_z"]) if pd.notna(raw_row["ball_z"]) else np.nan

        for player in player_meta.itertuples(index=False):
            player_id = int(player.player_id)
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            if x_col not in raw_row or y_col not in raw_row:
                continue

            px = raw_row[x_col]
            py = raw_row[y_col]
            if pd.isna(px) or pd.isna(py):
                continue

            internal_prefix = "home" if int(player.team_id) == possession_team_id else "away"
            pos_x = -float(px) if rotate else float(px)
            pos_y = -float(py) if rotate else float(py)
            normalized_row[f"{internal_prefix}_{player_id}_x"] = _field_x_from_centered(pos_x)
            normalized_row[f"{internal_prefix}_{player_id}_y"] = _field_y_from_centered(pos_y)

        rows.append(normalized_row)

    tracking = pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)
    tracking = proc.label_frames_and_episodes(tracking, fps=int(round(event_row["_fps"])))
    tracking = proc.calc_physical_features(tracking, fps=int(round(event_row["_fps"])))

    objects = sorted(
        {
            column[:-2]
            for column in tracking.columns
            if column.startswith(("home_", "away_")) and column.endswith("_x")
        }
    )
    for object_id in objects:
        x_col = f"{object_id}_x"
        y_col = f"{object_id}_y"
        feature_cols = [f"{object_id}_{suffix}" for suffix in ["vx", "vy", "speed", "accel"] if f"{object_id}_{suffix}" in tracking.columns]
        if not feature_cols:
            continue

        present_mask = tracking[x_col].notna() & tracking[y_col].notna()
        tracking.loc[~present_mask, feature_cols] = np.nan
        tracking.loc[present_mask, feature_cols] = tracking.loc[present_mask, feature_cols].fillna(0.0)

    if "ball_z" in tracking.columns:
        tracking["ball_z"] = tracking["ball_z"].fillna(0.0)
    return tracking


def _keeper_object_ids(player_meta: pd.DataFrame, possession_player_id: int) -> list[str]:
    possession_team_id = int(player_meta.loc[player_meta["player_id"] == possession_player_id, "team_id"].iloc[0])
    keepers = player_meta.loc[player_meta["is_goalkeeper"]].copy()
    object_ids = []
    for row in keepers.itertuples(index=False):
        prefix = "home" if int(row.team_id) == possession_team_id else "away"
        object_ids.append(f"{prefix}_{int(row.player_id)}")
    return object_ids


def _add_goal_nodes(tracking: pd.DataFrame) -> pd.DataFrame:
    tracking = tracking.copy()
    features = ["x", "y", "vx", "vy", "speed", "accel"]
    goal_cols = [f"{team}_goal_{feature}" for team in ["home", "away"] for feature in features]
    tracking[goal_cols] = 0.0
    tracking["home_goal_x"] = config.FIELD_SIZE[0]
    tracking["home_goal_y"] = config.FIELD_SIZE[1] / 2
    tracking["away_goal_x"] = 0.0
    tracking["away_goal_y"] = config.FIELD_SIZE[1] / 2
    return tracking


def _build_frame_meta(event_row: pd.Series, tracking: pd.DataFrame) -> pd.DataFrame:
    frame_start = int(event_row["frame_start"])
    frame_end = int(event_row["frame_end"])
    possession_player_id = int(event_row["player_id"])
    columns = [
        "frame_id",
        "match_id",
        "index",
        "period",
        "player_id",
        "attacking_side",
        "possessor_object_id",
        "has_ball",
    ]
    rows: list[dict[str, Any]] = []

    for frame_id in range(frame_start, frame_end + 1):
        if frame_id not in tracking.index:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "match_id": str(event_row["match_id"]),
                "index": int(event_row["index"]),
                "period": int(event_row["period"]),
                "player_id": possession_player_id,
                "attacking_side": str(event_row["attacking_side"]),
                "possessor_object_id": f"home_{possession_player_id}",
                "has_ball": not pd.isna(tracking.at[frame_id, "ball_x"]) and not pd.isna(tracking.at[frame_id, "ball_y"]),
            }
        )

    frame_meta = pd.DataFrame(rows, columns=columns).set_index("frame_id", drop=True).sort_index()
    frame_meta.index.name = "frame_id"
    return frame_meta


def _valid_possessor_snapshot(tracking: pd.DataFrame, frame_id: int, possessor_object_id: str) -> bool:
    if frame_id not in tracking.index:
        return False
    x_col = f"{possessor_object_id}_x"
    y_col = f"{possessor_object_id}_y"
    if x_col not in tracking.columns or y_col not in tracking.columns:
        return False
    return not pd.isna(tracking.at[frame_id, x_col]) and not pd.isna(tracking.at[frame_id, y_col])


def _initial_skillcorner_frame_stats(total_frames: int) -> dict[str, int]:
    return {
        "total_frames": int(total_frames),
        "valid_frames": 0,
        "evaluated_frames": 0,
        "selected_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }


def _build_frame_action_label_graph(
    possession: SkillcornerPossession,
    frame_id: int,
    frame_row: pd.Series,
    period_tracking: pd.DataFrame,
    feature_dim: int,
    add_v_edge_features: bool,
) -> tuple[dict[str, Any], list[float], Data, None] | tuple[None, None, None, str]:
    possessor_object_id = str(frame_row["possessor_object_id"])

    if not bool(frame_row["has_ball"]):
        return None, None, None, "skipped_missing_ball"
    if not _valid_possessor_snapshot(possession.tracking, frame_id, possessor_object_id):
        return None, None, None, "skipped_missing_possessor"

    graph = construct_graph_for_frame(
        possession,
        frame_id,
        possessor_object_id,
        period_tracking,
        feature_dim,
        extend=True,
        rotate_to_ltr=False,
        add_v_edge_features=add_v_edge_features,
    )
    if graph is None:
        return None, None, None, "skipped_missing_graph"

    attacking_players, defending_players = utils.find_active_players(
        possession.tracking,
        frame_id,
        team="home",
        include_goals=True,
    )
    if possessor_object_id not in attacking_players:
        return None, None, None, "skipped_missing_possessor"

    intent_index = attacking_players.index(possessor_object_id)
    receiver_index = intent_index
    start_x = float(possession.tracking.at[frame_id, f"{possessor_object_id}_x"])
    start_y = float(possession.tracking.at[frame_id, f"{possessor_object_id}_y"])

    action = {
        "frame_id": frame_id,
        "object_id": possessor_object_id,
        "player_id": int(frame_row["player_id"]),
        "period_id": int(frame_row["period"]),
        "spadl_type": "synthetic_frame",
        "action_type": "pass",
        "receiver_id": possessor_object_id,
        "next_player_id": possessor_object_id,
        "receive_frame_id": frame_id,
        "next_type": "synthetic_frame",
        "start_x": start_x,
        "start_y": start_y,
        "end_x": start_x,
        "end_y": start_y,
        "success": True,
        "blocked": False,
    }
    label = [
        frame_id,
        1,
        0,
        0,
        len(attacking_players) + len(defending_players),
        intent_index,
        receiver_index,
        0.0,
        start_x,
        start_y,
        start_x,
        start_y,
        start_x,
        start_y,
        1,
        0,
        1,
        *([0.0] * (len(config.LABEL_COLUMNS) - 17)),
    ]
    return action, label, graph, None


def _append_skillcorner_frame_result(
    result: tuple[dict[str, Any], list[float], Data],
    actions: list[dict[str, Any]],
    labels: list[list[float]],
    graphs: list[Data],
    stats: dict[str, int],
) -> None:
    action, label, graph = result
    actions.append(action)
    labels.append(label)
    graphs.append(graph)
    stats["valid_frames"] += 1
    stats["selected_frames"] += 1


def _evaluate_skillcorner_frame(
    possession: SkillcornerPossession,
    frame_id: int,
    frame_row: pd.Series,
    period_tracking: pd.DataFrame,
    feature_dim: int,
    add_v_edge_features: bool,
    stats: dict[str, int],
) -> tuple[dict[str, Any], list[float], Data] | None:
    stats["evaluated_frames"] += 1
    action, label, graph, skip_key = _build_frame_action_label_graph(
        possession,
        frame_id,
        frame_row,
        period_tracking,
        feature_dim,
        add_v_edge_features,
    )
    if skip_key is not None:
        stats[skip_key] += 1
        return None
    return action, label, graph


def _build_actions_and_labels(
    possession: SkillcornerPossession,
    add_v_edge_features: bool = False,
    frames_mode: str = "first_and_last",
) -> tuple[pd.DataFrame, torch.Tensor, list[Data], dict[str, int]]:
    actions: list[dict[str, Any]] = []
    labels: list[list[float]] = []
    graphs: list[Data] = []
    feature_dim = infer_node_feature_dim(extend=True)
    period_id = int(possession.frame_meta["period"].iloc[0])
    period_tracking = possession.tracking[possession.tracking["period_id"] == period_id]
    stats = _initial_skillcorner_frame_stats(len(possession.frame_meta))

    if frames_mode not in SKILLCORNER_FRAME_MODES:
        raise ValueError(f"Unsupported SkillCorner frames_mode: {frames_mode}")

    frame_items = [(int(frame_id), frame_row) for frame_id, frame_row in possession.frame_meta.iterrows()]
    if frames_mode == "all":
        for frame_id, frame_row in frame_items:
            result = _evaluate_skillcorner_frame(
                possession,
                frame_id,
                frame_row,
                period_tracking,
                feature_dim,
                add_v_edge_features,
                stats,
            )
            if result is not None:
                _append_skillcorner_frame_result(result, actions, labels, graphs, stats)
    else:
        first_frame_id: int | None = None
        first_result: tuple[dict[str, Any], list[float], Data] | None = None
        for frame_id, frame_row in frame_items:
            first_result = _evaluate_skillcorner_frame(
                possession,
                frame_id,
                frame_row,
                period_tracking,
                feature_dim,
                add_v_edge_features,
                stats,
            )
            if first_result is not None:
                first_frame_id = frame_id
                _append_skillcorner_frame_result(first_result, actions, labels, graphs, stats)
                break

        if first_frame_id is not None and first_result is not None:
            for frame_id, frame_row in reversed(frame_items):
                if frame_id == first_frame_id:
                    break
                result = _evaluate_skillcorner_frame(
                    possession,
                    frame_id,
                    frame_row,
                    period_tracking,
                    feature_dim,
                    add_v_edge_features,
                    stats,
                )
                if result is not None:
                    _append_skillcorner_frame_result(result, actions, labels, graphs, stats)
                    break

    actions_df = (
        pd.DataFrame(actions).set_index("frame_id", drop=False)
        if actions
        else pd.DataFrame(columns=["frame_id"]).set_index("frame_id", drop=False)
    )
    label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, len(config.LABEL_COLUMNS)))
    return actions_df, label_tensor, graphs, stats


def build_skillcorner_possession(
    context: dict[str, Any],
    event_index: int,
    add_v_edge_features: bool = False,
    frames_mode: str = "first_and_last",
) -> tuple[SkillcornerPossession, dict[str, int]]:
    events = context["events"]
    event_rows = events.loc[events["index"] == int(event_index)]
    if event_rows.empty:
        raise KeyError(f"SkillCorner player_possession {event_index} was not found for match {context['match_id']}.")
    event_row = event_rows.iloc[0].copy()
    event_row["_fps"] = context["fps"]

    tracking = _build_normalized_tracking_subset(context["tracking"], context["player_meta"], event_row)
    keepers = _keeper_object_ids(context["player_meta"], int(event_row["player_id"]))
    phases = proc.summarize_phases(tracking, keepers)
    tracking = tracking.copy()
    tracking["phase_id"] = 0
    for phase_id in phases.index:
        start_frame = int(phases.at[phase_id, "start_frame"])
        end_frame = int(phases.at[phase_id, "end_frame"])
        tracking.loc[start_frame:end_frame, "phase_id"] = int(phase_id)
    tracking = _add_goal_nodes(tracking)

    frame_meta = _build_frame_meta(event_row, tracking)
    possession = SkillcornerPossession(
        match_id=str(context["match_id"]),
        event_index=int(event_row["index"]),
        fps=float(context["fps"]),
        tracking=tracking,
        phases=phases,
        frame_meta=frame_meta,
        actions=pd.DataFrame(),
        labels=torch.empty((0, len(config.LABEL_COLUMNS))),
        graph_features_0=[],
    )
    if frame_meta.empty:
        empty_stats = _initial_skillcorner_frame_stats(0)
        return possession, empty_stats

    actions, labels, graphs, stats = _build_actions_and_labels(
        possession,
        add_v_edge_features=add_v_edge_features,
        frames_mode=frames_mode,
    )
    possession.actions = actions
    possession.labels = labels
    possession.graph_features_0 = graphs
    possession.graph_features_by_dir["action_graphs"] = graphs
    return possession, stats


def infer_skillcorner_components(
    possession: SkillcornerPossession,
    model_specs: dict[str, Any],
    device: str = "cpu",
) -> dict[str, pd.DataFrame]:
    components: dict[str, pd.DataFrame] = {}
    if possession.labels.numel() == 0 or not possession.graph_features_0:
        return components

    action_intent, _ = inference_gnn(possession, model_specs["action_intent"], device=device, post_action=False)
    pass_intent, _ = inference_gnn(possession, model_specs["pass_intent"], device=device, post_action=False)
    try:
        pass_success, _ = inference_gnn(possession, model_specs["pass_success"], device=device, post_action=False)
    except PhysicalXPassNoUsableRowsError:
        pass_success = pd.DataFrame()
    if "pass_height" in model_specs:
        pass_height, _ = inference_gnn(possession, model_specs["pass_height"], device=device, post_action=False)
    else:
        pass_height = pd.DataFrame()
    scoring_failure, scoring_success = inference_gnn(
        possession,
        model_specs["outcome_scoring"],
        device=device,
        post_action=False,
    )
    conceding_failure, conceding_success = inference_gnn(
        possession,
        model_specs["outcome_conceding"],
        device=device,
        post_action=False,
    )

    components["action_intent"] = action_intent
    components["pass_intent"] = pass_intent
    components["pass_success"] = pass_success
    components["pass_height"] = pass_height
    components["outcome_scoring_success"] = scoring_success
    components["outcome_scoring_failure"] = scoring_failure
    components["outcome_conceding_success"] = conceding_success
    components["outcome_conceding_failure"] = conceding_failure
    return components


def _expected_option_columns(player_meta: pd.DataFrame, include_shot: bool) -> list[str]:
    option_columns = [str(player_id) for player_id in sorted(player_meta["player_id"].astype(int).unique().tolist())]
    if include_shot:
        option_columns.append("shot")
    return option_columns


def _map_component_columns(component_frame: pd.DataFrame | None, include_shot: bool) -> pd.DataFrame:
    if component_frame is None or component_frame.empty:
        columns = ["shot"] if include_shot else []
        return pd.DataFrame(columns=columns)

    valid_cols = [column for column in component_frame.columns if column.startswith("home_")]
    mapped = component_frame[valid_cols].copy()
    rename_map = {}
    for column in valid_cols:
        if column == "home_goal":
            rename_map[column] = "shot"
        else:
            rename_map[column] = column.split("_", 1)[1]
    mapped = mapped.rename(columns=rename_map)
    mapped.columns = [str(column) for column in mapped.columns]
    if include_shot and "shot" not in mapped.columns:
        mapped["shot"] = np.nan
    return mapped


def build_skillcorner_component_table(
    possession: SkillcornerPossession,
    component_frame: pd.DataFrame | None,
    player_meta: pd.DataFrame,
    include_shot: bool,
) -> pd.DataFrame:
    option_columns = _expected_option_columns(player_meta, include_shot)
    ordered_columns = METADATA_COLUMNS + option_columns
    metadata = (
        possession.frame_meta.reset_index()[["match_id", "frame_id", "index", "period", "player_id", "attacking_side"]]
        .rename(columns={"frame_id": "frame"})
        .copy()
    )
    if component_frame is None or component_frame.empty:
        return pd.DataFrame(columns=ordered_columns)

    mapped = _map_component_columns(component_frame, include_shot).rename_axis("frame").reset_index()
    metadata = metadata.loc[metadata["frame"].isin(mapped["frame"])].copy()
    table = metadata.merge(mapped, on="frame", how="inner")

    for column in option_columns:
        if column not in table.columns:
            table[column] = np.nan
    return table[ordered_columns].sort_values(["frame", "index"]).reset_index(drop=True)


def summarize_skillcorner_stats(
    stats_by_match: dict[str, dict[str, int]],
    skipped_matches: dict[str, list[str]],
) -> dict[str, Any]:
    totals = {
        "matches": len(stats_by_match),
        "possessions": 0,
        "total_frames": 0,
        "valid_frames": 0,
        "evaluated_frames": 0,
        "selected_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }
    for match_stats in stats_by_match.values():
        totals["possessions"] += int(match_stats.get("possessions", 0))
        for key in [
            "total_frames",
            "valid_frames",
            "evaluated_frames",
            "selected_frames",
            "skipped_missing_ball",
            "skipped_missing_possessor",
            "skipped_missing_graph",
        ]:
            totals[key] += int(match_stats.get(key, 0))

    return {
        "totals": totals,
        "skipped_matches": {key: sorted(value) for key, value in skipped_matches.items() if value},
    }


def load_skillcorner_models(
    action_intent_model_id: str,
    pass_intent_model_id: str,
    pass_success_model_id: str,
    outcome_scoring_model_id: str,
    outcome_conceding_model_id: str,
    device: str,
    pass_height_model_id: str | None = None,
) -> dict[str, Any]:
    from models.utils import load_model

    model_specs = {
        "action_intent": load_model(action_intent_model_id, device),
        "pass_intent": load_model(pass_intent_model_id, device),
        "pass_success": load_model(pass_success_model_id, device),
        "outcome_scoring": load_model(outcome_scoring_model_id, device),
        "outcome_conceding": load_model(outcome_conceding_model_id, device),
    }
    if pass_height_model_id:
        model_specs["pass_height"] = load_model(pass_height_model_id, device)
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    return model_specs


def load_skillcorner_component_tables(component_dir: str | Path, match_id: str) -> dict[str, pd.DataFrame]:
    match_dir = Path(component_dir) / str(match_id)
    tables: dict[str, pd.DataFrame] = {}
    for component in COMPONENT_COLUMNS:
        path = match_dir / f"{component}.parquet"
        if not path.exists():
            if component == "pass_height":
                tables[component] = pd.DataFrame()
                continue
            raise FileNotFoundError(
                f"Component parquet not found at {path}. Run scripts/run_skillcorner.py for match {match_id} first."
            )
        tables[component] = pd.read_parquet(path)
    return tables


def build_visualization_probs(row: pd.Series | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)

    probs: dict[str, float] = {}
    for column, value in row.items():
        if column in METADATA_COLUMNS or column == "shot" or pd.isna(value):
            continue
        probs[f"home_{column}"] = float(value)
    return pd.Series(probs, dtype=float).sort_values(ascending=False)
