from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from datatools import config, utils
import datatools.preprocess as proc
from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
from inference import inference_gnn

FRAME_KEY_COLUMNS = ["game_id", "half", "abs_time"]
SITUATION_FRAME_KEY_COLUMNS = ["id", *FRAME_KEY_COLUMNS]
TRACKING_REQUIRED_COLUMNS = [
    "game_id",
    "half",
    "abs_time",
    "uefa_player_id",
    "role",
    "centroid_x",
    "centroid_y",
    "PlayerID",
    "id",
    "team",
    "possession_team",
]
BALL_REQUIRED_COLUMNS = ["game_id", "half", "abs_time", "ball_x", "ball_y", "ball_z"]
COMPONENT_COLUMNS = [
    "action_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]


@dataclass
class HawkeyeSituation:
    situation_id: str
    tracking: pd.DataFrame
    phases: pd.DataFrame
    frame_meta: pd.DataFrame
    actions: pd.DataFrame
    labels: torch.Tensor
    graph_features_0: list[Data]
    team_map: dict[str, str]
    prefix_to_team: dict[str, str]
    include_keepers: bool = True
    include_goals: bool = True
    max_players: int = 24
    match_id: str | None = None
    graph_features_1: list[Data] | None = None
    graph_features_by_dir: dict[str, object] = field(default_factory=dict)
    lineup: pd.DataFrame = field(default_factory=pd.DataFrame)
    tabular_features_0: None = None
    tabular_features_1: None = None

    def __post_init__(self) -> None:
        if self.match_id is None:
            self.match_id = self.situation_id


def load_hawkeye_tracking(tracking_path: str | Path) -> pd.DataFrame:
    tracking = pd.read_csv(tracking_path, low_memory=False)
    missing = [column for column in TRACKING_REQUIRED_COLUMNS if column not in tracking.columns]
    if missing:
        raise KeyError(f"Hawkeye tracking file is missing required columns: {missing}")
    return tracking


def load_hawkeye_ball(ball_path: str | Path) -> pd.DataFrame:
    ball = pd.read_csv(ball_path, low_memory=False)
    missing = [column for column in BALL_REQUIRED_COLUMNS if column not in ball.columns]
    if missing:
        raise KeyError(f"Hawkeye ball file is missing required columns: {missing}")
    return ball


def clean_hawkeye_tracking(tracking: pd.DataFrame) -> pd.DataFrame:
    cleaned = tracking.copy()
    cleaned = cleaned[cleaned["role"].isin([1, 2])].copy()
    cleaned = cleaned.drop_duplicates().copy()

    cleaned["game_id"] = pd.to_numeric(cleaned["game_id"], errors="coerce").astype("Int64")
    cleaned["half"] = pd.to_numeric(cleaned["half"], errors="coerce").astype("Int64")
    cleaned["abs_time"] = pd.to_numeric(cleaned["abs_time"], errors="coerce")
    cleaned["uefa_player_id"] = pd.to_numeric(cleaned["uefa_player_id"], errors="coerce").astype("Int64")
    cleaned["PlayerID"] = pd.to_numeric(cleaned["PlayerID"], errors="coerce").astype("Int64")
    cleaned["centroid_x"] = pd.to_numeric(cleaned["centroid_x"], errors="coerce")
    cleaned["centroid_y"] = pd.to_numeric(cleaned["centroid_y"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=["game_id", "half", "abs_time", "uefa_player_id", "PlayerID", "id", "team", "possession_team"]
    ).copy()
    cleaned["game_id"] = cleaned["game_id"].astype(int)
    cleaned["half"] = cleaned["half"].astype(int)
    cleaned["uefa_player_id"] = cleaned["uefa_player_id"].astype(int)
    cleaned["PlayerID"] = cleaned["PlayerID"].astype(int)
    cleaned["id"] = cleaned["id"].astype(str)
    cleaned["team"] = cleaned["team"].astype(str)
    cleaned["possession_team"] = cleaned["possession_team"].astype(str)

    return cleaned.sort_values(SITUATION_FRAME_KEY_COLUMNS + ["team", "uefa_player_id"]).reset_index(drop=True)


def clean_hawkeye_ball(ball: pd.DataFrame) -> pd.DataFrame:
    cleaned = ball.copy().drop_duplicates().copy()
    cleaned["game_id"] = pd.to_numeric(cleaned["game_id"], errors="coerce").astype("Int64")
    cleaned["half"] = pd.to_numeric(cleaned["half"], errors="coerce").astype("Int64")
    cleaned["abs_time"] = pd.to_numeric(cleaned["abs_time"], errors="coerce")
    cleaned["ball_x"] = pd.to_numeric(cleaned["ball_x"], errors="coerce")
    cleaned["ball_y"] = pd.to_numeric(cleaned["ball_y"], errors="coerce")
    cleaned["ball_z"] = pd.to_numeric(cleaned["ball_z"], errors="coerce")
    cleaned = cleaned.dropna(subset=["game_id", "half", "abs_time"]).copy()
    cleaned["game_id"] = cleaned["game_id"].astype(int)
    cleaned["half"] = cleaned["half"].astype(int)
    cleaned = cleaned.drop_duplicates(subset=FRAME_KEY_COLUMNS, keep="first").copy()
    return cleaned.sort_values(FRAME_KEY_COLUMNS).reset_index(drop=True)


def resolve_situation_ids(
    tracking: pd.DataFrame,
    requested_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    situation_ids = sorted(tracking["id"].dropna().astype(str).unique().tolist())
    if requested_ids:
        requested = {str(situation_id) for situation_id in requested_ids}
        situation_ids = [situation_id for situation_id in situation_ids if situation_id in requested]
    if limit is not None:
        situation_ids = situation_ids[:limit]
    if not situation_ids:
        raise ValueError("No Hawkeye situations were selected.")
    return situation_ids


def _field_x(x: float) -> float:
    return float(x) + config.FIELD_SIZE[0] / 2


def _field_y(y: float) -> float:
    return float(y) + config.FIELD_SIZE[1] / 2


def _stable_team_map(situation_tracking: pd.DataFrame) -> dict[str, str]:
    teams = sorted(situation_tracking["team"].dropna().astype(str).unique().tolist())
    if len(teams) != 2:
        raise ValueError(
            f"Expected exactly 2 teams in Hawkeye situation {situation_tracking['id'].iloc[0]}, found {teams}."
        )
    return {teams[0]: "home", teams[1]: "away"}


def _object_id(prefix: str, uefa_player_id: int) -> str:
    return f"{prefix}_{int(uefa_player_id)}"


def _build_object_map(situation_tracking: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    mapping = situation_tracking[["team", "uefa_player_id", "role"]].drop_duplicates().copy()
    mapping["prefix"] = mapping["team"].map(team_map)
    mapping["object_id"] = mapping.apply(lambda row: _object_id(row["prefix"], int(row["uefa_player_id"])), axis=1)
    return mapping


def _build_frame_meta(
    situation_tracking: pd.DataFrame,
    ball: pd.DataFrame,
    team_map: dict[str, str],
    object_map: pd.DataFrame,
) -> pd.DataFrame:
    player_lookup = object_map.set_index(["team", "uefa_player_id"])["object_id"]
    frame_rows: list[dict[str, Any]] = []
    start_time = float(situation_tracking["abs_time"].min())
    ball_lookup = ball.set_index(FRAME_KEY_COLUMNS)

    grouped = situation_tracking.groupby(FRAME_KEY_COLUMNS, sort=True)
    for frame_id, ((game_id, half, abs_time), frame_rows_df) in enumerate(grouped):
        frame_info = frame_rows_df.iloc[0]
        possession_team = str(frame_info["possession_team"])
        possessor_id = int(frame_info["PlayerID"])
        possession_prefix = team_map[possession_team]
        possessor_object_id = player_lookup.get((possession_team, possessor_id))

        ball_key = (int(game_id), int(half), float(abs_time))
        has_ball = ball_key in ball_lookup.index
        if has_ball:
            ball_row = ball_lookup.loc[ball_key]
            if isinstance(ball_row, pd.DataFrame):
                ball_row = ball_row.iloc[0]
            ball_x = _field_x(float(ball_row["ball_x"]))
            ball_y = _field_y(float(ball_row["ball_y"]))
            ball_z = float(ball_row["ball_z"]) if not pd.isna(ball_row["ball_z"]) else np.nan
        else:
            ball_x = np.nan
            ball_y = np.nan
            ball_z = np.nan

        frame_rows.append(
            {
                "id": str(frame_info["id"]),
                "frame_id": frame_id,
                "game_id": int(game_id),
                "half": int(half),
                "period_id": int(half),
                "abs_time": float(abs_time),
                "timestamp": float(abs_time) - start_time,
                "possession_team": possession_team,
                "possession_prefix": possession_prefix,
                "PlayerID": possessor_id,
                "possessor_object_id": possessor_object_id,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_z": ball_z,
                "has_ball": bool(has_ball),
            }
        )

    frame_meta = pd.DataFrame(frame_rows).set_index("frame_id").sort_index()
    frame_meta.index.name = "frame_id"
    return frame_meta


def _build_tracking_wide(
    situation_tracking: pd.DataFrame,
    frame_meta: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    grouped = situation_tracking.groupby("abs_time", sort=True)
    for frame_id, frame_row in frame_meta.iterrows():
        frame_players = grouped.get_group(frame_row["abs_time"])
        wide_row: dict[str, Any] = {
            "frame_id": int(frame_id),
            "period_id": int(frame_row["period_id"]),
            "timestamp": float(frame_row["timestamp"]),
            "ball_state": "alive",
            "ball_owning_home_away": frame_row["possession_prefix"],
            "ball_x": frame_row["ball_x"],
            "ball_y": frame_row["ball_y"],
            "ball_z": frame_row["ball_z"],
        }

        for _, player in frame_players.iterrows():
            prefix = team_map[str(player["team"])]
            player_id = int(player["uefa_player_id"])
            object_id = _object_id(prefix, player_id)
            wide_row[f"{object_id}_x"] = _field_x(float(player["centroid_x"]))
            wide_row[f"{object_id}_y"] = _field_y(float(player["centroid_y"]))

        rows.append(wide_row)

    tracking = pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)
    tracking = proc.label_frames_and_episodes(tracking, fps=25)
    tracking = proc.calc_physical_features(tracking, fps=25)
    return tracking


def _assign_phase_ids(tracking: pd.DataFrame, keepers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracking = tracking.copy()
    phases = proc.summarize_phases(tracking, keepers)
    tracking["phase_id"] = 0

    for phase_id in phases.index:
        start_frame = int(phases.at[phase_id, "start_frame"])
        end_frame = int(phases.at[phase_id, "end_frame"])
        tracking.loc[start_frame:end_frame, "phase_id"] = int(phase_id)

    return tracking, phases


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


def _valid_possessor_snapshot(tracking: pd.DataFrame, frame_id: int, possessor_object_id: str | None) -> bool:
    if possessor_object_id is None or not isinstance(possessor_object_id, str):
        return False
    if frame_id not in tracking.index:
        return False
    x_col = f"{possessor_object_id}_x"
    y_col = f"{possessor_object_id}_y"
    if x_col not in tracking.columns or y_col not in tracking.columns:
        return False
    return not pd.isna(tracking.at[frame_id, x_col]) and not pd.isna(tracking.at[frame_id, y_col])


def _build_actions_and_labels(situation: HawkeyeSituation) -> tuple[pd.DataFrame, torch.Tensor, list[Data], dict[str, int]]:
    actions: list[dict[str, Any]] = []
    labels: list[list[float]] = []
    graphs: list[Data] = []
    feature_dim = infer_node_feature_dim(extend=True)
    period_tracking = situation.tracking[situation.tracking["period_id"] == situation.frame_meta["period_id"].iloc[0]]
    stats = {
        "total_frames": int(len(situation.frame_meta)),
        "valid_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }

    for frame_id, frame_row in situation.frame_meta.iterrows():
        frame_id = int(frame_id)
        possessor_object_id = frame_row["possessor_object_id"]

        if not bool(frame_row["has_ball"]):
            stats["skipped_missing_ball"] += 1
            continue
        if not _valid_possessor_snapshot(situation.tracking, frame_id, possessor_object_id):
            stats["skipped_missing_possessor"] += 1
            continue

        graph = construct_graph_for_frame(
            situation,
            frame_id,
            possessor_object_id,
            period_tracking,
            feature_dim,
            extend=True,
            rotate_to_ltr=False,
        )
        if graph is None:
            stats["skipped_missing_graph"] += 1
            continue

        attacking_players, defending_players = utils.find_active_players(
            situation.tracking,
            frame_id,
            team=frame_row["possession_prefix"],
            include_goals=True,
        )
        if possessor_object_id not in attacking_players:
            stats["skipped_missing_possessor"] += 1
            continue

        intent_index = attacking_players.index(possessor_object_id)
        receiver_index = intent_index
        start_x = float(situation.tracking.at[frame_id, f"{possessor_object_id}_x"])
        start_y = float(situation.tracking.at[frame_id, f"{possessor_object_id}_y"])

        actions.append(
            {
                "frame_id": frame_id,
                "object_id": possessor_object_id,
                "player_id": int(frame_row["PlayerID"]),
                "period_id": int(frame_row["period_id"]),
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
        )
        labels.append(
            [
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
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        graphs.append(graph)
        stats["valid_frames"] += 1

    actions_df = pd.DataFrame(actions).set_index("frame_id") if actions else pd.DataFrame(columns=["frame_id"]).set_index("frame_id")
    label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, len(config.LABEL_COLUMNS)))
    return actions_df, label_tensor, graphs, stats


def build_hawkeye_situation(
    situation_tracking: pd.DataFrame,
    ball: pd.DataFrame,
) -> tuple[HawkeyeSituation, pd.DataFrame, dict[str, int]]:
    if situation_tracking.empty:
        raise ValueError("Cannot build a Hawkeye situation from an empty tracking frame.")

    situation_tracking = situation_tracking.copy()
    team_map = _stable_team_map(situation_tracking)
    prefix_to_team = {prefix: team for team, prefix in team_map.items()}
    object_map = _build_object_map(situation_tracking, team_map)
    frame_meta = _build_frame_meta(situation_tracking, ball, team_map, object_map)
    tracking = _build_tracking_wide(situation_tracking, frame_meta, team_map)

    keepers = object_map.loc[object_map["role"] == 2, "object_id"].dropna().tolist()
    tracking, phases = _assign_phase_ids(tracking, keepers)
    tracking = _add_goal_nodes(tracking)

    situation = HawkeyeSituation(
        situation_id=str(situation_tracking["id"].iloc[0]),
        tracking=tracking,
        phases=phases,
        frame_meta=frame_meta,
        actions=pd.DataFrame(),
        labels=torch.empty((0, len(config.LABEL_COLUMNS))),
        graph_features_0=[],
        team_map=team_map,
        prefix_to_team=prefix_to_team,
        max_players=20 + 2 + 2,
    )
    actions, labels, graphs, stats = _build_actions_and_labels(situation)
    situation.actions = actions
    situation.labels = labels
    situation.graph_features_0 = graphs
    situation.graph_features_by_dir["action_graphs"] = graphs

    attacking_rows = situation_tracking[situation_tracking["team"] == situation_tracking["possession_team"]].copy()
    attacking_rows["uefa_player_id"] = attacking_rows["uefa_player_id"].astype(int)
    attacking_rows["PlayerID"] = attacking_rows["PlayerID"].astype(int)
    attacking_rows["object_id"] = attacking_rows.apply(
        lambda row: _object_id(team_map[str(row["team"])], int(row["uefa_player_id"])),
        axis=1,
    )
    attacking_rows = attacking_rows.merge(
        frame_meta.reset_index()[["frame_id", "game_id", "half", "abs_time", "id", "has_ball"]],
        on=["game_id", "half", "abs_time", "id"],
        how="left",
    )

    return situation, attacking_rows, stats


def infer_hawkeye_components(
    situation: HawkeyeSituation,
    model_specs: dict[str, Any],
    device: str = "cpu",
) -> dict[str, pd.DataFrame]:
    components: dict[str, pd.DataFrame] = {}
    if situation.labels.numel() == 0 or not situation.graph_features_0:
        return components

    action_intent, _ = inference_gnn(situation, model_specs["action_intent"], device=device, post_action=False)
    pass_success, _ = inference_gnn(situation, model_specs["pass_success"], device=device, post_action=False)
    scoring_failure, scoring_success = inference_gnn(
        situation,
        model_specs["outcome_scoring"],
        device=device,
        post_action=False,
    )
    conceding_failure, conceding_success = inference_gnn(
        situation,
        model_specs["outcome_conceding"],
        device=device,
        post_action=False,
    )

    components["action_intent"] = action_intent
    components["pass_success"] = pass_success
    components["outcome_scoring_success"] = scoring_success
    components["outcome_scoring_failure"] = scoring_failure
    components["outcome_conceding_success"] = conceding_success
    components["outcome_conceding_failure"] = conceding_failure
    return components


def _merge_component_column(
    export_rows: pd.DataFrame,
    component_frame: pd.DataFrame | None,
    column_name: str,
) -> pd.DataFrame:
    export_rows[column_name] = np.nan
    if component_frame is None or component_frame.empty:
        return export_rows

    long_component = (
        component_frame.rename_axis("frame_id")
        .reset_index()
        .melt(id_vars="frame_id", var_name="object_id", value_name=column_name)
    )
    export_rows = export_rows.merge(long_component, on=["frame_id", "object_id"], how="left", suffixes=("", "_pred"))
    export_rows[column_name] = export_rows[f"{column_name}_pred"]
    export_rows = export_rows.drop(columns=[f"{column_name}_pred"])
    return export_rows


def build_hawkeye_export(
    attacking_rows: pd.DataFrame,
    situation: HawkeyeSituation,
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    export_rows = attacking_rows.copy()
    for column in COMPONENT_COLUMNS:
        export_rows = _merge_component_column(export_rows, components.get(column), column)

    export_rows = export_rows.drop(columns=["object_id"], errors="ignore")
    export_rows["_synthetic_order"] = 0

    shot_rows: list[dict[str, Any]] = []
    action_intent = components.get("action_intent")

    if action_intent is not None and not action_intent.empty:
        base_columns = export_rows.columns.tolist()
        for frame_id, frame_row in situation.frame_meta.iterrows():
            frame_id = int(frame_id)
            if frame_id not in action_intent.index:
                continue

            goal_object_id = f"{frame_row['possession_prefix']}_goal"
            shot_prob = action_intent.at[frame_id, goal_object_id] if goal_object_id in action_intent.columns else np.nan
            if pd.isna(shot_prob):
                continue

            shot_row = {column: np.nan for column in base_columns}
            shot_row.update(
                {
                    "game_id": int(frame_row["game_id"]),
                    "half": int(frame_row["half"]),
                    "abs_time": float(frame_row["abs_time"]),
                    "uefa_player_id": 1,
                    "PlayerID": int(frame_row["PlayerID"]),
                    "id": frame_row["id"],
                    "possession_team": frame_row["possession_team"],
                    "frame_id": frame_id,
                    "has_ball": True,
                    "action_intent": float(shot_prob),
                    "_synthetic_order": 1,
                }
            )
            shot_rows.append(shot_row)

    if shot_rows:
        export_rows = pd.concat([export_rows, pd.DataFrame(shot_rows)], ignore_index=True, sort=False)

    export_rows = export_rows.sort_values(["id", "abs_time", "_synthetic_order", "team", "uefa_player_id"], na_position="last")
    export_rows = export_rows.drop(columns=["frame_id", "has_ball", "_synthetic_order"], errors="ignore")
    return export_rows.reset_index(drop=True)


def summarize_hawkeye_stats(stats_by_situation: dict[str, dict[str, int]]) -> dict[str, int]:
    totals = {
        "situations": len(stats_by_situation),
        "total_frames": 0,
        "valid_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }
    for stats in stats_by_situation.values():
        for key in totals:
            if key == "situations":
                continue
            totals[key] += int(stats.get(key, 0))
    return totals


def load_hawkeye_models(
    action_intent_model_id: str,
    pass_success_model_id: str,
    outcome_scoring_model_id: str,
    outcome_conceding_model_id: str,
    device: str,
) -> dict[str, Any]:
    from models.utils import load_model

    model_specs = {
        "action_intent": load_model(action_intent_model_id, device),
        "pass_success": load_model(pass_success_model_id, device),
        "outcome_scoring": load_model(outcome_scoring_model_id, device),
        "outcome_conceding": load_model(outcome_conceding_model_id, device),
    }
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    return model_specs
