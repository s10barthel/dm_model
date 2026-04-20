from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from datatools import config, utils
from datatools.graph_feature import construct_graph_for_frame, infer_node_feature_dim
from inference import inference_gnn

COMPONENT_COLUMNS = [
    "action_intent",
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]
REQUIRED_STATE_COLUMNS = [
    "team",
    "player",
    "pos_x",
    "pos_y",
    "pos_z",
    "smooth_x_speed",
    "smooth_y_speed",
    "event_player",
    "playing_direction_event",
]
EXPORT_COLUMNS = [
    "modification",
    "game_state",
    "higher_state_id",
    "team",
    "player",
    "pos_x",
    "pos_y",
    "pos_z",
    "smooth_x_speed",
    "smooth_y_speed",
    "event_player",
    "action_intent",
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_conceding_success",
    "outcome_scoring_failure",
    "outcome_conceding_failure",
]
MODIFICATION_DIR_RE = re.compile(r"^modification_(?P<id>\d+)$")


@dataclass
class BenchmarkState:
    modification_id: int
    game_state_id: int
    higher_state_id: int
    tracking: pd.DataFrame
    phases: pd.DataFrame
    frame_meta: pd.DataFrame
    actions: pd.DataFrame
    labels: torch.Tensor
    graph_features_0: list[Data]
    team_map: dict[int, str]
    prefix_to_team: dict[str, int]
    possessor_team: int
    possessor_player: int
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
            self.match_id = self.state_id

    @property
    def state_id(self) -> str:
        return f"modification_{self.modification_id}_game_state_{self.game_state_id}"


def _field_x(x: float) -> float:
    return float(x) + config.FIELD_SIZE[0] / 2


def _field_y(y: float) -> float:
    return float(y) + config.FIELD_SIZE[1] / 2


def _object_id(prefix: str, player_id: int) -> str:
    return f"{prefix}_{int(player_id)}"


def _component_run_paths(component_dir: str | Path) -> tuple[Path, Path]:
    component_root = Path(component_dir)
    return component_root / "benchmark_data.parquet", component_root / "metadata.json"


def discover_benchmark_modifications(
    input_dir: str | Path,
    requested_modifications: list[int] | None = None,
    limit: int | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Benchmark input directory does not exist: {root}")

    skipped: dict[str, list[int]] = {
        "missing_modification_dir": [],
        "missing_game_state_1": [],
        "missing_game_state_2": [],
        "missing_modification_csv": [],
    }

    if requested_modifications:
        candidate_ids = list(dict.fromkeys(int(modification_id) for modification_id in requested_modifications))
    else:
        candidate_ids = sorted(
            int(match.group("id"))
            for path in root.iterdir()
            if path.is_dir() and (match := MODIFICATION_DIR_RE.match(path.name))
        )

    valid_ids: list[int] = []
    for modification_id in candidate_ids:
        modification_dir = root / f"modification_{int(modification_id)}"
        if not modification_dir.exists():
            skipped["missing_modification_dir"].append(int(modification_id))
            continue
        if not (modification_dir / "game_state_1.csv").exists():
            skipped["missing_game_state_1"].append(int(modification_id))
            continue
        if not (modification_dir / "game_state_2.csv").exists():
            skipped["missing_game_state_2"].append(int(modification_id))
            continue
        if not (modification_dir / "modification.csv").exists():
            skipped["missing_modification_csv"].append(int(modification_id))
            continue
        valid_ids.append(int(modification_id))

    if limit is not None:
        valid_ids = valid_ids[:limit]
    if not valid_ids:
        raise ValueError("No usable benchmark modifications were selected.")
    return valid_ids, skipped


def _coerce_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if normalized.isna().any():
        raise ValueError(f"Benchmark column {column_name!r} contains non-boolean values.")
    return normalized.astype(bool)


def load_benchmark_game_state(state_path: str | Path) -> pd.DataFrame:
    state = pd.read_csv(state_path, low_memory=False)
    missing = [column for column in REQUIRED_STATE_COLUMNS if column not in state.columns]
    if missing:
        raise KeyError(f"Benchmark game-state file {state_path} is missing required columns: {missing}")

    cleaned = state.copy()
    integer_cols = ["team", "player", "event_player"]
    float_cols = ["pos_x", "pos_y", "pos_z", "smooth_x_speed", "smooth_y_speed"]
    for column in integer_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype("Int64")
    for column in float_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["playing_direction_event"] = _coerce_bool_series(cleaned["playing_direction_event"], "playing_direction_event")
    cleaned = cleaned.dropna(subset=integer_cols + float_cols).copy()
    for column in integer_cols:
        cleaned[column] = cleaned[column].astype(int)

    ordered_columns = [*REQUIRED_STATE_COLUMNS]
    return cleaned[ordered_columns].reset_index(drop=True)


def load_benchmark_modification_label(label_path: str | Path) -> int:
    label_frame = pd.read_csv(label_path)
    if "higher_state_id" not in label_frame.columns:
        raise KeyError(f"Benchmark modification file {label_path} is missing the required 'higher_state_id' column.")
    if len(label_frame) != 1:
        raise ValueError(f"Benchmark modification file {label_path} must contain exactly one row.")

    higher_state_id = pd.to_numeric(label_frame.at[0, "higher_state_id"], errors="coerce")
    if pd.isna(higher_state_id):
        raise ValueError(f"Benchmark modification file {label_path} contains an invalid 'higher_state_id' value.")
    higher_state_id = int(higher_state_id)
    if higher_state_id not in {1, 2}:
        raise ValueError(f"Benchmark higher_state_id must be 1 or 2, got {higher_state_id}.")
    return higher_state_id


def load_benchmark_modification_data(modification_id: int, input_dir: str | Path) -> dict[str, Any]:
    modification_dir = Path(input_dir) / f"modification_{int(modification_id)}"
    return {
        "modification_id": int(modification_id),
        "higher_state_id": load_benchmark_modification_label(modification_dir / "modification.csv"),
        "game_state_1": load_benchmark_game_state(modification_dir / "game_state_1.csv"),
        "game_state_2": load_benchmark_game_state(modification_dir / "game_state_2.csv"),
    }


def _resolve_state_constants(raw_state: pd.DataFrame) -> tuple[int, bool]:
    event_players = raw_state["event_player"].dropna().astype(int).unique().tolist()
    if len(event_players) != 1:
        raise ValueError(f"Benchmark state must contain exactly one event_player value, found {event_players}.")
    playing_directions = raw_state["playing_direction_event"].dropna().astype(bool).unique().tolist()
    if len(playing_directions) != 1:
        raise ValueError(
            "Benchmark state must contain exactly one playing_direction_event value, "
            f"found {playing_directions}."
        )
    return int(event_players[0]), bool(playing_directions[0])


def _resolve_team_map(raw_state: pd.DataFrame, possessor_player: int) -> tuple[int, dict[int, str]]:
    player_rows = raw_state.loc[(raw_state["team"] != 0) & (raw_state["player"] == int(possessor_player))].copy()
    if len(player_rows) != 1:
        raise ValueError(
            f"Benchmark state must contain exactly one possessor row where player == event_player ({possessor_player}), "
            f"found {len(player_rows)}."
        )

    possessor_team = int(player_rows.iloc[0]["team"])
    teams = sorted(int(team) for team in raw_state["team"].dropna().astype(int).unique().tolist() if int(team) != 0)
    if len(teams) != 2:
        raise ValueError(f"Benchmark state must contain exactly two non-ball teams, found {teams}.")
    other_team = next(team for team in teams if team != possessor_team)
    return possessor_team, {possessor_team: "home", other_team: "away"}


def _normalize_state_rows(raw_state: pd.DataFrame, team_map: dict[int, str], rotate: bool) -> pd.DataFrame:
    normalized = raw_state.copy()
    flip = -1.0 if rotate else 1.0
    normalized["norm_pos_x"] = normalized["pos_x"].astype(float) * flip
    normalized["norm_pos_y"] = normalized["pos_y"].astype(float) * flip
    normalized["norm_speed_x"] = normalized["smooth_x_speed"].astype(float) * flip
    normalized["norm_speed_y"] = normalized["smooth_y_speed"].astype(float) * flip
    normalized["prefix"] = normalized["team"].map(team_map)
    normalized["object_id"] = pd.NA
    player_mask = normalized["team"] != 0
    normalized.loc[player_mask, "object_id"] = normalized.loc[player_mask].apply(
        lambda row: _object_id(str(row["prefix"]), int(row["player"])),
        axis=1,
    )
    return normalized


def _build_tracking_and_phases(normalized_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ball_rows = normalized_rows.loc[(normalized_rows["team"] == 0) & (normalized_rows["player"] == 0)].copy()
    if len(ball_rows) != 1:
        raise ValueError(f"Benchmark state must contain exactly one ball row, found {len(ball_rows)}.")

    ball_row = ball_rows.iloc[0]
    wide_row: dict[str, Any] = {
        "frame_id": 0,
        "period_id": 1,
        "timestamp": 0.0,
        "episode_id": 0,
        "phase_id": 1,
        "ball_state": "alive",
        "ball_owning_home_away": "home",
        "ball_x": _field_x(float(ball_row["norm_pos_x"])),
        "ball_y": _field_y(float(ball_row["norm_pos_y"])),
        "ball_vx": float(ball_row["norm_speed_x"]),
        "ball_vy": float(ball_row["norm_speed_y"]),
        "ball_speed": float(np.hypot(ball_row["norm_speed_x"], ball_row["norm_speed_y"])),
        "ball_accel": 0.0,
        "ball_vz": 0.0,
        "ball_z": float(ball_row["pos_z"]),
    }

    players = normalized_rows.loc[normalized_rows["team"] != 0].copy()
    players["sort_key"] = players["object_id"].map(utils.player_sort_key)
    players = players.sort_values("sort_key").reset_index(drop=True)

    home_players = players.loc[players["prefix"] == "home", "object_id"].astype(str).tolist()
    away_players = players.loc[players["prefix"] == "away", "object_id"].astype(str).tolist()
    if not home_players or not away_players:
        raise ValueError("Benchmark state must contain at least one home player and one away player after normalization.")

    for row in players.itertuples(index=False):
        object_id = str(row.object_id)
        vx = float(row.norm_speed_x)
        vy = float(row.norm_speed_y)
        wide_row[f"{object_id}_x"] = _field_x(float(row.norm_pos_x))
        wide_row[f"{object_id}_y"] = _field_y(float(row.norm_pos_y))
        wide_row[f"{object_id}_vx"] = vx
        wide_row[f"{object_id}_vy"] = vy
        wide_row[f"{object_id}_speed"] = float(np.hypot(vx, vy))
        wide_row[f"{object_id}_accel"] = 0.0

    goal_features = ["x", "y", "vx", "vy", "speed", "accel"]
    for team in ["home", "away"]:
        for feature in goal_features:
            wide_row[f"{team}_goal_{feature}"] = 0.0
    wide_row["home_goal_x"] = config.FIELD_SIZE[0]
    wide_row["home_goal_y"] = config.FIELD_SIZE[1] / 2
    wide_row["away_goal_x"] = 0.0
    wide_row["away_goal_y"] = config.FIELD_SIZE[1] / 2

    tracking = pd.DataFrame([wide_row]).set_index("frame_id", drop=False)
    home_keeper = min(home_players, key=lambda player_id: float(tracking.at[0, f"{player_id}_x"]))
    away_keeper = max(away_players, key=lambda player_id: float(tracking.at[0, f"{player_id}_x"]))
    phases = pd.DataFrame(
        [
            {
                "phase_id": 1,
                "period_id": 1,
                "start_frame": 0,
                "end_frame": 0,
                "active_players": home_players + away_players,
                "active_keepers": [home_keeper, away_keeper],
            }
        ]
    ).set_index("phase_id")
    return tracking, phases


def _build_frame_meta(
    tracking: pd.DataFrame,
    possessor_player: int,
    modification_id: int,
    game_state_id: int,
    higher_state_id: int,
) -> pd.DataFrame:
    frame_meta = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "period_id": 1,
                "modification": int(modification_id),
                "game_state": int(game_state_id),
                "higher_state_id": int(higher_state_id),
                "possessor_object_id": f"home_{int(possessor_player)}",
                "possessor_player": int(possessor_player),
                "possession_prefix": "home",
                "ball_x": float(tracking.at[0, "ball_x"]),
                "ball_y": float(tracking.at[0, "ball_y"]),
                "ball_z": float(tracking.at[0, "ball_z"]),
                "has_ball": not pd.isna(tracking.at[0, "ball_x"]) and not pd.isna(tracking.at[0, "ball_y"]),
            }
        ]
    ).set_index("frame_id", drop=True)
    frame_meta.index.name = "frame_id"
    return frame_meta


def _build_export_rows(
    raw_state: pd.DataFrame,
    modification_id: int,
    game_state_id: int,
    higher_state_id: int,
    possessor_team: int,
) -> pd.DataFrame:
    export_rows = raw_state.copy()
    export_rows.insert(0, "higher_state_id", int(higher_state_id))
    export_rows.insert(0, "game_state", int(game_state_id))
    export_rows.insert(0, "modification", int(modification_id))
    export_rows["frame_id"] = 0
    export_rows["object_id"] = np.where(
        export_rows["team"] == int(possessor_team),
        export_rows["player"].astype(int).map(lambda player_id: _object_id("home", int(player_id))),
        pd.NA,
    )
    export_rows.loc[export_rows["team"] == 0, "object_id"] = pd.NA
    return export_rows


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


def _build_actions_and_labels(
    state: BenchmarkState,
    add_v_edge_features: bool = False,
) -> tuple[pd.DataFrame, torch.Tensor, list[Data], dict[str, int]]:
    actions: list[dict[str, Any]] = []
    labels: list[list[float]] = []
    graphs: list[Data] = []
    feature_dim = infer_node_feature_dim(extend=True)
    period_tracking = state.tracking[state.tracking["period_id"] == state.frame_meta["period_id"].iloc[0]]
    stats = {
        "total_frames": int(len(state.frame_meta)),
        "valid_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }

    for frame_id, frame_row in state.frame_meta.iterrows():
        frame_id = int(frame_id)
        possessor_object_id = str(frame_row["possessor_object_id"])

        if not bool(frame_row["has_ball"]):
            stats["skipped_missing_ball"] += 1
            continue
        if not _valid_possessor_snapshot(state.tracking, frame_id, possessor_object_id):
            stats["skipped_missing_possessor"] += 1
            continue

        graph = construct_graph_for_frame(
            state,
            frame_id,
            possessor_object_id,
            period_tracking,
            feature_dim,
            extend=True,
            rotate_to_ltr=False,
            add_v_edge_features=add_v_edge_features,
        )
        if graph is None:
            stats["skipped_missing_graph"] += 1
            continue

        attacking_players, defending_players = utils.find_active_players(
            state.tracking,
            frame_id,
            team="home",
            include_goals=True,
        )
        if possessor_object_id not in attacking_players:
            stats["skipped_missing_possessor"] += 1
            continue

        intent_index = attacking_players.index(possessor_object_id)
        receiver_index = intent_index
        start_x = float(state.tracking.at[frame_id, f"{possessor_object_id}_x"])
        start_y = float(state.tracking.at[frame_id, f"{possessor_object_id}_y"])

        actions.append(
            {
                "frame_id": frame_id,
                "object_id": possessor_object_id,
                "player_id": int(frame_row["possessor_player"]),
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

    actions_df = (
        pd.DataFrame(actions).set_index("frame_id", drop=False)
        if actions
        else pd.DataFrame(columns=["frame_id"]).set_index("frame_id", drop=False)
    )
    label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0, len(config.LABEL_COLUMNS)))
    return actions_df, label_tensor, graphs, stats


def build_benchmark_state(
    raw_state: pd.DataFrame,
    modification_id: int,
    game_state_id: int,
    higher_state_id: int,
    add_v_edge_features: bool = False,
    build_graphs: bool = True,
) -> tuple[BenchmarkState, pd.DataFrame, dict[str, int]]:
    if raw_state.empty:
        raise ValueError("Cannot build a benchmark state from an empty dataframe.")

    raw_state = raw_state.copy().reset_index(drop=True)
    possessor_player, playing_direction_event = _resolve_state_constants(raw_state)
    rotate = not playing_direction_event
    possessor_team, team_map = _resolve_team_map(raw_state, possessor_player)
    prefix_to_team = {prefix: team for team, prefix in team_map.items()}
    normalized_rows = _normalize_state_rows(raw_state, team_map, rotate)
    tracking, phases = _build_tracking_and_phases(normalized_rows)
    frame_meta = _build_frame_meta(tracking, possessor_player, modification_id, game_state_id, higher_state_id)

    state = BenchmarkState(
        modification_id=int(modification_id),
        game_state_id=int(game_state_id),
        higher_state_id=int(higher_state_id),
        tracking=tracking,
        phases=phases,
        frame_meta=frame_meta,
        actions=pd.DataFrame(),
        labels=torch.empty((0, len(config.LABEL_COLUMNS))),
        graph_features_0=[],
        team_map=team_map,
        prefix_to_team=prefix_to_team,
        possessor_team=int(possessor_team),
        possessor_player=int(possessor_player),
        max_players=22 + 2,
    )
    export_rows = _build_export_rows(raw_state, modification_id, game_state_id, higher_state_id, possessor_team)

    if not build_graphs:
        stats = {
            "total_frames": int(len(state.frame_meta)),
            "valid_frames": 0,
            "skipped_missing_ball": 0,
            "skipped_missing_possessor": 0,
            "skipped_missing_graph": 0,
        }
        return state, export_rows, stats

    actions, labels, graphs, stats = _build_actions_and_labels(
        state,
        add_v_edge_features=add_v_edge_features,
    )
    state.actions = actions
    state.labels = labels
    state.graph_features_0 = graphs
    state.graph_features_by_dir["action_graphs"] = graphs
    return state, export_rows, stats


def infer_benchmark_components(
    state: BenchmarkState,
    model_specs: dict[str, Any],
    device: str = "cpu",
) -> dict[str, pd.DataFrame]:
    components: dict[str, pd.DataFrame] = {}
    if state.labels.numel() == 0 or not state.graph_features_0:
        return components

    action_intent, _ = inference_gnn(state, model_specs["action_intent"], device=device, post_action=False)
    pass_intent, _ = inference_gnn(state, model_specs["pass_intent"], device=device, post_action=False)
    pass_success, _ = inference_gnn(state, model_specs["pass_success"], device=device, post_action=False)
    scoring_failure, scoring_success = inference_gnn(
        state,
        model_specs["outcome_scoring"],
        device=device,
        post_action=False,
    )
    conceding_failure, conceding_success = inference_gnn(
        state,
        model_specs["outcome_conceding"],
        device=device,
        post_action=False,
    )

    components["action_intent"] = action_intent
    components["pass_intent"] = pass_intent
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


def build_benchmark_export(
    export_rows: pd.DataFrame,
    state: BenchmarkState,
    components: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    export_rows = export_rows.copy()
    for column in COMPONENT_COLUMNS:
        export_rows = _merge_component_column(export_rows, components.get(column), column)

    export_rows = export_rows.drop(columns=["object_id"], errors="ignore")
    export_rows["_synthetic_order"] = 0

    shot_rows: list[dict[str, Any]] = []
    action_intent = components.get("action_intent")
    if action_intent is not None and not action_intent.empty:
        base_columns = export_rows.columns.tolist()
        for frame_id, frame_row in state.frame_meta.iterrows():
            frame_id = int(frame_id)
            if frame_id not in action_intent.index:
                continue

            shot_prob = action_intent.at[frame_id, "home_goal"] if "home_goal" in action_intent.columns else np.nan
            if pd.isna(shot_prob):
                continue

            shot_row = {column: np.nan for column in base_columns}
            shot_row.update(
                {
                    "modification": int(frame_row["modification"]),
                    "game_state": int(frame_row["game_state"]),
                    "higher_state_id": int(frame_row["higher_state_id"]),
                    "frame_id": frame_id,
                    "action_intent": float(shot_prob),
                    "_synthetic_order": 1,
                }
            )
            shot_rows.append(shot_row)

    if shot_rows:
        export_rows = pd.concat([export_rows, pd.DataFrame(shot_rows)], ignore_index=True, sort=False)

    export_rows = export_rows.sort_values(
        ["modification", "game_state", "_synthetic_order", "team", "player"],
        na_position="last",
    )
    export_rows = export_rows.drop(columns=["frame_id", "_synthetic_order"], errors="ignore")
    return export_rows[EXPORT_COLUMNS].reset_index(drop=True)


def summarize_benchmark_stats(stats_by_state: dict[str, dict[str, int]]) -> dict[str, int]:
    totals = {
        "states": len(stats_by_state),
        "total_frames": 0,
        "valid_frames": 0,
        "skipped_missing_ball": 0,
        "skipped_missing_possessor": 0,
        "skipped_missing_graph": 0,
    }
    for stats in stats_by_state.values():
        for key in totals:
            if key == "states":
                continue
            totals[key] += int(stats.get(key, 0))
    return totals


def load_benchmark_component_run(component_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet_path, metadata_path = _component_run_paths(component_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Benchmark component parquet not found at {parquet_path}. Run scripts/run_benchmark.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Benchmark component metadata not found at {metadata_path}. Run scripts/run_benchmark.py first."
        )

    component_export = pd.read_parquet(parquet_path)
    required = {"modification", "game_state"}
    missing = sorted(required - set(component_export.columns))
    if missing:
        raise KeyError(f"Benchmark component export at {parquet_path} is missing required columns: {missing}")

    component_export["modification"] = pd.to_numeric(component_export["modification"], errors="coerce").astype("Int64")
    component_export["game_state"] = pd.to_numeric(component_export["game_state"], errors="coerce").astype("Int64")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return component_export, metadata


def resolve_benchmark_component_states(
    component_export: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    requested_modifications: list[int] | None = None,
    requested_game_states: list[int] | None = None,
) -> list[tuple[int, int]]:
    metadata = metadata or {}
    available_pairs = [
        (int(item["modification"]), int(item["game_state"]))
        for item in metadata.get("processed_states", [])
        if {"modification", "game_state"} <= set(item)
    ]
    if not available_pairs:
        rows = component_export[["modification", "game_state"]].dropna().drop_duplicates()
        available_pairs = [(int(row.modification), int(row.game_state)) for row in rows.itertuples(index=False)]

    if not available_pairs:
        raise ValueError("No benchmark states were found in the selected component run.")

    selected_pairs = list(dict.fromkeys(available_pairs))
    if requested_modifications:
        requested_modifications = list(dict.fromkeys(int(value) for value in requested_modifications))
        selected_pairs = [pair for pair in selected_pairs if pair[0] in set(requested_modifications)]
    if requested_game_states:
        requested_game_states = list(dict.fromkeys(int(value) for value in requested_game_states))
        selected_pairs = [pair for pair in selected_pairs if pair[1] in set(requested_game_states)]

    if requested_modifications or requested_game_states:
        available_set = set(available_pairs)
        requested_pairs = [
            (modification_id, game_state_id)
            for modification_id, game_state_id in available_set
            if (not requested_modifications or modification_id in set(requested_modifications))
            and (not requested_game_states or game_state_id in set(requested_game_states))
        ]
        if not requested_pairs:
            raise KeyError("Requested benchmark modifications/game states are not present in the selected component run.")

    if not selected_pairs:
        raise ValueError("No benchmark states remain after applying the requested filters.")
    return sorted(selected_pairs)


def _resolve_component_object_id(
    row: pd.Series,
    state: BenchmarkState,
    component_name: str,
) -> str | None:
    team = row.get("team")
    player = row.get("player")
    if pd.notna(team) and pd.notna(player):
        team_value = int(team)
        if team_value == state.possessor_team:
            return _object_id("home", int(player))
        if team_value in state.team_map:
            return _object_id(state.team_map[team_value], int(player))

    if component_name == "action_intent" and pd.isna(team) and pd.isna(player):
        return "home_goal"
    return None


def build_benchmark_component_tables(
    component_export: pd.DataFrame,
    state: BenchmarkState,
) -> dict[str, pd.DataFrame]:
    state_rows = component_export.loc[
        (component_export["modification"] == int(state.modification_id))
        & (component_export["game_state"] == int(state.game_state_id))
    ].copy()
    component_tables: dict[str, pd.DataFrame] = {}

    for component_name in COMPONENT_COLUMNS:
        if component_name not in state_rows.columns:
            component_tables[component_name] = pd.DataFrame()
            continue

        component_rows = state_rows.loc[state_rows[component_name].notna()].copy()
        if component_rows.empty:
            component_tables[component_name] = pd.DataFrame()
            continue

        records: list[dict[str, Any]] = []
        for row in component_rows.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            object_id = _resolve_component_object_id(row_series, state, component_name)
            if object_id is None:
                continue
            records.append({"frame_id": 0, "object_id": object_id, "value": float(row_series[component_name])})

        if not records:
            component_tables[component_name] = pd.DataFrame()
            continue

        component_frame = (
            pd.DataFrame(records)
            .drop_duplicates(subset=["frame_id", "object_id"], keep="last")
            .pivot(index="frame_id", columns="object_id", values="value")
            .sort_index()
        )
        component_frame.columns = component_frame.columns.astype(str)
        component_tables[component_name] = component_frame

    return component_tables


def build_benchmark_visualization_probs(row: pd.Series | None) -> pd.Series:
    if row is None:
        return pd.Series(dtype=float)

    probs = pd.to_numeric(row, errors="coerce").dropna()
    return probs.astype(float).sort_values(ascending=False)


def load_benchmark_models(
    action_intent_model_id: str,
    pass_intent_model_id: str,
    pass_success_model_id: str,
    outcome_scoring_model_id: str,
    outcome_conceding_model_id: str,
    device: str,
) -> dict[str, Any]:
    from models.utils import load_model

    model_specs = {
        "action_intent": load_model(action_intent_model_id, device),
        "pass_intent": load_model(pass_intent_model_id, device),
        "pass_success": load_model(pass_success_model_id, device),
        "outcome_scoring": load_model(outcome_scoring_model_id, device),
        "outcome_conceding": load_model(outcome_conceding_model_id, device),
    }
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    return model_specs
