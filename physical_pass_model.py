from __future__ import annotations

import hashlib
import math
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from datatools import config
from datatools.config import FIELD_SIZE, LABEL_INDEX
from project_config import get_physical_xpass_match_path

PHYSICAL_XPASS_SOURCE = "accessible_space_max_player_cum_prob_as_defaults"
PHYSICAL_XPASS_LEGACY_SOURCE = "accessible_space_player_cum_prob"
PHYSICAL_XPASS_SOURCES = {PHYSICAL_XPASS_SOURCE, PHYSICAL_XPASS_LEGACY_SOURCE}
PHYSICAL_XPASS_NEUTRAL_PROB = 0.5
PHYSICAL_XPASS_LOGIT_ATTR = "physical_xpass_logit"
PHYSICAL_XPASS_PROB_ATTR = "physical_xpass"
PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE = "ignore_teammates"
PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER = "consider_teammates"
PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX = "package_max"
PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED = "exact_separate_speed"
PHYSICAL_XPASS_SPEED_AGGREGATIONS = {
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
}
PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION = PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX
PHYSICAL_MODEL_VARIANTS = {
    "gat_baseline",
    "gat_plus_phys_feature",
    "gat_phys_logit_offset",
    "gat_phys_logit_offset_regularized",
}
PHYSICAL_XPASS_VARIANTS = {
    "gat_plus_phys_feature",
    "gat_phys_logit_offset",
    "gat_phys_logit_offset_regularized",
}
PHYSICAL_XPASS_ID_COLUMNS = {"match_id", "action_index", "action_id", "physical_state_hash"}
DEFAULT_V0_MIN = 8.886015553615485
DEFAULT_V0_MAX = 42.18118275402132
DEFAULT_N_V0 = 14
AS_DEFAULT_N_ANGLES = 30
AS_DEFAULT_PHI_OFFSET = 0.0
AS_DEFAULT_N_V0 = 15
AS_DEFAULT_V0_MIN = 3.0
AS_DEFAULT_V0_MAX = 30.0
AS_DEFAULT_PASS_START_LOCATION_OFFSET = 0.0
AS_DEFAULT_TIME_OFFSET_BALL = 0.0
AS_DEFAULT_RADIAL_GRIDSIZE = 3.0
AS_DEFAULT_B0 = -4.565680899844368
AS_DEFAULT_B1 = -2000.0
AS_DEFAULT_PLAYER_VELOCITY = 9.0
AS_DEFAULT_KEEP_INERTIAL_VELOCITY = True
AS_DEFAULT_USE_MAX = False
AS_DEFAULT_V_MAX = 19.85563874348074
AS_DEFAULT_A_MAX = 10.659091365334193
AS_DEFAULT_INERTIAL_SECONDS = 0.17
AS_DEFAULT_TOL_DISTANCE = 5.0
AS_DEFAULT_USE_APPROX_TWO_POINT = True
AS_DEFAULT_V0_PROB_AGGREGATION_MODE = "mean"
AS_DEFAULT_NORMALIZE = True
AS_DEFAULT_USE_EFFICIENT_SIGMOID = True
AS_DEFAULT_FACTOR = 5.077423030272923
AS_DEFAULT_FACTOR2 = 1.0063028450754512
AS_DEFAULT_RESPECT_OFFSIDE = True
AS_DEFAULT_EXCLUDE_PASSER = False


def physical_xpass_as_default_metadata(
    teammate_policy: str = PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    *,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
) -> dict[str, Any]:
    if teammate_policy not in {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE, PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER}:
        raise ValueError(
            f"Unsupported teammate_policy={teammate_policy!r}. "
            f"Expected {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE!r} or {PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER!r}."
        )
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    return {
        "metric": "max_player_cum_prob",
        "source": PHYSICAL_XPASS_SOURCE,
        "teammate_policy": teammate_policy,
        "speed_aggregation": speed_aggregation,
        "coordinate_system": "centered_pitch",
        "n_angles": AS_DEFAULT_N_ANGLES,
        "phi_offset": AS_DEFAULT_PHI_OFFSET,
        "n_v0": AS_DEFAULT_N_V0,
        "v0_min": AS_DEFAULT_V0_MIN,
        "v0_max": AS_DEFAULT_V0_MAX,
        "pass_start_location_offset": AS_DEFAULT_PASS_START_LOCATION_OFFSET,
        "time_offset_ball": AS_DEFAULT_TIME_OFFSET_BALL,
        "radial_gridsize": AS_DEFAULT_RADIAL_GRIDSIZE,
        "b0": AS_DEFAULT_B0,
        "b1": AS_DEFAULT_B1,
        "player_velocity": AS_DEFAULT_PLAYER_VELOCITY,
        "keep_inertial_velocity": AS_DEFAULT_KEEP_INERTIAL_VELOCITY,
        "use_max": AS_DEFAULT_USE_MAX,
        "v_max": AS_DEFAULT_V_MAX,
        "a_max": AS_DEFAULT_A_MAX,
        "inertial_seconds": AS_DEFAULT_INERTIAL_SECONDS,
        "tol_distance": AS_DEFAULT_TOL_DISTANCE,
        "use_approx_two_point": AS_DEFAULT_USE_APPROX_TWO_POINT,
        "v0_prob_aggregation_mode": (
            "max"
            if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX
            else AS_DEFAULT_V0_PROB_AGGREGATION_MODE
        ),
        "normalize": AS_DEFAULT_NORMALIZE,
        "use_efficient_sigmoid": AS_DEFAULT_USE_EFFICIENT_SIGMOID,
        "factor": AS_DEFAULT_FACTOR,
        "factor2": AS_DEFAULT_FACTOR2,
        "respect_offside": AS_DEFAULT_RESPECT_OFFSIDE,
        "exclude_passer": teammate_policy == PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    }


def normalize_physical_xpass_speed_aggregation(value: str | None) -> str:
    if value is None:
        return PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED
    value = str(value)
    if value not in PHYSICAL_XPASS_SPEED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported physical_xpass_speed_aggregation={value!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SPEED_AGGREGATIONS)}."
        )
    return value


def validate_physical_xpass_cache_metadata(
    cache_dir: str | Path,
    *,
    expected_source: str | None = None,
    expected_speed_aggregation: str | None = None,
) -> dict[str, Any]:
    metadata_path = Path(cache_dir) / "metadata.json"
    expected_source = expected_source or PHYSICAL_XPASS_SOURCE
    if expected_source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={expected_source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    rerun_message = (
        "Run scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> --overwrite "
        "to regenerate compatible physical xPass sidecars."
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Physical xPass metadata not found at {metadata_path}. {rerun_message}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    source = metadata.get("source")
    if source != expected_source:
        raise ValueError(
            f"Physical xPass sidecars at {cache_dir} use incompatible source {source!r}; "
            f"expected {expected_source!r}. {rerun_message}"
        )
    if expected_speed_aggregation is not None and source == PHYSICAL_XPASS_SOURCE:
        actual_speed_aggregation = normalize_physical_xpass_speed_aggregation(metadata.get("speed_aggregation"))
        expected_speed_aggregation = normalize_physical_xpass_speed_aggregation(expected_speed_aggregation)
        if actual_speed_aggregation != expected_speed_aggregation:
            raise ValueError(
                f"Physical xPass sidecars at {cache_dir} use incompatible speed_aggregation "
                f"{actual_speed_aggregation!r}; expected {expected_speed_aggregation!r}. {rerun_message}"
            )
    return metadata


def _get_arg(args: Any, name: str, default: Any = None) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def physical_xpass_enabled(args: Any) -> bool:
    return bool(_get_arg(args, "use_physical_xpass", False))


def physical_xpass_model_variant(args: Any) -> str:
    variant = str(_get_arg(args, "model_variant", "gat_baseline") or "gat_baseline")
    if variant not in PHYSICAL_MODEL_VARIANTS:
        raise ValueError(f"Unsupported model_variant={variant!r}. Expected one of {sorted(PHYSICAL_MODEL_VARIANTS)}.")
    return variant


def physical_xpass_source(args: Any) -> str:
    source = _get_arg(args, "physical_xpass_source", None) or PHYSICAL_XPASS_SOURCE
    source = str(source)
    if source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    return source


def physical_xpass_teammate_policy(args: Any, *, source: str | None = None) -> str:
    policy = _get_arg(args, "physical_xpass_teammate_policy", None)
    if policy is None:
        resolved_source = source or physical_xpass_source(args)
        return (
            PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
            if resolved_source == PHYSICAL_XPASS_LEGACY_SOURCE
            else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
        )
    policy = str(policy)
    if policy not in {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE, PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER}:
        raise ValueError(
            f"Unsupported physical_xpass_teammate_policy={policy!r}. "
            f"Expected {PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE!r} or {PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER!r}."
        )
    return policy


def physical_xpass_speed_aggregation(args: Any) -> str:
    value = _get_arg(args, "physical_xpass_speed_aggregation", None)
    if value is None:
        value = _get_arg(args, "speed_aggregation", None)
    return normalize_physical_xpass_speed_aggregation(value)


def model_uses_physical_xpass(args: Any) -> bool:
    task = _get_arg(args, "task", None)
    return task == "pass_success" and physical_xpass_enabled(args) and physical_xpass_model_variant(args) in PHYSICAL_XPASS_VARIANTS


def validate_physical_xpass_args(args: Any) -> None:
    variant = physical_xpass_model_variant(args)
    eps = float(_get_arg(args, "physical_eps", 1e-4))
    residual_clip_value = _get_arg(args, "residual_clip_value", None)
    residual_regularization_lambda = float(_get_arg(args, "residual_regularization_lambda", 0.0) or 0.0)

    if not (0.0 < eps < 0.5):
        raise ValueError(f"--physical-eps must be between 0 and 0.5, got {eps}.")
    if residual_regularization_lambda < 0:
        raise ValueError("--residual-regularization-lambda must be non-negative.")
    if residual_clip_value is not None and float(residual_clip_value) <= 0:
        raise ValueError("--residual-clip-value must be positive when provided.")
    if _get_arg(args, "physical_xpass_speed_aggregation", None) is not None or _get_arg(args, "speed_aggregation", None) is not None:
        physical_xpass_speed_aggregation(args)
    if physical_xpass_enabled(args) and _get_arg(args, "task", None) != "pass_success":
        raise ValueError("--use_physical_xpass is only supported for task=pass_success.")
    if physical_xpass_enabled(args) and variant == "gat_baseline":
        raise ValueError("--use_physical_xpass requires a physical model variant, not gat_baseline.")
    if model_uses_physical_xpass(args) and bool(_get_arg(args, "include_out", False)):
        raise ValueError("--include_out is not supported with physical xPass because player_cum_prob has no out node.")


def probability_to_logit(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    prob = torch.as_tensor(prob, dtype=torch.float32)
    prob = torch.clamp(prob, float(eps), 1.0 - float(eps))
    return torch.log(prob / (1.0 - prob))


def probability_to_logit_numpy(prob: np.ndarray | pd.Series | list[float], eps: float = 1e-4) -> np.ndarray:
    values = np.asarray(prob, dtype=float)
    values = np.clip(values, float(eps), 1.0 - float(eps))
    return np.log(values / (1.0 - values))


def _node_ids(graph: Data) -> list[str]:
    node_ids = getattr(graph, "node_ids", None)
    if node_ids is None:
        raise ValueError("Graph is missing node_ids; physical xPass requires player-id aligned graph nodes.")
    node_ids = [str(node_id) for node_id in node_ids]
    if len(node_ids) != int(graph.x.shape[0]):
        raise ValueError(f"Graph node_ids length {len(node_ids)} does not match node count {int(graph.x.shape[0])}.")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Graph node_ids contain duplicates; physical xPass alignment would be ambiguous.")
    return node_ids


def _candidate_target_indices(graph: Data) -> list[int]:
    teammate = graph.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
    possessor = graph.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
    goal = (
        graph.x[:, config.NODE_FEATURE_IS_GOAL] == 1
        if graph.x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.zeros(graph.x.shape[0], dtype=torch.bool)
    )
    finite_xy = torch.isfinite(graph.x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1]).all(dim=1)
    mask = teammate & (~possessor) & (~goal) & finite_xy
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def physical_state_hash(graph: Data) -> str:
    node_ids = _node_ids(graph)
    x = graph.x.detach().cpu().to(torch.float32)
    feature_indices = [
        config.NODE_FEATURE_IS_TEAMMATE,
        config.NODE_FEATURE_X,
        config.NODE_FEATURE_Y,
        config.NODE_FEATURE_VX,
        config.NODE_FEATURE_VY,
        config.NODE_FEATURE_IS_GOAL,
        config.NODE_FEATURE_IS_POSSESSOR,
    ]
    rows: list[dict[str, Any]] = []
    for node_index, node_id in enumerate(node_ids):
        values: dict[str, Any] = {"node_id": node_id}
        for feature_index in feature_indices:
            if feature_index >= x.shape[1]:
                values[str(feature_index)] = None
                continue
            value = float(x[node_index, feature_index].item())
            values[str(feature_index)] = None if not math.isfinite(value) else round(value, 6)
        rows.append(values)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_simulation_probability(
    simulation_result: Any,
    *,
    frame_index: int,
    target_player_index: int,
    angle_index: int,
    target_distance: float,
) -> tuple[float, dict[str, Any]]:
    player_cum_prob = getattr(simulation_result, "player_cum_prob", None)
    if player_cum_prob is None:
        raise ValueError("accessible-space simulation_result did not return player_cum_prob.")

    arr = np.asarray(player_cum_prob, dtype=float)
    if arr.ndim != 4:
        raise ValueError(f"player_cum_prob must be 4D [frame, player, angle, distance], got shape={arr.shape}.")
    if frame_index < 0 or frame_index >= arr.shape[0]:
        raise ValueError(f"frame_index={frame_index} out of bounds for player_cum_prob shape={arr.shape}.")
    if target_player_index < 0 or target_player_index >= arr.shape[1]:
        raise ValueError(f"target_player_index={target_player_index} out of bounds for player_cum_prob shape={arr.shape}.")
    if angle_index < 0 or angle_index >= arr.shape[2]:
        raise ValueError(f"angle_index={angle_index} out of bounds for player_cum_prob shape={arr.shape}.")

    r_grid = np.asarray(getattr(simulation_result, "r_grid", None), dtype=float)
    if r_grid.ndim != 1 or r_grid.shape[0] != arr.shape[3]:
        raise ValueError(
            f"simulation_result.r_grid must be 1D with length {arr.shape[3]}, got shape={r_grid.shape}."
        )
    distance_index = int(np.nanargmin(np.abs(r_grid - float(target_distance))))
    value = float(arr[frame_index, target_player_index, angle_index, distance_index])
    if not math.isfinite(value):
        raise ValueError(
            "Extracted non-finite player_cum_prob "
            f"for frame={frame_index}, player={target_player_index}, angle={angle_index}, distance={distance_index}."
        )
    return value, {
        "distance": float(target_distance),
        "distance_index": distance_index,
        "player_index": int(target_player_index),
        "frame_index": int(frame_index),
        "angle_index": int(angle_index),
    }


def _resolve_simulate_passes_fn() -> Callable[..., Any]:
    try:
        import accessible_space  # noqa: F401
        from accessible_space.core import simulate_passes_chunked
    except ImportError as exc:
        raise ImportError(
            "accessible-space is required to generate physical xPass sidecars. "
            "Install project dependencies, then rerun scripts/generate_physical_xpass.py."
        ) from exc
    return simulate_passes_chunked


def _validate_simulation_contract(
    *,
    players: np.ndarray,
    passers: np.ndarray,
    exclude_passer: bool,
) -> None:
    if not exclude_passer:
        return
    missing_passers = [str(passer) for passer in np.asarray(passers, dtype=object).tolist() if passer not in players.tolist()]
    if missing_passers:
        raise ValueError(
            "accessible-space requires every passer id to be present in the simulation players array when "
            f"exclude_passer=True. Missing passers: {missing_passers}."
        )


def _build_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
    consider_teammates: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    x = graph.x.detach().cpu().to(torch.float32)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    if consider_teammates:
        sim_player_indices = player_indices
        player_index_to_sim_index = {node_index: sim_index for sim_index, node_index in enumerate(sim_player_indices)}
        missing_targets = [idx for idx in candidate_indices if idx not in player_index_to_sim_index]
        if missing_targets:
            missing_ids = [str(graph.node_ids[idx]) for idx in missing_targets]
            raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

        player_pos = x[sim_player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy()
        player_teams = np.where(
            (x[sim_player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1),
            "attack",
            "defense",
        ).astype(object)
        players = np.array([str(graph.node_ids[idx]) for idx in sim_player_indices], dtype=object)
        target_index_lookup = {graph_target_index: player_index_to_sim_index[graph_target_index] for graph_target_index in candidate_indices}
        return (
            np.repeat(player_pos[np.newaxis, :, :], len(candidate_indices), axis=0),
            player_teams,
            players,
            x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy(),
            np.repeat("attack", len(candidate_indices)).astype(object),
            target_index_lookup,
        )

    defender_indices = [
        node_index
        for node_index in player_indices
        if node_index != possessor_index and int(x[node_index, config.NODE_FEATURE_IS_TEAMMATE].item()) == 0
    ]
    frame_player_indices = [[possessor_index, graph_target_index] + defender_indices for graph_target_index in candidate_indices]
    player_counts = {len(indices) for indices in frame_player_indices}
    if len(player_counts) != 1:
        raise ValueError(f"Physical xPass reduced simulation player count must be constant across candidates, got {sorted(player_counts)}.")

    player_pos = np.stack(
        [x[indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy() for indices in frame_player_indices],
        axis=0,
    )
    player_teams = np.array(["attack", "attack"] + ["defense"] * len(defender_indices), dtype=object)
    players = np.array(
        [str(graph.node_ids[possessor_index]), "target_player"] + [str(graph.node_ids[idx]) for idx in defender_indices],
        dtype=object,
    )
    target_index_lookup = {graph_target_index: 1 for graph_target_index in candidate_indices}
    return (
        player_pos,
        player_teams,
        players,
        x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy(),
        np.repeat("attack", len(candidate_indices)).astype(object),
        target_index_lookup,
    )


def _infer_playing_direction_from_centered_players(player_pos: np.ndarray, player_teams: np.ndarray) -> float:
    finite_x = np.isfinite(player_pos[:, 0])
    attack_x = player_pos[(player_teams == "attack") & finite_x, 0]
    defense_x = player_pos[(player_teams == "defense") & finite_x, 0]
    if attack_x.size == 0 or defense_x.size == 0:
        return 1.0
    return 1.0 if float(np.nanmean(attack_x)) <= float(np.nanmean(defense_x)) else -1.0


def _full_as_default_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], int]:
    x = graph.x.detach().cpu().to(torch.float32)
    node_ids = _node_ids(graph)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    player_index_to_sim_index = {node_index: sim_index for sim_index, node_index in enumerate(player_indices)}
    missing_targets = [node_index for node_index in candidate_indices if node_index not in player_index_to_sim_index]
    if missing_targets:
        missing_ids = [node_ids[node_index] for node_index in missing_targets]
        raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

    player_pos = x[player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy()
    player_pos = np.asarray(player_pos, dtype=float)
    player_pos[:, 0] -= FIELD_SIZE[0] / 2.0
    player_pos[:, 1] -= FIELD_SIZE[1] / 2.0
    player_teams = np.where((x[player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1), "attack", "defense").astype(object)
    players = np.array([node_ids[node_index] for node_index in player_indices], dtype=object)
    ball_pos = x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy().astype(float)
    ball_pos = np.array([ball_pos[0] - FIELD_SIZE[0] / 2.0, ball_pos[1] - FIELD_SIZE[1] / 2.0], dtype=float)
    target_index_lookup = {node_index: player_index_to_sim_index[node_index] for node_index in candidate_indices}
    return player_pos, player_teams, players, ball_pos, target_index_lookup, possessor_index


def _reduced_as_default_simulation_inputs(
    graph: Data,
    *,
    candidate_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[int, int]], int, float]:
    x = graph.x.detach().cpu().to(torch.float32)
    node_ids = _node_ids(graph)
    player_mask = (
        (x[:, config.NODE_FEATURE_IS_GOAL] == 0)
        if x.shape[1] > config.NODE_FEATURE_IS_GOAL
        else torch.ones(x.shape[0], dtype=torch.bool)
    )
    player_mask &= torch.isfinite(x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1]).all(dim=1)
    player_indices = torch.nonzero(player_mask, as_tuple=False).flatten().tolist()
    if not player_indices:
        raise ValueError("No finite non-goal players found for physical xPass simulation.")

    possessor_indices = torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).flatten().tolist()
    if len(possessor_indices) != 1:
        raise ValueError(f"Expected exactly one possessor node, found {len(possessor_indices)}.")
    possessor_index = int(possessor_indices[0])
    if possessor_index not in player_indices:
        raise ValueError("Possessor node is not part of the simulated player set.")

    missing_targets = [node_index for node_index in candidate_indices if node_index not in player_indices]
    if missing_targets:
        missing_ids = [node_ids[node_index] for node_index in missing_targets]
        raise ValueError(f"Candidate target nodes are missing from the physical player mapping: {missing_ids}.")

    full_player_pos = x[player_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy().astype(float)
    full_player_pos[:, 0] -= FIELD_SIZE[0] / 2.0
    full_player_pos[:, 1] -= FIELD_SIZE[1] / 2.0
    full_player_teams = np.where(
        (x[player_indices, config.NODE_FEATURE_IS_TEAMMATE].numpy() == 1),
        "attack",
        "defense",
    ).astype(object)
    playing_direction = _infer_playing_direction_from_centered_players(full_player_pos, full_player_teams)

    defender_indices = [
        node_index
        for node_index in player_indices
        if node_index != possessor_index and int(x[node_index, config.NODE_FEATURE_IS_TEAMMATE].item()) == 0
    ]
    frame_player_indices = [[possessor_index, graph_target_index] + defender_indices for graph_target_index in candidate_indices]
    player_counts = {len(indices) for indices in frame_player_indices}
    if len(player_counts) != 1:
        raise ValueError(
            f"Physical xPass reduced simulation player count must be constant across candidates, got {sorted(player_counts)}."
        )

    player_pos = np.stack(
        [x[indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy() for indices in frame_player_indices],
        axis=0,
    ).astype(float)
    player_pos[:, :, 0] -= FIELD_SIZE[0] / 2.0
    player_pos[:, :, 1] -= FIELD_SIZE[1] / 2.0
    player_teams = np.array(["attack", "attack"] + ["defense"] * len(defender_indices), dtype=object)
    players = np.array(
        [node_ids[possessor_index], "target_player"] + [node_ids[node_index] for node_index in defender_indices],
        dtype=object,
    )
    ball_pos = x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy().astype(float)
    ball_pos = np.array([ball_pos[0] - FIELD_SIZE[0] / 2.0, ball_pos[1] - FIELD_SIZE[1] / 2.0], dtype=float)
    target_index_lookup = {node_index: (frame_index, 1) for frame_index, node_index in enumerate(candidate_indices)}
    return player_pos, player_teams, players, ball_pos, target_index_lookup, possessor_index, playing_direction


def _on_pitch_mask(simulation_result: Any, frame_index: int = 0) -> np.ndarray:
    x_grid = getattr(simulation_result, "x_grid", None)
    y_grid = getattr(simulation_result, "y_grid", None)
    player_cum_prob = np.asarray(getattr(simulation_result, "player_cum_prob", None), dtype=float)
    if x_grid is None or y_grid is None:
        return np.ones(player_cum_prob.shape[2:], dtype=bool)
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if x_grid.ndim != 3 or y_grid.ndim != 3:
        raise ValueError(f"simulation_result x_grid/y_grid must be 3D [frame, angle, distance], got {x_grid.shape}/{y_grid.shape}.")
    return (
        np.isfinite(x_grid[frame_index])
        & np.isfinite(y_grid[frame_index])
        & (x_grid[frame_index] >= -FIELD_SIZE[0] / 2.0)
        & (x_grid[frame_index] <= FIELD_SIZE[0] / 2.0)
        & (y_grid[frame_index] >= -FIELD_SIZE[1] / 2.0)
        & (y_grid[frame_index] <= FIELD_SIZE[1] / 2.0)
    )


def _update_as_default_maxima_from_simulation_result(
    *,
    maxima: dict[int, float],
    simulation_result: Any,
    candidate_indices: list[int],
    target_index_lookup: dict[int, tuple[int, int]],
    frame_count: int,
    player_count: int,
) -> None:
    player_cum_prob = np.asarray(getattr(simulation_result, "player_cum_prob", None), dtype=float)
    if player_cum_prob.ndim != 4:
        raise ValueError(
            "accessible-space simulation_result.player_cum_prob must be 4D "
            f"[frame, player, angle, distance], got shape={player_cum_prob.shape}."
        )
    if (
        player_cum_prob.shape[0] != frame_count
        or player_cum_prob.shape[1] != player_count
        or player_cum_prob.shape[2] != AS_DEFAULT_N_ANGLES
    ):
        raise ValueError(
            "Unexpected player_cum_prob shape for AS-default physical xPass: "
            f"{player_cum_prob.shape}, frames={frame_count}, players={player_count}, "
            f"n_angles={AS_DEFAULT_N_ANGLES}."
        )
    for graph_target_index in candidate_indices:
        frame_index, target_sim_index = target_index_lookup[graph_target_index]
        on_pitch = _on_pitch_mask(simulation_result, frame_index=frame_index)
        if on_pitch.shape != player_cum_prob.shape[2:]:
            raise ValueError(
                f"On-pitch mask shape {on_pitch.shape} does not match "
                f"player_cum_prob grid {player_cum_prob.shape[2:]}."
            )
        values = np.where(on_pitch, player_cum_prob[frame_index, target_sim_index, :, :], np.nan)
        if np.isfinite(values).any():
            maxima[graph_target_index] = max(maxima[graph_target_index], float(np.nanmax(values)))


def _as_default_simulation_kwargs(
    *,
    PLAYER_POS: np.ndarray,
    BALL_POS: np.ndarray,
    phi_grid: np.ndarray,
    v0_grid: np.ndarray,
    passer_teams: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    passers: np.ndarray,
    exclude_passer: bool,
    playing_direction: np.ndarray,
    use_progress_bar: bool,
    chunk_size: int,
    v0_prob_aggregation_mode: str,
) -> dict[str, Any]:
    return {
        "PLAYER_POS": PLAYER_POS,
        "BALL_POS": BALL_POS,
        "phi_grid": phi_grid,
        "v0_grid": v0_grid,
        "passer_teams": passer_teams,
        "player_teams": player_teams,
        "players": players,
        "passers": passers,
        "exclude_passer": bool(exclude_passer),
        "respect_offside": AS_DEFAULT_RESPECT_OFFSIDE,
        "playing_direction": playing_direction,
        "x_pitch_min": -FIELD_SIZE[0] / 2.0,
        "x_pitch_max": FIELD_SIZE[0] / 2.0,
        "y_pitch_min": -FIELD_SIZE[1] / 2.0,
        "y_pitch_max": FIELD_SIZE[1] / 2.0,
        "use_progress_bar": use_progress_bar,
        "chunk_size": chunk_size,
        "fields_to_return": ("player_cum_prob",),
        "normalize": AS_DEFAULT_NORMALIZE,
        "pass_start_location_offset": AS_DEFAULT_PASS_START_LOCATION_OFFSET,
        "time_offset_ball": AS_DEFAULT_TIME_OFFSET_BALL,
        "radial_gridsize": AS_DEFAULT_RADIAL_GRIDSIZE,
        "b0": AS_DEFAULT_B0,
        "b1": AS_DEFAULT_B1,
        "player_velocity": AS_DEFAULT_PLAYER_VELOCITY,
        "keep_inertial_velocity": AS_DEFAULT_KEEP_INERTIAL_VELOCITY,
        "use_max": AS_DEFAULT_USE_MAX,
        "v_max": AS_DEFAULT_V_MAX,
        "a_max": AS_DEFAULT_A_MAX,
        "inertial_seconds": AS_DEFAULT_INERTIAL_SECONDS,
        "tol_distance": AS_DEFAULT_TOL_DISTANCE,
        "use_approx_two_point": AS_DEFAULT_USE_APPROX_TWO_POINT,
        "v0_prob_aggregation_mode": v0_prob_aggregation_mode,
        "use_efficient_sigmoid": AS_DEFAULT_USE_EFFICIENT_SIGMOID,
        "factor": AS_DEFAULT_FACTOR,
        "factor2": AS_DEFAULT_FACTOR2,
    }


def _compute_as_default_max_from_simulation_inputs(
    *,
    node_ids: list[str],
    candidate_indices: list[int],
    player_pos: np.ndarray,
    player_teams: np.ndarray,
    players: np.ndarray,
    ball_pos: np.ndarray,
    target_index_lookup: dict[int, tuple[int, int]],
    possessor_index: int,
    playing_direction: float,
    exclude_passer: bool,
    eps: float,
    simulate_passes_fn: Callable[..., Any] | None,
    use_progress_bar: bool,
    chunk_size: int,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
) -> pd.Series:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    frame_count = int(player_pos.shape[0])
    PLAYER_POS = np.asarray(player_pos, dtype=float)
    BALL_POS = np.repeat(np.asarray(ball_pos, dtype=float)[np.newaxis, :], frame_count, axis=0)
    phi_grid = np.repeat(
        np.linspace(
            AS_DEFAULT_PHI_OFFSET,
            2.0 * np.pi + AS_DEFAULT_PHI_OFFSET,
            AS_DEFAULT_N_ANGLES,
            endpoint=False,
            dtype=float,
        )[np.newaxis, :],
        frame_count,
        axis=0,
    )
    v0_values = np.linspace(AS_DEFAULT_V0_MIN, AS_DEFAULT_V0_MAX, AS_DEFAULT_N_V0, dtype=float)
    passer_teams = np.repeat("attack", frame_count).astype(object)
    passers = np.repeat(node_ids[possessor_index], frame_count).astype(object)
    _validate_simulation_contract(players=players, passers=passers, exclude_passer=exclude_passer)

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    maxima = {node_index: -np.inf for node_index in candidate_indices}
    if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX:
        simulation_result = simulate_passes_fn(
            **_as_default_simulation_kwargs(
                PLAYER_POS=PLAYER_POS,
                BALL_POS=BALL_POS,
                phi_grid=phi_grid,
                v0_grid=np.repeat(v0_values[np.newaxis, :], frame_count, axis=0),
                passer_teams=passer_teams,
                player_teams=player_teams,
                players=players,
                passers=passers,
                exclude_passer=exclude_passer,
                playing_direction=np.repeat(float(playing_direction), frame_count),
                use_progress_bar=use_progress_bar,
                chunk_size=chunk_size,
                v0_prob_aggregation_mode="max",
            )
        )
        _update_as_default_maxima_from_simulation_result(
            maxima=maxima,
            simulation_result=simulation_result,
            candidate_indices=candidate_indices,
            target_index_lookup=target_index_lookup,
            frame_count=frame_count,
            player_count=len(players),
        )
    else:
        for speed in v0_values:
            simulation_result = simulate_passes_fn(
                **_as_default_simulation_kwargs(
                    PLAYER_POS=PLAYER_POS,
                    BALL_POS=BALL_POS,
                    phi_grid=phi_grid,
                    v0_grid=np.full((frame_count, 1), float(speed), dtype=float),
                    passer_teams=passer_teams,
                    player_teams=player_teams,
                    players=players,
                    passers=passers,
                    exclude_passer=exclude_passer,
                    playing_direction=np.repeat(float(playing_direction), frame_count),
                    use_progress_bar=use_progress_bar,
                    chunk_size=chunk_size,
                    v0_prob_aggregation_mode=AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
                )
            )
            _update_as_default_maxima_from_simulation_result(
                maxima=maxima,
                simulation_result=simulation_result,
                candidate_indices=candidate_indices,
                target_index_lookup=target_index_lookup,
                frame_count=frame_count,
                player_count=len(players),
            )

    for graph_target_index, value in maxima.items():
        if math.isfinite(value):
            result.loc[node_ids[graph_target_index]] = float(np.clip(value, float(eps), 1.0 - float(eps)))
    return result


def compute_graph_max_player_cum_prob_as_defaults(
    graph: Data,
    *,
    eps: float = 1e-4,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    node_ids = _node_ids(graph)
    candidate_indices = _candidate_target_indices(graph)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    if not candidate_indices:
        return result

    if consider_teammates:
        player_pos, player_teams, players, ball_pos, raw_target_lookup, possessor_index = _full_as_default_simulation_inputs(
            graph,
            candidate_indices=candidate_indices,
        )
        player_pos = player_pos[np.newaxis, :, :]
        target_index_lookup = {node_index: (0, sim_index) for node_index, sim_index in raw_target_lookup.items()}
        playing_direction = _infer_playing_direction_from_centered_players(player_pos[0], player_teams)
        exclude_passer = False
    else:
        (
            player_pos,
            player_teams,
            players,
            ball_pos,
            target_index_lookup,
            possessor_index,
            playing_direction,
        ) = _reduced_as_default_simulation_inputs(graph, candidate_indices=candidate_indices)
        exclude_passer = True

    return _compute_as_default_max_from_simulation_inputs(
        node_ids=node_ids,
        candidate_indices=candidate_indices,
        player_pos=player_pos,
        player_teams=player_teams,
        players=players,
        ball_pos=ball_pos,
        target_index_lookup=target_index_lookup,
        possessor_index=possessor_index,
        playing_direction=playing_direction,
        exclude_passer=exclude_passer,
        eps=eps,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
        speed_aggregation=speed_aggregation,
    )


def compute_graphs_max_player_cum_prob_as_defaults(
    graphs: list[Data],
    *,
    eps: float = 1e-4,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
    batch_size: int = 16,
) -> list[pd.Series]:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    if not consider_teammates:
        return [
            compute_graph_max_player_cum_prob_as_defaults(
                graph,
                eps=eps,
                consider_teammates=False,
                speed_aggregation=speed_aggregation,
                simulate_passes_fn=simulate_passes_fn,
                use_progress_bar=use_progress_bar,
                chunk_size=chunk_size,
            )
            for graph in graphs
        ]

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    batch_size = max(1, int(batch_size))
    results: list[pd.Series | None] = [None] * len(graphs)
    prepared_groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    for graph_index, graph in enumerate(graphs):
        node_ids = _node_ids(graph)
        candidate_indices = _candidate_target_indices(graph)
        result = pd.Series(np.nan, index=node_ids, dtype=float)
        if not candidate_indices:
            results[graph_index] = result
            continue
        player_pos, player_teams, players, ball_pos, raw_target_lookup, possessor_index = _full_as_default_simulation_inputs(
            graph,
            candidate_indices=candidate_indices,
        )
        key = (tuple(str(player) for player in players.tolist()), tuple(str(team) for team in player_teams.tolist()))
        prepared_groups.setdefault(key, []).append(
            {
                "graph_index": graph_index,
                "node_ids": node_ids,
                "candidate_indices": candidate_indices,
                "player_pos": player_pos,
                "player_teams": player_teams,
                "players": players,
                "ball_pos": ball_pos,
                "target_lookup": raw_target_lookup,
                "passer": node_ids[possessor_index],
                "playing_direction": _infer_playing_direction_from_centered_players(player_pos, player_teams),
            }
        )

    for group_frames in prepared_groups.values():
        for batch_start in range(0, len(group_frames), batch_size):
            batch = group_frames[batch_start : batch_start + batch_size]
            players = np.asarray(batch[0]["players"], dtype=object)
            player_teams = np.asarray(batch[0]["player_teams"], dtype=object)
            player_pos = np.stack([np.asarray(item["player_pos"], dtype=float) for item in batch], axis=0)
            ball_pos = np.stack([np.asarray(item["ball_pos"], dtype=float) for item in batch], axis=0)
            frame_count = len(batch)
            phi_grid = np.repeat(
                np.linspace(
                    AS_DEFAULT_PHI_OFFSET,
                    2.0 * np.pi + AS_DEFAULT_PHI_OFFSET,
                    AS_DEFAULT_N_ANGLES,
                    endpoint=False,
                    dtype=float,
                )[np.newaxis, :],
                frame_count,
                axis=0,
            )
            passers = np.asarray([item["passer"] for item in batch], dtype=object)
            passer_teams = np.repeat("attack", frame_count).astype(object)
            playing_direction = np.asarray([float(item["playing_direction"]) for item in batch], dtype=float)
            _validate_simulation_contract(players=players, passers=passers, exclude_passer=False)
            maxima: dict[int, float] = {}
            target_index_lookup: dict[int, tuple[int, int]] = {}
            target_keys: dict[int, tuple[int, int]] = {}
            next_target_key = 0
            for frame_index, item in enumerate(batch):
                for graph_target_index, sim_index in item["target_lookup"].items():
                    target_key = next_target_key
                    next_target_key += 1
                    maxima[target_key] = -np.inf
                    target_index_lookup[target_key] = (frame_index, int(sim_index))
                    target_keys[target_key] = (int(item["graph_index"]), int(graph_target_index))

            v0_values = np.linspace(AS_DEFAULT_V0_MIN, AS_DEFAULT_V0_MAX, AS_DEFAULT_N_V0, dtype=float)
            if speed_aggregation == PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX:
                simulation_results = [
                    simulate_passes_fn(
                        **_as_default_simulation_kwargs(
                            PLAYER_POS=player_pos,
                            BALL_POS=ball_pos,
                            phi_grid=phi_grid,
                            v0_grid=np.repeat(v0_values[np.newaxis, :], frame_count, axis=0),
                            passer_teams=passer_teams,
                            player_teams=player_teams,
                            players=players,
                            passers=passers,
                            exclude_passer=False,
                            playing_direction=playing_direction,
                            use_progress_bar=use_progress_bar,
                            chunk_size=chunk_size,
                            v0_prob_aggregation_mode="max",
                        )
                    )
                ]
            else:
                simulation_results = [
                    simulate_passes_fn(
                        **_as_default_simulation_kwargs(
                            PLAYER_POS=player_pos,
                            BALL_POS=ball_pos,
                            phi_grid=phi_grid,
                            v0_grid=np.full((frame_count, 1), float(speed), dtype=float),
                            passer_teams=passer_teams,
                            player_teams=player_teams,
                            players=players,
                            passers=passers,
                            exclude_passer=False,
                            playing_direction=playing_direction,
                            use_progress_bar=use_progress_bar,
                            chunk_size=chunk_size,
                            v0_prob_aggregation_mode=AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
                        )
                    )
                    for speed in v0_values
                ]

            for simulation_result in simulation_results:
                _update_as_default_maxima_from_simulation_result(
                    maxima=maxima,
                    simulation_result=simulation_result,
                    candidate_indices=list(maxima.keys()),
                    target_index_lookup=target_index_lookup,
                    frame_count=frame_count,
                    player_count=len(players),
                )

            for item in batch:
                graph_index = int(item["graph_index"])
                if results[graph_index] is None:
                    results[graph_index] = pd.Series(np.nan, index=item["node_ids"], dtype=float)
            for target_key, value in maxima.items():
                graph_index, graph_target_index = target_keys[target_key]
                if math.isfinite(value):
                    node_id = graphs[graph_index].node_ids[graph_target_index]
                    results[graph_index].loc[str(node_id)] = float(np.clip(value, float(eps), 1.0 - float(eps)))

    return [result if result is not None else pd.Series(dtype=float) for result in results]


def compute_graph_player_cum_prob(
    graph: Data,
    *,
    eps: float = 1e-4,
    normalize: bool = True,
    consider_teammates: bool = False,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    if not normalize:
        warnings.warn("--no-normalize is ignored for AS-default max physical xPass; normalize=True is required.")
    return compute_graph_max_player_cum_prob_as_defaults(
        graph,
        eps=eps,
        consider_teammates=consider_teammates,
        speed_aggregation=speed_aggregation,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )


def compute_graph_player_cum_prob_at_target_location(
    graph: Data,
    *,
    eps: float = 1e-4,
    normalize: bool = True,
    consider_teammates: bool = False,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    node_ids = _node_ids(graph)
    candidate_indices = _candidate_target_indices(graph)
    result = pd.Series(np.nan, index=node_ids, dtype=float)
    if not candidate_indices:
        return result

    x = graph.x.detach().cpu().to(torch.float32)
    target_xy = x[candidate_indices, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy()
    player_pos, player_teams, players, ball_pos, passer_teams, target_index_lookup = _build_simulation_inputs(
        graph,
        candidate_indices=candidate_indices,
        consider_teammates=consider_teammates,
    )
    deltas = target_xy - ball_pos[np.newaxis, :]
    distances = np.linalg.norm(deltas, axis=1)
    phis = np.arctan2(deltas[:, 1], deltas[:, 0])

    finite_targets = np.isfinite(distances) & (distances > 1e-6) & np.isfinite(phis)
    if not finite_targets.all():
        bad_ids = [node_ids[candidate_indices[i]] for i, ok in enumerate(finite_targets.tolist()) if not ok]
        raise ValueError(f"Physical xPass candidate targets have invalid distance/angle: {bad_ids}.")

    frame_count = len(candidate_indices)
    PLAYER_POS = np.asarray(player_pos, dtype=float)
    BALL_POS = np.repeat(ball_pos[np.newaxis, :], frame_count, axis=0)
    phi_grid = phis.reshape(frame_count, 1)
    v0_grid = np.repeat(
        np.linspace(DEFAULT_V0_MIN, DEFAULT_V0_MAX, DEFAULT_N_V0, dtype=float)[np.newaxis, :],
        frame_count,
        axis=0,
    )
    possessor_index = int(torch.nonzero(x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1, as_tuple=False).item())
    passers = np.repeat(node_ids[possessor_index], frame_count).astype(object)
    _validate_simulation_contract(players=players, passers=passers, exclude_passer=True)

    simulate_passes_fn = simulate_passes_fn or _resolve_simulate_passes_fn()
    simulation_result = simulate_passes_fn(
        PLAYER_POS=PLAYER_POS,
        BALL_POS=BALL_POS,
        phi_grid=phi_grid,
        v0_grid=v0_grid,
        passer_teams=passer_teams,
        player_teams=player_teams,
        players=players,
        passers=passers,
        exclude_passer=True,
        x_pitch_min=0.0,
        x_pitch_max=float(FIELD_SIZE[0]),
        y_pitch_min=0.0,
        y_pitch_max=float(FIELD_SIZE[1]),
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
        fields_to_return=("player_cum_prob",),
        normalize=normalize,
    )

    for frame_index, graph_target_index in enumerate(candidate_indices):
        target_sim_index = target_index_lookup[graph_target_index]
        value, _ = _extract_simulation_probability(
            simulation_result,
            frame_index=frame_index,
            target_player_index=target_sim_index,
            angle_index=0,
            target_distance=float(distances[frame_index]),
        )
        value = float(np.clip(value, float(eps), 1.0 - float(eps)))
        result.loc[node_ids[graph_target_index]] = value

    return result


def compute_graph_physical_xpass_for_source(
    graph: Data,
    *,
    source: str,
    eps: float = 1e-4,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> pd.Series:
    source = str(source)
    if source not in PHYSICAL_XPASS_SOURCES:
        raise ValueError(
            f"Unsupported physical_xpass_source={source!r}. "
            f"Expected one of {sorted(PHYSICAL_XPASS_SOURCES)}."
        )
    policy = teammate_policy or (
        PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE
        if source == PHYSICAL_XPASS_LEGACY_SOURCE
        else PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    )
    consider_teammates = policy == PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    if source == PHYSICAL_XPASS_LEGACY_SOURCE:
        return compute_graph_player_cum_prob_at_target_location(
            graph,
            eps=eps,
            normalize=True,
            consider_teammates=consider_teammates,
            simulate_passes_fn=simulate_passes_fn,
            use_progress_bar=use_progress_bar,
            chunk_size=chunk_size,
        )
    return compute_graph_max_player_cum_prob_as_defaults(
        graph,
        eps=eps,
        consider_teammates=consider_teammates,
        speed_aggregation=speed_aggregation,
        simulate_passes_fn=simulate_passes_fn,
        use_progress_bar=use_progress_bar,
        chunk_size=chunk_size,
    )


def load_physical_xpass_match(cache_dir: str | Path, match_id: str) -> pd.DataFrame:
    cache_root = Path(cache_dir)
    direct_path = cache_root / "matches" / f"{match_id}.parquet"
    path = direct_path if cache_root.name == "physical_xpass" or direct_path.exists() else get_physical_xpass_match_path(str(match_id), root=cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Physical xPass sidecar not found at {path}. "
            "Run scripts/generate_physical_xpass.py --feature-run-id <feature_run_id> before training with --use_physical_xpass."
        )
    frame = pd.read_parquet(path)
    if "action_index" not in frame.columns:
        raise ValueError(f"Physical xPass sidecar {path} is missing required column 'action_index'.")
    frame = frame.copy()
    frame["action_index"] = frame["action_index"].astype(int)
    if frame["action_index"].duplicated().any():
        duplicates = frame.loc[frame["action_index"].duplicated(), "action_index"].head(5).tolist()
        raise ValueError(f"Physical xPass sidecar {path} contains duplicate action_index rows, e.g. {duplicates}.")
    return frame.set_index("action_index", drop=False)


def load_physical_xpass_component(
    cache_dir: str | Path,
    match_id: str,
    action_index: int,
) -> pd.Series:
    frame = load_physical_xpass_match(cache_dir, match_id)
    if int(action_index) not in frame.index:
        raise KeyError(f"Physical xPass sidecar has no row for match_id={match_id}, action_index={int(action_index)}.")
    row = frame.loc[int(action_index)]
    player_columns = [column for column in row.index if column not in PHYSICAL_XPASS_ID_COLUMNS]
    series = pd.to_numeric(row[player_columns], errors="coerce").astype(float)
    series.name = "max_player_cum_prob"
    return series


def attach_physical_xpass_to_graph(
    graph: Data,
    labels: torch.Tensor,
    physical_rows: pd.DataFrame,
    *,
    match_id: str,
    eps: float = 1e-4,
    require_observed_target: bool = True,
) -> Data:
    if physical_rows is None:
        raise ValueError("physical_rows must be provided when attaching physical xPass.")
    action_index = int(labels[LABEL_INDEX["action_index"]].item())
    if action_index not in physical_rows.index:
        raise FileNotFoundError(
            f"Physical xPass sidecar for match {match_id} has no row for action_index={action_index}. "
            "Run scripts/generate_physical_xpass.py for the full feature run."
        )

    row = physical_rows.loc[action_index]
    node_ids = _node_ids(graph)
    probs = torch.full((len(node_ids),), PHYSICAL_XPASS_NEUTRAL_PROB, dtype=torch.float32)
    missing_columns = []
    for node_index, node_id in enumerate(node_ids):
        if node_id not in row.index:
            missing_columns.append(node_id)
            continue
        value = row[node_id]
        if pd.isna(value):
            continue
        probs[node_index] = float(value)

    if missing_columns and require_observed_target:
        target_index = int(labels[LABEL_INDEX["intent_index"]].item())
        if 0 <= target_index < len(node_ids) and node_ids[target_index] in missing_columns:
            raise ValueError(
                f"Physical xPass sidecar for match {match_id}, action_index={action_index} is missing "
                f"observed target player column {node_ids[target_index]!r}."
            )

    probs = torch.clamp(probs, float(eps), 1.0 - float(eps))
    logits = probability_to_logit(probs, eps=eps)
    if not torch.isfinite(probs).all() or not torch.isfinite(logits).all():
        raise ValueError(f"Physical xPass values are non-finite after clipping for match {match_id}, action_index={action_index}.")

    if require_observed_target:
        target_index = int(labels[LABEL_INDEX["intent_index"]].item())
        if 0 <= target_index < len(node_ids):
            target_value = row[node_ids[target_index]] if node_ids[target_index] in row.index else np.nan
            if pd.isna(target_value):
                raise ValueError(
                    f"Physical xPass sidecar has no finite probability for observed target {node_ids[target_index]!r} "
                    f"in match {match_id}, action_index={action_index}."
                )

    setattr(graph, PHYSICAL_XPASS_PROB_ATTR, probs)
    setattr(graph, PHYSICAL_XPASS_LOGIT_ATTR, logits)
    return graph


def attach_physical_xpass_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    cache_dir: str | Path,
    match_id: str,
    eps: float = 1e-4,
    require_observed_target: bool = True,
) -> list[Data]:
    rows = load_physical_xpass_match(cache_dir, match_id)
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                rows,
                match_id=match_id,
                eps=eps,
                require_observed_target=require_observed_target,
            )
        )
    return attached


def attach_physical_xpass_online_to_graphs(
    graphs: list[Data],
    labels: torch.Tensor,
    *,
    source: str,
    eps: float = 1e-4,
    teammate_policy: str | None = None,
    speed_aggregation: str | None = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    require_observed_target: bool = False,
    simulate_passes_fn: Callable[..., Any] | None = None,
    use_progress_bar: bool = False,
    chunk_size: int = 150,
) -> list[Data]:
    attached: list[Data] = []
    for graph, label in zip(graphs, labels):
        action_index = int(label[LABEL_INDEX["action_index"]].item())
        probs = compute_graph_physical_xpass_for_source(
            graph,
            source=source,
            eps=eps,
            teammate_policy=teammate_policy,
            speed_aggregation=speed_aggregation,
            simulate_passes_fn=simulate_passes_fn,
            use_progress_bar=use_progress_bar,
            chunk_size=chunk_size,
        )
        row = {"action_index": action_index}
        row.update({str(player_id): float(value) for player_id, value in probs.items()})
        physical_rows = pd.DataFrame([row]).set_index("action_index", drop=False)
        attached.append(
            attach_physical_xpass_to_graph(
                graph,
                label,
                physical_rows,
                match_id="<runtime>",
                eps=eps,
                require_observed_target=require_observed_target,
            )
        )
    return attached
