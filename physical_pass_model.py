from __future__ import annotations

import math
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

PHYSICAL_XPASS_SOURCE = "accessible_space_player_cum_prob"
PHYSICAL_XPASS_NEUTRAL_PROB = 0.5
PHYSICAL_XPASS_LOGIT_ATTR = "physical_xpass_logit"
PHYSICAL_XPASS_PROB_ATTR = "physical_xpass"
PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE = "ignore_teammates"
PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER = "consider_teammates"
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
PHYSICAL_XPASS_ID_COLUMNS = {"match_id", "action_index", "action_id"}
DEFAULT_V0_MIN = 8.886015553615485
DEFAULT_V0_MAX = 42.18118275402132
DEFAULT_N_V0 = 14


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
    frame_player_indices = [[graph_target_index] + defender_indices for graph_target_index in candidate_indices]
    player_counts = {len(indices) for indices in frame_player_indices}
    if len(player_counts) != 1:
        raise ValueError(f"Physical xPass reduced simulation player count must be constant across candidates, got {sorted(player_counts)}.")

    player_pos = np.stack(
        [x[indices, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1].numpy() for indices in frame_player_indices],
        axis=0,
    )
    player_teams = np.array(["attack"] + ["defense"] * len(defender_indices), dtype=object)
    players = np.array(
        ["target_player"] + [str(graph.node_ids[idx]) for idx in defender_indices],
        dtype=object,
    )
    target_index_lookup = {graph_target_index: 0 for graph_target_index in candidate_indices}
    return (
        player_pos,
        player_teams,
        players,
        x[possessor_index, config.NODE_FEATURE_X : config.NODE_FEATURE_Y + 1].numpy(),
        np.repeat("attack", len(candidate_indices)).astype(object),
        target_index_lookup,
    )


def compute_graph_player_cum_prob(
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
    series.name = "player_cum_prob"
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
