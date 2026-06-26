"""Compare hybrid xPass receiver-control integration windows on one Hawkeye frame."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from datatools.hawkeye import (
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    load_hawkeye_ball,
    load_hawkeye_tracking,
)
from scripts.visualize_hawkeye import resolve_ballreceipt, resolve_hawkeye_png_frames
import datatools.config as config
import physical_pass_model as ppm


DEFAULT_SITUATION_ID = "a97f38e5-f4fd-4174-9b48-1436832ff654"
WINDOWS = (0.5, 1.0, 2.0, 10.0)

PC_MAX_PLAYER_SPEED = 5.0
PC_REACTION_TIME = 0.7
PC_TTI_SIGMA = 0.45
PC_LAMBDA_RECEIVER = 4.3
PC_LAMBDA_OPPONENT = 4.3
PC_DT = 0.04


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation-id", default=DEFAULT_SITUATION_ID)
    parser.add_argument("--time-norm", type=float, default=0.0)
    parser.add_argument("--tracking-csv", default=str(ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--ball-csv", default=str(ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    return parser.parse_args()


def build_graph(args: argparse.Namespace):
    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    situation_tracking = tracking.loc[tracking["id"].astype(str) == str(args.situation_id)].copy()
    situation, _, _ = build_hawkeye_situation(situation_tracking, ball)
    selection = resolve_hawkeye_png_frames(
        situation,
        resolve_ballreceipt(situation_tracking),
        [float(args.time_norm)],
    )[0]
    frame_id = int(selection["frame_id"])
    frame_ids = [int(value) for value in situation.actions.index.tolist()]
    return situation.graph_features_0[frame_ids.index(frame_id)], selection


def make_setup(graph):
    candidate_indices = ppm._candidate_target_indices(graph)
    node_ids = ppm._node_ids(graph)
    player_pos, player_teams, players, ball_pos, raw_lookup, possessor_index = ppm._full_as_default_simulation_inputs(
        graph,
        candidate_indices=candidate_indices,
    )
    player_pos = player_pos[np.newaxis, :, :]
    return {
        "candidate_indices": candidate_indices,
        "node_ids": node_ids,
        "player_pos": player_pos,
        "player_teams": player_teams,
        "players": players,
        "ball_pos": np.repeat(np.asarray(ball_pos, dtype=float)[np.newaxis, :], 1, axis=0),
        "raw_ball_pos": np.asarray(ball_pos, dtype=float),
        "target_lookup": {node_index: (0, int(sim_index)) for node_index, sim_index in raw_lookup.items()},
        "passers": np.asarray([node_ids[possessor_index]], dtype=object),
        "playing_direction": np.asarray(
            [ppm._infer_playing_direction_from_centered_players(player_pos[0], player_teams)],
            dtype=float,
        ),
        "exclude_passer": True,
        "frame_count": 1,
    }


def simulate(setup, angles, speeds, fields, player_pos_override=None):
    sim_fn = ppm._resolve_simulate_passes_fn()
    phi_grid = np.repeat(np.asarray(angles, dtype=float)[np.newaxis, :], setup["frame_count"], axis=0)
    passer_teams = np.repeat("attack", setup["frame_count"]).astype(object)
    player_pos = setup["player_pos"] if player_pos_override is None else player_pos_override
    results = []
    for speed in speeds:
        kwargs = ppm._as_default_simulation_kwargs(
            PLAYER_POS=player_pos,
            BALL_POS=setup["ball_pos"],
            phi_grid=phi_grid,
            v0_grid=np.full((setup["frame_count"], 1), float(speed), dtype=float),
            passer_teams=passer_teams,
            player_teams=setup["player_teams"],
            players=setup["players"],
            passers=setup["passers"],
            exclude_passer=setup["exclude_passer"],
            playing_direction=setup["playing_direction"],
            use_progress_bar=False,
            chunk_size=150,
            v0_prob_aggregation_mode=ppm.AS_DEFAULT_V0_PROB_AGGREGATION_MODE,
        )
        kwargs["fields_to_return"] = fields
        results.append(sim_fn(**kwargs))
    return results


def stack_cum_p0(results):
    values_by_speed = []
    r_grid = None
    for result in results:
        arr = np.asarray(result.cum_p0, dtype=float)
        on_pitch = ppm._on_pitch_mask(result, frame_index=0)
        values = np.where(on_pitch, arr[0, :, :], np.nan)
        values_by_speed.append(values)
        if r_grid is None:
            r_grid = ppm._simulation_r_grid(result, values.shape[1])
    return np.stack(values_by_speed, axis=0), r_grid


def stack_player_cum_prob(results, setup, target_sim_index):
    values_by_speed = []
    r_grid = None
    for result in results:
        arr = np.asarray(result.player_cum_prob, dtype=float)
        on_pitch = ppm._on_pitch_mask(result, frame_index=0)
        values = np.where(on_pitch, arr[0, target_sim_index, :, :], np.nan)
        values_by_speed.append(values)
        if r_grid is None:
            r_grid = ppm._simulation_r_grid(result, values.shape[1])
    return np.stack(values_by_speed, axis=0), r_grid


def time_to_arrive(player_xyv, target_x, target_y):
    x, y, vx, vy = [float(value) for value in player_xyv]
    inertial_x = x + vx * PC_REACTION_TIME
    inertial_y = y + vy * PC_REACTION_TIME
    return PC_REACTION_TIME + np.hypot(target_x - inertial_x, target_y - inertial_y) / PC_MAX_PLAYER_SPEED


def time_to_arrive_with_params(player_xyv, target_x, target_y, *, reaction_time: float, max_player_speed: float):
    x, y, vx, vy = [float(value) for value in player_xyv]
    inertial_x = x + vx * float(reaction_time)
    inertial_y = y + vy * float(reaction_time)
    return float(reaction_time) + np.hypot(target_x - inertial_x, target_y - inertial_y) / float(max_player_speed)


def pitch_control_receiver_by_window(setup, target_sim_index, target_x, target_y, t_ball):
    player_pos = setup["player_pos"][0]
    player_teams = setup["player_teams"]
    receiver_tta = time_to_arrive(player_pos[target_sim_index], target_x, target_y).reshape(-1)
    opponent_indices = np.flatnonzero(player_teams == "defense").tolist()
    opponent_ttas = np.stack(
        [time_to_arrive(player_pos[index], target_x, target_y).reshape(-1) for index in opponent_indices],
        axis=1,
    )
    t_ball_flat = t_ball.reshape(-1)
    p_receiver = np.zeros(int(t_ball_flat.size), dtype=float)
    p_total = np.zeros_like(p_receiver)
    outputs: dict[float, np.ndarray] = {}
    k = math.pi / math.sqrt(3.0) / PC_TTI_SIGMA
    max_window = max(WINDOWS)
    steps = int(math.ceil(max_window / PC_DT))
    pending = list(WINDOWS)
    for step in range(steps):
        elapsed = float((step + 1) * PC_DT)
        t = t_ball_flat + elapsed
        receiver_rate = PC_LAMBDA_RECEIVER * sigmoid(k * (t - receiver_tta))
        opponent_rate = PC_LAMBDA_OPPONENT * sigmoid(k * (t[:, np.newaxis] - opponent_ttas)).sum(axis=1)
        uncontrolled = np.maximum(0.0, 1.0 - p_total)
        p_receiver += uncontrolled * receiver_rate * PC_DT
        p_total += uncontrolled * (receiver_rate + opponent_rate) * PC_DT
        p_total = np.minimum(p_total, 1.0)
        while pending and elapsed + 1e-12 >= pending[0]:
            outputs[pending.pop(0)] = np.clip(p_receiver.reshape(t_ball.shape), 0.0, 1.0).copy()
    return outputs


def main() -> None:
    args = parse_args()
    graph, selection = build_graph(args)
    setup = make_setup(graph)
    speeds = ppm.as_default_v0_values(max_speed=25, speed_step=2)
    angles = np.deg2rad(np.arange(0.0, 360.0, 2.5, dtype=float))
    print("selection", selection)
    print("params", {
        "max_player_speed": PC_MAX_PLAYER_SPEED,
        "reaction_time": PC_REACTION_TIME,
        "tti_sigma": PC_TTI_SIGMA,
        "lambda_receiver": PC_LAMBDA_RECEIVER,
        "lambda_opponent": PC_LAMBDA_OPPONENT,
        "dt": PC_DT,
    })
    normal_results = simulate(setup, angles, speeds, ("player_cum_prob",))
    _, r_grid = stack_player_cum_prob(normal_results, setup, setup["target_lookup"][setup["candidate_indices"][0]][1])
    target_x = setup["raw_ball_pos"][0] + np.cos(angles)[np.newaxis, :, np.newaxis] * r_grid[np.newaxis, np.newaxis, :]
    target_y = setup["raw_ball_pos"][1] + np.sin(angles)[np.newaxis, :, np.newaxis] * r_grid[np.newaxis, np.newaxis, :]
    target_x = np.repeat(target_x, len(speeds), axis=0)
    target_y = np.repeat(target_y, len(speeds), axis=0)
    t_ball = np.repeat((r_grid[np.newaxis, np.newaxis, :] / speeds[:, np.newaxis, np.newaxis]), len(angles), axis=1)

    graph_node_ids = ppm._node_ids(graph)
    graph_x = graph.x.detach().cpu().numpy()
    positions = {
        str(player_id): (
            float(graph_x[i, config.NODE_FEATURE_X]),
            float(graph_x[i, config.NODE_FEATURE_Y]),
        )
        for i, player_id in enumerate(graph_node_ids)
    }
    rows = []
    detail_rows = []
    for graph_target_index in setup["candidate_indices"]:
        node_id = str(setup["node_ids"][graph_target_index])
        print("receiver", node_id, flush=True)
        frame_index, target_sim_index = setup["target_lookup"][graph_target_index]
        ignored_pos = np.asarray(setup["player_pos"], dtype=float).copy()
        ignored_pos[frame_index, target_sim_index, :] = np.nan
        lane_results = simulate(setup, angles, speeds, ("cum_p0",), player_pos_override=ignored_pos)
        lane_survival, _ = stack_cum_p0(lane_results)
        receiver_controls = pitch_control_receiver_by_window(setup, target_sim_index, target_x, target_y, t_ball)
        x, y = positions[node_id]
        row = {"player_id": node_id, "x": x, "y": y}
        for window in WINDOWS:
            score_grid = lane_survival * receiver_controls[window]
            flat = int(np.nanargmax(np.where(np.isfinite(score_grid), score_grid, np.nan)))
            speed_i, angle_i, distance_i = np.unravel_index(flat, score_grid.shape)
            label = str(window).replace(".", "p")
            row[f"score_{label}"] = float(score_grid[speed_i, angle_i, distance_i])
            detail_rows.append({
                "player_id": node_id,
                "window": window,
                "score": float(score_grid[speed_i, angle_i, distance_i]),
                "lane_survival": float(lane_survival[speed_i, angle_i, distance_i]),
                "receiver_control": float(receiver_controls[window][speed_i, angle_i, distance_i]),
                "best_speed": float(speeds[speed_i]),
                "best_angle": math.degrees(float(angles[angle_i])) % 360.0,
                "best_distance": float(r_grid[distance_i]),
            })
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("x")
    details = pd.DataFrame(detail_rows)
    print("\nSCORES")
    print(table.to_string(index=False, formatters={
        "x": "{:.2f}".format,
        "y": "{:.2f}".format,
        "score_0p5": "{:.6f}".format,
        "score_1p0": "{:.6f}".format,
        "score_2p0": "{:.6f}".format,
        "score_10p0": "{:.6f}".format,
    }))
    print("\nBEST DETAILS")
    print(details.to_string(index=False, formatters={
        "window": "{:.1f}".format,
        "score": "{:.6f}".format,
        "lane_survival": "{:.6f}".format,
        "receiver_control": "{:.6f}".format,
        "best_speed": "{:.1f}".format,
        "best_angle": "{:.2f}".format,
        "best_distance": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
