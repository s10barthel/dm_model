"""Prototype sigmoid-arrival lane survival and receiver control xPass.

This diagnostic is intentionally separate from production code. It evaluates a
single Hawkeye frame on a speed/angle/radial grid and prints the max score per
candidate teammate.
"""

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

from scripts.diagnostics import compare_hybrid_control_windows as chw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation-id", default=chw.DEFAULT_SITUATION_ID)
    parser.add_argument("--time-norm", type=float, default=0.0)
    parser.add_argument("--tracking-csv", default=str(ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--ball-csv", default=str(ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    parser.add_argument("--max-speed", type=float, default=25.0)
    parser.add_argument("--speed-step", type=float, default=2.0)
    parser.add_argument("--angle-step", type=float, default=2.5)
    parser.add_argument("--sigmoid-scale", type=float, default=15.0)
    parser.add_argument("--sigmoid-offset", type=float, default=0.3)
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def arrival_raw_control(ball_minus_player: np.ndarray, *, scale: float, offset: float) -> np.ndarray:
    """Raw control score f(x)=1/(1+exp(-scale*(x+offset)))."""
    return sigmoid(float(scale) * (ball_minus_player + float(offset)))


def normalize_if_sum_above_one(raw: np.ndarray, axis: int = 0) -> np.ndarray:
    sums = np.nansum(raw, axis=axis, keepdims=True)
    normalized = np.divide(raw, sums, out=np.zeros_like(raw), where=sums > 0)
    return np.where(sums > 1.0, normalized, raw)


def player_arrival_margins(setup, target_x: np.ndarray, target_y: np.ndarray, t_ball: np.ndarray) -> np.ndarray:
    player_pos = setup["player_pos"][0]
    margins = []
    for index in range(player_pos.shape[0]):
        tta = chw.time_to_arrive(player_pos[index], target_x, target_y)
        margins.append(t_ball - tta)
    return np.stack(margins, axis=0)


def main() -> None:
    args = parse_args()
    graph, selection = chw.build_graph(args)
    setup = chw.make_setup(graph)
    speeds = chw.ppm.as_default_v0_values(max_speed=args.max_speed, speed_step=args.speed_step)
    angles = np.deg2rad(np.arange(0.0, 360.0, float(args.angle_step), dtype=float))

    # One cheap accessible-space call gives us the radial grid and on-pitch mask.
    normal_results = chw.simulate(setup, angles, speeds[:1], ("cum_p0",))
    cum_p0, r_grid = chw.stack_cum_p0(normal_results)
    on_pitch = np.isfinite(cum_p0[0])

    target_x_base = setup["raw_ball_pos"][0] + np.cos(angles)[:, np.newaxis] * r_grid[np.newaxis, :]
    target_y_base = setup["raw_ball_pos"][1] + np.sin(angles)[:, np.newaxis] * r_grid[np.newaxis, :]
    target_x = np.repeat(target_x_base[np.newaxis, :, :], len(speeds), axis=0)
    target_y = np.repeat(target_y_base[np.newaxis, :, :], len(speeds), axis=0)
    t_ball = np.repeat((r_grid[np.newaxis, np.newaxis, :] / speeds[:, np.newaxis, np.newaxis]), len(angles), axis=1)
    on_pitch_grid = np.repeat(on_pitch[np.newaxis, :, :], len(speeds), axis=0)

    margins = player_arrival_margins(setup, target_x, target_y, t_ball)
    raw = arrival_raw_control(margins, scale=args.sigmoid_scale, offset=args.sigmoid_offset)
    raw[:, :, ~on_pitch] = np.nan

    node_ids = [str(value) for value in setup["node_ids"]]
    players = [str(value) for value in setup["players"]]
    possessor_player = str(setup["passers"][0])
    passer_sim_index = players.index(possessor_player)
    graph_x = graph.x.detach().cpu().numpy()
    positions = {
        node_id: (
            float(graph_x[index, chw.config.NODE_FEATURE_X]),
            float(graph_x[index, chw.config.NODE_FEATURE_Y]),
        )
        for index, node_id in enumerate(node_ids)
    }

    rows = []
    for graph_target_index in setup["candidate_indices"]:
        receiver = str(setup["node_ids"][graph_target_index])
        _, receiver_sim_index = setup["target_lookup"][graph_target_index]

        endpoint_raw = raw.copy()
        endpoint_raw[passer_sim_index] = 0.0
        endpoint_probs = normalize_if_sum_above_one(endpoint_raw, axis=0)
        receiver_control = endpoint_probs[receiver_sim_index]

        lane_raw = raw.copy()
        lane_raw[passer_sim_index] = 0.0
        lane_raw[receiver_sim_index] = 0.0
        lane_probs = normalize_if_sum_above_one(lane_raw, axis=0)
        other_control = np.nansum(lane_probs, axis=0)
        per_location_survival = np.clip(1.0 - other_control, 0.0, 1.0)

        lane_survival = np.ones_like(receiver_control, dtype=float)
        for distance_i in range(len(r_grid)):
            if distance_i == 0:
                lane_survival[:, :, distance_i] = 1.0
                continue
            previous = per_location_survival[:, :, :distance_i]
            lane_survival[:, :, distance_i] = np.prod(previous, axis=2)
        lane_survival = np.where(on_pitch_grid, lane_survival, np.nan)

        score = lane_survival * receiver_control
        if not np.isfinite(score).any():
            continue
        flat = int(np.nanargmax(score))
        speed_i, angle_i, distance_i = np.unravel_index(flat, score.shape)
        x, y = positions[receiver]
        rows.append({
            "player_id": receiver,
            "x": x,
            "y": y,
            "score": float(score[speed_i, angle_i, distance_i]),
            "lane_survival": float(lane_survival[speed_i, angle_i, distance_i]),
            "control_prob": float(receiver_control[speed_i, angle_i, distance_i]),
            "speed": float(speeds[speed_i]),
            "angle": math.degrees(float(angles[angle_i])) % 360.0,
            "distance": float(r_grid[distance_i]),
            "target_x": float(target_x[speed_i, angle_i, distance_i]),
            "target_y": float(target_y[speed_i, angle_i, distance_i]),
            "ball_minus_player": float(margins[receiver_sim_index, speed_i, angle_i, distance_i]),
        })

    table = pd.DataFrame(rows).sort_values("x")
    print("selection", selection)
    print("metric", {
        "raw_control": "1 / (1 + exp(-15 * (ball_minus_player + 0.3)))",
        "sigmoid_scale": args.sigmoid_scale,
        "sigmoid_offset": args.sigmoid_offset,
        "normalization": "raw / sum(raw) only where sum(raw) > 1",
        "lane_survival": "product over prior radial cells of 1 - normalized non-passer/non-receiver control",
        "receiver_control": "normalized receiver control at target cell, passer excluded",
        "speeds": f"{float(speeds[0])}..{float(speeds[-1])} step {args.speed_step}",
        "angle_step": args.angle_step,
    })
    print(table.to_string(index=False, formatters={
        "x": "{:.2f}".format,
        "y": "{:.2f}".format,
        "score": "{:.6f}".format,
        "lane_survival": "{:.6f}".format,
        "control_prob": "{:.6f}".format,
        "speed": "{:.1f}".format,
        "angle": "{:.2f}".format,
        "distance": "{:.2f}".format,
        "target_x": "{:.2f}".format,
        "target_y": "{:.2f}".format,
        "ball_minus_player": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()
