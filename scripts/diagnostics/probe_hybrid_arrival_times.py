"""Print arrival-time diagnostics for the experimental hybrid xPass metric.

This script is intentionally diagnostic, not part of the production pipeline.
It reuses the one-frame Hawkeye setup from compare_hybrid_control_windows.py and
prints player arrival margins at the 1.0s-window optimal pass locations.
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


DEFAULT_RECEIVERS = ("away_107232", "away_250012106")
DEFAULT_WINDOW = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation-id", default=chw.DEFAULT_SITUATION_ID)
    parser.add_argument("--time-norm", type=float, default=0.0)
    parser.add_argument("--receiver", action="append", default=None)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--selection-mode",
        choices=("score", "receiver-arrival-margin"),
        default="score",
        help="Choose the optimal cell by hybrid score or by receiver T_ball - TTA.",
    )
    parser.add_argument("--speed", type=float, default=None, help="Use an explicit pass speed from the grid.")
    parser.add_argument("--angle-deg", type=float, default=None, help="Use an explicit pass angle in degrees from the grid.")
    parser.add_argument("--distance", type=float, default=None, help="Use an explicit pass distance from the radial grid.")
    parser.add_argument("--tracking-csv", default=str(ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--ball-csv", default=str(ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    return parser.parse_args()


def receiver_arrival_margin_grid(setup, target_sim_index: int, target_x: np.ndarray, target_y: np.ndarray, t_ball: np.ndarray) -> np.ndarray:
    player_pos = setup["player_pos"][0, target_sim_index]
    tta = chw.time_to_arrive(player_pos, target_x, target_y)
    return t_ball - tta


def pitch_control_all_players_for_cell(setup, target_x: float, target_y: float, t_ball: float, window: float) -> np.ndarray:
    player_pos = setup["player_pos"][0]
    player_teams = setup["player_teams"]
    ttas = np.asarray(
        [chw.time_to_arrive(player_pos[index], target_x, target_y) for index in range(player_pos.shape[0])],
        dtype=float,
    )
    p_players = np.zeros(player_pos.shape[0], dtype=float)
    p_total = 0.0
    k = math.pi / math.sqrt(3.0) / chw.PC_TTI_SIGMA
    steps = int(math.ceil(window / chw.PC_DT))
    for step in range(steps):
        elapsed = float((step + 1) * chw.PC_DT)
        t = float(t_ball) + elapsed
        rates = np.zeros_like(ttas)
        attack_mask = player_teams == "attack"
        defense_mask = player_teams == "defense"
        rates[attack_mask] = chw.PC_LAMBDA_RECEIVER * chw.sigmoid(k * (t - ttas[attack_mask]))
        rates[defense_mask] = chw.PC_LAMBDA_OPPONENT * chw.sigmoid(k * (t - ttas[defense_mask]))
        uncontrolled = max(0.0, 1.0 - p_total)
        delta = uncontrolled * rates * chw.PC_DT
        p_players += delta
        p_total = min(1.0, p_total + float(delta.sum()))
    return np.clip(p_players, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    receivers = args.receiver or list(DEFAULT_RECEIVERS)
    graph, selection = chw.build_graph(args)
    setup = chw.make_setup(graph)
    speeds = chw.ppm.as_default_v0_values(max_speed=25, speed_step=2)
    angles = np.deg2rad(np.arange(0.0, 360.0, 2.5, dtype=float))

    normal_results = chw.simulate(setup, angles, speeds, ("player_cum_prob",))
    _, r_grid = chw.stack_player_cum_prob(
        normal_results,
        setup,
        setup["target_lookup"][setup["candidate_indices"][0]][1],
    )
    target_x = setup["raw_ball_pos"][0] + np.cos(angles)[np.newaxis, :, np.newaxis] * r_grid[np.newaxis, np.newaxis, :]
    target_y = setup["raw_ball_pos"][1] + np.sin(angles)[np.newaxis, :, np.newaxis] * r_grid[np.newaxis, np.newaxis, :]
    target_x = np.repeat(target_x, len(speeds), axis=0)
    target_y = np.repeat(target_y, len(speeds), axis=0)
    t_ball = np.repeat((r_grid[np.newaxis, np.newaxis, :] / speeds[:, np.newaxis, np.newaxis]), len(angles), axis=1)

    print("selection", selection)
    print("window", args.window)
    print("pitch_control_params", {
        "max_player_speed": chw.PC_MAX_PLAYER_SPEED,
        "reaction_time": chw.PC_REACTION_TIME,
        "tti_sigma": chw.PC_TTI_SIGMA,
        "lambda_receiver": chw.PC_LAMBDA_RECEIVER,
        "lambda_opponent": chw.PC_LAMBDA_OPPONENT,
        "dt": chw.PC_DT,
    })

    for receiver in receivers:
        graph_target_index = setup["node_ids"].index(receiver)
        frame_index, target_sim_index = setup["target_lookup"][graph_target_index]
        ignored_pos = np.asarray(setup["player_pos"], dtype=float).copy()
        ignored_pos[frame_index, target_sim_index, :] = np.nan
        lane_results = chw.simulate(setup, angles, speeds, ("cum_p0",), player_pos_override=ignored_pos)
        lane_survival, _ = chw.stack_cum_p0(lane_results)
        receiver_controls = chw.pitch_control_receiver_by_window(setup, target_sim_index, target_x, target_y, t_ball)
        if args.window not in receiver_controls:
            raise ValueError(f"window {args.window} is not available; use one of {sorted(receiver_controls)}")
        score_grid = lane_survival * receiver_controls[args.window]
        if args.speed is not None or args.angle_deg is not None or args.distance is not None:
            if args.speed is None or args.angle_deg is None or args.distance is None:
                raise ValueError("--speed, --angle-deg, and --distance must be provided together.")
            speed_i = int(np.argmin(np.abs(speeds - float(args.speed))))
            angle_i = int(np.argmin(np.abs(np.rad2deg(angles) - (float(args.angle_deg) % 360.0))))
            distance_i = int(np.argmin(np.abs(r_grid - float(args.distance))))
        else:
            if args.selection_mode == "receiver-arrival-margin":
                selection_grid = receiver_arrival_margin_grid(setup, target_sim_index, target_x, target_y, t_ball)
                selection_grid = np.where(np.isfinite(lane_survival), selection_grid, np.nan)
            else:
                selection_grid = score_grid
            flat = int(np.nanargmax(np.where(np.isfinite(selection_grid), selection_grid, np.nan)))
            speed_i, angle_i, distance_i = np.unravel_index(flat, score_grid.shape)
        best_x = float(target_x[speed_i, angle_i, distance_i])
        best_y = float(target_y[speed_i, angle_i, distance_i])
        best_t_ball = float(t_ball[speed_i, angle_i, distance_i])
        p_players = pitch_control_all_players_for_cell(setup, best_x, best_y, best_t_ball, float(args.window))

        rows = []
        for index, player_id in enumerate(setup["players"]):
            x, y, _, _ = setup["player_pos"][0, index]
            tta = float(chw.time_to_arrive(setup["player_pos"][0, index], best_x, best_y))
            margin = best_t_ball - tta
            rows.append({
                "player_id": str(player_id),
                "team": str(setup["player_teams"][index]),
                "x": float(x),
                "y": float(y),
                "t_ball": best_t_ball,
                "tta_player": tta,
                "ball_minus_player": margin,
                "control_prob": float(p_players[index]),
            })
        table = pd.DataFrame(rows)
        table = table.sort_values(["ball_minus_player", "control_prob"], ascending=[False, False])
        relevant = table.loc[
            (table["control_prob"] > 1e-4)
            | (table["ball_minus_player"] > -2.0)
            | (table["player_id"] == receiver)
        ].copy()

        print("\nreceiver", receiver)
        print("best", {
            "selection_mode": args.selection_mode,
            "score": float(score_grid[speed_i, angle_i, distance_i]),
            "lane_survival": float(lane_survival[speed_i, angle_i, distance_i]),
            "receiver_control": float(receiver_controls[args.window][speed_i, angle_i, distance_i]),
            "receiver_ball_minus_player": float(
                receiver_arrival_margin_grid(setup, target_sim_index, target_x, target_y, t_ball)[speed_i, angle_i, distance_i]
            ),
            "speed": float(speeds[speed_i]),
            "angle": math.degrees(float(angles[angle_i])) % 360.0,
            "distance": float(r_grid[distance_i]),
            "target_x": best_x,
            "target_y": best_y,
            "t_ball": best_t_ball,
        })
        print(relevant.to_string(index=False, formatters={
            "x": "{:.2f}".format,
            "y": "{:.2f}".format,
            "t_ball": "{:.3f}".format,
            "tta_player": "{:.3f}".format,
            "ball_minus_player": "{:.3f}".format,
            "control_prob": "{:.6f}".format,
        }))


if __name__ == "__main__":
    main()
