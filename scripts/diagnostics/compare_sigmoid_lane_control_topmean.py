"""Compare sigmoid-arrival lane/control xPass functions and top-mean scores."""

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
from scripts.diagnostics import probe_sigmoid_lane_control_xpass as sigproto


FUNCTIONS = {
    "original_15_0p3": (15.0, 0.3),
    "new_10_0p5": (10.0, 0.5),
}
TOP_NS = (10, 20, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation-id", default=chw.DEFAULT_SITUATION_ID)
    parser.add_argument("--time-norm", type=float, default=0.0)
    parser.add_argument("--tracking-csv", default=str(ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--ball-csv", default=str(ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    parser.add_argument("--max-speed", type=float, default=25.0)
    parser.add_argument("--speed-step", type=float, default=2.0)
    parser.add_argument("--angle-step", type=float, default=2.5)
    parser.add_argument("--reaction-time", type=float, default=chw.PC_REACTION_TIME)
    parser.add_argument("--max-player-speed", type=float, default=chw.PC_MAX_PLAYER_SPEED)
    return parser.parse_args()


def top_mean(values: np.ndarray, n: int) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return float("nan")
    k = min(int(n), int(finite.size))
    return float(np.mean(np.partition(finite, -k)[-k:]))


def score_grids(args: argparse.Namespace):
    graph, selection = chw.build_graph(args)
    setup = chw.make_setup(graph)
    speeds = chw.ppm.as_default_v0_values(max_speed=args.max_speed, speed_step=args.speed_step)
    angles = np.deg2rad(np.arange(0.0, 360.0, float(args.angle_step), dtype=float))

    normal_results = chw.simulate(setup, angles, speeds[:1], ("cum_p0",))
    cum_p0, r_grid = chw.stack_cum_p0(normal_results)
    on_pitch = np.isfinite(cum_p0[0])

    target_x_base = setup["raw_ball_pos"][0] + np.cos(angles)[:, np.newaxis] * r_grid[np.newaxis, :]
    target_y_base = setup["raw_ball_pos"][1] + np.sin(angles)[:, np.newaxis] * r_grid[np.newaxis, :]
    target_x = np.repeat(target_x_base[np.newaxis, :, :], len(speeds), axis=0)
    target_y = np.repeat(target_y_base[np.newaxis, :, :], len(speeds), axis=0)
    t_ball = np.repeat((r_grid[np.newaxis, np.newaxis, :] / speeds[:, np.newaxis, np.newaxis]), len(angles), axis=1)
    on_pitch_grid = np.repeat(on_pitch[np.newaxis, :, :], len(speeds), axis=0)
    margins = []
    for index in range(setup["player_pos"].shape[1]):
        tta = chw.time_to_arrive_with_params(
            setup["player_pos"][0, index],
            target_x,
            target_y,
            reaction_time=args.reaction_time,
            max_player_speed=args.max_player_speed,
        )
        margins.append(t_ball - tta)
    margins = np.stack(margins, axis=0)

    players = [str(value) for value in setup["players"]]
    node_ids = [str(value) for value in setup["node_ids"]]
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

    outputs = {}
    for function_name, (scale, offset) in FUNCTIONS.items():
        raw = sigproto.arrival_raw_control(margins, scale=scale, offset=offset)
        raw[:, :, ~on_pitch] = np.nan
        rows = []
        for graph_target_index in setup["candidate_indices"]:
            receiver = str(setup["node_ids"][graph_target_index])
            _, receiver_sim_index = setup["target_lookup"][graph_target_index]

            endpoint_raw = raw.copy()
            endpoint_raw[passer_sim_index] = 0.0
            endpoint_probs = sigproto.normalize_if_sum_above_one(endpoint_raw, axis=0)
            receiver_control = endpoint_probs[receiver_sim_index]

            lane_raw = raw.copy()
            lane_raw[passer_sim_index] = 0.0
            lane_raw[receiver_sim_index] = 0.0
            lane_probs = sigproto.normalize_if_sum_above_one(lane_raw, axis=0)
            other_control = np.nansum(lane_probs, axis=0)
            per_location_survival = np.clip(1.0 - other_control, 0.0, 1.0)
            lane_survival = np.ones_like(receiver_control, dtype=float)
            for distance_i in range(len(r_grid)):
                if distance_i == 0:
                    lane_survival[:, :, distance_i] = 1.0
                else:
                    lane_survival[:, :, distance_i] = np.prod(per_location_survival[:, :, :distance_i], axis=2)
            lane_survival = np.where(on_pitch_grid, lane_survival, np.nan)
            score = lane_survival * receiver_control
            flat = int(np.nanargmax(score))
            speed_i, angle_i, distance_i = np.unravel_index(flat, score.shape)
            x, y = positions[receiver]
            row = {
                "player_id": receiver,
                "x": x,
                "y": y,
                "score_max": float(score[speed_i, angle_i, distance_i]),
                "lane_survival": float(lane_survival[speed_i, angle_i, distance_i]),
                "control_prob": float(receiver_control[speed_i, angle_i, distance_i]),
                "speed": float(speeds[speed_i]),
                "angle": math.degrees(float(angles[angle_i])) % 360.0,
                "distance": float(r_grid[distance_i]),
                "target_x": float(target_x[speed_i, angle_i, distance_i]),
                "target_y": float(target_y[speed_i, angle_i, distance_i]),
                "ball_minus_player": float(margins[receiver_sim_index, speed_i, angle_i, distance_i]),
            }
            for n in TOP_NS:
                row[f"topmean_{n}"] = top_mean(score, n)
            rows.append(row)
        outputs[function_name] = pd.DataFrame(rows).sort_values("x")
    return selection, outputs


def main() -> None:
    args = parse_args()
    selection, outputs = score_grids(args)
    print("selection", selection)
    print("settings", {
        "functions": {
            name: f"1 / (1 + exp(-{scale:g} * (ball_minus_player + {offset:g})))"
            for name, (scale, offset) in FUNCTIONS.items()
        },
        "normalization": "raw / sum(raw) only where sum(raw) > 1",
        "top_ns": TOP_NS,
        "reaction_time": args.reaction_time,
        "max_player_speed": args.max_player_speed,
    })

    print("\nTABLE 1 - new function max details")
    print(outputs["new_10_0p5"].to_string(index=False, columns=[
        "player_id", "x", "y", "score_max", "lane_survival", "control_prob",
        "speed", "angle", "distance", "target_x", "target_y", "ball_minus_player",
    ], formatters={
        "x": "{:.2f}".format,
        "y": "{:.2f}".format,
        "score_max": "{:.6f}".format,
        "lane_survival": "{:.6f}".format,
        "control_prob": "{:.6f}".format,
        "speed": "{:.1f}".format,
        "angle": "{:.2f}".format,
        "distance": "{:.2f}".format,
        "target_x": "{:.2f}".format,
        "target_y": "{:.2f}".format,
        "ball_minus_player": "{:.3f}".format,
    }))

    table2 = outputs["original_15_0p3"][["player_id", "x", "y", "score_max", "topmean_10", "topmean_20", "topmean_25"]].copy()
    table2 = table2.rename(columns={
        "score_max": "original_max",
        "topmean_10": "original_topmean_10",
        "topmean_20": "original_topmean_20",
        "topmean_25": "original_topmean_25",
    })
    new_cols = outputs["new_10_0p5"][["player_id", "score_max", "topmean_10", "topmean_20", "topmean_25"]].rename(columns={
        "score_max": "new_max",
        "topmean_10": "new_topmean_10",
        "topmean_20": "new_topmean_20",
        "topmean_25": "new_topmean_25",
    })
    table2 = table2.merge(new_cols, on="player_id", how="left")
    print("\nTABLE 2 - max and topmean comparison")
    print(table2.to_string(index=False, formatters={
        "x": "{:.2f}".format,
        "y": "{:.2f}".format,
        "original_max": "{:.6f}".format,
        "original_topmean_10": "{:.6f}".format,
        "original_topmean_20": "{:.6f}".format,
        "original_topmean_25": "{:.6f}".format,
        "new_max": "{:.6f}".format,
        "new_topmean_10": "{:.6f}".format,
        "new_topmean_20": "{:.6f}".format,
        "new_topmean_25": "{:.6f}".format,
    }))


if __name__ == "__main__":
    main()
