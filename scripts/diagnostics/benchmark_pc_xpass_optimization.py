"""Create and compare a small deterministic PC-xPass optimization reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from datatools.benchmark import build_benchmark_state, load_benchmark_modification_data
from physical_pass_model import compute_graph_pc_xpass_metrics


REFERENCE_ROOT = ROOT / "data" / "pc_xpass_optimization_reference"
STATE_PAIRS = [(modification, game_state) for modification in (1, 2, 9, 14, 20, 28) for game_state in (1, 2)]
GRID_SIZES = (3.0, 1.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "compare", "time"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grids", nargs="+", type=float, default=list(GRID_SIZES))
    return parser.parse_args()


def load_graphs() -> list[tuple[int, int, object]]:
    graphs = []
    for modification, game_state in STATE_PAIRS:
        data = load_benchmark_modification_data(modification, ROOT / "benchmark")
        state, _, _ = build_benchmark_state(
            data[f"game_state_{game_state}"], modification, game_state, int(data["higher_state_id"])
        )
        graphs.append((modification, game_state, state.graph_features_0[0]))
    return graphs


def compute(graph: object, radial_gridsize: float) -> pd.Series:
    return compute_graph_pc_xpass_metrics(
        graph,
        ignore_teammates_lane_survival=True,
        max_speed=25.0,
        min_speed=5.0,
        speed_step=2.0,
        angle_step=2.5,
        radial_gridsize=float(radial_gridsize),
        top_n=10,
        top_n_values=[10, 25],
        enabled_metrics=["max_xpass", "top10_xpass", "top25_xpass"],
        lane_power=15.0,
        lane_inflection_point=0.2,
        control_power=15.0,
        control_inflection_point=0.2,
        endpoint_normalization="share",
        boost_def_endpoint_control=1.0,
        reaction_time=None,
        reaction_time_mode="dist_pass",
        dist_pass_div=50.0,
        dist_pass_min=0.2,
        dist_pass_max=0.7,
        max_player_speed=5.0,
        max_player_speed_off=5.0,
        max_player_speed_def=5.0,
        use_position_discount=False,
        top_xt=True,
    )


def output_frame(graphs: list[tuple[int, int, object]], grids: list[float]) -> pd.DataFrame:
    rows = []
    for grid in grids:
        for modification, game_state, graph in graphs:
            row = compute(graph, grid).to_dict()
            row.update(modification=modification, game_state=game_state, radial_gridsize=grid)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["radial_gridsize", "modification", "game_state"]).sort_index(axis=1)


def timing(graphs: list[tuple[int, int, object]], grids: list[float], repeats: int) -> dict[str, object]:
    compute(graphs[0][2], grids[0])
    records = []
    for grid in grids:
        for repeat in range(int(repeats)):
            started = time.perf_counter()
            for _, _, graph in graphs:
                compute(graph, grid)
            elapsed = time.perf_counter() - started
            records.append({"radial_gridsize": grid, "repeat": repeat + 1, "seconds": elapsed})
    frame = pd.DataFrame(records)
    medians = frame.groupby("radial_gridsize")["seconds"].median().to_dict()
    return {"records": records, "median_seconds": {str(key): value for key, value in medians.items()}}


def compare(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    actual = actual.reindex(index=expected.index, columns=expected.columns)
    numeric_expected = expected.select_dtypes(include=[np.number])
    numeric_actual = actual[numeric_expected.columns]
    same_nan = np.array_equal(np.isnan(numeric_actual.to_numpy()), np.isnan(numeric_expected.to_numpy()))
    finite = np.isfinite(numeric_expected.to_numpy()) & np.isfinite(numeric_actual.to_numpy())
    differences = np.abs(numeric_actual.to_numpy() - numeric_expected.to_numpy())
    max_abs = float(np.max(differences[finite])) if np.any(finite) else 0.0
    equivalent = same_nan and bool(np.allclose(numeric_actual, numeric_expected, rtol=1e-12, atol=1e-12, equal_nan=True))
    return {"equivalent": equivalent, "same_nan_mask": same_nan, "max_abs_difference": max_abs}


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    graphs = load_graphs()
    grids = [float(value) for value in args.grids]
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "create":
        outputs = output_frame(graphs, grids)
        outputs.to_parquet(REFERENCE_ROOT / "reference.parquet")
        timings = timing(graphs, grids, args.repeats)
        metadata = {"states": STATE_PAIRS, "grids": grids, "settings": "future_pc_xpass_share_no_position_discount"}
        (REFERENCE_ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (REFERENCE_ROOT / "reference_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        print(json.dumps({"rows": len(outputs), **timings}, indent=2))
    elif args.mode == "compare":
        expected = pd.read_parquet(REFERENCE_ROOT / "reference.parquet")
        result = compare(output_frame(graphs, grids), expected)
        print(json.dumps(result, indent=2))
        if not result["equivalent"]:
            raise SystemExit(1)
    else:
        print(json.dumps(timing(graphs, grids, args.repeats), indent=2))


if __name__ == "__main__":
    main()
