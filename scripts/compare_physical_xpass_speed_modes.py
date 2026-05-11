from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch

from datatools.config import LABEL_INDEX
from physical_pass_model import (
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
)
from project_config import (
    get_action_graph_dir,
    get_physical_xpass_dir,
    resolve_feature_root,
    resolve_feature_run_id,
    write_run_metadata,
)
from scripts.generate_physical_xpass import (
    compute_match_rows,
    configure_worker_thread_limit,
    resolve_match_ids,
    resolve_num_workers,
    resolve_reference_label_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact and package-max physical xPass speed aggregation.")
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--match-id", action="append", help="Restrict to one match id. Repeat for multiple matches.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum sampled pass actions.")
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--worker-thread-limit", type=int, default=1)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--return-type", "--return_type", dest="return_type", default=None)
    parser.add_argument("--intended-receiver-mode", default=None)
    parser.add_argument(
        "--ignore-teammates",
        dest="consider_teammates",
        action="store_false",
        help="Use reduced target-plus-defenders policy for the comparison.",
    )
    parser.add_argument(
        "--consider-teammates",
        dest="consider_teammates",
        action="store_true",
        help="Default. Use all finite non-goal players.",
    )
    parser.set_defaults(consider_teammates=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    if args.worker_thread_limit < 1:
        parser.error("--worker-thread-limit must be positive.")
    return args


def collect_sample_row_indices(
    match_ids: list[str],
    graph_dir: Path,
    label_dir: Path,
    *,
    limit: int | None,
) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {}
    count = 0
    for match_id in match_ids:
        graph_path = graph_dir / f"{match_id}.pt"
        label_path = label_dir / f"{match_id}.pt"
        if not graph_path.exists() or not label_path.exists():
            continue
        labels = torch.load(label_path, weights_only=False)
        if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
            continue
        row_indices: set[int] = set()
        for row_index in range(int(labels.shape[0])):
            if int(labels[row_index, LABEL_INDEX["is_pass"]].item()) != 1:
                continue
            row_indices.add(row_index)
            count += 1
            if limit is not None and count >= limit:
                break
        if row_indices:
            selected[str(match_id)] = row_indices
        if limit is not None and count >= limit:
            break
    return selected


def compute_mode_for_sample(
    selected: dict[str, set[int]],
    graph_dir: Path,
    label_dir: Path,
    *,
    speed_aggregation: str,
    consider_teammates: bool,
    physical_batch_size: int,
    num_workers: int,
    worker_thread_limit: int,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    tasks = [
        {
            "match_id": match_id,
            "graph_path": str(graph_dir / f"{match_id}.pt"),
            "label_path": str(label_dir / f"{match_id}.pt"),
            "row_indices": sorted(int(value) for value in row_indices),
            "speed_aggregation": speed_aggregation,
            "consider_teammates": bool(consider_teammates),
            "physical_batch_size": int(physical_batch_size),
        }
        for match_id, row_indices in selected.items()
    ]
    frames: list[pd.DataFrame] = []
    if num_workers == 1:
        for task in tasks:
            frame = compute_mode_match_task(task)
            if not frame.empty:
                frames.append(frame)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=configure_worker_thread_limit,
            initargs=(int(worker_thread_limit),),
        ) as executor:
            futures = [executor.submit(compute_mode_match_task, task) for task in tasks]
            for future in as_completed(futures):
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
    elapsed = time.perf_counter() - start
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), elapsed


def compute_mode_match_task(task: dict[str, object]) -> pd.DataFrame:
    frame, _computed = compute_match_rows(
        str(task["match_id"]),
        Path(str(task["graph_path"])),
        Path(str(task["label_path"])),
        eps=1e-4,
        normalize=True,
        consider_teammates=bool(task["consider_teammates"]),
        speed_aggregation=str(task["speed_aggregation"]),
        physical_batch_size=int(task["physical_batch_size"]),
        limit=None,
        selected_row_indices={int(value) for value in task["row_indices"]},
        show_progress=False,
    )
    return frame


def long_compare_frame(exact: pd.DataFrame, package: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"match_id", "action_index", "action_id", "physical_state_hash"}
    exact = exact.set_index(["match_id", "action_index"], drop=False)
    package = package.set_index(["match_id", "action_index"], drop=False)
    keys = exact.index.intersection(package.index)
    rows = []
    for key in keys:
        exact_row = exact.loc[key]
        package_row = package.loc[key]
        player_columns = sorted(
            (set(exact_row.index) | set(package_row.index)) - id_columns,
            key=str,
        )
        for player_id in player_columns:
            exact_value = pd.to_numeric(exact_row.get(player_id, np.nan), errors="coerce")
            package_value = pd.to_numeric(package_row.get(player_id, np.nan), errors="coerce")
            if not (math.isfinite(float(exact_value)) and math.isfinite(float(package_value))):
                continue
            rows.append(
                {
                    "match_id": key[0],
                    "action_index": int(key[1]),
                    "player_id": str(player_id),
                    "exact_separate_speed": float(exact_value),
                    "package_max": float(package_value),
                    "abs_diff": abs(float(exact_value) - float(package_value)),
                }
            )
    return pd.DataFrame(rows)


def top_option_agreement(exact: pd.DataFrame, package: pd.DataFrame) -> float | None:
    id_columns = {"match_id", "action_index", "action_id", "physical_state_hash"}
    exact = exact.set_index(["match_id", "action_index"], drop=False)
    package = package.set_index(["match_id", "action_index"], drop=False)
    keys = exact.index.intersection(package.index)
    if len(keys) == 0:
        return None
    agreements = []
    for key in keys:
        exact_row = pd.to_numeric(exact.loc[key].drop(labels=[col for col in id_columns if col in exact.loc[key].index]), errors="coerce")
        package_row = pd.to_numeric(
            package.loc[key].drop(labels=[col for col in id_columns if col in package.loc[key].index]),
            errors="coerce",
        )
        common = exact_row.index.intersection(package_row.index)
        exact_row = exact_row[common].dropna()
        package_row = package_row[common].dropna()
        common = exact_row.index.intersection(package_row.index)
        if len(common) == 0:
            continue
        agreements.append(str(exact_row[common].idxmax()) == str(package_row[common].idxmax()))
    if not agreements:
        return None
    return float(np.mean(agreements))


def main() -> None:
    args = parse_args()
    num_workers = resolve_num_workers(args.num_workers)
    configure_worker_thread_limit(int(args.worker_thread_limit))
    feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(feature_run_id)
    graph_dir = get_action_graph_dir(feature_root)
    label_dir = resolve_reference_label_dir(feature_run_id, feature_root, args)
    match_ids = resolve_match_ids(args, graph_dir)
    selected = collect_sample_row_indices(match_ids, graph_dir, label_dir, limit=args.limit)
    if not selected:
        raise RuntimeError("No pass actions were found for the requested comparison sample.")

    exact, exact_runtime = compute_mode_for_sample(
        selected,
        graph_dir,
        label_dir,
        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
        consider_teammates=bool(args.consider_teammates),
        physical_batch_size=int(args.physical_batch_size),
        num_workers=int(num_workers),
        worker_thread_limit=int(args.worker_thread_limit),
    )
    package, package_runtime = compute_mode_for_sample(
        selected,
        graph_dir,
        label_dir,
        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
        consider_teammates=bool(args.consider_teammates),
        physical_batch_size=int(args.physical_batch_size),
        num_workers=int(num_workers),
        worker_thread_limit=int(args.worker_thread_limit),
    )
    comparison = long_compare_frame(exact, package)
    if comparison.empty:
        raise RuntimeError("No overlapping finite candidate values were produced by both speed modes.")

    diff = comparison["abs_diff"].astype(float)
    report = {
        "feature_run_id": feature_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "teammate_policy": (
            PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER if bool(args.consider_teammates) else "ignore_teammates"
        ),
        "sampled_actions": int(sum(len(values) for values in selected.values())),
        "candidate_count": int(len(comparison)),
        "exact_runtime_seconds": float(exact_runtime),
        "package_max_runtime_seconds": float(package_runtime),
        "speedup": float(exact_runtime / package_runtime) if package_runtime > 0 else None,
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(diff.median()),
        "p90_abs_diff": float(diff.quantile(0.90)),
        "p95_abs_diff": float(diff.quantile(0.95)),
        "max_abs_diff": float(diff.max()),
        "pearson": float(comparison["exact_separate_speed"].corr(comparison["package_max"], method="pearson")),
        "spearman": float(comparison["exact_separate_speed"].corr(comparison["package_max"], method="spearman")),
        "top_option_agreement": top_option_agreement(exact, package),
    }

    output_dir = get_physical_xpass_dir(feature_root) / "comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("speed_modes_%Y%m%dT%H%M%S")
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    comparison.to_csv(csv_path, index=False)
    write_run_metadata(json_path.parent, {"latest_speed_mode_comparison": report})
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
