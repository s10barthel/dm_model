from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from datatools.benchmark import (
    build_benchmark_state,
    discover_benchmark_modifications,
    load_benchmark_modification_data,
)
from datatools.config import LABEL_INDEX
from datatools.hawkeye import (
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    load_hawkeye_ball,
    load_hawkeye_tracking,
    resolve_situation_ids,
)
from datatools.skillcorner import (
    build_skillcorner_match_context,
    build_skillcorner_possession,
    discover_skillcorner_matches,
)
from physical_pass_model import (
    AS_DEFAULT_V0_MIN,
    AS_DEFAULT_ANGLE_STEP_DEG,
    AS_DEFAULT_COARSE_N_ANGLES,
    AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    AS_DEFAULT_REFINE_TOP_K_ANGLES,
    AS_DEFAULT_SPEED_STEP,
    AS_DEFAULT_V0_MAX,
    PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    PHYSICAL_DEFAULT_MAX_AUTO_WORKERS,
    PHYSICAL_XPASS_SOURCE,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    PHYSICAL_XPASS_SPEED_AGGREGATIONS,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
    PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    as_default_v0_values,
    compute_graph_max_player_cum_prob_as_defaults,
    compute_graphs_max_player_cum_prob_as_defaults,
    configure_physical_worker_thread_limit,
    load_physical_xpass_match,
    normalize_physical_xpass_speed_aggregation,
    observed_pass_distance,
    physical_state_hash,
    physical_xpass_as_default_metadata,
    prewarm_physical_xpass_runtime_cache,
    resolve_physical_num_workers,
    validate_physical_xpass_cache_metadata,
)
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    PROJECT_ROOT,
    get_action_graph_dir,
    get_action_label_dir,
    get_physical_xpass_dir,
    get_runtime_physical_xpass_dir,
    infer_feature_run_intended_receiver_modes,
    infer_feature_run_return_types,
    load_base_splits,
    resolve_feature_root,
    resolve_feature_run_id,
    write_run_metadata,
)


RUNTIME_DATASETS = ("sportec", "skillcorner", "benchmark", "hawkeye")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate physical xPass caches for inference, or legacy feature-run sidecars when --feature-run-id is set."
    )
    parser.add_argument("--feature-run-id", help="Legacy mode: write Sportec sidecars under data/features/runs/<id>/physical_xpass.")
    parser.add_argument("--match-id", action="append", help="Restrict Sportec matches. Repeat for multiple matches.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Sportec split subset.")
    parser.add_argument("--limit", type=int, default=None, help="Legacy/Sportec pass-action compute limit.")
    parser.add_argument("--sportec-runtime-match-window", type=int, default=4, help="Number of Sportec matches to prewarm per worker-pool window.")
    parser.add_argument("--overwrite", action="store_true", help="Deprecated outside legacy --feature-run-id mode.")
    parser.add_argument(
        "--runtime-sportec-cache",
        action="store_true",
        help="Deprecated compatibility flag; runtime cache mode is now the default when --feature-run-id is omitted.",
    )
    parser.add_argument("--reuse-cache-dir", default=None, help="Compatible physical_xpass directory to reuse before computing missing Sportec rows.")

    parser.add_argument("--no-sportec", action="store_true", help="Skip Sportec runtime cache generation.")
    parser.add_argument("--no-skillcorner", action="store_true", help="Skip SkillCorner runtime cache generation.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmark runtime cache generation.")
    parser.add_argument("--no-hawkeye", action="store_true", help="Skip Hawkeye runtime cache generation.")

    parser.add_argument("--skillcorner-input-dir", default=str(PROJECT_ROOT / "skillcorner_data"))
    parser.add_argument("--skillcorner-match-id", action="append", help="Restrict SkillCorner matches.")
    parser.add_argument("--skillcorner-limit", type=int, help="Only process the first N selected SkillCorner matches.")
    skillcorner_frames = parser.add_mutually_exclusive_group()
    skillcorner_frames.add_argument(
        "--skillcorner-frames-first-and-last",
        dest="skillcorner_frames_mode",
        action="store_const",
        const="first_and_last",
        default="first_and_last",
    )
    skillcorner_frames.add_argument(
        "--skillcorner-frames-all",
        dest="skillcorner_frames_mode",
        action="store_const",
        const="all",
    )

    parser.add_argument("--benchmark-input-dir", default=str(PROJECT_ROOT / "benchmark"))
    parser.add_argument("--benchmark-modification", "--modification", dest="benchmark_modification", action="append", type=int)
    parser.add_argument("--benchmark-limit", type=int, help="Only process the first N selected benchmark modifications.")

    parser.add_argument("--hawkeye-tracking-csv", "--tracking-csv", dest="hawkeye_tracking_csv", default=str(PROJECT_ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--hawkeye-ball-csv", "--ball-csv", dest="hawkeye_ball_csv", default=str(PROJECT_ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    parser.add_argument("--hawkeye-situation-id", "--situation-id", dest="hawkeye_situation_id", action="append")
    parser.add_argument("--hawkeye-limit", type=int, help="Only process the first N selected Hawkeye situations.")
    parser.add_argument("--freeze-ballreceipt", dest="freeze_ballreceipt", action="store_true")
    parser.add_argument("--no-freeze-ballreceipt", dest="freeze_ballreceipt", action="store_false")

    parser.add_argument(
        "--return-type",
        "--return_type",
        dest="return_type",
        default=None,
        help="Legacy/Sportec label return type. Defaults to the first return type in the feature run.",
    )
    parser.add_argument(
        "--intended-receiver-mode",
        default=None,
        help="Legacy/Sportec intended-receiver mode. Defaults to angle_only when available.",
    )
    parser.add_argument("--physical-eps", type=float, default=1e-4)
    parser.add_argument(
        "--speed-aggregation",
        choices=sorted(PHYSICAL_XPASS_SPEED_AGGREGATIONS),
        default=PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    )
    parser.add_argument("--max-speed", "--max_speed", dest="max_speed", type=float, default=AS_DEFAULT_V0_MAX)
    parser.add_argument("--speed-step", "--speed_step", dest="speed_step", type=float, default=AS_DEFAULT_SPEED_STEP)
    parser.add_argument("--coarse-n-angles", "--coarse_n_angles", dest="coarse_n_angles", type=int, default=AS_DEFAULT_COARSE_N_ANGLES)
    parser.add_argument("--refine-top-k-angles", "--refine_top_k_angles", dest="refine_top_k_angles", type=int, default=AS_DEFAULT_REFINE_TOP_K_ANGLES)
    parser.add_argument("--refine-angle-radius", "--refine_angle_radius", dest="refine_angle_radius", type=float, default=AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG)
    parser.add_argument("--angle-step", "--angle_step", dest="angle_step", type=float, default=AS_DEFAULT_ANGLE_STEP_DEG)
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--max-auto-workers", type=int, default=PHYSICAL_DEFAULT_MAX_AUTO_WORKERS)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.add_argument("--worker-thread-limit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Scan runtime cache hits/misses without computing or writing rows.")
    teammate_policy_group = parser.add_mutually_exclusive_group()
    teammate_policy_group.add_argument("--ignore-teammates", dest="consider_teammates", action="store_false")
    teammate_policy_group.add_argument("--consider-teammates", dest="consider_teammates", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Deprecated compatibility flag; ignored.")
    parser.set_defaults(normalize=True, consider_teammates=True, freeze_ballreceipt=True)

    args = parser.parse_args(argv)
    for name in ["limit", "skillcorner_limit", "benchmark_limit", "hawkeye_limit"]:
        value = getattr(args, name, None)
        if value is not None and int(value) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.sportec_runtime_match_window < 1:
        parser.error("--sportec-runtime-match-window must be positive.")
    if not (0.0 < args.physical_eps < 0.5):
        parser.error("--physical-eps must be between 0 and 0.5.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    if args.worker_thread_limit < 1:
        parser.error("--worker-thread-limit must be positive.")
    if args.max_auto_workers < 1:
        parser.error("--max-auto-workers must be positive.")
    if args.max_speed is not None and args.max_speed < int(AS_DEFAULT_V0_MIN):
        parser.error(f"--max-speed must be at least {AS_DEFAULT_V0_MIN:g} m/s.")
    if args.speed_step <= 0:
        parser.error("--speed-step must be positive.")
    if args.coarse_n_angles < 1:
        parser.error("--coarse-n-angles must be positive.")
    if args.refine_top_k_angles < 1:
        parser.error("--refine-top-k-angles must be positive.")
    if args.refine_angle_radius < 0:
        parser.error("--refine-angle-radius must be non-negative.")
    if args.angle_step <= 0:
        parser.error("--angle-step must be positive.")
    try:
        resolve_physical_num_workers(args.num_workers, max_auto_workers=args.max_auto_workers)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def resolve_reference_label_dir(feature_run_id: str, feature_root: Path, args: argparse.Namespace) -> Path:
    return_types = infer_feature_run_return_types(feature_run_id)
    if args.return_type:
        return_type = str(args.return_type)
    elif return_types:
        return_type = return_types[0]
    else:
        raise FileNotFoundError(f"Feature run {feature_run_id} does not expose any action-label return types.")

    modes = infer_feature_run_intended_receiver_modes(feature_run_id)
    if args.intended_receiver_mode:
        mode = str(args.intended_receiver_mode)
    elif DEFAULT_INTENDED_RECEIVER_MODE in modes:
        mode = DEFAULT_INTENDED_RECEIVER_MODE
    elif modes:
        mode = modes[0]
    else:
        mode = DEFAULT_INTENDED_RECEIVER_MODE

    label_dir = get_action_label_dir(return_type, intended_receiver_mode=mode, root=feature_root)
    if not label_dir.exists():
        raise FileNotFoundError(
            f"Reference labels not found at {label_dir}. Provide --return-type/--intended-receiver-mode for an existing label set."
        )
    return label_dir


def resolve_match_ids(args: argparse.Namespace, graph_dir: Path) -> list[str]:
    if args.match_id:
        return [str(match_id) for match_id in args.match_id]

    train_ids, test_ids = load_base_splits(feature_dir=graph_dir)
    if args.split == "train":
        return [str(match_id) for match_id in train_ids.tolist()]
    if args.split == "test":
        return [str(match_id) for match_id in test_ids.tolist()]
    return [str(match_id) for match_id in train_ids.tolist()] + [str(match_id) for match_id in test_ids.tolist()]


def teammate_policy_from_args(args: argparse.Namespace) -> str:
    return PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER if bool(args.consider_teammates) else PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE


def resolve_num_workers(value: str | int, *, max_auto_workers: int = PHYSICAL_DEFAULT_MAX_AUTO_WORKERS) -> int:
    return resolve_physical_num_workers(value, max_auto_workers=max_auto_workers)


def configure_worker_thread_limit(limit: int) -> None:
    configure_physical_worker_thread_limit(limit)


def validate_reuse_cache_dir(
    cache_dir: str | Path,
    *,
    teammate_policy: str,
    speed_aggregation: str,
    physical_eps: float,
    max_speed: int | None = None,
    speed_step: float | None = None,
) -> Path:
    cache_path = Path(cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_path)
    expected_metadata = physical_xpass_as_default_metadata(
        teammate_policy,
        speed_aggregation=speed_aggregation,
        max_speed=max_speed,
        speed_step=speed_step,
    )
    mismatches: list[str] = []
    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if key == "speed_aggregation":
            actual_value = normalize_physical_xpass_speed_aggregation(actual_value)
            expected_value = normalize_physical_xpass_speed_aggregation(expected_value)
        if key in {"max_speed", "n_v0", "v0_max"} and actual_value is None and expected_metadata["max_speed"] is None:
            continue
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    metadata_eps = metadata.get("physical_eps")
    if metadata_eps is not None and abs(float(metadata_eps) - float(physical_eps)) > 1e-12:
        mismatches.append(f"physical_eps: expected {float(physical_eps)!r}, got {metadata_eps!r}")
    if mismatches:
        details = "; ".join(mismatches[:8])
        raise ValueError(f"--reuse-cache-dir {cache_path} is not compatible with this physical xPass run. {details}.")
    return cache_path


def load_reuse_rows(cache_dir: Path | None, match_id: str) -> pd.DataFrame | None:
    if cache_dir is None:
        return None
    try:
        return load_physical_xpass_match(cache_dir, match_id)
    except FileNotFoundError:
        return None


def resolve_runtime_sportec_reuse_cache(
    args: argparse.Namespace,
    feature_cache_dir: Path,
) -> tuple[Path | None, str | None]:
    explicit_reuse = bool(args.reuse_cache_dir)
    candidate = Path(args.reuse_cache_dir) if explicit_reuse else (feature_cache_dir if feature_cache_dir.exists() else None)
    if candidate is None:
        return None, None

    try:
        return (
            validate_reuse_cache_dir(
                candidate,
                teammate_policy=teammate_policy_from_args(args),
                speed_aggregation=normalize_physical_xpass_speed_aggregation(args.speed_aggregation),
                max_speed=args.max_speed,
                speed_step=args.speed_step,
                physical_eps=float(args.physical_eps),
            ),
            None,
        )
    except ValueError as exc:
        if explicit_reuse:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        print(f"Skipping incompatible implicit Sportec physical xPass reuse cache {candidate}: {exc}")
        return None, reason


def compute_match_rows(
    match_id: str,
    graph_path: Path,
    label_path: Path,
    *,
    eps: float,
    normalize: bool,
    consider_teammates: bool,
    speed_aggregation: str = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    max_speed: int | None = None,
    speed_step: float | None = None,
    physical_batch_size: int = 16,
    limit: int | None,
    reuse_rows: pd.DataFrame | None = None,
    reuse_stats: dict[str, int] | None = None,
    compute_fn=compute_graph_max_player_cum_prob_as_defaults,
    compute_batch_fn=compute_graphs_max_player_cum_prob_as_defaults,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, int]:
    del normalize
    speed_aggregation = normalize_physical_xpass_speed_aggregation(speed_aggregation)
    reuse_stats = reuse_stats if reuse_stats is not None else {}
    graphs = torch.load(graph_path, weights_only=False)
    labels = torch.load(label_path, weights_only=False)
    if not isinstance(graphs, list):
        raise ValueError(f"Graph artifact {graph_path} is not a list.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError(f"Label artifact {label_path} has invalid shape.")
    if len(graphs) != int(labels.shape[0]):
        raise ValueError(f"Graph/label length mismatch for match {match_id}: {len(graphs)} != {int(labels.shape[0])}.")

    rows = []
    pending: list[tuple[int, int, object, str, float]] = []
    computed = 0
    iterator = tqdm(range(int(labels.shape[0])), desc=f"physical_xpass {match_id}", leave=False) if show_progress else range(int(labels.shape[0]))
    for row_index in iterator:
        if limit is not None and computed >= limit and reuse_rows is None:
            break
        label = labels[row_index]
        if int(label[LABEL_INDEX["is_pass"]].item()) != 1:
            continue
        graph = graphs[row_index]
        if graph is None:
            continue
        action_index = int(label[LABEL_INDEX["action_index"]].item())
        current_hash = physical_state_hash(graph)
        if reuse_rows is not None and action_index in reuse_rows.index:
            reuse_row = reuse_rows.loc[action_index].copy()
            reusable_hash = reuse_row.get("physical_state_hash", None)
            if pd.isna(reusable_hash) or reusable_hash is None:
                reuse_stats["reused_without_state_hash"] = reuse_stats.get("reused_without_state_hash", 0) + 1
                reused = True
            elif str(reusable_hash) == current_hash:
                reuse_stats["hash_verified"] = reuse_stats.get("hash_verified", 0) + 1
                reused = True
            else:
                reused = False
            if reused:
                row = reuse_row.to_dict()
                row["match_id"] = str(match_id)
                row["action_index"] = action_index
                row["physical_state_hash"] = current_hash
                if pd.isna(row.get("pass_distance", None)):
                    row["pass_distance"] = observed_pass_distance(graph, label)
                    reuse_stats["pass_distance_filled"] = reuse_stats.get("pass_distance_filled", 0) + 1
                rows.append(row)
                reuse_stats["reused_actions"] = reuse_stats.get("reused_actions", 0) + 1
                continue
            reuse_stats["hash_mismatch_recomputed"] = reuse_stats.get("hash_mismatch_recomputed", 0) + 1
        if limit is not None and computed >= limit:
            reuse_stats["compute_limit_skipped"] = reuse_stats.get("compute_limit_skipped", 0) + 1
            continue
        pending.append((int(row_index), action_index, graph, current_hash, observed_pass_distance(graph, label)))
        computed += 1

    if pending:
        use_batch_fn = compute_batch_fn is not None and compute_fn is compute_graph_max_player_cum_prob_as_defaults
        if use_batch_fn:
            probs_list = compute_batch_fn(
                [item[2] for item in pending],
                eps=eps,
                consider_teammates=consider_teammates,
                speed_aggregation=speed_aggregation,
                max_speed=max_speed,
                speed_step=speed_step,
                batch_size=physical_batch_size,
            )
        else:
            probs_list = [
                compute_fn(
                    item[2],
                    eps=eps,
                    consider_teammates=consider_teammates,
                    speed_aggregation=speed_aggregation,
                    max_speed=max_speed,
                    speed_step=speed_step,
                )
                for item in pending
            ]
        if len(probs_list) != len(pending):
            raise ValueError(f"Physical xPass batch compute returned {len(probs_list)} rows for {len(pending)} graphs.")
        for (_, action_index, _graph, current_hash, pass_distance), probs in zip(pending, probs_list):
            row = {
                "match_id": str(match_id),
                "action_index": action_index,
                "physical_state_hash": current_hash,
                "pass_distance": pass_distance,
            }
            row.update({str(player_id): float(value) for player_id, value in probs.items()})
            rows.append(row)

    return pd.DataFrame(rows), computed


def process_match_task(task: dict[str, object]) -> dict[str, object]:
    match_id = str(task["match_id"])
    reuse_cache_dir = task.get("reuse_cache_dir")
    reuse_cache_path = Path(str(reuse_cache_dir)) if reuse_cache_dir else None
    reuse_stats: dict[str, int] = {}
    frame, computed = compute_match_rows(
        match_id,
        Path(str(task["graph_path"])),
        Path(str(task["label_path"])),
        eps=float(task["eps"]),
        normalize=True,
        consider_teammates=bool(task["consider_teammates"]),
        speed_aggregation=str(task["speed_aggregation"]),
        max_speed=task.get("max_speed"),
        speed_step=task.get("speed_step"),
        physical_batch_size=int(task["physical_batch_size"]),
        limit=task.get("limit"),
        reuse_rows=load_reuse_rows(reuse_cache_path, match_id),
        reuse_stats=reuse_stats,
        show_progress=bool(task.get("show_progress", False)),
    )
    reused = int(reuse_stats.get("reused_actions", 0))
    if frame.empty:
        return {"match_id": match_id, "computed": int(computed), "reused": reused, "written": False, "skip_reason": "no_pass_actions", "reuse_stats": reuse_stats}
    output_path = Path(str(task["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {"match_id": match_id, "computed": int(computed), "reused": reused, "written": True, "skip_reason": None, "reuse_stats": reuse_stats}


def runtime_cache_items_from_graphs(match_id: str, graphs: list[Any], labels: torch.Tensor) -> list[dict[str, Any]]:
    if labels is None or not isinstance(labels, torch.Tensor) or labels.numel() == 0 or not graphs:
        return []
    kept_graphs: list[Any] = []
    kept_labels: list[torch.Tensor] = []
    for graph, label in zip(graphs, labels):
        if graph is None:
            continue
        if int(label[LABEL_INDEX["is_pass"]].item()) != 1:
            continue
        kept_graphs.append(graph)
        kept_labels.append(label)
    if not kept_labels:
        return []
    return [{"match_id": str(match_id), "graphs": kept_graphs, "labels": torch.stack(kept_labels, axis=0)}]


def merge_stats(target: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    if not source:
        return target
    for key in [
        "rows_scanned",
        "pass_rows",
        "cache_hits",
        "cache_misses",
        "cache_written",
        "copied_from_reuse",
        "pass_distance_filled",
        "hash_mismatch_recomputed",
        "online_graphs",
        "compute_chunks",
        "skipped_all_nan",
    ]:
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))
    target["cache_dir"] = source.get("cache_dir", target.get("cache_dir"))
    target["dry_run"] = bool(target.get("dry_run", False)) or bool(source.get("dry_run", False))
    target["num_workers"] = source.get("num_workers", target.get("num_workers"))
    target["max_auto_workers"] = source.get("max_auto_workers", target.get("max_auto_workers"))
    target["worker_thread_limit"] = source.get("worker_thread_limit", target.get("worker_thread_limit"))
    target["physical_batch_size"] = source.get("physical_batch_size", target.get("physical_batch_size"))
    for key in ["cache_scan_seconds", "compute_seconds", "write_seconds"]:
        target[key] = float(target.get(key, 0.0) or 0.0) + float(source.get(key, 0.0) or 0.0)
    compute_seconds = float(target.get("compute_seconds", 0.0) or 0.0)
    if compute_seconds > 0.0:
        target["rows_per_second"] = float(target.get("online_graphs", 0) or 0) / compute_seconds
        target["chunks_per_second"] = float(target.get("compute_chunks", 0) or 0) / compute_seconds
    matches = dict(target.get("matches", {}))
    for match_id, source_match_stats in (source.get("matches", {}) or {}).items():
        current = dict(matches.get(str(match_id), {}))
        if isinstance(source_match_stats, dict):
            for key, value in source_match_stats.items():
                if isinstance(value, int):
                    current[key] = int(current.get(key, 0)) + int(value)
                elif isinstance(value, float):
                    current[key] = float(current.get(key, 0.0) or 0.0) + float(value)
                else:
                    current[key] = value
        matches[str(match_id)] = current
    target["matches"] = matches
    return target


def empty_runtime_stats(cache_dir: Path) -> dict[str, Any]:
    return {
        "cache_dir": str(cache_dir),
        "rows_scanned": 0,
        "pass_rows": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_written": 0,
        "copied_from_reuse": 0,
        "pass_distance_filled": 0,
        "hash_mismatch_recomputed": 0,
        "online_graphs": 0,
        "compute_chunks": 0,
        "skipped_all_nan": 0,
        "dry_run": False,
        "max_auto_workers": None,
        "cache_scan_seconds": 0.0,
        "compute_seconds": 0.0,
        "write_seconds": 0.0,
        "rows_per_second": None,
        "chunks_per_second": None,
        "matches": {},
    }


def prewarm_runtime_items(
    items: list[dict[str, Any]],
    *,
    cache_dir: Path,
    args: argparse.Namespace,
    reuse_cache_dir: Path | None = None,
    progress_desc: str | None = None,
) -> dict[str, Any] | None:
    if not items:
        return None
    if progress_desc is None:
        match_ids = sorted({str(item.get("match_id")) for item in items})
        progress_desc = f"physical xPass {match_ids[0]}" if len(match_ids) == 1 else "physical xPass runtime"
    return prewarm_physical_xpass_runtime_cache(
        items,
        cache_dir=cache_dir,
        source=PHYSICAL_XPASS_SOURCE,
        eps=float(args.physical_eps),
        teammate_policy=teammate_policy_from_args(args),
        speed_aggregation=normalize_physical_xpass_speed_aggregation(args.speed_aggregation),
        refresh=False,
        num_workers=args.num_workers,
        max_auto_workers=int(args.max_auto_workers),
        worker_thread_limit=int(args.worker_thread_limit),
        physical_batch_size=int(args.physical_batch_size),
        reuse_cache_dir=reuse_cache_dir,
        max_speed=args.max_speed,
        speed_step=args.speed_step,
        coarse_n_angles=int(args.coarse_n_angles),
        refine_top_k_angles=int(args.refine_top_k_angles),
        refine_angle_radius=float(args.refine_angle_radius),
        angle_step=float(args.angle_step),
        dry_run=bool(args.dry_run),
        show_progress=not bool(args.dry_run),
        progress_desc=progress_desc,
        verbose_status=True,
    )


def write_runtime_dataset_metadata(
    dataset: str,
    cache_dir: Path,
    args: argparse.Namespace,
    *,
    stats: dict[str, Any],
    source_inputs: dict[str, Any],
    skipped: dict[str, Any],
) -> None:
    if bool(getattr(args, "dry_run", False)):
        print(f"Dry run: not writing runtime metadata for {dataset} at {cache_dir}.")
        return
    metadata = {
        **physical_xpass_as_default_metadata(
            teammate_policy_from_args(args),
            speed_aggregation=normalize_physical_xpass_speed_aggregation(args.speed_aggregation),
            max_speed=args.max_speed,
            speed_step=args.speed_step,
            coarse_n_angles=args.coarse_n_angles,
            refine_top_k_angles=args.refine_top_k_angles,
            refine_angle_radius=args.refine_angle_radius,
            angle_step=args.angle_step,
        ),
        "created_for": "runtime_physical_xpass_cache",
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_inputs": source_inputs,
        "skipped": skipped,
        "stats": stats,
        "dry_run": bool(getattr(args, "dry_run", False)),
        "physical_eps": float(args.physical_eps),
        "num_workers": args.num_workers,
        "max_auto_workers": int(args.max_auto_workers),
        "resolved_num_workers": int(resolve_physical_num_workers(args.num_workers, max_auto_workers=args.max_auto_workers)),
        "physical_batch_size": int(args.physical_batch_size),
        "worker_thread_limit": int(args.worker_thread_limit),
        "effective_v0_grid": [float(value) for value in as_default_v0_values(max_speed=args.max_speed, speed_step=args.speed_step).tolist()],
        "storage": "wide_parquet_one_row_per_action_player_id_columns",
    }
    write_run_metadata(cache_dir, metadata)


def run_legacy_feature_mode(args: argparse.Namespace) -> None:
    speed_aggregation = normalize_physical_xpass_speed_aggregation(args.speed_aggregation)
    num_workers = resolve_physical_num_workers(args.num_workers, max_auto_workers=args.max_auto_workers)
    if args.limit is not None and num_workers > 1:
        raise ValueError("--limit cannot be used with --num-workers > 1. Use --num-workers 1 for limited smoke tests.")
    if args.runtime_sportec_cache:
        warnings.warn("--runtime-sportec-cache is ignored when --feature-run-id is provided; legacy feature-run mode writes data/features sidecars.")

    feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(feature_run_id)
    graph_dir = get_action_graph_dir(feature_root)
    label_dir = resolve_reference_label_dir(feature_run_id, feature_root, args)
    output_root = get_physical_xpass_dir(feature_root)
    match_ids = resolve_match_ids(args, graph_dir)
    reuse_cache_dir = (
        validate_reuse_cache_dir(
            args.reuse_cache_dir,
            teammate_policy=teammate_policy_from_args(args),
            speed_aggregation=speed_aggregation,
            max_speed=args.max_speed,
            speed_step=args.speed_step,
            physical_eps=float(args.physical_eps),
        )
        if args.reuse_cache_dir
        else None
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matches").mkdir(parents=True, exist_ok=True)
    total_computed = 0
    total_reused = 0
    reuse_stats: dict[str, int] = {}
    written_match_ids: list[str] = []
    skipped_match_ids: dict[str, str] = {}
    tasks: list[dict[str, object]] = []
    for match_id in tqdm(match_ids, desc="sportec legacy matches"):
        if args.limit is not None and total_computed >= args.limit and reuse_cache_dir is None:
            break
        output_path = output_root / "matches" / f"{match_id}.parquet"
        if output_path.exists() and not args.overwrite:
            skipped_match_ids[match_id] = "exists"
            continue
        graph_path = graph_dir / f"{match_id}.pt"
        label_path = label_dir / f"{match_id}.pt"
        if not graph_path.exists() or not label_path.exists():
            skipped_match_ids[match_id] = "missing_graph_or_label"
            continue
        remaining = None if args.limit is None else args.limit - total_computed
        task = {
            "match_id": match_id,
            "graph_path": str(graph_path),
            "label_path": str(label_path),
            "output_path": str(output_path),
            "eps": float(args.physical_eps),
            "consider_teammates": bool(args.consider_teammates),
            "speed_aggregation": speed_aggregation,
            "max_speed": args.max_speed,
            "speed_step": args.speed_step,
            "physical_batch_size": int(args.physical_batch_size),
            "limit": remaining,
            "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir is not None else None,
            "show_progress": num_workers == 1,
        }
        if args.limit is not None:
            result = process_match_task(task)
        else:
            tasks.append(task)
            continue
        for key, value in result["reuse_stats"].items():
            reuse_stats[key] = reuse_stats.get(key, 0) + int(value)
        if result["written"]:
            total_computed += int(result["computed"])
            total_reused += int(result["reused"])
            written_match_ids.append(match_id)
        else:
            skipped_match_ids[match_id] = str(result["skip_reason"])

    if args.limit is None and tasks:
        if num_workers == 1:
            results = (process_match_task(task) for task in tasks)
            iterator = tqdm(results, total=len(tasks), desc="sportec legacy compute")
        else:
            executor = ProcessPoolExecutor(max_workers=num_workers, initializer=configure_physical_worker_thread_limit, initargs=(int(args.worker_thread_limit),))
            futures = [executor.submit(process_match_task, task) for task in tasks]
            iterator = tqdm((future.result() for future in as_completed(futures)), total=len(futures), desc="sportec legacy compute")
        for result in iterator:
            match_id = str(result["match_id"])
            for key, value in result["reuse_stats"].items():
                reuse_stats[key] = reuse_stats.get(key, 0) + int(value)
            if result["written"]:
                total_computed += int(result["computed"])
                total_reused += int(result["reused"])
                written_match_ids.append(match_id)
            else:
                skipped_match_ids[match_id] = str(result["skip_reason"])
        if num_workers != 1:
            executor.shutdown()

    metadata = {
        **physical_xpass_as_default_metadata(
            teammate_policy_from_args(args),
            speed_aggregation=speed_aggregation,
            max_speed=args.max_speed,
            speed_step=args.speed_step,
            default_metric="max_xpass",
            available_metrics=["max_xpass"],
            metric_schema_version=1,
        ),
        "feature_run_id": feature_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "graph_dir": str(graph_dir),
        "label_dir": str(label_dir),
        "split": args.split,
        "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir is not None else None,
        "output_root": str(output_root),
        "match_ids": written_match_ids,
        "skipped_match_ids": skipped_match_ids,
        "n_actions": int(total_computed + total_reused),
        "n_computed_actions": int(total_computed),
        "n_reused_actions": int(total_reused),
        "reuse_stats": {key: int(value) for key, value in sorted(reuse_stats.items())},
        "physical_eps": float(args.physical_eps),
        "effective_v0_grid": [float(value) for value in as_default_v0_values(max_speed=args.max_speed, speed_step=args.speed_step).tolist()],
        "num_workers": int(num_workers),
        "max_auto_workers": int(args.max_auto_workers),
        "physical_batch_size": int(args.physical_batch_size),
        "worker_thread_limit": int(args.worker_thread_limit),
        "storage": "wide_parquet_one_row_per_action_player_id_columns",
    }
    write_run_metadata(output_root, metadata)
    print(f"Wrote legacy Sportec physical xPass sidecars at {output_root}: {total_computed} computed, {total_reused} reused.")


def run_runtime_sportec(args: argparse.Namespace) -> dict[str, Any]:
    feature_run_id = resolve_feature_run_id(None, required=True, allow_latest=True)
    feature_root = resolve_feature_root(feature_run_id)
    graph_dir = get_action_graph_dir(feature_root)
    label_dir = resolve_reference_label_dir(str(feature_run_id), feature_root, args)
    cache_dir = get_runtime_physical_xpass_dir("sportec")
    feature_cache_dir = get_physical_xpass_dir(feature_root)
    reuse_cache_dir, implicit_reuse_cache_skipped_reason = resolve_runtime_sportec_reuse_cache(args, feature_cache_dir)

    stats = empty_runtime_stats(cache_dir)
    skipped: dict[str, Any] = {}
    pending_items: list[dict[str, Any]] = []
    pending_match_ids: list[str] = []

    def flush_pending_window() -> None:
        nonlocal pending_items, pending_match_ids
        if not pending_items:
            pending_match_ids = []
            return
        first_match = pending_match_ids[0]
        last_match = pending_match_ids[-1]
        try:
            merge_stats(
                stats,
                prewarm_runtime_items(
                    pending_items,
                    cache_dir=cache_dir,
                    args=args,
                    reuse_cache_dir=reuse_cache_dir,
                    progress_desc=f"sportec {first_match}..{last_match}",
                ),
            )
            for pending_match_id in pending_match_ids:
                match_stats = (stats.get("matches", {}) or {}).get(str(pending_match_id), {})
                print(
                    f"sportec {pending_match_id}: hits={int(match_stats.get('cache_hits', 0) or 0)} "
                    f"misses={int(match_stats.get('cache_misses', 0) or 0)} "
                    f"written={int(match_stats.get('cache_written', 0) or 0)} "
                    f"copied={int(match_stats.get('copied_from_reuse', 0) or 0)} "
                    f"pass_distance_filled={int(match_stats.get('pass_distance_filled', 0) or 0)} "
                    f"skipped_all_nan={int(match_stats.get('skipped_all_nan', 0) or 0)}"
                )
        except Exception as exc:
            for pending_match_id in pending_match_ids:
                skipped[str(pending_match_id)] = f"{type(exc).__name__}: {exc}"
        finally:
            pending_items = []
            pending_match_ids = []

    for match_id in tqdm(resolve_match_ids(args, graph_dir), desc="sportec runtime"):
        graph_path = graph_dir / f"{match_id}.pt"
        label_path = label_dir / f"{match_id}.pt"
        if not graph_path.exists() or not label_path.exists():
            skipped[str(match_id)] = "missing_graph_or_label"
            continue
        try:
            graphs = torch.load(graph_path, weights_only=False)
            labels = torch.load(label_path, weights_only=False)
            items = runtime_cache_items_from_graphs(str(match_id), list(graphs), labels)
            if items:
                pending_items.extend(items)
                pending_match_ids.append(str(match_id))
            if len(pending_match_ids) >= int(args.sportec_runtime_match_window):
                flush_pending_window()
        except Exception as exc:
            skipped[str(match_id)] = f"{type(exc).__name__}: {exc}"
    flush_pending_window()
    write_runtime_dataset_metadata(
        "sportec",
        cache_dir,
        args,
        stats=stats,
        source_inputs={
            "feature_run_id": str(feature_run_id),
            "graph_dir": str(graph_dir),
            "label_dir": str(label_dir),
            "split": args.split,
            "match_window": int(args.sportec_runtime_match_window),
            "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir is not None else None,
            "implicit_reuse_cache_dir": str(feature_cache_dir) if not args.reuse_cache_dir and feature_cache_dir.exists() else None,
            "implicit_reuse_cache_skipped_reason": implicit_reuse_cache_skipped_reason,
        },
        skipped=skipped,
    )
    return {"dataset": "sportec", "cache_dir": str(cache_dir), "stats": stats, "skipped": skipped}


def run_runtime_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = get_runtime_physical_xpass_dir("benchmark")
    stats = empty_runtime_stats(cache_dir)
    skipped: dict[str, Any] = {}
    selected, skipped_discovery = discover_benchmark_modifications(
        args.benchmark_input_dir,
        requested_modifications=args.benchmark_modification,
        limit=args.benchmark_limit,
    )
    for modification_id in tqdm(selected, desc="benchmark runtime"):
        try:
            data = load_benchmark_modification_data(int(modification_id), args.benchmark_input_dir)
            states = [
                build_benchmark_state(data["game_state_1"], int(modification_id), 1, int(data["higher_state_id"]))[0],
                build_benchmark_state(data["game_state_2"], int(modification_id), 2, int(data["higher_state_id"]))[0],
            ]
            items: list[dict[str, Any]] = []
            for state in states:
                items.extend(runtime_cache_items_from_graphs(str(state.match_id), state.graph_features_0, state.labels))
            merge_stats(
                stats,
                prewarm_runtime_items(
                    items,
                    cache_dir=cache_dir,
                    args=args,
                    progress_desc=f"benchmark {modification_id}",
                ),
            )
        except Exception as exc:
            skipped[str(modification_id)] = f"{type(exc).__name__}: {exc}"
    write_runtime_dataset_metadata(
        "benchmark",
        cache_dir,
        args,
        stats=stats,
        source_inputs={"input_dir": str(args.benchmark_input_dir), "modifications": selected},
        skipped={"discovery": skipped_discovery, "processing": skipped},
    )
    return {"dataset": "benchmark", "cache_dir": str(cache_dir), "stats": stats, "skipped": skipped}


def run_runtime_hawkeye(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = get_runtime_physical_xpass_dir("hawkeye")
    stats = empty_runtime_stats(cache_dir)
    skipped: dict[str, Any] = {}
    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.hawkeye_tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.hawkeye_ball_csv))
    situation_ids = resolve_situation_ids(tracking, requested_ids=args.hawkeye_situation_id, limit=args.hawkeye_limit)
    for situation_id in tqdm(situation_ids, desc="hawkeye runtime"):
        try:
            situation_tracking = tracking.loc[tracking["id"] == situation_id].copy()
            situation = build_hawkeye_situation(
                situation_tracking,
                ball,
                freeze_ballreceipt=args.freeze_ballreceipt,
            )[0]
            merge_stats(
                stats,
                prewarm_runtime_items(
                    runtime_cache_items_from_graphs(str(situation.match_id), situation.graph_features_0, situation.labels),
                    cache_dir=cache_dir,
                    args=args,
                    progress_desc=f"hawkeye {situation_id}",
                ),
            )
        except Exception as exc:
            skipped[str(situation_id)] = f"{type(exc).__name__}: {exc}"
    write_runtime_dataset_metadata(
        "hawkeye",
        cache_dir,
        args,
        stats=stats,
        source_inputs={"tracking_csv": str(args.hawkeye_tracking_csv), "ball_csv": str(args.hawkeye_ball_csv), "situation_ids": situation_ids},
        skipped=skipped,
    )
    return {"dataset": "hawkeye", "cache_dir": str(cache_dir), "stats": stats, "skipped": skipped}


def run_runtime_skillcorner(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = get_runtime_physical_xpass_dir("skillcorner")
    stats = empty_runtime_stats(cache_dir)
    skipped: dict[str, Any] = {}
    selected, skipped_discovery = discover_skillcorner_matches(
        args.skillcorner_input_dir,
        requested_match_ids=args.skillcorner_match_id,
        limit=args.skillcorner_limit,
    )
    for match_id in tqdm(selected, desc="skillcorner runtime"):
        try:
            context = build_skillcorner_match_context(str(match_id), args.skillcorner_input_dir)
            events = context["events"]
            for event_index in events["index"].astype(int).tolist():
                try:
                    possession, _stats = build_skillcorner_possession(
                        context,
                        int(event_index),
                        frames_mode=args.skillcorner_frames_mode,
                    )
                    merge_stats(
                        stats,
                        prewarm_runtime_items(
                            runtime_cache_items_from_graphs(str(possession.match_id), possession.graph_features_0, possession.labels),
                            cache_dir=cache_dir,
                            args=args,
                            progress_desc=f"skillcorner {match_id}:{event_index}",
                        ),
                    )
                except Exception as exc:
                    skipped[f"{match_id}:{event_index}"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            skipped[str(match_id)] = f"{type(exc).__name__}: {exc}"
    write_runtime_dataset_metadata(
        "skillcorner",
        cache_dir,
        args,
        stats=stats,
        source_inputs={"input_dir": str(args.skillcorner_input_dir), "match_ids": selected, "frames_mode": args.skillcorner_frames_mode},
        skipped={"discovery": skipped_discovery, "processing": skipped},
    )
    return {"dataset": "skillcorner", "cache_dir": str(cache_dir), "stats": stats, "skipped": skipped}


def selected_runtime_datasets(args: argparse.Namespace) -> list[str]:
    selected = []
    if not args.no_sportec:
        selected.append("sportec")
    if not args.no_skillcorner:
        selected.append("skillcorner")
    if not args.no_benchmark:
        selected.append("benchmark")
    if not args.no_hawkeye:
        selected.append("hawkeye")
    return selected


def run_runtime_mode(args: argparse.Namespace) -> None:
    if args.overwrite:
        warnings.warn("--overwrite is ignored in runtime mode; runtime physical xPass caches are updated in place.")
    if not args.normalize:
        warnings.warn("--no-normalize is ignored; AS-default physical xPass always uses normalize=True.")
    configure_physical_worker_thread_limit(int(args.worker_thread_limit))
    runners = {
        "sportec": run_runtime_sportec,
        "skillcorner": run_runtime_skillcorner,
        "benchmark": run_runtime_benchmark,
        "hawkeye": run_runtime_hawkeye,
    }
    summaries = []
    for dataset in selected_runtime_datasets(args):
        print(f"Generating runtime physical xPass cache for {dataset}...")
        summaries.append(runners[dataset](args))
    print("Runtime physical xPass cache generation complete.")
    for summary in summaries:
        stats = summary["stats"]
        print(
            f"  {summary['dataset']}: {summary['cache_dir']} | "
            f"hits={int(stats.get('cache_hits', 0))}, misses={int(stats.get('cache_misses', 0))}, "
            f"written={int(stats.get('cache_written', 0))}, filled_distance={int(stats.get('pass_distance_filled', 0))}, "
            f"skipped={len(summary.get('skipped') or {})}"
        )


def main() -> None:
    args = parse_args()
    if args.feature_run_id:
        if not args.normalize:
            warnings.warn("--no-normalize is ignored; AS-default physical xPass always uses normalize=True.")
        configure_physical_worker_thread_limit(int(args.worker_thread_limit))
        run_legacy_feature_mode(args)
    else:
        run_runtime_mode(args)


if __name__ == "__main__":
    main()
