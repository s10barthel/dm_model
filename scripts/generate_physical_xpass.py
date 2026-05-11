from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from datatools.config import LABEL_INDEX
from physical_pass_model import (
    PHYSICAL_XPASS_SOURCE,
    PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_SPEED_AGGREGATIONS,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
    PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    compute_graphs_max_player_cum_prob_as_defaults,
    compute_graph_max_player_cum_prob_as_defaults,
    load_physical_xpass_match,
    normalize_physical_xpass_speed_aggregation,
    physical_state_hash,
    physical_xpass_as_default_metadata,
    validate_physical_xpass_cache_metadata,
)
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    get_action_graph_dir,
    get_action_label_dir,
    get_physical_xpass_dir,
    get_physical_xpass_match_path,
    infer_feature_run_intended_receiver_modes,
    infer_feature_run_return_types,
    load_base_splits,
    resolve_feature_root,
    resolve_feature_run_id,
    write_run_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute AS-default max player_cum_prob physical xPass sidecars for pass actions."
    )
    parser.add_argument("--feature-run-id", required=True, help="Feature run whose action graphs should be used.")
    parser.add_argument("--match-id", action="append", help="Restrict to one match id. Repeat for multiple matches.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pass actions to compute across selected matches.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing match sidecars.")
    parser.add_argument(
        "--reuse-cache-dir",
        default=None,
        help="Compatible physical_xpass directory to reuse before computing missing pass actions.",
    )
    parser.add_argument("--split", choices=["train", "test", "all"], default="all", help="Split subset to precompute.")
    parser.add_argument(
        "--return-type",
        "--return_type",
        dest="return_type",
        default=None,
        help="Reference action-label return type. Defaults to the first return type in the feature run.",
    )
    parser.add_argument(
        "--intended-receiver-mode",
        default=None,
        help="Reference action-label intended-receiver mode. Defaults to angle_only when available.",
    )
    parser.add_argument(
        "--physical-eps",
        type=float,
        default=1e-4,
        help="Clamp stored max player_cum_prob to [eps, 1-eps].",
    )
    parser.add_argument(
        "--speed-aggregation",
        choices=sorted(PHYSICAL_XPASS_SPEED_AGGREGATIONS),
        default=PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
        help=(
            "How to aggregate the AS/DAS speed grid. package_max uses one accessible-space call with "
            "v0_prob_aggregation_mode='max'; exact_separate_speed preserves the old 15-call semantics."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default="auto",
        help="Number of match-level worker processes, or 'auto' (6 on a 16-logical-core machine).",
    )
    parser.add_argument(
        "--physical-batch-size",
        type=int,
        default=16,
        help="Number of compatible actions to batch into one accessible-space call for --consider-teammates.",
    )
    parser.add_argument(
        "--worker-thread-limit",
        type=int,
        default=1,
        help="OMP/MKL/NUMEXPR thread limit per worker process.",
    )
    teammate_policy_group = parser.add_mutually_exclusive_group()
    teammate_policy_group.add_argument(
        "--ignore-teammates",
        dest="consider_teammates",
        action="store_false",
        help="Simulate each candidate with passer, target teammate, and defenders only.",
    )
    teammate_policy_group.add_argument(
        "--consider-teammates",
        dest="consider_teammates",
        action="store_true",
        help="Default. Use all finite non-goal players in the AS-default max physical simulation.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Deprecated compatibility flag; ignored because the AS-default max metric always uses normalize=True.",
    )
    parser.set_defaults(normalize=True, consider_teammates=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not (0.0 < args.physical_eps < 0.5):
        parser.error("--physical-eps must be between 0 and 0.5.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    if args.worker_thread_limit < 1:
        parser.error("--worker-thread-limit must be positive.")
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


def resolve_num_workers(value: str | int) -> int:
    if isinstance(value, int):
        workers = value
    else:
        text = str(value).strip().lower()
        if text == "auto":
            cpu_count = os.cpu_count() or 1
            workers = max(1, min(6, cpu_count - 2))
        else:
            workers = int(text)
    if workers < 1:
        raise ValueError("--num-workers must be a positive integer or 'auto'.")
    return workers


def configure_worker_thread_limit(limit: int) -> None:
    value = str(int(limit))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def validate_reuse_cache_dir(
    cache_dir: str | Path,
    *,
    teammate_policy: str,
    speed_aggregation: str,
    physical_eps: float,
) -> Path:
    cache_path = Path(cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_path)
    expected_metadata = physical_xpass_as_default_metadata(
        teammate_policy,
        speed_aggregation=speed_aggregation,
    )
    mismatches: list[str] = []
    for key, expected_value in expected_metadata.items():
        actual_value = metadata.get(key)
        if key == "speed_aggregation":
            actual_value = normalize_physical_xpass_speed_aggregation(actual_value)
            expected_value = normalize_physical_xpass_speed_aggregation(expected_value)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")
    metadata_eps = metadata.get("physical_eps")
    if metadata_eps is None or abs(float(metadata_eps) - float(physical_eps)) > 1e-12:
        mismatches.append(f"physical_eps: expected {float(physical_eps)!r}, got {metadata_eps!r}")
    if mismatches:
        details = "; ".join(mismatches[:8])
        raise ValueError(
            f"--reuse-cache-dir {cache_path} is not compatible with this physical xPass run. "
            f"{details}. Regenerate that cache with source={PHYSICAL_XPASS_SOURCE!r}, "
            f"teammate_policy={teammate_policy!r}, speed_aggregation={speed_aggregation!r}, "
            f"and physical_eps={float(physical_eps)!r}."
        )
    return cache_path


def load_reuse_rows(cache_dir: Path | None, match_id: str) -> pd.DataFrame | None:
    if cache_dir is None:
        return None
    try:
        return load_physical_xpass_match(cache_dir, match_id)
    except FileNotFoundError:
        return None


def compute_match_rows(
    match_id: str,
    graph_path: Path,
    label_path: Path,
    *,
    eps: float,
    normalize: bool,
    consider_teammates: bool,
    speed_aggregation: str = PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    physical_batch_size: int = 16,
    limit: int | None,
    selected_row_indices: set[int] | None = None,
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
    pending: list[tuple[int, int, object, str]] = []
    computed = 0
    iterator = range(int(labels.shape[0]))
    if selected_row_indices is not None:
        selected = {int(row_index) for row_index in selected_row_indices}
        iterator = [row_index for row_index in iterator if row_index in selected]
    iterator = tqdm(iterator, desc=f"physical_xpass {match_id}", leave=False) if show_progress else iterator
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
        reused = False
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
                reuse_stats["hash_mismatch_recomputed"] = reuse_stats.get("hash_mismatch_recomputed", 0) + 1

            if reused:
                row = reuse_row.to_dict()
                row["match_id"] = str(match_id)
                row["action_index"] = action_index
                if pd.isna(row.get("physical_state_hash", None)):
                    row["physical_state_hash"] = current_hash
                rows.append(row)
                reuse_stats["reused_actions"] = reuse_stats.get("reused_actions", 0) + 1
                continue

        if limit is not None and computed >= limit:
            reuse_stats["compute_limit_skipped"] = reuse_stats.get("compute_limit_skipped", 0) + 1
            continue

        pending.append((int(row_index), action_index, graph, current_hash))
        computed += 1

    if pending:
        use_batch_fn = compute_batch_fn is not None and compute_fn is compute_graph_max_player_cum_prob_as_defaults
        if use_batch_fn:
            probs_list = compute_batch_fn(
                [item[2] for item in pending],
                eps=eps,
                consider_teammates=consider_teammates,
                speed_aggregation=speed_aggregation,
                batch_size=physical_batch_size,
            )
        else:
            probs_list = [
                compute_fn(
                    item[2],
                    eps=eps,
                    consider_teammates=consider_teammates,
                    speed_aggregation=speed_aggregation,
                )
                for item in pending
            ]
        if len(probs_list) != len(pending):
            raise ValueError(f"Physical xPass batch compute returned {len(probs_list)} rows for {len(pending)} graphs.")
        for (_, action_index, _graph, current_hash), probs in zip(pending, probs_list):
            row = {
                "match_id": str(match_id),
                "action_index": action_index,
                "physical_state_hash": current_hash,
            }
            row.update({str(player_id): float(value) for player_id, value in probs.items()})
            rows.append(row)

    return pd.DataFrame(rows), computed


def process_match_task(task: dict[str, object]) -> dict[str, object]:
    match_id = str(task["match_id"])
    graph_path = Path(str(task["graph_path"]))
    label_path = Path(str(task["label_path"]))
    output_path = Path(str(task["output_path"]))
    reuse_cache_dir = task.get("reuse_cache_dir")
    reuse_cache_path = Path(str(reuse_cache_dir)) if reuse_cache_dir else None
    reuse_stats: dict[str, int] = {}
    reuse_rows = load_reuse_rows(reuse_cache_path, match_id)
    frame, computed = compute_match_rows(
        match_id,
        graph_path,
        label_path,
        eps=float(task["eps"]),
        normalize=True,
        consider_teammates=bool(task["consider_teammates"]),
        speed_aggregation=str(task["speed_aggregation"]),
        physical_batch_size=int(task["physical_batch_size"]),
        limit=task.get("limit"),
        reuse_rows=reuse_rows,
        reuse_stats=reuse_stats,
        show_progress=bool(task.get("show_progress", False)),
    )
    reused = int(reuse_stats.get("reused_actions", 0))
    if frame.empty:
        return {
            "match_id": match_id,
            "computed": int(computed),
            "reused": int(reused),
            "written": False,
            "skip_reason": "no_pass_actions",
            "reuse_stats": reuse_stats,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "match_id": match_id,
        "computed": int(computed),
        "reused": int(reused),
        "written": True,
        "skip_reason": None,
        "reuse_stats": reuse_stats,
    }


def main() -> None:
    args = parse_args()
    if not args.normalize:
        warnings.warn("--no-normalize is ignored; AS-default physical xPass always uses normalize=True.")
    speed_aggregation = normalize_physical_xpass_speed_aggregation(args.speed_aggregation)
    num_workers = resolve_num_workers(args.num_workers)
    if args.limit is not None and num_workers > 1:
        raise ValueError("--limit cannot be used with --num-workers > 1. Use --num-workers 1 for limited smoke tests.")
    configure_worker_thread_limit(int(args.worker_thread_limit))
    teammate_policy = teammate_policy_from_args(args)
    feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(feature_run_id)
    graph_dir = get_action_graph_dir(feature_root)
    label_dir = resolve_reference_label_dir(feature_run_id, feature_root, args)
    output_root = get_physical_xpass_dir(feature_root)
    match_ids = resolve_match_ids(args, graph_dir)
    reuse_cache_dir = (
        validate_reuse_cache_dir(
            args.reuse_cache_dir,
            teammate_policy=teammate_policy,
            speed_aggregation=speed_aggregation,
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
    written_match_ids = []
    skipped_match_ids: dict[str, str] = {}
    tasks: list[dict[str, object]] = []
    for match_id in tqdm(match_ids, desc="matches"):
        if args.limit is not None and total_computed >= args.limit and reuse_cache_dir is None:
            break

        output_path = get_physical_xpass_match_path(match_id, root=feature_root)
        if output_path.exists() and not args.overwrite:
            skipped_match_ids[match_id] = "exists"
            continue

        graph_path = graph_dir / f"{match_id}.pt"
        label_path = label_dir / f"{match_id}.pt"
        if not graph_path.exists() or not label_path.exists():
            skipped_match_ids[match_id] = "missing_graph_or_label"
            continue

        remaining = None if args.limit is None else args.limit - total_computed
        tasks.append(
            {
                "match_id": match_id,
                "graph_path": str(graph_path),
                "label_path": str(label_path),
                "output_path": str(output_path),
                "eps": float(args.physical_eps),
                "consider_teammates": bool(args.consider_teammates),
                "speed_aggregation": speed_aggregation,
                "physical_batch_size": int(args.physical_batch_size),
                "limit": remaining,
                "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir is not None else None,
                "show_progress": num_workers == 1,
            }
        )
        if args.limit is not None:
            result = process_match_task(tasks.pop())
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
            task_iter = (process_match_task(task) for task in tasks)
            for result in tqdm(task_iter, total=len(tasks), desc="compute matches"):
                match_id = str(result["match_id"])
                for key, value in result["reuse_stats"].items():
                    reuse_stats[key] = reuse_stats.get(key, 0) + int(value)
                if result["written"]:
                    total_computed += int(result["computed"])
                    total_reused += int(result["reused"])
                    written_match_ids.append(match_id)
                else:
                    skipped_match_ids[match_id] = str(result["skip_reason"])
        else:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=configure_worker_thread_limit,
                initargs=(int(args.worker_thread_limit),),
            ) as executor:
                futures = [executor.submit(process_match_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="compute matches"):
                    result = future.result()
                    match_id = str(result["match_id"])
                    for key, value in result["reuse_stats"].items():
                        reuse_stats[key] = reuse_stats.get(key, 0) + int(value)
                    if result["written"]:
                        total_computed += int(result["computed"])
                        total_reused += int(result["reused"])
                        written_match_ids.append(match_id)
                    else:
                        skipped_match_ids[match_id] = str(result["skip_reason"])

    metadata = {
        **physical_xpass_as_default_metadata(teammate_policy, speed_aggregation=speed_aggregation),
        "feature_run_id": feature_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "graph_dir": str(graph_dir),
        "label_dir": str(label_dir),
        "split": args.split,
        "reuse_cache_dir": str(reuse_cache_dir) if reuse_cache_dir is not None else None,
        "match_ids": written_match_ids,
        "skipped_match_ids": skipped_match_ids,
        "n_actions": int(total_computed + total_reused),
        "n_computed_actions": int(total_computed),
        "n_reused_actions": int(total_reused),
        "n_reused_without_state_hash": int(reuse_stats.get("reused_without_state_hash", 0)),
        "n_hash_verified": int(reuse_stats.get("hash_verified", 0)),
        "n_hash_mismatch_recomputed": int(reuse_stats.get("hash_mismatch_recomputed", 0)),
        "reuse_stats": {key: int(value) for key, value in sorted(reuse_stats.items())},
        "physical_eps": float(args.physical_eps),
        "num_workers": int(num_workers),
        "physical_batch_size": int(args.physical_batch_size),
        "worker_thread_limit": int(args.worker_thread_limit),
        "deprecated_normalize_requested": bool(args.normalize),
        "storage": "wide_parquet_one_row_per_action_player_id_columns",
    }
    write_run_metadata(output_root, metadata)
    print(
        "Wrote physical xPass sidecars for "
        f"{len(written_match_ids)} match(es), {total_computed} computed and {total_reused} reused pass action(s)."
    )
    print(f"Metadata: {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
