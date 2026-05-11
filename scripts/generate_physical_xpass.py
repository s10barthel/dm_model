from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from datatools.config import LABEL_INDEX
from physical_pass_model import (
    PHYSICAL_XPASS_SOURCE,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
    PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    compute_graph_max_player_cum_prob_as_defaults,
    load_physical_xpass_match,
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


def validate_reuse_cache_dir(
    cache_dir: str | Path,
    *,
    teammate_policy: str,
    physical_eps: float,
) -> Path:
    cache_path = Path(cache_dir)
    metadata = validate_physical_xpass_cache_metadata(cache_path)
    expected_metadata = physical_xpass_as_default_metadata(teammate_policy)
    mismatches: list[str] = []
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            mismatches.append(f"{key}: expected {expected_value!r}, got {metadata.get(key)!r}")
    metadata_eps = metadata.get("physical_eps")
    if metadata_eps is None or abs(float(metadata_eps) - float(physical_eps)) > 1e-12:
        mismatches.append(f"physical_eps: expected {float(physical_eps)!r}, got {metadata_eps!r}")
    if mismatches:
        details = "; ".join(mismatches[:8])
        raise ValueError(
            f"--reuse-cache-dir {cache_path} is not compatible with this physical xPass run. "
            f"{details}. Regenerate that cache with source={PHYSICAL_XPASS_SOURCE!r}, "
            f"teammate_policy={teammate_policy!r}, and physical_eps={float(physical_eps)!r}."
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
    limit: int | None,
    reuse_rows: pd.DataFrame | None = None,
    reuse_stats: dict[str, int] | None = None,
    compute_fn=compute_graph_max_player_cum_prob_as_defaults,
) -> tuple[pd.DataFrame, int]:
    del normalize
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
    computed = 0
    for row_index in tqdm(range(int(labels.shape[0])), desc=f"physical_xpass {match_id}", leave=False):
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

        probs = compute_fn(
            graph,
            eps=eps,
            consider_teammates=consider_teammates,
        )
        row = {
            "match_id": str(match_id),
            "action_index": action_index,
            "physical_state_hash": current_hash,
        }
        row.update({str(player_id): float(value) for player_id, value in probs.items()})
        rows.append(row)
        computed += 1

    return pd.DataFrame(rows), computed


def main() -> None:
    args = parse_args()
    if not args.normalize:
        warnings.warn("--no-normalize is ignored; AS-default physical xPass always uses normalize=True.")
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

        reuse_rows = load_reuse_rows(reuse_cache_dir, match_id)
        before_reused = reuse_stats.get("reused_actions", 0)
        remaining = None if args.limit is None else args.limit - total_computed
        frame, computed = compute_match_rows(
            match_id,
            graph_path,
            label_path,
            eps=args.physical_eps,
            normalize=args.normalize,
            consider_teammates=bool(args.consider_teammates),
            limit=remaining,
            reuse_rows=reuse_rows,
            reuse_stats=reuse_stats,
        )
        reused = reuse_stats.get("reused_actions", 0) - before_reused
        if frame.empty:
            skipped_match_ids[match_id] = "no_pass_actions"
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        total_computed += computed
        total_reused += reused
        written_match_ids.append(match_id)

    metadata = {
        **physical_xpass_as_default_metadata(teammate_policy),
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
