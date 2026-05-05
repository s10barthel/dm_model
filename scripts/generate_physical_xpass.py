from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from datatools.config import LABEL_INDEX
from physical_pass_model import PHYSICAL_XPASS_SOURCE, compute_graph_player_cum_prob
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute accessible-space player_cum_prob sidecars for pass actions.")
    parser.add_argument("--feature-run-id", required=True, help="Feature run whose action graphs should be used.")
    parser.add_argument("--match-id", action="append", help="Restrict to one match id. Repeat for multiple matches.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pass actions to compute across selected matches.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing match sidecars.")
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
    parser.add_argument("--physical-eps", type=float, default=1e-4, help="Clamp stored player_cum_prob to [eps, 1-eps].")
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable accessible-space cumulative probability normalization.",
    )
    parser.set_defaults(normalize=True)
    args = parser.parse_args()
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


def compute_match_rows(
    match_id: str,
    graph_path: Path,
    label_path: Path,
    *,
    eps: float,
    normalize: bool,
    limit: int | None,
) -> tuple[pd.DataFrame, int]:
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
        if limit is not None and computed >= limit:
            break
        label = labels[row_index]
        if int(label[LABEL_INDEX["is_pass"]].item()) != 1:
            continue
        graph = graphs[row_index]
        if graph is None:
            continue
        action_index = int(label[LABEL_INDEX["action_index"]].item())
        probs = compute_graph_player_cum_prob(graph, eps=eps, normalize=normalize)
        row = {
            "match_id": str(match_id),
            "action_index": action_index,
        }
        row.update({str(player_id): float(value) for player_id, value in probs.items()})
        rows.append(row)
        computed += 1

    return pd.DataFrame(rows), computed


def main() -> None:
    args = parse_args()
    feature_run_id = resolve_feature_run_id(args.feature_run_id, required=True, allow_latest=False)
    feature_root = resolve_feature_root(feature_run_id)
    graph_dir = get_action_graph_dir(feature_root)
    label_dir = resolve_reference_label_dir(feature_run_id, feature_root, args)
    output_root = get_physical_xpass_dir(feature_root)
    match_ids = resolve_match_ids(args, graph_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matches").mkdir(parents=True, exist_ok=True)

    total_computed = 0
    written_match_ids = []
    skipped_match_ids: dict[str, str] = {}
    for match_id in tqdm(match_ids, desc="matches"):
        if args.limit is not None and total_computed >= args.limit:
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
        frame, computed = compute_match_rows(
            match_id,
            graph_path,
            label_path,
            eps=args.physical_eps,
            normalize=args.normalize,
            limit=remaining,
        )
        if frame.empty:
            skipped_match_ids[match_id] = "no_pass_actions"
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        total_computed += computed
        written_match_ids.append(match_id)

    metadata = {
        "feature_run_id": feature_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": PHYSICAL_XPASS_SOURCE,
        "graph_dir": str(graph_dir),
        "label_dir": str(label_dir),
        "split": args.split,
        "match_ids": written_match_ids,
        "skipped_match_ids": skipped_match_ids,
        "n_actions": int(total_computed),
        "physical_eps": float(args.physical_eps),
        "normalize": bool(args.normalize),
        "storage": "wide_parquet_one_row_per_action_player_id_columns",
    }
    write_run_metadata(output_root, metadata)
    print(f"Wrote physical xPass sidecars for {len(written_match_ids)} match(es), {total_computed} pass action(s).")
    print(f"Metadata: {output_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
