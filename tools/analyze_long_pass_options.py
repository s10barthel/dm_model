"""Compare observed and modelled pass-success rates for long passing options."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datatools import config
from models.utils import adapt_batch_graphs_for_model, load_model, load_splits


def wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(centre - half), float(centre + half)]


def summarise(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    values_np = np.asarray(values, dtype=float)
    return {
        "n": int(values_np.size),
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "p10": float(np.percentile(values_np, 10)),
        "p90": float(np.percentile(values_np, 90)),
    }


def iter_long_candidate_graphs(match_ids, feature_dir: Path, label_dir: Path, threshold: float):
    """Yield pass graphs that have at least one >threshold teammate option.

    This performs only the pass-success row selection before batching; model-specific
    edge adaptation remains in the scoring path below.
    """
    for match_id in match_ids:
        graphs = torch.load(feature_dir / f"{match_id}.pt", weights_only=False)
        labels = torch.load(label_dir / f"{match_id}.pt", weights_only=False)
        for graph, label in zip(graphs, labels):
            if graph is None or int(label[1].item()) != 1 or int(label[config.LABEL_INDEX["intent_index"]].item()) < 0:
                continue
            if float(label[config.LABEL_INDEX["duration"]].item()) < 0.5:
                continue
            teammate = graph.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
            possessor = graph.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
            distance = graph.x[:, config.NODE_FEATURE_POSS_DIST]
            target = int(label[config.LABEL_INDEX["intent_index"]].item())
            if target >= graph.x.shape[0] or not bool(teammate[target].item()) or bool(possessor[target].item()):
                continue
            has_long_option = bool((teammate & ~possessor & torch.isfinite(distance) & (distance > threshold)).any().item())
            if not has_long_option:
                continue
            yield graph, label


def analyse_split(name: str, match_ids, feature_dir: Path, label_dir: Path, model, device: str, batch_size: int, threshold: float) -> dict:
    all_option_probs: list[float] = []
    observed_probs: list[float] = []
    observed_successes = 0
    observed_count = 0
    observed_offside = 0
    candidate_offside = 0
    candidate_count_before_offside = 0

    pending_graphs = []
    pending_labels = []

    def score_pending():
        nonlocal candidate_count_before_offside, candidate_offside, observed_offside, observed_count, observed_successes
        if not pending_graphs:
            return
        graphs = Batch.from_data_list(pending_graphs).to(device)
        labels = torch.stack(pending_labels).to(device)
        graphs = adapt_batch_graphs_for_model(graphs, model.args, context="long-pass analysis")
        with torch.no_grad():
            probs = torch.sigmoid(model(graphs))

        teammate = graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
        possessor = graphs.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
        distances = graphs.x[:, config.NODE_FEATURE_POSS_DIST]
        long_mask = teammate & ~possessor & torch.isfinite(distances) & (distances > threshold)
        offside = graphs.x[:, -1].bool() if bool(model.args.get("offside_aware", False)) else torch.zeros_like(long_mask)

        candidate_count_before_offside += int(long_mask.sum().item())
        candidate_offside += int((long_mask & offside).sum().item())
        onside_long = long_mask & ~offside
        all_option_probs.extend(probs[onside_long].detach().cpu().numpy().astype(float).tolist())

        target_global = graphs.ptr[:-1] + labels[:, config.LABEL_INDEX["intent_index"]].long()
        target_distance = distances[target_global]
        target_long = torch.isfinite(target_distance) & (target_distance > threshold)
        target_offside = offside[target_global]
        observed_offside += int((target_long & target_offside).sum().item())
        observed_onside_long = target_long & ~target_offside
        if observed_onside_long.any():
            target_probs = probs[target_global[observed_onside_long]]
            observed_probs.extend(target_probs.detach().cpu().numpy().astype(float).tolist())
            observed_count += int(observed_onside_long.sum().item())
            observed_successes += int(
                labels[observed_onside_long, config.LABEL_INDEX["success"]].sum().item()
            )

    for graph, label in iter_long_candidate_graphs(match_ids, feature_dir, label_dir, threshold):
        pending_graphs.append(graph)
        pending_labels.append(label)
        if len(pending_graphs) >= batch_size:
            score_pending()
            pending_graphs.clear()
            pending_labels.clear()
    score_pending()

    observed_rate = observed_successes / observed_count if observed_count else None
    return {
        "split": name,
        "observed_selected_onside_long_passes": {
            "n": observed_count,
            "successful": observed_successes,
            "empirical_success_rate": observed_rate,
            "wilson_95_ci": wilson_interval(observed_successes, observed_count),
            "model_probability": summarise(observed_probs),
            "calibration_gap_model_minus_observed": (
                float(np.mean(observed_probs) - observed_rate) if observed_count else None
            ),
            "offside_excluded": observed_offside,
        },
        "all_onside_long_options": summarise(all_option_probs),
        "candidate_options": {
            "long_before_offside_filter": candidate_count_before_offside,
            "offside_excluded": candidate_offside,
            "onside": len(all_option_probs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = load_model(args.model_id, device).eval()
    train_ids, valid_ids, test_ids = load_splits(feature_dir=args.feature_dir)
    result = {
        "model_id": args.model_id,
        "feature_dir": args.feature_dir,
        "label_dir": args.label_dir,
        "threshold_m": args.threshold,
        "observed_pass_length_definition": "Pre-pass possessor-to-assigned-intended-receiver distance (NODE_FEATURE_POSS_DIST).",
        "device": device,
        "offside_policy": "Exclude offside receiver options; standard inference would assign them probability zero.",
        "splits": [],
    }
    for name, match_ids in (("train", train_ids), ("validation", valid_ids), ("test", test_ids)):
        result["splits"].append(
            analyse_split(name, match_ids, Path(args.feature_dir), Path(args.label_dir), model, device, args.batch_size, args.threshold)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
