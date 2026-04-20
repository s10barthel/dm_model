import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from models import utils
from models.utils import get_losses_str, infer_feature_graph_schema, load_splits, run_epoch
from models.utils import validate_feature_graph_schema
from project_config import resolve_feature_root


def print_skipped_matches(name: str, dataset: ActionDataset, max_items: int = 10) -> None:
    skipped = getattr(dataset, "skipped_matches", {})
    if not skipped:
        return

    print(f"Skipped {len(skipped)} {name} matches due to unreadable or mismatched artifacts.")
    for match_id, reason in list(skipped.items())[:max_items]:
        print(f"  {match_id}: {reason}")
    if len(skipped) > max_items:
        print(f"  ... and {len(skipped) - max_items} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="task/trial, e.g., pass_success/01")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--feature_dir", type=str, required=False, default=None)
    parser.add_argument("--feature-run-id", type=str, required=False, default=None)
    args, _ = parser.parse_known_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)

    print("\nGenerating test datasets...")
    resolved_feature_run_id = args.feature_run_id or getattr(model_args, "feature_run_id", None)
    if resolved_feature_run_id:
        feature_root = resolve_feature_root(resolved_feature_run_id)
        feature_name = Path(getattr(model_args, "feature_dir", "data/features/action_graphs")).name
        label_name = Path(
            getattr(model_args, "label_dir", f"data/features/action_labels_{model_args.return_type}")
        ).name
        feature_dir = args.feature_dir or str(feature_root / feature_name)
        label_dir = str(feature_root / label_name)
    else:
        feature_dir = args.feature_dir or getattr(model_args, "feature_dir", "data/features/action_graphs")
        label_dir = getattr(model_args, "label_dir", f"data/features/action_labels_{model_args.return_type}")
    feature_schema = infer_feature_graph_schema(feature_dir)
    model_schema = {
        "edge_in_dim": int(getattr(model_args, "edge_in_dim", 2)),
        "add_v_edge_features": bool(getattr(model_args, "add_v_edge_features", getattr(model_args, "edge_in_dim", 2) > 2)),
    }
    validate_feature_graph_schema(feature_schema, model_schema, context="Selected feature artifacts")

    _, _, test_match_ids = load_splits(feature_dir=feature_dir)

    dataset_args = {
        "feature_dir": feature_dir,
        "label_dir": label_dir,
        "task": model_args.task,
        "inplay_only": model_args.task.split("_")[1] == "receiver" and not model_args.include_out,
        "min_pass_dur": model_args.min_pass_dur,
        "shot_success_type": getattr(model_args, "shot_success", "unblocked"),
        "xy_only": model_args.xy_only,
        "possessor_aware": model_args.possessor_aware,
        "keeper_aware": model_args.keeper_aware,
        "ball_z_aware": model_args.ball_z_aware,
        "poss_vel_aware": model_args.poss_vel_aware,
        "extend_features": model_args.extend_features,
        "drop_non_blockers": model_args.filter_blockers,
        "sparsify": model_args.sparsify,
        "max_edge_dist": model_args.max_edge_dist,
        "edge_in_dim": int(getattr(model_args, "edge_in_dim", 2)),
        "train": False,
    }
    test_dataset = ActionDataset(test_match_ids, **dataset_args)
    print_skipped_matches("test", test_dataset)
    if len(test_dataset) == 0:
        raise ValueError("No usable test samples remained after loading graph and label artifacts.")
    
    #On Windows, num_workers > 0 can cause issues with PyTorch DataLoader, so we set it to 0 for better compatibility. 
    #Adjust as needed for your environment.
    #test_loader = DataLoader(test_dataset, model_args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, model_args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Evaluating {args.model_id} on {len(test_match_ids)} matches with {len(test_dataset)} samples")
    test_metrics = run_epoch(model_args, model, test_loader, device=device, train=False)
    print("Test:\t" + get_losses_str(test_metrics))
