import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from models import utils
from models.utils import get_losses_str, load_splits, run_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="task/trial, e.g., pass_success/01")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--feature_dir", type=str, required=False, default=None)
    args, _ = parser.parse_known_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)

    print("\nGenerating test datasets...")
    feature_dir = args.feature_dir or getattr(model_args, "feature_dir", "data/ajax/features/action_graphs")
    label_dir = f"data/ajax/features/action_labels_{model_args.return_type}"

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
        "train": False,
    }
    test_dataset = ActionDataset(test_match_ids, **dataset_args)
    test_loader = DataLoader(test_dataset, model_args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    print(f"Evaluating {args.model_id} on {len(test_match_ids)} matches with {len(test_dataset)} samples")
    test_metrics = run_epoch(model_args, model, test_loader, device=device, train=False)
    print("Test:\t" + get_losses_str(test_metrics))
