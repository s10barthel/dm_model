import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from datatools import config
from datatools.config import LABEL_INDEX
from models.gnn import GNN
from models.utils import (
    validate_target_flags,
    load_splits,
    estimate_propensity,
    get_args_str,
    get_losses_str,
    num_trainable_params,
    printlog,
    run_epoch,
)

parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="gat")
parser.add_argument("--ipw_model_id", type=str, default="none", help="model ID to estimate propensity scores")
parser.add_argument("--weight_bce", action="store_true", default=False, help="use weighted BCE to balance classes")

parser.add_argument("--augment_blocks", action="store_true", default=False, help="include augmented data")
parser.add_argument("--min_pass_dur", type=float, default=0, help="min duration of a valid pass")
parser.add_argument("--shot_success", type=str, required=False, default="unblocked", choices=["goal", "unblocked"])
parser.add_argument("--xy_only", action="store_true", default=False, help="only use xy locations as features")
parser.add_argument("--possessor_aware", action="store_true", default=False, help="use possessor features")
parser.add_argument("--keeper_aware", action="store_true", default=False, help="distinguish keeper & goal nodes")
parser.add_argument("--ball_z_aware", action="store_true", default=False, help="consider the ball height")
parser.add_argument("--poss_vel_aware", action="store_true", default=False, help="consider possessor's velocity")
parser.add_argument("--extend_features", action="store_true", default=False, help="handcraft more node features")

parser.add_argument("--more_dest_features", action="store_true", default=False, help="handcraft more dest features")
parser.add_argument("--adjust_dest", action="store_true", default=False, help="adjust destinations of failed actions")
parser.add_argument("--normalize_dest", action="store_true", default=False, help="normalize action destinations")
parser.add_argument("--polar_dest", action="store_true", default=False, help="use polar coordinates for destinations")
parser.add_argument("--dest_sigma", type=float, default=3.0, help="sigma for smoothing target dest distribution")

parser.add_argument("--use_xg", action="store_true", default=False, help="use xG instead of actual goal labels")
parser.add_argument("--use_xt", action="store_true", default=False, help="use xT instead of xG or actual goal labels")
parser.add_argument("--return_type", type=str, required=False, default="disc_0.9", help="way of defining return")
parser.add_argument("--include_out", action="store_true", default=False, help="attach a component for ball out of play")
parser.add_argument("--filter_blockers", action="store_true", default=False, help="only include potential blockers")
parser.add_argument("--sparsify", type=str, choices=["distance", "delaunay", "none"], help="how to filter edges")
parser.add_argument("--max_edge_dist", type=int, default=10, help="max distance between off-ball nodes")
parser.add_argument("--feature_dir", type=str, default=None, help="override graph feature directory")
parser.add_argument("--label_dir", type=str, default=None, help="override label directory for evaluation/inference")
parser.add_argument("--train_feature_dir", type=str, default=None, help="feature directory used for training")
parser.add_argument("--train_label_dir", type=str, default=None, help="label directory used for training")
parser.add_argument("--valid_feature_dir", type=str, default=None, help="feature directory used for validation")
parser.add_argument("--valid_label_dir", type=str, default=None, help="label directory used for validation")
parser.add_argument("--ipw_feature_dir", type=str, default=None, help="feature directory used for IPW estimation")

parser.add_argument("--node_emb_dim", type=int, required=False, default=128, help="node embedding dim")
parser.add_argument("--graph_emb_dim", type=int, required=False, default=128, help="graph embedding dim")
parser.add_argument("--mlp_h1_dim", type=int, required=False, default=32, help="MLP 1st hidden dim")
parser.add_argument("--mlp_h2_dim", type=int, required=False, default=8, help="MLP 2nd hidden dim")
parser.add_argument("--gnn_layers", type=int, required=False, default=2, help="num GNN layers")
parser.add_argument("--gnn_heads", type=int, required=False, default=4, help="num heads of GNN layers")
parser.add_argument("--dropout", type=float, required=False, default=0, help="dropout prob")
parser.add_argument("--skip_conn", action="store_true", default=False, help="adopt skip-connection")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument("--lambda_l1", type=float, required=False, default=0, help="coeff of L1 regularizer")
parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="starting learning rate")
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=int, required=False, default=10, help="gradient clipping")
parser.add_argument("--print_freq", type=int, required=False, default=50, help="periodically print performance")
parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")

parser.add_argument("--cont", action="store_true", default=False, help="continue training previous best model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best loss")
parser.add_argument("--best_acc", type=float, required=False, default=0, help="best accuracy")

args, _ = parser.parse_known_args()


def infer_node_in_dim(feature_dir: str, task: str) -> int:
    feature_path = Path(feature_dir)
    for graph_file in sorted(feature_path.glob("*.pt")):
        try:
            graphs = torch.load(graph_file, weights_only=False)
            if not isinstance(graphs, list):
                continue
            first_graph = next((graph for graph in graphs if graph is not None), None)
            if first_graph is not None:
                base_dim = int(first_graph.x.shape[1])
                return base_dim + int(task == "failure_receiver")
        except Exception:
            continue
    raise FileNotFoundError(f"Could not infer node input dimension from {feature_dir}.")


def log_skipped_matches(name: str, dataset: ActionDataset, trial_path: str, max_items: int = 10) -> None:
    skipped = getattr(dataset, "skipped_matches", {})
    if not skipped:
        return

    printlog(f"Skipped {len(skipped)} {name} matches due to unreadable or mismatched artifacts.", trial_path)
    for match_id, reason in list(skipped.items())[:max_items]:
        printlog(f"  {match_id}: {reason}", trial_path)
    if len(skipped) > max_items:
        printlog(f"  ... and {len(skipped) - max_items} more", trial_path)


if __name__ == "__main__":
    # Set device and manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = "cpu"

    validate_target_flags(args)

    args.gnn_task = config.TASK_CONFIG.at[args.task, "gnn_task"]
    args.condition = config.TASK_CONFIG.at[args.task, "condition"]
    args.edge_in_dim = 2
    args.out_dim = config.TASK_CONFIG.at[args.task, "out_dim"]

    if args.task == "shot_blocking":
        feature_dir = "data/ajax/features/augmented_shot_graphs"
        label_dir = "data/ajax/features/augmented_shot_labels"
    elif args.task == "failure_receiver" and args.augment_blocks:
        feature_dir = "data/ajax/features/augmented_graphs"
        label_dir = "data/ajax/features/augmented_labels"
    else:
        feature_dir = getattr(args, "feature_dir", None) or "data/ajax/features/action_graphs"
        label_dir = getattr(args, "label_dir", None) or f"data/ajax/features/action_labels_{args.return_type}"
    args.feature_dir = feature_dir
    args.label_dir = label_dir
    args.train_feature_dir = getattr(args, "train_feature_dir", None) or feature_dir
    args.train_label_dir = getattr(args, "train_label_dir", None) or label_dir
    args.valid_feature_dir = getattr(args, "valid_feature_dir", None) or feature_dir
    args.valid_label_dir = getattr(args, "valid_label_dir", None) or label_dir
    args.ipw_feature_dir = getattr(args, "ipw_feature_dir", None) or feature_dir
    args.node_in_dim = infer_node_in_dim(args.train_feature_dir, args.task)

    # Load model
    args_dict = vars(args)
    model = GNN(args_dict).to(device)
    model = nn.DataParallel(model)
    args_dict["total_params"] = num_trainable_params(model)

    # Create a path to save model arguments and parameters
    trial_path = f"saved/{args.task}/{args.trial:02d}"
    os.makedirs(f"saved/{args.task}", exist_ok=True)
    os.makedirs(trial_path, exist_ok=True)
    with open(f"{trial_path}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Continue a previous experiment, or start a new one
    if args.cont:
        state_dict = torch.load(f"{trial_path}/best_weights.pt", weights_only=False)
        model.module.load_state_dict(state_dict)

    train_match_ids, valid_match_ids, _ = load_splits(feature_dir=feature_dir)

    print("Generating datasets...")
    common_dataset_args = {
        "task": args.task,
        "inplay_only": args.task.split("_")[1] == "receiver" and not args.include_out,
        "min_pass_dur": args.min_pass_dur,
        "shot_success_type": args.shot_success,
        "xy_only": args.xy_only,
        "possessor_aware": args.possessor_aware,
        "keeper_aware": args.keeper_aware,
        "ball_z_aware": args.ball_z_aware,
        "poss_vel_aware": args.poss_vel_aware,
        "extend_features": args.extend_features,
        "drop_non_blockers": args.filter_blockers,
        "sparsify": args.sparsify,
        "max_edge_dist": args.max_edge_dist,
    }
    train_dataset = ActionDataset(
        train_match_ids,
        feature_dir=args.train_feature_dir,
        label_dir=args.train_label_dir,
        **common_dataset_args,
    )
    valid_dataset = ActionDataset(
        valid_match_ids,
        feature_dir=args.valid_feature_dir,
        label_dir=args.valid_label_dir,
        **common_dataset_args,
    )
    log_skipped_matches("training", train_dataset, trial_path)
    log_skipped_matches("validation", valid_dataset, trial_path)
    if len(train_dataset) == 0:
        raise ValueError("No usable training samples remained after loading graph and label artifacts.")
    if len(valid_dataset) == 0:
        raise ValueError("No usable validation samples remained after loading graph and label artifacts.")

    if args.task == "pass_success" and args.weight_bce:
        n_positives = train_dataset.labels[train_dataset.labels[:, LABEL_INDEX["success"]] == 1].shape[0]
        n_negatives = train_dataset.labels[train_dataset.labels[:, LABEL_INDEX["success"]] == 0].shape[0]
        pos_weight = n_negatives / n_positives
    else:
        pos_weight = 1

    loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 16, "pin_memory": True}

    if args.ipw_model_id != "none":
        print("\nCalculating inverse propensity weights...")
        ipw_train_dataset = ActionDataset(
            train_match_ids,
            feature_dir=args.ipw_feature_dir,
            label_dir=args.label_dir,
            **common_dataset_args,
        )
        ipw_valid_dataset = ActionDataset(
            valid_match_ids,
            feature_dir=args.ipw_feature_dir,
            label_dir=args.label_dir,
            **common_dataset_args,
        )
        if len(ipw_train_dataset) == 0 or len(ipw_valid_dataset) == 0:
            raise ValueError("No usable samples remained for inverse-propensity weighting.")

        inverse_propensity = 1 / estimate_propensity(ipw_train_dataset, model_id=args.ipw_model_id, device=device)
        train_ipw = inverse_propensity / inverse_propensity.mean()
        train_dataset.set_inverse_propensity_weights(train_ipw)

        inverse_propensity = 1 / estimate_propensity(ipw_valid_dataset, model_id=args.ipw_model_id, device=device)
        valid_ipw = inverse_propensity / inverse_propensity.mean()
        valid_dataset.set_inverse_propensity_weights(valid_ipw)

    train_loader = DataLoader(train_dataset, **loader_args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Train loop
    best_loss = args.best_loss
    best_acc = args.best_acc
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for epoch in np.arange(args.n_epochs) + 1:
        # Set a custom learning rate schedule
        if epochs_since_best == 3 and lr > args.min_lr:
            # Load previous best model
            path = f"{trial_path}/best_weights.pt"
            state_dict = torch.load(path, weights_only=False)

            # Decrease learning rate
            lr = max(lr * 0.5, args.min_lr)
            printlog(f"########## lr {lr} ##########", trial_path)
            epochs_since_best = 0

        else:
            epochs_since_best += 1

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(f"\nEpoch: {epoch:d}", trial_path)
        start_time = time.time()

        train_metrics = run_epoch(args, model, train_loader, optimizer, device, pos_weight, train=True)
        printlog("Train:\t" + get_losses_str(train_metrics), trial_path)

        valid_metrics = run_epoch(args, model, valid_loader, optimizer, device, pos_weight, train=False)
        printlog("Valid:\t" + get_losses_str(valid_metrics), trial_path)

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time), trial_path)

        epoch_loss = valid_metrics["ce_loss"] if "ce_loss" in valid_metrics else valid_metrics["mse_loss"]

        # Best model on test set
        if best_loss == 0 or epoch_loss < best_loss:
            epochs_since_best = 0
            best_loss = epoch_loss

            torch.save(model.module.state_dict(), f"{trial_path}/best_weights.pt")
            printlog("######## Best Loss ########", trial_path)

        if "accuracy" in valid_metrics or "f1" in valid_metrics:
            epoch_acc = valid_metrics["accuracy"] if "accuracy" in valid_metrics else valid_metrics["f1"]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                if epochs_since_best > 0:
                    epochs_since_best = 0
                    torch.save(model.module.state_dict(), f"{trial_path}/best_acc_weights.pt")
                    printlog("###### Best Accuracy ######", trial_path)

    printlog(f"Best loss: {best_loss:.4f}", trial_path)
