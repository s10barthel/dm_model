import argparse
import faulthandler
import json
import os
import subprocess
import sys
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
    extract_model_feature_signature,
    get_args_str,
    get_losses_str,
    infer_training_edge_schema,
    estimate_propensity,
    is_validation_loss_improved,
    load_splits,
    num_trainable_params,
    printlog,
    run_epoch,
    should_stop_early,
    unwrap_model,
    validate_target_flags,
)
from project_config import (
    DEFAULT_INTENDED_RECEIVER_MODE,
    generate_model_run_id,
    get_action_graph_dir,
    get_action_label_dir,
    get_augmented_feature_dir,
    get_augmented_label_dir,
    get_model_run_root,
    get_success_intent_graph_dir,
    get_success_intent_label_dir,
    infer_target_family,
    resolve_effective_return_type,
    resolve_feature_root,
    resolve_feature_run_id,
    write_run_metadata,
)

parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str, required=True)
parser.add_argument("--trial", type=int, required=False, default=None)
parser.add_argument("--run-id", type=str, default=None, help="Checkpoint run id. Auto-generated when omitted.")
parser.add_argument("--resume-run-id", type=str, default=None, help="Resume an existing checkpoint run id.")
parser.add_argument("--device", type=str, default=None, help="Training device. Defaults to cuda when available, otherwise cpu.")
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
accel_group = parser.add_mutually_exclusive_group()
accel_group.add_argument(
    "--accel",
    dest="accel_aware",
    action="store_true",
    help="Include player-acceleration node features.",
)
accel_group.add_argument(
    "--no-accel",
    dest="accel_aware",
    action="store_false",
    help="Ignore player-acceleration node features and zero that feature slot.",
)
parser.set_defaults(accel_aware=True)
parser.add_argument("--extend_features", action="store_true", default=False, help="handcraft more node features")

parser.add_argument("--more_dest_features", action="store_true", default=False, help="handcraft more dest features")
parser.add_argument("--adjust_dest", action="store_true", default=False, help="adjust destinations of failed actions")
parser.add_argument("--normalize_dest", action="store_true", default=False, help="normalize action destinations")
parser.add_argument("--polar_dest", action="store_true", default=False, help="use polar coordinates for destinations")
parser.add_argument("--dest_sigma", type=float, default=3.0, help="sigma for smoothing target dest distribution")

parser.add_argument("--use_xg", action="store_true", default=False, help="use xG instead of actual goal labels")
parser.add_argument("--use_xt", action="store_true", default=False, help="use xT instead of xG or actual goal labels")
parser.add_argument(
    "--use_goal_distance",
    action="store_true",
    default=False,
    help="use goal-distance labels instead of xG, xT, EPV, or actual goal labels",
)
parser.add_argument("--use_epv", action="store_true", default=False, help="use EPV labels instead of xG, xT, goal-distance, or actual goal labels")
parser.add_argument(
    "--return_type",
    type=str,
    required=False,
    default=None,
    help=(
        "way of defining return: disc_<gamma>, disc_<gamma>_skip1, next_<N>, next_<N>_skip1, "
        "or in_<N> (xt/goal_distance/epv only)"
    ),
)
parser.add_argument("--include_out", action="store_true", default=False, help="attach a component for ball out of play")
parser.add_argument("--filter_blockers", action="store_true", default=False, help="only include potential blockers")
parser.add_argument("--sparsify", type=str, choices=["distance", "delaunay", "none"], help="how to filter edges")
parser.add_argument("--max_edge_dist", type=int, default=10, help="max distance between off-ball nodes")
parser.add_argument("--feature_run_id", "--feature-run-id", dest="feature_run_id", type=str, default=None, help="Pinned feature-artifact run id.")
parser.add_argument("--intended-receiver-mode", type=str, default="unknown", help="Resolved intended-receiver mode.")
parser.add_argument("--label-source", type=str, default=None, help="Optional label provenance descriptor saved with the checkpoint.")
parser.add_argument("--training-filter", type=str, default=None, help="Optional training-filter descriptor saved with the checkpoint.")
parser.add_argument("--feature_dir", type=str, default=None, help="override graph feature directory")
parser.add_argument("--label_dir", type=str, default=None, help="override label directory for evaluation/inference")
parser.add_argument("--train_feature_dir", type=str, default=None, help="feature directory used for training")
parser.add_argument("--train_label_dir", type=str, default=None, help="label directory used for training")
parser.add_argument("--valid_feature_dir", type=str, default=None, help="feature directory used for validation")
parser.add_argument("--valid_label_dir", type=str, default=None, help="label directory used for validation")
parser.add_argument("--ipw_feature_dir", type=str, default=None, help="feature directory used for IPW estimation")
edge_feature_group = parser.add_mutually_exclusive_group()
edge_feature_group.add_argument(
    "--v-edge-features",
    dest="use_v_edge_features",
    action="store_true",
    help="Use the stored velocity-angle edge features during training.",
)
edge_feature_group.add_argument(
    "--no-v-edge-features",
    dest="use_v_edge_features",
    action="store_false",
    help="Ignore the stored velocity-angle edge features and use only the base edge features.",
)
parser.set_defaults(use_v_edge_features=True)
pin_memory_group = parser.add_mutually_exclusive_group()
pin_memory_group.add_argument(
    "--pin-memory",
    dest="pin_memory",
    action="store_true",
    help="Pin host tensors before CUDA transfer.",
)
pin_memory_group.add_argument(
    "--no-pin-memory",
    dest="pin_memory",
    action="store_false",
    help="Avoid pinned host-memory transfers.",
)
parser.set_defaults(pin_memory=False)

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
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=10,
    help="Stop after this many consecutive validation-loss misses once min epochs are complete.",
)
parser.add_argument(
    "--early-stopping-min-epochs",
    type=int,
    default=30,
    help="Minimum epoch before early stopping can terminate training.",
)
parser.add_argument(
    "--early-stopping-min-delta",
    type=float,
    default=1e-5,
    help="Minimum validation-loss improvement required to reset early-stopping patience.",
)
parser.add_argument(
    "--no-early-stopping",
    dest="early_stopping",
    action="store_false",
    help="Disable validation-loss early stopping.",
)
parser.set_defaults(early_stopping=True)

parser.add_argument("--cont", action="store_true", default=False, help="continue training previous best model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best loss")
parser.add_argument("--best_acc", type=float, required=False, default=0, help="best accuracy")
parser.add_argument("--training-step-index", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument("--training-step-total", type=int, default=None, help=argparse.SUPPRESS)

args, _ = parser.parse_known_args()


def infer_node_in_dim(feature_dir: str, task: str) -> int:
    node_in_dim, _ = infer_graph_input_dims(feature_dir)
    return node_in_dim + int(task == "failure_receiver")


def infer_graph_input_dims(feature_dir: str) -> tuple[int, int]:
    feature_path = Path(feature_dir)
    for graph_file in sorted(feature_path.glob("*.pt")):
        try:
            graphs = torch.load(graph_file, weights_only=False)
            if not isinstance(graphs, list):
                continue
            first_graph = next((graph for graph in graphs if graph is not None), None)
            if first_graph is not None:
                node_dim = int(first_graph.x.shape[1])
                edge_dim = int(first_graph.edge_attr.shape[1]) if getattr(first_graph, "edge_attr", None) is not None else 0
                return node_dim, edge_dim
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


def resolve_training_device(requested_device: str | None) -> str:
    if requested_device:
        device = str(requested_device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device {device!r}, but CUDA is not available.")
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_pin_memory(requested_pin_memory: bool | None, device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    if requested_pin_memory is not None:
        return bool(requested_pin_memory)
    return False


def load_existing_metadata(trial_path: str) -> dict:
    metadata_path = Path(trial_path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def infer_last_completed_epoch_from_log(trial_path: str) -> int:
    log_path = Path(trial_path) / "log.txt"
    if not log_path.exists():
        return 0

    current_epoch = 0
    completed_epoch = 0
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("Epoch:"):
                try:
                    current_epoch = int(line.split(":", 1)[1].strip())
                except ValueError:
                    continue
            elif line.startswith("Time:") and current_epoch:
                completed_epoch = current_epoch
    except OSError:
        return 0
    return completed_epoch


def infer_last_lr_from_log(trial_path: str, default_lr: float) -> float:
    log_path = Path(trial_path) / "log.txt"
    if not log_path.exists():
        return default_lr

    lr = float(default_lr)
    try:
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("########## lr") and line.endswith("##########"):
                try:
                    lr = float(line.replace("#", "").strip().split("lr", 1)[1].strip())
                except (IndexError, ValueError):
                    continue
    except OSError:
        return default_lr
    return lr


def checkpoint_path_for_resume(trial_path: str) -> Path:
    for name in ("last_weights.pt", "best_weights.pt"):
        path = Path(trial_path) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No resumable checkpoint found in {trial_path}. Expected last_weights.pt or best_weights.pt.")


def update_training_metadata(trial_path: str, metadata: dict, **updates) -> None:
    metadata.update(updates)
    write_run_metadata(Path(trial_path), metadata)


if __name__ == "__main__":
    # Set device and manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_training_device(args.device)
    args.device = device
    args.pin_memory = resolve_pin_memory(args.pin_memory, device)
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")
    if args.early_stopping_min_epochs < 1:
        raise ValueError("--early-stopping-min-epochs must be at least 1.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    validate_target_flags(args)
    args.target_family = infer_target_family(
        bool(args.use_xg),
        bool(args.use_xt),
        bool(args.use_goal_distance),
        bool(args.use_epv),
    )
    args.return_type = resolve_effective_return_type(args.target_family, args.return_type)

    args.gnn_task = config.TASK_CONFIG.at[args.task, "gnn_task"]
    args.condition = config.TASK_CONFIG.at[args.task, "condition"]
    args.out_dim = config.TASK_CONFIG.at[args.task, "out_dim"]
    if args.resume_run_id:
        if args.run_id and args.run_id != args.resume_run_id:
            raise ValueError("--run-id must match --resume-run-id when both are provided.")
        args.run_id = args.resume_run_id
    args.run_id = args.run_id or (f"{args.trial:02d}" if args.trial is not None else generate_model_run_id(args.task))
    args.model_id = f"{args.task}/{args.run_id}"
    args.feature_run_id = resolve_feature_run_id(args.feature_run_id, required=False)
    feature_root = resolve_feature_root(args.feature_run_id)

    if args.task == "shot_blocking":
        feature_dir = getattr(args, "feature_dir", None) or str(feature_root / "augmented_shot_graphs")
        label_dir = getattr(args, "label_dir", None) or str(feature_root / "augmented_shot_labels")
    elif args.task == "success_intent":
        feature_dir = getattr(args, "feature_dir", None) or str(get_success_intent_graph_dir(feature_root))
        label_dir = getattr(args, "label_dir", None) or str(get_success_intent_label_dir(root=feature_root))
    elif args.task == "failure_receiver" and args.augment_blocks:
        feature_dir = getattr(args, "feature_dir", None) or str(
            get_augmented_feature_dir(DEFAULT_INTENDED_RECEIVER_MODE, root=feature_root)
        )
        label_dir = getattr(args, "label_dir", None) or str(
            get_augmented_label_dir(DEFAULT_INTENDED_RECEIVER_MODE, root=feature_root)
        )
    else:
        feature_dir = getattr(args, "feature_dir", None) or str(get_action_graph_dir(feature_root))
        label_dir = getattr(args, "label_dir", None) or str(
            get_action_label_dir(
                args.return_type,
                intended_receiver_mode=DEFAULT_INTENDED_RECEIVER_MODE,
                root=feature_root,
            )
        )
    args.feature_dir = feature_dir
    args.label_dir = label_dir
    args.train_feature_dir = getattr(args, "train_feature_dir", None) or feature_dir
    args.train_label_dir = getattr(args, "train_label_dir", None) or label_dir
    args.valid_feature_dir = getattr(args, "valid_feature_dir", None) or feature_dir
    args.valid_label_dir = getattr(args, "valid_label_dir", None) or label_dir
    args.ipw_feature_dir = getattr(args, "ipw_feature_dir", None) or feature_dir
    args.node_in_dim = infer_node_in_dim(args.train_feature_dir, args.task)
    _, feature_edge_dim = infer_graph_input_dims(args.train_feature_dir)
    feature_schema = {
        "edge_in_dim": int(feature_edge_dim),
        "add_v_edge_features": bool(feature_edge_dim > 2),
    }
    training_schema = infer_training_edge_schema(feature_schema, use_v_edge_features=bool(args.use_v_edge_features))
    args.edge_in_dim = int(training_schema["edge_in_dim"])
    args.add_v_edge_features = bool(training_schema["add_v_edge_features"])

    # Load model
    args_dict = vars(args)
    model = GNN(args_dict).to(device)
    if str(device).startswith("cuda") and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    args_dict["total_params"] = num_trainable_params(model)
    args_dict["run_id"] = args.run_id
    args_dict["model_id"] = args.model_id

    # Create a path to save model arguments and parameters
    trial_path = str(get_model_run_root(args.task, args.run_id))
    os.makedirs(trial_path, exist_ok=True)
    crash_log_file = open(f"{trial_path}/crash.log", "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=crash_log_file, all_threads=True)
    existing_metadata = load_existing_metadata(trial_path) if args.resume_run_id else {}
    with open(f"{trial_path}/args.json", "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4)

    metadata = {
        "model_id": args.model_id,
        "task": args.task,
        "run_id": args.run_id,
        "trial": args.trial,
        "created_at": existing_metadata.get("created_at", datetime.now().isoformat(timespec="seconds")),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline(sys.argv),
        "resume_run_id": args.resume_run_id,
        "feature_run_id": args.feature_run_id,
        "intended_receiver_mode": None if args.task == "success_intent" and args.intended_receiver_mode == "unknown" else args.intended_receiver_mode,
        "target_family": args.target_family,
        "label_source": args.label_source,
        "training_filter": args.training_filter,
        "resolved_dirs": {
            "feature_dir": args.feature_dir,
            "label_dir": args.label_dir,
            "train_feature_dir": args.train_feature_dir,
            "train_label_dir": args.train_label_dir,
            "valid_feature_dir": args.valid_feature_dir,
            "valid_label_dir": args.valid_label_dir,
            "ipw_feature_dir": args.ipw_feature_dir,
        },
        "feature_signature": extract_model_feature_signature(args_dict),
        "training_args": args_dict,
        "early_stopping": bool(args.early_stopping),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_epochs": int(args.early_stopping_min_epochs),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "crash_log": str(Path(trial_path) / "crash.log"),
        "status": "running",
    }
    for key in (
        "last_epoch",
        "best_loss",
        "best_acc",
        "lr",
        "epochs_since_best",
        "epochs_since_loss_improvement",
        "epochs_since_lr_loss_improvement",
    ):
        if key in existing_metadata:
            metadata[key] = existing_metadata[key]
    write_run_metadata(Path(trial_path), metadata)

    # Continue a previous experiment, or start a new one
    if args.resume_run_id:
        resume_path = checkpoint_path_for_resume(trial_path)
        state_dict = torch.load(resume_path, weights_only=False, map_location=device)
        unwrap_model(model).load_state_dict(state_dict)
        printlog(f"Resumed weights from {resume_path}", trial_path)
    elif args.cont:
        state_dict = torch.load(f"{trial_path}/best_weights.pt", weights_only=False, map_location=device)
        unwrap_model(model).load_state_dict(state_dict)

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
        "accel_aware": args.accel_aware,
        "extend_features": args.extend_features,
        "drop_non_blockers": args.filter_blockers,
        "sparsify": args.sparsify,
        "max_edge_dist": args.max_edge_dist,
        "edge_in_dim": args.edge_in_dim,
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
    #On Windows, num_workers > 0 can cause issues with PyTorch DataLoader, so we set it to 0 for better compatibility. 
    #Adjust as needed for your environment.
    #loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 16, "pin_memory": True}
    loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0, "pin_memory": args.pin_memory}
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

        inverse_propensity = 1 / estimate_propensity(
            ipw_train_dataset,
            model_id=args.ipw_model_id,
            device=device,
            pin_memory=args.pin_memory,
        )
        train_ipw = inverse_propensity / inverse_propensity.mean()
        train_dataset.set_inverse_propensity_weights(train_ipw)

        inverse_propensity = 1 / estimate_propensity(
            ipw_valid_dataset,
            model_id=args.ipw_model_id,
            device=device,
            pin_memory=args.pin_memory,
        )
        valid_ipw = inverse_propensity / inverse_propensity.mean()
        valid_dataset.set_inverse_propensity_weights(valid_ipw)

    train_loader = DataLoader(train_dataset, **loader_args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)

    # Train loop
    default_lr = max(args.start_lr, args.min_lr)
    best_loss = args.best_loss or float(existing_metadata.get("best_loss", 0) or 0)
    best_acc = args.best_acc or float(existing_metadata.get("best_acc", 0) or 0)
    if args.resume_run_id:
        legacy_epochs_since_best = int(existing_metadata.get("epochs_since_best", 0) or 0)
        epochs_since_loss_improvement = int(
            existing_metadata.get("epochs_since_loss_improvement", legacy_epochs_since_best) or 0
        )
        epochs_since_lr_loss_improvement = int(
            existing_metadata.get("epochs_since_lr_loss_improvement", legacy_epochs_since_best) or 0
        )
    else:
        epochs_since_loss_improvement = 0
        epochs_since_lr_loss_improvement = 0
    lr = (
        float(existing_metadata.get("lr", infer_last_lr_from_log(trial_path, default_lr)) or default_lr)
        if args.resume_run_id
        else default_lr
    )
    last_epoch = int(existing_metadata.get("last_epoch", 0) or 0) if args.resume_run_id else 0
    if args.resume_run_id and last_epoch == 0:
        last_epoch = infer_last_completed_epoch_from_log(trial_path)
    start_epoch = min(last_epoch + 1, args.n_epochs + 1) if args.resume_run_id else 1
    update_training_metadata(
        trial_path,
        metadata,
        start_epoch=start_epoch,
        last_epoch=last_epoch,
        best_loss=best_loss,
        best_acc=best_acc,
        lr=lr,
        epochs_since_best=epochs_since_loss_improvement,
        epochs_since_loss_improvement=epochs_since_loss_improvement,
        epochs_since_lr_loss_improvement=epochs_since_lr_loss_improvement,
        stopped_early=False,
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )

    stopped_early = False
    early_stop_epoch = None
    early_stop_reason = None
    for epoch in range(start_epoch, args.n_epochs + 1):
        # Set a custom learning rate schedule
        if epochs_since_lr_loss_improvement >= 3 and lr > args.min_lr and best_loss > 0:
            # Load previous best model
            path = f"{trial_path}/best_weights.pt"
            state_dict = torch.load(path, weights_only=False, map_location=device)
            unwrap_model(model).load_state_dict(state_dict)

            # Decrease learning rate
            lr = max(lr * 0.5, args.min_lr)
            printlog(f"########## lr {lr} ##########", trial_path)
            epochs_since_lr_loss_improvement = 0

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, unwrap_model(model).parameters()), lr=lr)

        if args.training_step_index is not None and args.training_step_total is not None:
            printlog(f"\nTraining model {args.training_step_index}/{args.training_step_total}: {args.task}", trial_path)
            printlog(f"Run id: {args.run_id}", trial_path)
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
        if is_validation_loss_improved(epoch_loss, best_loss, args.early_stopping_min_delta):
            epochs_since_loss_improvement = 0
            epochs_since_lr_loss_improvement = 0
            best_loss = epoch_loss

            torch.save(unwrap_model(model).state_dict(), f"{trial_path}/best_weights.pt")
            printlog("######## Best Loss ########", trial_path)
        else:
            epochs_since_loss_improvement += 1
            epochs_since_lr_loss_improvement += 1

        if "accuracy" in valid_metrics or "f1" in valid_metrics:
            epoch_acc = valid_metrics["accuracy"] if "accuracy" in valid_metrics else valid_metrics["f1"]
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(unwrap_model(model).state_dict(), f"{trial_path}/best_acc_weights.pt")
                printlog("###### Best Accuracy ######", trial_path)

        torch.save(unwrap_model(model).state_dict(), f"{trial_path}/last_weights.pt")
        update_training_metadata(
            trial_path,
            metadata,
            last_epoch=epoch,
            best_loss=best_loss,
            best_acc=best_acc,
            lr=lr,
            epochs_since_best=epochs_since_loss_improvement,
            epochs_since_loss_improvement=epochs_since_loss_improvement,
            epochs_since_lr_loss_improvement=epochs_since_lr_loss_improvement,
            last_epoch_train_metrics=train_metrics,
            last_epoch_valid_metrics=valid_metrics,
            updated_at=datetime.now().isoformat(timespec="seconds"),
            status="running",
        )

        if should_stop_early(
            args.early_stopping,
            epoch,
            args.early_stopping_min_epochs,
            epochs_since_loss_improvement,
            args.early_stopping_patience,
        ):
            stopped_early = True
            early_stop_epoch = epoch
            early_stop_reason = (
                f"validation loss did not improve by at least {args.early_stopping_min_delta:g} "
                f"for {epochs_since_loss_improvement} consecutive epochs"
            )
            printlog(f"######## Early stopping at epoch {epoch}: {early_stop_reason}. ########", trial_path)
            break

    printlog(f"Best loss: {best_loss:.4f}", trial_path)
    update_training_metadata(
        trial_path,
        metadata,
        status="completed",
        best_loss=best_loss,
        best_acc=best_acc,
        epochs_since_best=epochs_since_loss_improvement,
        epochs_since_loss_improvement=epochs_since_loss_improvement,
        epochs_since_lr_loss_improvement=epochs_since_lr_loss_improvement,
        stopped_early=stopped_early,
        early_stop_epoch=early_stop_epoch,
        early_stop_reason=early_stop_reason,
        completed_at=datetime.now().isoformat(timespec="seconds"),
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    faulthandler.disable()
    crash_log_file.close()
