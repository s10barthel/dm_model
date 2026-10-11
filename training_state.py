"""Versioned epoch checkpoints and shared training controls.

Checkpoints are trusted local artifacts (they contain Python RNG state). The
single recovery file owns both current and best state, avoiding multi-file commits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

CHECKPOINT_VERSION = 1
CHECKPOINT_NAME = "training_checkpoint.pt"


def environment_versions():
    import platform
    versions = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda}
    for package in ("torch-geometric", "torch-scatter", "torch-sparse", "pyg-lib"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def reduction_factor(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise argparse.ArgumentTypeError("must be finite and strictly between 0 and 1")
    return result


def random_seed(value):
    result = int(value)
    if not 0 <= result <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be an integer from 0 through 2**32 - 1")
    return result


def add_training_control_arguments(parser, *, wrapper=False):
    parser.add_argument("--lr-reduction-patience", type=positive_int, default=3,
                        help="Validation-loss misses before reducing the learning rate (default: 3).")
    parser.add_argument("--lr-reduction-factor", type=reduction_factor, default=0.5,
                        help="Multiplier on learning-rate reduction (default: 0.5).")
    parser.add_argument("--early-stopping-min-epochs-after-lr-reduction", type=nonnegative_int, default=5,
                        help="Completed epochs after the latest LR reduction before early stopping (default: 5; 0 disables).")
    rollback = parser.add_mutually_exclusive_group()
    rollback.add_argument("--lr-rollback", dest="lr_rollback", action="store_true",
                          help="Restore best model and Adam state on LR reduction (default).")
    rollback.add_argument("--no-lr-rollback", dest="lr_rollback", action="store_false",
                          help="Reduce LR without restoring model or Adam state.")
    parser.set_defaults(lr_rollback=True)
    if wrapper:
        parser.add_argument("--max-epochs", type=positive_int, default=100,
                            help="Hard epoch cap per validation stage (default: 100).")
    else:
        epochs = parser.add_mutually_exclusive_group()
        epochs.add_argument("--n_epochs", dest="n_epochs", type=positive_int, help="Hard epoch cap (default: 200).")
        epochs.add_argument("--max-epochs", dest="n_epochs", type=positive_int, help="Alias for --n_epochs.")
        parser.set_defaults(n_epochs=200)
    parser.add_argument("--seed", type=random_seed, default=100 if wrapper else 128,
                        help="Random seed for Python, NumPy and PyTorch.")


def restore_training_control_defaults(args):
    """Old checkpoint arguments retain the behavior used when they were saved."""
    for name, value in {"lr_reduction_patience": 3, "lr_reduction_factor": 0.5,
                        "lr_rollback": True, "early_stopping_min_epochs_after_lr_reduction": 0}.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def reduce_learning_rate(model, optimizer, best, lr, minimum, factor, rollback):
    new_lr = max(lr * factor, minimum)
    if new_lr >= lr:
        return lr
    if rollback:
        if best is None:
            raise RuntimeError("Missing best model/optimizer state for scheduler rollback.")
        restore_optimizer_snapshot(best, model, optimizer, lr=new_lr)
    else:
        for group in optimizer.param_groups:
            group["lr"] = new_lr
    return new_lr


def lr_reduction_grace_complete(epoch, reduction_start_epoch, grace):
    return reduction_start_epoch is None or epoch - reduction_start_epoch + 1 >= grace


def add_training_runtime_arguments(parser):
    parser.add_argument("--accumulation-steps", type=positive_int, default=1,
                        help="Physical batches per optimizer update (default: 1).")
    parser.add_argument("--monitoring", choices=("on", "off"), default="off",
                        help="Optional GPU/training telemetry and main training/validation graph collation diagnostics (default: off).")


def validate_learning_rates(start, minimum):
    for name, value in (("start_lr", start), ("min_lr", minimum)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name} must be finite and positive.")
    if start is not None and minimum is not None and minimum > start:
        raise ValueError("--min_lr must not exceed --start_lr after resolving model defaults.")


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def optimizer_snapshot(model, optimizer):
    return {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict())}


def restore_optimizer_snapshot(snapshot, model, optimizer, lr=None):
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    if lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = lr


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA device count differs from the current environment.")
        torch.cuda.set_rng_state_all(state["cuda"])


def make_checkpoint(model, optimizer, args, metadata, dataset_identity, best, best_acc_weights, finished=False):
    settings = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    return {"version": CHECKPOINT_VERSION, **optimizer_snapshot(model, optimizer),
            "args": cpu_copy(settings), "metadata": cpu_copy(metadata),
            "rng": capture_rng(), "dataset_identity": dataset_identity,
            "best": cpu_copy(best), "best_acc_weights": cpu_copy(best_acc_weights),
            "finished": bool(finished)}


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"No supported training checkpoint at {path}. Legacy weights-only checkpoints cannot be resumed.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"version", "model", "optimizer", "args", "metadata", "rng", "dataset_identity", "best", "best_acc_weights", "finished"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint) or checkpoint["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported training checkpoint format: {path}")
    return checkpoint


def publish_stage_artifacts(directory):
    """Publish the latest completed stage, removing stale optional root weights."""
    import shutil
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    stage = metadata.get("artifact_stage")
    if metadata.get("status") != "completed" or not stage:
        return
    root_metadata_path = directory.parent / "metadata.json"
    def order(name):
        if name == "final_refit":
            return float("inf")
        return int(name[5:]) if name and name.startswith("fold_") and name[5:].isdigit() else 0
    if root_metadata_path.is_file():
        root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
        if root_metadata.get("status") == "completed" and order(root_metadata.get("artifact_stage")) > order(stage):
            return
    for name in ("args.json", "metadata.json", "best_weights.pt", "best_acc_weights.pt", "last_weights.pt"):
        source, target = directory / name, directory.parent / name
        if source.exists():
            shutil.copy2(source, target)
        elif name == "best_acc_weights.pt":
            target.unlink(missing_ok=True)


def publish_checkpoint_artifacts(checkpoint, directory):
    """Repair interrupted publication while keeping completed stage files immutable."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    required = ["args.json", "metadata.json", "last_weights.pt"]
    if checkpoint["best"] is not None:
        required.append("best_weights.pt")
    if checkpoint["best_acc_weights"] is not None:
        required.append("best_acc_weights.pt")
    published = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    immutable = (checkpoint["finished"] and checkpoint["args"].get("artifact_stage")
                 and published.get("status") == "completed"
                 and published.get("last_epoch") == checkpoint["metadata"].get("last_epoch")
                 and all((directory / name).is_file() for name in required))
    if not immutable:
        # args.json is a convenience export; resume itself reads checkpoint args.
        (directory / "args.json").write_text(json.dumps(checkpoint["args"], indent=2), encoding="utf-8")
        atomic_torch_save(checkpoint["model"], directory / "last_weights.pt")
        if checkpoint["best"] is not None:
            atomic_torch_save(checkpoint["best"]["model"], directory / "best_weights.pt")
        if checkpoint["best_acc_weights"] is not None:
            atomic_torch_save(checkpoint["best_acc_weights"], directory / "best_acc_weights.pt")
        from project_config import write_run_metadata
        write_run_metadata(directory, checkpoint["metadata"])
    if checkpoint["finished"] and checkpoint["args"].get("artifact_stage"):
        publish_stage_artifacts(directory)


def resolve_resume_checkpoint(model_id, saved_root):
    """Resolve task/run[/stage] or an unambiguous bare run ID; never a bundle."""
    parts = model_id.replace("\\", "/").split("/")
    if len(parts) not in (1, 2, 3) or any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError("--resume-id must be task/run_id[/stage] or a bare run_id.")
    root = Path(saved_root).resolve()
    if len(parts) >= 2:
        if parts[0] in ("bundles", "refit_batches"):
            raise ValueError("Bundle resume is not supported; supply an individual model ID.")
        candidates = [root.joinpath(*parts)]
    else:
        candidates = [p / parts[0] for p in root.iterdir() if p.is_dir() and p.name not in ("bundles", "refit_batches") and (p / parts[0]).is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one model run for {model_id!r}; found {len(candidates)}.")
    run = candidates[0]
    direct = run / CHECKPOINT_NAME
    if direct.is_file():
        return direct
    # Expanding-validation stages are individual training artifacts. Resume only
    # the interrupted stage; this does not orchestrate remaining folds/refits.
    stages = sorted(run.glob(f"*/{CHECKPOINT_NAME}"),
                    key=lambda p: int(p.parent.name[5:]) if p.parent.name.startswith("fold_") and p.parent.name[5:].isdigit() else 0)
    staged = [p for p in stages if not load_checkpoint(p)["finished"]]
    if len(staged) == 1:
        return staged[0]
    if len(staged) > 1:
        raise ValueError("Multiple unfinished model stages found; cannot choose one safely.")
    if stages:
        return next((p for p in stages if p.parent.name == "final_refit"), stages[-1])
    raise ValueError(f"No supported training checkpoint in {run}. Legacy checkpoints cannot be resumed.")


def dataset_signature(args):
    """Record source file identity without rereading the complete graph corpus."""
    paths = set()
    for key, value in vars(args).items():
        if value and isinstance(value, str) and (key.endswith("feature_dir") or key.endswith("label_dir")
                                                  or key in ("physical_cache_dir", "lane_survival_cache_dir")):
            paths.add(Path(value).resolve())
    entries = []
    for directory in sorted(paths):
        if not directory.is_dir():
            entries.append((str(directory), None))
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            stat = path.stat()
            entries.append((str(path), stat.st_size, stat.st_mtime_ns))
    # Also cover the model used to construct inverse-propensity sample weights.
    dependency = getattr(args, "ipw_model_id", "none")
    if dependency != "none":
        path = Path(__file__).parent / "saved" / dependency / "best_weights.pt"
        stat = path.stat()
        entries.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    payload = {"files": entries, "split": getattr(args, "split_manifest", None)}
    if getattr(args, "goal_context_version", 1) == 2:
        from models.goal_context import goal_policy
        payload["goal_context"] = goal_policy(args).settings()
    if getattr(args, "pass_height_definition", None) is not None:
        payload["pass_height_definition"] = args.pass_height_definition
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class GradientAccumulator:
    """Sample-weighted gradient means, including a short final update group."""
    def __init__(self, optimizer, parameters, steps, clip):
        self.optimizer, self.parameters = optimizer, list(parameters)
        self.steps, self.clip = positive_int(steps), clip
        self.samples = self.batches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, mean_loss, samples):
        (mean_loss * samples).backward()
        self.samples += samples
        self.batches += 1
        if self.batches == self.steps:
            self.flush()

    def flush(self):
        if not self.samples:
            return
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self.samples)
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.samples = self.batches = 0
