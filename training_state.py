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


def add_training_runtime_arguments(parser):
    parser.add_argument("--accumulation-steps", type=positive_int, default=1,
                        help="Physical batches per optimizer update (default: 1).")
    parser.add_argument("--monitoring", choices=("on", "off"), default="off",
                        help="Optional GPU and training telemetry (default: off).")


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


def publish_checkpoint_artifacts(checkpoint, directory):
    """Repair inference files/metadata if a crash interrupted publication."""
    directory = Path(directory)
    # args.json is a convenience export; resume itself reads checkpoint args.
    (directory / "args.json").write_text(json.dumps(checkpoint["args"], indent=2), encoding="utf-8")
    atomic_torch_save(checkpoint["model"], directory / "last_weights.pt")
    if checkpoint["best"] is not None:
        atomic_torch_save(checkpoint["best"]["model"], directory / "best_weights.pt")
    if checkpoint["best_acc_weights"] is not None:
        atomic_torch_save(checkpoint["best_acc_weights"], directory / "best_acc_weights.pt")
    from project_config import write_run_metadata
    write_run_metadata(directory, checkpoint["metadata"])
    # The wrapper's expanding-validation pipeline publishes a completed stage
    # at the model root. Finish that publication even after a native crash.
    if checkpoint["finished"] and checkpoint["args"].get("artifact_stage"):
        import shutil
        for name in ("args.json", "metadata.json", "best_weights.pt", "best_acc_weights.pt", "last_weights.pt"):
            source = directory / name
            if source.exists():
                shutil.copy2(source, directory.parent / name)


def resolve_resume_checkpoint(model_id, saved_root):
    """Resolve task/run or an unambiguous bare run ID; never resolve a bundle."""
    parts = model_id.replace("\\", "/").split("/")
    if len(parts) not in (1, 2) or any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError("--resume-id must be an individual task/run_id or bare run_id.")
    root = Path(saved_root).resolve()
    if len(parts) == 2:
        if parts[0] == "bundles":
            raise ValueError("Bundle resume is not supported; supply an individual model ID.")
        candidates = [root.joinpath(*parts)]
    else:
        candidates = [p / parts[0] for p in root.iterdir() if p.is_dir() and p.name != "bundles" and (p / parts[0]).is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one model run for {model_id!r}; found {len(candidates)}.")
    run = candidates[0]
    direct = run / CHECKPOINT_NAME
    if direct.is_file():
        return direct
    # Expanding-validation stages are individual training artifacts. Resume only
    # the interrupted stage; this does not orchestrate remaining folds/refits.
    stages = sorted(run.glob(f"*/{CHECKPOINT_NAME}"))
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
