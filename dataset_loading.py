"""CLI shared by training entry points (kept independent of torch imports)."""
from pathlib import Path

DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / "data" / "cache" / "intent_datasets")
INTENT_TASKS = {"pass_intent", "action_intent"}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be a positive integer")
    return value


def add_dataset_loading_arguments(parser):
    parser.add_argument("--dataset-loading", choices=("auto", "memory", "disk"), default="auto")
    parser.add_argument("--dataset-buffer-matches", type=positive_int, default=4)
    parser.add_argument("--dataset-cache-dir", default=DEFAULT_CACHE_DIR)


def dataset_loading_flags(args):
    return ["--dataset-loading", getattr(args, "dataset_loading", "auto"),
            "--dataset-buffer-matches", str(getattr(args, "dataset_buffer_matches", 4)),
            "--dataset-cache-dir", str(getattr(args, "dataset_cache_dir", DEFAULT_CACHE_DIR))]


def resolve_dataset_loading(mode, task, ipw_model_id="none"):
    resolved = ("disk" if task in INTENT_TASKS else "memory") if mode == "auto" else mode
    if resolved == "disk" and (task not in INTENT_TASKS or ipw_model_id != "none"):
        raise ValueError("Disk datasets support pass_intent/action_intent without inverse-propensity weighting only.")
    return resolved
