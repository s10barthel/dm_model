"""CLI shared by training entry points (kept independent of torch imports)."""
from pathlib import Path

DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / "data" / "cache" / "intent_datasets")
INTENT_TASKS = {"pass_intent", "action_intent"}
PREPARED_TASKS = INTENT_TASKS | {
    "success_intent", "pass_success", "pass_height", "outcome_scoring",
    "outcome_conceding", "failure_receiver",
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise ValueError("must be a positive integer")
    return value


def add_dataset_loading_arguments(parser):
    from datatools.graph_validation import add_graph_validation_argument
    add_graph_validation_argument(parser)
    parser.add_argument("--dataset-loading", choices=("auto", "memory", "disk"), default="auto")
    parser.add_argument("--dataset-buffer-matches", type=positive_int, default=4)
    parser.add_argument("--dataset-cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--keep-cache", action="store_true", default=False,
                        help="Retain prepared dataset caches after successful training.")


def dataset_loading_flags(args):
    from datatools.graph_validation import graph_validation_flags
    return ["--dataset-loading", getattr(args, "dataset_loading", "auto"),
            "--dataset-buffer-matches", str(getattr(args, "dataset_buffer_matches", 4)),
            "--dataset-cache-dir", str(getattr(args, "dataset_cache_dir", DEFAULT_CACHE_DIR))] + (
                ["--keep-cache"] if getattr(args, "keep_cache", False) else []) + graph_validation_flags(args)


def resolve_dataset_loading(mode, task, ipw_model_id="none"):
    resolved = ("disk" if task in PREPARED_TASKS else "memory") if mode == "auto" else mode
    if resolved == "disk" and task not in PREPARED_TASKS:
        raise ValueError(f"Disk datasets do not support task {task!r}.")
    return resolved


def saved_dataset_loading(settings):
    """Do not reinterpret historical auto settings when resuming or refitting."""
    resolved = settings.get("resolved_dataset_loading")
    if resolved in ("disk", "memory"):
        return resolved
    mode = settings.get("dataset_loading", "auto")
    if mode == "auto":
        return "disk" if settings["task"] in INTENT_TASKS else "memory"
    return mode
