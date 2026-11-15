"""CLI shared by training and evaluation (kept independent of torch imports)."""
from pathlib import Path
import argparse

REPOSITORY_ROOT = Path(__file__).resolve().parent
LEGACY_CACHE_DIR = str(REPOSITORY_ROOT / "data" / "cache" / "intent_datasets")
DEFAULT_TRAINING_CACHE_DIR = str(REPOSITORY_ROOT / "cache" / "training")
DEFAULT_EVALUATION_CACHE_DIR = str(REPOSITORY_ROOT / "cache" / "evaluation")
DEFAULT_CACHE_DIR = DEFAULT_TRAINING_CACHE_DIR  # Compatibility for library callers.
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


def default_cache_dir(purpose):
    return {"training": DEFAULT_TRAINING_CACHE_DIR, "evaluation": DEFAULT_EVALUATION_CACHE_DIR}[purpose]


def resolve_dataset_cache_dir(purpose, *, requested=None, saved=None):
    """An explicit root wins; only a saved legacy default is redirected."""
    if requested is not None:
        return str(Path(requested).resolve())
    if saved and Path(saved).resolve() != Path(LEGACY_CACHE_DIR).resolve():
        return str(Path(saved).resolve())
    return default_cache_dir(purpose)


def configure_dataset_cache(args, purpose, *, saved=None):
    requested = getattr(args, "dataset_cache_dir", None)
    if saved is not None and not getattr(args, "_dataset_cache_dir_explicit", False):
        requested = None
    args.dataset_cache_dir = resolve_dataset_cache_dir(
        purpose, requested=requested, saved=(saved or {}).get("dataset_cache_dir"))
    args.cache_purpose = purpose
    return args.dataset_cache_dir


class CacheDirectoryAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace._dataset_cache_dir_explicit = True


def add_dataset_loading_arguments(parser, *, purpose="training", cache_options=("--dataset-cache-dir",)):
    from datatools.graph_validation import add_graph_validation_argument
    add_graph_validation_argument(parser)
    parser.add_argument("--dataset-loading", choices=("auto", "memory", "disk"), default="auto")
    parser.add_argument("--dataset-buffer-matches", type=positive_int, default=4)
    parser.add_argument("--dataset-preparation-workers", type=positive_int, default=1,
                        help="CPU processes for per-match preparation (default: 1).")
    parser.add_argument("--dataset-worker-threads", type=positive_int, default=1,
                        help="Native tensor threads per preparation worker (default: 1).")
    parser.set_defaults(_dataset_cache_dir_explicit=False)
    parser.add_argument(*cache_options, dest="dataset_cache_dir", default=default_cache_dir(purpose),
                        action=CacheDirectoryAction, help=f"Prepared {purpose} cache root (default: %(default)s).")
    parser.add_argument("--keep-cache", action="store_true", default=False,
                        help="Persistently protect prepared caches from later model-start pruning; "
                             "otherwise caches remain until the next model starts.")


def dataset_loading_flags(args, *, purpose="training"):
    from datatools.graph_validation import graph_validation_flags
    return ["--dataset-loading", getattr(args, "dataset_loading", "auto"),
            "--dataset-buffer-matches", str(getattr(args, "dataset_buffer_matches", 4)),
            "--dataset-cache-dir", str(getattr(args, "dataset_cache_dir", None) or default_cache_dir(purpose)),
            "--dataset-preparation-workers", str(getattr(args, "dataset_preparation_workers", 1)),
            "--dataset-worker-threads", str(getattr(args, "dataset_worker_threads", 1))] + (
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
