"""Shared IPW controls, without importing the training implementation."""
import argparse
from pathlib import Path

DEFAULT_IPW_CACHE_DIR = str(Path(__file__).resolve().parent / "data" / "cache" / "ipw_probabilities")
IPW_DEFAULTS = {"ipw_batch_size": 256, "ipw_probability_cache": "on",
                "ipw_probability_cache_dir": DEFAULT_IPW_CACHE_DIR}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_ipw_arguments(parser):
    parser.add_argument("--ipw-batch-size", type=positive_int, default=256,
                        help="IPW inference batch size, independent of training (default: 256).")
    parser.add_argument("--ipw-probability-cache", choices=("on", "off"), default="on")
    parser.add_argument("--ipw-probability-cache-dir", default=DEFAULT_IPW_CACHE_DIR)


def restore_ipw_defaults(args):
    for key, value in IPW_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def ipw_flags(args):
    return [item for key, default in IPW_DEFAULTS.items()
            for item in ("--" + key.replace("_", "-"), str(getattr(args, key, default)))]
