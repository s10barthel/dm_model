"""Pass-height target definitions, separate from probability decision thresholds."""
from __future__ import annotations

import argparse
import math

import torch

from datatools.config import LABEL_COLUMNS


def positive_height(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("height cutoff must be finite and positive metres") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("height cutoff must be finite and positive metres")
    return result


def add_pass_height_argument(parser):
    if "--pass-height-threshold" in parser._option_string_actions:
        return
    parser.add_argument("--pass-height-threshold", type=positive_height, default=None,
                        help="Observed maximum-height cutoff in metres (>=); distinct from the probability threshold.")


def definition(threshold, source):
    return {"threshold_meters": positive_height(threshold) if threshold is not None else None,
            "operator": ">=", "target_source": "pass_max_ball_z", "resolution_source": source}


def model_definition(args, metadata=None):
    args = vars(args) if not isinstance(args, dict) else args
    recorded = args.get("pass_height_definition") or (metadata or {}).get("pass_height_definition")
    if recorded:
        if recorded.get("operator") != ">=" or recorded.get("target_source") != "pass_max_ball_z":
            raise ValueError("Unsupported recorded pass-height target definition.")
        positive_height(recorded["threshold_meters"])
        return definition(recorded["threshold_meters"], recorded.get("resolution_source", "checkpoint"))
    # Legacy inference is descriptive only: callers must retain stored-label behavior.
    from project_config import load_feature_run_metadata
    run_id = args.get("feature_run_id") or (metadata or {}).get("feature_run_id")
    feature = (load_feature_run_metadata(run_id, required=False) or {}) if run_id else {}
    cutoff = feature.get("pass_height_threshold_meters")
    return definition(cutoff, "legacy_feature_metadata" if cutoff is not None else "unknown")


def resolve_training_height(args, feature_metadata, *, resume=False):
    requested = getattr(args, "pass_height_threshold", None)
    saved = getattr(args, "pass_height_definition", None)
    if resume:
        if requested is not None and (saved is None or positive_height(requested) != saved["threshold_meters"]):
            raise ValueError("Resume pass-height cutoff conflicts with the checkpoint target definition.")
        return
    cutoff = requested if requested is not None else feature_metadata.get("pass_height_threshold_meters")
    if cutoff is None:
        if args.task == "pass_height" or requested is not None:
            raise ValueError("Pass-height training requires --pass-height-threshold or feature-run threshold metadata.")
        return
    args.pass_height_threshold = positive_height(cutoff)
    args.pass_height_definition = definition(cutoff, "explicit" if requested is not None else "feature_run")


def evaluation_height(model_args, requested=None):
    args = vars(model_args) if not isinstance(model_args, dict) else model_args
    original = model_definition(args)
    recorded = args.get("pass_height_definition")
    cutoff = positive_height(requested) if requested is not None else (original["threshold_meters"] if recorded else None)
    effective = definition(cutoff, "explicit_evaluation" if requested is not None else "checkpoint") if cutoff is not None else original.copy()
    return cutoff, {"model_pass_height_definition": original,
                    "evaluation_pass_height_definition": effective,
                    "pass_height_target_override": requested is not None,
                    "pass_height_label_mode": "continuous" if cutoff is not None else "legacy_stored"}


def relabel_height(labels, threshold):
    """Return private labels; nonfinite observations stay unknown, never negative."""
    result = labels.clone()
    height = result[..., LABEL_COLUMNS.index("pass_max_ball_z")]
    result[..., LABEL_COLUMNS.index("pass_high")] = torch.where(
        torch.isfinite(height), (height >= positive_height(threshold)).to(result.dtype),
        torch.full_like(height, float("nan")))
    return result


def check_height_probability(cached_definition, requested):
    if requested is None:
        return
    cutoff = (cached_definition or {}).get("threshold_meters")
    if cutoff is None or positive_height(cutoff) != positive_height(requested):
        raise ValueError("Cached pass-height probability has an unknown or incompatible height cutoff.")


def cached_height_definition(metadata):
    recorded = metadata.get("pass_height_definition") or (metadata.get("pass_height_model_record") or {}).get("pass_height_definition")
    if not recorded and metadata.get("pass_height_model_id"):
        from models.utils import get_model_record
        try:
            model = get_model_record(str(metadata["pass_height_model_id"]))
        except (FileNotFoundError, ValueError):
            pass
        else:
            recorded = model_definition(model["args"], model["metadata"])
    return recorded or definition(None, "unknown")


def height_export_context(context):
    return {
        "model_pass_height_threshold_meters": context["model_pass_height_definition"]["threshold_meters"],
        "evaluation_pass_height_threshold_meters": context["evaluation_pass_height_definition"]["threshold_meters"],
        "pass_height_target_override": context["pass_height_target_override"],
        "pass_height_label_mode": context["pass_height_label_mode"],
    }
