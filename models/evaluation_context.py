"""Evaluation-only selection and provenance, independent of training settings."""
from __future__ import annotations

import argparse
import math

from project_config import (INTENDED_RECEIVER_MODES, load_recorded_split_manifest, parse_return_type,
                            load_feature_run_metadata, LEGACY_FEATURE_SPLIT_MANIFESTS)
from datatools.sportec_seasons import add_season_argument, heldout_season_match_ids


def nonnegative_duration(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("duration must be finite, nonnegative seconds") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("duration must be finite, nonnegative seconds")
    return number


def binary_goal_return(value):
    try:
        kind, parameter, _ = parse_return_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if kind != "next" and not (kind == "disc" and parameter == 1):
        raise argparse.ArgumentTypeError("diagnostic must be binary: use next_<N> or disc_1, optionally with _skip1")
    return value


def add_evaluation_arguments(parser):
    parser.add_argument("--intended-receiver-mode", choices=INTENDED_RECEIVER_MODES)
    parser.add_argument("--diagnostic-return-type", type=binary_goal_return, default=None,
                        help="Binary goal diagnostic return definition; defaults to next_10.")
    parser.add_argument("--min-pass-duration", type=nonnegative_duration, default=None)
    add_season_argument(parser, help_text="Explicit held-out season(s); repeat for multiple seasons. Training/validation overlap is an error.")
    parser.add_argument("--season-half", choices=("first", "second"),
                        help="Requires --season: first selects matchdays 1-17; second selects 18-34.")


def evaluation_selection(args):
    if getattr(args, "season_half", None) and not getattr(args, "season", None):
        raise ValueError("--season-half requires --season.")
    return heldout_season_match_ids(args.season, args.season_half) if getattr(args, "season", None) else None


def recorded_manifest_id(record):
    metadata, arguments = record.get("metadata") or {}, record.get("args") or {}
    manifest_id = record.get("split_manifest_id") or metadata.get("split_manifest_id") or arguments.get("split_manifest_id")
    if manifest_id:
        return manifest_id
    feature_id = record.get("feature_run_id") or metadata.get("feature_run_id") or arguments.get("feature_run_id")
    if feature_id:
        feature = load_feature_run_metadata(feature_id, required=False) or {}
        return feature.get("split_manifest_id") or LEGACY_FEATURE_SPLIT_MANIFESTS.get(feature_id)
    return None


def excluded_membership(record):
    metadata = record.get("metadata") or {}
    result = {}
    datasets = metadata.get("split_datasets") or {}
    for category, key in (("training", "train_match_ids"), ("validation", "validation_match_ids")):
        dataset = datasets.get("train" if category == "training" else "validation") or {}
        ids = metadata.get(key)
        if ids is None:
            ids = dataset.get("requested_match_ids", dataset.get("contributing_match_ids"))
        if ids is not None:
            result[category] = set(map(str, ids))
    manifest_id = recorded_manifest_id(record)
    if manifest_id:
        # The development pool includes validation and any earlier folds/refits.
        result["development"] = set(load_recorded_split_manifest(manifest_id)["train"])
    if "development" not in result and not {"training", "validation"}.issubset(result):
        raise ValueError(f"Cannot verify training/validation exclusion for {record.get('model_id', 'checkpoint')}: missing recorded membership.")
    return result


def check_heldout(record, match_ids):
    requested = set(map(str, match_ids))
    exclusions = excluded_membership(record)
    overlaps = {name: sorted(requested & ids) for name, ids in exclusions.items() if requested & ids}
    if overlaps:
        details = "; ".join(f"{name}: {len(ids)} matches ({', '.join(ids)})" for name, ids in overlaps.items())
        raise ValueError(f"Held-out selection overlaps {record.get('model_id', 'checkpoint')}: {details}.")
    return {"model_id": record.get("model_id"), "requested_match_count": len(requested),
            "exclusion_sources": {name: len(ids) for name, ids in exclusions.items()}, "overlap_count": 0}


def diagnostic_name(return_type):
    return "goal_next10_diagnostic" if return_type == "next_10" else f"goal_{return_type}_diagnostic"
