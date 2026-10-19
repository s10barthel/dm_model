"""Evaluation-only selection and provenance, independent of training settings."""
from __future__ import annotations

import argparse
import math

from project_config import (INTENDED_RECEIVER_MODES, load_recorded_split_manifest, parse_return_type,
                            load_feature_run_metadata, LEGACY_FEATURE_SPLIT_MANIFESTS)
from datatools.sportec_seasons import add_season_argument, heldout_season_match_ids
from models.membership import (recorded_manifest_id as _recorded_manifest_id, resolve_membership_evidence,
                               MissingMembershipError)


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
    parser.add_argument("--filter", dest="evaluation_filter", choices=("pass",), default=None,
                        help="Evaluate only is_pass events; omission retains the checkpoint's action population.")
    parser.add_argument("--intended-receiver-mode", choices=INTENDED_RECEIVER_MODES)
    parser.add_argument("--diagnostic-return-type", type=binary_goal_return, default=None,
                        help="Binary goal diagnostic return definition; defaults to next_10.")
    parser.add_argument("--min-pass-duration", type=nonnegative_duration, default=None,
                        help="Minimum pass duration in seconds (inclusive); 0 removes the minimum, omission uses checkpoint settings.")
    add_season_argument(parser, help_text="Explicit held-out season(s); repeat for multiple seasons. Training/validation overlap is an error.")
    parser.add_argument("--season-half", choices=("first", "second"),
                        help="Requires --season: first selects matchdays 1-17; second selects 18-34.")


def evaluation_selection(args):
    if getattr(args, "season_half", None) and not getattr(args, "season", None):
        raise ValueError("--season-half requires --season.")
    return heldout_season_match_ids(args.season, args.season_half) if getattr(args, "season", None) else None


def recorded_manifest_id(record):
    return _recorded_manifest_id(record, feature_loader=load_feature_run_metadata)


def excluded_membership(record):
    return resolve_membership_evidence(record, strict=True, manifest_loader=load_recorded_split_manifest,
                                       feature_loader=load_feature_run_metadata)["memberships"]


def check_heldout(record, match_ids):
    requested = set(map(str, match_ids))
    evidence = resolve_membership_evidence(record, strict=True, require_complete=False,
        manifest_loader=load_recorded_split_manifest, feature_loader=load_feature_run_metadata)
    exclusions = evidence["memberships"]
    overlaps = {name: sorted(requested & ids) for name, ids in exclusions.items() if requested & ids}
    if overlaps:
        details = "; ".join(f"{name}: {len(ids)} matches ({', '.join(ids)})" for name, ids in overlaps.items())
        raise ValueError(f"Held-out selection overlaps {record.get('model_id', 'checkpoint')}: {details}.")
    check = {"model_id": record.get("model_id"), "requested_match_count": len(requested),
             "requested_match_ids": sorted(requested),
             "exclusion_sources": {name: len(ids) for name, ids in exclusions.items()},
             "status": "verified" if evidence["complete"] else "unverified",
             "overlap_count": 0 if evidence["complete"] else None}
    if not evidence["complete"]:
        check["reason"] = "missing_recorded_membership"
        raise MissingMembershipError(
            f"Cannot verify training/validation exclusion for {record.get('model_id', 'checkpoint')}: missing recorded membership.",
            check=check)
    return check


def diagnostic_name(return_type):
    return "goal_next10_diagnostic" if return_type == "next_10" else f"goal_{return_type}_diagnostic"
