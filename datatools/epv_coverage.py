"""Optional EPV inputs and verified, whole-match availability for outcome labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datatools import config

VERSION = 1
POLICY = "complete_match"
ATTRIBUTE = "epv_coverage"
DIRECTORY = "epv_coverage"
TARGETS = ("scores_epv", "concedes_epv")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotations(events, match_id, directory):
    """Require finite values on candidates, retaining the original event index/order."""
    path = Path(directory) / f"{match_id}.csv"
    candidates = events.spadl_type.isin(config.XT_ACTION_TYPES) & events.object_id.astype(str).str[:4].isin(["home", "away"])
    identity = events[["action_id", "spadl_type", "object_id"]].to_csv(index=False).encode()
    report = dict(version=VERSION, policy=POLICY, match_id=str(match_id), complete=False,
                  annotation=str(path.resolve()), annotation_sha256=None,
                  event_identity_sha256=hashlib.sha256(identity).hexdigest(),
                  required_actions=int(candidates.sum()), finite_actions=0,
                  missing_required_count=0, nonfinite_required_count=0,
                  missing_required_action_ids=[], nonfinite_required_action_ids=[], reasons=[])
    values = pd.Series(np.nan, index=events.index, dtype=float)
    if not path.is_file():
        report["reasons"].append("missing_file")
        report["missing_required_count"] = int(candidates.sum())
        report["missing_required_action_ids"] = events.loc[candidates, "action_id"].head(10).tolist()
        return values, report
    report["annotation_sha256"] = fingerprint(path)
    try:
        side = pd.read_csv(path, usecols=lambda name: name in {"action_id", "epv"})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        report["reasons"].append("malformed_file")
        return values, report
    if not {"action_id", "epv"} <= set(side):
        report["reasons"].append("missing_columns")
        return values, report
    if side.action_id.isna().any() or not side.action_id.is_unique:
        report["reasons"].append("invalid_or_duplicate_action_ids")
        invalid = side.action_id.isna() | side.action_id.duplicated(keep=False)
        report["invalid_action_ids"] = side.loc[invalid, "action_id"].astype(str).head(10).tolist()
        return values, report
    unknown = side.loc[~side.action_id.isin(events.action_id), "action_id"]
    if len(unknown):
        report["reasons"].append("unknown_action_ids")
        report["unknown_action_ids"] = unknown.head(10).tolist()
        return values, report
    aligned = side.set_index("action_id").reindex(events.action_id)
    values = pd.Series(pd.to_numeric(aligned.epv, errors="coerce").to_numpy(dtype=float), index=events.index)
    present = events.action_id.isin(side.action_id)
    missing = candidates & ~present
    nonfinite = candidates & present & ~np.isfinite(values)
    report.update(finite_actions=int((candidates & np.isfinite(values)).sum()),
                  missing_required_count=int(missing.sum()), nonfinite_required_count=int(nonfinite.sum()),
                  missing_required_action_ids=events.loc[missing, "action_id"].head(10).tolist(),
                  nonfinite_required_action_ids=events.loc[nonfinite, "action_id"].head(10).tolist(),
                  ignored_missing_events=int((~candidates & ~present).sum()))
    if missing.any():
        report["reasons"].append("missing_required_actions")
    if nonfinite.any():
        report["reasons"].append("nonfinite_required_values")
    report["complete"] = not report["reasons"]
    if not report["complete"]:
        values[:] = np.nan
    return values, report


def unavailable(events):
    coverage = events.attrs.get(ATTRIBUTE)
    return coverage is not None and not coverage["complete"]


def mask_targets(frame, events):
    if unavailable(events):
        frame[list(TARGETS)] = np.nan
    return frame


def label_fingerprint(labels):
    """Bind coverage to EPV values and identities, allowing other target refreshes."""
    columns = [config.LABEL_INDEX[name] for name in ("action_index", *TARGETS)]
    values = labels[:, columns].detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((values.shape, values.dtype)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def verified_report(root, report, requests, label_hashes=None):
    """Carry forward certifications only while the underlying coverage is identical."""
    path = Path(root) / DIRECTORY / f"{report['match_id']}.json"
    variants, labels = {}, {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        prior_variants = previous.pop("verified_variants", {})
        prior_labels = previous.pop("verified_labels", {})
        if previous == report:
            variants = prior_variants
            labels = prior_labels
    for ret, conditions in requests.items():
        variants[ret] = sorted(set(variants.get(ret, [])) | set(conditions))
    labels.update(label_hashes or {})
    return dict(report, verified_variants=variants, verified_labels=labels)


def validate_labels(root, metadata, match_ids, label_dir, return_type, condition="base", *, context="training"):
    """Preflight only selected matches, before memory/disk dataset preparation."""
    import torch

    policy = metadata.get(ATTRIBUTE) or {}
    if policy.get("version") != VERSION or policy.get("policy") != POLICY:
        raise ValueError("EPV coverage is unverified. Run --refresh-stop-conditions on the completed feature run "
                         "before EPV training/evaluation; finite legacy zero labels do not certify annotations.")
    root, label_dir = Path(root), Path(label_dir)
    matches = list(dict.fromkeys(map(str, match_ids)))
    failures = []
    columns = [config.LABEL_INDEX[name] for name in TARGETS]
    for mid in matches:
        path = root / DIRECTORY / f"{mid}.json"
        if not path.is_file():
            failures.append(f"{mid}: missing EPV coverage record")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("version") != VERSION or record.get("policy") != POLICY
                or record.get("match_id") != mid):
            failures.append(f"{mid}: invalid EPV coverage record")
            continue
        if not record.get("complete"):
            failures.append(f"{mid}: {', '.join(record.get('reasons', []))}; "
                            f"missing={record.get('missing_required_count', 0)}, "
                            f"nonfinite={record.get('nonfinite_required_count', 0)}; "
                            f"action IDs={record.get('missing_required_action_ids', []) + record.get('nonfinite_required_action_ids', []) + record.get('unknown_action_ids', []) + record.get('invalid_action_ids', [])}")
            continue
        if condition not in record.get("verified_variants", {}).get(return_type, []):
            failures.append(f"{mid}: unverified EPV labels for {return_type}/{condition}; refresh this variant")
            continue
        label_path = label_dir / f"{mid}.pt"
        if not label_path.is_file():
            failures.append(f"{mid}: missing EPV labels at {label_path}")
            continue
        labels = torch.load(label_path, map_location="cpu", weights_only=True)
        if (not isinstance(labels, torch.Tensor) or labels.ndim != 2
                or labels.shape[1] <= max(columns) or not torch.isfinite(labels[:, columns]).all()):
            failures.append(f"{mid}: EPV label columns are missing or nonfinite at {label_path}")
            continue
        try:
            relative = label_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = None
        if (relative is None or record.get("verified_labels", {}).get(relative) != label_fingerprint(labels)):
            failures.append(f"{mid}: EPV labels are unverified or changed at {label_path}; refresh this variant")
    if failures:
        details = "\n- ".join(failures[:10])
        raise ValueError(f"Incomplete EPV annotations/labels for {context}: {len(failures)}/{len(matches)} selected matches.\n- "
                         + details + (f"\n(and {len(failures) - 10} more)" if len(failures) > 10 else ""))
    return dict(version=VERSION, policy=POLICY, selected_matches=len(matches))
