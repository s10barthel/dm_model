"""Explicit, cohort-scoped acceptance of missing evaluation membership evidence."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

from models.evaluation_context import check_heldout
from models.membership import MissingMembershipError


class EvaluationMembershipCancelled(RuntimeError):
    pass


def collect_evaluation_membership(record_cohorts):
    """Check every participant before any missing-evidence confirmation is offered."""
    checks = []
    for record, cohort in record_cohorts:
        try:
            check = check_heldout(record, cohort)
        except MissingMembershipError as exc:
            check = exc.check
        if check not in checks:
            checks.append(check)
    return membership_report(checks)


def membership_report(checks, confirmation=None):
    affected = sorted({check["model_id"] for check in checks if check["status"] == "unverified"})
    return {"status": "unverified" if affected else "verified", "checks": checks,
            "affected_model_ids": affected, "confirmation": confirmation}


def membership_for_models(report, model_ids, cohort):
    """Keep only the checks applicable to one evaluator and its auxiliary models."""
    requested = sorted(set(map(str, cohort)))
    checks = [deepcopy(check) for check in report["checks"]
              if check["model_id"] in model_ids and check["requested_match_ids"] == requested]
    if {check["model_id"] for check in checks} != set(model_ids):
        raise ValueError("Evaluation membership checks do not cover all participating models and the requested cohort.")
    return membership_report(checks, deepcopy(report["confirmation"]))


def accept_recorded_membership(report, approval):
    """Validate an existing interactive decision without offering another prompt."""
    if not isinstance(approval, dict):
        raise ValueError("Membership acceptance manifest has no evaluation_membership record.")
    confirmation = approval.get("confirmation")
    checks = approval.get("checks")
    affected = approval.get("affected_model_ids")
    if (not isinstance(confirmation, dict) or not isinstance(checks, list)
            or any(not isinstance(item, dict) for item in checks) or not isinstance(affected, list)):
        raise ValueError("Membership acceptance manifest contains malformed approval records.")
    if (approval.get("status") != "unverified" or confirmation.get("decision") != "accepted"
            or confirmation.get("method") != "interactive"):
        raise ValueError("Membership acceptance manifest does not record interactive approval.")
    for check in report["checks"]:
        if check["status"] != "unverified":
            continue
        if check["model_id"] not in affected or not any(
            item.get("model_id") == check["model_id"] and item.get("status") == "unverified"
            and item.get("requested_match_ids") == check["requested_match_ids"]
            for item in checks
        ):
            raise ValueError(f"Membership approval does not cover {check['model_id']} and its exact requested match cohort.")
    report["confirmation"] = deepcopy(confirmation)


def _accept_from_manifest(report, manifest_path):
    saved = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Membership acceptance manifest must contain an object.")
    accept_recorded_membership(report, saved.get("evaluation_membership"))
    report["confirmation"].update(source="wrapper_manifest", manifest_path=str(Path(manifest_path).resolve()))


def confirm_evaluation_membership(report, *, acceptance_manifest=None):
    """Ask once; general comparison-warning acceptance never authorizes this step."""
    if acceptance_manifest is not None:
        _accept_from_manifest(report, acceptance_manifest)
        return
    if report["status"] == "verified":
        return
    print("Cannot verify training/validation exclusion for:", file=sys.stderr, flush=True)
    for model_id in report["affected_model_ids"]:
        print(f"- {model_id}", file=sys.stderr, flush=True)
    print("Results may include matches used during model development. Evaluation membership will be marked unverified.",
          file=sys.stderr, flush=True)

    def cancel(reason):
        report["confirmation"] = {"decision": "cancelled", "method": "interactive", "reason": reason}
        raise EvaluationMembershipCancelled(f"Evaluation cancelled: {reason}.")

    if not sys.stdin.isatty():
        cancel("an interactive yes is required for unverified membership")
    while True:
        try:
            answer = input("Continue with unverified evaluation membership? [yes/no] ").strip().lower()
        except EOFError:
            cancel("no explicit answer received")
        except KeyboardInterrupt:
            cancel("confirmation interrupted")
        if answer in {"yes", "y"}:
            report["confirmation"] = {"decision": "accepted", "method": "interactive"}
            return
        if answer in {"no", "n"}:
            cancel("declined by user")
        print("Please answer yes or no; an empty answer does not authorize evaluation.", flush=True)


def confirm_direct_evaluation_membership(report, args):
    """Persist a standalone cancellation without overwriting earlier metric artifacts."""
    try:
        confirm_evaluation_membership(report, acceptance_manifest=getattr(args, "membership_acceptance_manifest", None))
    except EvaluationMembershipCancelled as exc:
        output = getattr(args, "evaluation_output_dir", None)
        if output:
            root = Path(output)
            root.mkdir(parents=True, exist_ok=True)
            path = root / "membership_cancellation.json"
            path.write_text(json.dumps({"model_id": args.model_id, "status": "cancelled", "error": str(exc),
                                        "evaluation_membership": report}, indent=2), encoding="utf-8")
        raise


def combine_evaluation_membership(reports):
    """Preserve each evaluator's decision in portable comparison reports."""
    reports = list(reports)
    checks = []
    for report in reports:
        for check in report["checks"]:
            if check not in checks:
                checks.append(deepcopy(check))
    result = membership_report(checks)
    result["confirmations"] = [deepcopy(report["confirmation"]) for report in reports if report.get("confirmation")]
    return result
