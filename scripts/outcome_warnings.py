"""Record explicit acceptance of mixed outcome definitions in ordinary inference."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.comparison_warnings import ComparisonWarnings, ComparisonCancelled


def outcome_warning_policy(args):
    return ComparisonWarnings(getattr(args, "accept_comparison_warnings", False))


def outcome_warning_metadata(policy, shared):
    return {"outcome_compatibility": {
        **{key: shared.get(key) for key in (
            "source_xt_target_definitions", "source_outcome_learning_contracts",
            "source_outcome_model_ids", "runtime_xt_reference")},
        **policy.metadata(),
    }}


def confirm_outcome_warnings(policy, root, shared):
    if not policy.messages:
        return
    root = Path(root)
    path = root / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"run_id": root.name}
    try:
        policy.confirm()
    except (ComparisonCancelled, KeyboardInterrupt):
        policy.cancelled = True
        metadata.update(status="cancelled", **outcome_warning_metadata(policy, shared))
        from project_config import write_run_metadata
        write_run_metadata(root, metadata)
        raise
    metadata.update(status="running", **outcome_warning_metadata(policy, shared))
    from project_config import write_run_metadata
    write_run_metadata(root, metadata)
