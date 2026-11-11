"""Mandatory explicit acceptance of omitted Hawkeye component checkpoints."""
from __future__ import annotations

import json
import sys

from project_config import write_run_metadata
from scripts.comparison_warnings import ComparisonCancelled


HAWKEYE_MODEL_TASKS = (
    "action_intent", "pass_intent", "pass_success", "pass_height",
    "outcome_scoring", "outcome_conceding",
)


def _record_selection(selection, output_dir, *, status):
    path = output_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"run_id": output_dir.name}
    metadata.update(status=status, command=" ".join(sys.argv), model_selection=selection)
    write_run_metadata(output_dir, metadata)


def confirm_missing_hawkeye_models(model_ids, args, output_dir):
    omitted = [task for task in HAWKEYE_MODEL_TASKS if not model_ids.get(task)]
    selection = {
        "selected_models": dict(model_ids), "omitted_models": omitted,
        "warning_acceptances": [], "cancelled": False,
    }
    if not omitted:
        _record_selection(selection, output_dir, status="running")
        return selection
    print("WARNING: Hawkeye component model IDs were omitted:", file=sys.stderr, flush=True)
    for task in omitted:
        print(f"- {task}: no checkpoint selected; its model predictions will be NaN.", file=sys.stderr, flush=True)
    if "pass_success" in omitted and args.use_physical_xpass and args.xpass_weight == "v0":
        source = "PC xPass" if args.pc_xpass else "physical xPass"
        print(f"  pass_success will copy cached {source} for passes; model_pass_success and carry success will be NaN.",
              file=sys.stderr, flush=True)
    try:
        while True:
            try:
                answer = input("Proceed with these omitted models? [yes/no] ").strip().lower()
            except EOFError:
                raise ComparisonCancelled("Hawkeye run cancelled: no explicit omission decision received.") from None
            if answer in {"y", "yes"}:
                selection["warning_acceptances"].append({"method": "interactive", "omitted_models": omitted})
                _record_selection(selection, output_dir, status="running")
                return selection
            if answer in {"n", "no"}:
                raise ComparisonCancelled("Hawkeye run cancelled by user.")
            print("Please answer yes or no; an empty answer does not accept the omissions.", flush=True)
    except (ComparisonCancelled, KeyboardInterrupt):
        selection["cancelled"] = True
        _record_selection(selection, output_dir, status="cancelled")
        raise
