"""Shared optional checkpoints, prediction availability, and xPass runtime policy."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pc_xpass_auxiliary import validate_auxiliary_selectors
from physical_pass_model import inference_uses_physical_xpass, model_uses_physical_xpass
from project_config import write_run_metadata
from scripts.comparison_warnings import ComparisonCancelled

COMPONENT_MODEL_TASKS = (
    "action_intent", "pass_intent", "pass_success", "pass_height",
    "outcome_scoring", "outcome_conceding",
)


class ComponentModels(dict):
    """Checkpoint mapping with a separate, unreported physical runtime context."""
    xpass_runtime = None


def uses_cached_v0(args):
    version = getattr(args, "xpass_weight", "v3")
    versions = version if isinstance(version, (list, tuple)) else [version]
    return bool(getattr(args, "use_physical_xpass", False) and versions and all(item == "v0" for item in versions))


def component_availability(model_ids, args):
    """Configuration availability; per-state cache gaps remain runtime checks."""
    available = {task: bool(model_ids.get(task)) for task in COMPONENT_MODEL_TASKS}
    available["model_pass_success"] = available["carry_success"] = available["pass_success"]
    available["pass_success"] |= uses_cached_v0(args)
    dependencies = {
        "outcome_success": ("outcome_scoring", "outcome_conceding"),
        "outcome_failure": ("outcome_scoring", "outcome_conceding"),
        "pass_score": ("pass_success", "outcome_scoring", "outcome_conceding"),
        "weighted_pass_score": ("pass_success", "pass_intent", "outcome_scoring", "outcome_conceding"),
        "epv": ("pass_success", "pass_intent", "outcome_scoring", "outcome_conceding"),
        "carry_score": ("carry_success", "outcome_scoring", "outcome_conceding"),
    }
    missing = {name: [task for task in tasks if not available[task]] for name, tasks in dependencies.items()}
    available.update({name: not tasks for name, tasks in missing.items()})
    return {"available": available, "unavailable_metrics": {name: tasks for name, tasks in missing.items() if tasks},
            "pass_success_source": ("pc_xpass_v0" if getattr(args, "pc_xpass", False) else "physical_xpass_v0")
                if uses_cached_v0(args) else ("checkpoint" if model_ids.get("pass_success") else "unavailable")}


def validate_component_dependencies(args, model_ids, *, require_epv=False):
    active = bool(getattr(args, "use_physical_xpass", False))
    if active and not uses_cached_v0(args) and not model_ids.get("pass_success"):
        raise ValueError(f"--xpass-weight {args.xpass_weight} requires --pass-success-model-id or a pass_success model in --bundle-id.")
    version = getattr(args, "xpass_weight", "v3")
    if isinstance(version, (list, tuple)):
        from copy import copy
        args = copy(args)
        args.xpass_weight = next((item for item in ("v5", "v4") if item in version), version[0])
    validate_auxiliary_selectors(args)
    if require_epv:
        missing = component_availability(model_ids, args)["unavailable_metrics"].get("epv", [])
        if missing:
            raise ValueError("EPV generation requires inputs for: " + ", ".join(missing))


def _record_selection(selection, output_dir, *, status):
    output_dir = Path(output_dir)
    path = output_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {"run_id": output_dir.name}
    metadata.update(status=status, command=" ".join(sys.argv), model_selection=selection)
    write_run_metadata(output_dir, metadata)


def confirm_component_selection(model_ids, args, output_dir, *, tasks=COMPONENT_MODEL_TASKS, metrics=None):
    omitted = [task for task in tasks if not model_ids.get(task)]
    availability = component_availability(model_ids, args)
    unavailable = availability["unavailable_metrics"]
    if metrics is not None:
        unavailable = {name: missing for name, missing in unavailable.items() if name in metrics}
    selection = {"selected_models": dict(model_ids), "omitted_models": omitted,
                 "availability": availability, "warning_acceptances": [], "cancelled": False}
    if not omitted and not unavailable:
        _record_selection(selection, output_dir, status="running")
        return selection
    print("WARNING: omitted inference models and unavailable metrics:", file=sys.stderr, flush=True)
    for task in omitted:
        print(f"- {task}: no checkpoint selected; its model predictions will be NaN.", file=sys.stderr, flush=True)
    for name, missing in unavailable.items():
        print(f"- {name}: unavailable; missing inputs: {', '.join(missing)}.", file=sys.stderr, flush=True)
    if uses_cached_v0(args) and not model_ids.get("pass_success"):
        print("- Pass success comes from cached xPass v0. Pass score and EPV can use it without a checkpoint; "
              "model_pass_success, carry success, and carry score remain NaN.", file=sys.stderr, flush=True)
    try:
        while True:
            try:
                answer = input("Proceed with these omissions and limitations? [yes/no] ").strip().lower()
            except (EOFError, OSError):
                raise ComparisonCancelled("Inference cancelled: no explicit omission decision received.") from None
            if answer in {"y", "yes"}:
                selection["warning_acceptances"].append({"method": "interactive", "omitted_models": omitted,
                    "unavailable_metrics": unavailable})
                _record_selection(selection, output_dir, status="running")
                return selection
            if answer in {"n", "no"}:
                raise ComparisonCancelled("Inference cancelled by user.")
            print("Please answer yes or no; an empty answer does not accept the omissions.", flush=True)
    except (ComparisonCancelled, KeyboardInterrupt):
        selection["cancelled"] = True
        _record_selection(selection, output_dir, status="cancelled")
        raise


def configure_xpass_runtime(args, models, cache_dir):
    """Keep a checkpoint-free runtime outside the loaded checkpoint dictionary."""
    runtime = models.get("pass_success")
    if runtime is None and uses_cached_v0(args):
        runtime = SimpleNamespace(args={"task": "pass_success"})
    if runtime is None:
        args._xpass_runtime = None
        if isinstance(models, ComponentModels):
            models.xpass_runtime = None
        return None
    if getattr(args, "use_physical_xpass", False):
        runtime.args.update(inference_use_physical_xpass=True, pc_xpass=bool(getattr(args, "pc_xpass", False)),
            pc_xpass_id=getattr(args, "pc_xpass_id", None), x_pass_version=getattr(args, "x_pass_version", "top10"),
            xpass_weight=getattr(args, "xpass_weight", "v3"), ball_z_limit=getattr(args, "ball_z_limit", "none"))
        for name in ("v4_power", "v4_zero", "v4_discount", "v5_intent_threshold", "v5_discount", "pass_height_threshold"):
            value = getattr(args, name, None)
            if value is not None:
                runtime.args[name] = value
    if runtime_uses_xpass(runtime):
        runtime.args.update(physical_runtime_cache_disabled=bool(getattr(args, "no_physical_cache", False)),
            physical_runtime_cache_refresh=False, physical_runtime_cache_read_only=True,
            physical_cache_dir=str(cache_dir), physical_num_workers=getattr(args, "physical_num_workers", "auto"),
            physical_worker_thread_limit=getattr(args, "physical_worker_thread_limit", 1),
            physical_batch_size=getattr(args, "physical_batch_size", 16))
    args._xpass_runtime = runtime
    if isinstance(models, ComponentModels):
        models.xpass_runtime = runtime
    return runtime


def runtime_uses_xpass(runtime):
    return runtime is not None and (model_uses_physical_xpass(runtime.args) or inference_uses_physical_xpass(runtime.args))


def runtime_kwargs(models, runtime):
    return {"xpass_runtime": runtime} if "pass_success" not in models and runtime is not None else {}


def require_sportec_feature_context(args, shared, bundle, *, cache_selected=False):
    if (not getattr(args, "feature_run_id", None) and not cache_selected and not bundle
            and not any(getattr(args, task + "_model_id", None) for task in (*COMPONENT_MODEL_TASKS, "success_intent"))
            and not shared.get("model_records") and not shared.get("source_feature_run_ids")
            and not shared.get("feature_run_id")):
        raise ValueError("Sportec inference without model, bundle, or cache context requires --feature-run-id.")


def confirm_visualization_selection(model_ids, args, output_dir, *, include_success_intent=False):
    from scripts.visualization_selection import resolve_component_selection, MODEL_TASKS
    requested = resolve_component_selection(args, include_success_intent=include_success_intent)
    tasks = [task for task in MODEL_TASKS if task in requested.required_component_groups]
    return confirm_component_selection(model_ids, args, output_dir, tasks=tasks,
        metrics=requested.requested_component_groups)
