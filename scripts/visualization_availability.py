"""Read-only availability discovery for visualization inputs."""
from pathlib import Path

from scripts.visualization_selection import (
    GROUP_TO_COMPONENTS, MODEL_TASKS, available_model_components, bind_component_selection,
    filter_component_selection, resolve_component_selection,
)


def select_inference_components(args, model_ids, *, include_success_intent=False):
    selection = resolve_component_selection(args, include_success_intent=include_success_intent)
    available = available_model_components(model_ids) | {"physical_xpass"}
    args._available_component_inputs = available
    reasons = {component: f"missing {task} checkpoint"
               for task in MODEL_TASKS if not model_ids.get(task)
               for component in GROUP_TO_COMPONENTS[task]}
    if not model_ids.get("pass_success"):
        reasons["model_pass_success"] = "missing pass_success checkpoint"
    return bind_component_selection(args, filter_component_selection(selection, available, reasons=reasons), report=False)


def finish_inference_selection(args, cache_dir, metric, version):
    from physical_pass_model import PhysicalXpassMetricUnavailable, validate_x_pass_version_available
    selection = args._component_selection
    available = set(args._available_component_inputs)
    reasons = {}
    if "physical_xpass" in selection.rendered_components:
        required = bool({"physical_xpass", "compare_pass_success"} & set(selection.explicit_component_groups))
        explicit_cache = getattr(args, "_explicit_physical_cache", bool(getattr(args, "physical_cache_dir", None) or getattr(args, "pc_xpass_id", None)))
        if getattr(args, "no_physical_cache", False):
            available.discard("physical_xpass")
            reasons["physical_xpass"] = "runtime physical cache is disabled"
        elif required:
            # The existing renderer performs full preflight, including row coverage.
            pass
        elif not (Path(cache_dir) / "metadata.json").exists():
            if explicit_cache:
                raise FileNotFoundError(f"Physical xPass metadata not found at {cache_dir}.")
            available.discard("physical_xpass")
            reasons["physical_xpass"] = f"no physical xPass cache at {cache_dir}"
        else:
            try:
                validate_x_pass_version_available(cache_dir, x_pass_version=version, metric=metric)
            except PhysicalXpassMetricUnavailable as exc:
                available.discard("physical_xpass")
                reasons["physical_xpass"] = str(exc)
    return bind_component_selection(args, filter_component_selection(selection, available, reasons=reasons))


def saved_component_inputs(raw, metadata=None):
    """Inspect source data before dataset adapters synthesize empty placeholders."""
    if raw is None:
        return set()
    if isinstance(raw, dict):
        return {name for name, table in raw.items() if table is not None and not table.empty}
    metadata = metadata or {}
    models = metadata.get("models", {}) or {}
    available = set()
    for group, components in GROUP_TO_COMPONENTS.items():
        for component in components:
            if component in raw and (raw[component].notna().any() or group in models or
                                     (component == "model_pass_success" and ("models" not in metadata or "pass_success" in models))):
                available.add(component)
    return available


def select_saved_components(args, raw, metadata, dataset, resolver, *, component_id=None, report=True, allow_empty=False):
    selection = resolve_component_selection(args)
    component_id = component_id or metadata.get("run_id")
    available = saved_component_inputs(raw, metadata)
    if {"model_pass_success", "compare_pass_success"} & set(selection.explicit_component_groups):
        from datatools.pass_success import require_model_pass_success
        require_model_pass_success(raw if raw is not None else {}, True)
    # Probe optional physical output using the same provenance policy as explicit output.
    bind_component_selection(args, selection, report=False)
    xpass, reason = None, None
    if "physical_xpass" in selection.rendered_components or "compare_pass_success" in selection.rendered_components:
        explicit = bool({"physical_xpass", "compare_pass_success"} & set(selection.explicit_component_groups))
        xpass, reason = probe_saved_xpass(args, metadata, dataset, resolver, component_id=component_id, required=explicit)
        if xpass is not None:
            available.add("physical_xpass")
    result = filter_component_selection(selection, available, reasons={"physical_xpass": reason} if reason else None,
                                        context=f"Component run {component_id or '<unidentified>'}", allow_empty=allow_empty)
    return bind_component_selection(args, result, report=report), xpass


def probe_saved_xpass(args, metadata, dataset, resolver, *, component_id=None, required=False):
    from scripts.saved_visualization_xpass import UnavailablePhysicalXpass
    from physical_pass_model import PhysicalXpassMetricUnavailable
    try:
        return resolver(args, metadata, dataset, component_id=component_id), None
    except (UnavailablePhysicalXpass, PhysicalXpassMetricUnavailable) as exc:
        if required or getattr(args, "physical_cache_dir", None) or getattr(args, "pc_xpass_id", None):
            raise
        return None, str(exc)
    except FileNotFoundError as exc:
        if required or getattr(args, "physical_cache_dir", None) or getattr(args, "pc_xpass_id", None):
            raise
        return None, str(exc)
