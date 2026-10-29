"""Shared output selection, dependency resolution, and availability filtering."""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, replace

from scripts.visualization_pass_success import COMPARISON_COMPONENT

COMPONENT_GROUPS = (
    "action_intent", "pass_intent", "pass_success", "outcome_scoring",
    "outcome_conceding", "pass_score", "outcome_failure", "outcome_success",
)
OPTIONAL_COMPONENT_GROUPS = ("pass_height",)
SUCCESS_INTENT_GROUP = "success_intent"
PASS_SCORE_DEPENDENCIES = {"pass_success", "outcome_scoring", "outcome_conceding"}
GROUP_TO_COMPONENTS = {
    "action_intent": ("action_intent",), "pass_intent": ("pass_intent",),
    "pass_success": ("pass_success",),
    "outcome_scoring": ("outcome_scoring_success", "outcome_scoring_failure"),
    "outcome_conceding": ("outcome_conceding_success", "outcome_conceding_failure"),
    "pass_score": ("pass_score",), "outcome_failure": ("outcome_failure",),
    "outcome_success": ("outcome_success",), "pass_height": ("pass_height",),
    "success_intent": ("success_intent",), "model_pass_success": ("model_pass_success",),
    COMPARISON_COMPONENT: (COMPARISON_COMPONENT,), "physical_xpass": ("physical_xpass",),
}
MODEL_TASKS = ("action_intent", "pass_intent", "pass_success", "pass_height",
               "outcome_scoring", "outcome_conceding", "success_intent")


def source_inputs(component):
    from datatools.viz_helpers import DERIVED_COMPONENT_INPUTS
    if component == COMPARISON_COMPONENT:
        return ("model_pass_success", "physical_xpass", "pass_success")
    return DERIVED_COMPONENT_INPUTS.get(component, (component,))


@dataclass(frozen=True)
class ComponentSelection:
    requested_component_groups: list[str]
    disabled_component_groups: list[str]
    rendered_components: list[str]
    disabled_components: list[str]
    explicit_component_groups: list[str] = field(default_factory=list)
    skipped_components: dict[str, str] = field(default_factory=dict)
    selection_mode: str = "available"

    @property
    def required_component_groups(self):
        required = set(self.rendered_components)
        if {"model_pass_success", COMPARISON_COMPONENT} & required:
            required.add("pass_success")
        if "pass_score" in required:
            required.update(PASS_SCORE_DEPENDENCIES)
        if required & {"outcome_failure", "outcome_success"}:
            required.update({"outcome_scoring", "outcome_conceding"})
        for task in ("outcome_scoring", "outcome_conceding"):
            if {f"{task}_success", f"{task}_failure"} & required:
                required.add(task)
        return [group for group in GROUP_TO_COMPONENTS if group in required and group not in {"model_pass_success", "physical_xpass", COMPARISON_COMPONENT}]

    def metadata(self):
        return {name: getattr(self, name) for name in (
            "selection_mode", "requested_component_groups", "explicit_component_groups",
            "disabled_component_groups", "rendered_components", "disabled_components", "skipped_components",
        )}


class LegacyInclusion(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn(f"{option_string} is deprecated: available components are enabled by default; "
                      "use --only-* to select outputs or --no-* to exclude them.", FutureWarning, stacklevel=2)
        setattr(namespace, self.dest, True)


def add_component_selection_args(parser, *, include_success_intent=False):
    groups = [*COMPONENT_GROUPS, *OPTIONAL_COMPONENT_GROUPS, "model_pass_success", "physical_xpass"]
    if include_success_intent:
        groups.append(SUCCESS_INTENT_GROUP)
        parser.add_argument("--success-intent", action=LegacyInclusion, default=False,
                            help="Deprecated: explicitly include success intent; available outputs default on.")
    parser.add_argument("--compare-pass-success", action="store_true",
                        help="Stack model pass success, cached pc-xPass, and final pass success in one PNG.")
    parser.add_argument("--only-compare-pass-success", action="store_true",
                        help="Select the pass-success comparison PNG; combine with other --only-* outputs.")
    for group in groups:
        option = group.replace("_", "-")
        selector = parser.add_mutually_exclusive_group()
        selector.add_argument(f"--no-{option}", action="store_true", help=f"Exclude {option} plots; inputs may still be computed.")
        selector.add_argument(f"--only-{option}", action="store_true", help=f"Select {option}; combine --only-* flags to add outputs.")
        if group in {"pass_height", "model_pass_success", "physical_xpass"}:
            flag = f"--show-{option}"
            existing = parser._option_string_actions.get(flag)
            if existing:
                # Dataset helpers historically register physical-xPass display separately.
                existing.help = f"Deprecated: explicitly include {option}; available outputs default on."
            else:
                parser.add_argument(flag, action=LegacyInclusion, default=False,
                                    help=f"Deprecated: explicitly include {option}; available outputs default on.")


def resolve_component_selection(args, *, include_success_intent=False):
    groups = [g for g in GROUP_TO_COMPONENTS if include_success_intent or g != SUCCESS_INTENT_GROUP]
    only = {g for g in groups if getattr(args, f"only_{g}", False)}
    excluded = {g for g in groups if getattr(args, f"no_{g}", False)}
    conflicts = only & excluded
    if conflicts:
        group = next(g for g in groups if g in conflicts).replace("_", "-")
        raise ValueError(f"--only-{group} cannot be combined with --no-{group}.")
    if getattr(args, "compare_intended_receiver", False) and "success_intent" in excluded:
        raise ValueError("--compare-intended-receiver cannot be combined with --no-success-intent.")
    explicit = only | {g for g in groups if getattr(args, f"show_{g}", False)}
    if include_success_intent and (getattr(args, "success_intent", False) or getattr(args, "compare_intended_receiver", False)):
        explicit.add(SUCCESS_INTENT_GROUP)
    if getattr(args, "compare_pass_success", False):
        explicit.add(COMPARISON_COMPONENT)
    selected = (set(only) if only else set(groups) - {COMPARISON_COMPONENT}) | explicit
    selected -= excluded
    rendered = [c for g in groups if g in selected for c in GROUP_TO_COMPONENTS[g]]
    if not rendered:
        raise ValueError("No visualization components were selected after applying --only-* and --no-* flags.")
    return ComponentSelection(
        [g for g in groups if g in selected], [g for g in groups if g not in selected], rendered,
        [c for g in groups if g not in selected for c in GROUP_TO_COMPONENTS[g]],
        [g for g in groups if g in explicit and g in selected], selection_mode="only" if only else "available",
    )


def available_model_components(model_ids):
    available = {c for task in MODEL_TASKS if model_ids.get(task) for c in GROUP_TO_COMPONENTS[task]}
    if model_ids.get("pass_success"):
        available.add("model_pass_success")
    return available


def filter_component_selection(selection, available, *, reasons=None, context="Selected source", allow_empty=False):
    """Filter defaults, but require every explicitly included output's inputs."""
    available = set(available)
    reasons = reasons or {}
    rendered, skipped = [], dict(selection.skipped_components)
    for group in selection.requested_component_groups:
        components = GROUP_TO_COMPONENTS[group]
        missing = sorted({leaf for c in components for leaf in source_inputs(c)} - available)
        if missing:
            reason = "; ".join(reasons.get(leaf, f"missing {leaf} inputs") for leaf in missing)
            if group in selection.explicit_component_groups:
                raise ValueError(f"{context}: {group} visualization unavailable: {reason}.")
            skipped[group] = f"{context}: {reason}"
        else:
            skipped.pop(group, None)
            rendered.extend(components)
    if not rendered and not allow_empty:
        raise ValueError(f"{context}: no visualization outputs are available. " + "; ".join(skipped.values()))
    return replace(selection, rendered_components=rendered, skipped_components=skipped)


def bind_component_selection(args, selection, *, report=True):
    args._component_selection = selection
    if report:
        for group, reason in selection.skipped_components.items():
            print(f"Skipping {group} visualization: {reason}")
    return selection


def selection_metadata(args):
    selection = getattr(args, "_component_selection", None)
    if selection is None:
        return {}
    metadata = selection.metadata()
    for group in ("model_pass_success", "pass_height", "physical_xpass"):
        metadata[f"show_{group}"] = group in selection.rendered_components
    return metadata
