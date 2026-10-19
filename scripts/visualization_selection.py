from __future__ import annotations

import argparse
from dataclasses import dataclass

COMPONENT_GROUPS = (
    "action_intent",
    "pass_intent",
    "pass_success",
    "outcome_scoring",
    "outcome_conceding",
    "pass_score",
    "outcome_failure",
    "outcome_success",
)
OPTIONAL_COMPONENT_GROUPS = ("pass_height",)
SUCCESS_INTENT_GROUP = "success_intent"
PASS_SCORE_DEPENDENCIES = {"pass_success", "outcome_scoring", "outcome_conceding"}

GROUP_TO_COMPONENTS = {
    "action_intent": ("action_intent",),
    "pass_intent": ("pass_intent",),
    "pass_success": ("pass_success",),
    "pass_height": ("pass_height",),
    "outcome_scoring": ("outcome_scoring_success", "outcome_scoring_failure"),
    "outcome_conceding": ("outcome_conceding_success", "outcome_conceding_failure"),
    "pass_score": ("pass_score",),
    "outcome_failure": ("outcome_failure",),
    "outcome_success": ("outcome_success",),
    "success_intent": ("success_intent",),
}


@dataclass(frozen=True)
class ComponentSelection:
    requested_component_groups: list[str]
    disabled_component_groups: list[str]
    rendered_components: list[str]
    disabled_components: list[str]

    @property
    def required_component_groups(self) -> list[str]:
        required = set(self.requested_component_groups)
        if "model_pass_success" in self.rendered_components:
            required.add("pass_success")
        if "pass_score" in required:
            required.update(PASS_SCORE_DEPENDENCIES)
        if required & {"outcome_failure", "outcome_success"}:
            required.update({"outcome_scoring", "outcome_conceding"})
        return [group for group in GROUP_TO_COMPONENTS if group in required]


def add_component_selection_args(parser: argparse.ArgumentParser, *, include_success_intent: bool = False) -> None:
    parser.add_argument(
        "--show-model-pass-success", action="store_true",
        help="Render checkpoint pass success after offside handling and before physical xPass blending.",
    )
    groups = [*COMPONENT_GROUPS, *OPTIONAL_COMPONENT_GROUPS]
    if include_success_intent:
        groups.append(SUCCESS_INTENT_GROUP)
        parser.add_argument(
            "--success-intent", action="store_true",
            help="Include success-intent visualizations alongside selected components.",
        )

    for group in groups:
        option = group.replace("_", "-")
        parser.add_argument(
            f"--no-{option}",
            action="store_true",
            help=f"Do not render {group.replace('_', ' ')} visualizations.",
        )
        parser.add_argument(
            f"--only-{option}",
            action="store_true",
            help=f"Render only selected {group.replace('_', ' ')} visualizations. Repeat with other --only flags to add groups.",
        )
        if group in OPTIONAL_COMPONENT_GROUPS:
            parser.add_argument(
                f"--show-{option}",
                action="store_true",
                help=f"Render {group.replace('_', ' ')} visualizations.",
            )


def resolve_component_selection(
    args: argparse.Namespace,
    *,
    include_success_intent: bool = False,
) -> ComponentSelection:
    groups = [*COMPONENT_GROUPS, *OPTIONAL_COMPONENT_GROUPS]
    if include_success_intent:
        groups.append(SUCCESS_INTENT_GROUP)

    only_groups = {group for group in groups if getattr(args, f"only_{group}", False)}
    no_groups = {group for group in groups if getattr(args, f"no_{group}", False)}

    default_groups = [*COMPONENT_GROUPS]
    if include_success_intent:
        default_groups.append(SUCCESS_INTENT_GROUP)
    selected = set(only_groups) if only_groups else set(default_groups)
    selected.update(group for group in OPTIONAL_COMPONENT_GROUPS if getattr(args, f"show_{group}", False))
    compare_receivers = getattr(args, "compare_intended_receiver", False)
    if compare_receivers and getattr(args, "no_success_intent", False):
        raise ValueError("--compare-intended-receiver cannot be combined with --no-success-intent.")
    if include_success_intent and (getattr(args, "success_intent", False) or compare_receivers):
        selected.add(SUCCESS_INTENT_GROUP)
    selected -= no_groups

    rendered_components = [
        component
        for group in groups
        if group in selected
        for component in GROUP_TO_COMPONENTS[group]
    ]
    if getattr(args, "show_model_pass_success", False):
        rendered_components.append("model_pass_success")
    if not rendered_components:
        raise ValueError("No visualization components were selected after applying --only-* and --no-* flags.")

    disabled_groups = [group for group in groups if group not in selected]
    disabled_components = [
        component
        for group in disabled_groups
        for component in GROUP_TO_COMPONENTS[group]
    ]
    return ComponentSelection(
        requested_component_groups=[group for group in groups if group in selected],
        disabled_component_groups=disabled_groups,
        rendered_components=rendered_components,
        disabled_components=disabled_components,
    )
