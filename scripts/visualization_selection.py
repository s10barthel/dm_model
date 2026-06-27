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
)
OPTIONAL_COMPONENT_GROUPS = ("pass_height",)
INTENDED_RECIPIENT_GROUP = "intended_recipient"
PASS_SCORE_DEPENDENCIES = {"pass_success", "outcome_scoring", "outcome_conceding"}

GROUP_TO_COMPONENTS = {
    "action_intent": ("action_intent",),
    "pass_intent": ("pass_intent",),
    "pass_success": ("pass_success",),
    "pass_height": ("pass_height",),
    "outcome_scoring": ("outcome_scoring_success", "outcome_scoring_failure"),
    "outcome_conceding": ("outcome_conceding_success", "outcome_conceding_failure"),
    "pass_score": ("pass_score",),
    "intended_recipient": ("intended_recipient",),
}


@dataclass(frozen=True)
class ComponentSelection:
    requested_component_groups: list[str]
    disabled_component_groups: list[str]
    rendered_components: list[str]
    disabled_components: list[str]


def add_component_selection_args(parser: argparse.ArgumentParser, *, include_intended_recipient: bool = False) -> None:
    groups = [*COMPONENT_GROUPS, *OPTIONAL_COMPONENT_GROUPS]
    if include_intended_recipient:
        groups.append(INTENDED_RECIPIENT_GROUP)

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
    include_intended_recipient: bool = False,
) -> ComponentSelection:
    groups = [*COMPONENT_GROUPS, *OPTIONAL_COMPONENT_GROUPS]
    if include_intended_recipient:
        groups.append(INTENDED_RECIPIENT_GROUP)

    only_groups = {group for group in groups if getattr(args, f"only_{group}", False)}
    no_groups = {group for group in groups if getattr(args, f"no_{group}", False)}

    default_groups = [*COMPONENT_GROUPS]
    if include_intended_recipient:
        default_groups.append(INTENDED_RECIPIENT_GROUP)
    selected = set(only_groups) if only_groups else set(default_groups)
    selected.update(group for group in OPTIONAL_COMPONENT_GROUPS if getattr(args, f"show_{group}", False))
    selected -= no_groups

    if "pass_score" in selected and not PASS_SCORE_DEPENDENCIES <= selected:
        missing = sorted(PASS_SCORE_DEPENDENCIES - selected)
        raise ValueError(
            "pass_score visualization requires pass_success, outcome_scoring, and outcome_conceding. "
            "Also select/enable: "
            + ", ".join(missing)
        )

    rendered_components = [
        component
        for group in groups
        if group in selected
        for component in GROUP_TO_COMPONENTS[group]
    ]
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
