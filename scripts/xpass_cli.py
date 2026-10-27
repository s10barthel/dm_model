"""Shared cached xPass selector arguments for downstream commands."""

import argparse
import math

from physical_pass_model import normalize_x_pass_version


class XPassVersionSelection(argparse.Action):
    """Retain the order of explicit versions and their top-pass aliases."""

    def __call__(self, parser, namespace, values, option_string=None):
        selections = list(getattr(namespace, "_xpass_version_selections", []))
        selections.append((self.dest, values))
        namespace._xpass_version_selections = selections
        previous = getattr(namespace, self.dest, None)
        setattr(namespace, self.dest, values if previous is None else _selection_list(previous) + [values])


def _selection_list(value):
    return [] if value is None else value if isinstance(value, list) else [value]


def add_top_pass_selector(parser, *, repeatable=False):
    version_action = parser._option_string_actions["--xpass-version"]
    parser.set_defaults(_xpass_version_default=version_action.default)
    version_action.default = None
    parser.add_argument(
        "--top-pass", "--top_pass", type=int, default=None,
        **({"action": XPassVersionSelection} if repeatable else {}),
        help=("Select a cached pc-xPass top-pass count (alias for --xpass-version top-pass<N>)."
              + (" Repeat or mix with --xpass-version using --compare." if repeatable else "")),
    )
    if repeatable:
        parser.set_defaults(_xpass_repeatable=True)


def resolve_top_pass_selector(parser, args, *, pc_only=False):
    if getattr(args, "_xpass_repeatable", False):
        return _resolve_repeatable_versions(parser, args, pc_only=pc_only)
    # Only visualization parsers opt into repeatable cache IDs.
    if getattr(args, "compare", False) and isinstance(getattr(args, "pc_xpass_id", None), list):
        args.pc_xpass = True
    dest = parser._option_string_actions["--xpass-version"].dest
    version = getattr(args, dest)
    count = args.top_pass
    if count is not None:
        if count < 1:
            parser.error("--top-pass must be a positive integer.")
        selected = f"top-pass{count}"
        try:
            if version is not None and normalize_x_pass_version(version) != selected:
                parser.error("--top-pass conflicts with --xpass-version.")
        except ValueError as exc:
            parser.error(str(exc))
        version = selected
    if version is not None:
        try:
            normalized = normalize_x_pass_version(version)
        except ValueError as exc:
            parser.error(str(exc))
        if (normalized.startswith("top-pass") or normalized == "uncertainty") and not (pc_only or getattr(args, "pc_xpass", False)):
            parser.error("--top-pass, top-pass and uncertainty versions require --pc-xpass.")
        version = normalized
    setattr(args, dest, version if version is not None else args._xpass_version_default)
    del args._xpass_version_default
    return args


def _resolve_repeatable_versions(parser, args, *, pc_only):
    dest = parser._option_string_actions["--xpass-version"].dest
    selections = getattr(args, "_xpass_version_selections", [])
    if any(value < 1 for name, value in selections if name == "top_pass"):
        parser.error("--top-pass must be a positive integer.")
    if args.xpass_weight is None:
        args.xpass_weight = "v3"
    if not args.compare:
        if any(sum(name == selector for name, _ in selections) > 1 for selector in (dest, "top_pass")):
            parser.error("Repeated xPass version selectors require --compare.")
        # Retain the existing matching/conflicting alias behavior for a single selection.
        del args._xpass_repeatable
        if hasattr(args, "_xpass_version_selections"):
            del args._xpass_version_selections
        return resolve_top_pass_selector(parser, args, pc_only=pc_only)
    try:
        versions = [normalize_x_pass_version(f"top-pass{value}" if name == "top_pass" else value)
                    for name, value in selections]
    except ValueError as exc:
        parser.error(str(exc))
    if any(len(_selection_list(value)) > 1 for value in (args.pc_xpass_id, versions, args.xpass_weight)):
        args.pc_xpass = True
    if not (pc_only or args.pc_xpass) and any(value.startswith("top-pass") or value == "uncertainty" for value in versions):
        parser.error("--top-pass, top-pass and uncertainty versions require --pc-xpass.")
    setattr(args, dest, versions if len(versions) > 1 else versions[0] if versions else args._xpass_version_default)
    del args._xpass_version_default, args._xpass_repeatable
    if hasattr(args, "_xpass_version_selections"):
        del args._xpass_version_selections
    return args


def validate_xpass_weight_options(parser, args):
    """Validate explicit tuning flags against all selected weighting versions."""
    weights = _selection_list(args.xpass_weight)
    for key, flag in (("v4_power", "--v4-power"), ("v4_zero", "--v4-zero")):
        value = getattr(args, key)
        if value is not None:
            if not math.isfinite(value) or value <= 0:
                parser.error(f"{flag} must be positive.")
            if "v4" not in weights:
                parser.error(f"{flag} is only valid with --xpass-weight v4.")
    if args.v4_discount is not None and "v4" not in weights:
        parser.error("--discount is only valid with --xpass-weight v4.")
    if args.v4_discount is None:
        args.v4_discount = True
    if (args.v5_intent_threshold is not None or args.v5_discount is not None) and "v5" not in weights:
        parser.error("v5 options are only valid with --xpass-weight v5.")
    args.v5_intent_threshold = 0.01 if args.v5_intent_threshold is None else args.v5_intent_threshold
    args.v5_discount = True if args.v5_discount is None else args.v5_discount
    if not math.isfinite(args.v5_intent_threshold) or args.v5_intent_threshold <= 0:
        parser.error("--v5-intent-threshold must be a positive finite float.")
