"""Shared cached xPass selector arguments for downstream commands."""

from physical_pass_model import normalize_x_pass_version


def add_top_pass_selector(parser):
    version_action = parser._option_string_actions["--xpass-version"]
    parser.set_defaults(_xpass_version_default=version_action.default)
    version_action.default = None
    parser.add_argument(
        "--top-pass", "--top_pass", type=int, default=None,
        help="Select one cached pc-xPass top-pass count (alias for --xpass-version top-pass<N>).",
    )


def resolve_top_pass_selector(parser, args, *, pc_only=False):
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
        if normalized.startswith("top-pass") and not (pc_only or getattr(args, "pc_xpass", False)):
            parser.error("--top-pass and top-pass versions require --pc-xpass.")
        version = normalized
    setattr(args, dest, version if version is not None else args._xpass_version_default)
    del args._xpass_version_default
    return args
