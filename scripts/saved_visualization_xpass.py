"""Resolve pc-xPass panels from saved component provenance, without inference."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import pc_xpass_match_cache as match_cache
import pc_xpass_versions as versions
from physical_pass_model import (
    PC_XPASS_SOURCE,
    normalize_x_pass_version,
    physical_xpass_metric_for_version,
    validate_x_pass_version_available,
)
from project_config import PROJECT_ROOT
from scripts.xpass_cli import add_top_pass_selector, resolve_top_pass_selector
from scripts.visualization_pass_success import compare_pass_success_requested, needs_physical_xpass


def add_saved_xpass_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--show-physical-xpass", action="store_true",
                        help="Render cached pc-xPass; inherit settings from the component run.")
    parser.add_argument("--physical-cache-dir", help="Explicit pc-xPass dataset cache directory override.")
    parser.add_argument("--pc-xpass", "--pc_xpass", dest="pc_xpass", action="store_true",
                        help="Compatibility flag; saved-component visualization supports only pc-xPass.")
    parser.add_argument("--xpass-version", "--x-pass-version", "--x_pass_version",
                        dest="x_pass_version", default=None,
                        help="Override the inherited pc-xPass metric: max, top<N>, top-pass<N>, or uncertainty.")
    add_top_pass_selector(parser)
    versions.add_selection_argument(parser, repeatable=True)
    parser._option_string_actions["--pc-xpass-id"].help = (
        "Override the component's pc-xPass cache ID; repeat with --compare. Never defaults to latest."
    )


def resolve_saved_xpass_selectors(parser: ArgumentParser, args: Namespace) -> None:
    versions.check_selectors(args)
    resolve_top_pass_selector(parser, args, pc_only=True)
    if args.x_pass_version is not None:
        try:
            physical_xpass_metric_for_version(args.x_pass_version, pc_xpass=True)
        except ValueError as exc:
            parser.error(str(exc))


@dataclass(frozen=True)
class SavedXpassSettings:
    cache_dir: Path
    pc_xpass_id: str | None
    namespace: str
    x_pass_version: str
    metric: str
    component_usage: str
    selection: dict[str, str]
    overrides: dict[str, str]

    def metadata(self) -> dict[str, Any]:
        return {
            "physical_xpass_runtime_source": PC_XPASS_SOURCE,
            "physical_cache_dir": str(self.cache_dir),
            "pc_xpass_id": self.pc_xpass_id,
            "pc_xpass_namespace": self.namespace,
            "x_pass_version": self.x_pass_version,
            "physical_xpass_metric": self.metric,
            "physical_xpass_resolution": {
                "component_usage": self.component_usage,
                "selection": self.selection,
                "inherited_fields": [key for key, origin in self.selection.items()
                                     if origin.startswith("component.")],
                "overrides": self.overrides,
            },
        }


def saved_xpass_metadata(settings: SavedXpassSettings | None) -> dict[str, Any]:
    if settings is not None:
        return settings.metadata()
    return dict.fromkeys(("physical_xpass_runtime_source", "physical_cache_dir", "pc_xpass_id",
                          "pc_xpass_namespace", "x_pass_version", "physical_xpass_metric",
                          "physical_xpass_resolution"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(*items: tuple[Any, str]) -> tuple[Any, str]:
    return next(((value, origin) for value, origin in items if value is not None and value != ""),
                (None, ""))


def _component_usage(metadata: Mapping[str, Any], effective: Mapping[str, Any], dataset: str) -> str:
    required = _mapping(metadata.get("physical_xpass_cache_summary")).get("physical_xpass_required")
    used = required if required is not None else effective.get("use_physical_xpass")
    if used is False:
        return "disabled"
    source = metadata.get("physical_xpass_runtime_source")
    if source and source != PC_XPASS_SOURCE:
        return "ordinary_physical"
    if dataset == "hawkeye_loc" and metadata.get("inference_mode") in {"loc", "freeze"}:
        return "pc_xpass"
    if used is True:
        if source == PC_XPASS_SOURCE or (source is None and effective.get("pc_xpass") is True):
            return "pc_xpass"
        if effective.get("pc_xpass") is False:
            return "ordinary_physical"
    return "unknown"


def resolve_saved_xpass(
    args: Namespace,
    metadata: Mapping[str, Any],
    dataset: str,
    *,
    component_id: str | None = None,
) -> SavedXpassSettings | None:
    """Resolve and preflight a panel once, leaving CLI arguments unmodified."""
    if not needs_physical_xpass(args):
        return None
    identifier = component_id or metadata.get("run_id") or "<unidentified>"
    try:
        result = _resolve(args, metadata, dataset)
    except (ValueError, FileNotFoundError) as exc:
        raise type(exc)(f"Component run {identifier!r}: {exc}") from exc
    print(f"Component run {identifier}: pc-xPass cache {result.cache_dir}; metric {result.x_pass_version}")
    return result


def _resolve(args: Namespace, metadata: Mapping[str, Any], dataset: str) -> SavedXpassSettings:
    invocation = _mapping(metadata.get("invocation"))
    effective = _mapping(invocation.get("effective_cli_options"))
    usage = _component_usage(metadata, effective, dataset)
    explicit_dir = getattr(args, "physical_cache_dir", None)
    explicit_id = getattr(args, "pc_xpass_id", None)
    explicit_version = getattr(args, "x_pass_version", None)
    if explicit_id and explicit_dir:
        raise ValueError("--pc-xpass-id cannot be combined with --physical-cache-dir.")
    if usage != "pc_xpass" and not ((explicit_dir or explicit_id) and explicit_version):
        option = ("--compare-pass-success" if compare_pass_success_requested(args) else
                  "--only-physical-xpass" if getattr(args, "only_physical_xpass", False) else "--show-physical-xpass")
        raise ValueError(
            f"pc-xPass usage is {usage}. {option} requires both an explicit "
            "--pc-xpass-id or --physical-cache-dir and --xpass-version (or --top-pass)."
        )

    # Never inherit incidental/default settings from runs that did not use pc-xPass.
    saved = metadata if usage == "pc_xpass" else {}
    options = effective if usage == "pc_xpass" else {}
    run_id, id_origin = _first(
        (explicit_id, "explicit.--pc-xpass-id"),
        (saved.get("pc_xpass_id"), "component.pc_xpass_id"),
        (options.get("pc_xpass_id"), "component.invocation.effective_cli_options.pc_xpass_id"),
    )
    directory, dir_origin = _first(
        (explicit_dir, "explicit.--physical-cache-dir"),
        (saved.get("physical_cache_dir"), "component.physical_cache_dir"),
        (saved.get("pc_xpass_cache_dir"), "component.pc_xpass_cache_dir"),
        (options.get("physical_cache_dir"), "component.invocation.effective_cli_options.physical_cache_dir"),
        (options.get("pc_xpass_cache_dir"), "component.invocation.effective_cli_options.pc_xpass_cache_dir"),
    )
    location = dataset == "hawkeye_loc"
    if explicit_dir:
        run_id = None
    if run_id:
        root = versions.version_root(run_id, location=location)
        versions.read_metadata(root)
        cache_dir = (root if location else root / dataset).resolve()
        cache_origin = id_origin
    elif directory:
        cache_dir = Path(directory)
        if not explicit_dir and not cache_dir.is_absolute():
            base = Path(invocation.get("working_directory") or PROJECT_ROOT)
            if not base.is_absolute():
                base = PROJECT_ROOT / base
            cache_dir = base / cache_dir
        cache_dir = cache_dir.resolve()
        cache_origin = dir_origin
    else:
        raise ValueError("No recorded pc-xPass cache; specify --pc-xpass-id or --physical-cache-dir.")

    # A pc metric column can identify its selector; generic topmean cannot.
    saved_metric = saved.get("physical_xpass_metric")
    metric_version = None
    if isinstance(saved_metric, str) and re.fullmatch(r"(?:max|uncertainty|top[1-9][0-9]*|top_pass[1-9][0-9]*)_xpass", saved_metric):
        metric_version = saved_metric.removesuffix("_xpass").replace("top_pass", "top-pass")
    version, metric_origin = _first(
        (explicit_version, "explicit.--xpass-version"),
        (saved.get("x_pass_version"), "component.x_pass_version"),
        (metric_version, "component.physical_xpass_metric"),
        (_mapping(saved.get("pc_xpass_settings")).get("x_pass_version"), "component.pc_xpass_settings.x_pass_version"),
        (options.get("x_pass_version"), "component.invocation.effective_cli_options.x_pass_version"),
    )
    if version is None:
        raise ValueError("No recorded pc-xPass metric; specify --xpass-version or --top-pass.")
    version = normalize_x_pass_version(version)
    metric = physical_xpass_metric_for_version(version, pc_xpass=True)
    cache_metadata = versions.read_metadata(cache_dir)
    if not isinstance(cache_metadata, dict):
        raise ValueError(f"Cache metadata at {cache_dir} must be a JSON object identifying pc_xpass.")
    source = cache_metadata.get("source") or cache_metadata.get("metric_family")
    if source != PC_XPASS_SOURCE:
        raise ValueError(f"Cache at {cache_dir} must identify itself as pc_xpass; found {source!r}.")
    validate_x_pass_version_available(cache_dir, source=PC_XPASS_SOURCE,
                                     x_pass_version=version, metric=metric)
    if dataset in {"skillcorner", "sportec"}:
        match_cache.require_format(cache_dir)
    run_id = run_id or cache_metadata.get("pc_xpass_id")
    overrides = {key: str(value) for key, value in (
        ("physical_cache_dir", explicit_dir), ("pc_xpass_id", explicit_id),
        ("x_pass_version", explicit_version),
    ) if value is not None}
    return SavedXpassSettings(cache_dir, run_id, "hawkeye_loc" if location else "normal",
                              version, metric, usage,
                              {"cache": cache_origin, "metric": metric_origin}, overrides)
