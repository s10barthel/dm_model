"""Version lifecycle and CLI selection for pc-xPass caches."""
from __future__ import annotations

import json
import math
import re
import sys
import uuid
import warnings
from argparse import ArgumentParser, Namespace
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import project_config as config
import reachability as reach

SETTING_KEYS = set("""ball_dec physical_eps min_speed max_speed speed_step angle_step radial_gridsize
    top_n top_n_values top_pass_values top_xt reaction_time reaction_time_mode dist_pass_div dist_pass_min dist_pass_max
    max_player_speed max_player_speed_off max_player_speed_def lane_power lane_inflection_point control_power
    control_inflection_point endpoint_normalization boost_def_endpoint_control use_position_discount
    position_discount_power position_discount_distance consider_teammates ignore_teammates_lane_survival
    ignore_teammates_control export_max export_topmean export_noise_kernel x_pass_version
    pass_height_model_id""".split())
SETTING_KEYS.update({"margin", "reachability_fingerprint", *("reachability_" + k for k in reach.DEFAULTS)})
RESERVED = {"reachability", "hawkeye_loc", "sportec", "skillcorner", "benchmark", "hawkeye", "latest", "latest.json"}


def validate_id(value: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value.lower() in RESERVED or value.endswith("."):
        raise ValueError(f"Invalid or reserved pc-xPass ID: {value!r}")
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError(f"Invalid pc-xPass ID: {value!r}")
    return value


def version_root(run_id: str, location: bool = False) -> Path:
    base = config.PC_XPASS_DIR / "hawkeye_loc" if location else config.PC_XPASS_DIR
    return base / validate_id(run_id)


def read_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"pc-xPass metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def add_selection_argument(parser: ArgumentParser) -> None:
    parser.add_argument("--pc-xpass-id", help="pc-xPass cache version; normal consumers default to latest.")


def check_selectors(args: Namespace) -> None:
    if getattr(args, "pc_xpass_id", None) and not getattr(args, "_pc_version_root", None) and any(
        getattr(args, key, None) for key in ("pc_xpass_cache_dir", "physical_cache_dir", "lane_survival_cache_dir")
    ):
        raise ValueError("--pc-xpass-id cannot be combined with a pc-xPass cache directory override.")


def settings(args: Namespace) -> dict[str, Any]:
    excluded = {"pass_height_model_id"} if getattr(args, "_pc_location", False) else set()
    result = {key: getattr(args, key) for key in sorted(SETTING_KEYS - excluded) if hasattr(args, key)}
    if getattr(args, "margin", "tta") == "tta":
        result = {k: v for k, v in result.items() if k != "margin" and not k.startswith("reachability_")}
    # Store the CLI representation so existing validation can be reused on load.
    if result.get("reaction_time_mode") == "dist_pass":
        result["reaction_time"] = "dist_pass"
    return result


def prepare_generation_args(parser: ArgumentParser, args: Namespace, argv: list[str] | None = None, *, location: bool = False) -> None:
    check_selectors(args)
    tokens = list(sys.argv[1:] if argv is None else argv)
    explicit = {action.dest for token in tokens for action in [parser._option_string_actions.get(token.split("=", 1)[0])] if action}
    args._pc_explicit = explicit
    args._pc_location = location
    args.pc_xpass_overrides = []
    run_id = getattr(args, "pc_xpass_id", None)
    directory = getattr(args, "pc_xpass_cache_dir", None) if location else None
    if run_id:
        root = version_root(run_id, location)
        if location or root.exists():
            metadata = read_metadata(root)
            if "generation_settings" not in metadata:
                raise ValueError(f"Not a versioned pc-xPass cache: {root}")
            for key, value in metadata["generation_settings"].items():
                if key not in SETTING_KEYS or (location and key == "pass_height_model_id"):
                    continue
                requested = getattr(args, key, None)
                if key == "reaction_time" and isinstance(requested, str) and requested != "dist_pass":
                    try:
                        requested = float(requested)
                    except ValueError:
                        pass
                if key in explicit and requested != value:
                    message = f"--{key.replace('_', '-')}: requested {requested!r}; cached value {value!r}"
                    if not location:
                        raise ValueError(message + "; create a new pc-xPass version for different settings.")
                    warnings.warn(message + "; using cached value.", stacklevel=2)
                    args.pc_xpass_overrides.append(message)
                setattr(args, key, value)
            if location and getattr(args, "top_pass", None) is not None:
                selected = f"top-pass{args.top_pass}"
                cached_version = metadata["generation_settings"].get("x_pass_version", "top10")
                if selected != cached_version:
                    message = f"--top-pass: requested {args.top_pass!r}; cached selector {cached_version!r}; using cached value."
                    warnings.warn(message, stacklevel=2)
                    args.pc_xpass_overrides.append(message)
                args.top_pass = None
            recorded_margin = metadata["generation_settings"].get("margin", "tta")
            if "margin" in explicit and args.margin != recorded_margin:
                raise ValueError("Margin differs from selected pc-xPass version; create a new version.")
            args.margin = recorded_margin
            args._pc_existing_metadata = metadata
        args._pc_version_root = str(root)
    elif directory:
        root = Path(directory)
        if (root / "metadata.json").exists():
            legacy = read_metadata(root)
            cached_dec = float(legacy.get("ball_dec", 0.0))
            if "ball_dec" in explicit and float(args.ball_dec) != cached_dec:
                raise ValueError("Explicit cache directory uses different ball deceleration; create a new version.")
            args.ball_dec = cached_dec
    if not math.isfinite(float(args.ball_dec)) or args.ball_dec < 0:
        parser.error("--ball-dec must be a finite non-negative number")


def start_generation(args: Namespace, *, location: bool = False) -> None:
    if not getattr(args, "pc_xpass", False):
        return
    directory = getattr(args, "pc_xpass_cache_dir", None) if location else None
    if directory and not getattr(args, "pc_xpass_id", None):
        return
    if not getattr(args, "pc_xpass_id", None):
        args.pc_xpass_id = config.generate_run_id("pc_xpass") + "_" + uuid.uuid4().hex[:8]
    root = version_root(args.pc_xpass_id, location)
    args._pc_location = location
    args._pc_version_root = str(root)
    args.pc_xpass_namespace = "hawkeye_loc" if location else "normal"
    if location:
        args.pc_xpass_cache_dir = str(root)
    if getattr(args, "dry_run", False):
        return
    old = read_metadata(root) if (root / "metadata.json").exists() else {}
    effective = settings(args)
    if old.get("generation_settings", effective) != effective:
        raise ValueError("Effective pc-xPass settings differ from selected version.")
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(old)
    metadata.update(pc_xpass_id=args.pc_xpass_id, namespace=args.pc_xpass_namespace,
                    schema_version=1, physics_version=2, ball_dec=float(args.ball_dec),
                    generation_settings=effective, created_at=old.get("created_at", now),
                    updated_at=now, status="running")
    history = list(old.get("invocations", []))
    history.append({"started_at": now, "arguments": {k: v for k, v in vars(args).items() if not k.startswith("_")}, "status": "running"})
    metadata["invocations"] = history
    atomic_json(root / "metadata.json", metadata)
    args._pc_generation_started = True
    print(f"pc-xPass cache version: {args.pc_xpass_id} ({args.pc_xpass_namespace})")


def finish_generation(args: Namespace, *, success: bool = True, coverage: dict[str, Any] | None = None) -> None:
    if not getattr(args, "_pc_generation_started", False) or getattr(args, "dry_run", False):
        return
    root = Path(args._pc_version_root)
    if not (root / "metadata.json").exists():
        return
    metadata = read_metadata(root)
    now = datetime.now(timezone.utc).isoformat()
    metadata.update(updated_at=now, status="completed" if success else "incomplete")
    if coverage is not None:
        for dataset, summary in coverage.items():
            metadata.setdefault("coverage", {}).setdefault(dataset, []).append(summary)
    metadata["invocations"][-1].update(finished_at=now, status=metadata["status"], coverage=coverage)
    atomic_json(root / "metadata.json", metadata)
    if success and not getattr(args, "_pc_location", False):
        atomic_json(config.PC_XPASS_DIR / "latest.json", {"run_id": args.pc_xpass_id, "updated_at": now})
    args._pc_generation_started = False


def cache_dir(source: str, args: Namespace) -> Path:
    """Resolve once per invocation, checking dataset coverage for readers."""
    check_selectors(args)
    root_value = getattr(args, "_pc_version_root", None)
    if root_value:
        root = Path(root_value)
        result = root if source == "hawkeye_loc" else root / source
        if not hasattr(args, "_pc_explicit") and not (result / "metadata.json").is_file():
            raise FileNotFoundError(f"Selected pc-xPass version has no {source} dataset cache.")
        return result
    run_id = getattr(args, "pc_xpass_id", None)
    location = source == "hawkeye_loc"
    if not run_id:
        if location:
            raise ValueError("Location visualization requires a cache ID from component metadata or an explicit override.")
        latest = config.PC_XPASS_DIR / "latest.json"
        if not latest.exists():
            raise FileNotFoundError("No latest pc-xPass version; generate a cache or specify an explicit directory.")
        run_id = json.loads(latest.read_text(encoding="utf-8-sig"))["run_id"]
    root = version_root(run_id, location)
    read_metadata(root)
    result = root if location else root / source
    if not (result / "metadata.json").exists():
        raise FileNotFoundError(f"pc-xPass version {run_id!r} has no {source} dataset cache.")
    args.pc_xpass_id = run_id
    args.pc_xpass_namespace = "hawkeye_loc" if location else "normal"
    args._pc_version_root = str(root)
    return result


def lane_cache_dir(source: str, args: Namespace, models: dict[str, Any]) -> str | None:
    if not any(model is not None and getattr(model, "args", {}).get("lane_survival", False) for model in models.values()):
        return None
    override = getattr(args, "lane_survival_cache_dir", None) or getattr(args, "physical_cache_dir", None)
    result = str(override or cache_dir(source, args))
    if getattr(args, "pc_xpass_id", None):
        for model in models.values():
            if model is not None and getattr(model, "args", {}).get("lane_survival", False):
                model.args["pc_xpass_id"] = args.pc_xpass_id
                model.args["pc_xpass_namespace"] = getattr(args, "pc_xpass_namespace", "normal")
    return result


def model_cache_dir(source: str, model_args: dict[str, Any]) -> Path:
    args = Namespace(pc_xpass_id=model_args.get("pc_xpass_id"))
    result = cache_dir(source, args)
    model_args["pc_xpass_id"] = args.pc_xpass_id
    model_args["pc_xpass_namespace"] = args.pc_xpass_namespace
    return result
