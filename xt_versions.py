"""Paths and publication checks for immutable xT builds."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from project_config import XT_DIR, XT_MATCH_DIR

REQUIRED_FILES = (
    "xT.csv", "xT_grid.csv", "xT_source_grid.csv", "xT_xy_surface.csv",
    "xT_glm_fit_sample.csv", "xT_xy_surface_3d.png", "fit_metadata.json",
)
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_RESERVED = {"matches", "latest", "latest.json", ".staging", "con", "prn", "aux", "nul"}
_RESERVED.update({f"com{number}" for number in range(1, 10)})
_RESERVED.update({f"lpt{number}" for number in range(1, 10)})


def required_files(metadata):
    if metadata.get("surface_type", metadata.get("xt_model_type")) == "log_progression":
        return tuple(name for name in REQUIRED_FILES if name not in {"xT_source_grid.csv", "xT_glm_fit_sample.csv"})
    return REQUIRED_FILES


@dataclass(frozen=True)
class XTVersion:
    version: str
    root: Path
    metadata: dict
    metadata_sha256: str

    @property
    def match_dir(self) -> Path:
        return self.root / "matches"


def validate_xt_version_id(value: str) -> str:
    version = str(value)
    if not _VERSION_PATTERN.fullmatch(version) or version.lower() in _RESERVED:
        raise ValueError(f"Invalid xT version ID: {value!r}.")
    return version


def get_xt_version_root(version: str, *, xt_dir: Path = XT_DIR) -> Path:
    return Path(xt_dir) / validate_xt_version_id(version)


def resolve_xt_version(
    requested: str,
    *,
    xt_dir: Path = XT_DIR,
    required_match_ids: list[str] | tuple[str, ...] = (),
    split_manifest_id: str | None = None,
) -> XTVersion:
    if not requested:
        raise ValueError("An explicit --xt-version is required; use 'latest' explicitly if desired.")
    root_dir = Path(xt_dir)
    version = str(requested)
    if version == "latest":
        pointer = root_dir / "latest.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"No latest xT version is registered at {pointer}.")
        version = str(json.loads(pointer.read_text(encoding="utf-8"))["version"])
    root = get_xt_version_root(version, xt_dir=root_dir)
    metadata_path = root / "fit_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"xT version {version!r} has no fit metadata at {metadata_path}.")
    content = metadata_path.read_bytes()
    metadata = json.loads(content)
    if metadata.get("version") != version or metadata.get("status") != "complete":
        raise ValueError(f"xT version {version!r} is not a completed selectable build.")
    if split_manifest_id and metadata.get("split_manifest_id") != split_manifest_id:
        raise ValueError(
            f"xT version {version!r} uses split {metadata.get('split_manifest_id')!r}, "
            f"but the feature run uses {split_manifest_id!r}."
        )
    from datatools.xt_targets import surface_from_metadata
    if metadata.get("surface_type") == "log_progression":
        surface_from_metadata(metadata)  # Validate the persisted analytical definition.
    missing = [str(root / name) for name in required_files(metadata) if not (root / name).is_file()]
    missing += [str(root / "matches" / f"{match_id}.csv") for match_id in required_match_ids
                if not (root / "matches" / f"{match_id}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"xT version {version!r} is missing required outputs: {', '.join(missing[:5])}")
    return XTVersion(version, root, metadata, hashlib.sha256(content).hexdigest())


def xt_match_dir_for_feature_root(feature_root: Path | None) -> Path:
    """Use a run's pinned xT sidecars, with legacy fallback for old runs."""
    if feature_root is None:
        return XT_MATCH_DIR
    metadata_path = Path(feature_root) / "metadata.json"
    if not metadata_path.is_file():
        return XT_MATCH_DIR
    version = json.loads(metadata_path.read_text(encoding="utf-8-sig")).get("xt_version")
    return resolve_xt_version(version).match_dir if version else XT_MATCH_DIR


def xt_options_for_metadata(metadata):
    """Construct Match options for runtime and non-CLI feature regeneration."""
    from datatools.xt_targets import target_settings, surface_from_metadata
    settings = target_settings(metadata)
    selected = resolve_xt_version(metadata["xt_version"]) if metadata.get("xt_version") else None
    if selected and metadata.get("xt_fit_metadata_sha256") not in (None, selected.metadata_sha256):
        raise ValueError("Pinned xT surface metadata changed after feature generation.")
    if settings["representation"] == "gain" and selected is None:
        raise ValueError("Gain targets require a pinned xT version.")
    return dict(xt_targets=settings,
                xt_surface=surface_from_metadata(selected.metadata) if settings["representation"] == "gain" else None,
                xt_surface_metadata=selected.metadata if selected else None,
                xt_match_dir=selected.match_dir if selected else None)


def xt_options_for_feature_root(feature_root):
    path = Path(feature_root) / "metadata.json" if feature_root is not None else None
    metadata = json.loads(path.read_text(encoding="utf-8-sig")) if path and path.is_file() else {}
    options = xt_options_for_metadata(metadata)
    if options["xt_match_dir"] is None:
        options["xt_match_dir"] = XT_MATCH_DIR
    return options
