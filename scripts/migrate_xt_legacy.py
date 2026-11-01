"""Move the historical unversioned xT outputs into data/xT/legacy once."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import EVENT_SYNCED_DIR, XT_DIR, XT_LEGACY_VERSION, atomic_write_json
from xt_versions import REQUIRED_FILES, get_xt_version_root


def migrate_legacy_xt(*, xt_dir: Path = XT_DIR, event_dir: Path = EVENT_SYNCED_DIR) -> Path:
    xt_dir = Path(xt_dir)
    destination = get_xt_version_root(XT_LEGACY_VERSION, xt_dir=xt_dir)
    if destination.exists():
        raise FileExistsError(f"Legacy xT version already exists at {destination}.")
    source_names = [*REQUIRED_FILES, "matches"]
    missing = [name for name in source_names if not (xt_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Unversioned xT outputs are missing: {missing}.")
    metadata_path = xt_dir / "fit_metadata.json"
    original_metadata = metadata_path.read_bytes()
    metadata = json.loads(original_metadata)
    if metadata.get("fit_only"):
        raise ValueError("Cannot import a fit-only xT build as the legacy version.")
    required = {path.stem for path in Path(event_dir).glob("*.csv")}
    present = {path.stem for path in (xt_dir / "matches").glob("*.csv")}
    if required - present:
        raise ValueError(f"Legacy xT is missing {len(required - present)} synced match sidecars.")
    stage = xt_dir / ".staging" / f"legacy_import_{uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    try:
        for name in source_names:
            (xt_dir / name).rename(stage / name)
            moved.append(name)
        metadata["version"] = XT_LEGACY_VERSION
        metadata["status"] = "complete"
        metadata["migrated_from_unversioned"] = True
        for key, value in metadata.get("output_files", {}).items():
            if isinstance(value, str) and value.startswith("data/xT/"):
                metadata["output_files"][key] = value.replace("data/xT/", "data/xT/legacy/", 1)
        source_grid_path = metadata.get("source_grid_path")
        if isinstance(source_grid_path, str) and source_grid_path.startswith("data/xT/"):
            metadata["source_grid_path"] = source_grid_path.replace("data/xT/", "data/xT/legacy/", 1)
        (stage / "fit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        stage.rename(destination)
    except BaseException:
        if stage.exists():
            if "fit_metadata.json" in moved:
                (stage / "fit_metadata.json").write_bytes(original_metadata)
            for name in reversed(moved):
                (stage / name).rename(xt_dir / name)
            stage.rmdir()
        raise
    if not (xt_dir / "latest.json").exists():
        atomic_write_json(xt_dir / "latest.json", {"version": XT_LEGACY_VERSION})
    return destination


if __name__ == "__main__":
    print(f"Moved legacy xT outputs to {migrate_legacy_xt()}")
