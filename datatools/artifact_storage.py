"""Storage for feature artifacts shared between derived runs.

Hard links are independent directory entries, but in-place writes affect every
link. All feature writers must replace complete files using these helpers.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

import torch


@contextmanager
def staged_path(destination, *, attempt=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    primary = None
    try:
        yield temporary
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            message = f"Temporary-file cleanup failed: {temporary}: {error}"
            if attempt is not None:
                attempt.secondary("Temporary-file cleanup failed", error, temporary)
                attempt.data["leftover_temporary_paths"].append(str(temporary))
            if primary is None:
                raise
            primary.add_note(message)


def atomic_torch_save(value, destination):
    with staged_path(destination) as temporary:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
        temporary.replace(destination)


def atomic_parquet_save(frame, destination, **kwargs):
    with staged_path(destination) as temporary:
        frame.to_parquet(temporary, **kwargs)
        temporary.replace(destination)


def clone_feature_artifacts(source, destination, *, mode="hardlink", diagnostics=None):
    """Clone without reading graph payloads; copy only independent control files."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if mode not in {"hardlink", "copy"}:
        raise ValueError(f"Unknown artifact copy mode: {mode}")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ.")
    if destination.exists() and any(p.name != "metadata.json" for p in destination.iterdir()):
        raise FileExistsError(f"Derived feature run directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    stats = dict(mode=mode, linked_files=0, linked_bytes=0, copied_files=0, copied_bytes=0)
    if diagnostics is not None:
        diagnostics["artifact_clone"] = stats
    if mode == "hardlink":
        # Probe the actual source and destination directories, not their drive letters:
        # Windows feature directories may be junctions to another volume.
        probe_source = next((p for p in source.rglob("*") if p.is_file()), None)
        with staged_path(destination / "link_probe") as probe:
            probe.unlink()
            try:
                if probe_source is not None:
                    os.link(probe_source, probe)
            except OSError as error:
                raise OSError("Hard-link cloning requires a supporting filesystem on the same volume. "
                              "Use --artifact-copy-mode copy to explicitly allow a full copy.") from error
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative == Path("metadata.json"):
            continue
        if path.is_symlink():
            raise ValueError(f"Feature artifact symlinks are unsupported: {path}")
        output = destination / relative
        if path.is_dir():
            output.mkdir(parents=True, exist_ok=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        # Only data artifacts are shared. Logs, JSON and diagnostics stay independent.
        share = mode == "hardlink" and path.suffix in {".pt", ".parquet"}
        if share:
            try:
                os.link(path, output)
            except OSError as error:
                raise OSError(f"Cannot hard-link {path}; no copy fallback was performed. "
                              "Use a new output run and --artifact-copy-mode copy if desired.") from error
        else:
            shutil.copy2(path, output)
        kind = "linked" if share else "copied"
        stats[f"{kind}_files"] += 1
        stats[f"{kind}_bytes"] += path.stat().st_size
    return stats
