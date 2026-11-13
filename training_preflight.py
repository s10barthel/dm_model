"""Cheap input access and dataset completeness checks for training only.

Keep these operational checks outside the prepared-dataset fingerprint: they
neither transform samples nor change which rows are eligible for training.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class TrainingInputError(RuntimeError):
    """A required training input failed; continuing would omit requested data."""


def preflight_training_files(
    splits: Iterable[tuple[str, Iterable[str], str | Path, str | Path]],
) -> None:
    """Probe selected graph/label files once, without loading their payloads.

    Each split is (name, match_ids, feature_dir, label_dir). Resolve directories
    once and deduplicate paths across splits. The caller chooses the actual
    training/validation IDs and omits validation for a final refit.
    """
    directories = {}
    checked = set()
    for split, match_ids, feature_dir, label_dir in splits:
        for match_id in match_ids:
            for role, directory in (("graph", feature_dir), ("label", label_dir)):
                root = Path(directory)
                path = root / f"{match_id}.pt"
                try:
                    if root not in directories:
                        directories[root] = root.resolve()
                    path = directories[root] / path.name
                    if path in checked:
                        continue
                    with path.open("rb", buffering=0) as stream:
                        if not stream.read(1):
                            raise EOFError("Required input file is empty")
                except (OSError, EOFError) as error:
                    codes = ", ".join(f"{name}={getattr(error, name)}"
                                      for name in ("errno", "winerror")
                                      if getattr(error, name, None) is not None)
                    detail = f"{type(error).__name__}: {error}"
                    if codes:
                        detail += f" ({codes})"
                    raise TrainingInputError(
                        f"Training input preflight failed: split={split!r}, match_id={str(match_id)!r}, "
                        f"role={role!r}, path={path.absolute()}: {detail}"
                    ) from error
                checked.add(path)


def require_complete_dataset(dataset, requested_ids: Iterable[str], split: str) -> None:
    """Reject file-level skips; eligible-row filtering and empty matches are valid."""
    skipped = dataset.skipped_matches
    if skipped:
        match_id, reason = next(iter(skipped.items()))
        raise TrainingInputError(
            f"Training dataset loading failed: split={split!r}, match_id={str(match_id)!r}: {reason}"
        )
    loaded = {str(match_id) for match_id in dataset.loaded_match_ids}
    for match_id in requested_ids:
        if str(match_id) not in loaded:
            raise TrainingInputError(
                f"Training dataset incomplete: split={split!r}, match_id={str(match_id)!r}: "
                "requested match is absent from loaded_match_ids"
            )
