"""Saved split provenance for wrapper tests using the synthetic 'feature_run'."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import project_config


def install_wrapper_split(test_case):
    directory = tempfile.TemporaryDirectory()
    test_case.addCleanup(directory.cleanup)
    root = Path(directory.name)
    train = [f"m{i:03d}" for i in range(5)]
    test = [f"m{i:03d}" for i in range(5, 10)]
    fingerprint = project_config._match_universe_fingerprint(train + test)
    manifest_id = f"train_50pct_{fingerprint[:12]}"
    manifest = {
        "manifest_id": manifest_id, "train_split_percent": 50, "train": train, "test": test,
        "metadata": {"train_size": 5, "test_size": 5, "universe_count": 10,
                     "universe_fingerprint": fingerprint, "rounding": "floor", "ordering": "match_id"},
    }
    (root / f"{manifest_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    for context in (
        patch.object(project_config, "SPLIT_MANIFESTS_DIR", root),
        patch.dict(project_config.LEGACY_FEATURE_SPLIT_MANIFESTS, {"feature_run": manifest_id}),
    ):
        context.start()
        test_case.addCleanup(context.stop)
    test_case.split_manifest = project_config.load_recorded_split_manifest(manifest_id)
