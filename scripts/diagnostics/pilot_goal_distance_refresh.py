"""Isolated two-match repair pilot; never publishes or edits production artifacts."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from datatools import config
from datatools.artifact_storage import clone_feature_artifacts
from datatools.goal_distance import annotate_match_goal_distance
from datatools.goal_distance_refresh import preflight_goal_distance, refresh_goal_distance, PAIR_COLUMNS
from datatools.progression import supported
from datatools.xt_refresh import file_signature, fingerprint, refresh_inventory
from project_config import GOAL_DISTANCE_MATCH_DIR, atomic_write_json
from scripts.feature_extension_diagnostics import ExtensionAttempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/features/runs/feature_20261029T201116_375036_6a227f77"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.parent.parent / "goal_distance_refresh_pilots" / datetime.now().strftime("pilot_%Y%m%dT%H%M%S_%f")
    if output.exists():
        raise FileExistsError(output)
    seed, derived, annotations = [output / name for name in ("source_subset", "refreshed", "annotations")]
    seed.mkdir(parents=True)
    annotations.mkdir()
    match_ids = ["DFL-MAT-J03WEL", "DFL-MAT-J03YI3"]
    metadata = json.loads((args.source / "metadata.json").read_text())
    metadata["split_manifest"] = {**metadata.get("split_manifest", {}), "universe_count": len(match_ids)}
    source_files = []
    for directory in args.source.iterdir():
        if not directory.is_dir():
            continue
        # Include empty auxiliary directories as well, preserving recorded variants.
        (seed / directory.name).mkdir()
        for mid in match_ids:
            for suffix in (".pt", ".parquet"):
                source = directory / (mid + suffix)
                if source.is_file():
                    os.link(source, seed / directory.name / source.name)
                    source_files.append(source)
    atomic_write_json(seed / "metadata.json", metadata)
    signatures = {path: file_signature(path) for path in source_files}
    hashes = {path: fingerprint(path) for path in source_files if "graphs" not in path.parent.name}
    production_metadata_hash = fingerprint(args.source / "metadata.json")
    annotation_hashes = {mid: fingerprint(GOAL_DISTANCE_MATCH_DIR / f"{mid}.csv")
                         for mid in match_ids if (GOAL_DISTANCE_MATCH_DIR / f"{mid}.csv").is_file()}
    shutil.copy2(GOAL_DISTANCE_MATCH_DIR / f"{match_ids[1]}.csv", annotations / f"{match_ids[1]}.csv")
    try:
        preflight_goal_distance(seed, metadata, goal_distance_match_dir=annotations)
    except FileNotFoundError as error:
        missing_error = str(error)
    else:
        raise AssertionError("Missing-season annotations were accepted")
    assert not derived.exists()
    # Generate the absent annotation only inside the isolated pilot directory.
    raw = pd.read_csv(Path("data/event_synced") / f"{match_ids[0]}.csv")
    annotated, _ = annotate_match_goal_distance(raw)
    annotated[["action_id", "goal_distance", "scores_goal_distance", "concedes_goal_distance"]].to_csv(
        annotations / f"{match_ids[0]}.csv", index=False)
    preflight = preflight_goal_distance(seed, metadata, goal_distance_match_dir=annotations)
    clone_stats = {}
    clone_feature_artifacts(seed, derived, diagnostics=clone_stats)
    original_load, original_parquet = torch.load, pd.read_parquet
    def guarded_load(path, *args, **kwargs):
        if "graphs" in Path(path).parent.name:
            raise AssertionError(f"Graph payload loaded: {path}")
        return original_load(path, *args, **kwargs)
    def guarded_parquet(path, *args, **kwargs):
        if "tracking" in str(path):
            raise AssertionError(f"Tracking loaded: {path}")
        return original_parquet(path, *args, **kwargs)
    with (patch("torch.load", side_effect=guarded_load),
          patch("pandas.read_parquet", side_effect=guarded_parquet),
          patch("datatools.match.Match._apply_intended_receiver_model", side_effect=AssertionError("receiver inference"))):
        report = refresh_goal_distance(derived, metadata, target_families=("goal_distance", "progression"),
            preflight=preflight, attempt=ExtensionAttempt(derived, "pilot", "derived", {}))
        second = refresh_goal_distance(derived, metadata, target_families=("goal_distance", "progression"),
            preflight=preflight, attempt=ExtensionAttempt(derived, "pilot", "derived", {}))
        assert second["rewritten_files"] == 0
        # Standalone progression verifies freshness and also leaves every artifact untouched.
        standalone = refresh_goal_distance(derived, metadata, target_families=("progression",),
            preflight=preflight, attempt=ExtensionAttempt(derived, "pilot", "derived", {}))
        assert standalone["rewritten_files"] == 0
    inventory, _, _, _ = refresh_inventory(derived, metadata)
    compared_rows, compared_files, repaired_nonzero = 0, 0, 0
    for mid, artifacts in inventory.items():
        for artifact in artifacts:
            old = original_load(seed / artifact.path.relative_to(derived), weights_only=True)
            new = original_load(artifact.path, weights_only=True)
            keep = [i for i in range(old.shape[1]) if i not in PAIR_COLUMNS
                    and i != config.LABEL_INDEX["scores_progression"]]
            torch.testing.assert_close(old[:, keep], new[:, keep], rtol=0, atol=0, equal_nan=True)
            if mid == match_ids[1]:
                torch.testing.assert_close(old[:, PAIR_COLUMNS], new[:, PAIR_COLUMNS], rtol=0, atol=0)
            assert torch.isfinite(new[:, PAIR_COLUMNS]).all()
            progression = new[:, config.LABEL_INDEX["scores_progression"]]
            if supported(artifact.return_type):
                assert torch.isfinite(progression).all()
                assert ((progression >= 0) & (progression <= 1)).all()
            else:
                assert progression.isnan().all()
            if mid == match_ids[0]:
                repaired_nonzero += int(torch.count_nonzero(new[:, PAIR_COLUMNS]))
            compared_files += 1
            compared_rows += len(new)
    assert repaired_nonzero > 0
    assert all(file_signature(path) == signature for path, signature in signatures.items())
    assert all(fingerprint(path) == digest for path, digest in hashes.items())
    assert fingerprint(args.source / "metadata.json") == production_metadata_hash
    assert all(fingerprint(GOAL_DISTANCE_MATCH_DIR / f"{mid}.csv") == digest
               for mid, digest in annotation_hashes.items())
    graph_paths = [path for path in source_files if "graphs" in path.parent.name]
    assert all(os.path.samefile(path, derived / path.relative_to(args.source)) for path in graph_paths)
    result = dict(output=str(output), matches=match_ids, compared_files=compared_files,
                  compared_rows=compared_rows, repaired_nonzero_targets=repaired_nonzero,
                  already_annotated_goal_distance_preserved=True,
                  missing_annotation_rejected=missing_error, source_preserved=True,
                  graphs_still_linked=len(graph_paths), graph_and_tracking_loads=0,
                  first_rewritten_files=report["rewritten_files"], repeated_rewritten_files=second["rewritten_files"],
                  standalone_rewritten_files=standalone["rewritten_files"], artifact_clone=clone_stats)
    atomic_write_json(output / "pilot_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
