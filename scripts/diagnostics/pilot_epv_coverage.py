"""Migrate an incomplete real match privately and verify EPV rejection before training."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from datatools import config
from datatools.epv_coverage import ATTRIBUTE, DIRECTORY, TARGETS, validate_labels
from datatools.stop_conditions import refresh_labels
from datatools.xt_refresh import fingerprint
from project_config import atomic_write_json, get_action_label_dir


def run(source, output, match_id):
    source, output = Path(source), Path(output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("status") != "completed" or metadata.get("label_layout_version"):
        raise ValueError("Pilot requires a completed legacy source")
    output.mkdir(parents=True, exist_ok=False)
    originals = {source / "metadata.json": fingerprint(source / "metadata.json")}
    for folder in source.iterdir():
        if not folder.is_dir():
            continue
        for suffix in (".pt", ".parquet", ".json"):
            path = folder / (match_id + suffix)
            if path.is_file():
                target = output / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                originals[path] = fingerprint(path)
    control = source / "carry_sidecars" / "control_events" / f"{match_id}.parquet"
    if control.is_file():
        target = output / control.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(control, target)
        originals[control] = fingerprint(control)
    metadata.update(run_id="isolated_epv_pilot", split_manifest=dict(universe_count=1))
    atomic_write_json(output / "metadata.json", metadata)
    requests = {ret.removesuffix("_spstop"): ["base", "set-pieces"] for ret in metadata["return_types"]}
    report = refresh_labels(output, metadata, requests)
    atomic_write_json(output / "metadata.json", metadata)
    coverage = json.loads((output / DIRECTORY / f"{match_id}.json").read_text())
    assert not coverage["complete"]
    rows, files = 0, 0
    columns = [config.LABEL_INDEX[name] for name in TARGETS]
    for path in (output / "labels").rglob("*.pt"):
        labels = torch.load(path, weights_only=True)
        assert labels[:, columns].isnan().all()
        rows += len(labels)
        files += 1
    selected = next(ret for ret in metadata["return_types"] if ret.startswith("next_"))
    try:
        validate_labels(output, metadata, [match_id], get_action_label_dir(selected, root=output), selected)
    except ValueError as exc:
        rejection = str(exc)
        assert match_id in rejection and "nonfinite_required_values" in rejection
    else:
        raise AssertionError("Incomplete EPV was accepted for training")
    # Recomputing other families must leave these already refreshed values identical.
    other = refresh_labels(output, metadata, requests,
        target_families=["goal", "xg", "xt", "goal_distance", "progression"])
    assert other["rewritten_files"] == 0
    repeated = refresh_labels(output, metadata, requests)
    assert repeated["rewritten_files"] == 0
    assert all(fingerprint(path) == digest for path, digest in originals.items())
    graphs = [path for path in originals if "graphs" in path.parent.name]
    assert all(fingerprint(output / path.relative_to(source)) == originals[path] for path in graphs)
    summary = dict(source=str(source), match_id=match_id, output=str(output),
                   coverage={key: value for key, value in coverage.items() if not key.startswith("verified_")},
                   migrated_label_files=files, checked_label_rows=rows,
                   source_files_unchanged=len(originals), retained_graph_files_unchanged=len(graphs),
                   repeat_replacements=0, other_target_replacements=0, training_rejection=rejection,
                   production_latest_updates=0)
    atomic_write_json(output / "pilot_report.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-id", required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.match_id)
