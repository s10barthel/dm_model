"""Exercise the real extension planner and label generator in a private two-match run."""
import argparse
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from datatools import graph_feature as gf
from datatools.stop_conditions import label_path, refresh_labels
from datatools.xt_refresh import fingerprint
from project_config import atomic_write_json, load_recorded_split_manifest
from scripts import generate_relevant_features as generator


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    if original.get("status") != "completed" or not (original.get("carry_variant") or {}).get("available"):
        raise ValueError("Pilot requires a completed carry-enabled source")
    if {"next_5", "next_5+"} & set(original["return_types"]):
        raise ValueError("Pilot requires a source without next_5 or next_5+")
    saved_split = load_recorded_split_manifest(original["split_manifest_id"])
    train_mid, test_mid = str(saved_split["train"][0]), str(saved_split["test"][0])
    mids = {train_mid, test_mid}
    output.mkdir(parents=True, exist_ok=False)
    base = output / "source"
    base.mkdir()
    inputs = {source / "metadata.json": fingerprint(source / "metadata.json")}
    # Match files are one level below the root, except small sidecars and migrated labels.
    directories = [p for p in source.iterdir() if p.is_dir()]
    directories.extend(p for name in ("carry_sidecars", "labels", "reception_retirement")
                       if (source / name).is_dir() for p in (source / name).rglob("*") if p.is_dir())
    for directory in dict.fromkeys(directories):
        for mid in mids:
            for suffix in (".pt", ".parquet", ".json"):
                path = directory / (mid + suffix)
                if not path.is_file():
                    continue
                # Test matches have no intent-training/blocked-action auxiliary artifacts.
                if mid == test_mid and ("intent_train" in directory.name or "augmented" in directory.name):
                    continue
                target = base / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                inputs[path] = fingerprint(path)
    metadata = copy.deepcopy(original)
    # Keep the source split identity required by the pinned surface; substitute only
    # the private match slice, without creating or modifying any registered manifest.
    manifest = dict(manifest_id=original["split_manifest_id"], train=[train_mid], test=[test_mid],
                    metadata=dict(train_size=1, test_size=1, universe_count=2))
    metadata.update(run_id="source", split_manifest_id=manifest["manifest_id"],
                    train_count=1, train_split_percent=None, split_manifest=manifest["metadata"])
    atomic_write_json(base / "metadata.json", metadata)
    if not metadata.get("stop_conditions"):
        refresh_labels(base, metadata, {ret.removesuffix("_spstop"): ["base"] for ret in metadata["return_types"]})
        atomic_write_json(base / "metadata.json", metadata)
    source_state = {p: fingerprint(p) for p in base.rglob("*") if p.is_file()}
    commands = []
    def execute(command):
        commands.append(command)
        # Use the actual CLI parser and match workers; only private roots/splits are substituted.
        with patch.object(sys, "argv", command[1:]):
            gf.main()
    def load_metadata(run_id, **_):
        return json.loads((output / run_id / "metadata.json").read_text(encoding="utf-8-sig"))
    with ExitStack() as stack:
        stack.enter_context(patch.object(generator, "get_feature_run_root", lambda run_id: output / run_id))
        stack.enter_context(patch.object(generator, "resolve_feature_run_id", lambda run_id, **_: run_id))
        stack.enter_context(patch.object(generator, "load_feature_run_metadata", load_metadata))
        stack.enter_context(patch.object(generator, "resolve_split_manifest", lambda **_: manifest))
        stack.enter_context(patch.object(gf, "get_feature_run_root", lambda run_id: output / run_id))
        stack.enter_context(patch.object(gf, "load_base_splits", lambda **_: ([train_mid], [test_mid])))
        stack.enter_context(patch.object(generator, "run_command", execute))
        latest = stack.enter_context(patch.object(generator, "write_latest_run"))
        argv = ["generate_relevant_features.py", "--extend-feature-run-id", "source", "--run-id", "extended",
                "--train-count", "1", "--return-type", "next_5", "--return-type", "next_5+",
                "--stop-condition", "set-pieces", "--artifact-copy-mode", "copy"]
        if not metadata.get("next_action_conditions_enabled", True):
            argv.append("--next-action-conditions-off")
        with patch.object(sys, "argv", argv):
            args = generator.parse_args()
        plan = generator.build_extension_plan(args)
        generator.run_extension_generation(args)
        latest.assert_called_once_with("feature", "extended")
    target = output / "extended"
    result = json.loads((target / "metadata.json").read_text())
    assert result["status"] == "completed"
    assert len(commands) == 5 and sum("--use-carries" in command for command in commands) == 2
    assert all(fingerprint(path) == digest for path, digest in inputs.items())
    assert all(fingerprint(path) == digest for path, digest in source_state.items())
    graphs = [path for path in source_state if "graphs" in path.parent.name]
    assert all(fingerprint(target / path.relative_to(base)) == source_state[path] for path in graphs)
    rows = 0
    for mode in metadata["intended_receiver_modes"]:
        for carry in (False, True):
            for mid in mids:
                reference = torch.load(gf.get_action_label_dir(metadata["return_types"][0], mode, root=base,
                    use_carries=carry) / f"{mid}.pt", weights_only=False)
                for ret in ("next_5", "next_5+"):
                    path = gf.get_action_label_dir(ret, mode, root=target, use_carries=carry) / f"{mid}.pt"
                    for condition in ("base", "set-pieces"):
                        labels = torch.load(label_path(path, target, condition), weights_only=False)
                        assert torch.equal(labels[:, 0], reference[:, 0])
                        rows += len(labels)
    report = dict(source=str(source), output=str(output), train_match=train_mid, test_match=test_mid,
                  stages=len(commands), carry_stages=2, checked_rows=rows,
                  source_files_unchanged=len(inputs), private_source_files_unchanged=len(source_state),
                  retained_graph_files_unchanged=len(graphs), production_latest_updates=0,
                  return_types=["next_5", "next_5+"], stop_conditions=["base", "set-pieces"],
                  commands=[step.command for step in plan.command_steps])
    atomic_write_json(output / "pilot_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.source, args.output)
