"""Validate log progression against an isolated subset of an existing feature run."""
from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.match import Match
from datatools.xt import annotate_match_xt
from datatools.xt_targets import LogProgressionSurface, target_settings, unavailable_xt_returns
from datatools.xt_refresh import refresh_xt, refresh_inventory, fingerprint, file_signature, TARGET_COLUMNS
from project_config import (atomic_write_json, get_action_label_dir, get_intent_train_label_dir,
                            get_augmented_label_dir, get_resolved_action_dir, parse_return_type)
from scripts.generate_xt import save_model_outputs


def run_pilot(source, output, match_ids, *, k=2., min_goal_distance=1.):
    source, output = Path(source), Path(output)
    saved = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    if saved.get("status") != "completed":
        raise ValueError("Pilot source must be a completed feature run.")
    returns = list(saved["return_types"])
    if not returns or parse_return_type(returns[0])[0] not in {"next", "in"}:
        raise ValueError("Pilot requires a supported primary return in the source run.")
    modes = saved["intended_receiver_modes"]
    carries = bool((saved.get("carry_variant") or {}).get("available"))
    output.mkdir(parents=True, exist_ok=False)
    derived, version = output / "features", output / "surface"
    derived.mkdir()
    model = LogProgressionSurface(k, min_goal_distance)
    settings = target_settings({"xt_targets": dict(representation="gain", clip_negative_gains=True)})
    surface_metadata = dict(surface_type="log_progression", surface_definition_version=1,
                            k=k, min_goal_distance=min_goal_distance, shot_xT_policy="surface_only")
    save_model_outputs(model, version)
    atomic_write_json(version / "fit_metadata.json", surface_metadata)
    (version / "matches").mkdir()
    unavailable = unavailable_xt_returns(surface_metadata, returns)
    metadata = {**saved, "run_id": "isolated_log_progression_pilot", "return_types": returns,
                "xt_targets": settings, "xt_unavailable_return_types": unavailable,
                "split_manifest": {"universe_count": len(match_ids)}}
    directories = {"action_graphs", "post_action_graphs", "action_graphs_intent_train",
                   "action_graphs_success_intent", "success_intent_labels"}
    if carries:
        directories.update({"action_graphs_carries", "post_action_graphs_carries"})
    for mode in modes:
        augmented = get_augmented_label_dir(mode, root=source)
        directories.update({augmented.name, augmented.name.replace("labels", "graphs")})
        for carry in ([False, True] if carries else [False]):
            directories.add(get_resolved_action_dir(mode, root=source, use_carries=carry).name)
            for ret in returns:
                directories.add(get_action_label_dir(ret, mode, root=source, use_carries=carry).name)
        directories.update(get_intent_train_label_dir(ret, mode, root=source).name for ret in returns)
    originals = {}
    for name in sorted(directories):
        directory = source / name
        if not directory.is_dir():
            continue
        (derived / name).mkdir()
        for mid in match_ids:
            for suffix in (".pt", ".parquet"):
                path = directory / (mid + suffix)
                if path.exists():
                    os.link(path, derived / name / path.name)
                    originals[path] = (file_signature(path), None if "graphs" in name else fingerprint(path))
    atomic_write_json(derived / "metadata.json", metadata)
    raw_events = {}
    for mid in match_ids:
        events = pd.read_csv(ROOT / "data/event_synced" / f"{mid}.csv")
        annotated, _ = annotate_match_xt(events, model)
        annotated[["action_id", "xT", "scores_xT", "concedes_xT"]].to_csv(version / "matches" / f"{mid}.csv", index=False)
        raw_events[mid] = annotated
    report = refresh_xt(derived, metadata, xt_match_dir=version / "matches", surface_metadata=surface_metadata,
                        event_dir=ROOT / "data/event_synced", xt_targets=settings)
    inventory, _, _, _ = refresh_inventory(derived, metadata)
    rows, maximum_difference, statistics = 0, 0., []
    for mid, artifacts in inventory.items():
        tracking = pd.read_parquet(ROOT / "data/tracking_processed" / f"{mid}.parquet")
        references = {}
        for artifact in artifacts:
            key = artifact.mode, artifact.carries, artifact.return_type
            if key not in references:
                actions = pd.read_parquet(get_resolved_action_dir(artifact.mode, root=derived, use_carries=artifact.carries) / f"{mid}.parquet")
                match = Match.__new__(Match)
                match.events, match.tracking = raw_events[mid].copy(), tracking
                match.action_type, match.fps, match.include_goals = "all", 25, False
                match.xt_targets, match.xt_surface, match.xt_surface_metadata = settings, model, surface_metadata
                labels = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                 intended_receiver_mode=artifact.mode, return_type=artifact.return_type)
                references[key] = pd.DataFrame(labels[:, TARGET_COLUMNS].numpy(), index=labels[:, 0].numpy().astype(int))
            actual = torch.load(artifact.path, map_location="cpu", weights_only=True)
            expected = torch.as_tensor(references[key].loc[actual[:, 0].numpy().astype(int)].to_numpy(), dtype=actual.dtype)
            torch.testing.assert_close(actual[:, TARGET_COLUMNS], expected, rtol=0, atol=0, equal_nan=True)
            finite = torch.isfinite(expected)
            if finite.any():
                maximum_difference = max(maximum_difference, float(
                    (actual[:, TARGET_COLUMNS][finite] - expected[finite]).abs().max()))
            before = torch.load(source / artifact.path.relative_to(derived), map_location="cpu", weights_only=True)
            keep = [i for i in range(actual.shape[1]) if i not in TARGET_COLUMNS]
            torch.testing.assert_close(actual[:, keep], before[:, keep], rtol=0, atol=0, equal_nan=True)
            rows += len(actual)
            if artifact.path.parent == get_action_label_dir(returns[0], modes[0], root=derived):
                x = actual[:, config.LABEL_INDEX["start_x"]].numpy()
                for region, mask in (("own_half", x < 52.5), ("middle", (x >= 52.5) & (x < 70)), ("attacking_third", x >= 70)):
                    gains = actual[mask, TARGET_COLUMNS[0]].numpy()
                    statistics.append(dict(match=mid, region=region, rows=len(gains),
                        mean=float(np.mean(gains)) if len(gains) else None,
                        zero_fraction=float(np.mean(gains == 0)) if len(gains) else None))
        print(f"Validated {mid}: {len(artifacts)} label files", flush=True)
    for path, (signature, digest) in originals.items():
        assert file_signature(path) == signature, path
        if digest:
            assert fingerprint(path) == digest, path
        if "graphs" in path.parent.name:
            assert os.path.samefile(path, derived / path.relative_to(source))
    repeated = refresh_xt(derived, metadata, xt_match_dir=version / "matches", surface_metadata=surface_metadata,
                          event_dir=ROOT / "data/event_synced", xt_targets=settings)
    assert repeated["rewritten_files"] == 0
    result = dict(source=str(source), matches=match_ids, return_types=returns,
        xt_unavailable_return_types=unavailable, k=k,
        min_goal_distance=min_goal_distance, compared_rows=rows, maximum_target_difference=maximum_difference,
        source_artifacts_unchanged=True, graph_payloads_loaded_by_refresh=report["graph_files_loaded"],
        repeated_refresh_rewritten_files=0, region_statistics=statistics,
        refresh={key: value for key, value in report.items() if key not in {"artifacts", "input_sha256"}})
    atomic_write_json(output / "pilot_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--match-id", action="append", required=True)
    parser.add_argument("--k", type=float, default=2.)
    parser.add_argument("--min-goal-distance", type=float, default=1.)
    args = parser.parse_args()
    output = args.output or ROOT / "data/features/log_progression_pilots" / datetime.now(UTC).strftime("pilot_%Y%m%dT%H%M%S")
    result = run_pilot(args.source, output, args.match_id, k=args.k, min_goal_distance=args.min_goal_distance)
    print(f"Validated {result['compared_rows']} rows. Report: {output / 'pilot_report.json'}")


if __name__ == "__main__":
    main()
