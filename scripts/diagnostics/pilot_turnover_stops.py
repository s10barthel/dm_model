"""Isolated one-match validation; never registers runs or changes latest pointers."""
import argparse
import json
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datatools.stop_conditions import refresh_labels
from datatools.xt_refresh import fingerprint, refresh_inventory
from project_config import atomic_write_json


def compare_generation(root, metadata, match_id, return_types):
    """Compare stored targets with normal label generation using real tracking."""
    import pandas as pd
    import torch
    from datatools import config
    from datatools.match import Match
    from datatools.epv import merge_epv_annotations
    from datatools.goal_distance import merge_goal_distance_annotations
    from datatools.reception_policy import policy_version
    from datatools.return_windows import TurnoverWindows
    from datatools.xt_refresh import read_annotated_events
    from datatools.xt_targets import surface_from_metadata, target_settings
    from project_config import get_action_label_dir, get_resolved_action_dir, parse_return_type
    from xt_versions import resolve_xt_version
    root = Path(root)
    selected = resolve_xt_version(metadata["xt_version"])
    pinned = root / "pass_endpoints" / f"{match_id}.parquet"
    events = read_annotated_events(pinned if pinned.exists() else ROOT / "data/event_synced" / f"{match_id}.csv",
                                   selected.match_dir / f"{match_id}.csv")
    from datatools.xt_acquisitions import acquisition_context, CONTEXT_KEY
    events, _, _ = acquisition_context(events, match_id, root)
    context = events.attrs.get(CONTEXT_KEY)
    events = merge_goal_distance_annotations(events, match_id)
    events = merge_epv_annotations(events, match_id)
    if context is not None:
        events.attrs[CONTEXT_KEY] = context
    tracking = pd.read_parquet(ROOT / "data/tracking_processed" / f"{match_id}.parquet")
    names = ["scores", "concedes", "scores_xg", "concedes_xg", "scores_xt", "concedes_xt",
             "scores_goal_distance", "concedes_goal_distance", "scores_epv", "concedes_epv", "scores_progression"]
    columns = [config.LABEL_INDEX[name] for name in names]
    mode = metadata["intended_receiver_modes"][0]
    cases, maximum_difference = [], 0.
    for carry in ([False, True] if (metadata.get("carry_variant") or {}).get("available") else [False]):
        actions = pd.read_parquet(get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{match_id}.parquet")
        for ret in return_types:
            for condition in metadata["stop_conditions"]["available"][ret]:
                match = Match.__new__(Match)
                match.events, match.tracking = events.copy(), tracking
                match.action_type, match.fps, match.include_goals = "all", 25, False
                match.reception_policy = policy_version(metadata)
                match.stop_condition, match.xt_targets = condition, target_settings(metadata)
                match.xt_surface, match.xt_surface_metadata = surface_from_metadata(selected.metadata), selected.metadata
                generated = match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                   intended_receiver_mode=mode, return_type=ret)
                actual = torch.load(root / "labels" / condition / get_action_label_dir(ret, mode, root=root,
                                    use_carries=carry).name / f"{match_id}.pt", map_location="cpu", weights_only=True)
                references = pd.DataFrame(generated[:, columns].numpy(), index=generated[:, 0].numpy().astype(int))
                expected = torch.as_tensor(references.loc[actual[:, 0].numpy().astype(int)].to_numpy(), dtype=actual.dtype)
                torch.testing.assert_close(actual[:, columns], expected, rtol=0, atol=0, equal_nan=True)
                finite = torch.isfinite(expected)
                if finite.any():
                    maximum_difference = max(maximum_difference, float((actual[:, columns][finite] - expected[finite]).abs().max()))
                kind, n, skip = parse_return_type(ret)
                windows = TurnoverWindows(events, condition)
                acquisitions, fallbacks = 0, 0
                anchors = actions.get("return_event_index", pd.Series(actions.index, index=actions.index)).fillna(pd.Series(actions.index, index=actions.index))
                for anchor, team in zip(anchors, actions.object_id.astype(str).str[:4]):
                    position = windows.events.index.get_indexer([anchor])[0]
                    _, acquisition = windows.select(position, team, int(n), kind == "next_plus", skip)
                    if acquisition is not None:
                        acquisitions += 1
                        fallbacks += int(acquisition in windows.eligible)
                xt = actual[:, [config.LABEL_INDEX["scores_xt"], config.LABEL_INDEX["concedes_xt"]]]
                cases.append(dict(return_type=ret, stop_condition=condition, carries=carry, rows=len(actual),
                    target_mean=xt.double().mean(dim=0).tolist(), zero_fraction=xt.eq(0).double().mean(dim=0).tolist(),
                    turnover_windows=acquisitions, acquisition_fallbacks=fallbacks))
    return dict(compared_rows=sum(case["rows"] for case in cases), maximum_target_difference=maximum_difference, cases=cases)


def run(source, output, match_id, copy_files=False):
    source, output = Path(source), Path(output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("status") != "completed" or metadata.get("label_layout_version"):
        raise ValueError("Pilot requires a completed legacy run")
    output.mkdir(parents=True, exist_ok=False)
    originals = {}
    for directory in source.iterdir():
        if not directory.is_dir():
            continue
        for suffix in (".pt", ".parquet"):
            path = directory / (match_id + suffix)
            if path.is_file():
                target = output / directory.name / path.name
                target.parent.mkdir(exist_ok=True)
                (shutil.copy2 if copy_files else os.link)(path, target)
                originals[path] = fingerprint(path)
    control = source / "carry_sidecars" / "control_events" / f"{match_id}.parquet"
    if control.is_file():
        target = output / control.relative_to(source)
        target.parent.mkdir(parents=True)
        (shutil.copy2 if copy_files else os.link)(control, target)
        originals[control] = fingerprint(control)
    metadata.update(run_id="isolated_turnover_stop_pilot", split_manifest={"universe_count": 1})
    atomic_write_json(output / "metadata.json", metadata)
    requests = {ret.removesuffix("_spstop"): ["base"] for ret in metadata["return_types"]}
    report = {"base": refresh_labels(output, metadata, requests)}
    atomic_write_json(output / "metadata.json", metadata)
    selected = next(ret for ret in metadata["return_types"] if ret.startswith("next_") and "skip" not in ret)
    report["set_pieces"] = refresh_labels(output, metadata, {selected: ["set-pieces"]})
    metadata["xt_targets"] = dict(version=2, representation="gain", clip_negative_gains=True)
    report["xt"] = refresh_labels(output, metadata, metadata["stop_conditions"]["available"], xt_only=True)
    atomic_write_json(output / "metadata.json", metadata)
    report["repeat"] = refresh_labels(output, metadata, metadata["stop_conditions"]["available"], xt_only=True)
    assert report["repeat"]["rewritten_files"] == 0
    # Exercise the existing add-return artifact shape without rebuilding graph payloads.
    inventory, _, _, _ = refresh_inventory(output, metadata)
    extended = selected + "+"
    for artifact in inventory[match_id]:
        if artifact.return_type != selected or selected not in artifact.path.parent.name:
            continue
        directory = artifact.path.parent.with_name(artifact.path.parent.name.replace(selected, extended))
        directory.mkdir(exist_ok=True)
        os.link(artifact.path, directory / artifact.path.name)
    metadata["return_types"].append(extended)
    report["extended"] = refresh_labels(output, metadata, {extended: ["base", "set-pieces"]})
    atomic_write_json(output / "metadata.json", metadata)
    refresh_inventory(output, metadata)
    report["generation_comparison"] = compare_generation(output, metadata, match_id, [selected, extended])
    assert all(fingerprint(path) == digest for path, digest in originals.items())
    report["source"] = str(source)
    report["source_files_unchanged"] = len(originals)
    report["match_id"] = match_id
    report["output"] = str(output)
    atomic_write_json(output / "pilot_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match-id")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    match_id = args.match_id or next((Path(args.source) / "action_graphs").glob("*.pt")).stem
    run(args.source, args.output, match_id, args.copy)
