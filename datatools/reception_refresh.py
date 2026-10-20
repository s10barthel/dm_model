"""Derived-only, resumable refresh of reception-dependent artifacts."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pandas as pd
import torch

from datatools import config
from datatools.artifact_storage import atomic_parquet_save, atomic_torch_save, clone_feature_artifacts
from datatools.carry_policy import CarryPolicy
from datatools.match import Match
from datatools.reception_policy import KEY, RETIRED, apply_policy, event_key, policy_version, retirement_records
from project_config import (DATA_ROOT, atomic_write_json, generate_run_id, get_feature_run_root,
                            get_resolved_action_path, write_latest_run, write_run_metadata)


def artifact_mode(name):
    return "model" if name.endswith("_model") else "angle_only" if name.endswith("_angle_only") else "original"


def patch_labels(labels, retired_ids, failed_ids, *, original=False):
    if labels.ndim != 2 or labels.shape[1] <= config.LABEL_INDEX["is_real"]:
        raise ValueError("Malformed reception refresh labels")
    result = labels.clone()
    ids = labels[:, 0].tolist()
    if any(not math.isfinite(i) or not float(i).is_integer() for i in ids):
        raise ValueError("Reception refresh requires finite integer action identities")
    mask = torch.tensor([int(i) in retired_ids for i in ids], device=labels.device)
    failed = torch.tensor([int(i) in failed_ids for i in ids], device=labels.device)
    for column in ("pass_max_ball_z", "pass_high"):
        if config.LABEL_INDEX[column] < result.shape[1]:
            result[mask, config.LABEL_INDEX[column]] = float("nan")
    for column, sentinel in (("intent_x", -config.FIELD_SIZE[0]), ("intent_y", -config.FIELD_SIZE[1])):
        result[mask, config.LABEL_INDEX[column]] = sentinel
    if original:
        result[failed, config.LABEL_INDEX["intent_index"]] = -1
    return result, mask, failed


def refresh_match(task):
    source, output, metadata, match_id, replace, policy_values, thread_limit = task
    source, output = Path(source), Path(output)
    from datatools.graph_feature import configure_worker_thread_limit
    from scripts import replace_carries as rc
    configure_worker_thread_limit(thread_limit)
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv")
    primary = pd.read_parquet(get_resolved_action_path(match_id,
        intended_receiver_mode=metadata["intended_receiver_modes"][0], root=source))
    if not primary.index.is_unique:
        raise ValueError("Duplicate canonical action identities")
    records = retirement_records(events)
    # Persisted provenance also covers source events no longer carrying provenance.
    existing = source / "reception_retirement" / f"{match_id}.json"
    if existing.exists():
        records += json.loads(existing.read_text())["receptions"]
    records += retirement_records(primary)
    records = list({r["event_id"]: r for r in records}.values())
    keys = {r["event_id"] for r in records}
    retired_ids = {int(i) for i, r in primary.iterrows() if event_key(r.original_event_id) in keys}
    failed_ids = {int(i) for i, r in primary.iterrows() if int(i) in retired_ids and not bool(r.success)}
    filtered = apply_policy(primary, RETIRED, records=records)
    post = Match.label_post_actions(SimpleNamespace(events=events, reception_policy=RETIRED), filtered)
    invalid_post = {int(i) for i in retired_ids if pd.isna(post.at[i, "end_frame_id"])}
    artifacts, counts = {}, Counter(receptions=len(records))
    for folder in sorted(p for p in source.iterdir() if p.is_dir() and not p.name.endswith("_carries")):
        name = folder.name
        if name.startswith("resolved_actions"):
            path = folder / f"{match_id}.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
                artifacts[output / path.relative_to(source)] = apply_policy(frame, RETIRED,
                    mode=artifact_mode(name), records=records)
        if "labels" not in name:
            continue
        path = folder / f"{match_id}.pt"
        if not path.exists():
            continue
        labels = torch.load(path, map_location="cpu", weights_only=False)
        original = artifact_mode(name) == "original" and name != "success_intent_labels"
        # Temporal variants repeat the targets of their own receiver mode.
        changed, mask, failed = patch_labels(labels, retired_ids, failed_ids, original=original)
        artifacts[output / path.relative_to(source)] = changed
        counts["height_label_rows"] += int(mask.sum())
        if original:
            counts["original_target_rows"] += int(failed.sum())
        if name.startswith("augmented_labels") and original:
            gp = source / name.replace("labels", "graphs", 1) / path.name
            if gp.exists():
                graphs = torch.load(gp, map_location="cpu", weights_only=False)
                if len(graphs) != len(labels):
                    raise ValueError(f"Augmented graph alignment mismatch: {gp}")
                dependent = failed & labels[:, config.LABEL_INDEX["is_real"]].eq(0)
                for pos in dependent.nonzero().flatten().tolist():
                    graphs[pos] = None
                artifacts[output / gp.relative_to(source)] = graphs
                counts["synthetic_states"] += int(dependent.sum())
    for folder in source.glob("post_action_graphs*"):
        gp = folder / f"{match_id}.pt"
        if folder.name.endswith("_carries") or not gp.exists():
            continue
        graphs = torch.load(gp, map_location="cpu", weights_only=False)
        from project_config import get_action_label_dir
        labels = torch.load(get_action_label_dir(metadata["return_types"][0],
            intended_receiver_mode=metadata["intended_receiver_modes"][0], root=source) / gp.name,
            map_location="cpu", weights_only=False)
        indices = labels[:, 0].long().tolist()
        if len(graphs) != len(indices) or not set(indices).issubset(primary.index):
            raise ValueError(f"Post-action graph alignment mismatch: {gp}")
        for pos, index in enumerate(indices):
            if int(index) in invalid_post:
                graphs[pos] = None
        artifacts[output / gp.relative_to(source)] = graphs
        counts["post_action_states"] += len(invalid_post)
    for path, value in artifacts.items():
        if path.suffix == ".pt":
            atomic_torch_save(value, path)
        else:
            atomic_parquet_save(value, path)
    record_path = output / "reception_retirement" / f"{match_id}.json"
    from project_config import RAW_SEASON_ROOTS
    layouts = {"22_23": "starting_players", "23_24": "match_information", "24_25": "match_information/starting_players"}
    season = next((s for s, root in RAW_SEASON_ROOTS.items() if any(
        (Path(root) / layouts[s] / (match_id + ext)).exists() for ext in ("", ".xml"))), "unknown")
    report = dict(policy=RETIRED, match_id=match_id, season=season, receptions=records, counts=dict(counts),
                  source_events_sha256=rc.digest(DATA_ROOT / "event_synced" / f"{match_id}.csv"),
                  artifacts={str(p.relative_to(output)): p.stat().st_size for p in artifacts})
    atomic_write_json(record_path, report)
    report["artifacts"][str(record_path.relative_to(output))] = record_path.stat().st_size
    if replace:
        carry_metadata = dict(metadata, **{KEY: RETIRED}, _canonical_root=str(output))
        carry_report = rc.rebuild_match((str(source), str(output), carry_metadata, match_id, policy_values, thread_limit))
        report["carry_segments"] = carry_report["segments"]
        report["artifacts"].update(carry_report["artifacts"])
    # Completion is separate from the policy manifest needed by runtime loading.
    atomic_write_json(output / "reception_retirement" / "completed" / f"{match_id}.json", report)
    return report


def complete(output, match_id):
    path = output / "reception_retirement" / "completed" / f"{match_id}.json"
    try:
        report = json.loads(path.read_text())
        if "source_events_sha256" in report:
            from scripts.replace_carries import digest
            if digest(DATA_ROOT / "event_synced" / f"{match_id}.csv") != report["source_events_sha256"]:
                return False
        return report["policy"] == RETIRED and all((output / p).is_file() and
            (output / p).stat().st_size == size for p, size in report["artifacts"].items())
    except (OSError, ValueError, KeyError):
        return False


def resolve_source(value):
    from scripts import replace_carries as rc
    roots = ([get_feature_run_root(value)] if value != "latest" else
             sorted(get_feature_run_root("placeholder").parent.glob("feature_*"), reverse=True))
    for root in roots:
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
            if metadata.get("status") != "completed":
                continue
            rc.match_ids(root, metadata)
            return root, metadata
        except (OSError, ValueError, KeyError):
            if value != "latest":
                raise
    raise ValueError("No compatible completed feature run for reception retirement")


def run_retirement(args):
    from scripts import replace_carries as rc
    source, original = resolve_source(args.extend_feature_run_id)
    replace = bool(args.replace_carries)
    if policy_version(original) == RETIRED and not replace:
        print(f"Elastic receptions already retired: {source.name}")
        return
    if (original.get("carry_variant") or {}).get("available") and not replace:
        raise ValueError("Retiring receptions in a carry-enabled run requires --replace-carries")
    ids = rc.match_ids(source, original)
    output = get_feature_run_root(args.run_id or generate_run_id("feature"))
    defaults = {**asdict(CarryPolicy()), **(original.get("carry_variant") or {}).get("validation_parameters", {})}
    explicit = getattr(args, "explicit_carry_parameters", None)
    values = {k: getattr(args, "carry_" + k, v) if explicit is None or k in explicit else v
              for k, v in defaults.items()}
    CarryPolicy(**values)
    settings = dict(source_sha256=rc.digest(source / "metadata.json"), policy=RETIRED,
                    replace_carries=replace, validation_parameters=values, matches=ids,
                    artifact_copy_mode=args.artifact_copy_mode)
    workers = int(args.num_workers)
    if workers < 1:
        raise ValueError("Reception refresh requires positive --num-workers")
    metadata = dict(original, run_id=output.name, base_feature_run_id=source.name,
                    status="in_progress", reception_refresh=settings,
                    created_at=datetime.now().isoformat(timespec="seconds"),
                    command=subprocess.list2cmdline(sys.argv), **{KEY: RETIRED})
    cloned = False
    if output.exists():
        old = json.loads((output / "metadata.json").read_text())
        if old.get("status") not in {"failed", "in_progress"} or old.get("reception_refresh") != settings:
            raise ValueError("Resume requires identical source, output, operation and policy")
        cloned = old.get("reception_clone_complete", False)
        metadata["created_at"] = old.get("created_at", metadata["created_at"])
    metadata["reception_clone_complete"] = cloned
    if replace:
        from datatools.carry_policy import DEFINITION
        metadata["carry_variant"] = dict(definition=DEFINITION, validation_parameters=values,
            available=False, sidecar_root="carry_sidecars", artifact_suffix="_carries")
    print(f"Reception retirement source: {source.name}\nDerived run: {output.name}", flush=True)
    write_run_metadata(output, metadata)
    try:
        if not cloned:
            clone_feature_artifacts(source, output, mode=args.artifact_copy_mode, resume=True,
                exclude=lambda p: (replace and rc.excluded_artifact(p)) or "reception_retirement" in p.parts)
            metadata["reception_clone_complete"] = True
            write_run_metadata(output, metadata)
        tasks = [(str(source), str(output), metadata, m, replace, values, args.worker_thread_limit)
                 for m in ids if not complete(output, m)]
        if workers == 1:
            for task in tasks:
                report = refresh_match(task)
                print(f"Retired receptions: {report['match_id']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for report in pool.map(refresh_match, tasks):
                    print(f"Retired receptions: {report['match_id']}", flush=True)
        if not all(complete(output, m) for m in ids):
            raise ValueError("Incomplete reception refresh")
        counts = Counter()
        reports = [json.loads((output / "reception_retirement/completed" / f"{m}.json").read_text()) for m in ids]
        for report in reports:
            counts.update(report["counts"])
        metadata["reception_retirement_counts"] = dict(counts)
        by_season = {}
        for report in reports:
            group = by_season.setdefault(report.get("season", "unknown"), Counter())
            group.update(report["counts"])
            group["carry_segments"] += report.get("carry_segments", 0)
        metadata["reception_retirement_by_season"] = {s: dict(v) for s, v in by_season.items()}
        if replace:
            metadata["carry_variant"].update(available=True, generated_at=datetime.now().isoformat(timespec="seconds"))
            metadata["reception_retirement_counts"]["carry_segments"] = sum(r["carry_segments"] for r in reports)
        metadata["status"] = "completed"
        write_run_metadata(output, metadata)
        write_latest_run("feature", output.name)
    except BaseException as error:
        metadata.update(status="failed", reception_refresh_error=str(error))
        write_run_metadata(output, metadata)
        print(f"Resume with --retire-elastic --extend-feature-run-id {source.name} --run-id {output.name}"
              + (" --replace-carries" if replace else ""), file=sys.stderr)
        raise
