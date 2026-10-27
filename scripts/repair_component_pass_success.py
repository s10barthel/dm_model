"""Rebuild missing Sportec v5 pass-success exports from saved component predictions.

The selected auxiliaries must be the exported, unmasked checkpoints. Physics is
read from its original cache; no models are fitted or evaluated. All matches are
staged and checked before publishing, with the original metadata retained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from datatools.config import FIELD_SIZE
from datatools.graph_feature import calculate_offside_flags
import pc_xpass_match_cache as cache
from pc_xpass_versions import atomic_json
from physical_pass_model import (
    _physical_xpass_lower_bound,
    blend_physical_xpass_predictions,
    summarize_physical_xpass_cache_usage,
)

KEYS = ["period_id", "state_frame_id", "possessor_id"]
PLAYER = re.compile(r"^(home|away)_\d+$")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def signature(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def rebuild_v5_table(model, height, intent, physical, offside, distances, *, metric, eps,
                     floor, intent_threshold, discount):
    """Align physical keys, then use the same float32 sidecar and NumPy blend as inference."""
    players = [column for column in model if PLAYER.fullmatch(column)]
    identifiers = [column for column in model if column not in players]
    if not players or model.state_id.duplicated().any() or model.duplicated(KEYS).any():
        raise ValueError("Missing players or duplicate component state identity")
    for role, auxiliary in (("pass_height", height), ("pass_intent", intent)):
        if set(auxiliary.columns) != set(model.columns) or not auxiliary[identifiers].equals(model[identifiers]):
            raise ValueError(f"{role} identifiers or candidate columns differ from model_pass_success")
    if physical.duplicated(KEYS).any():
        raise ValueError("Duplicate physical cache identity")
    requested = pd.MultiIndex.from_frame(model[KEYS])
    physical = physical.set_index(KEYS)
    if not requested.isin(physical.index).all():
        raise ValueError("Physical cache does not cover every component state")
    physical = physical.reindex(requested)
    values = model[players].to_numpy(dtype=float)
    candidate = np.isfinite(values) & (np.array(players)[None, :] != model.possessor_id.to_numpy()[:, None])
    offside = np.asarray(offside, dtype=bool) & candidate
    distances = np.asarray(distances, dtype=float)
    if offside.shape != values.shape or distances.shape != values.shape:
        raise ValueError("Tracking masks do not align with component candidates")
    if (values[offside] != 0).any():
        raise ValueError("Tracking offside flags disagree with saved model zeros")
    eligible = candidate & ~offside
    metric_columns = [f"{player}__{metric}" for player in players]
    # attach_physical_xpass_to_graph first assigns to a float32 tensor, then clips.
    xpass = physical[metric_columns].to_numpy(dtype=np.float32)
    xpass = np.clip(xpass, np.float32(_physical_xpass_lower_bound(eps, floor)),
                    np.float32(1.0 - eps)).astype(float)
    heights = height[players].to_numpy(dtype=float)
    intents = intent[players].to_numpy(dtype=float)
    for role, array in (("xpass", xpass), ("pass_height", heights), ("pass_intent", intents),
                        ("pass_distance", distances)):
        if not np.isfinite(array[eligible]).all():
            raise ValueError(f"Nonfinite {role} for an eligible pass")
    rebuilt = values.copy()
    rebuilt[eligible] = blend_physical_xpass_predictions(
        pass_success_model=values[eligible], xpass=xpass[eligible],
        pass_distance=distances[eligible], pass_height=heights[eligible],
        pass_intent=intents[eligible], weight_version="v5",
        v5_intent_threshold=intent_threshold, v5_discount=discount,
    )
    rebuilt[offside] = 0.0
    if not np.array_equal(np.isnan(rebuilt), np.isnan(values)):
        raise ValueError("Rebuilt values changed candidate availability")
    if not ((rebuilt[np.isfinite(rebuilt)] >= 0) & (rebuilt[np.isfinite(rebuilt)] <= 1)).all():
        raise ValueError("Rebuilt probabilities outside [0, 1]")
    result = model.copy()
    result[players] = rebuilt
    return result, {"rows": len(model), "finite_values": int(np.isfinite(rebuilt).sum()),
                    "blended_pass_candidates": int(eligible.sum()), "offside_zeros": int(offside.sum()),
                    "self_values_preserved": int((np.isfinite(values) & ~candidate).sum()),
                    "changed_values": int((eligible & (rebuilt != values)).sum())}


def tracking_inputs(model, path):
    """Reconstruct the graph's offside rule from original, unrounded tracking x coordinates."""
    parquet = pq.ParquetFile(path)
    tracking_players = [c[:-2] for c in parquet.schema.names if re.fullmatch(r"(home|away)_\d+_x", c)]
    columns = ["period_id", "frame_id", "ball_x"] + [f"{p}_{axis}" for p in tracking_players for axis in ("x", "y")]
    tracking = parquet.read(columns=columns, use_threads=False).to_pandas()
    if not tracking.index.is_unique:
        raise ValueError("Duplicate tracking frame identity")
    tracking = tracking.loc[model.state_frame_id]
    if not np.array_equal(tracking.period_id.to_numpy(), model.period_id.to_numpy()):
        raise ValueError("Tracking period differs from the component state")
    x = tracking[[p + "_x" for p in tracking_players]].to_numpy(dtype=float)
    y = tracking[[p + "_y" for p in tracking_players]].to_numpy(dtype=float)
    ball = tracking.ball_x.to_numpy(dtype=float)
    away = model.possessor_id.str.startswith("away_").to_numpy()
    x[away] = FIELD_SIZE[0] - x[away]
    y[away] = FIELD_SIZE[1] - y[away]
    ball[away] = FIELD_SIZE[0] - ball[away]
    teammate = np.array([p.startswith("home_") for p in tracking_players])[None, :] != away[:, None]
    flags = calculate_offside_flags(x, ball, teammate, np.zeros_like(teammate)).astype(bool)
    players = [c for c in model if PLAYER.fullmatch(c)]
    positions = {p: i for i, p in enumerate(tracking_players)}
    own = np.array([positions[p] for p in model.possessor_id])
    selected = [positions[p] for p in players]
    # v5 only needs finite distances; their magnitudes do not enter its weight.
    distances = np.hypot(x[:, selected] - x[np.arange(len(model)), own, None],
                         y[:, selected] - y[np.arange(len(model)), own, None])
    offside = flags[:, selected]
    values = model[players].to_numpy(dtype=float)
    candidate = np.isfinite(values) & (np.array(players)[None, :] != model.possessor_id.to_numpy()[:, None])
    offside &= candidate
    if not np.array_equal(offside, (values == 0) & candidate):
        raise ValueError("Tracking offside mask does not exactly match the saved zero mask")
    return offside, distances


def prepare_match(run, cache_root, metadata, settings, staging, mid):
    directory = run / mid
    if (directory / "pass_success.parquet").exists():
        raise FileExistsError(f"Refusing to replace existing pass_success for {mid}")
    originals = sorted(directory.glob("*.parquet")) + [directory / "possession_manifest.json"]
    before = {p.name: cache.checksum(p) for p in originals}
    model, height, intent = [pd.read_parquet(directory / f"{name}.parquet")
                             for name in ("model_pass_success", "pass_height", "pass_intent")]
    if not model.stats_perform_match_id.eq(mid).all():
        raise ValueError(f"Component match identity differs: {mid}")
    if len(model) != metadata["possession_reports"][mid]["selected_states"]:
        raise ValueError(f"Component state count differs: {mid}")
    manifest = cache.read_json(directory / "possession_manifest.json")
    frames, _, record = cache.load_match(cache_root, mid, include_lane=False)
    physical = frames["matches"]
    if not physical.frame_scope.eq("frame_id").all():
        raise ValueError(f"Unexpected physical cache frame scope: {mid}")
    partitions = {}
    for pid, group in model.groupby("state_partition_id", sort=False):
        identity = group[["period_id", "possessor_id", "pc_cache_match_id"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError(f"Ambiguous component partition: {mid}/{pid}")
        row = identity.iloc[0]
        descriptor = dict(match_id=mid, possession_id=int(pid), state_partition_id=int(pid),
                          period_id=int(row.period_id), possessor_id=row.possessor_id, **manifest["provenance"])
        selection = cache.Selection(int(pid), row.pc_cache_match_id, descriptor=descriptor)
        resolved, _, mode = cache._resolve_possession(cache_root, mid, selection, record["descriptors"])
        stored = physical.loc[physical.possession_id.eq(int(pid))]
        if not stored.possession_provenance_id.eq(resolved).all():
            raise ValueError(f"Physical row provenance differs: {mid}/{pid}")
        partitions[str(pid)] = {"rows": len(group), "resolution": mode,
                                "runtime_provenance_id": row.pc_cache_match_id,
                                "cached_provenance_id": resolved}
    tracking_path = ROOT / "data" / "tracking_processed" / f"{mid}.parquet"
    tracking_signature = signature(tracking_path)
    offside, distances = tracking_inputs(model, tracking_path)
    result, stats = rebuild_v5_table(model, height, intent, physical, offside, distances, **settings)
    destination = staging / f"{mid}.parquet"
    result.to_parquet(destination, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(destination), result, check_exact=True)
    return {**stats, "partitions": partitions, "original_files_sha256": before,
            "output_sha256": cache.checksum(destination), "tracking_signature": tracking_signature,
            "cache_output_sha256": {k: v["sha256"] for k, v in record["outputs"].items()},
            "cache_completion_signature": signature(cache_root / "completion" / f"{mid}.json"),
            "cache_output_signatures": {k: signature(cache_root / k / f"{mid}.parquet") for k in record["outputs"]}}


def repair_run(run, workers=4):
    run = Path(run).resolve()
    metadata_path = run / "metadata.json"
    original_metadata = metadata_path.read_bytes()
    metadata = json.loads(original_metadata)
    if (metadata.get("state_contract") != "sportec_possession_states_v3"
            or metadata.get("physical_xpass_weight_version") != "v5"
            or metadata.get("physical_xpass_runtime_source") != "pc_xpass"
            or metadata.get("use_carries") or metadata.get("physical_xpass_ball_z_limit") is not None):
        raise ValueError("Repair supports Sportec v3, v5 pc-xPass, without carries or a ball-z limit")
    if metadata.get("inference_masks", {}).get("configuration_id") != "unmasked":
        raise ValueError("Repair requires unmasked exported auxiliaries")
    for role in ("pass_height", "pass_intent"):
        if metadata["models"][role] != metadata["xpass_auxiliaries"][role]["model_id"]:
            raise ValueError(f"Exported {role} is not the selected blend auxiliary")
    model_args_path = ROOT / "saved" / metadata["models"]["pass_success"] / "args.json"
    model_args = cache.read_json(model_args_path)
    if not model_args.get("offside_aware") or model_args.get("node_in_dim") not in (20, 24, 26):
        raise ValueError("Repair requires the recorded offside-aware pass-success checkpoint")
    eps = float(model_args.get("physical_eps", 1e-4))
    settings = dict(metric=metadata["physical_xpass_metric"], eps=eps,
                    floor=model_args.get("physical_xpass_floor"),
                    intent_threshold=float(metadata["physical_xpass_v5_intent_threshold"]),
                    discount=bool(metadata["physical_xpass_v5_discount"]))
    matches = metadata["processed_match_ids"]
    if not matches or len(set(matches)) != len(matches) or metadata.get("skipped_matches"):
        raise ValueError("Repair requires a completed match inventory without skipped matches")
    for mid in matches:
        cache.validate_match_id(mid)
        if (run / mid / "pass_success.parquet").exists():
            raise FileExistsError(f"Refusing to replace existing pass_success for {mid}")
    cache_root = Path(metadata["physical_cache_dir"]).resolve()
    cache_metadata_hash = cache.checksum(cache_root / "metadata.json")
    repair_id = "pass_success_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    audit = run / "repairs" / repair_id
    staging = audit / "staged"
    staging.mkdir(parents=True)
    (audit / "metadata_before.json").write_bytes(original_metadata)
    report = {"repair_id": repair_id, "started_at": stamp(), "status": "preparing",
              "command": sys.argv, "run_id": metadata["run_id"], "settings": settings,
              "method": "saved_component_v5_reblend", "cache_metadata_sha256": cache_metadata_hash,
              "model_args_sha256": cache.checksum(model_args_path),
              "source_code_sha256": {p.name: cache.checksum(p) for p in
                                     (Path(__file__), ROOT / "pc_xpass_match_cache.py", ROOT / "physical_pass_model.py")},
              "matches": {}, "published_matches": []}
    report_path = audit / "report.json"
    atomic_json(report_path, report)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(prepare_match, run, cache_root, metadata, settings, staging, mid): mid
                       for mid in matches}
            for index, future in enumerate(as_completed(futures), 1):
                mid = futures[future]
                report["matches"][mid] = future.result()
                if index % 25 == 0 or index == len(matches):
                    print(f"Prepared {index}/{len(matches)} matches", flush=True)
        # Confirm inputs stayed unchanged while staging all results.
        for mid, details in report["matches"].items():
            for name, digest in details["original_files_sha256"].items():
                if cache.checksum(run / mid / name) != digest:
                    raise ValueError(f"Source component changed during repair: {mid}/{name}")
            if signature(ROOT / "data/tracking_processed" / f"{mid}.parquet") != details["tracking_signature"]:
                raise ValueError(f"Tracking changed during repair: {mid}")
            if signature(cache_root / "completion" / f"{mid}.json") != details["cache_completion_signature"]:
                raise ValueError(f"Cache completion changed during repair: {mid}")
            for name, before in details["cache_output_signatures"].items():
                if signature(cache_root / name / f"{mid}.parquet") != before:
                    raise ValueError(f"Cache changed during repair: {mid}/{name}")
        if (metadata_path.read_bytes() != original_metadata
                or cache.checksum(cache_root / "metadata.json") != cache_metadata_hash
                or cache.checksum(model_args_path) != report["model_args_sha256"]):
            raise ValueError("Run, cache metadata, or model settings changed during repair")
        report["status"] = "publishing"
        report["totals"] = {key: sum(record[key] for record in report["matches"].values()) for key in
                            ("rows", "finite_values", "blended_pass_candidates", "offside_zeros",
                             "self_values_preserved", "changed_values")}
        atomic_json(report_path, report)
        for mid in matches:
            destination = run / mid / "pass_success.parquet"
            # Path.rename refuses to replace an existing destination on Windows.
            if destination.exists():
                raise FileExistsError(destination)
            (staging / f"{mid}.parquet").rename(destination)
            if cache.checksum(destination) != report["matches"][mid]["output_sha256"]:
                raise ValueError(f"Published output checksum differs: {mid}")
            report["published_matches"].append(mid)
        runtime_stats = {mid: {pid: {"pass_success": {"cache_hits": part["rows"], "cache_misses": 0,
                                                     "execution_mode": "saved_component_v5_reblend"}}
                               for pid, part in details["partitions"].items()}
                         for mid, details in report["matches"].items()}
        metadata["physical_xpass_skipped_actions"] = {}
        metadata["physical_xpass_runtime_stats"] = runtime_stats
        metadata["physical_xpass_cache_summary"] = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True, cache_disabled=False, refresh_requested=False,
            cache_dir=cache_root, runtime_stats=runtime_stats, skipped_stats={})
        completed = stamp()
        metadata.setdefault("repair_history", []).append({
            "repair_id": repair_id, "component": "pass_success", "completed_at": completed,
            "method": report["method"], "report": str(report_path.relative_to(run)),
            "original_metadata": str((audit / "metadata_before.json").relative_to(run)),
            "match_count": len(matches), "rows": report["totals"]["rows"],
        })
        atomic_json(metadata_path, metadata)
        report.update(status="completed", completed_at=completed,
                      repaired_metadata_sha256=cache.checksum(metadata_path),
                      validation={"roundtrip_exact": True, "original_component_checksums_unchanged": True,
                                  "original_state_ids_preserved": True, "cache_checksums_verified": True,
                                  "tracking_offside_mask_verified": True})
        atomic_json(report_path, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(report_path, report)
        raise
    print(json.dumps({"report": str(report_path), "matches": len(matches), **report["totals"]}), flush=True)
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(1)
    repair_run(args.component_run_dir, args.workers)


if __name__ == "__main__":
    main()
