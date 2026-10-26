"""Prepare a resumable, independently validated pc-xPass acceleration table."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pc_xpass_acceleration as accel
import pc_xpass_distance as spatial
from pc_xpass_versions import atomic_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table-id", required=True)
    p.add_argument("--margin", choices=["tta", "distance"], default="tta")
    p.add_argument("--max-time", type=float, help="Distance-table time coverage in seconds; default covers max-distance at min-ball-speed")
    p.add_argument("--min-ball-speed", type=float, default=3.)
    p.add_argument("--ball-dec", default="drag", help="For automatic distance-table time coverage: drag or nonnegative deceleration")
    p.add_argument("--accel", required=True, type=float)
    p.add_argument("--source-table-id", help="Reuse a completed table and its preparation state in a new table ID")
    p.add_argument("--max-player-speed", type=float, default=5.)
    p.add_argument("--max-player-speed-off", type=float)
    p.add_argument("--max-player-speed-def", type=float)
    p.add_argument("--max-distance", type=float, default=150.)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--max-cells", type=int, default=None, help="Maximum tree nodes; difficult leaves use explicit solver fallback")
    p.add_argument("--max-depth", type=int, default=24)
    args = p.parse_args(argv)
    for key in ("accel", "max_player_speed", "max_distance", "max_player_speed_off", "max_player_speed_def", "max_time", "min_ball_speed"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error("--"+key.replace("_", "-")+" must be positive and finite")
    if args.max_cells is None:
        args.max_cells = 100000 if args.margin == "distance" else 20000
    if args.workers < 1 or args.max_cells < 1 or args.max_depth < 1:
        p.error("workers, max-cells, and max-depth must be positive")
    if args.margin == "distance" and args.max_time is None:
        from physical_pass_model import parse_ball_dec, pc_xpass_ball_arrival_times
        dec = parse_ball_dec(args.ball_dec)
        if dec == "drag":
            times, _ = pc_xpass_ball_arrival_times(np.array([args.max_distance]), np.array([args.min_ball_speed]), dec)
            args.max_time = float(times.max())
        elif not math.isfinite(dec) or dec < 0:
            p.error("--ball-dec must be drag or a nonnegative finite number")
        elif dec == 0:
            args.max_time = args.max_distance / args.min_ball_speed
        else:
            # Any reachable constant-deceleration trajectory takes at most V0/dec.
            args.max_time = max(args.min_ball_speed / dec, math.sqrt(2*args.max_distance/dec))
    return args


def _load_table(args):
    return (spatial.load_table if getattr(args, "margin", "tta") == "distance" else accel.load_table)(args.table_id)


def _baseline(points, distance=False):
    t = np.expm1(np.asarray(points)[..., 2])
    return spatial.standing_radius(t) if distance else t


def _key(point):
    return ",".join(float(x).hex() for x in point)


def _split(nodes, index, axis):
    node = nodes[index]
    lo, hi = np.array(node["low"]), np.array(node["high"])
    midpoint = .5*(lo[axis]+hi[axis])
    children = []
    for side in (0, 1):
        child_lo, child_hi = lo.copy(), hi.copy()
        if side == 0:
            child_hi[axis] = midpoint
        else:
            child_lo[axis] = midpoint
        inherited = [p for p in node.get("failed_probes", [])
                     if np.all(np.asarray(p) >= child_lo) and np.all(np.asarray(p) <= child_hi)]
        children.append(len(nodes))
        nodes.append(dict(low=child_lo.tolist(), high=child_hi.tolist(), depth=node["depth"]+1,
                          children=[-1, -1], axis=-1, values=[0.]*8, fallback=True, failed_probes=inherited))
    node["axis"], node["children"] = axis, children
    return children


def _seed_from_source(args, root, caps):
    import shutil
    source = accel.table_path(args.source_table_id)
    meta, arrays = (spatial.load_table if getattr(args, "margin", "tta") == "distance" else accel.load_table)(args.source_table_id)
    requested = dict(accel=args.accel, speeds=caps, max_distance=args.max_distance)
    if getattr(args, "margin", "tta") == "distance":
        requested["max_time"] = args.max_time
    if any(meta["definition"].get(k) != value for k, value in requested.items()):
        raise ValueError("Source table must use the same preparation accel, speed caps and distance")
    if not (source/"checkpoint.json").exists() or not (source/"reference.sqlite").exists():
        raise ValueError("Source-table refinement requires its checkpoint and reference.sqlite preparation state")
    state = json.loads((source/"checkpoint.json").read_text())
    nodes, pending = state["nodes"], []
    if args.max_cells < len(nodes)+2:
        raise ValueError("Source-table refinement needs --max-cells larger than the source node count")
    shutil.copy2(source/"reference.sqlite", root/"reference.sqlite")
    db = sqlite3.connect(root/"reference.sqlite")
    global_rng = np.random.default_rng(42)
    original_count = len(nodes)
    try:
        for index in range(original_count):
            node = nodes[index]
            if node["children"][0] >= 0:
                continue
            if getattr(args, "margin", "tta") == "distance" and node["fallback"]:
                # Re-evaluate every unvalidated support interval, not only final
                # holdout failures. Its reference probes are reused from SQLite.
                pending.append(index)
                continue
            if node["fallback"] and not any(node["values"]):
                continue
            rng = np.random.default_rng(42+index) if meta.get("holdout_seed") == "per_cell_index_42" else global_rng
            weights = rng.uniform(.05, .95, size=(2, 3))
            if not node["fallback"]:
                continue
            lo, hi = np.array(node["low"]), np.array(node["high"])
            node["failed_probes"] = []
            for weight in weights:
                point = (lo+(hi-lo)*weight).tolist()
                saved = db.execute("SELECT result FROM points WHERE key=?", (_key(point),)).fetchone()
                if saved:
                    reference = json.loads(saved[0])
                    estimate = float(accel._interpolate(np.array([node["values"]]), weight[None, :])[0]+_baseline(point, getattr(args, "margin", "tta") == "distance"))
                    if "unresolved" in reference or max(abs(estimate-reference["lower"]), abs(estimate-reference["upper"])) > meta.get("distance_error_normalized", meta.get("time_error_normalized")):
                        node["failed_probes"].append(point)
            if len(nodes)+2 <= args.max_cells and node["depth"] < args.max_depth:
                axis = int(np.argmax((hi-lo)/(np.array(nodes[0]["high"])-np.array(nodes[0]["low"]))))
                pending.extend(_split(nodes, index, axis))
    finally:
        db.close()
    atomic_json(root/"checkpoint.json", dict(nodes=nodes, pending=pending,
                statistics=state.get("statistics", {}), holdout_refinement=True))


def _build(args):
    root = accel.table_path(args.table_id)
    root.mkdir(parents=True, exist_ok=True)
    caps = [args.max_player_speed_off or args.max_player_speed, args.max_player_speed_def or args.max_player_speed]
    distance = getattr(args, "margin", "tta") == "distance"
    worker = spatial.reference_point if distance else accel.reference_point
    tau = spatial.LOOKUP_TOL*args.accel/max(caps)**2 if distance else accel.TIME_TOL*args.accel/max(caps)
    rho = accel.POSITION_TOL*args.accel/max(caps)**2
    radius_max = args.accel*args.max_distance/min(caps)**2
    # Polar coordinates and subtraction of the radial travel-time baseline make
    # large-distance cells inexpensive without changing the motion model.
    if any(not math.isfinite(value) or value <= 0 for value in (tau, rho, radius_max)):
        raise ValueError("Preparation settings exceed finite normalized numerical scales")
    low, high = [0., 0., math.log1p(min(1e-4, radius_max/1024))], [1., math.pi, math.log1p(radius_max)]
    definition = dict(movement_version=accel.VERSION, accel=args.accel, speeds=caps,
                      max_distance=args.max_distance, time_error_normalized=tau,
                      position_tolerance_normalized=rho, coordinates="speed_angle_log1p_radius",
                      max_cells=args.max_cells, max_depth=args.max_depth)
    if distance:
        low, high = [0., 0., 0.], [1., math.pi, math.log1p(args.max_time*args.accel/min(caps))]
        definition.update(margin="distance", spatial_version=spatial.VERSION, max_time=args.max_time,
                          distance_error_normalized=tau, coordinates="speed_normal_angle_log1p_time")
        definition.pop("time_error_normalized")
    if args.source_table_id:
        definition["source_table_id"] = args.source_table_id
    meta_path = root/"metadata.json"
    if args.source_table_id and not meta_path.exists():
        _seed_from_source(args, root, caps)
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("definition") != definition:
            raise ValueError("Table ID already has different preparation settings; use a new ID")
        if existing.get("status") == "complete":
            _load_table(args)
            print("Reusing completed table " + str(root), flush=True)
            return existing
    atomic_json(meta_path, dict(status="preparing", movement_version=accel.VERSION, definition=definition))
    checkpoint = root/"checkpoint.json"
    state = {}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        nodes, pending = state["nodes"], state["pending"]
        if not pending and not any(not node["fallback"] for node in nodes):
            nodes = [dict(low=low, high=high, depth=0, children=[-1, -1], axis=-1, values=[0.]*8, fallback=True)]
            pending = [0]
    else:
        nodes = [dict(low=low, high=high, depth=0, children=[-1, -1], axis=-1, values=[0.]*8, fallback=True)]
        pending = [0]
    reserve = min(max(64, args.max_cells//10), args.max_cells//2)
    primary_limit = args.max_cells if state.get("holdout_refinement") else max(1, args.max_cells-reserve)
    db = sqlite3.connect(root/"reference.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS points (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
    memo = {key: json.loads(value) for key, value in db.execute("SELECT key, result FROM points")}
    corners = accel._CORNERS
    probes = np.vstack([np.full(3, .5),
                        *[np.where(np.arange(3) == axis, side, .5) for axis in range(3) for side in (0., 1.)],
                        [.25, .25, .75], [.75, .25, .25], [.25, .75, .25], [.75, .75, .75]])
    started = time.perf_counter()
    last_progress = started
    validations, failures, maximum_error = 0, 0, 0.
    previous = {}
    if checkpoint.exists():
        previous = state.get("statistics", {})
        validations = previous.get("validations", 0)
        failures = previous.get("failures", 0)
        maximum_error = previous.get("maximum_error", 0.)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            while pending:
                batch = pending[:max(1, 2*args.workers)]
                points = {}
                for index in batch:
                    node = nodes[index]
                    lo, hi = np.array(node["low"]), np.array(node["high"])
                    sampled = list(lo+(hi-lo)*np.vstack([corners, probes]))+node.get("failed_probes", [])
                    for point in sampled:
                        if _key(point) not in memo:
                            points[_key(point)] = point.tolist()
                for point, result in pool.map(worker,
                                              [(p, tau*.5, rho) for p in points.values()], chunksize=1):
                    key = _key(point); memo[key] = result
                    db.execute("INSERT OR REPLACE INTO points VALUES (?, ?)", (key, json.dumps(result)))
                db.commit()
                pending = pending[len(batch):]
                for index in batch:
                    node = nodes[index]
                    lo, hi = np.array(node["low"]), np.array(node["high"])
                    all_points = np.asarray(list(lo+(hi-lo)*np.vstack([corners, probes]))+node.get("failed_probes", []))
                    probe_weights = (all_points[8:]-lo)/(hi-lo)
                    results = [memo[_key(p)] for p in all_points]
                    unresolved = any("unresolved" in r for r in results)
                    if unresolved:
                        failures += 1
                        curvature = np.eye(3)[node["depth"] % 3]
                    else:
                        center_times = np.array([.5*(r["lower"]+r["upper"]) for r in results])
                        delays = center_times-_baseline(all_points, distance)
                        values = delays[:8]
                        estimated = accel._interpolate(np.broadcast_to(values, (len(probe_weights), 8)), probe_weights)+_baseline(all_points[8:], distance)
                        errors = np.array([max(abs(t-r["lower"]), abs(t-r["upper"])) for t, r in zip(estimated, results[8:])])
                        worst = float(errors.max())
                        validations += len(probes)
                        if worst <= tau:
                            node["values"], node["fallback"] = values.tolist(), False
                            maximum_error = max(maximum_error, worst)
                            continue
                        curvature = [abs(.5*(delays[9+2*k]+delays[10+2*k])-delays[8]) for k in range(3)]
                    if node["depth"] >= args.max_depth or len(nodes)+2 > primary_limit:
                        continue
                    # Face-centre curvature chooses the interpolation dimension.
                    axis = int(np.argmax(curvature))
                    node["axis"] = axis
                    pending.extend(_split(nodes, index, axis))
                statistics = dict(validations=validations, failures=failures, maximum_error=maximum_error)
                if time.perf_counter()-last_progress >= 2. or not pending:
                    atomic_json(checkpoint, dict(nodes=nodes, pending=pending, statistics=statistics,
                                holdout_refinement=state.get("holdout_refinement", False)))
                    print(f"nodes={len(nodes)} pending={len(pending)} solved_points={len(memo)} "
                          f"validated_leaves={sum(not n['fallback'] for n in nodes)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                    last_progress = time.perf_counter()
    finally:
        db.close()
    # Final holdouts are independent of the probes used to choose refinements.
    holdouts = []
    for index, node in enumerate(nodes):
        if node["children"][0] < 0 and not node["fallback"]:
            lo, hi = np.array(node["low"]), np.array(node["high"])
            for weights in np.random.default_rng(42+index).uniform(.05, .95, size=(2, 3)):
                holdouts.append((index, weights, (lo+(hi-lo)*weights).tolist()))
    holdout_failures = set()
    holdout_progress = time.perf_counter()
    db = sqlite3.connect(root/"reference.sqlite")
    try:
        missing = { _key(point): point for _, _, point in holdouts if _key(point) not in memo }
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for count, (point, result) in enumerate(pool.map(worker,
                    [(point, tau*.5, rho) for point in missing.values()], chunksize=1), 1):
                key = _key(point); memo[key] = result
                db.execute("INSERT OR REPLACE INTO points VALUES (?, ?)", (key, json.dumps(result)))
                if count % 32 == 0:
                    db.commit()
                    if time.perf_counter()-holdout_progress >= 2.:
                        print(f"independent_holdouts={count}/{len(missing)}", flush=True)
                        holdout_progress = time.perf_counter()
        db.commit()
    finally:
        db.close()
    for index, weights, point in holdouts:
        reference = memo[_key(point)]
        estimate = float(accel._interpolate(np.array([nodes[index]["values"]]), weights[None, :])[0]+_baseline(point, getattr(args, "margin", "tta") == "distance"))
        if "unresolved" in reference or max(abs(estimate-reference["lower"]), abs(estimate-reference["upper"])) > tau:
            nodes[index]["fallback"] = True
            nodes[index].setdefault("failed_probes", []).append(point)
            holdout_failures.add(index)
    pending = []
    for index in sorted(holdout_failures):
        node = nodes[index]
        if node["depth"] < args.max_depth and len(nodes)+2 <= args.max_cells:
            lo, hi = np.array(node["low"]), np.array(node["high"])
            axis = int(np.argmax((hi-lo)/(np.array(nodes[0]["high"])-np.array(nodes[0]["low"]))))
            pending.extend(_split(nodes, index, axis))
    atomic_json(checkpoint, dict(nodes=nodes, pending=pending, statistics=dict(validations=validations,
                failures=failures, maximum_error=maximum_error), holdout_refinement=True))
    if pending:
        # Preserve the outer OS lock and reuse all solved reference points.
        del memo, nodes, holdouts
        return _build(args)
    arrays = dict(low=np.array([n["low"] for n in nodes]), high=np.array([n["high"] for n in nodes]),
                  values=np.array([n["values"] for n in nodes]), children=np.array([n["children"] for n in nodes]),
                  axis=np.array([n["axis"] for n in nodes]), fallback=np.array([n["fallback"] for n in nodes]))
    temporary = root/"table.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(root/"table.npz")
    leaves = [n for n in nodes if n["children"][0] < 0]
    meta = dict(status="complete", movement_version=accel.VERSION, definition=definition,
                fingerprint=accel._digest(root/"table.npz"), time_error_normalized=tau,
                position_tolerance_normalized=rho, max_distance=args.max_distance,
                reference_points=len(memo), validation_points=validations,
                validated_max_error_normalized=maximum_error, unresolved_cells=failures,
                leaves=len(leaves), fallback_leaves=sum(n["fallback"] for n in leaves),
                preparation_seconds=time.time()-root.stat().st_ctime,
                preparation_timing="wall time since table directory creation; includes interrupted/resumed periods",
                independent_holdout_points=len(holdouts), independent_holdout_failed_leaves=len(holdout_failures),
                holdout_seed="per_cell_index_42", preparation_workers=args.workers,
                validation="deterministic interior/face probes plus independent seeded holdouts; not an exhaustive interpolation-error proof")
    if not any(not n["fallback"] for n in leaves):
        raise accel.UnresolvedArrival("No interpolation leaves met acceptance criteria; table remains unpublished")
    source_seconds = 0.
    if args.source_table_id:
        source_meta, _ = (spatial.load_table if distance else accel.load_table)(args.source_table_id)
        source_seconds = source_meta.get("cumulative_preparation_seconds", source_meta["preparation_seconds"])
    if distance:
        meta.update(spatial_version=spatial.VERSION, margin="distance", supported_margins=["distance"],
                    max_time=args.max_time, distance_error_normalized=tau,
                    reference_bound_target_m=spatial.REFERENCE_TOL,
                    combined_spatial_target_m=spatial.SPATIAL_TOL,
                    angular_maximization="analytic_global_on_piecewise_linear_support_interpolant",
                    reachability_time="exact")
        meta.pop("time_error_normalized")
        meta["reference_failures_seen"] = meta["unresolved_cells"]
        meta["unresolved_cells"] = sum(
            node["fallback"] and any("unresolved" in memo[_key(point)]
                for point in list(np.array(node["low"])+(np.array(node["high"])-np.array(node["low"]))*np.vstack([corners, probes]))+node.get("failed_probes", []))
            for node in leaves)
    meta["cumulative_preparation_seconds"] = source_seconds+meta["preparation_seconds"]
    from importlib.metadata import version
    meta["solver"] = {"name": "clarabel", "version": version("clarabel")}
    atomic_json(meta_path, meta)
    _load_table(args)
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def build(args):
    # OS locks release on interruption; a leftover lock file is harmless.
    root = accel.table_path(args.table_id)
    root.mkdir(parents=True, exist_ok=True)
    with (root/"prepare.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another process is preparing this table ID") from exc
            try:
                return _build(args)
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another process is preparing this table ID") from exc
            try:
                return _build(args)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    build(parse_args())
