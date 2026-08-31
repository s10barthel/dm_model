from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.config import LABEL_INDEX
from datatools.hawkeye import (
    COMPONENT_COLUMNS,
    apply_hawkeye_possessor_offset,
    build_hawkeye_export,
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    infer_hawkeye_components,
    load_hawkeye_ball,
    load_hawkeye_models,
    load_hawkeye_tracking,
)
from inference import configure_lane_survival_runtime_cache
from models.utils import (
    extract_model_feature_signature,
    get_model_provenance,
    resolve_model_selection,
    validate_model_graph_schemas,
)
from physical_pass_model import (
    AS_DEFAULT_ANGLE_STEP_DEG,
    AS_DEFAULT_COARSE_N_ANGLES,
    AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    AS_DEFAULT_REFINE_TOP_K_ANGLES,
    PC_XPASS_DEFAULT_BOOST_DEF_ENDPOINT_CONTROL,
    PC_XPASS_DEFAULT_CONTROL_INFLECTION_POINT,
    PC_XPASS_DEFAULT_CONTROL_POWER,
    PC_XPASS_DEFAULT_DIST_PASS_DIV,
    PC_XPASS_DEFAULT_DIST_PASS_MAX,
    PC_XPASS_DEFAULT_DIST_PASS_MIN,
    PC_XPASS_DEFAULT_ENDPOINT_NORMALIZATION,
    PC_XPASS_DEFAULT_LANE_INFLECTION_POINT,
    PC_XPASS_DEFAULT_LANE_POWER,
    PC_XPASS_DEFAULT_MAX_PLAYER_SPEED,
    PC_XPASS_DEFAULT_MAX_SPEED,
    PC_XPASS_DEFAULT_MIN_SPEED,
    PC_XPASS_DEFAULT_POSITION_DISCOUNT_DISTANCE,
    PC_XPASS_DEFAULT_POSITION_DISCOUNT_POWER,
    PC_XPASS_DEFAULT_RADIAL_GRIDSIZE,
    PC_XPASS_DEFAULT_REACTION_TIME,
    PC_XPASS_DEFAULT_SPEED_STEP,
    PC_XPASS_DEFAULT_USE_POSITION_DISCOUNT,
    PC_XPASS_REACTION_TIME_MODE_DIST_PASS,
    PC_XPASS_REACTION_TIME_MODE_FIXED,
    PHYSICAL_DEFAULT_MAX_AUTO_WORKERS,
    PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR,
    PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR,
    PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR,
    PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    PHYSICAL_XPASS_SPEED_AGGREGATIONS,
    PHYSICAL_XPASS_DEFAULT_TOP_N,
    resolve_physical_num_workers,
)
from project_config import (
    HAWKEYE_LOC_COMPONENT_RUNS_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_pc_xpass_dir,
    write_latest_run,
    write_run_metadata,
)
from scripts.generate_physical_xpass import (
    enabled_physical_xpass_metrics_from_args,
    merge_stats,
    prewarm_runtime_items,
    resolve_runtime_row_window,
    runtime_cache_items_from_graphs,
    write_runtime_dataset_metadata,
)
from scripts.run_hawkeye import resolve_optional_model_id


DEFAULT_INPUT_FILE = REPOSITORY_ROOT / "data_analysis" / "data" / "dm_processed.csv"
DEFAULT_TIME_TOLERANCE = 0.1
DEFAULT_PC_XPASS_CACHE_DIR = get_pc_xpass_dir("hawkeye_loc")
REQUIRED_INPUT_COLUMNS = [
    "selection_row_id",
    "action_id",
    "SelectedPlayer",
    "pass_moment",
    "PositionX",
    "PositionY",
    "loc_info_missing",
]
MISSING_REPORT_COLUMNS = [
    "selection_row_id",
    "action_id",
    "SelectedPlayer",
    "pass_moment",
    "PositionX",
    "PositionY",
    "participant",
    "setting",
    "SceneNr",
    "loc_status",
    "loc_missing_reason",
]
REQUIRED_SCORE_COMPONENTS = [
    "action_intent",
    "pass_intent",
    "pass_success",
    "outcome_scoring_success",
    "outcome_scoring_failure",
    "outcome_conceding_success",
    "outcome_conceding_failure",
]


def parse_bool_text(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-frame location-adjusted Hawkeye inference.")
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--tracking-csv", default=str(PROJECT_ROOT / "hawkeye_data" / "centroid_data_team.csv"))
    parser.add_argument("--ball-csv", default=str(PROJECT_ROOT / "hawkeye_data" / "ball_data_selected.csv"))
    parser.add_argument("--time-tolerance", type=float, default=DEFAULT_TIME_TOLERANCE)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--situation-id",
        action="append",
        help="Select the first input row for an action_id; repeat for multiple ids, or use 'all'.",
    )
    selection_group.add_argument("--limit", type=int, help="Process the first N input CSV rows without deduplication.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-id")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--pass-height-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", help="Run parent directory; defaults to data/component_runs/hawkeye_loc.")
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse of an explicitly named run directory.")
    parser.add_argument(
        "--pc-xpass-cache-dir",
        "--physical-cache-dir",
        dest="pc_xpass_cache_dir",
        default=str(DEFAULT_PC_XPASS_CACHE_DIR),
    )
    parser.add_argument("--lane-survival-cache-dir")

    # Inference blending flags from run_hawkeye.py.
    parser.add_argument(
        "--xpass-version", "--x-pass-version", "--x_pass_version", dest="x_pass_version", default="top10"
    )
    parser.add_argument("--xpass-weight", "--xpass_weight", choices=["v1", "v2", "v3", "v4"], default="v3")
    parser.add_argument("--v4-power", type=float)
    parser.add_argument("--v4-zero", type=float)
    parser.add_argument("--discount", dest="v4_discount", type=parse_bool_text, default=None)
    parser.add_argument("--ball-z-limit", default="none")

    # Applicable pc-xPass generation flags from generate_physical_xpass.py.
    parser.add_argument("--physical-eps", type=float, default=1e-4)
    parser.add_argument(
        "--speed-aggregation",
        choices=sorted(PHYSICAL_XPASS_SPEED_AGGREGATIONS),
        default=PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    )
    parser.add_argument("--max-speed", "--max_speed", dest="max_speed", type=float, default=PC_XPASS_DEFAULT_MAX_SPEED)
    parser.add_argument("--min-speed", "--min_speed", dest="min_speed", type=float, default=PC_XPASS_DEFAULT_MIN_SPEED)
    parser.add_argument("--speed-step", "--speed_step", dest="speed_step", type=float, default=PC_XPASS_DEFAULT_SPEED_STEP)
    parser.add_argument("--coarse-n-angles", "--coarse_n_angles", dest="coarse_n_angles", type=int, default=AS_DEFAULT_COARSE_N_ANGLES)
    parser.add_argument("--refine-top-k-angles", "--refine_top_k_angles", dest="refine_top_k_angles", type=int, default=AS_DEFAULT_REFINE_TOP_K_ANGLES)
    parser.add_argument("--refine-angle-radius", "--refine_angle_radius", dest="refine_angle_radius", type=float, default=AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG)
    parser.add_argument("--angle-step", "--angle_step", dest="angle_step", type=float, default=AS_DEFAULT_ANGLE_STEP_DEG)
    parser.add_argument("--radial-gridsize", "--radial_gridsize", dest="radial_gridsize", type=float, default=PC_XPASS_DEFAULT_RADIAL_GRIDSIZE)
    parser.add_argument("--sigma-angle", "--sigma_angle", dest="sigma_angle", type=float, default=PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR)
    parser.add_argument("--sigma-speed", "--sigma_speed", dest="sigma_speed", type=float, default=PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR)
    parser.add_argument("--sigma-distance", "--sigma_distance", dest="sigma_distance", type=float, default=PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR)
    parser.add_argument("--top-n", "--top_n", dest="top_n", type=int, default=PHYSICAL_XPASS_DEFAULT_TOP_N)
    parser.add_argument("--top-n-values", "--top_n_values", dest="top_n_values", type=int, nargs="+", default=None)
    parser.add_argument("--top-xt", action="store_true")
    parser.add_argument("--reaction-time", "--reaction_time", dest="reaction_time", default=PC_XPASS_DEFAULT_REACTION_TIME)
    parser.add_argument("--dist-pass-div", "--dist_pass_div", dest="dist_pass_div", type=float, default=PC_XPASS_DEFAULT_DIST_PASS_DIV)
    parser.add_argument("--dist-pass-min", "--dist_pass_min", dest="dist_pass_min", type=float, default=PC_XPASS_DEFAULT_DIST_PASS_MIN)
    parser.add_argument("--dist-pass-max", "--dist_pass_max", dest="dist_pass_max", type=float, default=PC_XPASS_DEFAULT_DIST_PASS_MAX)
    parser.add_argument("--max-player-speed", "--max_player_speed", dest="max_player_speed", type=float, default=PC_XPASS_DEFAULT_MAX_PLAYER_SPEED)
    parser.add_argument("--max-player-speed-off", "--max_player_speed_off", dest="max_player_speed_off", type=float)
    parser.add_argument("--max-player-speed-def", "--max_player_speed_def", dest="max_player_speed_def", type=float)
    parser.add_argument("--lane-power", "--lane_power", dest="lane_power", type=float, default=PC_XPASS_DEFAULT_LANE_POWER)
    parser.add_argument("--lane-inflection-point", "--lane_inflection_point", dest="lane_inflection_point", type=float, default=PC_XPASS_DEFAULT_LANE_INFLECTION_POINT)
    parser.add_argument("--control-power", "--control_power", dest="control_power", type=float, default=PC_XPASS_DEFAULT_CONTROL_POWER)
    parser.add_argument("--control-inflection-point", "--control_inflection_point", dest="control_inflection_point", type=float, default=PC_XPASS_DEFAULT_CONTROL_INFLECTION_POINT)
    parser.add_argument(
        "--endpoint-normalization", "--endpoint_normalization",
        choices=["normal", "normal-one", "subtract", "subtract-one"],
        default=PC_XPASS_DEFAULT_ENDPOINT_NORMALIZATION,
    )
    parser.add_argument("--boost-def-endpoint-control", "--boost_def_endpoint_control", dest="boost_def_endpoint_control", type=float, default=PC_XPASS_DEFAULT_BOOST_DEF_ENDPOINT_CONTROL)
    parser.add_argument("--use-position-discount", "--use_position_discount", dest="use_position_discount", type=parse_bool_text, default=PC_XPASS_DEFAULT_USE_POSITION_DISCOUNT)
    parser.add_argument("--position-discount-power", "--position_discount_power", dest="position_discount_power", type=float, default=PC_XPASS_DEFAULT_POSITION_DISCOUNT_POWER)
    parser.add_argument("--position-discount-distance", "--position_discount_distance", dest="position_discount_distance", type=float, default=PC_XPASS_DEFAULT_POSITION_DISCOUNT_DISTANCE)
    parser.add_argument("--num-workers", "--physical-num-workers", dest="num_workers", default="auto")
    parser.add_argument("--max-auto-workers", type=int, default=PHYSICAL_DEFAULT_MAX_AUTO_WORKERS)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    parser.add_argument(
        "--runtime-row-window",
        type=int,
        default=None,
        help="Relocated graph rows to buffer per parallel pc-xPass prewarm; default: physical_batch_size * num_workers * 2.",
    )
    parser.add_argument("--worker-thread-limit", "--physical-worker-thread-limit", dest="worker_thread_limit", type=int, default=1)
    teammate_group = parser.add_mutually_exclusive_group()
    teammate_group.add_argument("--ignore-teammates", dest="consider_teammates", action="store_false")
    teammate_group.add_argument("--consider-teammates", dest="consider_teammates", action="store_true")
    parser.add_argument("--ignore-teammates-lane-survival", "--ignore_teammates_lane_survival", dest="ignore_teammates_lane_survival", action="store_true")
    parser.add_argument("--ignore-teammates-control", "--ignore_teammates_control", dest="ignore_teammates_control", action="store_true")
    parser.add_argument("--no-max", "--no_max", dest="export_max", action="store_false")
    parser.add_argument("--no-topmean", "--no_topmean", dest="export_topmean", action="store_false")
    parser.set_defaults(
        consider_teammates=True,
        ignore_teammates_lane_survival=False,
        ignore_teammates_control=False,
        export_max=True,
        export_topmean=True,
        export_noise_kernel=False,
        pc_xpass=True,
        dry_run=False,
    )
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.time_tolerance < 0:
        parser.error("--time-tolerance must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.runtime_row_window is not None and args.runtime_row_window < 1:
        parser.error("--runtime-row-window must be positive")
    requested_ids = [str(value).strip() for value in (args.situation_id or [])]
    if any(not value for value in requested_ids):
        parser.error("--situation-id must not be empty")
    if "all" in requested_ids and requested_ids != ["all"]:
        parser.error("--situation-id all cannot be combined with explicit situation ids")
    args.situation_id = requested_ids or None
    if not (0 < args.physical_eps < 0.5):
        parser.error("--physical-eps must be between 0 and 0.5")
    for name in ["max_speed", "min_speed", "speed_step", "radial_gridsize", "angle_step"]:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    if args.max_speed < args.min_speed:
        parser.error("--max-speed must be at least --min-speed")
    if not math.isfinite(float(args.refine_angle_radius)) or args.refine_angle_radius < 0:
        parser.error("--refine-angle-radius must be a non-negative finite number")
    for name in ["sigma_angle", "sigma_speed", "sigma_distance"]:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    for name in ["top_n", "coarse_n_angles", "refine_top_k_angles", "physical_batch_size", "worker_thread_limit", "max_auto_workers"]:
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.top_n_values is not None and any(int(value) < 1 for value in args.top_n_values):
        parser.error("--top-n-values must contain only positive integers")
    for name in ["lane_power", "lane_inflection_point", "control_power", "control_inflection_point"]:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    if not math.isfinite(float(args.boost_def_endpoint_control)) or args.boost_def_endpoint_control < 0:
        parser.error("--boost-def-endpoint-control must be a non-negative finite number")
    for name in ["position_discount_power", "position_discount_distance", "dist_pass_div", "max_player_speed"]:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    for name in ["dist_pass_min", "dist_pass_max"]:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be a non-negative finite number")
    if args.dist_pass_min > args.dist_pass_max:
        parser.error("--dist-pass-min must be less than or equal to --dist-pass-max")
    for name in ["max_player_speed_off", "max_player_speed_def"]:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    try:
        resolve_physical_num_workers(args.num_workers, max_auto_workers=args.max_auto_workers)
    except ValueError as exc:
        parser.error(str(exc))
    raw_reaction = args.reaction_time
    if isinstance(raw_reaction, str) and raw_reaction.lower() == PC_XPASS_REACTION_TIME_MODE_DIST_PASS:
        args.reaction_time_mode = PC_XPASS_REACTION_TIME_MODE_DIST_PASS
        args.reaction_time = None
    else:
        try:
            args.reaction_time = float(raw_reaction)
        except (TypeError, ValueError):
            parser.error("--reaction-time must be a non-negative number or 'dist_pass'")
        if not math.isfinite(args.reaction_time) or args.reaction_time < 0:
            parser.error("--reaction-time must be a non-negative number or 'dist_pass'")
        args.reaction_time_mode = PC_XPASS_REACTION_TIME_MODE_FIXED
    if args.x_pass_version.startswith("top"):
        try:
            requested_top_n = int(args.x_pass_version[3:])
        except ValueError:
            parser.error("--xpass-version must be max or top<N>")
        args.top_n_values = sorted({*(args.top_n_values or []), requested_top_n})
    elif args.x_pass_version != "max":
        parser.error("--xpass-version must be max or top<N>")
    if not args.export_max and not args.export_topmean:
        parser.error("At least one pc-xPass metric must be enabled")
    if args.x_pass_version == "max" and not args.export_max:
        parser.error("--xpass-version max cannot be combined with --no-max")
    if args.x_pass_version.startswith("top") and not args.export_topmean:
        parser.error("--xpass-version top<N> cannot be combined with --no-topmean")
    if args.v4_power is not None:
        if not math.isfinite(args.v4_power) or args.v4_power <= 0:
            parser.error("--v4-power must be positive")
        if args.xpass_weight != "v4":
            parser.error("--v4-power is only valid with --xpass-weight v4")
    if args.v4_zero is not None:
        if not math.isfinite(args.v4_zero) or args.v4_zero <= 0:
            parser.error("--v4-zero must be positive")
        if args.xpass_weight != "v4":
            parser.error("--v4-zero is only valid with --xpass-weight v4")
    if args.v4_discount is not None and args.xpass_weight != "v4":
        parser.error("--discount is only valid with --xpass-weight v4")
    if args.v4_discount is None:
        args.v4_discount = True


def resolve_target_frame(situation_tracking: pd.DataFrame, pass_moment: float, tolerance: float) -> dict[str, float | int]:
    receipts = pd.to_numeric(situation_tracking["BallReceipt"], errors="coerce").dropna().unique()
    if len(receipts) != 1:
        raise ValueError(f"expected one BallReceipt value, found {receipts.tolist()}")
    receipt = float(receipts[0])
    times = np.sort(pd.to_numeric(situation_tracking["abs_time"], errors="coerce").dropna().unique())
    candidates = pd.DataFrame({"abs_time": times})
    candidates["time_norm"] = candidates["abs_time"] - receipt
    candidates["time_diff"] = (candidates["time_norm"] - float(pass_moment)).abs()
    nearest = candidates.sort_values(["time_diff", "time_norm"], kind="mergesort").iloc[0]
    difference = float(nearest["time_diff"])
    if difference > tolerance:
        raise LookupError(
            f"nearest Hawkeye frame differs by {difference:.6f}s (tolerance={tolerance:.6f}s)"
        )
    # frame ids are assigned from the sorted unique frame-key order.
    key_rows = situation_tracking[["game_id", "half", "abs_time"]].drop_duplicates().sort_values(
        ["game_id", "half", "abs_time"]
    ).reset_index(drop=True)
    frame_matches = key_rows.index[np.isclose(key_rows["abs_time"], float(nearest["abs_time"]), atol=1e-9)]
    if len(frame_matches) != 1:
        raise ValueError(f"could not resolve unique frame id for abs_time={nearest['abs_time']}")
    return {
        "frame_id": int(frame_matches[0]),
        "abs_time": float(nearest["abs_time"]),
        "time_norm": float(nearest["time_norm"]),
        "time_diff": difference,
    }


def geometry_hash(
    action_id: str,
    selection_row_id: int,
    frame_id: int,
    position_x: float,
    position_y: float,
    adjusted_x: float,
    adjusted_y: float,
    situation_tracking: pd.DataFrame | None = None,
    situation_ball: pd.DataFrame | None = None,
) -> str:
    payload = {
        "action_id": str(action_id),
        "selection_row_id": int(selection_row_id),
        "frame_id": int(frame_id),
        "PositionX": round(float(position_x), 9),
        "PositionY": round(float(position_y), 9),
        "adjusted_x": round(float(adjusted_x), 9),
        "adjusted_y": round(float(adjusted_y), 9),
    }
    for name, table in [("tracking", situation_tracking), ("ball", situation_ball)]:
        if table is None:
            continue
        normalized = table.copy()
        normalized = normalized.reindex(sorted(normalized.columns), axis=1)
        sort_columns = [
            column
            for column in ["game_id", "half", "abs_time", "team", "uefa_player_id", "PlayerID"]
            if column in normalized.columns
        ]
        if sort_columns:
            normalized = normalized.sort_values(sort_columns, kind="mergesort", na_position="last")
        serialized = normalized.to_csv(index=False, float_format="%.12g", lineterminator="\n")
        payload[f"{name}_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compatibility_warnings(model_ids: dict[str, str], model_specs: dict[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task, model in model_specs.items():
        model_args = getattr(model, "args", {})
        signature = extract_model_feature_signature(model_args)
        incompatible: list[str] = []
        for name in ["poss_vel_aware", "poss_rel_vel_aware", "accel_aware"]:
            if bool(signature.get(name, False)):
                incompatible.append(name)
        if signature.get("v_edge_feature_mode") == "all":
            incompatible.append("v_edge_feature_mode=all")
        if signature.get("relative_speed_edge_feature_mode") == "all":
            incompatible.append("relative_speed_edge_feature_mode=all")
        if incompatible:
            record = {
                "task": task,
                "model_id": model_ids.get(task),
                "incompatible_features": incompatible,
                "message": "model may use possessor velocity/acceleration features in location inference",
            }
            records.append(record)
            warnings.warn(f"{record['model_id'] or task}: {record['message']}: {', '.join(incompatible)}")
    return records


def validate_adjusted_possessor_bounds(adjusted_tracking: pd.DataFrame, player_id: int) -> None:
    carrier_rows = adjusted_tracking.loc[adjusted_tracking["uefa_player_id"].eq(int(player_id))]
    if carrier_rows.empty:
        raise ValueError(f"adjusted tracking has no rows for possessor PlayerID={int(player_id)}")
    half_length = float(config.FIELD_SIZE[0]) / 2.0
    half_width = float(config.FIELD_SIZE[1]) / 2.0
    outside = carrier_rows["centroid_x"].abs().gt(half_length) | carrier_rows["centroid_y"].abs().gt(half_width)
    if outside.any():
        raise ArithmeticError("adjusted possessor position is outside pitch bounds")


def filter_situation_to_frame(situation, attacking_rows: pd.DataFrame, frame_id: int) -> pd.DataFrame:
    kept_graphs = []
    kept_labels = []
    for graph, label in zip(situation.graph_features_0, situation.labels):
        if int(label[LABEL_INDEX["action_index"]].item()) == int(frame_id):
            kept_graphs.append(graph)
            kept_labels.append(label)
    if not kept_graphs:
        raise LookupError(f"resolved Hawkeye frame {frame_id} has no valid graph")
    situation.graph_features_0 = kept_graphs
    situation.labels = torch.stack(kept_labels, axis=0)
    situation.graph_features_by_dir["action_graphs"] = kept_graphs
    if not situation.actions.empty:
        situation.actions = situation.actions.loc[situation.actions.index.astype(int) == int(frame_id)].copy()
    return attacking_rows.loc[attacking_rows["frame_id"].astype("Int64").eq(int(frame_id))].copy()


def _missing_record(row: pd.Series, status: str, reason: str) -> dict[str, object]:
    record = {column: row.get(column, pd.NA) for column in MISSING_REPORT_COLUMNS}
    record["loc_status"] = status
    record["loc_missing_reason"] = reason
    return record


def _configure_models(args: argparse.Namespace):
    resolved_ids, shared_context, bundle = resolve_model_selection(
        required_tasks=["action_intent", "pass_intent", "pass_success", "outcome_scoring", "outcome_conceding"],
        bundle_id=args.bundle_id,
        explicit_model_ids={
            "action_intent": args.action_intent_model_id,
            "pass_intent": args.pass_intent_model_id,
            "pass_success": args.pass_success_model_id,
            "outcome_scoring": args.outcome_scoring_model_id,
            "outcome_conceding": args.outcome_conceding_model_id,
        },
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )
    pass_height_id = resolve_optional_model_id("pass_height", args.pass_height_model_id, bundle)
    if pass_height_id:
        resolved_ids["pass_height"] = pass_height_id
    device = args.device if torch.cuda.is_available() else "cpu"
    specs = load_hawkeye_models(
        action_intent_model_id=resolved_ids["action_intent"],
        pass_intent_model_id=resolved_ids["pass_intent"],
        pass_success_model_id=resolved_ids["pass_success"],
        pass_height_model_id=resolved_ids.get("pass_height"),
        outcome_scoring_model_id=resolved_ids["outcome_scoring"],
        outcome_conceding_model_id=resolved_ids["outcome_conceding"],
        device=device,
    )
    schema = validate_model_graph_schemas(specs)
    pass_model = specs["pass_success"]
    pass_model.args.update(
        {
            "inference_use_physical_xpass": True,
            "pc_xpass": True,
            "x_pass_version": args.x_pass_version,
            "xpass_weight": args.xpass_weight,
            "v4_discount": bool(args.v4_discount),
            "ball_z_limit": args.ball_z_limit,
            "physical_runtime_cache_disabled": False,
            "physical_runtime_cache_refresh": False,
            "physical_runtime_cache_read_only": True,
            "physical_cache_dir": str(args.pc_xpass_cache_dir),
            "physical_num_workers": args.num_workers,
            "physical_worker_thread_limit": int(args.worker_thread_limit),
            "physical_batch_size": int(args.physical_batch_size),
        }
    )
    if args.v4_power is not None:
        pass_model.args["v4_power"] = float(args.v4_power)
    if args.v4_zero is not None:
        pass_model.args["v4_zero"] = float(args.v4_zero)
    configure_lane_survival_runtime_cache(specs, args.lane_survival_cache_dir or args.pc_xpass_cache_dir)
    args._runtime_graph_schema = schema
    args._pass_height_model = specs.get("pass_height")
    args._pass_height_model_record = (
        get_model_provenance(resolved_ids["pass_height"]) if "pass_height" in resolved_ids else None
    )
    args.pass_height_device = device
    args.pass_height_model_id = resolved_ids.get("pass_height")
    return resolved_ids, shared_context, specs, schema, device


def select_location_rows(
    selection: pd.DataFrame,
    *,
    situation_ids: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Apply CLI row selection while retaining the source CSV order."""
    if limit is not None:
        return selection.iloc[: int(limit)].copy()
    if not situation_ids:
        return selection.copy()

    action_values = selection["action_id"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    first_indices: dict[str, int] = {}
    for index, value in action_values.items():
        if value and value not in first_indices:
            first_indices[value] = index

    if situation_ids == ["all"]:
        return selection.loc[list(first_indices.values())].copy()

    missing = [value for value in situation_ids if value not in first_indices]
    if missing:
        raise ValueError("Requested --situation-id values were not found in action_id: " + ", ".join(missing))
    # dict.fromkeys prevents repeated CLI values from running the same action twice.
    indices = [first_indices[value] for value in dict.fromkeys(situation_ids)]
    return selection.loc[indices].copy()


def build_location_target(
    row: pd.Series,
    tracking: pd.DataFrame,
    ball: pd.DataFrame,
    graph_schema: dict[str, object],
    time_tolerance: float,
) -> dict[str, object]:
    """Build one relocated target graph and the diagnostics needed for export/cache identity."""
    row_id = int(row["selection_row_id"])
    if pd.isna(row["action_id"]):
        raise ValueError("action_id is missing")
    action_id = str(row["action_id"]).strip()
    pass_moment = float(row["pass_moment"])
    position_x = float(row["PositionX"])
    position_y = float(row["PositionY"])
    selected_player = float(row["SelectedPlayer"])
    if not action_id or not all(math.isfinite(value) for value in [pass_moment, position_x, position_y]):
        raise ValueError("invalid action_id, pass_moment, PositionX, or PositionY")
    if not math.isfinite(selected_player) or not selected_player.is_integer():
        raise ValueError("SelectedPlayer is missing or invalid")

    situation_tracking = tracking.loc[tracking["id"].eq(action_id)].copy()
    if situation_tracking.empty:
        raise LookupError(f"Hawkeye situation not found for action_id={action_id}")
    frame = resolve_target_frame(situation_tracking, pass_moment, float(time_tolerance))
    adjusted_tracking, adjusted_info = apply_hawkeye_possessor_offset(
        situation_tracking,
        offset_x=position_x / 100.0,
        offset_y=-position_y / 100.0,
    )
    try:
        validate_adjusted_possessor_bounds(adjusted_tracking, int(adjusted_info["PlayerID"]))
    except ArithmeticError as exc:
        raise ArithmeticError(
            f"{exc}; anchor=({adjusted_info['adjusted_x']:.6f}, {adjusted_info['adjusted_y']:.6f})"
        ) from exc

    situation_frame_keys = adjusted_tracking[["game_id", "half", "abs_time"]].drop_duplicates()
    situation_ball = ball.merge(situation_frame_keys, on=["game_id", "half", "abs_time"], how="inner")
    state_hash = geometry_hash(
        action_id,
        row_id,
        int(frame["frame_id"]),
        position_x,
        position_y,
        float(adjusted_info["adjusted_x"]),
        float(adjusted_info["adjusted_y"]),
        situation_tracking=adjusted_tracking,
        situation_ball=situation_ball,
    )
    synthetic_id = f"{action_id}__loc__{row_id}__{state_hash}"
    situation, attacking_rows, _stats = build_hawkeye_situation(
        adjusted_tracking,
        ball,
        freeze_ballreceipt=True,
        add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
        add_relative_speed_edge_features=bool(graph_schema.get("add_relative_speed_edge_features", False)),
        align_frozen_ball_to_possessor=True,
    )
    situation.match_id = synthetic_id
    attacking_rows = filter_situation_to_frame(situation, attacking_rows, int(frame["frame_id"]))
    if attacking_rows.empty:
        raise LookupError(f"target frame {frame['frame_id']} has no attacking-player rows")
    return {
        "row": row,
        "row_id": row_id,
        "action_id": action_id,
        "pass_moment": pass_moment,
        "position_x": position_x,
        "position_y": position_y,
        "selected_player": int(selected_player),
        "frame": frame,
        "adjusted_info": adjusted_info,
        "geometry_hash": state_hash,
        "synthetic_id": synthetic_id,
        "situation": situation,
        "attacking_rows": attacking_rows,
        "cache_items": runtime_cache_items_from_graphs(synthetic_id, situation.graph_features_0, situation.labels),
    }


def _cache_stats_have_misses(stats: dict[str, object] | None) -> bool:
    return bool(stats) and int(stats.get("cache_misses", 0) or 0) > 0


def _cache_stats_have_unusable_rows(stats: dict[str, object] | None) -> bool:
    return bool(stats) and int(stats.get("skipped_all_nan", 0) or 0) > 0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input selection file not found: {input_path}")
    selection = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in selection.columns]
    if missing_columns:
        raise ValueError(f"Input selection data is missing required columns: {', '.join(missing_columns)}")
    if selection["selection_row_id"].duplicated().any():
        raise ValueError("selection_row_id must be unique")
    selected_selection = select_location_rows(
        selection,
        situation_ids=args.situation_id,
        limit=args.limit,
    )

    run_id = args.run_id or generate_run_id("hawkeye_loc_component")
    output_parent = Path(args.output_dir) if args.output_dir else HAWKEYE_LOC_COMPONENT_RUNS_DIR
    output_dir = output_parent / run_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Component run directory already exists; use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    resolved_ids, shared_context, model_specs, graph_schema, device = _configure_models(args)
    model_warning_records = compatibility_warnings(resolved_ids, model_specs)

    exports: list[pd.DataFrame] = []
    missing_records: list[dict[str, object]] = []
    processed_rows: list[int] = []
    pc_stats: dict[str, object] = {}
    prepared_rows: list[pd.Series] = []
    cache_dir = Path(args.pc_xpass_cache_dir)
    runtime_row_window = resolve_runtime_row_window(args, "runtime_row_window")

    batch_errors: list[dict[str, object]] = []
    pending_targets: list[dict[str, object]] = []

    def flush_pending_targets() -> None:
        if not pending_targets:
            return
        window = list(pending_targets)
        pending_targets.clear()
        items = [item for unit in window for item in unit["cache_items"]]
        if not items:
            return
        try:
            stats = prewarm_runtime_items(
                items,
                cache_dir=cache_dir,
                args=args,
                progress_desc=(
                    f"hawkeye_loc cache rows {window[0]['row_id']}-{window[-1]['row_id']}"
                ),
            )
            merge_stats(pc_stats, stats)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            batch_errors.append(
                {
                    "selection_row_ids": [int(unit["row_id"]) for unit in window],
                    "reason": message,
                }
            )
            print(f"pc-xPass batch failed; auditing rows individually: {message}")

    # Phase 1: construct target graphs in bounded windows and prewarm pc-xPass in parallel.
    for _, row in selected_selection.iterrows():
        row_id = int(row["selection_row_id"])
        if int(pd.to_numeric(pd.Series([row.get("loc_info_missing")]), errors="coerce").fillna(1).iloc[0]) == 1:
            missing_records.append(
                _missing_record(row, str(row.get("loc_status", "location_missing")), str(row.get("loc_missing_reason", "location missing")))
            )
            continue
        try:
            target = build_location_target(row, tracking, ball, graph_schema, float(args.time_tolerance))
            prepared_rows.append(row.copy())
            pending_targets.append(target)
            if len(pending_targets) >= runtime_row_window:
                flush_pending_targets()
        except LookupError as exc:
            status = "hawkeye_situation_not_found" if "situation not found" in str(exc) else "hawkeye_frame_not_found"
            missing_records.append(_missing_record(row, status, str(exc)))
        except ArithmeticError as exc:
            missing_records.append(_missing_record(row, "adjusted_position_out_of_bounds", str(exc)))
        except Exception as exc:
            missing_records.append(_missing_record(row, "geometry_failed", f"{type(exc).__name__}: {exc}"))
            print(f"SKIP selection_row_id={row_id}: {type(exc).__name__}: {exc}")
    flush_pending_targets()

    # Phase 2: a single-worker prewarm is both a cache validity audit and a retry for misses.
    audit_args = argparse.Namespace(**vars(args))
    audit_args.num_workers = 1
    cache_audit = {"checked": 0, "initially_missing": 0, "recovered": 0, "failed": 0}
    cache_ready_rows: list[pd.Series] = []
    for row in prepared_rows:
        row_id = int(row["selection_row_id"])
        cache_audit["checked"] += 1
        was_missing = False
        try:
            unit = build_location_target(row, tracking, ball, graph_schema, float(args.time_tolerance))
            stats = prewarm_runtime_items(
                unit["cache_items"],
                cache_dir=cache_dir,
                args=audit_args,
                progress_desc=f"hawkeye_loc cache audit row {row_id}",
            )
            if _cache_stats_have_unusable_rows(stats):
                raise RuntimeError("pc-xPass cache generation skipped an all-NaN target row")
            if _cache_stats_have_misses(stats):
                was_missing = True
                cache_audit["initially_missing"] += 1
                merge_stats(pc_stats, stats)
                verification = prewarm_runtime_items(
                    unit["cache_items"],
                    cache_dir=cache_dir,
                    args=audit_args,
                    progress_desc=f"hawkeye_loc cache verify row {row_id}",
                )
                if _cache_stats_have_misses(verification) or _cache_stats_have_unusable_rows(verification):
                    raise RuntimeError("pc-xPass cache remained missing or invalid after individual retry")
                cache_audit["recovered"] += 1
            cache_ready_rows.append(row)
        except (LookupError, ArithmeticError, ValueError) as exc:
            cache_audit["failed"] += 1
            missing_records.append(_missing_record(row, "geometry_failed", f"{type(exc).__name__}: {exc}"))
            print(f"SKIP selection_row_id={row_id}: cache audit graph rebuild failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            if not was_missing:
                cache_audit["initially_missing"] += 1
            cache_audit["failed"] += 1
            missing_records.append(_missing_record(row, "pc_xpass_failed", f"{type(exc).__name__}: {exc}"))
            print(f"SKIP selection_row_id={row_id}: pc-xPass audit/retry failed: {type(exc).__name__}: {exc}")

    # Phase 3: rebuild each graph and infer sequentially from read-only caches.
    for row in cache_ready_rows:
        row_id = int(row["selection_row_id"])
        try:
            unit = build_location_target(row, tracking, ball, graph_schema, float(args.time_tolerance))
            situation = unit["situation"]
            attacking_rows = unit["attacking_rows"]
            components = infer_hawkeye_components(situation, model_specs, device=device)
            export = build_hawkeye_export(attacking_rows, situation, components)
            selected_export = export.loc[
                pd.to_numeric(export["uefa_player_id"], errors="coerce").eq(int(unit["selected_player"]))
            ]
            if len(selected_export) != 1:
                raise RuntimeError(
                    f"expected one target-frame row for SelectedPlayer={int(unit['selected_player'])}, "
                    f"found {len(selected_export)}"
                )
            missing_components = [column for column in REQUIRED_SCORE_COMPONENTS if pd.isna(selected_export.iloc[0][column])]
            if missing_components:
                raise RuntimeError("selected-player component values are missing: " + ", ".join(missing_components))
            frame = unit["frame"]
            adjusted_info = unit["adjusted_info"]
            export["selection_row_id"] = row_id
            export["original_action_id"] = unit["action_id"]
            export["SelectedPlayer"] = row["SelectedPlayer"]
            export["requested_pass_moment"] = unit["pass_moment"]
            export["resolved_time_norm"] = float(frame["time_norm"])
            export["hawkeye_time_diff"] = float(frame["time_diff"])
            export["PositionX"] = unit["position_x"]
            export["PositionY"] = unit["position_y"]
            export["adjusted_possessor_x"] = float(adjusted_info["adjusted_x"])
            export["adjusted_possessor_y"] = float(adjusted_info["adjusted_y"])
            export["loc_situation_id"] = unit["synthetic_id"]
            export["geometry_hash"] = unit["geometry_hash"]
            exports.append(export)
            processed_rows.append(row_id)
        except Exception as exc:
            missing_records.append(_missing_record(row, "inference_failed", f"{type(exc).__name__}: {exc}"))
            print(f"SKIP selection_row_id={row_id}: {type(exc).__name__}: {exc}")

    export_columns = [
        "selection_row_id", "original_action_id", "SelectedPlayer", "requested_pass_moment",
        "resolved_time_norm", "hawkeye_time_diff", "PositionX", "PositionY",
        "adjusted_possessor_x", "adjusted_possessor_y", "loc_situation_id", "geometry_hash",
        "id", "uefa_player_id", "PlayerID", "abs_time", *COMPONENT_COLUMNS,
    ]
    hawkeye_table = pd.concat(exports, ignore_index=True, sort=False) if exports else pd.DataFrame(columns=export_columns)
    missing_report = pd.DataFrame.from_records(missing_records, columns=MISSING_REPORT_COLUMNS)
    hawkeye_table.to_parquet(output_dir / "hawkeye_data.parquet", index=False)
    hawkeye_table.to_csv(output_dir / "hawkeye_data.csv", index=False)
    missing_report.to_csv(output_dir / "missing_data.csv", index=False)

    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_ids.items()}
    metadata = {
        "run_id": run_id,
        "run_type": "hawkeye_loc_component",
        "inference_mode": "loc",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_path.resolve()),
        "tracking_csv": str(Path(args.tracking_csv).resolve()),
        "ball_csv": str(Path(args.ball_csv).resolve()),
        "time_tolerance": float(args.time_tolerance),
        "position_units": "centimetres",
        "coordinate_transform": "centroid_x + PositionX/100; centroid_y - PositionY/100",
        "pc_xpass_cache_dir": str(Path(args.pc_xpass_cache_dir).resolve()),
        "pc_xpass_stats": pc_stats,
        "pc_xpass_batch_errors": batch_errors,
        "pc_xpass_cache_audit": cache_audit,
        "runtime_row_window": int(runtime_row_window),
        "num_workers": args.num_workers,
        "resolved_num_workers": int(resolve_physical_num_workers(args.num_workers, max_auto_workers=args.max_auto_workers)),
        "selection": {
            "requested_situation_ids": list(args.situation_id or []),
            "limit": args.limit,
            "input_rows": int(len(selection)),
            "selected_rows": int(len(selected_selection)),
            "selected_selection_row_ids": [int(value) for value in selected_selection["selection_row_id"].tolist()],
        },
        "processed_selection_row_ids": processed_rows,
        "missing_rows": int(len(missing_report)),
        "models": resolved_ids,
        "model_records": model_records,
        "model_compatibility_warnings": model_warning_records,
        "graph_schema": graph_schema,
        "shared_model_context": shared_context,
        "status": "completed" if missing_report.empty else "completed_with_errors",
    }
    write_run_metadata(output_dir, metadata)
    if output_parent.resolve() == HAWKEYE_LOC_COMPONENT_RUNS_DIR.resolve():
        write_latest_run("hawkeye_loc_component", run_id)
    # Keep cache metadata in the same schema as generate_physical_xpass.py.
    write_runtime_dataset_metadata(
        "hawkeye_loc",
        Path(args.pc_xpass_cache_dir),
        args,
        stats=pc_stats,
        source_inputs={
            "component_run_id": run_id,
            "input_file": str(input_path.resolve()),
            "requested_situation_ids": list(args.situation_id or []),
            "limit": args.limit,
            "selected_selection_row_ids": [int(value) for value in selected_selection["selection_row_id"].tolist()],
            "runtime_row_window": int(runtime_row_window),
        },
        skipped={str(record["selection_row_id"]): record["loc_missing_reason"] for record in missing_records},
    )
    print(f"Hawkeye location component run id: {run_id}")
    print(f"Processed selection rows: {len(processed_rows)}; missing: {len(missing_report)}")
    print(f"Saved component run to: {output_dir}")


if __name__ == "__main__":
    main()
