from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch

from datatools.graph_feature import construct_graph_features
from datatools.match import Match
from inference import PhysicalXPassNoUsableRowsError, inference_gnn, load_success_intent_labels
from models.utils import (
    get_model_provenance,
    infer_feature_graph_schema,
    load_model,
    resolve_model_selection,
    resolve_runtime_return_type,
    resolve_runtime_feature_run_context,
    validate_feature_graph_schema,
    validate_model_graph_schemas,
)
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    format_physical_xpass_cache_summary,
    inference_uses_physical_xpass,
    model_uses_physical_xpass,
    parse_physical_xpass_bool,
    physical_xpass_inference_lookup_config,
    physical_xpass_source,
    resolve_physical_num_workers,
    summarize_physical_xpass_cache_usage,
)
from project_config import (
    DATA_ROOT,
    DEFAULT_INTENDED_RECEIVER_MODE,
    INTENDED_RECEIVER_MODES,
    SPORTEC_COMPONENT_RUNS_DIR,
    generate_run_id,
    get_action_graph_dir,
    get_pc_xpass_dir,
    get_post_action_graph_dir,
    get_resolved_action_path,
    get_runtime_physical_xpass_dir,
    get_success_intent_graph_dir,
    load_base_splits,
    write_latest_run,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--match-id", action="append", help="Restrict inference to one or more match ids.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-id")
    parser.add_argument("--feature-run-id", help="Runtime feature run used to load graphs/resolved actions.")
    parser.add_argument("--intended-receiver-mode", choices=INTENDED_RECEIVER_MODES, help="Runtime resolved-action mode.")
    parser.add_argument("--return-type", "--return_type", dest="return_type", help="Runtime return type used for label construction.")
    parser.add_argument("--run-id")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--pass-height-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--use-physical-xpass", "--use_physical_xpass", dest="use_physical_xpass", action="store_true", help="Blend pass-success inference with physical xPass.")
    parser.add_argument("--pc-xpass", "--pc_xpass", dest="pc_xpass", action="store_true", help="Use pc-xPass cache values for physical xPass inference blending.")
    parser.add_argument("--xpass-version", "--x-pass-version", "--x_pass_version", dest="x_pass_version", default="top10", help="Cached xPass version to use: max, noise-kernel, or top<N> such as top10/top25/top50.")
    parser.add_argument("--xpass-weight", "--xpass_weight", dest="xpass_weight", choices=["v1", "v2", "v3", "v4"], default="v3", help="Physical xPass/model blend weighting version.")
    parser.add_argument("--v4-power", dest="v4_power", type=float, default=None, help="Power for --xpass-weight v4. Default: 2.0.")
    parser.add_argument("--v4-zero", dest="v4_zero", type=float, default=None, help="Zero point for --xpass-weight v4 distance discount. Default: 0.8.")
    parser.add_argument("--discount", dest="v4_discount", type=parse_physical_xpass_bool, default=None, help="For --xpass-weight v4, enable/disable the distance cosine discount. Default: true.")
    parser.add_argument("--ball-z-limit", dest="ball_z_limit", default="none", help="If set to a float, use 100%% pass-success model weight when cached ball_z exceeds this value. Use 'none' to disable.")
    parser.add_argument("--physical-cache-dir", help="Sportec runtime physical xPass cache override.")
    parser.add_argument("--no-physical-cache", action="store_true", help="Disable runtime physical xPass cache.")
    parser.add_argument("--refresh-physical-cache", action="store_true", help="Deprecated during inference; run scripts/generate_physical_xpass.py to refresh/fill caches.")
    parser.add_argument("--physical-num-workers", "--num-workers", dest="physical_num_workers", default="auto")
    parser.add_argument("--physical-worker-thread-limit", "--worker-thread-limit", dest="physical_worker_thread_limit", type=int, default=1)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        resolve_physical_num_workers(args.physical_num_workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.physical_worker_thread_limit < 1:
        parser.error("--physical-worker-thread-limit must be positive.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
    if args.v4_power is not None:
        if not math.isfinite(args.v4_power) or args.v4_power <= 0:
            parser.error("--v4-power must be positive.")
        if args.xpass_weight != "v4":
            parser.error("--v4-power is only valid with --xpass-weight v4.")
    if args.v4_zero is not None:
        if not math.isfinite(args.v4_zero) or args.v4_zero <= 0:
            parser.error("--v4-zero must be positive.")
        if args.xpass_weight != "v4":
            parser.error("--v4-zero is only valid with --xpass-weight v4.")
    if args.v4_discount is not None and args.xpass_weight != "v4":
        parser.error("--discount is only valid with --xpass-weight v4.")
    if args.v4_discount is None:
        args.v4_discount = True
    return args


def summarize_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def count_valid_graphs(graphs: list[object]) -> int:
    if not isinstance(graphs, list):
        raise TypeError("Cached graph artifact is not a list of graphs.")
    return sum(graph is not None for graph in graphs)


def load_match(
    match_id: str,
    intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE,
    return_type: str | None = None,
    feature_root: Path | None = None,
    add_v_edge_features: bool = False,
) -> Match:
    feature_root = Path(feature_root) if feature_root is not None else None
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineups = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()

    match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
    match.runtime_feature_root = feature_root
    resolved_action_path = get_resolved_action_path(
        match_id,
        intended_receiver_mode=intended_receiver_mode,
        root=feature_root,
    )
    if not resolved_action_path.exists():
        raise FileNotFoundError(
            f"Resolved actions not found at {resolved_action_path}. Run scripts/generate_relevant_features.py for this mode first."
        )
    resolved_actions = pd.read_parquet(resolved_action_path)
    match.labels = match.construct_labels(
        discount_xg=True,
        intended_receiver_mode=intended_receiver_mode,
        relabel_intended_receivers=False,
        resolved_actions=resolved_actions,
        return_type=return_type,
    )

    graph_path = get_action_graph_dir(feature_root) / f"{match_id}.pt"
    if graph_path.exists():
        try:
            match.graph_features_0 = torch.load(graph_path, weights_only=False)
        except Exception as exc:
            print(f"  Rebuilding cached action graphs after read failure: {summarize_exception(exc)}")
            match.graph_features_0 = construct_graph_features(
                match,
                extend=True,
                post_action=False,
                add_v_edge_features=add_v_edge_features,
            )
        else:
            if count_valid_graphs(match.graph_features_0) == 0:
                print("  Rebuilding cached action graphs because the cached file contains no usable graphs.")
                match.graph_features_0 = construct_graph_features(
                    match,
                    extend=True,
                    post_action=False,
                    add_v_edge_features=add_v_edge_features,
                )
    else:
        match.graph_features_0 = construct_graph_features(
            match,
            extend=True,
            post_action=False,
            add_v_edge_features=add_v_edge_features,
        )
    if count_valid_graphs(match.graph_features_0) == 0:
        raise ValueError("No usable action graphs are available for this match.")

    match.actions = match.label_post_actions(match.actions)
    post_graph_path = get_post_action_graph_dir(feature_root) / f"{match_id}.pt"
    if post_graph_path.exists():
        try:
            match.graph_features_1 = torch.load(post_graph_path, weights_only=False)
        except Exception as exc:
            print(f"  Rebuilding cached post-action graphs after read failure: {summarize_exception(exc)}")
            match.graph_features_1 = construct_graph_features(
                match,
                extend=True,
                post_action=True,
                add_v_edge_features=add_v_edge_features,
            )
        else:
            if count_valid_graphs(match.graph_features_1) == 0:
                print("  Rebuilding cached post-action graphs because the cached file contains no usable graphs.")
                match.graph_features_1 = construct_graph_features(
                    match,
                    extend=True,
                    post_action=True,
                    add_v_edge_features=add_v_edge_features,
                )
    else:
        match.graph_features_1 = construct_graph_features(
            match,
            extend=True,
            post_action=True,
            add_v_edge_features=add_v_edge_features,
        )
    if count_valid_graphs(match.graph_features_1) == 0:
        raise ValueError("No usable post-action graphs are available for this match.")

    return match


COMPONENT_IDENTIFIER_COLUMNS = ["stats_perform_match_id", "action_id", "original_event_id"]
FRAME_SCOPE_COLUMN = "frame_scope"
STATE_FRAME_ID_COLUMN = "state_frame_id"
FRAME_ID_SCOPE = "frame_id"
RECEIVE_FRAME_ID_SCOPE = "receive_frame_id"


def save_component_table(frame: pd.DataFrame, actions: pd.DataFrame, output_path: Path) -> None:
    missing_columns = [column for column in COMPONENT_IDENTIFIER_COLUMNS if column not in actions.columns]
    if missing_columns:
        raise ValueError(
            "Cannot export component table because match.actions is missing: "
            f"{', '.join(missing_columns)}"
        )
    if not frame.index.is_unique:
        raise ValueError("Cannot export component table because prediction rows have duplicate action indexes.")
    if not actions.index.is_unique:
        raise ValueError("Cannot export component table because match.actions has duplicate indexes.")

    missing_action_indexes = frame.index.difference(actions.index)
    if len(missing_action_indexes) > 0:
        sample = missing_action_indexes[:5].tolist()
        raise ValueError(
            "Cannot export component table because prediction action indexes are missing from match.actions: "
            f"{sample}"
        )

    identifiers = actions.loc[frame.index, COMPONENT_IDENTIFIER_COLUMNS].reset_index(drop=True)
    table = frame.reset_index(drop=True)
    duplicate_columns = [column for column in COMPONENT_IDENTIFIER_COLUMNS if column in table.columns]
    if duplicate_columns:
        raise ValueError(
            "Cannot export component table because prediction columns duplicate identifiers: "
            f"{', '.join(duplicate_columns)}"
        )

    table = pd.concat([identifiers, table], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)


def add_prediction_scope(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    frame_scope: str,
) -> pd.DataFrame:
    if frame_scope not in {FRAME_ID_SCOPE, RECEIVE_FRAME_ID_SCOPE}:
        raise ValueError(f"Unsupported frame scope: {frame_scope!r}")

    missing_columns = [
        column
        for column in COMPONENT_IDENTIFIER_COLUMNS + [frame_scope]
        if column not in actions.columns
    ]
    if missing_columns:
        raise ValueError(
            "Cannot export scoped component table because match.actions is missing: "
            f"{', '.join(missing_columns)}"
        )
    if not frame.index.is_unique:
        raise ValueError("Cannot export scoped component table because prediction rows have duplicate action indexes.")
    if not actions.index.is_unique:
        raise ValueError("Cannot export scoped component table because match.actions has duplicate indexes.")

    missing_action_indexes = frame.index.difference(actions.index)
    if len(missing_action_indexes) > 0:
        sample = missing_action_indexes[:5].tolist()
        raise ValueError(
            "Cannot export scoped component table because prediction action indexes are missing from match.actions: "
            f"{sample}"
        )

    identifiers = actions.loc[frame.index, COMPONENT_IDENTIFIER_COLUMNS].reset_index(drop=True)
    scoped = frame.reset_index(drop=True)
    duplicate_columns = [
        column
        for column in COMPONENT_IDENTIFIER_COLUMNS + [FRAME_SCOPE_COLUMN, STATE_FRAME_ID_COLUMN]
        if column in scoped.columns
    ]
    if duplicate_columns:
        raise ValueError(
            "Cannot export scoped component table because prediction columns duplicate identifiers: "
            f"{', '.join(duplicate_columns)}"
        )

    identifiers[FRAME_SCOPE_COLUMN] = frame_scope
    identifiers[STATE_FRAME_ID_COLUMN] = actions.loc[frame.index, frame_scope].reset_index(drop=True)
    return pd.concat([identifiers, scoped], axis=1)


def save_scoped_component_table(
    frame_predictions: pd.DataFrame,
    receive_predictions: pd.DataFrame | None,
    actions: pd.DataFrame,
    output_path: Path,
) -> None:
    scoped_frames = [add_prediction_scope(frame_predictions, actions, frame_scope=FRAME_ID_SCOPE)]
    if receive_predictions is not None:
        scoped_frames.append(
            add_prediction_scope(receive_predictions, actions, frame_scope=RECEIVE_FRAME_ID_SCOPE)
        )

    table = pd.concat(scoped_frames, ignore_index=True)
    duplicate_mask = table.duplicated(
        subset=COMPONENT_IDENTIFIER_COLUMNS + [FRAME_SCOPE_COLUMN],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(f"Scoped component table has duplicate action/scope rows for {output_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)


def save_component_table_with_optional_scope(
    frame_predictions: pd.DataFrame,
    receive_predictions: pd.DataFrame | None,
    actions: pd.DataFrame,
    output_path: Path,
) -> None:
    required_scope_columns = [FRAME_ID_SCOPE]
    if receive_predictions is not None:
        required_scope_columns.append(RECEIVE_FRAME_ID_SCOPE)
    if any(column not in actions.columns for column in required_scope_columns):
        save_component_table(frame_predictions, actions, output_path)
        return
    save_scoped_component_table(frame_predictions, receive_predictions, actions, output_path)


def save_match_component_tables(
    match_output_dir: Path,
    actions: pd.DataFrame,
    *,
    action_intent: pd.DataFrame,
    pass_intent: pd.DataFrame,
    pass_intent_receive: pd.DataFrame | None = None,
    pass_success: pd.DataFrame | None = None,
    pass_success_receive: pd.DataFrame | None = None,
    pass_height: pd.DataFrame | None = None,
    pass_height_receive: pd.DataFrame | None = None,
    scoring_success: pd.DataFrame | None = None,
    scoring_success_receive: pd.DataFrame | None = None,
    scoring_failure: pd.DataFrame | None = None,
    scoring_failure_receive: pd.DataFrame | None = None,
    conceding_success: pd.DataFrame | None = None,
    conceding_success_receive: pd.DataFrame | None = None,
    conceding_failure: pd.DataFrame | None = None,
    conceding_failure_receive: pd.DataFrame | None = None,
) -> None:
    save_component_table(action_intent, actions, match_output_dir / "action_intent.parquet")
    save_component_table_with_optional_scope(
        pass_intent,
        pass_intent_receive,
        actions,
        match_output_dir / "pass_intent.parquet",
    )
    if pass_success is not None:
        save_component_table_with_optional_scope(
            pass_success,
            pass_success_receive,
            actions,
            match_output_dir / "pass_success.parquet",
        )
    if pass_height is not None:
        save_component_table_with_optional_scope(
            pass_height,
            pass_height_receive,
            actions,
            match_output_dir / "pass_height.parquet",
        )
    save_component_table_with_optional_scope(
        scoring_success,
        scoring_success_receive,
        actions,
        match_output_dir / "outcome_scoring_success.parquet",
    )
    save_component_table_with_optional_scope(
        scoring_failure,
        scoring_failure_receive,
        actions,
        match_output_dir / "outcome_scoring_failure.parquet",
    )
    save_component_table_with_optional_scope(
        conceding_success,
        conceding_success_receive,
        actions,
        match_output_dir / "outcome_conceding_success.parquet",
    )
    save_component_table_with_optional_scope(
        conceding_failure,
        conceding_failure_receive,
        actions,
        match_output_dir / "outcome_conceding_failure.parquet",
    )


def resolve_optional_success_intent_model_id(
    args: argparse.Namespace,
    bundle: dict[str, object] | None,
) -> str | None:
    if args.success_intent_model_id:
        return str(args.success_intent_model_id)
    if bundle is None:
        return None
    bundle_model_ids = bundle.get("model_ids", {})
    if not isinstance(bundle_model_ids, dict):
        return None
    success_intent_model_id = bundle_model_ids.get("success_intent")
    return str(success_intent_model_id) if success_intent_model_id else None


def resolve_optional_model_id(task: str, explicit_model_id: str | None, bundle: dict[str, object] | None) -> str | None:
    if explicit_model_id:
        return str(explicit_model_id)
    if bundle is None:
        return None
    bundle_model_ids = bundle.get("model_ids", {})
    if not isinstance(bundle_model_ids, dict):
        return None
    model_id = bundle_model_ids.get(task)
    return str(model_id) if model_id else None


def load_optional_success_intent_model(
    success_intent_model_id: str | None,
    device: str,
    feature_root: Path,
) -> tuple[object | None, dict[str, object] | None, dict[str, int | bool] | None, dict[str, int | bool] | None]:
    if not success_intent_model_id:
        return None, None, None, None

    success_intent_model = load_model(success_intent_model_id, device)
    if success_intent_model is None:
        raise FileNotFoundError(f"Missing success_intent model checkpoint: {success_intent_model_id}")

    success_intent_graph_schema = validate_model_graph_schemas({"success_intent": success_intent_model})
    success_intent_feature_schema = infer_feature_graph_schema(get_success_intent_graph_dir(feature_root))
    validate_feature_graph_schema(
        success_intent_feature_schema,
        success_intent_graph_schema,
        context="Selected success_intent feature artifacts",
    )
    success_intent_model_record = get_model_provenance(success_intent_model_id)
    return (
        success_intent_model,
        success_intent_model_record,
        success_intent_graph_schema,
        success_intent_feature_schema,
    )


def run_success_intent_inference(
    match: Match,
    model: object,
    return_type: str | None,
    device: str,
    feature_root: Path | None = None,
) -> pd.DataFrame:
    del return_type
    original_actions = match.actions.copy()
    original_labels = match.labels.clone() if isinstance(match.labels, torch.Tensor) else match.labels
    original_stats = dict(getattr(match, "intended_receiver_stats", {}))
    original_runtime_feature_root = getattr(match, "runtime_feature_root", None)

    try:
        if feature_root is not None:
            match.runtime_feature_root = Path(feature_root)
        match.labels = load_success_intent_labels(match, feature_root)
        success_intent, _ = inference_gnn(match, model, device=device, post_action=False)
        if success_intent.empty:
            raise ValueError("No usable success_intent inference rows were produced for this match.")
        return success_intent
    finally:
        match.actions = original_actions
        match.labels = original_labels
        match.intended_receiver_stats = original_stats
        match.runtime_feature_root = original_runtime_feature_root


def resolve_match_ids(split: str, requested_match_ids: list[str] | None, feature_dir: Path) -> list[str]:
    train_ids, test_ids = load_base_splits(feature_dir)

    if split == "train":
        match_ids = train_ids.tolist()
    elif split == "test":
        match_ids = test_ids.tolist()
    else:
        match_ids = train_ids.tolist() + [match_id for match_id in test_ids.tolist() if match_id not in set(train_ids)]

    if requested_match_ids:
        requested = set(requested_match_ids)
        match_ids = [match_id for match_id in match_ids if match_id in requested]

    if not match_ids:
        raise ValueError("No matches selected for component inference.")

    return match_ids


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    resolved_model_ids, shared_context, bundle = resolve_model_selection(
        required_tasks=[
            "action_intent",
            "pass_intent",
            "pass_success",
            "outcome_scoring",
            "outcome_conceding",
        ],
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
    pass_height_model_id = resolve_optional_model_id("pass_height", getattr(args, "pass_height_model_id", None), bundle)
    if pass_height_model_id:
        resolved_model_ids["pass_height"] = pass_height_model_id
    component_run_id = args.run_id or generate_run_id("component")
    output_parent = Path(args.output_dir) if args.output_dir else SPORTEC_COMPONENT_RUNS_DIR
    output_dir = output_parent / component_run_id

    model_specs = {
        "action_intent": load_model(resolved_model_ids["action_intent"], device),
        "pass_intent": load_model(resolved_model_ids["pass_intent"], device),
        "pass_success": load_model(resolved_model_ids["pass_success"], device),
        "outcome_scoring": load_model(resolved_model_ids["outcome_scoring"], device),
        "outcome_conceding": load_model(
            resolved_model_ids["outcome_conceding"],
            device,
        ),
    }
    if pass_height_model_id:
        model_specs["pass_height"] = load_model(pass_height_model_id, device)
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    graph_schema = validate_model_graph_schemas(model_specs)
    runtime_feature_context = resolve_runtime_feature_run_context(
        getattr(args, "feature_run_id", None),
        shared_context,
        bundle,
        getattr(args, "intended_receiver_mode", None),
        graph_schema,
        context="Selected component feature artifacts",
    )
    feature_run_id = str(runtime_feature_context["feature_run_id"])
    intended_receiver_mode = str(
        runtime_feature_context.get("intended_receiver_mode")
        or shared_context.get("intended_receiver_mode")
        or DEFAULT_INTENDED_RECEIVER_MODE
    )
    return_type = resolve_runtime_return_type(shared_context, getattr(args, "return_type", None))
    feature_root = Path(runtime_feature_context["feature_root"])
    feature_schema = runtime_feature_context.get("feature_schema", {})
    shared_context = dict(shared_context)
    shared_context["feature_run_id"] = feature_run_id
    shared_context["runtime_feature_run_id"] = feature_run_id
    shared_context["intended_receiver_mode"] = intended_receiver_mode
    shared_context["runtime_intended_receiver_mode"] = intended_receiver_mode
    shared_context["return_type"] = return_type
    shared_context["runtime_return_type"] = return_type
    shared_context["runtime_feature_run_selection"] = runtime_feature_context.get("selection")
    no_physical_cache = bool(getattr(args, "no_physical_cache", False))
    refresh_physical_cache = bool(getattr(args, "refresh_physical_cache", False))
    physical_cache_dir = getattr(args, "physical_cache_dir", None) or str(
        get_pc_xpass_dir("sportec") if bool(getattr(args, "pc_xpass", False)) else get_runtime_physical_xpass_dir("sportec")
    )
    physical_num_workers = getattr(args, "physical_num_workers", "auto")
    physical_worker_thread_limit = int(getattr(args, "physical_worker_thread_limit", 1))
    physical_batch_size = int(getattr(args, "physical_batch_size", 16))
    pass_success_model = model_specs.get("pass_success")
    pass_success_args = getattr(pass_success_model, "args", None)
    pass_success_has_args = isinstance(pass_success_args, dict)
    if pass_success_has_args and bool(getattr(args, "use_physical_xpass", False)):
        pass_success_args["inference_use_physical_xpass"] = True
        pass_success_args["pc_xpass"] = bool(getattr(args, "pc_xpass", False))
        pass_success_args["x_pass_version"] = str(getattr(args, "x_pass_version", "top10"))
        pass_success_args["xpass_weight"] = str(getattr(args, "xpass_weight", "v3"))
        if getattr(args, "v4_power", None) is not None:
            pass_success_args["v4_power"] = float(args.v4_power)
        if getattr(args, "v4_zero", None) is not None:
            pass_success_args["v4_zero"] = float(args.v4_zero)
        pass_success_args["v4_discount"] = bool(getattr(args, "v4_discount", True))
        pass_success_args["ball_z_limit"] = getattr(args, "ball_z_limit", "none")
    if pass_success_has_args and (model_uses_physical_xpass(pass_success_args) or inference_uses_physical_xpass(pass_success_args)):
        pass_success_args["physical_runtime_cache_disabled"] = no_physical_cache
        pass_success_args["physical_runtime_cache_refresh"] = False
        pass_success_args["physical_num_workers"] = physical_num_workers
        pass_success_args["physical_worker_thread_limit"] = physical_worker_thread_limit
        pass_success_args["physical_batch_size"] = physical_batch_size
        pass_success_args["physical_runtime_cache_read_only"] = True
        if not no_physical_cache:
            pass_success_args["physical_cache_dir"] = physical_cache_dir
    model_records = {task: get_model_provenance(model_id) for task, model_id in resolved_model_ids.items()}
    success_intent_model_id = resolve_optional_success_intent_model_id(args, bundle)
    (
        success_intent_model,
        success_intent_model_record,
        success_intent_graph_schema,
        success_intent_feature_schema,
    ) = load_optional_success_intent_model(success_intent_model_id, device, feature_root)

    match_ids = resolve_match_ids(args.split, args.match_id, get_action_graph_dir(feature_root))

    physical_lookup_config = (
        physical_xpass_inference_lookup_config(pass_success_args, cache_dir=physical_cache_dir)
        if pass_success_has_args
        else {}
    )
    metadata = {
        "run_id": component_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "output_parent": str(output_parent),
        "split": args.split,
        "requested_match_ids": match_ids,
        "feature_run_id": feature_run_id,
        "runtime_feature_run_id": feature_run_id,
        "runtime_intended_receiver_mode": intended_receiver_mode,
        "runtime_return_type": return_type,
        "runtime_feature_run_selection": shared_context["runtime_feature_run_selection"],
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
        "feature_root": str(feature_root),
        "feature_schema": feature_schema,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": shared_context.get("target_family"),
        "graph_schema": graph_schema,
        "models": {
            "action_intent": resolved_model_ids["action_intent"],
            "pass_intent": resolved_model_ids["pass_intent"],
            "pass_success": resolved_model_ids["pass_success"],
            "outcome_scoring": resolved_model_ids["outcome_scoring"],
            "outcome_conceding": resolved_model_ids["outcome_conceding"],
        },
        "model_records": model_records,
        "model_feature_signatures": {task: record["feature_signature"] for task, record in model_records.items()},
        "processed_match_ids": [],
        "skipped_matches": [],
        "physical_xpass_skipped_actions": {},
        "physical_xpass_requested": bool(getattr(args, "use_physical_xpass", False)),
        "physical_xpass_hash_policy": PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
        "physical_xpass_lookup_policy": "dataset_event_frame_player_only",
        "physical_xpass_checkpoint_source": physical_xpass_source(pass_success_args) if pass_success_has_args else None,
        "physical_xpass_runtime_source": physical_lookup_config.get("source"),
        "physical_xpass_metric": physical_lookup_config.get("metric"),
        "x_pass_version": physical_lookup_config.get("x_pass_version"),
        "physical_xpass_weight_version": physical_lookup_config.get("weight_version"),
        "physical_xpass_v4_power": physical_lookup_config.get("v4_power") if physical_lookup_config.get("weight_version") == "v4" else None,
        "physical_xpass_v4_zero": physical_lookup_config.get("v4_zero") if physical_lookup_config.get("weight_version") == "v4" else None,
        "physical_xpass_v4_discount": physical_lookup_config.get("v4_discount") if physical_lookup_config.get("weight_version") == "v4" else None,
        "physical_xpass_ball_z_limit": physical_lookup_config.get("ball_z_limit"),
        "physical_cache_dir": None if no_physical_cache else physical_cache_dir,
        "physical_cache_disabled": no_physical_cache,
        "refresh_physical_cache": refresh_physical_cache,
        "physical_num_workers": physical_num_workers,
        "physical_worker_thread_limit": physical_worker_thread_limit,
        "physical_batch_size": physical_batch_size,
        "physical_xpass_runtime_stats": {},
        "status": "completed",
    }
    if success_intent_model_id and success_intent_model_record is not None:
        metadata["models"]["success_intent"] = success_intent_model_id
        metadata["model_records"]["success_intent"] = success_intent_model_record
        metadata["model_feature_signatures"]["success_intent"] = success_intent_model_record["feature_signature"]
        metadata["success_intent_graph_schema"] = success_intent_graph_schema
        metadata["success_intent_feature_schema"] = success_intent_feature_schema
        metadata["success_intent_skipped_matches"] = []
    if pass_height_model_id:
        metadata["models"]["pass_height"] = pass_height_model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, match_id in enumerate(match_ids, start=1):
        print(f"[{index}/{len(match_ids)}] {match_id}")
        try:
            match = load_match(
                match_id,
                intended_receiver_mode=intended_receiver_mode,
                return_type=return_type,
                feature_root=feature_root,
                add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
            )
            match_output_dir = output_dir / match_id

            action_intent, _ = inference_gnn(match, model_specs["action_intent"], device=device, post_action=False)
            pass_intent, _ = inference_gnn(match, model_specs["pass_intent"], device=device, post_action=False)
            pass_intent_receive, _ = inference_gnn(
                match,
                model_specs["pass_intent"],
                device=device,
                post_action=True,
            )
            pass_success = None
            pass_success_receive = None
            pass_height = None
            pass_height_receive = None
            try:
                pass_success, _ = inference_gnn(match, model_specs["pass_success"], device=device, post_action=False)
            except PhysicalXPassNoUsableRowsError as exc:
                print(f"  WARN {match_id}: pass_success frame_id export skipped: {summarize_exception(exc)}")
            try:
                pass_success_receive, _ = inference_gnn(
                    match,
                    model_specs["pass_success"],
                    device=device,
                    post_action=True,
                )
            except PhysicalXPassNoUsableRowsError as exc:
                print(f"  WARN {match_id}: pass_success receive_frame_id export skipped: {summarize_exception(exc)}")
            if "pass_height" in model_specs:
                pass_height, _ = inference_gnn(match, model_specs["pass_height"], device=device, post_action=False)
                pass_height_receive, _ = inference_gnn(
                    match,
                    model_specs["pass_height"],
                    device=device,
                    post_action=True,
                )
            scoring_failure, scoring_success = inference_gnn(
                match,
                model_specs["outcome_scoring"],
                device=device,
                post_action=False,
            )
            scoring_failure_receive, scoring_success_receive = inference_gnn(
                match,
                model_specs["outcome_scoring"],
                device=device,
                post_action=True,
            )
            conceding_failure, conceding_success = inference_gnn(
                match,
                model_specs["outcome_conceding"],
                device=device,
                post_action=False,
            )
            conceding_failure_receive, conceding_success_receive = inference_gnn(
                match,
                model_specs["outcome_conceding"],
                device=device,
                post_action=True,
            )
            if action_intent.empty:
                raise ValueError("No usable inference rows were produced for this match.")

            save_match_component_tables(
                match_output_dir,
                match.actions,
                action_intent=action_intent,
                pass_intent=pass_intent,
                pass_intent_receive=pass_intent_receive,
                pass_success=pass_success,
                pass_success_receive=pass_success_receive,
                pass_height=pass_height,
                pass_height_receive=pass_height_receive,
                scoring_success=scoring_success,
                scoring_success_receive=scoring_success_receive,
                scoring_failure=scoring_failure,
                scoring_failure_receive=scoring_failure_receive,
                conceding_success=conceding_success,
                conceding_success_receive=conceding_success_receive,
                conceding_failure=conceding_failure,
                conceding_failure_receive=conceding_failure_receive,
            )
            physical_skip_stats = getattr(match, "physical_xpass_skipped_actions", None)
            if physical_skip_stats:
                metadata["physical_xpass_skipped_actions"][match_id] = physical_skip_stats
            runtime_physical_stats = getattr(match, "physical_xpass_runtime_stats", None)
            if runtime_physical_stats:
                metadata["physical_xpass_runtime_stats"][match_id] = runtime_physical_stats

            if success_intent_model is not None:
                try:
                    success_intent = run_success_intent_inference(
                        match,
                        success_intent_model,
                        return_type=return_type,
                        device=device,
                        feature_root=feature_root,
                    )
                    save_component_table(success_intent, match.actions, match_output_dir / "success_intent.parquet")
                except Exception as exc:
                    error_summary = summarize_exception(exc)
                    metadata["success_intent_skipped_matches"].append({"match_id": match_id, "error": error_summary})
                    print(f"  WARN {match_id}: success_intent export skipped: {error_summary}")
            metadata["processed_match_ids"].append(match_id)
        except Exception as exc:
            error_summary = summarize_exception(exc)
            metadata["skipped_matches"].append({"match_id": match_id, "error": error_summary})
            print(f"  SKIP {match_id}: {error_summary}")

    write_run_metadata(output_dir, metadata)
    if not metadata["processed_match_ids"]:
        raise RuntimeError("No usable matches were available for component inference.")
    if metadata["skipped_matches"]:
        print(f"Skipped {len(metadata['skipped_matches'])} matches during component inference.")
    physical_xpass_required = bool(
        pass_success_has_args
        and (model_uses_physical_xpass(pass_success_args) or inference_uses_physical_xpass(pass_success_args))
    )
    physical_xpass_cache_summary = summarize_physical_xpass_cache_usage(
        physical_xpass_required=physical_xpass_required,
        cache_disabled=no_physical_cache,
        refresh_requested=refresh_physical_cache,
        cache_dir=None if no_physical_cache else physical_cache_dir,
        runtime_stats=metadata["physical_xpass_runtime_stats"],
        skipped_stats=metadata["physical_xpass_skipped_actions"],
    )
    metadata["physical_xpass_cache_summary"] = physical_xpass_cache_summary
    write_run_metadata(output_dir, metadata)

    if output_parent.resolve() == SPORTEC_COMPONENT_RUNS_DIR.resolve():
        write_latest_run("component", component_run_id)
    print(f"Component run id: {component_run_id}")
    print(f"Saved component predictions to {output_dir}")
    print(format_physical_xpass_cache_summary(physical_xpass_cache_summary))


if __name__ == "__main__":
    main()
