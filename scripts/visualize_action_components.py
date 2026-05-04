from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datatools import config
from datatools.graph_feature import construct_graph_features, summarize_ball_trajectory
from datatools.match import Match
from datatools.viz_helpers import compute_pass_score
from datatools.viz_snapshot import SnapshotVisualizer
from inference import inference_gnn, load_success_intent_labels
from models.utils import (
    load_model,
    resolve_model_selection,
    resolve_runtime_return_type,
    resolve_runtime_feature_run_context,
    validate_model_graph_schemas,
)
from project_config import (
    DATA_ROOT,
    DEFAULT_INTENDED_RECEIVER_MODE,
    INTENDED_RECEIVER_MODES,
    get_action_graph_dir,
    get_resolved_action_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument(
        "--action-id",
        action="append",
        type=int,
        help="Action id from data/event_synced/<match_id>.csv. Repeat to visualize multiple actions.",
    )
    parser.add_argument(
        "--row-index",
        action="append",
        type=int,
        help="Legacy modeled-action row index used by earlier versions of this script. Repeat to visualize multiple rows.",
    )
    parser.add_argument(
        "--original-event-id",
        action="append",
        help="Original Sportec event id from the original_event_id column. Repeat to visualize multiple events.",
    )
    parser.add_argument("--player-id", action="append", help="Filter by player_id. Repeat for OR within this column.")
    parser.add_argument("--object-id", action="append", help="Filter by object_id. Repeat for OR within this column.")
    parser.add_argument(
        "--advanced-position",
        action="append",
        help="Filter by advanced_position. Repeat for OR within this column.",
    )
    parser.add_argument("--team-id", action="append", help="Filter by team_id. Repeat for OR within this column.")
    parser.add_argument("--spadl-type", action="append", help="Filter by spadl_type. Repeat for OR within this column.")
    parser.add_argument("--success", action="append", type=parse_bool, help="Filter by success true/false.")
    parser.add_argument("--offside", action="append", type=parse_bool, help="Filter by offside true/false.")
    parser.add_argument("--next-type", action="append", help="Filter by next_type. Repeat for OR within this column.")
    for column in ("start_x", "start_y", "end_x", "end_y"):
        option_name = column.replace("_", "-")
        parser.add_argument(f"--{option_name}-lt", type=float, help=f"Filter to rows where {column} is lower than this value.")
        parser.add_argument(f"--{option_name}-gt", type=float, help=f"Filter to rows where {column} is higher than this value.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--feature-run-id", help="Runtime feature run used to load graphs/resolved actions.")
    parser.add_argument("--intended-receiver-mode", choices=INTENDED_RECEIVER_MODES, help="Runtime resolved-action mode.")
    parser.add_argument("--return-type", "--return_type", dest="return_type", help="Runtime return type used for label construction.")
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--success-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations"))
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


EXACT_FILTERS = {
    "player_id": "player_id",
    "object_id": "object_id",
    "advanced_position": "advanced_position",
    "team_id": "team_id",
    "spadl_type": "spadl_type",
    "success": "success",
    "offside": "offside",
    "next_type": "next_type",
}

NUMERIC_FILTER_COLUMNS = ("start_x", "start_y", "end_x", "end_y")


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
        match.graph_features_0 = torch.load(graph_path, weights_only=False)
    else:
        match.graph_features_0 = construct_graph_features(
            match,
            extend=True,
            post_action=False,
            add_v_edge_features=add_v_edge_features,
        )

    return match


def run_success_intent_component(
    match: Match,
    model: object,
    feature_root: Path,
    device: str,
    action_index: int,
) -> pd.Series:
    labels = load_success_intent_labels(match, feature_root)
    saved_action_indices = {int(action_index_i) for action_index_i in labels[:, 0].detach().cpu().numpy().astype(int)}
    if int(action_index) not in saved_action_indices:
        action = match.actions.loc[action_index]
        action_id = action.get("action_id", action_index)
        raise ValueError(
            f"Action index {action_index} (action_id={action_id}) is not present in saved success-intent labels. "
            "The success-intent component is only available for successful pass actions in the selected feature run."
        )

    original_labels = match.labels.clone() if isinstance(match.labels, torch.Tensor) else match.labels
    original_runtime_feature_root = getattr(match, "runtime_feature_root", None)
    try:
        match.runtime_feature_root = Path(feature_root)
        match.labels = labels
        probs, _ = inference_gnn(match, model, device=device, post_action=False, event_indices=[action_index])
        return probs.loc[action_index]
    finally:
        match.labels = original_labels
        match.runtime_feature_root = original_runtime_feature_root


def describe_action_subset_exclusion(match: Match, event_index: int) -> str:
    event = match.events.loc[event_index]
    spadl_type = str(event.get("spadl_type", "unknown"))
    frame_id = event.get("frame_id")
    receive_frame_id = event.get("receive_frame_id")
    receiver_id = event.get("receiver_id")
    next_player_id = event.get("next_player_id")
    next_type = event.get("next_type")
    object_id = event.get("object_id")

    if spadl_type in config.PASS:
        if pd.isna(frame_id) or pd.isna(receive_frame_id):
            return "pass/cross is missing frame_id or receive_frame_id"
        if not (
            receiver_id == next_player_id
            or receiver_id == "out"
            or next_type in ["foul", "freekick_short"]
        ):
            return (
                "pass/cross is excluded because receiver_id does not match next_player_id, "
                "it is not an out-of-play pass, and it does not transition into foul/freekick_short"
            )
    elif spadl_type in {"take_on", "dispossessed"}:
        if pd.isna(frame_id):
            return "dribble candidate is missing frame_id"
    elif spadl_type in config.SHOT:
        if pd.isna(frame_id):
            return "shot is missing frame_id"
    else:
        return f"spadl_type {spadl_type!r} is not part of the modeled pass/dribble/shot subset"

    if pd.isna(next_type):
        return "event has no next_type and is dropped from the modeled action subset"
    if not match.has_valid_action_snapshot(frame_id, object_id):
        return "tracking snapshot for the possessor is not valid at frame_id"
    return "event is not part of the modeled action subset"


def build_not_modeled_error(match: Match, match_id: str, identifier_desc: str, event_index: int) -> KeyError:
    event = match.events.loc[event_index]
    reason = describe_action_subset_exclusion(match, event_index)
    return KeyError(
        f"{identifier_desc} resolves to row index {event_index} in match {match_id} "
        f"(action_id={event.get('action_id')}, original_event_id={event.get('original_event_id')}, "
        f"spadl_type={event.get('spadl_type')}) but that event is not part of the modeled action subset: {reason}."
    )


def warn_skip(message: str) -> None:
    print(f"Skipping: {message}", file=sys.stderr)


def _dedupe_ordered(values: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _resolve_candidate_event_indices(match: Match, args: argparse.Namespace) -> list[int]:
    match_id = match.match_id or args.match_id
    candidates: list[int] = []
    has_explicit_selectors = any([args.action_id, args.row_index, args.original_event_id])

    if not has_explicit_selectors:
        return [int(index) for index in match.events.index.tolist()]

    for event_index in args.row_index or []:
        identifier_desc = f"Row index {event_index}"
        if int(event_index) not in match.events.index:
            warn_skip(f"{identifier_desc} is not present in match {match_id}.")
            continue
        candidates.append(int(event_index))

    for action_id in args.action_id or []:
        identifier_desc = f"CSV action_id {action_id}"
        matches = match.events.index[match.events["action_id"] == int(action_id)].tolist()
        if not matches:
            warn_skip(f"{identifier_desc} is not present in match {match_id}.")
            continue
        if len(matches) > 1:
            warn_skip(f"{identifier_desc} is not unique in match {match_id}.")
            continue
        candidates.append(int(matches[0]))

    for original_event_id in args.original_event_id or []:
        requested_original_event_id = str(original_event_id)
        identifier_desc = f"Original event id {requested_original_event_id}"
        matches = match.events.index[
            match.events["original_event_id"].astype("string") == requested_original_event_id
        ].tolist()
        if not matches:
            warn_skip(f"{identifier_desc} is not present in match {match_id}.")
            continue
        if len(matches) > 1:
            warn_skip(f"{identifier_desc} is not unique in match {match_id}.")
            continue
        candidates.append(int(matches[0]))

    return _dedupe_ordered(candidates)


def _filter_mask_for_events(events: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    mask = pd.Series(True, index=events.index)

    for attr_name, column in EXACT_FILTERS.items():
        requested_values = getattr(args, attr_name, None)
        if not requested_values:
            continue
        if column in {"success", "offside"}:
            normalized_values = {bool(value) for value in requested_values}
            column_values = events[column].map(lambda value: value if pd.isna(value) else bool(value))
            mask &= column_values.isin(normalized_values)
        else:
            normalized_values = {str(value) for value in requested_values}
            mask &= events[column].astype("string").isin(normalized_values)

    for column in NUMERIC_FILTER_COLUMNS:
        values = pd.to_numeric(events[column], errors="coerce")
        lower_than = getattr(args, f"{column}_lt")
        higher_than = getattr(args, f"{column}_gt")
        if lower_than is not None:
            mask &= values < float(lower_than)
        if higher_than is not None:
            mask &= values > float(higher_than)

    return mask


def resolve_action_indices(match: Match, args: argparse.Namespace) -> list[tuple[int, str]]:
    match_id = match.match_id or args.match_id
    candidate_event_indices = _resolve_candidate_event_indices(match, args)
    if not candidate_event_indices:
        return []

    filter_mask = _filter_mask_for_events(match.events, args)
    selected: list[tuple[int, str]] = []
    for event_index in candidate_event_indices:
        if event_index not in filter_mask.index or not bool(filter_mask.at[event_index]):
            continue
        if event_index not in match.actions.index:
            reason = describe_action_subset_exclusion(match, event_index)
            event = match.events.loc[event_index]
            warn_skip(
                f"row index {event_index} in match {match_id} "
                f"(action_id={event.get('action_id')}, original_event_id={event.get('original_event_id')}, "
                f"spadl_type={event.get('spadl_type')}) is not part of the modeled action subset: {reason}."
            )
            continue
        selected.append((event_index, str(int(match.actions.at[event_index, "action_id"]))))

    return selected


def resolve_highlight_players(
    snapshot: pd.DataFrame,
    action: pd.Series,
    component_name: str,
) -> dict[str, str] | None:
    highlight_players: dict[str, str] = {}
    object_id = action.get("object_id")
    if isinstance(object_id, str):
        highlight_players[object_id] = "#ffd400"

    if component_name == "intended_recipient":
        intent_id = action.get("intent_id")
        if (
            isinstance(intent_id, str)
            and intent_id != object_id
            and intent_id.startswith(("home_", "away_"))
            and f"{intent_id}_x" in snapshot.columns
            and f"{intent_id}_y" in snapshot.columns
            and not pd.isna(snapshot[f"{intent_id}_x"].iloc[-1])
            and not pd.isna(snapshot[f"{intent_id}_y"].iloc[-1])
        ):
            highlight_players[intent_id] = "#00b050"

    return highlight_players or None


def resolve_ball_velocity_xy(match: Match, action_index: int, component_name: str) -> tuple[float, float] | None:
    action = match.actions.loc[action_index]
    if component_name != "intended_recipient" or action.get("action_type") != "pass":
        return None

    summary = summarize_ball_trajectory(match, action_index, fps=match.fps, rotate_to_ltr=False)
    if summary.shape[0] < 3:
        return None

    ball_velocity_xy = np.asarray([summary[0] * summary[1], summary[0] * summary[2]], dtype=float)
    if not np.isfinite(ball_velocity_xy).all() or np.linalg.norm(ball_velocity_xy) <= 1e-6:
        return None

    return float(ball_velocity_xy[0]), float(ball_velocity_xy[1])


def render_component(
    match: Match,
    action_index: int,
    display_action_id: str,
    component_name: str,
    probs: pd.Series,
    output_path: Path,
    show_trajectories: bool = False,
) -> None:
    action = match.actions.loc[action_index]
    frame_id = int(action["frame_id"])
    snapshot = match.tracking.loc[max(frame_id - 24, 0) : frame_id].copy()
    ball_xy = snapshot[["ball_x", "ball_y"]].copy()

    attacking_prefix = action["object_id"][:4]
    attack_targets = [player_id for player_id in probs.index if player_id.startswith(attacking_prefix)]
    component_probs = probs.loc[attack_targets].dropna().sort_values(ascending=False)
    player_annots = component_probs if not component_probs.empty else None
    highlight_players = resolve_highlight_players(snapshot, action, component_name)
    ball_velocity_xy = resolve_ball_velocity_xy(match, action_index, component_name)

    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_annots=player_annots,
        ball_velocity_xy=ball_velocity_xy,
        show_velocities=True,
        show_trajectories=show_trajectories,
        highlight_players=highlight_players,
        style="pitchcontrol",
        attacking_team_prefix=attacking_prefix,
    )

    rotate_pitch = attacking_prefix == "away"
    title = f"{action['spadl_type']} {display_action_id} - {component_name.replace('_', ' ').title()}"
    fig, ax = visualizer.plot(rotate_pitch=rotate_pitch, anonymize=False, annot_type=component_name, show=False)
    fig.subplots_adjust(top=0.92, left=0.02, right=0.98, bottom=0.02)
    ax.text(
        0.5,
        1.01,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_action_components(
    match: Match,
    loaded_models: dict[str, object],
    feature_root: Path,
    device: str,
    action_index: int,
    display_action_id: str,
    output_dir: Path,
    show_trajectories: bool = False,
) -> None:
    component_prob_rows: dict[str, pd.Series] = {}

    for component_name, model in loaded_models.items():
        if component_name.startswith("outcome_"):
            failure_probs, success_probs = inference_gnn(
                match,
                model,
                device=device,
                post_action=False,
                event_indices=[action_index],
            )
            for outcome_case, probs in (("success", success_probs), ("failure", failure_probs)):
                component_prob_rows[f"{component_name}_{outcome_case}"] = probs.loc[action_index]
        else:
            if component_name == "intended_recipient":
                component_prob_rows[component_name] = run_success_intent_component(
                    match,
                    model,
                    feature_root,
                    device,
                    action_index,
                )
            else:
                probs, _ = inference_gnn(match, model, device=device, post_action=False, event_indices=[action_index])
                component_prob_rows[component_name] = probs.loc[action_index]

    component_prob_rows["pass_score"] = compute_pass_score(
        pass_success=component_prob_rows["pass_success"],
        outcome_scoring_success=component_prob_rows["outcome_scoring_success"],
        outcome_scoring_failure=component_prob_rows["outcome_scoring_failure"],
        outcome_conceding_success=component_prob_rows["outcome_conceding_success"],
        outcome_conceding_failure=component_prob_rows["outcome_conceding_failure"],
    )

    component_order = ["action_intent", "pass_intent"]
    if "intended_recipient" in component_prob_rows:
        component_order.append("intended_recipient")
    component_order.extend(
        [
            "pass_success",
            "outcome_scoring_success",
            "outcome_scoring_failure",
            "outcome_conceding_success",
            "outcome_conceding_failure",
            "pass_score",
        ]
    )
    for component_name in component_order:
        render_component(
            match=match,
            action_index=action_index,
            display_action_id=display_action_id,
            component_name=component_name,
            probs=component_prob_rows[component_name],
            output_path=output_dir / f"{component_name}.png",
            show_trajectories=show_trajectories,
        )


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
    success_intent_model_id = args.success_intent_model_id
    if not success_intent_model_id and bundle is not None:
        success_intent_model_id = bundle.get("model_ids", {}).get("success_intent")

    model_ids = {
        "action_intent": resolved_model_ids["action_intent"],
        "pass_intent": resolved_model_ids["pass_intent"],
        "pass_success": resolved_model_ids["pass_success"],
        "outcome_scoring": resolved_model_ids["outcome_scoring"],
        "outcome_conceding": resolved_model_ids["outcome_conceding"],
    }
    if success_intent_model_id:
        model_ids["intended_recipient"] = str(success_intent_model_id)
    loaded_models = {name: load_model(model_id, device) for name, model_id in model_ids.items()}
    missing = [name for name, model in loaded_models.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Could not load model checkpoints for: {', '.join(missing)}.")
    graph_schema = validate_model_graph_schemas(loaded_models)
    runtime_feature_context = resolve_runtime_feature_run_context(
        args.feature_run_id,
        shared_context,
        bundle,
        args.intended_receiver_mode,
        graph_schema,
        context="Selected visualization feature artifacts",
    )
    feature_run_id = str(runtime_feature_context["feature_run_id"])
    intended_receiver_mode = str(runtime_feature_context["intended_receiver_mode"])
    return_type = resolve_runtime_return_type(shared_context, args.return_type)
    feature_root = Path(runtime_feature_context["feature_root"])
    shared_context = dict(shared_context)
    shared_context["feature_run_id"] = feature_run_id
    shared_context["runtime_feature_run_id"] = feature_run_id
    shared_context["intended_receiver_mode"] = intended_receiver_mode
    shared_context["runtime_intended_receiver_mode"] = intended_receiver_mode
    shared_context["return_type"] = return_type
    shared_context["runtime_return_type"] = return_type
    shared_context["runtime_feature_run_selection"] = runtime_feature_context["selection"]

    match = load_match(
        args.match_id,
        intended_receiver_mode=intended_receiver_mode,
        return_type=return_type,
        feature_root=feature_root,
        add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
    )
    selected_actions = resolve_action_indices(match, args)
    if not selected_actions:
        raise ValueError("No matching modeled actions were selected.")

    saved_dirs: list[Path] = []
    for action_index, display_action_id in selected_actions:
        output_dir = Path(args.output_dir) / args.match_id / display_action_id
        try:
            render_action_components(
                match=match,
                loaded_models=loaded_models,
                feature_root=feature_root,
                device=device,
                action_index=action_index,
                display_action_id=display_action_id,
                output_dir=output_dir,
                show_trajectories=args.show_trajectories,
            )
        except Exception as exc:
            warn_skip(f"action_id={display_action_id} in match {args.match_id} failed during rendering: {exc}")
            continue
        saved_dirs.append(output_dir)
        print(f"Saved component plots to {output_dir}")

    if not saved_dirs:
        raise RuntimeError("All selected actions were skipped or failed during rendering.")
    print(f"Saved component plots for {len(saved_dirs)} action(s).")


if __name__ == "__main__":
    main()
