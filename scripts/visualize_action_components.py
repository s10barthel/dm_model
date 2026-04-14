from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch

from datatools import config
from datatools.graph_feature import construct_graph_features
from datatools.match import Match
from datatools.viz_helpers import compute_pass_score
from datatools.viz_snapshot import SnapshotVisualizer
from inference import inference_gnn
from models.utils import load_model
from project_config import (
    ACTION_GRAPH_DIR,
    DATA_ROOT,
    DEFAULT_INTENDED_RECEIVER_MODE,
    get_relevant_model_ids,
    get_resolved_action_path,
    resolve_intended_receiver_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    identifier_group = parser.add_mutually_exclusive_group(required=True)
    identifier_group.add_argument(
        "--action-id",
        type=int,
        help="Action id from data/ajax/event_synced/<match_id>.csv.",
    )
    identifier_group.add_argument(
        "--row-index",
        type=int,
        help="Legacy modeled-action row index used by earlier versions of this script.",
    )
    identifier_group.add_argument(
        "--original-event-id",
        help="Original Sportec event id from the original_event_id column.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--use_xt", action="store_true")
    parser.add_argument("--use-original-intended-receiver", action="store_true")
    parser.add_argument("--use-intended-receiver-model", action="store_true")
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations"))
    return parser.parse_args()


def load_match(match_id: str, intended_receiver_mode: str = DEFAULT_INTENDED_RECEIVER_MODE) -> Match:
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineups = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()

    match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
    resolved_action_path = get_resolved_action_path(match_id, intended_receiver_mode=intended_receiver_mode)
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
    )

    graph_path = ACTION_GRAPH_DIR / f"{match_id}.pt"
    if graph_path.exists():
        match.graph_features_0 = torch.load(graph_path, weights_only=False)
    else:
        match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)

    return match


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


def resolve_action_index(match: Match, args: argparse.Namespace) -> tuple[int, str]:
    match_id = match.match_id or args.match_id

    if args.row_index is not None:
        event_index = int(args.row_index)
        identifier_desc = f"Row index {event_index}"
        if event_index not in match.events.index:
            raise KeyError(f"{identifier_desc} is not present in match {match_id}.")
        if event_index not in match.actions.index:
            raise build_not_modeled_error(match, match_id, identifier_desc, event_index)
        return event_index, str(int(match.actions.at[event_index, "action_id"]))

    if args.action_id is not None:
        identifier_desc = f"CSV action_id {args.action_id}"
        matches = match.events.index[match.events["action_id"] == args.action_id].tolist()
    else:
        requested_original_event_id = str(args.original_event_id)
        identifier_desc = f"Original event id {requested_original_event_id}"
        matches = match.events.index[
            match.events["original_event_id"].astype("string") == requested_original_event_id
        ].tolist()

    if not matches:
        raise KeyError(f"{identifier_desc} is not present in match {match_id}.")
    if len(matches) > 1:
        raise ValueError(f"{identifier_desc} is not unique in match {match_id}.")

    event_index = int(matches[0])
    if event_index not in match.actions.index:
        raise build_not_modeled_error(match, match_id, identifier_desc, event_index)

    return event_index, str(int(match.actions.at[event_index, "action_id"]))


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
    highlight_players = {action["object_id"]: "#ffd400"} if isinstance(action["object_id"], str) else None

    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_annots=player_annots,
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


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    intended_receiver_mode = resolve_intended_receiver_mode(
        use_original_intended_receiver=args.use_original_intended_receiver,
        use_intended_receiver_model=args.use_intended_receiver_model,
    )
    default_model_ids = get_relevant_model_ids(intended_receiver_mode=intended_receiver_mode, use_xt=args.use_xt)

    match = load_match(args.match_id, intended_receiver_mode=intended_receiver_mode)
    action_index, display_action_id = resolve_action_index(match, args)
    output_dir = Path(args.output_dir) / args.match_id / display_action_id

    model_specs = {
        "action_intent": args.action_intent_model_id or default_model_ids["action_intent"],
        "pass_success": args.pass_success_model_id or default_model_ids["pass_success"],
        "outcome_scoring": args.outcome_scoring_model_id or default_model_ids["outcome_scoring"],
        "outcome_conceding": args.outcome_conceding_model_id or default_model_ids["outcome_conceding"],
    }
    component_prob_rows: dict[str, pd.Series] = {}

    for component_name, model_id in model_specs.items():
        model = load_model(model_id, device)
        if model is None:
            raise FileNotFoundError(f"Could not load model checkpoint {model_id}.")

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
            probs, _ = inference_gnn(match, model, device=device, post_action=False, event_indices=[action_index])
            component_prob_rows[component_name] = probs.loc[action_index]

    component_prob_rows["pass_score"] = compute_pass_score(
        pass_success=component_prob_rows["pass_success"],
        outcome_scoring_success=component_prob_rows["outcome_scoring_success"],
        outcome_scoring_failure=component_prob_rows["outcome_scoring_failure"],
        outcome_conceding_success=component_prob_rows["outcome_conceding_success"],
        outcome_conceding_failure=component_prob_rows["outcome_conceding_failure"],
    )

    component_order = [
        "action_intent",
        "pass_success",
        "outcome_scoring_success",
        "outcome_scoring_failure",
        "outcome_conceding_success",
        "outcome_conceding_failure",
        "pass_score",
    ]
    for component_name in component_order:
        render_component(
            match=match,
            action_index=action_index,
            display_action_id=display_action_id,
            component_name=component_name,
            probs=component_prob_rows[component_name],
            output_path=output_dir / f"{component_name}.png",
            show_trajectories=args.show_trajectories,
        )

    print(f"Saved component plots to {output_dir}")


if __name__ == "__main__":
    main()
