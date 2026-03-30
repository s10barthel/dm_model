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

from datatools.graph_feature import construct_graph_features
from datatools.match import Match
from datatools.viz_snapshot import SnapshotVisualizer
from inference import inference_gnn
from models.utils import load_model
from project_config import DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--action-id", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations"))
    return parser.parse_args()


def load_match(match_id: str) -> Match:
    events = pd.read_csv(DATA_ROOT / "event_synced" / f"{match_id}.csv", parse_dates=["utc_timestamp"])
    tracking = pd.read_parquet(DATA_ROOT / "tracking_processed" / f"{match_id}.parquet")
    lineups = pd.read_parquet(DATA_ROOT / "lineup" / "line_up.parquet")
    match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id].copy()

    match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
    match.labels = match.construct_labels(discount_xg=True)

    graph_path = DATA_ROOT / "features" / "action_graphs" / f"{match_id}.pt"
    if graph_path.exists():
        match.graph_features_0 = torch.load(graph_path, weights_only=False)
    else:
        match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)

    return match


def render_component(
    match: Match,
    action_id: int,
    component_name: str,
    probs: pd.Series,
    output_path: Path,
    show_trajectories: bool = False,
) -> None:
    action = match.actions.loc[action_id]
    frame_id = int(action["frame_id"])
    snapshot = match.tracking.loc[max(frame_id - 24, 0) : frame_id].copy()
    ball_xy = snapshot[["ball_x", "ball_y"]].copy()

    attacking_prefix = action["object_id"][:4]
    attack_targets = [player_id for player_id in probs.index if player_id.startswith(attacking_prefix)]
    component_probs = probs.loc[attack_targets].dropna().sort_values(ascending=False)
    player_colors = component_probs if not component_probs.empty else None
    player_annots = component_probs if not component_probs.empty else None
    highlight_players = {action["object_id"]: "#ffd400"} if isinstance(action["object_id"], str) else None

    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_colors=player_colors,
        player_annots=player_annots,
        show_velocities=True,
        show_trajectories=show_trajectories,
        highlight_players=highlight_players,
    )

    rotate_pitch = attacking_prefix == "away"
    title = f"{action['spadl_type']} {action_id} - {component_name.replace('_', ' ').title()}"
    visualizer.plot(rotate_pitch=rotate_pitch, anonymize=False, annot_type=component_name)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.match_id / str(args.action_id)
    device = args.device if torch.cuda.is_available() else "cpu"

    match = load_match(args.match_id)
    if args.action_id not in match.actions.index:
        raise KeyError(f"Action id {args.action_id} is not present in match {args.match_id}.")

    model_specs = {
        "action_intent": args.action_intent_model_id,
        "pass_success": args.pass_success_model_id,
        "outcome_scoring": args.outcome_scoring_model_id,
        "outcome_conceding": args.outcome_conceding_model_id,
    }

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
                event_indices=[args.action_id],
            )
            for outcome_case, probs in (("success", success_probs), ("failure", failure_probs)):
                component_probs = probs.loc[args.action_id]
                render_component(
                    match=match,
                    action_id=args.action_id,
                    component_name=f"{component_name}_{outcome_case}",
                    probs=component_probs,
                    output_path=output_dir / f"{component_name}_{outcome_case}.png",
                    show_trajectories=args.show_trajectories,
                )
        else:
            probs, _ = inference_gnn(match, model, device=device, post_action=False, event_indices=[args.action_id])
            component_probs = probs.loc[args.action_id]
            render_component(
                match=match,
                action_id=args.action_id,
                component_name=component_name,
                probs=component_probs,
                output_path=output_dir / f"{component_name}.png",
                show_trajectories=args.show_trajectories,
            )

    print(f"Saved component plots to {output_dir}")


if __name__ == "__main__":
    main()
