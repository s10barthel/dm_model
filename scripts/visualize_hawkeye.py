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
from PIL import Image

from datatools.hawkeye import (
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    infer_hawkeye_components,
    load_hawkeye_ball,
    load_hawkeye_models,
    load_hawkeye_tracking,
)
from models.utils import validate_model_graph_schemas
from datatools.viz_helpers import compute_pass_score, figure_to_rgb_image, save_animation
from datatools.viz_snapshot import SnapshotVisualizer
from project_config import DATA_ROOT, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situation-id", default=None, help="Hawkeye situation id to visualize.")
    parser.add_argument("--action-id", default=None, help="Alias for --situation-id.")
    parser.add_argument(
        "--tracking-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "centroid_data_team.csv"),
    )
    parser.add_argument(
        "--ball-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "ball_data_selected.csv"),
    )
    parser.add_argument("--freeze-ballreceipt", dest="freeze_ballreceipt", action="store_true")
    parser.add_argument("--no-freeze-ballreceipt", dest="freeze_ballreceipt", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Save GIFs instead of the default MP4 animations.")
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations" / "hawkeye"))
    parser.set_defaults(freeze_ballreceipt=True)
    return parser.parse_args()


def render_frame_image(
    situation,
    frame_id: int,
    component_name: str,
    probs: pd.Series | None,
    show_trajectories: bool = False,
) -> Image.Image:
    frame_start = max(frame_id - 24, int(situation.frame_meta.index.min()))
    snapshot = situation.tracking.loc[frame_start:frame_id].copy()
    ball_xy = snapshot[["ball_x", "ball_y"]].copy()
    frame_info = situation.frame_meta.loc[frame_id]
    attacking_prefix = frame_info["possession_prefix"]

    if probs is None or probs.empty:
        component_probs = pd.Series(dtype=float)
    else:
        attack_targets = [player_id for player_id in probs.index if isinstance(player_id, str) and player_id.startswith(attacking_prefix)]
        component_probs = probs.loc[attack_targets].dropna().sort_values(ascending=False)

    player_annots = component_probs if not component_probs.empty else None
    highlight_players = (
        {frame_info["possessor_object_id"]: "#ffd400"}
        if isinstance(frame_info.get("possessor_object_id"), str)
        else None
    )

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

    title = f"{situation.situation_id} | {frame_info['abs_time']:.3f} | {component_name.replace('_', ' ').title()}"
    fig, ax = visualizer.plot(rotate_pitch=False, anonymize=True, annot_type=component_name, show=False)
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

    image = figure_to_rgb_image(fig, dpi=150)
    plt.close(fig)
    return image


def main() -> None:
    args = parse_args()
    situation_id = args.situation_id or args.action_id
    if not situation_id:
        raise ValueError("Please provide --situation-id (or --action-id) for Hawkeye visualization.")

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / str(situation_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    situation_tracking = tracking.loc[tracking["id"] == str(situation_id)].copy()
    if situation_tracking.empty:
        raise KeyError(f"Hawkeye situation id {situation_id} was not found in {args.tracking_csv}.")

    model_specs = load_hawkeye_models(
        action_intent_model_id=args.action_intent_model_id,
        pass_success_model_id=args.pass_success_model_id,
        outcome_scoring_model_id=args.outcome_scoring_model_id,
        outcome_conceding_model_id=args.outcome_conceding_model_id,
        device=device,
    )
    graph_schema = validate_model_graph_schemas(model_specs)
    situation, _, _ = build_hawkeye_situation(
        situation_tracking,
        ball,
        freeze_ballreceipt=args.freeze_ballreceipt,
        add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
    )
    components = infer_hawkeye_components(situation, model_specs, device=device)

    component_frames: dict[str, pd.DataFrame | None] = {
        "action_intent": components.get("action_intent"),
        "pass_success": components.get("pass_success"),
        "outcome_scoring_success": components.get("outcome_scoring_success"),
        "outcome_scoring_failure": components.get("outcome_scoring_failure"),
        "outcome_conceding_success": components.get("outcome_conceding_success"),
        "outcome_conceding_failure": components.get("outcome_conceding_failure"),
    }
    if all(
        component_frames[name] is not None
        for name in [
            "pass_success",
            "outcome_scoring_success",
            "outcome_scoring_failure",
            "outcome_conceding_success",
            "outcome_conceding_failure",
        ]
    ):
        component_frames["pass_score"] = compute_pass_score(
            pass_success=component_frames["pass_success"],
            outcome_scoring_success=component_frames["outcome_scoring_success"],
            outcome_scoring_failure=component_frames["outcome_scoring_failure"],
            outcome_conceding_success=component_frames["outcome_conceding_success"],
            outcome_conceding_failure=component_frames["outcome_conceding_failure"],
        )
    else:
        component_frames["pass_score"] = None

    frame_ids = [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
    for component_name, component_table in component_frames.items():
        def iter_component_images():
            for frame_id in frame_ids:
                frame_probs = component_table.loc[frame_id] if component_table is not None and frame_id in component_table.index else None
                yield render_frame_image(
                    situation,
                    frame_id,
                    component_name,
                    frame_probs,
                    show_trajectories=args.show_trajectories,
                )

        suffix = "gif" if args.gif else "mp4"
        output_path = output_dir / f"{component_name}.{suffix}"
        save_animation(iter_component_images(), output_path, fps=25.0, gif=args.gif)

    print(f"Saved Hawkeye animations to {output_dir}")


if __name__ == "__main__":
    main()
