from __future__ import annotations

import argparse
import io
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outcome-case", default="success", choices=["success", "failure"])
    parser.add_argument("--action-intent-model-id", default="action_intent/00")
    parser.add_argument("--pass-success-model-id", default="pass_success/20")
    parser.add_argument("--outcome-scoring-model-id", default="outcome_scoring/20")
    parser.add_argument("--outcome-conceding-model-id", default="outcome_conceding/20")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations" / "hawkeye"))
    return parser.parse_args()


def normalize_for_sizes(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    max_value = float(values.max())
    if max_value <= 0:
        return values * 0 + 0.2
    return (values / max_value).clip(lower=0.15)


def render_frame_image(
    situation,
    frame_id: int,
    component_name: str,
    probs: pd.Series | None,
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

    player_sizes = normalize_for_sizes(component_probs) if not component_probs.empty else None
    player_colors = component_probs if not component_probs.empty else None
    player_annots = component_probs if not component_probs.empty else None

    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_sizes=player_sizes,
        player_colors=player_colors,
        player_annots=player_annots,
        show_velocities=True,
    )

    title = f"{situation.situation_id} | {frame_info['abs_time']:.3f} | {component_name.replace('_', ' ').title()}"
    visualizer.plot(rotate_pitch=False, anonymize=True, annot_type=component_name)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=18)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


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

    situation, _, _ = build_hawkeye_situation(situation_tracking, ball)
    model_specs = load_hawkeye_models(
        action_intent_model_id=args.action_intent_model_id,
        pass_success_model_id=args.pass_success_model_id,
        outcome_scoring_model_id=args.outcome_scoring_model_id,
        outcome_conceding_model_id=args.outcome_conceding_model_id,
        device=device,
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

    selected_components = {
        "action_intent": component_frames["action_intent"],
        "pass_success": component_frames["pass_success"],
        f"outcome_scoring_{args.outcome_case}": component_frames[f"outcome_scoring_{args.outcome_case}"],
        f"outcome_conceding_{args.outcome_case}": component_frames[f"outcome_conceding_{args.outcome_case}"],
    }

    frame_ids = [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
    for component_name, component_table in selected_components.items():
        images: list[Image.Image] = []
        for frame_id in frame_ids:
            frame_probs = component_table.loc[frame_id] if component_table is not None and frame_id in component_table.index else None
            images.append(render_frame_image(situation, frame_id, component_name, frame_probs))

        output_path = output_dir / f"{component_name}.gif"
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=40,
            loop=0,
        )

    print(f"Saved Hawkeye GIFs to {output_dir}")


if __name__ == "__main__":
    main()
