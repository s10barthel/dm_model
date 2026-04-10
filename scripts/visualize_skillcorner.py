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
from PIL import Image

from datatools.skillcorner import (
    COMPONENT_COLUMNS,
    build_skillcorner_match_context,
    build_skillcorner_possession,
    build_visualization_probs,
    load_skillcorner_component_tables,
)
from datatools.viz_snapshot import SnapshotVisualizer
from project_config import COMPONENT_DIR, DATA_ROOT, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--index", type=int, required=True, help="SkillCorner player_possession index to visualize.")
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "skillcorner_data"))
    parser.add_argument("--component-dir", default=str(COMPONENT_DIR / "skillcorner"))
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations" / "skillcorner"))
    parser.add_argument("--show-trajectories", action="store_true")
    return parser.parse_args()


def render_frame_image(
    possession,
    frame_id: int,
    component_name: str,
    probs: pd.Series | None,
    show_trajectories: bool = False,
) -> Image.Image:
    history_frames = max(int(round(possession.fps)) - 1, 0)
    snapshot_start = max(frame_id - history_frames, int(possession.tracking.index.min()))
    snapshot = possession.tracking.loc[snapshot_start:frame_id].copy()
    ball_xy = snapshot[["ball_x", "ball_y"]].copy()
    frame_info = possession.frame_meta.loc[frame_id]
    attacking_prefix = "home"

    if probs is None or probs.empty:
        component_probs = pd.Series(dtype=float)
    else:
        def player_visible(player_id: str) -> bool:
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            if x_col not in snapshot.columns or y_col not in snapshot.columns:
                return False
            return not snapshot[[x_col, y_col]].isna().all().all()

        attack_targets = [
            player_id
            for player_id in probs.index
            if isinstance(player_id, str)
            and player_id.startswith(f"{attacking_prefix}_")
            and player_visible(player_id)
        ]
        component_probs = probs.loc[attack_targets].dropna().sort_values(ascending=False)

    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_annots=component_probs if not component_probs.empty else None,
        show_velocities=True,
        show_trajectories=show_trajectories,
        highlight_players={str(frame_info["possessor_object_id"]): "#ffd400"},
        style="pitchcontrol",
        attacking_team_prefix=attacking_prefix,
    )

    title = f"{possession.match_id} | possession {possession.event_index} | frame {frame_id} | {component_name.replace('_', ' ').title()}"
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

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def _row_for_frame(component_table: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if component_table.empty or frame_id not in component_table.index:
        return None

    row = component_table.loc[frame_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / str(args.match_id) / str(args.index)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_skillcorner_match_context(args.match_id, args.input_dir)
    possession, _ = build_skillcorner_possession(context, args.index)
    if possession.frame_meta.empty:
        raise ValueError(f"Player-possession {args.index} in match {args.match_id} does not have any addressable frames.")

    component_tables = load_skillcorner_component_tables(args.component_dir, args.match_id)
    frame_ids = [int(frame_id) for frame_id in possession.frame_meta.index.tolist()]
    duration_ms = max(1, int(round(1000 / possession.fps)))

    for component_name in COMPONENT_COLUMNS:
        component_table = component_tables[component_name]
        component_table = component_table.loc[component_table["index"] == int(args.index)].copy()
        if not component_table.empty:
            component_table = component_table.sort_values("frame").drop_duplicates(subset=["frame"], keep="last")
            component_table = component_table.set_index("frame")

        images: list[Image.Image] = []
        for frame_id in frame_ids:
            row = _row_for_frame(component_table, frame_id)
            probs = build_visualization_probs(row)
            images.append(
                render_frame_image(
                    possession,
                    frame_id,
                    component_name,
                    probs,
                    show_trajectories=args.show_trajectories,
                )
            )

        output_path = output_dir / f"{component_name}.gif"
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )

    print(f"Saved SkillCorner GIFs to {output_dir}")


if __name__ == "__main__":
    main()
