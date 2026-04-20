from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from datatools.benchmark import (
    COMPONENT_COLUMNS,
    build_benchmark_component_tables,
    build_benchmark_state,
    build_benchmark_visualization_probs,
    load_benchmark_component_run,
    load_benchmark_modification_data,
    resolve_benchmark_component_states,
)
from datatools.viz_helpers import compute_pass_score, figure_to_rgb_image
from datatools.viz_snapshot import SnapshotVisualizer
from project_config import (
    DATA_ROOT,
    PROJECT_ROOT,
    get_benchmark_component_run_root,
    resolve_named_component_run_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "benchmark"))
    parser.add_argument("--modification", action="append", type=int, help="Restrict visualization to one or more benchmark modifications.")
    parser.add_argument("--game-state", action="append", type=int, choices=[1, 2], help="Restrict visualization to one or more benchmark game states.")
    parser.add_argument("--component-run-id", default=None, help="Optional versioned benchmark component run id.")
    parser.add_argument("--component-dir", default=None, help="Optional explicit benchmark component-run root override.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT / "visualizations" / "benchmark"))
    parser.add_argument("--show-trajectories", action="store_true")
    return parser.parse_args()


def render_state_image(
    state,
    component_name: str,
    probs: pd.Series | None,
    show_trajectories: bool = False,
) -> Image.Image:
    frame_id = int(state.frame_meta.index.min())
    snapshot = state.tracking.loc[frame_id:frame_id].copy()
    ball_xy = snapshot[["ball_x", "ball_y"]].copy()
    ball_velocity_xy = None
    if {"ball_vx", "ball_vy"} <= set(snapshot.columns):
        ball_vx = float(snapshot["ball_vx"].iloc[-1])
        ball_vy = float(snapshot["ball_vy"].iloc[-1])
        if pd.notna(ball_vx) and pd.notna(ball_vy):
            ball_velocity_xy = (ball_vx, ball_vy)

    if probs is None or probs.empty:
        component_probs = pd.Series(dtype=float)
    else:
        attack_targets = [
            player_id
            for player_id in probs.index
            if isinstance(player_id, str) and player_id.startswith("home_")
        ]
        component_probs = probs.loc[attack_targets].dropna().sort_values(ascending=False)

    possessor_object_id = str(state.frame_meta.at[frame_id, "possessor_object_id"])
    visualizer = SnapshotVisualizer(
        snapshot=snapshot,
        ball_xy=ball_xy,
        player_annots=component_probs if not component_probs.empty else None,
        ball_velocity_xy=ball_velocity_xy,
        show_velocities=True,
        show_trajectories=show_trajectories,
        highlight_players={possessor_object_id: "#ffd400"},
        style="pitchcontrol",
        attacking_team_prefix="home",
    )

    title = (
        f"Modification {state.modification_id} | "
        f"Game State {state.game_state_id} | "
        f"{component_name.replace('_', ' ').title()}"
    )
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


def _row_for_frame(component_table: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if component_table.empty or frame_id not in component_table.index:
        return None

    row = component_table.loc[frame_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _probs_for_component_frame(
    component_name: str,
    component_tables: dict[str, pd.DataFrame],
    frame_id: int,
) -> pd.Series:
    if component_name == "pass_score":
        return compute_pass_score(
            pass_success=build_benchmark_visualization_probs(_row_for_frame(component_tables["pass_success"], frame_id)),
            outcome_scoring_success=build_benchmark_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_success"], frame_id)
            ),
            outcome_scoring_failure=build_benchmark_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_failure"], frame_id)
            ),
            outcome_conceding_success=build_benchmark_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_success"], frame_id)
            ),
            outcome_conceding_failure=build_benchmark_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_failure"], frame_id)
            ),
        )

    return build_benchmark_visualization_probs(_row_for_frame(component_tables[component_name], frame_id))


def main() -> None:
    args = parse_args()
    if args.component_dir:
        component_dir = Path(args.component_dir)
    else:
        component_run_id = resolve_named_component_run_id("benchmark_component", args.component_run_id, required=True)
        component_dir = get_benchmark_component_run_root(component_run_id)

    component_export, component_metadata = load_benchmark_component_run(component_dir)
    state_pairs = resolve_benchmark_component_states(
        component_export,
        metadata=component_metadata,
        requested_modifications=args.modification,
        requested_game_states=args.game_state,
    )

    output_root = Path(args.output_dir)
    component_names = [*COMPONENT_COLUMNS, "pass_score"]

    for modification_id, game_state_id in state_pairs:
        modification_data = load_benchmark_modification_data(modification_id, args.input_dir)
        state, _, _ = build_benchmark_state(
            modification_data[f"game_state_{game_state_id}"],
            modification_id=int(modification_id),
            game_state_id=int(game_state_id),
            higher_state_id=int(modification_data["higher_state_id"]),
            build_graphs=False,
        )
        component_tables = build_benchmark_component_tables(component_export, state)
        frame_id = int(state.frame_meta.index.min())
        output_dir = output_root / f"modification_{modification_id}" / f"game_state_{game_state_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        for component_name in component_names:
            probs = _probs_for_component_frame(component_name, component_tables, frame_id)
            image = render_state_image(
                state,
                component_name,
                probs,
                show_trajectories=args.show_trajectories,
            )
            image.save(output_dir / f"{component_name}.png")

        print(f"Saved benchmark visualizations to {output_dir}")


if __name__ == "__main__":
    main()
