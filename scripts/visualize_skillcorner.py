from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
from datatools.viz_helpers import compute_pass_score, figure_to_rgb_image, save_animation
from datatools.viz_snapshot import SnapshotVisualizer
from project_config import (
    COMPONENT_DIR,
    PROJECT_ROOT,
    SKILLCORNER_VISUALIZATION_DIR,
    generate_run_id,
    get_skillcorner_component_run_root,
    load_run_metadata,
    resolve_named_component_run_id,
    write_run_metadata,
)
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument(
        "--index",
        action="append",
        type=int,
        required=True,
        help="SkillCorner player_possession index to visualize. Repeat to visualize multiple possessions.",
    )
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "skillcorner_data"))
    parser.add_argument("--component-run-id", default=None, help="Optional versioned SkillCorner component run id.")
    parser.add_argument("--component-dir", default=None, help="Optional explicit component-run root override.")
    add_component_selection_args(parser)
    parser.add_argument("--run-id", help="Pin the created SkillCorner visualization run id. Default: auto-generate one.")
    parser.add_argument("--output-dir", default=str(SKILLCORNER_VISUALIZATION_DIR))
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Save GIFs instead of the default MP4 animations.")
    return parser.parse_args()


def resolve_indices(args: argparse.Namespace) -> list[int]:
    seen: set[int] = set()
    indices: list[int] = []
    for index in args.index:
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


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

    image = figure_to_rgb_image(fig, dpi=150, tight=False)
    plt.close(fig)
    return image


def _row_for_frame(component_table: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if component_table.empty or frame_id not in component_table.index:
        return None

    row = component_table.loc[frame_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _probs_for_component_frame(
    component_name: str,
    component_tables: dict[str, pd.DataFrame],
    frame_id: int,
) -> pd.Series:
    if component_name == "pass_score":
        return compute_pass_score(
            pass_success=build_visualization_probs(_row_for_frame(component_tables["pass_success"], frame_id)),
            outcome_scoring_success=build_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_success"], frame_id)
            ),
            outcome_scoring_failure=build_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_failure"], frame_id)
            ),
            outcome_conceding_success=build_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_success"], frame_id)
            ),
            outcome_conceding_failure=build_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_failure"], frame_id)
            ),
        )

    return build_visualization_probs(_row_for_frame(component_tables[component_name], frame_id))


def render_possession(
    args: argparse.Namespace,
    context,
    component_tables: dict[str, pd.DataFrame],
    possession_index: int,
    output_root: Path,
    rendered_components: list[str],
) -> Path:
    output_dir = output_root / str(args.match_id) / str(possession_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    possession, _ = build_skillcorner_possession(context, possession_index)
    if possession.frame_meta.empty:
        raise ValueError(f"Player-possession {possession_index} in match {args.match_id} does not have any addressable frames.")

    possession_component_tables = {name: table.copy() for name, table in component_tables.items()}
    frame_ids = [int(frame_id) for frame_id in possession.frame_meta.index.tolist()]
    component_names = list(rendered_components)

    for component_name in COMPONENT_COLUMNS:
        component_table = possession_component_tables[component_name]
        component_table = component_table.loc[component_table["index"] == int(possession_index)].copy()
        if not component_table.empty:
            component_table = component_table.sort_values("frame").drop_duplicates(subset=["frame"], keep="last")
            component_table = component_table.set_index("frame")
        possession_component_tables[component_name] = component_table

    for component_name in component_names:
        def iter_component_images():
            for frame_id in frame_ids:
                probs = _probs_for_component_frame(component_name, possession_component_tables, frame_id)
                yield render_frame_image(
                    possession,
                    frame_id,
                    component_name,
                    probs,
                    show_trajectories=args.show_trajectories,
                )

        suffix = "gif" if args.gif else "mp4"
        output_path = output_dir / f"{component_name}.{suffix}"
        save_animation(iter_component_images(), output_path, fps=float(possession.fps), gif=args.gif)

    return output_dir


def main() -> None:
    args = parse_args()
    component_selection = resolve_component_selection(args)
    visualization_run_id = args.run_id or generate_run_id("skillcorner_visualization")
    output_parent = Path(args.output_dir)
    output_root = output_parent / visualization_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    component_run_id = None
    if args.component_dir:
        component_dir = Path(args.component_dir)
    else:
        component_run_id = resolve_named_component_run_id("skillcorner_component", args.component_run_id, required=False)
        component_dir = (
            get_skillcorner_component_run_root(component_run_id) if component_run_id is not None else COMPONENT_DIR / "skillcorner"
        )

    context = build_skillcorner_match_context(args.match_id, args.input_dir)
    component_tables = load_skillcorner_component_tables(component_dir, args.match_id)
    component_metadata = load_run_metadata(component_dir, required=False) or {}

    output_dirs: list[Path] = []
    rendered_possessions: list[dict[str, object]] = []
    for possession_index in resolve_indices(args):
        output_dir = render_possession(
            args=args,
            context=context,
            component_tables=component_tables,
            possession_index=possession_index,
            output_root=output_root,
            rendered_components=component_selection.rendered_components,
        )
        output_dirs.append(output_dir)
        rendered_possessions.append(
            {
                "match_id": str(args.match_id),
                "index": int(possession_index),
                "output_dir": str(output_dir.resolve()),
                "output_paths": [str(path.resolve()) for path in sorted(output_dir.glob("*")) if path.is_file()],
            }
        )
        print(f"Saved SkillCorner animations to {output_dir}")

    metadata = {
        "run_id": visualization_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "script": Path(__file__).name,
        "output_parent": str(output_parent),
        "output_dir": str(output_root.resolve()),
        "status": "completed",
        "component_run_id": component_run_id,
        "component_dir": str(component_dir.resolve()),
        "component_metadata_run_id": component_metadata.get("run_id"),
        "match_id": str(args.match_id),
        "requested_indices": [int(value) for value in (args.index or [])],
        "rendered_possessions": rendered_possessions,
        "input_dir": str(Path(args.input_dir).resolve()),
        "gif": bool(args.gif),
        "show_trajectories": bool(args.show_trajectories),
        "source_models": component_metadata.get("models", {}),
        "requested_component_groups": component_selection.requested_component_groups,
        "disabled_component_groups": component_selection.disabled_component_groups,
        "rendered_components": component_selection.rendered_components,
        "disabled_components": component_selection.disabled_components,
    }
    metadata_path = write_run_metadata(output_root, metadata)
    print(f"Saved SkillCorner animations for {len(output_dirs)} possession(s).")
    print(f"SkillCorner visualization run id: {visualization_run_id}")
    print(f"SkillCorner visualization metadata: {metadata_path}")


if __name__ == "__main__":
    main()
