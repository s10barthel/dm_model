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
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    PHYSICAL_XPASS_SOURCE,
    load_runtime_physical_xpass_visualization_table,
    physical_xpass_metric,
)
from project_config import (
    COMPONENT_DIR,
    PROJECT_ROOT,
    SKILLCORNER_VISUALIZATION_DIR,
    generate_run_id,
    get_skillcorner_component_run_root,
    get_runtime_physical_xpass_dir,
    load_run_metadata,
    resolve_named_component_run_id,
    write_run_metadata,
)
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--show-physical-xpass", action="store_true", help="Render cached runtime physical xPass.")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass cache override.")
    parser.add_argument("--max-xpass", "--max_xpass", dest="max_xpass", action="store_true", help="Use max physical xPass columns for visualization.")
    parser.add_argument("--topmean-xpass", "--topmean_xpass", dest="topmean_xpass", action="store_true", help="Use top-N-mean physical xPass columns for visualization.")
    parser.add_argument("--top10mean-xpass", "--top10mean_xpass", dest="top10mean_xpass", action="store_true", help="Deprecated alias for --topmean-xpass.")
    add_component_selection_args(parser)
    parser.add_argument("--run-id", help="Pin the created SkillCorner visualization run id. Default: auto-generate one.")
    parser.add_argument("--output-dir", default=str(SKILLCORNER_VISUALIZATION_DIR))
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--output", choices=["png", "mp4", "gif"], default="png")
    parser.add_argument("--only-first", action="store_true", help="In PNG mode, render only the first possession frame.")
    parser.add_argument("--only-last", action="store_true", help="In PNG mode, render only the last possession frame.")
    args = parser.parse_args(argv)
    if args.output != "png" and (args.only_first or args.only_last):
        parser.error("--only-first/--only-last are only valid with --output png.")
    if args.only_first and args.only_last:
        parser.error("--only-first and --only-last cannot be combined.")
    return args


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

    if component_name == "physical_xpass":
        row = _row_for_frame(component_tables[component_name], frame_id)
        return pd.Series(dtype=float) if row is None else pd.to_numeric(row, errors="coerce").dropna().astype(float).sort_values(ascending=False)

    return build_visualization_probs(_row_for_frame(component_tables[component_name], frame_id))


def output_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "output", "gif" if getattr(args, "gif", False) else "png"))


def resolve_skillcorner_png_frames(frame_ids: list[int], args: argparse.Namespace) -> list[dict[str, object]]:
    if not frame_ids:
        return []
    first_frame = int(frame_ids[0])
    last_frame = int(frame_ids[-1])
    if bool(getattr(args, "only_last", False)):
        return [{"label": "last", "frame_id": last_frame}]
    if bool(getattr(args, "only_first", False)) or first_frame == last_frame:
        return [{"label": "first", "frame_id": first_frame}]
    return [{"label": "first", "frame_id": first_frame}, {"label": "last", "frame_id": last_frame}]


def render_possession(
    args: argparse.Namespace,
    context,
    component_tables: dict[str, pd.DataFrame],
    possession_index: int,
    output_root: Path,
    rendered_components: list[str],
    physical_cache_dir: str | Path | None = None,
    physical_xpass_metric_name: str | None = None,
) -> tuple[Path, dict[str, object]]:
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

    if bool(getattr(args, "show_physical_xpass", False)):
        physical_frame_ids = (
            [int(index) for index in possession_component_tables["pass_success"].index.tolist()]
            if not possession_component_tables["pass_success"].empty
            else frame_ids
        )
        possession_component_tables["physical_xpass"] = load_runtime_physical_xpass_visualization_table(
            physical_cache_dir,
            str(possession.match_id),
            physical_frame_ids,
            metric=physical_xpass_metric_name,
        )

    selected_output_mode = output_mode(args)
    output_paths: list[str] = []
    selected_frames: list[dict[str, object]] = []
    if selected_output_mode == "png":
        selected_frames = resolve_skillcorner_png_frames(frame_ids, args)
        for component_name in component_names:
            for frame_selection in selected_frames:
                frame_id = int(frame_selection["frame_id"])
                probs = _probs_for_component_frame(component_name, possession_component_tables, frame_id)
                image = render_frame_image(
                    possession,
                    frame_id,
                    component_name,
                    probs,
                    show_trajectories=args.show_trajectories,
                )
                output_path = output_dir / f"{component_name}_{frame_selection['label']}.png"
                image.save(output_path)
                output_paths.append(str(output_path.resolve()))
    else:
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

            output_path = output_dir / f"{component_name}.{selected_output_mode}"
            save_animation(iter_component_images(), output_path, fps=float(possession.fps), gif=selected_output_mode == "gif")
            output_paths.append(str(output_path.resolve()))

    return output_dir, {
        "frame_ids": frame_ids,
        "selected_frames": selected_frames,
        "output_paths": output_paths,
    }


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
    physical_cache_dir = args.physical_cache_dir or str(get_runtime_physical_xpass_dir("skillcorner"))
    selected_physical_xpass_metric = physical_xpass_metric(args)
    rendered_components = list(component_selection.rendered_components)
    if bool(args.show_physical_xpass):
        rendered_components.append("physical_xpass")

    output_dirs: list[Path] = []
    rendered_possessions: list[dict[str, object]] = []
    selected_output_mode = output_mode(args)
    for possession_index in resolve_indices(args):
        output_dir, render_info = render_possession(
            args=args,
            context=context,
            component_tables=component_tables,
            possession_index=possession_index,
            output_root=output_root,
            rendered_components=rendered_components,
            physical_cache_dir=physical_cache_dir,
            physical_xpass_metric_name=selected_physical_xpass_metric,
        )
        output_dirs.append(output_dir)
        rendered_possessions.append(
            {
                "match_id": str(args.match_id),
                "index": int(possession_index),
                "frame_ids": render_info["frame_ids"],
                "selected_frames": render_info["selected_frames"],
                "output_dir": str(output_dir.resolve()),
                "output_paths": render_info["output_paths"],
            }
        )
        print(f"Saved SkillCorner {selected_output_mode} visualizations to {output_dir}")

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
        "output": selected_output_mode,
        "only_first": bool(getattr(args, "only_first", False)),
        "only_last": bool(getattr(args, "only_last", False)),
        "show_trajectories": bool(args.show_trajectories),
        "source_models": component_metadata.get("models", {}),
        "requested_component_groups": component_selection.requested_component_groups,
        "disabled_component_groups": component_selection.disabled_component_groups,
        "rendered_components": component_selection.rendered_components,
        "show_physical_xpass": bool(args.show_physical_xpass),
        "physical_xpass_hash_policy": PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
        "physical_xpass_lookup_policy": "dataset_event_frame_player_only",
        "physical_xpass_checkpoint_source": None,
        "physical_xpass_runtime_source": PHYSICAL_XPASS_SOURCE,
        "physical_xpass_metric": selected_physical_xpass_metric,
        "physical_cache_dir": str(physical_cache_dir),
        "physical_xpass_output_paths": [str(path.resolve()) for path in sorted(output_root.rglob("physical_xpass.*"))],
        "disabled_components": component_selection.disabled_components,
    }
    metadata_path = write_run_metadata(output_root, metadata)
    print(f"Saved SkillCorner {selected_output_mode} visualizations for {len(output_dirs)} possession(s).")
    print(f"SkillCorner visualization run id: {visualization_run_id}")
    print(f"SkillCorner visualization metadata: {metadata_path}")


if __name__ == "__main__":
    main()
