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

from datatools.hawkeye import (
    COMPONENT_COLUMNS,
    build_hawkeye_component_tables,
    build_hawkeye_situation,
    build_hawkeye_visualization_probs,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    load_hawkeye_ball,
    load_hawkeye_component_run,
    load_hawkeye_tracking,
    resolve_hawkeye_component_situation_ids,
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
    HAWKEYE_VISUALIZATION_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_hawkeye_component_run_root,
    get_runtime_physical_xpass_dir,
    resolve_named_component_run_id,
    write_run_metadata,
)
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--situation-id",
        action="append",
        help="Restrict visualization to one or more Hawkeye situation ids.",
    )
    parser.add_argument(
        "--tracking-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "centroid_data_team.csv"),
    )
    parser.add_argument(
        "--ball-csv",
        default=str(PROJECT_ROOT / "hawkeye_data" / "ball_data_selected.csv"),
    )
    parser.add_argument("--component-run-id", default=None, help="Optional versioned Hawkeye component run id.")
    parser.add_argument("--component-dir", default=None, help="Optional explicit component-run root override.")
    parser.add_argument("--show-trajectories", action="store_true")
    parser.add_argument("--show-physical-xpass", action="store_true", help="Render cached runtime physical xPass.")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass cache override.")
    parser.add_argument("--max-xpass", "--max_xpass", dest="max_xpass", action="store_true", help="Use max physical xPass columns for visualization.")
    parser.add_argument("--top10mean-xpass", "--top10mean_xpass", dest="top10mean_xpass", action="store_true", help="Use top-10%-mean physical xPass columns for visualization.")
    parser.add_argument("--output", choices=["png", "mp4", "gif"], default="png")
    parser.add_argument(
        "--time-norm",
        "--time_norm",
        dest="time_norm",
        action="append",
        type=float,
        help="BallReceipt-relative Hawkeye frame time to export in PNG mode. Repeat to export multiple frames.",
    )
    add_component_selection_args(parser)
    parser.add_argument("--run-id", help="Pin the created Hawkeye visualization run id. Default: auto-generate one.")
    parser.add_argument("--output-dir", default=str(HAWKEYE_VISUALIZATION_DIR))
    args = parser.parse_args(argv)
    if args.output != "png" and args.time_norm is not None:
        parser.error("--time-norm is only valid with --output png.")
    if args.output == "png" and args.time_norm is None:
        args.time_norm = [0.0]
    return args


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
        attack_targets = [
            player_id
            for player_id in probs.index
            if isinstance(player_id, str) and player_id.startswith(attacking_prefix)
        ]
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

    image = figure_to_rgb_image(fig, dpi=150, tight=False)
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
            pass_success=build_hawkeye_visualization_probs(_row_for_frame(component_tables["pass_success"], frame_id)),
            outcome_scoring_success=build_hawkeye_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_success"], frame_id)
            ),
            outcome_scoring_failure=build_hawkeye_visualization_probs(
                _row_for_frame(component_tables["outcome_scoring_failure"], frame_id)
            ),
            outcome_conceding_success=build_hawkeye_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_success"], frame_id)
            ),
            outcome_conceding_failure=build_hawkeye_visualization_probs(
                _row_for_frame(component_tables["outcome_conceding_failure"], frame_id)
            ),
        )

    return build_hawkeye_visualization_probs(_row_for_frame(component_tables[component_name], frame_id))


def output_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "output", "gif" if getattr(args, "gif", False) else "png"))


def requested_time_norms(args: argparse.Namespace) -> list[float]:
    values = getattr(args, "time_norm", None)
    return [0.0] if values is None else [float(value) for value in values]


def format_time_norm_label(time_norm: float) -> str:
    label = f"{float(time_norm):g}"
    return label.replace("-", "minus_").replace(".", "p")


def resolve_ballreceipt(situation_tracking: pd.DataFrame) -> float:
    if "BallReceipt" not in situation_tracking.columns:
        raise KeyError("Hawkeye tracking is missing BallReceipt, which is required for --output png.")
    values = pd.to_numeric(situation_tracking["BallReceipt"], errors="coerce").dropna().unique()
    if len(values) != 1:
        situation_id = situation_tracking["id"].iloc[0] if "id" in situation_tracking.columns and not situation_tracking.empty else "unknown"
        raise ValueError(
            f"Hawkeye situation {situation_id} must contain exactly one BallReceipt value for --output png, "
            f"found {values.tolist()}."
        )
    return float(values[0])


def resolve_hawkeye_png_frames(situation, ballreceipt: float, time_norms: list[float]) -> list[dict[str, object]]:
    frame_times = pd.to_numeric(situation.frame_meta["abs_time"], errors="coerce") - float(ballreceipt)
    frame_times = frame_times.dropna()
    if frame_times.empty:
        raise ValueError(f"Hawkeye situation {situation.situation_id} does not have any frame times for PNG export.")

    min_time = float(frame_times.min())
    max_time = float(frame_times.max())
    tolerance = 1e-9
    selections: list[dict[str, object]] = []
    for requested in time_norms:
        requested_float = float(requested)
        if requested_float < min_time - tolerance or requested_float > max_time + tolerance:
            raise ValueError(
                f"Requested time_norm {requested_float:g} is outside available range "
                f"[{min_time:g}, {max_time:g}] for Hawkeye situation {situation.situation_id}."
            )
        distances = (frame_times - requested_float).abs()
        frame_id = int(distances.sort_values(kind="stable").index[0])
        selections.append(
            {
                "label": f"time_norm_{format_time_norm_label(requested_float)}",
                "requested_time_norm": requested_float,
                "frame_id": frame_id,
                "resolved_time_norm": float(frame_times.loc[frame_id]),
                "abs_time": float(situation.frame_meta.at[frame_id, "abs_time"]),
            }
        )
    return selections


def main() -> None:
    args = parse_args()
    component_selection = resolve_component_selection(args)
    visualization_run_id = args.run_id or generate_run_id("hawkeye_visualization")
    output_parent = Path(args.output_dir)
    output_root = output_parent / visualization_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    component_run_id = None
    if args.component_dir:
        component_dir = Path(args.component_dir)
    else:
        component_run_id = resolve_named_component_run_id("hawkeye_component", args.component_run_id, required=True)
        component_dir = get_hawkeye_component_run_root(component_run_id)

    component_export, component_metadata = load_hawkeye_component_run(component_dir)
    situation_ids = resolve_hawkeye_component_situation_ids(
        component_export,
        metadata=component_metadata,
        requested_ids=args.situation_id,
    )
    freeze_ballreceipt = bool(component_metadata.get("freeze_ballreceipt", True))
    physical_cache_dir = args.physical_cache_dir or str(get_runtime_physical_xpass_dir("hawkeye"))
    selected_physical_xpass_metric = physical_xpass_metric(args)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    component_names = list(component_selection.rendered_components)
    if bool(args.show_physical_xpass):
        component_names.append("physical_xpass")
    rendered_situations: list[dict[str, object]] = []
    selected_output_mode = output_mode(args)

    for situation_id in situation_ids:
        situation_tracking = tracking.loc[tracking["id"] == str(situation_id)].copy()
        if situation_tracking.empty:
            raise KeyError(f"Hawkeye situation id {situation_id} was not found in {args.tracking_csv}.")

        situation, _, _ = build_hawkeye_situation(
            situation_tracking,
            ball,
            freeze_ballreceipt=freeze_ballreceipt,
            build_graphs=False,
        )
        component_tables = build_hawkeye_component_tables(component_export, situation)
        if bool(args.show_physical_xpass):
            physical_frame_ids = (
                [int(index) for index in component_tables["pass_success"].index.tolist()]
                if not component_tables["pass_success"].empty
                else [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
            )
            component_tables["physical_xpass"] = load_runtime_physical_xpass_visualization_table(
                physical_cache_dir,
                str(situation.match_id),
                physical_frame_ids,
                metric=selected_physical_xpass_metric,
            )
        frame_ids = [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
        output_dir = output_root / str(situation_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[str] = []
        selected_frames: list[dict[str, object]] = []

        if selected_output_mode == "png":
            selected_frames = resolve_hawkeye_png_frames(
                situation,
                resolve_ballreceipt(situation_tracking),
                requested_time_norms(args),
            )
            for component_name in component_names:
                for frame_selection in selected_frames:
                    frame_id = int(frame_selection["frame_id"])
                    probs = _probs_for_component_frame(component_name, component_tables, frame_id)
                    image = render_frame_image(
                        situation,
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
                        probs = _probs_for_component_frame(component_name, component_tables, frame_id)
                        yield render_frame_image(
                            situation,
                            frame_id,
                            component_name,
                            probs,
                            show_trajectories=args.show_trajectories,
                        )

                output_path = output_dir / f"{component_name}.{selected_output_mode}"
                save_animation(iter_component_images(), output_path, fps=25.0, gif=selected_output_mode == "gif")
                output_paths.append(str(output_path.resolve()))

        print(f"Saved Hawkeye {selected_output_mode} visualizations to {output_dir}")
        rendered_situations.append(
            {
                "situation_id": str(situation_id),
                "frame_ids": frame_ids,
                "selected_frames": selected_frames,
                "output_dir": str(output_dir.resolve()),
                "output_paths": output_paths,
            }
        )

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
        "requested_situation_ids": [str(value) for value in (args.situation_id or [])],
        "rendered_situation_ids": [item["situation_id"] for item in rendered_situations],
        "rendered_situations": rendered_situations,
        "tracking_csv": str(Path(args.tracking_csv).resolve()),
        "ball_csv": str(Path(args.ball_csv).resolve()),
        "output": selected_output_mode,
        "time_norm": requested_time_norms(args) if selected_output_mode == "png" else [],
        "show_trajectories": bool(args.show_trajectories),
        "freeze_ballreceipt": freeze_ballreceipt,
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
    print(f"Hawkeye visualization run id: {visualization_run_id}")
    print(f"Hawkeye visualization metadata: {metadata_path}")


if __name__ == "__main__":
    main()
