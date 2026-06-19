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

from datatools.benchmark import (
    build_benchmark_component_tables,
    build_benchmark_state,
    build_benchmark_visualization_probs,
    load_benchmark_component_run,
    load_benchmark_modification_data,
    resolve_benchmark_component_states,
)
from datatools.viz_helpers import compute_pass_score, figure_to_rgb_image
from datatools.viz_snapshot import SnapshotVisualizer
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    PHYSICAL_XPASS_SOURCE,
    load_runtime_physical_xpass_visualization_table,
    physical_xpass_metric,
)
from project_config import (
    BENCHMARK_VISUALIZATION_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_benchmark_component_run_root,
    get_runtime_physical_xpass_dir,
    resolve_named_component_run_id,
    write_run_metadata,
)
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "benchmark"))
    parser.add_argument("--modification", action="append", type=int, help="Restrict visualization to one or more benchmark modifications.")
    parser.add_argument("--game-state", action="append", type=int, choices=[1, 2], help="Restrict visualization to one or more benchmark game states.")
    parser.add_argument("--component-run-id", default=None, help="Optional versioned benchmark component run id.")
    parser.add_argument("--component-dir", default=None, help="Optional explicit benchmark component-run root override.")
    parser.add_argument("--show-physical-xpass", action="store_true", help="Render cached runtime physical xPass.")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass cache override.")
    parser.add_argument("--max-xpass", "--max_xpass", dest="max_xpass", action="store_true", help="Use max physical xPass columns for visualization.")
    parser.add_argument("--topmean-xpass", "--topmean_xpass", dest="topmean_xpass", action="store_true", help="Use top-N-mean physical xPass columns for visualization.")
    parser.add_argument("--top10mean-xpass", "--top10mean_xpass", dest="top10mean_xpass", action="store_true", help="Deprecated alias for --topmean-xpass.")
    add_component_selection_args(parser)
    parser.add_argument("--run-id", help="Pin the created benchmark visualization run id. Default: auto-generate one.")
    parser.add_argument("--output-dir", default=str(BENCHMARK_VISUALIZATION_DIR))
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


def combine_state_images(top_image: Image.Image, bottom_image: Image.Image) -> Image.Image:
    top_rgb = top_image.convert("RGB")
    bottom_rgb = bottom_image.convert("RGB")
    width = max(top_rgb.width, bottom_rgb.width)
    height = top_rgb.height + bottom_rgb.height
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(top_rgb, (0, 0))
    combined.paste(bottom_rgb, (0, top_rgb.height))
    return combined


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

    if component_name == "physical_xpass":
        row = _row_for_frame(component_tables[component_name], frame_id)
        return pd.Series(dtype=float) if row is None else pd.to_numeric(row, errors="coerce").dropna().astype(float).sort_values(ascending=False)

    return build_benchmark_visualization_probs(_row_for_frame(component_tables[component_name], frame_id))


def main() -> None:
    args = parse_args()
    component_selection = resolve_component_selection(args)
    visualization_run_id = args.run_id or generate_run_id("benchmark_visualization")
    output_parent = Path(args.output_dir)
    output_root = output_parent / visualization_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    component_run_id = None
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

    physical_cache_dir = args.physical_cache_dir or str(get_runtime_physical_xpass_dir("benchmark"))
    selected_physical_xpass_metric = physical_xpass_metric(args)
    component_names = list(component_selection.rendered_components)
    if bool(args.show_physical_xpass):
        component_names.append("physical_xpass")
    pairs_by_modification: dict[int, set[int]] = {}
    for modification_id, game_state_id in state_pairs:
        pairs_by_modification.setdefault(int(modification_id), set()).add(int(game_state_id))

    rendered_modifications: list[dict[str, object]] = []
    skipped_modifications: list[dict[str, object]] = []

    for modification_id in sorted(pairs_by_modification):
        available_game_states = pairs_by_modification[modification_id]
        if not {1, 2} <= available_game_states:
            skipped_modifications.append(
                {
                    "modification": int(modification_id),
                    "available_game_states": sorted(int(value) for value in available_game_states),
                    "reason": "paired_visualization_requires_game_states_1_and_2",
                }
            )
            print(
                f"Skipping modification_{modification_id}: paired benchmark visualization requires "
                "both game_state_1 and game_state_2."
            )
            continue

        modification_data = load_benchmark_modification_data(modification_id, args.input_dir)
        state_contexts: dict[int, tuple[object, dict[str, pd.DataFrame], int]] = {}
        for game_state_id in (1, 2):
            state, _, _ = build_benchmark_state(
                modification_data[f"game_state_{game_state_id}"],
                modification_id=int(modification_id),
                game_state_id=int(game_state_id),
                higher_state_id=int(modification_data["higher_state_id"]),
                build_graphs=False,
            )
            state_contexts[game_state_id] = (
                state,
                build_benchmark_component_tables(component_export, state),
                int(state.frame_meta.index.min()),
            )
            if bool(args.show_physical_xpass):
                frame_id = int(state.frame_meta.index.min())
                state_contexts[game_state_id][1]["physical_xpass"] = load_runtime_physical_xpass_visualization_table(
                    physical_cache_dir,
                    str(state.match_id),
                    [frame_id],
                    metric=selected_physical_xpass_metric,
                )

        output_paths: list[str] = []
        modification_output_dir = output_root / f"modification_{modification_id}"
        modification_output_dir.mkdir(parents=True, exist_ok=True)
        for component_name in component_names:
            state_images: dict[int, Image.Image] = {}
            for game_state_id in (1, 2):
                state, component_tables, frame_id = state_contexts[game_state_id]
                probs = _probs_for_component_frame(component_name, component_tables, frame_id)
                state_images[game_state_id] = render_state_image(
                    state,
                    component_name,
                    probs,
                    show_trajectories=args.show_trajectories,
                )

            output_path = modification_output_dir / f"{component_name}.png"
            combine_state_images(state_images[1], state_images[2]).save(output_path)
            output_paths.append(str(output_path.resolve()))

        print(f"Saved benchmark visualizations for modification_{modification_id} to {modification_output_dir}")
        rendered_modifications.append(
            {
                "modification": int(modification_id),
                "game_states": [1, 2],
                "output_dir": str(modification_output_dir.resolve()),
                "output_paths": output_paths,
            }
        )

    if not rendered_modifications:
        raise ValueError("No benchmark modifications with both game_state_1 and game_state_2 were available to visualize.")

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
        "requested_modifications": [int(value) for value in (args.modification or [])],
        "requested_game_states": [int(value) for value in (args.game_state or [])],
        "rendered_modifications": rendered_modifications,
        "skipped_modifications": skipped_modifications,
        "input_dir": str(Path(args.input_dir).resolve()),
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
    print(f"Benchmark visualization run id: {visualization_run_id}")
    print(f"Benchmark visualization metadata: {metadata_path}")


if __name__ == "__main__":
    main()
