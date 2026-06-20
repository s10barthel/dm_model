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
import torch
from PIL import Image

from datatools.hawkeye import (
    build_hawkeye_situation,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    load_hawkeye_ball,
    load_hawkeye_tracking,
    resolve_situation_ids as resolve_hawkeye_situation_ids,
)
from inference import inference_gnn
from models.utils import load_model, resolve_model_selection, validate_model_graph_schemas
from physical_pass_model import (
    PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
    format_physical_xpass_cache_summary,
    inference_uses_physical_xpass,
    load_runtime_physical_xpass_visualization_table,
    model_uses_physical_xpass,
    physical_xpass_inference_lookup_config,
    physical_xpass_metric,
    physical_xpass_source,
    resolve_physical_num_workers,
    summarize_physical_xpass_cache_usage,
)
from datatools.viz_helpers import compute_pass_score, figure_to_rgb_image, save_animation
from datatools.viz_snapshot import SnapshotVisualizer
from project_config import (
    HAWKEYE_VISUALIZATION_DIR,
    PROJECT_ROOT,
    generate_run_id,
    get_runtime_physical_xpass_dir,
    write_run_metadata,
)
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection
from scripts.visualize_hawkeye import (
    output_mode,
    requested_time_norms,
    resolve_ballreceipt,
    resolve_hawkeye_png_frames,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--situation-id",
        action="append",
        help="Hawkeye situation id to visualize. Repeat to visualize multiple situations.",
    )
    parser.add_argument("--action-id", action="append", help="Alias for --situation-id. Repeat to visualize multiple situations.")
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
    parser.add_argument("--show-physical-xpass", action="store_true", help="Render cached runtime physical xPass.")
    parser.add_argument("--output", choices=["png", "mp4", "gif"], default="png")
    parser.add_argument(
        "--time-norm",
        "--time_norm",
        dest="time_norm",
        action="append",
        type=float,
        help="BallReceipt-relative Hawkeye frame time to export in PNG mode. Repeat to export multiple frames.",
    )
    parser.add_argument("--bundle-id", default=None)
    parser.add_argument("--action-intent-model-id")
    parser.add_argument("--pass-intent-model-id")
    parser.add_argument("--pass-success-model-id")
    parser.add_argument("--outcome-scoring-model-id")
    parser.add_argument("--outcome-conceding-model-id")
    parser.add_argument("--use-physical-xpass", "--use_physical_xpass", dest="use_physical_xpass", action="store_true", help="Blend pass-success inference with physical xPass.")
    parser.add_argument("--max-xpass", "--max_xpass", dest="max_xpass", action="store_true", help="Use max physical xPass columns for inference blending.")
    parser.add_argument("--topmean-xpass", "--topmean_xpass", dest="topmean_xpass", action="store_true", help="Use top-N-mean physical xPass columns for inference blending and physical xPass rendering.")
    parser.add_argument("--top10mean-xpass", "--top10mean_xpass", dest="top10mean_xpass", action="store_true", help="Deprecated alias for --topmean-xpass.")
    parser.add_argument("--physical-cache-dir", help="Runtime physical xPass cache override.")
    parser.add_argument("--no-physical-cache", action="store_true", help="Disable runtime physical xPass cache.")
    parser.add_argument("--refresh-physical-cache", action="store_true", help="Deprecated during inference; run scripts/generate_physical_xpass.py to refresh/fill caches.")
    parser.add_argument("--physical-num-workers", "--num-workers", dest="physical_num_workers", default="auto")
    parser.add_argument("--physical-worker-thread-limit", "--worker-thread-limit", dest="physical_worker_thread_limit", type=int, default=1)
    parser.add_argument("--physical-batch-size", type=int, default=16)
    add_component_selection_args(parser)
    parser.add_argument("--run-id", help="Pin the created Hawkeye visualization run id. Default: auto-generate one.")
    parser.add_argument("--output-dir", default=str(HAWKEYE_VISUALIZATION_DIR))
    parser.set_defaults(freeze_ballreceipt=True)
    args = parser.parse_args(argv)
    if args.output != "png" and args.time_norm is not None:
        parser.error("--time-norm is only valid with --output png.")
    if args.output == "png" and args.time_norm is None:
        args.time_norm = [0.0]
    try:
        resolve_physical_num_workers(args.physical_num_workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.physical_worker_thread_limit < 1:
        parser.error("--physical-worker-thread-limit must be positive.")
    if args.physical_batch_size < 1:
        parser.error("--physical-batch-size must be positive.")
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

    image = figure_to_rgb_image(fig, dpi=150, tight=False)
    plt.close(fig)
    return image


def render_situation(
    situation_id: str,
    tracking: pd.DataFrame,
    ball: pd.DataFrame,
    model_specs: dict[str, object],
    graph_schema: dict[str, object],
    args: argparse.Namespace,
    device: str,
    output_root: Path,
    rendered_components: list[str],
) -> tuple[Path, dict[str, object], dict[str, object]]:
    output_dir = output_root / str(situation_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    situation_tracking = tracking.loc[tracking["id"] == str(situation_id)].copy()
    if situation_tracking.empty:
        raise KeyError(f"Hawkeye situation id {situation_id} was not found in {args.tracking_csv}.")

    situation, _, _ = build_hawkeye_situation(
        situation_tracking,
        ball,
        freeze_ballreceipt=args.freeze_ballreceipt,
        add_v_edge_features=bool(graph_schema["add_v_edge_features"]),
    )
    components: dict[str, pd.DataFrame] = {}
    if situation.labels.numel() != 0 and situation.graph_features_0:
        if "action_intent" in model_specs:
            components["action_intent"], _ = inference_gnn(
                situation,
                model_specs["action_intent"],
                device=device,
                post_action=False,
            )
        if "pass_intent" in model_specs:
            components["pass_intent"], _ = inference_gnn(
                situation,
                model_specs["pass_intent"],
                device=device,
                post_action=False,
            )
        if "pass_success" in model_specs:
            components["pass_success"], _ = inference_gnn(
                situation,
                model_specs["pass_success"],
                device=device,
                post_action=False,
            )
        if "outcome_scoring" in model_specs:
            scoring_failure, scoring_success = inference_gnn(
                situation,
                model_specs["outcome_scoring"],
                device=device,
                post_action=False,
            )
            components["outcome_scoring_success"] = scoring_success
            components["outcome_scoring_failure"] = scoring_failure
        if "outcome_conceding" in model_specs:
            conceding_failure, conceding_success = inference_gnn(
                situation,
                model_specs["outcome_conceding"],
                device=device,
                post_action=False,
            )
            components["outcome_conceding_success"] = conceding_success
            components["outcome_conceding_failure"] = conceding_failure

    component_frames: dict[str, pd.DataFrame | None] = {
        component_name: components.get(component_name)
        for component_name in rendered_components
        if component_name != "pass_score"
    }
    if "pass_score" in rendered_components and all(
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

    if bool(getattr(args, "show_physical_xpass", False)):
        physical_frame_ids = (
            [int(index) for index in component_frames["pass_success"].index.tolist()]
            if component_frames.get("pass_success") is not None
            else [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
        )
        component_frames["physical_xpass"] = load_runtime_physical_xpass_visualization_table(
            args.physical_cache_dir,
            str(situation.match_id),
            physical_frame_ids,
            metric=physical_xpass_metric(args),
        )

    frame_ids = [int(frame_id) for frame_id in situation.frame_meta.index.tolist()]
    component_names = list(rendered_components)
    if bool(getattr(args, "show_physical_xpass", False)):
        component_names.append("physical_xpass")
    selected_output_mode = output_mode(args)
    output_paths: list[str] = []
    selected_frames: list[dict[str, object]] = []

    if selected_output_mode == "png":
        selected_frames = resolve_hawkeye_png_frames(
            situation,
            resolve_ballreceipt(situation_tracking),
            requested_time_norms(args),
        )
        for component_name in component_names:
            component_table = component_frames.get(component_name)
            for frame_selection in selected_frames:
                frame_id = int(frame_selection["frame_id"])
                frame_probs = component_table.loc[frame_id] if component_table is not None and frame_id in component_table.index else None
                image = render_frame_image(
                    situation,
                    frame_id,
                    component_name,
                    frame_probs,
                    show_trajectories=args.show_trajectories,
                )
                output_path = output_dir / f"{component_name}_{frame_selection['label']}.png"
                image.save(output_path)
                output_paths.append(str(output_path.resolve()))
    else:
        for component_name in component_names:
            component_table = component_frames.get(component_name)

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

            output_path = output_dir / f"{component_name}.{selected_output_mode}"
            save_animation(iter_component_images(), output_path, fps=25.0, gif=selected_output_mode == "gif")
            output_paths.append(str(output_path.resolve()))

    render_info = {
        "frame_ids": frame_ids,
        "selected_frames": selected_frames,
        "output_paths": output_paths,
    }
    render_info["physical_xpass_skipped_actions"] = getattr(situation, "physical_xpass_skipped_actions", {})
    return output_dir, getattr(situation, "physical_xpass_runtime_stats", {}), render_info


def main() -> None:
    args = parse_args()
    component_selection = resolve_component_selection(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    visualization_run_id = args.run_id or generate_run_id("hawkeye_visualization")
    output_parent = Path(args.output_dir)
    output_root = output_parent / visualization_run_id
    output_root.mkdir(parents=True, exist_ok=True)

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    requested_situation_ids = [str(value) for value in (args.situation_id or [])]
    requested_action_ids = [str(value) for value in (args.action_id or [])]
    situation_ids = resolve_hawkeye_situation_ids(
        tracking,
        requested_ids=requested_situation_ids + requested_action_ids,
    )
    explicit_model_ids = {
        "action_intent": args.action_intent_model_id,
        "pass_intent": args.pass_intent_model_id,
        "pass_success": args.pass_success_model_id,
        "outcome_scoring": args.outcome_scoring_model_id,
        "outcome_conceding": args.outcome_conceding_model_id,
    }
    required_model_tasks = [
        task
        for task in [
            "action_intent",
            "pass_intent",
            "pass_success",
            "outcome_scoring",
            "outcome_conceding",
        ]
        if task in component_selection.requested_component_groups
    ]
    model_ids, shared_context, _ = resolve_model_selection(
        required_tasks=required_model_tasks,
        bundle_id=args.bundle_id,
        explicit_model_ids=explicit_model_ids,
        require_feature_run_id=False,
        require_intended_receiver_mode=False,
        require_return_type=False,
        require_target_family=False,
    )
    selected_model_ids = {
        name: model_id
        for name, model_id in model_ids.items()
        if name in component_selection.requested_component_groups
    }
    model_specs = {name: load_model(model_id, device) for name, model_id in selected_model_ids.items()}
    missing = [name for name, model in model_specs.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(missing)}")
    no_physical_cache = bool(getattr(args, "no_physical_cache", False))
    refresh_physical_cache = bool(getattr(args, "refresh_physical_cache", False))
    physical_cache_dir = getattr(args, "physical_cache_dir", None) or str(get_runtime_physical_xpass_dir("hawkeye"))
    args.physical_cache_dir = physical_cache_dir
    selected_physical_xpass_metric = physical_xpass_metric(args)
    pass_success_model = model_specs.get("pass_success")
    if pass_success_model is not None and bool(getattr(args, "use_physical_xpass", False)):
        pass_success_model.args["inference_use_physical_xpass"] = True
        pass_success_model.args["max_xpass"] = bool(getattr(args, "max_xpass", False))
        use_topmean_xpass = bool(getattr(args, "topmean_xpass", False) or getattr(args, "top10mean_xpass", False))
        pass_success_model.args["topmean_xpass"] = use_topmean_xpass
        pass_success_model.args["top10mean_xpass"] = bool(getattr(args, "top10mean_xpass", False))
    if pass_success_model is not None and (model_uses_physical_xpass(pass_success_model.args) or inference_uses_physical_xpass(pass_success_model.args)):
        pass_success_model.args["physical_runtime_cache_disabled"] = no_physical_cache
        pass_success_model.args["physical_runtime_cache_refresh"] = False
        pass_success_model.args["physical_num_workers"] = getattr(args, "physical_num_workers", "auto")
        pass_success_model.args["physical_worker_thread_limit"] = int(getattr(args, "physical_worker_thread_limit", 1))
        pass_success_model.args["physical_batch_size"] = int(getattr(args, "physical_batch_size", 16))
        pass_success_model.args["physical_runtime_cache_read_only"] = True
        if not no_physical_cache:
            pass_success_model.args["physical_cache_dir"] = physical_cache_dir
    graph_schema = validate_model_graph_schemas(model_specs)

    output_dirs: list[Path] = []
    rendered_situations: list[dict[str, object]] = []
    physical_xpass_runtime_stats: dict[str, dict[str, object]] = {}
    physical_xpass_skipped_actions: dict[str, dict[str, object]] = {}
    selected_output_mode = output_mode(args)
    for situation_id in situation_ids:
        output_dir, runtime_physical_stats, render_info = render_situation(
            situation_id=situation_id,
            tracking=tracking,
            ball=ball,
            model_specs=model_specs,
            graph_schema=graph_schema,
            args=args,
            device=device,
            output_root=output_root,
            rendered_components=component_selection.rendered_components,
        )
        if runtime_physical_stats:
            physical_xpass_runtime_stats[str(situation_id)] = runtime_physical_stats
        physical_skip_stats = render_info.get("physical_xpass_skipped_actions")
        if physical_skip_stats:
            physical_xpass_skipped_actions[str(situation_id)] = physical_skip_stats
        output_dirs.append(output_dir)
        rendered_situations.append(
            {
                "situation_id": str(situation_id),
                "frame_ids": render_info["frame_ids"],
                "selected_frames": render_info["selected_frames"],
                "output_dir": str(output_dir.resolve()),
                "output_paths": render_info["output_paths"],
            }
        )
        print(f"Saved Hawkeye {selected_output_mode} visualizations to {output_dir}")

    metadata = {
        "run_id": visualization_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "script": Path(__file__).name,
        "output_parent": str(output_parent),
        "output_dir": str(output_root.resolve()),
        "status": "completed",
        "bundle_id": args.bundle_id,
        "model_ids": model_ids,
        "selected_model_ids": selected_model_ids,
        "intended_receiver_mode": shared_context.get("intended_receiver_mode"),
        "return_type": shared_context.get("return_type"),
        "target_family": shared_context.get("target_family"),
        "source_feature_run_ids": shared_context.get("source_feature_run_ids", {}),
        "source_intended_receiver_modes": shared_context.get("source_intended_receiver_modes", {}),
        "source_return_types": shared_context.get("source_return_types", {}),
        "source_target_families": shared_context.get("source_target_families", {}),
        "requested_component_groups": component_selection.requested_component_groups,
        "disabled_component_groups": component_selection.disabled_component_groups,
        "rendered_components": component_selection.rendered_components,
        "disabled_components": component_selection.disabled_components,
        "requested_situation_ids": requested_situation_ids,
        "requested_action_ids": requested_action_ids,
        "rendered_situation_ids": [item["situation_id"] for item in rendered_situations],
        "rendered_situations": rendered_situations,
        "physical_xpass_runtime_stats": physical_xpass_runtime_stats,
        "physical_xpass_skipped_actions": physical_xpass_skipped_actions,
        "physical_xpass_requested": bool(getattr(args, "use_physical_xpass", False)),
        "physical_xpass_hash_policy": PHYSICAL_XPASS_INFERENCE_HASH_POLICY,
        "physical_xpass_lookup_policy": "dataset_event_frame_player_only",
        "physical_xpass_checkpoint_source": physical_xpass_source(pass_success_model.args) if pass_success_model is not None else None,
        "physical_xpass_runtime_source": physical_xpass_inference_lookup_config(pass_success_model.args, cache_dir=physical_cache_dir)["source"] if pass_success_model is not None else None,
        "show_physical_xpass": bool(getattr(args, "show_physical_xpass", False)),
        "physical_xpass_metric": selected_physical_xpass_metric,
        "physical_cache_dir": None if no_physical_cache else physical_cache_dir,
        "physical_xpass_output_paths": [str(path.resolve()) for path in sorted(output_root.rglob("physical_xpass.*"))],
        "physical_cache_disabled": no_physical_cache,
        "refresh_physical_cache": refresh_physical_cache,
        "tracking_csv": str(Path(args.tracking_csv).resolve()),
        "ball_csv": str(Path(args.ball_csv).resolve()),
        "freeze_ballreceipt": bool(args.freeze_ballreceipt),
        "output": selected_output_mode,
        "time_norm": requested_time_norms(args) if selected_output_mode == "png" else [],
        "show_trajectories": bool(args.show_trajectories),
        "graph_schema": graph_schema,
    }
    metadata_path = write_run_metadata(output_root, metadata)
    physical_xpass_required = pass_success_model is not None and (
        model_uses_physical_xpass(pass_success_model.args) or inference_uses_physical_xpass(pass_success_model.args)
    )
    physical_xpass_cache_summary = summarize_physical_xpass_cache_usage(
        physical_xpass_required=physical_xpass_required,
        cache_disabled=no_physical_cache,
        refresh_requested=refresh_physical_cache,
        cache_dir=None if no_physical_cache else physical_cache_dir,
        runtime_stats=physical_xpass_runtime_stats,
        skipped_stats=physical_xpass_skipped_actions,
    )
    metadata["physical_xpass_cache_summary"] = physical_xpass_cache_summary
    metadata_path = write_run_metadata(output_root, metadata)
    print(f"Saved Hawkeye {selected_output_mode} visualizations for {len(output_dirs)} situation(s).")
    print(f"Hawkeye visualization run id: {visualization_run_id}")
    print(f"Hawkeye visualization metadata: {metadata_path}")
    print(format_physical_xpass_cache_summary(physical_xpass_cache_summary))


if __name__ == "__main__":
    main()
