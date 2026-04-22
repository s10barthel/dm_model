from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DM_MODEL_ROOT = Path(__file__).resolve().parents[2]
DATATOOLS_ROOT = DM_MODEL_ROOT / "datatools"
if str(DATATOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATATOOLS_ROOT))
if str(DM_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(DM_MODEL_ROOT))

from hawkeye import (
    build_hawkeye_component_tables,
    build_hawkeye_situation,
    build_hawkeye_visualization_probs,
    clean_hawkeye_ball,
    clean_hawkeye_tracking,
    load_hawkeye_ball,
    load_hawkeye_component_run,
    load_hawkeye_tracking,
)
from viz_helpers import compute_pass_score, figure_to_rgb_image, save_animation
from viz_snapshot import SnapshotVisualizer


COACH_RATINGS_PATH = DM_MODEL_ROOT / "validation" / "coach_ratings" / "coach_ratings.csv"
DEFAULT_TRACKING_CSV = DM_MODEL_ROOT / "hawkeye_data" / "centroid_data_team.csv"
DEFAULT_BALL_CSV = DM_MODEL_ROOT / "hawkeye_data" / "ball_data_selected.csv"
DEFAULT_COMPONENT_DIR = DM_MODEL_ROOT / "data" / "component_runs" / "hawkeye"
DEFAULT_OUTPUT_DIR = DM_MODEL_ROOT / "validation" / "coach_ratings" / "visualizations"
BALLRECEIPT_ATOL = 1e-9
ANIMATION_FPS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--situation-id",
        action="append",
        help="Restrict visualization to one or more coach-rated Hawkeye situation ids.",
    )
    parser.add_argument(
        "--tracking-csv",
        default=str(DEFAULT_TRACKING_CSV),
        help="Hawkeye player-tracking CSV.",
    )
    parser.add_argument(
        "--ball-csv",
        default=str(DEFAULT_BALL_CSV),
        help="Hawkeye ball-tracking CSV.",
    )
    parser.add_argument(
        "--component-dir",
        default=str(DEFAULT_COMPONENT_DIR),
        help="Hawkeye component-run directory containing hawkeye_data.parquet and metadata.json.",
    )
    parser.add_argument(
        "--show-trajectories",
        action="store_true",
        help="Draw dashed recent player trajectories.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Save GIFs instead of the default MP4 animations.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Visualization output directory.",
    )
    return parser.parse_args()


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{label} is missing required columns: {', '.join(missing_columns)}")


def normalize_text_id(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned == "")


def normalize_player_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def format_coach_score(value: float) -> str:
    text = f"{float(value):.2f}"
    return text.rstrip("0").rstrip(".")


def load_scored_coach_ratings(path: Path) -> pd.DataFrame:
    coach_ratings = pd.read_csv(path, low_memory=False)
    validate_required_columns(
        coach_ratings,
        ["id", "uefa_player_id", "Scores", "team"],
        "Coach ratings",
    )
    normalized = coach_ratings.copy()
    normalized["id"] = normalize_text_id(normalized["id"])
    normalized["uefa_player_id"] = normalize_player_id(normalized["uefa_player_id"])
    normalized["team"] = normalized["team"].astype("string").str.strip()
    normalized["Scores"] = pd.to_numeric(normalized["Scores"], errors="coerce")
    normalized = normalized.dropna(subset=["id", "uefa_player_id", "Scores", "team"]).copy()
    normalized["uefa_player_id"] = normalized["uefa_player_id"].astype(int)
    normalized["team"] = normalized["team"].astype(str)
    return normalized


def resolve_selected_situation_ids(
    component_export: pd.DataFrame,
    scored_coach_ratings: pd.DataFrame,
    requested_ids: list[str] | None,
) -> list[str]:
    available_component_ids = {
        str(situation_id) for situation_id in component_export["id"].dropna().astype(str).tolist()
    }
    scored_ids = sorted({str(situation_id) for situation_id in scored_coach_ratings["id"].tolist()})
    scored_id_set = set(scored_ids)

    if requested_ids:
        requested = list(dict.fromkeys(str(situation_id).strip() for situation_id in requested_ids))
        missing_in_scores = [situation_id for situation_id in requested if situation_id not in scored_id_set]
        if missing_in_scores:
            raise KeyError(
                "Requested situation ids do not have any non-empty coach Scores: "
                + ", ".join(missing_in_scores)
            )
        missing_in_components = [
            situation_id for situation_id in requested if situation_id not in available_component_ids
        ]
        if missing_in_components:
            raise KeyError(
                "Requested situation ids are not present in the selected Hawkeye component data: "
                + ", ".join(missing_in_components)
            )
        return requested

    selected_ids = [situation_id for situation_id in scored_ids if situation_id in available_component_ids]
    if not selected_ids:
        raise ValueError("No coach-rated Hawkeye situations were found in the selected component data.")
    return selected_ids


def load_component_export(component_dir: Path) -> tuple[pd.DataFrame, dict]:
    if not component_dir.is_dir():
        raise NotADirectoryError(f"--component-dir must point to a directory, got: {component_dir}")
    return load_hawkeye_component_run(component_dir)


def get_ballreceipt_value(situation_tracking: pd.DataFrame) -> float:
    if "BallReceipt" not in situation_tracking.columns:
        raise KeyError(
            f"Hawkeye situation {situation_tracking['id'].iloc[0]} is missing the BallReceipt column."
        )

    ballreceipt_values = pd.to_numeric(situation_tracking["BallReceipt"], errors="coerce").dropna().unique()
    if len(ballreceipt_values) != 1:
        raise ValueError(
            f"Hawkeye situation {situation_tracking['id'].iloc[0]} must contain exactly one BallReceipt value, "
            f"found {ballreceipt_values.tolist()}."
        )
    return float(ballreceipt_values[0])


def get_frame_ids_for_window(situation, ballreceipt: float) -> list[int]:
    frame_meta = situation.frame_meta.reset_index()
    frame_mask = frame_meta["abs_time"].between(
        ballreceipt - BALLRECEIPT_ATOL,
        ballreceipt + 1.0 + BALLRECEIPT_ATOL,
        inclusive="both",
    )
    frame_ids = frame_meta.loc[frame_mask, "frame_id"].astype(int).tolist()
    if not frame_ids:
        raise ValueError(
            f"No frames fall inside the BallReceipt window for situation {situation.situation_id}."
        )
    return frame_ids


def get_ballreceipt_frame_id(situation, ballreceipt: float) -> int:
    frame_meta = situation.frame_meta.reset_index()
    exact_matches = frame_meta.loc[
        np.isclose(frame_meta["abs_time"], ballreceipt, atol=BALLRECEIPT_ATOL),
        "frame_id",
    ].astype(int)
    if exact_matches.empty:
        raise ValueError(
            f"Could not find the BallReceipt frame for situation {situation.situation_id} at {ballreceipt}."
        )
    return int(exact_matches.iloc[0])


def _row_for_frame(component_table: pd.DataFrame, frame_id: int) -> pd.Series | None:
    if component_table.empty or frame_id not in component_table.index:
        return None

    row = component_table.loc[frame_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def probs_for_pass_score_frame(component_tables: dict[str, pd.DataFrame], frame_id: int) -> pd.Series:
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


def build_coach_score_series(
    scored_coach_ratings: pd.DataFrame,
    situation,
    situation_id: str,
) -> pd.Series:
    coach_rows = scored_coach_ratings.loc[
        scored_coach_ratings["id"] == str(situation_id),
        ["team", "uefa_player_id", "Scores"],
    ].copy()
    if coach_rows.empty:
        raise ValueError(f"No scored coach-rating rows were found for situation {situation_id}.")

    coach_rows = coach_rows.loc[coach_rows["team"].isin(situation.team_map)].copy()
    if coach_rows.empty:
        raise ValueError(
            f"Coach-rating rows for situation {situation_id} could not be mapped to Hawkeye team prefixes."
        )

    coach_rows["object_id"] = coach_rows.apply(
        lambda row: f"{situation.team_map[str(row['team'])]}_{int(row['uefa_player_id'])}",
        axis=1,
    )
    coach_rows = coach_rows.drop_duplicates(subset=["object_id"], keep="first")
    return coach_rows.set_index("object_id")["Scores"].astype(float).sort_index()


def add_coach_score_annotations(
    ax,
    snapshot: pd.DataFrame,
    coach_scores: pd.Series,
    attacking_prefix: str,
) -> None:
    if coach_scores.empty:
        return

    for object_id, value in coach_scores.dropna().items():
        if not str(object_id).startswith(attacking_prefix):
            continue
        x_column = f"{object_id}_x"
        y_column = f"{object_id}_y"
        if x_column not in snapshot.columns or y_column not in snapshot.columns:
            continue

        x_value = snapshot[x_column].iloc[-1]
        y_value = snapshot[y_column].iloc[-1]
        ax.annotate(
            format_coach_score(float(value)),
            xy=(x_value, y_value),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            color="#1f4e79",
            fontsize=11,
            fontweight="normal",
            zorder=7,
        )


def render_frame_image(
    situation,
    frame_id: int,
    coach_scores: pd.Series,
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

    fig, ax = visualizer.plot(rotate_pitch=False, anonymize=True, annot_type="pass_score", show=False)
    fig.subplots_adjust(top=0.92, left=0.02, right=0.98, bottom=0.02)
    ax.text(
        0.5,
        1.01,
        f"{situation.situation_id} | {frame_info['abs_time']:.3f} | Pass Score",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="black",
    )
    add_coach_score_annotations(ax, snapshot, coach_scores, attacking_prefix)

    image = figure_to_rgb_image(fig, dpi=150)
    plt.close(fig)
    return image


def save_ballreceipt_snapshot(
    output_root: Path,
    situation_id: str,
    situation,
    ballreceipt_frame_id: int,
    coach_scores: pd.Series,
    component_tables: dict[str, pd.DataFrame],
    show_trajectories: bool,
) -> None:
    probs = probs_for_pass_score_frame(component_tables, ballreceipt_frame_id)
    image = render_frame_image(
        situation,
        ballreceipt_frame_id,
        coach_scores,
        probs,
        show_trajectories=show_trajectories,
    )
    image.save(output_root / f"{situation_id}.png")


def main() -> None:
    args = parse_args()
    component_dir = Path(args.component_dir).expanduser()
    component_export, component_metadata = load_component_export(component_dir)
    freeze_ballreceipt = bool(component_metadata.get("freeze_ballreceipt", True))

    scored_coach_ratings = load_scored_coach_ratings(COACH_RATINGS_PATH)
    situation_ids = resolve_selected_situation_ids(
        component_export,
        scored_coach_ratings,
        args.situation_id,
    )

    tracking = clean_hawkeye_tracking(load_hawkeye_tracking(args.tracking_csv))
    ball = clean_hawkeye_ball(load_hawkeye_ball(args.ball_csv))
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    suffix = "gif" if args.gif else "mp4"
    for situation_id in situation_ids:
        situation_tracking = tracking.loc[tracking["id"] == str(situation_id)].copy()
        if situation_tracking.empty:
            raise KeyError(f"Hawkeye situation id {situation_id} was not found in {args.tracking_csv}.")

        ballreceipt = get_ballreceipt_value(situation_tracking)
        situation, _, _ = build_hawkeye_situation(
            situation_tracking,
            ball,
            freeze_ballreceipt=freeze_ballreceipt,
            build_graphs=False,
        )
        component_tables = build_hawkeye_component_tables(component_export, situation)
        coach_scores = build_coach_score_series(scored_coach_ratings, situation, situation_id)
        frame_ids = get_frame_ids_for_window(situation, ballreceipt)
        ballreceipt_frame_id = get_ballreceipt_frame_id(situation, ballreceipt)

        def iter_images():
            for frame_id in frame_ids:
                probs = probs_for_pass_score_frame(component_tables, frame_id)
                yield render_frame_image(
                    situation,
                    frame_id,
                    coach_scores,
                    probs,
                    show_trajectories=args.show_trajectories,
                )

        animation_path = output_root / f"{situation_id}.{suffix}"
        save_animation(iter_images(), animation_path, fps=ANIMATION_FPS, gif=args.gif)
        save_ballreceipt_snapshot(
            output_root,
            str(situation_id),
            situation,
            ballreceipt_frame_id,
            coach_scores,
            component_tables,
            show_trajectories=args.show_trajectories,
        )
        print(f"Saved coach-rating visualizations to {animation_path} and {output_root / f'{situation_id}.png'}")


if __name__ == "__main__":
    main()
