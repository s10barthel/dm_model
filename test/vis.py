from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

import datatools.matplotsoccer as mps

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one synchronized action snapshot from smoke-test exports.")
    parser.add_argument("--action-id", type=int, required=True, help="Synchronized SPADL action_id to visualize.")
    parser.add_argument("--events-csv", default="test/elastic_events.csv", help="Path to the smoke-test elastic events CSV.")
    parser.add_argument("--tracking-csv", default="test/kloppy_tracking.csv", help="Path to the smoke-test tracking CSV.")
    parser.add_argument("--output", help="Optional output PNG path. Defaults to test/snapshot_<action_id>.png.")
    return parser.parse_args()


def load_action_row(events_csv: Path, action_id: int) -> pd.Series:
    events = pd.read_csv(events_csv)
    event_ids = pd.to_numeric(events["action_id"], errors="coerce")
    matches = events.loc[event_ids == action_id]
    if matches.empty:
        raise ValueError(f"Action id {action_id} was not found in {events_csv}")
    return matches.iloc[0]


def load_tracking_snapshot(tracking_csv: Path, frame_id: int) -> pd.Series:
    tracking = pd.read_csv(tracking_csv)
    frame_ids = pd.to_numeric(tracking["frame_id"], errors="coerce")
    matches = tracking.loc[frame_ids == frame_id]
    if matches.empty:
        raise ValueError(f"Frame id {frame_id} was not found in {tracking_csv}")
    return matches.iloc[0]


def iter_team_positions(snapshot: pd.Series, prefix: str):
    x_cols = sorted(col for col in snapshot.index if col.startswith(prefix) and col.endswith("_x"))
    for x_col in x_cols:
        player_id = x_col[:-2]
        y_col = f"{player_id}_y"
        if y_col not in snapshot.index:
            continue

        x = snapshot[x_col]
        y = snapshot[y_col]
        if pd.isna(x) or pd.isna(y):
            continue

        yield player_id, float(x), float(y)


def format_player_label(player_id: str) -> str:
    return player_id.rsplit("_", 1)[-1]


def build_title(event_row: pd.Series) -> str:
    start_x = float(event_row["start_x"]) if pd.notna(event_row["start_x"]) else float("nan")
    start_y = float(event_row["start_y"]) if pd.notna(event_row["start_y"]) else float("nan")

    line_1 = (
        f"action_id={int(event_row['action_id'])} | original_event_id={event_row['original_event_id']} | "
        f"period={int(event_row['period_id'])} | seconds={float(event_row['seconds']):.3f}"
    )
    line_2 = (
        f"frame_id={int(event_row['frame_id'])} | type={event_row['spadl_type']} | success={event_row['success']} | "
        f"player_id={event_row['player_id']} | object_id={event_row['object_id']} | "
        f"start=({start_x:.2f}, {start_y:.2f})"
    )
    return f"{line_1}\n{line_2}"


def plot_team(ax, snapshot: pd.Series, prefix: str, color: str, acting_object_id: str | None) -> None:
    for player_id, x, y in iter_team_positions(snapshot, prefix):
        edge_color = "gold" if player_id == acting_object_id else "black"
        linewidth = 2.5 if player_id == acting_object_id else 1.0
        ax.scatter(x, y, s=260, c=color, edgecolors=edge_color, linewidths=linewidth, zorder=3)
        ax.annotate(
            format_player_label(player_id),
            xy=(x, y),
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )


def main() -> None:
    args = parse_args()
    events_csv = (ROOT / args.events_csv).resolve()
    tracking_csv = (ROOT / args.tracking_csv).resolve()
    output_path = Path(args.output).resolve() if args.output else (ROOT / "test" / f"snapshot_{args.action_id}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_row = load_action_row(events_csv, args.action_id)
    if pd.isna(event_row["frame_id"]):
        raise ValueError(f"Action id {args.action_id} does not have a synchronized frame_id.")
    snapshot = load_tracking_snapshot(tracking_csv, int(event_row["frame_id"]))

    fig, ax = plt.subplots(figsize=(14, 9))
    mps.field("green", PITCH_LENGTH, PITCH_WIDTH, fig=fig, ax=ax, show=False)

    acting_object_id = event_row["object_id"] if pd.notna(event_row["object_id"]) else None
    plot_team(ax, snapshot, "home_", "tab:red", acting_object_id)
    plot_team(ax, snapshot, "away_", "tab:blue", acting_object_id)

    if pd.notna(snapshot.get("ball_x")) and pd.notna(snapshot.get("ball_y")):
        ax.scatter(
            float(snapshot["ball_x"]),
            float(snapshot["ball_y"]),
            s=180,
            c="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    if pd.notna(event_row["start_x"]) and pd.notna(event_row["start_y"]):
        ax.scatter(
            float(event_row["start_x"]),
            float(event_row["start_y"]),
            s=120,
            c="black",
            marker="x",
            linewidths=2.0,
            zorder=6,
        )

    ax.set_title(build_title(event_row), fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved snapshot to: {output_path}")


if __name__ == "__main__":
    main()
