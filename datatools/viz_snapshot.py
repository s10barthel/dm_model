import os
import sys
from typing import Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes
from matplotlib.patches import Arc, Circle, Rectangle

import datatools.matplotsoccer as mps
from datatools import config

COMPONENT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "component_orange_red",
    plt.get_cmap("jet")(np.linspace(0.7, 1.0, 256)),
)
COMPONENT_ANNOT_TYPES = (
    "action_intent",
    "pass_success",
    "outcome_scoring",
    "outcome_conceding",
    "pass_score",
)

PITCHCONTROL_PLAYER_DOT_SIZE = 60.0
PITCHCONTROL_BALL_DOT_SIZE = PITCHCONTROL_PLAYER_DOT_SIZE / 2.0
PITCHCONTROL_PITCH_LINE_ALPHA = 0.35
PITCHCONTROL_PITCH_LINE_WIDTH = 1.5


def _column_suffix(column: str) -> str:
    return column.rsplit("_", 1)[-1]


def _column_entity(column: str) -> str:
    return column.rsplit("_", 1)[0]


class SnapshotVisualizer:
    def __init__(
        self,
        snapshot: pd.DataFrame,
        ball_xy: pd.DataFrame = None,
        player_sizes: pd.Series = None,
        player_colors: pd.Series = None,
        player_annots: pd.Series = None,
        player_marks: pd.DataFrame | Dict[str, List[str]] = None,
        edges: np.ndarray = None,
        arrows: List[Tuple[str, str]] = None,
        heatmap: np.ndarray = None,
        show_velocities=True,
        show_trajectories=False,
        highlight_players: Dict[str, str] = None,
        unnormalize=False,
        style: str = "default",
        attacking_team_prefix: str = None,
        show_player_numbers: bool | None = None,
    ):
        self.snapshot = snapshot.copy()
        self.sizes = player_sizes
        self.colors = player_colors
        self.annots = player_annots
        self.marks = player_marks
        self.ball_xy = ball_xy
        self.edges = edges
        self.arrows = arrows
        self.heatmap = heatmap
        self.show_velocities = show_velocities
        self.show_trajectories = show_trajectories
        self.highlight_players = highlight_players or {}
        self.style = style
        self.attacking_team_prefix = attacking_team_prefix if attacking_team_prefix in {"home", "away"} else None
        self.show_player_numbers = show_player_numbers

        if unnormalize:
            x_cols = [c for c in self.snapshot.columns if c.endswith("_x")]
            y_cols = [c for c in self.snapshot.columns if c.endswith("_y")]
            self.snapshot[x_cols] = (self.snapshot[x_cols] + 1) * config.FIELD_SIZE[0] / 2
            self.snapshot[y_cols] = (self.snapshot[y_cols] + 1) * config.FIELD_SIZE[1] / 2

    def _style_profile(self) -> dict:
        if self.style == "pitchcontrol":
            return {
                "pitch": "pitchcontrol",
                "default_player_size": PITCHCONTROL_PLAYER_DOT_SIZE,
                "player_size_range": (PITCHCONTROL_PLAYER_DOT_SIZE, PITCHCONTROL_PLAYER_DOT_SIZE),
                "ball_size": PITCHCONTROL_BALL_DOT_SIZE,
                "ball_facecolor": "black",
                "ball_edgecolor": "none",
                "ball_linewidth": 0.0,
                "annot_fontsize": 12,
                "annot_color": "black",
                "show_player_numbers": False if self.show_player_numbers is None else self.show_player_numbers,
                "velocity_scale": 1.0,
                "velocity_width": 0.0015,
                "velocity_headwidth": 3.0,
                "velocity_headlength": 3.0,
                "velocity_headaxislength": 3.0,
                "velocity_alpha": 0.75,
            }

        return {
            "pitch": "default",
            "default_player_size": 600,
            "player_size_range": (400, 2000),
            "ball_size": 200,
            "ball_facecolor": "white",
            "ball_edgecolor": "k",
            "ball_linewidth": 1.0,
            "annot_fontsize": 14,
            "annot_color": "k",
            "show_player_numbers": True if self.show_player_numbers is None else self.show_player_numbers,
            "velocity_scale": 1.5,
            "velocity_width": 0.003,
            "velocity_headwidth": 5,
            "velocity_headlength": 5,
            "velocity_headaxislength": 4.5,
            "velocity_alpha": 1.0,
        }

    def _draw_pitchcontrol_pitch(self, fig: plt.Figure, ax: axes.Axes) -> None:
        field_length, field_width = config.FIELD_SIZE
        x_min, y_min = 0.0, 0.0
        x_max, y_max = field_length, field_width
        center_x, center_y = field_length / 2.0, field_width / 2.0

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                field_length,
                field_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )
        ax.plot(
            [center_x, center_x],
            [y_min, y_max],
            color="black",
            lw=PITCHCONTROL_PITCH_LINE_WIDTH,
            alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
            zorder=0,
        )
        ax.add_patch(
            Circle(
                (center_x, center_y),
                9.15,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                (center_x, center_y),
                0.2,
                fill=True,
                edgecolor="none",
                facecolor="black",
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        penalty_area_depth = 16.5
        penalty_area_half_width = 40.32 / 2.0
        goal_area_depth = 5.5
        goal_area_half_width = 18.32 / 2.0

        ax.add_patch(
            Rectangle(
                (x_min, center_y - penalty_area_half_width),
                penalty_area_depth,
                2 * penalty_area_half_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (x_max - penalty_area_depth, center_y - penalty_area_half_width),
                penalty_area_depth,
                2 * penalty_area_half_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        ax.add_patch(
            Rectangle(
                (x_min, center_y - goal_area_half_width),
                goal_area_depth,
                2 * goal_area_half_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (x_max - goal_area_depth, center_y - goal_area_half_width),
                goal_area_depth,
                2 * goal_area_half_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        goal_width = 7.32
        goal_depth = 2.0
        ax.add_patch(
            Rectangle(
                (x_min - goal_depth, center_y - goal_width / 2.0),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                clip_on=False,
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (x_max, center_y - goal_width / 2.0),
                goal_depth,
                goal_width,
                fill=False,
                edgecolor="black",
                linewidth=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                clip_on=False,
                zorder=0,
            )
        )

        penalty_spot_distance = 11.0
        left_penalty_spot = (x_min + penalty_spot_distance, center_y)
        right_penalty_spot = (x_max - penalty_spot_distance, center_y)
        ax.add_patch(
            Circle(
                left_penalty_spot,
                0.2,
                fill=True,
                edgecolor="none",
                facecolor="black",
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )
        ax.add_patch(
            Circle(
                right_penalty_spot,
                0.2,
                fill=True,
                edgecolor="none",
                facecolor="black",
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        arc_radius = 9.15
        left_x_line = x_min + penalty_area_depth
        left_cos_theta = float(np.clip((left_x_line - left_penalty_spot[0]) / arc_radius, -1.0, 1.0))
        left_theta = float(np.degrees(np.arccos(left_cos_theta)))
        ax.add_patch(
            Arc(
                left_penalty_spot,
                2 * arc_radius,
                2 * arc_radius,
                angle=0,
                theta1=360 - left_theta,
                theta2=left_theta,
                color="black",
                lw=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        right_x_line = x_max - penalty_area_depth
        right_cos_theta = float(np.clip((right_x_line - right_penalty_spot[0]) / arc_radius, -1.0, 1.0))
        right_theta = float(np.degrees(np.arccos(right_cos_theta)))
        right_delta = 180.0 - right_theta
        ax.add_patch(
            Arc(
                right_penalty_spot,
                2 * arc_radius,
                2 * arc_radius,
                angle=0,
                theta1=180 - right_delta,
                theta2=180 + right_delta,
                color="black",
                lw=PITCHCONTROL_PITCH_LINE_WIDTH,
                alpha=PITCHCONTROL_PITCH_LINE_ALPHA,
                zorder=0,
            )
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_pitch(self, fig: plt.Figure, ax: axes.Axes, profile: dict) -> None:
        if profile["pitch"] == "pitchcontrol":
            self._draw_pitchcontrol_pitch(fig, ax)
            return
        mps.field("green", config.FIELD_SIZE[0], config.FIELD_SIZE[1], fig, ax, show=False)

    def _resolve_team_colors(self, switch_teams: bool) -> tuple[str, str]:
        if self.style == "pitchcontrol" and self.attacking_team_prefix in {"home", "away"}:
            if self.attacking_team_prefix == "home":
                return "blue", "red"
            return "red", "blue"

        home_color = "tab:blue" if switch_teams else "tab:red"
        away_color = "tab:red" if switch_teams else "tab:blue"
        return home_color, away_color

    def _get_player_sizes(self, players: List[str], profile: dict) -> np.ndarray | float:
        if self.sizes is not None and players and players[0] in self.sizes.index:
            smin, smax = profile["player_size_range"]
            return self.sizes[players].values * (smax - smin) + smin
        return profile["default_player_size"]

    def _resolve_velocity_colors(self, colors, profile: dict):
        alpha = profile["velocity_alpha"]
        if isinstance(colors, str):
            return mpl.colors.to_rgba(colors, alpha=alpha)

        color_array = np.array(colors, dtype=float, copy=True)
        if color_array.ndim == 1:
            if color_array.shape[0] == 3:
                color_array = np.append(color_array, alpha)
            else:
                color_array[3] = alpha
            return color_array

        if color_array.shape[1] == 3:
            alpha_column = np.full((color_array.shape[0], 1), alpha, dtype=float)
            return np.concatenate([color_array, alpha_column], axis=1)

        color_array[:, 3] = alpha
        return color_array

    def _get_highlight_styles(self, players: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        edgecolors = np.array(["none"] * len(players), dtype=object)
        linewidths = np.zeros(len(players), dtype=float)

        for idx, player in enumerate(players):
            if player in self.highlight_players:
                edgecolors[idx] = self.highlight_players[player]
                linewidths[idx] = 1.5

        return edgecolors, linewidths

    def plot_team_players(
        self,
        ax: axes.Axes,
        xy: pd.DataFrame,
        sizes: np.ndarray,
        colors: np.ndarray,
        profile: dict,
        anonymize=True,
        annotate=None,
    ):
        n_features = len(np.unique([c.split("_")[-1] for c in xy.columns]))  # ["x", "y", "vx", "vy"] for each player
        x = xy[xy.columns[0::n_features]].values[-1]
        y = xy[xy.columns[1::n_features]].values[-1]

        players = [c[:-2] for c in xy.columns[0::n_features]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))
        highlight_edgecolors, highlight_linewidths = self._get_highlight_styles(players)

        if self.marks is None:
            ax.scatter(x, y, s=sizes, c=colors, edgecolors=highlight_edgecolors, linewidths=highlight_linewidths, zorder=2)

        else:
            linewidth = 3
            player_zorder = 2

            if isinstance(self.marks, pd.DataFrame):
                self.marks = self.marks[self.marks["object_id"].str.startswith(players[0][0])]
                edgecolors = np.where(np.isin(players, self.marks["object_id"]), "gold", "none")

                for i in self.marks.index:
                    p = self.marks.at[i, "object_id"]
                    start_frame = self.marks.at[i, "start_frame_id"]
                    end_frame = self.marks.at[i, "end_frame_id"]
                    player_x = xy.loc[start_frame:end_frame, f"{p}_x"].values
                    player_y = xy.loc[start_frame:end_frame, f"{p}_y"].values
                    ax.plot(player_x, player_y, c="gold", linewidth=5, zorder=2)

                    if annotate is not None:
                        text = self.marks.at[i, annotate]

                        dx = player_x[-1] - player_x[-4]
                        dy = player_y[-1] - player_y[-4]
                        norm = np.sqrt(dx**2 + dy**2)
                        text_xy = [player_x[-1] + dx / norm * 3, player_y[-1] + dy / norm * 3]

                        assert isinstance(colors, str)
                        text_color = colors.replace("tab:", "dark")

                        ax.annotate(text, xy=text_xy, ha="center", va="center", color=text_color, fontsize=24, zorder=3)

            else:
                # highlights = [p for p in self.highlights if p[0] == players[0][0]]
                edgecolors = np.array(["none"] * len(players), dtype=object)

                for color, players_to_highlight in self.marks.items():
                    teammates_to_highlight = [p for p in players_to_highlight if p[:4] == players[0][:4]]
                    edgecolors[np.isin(players, teammates_to_highlight)] = color

                    for p in teammates_to_highlight:
                        # If any player is highlighted in the team, place all teammates above their opponents
                        player_zorder = 3
                        player_x = xy[f"{p}_x"].values
                        player_y = xy[f"{p}_y"].values
                        ax.plot(player_x, player_y, c=color, linewidth=5, zorder=player_zorder)

            scatter_edgecolors = np.array(edgecolors, dtype=object)
            scatter_linewidths = np.full(len(players), linewidth, dtype=float)
            highlight_mask = highlight_edgecolors != "none"
            scatter_edgecolors[highlight_mask] = highlight_edgecolors[highlight_mask]
            scatter_linewidths[highlight_mask] = np.maximum(
                scatter_linewidths[highlight_mask], highlight_linewidths[highlight_mask]
            )
            ax.scatter(
                x,
                y,
                s=sizes,
                c=colors,
                edgecolors=scatter_edgecolors,
                linewidths=scatter_linewidths,
                zorder=player_zorder,
            )

        if self.show_velocities:
            assert n_features >= 4
            vx = xy[xy.columns[2::n_features]].values[-1]
            vy = xy[xy.columns[3::n_features]].values[-1]
            ax.quiver(
                x,
                y,
                vx * profile["velocity_scale"],
                vy * profile["velocity_scale"],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=self._resolve_velocity_colors(colors, profile),
                width=profile["velocity_width"],
                headwidth=profile["velocity_headwidth"],
                headlength=profile["velocity_headlength"],
                headaxislength=profile["velocity_headaxislength"],
            )

        for i, p in enumerate(players):
            player_xy = xy[[f"{p}_x", f"{p}_y"]].values
            player_num = player_dict[p] if anonymize else int(p.split("_")[1])
            player_color = colors if isinstance(colors, str) else colors[i]

            if self.show_trajectories:
                ax.plot(player_xy[:, 0], player_xy[:, 1], c=player_color, ls="--", zorder=0)
            if profile["show_player_numbers"]:
                ax.annotate(
                    player_num,
                    xy=player_xy[-1],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=12,
                    fontweight="normal",
                    zorder=3,
                )

        return ax

    def plot(
        self,
        rotate_pitch=False,
        half=None,
        focus_xy: Tuple[float, float] = None,
        switch_teams=False,
        anonymize=False,
        annotate=None,
        smin=400,
        smax=2000,
        cmin=None,
        cmax=None,
        annot_type=None,
        hm_cmap="jet",
        show=True,
    ):
        profile = self._style_profile()
        if focus_xy is None:
            figsize = (13.5, 9.0) if half is None else (10.4, 14.4)  # (9, 6)
        else:
            figsize = (4, 5)

        if self.colors is not None:
            figsize = (figsize[0] * 1.2, figsize[1])

        fig, ax = plt.subplots(figsize=figsize)
        self._draw_pitch(fig, ax, profile)

        if half == "left":
            ax.set_xlim(-4, config.FIELD_SIZE[0] / 2)
        elif half == "right":
            ax.set_xlim(config.FIELD_SIZE[0] / 2, config.FIELD_SIZE[0] + 4)

        if focus_xy is not None:
            ax.set_xlim(focus_xy[0] - 20, focus_xy[0] + 20)
            ax.set_ylim(focus_xy[1] - 20, focus_xy[1] + 20)

        snapshot = self.snapshot.dropna(axis=1, how="all").copy()
        ball_xy = self.ball_xy.copy() if self.ball_xy is not None else None

        x_cols = [c for c in snapshot.columns if c.endswith("_x")]
        y_cols = [c for c in snapshot.columns if c.endswith("_y")]
        vx_cols = [c for c in snapshot.columns if c.endswith("_vx")]
        vy_cols = [c for c in snapshot.columns if c.endswith("_vy")]

        if rotate_pitch:
            snapshot[x_cols] = config.FIELD_SIZE[0] - snapshot[x_cols]
            snapshot[y_cols] = config.FIELD_SIZE[1] - snapshot[y_cols]
            snapshot[vx_cols] = -snapshot[vx_cols]
            snapshot[vy_cols] = -snapshot[vy_cols]
            if ball_xy is not None:
                ball_xy["ball_x"] = config.FIELD_SIZE[0] - ball_xy["ball_x"]
                ball_xy["ball_y"] = config.FIELD_SIZE[1] - ball_xy["ball_y"]

        player_cols = [
            c
            for c in snapshot.columns
            if c.startswith(("home_", "away_")) and _column_suffix(c) in ["x", "y", "vx", "vy"] and "_goal_" not in c
        ]
        home_xy = snapshot[[c for c in player_cols if c.startswith("home_")]]
        away_xy = snapshot[[c for c in player_cols if c.startswith("away_")]]

        home_players = [_column_entity(c) for c in home_xy.columns if c.endswith("_x")]
        away_players = [_column_entity(c) for c in away_xy.columns if c.endswith("_x")]

        home_sizes = self._get_player_sizes(home_players, profile)
        away_sizes = self._get_player_sizes(away_players, profile)

        home_colors = "tab:blue"
        away_colors = "tab:red"
        home_colors, away_colors = self._resolve_team_colors(switch_teams)
        annot_args = {
            "ha": "center",
            "va": "center",
            "color": profile["annot_color"],
            "fontsize": profile["annot_fontsize"],
            "fontweight": "normal",
            "zorder": 7,
        }

        if self.annots is not None:
            annot_type = annot_type or ""
            if any(component_type in annot_type for component_type in COMPONENT_ANNOT_TYPES):
                annots = self.annots.dropna()
            elif "epv" in annot_type:
                annots = self.annots[self.annots != 0.0]
            elif "select" in annot_type or "posterior" in annot_type:
                annots = self.annots[self.annots > 0.05]
            elif "credit" in annot_type:
                annots = self.annots[(self.annots > 0.001) | (self.annots < -0.001)]
            else:
                annots = self.annots[self.annots >= 0.005]

            for p, value in annots.items():
                if "credit" in annot_type and value > 0:
                    text = f"+{value:.3f}"
                elif "credit" in annot_type and value < 0:
                    text = f"-{-value:.3f}"
                else:
                    text = f"{value:.3f}"

                text_xy = snapshot[[f"{p}_x", f"{p}_y"]].values[-1]
                annot_kwargs = dict(annot_args)
                if not p.endswith("_goal"):
                    annot_kwargs.update({"xytext": (0, 8), "textcoords": "offset points"})
                elif p == "home_goal":
                    text_xy[0] += 3
                elif p == "away_goal":
                    text_xy[0] -= 3
                ax.annotate(text, xy=text_xy, **annot_kwargs)

        if self.colors is not None:
            colorbar_norm = None
            colorbar_cmap = None

            def resolve_colors(players: List[str], base_color: str):
                player_values = self.colors.reindex(players)
                valid_values = player_values.dropna()
                if valid_values.empty:
                    return base_color, None, None

                local_cmin = float(valid_values.min()) if cmin is None else float(cmin)
                local_cmax = float(valid_values.max()) if cmax is None else float(cmax)
                if local_cmax <= local_cmin:
                    eps = max(abs(local_cmin) * 0.01, 1e-6)
                    local_cmin -= eps
                    local_cmax += eps

                norm = mpl.colors.Normalize(vmin=local_cmin, vmax=local_cmax)
                cmap = COMPONENT_CMAP
                base_rgba = mpl.colors.to_rgba(base_color)
                mapped_colors = [
                    base_rgba if pd.isna(value) else cmap(norm(float(value)))
                    for value in player_values
                ]
                return np.array(mapped_colors), norm, cmap

            if any(player in self.colors.index for player in home_players):
                home_colors, colorbar_norm, colorbar_cmap = resolve_colors(home_players, home_colors)
            elif any(player in self.colors.index for player in away_players):
                away_colors, colorbar_norm, colorbar_cmap = resolve_colors(away_players, away_colors)

            if colorbar_norm is not None and colorbar_cmap is not None:
                sm = mpl.cm.ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap)
                sm.set_array([])

                cbar = fig.colorbar(sm, ax=ax)
                cbar.ax.tick_params(labelsize=20)

        if self.edges is not None:
            for src, dst in self.edges:
                edge_x = snapshot[[f"{src}_x", f"{dst}_x"]].values[-1]
                edge_y = snapshot[[f"{src}_y", f"{dst}_y"]].values[-1]
                if src[:4] == "home" and dst[:4] == "home":
                    edgecolor = home_colors if isinstance(home_colors, str) else "dimgray"
                elif src[:4] == "away" and dst[:4] == "away":
                    edgecolor = away_colors if isinstance(away_colors, str) else "dimgray"
                else:
                    edgecolor = "dimgray"
                ax.plot(edge_x, edge_y, color=edgecolor, linewidth=2, alpha=0.5, zorder=-100)

        if self.arrows is not None:
            for src, dst in self.arrows:
                if isinstance(src, str):
                    src_x = snapshot[f"{src}_x"].values[-1]
                    src_y = snapshot[f"{src}_y"].values[-1]
                else:
                    src_x, src_y = src

                if isinstance(dst, str):
                    dst_x = snapshot[f"{dst}_x"].values[-1]
                    dst_y = snapshot[f"{dst}_y"].values[-1]
                else:
                    dst_x, dst_y = dst

                ax.arrow(
                    src_x,
                    src_y,
                    dst_x - src_x,
                    dst_y - src_y,
                    color="k",
                    width=0.5,
                    length_includes_head=True,
                    zorder=4,
                )

        if self.heatmap is not None:
            hm_extent = (0, config.FIELD_SIZE[0], 0, config.FIELD_SIZE[1])
            ax.imshow(self.heatmap, cmap=hm_cmap, vmin=cmin, vmax=cmax, extent=hm_extent, alpha=0.7, zorder=0)

        if len(home_xy.columns) > 0:
            self.plot_team_players(ax, home_xy, home_sizes, home_colors, profile, anonymize, annotate)

        if len(away_xy.columns) > 0:
            self.plot_team_players(ax, away_xy, away_sizes, away_colors, profile, anonymize, annotate)

        if ball_xy is not None:
            ball_x = ball_xy["ball_x"].values
            ball_y = ball_xy["ball_y"].values
            ax.scatter(
                ball_x[-1],
                ball_y[-1],
                s=profile["ball_size"],
                c=profile["ball_facecolor"],
                edgecolors=profile["ball_edgecolor"],
                linewidths=profile["ball_linewidth"],
                marker="o",
                zorder=5,
            )

        elif "ball_x" in snapshot.columns:
            ball_x = snapshot["ball_x"].values
            ball_y = snapshot["ball_y"].values
            ax.scatter(
                ball_x[-1],
                ball_y[-1],
                s=profile["ball_size"],
                c=profile["ball_facecolor"],
                edgecolors=profile["ball_edgecolor"],
                linewidths=profile["ball_linewidth"],
                marker="o",
                zorder=5,
            )

        if show:
            plt.show()

        return fig, ax
