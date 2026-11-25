from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from datatools.config import FIELD_SIZE
from datatools.field_of_view import CoveringSector, covering_sector
from datatools.hawkeye import build_hawkeye_situation, clean_hawkeye_tracking
from datatools.viz_snapshot import SnapshotVisualizer
from scripts import visualize_hawkeye as viz
from scripts.visualization_field_of_view import (
    hawkeye_field_of_view, prepare_hawkeye_visualization_tracking,
    report_unsupported_field_of_view, skillcorner_field_of_view, sportec_field_of_view,
)


def unit_vectors(bearings):
    radians = np.radians(bearings)
    return np.column_stack((np.cos(radians), np.sin(radians)))


@pytest.mark.parametrize(
    "bearings,start,width",
    [([350, 15, 30], 350, 40), ([0, 90, 180, 270], 90, 270), ([30, 140, 240], 30, 210),
     ([20, 20], 20, 0)],
)
def test_minimum_covering_sector(bearings, start, width):
    sector = covering_sector([0, 0], unit_vectors(bearings))
    assert sector.start_deg == pytest.approx(start)
    assert sector.width_deg == pytest.approx(width)


def test_undefined_bearings():
    assert covering_sector([0, 0], [[0, 0], [1, 1]]) is None
    assert covering_sector([0, 0], [[1, 1], [np.nan, 1]]) is None


@pytest.mark.parametrize("heatmap", [False, True])
def test_wide_sector_is_rotated_clipped_and_behind_foreground(heatmap):
    length, width = FIELD_SIZE
    # Rotating this wedge puts its apex at (25, 20) and opens CCW from the east.
    section = CoveringSector((length - 25, width - 20), 180, 270)
    snapshot = pd.DataFrame({"home_1_x": [length - 25], "home_1_y": [width - 20]})
    ball = pd.DataFrame({"ball_x": [length - 50], "ball_y": [width - 30]})
    kwargs = dict(snapshot=snapshot, ball_xy=ball, style="pitchcontrol", show_velocities=False)
    if heatmap:
        kwargs["heatmap"] = np.full((6, 10), 0.5)
    fig, ax = SnapshotVisualizer(**kwargs, field_of_view=section).plot(rotate_pitch=True, show=False)
    baseline, baseline_ax = SnapshotVisualizer(**kwargs).plot(rotate_pitch=True, show=False)
    try:
        cones = [patch for patch in ax.patches if isinstance(patch, Wedge)]
        assert len(cones) == 1
        cone = cones[0]
        assert cone.center == (25, 20)
        assert cone.theta2 - cone.theta1 == 270
        assert cone.get_facecolor() == ((1, 1, 1, 0.75) if heatmap else (0.5, 0.5, 0.5, 0.25))
        assert cone.get_clip_path() is not None
        assert all(cone.zorder < artist.zorder for artist in [*ax.lines, *ax.collections])
        if heatmap:
            assert ax.images[0].zorder < cone.zorder
        assert not any(isinstance(patch, Wedge) for patch in baseline_ax.patches)

        def pixel(figure, axes, xy):
            x, y = axes.transData.transform(xy).round().astype(int)
            pixels = np.asarray(figure.canvas.buffer_rgba())
            return tuple(pixels[-y - 1, x, :3])

        # Give the pitch visible padding: the axes clip alone would not keep it clear.
        ax.set_ylim(-2, width)
        baseline_ax.set_ylim(-2, width)
        fig.canvas.draw()
        baseline.canvas.draw()
        for xy in [(50, 30), (25, 20), (50, 10), (10, -0.8)]:
            # Ball, player, excluded quadrant, and padding all stay unchanged.
            assert pixel(fig, ax, xy) == pixel(baseline, baseline_ax, xy)
        assert pixel(fig, ax, (10, 10)) != pixel(baseline, baseline_ax, (10, 10))
    finally:
        plt.close(fig)
        plt.close(baseline)


def hawkeye_inputs():
    players = [(i + 1, "A" if i < 10 else "B", 1, x, y)
               for i, (x, y) in enumerate(unit_vectors(np.linspace(5, 130, 20)) * 10)]
    players += [(21, "A", 2, 0, 0), (22, "B", 2, -10, -10), (23, "B", 3, 0, -10)]
    # The goalkeeper is the clip recipient; the field-of-view code must ignore the
    # other keeper and the official, even though neither is inside the sector.
    rows, balls = [], []
    for time in [100.0, 100.04, 100.08]:
        balls.append(dict(game_id=1, half=1, abs_time=time, ball_x=5, ball_y=0, ball_z=0))
        for pid, team, role, x, y in players:
            if pid == 23 and time == 100.04:
                role = 9  # This raw row is discarded by the modeling loader.
            rows.append(dict(game_id=1, half=1, abs_time=time, id="clip", BallReceipt=100.0,
                             uefa_player_id=pid, role=role, team=team, centroid_x=x, centroid_y=y,
                             PlayerID=21, possession_team="A"))
    return pd.DataFrame(rows), pd.DataFrame(balls)


def test_raw_qc_and_displayed_coordinates():
    raw, ball = hawkeye_inputs()
    raw = pd.concat([raw, raw.iloc[[0]]])  # Identical roster rows are allowed.
    clean, source = prepare_hawkeye_visualization_tracking(
        raw, clean_hawkeye_tracking, SimpleNamespace(field_of_view=True),
    )
    situation, _, _ = build_hawkeye_situation(clean, ball, freeze_ballreceipt=False, build_graphs=False)
    # Do not use per-frame possessor metadata, and follow displayed pose adjustments.
    situation.frame_meta.loc[:, "possessor_object_id"] = "home_1"
    situation.tracking.loc[2, "home_21_y"] += 3
    with pytest.warns(UserWarning, match=r"1/3 frame\(s\).*unknown player role") as warnings:
        sectors = hawkeye_field_of_view(source, situation, [0, 1, 2])
    assert len(warnings) == 1
    assert set(sectors) == {0, 2}
    assert sectors[0].center == (FIELD_SIZE[0] / 2, FIELD_SIZE[1] / 2)
    assert sectors[0].width_deg == pytest.approx(125)
    assert sectors[2].center[1] == sectors[0].center[1] + 3
    assert sectors[2].width_deg != sectors[0].width_deg


@pytest.mark.parametrize("invalid,warning", [
    ("missing_outfielder", "incomplete outfield roster"),
    ("missing_coordinates", "missing player coordinates"),
    ("coincident_with_recipient", "undefined bearing"),
])
def test_no_cone_for_invalid_frame(invalid, warning):
    raw, ball = hawkeye_inputs()
    target = raw.abs_time.eq(100.0) & raw.uefa_player_id.eq(1)
    if invalid == "missing_outfielder":
        raw = raw.loc[~target]
    elif invalid == "missing_coordinates":
        raw.loc[target, "centroid_x"] = np.nan
    else:
        raw.loc[target, ["centroid_x", "centroid_y"]] = 0
    clean, source = prepare_hawkeye_visualization_tracking(
        raw, clean_hawkeye_tracking, SimpleNamespace(field_of_view=True),
    )
    situation, _, _ = build_hawkeye_situation(clean, ball, freeze_ballreceipt=False, build_graphs=False)
    with pytest.warns(UserWarning, match=warning):
        assert hawkeye_field_of_view(source, situation, [0]) == {}


def test_hawkeye_png_and_animation_share_the_overlay(monkeypatch, tmp_path):
    raw, ball = hawkeye_inputs()
    monkeypatch.setattr(viz, "load_hawkeye_tracking", lambda _: raw)
    monkeypatch.setattr(viz, "load_hawkeye_ball", lambda _: ball)
    video_frames = []
    monkeypatch.setattr(viz, "save_animation", lambda images, *_args, **_kw: video_frames.extend(images))
    base_args = ["--only-tracking", "--situation-id", "clip"]
    for name, output, flags in [("plain", "png", []), ("still", "png", ["--field-of-view"]),
                               ("movie", "mp4", ["--field-of-view"])]:
        root = tmp_path / name
        root.mkdir()
        args = viz.parse_args([*base_args, "--output", output, *flags])
        assert args.field_of_view is bool(flags)
        if name == "movie":
            with pytest.warns(UserWarning, match=r"1/3 frame\(s\)"):
                viz.run_tracking_only_visualization(args, tmp_path, root, name)
        else:
            viz.run_tracking_only_visualization(args, tmp_path, root, name)

    with (
        Image.open(next((tmp_path / "still").rglob("tracking*.png"))) as still,
        Image.open(next((tmp_path / "plain").rglob("tracking*.png"))) as plain,
    ):
        assert np.array_equal(np.asarray(still), np.asarray(video_frames[0]))
        assert not np.array_equal(np.asarray(still), np.asarray(plain))
    # Re-render the invalid frame without a cone: it must not reuse the first frame's cone.
    clean = clean_hawkeye_tracking(raw)
    situation, _, _ = build_hawkeye_situation(clean, ball, build_graphs=False)
    assert np.array_equal(np.asarray(video_frames[1]), np.asarray(viz.render_frame_image(situation, 1, "tracking", None)))


def test_sportec_uses_confirmed_receiver_and_never_the_sender():
    frame = {}
    lineup = []
    for team, offset in [("home", 0), ("away", 20)]:
        for number in range(11):
            name = f"{team}_{number}"
            frame[f"{name}_x"] = number + offset
            frame[f"{name}_y"] = number * 2
            lineup.append((name, "goal_keeper" if number == 0 else "center_back"))
    action = dict(frame_id=0, spadl_type="pass", object_id="home_1", receiver_id="home_0", success=True)
    match = SimpleNamespace(actions=pd.DataFrame([action]), tracking=pd.DataFrame([frame]),
                            lineup=pd.DataFrame(lineup, columns=["object_id", "advanced_position"]))
    assert sportec_field_of_view(match, 0, True).center == (0, 0)
    match.actions["source_has_receiver_identifier"] = False
    with pytest.warns(UserWarning, match="no source recipient identifier"):
        assert sportec_field_of_view(match, 0, True) is None
    match.actions.loc[0, "receiver_id"] = "away_5"
    with pytest.warns(UserWarning, match="no confirmed teammate recipient"):
        assert sportec_field_of_view(match, 0, True) is None
    with pytest.warns(UserWarning, match="does not provide a designated recipient"):
        report_unsupported_field_of_view(SimpleNamespace(field_of_view=True), "test source")


def test_skillcorner_uses_matched_outgoing_recipient_and_not_possessor():
    frame, metadata, raw_players = {}, [], []
    for team_id, prefix in [(0, "home"), (1, "away")]:
        for number in range(11):
            player_id = team_id * 11 + number
            frame[f"{prefix}_{player_id}_x"] = number + team_id * 20
            frame[f"{prefix}_{player_id}_y"] = number * 2
            metadata.append(dict(player_id=player_id, team_id=team_id, is_goalkeeper=number == 0))
            raw_players.append(dict(id=player_id, player_role={"acronym": "GK" if number == 0 else "CB"}))
    event = dict(index=1, player_id=2, player_targeted_id=5, end_type="pass", pass_outcome="successful",
                 is_pass_reception_matched=True)
    context = dict(events=pd.DataFrame([event]), player_meta=pd.DataFrame(metadata),
                   match_data={"players": raw_players})
    possession = SimpleNamespace(match_id="test", event_index=1, tracking=pd.DataFrame([frame]))
    assert skillcorner_field_of_view(context, possession, [0], True)[0].center == (5, 10)
    context["events"].loc[0, "is_pass_reception_matched"] = False
    with pytest.warns(UserWarning, match="no designated recipient"):
        assert skillcorner_field_of_view(context, possession, [0], True) == {}
