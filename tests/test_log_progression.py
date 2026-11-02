from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from datatools.xt import annotate_match_xt
from datatools.xt_targets import (LogProgressionSurface, configure_targets, target_settings,
    surface_from_metadata, transform_action_targets, validate_surface_returns, validate_training_targets)
from scripts import generate_xt
from xt_versions import required_files, resolve_xt_version


def surface_metadata(k=2):
    return dict(surface_type="log_progression", surface_definition_version=1,
                k=k, min_goal_distance=1.0)


def example_events():
    return pd.DataFrame(dict(game_id=["m"] * 4, action_id=[0, 1, 2, 3], original_event_id=[10, 11, 12, 13],
        period_id=[1] * 4, seconds=[0., 1., 2., 3.], team_id=["h", "h", "a", "h"],
        player_id=["p1", "p1", "p2", "p1"], object_id=["home_1", "home_1", "away_1", "home_1"],
        spadl_type=["pass", "shot", "pass", "pass"], success=[True, False, True, True],
        offside=[False] * 4, utc_timestamp=pd.date_range("2026-01-01", periods=4, freq="s"),
        start_x=[5., 85., 45., 75.], start_y=[34., 34., 15., 34.],
        end_x=[85., 105., 15., 90.], end_y=[34., 34., 20., 34.], expected_goal=[np.nan, .95, np.nan, np.nan]))


def test_surface_geometry_and_halving():
    model = LogProgressionSurface()
    assert model.predict_xy([0], [0])[0] == 0
    np.testing.assert_allclose(model.predict_xy([104, 105], [34, 34]), 1)
    distances = np.array([100., 50., 25., 12.5])
    values = model.predict_xy(105 - distances, np.full(4, 34.))
    np.testing.assert_allclose(np.diff(values), np.full(3, np.log(2) / np.log(model.max_distance)))
    np.testing.assert_allclose(model.predict_xy([20, 75], [10, 20]), model.predict_xy([20, 75], [58, 48]))
    x = np.linspace(0, 105, 200)
    assert (np.diff(model.predict_xy(x, np.full(200, 10))) >= 0).all()
    assert (model.predict_xy(x, np.full(200, 34)) >= model.predict_xy(x, np.zeros(200))).all()
    assert LogProgressionSurface(k=3).predict_xy([70], [0])[0] < LogProgressionSurface(k=1).predict_xy([70], [0])[0]
    assert LogProgressionSurface(min_goal_distance=5).predict_xy([101], [34])[0] == 1


@pytest.mark.parametrize("kwargs", [dict(k=0), dict(k=-1), dict(k=np.inf), dict(k=np.nan),
    dict(min_goal_distance=0), dict(min_goal_distance=200), dict(min_goal_distance=np.nan)])
def test_invalid_surface_parameters(kwargs):
    with pytest.raises(ValueError):
        LogProgressionSurface(**kwargs)


def test_shots_use_surface_and_away_annotations_rotate():
    events = example_events()
    model = LogProgressionSurface()
    annotated, _ = annotate_match_xt(events, model)
    assert annotated.loc[1, "xT"] == pytest.approx(model.predict_xy([85], [34])[0])
    assert annotated.loc[1, "xT"] < .95
    assert annotated.loc[2, "xT"] == pytest.approx(model.predict_xy([60], [53])[0])


def test_gain_transform_baselines_perspective_clipping():
    actions = pd.DataFrame(dict(object_id=["home_1", "away_1", "home_1"],
                                start_x=[75., 30., 75.], start_y=[10., 58., 10.]))
    pairs = pd.DataFrame(dict(scores_xt=[.7, .7, 0.], concedes_xt=[.1, .2, .3]))
    model = LogProgressionSurface()
    settings = dict(representation="gain", clip_negative_gains=True)
    transformed = transform_action_targets(actions, pairs, settings, model)
    assert transformed.scores_xt[0] == pytest.approx(transformed.scores_xt[1])
    assert transformed.scores_xt[2] == 0
    pd.testing.assert_series_equal(transformed.concedes_xt, pairs.concedes_xt)
    settings["clip_negative_gains"] = False
    assert transform_action_targets(actions, pairs, settings, model).scores_xt[2] < 0
    pd.testing.assert_frame_equal(transform_action_targets(actions, pairs), pairs)


def test_target_option_inheritance_and_validation():
    saved = dict(xt_targets=dict(representation="gain", clip_negative_gains=True))
    args = SimpleNamespace()
    assert configure_targets(args, saved) == target_settings(saved)
    with pytest.raises(ValueError, match="refresh-target-family"):
        configure_targets(SimpleNamespace(xt_target_representation="absolute"), saved)
    updated = configure_targets(SimpleNamespace(xt_target_representation="absolute"), saved, allow_change=True)
    assert updated["representation"] == "absolute" and not updated["clip_negative_gains"]
    with pytest.raises(ValueError, match="requires"):
        target_settings(dict(xt_targets=dict(clip_negative_gains=True)))
    with pytest.raises(ValueError, match="training requires"):
        validate_training_targets(dict(xt_targets=dict(representation="gain")))
    validate_training_targets(saved)
    validate_surface_returns(surface_metadata(), ["next_3", "next_3_skip1", "in_1"])
    with pytest.raises(ValueError, match="returns only"):
        validate_surface_returns(surface_metadata(), ["disc_max_0.9"])


def test_direct_xt_training_rejects_nonfinite_labels_without_metadata():
    from datatools import config
    from models.utils import get_outcome_targets

    labels = torch.zeros((2, len(config.LABEL_COLUMNS)), dtype=torch.float32)
    labels[0, config.LABEL_INDEX["scores_xt"]] = float("nan")
    args = SimpleNamespace(use_xt=True, use_xg=False, use_epv=False,
                           use_goal_distance=False, use_progression=False)
    with pytest.raises(ValueError, match="xT labels are missing or invalid"):
        get_outcome_targets(labels, args)


def test_learned_xt_gains_preserve_conceding_values():
    metadata = dict(coefficients=dict(intercept=-4., normalized_x=3., normalized_centrality=1.))
    surface = surface_from_metadata(metadata)
    actions = pd.DataFrame(dict(object_id=["home_1", "away_1"], start_x=[20., 85.], start_y=[34., 34.]))
    pairs = pd.DataFrame(dict(scores_xt=[.3, .3], concedes_xt=[.1, .2]))
    transformed = transform_action_targets(actions, pairs, dict(representation="gain", clip_negative_gains=True), surface)
    np.testing.assert_allclose(transformed.scores_xt, .3 - surface.predict_xy([20., 20.], [34., 34.]))
    pd.testing.assert_series_equal(transformed.concedes_xt, pairs.concedes_xt)
    # Learned xT continues to support discounted return definitions.
    validate_surface_returns(metadata, ["disc_0.9", "disc_max_0.9"])


def test_bundle_rejects_different_xt_target_definitions():
    from models.utils import validate_model_record_consistency
    from datatools.xt_targets import training_target_definition
    record = dict(feature_run_id="f", intended_receiver_mode="original", return_type="next_3",
                  target_family="xt", graph_schema=dict(node_in_dim=26, edge_in_dim=2),
                  xt_target_definition=training_target_definition(dict(xt_version="log_k2")))
    records = dict(outcome_scoring=record, outcome_conceding={**record})
    shared = validate_model_record_consistency(records)
    assert shared["xt_target_definition"] == record["xt_target_definition"]
    records["outcome_conceding"]["xt_target_definition"] = training_target_definition(dict(xt_version="log_k3"))
    with pytest.raises(ValueError, match="surface and target definition"):
        validate_model_record_consistency(records)


def test_log_version_generation_without_fitting(tmp_path, monkeypatch):
    events_dir, versions = tmp_path / "events", tmp_path / "versions"
    events_dir.mkdir()
    example_events().to_csv(events_dir / "m.csv", index=False)
    monkeypatch.setattr(generate_xt, "EVENT_SYNCED_DIR", events_dir)
    monkeypatch.setattr(generate_xt, "XT_DIR", versions)
    stage = versions / ".staging" / "log_test"
    stage.mkdir(parents=True)
    args = SimpleNamespace(log_progression=True, k=2., min_goal_distance=1., fit_only=False,
                           match_id=None, limit=None, train_split_percent=50, train_count=None)
    manifest = dict(manifest_id="test_split", metadata={"universe_count": 1}, train=["m"])
    destination = versions / "log_test"
    with patch.object(generate_xt, "fit_actions_for_train_split", side_effect=AssertionError("fitting prohibited")):
        generate_xt._build_version(args, manifest, "log_test", destination, stage)
    selected = resolve_xt_version("latest", xt_dir=versions, required_match_ids=["m"], split_manifest_id="test_split")
    assert selected.version == "log_test"
    assert not (destination / "xT_glm_fit_sample.csv").exists()
    assert not (destination / "xT_source_grid.csv").exists()
    assert all((destination / name).is_file() for name in required_files(selected.metadata))
    np.testing.assert_allclose(surface_from_metadata(selected.metadata).predict_xy([75], [20]),
                               LogProgressionSurface().predict_xy([75], [20]))


@pytest.mark.parametrize("arguments", [["--k", "2"], ["--log-progression", "--use-interaction"],
    ["--log-progression", "--k", "0"], ["--log-progression", "--min-goal-distance", "200"]])
def test_generation_rejects_invalid_options(arguments, monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_xt.py", *arguments])
    with pytest.raises(SystemExit):
        generate_xt.parse_args()
