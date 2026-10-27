"""Existing-grid execution uncertainty, selection and cache contracts."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import pc_xpass_high_pass as hp
import pc_xpass_uncertainty as uncertainty
import pc_xpass_versions as versions
import physical_pass_model as ppm
from scripts import generate_physical_xpass as generate
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph, make_label
from test_pc_xpass_acceleration import table as accel_table
from test_pc_xpass_distance import table as distance_table
from test_reachability import artifact as reachability_artifact


def options(**overrides):
    values = dict(min_speed=19., max_speed=25., speed_step=2., angle_step=3.,
                  radial_gridsize=3., top_n=None, top_pass_values=[1, 5], use_position_discount=False)
    return {**values, **overrides}


def graph():
    result = make_graph()
    result.x[:, 3] = torch.tensor([0., 50., 18.])
    return result


def test_default_scales_and_disabled_settings():
    args = generate.parse_args(["--pc-xpass"])
    assert not args.uncertainty and args.uncertainty_config is None
    assert not (uncertainty.KEYS & versions.settings(args).keys())
    args = generate.parse_args(["--pc-xpass", "--uncertainty", "true"])
    assert args.uncertainty_config == uncertainty.configuration(True)
    assert (args.speed_uncertainty, args.angle_uncertainty, args.elevation_uncertainty) == (2, 3, 6)
    assert generate.enabled_physical_xpass_metrics_from_args(args) == ["max_xpass"]
    assert args.top_n_values is None and args.top_n is None and not args.export_topmean
    args = generate.parse_args(["--pc-xpass", "--uncertainty", "true", "--top-pass", "5", "10", "--no-max",
                                "--speed-uncertainty", "0", "--angle-uncertainty", "2", "--elevation-uncertainty", "7"])
    assert generate.enabled_physical_xpass_metrics_from_args(args) == ["top_pass5_xpass", "top_pass10_xpass"]
    assert args.uncertainty_config == uncertainty.configuration(True, 0, 2, 7)


@pytest.mark.parametrize("flags", [
    ["--uncertainty", "true"], ["--uncertainty", "false"], ["--speed-uncertainty", "2"],
    ["--pc-xpass", "--uncertainty", "yes"],
    ["--pc-xpass", "--angle-uncertainty", "2"],
    ["--pc-xpass", "--uncertainty", "true", "--speed-uncertainty", "-1"],
    ["--pc-xpass", "--uncertainty", "true", "--elevation-uncertainty", "nan"],
    ["--pc-xpass", "--uncertainty", "true", "--angle-uncertainty", "inf"],
    ["--pc-xpass", "--uncertainty", "true", "--top-n", "5"],
    ["--pc-xpass", "--uncertainty", "true", "--top-n-values", "5", "10"],
    ["--pc-xpass", "--uncertainty", "true", "--no-max"],
])
def test_invalid_generation_cli(flags):
    with pytest.raises((SystemExit, ValueError)):
        generate.parse_args(flags)


@pytest.mark.parametrize("selector", ["max", "top-pass5"])
def test_location_flags_and_metric_selection(selector):
    args = location.parse_args(["--uncertainty", "true", "--xpass-version", selector])
    assert args.uncertainty_config == uncertainty.configuration(True)
    assert args.top_n is None and args.top_n_values is None
    assert args.top_pass_values == ([5] if selector == "top-pass5" else [])


def test_location_rejects_endpoint_metric():
    with pytest.raises(SystemExit):
        location.parse_args(["--uncertainty", "true", "--xpass-version", "top10"])


def test_elevation_examples_and_gaussian_weights():
    profiles, angles = uncertainty.neighbourhood([25]*4, [0, 20, 25, 30], [0], uncertainty.configuration(True))
    values = profiles.toarray()
    assert [np.flatnonzero(row).tolist() for row in values] == [[0], [1, 2], [1, 2, 3], [2, 3]]
    neighbour = math.exp(-.5*(5/6)**2)
    assert values[3, 3] == pytest.approx(1/(1+neighbour))
    assert values[3, 2] == pytest.approx(neighbour/(1+neighbour))
    np.testing.assert_allclose(values.sum(axis=1), 1.)
    np.testing.assert_array_equal(angles.toarray(), [[1.]])


def test_inclusive_rectangular_support_and_circular_angles():
    profile, angles = uncertainty.neighbourhood([21, 23, 23.01], [20, 26, 20],
        np.deg2rad([0, 3, 3.01, 357]), uncertainty.configuration(True))
    # Simultaneous one-scale changes in speed and elevation remain included.
    assert profile[0, 1] > 0 and profile[0, 2] == 0
    assert angles[0, 1] > 0 and angles[0, 2] == 0 and angles[0, 3] > 0
    assert angles[0, 1] == pytest.approx(angles[0, 3])


def test_zero_scales_and_isolated_profiles_are_identity():
    values = np.arange(6).reshape(2, 3)
    weights = uncertainty.neighbourhood([21, 22], [20, 25], np.deg2rad([0, 1, 359]),
                                         uncertainty.configuration(True, 0, 0, 0))
    np.testing.assert_array_equal(uncertainty.average(values, weights), values)


def test_failed_outcome_retains_weight_and_averaging_can_change_winner():
    weights = uncertainty.neighbourhood([1, 3, 10, 12], [0]*4, [0], uncertainty.configuration(True))
    score = np.array([[[1., .9]], [[np.nan, np.nan]], [[.2, .8]], [[.8, .3]]])
    outcomes = uncertainty.outcomes(score, score, np.ones_like(score), np.ones_like(score),
                                      np.ones((1, 2)), weights)
    assert outcomes["expected_xpass"][0, 0] == pytest.approx(1/(1+math.exp(-.5)))
    assert outcomes["endpoint"][:, 0].tolist() == [0, 0, 1, 0]
    assert outcomes["expected_xpass"][2, 0] == pytest.approx(.8)
    assert np.argmax(outcomes["expected_xpass"]) == 2
    # E[lane*control] is not generally E[lane]*E[control].
    lane = np.array([[[1]], [[0]]], dtype=float)
    control = 1-lane
    weights = uncertainty.neighbourhood([21, 23], [0, 0], [0], uncertainty.configuration(True))
    outcomes = uncertainty.outcomes(lane*control, lane*control, lane, control, np.ones((1, 1)), weights)
    assert outcomes["expected_xpass"].sum() == 0
    assert (outcomes["expected_lane_survival"] * outcomes["expected_control_prob"]).sum() > 0


@pytest.mark.parametrize("xt", [False, True])
@pytest.mark.parametrize("aerial", [False, True])
def test_integrated_scores_match_independent_discrete_expectation(xt, aerial):
    opts = options(top_xt=xt, high_pass_config=hp.configuration("on") if aerial else None)
    raw, details = {}, {}
    baseline = ppm.compute_graph_pc_xpass_metrics(graph(), diagnostic_details=raw, **opts)
    result = ppm.compute_graph_pc_xpass_metrics(graph(), diagnostic_details=details,
        uncertainty_config=uncertainty.configuration(True), **opts)
    for name in ("speeds", "elevations", "ball_times", "ball_heights"):
        np.testing.assert_array_equal(raw[name], details[name])
    score = raw["lane_survival"] * raw["receiver_control"]["home_2"]
    ranking = score * ppm._pc_xpass_xt_values(raw["target_x"], raw["target_y"]) if xt else score
    finite = np.isfinite(score) & np.isfinite(ranking)
    endpoint = np.argmax(np.where(finite, ranking, -np.inf), axis=-1)
    nominal = np.take_along_axis(score, endpoint[..., None], axis=-1)[..., 0]
    nominal_rank = np.take_along_axis(ranking, endpoint[..., None], axis=-1)[..., 0]
    nominal = np.where(finite.any(axis=-1), nominal, 0.)
    nominal_rank = np.where(finite.any(axis=-1), nominal_rank, 0.)
    expected, expected_rank = np.zeros_like(nominal), np.zeros_like(nominal)
    angles = np.arange(score.shape[1])*3.
    for p, (v, z) in enumerate(zip(raw["speeds"], raw["elevations"])):
        for a, angle in enumerate(angles):
            total = 0.
            for q, (v2, z2) in enumerate(zip(raw["speeds"], raw["elevations"])):
                for b, angle2 in enumerate(angles):
                    dv, dz, da = v2-v, z2-z, (angle2-angle+180)%360-180
                    if abs(dv) > 2+1e-10 or abs(dz) > 6+1e-10 or abs(da) > 3+1e-10:
                        continue
                    w = math.exp(-.5*((dv/2)**2+(dz/6)**2+(da/3)**2))
                    expected[p, a] += w*nominal[q, b]
                    expected_rank[p, a] += w*nominal_rank[q, b]
                    total += w
            expected[p, a] /= total
            expected_rank[p, a] /= total
    np.testing.assert_allclose(details["uncertainty_outcomes"]["home_2"]["expected_xpass"], expected, atol=1e-14)
    winner = np.unravel_index(np.argmax(expected_rank), expected_rank.shape)
    assert result["home_2__max_xpass"] == pytest.approx(np.clip(expected[winner], 1e-4, 1-1e-4))
    assert result["home_2__expected_xpass"] == pytest.approx(expected[winner])
    assert result["home_2__nominal_xpass"] == pytest.approx(nominal[winner])
    assert result["home_2__distance"] == endpoint[winner]*3
    # Stable ranking followed by one representative per speed/horizontal angle.
    representatives, seen = [], set()
    for flat in np.argsort(-expected_rank.ravel(), kind="stable"):
        p, a = np.unravel_index(flat, expected_rank.shape)
        key = (raw["speeds"][p], a)
        if key not in seen:
            representatives.append(flat)
            seen.add(key)
    assert result["home_2__top_pass5_xpass"] == pytest.approx(np.clip(expected.ravel()[representatives[:5]].mean(), 1e-4, 1-1e-4))
    assert result["home_2__top_pass1_xpass"] == result["home_2__max_xpass"]
    pd.testing.assert_series_equal(result, ppm.compute_graphs_pc_xpass_metrics([graph()],
        uncertainty_config=uncertainty.configuration(True), **opts)[0])
    assert baseline.index.difference(result.index).empty


def test_disabled_scores_metadata_and_fingerprints_equal():
    opts = options(high_pass_config=hp.configuration("on"))
    a = ppm.compute_graph_pc_xpass_metrics(graph(), **opts)
    b = ppm.compute_graph_pc_xpass_metrics(graph(), uncertainty_config=None, **opts)
    pd.testing.assert_series_equal(a, b)
    policy = ppm.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    a = ppm.pc_xpass_metadata(policy, **opts)
    b = ppm.pc_xpass_metadata(policy, uncertainty_config=None, **opts)
    assert a == b
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(a) == ppm.pc_xpass_lane_survival_metadata_fingerprint(b)
    c = ppm.pc_xpass_metadata(policy, uncertainty_config=uncertainty.configuration(True), **opts)
    assert c["top_pass_definition"] == uncertainty.TOP_PASS_DEFINITION
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(a) != ppm.pc_xpass_lane_survival_metadata_fingerprint(c)
    broken = copy.deepcopy(c)
    broken["uncertainty_config"]["speed_uncertainty"] = -1
    with pytest.raises(ValueError, match="uncertainty"):
        ppm.validate_pc_xpass_ball_motion_metadata(broken)
    broken = copy.deepcopy(c)
    broken["generation_settings"] = {"uncertainty": False}
    with pytest.raises(ValueError, match="uncertainty"):
        ppm.validate_pc_xpass_ball_motion_metadata(broken)


@pytest.mark.parametrize("location_mode", [False, True])
def test_version_resume_and_old_caches_default_off(tmp_path, monkeypatch, location_mode):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    parse = location.parse_args if location_mode else generate.parse_args
    prefix = ["--xpass-version", "top-pass5"] if location_mode else ["--pc-xpass", "--top-pass", "5"]
    args = parse([*prefix, "--uncertainty", "true", "--speed-uncertainty", "3"])
    args.pc_xpass_id = "uncertain"
    versions.start_generation(args, location=location_mode)
    resumed = parse([*([] if location_mode else ["--pc-xpass"]), "--pc-xpass-id", "uncertain"])
    assert resumed.uncertainty_config == args.uncertainty_config
    assert versions.settings(resumed) == versions.settings(args)
    opts = [*([] if location_mode else ["--pc-xpass"]), "--pc-xpass-id", "uncertain", "--speed-uncertainty", "2"]
    if location_mode:
        with pytest.warns(UserWarning, match="cached"):
            assert parse(opts).speed_uncertainty == 3
    else:
        with pytest.raises(ValueError, match="speed-uncertainty"):
            parse(opts)
    old = parse([] if location_mode else ["--pc-xpass"])
    old.pc_xpass_id = "old"
    versions.start_generation(old, location=location_mode)
    assert parse([*([] if location_mode else ["--pc-xpass"]), "--pc-xpass-id", "old"]).uncertainty_config is None


def test_runtime_cache_resume_repair_sidecars_and_selection(tmp_path, monkeypatch):
    items = [dict(match_id="example", graphs=[graph()], labels=torch.stack([make_label()]))]
    opts = dict(source=ppm.PC_XPASS_SOURCE, num_workers=1, export_lane_control=True,
                high_pass_config=hp.configuration("on"), uncertainty_config=uncertainty.configuration(True),
                **options(angle_step=30))
    assert ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)["cache_written"] == 1
    saved = pd.read_parquet(tmp_path / "matches/example.parquet")
    assert ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)["cache_hits"] == 1
    manifest = json.loads((tmp_path / "lane_control/coverage/example.json").read_text())
    assert manifest["schema_version"] == 3
    ppm.validate_x_pass_version_available(tmp_path, x_pass_version="top-pass5", metric="top_pass5_xpass")
    ppm.validate_pc_xpass_lane_survival_mode_cache_metadata(json.loads((tmp_path / "metadata.json").read_text()), "top_pass_5")
    with pytest.raises(ValueError):
        ppm.validate_x_pass_version_available(tmp_path, x_pass_version="top5", metric="top5_xpass")
    saved.drop(columns=["home_2__expected_xpass"]).to_parquet(tmp_path / "matches/example.parquet", index=True)
    assert ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)["cache_written"] == 1
    with pytest.raises(ValueError, match="uncertainty"):
        ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **{**opts, "uncertainty_config": None})
    monkeypatch.setattr(hp, "profile", lambda *args: pytest.fail("Cache-only read ran physics"))
    ppm.load_runtime_physical_xpass_visualization_component(tmp_path, "example", 7,
        metric="top_pass5_xpass", x_pass_version="top-pass5")


def test_windows_worker_results_equal_serial(tmp_path):
    items = [dict(match_id=f"example_{i}", graphs=[graph()], labels=torch.stack([make_label()])) for i in range(2)]
    opts = dict(source=ppm.PC_XPASS_SOURCE, physical_batch_size=1, export_lane_control=True,
                high_pass_config=hp.configuration("on"), uncertainty_config=uncertainty.configuration(True),
                **options(angle_step=30))
    ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path / "serial", num_workers=1, **opts)
    ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path / "parallel", num_workers=2, **opts)
    for item in items:
        for directory in ("matches", "lane_control"):
            name = f"{directory}/{item['match_id']}.parquet"
            pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "serial" / name),
                                          pd.read_parquet(tmp_path / "parallel" / name))


@pytest.mark.parametrize("margin", ["tta", "distance", "reachability"])
def test_zero_uncertainty_matches_all_movement_modes(accel_table, distance_table, reachability_artifact, margin):
    import pc_xpass_acceleration as acceleration
    import pc_xpass_distance as spatial
    import reachability
    movement = dict(margin=margin, reaction_time=0.)
    if margin == "tta":
        movement["acceleration_config"] = acceleration.configuration(7, "fixture")
    elif margin == "distance":
        movement["acceleration_config"] = spatial.configuration(7, "spatial", max_speeds=(5.,))
    else:
        movement["reachability_config"] = reachability.configuration("reachability", {"model_id": "test"})
    opts = options(**movement, angle_step=30, use_position_discount=True,
                   high_pass_config=hp.configuration("on", [20], high_pass_speed=[21]))
    raw = ppm.compute_graph_pc_xpass_metrics(graph(), **opts)
    averaged = ppm.compute_graph_pc_xpass_metrics(graph(),
        uncertainty_config=uncertainty.configuration(True, 0, 0, 0), **opts)
    pd.testing.assert_series_equal(raw, averaged.reindex(raw.index), atol=1e-14)


@pytest.mark.parametrize("dataset", ["sportec", "skillcorner"])
def test_transactional_match_cache(tmp_path, dataset):
    import pc_xpass_match_cache as cache
    from test_pc_xpass_match_cache import descriptor
    root = tmp_path / dataset
    cache.initialize(root)
    item = dict(match_id="one", graphs=[graph()], labels=torch.stack([make_label()]),
                frame_scope="frame_id", state_frame_id=[7])
    opts = dict(source=ppm.PC_XPASS_SOURCE, uncertainty_config=uncertainty.configuration(True),
                high_pass_config=hp.configuration("on"), export_lane_control=True, num_workers=1,
                **options(angle_step=30))
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.expect([item])
        assert ppm.prewarm_physical_xpass_runtime_cache([item], cache_dir=root, **opts)["cache_written"] == 1
        transaction.commit(export_lane=True)
    assert "home_2__expected_xpass" in pd.read_parquet(root / "matches/match.parquet")
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.expect([item])
        assert ppm.prewarm_physical_xpass_runtime_cache([item], cache_dir=root, **opts)["cache_hits"] == 1
        transaction.commit(export_lane=True)


def test_production_cli_generation_and_resume(tmp_path, monkeypatch):
    import sys
    inputs = tmp_path / "input/modification_1"
    inputs.mkdir(parents=True)
    state = pd.DataFrame([
        [1, 1, -30., 0., np.nan, 0., 0., 1, True],
        [1, 2, 20., 0., np.nan, 0., 0., 1, True],
        [2, 3, -12., 0., np.nan, 0., 0., 1, True],
        [0, 0, -30., 0., .13, 0., 0., 1, True],
    ], columns=["team", "player", "pos_x", "pos_y", "pos_z", "smooth_x_speed",
                "smooth_y_speed", "event_player", "playing_direction_event"])
    for number in (1, 2):
        state.to_csv(inputs / f"game_state_{number}.csv", index=False)
    pd.DataFrame({"higher_state_id": [1]}).to_csv(inputs / "modification.csv", index=False)
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "cache")
    common = ["generate_physical_xpass.py", "--pc-xpass", "--pc-xpass-id", "uncertainty_smoke",
        "--no-sportec", "--no-skillcorner", "--no-hawkeye", "--benchmark-input-dir", str(inputs.parent),
        "--benchmark-modification", "1", "--num-workers", "1", "--angle-step", "90",
        "--radial-gridsize", "10", "--top-pass", "5", "--export-lane-control"]
    monkeypatch.setattr(sys, "argv", common + ["--high-pass", "on", "--uncertainty", "true"])
    generate.main()
    root = tmp_path / "cache/uncertainty_smoke"
    metadata = json.loads((root / "metadata.json").read_text())
    dataset = json.loads((root / "benchmark/metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["uncertainty_config"] == dataset["uncertainty_config"] == uncertainty.configuration(True)
    assert dataset["top_pass_definition"] == metadata["top_pass_definition"] == uncertainty.TOP_PASS_DEFINITION
    assert dataset["available_metrics"] == ["max_xpass", "top_pass5_xpass"]
    assert dataset["lane_control_export"]["schema_version"] == 3
    saved = root / "benchmark/matches/modification_1_game_state_1.parquet"
    before = pd.read_parquet(saved)
    assert "home_2__expected_xpass" in before
    monkeypatch.setattr(sys, "argv", common)
    monkeypatch.setattr(hp, "profile", lambda *args: pytest.fail("Resume recalculated physics"))
    generate.main()
    pd.testing.assert_frame_equal(before, pd.read_parquet(saved))
