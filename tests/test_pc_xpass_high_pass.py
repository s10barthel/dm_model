"""Aerial trajectories, controls, selection and cache lifecycle regressions."""
from pathlib import Path
import copy
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import pc_xpass_high_pass as hp
import pc_xpass_versions as versions
import physical_pass_model as ppm
from scripts import generate_physical_xpass as generate
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph, make_label
from test_pc_xpass_acceleration import table as accel_table
from test_pc_xpass_distance import table as distance_table
from test_reachability import artifact as reachability_artifact


def settings(**overrides):
    return dict(min_speed=20, max_speed=20, speed_step=2, angle_step=30,
                radial_gridsize=2, top_n_values=[3, 10], top_n=None,
                top_pass_values=[1, 5], use_position_discount=False, **overrides)


def obstructed_graph():
    g = make_graph()
    g.x[:, 3] = torch.tensor([0., 50., 18.])
    return g


def test_cli_defaults_and_inheritance():
    off = generate.parse_args(["--pc-xpass"])
    assert off.high_pass_config is None
    assert not (hp.KEYS & versions.settings(off).keys())
    for parse in (generate.parse_args, location.parse_args):
        prefix = ["--pc-xpass"] if parse == generate.parse_args else []
        args = parse([*prefix, "--high-pass", "on", "--speed-step", "3",
                      "--high-pass-angle", "30", "15", "15", "20", "25"])
        assert args.high_pass_config["angles"] == [15, 20, 25, 30]
        assert args.high_pass_config["speeds"] == [20, 23, 26, 29]
        assert args.high_pass_speed_step == 3
        assert args.high_pass_cutoff is None
        assert args.high_pass_lane_cutoff == 2
        assert args.high_pass_control_cutoff == .5
        assert args.high_pass_config["schema_version"] == 2
        explicit = parse([*prefix, "--high-pass", "on", "--speed-step", "3", "--high-pass-speed-step", "2"])
        assert explicit.high_pass_config["speeds"] == [20, 22, 24, 26, 28, 30]


@pytest.mark.parametrize("flags", [
    ["--high-pass", "on"], ["--high-pass", "off"], ["--high-pass-angle", "20"],
])
def test_non_pc_rejected(flags):
    with pytest.raises(SystemExit):
        generate.parse_args(flags)


@pytest.mark.parametrize("flags", [
    ["--high-pass-angle", "0"], ["--high-pass-angle", "90"],
    ["--high-pass-angle", "nan"], ["--high-pass-angle"],
    ["--high-pass-speed-step", "0"], ["--high-pass-max-speed", "19"],
    ["--high-pass-min-speed", "-1"], ["--high-pass-cutoff", "0"],
    ["--high-pass-cutoff", "inf"], ["--high-pass-max-speed", "nan"],
])
def test_invalid_grid(flags):
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--high-pass", "on", *flags])


def test_explicit_tuning_requires_enabled():
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--high-pass-cutoff", "2"])


@pytest.mark.parametrize("parse", [generate.parse_args, location.parse_args])
def test_independent_cutoffs_and_deprecated_alias(parse):
    prefix = ["--pc-xpass"] if parse == generate.parse_args else []
    opts = [*prefix, "--high-pass", "on"]
    lane = parse([*opts, "--high-pass-lane-cutoff", "1.8"])
    assert (lane.high_pass_config["lane_cutoff"], lane.high_pass_config["control_cutoff"]) == (1.8, .5)
    control = parse([*opts, "--high-pass-control-cutoff", ".4"])
    assert (control.high_pass_config["lane_cutoff"], control.high_pass_config["control_cutoff"]) == (2., .4)
    with pytest.warns(FutureWarning, match="deprecated") as warnings:
        alias = parse([*opts, "--high-pass-cutoff", "1.5"])
    assert len(warnings) == 1
    assert (alias.high_pass_config["lane_cutoff"], alias.high_pass_config["control_cutoff"]) == (1.5, 1.5)
    assert "high_pass_cutoff" not in versions.settings(alias)
    for option in ("--high-pass-lane-cutoff", "--high-pass-control-cutoff"):
        with pytest.raises(SystemExit):
            parse([*opts, "--high-pass-cutoff", "2", option, "2"])


def test_configuration_api_alias():
    with pytest.warns(FutureWarning, match="deprecated"):
        config = hp.configuration("on", high_pass_cutoff=1.2)
    assert config["lane_cutoff"] == config["control_cutoff"] == 1.2
    for option in ("high_pass_lane_cutoff", "high_pass_control_cutoff"):
        with pytest.raises(ValueError, match="cannot be combined"):
            hp.configuration("on", high_pass_cutoff=2., **{option: 2.})


@pytest.mark.parametrize("option", ["--high-pass-lane-cutoff", "--high-pass-control-cutoff"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_independent_cutoffs(option, value):
    for parse in (generate.parse_args, location.parse_args):
        prefix = ["--pc-xpass"] if parse == generate.parse_args else []
        with pytest.raises(SystemExit):
            parse([*prefix, "--high-pass", "on", option, value])


def test_all_default_profiles_match_pinned_upstream_and_stop_bound():
    # Fixture generated with the unmodified pinned upstream function, 1001 samples
    # per profile; contains flight, first/subsequent impacts, and rolling tail.
    expected = np.load(Path(__file__).parent / "fixtures/high_pass_upstream.npz")
    config = hp.configuration("on")
    for angle in config["angles"]:
        for speed in config["speeds"]:
            path = hp.profile(speed, angle)
            reference = expected[f"v{int(speed)}_a{int(angle)}"]
            n = len(path)
            np.testing.assert_array_equal(path[:, [0, 2]].astype(np.float32), reference[:n, [0, 2]])
            assert not path.flags.writeable
            assert (np.diff(path[:, 0]) > 0).all()
            assert (path[:, 2] >= 0).all()
            assert path[:, 2].max() > .5
            assert path[-1, 0] >= 150 or abs(float(reference[-1, 0])-path[-1, 0]) <= .00101
            # At least one visible bounce after the first descending ground contact.
            ground = np.flatnonzero((path[:, 2] == 0) & (path[:, 1] > .1))
            assert len(ground) and np.any(path[ground[0]+1:, 2] > 0)
    assert hp.profile(20., 15.) is hp.profile(20., 15.)


def test_airborne_release_speed_and_gravity():
    path = hp.profile(30., 30.)
    dt, damping, g = .04, 1-.00648, 9.81
    vx0, vz0 = 30*math.cos(math.radians(30)), 15.
    assert path[1, 0]/dt == pytest.approx(damping*vx0)
    assert path[1, 2]/dt == pytest.approx(damping*vz0-g*dt)
    assert math.hypot(path[1, 0]/dt/damping, (path[1, 2]/dt+g*dt)/damping) == pytest.approx(30.)
    # Check the rotational symmetry used to share profiles across all azimuths.
    from itertools import islice
    from _vendor.montecarlo_pass_search.ball_motion import states
    for azimuth in (.7, 2.1):
        direction = np.array([math.cos(azimuth), math.sin(azimuth)])
        initial = np.array([vx0*direction[0], vx0*direction[1], vz0])
        rotated = np.array([pos for pos, _ in islice(states(np.zeros(3), initial), 120)])
        np.testing.assert_allclose(rotated[:, :2], path[:120, :1]*direction, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(rotated[:, 2], path[:120, 2], rtol=1e-13, atol=1e-12)
    # Check the rotational symmetry used to share profiles across all azimuths.
    from itertools import islice
    from _vendor.montecarlo_pass_search.ball_motion import states
    for azimuth in (.7, 2.1):
        direction = np.array([math.cos(azimuth), math.sin(azimuth)])
        initial = np.array([vx0*direction[0], vx0*direction[1], vz0])
        rotated = np.array([pos for pos, _ in islice(states(np.zeros(3), initial), 120)])
        np.testing.assert_allclose(rotated[:, :2], path[:120, :1]*direction, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(rotated[:, 2], path[:120, 2], rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("mode", ["share", "normal", "normal-one", "subtract", "subtract-one"])
@pytest.mark.parametrize("shared", [False, True])
def test_cutoff_before_prefix_and_endpoint_normalization(monkeypatch, mode, shared):
    # Cross both boundaries on ascent and descent, including exact equality.
    heights = np.array([0., .5, .5001, 2., 2.0001, 2., .5001, .5, 0.])
    monkeypatch.setattr(hp, "profile", lambda *args: np.column_stack((np.arange(9)*5., np.arange(9), heights)))
    config = hp.configuration("on", [20], 20, 20)
    details = {}
    ppm.compute_graph_pc_xpass_metrics(make_graph(), high_pass_config=config,
        min_speed=20, max_speed=20, speed_step=2, angle_step=90, radial_gridsize=5,
        endpoint_normalization=mode, lane_power=4., control_power=4. if shared else 3.,
        lane_inflection_point=0., control_inflection_point=0.,
        diagnostic_details=details, use_position_discount=False)
    lane, control = details["raw_lane"], details["raw_control"]
    np.testing.assert_array_equal(lane[:, 1, 0, :9][:, heights > 2], 0.)
    np.testing.assert_array_equal(control[:, 1, 0, :9][:, heights > .5], 0.)
    assert (lane[:, 1, 0, :9][:, heights <= 2] > 0).all()
    assert (control[:, 1, 0, :9][:, heights <= .5] > 0).all()
    assert not np.shares_memory(lane, control)
    assert np.isnan(control[:, 1, :, 9:]).all()
    margins = ppm._pc_xpass_arrival_margins(details["player_positions"],
        details["target_x"], details["target_y"], details["ball_times"])
    expected = ppm.pc_xpass_raw_control_with_params(margins, power=4. if shared else 3., inflection_point=0.)
    np.testing.assert_array_equal(control[:, 1, 0, 7], expected[:, 1, 0, 7])
    # Prefix lane interception still includes points that are too high to control.
    defender = lane[2, 1, 0, :9]
    np.testing.assert_allclose(details["lane_survival"][1, 0, 1:9],
                               1-np.maximum.accumulate(defender)[:8])
    for controls in details["receiver_control"].values():
        np.testing.assert_array_equal(controls[1, 0, :9][heights > .5], 0.)


@pytest.mark.parametrize("mode", ["share", "normal", "normal-one", "subtract", "subtract-one"])
@pytest.mark.parametrize("xt", [False, True])
def test_combined_ranking_and_all_normalizations(mode, xt):
    graph = obstructed_graph()
    config = hp.configuration("on")
    details, lanes = {}, []
    opts = settings(endpoint_normalization=mode, top_xt=xt)
    row = ppm.compute_graph_pc_xpass_metrics(graph, high_pass_config=config,
        diagnostic_details=details, lane_control_records=lanes, **opts)
    score = details["lane_survival"] * details["receiver_control"]["home_2"]
    ranking = score * ppm._pc_xpass_xt_values(details["target_x"], details["target_y"]) if xt else score
    ordered = sorted((i for i in range(score.size) if np.isfinite(score.flat[i]) and np.isfinite(ranking.flat[i])),
                     key=lambda i: (-ranking.flat[i], i))
    chosen = ordered[0]
    launch, angle, radius = np.unravel_index(chosen, score.shape)
    assert row["home_2__max_xpass"] == pytest.approx(np.clip(score.flat[chosen], 1e-4, 1-1e-4))
    for n in [3, 10]:
        assert row[f"home_2__top{n}_xpass"] == pytest.approx(np.clip(np.mean(score.ravel()[ordered[:n]]), 1e-4, 1-1e-4))
    distinct, seen = [], set()
    for i in ordered:
        profile, angle_i, _ = np.unravel_index(i, score.shape)
        candidate = (details["speeds"][profile], angle_i)
        if candidate not in seen:
            distinct.append(i)
            seen.add(candidate)
    for n in [1, 5]:
        assert row[f"home_2__top_pass{n}_xpass"] == pytest.approx(np.clip(np.mean(score.ravel()[distinct[:n]]), 1e-4, 1-1e-4))
        assert row[f"home_2__top_pass{n}_lane_survival"] == pytest.approx(np.mean(details["lane_survival"].ravel()[distinct[:n]]))
        assert row[f"home_2__top_pass{n}_control_prob"] == pytest.approx(np.mean(details["receiver_control"]["home_2"].ravel()[distinct[:n]]))
    assert row["home_2__top_pass1_xpass"] == row["home_2__max_xpass"]
    assert row["home_2__elevation"] == details["elevations"][launch]
    assert row["home_2__speed"] == details["speeds"][launch]
    assert row["home_2__target_z"] == details["ball_heights"][launch, 0, radius]
    assert row["home_2__flight_time"] == details["ball_times"][launch, 0, radius]
    assert np.prod([1-r["lane_control"] for r in lanes]) == pytest.approx(row["home_2__lane_survival"])
    assert lanes[0]["elevation"] == row["home_2__elevation"]
    pd.testing.assert_series_equal(row, ppm.compute_graphs_pc_xpass_metrics([graph], high_pass_config=config, **opts)[0])


def test_aerial_option_clears_obstructing_defender():
    g = obstructed_graph()
    opts = settings()
    off = ppm.compute_graph_pc_xpass_metrics(g, **opts)
    on = ppm.compute_graph_pc_xpass_metrics(g, high_pass_config=hp.configuration("on"), **opts)
    assert on["home_2__elevation"] > 0
    assert on["home_2__max_xpass"] > off["home_2__max_xpass"]


@pytest.mark.parametrize("xt", [False, True])
def test_top_pass_larger_than_number_of_speed_angle_pairs(xt):
    details = {}
    opts = {**settings(top_xt=xt), "angle_step": 90, "top_pass_values": [1, 100]}
    row = ppm.compute_graph_pc_xpass_metrics(obstructed_graph(),
        high_pass_config=hp.configuration("on", high_pass_min_speed=20, high_pass_max_speed=20),
        diagnostic_details=details, **opts)
    score = details["lane_survival"] * details["receiver_control"]["home_2"]
    ranking = score * ppm._pc_xpass_xt_values(details["target_x"], details["target_y"]) if xt else score
    selected = []
    # All five profiles have speed 20, so each azimuth contributes just one pass.
    for angle in range(score.shape[1]):
        candidates = [(p, angle, d) for p in range(score.shape[0]) for d in range(score.shape[2])
                      if np.isfinite(score[p, angle, d]) and np.isfinite(ranking[p, angle, d])]
        if candidates:
            selected.append(max(candidates, key=lambda index: ranking[index]))
    assert len(selected) <= 4
    expected = np.mean([score[index] for index in selected])
    assert row["home_2__top_pass100_xpass"] == pytest.approx(np.clip(expected, 1e-4, 1-1e-4))
    assert row["home_2__top_pass1_xpass"] == row["home_2__max_xpass"]


def test_metadata_fingerprints_and_corruption():
    policy = ppm.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER
    old = ppm.pc_xpass_metadata(policy)
    off = ppm.pc_xpass_metadata(policy, high_pass_config=None)
    assert old == off
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(old) == ppm.pc_xpass_lane_survival_metadata_fingerprint(off)
    a = ppm.pc_xpass_metadata(policy, high_pass_config=hp.configuration("on"))
    b = ppm.pc_xpass_metadata(policy, high_pass_config=hp.configuration("on", high_pass_lane_cutoff=3))
    c = ppm.pc_xpass_metadata(policy, high_pass_config=hp.configuration("on", high_pass_control_cutoff=1))
    assert a["top_pass_definition"] == hp.TOP_PASS_DEFINITION
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(a) != ppm.pc_xpass_lane_survival_metadata_fingerprint(b)
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(a) != ppm.pc_xpass_lane_survival_metadata_fingerprint(c)
    altered_ranking = {**a, "top_pass_definition": "different"}
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(a) != ppm.pc_xpass_lane_survival_metadata_fingerprint(altered_ranking)
    with pytest.raises(ValueError, match="top_pass_definition"):
        ppm.validate_pc_xpass_ball_motion_metadata(altered_ranking)
    broken = copy.deepcopy(a)
    broken["high_pass_config"]["physics"]["restitution"] = .5
    with pytest.raises(ValueError, match="high-pass"):
        ppm.validate_pc_xpass_ball_motion_metadata(broken)
    missing = copy.deepcopy(old)
    missing["generation_settings"] = {"high_pass": "on"}
    with pytest.raises(ValueError, match="high-pass"):
        ppm.validate_pc_xpass_ball_motion_metadata(missing)


def test_normal_and_location_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "normal", "--high-pass", "on", "--speed-step", "3"])
    versions.start_generation(args)
    resumed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "normal"])
    assert resumed.high_pass_config == args.high_pass_config
    with pytest.raises(ValueError, match="high-pass"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "normal", "--high-pass-lane-cutoff", "3"])
    legacy = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "ground"])
    versions.start_generation(legacy)
    with pytest.raises(ValueError, match="high-pass"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "ground", "--high-pass", "on"])
    loc = location.parse_args(["--high-pass", "on", "--high-pass-speed-step", "4"])
    loc.pc_xpass_id = "loc"
    versions.start_generation(loc, location=True)
    with pytest.warns(UserWarning, match="cached"):
        resumed = location.parse_args(["--pc-xpass-id", "loc", "--high-pass-control-cutoff", "1"])
    assert resumed.high_pass_config == loc.high_pass_config


def legacy_aerial_metadata(metadata):
    """Represent the pre-revision on-disk contract, without changing its meaning."""
    metadata = copy.deepcopy(metadata)
    config = metadata["high_pass_config"]
    config.pop("schema_version")
    config["cutoff"] = config.pop("lane_cutoff")
    config.pop("control_cutoff")
    metadata["top_pass_definition"] = hp.LEGACY_TOP_PASS_DEFINITION
    if "generation_settings" in metadata:
        saved = metadata["generation_settings"]
        saved["high_pass_cutoff"] = saved.pop("high_pass_lane_cutoff")
        saved.pop("high_pass_control_cutoff")
    return metadata


@pytest.mark.parametrize("location_mode", [False, True])
def test_legacy_version_readable_but_cannot_resume(tmp_path, monkeypatch, location_mode):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    parse = location.parse_args if location_mode else generate.parse_args
    prefix = [] if location_mode else ["--pc-xpass"]
    if location_mode:
        args = parse(["--high-pass", "on"])
        args.pc_xpass_id = "legacy"
    else:
        args = parse([*prefix, "--pc-xpass-id", "legacy", "--high-pass", "on"])
    versions.start_generation(args, location=location_mode)
    root = versions.version_root("legacy", location_mode)
    old = legacy_aerial_metadata(versions.read_metadata(root))
    versions.atomic_json(root / "metadata.json", old)
    before = (root / "metadata.json").read_bytes()
    assert versions.read_metadata(root)["high_pass_config"]["cutoff"] == 2.
    with pytest.raises(ValueError, match="read-only.*new pc-xPass version"):
        parse([*prefix, "--pc-xpass-id", "legacy"])
    with pytest.raises(ValueError, match="read-only"):
        versions.start_generation(args, location=location_mode)
    if location_mode:
        with pytest.raises(ValueError, match="read-only"):
            parse(["--pc-xpass-cache-dir", str(root)])
    assert (root / "metadata.json").read_bytes() == before


def test_legacy_stored_scores_and_visualization_without_simulation(tmp_path, monkeypatch):
    metadata = legacy_aerial_metadata(ppm.pc_xpass_metadata(
        ppm.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, high_pass_config=hp.configuration("on"),
        top_n=None, top_pass_values=[5]))
    versions.atomic_json(tmp_path / "metadata.json", metadata)
    (tmp_path / "matches").mkdir()
    row = pd.DataFrame([{"action_index": 7, "home_2__top_pass5_xpass": .371,
                         "home_2__elevation": 25.}])
    row.to_parquet(tmp_path / "matches/example.parquet", index=False)
    monkeypatch.setattr(hp, "profile", lambda *args: (_ for _ in ()).throw(AssertionError("Legacy read called simulator")))
    assert hp.validate(metadata["high_pass_config"], allow_legacy=True) == metadata["high_pass_config"]
    with pytest.raises(ValueError, match="read-only"):
        hp.validate(metadata["high_pass_config"])
    ppm.validate_pc_xpass_lane_survival_mode_cache_metadata(metadata, "top_pass_5")
    selected = ppm.load_runtime_physical_xpass_visualization_component(tmp_path, "example", 7,
        metric="top_pass5_xpass", x_pass_version="top-pass5")
    assert selected["home_2"] == .371
    assert selected.attrs["pc_xpass_elevations"] == {"home_2": 25.}
    graphs = ppm.attach_physical_xpass_read_only_to_graphs([make_graph()], torch.stack([make_label()]),
        cache_dir=tmp_path, match_id="example", metric="top_pass5_xpass")
    assert graphs[0].physical_xpass[1].item() == pytest.approx(.371)
    # Reproduce the old fingerprint's payload, with no newly introduced fields.
    import hashlib
    payload = {key: metadata.get(key) for key in ppm.PC_XPASS_LANE_SURVIVAL_FINGERPRINT_KEYS
               if key not in {"ball_dec", "physics_version"} or key in metadata}
    payload["high_pass_config"] = metadata["high_pass_config"]
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert ppm.pc_xpass_lane_survival_metadata_fingerprint(metadata) == expected


@pytest.mark.parametrize("operation", ["compute", "generate", "repair", "reuse"])
def test_legacy_aerial_rejects_numerical_work(tmp_path, operation):
    current = hp.configuration("on")
    metadata = legacy_aerial_metadata(ppm.pc_xpass_metadata(
        ppm.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, high_pass_config=current, **settings()))
    source = tmp_path / "legacy"
    versions.atomic_json(source / "metadata.json", metadata)
    before = (source / "metadata.json").read_bytes()
    graph = obstructed_graph()
    opts = dict(source=ppm.PC_XPASS_SOURCE, high_pass_config=current, num_workers=1, **settings())
    items = [dict(match_id="example", graphs=[graph], labels=torch.stack([make_label()]))]
    with pytest.raises(ValueError, match="read-only.*new pc-xPass version"):
        if operation == "compute":
            ppm.compute_graph_pc_xpass_metrics(graph, high_pass_config=metadata["high_pass_config"], **settings())
        elif operation == "reuse":
            ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path / "new",
                reuse_cache_dir=source, **opts)
        else:
            ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=source,
                refresh=operation == "repair", **opts)
    assert (source / "metadata.json").read_bytes() == before


def test_alias_resume_uses_resolved_cutoffs(tmp_path, monkeypatch):
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path)
    with pytest.warns(FutureWarning, match="deprecated"):
        args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "alias", "--high-pass", "on", "--high-pass-cutoff", "1.2"])
    versions.start_generation(args)
    resumed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "alias"])
    assert resumed.high_pass_config == args.high_pass_config
    with pytest.warns(FutureWarning, match="deprecated"):
        explicit = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "alias", "--high-pass-cutoff", "1.2"])
    assert explicit.high_pass_config == args.high_pass_config


def test_runtime_cache_lane_export_repair_and_cache_only(tmp_path, monkeypatch):
    graph = obstructed_graph()
    items = [{"match_id": "example", "graphs": [graph], "labels": torch.stack([make_label()])}]
    config = hp.configuration("on")
    opts = dict(source=ppm.PC_XPASS_SOURCE, num_workers=1, high_pass_config=config,
                export_lane_control=True, **settings())
    first = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)
    assert first["cache_written"] == 1
    second = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)
    assert second["cache_hits"] == 1
    lane = pd.read_parquet(tmp_path / "lane_control/example.parquet")
    assert set(hp.LANE_DETAILS) <= set(lane)
    manifest = json.loads((tmp_path / "lane_control/coverage/example.json").read_text())
    assert manifest["schema_version"] == 2
    row = ppm.load_physical_xpass_match(tmp_path, "example")
    assert lane.iloc[0].elevation == row.iloc[0]["home_2__elevation"]
    ppm.validate_x_pass_version_available(tmp_path, x_pass_version="top-pass5", metric="top_pass5_xpass")
    ppm.validate_pc_xpass_lane_survival_mode_cache_metadata(json.loads((tmp_path / "metadata.json").read_text()), "top_pass_5")
    # Missing new diagnostics cannot count as a complete enabled row.
    row.drop(columns=["home_2__elevation"]).to_parquet(tmp_path / "matches/example.parquet", index=True)
    repaired = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **opts)
    assert repaired["cache_written"] == 1
    with pytest.raises(ValueError, match="high-pass"):
        ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **{**opts, "high_pass_config": None})
    monkeypatch.setattr(hp, "profile", lambda *a: (_ for _ in ()).throw(AssertionError("Cache read invoked simulator")))
    selected = ppm.load_runtime_physical_xpass_visualization_component(tmp_path, "example", 7,
        metric="top_pass5_xpass", x_pass_version="top-pass5")
    assert selected.attrs["pc_xpass_elevations"]["home_2"] > 0
    table = ppm.load_runtime_physical_xpass_visualization_table(tmp_path, "example", [7], metric="max_xpass")
    assert table.attrs["pc_xpass_elevations_by_action"][7]["home_2"] > 0


def test_generation_forwards_aerial_settings(tmp_path):
    args = generate.parse_args(["--pc-xpass", "--high-pass", "on", "--num-workers", "1"])
    with patch.object(generate, "prewarm_physical_xpass_runtime_cache", return_value={}) as prewarm:
        generate.prewarm_runtime_items([{"match_id": "example"}], cache_dir=tmp_path, args=args, progress_desc="test")
    assert prewarm.call_args.kwargs["high_pass_config"] == args.high_pass_config


def test_production_cli_writes_metadata_and_resumes(tmp_path, monkeypatch):
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
    state.to_csv(inputs / "game_state_1.csv", index=False)
    state.to_csv(inputs / "game_state_2.csv", index=False)
    pd.DataFrame({"higher_state_id": [1]}).to_csv(inputs / "modification.csv", index=False)
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", tmp_path / "cache")
    common = ["generate_physical_xpass.py", "--pc-xpass", "--pc-xpass-id", "cli_smoke",
        "--no-sportec", "--no-skillcorner", "--no-hawkeye", "--benchmark-input-dir", str(inputs.parent),
        "--benchmark-modification", "1", "--num-workers", "1", "--angle-step", "90",
        "--radial-gridsize", "10", "--top-pass", "5", "--export-lane-control"]
    monkeypatch.setattr(sys, "argv", common + ["--high-pass", "on"])
    generate.main()
    root = tmp_path / "cache/cli_smoke"
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    dataset = json.loads((root / "benchmark/metadata.json").read_text())
    assert dataset["high_pass_config"] == metadata["high_pass_config"]
    assert metadata["high_pass_config"]["lane_cutoff"] == 2.
    assert metadata["high_pass_config"]["control_cutoff"] == .5
    assert metadata["top_pass_definition"] == hp.TOP_PASS_DEFINITION
    assert dataset["lane_control_export"]["schema_version"] == 2
    assert set(hp.LANE_DETAILS) <= set(dataset["lane_control_export"]["columns"])
    assert dataset["top_pass_definition"] == hp.TOP_PASS_DEFINITION
    before = pd.read_parquet(root / "benchmark/matches/modification_1_game_state_1.parquet")
    monkeypatch.setattr(sys, "argv", common)
    monkeypatch.setattr(hp, "profile", lambda *args: (_ for _ in ()).throw(AssertionError("Resume recomputed a cached trajectory")))
    generate.main()
    after = pd.read_parquet(root / "benchmark/matches/modification_1_game_state_1.parquet")
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("margin", ["tta", "distance", "reachability"])
def test_movement_modes_receive_all_flight_times(accel_table, distance_table, reachability_artifact, margin):
    import pc_xpass_acceleration as acceleration
    import pc_xpass_distance as spatial
    import reachability
    options = dict(margin=margin, reaction_time=0.)
    if margin == "tta":
        options["acceleration_config"] = acceleration.configuration(7, "fixture")
    elif margin == "distance":
        options["acceleration_config"] = spatial.configuration(7, "spatial", max_speeds=(5.,))
    else:
        options["reachability_config"] = reachability.configuration("reachability", {"model_id": "test"})
    details = {}
    config = hp.configuration("on", [20], 20, 20)
    opts = {**settings(), **options, "angle_step": 90, "radial_gridsize": 10}
    row = ppm.compute_graph_pc_xpass_metrics(obstructed_graph(), high_pass_config=config, diagnostic_details=details, **opts)
    assert details["ball_times"].shape[0] == 2
    assert np.isfinite(row["home_2__max_xpass"])
    raw = details["raw_control"][:, 1]
    above = np.broadcast_to(details["ball_heights"][1] > .5, raw.shape)
    np.testing.assert_array_equal(raw[above & np.isfinite(raw)], 0.)


@pytest.mark.parametrize("dataset", ["sportec", "skillcorner"])
def test_transactional_match_caches_retain_elevation(tmp_path, dataset):
    import pc_xpass_match_cache as cache
    from test_pc_xpass_match_cache import descriptor
    root = tmp_path / dataset
    cache.initialize(root)
    item = dict(match_id="one", graphs=[obstructed_graph()], labels=torch.stack([make_label()]),
                frame_scope="frame_id", state_frame_id=[7])
    opts = dict(source=ppm.PC_XPASS_SOURCE, high_pass_config=hp.configuration("on"),
                export_lane_control=True, num_workers=1, **settings())
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.register("one", descriptor())
        transaction.expect([item])
        assert ppm.prewarm_physical_xpass_runtime_cache([item], cache_dir=root, **opts)["cache_written"] == 1
        transaction.commit(export_lane=True)
    lane = pd.read_parquet(root / "lane_control/match.parquet")
    assert set(hp.LANE_DETAILS) <= set(lane)
    assert lane.elevation.iloc[0] > 0
    with cache.MatchAccumulator(root, "match") as transaction:
        transaction.expect([item])
        assert ppm.prewarm_physical_xpass_runtime_cache([item], cache_dir=root, **opts)["cache_hits"] == 1
        transaction.commit(export_lane=True)


def test_windows_workers_equal_serial(tmp_path):
    items = [dict(match_id=f"example_{i}", graphs=[obstructed_graph()], labels=torch.stack([make_label()])) for i in range(2)]
    opts = dict(source=ppm.PC_XPASS_SOURCE,
                high_pass_config=hp.configuration("on", high_pass_lane_cutoff=1.8, high_pass_control_cutoff=.35),
                export_lane_control=True, physical_batch_size=1, **settings())
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=serial, num_workers=1, **opts)
    ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=parallel, num_workers=2, **opts)
    for item in items:
        mid = item["match_id"]
        pd.testing.assert_frame_equal(pd.read_parquet(serial / f"matches/{mid}.parquet"),
                                      pd.read_parquet(parallel / f"matches/{mid}.parquet"))
        pd.testing.assert_frame_equal(pd.read_parquet(serial / f"lane_control/{mid}.parquet"),
                                      pd.read_parquet(parallel / f"lane_control/{mid}.parquet"))


@pytest.mark.parametrize("adapter", ["direct", "benchmark", "hawkeye", "skillcorner"])
def test_visualizer_labels_selected_elevation(adapter):
    import matplotlib.pyplot as plt
    from datatools.viz_snapshot import SnapshotVisualizer
    snapshot = pd.DataFrame({"home_1_x": [10.], "home_1_y": [20.],
                             "home_1_vx": [0.], "home_1_vy": [0.],
                             "ball_x": [10.], "ball_y": [20.]})
    probs = pd.Series({"home_1": .8})
    probs.attrs["pc_xpass_elevations"] = {"home_1": 25.}
    if adapter != "direct":
        import importlib
        module = importlib.import_module(f"scripts.visualize_{adapter}")
        table = pd.DataFrame({"home_1": [.8]}, index=[7])
        table.attrs["pc_xpass_elevations_by_action"] = {7: {"home_1": 25.}}
        probs = module._probs_for_component_frame("physical_xpass", {"physical_xpass": table}, 7)
    visualizer = SnapshotVisualizer(snapshot, player_annots=probs, show_velocities=False)
    fig, ax = visualizer.plot(annot_type="physical_xpass", show=False)
    assert any("best 25°" in text.get_text() for text in ax.texts)
    plt.close(fig)
