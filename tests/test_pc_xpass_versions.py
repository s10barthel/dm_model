from argparse import Namespace
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

import pc_xpass_versions as versions
import physical_pass_model as physics
from scripts import generate_physical_xpass as generate
from scripts import run_hawkeye_loc as location
from test_physical_xpass import make_graph, make_label


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "pc_xpass"
    monkeypatch.setattr(versions.config, "PC_XPASS_DIR", root)
    return root


def test_arrival_times_and_stopping_boundary():
    stop = 25 / 0.9
    times, reachable = physics.pc_xpass_ball_arrival_times(np.array([0, 10, stop, stop + 1e-7]), np.array([5.0]))
    np.testing.assert_array_equal(reachable.ravel(), [True, True, True, False])
    assert times[0, 0, 0] == 0
    assert times[0, 0, 2] == pytest.approx(5 / 0.45)
    assert 5 * times[0, 0, 1] - 0.225 * times[0, 0, 1] ** 2 == pytest.approx(10)
    constant, valid = physics.pc_xpass_ball_arrival_times(np.array([0, 30, 200]), np.array([10.0]), 0)
    np.testing.assert_array_equal(constant.ravel(), [0, 3, 20])
    assert valid.all()
    assert physics.pc_xpass_ball_arrival_times(np.array([30.]), np.array([10.]))[0].item() == pytest.approx(3.2355, abs=0.001)


def test_drag_arrival_times_shape_monotonicity_and_reachability():
    distances = np.array([0., 1e-8, 30., 200.])
    speeds = np.array([3., 10., 15., 20.])
    times, reachable = physics.pc_xpass_ball_arrival_times(distances, speeds, "drag")
    assert times.shape == reachable.shape == (4, 1, 4)
    assert reachable.dtype == bool and reachable.all()
    np.testing.assert_array_equal(times[:, 0, 0], 0.)
    np.testing.assert_allclose(times[1:, 0, 2], [3.71621413629886, 2.477476090865907, 1.85810706814943])
    np.testing.assert_allclose(times[:, 0, 1], distances[1] / speeds, rtol=1e-9)
    assert (np.diff(times, axis=2) > 0).all()
    assert (np.diff(times[:, 0, 1:], axis=0) < 0).all()
    assert np.isfinite(times).all()
    _, numeric_reachable = physics.pc_xpass_ball_arrival_times(distances, speeds, 3.)
    assert not numeric_reachable[0, 0, 2]


@pytest.mark.parametrize("value, expected", [("drag", "drag"), ("0", 0.), ("0.45", .45)])
def test_ball_motion_cli_values(value, expected):
    assert generate.parse_args(["--pc-xpass", "--ball-dec", value]).ball_dec == expected
    assert location.parse_args(["--ball-dec", value]).ball_dec == expected
    assert generate.parse_args(["--pc-xpass"]).ball_dec == .45
    assert location.parse_args([]).ball_dec == .45


@pytest.mark.parametrize("deceleration", [-1, float("nan"), float("inf"), "unknown"])
def test_invalid_deceleration(deceleration):
    with pytest.raises(ValueError):
        physics.pc_xpass_ball_arrival_times(np.array([1]), np.array([5]), deceleration)
    with pytest.raises(SystemExit):
        generate.parse_args(["--pc-xpass", "--ball-dec", str(deceleration)])
    with pytest.raises(SystemExit):
        location.parse_args(["--ball-dec", str(deceleration)])


@pytest.mark.parametrize("value", ["0", "drag"])
def test_pc_only_flag(value):
    with pytest.raises(SystemExit):
        generate.parse_args(["--ball-dec", value])


@pytest.mark.parametrize("ball_dec", [.45, "drag"])
def test_scoring_reachability_and_batch_agree(ball_dec):
    graph = make_graph()
    kwargs = dict(min_speed=3, max_speed=3, speed_step=1, angle_step=90,
                  radial_gridsize=5, top_n=100, top_pass_values=[100], ball_dec=ball_dec,
                  use_position_discount=False)
    row = physics.compute_graph_pc_xpass_metrics(graph, **kwargs)
    batch = physics.compute_graphs_pc_xpass_metrics([graph], **kwargs)[0]
    pd.testing.assert_series_equal(row, batch)
    for key, value in row.items():
        if ball_dec != "drag" and key.endswith("__distance") and np.isfinite(value):
            assert value <= 10
        if key.endswith("__speed") and np.isfinite(value):
            assert value == 3
    if ball_dec == "drag":
        prefix = "home_2"
        assert np.isfinite(row[f"{prefix}__max_xpass"])
        assert np.isfinite(row[f"{prefix}__top_pass100_xpass"])
        product = row[f"{prefix}__lane_survival"] * row[f"{prefix}__control_prob"]
        assert row[f"{prefix}__max_xpass"] == pytest.approx(np.clip(product, 1e-4, 1-1e-4))
        for suffix in ["lane_survival", "control_prob"]:
            assert 0 <= row[f"{prefix}__top_pass100_{suffix}"] <= 1


def test_no_reachable_sampled_endpoints(monkeypatch):
    monkeypatch.setattr(physics, "_pc_xpass_r_grid", lambda *a, **k: np.array([20., 30.]))
    row = physics.compute_graph_pc_xpass_metrics(make_graph(), min_speed=3, max_speed=3, angle_step=90)
    assert row.isna().all()
    drag = physics.compute_graph_pc_xpass_metrics(make_graph(), min_speed=3, max_speed=3, angle_step=90, ball_dec="drag")
    assert drag.notna().any()


def test_fingerprints_differ():
    a = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec=0)
    b = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec=0.45)
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(a) != physics.pc_xpass_lane_survival_metadata_fingerprint(b)


@pytest.mark.parametrize("ball_dec", [0., .45, "drag"])
def test_normal_create_resume_and_latest(cache_root, ball_dec):
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "first", "--ball-dec", str(ball_dec)])
    versions.start_generation(args)
    dataset = versions.cache_dir("hawkeye", args)
    versions.atomic_json(dataset / "metadata.json", physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec=ball_dec))
    versions.finish_generation(args, coverage={"hawkeye": {"rows": 1}})
    reader = Namespace(pc_xpass_id=None)
    assert versions.cache_dir("hawkeye", reader) == dataset
    assert reader.pc_xpass_id == "first"
    resumed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "first"])
    assert resumed.ball_dec == ball_dec
    versions.start_generation(resumed)
    versions.finish_generation(resumed)
    assert len(versions.read_metadata(cache_root / "first")["invocations"]) == 2
    with pytest.raises(ValueError, match="cached value"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "first", "--ball-dec", "0.2"])
    with pytest.raises(FileNotFoundError, match="sportec"):
        versions.cache_dir("sportec", reader)


def test_failed_and_dry_generation_never_publish(cache_root):
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "failed"])
    versions.start_generation(args)
    versions.finish_generation(args, success=False)
    assert not (cache_root / "latest.json").exists()
    dry = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "dry", "--dry-run"])
    versions.start_generation(dry)
    versions.finish_generation(dry)
    assert not (cache_root / "dry").exists()


@pytest.mark.parametrize("ball_dec", [0., "drag"])
def test_location_fresh_and_explicit_reuse(cache_root, ball_dec):
    args = location.parse_args(["--ball-dec", str(ball_dec)])
    versions.start_generation(args, location=True)
    root = Path(args.pc_xpass_cache_dir)
    assert root.parent == cache_root / "hawkeye_loc"
    args.angle_step = 90
    args.radial_gridsize = 10
    args.num_workers = 1
    # Keep the manifest's canonical settings consistent with this small smoke run.
    manifest = versions.read_metadata(root)
    manifest["generation_settings"] = versions.settings(args)
    versions.atomic_json(root / "metadata.json", manifest)
    item = {"match_id": "synthetic", "graphs": [make_graph()], "labels": torch.stack([make_label()])}
    first = generate.prewarm_runtime_items([item], cache_dir=root, args=args)
    second = generate.prewarm_runtime_items([item], cache_dir=root, args=args)
    assert first["cache_misses"] == 1
    assert second["cache_hits"] == 1
    assert "generation_settings" in versions.read_metadata(root)
    extended_item = {"match_id": "synthetic", "graphs": [make_graph(), make_graph()],
                     "labels": torch.stack([make_label(), make_label(action_index=8)])}
    extended = generate.prewarm_runtime_items([extended_item], cache_dir=root, args=args)
    assert extended["cache_hits"] == 1
    assert extended["cache_misses"] == 1
    versions.finish_generation(args)
    with pytest.warns(UserWarning, match="cached value"):
        reused = location.parse_args(["--pc-xpass-id", args.pc_xpass_id, "--ball-dec", "0.45"])
    assert reused.ball_dec == ball_dec
    versions.start_generation(reused, location=True)
    assert Path(reused.pc_xpass_cache_dir) == root
    assert not (cache_root / "latest.json").exists()
    fresh = location.parse_args([])
    versions.start_generation(fresh, location=True)
    assert fresh.pc_xpass_id != args.pc_xpass_id
    with pytest.raises(FileNotFoundError):
        location.parse_args(["--pc-xpass-id", "unknown"])


@pytest.mark.parametrize("value", ["../escape", "hawkeye_loc", "sportec", "a/b", "a\\b", "CON", "trail."])
def test_invalid_ids(value):
    with pytest.raises(ValueError):
        versions.validate_id(value)


def test_explicit_directory_conflict_and_legacy_deceleration(cache_root, tmp_path):
    with pytest.raises(ValueError, match="directory override"):
        location.parse_args(["--pc-xpass-id", "some", "--pc-xpass-cache-dir", str(tmp_path)])
    versions.atomic_json(tmp_path / "metadata.json", {"source": "pc_xpass"})
    args = location.parse_args(["--pc-xpass-cache-dir", str(tmp_path)])
    assert args.ball_dec == 0
    versions.start_generation(args, location=True)
    assert args.pc_xpass_id is None
    assert not cache_root.exists()


@pytest.mark.parametrize("ball_dec", [.7, "drag"])
def test_serial_and_worker_generation_match(cache_root, ball_dec):
    items = [{"match_id": "one", "graphs": [make_graph(), make_graph()],
              "labels": torch.stack([make_label(), make_label(action_index=8)])}]
    results = []
    for workers in (1, 2):
        args = generate.parse_args([
            "--pc-xpass", "--pc-xpass-id", f"workers{workers}", "--ball-dec", str(ball_dec),
            "--min-speed", "3", "--max-speed", "3", "--angle-step", "90",
            "--radial-gridsize", "5", "--num-workers", str(workers), "--physical-batch-size", "1",
        ])
        versions.start_generation(args)
        root = versions.cache_dir("hawkeye", args)
        first = generate.prewarm_runtime_items(items, cache_dir=root, args=args)
        assert first["cache_written"] == 2
        second = generate.prewarm_runtime_items(items, cache_dir=root, args=args)
        assert second["cache_hits"] == 2
        assert versions.read_metadata(root)["ball_dec"] == ball_dec
        frame = physics.load_physical_xpass_match(root, "one")
        reader = Namespace(pc_xpass_id=args.pc_xpass_id)
        assert versions.cache_dir("hawkeye", reader) == root
        probs = physics.load_runtime_physical_xpass_visualization_component(
            root, "one", int(frame.iloc[0].action_index), metric="max_xpass", x_pass_version="max")
        assert probs.notna().any()
        assert physics.validate_pc_xpass_lane_survival_mode_cache_metadata(
            frame.attrs["physical_xpass_cache_metadata"], "max") == "max"
        results.append(frame)
    pd.testing.assert_frame_equal(results[0], results[1])


def test_location_cached_settings_preserve_models_and_override_alias(cache_root):
    args = location.parse_args(["--ball-dec", "0", "--pass-height-model-id", "pass_height/old"])
    versions.start_generation(args, location=True)
    with pytest.warns(UserWarning):
        selected = location.parse_args([
            "--pc-xpass-id", args.pc_xpass_id, "--ball-dec", "-1",
            "--top-pass", "5", "--pass-height-model-id", "pass_height/new", "--limit", "2",
        ])
    assert selected.ball_dec == 0
    assert selected.x_pass_version == args.x_pass_version
    assert selected.pass_height_model_id == "pass_height/new"
    assert selected.limit == 2


def test_latest_is_pinned_and_failed_runs_do_not_replace_it(cache_root):
    for name in ("old", "new"):
        args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", name])
        versions.start_generation(args)
        root = versions.cache_dir("hawkeye", args)
        versions.atomic_json(root / "metadata.json", {})
        versions.finish_generation(args)
        if name == "old":
            reader = Namespace(pc_xpass_id=None)
            pinned = versions.cache_dir("hawkeye", reader)
    assert versions.cache_dir("hawkeye", reader) == pinned
    failed = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "failed"])
    versions.start_generation(failed)
    versions.finish_generation(failed, success=False)
    current = Namespace(pc_xpass_id=None)
    versions.cache_dir("hawkeye", current)
    assert current.pc_xpass_id == "new"


@pytest.mark.parametrize("ball_dec, fingerprint", [
    (0., "df0a3b3aac1cc4f04909a00dafb897e251d4d864ef7474cdcc805a7482a7cdb3"),
    (.45, "7afb8d999cf04881c86acbac41c381071d0d20727893ad33d8f2580c38b610a3"),
    (.7, "aca41f65a7962ce50fce8db5873ed9c8d425958ce2324059778376af4b1df4a3"),
])
def test_numeric_fingerprint_backwards_compatibility(ball_dec, fingerprint):
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec=ball_dec)
    assert metadata["physics_version"] == 2
    assert isinstance(metadata["ball_dec"], float)
    assert "ball_drag" not in metadata
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(metadata) == fingerprint


def test_drag_metadata_and_fingerprint():
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec="drag")
    assert metadata["physics_version"] == 3 and metadata["ball_dec"] == "drag"
    assert metadata["ball_drag"] == {
        "model": "quadratic_drag", "air_density_kg_m3": 1.22,
        "drag_coefficient": .25, "cross_sectional_area_m2": .038,
        "mass_kg": .42, "k_m_inv": 1.22 * .25 * .038 / (2 * .42),
    }
    assert physics.validate_pc_xpass_ball_motion_metadata(metadata) == "drag"
    numeric = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
    fingerprint = physics.pc_xpass_lane_survival_metadata_fingerprint(metadata)
    assert fingerprint != physics.pc_xpass_lane_survival_metadata_fingerprint(numeric)
    altered = copy.deepcopy(metadata)
    altered["ball_drag"]["k_m_inv"] *= 2
    assert fingerprint != physics.pc_xpass_lane_survival_metadata_fingerprint(altered)
    numeric.pop("ball_dec")
    numeric.pop("physics_version")
    assert physics.validate_pc_xpass_ball_motion_metadata(numeric) == 0
    assert physics.pc_xpass_lane_survival_metadata_fingerprint(numeric) == "4585c945ed3144ad3cdaf1bb185ead3f40835b31d2abf9cf510359f28c6fc87f"


@pytest.mark.parametrize("mutation", ["wrong_k", "wrong_constant", "missing_drag", "missing_dec", "wrong_version", "missing_version", "mixed_settings"])
def test_reject_incompatible_drag_cache(cache_root, mutation):
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "drag", "--ball-dec", "drag"])
    versions.start_generation(args)
    root = cache_root / "drag"
    metadata = versions.read_metadata(root)
    if mutation == "wrong_k":
        metadata["ball_drag"]["k_m_inv"] *= 2
    elif mutation == "wrong_constant":
        metadata["ball_drag"]["mass_kg"] = .5
    elif mutation == "missing_drag":
        metadata.pop("ball_drag")
    elif mutation == "missing_dec":
        metadata.pop("ball_dec")
    elif mutation == "wrong_version":
        metadata["physics_version"] = 2
    elif mutation == "missing_version":
        metadata.pop("physics_version")
    else:
        metadata["generation_settings"]["ball_dec"] = .45
    versions.atomic_json(root / "metadata.json", metadata)
    with pytest.raises(ValueError, match="new version"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "drag"])
    with pytest.raises(ValueError):
        physics._ensure_runtime_physical_xpass_cache(
            root, source=physics.PC_XPASS_SOURCE,
            teammate_policy=physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
            speed_aggregation=physics.PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
            ball_dec="drag", create_if_missing=False)


@pytest.mark.parametrize("cached, requested", [("drag", .45), (.45, "drag")])
def test_reject_mixed_motion_cache_reuse(tmp_path, cached, requested):
    source = tmp_path / "source"
    metadata = physics.pc_xpass_metadata(physics.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, ball_dec=cached)
    versions.atomic_json(source / "metadata.json", metadata)
    with pytest.raises(ValueError, match="ball deceleration"):
        physics.prewarm_physical_xpass_runtime_cache(
            [], cache_dir=tmp_path / "target", source=physics.PC_XPASS_SOURCE,
            reuse_cache_dir=source, ball_dec=requested)


@pytest.mark.parametrize("cached, requested", [("drag", ".45"), (.45, "drag")])
def test_location_inherits_cached_motion_law(cache_root, cached, requested):
    args = location.parse_args(["--ball-dec", str(cached)])
    versions.start_generation(args, location=True)
    with pytest.warns(UserWarning, match="cached value"):
        selected = location.parse_args(["--pc-xpass-id", args.pc_xpass_id, "--ball-dec", requested])
    assert selected.ball_dec == cached
    with pytest.warns(UserWarning, match="cached value"):
        selected = location.parse_args(["--pc-xpass-id", args.pc_xpass_id, "--ball-dec", "-1"])
    assert selected.ball_dec == cached


def test_drag_rejects_legacy_directory_without_motion_metadata(tmp_path):
    versions.atomic_json(tmp_path / "metadata.json", {"source": "pc_xpass"})
    with pytest.raises(ValueError, match="different ball deceleration"):
        location.parse_args(["--pc-xpass-cache-dir", str(tmp_path), "--ball-dec", "drag"])



@pytest.mark.parametrize("cached, requested", [("drag", ".45"), (.45, "drag")])
def test_normal_resume_rejects_motion_law_change(cache_root, cached, requested):
    args = generate.parse_args(["--pc-xpass", "--pc-xpass-id", "motion", "--ball-dec", str(cached)])
    versions.start_generation(args)
    with pytest.raises(ValueError, match="cached value"):
        generate.parse_args(["--pc-xpass", "--pc-xpass-id", "motion", "--ball-dec", requested])
