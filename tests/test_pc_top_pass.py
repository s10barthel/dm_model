import argparse
import importlib
import inspect
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

import physical_pass_model as ppm
from scripts import generate_physical_xpass as generate
from scripts.xpass_cli import add_top_pass_selector, resolve_top_pass_selector
from test_physical_xpass import make_graph, make_label


def test_distinct_pairs_and_stable_finite_endpoint_selection():
    values = np.array([[[0.9, 0.89, 0.88], [0.7, 0.7, np.nan]],
                       [[np.nan, np.nan, np.nan], [0.6, 0.5, 0.4]]])
    legacy = ppm._pc_xpass_top_option_indices(values, values, 3)
    distinct = ppm._pc_xpass_ranked_pass_indices(values, values)
    assert legacy.tolist() == [0, 1, 2]
    assert distinct.tolist() == [0, 3, 9]
    ranking = values.copy()
    ranking[0, 0, 0] = np.inf
    ranking[1, 1, 0] = 0.7
    assert ppm._pc_xpass_ranked_pass_indices(values, ranking).tolist() == [1, 3, 9]
    assert ppm._pc_xpass_ranked_pass_indices(values * np.nan, ranking).size == 0


def test_xt_changes_endpoint_and_pair_order_but_mean_uses_xpass():
    values = np.array([[[0.9, 0.4], [0.8, 0.2]]])
    ranking = values * np.array([[[0.1, 1.0], [0.1, 4.0]]])
    indices = ppm._pc_xpass_ranked_pass_indices(values, ranking)
    assert indices.tolist() == [3, 1]
    assert values.ravel()[indices].mean() == pytest.approx(0.3)


@pytest.mark.parametrize("top_xt", [False, True])
def test_compute_multiple_counts_matches_single_and_diagnostics(top_xt):
    graph = make_graph()
    kwargs = dict(top_n=None, max_speed=4, min_speed=3, speed_step=1,
                  angle_step=90, radial_gridsize=20, top_xt=top_xt)
    captured = []
    original = ppm._pc_xpass_ranked_pass_indices

    def capture(score, ranking):
        indices = original(score, ranking)
        captured.append((score.copy(), indices))
        return indices

    with patch.object(ppm, "_pc_xpass_ranked_pass_indices", side_effect=capture):
        result = ppm.compute_graph_pc_xpass_metrics(graph, top_pass_values=[1, 5, 10], **kwargs)
    assert result["home_2__top_pass1_xpass"] == pytest.approx(result["home_2__max_xpass"])
    assert result["home_2__top_pass1_lane_survival"] == pytest.approx(result["home_2__lane_survival"])
    assert result["home_2__top_pass1_control_prob"] == pytest.approx(result["home_2__control_prob"])
    assert result["home_2"] == result["home_2__max_xpass"]
    score, indices = captured[0]
    for n in [1, 5, 10]:
        separate = ppm.compute_graph_pc_xpass_metrics(graph, top_pass_values=[n], **kwargs)
        for suffix in ["xpass", "lane_survival", "control_prob"]:
            column = f"home_2__top_pass{n}_{suffix}"
            assert result[column] == pytest.approx(separate[column])
        expected = np.clip(score.ravel()[indices[:n]].mean(), 1e-4, 1 - 1e-4)
        assert result[f"home_2__top_pass{n}_xpass"] == pytest.approx(expected)


@pytest.mark.parametrize("flags,counts,metrics", [
    ([], None, ["max_xpass"]),
    (["--top-pass", "10", "5", "10"], [5, 10], ["max_xpass", "top_pass5_xpass", "top_pass10_xpass"]),
    (["--top_pass", "5", "--no-max", "--no-topmean"], [5], ["top_pass5_xpass"]),
    (["--top-n", "3", "--top-pass", "5"], [5], ["max_xpass", "top3_xpass", "top_pass5_xpass"]),
    (["--top-n-values", "5", "25"], None, ["max_xpass", "top5_xpass", "top25_xpass"]),
])
def test_generation_opt_in(flags, counts, metrics):
    args = generate.parse_args(["--pc-xpass", *flags])
    assert args.top_pass_values == counts
    assert generate.enabled_physical_xpass_metrics_from_args(args) == metrics


@pytest.mark.parametrize("flags", [
    ["--top-pass", "5"], ["--pc-xpass", "--top-pass"],
    ["--pc-xpass", "--top-pass", "0"], ["--pc-xpass", "--top-pass", "-2"],
    ["--pc-xpass", "--no-max"],
    ["--pc-xpass", "--top-n", "5", "--no-topmean"],
    ["--pc-xpass", "--top-n-values", "5", "--no-topmean"],
])
def test_generation_rejects_invalid_requests(flags):
    with pytest.raises(SystemExit):
        generate.parse_args(flags)


def parse_selector(flags):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpass-version", dest="x_pass_version", default="top10")
    parser.add_argument("--pc-xpass", action="store_true")
    add_top_pass_selector(parser)
    return resolve_top_pass_selector(parser, parser.parse_args(flags))


def test_downstream_alias_and_default():
    assert parse_selector([]).x_pass_version == "top10"
    for alias in ["--top-pass", "--top_pass"]:
        assert parse_selector(["--pc-xpass", alias, "5"]).x_pass_version == "top-pass5"
    assert parse_selector(["--pc-xpass", "--xpass-version", "top-pass5", "--top-pass", "5"]).x_pass_version == "top-pass5"
    assert ppm.physical_xpass_metric_for_version("top-pass5", pc_xpass=True) == "top_pass5_xpass"
    with pytest.raises(ValueError):
        ppm.physical_xpass_metric_for_version("top-pass5", pc_xpass=False)


@pytest.mark.parametrize("flags", [
    ["--top-pass", "5"], ["--pc-xpass", "--top-pass", "0"],
    ["--pc-xpass", "--top-pass", "5", "10"],
    ["--pc-xpass", "--top-pass", "5", "--xpass-version", "top10"],
    ["--xpass-version", "top-pass5"],
])
def test_downstream_rejects_invalid_requests(flags):
    with pytest.raises(SystemExit):
        parse_selector(flags)


@pytest.mark.parametrize("top_n,metrics,default", [
    (3, ["max_xpass", "top3_xpass", "top_pass5_xpass"], "top3"),
    (None, ["max_xpass", "top_pass5_xpass"], "max"),
    (None, ["top3_xpass", "top_pass5_xpass"], "top3"),
    (None, ["top_pass10_xpass", "top_pass5_xpass"], "top-pass5"),
])
def test_metadata_defaults(top_n, metrics, default):
    metadata = ppm.pc_xpass_metadata(ppm.PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER, top_n=top_n, top_n_values=[3],
                                     top_pass_values=[10, 5, 10], available_metrics=metrics)
    assert metadata["default_x_pass_version"] == default
    assert metadata["top_pass_values"] == [5, 10]
    assert metadata["top_pass_definition"] == ppm.PC_XPASS_TOP_PASS_DEFINITION


def test_runtime_cache_roundtrip_and_missing_metric(tmp_path):
    graph = make_graph()
    items = [{"match_id": "example", "graphs": [graph], "labels": torch.stack([make_label()])}]
    kwargs = dict(source=ppm.PC_XPASS_SOURCE, top_n=None, top_pass_values=[5, 10],
                  available_metrics=["max_xpass", "top_pass5_xpass", "top_pass10_xpass"],
                  max_speed=4, min_speed=3, speed_step=1, angle_step=90,
                  radial_gridsize=20, num_workers=1)
    first = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **kwargs)
    assert first["cache_written"] == 1
    second = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **kwargs)
    assert second["cache_hits"] == 1
    rows = ppm.load_physical_xpass_match(tmp_path, "example")
    # Metadata alone must not make incomplete output rows reusable.
    broken = rows.drop(columns=["home_2__top_pass10_control_prob"])
    broken.to_parquet(tmp_path / "matches" / "example.parquet", index=True)
    repaired = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **kwargs)
    assert repaired["cache_written"] == 1
    assert repaired["cache_hits"] == 0
    for n in [5, 10]:
        metric = f"top_pass{n}_xpass"
        selected = ppm.load_runtime_physical_xpass_visualization_component(
            tmp_path, "example", 7, metric=metric, x_pass_version=f"top-pass{n}")
        assert selected["home_2"] == rows.loc[7, f"home_2__{metric}"]
        mode = f"top_pass_{n}"
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert ppm.validate_pc_xpass_lane_survival_mode_cache_metadata(metadata, mode) == mode
        assert ppm.pc_xpass_lane_survival_column_for_mode("home_2", mode) in rows.columns
    with pytest.raises(ValueError, match="--top-pass 25"):
        ppm.validate_x_pass_version_available(tmp_path, x_pass_version="top-pass25", metric="top_pass25_xpass")
    metadata.pop("top_pass_definition")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="--top-pass 5"):
        ppm.validate_x_pass_version_available(tmp_path, x_pass_version="top-pass5", metric="top_pass5_xpass")
    refreshed = ppm.prewarm_physical_xpass_runtime_cache(items, cache_dir=tmp_path, **kwargs)
    assert refreshed["cache_written"] == 1


def test_generation_propagates_counts_to_workers(tmp_path):
    args = generate.parse_args(["--pc-xpass", "--top-pass", "5", "10", "--num-workers", "1"])
    with patch.object(generate, "prewarm_physical_xpass_runtime_cache", return_value={}) as prewarm:
        generate.prewarm_runtime_items([{"match_id": "example"}], cache_dir=tmp_path, args=args, progress_desc="test")
    assert prewarm.call_args.kwargs["top_pass_values"] == [5, 10]
    assert prewarm.call_args.kwargs["top_n"] is None


@pytest.mark.parametrize("module_name", [
    "generate_epv", "run_benchmark", "run_hawkeye", "run_skillcorner",
    "run_relevant_models", "run_and_visualize_hawkeye", "visualize_action_components",
    "visualize_benchmark", "visualize_hawkeye", "visualize_skillcorner",
    "evaluate_relevant_models", "run_hawkeye_loc",
])
def test_downstream_entrypoints(module_name):
    module = importlib.import_module(f"scripts.{module_name}")
    flags = ["--top-pass", "5"]
    if module_name not in {"evaluate_relevant_models", "run_hawkeye_loc"}:
        flags.insert(0, "--pc-xpass")
    if module_name == "visualize_action_components":
        flags.extend(["--match-id", "example", "--action-id", "1"])
    if module_name == "visualize_skillcorner":
        flags.extend(["--match-id", "example", "--index", "1"])
    with patch.object(sys, "argv", [module_name, *flags]):
        args = module.parse_args(flags) if "argv" in inspect.signature(module.parse_args).parameters else module.parse_args()
    version = getattr(args, "x_pass_version", getattr(args, "xpass_version", None))
    assert version == "top-pass5"
    if module_name == "run_hawkeye_loc":
        assert args.top_pass_values == [5]


def test_nonfinite_ranking_leaves_top_pass_missing():
    with patch.object(ppm, "_pc_xpass_xt_values", side_effect=lambda x, y: np.full_like(x, np.nan)):
        result = ppm.compute_graph_pc_xpass_metrics(make_graph(), top_n=None, top_pass_values=[5],
            top_xt=True, max_speed=3, min_speed=3, angle_step=90, radial_gridsize=20)
    assert np.isnan(result["home_2__top_pass5_xpass"])


def test_diagnostics_average_selected_endpoints():
    recorded = []
    original = ppm.pc_xpass_lane_survival_from_raw

    def capture_lane(raw):
        lane = original(raw)
        recorded.append(lane.copy())
        return lane

    selected = []
    rank = ppm._pc_xpass_ranked_pass_indices

    def capture_indices(score, ranking):
        indices = rank(score, ranking)
        selected.append((score.copy(), indices))
        return indices

    with patch.object(ppm, "pc_xpass_lane_survival_from_raw", side_effect=capture_lane), patch.object(
        ppm, "_pc_xpass_ranked_pass_indices", side_effect=capture_indices
    ):
        result = ppm.compute_graph_pc_xpass_metrics(make_graph(), top_n=None, top_pass_values=[5],
            max_speed=4, min_speed=3, angle_step=90, radial_gridsize=20, use_position_discount=False)
    score, indices = selected[0]
    indices = indices[:5]
    lane = recorded[0].ravel()[indices]
    assert result["home_2__top_pass5_lane_survival"] == pytest.approx(lane.mean())
    assert result["home_2__top_pass5_control_prob"] == pytest.approx((score.ravel()[indices] / lane).mean())
