"""Numerical parity and publication guarantees for feature worker optimizations."""
import copy
import os
import shutil
import time
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
import torch

from datatools import config
from datatools.artifact_storage import clone_feature_artifacts
from datatools.label_reuse import reuse_label_rows
from datatools.match import Match
from datatools.worker_runtime import map_tasks, map_shared_tasks, resolve_num_workers
from datatools.xt_refresh import fingerprint, refresh_xt
from datatools.goal_distance_refresh import refresh_goal_distance
from scripts.feature_extension_diagnostics import ExtensionAttempt
from tests.test_xt_refresh import sample


def worker_probe(value, settings):
    time.sleep(.25 if value == 0 else .05)
    return value, os.getpid(), torch.get_num_threads(), settings["value"]


def shared_probe(task):
    value, settings = task
    return value, settings["value"]


def test_bounded_workers_order_native_limits_and_shared_settings():
    previous = os.environ.get("OPENBLAS_NUM_THREADS")
    with closing(map_tasks(worker_probe, range(8), 2, 1, shared_args=({"value": 9},))) as results:
        values = list(results)
    assert [value[0] for value in values] == list(range(8))
    assert all(pid != os.getpid() and threads == 1 and value == 9 for _, pid, threads, value in values)
    assert len({value[1] for value in values}) == 2
    assert os.environ.get("OPENBLAS_NUM_THREADS") == previous
    settings = {"value": 12}
    with closing(map_shared_tasks(shared_probe, [(i, settings) for i in range(4)], 2,
                                   shared_indices=(1,))) as results:
        assert list(results) == [(i, 12) for i in range(4)]
    assert list(map_tasks(worker_probe, [], "auto", shared_args=({},))) == []


def test_auto_policy_and_rejection(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert resolve_num_workers("auto") == 6
    for value in (0, -1, "0"):
        with pytest.raises(ValueError):
            resolve_num_workers(value)


@pytest.fixture
def parallel_sample(sample):
    root, events, sidecars, metadata, *_ = sample
    for folder in (root, events, sidecars):
        for path in list(folder.rglob("m.*")):
            shutil.copy2(path, path.with_name("n" + path.suffix))
    metadata["split_manifest"] = dict(universe_count=2)
    return sample


def compare_artifacts(left, right):
    files = {p.relative_to(left) for p in left.rglob("*") if p.suffix in {".pt", ".parquet"}}
    assert files == {p.relative_to(right) for p in right.rglob("*") if p.suffix in {".pt", ".parquet"}}
    for relative in files:
        a, b = left / relative, right / relative
        if "graphs" in relative.parent.name:
            assert fingerprint(a) == fingerprint(b)
        elif relative.suffix == ".pt":
            torch.testing.assert_close(torch.load(a, weights_only=True), torch.load(b, weights_only=True),
                                       rtol=0, atol=0, equal_nan=True)
        else:
            pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
            assert pd.read_parquet(a).attrs == pd.read_parquet(b).attrs


@pytest.mark.parametrize("family", ["xt", "goal_distance"])
def test_legacy_parallel_refresh_exact_and_source_safe(parallel_sample, tmp_path, family):
    root, events, sidecars, metadata, *_ = parallel_sample
    if family == "goal_distance":
        annotations = pd.read_csv(sidecars / "m.csv")
        annotations["goal_distance"] = [.1, .4, .8]
        for mid in ("m", "n"):
            annotations.to_csv(sidecars / f"{mid}.csv", index=False)
    original = {p: fingerprint(p) for p in root.rglob("*") if p.is_file()}
    outputs, reports = [], []
    for workers in (1, 2):
        output = tmp_path / f"workers_{workers}"
        clone_feature_artifacts(root, output)
        kwargs = dict(num_workers=workers, worker_thread_limit=1, event_dir=events,
                      attempt=ExtensionAttempt(output, family, "derived", {}))
        if family == "xt":
            report = refresh_xt(output, copy.deepcopy(metadata), xt_match_dir=sidecars, **kwargs)
        else:
            report = refresh_goal_distance(output, copy.deepcopy(metadata), goal_distance_match_dir=sidecars,
                                           target_families=["goal_distance", "progression"], **kwargs)
        outputs.append(output)
        reports.append(report)
    compare_artifacts(*outputs)
    for key in ("changed_rows", "rewritten_files", "artifacts", "return_types"):
        assert reports[0][key] == reports[1][key]
    assert all(fingerprint(path) == digest for path, digest in original.items())
    assert not list(outputs[1].rglob("*.tmp"))


@pytest.mark.parametrize("family", ["xt", "goal_distance"])
def test_parallel_legacy_failure_leaves_destinations_unchanged(parallel_sample, family):
    root, events, sidecars, metadata, *_ = parallel_sample
    bad = pd.read_csv(sidecars / "n.csv")
    bad.loc[0, "xT"] = np.nan
    bad.to_csv(sidecars / "n.csv", index=False)
    if family == "goal_distance":
        for mid in ("m", "n"):
            side = pd.read_csv(sidecars / f"{mid}.csv")
            side["goal_distance"] = [.1, .4, .8]
            if mid == "n":
                side.loc[0, "goal_distance"] = np.nan
            side.to_csv(sidecars / f"{mid}.csv", index=False)
    original = {p: fingerprint(p) for p in root.rglob("*") if p.is_file()}
    kwargs = dict(num_workers=2, event_dir=events, attempt=ExtensionAttempt(root, family, "derived", {}))
    with pytest.raises(ValueError):
        if family == "xt":
            refresh_xt(root, metadata, xt_match_dir=sidecars, **kwargs)
        else:
            refresh_goal_distance(root, metadata, goal_distance_match_dir=sidecars, **kwargs)
    assert all(fingerprint(path) == digest for path, digest in original.items())
    assert not list(root.rglob("*.tmp"))


def test_label_rows_reused_exactly_and_invalidated(sample):
    _, _, _, _, events, actions, _ = sample
    def make_match():
        match = Match.__new__(Match)
        match.events = events.copy()
        match.action_type, match.fps, match.include_goals = "all", 25, False
        match.tracking = pd.DataFrame({"home_1_x": [1.], "home_1_y": [2.], "away_1_x": [3.], "away_1_y": [4.]})
        return match
    returns = ["next_3", "disc_0.9", "extended_possession_1", "in_1"]
    expected_match, reused_match = make_match(), make_match()
    expected = [expected_match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                return_type=ret) for ret in returns]
    with reuse_label_rows(reused_match) as cache:
        for ret, old in zip(returns, expected):
            result = reused_match.construct_labels(resolved_actions=actions, relabel_intended_receivers=False,
                                                   return_type=ret)
            torch.testing.assert_close(result, old, rtol=0, atol=0, equal_nan=True)
        changed = actions.copy()
        changed.loc[0, "start_x"] = 70.
        result = reused_match.construct_labels(resolved_actions=changed, relabel_intended_receivers=False,
                                               return_type=returns[0])
        assert result[0, config.LABEL_INDEX["start_x"]] == 70.
        assert cache["actions"].loc[0, "start_x"] == 70.
        from datatools.label_reuse import reused_labels
        reused_match.actions["scores"] = reused_match.actions.scores.astype(str)
        assert reused_labels(reused_match) is None
    assert not hasattr(reused_match, "_label_row_cache")


def test_success_intent_primary_only_preserves_state_and_primary_labels(sample):
    from datatools.graph_feature import build_success_intent_labels_by_return
    _, _, _, _, events, actions, _ = sample
    def make_match():
        match = Match.__new__(Match)
        match.events, match.actions = events.copy(), actions.copy()
        match.action_type, match.fps, match.include_goals = "all", 25, False
        match.tracking = pd.DataFrame({"home_1_x": [1.], "home_1_y": [2.], "away_1_x": [3.], "away_1_y": [4.]})
        return match
    returns = ["extended_possession_3", "next_3", "disc_0.9"]
    full, primary = make_match(), make_match()
    old_actions, old = build_success_intent_labels_by_return(full, returns)
    new_actions, new = build_success_intent_labels_by_return(primary, returns, primary_only=True)
    assert list(new) == [returns[0]]
    torch.testing.assert_close(old[returns[0]], new[returns[0]], rtol=0, atol=0, equal_nan=True)
    pd.testing.assert_frame_equal(old_actions, new_actions, check_exact=True)
    pd.testing.assert_frame_equal(full.actions, primary.actions, check_exact=True)
    pd.testing.assert_frame_equal(full.events, primary.events, check_exact=True)
    with pytest.raises(ValueError):
        build_success_intent_labels_by_return(make_match(), ["next_3", "invalid_return"], primary_only=True)


def test_parallel_edge_extensions_match_serial_and_cleanup(tmp_path):
    from torch_geometric.data import Data
    from models.edge_schema import CANONICAL_FEATURES
    from scripts.generate_relevant_features import append_relative_speed_to_graph, append_interactions_to_graph
    from scripts.feature_extension_diagnostics import extend_graph_files
    source = tmp_path / "source"
    source.mkdir()
    x = torch.arange(3 * 26, dtype=torch.float32).reshape(3, 26) / 10
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = torch.tensor([1., 1., 0.])
    edges = torch.cartesian_prod(torch.arange(3), torch.arange(3)).T
    graph = Data(x=x, edge_index=edges, edge_attr=torch.ones((9, 4)),
                 edge_feature_names=list(CANONICAL_FEATURES[:4]), edge_schema_version=1,
                 node_ids=["a", "b", "c"], node_self_edge_attr=torch.ones((3, 4)),
                 custom_provenance="preserve")
    for mid in ("a", "b", "c"):
        torch.save([graph.clone(), None], source / f"{mid}.pt")
    before = {path: fingerprint(path) for path in source.glob("*.pt")}
    outputs = []
    for workers in (1, 2):
        root = tmp_path / f"graphs_{workers}"
        clone_feature_artifacts(source, root)
        files = sorted(root.glob("*.pt"))
        attempt = ExtensionAttempt(root, "edges", "derived", {})
        for transform in (append_relative_speed_to_graph, append_interactions_to_graph):
            extend_graph_files(files, transform, ".edges.tmp", attempt, num_workers=workers)
        outputs.append(root)
    for left in outputs[0].glob("*.pt"):
        a = torch.load(left, weights_only=False)
        b = torch.load(outputs[1] / left.name, weights_only=False)
        assert a[1] is b[1] is None
        for key, value in a[0].to_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, b[0][key], rtol=0, atol=0, equal_nan=True)
            else:
                assert value == b[0][key]
    assert all(fingerprint(path) == digest for path, digest in before.items())
    root = tmp_path / "bad_graphs"
    clone_feature_artifacts(source, root)
    torch.save("not a graph list", root / "b.pt")
    before = {path: fingerprint(path) for path in root.glob("*.pt")}
    attempt = ExtensionAttempt(root, "edges", "derived", {})
    with pytest.raises(TypeError, match="not a list"):
        extend_graph_files(sorted(root.glob("*.pt")), append_relative_speed_to_graph, ".edges.tmp", attempt,
                           num_workers=2)
    assert attempt.data["current_file"] == str(root / "b.pt")
    assert all(fingerprint(path) == digest for path, digest in before.items())
    assert not list(root.glob("*.tmp"))


def test_auto_refresh_validation_precedes_output_mutation(monkeypatch):
    from types import SimpleNamespace
    from datatools import reception_refresh
    from scripts import replace_carries
    def probe(*args):
        raise RuntimeError("source probe")
    for module, function in ((reception_refresh, reception_refresh.run_retirement),
                             (replace_carries, replace_carries.replace_carries)):
        monkeypatch.setattr(module, "resolve_source", probe)
        with pytest.raises(RuntimeError, match="source probe"):
            function(SimpleNamespace(num_workers="auto", extend_feature_run_id="source"))
        with pytest.raises(ValueError, match="positive"):
            function(SimpleNamespace(num_workers="0", extend_feature_run_id="source"))


def test_carry_extension_forwards_edge_worker_settings(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from scripts import generate_relevant_features as generator
    from tests.test_feature_run_extension import make_args, make_metadata
    metadata = make_metadata()
    metadata[generator.PASS_DURATION_KEY] = generator.PASS_DURATION
    root = tmp_path / "base"
    root.mkdir()
    args = make_args(num_workers="4", worker_thread_limit=2, in_place=True, run_id=None)
    args.extend_interaction_edge_features = True
    monkeypatch.setattr(generator, "resolve_feature_run_id", lambda *a, **kw: "base")
    monkeypatch.setattr(generator, "load_feature_run_metadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(generator, "get_feature_run_root", lambda *a: root)
    monkeypatch.setattr(generator, "carry_extension_steps", lambda *a: [])
    monkeypatch.setattr(generator, "run_generation_steps", lambda *a, **kw: None)
    extend = Mock()
    monkeypatch.setattr(generator, "extend_interaction_edge_features", extend)
    generator.run_carry_extension_generation(args)
    assert extend.call_args.kwargs["num_workers"] == "4"
    assert extend.call_args.kwargs["worker_thread_limit"] == 2


def test_family_mapping_reuse_keeps_each_family_value_validation(sample, monkeypatch):
    from datatools import xt_refresh
    from datatools.stop_conditions import action_targets
    from datatools.xt_targets import target_settings
    _, _, _, _, events, actions, _ = sample
    computed = events.copy()
    for name in ("scores", "concedes", "scores_xg", "concedes_xg", "scores_xT", "concedes_xT",
                 "scores_goal_distance", "concedes_goal_distance", "scores_epv", "concedes_epv"):
        computed[name] = [.1, .2, .3]
    original, calls = xt_refresh._prepare_return_mapping, []
    def record(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(xt_refresh, "_prepare_return_mapping", record)
    options = dict(computed=computed, target_families=["goal", "xg", "xt", "goal_distance", "epv"])
    result = action_targets(actions, events, "next_3", "base", target_settings({}), None, None, **options)
    assert len(calls) == 1
    for name in ("scores", "scores_xg", "scores_xt", "scores_goal_distance", "scores_epv"):
        np.testing.assert_array_equal(result[name].to_numpy(), [.1, .2, .3])
    computed.loc[0, "scores_epv"] = np.nan
    with pytest.raises(ValueError, match="Non-finite computed returns"):
        action_targets(actions, events, "next_3", "base", target_settings({}), None, None, **options)


def test_single_return_avoids_label_cache_allocation(sample, monkeypatch):
    from datatools import label_reuse
    from datatools.graph_feature import build_labels_by_mode_and_return
    root, _, _, _, events, actions, _ = sample
    match = Match.__new__(Match)
    match.events, match.actions = events.copy(), actions.copy()
    match.action_type, match.fps, match.include_goals = "all", 25, False
    match.intended_receiver_stats = {}
    match.tracking = pd.DataFrame({"home_1_x": [1.], "home_1_y": [2.], "away_1_x": [3.], "away_1_y": [4.]})
    original, allocated = label_reuse.remember_label_rows, []
    def record(*args):
        allocated.append(getattr(args[0], "_label_row_cache", None) is not None)
        return original(*args)
    monkeypatch.setattr(label_reuse, "remember_label_rows", record)
    build_labels_by_mode_and_return(match, ["original"], ["next_3"], None, root, "m",
                                   prefer_existing_resolved_actions=True)
    assert allocated == [False]
    assert not hasattr(match, "_label_row_cache")
