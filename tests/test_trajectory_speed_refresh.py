from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import config, graph_feature
from datatools.artifact_storage import atomic_torch_save
from datatools.match import Match, clear_intended_receiver_model_cache
from datatools.trajectory import (LEGACY_TRAJECTORY_SPEED, TRAJECTORY_SPEED,
                                 PREVIOUS_TRAJECTORY_SPEED, resolve_training_speed, validate_trajectory_graphs)
from datatools import trajectory_refresh as refresh
from models.utils import extract_model_feature_signature
from project_config import assert_feature_run_usable
from scripts import generate_relevant_features as wrapper


def trajectory_match(*, end=35, away=False):
    frames = np.arange(10, 41)
    elapsed = (frames - 10) / 25
    tracking = pd.DataFrame(dict(period_id=1, timestamp=elapsed + 100, ball_x=10 + 10 * elapsed,
                                 ball_y=20 + 2 * elapsed, ball_z=1 + 3 * elapsed, ball_accel=0,
                                 ball_speed=1000, ball_vz=-1000), index=frames)
    actions = pd.DataFrame([dict(frame_id=10, receive_frame_id=end, period_id=1,
                                object_id="away_1" if away else "home_1", receiver_id="home_2",
                                action_type="pass", success=True, endpoint_source="kpi_csv", start_x=10, start_y=20,
                                start_z=1, end_x=20, end_y=22)], index=[17])
    return SimpleNamespace(tracking=tracking, actions=actions, fps=25)


def stored_graph(match, *, definition=None):
    summary = graph_feature.summarize_ball_trajectory(match, 17)
    x = torch.arange(72, dtype=torch.float32).reshape(3, 24)
    x[:, 19] = 123
    x[:, 20:23] = torch.tensor(summary[1:], dtype=torch.float32)
    graph = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 2]]), edge_attr=torch.randn(2, 2))
    graph.node_ids = ["home_1", "home_2", "away_1"]
    if definition:
        graph.trajectory_speed_definition = definition
    return graph


def test_net_speed_actual_timestamps_rotation_and_no_smoothed_dependency():
    match = trajectory_match()
    expected = np.sqrt(113)
    summary = graph_feature.summarize_ball_trajectory(match, 17)
    np.testing.assert_allclose(summary, [expected, 10 / expected, 2 / expected, 3 / expected])
    match.tracking[["ball_speed", "ball_vz"]] = np.nan
    np.testing.assert_allclose(graph_feature.summarize_ball_trajectory(match, 17), summary)
    match.tracking["timestamp"] = (match.tracking.index - 10) / 50
    assert graph_feature.summarize_ball_trajectory(match, 17)[0] == pytest.approx(expected * 2)
    away = graph_feature.summarize_ball_trajectory(trajectory_match(away=True), 17)
    np.testing.assert_allclose(away, summary * [1, -1, -1, 1])


def test_clipping_missing_samples_and_zero_displacement():
    match = trajectory_match(end=23)
    match.tracking.loc[11:15, ["ball_x", "ball_y"]] = np.nan
    assert graph_feature.summarize_ball_trajectory(match, 17)[0] == pytest.approx(np.sqrt(113))
    match.actions.at[17, "receiver_id"] = "out"
    with patch("datatools.utils.resolve_episode_end_frame", return_value=20):
        assert graph_feature.summarize_ball_trajectory(match, 17)[0] == pytest.approx(np.sqrt(113))
    match.tracking[["ball_x", "ball_y", "ball_z"]] = [10, 20, 1]
    np.testing.assert_equal(graph_feature.summarize_ball_trajectory(match, 17), np.zeros(4))


@pytest.mark.parametrize("invalid", ["time", "height", "position", "duration"])
def test_invalid_inputs_rejected(invalid):
    match = trajectory_match()
    if invalid == "time":
        match.tracking["timestamp"] = 1
    elif invalid == "height":
        match.tracking.loc[:, "ball_z"] = np.nan
    elif invalid == "position":
        match.tracking.loc[:, "ball_x"] = np.inf
    else:
        match.actions.at[17, "receive_frame_id"] = 10
    assert graph_feature.summarize_ball_trajectory(match, 17) is None


def test_no_coordinate_fallback_and_frame_rate_time_fallback():
    match = trajectory_match()
    match.tracking = match.tracking.drop(columns="timestamp")
    assert graph_feature.summarize_ball_trajectory(match, 17)[0] == pytest.approx(np.sqrt(113))
    match.tracking.loc[11:35, ["ball_x", "ball_y"]] = np.nan
    value = graph_feature.summarize_ball_trajectory(match, 17)
    assert value is None


def test_refresh_preserves_unrelated_features_and_matches_fresh_generation():
    match = trajectory_match()
    graph = stored_graph(match)
    before = graph.clone()
    labels = torch.tensor([[17.], [19.]])
    result = refresh.refresh_match_graphs([graph, None], labels, match, context="fixture")
    assert result[1] is None
    assert graph.trajectory_speed_definition == TRAJECTORY_SPEED
    torch.testing.assert_close(graph.x[:, :19], before.x[:, :19], rtol=0, atol=0)
    torch.testing.assert_close(graph.x[:, 20:], before.x[:, 20:], rtol=0, atol=0)
    torch.testing.assert_close(graph.edge_index, before.edge_index)
    torch.testing.assert_close(graph.edge_attr, before.edge_attr)
    assert graph.node_ids == before.node_ids
    match.actions.at[17, "object_id"] = "home_1"
    with patch.object(graph_feature, "construct_graph_for_frame", return_value=before.clone()) as construct:
        fresh = graph_feature.construct_graph_for_action(match, 17, feature_variant="success_intent")
    np.testing.assert_allclose(construct.call_args.kwargs["extra_node_features"],
                               graph_feature.summarize_ball_trajectory(match, 17))
    assert fresh.trajectory_speed_definition == TRAJECTORY_SPEED
    torch.testing.assert_close(graph.x[:, 19], graph.x.new_full((3,), construct.call_args.kwargs["extra_node_features"][0]))


def test_refresh_replaces_direction_and_handles_availability_changes():
    match = trajectory_match()
    graph = stored_graph(match, definition=PREVIOUS_TRAJECTORY_SPEED)
    before = graph.clone()
    graph.x[:, 20:23] = -1
    labels = torch.tensor([[17.]])
    refreshed = refresh.refresh_match_graphs([graph], labels, match, context="directions")[0]
    torch.testing.assert_close(refreshed.x[:, 19:23],
                               refreshed.x.new_tensor(graph_feature.summarize_ball_trajectory(match, 17)).expand(3, -1))
    torch.testing.assert_close(refreshed.x[:, :19], before.x[:, :19], rtol=0, atol=0)
    torch.testing.assert_close(refreshed.x[:, 23:], before.x[:, 23:], rtol=0, atol=0)
    with patch.object(refresh, "construct_graph_for_action", return_value=refreshed) as construct:
        assert refresh.refresh_match_graphs([None], labels, match, context="new")[0] is refreshed
        construct.assert_called_once()
    match.tracking[["ball_x", "ball_y"]] = np.nan
    assert refresh.refresh_match_graphs([graph], labels, match, context="unavailable") == [None]


@pytest.mark.parametrize("invalid", ["width", "action", "length", "duplicate"])
def test_refresh_rejects_inconsistent_artifacts(invalid):
    match = trajectory_match()
    graph, labels = stored_graph(match), torch.tensor([[17.]])
    graphs = [graph]
    if invalid == "width":
        graph.x = graph.x[:, :23]
    elif invalid == "action":
        labels[0, 0] = 99
    elif invalid == "length":
        labels = torch.empty((0, 1))
    else:
        graphs, labels = [graph, graph.clone()], torch.tensor([[17.], [17.]])
    with pytest.raises(ValueError):
        refresh.refresh_match_graphs(graphs, labels, match, context="invalid")


@pytest.fixture
def feature_runs(tmp_path):
    runs = tmp_path / "runs"
    base = runs / "base"
    for directory in ("action_graphs_success_intent", "success_intent_labels", "action_graphs"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    metadata = dict(run_id="base", status="completed", next_action_conditions_enabled=False,
                    intended_receiver_model_id="success_intent/old", return_types=["disc_0.9"])
    (base / "metadata.json").write_text(json.dumps(metadata))
    for match_id in ("a", "b"):
        torch.save([stored_graph(trajectory_match()), None], base / "action_graphs_success_intent" / f"{match_id}.pt")
        labels = torch.zeros((2, len(config.LABEL_COLUMNS)))
        labels[:, 0] = torch.tensor([17., 99.])
        labels[0, config.LABEL_INDEX["is_pass"]] = 1
        labels[0, config.LABEL_INDEX["duration"]] = 12
        torch.save(labels, base / "success_intent_labels" / f"{match_id}.pt")
    torch.save(["unchanged"], base / "action_graphs" / "a.pt")
    args = SimpleNamespace(extend_feature_run_id="base", run_id="derived", in_place=False,
                           num_workers="1", worker_thread_limit=1, artifact_copy_mode="hardlink")
    with (patch.object(refresh, "get_feature_run_root", side_effect=lambda run: runs / run),
          patch.object(refresh, "load_feature_run_metadata", side_effect=lambda run, **kw: json.loads((runs / run / "metadata.json").read_text())),
          patch.object(refresh, "write_run_metadata", side_effect=lambda directory, value: (directory.mkdir(parents=True, exist_ok=True), (directory / "metadata.json").write_text(json.dumps(value)))),
          patch.object(refresh, "load_refresh_match", side_effect=lambda *a: trajectory_match()),
          patch.object(refresh, "write_latest_run") as latest):
        yield runs, base, args, latest


def test_derived_hardlinks_preserve_base_and_noop(feature_runs):
    runs, base, args, latest = feature_runs
    old_graph_bytes = (base / "action_graphs_success_intent/a.pt").read_bytes()
    refresh.run_trajectory_speed_refresh(args)
    derived = runs / "derived"
    assert (base / "action_graphs_success_intent/a.pt").read_bytes() == old_graph_bytes
    assert (base / "action_graphs/a.pt").stat().st_ino == (derived / "action_graphs/a.pt").stat().st_ino
    assert (base / "success_intent_labels/a.pt").stat().st_ino != (derived / "success_intent_labels/a.pt").stat().st_ino
    assert torch.load(base / "success_intent_labels/a.pt", weights_only=False)[0, 7] == 12
    assert torch.load(derived / "success_intent_labels/a.pt", weights_only=False)[0, 7] == 1
    assert (base / "action_graphs_success_intent/a.pt").stat().st_ino != (derived / "action_graphs_success_intent/a.pt").stat().st_ino
    metadata = json.loads((derived / "metadata.json").read_text())
    assert metadata["trajectory_speed_definition"] == TRAJECTORY_SPEED
    assert metadata["intended_receiver_model_id"] == "success_intent/old"
    latest.assert_called_once_with("feature", "derived")
    args.extend_feature_run_id = "derived"
    with patch.object(refresh, "stage_refresh_match", side_effect=AssertionError("not a no-op")):
        refresh.run_trajectory_speed_refresh(args)


def test_in_place_failure_rejection_and_recovery(feature_runs):
    runs, base, args, latest = feature_runs
    args.in_place, args.run_id = True, None
    (base / "action_labels_next_3_angle_only").mkdir()
    labels_path = base / "action_labels_next_3_angle_only/a.pt"
    torch.save(torch.load(base / "success_intent_labels/a.pt", weights_only=False), labels_path)
    (base / "resolved_actions_angle_only").mkdir()
    actions_path = base / "resolved_actions_angle_only/a.parquet"
    actions = trajectory_match().actions
    actions["intent_id"] = "home_9"
    actions.to_parquet(actions_path)
    original_replace = Path.replace
    count = 0
    def interrupted_replace(source, target):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected interruption")
        return original_replace(source, target)
    with patch.object(Path, "replace", interrupted_replace), pytest.raises(OSError):
        refresh.run_trajectory_speed_refresh(args)
    with pytest.raises(ValueError, match="not usable"):
        assert_feature_run_usable(base)
    assert not list(base.rglob("*.tmp"))
    refresh.run_trajectory_speed_refresh(args)
    assert_feature_run_usable(base)
    assert json.loads((base / "metadata.json").read_text())["trajectory_speed_definition"] == TRAJECTORY_SPEED
    assert torch.load(labels_path, weights_only=False)[0, 7] == 1
    assert pd.read_parquet(actions_path).at[17, "observed_end_frame_id"] == 35
    assert pd.read_parquet(actions_path).at[17, "intent_id"] == "home_2"
    latest.assert_not_called()


def test_missing_labels_fail_before_mutation(feature_runs):
    runs, base, args, latest = feature_runs
    (base / "success_intent_labels/a.pt").unlink()
    with pytest.raises(FileNotFoundError):
        refresh.run_trajectory_speed_refresh(args)
    assert not (runs / "derived").exists()


def test_staging_failure_changes_no_graphs_and_can_retry(feature_runs):
    runs, base, args, latest = feature_runs
    args.in_place, args.run_id = True, None
    files = sorted((base / "action_graphs_success_intent").glob("*.pt"))
    before = [file.read_bytes() for file in files]
    stage = refresh.stage_refresh_match
    def fail_last(task):
        if task[0].stem == "b":
            raise ValueError("injected staging failure")
        stage(task)
    with patch.object(refresh, "stage_refresh_match", fail_last), pytest.raises(ValueError):
        refresh.run_trajectory_speed_refresh(args)
    assert [file.read_bytes() for file in files] == before
    assert not list(base.rglob("*.tmp"))
    assert_feature_run_usable(base)  # Unchanged input run is still safe to use.
    refresh.run_trajectory_speed_refresh(args)
    assert_feature_run_usable(base)


def test_refresh_all_model_label_variants_and_endpoint_metadata_preserves_receivers(feature_runs):
    runs, base, args, latest = feature_runs
    labels = torch.randn(3, len(config.LABEL_COLUMNS))
    labels[:, 0] = 17
    labels[:, config.LABEL_INDEX["is_pass"]] = 1
    labels[:, config.LABEL_INDEX["duration"]] = 12
    labels[1, config.LABEL_INDEX["is_real"]] = 0
    directories = ["action_labels_next_3_model", "action_labels_next_10_model_carries",
                   "action_labels_intent_train_next_3_model", "augmented_labels_model"]
    for directory in directories:
        (base / directory).mkdir()
        torch.save(labels, base / directory / "a.pt")
    actions = trajectory_match().actions
    actions["intent_id"] = "home_9"
    (base / "resolved_actions_model_carries").mkdir()
    resolved = base / "resolved_actions_model_carries/a.parquet"
    actions.to_parquet(resolved)
    before = resolved.read_bytes()
    refresh.run_trajectory_refresh(args)
    derived = runs / "derived"
    for directory in directories:
        saved = torch.load(derived / directory / "a.pt", weights_only=False)
        torch.testing.assert_close(saved[:, 7], torch.ones(3))
        for name in ("duration", "end_x", "end_y"):
            saved[:, config.LABEL_INDEX[name]] = labels[:, config.LABEL_INDEX[name]]
        torch.testing.assert_close(saved, labels, rtol=0, atol=0)
        torch.testing.assert_close(torch.load(base / directory / "a.pt", weights_only=False), labels)
    updated = pd.read_parquet(derived / resolved.relative_to(base))
    unchanged = [c for c in actions.columns if c not in {"endpoint_source", "end_x", "end_y", "receive_frame_id"}]
    pd.testing.assert_frame_equal(updated[unchanged], actions[unchanged])
    assert updated.at[17, "observed_end_frame_id"] == 35
    assert updated.at[17, "pass_duration"] == 1
    assert resolved.read_bytes() == before
    metadata = json.loads((derived / "metadata.json").read_text())
    assert metadata["angle_only_definition"] == refresh.ANGLE_ONLY
    assert metadata["pass_duration_definition"] == refresh.PASS_DURATION


def test_failed_repair_stays_unusable(feature_runs):
    runs, base, args, latest = feature_runs
    metadata = json.loads((base / "metadata.json").read_text())
    metadata.update(status="failed", extension_attempt=dict(operation="refresh_trajectory_speed", phase="replacing",
                                                           artifact_state="partial"))
    (base / "metadata.json").write_text(json.dumps(metadata))
    args.in_place, args.run_id = True, None
    with patch.object(refresh, "stage_refresh_match", side_effect=ValueError("broken source")), pytest.raises(ValueError):
        refresh.run_trajectory_speed_refresh(args)
    with pytest.raises(ValueError, match="not usable"):
        assert_feature_run_usable(base)


@pytest.mark.parametrize("alias", ["--in-place", "--overwrite-feature-run"])
@pytest.mark.parametrize("flag", ["--refresh-trajectory", "--refresh-trajectory-speed"])
def test_cli_destination_aliases(alias, flag):
    with (patch("sys.argv", ["generate", "--extend-feature-run-id", "base", flag, alias]),
          patch.object(wrapper, "load_feature_run_metadata", return_value={"next_action_conditions_enabled": False})):
        args = wrapper.parse_args()
    assert args.in_place and args.refresh_trajectory_speed
    assert not args.next_action_conditions_enabled


@pytest.mark.parametrize("extra", [[], ["--pass-height"], ["--return_type", "next_5"], ["--extend-interaction-edge-features"],
                                  ["--replace-intended-receiver-model", "--intended-receiver-model-id", "success_intent/new"]])
def test_cli_rejects_missing_source_and_combined_operations(extra):
    argv = ["generate", "--refresh-trajectory-speed"]
    if extra:
        argv.extend(["--extend-feature-run-id", "base", *extra])
    with patch("sys.argv", argv), pytest.raises(SystemExit):
        wrapper.parse_args()


def test_definition_compatibility_and_signatures(tmp_path):
    legacy, current = stored_graph(trajectory_match()), stored_graph(trajectory_match(), definition=TRAJECTORY_SPEED)
    validate_trajectory_graphs([legacy, None], LEGACY_TRAJECTORY_SPEED)
    validate_trajectory_graphs([current], TRAJECTORY_SPEED)
    with pytest.raises(ValueError, match="mismatch"):
        validate_trajectory_graphs([legacy, current], TRAJECTORY_SPEED)
    for name, graph in (("legacy", legacy), ("current", current)):
        directory = tmp_path / name
        directory.mkdir()
        torch.save([graph], directory / "a.pt")
    args = SimpleNamespace(task="success_intent", train_feature_dir=tmp_path / "current", valid_feature_dir=tmp_path / "current")
    resolve_training_speed(args)
    assert args.trajectory_speed_definition == TRAJECTORY_SPEED
    with pytest.raises(ValueError, match="resume/refit"):
        resolve_training_speed(args, checkpoint={})
    resolve_training_speed(args, checkpoint={"trajectory_speed_definition": TRAJECTORY_SPEED})
    args.valid_feature_dir = tmp_path / "legacy"
    with pytest.raises(ValueError, match="training/validation"):
        resolve_training_speed(args)
    signature = extract_model_feature_signature(dict(task="success_intent", trajectory_speed_definition=TRAJECTORY_SPEED))
    assert signature["trajectory_speed_definition"] == TRAJECTORY_SPEED
    assert extract_model_feature_signature(dict(task="success_intent"))["trajectory_speed_definition"] == LEGACY_TRAJECTORY_SPEED
    assert "trajectory_speed_definition" not in extract_model_feature_signature(dict(task="pass_intent"))


def test_legacy_on_demand_rejected_before_graph_generation():
    clear_intended_receiver_model_cache()
    match = Match.__new__(Match)
    match.actions = pd.DataFrame([dict(action_type="pass", success=False, receive_frame_id=20, receiver_id="away_1")])
    model = SimpleNamespace(args={"task": "success_intent"})
    with (patch.object(match, "_get_cached_intended_receiver_model", return_value=(model, {})),
          patch.object(graph_feature, "construct_graph_for_action", side_effect=AssertionError("should reject first")),
          pytest.raises(ValueError, match="On-demand")):
        match._apply_intended_receiver_model(match.actions)


def test_prepared_cache_identity_and_file_replacement(tmp_path):
    import prepared_dataset as prepared
    # Options participate in cache identity; file timestamps participate in entry reuse.
    file = tmp_path / "a.pt"
    torch.save([stored_graph(trajectory_match())], file)
    first = prepared._signature(file)
    atomic_torch_save([stored_graph(trajectory_match(), definition=TRAJECTORY_SPEED)], file)
    assert prepared._signature(file) != first
    assert prepared._digest({"trajectory_speed_definition": LEGACY_TRAJECTORY_SPEED}) != prepared._digest({"trajectory_speed_definition": TRAJECTORY_SPEED})


def test_prepared_success_intent_rebuilds_and_retains_definition(tmp_path):
    import prepared_dataset as prepared
    features, labels = tmp_path / "features", tmp_path / "labels"
    features.mkdir()
    labels.mkdir()
    graph = stored_graph(trajectory_match(), definition=TRAJECTORY_SPEED)
    graph.x[:, :19] = 0
    graph.x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
    graph.x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    row = torch.zeros((1, len(config.LABEL_COLUMNS)))
    for name, value in dict(action_index=17, is_pass=1, intent_index=1, receiver_index=1, duration=1,
                            success=1, is_real=1).items():
        row[0, config.LABEL_INDEX[name]] = value
    torch.save([graph], features / "a.pt")
    torch.save(row, labels / "a.pt")
    kwargs = dict(feature_dir=features, label_dir=labels, task="success_intent", trajectory_speed_definition=TRAJECTORY_SPEED,
                  extend_features=False, cache_dir=tmp_path / "cache", progress=False)
    with patch.object(prepared.PreparedActionDataset, "_check_space"):
        first = prepared.PreparedActionDataset(["a"], **kwargs)
        second = prepared.PreparedActionDataset(["a"], **kwargs)
        assert second.preparation["reused"] == 1
        graph.x[:, 19] = 45
        atomic_torch_save([graph], features / "a.pt")
        third = prepared.PreparedActionDataset(["a"], **kwargs)
    assert third.preparation["built"] == 1
    assert len(first) == len(third) == 1
    sample = next(iter(third))[0]
    assert sample.trajectory_speed_definition == TRAJECTORY_SPEED
    torch.testing.assert_close(sample.x[:, 19], torch.full((3,), 45.))


def test_stored_inference_rejects_same_width_definition_mismatch():
    from inference import inference_gnn
    model = SimpleNamespace(args={"task": "success_intent"})
    with pytest.raises(ValueError, match="definition mismatch"):
        inference_gnn(SimpleNamespace(), model, device="cpu",
                      graph_override=[stored_graph(trajectory_match(), definition=TRAJECTORY_SPEED)],
                      label_override=torch.zeros((1, len(config.LABEL_COLUMNS))))
