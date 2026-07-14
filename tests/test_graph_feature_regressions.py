from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from datatools.benchmark import build_benchmark_export
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from datatools import graph_feature
from datatools.match import Match, clear_intended_receiver_model_cache
from datatools.utils import filter_features_and_labels
from inference import inference_gnn, resolve_match_graphs
from models import utils as model_utils
from scripts import run_relevant_models


def make_minimal_match() -> SimpleNamespace:
    return SimpleNamespace(
        tracking=pd.DataFrame({"ball_accel": [0.0]}),
        fps=25,
        actions=pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3]),
        labels=None,
        label_post_actions=lambda actions: actions,
    )


def _player_columns(object_id: str, x: float, y: float = 34.0) -> dict[str, float]:
    return {
        f"{object_id}_x": x,
        f"{object_id}_y": y,
        f"{object_id}_vx": 0.0,
        f"{object_id}_vy": 0.0,
        f"{object_id}_speed": 0.0,
        f"{object_id}_accel": 0.0,
    }


def make_offside_match_and_snapshot(
    *,
    ball_x: float | None = 60.0,
    include_goals: bool = True,
    home_positions: dict[str, float] | None = None,
    away_positions: dict[str, float] | None = None,
) -> tuple[SimpleNamespace, pd.DataFrame]:
    home_positions = home_positions or {"home_1": 75.0, "home_2": 65.0, "home_3": 50.0}
    away_positions = away_positions or {"away_1": 15.0, "away_2": 70.0, "away_3": 80.0}

    row: dict[str, float | int] = {"phase_id": 1}
    if ball_x is not None:
        row["ball_x"] = ball_x
    row["ball_y"] = 34.0
    row["ball_z"] = 0.0

    for object_id, x in home_positions.items():
        row.update(_player_columns(object_id, x))
    if include_goals:
        row.update(_player_columns("home_goal", 105.0))

    for object_id, x in away_positions.items():
        row.update(_player_columns(object_id, x))
    if include_goals:
        row.update(_player_columns("away_goal", 0.0))

    snapshot = pd.DataFrame([row], index=pd.Index([10], name="frame_id"))
    match = SimpleNamespace(
        include_keepers=True,
        include_goals=include_goals,
        phases=pd.DataFrame(
            [
                {
                    "phase_id": 1,
                    "active_players": list(home_positions) + list(away_positions),
                    "active_keepers": ["home_3", "away_3"],
                }
            ]
        ).set_index("phase_id"),
    )
    return match, snapshot


class DummyNodeModel(torch.nn.Module):
    def __init__(self, task: str, logits: list[float] | list[list[float]], node_in_dim: int = 25) -> None:
        super().__init__()
        self.args = {
            "task": task,
            "node_in_dim": node_in_dim,
            "include_out": False,
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": False,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
            "filter_blockers": False,
        }
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self.seen_edge_attr: torch.Tensor | None = None

    def forward(self, graphs: Data, _batch_dests: torch.Tensor | None = None) -> torch.Tensor:
        self.seen_edge_attr = graphs.edge_attr.detach().cpu().clone()
        return self._logits[: graphs.x.shape[0]].to(graphs.x.device)


class EdgeFeatureConstructionTests(unittest.TestCase):
    def test_construct_graph_for_frame_appends_relative_speed_after_alignment(self) -> None:
        match, snapshot = make_offside_match_and_snapshot(
            include_goals=False,
            home_positions={"home_1": 10.0, "home_2": 20.0},
            away_positions={"away_1": 40.0, "away_2": 50.0},
        )
        snapshot.loc[:, "home_1_vx"] = 1.0
        snapshot.loc[:, "home_1_vy"] = 2.0
        snapshot.loc[:, "home_2_vx"] = 4.0
        snapshot.loc[:, "home_2_vy"] = 6.0
        snapshot.loc[:, "home_1_speed"] = float(np.hypot(1.0, 2.0))
        snapshot.loc[:, "home_2_speed"] = float(np.hypot(4.0, 6.0))
        match.tracking = snapshot
        match.max_players = 4

        graph = graph_feature.construct_graph_for_frame(
            match,
            frame=10,
            possessor="home_1",
            period_tracking=snapshot,
            feature_dim=graph_feature.infer_node_feature_dim(extend=False),
            extend=False,
            add_v_edge_features=True,
            add_relative_speed_edge_features=True,
        )

        assert graph is not None
        edge_mask = (graph.edge_index[0] == 0) & (graph.edge_index[1] == 1)
        self.assertEqual(graph.edge_attr.shape[1], 5)
        self.assertTrue(torch.allclose(graph.edge_attr[edge_mask, 4], torch.tensor([5.0])))

    def test_construct_graph_for_frame_rejects_relative_speed_without_alignment(self) -> None:
        match, snapshot = make_offside_match_and_snapshot(include_goals=False)
        match.tracking = snapshot
        match.max_players = 6

        with self.assertRaisesRegex(ValueError, "Relative-speed edge features require"):
            graph_feature.construct_graph_for_frame(
                match,
                frame=10,
                possessor="home_1",
                period_tracking=snapshot,
                feature_dim=graph_feature.infer_node_feature_dim(extend=False),
                extend=False,
                add_v_edge_features=False,
                add_relative_speed_edge_features=True,
            )


def make_inference_graph() -> Data:
    x = torch.zeros((4, 25), dtype=torch.float32)
    x[:3, 0] = 1.0
    x[0, 13] = 1.0
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.ones((edge_index.shape[1], 2), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_offside_inference_graph() -> Data:
    base = make_inference_graph()
    x = torch.zeros((base.x.shape[0], 26), dtype=torch.float32)
    x[:, : base.x.shape[1]] = base.x
    x[1, -1] = 1.0
    return Data(x=x, edge_index=base.edge_index, edge_attr=base.edge_attr)


def make_marked_inference_graph(marker: float) -> Data:
    graph = make_inference_graph()
    graph.x[0, 3] = marker
    return graph


def make_model_mode_graph() -> Data:
    graph = make_inference_graph()
    graph.edge_attr = torch.ones((graph.edge_index.shape[1], 4), dtype=torch.float32)
    graph.node_ids = ["home_10", "home_11", "home_12", "away_20"]
    return graph


def make_inference_labels(*, success: bool = False) -> torch.Tensor:
    labels = torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)
    labels[0, 0] = 0.0
    labels[0, 1] = 1.0
    labels[0, 4] = 4.0
    labels[0, 5] = 1.0
    labels[0, 6] = 1.0
    labels[0, 14] = 1.0
    labels[0, 16] = 1.0 if success else 0.0
    return labels


def make_inference_match(labels: torch.Tensor) -> SimpleNamespace:
    tracking = pd.DataFrame(
        [
            {
                "home_10_x": 10.0,
                "home_10_y": 34.0,
                "home_11_x": 20.0,
                "home_11_y": 30.0,
                "home_12_x": 30.0,
                "home_12_y": 38.0,
                "away_20_x": 60.0,
                "away_20_y": 34.0,
            }
        ],
        index=pd.Index([0], name="frame_id"),
    )
    actions = pd.DataFrame(
        [
            {
                "frame_id": 0,
                "object_id": "home_10",
                "end_frame_id": 0,
                "end_player_id": "home_10",
            }
        ],
        index=pd.Index([0], name="index"),
    )
    return SimpleNamespace(
        actions=actions,
        tracking=tracking,
        labels=labels,
        graph_features_0=[make_inference_graph()],
        graph_features_1=None,
        graph_features_by_dir={},
    )


def make_offside_inference_match(labels: torch.Tensor) -> SimpleNamespace:
    match = make_inference_match(labels)
    match.graph_features_0 = [make_offside_inference_graph()]
    return match


def make_pass_filter_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spadl_type": "pass",
                "frame_id": 10,
                "receive_frame_id": 20,
                "receiver_id": "home_02",
                "next_player_id": "away_03",
                "next_type": "interception",
                "object_id": "home_01",
                "offside": False,
            },
            {
                "spadl_type": "cross",
                "frame_id": 30,
                "receive_frame_id": 40,
                "receiver_id": "out",
                "next_player_id": "away_04",
                "next_type": "throw_in",
                "object_id": "home_01",
                "offside": False,
            },
            {
                "spadl_type": "pass",
                "frame_id": 50,
                "receive_frame_id": 60,
                "receiver_id": "away_05",
                "next_player_id": "away_06",
                "next_type": "foul",
                "object_id": "home_01",
                "offside": False,
            },
        ],
        index=[10, 20, 30],
    )


def make_pass_filter_match(*, next_action_conditions_enabled: bool) -> Match:
    match = Match.__new__(Match)
    match.events = make_pass_filter_events()
    match.next_action_conditions_enabled = next_action_conditions_enabled
    return match


class MatchPassFilterTests(unittest.TestCase):
    def test_next_action_conditions_on_keeps_current_exclusions_and_allowances(self) -> None:
        match = make_pass_filter_match(next_action_conditions_enabled=True)

        passes = match.filter_passes()

        self.assertEqual(passes.index.tolist(), [20, 30])
        self.assertNotIn(10, passes.index)

    def test_next_action_conditions_off_keeps_valid_frame_passes(self) -> None:
        match = make_pass_filter_match(next_action_conditions_enabled=False)

        passes = match.filter_passes()

        self.assertEqual(passes.index.tolist(), [10, 20, 30])


class GraphFeatureMatchConstructionTests(unittest.TestCase):
    def test_build_match_for_feature_generation_passes_next_action_setting(self) -> None:
        args = SimpleNamespace(action_type="all", next_action_conditions_enabled=False)

        with patch.object(graph_feature, "Match") as match_class:
            graph_feature.build_match_for_feature_generation(
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                args,
            )

        match_class.assert_called_once_with(
            ANY,
            ANY,
            ANY,
            "all",
            include_goals=True,
            next_action_conditions_enabled=False,
        )


class GraphFeatureRegressionTests(unittest.TestCase):
    def test_resolve_num_workers_auto_caps_at_six(self) -> None:
        with patch("datatools.graph_feature.os.cpu_count", return_value=16):
            self.assertEqual(graph_feature.resolve_num_workers("auto"), 6)

    def test_resolve_num_workers_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            graph_feature.resolve_num_workers("0")

    def test_parse_accepts_worker_flags(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "graph_feature.py",
                "--num-workers",
                "auto",
                "--worker-thread-limit",
                "2",
            ],
        ):
            args = graph_feature.parse_args()

        self.assertEqual(args.num_workers, "auto")
        self.assertEqual(args.worker_thread_limit, 2)

    def test_node_feature_dimensions_include_offside_tail(self) -> None:
        self.assertEqual(graph_feature.infer_node_feature_dim(extend=False), 20)
        self.assertEqual(graph_feature.infer_node_feature_dim(extend=True), 26)
        self.assertEqual(graph_feature.infer_node_feature_dim(extend=False, feature_variant="success_intent"), 24)

    def test_offside_flags_both_teams_and_excludes_goal_nodes(self) -> None:
        match, snapshot = make_offside_match_and_snapshot()

        features = graph_feature.calculate_event_features(match, snapshot, "home_1", extend=False)[0]

        self.assertEqual(features.shape[1], 20)
        self.assertEqual(features[:, -1].tolist(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def test_offside_uses_strict_level_comparisons(self) -> None:
        match, snapshot = make_offside_match_and_snapshot(
            home_positions={"home_1": 70.0, "home_2": 60.0, "home_3": 50.0},
            away_positions={"away_1": 60.0, "away_2": 70.0, "away_3": 80.0},
        )

        features = graph_feature.calculate_event_features(match, snapshot, "home_1", extend=False)[0]

        self.assertEqual(features[:, -1].tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_offside_rotates_away_possession_to_left_to_right(self) -> None:
        match, snapshot = make_offside_match_and_snapshot(
            ball_x=45.0,
            home_positions={"home_1": 95.0, "home_2": 85.0, "home_3": 65.0},
            away_positions={"away_1": 30.0, "away_2": 100.0, "away_3": 95.0},
        )

        features = graph_feature.calculate_event_features(match, snapshot, "away_1", extend=False, rotate_to_ltr=True)[0]

        self.assertEqual(features[:, -1].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_offside_returns_zero_without_ball_x_or_two_opponents(self) -> None:
        match, snapshot = make_offside_match_and_snapshot(ball_x=None)

        missing_ball_features = graph_feature.calculate_event_features(match, snapshot, "home_1", extend=False)[0]

        sparse_match, sparse_snapshot = make_offside_match_and_snapshot(
            include_goals=False,
            home_positions={"home_1": 75.0},
            away_positions={"away_1": 80.0},
        )
        sparse_features = graph_feature.calculate_event_features(sparse_match, sparse_snapshot, "home_1", extend=False)[0]

        self.assertEqual(missing_ball_features[:, -1].tolist(), [0.0] * 8)
        self.assertEqual(sparse_features[:, -1].tolist(), [0.0, 0.0])

    def test_old_model_prefix_truncates_new_offside_tail(self) -> None:
        x = torch.arange(26, dtype=torch.float32).reshape(1, 26)
        graph = Data(
            x=x,
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.ones((1, 2), dtype=torch.float32),
        )
        batch = Batch.from_data_list([graph])

        adapted = model_utils.adapt_batch_graphs_for_model(
            batch,
            {"node_in_dim": 25, "edge_in_dim": 2, "v_edge_feature_mode": "none"},
            context="old model",
        )

        self.assertEqual(adapted.x.shape[1], 25)
        torch.testing.assert_close(adapted.x[0], torch.arange(25, dtype=torch.float32))

    def test_construct_graph_features_requires_labels_or_action_indices(self) -> None:
        match = make_minimal_match()

        with self.assertRaises(ValueError) as exc:
            graph_feature.construct_graph_features(match, verbose=False)

        self.assertIn("construct_graph_features requires either action_indices or match.labels", str(exc.exception))

    def test_bind_canonical_graph_context_restores_multi_mode_base_feature_flow(self) -> None:
        match = make_minimal_match()
        canonical_actions = pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3])
        labels = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        resolved_actions_by_mode = {
            "original": canonical_actions,
            "angle_only": canonical_actions.copy(),
        }
        labels_by_key = {
            ("original", "next_5"): labels,
            ("angle_only", "next_5"): labels.clone(),
        }

        base_action_indices = graph_feature.bind_canonical_graph_context(
            match,
            resolved_actions_by_mode,
            labels_by_key,
            "original",
            "next_5",
        )

        self.assertEqual(base_action_indices.tolist(), [1, 3])
        self.assertTrue(match.actions.equals(canonical_actions))
        self.assertTrue(torch.equal(match.labels, labels))
        self.assertIsNot(match.labels, labels)

        with patch.object(
            graph_feature,
            "construct_graph_for_action",
            side_effect=lambda *_args, **kwargs: {"action_index": kwargs.get("action_index", _args[1])},
        ) as construct_graph_for_action:
            graphs = graph_feature.construct_graph_features(
                match,
                action_indices=base_action_indices,
                feature_variant="base",
                add_v_edge_features=True,
                verbose=False,
            )

        self.assertEqual(len(graphs), 2)
        self.assertEqual([graph["action_index"] for graph in graphs], [1, 3])
        self.assertEqual(
            [call.args[1] for call in construct_graph_for_action.call_args_list],
            [1, 3],
        )

    def test_bind_canonical_graph_context_rejects_mismatched_label_order(self) -> None:
        match = make_minimal_match()
        resolved_actions_by_mode = {
            "original": pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3]),
            "angle_only": pd.DataFrame({"frame_id": [10, 40]}, index=[1, 4]),
        }
        labels_by_key = {
            ("original", "next_5"): torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32),
            ("angle_only", "next_5"): torch.tensor([[1.0, 0.0], [4.0, 0.0]], dtype=torch.float32),
        }

        with self.assertRaises(ValueError) as exc:
            graph_feature.bind_canonical_graph_context(
                match,
                resolved_actions_by_mode,
                labels_by_key,
                "original",
                "next_5",
            )

        self.assertIn("Shared base graph artifacts require identical action ordering", str(exc.exception))

    def test_intent_train_labels_preserve_existing_augmented_action_sequence(self) -> None:
        base_labels = torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 30.0],
            ],
            dtype=torch.float32,
        )
        source_intent_labels = torch.tensor(
            [
                [3.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        )

        aligned = graph_feature.build_intent_train_labels_from_source_actions(base_labels, source_intent_labels)

        self.assertEqual(aligned[:, 0].tolist(), [3.0, 1.0, 3.0])
        self.assertEqual(aligned[:, 1].tolist(), [30.0, 10.0, 30.0])

    def test_labels_only_artifact_save_does_not_construct_graph_features(self) -> None:
        match = make_minimal_match()
        resolved_actions = pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3])
        labels = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            graph_dir = graph_feature.get_action_graph_dir(feature_root)
            graph_dir.mkdir(parents=True)
            (graph_dir / "match.pt").write_text("graph marker", encoding="utf-8")
            resolved_dir = feature_root / "resolved_actions_original"
            resolved_dir.mkdir(parents=True)
            (resolved_dir / "match.parquet").write_text(
                "not parquet, only used as an existence marker",
                encoding="utf-8",
            )

            with patch.object(graph_feature, "construct_graph_features") as construct_graph_features:
                graph_feature.save_labels_only_artifacts(
                    match,
                    {"original": resolved_actions},
                    {("original", "next_5"): labels},
                    ["original"],
                    ["next_5"],
                    "match",
                    feature_root,
                )

        construct_graph_features.assert_not_called()

    def test_labels_only_artifact_save_requires_copied_base_graphs(self) -> None:
        match = make_minimal_match()
        resolved_actions = pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3])
        labels = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            resolved_action_path = graph_feature.get_resolved_action_path(
                "match",
                intended_receiver_mode="original",
                root=feature_root,
            )
            label_path = graph_feature.get_action_label_dir(
                "next_5",
                intended_receiver_mode="original",
                root=feature_root,
            ) / "match.pt"

            with self.assertRaises(FileNotFoundError) as exc:
                graph_feature.save_labels_only_artifacts(
                    match,
                    {"original": resolved_actions},
                    {("original", "next_5"): labels},
                    ["original"],
                    ["next_5"],
                    "match",
                    feature_root,
                )

            self.assertFalse(resolved_action_path.exists())
            self.assertFalse(label_path.exists())

        self.assertIn("Base action graphs not found", str(exc.exception))

    def test_intent_train_labels_only_requires_copied_graphs_and_source_labels(self) -> None:
        match = make_minimal_match()
        resolved_actions = pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3])
        labels = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            resolved_action_path = graph_feature.get_resolved_action_path(
                "match",
                intended_receiver_mode="original",
                root=feature_root,
            )
            intent_label_path = graph_feature.get_intent_train_label_dir(
                "next_5",
                intended_receiver_mode="original",
                root=feature_root,
            ) / "match.pt"

            with self.assertRaises(FileNotFoundError) as missing_graph_exc:
                graph_feature.save_labels_only_artifacts(
                    match,
                    {"original": resolved_actions},
                    {("original", "next_5"): labels},
                    ["original"],
                    ["next_5"],
                    "match",
                    feature_root,
                    feature_variant="intent_train_augmented",
                    intent_train_label_source_mode="original",
                    intent_train_label_source_return_type="disc_0.9",
                )

            graph_dir = graph_feature.get_action_graph_intent_train_dir(feature_root)
            graph_dir.mkdir(parents=True)
            (graph_dir / "match.pt").write_text("intent graph marker", encoding="utf-8")

            with self.assertRaises(FileNotFoundError) as missing_source_exc:
                graph_feature.save_labels_only_artifacts(
                    match,
                    {"original": resolved_actions},
                    {("original", "next_5"): labels},
                    ["original"],
                    ["next_5"],
                    "match",
                    feature_root,
                    feature_variant="intent_train_augmented",
                    intent_train_label_source_mode="original",
                    intent_train_label_source_return_type="disc_0.9",
                )

            self.assertFalse(resolved_action_path.exists())
            self.assertFalse(intent_label_path.exists())

        self.assertIn("Base intent-train graphs not found", str(missing_graph_exc.exception))
        self.assertIn("Source intent-train labels not found", str(missing_source_exc.exception))

    def test_construct_labels_skips_shot_receiver_missing_from_frame_snapshot(self) -> None:
        match = object.__new__(Match)
        match.action_type = "all"
        match.fps = 25
        match.include_goals = True
        match.intended_receiver_stats = {}
        match.tracking = pd.DataFrame(
            [
                {
                    "home_1_x": 10.0,
                    "home_1_y": 34.0,
                    "home_goal_x": 105.0,
                    "home_goal_y": 34.0,
                    "away_1_x": 80.0,
                    "away_1_y": 34.0,
                    "away_goal_x": 0.0,
                    "away_goal_y": 34.0,
                },
                {
                    "home_1_x": 20.0,
                    "home_1_y": 34.0,
                    "home_goal_x": 105.0,
                    "home_goal_y": 34.0,
                    "away_1_x": 70.0,
                    "away_1_y": 34.0,
                    "away_goal_x": 0.0,
                    "away_goal_y": 34.0,
                },
            ],
            index=pd.Index([10, 20], name="frame_id"),
        )
        match.events = pd.DataFrame(
            {
                "spadl_type": ["shot", "shot"],
                "object_id": ["home_1", "home_1"],
                "period_id": [1, 1],
                "expected_goal": [0.0, 0.0],
                "xT": [0.0, 0.0],
                "goal_distance": [0.0, 0.0],
            },
            index=[1, 2],
        )
        actions = pd.DataFrame(
            [
                {
                    "frame_id": 10,
                    "object_id": "home_1",
                    "action_type": "shot",
                    "receiver_id": "home_22",
                    "receive_frame_id": pd.NA,
                    "intent_id": "home_goal",
                    "blocked": False,
                    "success": False,
                    "start_x": 10.0,
                    "start_y": 34.0,
                    "end_x": 105.0,
                    "end_y": 34.0,
                },
                {
                    "frame_id": 20,
                    "object_id": "home_1",
                    "action_type": "shot",
                    "receiver_id": "home_goal",
                    "receive_frame_id": pd.NA,
                    "intent_id": "home_goal",
                    "blocked": False,
                    "success": False,
                    "start_x": 20.0,
                    "start_y": 34.0,
                    "end_x": 105.0,
                    "end_y": 34.0,
                },
            ],
            index=[1, 2],
        )
        match.actions = actions.copy()

        def add_return_columns(events: pd.DataFrame, *_args: object, **_kwargs: object) -> pd.DataFrame:
            events = events.copy()
            for column in [
                "scores",
                "concedes",
                "scores_xg",
                "concedes_xg",
                "scores_xT",
                "concedes_xT",
                "scores_goal_distance",
                "concedes_goal_distance",
            ]:
                events[column] = 0.0
            return events

        with (
            patch("datatools.match.utils.label_returns", side_effect=add_return_columns),
            patch("datatools.match.utils.label_xt_returns", side_effect=add_return_columns),
            patch("datatools.match.utils.label_goal_distance_returns", side_effect=add_return_columns),
        ):
            labels = match.construct_labels(
                relabel_intended_receivers=False,
                resolved_actions=actions,
                return_type="next_1",
            )

        self.assertEqual(labels[:, 0].tolist(), [2.0])
        self.assertEqual(labels.shape[1], len(LABEL_COLUMNS))
        self.assertEqual(float(labels[0, LABEL_INDEX["scores_goal_next10"]]), 0.0)
        self.assertEqual(float(labels[0, LABEL_INDEX["concedes_goal_next10"]]), 0.0)

    def test_intended_receiver_model_cache_reuses_loaded_checkpoint_within_process(self) -> None:
        clear_intended_receiver_model_cache()
        match = object.__new__(Match)
        dummy_model = torch.nn.Linear(1, 1)

        with (
            patch("models.utils.load_model", return_value=dummy_model) as load_model,
            patch(
                "models.utils.get_model_graph_schema",
                return_value={"edge_in_dim": 4, "add_v_edge_features": True},
            ) as get_schema,
        ):
            cached_first = match._get_cached_intended_receiver_model("success_intent/demo")
            cached_second = match._get_cached_intended_receiver_model("success_intent/demo")

        self.assertIs(cached_first[0], dummy_model)
        self.assertIs(cached_second[0], dummy_model)
        self.assertEqual(load_model.call_count, 1)
        self.assertEqual(get_schema.call_count, 1)
        clear_intended_receiver_model_cache()

    def test_return_annotation_cache_reuses_per_return_type_annotations(self) -> None:
        match = object.__new__(Match)
        match._base_events = pd.DataFrame({"value": [1.0]})
        match._cached_events_by_return_type = {}
        match._cached_diagnostic_events = None

        def add_columns(events: pd.DataFrame, *_args: object, **_kwargs: object) -> pd.DataFrame:
            events = events.copy()
            for column in [
                "scores",
                "concedes",
                "scores_xg",
                "concedes_xg",
                "scores_xT",
                "concedes_xT",
                "scores_goal_distance",
                "concedes_goal_distance",
                "scores_epv",
                "concedes_epv",
            ]:
                events[column] = 0.0
            return events

        with (
            patch("datatools.match.utils.label_returns", side_effect=add_columns) as label_returns,
            patch("datatools.match.utils.label_xt_returns", side_effect=add_columns) as label_xt_returns,
            patch("datatools.match.utils.label_goal_distance_returns", side_effect=add_columns) as label_goal_distance_returns,
            patch("datatools.match.utils.label_epv_returns", side_effect=add_columns) as label_epv_returns,
        ):
            first = match._get_events_for_return_type("next_1", "next", 1, False)
            second = match._get_events_for_return_type("next_1", "next", 1, False)

        self.assertIs(first, second)
        self.assertEqual(label_returns.call_count, 1)
        self.assertEqual(label_xt_returns.call_count, 1)
        self.assertEqual(label_goal_distance_returns.call_count, 1)
        self.assertEqual(label_epv_returns.call_count, 1)


class PassOnlyIntentInferenceRegressionTests(unittest.TestCase):
    def _run_inference(self, task: str, *, success: bool = False) -> pd.DataFrame:
        model = DummyNodeModel(task, logits=[2.0, 1.0, 0.0, -1.0])
        match = make_inference_match(make_inference_labels(success=success))
        probs, _ = inference_gnn(match, model, device="cpu", post_action=False)
        return probs

    def test_pass_only_intent_tasks_exclude_possessor_and_renormalize(self) -> None:
        expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)

        for task in ["pass_intent", "pass_intent_oppo_agn"]:
            with self.subTest(task=task):
                probs = self._run_inference(task)

                self.assertEqual(probs.columns.tolist(), ["home_11", "home_12"])
                torch.testing.assert_close(torch.tensor(probs.loc[0].tolist()), expected, atol=1e-6, rtol=0.0)
                self.assertAlmostEqual(float(probs.loc[0].sum()), 1.0, places=6)

    def test_success_intent_excludes_possessor_and_renormalizes(self) -> None:
        probs = self._run_inference("success_intent", success=True)
        expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)

        self.assertEqual(probs.columns.tolist(), ["home_11", "home_12"])
        torch.testing.assert_close(torch.tensor(probs.loc[0].tolist()), expected, atol=1e-6, rtol=0.0)
        self.assertAlmostEqual(float(probs.loc[0].sum()), 1.0, places=6)

    def test_action_intent_keeps_possessor_option(self) -> None:
        probs = self._run_inference("action_intent")
        expected = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)

        self.assertEqual(probs.columns.tolist(), ["home_10", "home_11", "home_12"])
        torch.testing.assert_close(torch.tensor(probs.loc[0].tolist()), expected, atol=1e-6, rtol=0.0)

    def test_inference_gnn_trims_extra_edge_features_for_model(self) -> None:
        model = DummyNodeModel("action_intent", logits=[2.0, 1.0, 0.0, -1.0])
        match = make_inference_match(make_inference_labels())
        match.graph_features_0 = [make_model_mode_graph()]

        inference_gnn(match, model, device="cpu", post_action=False)

        self.assertIsNotNone(model.seen_edge_attr)
        self.assertEqual(model.seen_edge_attr.shape[1], 2)

    def test_inference_gnn_masks_possessor_velocity_edges_for_no_poss_model(self) -> None:
        model = DummyNodeModel("action_intent", logits=[2.0, 1.0, 0.0, -1.0])
        model.args["edge_in_dim"] = 4
        model.args["v_edge_feature_mode"] = "no_poss"
        match = make_inference_match(make_inference_labels())
        match.graph_features_0 = [make_model_mode_graph()]

        inference_gnn(match, model, device="cpu", post_action=False)

        self.assertIsNotNone(model.seen_edge_attr)
        graph = match.graph_features_0[0]
        incident_edges = (graph.edge_index[0] == 0) | (graph.edge_index[1] == 0)
        self.assertTrue(torch.equal(model.seen_edge_attr[incident_edges, 2:4], torch.zeros((6, 2))))
        self.assertTrue(torch.equal(model.seen_edge_attr[~incident_edges, 2:4], torch.ones((6, 2))))

    def test_pass_intent_keeps_offside_options_and_renormalizes(self) -> None:
        model = DummyNodeModel("pass_intent", logits=[0.0, 5.0, 0.0, 0.0], node_in_dim=26)
        match = make_offside_inference_match(make_inference_labels())

        probs, _ = inference_gnn(match, model, device="cpu", post_action=False)
        expected = torch.softmax(torch.tensor([5.0, 0.0]), dim=0)

        self.assertEqual(probs.columns.tolist(), ["home_11", "home_12"])
        torch.testing.assert_close(torch.tensor(probs.loc[0].tolist()), expected, atol=1e-6, rtol=0.0)

    def test_action_intent_keeps_offside_options(self) -> None:
        model = DummyNodeModel("action_intent", logits=[0.0, 5.0, 0.0, 0.0], node_in_dim=26)
        match = make_offside_inference_match(make_inference_labels())

        probs, _ = inference_gnn(match, model, device="cpu", post_action=False)
        expected = torch.softmax(torch.tensor([0.0, 5.0, 0.0]), dim=0)

        self.assertEqual(probs.columns.tolist(), ["home_10", "home_11", "home_12"])
        torch.testing.assert_close(torch.tensor(probs.loc[0].tolist()), expected, atol=1e-6, rtol=0.0)

    def test_pass_success_forces_offside_options_to_zero(self) -> None:
        model = DummyNodeModel("pass_success", logits=[0.0, 5.0, 0.0, 0.0], node_in_dim=26)
        match = make_offside_inference_match(make_inference_labels())

        probs, _ = inference_gnn(match, model, device="cpu", post_action=False)

        self.assertEqual(float(probs.loc[0, "home_11"]), 0.0)
        self.assertAlmostEqual(float(probs.loc[0, "home_12"]), 0.5, places=6)

    def test_outcome_success_branch_keeps_model_output_for_offside_options(self) -> None:
        model = DummyNodeModel(
            "outcome_scoring",
            logits=[[0.0, 0.0], [-2.0, 5.0], [0.0, 0.0], [0.0, 0.0]],
            node_in_dim=26,
        )
        match = make_offside_inference_match(make_inference_labels())

        failure_probs, success_probs = inference_gnn(match, model, device="cpu", post_action=False)

        self.assertAlmostEqual(float(failure_probs.loc[0, "home_11"]), float(torch.sigmoid(torch.tensor(-2.0))), places=6)
        self.assertAlmostEqual(float(success_probs.loc[0, "home_11"]), float(torch.sigmoid(torch.tensor(5.0))), places=6)
        self.assertAlmostEqual(float(success_probs.loc[0, "home_12"]), 0.5, places=6)

    def test_benchmark_export_leaves_passer_pass_intent_empty_and_renormalized(self) -> None:
        pass_intent = self._run_inference("pass_intent")
        export_rows = pd.DataFrame(
            [
                {
                    "modification": 1,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "frame_id": 0,
                    "team": 2,
                    "player": 10,
                    "pos_x": 10.0,
                    "pos_y": 34.0,
                    "pos_z": 0.0,
                    "smooth_x_speed": 0.0,
                    "smooth_y_speed": 0.0,
                    "event_player": 10,
                    "object_id": "home_10",
                },
                {
                    "modification": 1,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "frame_id": 0,
                    "team": 2,
                    "player": 11,
                    "pos_x": 20.0,
                    "pos_y": 30.0,
                    "pos_z": 0.0,
                    "smooth_x_speed": 0.0,
                    "smooth_y_speed": 0.0,
                    "event_player": 10,
                    "object_id": "home_11",
                },
                {
                    "modification": 1,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "frame_id": 0,
                    "team": 2,
                    "player": 12,
                    "pos_x": 30.0,
                    "pos_y": 38.0,
                    "pos_z": 0.0,
                    "smooth_x_speed": 0.0,
                    "smooth_y_speed": 0.0,
                    "event_player": 10,
                    "object_id": "home_12",
                },
            ]
        )
        state = SimpleNamespace(frame_meta=pd.DataFrame(index=pd.Index([0], name="frame_id")))

        exported = build_benchmark_export(export_rows, state, {"pass_intent": pass_intent})

        passer_row = exported.loc[exported["player"].eq(exported["event_player"])].iloc[0]
        receiver_rows = exported.loc[exported["player"].ne(exported["event_player"])]

        self.assertTrue(pd.isna(passer_row["pass_intent"]))
        self.assertAlmostEqual(float(receiver_rows["pass_intent"].sum()), 1.0, places=6)

    def test_success_intent_row_excludes_possessor_for_visualization_consumers(self) -> None:
        row = self._run_inference("success_intent", success=True).loc[0]

        self.assertEqual(row.index.tolist(), ["home_11", "home_12"])
        self.assertNotIn("home_10", row.index)

    def test_filter_features_aligns_by_feature_action_index(self) -> None:
        labels = torch.zeros((3, len(LABEL_COLUMNS)), dtype=torch.float32)
        labels[:, 0] = torch.tensor([1.0, 2.0, 3.0])
        labels[:, 1] = 1.0
        labels[:, 4] = 4.0
        labels[:, 5] = 1.0
        labels[:, 6] = 1.0
        labels[:, LABEL_INDEX["success"]] = 1.0
        args = DummyNodeModel("success_intent", logits=[0.0, 0.0, 0.0, 0.0]).args

        filtered_graphs, filtered_labels = filter_features_and_labels(
            [make_marked_inference_graph(20.0), make_marked_inference_graph(30.0)],
            labels,
            args,
            event_indices=[3],
            feature_action_indices=np.array([2, 3]),
        )

        self.assertEqual(filtered_labels[:, 0].tolist(), [3.0])
        self.assertEqual(float(filtered_graphs[0].x[0, 3]), 30.0)

    def test_filter_features_rejects_mismatched_rows_without_feature_action_index(self) -> None:
        labels = torch.zeros((3, len(LABEL_COLUMNS)), dtype=torch.float32)
        args = DummyNodeModel("success_intent", logits=[0.0, 0.0, 0.0, 0.0]).args

        with self.assertRaisesRegex(ValueError, "not row-aligned"):
            filter_features_and_labels(
                [make_marked_inference_graph(20.0), make_marked_inference_graph(30.0)],
                labels,
                args,
            )

    def test_resolve_match_graphs_uses_runtime_feature_root_and_distinct_cache_keys(self) -> None:
        model = DummyNodeModel("success_intent", logits=[0.0, 0.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            root_a = Path(tmpdir) / "feature_a"
            root_b = Path(tmpdir) / "feature_b"
            for root, marker, action_index in [(root_a, 10.0, 10.0), (root_b, 20.0, 20.0)]:
                graph_dir = root / "action_graphs_success_intent"
                label_dir = root / "success_intent_labels"
                graph_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                torch.save([make_marked_inference_graph(marker)], graph_dir / "match.pt")
                labels = torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)
                labels[0, 0] = action_index
                torch.save(labels, label_dir / "match.pt")

            model.args["feature_dir"] = str(root_a / "action_graphs_success_intent")
            match = SimpleNamespace(
                match_id="match",
                runtime_feature_root=root_b,
                graph_features_0=None,
                graph_features_1=None,
                graph_features_by_dir={},
                graph_feature_action_indices_by_dir={},
            )

            graphs_b, action_indices_b = resolve_match_graphs(match, model, post_action=False)
            match.runtime_feature_root = root_a
            graphs_a, action_indices_a = resolve_match_graphs(match, model, post_action=False)

        self.assertEqual(float(graphs_b[0].x[0, 3]), 20.0)
        self.assertEqual(action_indices_b.tolist(), [20])
        self.assertEqual(float(graphs_a[0].x[0, 3]), 10.0)
        self.assertEqual(action_indices_a.tolist(), [10])
        self.assertEqual(len(match.graph_features_by_dir), 2)

    def test_run_success_intent_inference_uses_saved_labels_and_restores_match_state(self) -> None:
        match = SimpleNamespace(
            actions=pd.DataFrame([{"action_id": 123}], index=[10]),
            labels=torch.tensor([[10.0]], dtype=torch.float32),
            intended_receiver_stats={"source": "original"},
            runtime_feature_root=Path("original_feature"),
        )
        original_actions = match.actions.copy()
        original_labels = match.labels.clone()
        original_runtime_feature_root = match.runtime_feature_root
        success_labels = torch.tensor([[10.0]], dtype=torch.float32)
        feature_root = Path("feature_root")

        with (
            patch.object(run_relevant_models, "load_success_intent_labels", return_value=success_labels) as load_labels_mock,
            patch.object(run_relevant_models, "inference_gnn", return_value=(pd.DataFrame({"home_11": [0.9]}, index=[10]), None)),
        ):
            result = run_relevant_models.run_success_intent_inference(
                match,
                model=object(),
                return_type="disc_0.9",
                device="cpu",
                feature_root=feature_root,
            )

        self.assertEqual(result.loc[10, "home_11"], 0.9)
        load_labels_mock.assert_called_once_with(match, feature_root)
        pd.testing.assert_frame_equal(match.actions, original_actions)
        self.assertTrue(torch.equal(match.labels, original_labels))
        self.assertEqual(match.intended_receiver_stats, {"source": "original"})
        self.assertEqual(match.runtime_feature_root, original_runtime_feature_root)

    def test_stored_model_mode_failed_pass_intent_matches_direct_recompute(self) -> None:
        match = Match.__new__(Match)
        match.actions = pd.DataFrame(
            {
                "action_type": ["pass"],
                "success": [False],
                "receive_frame_id": [12],
                "receiver_id": ["away_20"],
                "intent_id": ["home_10"],
            },
            index=[101],
        )
        match.actions.attrs["intended_receiver_stats"] = {}
        model = DummyNodeModel("success_intent", logits=[0.0, 5.0, 1.0, -1.0])
        model.args["edge_in_dim"] = 4
        model.args["add_v_edge_features"] = True

        angle_only_actions = match.actions.copy()
        stored_model_actions = angle_only_actions.copy()
        stored_model_actions["intent_id"] = ["home_11"]

        with (
            patch("models.utils.load_model", return_value=model),
            patch("datatools.graph_feature.construct_graph_for_action", return_value=make_model_mode_graph()),
            patch("datatools.match.utils.filter_features_and_labels", side_effect=lambda graphs, labels, *_args, **_kwargs: (graphs, labels)),
        ):
            direct_model_actions = match._apply_intended_receiver_model(
                angle_only_actions.copy(),
                "success_intent/test",
            )

        pd.testing.assert_series_equal(
            direct_model_actions.loc[stored_model_actions.index, "intent_id"],
            stored_model_actions["intent_id"],
            check_names=False,
        )


class ComponentExportRegressionTests(unittest.TestCase):
    def test_component_export_uses_true_event_action_id(self) -> None:
        frame = pd.DataFrame(
            {
                "home_1": [0.25, 0.75],
                "away_4": [0.75, 0.25],
            },
            index=pd.Index([4, 9], name="index"),
        )
        actions = pd.DataFrame(
            [
                {
                    "stats_perform_match_id": "DFL-MAT-TEST",
                    "action_id": 105,
                    "original_event_id": "EVENT-105",
                },
                {
                    "stats_perform_match_id": "DFL-MAT-TEST",
                    "action_id": 211,
                    "original_event_id": "EVENT-211",
                },
            ],
            index=pd.Index([4, 9], name="index"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "component.parquet"
            run_relevant_models.save_component_table(frame, actions, output_path)
            exported = pd.read_parquet(output_path)

        self.assertEqual(
            exported.columns.tolist(),
            ["stats_perform_match_id", "action_id", "original_event_id", "home_1", "away_4"],
        )
        self.assertEqual(exported["action_id"].tolist(), [105, 211])
        self.assertEqual(exported["original_event_id"].tolist(), ["EVENT-105", "EVENT-211"])
        self.assertNotIn("index", exported.columns)

    def test_match_component_export_keeps_other_outputs_when_pass_success_is_skipped(self) -> None:
        actions = pd.DataFrame(
            [
                {
                    "stats_perform_match_id": "DFL-MAT-TEST",
                    "action_id": 105,
                    "original_event_id": "EVENT-105",
                }
            ],
            index=pd.Index([4], name="index"),
        )
        frame = pd.DataFrame({"home_1": [0.25]}, index=pd.Index([4], name="index"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "DFL-MAT-TEST"
            run_relevant_models.save_match_component_tables(
                output_dir,
                actions,
                action_intent=frame,
                pass_intent=frame,
                pass_success=None,
                scoring_success=frame,
                scoring_failure=frame,
                conceding_success=frame,
                conceding_failure=frame,
            )

            self.assertTrue((output_dir / "action_intent.parquet").exists())
            self.assertTrue((output_dir / "pass_intent.parquet").exists())
            self.assertTrue((output_dir / "outcome_scoring_success.parquet").exists())
            self.assertTrue((output_dir / "outcome_conceding_failure.parquet").exists())
            self.assertFalse((output_dir / "pass_success.parquet").exists())


if __name__ == "__main__":
    unittest.main()
