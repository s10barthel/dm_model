from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from datatools import graph_feature
from datatools.match import Match
from scripts import run_relevant_models


def make_minimal_match() -> SimpleNamespace:
    return SimpleNamespace(
        tracking=pd.DataFrame({"ball_accel": [0.0]}),
        fps=25,
        actions=pd.DataFrame({"frame_id": [10, 20]}, index=[1, 3]),
        labels=None,
        label_post_actions=lambda actions: actions,
    )


class GraphFeatureRegressionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
