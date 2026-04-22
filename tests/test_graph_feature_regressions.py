from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from datatools import graph_feature


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


if __name__ == "__main__":
    unittest.main()
