from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import project_config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from datatools import utils
from models.utils import calc_binary_metrics, get_outcome_diagnostic_targets, get_outcome_targets
from scripts import main as main_script
from scripts import train_relevant_models as train_wrapper


def make_xt_events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spadl_type": spadl_type,
                "object_id": object_id,
                "xT": x_t,
                "period_id": 1,
                "success": False,
                "expected_goal": 0.0,
            }
            for spadl_type, object_id, x_t in rows
        ]
    )


def make_goal_distance_events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spadl_type": spadl_type,
                "object_id": object_id,
                "goal_distance": goal_distance,
                "period_id": 1,
                "success": False,
                "expected_goal": 0.0,
            }
            for spadl_type, object_id, goal_distance in rows
        ]
    )


def make_epv_events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spadl_type": spadl_type,
                "object_id": object_id,
                "epv": epv,
                "period_id": 1,
                "success": False,
                "expected_goal": 0.0,
            }
            for spadl_type, object_id, epv in rows
        ]
    )


class ReturnTypeValidationTests(unittest.TestCase):
    def test_validate_return_type_accepts_in_variant(self) -> None:
        self.assertEqual(project_config.validate_return_type("in_3"), "in_3")
        self.assertEqual(project_config.parse_return_type("in_3"), ("in", 3, False))

    def test_validate_return_type_accepts_skip1_variants(self) -> None:
        self.assertEqual(project_config.validate_return_type("next_3_skip1"), "next_3_skip1")
        self.assertEqual(project_config.validate_return_type("disc_0.5_skip1"), "disc_0.5_skip1")
        self.assertEqual(project_config.parse_return_type("next_3_skip1"), ("next", 3, True))
        self.assertEqual(project_config.parse_return_type("disc_0.5_skip1"), ("disc", 0.5, True))

    def test_validate_return_type_rejects_invalid_in_variant(self) -> None:
        with self.assertRaises(ValueError):
            project_config.validate_return_type("in_0")
        with self.assertRaises(ValueError):
            project_config.validate_return_type("in_three")
        with self.assertRaises(ValueError):
            project_config.validate_return_type("in_3_skip1")
        with self.assertRaises(ValueError):
            project_config.validate_return_type("next_three_skip1")

    def test_validate_return_type_for_target_family_rejects_goal_and_xg(self) -> None:
        for target_family in ["goal", "xg"]:
            with self.assertRaises(ValueError) as exc:
                project_config.validate_return_type_for_target_family("in_3", target_family=target_family)
            self.assertIn("only supported", str(exc.exception))

    def test_validate_return_type_for_target_family_accepts_xt_and_goal_distance(self) -> None:
        self.assertEqual(
            project_config.validate_return_type_for_target_family("in_3", target_family="xt"),
            "in_3",
        )
        self.assertEqual(
            project_config.validate_return_type_for_target_family("in_3", target_family="goal_distance"),
            "in_3",
        )

    def test_resolve_effective_return_type_rejects_invalid_in_state_target_family(self) -> None:
        with self.assertRaises(ValueError):
            project_config.resolve_effective_return_type("goal", "in_3")
        self.assertEqual(project_config.resolve_effective_return_type("xt", "in_3"), "in_3")

    def test_infer_feature_run_return_types_detects_in_state_and_skip1_label_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            (run_root / "action_labels_in_3").mkdir()
            (run_root / "action_labels_next_5_skip1_angle_only").mkdir()
            (run_root / "action_labels_disc_0.5_skip1_model").mkdir()

            with (
                patch.object(project_config, "load_feature_run_metadata", return_value={}),
                patch.object(project_config, "get_feature_run_root", return_value=run_root),
            ):
                return_types = project_config.infer_feature_run_return_types("feature_run")

        self.assertEqual(return_types, ["disc_0.5_skip1", "in_3", "next_5_skip1"])


class InStateLabelingTests(unittest.TestCase):
    def test_label_nth_future_state_value_matches_corrected_in_3_example(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.15),
                ("pass", "away_1", 0.20),
                ("pass", "away_1", 0.10),
                ("pass", "away_1", 0.15),
                ("pass", "home_1", 0.05),
            ]
        )

        labeled = utils.label_xt_in_state_returns(events, action_offset=3)

        self.assertEqual(labeled["scores_xT"].round(2).tolist(), [0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        self.assertEqual(labeled["concedes_xT"].round(2).tolist(), [0.00, 0.20, 0.10, 0.15, 0.05, 0.00, 0.00, 0.00])

    def test_label_nth_future_state_value_replaces_nth_action_with_earlier_shot(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("shot", "away_1", 0.70),
                ("pass", "away_1", 0.40),
                ("pass", "home_1", 0.30),
            ]
        )

        labeled = utils.label_xt_in_state_returns(events, action_offset=3)

        self.assertEqual(float(labeled.at[0, "scores_xT"]), 0.0)
        self.assertEqual(float(labeled.at[0, "concedes_xT"]), 0.70)

    def test_label_nth_future_state_value_uses_earlier_shot_when_nth_action_is_missing(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("shot", "away_1", 0.70),
                ("pass", "away_1", 0.40),
            ]
        )

        labeled = utils.label_xt_in_state_returns(events, action_offset=4)

        self.assertEqual(float(labeled.at[0, "scores_xT"]), 0.0)
        self.assertEqual(float(labeled.at[0, "concedes_xT"]), 0.70)

    def test_label_nth_future_state_value_returns_zero_without_nth_action_or_shot(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "away_1", 0.20),
                ("pass", "away_1", 0.30),
            ]
        )

        labeled = utils.label_xt_in_state_returns(events, action_offset=4)

        self.assertEqual(float(labeled.at[0, "scores_xT"]), 0.0)
        self.assertEqual(float(labeled.at[0, "concedes_xT"]), 0.0)

    def test_existing_next_and_discounted_xt_helpers_still_match_expected_behavior(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "away_1", 0.60),
                ("shot", "home_1", 0.40),
            ]
        )

        next_labeled = utils.label_xt_returns(events, lookahead_len=3)
        disc_labeled = utils.label_discounted_xt_returns(events, gamma=0.5)

        self.assertEqual(float(next_labeled.at[0, "scores_xT"]), 0.40)
        self.assertEqual(float(next_labeled.at[0, "concedes_xT"]), 0.60)
        self.assertAlmostEqual(float(disc_labeled.at[0, "scores_xT"]), 0.30)
        self.assertAlmostEqual(float(disc_labeled.at[0, "concedes_xT"]), 0.30)

    def test_discounted_goal_distance_returns_sum_future_values(self) -> None:
        events = make_goal_distance_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "away_1", 0.60),
                ("shot", "home_1", 0.40),
            ]
        )

        labeled = utils.label_discounted_goal_distance_returns(events, gamma=0.5)

        self.assertAlmostEqual(float(labeled.at[0, "scores_goal_distance"]), 0.30)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_goal_distance"]), 0.30)

    def test_discounted_epv_returns_sum_future_values(self) -> None:
        events = make_epv_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "away_1", 0.60),
                ("shot", "home_1", 0.40),
            ]
        )

        labeled = utils.label_discounted_epv_returns(events, gamma=0.5)

        self.assertAlmostEqual(float(labeled.at[0, "scores_epv"]), 0.30)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_epv"]), 0.30)

    def test_label_future_max_value_skip1_matches_next_3_example(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.15),
                ("pass", "away_1", 0.20),
                ("pass", "away_1", 0.10),
                ("pass", "away_1", 0.15),
                ("pass", "home_1", 0.05),
            ]
        )

        labeled = utils.label_xt_returns(events, lookahead_len=3, skip_first=True)

        self.assertEqual(labeled["scores_xT"].round(2).tolist(), [0.15, 0.15, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00])
        self.assertEqual(labeled["concedes_xT"].round(2).tolist(), [0.00, 0.20, 0.20, 0.15, 0.05, 0.05, 0.00, 0.00])

    def test_label_discounted_future_sum_value_skip1_skips_first_non_shot_without_consuming_rank(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "home_1", 0.70),
                ("pass", "home_1", 0.20),
                ("pass", "away_1", 0.10),
            ]
        )

        labeled = utils.label_discounted_xt_returns(events, gamma=0.5, skip_first=True)

        self.assertAlmostEqual(float(labeled.at[0, "scores_xT"]), 0.80)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_xT"]), 0.025)

    def test_skip1_next_and_discounted_xt_helpers_do_not_skip_first_rated_shot(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("shot", "away_1", 0.70),
                ("pass", "home_1", 0.40),
            ]
        )

        next_labeled = utils.label_xt_returns(events, lookahead_len=3)
        next_skip_labeled = utils.label_xt_returns(events, lookahead_len=3, skip_first=True)
        disc_labeled = utils.label_discounted_xt_returns(events, gamma=0.5)
        disc_skip_labeled = utils.label_discounted_xt_returns(events, gamma=0.5, skip_first=True)

        self.assertEqual(float(next_skip_labeled.at[0, "scores_xT"]), float(next_labeled.at[0, "scores_xT"]))
        self.assertEqual(float(next_skip_labeled.at[0, "concedes_xT"]), float(next_labeled.at[0, "concedes_xT"]))
        self.assertEqual(float(disc_skip_labeled.at[0, "scores_xT"]), float(disc_labeled.at[0, "scores_xT"]))
        self.assertEqual(float(disc_skip_labeled.at[0, "concedes_xT"]), float(disc_labeled.at[0, "concedes_xT"]))


class BinaryMetricsTests(unittest.TestCase):
    def test_binary_labels_are_supported(self) -> None:
        metrics = calc_binary_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0.05, 0.7, 0.2, 0.9]),
            threshold=0.5,
        )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_discounted_goal_labels_are_binarized_for_metrics(self) -> None:
        metrics = calc_binary_metrics(
            np.array([0.0, 0.729, 0.9, 1.0]),
            np.array([0.05, 0.8, 0.7, 0.95]),
            threshold=0.1,
        )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_soft_continuous_labels_are_binarized_for_metrics(self) -> None:
        metrics = calc_binary_metrics(
            np.array([0.0, 0.02, 0.4, 0.8]),
            np.array([0.2, 0.05, 0.6, 0.7]),
            threshold=0.1,
        )

        self.assertEqual(metrics["precision"], 0.6667)
        self.assertEqual(metrics["recall"], 0.6667)
        self.assertEqual(metrics["f1"], 0.6667)
        self.assertEqual(metrics["roc_auc"], 0.6667)

    def test_all_zero_labels_use_safe_fallbacks(self) -> None:
        metrics = calc_binary_metrics(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.3]),
            threshold=0.5,
        )

        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertFalse(math.isnan(metrics["brier"]))
        self.assertTrue(math.isnan(metrics["log_loss"]))


class OutcomeTargetSelectionTests(unittest.TestCase):
    def test_training_targets_remain_selected_target_family_while_diagnostics_use_goal_next10(self) -> None:
        labels = torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)
        labels[:, LABEL_INDEX["scores"]] = 0.729
        labels[:, LABEL_INDEX["scores_xg"]] = 0.42
        labels[:, LABEL_INDEX["concedes_xg"]] = 0.13
        labels[:, LABEL_INDEX["scores_goal_next10"]] = 1.0
        labels[:, LABEL_INDEX["concedes_goal_next10"]] = 0.0

        args = SimpleNamespace(use_xg=True, use_xt=False, use_goal_distance=False, use_epv=False)
        outcome_scoring, outcome_conceding = get_outcome_targets(labels, args)
        diagnostic_scoring, diagnostic_conceding = get_outcome_diagnostic_targets(labels)

        self.assertAlmostEqual(float(outcome_scoring[0]), 0.42, places=6)
        self.assertAlmostEqual(float(outcome_conceding[0]), 0.13, places=6)
        self.assertEqual(float(diagnostic_scoring[0]), 1.0)
        self.assertEqual(float(diagnostic_conceding[0]), 0.0)


class WrapperValidationTests(unittest.TestCase):
    def test_train_wrapper_accepts_xt_in_state_return_type(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["in_3"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--target-family",
                    "xt",
                    "--return_type",
                    "in_3",
                    "--intended-receiver-mode",
                    "original",
                ]
            )

        self.assertEqual(args.return_type, "in_3")

    def test_train_wrapper_rejects_xg_in_state_return_type(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["in_3"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--target-family",
                        "xg",
                        "--return_type",
                        "in_3",
                        "--intended-receiver-mode",
                        "original",
                    ]
                )

    def test_main_wrapper_rejects_xg_in_state_return_type(self) -> None:
        argv = [
            "scripts/main.py",
            "--skip-train",
            "--skip-evaluate",
            "--skip-run-relevant",
            "--skip-hawkeye",
            "--skip-benchmark",
            "--skip-skillcorner",
            "--target-family",
            "xg",
            "--return_type",
            "in_3",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                main_script.parse_args()


if __name__ == "__main__":
    unittest.main()
