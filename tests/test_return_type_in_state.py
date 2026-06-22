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
from datatools.goal_distance import (
    GOAL_DISTANCE_GOAL_X,
    GOAL_DISTANCE_GOAL_Y,
    GOAL_DISTANCE_MAX_RAW,
    GOAL_DISTANCE_MAX_VALUE,
    goal_distance_from_xy,
)
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from datatools import utils, xt
from models.utils import calc_binary_metrics, get_outcome_diagnostic_targets, get_outcome_targets
from scripts.generate_goal_distance import GOAL_DISTANCE_TARGET_RANGE
from scripts import generate_xt
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


class GoalDistanceScaleTest(unittest.TestCase):
    def test_goal_center_returns_max_value(self) -> None:
        values = goal_distance_from_xy(
            pd.Series([GOAL_DISTANCE_GOAL_X]),
            pd.Series([GOAL_DISTANCE_GOAL_Y]),
        )

        self.assertAlmostEqual(float(values[0]), GOAL_DISTANCE_MAX_VALUE)

    def test_opposite_corners_return_zero(self) -> None:
        values = goal_distance_from_xy(
            pd.Series([0.0, 0.0]),
            pd.Series([0.0, 68.0]),
        )

        self.assertEqual(values.tolist(), [0.0, 0.0])

    def test_midfield_point_is_bounded_and_uses_expected_formula(self) -> None:
        x = 52.5
        y = 34.0
        values = goal_distance_from_xy(pd.Series([x]), pd.Series([y]))
        expected = GOAL_DISTANCE_MAX_VALUE * (
            1.0 - math.hypot(GOAL_DISTANCE_GOAL_X - x, GOAL_DISTANCE_GOAL_Y - y) / GOAL_DISTANCE_MAX_RAW
        )

        self.assertGreater(float(values[0]), 0.0)
        self.assertLess(float(values[0]), GOAL_DISTANCE_MAX_VALUE)
        self.assertAlmostEqual(float(values[0]), expected)

    def test_nan_coordinates_remain_nan(self) -> None:
        values = goal_distance_from_xy(
            pd.Series([np.nan, GOAL_DISTANCE_GOAL_X]),
            pd.Series([GOAL_DISTANCE_GOAL_Y, np.nan]),
        )

        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(np.isnan(values[1]))

    def test_generated_metadata_target_range_uses_bounded_scale(self) -> None:
        self.assertEqual(GOAL_DISTANCE_TARGET_RANGE, [0.0, GOAL_DISTANCE_MAX_VALUE])


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
                ("interception", "home_1", 0.90),
                ("shot", "home_1", 0.40),
                ("pass", "away_1", 0.60),
                ("pass", "away_1", 0.30),
            ]
        )

        next_labeled = utils.label_xt_returns(events, lookahead_len=3)
        disc_labeled = utils.label_discounted_xt_returns(events, gamma=0.5)

        self.assertEqual(float(next_labeled.at[0, "scores_xT"]), 0.40)
        self.assertEqual(float(next_labeled.at[0, "concedes_xT"]), 0.60)
        expected_scoring = 1 - (1 - 0.20) * (1 - 0.5**2 * 0.40)
        expected_conceding = 1 - (1 - 0.5**3 * 0.60) * (1 - 0.5**4 * 0.30)
        self.assertAlmostEqual(float(disc_labeled.at[0, "scores_xT"]), expected_scoring)
        self.assertAlmostEqual(float(disc_labeled.at[0, "concedes_xT"]), expected_conceding)

    def test_discounted_future_probability_value_anchors_first_considered_row_at_one(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.70),
                ("pass", "home_1", 0.20),
                ("pass", "home_1", 0.30),
            ]
        )

        labeled = utils.label_discounted_xt_returns(events, gamma=0.5)

        expected_scoring = 1 - (1 - 0.70) * (1 - 0.5 * 0.20) * (1 - 0.5**2 * 0.30)
        self.assertAlmostEqual(float(labeled.at[0, "scores_xT"]), expected_scoring)

    def test_discounted_goal_distance_returns_probability_product(self) -> None:
        events = make_goal_distance_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("interception", "home_1", 0.90),
                ("shot", "home_1", 0.40),
                ("pass", "away_1", 0.60),
            ]
        )

        labeled = utils.label_discounted_goal_distance_returns(events, gamma=0.5)

        expected_scoring = 1 - (1 - 0.20) * (1 - 0.5**2 * 0.40)
        expected_conceding = 1 - (1 - 0.5**3 * 0.60)
        self.assertAlmostEqual(float(labeled.at[0, "scores_goal_distance"]), expected_scoring)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_goal_distance"]), expected_conceding)

    def test_discounted_epv_returns_probability_product(self) -> None:
        events = make_epv_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("interception", "home_1", 0.90),
                ("shot", "home_1", 0.40),
                ("pass", "away_1", 0.60),
            ]
        )

        labeled = utils.label_discounted_epv_returns(events, gamma=0.5)

        expected_scoring = 1 - (1 - 0.20) * (1 - 0.5**2 * 0.40)
        expected_conceding = 1 - (1 - 0.5**3 * 0.60)
        self.assertAlmostEqual(float(labeled.at[0, "scores_epv"]), expected_scoring)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_epv"]), expected_conceding)

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

    def test_label_discounted_future_probability_value_skip1_skips_first_non_shot(self) -> None:
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

        expected_scoring = 1 - (1 - 0.70) * (1 - 0.5 * 0.20)
        expected_conceding = 1 - (1 - 0.5**2 * 0.10)
        self.assertAlmostEqual(float(labeled.at[0, "scores_xT"]), expected_scoring)
        self.assertAlmostEqual(float(labeled.at[0, "concedes_xT"]), expected_conceding)

    def test_discounted_future_probability_value_clips_out_of_range_values(self) -> None:
        events = make_xt_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", -0.1),
                ("pass", "home_1", 1.2),
                ("pass", "away_1", 1.5),
            ]
        )

        labeled = utils.label_discounted_xt_returns(events, gamma=0.5)

        self.assertAlmostEqual(float(labeled.at[0, "scores_xT"]), 1 - (1 - 0.5 * 1.0))
        self.assertAlmostEqual(float(labeled.at[0, "concedes_xT"]), 1 - (1 - 0.5**2 * 1.0))

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


def _source_xt_grid(invert_x: bool = False) -> np.ndarray:
    grid = np.zeros((xt.XT_GRID_W, xt.XT_GRID_L), dtype=float)
    for row_index in range(xt.XT_GRID_W):
        y_index = xt.XT_GRID_W - 1 - row_index
        y = (y_index + 0.5) * (68.0 / xt.XT_GRID_W)
        centrality = math.sin(math.pi * y / 68.0)
        for x_index in range(xt.XT_GRID_L):
            x = (x_index + 0.5) * (105.0 / xt.XT_GRID_L)
            normalized_x = x / 105.0
            x_term = (1.0 - normalized_x) if invert_x else normalized_x
            grid[row_index, x_index] = 0.01 + 0.08 * x_term + 0.04 * centrality
    return grid


def _xt_calibration_actions() -> pd.DataFrame:
    rows = []
    action_id = 0
    for x in np.linspace(4.0, 101.0, 16):
        for y in np.linspace(4.0, 64.0, 12):
            rows.append(
                {
                    "game_id": "match_1",
                    "original_event_id": action_id,
                    "action_id": action_id,
                    "period_id": 1,
                    "seconds": float(action_id),
                    "team_id": "home_team",
                    "player_id": "player_1",
                    "start_x": float(x),
                    "start_y": float(y),
                    "end_x": float(min(x + 1.0, 105.0)),
                    "end_y": float(y),
                    "type_id": 0,
                    "result_id": 1,
                    "bodypart_id": 0,
                    "spadl_type": "pass",
                    "success": True,
                    "offside": False,
                    "xG": np.nan,
                }
            )
            action_id += 1
    return pd.DataFrame(rows)


def _xt_annotation_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "match_1",
                "stats_perform_match_id": "match_1",
                "original_event_id": 1,
                "action_id": 1,
                "period_id": 1,
                "seconds": 1.0,
                "team_id": "home_team",
                "player_id": "player_1",
                "object_id": "home_1",
                "start_x": 100.0,
                "start_y": 34.0,
                "end_x": 105.0,
                "end_y": 34.0,
                "spadl_type": "shot",
                "success": False,
                "offside": False,
                "expected_goal": 0.95,
            },
            {
                "game_id": "match_1",
                "stats_perform_match_id": "match_1",
                "original_event_id": 2,
                "action_id": 2,
                "period_id": 1,
                "seconds": 2.0,
                "team_id": "home_team",
                "player_id": "player_1",
                "object_id": "home_1",
                "start_x": 40.0,
                "start_y": 20.0,
                "end_x": 50.0,
                "end_y": 22.0,
                "spadl_type": "pass",
                "success": True,
                "offside": False,
                "expected_goal": np.nan,
            },
        ]
    )


class XTXYGLMTests(unittest.TestCase):
    def test_normalized_xt_xy_features_are_directional_and_symmetric(self) -> None:
        normalized_x, centrality = xt.normalized_xt_xy_features(
            np.array([0.0, 105.0, 52.5, 52.5, 52.5, 52.5]),
            np.array([34.0, 34.0, 20.0, 48.0, 0.0, 68.0]),
        )

        self.assertGreater(normalized_x[1], normalized_x[0])
        self.assertAlmostEqual(float(centrality[0]), 1.0)
        self.assertAlmostEqual(float(centrality[2]), float(centrality[3]))
        self.assertLess(centrality[2], centrality[0])
        self.assertAlmostEqual(float(centrality[4]), 0.0)
        self.assertAlmostEqual(float(centrality[5]), 0.0)

    def test_spatial_bin_weights_balance_populated_bins(self) -> None:
        normalized_x = np.array([0.1, 0.1, 0.1, 0.9])
        centrality = np.array([0.2, 0.2, 0.2, 0.8])
        weights = xt.compute_spatial_bin_weights(normalized_x, centrality, x_bins=2, centrality_bins=2)

        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_fit_xt_xy_glm_predicts_finite_monotonic_surface(self) -> None:
        fit = xt.fit_xt_xy_glm(_xt_calibration_actions(), _source_xt_grid())

        self.assertGreater(fit.model.coefficients["normalized_x"], 0)
        self.assertGreater(fit.model.coefficients["normalized_centrality"], 0)
        self.assertEqual(fit.model.terms, ("normalized_x", "normalized_centrality"))
        self.assertTrue(fit.diagnostics["monotonicity_check"]["passed"])
        self.assertTrue(fit.diagnostics["optimizer"]["success"])
        self.assertTrue(np.isfinite(fit.projection_grid).all())
        self.assertTrue(((fit.projection_grid >= 0.0) & (fit.projection_grid <= 1.0)).all())

        low = fit.model.predict_xy(np.array([20.0]), np.array([34.0]))[0]
        high = fit.model.predict_xy(np.array([90.0]), np.array([34.0]))[0]
        wide = fit.model.predict_xy(np.array([90.0]), np.array([2.0]))[0]
        self.assertGreater(high, low)
        self.assertGreater(high, wide)

    def test_zone_value_from_grid_uses_dynamic_source_grid_shape(self) -> None:
        grid = np.arange(24, dtype=float).reshape(4, 6)
        values = xt.zone_value_from_grid(pd.Series([1.0, 104.0]), pd.Series([67.0, 1.0]), grid)

        self.assertEqual(float(values[0]), float(grid[0, 0]))
        self.assertEqual(float(values[1]), float(grid[3, 5]))

    def test_fit_xt_xy_glm_enforces_monotonicity_for_inverted_source_grid(self) -> None:
        fit = xt.fit_xt_xy_glm(_xt_calibration_actions(), _source_xt_grid(invert_x=True))

        self.assertTrue(fit.diagnostics["optimizer"]["success"])
        self.assertTrue(fit.diagnostics["monotonicity_check"]["passed"])

    def test_xt_glm_nonlinear_terms_use_quadratic_cubic_features(self) -> None:
        terms = xt.resolve_xt_glm_terms(nonlinear_axes=["x", "y"])
        features = xt.build_xt_glm_design_frame(
            np.array([0.2, 0.5, 1.0]),
            np.array([0.2, 0.5, 1.0]),
            terms=terms,
        )

        self.assertEqual(
            terms,
            (
                "normalized_x",
                "normalized_centrality",
                "normalized_x_squared",
                "normalized_x_cubed",
                "normalized_centrality_squared",
                "normalized_centrality_cubed",
            ),
        )
        self.assertAlmostEqual(float(features.loc[1, "normalized_x_squared"]), 0.25)
        self.assertAlmostEqual(float(features.loc[1, "normalized_x_cubed"]), 0.125)
        self.assertAlmostEqual(float(features.loc[1, "normalized_centrality_squared"]), 0.25)
        self.assertAlmostEqual(float(features.loc[1, "normalized_centrality_cubed"]), 0.125)

    def test_fit_xt_xy_glm_supports_interaction_and_nonlinear_terms(self) -> None:
        fit = xt.fit_xt_xy_glm(
            _xt_calibration_actions(),
            _source_xt_grid(),
            use_interaction=True,
            nonlinear_axes=["x", "y"],
        )

        self.assertEqual(
            fit.model.terms,
            (
                "normalized_x",
                "normalized_centrality",
                "normalized_x_centrality",
                "normalized_x_squared",
                "normalized_x_cubed",
                "normalized_centrality_squared",
                "normalized_centrality_cubed",
            ),
        )
        for term in fit.model.terms:
            self.assertIn(term, fit.fit_sample.columns)
            self.assertIn(term, fit.model.coefficients)
        self.assertIn("monotonicity_check", fit.diagnostics)

    def test_fit_xt_xy_glm_interaction_with_nonlinear_y_records_monotonicity_violation(self) -> None:
        fit = xt.fit_xt_xy_glm(
            _xt_calibration_actions(),
            _source_xt_grid(),
            use_interaction=True,
            nonlinear_axes=["y"],
        )

        self.assertIn("monotonicity_check", fit.diagnostics)
        self.assertIn("monotonicity_error_raised", fit.diagnostics)

    def test_xt_glm_monotonicity_rejects_nonmonotonic_model(self) -> None:
        model = xt.XTXYLogitModel(
            coefficients={"intercept": -2.0, "normalized_x": -1.0, "normalized_centrality": 1.0},
            terms=("normalized_x", "normalized_centrality"),
        )

        with self.assertRaisesRegex(ValueError, "violated monotonicity"):
            xt.validate_xt_glm_monotonicity(model)

    def test_xt_glm_monotonicity_check_can_be_recorded_without_raising(self) -> None:
        model = xt.XTXYLogitModel(
            coefficients={"intercept": -2.0, "normalized_x": -1.0, "normalized_centrality": 1.0},
            terms=("normalized_x", "normalized_centrality"),
        )

        result = xt.check_xt_glm_monotonicity(model)

        self.assertFalse(result["passed"])
        self.assertGreater(result["x_violation_count"], 0)

    def test_annotate_match_xt_floors_shots_by_xg(self) -> None:
        model = xt.XTXYLogitModel(
            coefficients={"intercept": -6.0, "normalized_x": 1.0, "normalized_centrality": 1.0},
            terms=("normalized_x", "normalized_centrality"),
        )

        annotated, exported = xt.annotate_match_xt(_xt_annotation_events(), model)

        shot_row = exported.loc[exported["spadl_type"].eq("shot")].iloc[0]
        self.assertAlmostEqual(float(shot_row["xG"]), 0.95)
        self.assertAlmostEqual(float(shot_row["xT"]), 0.95)
        pass_row = exported.loc[exported["spadl_type"].eq("pass")].iloc[0]
        expected_pass_xt = model.predict_xy(np.array([40.0]), np.array([20.0]))[0]
        self.assertAlmostEqual(float(pass_row["xT"]), float(expected_pass_xt))
        self.assertIn("scores_xT", annotated.columns)
        self.assertIn("concedes_xT", annotated.columns)

    def test_build_xt_xy_surface_exports_whole_meter_grid(self) -> None:
        model = xt.XTXYLogitModel(
            coefficients={"intercept": -6.0, "normalized_x": 1.0, "normalized_centrality": 1.0},
            terms=("normalized_x", "normalized_centrality"),
        )
        surface = xt.build_xt_xy_surface(model)

        self.assertEqual(len(surface), 105 * 68)
        self.assertTrue({"x", "y", "normalized_x", "normalized_centrality", "xT"}.issubset(surface.columns))
        self.assertTrue(surface["xT"].between(0.0, 1.0).all())

    def test_generate_xt_load_source_grid_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                generate_xt.load_source_grid(Path(tmpdir) / "xT_source_grid.csv")

    def test_generate_xt_load_source_grid_infers_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xT_source_grid.csv"
            pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5)).to_csv(path, index=False)

            loaded = generate_xt.load_source_grid(path)

        self.assertEqual(loaded.shape, (3, 5))
        self.assertEqual(loaded.columns.tolist(), ["X0", "X1", "X2", "X3", "X4"])

    def test_generate_xt_parse_args_rejects_grid_size_with_reuse(self) -> None:
        with patch.object(sys, "argv", ["scripts/generate_xt.py", "--reuse-source-grid", "--source-grid-l", "16"]):
            with self.assertRaises(SystemExit):
                generate_xt.parse_args()

    def test_generate_xt_parse_args_accepts_custom_source_grid_size(self) -> None:
        with patch.object(sys, "argv", ["scripts/generate_xt.py", "--source-grid-l", "16", "--source-grid-w", "12"]):
            args = generate_xt.parse_args()

        self.assertEqual(args.source_grid_l, 16)
        self.assertEqual(args.source_grid_w, 12)

    def test_generate_xt_parse_args_accepts_interaction_and_repeated_nonlinear_axes(self) -> None:
        argv = [
            "scripts/generate_xt.py",
            "--use-interaction",
            "--use-nonlinear",
            "x",
            "--use-nonlinear",
            "y",
        ]
        with patch.object(sys, "argv", argv):
            args = generate_xt.parse_args()

        self.assertTrue(args.use_interaction)
        self.assertEqual(args.use_nonlinear, ["x", "y"])

    def test_generate_xt_parse_args_accepts_fit_only_with_model_flags(self) -> None:
        argv = [
            "scripts/generate_xt.py",
            "--fit-only",
            "--source-grid-l",
            "20",
            "--source-grid-w",
            "12",
            "--use-interaction",
            "--use-nonlinear",
            "x",
        ]
        with patch.object(sys, "argv", argv):
            args = generate_xt.parse_args()

        self.assertTrue(args.fit_only)
        self.assertEqual(args.source_grid_l, 20)
        self.assertEqual(args.source_grid_w, 12)
        self.assertTrue(args.use_interaction)
        self.assertEqual(args.use_nonlinear, ["x"])

    def test_generate_xt_fit_only_model_writer_does_not_write_match_exports(self) -> None:
        fit = xt.fit_xt_xy_glm(_xt_calibration_actions(), _source_xt_grid())
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "xT"
            match_dir = output_dir / "matches"
            with patch.object(generate_xt, "save_xt_xy_surface_plot") as save_plot:
                save_plot.side_effect = lambda surface, output_path: Path(output_path).write_text("plot", encoding="utf-8")
                generate_xt.save_model_outputs(fit, output_dir)

            self.assertTrue((output_dir / "xT_grid.csv").exists())
            self.assertTrue((output_dir / "xT_source_grid.csv").exists())
            self.assertTrue((output_dir / "xT_xy_surface.csv").exists())
            self.assertTrue((output_dir / "xT_glm_fit_sample.csv").exists())
            self.assertTrue((output_dir / "xT_xy_surface_3d.png").exists())
            self.assertFalse((output_dir / "xT.csv").exists())
            self.assertFalse(match_dir.exists())

    def test_generate_xt_ignored_export_filters_records_fit_only_filters(self) -> None:
        args = SimpleNamespace(match_id=["match_a", "match_b"], limit=3)

        ignored = generate_xt.ignored_export_filters(args)

        self.assertEqual(ignored, {"match_id": ["match_a", "match_b"], "limit": 3})


if __name__ == "__main__":
    unittest.main()
