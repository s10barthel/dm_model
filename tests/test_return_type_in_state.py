from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import project_config
from datatools import utils
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


class ReturnTypeValidationTests(unittest.TestCase):
    def test_validate_return_type_accepts_in_variant(self) -> None:
        self.assertEqual(project_config.validate_return_type("in_3"), "in_3")
        self.assertEqual(project_config.parse_return_type("in_3"), ("in", 3))

    def test_validate_return_type_rejects_invalid_in_variant(self) -> None:
        with self.assertRaises(ValueError):
            project_config.validate_return_type("in_0")
        with self.assertRaises(ValueError):
            project_config.validate_return_type("in_three")

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

    def test_infer_feature_run_return_types_detects_in_state_label_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir)
            (run_root / "action_labels_in_3").mkdir()
            (run_root / "action_labels_next_5_angle_only").mkdir()

            with (
                patch.object(project_config, "load_feature_run_metadata", return_value={}),
                patch.object(project_config, "get_feature_run_root", return_value=run_root),
            ):
                return_types = project_config.infer_feature_run_return_types("feature_run")

        self.assertEqual(return_types, ["in_3", "next_5"])


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
        self.assertEqual(float(disc_labeled.at[0, "scores_xT"]), 0.20)
        self.assertEqual(float(disc_labeled.at[0, "concedes_xT"]), 0.30)


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
