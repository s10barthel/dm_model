from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from datatools.success_intent import build_success_intent_resolved_actions
from project_config import get_success_intent_graph_dir, get_success_intent_label_dir
from scripts import train_relevant_models as train_wrapper


def make_enabled_tasks(**overrides: bool) -> dict[str, bool]:
    enabled = {
        "action_intent": True,
        "pass_intent": True,
        "success_intent": True,
        "pass_success": True,
        "outcome_scoring": True,
        "outcome_conceding": True,
        "failure_receiver": True,
    }
    enabled.update(overrides)
    return enabled


def make_training_args(feature_run_id: str, **overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        feature_run_id=feature_run_id,
        success_intent_only=False,
        enabled_tasks=make_enabled_tasks(),
        trained_tasks=[
            "action_intent",
            "pass_intent",
            "success_intent",
            "pass_success",
            "outcome_scoring",
            "outcome_conceding",
            "failure_receiver",
        ],
        intended_receiver_mode="original",
        target_family="goal",
        return_type="disc_0.9",
        use_v_edge_features=True,
        outcome_scoring_trial=None,
        outcome_conceding_trial=None,
        xy_only=None,
        possessor_aware=None,
        keeper_aware=None,
        ball_z_aware=None,
        poss_vel_aware=None,
        accel_aware=None,
        extend_features=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SuccessIntentModeIndependentTests(unittest.TestCase):
    def test_default_parse_args_enables_all_wrapper_models(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--target-family",
                    "goal",
                    "--return_type",
                    "disc_0.9",
                    "--intended-receiver-mode",
                    "original",
                ]
            )

        self.assertEqual(
            args.enabled_tasks,
            make_enabled_tasks(),
        )

    def test_success_intent_only_parse_args_does_not_require_mode(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(["--feature-run-id", "feature_run", "--success-intent-only"])

        self.assertIsNone(args.intended_receiver_mode)
        self.assertEqual(args.return_type, "disc_0.9")

    def test_success_intent_only_rejects_mode_argument(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--success-intent-only",
                        "--intended-receiver-mode",
                        "original",
                    ]
                )

    def test_success_intent_only_rejects_explicit_model_toggle(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--success-intent-only",
                        "--no-pass-intent",
                    ]
                )

    def test_regular_training_still_requires_mode(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--target-family",
                        "goal",
                        "--return_type",
                        "disc_0.9",
                    ]
                )

    def test_success_intent_toggle_can_run_without_mode_or_outcome_settings(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["next_5", "disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--no-action-intent",
                    "--no-pass-intent",
                    "--no-pass-success",
                    "--no-outcome-scoring",
                    "--no-outcome-conceding",
                    "--no-failure-receiver",
                ]
            )

        self.assertIsNone(args.intended_receiver_mode)
        self.assertIsNone(args.target_family)
        self.assertEqual(args.return_type, "next_5")
        self.assertEqual(
            args.enabled_tasks,
            make_enabled_tasks(
                action_intent=False,
                pass_intent=False,
                pass_success=False,
                outcome_scoring=False,
                outcome_conceding=False,
                failure_receiver=False,
            ),
        )

    def test_non_outcome_training_auto_selects_first_available_return_type(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["next_5", "disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--intended-receiver-mode",
                    "original",
                    "--no-outcome-scoring",
                    "--no-outcome-conceding",
                ]
            )

        self.assertEqual(args.return_type, "next_5")
        self.assertIsNone(args.target_family)

    def test_pass_success_requires_pass_intent(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--target-family",
                        "goal",
                        "--return_type",
                        "disc_0.9",
                        "--intended-receiver-mode",
                        "original",
                        "--no-pass-intent",
                    ]
                )

    def test_disabled_outcome_trial_is_rejected(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            with self.assertRaises(SystemExit):
                train_wrapper.parse_args(
                    [
                        "--feature-run-id",
                        "feature_run",
                        "--target-family",
                        "goal",
                        "--return_type",
                        "disc_0.9",
                        "--intended-receiver-mode",
                        "original",
                        "--no-outcome-scoring",
                        "--outcome-scoring-trial",
                        "7",
                    ]
                )

    def test_build_success_intent_resolved_actions_only_labels_successful_pass_receivers(self) -> None:
        actions = pd.DataFrame(
            [
                {"action_type": "pass", "success": True, "object_id": "home_1", "receiver_id": "home_2"},
                {"action_type": "pass", "success": False, "object_id": "home_1", "receiver_id": "home_3"},
                {"action_type": "pass", "success": True, "object_id": "home_1", "receiver_id": "away_4"},
                {"action_type": "dribble", "success": True, "object_id": "home_1", "receiver_id": "home_1"},
                {"action_type": "pass", "success": True, "object_id": "home_1", "receiver_id": pd.NA},
                {"action_type": "pass", "success": True, "object_id": "away_1", "receiver_id": "away_goal"},
            ]
        )

        resolved_actions = build_success_intent_resolved_actions(actions)

        self.assertEqual(resolved_actions.at[0, "intent_id"], "home_2")
        for index in [1, 2, 3, 4, 5]:
            self.assertTrue(pd.isna(resolved_actions.at[index, "intent_id"]))
        self.assertEqual(resolved_actions.attrs["success_intent_stats"]["labeled_successful_pass_actions"], 1)

    def test_success_intent_training_commands_use_dedicated_label_dir_and_no_mode_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                success_intent_only=True,
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    pass_success=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["success_intent"],
                intended_receiver_mode=None,
                target_family=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, model_ids, intended_receiver_mode, resolved_feature_run_id, _ = train_wrapper.build_training_commands(
                    args
                )

        self.assertEqual(model_ids.keys(), {"success_intent"})
        self.assertIsNone(intended_receiver_mode)
        self.assertEqual(resolved_feature_run_id, "feature_run")

        command = commands[0]
        self.assertNotIn("--intended-receiver-mode", command)
        self.assertIn("--label-source", command)
        self.assertIn("receiver_id", command)
        self.assertIn("--training-filter", command)
        self.assertIn("successful_pass_actions", command)

        feature_dir = command[command.index("--feature_dir") + 1]
        label_dir = command[command.index("--label_dir") + 1]
        self.assertEqual(feature_dir, str(get_success_intent_graph_dir(feature_root)))
        self.assertEqual(label_dir, str(get_success_intent_label_dir(root=feature_root)))

    def test_outcomes_only_build_training_commands_emit_exactly_two_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    pass_success=False,
                    failure_receiver=False,
                ),
                trained_tasks=["outcome_scoring", "outcome_conceding"],
                target_family="xt",
                return_type="next_5",
                intended_receiver_mode="original",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(len(commands), 2)
        self.assertEqual(set(model_ids.keys()), {"outcome_scoring", "outcome_conceding"})
        self.assertTrue(all("--use_xt" in command for command in commands))

    def test_success_intent_and_outcomes_build_training_commands_emit_three_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    pass_success=False,
                    failure_receiver=False,
                ),
                trained_tasks=["success_intent", "outcome_scoring", "outcome_conceding"],
                target_family="goal",
                return_type="disc_0.9",
                intended_receiver_mode="original",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(len(commands), 3)
        self.assertEqual(set(model_ids.keys()), {"success_intent", "outcome_scoring", "outcome_conceding"})
        success_intent_args = next(command for command in commands if command[1] == "success_intent")
        self.assertIn("--label-source", success_intent_args)
        self.assertIn("--training-filter", success_intent_args)

    def test_failure_receiver_can_be_disabled_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(failure_receiver=False),
                trained_tasks=[
                    "action_intent",
                    "pass_intent",
                    "success_intent",
                    "pass_success",
                    "outcome_scoring",
                    "outcome_conceding",
                ],
                target_family="goal",
                return_type="disc_0.9",
                intended_receiver_mode="original",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        task_names = [command[1] for command in commands]
        self.assertNotIn("failure_receiver", task_names)
        self.assertNotIn("failure_receiver", model_ids)


if __name__ == "__main__":
    unittest.main()
