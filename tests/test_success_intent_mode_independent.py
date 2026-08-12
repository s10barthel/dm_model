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
        "pass_height": False,
        "outcome_scoring": True,
        "outcome_conceding": True,
        "failure_receiver": False,
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
        pass_intent_model_id=None,
        pass_success_ipw=True,
        diagnostic_feature_run_id=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_pass_intent_record(
    *,
    task: str = "pass_intent",
    feature_run_id: str = "feature_run",
    intended_receiver_mode: str = "original",
    return_type: str = "disc_0.9",
    target_family: str | None = None,
    has_weights: bool = True,
    node_in_dim: int = 25,
    edge_in_dim: int = 4,
    add_v_edge_features: bool = True,
    feature_signature: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "task": task,
        "feature_run_id": feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": target_family,
        "has_weights": has_weights,
        "graph_schema": {
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "add_v_edge_features": add_v_edge_features,
        },
        "feature_signature": feature_signature or train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
        "metadata": metadata or {},
    }


def command_batch_size(command: list[str]) -> int:
    return int(command[command.index("--batch_size") + 1])


class SuccessIntentModeIndependentTests(unittest.TestCase):
    def test_default_parse_args_enables_failure_receiver_off(self) -> None:
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
        self.assertTrue(args.pass_success_ipw)

    def test_parse_args_disables_pass_success_ipw(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--return_type",
                    "disc_0.9",
                    "--intended-receiver-mode",
                    "original",
                    "--no-pass-success-ipw",
                    "--no-pass-intent",
                    "--no-outcome-scoring",
                    "--no-outcome-conceding",
                ]
            )

        self.assertFalse(args.pass_success_ipw)

    def test_pass_intent_model_id_rejected_when_pass_success_ipw_disabled(self) -> None:
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
                        "--return_type",
                        "disc_0.9",
                        "--intended-receiver-mode",
                        "original",
                        "--no-pass-intent",
                        "--no-pass-success-ipw",
                        "--pass-intent-model-id",
                        "pass_intent/old",
                        "--no-outcome-scoring",
                        "--no-outcome-conceding",
                    ]
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

    def test_train_wrapper_rejects_invalid_general_batch_size(self) -> None:
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
                        "--batch-size",
                        "0",
                    ]
                )

    def test_train_wrapper_rejects_invalid_model_batch_size(self) -> None:
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
                        "--pass-success-batch-size",
                        "-1",
                    ]
                )

    def test_pass_success_requires_pass_intent_or_external_model_id(self) -> None:
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

    def test_pass_success_accepts_external_pass_intent_model_id(self) -> None:
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
                    "--no-pass-intent",
                    "--pass-intent-model-id",
                    "pass_intent/old",
                ]
            )

        self.assertEqual(args.pass_intent_model_id, "pass_intent/old")
        self.assertFalse(args.enabled_tasks["pass_intent"])
        self.assertTrue(args.enabled_tasks["pass_success"])

    def test_pass_intent_model_id_rejects_same_run_pass_intent_training(self) -> None:
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
                        "--pass-intent-model-id",
                        "pass_intent/old",
                    ]
                )

    def test_pass_intent_model_id_rejects_disabled_pass_success(self) -> None:
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
                        "--no-pass-success",
                        "--pass-intent-model-id",
                        "pass_intent/old",
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

    def test_outcome_commands_pass_explicit_diagnostic_feature_run_id(self) -> None:
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
                diagnostic_feature_run_id="diagnostic_run",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertTrue(all("--diagnostic-feature-run-id" in command for command in commands))
        self.assertEqual(
            {command[command.index("--diagnostic-feature-run-id") + 1] for command in commands},
            {"diagnostic_run"},
        )

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

    def test_default_batch_sizes_are_task_specific(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(failure_receiver=False),
                target_family="goal",
                return_type="disc_0.9",
                intended_receiver_mode="original",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, _ = train_wrapper.build_training_commands(args)

        batch_sizes = {command[1]: command_batch_size(command) for command in commands}
        self.assertEqual(batch_sizes["action_intent"], 256)
        self.assertEqual(batch_sizes["pass_intent"], 256)
        self.assertEqual(batch_sizes["success_intent"], 256)
        self.assertEqual(batch_sizes["pass_success"], 512)
        self.assertEqual(batch_sizes["outcome_scoring"], 512)
        self.assertEqual(batch_sizes["outcome_conceding"], 512)

    def test_general_batch_size_overrides_all_task_defaults(self) -> None:
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
                batch_size=384,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual({command_batch_size(command) for command in commands}, {384})

    def test_model_specific_batch_size_overrides_general_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
                batch_size=384,
                pass_success_batch_size=640,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(train_wrapper, "get_model_record", return_value=make_pass_intent_record()),
            ):
                commands, _, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(command_batch_size(commands[0]), 640)

    def test_pass_success_uses_external_pass_intent_model_id_for_ipw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(return_type="next_5", target_family="xt"),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        task_names = [command[1] for command in commands]
        self.assertEqual(task_names, ["pass_success"])
        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        pass_success_args = commands[0]
        self.assertEqual(pass_success_args[pass_success_args.index("--ipw_model_id") + 1], "pass_intent/old")

    def test_pass_success_defaults_to_same_run_pass_intent_for_ipw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_intent", "pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "build_model_ids",
                    return_value={"pass_intent": "pass_intent/new", "pass_success": "pass_success/new"},
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_intent", "pass_success"})
        pass_success_args = commands[1]
        self.assertEqual(pass_success_args[pass_success_args.index("--ipw_model_id") + 1], "pass_intent/new")

    def test_pass_success_omits_ipw_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_success_ipw=False,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(train_wrapper, "get_model_record", side_effect=AssertionError("unexpected IPW validation")),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertNotIn("--ipw_model_id", commands[0])

    def test_pass_success_ipw_toggle_does_not_affect_other_model_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            common_overrides = dict(
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_success=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_intent"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
            )
            enabled_args = make_training_args("feature_run", pass_success_ipw=True, **common_overrides)
            disabled_args = make_training_args("feature_run", pass_success_ipw=False, **common_overrides)

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(train_wrapper, "build_model_ids", return_value={"pass_intent": "pass_intent/new"}),
            ):
                enabled_commands, _, _, _, _ = train_wrapper.build_training_commands(enabled_args)
                disabled_commands, _, _, _, _ = train_wrapper.build_training_commands(disabled_args)

        self.assertEqual(enabled_commands, disabled_commands)

    def test_external_pass_intent_accepts_mismatched_graph_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(edge_in_dim=2, add_v_edge_features=False),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertEqual(commands[0][commands[0].index("--ipw_model_id") + 1], "pass_intent/old")

    def test_external_pass_intent_accepts_smaller_ipw_schema_than_runtime(self) -> None:
        args = make_training_args(
            "feature_run",
            pass_intent_model_id="pass_intent/old",
        )
        runtime_schema = {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True}

        with patch.object(
            train_wrapper,
            "get_model_record",
            return_value=make_pass_intent_record(node_in_dim=19, edge_in_dim=2, add_v_edge_features=False),
        ):
            resolved = train_wrapper.validate_external_pass_intent_model_id(
                args,
                train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
                "feature_run",
                runtime_schema=runtime_schema,
            )

        self.assertEqual(resolved, "pass_intent/old")

    def test_external_pass_intent_rejects_runtime_edge_schema_too_small(self) -> None:
        args = make_training_args(
            "feature_run",
            pass_intent_model_id="pass_intent/old",
        )
        runtime_schema = {"node_in_dim": 25, "edge_in_dim": 2, "add_v_edge_features": False}

        with patch.object(
            train_wrapper,
            "get_model_record",
            return_value=make_pass_intent_record(node_in_dim=25, edge_in_dim=4, add_v_edge_features=True),
        ):
            with self.assertRaisesRegex(ValueError, "requires edge_in_dim=4.*Use --v-edge-features"):
                train_wrapper.validate_external_pass_intent_model_id(
                    args,
                    train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
                    "feature_run",
                    runtime_schema=runtime_schema,
                )

    def test_external_pass_intent_rejects_runtime_node_schema_too_small(self) -> None:
        args = make_training_args(
            "feature_run",
            pass_intent_model_id="pass_intent/old",
        )
        runtime_schema = {"node_in_dim": 19, "edge_in_dim": 4, "add_v_edge_features": True}

        with patch.object(
            train_wrapper,
            "get_model_record",
            return_value=make_pass_intent_record(node_in_dim=25, edge_in_dim=4, add_v_edge_features=True),
        ):
            with self.assertRaisesRegex(ValueError, "requires node_in_dim=25"):
                train_wrapper.validate_external_pass_intent_model_id(
                    args,
                    train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
                    "feature_run",
                    runtime_schema=runtime_schema,
                )

    def test_external_lane_survival_pass_intent_accepts_one_node_ipw_delta(self) -> None:
        args = make_training_args("feature_run", pass_intent_model_id="pass_intent/lane")
        runtime_schema = {"node_in_dim": 26, "edge_in_dim": 5, "add_v_edge_features": True}
        lane_signature = {**train_wrapper.WRAPPER_FEATURE_DEFAULTS, "lane_survival": True}

        with patch.object(
            train_wrapper,
            "get_model_record",
            return_value=make_pass_intent_record(
                node_in_dim=27,
                edge_in_dim=5,
                feature_signature=lane_signature,
                metadata={"lane_survival": {"enabled": True, "mode": "max"}},
            ),
        ):
            resolved = train_wrapper.validate_external_pass_intent_model_id(
                args,
                train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
                "feature_run",
                runtime_schema=runtime_schema,
            )

        self.assertEqual(resolved, "pass_intent/lane")

    def test_external_non_lane_pass_intent_rejects_one_node_ipw_delta(self) -> None:
        args = make_training_args("feature_run", pass_intent_model_id="pass_intent/not-lane")
        runtime_schema = {"node_in_dim": 26, "edge_in_dim": 5, "add_v_edge_features": True}

        with patch.object(
            train_wrapper,
            "get_model_record",
            return_value=make_pass_intent_record(node_in_dim=27, edge_in_dim=5),
        ):
            with self.assertRaisesRegex(ValueError, "requires node_in_dim=27"):
                train_wrapper.validate_external_pass_intent_model_id(
                    args,
                    train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy(),
                    "feature_run",
                    runtime_schema=runtime_schema,
                )

    def test_mixed_lane_survival_ipw_keeps_target_commands_without_lane_survival(self) -> None:
        lane_record = make_pass_intent_record(
            node_in_dim=27,
            edge_in_dim=5,
            feature_signature={**train_wrapper.WRAPPER_FEATURE_DEFAULTS, "lane_survival": True},
            metadata={"lane_survival": {"enabled": True, "mode": "max"}},
        )
        runtime_schema = {"node_in_dim": 26, "edge_in_dim": 5, "add_v_edge_features": True}
        cases = [
            ("pass_success", "pass_intent_model_id"),
            ("pass_height", "pass_height_ipw_model_id"),
        ]
        for task, model_id_attr in cases:
            enabled_tasks = make_enabled_tasks(
                action_intent=False,
                pass_intent=False,
                success_intent=False,
                pass_success=task == "pass_success",
                pass_height=task == "pass_height",
                outcome_scoring=False,
                outcome_conceding=False,
                failure_receiver=False,
            )
            args = make_training_args(
                "feature_run",
                enabled_tasks=enabled_tasks,
                trained_tasks=[task],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_success_ipw=task == "pass_success",
                pass_height_ipw=task == "pass_height",
                **{model_id_attr: "pass_intent/lane"},
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                with (
                    patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                    patch.object(train_wrapper, "resolve_feature_root", return_value=Path(tmpdir)),
                    patch.object(train_wrapper, "resolve_pass_success_runtime_schema", return_value=runtime_schema),
                    patch.object(train_wrapper, "get_model_record", return_value=lane_record),
                ):
                    commands, _, _, _, _ = train_wrapper.build_training_commands(args)
            self.assertIn("--no-lane-survival", commands[0])
            self.assertEqual(commands[0][commands[0].index("--ipw_model_id") + 1], "pass_intent/lane")

    def test_external_pass_intent_accepts_matching_possessor_masked_velocity_edge_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            source_signature = {
                **train_wrapper.WRAPPER_FEATURE_DEFAULTS,
                "v_edge_feature_mode": "no_poss",
            }
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
                v_edge_feature_mode="no_poss",
                mask_possessor_v_edge_features=True,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(feature_signature=source_signature),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertIn("--v-edge-features-no-poss", commands[0])

    def test_external_pass_intent_accepts_mismatched_velocity_edge_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            source_signature = {
                **train_wrapper.WRAPPER_FEATURE_DEFAULTS,
                "v_edge_feature_mode": "all",
            }
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
                v_edge_feature_mode="no_poss",
                mask_possessor_v_edge_features=True,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(feature_signature=source_signature),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertIn("--v-edge-features-no-poss", commands[0])

    def test_external_pass_intent_accepts_mismatched_feature_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            source_signature = train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy()
            source_signature["poss_vel_aware"] = False
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(feature_signature=source_signature),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertEqual(commands[0][commands[0].index("--ipw_model_id") + 1], "pass_intent/old")

    def test_external_pass_intent_accepts_mismatched_feature_run_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(
                    train_wrapper,
                    "get_model_record",
                    return_value=make_pass_intent_record(
                        feature_run_id="other_feature_run",
                        intended_receiver_mode="model",
                    ),
                ),
            ):
                commands, model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(set(model_ids.keys()), {"pass_success"})
        self.assertEqual(commands[0][commands[0].index("--ipw_model_id") + 1], "pass_intent/old")

    def test_external_pass_intent_rejects_wrong_task_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(train_wrapper, "get_model_record", return_value=make_pass_intent_record(task="action_intent")),
            ):
                with self.assertRaises(ValueError):
                    train_wrapper.build_training_commands(args)

    def test_external_pass_intent_rejects_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = make_training_args(
                "feature_run",
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_success"],
                target_family=None,
                return_type="disc_0.9",
                intended_receiver_mode="original",
                pass_intent_model_id="pass_intent/old",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
                patch.object(train_wrapper, "get_model_record", return_value=make_pass_intent_record(has_weights=False)),
            ):
                with self.assertRaises(ValueError):
                    train_wrapper.build_training_commands(args)

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
