from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from project_config import resolve_requested_return_types
from scripts import generate_relevant_features as generator

DEFAULT_METADATA = object()


def make_metadata(
    return_types: list[str] | None = None,
    modes: list[str] | None = None,
    model_id: str | None = None,
    status: str = "completed",
    graph_schema: dict[str, object] | None = None,
    next_action_conditions_enabled: bool | None = True,
) -> dict[str, object]:
    metadata = {
        "status": status,
        "graph_schema": graph_schema or generator.EXPECTED_GRAPH_SCHEMA.copy(),
        "return_types": return_types or ["disc_0.9"],
        "intended_receiver_modes": modes or ["original", "angle_only"],
        "intended_receiver_model_id": model_id,
    }
    if next_action_conditions_enabled is not None:
        metadata["next_action_conditions_enabled"] = next_action_conditions_enabled
    return metadata


def make_args(
    requested_return_types: list[str] | None = None,
    model_id: str | None = None,
    run_id: str | None = "derived",
    replace_model: bool = False,
    refresh_target_families: list[str] | None = None,
    pass_height: bool = False,
    in_place: bool = False,
    overwrite_feature_run: bool = False,
    next_action_conditions_enabled: bool = True,
    num_workers: str = "1",
    worker_thread_limit: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        extend_feature_run_id="base",
        run_id=run_id,
        requested_return_types=(
            resolve_requested_return_types(requested_return_types) if requested_return_types is not None else []
        ),
        intended_receiver_model_id=model_id,
        replace_intended_receiver_model=replace_model,
        refresh_target_families=refresh_target_families or [],
        pass_height=pass_height,
        in_place=in_place,
        overwrite_feature_run=overwrite_feature_run,
        next_action_conditions_enabled=next_action_conditions_enabled,
        num_workers=num_workers,
        worker_thread_limit=worker_thread_limit,
    )


class FeatureRunExtensionPlanTests(unittest.TestCase):
    def build_plan(
        self,
        args: SimpleNamespace,
        metadata: dict[str, object] | None | object = DEFAULT_METADATA,
        existing_output: bool = False,
    ) -> generator.FeatureExtensionPlan:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            if existing_output and args.run_id:
                (root / args.run_id).mkdir()
            loaded_metadata = make_metadata() if metadata is DEFAULT_METADATA else metadata

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=loaded_metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "generate_run_id", return_value="generated"),
            ):
                return generator.build_extension_plan(args, python="python")

    def test_inherits_base_return_types_when_adding_model_without_return_type(self) -> None:
        plan = self.build_plan(make_args(model_id="success_intent/42"))

        self.assertEqual(plan.final_return_types, ["disc_0.9"])
        self.assertEqual(plan.added_return_types, [])
        self.assertEqual(plan.added_intended_receiver_modes, ["model"])
        self.assertEqual(plan.intended_receiver_model_id, "success_intent/42")
        self.assertEqual(
            [step.description for step in plan.command_steps],
            [
                "train split with model mode (labels-only)",
                "test split with model mode (labels-only)",
                "train split with model mode intent_train_augmented (labels-only)",
            ],
        )

    def test_unions_requested_return_types_in_order(self) -> None:
        plan = self.build_plan(make_args(["next_5", "disc_0.9", "in_3"]))

        self.assertEqual(plan.final_return_types, ["disc_0.9", "next_5", "in_3"])
        self.assertEqual(plan.added_return_types, ["next_5", "in_3"])
        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only"])
        self.assertEqual(
            [step.description for step in plan.command_steps],
            [
                "train split (labels-only)",
                "test split (labels-only)",
                "train split with intent_train_augmented (labels-only)",
            ],
        )

    def test_model_mode_addition_updates_final_modes(self) -> None:
        plan = self.build_plan(make_args(["next_5"], model_id="success_intent/42"))

        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only", "model"])
        self.assertEqual(plan.added_intended_receiver_modes, ["model"])
        self.assertEqual(len(plan.command_steps), 6)
        self.assertEqual(
            [step.description for step in plan.command_steps],
            [
                "train split (labels-only)",
                "test split (labels-only)",
                "train split with intent_train_augmented (labels-only)",
                "train split with model mode (labels-only)",
                "test split with model mode (labels-only)",
                "train split with model mode intent_train_augmented (labels-only)",
            ],
        )

    def test_noop_extension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["disc_0.9"]))

    def test_refresh_only_extension_regenerates_existing_labels(self) -> None:
        plan = self.build_plan(make_args(refresh_target_families=["epv"]))

        self.assertEqual(plan.final_return_types, ["disc_0.9"])
        self.assertEqual(plan.added_return_types, [])
        self.assertEqual(plan.refresh_target_families, ["epv"])
        self.assertEqual(plan.refreshed_return_types, ["disc_0.9"])
        self.assertEqual(plan.refreshed_intended_receiver_modes, ["original", "angle_only"])
        self.assertEqual(
            [step.description for step in plan.command_steps],
            [
                "train split target-label refresh (labels-only)",
                "test split target-label refresh (labels-only)",
                "train split target-label refresh with intent_train_augmented (labels-only)",
            ],
        )
        self.assertTrue(all("--labels-only" in command for command in plan.commands))
        self.assertTrue(all("--overwrite-labels" in command for command in plan.commands))

    def test_refresh_target_families_are_deduplicated_in_order(self) -> None:
        plan = self.build_plan(make_args(refresh_target_families=["epv", "xt", "epv", "goal_distance"]))

        self.assertEqual(plan.refresh_target_families, ["epv", "xt", "goal_distance"])

    def test_unknown_refresh_target_family_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(refresh_target_families=["bad"]))

    def test_parse_rejects_unknown_refresh_target_family(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "generate_relevant_features.py",
                "--extend-feature-run-id",
                "base",
                "--refresh-target-family",
                "bad",
            ],
        ):
            with self.assertRaises(SystemExit):
                generator.parse_args()

    def test_parse_rejects_refresh_without_extension(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "generate_relevant_features.py",
                "--refresh-target-family",
                "epv",
            ],
        ):
            with self.assertRaises(SystemExit):
                generator.parse_args()

    def test_parse_defaults_next_action_conditions_on(self) -> None:
        with patch.object(sys, "argv", ["generate_relevant_features.py"]):
            args = generator.parse_args()

        self.assertTrue(args.next_action_conditions_enabled)

    def test_parse_accepts_next_action_conditions_off(self) -> None:
        with patch.object(sys, "argv", ["generate_relevant_features.py", "--next-action-conditions-off"]):
            args = generator.parse_args()

        self.assertFalse(args.next_action_conditions_enabled)

    def test_parse_accepts_worker_flags(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "generate_relevant_features.py",
                "--num-workers",
                "auto",
                "--worker-thread-limit",
                "2",
            ],
        ):
            args = generator.parse_args()

        self.assertEqual(args.num_workers, "auto")
        self.assertEqual(args.worker_thread_limit, 2)

    def test_next_action_conditions_off_propagates_to_full_generation_commands(self) -> None:
        command = generator.with_mode_flags(
            ["python", "datatools/graph_feature.py"],
            SimpleNamespace(
                return_types=[],
                intended_receiver_model_id=None,
                run_id=None,
                next_action_conditions_enabled=False,
                num_workers="1",
                worker_thread_limit=1,
            ),
        )

        self.assertIn("--next-action-conditions-off", command)
        self.assertNotIn("--next-action-conditions-on", command)

    def test_worker_flags_propagate_to_full_generation_commands(self) -> None:
        command = generator.with_mode_flags(
            ["python", "datatools/graph_feature.py"],
            SimpleNamespace(
                return_types=[],
                intended_receiver_model_id=None,
                run_id=None,
                next_action_conditions_enabled=True,
                num_workers="auto",
                worker_thread_limit=3,
            ),
        )

        self.assertIn("--num-workers", command)
        self.assertIn("auto", command)
        self.assertIn("--worker-thread-limit", command)
        self.assertIn("3", command)

    def test_output_run_must_not_already_exist(self) -> None:
        with self.assertRaises(FileExistsError):
            self.build_plan(make_args(["next_5"]), existing_output=True)

    def test_output_run_must_not_equal_base_run(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"], run_id="base"))

    def test_in_place_targets_base_run_for_additive_extension(self) -> None:
        plan = self.build_plan(make_args(["next_5"], run_id=None, in_place=True))

        self.assertEqual(plan.extension_mode, generator.EXTENSION_MODE_IN_PLACE)
        self.assertEqual(plan.output_run_id, "base")
        self.assertEqual(plan.target_run_id, "base")
        self.assertTrue(all("--run-id" in command and "base" in command for command in plan.commands))

    def test_in_place_rejects_run_id(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"], run_id="derived", in_place=True))

    def test_in_place_rejects_refresh_pass_height_and_model_replacement(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(run_id=None, in_place=True, refresh_target_families=["epv"]))

        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"], run_id=None, in_place=True, pass_height=True))

        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )
        with self.assertRaises(ValueError):
            self.build_plan(
                make_args(run_id=None, in_place=True, model_id="success_intent/new", replace_model=True),
                metadata=metadata,
            )

    def test_overwrite_feature_run_targets_base_run_for_refresh_and_replacement(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )
        plan = self.build_plan(
            make_args(
                run_id=None,
                overwrite_feature_run=True,
                model_id="success_intent/new",
                replace_model=True,
                refresh_target_families=["epv"],
            ),
            metadata=metadata,
        )

        self.assertEqual(plan.extension_mode, generator.EXTENSION_MODE_OVERWRITE_FEATURE_RUN)
        self.assertEqual(plan.output_run_id, "base")
        self.assertEqual(plan.target_run_id, "base")
        self.assertTrue(plan.regenerate_model_mode)
        self.assertTrue(all("--run-id" in command and "base" in command for command in plan.commands))

    def test_mutating_extension_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(
                make_args(["next_5"], run_id=None, in_place=True, overwrite_feature_run=True),
            )

    def test_parse_rejects_mutating_mode_without_extension(self) -> None:
        with patch.object(sys, "argv", ["generate_relevant_features.py", "--in-place"]):
            with self.assertRaises(SystemExit):
                generator.parse_args()

    def test_parse_rejects_mutating_mode_with_run_id(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "generate_relevant_features.py",
                "--extend-feature-run-id",
                "base",
                "--run-id",
                "derived",
                "--overwrite-feature-run",
            ],
        ):
            with self.assertRaises(SystemExit):
                generator.parse_args()

    def test_parse_rejects_in_place_regenerative_flags(self) -> None:
        for flag in ["--refresh-target-family", "--pass-height", "--replace-intended-receiver-model"]:
            argv = ["generate_relevant_features.py", "--extend-feature-run-id", "base", "--in-place"]
            if flag == "--refresh-target-family":
                argv.extend([flag, "epv"])
            elif flag == "--replace-intended-receiver-model":
                argv.extend(["--intended-receiver-model-id", "success_intent/new", flag])
            else:
                argv.append(flag)
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    generator.parse_args()

    def test_next_action_conditions_off_propagates_to_extension_commands(self) -> None:
        metadata = make_metadata(next_action_conditions_enabled=False)
        plan = self.build_plan(
            make_args(["next_5"], next_action_conditions_enabled=False),
            metadata=metadata,
        )

        self.assertFalse(plan.next_action_conditions_enabled)
        self.assertTrue(all("--next-action-conditions-off" in command for command in plan.commands))
        self.assertFalse(any("--next-action-conditions-on" in command for command in plan.commands))

    def test_extension_rejects_next_action_condition_mismatch(self) -> None:
        metadata = make_metadata(next_action_conditions_enabled=True)

        with self.assertRaises(ValueError) as exc:
            self.build_plan(
                make_args(["next_5"], next_action_conditions_enabled=False),
                metadata=metadata,
            )

        self.assertIn("same next-action condition setting", str(exc.exception))

    def test_legacy_metadata_defaults_next_action_conditions_on(self) -> None:
        metadata = make_metadata(next_action_conditions_enabled=None)
        plan = self.build_plan(make_args(["next_5"]), metadata=metadata)

        self.assertTrue(plan.next_action_conditions_enabled)
        self.assertTrue(all("--next-action-conditions-on" in command for command in plan.commands))

    def test_missing_metadata_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.build_plan(make_args(["next_5"]), metadata=None)

    def test_incomplete_metadata_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"]), metadata=make_metadata(status="failed"))

    def test_conflicting_existing_model_id_is_rejected(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"], model_id="success_intent/new"), metadata=metadata)

    def test_conflicting_existing_model_id_can_be_replaced(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        plan = self.build_plan(
            make_args(model_id="success_intent/new", replace_model=True),
            metadata=metadata,
        )

        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only", "model"])
        self.assertEqual(plan.added_intended_receiver_modes, [])
        self.assertEqual(plan.intended_receiver_model_id, "success_intent/new")
        self.assertTrue(plan.regenerate_model_mode)
        self.assertEqual(plan.replaced_intended_receiver_model_id, "success_intent/old")
        self.assertEqual(plan.replaced_intended_receiver_modes, ["model"])
        self.assertEqual(len(plan.commands), 3)
        self.assertTrue(all("--labels-only" in command for command in plan.commands))

    def test_refresh_existing_model_mode_preserves_model_id(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        plan = self.build_plan(make_args(refresh_target_families=["epv"]), metadata=metadata)

        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only", "model"])
        self.assertEqual(plan.intended_receiver_model_id, "success_intent/old")
        self.assertEqual(plan.refreshed_intended_receiver_modes, ["original", "angle_only", "model"])
        self.assertEqual(len(plan.commands), 6)
        model_commands = [
            command
            for command in plan.commands
            if "--only-intended-receiver-mode" in command and "model" in command
        ]
        self.assertEqual(len(model_commands), 3)
        self.assertTrue(all("success_intent/old" in command for command in model_commands))
        self.assertTrue(all("--overwrite-labels" in command for command in model_commands))
        self.assertFalse(any("--augment-blocks-from-existing-graphs" in command for command in model_commands))

    def test_replace_model_requires_requested_model_id(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        with self.assertRaises(ValueError):
            self.build_plan(make_args(replace_model=True), metadata=metadata)

    def test_added_return_types_with_replacement_generate_non_model_then_model_commands(self) -> None:
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        plan = self.build_plan(
            make_args(["next_5"], model_id="success_intent/new", replace_model=True),
            metadata=metadata,
        )

        self.assertEqual(plan.added_return_types, ["next_5"])
        self.assertEqual(plan.final_return_types, ["disc_0.9", "next_5"])
        self.assertEqual(len(plan.commands), 6)
        added_return_commands = plan.commands[:3]
        replacement_commands = plan.commands[3:]
        self.assertFalse(
            any("--only-intended-receiver-mode" in command and "model" in command for command in added_return_commands)
        )
        self.assertTrue(all("model" in command for command in replacement_commands))
        self.assertTrue(all("success_intent/new" in command for command in replacement_commands))

    def test_graph_schema_mismatch_is_rejected(self) -> None:
        metadata = make_metadata(graph_schema={"edge_in_dim": 2, "add_v_edge_features": False})

        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"]), metadata=metadata)

    def test_extension_commands_are_labels_only_and_not_full_generation(self) -> None:
        plan = self.build_plan(make_args(["next_5"], model_id="success_intent/42"))

        self.assertEqual(len(plan.commands), 6)
        self.assertEqual(plan.commands, [step.command for step in plan.command_steps])
        self.assertTrue(all("--labels-only" in command for command in plan.commands))
        self.assertTrue(all("--post_action" not in command for command in plan.commands))
        self.assertFalse(
            any(
                "--feature_variant" in command
                and command[command.index("--feature_variant") + 1] == "success_intent"
                for command in plan.commands
            )
        )

    def test_extension_commands_propagate_worker_flags(self) -> None:
        plan = self.build_plan(make_args(["next_5"], num_workers="auto", worker_thread_limit=2))

        self.assertTrue(all("--num-workers" in command for command in plan.commands))
        self.assertTrue(all("--worker-thread-limit" in command for command in plan.commands))
        self.assertTrue(all("auto" in command for command in plan.commands))
        self.assertTrue(all("2" in command for command in plan.commands))

    def test_full_generation_commands_have_documented_step_descriptions(self) -> None:
        steps = generator.full_generation_commands("python")

        self.assertEqual(len(steps), 5)
        self.assertEqual(
            [step.description for step in steps],
            [
                "train split with post_action + augment_blocks",
                "test split with post_action",
                "train split with intent_train_augmented",
                "train split with success_intent",
                "test split with success_intent",
            ],
        )


class FeatureRunExtensionExecutionTests(unittest.TestCase):
    def test_successful_extension_copies_base_artifacts_and_updates_latest_after_commands(self) -> None:
        args = make_args(["next_5"], run_id="derived")
        metadata = make_metadata()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "artifact.txt").write_text("copied", encoding="utf-8")
            (base_root / "metadata.json").write_text('{"status": "completed"}', encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            output_root = root / "derived"
            output_metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "artifact.txt").exists())
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["return_types"], ["disc_0.9", "next_5"])
            self.assertEqual(output_metadata["derived_from_feature_run_id"], "base")
            self.assertEqual(output_metadata["extension_refresh_target_families"], [])
            self.assertEqual(output_metadata["extension_refreshed_return_types"], [])
            self.assertEqual(output_metadata["extension_refreshed_intended_receiver_modes"], [])
            self.assertEqual(run_command.call_count, 3)
            write_latest_run.assert_called_once_with("feature", "derived")

    def test_in_place_extension_mutates_base_without_copy_or_latest_update(self) -> None:
        args = make_args(["next_5"], run_id=None, in_place=True)
        metadata = make_metadata()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "artifact.txt").write_text("stays", encoding="utf-8")
            (base_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "copy_base_feature_run") as copy_base_feature_run,
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            output_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertFalse((root / "derived").exists())
            copy_base_feature_run.assert_not_called()
            write_latest_run.assert_not_called()
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["run_id"], "base")
            self.assertEqual(output_metadata["return_types"], ["disc_0.9", "next_5"])
            self.assertEqual(output_metadata["last_extension_mode"], generator.EXTENSION_MODE_IN_PLACE)
            self.assertEqual(output_metadata["extension_history"][-1]["status"], "completed")
            self.assertEqual(output_metadata["extension_history"][-1]["mode"], generator.EXTENSION_MODE_IN_PLACE)
            self.assertEqual(run_command.call_count, 3)
            self.assertTrue(all("base" in call.args[0] for call in run_command.call_args_list))

    def test_failed_in_place_extension_keeps_advertised_base_metadata(self) -> None:
        args = make_args(["next_5"], run_id=None, in_place=True)
        metadata = make_metadata()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "copy_base_feature_run") as copy_base_feature_run,
                patch.object(generator, "run_command", side_effect=RuntimeError("boom")),
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                with self.assertRaises(RuntimeError):
                    generator.run_extension_generation(args)

            output_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
            copy_base_feature_run.assert_not_called()
            write_latest_run.assert_not_called()
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["return_types"], ["disc_0.9"])
            self.assertEqual(output_metadata["extension_added_return_types"], [])
            self.assertEqual(output_metadata["extension_history"][-1]["status"], "failed")
            self.assertIn("boom", output_metadata["extension_history"][-1]["error"])

    def test_overwrite_feature_run_refresh_mutates_base_without_copy_or_latest_update(self) -> None:
        args = make_args(run_id=None, overwrite_feature_run=True, refresh_target_families=["epv"])
        metadata = make_metadata(return_types=["next_5", "in_3"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "copy_base_feature_run") as copy_base_feature_run,
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            output_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
            copy_base_feature_run.assert_not_called()
            write_latest_run.assert_not_called()
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["run_id"], "base")
            self.assertEqual(output_metadata["return_types"], ["next_5", "in_3"])
            self.assertEqual(output_metadata["last_extension_mode"], generator.EXTENSION_MODE_OVERWRITE_FEATURE_RUN)
            self.assertEqual(output_metadata["extension_history"][-1]["status"], "completed")
            self.assertTrue(all("--overwrite-labels" in call.args[0] for call in run_command.call_args_list))

    def test_refresh_extension_records_metadata_and_overwrites_labels(self) -> None:
        args = make_args(run_id="derived", refresh_target_families=["epv"])
        metadata = make_metadata(return_types=["next_5", "in_3"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "artifact.txt").write_text("copied", encoding="utf-8")
            (base_root / "metadata.json").write_text('{"status": "completed"}', encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            output_root = root / "derived"
            output_metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "artifact.txt").exists())
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["return_types"], ["next_5", "in_3"])
            self.assertEqual(output_metadata["extension_refresh_target_families"], ["epv"])
            self.assertEqual(output_metadata["extension_refreshed_return_types"], ["next_5", "in_3"])
            self.assertEqual(
                output_metadata["extension_refreshed_intended_receiver_modes"],
                ["original", "angle_only"],
            )
            self.assertEqual(run_command.call_count, 3)
            self.assertTrue(all("--overwrite-labels" in call.args[0] for call in run_command.call_args_list))
            write_latest_run.assert_called_once_with("feature", "derived")

    def test_full_generation_metadata_records_worker_settings(self) -> None:
        args = SimpleNamespace(
            extend_feature_run_id=None,
            run_id="feature_demo",
            return_types=["disc_0.9"],
            intended_receiver_modes=["original", "angle_only"],
            intended_receiver_model_id=None,
            next_action_conditions_enabled=True,
            num_workers="auto",
            worker_thread_limit=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "run_generation_steps"),
                patch.object(generator, "write_latest_run"),
            ):
                generator.run_full_generation(args)

            metadata = json.loads((root / "feature_demo" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["num_workers"], "auto")
        self.assertEqual(metadata["worker_thread_limit"], 2)

    def test_replacement_removes_only_copied_model_mode_artifacts(self) -> None:
        args = make_args(model_id="success_intent/new", run_id="derived", replace_model=True)
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            for relative_path in [
                "resolved_actions_model/copied.parquet",
                "action_labels_disc_0.9_model/copied.pt",
                "action_labels_intent_train_disc_0.9_model/copied.pt",
                "augmented_graphs_model/copied.pt",
                "augmented_labels_model/copied.pt",
                "resolved_actions_original/keep.parquet",
                "action_labels_disc_0.9/keep.pt",
                "action_labels_intent_train_disc_0.9/keep.pt",
                "augmented_graphs/keep.pt",
                "augmented_labels/keep.pt",
            ]:
                path = base_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("copied", encoding="utf-8")
            (base_root / "metadata.json").write_text('{"status": "completed"}', encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            output_root = root / "derived"
            self.assertFalse((output_root / "resolved_actions_model").exists())
            self.assertFalse((output_root / "action_labels_disc_0.9_model").exists())
            self.assertFalse((output_root / "action_labels_intent_train_disc_0.9_model").exists())
            self.assertFalse((output_root / "augmented_graphs_model").exists())
            self.assertFalse((output_root / "augmented_labels_model").exists())
            self.assertTrue((output_root / "resolved_actions_original" / "keep.parquet").exists())
            self.assertTrue((output_root / "action_labels_disc_0.9" / "keep.pt").exists())
            self.assertTrue((output_root / "action_labels_intent_train_disc_0.9" / "keep.pt").exists())
            self.assertTrue((output_root / "augmented_graphs" / "keep.pt").exists())
            self.assertTrue((output_root / "augmented_labels" / "keep.pt").exists())

            output_metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["intended_receiver_model_id"], "success_intent/new")
            self.assertEqual(
                output_metadata["extension_replaced_intended_receiver_model_id"],
                "success_intent/old",
            )
            self.assertEqual(output_metadata["extension_replaced_intended_receiver_modes"], ["model"])
            self.assertEqual(run_command.call_count, 3)
            write_latest_run.assert_called_once_with("feature", "derived")

    def test_overwrite_feature_run_replacement_removes_base_model_mode_artifacts(self) -> None:
        args = make_args(
            model_id="success_intent/new",
            run_id=None,
            replace_model=True,
            overwrite_feature_run=True,
        )
        metadata = make_metadata(
            modes=["original", "angle_only", "model"],
            model_id="success_intent/old",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            for relative_path in [
                "resolved_actions_model/copied.parquet",
                "action_labels_disc_0.9_model/copied.pt",
                "action_labels_intent_train_disc_0.9_model/copied.pt",
                "augmented_graphs_model/copied.pt",
                "augmented_labels_model/copied.pt",
                "action_labels_disc_0.9/keep.pt",
            ]:
                path = base_root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("copied", encoding="utf-8")
            (base_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "copy_base_feature_run") as copy_base_feature_run,
                patch.object(generator, "run_command") as run_command,
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                generator.run_extension_generation(args)

            self.assertFalse((base_root / "resolved_actions_model").exists())
            self.assertFalse((base_root / "action_labels_disc_0.9_model").exists())
            self.assertFalse((base_root / "action_labels_intent_train_disc_0.9_model").exists())
            self.assertFalse((base_root / "augmented_graphs_model").exists())
            self.assertFalse((base_root / "augmented_labels_model").exists())
            self.assertTrue((base_root / "action_labels_disc_0.9" / "keep.pt").exists())
            output_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(output_metadata["status"], "completed")
            self.assertEqual(output_metadata["intended_receiver_model_id"], "success_intent/new")
            self.assertEqual(output_metadata["extension_replaced_intended_receiver_model_id"], "success_intent/old")
            copy_base_feature_run.assert_not_called()
            write_latest_run.assert_not_called()
            self.assertEqual(run_command.call_count, 3)

    def test_failed_extension_records_failure_without_updating_latest(self) -> None:
        args = make_args(["next_5"], run_id="derived")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "base").mkdir()

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=make_metadata()),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "run_command", side_effect=RuntimeError("boom")),
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                with self.assertRaises(RuntimeError):
                    generator.run_extension_generation(args)

            output_metadata = json.loads((root / "derived" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(output_metadata["status"], "failed")
            self.assertIn("boom", output_metadata["error"])
            write_latest_run.assert_not_called()

    def test_failed_overwrite_feature_run_marks_base_metadata_failed(self) -> None:
        args = make_args(run_id=None, overwrite_feature_run=True, refresh_target_families=["epv"])
        metadata = make_metadata()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_root = root / "base"
            base_root.mkdir()
            (base_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with (
                patch.object(generator, "resolve_feature_run_id", return_value="base"),
                patch.object(generator, "load_feature_run_metadata", return_value=metadata),
                patch.object(generator, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(generator, "copy_base_feature_run") as copy_base_feature_run,
                patch.object(generator, "run_command", side_effect=RuntimeError("boom")),
                patch.object(generator, "write_latest_run") as write_latest_run,
            ):
                with self.assertRaises(RuntimeError):
                    generator.run_extension_generation(args)

            output_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
            copy_base_feature_run.assert_not_called()
            write_latest_run.assert_not_called()
            self.assertEqual(output_metadata["status"], "failed")
            self.assertEqual(output_metadata["extension_history"][-1]["status"], "failed")
            self.assertIn("boom", output_metadata["error"])


if __name__ == "__main__":
    unittest.main()
