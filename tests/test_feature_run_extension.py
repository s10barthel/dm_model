from __future__ import annotations

import json
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
) -> dict[str, object]:
    return {
        "status": status,
        "graph_schema": graph_schema or generator.EXPECTED_GRAPH_SCHEMA.copy(),
        "return_types": return_types or ["disc_0.9"],
        "intended_receiver_modes": modes or ["original", "angle_only"],
        "intended_receiver_model_id": model_id,
    }


def make_args(
    requested_return_types: list[str] | None = None,
    model_id: str | None = None,
    run_id: str | None = "derived",
) -> SimpleNamespace:
    return SimpleNamespace(
        extend_feature_run_id="base",
        run_id=run_id,
        requested_return_types=(
            resolve_requested_return_types(requested_return_types) if requested_return_types is not None else []
        ),
        intended_receiver_model_id=model_id,
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

    def test_unions_requested_return_types_in_order(self) -> None:
        plan = self.build_plan(make_args(["next_5", "disc_0.9", "in_3"]))

        self.assertEqual(plan.final_return_types, ["disc_0.9", "next_5", "in_3"])
        self.assertEqual(plan.added_return_types, ["next_5", "in_3"])
        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only"])

    def test_model_mode_addition_updates_final_modes(self) -> None:
        plan = self.build_plan(make_args(["next_5"], model_id="success_intent/42"))

        self.assertEqual(plan.final_intended_receiver_modes, ["original", "angle_only", "model"])
        self.assertEqual(plan.added_intended_receiver_modes, ["model"])

    def test_noop_extension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["disc_0.9"]))

    def test_output_run_must_not_already_exist(self) -> None:
        with self.assertRaises(FileExistsError):
            self.build_plan(make_args(["next_5"]), existing_output=True)

    def test_output_run_must_not_equal_base_run(self) -> None:
        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"], run_id="base"))

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

    def test_graph_schema_mismatch_is_rejected(self) -> None:
        metadata = make_metadata(graph_schema={"edge_in_dim": 2, "add_v_edge_features": False})

        with self.assertRaises(ValueError):
            self.build_plan(make_args(["next_5"]), metadata=metadata)

    def test_extension_commands_are_labels_only_and_not_full_generation(self) -> None:
        plan = self.build_plan(make_args(["next_5"], model_id="success_intent/42"))

        self.assertEqual(len(plan.commands), 6)
        self.assertTrue(all("--labels-only" in command for command in plan.commands))
        self.assertTrue(all("--post_action" not in command for command in plan.commands))
        self.assertFalse(
            any(
                "--feature_variant" in command
                and command[command.index("--feature_variant") + 1] == "success_intent"
                for command in plan.commands
            )
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
            self.assertEqual(run_command.call_count, 3)
            write_latest_run.assert_called_once_with("feature", "derived")

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


if __name__ == "__main__":
    unittest.main()
