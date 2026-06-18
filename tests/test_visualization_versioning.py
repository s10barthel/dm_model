from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image

from scripts import run_and_visualize_hawkeye
from scripts import visualize_action_components
from scripts import visualize_benchmark
from scripts import visualize_hawkeye
from scripts import visualize_skillcorner


class FakeImage:
    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("image", encoding="utf-8")


def make_action_args(output_dir: Path) -> SimpleNamespace:
    args = SimpleNamespace(
        match_id="DFL-MAT-1",
        action_id=[123],
        row_index=None,
        original_event_id=None,
        first=None,
        player_id=None,
        object_id=None,
        advanced_position=None,
        team_id=None,
        spadl_type=["pass"],
        success=None,
        offside=None,
        next_type=None,
        device="cpu",
        bundle_id="bundle_1",
        feature_run_id=None,
        intended_receiver_mode=None,
        return_type=None,
        show_trajectories=False,
        show_physical_xpass=False,
        physical_cache_dir=None,
        action_intent_model_id=None,
        pass_intent_model_id=None,
        success_intent_model_id=None,
        pass_success_model_id=None,
        outcome_scoring_model_id=None,
        outcome_conceding_model_id=None,
        run_id="visualization_explicit",
        output_dir=str(output_dir),
    )
    for column in ("start_x", "start_y", "end_x", "end_y"):
        setattr(args, f"{column}_lt", None)
        setattr(args, f"{column}_gt", None)
    return args


def make_selection_args(**overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        match_id="DFL-MAT-1",
        action_id=None,
        row_index=None,
        original_event_id=None,
        first=None,
        player_id=None,
        object_id=None,
        advanced_position=None,
        team_id=None,
        spadl_type=None,
        success=None,
        offside=None,
        next_type=None,
    )
    for column in ("start_x", "start_y", "end_x", "end_y"):
        setattr(args, f"{column}_lt", None)
        setattr(args, f"{column}_gt", None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_selection_match() -> SimpleNamespace:
    events = pd.DataFrame(
        [
            {"action_id": 10, "spadl_type": "pass", "success": True, "offside": False, "original_event_id": "e10"},
            {"action_id": 11, "spadl_type": "pass", "success": True, "offside": False, "original_event_id": "e11"},
            {"action_id": 12, "spadl_type": "pass", "success": False, "offside": False, "original_event_id": "e12"},
            {"action_id": 13, "spadl_type": "pass", "success": True, "offside": False, "original_event_id": "e13"},
            {"action_id": 14, "spadl_type": "pass", "success": True, "offside": False, "original_event_id": "e14"},
        ],
        index=[0, 1, 2, 3, 4],
    )
    for column in ("start_x", "start_y", "end_x", "end_y"):
        events[column] = 0.0
    actions = pd.DataFrame({"action_id": [10, 12, 13, 14]}, index=[0, 2, 3, 4])
    return SimpleNamespace(match_id="DFL-MAT-1", events=events, actions=actions)


def success_intent_record(model_id: str = "success_intent/selected", feature_run_id: str = "feature_success") -> dict[str, object]:
    return {
        "model_id": model_id,
        "task": "success_intent",
        "feature_run_id": feature_run_id,
        "intended_receiver_mode": "unknown",
        "return_type": "next_5",
        "target_family": "goal",
        "graph_schema": {
            "node_in_dim": 23,
            "edge_in_dim": 4,
            "add_v_edge_features": True,
        },
    }


class VisualizationVersioningTests(unittest.TestCase):
    def test_resolve_action_indices_first_limits_eligible_modeled_events(self) -> None:
        match = make_selection_match()
        args = make_selection_args(first=2, spadl_type=["pass"])

        selected = visualize_action_components.resolve_action_indices(match, args)

        self.assertEqual(selected, [(0, "10"), (2, "12")])

    def test_resolve_action_indices_without_first_preserves_all_eligible_events(self) -> None:
        match = make_selection_match()
        args = make_selection_args(spadl_type=["pass"])

        selected = visualize_action_components.resolve_action_indices(match, args)

        self.assertEqual(selected, [(0, "10"), (2, "12"), (3, "13"), (4, "14")])

    def test_parse_args_rejects_invalid_first_values(self) -> None:
        with self.assertRaises(SystemExit):
            visualize_action_components.parse_args(["--match-id", "DFL-MAT-1", "--first", "0"])

        with self.assertRaises(SystemExit):
            visualize_action_components.parse_args(["--match-id", "DFL-MAT-1", "--first", "-1"])

    def test_parse_args_rejects_first_with_explicit_selectors(self) -> None:
        with self.assertRaises(SystemExit):
            visualize_action_components.parse_args(
                ["--match-id", "DFL-MAT-1", "--first", "10", "--action-id", "123"]
            )

    def test_action_visualization_writes_run_metadata_with_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = make_action_args(root)
            resolved_model_ids = {
                "action_intent": "action_intent/1",
                "pass_intent": "pass_intent/1",
                "pass_success": "pass_success/1",
                "outcome_scoring": "outcome_scoring/1",
                "outcome_conceding": "outcome_conceding/1",
            }
            runtime_context = {
                "feature_run_id": "feature_1",
                "intended_receiver_mode": "angle_only",
                "feature_root": root / "features",
                "selection": "newest_compatible",
            }

            def fake_render_action_components(**kwargs: object) -> None:
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "pass_score.png").write_text("plot", encoding="utf-8")

            with (
                patch.object(visualize_action_components, "parse_args", return_value=args),
                patch.object(visualize_action_components.torch.cuda, "is_available", return_value=False),
                patch.object(
                    visualize_action_components,
                    "resolve_model_selection",
                    return_value=(
                        resolved_model_ids,
                        {"return_type": "disc_0.9", "target_family": "goal"},
                        {"model_ids": {"success_intent": "success_intent/1"}},
                    ),
                ),
                patch.object(
                    visualize_action_components,
                    "load_model",
                    return_value=SimpleNamespace(args={}),
                ),
                patch.object(
                    visualize_action_components,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": False},
                ),
                patch.object(
                    visualize_action_components,
                    "resolve_runtime_feature_run_context",
                    return_value=runtime_context,
                ),
                patch.object(
                    visualize_action_components,
                    "get_model_record",
                    return_value=success_intent_record("success_intent/1", "feature_1"),
                ),
                patch.object(visualize_action_components, "resolve_runtime_return_type", return_value="disc_0.9"),
                patch.object(visualize_action_components, "load_match", return_value=SimpleNamespace()),
                patch.object(visualize_action_components, "resolve_action_indices", return_value=[(7, "123")]),
                patch.object(
                    visualize_action_components,
                    "render_action_components",
                    side_effect=fake_render_action_components,
                ),
            ):
                visualize_action_components.main()

            output_root = root / "visualization_explicit"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "DFL-MAT-1" / "123" / "pass_score.png").exists())
            self.assertEqual(metadata["run_id"], "visualization_explicit")
            self.assertEqual(metadata["model_ids"]["pass_success"], "pass_success/1")
            self.assertEqual(metadata["model_ids"]["intended_recipient"], "success_intent/1")
            self.assertEqual(metadata["feature_run_id"], "feature_1")
            self.assertEqual(metadata["filters"], {"spadl_type": ["pass"]})

    def test_action_visualization_only_pass_success_requires_only_pass_success_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = make_action_args(root)
            args.bundle_id = None
            args.only_pass_success = True
            args.pass_success_model_id = "pass_success/selected"
            runtime_context = {
                "feature_run_id": "feature_pass_success",
                "intended_receiver_mode": "angle_only",
                "feature_root": root / "features",
                "selection": "explicit",
            }

            def fake_resolve_model_selection(required_tasks: list[str], **_kwargs: object):
                self.assertEqual(required_tasks, ["pass_success"])
                return (
                    {"pass_success": "pass_success/selected"},
                    {"return_type": "disc_0.9", "target_family": "goal"},
                    None,
                )

            def fake_render_action_components(**kwargs: object) -> None:
                self.assertEqual(kwargs["rendered_components"], ["pass_success"])
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "pass_success.png").write_text("plot", encoding="utf-8")

            with (
                patch.object(visualize_action_components, "parse_args", return_value=args),
                patch.object(visualize_action_components.torch.cuda, "is_available", return_value=False),
                patch.object(
                    visualize_action_components,
                    "resolve_model_selection",
                    side_effect=fake_resolve_model_selection,
                ),
                patch.object(
                    visualize_action_components,
                    "load_model",
                    return_value=SimpleNamespace(args={}),
                ),
                patch.object(
                    visualize_action_components,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": False},
                ),
                patch.object(
                    visualize_action_components,
                    "resolve_runtime_feature_run_context",
                    return_value=runtime_context,
                ),
                patch.object(visualize_action_components, "resolve_runtime_return_type", return_value="disc_0.9"),
                patch.object(visualize_action_components, "load_match", return_value=SimpleNamespace()),
                patch.object(visualize_action_components, "resolve_action_indices", return_value=[(7, "123")]),
                patch.object(
                    visualize_action_components,
                    "render_action_components",
                    side_effect=fake_render_action_components,
                ),
            ):
                visualize_action_components.main()

            output_root = root / "visualization_explicit"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "DFL-MAT-1" / "123" / "pass_success.png").exists())
            self.assertEqual(metadata["model_ids"], {"pass_success": "pass_success/selected"})
            self.assertEqual(metadata["selected_model_ids"], {"pass_success": "pass_success/selected"})
            self.assertEqual(metadata["rendered_components"], ["pass_success"])

    def test_action_visualization_only_intended_recipient_uses_explicit_success_intent_without_core_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = make_action_args(root)
            args.bundle_id = None
            args.only_intended_recipient = True
            args.success_intent_model_id = "success_intent/selected"
            runtime_context = {
                "feature_run_id": "feature_success",
                "intended_receiver_mode": "angle_only",
                "feature_root": root / "features",
                "selection": "newest_compatible",
            }

            def fake_resolve_runtime_feature_run_context(
                _explicit_feature_run_id: object,
                shared_context: dict[str, object],
                bundle: dict[str, object] | None,
                *_args: object,
                **_kwargs: object,
            ) -> dict[str, object]:
                self.assertIsNone(bundle)
                self.assertEqual(
                    shared_context["source_feature_run_ids"],
                    {"intended_recipient": "feature_success"},
                )
                return runtime_context

            def fake_render_action_components(**kwargs: object) -> None:
                self.assertEqual(kwargs["rendered_components"], ["intended_recipient"])
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "intended_recipient.png").write_text("plot", encoding="utf-8")

            with (
                patch.object(visualize_action_components, "parse_args", return_value=args),
                patch.object(visualize_action_components.torch.cuda, "is_available", return_value=False),
                patch.object(visualize_action_components, "resolve_model_selection") as resolve_model_selection_mock,
                patch.object(
                    visualize_action_components,
                    "get_model_record",
                    return_value=success_intent_record("success_intent/selected", "feature_success"),
                ),
                patch.object(
                    visualize_action_components,
                    "load_model",
                    return_value=SimpleNamespace(args={}),
                ),
                patch.object(
                    visualize_action_components,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": True},
                ),
                patch.object(
                    visualize_action_components,
                    "resolve_runtime_feature_run_context",
                    side_effect=fake_resolve_runtime_feature_run_context,
                ),
                patch.object(visualize_action_components, "resolve_runtime_return_type", return_value="next_5"),
                patch.object(visualize_action_components, "load_match", return_value=SimpleNamespace()),
                patch.object(visualize_action_components, "resolve_action_indices", return_value=[(7, "123")]),
                patch.object(
                    visualize_action_components,
                    "render_action_components",
                    side_effect=fake_render_action_components,
                ),
            ):
                visualize_action_components.main()

            resolve_model_selection_mock.assert_not_called()
            metadata = json.loads((root / "visualization_explicit" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_ids"], {"intended_recipient": "success_intent/selected"})
            self.assertEqual(metadata["selected_model_ids"], {"intended_recipient": "success_intent/selected"})
            self.assertEqual(metadata["rendered_components"], ["intended_recipient"])

    def test_action_visualization_only_intended_recipient_keeps_bundle_context_with_explicit_success_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = make_action_args(root)
            args.only_intended_recipient = True
            args.success_intent_model_id = "success_intent/explicit"
            bundle = {
                "bundle_id": "bundle_1",
                "feature_run_id": "feature_bundle",
                "return_type": "disc_0.9",
                "model_ids": {"success_intent": "success_intent/bundle"},
                "source_feature_run_ids": {"pass_success": "feature_bundle"},
            }
            runtime_context = {
                "feature_run_id": "feature_bundle",
                "intended_receiver_mode": "model",
                "feature_root": root / "features",
                "selection": "newest_compatible",
            }

            def fake_resolve_runtime_feature_run_context(
                _explicit_feature_run_id: object,
                shared_context: dict[str, object],
                selected_bundle: dict[str, object] | None,
                *_args: object,
                **_kwargs: object,
            ) -> dict[str, object]:
                self.assertEqual(selected_bundle, bundle)
                self.assertEqual(shared_context["bundle_feature_run_id"], "feature_bundle")
                self.assertEqual(
                    shared_context["source_feature_run_ids"],
                    {"pass_success": "feature_bundle", "intended_recipient": "feature_success"},
                )
                return runtime_context

            def fake_render_action_components(**kwargs: object) -> None:
                self.assertEqual(kwargs["rendered_components"], ["intended_recipient"])
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "intended_recipient.png").write_text("plot", encoding="utf-8")

            with (
                patch.object(visualize_action_components, "parse_args", return_value=args),
                patch.object(visualize_action_components.torch.cuda, "is_available", return_value=False),
                patch.object(visualize_action_components, "resolve_model_selection") as resolve_model_selection_mock,
                patch.object(visualize_action_components, "load_bundle_record", return_value=bundle),
                patch.object(
                    visualize_action_components,
                    "get_model_record",
                    return_value=success_intent_record("success_intent/explicit", "feature_success"),
                ),
                patch.object(
                    visualize_action_components,
                    "load_model",
                    return_value=SimpleNamespace(args={}),
                ),
                patch.object(
                    visualize_action_components,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": True},
                ),
                patch.object(
                    visualize_action_components,
                    "resolve_runtime_feature_run_context",
                    side_effect=fake_resolve_runtime_feature_run_context,
                ),
                patch.object(visualize_action_components, "resolve_runtime_return_type", return_value="next_5"),
                patch.object(visualize_action_components, "load_match", return_value=SimpleNamespace()),
                patch.object(visualize_action_components, "resolve_action_indices", return_value=[(7, "123")]),
                patch.object(
                    visualize_action_components,
                    "render_action_components",
                    side_effect=fake_render_action_components,
                ),
            ):
                visualize_action_components.main()

            resolve_model_selection_mock.assert_not_called()
            metadata = json.loads((root / "visualization_explicit" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_ids"], {"intended_recipient": "success_intent/explicit"})
            self.assertEqual(metadata["selected_model_ids"], {"intended_recipient": "success_intent/explicit"})

    def test_action_visualization_only_intended_recipient_uses_bundle_success_intent_when_not_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = make_action_args(root)
            args.only_intended_recipient = True
            bundle = {
                "bundle_id": "bundle_1",
                "feature_run_id": "feature_bundle",
                "model_ids": {"success_intent": "success_intent/bundle"},
            }
            runtime_context = {
                "feature_run_id": "feature_bundle",
                "intended_receiver_mode": "model",
                "feature_root": root / "features",
                "selection": "newest_compatible",
            }

            def fake_render_action_components(**kwargs: object) -> None:
                self.assertEqual(kwargs["rendered_components"], ["intended_recipient"])
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "intended_recipient.png").write_text("plot", encoding="utf-8")

            with (
                patch.object(visualize_action_components, "parse_args", return_value=args),
                patch.object(visualize_action_components.torch.cuda, "is_available", return_value=False),
                patch.object(visualize_action_components, "resolve_model_selection") as resolve_model_selection_mock,
                patch.object(visualize_action_components, "load_bundle_record", return_value=bundle),
                patch.object(
                    visualize_action_components,
                    "get_model_record",
                    return_value=success_intent_record("success_intent/bundle", "feature_success"),
                ),
                patch.object(
                    visualize_action_components,
                    "load_model",
                    return_value=SimpleNamespace(args={}),
                ),
                patch.object(
                    visualize_action_components,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": True},
                ),
                patch.object(
                    visualize_action_components,
                    "resolve_runtime_feature_run_context",
                    return_value=runtime_context,
                ),
                patch.object(visualize_action_components, "resolve_runtime_return_type", return_value="next_5"),
                patch.object(visualize_action_components, "load_match", return_value=SimpleNamespace()),
                patch.object(visualize_action_components, "resolve_action_indices", return_value=[(7, "123")]),
                patch.object(
                    visualize_action_components,
                    "render_action_components",
                    side_effect=fake_render_action_components,
                ),
            ):
                visualize_action_components.main()

            resolve_model_selection_mock.assert_not_called()
            metadata = json.loads((root / "visualization_explicit" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["success_intent_model_id"], "success_intent/bundle")
            self.assertEqual(metadata["selected_model_ids"], {"intended_recipient": "success_intent/bundle"})

    def test_hawkeye_component_visualization_generates_run_id_and_records_component_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            component_root = root / "component"
            args = SimpleNamespace(
                situation_id=["sit1"],
                tracking_csv=str(root / "tracking.csv"),
                ball_csv=str(root / "ball.csv"),
                component_run_id=None,
                component_dir=None,
                show_trajectories=False,
                gif=False,
                run_id=None,
                output_dir=str(root),
            )

            def fake_save_animation(_images: object, output_path: str | Path, **_kwargs: object) -> None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("video", encoding="utf-8")

            with (
                patch.object(visualize_hawkeye, "parse_args", return_value=args),
                patch.object(visualize_hawkeye, "generate_run_id", return_value="hawkeye_visualization_generated"),
                patch.object(
                    visualize_hawkeye,
                    "resolve_named_component_run_id",
                    return_value="hawkeye_component_1",
                ),
                patch.object(visualize_hawkeye, "get_hawkeye_component_run_root", return_value=component_root),
                patch.object(
                    visualize_hawkeye,
                    "load_hawkeye_component_run",
                    return_value=(
                        pd.DataFrame({"id": ["sit1"]}),
                        {
                            "run_id": "hawkeye_component_1",
                            "freeze_ballreceipt": True,
                            "models": {"pass_success": "pass_success/1"},
                        },
                    ),
                ),
                patch.object(visualize_hawkeye, "resolve_hawkeye_component_situation_ids", return_value=["sit1"]),
                patch.object(visualize_hawkeye, "load_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(visualize_hawkeye, "clean_hawkeye_tracking", return_value=pd.DataFrame({"id": ["sit1"]})),
                patch.object(visualize_hawkeye, "load_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(visualize_hawkeye, "clean_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(
                    visualize_hawkeye,
                    "build_hawkeye_situation",
                    return_value=(SimpleNamespace(frame_meta=pd.DataFrame(index=[0])), None, None),
                ),
                patch.object(visualize_hawkeye, "build_hawkeye_component_tables", return_value={}),
                patch.object(visualize_hawkeye, "save_animation", side_effect=fake_save_animation),
            ):
                visualize_hawkeye.main()

            output_root = root / "hawkeye_visualization_generated"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "sit1" / "pass_score.mp4").exists())
            self.assertEqual(metadata["component_run_id"], "hawkeye_component_1")
            self.assertEqual(metadata["component_metadata_run_id"], "hawkeye_component_1")
            self.assertEqual(metadata["source_models"]["pass_success"], "pass_success/1")

    def test_hawkeye_component_visualization_only_flag_limits_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            component_root = root / "component"
            args = SimpleNamespace(
                situation_id=["sit1"],
                tracking_csv=str(root / "tracking.csv"),
                ball_csv=str(root / "ball.csv"),
                component_run_id="hawkeye_component_1",
                component_dir=None,
                show_trajectories=False,
                gif=False,
                only_pass_success=True,
                run_id="hawkeye_visualization_only",
                output_dir=str(root),
            )

            def fake_save_animation(_images: object, output_path: str | Path, **_kwargs: object) -> None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("video", encoding="utf-8")

            with (
                patch.object(visualize_hawkeye, "parse_args", return_value=args),
                patch.object(
                    visualize_hawkeye,
                    "resolve_named_component_run_id",
                    return_value="hawkeye_component_1",
                ),
                patch.object(visualize_hawkeye, "get_hawkeye_component_run_root", return_value=component_root),
                patch.object(
                    visualize_hawkeye,
                    "load_hawkeye_component_run",
                    return_value=(pd.DataFrame({"id": ["sit1"]}), {"run_id": "hawkeye_component_1"}),
                ),
                patch.object(visualize_hawkeye, "resolve_hawkeye_component_situation_ids", return_value=["sit1"]),
                patch.object(visualize_hawkeye, "load_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(visualize_hawkeye, "clean_hawkeye_tracking", return_value=pd.DataFrame({"id": ["sit1"]})),
                patch.object(visualize_hawkeye, "load_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(visualize_hawkeye, "clean_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(
                    visualize_hawkeye,
                    "build_hawkeye_situation",
                    return_value=(SimpleNamespace(frame_meta=pd.DataFrame(index=[0])), None, None),
                ),
                patch.object(visualize_hawkeye, "build_hawkeye_component_tables", return_value={}),
                patch.object(visualize_hawkeye, "save_animation", side_effect=fake_save_animation),
            ):
                visualize_hawkeye.main()

            output_root = root / "hawkeye_visualization_only"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "sit1" / "pass_success.mp4").exists())
            self.assertFalse((output_root / "sit1" / "pass_score.mp4").exists())
            self.assertEqual(metadata["requested_component_groups"], ["pass_success"])
            self.assertEqual(metadata["rendered_components"], ["pass_success"])
            self.assertIn("pass_score", metadata["disabled_components"])

    def test_benchmark_visualization_writes_paired_outputs_under_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = SimpleNamespace(
                input_dir=str(root / "benchmark"),
                modification=[1],
                game_state=[1, 2],
                component_run_id="benchmark_component_1",
                component_dir=None,
                run_id="benchmark_visualization_explicit",
                output_dir=str(root),
                show_trajectories=False,
            )

            with (
                patch.object(visualize_benchmark, "parse_args", return_value=args),
                patch.object(
                    visualize_benchmark,
                    "resolve_named_component_run_id",
                    return_value="benchmark_component_1",
                ),
                patch.object(visualize_benchmark, "get_benchmark_component_run_root", return_value=root / "component"),
                patch.object(
                    visualize_benchmark,
                    "load_benchmark_component_run",
                    return_value=(
                        pd.DataFrame({"modification": [1, 1], "game_state": [1, 2]}),
                        {"run_id": "benchmark_component_1", "models": {"pass_intent": "pass_intent/1"}},
                    ),
                ),
                patch.object(visualize_benchmark, "resolve_benchmark_component_states", return_value=[(1, 1), (1, 2)]),
                patch.object(
                    visualize_benchmark,
                    "load_benchmark_modification_data",
                    return_value={"game_state_1": pd.DataFrame(), "game_state_2": pd.DataFrame(), "higher_state_id": 2},
                ),
                patch.object(
                    visualize_benchmark,
                    "build_benchmark_state",
                    return_value=(SimpleNamespace(frame_meta=pd.DataFrame(index=[0])), None, None),
                ),
                patch.object(visualize_benchmark, "build_benchmark_component_tables", return_value={}),
                patch.object(visualize_benchmark, "_probs_for_component_frame", return_value=pd.Series(dtype=float)),
                patch.object(visualize_benchmark, "render_state_image", return_value=Image.new("RGB", (10, 8), "white")),
            ):
                visualize_benchmark.main()

            output_root = root / "benchmark_visualization_explicit"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            output_path = output_root / "modification_1" / "pass_score.png"
            self.assertTrue(output_path.exists())
            self.assertFalse((output_root / "modification_1_pass_score.png").exists())
            self.assertFalse((output_root / "modification_1" / "game_state_2" / "pass_score.png").exists())
            self.assertEqual(metadata["component_run_id"], "benchmark_component_1")
            rendered_modification = metadata["rendered_modifications"][0]
            self.assertEqual(rendered_modification["modification"], 1)
            self.assertEqual(rendered_modification["game_states"], [1, 2])
            self.assertEqual(rendered_modification["output_dir"], str(output_path.parent.resolve()))
            self.assertIn(str(output_path.resolve()), rendered_modification["output_paths"])

    def test_benchmark_visualization_combines_state_images_vertically(self) -> None:
        top = Image.new("RGB", (4, 3), "red")
        bottom = Image.new("RGB", (6, 2), "blue")

        combined = visualize_benchmark.combine_state_images(top, bottom)

        self.assertEqual(combined.size, (6, 5))
        self.assertEqual(combined.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(combined.getpixel((0, 3)), (0, 0, 255))

    def test_skillcorner_visualization_records_component_id_and_possession_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = SimpleNamespace(
                match_id="match1",
                index=[3],
                input_dir=str(root / "skillcorner"),
                component_run_id="skillcorner_component_1",
                component_dir=None,
                run_id="skillcorner_visualization_explicit",
                output_dir=str(root),
                show_trajectories=False,
                gif=True,
            )

            def fake_render_possession(
                args: SimpleNamespace,
                context: object,
                component_tables: dict[str, pd.DataFrame],
                possession_index: int,
                output_root: Path,
                rendered_components: list[str],
            ) -> Path:
                del context, component_tables, rendered_components
                output_dir = output_root / args.match_id / str(possession_index)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "pass_score.gif").write_text("gif", encoding="utf-8")
                return output_dir

            with (
                patch.object(visualize_skillcorner, "parse_args", return_value=args),
                patch.object(
                    visualize_skillcorner,
                    "resolve_named_component_run_id",
                    return_value="skillcorner_component_1",
                ),
                patch.object(visualize_skillcorner, "get_skillcorner_component_run_root", return_value=root / "component"),
                patch.object(visualize_skillcorner, "build_skillcorner_match_context", return_value={}),
                patch.object(visualize_skillcorner, "load_skillcorner_component_tables", return_value={}),
                patch.object(
                    visualize_skillcorner,
                    "load_run_metadata",
                    return_value={"run_id": "skillcorner_component_1", "models": {"pass_intent": "pass_intent/1"}},
                ),
                patch.object(visualize_skillcorner, "render_possession", side_effect=fake_render_possession),
            ):
                visualize_skillcorner.main()

            output_root = root / "skillcorner_visualization_explicit"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "match1" / "3" / "pass_score.gif").exists())
            self.assertEqual(metadata["component_run_id"], "skillcorner_component_1")
            self.assertEqual(metadata["rendered_possessions"][0]["index"], 3)
            self.assertEqual(metadata["source_models"]["pass_intent"], "pass_intent/1")

    def test_direct_hawkeye_visualization_records_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = SimpleNamespace(
                situation_id=["sit1"],
                action_id=None,
                tracking_csv=str(root / "tracking.csv"),
                ball_csv=str(root / "ball.csv"),
                freeze_ballreceipt=True,
                device="cpu",
                show_trajectories=False,
                gif=True,
                bundle_id=None,
                action_intent_model_id="action_intent/1",
                pass_intent_model_id="pass_intent/1",
                pass_success_model_id="pass_success/1",
                outcome_scoring_model_id="outcome_scoring/1",
                outcome_conceding_model_id="outcome_conceding/1",
                run_id="hawkeye_visualization_direct",
                output_dir=str(root),
            )

            def fake_render_situation(
                situation_id: str,
                tracking: pd.DataFrame,
                ball: pd.DataFrame,
                model_specs: dict[str, object],
                graph_schema: dict[str, object],
                args: SimpleNamespace,
                device: str,
                output_root: Path,
                rendered_components: list[str],
            ) -> Path:
                del tracking, ball, model_specs, graph_schema, args, device, rendered_components
                output_dir = output_root / situation_id
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "pass_score.gif").write_text("gif", encoding="utf-8")
                return output_dir, {}

            with (
                patch.object(run_and_visualize_hawkeye, "parse_args", return_value=args),
                patch.object(run_and_visualize_hawkeye.torch.cuda, "is_available", return_value=False),
                patch.object(run_and_visualize_hawkeye, "load_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "clean_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "load_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "clean_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(
                    run_and_visualize_hawkeye,
                    "resolve_model_selection",
                    return_value=(
                        {
                            "action_intent": "action_intent/1",
                            "pass_intent": "pass_intent/1",
                            "pass_success": "pass_success/1",
                            "outcome_scoring": "outcome_scoring/1",
                            "outcome_conceding": "outcome_conceding/1",
                        },
                        {},
                        None,
                    ),
                ),
                patch.object(run_and_visualize_hawkeye, "load_model", return_value=SimpleNamespace()),
                patch.object(
                    run_and_visualize_hawkeye,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": False},
                ),
                patch.object(run_and_visualize_hawkeye, "render_situation", side_effect=fake_render_situation),
            ):
                run_and_visualize_hawkeye.main()

            output_root = root / "hawkeye_visualization_direct"
            metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output_root / "sit1" / "pass_score.gif").exists())
            self.assertEqual(metadata["model_ids"]["pass_success"], "pass_success/1")
            self.assertEqual(metadata["rendered_situation_ids"], ["sit1"])

    def test_direct_hawkeye_only_outcome_scoring_loads_only_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = SimpleNamespace(
                situation_id=["sit1"],
                action_id=None,
                tracking_csv=str(root / "tracking.csv"),
                ball_csv=str(root / "ball.csv"),
                freeze_ballreceipt=True,
                device="cpu",
                show_trajectories=False,
                gif=False,
                bundle_id=None,
                action_intent_model_id="action_intent/1",
                pass_intent_model_id="pass_intent/1",
                pass_success_model_id="pass_success/1",
                outcome_scoring_model_id="outcome_scoring/1",
                outcome_conceding_model_id="outcome_conceding/1",
                only_outcome_scoring=True,
                run_id="hawkeye_visualization_only_scoring",
                output_dir=str(root),
            )
            loaded_ids: list[str] = []

            def fake_load_model(model_id: str, _device: str) -> SimpleNamespace:
                loaded_ids.append(model_id)
                return SimpleNamespace()

            def fake_resolve_model_selection(required_tasks: list[str], **_kwargs: object):
                self.assertEqual(required_tasks, ["outcome_scoring"])
                return ({"outcome_scoring": "outcome_scoring/1"}, {}, None)

            def fake_render_situation(
                situation_id: str,
                tracking: pd.DataFrame,
                ball: pd.DataFrame,
                model_specs: dict[str, object],
                graph_schema: dict[str, object],
                args: SimpleNamespace,
                device: str,
                output_root: Path,
                rendered_components: list[str],
            ) -> Path:
                del tracking, ball, graph_schema, args, device
                self.assertEqual(list(model_specs), ["outcome_scoring"])
                self.assertEqual(rendered_components, ["outcome_scoring_success", "outcome_scoring_failure"])
                output_dir = output_root / situation_id
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "outcome_scoring_success.mp4").write_text("video", encoding="utf-8")
                return output_dir, {}

            with (
                patch.object(run_and_visualize_hawkeye, "parse_args", return_value=args),
                patch.object(run_and_visualize_hawkeye.torch.cuda, "is_available", return_value=False),
                patch.object(run_and_visualize_hawkeye, "load_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "clean_hawkeye_tracking", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "load_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(run_and_visualize_hawkeye, "clean_hawkeye_ball", return_value=pd.DataFrame()),
                patch.object(
                    run_and_visualize_hawkeye,
                    "resolve_model_selection",
                    side_effect=fake_resolve_model_selection,
                ),
                patch.object(run_and_visualize_hawkeye, "load_model", side_effect=fake_load_model),
                patch.object(
                    run_and_visualize_hawkeye,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": False},
                ),
                patch.object(run_and_visualize_hawkeye, "render_situation", side_effect=fake_render_situation),
            ):
                run_and_visualize_hawkeye.main()

            metadata = json.loads((root / "hawkeye_visualization_only_scoring" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded_ids, ["outcome_scoring/1"])
            self.assertEqual(metadata["selected_model_ids"], {"outcome_scoring": "outcome_scoring/1"})
            self.assertEqual(metadata["rendered_components"], ["outcome_scoring_success", "outcome_scoring_failure"])

    def test_only_pass_score_without_dependencies_raises_clear_error(self) -> None:
        args = SimpleNamespace(
            situation_id=["sit1"],
            action_id=None,
            tracking_csv="tracking.csv",
            ball_csv="ball.csv",
            freeze_ballreceipt=True,
            device="cpu",
            show_trajectories=False,
            gif=False,
            action_intent_model_id="action_intent/1",
            pass_intent_model_id="pass_intent/1",
            pass_success_model_id="pass_success/1",
            outcome_scoring_model_id="outcome_scoring/1",
            outcome_conceding_model_id="outcome_conceding/1",
            only_pass_score=True,
            run_id="bad",
            output_dir="out",
        )
        with patch.object(run_and_visualize_hawkeye, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "pass_score visualization requires"):
                run_and_visualize_hawkeye.main()

    def test_visualization_scripts_do_not_write_latest_pointers(self) -> None:
        script_paths = [
            Path(visualize_action_components.__file__),
            Path(visualize_hawkeye.__file__),
            Path(visualize_benchmark.__file__),
            Path(visualize_skillcorner.__file__),
            Path(run_and_visualize_hawkeye.__file__),
        ]
        for path in script_paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("write_latest_run", source)
            self.assertNotIn("latest.json", source)


if __name__ == "__main__":
    unittest.main()
