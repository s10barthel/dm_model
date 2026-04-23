from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data

from dataset import ActionDataset
from datatools.config import LABEL_COLUMNS
from datatools.utils import filter_features_and_labels
from models import utils as model_utils
from models.gnn import GNN
from scripts import run_benchmark
from scripts import train_relevant_models as train_wrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datatools.viz_helpers import figure_to_rgb_image
from scripts import visualize_benchmark
from scripts import visualize_hawkeye


def make_graph(node_dim: int = 25, edge_dim: int = 2) -> Data:
    x = torch.zeros((2, node_dim), dtype=torch.float32)
    x[:, 0] = 1
    x[0, 13] = 1
    x[:, 7] = 7
    x[:, 8] = 8
    x[:, 9] = 9
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones((edge_index.shape[1], edge_dim), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_labels() -> torch.Tensor:
    return torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)


def make_model_args(*, accel_aware: bool | None = None) -> dict[str, object]:
    args: dict[str, object] = {
        "model": "gat",
        "task": "action_intent",
        "gnn_task": "node_selection",
        "node_in_dim": 25,
        "edge_in_dim": 2,
        "node_emb_dim": 8,
        "graph_emb_dim": 8,
        "gnn_layers": 1,
        "gnn_heads": 1,
        "skip_conn": False,
        "out_dim": 1,
        "include_out": False,
    }
    if accel_aware is not None:
        args["accel_aware"] = accel_aware
    return args


def write_checkpoint(checkpoint_dir: Path, *, accel_aware: bool | None = None) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args = make_model_args(accel_aware=accel_aware)
    (checkpoint_dir / "args.json").write_text(json.dumps(args), encoding="utf-8")
    model = GNN(args)
    torch.save(model.state_dict(), checkpoint_dir / "best_weights.pt")


def make_enabled_tasks(**overrides: bool) -> dict[str, bool]:
    enabled = {
        "action_intent": True,
        "pass_intent": True,
        "success_intent": True,
        "pass_success": True,
        "outcome_scoring": True,
        "outcome_conceding": True,
        "failure_receiver": False,
    }
    enabled.update(overrides)
    return enabled


def make_bundle_shared_context(
    *,
    feature_run_id: str = "feature_run",
    intended_receiver_mode: str | None = "angle_only",
    return_type: str = "disc_0.9",
    target_family: str | None = "goal",
    edge_in_dim: int = 4,
    add_v_edge_features: bool = True,
) -> dict[str, object]:
    return {
        "feature_run_id": feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": target_family,
        "graph_schema": {
            "edge_in_dim": edge_in_dim,
            "add_v_edge_features": add_v_edge_features,
        },
        "use_v_edge_features": add_v_edge_features,
    }


def make_model_record(
    task: str,
    *,
    feature_run_id: str = "feature_run",
    intended_receiver_mode: str | None = "angle_only",
    return_type: str = "disc_0.9",
    target_family: str | None = None,
    edge_in_dim: int = 4,
    add_v_edge_features: bool = True,
) -> dict[str, object]:
    return {
        "task": task,
        "feature_run_id": feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": target_family,
        "graph_schema": {
            "edge_in_dim": edge_in_dim,
            "add_v_edge_features": add_v_edge_features,
        },
    }


class BenchmarkNoAccelTests(unittest.TestCase):
    def test_benchmark_identifier_columns_export_as_nullable_integers(self) -> None:
        table = pd.DataFrame(
            [
                {"team": 1.0, "player": 10.0, "event_player": 10.0, "pos_x": 5.5},
                {"team": pd.NA, "player": pd.NA, "event_player": pd.NA, "pos_x": 6.5},
            ]
        )

        normalized = run_benchmark._coerce_benchmark_identifier_columns(table)

        for column in ["team", "player", "event_player"]:
            self.assertEqual(str(normalized[column].dtype), "Int64")
            self.assertEqual(int(normalized.at[0, column]), 10 if column != "team" else 1)
            self.assertTrue(pd.isna(normalized.at[1, column]))

        csv_export = normalized.to_csv(index=False)
        self.assertIn("1,10,10", csv_export)
        self.assertNotIn("1.0", csv_export)
        self.assertNotIn("10.0", csv_export)

    def test_figure_to_rgb_image_tight_false_keeps_canvas_size_stable_when_text_extents_change(self) -> None:
        fig_short, ax_short = plt.subplots(figsize=(4, 3))
        ax_short.plot([0.0, 1.0], [0.0, 1.0])
        ax_short.text(
            0.5,
            1.01,
            "Short title",
            transform=ax_short.transAxes,
            ha="center",
            va="bottom",
        )

        fig_long, ax_long = plt.subplots(figsize=(4, 3))
        ax_long.plot([0.0, 1.0], [0.0, 1.0])
        ax_long.text(
            0.5,
            1.01,
            "A much longer title that would normally expand a tight bounding box",
            transform=ax_long.transAxes,
            ha="center",
            va="bottom",
        )

        try:
            short_image = figure_to_rgb_image(fig_short, dpi=100, tight=False)
            long_image = figure_to_rgb_image(fig_long, dpi=100, tight=False)
        finally:
            plt.close(fig_short)
            plt.close(fig_long)

        self.assertEqual(short_image.size, long_image.size)

    def test_wrapper_parse_args_accepts_no_accel(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(["--feature-run-id", "feature_run", "--success-intent-only", "--no-accel"])

        self.assertFalse(args.accel_aware)

    def test_build_training_commands_emit_no_accel_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = SimpleNamespace(
                feature_run_id="feature_run",
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
                return_type="disc_0.9",
                use_v_edge_features=True,
                outcome_scoring_trial=None,
                outcome_conceding_trial=None,
                xy_only=None,
                possessor_aware=None,
                keeper_aware=None,
                ball_z_aware=None,
                poss_vel_aware=None,
                accel_aware=False,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertFalse(feature_flags["accel_aware"])
        self.assertIn("--no-accel", commands[0])
        self.assertNotIn("--accel", commands[0])

    def test_wrapper_main_records_accel_flag_in_bundle_metadata(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            success_intent_only=False,
            target_family="goal",
            return_type="disc_0.9",
        )

        def capture_write_run_metadata(_root: Path, payload: dict[str, object]) -> None:
            captured_metadata.update(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "bundle_under_test"
            with (
                patch.object(train_wrapper, "parse_args", return_value=cli_args),
                patch.object(
                    train_wrapper,
                    "build_training_commands",
                    return_value=(
                        [["--task", "pass_intent", "--no-accel"]],
                        {"pass_intent": "pass_intent/test"},
                        "angle_only",
                        "feature_run",
                        {
                            "xy_only": False,
                            "possessor_aware": True,
                            "keeper_aware": True,
                            "ball_z_aware": True,
                            "poss_vel_aware": True,
                            "accel_aware": False,
                            "extend_features": False,
                        },
                    ),
                ),
                patch.object(train_wrapper, "get_model_bundle_root", return_value=bundle_root),
                patch.object(train_wrapper, "load_model_bundle_metadata", return_value={}),
                patch.object(
                    train_wrapper,
                    "derive_bundle_shared_context",
                    return_value=make_bundle_shared_context(target_family=None),
                ),
                patch.object(train_wrapper, "load_feature_run_metadata", return_value={}),
                patch.object(train_wrapper, "write_run_metadata", side_effect=capture_write_run_metadata),
                patch.object(train_wrapper.subprocess, "run"),
            ):
                train_wrapper.main()

        self.assertIn("training_feature_flags", captured_metadata)
        self.assertFalse(captured_metadata["training_feature_flags"]["accel_aware"])
        self.assertEqual(captured_metadata["model_ids"], {"pass_intent": "pass_intent/test"})
        self.assertEqual(captured_metadata["trained_tasks"], ["pass_intent"])

    def test_wrapper_main_merges_existing_bundle_metadata_for_partial_outcome_rerun(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9", "in_3"],
            use_v_edge_features=True,
            success_intent_only=False,
            target_family="xt",
            return_type="in_3",
        )
        existing_bundle = {
            "created_at": "2026-04-20T10:00:00",
            "success_intent_label_source": "receiver_id",
            "success_intent_training_filter": "successful_pass_actions",
            "model_ids": {
                "action_intent": "action_intent/old",
                "pass_intent": "pass_intent/old",
                "pass_success": "pass_success/old",
                "outcome_scoring": "outcome_scoring/old",
                "outcome_conceding": "outcome_conceding/old",
            },
        }

        def capture_write_run_metadata(_root: Path, payload: dict[str, object]) -> None:
            captured_metadata.update(payload)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "bundle_under_test"
            with (
                patch.object(train_wrapper, "parse_args", return_value=cli_args),
                patch.object(
                    train_wrapper,
                    "build_training_commands",
                    return_value=(
                        [
                            ["--task", "outcome_scoring"],
                            ["--task", "outcome_conceding"],
                        ],
                        {
                            "outcome_scoring": "outcome_scoring/new",
                            "outcome_conceding": "outcome_conceding/new",
                        },
                        "angle_only",
                        "feature_run",
                        {
                            "xy_only": False,
                            "possessor_aware": True,
                            "keeper_aware": True,
                            "ball_z_aware": True,
                            "poss_vel_aware": True,
                            "accel_aware": True,
                            "extend_features": False,
                        },
                    ),
                ),
                patch.object(train_wrapper, "get_model_bundle_root", return_value=bundle_root),
                patch.object(train_wrapper, "load_model_bundle_metadata", return_value=existing_bundle),
                patch.object(
                    train_wrapper,
                    "derive_bundle_shared_context",
                    return_value=make_bundle_shared_context(return_type="in_3", target_family="xt"),
                ),
                patch.object(
                    train_wrapper,
                    "load_feature_run_metadata",
                    return_value={
                        "return_types": ["disc_0.9", "in_3"],
                        "intended_receiver_modes": ["original", "angle_only"],
                    },
                ),
                patch.object(train_wrapper, "write_run_metadata", side_effect=capture_write_run_metadata),
                patch.object(train_wrapper.subprocess, "run"),
            ):
                train_wrapper.main()

        self.assertEqual(captured_metadata["created_at"], "2026-04-20T10:00:00")
        self.assertIn("updated_at", captured_metadata)
        self.assertEqual(
            captured_metadata["model_ids"],
            {
                "action_intent": "action_intent/old",
                "pass_intent": "pass_intent/old",
                "pass_success": "pass_success/old",
                "outcome_scoring": "outcome_scoring/new",
                "outcome_conceding": "outcome_conceding/new",
            },
        )
        self.assertEqual(captured_metadata["trained_tasks"], ["outcome_scoring", "outcome_conceding"])
        self.assertEqual(captured_metadata["return_type"], "in_3")
        self.assertEqual(captured_metadata["target_family"], "xt")
        self.assertEqual(captured_metadata["success_intent_label_source"], "receiver_id")
        self.assertEqual(captured_metadata["success_intent_training_filter"], "successful_pass_actions")
        self.assertEqual(len(captured_metadata["commands"]), 2)

    def test_validate_model_record_consistency_uses_outcome_return_types_for_shared_context(self) -> None:
        model_records = {
            "action_intent": make_model_record("action_intent", return_type="next_5"),
            "pass_intent": make_model_record("pass_intent", return_type="next_3"),
            "pass_success": make_model_record("pass_success", return_type="disc_0.9"),
            "outcome_scoring": make_model_record(
                "outcome_scoring",
                return_type="in_3",
                target_family="xt",
            ),
            "outcome_conceding": make_model_record(
                "outcome_conceding",
                return_type="in_3",
                target_family="xt",
            ),
        }

        shared = model_utils.validate_model_record_consistency(model_records)

        self.assertEqual(shared["feature_run_id"], "feature_run")
        self.assertEqual(shared["intended_receiver_mode"], "angle_only")
        self.assertEqual(shared["return_type"], "in_3")
        self.assertEqual(shared["target_family"], "xt")

    def test_resolve_model_selection_uses_outcome_return_type_for_mixed_bundle(self) -> None:
        required_tasks = [
            "action_intent",
            "pass_intent",
            "pass_success",
            "outcome_scoring",
            "outcome_conceding",
        ]
        resolved_model_ids = {
            "action_intent": "action_intent/old",
            "pass_intent": "pass_intent/old",
            "pass_success": "pass_success/old",
            "outcome_scoring": "outcome_scoring/new",
            "outcome_conceding": "outcome_conceding/new",
        }
        bundle = {
            "bundle_id": "bundle_under_test",
            "feature_run_id": "feature_run",
            "intended_receiver_mode": "angle_only",
            "return_type": "in_3",
            "target_family": "xt",
            "model_ids": resolved_model_ids,
        }
        model_records = {
            "action_intent": make_model_record("action_intent", return_type="next_5"),
            "pass_intent": make_model_record("pass_intent", return_type="disc_0.9"),
            "pass_success": make_model_record("pass_success", return_type="next_3"),
            "outcome_scoring": make_model_record(
                "outcome_scoring",
                return_type="in_3",
                target_family="xt",
            ),
            "outcome_conceding": make_model_record(
                "outcome_conceding",
                return_type="in_3",
                target_family="xt",
            ),
        }

        with (
            patch.object(model_utils, "resolve_bundle_model_ids", return_value=(resolved_model_ids, bundle)),
            patch.object(model_utils, "get_model_records", return_value=model_records),
        ):
            selected_model_ids, shared, selected_bundle = model_utils.resolve_model_selection(
                required_tasks=required_tasks,
                bundle_id="bundle_under_test",
            )

        self.assertEqual(selected_model_ids, resolved_model_ids)
        self.assertEqual(shared["return_type"], "in_3")
        self.assertEqual(shared["target_family"], "xt")
        self.assertEqual(shared["feature_run_id"], "feature_run")
        self.assertEqual(selected_bundle, bundle)

    def test_resolve_bundle_model_ids_still_rejects_missing_required_tasks(self) -> None:
        with patch.object(
            model_utils,
            "load_bundle_record",
            return_value={"bundle_id": "bundle_under_test", "model_ids": {"action_intent": "action_intent/01"}},
        ):
            with self.assertRaises(ValueError):
                model_utils.resolve_bundle_model_ids(
                    "bundle_under_test",
                    required_tasks=["action_intent", "pass_success"],
                )

    def test_filter_features_and_labels_zeroes_only_accel_column(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": False,
            "extend_features": True,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
        }

        filtered_graphs, _ = filter_features_and_labels([make_graph()], labels, args)
        graph = filtered_graphs[0]

        self.assertTrue(torch.equal(graph.x[:, 8], torch.zeros_like(graph.x[:, 8])))
        self.assertTrue(torch.equal(graph.x[:, 7], torch.full_like(graph.x[:, 7], 7)))
        self.assertTrue(torch.equal(graph.x[:, 9], torch.full_like(graph.x[:, 9], 9)))

    def test_action_dataset_zeroes_only_accel_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_graph()], feature_dir / "match_1.pt")
            torch.save(make_labels(), label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="action_intent",
                accel_aware=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[:, 8], torch.zeros_like(graph.x[:, 8])))
        self.assertTrue(torch.equal(graph.x[:, 7], torch.full_like(graph.x[:, 7], 7)))
        self.assertTrue(torch.equal(graph.x[:, 9], torch.full_like(graph.x[:, 9], 9)))

    def test_get_model_record_defaults_legacy_checkpoint_to_accel_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "action_intent" / "legacy"
            write_checkpoint(checkpoint_dir, accel_aware=None)

            with patch.object(model_utils, "get_model_path", return_value=checkpoint_dir):
                record = model_utils.get_model_record("action_intent/legacy")

        self.assertTrue(record["feature_signature"]["accel_aware"])
        self.assertTrue(record["args"]["accel_aware"])

    def test_load_model_preserves_explicit_no_accel_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "action_intent" / "no_accel"
            write_checkpoint(checkpoint_dir, accel_aware=False)

            with patch.object(model_utils, "get_model_path", return_value=checkpoint_dir):
                model = model_utils.load_model("action_intent/no_accel", device="cpu")

        self.assertFalse(model.args["accel_aware"])

    def test_render_state_image_does_not_pass_ball_velocity_xy(self) -> None:
        tracking = pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "ball_x": 52.5,
                    "ball_y": 34.0,
                    "ball_vx": 3.0,
                    "ball_vy": 1.0,
                    "home_10_x": 50.0,
                    "home_10_y": 30.0,
                    "home_10_vx": 0.5,
                    "home_10_vy": 0.1,
                }
            ]
        ).set_index("frame_id", drop=False)
        frame_meta = pd.DataFrame([{"frame_id": 0, "possessor_object_id": "home_10"}]).set_index("frame_id")
        state = SimpleNamespace(
            modification_id=1,
            game_state_id=2,
            tracking=tracking,
            frame_meta=frame_meta,
        )

        with patch.object(visualize_benchmark, "SnapshotVisualizer") as mock_visualizer:
            fig, ax = plt.subplots()
            mock_visualizer.return_value.plot.return_value = (fig, ax)
            image = visualize_benchmark.render_state_image(state, "pass_success", pd.Series({"home_10": 0.7}))

        self.assertIsInstance(image, Image.Image)
        self.assertNotIn("ball_velocity_xy", mock_visualizer.call_args.kwargs)

    def test_hawkeye_render_frame_image_uses_fixed_canvas_export(self) -> None:
        tracking = pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "ball_x": 52.5,
                    "ball_y": 34.0,
                    "home_10_x": 50.0,
                    "home_10_y": 30.0,
                    "home_10_vx": 0.5,
                    "home_10_vy": 0.1,
                }
            ]
        ).set_index("frame_id", drop=False)
        frame_meta = pd.DataFrame(
            [
                {
                    "frame_id": 0,
                    "abs_time": 12.345,
                    "possession_prefix": "home",
                    "possessor_object_id": "home_10",
                }
            ]
        ).set_index("frame_id")
        situation = SimpleNamespace(
            situation_id="example-situation",
            tracking=tracking,
            frame_meta=frame_meta,
        )
        expected_image = Image.new("RGB", (16, 16))

        with (
            patch.object(visualize_hawkeye, "SnapshotVisualizer") as mock_visualizer,
            patch.object(visualize_hawkeye, "figure_to_rgb_image", return_value=expected_image) as mock_figure_to_rgb,
        ):
            fig, ax = plt.subplots()
            mock_visualizer.return_value.plot.return_value = (fig, ax)
            image = visualize_hawkeye.render_frame_image(
                situation,
                0,
                "pass_success",
                pd.Series({"home_10": 0.7}),
            )

        self.assertIs(image, expected_image)
        self.assertEqual(mock_figure_to_rgb.call_args.kwargs["dpi"], 150)
        self.assertFalse(mock_figure_to_rgb.call_args.kwargs["tight"])


if __name__ == "__main__":
    unittest.main()
