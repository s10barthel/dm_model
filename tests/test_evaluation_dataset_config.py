from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch_geometric.data import Data

import test as evaluation_script
from dataset import ActionDataset
from models import dataset_config
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.utils import calc_weighted_binary_probability_metrics
from physical_pass_model import pc_xpass_lane_survival_metadata_fingerprint, physical_xpass_blend_weight_v4
from scripts import evaluate_relevant_models


class EvaluationDatasetConfigTests(unittest.TestCase):
    def test_builder_covers_every_action_dataset_option(self) -> None:
        dataset_parameters = set(inspect.signature(ActionDataset.__init__).parameters)
        dataset_parameters.difference_update({"self", "match_ids", "feature_dir", "label_dir"})

        kwargs = build_action_dataset_kwargs(
            SimpleNamespace(task="pass_success", edge_in_dim=2),
            train=False,
            diagnostic_label_dir=None,
        )

        self.assertEqual(set(kwargs), dataset_parameters)

    def test_builder_reconstructs_all_feature_options(self) -> None:
        args = SimpleNamespace(
            task="pass_success",
            include_out=False,
            min_pass_dur=0.25,
            shot_success="goal",
            xy_only=True,
            possessor_aware=True,
            keeper_aware=True,
            ball_z_aware=True,
            poss_vel_aware=True,
            poss_rel_vel_aware=True,
            poss_geometry_aware=False,
            goal_features_aware=False,
            goal_nodes_aware=False,
            accel_aware=False,
            offside_aware=False,
            extend_features=True,
            filter_blockers=True,
            sparsify="distance",
            max_edge_dist=17.5,
            edge_in_dim=5,
            v_edge_feature_mode="no_poss",
            relative_speed_edge_feature_mode="no_poss",
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset",
            physical_eps=0.002,
            physical_xpass_floor=0.1,
            lane_survival=True,
            lane_survival_mode="top_10",
        )

        kwargs = build_action_dataset_kwargs(
            args,
            train=False,
            diagnostic_label_dir="diagnostics",
            physical_cache_dir="physical",
            lane_survival_cache_dir="lane",
        )

        self.assertFalse(kwargs["train"])
        self.assertTrue(kwargs["poss_rel_vel_aware"])
        self.assertFalse(kwargs["poss_geometry_aware"])
        self.assertFalse(kwargs["goal_features_aware"])
        self.assertFalse(kwargs["goal_nodes_aware"])
        self.assertFalse(kwargs["accel_aware"])
        self.assertFalse(kwargs["offside_aware"])
        self.assertEqual(kwargs["v_edge_feature_mode"], "no_poss")
        self.assertEqual(kwargs["relative_speed_edge_feature_mode"], "no_poss")
        self.assertTrue(kwargs["mask_possessor_v_edge_features"])
        self.assertTrue(kwargs["mask_possessor_relative_speed_edge_features"])
        self.assertTrue(kwargs["use_physical_xpass"])
        self.assertEqual(kwargs["physical_cache_dir"], "physical")
        self.assertEqual(kwargs["physical_eps"], 0.002)
        self.assertEqual(kwargs["physical_xpass_floor"], 0.1)
        self.assertTrue(kwargs["lane_survival"])
        self.assertEqual(kwargs["lane_survival_mode"], "top_10")
        self.assertEqual(kwargs["lane_survival_cache_dir"], "lane")

    def test_builder_applies_legacy_defaults(self) -> None:
        kwargs = build_action_dataset_kwargs(
            {"task": "outcome_scoring", "edge_in_dim": 2},
            train=False,
            diagnostic_label_dir="diagnostics",
        )

        self.assertFalse(kwargs["poss_rel_vel_aware"])
        self.assertTrue(kwargs["poss_geometry_aware"])
        self.assertTrue(kwargs["goal_features_aware"])
        self.assertTrue(kwargs["goal_nodes_aware"])
        self.assertTrue(kwargs["accel_aware"])
        self.assertTrue(kwargs["offside_aware"])
        self.assertEqual(kwargs["v_edge_feature_mode"], "none")
        self.assertEqual(kwargs["relative_speed_edge_feature_mode"], "none")
        self.assertTrue(kwargs["require_goal_next10_diagnostics"])

    def test_builder_rejects_relative_speed_without_velocity_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "Relative-speed edge features require velocity-angle"):
            build_action_dataset_kwargs(
                {
                    "task": "pass_success",
                    "edge_in_dim": 5,
                    "v_edge_feature_mode": "none",
                    "relative_speed_edge_feature_mode": "all",
                },
                train=False,
                diagnostic_label_dir=None,
            )

    def test_lane_survival_context_validates_checkpoint_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = {"source": "pc_xpass", "available_metrics": ["max_xpass"]}
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = SimpleNamespace(
                lane_survival=True,
                lane_survival_mode="max",
                lane_survival_cache_fingerprint=pc_xpass_lane_survival_metadata_fingerprint(metadata),
            )

            with patch.object(evaluation_script, "get_pc_xpass_dir", return_value=cache_dir):
                resolved = evaluation_script.resolve_lane_survival_context(args)

            self.assertEqual(resolved, str(cache_dir))
            self.assertEqual(args.lane_survival_mode, "max")

    def test_physical_xpass_context_uses_and_validates_recorded_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded_cache = Path(tmpdir) / "recorded"
            canonical_cache = Path(tmpdir) / "canonical"
            recorded_cache.mkdir()
            canonical_cache.mkdir()
            args = SimpleNamespace(
                task="pass_success",
                use_physical_xpass=True,
                model_variant="gat_phys_logit_offset",
                physical_cache_dir=str(recorded_cache),
                physical_eps=1e-4,
                physical_xpass_source="source",
                physical_xpass_speed_aggregation="package_max",
                physical_xpass_teammate_policy="ignore_teammates",
            )
            metadata = {
                "source": "source",
                "speed_aggregation": "package_max",
                "teammate_policy": "ignore_teammates",
            }

            with (
                patch.object(evaluation_script, "get_physical_xpass_dir", return_value=canonical_cache),
                patch.object(evaluation_script, "validate_physical_xpass_args") as validate_args,
                patch.object(
                    evaluation_script,
                    "validate_physical_xpass_cache_metadata",
                    return_value=metadata,
                ) as validate_cache,
            ):
                resolved = evaluation_script.resolve_physical_xpass_context(args, Path(tmpdir))

            self.assertEqual(resolved, str(recorded_cache))
            validate_args.assert_called_once_with(args)
            validate_cache.assert_called_once_with(
                recorded_cache,
                expected_source="source",
                expected_speed_aggregation="package_max",
            )

    def test_physical_xpass_context_can_remap_to_overridden_feature_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorded_cache = Path(tmpdir) / "recorded"
            canonical_cache = Path(tmpdir) / "canonical"
            recorded_cache.mkdir()
            canonical_cache.mkdir()
            args = SimpleNamespace(
                task="pass_success",
                use_physical_xpass=True,
                model_variant="gat_phys_logit_offset",
                physical_cache_dir=str(recorded_cache),
                physical_eps=1e-4,
            )
            metadata = {
                "source": "accessible_space_max_player_cum_prob_as_defaults",
                "speed_aggregation": "package_max",
            }

            with (
                patch.object(evaluation_script, "get_physical_xpass_dir", return_value=canonical_cache),
                patch.object(evaluation_script, "validate_physical_xpass_args"),
                patch.object(
                    evaluation_script,
                    "validate_physical_xpass_cache_metadata",
                    return_value=metadata,
                ),
            ):
                resolved = evaluation_script.resolve_physical_xpass_context(
                    args,
                    Path(tmpdir),
                    prefer_feature_root=True,
                )

            self.assertEqual(resolved, str(canonical_cache))

    def test_lane_survival_context_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = {"source": "pc_xpass", "available_metrics": ["max_xpass"]}
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = SimpleNamespace(
                lane_survival=True,
                lane_survival_mode="max",
                lane_survival_cache_fingerprint="different",
            )

            with patch.object(evaluation_script, "get_pc_xpass_dir", return_value=cache_dir):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    evaluation_script.resolve_lane_survival_context(args)

    def test_ipw_dataset_uses_lane_survival_checkpoint_features_only(self) -> None:
        target_kwargs = build_action_dataset_kwargs(
            {"task": "pass_success", "edge_in_dim": 5, "lane_survival": False},
            train=True,
            diagnostic_label_dir=None,
        )
        checkpoint_args = {
            "task": "pass_intent",
            "edge_in_dim": 5,
            "v_edge_feature_mode": "no_poss",
            "relative_speed_edge_feature_mode": "no_poss",
            "lane_survival": True,
            "lane_survival_mode": "max",
        }
        lane_metadata = {"enabled": True, "mode": "max", "cache_dir": "lane-cache", "cache_fingerprint": "fingerprint"}

        with (
            patch.object(dataset_config, "validate_physical_xpass_cache_metadata", return_value={"source": "pc_xpass"}),
            patch.object(dataset_config, "validate_pc_xpass_lane_survival_mode_cache_metadata", return_value="max"),
            patch.object(dataset_config, "pc_xpass_lane_survival_metadata_fingerprint", return_value="fingerprint"),
        ):
            ipw_kwargs = build_ipw_dataset_kwargs(
                target_kwargs,
                checkpoint_args,
                {"lane_survival": lane_metadata},
                diagnostic_label_dir=None,
                require_goal_next10_diagnostics=False,
            )

        self.assertFalse(target_kwargs["lane_survival"])
        self.assertEqual(target_kwargs["task"], "pass_success")
        self.assertTrue(ipw_kwargs["lane_survival"])
        self.assertEqual(ipw_kwargs["lane_survival_mode"], "max")
        self.assertEqual(ipw_kwargs["lane_survival_cache_dir"], "lane-cache")
        self.assertEqual(ipw_kwargs["task"], "pass_success")
        self.assertTrue(ipw_kwargs["mask_possessor_v_edge_features"])
        self.assertFalse(ipw_kwargs["use_physical_xpass"])

    def test_ipw_lane_survival_checkpoint_rejects_missing_cache_and_fingerprint_mismatch(self) -> None:
        target_kwargs = build_action_dataset_kwargs(
            {"task": "pass_height", "edge_in_dim": 5}, train=True, diagnostic_label_dir=None
        )
        checkpoint_args = {"task": "pass_intent", "edge_in_dim": 5, "lane_survival": True, "lane_survival_mode": "max"}

        with self.assertRaisesRegex(ValueError, "does not record a pc-xPass cache"):
            build_ipw_dataset_kwargs(
                target_kwargs,
                checkpoint_args,
                {"lane_survival": {"enabled": True}},
                diagnostic_label_dir=None,
                require_goal_next10_diagnostics=False,
            )

        with (
            patch.object(dataset_config, "validate_physical_xpass_cache_metadata", return_value={"source": "pc_xpass"}),
            patch.object(dataset_config, "validate_pc_xpass_lane_survival_mode_cache_metadata", return_value="max"),
            patch.object(dataset_config, "pc_xpass_lane_survival_metadata_fingerprint", return_value="different"),
        ):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                build_ipw_dataset_kwargs(
                    target_kwargs,
                    checkpoint_args,
                    {"lane_survival": {"enabled": True, "cache_dir": "lane-cache", "cache_fingerprint": "expected"}},
                    diagnostic_label_dir=None,
                    require_goal_next10_diagnostics=False,
                )

    def test_dimension_validation_checks_checkpoint_widths(self) -> None:
        dataset = SimpleNamespace(features=[Data(x=torch.zeros((3, 7)), edge_attr=torch.zeros((2, 5)))])
        evaluation_script.validate_test_dataset_dimensions(
            dataset,
            SimpleNamespace(node_in_dim=7, edge_in_dim=5),
        )

        with self.assertRaisesRegex(ValueError, "node feature width"):
            evaluation_script.validate_test_dataset_dimensions(
                dataset,
                SimpleNamespace(node_in_dim=8, edge_in_dim=5),
            )


class WeightedPassSuccessEvaluationTests(unittest.TestCase):
    def test_wrapper_cli_parsing_and_pass_success_propagation(self) -> None:
        args = evaluate_relevant_models.parse_args(
            [
                "--weighted-pass-success-metrics",
                "--pass-height-model-id",
                "pass_height/run_1",
                "--discount",
                "false",
                "--v4-power",
                "3.5",
                "--v4-zero",
                "0.65",
                "--pc-xpass-cache-dir",
                "cache-dir",
            ]
        )
        self.assertTrue(args.weighted_pass_success_metrics)
        self.assertFalse(args.discount)
        self.assertEqual(args.v4_power, 3.5)
        self.assertEqual(args.v4_zero, 0.65)

        command = evaluate_relevant_models.add_weighted_pass_success_options(
            ["python", "test.py", "--model_id", "pass_success/run_1"], args, args.pass_height_model_id
        )
        self.assertEqual(
            command[-11:],
            [
                "--weighted-pass-success-metrics",
                "--pass-height-model-id",
                "pass_height/run_1",
                "--discount",
                "false",
                "--v4-power",
                "3.5",
                "--v4-zero",
                "0.65",
                "--pc-xpass-cache-dir",
                "cache-dir",
            ],
        )

    def test_effective_v4_weights_cover_discount_and_custom_parameters(self) -> None:
        distances = np.array([0.0, 35.0, 70.0, 80.0])
        heights = np.array([0.0, 0.8, 0.6, 0.9])
        discounted = physical_xpass_blend_weight_v4(distances, heights, power=4.0, zero_point=0.7)
        expected_partial = 0.8 * np.cos((np.pi / 2.0) * (0.5**4))
        np.testing.assert_allclose(discounted, [0.0, expected_partial, 0.0, 0.0], rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(
            physical_xpass_blend_weight_v4(distances, heights, power=2.0, zero_point=1.0),
            heights * np.cos((np.pi / 2.0) * (distances / 100.0) ** 2),
            rtol=1e-7,
        )
        np.testing.assert_allclose(
            physical_xpass_blend_weight_v4(distances, heights, power=2.0, zero_point=0.2, use_discount=False),
            heights,
            rtol=1e-7,
        )

    def test_weighted_metrics_match_sklearn_and_fail_for_invalid_subsets(self) -> None:
        target = np.array([0, 1, 0, 1])
        prediction = np.array([0.05, 0.70, 0.80, 0.55])
        weight = np.array([0.2, 0.9, 0.5, 0.4])
        metrics = calc_weighted_binary_probability_metrics(target, prediction, weight)
        self.assertAlmostEqual(metrics["high_pass_weighted_roc_auc"], roc_auc_score(target, prediction, sample_weight=weight))
        self.assertAlmostEqual(
            metrics["high_pass_weighted_brier"], brier_score_loss(target, prediction, sample_weight=weight)
        )
        with self.assertRaisesRegex(ValueError, "zero total effective weight"):
            calc_weighted_binary_probability_metrics([0, 1], [0.2, 0.8], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "both successful and unsuccessful"):
            calc_weighted_binary_probability_metrics([0, 1], [0.2, 0.8], [1.0, 0.0])

    def test_weighted_cache_requires_pc_xpass_and_matching_pass_height_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_args = SimpleNamespace(
                weighted_pass_success_metrics=True,
                pass_height_model_id="pass_height/run_1",
                pc_xpass_cache_dir=str(Path(tmpdir) / "missing"),
                v4_power=4.0,
                v4_zero=0.7,
            )
            with self.assertRaises(FileNotFoundError):
                evaluation_script.resolve_weighted_pass_success_cache(missing_args, SimpleNamespace(task="pass_success"))

            cache_dir = Path(tmpdir) / "pc_xpass"
            cache_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps({"source": "pc_xpass", "pass_height_model_id": "pass_height/run_1"}), encoding="utf-8"
            )
            args = SimpleNamespace(
                weighted_pass_success_metrics=True,
                pass_height_model_id="pass_height/run_1",
                pc_xpass_cache_dir=str(cache_dir),
                v4_power=4.0,
                v4_zero=0.7,
            )
            model_args = SimpleNamespace(task="pass_success")
            self.assertEqual(evaluation_script.resolve_weighted_pass_success_cache(args, model_args), str(cache_dir))

            args.pass_height_model_id = "pass_height/other"
            with self.assertRaisesRegex(ValueError, "does not match"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)

            args.pass_height_model_id = "pass_height/run_1"
            args.v4_zero = 0.0
            with self.assertRaisesRegex(ValueError, "positive finite"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)

            args.v4_zero = 0.7
            (cache_dir / "metadata.json").write_text(json.dumps({"source": "physical_xpass"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible source"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)


if __name__ == "__main__":
    unittest.main()
