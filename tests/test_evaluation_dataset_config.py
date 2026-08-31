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
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch_geometric.data import Data

import test as evaluation_script
import dataset as dataset_module
from dataset import ActionDataset
from datatools import config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from models import dataset_config
from models.dataset_config import build_action_dataset_kwargs, build_ipw_dataset_kwargs
from models.utils import (
    calc_binary_cohort_metrics,
    calc_binary_metrics,
    calc_continuous_target_metrics,
    calc_equal_frequency_bins,
    calc_pass_success_height_metrics,
    calc_pass_success_predictor_metrics,
    calc_weighted_binary_probability_metrics,
)
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
    def test_wrapper_evaluation_options_default_and_opt_out(self) -> None:
        defaults = evaluate_relevant_models.parse_args([])
        self.assertFalse(defaults.no_observed_pass_height_stratification)
        self.assertIsNone(defaults.f1_outcome_threshold)
        opted_out = evaluate_relevant_models.parse_args(["--no-observed-pass-height-stratification"])
        self.assertTrue(opted_out.no_observed_pass_height_stratification)
        threshold = evaluate_relevant_models.parse_args(["--f1-outcome-threshold", "0.25"])
        self.assertEqual(threshold.f1_outcome_threshold, 0.25)
        with self.assertRaises(SystemExit):
            evaluate_relevant_models.parse_args(["--f1-outcome-threshold", "nan"])
        with self.assertRaises(SystemExit):
            evaluate_relevant_models.parse_args(["--f1-outcome-threshold", "1.1"])

    def test_wrapper_forwards_task_specific_options(self) -> None:
        args = evaluate_relevant_models.parse_args(["--f1-outcome-threshold", "0.2"])
        pass_command = evaluate_relevant_models.add_task_evaluation_options(["test.py"], args, "pass_success")
        scoring_command = evaluate_relevant_models.add_task_evaluation_options(["test.py"], args, "outcome_scoring")
        unrelated_command = evaluate_relevant_models.add_task_evaluation_options(["test.py"], args, "pass_height")
        self.assertIn("--observed-pass-height-stratification", pass_command)
        self.assertEqual(scoring_command[-2:], ["--f1-outcome-threshold", "0.2"])
        self.assertEqual(unrelated_command, ["test.py"])

        opted_out = evaluate_relevant_models.parse_args(["--no-observed-pass-height-stratification"])
        self.assertEqual(
            evaluate_relevant_models.add_task_evaluation_options(["test.py"], opted_out, "pass_success"),
            ["test.py"],
        )

    def test_xpass_evaluation_cli_validation_and_forwarding(self) -> None:
        standalone = evaluate_relevant_models.parse_args(
            ["--evaluate-xpass", "--xpass-version", "top25"]
        )
        evaluate_relevant_models.validate_pass_success_predictor_args(standalone)
        command = evaluate_relevant_models.add_task_evaluation_options(["test.py"], standalone, "pass_success")
        self.assertIn("--evaluate-xpass", command)
        self.assertEqual(command[-2:], ["--xpass-version", "top25"])

        combined = evaluate_relevant_models.parse_args(
            [
                "--evaluate-combined-success", "--xpass-version", "top25", "--xpass-weight", "v4",
                "--discount", "true", "--v4-power", "4", "--v4-zero", "0.75",
            ]
        )
        evaluate_relevant_models.validate_pass_success_predictor_args(combined)
        combined_command = evaluate_relevant_models.add_task_evaluation_options(
            ["test.py"], combined, "pass_success"
        )
        self.assertIn("--evaluate-combined-success", combined_command)
        self.assertIn("--xpass-weight", combined_command)
        self.assertIn("0.75", combined_command)

        invalid_argv = (
            ["--evaluate-xpass"],
            ["--evaluate-combined-success", "--xpass-version", "top25"],
            ["--evaluate-xpass", "--xpass-version", "top25", "--xpass-weight", "v3"],
            ["--evaluate-combined-success", "--xpass-version", "top25", "--xpass-weight", "v4"],
            [
                "--evaluate-combined-success", "--xpass-version", "top25", "--xpass-weight", "v3",
                "--v4-power", "4",
            ],
        )
        for argv in invalid_argv:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                evaluate_relevant_models.validate_pass_success_predictor_args(
                    evaluate_relevant_models.parse_args(argv)
                )

    def test_binary_metrics_omit_or_include_threshold_metrics(self) -> None:
        target = np.array([0, 1, 1])
        prediction = np.array([0.1, 0.4, 0.8])
        threshold_free = calc_binary_metrics(target, prediction, threshold=None)
        thresholded = calc_binary_metrics(target, prediction, threshold=0.5)
        self.assertTrue({"roc_auc", "brier", "log_loss"}.issubset(threshold_free))
        self.assertTrue({"precision", "recall", "f1"}.isdisjoint(threshold_free))
        self.assertTrue({"precision", "recall", "f1"}.issubset(thresholded))

    def test_binary_cohort_metrics_report_count_and_prevalence(self) -> None:
        metrics = calc_binary_cohort_metrics(np.array([0, 1, 1, 0, 1]))
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["positive_count"], 3)
        self.assertAlmostEqual(metrics["positive_prevalence"], 0.6)

    def test_binary_cohort_metrics_reject_non_binary_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite 0/1"):
            calc_binary_cohort_metrics(np.array([0, 0.5, 1]))

    def test_observed_height_metrics_match_direct_calculations(self) -> None:
        target = np.array([1, 0, 1, 0, 1])
        prediction = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        observed_high = np.array([1, 1, 1, 0, 0])
        flat, rows = calc_pass_success_height_metrics(target, prediction, observed_high)
        high = next(row for row in rows if row["stratum"] == "observed_high")
        non_high = next(row for row in rows if row["stratum"] == "observed_non_high")
        self.assertEqual(high["sample_count"], 3)
        self.assertEqual(high["positive_count"], 2)
        self.assertAlmostEqual(high["success_prevalence"], 2 / 3)
        self.assertAlmostEqual(high["roc_auc"], roc_auc_score(target[:3], prediction[:3]))
        self.assertAlmostEqual(high["brier"], brier_score_loss(target[:3], prediction[:3]))
        self.assertEqual(non_high["sample_count"], 2)
        self.assertIn("pass_success_observed_high_roc_auc", flat)
        self.assertNotIn("f1", high)

    def test_observed_height_one_class_stratum_keeps_brier(self) -> None:
        _, rows = calc_pass_success_height_metrics(
            np.array([1, 1, 0, 1]),
            np.array([0.8, 0.7, 0.4, 0.6]),
            np.array([1, 1, 0, 0]),
        )
        high = next(row for row in rows if row["stratum"] == "observed_high")
        self.assertTrue(np.isnan(high["roc_auc"]))
        self.assertTrue(np.isfinite(high["brier"]))

    def test_pass_success_predictors_share_cohort_and_metrics(self) -> None:
        target = np.array([1, 0, 1, 0])
        height = np.array([1, 1, 0, 0])
        predictors = {
            "learning": np.array([0.9, 0.6, 0.7, 0.1]),
            "physical_xpass_top25": np.array([0.8, 0.3, 0.6, 0.2]),
            "combined_v4": np.array([0.85, 0.45, 0.65, 0.15]),
        }
        flat, rows = calc_pass_success_predictor_metrics(target, height, predictors)
        self.assertEqual(len(rows), 9)
        for predictor, prediction in predictors.items():
            pooled = next(row for row in rows if row["predictor"] == predictor and row["stratum"] == "pooled")
            self.assertEqual(pooled["sample_count"], 4)
            self.assertAlmostEqual(pooled["roc_auc"], roc_auc_score(target, prediction))
            self.assertAlmostEqual(pooled["brier"], brier_score_loss(target, prediction))
            self.assertAlmostEqual(pooled["log_loss"], log_loss(target, prediction, labels=[0, 1]))
            self.assertIn(f"pass_success_predictor_{predictor}_f1", flat)
            self.assertIn(f"pass_success_predictor_{predictor}_log_loss", flat)

    def test_pass_success_predictor_csv_has_stable_schema(self) -> None:
        rows = [{
            "predictor": "learning", "stratum": "pooled", "sample_count": 2,
            "positive_count": 1, "success_prevalence": 0.5, "roc_auc": 1.0,
            "brier": 0.1, "log_loss": 0.2, "f1": 1.0,
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = evaluation_script.write_pass_success_predictor_metrics(tmpdir, "pass_success/1", rows)
            exported = pd.read_csv(path)
            self.assertListEqual(
                list(exported.columns),
                ["model_id", "predictor", "stratum", "sample_count", "positive_count",
                 "success_prevalence", "roc_auc", "brier", "log_loss", "f1"],
            )

    def test_pass_success_height_csv_has_stable_schema(self) -> None:
        rows = [
            {
                "stratum": "observed_high", "sample_count": 2, "positive_count": 1,
                "success_prevalence": 0.5, "roc_auc": 0.75, "brier": 0.2,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = evaluation_script.write_pass_success_height_metrics(tmpdir, "pass_success/1", rows)
            exported = pd.read_csv(path)
            self.assertListEqual(
                list(exported.columns),
                ["model_id", "stratum", "sample_count", "positive_count", "success_prevalence", "roc_auc", "brier"],
            )

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
                "--evaluation-output-dir",
                "evaluation-output",
            ]
        )
        self.assertTrue(args.weighted_pass_success_metrics)
        self.assertFalse(args.discount)
        self.assertEqual(args.v4_power, 3.5)
        self.assertEqual(args.v4_zero, 0.65)
        self.assertEqual(args.evaluation_output_dir, "evaluation-output")

        command = evaluate_relevant_models.add_weighted_pass_success_options(
            ["python", "test.py", "--model_id", "pass_success/run_1"], args
        )
        self.assertEqual(
            command[-9:],
            [
                "--weighted-pass-success-metrics",
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
        self.assertNotIn("--pass-height-model-id", command)

    def test_wrapper_selects_nonempty_explicit_model_subsets_in_registry_order(self) -> None:
        args = evaluate_relevant_models.parse_args(
            [
                "--pass-height-model-id", "pass_height/run_1",
                "--action-intent-model-id", "action_intent/run_1",
            ]
        )
        tasks, explicit, bundle = evaluate_relevant_models.requested_evaluation_tasks(args)
        self.assertEqual(tasks, ["action_intent", "pass_height"])
        self.assertEqual(
            explicit,
            {"action_intent": "action_intent/run_1", "pass_height": "pass_height/run_1"},
        )
        self.assertIsNone(bundle)

        empty_args = evaluate_relevant_models.parse_args([])
        with self.assertRaisesRegex(ValueError, "At least one explicit"):
            evaluate_relevant_models.requested_evaluation_tasks(empty_args)

    def test_wrapper_bundle_selection_ignores_unsupported_tasks_and_allows_explicit_additions(self) -> None:
        args = evaluate_relevant_models.parse_args(
            [
                "--bundle-id", "bundle/run_1",
                "--action-intent-model-id", "action_intent/override",
            ]
        )
        bundle = {"model_ids": {"pass_success": "pass_success/from_bundle", "unrelated": "other/run"}}
        with patch.object(evaluate_relevant_models, "load_bundle_record", return_value=bundle):
            tasks, explicit, selected_bundle = evaluate_relevant_models.requested_evaluation_tasks(args)
        self.assertEqual(tasks, ["action_intent", "pass_success"])
        self.assertEqual(explicit, {"action_intent": "action_intent/override"})
        self.assertEqual(selected_bundle, bundle)

        with patch.object(
            evaluate_relevant_models,
            "load_bundle_record",
            return_value={"model_ids": {"unrelated": "other/run"}},
        ):
            with self.assertRaisesRegex(ValueError, "none of the supported"):
                evaluate_relevant_models.requested_evaluation_tasks(
                    evaluate_relevant_models.parse_args(["--bundle-id", "bundle/empty"])
                )

    def test_wrapper_rejects_options_scoped_to_unselected_tasks(self) -> None:
        action_args = evaluate_relevant_models.parse_args(
            ["--action-intent-model-id", "action_intent/run_1", "--weighted-pass-success-metrics"]
        )
        with self.assertRaisesRegex(ValueError, "require pass_success"):
            evaluate_relevant_models.validate_selected_task_options(action_args, ["action_intent"])

        outcome_args = evaluate_relevant_models.parse_args(
            ["--action-intent-model-id", "action_intent/run_1", "--f1-outcome-threshold", "0.1"]
        )
        with self.assertRaisesRegex(ValueError, "outcome model"):
            evaluate_relevant_models.validate_selected_task_options(outcome_args, ["action_intent"])

        scoring_args = evaluate_relevant_models.parse_args(
            ["--outcome-scoring-model-id", "outcome_scoring/run_1", "--f1-outcome-threshold", "0.1"]
        )
        evaluate_relevant_models.validate_selected_task_options(scoring_args, ["outcome_scoring"])

    def test_wrapper_scopes_diagnostic_feature_run_to_pass_success_diagnostics(self) -> None:
        pass_args = evaluate_relevant_models.parse_args(
            [
                "--pass-success-model-id", "pass_success/run_1",
                "--diagnostic-feature-run-id", "feature_diagnostic",
            ]
        )
        evaluate_relevant_models.validate_selected_task_options(pass_args, ["pass_success"])
        self.assertTrue(evaluate_relevant_models.task_uses_diagnostic_feature_run(pass_args, "pass_success"))
        self.assertFalse(evaluate_relevant_models.task_uses_diagnostic_feature_run(pass_args, "action_intent"))

        unused_args = evaluate_relevant_models.parse_args(
            [
                "--pass-success-model-id", "pass_success/run_1",
                "--diagnostic-feature-run-id", "feature_diagnostic",
                "--no-observed-pass-height-stratification",
            ]
        )
        with self.assertRaisesRegex(ValueError, "pass-height diagnostics"):
            evaluate_relevant_models.validate_selected_task_options(unused_args, ["pass_success"])

    def test_wrapper_forwards_diagnostic_feature_run_to_pass_height(self) -> None:
        args = evaluate_relevant_models.parse_args(
            [
                "--pass-height-model-id", "pass_height/run_1",
                "--diagnostic-feature-run-id", "feature_diagnostic",
            ]
        )
        evaluate_relevant_models.validate_selected_task_options(args, ["pass_height"])
        self.assertTrue(evaluate_relevant_models.task_uses_diagnostic_feature_run(args, "pass_height"))
        command = ["python", "test.py", "--model_id", "pass_height/run_1"]
        if evaluate_relevant_models.task_uses_diagnostic_feature_run(args, "pass_height"):
            command.extend(["--diagnostic-feature-run-id", args.diagnostic_feature_run_id])
        self.assertEqual(command[-2:], ["--diagnostic-feature-run-id", "feature_diagnostic"])

    def test_wrapper_scopes_diagnostic_run_across_mixed_tasks(self) -> None:
        args = evaluate_relevant_models.parse_args(
            [
                "--action-intent-model-id", "action_intent/run_1",
                "--pass-height-model-id", "pass_height/run_1",
                "--outcome-scoring-model-id", "outcome_scoring/run_1",
                "--diagnostic-feature-run-id", "feature_diagnostic",
            ]
        )
        evaluate_relevant_models.validate_selected_task_options(
            args, ["action_intent", "pass_height", "outcome_scoring"]
        )
        self.assertFalse(evaluate_relevant_models.task_uses_diagnostic_feature_run(args, "action_intent"))
        self.assertTrue(evaluate_relevant_models.task_uses_diagnostic_feature_run(args, "pass_height"))
        self.assertTrue(evaluate_relevant_models.task_uses_diagnostic_feature_run(args, "outcome_scoring"))

    def test_pass_height_diagnostic_resolver_accepts_transitive_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_root = Path(tmpdir) / "diagnostic"
            label_dir = diagnostic_root / "action_labels_disc_0.7"
            label_dir.mkdir(parents=True)
            metadata = {
                "diagnostic": {
                    "derived_from_feature_run_id": "middle",
                    "pass_height_threshold_meters": 1.0,
                },
                "middle": {"derived_from_feature_run_id": "base"},
                "base": {"derived_from_feature_run_id": None},
            }
            args = SimpleNamespace(diagnostic_feature_run_id="diagnostic")
            model_args = SimpleNamespace(
                task="pass_success", return_type="disc_0.7", intended_receiver_mode="original"
            )
            with (
                patch.object(evaluation_script, "resolve_feature_run_id", side_effect=lambda value, **_: value),
                patch.object(evaluation_script, "resolve_feature_root", return_value=diagnostic_root),
                patch.object(
                    evaluation_script,
                    "load_feature_run_metadata",
                    side_effect=lambda run_id, required=True: metadata[run_id],
                ),
            ):
                run_id, resolved_dir, threshold, lineage = (
                    evaluation_script.resolve_pass_height_diagnostic_context(
                        args, model_args, "base", required=True
                    )
                )
            self.assertEqual(run_id, "diagnostic")
            self.assertEqual(resolved_dir, str(label_dir))
            self.assertEqual(threshold, 1.0)
            self.assertEqual(lineage, ["diagnostic", "middle", "base"])

            pass_height_args = SimpleNamespace(
                task="pass_height", return_type="disc_0.7", intended_receiver_mode="original"
            )
            with (
                patch.object(evaluation_script, "resolve_feature_run_id", side_effect=lambda value, **_: value),
                patch.object(evaluation_script, "resolve_feature_root", return_value=diagnostic_root),
                patch.object(
                    evaluation_script,
                    "load_feature_run_metadata",
                    side_effect=lambda run_id, required=True: metadata[run_id],
                ),
            ):
                height_run_id, height_dir, height_threshold, height_lineage = (
                    evaluation_script.resolve_pass_height_diagnostic_context(
                        args, pass_height_args, "base", required=True
                    )
                )
            self.assertEqual(height_run_id, "diagnostic")
            self.assertEqual(height_dir, str(label_dir))
            self.assertEqual(height_threshold, 1.0)
            self.assertEqual(height_lineage, ["diagnostic", "middle", "base"])

    def test_pass_height_diagnostic_resolver_rejects_unrelated_run_and_invalid_threshold(self) -> None:
        args = SimpleNamespace(diagnostic_feature_run_id="diagnostic")
        model_args = SimpleNamespace(
            task="pass_success", return_type="disc_0.7", intended_receiver_mode="original"
        )
        with (
            patch.object(evaluation_script, "resolve_feature_run_id", side_effect=lambda value, **_: value),
            patch.object(
                evaluation_script,
                "load_feature_run_metadata",
                side_effect=lambda run_id, required=True: {
                    "diagnostic": {"derived_from_feature_run_id": "other", "pass_height_threshold_meters": 1.0},
                    "other": {"derived_from_feature_run_id": None},
                }[run_id],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not equal to or derived"):
                evaluation_script.resolve_pass_height_diagnostic_context(args, model_args, "base", required=True)

        with (
            patch.object(evaluation_script, "resolve_feature_run_id", side_effect=lambda value, **_: value),
            patch.object(
                evaluation_script,
                "load_feature_run_metadata",
                return_value={"derived_from_feature_run_id": None, "pass_height_threshold_meters": 0.0},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                evaluation_script.resolve_pass_height_diagnostic_context(
                    SimpleNamespace(diagnostic_feature_run_id="base"), model_args, "base", required=True
                )

    def test_pass_height_diagnostic_columns_are_overlaid_and_strictly_validated(self) -> None:
        selected = torch.zeros((2, len(LABEL_COLUMNS) - 2), dtype=torch.float32)
        diagnostic = torch.zeros((2, len(LABEL_COLUMNS)), dtype=torch.float32)
        for row, action_index in enumerate((10, 11)):
            selected[row, LABEL_INDEX["action_index"]] = action_index
            diagnostic[row, LABEL_INDEX["action_index"]] = action_index
            selected[row, LABEL_INDEX["is_pass"]] = 1
            diagnostic[row, LABEL_INDEX["is_pass"]] = 1
            selected[row, LABEL_INDEX["success"]] = row
            diagnostic[row, LABEL_INDEX["success"]] = row
        diagnostic[:, LABEL_INDEX["pass_max_ball_z"]] = torch.tensor([1.2, 0.4])
        diagnostic[:, LABEL_INDEX["pass_high"]] = torch.tensor([1.0, 0.0])

        overlaid = dataset_module._copy_pass_height_diagnostics("match_1", selected, diagnostic)
        self.assertEqual(overlaid.shape[1], len(LABEL_COLUMNS))
        torch.testing.assert_close(
            overlaid[:, LABEL_INDEX["pass_high"]], torch.tensor([1.0, 0.0])
        )

        misaligned = diagnostic.clone()
        misaligned[0, LABEL_INDEX["action_index"]] = 99
        with self.assertRaisesRegex(ValueError, "do not align"):
            dataset_module._copy_pass_height_diagnostics("match_1", selected, misaligned)

        non_binary = diagnostic.clone()
        non_binary[0, LABEL_INDEX["pass_high"]] = 0.5
        with self.assertRaisesRegex(ValueError, "non-binary pass-row"):
            dataset_module._copy_pass_height_diagnostics("match_1", selected, non_binary)

        non_pass_nan = diagnostic.clone()
        non_pass_nan[1, LABEL_INDEX["is_pass"]] = 0
        selected_non_pass = selected.clone()
        selected_non_pass[1, LABEL_INDEX["is_pass"]] = 0
        non_pass_nan[1, LABEL_INDEX["pass_max_ball_z"]] = float("nan")
        non_pass_nan[1, LABEL_INDEX["pass_high"]] = float("nan")
        accepted = dataset_module._copy_pass_height_diagnostics("match_1", selected_non_pass, non_pass_nan)
        self.assertEqual(float(accepted[1, LABEL_INDEX["pass_max_ball_z"]]), 0.0)
        self.assertEqual(float(accepted[1, LABEL_INDEX["pass_high"]]), 0.0)

        pass_nan = diagnostic.clone()
        pass_nan[0, LABEL_INDEX["pass_max_ball_z"]] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite pass-row"):
            dataset_module._copy_pass_height_diagnostics("match_1", selected, pass_nan)

    def test_action_dataset_loads_pass_height_from_diagnostic_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_dir = root / "features"
            label_dir = root / "labels"
            diagnostic_dir = root / "diagnostics"
            feature_dir.mkdir()
            label_dir.mkdir()
            diagnostic_dir.mkdir()

            x = torch.zeros((2, config.NODE_FEATURE_CORE_DIM), dtype=torch.float32)
            x[:, config.NODE_FEATURE_IS_TEAMMATE] = 1
            x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
            x[:, config.NODE_FEATURE_X] = torch.tensor([40.0, 60.0])
            x[:, config.NODE_FEATURE_Y] = 34.0
            graph = Data(
                x=x,
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                edge_attr=torch.zeros((2, 2), dtype=torch.float32),
            )
            selected = torch.zeros((1, len(LABEL_COLUMNS) - 2), dtype=torch.float32)
            selected[0, LABEL_INDEX["action_index"]] = 7
            selected[0, LABEL_INDEX["is_pass"]] = 1
            selected[0, LABEL_INDEX["intent_index"]] = 1
            selected[0, LABEL_INDEX["success"]] = 1
            diagnostic = torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)
            diagnostic[:, : selected.shape[1]] = selected
            diagnostic[0, LABEL_INDEX["pass_max_ball_z"]] = 1.3
            diagnostic[0, LABEL_INDEX["pass_high"]] = 1
            torch.save([graph], feature_dir / "match_1.pt")
            torch.save(selected, label_dir / "match_1.pt")
            torch.save(diagnostic, diagnostic_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=feature_dir,
                label_dir=label_dir,
                pass_height_diagnostic_label_dir=diagnostic_dir,
                require_pass_height_labels=True,
                task="pass_success",
                edge_in_dim=2,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(float(dataset.labels[0][LABEL_INDEX["pass_high"]]), 1.0)
            self.assertAlmostEqual(float(dataset.labels[0][LABEL_INDEX["pass_max_ball_z"]]), 1.3, places=5)

            pass_height_dataset = ActionDataset(
                ["match_1"],
                feature_dir=feature_dir,
                label_dir=label_dir,
                pass_height_diagnostic_label_dir=diagnostic_dir,
                task="pass_height",
                edge_in_dim=2,
            )
            self.assertEqual(len(pass_height_dataset), 1)
            torch.testing.assert_close(pass_height_dataset.features[0].x, graph.x)
            torch.testing.assert_close(pass_height_dataset.features[0].edge_index, graph.edge_index)
            torch.testing.assert_close(pass_height_dataset.features[0].edge_attr, graph.edge_attr)
            self.assertEqual(float(pass_height_dataset.labels[0][LABEL_INDEX["pass_high"]]), 1.0)
            self.assertAlmostEqual(
                float(pass_height_dataset.labels[0][LABEL_INDEX["pass_max_ball_z"]]), 1.3, places=5
            )

    def test_model_evaluation_output_dir_uses_task_and_shared_timestamp(self) -> None:
        root = Path("evaluation-root")
        self.assertEqual(
            evaluate_relevant_models.model_evaluation_output_dir(root, "pass_success", "20260821T003732"),
            root / "pass_success_20260821T003732",
        )

    def test_model_evaluation_output_dirs_reject_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pass_success_20260821T003732").mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                evaluate_relevant_models.model_evaluation_output_dirs(
                    root,
                    ["pass_success", "outcome_scoring"],
                    "20260821T003732",
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

    def test_weighted_cache_requires_pc_xpass_provenance_but_not_matching_live_model(self) -> None:
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
            self.assertEqual(evaluation_script.resolve_weighted_pass_success_cache(args, model_args), str(cache_dir))

            args.pass_height_model_id = "pass_height/run_1"
            args.v4_zero = 0.0
            with self.assertRaisesRegex(ValueError, "positive finite"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)

            args.v4_zero = 0.7
            (cache_dir / "metadata.json").write_text(json.dumps({"source": "pc_xpass"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pass-height provenance"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)

            (cache_dir / "metadata.json").write_text(json.dumps({"source": "physical_xpass"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible source"):
                evaluation_script.resolve_weighted_pass_success_cache(args, model_args)

    def test_xpass_evaluation_cache_resolves_requested_metric_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pc_xpass"
            cache_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "source": "pc_xpass",
                        "available_metrics": ["max_xpass", "top10_xpass", "top25_xpass"],
                        "pass_height_model_id": "pass_height/1",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                evaluate_xpass=True,
                evaluate_combined_success=False,
                evaluation_output_dir=str(Path(tmpdir) / "output"),
                xpass_version="top25",
                xpass_weight=None,
                discount=None,
                v4_power=None,
                v4_zero=None,
                pc_xpass_cache_dir=str(cache_dir),
            )
            resolved_dir, metric, metadata = evaluation_script.resolve_evaluation_xpass_cache(
                args, SimpleNamespace(task="pass_success")
            )
            self.assertEqual(resolved_dir, str(cache_dir))
            self.assertEqual(metric, "top25_xpass")
            self.assertEqual(len(metadata["metadata_sha256"]), 64)

            args.xpass_version = "top50"
            with self.assertRaisesRegex(ValueError, "not available"):
                evaluation_script.resolve_evaluation_xpass_cache(args, SimpleNamespace(task="pass_success"))


class OutcomeEvaluationArtifactTests(unittest.TestCase):
    def test_binary_metrics_are_computed_from_the_pooled_sample(self) -> None:
        first_target, first_prediction = np.array([0, 1]), np.array([0.7, 0.8])
        second_target, second_prediction = np.array([0, 1]), np.array([0.1, 0.2])
        pooled = calc_binary_metrics(
            np.concatenate([first_target, second_target]),
            np.concatenate([first_prediction, second_prediction]),
            threshold=0.5,
        )
        batch_average_auc = np.nanmean(
            [
                calc_binary_metrics(first_target, first_prediction, threshold=0.5)["roc_auc"],
                calc_binary_metrics(second_target, second_prediction, threshold=0.5)["roc_auc"],
            ]
        )
        self.assertAlmostEqual(pooled["roc_auc"], roc_auc_score([0, 1, 0, 1], [0.7, 0.8, 0.1, 0.2]))
        self.assertNotAlmostEqual(pooled["roc_auc"], batch_average_auc)

    def test_continuous_metrics_and_equal_frequency_bins(self) -> None:
        target = np.array([0.1, 0.2, 0.6, 0.9])
        prediction = np.array([0.2, 0.1, 0.5, 0.8])
        metrics = calc_continuous_target_metrics(target, prediction)

        self.assertAlmostEqual(metrics["mae"], 0.1)
        self.assertAlmostEqual(metrics["rmse"], 0.1)
        self.assertAlmostEqual(metrics["mean_prediction_minus_target"], -0.05)
        self.assertGreater(metrics["pearson_r"], 0.9)
        self.assertGreater(metrics["spearman_rho"], 0.7)

        binned = calc_equal_frequency_bins(target, prediction, n_bins=10)
        self.assertEqual(len(binned), len(target))
        self.assertEqual(int(binned["sample_count"].sum()), len(target))
        self.assertListEqual(binned["bin"].tolist(), [1, 2, 3, 4])

    def test_binned_relationship_limits_adapt_to_small_values(self) -> None:
        binned = pd.DataFrame(
            {
                "evaluation_target": ["xt_training_target", "xt_training_target", "goal_next10_diagnostic"],
                "mean_prediction": [0.012, 0.045, 0.011],
                "mean_observed": [0.010, 0.040, 0.032],
            }
        )
        calibration_x, calibration_y = evaluation_script.binned_relationship_axis_limits(
            binned,
            evaluation_target="xt_training_target",
            show_identity=True,
        )
        association_x, association_y = evaluation_script.binned_relationship_axis_limits(
            binned,
            evaluation_target="goal_next10_diagnostic",
            show_identity=False,
        )
        self.assertEqual(calibration_x, calibration_y)
        self.assertLess(calibration_x[1], 0.1)
        self.assertLess(association_x[1], association_y[1])
        self.assertLess(association_y[1], 0.1)

    def test_outcome_artifacts_include_pooled_and_factual_strata(self) -> None:
        evaluation = {
            "prediction": np.array([0.1, 0.3, 0.7, 0.9]),
            "target": np.array([0.2, 0.2, 0.8, 0.9]),
            "diagnostic": np.array([0, 0, 1, 1]),
            "execution_branch": np.array([0, 1, 0, 1]),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir, outcome_metrics = evaluation_script.write_outcome_evaluation_artifacts(
                tmpdir,
                model_id="outcome_scoring/31",
                task="outcome_scoring",
                outcome_evaluation=evaluation,
            )
            summary_dir = Path(tmpdir) / "evaluations"
            with patch.object(evaluation_script, "EVALUATION_RUNS_DIR", summary_dir):
                evaluation_script.write_model_evaluation_artifacts(
                    output_dir,
                    model_id="outcome_scoring/31",
                    task="outcome_scoring",
                    feature_run_id="features_24_25",
                    diagnostic_feature_run_id="diagnostics_24_25",
                    evaluation_timestamp="20260821T003732",
                    evaluation_options={"weighted_pass_success_metrics": False},
                    test_metrics={"ce_loss": 0.2, "roc_auc": float("nan")},
                    outcome_metrics=outcome_metrics,
                )

                evaluation_script.write_model_evaluation_artifacts(
                    output_dir,
                    model_id="outcome_scoring/31",
                    task="outcome_scoring",
                    feature_run_id="features_24_25",
                    diagnostic_feature_run_id="diagnostics_24_25",
                    evaluation_timestamp="20260821T003732",
                    evaluation_options={"weighted_pass_success_metrics": False},
                    test_metrics={"ce_loss": 0.3, "roc_auc": float("nan")},
                    outcome_metrics=outcome_metrics,
                )

            self.assertTrue((output_dir / "metrics.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "outcome_metrics.csv").exists())
            self.assertTrue((output_dir / "calibration_bins.csv").exists())
            self.assertTrue((output_dir / "xt_target_calibration.png").exists())
            self.assertTrue((output_dir / "goal_next10_association.png").exists())
            metrics = pd.read_csv(output_dir / "outcome_metrics.csv")
            self.assertEqual(len(metrics), 6)
            self.assertSetEqual(set(metrics["stratum"]), {"pooled_factual", "observed_success", "observed_failure"})
            diagnostic_rows = metrics.loc[metrics["evaluation_target"] == "goal_next10_diagnostic"]
            self.assertEqual(int(diagnostic_rows.loc[diagnostic_rows["stratum"] == "pooled_factual", "positive_count"].iloc[0]), 2)
            self.assertNotIn("f1", metrics.columns)
            run_metadata = json.loads((Path(tmpdir) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(run_metadata["feature_run_id"], "features_24_25")
            self.assertIsNone(json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]["roc_auc"])
            summary = pd.read_csv(summary_dir / "metrics_summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary.loc[0, "ce_loss"]), 0.3)
            self.assertIn("outcome_xt_training_target_mae", summary.columns)

    def test_outcome_threshold_enables_precision_recall_and_f1(self) -> None:
        evaluation = {
            "prediction": np.array([0.05, 0.2, 0.4, 0.8]),
            "target": np.array([0.01, 0.02, 0.03, 0.04]),
            "diagnostic": np.array([0, 0, 1, 1]),
            "execution_branch": np.array([0, 1, 0, 1]),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            _, metrics = evaluation_script.write_outcome_evaluation_artifacts(
                tmpdir,
                model_id="outcome_scoring/threshold",
                task="outcome_scoring",
                outcome_evaluation=evaluation,
                f1_outcome_threshold=0.1,
            )
        diagnostic = metrics.loc[metrics["evaluation_target"] == "goal_next10_diagnostic"]
        self.assertTrue({"precision", "recall", "f1"}.issubset(diagnostic.columns))


if __name__ == "__main__":
    unittest.main()
