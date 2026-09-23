import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import learning_curve
from scripts.evaluate_relevant_models import learning_curve_selection, parse_args


class LearningCurveSelectionTests(unittest.TestCase):
    def test_individual_ids_expand_without_bundle(self):
        args = parse_args([
            "--learning-curve",
            "--outcome-scoring-model-id", "outcome_scoring/outcome_scoring_20260917T230020_827274_198dd085",
            "--outcome-conceding-model-id", "outcome_conceding/outcome_conceding_20260917T230020_827274_6f2aa381",
        ])
        with patch("scripts.evaluate_relevant_models.learning_curve.discover", side_effect=lambda model: [
            model + "/fold_1", model + "/fold_2", model + "/fold_3", model,
        ]):
            selected, skipped = learning_curve_selection(args)
        self.assertEqual(skipped, {})
        for task in ("outcome_scoring", "outcome_conceding"):
            root = getattr(args, f"{task}_model_id")
            self.assertEqual(selected[task], [root + "/fold_1", root + "/fold_2", root + "/fold_3", root])

    def test_bundle_override_and_direct_list(self):
        args = parse_args([
            "--learning-curve", "--bundle-id", "bundle_1",
            "--outcome-scoring-model-id", "outcome_scoring/override",
            "--learning-curve-model-id", "pass_height/a",
            "--learning-curve-model-id", "pass_height/b",
        ])
        with patch("scripts.evaluate_relevant_models.load_bundle_record", return_value={
            "model_ids": {"outcome_scoring": "outcome_scoring/bundle", "pass_success": "pass_success/one"}
        }), patch("scripts.evaluate_relevant_models.learning_curve.discover", side_effect=lambda model: [model + "/fold_1", model]):
            selected, skipped = learning_curve_selection(args)
        self.assertEqual(selected["outcome_scoring"], ["outcome_scoring/override/fold_1", "outcome_scoring/override"])
        self.assertEqual(selected["pass_height"], ["pass_height/a", "pass_height/b"])
        self.assertEqual(skipped, {})

    def test_root_is_not_counted_twice(self):
        root = "outcome_scoring/run_1"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for part in ("fold_1", "fold_2", "fold_3", "final_refit"):
                directory = path / part
                directory.mkdir()
                (directory / "metadata.json").write_text('{"status":"completed"}', encoding="utf-8")
                (directory / "best_weights.pt").touch()
            (path / "metadata.json").write_text('{"status":"completed","final_refit":true}', encoding="utf-8")
            with patch("scripts.learning_curve.get_model_path", return_value=path):
                ids = learning_curve.discover(root)
        self.assertEqual(ids, [root + "/fold_1", root + "/fold_2", root + "/fold_3", root])

    def test_direct_list_requires_two_distinct_sizes(self):
        with patch("scripts.learning_curve.checkpoint", side_effect=[
            {"model_id": "pass_height/a", "task": "pass_height", "train_matches": 100,
             "train_match_ids": ["a"], "test_match_ids": ["t"], "seed": 1},
            {"model_id": "pass_height/b", "task": "pass_height", "train_matches": 100,
             "train_match_ids": ["a"], "test_match_ids": ["t"], "seed": 1},
        ]):
            with self.assertRaisesRegex(ValueError, "two distinct training sizes"):
                learning_curve.validate_series(["pass_height/a", "pass_height/b"], expected_task="pass_height")

    def test_nested_membership_and_common_test_required(self):
        first = {"model_id": "pass_height/a", "task": "pass_height", "train_matches": 1,
                 "train_match_ids": ["a"], "test_match_ids": ["t"], "seed": 1}
        second = {"model_id": "pass_height/b", "task": "pass_height", "train_matches": 2,
                  "train_match_ids": ["b", "c"], "test_match_ids": ["t"], "seed": 1}
        with patch("scripts.learning_curve.checkpoint", side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, "not nested"):
                learning_curve.validate_series(["pass_height/a", "pass_height/b"], expected_task="pass_height")


class LearningCurveComparisonTests(unittest.TestCase):
    def test_paired_intervals_and_differences(self):
        records = [{"model_id": "pass_height/a", "train_matches": 100},
                   {"model_id": "pass_height/b", "train_matches": 200}]
        base = pd.DataFrame({"match_id": ["a", "a", "b", "b", "c", "c"],
                             "source_index": [0, 1, 0, 1, 0, 1],
                             "target": [0, 1, 0, 1, 0, 1]})
        first = base.assign(prediction=[.2, .8, .2, .8, .2, .8])
        second = base.assign(prediction=[.1, .9, .1, .9, .1, .9])
        summary, differences = learning_curve.compare("pass_height", records, [first, second], resamples=25, seed=7)
        summary_again, differences_again = learning_curve.compare("pass_height", records, [first, second], resamples=25, seed=7)
        pd.testing.assert_frame_equal(summary, summary_again)
        pd.testing.assert_frame_equal(differences, differences_again)
        brier = differences.loc[differences.metric == "brier"].iloc[0]
        self.assertLess(brier.difference, 0)
        self.assertEqual(brier.status, "ok")
        self.assertEqual(brier.valid_resamples, 25)
        self.assertEqual(len(summary.loc[summary.metric == "brier"]), 2)

    def test_identity_or_target_mismatch_fails(self):
        records = [{"model_id": "pass_height/a", "train_matches": 100, "test_match_ids": ["a"]},
                   {"model_id": "pass_height/b", "train_matches": 200, "test_match_ids": ["a"]}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for size, target in [(100, 0), (200, 1)]:
                directory = root / str(size)
                directory.mkdir()
                pd.DataFrame({"match_id": ["a"], "source_index": [0], "target": [target],
                              "prediction": [.5]}).to_csv(directory / "learning_curve_predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Evaluated examples or targets differ"):
                learning_curve.load_aligned_predictions(records, root)

    def test_all_task_metric_families(self):
        binary = pd.DataFrame({"target": [0, 1, 0, 1], "prediction": [.1, .9, .2, .8]})
        for task in ("pass_success", "pass_height"):
            self.assertIn("roc_auc", learning_curve.metrics(task, binary))
        outcome = binary.assign(soft_target=[.0, .8, .1, .7])
        for task in ("outcome_scoring", "outcome_conceding"):
            self.assertIn("xt_soft_bce", learning_curve.metrics(task, outcome))
        intent = pd.DataFrame({"target": [0, 1], "prediction": [0, 1],
                               "reciprocal_rank": [1.0, 1.0], "target_probability": [.8, .9]})
        for task in ("action_intent", "pass_intent", "success_intent"):
            self.assertIn("mrr", learning_curve.metrics(task, intent))


if __name__ == "__main__":
    unittest.main()
