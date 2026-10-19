import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import evaluate_relevant_models as wrapper, model_comparison


def checkpoint(model_id, *, expected_task=None):
    task = model_id.split("/", 1)[0]
    if expected_task and expected_task != task:
        raise ValueError("does not match selector task")
    return {"model_id": model_id, "task": task, "train_matches": 2,
            "train_match_ids": ["train1", "train2"], "test_match_ids": ["a", "b"],
            "target_definitions": {}, "split_manifest_id": "manifest",
            "checkpoint_record": {"args": {"task": task, "node_in_dim": 19, "edge_in_dim": 2},
                                  "model_id": model_id, "metadata": {"train_match_ids": ["train1", "train2"], "validation_match_ids": ["valid"]}, "is_complete": True}}


def binary_frame(predictions=(.2, .8, .2, .8)):
    return pd.DataFrame({"match_id": ["a", "a", "b", "b"], "source_index": [0, 1, 0, 1],
                         "target": [0, 1, 0, 1], "prediction": predictions})


class ComparisonEngineTests(unittest.TestCase):
    def test_all_pairs_direction_reproducible_bootstrap_and_single_match_disabled(self):
        records = [{"model_id": f"pass_success/{letter}", "configuration_id": "unmasked"} for letter in "abc"]
        frames = [binary_frame(), binary_frame((.1, .9, .1, .9)), binary_frame((.3, .7, .3, .7))]
        summary, differences = model_comparison.compare("pass_success", records, frames, resamples=15, seed=2)
        summary2, differences2 = model_comparison.compare("pass_success", records, frames, resamples=15, seed=2)
        pd.testing.assert_frame_equal(summary, summary2)
        pd.testing.assert_frame_equal(differences, differences2)
        brier = differences[differences.metric == "brier"]
        self.assertEqual(len(brier), 3)
        self.assertLess(brier.iloc[0].difference, 0)
        self.assertTrue(brier.direction.str.contains(" - ").all())
        model_comparison.compare("pass_success", records[:2], [frame.iloc[:2] for frame in frames[:2]], resamples=0, seed=2)
        with self.assertRaisesRegex(ValueError, "two contributing"):
            model_comparison.compare("pass_success", records[:2], [frame.iloc[:2] for frame in frames[:2]], resamples=1, seed=2)

    def test_row_target_and_duplicate_rejections(self):
        variants = [binary_frame().iloc[:3],
                    pd.concat([binary_frame(), binary_frame().iloc[:1]])]
        for bad in variants:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                records = []
                for index, frame in enumerate([binary_frame(), bad]):
                    output = root / str(index)
                    output.mkdir()
                    frame.to_csv(output / "learning_curve_predictions.csv", index=False)
                    records.append({**checkpoint(f"pass_success/{index}"), "output_dir": str(output)})
                with self.assertRaises(ValueError):
                    model_comparison.load_aligned_predictions(records)

    def test_semantic_intent_alignment_and_candidate_warning(self):
        first = binary_frame().assign(target=[0, 0, 0, 0], prediction=[0, 0, 0, 0],
                                      target_id="player_A", prediction_id="player_A", candidate_ids='["player_A","player_B"]',
                                      target_probability=.8, reciprocal_rank=1.)
        second = first.assign(target=1, prediction=1, candidate_ids='["player_B","player_A"]')
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for index, frame in enumerate([first, second]):
                output = Path(folder) / str(index)
                output.mkdir()
                frame.to_csv(output / "learning_curve_predictions.csv", index=False)
                records.append({**checkpoint(f"pass_intent/{index}"), "output_dir": str(output)})
            frames, warnings = model_comparison.load_aligned_predictions(records)
            self.assertEqual(warnings, [])
            summary, _ = model_comparison.compare("pass_intent", records, frames, resamples=0, seed=2)
            self.assertTrue((summary[summary.metric == "accuracy"].estimate == 1.).all())
            second.loc[0, "candidate_ids"] = '["player_A","player_C"]'
            second.to_csv(Path(records[1]["output_dir"]) / "learning_curve_predictions.csv", index=False)
            with self.assertWarnsRegex(RuntimeWarning, "Candidate membership"):
                _, messages = model_comparison.load_aligned_predictions(records)
            self.assertEqual(messages[0]["affected_row_count"], 1)

    def test_checkpoint_validation_accepts_features_seeds_and_same_sizes(self):
        records = [checkpoint("pass_success/a"), checkpoint("pass_success/b")]
        records[0]["checkpoint_record"]["args"].update(seed=1, model="gat", feature_run_id="one")
        records[1]["checkpoint_record"]["args"].update(seed=2, model="sage", feature_run_id="two")
        model_comparison.validate_target_definitions(records)
        for record in records:
            record["task"] = "outcome_scoring"
        records[0]["target_definitions"] = {"target_family": "goal"}
        records[1]["target_definitions"] = {"target_family": "xt"}
        self.assertTrue(model_comparison.validate_target_definitions(records))

    def test_evaluation_height_override_allows_different_training_cutoffs(self):
        with tempfile.TemporaryDirectory() as folder:
            records = []
            for index, cutoff in enumerate([1., 2.]):
                output = Path(folder) / str(index)
                output.mkdir()
                frame = binary_frame().assign(model_pass_height_threshold_meters=cutoff,
                                              evaluation_pass_height_threshold_meters=1.5,
                                              pass_height_label_mode="continuous")
                frame.to_csv(output / "learning_curve_predictions.csv", index=False)
                records.append({**checkpoint(f"pass_height/{index}"), "output_dir": str(output)})
            model_comparison.load_aligned_predictions(records)
        args = wrapper.parse_args(["--pass-height-threshold", "1.5", "--pass-height-model-id", "pass_height/a",
                                   "--pass-height-model-id", "pass_height/b"])
        def with_height(model_id, **kwargs):
            record = checkpoint(model_id, **kwargs)
            record["target_definitions"] = {"pass_height_definition": {"threshold_meters": 1. if model_id.endswith("/a") else 2.}}
            return record
        with patch.object(model_comparison, "checkpoint", side_effect=with_height):
            entries, _, _ = wrapper.resolve_evaluation_entries(args)
        model_comparison.validate_target_definitions(entries)


class EvaluationWrapperTests(unittest.TestCase):
    def run_wrapper(self, argv, output_root, *, checkpoint_fn=checkpoint):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            output = Path(command[command.index("--evaluation-output-dir") + 1])
            output.mkdir(parents=True)
            (output / "metadata.json").write_text('{}', encoding="utf-8")
            frame = binary_frame()
            if "--inference-mask" in command:
                frame["prediction"] = [.3, .7, .3, .7]
            if command[command.index("--model_id") + 1].startswith("pass_intent/"):
                frame = frame.assign(prediction=frame.target, target_id=["p0", "p1"] * 2,
                                     prediction_id=["p0", "p1"] * 2, candidate_ids='["p0","p1"]',
                                     target_probability=.8, reciprocal_rank=1.)
            frame.to_csv(output / "learning_curve_predictions.csv", index=False)
        args = wrapper.parse_args(["--accept-comparison-warnings", *argv, "--evaluation-output-dir", str(output_root), "--comparison-bootstrap-resamples", "0"])
        with patch.object(wrapper, "parse_args", return_value=args), \
             patch.object(model_comparison, "checkpoint", side_effect=checkpoint_fn), \
             patch.object(wrapper.subprocess, "run", side_effect=fake_run):
            wrapper.main()
        return commands

    def test_lane_survival_cache_reaches_comparison_and_mask_subprocesses(self):
        def with_lane_survival(model_id, **kwargs):
            record = checkpoint(model_id, **kwargs)
            record["checkpoint_record"]["args"]["lane_survival"] = model_id.endswith("/b")
            return record

        for mode in ("comparison", "masks"):
            for selection in ("directory", "id", "latest"):
                with self.subTest(mode=mode, selection=selection), tempfile.TemporaryDirectory() as folder:
                    cache_dir = str(Path(folder) / "compatible cache" / "sportec")
                    selected_id = "pc_xpass_selected" if selection == "id" else "pc_xpass_pinned_latest"
                    selector = (["--pc-xpass-cache-dir", cache_dir] if selection == "directory" else
                                ["--pc-xpass-id", selected_id] if selection == "id" else [])

                    def resolve_cache(source, args):
                        self.assertEqual(source, "sportec")
                        if selection == "id":
                            self.assertEqual(args.pc_xpass_id, selected_id)
                        else:
                            self.assertIsNone(args.pc_xpass_id)
                        args.pc_xpass_id = selected_id
                        args._pc_version_root = str(Path(folder) / selected_id)
                        return Path(args._pc_version_root) / source

                    flags = ["--compare"] if mode == "comparison" else ["--inference-mask", "accel"]
                    with patch.object(wrapper.pc_versions, "cache_dir", side_effect=resolve_cache) as resolve, \
                         patch.object(model_comparison, "save_plot"):
                        commands = self.run_wrapper([
                            *flags, "--pass-intent-model-id", "pass_intent/a",
                            "--pass-intent-model-id", "pass_intent/b", *selector,
                        ], Path(folder) / "reports", checkpoint_fn=with_lane_survival)

                    self.assertEqual(len(commands), 2 if mode == "comparison" else 4)
                    if selection == "directory":
                        resolve.assert_not_called()
                        option, value, absent = "--pc-xpass-cache-dir", cache_dir, "--pc-xpass-id"
                    else:
                        self.assertEqual(resolve.call_count, 1)
                        option, value, absent = "--pc-xpass-id", selected_id, "--pc-xpass-cache-dir"
                    for command in commands:
                        self.assertEqual(command.count(option), 1)
                        self.assertEqual(command[command.index(option) + 1], value)
                        self.assertNotIn(absent, command)
                    manifest = next((Path(folder) / "reports").glob("*/experiment_manifest.json"))
                    options = json.loads(manifest.read_text(encoding="utf-8"))["evaluation_options"]
                    self.assertEqual(options["pc_xpass_cache_dir"] if selection == "directory" else
                                     options["pc_xpass_id"], value)

    def test_pass_success_cache_selector_is_forwarded_only_once(self):
        for selection in ("directory", "id"):
            for predictor in ([], ["--evaluate-xpass", "--xpass-version", "max"],
                              ["--evaluate-combined-success", "--xpass-version", "max", "--xpass-weight", "v4",
                               "--discount", "true", "--v4-power", "4", "--v4-zero", "0.7"]):
                with self.subTest(selection=selection, predictor=predictor), tempfile.TemporaryDirectory() as folder:
                    option = "--pc-xpass-cache-dir" if selection == "directory" else "--pc-xpass-id"
                    value = str(Path(folder) / "cache" / "sportec") if selection == "directory" else "pc_xpass_selected"
                    with patch.object(wrapper.pc_versions, "cache_dir", return_value=Path(folder)), \
                         patch.object(model_comparison, "save_plot"):
                        commands = self.run_wrapper([
                            "--pass-success-model-id", "pass_success/a", "--weighted-pass-success-metrics",
                            "--xpass-pass-height-model-id", "pass_height/aux",
                            option, value, *predictor,
                        ], Path(folder) / "reports")
                    self.assertEqual(len(commands), 1)
                    self.assertEqual(commands[0].count(option), 1)
                    self.assertEqual(commands[0][commands[0].index(option) + 1], value)

    def test_cache_selection_reaches_every_learning_curve_subprocess(self):
        records = [{"model_id": f"pass_intent/{size}", "task": "pass_intent", "train_matches": size,
                    "test_match_ids": ["test"],
                    "comparison_config": {"feature_signature": {"lane_survival": True}}}
                   for size in (1, 2)]
        for selection in ("directory", "id", "latest"):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as folder:
                cache_dir = str(Path(folder) / "compatible cache" / "sportec")
                selected_id = "pc_xpass_selected" if selection == "id" else "pc_xpass_pinned_latest"
                selector = (["--pc-xpass-cache-dir", cache_dir] if selection == "directory" else
                            ["--pc-xpass-id", selected_id] if selection == "id" else [])
                args = wrapper.parse_args([
                    "--learning-curve", "--pass-intent-model-id", "pass_intent/anchor",
                    "--evaluation-output-dir", str(Path(folder) / "reports"),
                    "--learning-curve-bootstrap-resamples", "0", *selector,
                ])

                def resolve_cache(source, selection_args):
                    self.assertEqual(source, "sportec")
                    selection_args.pc_xpass_id = selected_id
                    return Path(folder) / selected_id / source

                def fake_run(command, **kwargs):
                    output = Path(command[command.index("--evaluation-output-dir") + 1])
                    output.mkdir(parents=True)
                    frame = pd.DataFrame({"match_id": ["test", "test"], "source_index": [0, 1],
                                          "target": [0, 1], "prediction": [0, 1],
                                          "target_probability": [.8, .9], "reciprocal_rank": [1., 1.]})
                    frame.to_csv(output / "learning_curve_predictions.csv", index=False)

                with patch.object(wrapper.learning_curve, "discover", return_value=[r["model_id"] for r in records]), \
                     patch.object(wrapper.learning_curve, "validate_series", return_value=records), \
                     patch.object(wrapper.pc_versions, "cache_dir", side_effect=resolve_cache) as resolve, \
                     patch.object(wrapper.subprocess, "run", side_effect=fake_run) as run, \
                     patch.object(wrapper.learning_curve, "save_plot"):
                    wrapper.run_learning_curve(args)

                self.assertEqual(run.call_count, 2)
                if selection == "directory":
                    resolve.assert_not_called()
                    option, value, absent = "--pc-xpass-cache-dir", cache_dir, "--pc-xpass-id"
                else:
                    self.assertEqual(resolve.call_count, 1)
                    option, value, absent = "--pc-xpass-id", selected_id, "--pc-xpass-cache-dir"
                for call in run.call_args_list:
                    command = call.args[0]
                    self.assertEqual(command.count(option), 1)
                    self.assertEqual(command[command.index(option) + 1], value)
                    self.assertNotIn(absent, command)
                manifest = next((Path(folder) / "reports").glob("*/experiment_manifest.json"))
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(metadata["pc_xpass_cache_dir"] if selection == "directory" else
                                 metadata["pc_xpass_id"], value)

    def test_cache_directory_and_id_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            wrapper.parse_args(["--pc-xpass-id", "pc_xpass_selected", "--pc-xpass-cache-dir", "cache/sportec"])

    def test_normal_repeated_selectors_and_shared_run_multi_task(self):
        with tempfile.TemporaryDirectory() as folder:
            commands = self.run_wrapper(["--pass-success-model-id", "pass_success/a",
                                         "--pass-success-model-id", "pass_success/b",
                                         "--pass-height-model-id", "pass_height/a"], Path(folder))
            self.assertEqual(len(commands), 3)
            self.assertTrue(all("--learning-curve-rows" not in command for command in commands))
            self.assertTrue(all("--pc-xpass-id" not in command and "--pc-xpass-cache-dir" not in command
                                for command in commands))
            self.assertEqual(len(set(command[command.index("--evaluation-output-dir") + 1] for command in commands)), 3)

    def test_mask_configs_compare_only_within_each_checkpoint_and_include_unmasked_once(self):
        for compare in ([], ["--compare"]):
            with self.subTest(compare=compare), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                commands = self.run_wrapper([*compare, "--pass-success-model-id", "pass_success/a",
                                             "--pass-success-model-id", "pass_success/b",
                                             "--inference-mask-2", "pos-node-features",
                                             "--inference-mask-1", "accel", "vel-node-features"], root)
                self.assertEqual(len(commands), 6)
                self.assertEqual(sum("--inference-mask" not in command for command in commands), 2)
                configuration_ids = [command[command.index("--inference-configuration-id") + 1] for command in commands]
                self.assertEqual(configuration_ids, ["unmasked", "mask_1", "mask_2"] * 2)
                reports = list(root.rglob("comparison_differences.csv"))
                self.assertEqual(len(reports), 2)
                for report in reports:
                    frame = pd.read_csv(report)
                    self.assertTrue((frame.model_a == frame.model_b).all())
                    self.assertEqual(len(frame[frame.metric == "brier"]), 3)

    def test_combined_masks_and_task_scopes_skip_unaffected_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            commands = self.run_wrapper(["--pass-success-model-id", "pass_success/a", "--pass-height-model-id", "pass_height/a",
                                         "--inference-mask", "pass_success:accel", "pass_success:pos-node-features"], root)
            self.assertEqual(len(commands), 3)
            manifests = list(root.glob("mask_comparison_*/experiment_manifest.json"))
            report = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(len(report["skipped_configurations"]), 1)

    def test_checkpoint_compare_mixed_tasks_duplicate_and_bundle_rejected(self):
        invalid = [["--compare", "--pass-success-model-id", "pass_success/a", "--pass-height-model-id", "pass_height/a"],
                   ["--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/a"],
                   ["--pass-success-model-id", "pass_height/a"]]
        with tempfile.TemporaryDirectory() as folder:
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    self.run_wrapper(argv, Path(folder))
            with patch.object(wrapper, "load_bundle_record", return_value={"model_ids": {}}), self.assertRaisesRegex(ValueError, "without --bundle-id"):
                self.run_wrapper(["--compare", "--bundle-id", "bundle",
                                  "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b"], Path(folder))

    def test_v5_explicit_auxiliaries_are_excluded_from_comparison(self):
        args = wrapper.parse_args(["--evaluate-combined-success", "--xpass-version", "max", "--xpass-weight", "v5",
                                   "--pass-intent-model-id", "pass_intent/component"])
        with self.assertRaisesRegex(ValueError, "required explicitly"):
            wrapper.validate_pass_success_predictor_args(args)
        with tempfile.TemporaryDirectory() as folder, patch.object(wrapper.pc_versions, "cache_dir", return_value=Path(folder)):
            commands = self.run_wrapper(["--compare", "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b",
                                         "--evaluate-combined-success", "--xpass-version", "max", "--xpass-weight", "v5",
                                         "--xpass-pass-height-model-id", "pass_height/aux", "--xpass-pass-intent-model-id", "pass_intent/aux"], Path(folder))
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--xpass-pass-height-model-id" in command for command in commands))
        self.assertTrue(all("--xpass-pass-intent-model-id" in command for command in commands))
        self.assertTrue(all("--pass-intent-model-id" not in command for command in commands))

    def test_parser_mask_conflicts_and_learning_curve_rejected(self):
        from models.inference_masks import resolve_inference_mask_configurations
        for argv in [["--inference-mask-1", "accel", "--inference-mask-1", "accel"],
                     ["--inference-mask", "accel", "--inference-mask-1", "accel"],
                     ["--inference-mask-0", "accel"]]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                resolve_inference_mask_configurations(wrapper.parse_args(argv))
        for flags in [["--compare"], ["--inference-mask", "accel"]]:
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "cannot be combined"):
                wrapper.learning_curve_selection(wrapper.parse_args(["--learning-curve", *flags]))
        with self.assertRaises(SystemExit):
            wrapper.parse_args(["--inference-mask-1"])

    def test_bundle_overrides_all_ids_for_explicit_task(self):
        args = wrapper.parse_args(["--bundle-id", "bundle", "--pass-success-model-id", "pass_success/a",
                                   "--pass-success-model-id", "pass_success/b"])
        with patch.object(wrapper, "load_bundle_record", return_value={"model_ids": {
            "pass_success": "pass_success/from_bundle", "pass_height": "pass_height/a"}}), \
             patch.object(model_comparison, "checkpoint", side_effect=checkpoint):
            entries, _, _ = wrapper.resolve_evaluation_entries(args)
        self.assertEqual([record["model_id"] for record in entries], ["pass_success/a", "pass_success/b", "pass_height/a"])

    def test_missing_feature_mask_fails_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            args = wrapper.parse_args(["--pass-success-model-id", "pass_success/a", "--inference-mask", "offside",
                                       "--evaluation-output-dir", folder])
            with patch.object(wrapper, "parse_args", return_value=args), \
                 patch.object(model_comparison, "checkpoint", side_effect=checkpoint), \
                 patch.object(wrapper.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
                    wrapper.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
