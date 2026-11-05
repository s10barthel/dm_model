from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import project_config
from datatools import utils
from datatools.epv import annotate_match_epv, compute_epv_values
from scripts import generate_epv
from scripts import train_relevant_models as train_wrapper


def make_epv_events(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": index + 1,
                "spadl_type": spadl_type,
                "object_id": object_id,
                "epv": epv,
                "period_id": 1,
                "success": False,
                "expected_goal": 0.0,
            }
            for index, (spadl_type, object_id, epv) in enumerate(rows)
        ]
    )


class EpvFormulaTests(unittest.TestCase):
    def test_compute_epv_values_aligns_components_and_sums_weighted_pass_score(self) -> None:
        pass_intent = pd.DataFrame({"home_2": [0.25], "home_3": [0.75]}, index=[10])
        pass_success = pd.DataFrame({"home_2": [0.8], "home_3": [0.4]}, index=[10])
        scoring_success = pd.DataFrame({"home_2": [0.5], "home_3": [0.3]}, index=[10])
        scoring_failure = pd.DataFrame({"home_2": [0.2], "home_3": [0.1]}, index=[10])
        conceding_success = pd.DataFrame({"home_2": [0.1], "home_3": [0.05]}, index=[10])
        conceding_failure = pd.DataFrame({"home_2": [0.3], "home_3": [0.2]}, index=[10])

        epv = compute_epv_values(
            pass_intent,
            pass_success,
            scoring_success,
            scoring_failure,
            conceding_success,
            conceding_failure,
        )

        self.assertAlmostEqual(float(epv.loc[10]), 0.105)

    def test_annotate_match_epv_uses_max_epv_xg_for_shots(self) -> None:
        events = pd.DataFrame(
            [
                {"action_id": 1, "spadl_type": "pass", "object_id": "home_1", "expected_goal": 0.0, "period_id": 1, "success": False},
                {"action_id": 2, "spadl_type": "shot", "object_id": "home_1", "expected_goal": 0.5, "period_id": 1, "success": False},
                {"action_id": 3, "spadl_type": "shot", "object_id": "home_1", "expected_goal": 0.5, "period_id": 1, "success": False},
            ]
        )
        epv_action_values = pd.DataFrame({"action_id": [1, 2, 3], "epv": [0.2, 0.3, 0.7]})

        annotated, _ = annotate_match_epv(events, epv_action_values)

        self.assertEqual(float(annotated.loc[annotated["action_id"].eq(1), "epv"].iloc[0]), 0.2)
        self.assertEqual(float(annotated.loc[annotated["action_id"].eq(2), "epv"].iloc[0]), 0.5)
        self.assertEqual(float(annotated.loc[annotated["action_id"].eq(3), "epv"].iloc[0]), 0.7)


class EpvReturnLabelTests(unittest.TestCase):
    def test_epv_return_helpers_match_xt_style_semantics(self) -> None:
        events = make_epv_events(
            [
                ("pass", "home_1", 0.10),
                ("pass", "home_1", 0.20),
                ("pass", "away_1", 0.60),
                ("shot", "home_1", 0.40),
            ]
        )

        next_labeled = utils.label_epv_returns(events, lookahead_len=3)
        disc_labeled = utils.label_discounted_epv_returns(events, gamma=0.5)
        in_labeled = utils.label_epv_in_state_returns(events, action_offset=3)

        self.assertEqual(float(next_labeled.at[0, "scores_epv"]), 0.40)
        self.assertEqual(float(next_labeled.at[0, "concedes_epv"]), 0.60)
        # The first future eligible action has exponent zero.
        self.assertAlmostEqual(float(disc_labeled.at[0, "scores_epv"]), 1 - (1 - 0.20) * (1 - 0.5**2 * 0.40))
        self.assertAlmostEqual(float(disc_labeled.at[0, "concedes_epv"]), 1 - (1 - 0.5 * 0.60))
        self.assertEqual(float(in_labeled.at[0, "scores_epv"]), 0.40)
        self.assertEqual(float(in_labeled.at[0, "concedes_epv"]), 0.0)

    def test_epv_skip1_does_not_skip_first_rated_shot(self) -> None:
        events = make_epv_events(
            [
                ("pass", "home_1", 0.10),
                ("shot", "away_1", 0.70),
                ("pass", "home_1", 0.40),
            ]
        )

        next_labeled = utils.label_epv_returns(events, lookahead_len=3)
        next_skip_labeled = utils.label_epv_returns(events, lookahead_len=3, skip_first=True)

        self.assertEqual(float(next_skip_labeled.at[0, "scores_epv"]), float(next_labeled.at[0, "scores_epv"]))
        self.assertEqual(float(next_skip_labeled.at[0, "concedes_epv"]), float(next_labeled.at[0, "concedes_epv"]))


class EpvWrapperTests(unittest.TestCase):
    def test_epv_accepts_in_state_return_type(self) -> None:
        self.assertEqual(project_config.validate_return_type_for_target_family("in_3", target_family="epv"), "in_3")
        self.assertEqual(project_config.resolve_effective_return_type("epv", None), "next_5")

    def test_outcome_command_emits_use_epv_flag(self) -> None:
        command = train_wrapper.outcome_command(
            "outcome_scoring",
            "outcome_scoring/run",
            "features",
            "labels",
            "epv",
            "next_5",
            "angle_only",
            512,
            "all",
            train_wrapper.WRAPPER_FEATURE_DEFAULTS,
        )

        self.assertIn("--use_epv", command)

    def test_generate_epv_model_selection_uses_bundle_and_explicit_overrides(self) -> None:
        args = SimpleNamespace(
            bundle_id="bundle_1",
            pass_intent_model_id=None,
            pass_success_model_id="pass_success/override",
            outcome_scoring_model_id=None,
            outcome_conceding_model_id=None,
        )

        with patch.object(
            generate_epv,
            "resolve_model_selection",
            return_value=(
                {
                    "pass_intent": "pass_intent/from_bundle",
                    "pass_success": "pass_success/override",
                    "outcome_scoring": "outcome_scoring/from_bundle",
                    "outcome_conceding": "outcome_conceding/from_bundle",
                },
                {},
                {},
            ),
        ) as resolve:
            resolved, _, _ = generate_epv.resolve_epv_model_selection(args)

        self.assertEqual(resolved["pass_success"], "pass_success/override")
        resolve.assert_called_once_with(
            required_tasks=generate_epv.REQUIRED_EPV_MODEL_TASKS,
            bundle_id="bundle_1",
            explicit_model_ids={
                "pass_intent": None,
                "pass_success": "pass_success/override",
                "outcome_scoring": None,
                "outcome_conceding": None,
            },
            require_feature_run_id=False,
            require_intended_receiver_mode=False,
            require_return_type=False,
            require_target_family=False,
        )

    def test_generate_epv_model_selection_fails_when_required_model_is_missing(self) -> None:
        args = SimpleNamespace(
            bundle_id="bundle_1",
            pass_intent_model_id=None,
            pass_success_model_id=None,
            outcome_scoring_model_id=None,
            outcome_conceding_model_id=None,
        )

        with patch.object(generate_epv, "resolve_model_selection", side_effect=ValueError("missing pass_success")):
            with self.assertRaisesRegex(ValueError, "missing pass_success"):
                generate_epv.resolve_epv_model_selection(args)


if __name__ == "__main__":
    unittest.main()
