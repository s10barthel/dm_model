from __future__ import annotations

import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from datatools.hawkeye import build_hawkeye_situation
from scripts import run_hawkeye_loc
from scripts.run_hawkeye_loc import (
    _geometry_warning_record,
    build_invocation_metadata,
    build_location_target,
    geometry_hash,
    parse_args,
    required_input_columns,
)


def make_tracking(points: list[tuple[float, float, float]], receipt: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for abs_time, carrier_x, carrier_y in points:
        rows.extend(
            [
                {
                    "game_id": 1,
                    "half": 1,
                    "abs_time": abs_time,
                    "uefa_player_id": 1,
                    "role": 1,
                    "centroid_x": carrier_x,
                    "centroid_y": carrier_y,
                    "PlayerID": 1,
                    "id": "s1",
                    "team": "A",
                    "possession_team": "A",
                    "BallReceipt": receipt,
                    "GameID": "G1",
                },
                {
                    "game_id": 1,
                    "half": 1,
                    "abs_time": abs_time,
                    "uefa_player_id": 2,
                    "role": 1,
                    "centroid_x": 0.0,
                    "centroid_y": 0.0,
                    "PlayerID": 1,
                    "id": "s1",
                    "team": "B",
                    "possession_team": "A",
                    "BallReceipt": receipt,
                    "GameID": "G1",
                },
            ]
        )
    return pd.DataFrame(rows)


def make_ball(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"game_id": 1, "half": 1, "abs_time": time, "ball_x": 0.0, "ball_y": 0.0, "ball_z": 0.0} for time in times]
    )


class HawkeyeTargetClampingTest(unittest.TestCase):
    def test_freeze_cli_and_required_columns(self) -> None:
        self.assertEqual(parse_args([]).mode, "loc")
        self.assertEqual(parse_args(["--mode", "freeze"]).mode, "freeze")
        self.assertIn("PositionX", required_input_columns("loc"))
        self.assertIn("loc_info_missing", required_input_columns("loc"))
        self.assertNotIn("PositionX", required_input_columns("freeze"))
        self.assertNotIn("PositionY", required_input_columns("freeze"))
        self.assertNotIn("loc_info_missing", required_input_columns("freeze"))

    def test_freeze_target_uses_zero_offset_without_location_fields(self) -> None:
        tracking = make_tracking([(0.0, 3.0, 4.0), (1.0, 5.0, 6.0)])
        row = pd.Series(
            {"selection_row_id": 9, "action_id": "s1", "SelectedPlayer": 1, "pass_moment": 0.0}
        )
        fake_graph = object()
        situation = SimpleNamespace(
            target_geometry={"loc_target_position_clamped": False},
            tracking=pd.DataFrame({"frame_id": [0]}),
            frame_meta=pd.DataFrame({"abs_time": [1.0]}),
            graph_features_0=[fake_graph],
            graph_features_by_dir={"action_graphs": [fake_graph]},
            labels=torch.tensor([[0.0]]),
            actions=pd.DataFrame(index=[0]),
            match_id="s1",
        )
        attacking = pd.DataFrame({"frame_id": [0]})
        adjusted_info = {"adjusted_x": 5.0, "adjusted_y": 6.0}
        with (
            patch.object(run_hawkeye_loc, "resolve_target_frame", return_value={"frame_id": 0, "abs_time": 1.0, "time_norm": 0.0, "time_diff": 0.0}),
            patch.object(run_hawkeye_loc, "apply_hawkeye_possessor_offset", return_value=(tracking, adjusted_info)) as apply_offset,
            patch.object(run_hawkeye_loc, "build_hawkeye_situation", return_value=(situation, attacking, {})),
            patch.object(run_hawkeye_loc, "filter_situation_to_frame", return_value=attacking),
            patch.object(run_hawkeye_loc, "runtime_cache_items_from_graphs", return_value=[]),
        ):
            target = build_location_target(row, tracking, pd.DataFrame(), {"add_v_edge_features": False}, 0.1, "freeze")

        self.assertEqual(target["position_x"], 0.0)
        self.assertEqual(target["position_y"], 0.0)
        self.assertIn("__freeze__", target["synthetic_id"])
        self.assertEqual(apply_offset.call_args.kwargs, {"offset_x": 0.0, "offset_y": 0.0})

    def test_geometry_hash_distinguishes_loc_and_freeze(self) -> None:
        loc_hash = geometry_hash("s1", 1, 0, 0.0, 0.0, 3.0, 4.0, mode="loc")
        freeze_hash = geometry_hash("s1", 1, 0, 0.0, 0.0, 3.0, 4.0, mode="freeze")
        self.assertNotEqual(loc_hash, freeze_hash)

    def test_post_receipt_target_is_frozen_then_clamped_with_aligned_ball(self) -> None:
        tracking = make_tracking([(0.0, 0.0, 0.0), (1.0, 60.0, 35.0), (2.0, 61.0, 36.0)])
        situation, _, stats = build_hawkeye_situation(
            tracking,
            make_ball([0.0, 1.0, 2.0]),
            align_frozen_ball_to_possessor=True,
            target_abs_time=2.0,
            clamp_target_possessor=True,
        )

        self.assertEqual(stats["valid_frames"], 3)
        self.assertTrue(situation.target_geometry["loc_target_position_clamped"])
        self.assertEqual(situation.target_geometry["raw_target_possessor_x"], 60.0)
        self.assertEqual(situation.target_geometry["raw_target_possessor_y"], 35.0)
        self.assertEqual(situation.target_geometry["effective_target_possessor_x"], 52.5)
        self.assertEqual(situation.target_geometry["effective_target_possessor_y"], 34.0)
        self.assertEqual(situation.target_geometry["clamped_boundaries"], ["x_max", "y_max"])
        self.assertEqual(situation.tracking.at[2, "home_1_x"], 105.0)
        self.assertEqual(situation.tracking.at[2, "home_1_y"], 68.0)
        self.assertEqual(situation.tracking.at[2, "ball_x"], 105.0)
        self.assertEqual(situation.tracking.at[2, "ball_y"], 68.0)

    def test_pre_receipt_target_is_clamped_without_changing_other_frames(self) -> None:
        tracking = make_tracking([(0.0, -53.0, -35.0), (1.0, 1.0, 2.0), (2.0, 2.0, 3.0)])
        situation, _, _ = build_hawkeye_situation(
            tracking,
            make_ball([0.0, 1.0, 2.0]),
            build_graphs=False,
            align_frozen_ball_to_possessor=True,
            target_abs_time=0.0,
            clamp_target_possessor=True,
        )

        self.assertEqual(situation.target_geometry["effective_target_possessor_x"], -52.5)
        self.assertEqual(situation.target_geometry["effective_target_possessor_y"], -34.0)
        self.assertEqual(situation.tracking.at[0, "home_1_x"], 0.0)
        self.assertEqual(situation.tracking.at[0, "home_1_y"], 0.0)
        self.assertEqual(situation.tracking.at[0, "ball_x"], 0.0)
        self.assertEqual(situation.tracking.at[0, "ball_y"], 0.0)
        self.assertEqual(situation.tracking.at[1, "home_1_x"], 53.5)
        self.assertEqual(situation.tracking.at[1, "home_1_y"], 36.0)

    def test_exact_pitch_boundary_is_not_reported_as_clamped(self) -> None:
        tracking = make_tracking([(0.0, 0.0, 0.0), (1.0, 52.5, -34.0), (2.0, 0.0, 0.0)])
        situation, _, _ = build_hawkeye_situation(
            tracking,
            make_ball([0.0, 1.0, 2.0]),
            build_graphs=False,
            align_frozen_ball_to_possessor=True,
            target_abs_time=1.0,
            clamp_target_possessor=True,
        )

        self.assertFalse(situation.target_geometry["loc_target_position_clamped"])
        self.assertEqual(situation.target_geometry["clamped_boundaries"], [])
        self.assertEqual(situation.target_geometry["effective_target_possessor_x"], 52.5)
        self.assertEqual(situation.target_geometry["effective_target_possessor_y"], -34.0)

    def test_effective_target_geometry_changes_hash_and_warning_record(self) -> None:
        base = pd.DataFrame({"frame_id": [0], "ball_x": [105.0], "ball_y": [68.0]})
        clipped = pd.DataFrame({"frame_id": [0], "ball_x": [104.0], "ball_y": [68.0]})
        base_hash = geometry_hash("s1", 1, 0, 10.0, 20.0, 52.5, 34.0, situation_tracking=base)
        clipped_hash = geometry_hash("s1", 1, 0, 10.0, 20.0, 52.5, 34.0, situation_tracking=clipped)
        self.assertNotEqual(base_hash, clipped_hash)

        row = pd.Series({"selection_row_id": 1, "action_id": "s1", "SelectedPlayer": 4, "pass_moment": 0.1, "PositionX": 10, "PositionY": 20})
        frame = {"time_norm": 0.08, "time_diff": 0.02}
        geometry = {
            "raw_target_possessor_x": 53.0,
            "raw_target_possessor_y": 35.0,
            "effective_target_possessor_x": 52.5,
            "effective_target_possessor_y": 34.0,
            "clamped_boundaries": ["x_max", "y_max"],
        }
        warning = _geometry_warning_record(row, frame, geometry)
        self.assertEqual(warning["clamped_boundaries"], "x_max,y_max")
        self.assertEqual(warning["effective_target_possessor_x"], 52.5)

    def test_invocation_metadata_captures_all_effective_cli_options(self) -> None:
        argv = [
            "--input-file", "selection data.csv",
            "--reaction-time", "dist_pass",
            "--xpass-version", "top25",
            "--xpass-weight", "v4",
            "--discount", "false",
            "--physical-num-workers", "8",
        ]
        args = parse_args(argv)
        metadata = build_invocation_metadata(
            args,
            argv=argv,
            script_argv=["scripts/run_hawkeye_loc.py", *argv],
            working_directory=Path("C:/dm_model"),
        )

        options = metadata["effective_cli_options"]
        self.assertEqual(set(options), set(vars(args)))
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["argv"], argv)
        self.assertEqual(metadata["working_directory"], str(Path("C:/dm_model").resolve()))
        self.assertEqual(options["reaction_time"], None)
        self.assertEqual(options["reaction_time_mode"], "dist_pass")
        self.assertEqual(options["top_n_values"], [25])
        self.assertFalse(options["v4_discount"])
        self.assertEqual(options["num_workers"], "8")
        self.assertIn('"selection data.csv"', metadata["command"])
        json.dumps(metadata)


if __name__ == "__main__":
    unittest.main()
