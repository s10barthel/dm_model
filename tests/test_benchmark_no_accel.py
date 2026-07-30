from __future__ import annotations

import json
import math
import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Batch, Data

from dataset import ActionDataset
from datatools import config
from datatools.benchmark import build_benchmark_state, load_benchmark_game_state
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from datatools.match import Match
from datatools.utils import filter_features_and_labels
from models import utils as model_utils
from models.gnn import GNN
from physical_pass_model import (
    AS_DEFAULT_N_ANGLES,
    AS_DEFAULT_N_V0,
    _candidate_target_indices,
    compute_graph_player_cum_prob,
    pc_xpass_lane_survival_column,
    pc_xpass_metadata,
)
from scripts import run_benchmark
from scripts import train_relevant_models as train_wrapper
from validation.benchmark import benchmark_postprocessing as benchmark_post

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datatools.viz_helpers import figure_to_rgb_image
from scripts import visualize_benchmark
from scripts import visualize_hawkeye
from scripts.visualization_selection import add_component_selection_args, resolve_component_selection


def make_graph(node_dim: int = 25, edge_dim: int = 2) -> Data:
    x = torch.zeros((2, node_dim), dtype=torch.float32)
    x[:, 0] = 1
    x[0, 13] = 1
    x[:, 5] = 5
    x[:, 6] = 6
    x[:, 7] = 7
    x[:, 8] = 8
    x[:, 9] = 9
    x[:, 10] = 10
    x[:, 11] = 11
    x[:, 12] = 12
    x[:, 14] = 14
    x[:, 15] = 15
    x[:, 16] = 16
    x[:, 17] = 17
    x[:, 18] = 18
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones((edge_index.shape[1], edge_dim), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_goal_node_graph(node_dim: int = 25, edge_dim: int = 2) -> Data:
    x = torch.zeros((3, node_dim), dtype=torch.float32)
    x[:, 0] = 1
    x[0, config.NODE_FEATURE_IS_POSSESSOR] = 1
    x[2, config.NODE_FEATURE_IS_GOAL] = 1
    x[:, config.NODE_FEATURE_X] = torch.tensor([0.0, 10.0, 105.0])
    x[:, config.NODE_FEATURE_Y] = torch.tensor([34.0, 30.0, 34.0])
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.ones((edge_index.shape[1], edge_dim), dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.node_ids = ["possessor", "target", "goal"]
    return graph


def make_velocity_edge_graph(node_dim: int = 25) -> Data:
    x = torch.zeros((3, node_dim), dtype=torch.float32)
    x[:, 0] = 1
    x[0, 13] = 1
    x[:, 7] = 7
    x[:, 8] = 8
    x[:, 9] = 9
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [10.0, 1.0, 0.2, 0.3],
            [10.0, 1.0, 0.4, 0.5],
            [5.0, 1.0, 0.6, 0.7],
            [5.0, 1.0, 0.8, 0.9],
        ],
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_relative_speed_edge_graph(node_dim: int = 25) -> Data:
    graph = make_velocity_edge_graph(node_dim=node_dim)
    graph.edge_attr = torch.cat(
        [graph.edge_attr, torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)],
        dim=1,
    )
    return graph


def make_physical_xpass_graph(node_dim: int = 26) -> Data:
    x = torch.zeros((5, node_dim), dtype=torch.float32)
    x[:, config.NODE_FEATURE_IS_TEAMMATE] = torch.tensor([1, 1, 1, 0, 1], dtype=torch.float32)
    x[:, config.NODE_FEATURE_IS_GOAL] = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)
    x[:, config.NODE_FEATURE_IS_POSSESSOR] = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
    x[:, config.NODE_FEATURE_X : config.NODE_FEATURE_VY + 1] = torch.tensor(
        [
            [40.0, 34.0, 1.0, 0.0],
            [70.0, 34.0, 2.0, 0.5],
            [105.0, 34.0, 0.0, 0.0],
            [55.0, 45.0, -1.0, 0.0],
            [float("nan"), 30.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x[:, -1] = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
    return Data(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_ids=["possessor", "target", "home_goal", "defender", "bad_xy"],
    )


def make_labels() -> torch.Tensor:
    return torch.zeros((1, len(LABEL_COLUMNS)), dtype=torch.float32)


def make_pass_labels(*, action_index: int = 7, intent_index: int = 1, success: float = 1.0) -> torch.Tensor:
    labels = make_labels()
    labels[0, LABEL_INDEX["action_index"]] = action_index
    labels[0, LABEL_INDEX["is_pass"]] = 1
    labels[0, LABEL_INDEX["intent_index"]] = intent_index
    labels[0, LABEL_INDEX["success"]] = float(success)
    return labels


def write_pc_xpass_lane_survival_cache(cache_dir: Path, *, match_id: str = "match_1", action_index: int = 7, target_value: float = 0.42) -> None:
    (cache_dir / "matches").mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").write_text(json.dumps(pc_xpass_metadata("consider_teammates")), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "match_id": match_id,
                "action_index": action_index,
                pc_xpass_lane_survival_column("target"): target_value,
            }
        ]
    ).to_parquet(cache_dir / "matches" / f"{match_id}.parquet", index=False)


def make_legacy_labels() -> torch.Tensor:
    return torch.zeros((1, len(LABEL_COLUMNS) - 2), dtype=torch.float32)


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
        "pass_height": False,
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
    node_in_dim: int = 25,
    edge_in_dim: int = 4,
    add_v_edge_features: bool = True,
) -> dict[str, object]:
    return {
        "feature_run_id": feature_run_id,
        "intended_receiver_mode": intended_receiver_mode,
        "return_type": return_type,
        "target_family": target_family,
        "graph_schema": {
            "node_in_dim": node_in_dim,
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
    node_in_dim: int = 25,
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
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "add_v_edge_features": add_v_edge_features,
        },
    }


class BenchmarkNoAccelTests(unittest.TestCase):
    @staticmethod
    def _make_runtime_feature_run(root: Path, run_id: str, *, mode: str = "model", with_resolved_actions: bool = True) -> Path:
        feature_root = root / run_id
        (feature_root / "action_graphs").mkdir(parents=True, exist_ok=True)
        if with_resolved_actions:
            model_utils.get_resolved_action_dir(mode, root=feature_root).mkdir(parents=True, exist_ok=True)
        return feature_root

    @staticmethod
    def _write_benchmark_state_csv(path: Path, *, ball_pos_z: float | None = 0.13) -> None:
        pd.DataFrame(
            [
                {
                    "team": 1,
                    "player": 21,
                    "pos_x": 47.63,
                    "pos_y": -25.64,
                    "pos_z": pd.NA,
                    "smooth_x_speed": -0.71,
                    "smooth_y_speed": -0.15,
                    "event_player": 21,
                    "playing_direction_event": True,
                },
                {
                    "team": 1,
                    "player": 13,
                    "pos_x": 40.69,
                    "pos_y": -16.03,
                    "pos_z": pd.NA,
                    "smooth_x_speed": 5.13,
                    "smooth_y_speed": 1.95,
                    "event_player": 21,
                    "playing_direction_event": True,
                },
                {
                    "team": 2,
                    "player": 20,
                    "pos_x": 30.0,
                    "pos_y": -18.92,
                    "pos_z": pd.NA,
                    "smooth_x_speed": 4.06,
                    "smooth_y_speed": -3.69,
                    "event_player": 21,
                    "playing_direction_event": True,
                },
                {
                    "team": 0,
                    "player": 0,
                    "pos_x": 47.4,
                    "pos_y": -25.34,
                    "pos_z": ball_pos_z,
                    "smooth_x_speed": 0.05,
                    "smooth_y_speed": 1.89,
                    "event_player": 21,
                    "playing_direction_event": True,
                },
            ]
        ).to_csv(path, index=False)

    @staticmethod
    def _make_benchmark_postprocessing_df() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "modification": 1,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.6,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 1,
                    "game_state": 2,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.2,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 1,
                    "game_state": 2,
                    "higher_state_id": 1,
                    "pass_intent": 0.1,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 1.0,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 2,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.1,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 2,
                    "game_state": 2,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.4,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 3,
                    "game_state": 1,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.3,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
                {
                    "modification": 3,
                    "game_state": 2,
                    "higher_state_id": 1,
                    "pass_intent": 1.0,
                    "pass_success": 1.0,
                    "outcome_scoring_success": 0.3,
                    "outcome_conceding_success": 0.0,
                    "outcome_scoring_failure": 0.0,
                    "outcome_conceding_failure": 0.0,
                },
            ]
        )

    def test_pass_height_label_uses_inclusive_two_meter_threshold(self) -> None:
        match = Match.__new__(Match)
        match.tracking = pd.DataFrame(
            {"ball_z": [0.0, 1.2, config.PASS_HEIGHT_THRESHOLD_METERS]},
            index=[10, 11, 12],
        )

        max_ball_z, pass_high = Match.pass_height_labels(match, 10, 12)

        self.assertEqual(max_ball_z, config.PASS_HEIGHT_THRESHOLD_METERS)
        self.assertEqual(pass_high, 1.0)

    def test_pass_height_label_below_threshold_is_low(self) -> None:
        match = Match.__new__(Match)
        match.tracking = pd.DataFrame({"ball_z": [0.0, 1.99]}, index=[10, 11])

        max_ball_z, pass_high = Match.pass_height_labels(match, 10, 11)

        self.assertAlmostEqual(max_ball_z, 1.99)
        self.assertEqual(pass_high, 0.0)

    def test_pass_height_label_uses_configured_threshold(self) -> None:
        match = Match.__new__(Match)
        match.tracking = pd.DataFrame({"ball_z": [0.0, 1.8]}, index=[10, 11])
        original_threshold = config.PASS_HEIGHT_THRESHOLD_METERS
        try:
            config.PASS_HEIGHT_THRESHOLD_METERS = 1.8
            _max_ball_z, pass_high = Match.pass_height_labels(match, 10, 11)
        finally:
            config.PASS_HEIGHT_THRESHOLD_METERS = original_threshold

        self.assertEqual(pass_high, 1.0)

    def test_benchmark_loader_preserves_players_with_blank_pos_z(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "game_state_1.csv"
            self._write_benchmark_state_csv(state_path)

            loaded = load_benchmark_game_state(state_path)

        possessor_rows = loaded.loc[(loaded["team"] != 0) & loaded["player"].eq(loaded["event_player"])]
        self.assertEqual(len(possessor_rows), 1)
        self.assertTrue(pd.isna(possessor_rows.iloc[0]["pos_z"]))
        self.assertEqual(len(loaded), 4)

    def test_benchmark_state_builds_when_player_pos_z_is_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "game_state_1.csv"
            self._write_benchmark_state_csv(state_path)
            loaded = load_benchmark_game_state(state_path)

        state, export_rows, stats = build_benchmark_state(
            loaded,
            modification_id=49,
            game_state_id=1,
            higher_state_id=1,
            build_graphs=False,
        )

        self.assertEqual(state.possessor_player, 21)
        self.assertEqual(state.possessor_team, 1)
        self.assertEqual(len(export_rows), 4)
        self.assertEqual(stats["total_frames"], 1)

    def test_benchmark_loader_rejects_blank_ball_pos_z(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "game_state_1.csv"
            self._write_benchmark_state_csv(state_path, ball_pos_z=None)

            with self.assertRaisesRegex(ValueError, "ball row.*pos_z"):
                load_benchmark_game_state(state_path)

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

    def test_benchmark_postprocessing_counts_agreements_disagreements_and_draws(self) -> None:
        summary_df, stats = benchmark_post.build_benchmark_summary(self._make_benchmark_postprocessing_df())

        self.assertEqual(
            summary_df.columns.tolist(),
            [
                "modification",
                "game_state_value_1",
                "game_state_value_2",
                "avg_pass_score_1",
                "avg_pass_score_2",
                "higher_state_id",
                "model_rating",
                "agreement",
                "pass_score_rating",
                "pass_score_agreement",
            ],
        )
        self.assertTrue(math.isclose(summary_df.loc[0, "game_state_value_2"], 0.3))
        self.assertTrue(math.isclose(summary_df.loc[0, "avg_pass_score_2"], 0.6))
        self.assertEqual(summary_df["model_rating"].tolist(), [1, 2, 0])
        self.assertEqual(summary_df["agreement"].tolist()[:2], [1, 0])
        self.assertTrue(pd.isna(summary_df.loc[2, "agreement"]))
        self.assertEqual(summary_df["pass_score_rating"].tolist(), [0, 2, 0])
        self.assertTrue(pd.isna(summary_df.loc[0, "pass_score_agreement"]))
        self.assertEqual(summary_df.loc[1, "pass_score_agreement"], 0)
        self.assertTrue(pd.isna(summary_df.loc[2, "pass_score_agreement"]))
        self.assertEqual(stats["modifications_total"], 3)
        self.assertEqual(stats["modifications_with_game_state_value_1"], 3)
        self.assertEqual(stats["modifications_with_game_state_value_2"], 3)
        self.assertEqual(stats["modifications_with_both_game_states"], 3)
        self.assertEqual(stats["modifications_with_avg_pass_score_1"], 3)
        self.assertEqual(stats["modifications_with_avg_pass_score_2"], 3)
        self.assertEqual(stats["modifications_with_both_avg_pass_scores"], 3)
        self.assertEqual(stats["draws"], 1)
        self.assertEqual(stats["agreements"], 1)
        self.assertEqual(stats["disagreements"], 1)
        self.assertEqual(stats["pass_score_draws"], 2)
        self.assertEqual(stats["pass_score_agreements"], 0)
        self.assertEqual(stats["pass_score_disagreements"], 1)
        self.assertEqual(stats["output_rows"], 3)

    def test_benchmark_postprocessing_writes_run_summary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "benchmark_component_test"
            run_root.mkdir()
            self._make_benchmark_postprocessing_df().to_csv(run_root / "benchmark_data.csv", index=False)

            summary_df, stats, summary_path, text_summary_path = benchmark_post.run_benchmark_postprocessing(
                "benchmark_component_test",
                output_dir=run_root,
            )

            self.assertEqual(summary_path, run_root / "benchmark_summary.csv")
            self.assertEqual(text_summary_path, run_root / "benchmark_summary.txt")
            self.assertTrue(summary_path.exists())
            self.assertTrue(text_summary_path.exists())
            written_summary = pd.read_csv(summary_path)
            self.assertEqual(written_summary.shape[0], len(summary_df))
            self.assertIn("avg_pass_score_1", written_summary.columns)
            self.assertIn("avg_pass_score_2", written_summary.columns)
            self.assertIn("pass_score_rating", written_summary.columns)
            self.assertIn("pass_score_agreement", written_summary.columns)
            self.assertEqual(stats["agreements"], 1)
            self.assertEqual(stats["disagreements"], 1)
            self.assertEqual(stats["pass_score_agreements"], 0)
            self.assertEqual(stats["pass_score_disagreements"], 1)
            self.assertEqual(
                text_summary_path.read_text(encoding="utf-8").splitlines(),
                [
                    "game_state_value",
                    "modifications_total: 3",
                    "modifications_with_game_state_value_1: 3",
                    "modifications_with_game_state_value_2: 3",
                    "modifications_with_both_game_states: 3",
                    "draws: 1",
                    "agreements: 1",
                    "disagreements: 1",
                    "output_rows: 3",
                    "",
                    "avg_pass_score",
                    "modifications_total: 3",
                    "modifications_with_avg_pass_score_1: 3",
                    "modifications_with_avg_pass_score_2: 3",
                    "modifications_with_both_avg_pass_scores: 3",
                    "pass_score_draws: 2",
                    "pass_score_agreements: 0",
                    "pass_score_disagreements: 1",
                    "output_rows: 3",
                ],
            )

    def test_benchmark_runs_ledger_creates_and_replaces_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "benchmark_runs.csv"

            run_benchmark.update_benchmark_runs_ledger(
                "run_a",
                {"agreements": 3, "disagreements": 1, "pass_score_agreements": 2, "pass_score_disagreements": 2},
                ledger_path=ledger_path,
            )
            run_benchmark.update_benchmark_runs_ledger(
                "run_a",
                {"agreements": 1, "disagreements": 1, "pass_score_agreements": 1, "pass_score_disagreements": 3},
                ledger_path=ledger_path,
            )
            run_benchmark.update_benchmark_runs_ledger(
                "run_b",
                {"agreements": 0, "disagreements": 0},
                ledger_path=ledger_path,
            )

            ledger = pd.read_csv(ledger_path)

        self.assertEqual(
            ledger.columns.tolist(),
            [
                "run_id",
                "agreements",
                "disagreements",
                "performance",
                "pass_score_agreements",
                "pass_score_disagreements",
                "pass_score_performance",
            ],
        )
        self.assertEqual(ledger["run_id"].tolist(), ["run_a", "run_b"])
        self.assertEqual(ledger.loc[0, "agreements"], 1)
        self.assertEqual(ledger.loc[0, "disagreements"], 1)
        self.assertTrue(math.isclose(ledger.loc[0, "performance"], 0.5))
        self.assertEqual(ledger.loc[0, "pass_score_agreements"], 1)
        self.assertEqual(ledger.loc[0, "pass_score_disagreements"], 3)
        self.assertTrue(math.isclose(ledger.loc[0, "pass_score_performance"], 0.25))
        self.assertTrue(pd.isna(ledger.loc[1, "performance"]))
        self.assertTrue(pd.isna(ledger.loc[1, "pass_score_performance"]))

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

    def test_wrapper_parse_args_accepts_no_offside(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(["--feature-run-id", "feature_run", "--success-intent-only", "--no-offside"])

        self.assertFalse(args.offside_aware)

    def test_wrapper_parse_args_accepts_feature_ablation_flags(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--success-intent-only",
                    "--no-poss-geometry",
                    "--no-goal-features",
                    "--no-goal-nodes",
                ]
            )

        self.assertFalse(args.poss_geometry_aware)
        self.assertFalse(args.goal_features_aware)
        self.assertFalse(args.goal_nodes_aware)

    def test_wrapper_parse_args_accepts_possessor_masked_velocity_edges(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(
                ["--feature-run-id", "feature_run", "--success-intent-only", "--v-edge-features-no-poss"]
            )

        self.assertEqual(args.v_edge_feature_mode, "no_poss")
        self.assertTrue(args.use_v_edge_features)
        self.assertTrue(args.mask_possessor_v_edge_features)

    def test_wrapper_parse_args_defaults_no_pin_memory_and_early_stopping(self) -> None:
        with (
            patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
            patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["original", "angle_only"]),
            patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]),
        ):
            args = train_wrapper.parse_args(["--feature-run-id", "feature_run", "--success-intent-only"])

        self.assertFalse(args.pin_memory)
        self.assertTrue(args.early_stopping)
        self.assertEqual(args.early_stopping_patience, 10)
        self.assertEqual(args.early_stopping_min_epochs, 30)
        self.assertEqual(args.early_stopping_min_delta, 1e-5)

    def test_early_stopping_helpers_use_min_delta_and_patience(self) -> None:
        self.assertTrue(model_utils.is_validation_loss_improved(0.9, 0, 1e-5))
        self.assertTrue(model_utils.is_validation_loss_improved(0.99998, 1.0, 1e-5))
        self.assertFalse(model_utils.is_validation_loss_improved(0.999995, 1.0, 1e-5))

        self.assertFalse(model_utils.should_stop_early(True, 29, 30, 10, 10))
        self.assertFalse(model_utils.should_stop_early(True, 30, 30, 9, 10))
        self.assertTrue(model_utils.should_stop_early(True, 30, 30, 10, 10))
        self.assertFalse(model_utils.should_stop_early(False, 30, 30, 10, 10))

    def test_wrapper_training_control_flags_can_disable_early_stopping(self) -> None:
        command = train_wrapper.append_training_control_flags(
            ["train.py"],
            {
                "early_stopping": False,
                "early_stopping_patience": 10,
                "early_stopping_min_epochs": 30,
                "early_stopping_min_delta": 1e-5,
            },
        )

        self.assertIn("--early-stopping-patience", command)
        self.assertIn("--no-early-stopping", command)

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
                poss_rel_vel_aware=None,
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

    def test_resolve_enabled_tasks_only_pass_height(self) -> None:
        args = SimpleNamespace(
            success_intent_only=False,
            only_pass_height=True,
            pass_intent_model_id=None,
            pass_height_ipw=False,
            pass_height_ipw_model_id=None,
            outcome_scoring_trial=None,
            outcome_conceding_trial=None,
        )
        for task, _, _, _ in train_wrapper.MODEL_TOGGLE_SPECS:
            setattr(args, task, None)

        enabled = train_wrapper.resolve_enabled_tasks(args)

        self.assertEqual([task for task, value in enabled.items() if value], ["pass_height"])

    def test_build_training_commands_emit_pass_height_only_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_root = Path(tmpdir)
            args = SimpleNamespace(
                feature_run_id="feature_run",
                success_intent_only=False,
                only_pass_height=True,
                enabled_tasks=make_enabled_tasks(
                    action_intent=False,
                    pass_intent=False,
                    success_intent=False,
                    pass_success=False,
                    pass_height=True,
                    outcome_scoring=False,
                    outcome_conceding=False,
                    failure_receiver=False,
                ),
                trained_tasks=["pass_height"],
                intended_receiver_mode="angle_only",
                target_family=None,
                return_type="disc_0.9",
                use_v_edge_features=True,
                pass_height_ipw=False,
                outcome_scoring_trial=None,
                outcome_conceding_trial=None,
                xy_only=None,
                possessor_aware=None,
                keeper_aware=None,
                ball_z_aware=None,
                poss_vel_aware=None,
                poss_rel_vel_aware=None,
                accel_aware=None,
                offside_aware=None,
                extend_features=None,
                lane_survival=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, trained_model_ids, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--task") + 1], "pass_height")
        self.assertIn("pass_height", trained_model_ids)
        self.assertNotIn("--model-variant", commands[0])

    def test_build_training_commands_emit_no_offside_flag(self) -> None:
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
                poss_rel_vel_aware=None,
                accel_aware=None,
                offside_aware=False,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertFalse(feature_flags["offside_aware"])
        self.assertIn("--no-offside", commands[0])
        self.assertNotIn("--offside", commands[0])

    def test_build_training_commands_default_velocity_flags(self) -> None:
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
                poss_rel_vel_aware=None,
                accel_aware=None,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertTrue(feature_flags["poss_vel_aware"])
        self.assertFalse(feature_flags["poss_rel_vel_aware"])
        self.assertIn("--poss_vel_aware", commands[0])
        self.assertNotIn("--poss_rel_vel_aware", commands[0])

    def test_build_training_commands_default_feature_ablation_flags(self) -> None:
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
                poss_rel_vel_aware=None,
                poss_geometry_aware=None,
                goal_features_aware=None,
                goal_nodes_aware=None,
                accel_aware=None,
                offside_aware=None,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertTrue(feature_flags["poss_geometry_aware"])
        self.assertTrue(feature_flags["goal_features_aware"])
        self.assertTrue(feature_flags["goal_nodes_aware"])
        self.assertFalse(feature_flags["lane_survival"])
        self.assertNotIn("--no-poss-geometry", commands[0])
        self.assertNotIn("--no-goal-features", commands[0])
        self.assertNotIn("--no-goal-nodes", commands[0])
        self.assertIn("--no-lane-survival", commands[0])

    def test_build_training_commands_emit_selected_lane_survival_mode(self) -> None:
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
                poss_rel_vel_aware=None,
                poss_geometry_aware=None,
                goal_features_aware=None,
                goal_nodes_aware=None,
                accel_aware=None,
                offside_aware=None,
                extend_features=None,
                lane_survival_mode="top_25",
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertTrue(feature_flags["lane_survival"])
        self.assertEqual(feature_flags["lane_survival_mode"], "top_25")
        self.assertIn("--lane-survival", commands[0])
        self.assertEqual(commands[0][commands[0].index("--lane-survival") + 1], "top_25")
        self.assertNotIn("--no-lane-survival", commands[0])

    def test_lane_survival_mode_parser_accepts_max_and_top_n_only(self) -> None:
        self.assertEqual(train_wrapper.parse_lane_survival_mode("max"), "max")
        self.assertEqual(train_wrapper.parse_lane_survival_mode("top_10"), "top_10")
        with self.assertRaises(argparse.ArgumentTypeError):
            train_wrapper.parse_lane_survival_mode("top_0")
        with self.assertRaises(argparse.ArgumentTypeError):
            train_wrapper.parse_lane_survival_mode("top10")

    def test_build_training_commands_emit_feature_ablation_flags(self) -> None:
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
                poss_rel_vel_aware=None,
                poss_geometry_aware=False,
                goal_features_aware=False,
                goal_nodes_aware=False,
                accel_aware=None,
                offside_aware=None,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertFalse(feature_flags["poss_geometry_aware"])
        self.assertFalse(feature_flags["goal_features_aware"])
        self.assertFalse(feature_flags["goal_nodes_aware"])
        self.assertIn("--no-poss-geometry", commands[0])
        self.assertIn("--no-goal-features", commands[0])
        self.assertIn("--no-goal-nodes", commands[0])

    def test_build_training_commands_forward_split_velocity_flags(self) -> None:
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
                poss_vel_aware=False,
                poss_rel_vel_aware=True,
                accel_aware=None,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, feature_flags = train_wrapper.build_training_commands(args)

        self.assertFalse(feature_flags["poss_vel_aware"])
        self.assertTrue(feature_flags["poss_rel_vel_aware"])
        self.assertNotIn("--poss_vel_aware", commands[0])
        self.assertIn("--poss_rel_vel_aware", commands[0])

    def test_build_training_commands_emit_possessor_masked_velocity_edge_flag(self) -> None:
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
                v_edge_feature_mode="no_poss",
                use_v_edge_features=True,
                mask_possessor_v_edge_features=True,
                outcome_scoring_trial=None,
                outcome_conceding_trial=None,
                xy_only=None,
                possessor_aware=None,
                keeper_aware=None,
                ball_z_aware=None,
                poss_vel_aware=None,
                poss_rel_vel_aware=None,
                accel_aware=None,
                extend_features=None,
            )

            with (
                patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"),
                patch.object(train_wrapper, "resolve_feature_root", return_value=feature_root),
            ):
                commands, _, _, _, _ = train_wrapper.build_training_commands(args)

        self.assertIn("--v-edge-features-no-poss", commands[0])
        self.assertNotIn("--no-v-edge-features", commands[0])

    def test_wrapper_main_records_accel_flag_in_bundle_metadata(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            device=None,
            pin_memory=None,
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

    def test_wrapper_forwards_runtime_flags_to_train(self) -> None:
        captured_command: list[str] = []
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            device="cuda:0",
            pin_memory=False,
            early_stopping=True,
            early_stopping_patience=10,
            early_stopping_min_epochs=30,
            early_stopping_min_delta=1e-5,
            success_intent_only=False,
            target_family="goal",
            return_type="disc_0.9",
        )

        def capture_run(command: list[str], **_kwargs: object) -> None:
            captured_command.extend(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "bundle_under_test"
            with (
                patch.object(train_wrapper, "parse_args", return_value=cli_args),
                patch.object(
                    train_wrapper,
                    "build_training_commands",
                    return_value=(
                        [["--task", "pass_intent", "--run-id", "pass_run"]],
                        {"pass_intent": "pass_intent/pass_run"},
                        "angle_only",
                        "feature_run",
                        train_wrapper.WRAPPER_FEATURE_DEFAULTS,
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
                patch.object(train_wrapper, "write_run_metadata"),
                patch.object(train_wrapper.subprocess, "run", side_effect=capture_run),
            ):
                train_wrapper.main()

        self.assertIn("--device", captured_command)
        self.assertEqual(captured_command[captured_command.index("--device") + 1], "cuda:0")
        self.assertIn("--no-pin-memory", captured_command)
        self.assertIn("--early-stopping-patience", captured_command)
        self.assertEqual(captured_command[captured_command.index("--early-stopping-patience") + 1], "10")
        self.assertIn("--early-stopping-min-epochs", captured_command)
        self.assertEqual(captured_command[captured_command.index("--early-stopping-min-epochs") + 1], "30")
        self.assertIn("--early-stopping-min-delta", captured_command)
        self.assertEqual(captured_command[captured_command.index("--early-stopping-min-delta") + 1], "1e-05")
        self.assertNotIn("--no-early-stopping", captured_command)

    def test_wrapper_records_failed_metadata_for_access_violation(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            device=None,
            pin_memory=False,
            success_intent_only=False,
            target_family="goal",
            return_type="disc_0.9",
        )

        def capture_write_run_metadata(_root: Path, payload: dict[str, object]) -> None:
            captured_metadata.update(payload)

        def fail_run(command: list[str], **_kwargs: object) -> None:
            raise train_wrapper.subprocess.CalledProcessError(3221225477, command)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_root = Path(tmpdir) / "bundle_under_test"
            with (
                patch.object(train_wrapper, "parse_args", return_value=cli_args),
                patch.object(
                    train_wrapper,
                    "build_training_commands",
                    return_value=(
                        [["--task", "action_intent", "--run-id", "action_run"]],
                        {"action_intent": "action_intent/action_run"},
                        "model",
                        "feature_run",
                        train_wrapper.WRAPPER_FEATURE_DEFAULTS,
                    ),
                ),
                patch.object(train_wrapper, "get_model_bundle_root", return_value=bundle_root),
                patch.object(train_wrapper, "load_model_bundle_metadata", return_value={}),
                patch.object(train_wrapper, "write_run_metadata", side_effect=capture_write_run_metadata),
                patch.object(train_wrapper.subprocess, "run", side_effect=fail_run),
            ):
                with self.assertRaises(train_wrapper.subprocess.CalledProcessError):
                    train_wrapper.main()

        self.assertEqual(captured_metadata["status"], "failed")
        self.assertEqual(captured_metadata["failed_model_id"], "action_intent/action_run")
        self.assertEqual(captured_metadata["returncode"], 3221225477)
        self.assertIn("0xC0000005", captured_metadata["returncode_description"])
        self.assertIn("crash.log", captured_metadata["failed_crash_log"])

    def test_wrapper_main_merges_existing_bundle_metadata_for_partial_outcome_rerun(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9", "in_3"],
            use_v_edge_features=True,
            device=None,
            pin_memory=None,
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
        self.assertEqual(captured_metadata["diagnostic_target"], "goal_next10")
        self.assertEqual(captured_metadata["diagnostic_return_type"], "next_10")
        self.assertEqual(captured_metadata["diagnostic_feature_run_id"], "feature_run")
        self.assertEqual(captured_metadata["success_intent_label_source"], "receiver_id")
        self.assertEqual(captured_metadata["success_intent_training_filter"], "successful_pass_actions")
        self.assertEqual(len(captured_metadata["commands"]), 2)

    def test_wrapper_main_records_external_pass_intent_source_in_bundle_metadata(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            device=None,
            pin_memory=None,
            success_intent_only=False,
            target_family=None,
            return_type="disc_0.9",
            pass_intent_model_id="pass_intent/source",
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
                        [["--task", "pass_success", "--run-id", "pass_success_new", "--ipw_model_id", "pass_intent/source"]],
                        {"pass_success": "pass_success/new"},
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

        self.assertEqual(
            captured_metadata["model_ids"],
            {
                "pass_intent": "pass_intent/source",
                "pass_success": "pass_success/new",
            },
        )
        self.assertEqual(captured_metadata["trained_tasks"], ["pass_success"])
        self.assertEqual(captured_metadata["source_model_ids"], {"pass_intent": "pass_intent/source"})
        self.assertEqual(captured_metadata["batch_sizes"], {"pass_success": 512})
        self.assertTrue(captured_metadata["pass_success_ipw"])

    def test_wrapper_main_records_effective_batch_sizes_in_bundle_metadata(self) -> None:
        captured_metadata: dict[str, object] = {}
        cli_args = SimpleNamespace(
            bundle_id="bundle_under_test",
            available_intended_receiver_modes=["original", "angle_only"],
            available_return_types=["disc_0.9"],
            use_v_edge_features=True,
            device=None,
            pin_memory=None,
            success_intent_only=False,
            target_family="goal",
            return_type="disc_0.9",
            batch_size=384,
            pass_success_batch_size=640,
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
                        [
                            ["--task", "pass_success", "--run-id", "pass_success_new"],
                            ["--task", "outcome_scoring", "--run-id", "outcome_scoring_new"],
                        ],
                        {
                            "pass_success": "pass_success/new",
                            "outcome_scoring": "outcome_scoring/new",
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
                patch.object(train_wrapper, "load_model_bundle_metadata", return_value={}),
                patch.object(
                    train_wrapper,
                    "derive_bundle_shared_context",
                    return_value=make_bundle_shared_context(target_family="goal"),
                ),
                patch.object(train_wrapper, "load_feature_run_metadata", return_value={}),
                patch.object(train_wrapper, "write_run_metadata", side_effect=capture_write_run_metadata),
                patch.object(train_wrapper.subprocess, "run"),
            ):
                train_wrapper.main()

        self.assertEqual(
            captured_metadata["batch_sizes"],
            {
                "pass_success": 640,
                "outcome_scoring": 384,
            },
        )

    def test_derive_bundle_shared_context_ignores_pass_intent_target_metadata(self) -> None:
        cli_args = SimpleNamespace(
            return_type="disc_0.9",
            target_family="goal",
            intended_receiver_mode="angle_only",
            use_v_edge_features=True,
        )
        model_records = {
            "pass_intent": make_model_record("pass_intent", return_type="next_5", target_family="xt"),
            "pass_success": make_model_record("pass_success", return_type="disc_0.9", target_family="goal"),
        }

        with patch.object(train_wrapper, "get_model_records", return_value=model_records):
            shared = train_wrapper.derive_bundle_shared_context(
                {
                    "pass_intent": "pass_intent/source",
                    "pass_success": "pass_success/new",
                },
                cli_args,
                "feature_run",
            )

        self.assertEqual(shared["return_type"], "disc_0.9")
        self.assertEqual(shared["target_family"], "goal")
        self.assertEqual(shared["feature_run_id"], "feature_run")
        self.assertEqual(shared["intended_receiver_mode"], "angle_only")

    def test_derive_bundle_shared_context_uses_existing_outcome_target_family_when_cli_omits_it(self) -> None:
        cli_args = SimpleNamespace(
            return_type="disc_0.9",
            target_family=None,
            intended_receiver_mode="angle_only",
            use_v_edge_features=True,
        )
        model_records = {
            "pass_intent": make_model_record("pass_intent", return_type="next_5", target_family="epv"),
            "pass_success": make_model_record("pass_success", return_type="disc_0.9", target_family="goal"),
            "outcome_scoring": make_model_record("outcome_scoring", return_type="in_3", target_family="xt"),
            "outcome_conceding": make_model_record("outcome_conceding", return_type="in_3", target_family="xt"),
        }

        with patch.object(train_wrapper, "get_model_records", return_value=model_records):
            shared = train_wrapper.derive_bundle_shared_context(
                {
                    "pass_intent": "pass_intent/source",
                    "pass_success": "pass_success/new",
                    "outcome_scoring": "outcome_scoring/old",
                    "outcome_conceding": "outcome_conceding/old",
                },
                cli_args,
                "feature_run",
            )

        self.assertEqual(shared["return_type"], "disc_0.9")
        self.assertIsNone(shared["target_family"])
        self.assertEqual(shared["intended_receiver_mode"], "angle_only")
        self.assertEqual(
            shared["source_target_families"],
            {
                "pass_intent": "epv",
                "pass_success": "goal",
                "outcome_scoring": "xt",
                "outcome_conceding": "xt",
            },
        )

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

    def test_validate_model_record_consistency_rejects_mixed_feature_run_by_default(self) -> None:
        model_records = {
            "pass_intent": make_model_record("pass_intent", feature_run_id="feature_old"),
            "pass_success": make_model_record("pass_success", feature_run_id="feature_new"),
        }

        with self.assertRaises(ValueError):
            model_utils.validate_model_record_consistency(model_records)

    def test_validate_model_record_consistency_accepts_mixed_feature_run_when_relaxed(self) -> None:
        model_records = {
            "pass_intent": make_model_record(
                "pass_intent",
                feature_run_id="feature_old",
                node_in_dim=20,
                edge_in_dim=2,
                add_v_edge_features=False,
            ),
            "pass_success": make_model_record(
                "pass_success",
                feature_run_id="feature_new",
                node_in_dim=25,
                edge_in_dim=4,
                add_v_edge_features=True,
            ),
        }

        shared = model_utils.validate_model_record_consistency(model_records, require_feature_run_id=False)

        self.assertIsNone(shared["feature_run_id"])
        self.assertEqual(
            shared["source_feature_run_ids"],
            {"pass_intent": "feature_old", "pass_success": "feature_new"},
        )
        self.assertEqual(
            shared["graph_schema"],
            {
                "node_in_dim": 25,
                "edge_in_dim": 4,
                "add_v_edge_features": True,
                "add_relative_speed_edge_features": False,
            },
        )

    def test_validate_model_record_consistency_accepts_all_mixed_metadata_when_relaxed(self) -> None:
        model_records = {
            "action_intent": make_model_record(
                "action_intent",
                feature_run_id="feature_action",
                intended_receiver_mode="model",
                return_type="next_5",
            ),
            "pass_success": make_model_record(
                "pass_success",
                feature_run_id="feature_pass",
                intended_receiver_mode="original",
                return_type="disc_0.9",
            ),
            "outcome_scoring": make_model_record(
                "outcome_scoring",
                feature_run_id="feature_score",
                intended_receiver_mode="model",
                return_type="disc_0.5",
                target_family="epv",
            ),
            "outcome_conceding": make_model_record(
                "outcome_conceding",
                feature_run_id="feature_concede",
                intended_receiver_mode="angle_only",
                return_type="in_3",
                target_family="xt",
            ),
        }

        shared = model_utils.validate_model_record_consistency(
            model_records,
            require_feature_run_id=False,
            require_intended_receiver_mode=False,
            require_return_type=False,
            require_target_family=False,
        )

        self.assertIsNone(shared["feature_run_id"])
        self.assertIsNone(shared["intended_receiver_mode"])
        self.assertIsNone(shared["return_type"])
        self.assertIsNone(shared["target_family"])
        self.assertEqual(
            shared["source_intended_receiver_modes"],
            {
                "action_intent": "model",
                "pass_success": "original",
                "outcome_scoring": "model",
                "outcome_conceding": "angle_only",
            },
        )
        self.assertEqual(shared["source_return_types"]["outcome_scoring"], "disc_0.5")
        self.assertEqual(shared["source_target_families"]["outcome_conceding"], "xt")

    def test_resolve_runtime_return_type_prefers_outcome_scoring_when_outcomes_disagree(self) -> None:
        shared = {
            "return_type": None,
            "source_return_types": {
                "pass_success": "disc_0.9",
                "outcome_scoring": "disc_0.5",
                "outcome_conceding": "in_3",
            },
        }

        self.assertEqual(model_utils.resolve_runtime_return_type(shared), "disc_0.5")

    def test_resolve_model_selection_relaxed_accepts_bundle_plus_override_mixed_feature_runs(self) -> None:
        resolved_model_ids = {
            "pass_intent": "pass_intent/old",
            "pass_success": "pass_success/new",
        }
        bundle = {
            "bundle_id": "bundle_under_test",
            "feature_run_id": "feature_bundle",
            "intended_receiver_mode": "angle_only",
            "return_type": "disc_0.9",
            "model_ids": {"pass_success": "pass_success/new"},
        }
        model_records = {
            "pass_intent": make_model_record("pass_intent", feature_run_id="feature_old"),
            "pass_success": make_model_record("pass_success", feature_run_id="feature_new"),
        }

        with (
            patch.object(model_utils, "resolve_bundle_model_ids", return_value=(resolved_model_ids, bundle)),
            patch.object(model_utils, "get_model_records", return_value=model_records),
        ):
            selected_model_ids, shared, selected_bundle = model_utils.resolve_model_selection(
                required_tasks=["pass_intent", "pass_success"],
                bundle_id="bundle_under_test",
                explicit_model_ids={"pass_intent": "pass_intent/old"},
                require_feature_run_id=False,
                require_intended_receiver_mode=False,
                require_return_type=False,
                require_target_family=False,
            )

        self.assertEqual(selected_model_ids, resolved_model_ids)
        self.assertEqual(selected_bundle, bundle)
        self.assertEqual(shared["feature_run_id"], "feature_bundle")
        self.assertEqual(shared["bundle_feature_run_id"], "feature_bundle")
        self.assertEqual(
            shared["source_feature_run_ids"],
            {"pass_intent": "feature_old", "pass_success": "feature_new"},
        )

    def test_resolve_model_selection_relaxed_accepts_bundle_metadata_mismatches(self) -> None:
        resolved_model_ids = {
            "outcome_scoring": "outcome_scoring/score",
            "outcome_conceding": "outcome_conceding/concede",
        }
        bundle = {
            "bundle_id": "bundle_under_test",
            "feature_run_id": "feature_bundle",
            "intended_receiver_mode": "original",
            "return_type": "next_10",
            "target_family": "goal",
            "model_ids": resolved_model_ids,
        }
        model_records = {
            "outcome_scoring": make_model_record(
                "outcome_scoring",
                feature_run_id="feature_score",
                intended_receiver_mode="model",
                return_type="disc_0.5",
                target_family="epv",
            ),
            "outcome_conceding": make_model_record(
                "outcome_conceding",
                feature_run_id="feature_concede",
                intended_receiver_mode="angle_only",
                return_type="in_3",
                target_family="xt",
            ),
        }

        with (
            patch.object(model_utils, "resolve_bundle_model_ids", return_value=(resolved_model_ids, bundle)),
            patch.object(model_utils, "get_model_records", return_value=model_records),
        ):
            selected_model_ids, shared, selected_bundle = model_utils.resolve_model_selection(
                required_tasks=["outcome_scoring", "outcome_conceding"],
                bundle_id="bundle_under_test",
                require_feature_run_id=False,
                require_intended_receiver_mode=False,
                require_return_type=False,
                require_target_family=False,
            )

        self.assertEqual(selected_model_ids, resolved_model_ids)
        self.assertEqual(selected_bundle, bundle)
        self.assertEqual(shared["feature_run_id"], "feature_bundle")
        self.assertEqual(shared["intended_receiver_mode"], "original")
        self.assertEqual(shared["return_type"], "next_10")
        self.assertEqual(shared["target_family"], "goal")
        self.assertEqual(shared["source_return_types"]["outcome_scoring"], "disc_0.5")
        self.assertEqual(shared["source_target_families"]["outcome_conceding"], "xt")

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

    def test_validate_model_graph_schemas_aggregates_compatible_requirements(self) -> None:
        small_model = SimpleNamespace(args={"node_in_dim": 20, "edge_in_dim": 2, "add_v_edge_features": False})
        large_model = SimpleNamespace(args={"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True})

        graph_schema = model_utils.validate_model_graph_schemas({"small": small_model, "large": large_model})

        self.assertEqual(
            graph_schema,
            {
                "node_in_dim": 25,
                "edge_in_dim": 4,
                "add_v_edge_features": True,
                "add_relative_speed_edge_features": False,
            },
        )

    def test_feature_signature_distinguishes_possessor_masked_velocity_edge_mode(self) -> None:
        base_args = {
            "node_in_dim": 25,
            "edge_in_dim": 4,
            "add_v_edge_features": True,
        }

        all_signature = model_utils.extract_model_feature_signature({**base_args, "v_edge_feature_mode": "all"})
        no_poss_signature = model_utils.extract_model_feature_signature({**base_args, "v_edge_feature_mode": "no_poss"})

        self.assertEqual(all_signature["edge_in_dim"], 4)
        self.assertTrue(all_signature["add_v_edge_features"])
        self.assertNotEqual(all_signature["v_edge_feature_mode"], no_poss_signature["v_edge_feature_mode"])

    def test_feature_signature_records_feature_ablation_flags(self) -> None:
        signature = model_utils.extract_model_feature_signature(
            {
                "node_in_dim": 25,
                "edge_in_dim": 4,
                "poss_geometry_aware": False,
                "goal_features_aware": False,
                "goal_nodes_aware": False,
            }
        )

        self.assertFalse(signature["poss_geometry_aware"])
        self.assertFalse(signature["goal_features_aware"])
        self.assertFalse(signature["goal_nodes_aware"])

    def test_feature_signature_records_lane_survival_flag(self) -> None:
        signature = model_utils.extract_model_feature_signature(
            {
                "node_in_dim": 26,
                "edge_in_dim": 4,
                "lane_survival": True,
            }
        )

        self.assertTrue(signature["lane_survival"])
        self.assertEqual(signature["lane_survival_mode"], "max")
        self.assertEqual(signature["node_in_dim"], 26)

    def test_feature_signature_records_top_n_lane_survival_mode(self) -> None:
        signature = model_utils.extract_model_feature_signature(
            {
                "node_in_dim": 26,
                "edge_in_dim": 4,
                "lane_survival": True,
                "lane_survival_mode": "top_10",
            }
        )

        self.assertEqual(signature["lane_survival_mode"], "top_10")

    def test_no_poss_velocity_edge_mode_requires_four_edge_features(self) -> None:
        with self.assertRaises(ValueError):
            model_utils.infer_training_edge_schema(
                {"edge_in_dim": 2, "add_v_edge_features": False},
                v_edge_feature_mode="no_poss",
            )

        schema = model_utils.infer_training_edge_schema(
            {"edge_in_dim": 4, "add_v_edge_features": True},
            v_edge_feature_mode="no_poss",
        )
        self.assertEqual(
            schema,
            {"edge_in_dim": 4, "add_v_edge_features": True, "add_relative_speed_edge_features": False},
        )

    def test_relative_speed_edge_mode_requires_alignment_and_five_edge_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "Relative-speed edge features require"):
            model_utils.infer_training_edge_schema(
                {"edge_in_dim": 5, "add_v_edge_features": True, "add_relative_speed_edge_features": True},
                v_edge_feature_mode="none",
                relative_speed_edge_feature_mode="all",
            )

        with self.assertRaisesRegex(ValueError, "required_edge_dim=5"):
            model_utils.infer_training_edge_schema(
                {"edge_in_dim": 4, "add_v_edge_features": True, "add_relative_speed_edge_features": False},
                v_edge_feature_mode="all",
                relative_speed_edge_feature_mode="all",
            )

        schema = model_utils.infer_training_edge_schema(
            {"edge_in_dim": 5, "add_v_edge_features": True, "add_relative_speed_edge_features": True},
            v_edge_feature_mode="all",
            relative_speed_edge_feature_mode="all",
        )
        self.assertEqual(
            schema,
            {"edge_in_dim": 5, "add_v_edge_features": True, "add_relative_speed_edge_features": True},
        )

    def test_pass_success_runtime_schema_adds_lane_survival_node_dim(self) -> None:
        with patch.object(
            train_wrapper,
            "infer_feature_graph_schema",
            return_value={"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
        ):
            schema = train_wrapper.resolve_pass_success_runtime_schema(
                Path("feature_root"),
                "all",
                lane_survival=True,
            )

        self.assertEqual(schema["node_in_dim"], 26)
        self.assertEqual(schema["edge_in_dim"], 4)

    def test_adapt_batch_graphs_for_model_truncates_extra_edge_features(self) -> None:
        batch = Batch.from_data_list([make_velocity_edge_graph()])
        adapted = model_utils.adapt_batch_graphs_for_model(
            batch,
            {"node_in_dim": 25, "edge_in_dim": 2, "v_edge_feature_mode": "none"},
            context="IPW model 'pass_intent/old'",
        )

        self.assertEqual(adapted.x.shape[1], 25)
        self.assertEqual(adapted.edge_attr.shape[1], 2)

    def test_adapt_batch_graphs_for_model_rejects_small_edge_schema(self) -> None:
        batch = Batch.from_data_list([make_graph(edge_dim=2)])

        with self.assertRaisesRegex(ValueError, "requires edge_in_dim=4"):
            model_utils.adapt_batch_graphs_for_model(
                batch,
                {"node_in_dim": 25, "edge_in_dim": 4, "v_edge_feature_mode": "all"},
                context="IPW model 'pass_intent/old'",
            )

    def test_adapt_batch_graphs_for_model_rejects_small_node_schema(self) -> None:
        batch = Batch.from_data_list([make_graph(node_dim=19, edge_dim=4)])

        with self.assertRaisesRegex(ValueError, "requires node_in_dim=25"):
            model_utils.adapt_batch_graphs_for_model(
                batch,
                {"node_in_dim": 25, "edge_in_dim": 4, "v_edge_feature_mode": "all"},
                context="IPW model 'pass_intent/old'",
            )

    def test_adapt_batch_graphs_for_model_masks_no_poss_velocity_edges(self) -> None:
        batch = Batch.from_data_list([make_velocity_edge_graph()])
        adapted = model_utils.adapt_batch_graphs_for_model(
            batch,
            {"node_in_dim": 25, "edge_in_dim": 4, "v_edge_feature_mode": "no_poss"},
            context="IPW model 'pass_intent/old'",
        )

        self.assertTrue(torch.equal(adapted.edge_attr[:2, 2:4], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(adapted.edge_attr[2:, 2:4], torch.tensor([[0.6, 0.7], [0.8, 0.9]])))

    def test_adapt_batch_graphs_for_model_masks_no_poss_relative_speed_edges(self) -> None:
        batch = Batch.from_data_list([make_relative_speed_edge_graph()])
        adapted = model_utils.adapt_batch_graphs_for_model(
            batch,
            {
                "node_in_dim": 25,
                "edge_in_dim": 5,
                "v_edge_feature_mode": "all",
                "relative_speed_edge_feature_mode": "no_poss",
            },
            context="IPW model 'pass_intent/old'",
        )

        self.assertTrue(torch.equal(adapted.edge_attr[:2, 4], torch.zeros(2)))
        self.assertTrue(torch.equal(adapted.edge_attr[2:, 4], torch.tensor([3.0, 4.0])))

    def test_estimate_propensity_adapts_extra_edge_features_before_forward(self) -> None:
        class FakePropensityModel:
            args = {"node_in_dim": 25, "edge_in_dim": 2, "v_edge_feature_mode": "none"}

            def __call__(self, batch_graphs):
                self.seen_edge_dim = int(batch_graphs.edge_attr.shape[1])
                return torch.zeros(batch_graphs.x.shape[0], dtype=torch.float32)

        label = torch.zeros(len(LABEL_COLUMNS), dtype=torch.float32)
        label[5] = 0
        fake_model = FakePropensityModel()

        with patch.object(model_utils, "load_model", return_value=fake_model):
            likelihoods = model_utils.estimate_propensity(
                [(make_velocity_edge_graph(), label, torch.tensor(1.0))],
                model_id="pass_intent/old",
                device="cpu",
                pin_memory=False,
            )

        self.assertEqual(fake_model.seen_edge_dim, 2)
        self.assertEqual(likelihoods.shape, (1,))

    def test_validate_feature_graph_schema_checks_node_dimension(self) -> None:
        with self.assertRaises(ValueError):
            model_utils.validate_feature_graph_schema(
                {"node_in_dim": 20, "edge_in_dim": 4, "add_v_edge_features": True},
                {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
            )

    def test_resolve_runtime_feature_run_context_explicit_feature_run_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_old")
            self._make_runtime_feature_run(root, "feature_new")
            metadata = {
                "feature_old": {
                    "created_at": "2026-04-01T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
                "feature_new": {
                    "created_at": "2026-04-02T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
            }
            schemas = {
                "feature_old": {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                "feature_new": {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
            }

            with (
                patch.object(model_utils, "FEATURE_RUNS_DIR", root),
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
                patch.object(model_utils, "infer_feature_graph_schema", side_effect=lambda path: schemas[Path(path).parent.name]),
            ):
                runtime = model_utils.resolve_runtime_feature_run_context(
                    "feature_old",
                    {"source_feature_run_ids": {"pass_success": "feature_old", "outcome_scoring": "feature_new"}},
                    None,
                    "model",
                    {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                )

        self.assertEqual(runtime["feature_run_id"], "feature_old")
        self.assertEqual(runtime["selection"], "explicit")

    def test_resolve_runtime_feature_run_context_uses_newest_compatible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_old")
            self._make_runtime_feature_run(root, "feature_new")
            metadata = {
                "feature_old": {
                    "created_at": "2026-04-01T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
                "feature_new": {
                    "created_at": "2026-04-02T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
            }
            schemas = {
                "feature_old": {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                "feature_new": {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
            }

            with (
                patch.object(model_utils, "FEATURE_RUNS_DIR", root),
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
                patch.object(model_utils, "infer_feature_graph_schema", side_effect=lambda path: schemas[Path(path).parent.name]),
            ):
                runtime = model_utils.resolve_runtime_feature_run_context(
                    None,
                    {"source_feature_run_ids": {"pass_success": "feature_old", "outcome_scoring": "feature_new"}},
                    None,
                    "model",
                    {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                )

        self.assertEqual(runtime["feature_run_id"], "feature_new")
        self.assertEqual(runtime["selection"], "newest_compatible")

    def test_resolve_runtime_feature_run_context_falls_back_to_all_feature_runs_and_prefers_model_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_old")
            feature_new = self._make_runtime_feature_run(root, "feature_new", mode="original")
            model_utils.get_resolved_action_dir("model", root=feature_new).mkdir(parents=True, exist_ok=True)
            metadata = {
                "feature_old": {
                    "created_at": "2026-04-01T00:00:00",
                    "intended_receiver_modes": ["model"],
                },
                "feature_new": {
                    "created_at": "2026-04-02T00:00:00",
                    "intended_receiver_modes": ["original", "model"],
                },
            }
            schemas = {
                "feature_old": {"node_in_dim": 25, "edge_in_dim": 2, "add_v_edge_features": False},
                "feature_new": {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
            }

            with (
                patch.object(model_utils, "FEATURE_RUNS_DIR", root),
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
                patch.object(model_utils, "infer_feature_graph_schema", side_effect=lambda path: schemas[Path(path).parent.name]),
            ):
                runtime = model_utils.resolve_runtime_feature_run_context(
                    None,
                    {"source_feature_run_ids": {"pass_success": "feature_old"}},
                    None,
                    None,
                    {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                )

        self.assertEqual(runtime["feature_run_id"], "feature_new")
        self.assertEqual(runtime["intended_receiver_mode"], "model")
        self.assertEqual(runtime["selection"], "newest_compatible_all_feature_runs")

    def test_resolve_runtime_feature_run_context_rejects_incompatible_graph_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_old")
            metadata = {
                "feature_old": {
                    "created_at": "2026-04-01T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
            }
            schemas = {"feature_old": {"node_in_dim": 25, "edge_in_dim": 2, "add_v_edge_features": False}}

            with (
                patch.object(model_utils, "FEATURE_RUNS_DIR", root),
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
                patch.object(model_utils, "infer_feature_graph_schema", side_effect=lambda path: schemas[Path(path).parent.name]),
            ):
                with self.assertRaises(ValueError):
                    model_utils.resolve_runtime_feature_run_context(
                        None,
                        {"source_feature_run_ids": {"pass_success": "feature_old"}},
                        None,
                        "model",
                        {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                    )

    def test_validate_runtime_feature_run_rejects_missing_resolved_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_old", with_resolved_actions=False)
            metadata = {
                "feature_old": {
                    "created_at": "2026-04-01T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/source",
                },
            }

            with (
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
            ):
                with self.assertRaises(FileNotFoundError):
                    model_utils.validate_runtime_feature_run(
                        "feature_old",
                        "model",
                        {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                        {},
                    )

    def test_validate_runtime_feature_run_allows_model_mode_intended_receiver_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._make_runtime_feature_run(root, "feature_runtime")
            metadata = {
                "feature_runtime": {
                    "created_at": "2026-04-02T00:00:00",
                    "intended_receiver_modes": ["model"],
                    "intended_receiver_model_id": "success_intent/runtime",
                },
            }

            with (
                patch.object(model_utils, "resolve_feature_run_id", side_effect=lambda run_id, **_kwargs: str(run_id)),
                patch.object(model_utils, "get_feature_run_root", side_effect=lambda run_id: root / str(run_id)),
                patch.object(model_utils, "load_feature_run_metadata", side_effect=lambda run_id, required=False: metadata.get(str(run_id))),
                patch.object(
                    model_utils,
                    "infer_feature_graph_schema",
                    return_value={"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                ),
            ):
                runtime = model_utils.validate_runtime_feature_run(
                    "feature_runtime",
                    "model",
                    {"node_in_dim": 25, "edge_in_dim": 4, "add_v_edge_features": True},
                    {"feature_source": {"intended_receiver_model_id": "success_intent/source"}},
                )

        self.assertEqual(runtime["feature_run_id"], "feature_runtime")
        self.assertEqual(runtime["intended_receiver_mode"], "model")

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

    def test_filter_features_and_labels_preserves_offside_tail_without_extended_features(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "extend_features": False,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
        }
        graph = make_graph(node_dim=26)
        graph.x[:, 19:25] = 19
        graph.x[:, 25] = 25

        filtered_graphs, _ = filter_features_and_labels([graph], labels, args)
        filtered = filtered_graphs[0]

        self.assertTrue(torch.equal(filtered.x[:, 19:25], torch.zeros_like(filtered.x[:, 19:25])))
        self.assertTrue(torch.equal(filtered.x[:, 25], torch.full_like(filtered.x[:, 25], 25)))

    def test_filter_features_and_labels_zeroes_offside_tail_when_disabled(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "offside_aware": False,
            "extend_features": False,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
        }
        graph = make_graph(node_dim=26)
        graph.x[:, 19:25] = 19
        graph.x[:, 25] = 25

        filtered_graphs, _ = filter_features_and_labels([graph], labels, args)
        filtered = filtered_graphs[0]

        self.assertTrue(torch.equal(filtered.x[:, 19:25], torch.zeros_like(filtered.x[:, 19:25])))
        self.assertTrue(torch.equal(filtered.x[:, 25], torch.zeros_like(filtered.x[:, 25])))

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

    def test_action_dataset_appends_lane_survival_before_failure_intent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            cache_dir = Path(tmpdir) / "pc_xpass"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            graph = make_graph()
            graph.node_ids = ["possessor", "target"]
            torch.save([graph], feature_dir / "match_1.pt")
            torch.save(make_pass_labels(action_index=7, intent_index=1, success=0.0), label_dir / "match_1.pt")
            write_pc_xpass_lane_survival_cache(cache_dir, action_index=7, target_value=0.42)

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="failure_receiver",
                lane_survival=True,
                lane_survival_cache_dir=str(cache_dir),
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertEqual(graph.x.shape[1], 27)
        self.assertTrue(torch.allclose(graph.x[:, -2], torch.tensor([0.0, 0.42])))
        self.assertTrue(torch.equal(graph.x[:, -1], torch.tensor([0.0, 1.0])))

    def test_action_dataset_lane_survival_is_independent_of_xy_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            cache_dir = Path(tmpdir) / "pc_xpass"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            graph = make_graph()
            graph.node_ids = ["possessor", "target"]
            torch.save([graph], feature_dir / "match_1.pt")
            torch.save(make_pass_labels(action_index=7, intent_index=1), label_dir / "match_1.pt")
            write_pc_xpass_lane_survival_cache(cache_dir, action_index=7, target_value=0.73)

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="pass_intent",
                xy_only=True,
                lane_survival=True,
                lane_survival_cache_dir=str(cache_dir),
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertEqual(graph.x.shape[1], 26)
        self.assertTrue(torch.allclose(graph.x[:, -1], torch.tensor([0.0, 0.73])))
        self.assertTrue(torch.equal(graph.x[:, config.NODE_FEATURE_IS_POSSESSOR], torch.zeros(2)))

    def test_action_dataset_zeroes_poss_geometry_but_preserves_possessor_flag(self) -> None:
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
                poss_geometry_aware=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[:, 13], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(graph.x[:, 14:17], torch.zeros_like(graph.x[:, 14:17])))

    def test_action_dataset_zeroes_goal_features_only(self) -> None:
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
                goal_features_aware=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[:, 8], torch.full_like(graph.x[:, 8], 8)))
        self.assertTrue(torch.equal(graph.x[:, 9:12], torch.zeros_like(graph.x[:, 9:12])))
        self.assertTrue(torch.equal(graph.x[:, 12], torch.full_like(graph.x[:, 12], 12)))

    def test_action_dataset_no_goal_nodes_removes_goal_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_goal_node_graph()], feature_dir / "match_1.pt")
            torch.save(make_labels(), label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="action_intent",
                goal_nodes_aware=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, labels, _ = dataset[0]
        self.assertEqual(graph.x.shape[0], 2)
        self.assertEqual(graph.edge_index.shape[1], 2)
        self.assertEqual(graph.node_ids, ["possessor", "target"])
        self.assertTrue(torch.equal(graph.x[:, config.NODE_FEATURE_IS_GOAL], torch.zeros(2)))
        self.assertEqual(float(labels[LABEL_INDEX["n_players"]]), 2.0)

    def test_action_dataset_preserves_offside_tail_without_extended_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            graph = make_graph(node_dim=26)
            graph.x[:, 19:25] = 19
            graph.x[:, 25] = 25
            torch.save([graph], feature_dir / "match_1.pt")
            torch.save(make_labels(), label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="action_intent",
                extend_features=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[:, 19:25], torch.zeros_like(graph.x[:, 19:25])))
        self.assertTrue(torch.equal(graph.x[:, 25], torch.full_like(graph.x[:, 25], 25)))

    def test_action_dataset_zeroes_offside_tail_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            graph = make_graph(node_dim=26)
            graph.x[:, 19:25] = 19
            graph.x[:, 25] = 25
            torch.save([graph], feature_dir / "match_1.pt")
            torch.save(make_labels(), label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="action_intent",
                offside_aware=False,
                extend_features=False,
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[:, 19:25], torch.zeros_like(graph.x[:, 19:25])))
        self.assertTrue(torch.equal(graph.x[:, 25], torch.zeros_like(graph.x[:, 25])))

    def test_physical_xpass_candidate_targets_ignore_offside_tail(self) -> None:
        graph = make_physical_xpass_graph()

        self.assertEqual(_candidate_target_indices(graph), [1])

    def test_physical_xpass_simulation_inputs_use_stable_prefix_with_new_tail(self) -> None:
        graph = make_physical_xpass_graph()
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            target_player_index = list(kwargs["players"]).index("target_player")
            probabilities = np.full((1, kwargs["PLAYER_POS"].shape[1], AS_DEFAULT_N_ANGLES, 1), 0.1, dtype=float)
            probabilities[0, target_player_index, 0, 0] = 0.73
            return SimpleNamespace(player_cum_prob=probabilities, r_grid=np.array([10.0], dtype=float))

        result = compute_graph_player_cum_prob(
            graph,
            simulate_passes_fn=fake_simulate_passes,
            consider_teammates=False,
        )

        self.assertEqual(result["target"], 0.73)
        self.assertTrue(result.drop(index="target").isna().all())
        self.assertEqual(len(calls), AS_DEFAULT_N_V0)
        self.assertEqual(calls[0]["players"].tolist(), ["possessor", "target_player", "defender"])
        np.testing.assert_allclose(calls[0]["PLAYER_POS"][0, 0], [-12.5, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(calls[0]["PLAYER_POS"][0, 1], [17.5, 0.0, 2.0, 0.5])
        np.testing.assert_allclose(calls[0]["PLAYER_POS"][0, 2], [2.5, 11.0, -1.0, 0.0])
        np.testing.assert_allclose(calls[0]["BALL_POS"][0], [-12.5, 0.0])

    def test_action_dataset_splits_possessor_and_relative_velocity_masks(self) -> None:
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
                poss_vel_aware=False,
                poss_rel_vel_aware=True,
            )

        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[0, 5:9], torch.zeros(4)))
        self.assertTrue(torch.equal(graph.x[1, 5:9], torch.tensor([5.0, 6.0, 7.0, 8.0])))
        self.assertTrue(torch.equal(graph.x[:, 17], torch.full_like(graph.x[:, 17], 17)))
        self.assertTrue(torch.equal(graph.x[:, 18], torch.full_like(graph.x[:, 18], 18)))

    def test_action_dataset_default_masks_relative_velocity_only(self) -> None:
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
            )

        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[0, 5:9], torch.tensor([5.0, 6.0, 7.0, 8.0])))
        self.assertTrue(torch.equal(graph.x[:, 17:19], torch.zeros_like(graph.x[:, 17:19])))

    def test_action_dataset_can_mask_both_possessor_velocity_groups(self) -> None:
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
                poss_vel_aware=False,
                poss_rel_vel_aware=False,
            )

        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.x[0, 5:9], torch.zeros(4)))
        self.assertTrue(torch.equal(graph.x[:, 17:19], torch.zeros_like(graph.x[:, 17:19])))

    def test_filter_features_and_labels_supports_feature_ablation_flags(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "poss_rel_vel_aware": True,
            "poss_geometry_aware": False,
            "goal_features_aware": False,
            "goal_nodes_aware": True,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": True,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
        }

        filtered_graphs, _ = filter_features_and_labels([make_graph()], labels, args)
        graph = filtered_graphs[0]

        self.assertTrue(torch.equal(graph.x[:, 13], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(graph.x[:, 14:17], torch.zeros_like(graph.x[:, 14:17])))
        self.assertTrue(torch.equal(graph.x[:, 9:12], torch.zeros_like(graph.x[:, 9:12])))
        self.assertTrue(torch.equal(graph.x[:, 12], torch.full_like(graph.x[:, 12], 12)))

    def test_filter_features_and_labels_no_goal_nodes_removes_goal_edges(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "poss_rel_vel_aware": True,
            "poss_geometry_aware": True,
            "goal_features_aware": True,
            "goal_nodes_aware": False,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": True,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
        }

        filtered_graphs, filtered_labels = filter_features_and_labels([make_goal_node_graph()], labels, args)
        graph = filtered_graphs[0]

        self.assertEqual(graph.x.shape[0], 2)
        self.assertEqual(graph.edge_index.shape[1], 2)
        self.assertEqual(graph.node_ids, ["possessor", "target"])
        self.assertEqual(float(filtered_labels[0, LABEL_INDEX["n_players"]]), 2.0)

    def test_filter_features_and_labels_masks_only_possessor_velocity_edges(self) -> None:
        labels = make_labels()
        args = {
            "task": "action_intent",
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "extend_features": True,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 4,
            "v_edge_feature_mode": "no_poss",
        }

        source_graph = make_velocity_edge_graph()
        filtered_graphs, _ = filter_features_and_labels([source_graph], labels, args)
        graph = filtered_graphs[0]

        self.assertTrue(torch.equal(graph.edge_attr[:, :2], source_graph.edge_attr[:, :2]))
        self.assertTrue(torch.equal(graph.edge_attr[:2, 2:4], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(graph.edge_attr[2:, 2:4], source_graph.edge_attr[2:, 2:4]))

    def test_action_dataset_masks_only_possessor_velocity_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            source_graph = make_velocity_edge_graph()
            torch.save([source_graph], feature_dir / "match_1.pt")
            torch.save(make_labels(), label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="action_intent",
                edge_in_dim=4,
                v_edge_feature_mode="no_poss",
            )

        self.assertEqual(len(dataset), 1)
        graph, _, _ = dataset[0]
        self.assertTrue(torch.equal(graph.edge_attr[:, :2], source_graph.edge_attr[:, :2]))
        self.assertTrue(torch.equal(graph.edge_attr[:2, 2:4], torch.zeros((2, 2))))
        self.assertTrue(torch.equal(graph.edge_attr[2:, 2:4], source_graph.edge_attr[2:, 2:4]))

    def test_action_dataset_loads_goal_next10_diagnostics_from_label_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels_disc"
            diagnostic_label_dir = Path(tmpdir) / "labels_next10"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_graph()], feature_dir / "match_1.pt")

            selected_labels = make_legacy_labels()
            selected_labels[:, LABEL_INDEX["scores"]] = 0.729
            diagnostic_labels = make_legacy_labels()
            diagnostic_labels[:, LABEL_INDEX["scores"]] = 1.0
            diagnostic_labels[:, LABEL_INDEX["concedes"]] = 0.0
            torch.save(selected_labels, label_dir / "match_1.pt")
            torch.save(diagnostic_labels, diagnostic_label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                diagnostic_label_dir=str(diagnostic_label_dir),
                task="outcome_scoring",
            )

        self.assertAlmostEqual(float(dataset.labels[0, LABEL_INDEX["scores"]]), 0.729, places=6)
        self.assertEqual(float(dataset.labels[0, LABEL_INDEX["scores_goal_next10"]]), 1.0)
        self.assertEqual(float(dataset.labels[0, LABEL_INDEX["concedes_goal_next10"]]), 0.0)

    def test_action_dataset_accepts_embedded_goal_next10_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels_disc"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_graph()], feature_dir / "match_1.pt")

            labels = make_labels()
            labels[:, LABEL_INDEX["scores"]] = 0.729
            labels[:, LABEL_INDEX["scores_goal_next10"]] = 1.0
            torch.save(labels, label_dir / "match_1.pt")

            dataset = ActionDataset(
                ["match_1"],
                feature_dir=str(feature_dir),
                label_dir=str(label_dir),
                task="outcome_scoring",
            )

        self.assertAlmostEqual(float(dataset.labels[0, LABEL_INDEX["scores"]]), 0.729, places=6)
        self.assertEqual(float(dataset.labels[0, LABEL_INDEX["scores_goal_next10"]]), 1.0)

    def test_action_dataset_requires_goal_next10_diagnostics_for_outcome_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels_disc"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_graph()], feature_dir / "match_1.pt")
            torch.save(make_legacy_labels(), label_dir / "match_1.pt")

            with self.assertRaises(FileNotFoundError):
                ActionDataset(
                    ["match_1"],
                    feature_dir=str(feature_dir),
                    label_dir=str(label_dir),
                    task="outcome_scoring",
                )

    def test_action_dataset_rejects_misaligned_goal_next10_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_dir = Path(tmpdir) / "features"
            label_dir = Path(tmpdir) / "labels_disc"
            diagnostic_label_dir = Path(tmpdir) / "labels_next10"
            feature_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_label_dir.mkdir(parents=True, exist_ok=True)
            torch.save([make_graph()], feature_dir / "match_1.pt")

            selected_labels = make_legacy_labels()
            diagnostic_labels = make_legacy_labels()
            diagnostic_labels[:, LABEL_INDEX["action_index"]] = 99.0
            torch.save(selected_labels, label_dir / "match_1.pt")
            torch.save(diagnostic_labels, diagnostic_label_dir / "match_1.pt")

            with self.assertRaises(ValueError):
                ActionDataset(
                    ["match_1"],
                    feature_dir=str(feature_dir),
                    label_dir=str(label_dir),
                    diagnostic_label_dir=str(diagnostic_label_dir),
                    task="outcome_scoring",
                )

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

    def test_visualization_selection_show_pass_height_is_optional(self) -> None:
        parser = argparse.ArgumentParser()
        add_component_selection_args(parser)

        default_selection = resolve_component_selection(parser.parse_args([]))
        pass_height_selection = resolve_component_selection(parser.parse_args(["--show-pass-height"]))

        self.assertNotIn("pass_height", default_selection.rendered_components)
        self.assertIn("pass_height", pass_height_selection.rendered_components)

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
