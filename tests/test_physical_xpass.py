from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datatools import config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset, pass_success_observed_target_invalid_reason
from models.gnn import Decoder
from models.utils import run_epoch
from physical_pass_model import (
    PHYSICAL_XPASS_LOGIT_ATTR,
    PHYSICAL_XPASS_PROB_ATTR,
    _validate_simulation_contract,
    attach_physical_xpass_to_graph,
    compute_graph_player_cum_prob,
    load_physical_xpass_match,
    load_physical_xpass_component,
)
from scripts import generate_physical_xpass
from scripts import train_relevant_models as train_wrapper


def make_graph(node_ids: list[str] | None = None) -> Data:
    node_ids = node_ids or ["home_1", "home_2", "away_3"]
    x = torch.zeros((len(node_ids), 25), dtype=torch.float32)
    x[:, 3] = torch.arange(len(node_ids), dtype=torch.float32) * 20.0
    x[:, 4] = 34.0
    x[:, 5:7] = 0.0
    x[0, 0] = 1.0
    x[1, 0] = 1.0
    x[0, 13] = 1.0
    if len(node_ids) > 2:
        x[2, 0] = 0.0
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones((2, 2), dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.node_ids = node_ids
    return graph


def make_label(*, action_index: int = 7, intent_index: int = 1) -> torch.Tensor:
    label = torch.zeros(len(LABEL_COLUMNS), dtype=torch.float32)
    label[LABEL_INDEX["action_index"]] = action_index
    label[LABEL_INDEX["is_pass"]] = 1
    label[LABEL_INDEX["intent_index"]] = intent_index
    label[LABEL_INDEX["success"]] = 1
    return label


class FakeSimulationResult:
    def __init__(self, shape: tuple[int, int, int, int], updates: list[tuple[int, int, int, int, float]]) -> None:
        self.player_cum_prob = np.zeros(shape, dtype=float)
        for frame_index, player_index, angle_index, distance_index, value in updates:
            self.player_cum_prob[frame_index, player_index, angle_index, distance_index] = value
        self.r_grid = np.array([0.0, 10.0, 20.0], dtype=float)


class FixedDelta(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.values[: inputs.shape[0]].unsqueeze(-1)


class DummyOffsetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.decoder = SimpleNamespace(latest_delta_gat=None)

    def forward(self, batch_graphs, batch_dests=None):
        del batch_dests
        delta = torch.tensor([0.0, 2.0], dtype=torch.float32, device=batch_graphs.x.device) + self.weight * 0.0
        self.decoder.latest_delta_gat = delta
        return torch.tensor([-10.0, 0.0], dtype=torch.float32, device=batch_graphs.x.device) + self.weight * 0.0


def decoder_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "task": "pass_success",
        "gnn_task": "node_binary",
        "node_in_dim": 25,
        "node_emb_dim": 4,
        "graph_emb_dim": 0,
        "skip_conn": False,
        "out_dim": 1,
        "include_out": False,
        "use_physical_xpass": True,
        "model_variant": "gat_phys_logit_offset",
        "learn_physical_scale": True,
        "physical_eps": 1e-4,
        "residual_clip_value": None,
    }
    args.update(overrides)
    return args


class PhysicalXPassTests(unittest.TestCase):
    def test_mocked_player_cum_prob_ignores_other_teammates_by_default(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            self.assertIn("home_1", kwargs["players"].tolist())
            self.assertIn("home_1", kwargs["passers"].tolist())
            return FakeSimulationResult((1, 3, 1, 3), [(0, 1, 0, 2, 0.73)])

        graph = make_graph()
        probs = compute_graph_player_cum_prob(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertAlmostEqual(float(probs["home_2"]), 0.73)
        self.assertTrue(np.isnan(probs["home_1"]))
        self.assertEqual(captured["PLAYER_POS"].shape, (1, 3, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "target_player", "away_3"])
        self.assertEqual(captured["player_teams"].tolist(), ["attack", "attack", "defense"])
        self.assertEqual(captured["passers"].tolist(), ["home_1"])
        self.assertTrue(captured["normalize"])
        self.assertEqual(captured["fields_to_return"], ("player_cum_prob",))

    def test_mocked_player_cum_prob_consider_teammates_preserves_previous_behavior(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            return FakeSimulationResult((1, 3, 1, 3), [(0, 1, 0, 2, 0.61)])

        graph = make_graph()
        probs = compute_graph_player_cum_prob(graph, consider_teammates=True, simulate_passes_fn=fake_simulate_passes)

        self.assertAlmostEqual(float(probs["home_2"]), 0.61)
        self.assertEqual(captured["PLAYER_POS"].shape, (1, 3, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "home_2", "away_3"])
        self.assertEqual(captured["player_teams"].tolist(), ["attack", "attack", "defense"])

    def test_mocked_player_cum_prob_uses_target_specific_slot_with_multiple_attackers(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            return FakeSimulationResult(
                (2, 3, 1, 3),
                [(0, 1, 0, 2, 0.2), (1, 1, 0, 2, 0.8)],
            )

        graph = make_graph(["home_1", "home_2", "home_4", "away_3"])
        graph.x[2, 0] = 1.0
        probs = compute_graph_player_cum_prob(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertEqual(captured["PLAYER_POS"].shape, (2, 3, 4))
        np.testing.assert_allclose(captured["PLAYER_POS"][:, 0, 0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(captured["PLAYER_POS"][:, 1, 0], np.array([20.0, 40.0]))
        np.testing.assert_allclose(captured["PLAYER_POS"][:, 2, 0], np.array([60.0, 60.0]))
        self.assertEqual(captured["players"].tolist(), ["home_1", "target_player", "away_3"])
        self.assertAlmostEqual(float(probs["home_2"]), 0.2)
        self.assertAlmostEqual(float(probs["home_4"]), 0.8)
        self.assertTrue(np.isnan(probs["away_3"]))

    def test_mocked_player_cum_prob_handles_no_opponents(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            return FakeSimulationResult((1, 2, 1, 3), [(0, 1, 0, 2, 0.5)])

        graph = make_graph(["home_1", "home_2"])
        probs = compute_graph_player_cum_prob(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertEqual(captured["PLAYER_POS"].shape, (1, 2, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "target_player"])
        self.assertAlmostEqual(float(probs["home_2"]), 0.5)

    def test_validate_simulation_contract_rejects_missing_passer(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires every passer id to be present"):
            _validate_simulation_contract(
                players=np.array(["target_player", "away_3"], dtype=object),
                passers=np.array(["home_1"], dtype=object),
                exclude_passer=True,
            )

    def test_accessible_space_smoke_reduced_mode_exclude_passer_contract(self) -> None:
        try:
            from accessible_space.core import simulate_passes_chunked
        except ImportError:
            self.skipTest("accessible_space is not installed")

        PLAYER_POS = np.array([[[0.0, 34.0, 0.0, 0.0], [20.0, 34.0, 0.0, 0.0], [60.0, 34.0, 0.0, 0.0]]], dtype=float)
        BALL_POS = np.array([[0.0, 34.0]], dtype=float)
        phi_grid = np.array([[0.0]], dtype=float)
        v0_grid = np.array([[10.0]], dtype=float)
        passer_teams = np.array(["attack"], dtype=object)
        player_teams = np.array(["attack", "attack", "defense"], dtype=object)
        players = np.array(["home_1", "target_player", "away_3"], dtype=object)
        passers = np.array(["home_1"], dtype=object)

        result = simulate_passes_chunked(
            PLAYER_POS,
            BALL_POS,
            phi_grid,
            v0_grid,
            passer_teams,
            player_teams,
            players=players,
            passers=passers,
            exclude_passer=True,
            fields_to_return=("player_cum_prob",),
            x_pitch_min=0.0,
            x_pitch_max=105.0,
            y_pitch_min=0.0,
            y_pitch_max=68.0,
            use_progress_bar=False,
        )

        self.assertEqual(np.asarray(result.player_cum_prob).shape[1], 3)

    def test_sidecar_attach_aligns_by_action_index_and_player_id_columns(self) -> None:
        graph = make_graph(["home_1", "home_2"])
        label = make_label()
        sidecar = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_2": 0.8}]).set_index("action_index", drop=False)

        attached = attach_physical_xpass_to_graph(graph, label, sidecar, match_id="m1")

        probs = getattr(attached, PHYSICAL_XPASS_PROB_ATTR)
        logits = getattr(attached, PHYSICAL_XPASS_LOGIT_ATTR)
        self.assertTrue(torch.allclose(probs, torch.tensor([0.5, 0.8])))
        self.assertAlmostEqual(float(logits[0]), 0.0, places=6)
        self.assertAlmostEqual(float(logits[1]), float(torch.logit(torch.tensor(0.8))), places=6)

    def test_sidecar_attach_fails_when_observed_target_is_missing(self) -> None:
        graph = make_graph(["home_1", "home_2"])
        label = make_label()
        sidecar = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_1": 0.5}]).set_index("action_index", drop=False)

        with self.assertRaisesRegex(ValueError, "observed target"):
            attach_physical_xpass_to_graph(graph, label, sidecar, match_id="m1")

    def test_sidecar_attach_fails_when_observed_target_is_nan(self) -> None:
        graph = make_graph(["home_1", "home_2"])
        label = make_label()
        sidecar = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_2": np.nan}]).set_index(
            "action_index",
            drop=False,
        )

        with self.assertRaisesRegex(ValueError, "observed target"):
            attach_physical_xpass_to_graph(graph, label, sidecar, match_id="m1")

    def test_pass_success_observed_target_validator(self) -> None:
        graph = make_graph()

        self.assertIsNone(pass_success_observed_target_invalid_reason(graph, make_label(intent_index=1)))
        self.assertEqual(
            pass_success_observed_target_invalid_reason(graph, make_label(intent_index=0)),
            "target_is_possessor",
        )
        self.assertEqual(
            pass_success_observed_target_invalid_reason(graph, make_label(intent_index=2)),
            "target_not_teammate",
        )
        self.assertEqual(
            pass_success_observed_target_invalid_reason(graph, make_label(intent_index=-1)),
            "target_index_out_of_bounds",
        )

        goal_graph = make_graph(["home_1", "home_goal"])
        goal_graph.x[1, config.NODE_FEATURE_IS_TEAMMATE] = 1.0
        goal_graph.x[1, config.NODE_FEATURE_IS_GOAL] = 1.0
        self.assertEqual(
            pass_success_observed_target_invalid_reason(goal_graph, make_label(intent_index=1)),
            "target_is_goal",
        )

        nonfinite_graph = make_graph()
        nonfinite_graph.x[1, config.NODE_FEATURE_X] = float("nan")
        self.assertEqual(
            pass_success_observed_target_invalid_reason(nonfinite_graph, make_label(intent_index=1)),
            "target_nonfinite_xy",
        )

    def test_action_dataset_skips_self_target_pass_success_rows_with_and_without_physical_xpass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_dir = root / "features"
            label_dir = root / "labels"
            physical_cache = root / "physical_xpass"
            match_dir = physical_cache / "matches"
            feature_dir.mkdir()
            label_dir.mkdir()
            match_dir.mkdir(parents=True)

            graphs = [make_graph(["home_1", "home_2", "away_3"]), make_graph(["home_1", "home_2", "away_3"])]
            labels = torch.stack(
                [
                    make_label(action_index=7, intent_index=1),
                    make_label(action_index=8, intent_index=0),
                ]
            )
            torch.save(graphs, feature_dir / "match_1.pt")
            torch.save(labels, label_dir / "match_1.pt")
            pd.DataFrame(
                [
                    {"match_id": "match_1", "action_index": 7, "home_2": 0.8},
                    {"match_id": "match_1", "action_index": 8, "home_1": np.nan},
                ]
            ).to_parquet(match_dir / "match_1.parquet", index=False)

            baseline = ActionDataset(["match_1"], feature_dir=feature_dir, label_dir=label_dir, task="pass_success")
            physical = ActionDataset(
                ["match_1"],
                feature_dir=feature_dir,
                label_dir=label_dir,
                task="pass_success",
                use_physical_xpass=True,
                physical_cache_dir=physical_cache,
            )

        expected_skips = {"invalid_pass_success_target:target_is_possessor": 1}
        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(physical), 1)
        self.assertEqual(baseline.skipped_rows, expected_skips)
        self.assertEqual(physical.skipped_rows, expected_skips)
        self.assertEqual(int(baseline.labels[0, LABEL_INDEX["action_index"]].item()), 7)
        self.assertEqual(int(physical.labels[0, LABEL_INDEX["action_index"]].item()), 7)
        self.assertTrue(hasattr(physical.features[0], PHYSICAL_XPASS_PROB_ATTR))
        self.assertAlmostEqual(float(getattr(physical.features[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.8)

    def test_missing_sidecar_failure_mentions_precompute_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "generate_physical_xpass.py"):
                load_physical_xpass_match(tmpdir, "missing_match")

    def test_decoder_offset_formula_and_beta_initialization(self) -> None:
        decoder = Decoder(decoder_args())
        decoder.nodewise_mlp = FixedDelta([0.1, -0.2])
        with torch.no_grad():
            decoder.physical_beta0.fill_(0.3)
            decoder.physical_beta1.fill_(2.0)

        physical_logits = torch.tensor([0.4, -0.6], dtype=torch.float32)
        out = decoder(
            node_embeddings=torch.zeros((2, 4), dtype=torch.float32),
            physical_xpass_logit=physical_logits,
        )

        expected = torch.tensor([0.3 + 2.0 * 0.4 + 0.1, 0.3 + 2.0 * -0.6 - 0.2])
        self.assertTrue(torch.allclose(out, expected))
        self.assertAlmostEqual(float(Decoder(decoder_args()).physical_beta0), 0.0)
        self.assertAlmostEqual(float(Decoder(decoder_args()).physical_beta1), 1.0)

    def test_decoder_fixed_beta_and_residual_clipping(self) -> None:
        decoder = Decoder(decoder_args(learn_physical_scale=False, residual_clip_value=1.0))
        decoder.nodewise_mlp = FixedDelta([10.0])

        out = decoder(
            node_embeddings=torch.zeros((1, 4), dtype=torch.float32),
            physical_xpass_logit=torch.tensor([0.0]),
        )

        self.assertFalse(decoder.physical_beta1.requires_grad)
        self.assertAlmostEqual(float(out[0]), float(torch.tanh(torch.tensor(10.0))), places=6)
        self.assertAlmostEqual(float(decoder.latest_delta_gat[0]), float(torch.tanh(torch.tensor(10.0))), places=6)

    def test_run_epoch_adds_observed_residual_l2_to_pass_success_loss(self) -> None:
        graph = make_graph(["home_1", "home_2"])
        label = make_label()
        loader = DataLoader([(graph, label, torch.tensor(1.0, dtype=torch.float32))], batch_size=1)
        args = SimpleNamespace(
            gnn_task="node_binary",
            task="pass_success",
            include_out=False,
            lambda_l1=0.0,
            residual_regularization_lambda=0.5,
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset",
            use_xg=False,
            use_xt=False,
            use_goal_distance=False,
            use_epv=False,
            print_freq=99,
            clip=10,
        )

        metrics = run_epoch(args, DummyOffsetModel(), loader, device="cpu", train=False)

        expected_bce = float(nn.BCEWithLogitsLoss()(torch.tensor([0.0]), torch.tensor([1.0])))
        self.assertAlmostEqual(metrics["residual_l2"], 4.0, places=6)
        self.assertAlmostEqual(metrics["ce_loss"], expected_bce + 0.5 * 4.0, places=6)

    def test_wrapper_physical_flags_reach_only_pass_success(self) -> None:
        args = SimpleNamespace(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset_regularized",
            physical_cache_dir="cache_dir",
            physical_eps=1e-3,
            learn_physical_scale=False,
            residual_regularization_lambda=0.25,
            residual_clip_value=2.0,
        )
        feature_flags = train_wrapper.WRAPPER_FEATURE_DEFAULTS.copy()

        pass_command = train_wrapper.pass_success_command(
            "pass_success",
            "pass_success/test",
            "features",
            "labels",
            "pass_intent/test",
            "angle_only",
            "disc_0.9",
            32,
            "none",
            feature_flags,
            args,
        )
        outcome_command = train_wrapper.outcome_command(
            "outcome_scoring",
            "outcome_scoring/test",
            "features",
            "labels",
            "goal",
            "disc_0.9",
            "angle_only",
            32,
            "none",
            feature_flags,
        )

        self.assertIn("--use_physical_xpass", pass_command)
        self.assertIn("--model-variant", pass_command)
        self.assertIn("--fixed-physical-scale", pass_command)
        self.assertIn("--residual-regularization-lambda", pass_command)
        self.assertNotIn("--use_physical_xpass", outcome_command)
        self.assertNotIn("--physical-cache-dir", outcome_command)

    def test_visualization_sidecar_loader_reads_player_cum_prob_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            match_dir = cache_dir / "matches"
            match_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"match_id": "m1", "action_index": 3, "home_1": 0.2, "home_2": np.nan}]
            ).to_parquet(match_dir / "m1.parquet", index=False)

            row = load_physical_xpass_component(cache_dir, "m1", 3)

        self.assertEqual(row.name, "player_cum_prob")
        self.assertAlmostEqual(float(row["home_1"]), 0.2)
        self.assertTrue(np.isnan(row["home_2"]))

    def test_generate_physical_xpass_cli_defaults_to_ignore_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run"])
        self.assertFalse(args.consider_teammates)

    def test_generate_physical_xpass_cli_accepts_consider_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--consider-teammates"])
        self.assertTrue(args.consider_teammates)


if __name__ == "__main__":
    unittest.main()
