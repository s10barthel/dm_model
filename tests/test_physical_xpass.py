from __future__ import annotations

import tempfile
import unittest
import warnings
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datatools import config
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset, pass_success_observed_target_invalid_reason
import inference
from models.gnn import Decoder
from models import utils as model_utils
from models.utils import run_epoch
from physical_pass_model import (
    AS_DEFAULT_N_ANGLES,
    AS_DEFAULT_N_V0,
    AS_DEFAULT_NORMALIZE,
    AS_DEFAULT_RESPECT_OFFSIDE,
    AS_DEFAULT_V0_MAX,
    AS_DEFAULT_V0_MIN,
    PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    PHYSICAL_XPASS_LEGACY_SOURCE,
    PHYSICAL_XPASS_LOGIT_ATTR,
    PHYSICAL_XPASS_PROB_ATTR,
    PHYSICAL_XPASS_SOURCE,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
    PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    _validate_simulation_contract,
    attach_physical_xpass_online_to_graphs,
    attach_physical_xpass_to_graph,
    compute_graphs_max_player_cum_prob_as_defaults,
    compute_graph_physical_xpass_for_source,
    compute_graph_max_player_cum_prob_as_defaults,
    compute_graph_player_cum_prob,
    load_physical_xpass_match,
    load_physical_xpass_component,
    physical_state_hash,
    physical_xpass_as_default_metadata,
    validate_physical_xpass_cache_metadata,
)
from scripts import compare_physical_xpass_speed_modes
from scripts import generate_physical_xpass
from scripts import train_relevant_models as train_wrapper


def make_graph(node_ids: list[str] | None = None) -> Data:
    node_ids = node_ids or ["home_1", "home_2", "away_3"]
    x = torch.zeros((len(node_ids), 25), dtype=torch.float32)
    x[:, 3] = torch.arange(len(node_ids), dtype=torch.float32) * 20.0
    x[:, 4] = 34.0
    x[:, 5:7] = 0.0
    x[:, config.NODE_FEATURE_POSS_DIST] = torch.abs(x[:, config.NODE_FEATURE_X] - x[0, config.NODE_FEATURE_X])
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
    def __init__(
        self,
        shape: tuple[int, int, int, int],
        updates: list[tuple[int, int, int, int, float]],
        *,
        x_grid: np.ndarray | None = None,
        y_grid: np.ndarray | None = None,
    ) -> None:
        self.player_cum_prob = np.zeros(shape, dtype=float)
        for frame_index, player_index, angle_index, distance_index, value in updates:
            self.player_cum_prob[frame_index, player_index, angle_index, distance_index] = value
        self.r_grid = np.array([0.0, 10.0, 20.0], dtype=float)
        if x_grid is not None:
            self.x_grid = x_grid
        if y_grid is not None:
            self.y_grid = y_grid


class FixedDelta(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.values[: inputs.shape[0]].unsqueeze(-1)


class DummyOffsetModel(nn.Module):
    def __init__(self, delta_values: list[float] | None = None, output_values: list[float] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.decoder = SimpleNamespace(latest_delta_gat=None)
        self.delta_values = delta_values or [0.0, 2.0]
        self.output_values = output_values or [-10.0, 0.0]

    def forward(self, batch_graphs, batch_dests=None):
        del batch_dests
        node_count = batch_graphs.x.shape[0]
        delta = torch.tensor(self.delta_values[:node_count], dtype=torch.float32, device=batch_graphs.x.device) + self.weight * 0.0
        self.decoder.latest_delta_gat = delta
        return torch.tensor(self.output_values[:node_count], dtype=torch.float32, device=batch_graphs.x.device) + self.weight * 0.0


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
        "residual_distance_threshold": 30.0,
        "short_residual_clip_value": None,
        "long_residual_clip_value": None,
    }
    args.update(overrides)
    return args


class PhysicalXPassTests(unittest.TestCase):
    def test_max_player_cum_prob_default_ignores_other_teammates_and_uses_speed_max(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            speed = float(kwargs["v0_grid"][0, 0])
            value = 0.73 if speed == AS_DEFAULT_V0_MAX else 0.41
            return FakeSimulationResult((1, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 4, 2, value)])

        graph = make_graph()
        probs = compute_graph_max_player_cum_prob_as_defaults(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertAlmostEqual(float(probs["home_2"]), 0.73)
        self.assertTrue(np.isnan(probs["home_1"]))
        self.assertTrue(np.isnan(probs["away_3"]))
        self.assertEqual(len(calls), AS_DEFAULT_N_V0)
        self.assertEqual(calls[0]["PLAYER_POS"].shape, (1, 3, 4))
        self.assertEqual(calls[0]["players"].tolist(), ["home_1", "target_player", "away_3"])
        self.assertEqual(calls[0]["player_teams"].tolist(), ["attack", "attack", "defense"])
        self.assertEqual(calls[0]["passers"].tolist(), ["home_1"])
        self.assertTrue(calls[0]["exclude_passer"])
        self.assertEqual(calls[0]["respect_offside"], AS_DEFAULT_RESPECT_OFFSIDE)
        self.assertEqual(calls[0]["normalize"], AS_DEFAULT_NORMALIZE)
        self.assertEqual(calls[0]["fields_to_return"], ("player_cum_prob",))
        self.assertEqual(calls[0]["phi_grid"].shape, (1, AS_DEFAULT_N_ANGLES))
        self.assertAlmostEqual(float(calls[0]["v0_grid"][0, 0]), AS_DEFAULT_V0_MIN)
        self.assertAlmostEqual(float(calls[-1]["v0_grid"][0, 0]), AS_DEFAULT_V0_MAX)
        np.testing.assert_allclose(calls[0]["BALL_POS"][0], np.array([-52.5, 0.0]), atol=1e-6)

    def test_max_player_cum_prob_package_max_uses_single_all_speed_call(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            return FakeSimulationResult((1, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 4, 2, 0.72)])

        graph = make_graph()
        probs = compute_graph_max_player_cum_prob_as_defaults(
            graph,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            simulate_passes_fn=fake_simulate_passes,
        )

        self.assertAlmostEqual(float(probs["home_2"]), 0.72)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["v0_grid"].shape, (1, AS_DEFAULT_N_V0))
        self.assertEqual(calls[0]["v0_prob_aggregation_mode"], "max")
        self.assertAlmostEqual(float(calls[0]["v0_grid"][0, 0]), AS_DEFAULT_V0_MIN)
        self.assertAlmostEqual(float(calls[0]["v0_grid"][0, -1]), AS_DEFAULT_V0_MAX)

    def test_batched_consider_teammates_uses_one_package_call_for_compatible_graphs(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            return FakeSimulationResult(
                (2, len(players), AS_DEFAULT_N_ANGLES, 3),
                [
                    (0, target_player_index, 0, 2, 0.44),
                    (1, target_player_index, 0, 2, 0.66),
                ],
            )

        graphs = [make_graph(), make_graph()]
        probs = compute_graphs_max_player_cum_prob_as_defaults(
            graphs,
            consider_teammates=True,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["PLAYER_POS"].shape, (2, 3, 4))
        self.assertEqual(calls[0]["v0_grid"].shape, (2, AS_DEFAULT_N_V0))
        self.assertFalse(calls[0]["exclude_passer"])
        self.assertAlmostEqual(float(probs[0]["home_2"]), 0.44)
        self.assertAlmostEqual(float(probs[1]["home_2"]), 0.66)

    def test_batched_exact_mode_makes_one_call_per_speed(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            speed = float(kwargs["v0_grid"][0, 0])
            value = 0.9 if speed == AS_DEFAULT_V0_MAX else 0.2
            return FakeSimulationResult((2, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, value)])

        probs = compute_graphs_max_player_cum_prob_as_defaults(
            [make_graph(), make_graph()],
            consider_teammates=True,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        self.assertEqual(len(calls), AS_DEFAULT_N_V0)
        self.assertEqual(calls[0]["v0_grid"].shape, (2, 1))
        self.assertEqual(calls[0]["v0_prob_aggregation_mode"], "mean")
        self.assertAlmostEqual(float(probs[0]["home_2"]), 0.9)

    def test_max_player_cum_prob_consider_teammates_uses_all_players(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            return FakeSimulationResult((1, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, 0.61)])

        graph = make_graph()
        probs = compute_graph_max_player_cum_prob_as_defaults(
            graph,
            consider_teammates=True,
            simulate_passes_fn=fake_simulate_passes,
        )

        self.assertAlmostEqual(float(probs["home_2"]), 0.61)
        self.assertEqual(captured["PLAYER_POS"].shape, (1, 3, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "home_2", "away_3"])
        self.assertFalse(captured["exclude_passer"])

    def test_max_player_cum_prob_ignores_off_pitch_distance_samples(self) -> None:
        x_grid = np.zeros((1, AS_DEFAULT_N_ANGLES, 3), dtype=float)
        y_grid = np.zeros((1, AS_DEFAULT_N_ANGLES, 3), dtype=float)
        x_grid[:, :, 2] = 80.0

        def fake_simulate_passes(**kwargs):
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            return FakeSimulationResult(
                (1, len(players), AS_DEFAULT_N_ANGLES, 3),
                [(0, target_player_index, 0, 1, 0.4), (0, target_player_index, 0, 2, 0.99)],
                x_grid=x_grid,
                y_grid=y_grid,
            )

        graph = make_graph()
        probs = compute_graph_max_player_cum_prob_as_defaults(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertAlmostEqual(float(probs["home_2"]), 0.4)

    def test_compute_graph_player_cum_prob_wrapper_keeps_no_normalize_warning(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            return FakeSimulationResult((1, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, 0.61)])

        graph = make_graph()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probs = compute_graph_player_cum_prob(
                graph,
                normalize=False,
                consider_teammates=True,
                simulate_passes_fn=fake_simulate_passes,
            )

        self.assertAlmostEqual(float(probs["home_2"]), 0.61)
        self.assertEqual(len(caught), 1)
        self.assertEqual(captured["players"].tolist(), ["home_1", "home_2", "away_3"])
        self.assertTrue(captured["normalize"])
        self.assertTrue(captured["respect_offside"])

    def test_physical_xpass_source_dispatches_legacy_to_target_location(self) -> None:
        graph = make_graph()
        expected = pd.Series({"home_1": np.nan, "home_2": 0.42, "away_3": np.nan})
        with patch("physical_pass_model.compute_graph_player_cum_prob_at_target_location", return_value=expected) as legacy_fn:
            with patch("physical_pass_model.compute_graph_max_player_cum_prob_as_defaults") as max_fn:
                result = compute_graph_physical_xpass_for_source(
                    graph,
                    source=PHYSICAL_XPASS_LEGACY_SOURCE,
                    eps=1e-4,
                )

        legacy_fn.assert_called_once()
        max_fn.assert_not_called()
        self.assertAlmostEqual(float(result["home_2"]), 0.42)

    def test_physical_xpass_source_dispatches_new_source_to_as_default_max(self) -> None:
        graph = make_graph()
        expected = pd.Series({"home_1": np.nan, "home_2": 0.73, "away_3": np.nan})
        with patch("physical_pass_model.compute_graph_max_player_cum_prob_as_defaults", return_value=expected) as max_fn:
            with patch("physical_pass_model.compute_graph_player_cum_prob_at_target_location") as legacy_fn:
                result = compute_graph_physical_xpass_for_source(
                    graph,
                    source=PHYSICAL_XPASS_SOURCE,
                    eps=1e-4,
                )

        max_fn.assert_called_once()
        legacy_fn.assert_not_called()
        self.assertAlmostEqual(float(result["home_2"]), 0.73)

    def test_online_physical_xpass_attachment_uses_source_dispatch(self) -> None:
        graph = make_graph()
        label = make_label(action_index=7)
        expected = pd.Series({"home_1": np.nan, "home_2": 0.64, "away_3": np.nan})
        with patch("physical_pass_model.compute_graph_player_cum_prob_at_target_location", return_value=expected):
            attached = attach_physical_xpass_online_to_graphs(
                [graph],
                torch.stack([label]),
                source=PHYSICAL_XPASS_LEGACY_SOURCE,
                eps=1e-4,
            )

        self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.64)

    def test_teammate_policy_switch_can_change_values(self) -> None:
        def fake_simulate_passes(**kwargs):
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player") if "target_player" in players else players.index("home_2")
            value = 0.8 if kwargs["exclude_passer"] else 0.3
            return FakeSimulationResult((kwargs["PLAYER_POS"].shape[0], len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, value)])

        graph = make_graph()
        ignore = compute_graph_max_player_cum_prob_as_defaults(graph, simulate_passes_fn=fake_simulate_passes)
        consider = compute_graph_max_player_cum_prob_as_defaults(
            graph,
            consider_teammates=True,
            simulate_passes_fn=fake_simulate_passes,
        )

        self.assertAlmostEqual(float(ignore["home_2"]), 0.8)
        self.assertAlmostEqual(float(consider["home_2"]), 0.3)

    def test_max_player_cum_prob_uses_player_id_mapping_with_multiple_attackers(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            return FakeSimulationResult(
                (2, 3, AS_DEFAULT_N_ANGLES, 3),
                [
                    (0, 1, 0, 2, 0.2),
                    (1, 1, 1, 2, 0.8),
                ],
            )

        graph = make_graph(["home_1", "home_2", "home_4", "away_3"])
        graph.x[2, 0] = 1.0
        probs = compute_graph_max_player_cum_prob_as_defaults(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertEqual(captured["PLAYER_POS"].shape, (2, 3, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "target_player", "away_3"])
        self.assertAlmostEqual(float(probs["home_2"]), 0.2)
        self.assertAlmostEqual(float(probs["home_4"]), 0.8)
        self.assertTrue(np.isnan(probs["away_3"]))

    def test_max_player_cum_prob_handles_no_opponents(self) -> None:
        captured = {}

        def fake_simulate_passes(**kwargs):
            captured.update(kwargs)
            return FakeSimulationResult((1, 2, AS_DEFAULT_N_ANGLES, 3), [(0, 1, 0, 2, 0.5)])

        graph = make_graph(["home_1", "home_2"])
        probs = compute_graph_max_player_cum_prob_as_defaults(graph, simulate_passes_fn=fake_simulate_passes)

        self.assertEqual(captured["PLAYER_POS"].shape, (1, 2, 4))
        self.assertEqual(captured["players"].tolist(), ["home_1", "target_player"])
        self.assertTrue(captured["exclude_passer"])
        np.testing.assert_allclose(captured["playing_direction"], np.array([1.0]))
        self.assertAlmostEqual(float(probs["home_2"]), 0.5)

    def test_validate_simulation_contract_rejects_missing_passer(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires every passer id to be present"):
            _validate_simulation_contract(
                players=np.array(["target_player", "away_3"], dtype=object),
                passers=np.array(["home_1"], dtype=object),
                exclude_passer=True,
            )

    def test_accessible_space_smoke_as_default_max_path(self) -> None:
        try:
            import accessible_space  # noqa: F401
        except ImportError:
            self.skipTest("accessible_space is not installed")

        graph = make_graph()
        result = compute_graph_max_player_cum_prob_as_defaults(graph, use_progress_bar=False, chunk_size=999)

        self.assertTrue(np.isfinite(float(result["home_2"])))
        self.assertTrue(np.isnan(result["home_1"]))
        self.assertTrue(np.isnan(result["away_3"]))

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

    def test_physical_xpass_metadata_accepts_new_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(f'{{"source": "{PHYSICAL_XPASS_SOURCE}"}}', encoding="utf-8")

            metadata = validate_physical_xpass_cache_metadata(tmpdir)

        self.assertEqual(metadata["source"], PHYSICAL_XPASS_SOURCE)

    def test_physical_xpass_metadata_missing_speed_aggregation_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(f'{{"source": "{PHYSICAL_XPASS_SOURCE}"}}', encoding="utf-8")

            metadata = validate_physical_xpass_cache_metadata(
                tmpdir,
                expected_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            )

        self.assertEqual(metadata["source"], PHYSICAL_XPASS_SOURCE)

    def test_physical_xpass_metadata_rejects_speed_aggregation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "source": PHYSICAL_XPASS_SOURCE,
                        "speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "speed_aggregation"):
                validate_physical_xpass_cache_metadata(
                    tmpdir,
                    expected_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

    def test_physical_xpass_metadata_rejects_old_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(f'{{"source": "{PHYSICAL_XPASS_LEGACY_SOURCE}"}}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "regenerate compatible physical xPass sidecars"):
                validate_physical_xpass_cache_metadata(tmpdir)

    def test_physical_xpass_metadata_accepts_expected_legacy_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(f'{{"source": "{PHYSICAL_XPASS_LEGACY_SOURCE}"}}', encoding="utf-8")

            metadata = validate_physical_xpass_cache_metadata(
                tmpdir,
                expected_source=PHYSICAL_XPASS_LEGACY_SOURCE,
            )

        self.assertEqual(metadata["source"], PHYSICAL_XPASS_LEGACY_SOURCE)

    def test_physical_xpass_metadata_rejects_new_source_for_legacy_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            metadata_path.write_text(f'{{"source": "{PHYSICAL_XPASS_SOURCE}"}}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, PHYSICAL_XPASS_LEGACY_SOURCE):
                validate_physical_xpass_cache_metadata(
                    tmpdir,
                    expected_source=PHYSICAL_XPASS_LEGACY_SOURCE,
                )

    def test_physical_xpass_metadata_missing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                validate_physical_xpass_cache_metadata(tmpdir)

    def test_load_model_copies_physical_xpass_source_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "args.json").write_text(
                json.dumps(
                    {
                        "model": "gat",
                        "task": "pass_success",
                        "edge_in_dim": 2,
                        "node_in_dim": 25,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "physical_xpass": {
                            "source": PHYSICAL_XPASS_LEGACY_SOURCE,
                            "teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
                            "speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                        }
                    }
                ),
                encoding="utf-8",
            )

            class FakeModel:
                def __init__(self, args):
                    self.args = args

                def to(self, device):
                    del device
                    return self

                def load_state_dict(self, state_dict):
                    del state_dict

            with patch.object(model_utils, "get_model_path", return_value=model_dir):
                with patch.object(model_utils, "GNN", FakeModel):
                    with patch.object(torch, "load", return_value={}):
                        model = model_utils.load_model("pass_success/fake_run", device="cpu")

        self.assertEqual(model.args["physical_xpass_source"], PHYSICAL_XPASS_LEGACY_SOURCE)
        self.assertEqual(model.args["physical_xpass_teammate_policy"], PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE)
        self.assertEqual(
            model.args["physical_xpass_speed_aggregation"],
            PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
        )

    def test_inference_missing_synthetic_sidecar_computes_physical_xpass_online(self) -> None:
        RuntimeState = type("BenchmarkState", (), {"__module__": "datatools.benchmark"})
        match = RuntimeState()
        match.match_id = "modification_50_game_state_2"
        graphs = [make_graph()]
        labels = torch.stack([make_label(action_index=0)])
        model = SimpleNamespace(
            args={
                "task": "pass_success",
                "model_id": "pass_success/fake",
                "use_physical_xpass": True,
                "model_variant": "gat_phys_logit_offset",
                "physical_xpass_source": PHYSICAL_XPASS_LEGACY_SOURCE,
                "physical_cache_dir": "missing_cache",
                "physical_eps": 1e-4,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata", side_effect=FileNotFoundError("missing")):
            with patch.object(inference, "attach_physical_xpass_online_to_graphs", return_value=graphs) as online_attach:
                result = inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        self.assertIs(result, graphs)
        online_attach.assert_called_once()
        self.assertEqual(online_attach.call_args.kwargs["source"], PHYSICAL_XPASS_LEGACY_SOURCE)
        self.assertEqual(
            online_attach.call_args.kwargs["speed_aggregation"],
            PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
        )
        self.assertEqual(match.physical_xpass_runtime_stats["pass_success"]["online_graphs"], 1)

    def test_inference_missing_real_sidecar_stays_strict(self) -> None:
        match = SimpleNamespace(match_id="DFL-MAT-REAL", runtime_feature_root=Path("feature_run"))
        graphs = [make_graph()]
        labels = torch.stack([make_label(action_index=0)])
        model = SimpleNamespace(
            args={
                "task": "pass_success",
                "use_physical_xpass": True,
                "model_variant": "gat_phys_logit_offset",
                "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
                "physical_cache_dir": "missing_cache",
                "physical_eps": 1e-4,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata", side_effect=FileNotFoundError("missing")):
            with patch.object(inference, "attach_physical_xpass_online_to_graphs") as online_attach:
                with self.assertRaises(FileNotFoundError):
                    inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        online_attach.assert_not_called()

    def test_generate_physical_xpass_reuse_cache_validation_rejects_old_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps({"source": PHYSICAL_XPASS_LEGACY_SOURCE}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incompatible source"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuse_cache_validation_rejects_policy_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "teammate_policy"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuse_cache_validation_rejects_speed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "speed_aggregation"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuses_row_without_hash_and_skips_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graphs.pt"
            label_path = root / "labels.pt"
            torch.save([make_graph(["home_1", "home_2", "away_3"])], graph_path)
            torch.save(torch.stack([make_label(action_index=7)]), label_path)
            reuse_rows = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_2": 0.88}]).set_index(
                "action_index",
                drop=False,
            )

            def fail_compute(*_args, **_kwargs):
                raise AssertionError("compute should not be called for reusable rows")

            stats: dict[str, int] = {}
            frame, computed = generate_physical_xpass.compute_match_rows(
                "m1",
                graph_path,
                label_path,
                eps=1e-4,
                normalize=True,
                consider_teammates=False,
                limit=None,
                reuse_rows=reuse_rows,
                reuse_stats=stats,
                compute_fn=fail_compute,
            )

        self.assertEqual(computed, 0)
        self.assertEqual(stats["reused_actions"], 1)
        self.assertEqual(stats["reused_without_state_hash"], 1)
        self.assertAlmostEqual(float(frame.loc[0, "home_2"]), 0.88)
        self.assertIn("physical_state_hash", frame.columns)

    def test_generate_physical_xpass_hash_match_reuses_and_hash_mismatch_recomputes(self) -> None:
        graph = make_graph(["home_1", "home_2", "away_3"])
        matching_hash = physical_state_hash(graph)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graphs.pt"
            label_path = root / "labels.pt"
            torch.save([graph, graph], graph_path)
            torch.save(
                torch.stack([make_label(action_index=7), make_label(action_index=8)]),
                label_path,
            )
            reuse_rows = pd.DataFrame(
                [
                    {"match_id": "m1", "action_index": 7, "home_2": 0.77, "physical_state_hash": matching_hash},
                    {"match_id": "m1", "action_index": 8, "home_2": 0.12, "physical_state_hash": "mismatch"},
                ]
            ).set_index("action_index", drop=False)

            def fake_compute(graph_arg, **_kwargs):
                return pd.Series({"home_1": np.nan, "home_2": 0.66, "away_3": np.nan})

            stats: dict[str, int] = {}
            frame, computed = generate_physical_xpass.compute_match_rows(
                "m1",
                graph_path,
                label_path,
                eps=1e-4,
                normalize=True,
                consider_teammates=False,
                limit=None,
                reuse_rows=reuse_rows,
                reuse_stats=stats,
                compute_fn=fake_compute,
            )

        by_action = frame.set_index("action_index")
        self.assertEqual(computed, 1)
        self.assertEqual(stats["reused_actions"], 1)
        self.assertEqual(stats["hash_verified"], 1)
        self.assertEqual(stats["hash_mismatch_recomputed"], 1)
        self.assertAlmostEqual(float(by_action.loc[7, "home_2"]), 0.77)
        self.assertAlmostEqual(float(by_action.loc[8, "home_2"]), 0.66)

    def test_generate_physical_xpass_limit_counts_computed_rows_only(self) -> None:
        graph = make_graph(["home_1", "home_2", "away_3"])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graph_path = root / "graphs.pt"
            label_path = root / "labels.pt"
            torch.save([graph, graph, graph], graph_path)
            torch.save(
                torch.stack(
                    [
                        make_label(action_index=7),
                        make_label(action_index=8),
                        make_label(action_index=9),
                    ]
                ),
                label_path,
            )
            reuse_rows = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_2": 0.7}]).set_index(
                "action_index",
                drop=False,
            )

            def fake_compute(graph_arg, **_kwargs):
                return pd.Series({"home_1": np.nan, "home_2": 0.5, "away_3": np.nan})

            stats: dict[str, int] = {}
            frame, computed = generate_physical_xpass.compute_match_rows(
                "m1",
                graph_path,
                label_path,
                eps=1e-4,
                normalize=True,
                consider_teammates=False,
                limit=1,
                reuse_rows=reuse_rows,
                reuse_stats=stats,
                compute_fn=fake_compute,
            )

        self.assertEqual(computed, 1)
        self.assertEqual(stats["reused_actions"], 1)
        self.assertEqual(stats["compute_limit_skipped"], 1)
        self.assertEqual(sorted(frame["action_index"].astype(int).tolist()), [7, 8])

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

    def test_decoder_applies_distance_specific_residual_clipping(self) -> None:
        decoder = Decoder(
            decoder_args(
                residual_distance_threshold=30.0,
                short_residual_clip_value=1.0,
                long_residual_clip_value=3.0,
            )
        )
        decoder.nodewise_mlp = FixedDelta([10.0, 10.0])
        node_features = torch.zeros((2, 25), dtype=torch.float32)
        node_features[:, config.NODE_FEATURE_POSS_DIST] = torch.tensor([20.0, 40.0])

        out = decoder(
            node_features=node_features,
            node_embeddings=torch.zeros((2, 4), dtype=torch.float32),
            physical_xpass_logit=torch.zeros(2, dtype=torch.float32),
        )

        expected = torch.tensor(
            [
                torch.tanh(torch.tensor(10.0)).item(),
                (3.0 * torch.tanh(torch.tensor(10.0 / 3.0))).item(),
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(out, expected))
        self.assertTrue(torch.allclose(decoder.latest_delta_gat, expected))

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

    def test_run_epoch_uses_distance_specific_residual_l2_weights(self) -> None:
        short_graph = make_graph(["home_1", "home_2"])
        long_graph = make_graph(["home_1", "home_2"])
        long_graph.x[1, config.NODE_FEATURE_POSS_DIST] = 40.0
        label = make_label()
        loader = DataLoader(
            [
                (short_graph, label, torch.tensor(1.0, dtype=torch.float32)),
                (long_graph, label, torch.tensor(1.0, dtype=torch.float32)),
            ],
            batch_size=2,
        )
        args = SimpleNamespace(
            gnn_task="node_binary",
            task="pass_success",
            include_out=False,
            lambda_l1=0.0,
            residual_regularization_lambda=0.0,
            residual_distance_threshold=30.0,
            short_residual_regularization_lambda=0.25,
            long_residual_regularization_lambda=0.5,
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset",
            use_xg=False,
            use_xt=False,
            use_goal_distance=False,
            use_epv=False,
            print_freq=99,
            clip=10,
        )

        metrics = run_epoch(
            args,
            DummyOffsetModel(delta_values=[0.0, 2.0, 0.0, 3.0], output_values=[-10.0, 0.0, -10.0, 0.0]),
            loader,
            device="cpu",
            train=False,
        )

        expected_bce = float(nn.BCEWithLogitsLoss()(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])))
        expected_penalty = (0.25 * 2.0**2 + 0.5 * 3.0**2) / 2.0
        self.assertAlmostEqual(metrics["residual_l2"], (2.0**2 + 3.0**2) / 2.0, places=6)
        self.assertAlmostEqual(metrics["ce_loss"], expected_bce + expected_penalty, places=6)

    def test_wrapper_physical_flags_reach_only_pass_success(self) -> None:
        args = SimpleNamespace(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset_regularized",
            physical_cache_dir="cache_dir",
            physical_eps=1e-3,
            learn_physical_scale=False,
            residual_regularization_lambda=0.25,
            residual_clip_value=2.0,
            residual_distance_threshold=35.0,
            short_residual_regularization_lambda=0.5,
            long_residual_regularization_lambda=0.1,
            short_residual_clip_value=1.0,
            long_residual_clip_value=3.0,
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
        self.assertIn("--residual-distance-threshold", pass_command)
        self.assertIn("--short-residual-regularization-lambda", pass_command)
        self.assertIn("--long-residual-regularization-lambda", pass_command)
        self.assertIn("--short-residual-clip-value", pass_command)
        self.assertIn("--long-residual-clip-value", pass_command)
        self.assertNotIn("--use_physical_xpass", outcome_command)
        self.assertNotIn("--physical-cache-dir", outcome_command)
        self.assertNotIn("--short-residual-clip-value", outcome_command)

    def test_visualization_sidecar_loader_reads_max_player_cum_prob_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            match_dir = cache_dir / "matches"
            match_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"match_id": "m1", "action_index": 3, "physical_state_hash": "abc", "home_1": 0.2, "home_2": np.nan}]
            ).to_parquet(match_dir / "m1.parquet", index=False)

            row = load_physical_xpass_component(cache_dir, "m1", 3)

        self.assertEqual(row.name, "max_player_cum_prob")
        self.assertAlmostEqual(float(row["home_1"]), 0.2)
        self.assertTrue(np.isnan(row["home_2"]))
        self.assertNotIn("physical_state_hash", row.index)

    def test_generate_physical_xpass_cli_defaults_to_consider_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run"])
        self.assertTrue(args.consider_teammates)
        self.assertEqual(args.speed_aggregation, PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION)

    def test_generate_physical_xpass_cli_accepts_ignore_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--ignore-teammates"])
        self.assertFalse(args.consider_teammates)

    def test_generate_physical_xpass_cli_accepts_consider_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--consider-teammates"])
        self.assertTrue(args.consider_teammates)

    def test_generate_physical_xpass_cli_accepts_reuse_cache_dir(self) -> None:
        args = generate_physical_xpass.parse_args(
            ["--feature-run-id", "feature_run", "--reuse-cache-dir", "old_cache"]
        )
        self.assertEqual(args.reuse_cache_dir, "old_cache")

    def test_generate_physical_xpass_cli_accepts_speed_aggregation_and_workers(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--feature-run-id",
                "feature_run",
                "--speed-aggregation",
                "exact_separate_speed",
                "--num-workers",
                "1",
                "--physical-batch-size",
                "4",
            ]
        )
        self.assertEqual(args.speed_aggregation, PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED)
        self.assertEqual(args.num_workers, "1")
        self.assertEqual(args.physical_batch_size, 4)

    def test_generate_physical_xpass_auto_workers_resolves_to_six_on_sixteen_cores(self) -> None:
        with patch.object(generate_physical_xpass.os, "cpu_count", return_value=16):
            self.assertEqual(generate_physical_xpass.resolve_num_workers("auto"), 6)

    def test_compare_speed_modes_long_frame_and_top_option_metrics(self) -> None:
        exact = pd.DataFrame(
            [
                {"match_id": "m1", "action_index": 1, "home_2": 0.8, "home_3": 0.1},
                {"match_id": "m1", "action_index": 2, "home_2": 0.2, "home_3": 0.7},
            ]
        )
        package = pd.DataFrame(
            [
                {"match_id": "m1", "action_index": 1, "home_2": 0.75, "home_3": 0.2},
                {"match_id": "m1", "action_index": 2, "home_2": 0.3, "home_3": 0.6},
            ]
        )

        long_frame = compare_physical_xpass_speed_modes.long_compare_frame(exact, package)
        agreement = compare_physical_xpass_speed_modes.top_option_agreement(exact, package)

        self.assertEqual(len(long_frame), 4)
        self.assertAlmostEqual(float(long_frame["abs_diff"].max()), 0.1)
        self.assertEqual(agreement, 1.0)


if __name__ == "__main__":
    unittest.main()
