from __future__ import annotations

import tempfile
import unittest
import warnings
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from datatools import config, metadata_summary
from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from dataset import ActionDataset, pass_success_observed_target_invalid_reason
import inference
import project_config
from models.gnn import Decoder
from models import utils as model_utils
from models.utils import run_epoch
from physical_pass_model import (
    AS_DEFAULT_N_ANGLES,
    AS_DEFAULT_N_V0,
    AS_DEFAULT_ANGLE_STEP_DEG,
    AS_DEFAULT_COARSE_N_ANGLES,
    AS_DEFAULT_NORMALIZE,
    AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG,
    AS_DEFAULT_REFINE_TOP_K_ANGLES,
    AS_DEFAULT_RESPECT_OFFSIDE,
    AS_DEFAULT_SPEED_STEP,
    AS_DEFAULT_V0_MAX,
    AS_DEFAULT_V0_MIN,
    PC_XPASS_AVAILABLE_METRICS,
    PC_XPASS_DEFAULT_CONTROL_FUNCTION_GAMMA,
    PC_XPASS_DEFAULT_CONTROL_FUNCTION_INFLECTION_POINT,
    PC_XPASS_DEFAULT_CONTROL_FUNCTION_POWER,
    PC_XPASS_DEFAULT_MAX_SPEED,
    PC_XPASS_DEFAULT_METRIC,
    PC_XPASS_DEFAULT_SPEED_STEP,
    PC_XPASS_METRIC_TOP10,
    PC_XPASS_METRIC_TOP25,
    PC_XPASS_SOURCE,
    PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION,
    PHYSICAL_DEFAULT_MAX_AUTO_WORKERS,
    PHYSICAL_XPASS_LEGACY_SOURCE,
    PHYSICAL_XPASS_LOGIT_ATTR,
    PHYSICAL_XPASS_NEAREST_OPPONENT_DISTANCE_ATTR,
    PHYSICAL_XPASS_NEAREST_OPPONENT_DISTANCE_SUFFIX,
    PHYSICAL_XPASS_PASS_DISTANCE_COLUMN,
    PHYSICAL_XPASS_PROB_ATTR,
    PHYSICAL_XPASS_SOURCE,
    PHYSICAL_XPASS_DEFAULT_METRIC,
    PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR,
    PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR,
    PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR,
    PHYSICAL_XPASS_DEFAULT_TOP_N,
    PHYSICAL_XPASS_BALL_Z_ATTR,
    PHYSICAL_XPASS_BALL_Z_COLUMN,
    PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
    PHYSICAL_XPASS_FRAME_SCOPE_COLUMN,
    PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE,
    PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN,
    PHYSICAL_XPASS_METRIC_MAX,
    PHYSICAL_XPASS_METRIC_NOISE_KERNEL,
    PHYSICAL_XPASS_METRIC_SCHEMA_VERSION,
    PHYSICAL_XPASS_METRIC_TOPMEAN,
    PHYSICAL_XPASS_NOISE_KERNEL_ALGORITHM,
    PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
    PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
    PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
    PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
    PHYSICAL_XPASS_TOPMEAN_DEFINITION,
    _robust_xpass_metrics_from_values,
    _validate_simulation_contract,
    as_default_v0_values,
    attach_physical_xpass_cached_online_to_graphs,
    attach_physical_xpass_online_to_graphs,
    attach_physical_xpass_read_only_to_graphs,
    attach_physical_xpass_to_graph,
    blend_physical_xpass_predictions,
    compute_graph_pc_xpass_metrics,
    compute_graphs_max_player_cum_prob_as_defaults,
    compute_graphs_physical_xpass_metrics_as_defaults,
    compute_graph_physical_xpass_for_source,
    compute_graph_max_player_cum_prob_as_defaults,
    compute_graph_physical_xpass_metrics_as_defaults,
    compute_graph_player_cum_prob,
    format_physical_xpass_cache_summary,
    graph_ball_z,
    inference_uses_physical_xpass,
    load_physical_xpass_match,
    load_physical_xpass_component,
    load_runtime_physical_xpass_visualization_component,
    graph_nearest_opponent_distances,
    graph_nearest_opponent_distance_row_values,
    observed_pass_distance,
    physical_state_hash,
    physical_xpass_as_default_metadata,
    physical_xpass_inference_lookup_config,
    physical_xpass_kernel_sigmas,
    physical_xpass_metric,
    physical_xpass_metric_column,
    pc_xpass_metadata,
    pc_xpass_endpoint_control_probabilities,
    pc_xpass_lane_survival_from_raw,
    pc_xpass_normalize_if_sum_above_one,
    pc_xpass_raw_control,
    pc_xpass_raw_control_with_params,
    physical_xpass_nearest_opponent_distance_column,
    physical_xpass_pass_height_column,
    physical_xpass_ball_z_limit,
    physical_xpass_weight_version,
    physical_xpass_blend_weight_v2,
    physical_xpass_blend_weight_v3,
    physical_xpass_source,
    prepare_runtime_physical_xpass_prewarm_items,
    prewarm_physical_xpass_runtime_cache,
    refined_angle_grid_from_coarse_angles,
    resolve_physical_num_workers,
    runtime_physical_xpass_source,
    runtime_physical_xpass_speed_aggregation,
    summarize_physical_xpass_cache_usage,
    validate_physical_xpass_cache_metadata,
)
from scripts import compare_physical_xpass_speed_modes
from scripts import generate_epv
from scripts import generate_physical_xpass
from scripts import run_and_visualize_hawkeye
from scripts import run_benchmark
from scripts import run_hawkeye
from scripts import run_skillcorner
from scripts import train_relevant_models as train_wrapper
from scripts import visualize_action_components


class DummyEpvModel:
    def __init__(self, *, task: str) -> None:
        self.args = {"task": task, "model_variant": "gat_baseline"}


class ConstantPassHeightModel(nn.Module):
    def __init__(self, *, probability: float = 0.8, node_in_dim: int = 25) -> None:
        super().__init__()
        self.args = {"task": "pass_height", "node_in_dim": int(node_in_dim)}
        self.logit = float(torch.logit(torch.tensor(float(probability))).item())

    def forward(self, batch: Data) -> torch.Tensor:
        return torch.full((int(batch.x.shape[0]),), self.logit, dtype=torch.float32, device=batch.x.device)


def make_epv_model_specs() -> dict[str, DummyEpvModel]:
    return {
        "pass_intent": DummyEpvModel(task="pass_intent"),
        "pass_success": DummyEpvModel(task="pass_success"),
        "outcome_scoring": DummyEpvModel(task="outcome_scoring"),
        "outcome_conceding": DummyEpvModel(task="outcome_conceding"),
    }


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


def make_pass_success_args(**overrides: object) -> dict[str, object]:
    args: dict[str, object] = {
        "task": "pass_success",
        "model_id": "pass_success/fake",
        "use_physical_xpass": True,
        "model_variant": "gat_phys_logit_offset",
        "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
        "physical_xpass_teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
        "physical_xpass_speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
        "physical_eps": 1e-4,
        "xy_only": False,
        "possessor_aware": True,
        "keeper_aware": True,
        "ball_z_aware": True,
        "poss_vel_aware": True,
        "poss_rel_vel_aware": False,
        "accel_aware": True,
        "offside_aware": True,
        "extend_features": True,
        "filter_blockers": False,
        "sparsify": "none",
        "max_edge_dist": 10,
        "edge_in_dim": 2,
        "node_in_dim": 25,
    }
    args.update(overrides)
    return args


def make_runtime_prewarm_stats(rows: int) -> dict[str, object]:
    return {
        "rows_scanned": int(rows),
        "pass_rows": int(rows),
        "cache_hits": 0,
        "cache_misses": int(rows),
        "cache_written": int(rows),
        "copied_from_reuse": 0,
        "pass_distance_filled": 0,
        "pass_height_filled": 0,
        "pass_height_refreshed": 0,
        "hash_mismatch_recomputed": 0,
        "online_graphs": int(rows),
        "compute_chunks": 1 if rows else 0,
        "skipped_all_nan": 0,
        "cache_scan_seconds": 0.0,
        "compute_seconds": 0.0,
        "write_seconds": 0.0,
        "matches": {},
    }


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


class DummyColumnNodeSelectionModel(nn.Module):
    def __init__(self, output_values: list[float]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.output_values = output_values

    def forward(self, batch_graphs, batch_dests=None):
        del batch_dests
        node_count = batch_graphs.x.shape[0]
        values = torch.tensor(self.output_values[:node_count], dtype=torch.float32, device=batch_graphs.x.device)
        return values.unsqueeze(-1) + self.weight * 0.0


class DummyPhysicalInferenceModel(nn.Module):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.args = {
            "task": "pass_success",
            "node_in_dim": 25,
            "include_out": False,
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": False,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
            "filter_blockers": False,
            "use_physical_xpass": True,
            "model_variant": "gat_phys_logit_offset",
            "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
            "physical_xpass_speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            "physical_xpass_teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
            "physical_cache_dir": str(cache_dir),
        }

    def forward(self, graphs: Data, _batch_dests: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(graphs.x.shape[0], dtype=torch.float32, device=graphs.x.device) + self.weight * 0.0


class DummyBaselineInferenceModel(nn.Module):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.args = {
            "task": "pass_success",
            "node_in_dim": 25,
            "include_out": False,
            "xy_only": False,
            "possessor_aware": True,
            "keeper_aware": True,
            "ball_z_aware": True,
            "poss_vel_aware": True,
            "accel_aware": True,
            "offside_aware": True,
            "extend_features": False,
            "sparsify": "none",
            "max_edge_dist": 10,
            "edge_in_dim": 2,
            "filter_blockers": False,
            "use_physical_xpass": False,
            "inference_use_physical_xpass": True,
            "model_variant": "gat_baseline",
            "x_pass_version": "noise-kernel",
            "xpass_weight": "v1",
            "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
            "physical_xpass_speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            "physical_xpass_teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
            "physical_cache_dir": str(cache_dir),
        }

    def forward(self, graphs: Data, _batch_dests: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.full((graphs.x.shape[0],), -10.0, dtype=torch.float32, device=graphs.x.device)
        teammate_mask = graphs.x[:, config.NODE_FEATURE_IS_TEAMMATE] == 1
        out[teammate_mask] = torch.logit(torch.tensor(0.9, dtype=torch.float32, device=graphs.x.device))
        return out + self.weight * 0.0


class RecordingTqdm:
    calls: list["RecordingTqdm"] = []

    def __init__(self, iterable, *args, **kwargs) -> None:
        self.iterable = list(iterable)
        self.args = args
        self.kwargs = kwargs
        self.postfixes: list[dict[str, object]] = []
        self.writes: list[str] = []
        RecordingTqdm.calls.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs) -> None:
        self.postfixes.append(dict(kwargs))

    def write(self, message: str) -> None:
        self.writes.append(message)


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
        "freeze_beta0": False,
        "freeze_beta1": False,
        "physical_eps": 1e-4,
        "residual_clip_value": None,
        "residual_distance_threshold": 30.0,
        "short_residual_clip_value": None,
        "long_residual_clip_value": None,
    }
    args.update(overrides)
    return args


def make_physical_inference_match(action_indexes: list[int], is_pass: list[int]) -> SimpleNamespace:
    labels = torch.zeros((len(action_indexes), len(LABEL_COLUMNS)), dtype=torch.float32)
    for row_index, (action_index, is_pass_i) in enumerate(zip(action_indexes, is_pass)):
        labels[row_index, LABEL_INDEX["action_index"]] = int(action_index)
        labels[row_index, LABEL_INDEX["is_pass"]] = int(is_pass_i)
        labels[row_index, LABEL_INDEX["intent_index"]] = 1
        labels[row_index, LABEL_INDEX["success"]] = 1

    tracking_rows = []
    action_rows = []
    for action_index in action_indexes:
        tracking_rows.append(
            {
                "home_1_x": 10.0,
                "home_1_y": 34.0,
                "home_2_x": 20.0,
                "home_2_y": 30.0,
                "away_3_x": 60.0,
                "away_3_y": 34.0,
            }
        )
        action_rows.append(
            {
                "frame_id": int(action_index),
                "object_id": "home_1",
                "end_frame_id": int(action_index),
                "end_player_id": "home_1",
            }
        )

    return SimpleNamespace(
        actions=pd.DataFrame(action_rows, index=pd.Index(action_indexes, name="index")),
        tracking=pd.DataFrame(tracking_rows, index=pd.Index(action_indexes, name="frame_id")),
        labels=labels,
        graph_features_0=[make_graph(["home_1", "home_2", "away_3"]) for _ in action_indexes],
        graph_features_1=None,
        graph_features_by_dir={},
        graph_feature_action_indices_by_dir={},
        match_id="match_1",
    )


def write_physical_inference_cache(cache_dir: Path, action_indexes: list[int]) -> None:
    (cache_dir / "matches").mkdir(parents=True)
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            )
        ),
        encoding="utf-8",
    )
    graph = make_graph(["home_1", "home_2", "away_3"])
    label = make_label(action_index=0, intent_index=1)
    rows = []
    for action_index in action_indexes:
        label[LABEL_INDEX["action_index"]] = int(action_index)
        rows.append(
            {
                "match_id": "match_1",
                "action_index": int(action_index),
                "physical_state_hash": physical_state_hash(graph),
                PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: observed_pass_distance(graph, label),
                "home_1": 0.2,
                "home_2": 0.8,
                "away_3": 0.1,
                "home_1__topmean_xpass": 0.2,
                "home_2__topmean_xpass": 0.8,
                "away_3__topmean_xpass": 0.1,
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "action_index",
            "physical_state_hash",
            PHYSICAL_XPASS_PASS_DISTANCE_COLUMN,
            "home_1",
            "home_2",
            "away_3",
            "home_1__topmean_xpass",
            "home_2__topmean_xpass",
            "away_3__topmean_xpass",
        ],
    ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)


def write_runtime_visualization_xpass_cache(cache_dir: Path, match_id: str = "match_1", action_indexes: list[int] | None = None) -> None:
    action_indexes = action_indexes or [0]
    (cache_dir / "matches").mkdir(parents=True)
    metadata = physical_xpass_as_default_metadata(
        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
        max_speed=22,
        speed_step=1,
    )
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = []
    for action_index in action_indexes:
        rows.append(
            {
                "match_id": str(match_id),
                "action_index": int(action_index),
                "physical_state_hash": "hash",
                PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 12.0,
                "home_1": 0.2,
                "home_2": 0.4,
                "home_1__max_xpass": 0.8,
                "home_2__max_xpass": 0.9,
                "home_1__topmean_xpass": 0.5,
                "home_2__topmean_xpass": 0.6,
            }
        )
    pd.DataFrame(rows).to_parquet(cache_dir / "matches" / f"{match_id}.parquet", index=False)


class PhysicalXPassTests(unittest.TestCase):
    def test_inference_physical_xpass_blend_formula(self) -> None:
        self.assertAlmostEqual(
            float(blend_physical_xpass_predictions(pass_success_model=0.9, xpass=0.5, pass_distance=10.0)),
            0.54,
        )
        self.assertAlmostEqual(
            float(blend_physical_xpass_predictions(pass_success_model=0.9, xpass=0.5, pass_distance=0.0)),
            0.5,
        )
        self.assertAlmostEqual(
            float(blend_physical_xpass_predictions(pass_success_model=0.9, xpass=0.5, pass_distance=100.0)),
            0.9,
        )
        self.assertAlmostEqual(
            float(blend_physical_xpass_predictions(pass_success_model=0.9, xpass=0.5, pass_distance=120.0)),
            0.9,
        )

    def test_inference_physical_xpass_blend_formula_v2(self) -> None:
        peak_weight = physical_xpass_blend_weight_v2(40.0, 20.0)
        weight_50m = physical_xpass_blend_weight_v2(50.0, 20.0)
        expected_50m_weight = 0.5 * np.sin((np.pi / 0.8) * 0.5) ** 3 * 1.4

        self.assertAlmostEqual(float(peak_weight), 0.7)
        self.assertAlmostEqual(float(weight_50m), expected_50m_weight)
        self.assertAlmostEqual(
            float(
                blend_physical_xpass_predictions(
                    pass_success_model=0.9,
                    xpass=0.5,
                    pass_distance=40.0,
                    distance_to_nearest_opponent=20.0,
                    weight_version="v2",
                )
            ),
            0.78,
        )
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v2(80.0, 20.0)), 0.0)
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v2(150.0, 20.0)), 0.0)
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v2(40.0, 100.0)), 1.0)

    def test_inference_physical_xpass_blend_formula_v2_requires_nearest_opponent_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "distance_to_nearest_opponent"):
            blend_physical_xpass_predictions(
                pass_success_model=0.9,
                xpass=0.5,
                pass_distance=50.0,
                weight_version="v2",
            )

    def test_inference_physical_xpass_blend_formula_v3(self) -> None:
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v3(0.0)), 0.0)
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v3(100.0)), 0.0)
        self.assertAlmostEqual(float(physical_xpass_blend_weight_v3(50.0)), 0.375)
        self.assertAlmostEqual(
            float(
                blend_physical_xpass_predictions(
                    pass_success_model=0.9,
                    xpass=0.5,
                    pass_distance=50.0,
                    weight_version="v3",
                )
            ),
            0.65,
        )

    def test_inference_physical_xpass_ball_z_limit_forces_model_weight(self) -> None:
        self.assertAlmostEqual(
            float(
                blend_physical_xpass_predictions(
                    pass_success_model=0.9,
                    xpass=0.5,
                    pass_distance=10.0,
                    ball_z=1.2,
                    ball_z_limit=1.0,
                )
            ),
            0.9,
        )
        self.assertAlmostEqual(
            float(
                blend_physical_xpass_predictions(
                    pass_success_model=0.9,
                    xpass=0.5,
                    pass_distance=10.0,
                    ball_z=0.8,
                    ball_z_limit=1.0,
                )
            ),
            0.54,
        )

    def test_inference_physical_xpass_ball_z_limit_requires_ball_z(self) -> None:
        with self.assertRaisesRegex(ValueError, "ball_z"):
            blend_physical_xpass_predictions(
                pass_success_model=0.9,
                xpass=0.5,
                pass_distance=10.0,
                ball_z_limit=1.0,
            )

    def test_graph_nearest_opponent_distances(self) -> None:
        distances = graph_nearest_opponent_distances(make_graph())

        self.assertAlmostEqual(float(distances[0]), 40.0)
        self.assertAlmostEqual(float(distances[1]), 20.0)
        self.assertAlmostEqual(float(distances[2]), 20.0)

    def test_graph_ball_z_uses_cached_graph_feature(self) -> None:
        graph = make_graph()
        graph.x[:, config.NODE_FEATURE_BALL_Z] = 1.25

        self.assertAlmostEqual(graph_ball_z(graph), 1.25)

    def test_physical_xpass_robust_defaults_and_kernel_sigmas(self) -> None:
        speeds = as_default_v0_values()
        self.assertEqual(speeds.tolist(), [float(value) for value in range(3, 23)])
        self.assertEqual(AS_DEFAULT_V0_MAX, 22.0)
        self.assertEqual(AS_DEFAULT_SPEED_STEP, 1.0)
        self.assertEqual(AS_DEFAULT_COARSE_N_ANGLES, 36)
        self.assertEqual(AS_DEFAULT_REFINE_TOP_K_ANGLES, 2)
        self.assertEqual(AS_DEFAULT_REFINE_ANGLE_RADIUS_DEG, 10.0)
        self.assertEqual(AS_DEFAULT_ANGLE_STEP_DEG, 2.5)

        sigma_angle, sigma_speed, sigma_distance = physical_xpass_kernel_sigmas(20.0, 30.0)

        self.assertEqual(PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR, 0.1)
        self.assertEqual(PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR, 0.05)
        self.assertEqual(PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR, 0.05)
        self.assertAlmostEqual(sigma_angle, np.deg2rad(2.0), places=8)
        self.assertAlmostEqual(sigma_speed, 1.0, places=8)
        self.assertAlmostEqual(sigma_distance, 1.5, places=8)

        custom_angle, custom_speed, custom_distance = physical_xpass_kernel_sigmas(
            20.0,
            30.0,
            sigma_angle=0.15,
            sigma_speed=0.1,
            sigma_distance=0.2,
        )

        self.assertAlmostEqual(custom_angle, np.deg2rad(3.0), places=8)
        self.assertAlmostEqual(custom_speed, 2.0, places=8)
        self.assertAlmostEqual(custom_distance, 6.0, places=8)

    def test_robust_physical_xpass_topmean_uses_top_n_values(self) -> None:
        values = np.arange(1, 19, dtype=float).reshape(2, 3, 3)
        metrics = _robust_xpass_metrics_from_values(
            values,
            speeds=np.asarray([10.0, 20.0], dtype=float),
            angles=np.deg2rad([0.0, 2.5, 5.0]),
            distances=np.asarray([10.0, 20.0, 30.0], dtype=float),
            top_n=3,
        )

        self.assertAlmostEqual(metrics[PHYSICAL_XPASS_METRIC_TOPMEAN], float(np.mean(np.arange(16, 19, dtype=float))))

    def test_angle_conditioned_noise_kernel_uses_angle_specific_best_speed_distance(self) -> None:
        values = np.full((2, 2, 2), np.nan, dtype=float)
        values[0, 0, 0] = 1.0
        values[1, 1, 1] = 0.8
        speeds = np.asarray([10.0, 20.0], dtype=float)
        angles = np.deg2rad([0.0, 3.0])
        distances = np.asarray([10.0, 20.0], dtype=float)

        metrics = _robust_xpass_metrics_from_values(
            values,
            speeds,
            angles,
            distances,
            sigma_angle=0.15,
            sigma_speed=0.05,
            sigma_distance=0.05,
        )

        angle_weight = math.exp(-0.5 * (np.deg2rad(3.0) / np.deg2rad(1.5)) ** 2)
        expected = (1.0 + angle_weight * 0.8) / (1.0 + angle_weight)
        self.assertAlmostEqual(metrics[PHYSICAL_XPASS_METRIC_NOISE_KERNEL], expected, places=8)

    def test_angle_conditioned_noise_kernel_multiplies_speed_and_distance_weights(self) -> None:
        values = np.full((2, 1, 2), np.nan, dtype=float)
        values[0, 0, 0] = 1.0
        values[1, 0, 1] = 0.5
        speeds = np.asarray([20.0, 21.0], dtype=float)
        angles = np.deg2rad([0.0])
        distances = np.asarray([20.0, 21.0], dtype=float)

        metrics = _robust_xpass_metrics_from_values(values, speeds, angles, distances)

        speed_weight = math.exp(-0.5 * (1.0 / 1.0) ** 2)
        distance_weight = math.exp(-0.5 * (1.0 / 1.0) ** 2)
        combined_weight = speed_weight * distance_weight
        expected = (1.0 + combined_weight * 0.5) / (1.0 + combined_weight)
        self.assertAlmostEqual(metrics[PHYSICAL_XPASS_METRIC_NOISE_KERNEL], expected, places=8)

    def test_adaptive_refined_angles_include_local_two_and_half_degree_grid(self) -> None:
        coarse = np.deg2rad(np.arange(0.0, 360.0, 10.0))

        refined = refined_angle_grid_from_coarse_angles(coarse, [0], refine_angle_radius=10.0, angle_step=2.5)

        expected = np.mod(np.deg2rad(np.arange(-10.0, 10.0 + 0.1, 2.5)), 2.0 * np.pi)
        for angle in expected:
            self.assertTrue(np.any(np.isclose(refined, angle, atol=1e-12)))

    def test_robust_physical_xpass_skips_all_nan_coarse_angle_candidates_without_warning(self) -> None:
        def fake_simulate_passes(**kwargs):
            frame_count = int(kwargs["PLAYER_POS"].shape[0])
            player_count = len(kwargs["players"])
            angle_count = int(kwargs["phi_grid"].shape[1])
            result = FakeSimulationResult((frame_count, player_count, angle_count, 3), [])
            result.player_cum_prob[:] = np.nan
            return result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probs = compute_graph_physical_xpass_metrics_as_defaults(
                make_graph(),
                max_speed=3,
                speed_step=1,
                simulate_passes_fn=fake_simulate_passes,
            )

        self.assertFalse(any("All-NaN slice encountered" in str(item.message) for item in caught))
        self.assertTrue(np.isnan(float(probs["home_2"])))

    def test_batched_robust_physical_xpass_matches_batch_size_one(self) -> None:
        def make_fake(calls):
            def fake_simulate_passes(**kwargs):
                calls.append(kwargs)
                frame_count = int(kwargs["PLAYER_POS"].shape[0])
                players = kwargs["players"].tolist()
                target_player_index = players.index("home_2")
                angle_count = int(kwargs["phi_grid"].shape[1])
                speed = float(kwargs["v0_grid"][0, 0])
                updates = []
                for frame_index in range(frame_count):
                    target_x = float(kwargs["PLAYER_POS"][frame_index, target_player_index, 0])
                    for angle_index, angle in enumerate(kwargs["phi_grid"][frame_index]):
                        value = 0.2 + 0.02 * speed + 0.03 * np.cos(float(angle)) + target_x / 1000.0
                        updates.append((frame_index, target_player_index, angle_index, 1, value))
                return FakeSimulationResult((frame_count, len(players), angle_count, 3), updates)

            return fake_simulate_passes

        graphs = [make_graph(), make_graph()]
        calls_one: list[dict[str, object]] = []
        calls_many: list[dict[str, object]] = []

        batch_one = compute_graphs_physical_xpass_metrics_as_defaults(
            graphs,
            max_speed=4,
            speed_step=1,
            coarse_n_angles=4,
            refine_top_k_angles=1,
            refine_angle_radius=5,
            angle_step=5,
            simulate_passes_fn=make_fake(calls_one),
            batch_size=1,
        )
        batch_many = compute_graphs_physical_xpass_metrics_as_defaults(
            graphs,
            max_speed=4,
            speed_step=1,
            coarse_n_angles=4,
            refine_top_k_angles=1,
            refine_angle_radius=5,
            angle_step=5,
            simulate_passes_fn=make_fake(calls_many),
            batch_size=16,
        )

        self.assertEqual(len(calls_one), 8)
        self.assertEqual(len(calls_many), 4)
        for actual, expected in zip(batch_many, batch_one):
            pd.testing.assert_series_equal(actual, expected)

    def test_batched_robust_physical_xpass_call_shape_scales_by_batch_not_graph(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            frame_count = int(kwargs["PLAYER_POS"].shape[0])
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            angle_count = int(kwargs["phi_grid"].shape[1])
            speed = float(kwargs["v0_grid"][0, 0])
            updates = []
            for frame_index in range(frame_count):
                for angle_index, angle in enumerate(kwargs["phi_grid"][frame_index]):
                    updates.append((frame_index, target_player_index, angle_index, 1, 0.2 + 0.01 * speed + 0.01 * np.cos(float(angle))))
            return FakeSimulationResult((frame_count, len(players), angle_count, 3), updates)

        compute_graphs_physical_xpass_metrics_as_defaults(
            [make_graph() for _ in range(32)],
            max_speed=4,
            speed_step=1,
            coarse_n_angles=4,
            refine_top_k_angles=1,
            refine_angle_radius=5,
            angle_step=5,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[0]["PLAYER_POS"].shape[0], 16)

    def test_robust_physical_xpass_outputs_only_enabled_metrics(self) -> None:
        values = np.arange(1, 19, dtype=float).reshape(2, 3, 3)
        metrics = _robust_xpass_metrics_from_values(
            values,
            speeds=np.asarray([10.0, 20.0], dtype=float),
            angles=np.deg2rad([0.0, 2.5, 5.0]),
            distances=np.asarray([10.0, 20.0, 30.0], dtype=float),
            enabled_metrics=[PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
        )

        self.assertEqual(set(metrics), {PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN})
        self.assertNotIn(PHYSICAL_XPASS_METRIC_NOISE_KERNEL, metrics)

    def test_batched_robust_physical_xpass_omits_disabled_metric_columns(self) -> None:
        def fake_simulate_passes(**kwargs):
            frame_count = int(kwargs["PLAYER_POS"].shape[0])
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            angle_count = int(kwargs["phi_grid"].shape[1])
            updates = []
            for frame_index in range(frame_count):
                for angle_index in range(angle_count):
                    updates.append((frame_index, target_player_index, angle_index, 1, 0.5))
            return FakeSimulationResult((frame_count, len(players), angle_count, 3), updates)

        rows = compute_graphs_physical_xpass_metrics_as_defaults(
            [make_graph()],
            max_speed=4,
            speed_step=1,
            coarse_n_angles=4,
            refine_top_k_angles=1,
            refine_angle_radius=5,
            angle_step=5,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
            enabled_metrics=[PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
        )

        self.assertNotIn("home_2", rows[0].index)
        self.assertIn("home_2__max_xpass", rows[0].index)
        self.assertIn("home_2__topmean_xpass", rows[0].index)

    def test_robust_physical_xpass_filters_speed_grid_by_max_speed(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            frame_count = int(kwargs["PLAYER_POS"].shape[0])
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            angle_count = int(kwargs["phi_grid"].shape[1])
            speed = float(kwargs["v0_grid"][0, 0])
            updates = []
            for frame_index in range(frame_count):
                for angle_index in range(angle_count):
                    updates.append((frame_index, target_player_index, angle_index, 1, 0.1 + 0.01 * speed))
            return FakeSimulationResult((frame_count, len(players), angle_count, 3), updates)

        compute_graphs_physical_xpass_metrics_as_defaults(
            [make_graph()],
            max_speed=20,
            speed_step=1,
            coarse_n_angles=4,
            refine_top_k_angles=1,
            refine_angle_radius=5,
            angle_step=5,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )
        simulated_speeds = [float(call["v0_grid"][0, 0]) for call in calls]

        self.assertTrue(simulated_speeds)
        self.assertLessEqual(max(simulated_speeds), 20.0)
        self.assertNotIn(21.0, simulated_speeds)

    def test_physical_xpass_metadata_records_robust_metric_schema(self) -> None:
        metadata = physical_xpass_as_default_metadata(PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)

        self.assertEqual(metadata["metric_schema_version"], PHYSICAL_XPASS_METRIC_SCHEMA_VERSION)
        self.assertEqual(metadata["default_metric"], PHYSICAL_XPASS_DEFAULT_METRIC)
        self.assertEqual(metadata["noise_kernel_algorithm"], PHYSICAL_XPASS_NOISE_KERNEL_ALGORITHM)
        self.assertEqual(metadata["topmean_definition"], PHYSICAL_XPASS_TOPMEAN_DEFINITION)
        self.assertEqual(metadata["top_n"], PHYSICAL_XPASS_DEFAULT_TOP_N)
        self.assertEqual(metadata["sigma_angle_factor"], PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR)
        self.assertEqual(metadata["sigma_speed_factor"], PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR)
        self.assertEqual(metadata["sigma_distance_factor"], PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR)
        self.assertEqual(
            metadata["available_metrics"],
            [PHYSICAL_XPASS_METRIC_NOISE_KERNEL, PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
        )
        self.assertEqual(metadata["disabled_metrics"], [])
        self.assertEqual(metadata["max_speed"], 22.0)
        self.assertEqual(metadata["speed_step"], 1.0)
        self.assertEqual(metadata["coarse_n_angles"], 36)
        self.assertEqual(metadata["refine_top_k_angles"], 2)
        self.assertEqual(metadata["refine_angle_radius_deg"], 10.0)
        self.assertEqual(metadata["angle_step_deg"], 2.5)

    def test_physical_xpass_attach_selects_exported_metric_columns(self) -> None:
        labels = make_label(action_index=5)
        rows = pd.DataFrame(
            [
                {
                    "match_id": "m1",
                    "action_index": 5,
                    PHYSICAL_XPASS_BALL_Z_COLUMN: 1.3,
                    "home_2": 0.4,
                    physical_xpass_metric_column("home_2", PHYSICAL_XPASS_METRIC_MAX): 0.9,
                    physical_xpass_metric_column("home_2", PHYSICAL_XPASS_METRIC_TOPMEAN): 0.7,
                    physical_xpass_nearest_opponent_distance_column("home_2"): 20.0,
                }
            ]
        ).set_index("action_index", drop=False)

        default_graph = attach_physical_xpass_to_graph(make_graph(), labels, rows, match_id="m1", require_observed_target=False)
        max_graph = attach_physical_xpass_to_graph(
            make_graph(),
            labels,
            rows,
            match_id="m1",
            require_observed_target=False,
            metric=PHYSICAL_XPASS_METRIC_MAX,
        )
        topmean_graph = attach_physical_xpass_to_graph(
            make_graph(),
            labels,
            rows,
            match_id="m1",
            require_observed_target=False,
            metric=PHYSICAL_XPASS_METRIC_TOPMEAN,
        )

        self.assertAlmostEqual(float(default_graph.physical_xpass[1]), 0.4)
        self.assertAlmostEqual(float(max_graph.physical_xpass[1]), 0.9)
        self.assertAlmostEqual(float(topmean_graph.physical_xpass[1]), 0.7)
        self.assertAlmostEqual(float(default_graph.physical_xpass_nearest_opponent_distance[1]), 20.0)
        self.assertAlmostEqual(float(getattr(default_graph, PHYSICAL_XPASS_BALL_Z_ATTR)[1]), 1.3)

    def test_physical_xpass_attach_requires_cached_ball_z_when_requested(self) -> None:
        labels = make_label(action_index=5)
        rows = pd.DataFrame([{"match_id": "m1", "action_index": 5, "home_2": 0.4}]).set_index(
            "action_index",
            drop=False,
        )

        with self.assertRaisesRegex(ValueError, "ball_z"):
            attach_physical_xpass_to_graph(
                make_graph(),
                labels,
                rows,
                match_id="m1",
                require_observed_target=False,
                require_ball_z=True,
            )

    def test_physical_xpass_metric_flags_select_inference_metric(self) -> None:
        self.assertEqual(physical_xpass_metric({"task": "pass_success"}), PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertEqual(physical_xpass_metric({"task": "pass_success", "x_pass_version": "max"}), PHYSICAL_XPASS_METRIC_MAX)
        self.assertEqual(
            physical_xpass_metric({"task": "pass_success", "x_pass_version": "top25"}),
            PHYSICAL_XPASS_METRIC_TOPMEAN,
        )
        self.assertEqual(
            physical_xpass_metric({"task": "pass_success", "x_pass_version": "noise-kernel"}),
            PHYSICAL_XPASS_METRIC_NOISE_KERNEL,
        )
        self.assertEqual(
            physical_xpass_metric({"task": "pass_success", "pc_xpass": True}),
            PC_XPASS_METRIC_TOP10,
        )
        self.assertEqual(
            physical_xpass_metric({"task": "pass_success", "pc_xpass": True, "x_pass_version": "top50"}),
            "top50_xpass",
        )
        with self.assertRaises(ValueError):
            physical_xpass_metric({"task": "pass_success", "pc_xpass": True, "x_pass_version": "noise-kernel"})

    def test_pc_xpass_raw_control_and_conditional_normalization(self) -> None:
        raw = pc_xpass_raw_control(np.asarray([-0.2, 0.0, 0.5], dtype=float))

        self.assertAlmostEqual(float(raw[0]), 0.5)
        self.assertGreater(float(raw[1]), 0.98)
        self.assertGreater(float(raw[2]), 0.999)
        custom = pc_xpass_raw_control_with_params(np.asarray([-0.3, 0.0], dtype=float), power=15.0, inflection_point=0.3)
        self.assertAlmostEqual(float(custom[0]), 0.5)
        self.assertGreater(float(custom[1]), 0.98)

        under_one = np.asarray([0.2, 0.3, 0.4], dtype=float)
        over_one = np.asarray([0.5, 0.5, 0.5], dtype=float)
        np.testing.assert_allclose(pc_xpass_normalize_if_sum_above_one(under_one), under_one)
        np.testing.assert_allclose(pc_xpass_normalize_if_sum_above_one(over_one), np.asarray([1 / 3, 1 / 3, 1 / 3]))

    def test_pc_xpass_endpoint_gamma_normalization(self) -> None:
        under_one = np.asarray([0.2, 0.3, 0.4], dtype=float)
        over_one = np.asarray([1.0, 0.5, 0.5], dtype=float)

        np.testing.assert_allclose(pc_xpass_endpoint_control_probabilities(under_one, gamma=2.0), under_one)
        np.testing.assert_allclose(pc_xpass_endpoint_control_probabilities(over_one, gamma=1.0), np.asarray([0.5, 0.25, 0.25]))
        np.testing.assert_allclose(
            pc_xpass_endpoint_control_probabilities(over_one, gamma=2.0),
            np.asarray([1.0 / 1.5, 0.25 / 1.5, 0.25 / 1.5]),
        )

    def test_pc_xpass_lane_survival_uses_player_max_then_independent_product(self) -> None:
        raw = np.zeros((3, 1, 1, 4), dtype=float)
        raw[0, 0, 0, :] = [0.2, 0.7, 0.6, 0.1]
        raw[1, 0, 0, :] = [0.0, 0.0, 0.2, 0.4]
        raw[2, 0, 0, :] = [np.nan, np.nan, np.nan, np.nan]

        survival = pc_xpass_lane_survival_from_raw(raw)

        self.assertAlmostEqual(float(survival[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(survival[0, 0, 1]), 0.8)
        self.assertAlmostEqual(float(survival[0, 0, 2]), 0.3)
        self.assertAlmostEqual(float(survival[0, 0, 3]), (1.0 - 0.7) * (1.0 - 0.2))

    def test_compute_graph_pc_xpass_exports_default_top10_and_detail_columns(self) -> None:
        row = compute_graph_pc_xpass_metrics(make_graph(), max_speed=7, speed_step=2, angle_step=90)

        self.assertIn("home_2", row.index)
        self.assertIn(physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP10), row.index)
        self.assertNotIn(physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP25), row.index)
        self.assertIn("home_2__lane_survival", row.index)
        self.assertIn("home_2__control_prob", row.index)
        self.assertIn("home_2__target_x", row.index)
        self.assertAlmostEqual(
            float(row["home_2"]),
            float(row[physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP10)]),
        )
        self.assertTrue(math.isfinite(float(row[physical_xpass_metric_column("home_2", PHYSICAL_XPASS_METRIC_MAX)])))

    def test_compute_graph_pc_xpass_exports_multiple_top_n_values(self) -> None:
        row = compute_graph_pc_xpass_metrics(
            make_graph(),
            max_speed=7,
            speed_step=2,
            angle_step=90,
            top_n=10,
            top_n_values=[5, 10, 25],
            enabled_metrics=[PHYSICAL_XPASS_METRIC_MAX, "top5_xpass", PC_XPASS_METRIC_TOP10, PC_XPASS_METRIC_TOP25],
        )

        self.assertIn("home_2", row.index)
        self.assertIn(physical_xpass_metric_column("home_2", "top5_xpass"), row.index)
        self.assertIn(physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP10), row.index)
        self.assertIn(physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP25), row.index)
        self.assertAlmostEqual(
            float(row["home_2"]),
            float(row[physical_xpass_metric_column("home_2", PC_XPASS_METRIC_TOP10)]),
        )

    def test_compute_graph_pc_xpass_splits_lane_and_control_teammate_ignoring(self) -> None:
        graph = make_graph(["home_1", "home_2", "home_3", "away_4"])
        graph.x[0, config.NODE_FEATURE_X] = 0.0
        graph.x[1, config.NODE_FEATURE_X] = 20.0
        graph.x[2, config.NODE_FEATURE_X] = 10.0
        graph.x[3, config.NODE_FEATURE_X] = 50.0
        graph.x[:, config.NODE_FEATURE_Y] = torch.tensor([34.0, 34.0, 34.0, 50.0])
        graph.x[:, config.NODE_FEATURE_POSS_DIST] = torch.abs(graph.x[:, config.NODE_FEATURE_X] - graph.x[0, config.NODE_FEATURE_X])
        graph.x[2, config.NODE_FEATURE_IS_TEAMMATE] = 1.0
        graph.x[3, config.NODE_FEATURE_IS_TEAMMATE] = 0.0

        lane_considers_teammate = compute_graph_pc_xpass_metrics(
            graph,
            max_speed=7,
            speed_step=2,
            angle_step=90,
            ignore_teammates_control=True,
        )
        lane_ignores_teammate = compute_graph_pc_xpass_metrics(
            graph,
            max_speed=7,
            speed_step=2,
            angle_step=90,
            ignore_teammates_lane_survival=True,
            ignore_teammates_control=True,
        )
        control_considers_teammate = compute_graph_pc_xpass_metrics(
            graph,
            max_speed=7,
            speed_step=2,
            angle_step=90,
            ignore_teammates_lane_survival=True,
        )
        control_ignores_teammate = compute_graph_pc_xpass_metrics(
            graph,
            max_speed=7,
            speed_step=2,
            angle_step=90,
            ignore_teammates_lane_survival=True,
            ignore_teammates_control=True,
        )

        self.assertGreater(float(lane_ignores_teammate["home_2__lane_survival"]), float(lane_considers_teammate["home_2__lane_survival"]))
        self.assertGreater(float(lane_ignores_teammate["home_2"]), float(lane_considers_teammate["home_2"]))
        self.assertGreater(float(control_ignores_teammate["home_2__control_prob"]), float(control_considers_teammate["home_2__control_prob"]))
        self.assertGreater(float(control_ignores_teammate["home_2"]), float(control_considers_teammate["home_2"]))

    def test_pc_xpass_metadata_records_split_teammate_policies(self) -> None:
        metadata = pc_xpass_metadata(
            PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
            ignore_teammates_lane_survival=True,
            ignore_teammates_control=False,
            top_n=10,
            top_n_values=[5, 10, 25],
            available_metrics=[PHYSICAL_XPASS_METRIC_MAX, "top5_xpass", PC_XPASS_METRIC_TOP10, PC_XPASS_METRIC_TOP25],
            control_function_power=20.0,
            control_function_inflection_point=0.2,
            control_function_gamma=2.0,
        )

        self.assertTrue(metadata["ignore_teammates_lane_survival"])
        self.assertFalse(metadata["ignore_teammates_control"])
        self.assertEqual(metadata["teammate_policy"], "split_teammate_policy")
        self.assertEqual(metadata["lane_survival_policy"], "non_passer_non_receiver_opponents_only")
        self.assertEqual(metadata["control_policy"], "all_players")
        self.assertEqual(metadata["control_function"], "sigmoid")
        self.assertEqual(metadata["control_function_power"], 20.0)
        self.assertEqual(metadata["control_function_inflection_point"], 0.2)
        self.assertEqual(metadata["control_function_gamma"], 2.0)
        self.assertEqual(metadata["lane_survival_aggregation"], "per_player_max_then_independent_product")
        self.assertEqual(metadata["endpoint_normalization"], "gamma_power_if_sum_gt_1")
        self.assertEqual(metadata["top_n_values"], [5, 10, 25])
        self.assertEqual(metadata["available_x_pass_versions"], ["max", "top5", "top10", "top25"])

    def test_pc_xpass_cache_accepts_old_single_teammate_policy_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pc_xpass"
            cache_dir.mkdir(parents=True, exist_ok=True)
            metadata = pc_xpass_metadata(PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE)
            metadata.pop("ignore_teammates_lane_survival", None)
            metadata.pop("ignore_teammates_control", None)
            metadata.pop("control_policy", None)
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            prewarm_physical_xpass_runtime_cache(
                [],
                cache_dir=cache_dir,
                source=PC_XPASS_SOURCE,
                teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
                dry_run=True,
            )
            with self.assertRaisesRegex(ValueError, "ignore_teammates_lane_survival"):
                prewarm_physical_xpass_runtime_cache(
                    [],
                    cache_dir=cache_dir,
                    source=PC_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    dry_run=True,
                )

    def test_pc_xpass_cache_rejects_old_control_function_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pc_xpass"
            cache_dir.mkdir(parents=True, exist_ok=True)
            metadata = pc_xpass_metadata(PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER)
            for key in [
                "control_function",
                "control_function_power",
                "control_function_inflection_point",
                "control_function_gamma",
                "lane_survival_aggregation",
                "endpoint_normalization",
            ]:
                metadata.pop(key, None)
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "control_function"):
                prewarm_physical_xpass_runtime_cache(
                    [],
                    cache_dir=cache_dir,
                    source=PC_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    dry_run=True,
                )

    def test_runtime_visualization_xpass_loader_selects_metric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_runtime_visualization_xpass_cache(cache_dir, action_indexes=[3])

            default = load_runtime_physical_xpass_visualization_component(cache_dir, "match_1", 3)
            max_row = load_runtime_physical_xpass_visualization_component(
                cache_dir,
                "match_1",
                3,
                metric=PHYSICAL_XPASS_METRIC_MAX,
            )
            topmean = load_runtime_physical_xpass_visualization_component(
                cache_dir,
                "match_1",
                3,
                metric=PHYSICAL_XPASS_METRIC_TOPMEAN,
            )

        self.assertAlmostEqual(float(default["home_2"]), 0.4)
        self.assertAlmostEqual(float(max_row["home_2"]), 0.9)
        self.assertAlmostEqual(float(topmean["home_2"]), 0.6)

    def test_runtime_visualization_xpass_loader_prefers_pc_default_unsuffixed_top25(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pc_xpass"
            matches_dir = cache_dir / "matches"
            matches_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 3,
                        "home_2": 0.42,
                        "home_2__top25_xpass": 0.84,
                    }
                ]
            ).to_parquet(matches_dir / "match_1.parquet", index=False)

            top25 = load_runtime_physical_xpass_visualization_component(
                cache_dir,
                "match_1",
                3,
                metric=PC_XPASS_METRIC_TOP25,
            )

        self.assertAlmostEqual(float(top25["home_2"]), 0.42)

    def test_runtime_visualization_xpass_loader_falls_back_to_legacy_top10_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_runtime_visualization_xpass_cache(cache_dir, action_indexes=[3])
            rows = pd.read_parquet(cache_dir / "matches" / "match_1.parquet")
            rows = rows.rename(
                columns={
                    "home_1__topmean_xpass": "home_1__top10mean_xpass",
                    "home_2__topmean_xpass": "home_2__top10mean_xpass",
                }
            )
            rows.to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)

            topmean = load_runtime_physical_xpass_visualization_component(
                cache_dir,
                "match_1",
                3,
                metric=PHYSICAL_XPASS_METRIC_TOPMEAN,
            )

        self.assertAlmostEqual(float(topmean["home_2"]), 0.6)

    def test_runtime_visualization_xpass_loader_allows_cache_setting_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_runtime_visualization_xpass_cache(cache_dir, action_indexes=[3])
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                max_speed=30,
                speed_step=1,
            )
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            max_row = load_runtime_physical_xpass_visualization_component(
                cache_dir,
                "match_1",
                3,
                metric=PHYSICAL_XPASS_METRIC_MAX,
            )

        self.assertAlmostEqual(float(max_row["home_2"]), 0.9)

    def test_runtime_visualization_xpass_loader_rejects_missing_metric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                max_speed=22,
                speed_step=1,
            )
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame([{"match_id": "match_1", "action_index": 3, "home_2": 0.4}]).to_parquet(
                cache_dir / "matches" / "match_1.parquet",
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "does not contain columns"):
                load_runtime_physical_xpass_visualization_component(
                    cache_dir,
                    "match_1",
                    3,
                    metric=PHYSICAL_XPASS_METRIC_MAX,
                )

    def test_visualize_action_components_uses_runtime_xpass_cache_by_default(self) -> None:
        match = SimpleNamespace(match_id="match_1")
        loaded_models: dict[str, object] = {}
        with patch.object(visualize_action_components, "get_runtime_physical_xpass_dir", return_value=Path("runtime_cache")):
            with patch.object(
                visualize_action_components,
                "load_runtime_physical_xpass_visualization_component",
                return_value=pd.Series({"home_2": 0.7}, dtype=float),
            ) as load_xpass:
                with patch.object(visualize_action_components, "render_component") as render_component:
                    visualize_action_components.render_action_components(
                        match=match,
                        loaded_models=loaded_models,
                        feature_root=Path("feature_root"),
                        device="cpu",
                        action_index=5,
                        display_action_id="a5",
                        output_dir=Path("out"),
                        show_physical_xpass=True,
                        physical_cache_dir=None,
                        physical_xpass_metric_name=PHYSICAL_XPASS_METRIC_TOPMEAN,
                        physical_xpass_version_name="top10",
                        rendered_components=["unused_component"],
                    )

        load_xpass.assert_called_once_with(
            Path("runtime_cache"),
            "match_1",
            5,
            metric=PHYSICAL_XPASS_METRIC_TOPMEAN,
            x_pass_version="top10",
            frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
        )
        render_component.assert_called_once()
        self.assertEqual(render_component.call_args.kwargs["component_name"], "physical_xpass")

    def test_run_and_visualize_hawkeye_physical_xpass_uses_selected_metric(self) -> None:
        frame_meta = pd.DataFrame(
            [{"possession_prefix": "home", "possessor_object_id": "home_1", "abs_time": 0.0}],
            index=pd.Index([0], name="frame_id"),
        )
        tracking = pd.DataFrame({"id": ["s1"], "ball_x": [0.0], "ball_y": [0.0]}, index=pd.Index([0], name="frame_id"))
        situation = SimpleNamespace(
            situation_id="s1",
            match_id="s1",
            tracking=tracking,
            frame_meta=frame_meta,
            labels=torch.empty((0, len(LABEL_COLUMNS))),
            graph_features_0=[],
        )
        args = SimpleNamespace(
            tracking_csv="tracking.csv",
            freeze_ballreceipt=True,
            show_physical_xpass=True,
            physical_cache_dir="hawkeye_cache",
            x_pass_version="max",
            gif=True,
            show_trajectories=False,
        )

        with patch.object(run_and_visualize_hawkeye, "build_hawkeye_situation", return_value=(situation, {}, {})):
            with patch.object(
                run_and_visualize_hawkeye,
                "load_runtime_physical_xpass_visualization_table",
                return_value=pd.DataFrame({"home_2": [0.9]}, index=pd.Index([0], name="action_index")),
            ) as load_xpass:
                with patch.object(run_and_visualize_hawkeye, "render_frame_image", return_value=object()):
                    with patch.object(run_and_visualize_hawkeye, "save_animation") as save_animation:
                        run_and_visualize_hawkeye.render_situation(
                            situation_id="s1",
                            tracking=pd.DataFrame({"id": ["s1"]}),
                            ball=pd.DataFrame(),
                            model_specs={},
                            graph_schema={"add_v_edge_features": False},
                            args=args,
                            device="cpu",
                            output_root=Path("out"),
                            rendered_components=[],
                        )

        load_xpass.assert_called_once_with(
            "hawkeye_cache",
            "s1",
            [0],
            metric=PHYSICAL_XPASS_METRIC_MAX,
            x_pass_version="max",
        )
        save_animation.assert_called_once()
        self.assertEqual(save_animation.call_args.args[1], Path("out") / "s1" / "physical_xpass.gif")

    def test_pass_distance_is_reserved_sidecar_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_2": 0.64,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)

            component = load_physical_xpass_component(cache_dir, "match_1", 5)

        self.assertIn("home_2", component.index)
        self.assertNotIn(PHYSICAL_XPASS_PASS_DISTANCE_COLUMN, component.index)

    def test_load_physical_xpass_match_filters_duplicate_action_indexes_by_frame_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
                        "home_2": 0.31,
                    },
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE,
                        "home_2": 0.72,
                    },
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)

            action_rows = load_physical_xpass_match(cache_dir, "match_1", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_ACTION)
            receive_rows = load_physical_xpass_match(cache_dir, "match_1", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE)
            with self.assertRaisesRegex(ValueError, "frame_scope"):
                load_physical_xpass_match(cache_dir, "match_1")

        self.assertEqual(action_rows.index.tolist(), [5])
        self.assertEqual(receive_rows.index.tolist(), [5])
        self.assertAlmostEqual(float(action_rows.loc[5, "home_2"]), 0.31)
        self.assertAlmostEqual(float(receive_rows.loc[5, "home_2"]), 0.72)

    def test_load_physical_xpass_match_treats_legacy_unscoped_rows_as_action_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            pd.DataFrame([{"match_id": "match_1", "action_index": 5, "home_2": 0.44}]).to_parquet(
                cache_dir / "matches" / "match_1.parquet",
                index=False,
            )

            action_rows = load_physical_xpass_match(cache_dir, "match_1", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_ACTION)
            receive_rows = load_physical_xpass_match(cache_dir, "match_1", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE)

        self.assertEqual(action_rows.index.tolist(), [5])
        self.assertTrue(receive_rows.empty)
        self.assertAlmostEqual(float(action_rows.loc[5, "home_2"]), 0.44)

    def test_observed_pass_distance_uses_passer_and_intended_receiver(self) -> None:
        graph = make_graph()
        label = make_label(intent_index=1)

        self.assertAlmostEqual(observed_pass_distance(graph, label), 20.0)

    def test_pass_success_inference_skips_non_pass_actions_missing_from_static_physical_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_physical_inference_cache(cache_dir, [0, 2])
            match = make_physical_inference_match([0, 1, 2], [1, 0, 1])
            model = DummyPhysicalInferenceModel(cache_dir)

            probs, _ = inference.inference_gnn(match, model, device="cpu", post_action=False)

        self.assertEqual(probs.index.tolist(), [0, 2])
        self.assertEqual(match.physical_xpass_skipped_actions["pass_success"]["skipped_count"], 1)
        self.assertEqual(match.physical_xpass_skipped_actions["pass_success"]["sample_action_indexes"], [1])

    def test_pass_success_inference_skips_pass_actions_missing_from_static_physical_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_physical_inference_cache(cache_dir, [0])
            match = make_physical_inference_match([0, 1], [1, 1])
            model = DummyPhysicalInferenceModel(cache_dir)

            probs, _ = inference.inference_gnn(match, model, device="cpu", post_action=False)

        self.assertEqual(probs.index.tolist(), [0])
        self.assertEqual(match.physical_xpass_skipped_actions["pass_success"]["skipped_count"], 1)
        self.assertEqual(match.physical_xpass_skipped_actions["pass_success"]["sample_action_indexes"], [1])

    def test_pass_success_inference_raises_specific_error_when_all_physical_rows_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            write_physical_inference_cache(cache_dir, [])
            match = make_physical_inference_match([0, 1], [1, 0])
            model = DummyPhysicalInferenceModel(cache_dir)

            with self.assertRaises(inference.PhysicalXPassNoUsableRowsError):
                inference.inference_gnn(match, model, device="cpu", post_action=False)

        self.assertEqual(match.physical_xpass_skipped_actions["pass_success"]["skipped_count"], 2)

    def test_pass_success_inference_uses_receive_scope_for_post_action_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
                        "home_2": 0.11,
                    },
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE,
                        "home_2": 0.91,
                    },
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)
            match = make_physical_inference_match([5], [1])
            model = DummyBaselineInferenceModel(cache_dir)

            graphs, labels = inference.filter_missing_physical_xpass_rows_for_inference(
                match,
                match.graph_features_0,
                match.labels,
                model,
                post_action=True,
            )
            attached = inference.attach_physical_xpass_for_inference(match, graphs, labels, model, post_action=True)

        self.assertEqual(int(labels[0, LABEL_INDEX["action_index"]].item()), 5)
        self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.91)

    def test_pass_success_inference_does_not_use_action_scope_for_post_action_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 5,
                        PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
                        "home_2": 0.11,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)
            match = make_physical_inference_match([5], [1])
            model = DummyBaselineInferenceModel(cache_dir)

            with self.assertRaises(inference.PhysicalXPassNoUsableRowsError):
                inference.filter_missing_physical_xpass_rows_for_inference(
                    match,
                    match.graph_features_0,
                    match.labels,
                    model,
                    post_action=True,
                )

        skipped = match.physical_xpass_skipped_actions["pass_success"]
        self.assertEqual(skipped["skipped_count"], 1)
        self.assertIn("missing_row_receive_frame_id", skipped["reason"])

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

    def test_max_player_cum_prob_package_max_filters_speed_grid_by_max_speed(self) -> None:
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
            max_speed=20,
            simulate_passes_fn=fake_simulate_passes,
        )
        expected_speeds = as_default_v0_values(max_speed=20)

        self.assertAlmostEqual(float(probs["home_2"]), 0.72)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["v0_grid"].shape, (1, len(expected_speeds)))
        self.assertLessEqual(float(calls[0]["v0_grid"][0, -1]), 20.0)
        np.testing.assert_allclose(calls[0]["v0_grid"][0], expected_speeds)

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

    def test_batched_consider_teammates_filters_speed_grid_by_max_speed(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("home_2")
            return FakeSimulationResult((2, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, 0.44)])

        probs = compute_graphs_max_player_cum_prob_as_defaults(
            [make_graph(), make_graph()],
            consider_teammates=True,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            max_speed=20,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )
        expected_speeds = as_default_v0_values(max_speed=20)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["v0_grid"].shape, (2, len(expected_speeds)))
        np.testing.assert_allclose(calls[0]["v0_grid"][0], expected_speeds)
        self.assertAlmostEqual(float(probs[0]["home_2"]), 0.44)

    def test_batched_ignore_teammates_uses_one_package_call_for_compatible_graphs(self) -> None:
        calls = []

        def make_extra_teammate_graph(node_ids: list[str]) -> Data:
            graph = make_graph(node_ids)
            graph.x[2, config.NODE_FEATURE_IS_TEAMMATE] = 1.0
            graph.x[3, config.NODE_FEATURE_IS_TEAMMATE] = 0.0
            return graph

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            return FakeSimulationResult(
                (4, len(players), AS_DEFAULT_N_ANGLES, 3),
                [
                    (0, target_player_index, 0, 2, 0.41),
                    (1, target_player_index, 0, 2, 0.42),
                    (2, target_player_index, 0, 2, 0.61),
                    (3, target_player_index, 0, 2, 0.62),
                ],
            )

        graphs = [
            make_extra_teammate_graph(["home_1", "home_2", "home_3", "away_4"]),
            make_extra_teammate_graph(["other_1", "other_2", "other_3", "away_9"]),
        ]
        probs = compute_graphs_max_player_cum_prob_as_defaults(
            graphs,
            consider_teammates=False,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["PLAYER_POS"].shape, (4, 3, 4))
        self.assertEqual(calls[0]["players"].tolist(), ["passer", "target_player", "defender_0"])
        self.assertEqual(calls[0]["player_teams"].tolist(), ["attack", "attack", "defense"])
        self.assertEqual(calls[0]["passers"].tolist(), ["passer", "passer", "passer", "passer"])
        self.assertTrue(calls[0]["exclude_passer"])
        self.assertAlmostEqual(float(probs[0]["home_2"]), 0.41)
        self.assertAlmostEqual(float(probs[0]["home_3"]), 0.42)
        self.assertAlmostEqual(float(probs[1]["other_2"]), 0.61)
        self.assertAlmostEqual(float(probs[1]["other_3"]), 0.62)

    def test_batched_ignore_teammates_exact_mode_batches_by_speed(self) -> None:
        calls = []
        expected_speeds = as_default_v0_values(max_speed=20)

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            speed = float(kwargs["v0_grid"][0, 0])
            value = 0.9 if np.isclose(speed, expected_speeds[-1]) else 0.2
            return FakeSimulationResult(
                (2, len(players), AS_DEFAULT_N_ANGLES, 3),
                [
                    (0, target_player_index, 0, 2, value),
                    (1, target_player_index, 0, 2, value - 0.1),
                ],
            )

        probs = compute_graphs_max_player_cum_prob_as_defaults(
            [make_graph(), make_graph(["other_1", "other_2", "away_9"])],
            consider_teammates=False,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            max_speed=20,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        self.assertEqual(len(calls), len(expected_speeds))
        self.assertEqual(calls[0]["v0_grid"].shape, (2, 1))
        self.assertEqual(calls[0]["v0_prob_aggregation_mode"], "mean")
        self.assertTrue(calls[0]["exclude_passer"])
        np.testing.assert_allclose([float(call["v0_grid"][0, 0]) for call in calls], expected_speeds)
        self.assertAlmostEqual(float(probs[0]["home_2"]), 0.9)
        self.assertAlmostEqual(float(probs[1]["other_2"]), 0.8)

    def test_batched_ignore_teammates_matches_batch_size_one(self) -> None:
        def fake_simulate_passes(**kwargs):
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            updates = []
            for frame_index in range(int(kwargs["PLAYER_POS"].shape[0])):
                target_x = float(kwargs["PLAYER_POS"][frame_index, target_player_index, 0])
                value = 0.3 + (target_x + 52.5) / 200.0
                updates.append((frame_index, target_player_index, 0, 2, value))
            return FakeSimulationResult(
                (int(kwargs["PLAYER_POS"].shape[0]), len(players), AS_DEFAULT_N_ANGLES, 3),
                updates,
            )

        graph_a = make_graph()
        graph_b = make_graph(["other_1", "other_2", "away_9"])
        graph_b.x[1, config.NODE_FEATURE_X] = 35.0

        batch_one = compute_graphs_max_player_cum_prob_as_defaults(
            [graph_a, graph_b],
            consider_teammates=False,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=1,
        )
        batch_many = compute_graphs_max_player_cum_prob_as_defaults(
            [graph_a, graph_b],
            consider_teammates=False,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            simulate_passes_fn=fake_simulate_passes,
            batch_size=16,
        )

        pd.testing.assert_series_equal(batch_many[0], batch_one[0])
        pd.testing.assert_series_equal(batch_many[1], batch_one[1])

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

    def test_exact_mode_filters_speed_grid_by_max_speed(self) -> None:
        calls = []

        def fake_simulate_passes(**kwargs):
            calls.append(kwargs)
            players = kwargs["players"].tolist()
            target_player_index = players.index("target_player")
            return FakeSimulationResult((1, len(players), AS_DEFAULT_N_ANGLES, 3), [(0, target_player_index, 0, 2, 0.5)])

        compute_graph_max_player_cum_prob_as_defaults(
            make_graph(),
            max_speed=20,
            simulate_passes_fn=fake_simulate_passes,
        )
        expected_speeds = as_default_v0_values(max_speed=20)

        self.assertEqual(len(calls), len(expected_speeds))
        self.assertEqual(calls[0]["v0_grid"].shape, (1, 1))
        self.assertLessEqual(float(calls[-1]["v0_grid"][0, 0]), 20.0)
        np.testing.assert_allclose([float(call["v0_grid"][0, 0]) for call in calls], expected_speeds)

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

    def test_sidecar_attach_applies_physical_xpass_floor_before_logit(self) -> None:
        graph = make_graph(["home_1", "home_2"])
        label = make_label()
        sidecar = pd.DataFrame([{"match_id": "m1", "action_index": 7, "home_2": 0.01}]).set_index("action_index", drop=False)

        attached = attach_physical_xpass_to_graph(graph, label, sidecar, match_id="m1", floor=0.2)

        probs = getattr(attached, PHYSICAL_XPASS_PROB_ATTR)
        logits = getattr(attached, PHYSICAL_XPASS_LOGIT_ATTR)
        self.assertTrue(torch.allclose(probs, torch.tensor([0.5, 0.2])))
        self.assertAlmostEqual(float(logits[1]), float(torch.logit(torch.tensor(0.2))), places=6)

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
                    {"match_id": "match_1", "action_index": 7, "home_2": 0.05},
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
                physical_xpass_floor=0.2,
            )

        expected_skips = {"invalid_pass_success_target:target_is_possessor": 1}
        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(physical), 1)
        self.assertEqual(baseline.skipped_rows, expected_skips)
        self.assertEqual(physical.skipped_rows, expected_skips)
        self.assertEqual(int(baseline.labels[0, LABEL_INDEX["action_index"]].item()), 7)
        self.assertEqual(int(physical.labels[0, LABEL_INDEX["action_index"]].item()), 7)
        self.assertTrue(hasattr(physical.features[0], PHYSICAL_XPASS_PROB_ATTR))
        self.assertAlmostEqual(float(getattr(physical.features[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.2)

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
                            "physical_xpass_floor": 0.2,
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
        self.assertEqual(model.args["physical_xpass_floor"], 0.2)

    def test_runtime_physical_xpass_cache_computes_writes_then_reuses(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.73}, dtype=float)],
            ) as compute:
                attached, stats = attach_physical_xpass_cached_online_to_graphs(
                    [make_graph()],
                    labels,
                    cache_dir=cache_dir,
                    match_id="runtime_match",
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

            compute.assert_called_once()
            self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.73)
            self.assertEqual(stats["cache_hits"], 0)
            self.assertEqual(stats["cache_misses"], 1)
            self.assertEqual(stats["cache_written"], 1)
            self.assertTrue((cache_dir / "metadata.json").exists())
            self.assertTrue((cache_dir / "matches" / "runtime_match.parquet").exists())

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute_again:
                attached_again, reuse_stats = attach_physical_xpass_cached_online_to_graphs(
                    [make_graph()],
                    labels,
                    cache_dir=cache_dir,
                    match_id="runtime_match",
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

            compute_again.assert_not_called()
            self.assertAlmostEqual(float(getattr(attached_again[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.73)
            self.assertEqual(reuse_stats["cache_hits"], 1)
            self.assertEqual(reuse_stats["cache_misses"], 0)

    def test_runtime_physical_xpass_cache_recomputes_old_metric_definition_rows(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            )
            metadata.pop("noise_kernel_algorithm", None)
            metadata.pop("topmean_definition", None)
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_2": 0.12,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.91}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")
            updated_metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

        compute.assert_called_once()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_written"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.91)
        self.assertEqual(updated_metadata["noise_kernel_algorithm"], PHYSICAL_XPASS_NOISE_KERNEL_ALGORITHM)
        self.assertEqual(updated_metadata["topmean_definition"], PHYSICAL_XPASS_TOPMEAN_DEFINITION)
        self.assertEqual(updated_metadata["top_n"], PHYSICAL_XPASS_DEFAULT_TOP_N)

    def test_runtime_physical_xpass_cache_recomputes_different_sigma_factor_rows(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                sigma_angle=0.15,
            )
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_2": 0.12,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.92}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    sigma_angle=PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR,
                    num_workers=1,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")
            updated_metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

        compute.assert_called_once()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.92)
        self.assertEqual(updated_metadata["sigma_angle_factor"], PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR)

    def test_runtime_physical_xpass_cache_recomputes_different_available_metric_rows(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                available_metrics=[PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
            )
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_2__max_xpass": 0.12,
                        "home_2__topmean_xpass": 0.13,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.91, "home_2__max_xpass": 0.92, "home_2__topmean_xpass": 0.93}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")
            updated_metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))

        compute.assert_called_once()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.91)
        self.assertEqual(
            updated_metadata["available_metrics"],
            [PHYSICAL_XPASS_METRIC_NOISE_KERNEL, PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
        )

    def test_runtime_physical_xpass_cache_copies_reuse_row_and_fills_pass_distance(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "runtime" / "physical_xpass"
            reuse_dir = root / "feature" / "physical_xpass"
            (reuse_dir / "matches").mkdir(parents=True)
            (reuse_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                    | {"physical_eps": 1e-4}
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        "home_2": 0.77,
                    }
                ]
            ).to_parquet(reuse_dir / "matches" / "runtime_match.parquet", index=False)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    reuse_cache_dir=reuse_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_not_called()
        self.assertEqual(stats["copied_from_reuse"], 1)
        self.assertEqual(stats["pass_distance_filled"], 1)
        self.assertEqual(stats["ball_z_filled"], 1)
        self.assertEqual(stats["cache_written"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.77)
        self.assertAlmostEqual(float(rows.loc[5, PHYSICAL_XPASS_PASS_DISTANCE_COLUMN]), 20.0)
        self.assertAlmostEqual(float(rows.loc[5, PHYSICAL_XPASS_BALL_Z_COLUMN]), 0.0)

    def test_runtime_physical_xpass_cache_fills_nearest_opponent_distance_without_recomputing_xpass(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_2": 0.77,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_not_called()
        self.assertEqual(stats["cache_misses"], 0)
        self.assertEqual(stats["copied_from_reuse"], 1)
        self.assertEqual(stats["nearest_opponent_distance_filled"], 1)
        self.assertEqual(stats["ball_z_filled"], 1)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_nearest_opponent_distance_column("home_2")]), 20.0)
        self.assertAlmostEqual(float(rows.loc[5, PHYSICAL_XPASS_BALL_Z_COLUMN]), 0.0)

    def test_runtime_physical_xpass_cache_fills_pass_height_without_recomputing_xpass(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        pass_height_model = ConstantPassHeightModel(probability=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        PHYSICAL_XPASS_BALL_Z_COLUMN: 0.0,
                        **graph_nearest_opponent_distance_row_values(graph),
                        "home_2": 0.77,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    pass_height_model=pass_height_model,
                    pass_height_model_id="pass_height/new_model",
                    pass_height_device="cpu",
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_not_called()
        self.assertEqual(stats["cache_misses"], 0)
        self.assertEqual(stats["copied_from_reuse"], 1)
        self.assertEqual(stats["pass_height_filled"], 1)
        self.assertEqual(stats["pass_height_refreshed"], 0)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.77)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_1")]), 0.8, places=6)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_2")]), 0.8, places=6)

    def test_runtime_physical_xpass_cache_hits_with_current_pass_height_metadata(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        pass_height_model = ConstantPassHeightModel(probability=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                    | {"pass_height_model_id": "pass_height/current_model"}
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        PHYSICAL_XPASS_BALL_Z_COLUMN: 0.0,
                        **graph_nearest_opponent_distance_row_values(graph),
                        "home_2": 0.77,
                        physical_xpass_pass_height_column("home_1"): 0.3,
                        physical_xpass_pass_height_column("home_2"): 0.4,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                with patch("physical_pass_model._pass_height_predictions_for_graphs") as predict_pass_height:
                    stats = prewarm_physical_xpass_runtime_cache(
                        [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                        cache_dir=cache_dir,
                        source=PHYSICAL_XPASS_SOURCE,
                        teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                        num_workers=1,
                        pass_height_model=pass_height_model,
                        pass_height_model_id="pass_height/current_model",
                        pass_height_device="cpu",
                    )

        compute.assert_not_called()
        predict_pass_height.assert_not_called()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_written"], 0)
        self.assertEqual(stats["pass_height_filled"], 0)
        self.assertEqual(stats["pass_height_refreshed"], 0)

    def test_runtime_physical_xpass_cache_refreshes_only_stale_pass_height_columns(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        pass_height_model = ConstantPassHeightModel(probability=0.9)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                    | {"pass_height_model_id": "pass_height/old_model"}
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        PHYSICAL_XPASS_BALL_Z_COLUMN: 0.0,
                        **graph_nearest_opponent_distance_row_values(graph),
                        "home_2": 0.77,
                        physical_xpass_pass_height_column("home_1"): 0.1,
                        physical_xpass_pass_height_column("home_2"): 0.2,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    pass_height_model=pass_height_model,
                    pass_height_model_id="pass_height/new_model",
                    pass_height_device="cpu",
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_not_called()
        self.assertEqual(stats["cache_misses"], 0)
        self.assertEqual(stats["copied_from_reuse"], 1)
        self.assertEqual(stats["pass_height_filled"], 0)
        self.assertEqual(stats["pass_height_refreshed"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.77)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_1")]), 0.9, places=6)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_2")]), 0.9, places=6)

    def test_runtime_physical_xpass_cache_miss_rows_include_pass_height(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        pass_height_model = ConstantPassHeightModel(probability=0.7)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.61}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [graph], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    pass_height_model=pass_height_model,
                    pass_height_model_id="pass_height/new_model",
                    pass_height_device="cpu",
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_called_once()
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_written"], 1)
        self.assertEqual(stats["pass_height_filled"], 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.61)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_1")]), 0.7, places=6)
        self.assertAlmostEqual(float(rows.loc[5, physical_xpass_pass_height_column("home_2")]), 0.7, places=6)

    def test_runtime_physical_xpass_cache_writes_two_sportec_frame_scopes_for_same_action(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.41}, dtype=float), pd.Series({"home_2": 0.82}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [
                        {
                            "match_id": "runtime_match",
                            "graphs": [graph],
                            "labels": labels,
                            PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_ACTION,
                            PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN: [100],
                        },
                        {
                            "match_id": "runtime_match",
                            "graphs": [graph],
                            "labels": labels,
                            PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE,
                            PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN: [140],
                        },
                    ],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            action_rows = load_physical_xpass_match(cache_dir, "runtime_match", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_ACTION)
            receive_rows = load_physical_xpass_match(cache_dir, "runtime_match", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE)

        compute.assert_called_once()
        self.assertEqual(stats["cache_written"], 2)
        self.assertAlmostEqual(float(action_rows.loc[5, "home_2"]), 0.41)
        self.assertAlmostEqual(float(receive_rows.loc[5, "home_2"]), 0.82)
        self.assertEqual(int(action_rows.loc[5, PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN]), 100)
        self.assertEqual(int(receive_rows.loc[5, PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN]), 140)

    def test_runtime_physical_xpass_cache_preserves_legacy_action_row_and_adds_receive_scope(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        PHYSICAL_XPASS_BALL_Z_COLUMN: 0.0,
                        physical_xpass_nearest_opponent_distance_column("home_2"): 20.0,
                        "home_2": 0.33,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.88}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [
                        {
                            "match_id": "runtime_match",
                            "graphs": [graph],
                            "labels": labels,
                            PHYSICAL_XPASS_FRAME_SCOPE_COLUMN: PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE,
                            PHYSICAL_XPASS_STATE_FRAME_ID_COLUMN: [140],
                        }
                    ],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            action_rows = load_physical_xpass_match(cache_dir, "runtime_match", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_ACTION)
            receive_rows = load_physical_xpass_match(cache_dir, "runtime_match", frame_scope=PHYSICAL_XPASS_FRAME_SCOPE_RECEIVE)

        compute.assert_called_once()
        self.assertEqual(stats["cache_written"], 1)
        self.assertAlmostEqual(float(action_rows.loc[5, "home_2"]), 0.33)
        self.assertAlmostEqual(float(receive_rows.loc[5, "home_2"]), 0.88)

    def test_runtime_physical_xpass_cache_hash_mismatch_recomputes_and_replaces(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.41}, dtype=float)],
            ):
                attach_physical_xpass_cached_online_to_graphs(
                    [make_graph()],
                    labels,
                    cache_dir=cache_dir,
                    match_id="runtime_match",
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

            changed_graph = make_graph()
            changed_graph.x[1, config.NODE_FEATURE_X] += 1.0
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.81}, dtype=float)],
            ) as compute:
                attached, stats = attach_physical_xpass_cached_online_to_graphs(
                    [changed_graph],
                    labels,
                    cache_dir=cache_dir,
                    match_id="runtime_match",
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

            compute.assert_called_once()
            self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.81)
            self.assertEqual(stats["hash_mismatch_recomputed"], 1)
            rows = load_physical_xpass_match(cache_dir, "runtime_match")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.81)
            self.assertEqual(str(rows.loc[5, "physical_state_hash"]), physical_state_hash(changed_graph))

    def test_runtime_prewarm_items_use_pass_success_filtered_graph_hash(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        graph = make_graph()
        graph.x[0, config.NODE_FEATURE_VX] = 3.5
        model = SimpleNamespace(args=make_pass_success_args(poss_vel_aware=False))
        runtime_object = SimpleNamespace(match_id="runtime_match", graph_features_0=[graph], labels=labels)

        items = prepare_runtime_physical_xpass_prewarm_items([runtime_object], model)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["match_id"], "runtime_match")
        self.assertNotEqual(physical_state_hash(graph), physical_state_hash(items[0]["graphs"][0]))
        self.assertEqual(float(items[0]["graphs"][0].x[0, config.NODE_FEATURE_VX].item()), 0.0)

    def test_runtime_read_only_attach_validates_hash_and_does_not_compute(self) -> None:
        graph = make_graph()
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            )
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": physical_state_hash(graph),
                        "home_2": 0.64,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            attached = attach_physical_xpass_read_only_to_graphs(
                [graph],
                labels,
                cache_dir=cache_dir,
                match_id="runtime_match",
            )

        self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.64)

    def test_runtime_read_only_attach_ignores_hash_mismatch_for_inference_lookup(self) -> None:
        graph = make_graph()
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            (cache_dir / "matches").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "match_id": "runtime_match",
                        "action_index": 5,
                        "physical_state_hash": "stale",
                        "home_2": 0.64,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "runtime_match.parquet", index=False)

            attached = attach_physical_xpass_read_only_to_graphs(
                [graph],
                labels,
                cache_dir=cache_dir,
                match_id="runtime_match",
            )

        self.assertAlmostEqual(float(getattr(attached[0], PHYSICAL_XPASS_PROB_ATTR)[1]), 0.64)

    def test_runtime_physical_xpass_cache_rejects_incompatible_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps({"source": PHYSICAL_XPASS_LEGACY_SOURCE}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incompatible source"):
                attach_physical_xpass_cached_online_to_graphs(
                    [make_graph()],
                    torch.stack([make_label(action_index=5)]),
                    cache_dir=cache_dir,
                    match_id="runtime_match",
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                )

    def test_runtime_physical_xpass_prewarm_detects_hits_and_misses(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.37}, dtype=float)],
            ):
                first_stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [make_graph()], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    physical_batch_size=4,
                )
            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute_again:
                second_stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [make_graph()], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    physical_batch_size=4,
                )

        self.assertEqual(first_stats["cache_misses"], 1)
        self.assertEqual(first_stats["cache_written"], 1)
        self.assertEqual(first_stats["num_workers"], 1)
        self.assertEqual(first_stats["physical_batch_size"], 4)
        compute_again.assert_not_called()
        self.assertEqual(second_stats["cache_hits"], 1)
        self.assertEqual(second_stats["cache_misses"], 0)

    def test_runtime_physical_xpass_prewarm_writes_batch_chunks_incrementally_and_resumes(self) -> None:
        labels = torch.stack(
            [
                make_label(action_index=5),
                make_label(action_index=6),
                make_label(action_index=7),
            ]
        )
        graphs = [make_graph(), make_graph(), make_graph()]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            call_count = 0

            def fake_compute(chunk_graphs, **kwargs):
                nonlocal call_count
                del kwargs
                call_count += 1
                if call_count == 2:
                    partial = load_physical_xpass_match(cache_dir, "runtime_match")
                    self.assertEqual(partial.index.tolist(), [5, 6])
                return [pd.Series({"home_2": 0.3 + index}, dtype=float) for index, _graph in enumerate(chunk_graphs)]

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults", side_effect=fake_compute) as compute:
                first_stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": graphs, "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    physical_batch_size=2,
                )
            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute_again:
                second_stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": graphs, "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    physical_batch_size=2,
                )
            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        self.assertEqual(compute.call_count, 2)
        self.assertEqual(first_stats["cache_misses"], 3)
        self.assertEqual(first_stats["cache_written"], 3)
        self.assertEqual(first_stats["compute_chunks"], 2)
        self.assertEqual(rows.index.tolist(), [5, 6, 7])
        compute_again.assert_not_called()
        self.assertEqual(second_stats["cache_hits"], 3)
        self.assertEqual(second_stats["cache_misses"], 0)

    def test_runtime_physical_xpass_prewarm_dry_run_scans_without_compute_or_writes(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [make_graph()], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                    dry_run=True,
                )

            self.assertFalse((cache_dir / "matches" / "runtime_match.parquet").exists())
            self.assertFalse((cache_dir / "metadata.json").exists())

        compute.assert_not_called()
        self.assertTrue(stats["dry_run"])
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_written"], 0)

    def test_runtime_physical_xpass_prewarm_refresh_recomputes_cached_row_once(self) -> None:
        labels = torch.stack([make_label(action_index=5)])
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.25}, dtype=float)],
            ):
                prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [make_graph()], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    num_workers=1,
                )
            with patch(
                "physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults",
                return_value=[pd.Series({"home_2": 0.92}, dtype=float)],
            ) as compute:
                stats = prewarm_physical_xpass_runtime_cache(
                    [{"match_id": "runtime_match", "graphs": [make_graph()], "labels": labels}],
                    cache_dir=cache_dir,
                    source=PHYSICAL_XPASS_SOURCE,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    refresh=True,
                    num_workers=1,
                )

            rows = load_physical_xpass_match(cache_dir, "runtime_match")

        compute.assert_called_once()
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows.loc[5, "home_2"]), 0.92)

    def test_physical_xpass_cache_summary_reports_cache_hit(self) -> None:
        summary = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True,
            cache_disabled=False,
            refresh_requested=False,
            cache_dir="cache",
            prewarm_stats={"all": {"cache_hits": 3, "cache_misses": 0}},
        )

        self.assertEqual(summary["reason"], "cache_hit")
        self.assertTrue(summary["cache_reused"])
        self.assertTrue(summary["cache_fully_reused"])
        self.assertIn("reused 3/3", format_physical_xpass_cache_summary(summary))

    def test_physical_xpass_cache_summary_reports_hash_mismatch(self) -> None:
        summary = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True,
            cache_disabled=False,
            refresh_requested=False,
            cache_dir="cache",
            prewarm_stats={
                "m1": {
                    "cache_hits": 0,
                    "cache_misses": 2,
                    "cache_written": 2,
                    "hash_mismatch_recomputed": 2,
                }
            },
        )

        self.assertEqual(summary["reason"], "hash_mismatch")
        self.assertFalse(summary["cache_reused"])
        self.assertIn("reason=hash_mismatch", format_physical_xpass_cache_summary(summary))

    def test_physical_xpass_cache_summary_reports_cold_cache(self) -> None:
        summary = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True,
            cache_disabled=False,
            refresh_requested=False,
            cache_dir="cache",
            prewarm_stats={"m1": {"cache_misses": 2, "cache_written": 2}},
        )

        self.assertEqual(summary["reason"], "missing_or_cold_cache_rows")

    def test_physical_xpass_cache_summary_reason_precedence(self) -> None:
        refresh = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True,
            cache_disabled=False,
            refresh_requested=True,
            cache_dir="cache",
            prewarm_stats={"m1": {"cache_hits": 1, "cache_misses": 2, "hash_mismatch_recomputed": 2}},
        )
        disabled = summarize_physical_xpass_cache_usage(
            physical_xpass_required=True,
            cache_disabled=True,
            refresh_requested=True,
            cache_dir=None,
            runtime_stats={"pass_success": {"online_graphs": 2}},
        )
        not_required = summarize_physical_xpass_cache_usage(
            physical_xpass_required=False,
            cache_disabled=False,
            refresh_requested=False,
            cache_dir="cache",
            prewarm_stats={"m1": {"cache_hits": 1}},
        )

        self.assertEqual(refresh["reason"], "refresh_requested")
        self.assertEqual(disabled["reason"], "cache_disabled")
        self.assertEqual(not_required["reason"], "physical_xpass_not_required")

    def test_inference_missing_synthetic_sidecar_does_not_compute_online(self) -> None:
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
                "physical_xpass_floor": 0.2,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata", side_effect=FileNotFoundError("missing")):
            with patch.object(inference, "attach_physical_xpass_cached_online_to_graphs") as cached_attach:
                with self.assertRaises(FileNotFoundError):
                    inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        cached_attach.assert_not_called()

    def test_inference_only_physical_xpass_lookup_skips_metadata_validation(self) -> None:
        RuntimeState = type("BenchmarkState", (), {"__module__": "datatools.benchmark"})
        match = RuntimeState()
        match.match_id = "modification_50_game_state_2"
        graphs = [make_graph()]
        labels = torch.stack([make_label(action_index=0)])
        model = SimpleNamespace(
            args={
                "task": "pass_success",
                "model_id": "pass_success/baseline",
                "use_physical_xpass": False,
                "inference_use_physical_xpass": True,
                "model_variant": "gat_baseline",
                "physical_xpass_source": PHYSICAL_XPASS_LEGACY_SOURCE,
                "physical_cache_dir": "missing_cache",
                "physical_eps": 1e-4,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata") as validate:
            with patch.object(inference, "validate_x_pass_version_available") as validate_version:
                with patch.object(inference, "attach_physical_xpass_read_only_to_graphs", return_value=graphs) as read_only:
                    with patch.object(inference, "attach_physical_xpass_cached_online_to_graphs") as cached_attach:
                        result = inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        self.assertIs(result, graphs)
        validate.assert_not_called()
        validate_version.assert_called_once()
        read_only.assert_called_once()
        cached_attach.assert_not_called()
        self.assertEqual(read_only.call_args.kwargs["metric"], PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertIsNone(read_only.call_args.kwargs["missing_player_value"])
        self.assertEqual(getattr(model, "physical_xpass_lookup_policy"), "dataset_event_frame_player_only")
        self.assertEqual(runtime_physical_xpass_source(model.args), PHYSICAL_XPASS_SOURCE)
        self.assertEqual(runtime_physical_xpass_speed_aggregation(model.args), PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX)

    def test_runtime_physical_xpass_source_decouples_inference_blend_from_checkpoint_source(self) -> None:
        inference_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            physical_xpass_source=PHYSICAL_XPASS_LEGACY_SOURCE,
        )
        legacy_model_input_args = make_pass_success_args(
            use_physical_xpass=True,
            inference_use_physical_xpass=False,
            model_variant="gat_phys_logit_offset",
            physical_xpass_source=PHYSICAL_XPASS_LEGACY_SOURCE,
        )

        self.assertEqual(physical_xpass_source(inference_args), PHYSICAL_XPASS_LEGACY_SOURCE)
        self.assertEqual(runtime_physical_xpass_source(inference_args), PHYSICAL_XPASS_SOURCE)
        self.assertEqual(runtime_physical_xpass_source(legacy_model_input_args), PHYSICAL_XPASS_LEGACY_SOURCE)

    def test_inference_lookup_config_ignores_checkpoint_physical_metadata(self) -> None:
        args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            physical_xpass_source=PHYSICAL_XPASS_LEGACY_SOURCE,
            physical_xpass_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            x_pass_version="top10",
        )

        config = physical_xpass_inference_lookup_config(args, cache_dir="runtime_cache")

        self.assertTrue(config["use_physical_xpass"])
        self.assertEqual(config["physical_cache_dir"], "runtime_cache")
        self.assertEqual(config["source"], PHYSICAL_XPASS_SOURCE)
        self.assertEqual(config["speed_aggregation"], PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX)
        self.assertEqual(config["metric"], PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertEqual(config["x_pass_version"], "top10")
        self.assertEqual(config["metric_schema_version"], PHYSICAL_XPASS_METRIC_SCHEMA_VERSION)
        self.assertEqual(config["weight_version"], "v3")

    def test_inference_lookup_config_records_selected_metric_variants(self) -> None:
        default_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
        )
        max_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            x_pass_version="max",
        )
        topmean_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            x_pass_version="top25",
        )

        default_config = physical_xpass_inference_lookup_config(default_args, cache_dir="runtime_cache")
        self.assertEqual(default_config["metric"], PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertEqual(default_config["x_pass_version"], "top10")
        self.assertEqual(physical_xpass_inference_lookup_config(max_args, cache_dir="runtime_cache")["metric"], PHYSICAL_XPASS_METRIC_MAX)
        self.assertEqual(physical_xpass_inference_lookup_config(topmean_args, cache_dir="runtime_cache")["metric"], PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertEqual(physical_xpass_inference_lookup_config(topmean_args, cache_dir="runtime_cache")["x_pass_version"], "top25")

    def test_inference_lookup_config_selects_pc_xpass_source_and_metrics(self) -> None:
        default_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            pc_xpass=True,
        )
        top10_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            pc_xpass=True,
            x_pass_version="top50",
        )

        default_config = physical_xpass_inference_lookup_config(default_args, cache_dir="pc_cache")
        top10_config = physical_xpass_inference_lookup_config(top10_args, cache_dir="pc_cache")

        self.assertEqual(default_config["source"], PC_XPASS_SOURCE)
        self.assertEqual(default_config["metric"], PC_XPASS_DEFAULT_METRIC)
        self.assertEqual(default_config["available_metrics"], PC_XPASS_AVAILABLE_METRICS)
        self.assertEqual(default_config["max_speed"], PC_XPASS_DEFAULT_MAX_SPEED)
        self.assertEqual(default_config["speed_step"], PC_XPASS_DEFAULT_SPEED_STEP)
        self.assertEqual(top10_config["metric"], "top50_xpass")
        self.assertEqual(top10_config["x_pass_version"], "top50")

    def test_metadata_summary_maps_recorded_xpass_metrics(self) -> None:
        base = {"physical_xpass_requested": True}

        self.assertEqual(
            metadata_summary._summary_xpass_metric({**base, "physical_xpass_metric": PHYSICAL_XPASS_METRIC_NOISE_KERNEL}),
            "noise_kernel",
        )
        self.assertEqual(
            metadata_summary._summary_xpass_metric({**base, "physical_xpass_metric": PHYSICAL_XPASS_METRIC_MAX}),
            "max_xpass",
        )
        self.assertEqual(
            metadata_summary._summary_xpass_metric({**base, "physical_xpass_metric": PHYSICAL_XPASS_METRIC_TOPMEAN}),
            "topmean_xpass",
        )
        self.assertEqual(
            metadata_summary._summary_xpass_metric({**base, "physical_xpass_metric": PC_XPASS_METRIC_TOP10}),
            "top10_xpass",
        )
        self.assertEqual(
            metadata_summary._summary_xpass_metric({**base, "physical_xpass_metric": PC_XPASS_METRIC_TOP25}),
            "top25_xpass",
        )

    def test_inference_lookup_config_selects_xpass_weight_versions(self) -> None:
        args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            xpass_weight="v2",
        )

        self.assertEqual(physical_xpass_weight_version(args), "v2")
        self.assertEqual(physical_xpass_inference_lookup_config(args, cache_dir="runtime_cache")["weight_version"], "v2")
        self.assertEqual(physical_xpass_weight_version(make_pass_success_args()), "v3")

    def test_inference_lookup_config_parses_ball_z_limit(self) -> None:
        disabled_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            ball_z_limit="none",
        )
        enabled_args = make_pass_success_args(
            use_physical_xpass=False,
            inference_use_physical_xpass=True,
            model_variant="gat_baseline",
            ball_z_limit="1.25",
        )

        self.assertIsNone(physical_xpass_ball_z_limit(disabled_args))
        self.assertAlmostEqual(physical_xpass_ball_z_limit(enabled_args), 1.25)
        self.assertAlmostEqual(physical_xpass_inference_lookup_config(enabled_args, cache_dir="runtime_cache")["ball_z_limit"], 1.25)

    def test_runtime_physical_xpass_speed_aggregation_uses_runtime_defaults_for_blending(self) -> None:
        inference_args = make_pass_success_args(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset",
            physical_xpass_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            inference_use_physical_xpass=True,
        )
        legacy_model_input_args = make_pass_success_args(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset",
            physical_xpass_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
            inference_use_physical_xpass=False,
        )

        self.assertEqual(
            runtime_physical_xpass_speed_aggregation(inference_args),
            PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
        )
        self.assertEqual(
            runtime_physical_xpass_speed_aggregation(legacy_model_input_args),
            PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
        )

    def test_inference_runtime_read_only_uses_cache_without_online_compute(self) -> None:
        RuntimeState = type("BenchmarkState", (), {"__module__": "datatools.benchmark"})
        match = RuntimeState()
        match.match_id = "runtime_match"
        graphs = [make_graph()]
        labels = torch.stack([make_label(action_index=5)])
        model = SimpleNamespace(
            args={
                **make_pass_success_args(
                    physical_xpass_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                ),
                "physical_cache_dir": "cache",
                "physical_runtime_cache_read_only": True,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata") as validate:
            with patch.object(inference, "attach_physical_xpass_read_only_to_graphs", return_value=graphs) as read_only:
                with patch.object(inference, "attach_physical_xpass_cached_online_to_graphs") as cached_online:
                    result = inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        self.assertIs(result, graphs)
        validate.assert_called_once()
        read_only.assert_called_once()
        cached_online.assert_not_called()

    def test_baseline_pass_success_inference_blends_with_physical_xpass_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            match = make_physical_inference_match([0], [1])
            graph = match.graph_features_0[0]
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                        max_speed=30,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 0,
                        "physical_state_hash": "stale_after_model_feature_filtering",
                        "home_1": 0.2,
                        "home_2": 0.5,
                        "away_3": 0.1,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)
            model = DummyBaselineInferenceModel(cache_dir)

            with patch("physical_pass_model.compute_graphs_physical_xpass_metrics_as_defaults") as compute:
                probs, _ = inference.inference_gnn(match, model, device="cpu", post_action=False)

        compute.assert_not_called()
        self.assertTrue(inference_uses_physical_xpass(model.args))
        self.assertAlmostEqual(float(probs.loc[0, "home_2"]), 0.58, places=5)

    def test_inference_physical_xpass_missing_player_value_outputs_nan_not_model_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            match = make_physical_inference_match([0], [1])
            (cache_dir / "matches").mkdir(parents=True)
            (cache_dir / "metadata.json").write_text(
                json.dumps(
                    physical_xpass_as_default_metadata(
                        PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                        speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    )
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "match_id": "match_1",
                        "action_index": 0,
                        "physical_state_hash": "stale",
                        PHYSICAL_XPASS_PASS_DISTANCE_COLUMN: 20.0,
                        "home_1": 0.2,
                        "home_2": np.nan,
                        "away_3": np.nan,
                    }
                ]
            ).to_parquet(cache_dir / "matches" / "match_1.parquet", index=False)
            model = DummyBaselineInferenceModel(cache_dir)

            probs, _ = inference.inference_gnn(match, model, device="cpu", post_action=False)

        self.assertTrue(np.isnan(float(probs.loc[0, "home_2"])))
        self.assertAlmostEqual(float(probs.loc[0, "home_1"]), 0.2, places=5)

    def test_inference_missing_synthetic_sidecar_stays_read_only_when_cache_disabled(self) -> None:
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
                "physical_xpass_floor": 0.2,
                "physical_runtime_cache_disabled": True,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata", side_effect=FileNotFoundError("missing")):
            with patch.object(inference, "attach_physical_xpass_online_to_graphs", return_value=graphs) as online_attach:
                with self.assertRaises(FileNotFoundError):
                    inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        online_attach.assert_not_called()

    def test_inference_cached_sidecar_uses_physical_xpass_floor(self) -> None:
        match = SimpleNamespace(match_id="DFL-MAT-REAL", runtime_feature_root=Path("feature_run"))
        graphs = [make_graph()]
        labels = torch.stack([make_label(action_index=0)])
        model = SimpleNamespace(
            args={
                "task": "pass_success",
                "use_physical_xpass": True,
                "model_variant": "gat_phys_logit_offset",
                "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
                "physical_cache_dir": "cache_dir",
                "physical_eps": 1e-4,
                "physical_xpass_floor": 0.2,
            }
        )

        with patch.object(inference, "validate_physical_xpass_cache_metadata", return_value={}):
            with patch.object(inference, "attach_physical_xpass_read_only_to_graphs", return_value=graphs) as cached_attach:
                result = inference.attach_physical_xpass_for_inference(match, graphs, labels, model)

        self.assertIs(result, graphs)
        cached_attach.assert_called_once()
        self.assertEqual(cached_attach.call_args.kwargs["floor"], 0.2)

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

    def test_generate_physical_xpass_reuse_cache_validation_rejects_max_speed_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                max_speed=20,
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_speed"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    max_speed=18,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuse_cache_validation_rejects_sigma_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                sigma_angle=0.15,
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sigma_angle_factor"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    sigma_angle=PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuse_cache_validation_rejects_top_n_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                top_n=3,
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "top_n"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    top_n=PHYSICAL_XPASS_DEFAULT_TOP_N,
                    physical_eps=1e-4,
                )

    def test_generate_physical_xpass_reuse_cache_validation_rejects_available_metric_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                available_metrics=[PHYSICAL_XPASS_METRIC_MAX, PHYSICAL_XPASS_METRIC_TOPMEAN],
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "available_metrics"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    available_metrics=[
                        PHYSICAL_XPASS_METRIC_NOISE_KERNEL,
                        PHYSICAL_XPASS_METRIC_MAX,
                        PHYSICAL_XPASS_METRIC_TOPMEAN,
                    ],
                    physical_eps=1e-4,
                )

    def test_physical_xpass_metadata_records_max_speed_effective_grid(self) -> None:
        metadata = physical_xpass_as_default_metadata(
            PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
            speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            max_speed=20,
            sigma_angle=0.2,
            sigma_speed=0.1,
            sigma_distance=0.15,
        )
        expected_speeds = as_default_v0_values(max_speed=20)

        self.assertEqual(metadata["max_speed"], 20)
        self.assertEqual(metadata["n_v0"], len(expected_speeds))
        self.assertAlmostEqual(metadata["v0_max"], float(expected_speeds[-1]))
        self.assertLessEqual(metadata["v0_max"], 20.0)
        self.assertEqual(metadata["sigma_angle_factor"], 0.2)
        self.assertEqual(metadata["sigma_speed_factor"], 0.1)
        self.assertEqual(metadata["sigma_distance_factor"], 0.15)

    def test_generate_physical_xpass_reuse_cache_validation_rejects_old_full_grid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
            )
            metadata.pop("max_speed")
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_speed"):
                generate_physical_xpass.validate_reuse_cache_dir(
                    cache_dir,
                    teammate_policy=PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                    physical_eps=1e-4,
                )

    def test_runtime_sportec_implicit_incompatible_feature_cache_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_IGNORE,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                max_speed=20,
                default_metric="max_xpass",
                available_metrics=["max_xpass"],
                metric_schema_version=1,
            )
            metadata["metric"] = "max_player_cum_prob"
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = generate_physical_xpass.parse_args([])

            with patch("builtins.print") as print_mock:
                reuse_cache_dir, reason = generate_physical_xpass.resolve_runtime_sportec_reuse_cache(args, cache_dir)

        self.assertIsNone(reuse_cache_dir)
        self.assertIsNotNone(reason)
        self.assertIn("metric", str(reason))
        print_mock.assert_called_once()

    def test_runtime_sportec_explicit_incompatible_reuse_cache_still_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            (cache_dir / "metadata.json").write_text(
                json.dumps({"source": PHYSICAL_XPASS_LEGACY_SOURCE}),
                encoding="utf-8",
            )
            args = generate_physical_xpass.parse_args(["--reuse-cache-dir", str(cache_dir)])

            with self.assertRaisesRegex(ValueError, "incompatible source"):
                generate_physical_xpass.resolve_runtime_sportec_reuse_cache(args, cache_dir)

    def test_runtime_sportec_implicit_compatible_feature_cache_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            cache_dir.mkdir()
            metadata = physical_xpass_as_default_metadata(
                PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                max_speed=22,
                speed_step=1,
            )
            metadata["physical_eps"] = 1e-4
            (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            args = generate_physical_xpass.parse_args([])

            reuse_cache_dir, reason = generate_physical_xpass.resolve_runtime_sportec_reuse_cache(args, cache_dir)

        self.assertEqual(reuse_cache_dir, cache_dir)
        self.assertIsNone(reason)

    def test_runtime_sportec_metadata_records_implicit_reuse_skip_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_root = root / "feature"
            graph_dir = feature_root / "action_graphs"
            label_dir = feature_root / "labels"
            cache_dir = root / "runtime" / "sportec"
            feature_cache_dir = feature_root / "physical_xpass"
            feature_cache_dir.mkdir(parents=True)
            (feature_cache_dir / "metadata.json").write_text(
                json.dumps({"source": PHYSICAL_XPASS_LEGACY_SOURCE}),
                encoding="utf-8",
            )
            args = generate_physical_xpass.parse_args(["--match-id", "m1"])

            with patch.object(generate_physical_xpass, "resolve_feature_run_id", return_value="feature_run"):
                with patch.object(generate_physical_xpass, "resolve_feature_root", return_value=feature_root):
                    with patch.object(generate_physical_xpass, "get_action_graph_dir", return_value=graph_dir):
                        with patch.object(
                            generate_physical_xpass,
                            "resolve_reference_label_context",
                            return_value=(label_dir, "disc_0.7", "model"),
                        ):
                            with patch.object(generate_physical_xpass, "get_runtime_physical_xpass_dir", return_value=cache_dir):
                                with patch.object(generate_physical_xpass, "get_physical_xpass_dir", return_value=feature_cache_dir):
                                    with patch.object(generate_physical_xpass, "resolve_match_ids", return_value=[]):
                                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata") as metadata_mock:
                                            with patch("builtins.print"):
                                                generate_physical_xpass.run_runtime_sportec(args)

        metadata_mock.assert_called_once()
        source_inputs = metadata_mock.call_args.kwargs["source_inputs"]
        self.assertEqual(source_inputs["implicit_reuse_cache_dir"], str(feature_cache_dir))
        self.assertIsNone(source_inputs["reuse_cache_dir"])
        self.assertIn("incompatible source", source_inputs["implicit_reuse_cache_skipped_reason"])

    def test_latest_run_loader_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "latest.json"
            latest_path.write_text(json.dumps({"run_id": "feature_bom"}), encoding="utf-8-sig")

            with patch.object(project_config, "FEATURE_LATEST_PATH", latest_path):
                self.assertEqual(project_config.load_latest_run_id("feature"), "feature_bom")

    def test_run_metadata_loader_accepts_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "run"
            run_root.mkdir()
            (run_root / "metadata.json").write_text(json.dumps({"return_type": "disc_0.7"}), encoding="utf-8-sig")

            self.assertEqual(project_config.load_run_metadata(run_root), {"return_type": "disc_0.7"})

    def test_feature_return_type_inference_accepts_utf8_bom_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_id = "feature_bom"
            feature_root = Path(tmpdir) / feature_id
            feature_root.mkdir()
            (feature_root / "metadata.json").write_text(
                json.dumps({"return_types": ["disc_0.7"]}),
                encoding="utf-8-sig",
            )

            with patch.object(project_config, "get_feature_run_root", return_value=feature_root):
                self.assertEqual(project_config.infer_feature_run_return_types(feature_id), ["disc_0.7"])

    def test_runtime_sportec_pc_xpass_uses_latest_feature_input_and_pc_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_root = root / "feature"
            graph_dir = feature_root / "action_graphs"
            label_dir = feature_root / "labels"
            pc_cache_dir = root / "pc_xpass" / "sportec"
            feature_cache_dir = feature_root / "physical_xpass"
            args = generate_physical_xpass.parse_args(["--pc-xpass"])

            with patch.object(generate_physical_xpass, "resolve_feature_run_id", return_value="feature_run") as resolve_mock:
                with patch.object(generate_physical_xpass, "resolve_feature_root", return_value=feature_root):
                    with patch.object(generate_physical_xpass, "get_action_graph_dir", return_value=graph_dir):
                        with patch.object(
                            generate_physical_xpass,
                            "resolve_reference_label_context",
                            return_value=(label_dir, "disc_0.7", "model"),
                        ):
                            with patch.object(generate_physical_xpass, "get_pc_xpass_dir", return_value=pc_cache_dir):
                                with patch.object(generate_physical_xpass, "get_physical_xpass_dir", return_value=feature_cache_dir):
                                    with patch.object(generate_physical_xpass, "resolve_match_ids", return_value=[]):
                                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata") as metadata_mock:
                                            with patch("builtins.print"):
                                                generate_physical_xpass.run_runtime_sportec(args)

        resolve_mock.assert_called_once_with(None, required=True, allow_latest=True)
        metadata_mock.assert_called_once()
        self.assertEqual(metadata_mock.call_args.args[1], pc_cache_dir)

    def test_runtime_physical_xpass_dir_points_directly_to_dataset_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            self.assertEqual(
                project_config.get_runtime_physical_xpass_dir("sportec", root=root),
                root / "sportec",
            )
            self.assertEqual(
                project_config.get_runtime_physical_xpass_dir("benchmark", root=root),
                root / "benchmark",
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
        self.assertTrue(Decoder(decoder_args()).physical_beta0.requires_grad)
        self.assertTrue(Decoder(decoder_args()).physical_beta1.requires_grad)

    def test_decoder_fixed_beta_and_residual_clipping(self) -> None:
        decoder = Decoder(decoder_args(freeze_beta1=True, residual_clip_value=1.0))
        decoder.nodewise_mlp = FixedDelta([10.0])

        out = decoder(
            node_embeddings=torch.zeros((1, 4), dtype=torch.float32),
            physical_xpass_logit=torch.tensor([0.0]),
        )

        self.assertFalse(decoder.physical_beta1.requires_grad)
        self.assertTrue(decoder.physical_beta0.requires_grad)
        self.assertAlmostEqual(float(out[0]), float(torch.tanh(torch.tensor(10.0))), places=6)
        self.assertAlmostEqual(float(decoder.latest_delta_gat[0]), float(torch.tanh(torch.tensor(10.0))), places=6)

    def test_decoder_can_freeze_beta0_and_beta1_independently(self) -> None:
        default_decoder = Decoder(decoder_args())
        freeze_beta0_decoder = Decoder(decoder_args(freeze_beta0=True))
        freeze_beta1_decoder = Decoder(decoder_args(freeze_beta1=True))
        freeze_both_decoder = Decoder(decoder_args(freeze_beta0=True, freeze_beta1=True))
        legacy_freeze_beta1_decoder = Decoder(decoder_args(learn_physical_scale=False, freeze_beta1=None))

        self.assertTrue(default_decoder.physical_beta0.requires_grad)
        self.assertTrue(default_decoder.physical_beta1.requires_grad)
        self.assertFalse(freeze_beta0_decoder.physical_beta0.requires_grad)
        self.assertTrue(freeze_beta0_decoder.physical_beta1.requires_grad)
        self.assertTrue(freeze_beta1_decoder.physical_beta0.requires_grad)
        self.assertFalse(freeze_beta1_decoder.physical_beta1.requires_grad)
        self.assertFalse(freeze_both_decoder.physical_beta0.requires_grad)
        self.assertFalse(freeze_both_decoder.physical_beta1.requires_grad)
        self.assertFalse(legacy_freeze_beta1_decoder.physical_beta1.requires_grad)

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

    def test_run_epoch_node_selection_accepts_singleton_column_logits(self) -> None:
        graph = make_graph(["home_1", "home_2", "away_3"])
        label = make_label(intent_index=0)
        loader = DataLoader([(graph, label, torch.tensor(1.0, dtype=torch.float32))], batch_size=1)
        args = SimpleNamespace(
            gnn_task="node_selection",
            task="success_intent",
            include_out=False,
            lambda_l1=0.0,
            use_xg=False,
            use_xt=False,
            use_goal_distance=False,
            use_epv=False,
            print_freq=99,
            clip=10,
        )

        metrics = run_epoch(
            args,
            DummyColumnNodeSelectionModel([2.0, 1.0, -1.0]),
            loader,
            device="cpu",
            train=False,
        )

        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mrr"], 1.0, places=6)

    def test_wrapper_physical_flags_reach_only_pass_success(self) -> None:
        args = SimpleNamespace(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset_regularized",
            physical_cache_dir="cache_dir",
            physical_eps=1e-3,
            physical_xpass_floor=0.2,
            freeze_beta0=True,
            freeze_beta1=True,
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
        self.assertIn("--freeze-beta0", pass_command)
        self.assertIn("--freeze-beta1", pass_command)
        self.assertIn("--physical-xpass-floor", pass_command)
        self.assertIn("--residual-regularization-lambda", pass_command)
        self.assertIn("--residual-distance-threshold", pass_command)
        self.assertIn("--short-residual-regularization-lambda", pass_command)
        self.assertIn("--long-residual-regularization-lambda", pass_command)
        self.assertIn("--short-residual-clip-value", pass_command)
        self.assertIn("--long-residual-clip-value", pass_command)
        self.assertNotIn("--use_physical_xpass", outcome_command)
        self.assertNotIn("--physical-cache-dir", outcome_command)
        self.assertNotIn("--physical-xpass-floor", outcome_command)
        self.assertNotIn("--freeze-beta0", outcome_command)
        self.assertNotIn("--freeze-beta1", outcome_command)
        self.assertNotIn("--short-residual-clip-value", outcome_command)

    def test_wrapper_legacy_fixed_physical_scale_maps_to_freeze_beta1(self) -> None:
        args = SimpleNamespace(
            use_physical_xpass=True,
            model_variant="gat_phys_logit_offset_regularized",
            physical_cache_dir=None,
            physical_eps=1e-4,
            physical_xpass_floor=None,
            learn_physical_scale=False,
            residual_regularization_lambda=0.0,
            residual_clip_value=None,
            residual_distance_threshold=30.0,
            short_residual_regularization_lambda=None,
            long_residual_regularization_lambda=None,
            short_residual_clip_value=None,
            long_residual_clip_value=None,
        )
        command = train_wrapper.append_physical_xpass_flags([], args)

        self.assertIn("--freeze-beta1", command)
        self.assertNotIn("--fixed-physical-scale", command)

    def test_wrapper_physical_xpass_floor_validation(self) -> None:
        base_args = [
            "--feature-run-id",
            "feature_run",
            "--target-family",
            "goal_distance",
            "--return_type",
            "disc_0.9",
            "--intended-receiver-mode",
            "angle_only",
        ]

        with patch.object(train_wrapper, "resolve_feature_run_id", return_value="feature_run"):
            with patch.object(train_wrapper, "infer_feature_run_intended_receiver_modes", return_value=["angle_only"]):
                with patch.object(train_wrapper, "infer_feature_run_return_types", return_value=["disc_0.9"]):
                    self.assertIsNone(train_wrapper.parse_args(base_args).physical_xpass_floor)
                    self.assertEqual(
                        train_wrapper.parse_args([*base_args, "--physical-xpass-floor", "0.0"]).physical_xpass_floor,
                        0.0,
                    )
                    self.assertEqual(
                        train_wrapper.parse_args([*base_args, "--physical_xpass_floor", "0.2"]).physical_xpass_floor,
                        0.2,
                    )
        with self.assertRaises(SystemExit):
            train_wrapper.parse_args([*base_args, "--physical-xpass-floor", "-0.1"])
        with self.assertRaises(SystemExit):
            train_wrapper.parse_args([*base_args, "--physical-xpass-floor", "1.0"])

    def test_visualization_sidecar_loader_reads_max_player_cum_prob_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "physical_xpass"
            match_dir = cache_dir / "matches"
            match_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "match_id": "m1",
                        "action_index": 3,
                        "physical_state_hash": "abc",
                        "home_1": 0.2,
                        "home_2": np.nan,
                        physical_xpass_pass_height_column("home_1"): 0.9,
                    }
                ]
            ).to_parquet(match_dir / "m1.parquet", index=False)

            row = load_physical_xpass_component(cache_dir, "m1", 3)

        self.assertEqual(row.name, "noise_kernel_xpass")
        self.assertAlmostEqual(float(row["home_1"]), 0.2)
        self.assertTrue(np.isnan(row["home_2"]))
        self.assertNotIn(physical_xpass_pass_height_column("home_1"), row.index)
        self.assertNotIn("physical_state_hash", row.index)

    def test_generate_physical_xpass_cli_defaults_to_consider_teammates(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run"])
        self.assertTrue(args.consider_teammates)
        self.assertEqual(args.speed_aggregation, PHYSICAL_XPASS_DEFAULT_SPEED_AGGREGATION)
        self.assertEqual(args.max_speed, 22.0)
        self.assertEqual(args.speed_step, 1.0)
        self.assertEqual(args.coarse_n_angles, 36)
        self.assertEqual(args.refine_top_k_angles, 2)
        self.assertEqual(args.refine_angle_radius, 10.0)
        self.assertEqual(args.angle_step, 2.5)
        self.assertEqual(args.sigma_angle, PHYSICAL_XPASS_DEFAULT_SIGMA_ANGLE_FACTOR)
        self.assertEqual(args.sigma_speed, PHYSICAL_XPASS_DEFAULT_SIGMA_SPEED_FACTOR)
        self.assertEqual(args.sigma_distance, PHYSICAL_XPASS_DEFAULT_SIGMA_DISTANCE_FACTOR)

    def test_generate_physical_xpass_cli_default_runtime_selects_all_datasets(self) -> None:
        args = generate_physical_xpass.parse_args([])

        self.assertIsNone(args.feature_run_id)
        self.assertEqual(
            generate_physical_xpass.selected_runtime_datasets(args),
            ["sportec", "skillcorner", "benchmark", "hawkeye"],
        )
        self.assertEqual(args.num_workers, "auto")
        self.assertEqual(args.max_auto_workers, PHYSICAL_DEFAULT_MAX_AUTO_WORKERS)
        self.assertEqual(args.sportec_runtime_match_window, 4)
        self.assertIsNone(args.skillcorner_runtime_row_window)
        self.assertIsNone(args.benchmark_runtime_row_window)
        self.assertEqual(args.physical_batch_size, 16)
        self.assertEqual(args.worker_thread_limit, 1)
        self.assertFalse(args.dry_run)

    def test_generate_physical_xpass_cli_accepts_runtime_row_windows(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--skillcorner-runtime-row-window",
                "32",
                "--benchmark-runtime-row-window",
                "12",
            ]
        )

        self.assertEqual(args.skillcorner_runtime_row_window, 32)
        self.assertEqual(args.benchmark_runtime_row_window, 12)

    def test_generate_physical_xpass_cli_rejects_non_positive_runtime_row_windows(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--skillcorner-runtime-row-window", "0"])
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--benchmark-runtime-row-window", "0"])

    def test_generate_physical_xpass_cli_accepts_dry_run(self) -> None:
        args = generate_physical_xpass.parse_args(["--dry-run", "--no-skillcorner"])

        self.assertTrue(args.dry_run)
        self.assertEqual(generate_physical_xpass.selected_runtime_datasets(args), ["sportec", "benchmark", "hawkeye"])

    def test_generate_physical_xpass_cli_accepts_pass_height_model_id(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--pass-height-model-id",
                "pass_height/pass_height_20260629T002934_085928_bc1e56c1",
                "--pass-height-device",
                "cpu",
            ]
        )

        self.assertEqual(args.pass_height_model_id, "pass_height/pass_height_20260629T002934_085928_bc1e56c1")
        self.assertEqual(args.pass_height_device, "cpu")

    def test_generate_physical_xpass_cli_rejects_non_pass_height_model_id(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--pass-height-model-id", "pass_success/run"])

    def test_generate_physical_xpass_cli_rejects_pass_height_with_legacy_feature_run(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(
                [
                    "--feature-run-id",
                    "feature_run",
                    "--pass-height-model-id",
                    "pass_height/pass_height_20260629T002934_085928_bc1e56c1",
                ]
            )

    def test_generate_physical_xpass_runtime_status_line_includes_core_counts(self) -> None:
        line = generate_physical_xpass._format_runtime_stats_line(
            "hawkeye situation s1",
            {
                "rows_scanned": 12,
                "pass_rows": 3,
                "cache_hits": 1,
                "cache_misses": 2,
                "cache_written": 2,
                "copied_from_reuse": 0,
                "pass_distance_filled": 1,
                "pass_height_filled": 2,
                "pass_height_refreshed": 1,
                "skipped_all_nan": 0,
            },
        )

        self.assertIn("hawkeye situation s1", line)
        self.assertIn("rows=12", line)
        self.assertIn("passes=3", line)
        self.assertIn("hits=1", line)
        self.assertIn("misses=2", line)
        self.assertIn("written=2", line)
        self.assertIn("pass_distance_filled=1", line)
        self.assertIn("pass_height_filled=2", line)
        self.assertIn("pass_height_refreshed=1", line)
        self.assertIn("skipped_all_nan=0", line)

    def test_generate_physical_xpass_skip_reason_summary_groups_common_errors(self) -> None:
        skipped = {
            "hawkeye": {
                "s1": "ValueError: Physical xPass sidecars use incompatible speed_aggregation",
                "s2": "FileNotFoundError: missing input",
            },
            "skillcorner": {"processing": {"m1:7": "RuntimeError: no pass rows"}},
        }

        summary = generate_physical_xpass.summarize_skip_reasons(skipped)

        self.assertIn("empty_or_no_pass_rows=1", summary)
        self.assertIn("incompatible_cache=1", summary)
        self.assertIn("missing_input=1", summary)

    def test_generate_physical_xpass_prewarm_runtime_items_passes_pass_height_context(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--pass-height-model-id",
                "pass_height/pass_height_20260629T002934_085928_bc1e56c1",
                "--pass-height-device",
                "cpu",
            ]
        )
        model = ConstantPassHeightModel()
        args._pass_height_model = model
        items = [{"match_id": "m1", "graphs": [make_graph()], "labels": torch.stack([make_label(action_index=5)])}]

        with patch.object(
            generate_physical_xpass,
            "prewarm_physical_xpass_runtime_cache",
            return_value={"cache_misses": 1},
        ) as prewarm:
            result = generate_physical_xpass.prewarm_runtime_items(
                items,
                cache_dir=Path("cache"),
                args=args,
            )

        self.assertEqual(result, {"cache_misses": 1})
        self.assertIs(prewarm.call_args.kwargs["pass_height_model"], model)
        self.assertEqual(prewarm.call_args.kwargs["pass_height_model_id"], "pass_height/pass_height_20260629T002934_085928_bc1e56c1")
        self.assertEqual(prewarm.call_args.kwargs["pass_height_device"], "cpu")

    def test_generate_physical_xpass_runtime_metadata_includes_pass_height_model(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--pass-height-model-id",
                "pass_height/pass_height_20260629T002934_085928_bc1e56c1",
                "--pass-height-device",
                "cpu",
            ]
        )
        args._pass_height_model_record = {"model_id": args.pass_height_model_id, "task": "pass_height"}

        with patch.object(generate_physical_xpass, "write_run_metadata") as write_metadata:
            generate_physical_xpass.write_runtime_dataset_metadata(
                "benchmark",
                Path("cache"),
                args,
                stats=generate_physical_xpass.empty_runtime_stats(Path("cache")),
                source_inputs={},
                skipped={},
            )

        metadata = write_metadata.call_args.args[1]
        self.assertEqual(metadata["pass_height_model_id"], "pass_height/pass_height_20260629T002934_085928_bc1e56c1")
        self.assertEqual(metadata["pass_height_model_record"], {"model_id": args.pass_height_model_id, "task": "pass_height"})
        self.assertEqual(metadata["pass_height_column_suffix"], "__pass_height")
        self.assertEqual(metadata["pass_height_storage"], "per_player_probability_columns")

    def test_generate_physical_xpass_skillcorner_batches_runtime_prewarm_by_rows(self) -> None:
        args = generate_physical_xpass.parse_args(["--skillcorner-runtime-row-window", "3"])
        context = {"events": pd.DataFrame({"index": [1, 2, 3]})}

        def fake_possession(_context, event_index: int, **_kwargs):
            return (
                SimpleNamespace(
                    match_id="m1",
                    graph_features_0=[object()],
                    labels=torch.stack([make_label(action_index=int(event_index))]),
                ),
                {},
            )

        with patch.object(generate_physical_xpass, "get_runtime_physical_xpass_dir", return_value=Path("cache")):
            with patch.object(generate_physical_xpass, "discover_skillcorner_matches", return_value=(["m1"], {})):
                with patch.object(generate_physical_xpass, "build_skillcorner_match_context", return_value=context):
                    with patch.object(generate_physical_xpass, "build_skillcorner_possession", side_effect=fake_possession):
                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata"):
                            with patch.object(generate_physical_xpass.tqdm, "write"):
                                with patch.object(
                                    generate_physical_xpass,
                                    "prewarm_runtime_items",
                                    side_effect=lambda items, **_kwargs: make_runtime_prewarm_stats(len(items)),
                                ) as prewarm:
                                    result = generate_physical_xpass.run_runtime_skillcorner(args)

        prewarm.assert_called_once()
        self.assertEqual(len(prewarm.call_args.args[0]), 3)
        self.assertEqual(result["stats"]["cache_misses"], 3)
        self.assertEqual(result["skipped"], {})

    def test_generate_physical_xpass_skillcorner_batch_failure_retries_per_event(self) -> None:
        args = generate_physical_xpass.parse_args(["--skillcorner-runtime-row-window", "2"])
        context = {"events": pd.DataFrame({"index": [1, 2]})}
        call_sizes: list[int] = []

        def fake_possession(_context, event_index: int, **_kwargs):
            return (
                SimpleNamespace(
                    match_id="m1",
                    graph_features_0=[object()],
                    labels=torch.stack([make_label(action_index=int(event_index))]),
                ),
                {},
            )

        def fake_prewarm(items, **_kwargs):
            call_sizes.append(len(items))
            if len(call_sizes) == 1:
                raise RuntimeError("batch failed")
            if len(call_sizes) == 3:
                raise ValueError("event failed")
            return make_runtime_prewarm_stats(len(items))

        with patch.object(generate_physical_xpass, "get_runtime_physical_xpass_dir", return_value=Path("cache")):
            with patch.object(generate_physical_xpass, "discover_skillcorner_matches", return_value=(["m1"], {})):
                with patch.object(generate_physical_xpass, "build_skillcorner_match_context", return_value=context):
                    with patch.object(generate_physical_xpass, "build_skillcorner_possession", side_effect=fake_possession):
                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata"):
                            with patch.object(generate_physical_xpass.tqdm, "write"):
                                with patch.object(
                                    generate_physical_xpass,
                                    "prewarm_runtime_items",
                                    side_effect=fake_prewarm,
                                ):
                                    result = generate_physical_xpass.run_runtime_skillcorner(args)

        self.assertEqual(call_sizes, [2, 1, 1])
        self.assertIn("m1:2", result["skipped"])
        self.assertIn("ValueError", result["skipped"]["m1:2"])
        self.assertEqual(result["stats"]["cache_misses"], 1)

    def test_generate_physical_xpass_benchmark_batches_runtime_prewarm_by_rows(self) -> None:
        args = generate_physical_xpass.parse_args(["--benchmark-runtime-row-window", "4"])
        call_sizes: list[int] = []

        def fake_build_state(_raw_state, modification_id: int, game_state_id: int, _higher_state_id: int):
            return (
                SimpleNamespace(
                    match_id=f"modification_{int(modification_id)}_game_state_{int(game_state_id)}",
                    graph_features_0=[object()],
                    labels=torch.stack([make_label(action_index=0)]),
                ),
                pd.DataFrame(),
                {},
            )

        def fake_prewarm(items, **_kwargs):
            call_sizes.append(len(items))
            return make_runtime_prewarm_stats(len(items))

        with patch.object(generate_physical_xpass, "get_runtime_physical_xpass_dir", return_value=Path("cache")):
            with patch.object(generate_physical_xpass, "discover_benchmark_modifications", return_value=([1, 2, 3], {})):
                with patch.object(
                    generate_physical_xpass,
                    "load_benchmark_modification_data",
                    return_value={"game_state_1": pd.DataFrame(), "game_state_2": pd.DataFrame(), "higher_state_id": 9},
                ):
                    with patch.object(generate_physical_xpass, "build_benchmark_state", side_effect=fake_build_state):
                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata"):
                            with patch.object(generate_physical_xpass.tqdm, "write"):
                                with patch.object(
                                    generate_physical_xpass,
                                    "prewarm_runtime_items",
                                    side_effect=fake_prewarm,
                                ):
                                    result = generate_physical_xpass.run_runtime_benchmark(args)

        self.assertEqual(call_sizes, [4, 2])
        self.assertEqual(result["stats"]["cache_misses"], 6)
        self.assertEqual(result["skipped"], {})

    def test_generate_physical_xpass_benchmark_batch_failure_retries_per_modification(self) -> None:
        args = generate_physical_xpass.parse_args(["--benchmark-runtime-row-window", "4"])
        call_sizes: list[int] = []

        def fake_build_state(_raw_state, modification_id: int, game_state_id: int, _higher_state_id: int):
            return (
                SimpleNamespace(
                    match_id=f"modification_{int(modification_id)}_game_state_{int(game_state_id)}",
                    graph_features_0=[object()],
                    labels=torch.stack([make_label(action_index=0)]),
                ),
                pd.DataFrame(),
                {},
            )

        def fake_prewarm(items, **_kwargs):
            call_sizes.append(len(items))
            if len(call_sizes) == 1:
                raise RuntimeError("batch failed")
            if len(call_sizes) == 3:
                raise ValueError("modification failed")
            return make_runtime_prewarm_stats(len(items))

        with patch.object(generate_physical_xpass, "get_runtime_physical_xpass_dir", return_value=Path("cache")):
            with patch.object(generate_physical_xpass, "discover_benchmark_modifications", return_value=([1, 2], {})):
                with patch.object(
                    generate_physical_xpass,
                    "load_benchmark_modification_data",
                    return_value={"game_state_1": pd.DataFrame(), "game_state_2": pd.DataFrame(), "higher_state_id": 9},
                ):
                    with patch.object(generate_physical_xpass, "build_benchmark_state", side_effect=fake_build_state):
                        with patch.object(generate_physical_xpass, "write_runtime_dataset_metadata"):
                            with patch.object(generate_physical_xpass.tqdm, "write"):
                                with patch.object(
                                    generate_physical_xpass,
                                    "prewarm_runtime_items",
                                    side_effect=fake_prewarm,
                                ):
                                    result = generate_physical_xpass.run_runtime_benchmark(args)

        self.assertEqual(call_sizes, [4, 2, 2])
        self.assertIn("2", result["skipped"])
        self.assertIn("ValueError", result["skipped"]["2"])
        self.assertEqual(result["stats"]["cache_misses"], 2)

    def test_generate_physical_xpass_cli_runtime_dataset_opt_out_flags(self) -> None:
        args = generate_physical_xpass.parse_args(["--no-skillcorner", "--no-hawkeye"])

        self.assertEqual(generate_physical_xpass.selected_runtime_datasets(args), ["sportec", "benchmark"])

    def test_generate_physical_xpass_cli_feature_run_uses_legacy_mode(self) -> None:
        args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run"])

        self.assertEqual(args.feature_run_id, "feature_run")
        self.assertFalse(args.runtime_sportec_cache)

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

    def test_generate_physical_xpass_cli_accepts_max_speed_aliases(self) -> None:
        hyphen_args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--max-speed", "20"])
        underscore_args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--max_speed", "20"])

        self.assertEqual(hyphen_args.max_speed, 20)
        self.assertEqual(underscore_args.max_speed, 20)

    def test_generate_physical_xpass_cli_accepts_sigma_factors(self) -> None:
        args = generate_physical_xpass.parse_args(
            [
                "--feature-run-id",
                "feature_run",
                "--sigma-angle",
                "0.2",
                "--sigma-speed",
                "0.1",
                "--sigma-distance",
                "0.15",
            ]
        )

        self.assertEqual(args.sigma_angle, 0.2)
        self.assertEqual(args.sigma_speed, 0.1)
        self.assertEqual(args.sigma_distance, 0.15)

    def test_generate_physical_xpass_cli_accepts_top_n(self) -> None:
        default_args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run"])
        custom_args = generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--top-n", "3"])

        self.assertEqual(default_args.top_n, PHYSICAL_XPASS_DEFAULT_TOP_N)
        self.assertEqual(custom_args.top_n, 3)

    def test_generate_physical_xpass_cli_accepts_metric_output_skip_flags(self) -> None:
        args = generate_physical_xpass.parse_args(["--no-noise-kernel", "--no-max"])

        self.assertFalse(args.export_noise_kernel)
        self.assertFalse(args.export_max)
        self.assertTrue(args.export_topmean)
        self.assertEqual(generate_physical_xpass.enabled_physical_xpass_metrics_from_args(args), [PHYSICAL_XPASS_METRIC_TOPMEAN])

    def test_generate_physical_xpass_cli_pc_xpass_uses_pc_defaults(self) -> None:
        args = generate_physical_xpass.parse_args(["--pc-xpass"])

        self.assertTrue(args.pc_xpass)
        self.assertEqual(args.max_speed, PC_XPASS_DEFAULT_MAX_SPEED)
        self.assertEqual(args.speed_step, PC_XPASS_DEFAULT_SPEED_STEP)
        self.assertEqual(args.control_function_power, PC_XPASS_DEFAULT_CONTROL_FUNCTION_POWER)
        self.assertEqual(args.control_function_inflection_point, PC_XPASS_DEFAULT_CONTROL_FUNCTION_INFLECTION_POINT)
        self.assertEqual(args.control_function_gamma, PC_XPASS_DEFAULT_CONTROL_FUNCTION_GAMMA)
        self.assertFalse(generate_physical_xpass.pc_ignore_teammates_lane_survival_from_args(args))
        self.assertFalse(generate_physical_xpass.pc_ignore_teammates_control_from_args(args))
        self.assertEqual(generate_physical_xpass.enabled_physical_xpass_metrics_from_args(args), [PHYSICAL_XPASS_METRIC_MAX, PC_XPASS_METRIC_TOP10])

        top25_args = generate_physical_xpass.parse_args(["--pc-xpass", "--top-n", "25"])
        self.assertEqual(generate_physical_xpass.enabled_physical_xpass_metrics_from_args(top25_args), [PHYSICAL_XPASS_METRIC_MAX, PC_XPASS_METRIC_TOP25])

        multi_top_args = generate_physical_xpass.parse_args(["--pc-xpass", "--top-n-values", "5", "10", "25"])
        self.assertEqual(generate_physical_xpass.pc_top_n_values_from_args(multi_top_args), [5, 10, 25])
        self.assertEqual(
            generate_physical_xpass.enabled_physical_xpass_metrics_from_args(multi_top_args),
            [PHYSICAL_XPASS_METRIC_MAX, "top5_xpass", PC_XPASS_METRIC_TOP10, PC_XPASS_METRIC_TOP25],
        )

        custom_args = generate_physical_xpass.parse_args(
            [
                "--pc-xpass",
                "--control-function-power",
                "15",
                "--control-function-inflection-point",
                "0.3",
                "--control-function-gamma",
                "2",
            ]
        )
        self.assertEqual(custom_args.control_function_power, 15)
        self.assertEqual(custom_args.control_function_inflection_point, 0.3)
        self.assertEqual(custom_args.control_function_gamma, 2)

    def test_generate_physical_xpass_cli_pc_xpass_accepts_split_teammate_ignoring(self) -> None:
        lane_args = generate_physical_xpass.parse_args(["--pc-xpass", "--ignore-teammates-lane-survival"])
        control_args = generate_physical_xpass.parse_args(["--pc-xpass", "--ignore-teammates-control"])
        both_args = generate_physical_xpass.parse_args(["--pc-xpass", "--ignore-teammates"])

        self.assertTrue(generate_physical_xpass.pc_ignore_teammates_lane_survival_from_args(lane_args))
        self.assertFalse(generate_physical_xpass.pc_ignore_teammates_control_from_args(lane_args))
        self.assertFalse(generate_physical_xpass.pc_ignore_teammates_lane_survival_from_args(control_args))
        self.assertTrue(generate_physical_xpass.pc_ignore_teammates_control_from_args(control_args))
        self.assertTrue(generate_physical_xpass.pc_ignore_teammates_lane_survival_from_args(both_args))
        self.assertTrue(generate_physical_xpass.pc_ignore_teammates_control_from_args(both_args))

    def test_generate_physical_xpass_cli_rejects_split_teammate_flags_without_pc_xpass(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--ignore-teammates-lane-survival"])
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--ignore-teammates-control"])

    def test_generate_physical_xpass_cli_rejects_pc_xpass_legacy_feature_mode(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--pc-xpass", "--feature-run-id", "feature_run"])

    def test_generate_physical_xpass_cli_rejects_disabling_all_metric_outputs(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--no-noise-kernel", "--no-max", "--no-topmean"])

    def test_generate_physical_xpass_cli_rejects_non_positive_top_n(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--top-n", "0"])
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--pc-xpass", "--top-n-values", "5", "0"])
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--top-n-values", "5"])

    def test_generate_physical_xpass_cli_rejects_non_positive_sigma_factors(self) -> None:
        for flag in ["--sigma-angle", "--sigma-speed", "--sigma-distance"]:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", flag, "0"])

    def test_generate_physical_xpass_cli_rejects_non_positive_control_function_values(self) -> None:
        for flag in ["--control-function-power", "--control-function-inflection-point", "--control-function-gamma"]:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    generate_physical_xpass.parse_args(["--pc-xpass", flag, "0"])

    def test_generate_physical_xpass_cli_rejects_max_speed_below_grid_min(self) -> None:
        with self.assertRaises(SystemExit):
            generate_physical_xpass.parse_args(["--feature-run-id", "feature_run", "--max-speed", "2"])

    def test_run_hawkeye_cli_accepts_xpass_weight_versions(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_hawkeye.py", "--use-physical-xpass", "--xpass-weight", "v2", "--x-pass-version", "top25", "--ball-z-limit", "1.0"],
        ):
            args = run_hawkeye.parse_args()

        self.assertTrue(args.use_physical_xpass)
        self.assertEqual(args.xpass_weight, "v2")
        self.assertEqual(args.x_pass_version, "top25")
        self.assertEqual(args.ball_z_limit, "1.0")

        with patch.object(sys, "argv", ["run_hawkeye.py", "--use-physical-xpass", "--xpass-weight-v2"]):
            with self.assertRaises(SystemExit):
                run_hawkeye.parse_args()
        for old_flag in ["--max-xpass", "--topmean-xpass", "--top10-xpass"]:
            with self.subTest(old_flag=old_flag):
                with patch.object(sys, "argv", ["run_hawkeye.py", "--use-physical-xpass", old_flag]):
                    with self.assertRaises(SystemExit):
                        run_hawkeye.parse_args()

    def test_generate_epv_cli_accepts_physical_xpass_flags(self) -> None:
        args = generate_epv.parse_args(
            [
                "--use-physical-xpass",
                "--x-pass-version",
                "top25",
                "--xpass-weight",
                "v2",
                "--ball-z-limit",
                "1.0",
                "--physical-cache-dir",
                "runtime_cache",
            ]
        )

        self.assertTrue(args.use_physical_xpass)
        self.assertEqual(args.x_pass_version, "top25")
        self.assertEqual(args.xpass_weight, "v2")
        self.assertEqual(args.ball_z_limit, "1.0")
        self.assertEqual(args.physical_cache_dir, "runtime_cache")

    def test_generate_epv_configures_only_pass_success_physical_xpass(self) -> None:
        args = generate_epv.parse_args(
            [
                "--use-physical-xpass",
                "--x-pass-version",
                "top25",
                "--xpass-weight",
                "v2",
                "--ball-z-limit",
                "1.0",
                "--physical-cache-dir",
                "runtime_cache",
            ]
        )
        model_specs = make_epv_model_specs()

        physical_cache_dir = generate_epv.configure_epv_physical_xpass(args, model_specs)

        self.assertEqual(physical_cache_dir, "runtime_cache")
        self.assertTrue(model_specs["pass_success"].args["inference_use_physical_xpass"])
        self.assertEqual(model_specs["pass_success"].args["x_pass_version"], "top25")
        self.assertEqual(model_specs["pass_success"].args["xpass_weight"], "v2")
        self.assertEqual(model_specs["pass_success"].args["ball_z_limit"], "1.0")
        self.assertTrue(model_specs["pass_success"].args["physical_runtime_cache_read_only"])
        self.assertFalse(model_specs["pass_success"].args["physical_runtime_cache_refresh"])
        self.assertEqual(model_specs["pass_success"].args["physical_cache_dir"], "runtime_cache")
        for task in ("pass_intent", "outcome_scoring", "outcome_conceding"):
            self.assertNotIn("inference_use_physical_xpass", model_specs[task].args)
            self.assertNotIn("physical_cache_dir", model_specs[task].args)

    def test_generate_epv_physical_xpass_metadata_records_topmean_v2(self) -> None:
        args = generate_epv.parse_args(
            [
                "--use-physical-xpass",
                "--x-pass-version",
                "top25",
                "--xpass-weight",
                "v2",
                "--ball-z-limit",
                "1.0",
                "--physical-cache-dir",
                "runtime_cache",
            ]
        )
        model_specs = make_epv_model_specs()
        generate_epv.configure_epv_physical_xpass(args, model_specs)

        metadata = generate_epv.epv_physical_xpass_metadata(
            args,
            model_specs,
            physical_cache_dir="runtime_cache",
            runtime_stats={},
            skipped_actions={"match_1": {"pass_success": {"reason": "missing_cache_row"}}},
        )

        self.assertTrue(metadata["physical_xpass_requested"])
        self.assertEqual(metadata["physical_xpass_metric"], PHYSICAL_XPASS_METRIC_TOPMEAN)
        self.assertEqual(metadata["x_pass_version"], "top25")
        self.assertEqual(metadata["physical_xpass_weight_version"], "v2")
        self.assertEqual(metadata["physical_xpass_ball_z_limit"], 1.0)
        self.assertEqual(metadata["physical_cache_dir"], "runtime_cache")
        self.assertEqual(metadata["physical_xpass_skipped_actions"], {"match_1": {"pass_success": {"reason": "missing_cache_row"}}})
        self.assertTrue(metadata["physical_xpass_cache_summary"]["physical_xpass_required"])

    def test_generate_epv_without_physical_xpass_leaves_model_args_unchanged(self) -> None:
        args = SimpleNamespace(
            use_physical_xpass=False,
            max_xpass=False,
            topmean_xpass=False,
            top10mean_xpass=False,
            xpass_weight_v2=False,
            physical_cache_dir="runtime_cache",
        )
        model_specs = make_epv_model_specs()

        generate_epv.configure_epv_physical_xpass(args, model_specs)
        metadata = generate_epv.epv_physical_xpass_metadata(
            args,
            model_specs,
            physical_cache_dir="runtime_cache",
            runtime_stats={},
            skipped_actions={},
        )

        self.assertNotIn("inference_use_physical_xpass", model_specs["pass_success"].args)
        self.assertNotIn("physical_cache_dir", model_specs["pass_success"].args)
        self.assertFalse(metadata["physical_xpass_requested"])
        self.assertIsNone(metadata["physical_xpass_metric"])
        self.assertIsNone(metadata["physical_xpass_weight_version"])
        self.assertFalse(metadata["physical_xpass_cache_summary"]["physical_xpass_required"])

    def test_runtime_physical_xpass_cache_cli_flags(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_hawkeye.py",
                "--physical-cache-dir",
                "cache",
                "--no-physical-cache",
                "--refresh-physical-cache",
                "--physical-num-workers",
                "2",
                "--physical-worker-thread-limit",
                "3",
                "--physical-batch-size",
                "4",
            ],
        ):
            hawkeye_args = run_hawkeye.parse_args()
        with patch.object(
            sys,
            "argv",
            [
                "run_benchmark.py",
                "--physical-cache-dir",
                "cache",
                "--no-physical-cache",
                "--refresh-physical-cache",
                "--physical-num-workers",
                "2",
                "--physical-worker-thread-limit",
                "3",
                "--physical-batch-size",
                "4",
            ],
        ):
            benchmark_args = run_benchmark.parse_args()
        skillcorner_args = run_skillcorner.parse_args(
            [
                "--physical-cache-dir",
                "cache",
                "--no-physical-cache",
                "--refresh-physical-cache",
                "--physical-num-workers",
                "2",
                "--physical-worker-thread-limit",
                "3",
                "--physical-batch-size",
                "4",
            ]
        )

        for args in [hawkeye_args, benchmark_args, skillcorner_args]:
            self.assertEqual(args.physical_cache_dir, "cache")
            self.assertTrue(args.no_physical_cache)
            self.assertTrue(args.refresh_physical_cache)
            self.assertEqual(args.physical_num_workers, "2")
            self.assertEqual(args.physical_worker_thread_limit, 3)
            self.assertEqual(args.physical_batch_size, 4)

    def test_runtime_prewarm_runner_helpers_batch_expected_objects(self) -> None:
        model = SimpleNamespace(
            args={
                **make_pass_success_args(
                    use_physical_xpass=False,
                    inference_use_physical_xpass=True,
                    model_variant="gat_baseline",
                    physical_xpass_source=PHYSICAL_XPASS_LEGACY_SOURCE,
                    physical_xpass_speed_aggregation=PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX,
                ),
                "physical_runtime_cache_refresh": True,
            }
        )
        model_specs = {"pass_success": model}
        labels = torch.stack([make_label(action_index=1)])
        graph = make_graph()
        hawkeye_situation = SimpleNamespace(match_id="hawkeye_1", labels=labels, graph_features_0=[graph])
        benchmark_states = [
            SimpleNamespace(match_id="benchmark_1", labels=labels, graph_features_0=[graph]),
            SimpleNamespace(match_id="benchmark_2", labels=labels, graph_features_0=[graph]),
        ]
        skillcorner_possessions = [
            SimpleNamespace(match_id="skillcorner_1", labels=labels, graph_features_0=[graph]),
            SimpleNamespace(match_id="skillcorner_1", labels=labels, graph_features_0=[graph]),
        ]

        with patch.object(run_hawkeye, "prewarm_physical_xpass_runtime_cache", return_value={"cache_misses": 1}) as hawkeye_prewarm:
            run_hawkeye._prewarm_hawkeye_physical_xpass(
                hawkeye_situation,
                model_specs,
                cache_dir="cache",
                num_workers="2",
                worker_thread_limit=3,
                physical_batch_size=4,
            )
        with patch.object(run_benchmark, "prewarm_physical_xpass_runtime_cache", return_value={"cache_misses": 2}) as benchmark_prewarm:
            run_benchmark._prewarm_benchmark_physical_xpass(
                benchmark_states,
                model_specs,
                cache_dir="cache",
                num_workers="2",
                worker_thread_limit=3,
                physical_batch_size=4,
            )
        with patch.object(run_skillcorner, "prewarm_physical_xpass_runtime_cache", return_value={"cache_misses": 2}) as skillcorner_prewarm:
            run_skillcorner._prewarm_skillcorner_physical_xpass(
                skillcorner_possessions,
                model_specs,
                cache_dir="cache",
                num_workers="2",
                worker_thread_limit=3,
                physical_batch_size=4,
            )

        self.assertEqual(hawkeye_prewarm.call_args.args[0][0]["match_id"], "hawkeye_1")
        self.assertEqual(hawkeye_prewarm.call_args.kwargs["num_workers"], "2")
        self.assertEqual(hawkeye_prewarm.call_args.kwargs["worker_thread_limit"], 3)
        self.assertEqual(hawkeye_prewarm.call_args.kwargs["physical_batch_size"], 4)
        self.assertTrue(hawkeye_prewarm.call_args.kwargs["refresh"])
        self.assertEqual(hawkeye_prewarm.call_args.kwargs["source"], PHYSICAL_XPASS_SOURCE)
        self.assertEqual(benchmark_prewarm.call_args.kwargs["source"], PHYSICAL_XPASS_SOURCE)
        self.assertEqual(skillcorner_prewarm.call_args.kwargs["source"], PHYSICAL_XPASS_SOURCE)
        self.assertEqual(hawkeye_prewarm.call_args.kwargs["speed_aggregation"], PHYSICAL_XPASS_SPEED_AGGREGATION_PACKAGE_MAX)
        self.assertEqual([item["match_id"] for item in benchmark_prewarm.call_args.args[0]], ["benchmark_1", "benchmark_2"])
        self.assertEqual([item["match_id"] for item in skillcorner_prewarm.call_args.args[0]], ["skillcorner_1", "skillcorner_1"])

    def test_benchmark_state_progress_formatting_helper(self) -> None:
        self.assertEqual(
            run_benchmark.format_benchmark_state_progress(17, 100, 9, 1),
            "benchmark state 17/100 | modification_9 game_state_1 | 83 states left",
        )
        self.assertEqual(
            run_benchmark.format_benchmark_state_progress(99, 100, 50, 1),
            "benchmark state 99/100 | modification_50 game_state_1 | 1 state left",
        )

    def test_benchmark_main_does_not_prewarm_during_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_parent = Path(tmpdir) / "runs"
            args = SimpleNamespace(
                input_dir=str(Path(tmpdir) / "benchmark"),
                modification=None,
                limit=None,
                device="cpu",
                bundle_id="bundle",
                action_intent_model_id=None,
                pass_intent_model_id=None,
                pass_success_model_id=None,
                outcome_scoring_model_id=None,
                outcome_conceding_model_id=None,
                run_id="benchmark_component_test",
                output_dir=str(output_parent),
                physical_cache_dir="cache",
                use_physical_xpass=True,
                max_xpass=False,
                top10mean_xpass=False,
                no_physical_cache=False,
                refresh_physical_cache=True,
                physical_num_workers="auto",
                physical_worker_thread_limit=1,
                physical_batch_size=16,
            )
            pass_success_model = SimpleNamespace(
                args={
                    "task": "pass_success",
                    "use_physical_xpass": True,
                    "model_variant": "gat_phys_logit_offset",
                    "physical_xpass_source": PHYSICAL_XPASS_LEGACY_SOURCE,
                    "physical_xpass_teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    "physical_xpass_speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                }
            )
            model_specs = {"pass_success": pass_success_model}

            def build_state(_game_state, *, modification_id, game_state_id, higher_state_id, add_v_edge_features):
                del higher_state_id, add_v_edge_features
                if modification_id == 2 and game_state_id == 2:
                    raise ValueError("bad state")
                state = SimpleNamespace(
                    match_id=f"modification_{modification_id}_game_state_{game_state_id}",
                    physical_xpass_runtime_stats=None,
                )
                rows = pd.DataFrame([{"team": 1, "frame": f"{modification_id}:{game_state_id}"}])
                stats = {
                    "states": 1,
                    "valid_frames": 1,
                    "total_frames": 1,
                    "skipped_missing_ball": 0,
                    "skipped_missing_possessor": 0,
                    "skipped_missing_graph": 0,
                }
                return state, rows, stats

            written_metadata: dict[str, object] = {}
            RecordingTqdm.calls = []

            with patch.object(run_benchmark, "parse_args", return_value=args), \
                patch.object(
                    run_benchmark,
                    "resolve_model_selection",
                    return_value=(
                        {
                            "action_intent": "action_intent/fake",
                            "pass_intent": "pass_intent/fake",
                            "pass_success": "pass_success/fake",
                            "outcome_scoring": "outcome_scoring/fake",
                            "outcome_conceding": "outcome_conceding/fake",
                        },
                        {
                            "intended_receiver_mode": "angle_only",
                            "return_type": "disc_0.9",
                            "target_family": "goal",
                        },
                        None,
                    ),
                ), \
                patch.object(run_benchmark, "discover_benchmark_modifications", return_value=([1, 2], {})), \
                patch.object(run_benchmark, "load_benchmark_models", return_value=model_specs), \
                patch.object(
                    run_benchmark,
                    "validate_model_graph_schemas",
                    return_value={"add_v_edge_features": True},
                ), \
                patch.object(run_benchmark, "get_model_provenance", return_value={"feature_signature": "sig"}), \
                patch.object(
                    run_benchmark,
                    "load_benchmark_modification_data",
                    side_effect=lambda modification_id, _input_dir: {
                        "higher_state_id": 1,
                        "game_state_1": object(),
                        "game_state_2": object(),
                    },
                ), \
                patch.object(run_benchmark, "build_benchmark_state", side_effect=build_state), \
                patch.object(
                    run_benchmark,
                    "_prewarm_benchmark_physical_xpass",
                    return_value={"cache_misses": 3},
                ) as prewarm, \
                patch.object(run_benchmark, "infer_benchmark_components", side_effect=lambda state, _models, device: {"state": state.match_id}), \
                patch.object(run_benchmark, "build_benchmark_export", side_effect=lambda rows, _state, _components: rows), \
                patch.object(run_benchmark, "summarize_benchmark_stats", return_value={"states": 3, "valid_frames": 3, "total_frames": 3, "skipped_missing_ball": 0, "skipped_missing_possessor": 0, "skipped_missing_graph": 0}), \
                patch.object(run_benchmark.pd.DataFrame, "to_parquet"), \
                patch.object(run_benchmark.pd.DataFrame, "to_csv"), \
                patch.object(run_benchmark, "write_run_metadata", side_effect=lambda _path, metadata: written_metadata.update(metadata)), \
                patch.object(run_benchmark, "write_latest_run"), \
                patch.object(
                    run_benchmark,
                    "run_benchmark_postprocessing",
                    return_value=(None, {"agreements": 0, "disagreements": 0}, Path("summary.csv"), Path("summary.txt")),
                ), \
                patch.object(run_benchmark, "tqdm", RecordingTqdm), \
                patch.object(run_benchmark, "update_benchmark_runs_ledger", return_value=Path("ledger.csv")):
                run_benchmark.main()

            self.assertEqual(len(RecordingTqdm.calls), 1)
            self.assertEqual(RecordingTqdm.calls[0].kwargs["desc"], "benchmark states")
            self.assertEqual(RecordingTqdm.calls[0].kwargs["total"], 3)
            self.assertEqual(
                RecordingTqdm.calls[0].postfixes,
                [
                    {"modification": 1, "game_state": 1},
                    {"modification": 1, "game_state": 2},
                    {"modification": 2, "game_state": 1},
                ],
            )
            self.assertEqual(
                RecordingTqdm.calls[0].writes,
                [
                    "benchmark state 1/3 | modification_1 game_state_1 | 2 states left",
                    "benchmark state 2/3 | modification_1 game_state_2 | 1 state left",
                    "benchmark state 3/3 | modification_2 game_state_1 | 0 states left",
                ],
            )
            prewarm.assert_not_called()
            self.assertFalse(pass_success_model.args["physical_runtime_cache_refresh"])
            self.assertTrue(pass_success_model.args["physical_runtime_cache_read_only"])
            self.assertEqual(written_metadata["physical_xpass_checkpoint_source"], PHYSICAL_XPASS_LEGACY_SOURCE)
            self.assertEqual(written_metadata["physical_xpass_runtime_source"], PHYSICAL_XPASS_SOURCE)
            self.assertEqual(written_metadata["physical_xpass_prewarm_stats"], {})
            self.assertEqual(written_metadata["physical_xpass_cache_summary"]["reason"], "refresh_requested")
            self.assertEqual(written_metadata["physical_xpass_cache_summary"]["cache_misses"], 0)
            self.assertEqual(written_metadata["processed_modifications"], [1, 2])
            self.assertEqual(
                written_metadata["skipped_states"],
                [{"modification": 2, "game_state": 2, "error": "ValueError: bad state"}],
            )

    def test_benchmark_main_no_physical_cache_skips_global_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                input_dir=str(Path(tmpdir) / "benchmark"),
                modification=None,
                limit=None,
                device="cpu",
                bundle_id="bundle",
                action_intent_model_id=None,
                pass_intent_model_id=None,
                pass_success_model_id=None,
                outcome_scoring_model_id=None,
                outcome_conceding_model_id=None,
                run_id="benchmark_component_test",
                output_dir=str(Path(tmpdir) / "runs"),
                physical_cache_dir="cache",
                no_physical_cache=True,
                refresh_physical_cache=False,
                physical_num_workers="auto",
                physical_worker_thread_limit=1,
                physical_batch_size=16,
            )
            pass_success_model = SimpleNamespace(
                args={
                    "task": "pass_success",
                    "use_physical_xpass": True,
                    "model_variant": "gat_phys_logit_offset",
                    "physical_xpass_source": PHYSICAL_XPASS_SOURCE,
                    "physical_xpass_teammate_policy": PHYSICAL_XPASS_TEAMMATE_POLICY_CONSIDER,
                    "physical_xpass_speed_aggregation": PHYSICAL_XPASS_SPEED_AGGREGATION_EXACT_SEPARATE_SPEED,
                }
            )
            state = SimpleNamespace(match_id="benchmark_1", physical_xpass_runtime_stats=None)
            stats = {
                "states": 1,
                "valid_frames": 1,
                "total_frames": 1,
                "skipped_missing_ball": 0,
                "skipped_missing_possessor": 0,
                "skipped_missing_graph": 0,
            }
            RecordingTqdm.calls = []

            with patch.object(run_benchmark, "parse_args", return_value=args), \
                patch.object(
                    run_benchmark,
                    "resolve_model_selection",
                    return_value=(
                        {
                            "action_intent": "action_intent/fake",
                            "pass_intent": "pass_intent/fake",
                            "pass_success": "pass_success/fake",
                            "outcome_scoring": "outcome_scoring/fake",
                            "outcome_conceding": "outcome_conceding/fake",
                        },
                        {
                            "intended_receiver_mode": "angle_only",
                            "return_type": "disc_0.9",
                            "target_family": "goal",
                        },
                        None,
                    ),
                ), \
                patch.object(run_benchmark, "discover_benchmark_modifications", return_value=([1], {})), \
                patch.object(run_benchmark, "load_benchmark_models", return_value={"pass_success": pass_success_model}), \
                patch.object(run_benchmark, "validate_model_graph_schemas", return_value={"add_v_edge_features": True}), \
                patch.object(run_benchmark, "get_model_provenance", return_value={"feature_signature": "sig"}), \
                patch.object(
                    run_benchmark,
                    "load_benchmark_modification_data",
                    return_value={"higher_state_id": 1, "game_state_1": object(), "game_state_2": object()},
                ), \
                patch.object(
                    run_benchmark,
                    "build_benchmark_state",
                    side_effect=[(state, pd.DataFrame([{"team": 1}]), stats), ValueError("bad state")],
                ), \
                patch.object(run_benchmark, "_prewarm_benchmark_physical_xpass") as prewarm, \
                patch.object(run_benchmark, "infer_benchmark_components", return_value={}), \
                patch.object(run_benchmark, "build_benchmark_export", side_effect=lambda rows, _state, _components: rows), \
                patch.object(run_benchmark, "summarize_benchmark_stats", return_value=stats), \
                patch.object(run_benchmark.pd.DataFrame, "to_parquet"), \
                patch.object(run_benchmark.pd.DataFrame, "to_csv"), \
                patch.object(run_benchmark, "write_run_metadata"), \
                patch.object(run_benchmark, "write_latest_run"), \
                patch.object(
                    run_benchmark,
                    "run_benchmark_postprocessing",
                    return_value=(None, {"agreements": 0, "disagreements": 0}, Path("summary.csv"), Path("summary.txt")),
                ), \
                patch.object(run_benchmark, "tqdm", RecordingTqdm), \
                patch.object(run_benchmark, "update_benchmark_runs_ledger", return_value=Path("ledger.csv")):
                run_benchmark.main()

            prewarm.assert_not_called()
            self.assertEqual(len(RecordingTqdm.calls), 1)
            self.assertEqual(RecordingTqdm.calls[0].kwargs["total"], 1)

    def test_generate_physical_xpass_auto_workers_resolves_with_higher_default_cap(self) -> None:
        with patch("physical_pass_model.os.cpu_count", return_value=16):
            self.assertEqual(resolve_physical_num_workers("auto"), PHYSICAL_DEFAULT_MAX_AUTO_WORKERS)
            self.assertEqual(generate_physical_xpass.resolve_num_workers("auto"), PHYSICAL_DEFAULT_MAX_AUTO_WORKERS)
            self.assertEqual(resolve_physical_num_workers("auto", max_auto_workers=8), 8)
            self.assertEqual(generate_physical_xpass.resolve_num_workers("auto", max_auto_workers=8), 8)

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
