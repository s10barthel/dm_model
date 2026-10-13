"""Exercise actual learned inference and artifacts inside the evaluation wrapper."""
import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import test as evaluation
from datatools import config
from models.gnn import GNN
from models.inference_masks import configure_inference_masks
from models.utils import run_epoch
from models.utils import calc_threshold_binary_metrics
from scripts import evaluate_relevant_models as wrapper, model_comparison


def model_args(task="pass_success"):
    return dict(task=task, model="gat", gnn_task="node_binary", node_in_dim=20,
                edge_in_dim=2, node_emb_dim=8, graph_emb_dim=8, gnn_layers=1,
                gnn_heads=1, skip_conn=True, out_dim=2 if task.startswith("outcome") else 1,
                include_out=False, possessor_aware=True, poss_vel_aware=True,
                poss_rel_vel_aware=True, lambda_l1=0., clip=10., print_freq=99,
                use_xg=False, use_xt=False, use_epv=False, use_goal_distance=False,
                classification_threshold=.5, return_pass_success_height_evaluation=task == "pass_success",
                observed_pass_height_stratification=task == "pass_success")


def samples():
    result = []
    for i in range(4):
        x = torch.arange(60, dtype=torch.float32).reshape(3, 20) / 10
        x[:, :3] = 0
        x[:2, 0] = 1
        x[:, 13] = 0
        x[0, 13] = 1
        x[:, 8] += i
        graph = Data(x=x, edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
                     edge_attr=torch.ones(3, 2), inference_possessor_mask=x[:, 13] == 1,
                     evaluation_match_id="a" if i < 2 else "b", evaluation_source_row=i % 2,
                     node_ids=["home_1", "home_2", "away_3"])
        label = torch.zeros(len(config.LABEL_COLUMNS))
        for field, value in {"action_index": i, "intent_index": 1, "is_pass": 1,
                             "success": i % 2, "pass_high": i % 2, "pass_max_ball_z": 1 + i,
                             "scores": i % 2, "concedes": (i + 1) % 2,
                             "scores_goal_next10": i % 2, "concedes_goal_next10": (i + 1) % 2}.items():
            label[config.LABEL_INDEX[field]] = value
        result.append((graph, label, torch.tensor(1.)))
    return result


def test_wrapper_mask_reports_use_real_predictions_and_preserve_summary_variants(tmp_path, monkeypatch):
    torch.manual_seed(7)
    args = model_args()
    models = {model_id: GNN(dict(args)).eval() for model_id in ("pass_success/a", "pass_success/b")}
    source = samples()
    originals = [sample[0].x.clone() for sample in source]
    monkeypatch.setattr(evaluation, "EVALUATION_RUNS_DIR", tmp_path / "summary")
    def checkpoint(model_id, expected_task=None):
        return {"model_id": model_id, "task": "pass_success", "train_matches": 1,
                "train_match_ids": ["train"], "test_match_ids": ["a", "b"], "target_definitions": {},
                "checkpoint_record": {"args": args, "metadata": {"train_match_ids": ["train"], "validation_match_ids": []}, "is_complete": True}}
    monkeypatch.setattr(model_comparison, "checkpoint", checkpoint)
    commands = []
    def evaluate_command(command, **kwargs):
        commands.append(command)
        model_id = command[command.index("--model_id") + 1]
        output = Path(command[command.index("--evaluation-output-dir") + 1])
        configuration = command[command.index("--inference-configuration-id") + 1]
        masks = []
        if "--inference-mask" in command:
            start = command.index("--inference-mask") + 1
            for token in command[start:]:
                if token.startswith("--"):
                    break
                masks.append(token)
        model = copy.deepcopy(models[model_id])
        metadata = configure_inference_masks({"pass_success": model}, masks)
        metrics, rows, diagnostics = run_epoch(
            argparse.Namespace(**args), model, DataLoader(source, batch_size=2), device="cpu",
            return_learning_curve_rows=True, return_semantic_prediction_rows=True)
        assert diagnostics["binary"]["height_rows"]
        output.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(output / "learning_curve_predictions.csv", index=False)
        evaluation.write_model_evaluation_artifacts(
            output, model_id=model_id, task="pass_success", feature_run_id=None,
            diagnostic_feature_run_id=None, evaluation_timestamp="fixed",
            evaluation_options={"configuration_id": configuration, "inference_masks": metadata}, test_metrics=metrics)
    monkeypatch.setattr(wrapper.subprocess, "run", evaluate_command)
    cli = wrapper.parse_args([
        "--pass-success-model-id", "pass_success/a", "--pass-success-model-id", "pass_success/b",
        "--inference-mask-1", "accel", "--inference-mask-2", "pos-node-features",
        "--comparison-bootstrap-resamples", "0", "--evaluation-output-dir", str(tmp_path / "reports")])
    monkeypatch.setattr(wrapper, "parse_args", lambda: cli)
    wrapper.main()
    assert len(commands) == 6
    summaries = pd.read_csv(tmp_path / "summary" / "metrics_summary.csv")
    assert len(summaries) == 6
    assert set(summaries.configuration_id) == {"unmasked", "mask_1", "mask_2"}
    assert len(list((tmp_path / "reports").rglob("comparison_differences.csv"))) == 2
    for sample, original in zip(source, originals):
        torch.testing.assert_close(sample[0].x, original)
    for manifest in (tmp_path / "reports").rglob("experiment_manifest.json"):
        report = json.loads(manifest.read_text())
        if "variants" in report:
            assert len({entry["model_id"] for entry in report["variants"]}) == 1


def test_comparison_rows_retain_outcome_diagnostics():
    torch.manual_seed(8)
    args = model_args("outcome_scoring")
    model = GNN(args).eval()
    configure_inference_masks({"outcome_scoring": model}, ["accel"])
    metrics, rows, diagnostics = run_epoch(
        argparse.Namespace(**args), model, DataLoader(samples(), batch_size=2), device="cpu",
        return_learning_curve_rows=True, return_semantic_prediction_rows=True,
        return_outcome_evaluation=True)
    assert len(rows) == 4
    assert all("soft_target" in row and "execution_branch" in row for row in rows)
    assert len(diagnostics["outcome"]["prediction"]) == 4
    assert len(diagnostics["outcome"]["match_id"]) == 4


def test_comparison_classification_boundary_matches_evaluation():
    frame = pd.DataFrame({"match_id": ["a", "a", "b", "b"], "source_index": [0, 1, 0, 1],
                          "target": [0, 1, 0, 1], "prediction": [.5, .5, .2, .8]})
    expected = calc_threshold_binary_metrics(frame.target.to_numpy(), frame.prediction.to_numpy(), .5)
    summary, _ = model_comparison.compare(
        "pass_success", [{"model_id": "pass_success/a"}, {"model_id": "pass_success/b"}],
        [frame, frame.copy()], resamples=0, seed=42, threshold=.5)
    for name in ("precision", "recall", "f1"):
        assert summary.loc[summary.metric == name, "estimate"].tolist() == [expected[name]] * 2
