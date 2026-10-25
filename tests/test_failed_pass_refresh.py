import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import failed_pass_refresh as refresh
from datatools.failed_pass_trajectory import KEY, VERSION
from datatools.pass_trajectory import PASS_DURATION_KEY, PASS_DURATION
from test_failed_pass_trajectory import sample


def test_match_refresh_preserves_success_and_other_inputs(tmp_path):
    action, tracking = sample()
    actions = pd.DataFrame([action, action.copy()], index=[1, 2])
    actions.at[2, "success"] = True
    actions.at[2, "original_event_id"] = "b"
    root, data = tmp_path / "run", tmp_path / "data"
    for folder in ("pass_endpoints", "resolved_actions_original", "action_graphs_success_intent", "success_intent_labels"):
        (root / folder).mkdir(parents=True)
    for folder in ("tracking_processed", "lineup"):
        (data / folder).mkdir(parents=True)
    actions.to_parquet(root / "pass_endpoints/m.parquet")
    actions.to_parquet(root / "resolved_actions_original/m.parquet")
    tracking.to_parquet(data / "tracking_processed/m.parquet")
    pd.DataFrame({"stats_perform_match_id": ["m"]}).to_parquet(data / "lineup/line_up.parquet")
    (root / "metadata.json").write_text(json.dumps({KEY: VERSION}))
    graphs = [Data(x=torch.ones((2, 24)),
                   edge_index=torch.cartesian_prod(torch.arange(2), torch.arange(2)).T.contiguous(),
                   edge_attr=torch.ones((4, 2))) for _ in range(2)]
    torch.save(graphs, root / "action_graphs_success_intent/m.pt")
    torch.save(torch.tensor([[1.], [2.]]), root / "success_intent_labels/m.pt")
    match = SimpleNamespace(actions=actions, tracking=tracking, fps=25, match_id="m")
    resolver = SimpleNamespace(resolve=lambda a: SimpleNamespace(trajectory=25))
    with patch.object(refresh, "DATA_ROOT", data), patch("datatools.graph_feature.build_match_for_feature_generation", return_value=match), patch("datatools.graph_feature.ensure_pass_endpoints", return_value=resolver), patch.object(refresh, "ensure_pass_endpoints", return_value=resolver):
        report = refresh.refresh_match((str(root), "m", {}, 1))
    assert report["counts"] == {"shortened": 1}
    result = torch.load(root / "action_graphs_success_intent/m.pt", weights_only=False)
    torch.testing.assert_close(result[1].x, graphs[1].x)
    torch.testing.assert_close(result[0].x[:, :19], graphs[0].x[:, :19])
    assert result[0].trajectory_end_frame_id == 11
    assert result[0].x[0, 19].item() == pytest.approx(10)
    pd.testing.assert_frame_equal(pd.read_parquet(root / "pass_endpoints/m.parquet"), actions)
    # Rerunning must use the original cutoff, with identical results.
    with patch.object(refresh, "DATA_ROOT", data), patch("datatools.graph_feature.build_match_for_feature_generation", return_value=SimpleNamespace(actions=actions, tracking=tracking, fps=25, match_id="m")), patch("datatools.graph_feature.ensure_pass_endpoints", return_value=resolver), patch.object(refresh, "ensure_pass_endpoints", return_value=resolver):
        repeated = refresh.refresh_match((str(root), "m", {}, 1))
        assert {k: v for k, v in repeated.items() if k != "graph_validation"} == {
            k: v for k, v in report.items() if k != "graph_validation"}
    second = torch.load(root / "action_graphs_success_intent/m.pt", weights_only=False)
    torch.testing.assert_close(second[0].x, result[0].x)


def test_derived_refresh_invalidation_failure_and_resume(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "pass_endpoints").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(source / "pass_endpoints/m.parquet")
    (source / "resolved_actions_model").mkdir()
    (source / "resolved_actions_model/m.pt").write_bytes(b"old")
    original = {"status": "completed", PASS_DURATION_KEY: PASS_DURATION,
                "intended_receiver_modes": ["original", "angle_only", "model"], "intended_receiver_model_id": "old"}
    (source / "metadata.json").write_text(json.dumps(original))
    monkeypatch.setattr(refresh, "get_feature_run_root", lambda run: tmp_path / run)
    monkeypatch.setattr(refresh, "load_feature_run_metadata", lambda run, **kw: json.loads((tmp_path / run / "metadata.json").read_text()))
    monkeypatch.setattr(refresh, "write_latest_run", lambda *a: None)
    args = SimpleNamespace(extend_feature_run_id="source", run_id="derived", in_place=False, artifact_copy_mode="copy", num_workers="1", worker_thread_limit=1)
    def fail(task):
        raise ValueError("interrupted")
    monkeypatch.setattr(refresh, "refresh_match", fail)
    with pytest.raises(ValueError, match="interrupted"):
        refresh.run_failed_pass_refresh(args)
    target = tmp_path / "derived"
    metadata = json.loads((target / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["intended_receiver_modes"] == ["original", "angle_only"]
    assert not (target / "resolved_actions_model").exists()
    assert json.loads((source / "metadata.json").read_text()) == original
    monkeypatch.setattr(refresh, "refresh_match", lambda task: {"total": 1, "counts": {"unchanged": 1}})
    args.extend_feature_run_id, args.in_place, args.run_id = "derived", True, None
    refresh.run_failed_pass_refresh(args)
    assert json.loads((target / "metadata.json").read_text())["status"] == "completed"


def test_model_replacement_removes_carry_artifacts_only(tmp_path):
    from scripts.generate_relevant_features import remove_model_mode_artifacts
    for name in ("resolved_actions_model", "resolved_actions_model_carries",
                 "action_labels_next_3_model_carries", "resolved_actions_original_carries"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "m.pt").write_bytes(b"old")
    remove_model_mode_artifacts(tmp_path, ["next_3"])
    assert not (tmp_path / "resolved_actions_model").exists()
    assert not (tmp_path / "resolved_actions_model_carries").exists()
    assert not (tmp_path / "action_labels_next_3_model_carries").exists()
    assert (tmp_path / "resolved_actions_original_carries/m.pt").exists()


def test_all_unavailable_run_keeps_diagnostic_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "pass_endpoints").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(source / "pass_endpoints/m.parquet")
    (source / "metadata.json").write_text(json.dumps({"status": "completed", PASS_DURATION_KEY: PASS_DURATION}))
    monkeypatch.setattr(refresh, "get_feature_run_root", lambda run: tmp_path / run)
    monkeypatch.setattr(refresh, "load_feature_run_metadata", lambda run, **kw: json.loads((tmp_path / run / "metadata.json").read_text()))
    monkeypatch.setattr(refresh, "write_latest_run", lambda *a: pytest.fail("Failed run must not become latest"))
    def worker(task):
        report = dict(match_id="m", total=1, counts={"unavailable": 1}, unavailable=[dict(action_index=1, reason="missing_tracking")])
        refresh.atomic_write_json(Path(task[0]) / "failed_pass_trajectory_refresh/m.json", report)
        return report
    from pathlib import Path
    monkeypatch.setattr(refresh, "refresh_match", worker)
    args = SimpleNamespace(extend_feature_run_id="source", run_id="derived", in_place=False, artifact_copy_mode="copy", num_workers="1", worker_thread_limit=1)
    with pytest.raises(ValueError, match="All failed passes unavailable"):
        refresh.run_failed_pass_refresh(args)
    result = json.loads((tmp_path / "derived/metadata.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_pass_refresh_reports"][0]["unavailable"][0]["reason"] == "missing_tracking"
