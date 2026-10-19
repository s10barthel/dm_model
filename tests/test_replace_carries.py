import json
from types import SimpleNamespace

import pytest
import torch

from datatools.artifact_storage import clone_feature_artifacts
from datatools.carry_policy import DEFINITION
from scripts import replace_carries as rc


def test_clone_excludes_old_carries_and_preserves_shared_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    for folder in ["action_graphs", "action_graphs_carries", "carry_sidecars/segments"]:
        path = source / folder / "m.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save([1], path)
    clone_feature_artifacts(source, output, exclude=rc.excluded_artifact)
    assert (output / "action_graphs/m.pt").exists()
    assert not (output / "action_graphs_carries").exists()
    assert not (output / "carry_sidecars").exists()
    from datatools.artifact_storage import atomic_torch_save
    atomic_torch_save([2], output / "action_graphs/m.pt")
    assert torch.load(source / "action_graphs/m.pt", weights_only=False) == [1]


def test_cloning_can_resume_after_partial_copy(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "a.parquet").write_bytes(b"a")
    (source / "b.parquet").write_bytes(b"b")
    output.mkdir()
    (output / "a.parquet").write_bytes(b"a")
    clone_feature_artifacts(source, output, resume=True, mode="copy")
    assert (output / "b.parquet").read_bytes() == b"b"


def test_failed_replacement_resumes_completed_matches_and_preserves_source(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "derived"
    source.mkdir()
    original = dict(status="completed", carry_variant=dict(available=True, definition="sportec_fernandez_1s_v2"))
    (source / "metadata.json").write_text(json.dumps(original))
    (source / "canonical.parquet").write_bytes(b"source")
    checksum = rc.digest(source / "canonical.parquet")
    monkeypatch.setattr(rc, "resolve_source", lambda value: source)
    monkeypatch.setattr(rc, "load_feature_run_metadata", lambda *a, **k: original)
    monkeypatch.setattr(rc, "match_ids", lambda *a: ["m1", "m2"])
    monkeypatch.setattr(rc, "get_feature_run_root", lambda name: output)
    def write(root, metadata):
        root.mkdir(parents=True, exist_ok=True)
        (root / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(rc, "write_run_metadata", write)
    latest = []
    monkeypatch.setattr(rc, "write_latest_run", lambda *a: latest.append(a))
    calls = []
    fail = [True]
    def rebuild(task):
        match_id = task[3]
        calls.append(match_id)
        if match_id == "m2" and fail[0]:
            raise RuntimeError("interrupted")
        artifact = output / f"{match_id}.parquet"
        artifact.write_bytes(b"done")
        marker = dict(definition=DEFINITION, exclusions={}, segments=2, elapsed_seconds=1,
                      artifacts={artifact.name: 4})
        (output / "carry_replacement").mkdir(exist_ok=True)
        (output / "carry_replacement" / f"{match_id}.json").write_text(json.dumps(marker))
        return dict(match_id=match_id, **marker)
    monkeypatch.setattr(rc, "rebuild_match", rebuild)
    args = SimpleNamespace(extend_feature_run_id="source", run_id="derived", artifact_copy_mode="hardlink",
                           num_workers="1", worker_thread_limit=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        rc.replace_carries(args)
    assert json.loads((output / "metadata.json").read_text())["status"] == "failed"
    assert not latest
    fail[0] = False
    rc.replace_carries(args)
    assert calls == ["m1", "m2", "m2"]
    updated = json.loads((output / "metadata.json").read_text())
    assert updated["status"] == "completed"
    assert updated["carry_variant"]["definition"] == DEFINITION
    assert rc.digest(source / "canonical.parquet") == checksum
    assert json.loads((source / "metadata.json").read_text()) == original
    assert latest == [("feature", "derived")]


def test_marker_requires_every_committed_artifact(tmp_path):
    path = tmp_path / "carry_replacement"
    path.mkdir()
    (path / "m.json").write_text(json.dumps(dict(definition=DEFINITION, artifacts={"missing.pt": 4})))
    assert not rc.marker_complete(tmp_path, "m")


def test_replacement_cli_inherits_settings_and_disallows_in_place(monkeypatch):
    from scripts.generate_relevant_features import parse_args
    monkeypatch.setattr("sys.argv", ["generate_relevant_features.py", "--replace-carries", "--extend-feature-run-id", "latest"])
    args = parse_args()
    assert args.replace_carries
    assert args.carry_acquisition_distance == 3
    monkeypatch.setattr("sys.argv", ["generate_relevant_features.py", "--replace-carries", "--extend-feature-run-id", "latest", "--in-place"])
    with pytest.raises(SystemExit):
        parse_args()
