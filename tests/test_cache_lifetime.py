import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cache_lifetime import (CacheWorkflow, cache_lock, database, training_context,
                            workflow_context, ENVIRONMENT_KEY)


def claim(workflow, root, name="b", *, existing=False):
    session = workflow.session_for(root)
    path = Path(root) / ("a" * 32) / (name * 32 + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        path.write_bytes(b"legacy")
        path.with_suffix(".json").write_text("{}")
    with session.entry(path):
        built = not path.exists()
        if built:
            path.write_bytes(b"graphs")
            path.with_suffix(".json").write_text("{}")
        session.recorded(path, built=built, size=path.stat().st_size)
    return path


def test_success_cleanup_and_failure_keep(tmp_path):
    workflow = CacheWorkflow(tmp_path)
    path = claim(workflow, tmp_path)
    report = workflow.finish(True)
    assert not path.exists() and not path.with_suffix(".json").exists()
    assert report["roots"][0]["deleted_bytes"] == 6
    assert report["roots"][0]["created_entries"] == 1
    assert json.loads((tmp_path / ".sessions" / (workflow.descriptor["id"] + ".json")).read_text())["status"] == "completed"
    failed = CacheWorkflow(tmp_path)
    path = claim(failed, tmp_path)
    failed.finish(False)
    reuse = CacheWorkflow(tmp_path)
    claim(reuse, tmp_path)
    report = reuse.finish(True)
    assert path.exists()
    assert report["roots"][0]["retention_reasons"]["unfinished_or_failed_workflow"] == 1


@pytest.mark.parametrize("legacy,keep", [(True, False), (False, True)])
def test_legacy_and_explicit_keep_are_persistent(tmp_path, legacy, keep):
    workflow = CacheWorkflow(tmp_path, keep=keep)
    path = claim(workflow, tmp_path, existing=legacy)
    workflow.finish(True)
    reuse = CacheWorkflow(tmp_path)
    claim(reuse, tmp_path)
    reuse.finish(True)
    assert path.exists()


def test_overlapping_workflows_defer_cleanup(tmp_path):
    first, second = CacheWorkflow(tmp_path), CacheWorkflow(tmp_path)
    path = claim(first, tmp_path)
    claim(second, tmp_path)
    first.finish(True)
    assert path.exists()
    second.finish(True)
    assert not path.exists()


def test_multiple_roots_and_path_checks(tmp_path):
    root, other = tmp_path / "first", tmp_path / "other"
    workflow = CacheWorkflow(root)
    paths = [claim(workflow, r) for r in (root, other)]
    with pytest.raises(ValueError):
        with workflow.session_for(root).entry(tmp_path / "outside.pt"):
            pass
    workflow.finish(True)
    assert all(not p.exists() for p in paths)


def test_cleanup_error_preserves_training_success(tmp_path):
    workflow = CacheWorkflow(tmp_path)
    path = claim(workflow, tmp_path)
    original = Path.unlink
    def fail(self, *args, **kwargs):
        if self == path:
            raise PermissionError("busy")
        return original(self, *args, **kwargs)
    with patch.object(Path, "unlink", fail):
        report = workflow.finish(True)
    assert report["status"] == "completed" and path.exists()
    assert report["roots"][0]["cleanup_errors"]


def test_child_stage_does_not_finish_wrapper_and_resume_does_not_finish_parent(tmp_path):
    args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
    with workflow_context(args) as parent:
        child_args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
        with training_context(child_args) as session:
            path = claim(parent, tmp_path)
            assert session.id == parent.descriptor["id"]
        assert path.exists()
    assert not path.exists()
    with pytest.raises(RuntimeError):
        with workflow_context(args) as parent:
            path = claim(parent, tmp_path)
            saved = dict(dataset_cache_dir=str(tmp_path), cache_session=parent.descriptor)
            raise RuntimeError("interrupted bundle")
    with workflow_context(args, saved=saved) as resumed:
        claim(resumed, tmp_path)
        assert resumed.descriptor["id"] != parent.descriptor["id"]
    assert path.exists()


def test_direct_resume_reconnects_failed_session(tmp_path):
    args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
    with pytest.raises(RuntimeError):
        with training_context(args):
            owner = CacheWorkflow.attached(args.cache_session)
            path = claim(owner, tmp_path)
            owner.release()
            raise RuntimeError("interrupted")
    previous_id = args.cache_session["id"]
    args.resume_checkpoint = "checkpoint"
    with training_context(args):
        assert args.cache_session["id"] == previous_id
    assert not path.exists()


def test_native_process_death_retains_claims(tmp_path):
    code = """import os,sys
from pathlib import Path
from cache_lifetime import CacheWorkflow
w=CacheWorkflow(sys.argv[1])
s=w.session_for(sys.argv[1])
p=Path(sys.argv[1])/('a'*32)/('b'*32+'.pt')
p.parent.mkdir(parents=True,exist_ok=True)
with s.entry(p):
 p.write_bytes(b'graphs')
 p.with_suffix('.json').write_text('{}')
 s.recorded(p,built=True,size=6)
os._exit(3)
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=False)
    assert result.returncode == 3
    next_workflow = CacheWorkflow(tmp_path)
    path = claim(next_workflow, tmp_path)
    report = next_workflow.finish(True)
    assert path.exists()
    with cache_lock(tmp_path), database(tmp_path) as db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE status='abandoned'").fetchone()[0] == 1


def test_cross_process_reader_prevents_deletion(tmp_path):
    workflow = CacheWorkflow(tmp_path)
    path = claim(workflow, tmp_path)
    code = """import json,sys
from cache_lifetime import CacheWorkflow
w=CacheWorkflow.attached(json.loads(sys.argv[1]))
w.session_for(w.root)
print('ready',flush=True)
sys.stdin.readline()
w.release()
"""
    child = subprocess.Popen([sys.executable, "-c", code, json.dumps(workflow.descriptor)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        workflow.finish(True)
        assert path.exists()
    finally:
        child.communicate("release\n", timeout=15)
    collector = CacheWorkflow(tmp_path)
    collector.finish(True)
    assert not path.exists()
