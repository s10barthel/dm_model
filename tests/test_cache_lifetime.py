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


def test_distinct_entries_can_build_while_another_entry_is_locked(tmp_path):
    workflow = CacheWorkflow(tmp_path)
    session = workflow.session_for(tmp_path)
    path = tmp_path / ("a" * 32) / ("a" * 32 + ".pt")
    code = '''import sys
from pathlib import Path
from cache_lifetime import CacheWorkflow
root = Path(sys.argv[1])
w = CacheWorkflow(root)
s = w.session_for(root)
p = root / ('a' * 32) / ('b' * 32 + '.pt')
with s.entry(p):
 p.parent.mkdir(exist_ok=True)
 p.write_bytes(b'graphs')
 s.recorded(p, built=True, size=6)
w.finish(True)
'''
    with session.entry(path):
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
    workflow.finish(True)


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


def test_child_models_release_claims_before_wrapper_finishes(tmp_path):
    args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
    starts = []
    with workflow_context(args) as parent:
        previous_path = None
        for name in ("b", "c", "d"):
            child_args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
            with training_context(child_args) as session:
                assert session.id != parent.descriptor["id"]
                assert session.descriptor["parent"] == parent.descriptor["id"]
                wanted = tmp_path / ("a" * 32) / (name * 32 + ".pt")
                starts.append(session.start_model([wanted]))
                if previous_path:
                    assert not previous_path.exists()
                owner = CacheWorkflow.attached(session.descriptor)
                path = claim(owner, tmp_path, name)
                owner.release()
            assert path.exists()
            previous_path = path
    assert path.exists()
    assert [r["deleted_entries"] for r in starts] == [0, 1, 1]
    assert len(parent.report["model_starts"]) == 3


def test_direct_resume_uses_new_claims_without_requiring_original_success(tmp_path):
    args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
    with pytest.raises(RuntimeError):
        with training_context(args) as session:
            path = tmp_path / ("a" * 32) / ("b" * 32 + ".pt")
            session.start_model([path])
            owner = CacheWorkflow.attached(session.descriptor)
            claim(owner, tmp_path)
            owner.release()
            raise RuntimeError("interrupted")
    previous_id = args.cache_session["id"]
    args.resume_checkpoint = "checkpoint"
    with training_context(args) as session:
        assert args.cache_session["id"] != previous_id
        report = session.start_model([path])
        assert report["compatible_entries"] == 1
    assert path.exists()
    with training_context(args) as session:
        session.start_model([])  # A model using memory needs no prepared cache.
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


@pytest.mark.parametrize('success', [True, False])
def test_next_model_prunes_historical_workflows_regardless_of_status(tmp_path, success):
    old = CacheWorkflow(tmp_path, policy='model_start' if success else 'completion')
    path = claim(old, tmp_path)
    old.finish(success)
    args = SimpleNamespace(dataset_cache_dir=str(tmp_path), keep_cache=False)
    with training_context(args) as session:
        report = session.start_model([])
        assert report['deleted_entries'] == 1
        assert not path.exists()


@pytest.mark.parametrize('existing,keep', [(True, False), (False, True)])
def test_model_start_preserves_explicit_and_preexisting_retention(tmp_path, existing, keep):
    old = CacheWorkflow(tmp_path, keep=keep)
    path = claim(old, tmp_path, existing=existing)
    old.finish(False)
    with training_context(SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)) as session:
        report = session.start_model([])
        assert path.exists()
        assert report['retention_reasons']['pre_existing' if existing else 'keep_cache'] == 1


def test_evaluation_completion_cannot_sweep_training_retention(tmp_path):
    from cache_lifetime import evaluation_context
    args = SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)
    with training_context(args) as session:
        owner = CacheWorkflow.attached(session.descriptor)
        path = claim(owner, tmp_path)
        session.start_model([path])
        owner.release()
    with evaluation_context(args) as session:
        with session.entry(path):
            session.recorded(path, built=False, size=6)
    assert path.exists()
    with training_context(args) as session:
        session.start_model([])
    assert not path.exists()


def test_reserved_entry_is_protected_across_processes_before_reading(tmp_path):
    old = CacheWorkflow(tmp_path, policy='model_start')
    path = claim(old, tmp_path)
    old.finish(True)
    code = '''import sys
from pathlib import Path
from types import SimpleNamespace
from cache_lifetime import training_context
with training_context(SimpleNamespace(dataset_cache_dir=sys.argv[1],keep_cache=False)) as s:
 s.start_model([Path(sys.argv[2])])
 print('ready',flush=True)
 sys.stdin.readline()
'''
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path), str(path)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    args = SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)
    try:
        while child.stdout.readline().strip() != 'ready':
            assert child.poll() is None
        with training_context(args) as session:
            report = session.start_model([])
            assert report['retention_reasons']['active_reader'] == 1
            assert path.exists()
    finally:
        child.communicate('release\n', timeout=15)
    assert child.returncode == 0
    with training_context(args) as session:
        session.start_model([])
    assert not path.exists()


def test_killed_training_process_does_not_permanently_protect_entries(tmp_path):
    code = '''import os,sys
from pathlib import Path
from cache_lifetime import CacheWorkflow
w=CacheWorkflow(sys.argv[1],policy='model_start')
s=w.session_for(w.root)
p=Path(sys.argv[1])/('a'*32)/('b'*32+'.pt')
s.start_model([p])
with s.entry(p):
 p.parent.mkdir(exist_ok=True)
 p.write_bytes(b'graphs')
 s.recorded(p,built=True,size=6)
os._exit(3)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, timeout=15)
    assert result.returncode == 3
    with training_context(SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)) as session:
        report = session.start_model([])
        assert report['deleted_entries'] == 1
    assert not list(tmp_path.glob('*/*.pt'))


def test_additive_migration_does_not_prune_until_model_start(tmp_path):
    old = CacheWorkflow(tmp_path)
    path = claim(old, tmp_path)
    old.finish(False)
    with cache_lock(tmp_path), database(tmp_path) as db:
        db.execute('DROP TABLE session_policies')
        db.execute('DROP TABLE model_starts')
        db.execute('PRAGMA user_version=0')
    with cache_lock(tmp_path), database(tmp_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM entries').fetchone()[0] == 1
    assert path.exists()
    with training_context(SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)) as session:
        assert path.exists()
        session.start_model([])
    assert not path.exists()


def test_partial_startup_deletion_is_retried_by_next_model(tmp_path):
    old = CacheWorkflow(tmp_path)
    path = claim(old, tmp_path)
    old.finish(False)
    original = Path.unlink
    def fail(self, *args, **kwargs):
        if self == path:
            raise PermissionError('busy')
        return original(self, *args, **kwargs)
    args = SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)
    with training_context(args) as session:
        with patch.object(Path, 'unlink', fail):
            report = session.start_model([])
        assert report['cleanup_errors'] and report['deleted_bytes'] == 0
        assert path.exists() and not path.with_suffix('.json').exists()
    with training_context(args) as session:
        report = session.start_model([])
        assert not report['cleanup_errors'] and not path.exists()


def test_wrapper_aggregates_model_starts_from_another_cache_root(tmp_path):
    args = SimpleNamespace(dataset_cache_dir=tmp_path / 'wrapper_cache', keep_cache=False)
    with workflow_context(args) as wrapper:
        child = SimpleNamespace(dataset_cache_dir=tmp_path / 'model_cache', keep_cache=False)
        with training_context(child) as session:
            session.start_model([])
    assert len(wrapper.report['model_starts']) == 1
    assert wrapper.report['model_starts'][0]['root'] == str(child.dataset_cache_dir.resolve())
    assert wrapper.report['model_start_totals']['deleted_bytes'] == 0


def test_startup_protects_an_active_evaluator_then_reclaims_after_it_exits(tmp_path):
    from cache_lifetime import evaluation_context
    args = SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)
    with pytest.raises(RuntimeError):
        with evaluation_context(args) as evaluator:
            owner = CacheWorkflow.attached(evaluator.descriptor)
            path = claim(owner, tmp_path)
            with training_context(args) as model:
                report = model.start_model([])
                assert report['retention_reasons']['active_reader'] == 1
                assert path.exists()
            owner.release()
            raise RuntimeError('evaluation failed')
    with training_context(args) as model:
        model.start_model([])
    assert not path.exists()


def test_untracked_payloads_are_reported_and_never_adopted_or_deleted(tmp_path):
    path = tmp_path / ('a' * 32) / ('b' * 32 + '.pt')
    path.parent.mkdir()
    path.write_bytes(b'legacy')
    with training_context(SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)) as session:
        report = session.start_model([])
        assert report['retention_bytes']['pre_existing'] == 6
    assert path.read_bytes() == b'legacy'
    with cache_lock(tmp_path), database(tmp_path) as db:
        assert db.execute('SELECT count(*) FROM entries').fetchone()[0] == 0


def test_alias_to_another_entry_cannot_bypass_its_live_claim(tmp_path):
    old = CacheWorkflow(tmp_path)
    alias = claim(old, tmp_path, 'b')
    old.finish(False)
    active = CacheWorkflow(tmp_path, policy='model_start')
    protected = claim(active, tmp_path, 'c')
    alias.unlink()
    try:
        alias.symlink_to(protected)
    except OSError:
        active.finish(True)
        pytest.skip('File symlinks unavailable')
    with training_context(SimpleNamespace(dataset_cache_dir=tmp_path, keep_cache=False)) as model:
        report = model.start_model([])
        assert report['cleanup_errors']
        assert protected.read_bytes() == b'graphs'
    active.finish(True)
