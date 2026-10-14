"""Workflow ownership of prepared caches; OS locks and SQLite survive process death.

A session claims files before preparing/reading them. Claims last for the whole
workflow, including idle time between child stages. Untracked files are retained.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import uuid

import psutil

ENVIRONMENT_KEY = "DM_MODEL_DATASET_CACHE_SESSION"


def process_identity(pid=None):
    process = psutil.Process(os.getpid() if pid is None else pid)
    return process.pid, process.create_time()


def process_alive(pid, created):
    try:
        return abs(psutil.Process(pid).create_time() - created) < .001
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Uncertainty must never authorize deletion.


@contextmanager
def cache_lock(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.ownership.lock').open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            import msvcrt

            class Overlapped(ctypes.Structure):
                _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                            ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD),
                            ('hEvent', wintypes.HANDLE)]

            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                         wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
            kernel.LockFileEx.restype = wintypes.BOOL
            kernel.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                           wintypes.DWORD, ctypes.POINTER(Overlapped)]
            kernel.UnlockFileEx.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(stream.fileno())
            overlap = Overlapped()
            # Blocking Windows locks wake a waiting workflow between entries;
            # polling can repeatedly miss that short gap and starve consumers.
            if not kernel.LockFileEx(handle, 2, 0, 1, 0, ctypes.byref(overlap)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield
            finally:
                if not kernel.UnlockFileEx(handle, 0, 1, 0, ctypes.byref(overlap)):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def database(root):
    db = sqlite3.connect(Path(root) / '.ownership.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, pid INTEGER, created REAL, kind TEXT,
            status TEXT, keep INTEGER, error TEXT);
        CREATE TABLE IF NOT EXISTS entries (
            path TEXT PRIMARY KEY, managed INTEGER, retained INTEGER, bytes INTEGER);
        CREATE TABLE IF NOT EXISTS uses (
            session TEXT, path TEXT, built INTEGER DEFAULT 0, reused INTEGER DEFAULT 0,
            PRIMARY KEY(session, path));
        CREATE TABLE IF NOT EXISTS readers (
            session TEXT, pid INTEGER, created REAL, PRIMARY KEY(session, pid, created));
        CREATE TABLE IF NOT EXISTS roots (
            session TEXT, root TEXT, PRIMARY KEY(session, root));
    ''')
    try:
        with db:
            yield db
    finally:
        db.close()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CacheWorkflow:
    def __init__(self, root, *, keep=False, kind='direct', descriptor=None):
        root = Path(root).resolve()
        pid, created = process_identity()
        self.descriptor = descriptor or dict(id=uuid.uuid4().hex, root=str(root),
                                             pid=pid, created=created, kind=kind, keep=bool(keep))
        self.descriptor = dict(self.descriptor)
        self.descriptor.update(pid=pid, created=created, keep=bool(keep or self.descriptor.get('keep')))
        self.root = Path(self.descriptor['root']).resolve()
        self.sessions = {}
        self.finished = False
        self.report = None
        self.session_for(self.root)

    @classmethod
    def attached(cls, descriptor):
        instance = cls.__new__(cls)
        instance.descriptor = dict(descriptor)
        instance.root = Path(descriptor['root']).resolve()
        instance.sessions = {}
        instance.finished = False
        instance.report = None
        return instance

    def session_for(self, root):
        root = Path(root).resolve()
        if root not in self.sessions:
            with cache_lock(self.root), database(self.root) as db:
                db.execute('INSERT OR IGNORE INTO roots VALUES (?,?)', (self.descriptor['id'], str(root)))
            self.sessions[root] = CacheSession(root, self.descriptor)
        return self.sessions[root]

    def finish(self, success):
        if self.finished:
            return self.report
        self.finished = True
        reports = []
        errors = []
        try:
            with cache_lock(self.root), database(self.root) as db:
                roots = [r['root'] for r in db.execute('SELECT root FROM roots WHERE session=?',
                                                     (self.descriptor['id'],))]
            for root in roots:
                try:
                    session = self.session_for(root)
                    reports.append(session.finish(success))
                except Exception as exc:
                    errors.append(f'{root}: {exc}')
        except Exception as exc:
            errors.append(str(exc))
        self.report = dict(session=self.descriptor, status='completed' if success else 'failed',
                           roots=reports, cleanup_errors=errors)
        try:
            atomic_json(self.root / '.sessions' / (self.descriptor['id'] + '.json'), self.report)
        except OSError as exc:
            errors.append(str(exc))
        if errors:
            print('Prepared cache cleanup incomplete (training artifacts preserved): ' + '; '.join(errors), flush=True)
        return self.report

    def release(self):
        for session in self.sessions.values():
            session.release()


class CacheSession:
    def __init__(self, root, descriptor):
        self.root, self.descriptor = Path(root).resolve(), dict(descriptor)
        self.id = descriptor['id']
        self.pid, self.created = process_identity()
        with cache_lock(self.root), database(self.root) as db:
            db.execute('''INSERT INTO sessions VALUES (?,?,?,?,?,?,NULL)
                ON CONFLICT(id) DO UPDATE SET pid=excluded.pid, created=excluded.created,
                keep=MAX(sessions.keep, excluded.keep), status='running', error=NULL''',
                (self.id, descriptor['pid'], descriptor['created'], descriptor['kind'],
                 'running', int(descriptor.get('keep', False))))
            db.execute('INSERT OR IGNORE INTO readers VALUES (?,?,?)', (self.id, self.pid, self.created))
        atexit.register(self.release)

    def checked_path(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(self.root)
        # Only the two files in a prepared match entry may be removed.
        if len(relative.parts) != 2 or len(relative.parts[0]) != 32 or path.suffix != '.pt':
            raise ValueError(f'Invalid prepared cache entry: {path}')
        if any(c not in '0123456789abcdef' for c in relative.parts[0] + path.stem) or len(path.stem) != 32:
            raise ValueError(f'Invalid prepared cache entry: {path}')
        return path, relative.as_posix()

    @contextmanager
    def entry(self, path):
        """Serialize validation/build/publication and durably claim before any use."""
        path, key = self.checked_path(path)
        with cache_lock(self.root):
            with database(self.root) as db:
                existed = path.exists() or path.with_suffix('.json').exists()
                db.execute('INSERT OR IGNORE INTO entries VALUES (?,?,?,0)',
                           (key, int(not existed), int(existed)))
                if self.descriptor.get('keep'):
                    db.execute('UPDATE entries SET retained=1 WHERE path=?', (key,))
                db.execute('INSERT OR IGNORE INTO uses(session,path) VALUES (?,?)', (self.id, key))
            yield

    def recorded(self, path, *, built, size):
        # Caller already holds entry()'s root lock.
        _, key = self.checked_path(path)
        with database(self.root) as db:
            db.execute('UPDATE entries SET bytes=? WHERE path=?', (size, key))
            db.execute('UPDATE uses SET built=MAX(built,?), reused=MAX(reused,?) WHERE session=? AND path=?',
                       (int(built), int(not built), self.id, key))

    def release(self):
        try:
            with cache_lock(self.root), database(self.root) as db:
                db.execute('DELETE FROM readers WHERE session=? AND pid=? AND created=?',
                           (self.id, self.pid, self.created))
        except (OSError, sqlite3.Error):
            pass

    def finish(self, success):
        summary = dict(root=str(self.root), created_entries=0, reused_entries=0,
                       deleted_bytes=0, deleted_entries=0, retained_bytes=0,
                       retention_reasons={}, cleanup_errors=[])
        with cache_lock(self.root), database(self.root) as db:
            db.execute('UPDATE sessions SET status=? WHERE id=?', ('completed' if success else 'failed', self.id))
            db.execute('DELETE FROM readers WHERE session=? AND pid=? AND created=?',
                       (self.id, self.pid, self.created))
            for row in db.execute("SELECT * FROM sessions WHERE status='running'").fetchall():
                if not process_alive(row['pid'], row['created']):
                    db.execute("UPDATE sessions SET status='abandoned' WHERE id=?", (row['id'],))
            readers = set()
            for row in db.execute('SELECT * FROM readers').fetchall():
                if process_alive(row['pid'], row['created']):
                    readers.add(row['session'])
                else:
                    db.execute('DELETE FROM readers WHERE session=? AND pid=? AND created=?',
                               (row['session'], row['pid'], row['created']))
            counts = db.execute('SELECT SUM(built),SUM(reused) FROM uses WHERE session=?', (self.id,)).fetchone()
            summary.update(created_entries=counts[0] or 0, reused_entries=counts[1] or 0)
            # Also collect entries deferred by a successful earlier workflow.
            for entry in db.execute('SELECT * FROM entries').fetchall():
                users = db.execute('SELECT s.* FROM uses u JOIN sessions s ON s.id=u.session WHERE u.path=?',
                                   (entry['path'],)).fetchall()
                reason = ('pre_existing' if not entry['managed'] else
                          'keep_cache' if entry['retained'] or any(u['keep'] for u in users) else
                          'active_reader' if any(u['id'] in readers for u in users) else
                          'unfinished_or_failed_workflow' if any(u['status'] != 'completed' for u in users) else None)
                path, _ = self.checked_path(self.root / entry['path'])
                if reason:
                    if path.exists():
                        summary['retained_bytes'] += path.stat().st_size
                        summary['retention_reasons'][reason] = summary['retention_reasons'].get(reason, 0) + 1
                    continue
                try:
                    # Resolve again under the lock, rejecting links that escape root.
                    path.with_suffix('.json').resolve().relative_to(self.root)
                    size = path.stat().st_size if path.exists() else 0
                    path.with_suffix('.json').unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
                    summary['deleted_bytes'] += size
                    summary['deleted_entries'] += 1
                    db.execute('DELETE FROM uses WHERE path=?', (entry['path'],))
                    db.execute('DELETE FROM entries WHERE path=?', (entry['path'],))
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
                except (OSError, ValueError) as exc:
                    summary['cleanup_errors'].append(f'{path}: {exc}')
        if summary['cleanup_errors']:
            print('Prepared cache cleanup incomplete: ' + '; '.join(summary['cleanup_errors']), flush=True)
        return summary


def record_report(directory, report):
    if directory is None:
        return
    path = Path(directory) / 'metadata.json'
    if path.is_file():
        try:
            metadata = json.loads(path.read_text(encoding='utf-8'))
            metadata['cache_lifetime'] = report
            atomic_json(path, metadata)
        except (OSError, ValueError) as exc:
            print(f'Could not record prepared cache cleanup in {path}: {exc}', flush=True)


def workflow_running(descriptor):
    if not process_alive(descriptor['pid'], descriptor['created']):
        return False
    with cache_lock(descriptor['root']), database(descriptor['root']) as db:
        row = db.execute('SELECT status FROM sessions WHERE id=?', (descriptor['id'],)).fetchone()
        return row is not None and row['status'] == 'running'


@contextmanager
def workflow_context(args, *, kind='wrapper', saved=None):
    """Wrapper scope; environment propagates the same workflow to child stages."""
    from dataset_loading import DEFAULT_CACHE_DIR
    saved = saved or {}
    root = saved.get('dataset_cache_dir') or getattr(args, 'dataset_cache_dir', DEFAULT_CACHE_DIR)
    previous = saved.get('cache_session')
    # Resuming a single stage must not complete its original wrapper workflow.
    reconnect = previous if previous and previous['kind'] == 'direct' else None
    if reconnect and workflow_running(reconnect):
        raise RuntimeError('The original cache workflow is still running.')
    workflow = CacheWorkflow(root, keep=getattr(args, 'keep_cache', False) or saved.get('keep_cache', False),
                             kind=kind, descriptor=reconnect)
    before = os.environ.get(ENVIRONMENT_KEY)
    os.environ[ENVIRONMENT_KEY] = json.dumps(workflow.descriptor)
    success = False
    try:
        yield workflow
        success = True
    finally:
        report = workflow.finish(success)
        record_report(getattr(args, '_cache_metadata_directory', None), report)
        if before is None:
            os.environ.pop(ENVIRONMENT_KEY, None)
        else:
            os.environ[ENVIRONMENT_KEY] = before


@contextmanager
def training_context(args):
    from dataset_loading import DEFAULT_CACHE_DIR
    inherited = os.environ.get(ENVIRONMENT_KEY)
    if inherited:
        descriptor = json.loads(inherited)
        descriptor['keep'] = descriptor.get('keep', False) or getattr(args, 'keep_cache', False)
        workflow = CacheWorkflow.attached(descriptor)
        owner = False
    else:
        previous = getattr(args, 'cache_session', None) if getattr(args, 'resume_checkpoint', None) else None
        reconnect = previous if previous and previous['kind'] == 'direct' else None
        if reconnect and workflow_running(reconnect):
            raise RuntimeError('The original cache workflow is still running.')
        workflow = CacheWorkflow(getattr(args, 'dataset_cache_dir', DEFAULT_CACHE_DIR),
                                 keep=getattr(args, 'keep_cache', False), descriptor=reconnect)
        owner = True
    args.cache_session = workflow.descriptor
    session = workflow.session_for(getattr(args, 'dataset_cache_dir', DEFAULT_CACHE_DIR))
    success = False
    try:
        yield session
        success = True
    finally:
        monitor = getattr(args, '_monitor', None)
        if monitor is not None:
            monitor.close()
        if owner:
            report = workflow.finish(success)
            record_report(getattr(args, '_cache_metadata_directory', None), report)
            if success and getattr(args, 'artifact_stage', None):
                from training_state import publish_stage_artifacts
                publish_stage_artifacts(args._cache_metadata_directory)
        else:
            workflow.release()
