"""Workflow ownership of prepared caches; OS locks and SQLite survive process death.

Training and evaluation claims last for one model, not its wrapper. Files remain
until a later model prunes obsolete entries in the same cache root.
"""
from __future__ import annotations

import atexit
from contextlib import closing, contextmanager
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
    with file_lock(root / '.ownership.lock'):
        yield


@contextmanager
def file_lock(path):
    """An OS-owned lock, released even if a cache builder dies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
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
        CREATE TABLE IF NOT EXISTS session_policies (
            session TEXT PRIMARY KEY, policy TEXT, parent TEXT);
        CREATE TABLE IF NOT EXISTS model_starts (
            session TEXT PRIMARY KEY, report TEXT);
        CREATE INDEX IF NOT EXISTS uses_path ON uses(path);
    ''')
    if db.execute('PRAGMA user_version').fetchone()[0] < 1:
        db.execute('PRAGMA user_version=1')
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
    def __init__(self, root, *, keep=False, kind='direct', descriptor=None,
                 policy=None, parent=None, purpose=None):
        root = Path(root).resolve()
        pid, created = process_identity()
        self.descriptor = descriptor or dict(id=uuid.uuid4().hex, root=str(root),
                                             pid=pid, created=created, kind=kind, keep=bool(keep))
        self.descriptor = dict(self.descriptor)
        self.descriptor.update(pid=pid, created=created, keep=bool(keep or self.descriptor.get('keep')))
        if policy is not None:
            self.descriptor['policy'] = policy
        if parent is not None:
            self.descriptor['parent'] = parent
        if purpose is not None:
            self.descriptor['purpose'] = purpose
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
        model_starts = []
        errors = []
        try:
            with cache_lock(self.root), database(self.root) as db:
                roots = [r['root'] for r in db.execute('SELECT root FROM roots WHERE session=?',
                                                     (self.descriptor['id'],))]
            for root in roots:
                try:
                    session = self.session_for(root)
                    reports.append(session.finish(success))
                    if self.descriptor.get('kind') in {'wrapper', 'evaluation_wrapper'}:
                        with cache_lock(root), database(root) as db:
                            model_starts.extend(json.loads(row['report']) for row in db.execute(
                                'SELECT m.report FROM model_starts m JOIN session_policies p ON p.session=m.session '
                                'WHERE p.parent=? ORDER BY m.rowid', (self.descriptor['id'],)))
                except Exception as exc:
                    errors.append(f'{root}: {exc}')
        except Exception as exc:
            errors.append(str(exc))
        self.report = dict(session=self.descriptor, status='completed' if success else 'failed',
                           roots=reports, cleanup_errors=errors)
        if self.descriptor.get('kind') in {'wrapper', 'evaluation_wrapper'}:
            self.report['model_starts'] = model_starts
            self.report['model_start_totals'] = {
                key: sum(report[key] for report in model_starts)
                for key in ('deleted_entries', 'deleted_bytes')}
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
            db.execute('INSERT OR REPLACE INTO session_policies VALUES (?,?,?)',
                       (self.id, descriptor.get('policy', 'completion'), descriptor.get('parent')))
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

    def entry_lock(self, path):
        return file_lock(self.root / '.entry_locks' / (path.parent.name + path.stem + '.lock'))

    @contextmanager
    def entry(self, path):
        """Serialize validation/build/publication and durably claim before any use."""
        path, key = self.checked_path(path)
        # Different entries may build concurrently. Register the durable claim
        # under the root lock before releasing it; cleanup sees this live reader.
        with self.entry_lock(path):
            with cache_lock(self.root), database(self.root) as db:
                existed = path.exists() or path.with_suffix('.json').exists()
                db.execute('INSERT OR IGNORE INTO entries VALUES (?,?,?,0)',
                           (key, int(not existed), int(existed)))
                if self.descriptor.get('keep'):
                    db.execute('UPDATE entries SET retained=1 WHERE path=?', (key,))
                db.execute('INSERT OR IGNORE INTO uses(session,path) VALUES (?,?)', (self.id, key))
            yield

    def recorded(self, path, *, built, size):
        # Caller holds the per-entry lock, not the root bookkeeping lock.
        _, key = self.checked_path(path)
        with cache_lock(self.root), database(self.root) as db:
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

    def start_model(self, paths, *, preparation_ids=()):
        """Reserve both splits before pruning, without opening any graph payload."""
        if self.descriptor.get('policy') != 'model_start':
            raise ValueError('Model-start pruning requires a model session.')
        desired = {self.checked_path(path)[1] for path in paths}
        with cache_lock(self.root), database(self.root) as db:
            if db.execute('SELECT 1 FROM model_starts WHERE session=?', (self.id,)).fetchone():
                raise RuntimeError('Model cache requirements have already been declared.')
            db.executemany('INSERT OR IGNORE INTO uses(session,path) VALUES (?,?)',
                           [(self.id, key) for key in desired])
        summary = self.collect(trigger='model_start', desired=desired)
        summary.update(model_use_id=self.id, purpose=self.descriptor.get('purpose'),
                       preparation_ids=sorted(set(preparation_ids)),
                       compatible_entries=summary['retention_reasons'].get('current_model', 0))
        with cache_lock(self.root), database(self.root) as db:
            db.execute('INSERT INTO model_starts VALUES (?,?)', (self.id, json.dumps(summary)))
        print(f"Prepared {self.descriptor.get('purpose', 'model')} cache at {self.root}: reclaimed {summary['deleted_entries']} entries "
              f"({summary['deleted_bytes'] / 1024**3:.2f} GiB); "
              f"{summary['compatible_entries']} compatible entries available; "
              f"retained {summary['retained_bytes'] / 1024**3:.2f} GiB "
              f"{summary['retention_reasons']}.", flush=True)
        return summary

    @staticmethod
    def live_readers(db, processes=None):
        processes = {} if processes is None else processes
        def alive(pid, created):
            key = (pid, created)
            if key not in processes:
                processes[key] = process_alive(pid, created)
            return processes[key]
        for row in db.execute("SELECT * FROM sessions WHERE status='running'").fetchall():
            if not alive(row['pid'], row['created']):
                db.execute("UPDATE sessions SET status='abandoned' WHERE id=?", (row['id'],))
        readers = set()
        for row in db.execute('SELECT * FROM readers').fetchall():
            if alive(row['pid'], row['created']):
                readers.add(row['session'])
            else:
                db.execute('DELETE FROM readers WHERE session=? AND pid=? AND created=?',
                           (row['session'], row['pid'], row['created']))
        return readers

    def retention_reason(self, db, entry, readers, trigger, desired):
        users = db.execute('SELECT s.*,p.policy FROM uses u JOIN sessions s ON s.id=u.session '
                           'LEFT JOIN session_policies p ON p.session=s.id WHERE u.path=?', (entry['path'],)).fetchall()
        reason = ('current_model' if entry['path'] in desired else
                  'pre_existing' if not entry['managed'] else
                  'keep_cache' if entry['retained'] or any(u['keep'] for u in users) else
                  'active_reader' if any(u['id'] in readers for u in users) else None)
        if reason is None and trigger != 'model_start':
            reason = ('awaiting_next_model' if self.descriptor.get('policy') == 'model_start'
                      or any(u['policy'] == 'model_start' for u in users) else
                      'unfinished_or_failed_workflow' if any(u['status'] != 'completed' for u in users)
                      else None)
        return reason

    @staticmethod
    def retain(summary, path, reason):
        if path.exists():
            size = path.stat().st_size
            summary['retained_bytes'] += size
            summary['retention_reasons'][reason] = summary['retention_reasons'].get(reason, 0) + 1
            summary['retention_bytes'][reason] = summary['retention_bytes'].get(reason, 0) + size

    def collect(self, *, trigger, desired=()):
        summary = dict(root=str(self.root), trigger=trigger, created_entries=0, reused_entries=0,
                       deleted_bytes=0, deleted_entries=0, retained_bytes=0,
                       retention_reasons={}, retention_bytes={}, cleanup_errors=[])
        processes = {}
        with cache_lock(self.root), database(self.root) as db:
            self.live_readers(db, processes)
            keys = [r['path'] for r in db.execute('SELECT path FROM entries')]
            counts = db.execute('SELECT SUM(built),SUM(reused) FROM uses WHERE session=?', (self.id,)).fetchone()
            summary.update(created_entries=counts[0] or 0, reused_entries=counts[1] or 0)
        for key in keys:
            try:
                path, resolved_key = self.checked_path(self.root / key)
                if resolved_key != key:
                    raise ValueError('Tracked cache path resolves to another entry')
                # Protected readers/builders need not finish before we can report.
                with cache_lock(self.root), database(self.root) as db:
                    entry = db.execute('SELECT * FROM entries WHERE path=?', (key,)).fetchone()
                    if entry is None:
                        continue
                    reason = self.retention_reason(db, entry, self.live_readers(db, processes), trigger, desired)
                    if reason:
                        self.retain(summary, path, reason)
                        continue
                # Never wait for an entry lock while holding the root lock.
                with self.entry_lock(path), cache_lock(self.root), database(self.root) as db:
                    path, resolved_key = self.checked_path(self.root / key)
                    if resolved_key != key:
                        raise ValueError('Tracked cache path resolves to another entry')
                    entry = db.execute('SELECT * FROM entries WHERE path=?', (key,)).fetchone()
                    if entry is None:
                        continue
                    readers = self.live_readers(db, processes)
                    reason = self.retention_reason(db, entry, readers, trigger, desired)
                    if reason:
                        self.retain(summary, path, reason)
                        continue
                    size = path.stat().st_size if path.exists() else 0
                    path.with_suffix('.json').resolve().relative_to(self.root)
                    path.with_suffix('.json').unlink(missing_ok=True)
                    path.unlink(missing_ok=True)
                    summary['deleted_bytes'] += size
                    summary['deleted_entries'] += 1
                    db.execute('DELETE FROM uses WHERE path=?', (key,))
                    db.execute('DELETE FROM entries WHERE path=?', (key,))
                    try:
                        path.parent.rmdir()
                    except OSError:
                        pass
            except (OSError, ValueError) as exc:
                summary['cleanup_errors'].append(f'{key}: {exc}')
        if summary['cleanup_errors']:
            print('Prepared cache cleanup incomplete: ' + '; '.join(summary['cleanup_errors']), flush=True)
        # Files predating ownership tracking remain untouched and visible in reports.
        with cache_lock(self.root), database(self.root) as db:
            tracked = {row['path'] for row in db.execute('SELECT path FROM entries')}
            for path in self.root.glob('*/*.pt'):
                try:
                    path, key = self.checked_path(path)
                    if key not in tracked:
                        self.retain(summary, path, 'pre_existing')
                except (OSError, ValueError):
                    pass  # Unrecognized or escaping paths are never adopted/deleted.
        return summary

    def finish(self, success):
        with cache_lock(self.root), database(self.root) as db:
            db.execute('UPDATE sessions SET status=? WHERE id=?', ('completed' if success else 'failed', self.id))
            db.execute('DELETE FROM readers WHERE session=? AND pid=? AND created=?',
                       (self.id, self.pid, self.created))
            startup = db.execute('SELECT report FROM model_starts WHERE session=?', (self.id,)).fetchone()
        summary = self.collect(trigger='model_exit' if self.descriptor.get('policy') == 'model_start' else 'completion')
        if startup:
            summary['model_start'] = json.loads(startup['report'])
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
    # Saved descriptors may point to an old root. Liveness checks must not
    # migrate it or create lock/database files there.
    try:
        uri = (Path(descriptor['root']).resolve() / '.ownership.sqlite').as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as db:
            row = db.execute('SELECT status FROM sessions WHERE id=?', (descriptor['id'],)).fetchone()
            return row is not None and row[0] == 'running'
    except (OSError, sqlite3.Error):
        return True  # A live owner with inaccessible records remains protected.


@contextmanager
def workflow_context(args, *, kind='wrapper', saved=None):
    """Wrapper scope; environment propagates the same workflow to child stages."""
    from dataset_loading import configure_dataset_cache
    purpose = 'evaluation' if kind == 'evaluation_wrapper' else 'training'
    root = configure_dataset_cache(args, purpose, saved=saved)
    saved = saved or {}
    previous = saved.get('cache_session')
    if previous and previous['kind'] == 'direct' and workflow_running(previous):
        raise RuntimeError('The original cache workflow is still running.')
    workflow = CacheWorkflow(root, keep=getattr(args, 'keep_cache', False) or saved.get('keep_cache', False),
                             kind=kind, policy='model_start', purpose=purpose)
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


def model_workflow(args, purpose):
    """Give each model fresh claims, inheriting only a matching wrapper's context."""
    from dataset_loading import configure_dataset_cache
    inherited = os.environ.get(ENVIRONMENT_KEY)
    parent = json.loads(inherited) if inherited else {}
    parent_purpose = parent.get('purpose') or (
        'evaluation' if parent.get('kind') in {'evaluation', 'evaluation_wrapper'} else 'training')
    if parent_purpose != purpose:
        parent = {}
    previous = getattr(args, 'cache_session', None) if getattr(args, 'resume_checkpoint', None) else None
    if previous and previous['kind'] == 'direct' and workflow_running(previous):
        raise RuntimeError('The original cache workflow is still running.')
    saved = dict(vars(args)) if purpose == 'training' and getattr(args, 'resume_checkpoint', None) else None
    root = configure_dataset_cache(args, purpose, saved=saved)
    workflow = CacheWorkflow(root, keep=getattr(args, 'keep_cache', False) or parent.get('keep', False),
                             kind='direct' if purpose == 'training' else 'evaluation',
                             policy='model_start', parent=parent.get('id'), purpose=purpose)
    if parent:
        with cache_lock(parent['root']), database(parent['root']) as db:
            db.execute('INSERT OR IGNORE INTO roots VALUES (?,?)', (parent['id'], str(workflow.root)))
    args.cache_session = workflow.descriptor
    return workflow


@contextmanager
def evaluation_context(args):
    workflow = model_workflow(args, 'evaluation')
    success = False
    try:
        yield workflow.session_for(args.dataset_cache_dir)
        success = True
    finally:
        record_report(getattr(args, 'evaluation_output_dir', None), workflow.finish(success))


@contextmanager
def training_context(args):
    workflow = model_workflow(args, 'training')
    session = workflow.session_for(args.dataset_cache_dir)
    success = False
    try:
        yield session
        success = True
    finally:
        monitor = getattr(args, '_monitor', None)
        if monitor is not None:
            monitor.close()
        report = workflow.finish(success)
        record_report(getattr(args, '_cache_metadata_directory', None), report)
        if success and getattr(args, 'artifact_stage', None):
            from training_state import publish_stage_artifacts
            publish_stage_artifacts(args._cache_metadata_directory)
