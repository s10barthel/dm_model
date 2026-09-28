"""Optional Windows ProcDump supervision; also a lightweight child bootstrap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

DEFAULT_PROCDUMP_PATH = r'C:\Tools\ProcDump\procdump64.exe'


def add_crash_capture_arguments(parser):
    parser.add_argument('--crash-dump', choices=('on', 'off'), default='off')
    parser.add_argument('--procdump-path', default=None,
                        help=f'Override ProcDump location (default lookup: {DEFAULT_PROCDUMP_PATH}, then PATH).')


def resolve_procdump(path=None):
    if sys.platform != 'win32':
        raise RuntimeError('--crash-dump on is supported only on Windows.')
    if path:
        executable = str(Path(path).resolve())
    elif Path(DEFAULT_PROCDUMP_PATH).is_file():
        executable = DEFAULT_PROCDUMP_PATH
    else:
        executable = shutil.which('procdump64.exe') or shutil.which('procdump.exe')
    if not executable or not Path(executable).is_file():
        raise RuntimeError('ProcDump is unavailable. Install Microsoft ProcDump and use --procdump-path or PATH.')
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Sysinternals\ProcDump') as key:
            accepted = winreg.QueryValueEx(key, 'EulaAccepted')[0] == 1
    except OSError:
        accepted = False
    if not accepted:
        raise RuntimeError('Open ProcDump manually and accept its license before enabling crash capture.')
    return executable


def _value(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def _run_directory(command, root):
    checkpoint = _value(command, '--resume-checkpoint')
    if checkpoint:
        return Path(checkpoint).resolve().parent
    from project_config import get_model_run_root
    task, run_id = _value(command, '--task'), _value(command, '--run-id')
    if not task or not run_id:
        raise ValueError('Crash capture requires an individual training run ID.')
    directory = get_model_run_root(task, run_id)
    stage = _value(command, '--artifact-stage')
    return directory / stage if stage else directory


def _write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_training(command, *, cwd, options):
    if getattr(options, 'crash_dump', 'off') == 'off':
        return subprocess.run(command, cwd=cwd, check=True)
    executable = resolve_procdump(getattr(options, 'procdump_path', None))
    directory = _run_directory(command, cwd)
    metadata_path = directory / 'metadata.json'
    previous = json.loads(metadata_path.read_text(encoding='utf-8')).get('crash_capture_attempts', []) if metadata_path.exists() else []
    attempt = directory / 'crash_dumps' / (datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    record = {'enabled': True, 'tool': executable, 'directory': str(attempt), 'status': 'starting',
              'started': datetime.now(timezone.utc).isoformat(), 'dumps': []}
    child = collector = None
    try:
        with (attempt / 'procdump.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen([command[0], str(Path(__file__).resolve()), '--child', str(attempt),
                                      *command[1:]], cwd=cwd)
            record['launcher_pid'] = child.pid
            deadline = time.monotonic() + 30
            pid_path = attempt / 'interpreter.json'
            while not pid_path.exists():
                if child.poll() is not None:
                    raise RuntimeError('Training bootstrap exited before reporting its interpreter PID.')
                if time.monotonic() >= deadline:
                    raise RuntimeError('Training bootstrap PID handshake timed out.')
                time.sleep(.05)
            interpreter_pid = json.loads(pid_path.read_text(encoding='utf-8'))['pid']
            if not isinstance(interpreter_pid, int) or isinstance(interpreter_pid, bool) or interpreter_pid <= 0:
                raise RuntimeError('Training bootstrap reported an invalid interpreter PID.')
            record['pid'] = interpreter_pid
            collector = subprocess.Popen([executable, '-mm', '-e', '-n', '1', str(interpreter_pid), str(attempt)],
                                         cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while not (attempt / 'attached').exists():
                if child.poll() is not None or collector.poll() is not None:
                    raise RuntimeError(f'Crash capture failed to attach; see {attempt / "procdump.log"}.')
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'ProcDump attachment timed out; see {attempt / "procdump.log"}.')
                time.sleep(.05)
            if collector.poll() is not None:
                raise RuntimeError('ProcDump exited before training started.')
            record['status'] = 'attached'
            _write_json(attempt / 'capture.json', record)
            (attempt / 'release').touch()
            returncode = child.wait()
            record['returncode'] = returncode
            try:
                collector.wait(timeout=60 if returncode else 5)
                record['status'] = 'completed' if returncode == 0 else 'no_dump'
            except subprocess.TimeoutExpired:
                record['status'] = 'capture_incomplete'
                print(f'Crash capture did not finish in time; inspect {attempt}.', flush=True)
            record['dumps'] = [str(p) for p in attempt.glob('*.dmp') if p.stat().st_size > 0]
            if record['dumps'] and record['status'] != 'capture_incomplete':
                record['status'] = 'captured'
            if returncode:
                print(f"Crash capture: {record['status']}; artifacts: {attempt}", flush=True)
                raise subprocess.CalledProcessError(returncode, command)
            return subprocess.CompletedProcess(command, returncode)
    except BaseException as exc:
        if record['status'] in {'starting', 'attached'}:
            record['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['error'] = str(exc)
        raise
    finally:
        (attempt / 'abort').touch()
        # Stop the child first: terminating a debugger may also terminate its target.
        for process in (child, collector):
            try:
                _stop(process)
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f'Crash capture cleanup failed: {exc}', flush=True)
        try:
            _write_json(attempt / 'capture.json', record)
            metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
            attempts = {r['directory']: r for r in previous + metadata.get('crash_capture_attempts', [])}
            attempts[str(attempt)] = record
            metadata['crash_capture_attempts'] = list(attempts.values())
            _write_json(metadata_path, metadata)
        except (OSError, ValueError, TypeError) as exc:
            print(f'Could not update run metadata with crash capture details: {exc}', flush=True)


def child_bootstrap(attempt, command):
    import ctypes
    import runpy
    attempt = Path(attempt)
    # Windows venv executables can be launcher processes. Only the interpreter
    # executing this bootstrap can report the PID that ProcDump must debug.
    _write_json(attempt / 'interpreter.json', {'pid': os.getpid()})
    deadline = time.monotonic() + 40
    while not (attempt / 'release').exists():
        if (attempt / 'abort').exists():
            return
        if ctypes.windll.kernel32.IsDebuggerPresent():
            (attempt / 'attached').touch()
        if time.monotonic() >= deadline:
            raise RuntimeError('Training startup handshake timed out.')
        time.sleep(.05)
    sys.argv = command
    sys.path.insert(0, str(Path(command[0]).resolve().parent))
    runpy.run_path(command[0], run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[1] != '--child':
        raise SystemExit('This module is launched by the training wrapper.')
    child_bootstrap(sys.argv[2], sys.argv[3:])
