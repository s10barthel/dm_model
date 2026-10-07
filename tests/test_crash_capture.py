import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
from unittest.mock import Mock

import pytest
import crash_capture as capture


def test_resume_arguments():
    from scripts.train_relevant_models import parse_args
    args = parse_args(['--resume-id', 'pass_success/run', '--crash-dump', 'on', '--procdump-path', 'tool.exe'])
    assert args.crash_dump == 'on' and args.procdump_path == 'tool.exe' and args.monitoring == 'off'


@pytest.mark.parametrize('status', [0, 3221225477])
def test_exit_code_and_unique_attempts(tmp_path, monkeypatch, status):
    monkeypatch.setattr(capture, 'resolve_procdump', lambda _: 'tool.exe')
    monkeypatch.setattr(capture, '_run_directory', lambda *_: tmp_path)
    def spawn(command, **kwargs):
        process = Mock()
        process.pid = 123
        process.poll.return_value = None
        process.wait.return_value = status if '--child' in command else 0
        if '--child' in command:
            attempt = Path(command[command.index('--child') + 1])
            capture._write_json(attempt / 'interpreter.json', {'pid': 456})
            (attempt / 'attached').touch()
        else:
            assert command[1:5] == ['-mm', '-e', '-n', '1']
            assert command[5] == '456' and '-accepteula' not in command
            if status:
                (Path(command[-1]) / 'crash.dmp').write_bytes(b'dump')
        return process
    monkeypatch.setattr(capture.subprocess, 'Popen', spawn)
    for _ in range(2):
        if status:
            with pytest.raises(subprocess.CalledProcessError) as error:
                capture.run_training(['python', 'train.py'], cwd=tmp_path, options=SimpleNamespace(crash_dump='on'))
            assert error.value.returncode == status
        else:
            assert capture.run_training(['python', 'train.py'], cwd=tmp_path,
                                        options=SimpleNamespace(crash_dump='on')).returncode == 0
    records = json.loads((tmp_path / 'metadata.json').read_text())['crash_capture_attempts']
    assert len(records) == 2
    assert all(r['pid'] == 456 and r['launcher_pid'] == 123 for r in records)
    assert all(r['status'] == ('captured' if status else 'completed') for r in records)


@pytest.mark.parametrize('mode', ['exit', 'timeout', 'cancel'])
def test_attach_failure_cleanup(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(capture, 'resolve_procdump', lambda _: 'tool.exe')
    monkeypatch.setattr(capture, '_run_directory', lambda *_: tmp_path)
    child, collector = Mock(pid=123), Mock()
    child.poll.return_value = None
    collector.poll.return_value = 1 if mode == 'exit' else None
    processes = iter([child, collector])
    def spawn(command, **kwargs):
        if '--child' in command:
            capture._write_json(Path(command[command.index('--child') + 1]) / 'interpreter.json', {'pid': 456})
        return next(processes)
    monkeypatch.setattr(capture.subprocess, 'Popen', spawn)
    monkeypatch.setattr(capture.time, 'monotonic', Mock(side_effect=[0, 31]))
    if mode == 'cancel':
        child.poll.side_effect = KeyboardInterrupt
        # Interrupt only once so cleanup can inspect the child.
        child.poll.side_effect = [KeyboardInterrupt(), None]
    with pytest.raises(KeyboardInterrupt if mode == 'cancel' else RuntimeError):
        capture.run_training(['python', 'train.py'], cwd=tmp_path, options=SimpleNamespace(crash_dump='on'))
    assert child.terminate.called
    assert not list(tmp_path.glob('crash_dumps/*/release'))


def test_disabled_and_missing_tool(tmp_path, monkeypatch):
    run = Mock(return_value='normal')
    monkeypatch.setattr(capture.subprocess, 'run', run)
    assert capture.run_training(['python', 'train.py'], cwd=tmp_path, options=SimpleNamespace()) == 'normal'
    monkeypatch.setattr(capture.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='Windows'):
        capture.resolve_procdump()
    monkeypatch.setattr(capture.sys, 'platform', 'win32')
    monkeypatch.setattr(capture, 'DEFAULT_PROCDUMP_PATH', str(tmp_path / 'missing.exe'))
    monkeypatch.setattr(capture.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='unavailable'):
        capture.resolve_procdump()


def test_dump_timeout_preserves_native_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, 'resolve_procdump', lambda _: 'tool.exe')
    monkeypatch.setattr(capture, '_run_directory', lambda *_: tmp_path)
    child, collector = Mock(pid=123), Mock()
    child.poll.return_value = 3221225477
    child.wait.return_value = 3221225477
    collector.poll.return_value = None
    collector.wait.side_effect = [subprocess.TimeoutExpired('tool', 60), 0]
    def spawn(command, **kwargs):
        if '--child' in command:
            attempt = Path(command[command.index('--child') + 1])
            capture._write_json(attempt / 'interpreter.json', {'pid': 456})
            (attempt / 'attached').touch()
            return child
        return collector
    monkeypatch.setattr(capture.subprocess, 'Popen', spawn)
    with pytest.raises(subprocess.CalledProcessError) as error:
        capture.run_training(['python', 'train.py'], cwd=tmp_path, options=SimpleNamespace(crash_dump='on'))
    assert error.value.returncode == 3221225477
    record = json.loads(next(tmp_path.glob('crash_dumps/*/capture.json')).read_text())
    assert record['status'] == 'capture_incomplete'


def test_executable_precedence_and_license(tmp_path, monkeypatch):
    import sys
    exe = tmp_path / 'tool.exe'
    exe.touch()
    monkeypatch.setattr(capture.sys, 'platform', 'win32')
    key = Mock()
    key.__enter__ = Mock(return_value=key)
    key.__exit__ = Mock(return_value=False)
    registry = SimpleNamespace(HKEY_CURRENT_USER=1, OpenKey=Mock(return_value=key),
                               QueryValueEx=Mock(return_value=(1, 4)))
    monkeypatch.setitem(sys.modules, 'winreg', registry)
    search = Mock(return_value=str(exe))
    monkeypatch.setattr(capture.shutil, 'which', search)
    monkeypatch.setattr(capture, 'DEFAULT_PROCDUMP_PATH', str(exe))
    assert capture.resolve_procdump(str(exe)) == str(exe)
    search.assert_not_called()
    assert capture.resolve_procdump() == str(exe)
    search.assert_not_called()
    monkeypatch.setattr(capture, 'DEFAULT_PROCDUMP_PATH', str(tmp_path / 'missing.exe'))
    assert capture.resolve_procdump() == str(exe)
    search.assert_called_once_with('procdump64.exe')
    registry.QueryValueEx.return_value = (0, 4)
    with pytest.raises(RuntimeError, match='license'):
        capture.resolve_procdump(str(exe))
