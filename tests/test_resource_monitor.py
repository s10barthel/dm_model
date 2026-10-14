import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import psutil
import pytest

import training_monitor as monitor


class Stopper:
    def __init__(self, count=1):
        self.remaining = count
    def is_set(self):
        return self.remaining <= 0
    def wait(self, seconds):
        self.remaining -= 1


def test_process_memory_and_windows_commit():
    pid = os.getpid()
    created = psutil.Process(pid).create_time()
    with patch.object(monitor, "windows_commit", return_value=dict(commit_charge_bytes=100,
            commit_limit_bytes=200, commit_remaining_bytes=100)):
        event = monitor.sample_resources(pid, created, include_top=True)
    assert event["process"]["rss_bytes"] > 0
    assert event["system"]["physical_total_bytes"] > 0
    if os.name == "nt":
        assert event["system"]["commit_remaining_bytes"] == 100
        assert event["process"]["private_committed_bytes"] > 0
    assert len(event["largest_processes"]) <= 5
    assert all("process_created" in p for p in event["largest_processes"])


def test_pid_reuse_and_inaccessible_processes():
    pid = os.getpid()
    event = monitor.sample_resources(pid, 0)
    assert "reused" in event["process_error"] and "process" not in event
    denied = SimpleNamespace(oneshot=lambda: (_ for _ in ()).throw(psutil.AccessDenied(123)))
    with patch.object(monitor.psutil, "process_iter", return_value=[denied]), \
         patch.object(monitor, "windows_commit", side_effect=OSError("unavailable")):
        event = monitor.sample_resources(pid, psutil.Process(pid).create_time(), include_top=True)
    assert event["largest_processes"] == []
    if os.name == "nt":
        assert event["system"]["commit_error"] == "unavailable"


def test_sampling_intervals_and_gpu_failure(tmp_path):
    with patch.object(monitor, "sample_gpu", side_effect=RuntimeError("GPU unavailable")):
        monitor.gpu_loop(tmp_path / "gpu_monitor.jsonl", Stopper())
    assert json.loads((tmp_path / "gpu_monitor.jsonl").read_text())["error"] == "GPU unavailable"
    top = []
    def sample(pid, created, include_top=False):
        top.append(include_top)
        return {"pid": pid, "process_created": created}
    with patch.object(monitor, "sample_resources", side_effect=sample), \
         patch.object(monitor.time, "monotonic", side_effect=range(0, 40, 5)):
        monitor.resource_loop(tmp_path / "resource_monitor.jsonl", 1, 2., Stopper(8))
    assert top == [True, False, False, False, False, False, True, False]
    assert len((tmp_path / "resource_monitor.jsonl").read_text().splitlines()) == 8


def test_disabled_monitor_does_not_sample_or_write(tmp_path):
    with patch.object(monitor, "sample_resources", side_effect=AssertionError("sampled")), \
         patch.object(monitor.subprocess, "Popen", side_effect=AssertionError("spawned")):
        instance = monitor.TrainingMonitor(tmp_path, False)
        instance.event("preparation_start", "cpu")
        instance.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows performance counters")
def test_live_windows_commit_counter():
    counters = monitor.windows_commit()
    assert 0 < counters["commit_charge_bytes"] <= counters["commit_limit_bytes"]
    assert counters["commit_remaining_bytes"] == counters["commit_limit_bytes"] - counters["commit_charge_bytes"]
