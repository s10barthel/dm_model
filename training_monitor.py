"""Opt-in telemetry; the GPU sampler survives a native training exception.

Closing the parent's pipe (including process death) stops the child. No CUDA
debugging or synchronization mode is enabled by this module.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil


def append_event(path, event):
    event = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with open(path, "a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(event, default=str) + "\n")


def windows_commit():
    """Actual commit counters; physical RAM usage cannot diagnose commit exhaustion."""
    import ctypes
    from ctypes import wintypes

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
                "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
                "KernelNonpaged", "PageSize")
        ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    query = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    query.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(commit_charge_bytes=info.CommitTotal * info.PageSize,
                commit_limit_bytes=info.CommitLimit * info.PageSize,
                commit_remaining_bytes=(info.CommitLimit - info.CommitTotal) * info.PageSize)


def memory_fields(memory):
    return dict(rss_bytes=memory.rss, private_committed_bytes=getattr(memory, "private", None),
                virtual_memory_bytes=memory.vms)


def sample_resources(pid, created, *, include_top=False):
    event = {"pid": pid, "process_created": created}
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - created) >= .001:
            raise RuntimeError("Training PID has been reused; process memory is unavailable.")
        event["process"] = memory_fields(process.memory_info())
    except (psutil.Error, RuntimeError) as exc:
        event["process_error"] = str(exc)
    try:
        memory = psutil.virtual_memory()
        event["system"] = dict(physical_total_bytes=memory.total, physical_used_bytes=memory.used,
                               physical_available_bytes=memory.available)
        if os.name == "nt":
            try:
                event["system"].update(windows_commit())
            except OSError as exc:
                event["system"]["commit_error"] = str(exc)
    except psutil.Error as exc:
        event["system_error"] = str(exc)
    if include_top:
        processes = []
        for process in psutil.process_iter():
            try:
                with process.oneshot():
                    processes.append(dict(pid=process.pid, name=process.name(),
                                          process_created=process.create_time(),
                                          **memory_fields(process.memory_info())))
            except psutil.Error:
                continue
        event["largest_processes"] = sorted(
            processes, key=lambda p: (p["private_committed_bytes"] if p["private_committed_bytes"] is not None
                                     else p["rss_bytes"]), reverse=True)[:5]
    return event


class TrainingMonitor:
    def __init__(self, directory, enabled=False, *, mode=None):
        self.mode = mode if mode is not None else ("on" if enabled else "off")
        self.enabled = self.mode != "off"
        enabled = self.enabled
        self.directory = Path(directory)
        self.process = None
        self.warned = False
        self.pid = os.getpid()
        self.created = psutil.Process(self.pid).create_time() if enabled else None
        if enabled:
            try:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(self.directory / "gpu_monitor.jsonl"),
                     str(self.pid), str(self.created)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                self._warning(exc)
            atexit.register(self.close)

    def _warning(self, error):
        if not self.warned:
            print(f"Monitoring unavailable or incomplete (training continues): {error}", flush=True)
            self.warned = True

    def should_sample_batch(self, batch_index):
        return self.enabled and (self.mode == "on" or batch_index % 50 == 0)

    def event(self, stage, device=None, **details):
        if not self.enabled:
            return
        try:
            import torch
            if device is not None and str(device).startswith("cuda"):
                details.update(allocated_bytes=torch.cuda.memory_allocated(device),
                               reserved_bytes=torch.cuda.memory_reserved(device),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            details["resources"] = sample_resources(self.pid, self.created)
            append_event(self.directory / "training_monitor.jsonl", {"stage": stage, **details})
        except Exception as exc:
            self._warning(exc)

    def epoch_start(self, device, epoch):
        if not self.enabled:
            return
        try:
            import torch
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            self.event("epoch_start", device, epoch=epoch)
        except Exception as exc:
            self._warning(exc)

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=7)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                    self.process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self.process = None


def sample_gpu():
    fields = "timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
    return {"fields": fields.split(","), "rows": [line.split(", ") for line in result.stdout.strip().splitlines()]}


def gpu_loop(path, stopped):
    while not stopped.is_set():
        try:
            event = sample_gpu()
        except Exception as exc:
            event = {"error": str(exc)}
        try:
            append_event(path, event)
        except OSError:
            return
        stopped.wait(30)


def resource_loop(path, pid, created, stopped):
    next_top = 0.0
    while not stopped.is_set():
        now = time.monotonic()
        include_top = now >= next_top
        if include_top:
            next_top = now + 30
        try:
            event = sample_resources(pid, created, include_top=include_top)
        except Exception as exc:
            event = {"pid": pid, "process_created": created, "error": str(exc)}
        try:
            append_event(path, event)
        except OSError:
            return
        stopped.wait(5)


def monitor_main(path, pid=None, created=None):
    pid = os.getppid() if pid is None else pid
    created = psutil.Process(pid).create_time() if created is None else created
    stopped = threading.Event()

    def watch_parent():
        sys.stdin.buffer.read()
        stopped.set()

    threading.Thread(target=watch_parent, daemon=True).start()
    gpu = threading.Thread(target=gpu_loop, args=(path, stopped), daemon=True)
    gpu.start()
    resource_loop(Path(path).with_name("resource_monitor.jsonl"), pid, created, stopped)
    gpu.join(timeout=6)


if __name__ == "__main__":
    monitor_main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None,
                 float(sys.argv[3]) if len(sys.argv) > 3 else None)
