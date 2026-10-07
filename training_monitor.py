"""Opt-in telemetry; the GPU sampler survives a native training exception.

Closing the parent's pipe (including process death) stops the child. No CUDA
debugging or synchronization mode is enabled by this module.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading


def append_event(path, event):
    event = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with open(path, "a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(event, default=str) + "\n")


class TrainingMonitor:
    def __init__(self, directory, enabled=False):
        self.enabled = enabled
        self.directory = Path(directory)
        self.process = None
        self.warned = False
        if enabled:
            try:
                self.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), str(self.directory / "gpu_monitor.jsonl")],
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


def monitor_main(path):
    stopped = threading.Event()

    def watch_parent():
        sys.stdin.buffer.read()
        stopped.set()

    threading.Thread(target=watch_parent, daemon=True).start()
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


if __name__ == "__main__":
    monitor_main(sys.argv[1])
