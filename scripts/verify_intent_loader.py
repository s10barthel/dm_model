"""Benchmark prepared graphs and optionally verify one full training epoch.

The subprocess memory guard stops verification above 24 GiB RSS or private commit.
Reports and logs are retained under --output-dir; existing runs are never overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import gc
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch

from dataset import ActionDataset
from dataset_loading import DEFAULT_CACHE_DIR
from models.dataset_config import build_action_dataset_kwargs
from prepared_dataset import PreparedActionDataset


def write_progress_report(path, report, *, required=False):
    """A transient Windows report-file error must not discard a training epoch."""
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        if required:
            raise
        report["report_write_errors"] = report.get("report_write_errors", 0) + 1
        report["last_report_write_error"] = str(exc)


def benchmark(metadata, cache_dir):
    args = metadata["training_args"]
    dirs = metadata["resolved_dirs"]
    options = build_action_dataset_kwargs(
        args, train=True, diagnostic_label_dir=args.get("diagnostic_label_dir"),
        require_goal_next10_diagnostics=args.get("require_goal_next10_diagnostics", False),
        physical_cache_dir=args.get("physical_cache_dir"),
        lane_survival_cache_dir=args.get("lane_survival_cache_dir"))
    ids = sorted(metadata["train_match_ids"], key=lambda m: (Path(dirs["train_feature_dir"]) / f"{m}.pt").stat().st_size)
    sample_ids = list(dict.fromkeys(ids[int((len(ids) - 1) * f)] for f in (0.2, 0.5, 0.8)))
    rows = []
    for match_id in sample_ids:
        original_times, prepared_times = [], []
        cache = PreparedActionDataset([match_id], feature_dir=dirs["train_feature_dir"],
                                      label_dir=dirs["train_label_dir"], cache_dir=cache_dir,
                                      progress=False, **options)
        for repeat in range(2):
            gc.collect()
            start = time.perf_counter()
            with contextlib.redirect_stderr(io.StringIO()):
                eager = ActionDataset([match_id], feature_dir=dirs["train_feature_dir"],
                                      label_dir=dirs["train_label_dir"], **options)
            original_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            samples = list(cache)
            prepared_times.append(time.perf_counter() - start)
            assert len(samples) == len(eager)
            for i, (graph, label, weight) in enumerate(samples):
                expected, expected_label, expected_weight = eager[i]
                assert set(graph.keys()) == set(expected.keys())
                for key in graph.keys():
                    if isinstance(graph[key], torch.Tensor):
                        torch.testing.assert_close(graph[key], expected[key], equal_nan=True)
                    else:
                        assert graph[key] == expected[key]
                torch.testing.assert_close(label, expected_label, equal_nan=True)
                torch.testing.assert_close(weight, expected_weight)
            del eager, samples
        row = {"match_id": match_id, "samples": len(cache),
               "source_bytes": (Path(dirs["train_feature_dir"]) / f"{match_id}.pt").stat().st_size,
               "cache_bytes": cache.preparation["bytes"], "build_seconds": cache.preparation["seconds"],
               "original_seconds": original_times, "prepared_seconds": prepared_times}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del cache
    original = sum(sum(row["original_seconds"]) for row in rows)
    prepared = sum(sum(row["prepared_seconds"]) for row in rows)
    return {"matches": rows, "equivalent": True, "speedup": original / prepared,
            "passed": prepared < original,
            "note": "First and repeat passes; OS file cache is not flushed. Timings exclude batch collation and GPU work."}


def verify_epoch(metadata, cache_dir, output):
    output = Path(output).resolve()
    command = shlex.split(metadata["command"], posix=False)
    command = [token[1:-1] if token.startswith('"') and token.endswith('"') else token for token in command]
    def set_flag(flag, value):
        if flag in command:
            command[command.index(flag) + 1] = str(value)
        else:
            command.extend([flag, str(value)])
    for flag in ("--resume-run-id",):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    command = [token for token in command if token != "--cont"]
    run_id = metadata["task"] + "_loader_verify_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    set_flag("--run-id", run_id)
    set_flag("--n_epochs", 1)
    set_flag("--dataset-loading", "disk")
    set_flag("--dataset-cache-dir", cache_dir)
    set_flag("--dataset-buffer-matches", 4)
    command = [sys.executable, "-u", *command]
    report = {"run_id": run_id, "command": command, "peak_rss": 0, "peak_private": 0,
              "memory_limit_bytes": 24 * 1024**3, "status": "running"}
    report_path = output / "epoch_report.json"
    started = time.perf_counter()
    with (output / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        watched = psutil.Process(process.pid)
        try:
            next_update = 0
            while process.poll() is None:
                try:
                    # Windows venv python.exe is a launcher; the actual trainer
                    # lives in a child. Guard the whole training process tree.
                    memories = []
                    for child in [watched, *watched.children(recursive=True)]:
                        try:
                            memories.append(child.memory_info())
                        except psutil.NoSuchProcess:
                            pass
                    report["peak_rss"] = max(report["peak_rss"], sum(m.rss for m in memories))
                    report["peak_private"] = max(report["peak_private"],
                                                 sum(getattr(m, "private", m.vms) for m in memories))
                except psutil.NoSuchProcess:
                    break
                if max(report["peak_rss"], report["peak_private"]) > report["memory_limit_bytes"]:
                    report["status"] = "memory_limit_exceeded"
                    break
                elapsed = time.perf_counter() - started
                if elapsed >= next_update:
                    report["seconds"] = elapsed
                    write_progress_report(report_path, report)
                    try:
                        print(f"Verification {elapsed:.0f}s; peak RSS {report['peak_rss']/1024**3:.2f} GiB, "
                              f"private {report['peak_private']/1024**3:.2f} GiB", flush=True)
                    except OSError:
                        pass  # The memory guard continues even if its output stream is unavailable.
                    next_update = elapsed + 30
                time.sleep(1)
        finally:
            if process.poll() is None:
                try:
                    for child in reversed(watched.children(recursive=True)):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                process.terminate()
            report["exit_code"] = process.wait()
            report["seconds"] = time.perf_counter() - started
            if report["status"] == "running":
                report["status"] = "completed" if report["exit_code"] == 0 else "failed"
            mark_verification_run(metadata["task"], report)
            write_progress_report(report_path, report, required=True)
    return report


def mark_verification_run(task, report):
    """Keep a short verification checkpoint out of automatic model selection."""
    from project_config import get_model_run_root
    run_id = report["run_id"]
    if not run_id.startswith(f"{task}_loader_verify_") or Path(run_id).name != run_id:
        raise ValueError("Expected an isolated loader verification run ID")
    path = get_model_run_root(task, run_id) / "metadata.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["training_status"] = data.get("status")
        data["status"] = "verification_" + report["status"]
        data["verification_only"] = True
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def verify_reuse(metadata, cache_dir, output):
    """Check both complete splits without starting another training epoch."""
    args = metadata["training_args"]
    dirs = metadata["resolved_dirs"]
    options = build_action_dataset_kwargs(
        args, train=True, diagnostic_label_dir=args.get("diagnostic_label_dir"),
        require_goal_next10_diagnostics=args.get("require_goal_next10_diagnostics", False),
        physical_cache_dir=args.get("physical_cache_dir"),
        lane_survival_cache_dir=args.get("lane_survival_cache_dir"))
    report = {}
    for split, ids_key, prefix in [("train", "train_match_ids", "train"),
                                   ("validation", "validation_match_ids", "valid")]:
        dataset = PreparedActionDataset(
            metadata[ids_key], feature_dir=dirs[f"{prefix}_feature_dir"],
            label_dir=dirs[f"{prefix}_label_dir"], cache_dir=cache_dir, **options)
        assert dataset.preparation["built"] == 0, f"Unexpected {split} cache rebuild"
        report[split] = dataset.metadata()
        del dataset
    (output / "reuse_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-one-epoch", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = benchmark(metadata, args.cache_dir)
    (args.output_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not results["passed"]:
        raise RuntimeError("Prepared loading did not beat original processing; investigate before a full build.")
    if args.train_one_epoch:
        result = verify_epoch(metadata, args.cache_dir, args.output_dir)
        if result["status"] != "completed":
            raise RuntimeError(f"Verification {result['status']}; see training.log")
        verify_reuse(metadata, args.cache_dir, args.output_dir)


if __name__ == "__main__":
    main()
