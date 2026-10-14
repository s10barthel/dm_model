"""Compare prepared samples and stream a complete split under a 24 GiB guard.

Verification writes only beneath --output-dir and never trains or alters a model.
Successful verification removes its managed graph cache unless --keep-cache is set.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil


def verify_samples(eager, disk):
    import torch
    count = 0
    for count, (graph, label, weight) in enumerate(disk, 1):
        expected_graph, expected_label, expected_weight = eager[count - 1]
        if set(graph.keys()) != set(expected_graph.keys()):
            raise AssertionError("Prepared graph attributes differ")
        for key in graph.keys():
            if isinstance(graph[key], torch.Tensor):
                assert graph[key].dtype == expected_graph[key].dtype
                torch.testing.assert_close(graph[key], expected_graph[key], rtol=0, atol=0, equal_nan=True)
            elif graph[key] != expected_graph[key]:
                raise AssertionError(f"Prepared graph attribute differs: {key}")
        torch.testing.assert_close(label, expected_label, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(weight, expected_weight)
    if count != len(eager):
        raise AssertionError("Prepared sample count differs")
    return count


def worker(options):
    import torch
    from cache_lifetime import atomic_json, training_context
    from dataset import ActionDataset
    from models.dataset_config import build_action_dataset_kwargs
    from prepared_dataset import PreparedActionDataset
    from training_monitor import TrainingMonitor

    # Small per-graph tensor operations otherwise spend most time in thread pools.
    torch.set_num_threads(1)
    output = Path(options.output_dir).resolve()
    previous = json.loads((output / "metadata.json").read_text()) if options.resume else None
    metadata = json.loads(Path(options.source_metadata).read_text(encoding="utf-8"))
    args = dict(metadata["training_args"])
    args.update(dataset_cache_dir=str(output / "cache"), keep_cache=options.keep_cache,
                resume_checkpoint=None, artifact_stage=None)
    if previous is not None:
        descriptor = previous["cache_session"]
        if Path(descriptor["root"]).resolve() != (output / "cache").resolve() or descriptor["kind"] != "direct":
            raise ValueError("Verification cache ownership does not match its output directory.")
        args.update(cache_session=descriptor, resume_checkpoint="verification")
    state = SimpleNamespace(**args)
    state._cache_metadata_directory = output
    report = dict(status="running", task=args["task"], source_metadata=str(Path(options.source_metadata).resolve()),
                  representative_matches=[], splits={})
    atomic_json(output / "metadata.json", report)
    monitor = TrainingMonitor(output, enabled=True)
    try:
        with training_context(state) as session:
            report["cache_session"] = state.cache_session
            atomic_json(output / "metadata.json", report)
            kwargs = build_action_dataset_kwargs(args, train=True,
                diagnostic_label_dir=args.get("diagnostic_label_dir"),
                require_goal_next10_diagnostics=args.get("require_goal_next10_diagnostics"),
                physical_cache_dir=args.get("physical_cache_dir"),
                lane_survival_cache_dir=args.get("lane_survival_cache_dir"))
            feature_dir = metadata["resolved_dirs"]["train_feature_dir"]
            label_dir = metadata["resolved_dirs"]["train_label_dir"]
            ids = sorted(metadata["train_match_ids"], key=lambda m: (Path(feature_dir) / (m + ".pt")).stat().st_size)
            sample_ids = list(dict.fromkeys(ids[int((len(ids) - 1) * f)] for f in (.2, .5, .8)))
            for match_id in sample_ids:
                monitor.event("representative_start", "cpu", match_id=match_id)
                eager = ActionDataset([match_id], feature_dir=feature_dir, label_dir=label_dir, **kwargs)
                disk = PreparedActionDataset([match_id], feature_dir=feature_dir, label_dir=label_dir,
                    cache_dir=args["dataset_cache_dir"], cache_session=session, progress=False, **kwargs)
                count = verify_samples(eager, disk)
                report["representative_matches"].append(dict(match_id=match_id, samples=count, equivalent=True))
                del eager, disk
                gc.collect()
                atomic_json(output / "metadata.json", report)
            for split, key in (("training", "train"), ("validation", "validation")):
                monitor.event("preparation_start", "cpu", split=split)
                dir_key = "train" if split == "training" else "valid"
                dataset = PreparedActionDataset(metadata[key + "_match_ids"],
                    feature_dir=metadata["resolved_dirs"][dir_key + "_feature_dir"],
                    label_dir=metadata["resolved_dirs"][dir_key + "_label_dir"],
                    cache_dir=args["dataset_cache_dir"], cache_session=session,
                    buffer_matches=args.get("dataset_buffer_matches", 4), shuffle=split == "training",
                    seed=args.get("seed", 100), **kwargs)
                monitor.event("preparation_complete", "cpu", split=split, samples=len(dataset))
                started = time.perf_counter()
                count = 0
                for graph, label, weight in dataset:
                    count += 1
                if count:
                    del graph, label, weight
                if count != len(dataset):
                    raise AssertionError("Full-split iteration sample count differs")
                report["splits"][split] = dict(dataset.metadata(), streamed_samples=count,
                    stream_seconds=time.perf_counter() - started)
                del dataset
                gc.collect()
                monitor.event("stream_complete", "cpu", split=split, samples=count)
                atomic_json(output / "metadata.json", report)
            report["status"] = "completed"
            atomic_json(output / "metadata.json", report)
    finally:
        monitor.close()


def write_progress(path, report):
    """Report contention must not terminate verification or disable its guard."""
    from cache_lifetime import atomic_json
    try:
        atomic_json(path, report)
    except OSError as exc:
        report["report_write_errors"] = report.get("report_write_errors", 0) + 1
        report["last_report_write_error"] = str(exc)
        if report["report_write_errors"] == 1:
            try:
                print(f"Verification progress report temporarily unavailable: {exc}", file=sys.stderr, flush=True)
            except OSError:
                pass


def supervise(options):
    from cache_lifetime import workflow_running
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=options.resume)
    previous = None
    if options.resume:
        state = json.loads((output / "metadata.json").read_text())
        if Path(state["source_metadata"]).resolve() != Path(options.source_metadata).resolve():
            raise ValueError("Cannot resume verification of different source metadata.")
        if workflow_running(state["cache_session"]):
            raise RuntimeError("The original verification is still running.")
        previous = json.loads((output / "memory_report.json").read_text())
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--source-metadata", str(Path(options.source_metadata).resolve()), "--output-dir", str(output)]
    if options.keep_cache:
        command.append("--keep-cache")
    if options.resume:
        command.append("--resume")
    report = dict(status="running", command=command, peak_rss_bytes=0,
                  peak_private_bytes=0, memory_limit_bytes=24 * 1024**3)
    previous_elapsed = 0.0
    if previous:
        report.update(previous_attempt=previous, peak_rss_bytes=previous["peak_rss_bytes"],
                      peak_private_bytes=previous["peak_private_bytes"])
        previous_elapsed = previous.get("elapsed_seconds", 0.0)
    started = time.monotonic()
    with (output / "verification.log").open("a" if options.resume else "w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        watched = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                memories = []
                try:
                    processes = [watched, *watched.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                for process in processes:
                    try:
                        memories.append(process.memory_info())
                    except psutil.Error:
                        continue
                report["peak_rss_bytes"] = max(report["peak_rss_bytes"], sum(m.rss for m in memories))
                report["peak_private_bytes"] = max(report["peak_private_bytes"],
                    sum(getattr(m, "private", m.vms) for m in memories))
                report["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
                write_progress(output / "memory_report.json", report)
                if max(report["peak_rss_bytes"], report["peak_private_bytes"]) > report["memory_limit_bytes"]:
                    report["status"] = "memory_limit_exceeded"
                    for process in reversed(processes):
                        try:
                            process.terminate()
                        except psutil.Error:
                            pass
                    break
                time.sleep(.5)
            returncode = child.wait(timeout=15)
            report.update(returncode=returncode, elapsed_seconds=previous_elapsed + time.monotonic() - started)
            if report["status"] == "running":
                report["status"] = "completed" if returncode == 0 else "failed"
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
            write_progress(output / "memory_report.json", report)
    print(json.dumps(report), flush=True)
    if report["status"] != "completed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted verification in the same output directory.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    worker(options) if options.worker else supervise(options)


if __name__ == "__main__":
    main()
