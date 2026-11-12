"""Run a saved training command once its feature run has completed successfully.

Usage: python scripts/train_when_features_ready.py --job-file path/to/job.json
The job contains feature_run_id and command (a JSON array of command arguments).
Status and output files are written alongside the job. An existing status file
prevents the same job from being launched twice, including after completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def features_ready(metadata):
    status = metadata.get("status")
    attempt = metadata.get("extension_attempt") or {}
    if status == "completed":
        if attempt.get("artifact_state") in ("partial", "unknown"):
            raise ValueError("Feature run reports completion with incomplete artifacts")
        return True
    if status in ("in_progress", "running", "building"):
        return False
    raise ValueError(
        f"Feature run did not complete successfully: status={status!r}, "
        f"phase={attempt.get('phase')!r}, errors={attempt.get('primary_error_chain', [])!r}"
    )


def run_job(job_file, poll_seconds=60, *, project_root=ROOT):
    job_file = Path(job_file).resolve()
    job = json.loads(job_file.read_text(encoding="utf-8-sig"))
    run_id = job["feature_run_id"]
    command = job["command"]
    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        raise ValueError("feature_run_id must be a run name")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a nonempty array of strings")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    metadata_path = project_root / "data" / "features" / "runs" / run_id / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    state_path = job_file.with_suffix(".status.json")
    state = dict(status="waiting", waiter_pid=os.getpid(), feature_run_id=run_id,
                 command=command, queued_at=datetime.now().astimezone().isoformat())
    # Exclusive creation claims this job before any training can be launched.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)

    def update(**values):
        state.update(values, updated_at=datetime.now().astimezone().isoformat())
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    def log(message):
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)

    try:
        log(f"Waiting for feature run {run_id}; checking every {poll_seconds:g} seconds")
        while True:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if metadata.get("run_id") != run_id:
                raise ValueError("Feature metadata run_id does not match the queued job")
            ready = features_ready(metadata)
            update(feature_status=metadata.get("status"),
                   feature_phase=(metadata.get("extension_attempt") or {}).get("phase"))
            if ready:
                break
            time.sleep(poll_seconds)

        stdout_path = job_file.with_suffix(".training.stdout.log")
        stderr_path = job_file.with_suffix(".training.stderr.log")
        update(status="starting", started_at=datetime.now().astimezone().isoformat())
        log("Feature run completed successfully; starting training")
        # Pass arguments directly, preserving the saved command without shell expansion.
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
            process = subprocess.Popen(command, cwd=project_root, stdout=stdout, stderr=stderr,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
            update(status="running", training_pid=process.pid,
                   stdout_log=str(stdout_path), stderr_log=str(stderr_path))
            return_code = process.wait()
        update(status="completed" if return_code == 0 else "failed", exit_code=return_code,
               finished_at=datetime.now().astimezone().isoformat())
        log(f"Training exited with code {return_code}")
        return return_code
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    raise SystemExit(run_job(args.job_file, args.poll_seconds))


if __name__ == "__main__":
    main()
