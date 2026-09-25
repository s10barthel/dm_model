# Training controls, recovery and monitoring

## New runs

`scripts/train_relevant_models.py` accepts these optional flags in addition to
the existing model/feature selection options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--start_lr FLOAT` | Model-specific default | Starting learning rate for every selected model |
| `--min_lr FLOAT` | Model-specific default | Minimum learning rate for every selected model |
| `--accumulation-steps INT` | `1` | Physical batches per optimizer update, for every selected model |
| `--monitoring {on,off}` | `off` | GPU and training telemetry |

Either learning-rate override can be supplied independently. Resolved rates must
be finite and positive, with minimum no greater than start. Existing per-model
defaults remain in effect for omitted overrides. Accumulation must be a positive
integer; there are no per-model accumulation overrides.

For example, append the following to your normal wrapper command for a new
pass-intent run:

```text
--pass-intent-batch-size 128 --accumulation-steps 2 --monitoring on
```

This processes physical batches of 128 with an effective batch size of 256. It
does not require halving the learning rate merely because the physical batch is
smaller. With `--batch-size 256 --accumulation-steps 2`, the effective batch is
512, but individual forward/backward passes still require batch-256 memory.
Accumulation is off by default for all tasks, including intent models.

Gradients are sample-weighted across each group, including the final short group.
Regularization follows the same weighting; clipping and Adam updates occur once
per group. Each microbatch's autograd graph is released after backward. This is
not a guarantee of numerical identity with a large batch when stochastic or
batch-dependent operations are involved.

The direct `train.py` entry point also accepts accumulation and monitoring. Its
batch-size option is spelled `--batch_size`. The top-level `scripts/main.py`
interface has not acquired these new overrides; use the training wrapper.

## Optimizer and scheduler

New training keeps Adam's optimizer state across epochs. Previously Adam was
recreated at each epoch, discarding its moments. This deliberately changes the
optimization behavior of new runs.

On the existing validation-plateau learning-rate reduction, both the best model
weights and their matching optimizer state are restored, then the current
learning rate is halved subject to the configured minimum. Stopping counters
continue to track the current run rather than rewinding to the best epoch.

Intent losses and ranking metrics are grouped by candidate count. Candidate
selection, per-graph weighting, and the existing descending-argsort tie policy
are preserved. Graph features, edges and node-selection targets are validated on
the CPU before the batch is transferred to CUDA.

## Resume one model

```powershell
.\.venv\Scripts\python.exe scripts/train_relevant_models.py --resume-id pass_intent/<run_id>
```

To enable monitoring during that resume:

```powershell
.\.venv\Scripts\python.exe scripts/train_relevant_models.py --resume-id pass_intent/<run_id> --monitoring on
```

A unique bare run ID is also accepted. Bundle IDs are not supported. Resume
restores the individual model's saved settings; other training flags are rejected,
even when they repeat a saved value. Monitoring is an independent operational
option and defaults to off for each invocation. A completed checkpoint reports
completion without running more training. Existing bundle IDs/references are
not rewritten and the remaining models in a failed bundle are not launched.

Only the new `training_checkpoint.pt` format is supported. Older weights-only
checkpoints cannot be resumed. The former manual `train.py --resume-run-id`
workflow now points users to the wrapper, avoiding partial configuration restore.

Checkpoints contain model and optimizer state, best-model/optimizer state,
learning-rate/stopping counters, resolved settings, data identity and Python,
NumPy, CPU/CUDA RNG states. Recovery state is saved after dataset initialization
(epoch zero) and after each completed epoch. An interrupted epoch is repeated in
full. There are no periodic checkpoints within epochs and no automatic retries.

One authoritative recovery file is written to a temporary file, flushed, and
atomically replaced. The old checkpoint survives a failed replacement. Inference
weight files remain compatible with existing consumers and can be regenerated
from the authoritative checkpoint if their publication was interrupted.

Resume checks source paths, file sizes/modification times, split identity and
preprocessing/training code fingerprints. It refuses changed inputs/code rather
than silently training under different conditions. These source signatures are
not full content hashes of the graph corpus. Software versions and resume events
are recorded; changed environments can still prevent exact numerical replay.
The disk loader uses its saved seed and resumed epoch to reconstruct shuffling.

For expanding validation, an individual model ID resumes its unique unfinished
training stage. It does not orchestrate subsequent folds or final refitting. If
all existing stages are complete, it reports completion of the saved stage;
the original wrapper pipeline may still have unstarted stages.

## Optional monitoring

`--monitoring on` creates these files in the individual model/stage directory:

- `gpu_monitor.jsonl`: a separate process samples NVIDIA telemetry approximately
  every 30 seconds, including memory, utilization, temperature, power and clocks.
- `training_monitor.jsonl`: epoch/batch/stage markers, allocated/reserved/peak
  PyTorch memory, graph sizes, and match/source-row identifiers.

Logs are flushed regularly. The sampler exits when the parent closes its pipe
or dies. Unsupported counters and telemetry failures are nonfatal. With monitoring
off, there is no telemetry subprocess or periodic telemetry collection. Ordinary
progress logs, `crash.log`, and CPU validation remain available.

No continuous synchronous CUDA debugging, timeout-registry changes, GPU clock
changes or driver changes are made. NVIDIA Control Panel Debug Mode, enabled
manually, is separate from these logs.

The changes reduce avoidable GPU operations and improve recovery/diagnostics;
they are not a confirmed fix for intermittent native driver crashes. A bounded
GPU smoke test cannot establish overnight stability. The prepared-cache code
fingerprint includes `models/utils.py`, so this update invalidates older prepared
intent caches and the next full run will prepare a new cache. Old caches are not
automatically removed.
