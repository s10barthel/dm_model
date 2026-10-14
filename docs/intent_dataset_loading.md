# Training with a prepared graph cache

Fresh training runs use a prepared disk cache by default for all eight wrapper
tasks: action/pass/success intent, pass success, pass height, scoring/conceding
outcomes, and failure receiver. Supported IPW and physical-feature configurations
use the same bounded loader.
The existing graph and label files remain the source of truth. Preparation runs
the existing dataset filters and feature transformations once per match and
saves the accepted CPU graph tensors and labels in the version 2
`packed_tensors` format. Compatible later runs reuse them.
Tasks outside this set retain their existing loading behavior.

## Packed tensor format

Each match stores contiguous CPU tensors with per-sample offsets, plus labels
and graph metadata. Node/edge indices remain local to each graph. The loader
constructs graph objects only when requested, using tensor views; it does not
materialize or memoize the complete graph list. Optional attributes, scalar and
empty tensors, diagnostics, dtypes and precision are preserved. Unsupported
attribute structures raise an error. Tensor views are read-only inputs for the
training pipeline; PyG collation creates the batch tensors. Memory remains
bounded by one match during preparation, the training match group, and batches
that may cross group boundaries. This is a match bound, not a byte limit.

Plain `torch.load` is used, with memory mapping disabled. There is no additional
format flag: disk loading automatically selects packed tensors for all eight
tasks. IPW still attaches split-normalized weights by stable sample position;
shared packed files never contain these weights. Pass-height thresholds remain
applied at load time, so compatible threshold experiments can share raw labels.

Version 1 graph-object caches are rebuilt from source into a separate identity.
There is no legacy reader or migration. Existing retained cache files are left
untouched and can be removed manually when unused. Saved loading modes, buffer
sizes and deterministic sample ordering remain subject to the existing
resume/refit compatibility checks. Checkpoints written before this update fail
the existing preprocessing-fingerprint check under updated code; resuming those
checkpoints requires matching preprocessing code. Source-based refits can rebuild
with the current format while retaining saved loading choices. The codec is part
of the preparation code fingerprint; this change can also trigger one-time IPW
probability recomputation.
IPW probability files keep their separate retention policy.

Manifests and run metadata record `format` and `version`. Preparation reports
`packing_seconds`, `transform_seconds`, and `save_and_hash_seconds`; these measure
work performed in the current invocation. `last_load_seconds` measures epoch
deserialization, `last_reconstruct_seconds` measures lazy graph reconstruction,
and `last_load_and_reconstruct_seconds` is their sum. Epoch logs show both and
the combined value; batch collation/model work is outside these timers.

The initial prototype benchmark used 7,018 outcome-scoring samples, warm OS file
cache and two-epoch GPU trials. Plain packed tensors reduced cache size by 5.2%,
training time by 28.1%, and preparation plus training by 15.8%. These subset
measurements motivated integration; they are not promised full-run speedups.
The repeated production comparison recorded a 25.0% reduction in preparation,
startup and two-epoch training combined (39.96 s to 29.95 s), with exact equality
for all 7,018 samples and 22 shuffled batches. Median training/validation time was
16.24 s for graph objects and 9.73 s for packed tensors. Timing ranges overlapped
between some trials; treat these as subset results. Production measurements
include dataset initialization and validation, which the original prototype did
not measure. The full original outcome-scoring split also streamed all
1,347,898 samples successfully: peak private memory 2.09 GiB and RSS 0.94 GiB
under a 24 GiB guard. Its 36.61 GiB packed cache was automatically removed after
successful verification, with zero cleanup errors. Reproduce the comparison with:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_cache_formats.py --source-metadata <saved-run-metadata.json> --output-dir <new-output-directory>
```

The comparison writes temporary baseline/packed caches and model weights only
under its output directory and retains them for inspection. It uses fresh
processes with a 24 GiB memory guard, identical sample/batch checks, repeated
creation/reading/training trials and the production packed loader. It does not
flush the Windows file cache. The full-split verifier can separately confirm
bounded memory and automatic cleanup without training a model.

## Controls

These options work in `train.py`, `scripts/train_relevant_models.py`, and
`scripts/main.py`:

| Option | Default | Effect |
| --- | --- | --- |
| `--dataset-loading` | `auto` | `auto` selects disk for all eight supported tasks in fresh runs; `memory` explicitly selects the full in-memory loader; `disk` requires a supported task. |
| `--dataset-buffer-matches` | `4` | Number of prepared matches mixed during training. Must be positive. Validation always reads one match at a time. |
| `--dataset-cache-dir` | `data/cache/intent_datasets` under the repository | Shared cache location; the historical directory name remains for compatibility. A different drive can be used. |
| `--keep-cache` | Disabled | Keep prepared graph/label caches after the complete workflow succeeds. |

Add these options to an existing training command; feature regeneration is not
needed. Disk loading supports inverse-propensity weighting. Preparation releases
the main match graphs before constructing the corresponding propensity-model
view. Only scalar probabilities, carry masks, and weights span the full split;
weights remain separate from shared graph caches.

Training shuffles match order each epoch, loads a group of matches, and shuffles
all samples in that group. Each accepted sample appears once per epoch, including
each earlier-frame augmentation. Batches can cross group boundaries. Validation
retains source order. The shuffle is deterministic for a given seed and epoch,
including resumed epoch numbers, but differs from the old global shuffle.
Fresh disk and memory runs can therefore produce different trained weights.
Resume and source-based refits preserve the recorded loading mode and buffer
size; historical `auto` settings use their original interpretation if a resolved
mode was not saved. Existing source/code compatibility checks still apply.
The buffer bounds graph counts by matches, not a fixed number of bytes.

## Cache lifetime and disk use

Cache identity includes task, all resolved `ActionDataset` options, source
directories, preparation-specific code, and the cache schema version. Per-match
source signatures validate individual entries. Batch size, accumulation, seed,
learning rate, epoch count, optimizer, early stopping, monitoring, device, and
training or metric code do not affect prepared graph contents. Compatible
overlapping splits reuse the same per-match entries. Different tasks or feature
settings can need separate caches, so their sizes add up.

The code fingerprint covers the prepared loader, `ActionDataset`, dataset-option
construction, edge-feature configuration, and the graph/label helpers used to
construct serialized samples. Physical-xPass and reachability code is included
only for datasets that enable physical-xPass, observed-height, evaluation-xPass,
or lane-survival inputs. It deliberately excludes the training loop and model,
loss, metric, checkpoint, and monitoring code.

Preparation checks manifests and file checksums before reuse. Missing, stale, or
corrupt entries are rebuilt. Unreadable source matches are reported and retried
on the next run. Temporary writes are atomically published; completed entries
survive interruption. Source or cache changes detected during training cause an
error rather than changing the dataset silently.
Preprocessing identity is captured when its code is imported. If that code is
edited before another dataset is prepared, restart the process so cached graphs
cannot be assigned the identity of code that the process has not loaded.

The initial scan estimates additional disk usage conservatively from source file
sizes. The writer checks space between entries and keeps a 10 GiB free-space
reserve. If space is insufficient, free space or choose another cache directory
and rerun. No fallback to full in-memory loading occurs. Source signatures use paths,
sizes, and modification times; do not replace source contents while preserving
their timestamps deliberately.

The narrower preparation fingerprint introduced after the initial disk-loader
release intentionally does not accept the older broad-fingerprint caches. Delete
those caches manually and allow one rebuild. No automatic cache migration or
legacy-ID aliasing is performed.

Prepared entries created by training have durable ownership records in
`.ownership.sqlite`; a process-safe root lock protects claims, validation,
publication, and deletion. One wrapper workflow owns caches through all selected
models, validation stages, and final refits. Direct training owns its own session.

After successful model and workflow artifact publication, entries are deleted
only when every participating workflow succeeded, no process still reads them,
and no workflow requested `--keep-cache`. A later workflow can collect entries
whose cleanup was deferred by an active reader. Failure, cancellation, and
abrupt exits retain completed entries. PID creation times prevent PID reuse from
mistaking a different process for the original owner. Uncertain process access
is treated conservatively.

`--keep-cache` persistently retains entries for compatible later runs. Entries
that existed before ownership tracking, and caches prepared directly through
the library without a training session, are retained. Automatic cleanup never
prunes source features, labels, model checkpoints, physical caches, or the
separately controlled IPW probability cache. Retained entries can be deleted
manually when no process uses them.

Resume of a direct run reconnects its failed cache session. Resuming one stage
from an interrupted wrapper does not complete the original multi-stage
workflow, so its retained caches remain protected. Cleanup errors are reported
without changing successful training status.

Workflow manifests under `.sessions/` and `cache_lifetime` in workflow/model
metadata report created/reused entries, deleted/retained bytes, retention
reasons, and cleanup errors. Ownership and retention options do not enter the
prepared-data identity.

Run metadata also records each cache directory and identity, accepted sample counts,
built/reused entry counts, cache bytes, preparation duration, and graph loading
time for the latest epoch. When no training process uses a cache, its corresponding
directory can be removed manually; a future run rebuilds it. This does not remove
source features or model checkpoints.

## Verification and timing

The first run pays the preparation cost. Every epoch still reads prepared graphs,
collates batches, and runs the model; the cache removes repeated filtering and
graph transformations, not these remaining costs. Loading is synchronous and uses
zero DataLoader workers to keep the memory bound predictable on Windows.

To benchmark three representative matches against an existing intent run:

```powershell
.\.venv\Scripts\python.exe scripts/verify_intent_loader.py `
  --source-metadata saved/pass_intent/<run-id>/metadata.json `
  --output-dir tmp/intent_loader_verification
```

The benchmark checks graph/label equivalence and reports repeated loading times
and actual cache sizes. Its timings do not flush the operating system file cache
and exclude GPU work. Add `--train-one-epoch` to build the full split caches and
run one isolated epoch with the source configuration. It creates a new run, marks
it as verification-only to exclude it from automatic model selection, and
records a training log and memory report, terminating the verification subprocess
if resident memory or private committed memory exceeds 24 GiB. It does not launch
a full multi-epoch training run.

For any supported task, compare representative matches and then prepare and
stream the complete saved training/validation split, without training a model:

```powershell
.\.venv\Scripts\python.exe scripts/verify_dataset_loading.py `
  --source-metadata saved/<task>/<run-id>/<stage>/metadata.json `
  --output-dir tmp/dataset_loading_verification
```

Use a new output directory, or add `--resume` to reconnect the cache ownership
of an interrupted verification in the same directory. Active verification
processes cannot be resumed concurrently. The verifier uses one CPU tensor
thread for its small graph transformations; training thread settings are unchanged.
The supervisor enforces a 24 GiB process-tree RSS or
private-commit limit and writes `memory_report.json`, `metadata.json`, resource
telemetry, and `verification.log`. Successful verification removes its managed
cache by default; add `--keep-cache` to retain it. Failure preserves completed
entries for inspection. This verifies data equivalence and loading behavior,
not model-training throughput or full-epoch GPU memory.

## Historical intent verification on this machine

The completed September 23, 2026 check used all 612 training matches
(1,572,460 samples) and 153 validation matches (136,978 samples). One training
epoch plus validation took 2,318.61 seconds (38.6 minutes), including 337.6
seconds reading prepared graphs. Peak private committed memory was 3.75 GiB;
peak resident memory was 1.02 GiB, below the 24 GiB verification limit.

The active training and validation caches total 33.34 GiB. A subsequent check
reused all 765 entries with zero rebuilds; verification took 72 seconds for the
training cache and 7 seconds for validation. Older cache versions from interrupted
verification attempts are retained separately and are not included in that size.
The checkpoint is marked `verification_completed` and excluded from automatic
selection of trained models. This was a one-epoch integration check, not a full
model-training run.


## Outcome-scoring verification, October 14, 2026

The configuration from the interrupted October 13 outcome-scoring run was
verified on all 612 training and 153 validation matches. Three representative
matches produced identical eager and prepared graphs, attributes, labels, and
weights. Complete iteration streamed 1,068,759 training samples and 279,139
validation samples using four-match training groups and one-match validation.

The supervised process tree peaked at 2.01 GiB private committed memory and
0.89 GiB RSS, below the 24 GiB guard. This measures preparation and streaming,
not a full model-training epoch or GPU memory. Verification resumed its original
ownership session after a controlled restart and reused 191 completed entries.
After successful completion, automatic cleanup removed all 765 graph/label
entries, freeing 38.66 GiB, with zero retained graph bytes or cleanup errors.

Reports, resource telemetry and the verification log are retained locally under
`tmp/outcome_scoring_cache_verification_20261014/`. The source model run and its
feature/label artifacts were not modified.
