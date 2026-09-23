# Intent training with a prepared graph cache

Pass-intent and action-intent training now use a prepared disk cache by default.
The existing graph and label files remain the source of truth. Preparation runs
the existing dataset filters and feature transformations once per match and
saves the accepted CPU graphs and labels. Compatible later runs reuse them.
Other model tasks retain the memory loader.

## Controls

These options work in `train.py`, `scripts/train_relevant_models.py`, and
`scripts/main.py`:

| Option | Default | Effect |
| --- | --- | --- |
| `--dataset-loading` | `auto` | `auto` selects disk for pass/action intent; `memory` uses the previous loader; `disk` explicitly requires a supported intent task. |
| `--dataset-buffer-matches` | `4` | Number of prepared matches mixed during training. Must be positive. Validation always reads one match at a time. |
| `--dataset-cache-dir` | `data/cache/intent_datasets` under the repository | Shared cache location. A different drive can be used if space is limited. |

Add these options to an existing training command; feature regeneration is not
needed. Disk loading currently requires no inverse-propensity model.

Training shuffles match order each epoch, loads a group of matches, and shuffles
all samples in that group. Each accepted sample appears once per epoch, including
each earlier-frame augmentation. Batches can cross group boundaries. Validation
retains source order. The shuffle is deterministic for a given seed and epoch,
including resumed epoch numbers, but differs from the old global shuffle.

## Cache lifetime and disk use

Cache identity includes task, dataset options, source directories, preprocessing
code, schema version, and source file signatures. Batch size, seed, learning rate,
and epoch count do not affect the prepared graph contents. Compatible overlapping
splits reuse the same per-match entries. Different tasks or feature settings can
need separate caches, so their sizes add up.

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
and rerun. No other cache is deleted automatically. Source signatures use paths,
sizes, and modification times; do not replace source contents while preserving
their timestamps deliberately.

Run metadata records each cache directory and identity, accepted sample counts,
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

## Verified on this machine

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
