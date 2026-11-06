# Training runtime optimizations — 6 November 2026

Implemented bounded parallel preparation, per-entry build locks, shared outcome
preparation, cheaper console progress, bulk tensor gathering, candidate grouping,
and batched IPW probability extraction. Default training remains FP32. Model
architecture, optimizer, batch size, sample order, full epoch metrics, and
validation/stopping rules retain their existing behavior.

Use `--dataset-preparation-workers 2 --dataset-worker-threads 1` to start with
moderate preparation memory. For this workstation with no competing workload,
eight workers were fastest in the extended sweep below. One process remains the
default; epoch DataLoader workers remain zero.
`--interim-metrics full` restores expensive cumulative console probability
metrics; the default `loss` retains running losses and counters. Full metrics
are still computed at each epoch end.

## Measurements

Benchmarks used the November 6 outcome-scoring run and its real source matches
on this workstation. Original runtime functions were extracted from the
recorded Git revision, then compared with the optimized implementations under
the same installed environment. Source runs and their checkpoints were not
modified. New benchmark caches were isolated under the output directory and
removed through normal workflow cleanup after success. No `train.py` process
was active. File caches were not flushed; these are fresh prepared-cache builds,
not measurements of cold physical storage.

Preparation trials used eight matches and 9,763 accepted samples. Case order was
randomized, with three trials per worker count. The parent used one tensor
thread, and spawned workers used one tensor/BLAS thread each. Time includes
process startup, source loading/transformation, packing, checksums, and ownership
bookkeeping; cleanup is outside the preparation timer.

| Preparation processes | Median wall time | Peak process-tree private memory |
| --- | ---: | ---: |
| 1 | 55.24 s | 2.19 GiB |
| 2 | 38.67 s | 4.29 GiB |
| 4 | 34.58 s | 6.50 GiB |

Two workers reduced preparation time by 30.0%; four reduced it by 37.4%.
Worker counts and memory costs are workload dependent. These percentages do not
include the separate benefit of scoring/conceding sharing a compatible cache.

## Extended worker-count sweep

The initial 1/2/4 sweep did not establish an optimum beyond four workers. A
follow-up compared 4, 6 and 8 workers on the same 24 requested matches, with
three trials per count in randomized order. Every trial built 23 entries and
produced the same 34,823 accepted samples. The machine had eight physical cores,
16 logical CPUs and 32 GiB RAM, with approximately 22 GiB available before the
benchmark and no active training or feature-generation process. Each preparation
worker used one tensor/BLAS thread. Fresh isolated prepared caches were created
and cleaned up per trial; the OS file cache was not flushed.

| Processes | Median wall time | Observed range | Peak process-tree private memory |
| --- | ---: | ---: | ---: |
| 4 | 88.58 s | 88.42–90.56 s | 6.38 GiB |
| 6 | 104.60 s | 85.06–113.53 s | 8.72 GiB |
| 8 | 75.44 s | 75.44–75.57 s | 10.79 GiB |

Eight workers reduced elapsed time by 14.8% versus four and were consistently
fastest. Six-worker results were substantially more variable; this sweep does
not isolate the cause. The recommendation for the current workload on an idle
workstation is now `--dataset-preparation-workers 8 --dataset-worker-threads 1`.
The memory figures describe the preparation benchmark, not a complete CUDA
training process. Counts above eight and full-corpus scaling remain unmeasured.

Reproduce with the common benchmark arguments below plus
`--suite prepare --matches 24 --repeats 3 --workers 4,6,8`.
Raw results: `out/training_optimizations_20261106/workers_4_6_8/results.json`.
All nine trials completed and left no prepared `.pt` files in their isolated
cache directories.

## Component and short-loop measurements

Component trials used 512 shuffled samples from four real matches, three timed
repeats after warm-up, and one CPU tensor thread. Candidate layouts and complete
batched graphs, labels, weights, metadata, slices and increments matched exactly.
The IPW trial used a lightweight deterministic scorer on CUDA to isolate data
batching and probability extraction; its percentage is not an end-to-end IPW
forecast. Grouped probabilities matched the scalar reference within tolerance.

| Component | Original median | Optimized median | Time reduction |
| --- | ---: | ---: | ---: |
| Packed batch construction | 22.78 ms | 15.72 ms | 31.0% |
| Intent candidate construction | 58.31 ms | 28.25 ms | 51.6% |
| IPW batching/probability extraction | 205.66 ms | 125.51 ms | 39.0% |
| Full-split metric replay | 11.49 s | 0.97 s | 91.5% |

Metric replay used deterministic synthetic scores/labels at the saved training
split's actual sample count. It compared cumulative metrics every 50 batches
plus complete final metrics against complete final metrics alone. It measures
CPU reporting overhead rather than model throughput or model quality.

Short production trials used 20 real matches, 50 batches (25,600 examples) per
trial, three repeats per case, eight CPU tensor threads, and the real FP32 GNN,
Adam, CPU graph validation, losses and metrics. Each trial reset weights,
optimizer and random seeds. Case order was randomized after warm-up. Both phases
used the selected training examples; validation-mode timing is a throughput
check, not a holdout evaluation.

| Production loop | Original median | Optimized median | Time reduction |
| --- | ---: | ---: | ---: |
| Training | 9.778 s | 9.758 s | 0.2% |
| Validation mode | 10.255 s | 9.633 s | 6.1% |

Short training times overlap and do not establish a training-loop speedup.
These short trials include only one interim probability-metric calculation in
the original loop and therefore miss the cumulative reporting cost of a full
epoch. Targets and their order matched exactly; predictions matched within
`rtol=1e-5, atol=1e-6`. CUDA weights were not bitwise identical, including repeat
trials. CPU optimizer/dropout and checkpoint-resume regressions separately check
the required numerical behavior.

Independent real-data comparisons confirmed that scoring and conceding produce
identical graph attributes and complete labels for two selected matches. Fixture
tests also cover goal-node inclusion/exclusion and distinct feature settings.
The shared profile retains every other resolved option and source path in its
identity; other task families remain separate.

## Full training epoch confirmation

A further comparison streamed all 612 training matches (912,430 samples) from
the source run's retained packed cache. Checksums, payloads and source signatures
were verified before timing. The cache and its ownership records were read only.
Each variant started with the same saved weights, fresh Adam state and seeds;
no checkpoint or model artifact was written. There was one complete epoch per
variant, in randomized order, with eight tensor threads and FP32 CUDA execution.

| Full training epoch | Original | Optimized |
| --- | ---: | ---: |
| Wall time | 273.15 s | 254.77 s |
| Recorded batching time | 57.21 s | 45.58 s |
| Recorded deserialization time | 24.27 s | 18.21 s |
| Mean training CE loss | 0.36973582 | 0.36973319 |

Wall time fell by 6.7%. Storage/cache variation contributes to that comparison:
deserialization was 6.1 seconds faster in the optimized trial. Subtracting each
trial's recorded deserialization time gives a 4.9% reduction in the remainder;
that is not an isolated measurement of GPU compute. This single full-epoch pair
confirms a useful result on this workload, not a guaranteed training forecast.

Targets, diagnostics and branch order matched exactly. Maximum prediction
difference over the long CUDA trajectories was 0.00546, and maximum parameter
difference was 0.02381. Exact CUDA replay is not claimed; short repeated trials
also showed nonidentical CUDA weights. Deterministic CPU batch, gradient,
optimizer/dropout and actual checkpoint/resume tests cover semantic equivalence.
The full-epoch diagnostic explicitly records long-run CUDA drift instead of
applying the short-trial prediction tolerance.

## Reproduction

Run `scripts/diagnostics/benchmark_training_optimizations.py` with
`--source-metadata <saved-run>/metadata.json --output <new-directory>
--baseline-revision c4ce4b98a640ca4f74507c9c3c18d619417b9948` and one of:

- `--suite components --matches 4 --repeats 3 --device cuda`
- `--suite prepare --matches 8 --repeats 3 --workers 1,2,4`
- `--suite training --matches 20 --steps 50 --repeats 3 --device cuda --threads 8`
- Full training epoch: `--suite training --existing-cache --matches 612 --steps 100000 --repeats 1 --device cuda --threads 8 --phases train --skip-warmup --record-gpu-drift`

The output directory must be new. The script refuses to benchmark while
`train.py` is active and never suspends existing work. It records the baseline
Git revision, installed versions, exact selected matches, all repeats, and cache
cleanup reports. `--existing-cache` supports read-only runtime measurements on a
retained source cache, verifying file checksums, payloads and source signatures.
It does not change that cache's ownership or retention settings.

Raw results are under `out/training_optimizations_20261106/{components,preparation,training}/`.
The full-epoch result is under `out/training_optimizations_20261106/full_epoch_verified/`.
The component and preparation percentages must not be added together or treated
as guaranteed full-run savings. Existing benchmarks did not support enabling
background prefetch or pinning by default, and this change does not enable them.

Preparation fingerprints change once for this implementation, requiring cache
rebuilding and preserving the existing rejection of incompatible unfinished
checkpoints. Completed model artifacts remain readable. IPW fingerprints now
include the shared candidate-grouping helper.

Validation passed for 518 selected cases across the compatibility suite and
targeted rechecks. Coverage includes exact mixed/variable-size batches and dtype
promotion, CPU gradients/Adam/dropout, actual CPU/CUDA training and checkpoint
resume, all wrapper tasks, IPW alignment/normalization/reuse, Windows spawned
preparation, failure/retry, concurrent entries, and ownership cleanup.
`git diff --check` also passed.
