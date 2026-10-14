**Training pipeline optimization measurements - 15 October 2026**

The measurements favor direct batching from packed tensors and a separate lightweight monitoring mode. Ordinary one-batch background prefetching helps the existing loader, but adding it to the faster direct reader slowed training in the follow-up. Pinned transfers and deferred metric copies did not justify changing their defaults.

During the original prototype measurements below, production training code, checkpoints, and cache files were not changed. Only a reusable diagnostic harness and this report were added. The subsequent production implementation and verification are described at the end. The pc-xPass process tree was suspended with explicit user permission and resumed after each suite. Its CPU counters did not advance during the isolation checks; all eight workers were verified running and consuming CPU afterward. Total suspension time was 218 seconds, approximately 3.6 minutes. A separate resume watchdog protected both runs.

The reference was `saved/outcome_scoring/outcome_scoring_20261014T134452_571990_ade6798f/holdout`: relationship-attention, two layers, width 128, four heads, batch size 512, FP32, and an RTX 2080 SUPER (8 GiB). Python was 3.11.5, PyTorch 2.7.1+cu118, and PyG 2.7.0. torch-scatter, torch-sparse, and pyg-lib were absent in this environment.

Each trial processed the same 6,144 distinct samples, shuffled deterministically across four real training matches. There were 12 batches per trial and three repeats per configuration: 48 measured trials in the main suite and 30 in the follow-up. The model weights, fresh Adam optimizer, and random seeds were reset before every trial. One warm-up trial preceded each suite; case order was shuffled between repeats. These are short training-loop measurements, not convergence tests.

Timing used wall time with CUDA synchronization at the two boundaries. It includes graph reconstruction, collation, mandatory CPU graph validation, forward/backward/optimizer work, and final metric calculations. Initialization, model resets, and numerical comparisons are outside the timing. Interim console metric calculation was disabled equally for all cases. Throughput, rather than sampled GPU utilization, is the basis for the recommendations.

The main suite held four packed cache payloads in RAM while still reconstructing and batching samples on each pass. The comparison baseline below uses two CPU threads, selected by a preliminary 1/2/4/8-thread sweep. A subsequent eight-thread baseline had essentially the same median, so the thread sweep does not establish a robust reason to change the default.

| Main suite configuration | Median ms/batch | Observed range, ms/batch | Time saved versus baseline |
|---|---:|---:|---:|
| Current loader, monitoring off | 185.5 | 176.9-195.6 | +0.0% |
| Direct packed batching | 139.7 | 139.4-141.5 | +24.7% |
| Current loader + one-batch background prefetch | 158.2 | 149.2-161.1 | +14.7% |
| Direct + prefetch + deferred metrics + pinned nonblocking transfers | 139.3 | 133.7-141.1 | +24.9% |
| Deferred metric copies only | 184.9 | 174.7-219.4 | +0.3% |
| Pinned memory, blocking transfers | 212.4 | 210.2-213.9 | -14.5% |
| Pinned memory, nonblocking transfers | 204.1 | 204.1-205.2 | -10.1% |
| Extra collation diagnostics only | 213.4 | 207.0-225.9 | -15.1% |
| Per-batch telemetry only | 246.3 | 230.1-274.1 | -32.8% |
| Full monitoring and collation diagnostics | 295.9 | 282.1-342.5 | -59.5% |

A positive time-saved percentage means a shorter run; a negative value means overhead. Disabling full monitoring reduced measured time by 30% in the eight-thread comparison and 37% in the two-thread comparison. Per-graph collation checks and per-batch telemetry both contribute. Mandatory CPU validation before device transfer remained enabled in every case.

CPU-only reconstruction and collation took about 74-80 ms/batch with the existing loader, versus 24-26 ms/batch for the direct prototype: roughly two-thirds less preparation time. That translates to a smaller 23-25% saving in the training loop because GPU computation and validation still take time.

The follow-up used the normal eight CPU threads and reread the four cache files inside each timed trial. These files were warm in the operating-system cache, and each trial represented one four-match group. This does not measure cold storage reads or the full approximately 40 GB cache. The same training samples and full model/loss/optimizer path were retained.

| Follow-up training configuration | Median ms/batch | Observed range, ms/batch | Time saved versus baseline |
|---|---:|---:|---:|
| Current loader, monitoring off | 201.7 | 189.8-202.5 | +0.0% |
| Direct packed batching | 156.3 | 154.5-162.8 | +22.5% |
| Current loader + background prefetch | 171.9 | 161.4-172.1 | +14.8% |
| Direct packed batching + background prefetch | 175.5 | 168.0-180.8 | +13.0% |
| Direct + prefetch + nonblocking transfers, no pinning | 174.0 | 168.8-175.8 | +13.7% |
| Direct + prefetch + pinned nonblocking transfers | 186.6 | 180.3-194.9 | +7.5% |
| Direct + prefetch + deferred metrics | 181.1 | 172.9-182.7 | +10.2% |
| Current loader + telemetry every 50 batches, no extra collation diagnostics | 201.4 | 196.2-202.3 | +0.1% |

Adding background prefetch to direct batching increased its measured training time by 12%. The gains are not additive. Python-thread contention is a possible explanation, but this experiment does not prove the cause. Prefetch is useful as an independent option for the existing loader, not as a default addition to the direct reader.

Validation-mode processing of the same samples fell from 187.2 to 131.8 ms/batch with direct batching plus prefetch, a 29.6% saving. This was a throughput check on the selected training examples, not a new holdout evaluation. Direct batching alone was not isolated in validation mode, so the incremental validation benefit of prefetch is unknown.

The sampled telemetry prototype logged batch 0 and then every 50 batches, retaining the existing background resource sampler and mandatory CPU graph validation, but omitting the additional collation diagnostic wrapper. Its median was 201.4 versus 201.7 ms/batch with monitoring off. On a 12-batch trial this samples one in 12 batches, making the logging frequency higher than it would be over a full epoch. This supports separating lightweight progress/resource telemetry from full diagnostic monitoring.

Every one of the 12 direct batches was checked against PyG: tensors, labels, weights, object metadata, graph boundaries, and both batching bookkeeping dictionaries matched exactly. A separate deterministic CPU comparison of the real loss and two Adam updates matched all parameter values and prediction/target values exactly; reported metrics matched to an absolute tolerance of 1e-12. GPU predictions differed by at most 3.73e-8 across the trials. GPU weights were not bitwise identical, including repeated baseline trials (maximum parameter difference about 1.84e-4), so exact GPU optimizer-state equivalence is not claimed.

Training peak allocated GPU memory was approximately 2,894 MiB across the compared variants; validation was approximately 762 MiB. The tests do not show a need to increase batch size. The direct prototype is intentionally restricted to this run's homogeneous schema; it is not a general replacement for PyG collation.

The implementation priorities are:

1. Add a lightweight monitoring mode that samples training events and keeps background resource sampling, while retaining mandatory graph validation. Preserve full collation diagnostics as an explicit diagnostic option. Existing `--monitoring off` is the immediate way to avoid the measured overhead.
2. Implement a direct packed-batch reader with a tested PyG fallback for unsupported fields. Preserve sample/shuffle order, partial batches, labels, IPW, edge increments, metadata, cache validation, and a bounded memory footprint. Keep the existing packed cache format; consider preprocessing/cache identity carefully so a reader-only optimization does not unnecessarily force cache regeneration. The short-run evidence suggests approximately 20-25% less training-loop time with monitoring off.
3. Keep one-batch CPU prefetch optional for the current loader. Do not enable it automatically alongside direct batching based on these results.
4. Leave pinning, transfer flags, deferred metric collection, and CPU-thread defaults unchanged for now. They showed no clear additional gain or were slower in these trials.

The percentages describe the tested short loops. Full-epoch speed depends on storage behavior, other matches, metric/reporting costs, and system load; the savings should not be added together or treated as guaranteed epoch forecasts. At the end of the prototype measurements, no production optimization had been applied.

Benchmark harness: [benchmark_training_pipeline.py](../scripts/diagnostics/benchmark_training_pipeline.py). Local raw results: [main suite](../out/training_pipeline_benchmark/isolated_20261015/results.json), [follow-up](../out/training_pipeline_benchmark/followup_20261015/results.json), and [CPU correctness](../out/training_pipeline_benchmark/correctness_20261015/results.json). Suspension and resume audit files are alongside the corresponding results.

To reproduce after competing workloads have been stopped, run the harness with the saved run path, a new output directory, `--steps 12 --repeats 3 --contention isolated`. Add `--suite followup` for the second suite, or `--correctness-only --steps 2` for the CPU equivalence check. The harness itself does not suspend other workloads and refuses GPU measurements while a `train.py` process is active.


## Production implementation verification

Implemented `--batch-loader auto|pyg` (new-run default `auto`) and `--monitoring off|light|on` (default `off`). Prepared disk datasets use direct tensor batching in auto mode; unsupported schemas fall back per batch. Memory-backed data and full monitoring use PyG. Cache format and preprocessing fingerprints are unchanged. Older argument records retain PyG on resume unless explicitly overridden, subject to the existing source/code fingerprint checks.

The new `scripts/diagnostics/benchmark_packed_loader.py` runs the production reader and `run_epoch`, including warm cache file reads, mandatory CPU validation, device transfers, ordinary interim and final metrics. It checks existing cache checksums and source signatures without preparing or modifying caches. Each configuration resets weights, Adam, and random seeds. Case order is randomized, CPU threads remain eight, and CUDA synchronization occurs at timing boundaries only. Initialization and numerical comparisons are excluded.

An initial production comparison repeated 12 batches across four matches, three times per configuration. A follow-up repeated 50 batches (25,600 examples) across 20 selected matches, three times per configuration, to exercise multiple match buffers and the normal one-in-50 sampling cadence. Training uses shuffled four-match buffers; validation uses sequential one-match buffers. Validation examples are selected training examples for throughput measurement, not a holdout assessment. Both checks were isolated by pausing pc-xPass with a watchdog and resuming all ten processes afterward. Neither launched a complete training run.

Median production results:

| Comparison | Existing PyG, off | Packed, off | Packed, light | Packed time saved vs PyG |
|---|---:|---:|---:|---:|
| 12 batches, training | 212.4 ms | 170.0 ms | 179.0 ms | 20.0% |
| 12 batches, validation | 196.7 ms | 174.8 ms | 168.6 ms | 11.1% |
| 50 batches, training | 218.9 ms | 170.6 ms | 167.5 ms | 22.1% |
| 50 batches, validation | 231.0 ms | 161.9 ms | 168.5 ms | 29.9% |

The 50-batch run met the 15% training-time reduction target: 22.1% less time, with throughput increasing from 2,339 to 3,002 samples/second. Validation used 29.9% less time. Loader wait (including file reads) fell from 5.69 to 2.78 seconds for training and from 5.21 to 3.15 seconds for validation. No measured packed batch required schema fallback.

The initial short trial showed 5.3% extra training time with light monitoring. Its child sampler was still performing its first asynchronous resource/GPU query during the timed epoch. The follow-up waited for that initial sample before timing, matching ordinary training, which starts monitoring before dataset preparation. At 50 batches, light monitoring was 1.8% faster than off in training (timing variation, not a claimed optimization) and 4.1% slower in validation. This supports the sampled mode while retaining monitoring off as the default.

Peak GPU allocations were identical between loaders: approximately 2,893 MiB during training and 762 MiB during validation. Sampled process RSS peaks were roughly 1.2-1.5 GiB, with variation from allocator reuse and case order; this does not establish a full-split RAM estimate. Storage-release tests independently verify that old match buffers are released, including partial batches spanning buffers.

Every selected production batch matched PyG tensors, labels, weights, metadata, slice dictionaries, and increment dictionaries exactly. CUDA predictions differed by at most 4.1e-8 across the two production suites. After 50 training updates, parameter differences relative to the first training trial reached about 2.45e-4; exact CUDA optimizer replay is not claimed. Deterministic CPU tests check exact predictions, dropout RNG, parameter updates and Adam state, and actual entrypoint tests check interruption/resume with both loader selections and all monitoring modes.

Raw production trials and suspension/resume records are under `out/training_pipeline_benchmark/production_20261015/` and `out/training_pipeline_benchmark/production_long_20261015/`. Total pc-xPass suspension for these implementation checks was 293.0 seconds. All eight workers were verified running and accumulating CPU time afterward. These are warm-cache short-run measurements, not cold-disk or full-epoch guarantees.


Reproduce the production comparison after competing workloads have been stopped:

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/benchmark_packed_loader.py `
  --run saved/outcome_scoring/outcome_scoring_20261014T134452_571990_ade6798f/holdout `
  --output out/training_pipeline_benchmark/production_recheck `
  --matches 20 --steps 50 --repeats 3 --contention isolated
```

The benchmark does not suspend competing workloads itself. It reads a trusted
existing run and writes only diagnostic results under the chosen output directory.


Implementation validation: **271 tests passed** across packed batching/packing,
prepared datasets, graph diagnostics, runtime controls, resource monitoring,
training controls, source refits and IPW preparation. This includes real CPU and
CUDA training/checkpoint-resume comparisons, legacy missing-loader arguments,
explicit loader overrides, exact CPU Adam/dropout replay, fallback errors and
match-storage release. `git diff --check` also passed.
