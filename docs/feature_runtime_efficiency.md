# Feature generation and refresh efficiency

`scripts/generate_relevant_features.py` uses `--num-workers <N|auto>` for
stop-condition refreshes, legacy xT and goal-distance/progression refreshes,
trajectory and failed-pass refreshes, reception retirement, carry replacement,
and graph edge extensions. Generation and label-only subprocesses retain their
match-level parallelism. The default is `1`. `auto` retains the existing policy
`max(1, min(6, cpu_count - 2))`; the worker runner caps the pool at the task count.

Spawned workers inherit native thread limits before importing numerical
libraries, then apply the requested PyTorch and BLAS/OpenMP limits. Workers
receive immutable configuration once, and only a bounded number of tasks is
submitted. Results and persisted reports are assembled in input order.
Reception retirement and standalone carry replacement accept `auto` too.

Independent target-refresh matches and graph-extension files are staged by
workers. The parent registers temporary paths, waits for workers to stop before
cleanup, retains input and output validation checkpoints, and publishes staged
files in canonical order. Existing resumable refreshes retain their per-match
commit and recovery rules. Parallel execution does not combine separate stages
or change the order of dependent edge extensions.

## Reusing work across return types

Graph generation already batches return types and reuses graphs and resolved
receivers. It additionally reuses return-independent label rows within one
receiver/action variant. Current action data and configuration must match the
cached context; otherwise the original row construction runs again. Return
calculations, action filtering, fixed diagnostic targets, and match-state updates
retain their existing definitions. Caches end with the variant and never become
an authoritative source for a later run.

Success-intent output still uses the first requested return type. Secondary
types are calculated and validated without retaining full unused tensors when
the common label rows can be reused. Changing the order of requested return
types can select a different primary artifact, just as before.

Stop-condition target construction validates the action-to-event mapping once
per calculation and reuses it across target families. Each family's source
columns and finite values are still checked. An already loaded canonical label
template is reused rather than reading the same file a second time.

Tracking is still loaded for label-only stages. Existing saved labels do not
prove that current tracking-dependent fields are unchanged. Similarly, the final
stop-condition refresh still establishes the canonical label layout, updates
resolved-action targets, and produces its reports. Skipping these operations
would require additional provenance or could change persisted artifacts.

## Exactness and benchmarks

Optimizations must preserve exact tensor values, graph attributes, table values,
dtypes, indexes, ordering, missing-value positions, artifact coverage, and
semantic metadata. Timing, worker diagnostics, progress ordering, and serialized
file bytes may differ. Source artifacts remain protected by atomic replacement
of complete files, including when derived artifacts share hard links.

The diagnostic below compares optimized paths with sources captured before the
change (or with a Git revision). It uses real computation and process workers
on isolated synthetic matches, verifies every data artifact and semantic report,
and records cold/warm timings and sampled aggregate process RSS. Shared memory
pages count once per process in the RSS totals.

```powershell
.venv\Scripts\python.exe -B scripts/diagnostics/benchmark_feature_efficiency.py `
  --baseline-ref <revision-before-change> `
  --output tmp/feature_efficiency/comparison `
  --actions 500 --refresh-actions 1500 --matches 8 `
  --workers 1 2 4 8 --repeats 3
```

Alternatively, use `--baseline-sources <directory>` with captured source files
under their repository-relative paths. The output directory must be new.
The built-in label cases use one, three, and six return types. Other return
orderings must be compared against their corresponding baseline, because the
first return type intentionally selects the primary artifact.

Small refresh batches can take longer with multiple workers because importing
numerical libraries and starting processes outweigh the computation saved.
Benchmark representative workloads before choosing a worker count. Synthetic
benchmarks do not establish production storage throughput or a full-run ETA.

## Validation on 5 November 2026

The targeted and adjacent regression runs passed 744 test cases. Six additional
failures reproduced with the captured pre-change sources from revision
`f7951ca42bd2e546fe86483421a2bdb4772451c2` in the same environment:

- Two synchronized-pass tests: a fixture assigns three values to two rows, and
  an expected duration-safeguard exception is not raised.
- Two graph-validation forwarding cases require an explicit xT version.
- Two success-intent CLI cases use feature metadata without the required
  stop-condition definition.

Those pre-existing behaviors and tests were not changed by this optimization.
New coverage includes actual worker processes and thread limits, numeric/auto
selection, serial/parallel equality, source hardlink preservation, worker crash
cleanup, edge extension equality, cache invalidation, and secondary target
validation. A single return type does not allocate the label-row cache.

Captured-source comparisons verified exact label values, action tables, artifact
membership, and semantic reports for legacy xT, goal-distance/progression, and
stop-condition refreshes with 1, 2, 4, and 8 workers. The final synthetic
stop-condition workload used 16 matches with 500 actions each:

| Implementation | Wall time | Peak aggregate RSS |
|---|---:|---:|
| Pre-change, one worker | 51.7 s | 668 MiB |
| Optimized, one worker | 45.0 s | 671 MiB |
| Optimized, two workers | 31.2 s | 1,950 MiB |
| Optimized, four workers | 27.5 s | 3,225 MiB |
| Optimized, eight workers | 28.1 s | 5,784 MiB |

Multi-return label construction was approximately 15% faster in the final
250-action synthetic comparison. Raw results are retained locally under
`tmp/feature_efficiency/comparison_stop_final/benchmark.json`; earlier legacy
refresh comparisons are under `tmp/feature_efficiency/comparison_large/`.
Production-data benchmarking was blocked by read permissions on the source
Parquet files.
