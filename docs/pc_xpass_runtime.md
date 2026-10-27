# PC-xPass runtime scheduling

Optional [aerial candidates](pc_xpass_high_pass.md) reuse immutable 25 Hz flight
profiles per worker. `--high-pass on` adds elevations 15/20/25/30 degrees and
20..30 m/s total speeds by default; the aerial step inherits `--speed-step`.
Lane interception defaults to a 2 m height cutoff; endpoint control uses 0.5 m.
Measure the additional player/time
queries with `scripts/diagnostics/benchmark_pc_xpass_high_pass.py`.

PC-xPass generation uses one lazy, reusable compute session per dataset. Sportec
and SkillCorner keep match publication atomic; Hawkeye combines situations into
compute windows while retaining complete situation outputs; Benchmark retains
its modification/state outputs. Preparation and computation are sequential by
default; Sportec supports opt-in bounded preparation overlap.

## Controls and compatibility

Use `--num-workers 8 --physical-batch-size 16 --worker-thread-limit 1` with an
optional dataset window:

```powershell
--sportec-runtime-row-window 256
--skillcorner-runtime-row-window 256
--hawkeye-runtime-row-window 256
--benchmark-runtime-row-window 256
```

Each flag configures its own dataset. The automatic value is batch size times
resolved workers times two. The Sportec and Hawkeye flags require `--pc-xpass`.
Explicit `--sportec-runtime-match-window` is rejected with PC-xPass; it retains
its old meaning for ordinary physical-xPass generation.

Input buffering counts eligible states, including cache hits, and can exceed the
window when a single logical group is large. Computation submits at most the
configured number of rows at once. Tasks retain the numerical batch size unless
the explicit row budget is smaller. Complete groups are collected before the
existing publication boundary. Prepared graphs and accumulated match results are
not subject to this queue budget.

A cold workload with only one available task runs serially. An established pool
is reused for tails; empty, dry-run, cache-hit-only, and single-worker workloads
create no pool. Failures settle outstanding work before serial recovery or
transaction cleanup. Scheduling settings are excluded from numerical fingerprints.
This scheduler needs no new cache format or rebuild of already compatible caches.

## Sportec preparation overlap

Add `--sportec-runtime-prefetch` to PC-xPass generation to run one ordered
preparation thread alongside the existing numerical pool. Default: off. The
thread exclusively owns match loading, reconstruction, and graph preparation.
The main thread owns source registration, cache reads/writes, result handling,
transactions, and publication. Workers use the spawn process context when
prefetch is enabled. Dry runs and single-worker execution stay sequential.

The producer needs permission **before constructing** the next window. There is
at most one window ahead, including construction in progress, alongside the
current consumer window. Complete period/possessor groups remain intact, so one
oversized group can exceed the row target. This is not a byte limit: tracking,
the reconstruction plan, and accumulated match outputs are separate allocations.
The outstanding numerical row cap and physical task size still apply.

Graphs and descriptors transfer to the consumer without later producer mutation.
Reports and manifests arrive after preparation finishes. A match is published
only after the preparation thread has joined, before the next match is loaded.
On a batch failure, the numerical session settles outstanding tasks and the
producer joins before serial recovery. Already prepared work is retained, and
remaining preparation resumes sequentially for that match. Cancellation waits
for the in-progress preparation window to finish; it cannot forcibly interrupt
a graph-building operation. Failed matches retain their previous published cache.

`scheduling.preparation` records each match's preparation work, initial/total
consumer wait, producer backpressure, preparation/compute interval intersection,
and maximum prepared window rows. Existing `preparation_seconds` window fields
remain gaps before consumption; they are not producer work when overlap is on.
Do not add overlapping work durations to derive elapsed time. Other datasets'
preparation is unchanged. Prefetch and row-window options do not affect numerical
fingerprints or require a fresh compatible cache.

Run the isolated, fully warmed comparison with:

```powershell
.venv/Scripts/python.exe scripts/measure_sportec_runtime.py --budget-seconds 900
```

The diagnostic uses Sportec match `DFL-MAT-J03WEL` and feature run
`feature_20260910T110759_053479_8b508137` by default; override them with
`--match-id` and `--feature-run-id`. It requires psutil for memory sampling.
Three cases use actual generation and match publication: sequential 256,
sequential 512, then prefetch at the better window (256 within a 5% difference).
Every case has a fresh temporary cache and eight fully warmed workers with
16-state tasks. Output comparisons include exact values, dtypes, NaNs, identities,
lane rows, manifests, associations, selection reports, and coverage. Progress
goes to stdout and incremental `results.json` under `tmp/pc_prefetch_<timestamp>`.
Production cache versions and latest pointers remain untouched. The budget
includes imports, warm-up, verification, and shutdown; incomplete cases are
reported explicitly.

## Fully warmed preparation comparison (alternative baseline)

The final comparison used completed feature run
`feature_20260827T004621_803230_f03ded3f` and available match `DFL-MAT-J03YDU`.
The originally requested feature run was being extended and did not pass its
usability guard; the completed alternative lacks the original test match.
These absolute times are therefore a different baseline from the probes below.

All three cases ran actual generation with fresh temporary caches and eight
fully warmed workers, 16-state tasks, and one thread per worker. They published
identical 2,100 physical states and 231,000 lane rows, with exact equality of
values, NaNs, dtypes, identities, manifests, associations, selection reports,
and coverage. There were no retries or pool discards. Existing caches and
production latest pointers remained untouched.

| Case | Full runner time | Reduction from sequential 256 |
| --- | ---: | ---: |
| Sequential 256 | 208.31 s | Reference |
| Sequential 512 | 203.57 s | 2.28% |
| Prefetch 256 | 182.52 s | 12.38% |

The 512 setting improved worker-slot occupancy from 84.8% to 93.0% and reduced
numerical dispatch from 106.87 s to 100.22 s. Its whole-run improvement stayed
below the agreed 5% threshold, so the prefetch test retained 256.

Prefetch overlapped 44.19 s of preparation with numerical dispatch. Consumer
waiting after the initial window totaled only 0.035 s. Numerical dispatch stayed
near 107 s, while preparation wall time increased from 67.65 s to 93.60 s and
cache scans increased from 8.44 s to 33.45 s. Concurrent parent work therefore
reduced the benefit relative to the theoretical ceiling; the timings do not
isolate the cause of this contention. Overlapping phase durations must not be
summed as sequential costs.

The one-ahead-window bound held; the largest prepared window was 364 states at
the soft 256 target. Sampled parent/worker-total peak RSS was 6.92, 7.12, and
7.29 GiB respectively. The harness retained verification tables between cases,
so RSS differences cannot be attributed entirely to the scheduling changes.

**Provisional recommendation: keep 256 and add `--sportec-runtime-prefetch`** for
comparable Sportec generation. Defaults remain unchanged. This is one comparison
per case on one match; confirm behavior during the next normal multi-match run
before extrapolating. Other datasets remain untuned.

The completed comparison, including warm-ups and verification, took 688.10 s;
including guarded preflights, the measurement total was about 712.56 s, within
the 900-second budget. No repeats were launched. Raw results, derived metrics,
and the detailed report are under `tmp/pc_prefetch_20261013T180229`.

Reproduce this baseline with:

```powershell
.venv/Scripts/python.exe scripts/measure_sportec_runtime.py --budget-seconds 900 --feature-run-id feature_20260827T004621_803230_f03ded3f --match-id DFL-MAT-J03YDU
```

## Measurements on 2026-10-13

Recorded real-data performance phases totaled **297.6 seconds**, including initial
preparation, the reference, warm-ups, comparisons, resume, and pool shutdowns.
Parent interpreter/module imports and automated test time are outside that total.
All outputs went to `tmp/pc_runtime_20261013`; existing caches and latest pointers
were left intact. Raw records are in `baseline.json` and `optimized.json` there.

The sample used Sportec match `DFL-MAT-J03WEL`, feature run
`feature_20260910T110759_053479_8b508137`, actions scope, eight workers, 16-state
tasks, and one thread per worker. Numerical settings included speeds 3..25 in
steps of 2, angle step 3, radial grid size 1, distance-based reaction time,
maximum player speed 5, top-pass 5/10, top-N 10/25, xT ranking, and lane export.
No pass-height model was used.

Physical rows, dtypes, NaNs, identity columns, and lane-control rows matched
exactly between the original and optimized 64-state reference. The 512-state
window candidates also matched exactly. These are logical comparisons, not
byte-for-byte Parquet comparisons.

| Comparison | Original pools per window | Reusable pool |
| --- | ---: | ---: |
| Two 32-state windows, 64 identical states | 21.04 s | 13.18 s |
| Pool creations | 2 | 1 |

This is a **37% reduction for that small-window probe**, including pool lifecycle
costs. It is not an estimate of full-match or full-dataset speedup. The measured
startup-to-first-task span was about 7.5 seconds.

The larger comparison used 512 identical, already prepared states and isolated
fresh physical caches for each run:

| Row window | Tasks per full window | First run | Reverse-order repeat |
| --- | ---: | ---: | ---: |
| 128 | 8 | 35.51 s | Not run: budget exhausted |
| 256 | 16 | 35.46 s | 36.14 s |
| 512 | 32 | 35.68 s | 35.15 s |

Every candidate computed 32 tasks of 16 states in total. Peak outstanding rows
respected each configured budget. Scan time was 1.2..1.9 seconds per candidate;
write time was 1.6..2.5 seconds; parent result handling was about 0.05 seconds.
These measurements use the existing partition-file writer, rather than an actual
match transaction, so publication costs cannot be extrapolated to complete matches.

**Warm-up limitation:** each pool was initially given only two tasks. On Windows,
additional processes started when larger windows were submitted. The initial
measured window therefore includes further worker startup, despite reusing an
executor. Overall times are comparable as startup-inclusive runs, but do not
establish an optimal steady-state window.

Later windows reached approximately **95.6%..98.4% task occupancy**. Their final
16-state task drain lasted approximately 5.2..5.9 seconds, but aggregate unused
worker capacity during that drain was only about 0.5..1.9 worker-seconds. A long
final task alone therefore does not imply substantial imbalance. For the same
last 256 states, the 128-state setting took 14.04 seconds versus 13.40 seconds
for the first 256-state run: within the chosen 5% band, with insufficient repeat
evidence to select the smaller setting confidently.

**Recommendation: retain the automatic 256-state window.** No larger window
showed a clear improvement. The smaller setting is a provisional candidate only.
Other datasets have automated scheduling coverage; their real-data window sizes
were not measured within this shared budget and remain untuned.

A 512-state cache reuse check took 7.81 seconds with **zero physical misses and
zero compute tasks**. Existing auxiliary-data handling classified these rows as
copies/refreshes and rewrote them; this is not a zero-I/O cache-hit measurement.
The scheduler preserves that behavior.

## Preparation overlap and next step

Initial match loading, reconstruction, and graph preparation took 12.88 seconds
before the numerical probe. The adapter prepared whole partitions (550 graphs)
and the sample retained 512 states. The first 45-state partition was available
at 6.14 seconds; later partition preparation took another 6.64 seconds. This
shows preparation cost, but does not measure two complete steady-state
preparation/compute windows with real match publication.

Do not implement preparation/computation overlap from this evidence alone.
Retain the 256-state default and use the new diagnostics on a bounded follow-up:

1. Warm the executor with at least eight numerical tasks and check
   `worker_processes_used`; exclude remaining startup from steady-state analysis.
2. Compare the same inputs at 128/256/512 rows, reversing the order for the leader
   and 256. Recommend the smallest setting within 5% of the fastest confirmed
   throughput. Treat unconfirmed settings as provisional.
3. Observe at least two steady-state preparation/compute windows through the
   actual dataset runner, with temporary output and complete match publication.
   Keep this follow-up short; report uncertainty if the budget is insufficient.
4. Calculate the optimistic pipeline opportunity as
   `min(eligible preparation time, parallel compute time) / steady-state elapsed`.
   Include only preparation compatible with the retained publication boundaries.
   Exclude initial preparation and startup from the steady-state comparison.
   Writes, commits, and serial recovery belong in elapsed time, but must not be
   counted as overlapping preparation or parallel computation.

If that opportunity is at least 20% across at least two measured windows, prepare
a separate bounded producer/consumer plan. Otherwise retain sequential
preparation and computation, and target the largest measured remaining cost.
The high later-window task occupancy currently gives little evidence that
larger queues alone will materially improve compute utilization.

## Validation

Focused tests cover all four runners, lazy creation, reuse across windows and
matches, serial execution, bounded oversized groups, small budgets, hit/dry-run
workloads, resume, failures, interruption, pool replacement, publication
boundaries, and exact physical/lane-control equality. Existing possession,
cache, resume, auxiliary, and physical-model tests also pass: **638 tests and
25 subtests**, in 44.3 seconds. Real Windows spawn,
reuse, and shutdown were exercised by the bounded numerical probe.
