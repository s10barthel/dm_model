# PC-xPass runtime scheduling

PC-xPass generation uses one lazy, reusable compute session per dataset. Sportec
and SkillCorner keep match publication atomic; Hawkeye combines situations into
compute windows while retaining complete situation outputs; Benchmark retains
its modification/state outputs. Preparation and computation remain sequential.

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
