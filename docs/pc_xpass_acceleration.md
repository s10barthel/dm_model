# Acceleration-constrained pc-xPass

Acceleration is optional and applies only to pc-xPass. The original time-arrival
margin is described here; [distance control](pc_xpass_distance.md) uses separate
exact-time reachable-region tables.
Omitting `--accel`, or supplying `--accel off`, preserves existing movement and
legacy fingerprints. Original physical xPass and empirical reachability do not
use this model.

## Prepare a movement table

~~~powershell
python scripts/prepare_pc_xpass_acceleration.py --table-id accel_v1 --accel 7 --max-player-speed 5
~~~

Maximum vector acceleration is in m/s²; player speed is in m/s. Attack/defence
speed overrides are accepted. Default `--max-distance 150` covers straight-line
player-to-target separations, not trajectory lengths; it does not exclude players.
Preparation is an explicit, potentially expensive one-time operation. It reports
progress and resumes interrupted preparation with identical settings. `--workers`
controls independent solver processes.

Completed artifacts live under `data/pc_xpass/acceleration/<table-id>/`. They are
immutable and checksummed. Changed preparation settings require a new ID.
`--max-cells` (default 20000 tree nodes) and `--max-depth` (default 24) bound
refinement, reserving capacity for refining independent-holdout failures. Known
failed probes remain validation cases in child cells. Unsuitable cells explicitly
use reference-solver fallback; they do
not contain trusted interpolation. Reports include their count. Many fallback
queries can make generation very slow: measure frequency before a large run.

To refine an existing table without changing it, reuse its preparation state:

~~~powershell
python scripts/prepare_pc_xpass_acceleration.py --table-id accel_v1_refined --source-table-id accel_v1 --accel 7 --max-player-speed 5 --max-cells 22000
~~~

The source must have matching physical preparation settings and retain its
checkpoint and reference.sqlite files. Increase the node budget above its current
node count. Completed source artifacts remain immutable.

## Generate predictions

~~~powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id pc_accel7_v5 --accel 7 --accel-table-id accel_v1 --max-player-speed 5 --reaction-time 0
~~~

Append normal dataset, pass-grid, scoring and worker settings. Team speed caps
override the common cap. Ball launch-speed flags retain their meaning. Reaction
time defaults to zero with acceleration. Explicitly nonzero/distance reaction
time, `--dist-pass-*`, and `--margin reachability` are rejected in acceleration
mode. Hawkeye location/freeze generation accepts the same movement flags.

New acceleration runs require an explicit prepared table ID. Generation and dry
runs never prepare tables automatically. A resumed prediction cache inherits its
settings under the existing normal/location version-selection rules. Missing
acceleration metadata means off. Cache-only inference needs neither the movement
table nor the solver: it consumes stored predictions.

## Physical and numerical definitions

Players retain their full tracked velocity, with magnitude clipped to the cap.
The solver minimizes time to a stationary point with dr/dt=v, dv/dt=u,
||u||<=a and ||v||<=V. There is no reaction delay, terminal stopping requirement,
body orientation, obstacle, or pitch-boundary path constraint. The default
interception radius is zero; optional disk arrival is described below.
Braking and turning share the vector acceleration budget. This reconstructs
Spearman et al. (2017)'s constraints/objective, not an unpublished implementation.
Acceleration 7 m/s² is an experiment setting, not a verified value from that paper.

Reference arrival bounds must differ by at most **10 ms**, with terminal residual
at most **1 cm**. Residual tolerance is not a physical interception disk. Initial
velocity, speed and acceleration are checked. Meshes start at 32 constant-
acceleration intervals and double as needed. Exact analytic cases need no mesh.
Clarabel solves fixed-time trajectory problems. Feasible paths give upper bounds;
continuous-time costate support bounds give lower bounds, including paths between
mesh points. The time search permits nonmonotonic exact-time reachability. Mesh
agreement alone is insufficient. Unresolved results raise errors rather than
providing guessed arrivals.

## Normalization and validation

Position units are V²/a; time units are V/a. The table represents speed ratio,
target angle relative to initial velocity, and log1p of normalized distance.
Adaptive rectilinear cells interpolate the correction to radial travel time.
Polar coordinates make distant regions cheaper to represent than a uniform
Cartesian grid without changing the motion model.

Preparation settings set normalized coverage and precision. Later generation
settings may differ if scaled tolerances remain valid; both team caps are checked.
Prediction caches retain their actual acceleration and speed settings. Sharing a
table does not make predictions interchangeable.

Interpolation is checked against separate reference solves at deterministic
interior and face probes, followed by independent seeded holdouts, to a
**10 ms physical error target** at preparation
settings. This is sampled validation, not an error proof at every interior point.
Reference and interpolation errors are separate budgets, not a combined 10 ms
guarantee. Stationary players and coincident targets use analytic solutions.
Outside coverage and in failed validation cells, generation uses the reference
solver; it never extrapolates or drops distant players. Statistics report
acceleration_queries, acceleration_fallbacks and acceleration_fallback_seconds.

## TTA with a blocking radius

`--margin tta --blocking-radius R` changes raw control to use the earliest time
the player's tracked centre can reach the closed disk of radius R metres around
each target. It applies to **all players**, for both lane and endpoint raw control.
Lane aggregation, endpoint competition, xT ranking and averaging retain their
existing definitions. Radius defaults to zero, preserving old results, metadata
fingerprints and cache reuse. The option is pc-xPass only and conflicts with
distance and empirical-reachability margins.

For a positive radius, omitted lane/control temporal offsets default to **zero**.
Explicit `--lane-inflection-point` and `--control-inflection-point` values are
respected; specifying 0.2 would retain a temporal allowance in addition to the disk.
Resumed prediction caches inherit their recorded offsets and radius. Power 15
retains its units of inverse seconds. Raw control is
`sigmoid(power * (ball_time - disk_arrival_time + offset))`.

With zero offset, 50% means disk arrival coincides with ball arrival. A player
initially in/on the disk has arrival time zero, including when moving away. This
is an **earliest-contact model**, not guaranteed occupancy at the later ball time.
Terminal stopping is not required. A radius is distinct from spatial uncertainty.

Prepare a separate bundle for accelerated disk arrival:

~~~powershell
python scripts/prepare_pc_xpass_acceleration.py --table-id pc_disk_v1 --margin tta --accel 7 --max-player-speed 8 --blocking-radius 1 --max-cells 22000
python scripts/generate_physical_xpass.py --pc-xpass --accel 7 --accel-table-id pc_disk_v1 --max-player-speed 8 --blocking-radius 1 --reaction-time 0 --lane-inflection-point 0 --control-inflection-point 0
~~~

A bundle contains a three-dimensional slice for each distinct normalized radius
`accel * R / V^2` needed by the attacking/defending speed caps. Preparation uses
the same speed flags as generation. The slice coordinates are initial speed ratio,
relative target angle, and log1p of normalized distance **beyond the disk**. Inside
the disk is handled exactly. The normal 150 m player-target coverage is retained;
targets outside table coverage and unsuitable cells use explicit reference solving.

`--source-table-id` can refine a bundle into a new ID with matching preparation
acceleration, caps, radius and coverage. Point tables and positive-radius bundles
are not interchangeable. Other physical settings may reuse a bundle only if all
normalized radii match and scaled time/position tolerances still pass. Cache
identity records the actual physical radius and bundle checksum. Cache-only
inference does not load the bundle or optimizer.

The disk reference solver minimizes arrival time over all admissible trajectories,
including sideways movement and reversal. It does not subtract R/V from point TTA
or shrink the target along the initial radial line. Fixed-time trajectory residuals
and continuous support bounds certify disk intersection/exclusion. Numerical
acceptance remains a 10 ms arrival bracket and at most 1 cm terminal excess beyond
the disk. Independent interpolation validation retains its separate 10 ms target.
Unresolved probes never become trusted interpolation entries.

With acceleration off, a positive radius also works under the existing fixed
initial trajectory during reaction, then instantaneous capped-speed movement.
Contact during that reaction segment is detected analytically. Radius zero retains
the original reaction-time calculation exactly.

Runtime looks up disk TTA once per player and target and reuses it across ball
launch speeds. Repeated target coordinates, including the common origin of all
polar rays, are evaluated once and expanded back to their original locations.
No radius dimension is searched during normal prediction generation.

Benchmark numerical compute, fresh temporary prediction-cache writes and independent
reference checks using:

~~~powershell
python scripts/diagnostics/benchmark_pc_xpass_blocking.py --point-table-id accel7_v8 --radius-table-id pc_disk_v1 --blocking-radius 1
~~~

The optional `--hawkeye-graphs out/acceleration_lane_investigation/graphs.pt` adds
the two previously investigated Hawkeye scenes to numerical timing/comparison.
Input loading and one-time table preparation are excluded from generation timings.

The local example bundle **accel7_v8_radius1_refined** is ready for a=7 m/s²,
V=8 m/s and R=1 m. Preparing its source and refining it took 300.3 s in total
with eight workers. Its compressed lookup files occupy 905,029 bytes. It contains
22,000 leaves; unvalidated leaves use reference solving. Of 22,224 independent
holdout probes, four leaves failed and were retained as explicit fallback rather
than trusted interpolation. Validation is sampled, not an exhaustive error proof.

An additional 48 real-state queries were checked with reference brackets tightened
to 1 ms solely for the validation measurement. Their largest lookup deviation from
either reference endpoint was 7.61 ms, within the 10 ms interpolation target. Normal
reference solving still uses the agreed 10 ms bracket and 1 cm terminal-excess limits.

Final local timing (26 October 2026), a=7, V=8, R=1, three repetitions and one
computation worker:

| Measurement | Point TTA, 0.2 s offsets | Disk TTA, zero offsets |
|---|---:|---:|
| Numerical computation, 12 benchmark states + two Hawkeye scenes | 3.869 s | 3.904 s |
| Fresh cache creation, 12 loaded benchmark states | 3.715 s | 3.653 s |

The measured numerical overhead is about 0.9%; cache-write timings are comparable.
This is a small-state benchmark, not a whole-season runtime guarantee. Cached
prediction reuse took about 1.00 s in both modes. The disk run had 60 reference
fallbacks across 4,604,666 unique player-target evaluations including warmup and
repetitions; fallback solving took 0.012 s in total. Scale-aware machine-precision
contact comparisons prevent numerical roundoff at the closed-disk boundary from
triggering expensive solves. They do not add a centimetre-scale physical radius.

The two investigated Hawkeye top-pass-10 values change from 0.902 to 0.511
(right-back, time_norm 0) and 0.886 to 0.856 (CM, time_norm 1). Both radius and the
removal of temporal offsets participate in that comparison. The diagnostic also
records a point-TTA variant with zero offsets to separate their effects.

Numerical results, temporary fresh-cache timings, movement settings and the 48
independent arrival checks are saved in
`out/pc_xpass_blocking_benchmark_ready.json`.

## Benchmark

~~~powershell
python scripts/diagnostics/benchmark_pc_xpass_acceleration.py --accel-table-id accel_v1 --accel 7 --repeats 3
~~~

Uses the existing 12 real benchmark states, 3°/1 m grid, ball speeds 3–25 m/s,
and top-pass 10 with xT ranking. Compares distance-based reaction, zero reaction
without acceleration, and acceleration. Preparation, numerical compute time,
peak RSS and fallback time are separate. Graph loading and prediction-cache I/O
are excluded from numerical repetitions. Add
`--hawkeye-id 0291ae3b-9848-4f16-ab2e-2dc194ed8649` for that local BallReceipt state.



## Local measurements, 26 October 2026

With a=7 m/s?, V=5 m/s, all players, 150 m coverage, and the documented pass grid:

| Measurement | Existing model | Acceleration | Increase |
|---|---:|---:|---:|
| Fresh prediction-cache creation, 12 benchmark states | 2.788 s | 3.802 s | 36.4% |
| Numerical computation, 12 benchmark states plus the specified Hawkeye scene | 2.428 s | 3.529 s | 45.4% |

These are medians of three repeats with one computation worker. The cache-creation
measurement includes metadata/Parquet writes but excludes loading graph inputs.
Reusing already computed prediction rows took approximately 1.04 s in both modes.
These small-sample measurements are not a whole-season runtime guarantee.

The prepared example table **accel7_v5_refined** is available locally for immediate
use with those movement settings. Building its source and refining the independent
holdout failures took 427.6 s in total with 12 preparation workers. The compressed
lookup data occupies 428,657 bytes. It has 10,061 leaves, of which 4,911 explicitly
use fallback; those difficult cells mostly cover rarely queried regions. The
13-state benchmark used 4,928 reference fallbacks across 4,365,262 queries
(including warmup), taking 0.354 s in total. Many fallbacks have exact analytic
solutions. Leaf counts alone do not predict runtime fallback cost.

The refined table passed 10,300 final independent holdout checks. Another 48
queries sampled from real benchmark states had maximum error 5.43 ms outside the
reference bracket, within the 10 ms interpolation target. Validation is sampled.

Local reports are in out/pc_xpass_acceleration_refined_benchmark.json and .csv,
out/pc_xpass_acceleration_cache_benchmark.json, and
out/pc_xpass_acceleration_validation.json. The last report validates the refined
movement table on the 48 independent real-state queries.
