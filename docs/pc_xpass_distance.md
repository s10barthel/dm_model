# Acceleration-based distance control for pc-xPass

`--margin distance` changes raw control to use signed Euclidean distance from a
ball target to the player's reachable region **at the ball's arrival time**.
Acceleration must be enabled. Lane aggregation, endpoint competition, pass
ranking, top-option averaging, and position discount retain their definitions.

For signed distance d in metres (positive outside, negative inside), raw control is
`sigmoid(distance_power * (distance_allowance - d))`. Defaults are power 3 per
metre and allowance 1 metre: 50% control at a 1 m shortfall, approximately 95.3%
at the reachable boundary, and approximately 4.7% at a 2 m shortfall. This is raw
control, before aggregation and competition. The same mapping applies to every
player and to lane and endpoint raw controls.

## Prepare a distance movement table

~~~powershell
python scripts/prepare_pc_xpass_acceleration.py `
  --margin distance `
  --table-id distance_v1 `
  --accel 7 `
  --max-player-speed 8 `
  --max-distance 150 `
  --workers 8
~~~

Distance preparation defaults to a 100,000-node refinement budget.

Existing TTA tables cannot be used for distance mode, and distance tables cannot
be used for TTA. Completed artifacts are immutable and checksummed; preparation
resumes interrupted work with the same settings. Source-table refinement is
available through `--source-table-id` with matching movement/time coverage.

Distance tables depend on acceleration, speed, precision and coverage. They do
not depend on the sigmoid power or allowance. Normalized data can be reused for
other movement settings when scaled spatial accuracy remains acceptable.

`--max-time` specifies physical time coverage. If omitted, preparation covers
`--max-distance` for `--min-ball-speed 3` with `--ball-dec drag`. Other supported
ball decelerations may be supplied to derive the horizon. These ball settings
only select coverage; they do not enter the movement dynamics. Generation with
different ball settings is allowed, with explicit reference fallback outside
validated coverage. No players are excluded and no spatial extrapolation is used.

The table approximates directional support of the exact-time reachable region,
indexed by initial speed/cap, normal angle relative to initial velocity, and
log1p(normalized time). It subtracts the stationary-player radius to make long
flight times inexpensive to interpolate. At a fixed initial speed and time, the
interpolated support is piecewise linear in angle. The runtime computes its
**global angular maximum analytically**, including interval interiors, to obtain
signed distance. It does not merely multiply a TTA difference by a speed.

## Generate predictions

~~~powershell
python scripts/generate_physical_xpass.py `
  --pc-xpass --pc-xpass-id distance_predictions `
  --margin distance `
  --accel 7 --accel-table-id distance_v1 `
  --max-player-speed 8 --max-player-speed-off 8 --max-player-speed-def 8 `
  --distance-power 3 --distance-allowance 1 `
  --reaction-time 0 --ball-dec drag --top-pass 10 --top-xt
~~~

Append the usual dataset and pass-grid selections. Acceleration defaults to zero
explicit reaction delay. `--margin tta` remains the overall default. Original
physical xPass and empirical `--margin reachability` do not use distance control.
Hawkeye location/freeze generation accepts the same distance settings.

Distance power must be positive and finite; allowance must be nonnegative and
finite, including zero. Explicit temporal sigmoid flags (`--lane-power`,
`--lane-inflection-point`, `--control-power`, `--control-inflection-point`) conflict
with distance mode. Explicit distance sigmoid flags conflict with other modes.
Nonzero/distance reaction delay and explicit `--dist-pass-*` conflict with
acceleration, as before. There is no blocking-radius option.

Prediction metadata and fingerprints record the margin definition, distance
parameters, and movement artifact checksum. Resumes inherit settings and reject
incompatible normal-cache overrides. Location caches retain their existing
cached-setting precedence. Cache-only inference/visualization consumes stored
predictions without loading movement artifacts or a numerical solver.

## Numerical interpretation

Movement remains dr/dt=v, dv/dt=u, norm(u)<=a, norm(v)<=V, using the full initial
velocity clipped to V. There is no final-velocity requirement, body orientation,
body/foot radius, obstacle or pitch-boundary constraint. Players may brake before
arrival, intercept while moving, or wait whenever those trajectories are feasible.
A location visited earlier can temporarily be unreachable at the ball time.

The reference support solver starts at 32 trajectory intervals and doubles as
needed. Feasible trajectories provide lower support bounds; continuous-time
costate bounds provide upper bounds. Speed, acceleration, and initial velocity
are checked. The signed-distance reference solver refines a feasible inner
polygon and directional outer bounds until the signed-distance bracket is at
most 5 mm. Exact analytic cases use no optimizer.

Table support interpolation is validated to 5 mm at preparation settings, using
reference support brackets targeted at 1 mm, deterministic face/interior probes,
and independent seeded holdouts. The reference uncertainty is included in the
probe acceptance calculation. The overall spatial accuracy target is 1 cm.
**Interpolation validation is sampled, not a proof for every interior query.**
A new table must also pass independent signed-distance checks on real states
before its measured accuracy is reported. No unresolved reference result becomes
trusted interpolation. Unvalidated cells and out-of-coverage queries use the
reference solver; unresolved fallback raises an error rather than guessing.

Runtime statistics retain `acceleration_queries`, `acceleration_fallbacks`, and
`acceleration_fallback_seconds`. Distance mode depends on ball flight time and
cannot share a single player-target arrival time across all launch speeds.
Benchmark numerical computation and fresh prediction-cache writes separately
from graph loading and one-time table preparation.


## Validation and local measurements

A prepared table named **distance_accel7_v8_v1** is available locally for a=7
m/s^2 and speed caps 8 m/s, covering 150 m separation and 167.227 s ball time.
It has 16,252 validated leaves, no fallback leaves, and passed 32,504 final
independent support holdouts. Numerical/reference failures encountered during
refinement are reported separately from unresolved final cells; the latter is
zero. Validation remains sampled.

The reproducible diagnostic is:

~~~powershell
python scripts/diagnostics/benchmark_pc_xpass_distance.py --table-id distance_accel7_v8_v1
~~~

It compares distance mode with acceleration-enabled TTA at the same a=7, V=8,
zero reaction delay, 3-degree/1 m pass grid, release speeds 3..25 m/s, and top-pass
10 with xT ranking. It includes independent signed-distance reference checks,
separate fresh-cache writes in temporary directories, and warm cache reuse.
Normal prediction caches and latest pointers are not changed by this diagnostic.
Its output directory defaults to out/pc_xpass_distance.

Distance queries depend on ball-arrival time and are materially more expensive
than TTA lookup. Use the recorded benchmark rather than assuming the earlier
constant-acceleration TTA overhead applies to this mode.


Local measurement on 26 October 2026 (one computation worker):

| Measurement | Acceleration TTA | Distance | Ratio |
|---|---:|---:|---:|
| Numerical computation, 12 benchmark states plus the two Hawkeye examples | 11.262 s | 319.883 s | 28.4x |
| Fresh prediction caches, two benchmark states | 1.789 s | 45.934 s | 25.7x |

The numerical run used one timed repetition after warmup. Cache numbers are
medians of three fresh writes, including Parquet/metadata writes and excluding
graph loading. Both modes use a=7, V=8 and zero reaction delay. Distance mode
performed 18,481,320 player/target/time queries with zero reference fallbacks in
the numerical run; TTA performed 1,540,110 speed-independent arrival queries.
Cached prediction reuse remained fast (approximately 0.3 s for the two states).
The process peak working set including loaded graph inputs was 875,606,016 bytes.
These are sample measurements, not a whole-season runtime guarantee.

The available table occupies 609,864 compressed bytes. Its cumulative preparation
wall time, including refinements and an interrupted attempt, was 948.8 s with
eight preparation workers. This is not a clean-build timing estimate.

Independent validation checked 96 signed distances against reference brackets.
The maximum error against either bracket endpoint was **5.65 mm**, within the
1 cm target. The maximum reference bracket width was 5.00 mm. This provides
sampled evidence, not an exhaustive interpolation-error guarantee.

For the two previously investigated top-pass-10 receiver scores:

| Scene / receiver | Acceleration TTA | Distance |
|---|---:|---:|
| 5b9966a1..., time_norm 0, away_250017233 | 0.902340 | 0.491838 |
| fe635459..., time_norm 1, home_1909098 | 0.886310 | 0.851899 |

Reports: out/pc_xpass_distance/benchmark.json, validation.json and predictions.csv.
The original prediction caches and their latest pointers were not modified.
