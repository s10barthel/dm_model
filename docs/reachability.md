# Empirical reachability for pc-xPass

The default `--margin tta` preserves existing pc-xPass movement and cache behaviour.
`--margin reachability` uses fitted, speed-dependent shifted circles and a spatial
sigmoid. The offline fitting pipeline does not label interception attempts.

## Prepare and fit

Run from the repository root using its Python environment. Select the same outer
training split as the downstream experiment. `--train-count` is an alternative to
`--train-split`; exactly one is required. The percentage below is an example.

```powershell
.venv/Scripts/python.exe scripts/fit_reachability.py prepare --model-id bundesliga_reach_v1 --train-split 50
.venv/Scripts/python.exe scripts/fit_reachability.py fit --model-id bundesliga_reach_v1
.venv/Scripts/python.exe scripts/fit_reachability.py diagnose --model-id bundesliga_reach_v1
```

Preparation requires existing `data/tracking`, `data/tracking_processed`, and lineup
Parquet inputs for every selected training-pool match. It reserves 10% of that pool
(rounding up) for diagnostics, deterministically with seed 42. Outer-test matches
are excluded. Goalkeepers and unresolved roles do not contribute to fitting; the
shared ground-movement model applies to all players during inference.

The model directory is `data/pc_xpass/reachability/<model-id>/`:

- `preparation.json`: exact split, fitting/holdout match lists, preprocessing convention,
  source SHA-256 fingerprints, extraction settings.
- `shards/<match>.parquet` and `.json`: compressed aligned endpoints and checksummed
  per-match quality/exclusion reports. Preparation resumes verified shards.
- `circles.npz`, `metadata.json`: compact inference tables and artifact fingerprint.
- `fit_report.csv`, `diagnostics/`: support counts, pooling widths, directional fitting
  errors, holdout coverage/plots, extrapolation comparison and match-bootstrap results.

Preparation uses every start frame by default. `--frame-stride 25` makes a cheaper
pilot with one start per second; use a **different model ID** for the full fit.
All original source files are read-only. Hashing inputs itself can take time.
Shards can be large: the default expands every valid start into up to 50 horizons.
Preparation streams player/horizon blocks. Fitting and diagnosis stage one horizon
into temporary on-disk speed partitions, then load one pooled speed group at a time.
Allow disk space in the OS temporary directory as well as the model directory.

To change fitting settings while reusing prepared shards:

```powershell
.venv/Scripts/python.exe scripts/fit_reachability.py fit --model-id bundesliga_reach_v2 --source-model-id bundesliga_reach_v1 --min-samples 10000 --envelopes 0.999 max
```

Completed artifacts are immutable; use a new model ID for a changed fit. Interrupted
preparation can resume with identical settings and unchanged source fingerprints.
An interrupted fit that has already published `circles.npz` requires a new model ID,
reusing preparation through `--source-model-id`.

## Geometry and modelling conventions

Initial velocities come from the existing processed tracking. The recorded
convention is frame differences times 25 Hz, smoothed with the existing 15-sample,
order-2 Savitzky–Golay velocity filter. No velocities are recomputed. Trajectories
must be continuous, finite, in play, and within one period/episode. Windows crossing
invalid samples or discontinuities are excluded; starts also receive a +/-7-frame
guard for velocity smoothing.

Configurable preparation quality limits are `--max-speed 15` m/s,
`--max-acceleration 30` m/s² and `--jump-tolerance 0.5` m. A raw-frame jump is rejected
above `max_speed / 25 + jump_tolerance`. These conservative engineering thresholds
detect errors; they do not define the fitted capability limit. Quality report counts
are flagged frames by reason and can overlap.

Speed bins are 0.5 km/h; horizons are every 0.04 s through 1 s and every 0.2 s through
6 s, with zero centre/radius at time zero. Directions are aligned to initial velocity.
For each speed/horizon, project endpoints onto directions every 5 degrees. Fit
`h(theta) = c*cos(theta) + R` by equal-weight least squares with nonnegative radius.
The lowest bin uses `c=0`; its radius is the mean directional frontier. At inference,
the centre stays zero throughout that lowest bin. All other centre shifts rotate
with the player's initial velocity direction.

The four default envelopes are `0.99`, `0.995`, `0.999`, and `max`. Percentiles are
**directional projection quantiles**, not guaranteed two-dimensional coverage and
not control probabilities. `max` samples the cleaned convex hull's directional support;
the fitted circle need not enclose the entire hull. Actual holdout coverage is reported.
Groups below 10,000 samples pool symmetrically across adjacent speed bins, recording
the effective width and episode count. Overlapping starts are not independent samples.
Circle tables are linearly interpolated in speed and time; empirical circles are
not forced to nest. Speeds outside the fitted grid clamp to its endpoints.

After `T=5` s, the centre freezes and radius grows at `8` m/s:
`C(t)=C(T)`, `R(t)=R(T)+8*(t-T)`. The transition and expansion speed are configurable.
This is a capability approximation, not a claim about each player's maximum speed.

## Generate a new pc-xPass version

```powershell
.venv/Scripts/python.exe scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id reach_v1 --margin reachability --reachability-model-id bundesliga_reach_v1 --reachability-envelope 0.999 --reachability-lane-power 3 --reachability-lane-inflection-point 1 --reachability-control-power 3 --reachability-control-inflection-point 1 --reachability-extrapolation-start 5 --reachability-extrapolation-speed 8
```

Append your usual dataset selectors, pass-grid settings and worker limits. The shared
generator supports Sportec, SkillCorner, benchmark and Hawkeye inputs. Their existing
velocity inputs are used; the movement artifact records Bundesliga as the fitting source.
No full fit or full cache regeneration runs automatically.

The spatial margin is `g = radius - distance(target, circle_center)` in metres.
Lane and endpoint raw probabilities each use `sigmoid(power * (g + offset))`.
The defaults give `sigmoid(3*(g+1))`, corresponding to the old 15/0.2 sigmoid at a
5 m/s speed **when evaluated on equivalent geometry**. They are a starting calibration,
not probabilities learned from the hull. Reaction-time and constant player-speed
options are inactive in reachability mode. Ball deceleration and stopped-ball masks,
lane survival, endpoint normalisation and ranking retain their existing behaviour.

Artifact fingerprints and active settings travel through worker tasks, version metadata
and lane-survival fingerprints. Selecting an existing pc-xPass ID restores its settings.
Changing its reachability settings or artifact is rejected. Old caches without margin
metadata remain TTA, and TTA generation does not require a movement artifact.

## Compare before a full cache run

```powershell
.venv/Scripts/python.exe scripts/diagnostics/compare_reachability_xpass.py --model-id bundesliga_reach_v1 --feature-run-id YOUR_FEATURE_RUN --output out/reachability_comparison --count 100 --seed 42
```

Here `--feature-run-id` selects graph inputs for this diagnostic only; it does not
change the generator's legacy `--feature-run-id` behaviour. All movement-holdout
graph files must exist. The command saves the exact match/index sample and source
fingerprints for repeatable comparisons. It evaluates TTA, 99.9% reachability,
remaining fitted envelopes, and optional sweeps in that order.

`--grid-json` accepts shared computation settings. Defaults are ball speeds 5–25 m/s
in 2 m/s steps, 2.5-degree angles, 3 m radial steps, top10/top25, existing ball
deceleration, and no position discount. The TTA baseline uses dist_pass 50/0.2/0.7
and sigmoid 15/0.2. `--sweeps-json` accepts a JSON list such as:

```json
[
  {"lane_power": 4, "control_power": 4},
  {"extrapolation_start": 4, "extrapolation_speed": 8}
]
```

Reports include receiver xPass/ranks and components, raw player-level controls,
differences by speed/distance, component arrays and plots of the largest changes.
Timings exclude startup and include warmup before repeated measurement. Each variant
gets its own worker; RSS is sampled every 2 ms, with warmed baseline and incremental
peak reported. Samples may miss extremely short memory peaks.

`diagnose` reports actual coverage, directional errors and circle curves, and uses
20 match-bootstrap resamples with seed 42 on a fixed speed/horizon panel. These
diagnostics do not automatically choose a percentile or calibrate control probabilities.

## Verification benchmark

```powershell
.venv/Scripts/python.exe scripts/diagnostics/benchmark_reachability.py --repeats 3
.venv/Scripts/python.exe -m pytest tests/test_reachability.py tests/test_pc_xpass_versions.py tests/test_physical_xpass.py
```

On 12 existing benchmark states in this workspace, a three-repeat synthetic-circle
benchmark measured TTA at 3.20 s and reachability at 3.33 s (1.043x). Sampled peak
worker RSS was 730.3 MB versus 731.9 MB (1.002x); two independent workers returned
identical results. These figures measure the new inference machinery using synthetic
circles, **not a fitted Bundesliga model or the full cache pipeline**. Repeat the
holdout comparison with the fitted artifact before estimating a full-run duration.
