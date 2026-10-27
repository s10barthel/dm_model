# Implementation plan: optional aerial candidates for PC xPass

Status: all recommendations approved by the user and implemented on 27 October 2026. See [usage and validation](pc_xpass_high_pass.md). The following records the agreed implementation plan.

## Agreed behavior

Add aerial pass candidates to PC xPass when enabled. Use the existing horizontal control estimates and endpoint competition, with a common height eligibility cutoff for every player. No additional aerial reception penalty or pass-height probability mixture is part of this change.

The user has obtained the authors' permission to reuse the trajectory code from `andrewkang12345/monteCarloPassSearch`. Preserve attribution and pin the reviewed revision `cc6af7b00769fc780d890eca49f530c2dce97e4b`.

The requested cutoff is now **2 m**, superseding the earlier 1.5 m proposal. This is the maximum contact height along a simulated trajectory. It is independent of the learned pass-height classification threshold and of the maximum height an aerial trajectory may attain.

## CLI contract

| Option | Default | Meaning |
|---|---|---|
| `--high-pass {on,off}` | `off` | Enable aerial candidates alongside ground candidates. |
| `--high-pass-angle FLOAT [FLOAT ...]` | `15 20 25 30` | Elevation angles in degrees, measured upwards from the pitch plane. |
| `--high-pass-max-speed FLOAT` | `30` | Upper limit on total 3D aerial release speed, m/s. |
| `--high-pass-min-speed FLOAT` | `20` | Lower limit on total 3D aerial release speed, m/s. |
| `--high-pass-speed-step FLOAT` | Effective `--speed-step` | Aerial release-speed spacing, m/s. |
| `--high-pass-cutoff FLOAT` | `2` | Maximum ball height at which a player is eligible to intercept/control, m. |

Implement shared argument registration and normalization for `scripts/generate_physical_xpass.py` and `scripts/run_hawkeye_loc.py`. Thread the normalized configuration through dataset adapters, batch/worker calls and runtime cache generation. Consumers that only read prediction caches use the saved settings; they do not need duplicate generation flags.

Validation:

- These flags require PC xPass. Reject explicitly supplied aerial tuning flags with `--high-pass off`, rather than silently ignoring them. Parser defaults alone are not explicit tuning.
- Require a nonempty list of finite angles strictly between 0 and 90 degrees. Sort and deduplicate it for deterministic grids and fingerprints. Ground candidates already supply elevation zero.
- Speeds and steps must be finite and positive; minimum must not exceed maximum. Cutoff must be finite and positive.
- Resolve the inherited aerial speed step only after ordinary defaults and existing cache-resume precedence have been applied. Persist the effective value and grid for enabled runs.
- Follow the existing speed-grid convention: `min + k*step <= max`, with floating-point boundary tolerance. Include the maximum only if it lies on that grid. For 20..30 with step 3, use 20, 23, 26, 29; do not silently append a shorter last step.
- A fresh run with the feature off retains the existing computation and cache fingerprint. Missing aerial metadata in old caches means off.

Example planned invocation, with usual dataset/movement options added as required:

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_trial `
  --high-pass on --high-pass-angle 15 20 25 30 `
  --high-pass-min-speed 20 --high-pass-max-speed 30 `
  --speed-step 2 --high-pass-cutoff 2
```

This yields aerial speeds 20, 22, 24, 26, 28 and 30 m/s: 24 speed/elevation combinations for every horizontal angle.

## Approved decisions

| Decision | Options | Recommendation |
|---|---|---|
| Aerial physics | Preserve the authors' fixed damping; substitute quadratic drag. | Preserve the authors' fixed damping at 25 Hz for the initial implementation, including update ordering. Evaluate quadratic drag separately later. |
| Ground trajectory behavior when enabled | Keep the existing ground solver and `--ball-dec`; run ground passes through the imported simulator too. | Keep the existing ground solver. The feature adds aerial choices while retaining the current meaning of the ground configuration. `--ball-dec` applies to ground candidates only. |
| Aerial launch height | Start at `z0=0`; use current tracked ball height. | Start at `z0=0` for hypothetical footed passes, with XY origin at the current PC possessor position. This matches the existing hypothetical-pass setup and allows reusable flight tables. Tracked-height launches are more general but require a new height-sensitive input/cache contract and missing-height policy. |
| Aerial trajectory after landing | Continue the authors' bounce/rolling model; ground immediately and switch to our ground law; stop at first landing. | Continue the authors' bounce/rolling model through later receiving opportunities. This preserves the complete source trajectory. Ground candidates still use their separately selected ground law. |
| Ranked output | Rank all ground and aerial candidates together; export separate probabilities per family. | Rank together using the existing max/top-N/top-pass and xT rules. Add selected elevation and height/time diagnostics. Separate family scores can be an additional later feature. |

The user explicitly accepted all recommendations before implementation.

## Implementation sequence

### 1. Isolate and verify the flight model

Introduce `pc_xpass_high_pass.py` for settings, trajectory adaptation and flight-table preparation. Keep the upstream reference/helper and provenance in a small dedicated vendor directory or clearly attributed source file; importing the external search runner would unnecessarily import its learned-model stack.

Relevant source:

- [Simulation function](https://github.com/andrewkang12345/monteCarloPassSearch/blob/cc6af7b00769fc780d890eca49f530c2dce97e4b/monteCarloPassSearch/initParamVar/sim_core.py#L135-L170).
- [Constants](https://github.com/andrewkang12345/monteCarloPassSearch/blob/cc6af7b00769fc780d890eca49f530c2dce97e4b/monteCarloPassSearch/initParamVar/sim_viz_runtime.py#L42-L51).

Recommended baseline: spin zero; timestep 0.04 s; gravity 9.81 m/s^2; per-step air damping fraction 0.00648; restitution 0.109; horizontal ground-impact loss zero; rolling damping fraction 0.02731; rolling-height threshold 0.054 m. Preserve the source's vertical-speed rolling condition and numerical update order as well as its constants. Input data frequency does not change this internal integration step.

Construct release velocity as:

`v0 = speed * [cos(elevation)*cos(azimuth), cos(elevation)*sin(azimuth), sin(elevation)]`.

Expose internally position/time samples and the corresponding horizontal range. Zero spin makes the range/height/time profile independent of azimuth. Build each profile once per speed/elevation/launch-height/physics configuration and reuse it across states and azimuths where those inputs agree.

Continue through bouncing and rolling. Generate only to the needed pitch-diagonal range or until a conservative bound on all remaining horizontal travel is below a small documented tolerance (recommend 1 mm). Air damping alone supplies an upper bound even when rolling applies intermittently. Do not create arbitrary long-time arrivals beyond the computed path. An iteration safety limit must raise a diagnostic error rather than quietly mark truncated paths unreachable. Record the termination definition/tolerance in physics metadata.

### 2. Add aerial candidates to the existing scorer

Update `physical_pass_model.py`:

- Generalize the candidate dimension from launch speed to a deterministic list of `(family, speed, elevation)` entries, with the existing ground entries first. This handles different ground/aerial speed grids without padding invalid speed/elevation combinations.
- Retain the existing azimuth and horizontal radial grids. Interpolate arrival time and ball height from each flight profile at those radial distances. Use horizontal distance, not 3D arc length.
- Preserve on-pitch checks and mask physically unreached distances. A point above the contact cutoff remains a valid ball-path point; it simply has zero player contact control. Later descending points remain candidates.
- Pass the aerial arrival times into all supported movement modes: legacy TTA, acceleration TTA, exact-time distance and empirical reachability. Reuse acceleration TTA by XY target; exact-time distance/reachability require time-dependent queries. Preserve existing coverage checks and solver fallbacks.
- Apply `height <= high_pass_cutoff` to both raw lane and raw endpoint controls, for every player. Equality is eligible. Apply the mask before each opponent's prefix maximum and before endpoint normalization. Ground samples have height zero.
- Keep opponent-only lane aggregation, endpoint competition, optional position discount and xT ranking definitions.

The cutoff changes contact eligibility, not the horizontal control mapping. Passing over an opponent can improve lane survival, while additional flight time can improve opponents' endpoint reachability.

### 3. Preserve global ranking and enrich outputs

- Max and ordinary top-N operate over the union of all valid candidate cells. Do not average separate ground/aerial means.
- Top-pass retains the best endpoint for each `(speed, azimuth, elevation)` launch, then ranks those distinct launches globally. Ground has elevation zero. Introduce an enabled-mode definition/version; continue recognizing the legacy speed/azimuth definition for old and disabled caches.
- Keep deterministic ties: ground before aerial, followed by canonical grid order. Verify both direct probability ranking and xT-weighted ranking. With xT ranking, the selected raw probability is not guaranteed to increase when candidates are added.
- Preserve `__angle` as azimuth and `__speed` as total release speed. For enabled caches add receiver detail columns `__elevation`, `__target_z` and `__flight_time`; ground selections report elevation/height zero.
- Add the same selected trajectory information to opponent lane-control exports and their schema/version. Their product reconstruction must still equal the selected lane survival.
- Update visualizers to read and label the selected elevation; legacy caches display ground elevation zero. A 3D visualization is not required for this change.

### 4. Extend metadata, versioning and resume contracts

Primary files: `pc_xpass_versions.py`, `pc_xpass_resume.py`, `pc_xpass_match_cache.py`, `pc_xpass_contract.py`, and metadata/validation/runtime helpers in `physical_pass_model.py`.

- Register all six options in generation settings, including explicit-argument detection.
- For enabled runs fingerprint effective aerial angles/speeds/cutoff, simulator revision and coefficients, integration timestep, launch policy, interpolation and termination definitions, and the changed top-pass/output contracts.
- Keep the ground motion metadata and add a separately versioned aerial-physics block. Ground `--ball-dec` and aerial damping must not be conflated.
- Preserve existing normal-cache resume rejection of incompatible overrides and HawkEye-location cached-setting precedence. Apply that precedence before resolving inherited defaults. Do not regenerate scores silently during cache-only inference.
- Audit exact comparisons against `PC_XPASS_TOP_PASS_DEFINITION`; update readers as well as writers to accept the correct definition for the saved mode.
- Legacy/off runs retain old fingerprints and output compatibility. Enabled and disabled computations cannot share prediction-cache identities.
- Current `physical_state_hash` excludes ball height. If tracked launch height is selected, introduce a mode-sensitive physical input contract/hash covering normalized launch height throughout source creation, cache lookup and graph validation. Missing/nonfinite height must have an explicit recorded policy. Do not silently feed tracked height into the current hash.
- Cached inference should continue to consume stored values without loading the flight simulator or movement solvers.

### 5. Bound runtime and memory

Batch aerial candidate profiles if necessary and merge selections globally with exact ranking/tie equivalence. Reuse immutable flight tables within persistent workers. Avoid retaining every player's complete aerial control tensor across all chunks.

At step 2, the requested aerial grid adds 24 speed/elevation profiles to the common 12-profile ground grid. This triples the trajectory count, but does not imply an exact threefold end-to-end runtime increase: TTA reuse, exact-time queries, batching and IO differ. Measure warm computation, first-worker initialization, fresh-cache writes, peak memory and cache reuse separately. Include Windows spawned-worker coverage.

### 6. Validate behavior and document usage

Meaningful checks:

1. Disabled-mode regressions: predictions, candidate ordering, fingerprints, metadata compatibility, ordinary/location resumes and old-cache reads agree with the existing implementation.
2. CLI: list normalization, speed-step inheritance after cache resolution, finite/range validation, independent aerial min/max, grid endpoints and explicit incompatible settings.
3. Physics: compare adapted trajectories with the pinned upstream implementation for the full default grid through flight, impact and rolling. Check rotation invariance, launch-speed norm, nonnegative heights, correct range/time interpolation and bounded stopping error.
4. Contact eligibility: below/at/above 2 m; an opponent under an elevated segment cannot intercept there; the same opponent can intercept on descent; receivers above the cutoff have zero endpoint raw control. Check the cutoff is not confused with pass-height labels or blend thresholds.
5. Aggregation: compare a small exhaustive ground-plus-aerial reference with production batching for all endpoint normalization modes, global top-N, distinct top-pass, xT selection, ties and per-opponent lane reconstruction.
6. Movement compatibility: fixed-reaction TTA, accelerated TTA with blocking radius, exact-time distance, empirical reachability, and existing out-of-coverage behavior.
7. Cache integration: fresh generation, interrupted resume, mismatched aerial settings/source revision, matching reuse, cache-only visualization and lane exports for ground and aerial selections. Cover Sportec and the supported external-data paths with small fixtures.
8. Representative-state diagnostics: compare off/on candidate counts, selected elevation, lane survival, endpoint control, final ranking and runtime. Use saved states with obstructed lanes and long-pass options. Numerical correctness and tactical plausibility do not establish predictive accuracy.

Update README generation examples and `docs/pc_xpass_runtime.md`, `docs/pc_xpass_caches.md`, `docs/pc_xpass_output.md`, and `docs/lane_control_export.md`. Explain total 3D speed, the shared 2 m contact cutoff, ground/aerial motion laws, inherited step resolution and fresh-version requirements.

## Deliverables and acceptance

Deliver the attributed flight module, integrated generation/scoring/cache/export support, focused regression tests, documentation, and a small reproducible off/on benchmark report. Acceptance requires exact legacy behavior when off, source-equivalent flight updates when on, consistent global ranking and exports, cache isolation, and reported measured overhead. Full-season cache generation or recalibration is a separate execution step after implementation validation.
