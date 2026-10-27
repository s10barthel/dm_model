# Existing-grid uncertainty for PC xPass

Enable uncertainty during generation with `--uncertainty true`. The default is
`false`. The same flags are accepted by `scripts/run_hawkeye_loc.py`; location
runs must select `--xpass-version max` or `--top-pass N`.

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_uncertainty `
  --high-pass on --uncertainty true --speed-uncertainty 2 `
  --angle-uncertainty 3 --elevation-uncertainty 6 --top-pass 5 10
```

Add the usual dataset/frame selection. Uncertainty also works with ground-only
candidates. It supports `max` and `top-pass<N>`; ordinary endpoint `top<N>` and the
original physical-xPass `noise-kernel` are unavailable in this mode. Implicit
endpoint metrics are omitted, and explicit `--top-n`/`--top-n-values` requests
are rejected. Saved-cache readers select the appropriate existing metric and
inherit its uncertainty configuration; they perform no new simulation.

| Flag | Default | Definition |
|---|---:|---|
| `--uncertainty true/false` | `false` | Average execution outcomes before launch selection. |
| `--speed-uncertainty FLOAT` | `2` | Gaussian scale and inclusive radius in total initial speed, m/s. |
| `--angle-uncertainty FLOAT` | `3` | Gaussian scale and inclusive radius in horizontal direction, degrees. |
| `--elevation-uncertainty FLOAT` | `6` | Gaussian scale and inclusive radius in elevation, degrees. |

Scales are fixed across intended launches, finite and nonnegative. Zero freezes
that dimension. Explicit scale flags require `--uncertainty true`; all four
flags are PC-xPass only.

## Averaging and selection

For each receiver:

1. Evaluate the existing trajectories and reception locations with the existing
   control, lane survival, height cutoffs and position discount.
2. Select the best reception location along each executed trajectory. Its score
   is the product of lane survival, endpoint control and position discount.
   With `--top-xt`, select that location by score times endpoint xT.
3. For each intended launch, include only computed launch vectors whose speed,
   circular horizontal-angle and elevation differences are each within one
   corresponding uncertainty scale, inclusive. Include the intention itself.
4. Weight each included vector by `exp(-0.5 * sum((difference/scale)**2))` and
   normalize over the included vectors. A frozen dimension contributes zero to
   the exponent and allows only matching values. Computed unsuccessful outcomes
   contribute zero while retaining their weight; absent profiles are excluded.
5. Average complete outcome scores, then select the best intended launch.
   For `top-pass<N>`, retain the best intended elevation for each speed and
   horizontal angle, rank those representatives and average the best N.

With xT ranking, average **score times each outcome's endpoint xT**, while
exporting the corresponding expected xPass probability. Reception distance
varies with the executed trajectory and has no independent uncertainty.

At the default elevations `0, 20, 25, 30`, eligible elevation neighbours are
`0 -> {0}`, `20 -> {20,25}`, `25 -> {20,25,30}`, `30 -> {25,30}`. For elevation
alone, the 30-degree intention assigns about 58.6% to 30 and 41.4% to 25.

The Gaussian assumption and default scales are adapted from
[Dick, Link and Brefeld (2022), sections 3.4 and 5](https://link.springer.com/article/10.1007/s10618-022-00827-2).
The hard cutoff and renormalization over existing grid points are a deliberate
local robustness approximation, rather than integration over the full Gaussian.
There is no grid expansion or additional trajectory/reachability evaluation.

## Outputs and cache compatibility

Existing xPass metric columns contain expected probabilities when enabled.
The selected `__speed`, `__angle`, `__elevation` describe the **intended launch**.
`__distance`, `__target_x`, `__target_y`, aerial `__target_z`/`__flight_time`,
and `__lane_survival`/`__control_prob` describe that intention's **nominal**
reception. They do not describe a single averaged physical endpoint. If the
selected intention has no nominal reception location, its location/components
are NaN, although neighbouring successful executions can yield a positive mean.

New receiver columns describe the highest-ranked intention, even when the
unsuffixed receiver estimate selects a top-pass mean. Top-pass metrics have
their own component means.

| Suffix | Meaning for the selected intended launch |
|---|---|
| `__nominal_xpass` | Its unperturbed reception score, before numerical clipping. |
| `__expected_xpass` | Weighted outcome xPass before numerical clipping. |
| `__expected_lane_survival` | Weighted lane-survival marginal. |
| `__expected_control_prob` | Weighted control marginal. |
| `__expected_position_discount` | Weighted position-discount marginal. |
| `__expected_ranking` | Expected ranking score; includes outcome xT when enabled. |

Top-pass `*_lane_survival` and `*_control_prob` are means of expected marginals
for the selected representatives. Additional `*_nominal_xpass`,
`*_nominal_lane_survival`, `*_nominal_control_prob` record their nominal means.
**Products of expected marginals do not generally equal expected xPass.**

Opponent lane sidecars describe the nominal trajectory of the selected
intention, with schema 3 and definition
`per_opponent_nominal_intended_launch_before_endpoint_v1`. Their opponent
product reconstructs nominal lane survival, rather than its expected marginal.

Enabled caches record the full algorithm/scales in `uncertainty_config`, the
settings, computation dependencies and lane fingerprints. Changes require a new
cache ID. Historical caches default to uncertainty off. Disabled mode preserves
existing numerical results, column schemas and compatibility fingerprints.

## Validation

`tests/test_pc_xpass_uncertainty.py` covers Gaussian weights, support boundaries,
wraparound, zero scales, independent reference expectations, xT and top-pass
selection, movement modes, version inheritance, cache repair, transactional
match storage, the production CLI and Windows worker equality.

Run a timing comparison on saved states without writing prediction caches:

```powershell
python scripts/diagnostics/benchmark_pc_xpass_uncertainty.py --states 2 --repeats 3
```

Use `--margin distance --accel-table-id distance_accel7_v8_v1` for distance
control, or `--high-pass off` for ground-only scoring. The diagnostic alternates
off/on timing order, separates initialization, and checks that averaging adds
no flight profiles. Results are small-sample measurements, not whole-season
throughput estimates.

Validation results and measured timings are recorded in
[the validation report](pc_xpass_uncertainty_validation.md).
