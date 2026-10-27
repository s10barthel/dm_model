# Separate uncertainty export for PC xPass

Enable an additional uncertainty export during generation with `--uncertainty true`
(default `false`). It leaves raw max, top-N, top-pass and their geometry unchanged.
The same flags are accepted by `scripts/run_hawkeye_loc.py`. Select the separate
estimate with `--xpass-version uncertainty` during inference or visualization;
location generation additionally requires `--uncertainty true` for that selector.

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id aerial_uncertainty `
  --high-pass on --uncertainty true --speed-uncertainty 2 `
  --angle-uncertainty 3 --elevation-uncertainty 6 --top-pass 5 10
```

Add the usual dataset/frame selection. Ground-only and combined ground/aerial
candidates are supported. One run can export raw max, raw endpoint `top<N>`, raw
`top-pass<N>` and `uncertainty` independently. `--no-max` controls only raw max;
`--uncertainty true --no-max --no-topmean` permits uncertainty-only generation.
The original physical-xPass `noise-kernel` remains a separate non-PC model.
Cache-only readers require the requested exported metric and never simulate
missing outcomes or fall back from uncertainty to raw max.

| Flag | Default | Definition |
|---|---:|---|
| `--uncertainty true/false` | `false` | Add an independent expected-success maximum. |
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
5. Average complete outcome scores, then select the best intended launch for
   `uncertainty_xpass`. Raw max, endpoint top-N and top-pass selection continue
   to use the original unsmoothed scores. There is no uncertainty-adjusted
   top-pass mode.

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

`<player>__uncertainty_xpass` contains the selected uncertainty winner's expected
probability with the usual numerical clipping. Existing max and top metrics,
their default receiver-column alias and ordinary geometry remain identical
when the flag is enabled. If uncertainty is the only export, its score becomes
the default receiver-column alias.

All uncertainty diagnostics use `<player>__uncertainty_<detail>`:

| Detail | Meaning for the selected uncertainty intention |
|---|---|
| `speed`, `angle`, `elevation` | Intended total launch speed (m/s) and angles (degrees). |
| `distance`, `target_x`, `target_y`, `target_z`, `flight_time` | Its nominal reception geometry and time, rather than an averaged physical endpoint. |
| `nominal_xpass`, `nominal_lane_survival`, `nominal_control_prob`, `nominal_position_discount`, `nominal_ranking` | Unperturbed values for that intention's best endpoint. |
| `expected_xpass` | Unclipped expected probability; the selectable metric is `uncertainty_xpass`. |
| `expected_lane_survival`, `expected_control_prob`, `expected_position_discount`, `expected_ranking` | Averaged marginals and ranking, including outcome xT when enabled. |

Elevation, height and flight time are also present for ground-only uncertainty;
elevation and height are zero. If the winning intention has no nominal
reception, nominal reception geometry is NaN. Products of expected component
marginals do not generally equal expected xPass. Previous unprefixed uncertainty
and uncertainty-specific top-pass diagnostics are not exported.

Readers and visualization use the uncertainty winner's prefixed elevation and
geometry when that metric is selected. Raw max and lane-control sidecars continue
to describe the raw maximum, using the usual ground/aerial sidecar schemas.

New enabled caches record uncertainty configuration schema 2, the independent
metric definition, scales, available metric and output semantics. Previous
schema-1 uncertainty-enabled caches stored adjusted values in `max_xpass` and
are rejected for both generation and consumption: regenerate under a new cache
ID. Caches without uncertainty remain compatible. Changes to scales/settings
require a new cache ID; missing metric or diagnostic columns are repaired during
generation and rejected during cache-only consumption as appropriate.

## Validation

`tests/test_pc_xpass_uncertainty.py` covers Gaussian weights, support boundaries,
wraparound, zero scales, independent reference expectations, xT and independent
raw top-pass selection, raw-export equivalence, legacy rejection, movement modes,
version inheritance, cache repair, transactional match storage, the production CLI
and Windows worker equality.

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
