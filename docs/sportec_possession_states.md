# Sportec possession states and valuation

Sportec inference reconstructs spells from canonical actions and synchronized
control events. It preserves canonical pre/post observations independently of
training eligibility or complete spell reconstruction. Existing carry derivation,
one-second segments, eligibility, labels, and targets are unchanged.

A spell records acquisition and release/loss. Passing restarts are one-frame
spells. Unknown acquisitions or endings remain missing rather than becoming
zero-length carries. Supported loss endpoints use the same winning player as the
training post-carry convention. The spell owner remains the original player.
Typed passing restarts (`corner_short`, `corner_crossed`, `throw_in`, `goalkick`,
and free kicks) also supply their subsequent receipt. Duplicate restart
representations share one spell. Penalty/shooting-restart snapshots remain
available, without substituting a hypothetical pass for their shot-specific value.

For interception -> pass -> receipt -> pass -> corner -> receipt -> pass, the
spells are interception/pass, receipt/pass, corner/corner, and receipt/pass.
The corner start credits the preceding action; its own carry value is zero.

Ranking resolves values through the spell manifest:

- `game_state_value_start`: current spell's acquisition.
- `game_state_value_end`: current spell's endpoint.
- `game_state_value_next`: next genuine spell's start.
- `carry_epv = V(end) - V(start)`.
- `pass_epv = V(next) - V(end)`.
- `action_epv = V(next) - V(start)`.

Values use the original actor's team perspective. Opponent values change sign;
standard deviations do not. Missing successors stay missing, and goals/period
boundaries terminate links. Canonical post-state snapshots remain available even
when a genuine acquisition changes the successor used for valuation. Intermediate
frame predictions are never averaged into these values.

## Selection and fresh caches

`--scope actions` evaluates exact spell endpoints and preserved canonical
observations. `--scope frames --frames N` adds strided interior frames where control
is supported. Both paths share reconstruction and graph building. Invalid endpoints
are reported and stay missing; nearby valid frames do not replace them.
Corroborating control observations are retained in the manifest; they do not add
spell boundaries or bypass the selected scope/stride.

Sportec now requires `match_physical_states_v2` / `sportec_possession_states_v2`.
Physical identity includes match, period, frame, actual possessor and provenance;
semantic spells and event associations live in a separate manifest. Cache manifests
are published with match checkpoints; component runs export
`<match_id>/possession_manifest.json`. Training/evaluation readers explicitly map
canonical action indexes to physical rows. Older Sportec caches and component runs
must be regenerated. Existing feature runs remain usable with matching synchronized
source data. SkillCorner, Benchmark and Hawkeye retain their behavior.

The command below reproduces the numerical options recorded in
`pc_xpass_20260920T200417_760859_b7cb1e62_9a483db3`, with the new Sportec state
selection and an explicitly selected feature run. Choose an unused new cache ID.
Run the full rebuild after the separately planned batching work; it has not been
started as part of this implementation.

```powershell
python scripts/generate_physical_xpass.py `
  --pc-xpass --pc-xpass-id sportec_spells_v2 `
  --sportec-feature-run-id feature_20260910T110759_053479_8b508137 `
  --top-pass 5 10 --top-n-values 10 25 --top-xt `
  --reaction-time dist_pass --dist-pass-div 50 --dist-pass-min 0 --dist-pass-max 0.7 `
  --max-player-speed 5 --max-player-speed-off 5 --max-player-speed-def 5 `
  --lane-power 15 --lane-inflection-point 0.2 `
  --control-power 15 --control-inflection-point 0.2 `
  --endpoint-normalization share --boost-def-endpoint-control 1 `
  --use-position-discount false --ball-dec 0.45 `
  --max-speed 25 --min-speed 3 --speed-step 2 --angle-step 3 --radial-gridsize 1 `
  --physical-eps 0.0001 --num-workers 8 --physical-batch-size 16 --worker-thread-limit 1 `
  --scope actions `
  --pass-height-model-id pass_height/pass_height_20260703T120403_205360_c72e946d `
  --no-skillcorner --no-hawkeye --no-benchmark
```

Add your existing model/split options to these consumer examples:

```powershell
python scripts/run_relevant_models.py --bundle-id BUNDLE --feature-run-id feature_20260910T110759_053479_8b508137 --scope actions --use-physical-xpass --pc-xpass --pc-xpass-id sportec_spells_v2
python scripts/train_relevant_models.py --feature-run-id feature_20260910T110759_053479_8b508137 --only-pass-height --lane-survival top_pass_10 --pc-xpass-id sportec_spells_v2
```

## Six-match coverage audit

The audit compares physical graph hashes against the old cache, separately from
successor associations, and recomputes training carry segments against saved
sidecars. It reads source/cache data and writes only audit reports.

```powershell
python scripts/audit_sportec_possession_states.py --feature-run-id feature_20260910T110759_053479_8b508137 --old-pc-xpass-id pc_xpass_20260920T200417_760859_b7cb1e62_9a483db3 --output data/audits/sportec_possession_states_final_20261012.json
```

The JSON reports coverage, exclusions, changed successors, unchanged training
segments, and loss-endpoint comparisons per match. The adjacent CSV records
exclusions; `_successors.csv` lists changed associations with their old/new frames,
players, and reasons. Snapshot coverage and valuation changes are separate
acceptance criteria; matching raw row counts is not required.

## Validation record: 2026-10-12

The six-match audit restored every old cached graph state: all **10,169 rows**
(9,997 distinct graph hashes), with no exclusions. Every graph also matches its
explicit canonical pre/post association. Scope `actions` now evaluates
12,403 physical states across these matches.

| Match | Old rows represented | New physical states | Unchanged carry segments | Changed successors |
| --- | ---: | ---: | ---: | ---: |
| DFL-MAT-J03WEL | 1,976/1,976 | 2,294 | 929 | 66 |
| DFL-MAT-J03WEM | 1,656/1,656 | 2,064 | 892 | 75 |
| DFL-MAT-J03WEN | 1,405/1,405 | 1,845 | 587 | 89 |
| DFL-MAT-J03WEO | 1,574/1,574 | 1,942 | 770 | 74 |
| DFL-MAT-J03WEP | 1,923/1,923 | 2,215 | 957 | 66 |
| DFL-MAT-J03WEQ | 1,635/1,635 | 2,043 | 629 | 81 |

All **4,764 training carry segments** match their saved sidecars. Of **451** changed
successor associations, 447 point to a different genuine acquisition and four
remain missing at a chain boundary or uncertain continuation. The audit CSV gives
each affected action and its reason.

All **187 corresponding carry-loss endpoints** retain training's exact frame/player
convention. Two historical training intervals have no corresponding inference loss
spell because an opponent reception now establishes possession earlier:

| Match / training carry | Training loss target | Earlier opponent acquisition |
| --- | --- | --- |
| J03WEM / 367 | frame 65868, `away_30` | frame 65829, `away_30` |
| J03WEN / 507 | frame 105427, `away_16` | frame 105424, `away_16` |

Those training segments remain unchanged. Inference preserves the later control
observation in its manifest, while leaving the original owner's unestablished
release missing; it does not extend that spell through the opponent's acquisition.

Automated validation: **958 tests passed**, plus **41 subtests**, covering
reconstruction, cache/resume/height enrichment, ranking, unchanged carry training,
graph features, and the other datasets. Exact endpoint graph hashes also match
across `actions` and multiple `frames` strides.

A bounded smoke check used two real J03WEL states on a coarse numerical grid. It
checked pc-xPass calculation, lane-control round trips, canonical `top_pass_10`
training attachment, and an action-intent forward pass with a test model. Resume
computed zero additional states and left the match Parquet bytes unchanged.

Existing cache versions remain untouched. No full cache rebuild or batching
optimization was performed.
