# Optional component inference

The component runners for Hawkeye, Sportec, SkillCorner, Benchmark, and Hawkeye
location/freeze accept any subset of the six core model IDs, including none.
Explicit IDs override bundle IDs. A supplied invalid checkpoint remains an error.
Sportec requires `--feature-run-id` when models, a bundle, or a selected physical
cache cannot identify the feature context.

Before processing, the runner lists omitted models and derived metrics that
cannot be calculated, with the missing dependencies. Answer `yes` or `no` to
proceed. Blank input repeats the question; EOF, interruption, and `no` cancel.
There is no default answer or timeout. `--accept-comparison-warnings` does not
accept model omissions. Decisions and availability are recorded in
`metadata.json` under `model_selection`.

Exports retain their existing columns and frame/player identities. Unselected
predictions are NaN in Parquet and blank in CSV. These rules do not add derived
metric columns to raw-component export formats.

## PC xPass v0

`--use-physical-xpass --pc-xpass --xpass-weight v0` copies the selected cached
PC xPass metric without requiring a pass-success checkpoint. Cache validation,
state provenance, offside handling, and coverage checks still apply.

| Metric | Required sources |
| --- | --- |
| Raw scoring outcomes | Scoring checkpoint |
| Raw conceding outcomes | Conceding checkpoint |
| Net success/failure outcome | Both outcome checkpoints |
| Pass score | Pass-success probability and both outcome checkpoints |
| EPV / intent-weighted pass score | Pass score inputs and pass-intent checkpoint |
| Model pass success / carry success / carry score | Learned pass-success checkpoint; carry score also needs both outcome checkpoints |

PC xPass v0 supplies the pass-success probability for pass score and EPV.
Without a learned pass-success checkpoint, carry success and carry score remain
NaN. If the checkpoint is supplied, its separate predictions and supported carry
predictions remain available. v1–v5 blending requires a learned pass-success
checkpoint; PC xPass v4/v5 also require their explicit auxiliary checkpoints.

Dedicated `generate_epv.py` accepts v0 without a pass-success checkpoint but
still requires pass intent and both outcome checkpoints. A warning acceptance
cannot supply these required EPV inputs. Positive-intent candidates with missing
score inputs keep the aggregated value unavailable rather than producing a
partial sum.

## Examples

Raw Sportec outcomes and PC xPass, with the feature run supplied explicitly:

```powershell
python scripts/run_relevant_models.py --feature-run-id <features> --outcome-scoring-model-id outcome_scoring/<id> --outcome-conceding-model-id outcome_conceding/<id> --use-physical-xpass --pc-xpass --xpass-weight v0 --pc-xpass-id <cache>
```

EPV from PC xPass v0, pass intent, and the two outcome checkpoints:

```powershell
python scripts/generate_epv.py --feature-run-id <features> --pass-intent-model-id pass_intent/<id> --outcome-scoring-model-id outcome_scoring/<id> --outcome-conceding-model-id outcome_conceding/<id> --use-physical-xpass --pc-xpass --xpass-weight v0 --pc-xpass-id <cache>
```

Direct-inference visualizers (`run_and_visualize_hawkeye.py` and
`visualize_action_components.py`, including comparison modes) use the same
source-availability rules. Missing dependencies for requested displays are
included in the confirmation warning. Unavailable displays are skipped; an
invocation with no available displays fails clearly. Saved-run visualizers use
the availability metadata so NaN placeholder tables are not mistaken for model
predictions.

Benchmark retains intent-weighted state values as unavailable without pass
intent, while its unweighted pass-score summary can use v0 and both outcome
models. Existing complete-model postprocessing keeps its candidate selection.
