# Configurable validation and final refitting

`scripts/train_relevant_models.py` defaults to `--validation-mode holdout` with
20% validation. `holdout_80_20` remains an alias requiring that same split.
`expanding` runs fresh models at one or more explicitly requested training sizes,
using the same latest validation cohort for every stage.

## Split options

| Option | Meaning |
| --- | --- |
| `--validation-count N` | Reserve the last N development games for validation |
| `--validation-fraction F` | Reserve the latest fraction of development games; default 0.2 |
| `--training-counts N ...` | Actual training-game counts, excluding validation |
| `--training-fractions F ...` | Fractions of the remaining training pool |
| `--final-refit` | Refit using the largest validation stage's recipe |

Count and fraction options are mutually exclusive within each pair. Expanding
requires a nonempty training-size list; holdout permits one size and otherwise
uses all non-validation development games. A fraction of 1.0 uses the entire
remaining training pool. Resolved sizes must be positive, strictly increasing,
and no larger than that pool. There is no fixed number of expanding stages.

The outer development/test split comes from the feature run's immutable saved
manifest. `--train-count` and `--train-split` only assert that recorded selector;
they cannot redefine an existing feature run's split. Omitting them infers it.
Games follow manifest order, currently MatchId order. Validation is reserved
first, and each training subset is a prefix before the validation boundary.
Missing files and action filtering never move those boundaries or replace games.
Requested, loaded, and contributing game counts are reported separately.

For fractional validation, training capacity is
`floor(development_count * (1 - validation_fraction))`; validation receives the
remainder. Training fractions are floored against this capacity. Rounding that
produces an empty split or duplicate training sizes is rejected.

Example, using a feature run recorded with 765 development games:

```powershell
python scripts/train_relevant_models.py `
  --feature-run-id FEATURE_RUN_ID `
  --validation-mode expanding `
  --train-count 765 `
  --validation-count 153 `
  --training-counts 306 459 612 `
  --final-refit
```

Add the usual model/target options for your experiment. The three stages train on
the first 306, 459, and 612 development games and validate on games 613-765. The
optional refit trains on all 765. Recorded test games remain excluded throughout.
For an ordinary holdout, omit the training list and use `--validation-mode holdout`.
Refitting is disabled unless explicitly requested.

## Validation-derived refit recipe

Each new validation run records a versioned `learning_rate_history`, with the
actual learning rate used for each completed epoch. Resume preserves that history;
rollback events do not erase executed epochs.

A final refit copies the largest stage's best validation epoch and its rate
history through that epoch. For holdout, the source is its single validation run.
It reconstructs the source model/training configuration and starts fresh with the
same seed and a new Adam optimizer. No source weights or optimizer state are
loaded. The recorded rate is applied before each corresponding epoch.

Refits have no validation, adaptive rate decisions, rollback, or early stopping.
The selected best epoch is their duration; the validation run's 30-epoch minimum
does not extend it. Final-epoch weights are delivered. Reductions after the best
epoch are excluded. This is an approximation: more data gives more optimizer
updates per epoch, and updates discarded by source rollbacks remain in the refit.

Validation artifacts remain under `holdout` or `fold_1` through `fold_N`; refits
remain under `final_refit`. Published model-root inference artifacts represent
the largest validation stage without refit, or the completed refit with it.
Recorded propensity dependencies use immutable stage IDs when trained together;
their weight fingerprints are checked before refitting.

## Refitting saved sources without validation

One source is sufficient to run the command with no further options:

```powershell
python scripts/train_relevant_models.py `
  --refit-source-model-id outcome_conceding/RUN_ID/holdout
```

Repeat the flag to refit independent sources, including multiple runs of one task:

```powershell
python scripts/train_relevant_models.py `
  --refit-source-model-id outcome_conceding/RUN_A/fold_3 `
  --refit-source-model-id outcome_conceding/RUN_B/holdout `
  --refit-source-model-id outcome_scoring/RUN_C/fold_4
```

Sources imply refitting and skip all validation. They determine the jobs instead
of default task selection. Different feature runs and configurations are allowed.
Identical repeated IDs are rejected. Sources must be completed validation models
with supported structured rate history, a best epoch, and available artifacts
and dependencies. Old runs without that history and existing refits are rejected;
logs are not parsed. A root ID is accepted when it represents a validation model.

Configuration comes from each source. Explicit model/task, feature, split,
training-size, learning-rate, seed, or batch-size overrides are rejected.
Operational options include device, monitoring, pin-memory, crash capture, and
dataset cache settings. A loading-mode override must preserve the resolved
memory/disk mode; a disk buffer-size change is rejected because it changes sample
order. Dependencies are reused, not retrained or replaced by other batch jobs.

All sources are checked before training starts. Jobs run sequentially, each with
a unique new model ID. A batch manifest at
`saved/refit_batches/<batch_id>/metadata.json` lists source/output IDs, recipes,
statuses, and failures. This is not an ordinary model bundle. A failure stops the
batch, preserves completed outputs, and leaves later jobs unstarted. Evaluate
individual outputs using existing explicit model-ID options.

An interrupted model can be resumed with `--resume-id TASK/RUN/final_refit`.
Its checkpoint contains its recipe, so resume does not need the original source
metadata or batch configuration file. Existing data/code identity checks apply.
Resuming a model does not automatically continue the rest of its batch.

## Learning curves and historical artifacts

New bundle metadata lists all stage IDs explicitly. Learning-curve evaluation
uses that list, while individual IDs and old bundles retain artifact discovery.
Series must have distinct nested training sizes and compatible configurations,
seeds, test membership, and aligned test examples. New shared-validation stages
also check requested/usable validation membership and example identities.

Validation summaries retain separate stage metrics. They do not average shared
validation results into a cross-validation estimate. Refits have no point in
validation-score plots; they can be included in the common-test learning curve.

Historical successive-block expanding splits remain interpretable for recorded
artifacts. New expanding commands must provide training sizes and use the shared
validation protocol. Historical checkpoints are not rewritten or migrated.
