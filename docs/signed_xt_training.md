# Training on unclipped xT gains

Outcome scoring and conceding GNNs support feature runs with
`xt_targets.representation = "gain"` and `clip_negative_gains = false`.
Training selects signed learning automatically from that saved metadata. There
is no additional training flag, and the selection does not depend on whether a
particular training sample contains a negative value.

Generate or refresh the feature run using `--xt-target-representation gain`
and `--no-xt-clip-negative-gains`. Wait for the feature run to complete, then use
its ID with the usual training command and `--target-family xt` in
`scripts/train_relevant_models.py`, or `--use_xt` in `train.py`.

## Learning behavior

| Configuration | Output | Training loss | Validation selection |
| --- | --- | --- | --- |
| Signed scalar (default) | Raw predicted gain | Sample-weighted MSE | `mse_loss` |
| Signed `--learning-mode distributional` | Expected gain over categorical atoms | Sample-weighted categorical cross-entropy | `mse_loss` of expected gain |
| Existing absolute/clipped scalar | Sigmoid | Existing BCE | Existing selection |
| Existing absolute/clipped distributional | Expected value on `[0, 1]` | Existing categorical cross-entropy | Existing mean soft BCE |

The signed scalar model retains the existing architecture and two execution
branches. Outputs are not clipped, including values outside `[-1, 1]`; evaluation
reports their frequency. Both signed losses retain the existing sample-weight
reduction (`mean(weight * loss)`). `--weight_bce` is rejected for signed gains.

Signed distributions use definition version 2. Atoms include `-1`, `0`, and `1`;
remaining atoms are training-interior quantiles. `--distribution-atoms` must be
at least 3. The endpoints cover differences of bounded surface values for the
supported next, next-plus, time, possession, in-state, and discounted returns.
Surface-specific unavailable returns remain unavailable. Nonfinite labels and
labels outside the declared domain are rejected, never clipped.

Support is fitted only on retained training labels, then frozen for evaluation,
resume, and source refit. Version-1 probability distributions remain unchanged.

## Evaluation and inference

Signed predictions retain original gain units in component tables, distribution
sidecars, pass-score calculations, and comparison exports. Evaluation reports
MAE, RMSE, correlations, mean bias, and the out-of-bounds prediction fraction.
Distributional evaluation also reports categorical loss and CRPS.

Binary goal diagnostics report ranking metrics (ROC AUC and PR AUC). BCE, Brier
score, probability calibration, and probability thresholds are not applied to
signed values. Plots include negative values. Comparison rows carry
`outcome_domain = "signed_gain"`; paired comparisons with probability models
use only metrics defined for both models.

The versioned `outcome_learning_contract` is saved in arguments, metadata,
checkpoints, model provenance, and bundles. Resume/refit cannot change the target
interpretation or loss. Old checkpoints retain their saved probability behavior;
signed checkpoints require the explicit contract. Scoring/conceding bundle
members must agree on the xT target definition and learning contract.

Inference commands can combine checkpoints with different xT surfaces, target
definitions, or learning contracts after explicit confirmation. They print the
differences and ask `Proceed despite these warnings? [yes/no]`. Answer `yes` to
continue or `no` to cancel. Empty answers do not accept the warning. For scripted
execution, the existing `--accept-comparison-warnings` flag explicitly accepts
these limitations; noninteractive runs without that flag cancel.

For example, to inspect signed scoring gain together with absolute conceding:

```powershell
.venv/Scripts/python.exe scripts/run_and_visualize_hawkeye.py `
  --outcome-scoring-model-id outcome_scoring/outcome_scoring_20261111T100349_602072_96195465 `
  --outcome-conceding-model-id outcome_conceding/outcome_conceding_20261110T080157_625632_b30a6a77 `
  --situation-id 41558c01-d87a-4efd-b0e9-ddfcf8470a77 `
  --only-outcome-success --time-norm 1 --device cpu
```

The confirmation applies to ordinary inference, live visualization, comparisons,
and EPV generation. Training/bundle creation and library validation remain strict
unless their caller explicitly supplies a warning collector. Each checkpoint
keeps its own decoder and original units; acceptance does not convert gain to
absolute value or calibrate their combination.

Ordinary inference metadata records definitions and acceptance under
`outcome_compatibility`. Comparisons use their existing `warnings` and
`warning_acceptances` fields with per-variant source context. Sportec runtime
feature selection records which checkpoints' target definitions the runtime
labels match. A new runtime definition requires an additional warning; labels
that differ from a model's target are not its native evaluation reference.
Graph/schema, required-artifact, stop-condition, and physical-cache checks keep
their existing errors.

## Scope

Supported: scalar and distributional `outcome_scoring`/`outcome_conceding` GNNs,
their evaluation, inference, and component exports. Other target consumers and
tree models reject unclipped gains. The legacy `DEFCON` valuation engine also
rejects signed models because its shot, penalty, and terminal-state overrides
assume absolute goal probabilities. Component inference and pass-score exports
support signed models without those overrides.

Feature generation and existing non-signed training settings are unchanged.
