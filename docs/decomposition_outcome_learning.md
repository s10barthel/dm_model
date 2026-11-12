# Occurrence/severity outcome learning

Outcome scoring and outcome conceding support an explicit decomposition alternative. Add these options to an existing training command:

```powershell
--learning-mode decomposition `
--severity-mode scalar `
```

Scalar severity is the default when --severity-mode is omitted. To learn a conditional severity distribution instead:

```powershell
--learning-mode decomposition `
--severity-mode distributional `
--distribution-atoms 65 `
```

Both train.py and scripts/train_relevant_models.py accept the options. The wrapper routes them to selected outcome tasks. Direct training uses --use_xt; the wrapper uses --target-family xt. Choose a feature run whose saved xT representation is absolute. Training rejects both clipped and signed gain, other target families, non-GNN estimators, include_out and outcome-dependent class balancing. Setting --severity-mode in another learning mode is an error; --distribution-atoms in scalar decomposition is also an error. The default overall learning mode remains scalar.

## Predictions and objective

One graph encoder serves failure and success execution branches. For each candidate and branch the model predicts:

- occurrence_probability q: probability that the original absolute target is strictly positive;
- conditional_severity m: expected target given that it is positive;
- ordinary component prediction q*m, in the original xT units.

This preserves outcome_scoring_success, outcome_scoring_failure, outcome_conceding_success and outcome_conceding_failure, as well as the existing pass-score formula. Occurrence follows the existing absolute-target zero/positive semantics; it is an eligible-action-within-window event, not necessarily the instant possession changes. Excluded/invalid windows retain their existing exclusion policy. No epsilon threshold is used.

The occurrence loss uses every observed example. The severity loss uses positives only. Only the observed candidate and execution branch receive a direct label loss:

`loss = mean(sample_weight * [occurrence_BCE + is_positive * severity_loss])`.

Scalar severity uses bounded soft-target BCE. Distributional severity uses mean-preserving categorical CE. Positive loss contributions are averaged over the full batch, rather than independently normalized and reweighted. No additional loss multiplier or post-hoc scoring/conceding weight is introduced. Soft-target BCE estimates a conditional mean; it is not a continuous severity density.

All-zero minibatches train occurrence without a direct severity gradient. A fresh training population must have observations and positive examples in each execution branch; otherwise training fails with counts identifying the unlearnable branch. All-positive branches are allowed and retain their count diagnostics.

Checkpoint selection, LR scheduling and early stopping use mean_soft_bce of the combined q*m. ce_loss retains that same value for existing monitor consumers. decomposition_loss, occurrence_bce and severity_loss_contribution record optimization separately. Regularization applies once.

## Distributional severity

The optional severity distribution uses up to 65 atoms, with positive retained training-label quantiles and endpoints zero and one. The endpoints preserve the arithmetic value of unseen tiny positives without clipping. Zero is a numerical interpolation boundary, not the occurrence category. q always comes from its separate occurrence output. In particular, neither severity probability at zero nor total assembled probability at zero equals the exact non-occurrence probability.

Supports are fitted independently per outcome task, pooling positive labels across execution branches. Validation/test labels do not influence support. Metadata records support, counts, target/surface identity, label fingerprint, losses, normalization, branch order and definition hash. Resume and source-based refit inherit the definition and support exactly; changing learning mode, severity mode, atoms or target semantics is rejected. Legacy scalar, categorical and signed-gain checkpoints retain their original decoding.

## Evaluation and exports

Evaluate the parent model with the existing test.py or scripts/evaluate_relevant_models.py commands. Alongside the ordinary combined-value metrics, outcome_decomposition_metrics.json reports pooled, success/failure and existing region slices:

- Occurrence log loss, Brier score, ROC-AUC, average precision and calibration bins.
- Conditional severity MAE, RMSE, bias, soft BCE and calibration on actual positive examples only.
- Joint optimization score; distributional severity additionally gets conditional categorical CE, conditional CRPS and assembled-outcome CRPS.

One-class discrimination and empty-positive severity metrics are null with sample counts. Heads belong to one shared checkpoint; they do not acquire separate model IDs or separately selected epochs.

Use --export-outcome-decomposition for observed evaluation rows or all candidate/branch inference rows. The sidecar outcome_decomposition.parquet contains semantic scene/match, action/frame and candidate identities, execution branch, parent model, occurrence, conditional severity, expected value and definition identity. Inference rows also record runtime mask identity and forced/missing status. Forced-zero predictions have effective occurrence and value zero, and undefined conditional severity. Missing predictions remain missing.

The export option is accepted by test.py, scripts/evaluate_relevant_models.py, scripts/run_relevant_models.py, scripts/run_hawkeye.py, scripts/run_and_visualize_hawkeye.py and scripts/visualize_action_components.py. Live checkpoint comparisons export all selected models beneath decomposition_exports/. Evaluation needs --evaluation-output-dir; when combined with --learning-curve-rows it also needs --comparison-rows. Sportec exports are stored per match; model definitions remain in component provenance. Hawkeye standalone sidecars include a definitions JSON file.

With distributional severity, --export-outcome-distributions additionally exports conditional severity probability arrays and their support. Their distribution_kind is conditional_severity, to distinguish them from ordinary distributional-mode outcome distributions.

## Existing visualizations with below-dot labels

Add --decomposition to an existing individual-outcome visualization command, for example:

```powershell
--only-outcome-conceding `
--decomposition `
```

This renders the existing success and failure outcome plots. The normal expected value stays above each player dot. The sole added annotation below it is occurrence x conditional severity, using the existing three-decimal numerical precision:

```text
    0.100
      dot
0.500 x 0.200
```

There are no additional panels, colors or legends. The flag supports only the four individual components. It rejects combined outcome_success, outcome_failure, pass_score and other selected component types; select outcome_scoring and/or outcome_conceding explicitly. It also rejects --coach-ratings and --selections together with --decomposition. A decomposition checkpoint is required for live visualization.

Supported live entry points are scripts/run_and_visualize_hawkeye.py and scripts/visualize_action_components.py, including checkpoint comparisons. scripts/visualize_hawkeye.py supports standard saved Hawkeye exports generated by scripts/run_hawkeye.py --export-outcome-decomposition, including saved-run comparisons. It loads saved heads only when requested and verifies that their values match the displayed components. Older exports need regeneration; occurrence cannot be reconstructed from a scalar outcome alone. Saved loc/freeze modes currently reject the flag. Forced or missing heads display '- x -'. Away-team Sportec annotations follow the existing pitch rotation.

## Verification

The combined targeted suite passed 698 tests (plus eight subtests), with five pre-existing visualization failures excluded: three incomplete Sportec rendering fixtures and two v5 hidden-auxiliary expectations. The two v5 failures were reproduced by executing the original HEAD visualization modules in memory; the three fixture failures were previously reproduced against original code.

Focused tests cover both severity variants and outcome tasks; all four encoder architectures; observed branch gradients, zero-positive batches, weighting and extreme logits; deterministic positive-only support; actual CLI training and model loading; exact interrupted/uninterrupted resume; source refit inheritance; evaluation/inference export products; and annotation scope, content and placement.

A synthetic one-thread CPU benchmark used 128 six-node graphs, a small 1,082-parameter scalar fixture encoder and 20 timed forward/decode iterations. Scalar took 2.42 ms; decomposition with scalar severity took 2.54 ms (1,100 parameters); decomposition with 65 severity atoms took 4.64 ms (2,252 parameters). Raw output sizes were 6,144, 12,288 and 405,504 bytes respectively. This is a small-fixture measurement, not a full-match throughput estimate. The record is out/decomposition_learning_benchmark.json; a synthetic annotation preview is out/decomposition_annotation_example.png.

Correctness checks do not establish better predictive performance. Comparative full-data training and the cited Hawkeye passing-option analysis remain experiments to run with matched targets, splits and seeds.
