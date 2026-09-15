# Outcome evaluation confidence intervals

Outcome artifact generation now defaults to 2,000 whole-match bootstrap resamples,
seed 42, and 95% percentile intervals. Both `scripts/evaluate_relevant_models.py`
and `test.py` accept `--outcome-bootstrap-resamples` and `--outcome-bootstrap-seed`.
Set the resample count to `0` to disable intervals. Other model tasks do not run
this calculation. Inference runs once; resampling uses the collected predictions
on CPU.

For each replicate, draw as many matches as contributed observations, uniformly
with replacement. Include all retained observations of each selected match,
including repetitions. Metrics remain weighted by observations, as in the
existing evaluation, rather than giving each match's metric equal weight.
Sorted match IDs and a local random generator make draws reproducible; the same
draw is used for all metrics and targets. Identical match sets and seeds across
models give identical match draws.

## Outputs

`outcome_bootstrap_ci.csv` contains 11 rows for `pooled_factual`:

- Continuous training target: MAE, RMSE, Pearson and Spearman correlation,
  mean prediction minus target, linear calibration intercept and slope.
- Next-10-action goal diagnostic: ROC AUC, Brier score, logistic calibration
  intercept and slope.

Columns are `model_id`, `task`, `evaluation_target`, `stratum`, `metric`,
`point_estimate`, `lower`, `upper`, `confidence_level`, `match_count`,
`requested_resamples`, `valid_resamples`, `seed`, and `status`.
Bounds use the 2.5th and 97.5th percentiles with linear quantile interpolation.
At least two matches, a finite original estimate, and at least 95% valid
replicates are required. Undefined metrics and failed/nonconverged logistic
fits are excluded per metric. Unavailable intervals have empty bounds,
an explanatory status, and a warning.

The model's `metadata.json` includes `outcome_bootstrap` configuration,
contributing match IDs/count, elapsed seconds, and validity counts/statuses.
Disabled runs record `enabled: false` and omit the CI file. When reusing an
output directory, disabling bootstrap removes its previous CI file to avoid
presenting stale intervals. Existing point-estimate tables, comparison summary
columns, branch results, and plots retain their formats.

## Interpretation and runtime

Intervals describe held-out match sampling uncertainty conditional on the fitted
model and fixed targets. They do not include retraining, model selection, target
estimation, or distribution shift. Linear calibration describes agreement with
the continuous training target. For xT-derived predictions, logistic calibration
against goal labels is a proxy diagnostic rather than a direct assessment of
predicted goal probabilities. Match resampling preserves dependence within
matches but assumes matches are independent sampling units.

A bounded synthetic check on the development environment used 100 matches,
10,000 observations, and 50 resamples for all 11 metrics: approximately 0.57
seconds, excluding imports. This is not a production runtime estimate; sorting
for rank metrics and logistic fitting depend on sample size and data. Progress
is printed every 30 seconds between replicates and at completion. Only one
replicated dataset is held at a time. No historical evaluations were rerun.
