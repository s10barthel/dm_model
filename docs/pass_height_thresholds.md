# Pass-height target cutoffs

A feature run containing `pass_max_ball_z` can train models at different height cutoffs without copying its graphs or regenerating labels. Each training run selects one cutoff. The dataset derives `pass_high = pass_max_ball_z >= cutoff` in private label tensors; source files remain unchanged.

## Training

Add `--pass-height-threshold 1.0` to your existing training command. For example:

```powershell
.\.venv\Scripts\python.exe scripts\train_relevant_models.py `
  --feature-run-id feature_20260910T110759_053479_8b508137 `
  --train-count 765 `
  --intended-receiver-mode model `
  --return_type next_3 `
  --only-pass-height `
  --pass-height-threshold 1.0
```

Keep your architecture and feature-selection flags when conducting a controlled experiment. The metre cutoff does not change graph features, outcome return horizons, or split membership.

Without the option, new training inherits `pass_height_threshold_meters` from the selected feature run. Pass-height training fails if neither source supplies a cutoff. The resolved definition, including its source and inclusive `>=` boundary, is saved in model arguments and metadata. Resume restores it; a conflicting explicit cutoff is rejected.

`--dataset-loading disk` is supported for pass-height training without inverse-propensity weighting. `auto` retains the existing memory default for this task. Different cutoff values share the same prepared graph cache when their source paths and preprocessing match. Labels are converted after cache loading. Changes in preprocessing code may build a new cache once; old caches are not deleted automatically.

## Evaluation

Evaluation defaults to the saved model cutoff:

```powershell
python scripts\evaluate_relevant_models.py --pass-height-model-id pass_height/<run_id>
```

To deliberately score those predictions against a different height event:

```powershell
python scripts\evaluate_relevant_models.py `
  --pass-height-model-id pass_height/<run_id> `
  --pass-height-threshold 1.5
```

`test.py --model_id ...` accepts the same option. Overrides change observed targets and diagnostics, not the meaning of the model's predictions. Outputs record both model and evaluation cutoffs, target provenance, label mode, and whether an override was requested. The metre cutoff is separate from `--classification-threshold 0.5`, which controls classification of predicted probabilities.

Legacy checkpoints without a target definition continue using their stored binary labels by default. Their historical cutoff is inferred descriptively from available feature-run metadata, or reported as unknown. An explicit evaluation override uses the continuous measurements. It cannot recover missing continuous heights.

Diagnostic bands use `[T-0.5, T)` and `[T, T+0.5)` plus the lower and upper tails. CSV output records exact boundaries; empty bands remain present. If a legacy cutoff is unknown, boundary-relative slices are omitted rather than invented. Nonfinite observations are excluded when required for height targets or height-stratified evaluation. Single-class cohorts retain valid losses and report undefined ranking/calibration metrics as missing.

## Other height consumers

The option also selects the observed-height definition for other tasks, including pass-success stratification. Existing physical trajectory settings and blending formulas are unchanged.

Predicted height probabilities carry the source model's definition in model provenance, cache metadata, and component exports. In physical/xPass cache generation, the option asserts the selected height model's cutoff; it cannot convert a model trained at 2.0 m into a 1.0 m predictor. Consumers requiring a matching probability reject unknown or incompatible definitions when a cutoff is requested. Use a separate cache version for a different predicted height event to avoid mixed definitions in one cache.

Learning-curve comparisons require matching target definitions and effective evaluation definitions. Compare models at different height cutoffs as separate experiments: their positive prevalence and prediction task differ even with identical match and example membership.
