Outcome scoring and conceding support two learning modes:

```text
--learning-mode scalar          # default
--learning-mode distributional
--distribution-atoms 65        # maximum count, including endpoints
```

The flags are accepted by `train.py` and `scripts/train_relevant_models.py`. The wrapper applies the mode to selected `outcome_scoring` and `outcome_conceding` tasks. Other tasks retain their normal learning procedure. Distributional mode requires a GNN and retained targets in [0, 1]; unsupported tasks, out-of-range targets, `include_out` and outcome-dependent class balancing are rejected.

For example, add these options to an existing outcome-training command:

```powershell
  --learning-mode distributional `
  --distribution-atoms 65 `
```

The target family, return type, stopping rule, xT surface and absolute/gain representation are independent choices. For a controlled comparison, run scalar and distributional fits against the same feature run, labels, split, architecture and seed. Start with absolute legacy xT and `next_3` if testing whether distributional learning improves build-up discrimination. Full comparative retraining is separate from the implementation's small correctness tests.

**Model and target encoding**

Each candidate has a categorical value distribution for failure and another for success. With ordered atoms `v[k]` and predicted probabilities `p[k]`, the exported value is `sum(p[k] * v[k])`. The scalar component outputs and pass-score calculation consume this expectation in the original target units.

Training places a label between its two adjacent atoms, using interpolation weights that reconstruct its original arithmetic value exactly. Interpolation occurs in the original units. Atoms are not rounded bin centres, and labels are not log-transformed. Only the observed candidate and execution branch receive the direct categorical loss.

Atoms include 0 and 1, with interior quantiles fitted from the actual retained training labels. Repeated quantiles are removed, so the effective count can be smaller than requested. All-zero/binary populations use `[0, 1]`. Validation and test labels cannot affect atom fitting. Prepared training data is scanned for labels without reconstructing graphs or consuming training RNG state.

The saved `outcome_distribution_definition` contains the atoms, encoding version, provenance, label fingerprint, effective count, atom gaps and empirical encoded occupancy. Resume, continuation and source refit inherit the exact saved definition. Existing checkpoints without a learning mode remain scalar; changing a checkpoint's mode or support is rejected.

Zero labels map exactly to the zero atom. Small positive labels below the first positive atom also allocate interpolation mass to zero. Therefore, `1 - p[0]` is not an exact positive-target probability and must not be interpreted as turnover probability. Predictive spread describes variation in outcomes, not uncertainty about fitted model parameters.

**Training and evaluation**

`distributional_ce` is the categorical optimization loss. `mean_soft_bce` is BCE of the decoded expected value against the original scalar target. Checkpoint selection, early stopping and LR control use the expected-value BCE; `ce_loss` retains that same monitor value for existing training/reporting consumers. Regularization is applied once. Existing MAE, RMSE, correlation and calibration metrics use the decoded means. `distribution_crps` assesses the categorical distribution in original value units.

Outcome evaluation additionally reports success/failure and location slices. Build-up means the observed candidate's attacking-coordinate x is below 35 metres; advanced means x is at least 70 metres. The low-target slice uses targets at most 0.01. These definitions are fixed across compared modes; compare identical examples and target definitions. Greater visual spread among unchosen options is not labelled evidence of better accuracy.

**Optional probability exports**

Use `--export-outcome-distributions` with:

- `test.py --model_id ... --evaluation-output-dir ...`, or the corresponding evaluation wrapper. For distributional checkpoints, this exports the observed candidate/branch probabilities to `outcome_distributions.parquet`, with `outcome_distribution_definition.json` alongside it. When also exporting learning-curve rows, use `--comparison-rows`.
- `scripts/run_relevant_models.py` for Sportec component inference. A match's sidecar includes all candidate/branch distributions; support definitions are in the source-model provenance.
- `scripts/run_and_visualize_hawkeye.py` or `scripts/visualize_action_components.py` for live model diagnostics. Checkpoint comparisons write the sidecars beneath `distribution_exports/`.

Inference exports identify the model, scene/match, action, frame, candidate, failure/success branch, inference-mask configuration and support definition. The `prediction` column equals the probabilities' expectation. Evaluation sidecars use stable match, source-row, action and candidate identities. Atom values are stored once in accompanying definition/provenance metadata.

Normal inference outputs remain expected values. No new scoring-versus-conceding multiplier is introduced. Whether distributional learning improves discrimination or calibration must be established with controlled held-out evaluation.


Implementation verification covered 321 passing targeted tests, including original-scalar-decoder equality, categorical target reconstruction, all four supported encoder paths, real training/checkpoint publication for both outcome tasks, exact interrupted-versus-uninterrupted resume, source-refit support inheritance, and probability/expectation export round trips. Broader checks also encountered baseline self-loop, visualization-fixture and pc-xPass-cache failures; these were reproduced with the original relevant code.

A small CPU benchmark with one thread, 128 synthetic six-node graphs and ten timed forward-and-decode iterations measured 34.9 ms for scalar and 38.3 ms for 65-atom distributional output. Trainable parameters increased from 366,706 to 368,882. Raw output tensors increased from 6,144 to 399,360 bytes. These figures describe that synthetic workload, not full-match training performance. The benchmark record is `out/distributional_learning_benchmark.json`.


An additional explicit occurrence/severity alternative is documented in [decomposition_outcome_learning.md](decomposition_outcome_learning.md). It preserves the existing outcome components and is restricted to absolute xT targets.
