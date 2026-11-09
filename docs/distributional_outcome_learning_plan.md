Distributional outcome learning — implementation plan

Status: recommendations accepted; implementation and verification recorded in `distributional_outcome_learning.md`. This plan adds an alternative learning mode to existing outcome models. It does not introduce a continuation-value model or change feature generation, return horizons, stopping rules, xT surfaces, or target representations.

**Objective and scope**

Add `--learning-mode {scalar,distributional}`, defaulting to `scalar`, for `outcome_scoring` and `outcome_conceding`. Each still predicts failure-conditioned and success-conditioned values for each candidate. Distributional inference returns the arithmetic expectation in the selected target's original units, so the existing outcome and pass-score formulas remain applicable without additional scoring/conceding weights.

Support the existing GNN encoders through their shared decoder. The mode is independent of the selected target family, provided the actual retained labels are finite and in [0, 1]. Binary goal labels are a valid degenerate case, not an expected source of benefit. Signed/unbounded targets and other tasks should reject distributional mode explicitly. Selecting an existing target family does not create a new value model.

The implementation is complete when training, evaluation, inference, checkpoint loading, resume, and final refit all agree on output semantics. Demonstrating improved build-up discrimination is a separate experiment, not a correctness criterion that can be guaranteed in advance.

**Recommended distribution representation**

Use a categorical distribution over ordered value atoms, with a separate distribution for each execution branch. This is a refinement of the earlier occurrence/severity proposal: it reuses the current scalar labels directly and avoids introducing a new turnover-event definition.

For atoms `v[0] < ... < v[K-1]`, predict probabilities `p[k]` and export:

`expected_value = sum(p[k] * v[k])`.

For a label `y` between adjacent atoms `v[i]` and `v[i+1]`, encode it as:

`t[i] = (v[i+1] - y) / (v[i+1] - v[i])`

`t[i+1] = (y - v[i]) / (v[i+1] - v[i])`.

All other target masses are zero. Exact atom hits, including 0 and 1, are one-hot. This gives `sum(t) = 1` and `sum(t * v) = y`: no bin-centre rounding or change to the arithmetic training label. Interpolation must be in original value units, even if atoms are spaced nonlinearly.

At the unrestricted population optimum, cross-entropy recovers the conditional mean encoded distribution; its decoded mean is therefore the original conditional mean target. Finite data, regularization and limited model capacity can still produce calibration error, so this is a consistency property rather than a performance guarantee.

Optimize soft-target categorical cross-entropy using log-softmax, selecting only the observed candidate and observed execution branch as in current training. Preserve existing sample/IPW weighting semantics. Do not add an auxiliary scalar loss or outcome-dependent class balancing in version 1.

An explicit zero atom represents zero-valued labels exactly. However, positive labels between zero and the first positive atom also allocate some interpolation mass to zero. Consequently, `1 - p[0]` must not be advertised as a calibrated turnover probability or exact probability of a positive continuous target. If that exact occurrence probability is required, choose the hurdle alternative below.

**Atom placement and provenance**

Proposed default: `--distribution-atoms 65`, interpreted as a maximum requested count. Include endpoints 0 and 1 and fill the interior with deterministic quantiles of strictly interior training labels. Use the actual retained training population after the existing eligibility, missing-timing, empty-window and graph-validity filters. Pool success and failure branches for atom fitting within a task; scoring and conceding fit independently.

Deduplicate repeated quantiles. The effective number of atoms may be smaller than requested. With no interior labels, use [0, 1]; for binary targets, this reduces to a two-category model with the same expected-value objective as binary scalar prediction. Do not fit from validation or test labels. Endpoints ensure that validation values outside the observed training range remain representable.

Quantile fitting is a placement heuristic, not a guarantee that every build-up value is sufficiently resolved. Record atom gaps, occupancy, positive-label coverage and effective counts. Add low-value validation diagnostics; increase atom resolution in a subsequent controlled experiment if needed. Fitting and atom assignment must be deterministic and preserve training RNG state and sample order.

Read retained labels through the in-memory dataset or a labels-only prepared-dataset iterator; do not rebuild graphs solely to fit atoms. For distributional fresh training, resolve the split and support before final decoder construction, optimizer creation and finalized args/checkpoint serialization. Keep the existing scalar initialization sequence intact. A prepared-label scan should not change the meaning of the graph cache.

Persist a versioned `outcome_distribution_definition` with the representation, encoding, ordered atoms, requested/effective count, support-fitting policy, target definition, training membership/data fingerprint, selected-label count and a definition hash. Store it in args, metadata and checkpoints; keep the atom values available as model buffers for decoding. Validate agreement on load.

Each validation fold fits its own training-only support. Resume, continuation and source-based final refit inherit the saved support exactly. Final refit must not silently redefine output atoms while replaying a source recipe. A genuinely new model may fit a new support.

**CLI and compatibility contract**

- Add the flag to `train.py` and `scripts/train_relevant_models.py`. In the wrapper, apply it only to selected outcome-scoring/conceding tasks. Reject a distributional request when no eligible task is selected, rather than silently ignoring it.
- Keep all other selected tasks on their existing paths. Direct calls requesting distributional learning for unsupported tasks/estimators should fail early.
- Missing learning-mode metadata means legacy scalar. Preserve loading of old two-logit checkpoints and their numerical inference behavior.
- Inference/evaluation obtains the mode and support from the checkpoint; an inference flag must not reinterpret trained outputs.
- Reject attempts to resume scalar weights as distributional, or to change a distributional run's atoms, encoding or mode. A warm-start/head-migration workflow is outside this implementation.
- Permit scalar and distributional components in one bundle if their target definitions meet existing compatibility checks. Different learning modes are not different value units. Record per-model learning identities rather than imposing a shared atom grid across scoring and conceding.

**Implementation sequence and code locations**

1. Add `models/outcome_learning.py` for argument normalization, supported-task/range validation, versioned definitions, support fitting, target encoding, output reshaping, distributional loss, and expected-value decoding. Keep these functions independent of plotting and feature generation.
2. Integrate configuration and training-only support fitting into `train.py` and `scripts/train_relevant_models.py`. Add the labels-only prepared-data access needed by support fitting without changing graph transformations, retained rows or sampling semantics.
3. Extend the shared decoder in `models/gnn.py`. Scalar output remains `[nodes, 2]`; distributional output has the explicit contract `[nodes, 2, K]`, with branch 0=failure and 1=success. Keep logical branch count distinct from the final linear layer's `2*K` output width. The shared encoders do not need separate implementations. Handle or explicitly reject any unsupported `include_out` combination before training.
4. Update `models/utils.py` training/evaluation and verify `models/observed_selection.py` selects `[batch, K]` for the observed branch. Do not spread branch slicing and reshaping across callers. Compute expectations through the shared decoder helper, never sigmoid over categorical logits.
5. Update `inference.py` and all outcome consumers to use the same helper. Existing component tables and visualization inputs continue to receive `[nodes, 2]` expected values. Check both Hawkeye and Sportec, batching, masks, absent candidates, self/carry handling and candidate ordering. Apply existing forced-zero/missing-value semantics to decoded expectations as appropriate; do not softmax an all-negative-infinity categorical vector, which would produce NaNs. Leave pass-score arithmetic unchanged.
6. Update loading/provenance in `models/utils.py`, checkpoint integrity in `training_state.py`, and recipe inheritance in `training_refit.py`. Add learning-mode/distribution identity to model records, bundle provenance and applicable prediction-cache identities. Graph caches may remain reusable when their actual inputs and labels are identical.
7. Extend evaluation/reporting in `test.py`, `scripts/evaluate_relevant_models.py`, comparison scripts, training monitoring and `datatools/metadata_summary.py`. Audit any code that assumes `ce_loss` is simultaneously the optimization objective and checkpoint-selection metric.
8. Add documentation, CLI examples and focused automated tests, then perform a small training/evaluation/inference smoke run. Full comparative retraining is a separate final validation phase.

**Loss reporting and checkpoint selection**

Keep optimization loss and expected-value metrics explicitly separate:

- `distributional_ce`: categorical loss used for distributional backpropagation.
- `mean_soft_bce`: BCE against the original scalar label, using the decoded expectation.
- Existing mean-target metrics: MAE, RMSE, correlations and calibration.
- Distribution diagnostics: CRPS in original units and atom occupancy; report predictive spread as outcome dispersion, not confidence in model parameters.

Recommended checkpoint/early-stopping/LR monitor: validation `mean_soft_bce`, since the downstream product is the expected value used in pass scores. Scalar mode uses its equivalent existing BCE. Persist the monitor identity and ensure selection, rollback, histories and refit recipes refer to it consistently. Record training distributional loss separately instead of comparing it directly with scalar BCE. A distributional-loss monitor is an alternative requiring a deliberate choice.

Use numerically stable expectation/BCE computation, including endpoint-only support and extreme logits. Apply existing regularization once; report parameter count, L1 contribution and head size so their changed contribution can be assessed. Do not change scalar regularization or stopping behavior incidentally.

For validation, retain pooled metrics and add execution-branch and predeclared build-up/advanced-location slices. Scalar and distributional comparisons must use identical observed examples and target definitions. Candidate counterfactuals are not labelled ground truth: do not treat greater spread in one visualization as an accuracy test.

**Exports**

Recommend retaining expected values in standard component outputs. Add an opt-in distribution sidecar to evaluation/diagnostic exports, containing stable scene/action, frame, candidate and execution-branch keys, atom probabilities and the support-definition ID. Store atoms once in accompanying metadata. This allows inspecting the three defensive options without multiplying the size of every ordinary inference export. The zero atom is not a turnover-probability export under the recommended encoding.

**Required verification**

- Encoding sums to one and reconstructs original labels for endpoints, exact atoms, tiny positives and nonuniform gaps. Reject invalid ranges and nonfinite retained labels explicitly rather than silently clipping them.
- Deterministic training-only support fitting, repeated quantiles, all-zero/binary/constant labels, and values outside the training-observed range but inside [0, 1]. Validation/test labels cannot affect the support.
- Correct candidate and success/failure selection; finite gradients; unobserved candidate/branch outputs receive no direct label loss. Both supported tasks and representative shared encoders work with PyG and packed batches.
- Decoding consistency across training, evaluation, standard inference and visualizations. Output probabilities sum to one, and exported means equal the stored-probability expectation.
- Legacy scalar checkpoint predictions remain unchanged within existing numerical tolerance. Test full save/load/resume equivalence, source refit inheritance, conflicting configuration rejection and mixed-mode bundle compatibility.
- Smoke-test actual forward/backward, checkpoint publication and inference exports with tiny fixtures; benchmark additional memory/runtime before full experiments.
- Compare fresh scalar and distributional runs on identical data, architecture, seeds, horizon and target definitions. Start with absolute legacy xT and `next_3`; do not change the horizon or baseline in that first comparison. Also retain the favored progression-surface pair as a qualitative reference, not the same-target scalar control. Evaluate calibration and discrimination on held-out matches and multiple build-up situations, including `41558c01-d87a-4efd-b0e9-ddfcf8470a77`.

**Choices to settle before implementation**

| Choice | Options | Recommendation |
|---|---|---|
| Distribution representation | Categorical, mean-preserving interpolation; or explicit positive-occurrence gate plus conditional severity distribution | Categorical interpolation for version 1: unchanged label source, exact label-mean reconstruction and simpler integration. Choose the gate if a separately calibrated `P(target > 0)` is a requirement; it still does not automatically mean turnover probability. |
| Atom placement | Uniform; fixed nonlinear; or training-only quantile atoms | Training-only quantiles, up to 65 atoms including endpoints, shared across execution branches within each task. |
| Checkpoint monitor | Expected-value BCE; or distributional cross-entropy | Expected-value BCE for the exported pass-value use case; log both throughout training. |
| Scope and export defaults | xT-only versus existing bounded outcome targets; full distributions everywhere versus optional sidecars | Generic bounded outcome targets for the two tasks, with scalar defaults and opt-in distribution sidecars. Initial experiment uses legacy absolute xT. |

Selecting the recommended categorical implementation means the first release does not include a separate learned occurrence head. This distinction from the earlier discussion is intentional and should be agreed before coding.
