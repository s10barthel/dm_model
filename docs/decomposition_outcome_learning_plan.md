# Explicit occurrence/severity outcome learning

Status: recommendations accepted and implemented. See decomposition_outcome_learning.md for usage and verification. Existing scalar and distributional modes remain supported.

## Contract and recommended decisions

- Add `--learning-mode decomposition` for GNN outcome_scoring and outcome_conceding.
- Add `--severity-mode {scalar,distributional}`, default scalar in decomposition mode. Reject an explicitly supplied severity mode for other learning modes. Offer both variants in the implementation; use scalar first to isolate decomposition from distributional learning.
- Initially require target_family=xt with explicit absolute representation in resolved feature metadata. Reject clipped and signed gain regardless of observed label range. Other target families require a future documented semantic contract rather than inferring eligibility from nonnegative values.
- Occurrence is exactly `z = 1[y > 0]`, with no epsilon threshold. Under the agreed absolute-target semantics this means a relevant team's eligible action occurs within the return window, not necessarily the instant possession changes. Preserve existing exclusions: an excluded/invalid window must not become a negative example.
- Preserve four public component values, the original units, and pass-score arithmetic. No post-hoc weighting or extra baseline.
- Treat heads as first-class evaluation outputs of one parent model, not independently trained/deployed model families.

## Architecture and decoding

Keep one shared graph encoder per existing outcome model. Each failure/success execution branch has an occurrence logit and its own conditional-severity output. Logical out_dim stays two; physical head dimensions differ by severity mode. Prefer named decoder layers for occurrence and severity to make contracts explicit. Keep existing modes and their state-dict layouts unchanged.

Scalar severity: q=sigmoid(occurrence_logit), m=sigmoid(severity_logit).
Distributional severity: q as above; conditional categorical probabilities r=softmax(severity_logits); m=sum(r[k]*atom[k]).
Both: expected_value=q*m. Existing consumers receive [nodes,2], failure first and success second.

Provide one shared decoding helper returning expected_value, occurrence_probability, conditional_severity_mean and, when requested, severity probabilities. Inference, evaluation and exports must use it consistently. This requires internal loading/decoding changes but preserves the public component interface.

## Training objective

Use only the observed candidate and execution branch, as today. For each example:

`loss_i = BCEWithLogits(occurrence_logit_i, z_i) + z_i * severity_loss_i`.

For scalar severity use BCEWithLogits against the original positive continuous target; it is a mean-consistent bounded regression loss, not a literal continuous-target likelihood. For distributional severity use existing mean-preserving two-hot categorical CE against positive labels only.

Apply existing sample/IPW weights to the complete per-example loss and average over all examples using existing weighting conventions. Do not independently normalize the positive loss by the number of positives and then add it: that changes its relative influence. No class balancing, extra scalar product loss or user-selected loss multiplier in version one. Apply regularization once. Occurrence and severity can still compete through shared encoder gradients; absence of a configurable multiplier does not mean their optimization scales are identical across severity variants.

Use masking/index selection to compute severity loss only for positives; do not rely on zero times an undefined loss. A zero-positive minibatch has finite occurrence loss and no direct severity loss. Shared encoder updates can still change severity predictions.

Report occurrence BCE over all examples, severity loss over positives for interpretation, the all-example severity contribution, and joint optimization loss separately. Keep checkpoint selection, LR scheduling and early stopping on BCE of q*m against the original target, matching existing bounded outcome selection. Use numerically stable log-domain product/complement calculations; avoid probability clipping that masks extreme-logit bugs.

## Distributional severity support

Reuse deterministic training-only quantile placement, pooling positive examples across execution branches within each task, original-unit interpolation, maximum 65 atoms by default via --distribution-atoms, float32 deduplication, and frozen saved support on resume/refit. Scalar severity must not fit atoms; reject an explicitly supplied atom count for scalar decomposition.

Recommended numerical support includes endpoints 0 and 1, with interior quantiles fitted only to positive retained training labels. Zero is an interpolation boundary for positive severity, NOT the occurrence category. This covers arbitrarily small positive validation targets without clamping to a training-set minimum and preserves target means. Occurrence is always the separate q, never one minus a severity zero-atom mass.

Alternative: strictly positive support, which has a cleaner categorical interpretation but requires a lower bound and a policy for smaller unseen positive values. Do not choose it by silently clipping labels or adding an arbitrary epsilon. A continuous conditional severity distribution is another future alternative, with a larger likelihood/optimization change.

Document that the categorical severity and its CRPS concern a numerical approximation to positive outcomes; interpolation can put mass at zero. The assembled numerical outcome distribution has mass 1-q at no-occurrence plus q*r over severity atoms, merging coincident zero atoms for diagnostics. Its total zero mass is not 1-q; export q independently.

## Degenerate training populations

Scan retained training labels and record per-branch total/zero/positive counts for both severity modes. Fail clearly if either execution branch has no training observations. Recommend also failing if a branch has no positive observations: its severity cannot be learned and should not be exported as meaningful. Report counts and affected branch, without inventing pseudo-labels. A future explicit constant-zero branch policy can be added if needed.

All-positive branches are allowed; occurrence has no observed negative examples. Record this limitation, and return undefined discrimination metrics where necessary. Constant positive and binary-positive severity populations are allowed; they still need correct support and finite training. No-positive evaluation subsets return null severity metrics plus count zero.

## Evaluation: full support without separate model families

Each parent outcome checkpoint gets reports for failure and success separately, plus pooled and existing location slices:

1. Combined expected value: existing BCE, MAE, RMSE, calibration and goal diagnostics, using q*m.
2. Occurrence: binary log loss, Brier score, ROC-AUC, average precision, reliability bins, prevalence and example counts against z. Undefined one-class metrics must be null, not fabricated.
3. Conditional severity: MAE, RMSE, mean bias and predicted-versus-observed calibration on actual positives only. Add categorical CE and CRPS for distributional severity. Never score conditional severity against zero targets from non-occurrences, or select its evaluation population using predicted occurrence.
4. Joint diagnostics: joint optimization score and, for distributional severity, CRPS of the assembled numerical distribution. Do not directly compare scalar severity BCE with categorical severity CE as if they were the same metric.

Reuse one evaluation dataset/forward pass, split membership, model ID, checkpoints and provenance. Add occurrence/severity report sections and optional comparison columns, not separate model IDs or independently selected best heads. The best occurrence epoch and best severity epoch cannot simply be combined because the encoder is shared.

## Inference and exports

Always make decomposed fields available through the structured internal prediction result. Keep standard component matrices unchanged. Add opt-in --export-outcome-decomposition across evaluation, Sportec inference and live Hawkeye/visualization workflows, following existing distribution-export patterns. The accepted visualization design is only --decomposition: retain the normal value above the dot and add occurrence x severity below it. Reject combined outcomes/pass_score and conflicts with coach ratings or selections. No additional visual panels or color encodings.

Write outcome_decomposition.parquet with model/task, match/scene, source action/frame, candidate, execution branch, runtime mask identity, q, conditional severity mean, expected_value, definition ID, and forced/missing status. Evaluation rows additionally include original target and occurrence label. Export all candidate/branch predictions at inference; label-based evaluation remains limited to observed actions. Optional severity probability arrays use --export-outcome-distributions when severity is distributional, with definitions stored once.

Handle overrides after valid raw decoding. For forced-zero predictions, set effective q and expected_value to zero, mark forced status, and leave effective conditional severity undefined; preserve raw head fields separately if exported. Missing candidates have missing fields. Do not let a numerical all-negative-infinity sentinel masquerade as a learned severity prediction. No changes to pass-score weighting, forced-zero policy or candidate alignment.

## Persistence and integration work

Persist a versioned outcome_decomposition_definition including occurrence rule, absolute target/surface definition, branch order, severity mode, support/encoding if any, training counts/fingerprint, losses and normalization, decoder contract and monitor. Hash and validate it; include it in metadata, checkpoint provenance and prediction-cache identity. Resume/source refit inherits it exactly, including support. Reject changes of mode, severity type, target semantics or support; warm-starting a new head is outside scope.

Implementation sequence:

1. models/outcome_learning.py: CLI/config validation, decomposition contract, training-label scan/support, loss and shared structured decoder, stable mean BCE, diagnostics and export helpers. Preserve the current signed-gain contract for existing modes.
2. models/gnn.py and models/observed_selection.py: branch head layout, buffers, observed selection and legacy-state compatibility. Audit shape/sentinel assumptions before changing dispatch.
3. train.py and scripts/train_relevant_models.py: early eligibility checks, source inheritance, support initialization before optimizer creation, finalized metadata, wrapper flag routing to outcome tasks only.
4. models/utils.py, training_state.py, training_refit.py and datatools/metadata_summary.py: loading/validation, training loss/monitor dispatch, resume/refit and summaries. Avoid treating every non-distributional mode as legacy scalar, or every distribution-bearing mode as the existing categorical contract.
5. test.py, models/evaluation_context.py and scripts/evaluate_relevant_models.py: head evaluation, counts, per-branch/region reports and comparison exports.
6. inference.py and existing Sportec/Hawkeye/live comparison entry points: structured head decoding, optional exports, masks, cache/provenance identity and unchanged expected component values.
7. Documentation and focused tests, then smoke training for both tasks and severity variants.

## Verification and comparative experiment

- Exact occurrence labeling, both gain types rejected from metadata even for all-positive samples, unsupported families/tasks/estimators rejected; gain support in existing modes remains unchanged.
- Correct candidate/branch gradients, finite extremes, zero-positive batches, weighted loss normalization, positive-only severity evaluation, and degenerate population policies.
- Distribution encoding reconstructs tiny positive labels; support only uses retained training positives, consumes no RNG and persists exactly. Test held-out positives below the smallest positive training label.
- Products equal exported component values for all four heads; occurrence is independent of severity interpolation at zero. Test masks, absent candidates and full inference/evaluation export round trips.
- Checkpoint publication/load, exact interrupted resume, source-refit inheritance, configuration conflicts and definition/buffer tampering.
- Legacy scalar, categorical and signed-gain inference/regression tests; CLI wrappers and cache identity.
- Controlled fits: scalar, categorical distributional, decomposition+scalar severity and decomposition+distributional severity, using identical absolute targets, split, horizon, encoder and comparable seeds/budgets. Report occurrence/severity/combined metrics and runtime; bootstrap by match if reporting uncertainty. For old modes, occurrence comparisons are unavailable or explicitly approximate, never relabeled as exact.
- Revisit the cited Hawkeye defensive options as a qualitative diagnostic, without treating their spread as counterfactual ground truth. No guarantee of stronger discrimination or calibration improvement.

## Decisions to confirm before implementation

Recommended defaults: both severity variants supported, scalar default; absolute xT only initially; full head evaluation within parent models; optional decomposed exports; numerical zero boundary for distributional severity with separately defined occurrence; fail clearly on training branches with no positives; combined expected-value BCE checkpoint selection. Implementation was subsequently authorized by the user. Production comparative retraining is a separate experiment.
