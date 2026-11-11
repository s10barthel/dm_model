Mixed xT outcome checkpoints: confirmation before inference

Status: implemented, 2026-11-11. Usage is documented in
`signed_xt_training.md`; focused regression coverage is in
`tests/test_mixed_outcome_inference.py`.

**Verified trigger**

The selected checkpoints are:

- `outcome_scoring/outcome_scoring_20261111T100349_602072_96195465`
- `outcome_conceding/outcome_conceding_20261110T080157_625632_b30a6a77`

Both use `log_progression_k5_d5` with the same fit hash, `time_10`, and
`set-pieces`. Scoring uses unclipped signed gain (identity output, weighted
MSE); conceding uses absolute targets (legacy bounded scalar output).
Their source feature runs also differ.

Reading the actual model records reproduces the reported error in
`models/utils.py::validate_model_record_consistency`. That function then
has a second blocking check for `outcome_learning_contract`. Handling only
the first exception would leave this pair blocked by the second.

**Intended behavior**

Inference reports differing xT surfaces, fit fingerprints, target settings,
and learning contracts as limitations requiring explicit acceptance. It
continues only after yes, or the existing explicit acceptance CLI flag.
Compatible selections do not prompt.

Use `scripts/comparison_warnings.py::ComparisonWarnings`, including its
existing `Proceed despite these warnings? [yes/no]` prompt. No and EOF
cancel; empty/invalid answers repeat the question. Noninteractive execution
without `--accept-comparison-warnings` cancels with instructions. There is
no timeout or implicit acceptance.

For the reported pair, one confirmation should cover the known target and
contract differences together. A suitable warning is:

> Outcome checkpoints use different xT targets: scoring predicts signed
> gain, conceding predicts absolute value. Their output/learning contracts
> also differ. The xT surface and fit fingerprint agree. Each checkpoint
> will retain its own output interpretation; derived outcome and pass
> scores combine these definitions without automatic conversion.

Warnings include both model IDs and the precise differing fields. A true
surface or fit mismatch gets its own description; do not describe every
target mismatch as a different surface. Accepting a limitation does not
convert targets or establish calibration.

**Implementation sequence**

1. Separate mismatch detection from the decision to continue.

   Add an optional warning collector to `validate_model_record_consistency`
   and forward it through `resolve_model_selection`. The collector accepts
   structured xT definition/contract differences with stable keys based on
   the selected model IDs and definitions. With no collector, retain strict
   errors for existing noninteractive library and training callers. Keep
   prompting in the CLI layer; the shared validator must not call input or
   import a scripts-layer confirmation policy.

   Collect both target and contract differences in one pass. Retain
   per-task definitions/contracts in the returned source context. Populate
   a shared definition/contract only when all relevant records agree;
   never publish the first checkpoint's definition as shared for a mixed
   pair. A missing legacy contract remains valid when individual checkpoint
   validation accepts it; describe it as the legacy output contract rather
   than corrupted metadata.

2. Integrate the existing confirmation policy into inference entry points.

   Cover `run_and_visualize_hawkeye.py`, `visualize_action_components.py`,
   the direct comparison path in `inference_visualization_comparison.py`,
   and standalone inference in `run_relevant_models.py`, `run_hawkeye.py`,
   `run_hawkeye_loc.py`, `run_skillcorner.py`, and `run_benchmark.py`.
   Include `generate_epv.py`, which also combines these model outputs.

   Reuse `ComparisonExport.policy` in comparisons so the same limitation is
   not confirmed twice. Ordinary inference creates one run-scoped policy.
   Add the existing acceptance argument only to parsers that lack it.
   Confirm after collecting applicable startup differences and before
   prediction, physical-cache computation, or rendering. Existing run
   metadata may be created to record cancellation. A genuinely new issue
   discovered later requires a new confirmation.

3. Preserve runtime feature validation for mixed selections.

   Audit `resolve_runtime_feature_run_context`: it currently compares the
   selected runtime labels against one shared xT definition. Returning None
   for a mixed definition must not silently bypass that check.

   Compare runtime metadata against the per-task source definitions.
   Prefer candidates whose full xT definition matches at least one selected
   source, subject to existing graph/cache/stop/availability requirements;
   keep the existing newest-compatible selection behavior among equally
   suitable candidates. Record which model targets the runtime labels
   match. Differences against the other selected model go through the same
   warning policy. Explicit or fallback runtime choices with additional
   surface/target differences need their own explicit acceptance.

   Describe runtime labels as their actual single definition. Any consumer
   computing native-target metrics must use that model's matching reference,
   or mark the metric unavailable; it cannot treat one mixed run's labels
   as valid native targets for both checkpoints. Keep missing artifacts,
   incompatible graph schemas, stop-condition errors, unavailable required
   labels, and physical-cache integrity errors as existing hard failures.

4. Preserve checkpoint-specific prediction behavior and provenance.

   Continue loading each model with its own validated args. Signed scoring
   retains identity decoding and negative predictions; bounded conceding
   retains its saved decoding; distributional models retain their own atom
   support. Keep the current pass-score arithmetic.

   Save per-model definitions/contracts, runtime reference information,
   warnings, and acceptance method in run/comparison metadata. Record
   rejected runs as cancelled using existing cancellation handling. Scope
   acceptance to the current run and exact warning identity. Retain
   existing component-output keys and prediction-cache identities.

5. Verify the behavior and document usage.

   Extend existing consistency, comparison-confirmation, visualization, and
   signed-outcome tests. Cover: matching pairs; representation-only,
   contract-only, actual-surface, and combined mismatches; yes/no/empty/EOF;
   noninteractive execution; explicit acceptance; warning deduplication;
   unchanged strict training/library behavior; runtime feature selection;
   and persisted mixed provenance/cancellation.

   Test prediction equivalence against running each component separately,
   including negative signed outputs, and check the unchanged combined
   arithmetic. Exercise both normal inference and comparison paths with
   small fixtures. Use the supplied IDs in a read-only metadata regression
   check and a focused local inference smoke run where inputs are available;
   do not retrain or modify those checkpoints. Add a short usage example to
   the relevant inference documentation.

**Choices and recommendations**

| Choice | Options | Recommendation |
| --- | --- | --- |
| Which xT differences can be accepted? | Representation only; all cross-checkpoint surface/target/contract differences | All three, with precise warnings. This handles the reported pair and the requested error category. |
| Confirmation frequency | Per field/frame; once per unique startup issue set | One startup confirmation covering known differences; prompt again only for a new limitation. |
| Scripted execution | Always interactive; reuse explicit acceptance flag | Reuse `--accept-comparison-warnings`; never accept automatically. |
| Training and bundle creation | Relax globally; keep existing strict callers | Keep strict behavior. This request concerns running mixed checkpoints at inference. |
| Output semantics | Automatically convert to a common target; retain each checkpoint's outputs | Retain each checkpoint's validated outputs and expose the mixed definitions in provenance. |

No additional user information is needed for this plan. These recommendations
resolve routine implementation choices; runtime confirmation remains an
explicit decision by the person executing inference.
