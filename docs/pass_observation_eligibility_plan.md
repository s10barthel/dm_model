# Plan: separate pass receipts, trajectory observations, and training eligibility

Status: implementation plan, 2026-10-29. Policy decisions marked pending below require an explicit user answer. This document does not authorize a data rebuild or model-training run.

## Agreed behavior

- Successful passes use official Sportec/KPI receipts. Generic contacts, unrelated missing event timestamps, and launch live-state flags must not overwrite or invalidate an otherwise usable successful receipt.
- Missing or inconsistent receipt timing is recorded explicitly. It does not by itself invalidate successful-pass supervision for outcome, pass-intent, pass-success, or action-intent models.
- Failed passes retain conservative contact/stoppage flags. A later flag limits the observation window; it does not reject the action if enough clean trajectory precedes it.
- Failed-pass receiver inference requires at least three clean frame intervals: 0.12 seconds at Sportec's 25 fps. Contaminated contact frames do not count.
- The first-second observation window must be constructible independently of the full pass endpoint.
- A failed pass with valid intended-receiver supervision and a usable observation can remain in the affected training datasets even when full duration is unknown.
- Replace the blanket minimum-pass-duration filter with the failed-pass quality decision for the affected tasks. Successful passes are exempt from the failed-pass minimum.
- Excluding an action as a training example does not delete it from the underlying real-event sequence or shift other actions' future-return targets.
- A successful receipt with a known, usable timestamp and receiver is control evidence for starting a carry candidate, independently of pass-training eligibility. The handling of subsequent carry validation is pending below.

## Technical findings that affect the plan

### Base graph versus trajectory graph

All current affected models consume a graph at the action's launch. `dataset.py` skips `graph is None`, and `models/utils.py` calls the downstream model on `batch_graphs`. A receiver label cannot replace that input.

`success_intent` additionally consumes trajectory features. Its augmented graph can be unavailable while the ordinary launch graph remains usable. This distinction allows successful examples to survive missing trajectory inputs without changing the other model architectures.

Pending decision G below determines whether this distinction captures the user's meaning of "launch graph."

### Current carry start coupling

`datatools/carry_policy.py::reconstruct_spells` currently suppresses receipt evidence when `pass_endpoint_source` is present and differs from `official_reception`. It also consults `training_sample_eligible` and launch/receipt interval validity. Consequently, a successful receipt can disappear as a carry start because the shared pass endpoint was replaced or rejected.

After receipt evidence is created, acquisition distance, tracking continuity, live-state, separation, and terminal-control checks can still invalidate the reconstructed spell. Creating a carry candidate and accepting a complete carry are separate decisions.

## Implementation sequence

### 1. Define and version independent records

Record by stable match/event identity:

- Receipt: actual receiver, official receipt frame and source, validity status, and reason.
- Duration: value when derivable, status, and reason; missing is not zero.
- Observation: launch, last clean frame, cutoff event/source, clean interval count, quality status, and reason.
- Intended receiver: observed or predicted identity, validity, prediction model and input-policy provenance.
- Task eligibility: decision and explicit reason for each task, including task-specific graph availability.

Preserve original source fields. Do not overload `receive_frame_id` with a detector's clean-trajectory cutoff. Introduce new policy versions for changed endpoint, trajectory, carry-start, and eligibility semantics; leave historical metadata interpretable under its recorded versions.

Primary modules: `datatools/pass_trajectory.py`, `datatools/endpoint_policy.py`, `datatools/trajectory.py`, feature metadata helpers.

### 2. Restore successful receipts and separate failed-pass observation bounds

- Load successful receipts from original Sportec/KPI data, not an endpoint column already overwritten by a previous refresh.
- Give the successful receipt precedence. Check event/receiver identity, timestamp mapping, period, and ordering; treat problematic timing as a timing issue, not a universal action exclusion.
- Keep failed-pass endpoint information when defensible, but do not require it before constructing an observation of at most one second.
- Apply conservative failed-pass contact/stoppage bounds and tracking-quality checks to the observation only.
- Restrict event association to relevant periods/actions and audit raw/synchronized ordering conflicts. Do not silently reinterpret an earlier action as the end of a successful pass.
- Resolve untimed potentially relevant flags according to pending decision U.

Primary modules: `datatools/pass_trajectory.py`, `datatools/graph_feature.py`, `datatools/failed_pass_trajectory.py`.

### 3. Enforce the failed-pass minimum before receiver inference

Use the final clean endpoint after applying all cutoffs. At 25 fps, require `last_clean_frame - launch_frame >= 3` and the tracking samples required by the feature calculation.

Boundary cases:

- Contact at launch + 3, with that frame contaminated: last clean frame is launch + 2; reject.
- Contact at launch + 4: last clean frame is launch + 3; the minimum is satisfied if the prefix is otherwise usable.
- No early flag and a complete first second: eligible for inference even without a full pass endpoint.

The existing 0.12-second detector search guard is not the minimum-length implementation. Ensure early flags cannot be skipped and then reported as a clean observation. Keep detection evidence requirements explicit.

Only failed passes receive this rejection threshold. Successful `success_intent` training examples still require computable features, but do not inherit the failed-pass minimum. Apply one versioned observation definition consistently in training feature generation, on-demand inference, refreshes, and cached prediction reuse.

Primary modules: `datatools/failed_pass_trajectory.py`, `datatools/failed_pass_refresh.py`, `datatools/match.py`, `datatools/receiver_reuse.py`.

### 4. Apply task-specific eligibility and retire the blanket duration filter

Subject to decision G:

| Use | Required inputs |
|---|---|
| Successful passes in outcome/pass-intent/pass-success/action-intent training | Usable task base graph, observed receiver valid for that task, valid target |
| Failed passes in those tasks | Usable task base graph, passed failed-pass quality check, valid intended receiver, valid target |
| Successful examples for success-intent training | Known receiver and usable trajectory-augmented input |
| Failed-pass intended-receiver inference | Usable base graph plus passed trajectory check |
| Endpoint-dependent supervision | Its actual endpoint-dependent inputs/targets in addition to the task's other requirements |

Remove automatic exclusion through `duration >= min_pass_dur` for the affected tasks under the new policy. Update entry points and saved arguments so a legacy positive option cannot silently reintroduce the filter. Preserve explicit historical-policy replay, and make obsolete new-policy option combinations fail clearly instead of silently ignoring them.

Keep candidate-node mapping, finite targets, and structural input validation. Record a missing base graph separately from a missing success-intent trajectory graph or missing intended receiver. Apply identical eligibility in ordinary, prepared, and packed loaders, validation, and inference where relevant.

Primary modules: `dataset.py`, `models/dataset_config.py`, dataset-loading/preparation code, training/evaluation entry points and scripts.

### 5. Restore carry starts from independent receipt evidence

- Feed official successful receipt identity/time directly into control reconstruction.
- Do not require a valid pass trajectory, pass duration, intended-receiver prediction, or pass-training eligibility to create that receipt control event.
- Do not make receipt evidence depend on the old `pass_endpoint_source` classification or availability of the pass launch graph.
- Deduplicate companion control evidence and preserve the receipt frame as the acquisition time.
- Audit candidate creation separately from acceptance/rejection of the resulting carry.
- Apply the agreed choice from decision C to acquisition and subsequent spell validation.

Primary modules: `datatools/carry_policy.py`, `datatools/reception_policy.py`, `scripts/replace_carries.py`, carry sidecars and metadata.

### 6. Add focused regression coverage and a source-data audit

Test the identified failure modes and agreed boundaries:

- Successful KPI receipt survives an earlier generic tackle/foul, an unrelated untimed event, and a dead launch flag.
- Known successful receiver with unavailable duration remains eligible for the affected tasks.
- Failed receiver inference works from a valid first-second window without a full endpoint.
- Two clean intervals fail, three pass; contaminated contact frames never count.
- A later contact truncates without rejecting a sufficiently long prefix.
- Missing intended receiver excludes the four named downstream task families.
- Missing trajectory-augmented input does not suppress a successful example whose base input exists.
- Official receipt creates carry control evidence even if the pass is excluded from training or has no pass-launch graph.
- Candidate carry creation and full spell validation obey decision C.
- Task filtering preserves the real-event sequence and return-target alignment.
- Ordinary/prepared/packed loaders agree; incompatible policies cannot reuse old caches/predictions.

Use the audited real examples, including `DFL-MAT-J03YDU` events `18453000000074`, `18453000001429`, and `18453000001057`, as source checks. Cover the corresponding logic with small synthetic fixtures suitable for routine tests.

Then audit all 918 source matches while retaining the existing 612/153/153 training/validation/test assignment. Report every membership change by stable event identity, success/failure, reason, and task. Report carry starts recovered separately from final carries accepted. Do not use the test split for policy/model tuning.

### 7. Rebuild derived artifacts and retrain in dependency order

This is a later execution stage, after implementation verification and decisions below.

1. Create a new feature run from original provider evidence and the chosen source run; preserve previous artifacts.
2. Regenerate receipt/duration records, observations, successful success-intent inputs, and carry reconstruction under the new versions.
3. Train a compatible success-intent model when its input definition has changed.
4. Generate failed-pass intended receivers using that model and the new quality gate.
5. Build final task datasets and train downstream models in dependency order.
6. Compare full-cohort results and matched-example results separately; report counts, exclusions, and carry effects alongside performance.

Include the policy versions and relevant source/model identities in dataset, feature, receiver-prediction, and downstream cache fingerprints. Do not patch old checkpoints' recorded input semantics to label them compatible with new inputs.

## Pending decisions

### G. Meaning of launch graph

- **Recommended:** retain the base graph requirement for all current graph models; only success-intent requires trajectory-augmented input. A missing success-intent graph does not automatically mean the base graph is missing.
- **Alternative:** allow downstream models to train without a base graph. This requires a new input representation/model architecture and is a separate project.

### C. Meaning of "a successful receipt can always start a carry"

- **Recommended:** always create receipt control evidence/a carry candidate from the official receiver and usable receipt timestamp; retain validation of the resulting carry interval.
- **Alternative:** also let official receipt evidence override existing carry-acquisition rejection checks. Identify which checks should be overridden before implementation; structural impossibility and missing interval data still need explicit handling.

### U. An untimed contact flag potentially inside the early observation window

- **Recommended conservative option:** a missing timestamp elsewhere does not reject the observation, but if relevant evidence cannot rule out a contact inside the proposed clean prefix, record an explicit uncertain-observation rejection. Bound or truncate the uncertainty where reliable evidence permits it.
- **Alternative:** ignore untimed flags and rely on tracking-based detection. This can retain more examples but can also admit a contaminated prefix.

These choices remain pending until the user answers; elapsed time does not resolve them.
