# Pass receipts, observations and training selection

The independent pass policy keeps official receipt timing, usable trajectory,
and task eligibility separate. Successful passes use Sportec/KPI receipts even
when nearby annotations, a dead launch flag, or unrelated untimed events would
invalidate a failed-pass observation. Missing duration stays unknown.

Failed-pass receiver inference uses at most the first second after launch and
requires three clean frame intervals (four samples, 0.12 seconds at 25 fps).
An official contact frame is excluded from that prefix. A flag at launch + 3
therefore rejects the observation; a flag at launch + 4 can leave enough data.
Later flags shorten the prefix. A potentially early untimed contact rejects an
observation when the clean prefix cannot be established. The tracking detector
also checks early contacts without its former search guard.

The `pass_endpoints` tables retain `official_receipt_frame_id`, receipt status
and reason, duration status and reason, and the independent official observation
bound. `receive_frame_id` describes the full endpoint; it is never the model-only
detector cutoff. The `failed_pass_trajectories` tables record the final cutoff,
clean interval count, version, and explicit quality reason by match/event identity.

All current graph models require their ordinary launch graph. Only
`success_intent` requires trajectory augmentation. Successful examples with a
known receiver remain eligible for the other tasks without receipt timing or
trajectory features. An unusable launch live-state flag can still exclude the
trajectory-augmented input while leaving receipt timing and other eligibility intact. Failed examples require a valid intended receiver and a
usable observation even when full duration is unknown. Task-specific target
checks, such as observed pass height, still apply. Selection does not remove
events from the sequence used to calculate future returns.

For these inputs, `--min_pass_dur` must be zero (the new default). A positive
value is rejected explicitly. Historical feature runs retain their recorded
duration-filter behavior when requested. Prepared and ordinary datasets apply
the same gate; prepared-cache signatures include the quality sidecar.

A successful receipt with a known receiver and usable timestamp starts a carry
candidate independently of pass-training eligibility and launch-graph availability.
The resulting carry still undergoes acquisition and spell validation. Missing
launch timing cannot become an uncertainty barrier after its known receipt.

## Regenerating artifacts

Existing runs and checkpoints are preserved. For a carry-enabled source run:

```powershell
.venv\Scripts\python.exe -m scripts.generate_relevant_features --extend-feature-run-id SOURCE_RUN --refresh-trajectory --replace-carries --num-workers 4 --worker-thread-limit 1
```

The refresh stages endpoint and quality records alongside graphs and labels,
then publishes them together. Model-mode predictions are invalidated. Train a
compatible `success_intent` model from the refreshed inputs, extend the new
feature run with that model to regenerate failed-pass receivers, then train
downstream models. Source membership is retained by a refresh; use fresh feature
generation to recover actions already absent from the source graph/label files.

Policy versions are recorded in feature metadata, prediction-cache fingerprints,
and training metadata/arguments. Resume and refit cannot adopt a changed policy.

## Source audit

```powershell
.venv\Scripts\python.exe -m scripts.audit_pass_observation_policy --feature-run-id SOURCE_RUN --output-dir tmp/pass_policy_audit --workers 4
```

Add repeated `--match-id` options for a subset. This writes per-event CSVs and a
summary without changing feature artifacts. Historical receiver availability
does not certify compatibility with the new observation policy.


## Verification

The regression suite passed 489 tests, with 97 additional physical-cache and
inference compatibility checks passing. Coverage includes ordinary/prepared dataset
selection, carry starts, historical-policy replay, interrupted refresh recovery,
and training refits. The source audit completed all 918 matches (808,618 source
pass records) and verified the original 612/153/153 cohort assignment. These are
source-record checks; final training membership requires compatible regenerated
receiver predictions.

The reported successful-pass examples now use official receipt durations:

| Event | Previous duration | Official receipt duration |
| --- | ---: | ---: |
| 18453000000074 | 0.08 s | 1.20 s |
| 18453000001429 | 0.04 s | 3.56 s |
| 18453000001057 | 0.08 s | 0.84 s |
