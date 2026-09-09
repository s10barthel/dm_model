# Endpoint policy and pass-duration selection

## Behavior

New Sportec KPI synchronization records `endpoint_policy_version=selective_reception_v1` and preserves:

- `source_event_success` and nullable `source_xml_success`;
- `source_has_reception` and `source_has_receiver_identifier`;
- `endpoint_source`, `training_sample_eligible`, and `sample_exclusion_reason`.

For passes/crosses with affirmative XML evidence that both a linked Reception and receiver/reception identifiers are absent, only explicitly unsuccessful passes immediately preceding a throw-in, goal kick, or corner in the same canonical period are eligible for repair. Either source reporting success overrides a failure report; no known result means exclusion. The source-specific rule does not apply to legacy/CSV-only inputs without this provenance.

The repair uses the last alive frame of the release's tracking episode, strictly after a live release and before the next restart frame when available. It never uses another episode, a possession-end placeholder, or an ELASTIC reception fallback for this group. Other groups remain unresolved and ineligible. Known-receiver timestamp fallback remains unchanged. Source results and all canonical events, including unresolved trailing events, remain available for return calculations.

Excluded passes cannot re-enter regular, augmented, or predefined sampling or create synthetic carry receipts. Reversed, missing, and cross-period carry receipts are rejected; equal receipts remain eligible under existing receiver checks. Missing, nonpositive, or cross-period height/trajectory intervals produce unavailable labels/graphs without swapping or clamping endpoints. Height diagnostics propagate paired NaNs, allowing consumer-specific exclusion. Existing intended-receiver modes and the height threshold/inclusion policy are unchanged.

Carry artifacts now use `sportec_fernandez_1s_v2`. New carry feature generation rejects older sidecars, including unversioned empty ones. Carry derivation also normalizes a raw `frame_id` column to the index before looking up frames: raw row positions are not tracking frame IDs. This correction can change more carries than the receipt safeguard alone.

## Training flag

Both `scripts/train_relevant_models.py` and `train.py` accept `--min_pass_dur`, default **0.5 seconds**. Values must be finite and nonnegative; zero disables duration selection, but does not disable chronology safeguards.

The wrapper forwards the same value to every selected component, including pass height, outcomes, and carry-enabled action intent. Only pass samples are duration-filtered; carries and shots are unaffected. This adds duration filtering to outcome training and carry-enabled action intent compared with previous wrapper behavior. Train components separately when their thresholds should differ.

At 25 Hz, 0.1 seconds retains intervals of at least 3 frames (0.12 seconds), 0.2 retains 5 frames, and 0.5 retains 13 frames (0.52 seconds). The threshold is stored in model configuration and run metadata and reused for validation/evaluation. Legacy saved configurations without the field retain their historical zero fallback. This is dataset selection, not a prediction-time parameter; changing it alone does not require regenerating existing feature rows.

## Three-match audit

Executed against J03WEL, J03YDU, and J04034 using fresh KPI/ELASTIC synchronization, next-action conditions **off**, and isolated outputs in `out/endpoint_policy_v1_audit_02`. Production synchronized files, sidecars, and feature runs were not overwritten.

| Result | Count |
|---|---:|
| Canonical events preserved | 3,302 |
| Repaired unsuccessful out-of-play passes | 23 |
| Excluded missing-reception passes | 12 |
| Exclusions because a source reports success | 4 |
| Exclusions because the next event is not a permitted restart | 8 |
| Remaining candidates without valid height/trajectory intervals | 8 |
| Old synthetic receipts rejected by chronology checks | 1 |

No eligible repair failed termination validation in this small subset. Synthetic fixtures additionally cover missing endpoints, equal/reversed endpoints, restart overlap, and period boundaries.

Counts below are after pass eligibility and start-snapshot checks, before component-specific label/graph filtering:

| Minimum duration | Successful | Unsuccessful | Total |
|---|---:|---:|---:|
| 0.1 s | 2,462 | 408 | 2,870 |
| 0.2 s | 2,461 | 393 | 2,854 |
| 0.5 s | 2,422 | 356 | 2,778 |

The isolated artifacts include per-action sample tables, synchronized outputs, regenerated controls/carries, `summary.json`, and `carry_attribution.json`. All release frames match the previous synchronized files. Previously stored carry counts were 700/920/1,022; full regeneration produces 929/953/1,029. These differences are not solely receipt filtering: using a correct frame index with the previous algorithm and frozen control sidecars produces 929/920/1,022; adding the receipt safeguards produces 929/923/1,022. Fresh control reconstruction also changes existing control-frame assignments in the latter two matches. In particular, the single reversed receipt had interrupted a valid carry spell; rejecting it restores three segments in the frozen-control comparison. This subset does not establish effects across the full dataset.

## Reproduction and rollout

Repeat the isolated audit with a new directory:

```powershell
.venv\Scripts\python.exe scripts/audit_endpoint_policy.py --output-dir out/endpoint_policy_recheck
```

The following are rollout commands, not commands executed by this change. Rebuild synchronization and carry sidecars for the selected matches while reusing cached tracking:

```powershell
.venv\Scripts\python.exe scripts/preprocess_sportec.py --match-id DFL-MAT-J03WEL --match-id DFL-MAT-J03YDU --match-id DFL-MAT-J04034
```

For the full dataset, omit the match selectors. Do not use `--skip-sync` or just `--carry-artifacts-only` to roll out the endpoint repair. `--overwrite` is unnecessary for synchronization and would also rebuild tracking.

After rebuilding every match used by the intended feature split, generate a fresh feature run (retain your usual split, return-type, model-variant, and graph-feature options):

```powershell
.venv\Scripts\python.exe scripts/generate_relevant_features.py --run-id endpoint_policy_v1 --next-action-conditions-off --use-carries
```

Omit `--use-carries` when unused. Do not extend an old feature run to apply this repair: old sample membership, graphs, and labels need rebuilding together. New feature metadata records `endpoint_consumer_version`; this describes the consumer code, not proof that all source files were regenerated. The next-action flag's default remains unchanged.

Then append `--min_pass_dur 0.1` (or another value) to the usual training invocation. Omit it for 0.5. No full-dataset regeneration or training was performed here.

## Verification

Focused regression tests cover XML provenance and source conflicts, selective synchronization through ELASTIC, CSV persistence, canonical history, known-receiver fallback, carry chronology and frame axes, height/trajectory availability and alignment, duration forwarding and actual dataset selection, legacy configuration defaults, and artifact versions.

Final verification: **880 tests and 78 subtests passed; two tests failed** in existing, unchanged EPV behavior (`test_epv_return_helpers_match_xt_style_semantics` and `test_generate_epv_model_selection_uses_bundle_and_explicit_overrides`). EPV behavior was not changed as part of this work. `git diff --check` passed.
