# Pass endpoints, receiver modes, and trajectory refresh

Fresh feature generation and `--refresh-trajectory` use the same synchronized official endpoint policy. No touch detector, raw-time difference, raw-time interpolation, or ELASTIC reception fallback is used to infer a pass endpoint. ELASTIC launch synchronization retains its existing policy.

## Endpoint policy

Active raw events are joined to official KPI frames using the existing CSV/AdvancedEvents readers. This includes events absent from the carry-control sidecar, such as offsides. Deleted and administrative events are excluded. KPI frames are mapped by period and raw frame ID to normalized tracking IDs. Candidates are ordered by tracking frame, with deterministic handling of simultaneous events. Source event order retains uncertainty barriers: an intervening potentially terminating event without usable official timing prevents jumping to a later contact. XML possession-end fallbacks are not promoted to receptions.

The earliest valid associated endpoint is selected from:

- Official linked reception frames, for successful passes and interceptions alike.
- Official physical-contact events, including plays, shots, other ball actions, claims, and duels.
- Official stoppages, including fouls, offsides, and the final whistle. These are recorded as stoppages, not as confirmed touches or receivers.
- Explicit out-of-play events, or an identified out boundary supported by out/restart semantics and the launch live episode.

Chronology, tracking membership, period, live-episode boundaries, and intervening restarts constrain selection. A stoppage can lie on the first dead frame. A later restart is never substituted for a contact. Arbitrary episode/period ends and `receiver_id="out"` alone do not establish an endpoint. Remaining annotation errors are accepted without tracking-based touch validation.

`receive_frame_id` is the authoritative pass endpoint, including qualifying stoppages and out-of-play boundaries. `observed_end_frame_id` is an equal compatibility alias. Resolved actions record the capped trajectory frame, endpoint source/event/version, duration, and unavailable reason. Original reception values and retired ELASTIC values remain audit data. The full prepared event table is pinned under `pass_endpoints/<match_id>.parquet`; later generation and inference reuse it by event ID. Candidate frames and durations remain available for diagnostics when a duration safeguard rejects an endpoint. Physical reception and possession annotations remain separate.

| Consumer | Interval or positions |
| --- | --- |
| Pass duration, height and observed-end marker | Launch to `receive_frame_id` |
| Failed-pass `original` | Existing distance/angle scoring; tracked ball endpoint and candidate teammates at observed end |
| Failed-pass `angle_only` | Horizontal displacement and candidates at trajectory end |
| Success-intent inputs | Net 3D displacement over the capped trajectory interval |
| Intended-receiver coordinates in every mode | Selected player's tracked position at `receive_frame_id` |

Trajectory end is the earlier of observed end and launch plus one second. The four inputs remain `[speed, direction_x, direction_y, direction_z]`; this is an early net-displacement summary, not instantaneous launch velocity or peak height. Finite start/end positions and positive elapsed time are required. Missing interior samples are tolerated. Unavailable trajectories produce unavailable graphs.

Successful passes retain the actual receiver for all modes. Failed-pass comparison model rings use the displayed model probabilities. Overlays do not replace runtime receiver labels.

## Duration safeguard

`--max-pass-duration SECONDS` accepts positive finite values and defaults to **disabled (`None`)**. An observed duration strictly greater than the limit makes its endpoint-dependent samples unavailable; equality remains eligible. The original candidate duration is retained, never shortened to the limit. The policy is recorded in run metadata and honored by visualization and dataset selection. Changing it on an existing run requires `--refresh-trajectory`.

The one-second trajectory window and training's `min_pass_dur` are separate settings.

## Refresh and composition

```powershell
python scripts\generate_relevant_features.py `
  --replace-carries `
  --retire-elastic `
  --refresh-trajectory `
  --extend-feature-run-id feature_20261019T215248_909646_f6c38c72 `
  --num-workers 6 `
  --worker-thread-limit 1
```

This creates one derived run: clone once, retire receptions, refresh canonical pass artifacts, rebuild carries from the updated canonical artifacts, then publish completion. Each match has a completion marker; failed derived runs resume with the same arguments and an explicit `--run-id` naming the failed target. Changed files are replaced atomically, preserving source hard links. Combined operations reject `--in-place`.

Standalone `--refresh-trajectory` retains derived and in-place operation/recovery. Its deprecated spelling `--refresh-trajectory-speed` remains supported. Refresh now updates:

- All four success-intent inputs and their contract tags.
- Pass duration, maximum height, height labels, endpoint coordinates and metadata across persisted variants, including repeated/synthetic source rows.
- Endpoint-dependent post-action observations.
- Both saved heuristic receiver modes, target indices and coordinates.
- Dependent temporal labels and blocked augmentation graphs/labels; synthetic membership is regenerated where it depends on the changed receiver.

Real-action identities, source selection, return targets, and unrelated graph inputs are preserved. Model-mode artifacts are excluded from the derived run, and metadata records `model_mode_regeneration_required=true` plus the previous checkpoint as provenance. Success-intent training remains available. Endpoint refreshes on carry-enabled runs require `--replace-carries`.

## Compatibility and visualization

New contracts are:

- `pass_duration_definition = canonical_official_pass_end_v3`
- `trajectory_speed_definition = canonical_official_net_displacement_3d_1s_v5`
- `angle_only_definition = canonical_official_ball_direction_1s_v3`
- `original_receiver_definition = canonical_official_distance_angle_v3`

Legacy trajectory v1/v2/v3/v4 artifacts remain recognizable for matching legacy inference. Checkpoints are never upgraded through mutable feature metadata, and incompatible input/checkpoint contracts are rejected. Prepared caches include implementation fingerprints and changed artifact signatures.

The black arrow uses the trajectory from the exact graph supplied for the displayed prediction. New graphs record their sampling frames. Legacy graphs without sampling metadata retain an unknown interval rather than inventing one. Comparison panels carry their own trajectory provenance. Current-policy endpoint markers and heuristic rings are recorded separately from the model-input contract.

## Read-only six-second audit

```powershell
python scripts\audit_pass_endpoints.py `
  --feature-run-id feature_20261019T215248_909646_f6c38c72 `
  --max-pass-duration 6 `
  --num-workers 6 `
  --count-label-rows
```

This resolves endpoints under the current policy without changing feature files or enabling a safeguard. It exports all selected unique passes, affected actions, endpoint provenance and durations, existing unavailability, and optional counts of repeated label rows. Carry copies and multiple return variants do not inflate unique-action counts. Use `--match-id` to restrict the audit.

Regression examples in DFL-MAT-J03YDU: 105 ends at frame 26767 (0.64 s), 313 at 49217 (0.16 s), 484 at official offside frame 68452 (2.40 s). Action 97 uses the earliest available synchronized contact at 24616 (3.32 s); no tracking-based correction is applied.

## Retraining workflow

1. Run the combined refresh above. Its new run provides shared graphs, success-intent training targets, both heuristic modes and rebuilt carries. Model mode is intentionally unavailable.
2. Train a new success-intent checkpoint against that updated feature run.
3. Add the new checkpoint using the existing extension interface:

```powershell
python scripts\generate_relevant_features.py `
  --extend-feature-run-id <updated-feature-run-id> `
  --intended-receiver-model-id success_intent/<new-model-run-id> `
  --num-workers 6 `
  --worker-thread-limit 1
```

The extension regenerates model-mode labels, coordinates, augmentations and carry variants before advertising that mode as available. No replacement flag is needed when the refreshed run has no active model mode. Replacing an already active model still requires `--replace-intended-receiver-model`.

The canonical endpoint contract is `canonical_official_pass_end_v3`; success-intent trajectories use `canonical_official_net_displacement_3d_1s_v5`. Older checkpoints must not be silently relabeled with the new input contract. The six-second safeguard remains disabled by default.
