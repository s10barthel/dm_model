# Pass trajectories and targeted refresh

Pass and cross actions have separate observed and trajectory endpoints. Neither replaces `receive_frame_id`, so possession reconstruction keeps its existing inputs.

| Endpoint | Definition | Consumers |
| --- | --- | --- |
| Observed end | Successful pass: official KPI reception. Failed pass, or success without a valid KPI reception: next relevant official event raw time difference anchored to synchronized launch. Bound by launch episode last live frame and period end. | Pass duration, observed-end marker |
| Trajectory end | Earlier of observed end and launch plus one second | Success-intent inputs, black arrow, angle_only candidates |

Relevant events include Other Ball Action, ball won, tackles, passes, shots, restarts and stoppages. Deleted and administrative events and annotations at launch time do not terminate the segment. Contact KPI frames and ELASTIC reception frames do not determine failed-pass endpoints. Raw XML is preferred; saved control events can supply their original timestamps if XML is unavailable. There is no ball-motion contact detection or timestamp refinement.

Resolved actions record `observed_end_frame_id`, `trajectory_end_frame_id`, `pass_endpoint_source`, `pass_endpoint_event_id`, `pass_endpoint_definition` and `pass_duration`. An official live/period boundary can supply an endpoint when no later contact exists.

Success-intent inputs remain `[speed, direction_x, direction_y, direction_z]`: normalized net tracked-ball displacement and its magnitude divided by elapsed tracking time. Without tracking timestamps, elapsed time uses frame difference / frame rate. At least two finite positions and positive elapsed time are necessary. There is no event-coordinate fallback or extension beyond the endpoint. A 0.04-second segment uses just its two frames; unusable trajectories follow the existing unavailable-graph handling. Zero displacement produces a zero vector.

`min_pass_dur` uses launch-to-observed-end duration, without the one-second cap, for every component model. Omitted or zero adds no minimum-duration exclusion. Inference does not impose this training filter.

Fresh `angle_only` generation uses horizontal ball displacement and candidate positions at trajectory end, retaining angular scoring and tie-breaking rules. Comparison visualizations recompute the heuristic using this definition; rings remain at launch positions. The black arrow uses the same trajectory as the model. The Observed pass end marker uses tracked ball position at observed end and is omitted if unavailable.

## Refresh workflow

```powershell
python scripts/generate_relevant_features.py --extend-feature-run-id OLD_RUN --refresh-trajectory --run-id NEW_RUN
python scripts/train_relevant_models.py --feature-run-id NEW_RUN --only-success-intent
```

`--refresh-trajectory-speed` remains a deprecated alias for the expanded operation. Refresh updates all four success-intent inputs, duration in every applicable label variant (including repeated and synthetic rows), and persisted pass endpoint metadata. It preserves saved receiver IDs, receiver coordinates, return targets, synthetic membership, unrelated node inputs and graph structure. It does **not** upgrade saved `angle_only` labels; generate a fresh feature run for those. Model-backed receiver replacement remains a separate operation after retraining.

- Requires `--extend-feature-run-id`, inherits source selection settings, and cannot be combined with unrelated generation/refresh options.
- Derived runs share unchanged artifacts using hard links by default. Use `--artifact-copy-mode copy` for full copies. All changed artifacts are staged and replaced by complete-file renames, preserving the source run.
- `--in-place` (alias `--overwrite-feature-run`) updates the existing run and cannot be combined with `--run-id`.
- `--num-workers` and `--worker-thread-limit` control staging. Existing graphs reuse their structure; newly available success-intent graphs alone are rebuilt.
- All files are staged before publication. Interrupted replacement leaves the run unusable until repaired with `--refresh-trajectory --in-place --extend-feature-run-id TARGET_RUN`. An interrupted clone must be recreated from the original. A completed run with both current contracts is a no-op.

## Semantic compatibility

New/refreshed inputs use `trajectory_speed_definition = official_event_net_displacement_3d_1s_v3`. The old `net_displacement_3d_speed_v2` and missing-tag legacy v1 remain recognized for matching stored inputs. Training records the trajectory contract in checkpoint arguments, metadata and feature signatures. Resume/refit and inference reject incompatible inputs despite identical dimensions. Old checkpoints are never upgraded from mutable source-run metadata; on-demand construction requires a current checkpoint.

`pass_duration_definition = official_event_observed_end_v1` is recorded separately. Fresh runs record `angle_only_definition = official_event_ball_direction_1s_v1`; refresh preserves the source receiver definition (or identifies its legacy policy). Prepared cache identities include the trajectory implementation/contract; replaced label/graph signatures invalidate cached samples.

Full-corpus refresh and model retraining are explicit operations, not part of installing this change. Ball-carry/possession reconstruction is unaffected.

## Validation example: DFL-MAT-J03YDU

Read-only inspection of the stored match and in-memory refresh selected these official contacts. Directions are unit 3D vectors in native tracking coordinates; both actions are home-team passes.

| Action | Subsequent event | Observed frame | Trajectory frame | Duration (s) | Direction (x, y, z) | Live angle_only |
| --- | --- | --- | --- | --- | --- | --- |
| 47 | Kim OtherBallAction, 18453000000073 | 14576 | 14566 | 1.40 | (0.526209, 0.786587, 0.323087) | home_8 (Weiser) |
| 97 | Davies OtherBallAction, 18453000000151 | 24552 | 24552 | 0.76 | (0.552121, 0.833759, 0.002788) | home_11 |

The black-arrow XY velocities are respectively (3.86, 5.77) and (2.605263, 3.934211) m/s. Observed-end markers are (82.17, 17.62) and (63.66, 5.64). Fresh and refreshed success-intent graphs match for both actions, including unrelated node and edge inputs. All stored label variants for this match passed in-memory duration refresh/alignment checks. Existing feature files, receiver labels, models and visualizations were not overwritten.
