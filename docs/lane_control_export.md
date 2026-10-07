# Best-pass opponent lane-control export

Use a fresh pc-xPass run:

```powershell
python scripts/generate_physical_xpass.py --pc-xpass --export-lane-control --pc-xpass-id my_new_run
```

The optional export does not change scores or the main wide cache. It writes
`data/pc_xpass/<run_id>/<dataset>/lane_control/<match_id>.parquet`.
Each row represents a state, candidate receiver, and defending player.

Columns: match_id, action_index, physical_state_hash, frame_scope, state_frame_id,
receiver_id, opponent_id, lane_control, speed, angle, distance, target_x, target_y.
Scope and state-frame ID are nullable. Angle is in degrees; speed is m/s and distances
and coordinates use the main cache's metre-based coordinate system.

Lane control is the opponent's maximum raw interception control over samples strictly
before the selected endpoint. Nonfinite samples contribute zero; finite controls are
clipped to [0, 1]. An empty prefix contributes zero. All simulated opponents, including
zero contributions, are included. Their product of (1 - lane_control) reconstructs the
selected pass's lane_survival. Endpoint control is not included.

Selection exactly follows the main cache, including --top-xt and its tie-breaking.
--no-max is allowed. Invalid receiver options produce no rows. A coverage manifest in
lane_control/coverage records completed states, including valid empty exports.

The flag defaults off and is persisted in run settings. It cannot be enabled on a
version created without it. There is no backfill: missing sidecars for existing cache
hits raise an incomplete-coverage error, never trigger recomputation. Resume an
export-enabled version to compute unfinished states. Use a fresh version if its
existing sidecar is missing or incompatible. --dry-run does not write exports.

Only compact selected-pass contributions are returned from workers. Full grids are
not exported or retained for this feature. Benchmark runtime and output size on a
representative dataset before estimating production overhead.

## Validation benchmark

A local synthetic benchmark (8 states, 10 receivers, 11 opponents, 10-degree
angle grid, one worker, four fresh runs per mode) measured median wall time
0.384 s without export and 0.407 s with export (about 6% overhead). The sidecar
contained 880 rows and occupied 9,472 bytes; the main Parquet tables were identical.
This small synthetic result is not a production-runtime estimate.
