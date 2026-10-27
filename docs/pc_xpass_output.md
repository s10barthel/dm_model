# Sportec pc-xPass output

For newer [aerial-enabled runs](pc_xpass_high_pass.md), receiver details also
include `__elevation` (degrees), `__target_z` (metres) and `__flight_time`
(seconds). These describe the best ranked option; `__angle` remains azimuth and
`__speed` means total 3D release speed. Ground selections have zero elevation
and height. Top-pass retains one representative per initial total speed and
azimuth, selecting its best elevation and endpoint. Historical aerial caches
retain their recorded speed/azimuth/elevation grouping.
The specific historical run documented below is ground-only.

Directory: `data/pc_xpass/pc_xpass_20261013T223051_439659_c2a48dc1_2f6b95d5/sportec/`

This run stores physical pass-completion estimates for match `DFL-MAT-J040A0`,
using `sportec_possession_states_v3` and cache format `match_physical_states_v3`.
Selection used `scope=frames`, `frames=5`: exact endpoints/canonical observations
plus interior frames sampled at a stride of five where control is supported.

| File | Format and contents |
| --- | --- |
| `matches/DFL-MAT-J040A0.parquet` | Wide Apache Parquet table: 6,815 physical-state rows and 717 columns, including receiver-specific estimates and selected-pass geometry. |
| `lane_control/DFL-MAT-J040A0.parquet` | Long Apache Parquet table: 749,650 rows and 19 columns; one row per state, candidate receiver, and opponent for the selected pass. |
| `completion/DFL-MAT-J040A0.json` | JSON checkpoint with completion status, output row counts/checksums, state and lane-export coverage, source provenance, and the possession manifest. |
| `metadata.json` | JSON run configuration, available metrics, physics/ranking settings, source fingerprints, sampling reports, and runtime statistics. |

## Main table

Identify a physical state by `(match_id, state_id)`. Its identity includes period,
tracking frame, actual possessor, and reconstruction provenance. `state_frame_id`
is the tracking frame; `action_index` repeats that frame and is **not** a canonical
event index. `frame_scope` is `frame_id` in this run. `physical_state_hash` checks
physical inputs. `period_id`, `possessor_id`, and `possession_provenance_id` retain
context. `possession_id` aliases `state_partition_id`, an internal worker grouping
by period/possessor; semantic possession spells and event links live in the manifest.

Receiver columns use tracking IDs such as `away_17` or `home_27`:

| Column pattern (`<player>` is the receiver ID) | Meaning |
| --- | --- |
| `<player>` | Default estimate; equals `<player>__max_xpass` in this run. |
| `<player>__max_xpass` | Completion estimate for the highest-ranked simulated option. This run ranks by **xPass x endpoint xT**, so this need not be the largest raw xPass. |
| `<player>__top10_xpass`, `__top25_xpass` | Mean completion estimate over the best 10 or 25 ranked grid options. |
| `<player>__top_pass5_xpass`, `__top_pass10_xpass` | Mean over the best 5 or 10 ranked distinct passes, retaining the best endpoint per speed/angle pair. |
| `<player>__lane_survival`, `__control_prob` | Selected pass's lane survival and the receiver's normalized endpoint control. Position discount is disabled here, so their product gives raw xPass before numerical clipping. |
| `<player>__top10_lane_survival`, `__top25_lane_survival`, `__top_pass5_lane_survival`, `__top_pass10_lane_survival` (and matching `*_control_prob`) | Separate means of these components over the corresponding selected options. The product of these means need not equal the mean xPass. |
| `<player>__speed`, `__angle`, `__distance`, `__target_x`, `__target_y` | Selected pass's launch speed (m/s), direction (degrees), travel distance (m), and endpoint (m in the centered pitch coordinate system). |
| `<player>__distance_to_nearest_opponent` | Receiver's distance to its nearest opponent (m). |

`pass_distance` and `ball_z` are state input fields in metres, separate from each
receiver's selected simulated distance. Estimates are floating-point values;
missing/inapplicable options remain null/NaN. Goal-node columns can also be present
in the wide schema; use player IDs for receiver analysis.

## Opponent lane controls

The long table carries the same state identifiers plus `receiver_id`, `opponent_id`,
`lane_control`, and selected-pass geometry (`speed`, `angle`, `distance`, `target_x`,
`target_y`). `lane_control` is each opponent's maximum interception control over
lane samples strictly before the endpoint. For a state/receiver,
`product(1 - lane_control)` across opponents reconstructs the selected pass's
`__lane_survival`; endpoint control is separate. These rows describe the single
selected pass, rather than the top-N averages.

Read either table with `pandas.read_parquet(path)`. For details, see
[cache/state identities](pc_xpass_caches.md) and [lane controls](lane_control_export.md).
