# Sportec inference output

Directory: `data/component_runs/sportec/component_20261014T001732_245661_c1de161b/`

This completed run contains component predictions for match `DFL-MAT-J040A0`.
It uses `sportec_possession_states_v3`, with `scope=frames`, `frames=5`: exact
endpoints/canonical observations plus supported interior frames at a stride of five.
Each component has 6,815 physical-state rows.

| File | Contents |
| --- | --- |
| `metadata.json` | JSON run configuration and provenance: model/checkpoint IDs, source feature run, sampling settings, pc-xPass dependency, auxiliary models, processed/skipped matches, and completion status. |
| `DFL-MAT-J040A0/possession_manifest.json` | JSON reconstruction of semantic spells, physical observations, and canonical action-to-state associations, including endpoint/successor links and exclusions. |
| `DFL-MAT-J040A0/action_intent.parquet` | Action-option probabilities, including player options and goal nodes for shooting; 43 columns. |
| `DFL-MAT-J040A0/pass_intent.parquet` | Receiver-selection probabilities conditional on passing; 41 columns. |
| `DFL-MAT-J040A0/pass_success.parquet` | Completion estimates for candidate receivers; 41 columns. The possessor column represents the self/carry option. |
| `DFL-MAT-J040A0/pass_height.parquet` | Probability that a pass to each candidate player reaches a maximum ball height of at least 2.0 m; 41 columns (11 identifiers/context fields and 30 player columns). |
| `DFL-MAT-J040A0/outcome_scoring_success.parquet` | Predicted attacking outcome value conditional on a successful option; 41 columns. |
| `DFL-MAT-J040A0/outcome_scoring_failure.parquet` | Predicted attacking outcome value conditional on a failed option; 41 columns. |
| `DFL-MAT-J040A0/outcome_conceding_success.parquet` | Predicted conceding outcome value conditional on a successful option; 41 columns. |
| `DFL-MAT-J040A0/outcome_conceding_failure.parquet` | Predicted conceding outcome value conditional on a failed option; 41 columns. |

The outcome checkpoints use target family **xT** and return type **`next_3`**;
interpret their outputs as learned xT-based values over that horizon. Pass success
uses the pc-xPass run described in [pc_xpass_output.md](pc_xpass_output.md), selecting
`top-pass10` with weighting version `v5` and runtime height/intent auxiliaries.
Use the exported `pass_success` values directly when computing `pass_score`.

## Table format and alignment

All prediction files are wide Apache Parquet tables, readable with
`pandas.read_parquet(path)`. The 11 identifier/context columns are
`stats_perform_match_id`, `action_id`, `original_event_id`, `state_id`,
`state_partition_id`, `possessor_id`, `period_id`, `state_frame_id`, `frame_role`,
`pc_cache_match_id`, and `state_contract`. Remaining columns contain floating-point
values keyed by tracking player IDs such as `away_17` and `home_27`;
`action_intent` additionally contains `away_goal` and `home_goal`.

Align tables by `(stats_perform_match_id, state_id)` and by player column name.
`action_id` equals `state_id` here; `original_event_id` is null for these synthetic
state rows. Use the manifest to recover canonical event associations.
`state_frame_id` identifies the tracking frame, `frame_role` records observation
roles (potentially joined by `|`), and `state_partition_id` is an internal
period/possessor grouping. `pc_cache_match_id` identifies the provenance partition;
it is not the actual match ID. To join the physical cache, match
`stats_perform_match_id` to `match_id` and use the shared `state_id`.

Null/NaN cells represent missing or inapplicable options, including opponents and
players absent from that state. Retain them as missing.

## Pass-height predictions

`pass_height.parquet` contains independent sigmoid probabilities in `[0, 1]` for
each state/player pair. They do not sum to one across players. A value of `0.9`
means an estimated 90% probability that the candidate pass's maximum ball height
is **at least 2.0 m**; the value is a probability, rather than a height in metres.
The target is `pass_max_ball_z >= 2.0`, with the inclusive boundary recorded in
`metadata.json` under `pass_height_probability_definition`.

The exported checkpoint is
`pass_height/pass_height_20260703T120403_205360_c72e946d`, also selected as this run's
height auxiliary for v5 pass-success blending. Height uses the same state keys and
player-column naming as the other components; exclude the possessor's self option
when interpreting candidate passes. Missing cells remain null/NaN.

The `pass_score` formula below uses the exported `pass_success`, which already
includes the configured height-dependent blending. No extra multiplication by
`pass_height` is needed. See [pass-height cutoffs](pass_height_thresholds.md) for
details on the height event and its provenance.

## Computing `pass_score`

For each state and candidate receiver, compute the expected net outcome from the
possessing team's perspective:

```text
pass_score = pass_success * (outcome_scoring_success - outcome_conceding_success)
           + (1 - pass_success) * (outcome_scoring_failure - outcome_conceding_failure)
```

The following example runs from the repository root and produces a wide score
table indexed by match and state, with one column per player:

```python
from pathlib import Path
import pandas as pd

match_dir = Path(
    "data/component_runs/sportec/"
    "component_20261014T001732_245661_c1de161b/DFL-MAT-J040A0"
)
keys = ["stats_perform_match_id", "state_id"]

def load_values(name):
    table = pd.read_parquet(match_dir / f"{name}.parquet")
    players = table.filter(regex=r"^(home|away)_\d+$").columns
    return table.set_index(keys, verify_integrity=True)[players]

p = load_values("pass_success")
s = load_values("outcome_scoring_success")
f = load_values("outcome_scoring_failure")
cs = load_values("outcome_conceding_success")
cf = load_values("outcome_conceding_failure")
pass_score = p * (s - cs) + (1 - p) * (f - cf)
```

Pandas aligns state keys and player columns automatically and preserves missing
operands. For pass-only rankings, exclude the `possessor_id` option in each row.
Higher scores indicate better expected net outcomes; negative scores are possible.
`pass_intent` is not part of the per-option score; weighting scores by pass intent
and summing across receivers produces the corresponding state EPV.

## Repairing missing v5 pass-success exports

For a Sportec v3 run that saved `model_pass_success`, `pass_height`, and
`pass_intent` but rejected its physical cache before blending, use:

```powershell
.venv/Scripts/python.exe scripts/repair_component_pass_success.py --component-run-dir data/component_runs/sportec/<component_run_id> --workers 4
```

The repair requires unmasked exports from the selected height and intent
auxiliary checkpoints, v5 weighting, and no carries or ball-z limit. It verifies
cache checksums and provenance, aligns physical keys across feature runs, checks
offside zeros against tracking, and reproduces inference's float32 xPass clipping
before applying the recorded v5 blend. A missing legacy `reception_policy` is
compared using its documented default; a different policy still rejects reuse.

All matches are staged and round-trip checked before publishing. Existing
`pass_success.parquet` files are refused. The utility retains original metadata
and a checksum report under `repairs/<repair_id>/`, updates current cache-use
statistics, and records the repair in `metadata.json`. Other component exports,
their state IDs, and the physical cache are preserved.

## Required pc-xPass blending

Enabling `--pc-xpass --use-physical-xpass` makes the configured blend required.
Startup validates the cache, metric, weighting configuration, and auxiliary
checkpoints. Each prepared inference unit is checked before its component models
run. A rejected cache, missing required state/player value, or nonfinite required
input aborts the entire run. Capturing `model_pass_success` or enabling comparison
skip modes does not permit fallback to incomplete predictions.

Self/carry options, offside options, opponents, and absent players retain their
normal handling. Valid weights that select the model prediction are allowed.
Checks preserve normal clipping and numerical blending behavior.

Required-blend attempts record `status: running`, then `completed` or `failed`.
Failure metadata includes the stage, error, and available cache/match/state/player
context. Failed runs exit nonzero and do not update the latest-successful pointer;
earlier completed outputs remain available in the failed run directory. EPV
attempts stage under `data/epv/inference_runs/` and publish canonical exports only
after inference succeeds. Existing explicit Hawkeye location cache preparation
and audit stages remain; ordinary inference does not generate replacement caches.

This policy applies to pc-xPass inference blending. Blending-disabled inference,
checkpoint-internal physical inputs without blending, other physical-xPass
sources, and reading saved component outputs retain their existing behavior.
