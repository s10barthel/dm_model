# Turnover gains and stop conditions

Three separate feature operations are supported. Use the source run's split
selector (`--train-count` or `--train-split`) for extensions. Extensions create
derived runs by default; source artifacts remain unchanged.

## 1. Migrate base stop conditions

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <source> --refresh-stop-conditions `
  --num-workers 8 --worker-thread-limit 1
```

This recomputes outcome labels for all existing returns and target families,
using saved action identities and without rebuilding graphs. It preserves the
source xT target definition. It does **not** add set-piece variants.

`--num-workers <N|auto>` also controls this refresh stage (default: `1`). Each
worker computes and stages one match at a time; the parent validates all inputs
before replacing any output files. `--worker-thread-limit` limits native compute
threads in each worker. Progress messages show staged matches and the worker
count. These options also apply when refreshing existing stop-condition variants
for xT, goal distance, or newly generated labels. An already running refresh
keeps its original worker count; it must be restarted to use changed code.

Base stops include a detected goal, then stop; stop at period end and before a
subsequent goal-kick. No separate kick-off check is added. Set-piece stops also
stop before throw-ins, corners, free kicks and penalties. Boundaries are checked
in the full event stream, even when the boundary event is not an eligible xT
action. A restart used as the labeled action can look forward into its own play.

Each family retains its existing counting and aggregation rules. For `in_N`, a
boundary before the selected state gives zero raw return, except when the
existing early-shot rule selects an earlier permitted shot. Fixed diagnostic
labels retain their existing definitions.

Migration is required before adding returns or starting new outcome training
on legacy runs. Existing model records retain legacy semantics. Legacy
`_spstop` artifacts remain readable; migration normalizes their return names and
recomputes base labels. New commands use `--stop-condition`, not `_spstop`.

## 2. Refresh xT gain targets

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <migrated> --refresh-target-family xt `
  --xt-version <version> --xt-target-representation gain --xt-clip-negative-gains
```

For `next_N`, `next_N_skip1`, `next_N+`, and `next_N+_skip1`, definition version 2
uses maximum teammate xT minus the labeled action's starting surface value for
scoring, and maximum opponent xT minus the opponent acquisition-start surface
value for conceding. Clipping applies to both gains. No opponent candidate
means zero conceding. Absolute targets and non-`next` gain definitions retain
their previous transformations.

The first opponent pass/cross/shot in the original N-action window identifies
the transition. Search backward, bounded by the last eligible original-team
action, the labeled action's saved return anchor, and applicable boundaries.
Use the earliest opponent interception, recovery, tackle, goalkeeper
claim/pick-up/save/sweeper, block, punch, or clearance with usable coordinates.
Bad touches are excluded. There are no success/control/possession-change checks.
If no matching event is found, use the first opponent pass/cross/shot location.
Coordinates use the opponent's attacking perspective. Keep this first baseline
through subsequent turnovers. A possession without an eligible opponent action
does not trigger the lookup.

Sportec canonical files omit defensive actions. Their synchronized control
events supply extra acquisition locations in an xT-only timeline; these never
change canonical indices or another family's action counting. Tackles use the
recorded winner, claims/recoveries use their actor, and clearances use their
actor. No possession/control checks are added.

Initial preparation reads only `ball_x`/`ball_y` tracking columns; generation
reuses tracking it already loaded. Locations are saved under
`acquisition_locations/<match>.parquet`. Later refreshes read this frozen cache
without loading tracking. When controls or usable ball coordinates are absent,
the canonical-event fallback is recorded in the refresh report.

The refresh changes only xT targets and updates every stop variant already
stored; it does not create unrequested variants. Existing runs retain their
recorded gain definition until explicitly refreshed.

## 3. Add returns and optional set-piece variants

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <gain-refreshed> --return-type next_3+

.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <run> --return-type next_3 --return-type next_3+ `
  --stop-condition set-pieces
```

`--return-type` is an alias for the existing `--return_type`. `next_N+` is xT
only. It extends the shared scoring/conceding window once, to include N eligible
passes/crosses/shots strictly after acquisition. An eligible acquisition action
remains a maximum candidate but does not consume the restarted budget.
Noneligible acquisition events contribute no maximum candidate. Earlier
candidates remain in the window. Later turnovers never restart the budget.

For N=3 and acquisition at eligible position 2, the endpoint is position 5.
When acquisition is an interception immediately before the first opponent
pass, that pass consumes the first restarted slot. `_skip1` excludes its normal
candidate (unless a shot) without hiding acquisition or replenishing the budget.

Fresh generation defaults to base only. `--stop-condition set-pieces` generates
base plus set-pieces for the listed returns. On an extension it requires at
least one explicit return, including when adding set-piece support to an
already existing return. Availability is recorded separately for each return.

Adding a return type to a carry-enabled run also generates its carry labels for
the saved train/test splits and all available receiver modes. Saved carry actions
and graphs are reused automatically; no `--use-carries` flag is needed for this
extension. Before publication, the extension checks label coverage and action
order for canonical, carry, and intent-training labels, including requested
set-piece variants. An incomplete extension cannot update the latest-run pointer.

## Extended possession

`--return-type extended_possession` uses the remainder of the initial team's
eligible-action sequence and the complete next opponent sequence. Only passes,
crosses, and shots define sequences. Stop before the first eligible action by
the initial team after that opponent sequence. A failed initial pass still
establishes the first possession; interceptions, tackles, carries, and other
ineligible events never change sequences. Missing target annotations do not
hide team changes. If there is no second team change, use the remaining events
up to the stop boundary.

For `A A A A A A B A`, action 0's window ends at action 6, inclusive. For
`A B B B B A`, it ends at action 4, inclusive. Starting midway through an A
sequence uses only its remainder and the next B sequence.

`extended_possession_N` additionally limits the window to at most N future
eligible actions strictly after the saved return anchor. N must be a positive
integer. The initial action establishes the first possession and consumes no
slot. Goal/xG returns retain its existing contribution in addition to the N
future slots. Passes, crosses, and shots with missing target values still
consume slots; other event types, including acquisition events, do not.

The cap can end within either possession, including before any opponent action.
The earliest of the cap, the possession endpoint, and the stop boundary wins.
In `A A A A A A B A`, action 0 with `extended_possession_3` ends at action 3,
while `extended_possession_6` and `extended_possession_20` both end at action 6.
In `A B B B B A`, `extended_possession_3` ends at action 3. Plain
`extended_possession` has no numerical action cap and retains its existing
behavior. Capped and unlimited returns can coexist in the same feature run.

The target aggregations match `next_N`: goal and combined xG returns include
the initial eligible action, while xT, EPV, and goal-distance maxima use only
eligible actions strictly after the saved return anchor. All supported target
families, including progression, can use this return. Version-2 xT gains use
the same acquisition-start conceding baseline and clipping as `next_N`.
Carries establish the initial team using their own actor and look forward from
their saved return anchor, even when that anchor belongs to the other team.

Base stops (including period ends, goals, and goal-kicks) always apply.
`--stop-condition set-pieces` also generates the shorter set-piece variant.
There is no `_skip1` or `_spstop` name modifier for this return.

```powershell
.\.venv\Scripts\python.exe scripts/generate_relevant_features.py `
  --extend-feature-run-id <run> --return-type extended_possession `
  --return-type extended_possession_10
```

## Training and artifacts

```text
--target-family xt --return_type next_3+ --stop-condition set-pieces
```

Outcome training defaults to `--stop-condition base`. Missing variants fail
explicitly. Model records, bundles, evaluation and resume checks preserve the
stop definition; incompatible outcome definitions cannot be mixed.

Labels live under `labels/base/` and, only when requested, `labels/set-pieces/`.
Graphs and action identities are shared. Legacy label files remain as source
artifacts; metadata selects the new layout after successful migration. Saved
action target columns describe the primary base return. Cache signatures see
the selected label files and updated metadata.

Refreshes stage replacements and check input/artifact stability before
publishing. Missing required annotations fail explicitly. EPV sidecars were
optional in historical generation: absent EPV annotations are reported, and
the existing zero-return convention is retained only when saved EPV targets
are zero; missing annotations with nonzero saved EPV targets fail.

An isolated pilot is available in `scripts/diagnostics/pilot_turnover_stops.py`.
It accepts `--source`, `--output`, optional `--match-id`, and `--copy` when
hard-linking is unavailable. It never registers runs or updates latest pointers.

`scripts/diagnostics/pilot_carry_return_extension.py --source <completed-run>
--output <new-private-directory>` exercises the actual extension planner and
label generator on one saved train match and one saved test match. It adds
`next_5` and `next_5+`, checks both stop modes, and verifies that source files and
retained graphs remain unchanged. The source must not already contain these
two returns.
