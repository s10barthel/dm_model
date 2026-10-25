# Feature selections for training and inference

Both `scripts/train_relevant_models.py` and `train.py` accept explicit selectors.
Omitting them preserves each entry point's current defaults. Direct training
retains its historical feature profile, which differs from the wrapper. Saved
settings determine evaluation, inference, resume, refit, and auxiliary/IPW inputs.

## Node selections

| Selection | Retained values |
| --- | --- |
| `all` | Every node |
| `attacking` | Players on the possessing team, including the possessor |
| `defending` | Players on the opposing team |
| `no-poss` | Every node except the possessor |
| `none` | No nodes |

Both goalkeepers are players. Team selections exclude synthetic goal nodes;
their selected columns become zero. `all` and `no-poss` can retain goal inputs,
subject to the separate goal-geometry setting. `true` and `false` are accepted
as aliases for `all` and `none`.

| Training flag | Node inputs |
| --- | --- |
| `--pos-node-features SELECTION` | `x`, `y` |
| `--vel-node-features SELECTION` | `vx`, `vy`, `speed`, `accel` |
| `--vx SELECTION`, `--vy SELECTION` | Individual velocity components |
| `--speed SELECTION`, `--accel SELECTION` | Scalar speed or acceleration |
| `--goal-features SELECTION` | Goal distance and normalized goal direction |
| `--poss-geometry SELECTION` | Possessor distance and normalized possessor direction |
| `--poss-velocity-angles SELECTION` | Cosine/sine of velocity angle relative to the possessor |
| `--offside SELECTION` | Offside indicator, when present |
| `--extended-node-features SELECTION` | Six extended handcrafted inputs |
| `--pass-lane-features SELECTION` | Nearest-opponent-to-pass and potential-interceptor inputs |
| `--lane-survival SELECTION` | Cached lane-survival input |

Groups provide defaults for their members. Narrower groups and individual
features override broader groups, independent of argument order. Explicit
selectors override the legacy profile for their selected columns. Goal-node
possessor geometry still requires `--goal-poss-geometry true`.

Append these options to a normal training command:

```powershell
# Only defenders retain vx and vy; speed and acceleration are zero everywhere.
--vel-node-features none --vx defending --vy defending

# Defenders retain all four motion inputs.
--vel-node-features defending

# Everyone except the possessor retains motion; acceleration is zero everywhere.
--vel-node-features no-poss --accel none
```

These masks affect the named learned inputs. Velocity-derived edge features and
possessor-relative velocity angles remain independently controlled. Components
use the existing graph coordinate system.

## Boolean and structural controls

| Flag | Meaning |
| --- | --- |
| `--possessor-id true/false` | Retain/zero only `is_possessor` |
| `--keeper-id true/false` | Retain/zero only `is_keeper` |
| `--ball-z true/false` | Retain/zero ball height |
| `--goal-poss-geometry true/false` | Allow/suppress possessor geometry on goal nodes |
| `--goal-nodes true/false` | Retain/remove goal nodes and incident edges |
| `--action-intent-include-goals true/false` | Include/exclude action-intent goal candidates (wrapper) |
| `--include-goals true/false` | Same candidate setting in direct training |

`possessor-id` is independent of possessor-relative geometry and velocity angles.
The historical `possessor-aware` alias keeps its broader effect on columns 13-18.
Original identities remain available to graph processing and subset masks.

Lane-survival selection and computation are separate:

```powershell
--lane-survival defending --lane-survival-mode top_pass_10
```

Computation defaults to `max` when enabled. Existing `--lane-survival`,
`--lane-survival max`, `--lane-survival top_10`, and `--no-lane-survival` remain
valid. Requested metrics must exist in the selected physical cache.

## Edge selections

These flags accept `all`, `no-poss`, and `none`:

- `--velocity-angle-edge-features`
- `--relative-speed-edge-features`
- `--position-direction-edge-features`
- `--relative-velocity-edge-features`
- `--separation-rate-edge-features`

`no-poss` zeros inputs on both incoming and outgoing possessor edges. Team
selections are not supported for edges. Existing stored-schema requirements
still apply, including relative-speed's dependency on velocity-angle features.
`--goal-edge-movement true/false` controls movement inputs on goal edges.

## Inference masks

Evaluation and inference/visualization scripts accept
`[TASK:|all:]FEATURE[=SELECTION]`. A bare feature means `=none`.
Omitting inference masks entirely preserves normal checkpoint inference.

```powershell
--inference-mask all:vx all:vy
--inference-mask all:vx=defending all:vy=defending

--inference-mask-1 all:vel-node-features
--inference-mask-2 all:vel-node-features=defending

--inference-mask pass_success:vel-node-features=no-poss
```

Explicit values replace bare shorthand for the same feature/group, independent
of argument order. Narrower groups and individual features override broader
groups. Conflicting explicit values for the same feature/group are rejected;
identical assignments are deduplicated. Numbered configurations remain separate.
Unscoped names require exactly one participating model task, including auxiliaries.

Inference can only further restrict inputs allowed by training. For example,
`vx=defending` on an attacking-only checkpoint zeros all `vx`; it cannot restore
defenders' values. `vx=all` retains normal checkpoint inputs. Already excluded
inputs are reported as having no additional effect. Missing or ambiguous schema
features produce a compatibility error.

Use `--list-inference-masks` to inspect names, selections, and expansions.
Schema-qualified names such as `node.vx` and `edge.relative_speed` remain supported.

## Compatibility and provenance

Existing model IDs and checkpoints remain usable without retraining or rewriting.
The resolver reconstructs legacy defaults and coupled masks. Historical aliases
keep their meanings, including these special inference shorthands:

| Legacy inference name | Equivalent selection |
| --- | --- |
| `poss-vel-aware` | `vel-node-features=no-poss` |
| `mask-possessor-v-edge-features`, `v-edge-features-no-poss` | `velocity-angle-edge-features=no-poss` |
| `mask-possessor-relative-speed-edge-features`, `relative-speed-edge-features-no-poss` | `relative-speed-edge-features=no-poss` |
| `possessor-aware` | Remove all six historical possessor-awareness inputs |

Negative training flags remain compatibility aliases. Bare legacy positive
training flags retain their historical meaning; bare inference names remove
their named inputs. `xy-only` remains a legacy preset with its historical effects.

New checkpoints record explicit selections and a versioned, resolved per-feature
permission signature. Metadata summaries include selections and resolved masks.
Inference metadata records the original request, canonical selections, resolved
operations, and configuration identity. Equivalent shorthand and explicit forms
share an identity. Saved outputs with legacy metadata remain readable.

Prepared dataset cache identities include selections and preprocessing code.
Existing source graphs can be reused; prepared caches may rebuild after this
update. Node masks preserve width and source graphs. Coordinates remain available
for spatial processing, physical inputs, and cache identities.
