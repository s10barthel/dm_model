# Sportec carry definition v4

The shared policy governs Sportec training/evaluation carries and inference.
Legacy feature runs retain their original inference definition. SkillCorner
continues using its own possession events. Valid canonical actions survive
rejection of their preceding carry interval.

Eligible acquisitions are same-team receipts, InterceptedBall, BallClaimed and
consistent tackles. BallHeld and Block do not start carries. Holds suppress
companion acquisitions. Generic OtherBallAction needs corroborated control.
Same-frame release/tackle evidence cannot invent a post-release carry.
Missing-frame events create uncertainty barriers.

Defaults: acquisition distance <=3 m; separation >5 m sustained for >=1 s,
excluding the final 0.2 s. Missing tracking, invalid chronology, period crossings
and dead-ball frames reject the whole training spell. Terminal distance and ball
height are diagnostics. Legacy reception policies validate provider and fallback
boundaries identically; the retired policy removes Elastic receipts first.
One-second segmentation and canonical action-count return horizons are preserved.

## Create a corrected derivative

Run from C:\dm_model:

```powershell
.\.venv\Scripts\python.exe scripts\generate_relevant_features.py `
  --replace-carries `
  --retire-elastic `
  --refresh-trajectory `
  --extend-feature-run-id latest `
  --num-workers 4 `
  --worker-thread-limit 1
```

The latest completed source with complete canonical artifacts is selected. The
combined update refreshes endpoint-dependent artifacts and rebuilds carries.
Standalone carry replacement on an already refreshed run reuses canonical rows inside
combined artifacts. Event IDs normalize to nullable strings for numeric Sportec
and synthetic carry IDs. No receiver model is rerun. Use --artifact-copy-mode copy
if hard links are unavailable. No source artifacts are overwritten.

The derived run pins controls, segments and audits under carry_sidecars/.
Definition-aware inference and cache identities preserve legacy support. Metadata
records source provenance and validation parameters. Each match writes its
carry_replacement/ completion marker last. Only fully completed runs update the
latest pointer. Interrupted runs can be resumed with the explicit printed source
and derived run IDs:

```powershell
.\.venv\Scripts\python.exe scripts\generate_relevant_features.py `
  --replace-carries `
  --extend-feature-run-id <original-source-run-id> `
  --run-id <interrupted-derived-run-id> `
  --num-workers 4 `
  --worker-thread-limit 1
```

Resume with the same validation parameters. Configurable flags:
--carry-acquisition-distance, --carry-separation-distance,
--carry-separation-seconds, --carry-terminal-grace-seconds.
In-place replacement and unrelated combined generation options are rejected.

## Bounded audit

```powershell
.\.venv\Scripts\python.exe scripts\audit_carry_policy.py `
  --source-run-id latest `
  --output tmp\carry_policy_audit `
  --per-season 2 `
  --benchmark 1
```

Audits DFL-MAT-J03YDU plus the first two available matches by ID in each season,
deduplicating overlaps. At most three individual match rebuild benchmarks are
allowed. This never creates a full feature run. Reports include exclusions by
source, start/end, reason, diagnostics and a serial runtime estimate. Conservative
thresholds identify uncertain examples, not proven data errors.


## Retire Elastic reception timing

New full feature generations default to `provider_receptions_only_v1`. Legacy
runs retain their recorded behaviour. Launch synchronization is unaffected.
The synchronized CSV files are not modified. Provider receiver identities and
known outcomes remain available when reception timing is missing.

Refresh a completed carry-enabled source into a new derived run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_relevant_features.py `
  --extend-feature-run-id latest `
  --retire-elastic `
  --replace-carries `
  --num-workers 4 `
  --worker-thread-limit 1
```

`--retire-elastic` alone supports sources without carries. A carry-enabled source
requires `--replace-carries`. In-place retirement and unrelated combined refreshes
are rejected. Source settings are inherited. The canonical refresh runs before
carry assembly so corrected labels and post-action states are reused.

Unavailable receptions invalidate height labels, reception-time position metadata,
and failed-pass original-mode intended-receiver targets. Model/angle-only targets
and successful-pass receiver targets are retained. Dependent original-mode synthetic
augmentations become unavailable. Valid pass launch graphs and independent restart
states survive. Every tensor/graph slot stays aligned with its source action.

Pinned per-match records live under `reception_retirement/`; runtime loading and
physical-cache provenance respect the run policy. Old caches are not overwritten.
Completion markers are written last. Resume with the explicit source and output
IDs printed by the command, plus the same options and validation parameters.
Already retired runs are a no-op unless carry rebuilding is requested.

Bounded integration audit (four matches; never updates the latest pointer):

```powershell
.\.venv\Scripts\python.exe scripts\audit_reception_retirement.py `
  --source-run-id latest `
  --output tmp\reception_retirement_audit
```

Use a new audit directory for each attempt. If production hard linking is unavailable,
select `--artifact-copy-mode copy` explicitly; this copies reusable artifacts and
requires the corresponding disk space.

## Canonical pass endpoints

New carry generation and replacement consume the run-pinned `receive_frame_id` policy, after ELASTIC retirement and endpoint resolution. Only a confirmed official reception can supply a synthetic receipt; an official touch, duel or stoppage does not by itself establish possession. Explicit control events can still establish acquisition under the existing checks. Acquisition/separation thresholds are unchanged, and unavailable training passes remain in the control timeline.

When pass timing changes, combine `--replace-carries --retire-elastic --refresh-trajectory`. Fresh `--use-carries` uses the same preparation. Legacy v3 artifacts retain their validated inference engine; new v4 artifacts use the updated endpoint contract.
