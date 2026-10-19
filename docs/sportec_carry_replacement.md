# Sportec carry definition v3

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
height are diagnostics. Provider and fallback boundaries receive identical checks.
One-second segmentation and canonical action-count return horizons are preserved.

## Create a corrected derivative

Run from C:\dm_model:

```powershell
.\.venv\Scripts\python.exe scripts\generate_relevant_features.py `
  --replace-carries `
  --extend-feature-run-id latest `
  --num-workers 4 `
  --worker-thread-limit 1
```

The latest completed carry-enabled source with complete canonical artifacts is
selected. A new run reuses non-carry artifacts using hard links and rebuilds the
complete carry variant. Canonical graph and label rows are reused exactly inside
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
