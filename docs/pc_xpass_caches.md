# pc-xPass deceleration and cache versions

pc-xPass models each speed-grid value as the initial release speed. The ball slows
at `--ball-dec` metres per second squared (default `0.45`). Use `--ball-dec 0` for
constant-speed passes. The ball never reverses: sampled targets beyond its stopping
distance are excluded from scoring and ranking. Accessible Space is unchanged.

## Normal caches

```powershell
# Create a fresh version with an automatically generated ID.
python scripts/generate_physical_xpass.py --pc-xpass --ball-dec 0.45

# Create a named version, or resume it with its recorded settings.
python scripts/generate_physical_xpass.py --pc-xpass --pc-xpass-id dec045

# Explicitly select the version for inference.
python scripts/run_hawkeye.py --pc-xpass --pc-xpass-id dec045
```

Each version lives in `data/pc_xpass/<pc_xpass_id>/`, with root metadata and separate
`sportec`, `skillcorner`, `benchmark`, and `hawkeye` subdirectories. Dataset files
retain their existing layout. The command prints its generated ID.

Generation without an ID always creates a new version. Resuming an ID inherits
omitted generation settings and rejects explicitly conflicting settings. The
metadata records settings, physics/schema versions, invocation history, coverage,
and completion status. Compatible missing rows can be added to a version.

Normal consumers default to the ID in `data/pc_xpass/latest.json`. Successful
generation updates this global pointer; failed, incomplete, and dry runs do not.
A deliberately limited successful run can become latest. If it lacks your dataset,
select an appropriate ID explicitly; consumers do not search older versions.

`--pc-xpass-id` is also available for training with lane-survival features,
evaluation, EPV generation, and visualization. It selects a cache without enabling
additional model features. Outputs record the resolved ID for reproducibility.

## Hawkeye location and freeze caches

```powershell
# Always create a fresh location cache, independently of normal latest.
python scripts/run_hawkeye_loc.py --ball-dec 0.45

# Reuse an existing location cache; compute missing rows using its settings.
python scripts/run_hawkeye_loc.py --pc-xpass-id <location-cache-id>
```

These caches live in `data/pc_xpass/hawkeye_loc/<pc_xpass_id>/`. An explicit ID must
already exist. Cached generation settings override conflicting CLI flags with a
warning, including deceleration and metric-generation settings. Input selections,
geometry, model selection, and execution controls still come from the invocation.
Location runs never advance the normal latest pointer.

Location visualization inherits the cache ID recorded in the selected component
run. An explicit `--pc-xpass-id` or cache-directory override takes precedence.

## Existing caches and directory overrides

Existing unversioned caches remain in place and are not registered as latest.
Select them explicitly, for example:

```powershell
python scripts/run_hawkeye.py --pc-xpass --physical-cache-dir data/pc_xpass/hawkeye
python scripts/run_hawkeye_loc.py --pc-xpass-cache-dir data/pc_xpass/legacy_location
```

Do not combine an ID with a cache-directory override. Legacy metadata without
deceleration is interpreted as constant speed; location generation through that
explicit directory retains zero deceleration. No automatic migration occurs.

## Consistent resumes and pass-height replacement

Resume with `--pc-xpass --pc-xpass-id <id>`. Omitted calculation settings and
`--export-lane-control` inherit recorded values; explicit conflicts require a new
version. Physics and ranking dependencies are validated. Sportec pins the resolved
feature run and whole-spell provenance; Hawkeye pins `--freeze-ballreceipt`
(initial default: true). Contracts and reconstruction sources are saved before
computation. Paths may move if source identities and reconstructed hashes match.

Data selection remains flexible: season, match/situation IDs, dataset switches,
limits, `--scope`, `--frames`, and Hawkeye times are not frozen or automatically
restored. Workers, device, and batch/window sizes may change. Repeat the original
selection to finish the same requested coverage, or change it to extend the cache.
Partition/source coverage accumulates, and sampling reports are saved per invocation.

An older interrupted Sportec cache lacking resolved input provenance requires
explicit `--sportec-feature-run-id <id>`. Saved states are reconstructed and checked
before new partitions are added; unknown or incompatible sources cause an error.
Ctrl+C marks the invocation incomplete where possible. Saved rows remain reusable
if abrupt shutdown leaves status marked running.

Pass-height predictions are mutable enrichment. Omitting `--pass-height-model-id`
inherits the recorded model, including a pending refresh target. Selecting another
compatible height model refreshes all saved states across every populated dataset,
including datasets excluded by current generation switches. Physical metrics and
lane-control artifacts are preserved. Additional selected physical states are
generated after the full refresh completes.

Refresh requires the original state sources. Target artifact and pending status
are saved before predictions change. Model/input fingerprints are stored with
predictions in the same atomic Parquet write. Resuming skips verified rows; another
model switch is rejected until the pending refresh completes. Legacy predictions
without trustworthy provenance are refreshed, rather than certified from a single
dataset-level model ID.

While refresh is pending, height-consuming reads fail, physical-only reads remain
available, and the version cannot advance `latest`. Missing sources, changed state
hashes, incompatible height definitions, or incomplete predictions retain pending
status and identify unresolved work. Completed model metadata is committed only
after full verification. Dry runs leave cache files and metadata unchanged.
