# xT versioning implementation plan

## Goal and contract

Each xT build creates a distinct directory at `data/xT/<version>/`. A full
version contains the fit artifacts, `fit_metadata.json`, `xT.csv`, and
`matches/<match_id>.csv` sidecars produced by that build. Feature generation
and xT target refresh select an xT version by ID and record it in the feature
run metadata. A completed version is never overwritten by another build.

The repository uses `data/xT` in code and documentation. This is the same path
as `data/xt` on the current Windows workspace, but the implementation should
keep the existing spelling so it also works on case-sensitive filesystems.

## Implementation steps

1. **Add version path and validation helpers.** In `project_config.py`, keep
   `XT_DIR` as the container and add a version ID validator plus helpers for
   `data/xT/<version>` and its `matches` directory. Reject path separators,
   reserved names, and existing destination IDs. Add a reader for completed
   xT version metadata that checks the selected split, required fit files, and
   sidecars for the matches a feature run will process. Do not let a missing
   sidecar silently fall back to xT columns in synced events.

2. **Publish new versions from `scripts/generate_xt.py`.** Generate a unique
   ID (with an optional caller-supplied ID for scripted pipelines), create all
   outputs under a staging directory on the same filesystem, and publish to
   `data/xT/<version>` only after output validation succeeds. Write the ID,
   build options, split manifest, output paths, exported match IDs, and a
   completion/partial status in `fit_metadata.json`. Update a `latest.json`
   pointer only after a complete version is published. A requested ID that
   already exists must fail; the xT generator has no `--overwrite` flag.
   Make `--reuse-source-grid` read from an explicitly selected prior
   version, rather than from the new output
   directory. Keep `--fit-only`, `--match-id`, and `--limit` outputs separate
   from full versions: they may be saved for inspection, but cannot be
   selected for a feature run. A full version must cover the required match
   set before it can become `latest`.

3. **Pin xT for full feature generation.** Add `--xt-version` to
   `scripts/generate_relevant_features.py` and pass the resolved, validated
   version to every `datatools/graph_feature.py` worker command. Give
   `datatools.match.Match` an explicit xT sidecar directory and pass it to
   `merge_xt_annotations`; this is essential because full generation currently
   uses `Match`'s global `XT_MATCH_DIR` default. Store `xt_version` and the
   selected fit metadata fingerprint in the feature run's `metadata.json`.
   Validate required sidecars before starting workers and ensure all workers
   use the same resolved path. Audit other `Match` callers that operate under
   a feature run and have them use its recorded version.

4. **Pin xT for refresh and extension.** Resolve `--xt-version` before
   `--refresh-target-family xt`, pass its `matches` directory to
   `datatools.xt_refresh.refresh_xt`, and record the selected version in the
   refresh report and completed feature run metadata. A derived refresh may
   change `xt_version`; other extensions inherit the base run's version and
   reject an attempted change without an xT refresh. For an in-place xT
   refresh, record the old and new IDs in `extension_history` and update the
   current ID only after a successful refresh. Preserve the current failure
   handling for partial file replacement. Remove `--xt-match-dir` from the
   feature-run CLI because an arbitrary directory has no stable version ID.

5. **Update the pipeline wrapper and other xT consumers.** In
   `scripts/main.py`, allocate the xT version ID before building subprocess
   commands and pass it to both `generate_xt.py` and feature generation. With
   `--skip-xt`, validate the explicitly selected version instead of the
   unversioned `XT_DIR`. Update pc-xPass's `--top-xt` surface lookup in
   `physical_pass_model.py` to use the selected feature run's xT version (or
   an explicit version if no feature run is available), and keep its existing
   surface hash in cache metadata. Review direct `Match` callers and
   diagnostics for any remaining implicit reads of `data/xT`.

6. **Migrate, document, and verify.** Move existing unversioned files
   into `data/xT/legacy/` with a one-time migration command; mark older feature
   runs with no recorded `xt_version` as legacy/unpinned rather than claiming
   an exact version. Update README commands and the targeted xT refresh guide.
   Test unique version creation, collision/failure cleanup, source-grid reuse,
   partial export rejection, version selection through every full-generation
   worker, refresh metadata and labels (derived and in place), inheritance
   across other extensions, legacy reads, and pipeline/pc-xPass propagation.
   Run a small end-to-end feature build and targeted refresh using two xT
   versions; confirm each run's labels match its selected sidecars while the
   older version remains unchanged.

## Agreed decisions

| Decision | Choice |
| --- | --- |
| Selection default for new feature runs and refreshes | Require `--xt-version`; explicit `latest` resolves once to a concrete ID. |
| Treatment of incomplete xT builds | Save fit-only and subset builds as `partial`; do not allow feature runs to select them. |
| Existing unversioned xT artifacts | Move into `data/xT/legacy/`. |
| pc-xPass `--top-xt` surface | Use the feature run's xT version, or an explicit version without a feature run. |
| Existing `--overwrite` flag | Remove it from xT generation. |
