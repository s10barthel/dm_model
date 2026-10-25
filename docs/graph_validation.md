# Graph validation

Graph generation, refresh, training and evaluation default to `--graph-validation on`.
Use `--graph-validation off` to skip the added raw-graph scans, preservation fingerprints
and saved-output readback when their CPU/I/O cost is undesirable.

```powershell
python scripts/generate_relevant_features.py --return_type next_10 --graph-validation on
python scripts/train_relevant_models.py --feature-run-id YOUR_FEATURE_RUN --only-success-intent --graph-validation off
python scripts/evaluate_relevant_models.py --success-intent-model-id success_intent/YOUR_MODEL_RUN --graph-validation on
```

The same option is accepted by `datatools/graph_feature.py`, `train.py`, dedicated refresh
modes, and training resume/refit commands. Each invocation defaults to on, even when the
source run or checkpoint recorded off. The setting is operational and does not change
model inputs or labels. Library loading, refresh and graph-writing APIs use the keyword
`graph_validation=True` by default; stored-graph inference has the same keyword.

Enabled validation requires raw graphs to contain each directed node pair exactly once,
including self-loops. It checks index dtype/shape/bounds, node dimensions, edge-attribute
counts/schema, optional node IDs, and graph/label alignment. Null graphs and empty
artifacts remain supported. Valid edge permutations and supported legacy schemas are
accepted. Failures identify the artifact, match, source row and action when available.
Prepared/sparsified graphs retain their sparse contract.

Graph writers validate temporary files after closing and rereading them, before atomic
publication. Refreshes also check immutable fingerprints of retained topology and node
order; trajectory-only changes must preserve edge attributes. Edge-feature extensions
may add columns while preserving existing named columns. Disabling validation retains
existing checks and atomic publication; it cannot make an invalid Delaunay edge lookup
succeed. Labels and tabular outputs use their existing writers without graph readback.

Checked and unchecked prepared caches have separate identities. Raw graphs are checked
at preparation, not every epoch. The validator code participates in cache fingerprints.
The packed format is unchanged. Operation metadata records the mode and validation,
fingerprinting and readback timings; file/graph check counts count validation passes,
including input, output and readback checks, while verified-output counts count files.
Generation workers write summaries under the run's `graph_validation` directory;
refresh summaries include their validation statistics. Disabled counters stay zero.

Measurements on this machine estimated approximately 24 seconds per structural pass
across 797,000 success-intent graphs. A complete refresh with preservation checks and
cached readback was estimated at 6-7 additional minutes for that 26.5 GiB family.
Full rebuilds of all graph families (approximately 322 GiB in the investigated run)
cost substantially more. Disk contention and fingerprint implementation affect these
estimates. Only graphs actually read or rewritten incur the applicable checks;
untouched outputs and label-only operations do not gain graph reads.

The latest run `feature_20261021T000152_142839_d201b27b` was already repaired separately.
This feature does not repair historical runs or alter feature files automatically.

The full production-validator acceptance audits took 475-496 seconds including loading
all 918 files, with 92-100 seconds spent in structural validation. This is a more conservative
measurement than the earlier 24-second connectivity-only benchmark; actual overhead
depends on the schema and metadata checks as well as disk/cache conditions.
