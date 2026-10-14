# xT refresh pilot - 2026-10-15

The isolated three-match pilot passed. Production feature artifacts and annotations
were unchanged. No production refresh, cache rebuild, training, or model fit was
performed by this implementation.

| Check | Result |
| --- | --- |
| Matches | DFL-MAT-J03WEL, DFL-MAT-J03YDU, DFL-MAT-J04034 |
| Compared label files / rows | 93 / 176,432 |
| Maximum xT target difference | 0.0 (exact agreement) |
| Non-xT tensor columns | Unchanged, including row identities and synthetic fields |
| Source labels/action tables | SHA-256 hashes unchanged |
| Source graphs | File signatures unchanged; all 27 graph files remain shared |
| Resolved-action non-xT fields | Unchanged across all 18 tables, including receiver choices and carry anchors |
| Production annotations | SHA-256 hashes unchanged |
| Missing annotation fixture | Rejected before replacement |
| Second refresh | 0 rewritten files |
| Graphs loaded / written during refresh | 0 / 0 |
| Full clone payload copies | 0 bytes |
| Shared graph payload | 1,530,513,080 bytes (1.425 GiB) |
| All linked artifacts | 138 files, 1,555,857,528 logical bytes |
| Refreshed artifacts | 111 files, 25,380,484 bytes (24.20 MiB) |
| Target refresh time | 3.35 seconds |
| Peak process RSS during refresh | 657.0 MiB |
| RSS increase over baseline | 54.9 MiB |
| Reference validation time | 148.01 seconds |

The refresh timing excludes imports, fixture setup, cloning and reference label
construction. RSS was sampled every 50 ms. These three matches do not establish
a guaranteed runtime or memory maximum for the full 918-match run.

The reference used normal `Match.construct_labels` with saved resolved actions
and original tracking data, using the same isolated annotation fixtures. All
three receiver modes and three recorded return types were included, across
canonical, carry, intent-training, augmented and success-intent labels. The
historically affected 2022/23 match's fixture was exported from persisted xT
coefficients without fitting a model.

The first pilot completed target comparison but encountered production annotation
changes during a concurrent xT regeneration. The successful rerun above used
frozen fixtures and passed all production-integrity checks.

Machine-readable report: `data/features/xt_refresh_pilots/pilot_20261015T005748/pilot_report.json`. Source hashes and signatures are also
saved in `source_baseline.json` in the same pilot directory. These local data
artifacts are not published feature runs.

Regression coverage: 406 tests and 60 subtests passed in the broader feature,
extension, return-type, prepared-cache and success-intent suite; subsequent
checks cover additional publication/replacement/cleanup failure cases.

See [the operational guide](targeted_xt_refresh.md) for production commands and
storage behavior.
