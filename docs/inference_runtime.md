# Runtime inference efficiency

Runtime inference can omit expensive extended node calculations when every
participating checkpoint discards the corresponding inputs. Hawkeye, benchmark,
SkillCorner, and Sportec possession builders retain the same frame selection,
tracking derivatives, tensor widths, labels, candidate identities, and edge data.
Unused slots in these temporary graphs are zero-filled. Requirements include
separately selected PC-xPass auxiliaries and the union of comparison variants;
retaining a feature for any player role requires its computation.

The policy is applied by the component runners, direct Hawkeye inference
visualizer, Sportec action visualizer, and inference comparisons. Builders still
compute all features by default. Training, persisted graph generation, physical
cache generation, and the location runner's shared cache-generation graphs keep
their original computation. Checkpoints requiring raw-graph physical inputs or
lane-survival hashes, and unknown/custom feature layouts, also keep full features.
No cache format, fingerprint policy, worker setting, or CLI option changes.

The shared GNN inference helper avoids tracking lookups when node candidate IDs
already identify the outputs. Ordinary scalar and signed outcome heads decode
their independent rows in bulk on CUDA and transfer results once per batch.
CPU activations retain their original per-graph shapes to preserve rounding.
Distributional, decomposition, selection, pass-success, and explicit out-candidate
decoders retain their existing specialized paths. Their unnecessary tracking
lookups are still avoided where candidate IDs suffice. Outcome decomposition
attributes, distribution exports, offside rules, carry behavior, and raw xPass v0
precision remain governed by their existing inference contracts.

Early frame selection is deliberately unchanged: graph construction still
precedes the existing time-norm filter, and frame statistics retain their scope.

## Reproducing comparisons

Capture the current working sources before editing, preserving repository-relative
paths. This is preferable to an older Git revision when working changes are part
of the baseline. Then run:

```powershell
.venv\Scripts\python.exe scripts/diagnostics/benchmark_inference_runtime.py `
  --baseline-sources tmp/<captured_sources> `
  --output-dir tmp/<new_comparison_directory> `
  --device cuda:0 --repeats 3
```

The diagnostic uses the newest saved scoring/conceding checkpoints unless their
`--outcome-scoring-model-id` and `--outcome-conceding-model-id` are specified.
It reads the first 20,000 tracking CSV rows by default and the complete ball CSV;
`--sample-rows 0` reads the complete tracking source. An optional `--situation-id`
selects the situation, otherwise the first cleaned/sorted situation is used.
Source files and production cache/run pointers are never written. Only the new
comparison directory receives measurements.

Both paths are warmed, then measured in alternating order. Timings include input
loading/cleaning, complete tracking and graph preparation, and both model
predictions. Model loading, warm-up, verification, exports, and interpreter startup
are excluded. The diagnostic checks exact prepared node/edge tensors, candidate
ordering and NaN positions. It replays captured real-model logits through both
inference implementations and requires exactly equal prediction tables, isolating
decoding from nondeterministic CUDA model reductions. It separately records output
differences across fresh forwards and the baseline's observed repeatability; a
maximum from three repeats is not a bound on the next forward's variation. Tests
also compare decoding exactly on identical CPU/CUDA logits.

## Local sample, 12 November 2026

The captured working baseline is under
`tmp/inference_optimization_baseline_oejd0yaz`. A 375-frame real Hawkeye situation
and two saved outcome checkpoints on the RTX 2080 SUPER produced these medians
over three measured runs per implementation, after warm-up:

| Stage | Baseline | Optimized |
|---|---:|---:|
| Sample input loading/cleaning | 0.19 s | 0.19 s |
| Tracking and graph preparation | 4.69 s | 3.23 s |
| Both outcome predictions | 4.08 s | 3.17 s |
| Sample pipeline | 8.97 s | 6.59 s |

The sample pipeline was **26.5% faster**. Prepared model inputs and predictions
decoded from replayed real logits were identical. Across fresh CUDA forwards, the
maximum difference was `1.7881393432617188e-7`; baseline repeat variation was
`1.4901161193847656e-7`. These observations illustrate why finite-sample forward
variation is recorded separately from exact decoding checks. The timings include
sampled CSV loading and are not an estimate for processing the entire dataset.
Raw settings, checkpoint IDs, and measurements are in
`tmp/inference_runtime_20261112_verified/results.json`.

A second comparison read and cleaned the complete tracking CSV on every run,
then processed one 376-frame situation with both models. Over two measured runs
per implementation, median pipeline time fell from **14.77 s to 11.92 s (19.3%)**.
Input loading/cleaning alone took about 6.1 s and is unchanged by these
optimizations. Tracking/graph preparation fell from 4.70 s to 2.84 s, and both
predictions from 3.91 s to 2.97 s. Exact input and captured-logit replay checks
passed; both paths' maximum fresh-forward difference was
`2.086162567138672e-7`. Raw results are in
`tmp/inference_runtime_20261112_full_verified/results.json`.

Validation includes 36 new cases covering retained roles, mixed model/auxiliary
requirements, raw-hash fallbacks, three dataset builders, Sportec provenance,
exact CPU/CUDA decoding, and reused visualization state. Adjacent optional-model,
cache, frame-selection, outcome, mask, and model-selection suites passed. Broader
testing also encountered five existing visualization fixture failures (missing
tracking and historical v5 dependency expectations) and a training-parser fixture
that no longer intercepts parser construction. Those failures reproduced with the
captured pre-change source and were left unchanged.
