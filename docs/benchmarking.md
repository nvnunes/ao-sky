# Benchmarking

This source-tree research-log theme records benchmark evidence that informs `ao-sky` storage, traversal, prediction, and runtime decisions.

Detailed evidence lives under `docs/benchmarking/entries/`.

## Contents

- [Entries](#entries)
- [Summary](#summary)
- [Follow-Up](#follow-up)

## Entries

- `2026-04-01` [Storage, Cache and Format Baselines](benchmarking/entries/2026-04-01-e001-storage-cache-format-baselines/e001.md)
- `2026-04-16` Traversal I/O Optimization and Compression:
  - [Gaia Read-Path Optimization](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002a.md)
  - [Artifact Compression and Write Path](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002b.md)
- `2026-04-17` [Legacy-Compatible Traversal Baseline](benchmarking/entries/2026-04-17-e003-legacy-compatible-baseline-and-compatibility/e003.md)
- `2026-04-17` Winner-Map Asterism Selection Implementation:
  - [Winner-Map Asterism Selection Implementation](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)
  - [Additional Traversal Optimizations](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004b.md)
- `2026-04-17` [Memory Usage Audit with CPU Prediction](benchmarking/entries/2026-04-17-e005-cpu-prediction-memory-audit/e005.md)
- `2026-04-18` [Prediction Device and Batch Shape](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)
- `2026-04-18` [Dense Gaia Pixels and Performance Scaling](benchmarking/entries/2026-04-18-e007-dense-gaia-pixels-and-performance-scaling/e007.md)
- `2026-04-19` Static Scheduler Full-Sky Runtime and Modeling:
  - [Worker Trade Study](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008a.md)
  - [Memory-Aware Regional Pre-Planning](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008b.md)
  - [Runtime Simulation](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008c.md)
  - [First Full-Sky Memory-Aware Scheduler Runs](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008d.md)
- `2026-04-21` Dynamic Scheduler Runtime Validation:
  - [Dynamic Scheduler and Stochastic Simulation](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)
  - [v2 Dynamic Scheduler Runtime Evidence](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md)
- `2026-05-11` No-Winner Recovery Performance:
  - [v3 No-Winner Recovery Run Performance](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md)
  - [Update No-Winner Recovery Dynamic Scheduler Models](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010b.md)

## Summary

### Storage

#### Observations

- Storage tests showed SSD was much better for many-small-file reads, but more realistic Gaia neighbour and cluster access patterns narrowed the HDD-versus-SSD gap enough that an elaborate hot-cache design is not justified. Simple SSD staging looked like the better option ([e001](benchmarking/entries/2026-04-01-e001-storage-cache-format-baselines/e001.md)).
- Compressed HDF5 reduced a representative Gaia file by `38.5%` relative to FITS. Grouping four outer pixels compressed further, but would require reading more data when loading neighbours ([e001](benchmarking/entries/2026-04-01-e001-storage-cache-format-baselines/e001.md)).
- Reading Gaia from the SSD mirror was much faster than reading it from HDD: average Gaia read time fell from `0.180 s` to `0.021 s` without read prewarm ([e002a](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002a.md)).
- Worker-local Gaia table caching reduced measured Gaia load time by about `25%` on both HDD and SSD roots with only `2.4 MiB` peak cache, but it did not improve end-to-end wall time because most time was spent outside Gaia loading ([e002a](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002a.md)).
- The old `gzip=9` dense-inner compression cost dominated artifact-write time; Blosc Zstd kept artifacts small with much lower write and warm-read cost ([e002b](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002b.md)).
- The speculative Gaia read prewarmer had too little benefit to justify its added complexity; SSD artifact staging and write-behind also had small, mixed, or queueing-dependent gains ([e002a](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002a.md), [e002b](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002b.md)).

#### Decisions

- Keep canonical Gaia as compressed HDF5, one file per outer pixel, under the hour/declination/pixel directory layout ([e001](benchmarking/entries/2026-04-01-e001-storage-cache-format-baselines/e001.md)).
- Serve active-build Gaia reads from a local SSD mirror when the source store has HDD-class seek behavior, and keep worker-local Gaia table caching for neighbour-oriented traversal reuse ([e002a](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002a.md)).
- Use Blosc Zstd through `hdf5plugin` as the HDF5 compression default ([e002b](benchmarking/entries/2026-04-16-e002-traversal-io-optimization-and-compression/e002b.md)).

### Traversal

#### Observations

- Winner-map asterism selection improved the fixed 288-pixel CPU sample from `137.49 s` to `97.48 s` while doing more useful work: it retained about `5.1x` as many saved asterisms as the smaller overlap-pruned legacy-compatible catalog, while artifact size grew about `2.0x` ([e004a](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)).
- Vectorizing the local regularizer removed the dominant local-selection cost on the 12-pixel sample: local selection fell from `2.230 s/pixel` to `0.076 s/pixel` ([e004a](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)).
- Vectorized feature construction cut resolved plus averaged prediction work from about `1.41 s/pixel` to about `0.63 s/pixel` on the real-data sample ([e004a](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)).
- Compatibility checks kept the legacy-compatible code path as a reference and showed where winner-map asterism selection and NGS pre-selection in dense regions intentionally changed which asterisms are kept ([e003](benchmarking/entries/2026-04-17-e003-legacy-compatible-baseline-and-compatibility/e003.md)).
- A global adaptive NGS magnitude cap controlled candidate count but lost dense-field coverage; NGS pre-selection in dense regions kept dense-field coverage near complete while controlling per-region combination work ([e007](benchmarking/entries/2026-04-18-e007-dense-gaia-pixels-and-performance-scaling/e007.md)).

#### Conclusions

- Winner-map asterism selection is worth retaining because it is faster and keeps far more useful winning asterisms, despite higher memory and larger artifacts ([e004a](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)).
- The useful traversal optimizations were structural vectorization, not small scheduling or compatibility tweaks ([e004b](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004b.md)).
- NGS pre-selection in dense regions lets the build include dense areas, especially the Galactic plane, instead of excluding them because asterism finding cannot handle the raw candidate volume ([e007](benchmarking/entries/2026-04-18-e007-dense-gaia-pixels-and-performance-scaling/e007.md)).

#### Decisions

- Keep winner-map asterism selection, including the vectorized regularizer, and keep vectorized feature construction ([e004a](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004a.md)).
- Use NGS pre-selection in dense regions to keep dense areas in scope ([e007](benchmarking/entries/2026-04-18-e007-dense-gaia-pixels-and-performance-scaling/e007.md)).
- Accept leaving strict legacy exact equivalence where needed to support dense areas ([e003](benchmarking/entries/2026-04-17-e003-legacy-compatible-baseline-and-compatibility/e003.md), [e007](benchmarking/entries/2026-04-18-e007-dense-gaia-pixels-and-performance-scaling/e007.md)).

### Prediction

#### Observations

- Traversal arrays, feature matrices, and artifact write buffers were too small to explain the multi-GiB worker memory growth; memory mainly grew because prediction calls left memory reserved after they finished ([e005](benchmarking/entries/2026-04-17-e005-cpu-prediction-memory-audit/e005.md)).
- Prediction calls with many different row counts made retained memory worse, which led to testing padded prediction batches with a fixed set of allowed sizes ([e005](benchmarking/entries/2026-04-17-e005-cpu-prediction-memory-audit/e005.md)).
- GPU prediction cut the fixed-sample wall time by about `33%` compared with CPU prediction, but raised total RAM from about `4.6 GiB` to `11.6 GiB` across three workers ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Padding GPU prediction batches to the dense `1000..25000` row ladder was faster and used less memory than letting every call use its natural row count ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Explicit GPU cache clearing saved RAM but cost time; running without routine cache clearing was fastest, and the retained-policy smoke completed without memory-pressure failure ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Removing row-wise `zd`/`az` tie sorting cut feature-construction time. The existing models were not trained for that ordering, but the forced-tie stress test found only small prediction differences, and the real 12-pixel artifact comparison matched exactly ([e004b](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004b.md)).

#### Conclusions

- GPU prediction is worth using, with worker count and scheduling constrained by memory headroom ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Prediction memory should be managed by controlling prediction batch sizes and device choice, not by trimming traversal data structures ([e005](benchmarking/entries/2026-04-17-e005-cpu-prediction-memory-audit/e005.md)).
- The dense batch ladder is the retained speed/memory tradeoff for prediction ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).

#### Decisions

- Use GPU prediction where memory headroom allows it ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Pad prediction batches to the dense `1000..25000` row ladder ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Use explicit cache clearing only as a memory-pressure tool, not as the normal fast path ([e006](benchmarking/entries/2026-04-18-e006-prediction-device-and-batch-shape/e006.md)).
- Use magnitude-ordered guide-star features and do not reorder tied stars by inner-pixel `zd`/`az` location ([e004b](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004b.md)).

### Runtime

#### Observations

- GPU prediction made worker memory high enough that work scheduling became the main RAM mitigation: dense or memory-heavy pixels need to be placed deliberately instead of letting too many expensive workers run at once ([e008a](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008a.md), [e008b](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008b.md), [e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)).
- Memory-aware regional pre-planning produced a static scheduler that uses Galactic latitude as a density proxy and assigns fixed worker pools to low- and high-latitude regions. The fixed `9/6` split pushed memory too hard: workers spent most samples clearing memory, peaked at `25.48 GiB`, and only reached `0.80 pix/s` ([e008b](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008b.md), [e008c](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008c.md), [e008d](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008d.md)).
- Gentler fixed splits, including `8/5` and `8/2`, used less memory and spent less time clearing it, but we did not rerun a full comparable build to measure the end-to-end effect ([e008d](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008d.md)).
- To compare policies without rerunning every full build, we built a scheduler simulation from measured runtime and RAM behavior. It estimated throughput, memory pressure, and memory-clearing events for fixed and dynamic worker policies ([e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)).
- Added a dynamic scheduler that uses Gaia star count and the RAM model instead of Galactic latitude to decide which work can run together. The simulation favored this approach because it kept high-memory work moving without pinning workers to a fixed latitude split: dynamic `9` reached `1.12 pix/s` versus about `0.81 pix/s` for the fixed splits, with lower peak RAM than fixed `9/6` ([e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)).
- The first measured dynamic `9` run matched the policy expectation: it processed `41,184` outer pixels at `1.36 pix/s`, reached only `3` memory clearing events, never paused, and peaked at `22.18 GiB` ([e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md)).
- The dynamic `9` simulation matched measured throughput over the cropped run window, while the RAM simulation overestimated memory pressure ([e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md)).
- The completed `v3` dynamic `9` run processed all `49,152` outer pixels in `13.78 h` of active traversal time at `0.99 pix/s`, with only `2` memory clearing events, no pauses, and `22.13 GiB` peak RAM ([e010a](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md)).
- Active-time accounting for stopped-and-continued builds now skips stopped intervals and de-duplicates restarted `outer_pixel_done` events, so throughput reflects traversal work rather than supervision gaps ([e010a](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md)).
- Updated dynamic scheduler simulations now distinguish recovery-enabled and
  no-recovery operation. The no-recovery full-run simulation reaches
  `1.10 pix/s`, while the recovery-enabled full-run simulation reaches
  `0.995 pix/s`, leaving about a `10%` throughput gap; both RAM simulations
  remain conservative relative to measured peak RAM ([e010b](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010b.md)).

#### Conclusions

- Dynamic scheduling is superior to fixed latitude splits for the current GPU build ([e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md), [e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md), [e010a](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md)).
- Scheduler simulation is effective for testing scheduling algorithms before committing to full-build runs ([e008c](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008c.md), [e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)).
- Staying within the MacBook Pro M4 Max RAM ceiling limits throughput, so careful memory scheduling is part of how throughput is increased ([e008b](benchmarking/entries/2026-04-19-e008-static-scheduler-full-sky-runtime-and-modeling/e008b.md), [e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md), [e010a](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md), [e010b](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010b.md)).
- Use the `v3` dynamic model family as the baseline and vary only the first
  runtime segment and normal-worker RAM decay when no-winner recovery is
  disabled ([e010b](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010b.md)).

#### Decisions

- Use dynamic `9` under the current local memory ceiling, with dynamic `8` as the lower-pressure fallback ([e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md)).
- Keep measured total-RAM guards in control during the run to guard against memory overflow ([e009a](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009a.md), [e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md), [e010a](benchmarking/entries/2026-05-11-e010-no-winner-recovery-performance/e010a.md)).

## Follow-Up

### Predictions

1. Retrain and validate the prediction models to only sort NGS by magnitude ([e004b](benchmarking/entries/2026-04-17-e004-winner-map-algorithm-implementation/e004b.md)).


### Runtime

1. Study how sensitive the dynamic scheduler is to the runtime/RAM models when run on other computers ([e009b](benchmarking/entries/2026-04-21-e009-dynamic-scheduler-runtime-validation/e009b.md)).
