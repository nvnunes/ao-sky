# Benchmarking

This source-tree research-log theme records benchmark evidence that informs `ao-sky` storage, traversal, prediction, and runtime decisions.

Timeline and entry routing live in `docs/benchmarking/index.md`. Detailed evidence lives under `docs/benchmarking/entries/`.

## Summary

### Storage

#### Observations

- HDD is too slow for seek-heavy Gaia access: historical many-small-file reads took `27.38 s` on HDD versus about `1.30 s` on local SSD, and cold Gaia access-pattern reads remained much worse on HDD ([e001](benchmarking/entries/2026-04-01-storage-e001-storage-cache-format-baselines/index.md)).
- Compressed HDF5 reduced a representative Gaia file by `38.5%` relative to FITS, so each canonical Gaia read has less data to pull from disk ([e001](benchmarking/entries/2026-04-01-storage-e001-storage-cache-format-baselines/index.md)).
- Reading Gaia from SSD was much faster than reading it from HDD: the average raw-load metric fell from `0.180 s` to `0.021 s` when the Gaia root moved to a local SSD mirror ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- Worker-local prepared Gaia table caching made Gaia load and preparation about `25%` faster on both HDD and SSD roots with only `2.4 MiB` peak cache, but the whole moderate-density benchmark did not get much faster because most time was spent outside Gaia loading ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- The old `gzip=9` dense-inner compression cost dominated artifact-write time; Blosc Zstd kept artifacts small with much lower write and warm-read cost ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- SSD artifact staging, speculative read prewarm, and write-behind had small, mixed, or queueing-dependent gains ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).

#### Conclusions

- Gaia files should be served from SSD for active builds; HDD-backed Gaia reads are too slow for the seek-heavy access pattern ([e001](benchmarking/entries/2026-04-01-storage-e001-storage-cache-format-baselines/index.md), [e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- Compression should be used, but `gzip=9` is too slow for build artifacts; Blosc Zstd is the better retained speed/size trade ([e001](benchmarking/entries/2026-04-01-storage-e001-storage-cache-format-baselines/index.md), [e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- Additional I/O optimizations were not worth their runtime complexity ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).

#### Decisions

- Keep canonical Gaia as compressed HDF5, one file per outer pixel, under the hour/declination/pixel directory layout ([e001](benchmarking/entries/2026-04-01-storage-e001-storage-cache-format-baselines/index.md)).
- Serve active-build Gaia reads from a local SSD mirror when the source store has HDD-class seek behavior, with worker-local prepared Gaia table caching for neighbour-oriented traversal reuse ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).
- Use Blosc Zstd through `hdf5plugin` for build artifacts ([e002](benchmarking/entries/2026-04-16-storage-e002-traversal-io-compression-and-artifact-baselines/index.md)).

### Traversal

#### Observations

- Winner-map-based traversal improved the fixed 288-pixel CPU sample from `137.49 s` to `97.48 s` while doing more useful work: it retained unique regularized winners instead of the smaller overlap-pruned catalog, roughly doubling artifact size ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Vectorizing the local regularizer removed the dominant local-selection cost on the 12-pixel sample: local selection fell from `2.230 s/pixel` to `0.076 s/pixel` ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Vectorized feature construction cut resolved plus averaged prediction work from about `1.41 s/pixel` to about `0.63 s/pixel` on the real-data sample ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Compatibility checks kept the old traversal output as a reference and showed where winner-map traversal and dense-field support intentionally changed which asterisms are kept ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- A global adaptive NGS magnitude cap controlled candidate count but lost dense-field coverage; regional FOR-optimized NGS selection kept dense-field coverage near complete while controlling per-region combination work ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).

#### Conclusions

- Winner-map-based traversal is worth retaining because it is faster and keeps far more useful winning asterisms, despite higher memory and larger artifacts ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- The useful traversal optimizations were structural vectorization, not small scheduling or compatibility tweaks ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Regional FOR-optimized NGS selection lets the build include dense areas, especially the Galactic plane, instead of excluding them because asterism finding cannot handle the raw candidate volume ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).

#### Decisions

- Keep winner-map-based traversal, vectorized local regularization, and vectorized feature construction ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Use regional FOR-optimized NGS selection to keep dense areas in scope ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Accept leaving strict legacy exact equivalence where needed to support dense areas ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md), [e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).

### Prediction

#### Observations

- Traversal arrays, feature matrices, and artifact write buffers were too small to explain the multi-GiB worker memory growth; memory mainly grew because prediction calls left memory reserved after they finished ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- Prediction calls with many different row counts made retained memory worse, which led to testing padded prediction batches with a fixed set of allowed sizes ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- GPU prediction cut the fixed-sample wall time by about `33%` compared with CPU prediction, but raised total RAM from about `4.6 GiB` to `11.6 GiB` across three workers ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Padding GPU prediction batches to the dense `1000..25000` row ladder was faster and used less memory than letting every call use its natural row count ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Explicit GPU cache clearing saved RAM but cost time; running without routine cache clearing was fastest, and the retained-policy smoke completed without memory-pressure failure ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Removing row-wise `zd`/`az` tie sorting cut feature-construction time. The existing models were not trained for that ordering, but the forced-tie stress test found only small prediction differences, and the real 12-pixel artifact comparison matched exactly ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).

#### Conclusions

- GPU prediction is worth using, with worker count and scheduling constrained by memory headroom ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Prediction memory should be managed by controlling prediction batch sizes and device choice, not by trimming traversal data structures ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).
- The dense batch ladder is the retained speed/memory tradeoff for prediction ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).

#### Decisions

- Use GPU prediction where memory headroom allows it ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Pad prediction batches to the dense `1000..25000` row ladder ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Use explicit cache clearing only as a memory-pressure tool, not as the normal fast path ([e004](benchmarking/entries/2026-04-18-prediction-e004-device-shape-control-and-dense-field-policy/index.md)).
- Use magnitude-ordered guide-star features and do not reorder tied stars by inner-pixel `zd`/`az` location ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).

#### Follow-Up

1. Retrain and validate the prediction models to only sort NGS by magnitude ([e003](benchmarking/entries/2026-04-17-traversal-e003-winner-map-traversal-cpu-memory-and-compatibility/index.md)).

### Runtime

#### Observations

- GPU prediction made worker memory high enough that work scheduling became the main RAM mitigation: dense or memory-heavy pixels need to be placed deliberately instead of letting too many expensive workers run at once ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md), [e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).
- Memory-aware regional pre-planning produced a static scheduler that uses Galactic latitude as a density proxy and assigns fixed worker pools to low- and high-latitude regions. The fixed `9/6` split pushed memory too hard: workers spent most samples clearing memory, peaked at `25.48 GiB`, and only reached `0.80 pix/s` ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md)).
- Gentler fixed splits, including `8/5` and `8/2`, used less memory and spent less time clearing it, but we did not rerun a full comparable build to measure the end-to-end effect ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md)).
- To compare policies without rerunning every full build, we built a scheduler simulation from measured runtime and RAM behavior. It estimated throughput, memory pressure, and memory-clearing events for fixed and dynamic worker policies ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md)).
- Added a dynamic scheduler that uses Gaia star count and the RAM model instead of Galactic latitude to decide which work can run together. The simulation favored this approach because it kept high-memory work moving without pinning workers to a fixed latitude split: dynamic `9` reached `1.11 pix/s` versus about `0.83 pix/s` for the fixed splits, with lower peak RAM than fixed `9/6` ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md)).
- The first measured dynamic `9` run matched the policy expectation: it processed `41,184` outer pixels at `1.36 pix/s`, reached only `3` memory clearing events, never paused, and peaked at `22.18 GiB` ([e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).
- The dynamic `9` simulation matched measured throughput over the cropped run window, while the RAM simulation overestimated memory pressure ([e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).

#### Conclusions

- Dynamic scheduling is superior to fixed latitude splits for the current GPU build ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md), [e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).
- Scheduler simulation is effective for testing scheduling algorithms before committing to full-build runs ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md), [e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).
- Staying within the MacBook Pro M4 Max RAM ceiling limits throughput, so careful memory scheduling is part of how throughput is increased ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md), [e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).

#### Decisions

- Use dynamic `9` under the current local memory ceiling, with dynamic `8` as the lower-pressure fallback ([e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).
- Keep measured total-RAM guards in control during the run to guard against memory overflow ([e005](benchmarking/entries/2026-04-19-runtime-e005-static-scheduler-full-sky-runtime-and-modeling/index.md), [e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).

#### Follow-Up

1. Study how sensitive the dynamic scheduler is to the runtime/RAM models when run on other computers ([e006](benchmarking/entries/2026-04-21-runtime-e006-dynamic-scheduler-runtime-validation/index.md)).

## Next Steps

1. Run the human-guided retrospective `AI Use:` review entry by entry.
