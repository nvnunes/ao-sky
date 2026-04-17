# ao-sky Benchmarking

## Purpose

This document records benchmark evidence that informs `ao-sky` storage,
scheduling, and cache decisions.

It is now the active benchmark record for this repository. Relevant historical
benchmark material from the legacy implementation has been consolidated here so
`ao-sky` planning does not depend on scattered documents in other repositories.

## Source And Status

- The historical measurements below were originally gathered on `2026-04-01`
  during the pre-`ao-sky` planning work in the legacy implementation.
- They were measured before the current local-SSD failure and should therefore
  be treated as historical baselines rather than current operating truth.
- They still matter because they constrain what had already been learned about
  Gaia I/O, end-to-end build behavior, and HDF5 storage.
- Any new cache-policy decision for `ao-sky` should refresh the relevant
  benchmarks under the current storage setup before the implementation is
  treated as settled.

## Retained Planning Conclusions

- Canonical Gaia storage should use compressed HDF5.
- The canonical storage unit should remain one file per outer pixel.
- The hour/declination/pixel directory organization should remain in place.
- Direct HDD access is poor for many-small-files Gaia access patterns.
- Neighbour-oriented traversal, restart-aware execution, and better scheduling
  remain the main optimization targets.
- The historical evidence did not justify a custom multi-tier cache
  architecture or an always-on local hot cache layered over a fast staged copy.
- The Phase 13 optimization pass kept regional long-lived workers and the
  worker-local prepared Gaia table cache, but did not retain speculative read
  prewarm or write-behind artifact writes as active execution paths.

## 2026-04-01 Storage Baseline

Purpose:

- measure raw device behavior before changing Gaia storage or caching design
- separate sustained transfer performance from many-small-files seek behavior

Devices tested:

- HDD: `/Volumes/Data`
- External SSD: `/Volumes/DataSSD`
- Local SSD: system-local storage

### Sequential Read Baseline

| Device | Read Size | Time | Throughput |
| --- | ---: | ---: | ---: |
| HDD | 1024.0 MiB | 4.31 s | 237.4 MiB/s |
| DataSSD | 1024.0 MiB | 1.07 s | 959.6 MiB/s |
| Local SSD | 1024.0 MiB | 0.17 s | 6147.6 MiB/s |

### Many-Small-Files Baseline

| Device | Files | Bytes Read | Time | Files/s |
| --- | ---: | ---: | ---: | ---: |
| HDD | 5000 | 19.5 MiB | 27.38 s | 182.6 |
| DataSSD | 5000 | 19.5 MiB | 1.41 s | 3548.4 |
| Local SSD | 5000 | 19.5 MiB | 1.30 s | 3847.3 |

Retained interpretation:

- HDD was acceptable for sustained transfer but very poor for seek-heavy
  many-file access.
- DataSSD was already close to local SSD for the critical small-file pattern.
- The historical evidence supported testing staging away from HDD before
  considering a more complex local-cache design. Later Traversal benchmarks did
  not justify retaining artifact staging as an active execution path.

## 2026-04-01 Gaia Access-Pattern Baseline

Purpose:

- measure Gaia HEALPix read behavior more directly than the raw storage
  baseline
- compare one-file, neighbour-heavy, and small-cluster access patterns

Benchmark region:

- central outer pixel: `41430`
- cluster: `41425, 41427, 41428, 41429, 41430, 41433`
- local subset staged for comparison: `22` Gaia files including required
  neighbours

### Initial Access-Pattern Comparison

| Scenario | HDD | DataSSD | Local Workspace Copy |
| --- | ---: | ---: | ---: |
| `SINGLE` | 0.0976 s | 0.0051 s | 0.0107 s |
| `NEIGHBOURS` | 0.2546 s | 0.0659 s | 0.0814 s |
| `CLUSTER_PASS1` | 0.6492 s | 0.3886 s | 0.5232 s |
| `CLUSTER_PASS2` | 0.3730 s | 0.3977 s | 0.5092 s |

### Repeated Local Comparison Using `/tmp`

| Scenario | DataSSD | `/tmp` Local SSD |
| --- | ---: | ---: |
| `SINGLE` | 0.0156 s | 0.0097 s |
| `NEIGHBOURS` | 0.0624 s | 0.0814 s |
| `CLUSTER_PASS1` | 0.3646 s | 0.3799 s |
| `CLUSTER_PASS2` | 0.3975 s | 0.4056 s |

Retained interpretation:

- HDD remained much worse for cold Gaia access.
- DataSSD was already strong for the actual Gaia HEALPix read pattern.
- A true local-SSD subset copy under `/tmp` was only marginally better for the
  one-file case and effectively the same as DataSSD for neighbour-heavy and
  clustered reads.
- The historical evidence favored `DataSSD Pre-stage` over a more elaborate
  local hot-cache design.

## 2026-04-01 Build Worker Baseline

Purpose:

- measure a build-like outer-pixel workflow rather than raw file reads
- compare single-pixel and mixed-density runs
- check whether storage remained dominant once asterism work was included

Important setup notes:

- The benchmark used `_build_outer_pix()` directly because the full
  `build_inner()` path could not be run in that workspace.
- Asterism quality was forced onto the built-in non-model heuristic.
- Gaia was accessed through temporary wrapper roots so the benchmark could read
  the restored legacy tree without changing the benchmark contract.

### Single Outer Pixel (`4727`)

| Storage | Total Wall | Inner Total | Inner Gaia Load | Asterism Total | Asterism Star Load | Asterisms Found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HDD | 3.304 s | 0.091 s | 0.038 s | 3.212 s | 0.392 s | 856 |
| DataSSD | 2.540 s | 0.027 s | 0.006 s | 2.512 s | 0.052 s | 856 |

### 5-Pixel Mixed-Density Batch

| Storage | Workers | Total Wall | Per Pixel |
| --- | ---: | ---: | ---: |
| HDD | 1 | 83.182 s | 16.636 s |
| HDD | 5 | 61.240 s | 12.248 s |
| DataSSD | 1 | 82.336 s | 16.467 s |
| DataSSD | 5 | 105.300 s | 21.060 s |

Retained interpretation:

- DataSSD clearly improved cold single-pixel Gaia load time.
- Even so, most single-pixel wall time was not raw Gaia I/O.
- For the 5-pixel mixed-density run at `cores=1`, HDD and DataSSD were nearly
  identical.
- For this benchmark shape, compute and scheduling overhead dominated enough
  that storage alone did not explain most end-to-end runtime.
- The historical evidence did not justify assuming that more workers or more
  cache layers automatically improve throughput.

## 2026-04-01 Local-Only Comparison

Purpose:

- compare a true local-SSD staging run against the prior HDD and DataSSD worker
  benchmarks
- decide whether a bounded local cache had meaningful upside over DataSSD

Setup:

- staged the working set into `/tmp/gaia-benchmark-local-full/dr3-hpx6`
- copied the exact 5 benchmark pixels plus their immediate neighbours
- reused the same `_build_outer_pix()` worker benchmark and non-model heuristic

### Single Outer Pixel (`4727`)

| Storage | Total Wall | Inner Gaia Load | Asterism Star Load |
| --- | ---: | ---: | ---: |
| HDD | 3.304 s | 0.038 s | 0.392 s |
| DataSSD | 2.540 s | 0.006 s | 0.052 s |
| Local `/tmp` | 2.857 s | 0.017 s | 0.093 s |

### 5-Pixel Mixed-Density Batch At `cores=1`

| Storage | Total Wall | Per Pixel |
| --- | ---: | ---: |
| HDD | 83.182 s | 16.636 s |
| DataSSD | 82.336 s | 16.467 s |
| Local `/tmp` | 81.510 s | 16.302 s |

Retained interpretation:

- The local-only run did not beat DataSSD on the single-pixel benchmark.
- For the 5-pixel mixed-density batch at `cores=1`, local `/tmp`, DataSSD, and
  HDD were effectively the same.
- The historical evidence did not support preferring a bounded local cache over
  DataSSD for that benchmark shape.

## 2026-04-01 HDF5 Size Comparison

Purpose:

- compare existing Gaia FITS files against compressed HDF5 equivalents
- decide whether format should be chosen on storage-efficiency grounds

### Single-File Comparison

| Artifact | Size |
| --- | ---: |
| FITS | 13,826,880 bytes |
| HDF5 (`gzip=9`, `shuffle=True`) | 8,498,866 bytes |
| Size change | -38.5% |

### Worst-Case Level-5 Parent Comparison

| Artifact | Size |
| --- | ---: |
| Four child FITS files total | 301,858,560 bytes |
| Grouped HDF5 with one dataset per child | 151,859,481 bytes |
| Grouped size change | -49.7% |
| Combined HDF5 with one dataset | 131,826,282 bytes |
| Combined size change | -56.3% |

Retained interpretation:

- Compressed HDF5 was materially smaller than the FITS representation for real
  Gaia files.
- Grouping at level 5 was physically feasible, but remained a separate design
  choice from format.
- The retained design decision is compressed HDF5 with one file per outer
  pixel.

## 2026-04-16 Phase 13 Traversal Optimization Samples

Purpose:

- measure the Phase 13 Traversal optimization path across IO, memory,
  compression, and worker-count scaling on the current live GNAO build path
- keep the samples conservative, restartable, and memory-guarded
- separate useful execution settings from settings that only move bottlenecks

Common setup for the initial probes:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Volumes/Data/Galaxy/aosky`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- measurement method: short restart/run samples, stopped before completion
- benchmark caveat: the live build scheduler can hit uneven-density work; the
  write-behind comparison used scratch copies of the same `build.h5` and
  `build.yaml` under `/tmp` so each cap started from the same state
- later fixed-sample subsections record their own Gaia root, worker count,
  telemetry, compression, and memory-guard overrides

### Current Traversal Stage Baseline

Purpose:

- measure the current baseline after settling on SSD Gaia, worker-local prepared
  Gaia table caching, Blosc Zstd derived artifacts, and direct HDD writes
- break worker pixel time into the main Traversal stages that were previously
  grouped under "Other Traversal work"
- keep the same `18` region / `288` outer-pixel sample used for the compression
  and staging comparisons

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- benchmark build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-traversal-baseline-20260416-225527`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample source:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-compression-sweep-bench-20260416-212643/sample-pixels.ecsv`

Results:

- Gaia read/prep is separated from asterism star selection in this view, so
  "Star selection, excluding Gaia read/prep" is the measured star-selection
  bucket minus Gaia load/prep telemetry.
- Rows are additive and sum to the measured `1.470 s/pixel` worker total.

| Stage | Per Pixel | Interpretation |
| --- | ---: | --- |
| Overlap geometry filtering | `0.442 s` | FOV-level grouping, neighbour search, center separations, overlap-area tests, and keep/drop decisions |
| Bright-star filtering | `0.246 s` | Candidate-center rejection around bright stars |
| Candidate generation | `0.216 s` | `find_asterisms()` over prepared NGS rows |
| Point prediction + winner selection | `0.216 s` | Point-model prediction plus best/winner field updates |
| Dust injection | `0.117 s` | Gaia TGE `gaia_A0` injection |
| Field-mean prediction | `0.044 s` | Averaged EE prediction for selected winners |
| Overlap quality prediction | `0.043 s` | Asterism-quality prediction used to rank overlap choices |
| HDF5 artifact write | `0.041 s` | Direct Blosc Zstd `outer.h5` write |
| Prediction context setup | `0.032 s` | Inner/asterism/star plane offsets and candidate inner-pixel matching |
| Star selection, excluding Gaia read/prep | `0.029 s` | Boundary-neighbour star assembly, locality filtering, runtime row validation, NGS filtering |
| Gaia read/prep | `0.021 s` | SSD Gaia file reads plus runtime prep: proper motion, `R`, `hpx14`, read-only table preparation |
| Base inner table construction | `0.018 s` | Dense inner rows plus star, NGS, and asterism counts |
| Unprofiled Traversal overhead | `0.004 s` | Residual timing gap |
| Coverage assignment | `<0.001 s` | Resolved and averaged coverage threshold booleans |
| Inner-pixel assignment | `<0.001 s` | Assign filtered asterism centers to inner HEALPix pixels |
| Local asterism selection | `<0.001 s` | Split retained local asterisms from expanded-neighbour asterisms |
| Persisted asterism shaping | `<0.001 s` | Retained local asterism output shaping |
| **Total worker pixel time** | **`1.470 s`** | Sum of measured per-pixel worker buckets |

Run summary:

- wall time: `152.081 s`
- worker elapsed time: `148.029 s`
- throughput: `1.946 pixels/s`
- artifact size: `152.8 MiB` total, `0.531 MiB/pixel`
- Gaia load/prep telemetry: `6.185 s` total (`3.746 s` raw load,
  `2.438 s` runtime prep). This is not an additive row in the raw table because it
  is part of asterism star selection.
- worker cache: `829` hits, `2051` misses, `2027` evictions, `0.288` hit ratio
- Gaia cache peak: `2.4 MiB`
- peak worker RSS: `3521.1 MiB`

Memory-shape telemetry:

This table comes from a detailed telemetry run. Detailed telemetry is opt-in
(`--telemetry detailed`) and writes raw per-pixel RSS checkpoints plus row-count
context to `<build>/diagnostics/traversal-memory.csv`; normal production runs
keep only the lower-cost build-log telemetry.

| Intermediate | Average Per Pixel | Peak Pixel | Interpretation |
| --- | ---: | ---: | --- |
| Expanded search stars | `2474` rows | `3151` rows | Stars loaded for the outer pixel plus two-ring boundary footprint |
| Search NGS rows | `1100` rows | `1399` rows | Rows entering `find_asterisms()` after magnitude filtering |
| Close star-pair rows | `5057` rows | `7733` rows | Full `search_around_sky()` pair result materialized before pruning |
| Raw asterism rows | `3260` rows | `5684` rows | Candidate asterisms before bright-star and overlap filtering |
| Post-bright-star rows | `3209` rows | n/a | Bright-star exclusion removes little in this sample |
| Post-overlap rows | `586` rows | n/a | Overlap filtering is the largest row-count reducer |
| Local retained asterisms | `471` rows | `622` rows | Persisted local asterism subset |
| Inner/asterism candidate pairs | `25585` rows | `33640` rows | Full inner-pixel/asterism match table materialized before point prediction batching |
| Winner payloads | `9452` objects | `12766` objects | Python NGS payload objects retained until field-mean prediction |
| Inner artifact structured array | `6.125 MiB` | n/a | Dense inner data copied for HDF5 writing |
| Asterism artifact structured array | `0.061 MiB` | n/a | Retained asterism data copied for HDF5 writing |

Detailed RSS checkpoint run:

- benchmark build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-traversal-baseline-20260416-232141`
- detailed output:
  `<build>/diagnostics/traversal-memory.csv` and
  `<build>/diagnostics/traversal-worker-memory.csv`
- rows: `288`
- wall time: `148.359 s`
- worker elapsed time: `147.174 s`
- current RSS max from detailed pixel checkpoints: `1231.5 MiB`
- `resource.ru_maxrss` worker high-water max: `3540.4 MiB`
- worker lifecycle RSS reached only `~419 MiB` after model warmup and runtime
  setup; the multi-GiB high-water mark is therefore not caused by worker
  initialization
- stage high-water checkpoints show the first large `ru_maxrss` jump happens
  during dust injection. A later artifact-writer split confirmed the high-water
  mark is already present before artifact conversion, HDF5 open, or HDF5 dataset
  writes. This points to the first Gaia TGE `dustmaps` query/load in each worker,
  not HDF5/filter/compression.

Largest per-pixel RSS growth:

| Outer Pix | Worker | Start RSS | Peak RSS | Post-GC RSS | Growth | Context Pairs | Close Pairs | Winner Payloads |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `34448` | `0` | `417.4 MiB` | `825.8 MiB` | `824.2 MiB` | `408.4 MiB` | `30877` | `7614` | `12035` |
| `27027` | `2` | `418.5 MiB` | `863.4 MiB` | `862.3 MiB` | `444.9 MiB` | `31551` | `7637` | `12762` |
| `33764` | `1` | `419.3 MiB` | `843.7 MiB` | `843.4 MiB` | `424.4 MiB` | `30286` | `6649` | `11619` |
| `34449` | `0` | `824.2 MiB` | `1004.2 MiB` | `1004.2 MiB` | `180.0 MiB` | `33640` | `7255` | `12380` |
| `33761` | `1` | `843.4 MiB` | `1012.2 MiB` | `1012.2 MiB` | `168.8 MiB` | `31174` | `6581` | `11463` |
| `27028` | `2` | `862.3 MiB` | `1020.1 MiB` | `1020.1 MiB` | `157.8 MiB` | `30059` | `6861` | `11745` |

Stage RSS deltas across the sample:

| Stage Boundary | Median Delta | P95 Delta | Max Delta |
| --- | ---: | ---: | ---: |
| Star selection | `0.0 MiB` | `1.4 MiB` | `14.0 MiB` |
| Find asterisms | `0.0 MiB` | `0.0 MiB` | `7.2 MiB` |
| Filtering | `0.0 MiB` | `4.3 MiB` | `79.0 MiB` |
| Context setup | `0.0 MiB` | `0.3 MiB` | `16.8 MiB` |
| Point prediction | `0.0 MiB` | `9.5 MiB` | `150.8 MiB` |
| Field-mean prediction | `0.0 MiB` | `2.1 MiB` | `130.7 MiB` |
| Artifact write | `0.0 MiB` | `5.0 MiB` | `6.5 MiB` |
| Post-GC change | `0.0 MiB` | `0.0 MiB` | `191.0 MiB` |

Stage high-water (`ru_maxrss`) increments:

| Stage Boundary | Max High-Water Increase |
| --- | ---: |
| Star selection | `16.0 MiB` |
| Find asterisms | `7.3 MiB` |
| Filtering | `82.0 MiB` |
| Context setup | `17.9 MiB` |
| Point prediction | `148.5 MiB` |
| Field-mean prediction | `129.7 MiB` |
| Coverage | `0.0 MiB` |
| Dust injection | `2881.2 MiB` |
| Persisted asterism shaping | `0.0 MiB` |
| Artifact writer entry | `0.0 MiB` |
| Artifact convert inner | `0.0 MiB` |
| Artifact convert asterisms | `0.0 MiB` |
| HDF5 open | `0.0 MiB` |
| HDF5 inner dataset write | `0.0 MiB` |
| HDF5 asterisms dataset write | `0.0 MiB` |
| HDF5 close | `0.0 MiB` |
| Artifact replace | `0.0 MiB` |
| Post-GC change | `0.0 MiB` |

Representative first high-water pixels:

| Outer Pix | Worker | Start High-Water | Final High-Water | Current RSS After Dust | Dust High-Water Increase |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `33764` | `1` | `419.7 MiB` | `3637.6 MiB` | `417.4 MiB` | `2881.2 MiB` |
| `27027` | `2` | `416.8 MiB` | `3318.2 MiB` | `437.9 MiB` | `2559.9 MiB` |
| `34448` | `0` | `418.3 MiB` | `3291.2 MiB` | `417.2 MiB` | `2470.5 MiB` |

Retained interpretation:

- overlap geometry filtering is the largest measured bucket, followed by
  bright-star filtering
- candidate generation and point prediction/winner selection are the next
  largest buckets and are nearly equal
- overlap-quality prediction is not the dominant part of filtering; the
  expensive work is primarily geometric filtering
- dust injection is now larger than artifact writes, which reinforces that
  HDF5 artifact IO is no longer the near-term optimization target
- Gaia access is visible but small with SSD Gaia plus the prepared-table cache;
  optimizing the science pipeline stages is more promising than adding IO
  machinery
- Gaia cache memory is not the cause of multi-GiB worker RSS in this sample;
  cache peak was only `2.4 MiB`
- the largest materialized row-count structure is the inner/asterism candidate
  pair table, not the Gaia table or artifact write copy
- artifact conversion duplicates the dense inner table during write, but at
  about `6.1 MiB/pixel` it is not the primary memory issue for this sample
- artifact writing is not a wall-time bottleneck and is not the source of the
  large `ru_maxrss` high-water mark in this sample
- the high-water mark comes from dust injection, most likely the first
  `dustmaps.gaia_tge.GaiaTGEQuery` load/query in each worker. The current RSS
  after dust is far smaller, so future memory-limit policy should distinguish
  current RSS from high-water RSS.
- the current non-crowded sample does not reproduce a pathological close-pair or
  context-pair explosion; dense pixels remain the place where those structures
  need proactive guardrails

Follow-up dust-cache probe:

- implementation change: `init` now converts the source Gaia TGE CSV into a
  build-local dense A0 NumPy cache at
  `<build>/dust/ao-sky-gaia-tge-a0-hpx<max_level>.npy`; traversal and
  aggregation memory-map that cache instead of constructing
  `dustmaps.gaia_tge.GaiaTGEQuery` inside worker processes
- real cache created from `/Volumes/Data/Galaxy/aosky/gaia_tge` for level 9
  and moved into the version-local build dust directory:
  `12,583,040` bytes, about `12.0 MiB`
- isolated probe used outer pixel `33764`, `outer_level=6`, `max_data_level=9`,
  and `inner_level=14`

| Dust Path | Current RSS Start | Current RSS After Sample | Peak RSS | Mean A0 |
| --- | ---: | ---: | ---: | ---: |
| `dustmaps.GaiaTGEQuery` constructor + query | `120.8 MiB` | `435.5 MiB` | `3488.7 MiB` | `0.081100` |
| build-local mmap A0 cache | `117.8 MiB` | `118.4 MiB` | `118.8 MiB` | `0.081100` |

Full fixed-sample traversal confirmation:

- benchmark root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-traversal-baseline-20260416-235633`
- sample: same fixed 288-pixel non-crowded sample
- workers: `3`
- Gaia root: SSD mirror
- artifact writes: direct HDD writes with Blosc Zstd

| Metric | Result |
| --- | ---: |
| Wall time | `146.804 s` |
| Pixels/s | `1.974` |
| Avg total worker time | `1.436 s/pixel` |
| Avg dust time | `0.099 s/pixel` |
| Peak worker RSS | `1044.8 MiB` |
| Failed pixels | `0` |

Retained decision:

- keep `fetch-gaia` store-scoped: it fetches the source Gaia TGE CSV and Gaia
  star store only
- create the native A0 cache during `init`, where `maps.max_level` is known
- persist the build-local dust root in `build.h5` so all later `run`,
  `restart`, and aggregation paths use the mmap cache

### Traversal Worker Scaling With Memory Guards

Purpose:

- measure fixed-sample throughput as the worker count approaches the available
  performance-core budget on the M4 Max test system
- keep the same 288-pixel non-crowded sample for each run
- confirm the new memory guards are practical during real Traversal execution

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- artifact writes: direct HDD writes with Blosc Zstd
- telemetry: basic profile logging
- worker memory guard: `2048 MiB`
- parent aggregate current-RSS guard: `12288 MiB`
- benchmark roots:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-traversal-baseline-20260417-001121`
  and
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-traversal-baseline-20260417-001755`

Results:

| Workers | Wall Time | Pixels/s | Avg Worker Time/px | Avg Other Traversal/px | Avg Artifact Write/px | Peak Worker RSS | Failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3` | `137.491 s` | `2.095` | `1.332 s` | `1.275 s` | `0.036 s` | `1060.7 MiB` | `0` |
| `4` | `109.566 s` | `2.629` | `1.332 s` | `1.275 s` | `0.039 s` | `1154.2 MiB` | `0` |
| `5` | `92.863 s` | `3.101` | `1.334 s` | `1.276 s` | `0.038 s` | `1541.8 MiB` | `0` |
| `6` | `72.540 s` | `3.970` | `1.354 s` | `1.297 s` | `0.037 s` | `1372.1 MiB` | `0` |
| `7` | `78.465 s` | `3.670` | `1.557 s` | `1.489 s` | `0.046 s` | `1427.8 MiB` | `0` |
| `8` | `69.404 s` | `4.150` | `1.393 s` | `1.335 s` | `0.039 s` | `1417.4 MiB` | `0` |
| `9` | `59.745 s` | `4.821` | `1.599 s` | `1.531 s` | `0.045 s` | `1370.8 MiB` | `0` |

Retained interpretation:

- `9` workers was fastest on this fixed non-crowded sample, completing the
  sample in `59.745 s`
- `6` workers is the conservative fallback point if dense regions or other
  desktop activity make full-core execution less desirable
- the `2048 MiB` per-worker guard did not trip; observed peak worker RSS stayed
  below `1542 MiB` across all runs
- the `12288 MiB` parent aggregate guard did not trip; it remains a whole-system
  safety guard rather than a normal scheduling limit
- the 7-worker run was slower than 6 and 8 in this sample, so worker scaling
  should be treated as empirical rather than perfectly monotonic

### Fixed-Pixel Gaia Root Comparison

Purpose:

- compare the HDD Gaia root against the SSD Gaia mirror while holding Traversal
  work fixed
- compare read prewarm on and off against the exact same outer-pixel sample
- isolate Gaia read-root behavior from final build-artifact HDD writes

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- HDD Gaia root: `/Volumes/Data/Galaxy/aosky`
- SSD Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- scratch build root: `/tmp/ao-sky-fixed-gaia-bench-20260416-192625`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- artifact writes: scratch build directories under `/tmp`
- sample criteria: `60` loaded outer pixels with `1000 <= star_count <= 2500`
  and `abs(galactic latitude) >= 40 deg`
- exact outer-pixel list:
  `0, 4478, 5203, 5305, 5400, 5496, 5591, 5711, 5907, 6002, 6110, 10292, 10407, 10537, 10729, 10826, 10924, 11019, 11116, 11212, 16400, 16597, 16714, 16813, 16988, 17251, 17410, 17505, 17601, 17696, 17799, 17898, 17999, 18138, 18261, 27070, 27495, 27645, 27818, 27940, 28061, 28158, 28253, 28348, 28444, 28539, 28635, 35253, 35511, 35623, 35719, 35815, 35993, 36091, 36365, 36461, 36556, 36652, 36767, 36863`

Results:

| Case | Prewarm Window | Completed Pixels | Elapsed | Pixels/s | Avg Pixel | Avg Artifact Write | Avg Gaia Load | Avg Raw Gaia Load | Avg Prepare | Prewarm Completed | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDD Gaia | `0` | `60` | `44.128 s` | `1.360` | `2.129 s` | `0.363 s` | `0.193 s` | `0.180 s` | `0.013 s` | `0` | `3884.5 MiB` |
| SSD Gaia | `0` | `60` | `40.416 s` | `1.485` | `1.950 s` | `0.359 s` | `0.033 s` | `0.021 s` | `0.012 s` | `0` | `3912.3 MiB` |
| HDD Gaia | `2` | `60` | `42.845 s` | `1.400` | `2.074 s` | `0.360 s` | `0.184 s` | `0.159 s` | `0.025 s` | `57` | `3895.1 MiB` |
| SSD Gaia | `2` | `60` | `41.000 s` | `1.463` | `1.976 s` | `0.360 s` | `0.053 s` | `0.030 s` | `0.023 s` | `57` | `3903.5 MiB` |

Retained interpretation:

- the SSD Gaia mirror materially reduced raw Gaia load time, but total
  Traversal elapsed time improved by only about `8%` on this moderate-density
  sample because asterism/prediction work and artifact writing still dominate
- HDD read prewarm with `gaia_prewarm_window=2` produced a small improvement,
  about `3%` elapsed, and completed almost all planned prewarm reads
- SSD read prewarm slightly hurt this sample, likely because SSD raw loads are
  already cheap while prewarm still adds queueing and preparation overhead
- for SSD-backed runs, the simplest path is no speculative read prewarm
- cache hit ratio was `0.0` in this exact sample because the selected pixels
  were intentionally spread out; this comparison measures cold per-pixel Gaia
  root behavior rather than neighbour-local cache reuse

### Fixed-Region Prepared Gaia Cache Signal

Purpose:

- isolate the worker-local prepared Gaia table cache from speculative read
  prewarm
- hold Traversal work fixed by selecting complete low-density HEALPix regions
  instead of scattered individual outer pixels
- keep the regional long-lived worker path active in both cases so
  `gaia_cache_entries=0` disables only table retention
- repeat the same cache-on/cache-off comparison against both the HDD Gaia root
  and the SSD Gaia mirror

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- HDD Gaia root: `/Volumes/Data/Galaxy/aosky`
- SSD Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- HDD scratch build root: `/tmp/ao-sky-cache-signal-bench-20260416-201046`
- SSD scratch build root: `/tmp/ao-sky-cache-signal-ssd-bench-20260416-202355`
- worker count: `3`
- worker memory guard: `6144 MiB`
- artifact writes: scratch build directories under `/tmp`
- sample: `18` complete HEALPix regions at level `4`, totaling `288` level-6
  outer pixels
- sample criteria: every region child loaded, region total star count between
  `16000` and `45000`, and `abs(galactic latitude) >= 40 deg`
- exact region list:
  `1784, 324, 1769, 1762, 370, 1124, 1090, 1140, 277, 1144, 383, 1081, 554, 2247, 663, 1689, 2110, 2153`

Results:

| Gaia Root | Case | Cache Entries | Cache MiB | Completed Pixels | Wall | Elapsed | Pixels/s | Avg Pixel | Avg Artifact Write | Gaia Load | Raw Gaia Load | Gaia Prepare | Cache Hits | Cache Misses | Hit Ratio | Evictions | Peak Cache | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HDD | cache off | `0` | `0` | `288` | `182.029 s` | `179.198 s` | `1.607` | `1.805 s` | `0.388 s` | `14.357 s` | `10.965 s` | `3.392 s` | `0` | `0` | `0.000` | `0` | `0.0 MiB` | `3920.5 MiB` |
| HDD | cache on | `8` | `128` | `288` | `183.013 s` | `179.929 s` | `1.601` | `1.807 s` | `0.392 s` | `10.736 s` | `8.326 s` | `2.410 s` | `829` | `2051` | `0.288` | `2027` | `2.4 MiB` | `3918.6 MiB` |
| SSD | cache off | `0` | `0` | `288` | `178.547 s` | `175.818 s` | `1.638` | `1.775 s` | `0.389 s` | `8.149 s` | `4.835 s` | `3.315 s` | `0` | `0` | `0.000` | `0` | `0.0 MiB` | `3921.7 MiB` |
| SSD | cache on | `8` | `128` | `288` | `180.336 s` | `177.327 s` | `1.624` | `1.785 s` | `0.392 s` | `6.087 s` | `3.666 s` | `2.419 s` | `829` | `2051` | `0.288` | `2027` | `2.4 MiB` | `3929.1 MiB` |

Retained interpretation:

- the prepared Gaia table cache reduced measured Gaia load and preparation time
  on both roots: about `25%` on HDD, from `14.357 s` to `10.736 s`, and about
  `25%` on SSD, from `8.149 s` to `6.087 s`
- the same runs did not improve end-to-end wall time because Gaia loading was
  only a small part of total Traversal time for this sample; prediction,
  asterism work, artifact writes, and worker critical-path balance dominated
- moving Gaia reads from HDD to SSD was more visible than enabling the
  prepared-table cache for this sample: cache-off wall time improved by about
  `3.5 s`, from `182.029 s` to `178.547 s`
- keeping the worker-local prepared-table cache is still worthwhile because it
  removes redundant Gaia transforms on neighbour-oriented traversal and has low
  memory cost when bounded, but it should not be described as a guaranteed
  all-sky wall-time win on moderate-density samples
- speculative read prewarm remains out of the active implementation because
  the cleaner cache-on/cache-off signal shows the important retained mechanism
  is prepared-table reuse, not background loading

### Initial Read Prewarm Probe

This first probe measured prewarm on the live lineage with `workers=3` and
write-behind disabled. It was not a clean prewarm-only comparison because it
also exercised the regional long-lived worker path and the prepared Gaia table
cache.

| Setting | Representative Warmed Worker Rate | Prewarm Behavior | Interpretation |
| --- | ---: | --- | --- |
| `gaia_prewarm_window=0` | about `0.238-0.261 s/pixel` | no cache-hit help | control path |
| `gaia_prewarm_window=1` | about `0.170-0.179 s/pixel` | near one cache hit per computed pixel | clear improvement |
| `gaia_prewarm_window=2` | about `0.163-0.169 s/pixel` | no prewarm failures or drops | best observed balance |
| `gaia_prewarm_window=4` | about `0.169-0.174 s/pixel` | stable but no meaningful gain | not worth extra lookahead |

Retained interpretation:

- the initial `0` versus nonzero window comparison overstated the value of
  prewarm because it mixed prewarm with regional scheduling and prepared-table
  cache reuse
- later fixed-pixel measurements showed that prewarm itself provided little
  benefit once the regional worker path and prepared-table cache were already
  active
- speculative read prewarm was removed from the active implementation; revisit
  only if future profiling shows workers blocked on cold Gaia reads that cannot
  be amortized by neighbour-oriented cache reuse

### Expanded Artifact Write Telemetry

Purpose:

- split the previous `artifact_write_s` counter into conversion, HDF5 dataset
  creation/compression, HDF5 close, replace, and promotion components
- keep the same fixed-region sample and the same SSD-stage-then-promote
  execution shape

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- benchmark build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-artifact-stage-profile-bench-20260416-210109`
- artifact stage root: `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample: same `18` region / `288` outer-pixel sample as above

Results:

| Bucket | Total | Avg Per Pixel | Interpretation |
| --- | ---: | ---: | --- |
| Total worker pixel time | `514.395 s` | `1.786 s` | summed over workers |
| Other Traversal work | `395.578 s` | `1.374 s` | asterisms, prediction, winner selection, dust, inner shaping |
| Gaia load/prep | `6.095 s` | `0.021 s` | SSD Gaia plus prepared-table cache |
| Artifact write | `112.722 s` | `0.391 s` | worker-side staged HDF5 creation |
| Artifact table conversion | `0.358 s` | `0.001 s` | negligible |
| HDF5 total | `112.318 s` | `0.390 s` | dominant artifact-write cost |
| HDF5 inner dataset | `110.803 s` | `0.385 s` | dominant HDF5 cost |
| HDF5 asterisms dataset | `1.370 s` | `0.005 s` | small |
| HDF5 open + close | `0.144 s` | `0.001 s` | negligible |
| Worker-side replace | `0.031 s` | `<0.001 s` | negligible |
| Parent promotion | `0.166 s` | `0.001 s` | SSD stage to HDD build tree |

Artifact volume:

- inner rows: `18,874,368` total, exactly `65,536` rows per outer pixel
- asterism rows: `135,718` total, about `471` rows per outer pixel
- artifact size: `154.9 MiB` total, about `0.538 MiB` per outer pixel

Retained interpretation:

- artifact write time is not raw disk copy time and not Astropy-to-NumPy table
  conversion time
- the dominant cost is HDF5 creation/compression of the dense `inner` dataset
- because each level-6 outer pixel writes `65,536` dense inner rows, this cost
  exists even in low-density regions
- useful next experiments are compression-level or codec comparisons for
  `inner`, and possibly separating dense `inner` representation choices from
  asterism output; SSD staging alone is not the lever

### No-Compression Artifact Comparison

Purpose:

- measure whether the dominant HDF5 `inner` cost is compression rather than
  basic dataset creation
- use the same fixed-region sample, SSD Gaia root, prepared Gaia cache, and
  SSD-stage-then-HDD-promote execution shape as the expanded telemetry run
- disable derived-artifact compression only for the benchmark process with
  `AO_SKY_ARTIFACT_COMPRESSION=none`

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- compressed reference build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-artifact-stage-profile-bench-20260416-210109`
- no-compression build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-artifact-stage-profile-bench-20260416-211413`
- artifact stage root: `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample: same `18` region / `288` outer-pixel sample as above

Results:

| Bucket | Compressed | No Compression | Change |
| --- | ---: | ---: | ---: |
| Wall time | `180.775 s` | `144.572 s` | `-20.0%` |
| Worker elapsed | `177.261 s` | `140.575 s` | `-20.7%` |
| Pixels/s | `1.625` | `2.049` | `+26.1%` |
| Artifact write | `112.722 s` | `0.777 s` | `-99.3%` |
| HDF5 inner dataset | `110.803 s` | `0.168 s` | `-99.8%` |
| Parent promotion | `0.166 s` | `0.569 s` | `+0.403 s` |
| Artifact volume | `154.9 MiB` | `1783.3 MiB` | `11.5x` |
| Avg artifact size | `0.538 MiB/pixel` | `6.192 MiB/pixel` | `11.5x` |
| Peak RSS | `3934.0 MiB` | `3754.3 MiB` | roughly unchanged |

Per-pixel worker-time comparison:

| Chrono Bucket | Compressed Avg/Pixel | No-Compression Avg/Pixel |
| --- | ---: | ---: |
| Gaia load/prep | `0.021 s` | `0.021 s` |
| Traversal compute after Gaia | `1.374 s` | `1.387 s` |
| Artifact table conversion | `0.001 s` | `0.001 s` |
| HDF5 open/close | `0.001 s` | `0.001 s` |
| HDF5 inner dataset write/compress | `0.385 s` | `0.001 s` |
| HDF5 asterisms dataset write/compress | `0.005 s` | `<0.001 s` |
| Worker-side replace | `<0.001 s` | `<0.001 s` |
| Total worker pixel time | `1.786 s` | `1.411 s` |

Parent-side SSD-stage promotion is outside worker pixel time:

| Promotion | Compressed | No Compression |
| --- | ---: | ---: |
| Promote time | `0.001 s/pixel` | `0.002 s/pixel` |
| Promoted volume | `0.538 MiB/pixel` | `6.192 MiB/pixel` |

No-compression additive worker-time breakdown:

| Bucket | Total | Avg Per Pixel | Share |
| --- | ---: | ---: | ---: |
| Gaia load/prep | `6.138 s` | `0.021 s` | `1.5%` |
| Traversal compute after Gaia | `399.399 s` | `1.387 s` | `98.3%` |
| Artifact table conversion | `0.357 s` | `0.001 s` | `<0.1%` |
| HDF5 inner dataset | `0.168 s` | `0.001 s` | `<0.1%` |
| HDF5 asterisms dataset | `0.051 s` | `<0.001 s` | `<0.1%` |
| HDF5 open/close | `0.161 s` | `0.001 s` | `<0.1%` |
| Worker-side replace | `0.026 s` | `<0.001 s` | `<0.1%` |
| Total worker pixel time | `406.312 s` | `1.411 s` | `100.0%` |

Retained interpretation:

- the dominant artifact-write cost was gzip compression of the dense `inner`
  dataset, not HDF5 dataset creation itself
- disabling artifact compression produced a large speedup on this sample, but
  increased artifact volume by `11.5x`
- a full level-6 traversal extrapolation from this sample would be roughly
  `26 GiB` compressed versus roughly `300 GiB` uncompressed for outer
  artifacts, before maps and other build products
- the next useful benchmark is not simply compression on/off; it is a
  compression-level or codec sweep that searches for a better speed/size
  balance than `gzip=9`

### Compression Codec Sweep

Purpose:

- compare faster HDF5 compression modes against the former `gzip=9` default
  and the no-compression control
- use the same fixed-region sample, SSD Gaia root, prepared Gaia cache, and
  SSD-stage-then-HDD-promote execution shape
- keep the agreed summary dimensions focused on total worker time per pixel,
  HDF5 time per pixel, and average artifact size

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- sweep build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-compression-sweep-bench-20260416-212643`
- artifact stage root: `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample: same `18` region / `288` outer-pixel sample as above
- plugin codecs used `hdf5plugin` with Blosc `clevel=5`

Results:

| Artifact Mode | Total Time/Px | HDF5 Time/Px | Avg Artifact Size |
| --- | ---: | ---: | ---: |
| `gzip=9 + shuffle` | `1.786 s` | `0.390 s` | `0.538 MiB` |
| `gzip=1 + shuffle` | `1.436 s` | `0.028 s` | `0.618 MiB` |
| `gzip=4 + shuffle` | `1.447 s` | `0.046 s` | `0.562 MiB` |
| `lzf + shuffle` | `1.421 s` | `0.010 s` | `0.861 MiB` |
| `blosc-lz4` | `1.420 s` | `0.006 s` | `0.798 MiB` |
| `blosc-zstd` | `1.439 s` | `0.036 s` | `0.531 MiB` |
| `bitshuffle-lz4` | `1.422 s` | `0.015 s` | `0.817 MiB` |
| `uncompressed` | `1.411 s` | `0.001 s` | `6.192 MiB` |

Read/decompression check:

- read comparison used `96` existing `outer.h5` files per mode from the same
  benchmark artifacts
- "warm read" means the second full read pass, after the operating system page
  cache has already seen the files; it is useful for comparing decompression
  and HDF5 materialization cost, but it is not a cold disk-read benchmark
- uncompressed warm reads are expected to be fastest because they avoid
  decompression and read cached bytes from memory

| Artifact Mode | Bytes Read | Warm Read Time | Warm Read/File |
| --- | ---: | ---: | ---: |
| `gzip=9 + shuffle` | `49.3 MiB` | `1.282 s` | `13.36 ms` |
| `gzip=1 + shuffle` | `56.8 MiB` | `1.091 s` | `11.36 ms` |
| `blosc-zstd` | `48.6 MiB` | `0.542 s` | `5.65 ms` |
| `blosc-lz4` | `73.2 MiB` | `0.380 s` | `3.96 ms` |
| `uncompressed` | `594.1 MiB` | `0.091 s` | `0.95 ms` |

Retained interpretation:

- every tested alternative to `gzip=9` removed most of the artifact-write
  bottleneck
- `blosc-lz4` was the fastest compressed option in this sample and stayed
  close to uncompressed runtime while reducing artifact size by about `7.8x`
  compared with uncompressed
- `blosc-zstd` produced the smallest artifacts, slightly smaller than
  `gzip=9`, while reducing HDF5 time by about `91%`; it also decompressed
  about `2x` faster than gzip in the warm-read check
- `gzip=1` is an attractive built-in fallback: it is much faster than
  `gzip=9`, keeps artifacts under `0.7 MiB/pixel`, and does not require HDF5
  plugin support
- a production codec decision should weigh dependency and portability: Blosc
  modes require plugin availability when reading artifacts, while gzip and lzf
  are available through the standard HDF5/h5py stack

Decision:

- use `blosc-zstd` as the default compression for derived build artifacts
  (`outer.h5`, `maps-hpx<level>.h5`, and augmentation datasets)
- require `hdf5plugin` as a normal runtime dependency so build writers and
  package readers can register the Blosc HDF5 filter
- keep canonical Gaia store compression unchanged; the codec decision applies
  to derived build artifacts, not raw shared Gaia files

Rationale:

- `blosc-zstd` gave the smallest measured compressed artifacts in this sample:
  `0.531 MiB/pixel`, slightly smaller than `gzip=9`
- write-time cost was close to the fast alternatives: `1.439 s/pixel` total
  versus `1.420 s/pixel` for `blosc-lz4` and `1.411 s/pixel` uncompressed
- warm-read decompression was about `2x` faster than gzip while preserving the
  strong size reduction
- the dependency tradeoff is acceptable for the intended macOS/Linux build
  environments; direct `h5py` readers must import `hdf5plugin` before opening
  plugin-compressed datasets

### Fixed-Region Artifact Staging Signal

Purpose:

- compare direct HDD artifact writes against SSD-staged artifact writes
- keep the same `18` region / `288` outer-pixel sample as the prepared-cache
  signal
- use SSD Gaia and the prepared Gaia cache in both cases so artifact behavior
  is the only intended difference

Implementation tested:

- workers wrote flat staged files named `<outer_pix>.h5` under a per-build
  subdirectory of `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- the parent process computed the canonical HEALPix destination in the HDD
  build tree, copied each staged file into a temporary file beside the final
  artifact, atomically replaced the final `outer.h5`, removed the staged file,
  and only then marked the outer pixel done
- this is intentionally different from write-behind: completed tables are not
  buffered in memory, and the worker reports completion only after the staged
  HDF5 file exists
- the design keeps artifact ownership worker-local: large `inner` and
  `asterisms` tables are never serialized back to the parent process, and the
  parent only sees a completed staged HDF5 file path plus telemetry

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- benchmark build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-artifact-stage-hdd-bench-20260416-204051`
- artifact stage root: `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample: same region list as the fixed-region prepared-cache benchmark

Results:

| Case | Completed Pixels | Wall | Elapsed | Pixels/s | Avg Pixel | Avg Artifact Write | Promotions | Promote Time | Promoted Size | Gaia Load | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct HDD write | `288` | `179.668 s` | `176.981 s` | `1.627` | `1.781 s` | `0.396 s` | `0` | `0.000 s` | `0.0 MiB` | `5.965 s` | `3930.6 MiB` |
| SSD stage then promote | `288` | `178.843 s` | `175.837 s` | `1.638` | `1.779 s` | `0.390 s` | `288` | `0.242 s` | `154.9 MiB` | `5.911 s` | `3910.3 MiB` |

Retained interpretation:

- SSD artifact staging improved wall time by only `0.825 s`, about `0.5%`, on
  this sample
- promotion from SSD staging into the HDD build tree was cheap in aggregate:
  `154.9 MiB` promoted in `0.242 s`
- the measured `artifact_write_s` remained about `112-114 s` summed across
  workers even when workers wrote the HDF5 files to SSD, which suggests HDF5
  dataset construction/compression dominates this artifact-write counter more
  than raw disk write latency for these pixels
- do not retain artifact staging as an active runtime knob unless future
  evidence shows direct worker-owned writes have again become a critical-path
  bottleneck
- the implementation is useful evidence, but the measured result does not
  justify keeping staging in the build implementation

### Blosc Zstd Staging Recheck

Purpose:

- recheck artifact staging after the derived-artifact compression default moved
  from `gzip=9` to Blosc Zstd
- keep the same SSD Gaia root, prepared Gaia cache, worker count, memory guard,
  and `18` region / `288` outer-pixel sample used for the compression sweep
- compare direct HDD writes against SSD-stage-then-promote with the new default
  codec

Setup:

- source build lineage: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- benchmark build root:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-blosc-staging-value-20260416-215536`
- artifact stage root: `/Users/nelsonnunes/ao-sky-cache/artifact-stage`
- worker count: `3`
- worker Gaia cache: `8` entries, `128 MiB`
- worker memory guard: `6144 MiB`
- sample source:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-compression-sweep-bench-20260416-212643/sample-pixels.ecsv`

Results:

| Case | Completed Pixels | Wall | Elapsed | Pixels/s | Avg Pixel | Avg Artifact Write | Avg HDF5 | Artifact Size/Px | Promotions | Promote Time | Gaia Load | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct HDD write | `288` | `146.938 s` | `143.621 s` | `2.005` | `1.440 s` | `0.040 s` | `0.039 s` | `0.531 MiB` | `0` | `0.000 s` | `6.115 s` | `3710.3 MiB` |
| SSD stage then promote | `288` | `147.165 s` | `144.109 s` | `1.998` | `1.437 s` | `0.037 s` | `0.036 s` | `0.531 MiB` | `288` | `0.175 s` | `5.981 s` | `3825.6 MiB` |

Retained interpretation:

- with Blosc Zstd, artifact write time fell from the old `~0.39 s/pixel`
  `gzip=9` cost to `~0.04 s/pixel`
- SSD staging made worker-side HDF5 writes slightly faster, but wall time did
  not improve; the staged run was `0.227 s` slower overall on this sample
- parent promotion remained cheap (`152.8 MiB` in `0.175 s`), but once the
  compression bottleneck was removed there was no meaningful staging value to
  recover
- the likely storage explanation is layered buffering: macOS page cache, APFS
  buffering, HDF5 buffering, and the HDD or enclosure's onboard cache can
  absorb these small compressed writes and flush them outside the Traversal
  critical path
- this result should not be read as "HDD equals SSD" in general; it means the
  synchronous write component for `~0.53 MiB` Blosc Zstd `outer.h5` artifacts is
  not currently a bottleneck when Gaia reads are already served from SSD
- direct writes are the normal path; artifact staging is not retained in the
  active implementation
- do not build further write-pipeline complexity without new benchmark evidence
  that direct worker-owned writes have become a critical-path bottleneck

### Artifact Write-Behind

Write-behind was measured on scratch build copies so each cap began from the
same scheduler state. The main comparison used the then-current
`gaia_prewarm_window=2` path.

Implementation tested:

- split Traversal work into a build step and a write step, so the worker could
  enqueue completed `inner` and `asterisms` tables for `outer.h5` writing
- used one worker-local IO thread for both Gaia prewarm reads and artifact
  writes
- kept artifact writes worker-owned so large tables were not serialized back to
  the parent process
- reported an outer pixel complete only after its write result succeeded
- tracked pending write memory from the estimated in-memory table sizes and
  applied backpressure when the next queued write would exceed the configured
  cap
- logged write-submitted, write-completed, write-failed, backpressure seconds,
  and pending-write peak MiB in worker profile lines
- initially prioritized writes strictly over prewarm reads, then changed the IO
  loop to service one queued prewarm read opportunistically after each write
  because strict write priority starved read prewarm

| Setting | Representative Worker 0 At 200 Pixels | Prewarm Behavior | Pending Write Peak | Interpretation |
| --- | ---: | --- | ---: | --- |
| disabled | `85.3 s elapsed`, `0.380 s/pixel` | preserved | `0.0 MiB` | safe control |
| `16 MiB` | `83.1 s elapsed`, `0.369 s/pixel` | mostly starved before fairness; backpressure observed | `12.4 MiB` | too tight |
| `32 MiB` with fair IO loop | `75.2 s elapsed`, `0.329 s/pixel` | preserved, no drops | `30.6 MiB` | useful |
| `64 MiB` with fair IO loop | `74.4 s elapsed`, `0.325 s/pixel` | preserved, no drops | `30.6 MiB` | best observed balance |
| early `256 MiB` shape | `81.2 s elapsed`, `0.358 s/pixel` | starved prewarm before fairness | `24.5 MiB` | too aggressive before fairness |

Retained interpretation:

- naive write-priority write-behind can make the total path worse by starving
  read prewarm
- the least-bad tested IO loop prioritized writes but still serviced one queued
  prewarm read opportunistically after each write
- `artifact_write_behind_mb=64` was the best observed balance in these samples:
  it improved elapsed time without increasing worker peak RSS materially in the
  prewarm-enabled prototype
- the observed benefit was not large enough to justify the added async
  completion path, failure surface, memory accounting, backpressure logic, and
  profiling complexity
- write-behind was removed from the active implementation; revisit only if
  future dense-region evidence shows synchronous artifact writes are a dominant
  bottleneck

## 2026-04-17 Known-Good Legacy Compatibility Check

Purpose:

- preserve a reference point before later phases intentionally change the
  winner/overlap algorithm
- confirm that the current optimized Traversal path still matches the legacy
  build path where compatibility is still expected
- use the live legacy implementation as the comparison source, not only
  repo-native fixtures

Command shape:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py \
  --sample full \
  --workers 3 \
  --gaia-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --dust-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --model-root ../survey_tools/data/models \
  --allow-local-winner-divergence \
  --allow-boundary-overlap-divergence
```

Sample:

- full comparison sample:
  `28559, 28550, 28607, 28383, 28463, 5407, 28589, 5380, 5424, 28597, 5448, 5391`
- ao-sky side used the runner path with `workers=3`
- runner-derived region level: `4`
- ao-sky runner elapsed time before live legacy comparisons: `5.949 s`

Accepted divergence scope:

- inner fields intentionally ignored because ao-sky now derives counts from the
  runtime Gaia rows and only persists traceable local winners:
  `star_count`, `ngs_count`, `winner_asterism_id`,
  `winner_distance_arcsec`, `winner_ee_resolved`,
  `winner_ee_averaged`, `coverage_resolved`, and `coverage_averaged`
- boundary-footprint fields accepted under the self-contained two-ring
  footprint rule:
  `asterism_count`, `best_ee`, `best_fwhm`, and `best_sr`

Result:

| Outer Pixel | Legacy Asterisms | ao-sky Asterisms | Membership | Inner Rows | Result |
| ---: | ---: | ---: | --- | ---: | --- |
| `28559` | `324` | `324` | exact | `65536` | OK |
| `28550` | `342` | `342` | exact | `65536` | OK |
| `28607` | `351` | `351` | exact | `65536` | OK |
| `28383` | `336` | `336` | exact | `65536` | OK |
| `28463` | `368` | `368` | exact | `65536` | OK |
| `5407` | `306` | `306` | exact | `65536` | OK |
| `28589` | `343` | `343` | exact | `65536` | OK |
| `5380` | `340` | `340` | exact | `65536` | OK |
| `5424` | `365` | `365` | exact | `65536` | OK |
| `28597` | `328` | `328` | exact | `65536` | OK |
| `5448` | `371` | `372` | accepted boundary divergence: `1` extra | `65536` | OK |
| `5391` | `356` | `356` | accepted boundary divergence: `1` missing, `1` extra | `65536` | OK |

Retained interpretation:

- the current implementation is still legacy-compatible for the retained
  Traversal contract on this full 12-pixel sample
- the only observed asterism membership differences are the expected edge
  effects from the ao-sky two-ring, self-contained boundary footprint
- every outer pixel produced the same dense inner row domain and exact inner row
  ordering
- this is the known-good legacy-preserving state before Phase 14 changes the
  winner/overlap algorithm more substantially
