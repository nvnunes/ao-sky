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
- The current storage situation re-opens the case for a bounded local
  pre-warming cache, but that should still be treated as an execution-path
  question rather than a package-boundary question.

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
- The historical evidence supported staging away from HDD before considering a
  more complex local-cache design.

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

## Current Benchmark Refresh Targets

The current storage situation is different enough that some benchmark work
should be repeated before cache policy is treated as closed.

The most useful refresh targets are:

- cold-read and warm-read timing for Gaia files under the current source
  storage path
- end-to-end timing for a representative single outer pixel
- neighbour-heavy clustered timing over a small batch
- the cost and overlap behavior of pre-warming Gaia and inner files into a
  bounded local cache while workers continue processing
- throughput behavior with different worker counts once staging is active

Each refreshed benchmark should record:

- source storage location
- local cache target, if any
- whether staging is full, bounded, or on-demand pre-warming
- benchmark region or outer-pixel list
- worker count
- whether timings are cold, warm, or mixed
