# Python API Documentation

This document describes the current code-first API exposed by `ao_sky`.

The supported public surface currently centers on:

- `ao_sky.gaia` for canonical Gaia storage and proper-motion transforms
- `ao_sky.dust` for Gaia TGE source validation and build-local A0 cache helpers
- `ao_sky.spatial` for reusable non-plotting HEALPix helpers
- `ao_sky.asterisms` for outer-pixel star assembly and in-memory search
- `ao_sky.predict` for native prediction runtime records and array prediction helpers
- `ao_sky.build` for persisted build roots, build state, and per-outer-pixel
  derived artifacts
- `ao_sky.artifacts` for read-only access to existing build artifacts

## Current Public Surface

### `ao_sky.gaia`

The current package-supported Gaia API exposes:

- `GaiaStoreConfig`
- `GaiaHealpixStore`
- `GaiaSummaryStore`
- `GaiaError`
- `fetch_gaia_store`
- `apply_proper_motion`
- `compute_r_magnitude`
- `GAIA_SCHEMA_COLUMNS`
- `HDF5_DATASET_NAME`
- `GAIA_SUMMARY_DATASET_NAME`
- `GAIA_SUMMARY_DTYPE`
- `GAIA_SUMMARY_FILENAME`
- `GaiaTableCacheStats`
- `HDF5_COMPRESSION` (`"blosc-zstd"`)
- `HDF5_COMPRESSION_OPTS` (`5`)
- `HDF5_SHUFFLE` (`True`)

The canonical raw Gaia schema is:

- `source_id`
- `ra`
- `dec`
- `G`
- `BP`
- `RP`
- `ref_epoch`
- `pmra`
- `pmdec`
- `non_single_star`
- `ruwe`

The canonical raw Gaia path contract is:

`<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5`

The shared Gaia summary path contract is:

`<root>/gaia-<release>-hpx<healpix_level>/summary.h5`

The canonical stored dataset is:

- HDF5 dataset name: `gaia`
- codec: Blosc Zstd
- Blosc compression level: `5`
- Blosc shuffle: enabled
- Python dependency: `hdf5plugin`

The shared Gaia summary dataset is:

- HDF5 dataset name: `summary`
- one dense row per outer pixel
- fields: `outer_pix`, `star_count`, `loaded`
- stored with the same Blosc Zstd level 5 HDF5 codec as per-pixel Gaia files

### `ao_sky.dust`

The current package-supported dust API exposes:

- `DustError`
- `fetch_gaia_tge_dataset`
- `gaia_tge_map_filename`
- `gaia_tge_a0_cache_filename`
- `prepare_gaia_tge_a0_cache`
- `sample_gaia_a0_for_outer_pixel`
- `add_gaia_a0_to_inner`

The source Gaia TGE path contract is:

`<gaia_root>/gaia_tge/TotalGalacticExtinctionMap_001.csv.gz`

Initialized builds also persist a build-local dense A0 cache:

`<build>/dust/ao-sky-gaia-tge-a0-hpx<max_level>.npy`

Traversal and aggregation sample this cache with `numpy.load(...,
mmap_mode="r")`, using the same `max_data_level` coarse-sampling rule as the
legacy pipeline.

### Repo-Owned HDF5 Compression

Repo-owned HDF5 files use one compression contract for new writes:

- codec: Blosc Zstd through `hdf5plugin`
- Blosc compression level: `5`
- Blosc shuffle: enabled
- Python dependency: `hdf5plugin`

This applies to canonical Gaia `gaia.h5` files, Gaia `summary.h5` files, and
derived build artifacts such as per-outer-pixel `outer.h5`,
`maps-hpx<level>.h5`, and augmentation datasets inside map files. Existing
legacy `gzip=9` Gaia files remain readable. Code that reads HDF5 files through
`ao_sky` readers gets HDF5 plugin registration automatically. Code that opens
repo-owned HDF5 files directly with `h5py` should import `hdf5plugin` before
reading plugin-compressed datasets.

### `ao_sky.spatial`

The current package-supported spatial API exposes:

- `SpatialError`
- `get_pixel_resolution`
- `get_pixel_area`
- `get_pixel_from_skycoord`
- `get_pixel_skycoord`
- `get_parent_pixel`
- `get_pixel_neighbours`
- `get_subpixels`
- `get_subpixel_indexes`

### `ao_sky.asterisms`

The current package-supported asterism API exposes:

- `AsterismError`
- `AsterismSearchOptions`
- `AsterismSearchProfile`
- `load_asterism_stars`
- `find_asterisms`
- `ASTERISM_TABLE_COLUMNS`

### `ao_sky.predict`

The current package-supported prediction API exposes:

- `AOSystemRuntime`
- `PredictRuntime`
- `PointPredictionBatch`
- `SeeingBaselinePerformance`
- `PredictError`
- `configure_inference_threads`
- `warm_model_cache`
- `clear_backend_cache`
- `get_point_model`
- `get_mean_model`
- `predict_point_arrays`
- `predict_field_mean_arrays`
- `get_seeing_baseline_performance`

These helpers are primarily the build Traversal prediction boundary. Direct
callers should pass homogeneous, magnitude-ordered NGS arrays whose second
dimension matches the requested star count and should obtain models through
`get_point_model` or `get_mean_model`. Model loading defaults to CPU unless the
caller passes `device="auto"` explicitly. The current implementation still uses
the temporary `girmos-aosims` backend adapter behind this native API.

### `ao_sky.build`

The current package-supported build API exposes:

- `BuildError`
- `check_runtime_roots`
- `fetch_gaia_data`
- `init_build`
- `load_build_definition`
- `resolve_build_root_only`
- `resolve_gaia_root_only`
- `resolve_build_roots`
- `restart_build`
- `run_build`
- `show_build`

### `ao_sky.artifacts`

The current package-supported artifact-reader API exposes:

- `AoSkyArtifactStore`
- `MapData`

`AoSkyArtifactStore` is a read-only facade over existing build artifacts. It
loads build-local runtime configuration, dense map fields, per-outer `inner`
and `asterisms` datasets, and pinned Gaia tables. It does not run build steps,
materialize missing artifacts, or provide paper-specific compatibility columns.

## Core Read And Search Paths

### `GaiaHealpixStore.healpix_filename(outer_pix: int) -> Path`

Return the canonical HDF5 filename for one outer nested HEALPix pixel.

### `GaiaHealpixStore.load_healpix(outer_pix: int, *, force_reload: bool = False, read_only: bool = False) -> Table`

Load one raw Gaia table for an outer pixel.

Behavior:

- if the canonical HDF5 file already exists and `force_reload=False`, read it
- otherwise query the Gaia archive seam, write the canonical file, then return
  the canonical table
- when `read_only=True`, mark existing table columns as non-writeable; callers
  that need to derive or mutate working columns should copy first

### `GaiaSummaryStore.summary_filename() -> Path`

Return the shared Gaia summary filename for one Gaia release and outer level.

### `GaiaSummaryStore.load_summary() -> Table`

Load the shared dense Gaia summary table for one Gaia release and outer level.

### `fetch_gaia_store(config, *, force_reload=False, output=None) -> Path`

Load the full-sky canonical Gaia store for one Gaia release and outer level,
then refresh the shared Gaia summary from the final store state.

Behavior:

- ensure the dense shared `summary.h5` file exists before the full-sky pass
- skip existing Gaia files unless `force_reload=True`
- after each outer-pixel step, count the final Gaia table and update the
  corresponding summary row immediately
- leave partial summary progress behind if the run is interrupted or fails
- when `output` is provided, emit legacy-style start, progress, and done lines

### `apply_proper_motion(table, *, epoch=None, dt_years=None) -> Table`

Return a canonical Gaia table with in-memory proper motion applied. The schema
is preserved exactly, and the returned table updates `ra`, `dec`, and
`ref_epoch`.

### `compute_r_magnitude(table) -> ndarray`

Return the empirical Gaia-to-`R` magnitude estimate used by the legacy guide-star
workflow. This is an in-memory Gaia-domain helper and is not part of the raw
canonical HDF5 contract.

### `load_asterism_stars(store, outer_pix, *, neighbour_level=None, boundary_rings=2, epoch=None, dt_years=None, include_locality=False) -> Table`

Return the Gaia rows needed to search one outer pixel for asterisms.

Behavior:

- load the local outer pixel fully
- when `neighbour_level` is set, include only the border-trimmed neighbour rows
  from `boundary_rings` fine-HEALPix rings around the target outer pixel
- optionally apply Gaia proper motion in-memory
- the default two-ring boundary provides edge completeness headroom before
  proper-motion shifting, so stars just outside the raw boundary can still
  contribute after epoch shifting
- add the legacy empirical `R` magnitude used by the current asterism search
- return Gaia-schema rows, optionally enriched with locality columns

### `find_asterisms(stars, options) -> Table`

Run the legacy-faithful in-memory asterism search over a prepared Gaia star
table.

Behavior:

- search brightness is fixed to the legacy empirical Gaia-derived `R`
- supported star-count range is 1-3
- output is a clean flat table with fixed `star1_*`/`star2_*`/`star3_*` slots
- the table carries the temporary full legacy-comparison field set

Non-goals of the current Gaia API:

- instrument-specific photometric proxies
- repo-root config discovery

## Build Lifecycle

### `ao-sky.yaml` Discovery

Helpers that accept `aosky_yaml` use a narrow discovery order when the argument
is omitted:

- `./ao-sky.yaml` in the current working directory
- `ao-sky.yaml` in the nearest parent Python project root identified by
  `pyproject.toml`
- no configuration defaults

Explicit function arguments and CLI flags still take precedence over values
loaded from `ao-sky.yaml`.

### `check_runtime_roots(...) -> tuple[bool, str]`

Resolve and validate the runtime root set without requiring a specific build.

Behavior:

- resolve `gaia_root`, `build_root`, `dust_root`, and `model_root` from explicit
  arguments or `ao-sky.yaml`; `dust_root` is derived from `gaia_root`
- require Gaia, build, and model roots to exist as directories
- require the dust root to contain the Gaia TGE dataset file
- return a success flag and concise human-readable report
- perform no downloads, build inspection, or persisted build mutation

### `fetch_gaia_data(*, gaia_root, gaia_release, outer_level, force=False, aosky_yaml=None, cwd=None, output=None) -> Path`

Resolve the Gaia root, ensure Gaia TGE dust is installed there, and run the
shared full-sky Gaia loading workflow.

Behavior:

- install or validate the Gaia TGE dataset under `<gaia_root>/gaia_tge`
- load one explicit Gaia release and outer level into the canonical Gaia store
- skip existing per-pixel Gaia files unless `force=True`
- refresh the shared dense `summary.h5` artifact as the run progresses
- return the written summary filename
- do not create build-local dust caches; that is a build-aware `init` step

### `init_build(...) -> Path`

Create a new build root from one merged build config, persist `build.h5`, seed
full-sky outer-pixel state, and create the build artifact layout.

Behavior:

- load a merged build-config YAML with `gaia.release`, `maps.max_level`,
  runtime policy sections, and optional `survey_overlays`
- require the matching shared Gaia summary to exist for the configured Gaia
  release and outer level
- resolve `gaia_root`, `build_root`, and `model_root` from explicit arguments
  or `ao-sky.yaml`
- convert the source Gaia TGE CSV under `gaia_root` into a build-local dense
  A0 cache at `<build>/dust/ao-sky-gaia-tge-a0-hpx<max_level>.npy`
  and persist `dust_root` as that build-local dust snapshot
- persist the resolved roots into `build.h5`
- copy the supplied native runtime policy into build-local `build.yaml`; later
  build execution reads that build-local config
- copy required `.pt` and `_metadata.pkl` files for configured resolved and
  averaged models into `<build>/models`
- write `<build>/models/manifest.json` and update persisted `model_root` to the
  build-local snapshot
- when `survey_overlays` are configured, resolve each `moc_files` entry as a
  filename under `survey_root`, copy it into `<build>/surveys`, write
  `<build>/surveys/manifest.json`, and persist build-root-relative
  `surveys/<filename>` paths for augmentation

### `run_build(build_path, *, workers=None, scheduler=None, low_latitude_workers=None, gaia_cache_entries=None, gaia_cache_mb=None, parent_memory_limit_mb=None, telemetry=None, aosky_yaml=None) -> Path`

Run one initialized build through its unfinished outer-pixel work and write
`outer.h5` artifact containers.

`workers` controls Traversal execution parallelism at execution time. When it
is `None`, `ao-sky.yaml` may provide `build.workers`; otherwise the fallback is
`1`. `scheduler` selects the multi-worker Traversal scheduler. `"static"` uses
the legacy regional Galactic-latitude split and accepts `low_latitude_workers`.
`"dynamic"` uses parent-owned RAM-aware batches and rejects
`low_latitude_workers`; workers request batches at runtime while keeping model
and Gaia caches warm. The single-worker fallback uses a long-lived in-process
Traversal worker so runtime setup, model caches, and Gaia table caches are
reused across outer pixels.
Multi-worker runs use long-lived process workers. Cache-aware
Traversal uses a worker-local runtime Gaia table cache controlled by
`gaia_cache_entries` and `gaia_cache_mb`; set either to `0` only for cache-off
benchmarks. Read prewarm and artifact write-behind were measured and rejected
as active execution paths because they added complexity without enough benefit
over regional traversal plus the prepared-table cache. Artifact writes remain
synchronous and worker-owned. The regional worker-assignment level is derived
internally from `outer_level` and `workers`, so it is not a public runtime
setting. Cached runtime Gaia rows are derived from raw canonical Gaia files,
shifted to the build epoch, enriched with `R` and `hpx14`, and marked
read-only. These settings and derived columns are runtime execution details
only; they are not persisted in the build definition, build metadata, or
canonical Gaia files. Native build Traversal expects these runtime rows for
inner counts, asterism search, and prediction.

`parent_memory_limit_mb` is the Python/CLI override for the aggregate total-RAM
safety guard configured in YAML as `build.memory_limit_mb`. A value of `0` or
`None` disables it. When enabled, the parent process periodically checks its
current RSS plus active worker current RSS, with an additional GPU-driver reserve
when GPU prediction is enabled. As total RAM approaches the limit, the parent
targets the heaviest active workers for cache trimming and brief backoff; if the
aggregate still exceeds the limit, Traversal stops with a restartable failure.
This is the only active memory-limit guard.

`telemetry` controls runtime diagnostics. The default `None`/`"basic"` keeps
low-cost operational profiling in `build.log`. `"detailed"` additionally writes
one per-outer-pixel diagnostic row to
`<build>/diagnostics/traversal-memory.csv` with RSS checkpoints and key
intermediate row counts. Detailed telemetry is for benchmark/debug runs and is
not part of the persisted build contract.

Artifact writes are direct and worker-owned. SSD artifact staging was benchmarked
and rejected as an active runtime option after Blosc Zstd made write latency
negligible relative to Traversal compute.

### `restart_build(..., workers=None, scheduler=None, low_latitude_workers=None, gaia_cache_entries=None, gaia_cache_mb=None, parent_memory_limit_mb=None, telemetry=None) -> Path`

Resume the latest `v<N>` build under the lineage workspace, using the same
execution-time worker-count contract as `run_build`.

### `show_build(build_path) -> str`

Return a human-readable build summary from the persisted `build.h5` state.

## Working Example

```python
from pathlib import Path

from ao_sky.asterisms import AsterismSearchOptions, find_asterisms, load_asterism_stars
from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig

config = GaiaStoreConfig(
    root=Path("/data/gaia"),
    release="dr3",
    healpix_level=6,
)
store = GaiaHealpixStore(config)

filename = store.healpix_filename(outer_pix=0)
stars = load_asterism_stars(store, outer_pix=0, neighbour_level=7, epoch=2016.0)
asterisms = find_asterisms(stars, AsterismSearchOptions(min_stars=1, max_stars=3))

print(filename)
print(stars.colnames)
print(asterisms.colnames)
print(len(asterisms))
```

## Reference

Use the generated reference page for the complete public module surface:

- [Gaia API Reference](reference/gaia/api.md)
- [Spatial API Reference](reference/spatial/api.md)
- [Asterisms API Reference](reference/asterisms/api.md)
- [Build API Reference](reference/build/api.md)
