# ao-sky Architecture

This document is the source of truth for `ao-sky` package boundaries, canonical
data layers, persisted artifact model, build execution model, and public
package/documentation shape.

It describes a mix of implemented and intended architecture. Use
[`plan.md`](plan.md) for the transitional migration context and implementation
sequencing around that architecture.

## Shared Guidance

This repo adopts the shared guidance in:
- `astro-agents/guidance/agent-surface.md`
- `astro-agents/guidance/public-python-projects.md`
- `astro-agents/guidance/python-development.md`

Repo-local package boundaries, persisted contracts, artifact rules, and
exceptions in this document remain the source of truth.

## Package Identity

`ao-sky` is a public Python package for Gaia-backed AO-sky mapping, all-sky
asterism search, and derived AO-sky products.

The package is organized around:

- canonical shared Gaia inputs
- derived build artifacts
- a build engine that materializes, resumes, and inspects those builds

`ao-sky` is a spatial data and build-orchestration package. Gaia storage
layout, build metadata, restartability, traversal policy, and artifact
provenance are part of the domain model.

## Locked Design Constraints

The following decisions should be treated as fixed inputs to the design:

- Canonical Gaia storage is raw Gaia DR3, not a working-data product.
- Derived working-band magnitudes, WFS-band proxies, and other AO-system-aware
  photometric products are not persisted in canonical Gaia files.
- Gaia-domain in-memory transforms may derive working quantities such as the
  current empirical `R` estimate used by asterism search, but those quantities
  are not part of the canonical persisted schema.
- The locked canonical Gaia schema is:
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
- Canonical Gaia file format is compressed HDF5.
- The storage unit remains one file per outer pixel.
- The hour/declination/pixel directory structure stays in place.
- Build-performance work should focus first on neighbour-oriented traversal,
  improved scheduling, restart-aware processing of unfinished work, and work
  balancing by star-count proxy.
- A custom multi-tier cache architecture is not an early design target.
- macOS page cache is the default first-level disk cache unless later evidence
  proves otherwise.

## Public Package Boundary

The deliberate public Python package is `ao_sky`.

- Re-export only supported user-facing entrypoints from `ao_sky.__init__`.
- Keep CLI commands thin wrappers over the Python API.
- Keep internal file layout and public import layout intentionally separate.
- Prefer typed request/config objects where they improve clarity.
- Keep `astropy.table.Table` as the initial scientific interchange type unless
  a stronger abstraction becomes necessary.
- Keep legacy-compatibility code outside the canonical package core.

## Canonical Subpackages

The canonical package is split by ownership:

- `ao_sky.gaia`
  - Gaia schema constants
  - archive querying
  - canonical HDF5 storage
  - raw per-outer-pixel Gaia readers
  - Gaia-domain in-memory proper-motion transforms
- `ao_sky.spatial`
  - HEALPix conversions
  - neighbour traversal primitives
  - geometric helpers used across loaders, artifacts, and build execution
- `ao_sky.asterisms`
  - star assembly for asterism search
  - asterism search
  - geometry
  - overlap logic
  - filtering and scoring seams
- `ao_sky.dust`
  - Gaia TGE source loading and build-local dense A0 cache creation
  - mmap-backed local dust-field sampling
  - build-time dust-field injection helpers
- `ao_sky.predict`
  - native traversal runtime records
  - temporary AO-model loading and caching
  - temporary `girmos-aosims` prediction adapter
  - native point and field-mean prediction helpers
- `ao_sky.build`
  - merged build-config loading
  - build metadata/state contracts
  - build layout rules
  - persisted outer-pixel artifact writers/readers
  - planning
  - scheduling
  - traversal
  - execution state
  - build runner orchestration
- `ao_sky.survey`
  - survey-extent overlays
  - other all-sky augmentation layers
- `ao_sky.cli`
  - thin command-line entrypoints over the Python API

## Data Model Layers

The architecture uses distinct data layers with explicit ownership.

### Canonical Gaia Inputs

- Raw Gaia DR3 per-outer-pixel HDF5 files.
- Versioned by Gaia release and outer HEALPix level.
- One shared Gaia summary artifact per Gaia release and outer HEALPix level.
- Shared across all derived builds.
- Stored without derived working bands.
- Loaded through explicit schema-aware readers.
- Stored at `<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5`.
- Stored as one HDF5 dataset named `gaia` using `gzip=9` and `shuffle=True`.
- The shared summary is stored at `<root>/gaia-<release>-hpx<healpix_level>/summary.h5`.

### Derived Build Artifacts

- Inner-pixel products.
- Asterism catalogs.
- Aggregate map products.
- Survey-extent overlays and other derived summaries.
- Stored as HDF5 datasets using Blosc Zstd compression through
  `hdf5plugin`; readers that bypass `ao_sky.build` and use `h5py` directly
  must import `hdf5plugin` before reading these artifacts.
- Versioned by build metadata and output layout version, not by ad hoc folder
  naming.

### Read Models And Contracts

- Loaders expose canonical scientific data through explicit APIs rather than ad
  hoc path logic.
- Gaia proper motion is an in-memory Gaia transform, not part of the persisted
  raw-store contract.
- Asterism-side neighbour stitching is an in-memory search-preparation step,
  not part of canonical Gaia storage.
- Persisted schema rules and path/version rules live in narrow contract modules.
- Path layout, schema ownership, and derived-field rules are treated as
  user-facing contracts.
 
## Build Model

`ao-sky` uses an explicit build model for derived artifacts.

- Gaia store has its own root and stable path contract.
- Gaia summary is a Gaia-side artifact, not a build-side artifact.
- Dust store has its own root and stable path contract.
- Derived work happens inside explicit build directories.
- Every build has:
  - a root `build.h5` control file for metadata and outer-pixel state
  - a root `build.log`
  - a root `build.yaml` native Traversal runtime config copied at `init`
  - one `outer.h5` artifact container per processed outer pixel
  - one `maps-hpx<level>.h5` all-sky map artifact per aggregated level once
    the build reaches `aggregation`
  - one `survey_extent` dataset inside each `maps-hpx<level>.h5` file once a
    build reaches `augmentation`, when survey overlays are configured
  - a `models/` snapshot directory created at `init`
  - a `surveys/` snapshot directory created at `init` for builds with survey
    overlays
- Build versions are named `v<N>` under the lineage workspace.
- `build.h5` stores:
  - the original merged build-config YAML
  - normalized build metadata
  - the build-local runtime-config path and source-provenance path
  - the full-sky outer-pixel state table for the configured outer level
- normalized build metadata includes the resolved `gaia_root`, build-local
  `dust_root`, `build_root`, and `model_root`, plus build-config fields such as
  `maps.max_level`; build-local snapshot paths such as `dust`, `models`, and
  `surveys/...` are persisted relative to the build root
- Per-outer-pixel build artifacts live under:
  - `hpx<outer-level>-<inner-level>/<hour>h/<sign><deg>/<outer_pix>/outer.h5`
- Initialized builds are inspectable and comparable without external model
  context.
- Initialized builds with survey overlays can augment survey extents without
  external survey-file context.
- Native build execution reads AO-system, traversal, and prediction policy
  from `build.yaml`. Legacy YAML import, when needed for comparison
  work, happens outside the build API before `init`.

## Build And Scheduling Model

Build execution is organized around restartable outer-pixel work.

- The unit of work remains the outer pixel.
- Scheduling is computed from unfinished work, not from the ideal full sky.
- Traversal should prefer neighbouring unfinished outer pixels whenever
  possible.
- Work balancing should use star count or another practical work proxy.
- Traversal uses a long-lived worker runtime even when `workers=1`, and can run
  with multiple regional process workers as an execution-time option; worker
  count, regional scheduling, and Gaia memory-cache settings are not part of
  the persisted build contract. The optional worker memory limit is also an
  execution-time guard: it fails the run cleanly if a worker reports peak RSS
  above the configured limit.
- Canonical Gaia files remain raw on disk, but long-lived Traversal workers
  use a runtime-local Gaia cache whose rows are shifted to the build epoch,
  enriched with `R` and `hpx14`, and then marked read-only. Downstream
  native Traversal logic expects this runtime row contract and derives
  lower-level pixel assignments from `hpx14` instead of recomputing sky
  projections. Raw-row compatibility remains at the lower-level asterism
  loader boundary, not in the build Traversal hot path.
- Retained local asterisms and rich inner Traversal products are derived from
  each outer pixel's two-ring expanded Gaia-star footprint; they do not depend
  on neighbouring pixels' derived asterism artifacts. The expansion is applied
  before build-epoch proper-motion shifting, so stars that start just outside
  the raw outer-pixel boundary can still be included if epoch shifting moves
  them into the relevant edge footprint.
- The two-ring asterism boundary is an internal fixed rule rather than a
  runtime setting; it is the minimal integer-ring margin used to avoid the
  insufficient one-ring self-contained boundary while providing a small
  proper-motion buffer without adding another user-facing knob.
- Cache-aware Traversal groups outer pixels by coarse HEALPix regions and keeps
  Gaia table caches worker-local so build state remains parent-owned.
- Build creation depends on a shared Gaia summary for the configured Gaia
  release and outer level.
- Build state should record:
  - the current build phase
  - explicit per-phase outer-pixel status for `gaia_loading` and `traversal`
  - per-phase attempt count and last error message
- Aggregation is build-global rather than outer-pixel-local.
- A normal build run may advance from `traversal` into `aggregation`
  automatically once all outer-pixel Traversal work is complete.
- When survey overlays are configured, a normal build run may also advance from
  `aggregation` into `augmentation` automatically.
- Cache policy should remain simple until neighbour-aware scheduling has been
  benchmarked.

## Scoring And AO-System Policy

AO-system-specific scoring does not define the generic package boundary.

- The generic asterism search and spatial coverage machinery belongs in
  `ao-sky`.
- Scorer backends are pluggable.
- Instrument-specific ranking and science policy stay outside the canonical
  package core.
- Algorithm-specific behavior is reflected in build identity and explicit scorer
  configuration, not in hidden global behavior.
- Build execution receives native `ao-sky` runtime configuration only. Legacy
  `survey_tools` YAML import, when needed for compatibility checks, belongs in
  comparison tooling outside the build API.

## Public Surface

`ao-sky` presents itself as a public Python library rather than a private script
repository.

### Public API Style

- Package-root exports for the primary entrypoints only.
- Explicit contract modules for persisted schemas and path/version rules.
- Typed request objects where inputs form a coherent family.
- Read models that hide storage details where possible.
- Minimal hidden globals and minimal path inference.

### CLI Surface

- CLI is a thin wrapper over the Python API.
- CLI verbs are deterministic and composable.
- CLI behavior should not depend on implicit working-directory conventions.

### Documentation Surface

- `README.md` provides the public summary.
- `docs/architecture.md` is the stable design source of truth.
- `docs/plan.md` holds transitional migration context.
- `docs/testing.md` is the source of truth for verification expectations.
- API, CLI, development, and reference docs explain supported workflows and
  interfaces as the repo grows.

The documentation surface explains:

- how Gaia storage is laid out
- how canonical Gaia inputs differ from later derived photometric layers
- what a build is
- what artifacts a build produces
- how restartable build execution works
- how build inspection and comparison work
