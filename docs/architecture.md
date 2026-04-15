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
  - Gaia TGE dust loading
  - local dust-field sampling
  - build-time dust-field injection helpers
- `ao_sky.build`
  - build-definition loading
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
- Shared across all derived builds.
- Stored without derived working bands.
- Loaded through explicit schema-aware readers.
- Stored at `<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5`.
- Stored as one HDF5 dataset named `gaia` using `gzip=9` and `shuffle=True`.

### Derived Build Artifacts

- Inner-pixel products.
- Asterism catalogs.
- Aggregate map products.
- Survey-extent overlays and other derived summaries.
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
- Dust store has its own root and stable path contract.
- Derived work happens inside explicit build directories.
- Every build has:
  - a root `build.h5` control file for metadata and outer-pixel state
  - a root `build.log`
  - one `outer.h5` artifact container per processed outer pixel
  - one `maps-hpx<level>.h5` all-sky map artifact per aggregated level once
    the build reaches `aggregation`
  - one `survey_extent` dataset inside each `maps-hpx<level>.h5` file once a
    build reaches `augmentation`, when survey overlays are configured
- Builds are named `<ao-system-short-name>-<config-short-name>-v<N>`.
- `build.h5` stores:
  - the original build-definition YAML
  - normalized build metadata
  - the full-sky outer-pixel state table for the configured outer level
- normalized build metadata includes the resolved `gaia_root`, `build_root`,
  and `dust_root`, plus build-definition fields such as `max_data_level`
- Per-outer-pixel build artifacts live under:
  - `hpx<outer-level>-<inner-level>/<hour>h/<sign><deg>/<outer_pix>/outer.h5`
- Builds are inspectable and comparable without external context.

## Build And Scheduling Model

Build execution is organized around restartable outer-pixel work.

- The unit of work remains the outer pixel.
- Scheduling is computed from unfinished work, not from the ideal full sky.
- Traversal should prefer neighbouring unfinished outer pixels whenever
  possible.
- Work balancing should use star count or another practical work proxy.
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
