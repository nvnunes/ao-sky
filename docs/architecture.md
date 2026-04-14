# ao-sky Architecture

This document is the source of truth for `ao-sky` package boundaries, canonical
data layers, persisted artifact model, build execution model, and public
package/documentation shape.

It describes a mix of implemented and intended architecture. Use
[`docs/plan.md`](docs/plan.md) for the transitional migration context and
phase-level details around that architecture.

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
- derived pass artifacts
- a build engine that materializes, resumes, and inspects those passes

`ao-sky` is a spatial data and build-orchestration package. Gaia storage
layout, pass manifests, restartability, traversal policy, and artifact
provenance are part of the domain model.

## Locked Design Constraints

The following decisions should be treated as fixed inputs to the design:

- Canonical Gaia storage is raw Gaia DR3, not a working-data product.
- Derived working-band magnitudes such as `R` are computed in the loader, not
  persisted.
- The locked canonical Gaia schema is:
  - `gaia_id`
  - `gaia_ra`
  - `gaia_dec`
  - `gaia_G`
  - `gaia_BP`
  - `gaia_RP`
  - `gaia_ref_epoch`
  - `gaia_pmra`
  - `gaia_pmdec`
  - `gaia_non_single_star`
  - `gaia_ruwe`
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
  - loader-time preparation such as proper motion and derived working bands
- `ao_sky.spatial`
  - HEALPix conversions
  - neighbour traversal primitives
  - geometric helpers used across loaders, passes, and build execution
- `ao_sky.asterisms`
  - asterism search
  - geometry
  - overlap logic
  - filtering and scoring seams
- `ao_sky.passes`
  - pass manifests
  - persisted artifact contracts
  - pass layout rules
  - inner, catalog, and aggregate readers/writers
- `ao_sky.build`
  - planning
  - scheduling
  - traversal
  - execution state
  - build runner orchestration
- `ao_sky.survey`
  - survey-extent overlays
  - other survey-scale derived products
- `ao_sky.cli`
  - thin command-line entrypoints over the Python API

## Data Model Layers

The architecture uses distinct data layers with explicit ownership.

### Canonical Gaia Inputs

- Raw Gaia DR3 per-outer-pixel HDF5 files.
- Versioned by Gaia release and outer HEALPix level.
- Shared across all derived passes.
- Stored without derived working bands.
- Loaded through explicit schema-aware readers.

### Derived Pass Artifacts

- Inner-pixel products.
- Asterism catalogs.
- Aggregate map products.
- Survey-extent overlays and other derived summaries.
- Versioned by pass manifest and output layout version, not by ad hoc folder
  naming.

### Read Models And Contracts

- Loaders expose canonical scientific data through explicit APIs rather than ad
  hoc path logic.
- Persisted schema rules and path/version rules live in narrow contract modules.
- Path layout, schema ownership, and derived-field rules are treated as
  user-facing contracts.
 
## Pass Model

`ao-sky` uses an explicit pass model for derived artifacts.

- Gaia store has its own root and stable path contract.
- Derived work happens inside explicit pass directories.
- Every pass has a manifest recording:
  - algorithm/version identity
  - config inputs
  - source Gaia release and level
  - scoring backend identity
  - output layout version
  - creation time and completion state

- Passes are inspectable and comparable without external context.

## Build And Scheduling Model

Build execution is organized around restartable outer-pixel work.

- The unit of work remains the outer pixel.
- Scheduling is computed from unfinished work, not from the ideal full sky.
- Traversal should prefer neighbouring unfinished outer pixels whenever
  possible.
- Work balancing should use star count or another practical work proxy.
- Worker state should record `pending`, `running`, `done`, `excluded`, and
  `failed` explicitly.
- Cache policy should remain simple until neighbour-aware scheduling has been
  benchmarked.

## Scoring And AO-System Policy

AO-system-specific scoring does not define the generic package boundary.

- The generic asterism search and spatial coverage machinery belongs in
  `ao-sky`.
- Scorer backends are pluggable.
- Instrument-specific ranking and science policy stay outside the canonical
  package core.
- Algorithm-specific behavior is reflected in pass identity and explicit scorer
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
- how working bands are derived at load time
- what a pass is
- what artifacts a pass produces
- how restartable build execution works
- how pass inspection and comparison work
