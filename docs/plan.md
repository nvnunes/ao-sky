# ao-sky Plan

## Purpose

This document holds the transitional context for the repository:

- why `ao-sky` exists now
- how it relates to `survey_tools`
- what should be re-implemented first
- how the migration should proceed
- how compatibility should eventually hand back through `survey_tools`

`docs/architecture.md` should stay focused on what `ao-sky` is as a package and
system design. This file is the planning and migration layer around that
architecture.

This repository is now also the active planning source of truth for `ao-sky`.
Historical planning, compatibility, and benchmark material from the legacy
implementation and downstream migration work has been absorbed here or into the
sibling docs in this repository. `ao-sky` planning should remain readable and
actionable without depending on planning documents in other repositories.

## Documentation Boundary

- Phase labels and other transitional sequencing language belong in this
  document.
- Source-of-truth docs such as `README.md`, `AGENTS.md`,
  `docs/architecture.md`, `docs/development.md`, `docs/testing.md`,
  `docs/benchmarking.md`, package metadata, and shipped CLI/API text should
  describe current state, supported workflows, and stable boundaries without
  phase language.
- When another document needs to point at future work, link to
  `docs/plan.md` rather than importing its phase terminology into that
  document.

## Current Situation

`survey_tools` should now be treated as legacy and compatibility
infrastructure, not as the place for major AO-sky modernization.

This repository is still in its bootstrap stage. The current repo surface is
mostly the design documents that define the package boundary and migration
direction.

The reason this repository exists is to create a clean upstream home for the
reusable AO-sky core without forcing that work to remain shaped by the old
`survey_tools` package layout, old config assumptions, or old downstream
compatibility pressure.

The legacy implementation still carries the old `aomap` shape, monolithic
workflow structure, and shared-input/derived-output coupling through
`config.folder`.

Within `survey_tools`, branch roles should be understood clearly:

- `develop` is the legacy implementation and the main behavioral baseline
- `codex/pre-ao-sky-phase-1` contains preliminary AO-sky-oriented exploratory
  work, but it is not the canonical architecture source for this repository

The migration should preserve algorithmic effect by default unless an explicit
decision is made to change it. It should not preserve the existing
`survey_tools` implementation structure for its own sake.

In particular:

- the goal is not a line-by-line or module-by-module transplant
- code organization can change substantially
- internal interfaces can change substantially
- implementation details can change substantially
- the new code should follow the cleaner package and ownership boundaries being
  established in `ao-sky`

Behavioral continuity matters. Legacy code organization does not.

`survey_tools` is the reference implementation and behavioral baseline during
this work, with `develop` as the default legacy baseline unless an explicit
decision says otherwise. `ao-sky` should re-implement the needed capabilities
here rather than carrying forward the old module structure.

## Migration Direction

The migration direction is intentionally not:

- add a thin `ao-sky` wrapper over `survey_tools`
- keep new architecture work inside `survey_tools`
- freeze the old `aomap` package shape as the long-term public interface

The migration direction is:

1. use `survey_tools` as the reference implementation and behavioral baseline
2. re-implement the reusable AO-sky core inside `ao-sky`
3. keep `survey_tools` available as the legacy reference implementation and
   compatibility baseline during migration
4. later let `survey_tools` depend on `ao-sky`
5. only then delete the superseded legacy implementation

This is the same general upstream/downstream pattern already used for
`ao-predict`: the new repo should own the reusable core, while legacy and
project-specific compatibility remain downstream until the new public surface is
stable.

## Broader Migration Intent

The larger repository split should keep following the same ownership rule:

- reusable generic AO-sky capability moves upward into `ao-sky`
- thesis-specific or project-specific workflow and science policy stays
  downstream
- dependency direction should converge toward `ao-sky` feeding downstream
  projects, not the reverse
- `survey_tools` remains transitional compatibility infrastructure during this
  handoff, not the long-term home of the reusable core

## Boundary During Transition

### Remains Legacy In survey_tools

- the transitional `aomap` package shape
- legacy reader contracts used by pinned downstream assets
- compatibility loaders for:
  - `inner.fits`
  - `asterisms-*.fits`
  - `data-hpx*.fits`
- existing notebooks, ad hoc scripts, and legacy config habits
- temporary downstream bridges used by `girmos-aosims` and paper figure code

### Becomes Canonical In ao-sky

- Gaia archive querying and canonical raw storage
- loader-time star-table preparation
- HEALPix and neighbour traversal primitives
- asterism search, geometric filtering, and overlap handling
- pass manifests and derived-artifact layout
- restart-aware build state and scheduling
- public Python API, CLI, docs, and examples

## What To Re-Implement First From survey_tools

The first re-implemented code should be the reusable scientific core, not the
entire legacy repo.

### Re-Implement First

1. `survey_tools/survey_tools/gaia.py`
   - Re-implement first, but split immediately into archive querying,
     canonical HDF5 storage, and loader-time preparation.
   - Keep the canonical Gaia schema and HDF5 behavior.
   - Do not carry legacy FITS cache path logic into the new core.

2. Non-plotting HEALPix helpers from `survey_tools/survey_tools/healpix.py`
   - Re-implement only the spatial primitives:
     - level and resolution helpers
     - parent and subpixel helpers
     - neighbour lookup
     - `SkyCoord` conversions

3. `survey_tools/survey_tools/asterism.py`
   - Re-implement the asterism search and geometry logic.
   - Split search, geometry, and overlap utilities into clearer modules.

4. Narrow build and path concepts from `survey_tools/aomap/aomap.py`
   - Re-implement only the parts that define the real AO-sky boundary:
     - config/path resolution ideas
     - outer/inner/asterism layout concepts
     - the Gaia loader seam
     - restart-aware work-state concepts
   - Do not carry the monolithic `aomap.py` structure forward.

5. The tests that lock the intended Gaia behavior
   - `survey_tools/tests/test_gaia_store.py`
   - the Gaia-loader integration pieces from
     `survey_tools/tests/test_aomap_gaia_integration.py`

### Re-Implement Later, If Needed

- `load_asterisms(...)`-style readers for derived asterism products
- `get_map_data(...)`-style aggregate readers
- survey-extent helpers
- plotting helpers

### Do Not Carry Forward Into ao-sky

These areas are not part of `ao-sky`'s intended scope. They should not be
re-implemented here and should not shape the package boundary for this
repository.

- `survey_tools.catalog`
- `survey_tools.sky`
- `survey_tools.targets`
- broad notebook- or script-driven workflow code
- the legacy top-level `aomap` compatibility package structure

## Implementation Phases

The phase order should optimize for long-term design clarity, not for minimum
wrapper code.

### Phase 0: Public Package Baseline

- Add `pyproject.toml` with public package metadata, dependency structure, and
  entrypoints appropriate for an installable Python package.
- Add `LICENSE`.
- Create a local Conda environment for repo development and verification.
- Establish the `src/ao_sky` package root and minimal package exports.
- Establish the minimum public document set for the repo surface:
  - `README.md`
  - `AGENTS.md`
  - `docs/architecture.md`
  - `docs/development.md`
  - `docs/benchmarking.md`
  - `docs/testing.md`
- Populate the bootstrap-stage workflow docs with exact repo-local commands:
  - `docs/development.md` should record the canonical environment creation,
    activation, install, and daily workflow commands.
  - `docs/testing.md` should record the canonical verification commands and
    completion expectations for the repo surface established in Phase 0.
- Add any minimal supporting repo metadata or docs-site configuration needed
  for `ao-sky` to present itself as a public Python package from the start.

### Phase 1: Lock The Gaia Boundary

- Implement the canonical Gaia schema constants and path contract.
- Re-implement the HDF5 store and raw per-outer-pixel loader behavior using
  `survey_tools` as the reference.
- Keep the Gaia boundary band-agnostic and free of instrument-specific derived
  photometry.
- Keep Phase 1 focused on canonical Gaia tables only:
  - no derived bands or photon-rate proxies
  - no proper-motion application
  - no neighbour stitching
  - no scratch cache tier
- Add an explicit Gaia store configuration object for:
  - storage root
  - Gaia release
  - outer HEALPix level
- Do not add implicit Gaia-root discovery or repo-root config-file lookup
  inside `ao_sky.gaia`.
- If a supported `aosky.conf` is added later, treat it as a CLI or
  application-layer convenience that constructs explicit store configuration
  rather than as hidden package-global behavior.
- Re-implement and modernize the Gaia store tests first.

### Phase 2: Re-Implement Spatial And Asterism Core

- Re-implement the non-plotting HEALPix helpers.
- Re-implement and split the asterism search code.
- Build a clean star-loading path for asterism construction around the Phase 1
  Gaia loader, including proper-motion application and neighbour-aware views as
  needed there.
- Use the legacy empirical Gaia-derived `R` estimate for the current in-memory
  asterism search path, implemented as a Gaia-domain in-memory helper rather
  than as persisted canonical Gaia data.
- Define the canonical in-memory asterism catalog contract and carry the full
  legacy-comparison field set for early validation, even though some of those
  fields are expected to be dropped later.

### Phase 3: Define The Derived Pass Model

- Replace `config.folder` coupling with explicit pass manifests and layouts.
- Select and define the new persisted asterism data format rather than
  assuming legacy FITS compatibility.
- Implement pass-aware paths for:
  - inner products
  - asterism catalogs
  - aggregate products
  - survey-extent overlays
- Define pass metadata and restart state.

### Phase 4: Rebuild The Execution Engine

- Implement restart-aware planning over unfinished outer pixels.
- Replace chunk-barrier scheduling with a more flexible scheduler.
- Add neighbour-oriented traversal and star-count balancing.
- Add a pre-warming local-cache scheme that can stage Gaia and inner files into
  local storage in parallel with ongoing processing.
- If a supported repo-root `aosky.conf` is added, define it here as a CLI or
  application-layer configuration surface rather than as hidden package-global
  behavior inside `ao_sky.gaia`.
- Benchmark against the recorded baseline in `docs/benchmarking.md` and refresh
  it under the current storage setup before considering cache complexity.

### Phase 5: Compatibility Adoption In survey_tools

- Add thin `survey_tools` adapters that call `ao-sky` public APIs.
- Keep pinned legacy assets readable through explicit compatibility paths.
- Repoint downstream consumers incrementally.
- Avoid deleting legacy code until the new path is documented, tested, and used
  in practice.

### Phase 6: Define The WFS Photometric Proxy Layer

- Keep canonical Gaia storage raw and free of derived bands, fluxes, and other
  AO-system-specific photometric products.
- Continue using the current legacy empirical Gaia-derived `R` estimate as the
  interim asterism-search input until this phase replaces it.
- Define AO-system-aware derived photometry on top of loaded Gaia stars rather
  than inside the canonical Gaia store.
- Support transient sensing-band outputs, including photon-rate proxies and
  magnitude-like proxies where useful to downstream consumers.
- Decide how Gaia XP synthetic photometry, empirical Gaia-to-band transforms,
  and flux-space combinations fit behind one interface.
- Define uncertainty handling and the downstream contract for asterism-building
  and AO-simulation consumers.

### Phase 7: Deduplication And Final Handoff

- Remove the superseded legacy implementation from `survey_tools`.
- Retain only the compatibility surface that is still worth carrying.
- Decide whether remaining legacy asset readers stay in `survey_tools` or move
  to a dedicated compatibility namespace.

## Compatibility Strategy

The compatibility plan should be explicit from the start.

### Early Migration

- `survey_tools` remains the compatibility baseline.
- `ao-sky` should not depend on `survey_tools`.
- Downstream code keeps using `survey_tools` until `ao-sky` can replace a
  stable slice cleanly.
- Compatibility is required at the level of pinned derived AO-sky assets:
  - `inner.fits`
  - `asterisms-*.fits`
  - `data-hpx*.fits`
- Preserving every legacy reader signature is not required if those pinned
  assets remain readable through an explicit compatibility path.

### Middle Migration

- Add thin `survey_tools -> ao_sky` adapters for selected new paths.
- Keep pinned asset readers in `survey_tools`.
- Prefer explicit new entrypoints over hidden import-time substitution.

### Late Migration

- Once `ao-sky` is stable, let `survey_tools` depend on it.
- Re-export or wrap the subset of `ao-sky` needed for backwards compatibility.
- Deprecate legacy reader and builder entrypoints deliberately.

### Important Constraint

Compatibility should be provided at the public-contract level, not by forcing
`ao-sky` to retain `survey_tools`' internal monolith structure. The stable
surface is the compatibility target, not the old implementation shape.

## ao-predict As A Reference

`ao-predict` is a good model for:

- `src/`-based packaging
- deliberate package-root exports
- strict docs builds
- thin CLI design
- bootstrap and development ergonomics
- writing architecture down early
- a public Python documentation surface that is broader than `README.md` alone

`ao-sky` should still differ where its domain requires it:

- it is a spatial data and build-orchestration package first
- Gaia storage layout and pass layout are domain concepts, not incidental
  storage details
- restartability, scheduling, and provenance are first-class concerns from the
  start
- compatibility pressure should stay in `survey_tools` longer because pinned
  AO-sky assets already exist downstream
- the generic package surface should keep scorer backends pluggable rather than
  absorb instrument-specific policy

## Immediate Next Step

After this planning pass, the best first implementation thread is:

1. re-implement the non-plotting HEALPix helpers needed beyond the raw Gaia
   store boundary
2. build the higher-level star-loading path for asterism construction,
   including proper-motion application and neighbour-aware views
3. re-implement and split the asterism search code on top of that loader path

That keeps the raw Gaia nucleus stable while moving the next thread to the
Phase 2 spatial and asterism core.
