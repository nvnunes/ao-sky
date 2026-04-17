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

This repository now has an implemented package, CLI, native runtime config,
Gaia summary prerequisite, build lifecycle, artifact layout, and comparison
tooling. This plan remains the migration sequencing source of truth; it is not
a claim that `ao-sky` is still design-only.

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
- build metadata, state, and derived-artifact layout
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
- If a supported `ao-sky.yaml` is added later, treat it as a CLI or
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

### Phase 3: Define The Derived Build Model

- Replace `config.folder` coupling with explicit build metadata and layouts.
- Select and define the new persisted asterism data format rather than
  assuming legacy FITS compatibility.
- Implement build-aware paths for:
  - inner products
  - asterism catalogs
- Define build metadata and restart state.

### Phase 4: Rebuild The Execution Engine

- Implement restart-aware planning over unfinished outer pixels.
- Replace chunk-barrier scheduling with a more flexible scheduler.
- Add neighbour-oriented traversal and star-count balancing.
- Keep this phase single-process; defer parallel worker execution and runtime
  optimization to later phases.
- If a supported repo-root `ao-sky.yaml` is added, define it here as a CLI or
  application-layer configuration surface rather than as hidden package-global
  behavior inside `ao_sky.gaia`.

### Phase 5: Add Asterism And Inner-Product Richness

- Treat the overall Build as four phases:
  Gaia Loading, Traversal, Aggregation, and Augmentation.
- Use this phase to flesh out the Traversal products beyond the minimal
  persisted outer-pixel contract established earlier.
- Treat Traversal as one outer-pixel analysis pipeline, not as multiple
  persisted passes.
- Structure Traversal around these explicit steps:
  Load Gaia Neighborhood, Generate Candidate Asterisms, Evaluate Candidate
  Performance, Select Winning Asterisms, Evaluate Winner Performance, and
  Persist Outer-Pixel Products.
- Keep raw candidate asterisms transient and in memory; do not persist the
  full raw candidate set.
- Keep the temporary live legacy Traversal shell-out isolated to comparison
  tooling so it cannot become part of the canonical build API.
- Do not persist `Av` in build `asterisms` during this phase.
- Do not use dust-based asterism filtering during Phase 5 Traversal; compare
  against the legacy path with the asterism dust cut disabled.
- Put the temporary Phase 5 overlap handling in Select Winning Asterisms
  rather than in candidate generation so the later winner-map-first algorithm
  can replace it without changing the earlier candidate-building contract.
- Implement the richer overlap-handling and winner-selection path needed for
  build parity with `survey_tools`.
- Expand the inner-product model beyond the initial count-oriented surface.
- Extend the temporary legacy-comparison harness as needed to validate the
  richer derived products against the live legacy path while this phase is in
  progress.

### Phase 6: Add Dust-Enriched Derived Products

- Implement local dust loading in `ao-sky` rather than extending the temporary
  legacy Traversal shell-out.
- Introduce `ao_sky.dust` as the owner for dust loading and sampling, with
  Gaia TGE as the only backend in this phase.
- Add `max_data_level` to the build definition and persisted build metadata,
  and validate `outer_level <= max_data_level <= inner_level`.
- Extend Traversal with an `Apply Dust Field` step that samples Gaia TGE `A0`
  on the outer pixel at `max_data_level`, repeats it to `inner_level` as
  needed, and persists it into dense `inner` as `gaia_A0`.
- Keep dust data-only in this phase:
  do not add dust-based Traversal filters and do not let dust alter `best_*`,
  `winner_*`, or coverage fields.
- Keep build `asterisms` dust-free in this phase; export-oriented dust fields
  still belong to the later export phase.
- Configure the local dust data location through the planned `ao-sky.yaml`
  surface rather than by copying the legacy relative-path behavior.
- Extend the temporary legacy-comparison harness to validate the new local
  `gaia_A0` path against the live legacy coarse-sampling behavior.

### Phase 7: Add All-Sky Derived Products

- Rebuild the Aggregation part of the Build on top of the new model.
- Extend the temporary legacy-comparison harness as needed to validate these
  all-sky aggregation outputs where useful during migration.

### Phase 8: Add All-Sky Augmentation

- Rebuild the Augmentation part of the Build, including survey-extent overlays
  and other richer all-sky derived products.

### Phase 9: Replace The Temporary Legacy Shell-Out

- Replace the temporary live legacy Traversal shell-out used for richer
  per-outer-pixel products with native `ao-sky` implementation.
- Move the remaining AO/winner prediction and selection path into the canonical
  package surface, leaving live legacy shell-outs in comparison tooling only.
- Keep the Phase 5/6 product contracts stable while removing the live
  dependency on legacy `survey_tools` execution during builds.
- Continue using the temporary live legacy-comparison harness to validate the
  native path while this replacement is in progress.

### Phase 10: Audit `survey_tools` `aomap` For Missing Implementation

- Perform a focused audit of the live `survey_tools` `aomap` implementation
  against the then-current `ao-sky` build surface.
- Identify any remaining legacy `aomap` capabilities that have not yet been
  re-implemented or intentionally deferred.
- Record the remaining gaps clearly enough that the compatibility-adoption work
  does not discover missing functionality late.
- Use the audit to confirm that the preceding build-product phases are
  sufficient before cache work and compatibility adoption proceed.

### Phase 11: Add Gaia Pre-Download And Summary Support

- Add a CLI command that can pre-download the canonical Gaia store for one
  explicit Gaia release and outer level.
- Build and persist a dense shared Gaia summary artifact with one row per outer
  pixel and fields `outer_pix`, `star_count`, and `loaded`.
- Require the matching shared Gaia summary during `init`; build creation should
  fail clearly when the summary is missing and direct the user to
  `fetch-gaia`.
- Use summary `star_count` as the scheduling proxy for Traversal.
- Keep `fetch-gaia` resumable by default by skipping already loaded files,
  checking file presence, and refreshing summary progress from the on-disk
  store state without bulk deletion.

### Phase 12: Add Parallel Execution Support

- Add parallel worker execution on top of the current single-process execution
  engine.
- Keep this phase focused on the worker model, worker ownership, and the
  interaction between parallel execution and the persisted build state.
- Refresh the relevant execution benchmarks under the current storage setup so
  the parallel worker design is grounded in current measurements rather than
  only the historical baseline in `docs/benchmarking.md`.
- Benchmark the parallel execution path against the non-cache single-process
  execution engine before adding cache complexity.

### Phase 13: Major Traversal Optimization Pass

Status: complete. Phase 13 is the known-good legacy-preserving state before
Phase 14 intentionally changes the winner/overlap algorithm.

- Treated this as an end-to-end Traversal optimization pass across scheduling,
  runtime caching, memory safety, artifact IO, compression, telemetry,
  build-local snapshots, and legacy validation.
- Implemented regional Traversal execution that groups outer pixels by a
  coarse HEALPix region level derived from `outer_level` and `workers`, then
  assigns balanced region sets to long-lived workers.
- Reused the long-lived Traversal worker runtime for the default single-worker
  path, running in process without spawn overhead so runtime setup, model
  caches, and Gaia table caches are reused by default.
- Kept Traversal outer-pixel independent by deriving rich inner performance and
  retained local asterisms from the current pixel's two-ring expanded
  Gaia-star footprint, not from neighbouring pixels' derived asterism
  artifacts.
- Kept the two-ring boundary as an internal fixed rule, not a runtime knob. One
  ring was insufficient for self-contained best-map fidelity, and two is the
  minimal integer-ring margin adopted for this phase. Because expansion happens
  before build-epoch proper-motion shifting, the extra ring also acts as a
  pragmatic buffer for high-proper-motion stars that originate just outside the
  raw outer-pixel boundary but can move into the relevant edge footprint.
- Implemented a worker-local in-memory LRU cache for runtime Gaia tables,
  controlled by both an entry target and a per-worker memory cap. Canonical
  Gaia files remain raw, but cached runtime rows are shifted to the build
  epoch, enriched with `R` and `hpx14`, and marked read-only so Traversal does
  not repeatedly apply proper motion or recompute fine HEALPix projections.
  Native build Traversal treats this as its row contract for inner counts,
  asterism search, and prediction.
- Hardened the runtime Gaia cache by fixing oversized-entry behavior,
  preserving force-reload semantics, adding basic thread safety, and making
  read-only prepared tables the normal cached representation.
- Used the shared Gaia summary as a cheap coarse stellar-density prefilter for
  Traversal asterism generation. If an outer pixel's average summary density is
  already above the configured NGS density cutoff, Traversal skips the
  expensive asterism path for that outer pixel while still building base inner
  counts from runtime Gaia rows. Finer density indexes are deferred to Phase
  15.
- Kept the parent process as the only owner of `build.h5` state while workers
  stream state messages back to the parent. Parent state writes are buffered
  and flushed periodically with retry/backoff rather than opening and writing
  `build.h5` for every row update.
- Kept successful build-log output periodic and failure logs per-pixel, avoiding
  high-volume per-pixel success log I/O.
- Exposed worker count, Gaia cache, per-worker memory guard, parent aggregate
  memory guard, and telemetry settings as runtime CLI options with optional
  `ao-sky.yaml` defaults, without persisting them into build metadata. The
  regional assignment level is derived internally from `outer_level` and
  `workers` rather than exposed as a config knob.
- Set the current local operating defaults to `workers=6`,
  `worker_memory_limit_mb=2048`, `parent_memory_limit_mb=12288`, Gaia data on
  the SSD mirror, worker-local cache enabled, and direct HDD artifact writes.
- Persisted a build-local native `build.yaml` at `init`. Legacy YAML conversion,
  when needed for comparison work, now happens outside the build API and passes
  the resulting native runtime config into `init`.
- Moved build-local model and survey snapshots into `init`: configured model
  files are copied into `<build>/models`, configured survey MOC files are copied
  into `<build>/surveys`, and the build metadata points at those build-local
  snapshots.
- Moved Gaia TGE dust into the Gaia root as part of `fetch-gaia`, then created a
  build-local dense `A0` cache under `<build>/dust` at every `init` so workers
  use a small mmap-friendly build artifact instead of repeatedly reading the
  source CSV.
- Switched derived build artifacts to Blosc Zstd by default. Benchmarks showed
  it preserved near-maximum compression while reducing artifact-write time from
  the old `gzip=9` path to a negligible part of Traversal.
- Added `basic` and `detailed` Traversal telemetry. Basic telemetry remains
  low-overhead operational profiling in `build.log`; detailed telemetry writes
  per-pixel RSS checkpoints and intermediate cardinalities for benchmark and
  memory investigations.
- Benchmarked worker scaling, memory guards, Gaia HDD versus SSD reads, cache
  on/off behavior, compression codecs, artifact staging, read prewarm, and
  write-behind. The retained path is regional workers plus worker-local
  prepared-table caching, SSD Gaia reads, Blosc Zstd artifacts, and direct
  worker-owned artifact writes.
- Confirmed the known-good legacy-preserving state with the 12-pixel live
  legacy comparison using `workers=3`. Retained asterism membership matched
  exactly except for accepted two-ring boundary-footprint divergences, and
  dense inner row domains and ordering matched for all sampled pixels.

Measured optimization decisions retained from Phase 13:

- Read prewarm stays out of the active implementation. The tested design used
  one worker-local IO thread per worker, a bounded queue, and the existing
  prepared Gaia table cache. Fixed-pixel HDD/SSD comparisons showed the benefit
  came primarily from regional traversal plus prepared-table cache reuse, not
  speculative prewarm. Revisit only if future profiling shows workers blocked
  on cold reads that the cache cannot amortize.
- Write-behind stays out of the active implementation. The tested design kept
  artifact writes worker-owned, bounded pending write bytes, applied
  backpressure, and reported pixels complete only after `outer.h5` writes
  succeeded. The measured benefit did not justify the async completion path,
  failure surface, memory accounting, and profiling complexity.
- Artifact staging stays out of the active implementation. The tested design
  wrote flat `<outer_pix>.h5` files to an SSD staging folder and had the parent
  promote them into canonical HEALPix paths before marking pixels done. After
  switching derived artifacts to Blosc Zstd, staging showed no wall-time
  benefit, so direct worker-owned writes remain the normal path.
- Build-log batching stays low priority because successful progress logging is
  already periodic; revisit only if measurement shows it contributes meaningful
  overhead.
- Higher-density memory pressure remains a Phase 15 problem. The Phase 13
  optimization pass added coarse summary-based skipping and memory guards, but
  the next algorithmic memory reductions should focus on bounded asterism
  search, bounded inner/asterism context construction, and finer density
  indexes.

### Phase 14: Improve The Winning-Asterism Algorithm

- Revisit the interim winning-asterism selection path after the preceding
  build-product and audit phases have exposed the remaining weaknesses.
- Replace the legacy-style overlap-plus-winner heuristic with a winner-map-first
  formulation that solves for the winning asterism at each inner pixel.
- Add regularization or smoothing to that winner field so the resulting winner
  map is not dominated by highly pixelated local switches driven by small EE
  differences.
- Treat retained asterisms as the unique asterisms that win somewhere in the
  solved winner field, rather than as the result of a separate heuristic
  overlap-pruning step.
- Improve the winner-selection algorithm itself rather than just reproducing
  the current legacy behavior.
- Validate the improved winner path against the richer per-outer-pixel and
  all-sky products before compatibility adoption proceeds.

### Phase 15: Explore Ways to Handle Higher Stellar Density

- Currently the asterism code explodes when there are too many stars in an outer pixel
- Explore ways to improve the algorithm so higher density fields can still be used
- Revisit the coarse Gaia-summary density prefilter from the Phase 13
  optimization pass and consider a finer persisted Gaia density index, such as
  per-file FOV-level counts, so Traversal can avoid loading or preparing rows
  from rejected high-density subcells rather than skipping whole outer pixels.

### Phase 16: Rebuild The Asterism Catalog Export Path

- Rebuild the asterism catalog export workflow on top of the richer `ao-sky`
  build products rather than the legacy `aomap` export path.
- Add export-oriented asterism-side fields such as `Av` there, rather than
  carrying them in the core build `asterisms` artifact.
- Define which richer asterism-side fields belong in exported catalogs versus
  remaining transient model-prediction context.
- Validate exported catalogs against the intended downstream use cases before
  compatibility adoption proceeds.

### Phase 17: Define The WFS Photometric Proxy Layer

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

### Phase 18: Add Build Provenance, Introspection, And Validation Support

- Add a build-root provenance manifest that summarizes existing authoritative
  metadata rather than replacing `build.h5`, `build.yaml`, Gaia
  summaries, model manifests, map artifacts, or overlay datasets as sources of
  truth.
- Include enough manifest detail to audit and compare builds:
  runtime-config path and content hash, merged build-config content hash, package
  version or code identity, Gaia release and level, Gaia summary path, dust
  dataset identity, model snapshot manifest, map levels, survey overlays, and
  implemented build phase completion.
- Add build introspection commands or APIs that can list available builds,
  inspect one build's provenance and artifact completeness, and compare two
  builds for meaningful policy, model, input, and artifact differences.
- Keep introspection read-only and contract-focused; do not add implicit
  migration, repair, or rebuild behavior to these commands.
- Define a lightweight science-validation workspace and toolkit for comparing
  important build outputs before compatibility adoption.
- Standardize representative sample pixels, low-density smoke samples, map
  comparison summaries, and human-inspection outputs so algorithm-changing
  phases can be reviewed consistently.
- Use this phase to make post-legacy comparisons reproducible after the
  winner-algorithm and photometric-proxy phases intentionally move beyond exact
  legacy parity.

### Phase 19: Compatibility Adoption In `survey_tools` And `girmos-aosims`

- Add thin `survey_tools` adapters that call `ao-sky` public APIs.
- Add the downstream adoption work needed for `girmos-aosims` to consume the
  new `ao-sky` build and photometric surfaces cleanly.
- Keep pinned legacy assets readable through explicit compatibility paths.
- Repoint downstream consumers incrementally across both `survey_tools` and
  `girmos-aosims`.
- Avoid deleting legacy code until the new path is documented, tested, and used
  in practice.

### Phase 20: Deduplication And Final Handoff

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
- Gaia storage layout and build layout are domain concepts, not incidental
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
