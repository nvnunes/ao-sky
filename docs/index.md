# ao-sky

`ao-sky` is the public Python package for Gaia-backed AO-sky mapping,
guide-star loading, asterism search, and derived all-sky AO products.

## Current Surface

The current repo surface establishes:

- `src/ao_sky` is the canonical package root
- `ao_sky.gaia` owns raw Gaia storage and proper-motion transforms
- `ao_sky.spatial` owns the reusable non-plotting HEALPix helpers
- `ao_sky.asterisms` owns in-memory star assembly and search APIs
- `ao_sky.build` owns persisted build roots, state, and per-outer-pixel
  derived artifacts
- `ao-sky` is the public CLI entrypoint
- `./.conda` is the canonical local development environment
- strict docs and packaging checks are part of the repo-local verification path

## Start Here

- [`architecture.md`](architecture.md): package boundaries, data layers, and
  persisted contracts
- [`api.md`](api.md): supported Python API and public helper contracts
- [`algorithms.md`](algorithms.md): traversal, dense-field NGS selection, and
  winner-selection algorithms
- [`development.md`](development.md): local environment creation, editable
  installs, and daily commands
- [`testing.md`](testing.md): canonical verification commands and completion
  expectations
