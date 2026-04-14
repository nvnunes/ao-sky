# ao-sky

`ao-sky` is the public Python package for Gaia-backed AO-sky mapping,
guide-star loading, asterism search, and derived all-sky AO products.

## Current Status

The current repo surface establishes:

- `src/ao_sky` is the canonical package root
- `ao_sky.gaia` is the implemented raw Gaia per-pixel store boundary
- `ao-sky` is the public CLI entrypoint
- `./.conda` is the canonical local development environment
- strict docs and packaging checks are part of the repo-local verification path

The next planned implementation areas are the higher-level spatial helpers,
asterism path, and derived photometric layers. Use [`plan.md`](plan.md) for
migration sequencing and [`architecture.md`](architecture.md) for the package
boundary.

## Start Here

- [`architecture.md`](architecture.md): package boundaries, data layers, and
  persisted contracts
- [`development.md`](development.md): local environment creation, editable
  installs, and daily commands
- [`testing.md`](testing.md): canonical verification commands and completion
  expectations
- [`benchmarking.md`](benchmarking.md): retained storage and scheduling
  benchmark evidence
- [`plan.md`](plan.md): migration sequencing and immediate next steps
