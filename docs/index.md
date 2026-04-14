# ao-sky

`ao-sky` is the public Python package for Gaia-backed AO-sky mapping,
guide-star loading, asterism search, and derived all-sky AO products.

## Current Status

The current repo surface establishes the installable package baseline:

- `src/ao_sky` is the canonical package root
- `ao-sky` is the public CLI entrypoint
- `./.conda` is the canonical local development environment
- strict docs and packaging checks are part of the repo-local verification path

The scientific core remains in the next planned implementation areas. Use
[`plan.md`](plan.md) for migration sequencing and
[`architecture.md`](architecture.md) for the intended package boundary.

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
