# ao-sky

`ao-sky` is a public Python package for AO-sky mapping, Gaia-backed
guide-star loading, asterism search, and derived all-sky AO products.

The repository currently provides:

- an installable `src/ao_sky` package
- a raw `ao_sky.gaia` store boundary for canonical per-pixel Gaia HDF5 files
- Gaia-domain proper-motion transforms on canonical Gaia tables
- reusable `ao_sky.spatial` HEALPix helpers
- public in-memory `ao_sky.asterisms` star-loading and search APIs
- persisted `ao_sky.build` build roots, build state, and per-outer-pixel
  derived artifact containers
- a public `ao-sky` CLI with build lifecycle commands
- a repo-local Conda workflow rooted at `./.conda`
- strict docs-site and packaging verification commands

## Quick Start

```bash
conda create -y -p ./.conda python=3.12
./.conda/bin/python -m pip install -e ".[dev,docs]"
./.conda/bin/ao-sky --version
```

Use [`docs/development.md`](docs/development.md) for the full local workflow
including root preflight, Gaia fetch, build definitions, runtime config, and
`ao-sky init|run|restart|show`. Use
[`docs/testing.md`](docs/testing.md) for the canonical verification path.

## Core Docs

- [`docs/architecture.md`](docs/architecture.md): package and data/build design
- [`docs/api.md`](docs/api.md): supported Python API and public helper contracts
- [`docs/development.md`](docs/development.md): local bootstrap and development workflow
- [`docs/plan.md`](docs/plan.md): transitional migration context and implementation plan
- [`docs/benchmarking.md`](docs/benchmarking.md): benchmark evidence that informs storage and cache decisions
- [`docs/testing.md`](docs/testing.md): verification expectations
