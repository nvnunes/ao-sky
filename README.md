# ao-sky

`ao-sky` is a public Python package for AO-sky mapping, Gaia-backed
guide-star loading, asterism search, and derived all-sky AO products.

The repository now includes the public package baseline:

- an installable `src/ao_sky` package
- a minimal `ao-sky` CLI entrypoint
- a repo-local Conda workflow rooted at `./.conda`
- strict docs-site and packaging verification commands

The scientific core is still in its bootstrap stage. Gaia storage, spatial
helpers, and asterism search are the next planned implementation areas
described in [`docs/plan.md`](docs/plan.md).

## Quick Start

```bash
conda create -y -p ./.conda python=3.12
./.conda/bin/python -m pip install -e ".[dev,docs]"
./.conda/bin/ao-sky --version
```

Use [`docs/development.md`](docs/development.md) for the full local workflow
and [`docs/testing.md`](docs/testing.md) for the canonical verification path.

## Core Docs

- [`docs/architecture.md`](docs/architecture.md): package and data/build design
- [`docs/development.md`](docs/development.md): local bootstrap and development workflow
- [`docs/plan.md`](docs/plan.md): transitional migration context and implementation plan
- [`docs/benchmarking.md`](docs/benchmarking.md): benchmark evidence that informs storage and cache decisions
- [`docs/testing.md`](docs/testing.md): verification expectations
