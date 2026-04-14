# Development

This document is the source of truth for local bootstrap, environment setup,
and daily development workflow in `ao-sky`.

`ao-sky` is still in its bootstrap stage, so some commands described here are
intended Phase 0 outcomes rather than commands that already exist today. When
Phase 0 is implemented, update this document immediately with the exact
canonical commands.

## Shared Guidance

This repo adopts the shared Python-development guidance in:

- `astro-agents/guidance/python-development.md`

Repo-local environment choices, bootstrap commands, and daily workflow
expectations in this document remain the source of truth for `ao-sky`.

## Local Environment

- Use a repo-local Conda environment for Python commands, test runs, docs
  builds, and CLI runs unless a task explicitly requires something else.
- The intended canonical location is `./.conda`.
- Do not rely on an undocumented user-global Python environment for normal repo
  work.

## Current Bootstrap Status

The repo does not yet have its Phase 0 Python package baseline.

Not yet implemented:

- `pyproject.toml`
- `src/ao_sky`
- editable install workflow
- docs-site build workflow
- canonical test commands

That means this document currently defines the intended workflow shape and the
expected local-environment policy, but not a complete runnable toolchain yet.

## Phase 0 Development Targets

Phase 0 should establish the following repo-local workflow:

1. Create or update the local `./.conda` environment.
2. Activate that environment for Python work in this repo.
3. Install `ao-sky` in editable mode once packaging exists.
4. Use the same environment for tests, docs builds, and CLI smoke checks.
5. Keep the canonical verification commands in `docs/testing.md`.

## Daily Workflow Expectations

- Keep bootstrap, environment, and daily command guidance here.
- Keep verification commands and completion expectations in `docs/testing.md`.
- Keep package boundaries, persisted contracts, and lifecycle-sensitive design
  rules in `docs/architecture.md`.
- Keep migration sequencing and compatibility-handoff planning in `docs/plan.md`.

## Commands To Add During Phase 0

When Phase 0 is implemented, add the exact canonical commands for:

- creating the local Conda environment
- activating the local Conda environment
- installing the package in editable mode
- running the canonical test path
- building docs, if docs build tooling is added in Phase 0
- running any canonical formatting, lint, or hook steps if they become part of
  the repo workflow
