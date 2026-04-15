# Development

This document is the source of truth for local bootstrap, environment setup,
and daily development workflow in `ao-sky`.

This document records the exact repo-local commands for environment creation,
editable installs, and daily development work.

## Shared Guidance

This repo adopts the shared Python-development guidance in:

- `astro-agents/guidance/python-development.md`

Repo-local environment choices, bootstrap commands, and daily workflow
expectations in this document remain the source of truth for `ao-sky`.

## Local Environment

- Use a repo-local Conda environment for Python commands, test runs, docs
  builds, and CLI runs unless a task explicitly requires something else.
- The canonical location is `./.conda`.
- Do not rely on an undocumented user-global Python environment for normal repo
  work.

## Bootstrap

Create the local environment with:

```bash
conda create -y -p ./.conda python=3.12
```

Install the package and the current development extras with:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

For an interactive shell, the canonical activation command is:

```bash
conda activate "$(pwd)/.conda"
```

Direct `./.conda/bin/...` invocations remain the preferred non-interactive
workflow because they do not depend on shell-specific Conda initialization.

## Git Hooks

The repo includes a versioned pre-commit hook at:

- `.githooks/pre-commit`

That hook runs:

- tests: `./.conda/bin/python -m pytest -q`
- docs build: `./.conda/bin/mkdocs build --strict`

If hooks are not active in your clone, run:

```bash
git config core.hooksPath .githooks
```

## Daily Workflow Expectations

- Keep bootstrap, environment, and daily command guidance here.
- Keep verification commands and completion expectations in `docs/testing.md`.
- Keep package boundaries, persisted contracts, and lifecycle-sensitive design
  rules in `docs/architecture.md`.
- Keep migration sequencing and compatibility-handoff planning in `docs/plan.md`.
- Keep git hook activation and hook behavior here.

## Daily Commands

Use the local environment directly for normal repo work:

```bash
./.conda/bin/python -m pytest -q
./.conda/bin/python -m build --no-isolation
./.conda/bin/mkdocs build --strict
./.conda/bin/ao-sky --version
```

Refresh the editable install whenever package metadata or dependencies change:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```
