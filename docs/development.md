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

## Minimal Build Workflow

The current public build surface is centered on:

- `ao-sky init`
- `ao-sky run`
- `ao-sky restart`
- `ao-sky show`

`init` takes a minimal build-definition YAML file. The required fields are:

- `ao_system_short_name`
- `config_short_name`
- `gaia_release`
- `outer_level`
- `inner_level`
- `max_data_level`
- `epoch`

Optional fields:

- `min_galactic_latitude`
- `survey_extent_overlays`

Example build definition:

```yaml
ao_system_short_name: GNAO
config_short_name: baseline
gaia_release: dr3
outer_level: 6
inner_level: 14
max_data_level: 9
epoch: 2028.0
min_galactic_latitude: 10.0
survey_extent_overlays:
  - name: ews
    moc_files:
      - ../data/euclid/rsd2024a-footprint-equ-13-year1-MOC.fits
```

`gaia_root`, `build_root`, `dust_root`, and `model_root` are not part of the
build definition. Resolve them either with CLI options or with a project-root
`aosky.conf` file:

```yaml
gaia_root: /data/gaia
build_root: /data/ao-builds
dust_root: /data/dust
model_root: /data/models
```

If `aosky.conf` is present in the working project root, `init`, `restart`,
`check`, and `fetch-dust` can use it automatically. Otherwise pass the root
flags explicitly.

Example CLI flow with explicit roots:

```bash
./.conda/bin/ao-sky check \
  --gaia-root /data/gaia \
  --build-root /data/ao-builds \
  --dust-root /data/dust \
  --model-root /data/models

./.conda/bin/ao-sky fetch-dust \
  --dust-root /data/dust

./.conda/bin/ao-sky init build.yaml \
  --gaia-root /data/gaia \
  --build-root /data/ao-builds \
  --dust-root /data/dust \
  --model-root /data/models

./.conda/bin/ao-sky show /data/ao-builds/GNAO-baseline-v1
./.conda/bin/ao-sky run /data/ao-builds/GNAO-baseline-v1
./.conda/bin/ao-sky restart GNAO baseline --build-root /data/ao-builds
```

During the current migration phase, `init` also accepts `--legacy-config` to
override the temporary legacy `survey_tools/aomap/config.yaml` runtime-policy
source. Normal repo usage should rely on the default unless a task explicitly
needs a different legacy config. When `model_root` is omitted, `init` falls
back to the sibling legacy models cache under `../survey_tools/data/models`.
