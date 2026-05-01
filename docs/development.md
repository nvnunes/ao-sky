# Development

This document is the source of truth for local bootstrap, environment setup,
and daily development workflow in `ao-sky`.

This document records the exact repo-local commands for environment creation,
editable installs, and daily development work.

## Shared Skills

For Python code changes, use `$python-code-writing` alongside this project's
local environment and workflow rules.

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

`init` takes one merged build-config YAML file. The config filename identifies
the build lineage, while the AO-system, prediction, traversal, Gaia, maps,
asterism, best, and coverage sections define the runtime policy.
`survey_overlays` is optional.

Example build config:

```yaml
schema_version: 2
build:
  workers: 9
  scheduler: dynamic
  memory_limit_mb: 26624
  roots:
    gaia: /data/gaia
    model: /data/models
    survey: /data/surveys
ao_system:
  band: R
  fov_arcsec: 120.0
  lgs: []
  min_wfs: 1
  max_wfs: 3
  min_mag: 8.0
  max_mag: 18.5
  min_sep_arcsec: 5.0
prediction:
  resolved_device: gpu
  averaged_device: gpu
  wavelength_micron: 1.654
  resolved_models:
    1star: point_one
    2star: point_two
    3star: point_three
  averaged_models:
    1star: mean_one
    2star: mean_two
    3star: mean_three
traversal:
  outer_level: 6
  inner_level: 14
gaia:
  release: dr3
  epoch: 2028.0
  max_bright_star_mag: 8.0
  max_bright_star_exclusion_arcsec: null
maps:
  max_level: 9
asterism:
  winner_ee_epsilon: 0.01
best:
  seeing_baseline:
    wavelength_micron: 0.5
    sr: 0.0
    ee: 0.02
    fwhm_mas: 650.0
coverage:
  resolved_ee_threshold: 0.4
  averaged_ee_threshold: 0.3
survey_overlays:
  - name: ews
    moc_files:
      - rsd2024a-footprint-equ-13-year1-MOC.fits
```

External runtime roots live under `build.roots`. Build versions are created in
the current working directory by default. The source Gaia TGE CSV lives under
the Gaia root, while `init` converts it into a build-local dense A0 cache under
`<build>/dust` so traversal workers can memory-map the small build artifact.
CLI root flags can still override roots for tests and ad hoc runs.

`ao-sky.yaml` discovery is deliberately narrow. Commands first use an explicit
`--ao-sky-yaml` path, then `./ao-sky.yaml` in the current working directory, then
`ao-sky.yaml` in the nearest parent Python project root identified by
`pyproject.toml`. `init`, `run`, `restart`, `check`, and `fetch-gaia` can use
that discovered file
automatically. Otherwise pass the root flags explicitly. The execution settings
are runtime defaults only and are not persisted into build metadata. Traversal
derives its regional worker-assignment level from `outer_level` and `workers`;
there is no user-facing region-level setting. The current derivation targets at
least `max(4, workers * 4)` outer pixels per region, clamped by `outer_level`.
Worker-local Gaia table cache settings are also runtime-only. Set
`--gaia-cache-entries 0` or `--gaia-cache-mb 0` only for cache comparison
benchmarks; normal Traversal should keep the prepared Gaia table cache enabled.
Repo-owned HDF5 datasets use Blosc Zstd level 5 compression through
`hdf5plugin` for new writes, including canonical Gaia files and derived
artifacts. Existing legacy `gzip=9` Gaia files remain readable. If you inspect
these files directly with `h5py`, import `hdf5plugin` first so the HDF5 filter
is registered in the process.
Artifact writes are direct and worker-owned; SSD artifact staging was measured
and removed from the active execution surface.

Example lineage-workspace flow:

```bash
cd /data/ao-builds/gnao-baseline
./ao-sky check
./ao-sky fetch-gaia
./ao-sky init
./ao-sky restart gnao-baseline
```

Explicit root flags remain available for tests and ad hoc runs:

```bash
./.conda/bin/ao-sky fetch-gaia \
  --gaia-root /data/gaia \
  --gaia-release dr3 \
  --outer-level 6

./.conda/bin/ao-sky check \
  --gaia-root /data/gaia \
  --build-root /data/ao-builds \
  --model-root /data/models

./.conda/bin/ao-sky init build.yaml \
  --gaia-root /data/gaia \
  --build-root /data/ao-builds \
  --model-root /data/models \
  --survey-root /data/surveys

./.conda/bin/ao-sky show /data/ao-builds/v1
./.conda/bin/ao-sky run /data/ao-builds/v1 \
  --workers 3 \
  --gaia-cache-entries 64 \
  --gaia-cache-mb 2048 \
  --parent-memory-limit-mb 12288
./.conda/bin/ao-sky restart build \
  --build-root /data/ao-builds \
  --workers 3 \
  --parent-memory-limit-mb 12288
```

`init` copies the runtime policy from the merged build config into build-local
`build.yaml`; `run` and `restart` read that build-local config.
`ao-sky.yaml` remains path-only configuration. `init` also snapshots the
configured model files into `<build>/models`, writes the build-local Gaia TGE
A0 cache into `<build>/dust`, and updates the build to use those snapshots. For
builds with configured survey overlays, `init` resolves
each `moc_files` entry as a filename under `survey_root`, copies it under
`<build>/surveys`, and updates the build to use build-root-relative
`surveys/<filename>` paths.

For performance or memory investigations, add `--telemetry detailed` to `run`
or `restart`. Detailed telemetry writes per-pixel RSS checkpoints to
`<build>/diagnostics/traversal-memory.csv`; normal runs should use the default
basic telemetry.
