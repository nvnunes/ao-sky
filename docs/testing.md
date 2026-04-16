# Testing

This document is the source of truth for local verification commands and
completion expectations in `ao-sky`.

## Shared Validation

Use the shared base testing guidance in `astro-agents/validation/base-testing.md`.

## Environment

Use the local `./.conda` environment for Python commands, test runs, packaging
checks, and docs builds unless a task explicitly requires something else.

Create the environment and install the package with:

```bash
conda create -y -p ./.conda python=3.12
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

## Canonical Verification Commands

Run the Python test suite with:

```bash
./.conda/bin/python -m pytest -q
```

Build the package artifacts with:

```bash
./.conda/bin/python -m build --no-isolation
```

Build the docs site in strict mode with:

```bash
./.conda/bin/mkdocs build --strict
```

Smoke-check the installed CLI entrypoint with:

```bash
./.conda/bin/ao-sky --version
```

When the runtime-root CLI surface changes, also smoke-check the preflight
commands:

```bash
./.conda/bin/ao-sky check --help
./.conda/bin/ao-sky fetch-dust --help
```

When the public build surface changes, also smoke-check the build lifecycle.
The repository test suite covers this contract, and an additional manual CLI
smoke path is:

```bash
tmpdir="$(mktemp -d)"
cat >"$tmpdir/build.yaml" <<'YAML'
ao_system_short_name: GNAO
config_short_name: baseline
gaia_release: dr3
outer_level: 0
inner_level: 1
max_data_level: 1
epoch: 2028.0
YAML
cat >"$tmpdir/legacy.yaml" <<'YAML'
ao_systems:
  - name: GNAO
    band: R
    fov: 120.0
    fov_1ngs: 60.0
    min_wfs: 2
    max_wfs: 3
    min_mag: 8.0
    nom_mag: 16.0
    max_mag: 18.5
    min_sep: 5.0
    max_sep: 120.0
asterisms_max_star_density: 6.0
asterisms_max_bright_star_mag: 8.0
asterisms_max_overlap: 0.66
YAML
build_path="$(
  ./.conda/bin/ao-sky init "$tmpdir/build.yaml" \
    --gaia-root "$tmpdir/gaia" \
    --build-root "$tmpdir/builds" \
    --dust-root "$tmpdir/dust" \
    --model-root "$tmpdir/models" \
    --legacy-config "$tmpdir/legacy.yaml"
)"
./.conda/bin/ao-sky check \
  --gaia-root "$tmpdir/gaia" \
  --build-root "$tmpdir/builds" \
  --dust-root "$tmpdir/dust" \
  --model-root "$tmpdir/models"
./.conda/bin/ao-sky show "$build_path"
```

When survey overlays are configured in the build definition, the current build
surface may also auto-advance from `aggregation` into `augmentation` and add a
`survey_extent` dataset to each `maps-hpx<level>.h5` file.

Refresh the editable install whenever package metadata, dependencies, or entry
points change:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

The repo includes a versioned pre-commit hook at `.githooks/pre-commit`.
That hook runs:

- `./.conda/bin/python -m pytest -q`
- `./.conda/bin/mkdocs build --strict`

For live migration comparisons against `survey_tools`, use the repo helper:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py --sample smoke --model-root ../survey_tools/data/models
```

The default `smoke` sample intentionally avoids the slower pathological outer
pixels so routine phase work does not spend most of its time in a few crowded
regions. Use the broader sample only when you want a heavier check:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py --sample full --model-root ../survey_tools/data/models
```

For sparse all-sky aggregation comparisons against legacy, use the same helper
with `--maps`:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py --maps --sample smoke --model-root ../survey_tools/data/models
```

If the hooks path is not active in your clone, set it with:

```bash
git config core.hooksPath .githooks
```

## Completion Expectations

Run the full package-surface verification path before concluding substantial
changes to the public package surface:

- `./.conda/bin/python -m pytest -q`
- `./.conda/bin/python -m build --no-isolation`
- `./.conda/bin/mkdocs build --strict`
- `./.conda/bin/ao-sky --version`

Docs-only work is complete when:

- the changed docs remain internally consistent
- source-of-truth ownership remains clear across `README.md`, `AGENTS.md`, and
  the docs surface
- cross-links and package/workflow claims remain current
- the strict docs build passes

Package, CLI, or packaging changes are complete when:

- the package installs cleanly in editable mode
- the Python test suite passes
- the package build succeeds
- the CLI smoke check succeeds
- build-surface changes also cover the minimal build CLI smoke path
- docs are updated when supported usage or workflow changed

Planning, architecture, or benchmark-sensitive doc changes are complete when:

- the owning document reflects the decision
- adjacent docs are updated when discovery or ownership changes
- benchmark-sensitive claims remain consistent with `docs/benchmarking.md`

Gaia store tests should remain offline:

- materialization tests should monkeypatch the archive query seam
- normal repo verification should not depend on live Gaia archive access

Phase 2 spatial and asterism tests should also remain repo-native:

- the normal test suite should not import or execute `survey_tools`
- neighbour-stitching tests should build synthetic Gaia tables locally
- search tests should verify deterministic output and schema contracts rather
  than comparing against legacy assets in the default repo test path

## Verification Scope Boundaries

- Keep environment creation, activation, and daily workflow commands in
  `docs/development.md`.
- Keep verification commands and completion expectations here.
- Keep benchmark methodology and benchmark results in `docs/benchmarking.md`.
