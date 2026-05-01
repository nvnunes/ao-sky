# Testing

This document is the source of truth for local verification commands and
completion expectations in `ao-sky`.

## Shared Validation

Use this document as the source of truth for local verification commands and
completion expectations.

For shared review, use:

- `$agent-surface-review` when agent-facing instructions, docs, or skill-selection context changes.
- `$documentation-surface-review` when documentation surface behavior changes.
- `$code-quality-review` for source-code quality review when requested or when the change needs review beyond tests.

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
./.conda/bin/ao-sky fetch-gaia --help
./.conda/bin/ao-sky run --help
./.conda/bin/ao-sky restart --help
```

When the public build surface changes, also smoke-check the build lifecycle.
The repository test suite covers this contract, and an additional manual CLI
smoke path is below. This path is intentionally repo-native and offline: it
creates a minimal shared Gaia summary fixture plus dust/model placeholders
rather than calling the live Gaia archive or downloading dust.

```bash
tmpdir="$(mktemp -d)"
cat >"$tmpdir/build.yaml" <<'YAML'
schema_version: 2
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
  outer_level: 0
  inner_level: 1
gaia:
  release: dr3
  epoch: 2028.0
  max_bright_star_mag: 8.0
  max_bright_star_exclusion_arcsec: null
maps:
  max_level: 1
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
YAML
mkdir -p "$tmpdir/models"
./.conda/bin/python - "$tmpdir" <<'PY'
from pathlib import Path
import gzip
import sys

import numpy as np
from astropy.table import Table

from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore

root = Path(sys.argv[1])
summary = np.zeros(
    12,
    dtype=[
        ("outer_pix", "<i8"),
        ("star_count", "<i8"),
        ("loaded", "?"),
    ],
)
summary["outer_pix"] = np.arange(12, dtype=np.int64)
summary["loaded"] = True
GaiaSummaryStore(
    GaiaStoreConfig(root=root / "gaia", release="dr3", healpix_level=0)
).write_summary(Table(summary))

dust_file = root / "gaia" / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
dust_file.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(dust_file, "wt", encoding="utf-8") as handle:
    handle.write("healpix_id,healpix_level,a0,optimum_hpx_flag\n")
    for healpix_id in range(48):
        handle.write(f"{healpix_id},1,0.5,true\n")

model_root = root / "models"
model_root.mkdir(parents=True, exist_ok=True)
for model_name in (
    "point_one",
    "point_two",
    "point_three",
    "mean_one",
    "mean_two",
    "mean_three",
):
    (model_root / f"{model_name}.pt").write_bytes(model_name.encode() + b":pt")
    (model_root / f"{model_name}_metadata.pkl").write_bytes(
        model_name.encode() + b":metadata"
    )
PY
./.conda/bin/ao-sky init "$tmpdir/build.yaml" \
  --gaia-root "$tmpdir/gaia" \
  --build-root "$tmpdir/builds" \
  --model-root "$tmpdir/models"
build_path="$tmpdir/builds/v1"
./.conda/bin/ao-sky check \
  --gaia-root "$tmpdir/gaia" \
  --build-root "$tmpdir/builds" \
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

The current full-build operating default on the local workstation is:

- `build.workers: 9`
- `build.scheduler: dynamic`
- `build.memory_limit_mb: 26624`
- `prediction.resolved_device: auto`
- `prediction.averaged_device: auto`
- `build.roots.gaia: /Users/nelsonnunes/ao-sky-cache/gaia`
- artifact writes go directly to the build tree on `/Volumes/Data/Galaxy/aosky`
- repo-owned HDF5 writes use Blosc Zstd level 5 compression, including
  canonical Gaia files and derived build artifacts
- speculative Gaia prewarm, write-behind artifact writes, and SSD artifact
  staging are documented in `docs/benchmarking.md` but are not retained in the
  active implementation

For the live GNAO workspace, run commands from
`/Volumes/Data/Galaxy/aosky/gnao-baseline` and use the local `./ao-sky` wrapper.
If a previous run was interrupted, inspect the build first and use `restart` so
stale `running` rows are repaired by the runner:

```bash
cd /Volumes/Data/Galaxy/aosky/gnao-baseline
./ao-sky check
./ao-sky show v1
./ao-sky restart v1
```

### Long-Running Build Supervision

Use this procedure for overnight or unattended local full-sky builds. It is an
operational monitoring loop, not a correctness test.

Launch the build under `caffeinate` so the workstation does not sleep:

```bash
cd /Volumes/Data/Galaxy/aosky/gnao-baseline
caffeinate -dimsu ./ao-sky restart v2
```

Every `10` minutes, check that the run is alive and progressing:

```bash
ps -axww | rg -i 'ao-sky restart v2|caffeinate -dimsu ./ao-sky restart v2'
./ao-sky show v2
```

Then inspect `build.log` for:

- completed outer-pixel count
- failed outer-pixel count
- pending/remaining outer-pixel count
- recent throughput, usually over the last `10` minutes
- overall throughput
- ETA from recent throughput
- memory-pressure state counts: normal, trim, and pause
- latest total RAM
- peak total RAM

The monitor should continue the run when progress is steady, failures are zero,
and memory pressure is not sustained. Stop and inspect when any of the following
occur:

- failed outer pixels appear
- the build process exits unexpectedly
- memory spends sustained time in pause
- trim/pause dominates throughput
- total RAM approaches the hard ceiling
- throughput stalls for multiple check intervals

If the process exits but the build is restartable, inspect `./ao-sky show`,
review the recent `build.log` tail, and restart with `./ao-sky restart <build>`.
The runner repairs stale `running` rows to `pending` at restart. If the run was
manually interrupted, expect the persisted state to show a small number of stale
`running` rows corresponding to in-flight worker pixels.

When stopping a supervised run for analysis, record:

- final process state
- `./ao-sky show <build>` output
- log-derived completed, failed, and remaining counts
- last-10-minute, last-30-minute, and last-hour throughput
- normal/trim/pause counts
- latest and peak total RAM
- archive location for `build.log`, `build.h5`, and `build.yaml`

## Developer Script Policy

Keep reusable migration scripts in `scripts/`, but do not promote
every one-off investigation into permanent project surface.

Do not retain one-off prewarm, write-behind, artifact-staging, compression, or
cache probes as separate scripts unless they become repeated workflows. Their
results belong in `docs/benchmarking.md`.

Stable offline correctness checks should move into `pytest` once they no longer
need live legacy assets. Live migration checks should stay as explicit script
commands so normal repo verification remains repo-native and offline.

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
