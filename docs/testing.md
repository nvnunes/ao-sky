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

## Live Legacy Compatibility Handoff

Use this section when a new thread needs to continue the legacy-preserving
Traversal work. The local paths below describe the current development machine,
not a portable CI contract.

Current live roots and fixtures:

- repo root: `/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/ao-sky`
- live build workspace: `/Volumes/Data/Galaxy/aosky/gnao-baseline`
- live build config: `/Volumes/Data/Galaxy/aosky/gnao-baseline/ao-sky.yaml`
- current live build directory: `/Volumes/Data/Galaxy/aosky/gnao-baseline/v1`
- SSD Gaia mirror: `/Users/nelsonnunes/ao-sky-cache/gaia`
- HDD Gaia and dust root: `/Volumes/Data/Galaxy/aosky`
- legacy model root: `../survey_tools/data/models`
- legacy config source used by comparison helpers:
  `../survey_tools/aomap/config.yaml`

The routine Traversal compatibility samples are fixed in
`scripts/compare_legacy_asterisms.py` so a new thread does not accidentally
change the sample while comparing behavior:

- smoke sample: `28559, 28550, 28607`
- full sample:
  `28559, 28550, 28607, 28383, 28463, 5407, 28589, 5380, 5424, 28597, 5448, 5391`
- sparse map seed pixels: `1456, 1717, 4008`

Use the smoke sample for quick checks:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py \
  --sample smoke \
  --gaia-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --dust-root /Volumes/Data/Galaxy/aosky \
  --model-root ../survey_tools/data/models \
  --allow-local-winner-divergence \
  --allow-boundary-overlap-divergence
```

Use the broader low-density 12-pixel sample before calling a legacy-preserving
Traversal phase complete:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py \
  --sample full \
  --gaia-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --dust-root /Volumes/Data/Galaxy/aosky \
  --model-root ../survey_tools/data/models \
  --allow-local-winner-divergence \
  --allow-boundary-overlap-divergence
```

When validating parallel Traversal changes, run the full comparison through the
`ao-sky` runner with three workers. This has been the required acceptance gate
for the parallel/cache-preserving path:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py \
  --sample full \
  --workers 3 \
  --gaia-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --dust-root /Volumes/Data/Galaxy/aosky \
  --model-root ../survey_tools/data/models \
  --allow-local-winner-divergence \
  --allow-boundary-overlap-divergence
```

Accepted compatibility differences:

- `star_count` and `ngs_count` are not compared against legacy because `ao-sky`
  now derives them from runtime Gaia rows after proper-motion preparation.
- `winner_asterism_id`, `winner_distance_arcsec`, `winner_ee_resolved`,
  `winner_ee_averaged`, `coverage_resolved`, and `coverage_averaged` may differ
  where `ao-sky` intentionally persists only traceable local winners.
- `asterism_count`, `best_ee`, `best_fwhm`, and `best_sr` may differ at the
  boundary under the self-contained two-ring footprint rule.
- Retained local asterism membership and the dense inner row domain should still
  match, apart from the accepted boundary-footprint effects.

For sparse all-sky aggregation comparisons against legacy, use the same helper
with `--maps`:

```bash
./.conda/bin/python scripts/compare_legacy_asterisms.py \
  --maps \
  --sample smoke \
  --gaia-root /Users/nelsonnunes/ao-sky-cache/gaia \
  --dust-root /Volumes/Data/Galaxy/aosky \
  --model-root ../survey_tools/data/models \
  --allow-local-winner-divergence \
  --allow-boundary-overlap-divergence
```

The comparison samples are intentionally low-density and already-loaded. Do not
use crowded or skipped pixels for routine regression checks unless the change is
specifically about density handling, memory guards, or skip behavior.

## Traversal Benchmark Handoff

Use `docs/benchmarking.md` as the benchmark record. This section records how the
latest Traversal benchmark data was produced so another thread can reproduce or
extend it without changing the sample.

The current benchmark script is:

```bash
./.conda/bin/python scripts/benchmark_traversal_baseline.py
```

The script currently uses:

- source config: `/Volumes/Data/Galaxy/aosky/gnao-baseline/ao-sky.yaml`
- benchmark config: a schema-version-2 scratch copy derived from the source
  config for each benchmark build
- Gaia root: `/Users/nelsonnunes/ao-sky-cache/gaia`
- model root:
  `/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/survey_tools/data/models`
- survey root:
  `/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/survey_tools/data/euclid`
- reference sample:
  `/Volumes/Data/Galaxy/aosky/benchmark-runs/ao-sky-compression-sweep-bench-20260416-212643/sample-pixels.ecsv`
- sample shape: `18` complete level-4 regions, `288` level-6 outer pixels
- default worker count: `3`
- default Gaia table cache: `8` prepared tables, `128 MiB`
- default parent memory guard: disabled unless `AO_SKY_BENCH_PARENT_MEMORY_LIMIT_MB`
  is set
- default telemetry mode: `detailed`

Use the same fixed sample for benchmark comparisons. If the sample changes,
record that explicitly in `docs/benchmarking.md`; otherwise cache, SSD, codec,
worker-count, and telemetry measurements are not directly comparable.

Run a small real-data smoke check before launching the full sample:

```bash
AO_SKY_BENCH_SAMPLE_LIMIT=12 \
AO_SKY_BENCH_WORKERS=3 \
AO_SKY_BENCH_TELEMETRY=detailed \
./.conda/bin/python scripts/benchmark_traversal_baseline.py
```

Use `AO_SKY_BENCH_OUTER_PIXS=4432,4435,...` when a specific pixel list is more
useful than the fixed sample prefix.

Run a single baseline refresh with:

```bash
AO_SKY_BENCH_WORKERS=3 \
./.conda/bin/python scripts/benchmark_traversal_baseline.py
```

Run the worker-count sweep used for the current recommendation with:

```bash
AO_SKY_BENCH_WORKERS=3,4,5,6,7,8,9 \
AO_SKY_BENCH_PARENT_MEMORY_LIMIT_MB=26624 \
AO_SKY_BENCH_TELEMETRY=basic \
./.conda/bin/python scripts/benchmark_traversal_baseline.py
```

Use detailed telemetry only when diagnosing performance or memory shape:

```bash
AO_SKY_BENCH_WORKERS=3 \
AO_SKY_BENCH_TELEMETRY=detailed \
./.conda/bin/python scripts/benchmark_traversal_baseline.py
```

Detailed telemetry writes raw traversal diagnostics under each benchmark build's
`diagnostics/` directory and adds more per-pixel profile fields to `build.log`.
Normal production runs should keep telemetry lower because detailed telemetry is
for diagnosis, not throughput.

## Saved Artifact Visual Checks

Use saved-artifact visual checks during Phase 15 physical validation when the
question is whether the new winner-map-first outputs remain spatially sensible,
not whether they are byte-for-byte compatible with legacy outputs.

The saved Phase 13 baseline artifact snapshot is:

```text
/Volumes/Data/Galaxy/aosky/gnao-baseline/phase13
```

This directory is a copied fixed-sample Traversal build from the end of the
Phase 13 optimization pass. It is the best compact artifact source for "where
Phase 13 landed" because it contains the representative 288-pixel sample built
with the retained Phase 13 settings: SSD Gaia, worker-local prepared Gaia cache,
Blosc Zstd artifacts, direct build-tree writes, build-local models, and
build-local dust. It is not an all-sky production build.

Use this snapshot as the legacy-compatible artifact baseline when comparing
later branches or phases against the `legacy-compatible` tag. Match artifacts by
relative path under the build root. For example:

```bash
rel="hpx6-14/0h/+00/18314/outer.h5"
./.conda/bin/python scripts/plot_outer_best_ee_comparison.py \
  "/Volumes/Data/Galaxy/aosky/gnao-baseline/phase13/$rel" \
  "/path/to/new/build/$rel" \
  --left-label "Phase 13 legacy-compatible" \
  --right-label "Current branch" \
  --output /tmp/ao_sky_phase13_current_best_ee_outer_18314.png
```

The snapshot is most useful for inspecting artifact contracts and spatial
behavior on the fixed sample. For branch-level code comparisons, use the
`legacy-compatible` tag as the code baseline and this directory as the saved
artifact baseline.

The true legacy artifact source is:

```text
/Volumes/Data/Galaxy/maps/inner
```

Use this path when the question is "what did the original legacy build write?"
rather than "what did the Phase 13 `ao-sky` legacy-compatible build write?" The
Phase 13 snapshot is the `ao-sky` compatibility baseline; this legacy path is
the source of original `survey_tools` outputs.

The reusable raw best-EE comparison helper is:

```bash
./.conda/bin/python scripts/plot_outer_best_ee_comparison.py \
  /path/to/left/artifact-or-root \
  /path/to/right/artifact-or-root \
  --left-label "Phase 13" \
  --right-label "Phase 14" \
  --outer-pix 18314 \
  --output /tmp/ao_sky_phase13_phase14_best_ee_outer_18314.png
```

The helper supports:

- schema version 1: legacy-preserving Phase 13 artifacts with
  `asterism_count` and `winner_distance_arcsec`
- schema version 2: winner-map-first Phase 14 artifacts without those
  legacy-only columns
- true legacy FITS artifacts under roots such as
  `/Volumes/Data/Galaxy/maps/inner`, where each outer pixel stores an
  `inner.fits` file with the default raw EE column `ASTERISM_EE_MAX_GNAO`

For `outer.h5` inputs, it reads the shared `inner.pix` and `inner.best_ee`
columns. For true legacy `inner.fits` inputs, it reads `PIX` and the configured
legacy EE column. It projects finite inner-pixel samples into local tangent-plane
offsets about the outer-pixel center, interpolates them through a Delaunay
triangulation onto a regular `300 x 300` grid, and writes a three-panel plot
showing left, right, and right-minus-left raw best EE. EE panels use Matplotlib's
`plasma` colormap with a fixed `0.0` to `0.60` scale by default so separate
pixels can be compared visually. The default figure size and DPI are kept modest
for chat and notebook previews; use the script's `--figure-width`,
`--figure-height`, and `--dpi` options when a larger saved image is useful. This
is a human-inspection tool for saved real-sky artifacts; keep deterministic
repo-native correctness checks in `pytest`.

The current full-build operating default on the local workstation is:

- `build.workers: 9`
- `build.scheduler: dynamic`
- `build.memory_limit_mb: 26624`
- `prediction.resolved_device: auto`
- `prediction.averaged_device: auto`
- `build.roots.gaia: /Users/nelsonnunes/ao-sky-cache/gaia`
- artifact writes go directly to the build tree on `/Volumes/Data/Galaxy/aosky`
- derived build artifacts use Blosc Zstd compression
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

Keep reusable migration and benchmark scripts in `scripts/`, but do not promote
every one-off investigation into permanent project surface.

Retained scripts:

- `scripts/compare_legacy_asterisms.py` is the live legacy compatibility
  harness. It is intentionally outside normal `pytest` because it depends on
  local `survey_tools`, Gaia, dust, and model assets, but it is the canonical
  acceptance tool when a phase still claims legacy compatibility.
- `scripts/benchmark_traversal_baseline.py` is the fixed-sample Traversal
  benchmark harness. It owns the current 288-pixel benchmark workflow, parses
  traversal telemetry, and writes reproducible CSV output for
  `docs/benchmarking.md`.
- `scripts/plot_outer_best_ee_comparison.py` is a Phase 15 saved-artifact
  visual inspection helper for comparing raw best-EE maps from true legacy
  `inner.fits` artifacts, v1 `outer.h5` artifacts, and v2 `outer.h5` artifacts.

Do not retain one-off prewarm, write-behind, artifact-staging, compression, or
cache probes as separate scripts unless they become repeated workflows. Their
results belong in `docs/benchmarking.md`; if a probe becomes useful again,
prefer adding a controlled mode to `scripts/benchmark_traversal_baseline.py`
over adding another hardcoded script.

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
