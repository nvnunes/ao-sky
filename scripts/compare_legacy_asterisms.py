#!/usr/bin/env python3
"""Temporary outer-pixel legacy asterism comparison helper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import tempfile
import time
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.table import Table
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LEGACY_CONFIG_FILENAME = REPO_ROOT.parent / "survey_tools" / "aomap" / "config.yaml"
LEGACY_RUNTIME_ROOT = REPO_ROOT.parent / "survey_tools"
GAIA_ROOT = Path("/Volumes/Data/Galaxy/aosky")
DUST_ROOT = LEGACY_RUNTIME_ROOT / "data" / "dust"
MODEL_ROOT = LEGACY_RUNTIME_ROOT / "data" / "models"
GAIA_RELEASE = "dr3"
DEFAULT_AO_SYSTEM = "GNAO"
FLOAT_ATOL = 1e-5
FLOAT_RTOL = 1e-10
SMOKE_SAMPLE_OUTER_PIXS = (
    28559,
    28550,
    28607,
)
FULL_SAMPLE_OUTER_PIXS = (
    28559,
    28550,
    28607,
    28383,
    28463,
    5407,
    28589,
    5380,
    5424,
    28597,
    5448,
    5391,
)
MAPS_SAMPLE_SEED_OUTER_PIXS = (
    1456,
    1717,
    4008,
)
SPECIAL_HOUR_PIXELS = {
    8960,
    8972,
    9023,
    9152,
    9200,
    9203,
    9215,
    11264,
    11312,
    11327,
    11468,
}

COMPARISON_COLUMNS = (
    "id",
    "ra",
    "dec",
    "num_stars",
    "star1_id",
    "star1_ra",
    "star1_dec",
    "star1_mag",
    "star2_id",
    "star2_ra",
    "star2_dec",
    "star2_mag",
    "star3_id",
    "star3_ra",
    "star3_dec",
    "star3_mag",
    "pix",
)

INNER_COMPARISON_COLUMNS = (
    "pix",
    "gaia_A0",
    "star_count",
    "ngs_count",
    "asterism_count",
    "best_ee",
    "best_sr",
    "best_fwhm",
    "winner_asterism_id",
    "winner_distance_arcsec",
    "winner_ee_resolved",
    "winner_ee_averaged",
    "coverage_resolved",
    "coverage_averaged",
)
LOCAL_WINNER_DIVERGENCE_COLUMNS = frozenset(
    {
        "winner_asterism_id",
        "winner_distance_arcsec",
        "winner_ee_resolved",
        "winner_ee_averaged",
        "coverage_resolved",
        "coverage_averaged",
    }
)
BOUNDARY_OVERLAP_DIVERGENCE_COLUMNS = frozenset(
    {
        "asterism_count",
        "best_sr",
        "best_ee",
        "best_fwhm",
    }
)

NEW_TO_LEGACY_COLUMNS = {
    "asterism_id": "id",
    "star1_source_id": "star1_id",
    "star1_ref_epoch": "star1_pmepoch",
    "star2_source_id": "star2_id",
    "star2_ref_epoch": "star2_pmepoch",
    "star3_source_id": "star3_id",
    "star3_ref_epoch": "star3_pmepoch",
    "radius_arcsec": "radius",
    "area_arcsec2": "area",
    "relative_area": "relarea",
    "separation_arcsec": "separation",
    "relative_separation": "relsep",
}

EMPTY_LEGACY_COLUMN_DTYPES = {
    "id": np.int64,
    "ra": np.float64,
    "dec": np.float64,
    "num_stars": np.int64,
    "star1_id": np.int64,
    "star1_ra": np.float64,
    "star1_dec": np.float64,
    "star1_mag": np.float64,
    "star2_id": np.int64,
    "star2_ra": np.float64,
    "star2_dec": np.float64,
    "star2_mag": np.float64,
    "star3_id": np.int64,
    "star3_ra": np.float64,
    "star3_dec": np.float64,
    "star3_mag": np.float64,
    "pix": np.int64,
}

GaiaHealpixStore = None
GaiaStoreConfig = None
aggregate_maps = None
init_build = None
read_outer_dataset = None
run_traversal_phase = None
resolve_traversal_execution_config = None
BuildDefinition = None
build_traversal_products = None
load_live_legacy_traversal_outputs = None
load_native_runtime = None
RuntimeGaiaHealpixStore = None
maps_artifact_filename = None
outer_artifact_filename = None
sample_gaia_a0_for_outer_pixel = None
write_outer_artifact = None
BUILD_FILENAME = None
OUTER_DATASET_ASTERISMS = None
OUTER_DATASET_INNER = None
WORK_STATUS_DONE = None
WORK_STATUS_PENDING = None
get_parent_pixel = None
get_subpixels = None

MAPS_COMPARISON_COLUMNS = (
    "gaia_A0",
    "star_count",
    "ngs_count",
    "asterism_count",
    "best_sr",
    "best_ee",
    "best_fwhm",
    "winner_ee_resolved",
    "coverage_resolved",
    "coverage_averaged",
)


def _load_runtime() -> None:
    global GaiaHealpixStore, GaiaStoreConfig
    global BuildDefinition
    global aggregate_maps, init_build, read_outer_dataset, run_traversal_phase
    global resolve_traversal_execution_config
    global build_traversal_products, load_live_legacy_traversal_outputs, load_native_runtime
    global RuntimeGaiaHealpixStore
    global maps_artifact_filename, outer_artifact_filename, sample_gaia_a0_for_outer_pixel
    global write_outer_artifact
    global BUILD_FILENAME, OUTER_DATASET_ASTERISMS, OUTER_DATASET_INNER
    global WORK_STATUS_DONE, WORK_STATUS_PENDING
    global get_parent_pixel
    global get_subpixels

    if GaiaHealpixStore is not None:
        return

    from ao_sky.build import init_build as _init_build
    from ao_sky.build.aggregation import aggregate_maps as _aggregate_maps
    from ao_sky.build._constants import (
        BUILD_FILENAME as _BUILD_FILENAME,
        OUTER_DATASET_ASTERISMS as _OUTER_DATASET_ASTERISMS,
        OUTER_DATASET_INNER as _OUTER_DATASET_INNER,
        WORK_STATUS_DONE as _WORK_STATUS_DONE,
        WORK_STATUS_PENDING as _WORK_STATUS_PENDING,
    )
    from ao_sky.build._models import BuildDefinition as _BuildDefinition
    from ao_sky.build.artifacts import (
        read_outer_dataset as _read_outer_dataset,
        write_outer_artifact as _write_outer_artifact,
    )
    from ao_sky.build.control import (
        maps_artifact_filename as _maps_artifact_filename,
        outer_artifact_filename as _outer_artifact_filename,
    )
    from ao_sky.build.legacy_config import load_native_runtime as _load_native_runtime
    from ao_sky.build.legacy_runtime import (
        load_live_legacy_traversal_outputs as _load_live_legacy_traversal_outputs,
    )
    from ao_sky.build.config import (
        resolve_traversal_execution_config as _resolve_traversal_execution_config,
    )
    from ao_sky.build.runner import _run_traversal_phase as _run_traversal_phase
    from ao_sky.build.runtime_gaia import RuntimeGaiaHealpixStore as _RuntimeGaiaHealpixStore
    from ao_sky.build.traversal import build_traversal_products as _build_traversal_products
    from ao_sky.dust import sample_gaia_a0_for_outer_pixel as _sample_gaia_a0_for_outer_pixel
    from ao_sky.gaia import (
        GaiaHealpixStore as _GaiaHealpixStore,
        GaiaStoreConfig as _GaiaStoreConfig,
    )
    from ao_sky.spatial import (
        get_parent_pixel as _get_parent_pixel,
        get_subpixels as _get_subpixels,
    )

    GaiaHealpixStore = _GaiaHealpixStore
    GaiaStoreConfig = _GaiaStoreConfig
    aggregate_maps = _aggregate_maps
    init_build = _init_build
    read_outer_dataset = _read_outer_dataset
    run_traversal_phase = _run_traversal_phase
    resolve_traversal_execution_config = _resolve_traversal_execution_config
    BuildDefinition = _BuildDefinition
    build_traversal_products = _build_traversal_products
    load_live_legacy_traversal_outputs = _load_live_legacy_traversal_outputs
    load_native_runtime = _load_native_runtime
    RuntimeGaiaHealpixStore = _RuntimeGaiaHealpixStore
    maps_artifact_filename = _maps_artifact_filename
    outer_artifact_filename = _outer_artifact_filename
    sample_gaia_a0_for_outer_pixel = _sample_gaia_a0_for_outer_pixel
    write_outer_artifact = _write_outer_artifact
    BUILD_FILENAME = _BUILD_FILENAME
    OUTER_DATASET_ASTERISMS = _OUTER_DATASET_ASTERISMS
    OUTER_DATASET_INNER = _OUTER_DATASET_INNER
    WORK_STATUS_DONE = _WORK_STATUS_DONE
    WORK_STATUS_PENDING = _WORK_STATUS_PENDING
    get_parent_pixel = _get_parent_pixel
    get_subpixels = _get_subpixels


@dataclass(frozen=True, slots=True)
class LegacyAOSystem:
    name: str
    band: str
    fov: u.Quantity
    fov_1ngs: u.Quantity
    min_wfs: int
    max_wfs: int
    min_mag: float
    nom_mag: float
    max_mag: float
    min_sep: u.Quantity
    max_sep: u.Quantity
    min_rel_sep: float
    max_rel_sep: float
    min_rel_area: float
    max_rel_area: float


@dataclass(frozen=True, slots=True)
class LegacyComparisonConfig:
    outer_level: int
    inner_level: int
    max_data_level: int
    asterism_epoch: float | None
    asterisms_min_galactic_latitude: float
    asterisms_galactic_latitude_bypass_pixs: tuple[int, ...]
    asterisms_max_star_density: float | None
    asterisms_max_bright_star_mag: float | None
    asterisms_max_overlap: float | None
    legacy_config_filename: Path
    model_root: Path
    ao_system: LegacyAOSystem


def _load_live_legacy_dust_values(
    *,
    config: LegacyComparisonConfig,
    outer_pix: int,
    dust_root: Path,
) -> np.ndarray:
    _load_runtime()
    return sample_gaia_a0_for_outer_pixel(
        dust_root=dust_root,
        outer_level=config.outer_level,
        outer_pix=outer_pix,
        inner_level=config.inner_level,
        max_data_level=config.max_data_level,
    )


def _add_live_legacy_dust_field(
    inner: Table,
    *,
    config: LegacyComparisonConfig,
    outer_pix: int,
    dust_root: Path,
) -> Table:
    result = inner.copy(copy_data=True)
    values = _load_live_legacy_dust_values(
        config=config,
        outer_pix=outer_pix,
        dust_root=dust_root,
    )
    if len(values) != len(result):
        raise RuntimeError(
            f"Legacy dust length {len(values)} does not match inner rows {len(result)}"
        )
    if "gaia_A0" in result.colnames:
        result["gaia_A0"] = values
    else:
        result.add_column(np.asarray(values, dtype=np.float64), name="gaia_A0", index=1)
    return result


def load_live_legacy_outputs(
    *,
    release: str,
    config: LegacyComparisonConfig,
    outer_pix: int,
    dust_root: Path,
) -> tuple[Table, Table]:
    """Run the live legacy outer-pixel path and return asterism and inner outputs."""
    runtime = _build_runtime(release=release, config=config)
    legacy_asterisms, inner = load_live_legacy_traversal_outputs(runtime, outer_pix)
    inner = _add_live_legacy_dust_field(
        inner,
        config=config,
        outer_pix=outer_pix,
        dust_root=dust_root,
    )
    if legacy_asterisms is None:
        legacy_asterisms = _empty_legacy_asterism_table()
    return legacy_asterisms, inner


def select_random_outer_pix(
    *,
    outer_level: int,
) -> int:
    """Select a random outer pixel id for the configured HEALPix level."""
    num_pixels = 12 * (4**outer_level)
    rng = np.random.default_rng()
    return int(rng.integers(0, num_pixels))


def _get_membership_key(table: Table, row_index: int) -> str:
    star_ids = [
        int(table["star1_id"][row_index]),
        int(table["star2_id"][row_index]),
        int(table["star3_id"][row_index]),
    ]
    valid_ids = sorted(star_id for star_id in star_ids if star_id >= 0)
    return "-".join(str(star_id) for star_id in valid_ids)


def _empty_legacy_asterism_table() -> Table:
    return Table(
        [np.array([], dtype=EMPTY_LEGACY_COLUMN_DTYPES[name]) for name in COMPARISON_COLUMNS],
        names=COMPARISON_COLUMNS,
    )


def load_legacy_config(filename: Path, ao_system_name: str) -> LegacyComparisonConfig:
    with filename.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    systems = raw.get("ao_systems", [])
    system = next((item for item in systems if item["name"] == ao_system_name), None)
    if system is None:
        raise ValueError(f"AO system {ao_system_name!r} not found in {filename}")

    fov_1ngs = system.get("fov_1ngs", system["fov"])
    ao_system = LegacyAOSystem(
        name=system["name"],
        band=system["band"],
        fov=float(system["fov"]) * u.arcsec,
        fov_1ngs=float(fov_1ngs) * u.arcsec,
        min_wfs=int(system["min_wfs"]),
        max_wfs=int(system["max_wfs"]),
        min_mag=float(system["min_mag"]),
        nom_mag=float(system.get("nom_mag", system["max_mag"])),
        max_mag=float(system["max_mag"]),
        min_sep=float(system["min_sep"]) * u.arcsec,
        max_sep=float(system["max_sep"]) * u.arcsec,
        min_rel_sep=float(system.get("min_rel_sep", 0.0)),
        max_rel_sep=float(system.get("max_rel_sep", 0.0)),
        min_rel_area=float(system.get("min_rel_area", 0.0)),
        max_rel_area=float(system.get("max_rel_area", 0.0)),
    )

    bypass_pixs = tuple(int(pix) for pix in raw.get("asterisms_galactic_latitude_bypass_pixs", []))
    return LegacyComparisonConfig(
        outer_level=int(raw["outer_level"]),
        inner_level=int(raw["inner_level"]),
        max_data_level=int(raw["max_data_level"]),
        asterism_epoch=(
            None if raw.get("asterism_epoch") is None else float(raw["asterism_epoch"])
        ),
        asterisms_min_galactic_latitude=float(raw.get("asterisms_min_galactic_latitude", 20.0)),
        asterisms_galactic_latitude_bypass_pixs=bypass_pixs,
        asterisms_max_star_density=(
            None
            if raw.get("asterisms_max_star_density") is None
            else float(raw["asterisms_max_star_density"])
        ),
        asterisms_max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        asterisms_max_overlap=(
            None
            if raw.get("asterisms_max_overlap") is None
            else float(raw["asterisms_max_overlap"])
        ),
        legacy_config_filename=filename.resolve(),
        model_root=MODEL_ROOT.resolve(),
        ao_system=ao_system,
    )


def _build_runtime(
    *,
    release: str,
    config: LegacyComparisonConfig,
) -> object:
    _load_runtime()
    definition = BuildDefinition(
        ao_system_short_name=config.ao_system.name,
        config_short_name="legacy-comparison",
        gaia_release=release,
        outer_level=config.outer_level,
        inner_level=config.inner_level,
        max_data_level=config.max_data_level,
        epoch=config.asterism_epoch if config.asterism_epoch is not None else 2016.0,
        min_galactic_latitude=config.asterisms_min_galactic_latitude,
    )
    return load_native_runtime(
        definition,
        legacy_config_path=config.legacy_config_filename,
        model_root=config.model_root,
    )


def build_new_outputs(
    *,
    gaia_root: Path,
    dust_root: Path,
    release: str,
    config: LegacyComparisonConfig,
    outer_pix: int,
) -> tuple[Table, Table]:
    _load_runtime()
    base_store = GaiaHealpixStore(
        GaiaStoreConfig(root=gaia_root, release=release, healpix_level=config.outer_level)
    )
    runtime = _build_runtime(release=release, config=config)
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=0,
        max_bytes=0,
    )
    asterisms, inner = build_traversal_products(
        store,
        runtime,
        outer_pix,
        dust_root=dust_root,
        max_data_level=config.max_data_level,
    )
    return asterisms, inner


def _write_subset_traversal_state(build_path: Path, outer_pixs: list[int]) -> None:
    selected = {int(outer_pix) for outer_pix in outer_pixs}
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["state"]["outer_pixels"]
        state = dataset[...]
        # Sparse comparison builds intentionally materialize only selected
        # traversal artifacts and call the Traversal phase directly, so
        # unselected rows can stay terminal without triggering aggregation.
        state["traversal_status"] = WORK_STATUS_DONE
        state["traversal_attempt_count"] = 0
        state["traversal_last_error_message"] = b""
        for outer_pix in selected:
            state["traversal_status"][outer_pix] = WORK_STATUS_PENDING
        dataset[...] = state


def build_new_outputs_with_runner(
    *,
    gaia_root: Path,
    dust_root: Path,
    release: str,
    config: LegacyComparisonConfig,
    outer_pixs: list[int],
    workers: int,
    gaia_cache_entries: int | None,
    gaia_cache_mb: int | None,
    region_level: int | None,
) -> dict[int, tuple[Table, Table]]:
    _load_runtime()
    with tempfile.TemporaryDirectory(prefix="ao-sky-traversal-new-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        definition_filename = _write_temporary_build_definition(
            tmpdir_path / "build.yaml",
            config=config,
            release=release,
        )
        definition = BuildDefinition(
            ao_system_short_name=config.ao_system.name,
            config_short_name="legacy-traversal",
            gaia_release=release,
            outer_level=config.outer_level,
            inner_level=config.inner_level,
            max_data_level=config.max_data_level,
            epoch=config.asterism_epoch if config.asterism_epoch is not None else 2016.0,
            min_galactic_latitude=config.asterisms_min_galactic_latitude,
        )
        build_path = init_build(
            definition_filename=definition_filename,
            gaia_root=gaia_root,
            build_root=tmpdir_path / "builds",
            dust_root=dust_root,
            legacy_config_path=config.legacy_config_filename,
            model_root=config.model_root,
        )
        _write_subset_traversal_state(build_path, outer_pixs)
        print(
            "ao-sky runner "
            f"workers={workers} "
            f"gaia_cache_entries={gaia_cache_entries} "
            f"gaia_cache_mb={gaia_cache_mb} "
            f"region_level={region_level}"
        )
        execution_config = resolve_traversal_execution_config(
            outer_level=config.outer_level,
            workers=workers,
            gaia_cache_entries=gaia_cache_entries,
            gaia_cache_mb=gaia_cache_mb,
            region_level=region_level,
        )
        start_time = time.perf_counter()
        traversal_complete, traversal_counts = run_traversal_phase(
            build_path,
            execution_config=execution_config,
        )
        elapsed = time.perf_counter() - start_time
        print(f"ao-sky runner elapsed: {elapsed:.3f} s")
        if not traversal_complete:
            log_filename = build_path / "build.log"
            if log_filename.exists():
                print(log_filename.read_text(encoding="utf-8"))
            raise RuntimeError(f"ao-sky traversal failed: {traversal_counts}")

        outputs: dict[int, tuple[Table, Table]] = {}
        for outer_pix in outer_pixs:
            filename = outer_artifact_filename(build_path, definition, outer_pix)
            outputs[int(outer_pix)] = (
                read_outer_dataset(filename, OUTER_DATASET_ASTERISMS),
                read_outer_dataset(filename, OUTER_DATASET_INNER),
            )
        return outputs


def prepare_legacy_table(table: Table) -> Table:
    missing = [name for name in COMPARISON_COLUMNS if name not in table.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Legacy table is missing required comparison columns: {missing_text}")
    return table[list(COMPARISON_COLUMNS)].copy(copy_data=True)


def prepare_new_table(table: Table) -> Table:
    working = table.copy(copy_data=True)
    for source_name, legacy_name in NEW_TO_LEGACY_COLUMNS.items():
        if source_name in working.colnames:
            working.rename_column(source_name, legacy_name)
    for column_name in ("star1_idx", "star2_idx", "star3_idx"):
        if column_name in working.colnames:
            working.remove_column(column_name)
    missing = [name for name in COMPARISON_COLUMNS if name not in working.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"New table is missing required comparison columns: {missing_text}")
    return working[list(COMPARISON_COLUMNS)].copy(copy_data=True)


def prepare_inner_table(table: Table) -> Table:
    missing = [name for name in INNER_COMPARISON_COLUMNS if name not in table.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Inner table is missing required comparison columns: {missing_text}")
    return table[list(INNER_COMPARISON_COLUMNS)].copy(copy_data=True)


def compare_tables(
    legacy: Table,
    new: Table,
    *,
    allow_membership_divergence: bool = False,
) -> int:
    legacy_keys = np.asarray([_get_membership_key(legacy, index) for index in range(len(legacy))])
    new_keys = np.asarray([_get_membership_key(new, index) for index in range(len(new))])

    missing_from_new = sorted(set(legacy_keys) - set(new_keys))
    extra_in_new = sorted(set(new_keys) - set(legacy_keys))

    print(f"legacy rows: {len(legacy)}")
    print(f"new rows:    {len(new)}")
    print(f"shared keys: {len(set(legacy_keys) & set(new_keys))}")
    print(f"missing:     {len(missing_from_new)}")
    print(f"extra:       {len(extra_in_new)}")

    if missing_from_new:
        print("missing keys:", ", ".join(missing_from_new[:10]))
    if extra_in_new:
        print("extra keys:", ", ".join(extra_in_new[:10]))

    if len(legacy) == 0 and len(new) == 0:
        print("row ordering: exact membership order match")
        return 0

    common_keys = sorted(set(legacy_keys) & set(new_keys))
    if not common_keys:
        return 1

    legacy_index = {key: int(np.flatnonzero(legacy_keys == key)[0]) for key in common_keys}
    new_index = {key: int(np.flatnonzero(new_keys == key)[0]) for key in common_keys}

    mismatch_count = 0
    for column_name in COMPARISON_COLUMNS:
        if column_name == "id":
            continue

        column_mismatch = 0
        max_abs_diff = 0.0
        for key in common_keys:
            legacy_value = legacy[column_name][legacy_index[key]]
            new_value = new[column_name][new_index[key]]
            if np.issubdtype(np.asarray([legacy_value]).dtype, np.floating):
                if np.isnan(legacy_value) and np.isnan(new_value):
                    continue
                if np.isclose(legacy_value, new_value, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
                    max_abs_diff = max(max_abs_diff, float(abs(legacy_value - new_value)))
                    continue
                max_abs_diff = max(max_abs_diff, float(abs(legacy_value - new_value)))
            elif legacy_value == new_value:
                continue

            column_mismatch += 1

        if column_mismatch > 0:
            mismatch_count += column_mismatch
            print(
                f"column mismatch: {column_name} mismatches={column_mismatch} "
                f"max_abs_diff={max_abs_diff:.6g}"
            )

    legacy_ordered = np.asarray([legacy_index[key] for key in common_keys], dtype=np.int64) + 1
    new_ordered = np.asarray([new_index[key] for key in common_keys], dtype=np.int64) + 1
    if np.array_equal(legacy_ordered, new_ordered):
        print("row ordering: exact membership order match")
    else:
        print("row ordering: differs")

    membership_status = bool(missing_from_new or extra_in_new)
    if membership_status and allow_membership_divergence:
        print("membership differences accepted: boundary overlap footprint divergence")
        membership_status = False

    return 1 if membership_status or mismatch_count else 0


def compare_inner_tables(
    legacy: Table,
    new: Table,
    *,
    ignored_columns: frozenset[str] = frozenset(),
) -> int:
    print(f"legacy inner rows: {len(legacy)}")
    print(f"new inner rows:    {len(new)}")

    if len(legacy) != len(new):
        print("inner row count differs")
        return 1

    mismatch_count = 0
    for column_name in INNER_COMPARISON_COLUMNS:
        if column_name in ignored_columns:
            continue
        legacy_values = np.asarray(legacy[column_name])
        new_values = np.asarray(new[column_name])
        if legacy_values.dtype.kind == "f" or new_values.dtype.kind == "f":
            equal = np.isclose(
                legacy_values,
                new_values,
                atol=FLOAT_ATOL,
                rtol=FLOAT_RTOL,
                equal_nan=True,
            )
        else:
            equal = legacy_values == new_values
        if np.all(equal):
            continue
        mismatches = int(np.count_nonzero(~equal))
        mismatch_count += mismatches
        print(f"inner column mismatch: {column_name} mismatches={mismatches}")

    if mismatch_count == 0:
        print("inner row ordering: exact match")
        return 0

    return 1


def _legacy_field_from_key(key: str) -> str:
    return key.replace("-", "_").upper()


def _legacy_map_column_name(column_name: str, config: LegacyComparisonConfig) -> str:
    ao_field = _legacy_field_from_key(config.ao_system.name)
    mapping = {
        "gaia_A0": "DUST_EXTINCTION",
        "star_count": "STAR_COUNT",
        "ngs_count": f"NGS_COUNT_{ao_field}",
        "asterism_count": f"ASTERISM_COUNT_{ao_field}",
        "best_sr": f"ASTERISM_SR_MAX_{ao_field}",
        "best_ee": f"ASTERISM_EE_MAX_{ao_field}",
        "best_fwhm": f"ASTERISM_FWHM_MIN_{ao_field}",
        "winner_ee_resolved": f"ASTERISM_EE_MAX_{ao_field}",
        "coverage_resolved": f"ASTERISM_COVERAGE_{ao_field}_RESOLVED",
        "coverage_averaged": f"ASTERISM_COVERAGE_{ao_field}_MEAN",
    }
    return mapping[column_name]


def _select_map_outer_pix_groups(
    *,
    outer_pixs: list[int],
    sample: str,
    outer_level: int,
) -> list[list[int]]:
    _load_runtime()
    if outer_pixs:
        return [[int(outer_pix) for outer_pix in outer_pixs]]
    if outer_level == 0:
        raise ValueError("Map comparison requires outer_level > 0 to build 4-pixel parent groups")

    target_groups = 1 if sample == "smoke" else 3 if sample == "full" else 1
    candidate_outer_pixs = (
        [select_random_outer_pix(outer_level=outer_level)] if sample == "random" else
        list(MAPS_SAMPLE_SEED_OUTER_PIXS)
    )
    groups: list[list[int]] = []
    seen_parents: set[int] = set()
    for outer_pix in candidate_outer_pixs:
        parent_pix = int(get_parent_pixel(outer_level, int(outer_pix), outer_level - 1))
        if parent_pix in seen_parents:
            continue
        seen_parents.add(parent_pix)
        groups.append([int(pix) for pix in get_subpixels(outer_level - 1, parent_pix, outer_level)])
        if len(groups) == target_groups:
            break
    if not groups:
        raise ValueError("Could not derive sparse map comparison groups")
    print(
        "selected sparse map groups:",
        " | ".join(" ".join(str(pix) for pix in group) for group in groups),
    )
    return groups


def _write_temporary_build_definition(
    filename: Path,
    *,
    config: LegacyComparisonConfig,
    release: str,
) -> Path:
    filename.write_text(
        "\n".join(
            (
                f"ao_system_short_name: {config.ao_system.name}",
                "config_short_name: legacy-maps",
                f"gaia_release: {release}",
                f"outer_level: {config.outer_level}",
                f"inner_level: {config.inner_level}",
                f"max_data_level: {config.max_data_level}",
                f"epoch: {config.asterism_epoch if config.asterism_epoch is not None else 2016.0}",
                (
                    ""
                    if config.asterisms_min_galactic_latitude is None
                    else f"min_galactic_latitude: {config.asterisms_min_galactic_latitude}"
                ),
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return filename


def _build_sparse_new_maps(
    *,
    gaia_root: Path,
    dust_root: Path,
    release: str,
    config: LegacyComparisonConfig,
    outer_pixs: list[int],
) -> dict[int, Table]:
    _load_runtime()
    with tempfile.TemporaryDirectory(prefix="ao-sky-maps-new-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        definition_filename = _write_temporary_build_definition(
            tmpdir_path / "build.yaml",
            config=config,
            release=release,
        )
        definition = BuildDefinition(
            ao_system_short_name=config.ao_system.name,
            config_short_name="legacy-maps",
            gaia_release=release,
            outer_level=config.outer_level,
            inner_level=config.inner_level,
            max_data_level=config.max_data_level,
            epoch=config.asterism_epoch if config.asterism_epoch is not None else 2016.0,
            min_galactic_latitude=config.asterisms_min_galactic_latitude,
        )
        build_path = init_build(
            definition_filename=definition_filename,
            gaia_root=gaia_root,
            build_root=tmpdir_path / "builds",
            dust_root=dust_root,
            legacy_config_path=config.legacy_config_filename,
        )
        for outer_pix in outer_pixs:
            asterisms, inner = build_new_outputs(
                gaia_root=gaia_root,
                dust_root=dust_root,
                release=release,
                config=config,
                outer_pix=outer_pix,
            )
            write_outer_artifact(
                outer_artifact_filename(build_path, definition, outer_pix),
                inner=inner,
                asterisms=asterisms,
            )
        return {level: Table(values) for level, values in aggregate_maps(build_path, outer_pixs=outer_pixs).items()}


def _build_sparse_legacy_maps(
    *,
    release: str,
    config: LegacyComparisonConfig,
    outer_pixs: list[int],
    dust_root: Path,
) -> dict[int, Table]:
    with tempfile.TemporaryDirectory(prefix="ao-sky-maps-legacy-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        definition_filename = _write_temporary_build_definition(
            tmpdir_path / "build.yaml",
            config=config,
            release=release,
        )
        definition = BuildDefinition(
            ao_system_short_name=config.ao_system.name,
            config_short_name="legacy-maps",
            gaia_release=release,
            outer_level=config.outer_level,
            inner_level=config.inner_level,
            max_data_level=config.max_data_level,
            epoch=config.asterism_epoch if config.asterism_epoch is not None else 2016.0,
            min_galactic_latitude=config.asterisms_min_galactic_latitude,
        )
        build_path = init_build(
            definition_filename=definition_filename,
            gaia_root=GAIA_ROOT,
            build_root=tmpdir_path / "builds",
            dust_root=dust_root,
            legacy_config_path=config.legacy_config_filename,
        )
        for outer_pix in outer_pixs:
            legacy_asterisms, legacy_inner = load_live_legacy_outputs(
                release=release,
                config=config,
                outer_pix=outer_pix,
                dust_root=dust_root,
            )
            del legacy_asterisms
            write_outer_artifact(
                outer_artifact_filename(build_path, definition, outer_pix),
                inner=legacy_inner,
                asterisms=None,
            )
        return {level: Table(values) for level, values in aggregate_maps(build_path, outer_pixs=outer_pixs).items()}


def _touched_level_pixs(
    *,
    outer_pixs: list[int],
    outer_level: int,
    level: int,
) -> np.ndarray:
    _load_runtime()
    if level == outer_level:
        return np.asarray(sorted(int(pix) for pix in outer_pixs), dtype=np.int64)
    return np.asarray(get_subpixels(outer_level, outer_pixs, level), dtype=np.int64)


def _compare_sparse_maps_for_group(
    *,
    outer_pixs: list[int],
    gaia_root: Path,
    dust_root: Path,
    release: str,
    config: LegacyComparisonConfig,
) -> int:
    _load_runtime()
    print("map outer pixels:", " ".join(str(pix) for pix in outer_pixs))
    legacy_tables = _build_sparse_legacy_maps(
        release=release,
        config=config,
        outer_pixs=outer_pixs,
        dust_root=dust_root,
    )
    new_tables = _build_sparse_new_maps(
        gaia_root=gaia_root,
        dust_root=dust_root,
        release=release,
        config=config,
        outer_pixs=outer_pixs,
    )

    mismatches = 0
    for level in range(config.outer_level, config.max_data_level + 1):
        touched_pixs = _touched_level_pixs(
            outer_pixs=outer_pixs,
            outer_level=config.outer_level,
            level=level,
        )
        legacy = legacy_tables[level]
        new = new_tables[level]
        print(f"level {level}: touched pixels={len(touched_pixs)}")
        for column_name in MAPS_COMPARISON_COLUMNS:
            legacy_name = _legacy_map_column_name(column_name, config)
            source_name = legacy_name if legacy_name in legacy.colnames else column_name
            legacy_values = np.asarray(legacy[source_name])[touched_pixs]
            new_values = np.asarray(new[column_name])[touched_pixs]
            equal = np.isclose(
                legacy_values,
                new_values,
                atol=FLOAT_ATOL,
                rtol=FLOAT_RTOL,
                equal_nan=True,
            )
            if np.all(equal):
                continue
            mismatches += int(np.count_nonzero(~equal))
            print(f"map column mismatch: level={level} column={column_name} mismatches={int(np.count_nonzero(~equal))}")
    print(f"map result: {'FAIL' if mismatches else 'OK'}")
    return 1 if mismatches else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare live legacy outer-pixel results against the new in-memory path.",
    )
    parser.add_argument(
        "outer_pix",
        nargs="*",
        type=int,
        help="Outer HEALPix pixel id(s) to compare. If omitted, use the selected sample set.",
    )
    parser.add_argument(
        "--sample",
        choices=("smoke", "full", "random"),
        default="smoke",
        help=(
            "Named sample set to use when no explicit outer_pix values are given. "
            "'smoke' avoids the slower pathological pixels used only in broader checks."
        ),
    )
    parser.add_argument("--ao-system", default=DEFAULT_AO_SYSTEM, help="Legacy AO system name.")
    parser.add_argument(
        "--config",
        type=Path,
        default=LEGACY_CONFIG_FILENAME,
        help="Legacy aomap config.yaml to read comparison settings from.",
    )
    parser.add_argument(
        "--gaia-root",
        type=Path,
        default=GAIA_ROOT,
        help="Canonical ao-sky Gaia root used by the new code path.",
    )
    parser.add_argument(
        "--dust-root",
        type=Path,
        default=DUST_ROOT,
        help="Dust root containing the Gaia TGE dustmaps data.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
        help="AO model root containing the temporary girmos-aosims models.",
    )
    parser.add_argument(
        "--release",
        default=GAIA_RELEASE,
        help="Gaia release identifier for the canonical ao-sky store.",
    )
    parser.add_argument(
        "--maps",
        action="store_true",
        help="Compare sparse all-sky aggregated maps instead of per-outer-pixel traversal outputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of ao-sky Traversal workers to use for non-map comparisons.",
    )
    parser.add_argument(
        "--gaia-cache-entries",
        type=int,
        default=None,
        help="Worker-local Gaia table cache entry target for ao-sky runner comparisons.",
    )
    parser.add_argument(
        "--gaia-cache-mb",
        type=int,
        default=None,
        help="Worker-local Gaia table cache memory cap in MiB for ao-sky runner comparisons.",
    )
    parser.add_argument(
        "--region-level",
        type=int,
        default=None,
        help="Regional HEALPix level for ao-sky runner comparisons.",
    )
    parser.add_argument(
        "--allow-local-winner-divergence",
        action="store_true",
        help=(
            "Ignore inner winner/coverage columns that intentionally diverge "
            "because ao-sky only persists traceable local winners."
        ),
    )
    parser.add_argument(
        "--allow-boundary-overlap-divergence",
        action="store_true",
        help=(
            "Accept boundary asterism membership and best-map differences caused "
            "by ao-sky's self-contained two-ring overlap/proper-motion buffer footprint."
        ),
    )
    return parser.parse_args()


def _select_outer_pixs(
    *,
    outer_pixs: list[int],
    sample: str,
    outer_level: int,
) -> list[int]:
    if outer_pixs:
        return [int(outer_pix) for outer_pix in outer_pixs]
    if sample == "random":
        outer_pix = select_random_outer_pix(outer_level=outer_level)
        print(f"selected random outer pixel: {outer_pix}")
        return [outer_pix]
    if sample == "full":
        print("selected full comparison sample:", " ".join(str(pix) for pix in FULL_SAMPLE_OUTER_PIXS))
        return list(FULL_SAMPLE_OUTER_PIXS)
    print("selected smoke comparison sample:", " ".join(str(pix) for pix in SMOKE_SAMPLE_OUTER_PIXS))
    return list(SMOKE_SAMPLE_OUTER_PIXS)


def _compare_outer_pixel(
    *,
    outer_pix: int,
    gaia_root: Path,
    dust_root: Path,
    release: str,
    config: LegacyComparisonConfig,
    new_outputs: dict[int, tuple[Table, Table]] | None = None,
    ignored_inner_columns: frozenset[str] = frozenset(),
    allow_asterism_membership_divergence: bool = False,
) -> int:
    print(f"outer pixel: {outer_pix}")
    legacy_asterisms, legacy_inner = load_live_legacy_outputs(
        release=release,
        config=config,
        outer_pix=outer_pix,
        dust_root=dust_root,
    )
    legacy_table = prepare_legacy_table(legacy_asterisms)
    legacy_inner = prepare_inner_table(legacy_inner)
    legacy_label = (
        "live legacy code: "
        f"{LEGACY_RUNTIME_ROOT / 'aomap'} using {config.legacy_config_filename}"
    )

    if new_outputs is None:
        new_table, new_inner = build_new_outputs(
            gaia_root=gaia_root,
            dust_root=dust_root,
            release=release,
            config=config,
            outer_pix=outer_pix,
        )
    else:
        new_table, new_inner = new_outputs[int(outer_pix)]
    prepared_new = prepare_new_table(new_table)
    prepared_new_inner = prepare_inner_table(new_inner)

    print(legacy_label)
    print("asterisms:")
    asterism_status = compare_tables(
        legacy_table,
        prepared_new,
        allow_membership_divergence=allow_asterism_membership_divergence,
    )
    print("inner:")
    inner_status = compare_inner_tables(
        legacy_inner,
        prepared_new_inner,
        ignored_columns=ignored_inner_columns,
    )
    status = 1 if asterism_status or inner_status else 0
    print(f"result: {'FAIL' if status else 'OK'}")
    return status


def main() -> int:
    args = parse_args()
    _load_runtime()
    dust_root = args.dust_root.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    config = load_legacy_config(args.config, args.ao_system)
    config = LegacyComparisonConfig(
        outer_level=config.outer_level,
        inner_level=config.inner_level,
        max_data_level=config.max_data_level,
        asterism_epoch=config.asterism_epoch,
        asterisms_min_galactic_latitude=config.asterisms_min_galactic_latitude,
        asterisms_galactic_latitude_bypass_pixs=config.asterisms_galactic_latitude_bypass_pixs,
        asterisms_max_star_density=config.asterisms_max_star_density,
        asterisms_max_bright_star_mag=config.asterisms_max_bright_star_mag,
        asterisms_max_overlap=config.asterisms_max_overlap,
        legacy_config_filename=config.legacy_config_filename,
        model_root=model_root,
        ao_system=config.ao_system,
    )
    if args.maps:
        groups = _select_map_outer_pix_groups(
            outer_pixs=args.outer_pix,
            sample=args.sample,
            outer_level=config.outer_level,
        )
        failures = 0
        for index, group in enumerate(groups):
            if index > 0:
                print()
            failures += _compare_sparse_maps_for_group(
                outer_pixs=group,
                gaia_root=args.gaia_root,
                dust_root=dust_root,
                release=args.release,
                config=config,
            )
        return 1 if failures else 0
    outer_pixs = _select_outer_pixs(
        outer_pixs=args.outer_pix,
        sample=args.sample,
        outer_level=config.outer_level,
    )

    new_outputs = None
    if args.workers != 1:
        new_outputs = build_new_outputs_with_runner(
            gaia_root=args.gaia_root,
            dust_root=dust_root,
            release=args.release,
            config=config,
            outer_pixs=outer_pixs,
            workers=args.workers,
            gaia_cache_entries=args.gaia_cache_entries,
            gaia_cache_mb=args.gaia_cache_mb,
            region_level=args.region_level,
        )

    ignored_inner_columns = (
        LOCAL_WINNER_DIVERGENCE_COLUMNS
        if args.allow_local_winner_divergence
        else frozenset()
    )
    if ignored_inner_columns:
        print(
            "ignoring intentional local-winner divergence columns:",
            " ".join(sorted(ignored_inner_columns)),
        )
    if args.allow_boundary_overlap_divergence:
        ignored_inner_columns = ignored_inner_columns | BOUNDARY_OVERLAP_DIVERGENCE_COLUMNS
        print(
            "accepting boundary overlap divergence:",
            " ".join(sorted(BOUNDARY_OVERLAP_DIVERGENCE_COLUMNS)),
        )

    failures = 0
    for index, outer_pix in enumerate(outer_pixs):
        if index > 0:
            print()
        failures += _compare_outer_pixel(
            outer_pix=outer_pix,
            gaia_root=args.gaia_root,
            dust_root=dust_root,
            release=args.release,
            config=config,
            new_outputs=new_outputs,
            ignored_inner_columns=ignored_inner_columns,
            allow_asterism_membership_divergence=args.allow_boundary_overlap_divergence,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
