#!/usr/bin/env python3
"""Temporary migration helper for legacy Gaia FITS outer-pixel files."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig


SOURCE_ROOT = Path("/Volumes/Data/Galaxy/maps/gaia")
DEST_ROOT = Path("/Volumes/Data/Galaxy/aosky")
RELEASE = "dr3"
HEALPIX_LEVEL = 6
PROGRESS_EVERY = 100

LEGACY_COLUMNS = (
    "gaia_id",
    "gaia_ra",
    "gaia_dec",
    "gaia_G",
    "gaia_BP",
    "gaia_RP",
    "gaia_ref_epoch",
    "gaia_pmra",
    "gaia_pmdec",
    "gaia_non_single_star",
    "gaia_R",
)

CANONICAL_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "G",
    "BP",
    "RP",
    "ref_epoch",
    "pmra",
    "pmdec",
    "non_single_star",
    "ruwe",
)

CANONICAL_DTYPE = (
    ("source_id", "<i8"),
    ("ra", "<f8"),
    ("dec", "<f8"),
    ("G", "<f8"),
    ("BP", "<f8"),
    ("RP", "<f8"),
    ("ref_epoch", "<f8"),
    ("pmra", "<f8"),
    ("pmdec", "<f8"),
    ("non_single_star", "?"),
    ("ruwe", "<f8"),
)

HDF5_DATASET_NAME = "gaia"
HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_OPTS = 9
HDF5_SHUFFLE = True


# Discovery

def discover_legacy_fits(source_root: Path) -> list[Path]:
    """Return legacy per-pixel Gaia FITS files in stable path order."""

    return sorted(path for path in source_root.rglob("gaia.fits") if path.is_file())


# Table coercion

def _coerce_source_id(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values)
    if np.ma.isMaskedArray(data) and np.any(data.mask):
        raise ValueError("gaia_id contains masked values")
    return np.asarray(data, dtype=np.int64)


def _coerce_float(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values, dtype=np.float64)
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled(np.nan), dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def _coerce_bool(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values)
    if np.ma.isMaskedArray(data):
        data = data.filled(False)

    array = np.asarray(data)
    if array.dtype.kind == "b":
        return array.astype(np.bool_)
    if array.dtype.kind in {"i", "u", "f"}:
        return array.astype(np.int64) != 0

    result = np.zeros(len(array), dtype=np.bool_)
    for index, value in enumerate(array):
        text = str(value).strip().lower()
        result[index] = text in {"1", "true", "t", "yes", "y"}
    return result


def _require_legacy_columns(table: Table, filename: Path) -> None:
    missing = [name for name in LEGACY_COLUMNS if name not in table.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{filename} is missing required legacy columns: {missing_text}")


def convert_legacy_table(table: Table, filename: Path) -> Table:
    """Convert one legacy Gaia FITS table into the canonical raw schema."""

    _require_legacy_columns(table, filename)

    canonical = Table()
    canonical["source_id"] = _coerce_source_id(table["gaia_id"])
    canonical["ra"] = _coerce_float(table["gaia_ra"])
    canonical["dec"] = _coerce_float(table["gaia_dec"])
    canonical["G"] = _coerce_float(table["gaia_G"])
    canonical["BP"] = _coerce_float(table["gaia_BP"])
    canonical["RP"] = _coerce_float(table["gaia_RP"])
    canonical["ref_epoch"] = _coerce_float(table["gaia_ref_epoch"])
    canonical["pmra"] = _coerce_float(table["gaia_pmra"])
    canonical["pmdec"] = _coerce_float(table["gaia_pmdec"])
    canonical["non_single_star"] = _coerce_bool(table["gaia_non_single_star"])
    canonical["ruwe"] = np.full(len(table), np.nan, dtype=np.float64)
    return canonical


def _table_to_structured_array(table: Table) -> np.ndarray:
    array = np.empty(len(table), dtype=np.dtype(list(CANONICAL_DTYPE)))
    for name in CANONICAL_COLUMNS:
        array[name] = np.asarray(table[name])
    return array


# Migration

def _get_store(dest_root: Path, release: str, healpix_level: int) -> GaiaHealpixStore:
    config = GaiaStoreConfig(root=dest_root, release=release, healpix_level=healpix_level)
    return GaiaHealpixStore(config)


def migrate_one_file(
    legacy_filename: Path,
    *,
    store: GaiaHealpixStore,
) -> str:
    """Migrate one legacy per-pixel FITS file into the canonical HDF5 layout."""

    try:
        outer_pix = int(legacy_filename.parent.name)
    except ValueError as exc:
        raise ValueError(f"Could not derive outer pixel from {legacy_filename}") from exc

    dest_filename = store.healpix_filename(outer_pix)
    if dest_filename.exists():
        return "skipped"

    legacy_table = Table.read(legacy_filename, format="fits")
    canonical_table = convert_legacy_table(legacy_table, legacy_filename)
    data = _table_to_structured_array(canonical_table)

    dest_filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest_filename, "w") as handle:
        handle.create_dataset(
            HDF5_DATASET_NAME,
            data=data,
            compression=HDF5_COMPRESSION,
            compression_opts=HDF5_COMPRESSION_OPTS,
            shuffle=HDF5_SHUFFLE,
        )

    return "processed"


def migrate_legacy_gaia(
    *,
    source_root: Path,
    dest_root: Path,
    release: str,
    healpix_level: int,
    progress_every: int = PROGRESS_EVERY,
    limit: int | None = None,
) -> int:
    """Run the legacy Gaia FITS to canonical HDF5 migration."""

    store = _get_store(dest_root, release, healpix_level)
    legacy_files = discover_legacy_fits(source_root)
    if limit is not None:
        legacy_files = legacy_files[:limit]

    total = len(legacy_files)
    processed = 0
    skipped = 0
    failed = 0
    started = time.monotonic()

    print(f"Discovered {total} legacy Gaia FITS files under {source_root}")

    for index, legacy_filename in enumerate(legacy_files, start=1):
        try:
            status = migrate_one_file(legacy_filename, store=store)
        except Exception as exc:  # temporary migration helper
            failed += 1
            print(f"[{index}/{total}] failed: {legacy_filename} :: {exc}")
            continue

        if status == "processed":
            processed += 1
        else:
            skipped += 1

        if index % progress_every == 0 or index == total:
            elapsed = time.monotonic() - started
            print(
                f"[{index}/{total}] processed={processed} "
                f"skipped={skipped} failed={failed} elapsed={elapsed:.1f}s"
            )

    elapsed = time.monotonic() - started
    print(
        "Migration complete: "
        f"processed={processed} skipped={skipped} failed={failed} elapsed={elapsed:.1f}s"
    )
    return 1 if failed else 0


def main() -> int:
    return migrate_legacy_gaia(
        source_root=SOURCE_ROOT,
        dest_root=DEST_ROOT,
        release=RELEASE,
        healpix_level=HEALPIX_LEVEL,
    )


if __name__ == "__main__":
    raise SystemExit(main())
