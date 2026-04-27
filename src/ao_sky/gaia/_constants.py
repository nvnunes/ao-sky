"""Canonical Gaia store constants."""

from __future__ import annotations

from typing import Final

from .._hdf5 import HDF5_BLOSC_LEVEL, HDF5_CODEC_NAME, HDF5_SHUFFLE_ENABLED

GAIA_SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
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

GAIA_SCHEMA_DTYPE: Final[tuple[tuple[str, str], ...]] = (
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

HDF5_DATASET_NAME: Final[str] = "gaia"
GAIA_SUMMARY_FILENAME: Final[str] = "summary.h5"
GAIA_SUMMARY_DATASET_NAME: Final[str] = "summary"
GAIA_SUMMARY_DTYPE: Final[tuple[tuple[str, str], ...]] = (
    ("outer_pix", "<i8"),
    ("star_count", "<i8"),
    ("loaded", "?"),
)
HDF5_COMPRESSION: Final[str] = HDF5_CODEC_NAME
HDF5_COMPRESSION_OPTS: Final[int] = HDF5_BLOSC_LEVEL
HDF5_SHUFFLE: Final[bool] = HDF5_SHUFFLE_ENABLED
