"""Canonical Gaia store constants."""

from __future__ import annotations

from typing import Final

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
HDF5_COMPRESSION: Final[str] = "gzip"
HDF5_COMPRESSION_OPTS: Final[int] = 9
HDF5_SHUFFLE: Final[bool] = True

HOUR_FOLDER_OVERRIDE_PIXELS: Final[frozenset[int]] = frozenset(
    {
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
)
