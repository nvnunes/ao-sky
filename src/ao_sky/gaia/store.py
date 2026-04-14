"""Canonical raw Gaia store contracts.

This module owns the persisted-file contract for raw Gaia outer-pixel tables.
It defines the runtime store configuration, the canonical schema shaping used
at the HDF5 boundary, and the read-or-materialize workflow that maps one outer
nested HEALPix pixel to one canonical ``gaia.h5`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

from ._constants import (
    GAIA_SCHEMA_COLUMNS,
    GAIA_SCHEMA_DTYPE,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_OPTS,
    HDF5_DATASET_NAME,
    HDF5_SHUFFLE,
    HOUR_FOLDER_OVERRIDE_PIXELS,
)
from ._exceptions import GaiaError
from ._healpix import get_pixel_skycoord
from ._query import query_healpix_table


# Config

@dataclass(frozen=True, slots=True)
class GaiaStoreConfig:
    """Runtime configuration for the canonical raw Gaia store.

    Attributes:
        root: Caller-owned filesystem root for canonical Gaia files. This path
            is used to build canonical per-pixel HDF5 filenames and is not
            persisted inside those files.
        release: Gaia release identifier such as ``"dr3"`` used to select the
            archive table family during materialization.
        healpix_level: Outer nested HEALPix level that defines both the
            per-file storage unit and the canonical path layout under
            ``root``.

    Raises:
        GaiaError: If ``root`` is empty, ``release`` is blank, or
            ``healpix_level`` is negative.
    """

    root: Path | str
    release: str
    healpix_level: int

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not str(root):
            raise GaiaError("root must be a non-empty filesystem path")
        if not self.release or not self.release.strip():
            raise GaiaError("release must be a non-empty string")
        if self.healpix_level < 0:
            raise GaiaError(
                f"healpix_level must be non-negative, got {self.healpix_level}"
            )

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "release", self.release.strip().lower())


# Canonical schema shaping

def _coerce_source_id_column(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values)
    if np.ma.isMaskedArray(data) and np.any(data.mask):
        raise GaiaError("source_id cannot contain masked values")
    return np.asarray(data, dtype=np.int64)


def _coerce_float_column(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values, dtype=np.float64)
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled(np.nan), dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def _coerce_bool_column(values: np.ndarray) -> np.ndarray:
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


def _coerce_table_to_canonical_schema(table: Table) -> Table:
    """Return a Gaia table coerced into the canonical raw-store schema.

    The input table must already expose exactly the canonical column set and
    order owned by ``GAIA_SCHEMA_COLUMNS``. Each column is then coerced into
    the stored on-disk representation used by the canonical HDF5 boundary:
    integer ``source_id``, floating-point astrometric and photometric fields,
    and boolean ``non_single_star`` values with masked entries filled to
    ``False``.

    Raises:
        GaiaError: If the input columns do not match the canonical schema or if
            ``source_id`` contains masked values.
    """

    column_names = tuple(table.colnames)
    if column_names != GAIA_SCHEMA_COLUMNS:
        raise GaiaError(
            "Gaia table columns do not match the canonical schema: "
            f"expected {GAIA_SCHEMA_COLUMNS}, got {column_names}"
        )

    canonical = Table()
    canonical["source_id"] = _coerce_source_id_column(table["source_id"])
    canonical["ra"] = _coerce_float_column(table["ra"])
    canonical["dec"] = _coerce_float_column(table["dec"])
    canonical["G"] = _coerce_float_column(table["G"])
    canonical["BP"] = _coerce_float_column(table["BP"])
    canonical["RP"] = _coerce_float_column(table["RP"])
    canonical["ref_epoch"] = _coerce_float_column(table["ref_epoch"])
    canonical["pmra"] = _coerce_float_column(table["pmra"])
    canonical["pmdec"] = _coerce_float_column(table["pmdec"])
    canonical["non_single_star"] = _coerce_bool_column(table["non_single_star"])
    canonical["ruwe"] = _coerce_float_column(table["ruwe"])
    return canonical


def _table_to_structured_array(table: Table) -> np.ndarray:
    """Encode a canonical Gaia table into the stored structured-array layout.

    The returned array matches the exact field order and dtypes declared by
    ``GAIA_SCHEMA_DTYPE`` and is written directly to the canonical HDF5
    dataset owned by this module.
    """

    canonical = _coerce_table_to_canonical_schema(table)
    array = np.empty(len(canonical), dtype=np.dtype(list(GAIA_SCHEMA_DTYPE)))
    for name in GAIA_SCHEMA_COLUMNS:
        array[name] = np.asarray(canonical[name])
    return array


def _structured_array_to_table(array: np.ndarray) -> Table:
    """Decode one stored Gaia dataset into the canonical in-memory table form.

    The input array must come from the canonical ``gaia`` HDF5 dataset and
    therefore must expose exactly the canonical schema field names in canonical
    order.

    Raises:
        GaiaError: If the structured array field names do not match the
            canonical schema.
    """

    dtype_names = tuple(array.dtype.names or ())
    if dtype_names != GAIA_SCHEMA_COLUMNS:
        raise GaiaError(
            "Stored Gaia dataset does not match the canonical schema: "
            f"expected {GAIA_SCHEMA_COLUMNS}, got {dtype_names}"
        )

    table = Table()
    for name in GAIA_SCHEMA_COLUMNS:
        table[name] = array[name]
    return table


# Public store surface

class GaiaHealpixStore:
    """Read and materialize canonical raw Gaia per-pixel HDF5 files.

    This class is the public owner of the canonical Gaia path contract and the
    simple read-or-materialize workflow for raw outer-pixel tables. Each outer
    nested HEALPix pixel maps to one canonical file at:

    ``<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5``

    The store only exposes raw Gaia tables in the canonical schema and does not
    apply proper motion, derive photometric proxy columns, stitch neighbouring
    pixels, or perform repo-root configuration discovery.
    """

    def __init__(self, config: GaiaStoreConfig) -> None:
        self.config = config

    def healpix_filename(self, outer_pix: int) -> Path:
        """Return the canonical HDF5 filename for one outer pixel.

        Paths follow the stored Gaia contract:
        ``<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5``.
        The ``<hour>h`` and signed declination bucket segments are derived from
        the nested HEALPix pixel centre. For the retained legacy override pixel
        set, the hour bucket is forced to ``14h`` so existing stored layouts
        remain discoverable.

        Returns:
            The canonical filesystem path under the configured Gaia store root
            for the requested outer pixel.

        Raises:
            GaiaError: If ``outer_pix`` is outside the valid range for the
                configured HEALPix level.
        """

        coord = get_pixel_skycoord(self.config.healpix_level, outer_pix)
        hour = int(np.floor(coord.ra.degree / 15.0))
        if outer_pix in HOUR_FOLDER_OVERRIDE_PIXELS:
            hour = 14

        dec_bucket = int(np.floor(np.abs(coord.dec.degree / 10.0)) * 10)
        dec_sign = "+" if coord.dec.degree >= 0 else "-"

        return (
            self.config.root
            / f"gaia-{self.config.release}-hpx{self.config.healpix_level}"
            / f"{hour}h"
            / f"{dec_sign}{dec_bucket:02d}"
            / str(outer_pix)
            / "gaia.h5"
        )

    def load_healpix(self, outer_pix: int, *, force_reload: bool = False) -> Table:
        """Load one canonical raw Gaia table for an outer pixel.

        This method is the public entry point for raw Gaia access. When the
        canonical HDF5 file already exists and ``force_reload`` is false, the
        stored file is read directly. Otherwise the Gaia archive seam is
        queried, the canonical file is written at the configured path, and the
        freshly materialized canonical table is returned.

        The returned table always uses the canonical raw-store schema:
        ``source_id``, ``ra``, ``dec``, ``G``, ``BP``, ``RP``, ``ref_epoch``,
        ``pmra``, ``pmdec``, ``non_single_star``, and ``ruwe``. The method does
        not derive additional columns, rename the canonical fields, or mutate
        coordinates into another epoch.

        Args:
            outer_pix: Outer nested HEALPix pixel index at the configured
                ``healpix_level``.
            force_reload: When true, bypass any existing canonical file and
                rewrite it from a fresh Gaia archive query.

        Returns:
            An `astropy.table.Table` in the canonical raw-store schema.

        Raises:
            GaiaError: If the pixel is invalid, the stored file is malformed,
                the archive query fails, or the table does not match the
                canonical schema.
        """

        filename = self.healpix_filename(outer_pix)
        if filename.exists() and not force_reload:
            return self._read_healpix_file(filename)

        table = query_healpix_table(
            self.config.release,
            self.config.healpix_level,
            outer_pix,
        )
        self._write_healpix_file(filename, table)
        return _coerce_table_to_canonical_schema(table)

    def _read_healpix_file(self, filename: Path) -> Table:
        with h5py.File(filename, "r") as handle:
            if HDF5_DATASET_NAME not in handle:
                raise GaiaError(
                    f"Missing {HDF5_DATASET_NAME!r} dataset in Gaia file {filename}"
                )
            return _structured_array_to_table(handle[HDF5_DATASET_NAME][...])

    def _write_healpix_file(self, filename: Path, table: Table) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        data = _table_to_structured_array(table)
        with h5py.File(filename, "w") as handle:
            handle.create_dataset(
                HDF5_DATASET_NAME,
                data=data,
                compression=HDF5_COMPRESSION,
                compression_opts=HDF5_COMPRESSION_OPTS,
                shuffle=HDF5_SHUFFLE,
            )
