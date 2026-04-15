"""Persisted build artifact readers and writers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

from ..gaia._constants import HDF5_COMPRESSION, HDF5_COMPRESSION_OPTS, HDF5_SHUFFLE
from ._constants import ASTERISMS_DTYPE, INNER_DTYPE, OUTER_DATASET_ASTERISMS, OUTER_DATASET_INNER


def _table_to_structured_array(table: Table, dtype: np.dtype) -> np.ndarray:
    data = np.zeros(len(table), dtype=dtype)
    for name in dtype.names or ():
        data[name] = np.asarray(table[name], dtype=dtype[name])
    return data


def write_outer_artifact(
    filename: Path,
    *,
    inner: Table,
    asterisms: Table | None,
) -> None:
    """Write one per-outer-pixel HDF5 artifact container."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp_filename = filename.with_suffix(filename.suffix + ".tmp")
    if tmp_filename.exists():
        tmp_filename.unlink()

    with h5py.File(tmp_filename, "w") as handle:
        handle.create_dataset(
            OUTER_DATASET_INNER,
            data=_table_to_structured_array(inner, INNER_DTYPE),
            compression=HDF5_COMPRESSION,
            compression_opts=HDF5_COMPRESSION_OPTS,
            shuffle=HDF5_SHUFFLE,
        )
        handle.create_dataset(
            OUTER_DATASET_ASTERISMS,
            data=(
                np.zeros(0, dtype=ASTERISMS_DTYPE)
                if asterisms is None
                else _table_to_structured_array(asterisms, ASTERISMS_DTYPE)
            ),
            compression=HDF5_COMPRESSION,
            compression_opts=HDF5_COMPRESSION_OPTS,
            shuffle=HDF5_SHUFFLE,
        )

    tmp_filename.replace(filename)


def read_outer_dataset(filename: Path, dataset_name: str) -> Table:
    """Read one dataset from a persisted outer artifact."""

    with h5py.File(filename, "r") as handle:
        return Table(handle[dataset_name][...])
