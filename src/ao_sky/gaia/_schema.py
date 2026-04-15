"""Canonical Gaia schema shaping helpers.

This module owns the in-memory and on-disk canonical Gaia schema coercion used
by the raw store and Gaia-domain table transforms.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ._constants import GAIA_SCHEMA_COLUMNS, GAIA_SCHEMA_DTYPE
from ._exceptions import GaiaError


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


def coerce_table_to_canonical_gaia_schema(table: Table) -> Table:
    """Return a table coerced into the canonical Gaia schema.

    Args:
        table: Input `Table` that must already expose exactly the canonical Gaia
            columns in canonical order.

    Returns:
        Canonical Gaia `Table` with stable dtypes and filled missing-value
        policy.

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


def table_to_structured_array(table: Table) -> np.ndarray:
    """Encode a canonical Gaia table into the stored structured-array layout."""

    canonical = coerce_table_to_canonical_gaia_schema(table)
    array = np.empty(len(canonical), dtype=np.dtype(list(GAIA_SCHEMA_DTYPE)))
    for name in GAIA_SCHEMA_COLUMNS:
        array[name] = np.asarray(canonical[name])
    return array


def structured_array_to_table(array: np.ndarray) -> Table:
    """Decode one stored Gaia dataset into the canonical in-memory table form.

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
