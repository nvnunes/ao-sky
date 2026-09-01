"""Export retained winner asterism catalogs from build artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .._hdf5 import ensure_hdf5_filters, hdf5_dataset_options
from ..build._constants import ASTERISMS_DTYPE, WORK_STATUS_DONE
from ..build.control import load_build_definition, load_state
from ._exceptions import AsterismError
from .lookup import (
    ASTERISM_LOOKUP_MEAN_FIELDS,
    AsterismLookupFilters,
    _iter_lookup_observations,
    _load_moc_outer_pixels,
    _make_moc_inner_selector,
    _normalize_outer_pixels,
    _passes_lookup_metric_filters,
)

ASTERISM_EXPORT_FORMATS: tuple[str, ...] = ("hdf5", "fits")
ASTERISM_EXPORT_DTYPE: np.dtype = np.dtype(
    [
        ("asterism_id", "<i8"),
        ("outer_pix", "<i8"),
        *[
            (name, ASTERISMS_DTYPE[name])
            for name in (ASTERISMS_DTYPE.names or ())
            if name not in {"asterism_id", "pix"}
        ],
        ("inner_pixel_count", "<i8"),
        *[(name, "<f8") for name in ASTERISM_LOOKUP_MEAN_FIELDS],
    ]
)


@dataclass(frozen=True, slots=True)
class AsterismExportSummary:
    """Summary of a completed asterism catalog export."""

    output_path: Path
    format: str
    selected_outer_pixel_count: int
    exported_asterism_count: int
    chunk_count: int


def export_asterisms(
    build_path: Path | str,
    output_path: Path | str,
    *,
    format: str = "hdf5",
    outer_pixels: int | list[int] | tuple[int, ...] | None = None,
    moc_file: Path | str | None = None,
    filters: AsterismLookupFilters | None = None,
    overwrite: bool = False,
    chunk_count: int = 1,
) -> AsterismExportSummary:
    """Export retained regularized winner asterisms from build artifacts.

    The export is read-only with respect to the build. Rows are globally
    deduplicated by sorted real member ``source_id`` values across the selected
    region, then assigned deterministic export-local ``asterism_id`` values in
    first-seen build stream order.
    """

    resolved_format = str(format).lower()
    if resolved_format not in ASTERISM_EXPORT_FORMATS:
        raise AsterismError("format must be one of: " + ", ".join(ASTERISM_EXPORT_FORMATS))
    chunk_count = int(chunk_count)
    if chunk_count < 1:
        raise AsterismError("chunk_count must be at least 1")

    resolved_build_path = Path(build_path)
    resolved_output_path = Path(output_path)
    if resolved_output_path.exists() and not overwrite:
        raise AsterismError(f"output already exists: {resolved_output_path}")

    definition, selected_outer_pixels, inner_selector = _resolve_export_selection(
        resolved_build_path,
        outer_pixels=outer_pixels,
        moc_file=moc_file,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{resolved_output_path.name}.",
        dir=str(resolved_output_path.parent or Path(".")),
    ) as temp_dir:
        temp_root = Path(temp_dir)
        database_path = temp_root / "asterisms.sqlite"
        _build_export_database(
            database_path,
            resolved_build_path,
            selected_outer_pixels,
            inner_selector=inner_selector,
            filters=filters,
            definition=definition,
        )
        rows = _collect_export_rows(database_path, filters=filters)
        actual_chunk_count = min(chunk_count, len(rows)) if rows else 0
        chunks = _split_rows(rows, actual_chunk_count)
        temp_output_path = temp_root / resolved_output_path.name
        if resolved_format == "hdf5":
            _write_hdf5_export(
                temp_output_path,
                chunks,
                selected_outer_pixel_count=len(selected_outer_pixels),
                exported_asterism_count=len(rows),
                requested_chunk_count=chunk_count,
            )
        else:
            _write_fits_export(
                temp_output_path,
                chunks,
                selected_outer_pixel_count=len(selected_outer_pixels),
                exported_asterism_count=len(rows),
                requested_chunk_count=chunk_count,
            )
        os.replace(temp_output_path, resolved_output_path)

    return AsterismExportSummary(
        output_path=resolved_output_path,
        format=resolved_format,
        selected_outer_pixel_count=len(selected_outer_pixels),
        exported_asterism_count=len(rows),
        chunk_count=actual_chunk_count,
    )


def _resolve_export_selection(
    build_path: Path,
    *,
    outer_pixels: int | list[int] | tuple[int, ...] | None,
    moc_file: Path | str | None,
):
    if outer_pixels is not None and moc_file is not None:
        raise AsterismError("export_asterisms accepts only one selector")

    definition = load_build_definition(build_path)
    state = load_state(build_path)
    if moc_file is not None:
        moc, selected_outer_pixels = _load_moc_outer_pixels(
            Path(moc_file),
            outer_level=definition.outer_level,
        )
        inner_selector = _make_moc_inner_selector(moc, inner_level=definition.inner_level)
    elif outer_pixels is None:
        selected_outer_pixels = tuple(
            int(row["outer_pix"])
            for row in state
            if int(row["traversal_status"]) == WORK_STATUS_DONE
        )
        inner_selector = None
    else:
        selected_outer_pixels = _normalize_outer_pixels(outer_pixels)
        inner_selector = None
    return definition, tuple(sorted(selected_outer_pixels)), inner_selector


def _build_export_database(
    database_path: Path,
    build_path: Path,
    outer_pixels: tuple[int, ...],
    *,
    inner_selector,
    filters: AsterismLookupFilters | None,
    definition,
) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE asterisms (
                key TEXT PRIMARY KEY,
                first_ordinal INTEGER NOT NULL,
                representative_json TEXT NOT NULL,
                outer_pix INTEGER NOT NULL,
                inner_pixel_count INTEGER NOT NULL,
                sum_on_axis_winner_ee REAL NOT NULL,
                count_on_axis_winner_ee INTEGER NOT NULL,
                sum_field_averaged_winner_ee REAL NOT NULL,
                count_field_averaged_winner_ee INTEGER NOT NULL,
                sum_gaia_A0 REAL NOT NULL,
                count_gaia_A0 INTEGER NOT NULL
            )
            """
        )
        ordinal = 0
        for observation in _iter_lookup_observations(
            build_path,
            outer_pixels,
            inner_selector=inner_selector,
            filters=filters,
            definition=definition,
        ):
            sums, counts = _support_sums(observation.support)
            key = _key_to_text(observation.key)
            existing = connection.execute(
                "SELECT key FROM asterisms WHERE key = ?",
                (key,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO asterisms VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        key,
                        ordinal,
                        json.dumps(_jsonable_representative(observation.representative)),
                        int(observation.outer_pix),
                        int(len(observation.support)),
                        sums["on_axis_winner_ee"],
                        counts["on_axis_winner_ee"],
                        sums["field_averaged_winner_ee"],
                        counts["field_averaged_winner_ee"],
                        sums["gaia_A0"],
                        counts["gaia_A0"],
                    ),
                )
                ordinal += 1
            else:
                connection.execute(
                    """
                    UPDATE asterisms
                    SET inner_pixel_count = inner_pixel_count + ?,
                        sum_on_axis_winner_ee = sum_on_axis_winner_ee + ?,
                        count_on_axis_winner_ee = count_on_axis_winner_ee + ?,
                        sum_field_averaged_winner_ee = sum_field_averaged_winner_ee + ?,
                        count_field_averaged_winner_ee = count_field_averaged_winner_ee + ?,
                        sum_gaia_A0 = sum_gaia_A0 + ?,
                        count_gaia_A0 = count_gaia_A0 + ?
                    WHERE key = ?
                    """,
                    (
                        int(len(observation.support)),
                        sums["on_axis_winner_ee"],
                        counts["on_axis_winner_ee"],
                        sums["field_averaged_winner_ee"],
                        counts["field_averaged_winner_ee"],
                        sums["gaia_A0"],
                        counts["gaia_A0"],
                        key,
                    ),
                )


def _support_sums(support: Table) -> tuple[dict[str, float], dict[str, int]]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for field_name in ASTERISM_LOOKUP_MEAN_FIELDS:
        values = np.asarray(support[field_name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        sums[field_name] = float(np.sum(finite)) if len(finite) else 0.0
        counts[field_name] = int(len(finite))
    return sums, counts


def _collect_export_rows(
    database_path: Path,
    *,
    filters: AsterismLookupFilters | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with sqlite3.connect(database_path) as connection:
        for record in connection.execute(
            """
            SELECT
                representative_json,
                outer_pix,
                inner_pixel_count,
                sum_on_axis_winner_ee,
                count_on_axis_winner_ee,
                sum_field_averaged_winner_ee,
                count_field_averaged_winner_ee,
                sum_gaia_A0,
                count_gaia_A0
            FROM asterisms
            ORDER BY first_ordinal
            """
        ):
            row = _record_to_export_row(record)
            if _passes_lookup_metric_filters(row, filters):
                rows.append(row)

    for asterism_id, row in enumerate(rows, start=1):
        row["asterism_id"] = int(asterism_id)
    return rows


def _record_to_export_row(record: tuple[Any, ...]) -> dict[str, object]:
    representative = json.loads(str(record[0]))
    row: dict[str, object] = {"asterism_id": 0, "outer_pix": int(record[1])}
    for name in ASTERISMS_DTYPE.names or ():
        if name in {"asterism_id", "pix"}:
            continue
        row[name] = representative[name]
    row["inner_pixel_count"] = int(record[2])
    metric_records = {
        "on_axis_winner_ee": (float(record[3]), int(record[4])),
        "field_averaged_winner_ee": (float(record[5]), int(record[6])),
        "gaia_A0": (float(record[7]), int(record[8])),
    }
    for field_name, (value_sum, value_count) in metric_records.items():
        row[field_name] = np.nan if value_count == 0 else value_sum / value_count
    return row


def _split_rows(rows: list[dict[str, object]], chunk_count: int) -> list[list[dict[str, object]]]:
    if chunk_count == 0:
        return []
    base_size, remainder = divmod(len(rows), chunk_count)
    chunks: list[list[dict[str, object]]] = []
    start = 0
    for chunk_index in range(chunk_count):
        size = base_size + (1 if chunk_index < remainder else 0)
        end = start + size
        if start != end:
            chunks.append(rows[start:end])
        start = end
    return chunks


def _write_hdf5_export(
    output_path: Path,
    chunks: list[list[dict[str, object]]],
    *,
    selected_outer_pixel_count: int,
    exported_asterism_count: int,
    requested_chunk_count: int,
) -> None:
    ensure_hdf5_filters()
    with h5py.File(output_path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["format"] = "hdf5"
        metadata.attrs["selected_outer_pixel_count"] = int(selected_outer_pixel_count)
        metadata.attrs["exported_asterism_count"] = int(exported_asterism_count)
        metadata.attrs["requested_chunk_count"] = int(requested_chunk_count)
        metadata.attrs["chunk_count"] = int(len(chunks))
        chunk_group = handle.create_group("chunks")
        for chunk_index, rows in enumerate(chunks, start=1):
            group = chunk_group.create_group(f"chunk_{chunk_index:06d}")
            group.attrs["start_asterism_id"] = int(rows[0]["asterism_id"])
            group.attrs["end_asterism_id"] = int(rows[-1]["asterism_id"])
            group.create_dataset(
                "asterisms",
                data=_rows_to_structured_array(rows),
                **hdf5_dataset_options(),
            )


def _write_fits_export(
    output_path: Path,
    chunks: list[list[dict[str, object]]],
    *,
    selected_outer_pixel_count: int,
    exported_asterism_count: int,
    requested_chunk_count: int,
) -> None:
    primary = fits.PrimaryHDU()
    primary.header["AOSKYFMT"] = "fits"
    primary.header["SELOUTER"] = int(selected_outer_pixel_count)
    primary.header["NROWS"] = int(exported_asterism_count)
    primary.header["REQCHUNK"] = int(requested_chunk_count)
    primary.header["NCHUNKS"] = int(len(chunks))
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [primary]
    for chunk_index, rows in enumerate(chunks, start=1):
        table = Table(_rows_to_structured_array(rows))
        hdu = fits.BinTableHDU(table, name=f"ASTERISMS_{chunk_index:06d}")
        hdu.header["STARTID"] = int(rows[0]["asterism_id"])
        hdu.header["ENDID"] = int(rows[-1]["asterism_id"])
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(output_path, overwrite=True)


def _rows_to_structured_array(rows: list[dict[str, object]]) -> np.ndarray:
    array = np.zeros(len(rows), dtype=ASTERISM_EXPORT_DTYPE)
    for row_index, row in enumerate(rows):
        for name in ASTERISM_EXPORT_DTYPE.names or ():
            array[name][row_index] = row[name]
    return array


def _jsonable_representative(row: dict[str, object]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for name in ASTERISMS_DTYPE.names or ():
        dtype = ASTERISMS_DTYPE[name]
        value = row[name]
        if np.issubdtype(dtype, np.integer):
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def _key_to_text(key: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in key)
