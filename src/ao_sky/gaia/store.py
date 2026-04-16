"""Canonical raw Gaia store contracts.

This module owns the persisted-file contract for raw Gaia outer-pixel tables.
It defines the runtime store configuration, the canonical schema shaping used
at the HDF5 boundary, and the read-or-materialize workflow that maps one outer
nested HEALPix pixel to one canonical ``gaia.h5`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import TextIO

import h5py
import numpy as np
from astropy.table import Table

from ._constants import (
    GAIA_SUMMARY_DATASET_NAME,
    GAIA_SUMMARY_DTYPE,
    GAIA_SUMMARY_FILENAME,
    GAIA_SCHEMA_COLUMNS,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_OPTS,
    HDF5_DATASET_NAME,
    HDF5_SHUFFLE,
)
from ._exceptions import GaiaError
from ._schema import (
    coerce_table_to_canonical_gaia_schema,
    structured_array_to_table,
    table_to_structured_array,
)
from ._query import query_healpix_table
from .._paths import get_outer_pixel_bucket_path


def _gaia_root_prefix(root: Path, release: str, healpix_level: int) -> Path:
    return Path(root) / f"gaia-{release}-hpx{healpix_level}"


def _build_empty_summary(num_pixels: int) -> np.ndarray:
    summary = np.zeros(num_pixels, dtype=np.dtype(list(GAIA_SUMMARY_DTYPE)))
    summary["outer_pix"] = np.arange(num_pixels, dtype=np.int64)
    return summary


def _emit_fetch_gaia_progress(
    output: TextIO | None,
    message: str,
    *,
    transient: bool = False,
) -> None:
    if output is None:
        return
    prefix = "\r" if transient else ""
    suffix = "" if transient else "\n"
    print(f"{prefix}{message}", end=suffix, file=output, flush=True)


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

        return (
            _gaia_root_prefix(
                self.config.root,
                self.config.release,
                self.config.healpix_level,
            )
            / get_outer_pixel_bucket_path(self.config.healpix_level, outer_pix)
            / "gaia.h5"
        )

    def load_healpix(
        self,
        outer_pix: int,
        *,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
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
            read_only: When true, mark the returned table columns as
                non-writeable. This protects canonical cached Gaia rows from
                accidental in-place value mutation, but does not freeze table
                structure.

        Returns:
            An `astropy.table.Table` in the canonical raw-store schema.

        Raises:
            GaiaError: If the pixel is invalid, the stored file is malformed,
                the archive query fails, or the table does not match the
                canonical schema.
        """

        filename = self.healpix_filename(outer_pix)
        if filename.exists() and not force_reload:
            return _maybe_mark_read_only(
                self._read_healpix_file(filename),
                read_only=read_only,
            )

        table = query_healpix_table(
            self.config.release,
            self.config.healpix_level,
            outer_pix,
        )
        self._write_healpix_file(filename, table)
        return _maybe_mark_read_only(
            coerce_table_to_canonical_gaia_schema(table),
            read_only=read_only,
        )

    def _read_healpix_file(self, filename: Path) -> Table:
        with h5py.File(filename, "r") as handle:
            if HDF5_DATASET_NAME not in handle:
                raise GaiaError(
                    f"Missing {HDF5_DATASET_NAME!r} dataset in Gaia file {filename}"
                )
            return structured_array_to_table(handle[HDF5_DATASET_NAME][...])

    def _write_healpix_file(self, filename: Path, table: Table) -> None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        data = table_to_structured_array(table)
        with h5py.File(filename, "w") as handle:
            handle.create_dataset(
                HDF5_DATASET_NAME,
                data=data,
                compression=HDF5_COMPRESSION,
                compression_opts=HDF5_COMPRESSION_OPTS,
                shuffle=HDF5_SHUFFLE,
            )


def _maybe_mark_read_only(table: Table, *, read_only: bool) -> Table:
    if read_only:
        for name in table.colnames:
            table[name].flags.writeable = False
    return table


class GaiaSummaryStore:
    """Read and write the shared Gaia per-pixel summary for one release and level."""

    def __init__(self, config: GaiaStoreConfig) -> None:
        self.config = config

    def summary_filename(self) -> Path:
        return (
            _gaia_root_prefix(
                self.config.root,
                self.config.release,
                self.config.healpix_level,
            )
            / GAIA_SUMMARY_FILENAME
        )

    def load_summary(self) -> Table:
        filename = self.summary_filename()
        if not filename.is_file():
            raise GaiaError(f"Missing Gaia summary file: {filename}")
        with h5py.File(filename, "r") as handle:
            array = self._require_summary_dataset(handle)[...]
        return Table(array)

    def write_summary(self, table: Table) -> Path:
        filename = self.summary_filename()
        filename.parent.mkdir(parents=True, exist_ok=True)
        data = np.asarray(table.as_array(), dtype=np.dtype(list(GAIA_SUMMARY_DTYPE)))
        with NamedTemporaryFile(
            dir=filename.parent,
            prefix=f".{filename.stem}-",
            suffix=".tmp",
            delete=False,
        ) as tmp_handle:
            tmp_path = Path(tmp_handle.name)
        try:
            with h5py.File(tmp_path, "w") as handle:
                handle.create_dataset(
                    GAIA_SUMMARY_DATASET_NAME,
                    data=data,
                    compression=HDF5_COMPRESSION,
                    compression_opts=HDF5_COMPRESSION_OPTS,
                    shuffle=HDF5_SHUFFLE,
                )
            tmp_path.replace(filename)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return filename

    def initialize_summary(self, num_pixels: int) -> Path:
        """Ensure the shared summary file exists and matches one dense level."""

        filename = self.summary_filename()
        if filename.is_file():
            with h5py.File(filename, "r") as handle:
                self._require_summary_dataset(handle, expected_num_pixels=num_pixels)
            return filename
        return self.write_summary(Table(_build_empty_summary(num_pixels)))

    def open_summary_for_update(self, num_pixels: int) -> tuple[h5py.File, h5py.Dataset]:
        """Open the shared summary file for in-place row updates."""

        filename = self.initialize_summary(num_pixels)
        handle = h5py.File(filename, "r+")
        try:
            dataset = self._require_summary_dataset(
                handle,
                expected_num_pixels=num_pixels,
            )
        except Exception:
            handle.close()
            raise
        return handle, dataset

    def _require_summary_dataset(
        self,
        handle: h5py.File,
        *,
        expected_num_pixels: int | None = None,
    ) -> h5py.Dataset:
        filename = Path(handle.filename)
        if GAIA_SUMMARY_DATASET_NAME not in handle:
            raise GaiaError(
                f"Missing {GAIA_SUMMARY_DATASET_NAME!r} dataset in Gaia summary file {filename}"
            )
        dataset = handle[GAIA_SUMMARY_DATASET_NAME]
        dtype_names = tuple(dataset.dtype.names or ())
        expected_names = tuple(name for name, _ in GAIA_SUMMARY_DTYPE)
        if dtype_names != expected_names:
            raise GaiaError(
                "Stored Gaia summary dataset does not match the summary schema: "
                f"expected {expected_names}, got {dtype_names}"
            )
        if expected_num_pixels is not None and len(dataset) != expected_num_pixels:
            raise GaiaError(
                "Stored Gaia summary dataset does not match the requested HEALPix level: "
                f"expected {expected_num_pixels} rows, got {len(dataset)}"
            )
        if expected_num_pixels is not None and not np.array_equal(
            dataset["outer_pix"],
            np.arange(expected_num_pixels, dtype=np.int64),
        ):
            raise GaiaError(
                "Stored Gaia summary dataset does not use the required dense outer_pix ordering"
            )
        return dataset


def fetch_gaia_store(
    config: GaiaStoreConfig,
    *,
    force_reload: bool = False,
    output: TextIO | None = None,
) -> Path:
    """Materialize one full-sky Gaia store and refresh its shared summary."""

    store = GaiaHealpixStore(config)
    summary_store = GaiaSummaryStore(config)
    num_pixels = 12 * (4 ** config.healpix_level)
    _emit_fetch_gaia_progress(
        output,
        f"Loading Gaia outer pixels for release {config.release} at level {config.healpix_level}:",
    )
    summary_handle, summary_dataset = summary_store.open_summary_for_update(num_pixels)
    start_time = time.time()
    last_time = start_time
    loaded_pixels = 0
    skipped_pixels = 0
    try:
        for outer_pix in range(num_pixels):
            filename = store.healpix_filename(outer_pix)
            existed = filename.is_file()
            already_loaded = bool(summary_dataset[outer_pix]["loaded"])
            if existed and already_loaded and not force_reload:
                skipped_pixels += 1
            else:
                summary_dataset[outer_pix] = (outer_pix, 0, False)
                summary_handle.flush()
                final_table = store.load_healpix(outer_pix, force_reload=force_reload)
                summary_dataset[outer_pix] = (
                    outer_pix,
                    len(final_table),
                    True,
                )
                summary_handle.flush()
                loaded_pixels += 1
            now = time.time()
            current_time = time.strftime("%H:%M:%S", time.localtime(now))
            _emit_fetch_gaia_progress(
                output,
                (
                    f"  {current_time}: {outer_pix + 1}/{num_pixels} "
                    f"(1px in {now - last_time:.2f}s, {loaded_pixels} loaded, "
                    f"{skipped_pixels} skipped)          "
                ),
                transient=True,
            )
            last_time = now
    finally:
        summary_handle.close()

    _emit_fetch_gaia_progress(output, "")
    total_time = time.time() - start_time
    _emit_fetch_gaia_progress(
        output,
        (
            f"  done: {num_pixels}px in {total_time:.1f}s "
            f"({loaded_pixels} loaded, {skipped_pixels} skipped)"
        ),
    )
    return summary_store.summary_filename()
