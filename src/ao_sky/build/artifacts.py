"""Persisted build artifact readers and writers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import h5py
import hdf5plugin
import numpy as np
from astropy.table import Table

from ..gaia._constants import HDF5_SHUFFLE
from ._constants import (
    ASTERISMS_DTYPE,
    INNER_DTYPE,
    MAPS_DATASET,
    MAPS_DTYPE,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
)

ARTIFACT_COMPRESSION_ENV = "AO_SKY_ARTIFACT_COMPRESSION"
DEFAULT_ARTIFACT_COMPRESSION = "blosc-zstd"
DEFAULT_ARTIFACT_BLOSC_LEVEL = 5
HDF5_BLOSC_FILTER_ID = 32001


@dataclass(frozen=True, slots=True)
class ArtifactWriteProfile:
    """Timing and size profile for one outer-artifact write."""

    total_seconds: float = 0.0
    convert_inner_seconds: float = 0.0
    convert_asterisms_seconds: float = 0.0
    hdf5_open_seconds: float = 0.0
    hdf5_inner_seconds: float = 0.0
    hdf5_asterisms_seconds: float = 0.0
    hdf5_close_seconds: float = 0.0
    replace_seconds: float = 0.0
    output_bytes: int = 0
    inner_input_bytes: int = 0
    asterism_input_bytes: int = 0
    inner_structured_bytes: int = 0
    asterism_structured_bytes: int = 0
    inner_rows: int = 0
    asterism_rows: int = 0


@dataclass(slots=True)
class ArtifactMemoryProfile:
    """Mutable detailed memory checkpoints for one outer-artifact write."""

    rss_before_convert_inner_mb: float = 0.0
    peak_before_convert_inner_mb: float = 0.0
    rss_after_convert_inner_mb: float = 0.0
    peak_after_convert_inner_mb: float = 0.0
    rss_after_convert_asterisms_mb: float = 0.0
    peak_after_convert_asterisms_mb: float = 0.0
    rss_after_hdf5_open_mb: float = 0.0
    peak_after_hdf5_open_mb: float = 0.0
    rss_after_hdf5_inner_mb: float = 0.0
    peak_after_hdf5_inner_mb: float = 0.0
    rss_after_hdf5_asterisms_mb: float = 0.0
    peak_after_hdf5_asterisms_mb: float = 0.0
    rss_after_hdf5_close_mb: float = 0.0
    peak_after_hdf5_close_mb: float = 0.0
    rss_after_replace_mb: float = 0.0
    peak_after_replace_mb: float = 0.0


def _sample_artifact_memory(
    profile: ArtifactMemoryProfile | None,
    current_attr: str,
    peak_attr: str,
    *,
    rss_sampler=None,
    peak_sampler=None,
) -> None:
    if profile is None or rss_sampler is None or peak_sampler is None:
        return
    setattr(profile, current_attr, float(rss_sampler()))
    setattr(profile, peak_attr, float(peak_sampler()))


def _table_nbytes(table: Table | None) -> int:
    if table is None:
        return 0
    total = 0
    for name in table.colnames:
        total += int(np.asarray(table[name]).nbytes)
    return total


def _table_to_structured_array(table: Table, dtype: np.dtype) -> np.ndarray:
    data = np.zeros(len(table), dtype=dtype)
    for name in dtype.names or ():
        data[name] = np.asarray(table[name], dtype=dtype[name])
    return data


def _artifact_hdf5_dataset_options() -> dict[str, object]:
    """Return HDF5 dataset options for derived build artifacts."""

    mode = os.environ.get(ARTIFACT_COMPRESSION_ENV, "").lower().strip()
    if mode in {"", "default", DEFAULT_ARTIFACT_COMPRESSION, "blosc_zstd"}:
        return _blosc_zstd_dataset_options()
    if mode in {"gzip9", "gzip-9"}:
        return {"compression": "gzip", "compression_opts": 9, "shuffle": HDF5_SHUFFLE}
    if mode in {"0", "false", "none", "off", "uncompressed"}:
        return {}
    if mode in {"gzip1", "gzip-1"}:
        return {"compression": "gzip", "compression_opts": 1, "shuffle": HDF5_SHUFFLE}
    if mode in {"gzip4", "gzip-4"}:
        return {"compression": "gzip", "compression_opts": 4, "shuffle": HDF5_SHUFFLE}
    if mode == "lzf":
        return {"compression": "lzf", "shuffle": HDF5_SHUFFLE}
    if mode in {"blosc-lz4", "blosc_lz4"}:
        return dict(
            _load_hdf5plugin().Blosc(
                cname="lz4",
                clevel=DEFAULT_ARTIFACT_BLOSC_LEVEL,
                shuffle=1,
            )
        )
    if mode in {"bitshuffle-lz4", "bitshuffle_lz4"}:
        return dict(
            _load_hdf5plugin().Bitshuffle(
                cname="lz4",
                clevel=DEFAULT_ARTIFACT_BLOSC_LEVEL,
            )
        )
    raise ValueError(
        f"Unsupported {ARTIFACT_COMPRESSION_ENV}={mode!r}; expected one of "
        "gzip9, gzip1, gzip4, lzf, blosc-lz4, blosc-zstd, bitshuffle-lz4, none"
    )


def _blosc_zstd_dataset_options() -> dict[str, object]:
    return dict(
        _load_hdf5plugin().Blosc(
            cname="zstd",
            clevel=DEFAULT_ARTIFACT_BLOSC_LEVEL,
            shuffle=1,
        )
    )


def _load_hdf5plugin():
    """Return the imported hdf5plugin module for derived artifact filters."""

    return hdf5plugin


def _ensure_artifact_hdf5_filters() -> None:
    _load_hdf5plugin()


def write_outer_artifact(
    filename: Path,
    *,
    inner: Table,
    asterisms: Table | None,
) -> None:
    """Write one per-outer-pixel HDF5 artifact container."""

    write_outer_artifact_profiled(filename, inner=inner, asterisms=asterisms)


def write_outer_artifact_profiled(
    filename: Path,
    *,
    inner: Table,
    asterisms: Table | None,
    measure_input_bytes: bool = False,
    memory_profile: ArtifactMemoryProfile | None = None,
    rss_sampler=None,
    peak_sampler=None,
) -> ArtifactWriteProfile:
    """Write one per-outer-pixel HDF5 artifact container and return timings."""

    total_started = time.perf_counter()
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp_filename = filename.with_suffix(filename.suffix + ".tmp")
    if tmp_filename.exists():
        tmp_filename.unlink()

    _sample_artifact_memory(
        memory_profile,
        "rss_before_convert_inner_mb",
        "peak_before_convert_inner_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )
    convert_started = time.perf_counter()
    inner_input_bytes = _table_nbytes(inner) if measure_input_bytes else 0
    inner_data = _table_to_structured_array(inner, INNER_DTYPE)
    inner_structured_bytes = int(inner_data.nbytes)
    convert_inner_seconds = time.perf_counter() - convert_started
    _sample_artifact_memory(
        memory_profile,
        "rss_after_convert_inner_mb",
        "peak_after_convert_inner_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )

    convert_started = time.perf_counter()
    asterism_input_bytes = _table_nbytes(asterisms) if measure_input_bytes else 0
    if asterisms is None:
        asterism_data = np.zeros(0, dtype=ASTERISMS_DTYPE)
        asterism_rows = 0
    else:
        asterism_data = _table_to_structured_array(asterisms, ASTERISMS_DTYPE)
        asterism_rows = len(asterisms)
    asterism_structured_bytes = int(asterism_data.nbytes)
    convert_asterisms_seconds = time.perf_counter() - convert_started
    _sample_artifact_memory(
        memory_profile,
        "rss_after_convert_asterisms_mb",
        "peak_after_convert_asterisms_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )

    hdf5_open_seconds = 0.0
    hdf5_inner_seconds = 0.0
    hdf5_asterisms_seconds = 0.0
    hdf5_close_seconds = 0.0
    dataset_options = _artifact_hdf5_dataset_options()
    open_started = time.perf_counter()
    handle = h5py.File(tmp_filename, "w")
    hdf5_open_seconds = time.perf_counter() - open_started
    _sample_artifact_memory(
        memory_profile,
        "rss_after_hdf5_open_mb",
        "peak_after_hdf5_open_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )
    try:
        dataset_started = time.perf_counter()
        handle.create_dataset(
            OUTER_DATASET_INNER,
            data=inner_data,
            **dataset_options,
        )
        hdf5_inner_seconds = time.perf_counter() - dataset_started
        _sample_artifact_memory(
            memory_profile,
            "rss_after_hdf5_inner_mb",
            "peak_after_hdf5_inner_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )

        dataset_started = time.perf_counter()
        handle.create_dataset(
            OUTER_DATASET_ASTERISMS,
            data=asterism_data,
            **dataset_options,
        )
        hdf5_asterisms_seconds = time.perf_counter() - dataset_started
        _sample_artifact_memory(
            memory_profile,
            "rss_after_hdf5_asterisms_mb",
            "peak_after_hdf5_asterisms_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )
    finally:
        close_started = time.perf_counter()
        handle.close()
        hdf5_close_seconds = time.perf_counter() - close_started
        _sample_artifact_memory(
            memory_profile,
            "rss_after_hdf5_close_mb",
            "peak_after_hdf5_close_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )

    output_bytes = tmp_filename.stat().st_size
    replace_started = time.perf_counter()
    tmp_filename.replace(filename)
    replace_seconds = time.perf_counter() - replace_started
    _sample_artifact_memory(
        memory_profile,
        "rss_after_replace_mb",
        "peak_after_replace_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )
    return ArtifactWriteProfile(
        total_seconds=time.perf_counter() - total_started,
        convert_inner_seconds=convert_inner_seconds,
        convert_asterisms_seconds=convert_asterisms_seconds,
        hdf5_open_seconds=hdf5_open_seconds,
        hdf5_inner_seconds=hdf5_inner_seconds,
        hdf5_asterisms_seconds=hdf5_asterisms_seconds,
        hdf5_close_seconds=hdf5_close_seconds,
        replace_seconds=replace_seconds,
        output_bytes=int(output_bytes),
        inner_input_bytes=int(inner_input_bytes),
        asterism_input_bytes=int(asterism_input_bytes),
        inner_structured_bytes=int(inner_structured_bytes),
        asterism_structured_bytes=int(asterism_structured_bytes),
        inner_rows=len(inner),
        asterism_rows=int(asterism_rows),
    )



def write_maps_artifact(
    filename: Path,
    *,
    maps: np.ndarray,
) -> None:
    """Write one dense all-sky maps artifact."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp_filename = filename.with_suffix(filename.suffix + ".tmp")
    if tmp_filename.exists():
        tmp_filename.unlink()

    dataset_options = _artifact_hdf5_dataset_options()
    with h5py.File(tmp_filename, "w") as handle:
        handle.create_dataset(
            MAPS_DATASET,
            data=np.asarray(maps, dtype=MAPS_DTYPE),
            **dataset_options,
        )

    tmp_filename.replace(filename)


def write_maps_family_dataset(
    filename: Path,
    *,
    dataset_name: str,
    data: np.ndarray,
) -> None:
    """Atomically add or replace one auxiliary dataset in a maps artifact."""

    if not filename.is_file():
        raise FileNotFoundError(f"Maps artifact not found: {filename}")

    tmp_filename = filename.with_suffix(filename.suffix + ".tmp")
    if tmp_filename.exists():
        tmp_filename.unlink()

    dataset_options = _artifact_hdf5_dataset_options()
    with h5py.File(filename, "r") as source, h5py.File(tmp_filename, "w") as target:
        for name in source.keys():
            if name == dataset_name:
                continue
            source.copy(name, target)
        target.create_dataset(
            dataset_name,
            data=np.asarray(data),
            **dataset_options,
        )

    tmp_filename.replace(filename)


def read_outer_dataset(filename: Path, dataset_name: str) -> Table:
    """Read one dataset from a persisted outer artifact."""

    _ensure_artifact_hdf5_filters()
    with h5py.File(filename, "r") as handle:
        return Table(handle[dataset_name][...])


def read_maps_dataset(filename: Path, dataset_name: str = MAPS_DATASET) -> Table:
    """Read one dataset from a dense all-sky maps artifact."""

    _ensure_artifact_hdf5_filters()
    with h5py.File(filename, "r") as handle:
        return Table(handle[dataset_name][...])
