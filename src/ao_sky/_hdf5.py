"""Shared HDF5 filter contracts for repo-owned persisted datasets."""

from __future__ import annotations

from typing import Final

import hdf5plugin

HDF5_CODEC_NAME: Final[str] = "blosc-zstd"
HDF5_BLOSC_FILTER_ID: Final[int] = 32001
HDF5_BLOSC_CNAME: Final[str] = "zstd"
HDF5_BLOSC_LEVEL: Final[int] = 5
HDF5_BLOSC_SHUFFLE: Final[int] = 1
HDF5_SHUFFLE_ENABLED: Final[bool] = True


def hdf5_dataset_options() -> dict[str, object]:
    """Return dataset options for repo-owned HDF5 datasets."""

    return dict(
        hdf5plugin.Blosc(
            cname=HDF5_BLOSC_CNAME,
            clevel=HDF5_BLOSC_LEVEL,
            shuffle=HDF5_BLOSC_SHUFFLE,
        )
    )


def ensure_hdf5_filters() -> None:
    """Register HDF5 plugin filters needed by repo-owned datasets."""

    # Importing hdf5plugin registers the Blosc filter with HDF5 for this process.
    _ = hdf5plugin
