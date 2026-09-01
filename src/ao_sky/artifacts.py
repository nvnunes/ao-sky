"""Read-only access to persisted `ao-sky` artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ._hdf5 import ensure_hdf5_filters
from ._paths import get_outer_pixel_bucket_path
from .build._constants import (
    MAPS_DATASET,
    MAPS_FILENAME_TEMPLATE,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
    OUTER_FILENAME,
    RUNTIME_CONFIG_FILENAME,
)
from .build.artifacts import read_maps_dataset, read_outer_dataset, read_outer_products
from .build.runtime_config import load_runtime_config
from .gaia._constants import HDF5_DATASET_NAME
from .gaia.transform import apply_proper_motion, compute_r_magnitude
from .plotting.healpix import get_pixel_skycoord


@dataclass(frozen=True)
class MapData:
    """A selected dense map field with pixel centers."""

    filename: Path
    level: int
    field: str
    values: np.ndarray
    pixs: np.ndarray
    coords: SkyCoord
    table: Table


class AoSkyArtifactStore:
    """Read existing `ao-sky` build artifacts without running a build."""

    def __init__(
        self,
        root: Path | str,
        *,
        outer_root: Path | str | None = None,
        gaia_root: Path | str | None = None,
        model_root: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.outer_root = Path(outer_root) if outer_root is not None else self.root
        self.gaia_root = Path(gaia_root) if gaia_root is not None else self.root
        self.model_root = Path(model_root) if model_root is not None else self.root / "models"

    def runtime(self, filename: Path | str | None = None):
        """Load the build-local runtime config."""

        runtime_file = self.root / RUNTIME_CONFIG_FILENAME if filename is None else Path(filename)
        if not runtime_file.is_absolute():
            runtime_file = self.root / runtime_file
        return load_runtime_config(runtime_file, model_root=self.model_root)

    def maps_filename(self, level: int) -> Path:
        """Return the dense map artifact path for a HEALPix level."""

        return self.root / MAPS_FILENAME_TEMPLATE.format(level=int(level))

    def maps_table(self, level: int, *, dataset_name: str = MAPS_DATASET) -> Table:
        """Read a dense map artifact dataset."""

        return read_maps_dataset(self.maps_filename(level), dataset_name)

    def map_data(
        self,
        field: str,
        *,
        level: int,
        nan_below: float | None = None,
    ) -> MapData:
        """Read one dense map field and its pixel centers."""

        filename = self.maps_filename(level)
        table = read_maps_dataset(filename, MAPS_DATASET)
        pixs = np.asarray(table["pix"], dtype=np.int64)
        values = np.asarray(table[field], dtype=float)
        if nan_below is not None:
            values = values.copy()
            values[values < float(nan_below)] = np.nan
        return MapData(
            filename=filename,
            level=int(level),
            field=field,
            values=values,
            pixs=pixs,
            coords=get_pixel_skycoord(level, pixs),
            table=table,
        )

    def outer_path(self, outer_pix: int, *, outer_level: int, inner_level: int) -> Path:
        """Return the per-outer artifact path for one outer pixel."""

        return (
            self.outer_root
            / f"hpx{int(outer_level)}-{int(inner_level)}"
            / get_outer_pixel_bucket_path(int(outer_level), int(outer_pix))
            / OUTER_FILENAME
        )

    def outer_products(self, outer_pix: int, *, outer_level: int, inner_level: int) -> tuple[Table, Table]:
        """Read both inner and retained-asterism tables for one outer pixel."""

        return read_outer_products(
            self.outer_path(outer_pix, outer_level=outer_level, inner_level=inner_level)
        )

    def inner(self, outer_pix: int, *, outer_level: int, inner_level: int) -> Table:
        """Read the inner table for one outer pixel."""

        return read_outer_dataset(
            self.outer_path(outer_pix, outer_level=outer_level, inner_level=inner_level),
            OUTER_DATASET_INNER,
        )

    def asterisms(
        self,
        outer_pix: int,
        *,
        outer_level: int,
        inner_level: int,
        missing_ok: bool = False,
    ) -> Table | None:
        """Read the retained-asterism table for one outer pixel."""

        filename = self.outer_path(outer_pix, outer_level=outer_level, inner_level=inner_level)
        if missing_ok and not filename.is_file():
            return None
        return read_outer_dataset(filename, OUTER_DATASET_ASTERISMS)

    def gaia_path(self, outer_pix: int, *, outer_level: int) -> Path:
        """Return the raw Gaia artifact path for one outer pixel."""

        return (
            self.gaia_root
            / f"hpx{int(outer_level)}"
            / get_outer_pixel_bucket_path(int(outer_level), int(outer_pix))
            / "gaia.h5"
        )

    def gaia(self, outer_pix: int, *, outer_level: int) -> Table:
        """Read one raw Gaia table from the artifact tree."""

        ensure_hdf5_filters()
        with h5py.File(self.gaia_path(outer_pix, outer_level=outer_level), "r") as handle:
            return Table(handle[HDF5_DATASET_NAME][...])

    def gaia_stars(
        self,
        outer_pix: int,
        *,
        outer_level: int,
        epoch: float | None = None,
        band: str = "R",
        finite: bool = True,
    ) -> Table:
        """Read Gaia rows prepared for guide-star use."""

        stars = apply_proper_motion(self.gaia(outer_pix, outer_level=outer_level), epoch=epoch)
        if band == "R" and "R" not in stars.colnames:
            stars["R"] = compute_r_magnitude(stars)
        if finite:
            mask = ~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"])
            if band in stars.colnames:
                mask &= ~np.isnan(stars[band])
            stars = stars[mask]
        return stars


__all__ = [
    "AoSkyArtifactStore",
    "MapData",
]
