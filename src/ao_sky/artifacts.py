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
from .predict import PredictRuntime


@dataclass(frozen=True)
class MapData:
    """A selected dense map field with pixel centers.

    Attributes
    ----------
    filename
        Source map artifact.
    level
        Dense-map HEALPix level.
    field
        Selected map field name.
    values
        Selected field values in pixel order.
    pixs
        HEALPix pixel identifiers corresponding to ``values``.
    coords
        Sky coordinates at the selected pixel centers.
    table
        Full source map table.
    """

    filename: Path
    level: int
    field: str
    values: np.ndarray
    pixs: np.ndarray
    coords: SkyCoord
    table: Table


class AoSkyArtifactStore:
    """Read existing `ao-sky` build artifacts without running a build.

    Relative artifact paths are resolved beneath ``root``. Per-outer artifacts,
    canonical Gaia files, and model snapshots can instead use explicit roots.
    Readers normalize supported layout-version-2 result names in memory and do
    not modify source artifacts.

    Attributes:
        root: Build root containing ``build.yaml`` and dense map artifacts.
        outer_root: Root containing per-outer ``outer.h5`` artifacts.
        gaia_root: Root containing per-outer canonical ``gaia.h5`` files.
        model_root: Directory containing model snapshots named by
            ``build.yaml``.
    """

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

    def runtime(self, filename: Path | str | None = None) -> PredictRuntime:
        """Load normalized prediction policy from the build-local config.

        ``filename`` defaults to ``<root>/build.yaml``. A relative filename is
        resolved beneath ``root``. Supported schema-version-2 fields are
        normalized to the current in-memory names without rewriting the file.
        """

        runtime_file = self.root / RUNTIME_CONFIG_FILENAME if filename is None else Path(filename)
        if not runtime_file.is_absolute():
            runtime_file = self.root / runtime_file
        return load_runtime_config(
            runtime_file,
            model_root=self.model_root,
            allow_legacy=True,
        )

    def maps_filename(self, level: int) -> Path:
        """Return ``<root>/maps-hpx<level>.h5`` for a HEALPix level."""

        return self.root / MAPS_FILENAME_TEMPLATE.format(level=int(level))

    def maps_table(self, level: int, *, dataset_name: str = MAPS_DATASET) -> Table:
        """Read and normalize one dataset from a dense map artifact.

        Layout-version-2 result fields are returned under their current names.
        The source artifact is never modified.
        """

        return read_maps_dataset(self.maps_filename(level), dataset_name)

    def map_data(
        self,
        field: str,
        *,
        level: int,
        nan_below: float | None = None,
    ) -> MapData:
        """Read one dense map field and its nested-HEALPix pixel centers.

        Values below ``nan_below`` are replaced with NaN in the returned field
        array only; the full source table and artifact remain unchanged.
        """

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
        """Return the bucketed ``outer.h5`` path for one outer pixel."""

        return (
            self.outer_root
            / f"hpx{int(outer_level)}-{int(inner_level)}"
            / get_outer_pixel_bucket_path(int(outer_level), int(outer_pix))
            / OUTER_FILENAME
        )

    def outer_products(self, outer_pix: int, *, outer_level: int, inner_level: int) -> tuple[Table, Table]:
        """Read normalized inner and retained-asterism tables for one outer pixel."""

        return read_outer_products(
            self.outer_path(outer_pix, outer_level=outer_level, inner_level=inner_level)
        )

    def inner(self, outer_pix: int, *, outer_level: int, inner_level: int) -> Table:
        """Read the normalized inner table for one outer pixel."""

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
        """Read the normalized retained-asterism table for one outer pixel.

        Return ``None`` only when the ``outer.h5`` file is absent and
        ``missing_ok`` is true. Missing datasets and invalid artifacts still
        raise their normal read errors.
        """

        filename = self.outer_path(outer_pix, outer_level=outer_level, inner_level=inner_level)
        if missing_ok and not filename.is_file():
            return None
        return read_outer_dataset(filename, OUTER_DATASET_ASTERISMS)

    def gaia_path(self, outer_pix: int, *, outer_level: int) -> Path:
        """Return the bucketed canonical ``gaia.h5`` path for one outer pixel."""

        return (
            self.gaia_root
            / f"hpx{int(outer_level)}"
            / get_outer_pixel_bucket_path(int(outer_level), int(outer_pix))
            / "gaia.h5"
        )

    def gaia(self, outer_pix: int, *, outer_level: int) -> Table:
        """Read one raw canonical Gaia table and register its HDF5 filters."""

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
        """Read Gaia rows prepared for guide-star use.

        Proper motions are applied at ``epoch`` when supplied. The empirical
        ``R`` magnitude is derived on demand when ``band="R"``. When ``finite``
        is true, rows with non-finite coordinates or requested-band magnitudes
        are removed.
        """

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
