"""Gaia TGE dust loading and sampling helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from astropy.table import Table
import numpy as np

from ..spatial import get_pixel_skycoord, get_subpixels
from ._exceptions import DustError

GAIA_TGE_RELATIVE_FILENAME = Path("gaia_tge") / "TotalGalacticExtinctionMap_001.csv.gz"


def gaia_tge_map_filename(dust_root: Path) -> Path:
    """Return the expected Gaia TGE map filename under one dust root."""

    filename = Path(dust_root).expanduser().resolve() / GAIA_TGE_RELATIVE_FILENAME
    if not filename.is_file():
        raise DustError(f"Gaia TGE map not found under dust_root: {filename}")
    return filename


@lru_cache(maxsize=None)
def _load_gaia_tge_query(map_filename: str):
    try:
        import dustmaps.gaia_tge as gaia_tge
    except ImportError as exc:  # pragma: no cover - import availability is environment-specific
        raise DustError(
            "dustmaps must be installed to load Gaia TGE dust fields"
        ) from exc
    return gaia_tge.GaiaTGEQuery(map_fname=map_filename, healpix_level="optimum")


def sample_gaia_a0_for_outer_pixel(
    *,
    dust_root: Path,
    outer_level: int,
    outer_pix: int,
    inner_level: int,
    max_data_level: int,
) -> np.ndarray:
    """Sample Gaia TGE A0 using the legacy max-data-level coarse-sampling rule."""

    if max_data_level < outer_level:
        raise DustError("max_data_level must be greater than or equal to outer_level")
    if max_data_level > inner_level:
        raise DustError("max_data_level must be less than or equal to inner_level")

    map_filename = gaia_tge_map_filename(dust_root)
    query = _load_gaia_tge_query(str(map_filename))

    sample_pixs = get_subpixels(outer_level, outer_pix, max_data_level)
    sample_coords = get_pixel_skycoord(max_data_level, sample_pixs)
    values = np.asarray(query.query(sample_coords), dtype=np.float64)
    if inner_level > max_data_level:
        values = np.repeat(values, 4 ** (inner_level - max_data_level))
    return values


def add_gaia_a0_to_inner(
    inner: Table,
    *,
    dust_root: Path,
    outer_level: int,
    outer_pix: int,
    inner_level: int,
    max_data_level: int,
) -> Table:
    """Return a copy of `inner` with `gaia_A0` inserted after `pix`."""

    values = sample_gaia_a0_for_outer_pixel(
        dust_root=dust_root,
        outer_level=outer_level,
        outer_pix=outer_pix,
        inner_level=inner_level,
        max_data_level=max_data_level,
    )
    if len(values) != len(inner):
        raise DustError(
            f"Gaia TGE A0 length {len(values)} does not match inner rows {len(inner)}"
        )

    result = inner.copy(copy_data=True)
    if "gaia_A0" in result.colnames:
        result["gaia_A0"] = values
        return result

    result.add_column(np.asarray(values, dtype=np.float64), name="gaia_A0", index=1)
    return result

