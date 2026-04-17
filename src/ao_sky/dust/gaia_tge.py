"""Gaia TGE dust loading and sampling helpers."""

from __future__ import annotations

import csv
from gzip import open as gzip_open
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from astropy.table import Table
import numpy as np

from ..spatial import get_pixel_from_skycoord, get_pixel_skycoord, get_subpixels
from ._exceptions import DustError

GAIA_TGE_RELATIVE_FILENAME = Path("gaia_tge") / "TotalGalacticExtinctionMap_001.csv.gz"
GAIA_TGE_A0_CACHE_FILENAME_TEMPLATE = "ao-sky-gaia-tge-a0-hpx{level}.npy"


def gaia_tge_map_filename(dust_root: Path) -> Path:
    """Return the expected Gaia TGE map filename under one dust root."""

    filename = Path(dust_root).expanduser().resolve() / GAIA_TGE_RELATIVE_FILENAME
    if not filename.is_file():
        raise DustError(f"Gaia TGE map not found under dust_root: {filename}")
    return filename


def gaia_tge_a0_cache_filename(dust_root: Path, level: int) -> Path:
    """Return the build-local dense Gaia TGE A0 cache filename for one level."""

    return Path(dust_root).expanduser().resolve() / (
        GAIA_TGE_A0_CACHE_FILENAME_TEMPLATE.format(level=int(level))
    )


def _load_dustmaps_fetch_dependencies():
    try:
        import dustmaps.gaia_tge as gaia_tge
        from dustmaps.config import config as dustmaps_config
    except ImportError as exc:  # pragma: no cover - import availability is environment-specific
        raise DustError(
            "dustmaps must be installed to fetch Gaia TGE dust fields"
        ) from exc
    return dustmaps_config, gaia_tge


def fetch_gaia_tge_dataset(dust_root: Path) -> Path:
    """Fetch Gaia TGE into one explicit `dust_root` and return the dataset path."""

    dust_root = Path(dust_root).expanduser().resolve()
    filename = dust_root / GAIA_TGE_RELATIVE_FILENAME
    if filename.is_file():
        return filename

    dust_root.mkdir(parents=True, exist_ok=True)
    filename.parent.mkdir(parents=True, exist_ok=True)

    dustmaps_config, gaia_tge = _load_dustmaps_fetch_dependencies()
    previous_data_dir = dustmaps_config.get("data_dir")
    try:
        dustmaps_config["data_dir"] = str(dust_root)
        gaia_tge.fetch()
    except Exception as exc:
        raise DustError(f"Failed to fetch Gaia TGE dust data into {dust_root}") from exc
    finally:
        if previous_data_dir is None:
            dustmaps_config.remove("data_dir")
        else:
            dustmaps_config["data_dir"] = previous_data_dir

    if not filename.is_file():
        raise DustError(f"Gaia TGE fetch did not produce expected file: {filename}")
    return filename


def prepare_gaia_tge_a0_cache(
    *,
    source_dust_root: Path,
    destination_dust_root: Path,
    level: int,
    force: bool = False,
) -> Path:
    """Build a dense mmap-friendly Gaia TGE A0 cache for one HEALPix level."""

    level = int(level)
    if level < 0:
        raise DustError(f"level must be non-negative, got {level}")

    destination = gaia_tge_a0_cache_filename(destination_dust_root, level)
    if destination.is_file() and not force:
        return destination

    source = gaia_tge_map_filename(source_dust_root)
    source_max_level = _get_source_max_optimum_level(source)
    source_values = _build_dense_a0_values(source, source_max_level)
    if level == source_max_level:
        values = source_values
    elif level > source_max_level:
        values = np.repeat(source_values, 4 ** (level - source_max_level))
    else:
        sample_pixs = np.arange(12 * (4**level), dtype=np.int64)
        sample_coords = get_pixel_skycoord(level, sample_pixs)
        source_pixs = np.asarray(
            get_pixel_from_skycoord(source_max_level, sample_coords),
            dtype=np.int64,
        )
        values = source_values[source_pixs]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}-",
        suffix=".tmp",
        delete=False,
    ) as tmp_handle:
        tmp_path = Path(tmp_handle.name)
        np.save(tmp_handle, np.asarray(values, dtype=np.float32))
    try:
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    _load_gaia_tge_a0_values.cache_clear()
    return destination


def _get_source_max_optimum_level(map_filename: Path) -> int:
    levels = [
        int(row["healpix_level"])
        for row in _iter_gaia_tge_rows(map_filename)
        if _parse_ecsv_bool(row["optimum_hpx_flag"])
    ]
    if not levels:
        raise DustError(f"Gaia TGE map has no optimum rows: {map_filename}")
    return max(levels)


def _build_dense_a0_values(map_filename: Path, level: int) -> np.ndarray:
    values = np.full(12 * (4**level), np.nan, dtype=np.float32)
    for row in _iter_gaia_tge_rows(map_filename):
        if not _parse_ecsv_bool(row["optimum_hpx_flag"]):
            continue
        row_level = int(row["healpix_level"])
        healpix_id = int(row["healpix_id"])
        a0 = _parse_ecsv_float(row["a0"])
        if row_level > level:
            raise DustError(
                "Cannot build Gaia TGE A0 cache because source rows are finer "
                f"than the dense source level ({row_level} > {level})"
            )
        block_size = 4 ** (level - row_level)
        start = healpix_id * block_size
        values[start : start + block_size] = a0
    return values


def _iter_gaia_tge_rows(map_filename: Path):
    with gzip_open(map_filename, "rt", encoding="utf-8", newline="") as handle:
        rows = (line for line in handle if line.strip() and not line.startswith("#"))
        yield from csv.DictReader(rows)


def _parse_ecsv_bool(value: str) -> bool:
    return value.strip().strip('"').lower() == "true"


def _parse_ecsv_float(value: str) -> float:
    normalized = value.strip().strip('"').lower()
    if normalized in {"", "null", "nan"}:
        return float(np.nan)
    return float(value)


@lru_cache(maxsize=None)
def _load_gaia_tge_a0_values(cache_filename: str) -> np.ndarray:
    return np.load(cache_filename, mmap_mode="r")


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

    cache_filename = gaia_tge_a0_cache_filename(dust_root, max_data_level)
    if not cache_filename.is_file():
        try:
            cache_filename = prepare_gaia_tge_a0_cache(
                source_dust_root=dust_root,
                destination_dust_root=dust_root,
                level=max_data_level,
            )
        except DustError as exc:
            raise DustError(
                "Gaia TGE A0 cache not found. Build initialization should create "
                f"{cache_filename}"
            ) from exc
    a0_values = _load_gaia_tge_a0_values(str(cache_filename))

    sample_pixs = get_subpixels(outer_level, outer_pix, max_data_level)
    values = np.asarray(a0_values[sample_pixs], dtype=np.float64)
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
