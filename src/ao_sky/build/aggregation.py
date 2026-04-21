"""All-sky map aggregation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..dust import sample_gaia_a0_for_outer_pixel
from ._constants import MAPS_DTYPE, MAPS_MEAN_FIELDS, MAPS_SUM_FIELDS
from ._exceptions import BuildError
from .artifacts import read_outer_aggregation_products, write_maps_artifact
from .control import (
    load_build_definition,
    load_build_roots,
    maps_artifact_filename,
    outer_artifact_filename,
)


def _create_maps_array(level: int) -> np.ndarray:
    npix = 12 * (4**level)
    maps = np.zeros(npix, dtype=MAPS_DTYPE)
    maps["pix"] = np.arange(npix, dtype=np.int64)
    for field in MAPS_MEAN_FIELDS:
        maps[field] = np.nan
    return maps


def _reduce_level(
    pix: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reduced_pix = pix[::4] // 4
    reduced_values: dict[str, np.ndarray] = {}
    for field in MAPS_SUM_FIELDS:
        reduced_values[field] = values[field].reshape(-1, 4).sum(axis=1)
    for field in MAPS_MEAN_FIELDS:
        # Legacy survey_tools used plain mean here, so one NaN child poisoned
        # the coarser pixel. ao-sky intentionally averages finite children for
        # mean fields so partial/missing winner values do not erase valid data.
        grouped = values[field].reshape(-1, 4)
        finite = np.isfinite(grouped)
        counts = finite.sum(axis=1)
        reduced = np.full(len(grouped), np.nan, dtype=np.float64)
        populated = counts > 0
        reduced[populated] = (
            np.where(finite[populated], grouped[populated], 0.0).sum(axis=1)
            / counts[populated]
        )
        reduced_values[field] = reduced
    return reduced_pix, reduced_values


def _add_winner_asterism_counts(
    level_maps: dict[int, np.ndarray],
    *,
    outer_level: int,
    inner_level: int,
    max_data_level: int,
    asterism_pix: np.ndarray,
) -> None:
    """Populate center-owned retained-asterism counts for one outer artifact."""

    if len(asterism_pix) == 0:
        return

    asterism_pix = np.asarray(asterism_pix, dtype=np.int64)
    if int(max_data_level) > int(inner_level):
        raise BuildError(
            "winner_asterism_count cannot be projected above the retained "
            "asterism center-pixel level"
        )

    for level in range(int(outer_level), int(max_data_level) + 1):
        pixels = asterism_pix // 4 ** (int(inner_level) - int(level))
        if np.any(pixels < 0) or np.any(pixels >= len(level_maps[level])):
            raise BuildError(
                "Retained asterism center pixel is outside the map domain "
                f"at level {level}"
            )
        unique_pixels, counts = np.unique(pixels, return_counts=True)
        level_maps[level]["winner_asterism_count"][unique_pixels] += counts


def aggregate_maps(
    build_path: Path,
    *,
    outer_pixs: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Aggregate per-outer-pixel inner products into dense all-sky level maps."""

    definition = load_build_definition(build_path)
    roots = load_build_roots(build_path)
    if outer_pixs is None:
        all_outer_pixs = tuple(int(pix) for pix in range(12 * (4**definition.outer_level)))
    else:
        all_outer_pixs = tuple(int(pix) for pix in outer_pixs)

    level_maps = {
        level: _create_maps_array(level)
        for level in range(definition.outer_level, definition.max_data_level + 1)
    }

    for outer_pix in all_outer_pixs:
        filename = outer_artifact_filename(build_path, definition, outer_pix)
        if not filename.is_file():
            raise BuildError(f"Outer artifact not found for aggregation: {filename}")

        inner, asterism_pix = read_outer_aggregation_products(filename)
        dust = sample_gaia_a0_for_outer_pixel(
            dust_root=roots.dust_root,
            outer_level=definition.outer_level,
            outer_pix=outer_pix,
            inner_level=definition.inner_level,
            max_data_level=definition.max_data_level,
        )
        if len(inner) != len(dust):
            raise BuildError(
                f"Dust values length {len(dust)} does not match inner rows {len(inner)} "
                f"for outer pixel {outer_pix}"
            )

        local_pix = np.asarray(inner["pix"], dtype=np.int64)
        order = np.argsort(local_pix, kind="stable")
        global_pix = local_pix[order]
        level_values = {
            "gaia_A0": np.asarray(dust, dtype=np.float64)[order],
            "star_count": np.asarray(inner["star_count"], dtype=np.int64)[order],
            "ngs_count": np.asarray(inner["ngs_count"], dtype=np.int64)[order],
            "best_sr": np.asarray(inner["best_sr"], dtype=np.float64)[order],
            "best_ee": np.asarray(inner["best_ee"], dtype=np.float64)[order],
            "best_fwhm": np.asarray(inner["best_fwhm"], dtype=np.float64)[order],
            "winner_ee_resolved": np.asarray(inner["winner_ee_resolved"], dtype=np.float64)[order],
            "winner_ee_averaged": np.asarray(inner["winner_ee_averaged"], dtype=np.float64)[order],
            "coverage_resolved": np.asarray(inner["coverage_resolved"], dtype=np.float64)[order],
            "coverage_averaged": np.asarray(inner["coverage_averaged"], dtype=np.float64)[order],
        }

        current_pix = global_pix
        current_values = level_values
        for level in range(definition.inner_level, definition.outer_level - 1, -1):
            if level <= definition.max_data_level:
                maps = level_maps[level]
                for field, values in current_values.items():
                    maps[field][current_pix] = values
            if level > definition.outer_level:
                current_pix, current_values = _reduce_level(current_pix, current_values)

        _add_winner_asterism_counts(
            level_maps,
            outer_level=definition.outer_level,
            inner_level=definition.inner_level,
            max_data_level=definition.max_data_level,
            asterism_pix=asterism_pix,
        )

    return level_maps


def write_maps(
    build_path: Path,
    *,
    level_maps: dict[int, np.ndarray],
) -> None:
    """Persist dense all-sky maps for the given levels."""

    for level, maps in sorted(level_maps.items()):
        write_maps_artifact(maps_artifact_filename(build_path, level), maps=maps)


def build_maps(
    build_path: Path,
    *,
    outer_pixs: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Aggregate and persist all-sky maps."""

    level_maps = aggregate_maps(build_path, outer_pixs=outer_pixs)
    write_maps(build_path, level_maps=level_maps)
    return level_maps
