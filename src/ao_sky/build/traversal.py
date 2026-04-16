"""Native Traversal pipeline for retained asterisms and rich inner products."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
import numpy as np

from ..asterisms import AsterismSearchOptions, find_asterisms, load_asterism_stars
from ..dust import add_gaia_a0_to_inner
from ..gaia import GAIA_SCHEMA_COLUMNS, GaiaHealpixStore, compute_legacy_r_magnitude
from ..predict import (
    clear_backend_cache,
    get_mean_model,
    get_point_model,
    get_rotation_angles,
    get_seeing_baseline_performance,
    predict_asterism_ee,
    predict_field_mean_batch,
    predict_point_batch,
)
from ..predict._models import PredictRuntime, SeeingBaselinePerformance
from ..spatial import (
    get_parent_pixel,
    get_pixel_area,
    get_pixel_from_skycoord,
    get_pixel_neighbours,
    get_pixel_resolution,
    get_pixel_skycoord,
    get_subpixels,
)
from ._exceptions import BuildError
from .runtime_gaia import RUNTIME_HPX_COLUMN, RUNTIME_HPX_LEVEL

ASTERISM_BOUNDARY_RINGS = 2


@dataclass(slots=True)
class TraversalGeometry:
    """Worker-local geometry values reused while building outer pixels."""

    outer_level: int
    inner_level: int
    fov_level: int
    fov_level_area_arcmin2: float
    inner_resolution: u.Quantity
    baseline: SeeingBaselinePerformance
    min_galactic_latitude: float | None
    _inner_pixs: dict[int, np.ndarray] = field(default_factory=dict)
    _inner_centres: dict[int, SkyCoord] = field(default_factory=dict)
    _skip_asterisms: dict[int, tuple[bool, str]] = field(default_factory=dict)

    @classmethod
    def from_runtime(cls, runtime: PredictRuntime) -> TraversalGeometry:
        """Build static traversal geometry from the native runtime contract."""

        fov_level = _get_level_with_resolution(runtime.ao_system.fov)
        return cls(
            outer_level=runtime.outer_level,
            inner_level=runtime.inner_level,
            fov_level=fov_level,
            fov_level_area_arcmin2=get_pixel_area(fov_level).to(u.arcmin**2).value,
            inner_resolution=get_pixel_resolution(runtime.inner_level),
            baseline=get_seeing_baseline_performance(runtime),
            min_galactic_latitude=runtime.min_galactic_latitude,
        )

    def inner_pixs(self, outer_pix: int) -> np.ndarray:
        outer_pix = int(outer_pix)
        if outer_pix not in self._inner_pixs:
            self._inner_pixs[outer_pix] = get_subpixels(
                self.outer_level,
                outer_pix,
                self.inner_level,
            )
        return self._inner_pixs[outer_pix]

    def inner_centres(self, outer_pix: int) -> SkyCoord:
        outer_pix = int(outer_pix)
        if outer_pix not in self._inner_centres:
            self._inner_centres[outer_pix] = get_pixel_skycoord(
                self.inner_level,
                self.inner_pixs(outer_pix),
            )
        return self._inner_centres[outer_pix]

    def should_skip_asterisms(self, outer_pix: int) -> tuple[bool, str]:
        """Return the cached Galactic-latitude skip decision for an outer pixel."""

        outer_pix = int(outer_pix)
        if outer_pix not in self._skip_asterisms:
            if self.min_galactic_latitude is None:
                self._skip_asterisms[outer_pix] = (False, "")
            else:
                coord = get_pixel_skycoord(self.outer_level, outer_pix)
                if abs(coord.galactic.b.degree) < self.min_galactic_latitude:
                    self._skip_asterisms[outer_pix] = (
                        True,
                        f"min_galactic_latitude<{self.min_galactic_latitude:g}",
                    )
                else:
                    self._skip_asterisms[outer_pix] = (False, "")
        return self._skip_asterisms[outer_pix]


def _require_runtime_gaia_columns(table: Table) -> None:
    missing = [name for name in ("R", RUNTIME_HPX_COLUMN) if name not in table.colnames]
    if missing:
        raise BuildError(
            "Native Traversal requires runtime Gaia rows with columns: "
            + ", ".join(missing)
        )


def _get_runtime_table_pixels(table: Table, level: int) -> np.ndarray:
    _require_runtime_gaia_columns(table)
    if level > RUNTIME_HPX_LEVEL:
        raise BuildError(
            f"Runtime Gaia {RUNTIME_HPX_COLUMN} cannot derive finer level {level}"
        )
    hpx = np.asarray(table[RUNTIME_HPX_COLUMN], dtype=np.int64)
    pixels = np.full(hpx.shape, -1, dtype=np.int64)
    valid = hpx >= 0
    if np.any(valid):
        pixels[valid] = np.asarray(
            get_parent_pixel(RUNTIME_HPX_LEVEL, hpx[valid], level),
            dtype=np.int64,
        )
    return pixels


def _valid_runtime_gaia_mask(table: Table) -> np.ndarray:
    _require_runtime_gaia_columns(table)
    return (
        np.isfinite(np.asarray(table["ra"], dtype=np.float64))
        & np.isfinite(np.asarray(table["dec"], dtype=np.float64))
        & np.isfinite(np.asarray(table["R"], dtype=np.float64))
        & (np.asarray(table[RUNTIME_HPX_COLUMN], dtype=np.int64) >= 0)
    )


def _get_level_with_resolution(target_resolution: u.Quantity) -> int:
    target = target_resolution.to(u.arcmin)
    for level in range(0, 30):
        if get_pixel_resolution(level).to(u.arcmin) <= target:
            return level
    raise BuildError(f"Could not find HEALPix level for resolution {target_resolution}")


def should_skip_asterisms(
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
) -> tuple[bool, str]:
    """Return whether asterisms should be skipped for one outer pixel."""

    if geometry is not None:
        return geometry.should_skip_asterisms(outer_pix)

    if runtime.min_galactic_latitude is None:
        return False, ""

    coord = get_pixel_skycoord(runtime.outer_level, outer_pix)
    if abs(coord.galactic.b.degree) < runtime.min_galactic_latitude:
        return True, f"min_galactic_latitude<{runtime.min_galactic_latitude:g}"
    return False, ""


def _get_band_values(table: Table, band: str) -> np.ndarray:
    if band in table.colnames:
        return np.asarray(table[band], dtype=np.float64)
    raise BuildError(f"Unsupported build band {band!r}")


def _get_inner_count_band_values(table: Table, band: str) -> np.ndarray:
    if band in table.colnames:
        return np.asarray(table[band], dtype=np.float64)
    if band == "R":
        return np.asarray(
            compute_legacy_r_magnitude(table[list(GAIA_SCHEMA_COLUMNS)]),
            dtype=np.float64,
        )
    raise BuildError(f"Unsupported build band {band!r}")


def _get_inner_count_pixels(table: Table, level: int) -> np.ndarray:
    pixels = np.full((len(table),), -1, dtype=np.int64)
    valid = np.isfinite(np.asarray(table["ra"], dtype=np.float64)) & np.isfinite(
        np.asarray(table["dec"], dtype=np.float64)
    )
    if np.any(valid):
        pixels[valid] = np.asarray(
            get_pixel_from_skycoord(
                level,
                SkyCoord(
                    ra=table["ra"][valid],
                    dec=table["dec"][valid],
                    unit=(u.degree, u.degree),
                ),
            ),
            dtype=np.int64,
        )
    return pixels


def _load_inner_count_stars(store: GaiaHealpixStore, outer_pix: int) -> Table:
    raw_store = getattr(store, "store", store)
    return raw_store.load_healpix(outer_pix, read_only=True)


def _filter_neighbours_by_galactic_latitude(
    stars: Table,
    *,
    outer_pix: int,
    geometry: TraversalGeometry,
) -> Table:
    if (
        geometry.min_galactic_latitude is None
        or "source_outer_pix" not in stars.colnames
    ):
        return stars

    keep = np.ones(len(stars), dtype=np.bool_)
    source_outer_pixs = np.asarray(stars["source_outer_pix"], dtype=np.int64)
    unique_source_pixels = np.unique(source_outer_pixs)
    for source_outer_pix in unique_source_pixels:
        if source_outer_pix == outer_pix:
            continue
        skip, _ = geometry.should_skip_asterisms(int(source_outer_pix))
        if skip:
            keep &= source_outer_pixs != int(source_outer_pix)
    return stars[keep]


def prepare_search_inputs(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
    boundary_rings: int = ASTERISM_BOUNDARY_RINGS,
) -> tuple[Table, Table]:
    """Return the prepared search-star and NGS tables for one outer pixel."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=geometry.fov_level,
        boundary_rings=boundary_rings,
        include_locality=True,
    )
    stars = _filter_neighbours_by_galactic_latitude(
        stars,
        outer_pix=outer_pix,
        geometry=geometry,
    )
    _require_runtime_gaia_columns(stars)
    stars = stars[_valid_runtime_gaia_mask(stars)]
    stars["pix"] = _get_runtime_table_pixels(stars, geometry.fov_level)

    ngs = stars[
        (stars["R"] >= runtime.ao_system.min_mag) & (stars["R"] < runtime.ao_system.max_mag)
    ]

    if runtime.max_star_density is not None and len(stars) > 0:
        unique_pixs, star_counts = np.unique(
            np.asarray(stars["pix"], dtype=np.int64),
            return_counts=True,
        )
        remove_pixs = unique_pixs[
            star_counts / geometry.fov_level_area_arcmin2 > runtime.max_star_density
        ]
        if len(remove_pixs) > 0:
            ngs = ngs[~np.isin(ngs["pix"], remove_pixs)]

    return stars, ngs


def _get_circle_overlap_area(radius1: float, radius2: float, separation: float) -> float:
    if separation >= radius1 + radius2:
        return 0.0
    if separation <= abs(radius1 - radius2):
        return float(np.pi * min(radius1, radius2) ** 2)

    term1 = radius1**2 * np.arccos(
        (separation**2 + radius1**2 - radius2**2) / (2.0 * separation * radius1)
    )
    term2 = radius2**2 * np.arccos(
        (separation**2 + radius2**2 - radius1**2) / (2.0 * separation * radius2)
    )
    term3 = 0.5 * np.sqrt(
        (-separation + radius1 + radius2)
        * (separation + radius1 - radius2)
        * (separation - radius1 + radius2)
        * (separation + radius1 + radius2)
    )
    return float(term1 + term2 - term3)


def _compute_relative_polar_coords(
    center_ra: float,
    center_dec: float,
    ra: float,
    dec: float,
) -> tuple[float, float]:
    ra0 = np.deg2rad(center_ra)
    dec0 = np.deg2rad(center_dec)
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    delta_ra = ra_rad - ra0
    delta_dec = dec_rad - dec0
    sin_ddec2 = np.sin(delta_dec / 2.0)
    sin_dra2 = np.sin(delta_ra / 2.0)
    a = sin_ddec2**2 + np.cos(dec0) * np.cos(dec_rad) * sin_dra2**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = np.rad2deg(c) * 3600.0
    y = np.sin(delta_ra) * np.cos(dec_rad)
    x = np.cos(dec0) * np.sin(dec_rad) - np.sin(dec0) * np.cos(dec_rad) * np.cos(delta_ra)
    theta = np.rad2deg(np.arctan2(y, x))
    return r, theta


def _get_ngs_from_asterisms(asterisms: Table) -> list[list[dict[str, float]]]:
    ngs: list[list[dict[str, float]]] = []
    for asterism in asterisms:
        stars: list[dict[str, float]] = []
        for star_idx in range(1, int(asterism["num_stars"]) + 1):
            r, theta = _compute_relative_polar_coords(
                float(asterism["ra"]),
                float(asterism["dec"]),
                float(asterism[f"star{star_idx}_ra"]),
                float(asterism[f"star{star_idx}_dec"]),
            )
            stars.append(
                {
                    "zd": float(r),
                    "az": float(theta),
                    "mag": float(asterism[f"star{star_idx}_mag"]),
                }
            )
        ngs.append(stars)
    return ngs


def _get_asterisms_ee(asterisms: Table, runtime: PredictRuntime, batch_size: int = 10000) -> np.ndarray:
    if len(asterisms) == 0 or not runtime.ao_system.mean_models:
        return np.zeros((len(asterisms),), dtype=np.float64)

    qualities = np.zeros((len(asterisms),), dtype=np.float64)
    best_angles = np.zeros((len(asterisms),), dtype=np.float64)
    for num_stars in range(runtime.ao_system.min_wfs, runtime.ao_system.max_wfs + 1):
        indexes = np.flatnonzero(np.asarray(asterisms["num_stars"], dtype=np.int64) == num_stars)
        if len(indexes) == 0:
            continue
        model = get_mean_model(runtime, num_stars)
        if model is None:
            continue
        num_batches = int(np.ceil(len(indexes) / batch_size))
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(indexes))
            batch_indexes = indexes[start_idx:end_idx]
            if len(batch_indexes) == 0:
                continue
            batch_qualities, batch_angles = predict_asterism_ee(
                runtime,
                num_stars=num_stars,
                model=model,
                ngs=_get_ngs_from_asterisms(asterisms[batch_indexes]),
            )
            qualities[batch_indexes] = batch_qualities
            best_angles[batch_indexes] = batch_angles
        clear_backend_cache()

    asterisms["best_ee"] = qualities
    asterisms["best_angle"] = best_angles
    return qualities


def _get_asterism_quality(asterisms: Table, runtime: PredictRuntime) -> np.ndarray:
    if runtime.ao_system.mean_models:
        return _get_asterisms_ee(asterisms, runtime)

    max_separation = runtime.ao_system.fov.to(u.arcsec).value
    radius_1ngs = runtime.ao_system.fov_1ngs.to(u.arcsec).value / 2.0
    min_rel_factor_small = 0.25
    min_rel_factor_large = 0.5
    min_rel_sep = radius_1ngs / max_separation
    mid_rel_sep = 0.5
    max_rel_sep = 1.0
    below_mid_slope = (1.0 - min_rel_factor_small) / (mid_rel_sep - min_rel_sep)
    above_mid_slope = (1.0 - min_rel_factor_large) / (max_rel_sep - mid_rel_sep)
    qualities = np.zeros((len(asterisms),), dtype=np.float64)

    for index, asterism in enumerate(asterisms):
        rel_sep = (
            float(asterism["relative_separation"])
            if int(asterism["num_stars"]) > 1
            else min_rel_sep
        )
        if rel_sep < 0.5:
            rel_factor = max(
                min_rel_factor_small,
                1.0 - below_mid_slope * (mid_rel_sep - rel_sep),
            )
        else:
            rel_factor = max(
                min_rel_factor_large,
                1.0 - above_mid_slope * (rel_sep - mid_rel_sep),
            )

        mag_factors: list[float] = []
        for star_idx in range(1, int(asterism["num_stars"]) + 1):
            if runtime.ao_system.max_mag == runtime.ao_system.nom_mag:
                mag_factors.append(1.0)
            else:
                mag_factors.append(
                    min(
                        1.0,
                        (runtime.ao_system.max_mag - float(asterism[f"star{star_idx}_mag"]))
                        / (runtime.ao_system.max_mag - runtime.ao_system.nom_mag),
                    )
                )
        while len(mag_factors) < 3:
            mag_factors.append(0.0)
        qualities[index] = rel_factor * (sum(mag_factors) / 3.0)
    return qualities


def _filter_bright_star_exclusion(
    asterisms: Table,
    centres: SkyCoord,
    stars: Table,
    runtime: PredictRuntime,
) -> tuple[Table, SkyCoord]:
    threshold = runtime.max_bright_star_mag
    if threshold is None or len(asterisms) == 0:
        return asterisms, centres

    bright_stars = stars[stars["R"] < threshold]
    if len(bright_stars) == 0:
        return asterisms, centres

    bright_star_coords = SkyCoord(
        ra=bright_stars["ra"],
        dec=bright_stars["dec"],
        unit=(u.degree, u.degree),
    )
    keep = np.asarray(
        [
            np.min(centre.separation(bright_star_coords))
            > 2.0 * runtime.ao_system.fov
            for centre in centres
        ],
        dtype=np.bool_,
    )
    return asterisms[keep], centres[keep]


def _filter_overlaps(
    asterisms: Table,
    centres: SkyCoord,
    runtime: PredictRuntime,
    geometry: TraversalGeometry,
) -> tuple[Table, SkyCoord]:
    threshold = runtime.max_overlap
    if threshold is None or len(asterisms) == 0:
        return asterisms, centres

    fov_radius = runtime.ao_system.fov.to(u.rad).value
    fov_1ngs_radius = runtime.ao_system.fov_1ngs.to(u.rad).value
    asterism_pixs = get_pixel_from_skycoord(geometry.fov_level, centres)
    keep = np.ones(len(asterisms), dtype=np.bool_)
    qualities = _get_asterism_quality(asterisms, runtime)

    for pix in np.unique(asterism_pixs):
        search_pixs = np.concatenate([[pix], get_pixel_neighbours(geometry.fov_level, int(pix))])
        candidate_indexes = np.flatnonzero(keep & np.isin(asterism_pixs, search_pixs))
        if len(candidate_indexes) == 0:
            continue
        candidate_indexes = candidate_indexes[np.argsort(qualities[candidate_indexes])[::-1]]
        skip = np.zeros(len(candidate_indexes), dtype=np.bool_)
        for j, idx1 in enumerate(candidate_indexes):
            if skip[j]:
                continue
            separations = centres[idx1].separation(centres[candidate_indexes[j + 1 :]]).to(u.rad).value
            for offset, separation in enumerate(separations, start=j + 1):
                if skip[offset] or separation > fov_radius:
                    continue
                idx2 = candidate_indexes[offset]
                radius1 = fov_radius if asterisms["num_stars"][idx1] > 1 else fov_1ngs_radius
                radius2 = fov_radius if asterisms["num_stars"][idx2] > 1 else fov_1ngs_radius
                overlap_area = _get_circle_overlap_area(radius1, radius2, float(separation))
                overlap = overlap_area / (np.pi * min(radius1, radius2) ** 2)
                if overlap > threshold:
                    skip[offset] = True
        current_pix = asterism_pixs[candidate_indexes] == pix
        keep[candidate_indexes[current_pix & skip]] = False
    return asterisms[keep], centres[keep]


def _filter_relative_constraints(
    asterisms: Table,
    centres: SkyCoord,
    runtime: PredictRuntime,
) -> tuple[Table, SkyCoord]:
    result = asterisms
    result_centres = centres
    if runtime.ao_system.max_rel_sep > 0:
        keep = (
            np.asarray(result["relative_separation"], dtype=np.float64) >= runtime.ao_system.min_rel_sep
        ) & (
            np.asarray(result["relative_separation"], dtype=np.float64) < runtime.ao_system.max_rel_sep
        )
        result = result[keep]
        result_centres = result_centres[keep]
    if runtime.ao_system.max_rel_area > 0 and len(result) > 0:
        keep = (
            np.asarray(result["relative_area"], dtype=np.float64) >= runtime.ao_system.min_rel_area
        ) & (
            np.asarray(result["relative_area"], dtype=np.float64) < runtime.ao_system.max_rel_area
        )
        result = result[keep]
        result_centres = result_centres[keep]
    return result, result_centres


def build_outer_pixel_asterisms(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
) -> tuple[Table, Table, Table]:
    """Return the search stars, filtered NGS, and retained asterisms."""

    stars, ngs, asterisms, _ = _build_expanded_outer_pixel_asterisms(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
    )
    if len(asterisms) == 0:
        return stars, ngs, asterisms

    keep = (
        get_parent_pixel(
            runtime.inner_level,
            np.asarray(asterisms["pix"], dtype=np.int64),
            runtime.outer_level,
        )
        == int(outer_pix)
    )
    return stars, ngs, asterisms[keep]


def _build_expanded_outer_pixel_asterisms(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
    boundary_rings: int = ASTERISM_BOUNDARY_RINGS,
) -> tuple[Table, Table, Table, SkyCoord]:
    """Return retained asterisms from one outer pixel's expanded star footprint."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    stars, ngs = prepare_search_inputs(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
        boundary_rings=boundary_rings,
    )
    options = AsterismSearchOptions(
        min_stars=runtime.ao_system.min_wfs,
        max_stars=runtime.ao_system.max_wfs,
        min_separation_arcsec=runtime.ao_system.min_sep.to(u.arcsec).value,
        max_separation_arcsec=runtime.ao_system.max_sep.to(u.arcsec).value,
        max_single_star_radius_arcsec=runtime.ao_system.fov_1ngs.to(u.arcsec).value / 2.0,
    )
    asterisms = find_asterisms(ngs, options).copy(copy_data=True)
    centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
    asterisms, centres = _filter_bright_star_exclusion(asterisms, centres, stars, runtime)
    asterisms, centres = _filter_overlaps(asterisms, centres, runtime, geometry)
    asterisms, centres = _filter_relative_constraints(asterisms, centres, runtime)
    if len(asterisms) > 0:
        asterisms["pix"] = get_pixel_from_skycoord(runtime.inner_level, centres)
    else:
        asterisms["pix"] = np.array([], dtype=np.int64)
    return stars, ngs, asterisms, centres


def _empty_persisted_asterisms() -> Table:
    return Table(
        [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ],
        names=(
            "asterism_id",
            "ra",
            "dec",
            "num_stars",
            "pix",
            "star1_source_id",
            "star1_ra",
            "star1_dec",
            "star1_mag",
            "star2_source_id",
            "star2_ra",
            "star2_dec",
            "star2_mag",
            "star3_source_id",
            "star3_ra",
            "star3_dec",
            "star3_mag",
        ),
    )


def _to_persisted_asterisms(asterisms: Table | None) -> Table:
    if asterisms is None or len(asterisms) == 0:
        return _empty_persisted_asterisms()

    result = Table()
    result["asterism_id"] = np.asarray(asterisms["asterism_id"], dtype=np.int64)
    result["ra"] = np.asarray(asterisms["ra"], dtype=np.float64)
    result["dec"] = np.asarray(asterisms["dec"], dtype=np.float64)
    result["num_stars"] = np.asarray(asterisms["num_stars"], dtype=np.int64)
    result["pix"] = np.asarray(asterisms["pix"], dtype=np.int64)
    for index in (1, 2, 3):
        result[f"star{index}_source_id"] = np.asarray(asterisms[f"star{index}_source_id"], dtype=np.int64)
        result[f"star{index}_ra"] = np.asarray(asterisms[f"star{index}_ra"], dtype=np.float64)
        result[f"star{index}_dec"] = np.asarray(asterisms[f"star{index}_dec"], dtype=np.float64)
        result[f"star{index}_mag"] = np.asarray(asterisms[f"star{index}_mag"], dtype=np.float64)
    return result


def build_base_inner_table(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
    asterisms: Table | None = None,
) -> Table:
    """Return the dense base inner table for one outer pixel."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    local_stars = _load_inner_count_stars(store, outer_pix)
    pixs = geometry.inner_pixs(outer_pix)

    gaia_pixs = _get_inner_count_pixels(local_stars, runtime.inner_level)
    valid_gaia_pixs = np.asarray(gaia_pixs, dtype=np.int64) >= 0
    unique_pixs, counts = np.unique(
        np.asarray(gaia_pixs, dtype=np.int64)[valid_gaia_pixs],
        return_counts=True,
    )
    count_map = dict(zip(unique_pixs, counts, strict=False))

    band_values = _get_inner_count_band_values(local_stars, runtime.ao_system.band)
    ngs_mask = valid_gaia_pixs & np.isfinite(band_values)
    ngs_mask &= band_values >= runtime.ao_system.min_mag
    ngs_mask &= band_values < runtime.ao_system.max_mag
    if np.any(ngs_mask):
        ngs_unique, ngs_counts = np.unique(
            np.asarray(gaia_pixs, dtype=np.int64)[ngs_mask],
            return_counts=True,
        )
        ngs_count_map = dict(zip(ngs_unique, ngs_counts, strict=False))
    else:
        ngs_count_map = {}

    if asterisms is not None and len(asterisms) > 0:
        asterism_unique, asterism_counts = np.unique(
            np.asarray(asterisms["pix"], dtype=np.int64),
            return_counts=True,
        )
        asterism_count_map = dict(zip(asterism_unique, asterism_counts, strict=False))
    else:
        asterism_count_map = {}

    baseline = geometry.baseline
    size = len(pixs)
    return Table(
        [
            np.asarray(pixs, dtype=np.int64),
            np.asarray([count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.asarray([ngs_count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.asarray([asterism_count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.full((size,), np.nan if baseline.ee is None else baseline.ee, dtype=np.float64),
            np.full((size,), np.nan if baseline.sr is None else baseline.sr, dtype=np.float64),
            np.full((size,), np.nan if baseline.fwhm is None else baseline.fwhm, dtype=np.float64),
            np.full((size,), -1, dtype=np.int64),
            np.full((size,), np.nan, dtype=np.float64),
            np.full((size,), np.nan if baseline.ee is None else baseline.ee, dtype=np.float64),
            np.full((size,), np.nan, dtype=np.float64),
            np.zeros((size,), dtype=np.bool_),
            np.zeros((size,), dtype=np.bool_),
        ],
        names=(
            "pix",
            "star_count",
            "ngs_count",
            "asterism_count",
            "best_ee",
            "best_sr",
            "best_fwhm",
            "winner_asterism_id",
            "winner_distance_arcsec",
            "winner_ee_resolved",
            "winner_ee_averaged",
            "coverage_resolved",
            "coverage_averaged",
        ),
    )


def _get_plane_offsets_arcsec(reference_coord: SkyCoord, skycoords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    lon_offset, lat_offset = reference_coord.spherical_offsets_to(skycoords)
    return lon_offset.to(u.arcsec).value, lat_offset.to(u.arcsec).value


def _build_context(
    runtime: PredictRuntime,
    outer_pix: int,
    asterisms: Table,
    local_asterism_mask: np.ndarray,
    *,
    geometry: TraversalGeometry | None = None,
) -> SimpleNamespace:
    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    context = SimpleNamespace()
    context.outer_pix = outer_pix
    context.pixel_idxs = np.array([], dtype=np.int64)
    context.asterism_idxs = np.array([], dtype=np.int64)
    context.inner_centres = geometry.inner_centres(outer_pix)
    outer_centre = get_pixel_skycoord(runtime.outer_level, outer_pix)
    context.inner_x, context.inner_y = _get_plane_offsets_arcsec(outer_centre, context.inner_centres)

    if len(asterisms) == 0:
        context.asterisms = None
        context.local_asterism_mask = np.array([], dtype=np.bool_)
        return context

    context.asterisms = asterisms
    context.local_asterism_mask = np.asarray(local_asterism_mask, dtype=np.bool_)
    asterism_catalog = SkyCoord(
        ra=context.asterisms["ra"],
        dec=context.asterisms["dec"],
        unit=(u.degree, u.degree),
    )
    context.asterism_x, context.asterism_y = _get_plane_offsets_arcsec(outer_centre, asterism_catalog)
    context.pixel_idxs, context.asterism_idxs, _, _ = search_around_sky(
        context.inner_centres,
        asterism_catalog,
        (runtime.ao_system.fov - geometry.inner_resolution) / 2,
    )
    context.star_x = {}
    context.star_y = {}
    num_stars = np.asarray(context.asterisms["num_stars"], dtype=np.int64)
    for star_idx in range(1, 4):
        if not np.any(num_stars >= star_idx):
            continue
        star_coords = SkyCoord(
            ra=context.asterisms[f"star{star_idx}_ra"],
            dec=context.asterisms[f"star{star_idx}_dec"],
            unit=(u.degree, u.degree),
        )
        context.star_x[star_idx], context.star_y[star_idx] = _get_plane_offsets_arcsec(
            outer_centre,
            star_coords,
        )
    return context


def _get_valid_ngs_from_context_pair(
    context: SimpleNamespace,
    pixel_idx: int,
    asterism_idx: int,
    *,
    field_radius: float,
    field_radius_1ngs: float,
) -> list[dict[str, float]]:
    all_stars: list[dict[str, float]] = []
    pixel_x = context.inner_x[pixel_idx]
    pixel_y = context.inner_y[pixel_idx]
    num_stars = int(context.asterisms["num_stars"][asterism_idx])
    for star_idx in range(1, num_stars + 1):
        dx = context.star_x[star_idx][asterism_idx] - pixel_x
        dy = context.star_y[star_idx][asterism_idx] - pixel_y
        zd = float(np.hypot(dx, dy))
        all_stars.append(
            {
                "zd": zd,
                "az": float(np.rad2deg(np.arctan2(dx, dy))),
                "mag": float(context.asterisms[f"star{star_idx}_mag"][asterism_idx]),
            }
        )

    stars_in_fov = [star for star in all_stars if star["zd"] <= field_radius]
    if len(stars_in_fov) >= 2:
        return stars_in_fov
    return [star for star in all_stars if star["zd"] <= field_radius_1ngs]


def _update_inner_pixel_asterism_performance(
    runtime: PredictRuntime,
    inner: Table,
    context: SimpleNamespace,
    *,
    batch_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field_radius = runtime.ao_system.fov.to(u.arcsec).value / 2.0
    field_radius_1ngs = runtime.ao_system.fov_1ngs.to(u.arcsec).value / 2.0
    winner_asterism_idxs = np.full((len(inner),), -1, dtype=np.int64)
    winner_angles = np.full((len(inner),), np.nan, dtype=np.float64)
    winner_ngs_payloads = np.empty((len(inner),), dtype=object)
    winner_ngs_payloads[:] = None
    if len(context.pixel_idxs) == 0:
        return winner_asterism_idxs, winner_angles, winner_ngs_payloads

    candidate_pixel_idxs = context.pixel_idxs
    candidate_asterism_idxs = context.asterism_idxs
    num_batches = int(np.ceil(len(candidate_pixel_idxs) / batch_size))
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(candidate_pixel_idxs))
        batch_pixel_idxs = candidate_pixel_idxs[start_idx:end_idx]
        batch_asterism_idxs = candidate_asterism_idxs[start_idx:end_idx]
        grouped_pairs: dict[int, dict[str, list]] = {}
        for row_idx, (pixel_idx, asterism_idx) in enumerate(zip(batch_pixel_idxs, batch_asterism_idxs, strict=True)):
            ngs = _get_valid_ngs_from_context_pair(
                context,
                int(pixel_idx),
                int(asterism_idx),
                field_radius=field_radius,
                field_radius_1ngs=field_radius_1ngs,
            )
            surviving_stars = len(ngs)
            if surviving_stars < runtime.ao_system.min_wfs:
                continue
            group = grouped_pairs.setdefault(
                surviving_stars,
                {"row_idxs": [], "pixel_idxs": [], "asterism_idxs": [], "ngs": []},
            )
            group["row_idxs"].append(row_idx)
            group["pixel_idxs"].append(int(pixel_idx))
            group["asterism_idxs"].append(int(asterism_idx))
            group["ngs"].append(ngs)

        for surviving_stars, group in grouped_pairs.items():
            model = get_point_model(runtime, surviving_stars)
            if model is None:
                continue
            metrics = predict_point_batch(
                runtime,
                num_stars=surviving_stars,
                model=model,
                ngs=group["ngs"],
            )
            for result_idx, (pixel_idx, asterism_idx, ngs, sr, ee, fwhm) in enumerate(
                zip(
                    group["pixel_idxs"],
                    group["asterism_idxs"],
                    group["ngs"],
                    metrics.sr,
                    metrics.ee,
                    metrics.fwhm,
                    strict=True,
                )
            ):
                if np.isfinite(sr) and (
                    np.isnan(inner["best_sr"][pixel_idx]) or sr > inner["best_sr"][pixel_idx]
                ):
                    inner["best_sr"][pixel_idx] = float(sr)
                if np.isfinite(ee) and (
                    np.isnan(inner["best_ee"][pixel_idx]) or ee > inner["best_ee"][pixel_idx]
                ):
                    inner["best_ee"][pixel_idx] = float(ee)
                if (
                    context.local_asterism_mask[asterism_idx]
                    and np.isfinite(ee)
                    and (
                        np.isnan(inner["winner_ee_resolved"][pixel_idx])
                        or ee > inner["winner_ee_resolved"][pixel_idx]
                    )
                ):
                    inner["winner_ee_resolved"][pixel_idx] = float(ee)
                    winner_angles[pixel_idx] = float(metrics.ee_angle[result_idx])
                    winner_asterism_idxs[pixel_idx] = int(asterism_idx)
                    winner_ngs_payloads[pixel_idx] = ngs
                    inner["winner_asterism_id"][pixel_idx] = int(
                        context.asterisms["asterism_id"][asterism_idx]
                    )
                    inner["winner_distance_arcsec"][pixel_idx] = float(
                        np.hypot(
                            context.asterism_x[asterism_idx] - context.inner_x[pixel_idx],
                            context.asterism_y[asterism_idx] - context.inner_y[pixel_idx],
                        )
                    )
                if np.isfinite(fwhm) and (
                    np.isnan(inner["best_fwhm"][pixel_idx]) or fwhm < inner["best_fwhm"][pixel_idx]
                ):
                    inner["best_fwhm"][pixel_idx] = float(fwhm)
        clear_backend_cache()
    return winner_asterism_idxs, winner_angles, winner_ngs_payloads


def _update_inner_pixel_asterism_field_mean(
    runtime: PredictRuntime,
    inner: Table,
    context: SimpleNamespace,
    winner_asterism_idxs: np.ndarray,
    winner_angles: np.ndarray,
    winner_ngs_payloads: np.ndarray | None = None,
    *,
    batch_size: int = 10000,
) -> None:
    winner_pixel_idxs = np.flatnonzero(winner_asterism_idxs >= 0)
    if len(winner_pixel_idxs) == 0:
        return

    field_radius = runtime.ao_system.fov.to(u.arcsec).value / 2.0
    field_radius_1ngs = runtime.ao_system.fov_1ngs.to(u.arcsec).value / 2.0
    grouped_pairs: dict[int, dict[str, list]] = {}
    for pixel_idx in winner_pixel_idxs:
        asterism_idx = int(winner_asterism_idxs[pixel_idx])
        ngs = None if winner_ngs_payloads is None else winner_ngs_payloads[pixel_idx]
        if ngs is None:
            ngs = _get_valid_ngs_from_context_pair(
                context,
                int(pixel_idx),
                asterism_idx,
                field_radius=field_radius,
                field_radius_1ngs=field_radius_1ngs,
            )
        surviving_stars = len(ngs)
        if surviving_stars < runtime.ao_system.min_wfs:
            continue
        group = grouped_pairs.setdefault(
            surviving_stars,
            {"pixel_idxs": [], "ngs": [], "angles": []},
        )
        group["pixel_idxs"].append(int(pixel_idx))
        group["ngs"].append(ngs)
        group["angles"].append(float(winner_angles[pixel_idx]))

    for surviving_stars, group in grouped_pairs.items():
        model = get_mean_model(runtime, surviving_stars)
        if model is None:
            continue
        num_rows = len(group["pixel_idxs"])
        num_batches = int(np.ceil(num_rows / batch_size))
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_rows)
            batch_pixel_idxs = group["pixel_idxs"][start_idx:end_idx]
            batch_ngs = group["ngs"][start_idx:end_idx]
            batch_angles = group["angles"][start_idx:end_idx]
            ee_field_mean = predict_field_mean_batch(
                runtime,
                num_stars=surviving_stars,
                model=model,
                ngs=batch_ngs,
                rot_angles=batch_angles,
            )
            inner["winner_ee_averaged"][batch_pixel_idxs] = ee_field_mean
        clear_backend_cache()


def build_traversal_products(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    dust_root: Path,
    max_data_level: int,
    geometry: TraversalGeometry | None = None,
) -> tuple[Table, Table]:
    """Return retained asterisms and the rich inner table for one outer pixel."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    skip_asterisms, _ = should_skip_asterisms(runtime, outer_pix, geometry=geometry)
    if skip_asterisms:
        inner = build_base_inner_table(
            store,
            runtime,
            outer_pix,
            geometry=geometry,
            asterisms=None,
        )
        inner = add_gaia_a0_to_inner(
            inner,
            dust_root=dust_root,
            outer_level=runtime.outer_level,
            outer_pix=outer_pix,
            inner_level=runtime.inner_level,
            max_data_level=max_data_level,
        )
        return _empty_persisted_asterisms(), inner

    _, _, expanded_asterisms, _ = _build_expanded_outer_pixel_asterisms(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
    )
    if len(expanded_asterisms) > 0:
        local_asterism_mask = (
            get_parent_pixel(
                runtime.inner_level,
                np.asarray(expanded_asterisms["pix"], dtype=np.int64),
                runtime.outer_level,
            )
            == int(outer_pix)
        )
        local_asterisms = expanded_asterisms[local_asterism_mask]
    else:
        local_asterism_mask = np.array([], dtype=np.bool_)
        local_asterisms = expanded_asterisms
    inner = build_base_inner_table(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
        asterisms=local_asterisms,
    )
    if len(expanded_asterisms) > 0:
        context = _build_context(
            runtime,
            outer_pix,
            expanded_asterisms,
            local_asterism_mask,
            geometry=geometry,
        )
        if context.asterisms is not None and len(context.asterisms) > 0 and len(context.pixel_idxs) > 0:
            (
                winner_asterism_idxs,
                winner_angles,
                winner_ngs_payloads,
            ) = _update_inner_pixel_asterism_performance(
                runtime,
                inner,
                context,
            )
            if runtime.ao_system.mean_models:
                _update_inner_pixel_asterism_field_mean(
                    runtime,
                    inner,
                    context,
                    winner_asterism_idxs,
                    winner_angles,
                    winner_ngs_payloads,
                )
            inner["coverage_resolved"] = (
                np.asarray(inner["winner_ee_resolved"], dtype=np.float64)
                >= runtime.coverage_ee_threshold_resolved
            )
            inner["coverage_averaged"] = (
                np.asarray(inner["winner_ee_averaged"], dtype=np.float64)
                >= runtime.coverage_ee_threshold_mean
            )

    inner = add_gaia_a0_to_inner(
        inner,
        dust_root=dust_root,
        outer_level=runtime.outer_level,
        outer_pix=outer_pix,
        inner_level=runtime.inner_level,
        max_data_level=max_data_level,
    )
    return _to_persisted_asterisms(local_asterisms), inner
