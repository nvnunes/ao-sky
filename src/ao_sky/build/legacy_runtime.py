"""Temporary legacy-derived runtime policy used by Phase 3 builds."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np
import yaml

from ..asterisms import AsterismSearchOptions, find_asterisms, load_asterism_stars
from ..gaia import GAIA_SCHEMA_COLUMNS, GaiaHealpixStore, apply_proper_motion, compute_legacy_r_magnitude
from ..spatial import (
    get_pixel_area,
    get_pixel_from_skycoord,
    get_pixel_neighbours,
    get_pixel_resolution,
    get_pixel_skycoord,
    get_subpixels,
)
from ._exceptions import BuildError
from ._models import BuildDefinition, LegacyAOSystemRuntime, LegacyBuildRuntime


def _get_legacy_runtime_root(config_path: Path) -> Path:
    return Path(config_path).resolve().parents[1]


def _get_level_with_resolution(target_resolution: u.Quantity) -> int:
    target = target_resolution.to(u.arcmin)
    for level in range(0, 30):
        if get_pixel_resolution(level).to(u.arcmin) <= target:
            return level
    raise BuildError(f"Could not find HEALPix level for resolution {target_resolution}")


def load_legacy_runtime(
    definition: BuildDefinition,
    legacy_config_path: Path,
) -> LegacyBuildRuntime:
    """Load the temporary Phase 3 AO/runtime policy from legacy config."""

    with Path(legacy_config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    systems = raw.get("ao_systems", [])
    ao_system_raw = next(
        (
            item
            for item in systems
            if str(item.get("name", "")).strip() == definition.ao_system_short_name
        ),
        None,
    )
    if ao_system_raw is None:
        raise BuildError(
            f"AO system {definition.ao_system_short_name!r} not found in {legacy_config_path}"
        )

    fov_1ngs = ao_system_raw.get("fov_1ngs", ao_system_raw["fov"])
    ao_system = LegacyAOSystemRuntime(
        name=str(ao_system_raw["name"]),
        band=str(ao_system_raw["band"]),
        fov=float(ao_system_raw["fov"]) * u.arcsec,
        fov_1ngs=float(fov_1ngs) * u.arcsec,
        min_wfs=int(ao_system_raw["min_wfs"]),
        max_wfs=int(ao_system_raw["max_wfs"]),
        min_mag=float(ao_system_raw["min_mag"]),
        nom_mag=float(ao_system_raw.get("nom_mag", ao_system_raw["max_mag"])),
        max_mag=float(ao_system_raw["max_mag"]),
        min_sep=float(ao_system_raw["min_sep"]) * u.arcsec,
        max_sep=float(ao_system_raw["max_sep"]) * u.arcsec,
        min_rel_sep=float(ao_system_raw.get("min_rel_sep", 0.0)),
        max_rel_sep=float(ao_system_raw.get("max_rel_sep", 0.0)),
        min_rel_area=float(ao_system_raw.get("min_rel_area", 0.0)),
        max_rel_area=float(ao_system_raw.get("max_rel_area", 0.0)),
    )

    return LegacyBuildRuntime(
        ao_system=ao_system,
        outer_level=definition.outer_level,
        inner_level=definition.inner_level,
        epoch=definition.epoch,
        min_galactic_latitude=definition.min_galactic_latitude,
        max_star_density=(
            None
            if raw.get("asterisms_max_star_density") is None
            else float(raw["asterisms_max_star_density"])
        ),
        max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        max_overlap=(
            None if raw.get("asterisms_max_overlap") is None else float(raw["asterisms_max_overlap"])
        ),
        legacy_config_path=Path(legacy_config_path).resolve(),
    )


def should_skip_asterisms(runtime: LegacyBuildRuntime, outer_pix: int) -> tuple[bool, str]:
    """Return whether asterisms should be skipped for one outer pixel."""

    if runtime.min_galactic_latitude is None:
        return False, ""

    coord = get_pixel_skycoord(runtime.outer_level, outer_pix)
    if abs(coord.galactic.b.degree) < runtime.min_galactic_latitude:
        return True, f"min_galactic_latitude<{runtime.min_galactic_latitude:g}"
    return False, ""


def _get_band_values(table: Table, band: str) -> np.ndarray:
    if band in table.colnames:
        return np.asarray(table[band], dtype=np.float64)
    if band == "R":
        return compute_legacy_r_magnitude(table[list(GAIA_SCHEMA_COLUMNS)])
    raise BuildError(f"Unsupported build band {band!r}")


def build_inner_table(
    store: GaiaHealpixStore,
    runtime: LegacyBuildRuntime,
    outer_pix: int,
    *,
    asterisms: Table | None = None,
) -> Table:
    """Return the minimal persisted inner table for one outer pixel."""

    local_stars = store.load_healpix(outer_pix)
    pixs = get_subpixels(runtime.outer_level, outer_pix, runtime.inner_level)

    gaia_pixs = get_pixel_from_skycoord(
        runtime.inner_level,
        SkyCoord(ra=local_stars["ra"], dec=local_stars["dec"], unit=(u.degree, u.degree)),
    )
    unique_pixs, counts = np.unique(np.asarray(gaia_pixs, dtype=np.int64), return_counts=True)
    count_map = dict(zip(unique_pixs, counts, strict=False))

    band_values = _get_band_values(local_stars, runtime.ao_system.band)
    ngs_mask = np.isfinite(band_values)
    ngs_mask &= band_values >= runtime.ao_system.min_mag
    ngs_mask &= band_values < runtime.ao_system.max_mag
    ngs_unique, ngs_counts = np.unique(
        np.asarray(gaia_pixs, dtype=np.int64)[ngs_mask], return_counts=True
    ) if np.any(ngs_mask) else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    ngs_count_map = dict(zip(ngs_unique, ngs_counts, strict=False))

    if asterisms is not None and len(asterisms) > 0:
        asterism_unique, asterism_counts = np.unique(
            np.asarray(asterisms["pix"], dtype=np.int64),
            return_counts=True,
        )
        asterism_count_map = dict(zip(asterism_unique, asterism_counts, strict=False))
    else:
        asterism_count_map = {}

    return Table(
        [
            np.asarray(pixs, dtype=np.int64),
            np.asarray([ngs_count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.asarray([asterism_count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
        ],
        names=("pix", "ngs_count", "asterism_count"),
    )


def _filter_neighbours_by_galactic_latitude(
    stars: Table,
    *,
    runtime: LegacyBuildRuntime,
    outer_pix: int,
    neighbour_level: int | None,
) -> Table:
    if (
        runtime.min_galactic_latitude is None
        or "source_outer_pix" not in stars.colnames
        or neighbour_level is None
    ):
        return stars

    keep = np.ones(len(stars), dtype=np.bool_)
    unique_source_pixels = np.unique(np.asarray(stars["source_outer_pix"], dtype=np.int64))
    for source_outer_pix in unique_source_pixels:
        if source_outer_pix == outer_pix:
            continue
        source_coord = get_pixel_skycoord(neighbour_level, int(source_outer_pix))
        if abs(source_coord.galactic.b.degree) < runtime.min_galactic_latitude:
            keep &= np.asarray(stars["source_outer_pix"], dtype=np.int64) != int(source_outer_pix)
    return stars[keep]


def prepare_search_inputs(
    store: GaiaHealpixStore,
    runtime: LegacyBuildRuntime,
    outer_pix: int,
) -> tuple[Table, Table]:
    """Return the live-build star and NGS tables for one outer pixel."""

    fov_level = _get_level_with_resolution(runtime.ao_system.fov)
    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=fov_level,
        include_locality=True,
    )
    stars = _filter_neighbours_by_galactic_latitude(
        stars,
        runtime=runtime,
        outer_pix=outer_pix,
        neighbour_level=fov_level,
    )
    stars["pix"] = get_pixel_from_skycoord(
        fov_level,
        SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree)),
    )

    locality_columns = {
        "is_local": np.asarray(stars["is_local"]).copy(),
        "source_outer_pix": np.asarray(stars["source_outer_pix"]).copy(),
        "pix": np.asarray(stars["pix"]).copy(),
        "R": np.asarray(stars["R"]).copy(),
    }
    shifted = apply_proper_motion(
        stars[list(GAIA_SCHEMA_COLUMNS)],
        epoch=runtime.epoch,
    )
    for name, values in locality_columns.items():
        shifted[name] = values
    stars = shifted

    stars = stars[~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"]) & ~np.isnan(stars["R"])]
    ngs = stars[(stars["R"] >= runtime.ao_system.min_mag) & (stars["R"] < runtime.ao_system.max_mag)]

    if runtime.max_star_density is not None and len(stars) > 0:
        unique_pixs, star_counts = np.unique(
            np.asarray(stars["pix"], dtype=np.int64),
            return_counts=True,
        )
        fov_level_area = get_pixel_area(fov_level).to(u.arcmin**2).value
        remove_pixs = unique_pixs[star_counts / fov_level_area > runtime.max_star_density]
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


def _get_asterism_quality(asterisms: Table, runtime: LegacyBuildRuntime) -> np.ndarray:
    model_qualities = _get_model_based_asterism_quality(asterisms, runtime)
    if model_qualities is not None:
        return model_qualities

    ao_system = runtime.ao_system
    max_separation = ao_system.fov.to(u.arcsec).value
    radius_1ngs = ao_system.fov_1ngs.to(u.arcsec).value / 2.0
    min_rel_factor_small = 0.25
    min_rel_factor_large = 0.5
    min_rel_sep = radius_1ngs / max_separation
    mid_rel_sep = 0.5
    max_rel_sep = 1.0

    below_mid_slope = (1.0 - min_rel_factor_small) / (mid_rel_sep - min_rel_sep)
    above_mid_slope = (1.0 - min_rel_factor_large) / (max_rel_sep - mid_rel_sep)
    qualities = np.zeros(len(asterisms), dtype=np.float64)

    for index, asterism in enumerate(asterisms):
        rel_sep = float(asterism["relative_separation"]) if int(asterism["num_stars"]) > 1 else min_rel_sep
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

        star_mags = [float(asterism["star1_mag"])]
        if int(asterism["num_stars"]) >= 2:
            star_mags.append(float(asterism["star2_mag"]))
        if int(asterism["num_stars"]) >= 3:
            star_mags.append(float(asterism["star3_mag"]))

        mag_factors = []
        for star_mag in star_mags:
            if ao_system.max_mag == ao_system.nom_mag:
                mag_factors.append(1.0)
            else:
                mag_factors.append(
                    min(1.0, (ao_system.max_mag - star_mag) / (ao_system.max_mag - ao_system.nom_mag))
                )
        while len(mag_factors) < 3:
            mag_factors.append(0.0)

        qualities[index] = rel_factor * (sum(mag_factors) / 3.0)

    return qualities


def _get_model_based_asterism_quality(
    asterisms: Table,
    runtime: LegacyBuildRuntime,
) -> np.ndarray | None:
    legacy_python = _get_legacy_runtime_root(runtime.legacy_config_path) / ".conda" / "bin" / "python"
    if not legacy_python.exists() or len(asterisms) == 0:
        return None

    working = asterisms.copy(copy_data=True)
    rename_map = {
        "asterism_id": "id",
        "star1_source_id": "star1_id",
        "star1_ref_epoch": "star1_pmepoch",
        "star2_source_id": "star2_id",
        "star2_ref_epoch": "star2_pmepoch",
        "star3_source_id": "star3_id",
        "star3_ref_epoch": "star3_pmepoch",
        "radius_arcsec": "radius",
        "area_arcsec2": "area",
        "relative_area": "relarea",
        "separation_arcsec": "separation",
        "relative_separation": "relsep",
    }
    for source_name, legacy_name in rename_map.items():
        if source_name in working.colnames:
            working.rename_column(source_name, legacy_name)
    for column_name in ("star1_idx", "star2_idx", "star3_idx"):
        if column_name in working.colnames:
            working.remove_column(column_name)

    with tempfile.TemporaryDirectory(prefix="ao-sky-legacy-quality-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_filename = tmpdir_path / "asterisms.fits"
        output_filename = tmpdir_path / "qualities.npy"
        working.write(input_filename, format="fits", overwrite=True)

        code = """
import os
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table

survey_root = Path(sys.argv[1]).resolve()
config_filename = Path(sys.argv[2]).resolve()
input_filename = Path(sys.argv[3]).resolve()
output_filename = Path(sys.argv[4]).resolve()
ao_system_name = sys.argv[5]

sys.path.insert(0, str(survey_root))
import aomap.aomap as aomap

config = aomap.read_config(str(config_filename))
ao_system = aomap.get_ao_system(config, ao_system_name)
asterisms = Table.read(input_filename, format="fits")
qualities = aomap._get_asterism_quality(config, asterisms, ao_system)
np.save(output_filename, np.asarray(qualities, dtype=np.float64))
"""
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mpl"))
        result = subprocess.run(
            [
                str(legacy_python),
                "-c",
                code,
                str(_get_legacy_runtime_root(runtime.legacy_config_path)),
                str(runtime.legacy_config_path),
                str(input_filename),
                str(output_filename),
                runtime.ao_system.name,
            ],
            cwd=runtime.legacy_config_path.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output_filename.exists():
            return None
        return np.load(output_filename)


def _filter_bright_star_exclusion(
    asterisms: Table,
    stars: Table,
    runtime: LegacyBuildRuntime,
) -> Table:
    threshold = runtime.max_bright_star_mag
    if threshold is None or len(asterisms) == 0:
        return asterisms

    bright_stars = stars[stars["R"] < threshold]
    if len(bright_stars) == 0:
        return asterisms

    bright_star_coords = SkyCoord(
        ra=bright_stars["ra"],
        dec=bright_stars["dec"],
        unit=(u.degree, u.degree),
    )
    asterism_centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
    keep = np.asarray(
        [
            np.min(centre.separation(bright_star_coords))
            > 2.0 * runtime.ao_system.fov
            for centre in asterism_centres
        ],
        dtype=np.bool_,
    )
    return asterisms[keep]


def _filter_overlaps(asterisms: Table, runtime: LegacyBuildRuntime) -> Table:
    threshold = runtime.max_overlap
    if threshold is None or len(asterisms) == 0:
        return asterisms

    fov_level = _get_level_with_resolution(runtime.ao_system.fov)
    fov_radius = runtime.ao_system.fov.to(u.rad).value
    fov_1ngs_radius = runtime.ao_system.fov_1ngs.to(u.rad).value
    centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
    asterism_pixs = get_pixel_from_skycoord(fov_level, centres)
    keep = np.ones(len(asterisms), dtype=np.bool_)
    qualities = _get_asterism_quality(asterisms, runtime)

    for pix in np.unique(asterism_pixs):
        search_pixs = np.concatenate([[pix], get_pixel_neighbours(fov_level, int(pix))])
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

    return asterisms[keep]


def _filter_relative_constraints(asterisms: Table, runtime: LegacyBuildRuntime) -> Table:
    result = asterisms
    if runtime.ao_system.max_rel_sep > 0:
        keep = (result["relative_separation"] >= runtime.ao_system.min_rel_sep) & (
            result["relative_separation"] < runtime.ao_system.max_rel_sep
        )
        result = result[keep]
    if runtime.ao_system.max_rel_area > 0 and len(result) > 0:
        keep = (result["relative_area"] >= runtime.ao_system.min_rel_area) & (
            result["relative_area"] < runtime.ao_system.max_rel_area
        )
        result = result[keep]
    return result


def build_outer_pixel_asterisms(
    store: GaiaHealpixStore,
    runtime: LegacyBuildRuntime,
    outer_pix: int,
) -> tuple[Table, Table, Table]:
    """Return the live-build star, NGS, and final asterism tables."""

    stars, ngs = prepare_search_inputs(store, runtime, outer_pix)
    options = AsterismSearchOptions(
        min_stars=runtime.ao_system.min_wfs,
        max_stars=runtime.ao_system.max_wfs,
        min_separation_arcsec=runtime.ao_system.min_sep.to(u.arcsec).value,
        max_separation_arcsec=runtime.ao_system.max_sep.to(u.arcsec).value,
        max_single_star_radius_arcsec=runtime.ao_system.fov_1ngs.to(u.arcsec).value / 2.0,
    )

    asterisms = find_asterisms(ngs, options).copy(copy_data=True)
    asterisms = _filter_bright_star_exclusion(asterisms, stars, runtime)
    asterisms = _filter_overlaps(asterisms, runtime)
    asterisms = _filter_relative_constraints(asterisms, runtime)

    centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
    keep = get_pixel_from_skycoord(runtime.outer_level, centres) == int(outer_pix)
    asterisms = asterisms[keep]
    if len(asterisms) > 0:
        centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
        asterisms["pix"] = get_pixel_from_skycoord(runtime.inner_level, centres)
    else:
        asterisms["pix"] = np.array([], dtype=np.int64)
    return stars, ngs, asterisms
