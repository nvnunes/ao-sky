#!/usr/bin/env python3
"""Temporary outer-pixel legacy asterism comparison helper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


LEGACY_CONFIG_FILENAME = REPO_ROOT.parent / "survey_tools" / "aomap" / "config.yaml"
LEGACY_RUNTIME_ROOT = REPO_ROOT.parent / "survey_tools"
LEGACY_PYTHON = LEGACY_RUNTIME_ROOT / ".conda" / "bin" / "python"
GAIA_ROOT = Path("/Volumes/Data/Galaxy/aosky")
GAIA_RELEASE = "dr3"
DEFAULT_AO_SYSTEM = "GNAO"
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-10
SPECIAL_HOUR_PIXELS = {
    8960,
    8972,
    9023,
    9152,
    9200,
    9203,
    9215,
    11264,
    11312,
    11327,
    11468,
}

COMPARISON_COLUMNS = (
    "id",
    "ra",
    "dec",
    "num_stars",
    "star1_id",
    "star1_ra",
    "star1_dec",
    "star1_pmra",
    "star1_pmdec",
    "star1_pmepoch",
    "star1_mag",
    "star2_id",
    "star2_ra",
    "star2_dec",
    "star2_pmra",
    "star2_pmdec",
    "star2_pmepoch",
    "star2_mag",
    "star3_id",
    "star3_ra",
    "star3_dec",
    "star3_pmra",
    "star3_pmdec",
    "star3_pmepoch",
    "star3_mag",
    "radius",
    "area",
    "relarea",
    "separation",
    "relsep",
    "pix",
)

NEW_TO_LEGACY_COLUMNS = {
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

GaiaHealpixStore = None
GaiaStoreConfig = None
GAIA_SCHEMA_COLUMNS = None
AsterismSearchOptions = None
apply_proper_motion = None
find_asterisms = None
load_asterism_stars = None
get_pixel_area = None
get_pixel_from_skycoord = None
get_pixel_neighbours = None
get_pixel_resolution = None
get_pixel_skycoord = None


def _load_runtime() -> None:
    global GaiaHealpixStore, GaiaStoreConfig, GAIA_SCHEMA_COLUMNS
    global AsterismSearchOptions, find_asterisms, load_asterism_stars
    global apply_proper_motion
    global get_pixel_area, get_pixel_from_skycoord, get_pixel_neighbours
    global get_pixel_resolution, get_pixel_skycoord

    if GaiaHealpixStore is not None:
        return

    from ao_sky.asterisms import (
        AsterismSearchOptions as _AsterismSearchOptions,
        find_asterisms as _find_asterisms,
        load_asterism_stars as _load_asterism_stars,
    )
    from ao_sky.gaia import (
        GaiaHealpixStore as _GaiaHealpixStore,
        GaiaStoreConfig as _GaiaStoreConfig,
        GAIA_SCHEMA_COLUMNS as _GAIA_SCHEMA_COLUMNS,
        apply_proper_motion as _apply_proper_motion,
    )
    from ao_sky.spatial import (
        get_pixel_area as _get_pixel_area,
        get_pixel_from_skycoord as _get_pixel_from_skycoord,
        get_pixel_neighbours as _get_pixel_neighbours,
        get_pixel_resolution as _get_pixel_resolution,
        get_pixel_skycoord as _get_pixel_skycoord,
    )

    GaiaHealpixStore = _GaiaHealpixStore
    GaiaStoreConfig = _GaiaStoreConfig
    GAIA_SCHEMA_COLUMNS = _GAIA_SCHEMA_COLUMNS
    AsterismSearchOptions = _AsterismSearchOptions
    apply_proper_motion = _apply_proper_motion
    find_asterisms = _find_asterisms
    load_asterism_stars = _load_asterism_stars
    get_pixel_area = _get_pixel_area
    get_pixel_from_skycoord = _get_pixel_from_skycoord
    get_pixel_neighbours = _get_pixel_neighbours
    get_pixel_resolution = _get_pixel_resolution
    get_pixel_skycoord = _get_pixel_skycoord


@dataclass(frozen=True, slots=True)
class LegacyAOSystem:
    name: str
    band: str
    fov: u.Quantity
    fov_1ngs: u.Quantity
    min_wfs: int
    max_wfs: int
    min_mag: float
    nom_mag: float
    max_mag: float
    min_sep: u.Quantity
    max_sep: u.Quantity
    min_rel_sep: float
    max_rel_sep: float
    min_rel_area: float
    max_rel_area: float


@dataclass(frozen=True, slots=True)
class LegacyComparisonConfig:
    outer_level: int
    inner_level: int
    asterism_epoch: float | None
    asterisms_min_galactic_latitude: float
    asterisms_galactic_latitude_bypass_pixs: tuple[int, ...]
    asterisms_max_star_density: float | None
    asterisms_max_bright_star_mag: float | None
    asterisms_max_overlap: float | None
    legacy_config_filename: Path
    ao_system: LegacyAOSystem


def _get_level_with_resolution(target_resolution: u.Quantity) -> int:
    _load_runtime()
    target = target_resolution.to(u.arcmin)
    for level in range(0, 30):
        if get_pixel_resolution(level).to(u.arcmin) <= target:
            return level
    raise ValueError(f"Could not find HEALPix level for resolution {target_resolution}")


def _get_hour_deg_for_path(outer_pix: int, coord: SkyCoord) -> tuple[int, int]:
    if outer_pix in SPECIAL_HOUR_PIXELS:
        return 0, 50
    hour = int(coord.ra.degree / 15.0)
    deg = int(np.abs(coord.dec.degree) / 10.0) * 10
    return hour, deg


def load_live_legacy_asterisms(
    *,
    config: LegacyComparisonConfig,
    outer_pix: int,
) -> Table:
    """Run the live legacy outer-pixel asterism path and return its table output."""
    if not LEGACY_PYTHON.exists():
        raise FileNotFoundError(f"Legacy Python runtime not found: {LEGACY_PYTHON}")

    with tempfile.TemporaryDirectory(prefix="ao-sky-live-legacy-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_filename = tmpdir_path / "legacy-asterisms.fits"
        code = textwrap.dedent("""
import os
import sys
from pathlib import Path

survey_root = Path(sys.argv[1]).resolve()
config_filename = Path(sys.argv[2]).resolve()
output_filename = Path(sys.argv[3]).resolve()
outer_pix = int(sys.argv[4])
ao_system_name = sys.argv[5]

sys.path.insert(0, str(survey_root))
import aomap.aomap as aomap

config = aomap.read_config(str(config_filename))
asterisms = aomap.find_outer_asterisms(config, outer_pix, ao_system_name)
asterisms.write(output_filename, format="fits", overwrite=True)
""")
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(tmpdir_path / "mpl"))
        result = subprocess.run(
            [
                str(LEGACY_PYTHON),
                "-c",
                code,
                str(LEGACY_RUNTIME_ROOT),
                str(config.legacy_config_filename),
                str(output_filename),
                str(outer_pix),
                config.ao_system.name,
            ],
            cwd=LEGACY_RUNTIME_ROOT / "aomap",
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "no output"
            raise RuntimeError(f"Legacy live comparison failed: {details}")
        return Table.read(output_filename, format="fits")


def select_random_outer_pix(
    *,
    outer_level: int,
) -> int:
    """Select a random outer pixel id for the configured HEALPix level."""
    num_pixels = 12 * (4**outer_level)
    rng = np.random.default_rng()
    return int(rng.integers(0, num_pixels))


def _get_membership_key(table: Table, row_index: int) -> str:
    star_ids = [
        int(table["star1_id"][row_index]),
        int(table["star2_id"][row_index]),
        int(table["star3_id"][row_index]),
    ]
    valid_ids = sorted(star_id for star_id in star_ids if star_id >= 0)
    return "-".join(str(star_id) for star_id in valid_ids)


def _get_circle_overlap_area(radius1: float, radius2: float, separation: float) -> float:
    if separation >= radius1 + radius2:
        return 0.0
    if separation <= abs(radius1 - radius2):
        return np.pi * min(radius1, radius2) ** 2

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


def load_legacy_config(filename: Path, ao_system_name: str) -> LegacyComparisonConfig:
    with filename.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    systems = raw.get("ao_systems", [])
    system = next((item for item in systems if item["name"] == ao_system_name), None)
    if system is None:
        raise ValueError(f"AO system {ao_system_name!r} not found in {filename}")

    fov_1ngs = system.get("fov_1ngs", system["fov"])
    ao_system = LegacyAOSystem(
        name=system["name"],
        band=system["band"],
        fov=float(system["fov"]) * u.arcsec,
        fov_1ngs=float(fov_1ngs) * u.arcsec,
        min_wfs=int(system["min_wfs"]),
        max_wfs=int(system["max_wfs"]),
        min_mag=float(system["min_mag"]),
        nom_mag=float(system.get("nom_mag", system["max_mag"])),
        max_mag=float(system["max_mag"]),
        min_sep=float(system["min_sep"]) * u.arcsec,
        max_sep=float(system["max_sep"]) * u.arcsec,
        min_rel_sep=float(system.get("min_rel_sep", 0.0)),
        max_rel_sep=float(system.get("max_rel_sep", 0.0)),
        min_rel_area=float(system.get("min_rel_area", 0.0)),
        max_rel_area=float(system.get("max_rel_area", 0.0)),
    )

    bypass_pixs = tuple(int(pix) for pix in raw.get("asterisms_galactic_latitude_bypass_pixs", []))
    return LegacyComparisonConfig(
        outer_level=int(raw["outer_level"]),
        inner_level=int(raw["inner_level"]),
        asterism_epoch=(
            None if raw.get("asterism_epoch") is None else float(raw["asterism_epoch"])
        ),
        asterisms_min_galactic_latitude=float(raw.get("asterisms_min_galactic_latitude", 20.0)),
        asterisms_galactic_latitude_bypass_pixs=bypass_pixs,
        asterisms_max_star_density=(
            None
            if raw.get("asterisms_max_star_density") is None
            else float(raw["asterisms_max_star_density"])
        ),
        asterisms_max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        asterisms_max_overlap=(
            None
            if raw.get("asterisms_max_overlap") is None
            else float(raw["asterisms_max_overlap"])
        ),
        legacy_config_filename=filename.resolve(),
        ao_system=ao_system,
    )


def _filter_neighbours_by_galactic_latitude(
    stars: Table,
    *,
    outer_level: int,
    outer_pix: int,
    neighbour_level: int | None,
    min_galactic_latitude: float,
    bypass_pixs: tuple[int, ...],
) -> Table:
    _load_runtime()
    if outer_pix in bypass_pixs or "source_outer_pix" not in stars.colnames or neighbour_level is None:
        return stars

    keep = np.ones(len(stars), dtype=np.bool_)
    unique_source_pixels = np.unique(np.asarray(stars["source_outer_pix"], dtype=np.int64))
    for source_outer_pix in unique_source_pixels:
        if source_outer_pix == outer_pix:
            continue
        source_coord = get_pixel_skycoord(neighbour_level, int(source_outer_pix))
        if np.abs(source_coord.galactic.b.degree) < min_galactic_latitude:
            keep &= np.asarray(stars["source_outer_pix"], dtype=np.int64) != int(source_outer_pix)
    return stars[keep]


def _prepare_new_star_table(
    store: object,
    config: LegacyComparisonConfig,
    outer_pix: int,
) -> tuple[Table, Table]:
    _load_runtime()
    fov_level = _get_level_with_resolution(config.ao_system.fov)
    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=fov_level,
        include_locality=True,
    )
    stars = _filter_neighbours_by_galactic_latitude(
        stars,
        outer_level=config.outer_level,
        outer_pix=outer_pix,
        neighbour_level=fov_level,
        min_galactic_latitude=config.asterisms_min_galactic_latitude,
        bypass_pixs=config.asterisms_galactic_latitude_bypass_pixs,
    )
    stars["pix"] = get_pixel_from_skycoord(
        fov_level,
        SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree)),
    )

    if config.asterism_epoch is not None:
        locality_columns = {
            "is_local": np.asarray(stars["is_local"]).copy(),
            "source_outer_pix": np.asarray(stars["source_outer_pix"]).copy(),
            "pix": np.asarray(stars["pix"]).copy(),
            "R": np.asarray(stars["R"]).copy(),
        }
        shifted = apply_proper_motion(stars[list(GAIA_SCHEMA_COLUMNS)], epoch=config.asterism_epoch)
        for name, values in locality_columns.items():
            shifted[name] = values
        stars = shifted

    stars = stars[~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"]) & ~np.isnan(stars["R"])]
    filtered = stars[(stars["R"] >= config.ao_system.min_mag) & (stars["R"] < config.ao_system.max_mag)]

    if config.asterisms_max_star_density is not None and len(stars) > 0:
        unique_pixs, star_counts = np.unique(np.asarray(stars["pix"], dtype=np.int64), return_counts=True)
        fov_level_area = get_pixel_area(fov_level).to(u.arcmin**2).value
        remove_pixs = unique_pixs[star_counts / fov_level_area > config.asterisms_max_star_density]
        if len(remove_pixs) > 0:
            filtered = filtered[~np.isin(filtered["pix"], remove_pixs)]

    return stars, filtered


def _filter_bright_star_exclusion(asterisms: Table, stars: Table, config: LegacyComparisonConfig) -> Table:
    threshold = config.asterisms_max_bright_star_mag
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
        [np.min(centre.separation(bright_star_coords)) > 2.0 * config.ao_system.fov for centre in asterism_centres],
        dtype=np.bool_,
    )
    return asterisms[keep]


def _filter_overlaps(asterisms: Table, config: LegacyComparisonConfig) -> Table:
    _load_runtime()
    threshold = config.asterisms_max_overlap
    if threshold is None or len(asterisms) == 0:
        return asterisms

    fov_level = _get_level_with_resolution(config.ao_system.fov)
    fov_radius = config.ao_system.fov.to(u.rad).value
    fov_1ngs_radius = config.ao_system.fov_1ngs.to(u.rad).value
    centres = SkyCoord(ra=asterisms["ra"], dec=asterisms["dec"], unit=(u.degree, u.degree))
    asterism_pixs = get_pixel_from_skycoord(fov_level, centres)
    keep = np.ones(len(asterisms), dtype=np.bool_)
    qualities = _get_asterism_quality(asterisms, config)

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


def _get_asterism_quality(asterisms: Table, config: LegacyComparisonConfig) -> np.ndarray:
    model_qualities = _get_model_based_asterism_quality(asterisms, config)
    if model_qualities is not None:
        return model_qualities

    ao_system = config.ao_system
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
        rel_sep = float(asterism["relsep"]) if int(asterism["num_stars"]) > 1 else min_rel_sep
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

        mag_factor = sum(mag_factors) / 3.0
        qualities[index] = rel_factor * mag_factor

    return qualities


def _get_model_based_asterism_quality(
    asterisms: Table,
    config: LegacyComparisonConfig,
) -> np.ndarray | None:
    if not LEGACY_PYTHON.exists():
        return None
    if len(asterisms) == 0:
        return np.array([], dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="ao-sky-legacy-quality-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_filename = tmpdir_path / "asterisms.fits"
        output_filename = tmpdir_path / "qualities.npy"
        asterisms.write(input_filename, format="fits", overwrite=True)

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
                str(LEGACY_PYTHON),
                "-c",
                code,
                str(LEGACY_RUNTIME_ROOT),
                str(config.legacy_config_filename),
                str(input_filename),
                str(output_filename),
                config.ao_system.name,
            ],
            cwd=LEGACY_RUNTIME_ROOT / "aomap",
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output_filename.exists():
            return None
        return np.load(output_filename)


def _filter_relative_constraints(asterisms: Table, config: LegacyComparisonConfig) -> Table:
    result = asterisms
    if config.ao_system.max_rel_sep > 0:
        keep = (result["relsep"] >= config.ao_system.min_rel_sep) & (
            result["relsep"] < config.ao_system.max_rel_sep
        )
        result = result[keep]
    if config.ao_system.max_rel_area > 0 and len(result) > 0:
        keep = (result["relarea"] >= config.ao_system.min_rel_area) & (
            result["relarea"] < config.ao_system.max_rel_area
        )
        result = result[keep]
    return result


def _prepare_new_asterisms(
    stars: Table,
    ngs: Table,
    config: LegacyComparisonConfig,
    outer_pix: int,
) -> Table:
    _load_runtime()
    options = AsterismSearchOptions(
        min_stars=config.ao_system.min_wfs,
        max_stars=config.ao_system.max_wfs,
        min_separation_arcsec=config.ao_system.min_sep.to(u.arcsec).value,
        max_separation_arcsec=config.ao_system.max_sep.to(u.arcsec).value,
        max_single_star_radius_arcsec=config.ao_system.fov_1ngs.to(u.arcsec).value / 2.0,
    )
    asterisms = find_asterisms(ngs, options)
    working = asterisms.copy(copy_data=True)
    for source_name, legacy_name in NEW_TO_LEGACY_COLUMNS.items():
        if source_name in working.colnames:
            working.rename_column(source_name, legacy_name)
    for column_name in ("star1_idx", "star2_idx", "star3_idx"):
        if column_name in working.colnames:
            working.remove_column(column_name)

    working = _filter_bright_star_exclusion(working, stars, config)
    working = _filter_overlaps(working, config)
    working = _filter_relative_constraints(working, config)

    centres = SkyCoord(ra=working["ra"], dec=working["dec"], unit=(u.degree, u.degree))
    keep = get_pixel_from_skycoord(config.outer_level, centres) == int(outer_pix)
    working = working[keep]
    if len(working) == 0:
        working["pix"] = np.array([], dtype=np.int64)
        return working

    centres = SkyCoord(ra=working["ra"], dec=working["dec"], unit=(u.degree, u.degree))
    working["pix"] = get_pixel_from_skycoord(config.inner_level, centres)
    return working


def build_new_asterisms(
    *,
    gaia_root: Path,
    release: str,
    config: LegacyComparisonConfig,
    outer_pix: int,
) -> tuple[Table, Table, Table]:
    _load_runtime()
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=gaia_root, release=release, healpix_level=config.outer_level)
    )
    stars, ngs = _prepare_new_star_table(store, config, outer_pix)
    asterisms = _prepare_new_asterisms(stars, ngs, config, outer_pix)
    return stars, ngs, asterisms


def prepare_legacy_table(table: Table) -> Table:
    missing = [name for name in COMPARISON_COLUMNS if name not in table.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Legacy table is missing required comparison columns: {missing_text}")
    return table[list(COMPARISON_COLUMNS)].copy(copy_data=True)


def prepare_new_table(table: Table) -> Table:
    missing = [name for name in COMPARISON_COLUMNS if name not in table.colnames]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"New table is missing required comparison columns: {missing_text}")
    return table[list(COMPARISON_COLUMNS)].copy(copy_data=True)


def compare_tables(legacy: Table, new: Table) -> int:
    legacy_keys = np.asarray([_get_membership_key(legacy, index) for index in range(len(legacy))])
    new_keys = np.asarray([_get_membership_key(new, index) for index in range(len(new))])

    missing_from_new = sorted(set(legacy_keys) - set(new_keys))
    extra_in_new = sorted(set(new_keys) - set(legacy_keys))

    print(f"legacy rows: {len(legacy)}")
    print(f"new rows:    {len(new)}")
    print(f"shared keys: {len(set(legacy_keys) & set(new_keys))}")
    print(f"missing:     {len(missing_from_new)}")
    print(f"extra:       {len(extra_in_new)}")

    if missing_from_new:
        print("missing keys:", ", ".join(missing_from_new[:10]))
    if extra_in_new:
        print("extra keys:", ", ".join(extra_in_new[:10]))

    if len(legacy) == 0 and len(new) == 0:
        print("row ordering: exact membership order match")
        return 0

    common_keys = sorted(set(legacy_keys) & set(new_keys))
    if not common_keys:
        return 1

    legacy_index = {key: int(np.flatnonzero(legacy_keys == key)[0]) for key in common_keys}
    new_index = {key: int(np.flatnonzero(new_keys == key)[0]) for key in common_keys}

    mismatch_count = 0
    for column_name in COMPARISON_COLUMNS:
        if column_name == "id":
            continue

        column_mismatch = 0
        max_abs_diff = 0.0
        for key in common_keys:
            legacy_value = legacy[column_name][legacy_index[key]]
            new_value = new[column_name][new_index[key]]
            if np.issubdtype(np.asarray([legacy_value]).dtype, np.floating):
                if np.isnan(legacy_value) and np.isnan(new_value):
                    continue
                if np.isclose(legacy_value, new_value, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
                    max_abs_diff = max(max_abs_diff, float(abs(legacy_value - new_value)))
                    continue
                max_abs_diff = max(max_abs_diff, float(abs(legacy_value - new_value)))
            elif legacy_value == new_value:
                continue

            column_mismatch += 1

        if column_mismatch > 0:
            mismatch_count += column_mismatch
            print(
                f"column mismatch: {column_name} mismatches={column_mismatch} "
                f"max_abs_diff={max_abs_diff:.6g}"
            )

    legacy_ordered = np.asarray([legacy_index[key] for key in common_keys], dtype=np.int64) + 1
    new_ordered = np.asarray([new_index[key] for key in common_keys], dtype=np.int64) + 1
    if np.array_equal(legacy_ordered, new_ordered):
        print("row ordering: exact membership order match")
    else:
        print("row ordering: differs")

    return 1 if missing_from_new or extra_in_new or mismatch_count else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one outer-pixel live legacy asterism result against the new in-memory search path.",
    )
    parser.add_argument(
        "outer_pix",
        nargs="?",
        type=int,
        help="Outer HEALPix pixel id to compare. If omitted, choose one uniformly at random from the outer HEALPix level.",
    )
    parser.add_argument("--ao-system", default=DEFAULT_AO_SYSTEM, help="Legacy AO system name.")
    parser.add_argument(
        "--config",
        type=Path,
        default=LEGACY_CONFIG_FILENAME,
        help="Legacy aomap config.yaml to read comparison settings from.",
    )
    parser.add_argument(
        "--gaia-root",
        type=Path,
        default=GAIA_ROOT,
        help="Canonical ao-sky Gaia root used by the new code path.",
    )
    parser.add_argument(
        "--release",
        default=GAIA_RELEASE,
        help="Gaia release identifier for the canonical ao-sky store.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_runtime()
    config = load_legacy_config(args.config, args.ao_system)
    outer_pix = args.outer_pix
    if outer_pix is None:
        outer_pix = select_random_outer_pix(
            outer_level=config.outer_level,
        )
        print(f"selected random outer pixel: {outer_pix}")

    legacy_table = prepare_legacy_table(
        load_live_legacy_asterisms(config=config, outer_pix=outer_pix)
    )
    legacy_label = (
        "live legacy code: "
        f"{LEGACY_RUNTIME_ROOT / 'aomap'} using {config.legacy_config_filename}"
    )

    stars, ngs, new_table = build_new_asterisms(
        gaia_root=args.gaia_root,
        release=args.release,
        config=config,
        outer_pix=outer_pix,
    )
    prepared_new = prepare_new_table(new_table)

    print(legacy_label)
    print(f"new stars loaded: {len(stars)}")
    print(f"new NGS after policy filters: {len(ngs)}")
    return compare_tables(legacy_table, prepared_new)


if __name__ == "__main__":
    raise SystemExit(main())
