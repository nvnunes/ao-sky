"""In-memory asterism search with legacy-faithful execution shape."""

from __future__ import annotations

from dataclasses import dataclass
import time

from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
import astropy.units as u
import numpy as np

from ._constants import ASTERISM_TABLE_COLUMNS
from ._exceptions import AsterismError


# Public search options

@dataclass(frozen=True, slots=True)
class AsterismSearchOptions:
    """Search controls for in-memory asterism construction.

    Attributes:
        min_stars: Minimum supported star count in each asterism.
        max_stars: Maximum supported star count in each asterism.
        min_separation_arcsec: Minimum star-to-star separation for multi-star
            asterisms.
        max_separation_arcsec: Maximum star-to-star separation used by the
            legacy neighbour search.
        max_single_star_radius_arcsec: Radius used for one-star asterisms.

    Raises:
        AsterismError: If the requested search options are outside the Phase 2
            1-3 star search contract.
    """

    min_stars: int = 1
    max_stars: int = 1
    min_separation_arcsec: float = 0.0
    max_separation_arcsec: float = 60.0
    max_single_star_radius_arcsec: float = 30.0

    def __post_init__(self) -> None:
        min_stars = int(self.min_stars)
        max_stars = int(self.max_stars)
        min_separation = float(self.min_separation_arcsec)
        max_separation = float(self.max_separation_arcsec)
        max_single_star_radius = float(self.max_single_star_radius_arcsec)

        if min_stars < 1 or max_stars > 3 or min_stars > max_stars:
            raise AsterismError("Phase 2 asterism search supports only 1-3 stars")
        if min_separation < 0.0:
            raise AsterismError("min_separation_arcsec must be non-negative")
        if max_separation <= 0.0:
            raise AsterismError("max_separation_arcsec must be positive")
        if min_separation > max_separation:
            raise AsterismError("min_separation_arcsec cannot exceed max_separation_arcsec")
        if max_single_star_radius <= 0.0:
            raise AsterismError("max_single_star_radius_arcsec must be positive")

        object.__setattr__(self, "min_stars", min_stars)
        object.__setattr__(self, "max_stars", max_stars)
        object.__setattr__(self, "min_separation_arcsec", min_separation)
        object.__setattr__(self, "max_separation_arcsec", max_separation)
        object.__setattr__(self, "max_single_star_radius_arcsec", max_single_star_radius)


# Search buffer

class _AsterismBuffer:
    def __init__(self) -> None:
        self.N = 0
        self.max_size = 10000
        self.idx = -1
        self.asterism_id: list[np.ndarray] = []
        self.ra: list[np.ndarray] = []
        self.dec: list[np.ndarray] = []
        self.num_stars: list[np.ndarray] = []
        self.star1_idx: list[np.ndarray] = []
        self.star1_source_id: list[np.ndarray] = []
        self.star1_ra: list[np.ndarray] = []
        self.star1_dec: list[np.ndarray] = []
        self.star1_pmra: list[np.ndarray] = []
        self.star1_pmdec: list[np.ndarray] = []
        self.star1_ref_epoch: list[np.ndarray] = []
        self.star1_mag: list[np.ndarray] = []
        self.star2_idx: list[np.ndarray] = []
        self.star2_source_id: list[np.ndarray] = []
        self.star2_ra: list[np.ndarray] = []
        self.star2_dec: list[np.ndarray] = []
        self.star2_pmra: list[np.ndarray] = []
        self.star2_pmdec: list[np.ndarray] = []
        self.star2_ref_epoch: list[np.ndarray] = []
        self.star2_mag: list[np.ndarray] = []
        self.star3_idx: list[np.ndarray] = []
        self.star3_source_id: list[np.ndarray] = []
        self.star3_ra: list[np.ndarray] = []
        self.star3_dec: list[np.ndarray] = []
        self.star3_pmra: list[np.ndarray] = []
        self.star3_pmdec: list[np.ndarray] = []
        self.star3_ref_epoch: list[np.ndarray] = []
        self.star3_mag: list[np.ndarray] = []
        self.radius_arcsec: list[np.ndarray] = []
        self.area_arcsec2: list[np.ndarray] = []
        self.relative_area: list[np.ndarray] = []
        self.separation_arcsec: list[np.ndarray] = []
        self.relative_separation: list[np.ndarray] = []


def _increment_buffer(buffer: _AsterismBuffer) -> None:
    buffer.asterism_id.append(np.zeros((buffer.max_size), dtype=np.int64))
    buffer.ra.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.dec.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.num_stars.append(np.zeros((buffer.max_size), dtype=np.int64))
    for prefix in ("star1", "star2", "star3"):
        getattr(buffer, f"{prefix}_idx").append(np.zeros((buffer.max_size), dtype=np.int64))
        getattr(buffer, f"{prefix}_source_id").append(np.zeros((buffer.max_size), dtype=np.int64))
        getattr(buffer, f"{prefix}_ra").append(np.zeros((buffer.max_size), dtype=np.float64))
        getattr(buffer, f"{prefix}_dec").append(np.zeros((buffer.max_size), dtype=np.float64))
        getattr(buffer, f"{prefix}_pmra").append(np.zeros((buffer.max_size), dtype=np.float64))
        getattr(buffer, f"{prefix}_pmdec").append(np.zeros((buffer.max_size), dtype=np.float64))
        getattr(buffer, f"{prefix}_ref_epoch").append(np.zeros((buffer.max_size), dtype=np.float64))
        getattr(buffer, f"{prefix}_mag").append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.radius_arcsec.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.area_arcsec2.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.relative_area.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.separation_arcsec.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.relative_separation.append(np.zeros((buffer.max_size), dtype=np.float64))
    buffer.idx = -1


def _merge_buffer(buffer: _AsterismBuffer) -> None:
    for name in ASTERISM_TABLE_COLUMNS:
        values = getattr(buffer, name)
        if buffer.N == 0:
            setattr(buffer, name, np.array([], dtype=values[0].dtype if values else np.float64))
            continue
        merged = np.concatenate(values)
        setattr(buffer, name, merged[: buffer.N])


# Search validation

def _require_search_columns(stars: Table) -> Table:
    required = ("source_id", "ra", "dec", "ref_epoch", "pmra", "pmdec", "R")
    missing = [name for name in required if name not in stars.colnames]
    if missing:
        raise AsterismError(
            "find_asterisms requires columns: " + ", ".join(required)
        )
    return stars


def _sort_star_indexes(star1_idx: int, star2_idx: int | None = None, star3_idx: int | None = None) -> tuple[np.ndarray, str]:
    if star3_idx is not None:
        star_indexes = [star1_idx, star2_idx, star3_idx]
    elif star2_idx is not None:
        star_indexes = [star1_idx, star2_idx]
    else:
        star_indexes = [star1_idx]

    indexes = np.sort(np.asarray(star_indexes, dtype=np.int64))
    key = "-".join(str(int(index)) for index in indexes)
    return indexes, key


# Flat geometry helpers

def _process_flat_2points(ra1: float, dec1: float, ra2: float, dec2: float, flat_func) -> tuple[float, float]:
    ra0 = np.mean([ra1, ra2])
    dec0 = np.mean([dec1, dec2])

    ux, uy = flat_func(
        (ra1 - ra0) * np.cos(dec0),
        (dec1 - dec0),
        (ra2 - ra0) * np.cos(dec0),
        (dec2 - dec0),
    )
    return ux / np.cos(dec0) + ra0, uy + dec0


def _get_midpoint_flat(ax: float, ay: float, bx: float, by: float) -> tuple[float, float]:
    return (ax + bx) / 2.0, (ay + by) / 2.0


def get_midpoint(ra1: float, dec1: float, ra2: float, dec2: float) -> tuple[float, float]:
    """Return the midpoint between two tangent-plane points."""

    return _process_flat_2points(ra1, dec1, ra2, dec2, _get_midpoint_flat)


def _process_flat_3points(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra3: float,
    dec3: float,
    flat_func,
) -> tuple[float, float]:
    ra0 = np.mean([ra1, ra2, ra3])
    dec0 = np.mean([dec1, dec2, dec3])

    ux, uy = flat_func(
        (ra1 - ra0) * np.cos(dec0),
        (dec1 - dec0),
        (ra2 - ra0) * np.cos(dec0),
        (dec2 - dec0),
        (ra3 - ra0) * np.cos(dec0),
        (dec3 - dec0),
    )
    return ux / np.cos(dec0) + ra0, uy + dec0


def _get_triangle_incenter_flat(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> tuple[float, float]:
    d1 = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    d2 = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    d3 = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
    p = d1 + d2 + d3
    return (d1 * ax + d2 * bx + d3 * cx) / p, (d1 * ay + d2 * by + d3 * cy) / p


def get_triangle_incenter(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra3: float,
    dec3: float,
) -> tuple[float, float]:
    """Return the incenter of three tangent-plane triangle vertices."""

    return _process_flat_3points(
        ra1,
        dec1,
        ra2,
        dec2,
        ra3,
        dec3,
        _get_triangle_incenter_flat,
    )


def get_triangle_angular_area(
    ra1: float,
    dec1: float,
    ra2: float,
    dec2: float,
    ra3: float,
    dec3: float,
    centre_ra: float,
    centre_dec: float,
) -> float:
    """Return the tangent-plane triangle area in radians squared."""

    dra1 = (ra1 - centre_ra) * np.cos(centre_dec)
    ddec1 = dec1 - centre_dec
    dra2 = (ra2 - centre_ra) * np.cos(centre_dec)
    ddec2 = dec2 - centre_dec
    dra3 = (ra3 - centre_ra) * np.cos(centre_dec)
    ddec3 = dec3 - centre_dec
    return 0.5 * np.abs(dra1 * (ddec2 - ddec3) + dra2 * (ddec3 - ddec1) + dra3 * (ddec1 - ddec2))


# Row writing

def _set_star_slot(
    buffer: _AsterismBuffer,
    prefix: str,
    star_data: Table,
    row_idx: int,
    star_idx: int | None,
) -> None:
    if star_idx is None:
        getattr(buffer, f"{prefix}_idx")[-1][row_idx] = -1
        getattr(buffer, f"{prefix}_source_id")[-1][row_idx] = -1
        getattr(buffer, f"{prefix}_ra")[-1][row_idx] = -1.0
        getattr(buffer, f"{prefix}_dec")[-1][row_idx] = -1.0
        getattr(buffer, f"{prefix}_pmra")[-1][row_idx] = -1.0
        getattr(buffer, f"{prefix}_pmdec")[-1][row_idx] = -1.0
        getattr(buffer, f"{prefix}_ref_epoch")[-1][row_idx] = -1.0
        getattr(buffer, f"{prefix}_mag")[-1][row_idx] = -1.0
        return

    getattr(buffer, f"{prefix}_idx")[-1][row_idx] = star_idx
    getattr(buffer, f"{prefix}_source_id")[-1][row_idx] = int(star_data["source_id"][star_idx])
    getattr(buffer, f"{prefix}_ra")[-1][row_idx] = float(star_data["ra"][star_idx])
    getattr(buffer, f"{prefix}_dec")[-1][row_idx] = float(star_data["dec"][star_idx])
    getattr(buffer, f"{prefix}_pmra")[-1][row_idx] = float(star_data["pmra"][star_idx])
    getattr(buffer, f"{prefix}_pmdec")[-1][row_idx] = float(star_data["pmdec"][star_idx])
    getattr(buffer, f"{prefix}_ref_epoch")[-1][row_idx] = float(star_data["ref_epoch"][star_idx])
    getattr(buffer, f"{prefix}_mag")[-1][row_idx] = float(star_data["R"][star_idx])


def _add_asterism(
    buffer: _AsterismBuffer,
    star_indexes: np.ndarray,
    star_data: Table,
    options: AsterismSearchOptions,
) -> None:
    optimal_star_area = (
        3.0 / 4.0 * np.sqrt(3.0) * np.power(options.max_separation_arcsec / 4.0, 2)
    )

    num_stars = len(star_indexes)
    if num_stars == 0:
        raise IndexError("Minimum of 1 star required")
    if num_stars > 3:
        raise IndexError("Maximum of 3 stars supported")

    idx1 = int(star_indexes[0])
    ra1 = float(star_data["ra"][idx1])
    dec1 = float(star_data["dec"][idx1])
    m1 = float(star_data["R"][idx1])
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)

    if num_stars >= 2:
        idx2 = int(star_indexes[1])
        ra2 = float(star_data["ra"][idx2])
        dec2 = float(star_data["dec"][idx2])
        m2 = float(star_data["R"][idx2])
        ra2_rad = np.deg2rad(ra2)
        dec2_rad = np.deg2rad(dec2)
        l1 = np.rad2deg(np.sqrt(((ra1_rad - ra2_rad) * np.cos(dec1_rad)) ** 2 + (dec1_rad - dec2_rad) ** 2)) * 3600.0
    else:
        idx2 = None

    if num_stars == 3:
        idx3 = int(star_indexes[2])
        ra3 = float(star_data["ra"][idx3])
        dec3 = float(star_data["dec"][idx3])
        m3 = float(star_data["R"][idx3])
        ra3_rad = np.deg2rad(ra3)
        dec3_rad = np.deg2rad(dec3)
        l2 = np.rad2deg(np.sqrt(((ra2_rad - ra3_rad) * np.cos(dec2_rad)) ** 2 + (dec2_rad - dec3_rad) ** 2)) * 3600.0
        l3 = np.rad2deg(np.sqrt(((ra3_rad - ra1_rad) * np.cos(dec3_rad)) ** 2 + (dec3_rad - dec1_rad) ** 2)) * 3600.0
    else:
        idx3 = None

    if num_stars == 1:
        centre_ra = ra1_rad
        centre_dec = dec1_rad
        separation = options.max_single_star_radius_arcsec * 2.0
        radius = options.max_single_star_radius_arcsec
        area = 0.0
    elif num_stars == 2:
        centre_ra = np.mean([ra1_rad, ra2_rad])
        centre_dec = np.mean([dec1_rad, dec2_rad])
        separation = l1
        radius = l1 / 2.0
        area = 0.0
    else:
        centre_ra, centre_dec = get_triangle_incenter(
            ra1_rad,
            dec1_rad,
            ra2_rad,
            dec2_rad,
            ra3_rad,
            dec3_rad,
        )

        d1 = np.sqrt(((ra1_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec1_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0
        d2 = np.sqrt(((ra2_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec2_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0
        d3 = np.sqrt(((ra3_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec3_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0

        if np.max([d1, d2, d3]) > options.max_separation_arcsec / 2.0:
            longest_length = np.max([l1, l2, l3])
            if l1 == longest_length:
                centre_ra, centre_dec = get_midpoint(ra1_rad, dec1_rad, ra2_rad, dec2_rad)
            elif l2 == longest_length:
                centre_ra, centre_dec = get_midpoint(ra2_rad, dec2_rad, ra3_rad, dec3_rad)
            else:
                centre_ra, centre_dec = get_midpoint(ra3_rad, dec3_rad, ra1_rad, dec1_rad)

            d1 = np.sqrt(((ra1_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec1_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0
            d2 = np.sqrt(((ra2_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec2_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0
            d3 = np.sqrt(((ra3_rad - centre_ra) * np.cos(centre_dec)) ** 2 + (dec3_rad - centre_dec) ** 2) * 180.0 / np.pi * 3600.0
            if np.max([d1, d2, d3]) > options.max_separation_arcsec / 2.0:
                return

        mean_mag = np.mean([m1, m2, m3])
        max_diff = np.max(np.abs([m1, m2, m3] - mean_mag))
        num_neg_diff = np.sum([m1, m2, m3] - mean_mag < 0)

        if max_diff < 1.0:
            separation = max(l1, l2, l3)
        elif num_neg_diff == 2:
            if m3 > m1 and m3 > m2:
                separation = l1
            elif m1 > m2 and m1 > m3:
                separation = l2
            else:
                separation = l3
        else:
            if m1 < m2 and m1 < m3:
                separation = max(l1, l3)
            elif m2 < m1 and m2 < m3:
                separation = max(l1, l2)
            else:
                separation = max(l2, l3)

        area = (
            get_triangle_angular_area(
                ra1_rad,
                dec1_rad,
                ra2_rad,
                dec2_rad,
                ra3_rad,
                dec3_rad,
                centre_ra,
                centre_dec,
            )
            * (180.0 / np.pi * 3600.0) ** 2
        )
        radius = area / ((l1 + l2 + l3) / 2.0)

    relative_separation = separation / options.max_separation_arcsec
    relative_area = area / optimal_star_area

    if buffer.idx == -1 or (buffer.idx + 1) >= buffer.max_size:
        _increment_buffer(buffer)

    buffer.N += 1
    buffer.idx += 1
    row_idx = buffer.idx
    buffer.asterism_id[-1][row_idx] = buffer.N
    buffer.ra[-1][row_idx] = np.rad2deg(centre_ra)
    buffer.dec[-1][row_idx] = np.rad2deg(centre_dec)
    buffer.num_stars[-1][row_idx] = num_stars
    _set_star_slot(buffer, "star1", star_data, row_idx, idx1)
    _set_star_slot(buffer, "star2", star_data, row_idx, idx2)
    _set_star_slot(buffer, "star3", star_data, row_idx, idx3)
    buffer.radius_arcsec[-1][row_idx] = radius
    buffer.area_arcsec2[-1][row_idx] = area
    buffer.relative_area[-1][row_idx] = relative_area
    buffer.separation_arcsec[-1][row_idx] = separation
    buffer.relative_separation[-1][row_idx] = relative_separation


# Public search

def find_asterisms(
    stars: Table,
    options: AsterismSearchOptions,
    *,
    verbose: bool = False,
) -> Table:
    """Return all asterisms found in a prepared Gaia star table.

    The implementation deliberately follows the legacy `survey_tools`
    search/deduping strategy before later optimization work. Search brightness
    is fixed to the legacy empirical Gaia-derived ``R`` magnitude.

    Args:
        stars: Prepared Gaia rows to search.
        options: Search controls for star count and separation policy.
        verbose: When true, emit the legacy-style progress logging for very
            large searches.

    Returns:
        Flat `Table` with deterministic sequential `asterism_id` values and
        fixed `star1_*`/`star2_*`/`star3_*` member slots.

    Raises:
        AsterismError: If the input table does not expose the required search
            fields.
    """

    _require_search_columns(stars)
    star_catalog = SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree))
    star_idx1s, star_idx2s, seps, _ = search_around_sky(
        star_catalog,
        star_catalog,
        options.max_separation_arcsec * u.arcsec,
    )
    sorting_indexes = np.argsort(star_idx1s)
    star_idx1s = star_idx1s[sorting_indexes]
    star_idx2s = star_idx2s[sorting_indexes]
    seps = seps[sorting_indexes]

    buffer = _AsterismBuffer()
    added_keys: dict[str, bool] = {}

    N = len(star_idx1s)
    start_time = time.time()
    close_idxs: list[int] = []
    close_seps: list[float] = []

    for i in np.arange(N):
        if verbose and N > 10000 and (i + 1) % 10000 == 0:
            print(f"{i + 1}/{N}: {time.time() - start_time:.2f}s", flush=True)
            start_time = time.time()

        if options.min_stars <= 1 and star_idx1s[i] == star_idx2s[i]:
            asterism_star_indexes, _ = _sort_star_indexes(int(star_idx1s[i]))
            _add_asterism(buffer, asterism_star_indexes, stars, options)

        if options.max_stars == 1:
            continue

        if seps[i] > options.min_separation_arcsec * u.arcsec:
            close_idxs.append(int(star_idx2s[i]))
            close_seps.append(float(seps[i].to(u.arcsec).value))

        if close_idxs and (i + 1 == N or star_idx1s[i + 1] != star_idx1s[i]):
            close_idx_array = np.asarray(close_idxs, dtype=np.int64)
            close_sep_array = np.asarray(close_seps, dtype=np.float64)
            sorted_close_idxs = close_idx_array[np.argsort(close_sep_array)]

            for j in np.arange(len(sorted_close_idxs)):
                asterism_star_indexes, asterism_key = _sort_star_indexes(
                    int(star_idx1s[i]),
                    int(sorted_close_idxs[j]),
                )
                if options.min_stars <= 2 and asterism_key not in added_keys:
                    _add_asterism(buffer, asterism_star_indexes, stars, options)
                    added_keys[asterism_key] = True

                if options.max_stars == 2:
                    continue

                for k in np.arange(j + 1, len(sorted_close_idxs)):
                    asterism_star_indexes, asterism_key = _sort_star_indexes(
                        int(star_idx1s[i]),
                        int(sorted_close_idxs[j]),
                        int(sorted_close_idxs[k]),
                    )
                    if asterism_key not in added_keys:
                        _add_asterism(buffer, asterism_star_indexes, stars, options)
                        added_keys[asterism_key] = True

            close_idxs = []
            close_seps = []

    if verbose and N > 10000:
        print(f"{N}/{N}: {time.time() - start_time:.2f}s", flush=True)

    _merge_buffer(buffer)
    return Table(
        [getattr(buffer, name) for name in ASTERISM_TABLE_COLUMNS],
        names=ASTERISM_TABLE_COLUMNS,
    )
