"""Local asterism diagnostic plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.coordinates import SkyCoord
from astropy.table import Table, unique, vstack
from astropy.table.row import Row
import astropy.units as u
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import griddata
import yaml

from ao_sky.build._constants import RUNTIME_CONFIG_FILENAME
from ao_sky.spatial import get_pixel_from_skycoord, get_pixel_neighbours, get_pixel_skycoord

from ._exceptions import PlottingError


@dataclass(frozen=True, slots=True)
class _AsterismPlotConfig:
    outer_level: int
    inner_level: int
    epoch: float
    release: str
    band: str
    fov: u.Quantity
    min_mag: float


def plot_asterism(
    asterism: Row | Table,
    *,
    fov: u.Quantity,
    stars: Table | None = None,
    center: SkyCoord | None = None,
    band: str = "R",
    min_mag: float = 8.0,
    ax: Axes | None = None,
    hide_stars: bool = False,
    hide_centered_fov: bool = False,
    hide_connections: bool = False,
    hide_valid_fov_centers: bool = False,
    hide_coverable_region: bool = False,
) -> Figure:
    """Plot one retained asterism as a local FoV geometry diagnostic.

    Args:
        asterism: One normalized retained-asterism row, or a one-row table,
            with `ra`, `dec`, `num_stars`, and `starN_*` member columns.
        fov: Asterism field-of-view diameter.
        stars: Optional normalized Gaia-star table to draw behind the asterism
            member stars.
        center: Optional scalar sky coordinate used as the plot origin. When
            omitted, the retained asterism `ra`/`dec` center is used.
        band: Magnitude column suffix used for member-star marker sizing.
            Retained asterism rows currently store member magnitudes in
            `starN_mag`, so this is reserved for API symmetry with
            `plot_asterisms`.
        min_mag: Bright-star threshold for member-star highlighting.
        ax: Optional Matplotlib axes to draw into. When omitted, a new figure
            and axes are created.
        hide_stars: Do not draw background stars or member-star markers.
        hide_centered_fov: Do not draw the retained asterism-centered FoV.
        hide_connections: Do not draw member-star connection lines.
        hide_valid_fov_centers: Do not draw the region where an FoV center can
            be placed while keeping all guide stars inside the FoV.
        hide_coverable_region: Do not draw the science-pixel region coverable
            by at least one valid FoV center.

    Returns:
        Matplotlib figure containing the single-asterism diagnostic plot.

    Raises:
        PlottingError: If the plot inputs are invalid.
    """

    table = _as_single_asterism_table(asterism)
    fov_radius_arcsec = _as_fov_radius_arcsec(fov)
    if center is None:
        row = table[0]
        center = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg, frame="icrs")
    center = _as_scalar_icrs(center)
    if stars is not None and not isinstance(stars, Table):
        raise PlottingError("stars must be an astropy.table.Table")
    extent_arcsec = _single_asterism_extent_arcsec(
        table[0],
        center=center,
        fov_radius_arcsec=fov_radius_arcsec,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
    else:
        fig = ax.figure

    if not hide_coverable_region or not hide_valid_fov_centers:
        valid_centers, coverable_region = _asterism_feasible_geometries(
            table[0],
            center=center,
            fov_radius_arcsec=fov_radius_arcsec,
        )
        if not hide_coverable_region:
            _draw_geometry(
                ax,
                coverable_region,
                facecolor="#2a9d8f",
                edgecolor="#087f73",
                alpha=0.14,
                linewidth=1.8,
                zorder=0,
            )
        if not hide_valid_fov_centers:
            _draw_geometry(
                ax,
                valid_centers,
                facecolor="#6c757d",
                edgecolor="#495057",
                alpha=0.18,
                linewidth=1.2,
                zorder=1,
            )

    if not hide_centered_fov:
        _draw_single_centered_fov(ax, table[0], center=center, fov_radius_arcsec=fov_radius_arcsec)
    if not hide_connections:
        _draw_connections_arcsec(ax, table, center)
    if not hide_stars:
        if stars is not None:
            _draw_background_stars_arcsec(
                ax,
                stars,
                center,
                extent_arcsec=extent_arcsec,
                band=band,
                min_mag=min_mag,
            )
        _draw_member_stars(ax, table[0], center=center, min_mag=min_mag, band=band)

    _style_single_asterism_axis(ax, extent_arcsec=extent_arcsec)
    return fig


def plot_asterisms(
    asterisms: Table,
    *,
    center: SkyCoord,
    width: u.Quantity,
    stars: Table | None = None,
    fov: u.Quantity | None = None,
    band: str = "R",
    min_mag: float = 8.0,
    ax: Axes | None = None,
    hide_stars: bool = False,
    hide_fov: bool = False,
    hide_connections: bool = False,
    coverable_region_mask: bool = False,
    coverable_region_asterisms: Table | None = None,
) -> Figure:
    """Plot a normalized retained-asterism table in a local square field.

    Args:
        asterisms: Normalized retained-asterism table with `ra`, `dec`,
            `num_stars`, and `starN_*` member columns.
        center: Scalar sky coordinate at the centre of the plotted field. The
            coordinate is interpreted in ICRS after any frame transform.
        width: Angular side length of the square field.
        stars: Optional normalized Gaia-star table to draw behind the retained
            asterisms. Required unless `hide_stars` is true.
        fov: Asterism field-of-view diameter. Required unless `hide_fov` is
            true.
        band: Magnitude column used for star marker sizing and bright-star
            highlighting.
        min_mag: Bright-star threshold in `band`.
        ax: Optional Matplotlib axes to draw into. When omitted, a new figure
            and axes are created.
        hide_stars: Do not draw Gaia stars.
        hide_fov: Do not draw asterism field-of-view circles.
        hide_connections: Do not draw member-star connection lines.
        coverable_region_mask: Draw a semi-transparent union mask of the
            science-pixel region coverable by at least one valid FoV center for
            each retained asterism.
        coverable_region_asterisms: Optional expanded retained-asterism table
            to use for the coverable-region mask. This lets build-backed
            callers include asterisms whose centers are just outside the
            plotted field but whose coverable regions enter it.

    Returns:
        Matplotlib figure containing the asterism diagnostic plot.

    Raises:
        PlottingError: If the plot inputs are invalid.
    """

    center = _as_scalar_icrs(center)
    width_deg = _as_width_deg(width)
    if not isinstance(asterisms, Table):
        raise PlottingError("asterisms must be an astropy.table.Table")
    if coverable_region_asterisms is not None and not isinstance(
        coverable_region_asterisms,
        Table,
    ):
        raise PlottingError("coverable_region_asterisms must be an astropy.table.Table")
    if not hide_stars and stars is None:
        raise PlottingError("stars are required unless hide_stars=True")
    if not hide_stars and not isinstance(stars, Table):
        raise PlottingError("stars must be an astropy.table.Table")
    if (not hide_fov or coverable_region_mask) and fov is None:
        raise PlottingError("fov is required unless hide_fov=True and coverable_region_mask=False")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
    else:
        fig = ax.figure

    half_width = width_deg / 2.0
    if coverable_region_mask:
        fov_radius_deg = float(fov.to_value(u.deg)) / 2.0
        mask_asterisms = (
            coverable_region_asterisms
            if coverable_region_asterisms is not None
            else asterisms
        )
        _draw_coverable_region_mask(
            ax,
            mask_asterisms,
            center=center,
            fov_radius_deg=fov_radius_deg,
            half_width_deg=half_width,
        )
    if not hide_fov:
        fov_radius_deg = float(fov.to_value(u.deg)) / 2.0
        _draw_fov_circles(ax, asterisms, center, fov_radius_deg)
    if not hide_connections:
        _draw_connections(ax, asterisms, center)
    if not hide_stars:
        _draw_stars(ax, stars, center, band=band, min_mag=min_mag)

    _style_axis(ax, half_width)
    return fig


def plot_build_asterisms(
    build_path: Path | str,
    *,
    gaia_root: Path | str | None = None,
    center: SkyCoord,
    width: u.Quantity,
    ax: Axes | None = None,
    hide_stars: bool = False,
    hide_fov: bool = False,
    hide_connections: bool = False,
    coverable_region_mask: bool = False,
    max_asterisms: int | None = None,
) -> Figure:
    """Plot retained asterisms in a square field from one build root.

    Args:
        build_path: Persisted `ao-sky` build root.
        gaia_root: Canonical Gaia store root. Required unless `hide_stars` is
            true. Stars are loaded from
            `<gaia_root>/gaia-<release>-hpx<outer_level>`.
        center: Scalar sky coordinate at the centre of the plotted field. The
            coordinate is interpreted in ICRS after any frame transform.
        width: Angular side length of the square field.
        ax: Optional Matplotlib axes to draw into. When omitted, a new figure
            and axes are created.
        hide_stars: Do not draw Gaia stars.
        hide_fov: Do not draw asterism field-of-view circles.
        hide_connections: Do not draw member-star connection lines.
        coverable_region_mask: Draw the coverable science-region mask.
        max_asterisms: Optional cap on plotted asterism rows after field
            filtering.

    Returns:
        Matplotlib figure containing the asterism diagnostic plot.

    Raises:
        PlottingError: If the build plotting metadata or location inputs are
            invalid.
    """

    center = _as_scalar_icrs(center)
    _as_width_deg(width)
    if max_asterisms is not None and int(max_asterisms) < 0:
        raise PlottingError("max_asterisms must be non-negative")

    if not hide_stars and gaia_root is None:
        raise PlottingError("gaia_root is required unless hide_stars=True")
    build_root = Path(build_path)
    gaia_root = None if gaia_root is None else Path(gaia_root)
    config = _load_plot_config(build_root)
    asterisms = read_build_asterisms(
        build_root,
        center=center,
        width=width,
        max_asterisms=max_asterisms,
    )
    coverable_asterisms = (
        read_build_asterisms(
            build_root,
            center=center,
            width=width,
            margin=config.fov,
            max_asterisms=max_asterisms,
        )
        if coverable_region_mask
        else None
    )
    stars = (
        Table()
        if hide_stars
        else read_asterism_stars(
            gaia_root=gaia_root,
            release=config.release,
            outer_level=config.outer_level,
            epoch=config.epoch,
            band=config.band,
            center=center,
            width=width,
        )
    )

    return plot_asterisms(
        asterisms,
        center=center,
        width=width,
        stars=stars,
        fov=config.fov,
        band=config.band,
        min_mag=config.min_mag,
        ax=ax,
        hide_stars=hide_stars,
        hide_fov=hide_fov,
        hide_connections=hide_connections,
        coverable_region_mask=coverable_region_mask,
        coverable_region_asterisms=coverable_asterisms,
    )


def plot_winner_ee(
    inner_pixels: Table,
    *,
    center: SkyCoord,
    width: u.Quantity,
    ee_kind: str = "resolved",
    ax: Axes | None = None,
    cmap: str = "plasma",
    vmin: float = 0.0,
    vmax: float = 0.6,
    grid_resolution: int = 300,
    add_colorbar: bool = True,
) -> Figure:
    """Plot a smoothed local winner-EE field from normalized inner pixels.

    Args:
        inner_pixels: Table with `ra`, `dec`, `winner_ee_resolved`, and
            `winner_ee_averaged` columns. Values are interpreted as inner-pixel
            centers and the winner performance assigned to each center.
        center: Scalar sky coordinate at the centre of the plotted field. The
            coordinate is interpreted in ICRS after any frame transform.
        width: Angular side length of the square field.
        ee_kind: Winner EE field to plot. Accepted values are `resolved`,
            `averaged`, `winner_ee_resolved`, and `winner_ee_averaged`.
        ax: Optional Matplotlib axes to draw into. When omitted, a new figure
            and axes are created.
        cmap: Matplotlib colormap name.
        vmin: Lower plotted EE value.
        vmax: Upper plotted EE value.
        grid_resolution: Number of interpolation samples along each plot axis.
        add_colorbar: Add an EE colorbar to the figure.

    Returns:
        Matplotlib figure containing the winner-EE field plot.

    Raises:
        PlottingError: If the plot inputs are invalid.
    """

    center = _as_scalar_icrs(center)
    width_deg = _as_width_deg(width)
    field = _winner_ee_field_name(ee_kind)
    if not isinstance(inner_pixels, Table):
        raise PlottingError("inner_pixels must be an astropy.table.Table")
    _require_winner_ee_columns(inner_pixels, field)
    grid_resolution = int(grid_resolution)
    if grid_resolution < 2:
        raise PlottingError("grid_resolution must be at least 2")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
    else:
        fig = ax.figure

    half_width = width_deg / 2.0
    image = _draw_winner_ee_field(
        ax,
        inner_pixels,
        center=center,
        half_width_deg=half_width,
        field=field,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        grid_resolution=grid_resolution,
    )
    if add_colorbar and image is not None:
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(r"Winner EE [100 mas]")

    _style_axis(ax, half_width)
    return fig


def plot_build_winner_ee(
    build_path: Path | str,
    *,
    center: SkyCoord,
    width: u.Quantity,
    ee_kind: str = "resolved",
    ax: Axes | None = None,
    cmap: str = "plasma",
    vmin: float = 0.0,
    vmax: float = 0.6,
    grid_resolution: int = 300,
    add_colorbar: bool = True,
) -> Figure:
    """Plot a local winner-EE field from one build root.

    Args:
        build_path: Persisted `ao-sky` build root.
        center: Scalar sky coordinate at the centre of the plotted field. The
            coordinate is interpreted in ICRS after any frame transform.
        width: Angular side length of the square field.
        ee_kind: Winner EE field to plot. Accepted values are `resolved`,
            `averaged`, `winner_ee_resolved`, and `winner_ee_averaged`.
        ax: Optional Matplotlib axes to draw into. When omitted, a new figure
            and axes are created.
        cmap: Matplotlib colormap name.
        vmin: Lower plotted EE value.
        vmax: Upper plotted EE value.
        grid_resolution: Number of interpolation samples along each plot axis.
        add_colorbar: Add an EE colorbar to the figure.

    Returns:
        Matplotlib figure containing the winner-EE field plot.

    Raises:
        PlottingError: If the build plotting metadata or location inputs are
            invalid.
    """

    inner_pixels = read_build_winner_ee(build_path, center=center, width=width)
    return plot_winner_ee(
        inner_pixels,
        center=center,
        width=width,
        ee_kind=ee_kind,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        grid_resolution=grid_resolution,
        add_colorbar=add_colorbar,
    )


def read_build_asterisms(
    build_path: Path | str,
    *,
    center: SkyCoord,
    width: u.Quantity,
    margin: u.Quantity | None = None,
    max_asterisms: int | None = None,
) -> Table:
    """Read normalized retained asterisms from one current build root.

    Args:
        build_path: Persisted `ao-sky` build root.
        center: Scalar sky coordinate at the centre of the selected field.
        width: Angular side length of the selected square field.
        margin: Optional angular margin added to all sides of the selected
            field before filtering asterism centers.
        max_asterisms: Optional cap on returned rows after field filtering.
    """

    center = _as_scalar_icrs(center)
    width_deg = _as_width_deg(width)
    margin_deg = 0.0 if margin is None else _as_margin_deg(margin)
    if max_asterisms is not None and int(max_asterisms) < 0:
        raise PlottingError("max_asterisms must be non-negative")

    build_root = Path(build_path)
    config = _load_plot_config(build_root)
    outer_pixs = _candidate_outer_pixels(config.outer_level, center)
    _require_center_outer_artifact(build_root, config=config, center_outer_pix=outer_pixs[0])
    return _load_field_asterisms(
        build_root,
        config=config,
        outer_pixs=outer_pixs,
        center=center,
        width_deg=width_deg + 2.0 * margin_deg,
        max_asterisms=max_asterisms,
    )


def read_build_winner_ee(
    build_path: Path | str,
    *,
    center: SkyCoord,
    width: u.Quantity,
) -> Table:
    """Read normalized winner-EE inner pixels from one current build root."""

    center = _as_scalar_icrs(center)
    width_deg = _as_width_deg(width)
    build_root = Path(build_path)
    config = _load_plot_config(build_root)
    outer_pixs = _candidate_outer_pixels(config.outer_level, center)
    _require_center_outer_artifact(build_root, config=config, center_outer_pix=outer_pixs[0])
    return _load_field_winner_ee(
        build_root,
        config=config,
        outer_pixs=outer_pixs,
        center=center,
        width_deg=width_deg,
    )


def read_asterism_stars(
    *,
    gaia_root: Path | str,
    release: str,
    outer_level: int,
    epoch: float,
    band: str,
    center: SkyCoord,
    width: u.Quantity,
) -> Table:
    """Read normalized Gaia stars for a local asterism diagnostic field."""

    center = _as_scalar_icrs(center)
    width_deg = _as_width_deg(width)
    return _load_field_stars(
        gaia_root=Path(gaia_root),
        release=release,
        outer_level=outer_level,
        epoch=epoch,
        band=band,
        outer_pixs=_candidate_outer_pixels(outer_level, center),
        center=center,
        width_deg=width_deg,
    )


def _load_plot_config(build_path: Path) -> _AsterismPlotConfig:
    config_path = build_path / RUNTIME_CONFIG_FILENAME
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise PlottingError(f"Build runtime config not found: {config_path}") from exc

    try:
        ao_system = _required_mapping(payload, "ao_system", config_path)
        traversal = _required_mapping(payload, "traversal", config_path)
        gaia = _required_mapping(payload, "gaia", config_path)
        return _AsterismPlotConfig(
            outer_level=_required_int(traversal, "outer_level", config_path),
            inner_level=_required_int(traversal, "inner_level", config_path),
            epoch=_required_float(gaia, "epoch", config_path),
            release=_optional_str(gaia, "release", default="dr3"),
            band=_required_str(ao_system, "band", config_path),
            fov=_required_float(ao_system, "fov_arcsec", config_path) * u.arcsec,
            min_mag=_required_float(ao_system, "min_mag", config_path),
        )
    except PlottingError:
        raise
    except (TypeError, ValueError) as exc:
        raise PlottingError(f"Invalid asterism plotting metadata in {config_path}") from exc


def _required_mapping(payload: dict[str, Any], key: str, filename: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PlottingError(f"{filename} is missing required section: {key}")
    return value


def _required_str(payload: dict[str, Any], key: str, filename: Path) -> str:
    if key not in payload:
        raise PlottingError(f"{filename} is missing required field: {key}")
    value = str(payload[key]).strip()
    if not value:
        raise PlottingError(f"{filename}.{key} must be non-empty")
    return value


def _optional_str(payload: dict[str, Any], key: str, *, default: str) -> str:
    value = str(payload.get(key, default)).strip()
    return value if value else default


def _required_int(payload: dict[str, Any], key: str, filename: Path) -> int:
    if key not in payload:
        raise PlottingError(f"{filename} is missing required field: {key}")
    return int(payload[key])


def _required_float(payload: dict[str, Any], key: str, filename: Path) -> float:
    if key not in payload:
        raise PlottingError(f"{filename} is missing required field: {key}")
    return float(payload[key])


def _as_scalar_icrs(center: SkyCoord) -> SkyCoord:
    if not isinstance(center, SkyCoord):
        raise PlottingError("center must be an astropy.coordinates.SkyCoord")
    if not center.isscalar:
        raise PlottingError("center must be a scalar SkyCoord")
    return center.icrs


def _as_width_deg(width: u.Quantity) -> float:
    if not isinstance(width, u.Quantity):
        raise PlottingError("width must be an astropy.units.Quantity")
    width_deg = float(width.to_value(u.deg))
    if not np.isfinite(width_deg) or width_deg <= 0.0:
        raise PlottingError("width must be a positive angular quantity")
    return width_deg


def _as_margin_deg(margin: u.Quantity) -> float:
    if not isinstance(margin, u.Quantity):
        raise PlottingError("margin must be an astropy.units.Quantity")
    margin_deg = float(margin.to_value(u.deg))
    if not np.isfinite(margin_deg) or margin_deg < 0.0:
        raise PlottingError("margin must be a non-negative angular quantity")
    return margin_deg


def _as_fov_radius_arcsec(fov: u.Quantity) -> float:
    if not isinstance(fov, u.Quantity):
        raise PlottingError("fov must be an astropy.units.Quantity")
    fov_arcsec = float(fov.to_value(u.arcsec))
    if not np.isfinite(fov_arcsec) or fov_arcsec <= 0.0:
        raise PlottingError("fov must be a positive angular quantity")
    return fov_arcsec / 2.0


def _as_single_asterism_table(asterism: Row | Table) -> Table:
    if isinstance(asterism, Row):
        table = Table(asterism)
    elif isinstance(asterism, Table):
        table = asterism
    else:
        raise PlottingError("asterism must be an astropy Row or one-row Table")
    if len(table) != 1:
        raise PlottingError("asterism must contain exactly one row")
    _require_asterism_columns(table)
    return table


def _require_asterism_columns(asterism: Table) -> None:
    required = {"ra", "dec", "num_stars"}
    missing = required - set(asterism.colnames)
    if missing:
        raise PlottingError(f"asterism is missing required columns: {', '.join(sorted(missing))}")
    num_stars = int(asterism["num_stars"][0])
    if num_stars < 1 or num_stars > 3:
        raise PlottingError("asterism num_stars must be between 1 and 3")
    for index in range(1, num_stars + 1):
        for suffix in ("ra", "dec"):
            column = f"star{index}_{suffix}"
            if column not in asterism.colnames:
                raise PlottingError(f"asterism is missing required column: {column}")


def _candidate_outer_pixels(outer_level: int, center: SkyCoord) -> tuple[int, ...]:
    center_pix = int(get_pixel_from_skycoord(int(outer_level), center))
    neighbours = get_pixel_neighbours(int(outer_level), center_pix)
    values = [center_pix]
    values.extend(int(pixel) for pixel in neighbours)
    return tuple(dict.fromkeys(values))


def _require_center_outer_artifact(
    build_path: Path,
    *,
    config: _AsterismPlotConfig,
    center_outer_pix: int,
) -> None:
    from ao_sky.artifacts import AoSkyArtifactStore

    store = AoSkyArtifactStore(build_path)
    filename = store.outer_path(
        center_outer_pix,
        outer_level=config.outer_level,
        inner_level=config.inner_level,
    )
    if not filename.is_file():
        raise PlottingError(
            "Center outer-pixel artifact is missing for asterism plot: "
            f"outer_pix={center_outer_pix}, path={filename}"
        )


def _load_field_stars(
    *,
    gaia_root: Path,
    release: str,
    outer_level: int,
    epoch: float,
    band: str,
    outer_pixs: tuple[int, ...],
    center: SkyCoord,
    width_deg: float,
) -> Table:
    tables = []
    for outer_pix in outer_pixs:
        stars = _load_outer_stars(
            gaia_root,
            outer_pix,
            release=release,
            outer_level=outer_level,
            epoch=epoch,
            band=band,
        )
        if stars is None:
            continue
        if len(stars) == 0:
            continue
        mask = _local_square_mask(stars["ra"], stars["dec"], center, width_deg / 2.0)
        if np.any(mask):
            tables.append(stars[mask])
    if not tables:
        return Table()
    return unique(vstack(tables), keys="source_id")


def _load_outer_stars(
    gaia_root: Path,
    outer_pix: int,
    *,
    release: str,
    outer_level: int,
    epoch: float,
    band: str,
) -> Table | None:
    from ao_sky.gaia.store import GaiaHealpixStore, GaiaStoreConfig
    from ao_sky.gaia.transform import apply_proper_motion, compute_r_magnitude

    store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=gaia_root,
            release=release,
            healpix_level=outer_level,
        )
    )
    filename = store.healpix_filename(outer_pix)
    if not filename.is_file():
        return None
    stars = apply_proper_motion(store.load_healpix(outer_pix, read_only=True), epoch=epoch)
    if band == "R" and "R" not in stars.colnames:
        stars["R"] = compute_r_magnitude(stars)

    mask = ~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"])
    if band in stars.colnames:
        mask &= ~np.isnan(stars[band])
    return stars[mask]


def _load_field_asterisms(
    build_path: Path,
    *,
    config: _AsterismPlotConfig,
    outer_pixs: tuple[int, ...],
    center: SkyCoord,
    width_deg: float,
    max_asterisms: int | None,
) -> Table:
    from ao_sky.artifacts import AoSkyArtifactStore

    store = AoSkyArtifactStore(build_path)
    tables = []
    for outer_pix in outer_pixs:
        asterisms = store.asterisms(
            outer_pix,
            outer_level=config.outer_level,
            inner_level=config.inner_level,
            missing_ok=True,
        )
        if asterisms is None or len(asterisms) == 0:
            continue
        mask = _local_square_mask(asterisms["ra"], asterisms["dec"], center, width_deg / 2.0)
        if np.any(mask):
            tables.append(asterisms[mask])
    if not tables:
        return Table()
    asterisms = _deduplicate_physical_asterisms(vstack(tables))
    if max_asterisms is not None:
        asterisms = asterisms[: int(max_asterisms)]
    return asterisms


def _load_field_winner_ee(
    build_path: Path,
    *,
    config: _AsterismPlotConfig,
    outer_pixs: tuple[int, ...],
    center: SkyCoord,
    width_deg: float,
) -> Table:
    from ao_sky.artifacts import AoSkyArtifactStore

    store = AoSkyArtifactStore(build_path)
    tables = []
    for outer_pix in outer_pixs:
        filename = store.outer_path(
            outer_pix,
            outer_level=config.outer_level,
            inner_level=config.inner_level,
        )
        if not filename.is_file():
            continue
        inner = store.inner(outer_pix, outer_level=config.outer_level, inner_level=config.inner_level)
        if len(inner) == 0:
            continue
        pixs = np.asarray(inner["pix"], dtype=np.int64)
        coords = get_pixel_skycoord(config.inner_level, pixs)
        mask = _local_square_mask(coords.ra.deg, coords.dec.deg, center, width_deg / 2.0)
        if not np.any(mask):
            continue
        selected = inner[mask].copy()
        selected["ra"] = coords.ra.deg[mask]
        selected["dec"] = coords.dec.deg[mask]
        tables.append(selected)
    if not tables:
        return Table()

    inner_pixels = vstack(tables)
    _, keep = np.unique(np.asarray(inner_pixels["pix"], dtype=np.int64), return_index=True)
    keep.sort()
    return inner_pixels[keep]


def _deduplicate_physical_asterisms(asterisms: Table) -> Table:
    seen: set[tuple[int, ...]] = set()
    keep = np.zeros(len(asterisms), dtype=bool)
    for index, row in enumerate(asterisms):
        key = _physical_asterism_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep[index] = True
    return asterisms[keep]


def _physical_asterism_key(row) -> tuple[int, ...]:  # noqa: ANN001
    num_stars = max(0, min(3, int(row["num_stars"])))
    source_ids = [
        int(row[f"star{index}_source_id"])
        for index in range(1, num_stars + 1)
        if int(row[f"star{index}_source_id"]) >= 0
    ]
    return tuple(sorted(source_ids))


def _winner_ee_field_name(ee_kind: str) -> str:
    value = str(ee_kind).strip().lower()
    if value in {"resolved", "winner_ee_resolved"}:
        return "winner_ee_resolved"
    if value in {"averaged", "winner_ee_averaged"}:
        return "winner_ee_averaged"
    raise PlottingError(
        "ee_kind must be one of: resolved, averaged, winner_ee_resolved, winner_ee_averaged"
    )


def _require_winner_ee_columns(inner_pixels: Table, field: str) -> None:
    required = {"ra", "dec", field}
    missing = required - set(inner_pixels.colnames)
    if missing:
        raise PlottingError(f"inner_pixels is missing required columns: {', '.join(sorted(missing))}")


def _draw_winner_ee_field(
    ax: Axes,
    inner_pixels: Table,
    *,
    center: SkyCoord,
    half_width_deg: float,
    field: str,
    cmap: str,
    vmin: float,
    vmax: float,
    grid_resolution: int,
):  # noqa: ANN202
    if len(inner_pixels) == 0:
        return None

    mask = _local_square_mask(inner_pixels["ra"], inner_pixels["dec"], center, half_width_deg)
    if not np.any(mask):
        return None

    values = np.asarray(inner_pixels[field], dtype=float)[mask]
    x_deg, y_deg = _local_offsets_deg(
        np.asarray(inner_pixels["ra"], dtype=float)[mask],
        np.asarray(inner_pixels["dec"], dtype=float)[mask],
        center,
    )
    valid = np.isfinite(values) & (values > 0.0)
    if np.count_nonzero(valid) < 3:
        return None

    xi = np.linspace(-half_width_deg, half_width_deg, grid_resolution)
    yi = np.linspace(-half_width_deg, half_width_deg, grid_resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata(
        np.column_stack((x_deg[valid], y_deg[valid])),
        values[valid],
        (grid_x, grid_y),
        method="cubic",
    )

    return ax.imshow(
        grid_z,
        extent=(-half_width_deg, half_width_deg, -half_width_deg, half_width_deg),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        zorder=0,
    )


def _draw_stars(
    ax: Axes,
    stars: Table,
    center: SkyCoord,
    *,
    band: str,
    min_mag: float,
) -> None:
    if len(stars) == 0:
        return

    x_deg, y_deg = _local_offsets_deg(stars["ra"], stars["dec"], center)
    magnitudes = (
        np.asarray(stars[band], dtype=float)
        if band in stars.colnames
        else np.full(len(stars), np.nan)
    )
    sizes = _linear_marker_sizes(
        magnitudes,
        min_mag=7.0,
        max_mag=19.0,
        min_size=2.0,
        max_size=35.0,
    )
    bright = np.isfinite(magnitudes) & (magnitudes < float(min_mag))
    normal = ~bright
    if np.any(normal):
        ax.scatter(
            x_deg[normal],
            y_deg[normal],
            s=sizes[normal],
            facecolors="white",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
    if np.any(bright):
        ax.scatter(
            x_deg[bright],
            y_deg[bright],
            s=sizes[bright],
            facecolors="#f4a261",
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )


def _draw_member_stars(
    ax: Axes,
    row: Row,
    *,
    center: SkyCoord,
    min_mag: float,
    band: str,
) -> None:
    points = _asterism_member_offsets_arcsec(row, center)
    if not points:
        return
    x_arcsec = np.asarray([point[0] for point in points], dtype=float)
    y_arcsec = np.asarray([point[1] for point in points], dtype=float)
    magnitudes = _asterism_member_magnitudes(row, band=band)
    sizes = _linear_marker_sizes(
        magnitudes,
        min_mag=7.0,
        max_mag=19.0,
        min_size=28.0,
        max_size=95.0,
    )
    bright = np.isfinite(magnitudes) & (magnitudes < float(min_mag))
    normal = ~bright
    if np.any(normal):
        ax.scatter(
            x_arcsec[normal],
            y_arcsec[normal],
            s=sizes[normal],
            facecolors="#f4a261",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    if np.any(bright):
        ax.scatter(
            x_arcsec[bright],
            y_arcsec[bright],
            s=sizes[bright],
            facecolors="#e76f51",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )


def _draw_background_stars_arcsec(
    ax: Axes,
    stars: Table,
    center: SkyCoord,
    *,
    extent_arcsec: float,
    band: str,
    min_mag: float,
) -> None:
    if len(stars) == 0:
        return
    x_deg, y_deg = _local_offsets_deg(stars["ra"], stars["dec"], center)
    x_arcsec = x_deg * 3600.0
    y_arcsec = y_deg * 3600.0
    mask = (np.abs(x_arcsec) <= extent_arcsec) & (np.abs(y_arcsec) <= extent_arcsec)
    if not np.any(mask):
        return

    magnitudes = (
        np.asarray(stars[band], dtype=float)
        if band in stars.colnames
        else np.full(len(stars), np.nan)
    )
    sizes = _linear_marker_sizes(
        magnitudes,
        min_mag=7.0,
        max_mag=19.0,
        min_size=5.0,
        max_size=32.0,
    )
    bright = np.isfinite(magnitudes) & (magnitudes < float(min_mag))
    normal = ~bright
    draw_mask = mask & normal
    if np.any(draw_mask):
        ax.scatter(
            x_arcsec[draw_mask],
            y_arcsec[draw_mask],
            s=sizes[draw_mask],
            facecolors="white",
            edgecolors="#343a40",
            linewidths=0.45,
            alpha=0.82,
            zorder=2,
        )
    draw_mask = mask & bright
    if np.any(draw_mask):
        ax.scatter(
            x_arcsec[draw_mask],
            y_arcsec[draw_mask],
            s=sizes[draw_mask],
            facecolors="#ffe8cc",
            edgecolors="#343a40",
            linewidths=0.5,
            alpha=0.9,
            zorder=2,
        )


def _draw_fov_circles(
    ax: Axes,
    asterisms: Table,
    center: SkyCoord,
    fov_radius_deg: float,
) -> None:
    if len(asterisms) == 0:
        return
    colors = {1: "#f4a261", 2: "#2a9d8f", 3: "#457b9d"}
    for row in asterisms:
        x_deg, y_deg = _local_offsets_deg([row["ra"]], [row["dec"]], center)
        ax.add_patch(
            Circle(
                (float(x_deg[0]), float(y_deg[0])),
                fov_radius_deg,
                facecolor="none",
                edgecolor=colors.get(int(row["num_stars"]), "#457b9d"),
                linewidth=0.9,
                alpha=0.8,
                zorder=1,
            )
        )


def _draw_coverable_region_mask(
    ax: Axes,
    asterisms: Table,
    *,
    center: SkyCoord,
    fov_radius_deg: float,
    half_width_deg: float,
) -> None:
    if len(asterisms) == 0:
        return
    try:
        from shapely.geometry import Point, box
        from shapely.ops import unary_union
    except ImportError as exc:
        raise PlottingError("shapely is required to draw coverable-region masks") from exc

    regions = []
    for row in asterisms:
        points = _asterism_member_offsets_deg(row, center)
        if not points:
            continue
        valid_centers = Point(points[0]).buffer(fov_radius_deg, quad_segs=96)
        for point in points[1:]:
            valid_centers = valid_centers.intersection(
                Point(point).buffer(fov_radius_deg, quad_segs=96)
            )
        if valid_centers.is_empty:
            continue
        regions.append(valid_centers.buffer(fov_radius_deg, quad_segs=96))
    if not regions:
        return

    field = box(-half_width_deg, -half_width_deg, half_width_deg, half_width_deg)
    mask = unary_union(regions).intersection(field)
    _draw_geometry(
        ax,
        mask,
        facecolor="#2a9d8f",
        edgecolor="#087f73",
        alpha=0.18,
        linewidth=0.8,
        zorder=0,
    )


def _draw_single_centered_fov(
    ax: Axes,
    row: Row,
    *,
    center: SkyCoord,
    fov_radius_arcsec: float,
) -> None:
    x_deg, y_deg = _local_offsets_deg([row["ra"]], [row["dec"]], center)
    ax.add_patch(
        Circle(
            (float(x_deg[0]) * 3600.0, float(y_deg[0]) * 3600.0),
            fov_radius_arcsec,
            facecolor="none",
            edgecolor="#e76f51",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
        )
    )


def _draw_connections(ax: Axes, asterisms: Table, center: SkyCoord) -> None:
    if len(asterisms) == 0:
        return
    for row in asterisms:
        num_stars = int(row["num_stars"])
        if num_stars < 2:
            continue
        points = []
        for index in range(1, min(num_stars, 3) + 1):
            x_deg, y_deg = _local_offsets_deg(
                [row[f"star{index}_ra"]],
                [row[f"star{index}_dec"]],
                center,
            )
            points.append((float(x_deg[0]), float(y_deg[0])))
        for start, stop in zip(points, points[1:], strict=False):
            ax.plot(
                [start[0], stop[0]],
                [start[1], stop[1]],
                color="#6c757d",
                linewidth=0.8,
                alpha=0.65,
                zorder=2,
            )
        if len(points) == 3:
            ax.plot(
                [points[2][0], points[0][0]],
                [points[2][1], points[0][1]],
                color="#6c757d",
                linewidth=0.8,
                alpha=0.65,
                zorder=2,
            )


def _draw_connections_arcsec(ax: Axes, asterisms: Table, center: SkyCoord) -> None:
    if len(asterisms) == 0:
        return
    for row in asterisms:
        points = _asterism_member_offsets_arcsec(row, center)
        if len(points) < 2:
            continue
        for start, stop in zip(points, points[1:], strict=False):
            ax.plot(
                [start[0], stop[0]],
                [start[1], stop[1]],
                color="#6c757d",
                linewidth=1.1,
                alpha=0.75,
                zorder=4,
            )
        if len(points) == 3:
            ax.plot(
                [points[2][0], points[0][0]],
                [points[2][1], points[0][1]],
                color="#6c757d",
                linewidth=1.1,
                alpha=0.75,
                zorder=4,
            )


def _asterism_feasible_geometries(
    row: Row,
    *,
    center: SkyCoord,
    fov_radius_arcsec: float,
):  # noqa: ANN202
    try:
        from shapely.geometry import Point
    except ImportError as exc:
        raise PlottingError(
            "shapely is required to draw asterism feasible-region overlays"
        ) from exc

    points = _asterism_member_offsets_arcsec(row, center)
    if not points:
        raise PlottingError("asterism has no member-star positions")

    valid_centers = Point(points[0]).buffer(fov_radius_arcsec, quad_segs=192)
    for point in points[1:]:
        valid_centers = valid_centers.intersection(
            Point(point).buffer(fov_radius_arcsec, quad_segs=192)
        )
    if valid_centers.is_empty:
        raise PlottingError("asterism guide stars do not fit inside one FoV")
    coverable_region = valid_centers.buffer(fov_radius_arcsec, quad_segs=192)
    return valid_centers, coverable_region


def _draw_geometry(
    ax: Axes,
    geometry,  # noqa: ANN001
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    geometries = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for geom in geometries:
        if geom.is_empty:
            continue
        ax.add_patch(
            PathPatch(
                _polygon_path_with_holes(geom),
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
        )


def _polygon_path_with_holes(polygon) -> MplPath:  # noqa: ANN001
    from shapely.geometry.polygon import orient

    polygon = orient(polygon, sign=1.0)
    vertices = []
    codes = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = np.asarray(ring.coords, dtype=float)
        if len(coords) < 2:
            continue
        vertices.extend(coords)
        codes.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(coords) - 2)
            + [MplPath.CLOSEPOLY]
        )
    return MplPath(np.asarray(vertices, dtype=float), np.asarray(codes, dtype=np.uint8))


def _asterism_member_offsets_arcsec(row: Row, center: SkyCoord) -> list[tuple[float, float]]:
    points = []
    for index in range(1, min(3, int(row["num_stars"])) + 1):
        x_deg, y_deg = _local_offsets_deg(
            [row[f"star{index}_ra"]],
            [row[f"star{index}_dec"]],
            center,
        )
        points.append((float(x_deg[0]) * 3600.0, float(y_deg[0]) * 3600.0))
    return points


def _asterism_member_offsets_deg(row: Row, center: SkyCoord) -> list[tuple[float, float]]:
    points = []
    for index in range(1, min(3, int(row["num_stars"])) + 1):
        x_deg, y_deg = _local_offsets_deg(
            [row[f"star{index}_ra"]],
            [row[f"star{index}_dec"]],
            center,
        )
        points.append((float(x_deg[0]), float(y_deg[0])))
    return points


def _asterism_member_magnitudes(row: Row, *, band: str) -> np.ndarray:
    values = []
    for index in range(1, min(3, int(row["num_stars"])) + 1):
        band_column = f"star{index}_{band}"
        mag_column = f"star{index}_mag"
        if band_column in row.colnames:
            values.append(float(row[band_column]))
        elif mag_column in row.colnames:
            values.append(float(row[mag_column]))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _linear_marker_sizes(
    magnitudes: np.ndarray,
    *,
    min_mag: float,
    max_mag: float,
    min_size: float,
    max_size: float,
) -> np.ndarray:
    values = np.asarray(magnitudes, dtype=float)
    clipped = np.clip(values, min_mag, max_mag)
    scale = (max_mag - clipped) / (max_mag - min_mag)
    sizes = min_size + scale * (max_size - min_size)
    sizes[~np.isfinite(values)] = min_size
    return sizes


def _style_axis(ax: Axes, half_width_deg: float) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-half_width_deg, half_width_deg)
    ax.set_ylim(-half_width_deg, half_width_deg)
    ax.set_xlabel(r"$\Delta \mathrm{RA}\cos\delta\ [\mathrm{deg}]$")
    ax.set_ylabel(r"$\Delta \mathrm{Dec}\ [\mathrm{deg}]$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)


def _single_asterism_extent_arcsec(
    row: Row,
    *,
    center: SkyCoord,
    fov_radius_arcsec: float,
) -> float:
    points = _asterism_member_offsets_arcsec(row, center)
    x_deg, y_deg = _local_offsets_deg([row["ra"]], [row["dec"]], center)
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_values.append(float(x_deg[0]) * 3600.0)
    y_values.append(float(y_deg[0]) * 3600.0)
    return max(
        fov_radius_arcsec * 2.3,
        max(abs(value) for value in x_values) + fov_radius_arcsec * 2.1,
        max(abs(value) for value in y_values) + fov_radius_arcsec * 2.1,
    )


def _style_single_asterism_axis(
    ax: Axes,
    *,
    extent_arcsec: float,
) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent_arcsec, extent_arcsec)
    ax.set_ylim(-extent_arcsec, extent_arcsec)
    ax.set_xlabel(r"$\Delta \mathrm{RA}\cos\delta\ [\mathrm{arcsec}]$")
    ax.set_ylabel(r"$\Delta \mathrm{Dec}\ [\mathrm{arcsec}]$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)


def _local_offsets_deg(
    ra_deg: Any,
    dec_deg: Any,
    center: SkyCoord,
) -> tuple[np.ndarray, np.ndarray]:
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    delta_ra = (ra_deg - center.ra.deg + 180.0) % 360.0 - 180.0
    x_deg = delta_ra * np.cos(center.dec.to_value(u.rad))
    y_deg = dec_deg - center.dec.deg
    return x_deg, y_deg


def _local_square_mask(
    ra_deg: Any,
    dec_deg: Any,
    center: SkyCoord,
    half_width_deg: float,
) -> np.ndarray:
    x_deg, y_deg = _local_offsets_deg(ra_deg, dec_deg, center)
    return (np.abs(x_deg) <= half_width_deg) & (np.abs(y_deg) <= half_width_deg)


__all__ = [
    "plot_asterism",
    "plot_asterisms",
    "plot_build_asterisms",
    "plot_build_winner_ee",
    "plot_winner_ee",
    "read_asterism_stars",
    "read_build_asterisms",
    "read_build_winner_ee",
]
