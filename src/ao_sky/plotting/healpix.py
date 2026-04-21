"""HEALPix sky-map plotting helpers.

This module is intentionally a close port of the working `survey_tools`
HEALPix plotting path. Plotting through `skyproj` is sensitive to small
ordering and option changes, so cleanup should happen in the artifact and
field-convention layers rather than in this renderer.
"""

from __future__ import annotations

import copy
from logging import warning

from astropy.coordinates import Latitude, Longitude, SkyCoord
from astropy_healpix import HEALPix
import astropy.units as u
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axisartist import angle_helper
import numpy as np
import skyproj

from ._exceptions import PlottingError


GLOBE_PROJECTIONS = ["aitoff", "hammer", "lambert", "mollweide"]


class WrappedFormatterHMS(angle_helper.FormatterHMS):
    """Skyproj-compatible HMS tick formatter copied from the legacy pattern."""

    def __init__(self, longitude_ticks: str) -> None:
        self._formatter = skyproj.mpl_utils.WrappedFormatterDMS(180, longitude_ticks)

    def __call__(self, direction, factor, values):  # noqa: ANN001
        return super().__call__(direction, factor, self._formatter._wrap_values(factor, values))


def get_nside(level: int) -> int:
    """Return the nested HEALPix nside for a level."""

    return 2 ** int(level)


def get_npix(level: int) -> int:
    """Return the nested HEALPix pixel count for a level."""

    return 12 * 4 ** int(level)


def is_npix_valid(npix: int) -> bool:
    """Return whether a dense HEALPix length can be mapped to a level."""

    nside = np.sqrt(int(npix) / 12.0)
    return nside == int(nside)


def get_level(npix: int) -> int | None:
    """Infer a HEALPix level from a dense map length."""

    if is_npix_valid(npix):
        return int(np.log2(np.sqrt(int(npix) / 12.0)))
    return None


def get_resolution(level: int) -> u.Quantity:
    """Return the approximate angular size of a nested HEALPix pixel."""

    area = 4 * np.pi * (180.0 / np.pi) ** 2 / get_npix(level) * 3600.0**2
    return np.sqrt(area * u.arcsec**2)


def get_healpix_from_skycoord(level: int, skycoords: SkyCoord, frame: str = "icrs") -> np.ndarray:
    """Return nested HEALPix pixels for sky coordinates."""

    hp = HEALPix(nside=get_nside(level), order="nested", frame=frame)
    return np.asarray(hp.skycoord_to_healpix(skycoords), dtype=np.int64)


def get_pixel_skycoord(
    level: int,
    pixs: np.ndarray | list[int] | tuple[int, ...] | None = None,
    *,
    frame: str = "icrs",
) -> SkyCoord:
    """Return nested HEALPix pixel-center coordinates."""

    hp = HEALPix(nside=get_nside(level), order="nested", frame=frame)
    if pixs is None:
        pixs = np.arange(hp.npix, dtype=np.int64)
    return hp.healpix_to_skycoord(pixs)


def get_boundaries_skycoord(
    level: int,
    pixs: np.ndarray | list[int] | tuple[int, ...] | None = None,
    step: int = 1,
    *,
    frame: str = "icrs",
) -> SkyCoord:
    """Return nested HEALPix pixel boundary coordinates."""

    hp = HEALPix(nside=get_nside(level), order="nested", frame=frame)
    if pixs is None:
        pixs = np.arange(hp.npix, dtype=np.int64)
    boundaries = hp.boundaries_skycoord(pixs, step)
    if np.size(pixs) == 1:
        return boundaries[0]
    return boundaries


def plot_healpix(
    values: np.ndarray | None,
    *,
    level: int | None = None,
    pixs: np.ndarray | None = None,
    skycoords: SkyCoord | None = None,
    contour_values: np.ndarray | None = None,
    plot_properties: dict[str, object] | None = None,
):
    """Plot a HEALPix map using the legacy `skyproj` rendering path."""

    if plot_properties is None:
        plot_properties = {}
    _set_default_plot_properties(values, contour_values, plot_properties)

    with mpl.rc_context(
        {
            "xtick.labelcolor": plot_properties["colors"]["xtick_label"],
            "xtick.labelsize": plot_properties["fontsize"]["xtick_label"],
            "ytick.labelcolor": plot_properties["colors"]["ytick_label"],
            "ytick.labelsize": plot_properties["fontsize"]["ytick_label"],
        }
    ):
        fig, ax = plt.subplots(
            figsize=(plot_properties["width"], plot_properties["height"]),
            dpi=plot_properties["dpi"],
        )

        kwargs = {
            "celestial": plot_properties["flip"],
            "galactic": plot_properties["mapcoord"] == "G",
            "lon_0": plot_properties["rotation"],
            "gridlines": plot_properties["grid"],
            "longitude_ticks": (
                "symmetric" if plot_properties["grid_longitude"] == "degrees" else "positive"
            ),
        }

        match plot_properties["projection"]:
            case "mollweide":
                sp = skyproj.MollweideSkyproj(ax, **kwargs)
            case "gnomonic":
                sp = skyproj.GnomonicSkyproj(ax, **kwargs)
            case _:
                sp = skyproj.Skyproj(ax, **kwargs)

        plot_properties["fig"] = fig
        plot_properties["ax"] = fig.gca()
        plot_properties["sp"] = sp

        _draw_map(values, level, pixs, skycoords, **plot_properties)

        if contour_values is not None:
            contour_plot_properties = plot_properties.copy()
            contour_plot_properties.update(
                {
                    "zoom": False,
                    "cmap": plot_properties["contour_cmap"],
                    "norm": plot_properties["contour_norm"],
                    "vmin": plot_properties["contour_vmin"],
                    "vmax": plot_properties["contour_vmax"],
                }
            )
            _draw_map(
                contour_values,
                None,
                None,
                None,
                plot_type="contour",
                **contour_plot_properties,
            )

        _draw_grid(**plot_properties)
        _draw_cbar(**plot_properties)
        _draw_boundaries(**plot_properties)

        if plot_properties.get("milkyway", False):
            sp.draw_milky_way(
                width=(
                    plot_properties["milkyway_width"]
                    if plot_properties.get("milkyway_width", None) is not None
                    else 10
                ),
                linewidth=1.2,
                color=plot_properties["colors"]["milkyway"],
                alpha=plot_properties["alphas"]["milkyway"],
                linestyle="-",
            )

        if plot_properties.get("ecliptic", False):
            _draw_ecliptic(
                sp=plot_properties["sp"],
                galactic=plot_properties["galactic"],
                width=(
                    plot_properties["ecliptic_width"]
                    if plot_properties.get("ecliptic_width", None) is not None
                    else 10
                ),
                linewidth=1.2,
                color=plot_properties["colors"]["ecliptic"],
                alpha=plot_properties["alphas"]["ecliptic"],
                linestyle="-",
            )

        if plot_properties.get("surveys", None) is not None:
            _draw_surveys(**plot_properties)

        if plot_properties.get("points", None) is not None:
            _draw_points(**plot_properties)

        if plot_properties.get("tissot", False):
            sp.tissot_indicatrices()

        _finish_plot(**plot_properties)
        return fig


def _update_dictionary(dict1: dict[str, object], dict2: dict[str, object]) -> dict[str, object]:
    for key in dict1.keys():
        if key in dict2:
            dict1[key] = dict2[key]
    return dict1


def _set_default_plot_properties(
    values: np.ndarray | None,
    contour_values: np.ndarray | None,
    plot_properties: dict[str, object],
) -> dict[str, object]:
    if plot_properties.get("mapcoord", None) is not None:
        raise PlottingError("mapcoord not supported with skyproj")
    plot_properties["galactic"] = bool(plot_properties.get("galactic", False))
    plot_properties["mapcoord"] = "G" if plot_properties["galactic"] else "C"

    if plot_properties.get("projection", None) is None:
        plot_properties["projection"] = "cartesian"

    if plot_properties["projection"] in [
        "cart",
        "equi",
        "equirectangular",
        "rectangular",
        "lacarte",
        "platecarree",
    ]:
        plot_properties["projection"] = "cartesian"

    if plot_properties.get("zoom", None) is None:
        plot_properties["zoom"] = False
    if plot_properties.get("rotation", None) is None:
        plot_properties["rotation"] = 0.0
    if plot_properties.get("flip", None) is None:
        plot_properties["flip"] = True

    if plot_properties.get("width", None) is None:
        plot_properties["width"] = 10
    if plot_properties.get("height", None) is None:
        plot_properties["height"] = plot_properties["width"] / 1.618
    if plot_properties.get("dpi", None) is None:
        plot_properties["dpi"] = 100
    if plot_properties.get("xsize", None) is None:
        plot_properties["xsize"] = int(plot_properties["width"] * 0.8 * plot_properties["dpi"])

    if plot_properties.get("grid", None) is None:
        plot_properties["grid"] = True
    if plot_properties.get("grid_longitude", None) is None:
        plot_properties["grid_longitude"] = "degrees"

    color_defaults = {
        "background": "white",
        "badvalue": "gray",
        "grid": (0.8, 0.8, 0.8) if values is not None else (0.5, 0.5, 0.5),
        "xtick_label": "black",
        "ytick_label": "black",
        "boundaries": "red",
        "milkyway": "black",
        "ecliptic": "black",
    }

    if plot_properties["projection"] in GLOBE_PROJECTIONS:
        color_defaults["xtick_label"] = color_defaults["grid"]

    if "colors" in plot_properties and plot_properties["colors"] is not None:
        plot_properties["colors"] = _update_dictionary(color_defaults, plot_properties["colors"])
    else:
        plot_properties["colors"] = color_defaults

    alpha_defaults = {
        "grid": 1.0,
        "boundaries": 1.0,
        "milkyway": 1.0,
        "ecliptic": 1.0,
    }
    if "alphas" in plot_properties and plot_properties["alphas"] is not None:
        plot_properties["alphas"] = _update_dictionary(alpha_defaults, plot_properties["alphas"])
    else:
        plot_properties["alphas"] = alpha_defaults

    fontsize_defaults = {
        "xlabel": 12,
        "ylabel": 12,
        "title": 14,
        "xtick_label": 10,
        "ytick_label": 10,
        "cbar_label": 12,
        "cbar_tick_label": 10,
        "boundaries_label": 10,
        "boundaries_label_small": max(8, 12.0 - (plot_properties.get("boundaries_level") or 0) / 2.0),
        "points_label": 8,
        "stars_legend": 10,
    }
    if "fontsize" in plot_properties and plot_properties["fontsize"] is not None:
        plot_properties["fontsize"] = _update_dictionary(fontsize_defaults, plot_properties["fontsize"])
    else:
        plot_properties["fontsize"] = fontsize_defaults

    cmap = plot_properties.get("cmap", None)
    if cmap is None:
        cmap = "viridis"

    if isinstance(cmap, str):
        cmap0 = plt.get_cmap(cmap)
    elif isinstance(cmap, mpl.colors.Colormap):
        cmap0 = cmap
    else:
        cmap0 = plt.get_cmap(mpl.rcParams["image.cmap"])

    cmap = copy.copy(cmap0)
    cmap.set_over(plot_properties["colors"]["badvalue"])
    cmap.set_under(plot_properties["colors"]["badvalue"])
    cmap.set_bad(plot_properties["colors"]["badvalue"])
    plot_properties["cmap"] = cmap

    if values is not None:
        if plot_properties.get("vmin", None) is None and plot_properties.get("cbar_ticks", None) is not None:
            plot_properties["vmin"] = np.min(plot_properties["cbar_ticks"])
        if plot_properties.get("vmax", None) is None and plot_properties.get("cbar_ticks", None) is not None:
            plot_properties["vmax"] = np.max(plot_properties["cbar_ticks"])
        plot_properties["norm"], plot_properties["vmin"], plot_properties["vmax"] = _prepare_norm(
            values,
            plot_properties.get("norm", None),
            plot_properties.get("vmin", None),
            plot_properties.get("vmax", None),
        )

    if contour_values is not None:
        if "contour_cmap" not in plot_properties:
            plot_properties["contour_cmap"] = None
        (
            plot_properties["contour_norm"],
            plot_properties["contour_vmin"],
            plot_properties["contour_vmax"],
        ) = _prepare_norm(
            contour_values,
            plot_properties.get("contour_norm", None),
            plot_properties.get("contour_vmin", None),
            plot_properties.get("contour_vmax", None),
        )
        if "contour_levels" not in plot_properties:
            plot_properties["contour_levels"] = None
        if plot_properties.get("contour_filled", None) is None:
            plot_properties["contour_filled"] = False
        if plot_properties.get("contour_colors", None) is None:
            plot_properties["contour_colors"] = "k"
        if plot_properties.get("contour_alpha", None) is None:
            plot_properties["contour_alpha"] = 0.25

    if plot_properties.get("hide_title", None) is None:
        plot_properties["hide_title"] = False
    if plot_properties.get("hide_cbar", None) is None:
        plot_properties["hide_cbar"] = False

    return plot_properties


def _prepare_norm(
    values: np.ndarray,
    norm: str | mpl.colors.Normalize | None,
    vmin: float | None,
    vmax: float | None,
) -> tuple[mpl.colors.Normalize, float | None, float | None]:
    is_valid = ~np.isnan(values) & ~np.isinf(values)

    if vmin is None and np.sum(is_valid) > 0:
        vmin = float(np.min(values[is_valid]))
    if vmax is None and np.sum(is_valid) > 0:
        vmax = float(np.max(values[is_valid]))

    if norm is None:
        norm = mpl.colors.NoNorm()
    elif isinstance(norm, str):
        match norm.lower():
            case "lin" | "linear" | "norm" | "normalize":
                norm = mpl.colors.Normalize()
            case "log":
                norm = mpl.colors.LogNorm()
            case "symlog":
                norm = mpl.colors.SymLogNorm(1, linscale=0.1, clip=True, base=10)
            case _:
                raise PlottingError("Unrecognized norm")

    norm.vmin = vmin
    norm.vmax = vmax
    norm.autoscale_None(values[is_valid])
    return norm, vmin, vmax


def _draw_map(
    values: np.ndarray | None,
    level: int | None,
    pixs: np.ndarray | None,
    skycoords: SkyCoord | None,
    *,
    plot_type: str = "mesh",
    sp=None,  # noqa: ANN001
    mapcoord: str | None = None,
    zoom: bool = False,
    xsize: int = 1000,
    cmap=None,  # noqa: ANN001
    norm=None,  # noqa: ANN001
    **kwargs,
) -> None:
    if values is None:
        return
    if len(values) < 2:
        raise PlottingError("multiple values required")
    if plot_type not in ["mesh", "contour"]:
        raise PlottingError("Invalid map type")

    num_values = len(values)
    if level is None:
        if not is_npix_valid(num_values):
            raise PlottingError("level and pixs required when passing a partial map")
        level = get_level(num_values)
    if level is None:
        raise PlottingError("Unable to infer HEALPix level")

    npix = get_npix(level)
    if pixs is None:
        if num_values != npix:
            raise PlottingError("pixs must be specified when values are not a full HEALpix map")
    else:
        if num_values == npix:
            pixs = None
        elif num_values != len(pixs):
            raise PlottingError("values and pixs must be the same size")

    if pixs is not None:
        if mapcoord != "C":
            raise PlottingError("rotating coordinates is not supported when pixs specified")
        if plot_type == "contour":
            raise PlottingError("contour plot is not supported when pixs specified")
        sp.draw_hpxpix(get_nside(level), pixs, values, nest=True, zoom=zoom, xsize=xsize, cmap=cmap, norm=norm)
    else:
        if mapcoord != "C":
            values = _remap_dense_values_to_mapcoord(values, level=level, skycoords=skycoords, mapcoord=mapcoord)

        if plot_type == "contour":
            _draw_contour_map(sp, values, xsize=xsize, cmap=cmap, norm=norm, **kwargs)
        else:
            sp.draw_hpxmap(values, nest=True, zoom=False, xsize=xsize, cmap=cmap, norm=norm)


def _remap_dense_values_to_mapcoord(
    values: np.ndarray,
    *,
    level: int,
    skycoords: SkyCoord | None,
    mapcoord: str,
) -> np.ndarray:
    if skycoords is None:
        skycoords = get_pixel_skycoord(level)
    if mapcoord == "G":
        coords = SkyCoord(
            l=skycoords.ra.degree * u.degree,
            b=skycoords.dec.degree * u.degree,
            frame="galactic",
        ).icrs
        lon = coords.ra.degree
        lat = coords.dec.degree
    else:
        raise PlottingError(f"Unsupported map coordinate system: {mapcoord}")
    remap_pixs = get_healpix_from_skycoord(
        level,
        SkyCoord(lon, lat, unit=(u.degree, u.degree), frame="icrs"),
    )
    return values[remap_pixs]


def _draw_contour_map(
    sp,  # noqa: ANN001
    values: np.ndarray,
    *,
    ax=None,  # noqa: ANN001
    projection: str = "cartesian",
    rotation: float | None = None,
    xsize: int = 1000,
    cmap=None,  # noqa: ANN001
    norm=None,  # noqa: ANN001
    contour_levels=None,  # noqa: ANN001
    contour_filled: bool = False,
    contour_colors=None,  # noqa: ANN001
    contour_alpha: float | None = None,
    **kwargs,  # noqa: ARG001
) -> None:
    if projection != "cartesian":
        warning("Contour plot only works correctly with the cartesian projection")

    extent = sp.get_extent()
    lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
    lat_range = [extent[2], extent[3]]

    aspect = 0.5
    lon, lat = np.meshgrid(
        np.linspace(lon_range[0], lon_range[1], xsize),
        np.linspace(lat_range[0], lat_range[1], int(aspect * xsize)),
    )

    level = get_level(len(values))
    if level is None:
        raise PlottingError("Unable to infer contour HEALPix level")
    hp = HEALPix(nside=get_nside(level), order="nested", frame="icrs")
    pixs = hp.lonlat_to_healpix(Longitude(lon, unit=u.degree), Latitude(lat, unit=u.degree))
    values_raster = values[pixs]

    if rotation is not None:
        lon = (lon - rotation + 180) % 360 - 180

    masked = np.ma.array(values_raster, mask=np.isnan(values_raster))
    if contour_filled:
        ax.contourf(
            lon,
            lat,
            masked,
            levels=contour_levels,
            transform=ax.projection if projection != "cartesian" else None,
            cmap=cmap,
            colors=contour_colors if cmap is None else None,
            norm=norm,
            vmin=norm.vmin if norm is None else None,
            vmax=norm.vmax if norm is None else None,
            alpha=contour_alpha,
        )
    else:
        ax.contour(
            lon,
            lat,
            masked,
            levels=contour_levels,
            transform=ax.projection if projection != "cartesian" else None,
            cmap=cmap,
            colors=contour_colors if cmap is None else None,
            norm=norm,
            vmin=norm.vmin if norm is None else None,
            vmax=norm.vmax if norm is None else None,
            linewidths=0.5,
            alpha=contour_alpha,
        )


def _as_overlay_list(items):  # noqa: ANN001
    if items is None:
        return []
    if isinstance(items, list) and all(isinstance(item, list) for item in items):
        return items
    return [items]


def _draw_surveys(
    *,
    sp=None,  # noqa: ANN001
    surveys=None,  # noqa: ANN001
    galactic: bool = False,
    fontsize: dict[str, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> None:
    if surveys is None or fontsize is None:
        return

    for survey in _as_overlay_list(surveys):
        if isinstance(survey, str):
            filename = survey
            styles = {}
            label = None
            label_offset = (0.0, 0.0)
            text_styles = {}
        elif isinstance(survey, list) and len(survey) >= 1:
            filename = survey[0]
            styles = dict(survey[1]) if len(survey) > 1 and isinstance(survey[1], dict) else {}
            label = survey[2] if len(survey) > 2 else None
            if len(survey) > 3 and isinstance(survey[3], list | tuple) and len(survey[3]) == 2:
                label_offset = (float(survey[3][0]), float(survey[3][1]))
            else:
                label_offset = (0.0, 0.0)
            text_styles = dict(survey[4]) if len(survey) > 4 and isinstance(survey[4], dict) else {}
        else:
            continue

        data = np.genfromtxt(filename, names=["lon", "lat", "poly"])
        styles.setdefault("edgecolor", "red")
        styles.setdefault("linewidth", 2.0)
        styles.setdefault("linestyle", "solid")

        for poly_id in np.unique(data["poly"]):
            poly = data[data["poly"] == poly_id]
            lon = (poly["lon"] + 180) % 360 - 180
            lat = poly["lat"]

            if galactic:
                coords = SkyCoord(ra=lon * u.degree, dec=lat * u.degree, frame="icrs")
                lon = coords.galactic.l.degree
                lat = coords.galactic.b.degree

            sp.draw_polygon(lon, lat, **styles)

            if label is None:
                continue
            center_lon = float(np.mean(lon)) + label_offset[0]
            center_lat = float(np.mean(lat)) + label_offset[1]
            plt.text(
                center_lon,
                center_lat,
                label,
                color=styles["edgecolor"],
                fontsize=fontsize["points_label"],
                fontweight="bold",
                ha="center",
                va="center",
                **text_styles,
            )


def _draw_points(
    *,
    sp=None,  # noqa: ANN001
    ax=None,  # noqa: ANN001
    points=None,  # noqa: ANN001
    galactic: bool = False,
    zoom: bool = False,
    fontsize: dict[str, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> None:
    if points is None or fontsize is None:
        return

    for point_layer in _as_overlay_list(points):
        if isinstance(point_layer, np.ndarray):
            points_ra_dec = point_layer
            scatter_options = {}
            text_styles = {}
        elif isinstance(point_layer, list) and len(point_layer) >= 1:
            points_ra_dec = np.asarray(point_layer[0])
            scatter_options = (
                dict(point_layer[1]) if len(point_layer) > 1 and isinstance(point_layer[1], dict) else {}
            )
            text_styles = dict(point_layer[2]) if len(point_layer) > 2 and isinstance(point_layer[2], dict) else {}
        else:
            continue

        if points_ra_dec.size == 0:
            continue
        if points_ra_dec.ndim == 1:
            points_ra_dec = points_ra_dec.reshape(1, -1)
        if points_ra_dec.shape[1] < 2:
            continue

        if points_ra_dec.shape[1] >= 3 and isinstance(points_ra_dec[0, 2], str):
            labels = points_ra_dec[:, 2]
            label_offsets = points_ra_dec[:, 3:5].astype(float) if points_ra_dec.shape[1] >= 5 else None
            points_ra_dec = points_ra_dec[:, 0:2].astype(float)
        else:
            labels = None
            label_offsets = None
            points_ra_dec = points_ra_dec[:, 0:2].astype(float)

        if zoom:
            [xlim, ylim] = np.sort(sp.crs.transform_points(ax.get_xlim(), ax.get_ylim(), inverse=True).transpose())
            xlim = (xlim + 360) % 360
            points_filter = (
                (points_ra_dec[:, 0] >= xlim[0])
                & (points_ra_dec[:, 0] <= xlim[1])
                & (points_ra_dec[:, 1] >= ylim[0])
                & (points_ra_dec[:, 1] <= ylim[1])
            )
            if not np.any(points_filter):
                continue
            points_ra_dec = points_ra_dec[points_filter, :]
            labels = labels[points_filter] if labels is not None else None
            label_offsets = label_offsets[points_filter] if label_offsets is not None else None

        scatter_options.setdefault("s", 10)
        scatter_options.setdefault("marker", "o")
        if not any(key in scatter_options for key in ("color", "c", "facecolor", "fc", "edgecolor", "ec")):
            scatter_options["color"] = "r"

        if labels is not None:
            label_color = _get_point_label_color(scatter_options)
            label_space = 0.015 * np.diff(xlim)[0] if zoom else 5.0

        if galactic:
            points_icrs = SkyCoord(
                ra=points_ra_dec[:, 0] * u.degree,
                dec=points_ra_dec[:, 1] * u.degree,
                frame="icrs",
            )
            plot_lon = points_icrs.galactic.l.degree
            plot_lat = points_icrs.galactic.b.degree
        else:
            plot_lon = points_ra_dec[:, 0]
            plot_lat = points_ra_dec[:, 1]

        ax.scatter(plot_lon, plot_lat, **scatter_options)
        if labels is None:
            continue
        for index, label in enumerate(labels):
            label_dx = label_offsets[index, 0] if label_offsets is not None else label_space
            label_dy = label_offsets[index, 1] if label_offsets is not None else 0.0
            ax.text(
                plot_lon[index] + label_dx,
                plot_lat[index] + label_dy,
                label,
                color=label_color,
                fontsize=fontsize["points_label"],
                fontweight="bold",
                ha="right",
                **text_styles,
            )


def _get_point_label_color(scatter_options: dict[str, object]) -> object:
    for key in ("color", "c", "facecolor", "fc", "edgecolor", "ec"):
        value = scatter_options.get(key)
        if value is not None and value != "none":
            return value
    return "r"


def _draw_grid(
    *,
    sp=None,  # noqa: ANN001
    ax=None,  # noqa: ANN001
    zoom: bool = False,
    grid: bool = True,
    grid_longitude: str | None = None,
    colors: dict[str, object] | None = None,
    alphas: dict[str, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> None:
    if colors is None or alphas is None:
        return
    if not hasattr(ax, "gridlines"):
        is_old_version = True
        aa = sp._aa
        gridlines = aa.gridlines
    else:
        is_old_version = False
        gridlines = ax.gridlines

    if not grid:
        if is_old_version:
            aa.axis["left"].major_ticklabels.set_visible(False)
            aa.axis["right"].major_ticklabels.set_visible(False)
            aa.axis["bottom"].major_ticklabels.set_visible(False)
            aa.axis["top"].major_ticklabels.set_visible(False)
            if sp._boundary_labels:
                for label in sp._boundary_labels:
                    label.remove()
                sp._boundary_labels = []
        return

    gridlines.set_edgecolor(colors["grid"])
    gridlines.set_alpha(alphas["grid"])

    grid_helper = gridlines._grid_helper
    if is_old_version:
        n_grid_lon, n_grid_lat = sp._compute_n_grid_from_extent(ax.get_extent(lonlat=True))
    else:
        n_grid_lon, n_grid_lat = grid_helper._compute_n_grid_from_extent(
            ax.get_extent(),
            n_grid_lon_default=6,
            n_grid_lat_default=6,
        )

    match grid_longitude:
        case "hours":
            lon_locator = angle_helper.LocatorHMS(n_grid_lon, include_last=zoom)
            lon_formatter = WrappedFormatterHMS(sp._longitude_ticks)
        case _:
            lon_locator = angle_helper.LocatorDMS(n_grid_lon, include_last=zoom)
            lon_formatter = None

    lat_locator = angle_helper.LocatorDMS(n_grid_lat, include_last=True)

    if is_old_version:
        grid_helper.update_grid_finder(grid_locator1=lon_locator, grid_locator2=lat_locator)
        if lon_formatter is not None:
            sp._tick_formatter1 = lon_formatter
            grid_helper.update_grid_finder(tick_formatter1=lon_formatter)
    else:
        grid_helper._grid_locator_lon = lon_locator
        grid_helper._grid_locator_lat = lat_locator
        if lon_formatter is not None:
            grid_helper._tick_formatters["lon"] = lon_formatter

    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    grid_helper._update_grid(x1, y1, x2, y2)
    grid_helper._old_limits = (x1, x2, y1, y2)

    if is_old_version:
        sp._draw_aa_bounds_and_labels()


def _draw_boundaries(
    *,
    ax=None,  # noqa: ANN001
    sp=None,  # noqa: ANN001
    galactic: bool = False,
    zoom: bool = False,
    boundaries_level: int | None = None,
    boundaries_pixs=None,  # noqa: ANN001
    colors: dict[str, object] | None = None,
    alphas: dict[str, float] | None = None,
    fontsize: dict[str, float] | None = None,
    **kwargs,  # noqa: ARG001
) -> None:
    if boundaries_level is None:
        return
    if colors is None or alphas is None or fontsize is None:
        return

    max_count = 250
    if boundaries_pixs is not None and not isinstance(boundaries_pixs, list | np.ndarray):
        boundaries_pixs = [boundaries_pixs]

    step = 2 ** max(0, 7 - boundaries_level)
    pixs = np.arange(get_npix(boundaries_level), dtype=np.int64)
    skycoords = get_pixel_skycoord(boundaries_level, pixs)

    [xlim, ylim] = np.sort(sp.crs.transform_points(ax.get_xlim(), ax.get_ylim(), inverse=True).transpose())
    xlim = (xlim + 360) % 360

    if zoom:
        resolution = get_resolution(boundaries_level).to(u.deg).value
        mean_dec = np.mean(ylim)
        dra = resolution * 2 / np.cos(np.deg2rad(mean_dec))
        ddec = resolution * 2
        pixs_filter = (
            (skycoords.ra.degree >= xlim[0] - dra)
            & (skycoords.ra.degree <= xlim[1] + dra)
            & (skycoords.dec.degree >= ylim[0] - ddec)
            & (skycoords.dec.degree <= ylim[1] + ddec)
        )
        pixs = pixs[pixs_filter]
        skycoords = skycoords[pixs_filter]

    boundaries = get_boundaries_skycoord(boundaries_level, pixs=pixs, step=step)
    boundaries_label_fontsize = fontsize["boundaries_label_small"] if not zoom else fontsize["boundaries_label"]

    if galactic:
        boundaries_lon = boundaries.galactic.l.to(u.degree).value
        boundaries_lat = boundaries.galactic.b.to(u.degree).value
        center_lon = skycoords.galactic.l.to(u.degree).value
        center_lat = skycoords.galactic.b.to(u.degree).value
    else:
        boundaries_lon = boundaries.ra.to(u.degree).value
        boundaries_lat = boundaries.dec.to(u.degree).value
        center_lon = skycoords.ra.to(u.degree).value
        center_lat = skycoords.dec.to(u.degree).value

    count = 0
    for i, pix in enumerate(pixs):
        vertices = np.vstack([boundaries_lon[i], boundaries_lat[i]]).transpose()
        if zoom and not np.any(
            (vertices[:, 0] >= xlim[0])
            & (vertices[:, 0] <= xlim[1])
            & (vertices[:, 1] >= ylim[0])
            & (vertices[:, 1] <= ylim[1])
        ):
            continue

        count += 1
        if count > max_count:
            plt.clf()
            raise PlottingError("Too many boundaries to plot")

        sp.draw_polygon(
            lon=vertices[:, 0],
            lat=vertices[:, 1],
            edgecolor=colors["boundaries"],
            alpha=alphas["boundaries"],
        )

        if boundaries_pixs is not None and pix not in boundaries_pixs:
            continue
        if zoom and (
            (center_lon[i] < xlim[0])
            or (center_lon[i] > xlim[1])
            or (center_lat[i] < ylim[0])
            or (center_lat[i] > ylim[1])
        ):
            continue
        ax.text(
            center_lon[i],
            center_lat[i],
            f"{pix}",
            color=colors["boundaries"],
            fontsize=boundaries_label_fontsize,
            ha="center",
            va="center",
        )


def _draw_ecliptic(
    *,
    sp=None,  # noqa: ANN001
    galactic: bool = False,
    width: float = 10,
    linewidth: float = 1.5,
    color: str = "black",
    linestyle: str = "-",
    **kwargs,
) -> None:
    elon = np.linspace(0, 360, 500)
    elat = np.zeros_like(elon)
    ec = SkyCoord(lon=elon * u.degree, lat=elat * u.degree, frame="barycentricmeanecliptic")

    if galactic:
        lon = ec.galactic.l.degree
        lat = ec.galactic.b.degree
    else:
        lon = ec.fk5.ra.degree
        lat = ec.fk5.dec.degree

    sp.plot(lon, lat, linewidth=linewidth, color=color, linestyle=linestyle, **kwargs)

    if width > 0:
        for delta in [+width, -width]:
            ec = SkyCoord(
                lon=elon * u.degree,
                lat=(elat + delta) * u.degree,
                frame="barycentricmeanecliptic",
            )
            if galactic:
                lon = ec.galactic.l.degree
                lat = ec.galactic.b.degree
            else:
                lon = ec.fk5.ra.degree
                lat = ec.fk5.dec.degree
            sp.plot(lon, lat, linewidth=1.0, color=color, linestyle="--", **kwargs)


def _draw_cbar(
    *,
    sp=None,  # noqa: ANN001
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    cbar_orientation: str = "vertical",
    cbar_ticks=None,  # noqa: ANN001
    cbar_format: str = "%g",
    cbar_pad: float = 0.03,
    cbar_shrink: float = 0.6,
    cbar_unit: str = "",
    fontsize: dict[str, float] | None = None,
    hide_cbar: bool = False,
    **kwargs,  # noqa: ARG001
):
    if not cbar or vmin is None or vmax is None or fontsize is None:
        return None

    location = "bottom" if cbar_orientation == "horizontal" else "right"
    ax = getattr(sp, "_ax", None)
    cax = None

    if ax is not None:
        divider = make_axes_locatable(ax)
        if cbar_orientation == "horizontal":
            cax = divider.append_axes("bottom", size="2.5%", pad=cbar_pad, axes_class=plt.Axes)
        else:
            cax = divider.append_axes("right", size="2.5%", pad=cbar_pad, axes_class=plt.Axes)

    if cax is not None and ax is not None:
        cb = ax.figure.colorbar(ax._gci(), cax=cax, ticks=cbar_ticks, format=cbar_format)
        cbar_axis = "y" if location in ("right", "left") else "x"
        cb.ax.tick_params(axis=cbar_axis, labelsize=fontsize["cbar_tick_label"])
        cb.set_label(label=cbar_unit, fontsize=fontsize["cbar_label"])
        ax.figure.sca(ax)
    else:
        cb = sp.draw_colorbar(
            ticks=cbar_ticks,
            format=cbar_format,
            fontsize=fontsize["cbar_tick_label"],
            location=location,
            shrink=cbar_shrink,
            pad=cbar_pad,
            ax=ax,
            cax=cax,
        )
        if cbar_format is not None:
            cb.minorformatter = mpl.ticker.FormatStrFormatter(cbar_format)
        cb.set_label(label=cbar_unit, fontsize=fontsize["cbar_label"])

    if hide_cbar:
        cb.remove()
        cb = None

    return cb


def _finish_plot(
    *,
    ax=None,  # noqa: ANN001
    projection: str = "cartesian",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: dict[str, float] | None = None,
    hide_title: bool = False,
    **kwargs,  # noqa: ARG001
) -> None:
    if fontsize is None:
        return
    if not hide_title and title is not None:
        title_pad = 25 if projection not in GLOBE_PROJECTIONS else 10
        ax.set_title(title, pad=title_pad, fontsize=fontsize["title"])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize["xlabel"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize["ylabel"])
