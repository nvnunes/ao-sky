"""Plot native `ao-sky` dense map artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any

from astropy.table import Table
import astropy.units as u
import h5py
from matplotlib import pyplot as plt
import numpy as np
import yaml

from ao_sky.build._constants import (
    BUILD_FILENAME,
    MAPS_DATASET,
    MAPS_FILENAME_TEMPLATE,
    RUNTIME_CONFIG_FILENAME,
)
from ao_sky.build.artifacts import read_maps_dataset
from ao_sky.spatial import get_pixel_area

from ._exceptions import PlottingError
from .fields import FieldConvention, get_field_convention, prepare_field_values
from .healpix import get_level, get_npix, get_pixel_skycoord, plot_healpix


_MAPS_FILENAME_RE = re.compile(r"^maps-hpx(?P<level>\d+)\.h5$")


@dataclass(frozen=True)
class MapPlotOptions:
    """Options for the artifact-aware map plotting layer."""

    galactic: bool = False
    projection: str = "astro"
    zoom: bool | None = None
    rotation: float | None = None
    width: float | None = 10
    height: float | None = None
    dpi: int | None = 120
    xsize: int | None = None
    grid: bool = True
    grid_longitude: str | None = None
    cmap: str | None = None
    norm: Any = None
    vmin: float | None = None
    vmax: float | None = None
    title: str | None = None
    cbar: bool = True
    cbar_orientation: str | None = None
    cbar_ticks: tuple[float, ...] | None = None
    cbar_format: str | None = None
    cbar_unit: str | None = None
    cbar_pad: float | None = None
    contour_norm: str | None = None
    contour_vmin: float | None = None
    contour_vmax: float | None = None
    contour_levels: tuple[float, ...] | None = None
    contour_filled: bool = False
    contour_colors: str | tuple[float, ...] | None = None
    contour_alpha: float | None = None
    boundaries_level: int | None = None
    boundaries_pixs: tuple[int, ...] | None = None
    milkyway: bool = False
    milkyway_width: float | None = None
    ecliptic: bool = False
    ecliptic_width: float | None = None
    surveys: Any = None
    points: Any = None
    colors: dict[str, Any] | None = None
    alphas: dict[str, Any] | None = None
    fontsize: dict[str, Any] | None = None
    hide_title: bool = False
    hide_cbar: bool = False


@dataclass(frozen=True)
class MapLayer:
    """A selected field from one dense map artifact."""

    filename: Path
    level: int
    field: str
    values: np.ndarray
    pixs: np.ndarray
    table: Table
    convention: FieldConvention


def infer_maps_level(filename: Path, table: Table | None = None) -> int:
    """Infer the HEALPix level represented by a maps artifact."""

    match = _MAPS_FILENAME_RE.match(filename.name)
    if match:
        return int(match.group("level"))
    if table is not None:
        level = get_level(len(table))
        if level is not None:
            return level
    raise PlottingError(f"Unable to infer maps artifact level from {filename}")


def maps_artifact_path(source: Path, *, level: int | None = None) -> Path:
    """Resolve a single maps artifact from a file or artifact directory."""

    source = Path(source)
    if source.is_file():
        return source
    if not source.is_dir():
        raise PlottingError(f"Map artifact source does not exist: {source}")

    if level is not None:
        filename = source / MAPS_FILENAME_TEMPLATE.format(level=int(level))
        if not filename.is_file():
            raise PlottingError(f"Maps artifact not found for level {level}: {filename}")
        return filename

    candidates: list[tuple[int, Path]] = []
    for filename in source.glob("maps-hpx*.h5"):
        match = _MAPS_FILENAME_RE.match(filename.name)
        if match:
            candidates.append((int(match.group("level")), filename))
    if not candidates:
        raise PlottingError(f"No maps-hpx*.h5 artifacts found in {source}")
    return max(candidates, key=lambda item: item[0])[1]


def read_map_layer(
    source: Path,
    *,
    field: str,
    level: int | None = None,
    field_convention: FieldConvention | None = None,
    field_overrides: dict[str, Any] | None = None,
) -> MapLayer:
    """Read one native map field from an artifact path or artifact directory."""

    filename = maps_artifact_path(source, level=level)
    table = read_maps_dataset(filename, MAPS_DATASET)
    artifact_level = infer_maps_level(filename, table)

    if "pix" not in table.colnames:
        raise PlottingError(f"Maps artifact is missing required 'pix' column: {filename}")

    expected_npix = get_npix(artifact_level)
    if len(table) != expected_npix:
        raise PlottingError(
            f"Maps artifact row count {len(table)} does not match level {artifact_level} "
            f"pixel count {expected_npix}: {filename}"
        )

    pixs = np.asarray(table["pix"], dtype=np.int64)
    raw_values = _map_field_values(table, field=field, level=artifact_level, filename=filename)
    if field_convention is None:
        field_convention = get_field_convention(field, **(field_overrides or {}))
    values = prepare_field_values(raw_values, field_convention)
    return MapLayer(
        filename=filename,
        level=artifact_level,
        field=field,
        values=values,
        pixs=pixs,
        table=table,
        convention=field_convention,
    )


def _map_field_values(
    table: Table,
    *,
    field: str,
    level: int,
    filename: Path,
) -> np.ndarray:
    if field == "coverage_mean":
        if "coverage_averaged" not in table.colnames:
            raise PlottingError(f"Map field 'coverage_mean' requires 'coverage_averaged' in {filename}")
        return np.asarray(table["coverage_averaged"])
    if field in table.colnames:
        return np.asarray(table[field])
    if field == "stellar_density":
        if "star_count" not in table.colnames:
            raise PlottingError(f"Map field 'stellar_density' requires 'star_count' in {filename}")
        area_arcmin2 = get_pixel_area(level).to_value(u.arcmin**2)
        return np.asarray(table["star_count"], dtype=np.float64) / area_arcmin2
    if field == "ngs_density":
        if "ngs_count" not in table.colnames:
            raise PlottingError(f"Map field 'ngs_density' requires 'ngs_count' in {filename}")
        area_arcmin2 = get_pixel_area(level).to_value(u.arcmin**2)
        return np.asarray(table["ngs_count"], dtype=np.float64) / area_arcmin2
    raise PlottingError(f"Map field '{field}' not found in {filename}")


def plot_map_layer(
    layer: MapLayer,
    *,
    options: MapPlotOptions | None = None,
    contour_layer: MapLayer | None = None,
    output_path: Path | None = None,
):
    """Plot a selected map layer and optionally save it to disk."""

    if options is None:
        options = MapPlotOptions()
    options = _merge_field_options(options, layer.convention)

    pixs = None
    skycoords = None
    if not np.array_equal(layer.pixs, np.arange(get_npix(layer.level), dtype=np.int64)):
        pixs = layer.pixs
        skycoords = get_pixel_skycoord(layer.level, pixs)

    projection, zoom, rotation, grid_longitude = _resolve_projection(
        level=layer.level,
        projection=options.projection,
        zoom=options.zoom,
        rotation=options.rotation,
        galactic=options.galactic,
        pixs=pixs,
        skycoords=skycoords,
    )

    plot_properties: dict[str, object] = {
        "galactic": options.galactic,
        "projection": projection,
        "zoom": zoom,
        "rotation": rotation,
        "xsize": options.xsize,
        "width": options.width,
        "height": options.height,
        "dpi": options.dpi,
        "cmap": options.cmap,
        "norm": options.norm,
        "vmin": options.vmin,
        "vmax": options.vmax,
        "grid": options.grid,
        "grid_longitude": options.grid_longitude or grid_longitude,
        "cbar": options.cbar,
        "cbar_orientation": options.cbar_orientation,
        "cbar_ticks": options.cbar_ticks,
        "cbar_format": options.cbar_format,
        "cbar_unit": options.cbar_unit,
        "cbar_pad": options.cbar_pad,
        "contour_norm": options.contour_norm,
        "contour_vmin": options.contour_vmin,
        "contour_vmax": options.contour_vmax,
        "contour_levels": options.contour_levels,
        "contour_filled": options.contour_filled,
        "contour_colors": options.contour_colors,
        "contour_alpha": options.contour_alpha,
        "boundaries_level": options.boundaries_level,
        "boundaries_pixs": options.boundaries_pixs,
        "title": options.title,
        "xlabel": "GLON" if options.galactic else "RA",
        "ylabel": "GLAT" if options.galactic else "DEC",
        "milkyway": options.milkyway,
        "milkyway_width": options.milkyway_width,
        "ecliptic": options.ecliptic,
        "ecliptic_width": options.ecliptic_width,
        "surveys": options.surveys,
        "points": options.points,
        "colors": options.colors,
        "alphas": options.alphas,
        "fontsize": options.fontsize,
        "hide_title": options.hide_title,
        "hide_cbar": options.hide_cbar,
    }
    plot_properties = {key: value for key, value in plot_properties.items() if value is not None}

    fig = plot_healpix(
        layer.values,
        level=layer.level,
        pixs=pixs,
        skycoords=skycoords,
        contour_values=None if contour_layer is None else contour_layer.values,
        plot_properties=plot_properties,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_map_artifact_field(
    source: Path,
    *,
    field: str,
    level: int | None = None,
    options: MapPlotOptions | None = None,
    output_path: Path | None = None,
    field_overrides: dict[str, Any] | None = None,
    contour_field: str | None = None,
    contour_field_overrides: dict[str, Any] | None = None,
):
    """Read and plot one field from a native maps artifact source."""

    layer = read_map_layer(
        source,
        field=field,
        level=level,
        field_overrides=field_overrides,
    )
    if layer.convention.title_uses_prediction_wavelength:
        options = _with_prediction_wavelength_title(
            source,
            options=options,
            convention=layer.convention,
        )
    contour_layer = None
    if contour_field is not None:
        contour_layer = read_map_layer(
            source,
            field=contour_field,
            level=layer.level,
            field_overrides=contour_field_overrides,
        )
    return plot_map_layer(
        layer,
        options=options,
        contour_layer=contour_layer,
        output_path=output_path,
    )


def _with_prediction_wavelength_title(
    source: Path,
    *,
    options: MapPlotOptions | None,
    convention: FieldConvention,
) -> MapPlotOptions | None:
    if options is not None and options.title is not None:
        return options
    wavelength_micron = _read_prediction_wavelength_micron(source)
    if wavelength_micron is None:
        return options
    options = MapPlotOptions() if options is None else options
    return replace(
        options,
        title=_format_prediction_wavelength_title(convention.title, wavelength_micron),
    )


def _read_prediction_wavelength_micron(source: Path) -> float | None:
    build_root = _build_root_for_map_source(source)
    runtime_config_path = build_root / RUNTIME_CONFIG_FILENAME
    if runtime_config_path.is_file():
        return _prediction_wavelength_from_yaml(
            runtime_config_path.read_text(encoding="utf-8"),
            source=runtime_config_path,
        )

    build_metadata_path = build_root / BUILD_FILENAME
    if build_metadata_path.is_file():
        with h5py.File(build_metadata_path, "r") as handle:
            if "metadata/config_yaml" not in handle:
                return None
            return _prediction_wavelength_from_yaml(
                _decode_hdf5_scalar(handle["metadata/config_yaml"][()]),
                source=build_metadata_path,
            )
    return None


def _build_root_for_map_source(source: Path) -> Path:
    source = Path(source)
    if source.is_file():
        return source.parent
    return source


def _decode_hdf5_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _prediction_wavelength_from_yaml(raw_yaml: str, *, source: Path) -> float | None:
    payload = yaml.safe_load(raw_yaml) or {}
    if not isinstance(payload, dict):
        return None
    prediction = payload.get("prediction")
    if not isinstance(prediction, dict) or "wavelength_micron" not in prediction:
        return None
    try:
        wavelength_micron = float(prediction["wavelength_micron"])
    except (TypeError, ValueError) as exc:
        raise PlottingError(
            f"Runtime config prediction.wavelength_micron in {source} must be numeric"
        ) from exc
    if wavelength_micron <= 0.0:
        raise PlottingError(
            f"Runtime config prediction.wavelength_micron in {source} must be positive"
        )
    return wavelength_micron


def _format_prediction_wavelength_title(title: str, wavelength_micron: float) -> str:
    return f"{title} [$\\lambda = {wavelength_micron:.3f}\\,\\mu\\mathrm{{m}}$]"


def _merge_field_options(options: MapPlotOptions, convention: FieldConvention) -> MapPlotOptions:
    return replace(
        options,
        cmap=options.cmap if options.cmap is not None else convention.cmap,
        norm=options.norm if options.norm is not None else convention.norm,
        vmin=options.vmin if options.vmin is not None else convention.vmin,
        vmax=options.vmax if options.vmax is not None else convention.vmax,
        title=options.title if options.title is not None else convention.title,
        cbar_ticks=options.cbar_ticks if options.cbar_ticks is not None else convention.cbar_ticks,
        cbar_format=options.cbar_format if options.cbar_format is not None else convention.cbar_format,
        cbar_unit=options.cbar_unit if options.cbar_unit is not None else convention.unit,
    )


def _resolve_projection(
    *,
    level: int,
    projection: str,
    zoom: bool | None,
    rotation: float | None,
    galactic: bool,
    pixs: np.ndarray | None,
    skycoords,
) -> tuple[str, bool, float | None, str]:
    projection_words = projection.lower().split()

    if zoom is None:
        if pixs is not None and len(pixs) < get_npix(level):
            zoom = skycoords is not None and (max(skycoords.ra.degree) - min(skycoords.ra.degree)) < 180
            if zoom:
                rotation = (rotation if rotation is not None else 0) + float(np.mean(skycoords.ra.degree))
        else:
            zoom = False

    if "astro" in projection_words:
        includes_poles = skycoords is not None and np.any(np.abs(skycoords.dec.degree) > 89)
        if zoom and not includes_poles:
            projection = "gnomonic"
        else:
            projection = "mollweide"
    elif "cart" in projection_words or "cartesian" in projection_words:
        projection = "cartesian"
    else:
        projection = projection_words[0]

    if "hours" in projection_words:
        grid_longitude = "hours"
    elif "degrees" in projection_words:
        grid_longitude = "degrees"
    elif "longitude" in projection_words:
        grid_longitude = "longitude"
    elif galactic:
        grid_longitude = "degrees"
    else:
        grid_longitude = "hours"

    return projection, bool(zoom), rotation, grid_longitude
