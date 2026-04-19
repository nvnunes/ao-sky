#!/usr/bin/env python3
"""Plot raw best-EE maps from two saved outer-pixel artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ao_sky.spatial import get_pixel_skycoord  # noqa: E402


LEVEL_PATH_RE = re.compile(r"^hpx(?P<outer>\d+)-(?P<inner>\d+)$")
DEFAULT_GRID_SIZE = 300
DEFAULT_FIGURE_WIDTH = 14.2
DEFAULT_FIGURE_HEIGHT = 4.6
DEFAULT_DPI = 120
DEFAULT_EE_CMAP = "plasma"
DEFAULT_EE_VMIN = 0.0
DEFAULT_EE_VMAX = 0.60
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


@dataclass(frozen=True)
class OuterBestEe:
    """Raw best-EE samples from one saved outer-pixel artifact."""

    path: Path
    source_format: str
    schema_label: str
    pix: np.ndarray
    best_ee: np.ndarray


@dataclass(frozen=True)
class ComparisonGeometry:
    """Coordinate and interpolation grid for a shared outer-pixel comparison."""

    outer_pix: int
    outer_level: int
    inner_level: int
    x_deg: np.ndarray
    y_deg: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    extent: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    """Return command-line arguments for the saved-artifact comparison plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare raw best_ee maps from two outer.h5 artifacts. The finite "
            "inner-pixel samples are projected to local tangent-plane offsets, "
            "interpolated through matplotlib.tri.LinearTriInterpolator on a "
            "regular grid, and plotted as left, right, and right-minus-left."
        )
    )
    parser.add_argument(
        "left_outer_h5",
        type=Path,
        help=(
            "First artifact input. Accepts an outer.h5 file, a true legacy "
            "inner.fits file, an outer-pixel artifact directory, or a root "
            "containing an hpx<outer>-<inner> tree."
        ),
    )
    parser.add_argument(
        "right_outer_h5",
        type=Path,
        help=(
            "Second artifact input. Accepts the same formats as the first "
            "input."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outer_best_ee_comparison.png"),
        help="Output PNG path. Defaults to ./outer_best_ee_comparison.png.",
    )
    parser.add_argument(
        "--left-label",
        default="Left",
        help="Panel label for the first artifact.",
    )
    parser.add_argument(
        "--right-label",
        default="Right",
        help="Panel label for the second artifact.",
    )
    parser.add_argument(
        "--outer-pix",
        type=int,
        default=None,
        help=(
            "Outer HEALPix pixel id. By default this is inferred from the "
            "artifact path."
        ),
    )
    parser.add_argument(
        "--outer-level",
        type=int,
        default=None,
        help=(
            "Outer nested HEALPix level. By default this is inferred from "
            "hpx<outer>-<inner>."
        ),
    )
    parser.add_argument(
        "--inner-level",
        type=int,
        default=None,
        help=(
            "Inner nested HEALPix level. By default this is inferred from "
            "hpx<outer>-<inner>."
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help=f"Regular interpolation grid side length. Defaults to {DEFAULT_GRID_SIZE}.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=DEFAULT_FIGURE_WIDTH,
        help=f"Figure width in inches. Defaults to {DEFAULT_FIGURE_WIDTH}.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT,
        help=f"Figure height in inches. Defaults to {DEFAULT_FIGURE_HEIGHT}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output image DPI. Defaults to {DEFAULT_DPI}.",
    )
    parser.add_argument(
        "--ee-low-percentile",
        type=float,
        default=1.0,
        help=(
            "Shared EE color-scale low percentile used with --auto-ee-scale. "
            "Defaults to 1."
        ),
    )
    parser.add_argument(
        "--ee-high-percentile",
        type=float,
        default=99.0,
        help=(
            "Shared EE color-scale high percentile used with --auto-ee-scale. "
            "Defaults to 99."
        ),
    )
    parser.add_argument(
        "--auto-ee-scale",
        action="store_true",
        help=(
            "Use percentile-derived EE color limits instead of the fixed "
            "default scale."
        ),
    )
    parser.add_argument(
        "--ee-vmin",
        type=float,
        default=DEFAULT_EE_VMIN,
        help=f"Fixed EE color-scale minimum. Defaults to {DEFAULT_EE_VMIN}.",
    )
    parser.add_argument(
        "--ee-vmax",
        type=float,
        default=DEFAULT_EE_VMAX,
        help=f"Fixed EE color-scale maximum. Defaults to {DEFAULT_EE_VMAX}.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_EE_CMAP,
        help=f"Matplotlib colormap for EE panels. Defaults to {DEFAULT_EE_CMAP}.",
    )
    parser.add_argument(
        "--diff-percentile",
        type=float,
        default=99.0,
        help="Symmetric absolute-difference color-scale percentile. Defaults to 99.",
    )
    parser.add_argument(
        "--ao-system",
        default="GNAO",
        help=(
            "AO system name used to choose the true legacy FITS EE column. "
            "Defaults to GNAO."
        ),
    )
    parser.add_argument(
        "--legacy-ee-column",
        default=None,
        help=(
            "Explicit true legacy FITS EE column. Defaults to "
            "ASTERISM_EE_MAX_<AO_SYSTEM>."
        ),
    )
    return parser.parse_args()


def read_outer_h5_best_ee(path: Path) -> OuterBestEe:
    """Read and sort the raw best-EE field from one v1/v2 outer.h5 artifact."""

    with h5py.File(path, "r") as handle:
        if "inner" not in handle:
            raise KeyError(f"{path} does not contain an 'inner' dataset")
        inner = handle["inner"][:]
    names = inner.dtype.names or ()
    missing = [name for name in ("pix", "best_ee") if name not in names]
    if missing:
        raise KeyError(
            f"{path} is missing inner column(s): {', '.join(missing)}. "
            "Both v1 and v2 traversal artifacts must expose inner.pix and "
            "inner.best_ee for this comparison."
        )

    pix = np.asarray(inner["pix"], dtype=np.int64)
    best_ee = np.asarray(inner["best_ee"], dtype=np.float64)
    order = np.argsort(pix)
    schema_version = detect_inner_schema_version(names)
    return OuterBestEe(
        path=path,
        source_format="outer.h5",
        schema_label=f"schema_v{schema_version}" if schema_version is not None else "unknown",
        pix=pix[order],
        best_ee=best_ee[order],
    )


def read_true_legacy_best_ee(path: Path, column_name: str) -> OuterBestEe:
    """Read and sort the raw best-EE field from one true legacy inner.fits file."""

    with fits.open(path, memmap=True) as handle:
        if len(handle) < 2:
            raise KeyError(f"{path} does not contain a FITS table extension")
        data = handle[1].data
        names = tuple(data.names or ())
        missing = [name for name in ("PIX", column_name) if name not in names]
        if missing:
            raise KeyError(f"{path} is missing FITS column(s): {', '.join(missing)}")
        pix = np.asarray(data["PIX"], dtype=np.int64).copy()
        best_ee = np.asarray(data[column_name], dtype=np.float64).copy()

    order = np.argsort(pix)
    return OuterBestEe(
        path=path,
        source_format="inner.fits",
        schema_label="true_legacy_fits",
        pix=pix[order],
        best_ee=best_ee[order],
    )


def detect_inner_schema_version(names: tuple[str, ...]) -> int | None:
    """Infer the traversal artifact schema version from inner column names."""

    name_set = set(names)
    legacy_columns = {"asterism_count", "winner_distance_arcsec"}
    if legacy_columns.issubset(name_set):
        return 1
    if "best_ee" in name_set and not legacy_columns.intersection(name_set):
        return 2
    return None


def legacy_ee_column(ao_system: str, explicit_column: str | None) -> str:
    """Return the true legacy FITS column used for the raw best-EE field."""

    if explicit_column:
        return explicit_column
    system_key = ao_system.replace("-", "_").upper()
    return f"ASTERISM_EE_MAX_{system_key}"


def infer_levels(path: Path) -> tuple[int, int] | None:
    """Return ``(outer_level, inner_level)`` inferred from an hpx path segment."""

    if LEVEL_PATH_RE.match(path.name):
        match = LEVEL_PATH_RE.match(path.name)
        if match:
            return int(match.group("outer")), int(match.group("inner"))
    for parent in path.resolve().parents:
        match = LEVEL_PATH_RE.match(parent.name)
        if match:
            return int(match.group("outer")), int(match.group("inner"))
    if path.is_dir():
        matches = []
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            match = LEVEL_PATH_RE.match(child.name)
            if match:
                matches.append(match)
        if len(matches) == 1:
            match = matches[0]
            return int(match.group("outer")), int(match.group("inner"))
    return None


def infer_outer_pix(path: Path) -> int | None:
    """Return the outer pixel id inferred from a conventional artifact path."""

    try:
        return int(path.resolve().parent.name)
    except ValueError:
        return None


def resolve_levels(
    left_path: Path,
    right_path: Path,
    outer_level: int | None,
    inner_level: int | None,
) -> tuple[int, int]:
    """Resolve HEALPix levels from CLI overrides or artifact paths."""

    if outer_level is not None and inner_level is not None:
        return int(outer_level), int(inner_level)

    inferred = [
        value
        for value in (infer_levels(left_path), infer_levels(right_path))
        if value is not None
    ]
    if not inferred:
        raise ValueError(
            "could not infer HEALPix levels from artifact paths; pass "
            "--outer-level and --inner-level"
        )
    if any(value != inferred[0] for value in inferred):
        raise ValueError(f"artifact paths disagree on hpx levels: {inferred}")

    resolved_outer, resolved_inner = inferred[0]
    if outer_level is not None:
        resolved_outer = int(outer_level)
    if inner_level is not None:
        resolved_inner = int(inner_level)
    return resolved_outer, resolved_inner


def resolve_outer_pix(left_path: Path, right_path: Path, outer_pix: int | None) -> int:
    """Resolve the compared outer pixel from a CLI override or artifact paths."""

    if outer_pix is not None:
        return int(outer_pix)

    inferred = [
        value
        for value in (infer_outer_pix(left_path), infer_outer_pix(right_path))
        if value is not None
    ]
    if not inferred:
        raise ValueError("could not infer outer pixel from artifact paths; pass --outer-pix")
    if any(value != inferred[0] for value in inferred):
        raise ValueError(f"artifact paths disagree on outer pixel: {inferred}")
    return inferred[0]


def hpx_tree_root(path: Path, outer_level: int, inner_level: int) -> Path:
    """Return the directory that owns the requested hpx tree."""

    hpx_name = f"hpx{outer_level}-{inner_level}"
    if path.name == hpx_name:
        return path
    candidate = path / hpx_name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"{path} does not contain {hpx_name}")


def bucket_parts(outer_pix: int, outer_level: int) -> tuple[str, str]:
    """Return legacy/build hour and declination bucket path components."""

    coord = get_pixel_skycoord(outer_level, outer_pix)
    hour = int(np.floor(coord.ra.degree / 15.0))
    deg = int(np.floor(abs(coord.dec.degree / 10.0)) * 10)
    if outer_pix in SPECIAL_HOUR_PIXELS:
        hour = 14
    sign = "+" if coord.dec.degree >= 0 else "-"
    return f"{hour}h", f"{sign}{deg:02}"


def resolve_artifact_input(
    path: Path,
    *,
    outer_pix: int,
    outer_level: int,
    inner_level: int,
) -> Path:
    """Resolve a file, pixel directory, or hpx-root input to a concrete artifact."""

    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"artifact input does not exist: {path}")

    direct_h5 = path / "outer.h5"
    direct_fits = path / "inner.fits"
    if direct_h5.is_file() and direct_fits.is_file():
        raise ValueError(f"artifact directory contains both outer.h5 and inner.fits: {path}")
    if direct_h5.is_file():
        return direct_h5
    if direct_fits.is_file():
        return direct_fits

    hour, dec_bucket = bucket_parts(outer_pix, outer_level)
    root = hpx_tree_root(path, outer_level, inner_level)
    pixel_dir = root / hour / dec_bucket / str(outer_pix)
    h5_candidate = pixel_dir / "outer.h5"
    fits_candidate = pixel_dir / "inner.fits"
    if h5_candidate.is_file() and fits_candidate.is_file():
        raise ValueError(f"artifact tree contains both outer.h5 and inner.fits: {pixel_dir}")
    if h5_candidate.is_file():
        return h5_candidate
    if fits_candidate.is_file():
        return fits_candidate
    raise FileNotFoundError(f"no outer.h5 or inner.fits found under {pixel_dir}")


def read_best_ee_input(path: Path, *, legacy_column: str) -> OuterBestEe:
    """Read one resolved artifact input in either supported persisted format."""

    if path.name == "outer.h5":
        return read_outer_h5_best_ee(path)
    if path.name == "inner.fits":
        return read_true_legacy_best_ee(path, legacy_column)
    raise ValueError(f"unsupported artifact file name: {path}")


def align_inner_pixels(
    left: OuterBestEe,
    right: OuterBestEe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared inner-pixel ids and the corresponding left/right EE values."""

    common, left_idx, right_idx = np.intersect1d(left.pix, right.pix, return_indices=True)
    if len(common) == 0:
        raise ValueError("artifacts do not share any inner pixels")
    return common, left.best_ee[left_idx], right.best_ee[right_idx]


def build_geometry(
    inner_pixs: np.ndarray,
    *,
    outer_pix: int,
    outer_level: int,
    inner_level: int,
    grid_size: int,
) -> ComparisonGeometry:
    """Build local tangent-plane offsets and a regular interpolation grid."""

    if grid_size < 3:
        raise ValueError("grid size must be at least 3")

    center = get_pixel_skycoord(outer_level, outer_pix)
    coords = get_pixel_skycoord(inner_level, inner_pixs)
    dlon, dlat = center.spherical_offsets_to(coords)
    x_deg = np.asarray(dlon.deg, dtype=np.float64)
    y_deg = np.asarray(dlat.deg, dtype=np.float64)

    x_span = float(np.nanmax(x_deg) - np.nanmin(x_deg))
    y_span = float(np.nanmax(y_deg) - np.nanmin(y_deg))
    x_pad = 0.02 * x_span if x_span > 0 else 0.01
    y_pad = 0.02 * y_span if y_span > 0 else 0.01
    xi = np.linspace(
        float(np.nanmin(x_deg) - x_pad),
        float(np.nanmax(x_deg) + x_pad),
        grid_size,
    )
    yi = np.linspace(
        float(np.nanmin(y_deg) - y_pad),
        float(np.nanmax(y_deg) + y_pad),
        grid_size,
    )
    grid_x, grid_y = np.meshgrid(xi, yi)
    return ComparisonGeometry(
        outer_pix=outer_pix,
        outer_level=outer_level,
        inner_level=inner_level,
        x_deg=x_deg,
        y_deg=y_deg,
        grid_x=grid_x,
        grid_y=grid_y,
        extent=(float(xi.min()), float(xi.max()), float(yi.min()), float(yi.max())),
    )


def interpolate_to_grid(geometry: ComparisonGeometry, values: np.ndarray) -> np.ma.MaskedArray:
    """Interpolate finite irregular samples onto the comparison grid."""

    import matplotlib.tri as mtri

    finite = np.isfinite(values)
    if int(np.count_nonzero(finite)) < 3:
        raise ValueError(
            "at least three finite best_ee samples are required for triangulation"
        )
    triangulation = mtri.Triangulation(geometry.x_deg[finite], geometry.y_deg[finite])
    interpolator = mtri.LinearTriInterpolator(triangulation, values[finite])
    return np.ma.masked_invalid(interpolator(geometry.grid_x, geometry.grid_y))


def percentile_limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Return robust plotting limits for finite values."""

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("no finite values available for color scaling")
    vmin = float(np.nanpercentile(finite, low))
    vmax = float(np.nanpercentile(finite, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def symmetric_abs_limit(values: np.ndarray, percentile: float) -> float:
    """Return a symmetric plotting limit for finite signed differences."""

    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit == 0.0:
        limit = float(np.nanmax(np.abs(finite)))
    return limit if limit > 0.0 else 1e-12


def summary_key(label: str) -> str:
    """Return a stable lowercase key fragment from a free-form plot label."""

    value = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    return value or "artifact"


def make_plot(
    geometry: ComparisonGeometry,
    left_image: np.ma.MaskedArray,
    right_image: np.ma.MaskedArray,
    *,
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_label: str,
    right_label: str,
    output: Path,
    ee_low_percentile: float,
    ee_high_percentile: float,
    auto_ee_scale: bool,
    ee_vmin: float,
    ee_vmax: float,
    cmap: str,
    diff_percentile: float,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    """Write the three-panel raw best-EE comparison plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if auto_ee_scale:
        combined = np.concatenate(
            [
                left_values[np.isfinite(left_values)],
                right_values[np.isfinite(right_values)],
            ]
        )
        ee_vmin, ee_vmax = percentile_limits(
            combined,
            ee_low_percentile,
            ee_high_percentile,
        )
    elif ee_vmax <= ee_vmin:
        raise ValueError("--ee-vmax must be greater than --ee-vmin")

    diff_image = np.ma.masked_invalid(np.ma.asarray(right_image) - np.ma.asarray(left_image))
    diff_values = np.asarray(diff_image.compressed(), dtype=np.float64)
    diff_limit = symmetric_abs_limit(diff_values, diff_percentile)

    if figure_width <= 0.0 or figure_height <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    image_kwargs = {
        "origin": "lower",
        "extent": geometry.extent,
        "interpolation": "bilinear",
    }

    left_artist = axes[0].imshow(
        left_image,
        cmap=cmap,
        vmin=ee_vmin,
        vmax=ee_vmax,
        **image_kwargs,
    )
    axes[1].imshow(
        right_image,
        cmap=cmap,
        vmin=ee_vmin,
        vmax=ee_vmax,
        **image_kwargs,
    )
    for axis, title in zip(axes[:2], (left_label, right_label), strict=True):
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$\Delta\mathrm{RA}\cos\delta$ [deg]")
    axes[0].set_ylabel(r"$\Delta\mathrm{Dec.}$ [deg]")
    fig.colorbar(left_artist, ax=axes[:2], shrink=0.86, label="raw best EE")

    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)
    diff_artist = axes[2].imshow(
        diff_image,
        cmap="coolwarm",
        norm=diff_norm,
        **image_kwargs,
    )
    axes[2].set_title(f"{right_label} - {left_label}")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_xlabel(r"$\Delta\mathrm{RA}\cos\delta$ [deg]")
    fig.colorbar(diff_artist, ax=axes[2], shrink=0.86, label=r"$\Delta$ raw best EE")

    fig.suptitle(
        f"Outer HEALPix {geometry.outer_pix}: raw best EE at inner-pixel centers",
        y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    *,
    output: Path,
    left: OuterBestEe,
    right: OuterBestEe,
    common_pixs: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_label: str,
    right_label: str,
    left_image: np.ma.MaskedArray,
    right_image: np.ma.MaskedArray,
) -> None:
    """Print a compact numeric summary for reproducible visual checks."""

    finite_left = np.isfinite(left_values)
    finite_right = np.isfinite(right_values)
    finite_both = finite_left & finite_right
    diff_image = np.ma.masked_invalid(np.ma.asarray(right_image) - np.ma.asarray(left_image))
    diff_values = np.asarray(diff_image.compressed(), dtype=np.float64)
    left_key = summary_key(left_label)
    right_key = summary_key(right_label)

    print(f"output={output}")
    print(f"left_source={left.source_format}")
    print(f"right_source={right.source_format}")
    print(f"left_schema={left.schema_label}")
    print(f"right_schema={right.schema_label}")
    print(f"left_path={left.path}")
    print(f"right_path={right.path}")
    print(f"common_inner_pixels={len(common_pixs)}")
    print(f"finite_{left_key}={int(np.count_nonzero(finite_left))}")
    print(f"finite_{right_key}={int(np.count_nonzero(finite_right))}")
    print(f"finite_both={int(np.count_nonzero(finite_both))}")
    if np.any(finite_left):
        print(
            f"{left_key}_best_ee_range="
            f"{float(np.nanmin(left_values[finite_left])):.12g},"
            f"{float(np.nanmax(left_values[finite_left])):.12g}"
        )
    if np.any(finite_right):
        print(
            f"{right_key}_best_ee_range="
            f"{float(np.nanmin(right_values[finite_right])):.12g},"
            f"{float(np.nanmax(right_values[finite_right])):.12g}"
        )
    if len(diff_values):
        print(
            "diff_grid_median_p05_p95_min_max="
            f"{float(np.nanmedian(diff_values)):.12g},"
            f"{float(np.nanpercentile(diff_values, 5)):.12g},"
            f"{float(np.nanpercentile(diff_values, 95)):.12g},"
            f"{float(np.nanmin(diff_values)):.12g},"
            f"{float(np.nanmax(diff_values)):.12g}"
        )


def main() -> int:
    """Run the saved-artifact raw best-EE comparison plotter."""

    args = parse_args()
    outer_level, inner_level = resolve_levels(
        args.left_outer_h5,
        args.right_outer_h5,
        args.outer_level,
        args.inner_level,
    )
    outer_pix = resolve_outer_pix(args.left_outer_h5, args.right_outer_h5, args.outer_pix)
    left_path = resolve_artifact_input(
        args.left_outer_h5,
        outer_pix=outer_pix,
        outer_level=outer_level,
        inner_level=inner_level,
    )
    right_path = resolve_artifact_input(
        args.right_outer_h5,
        outer_pix=outer_pix,
        outer_level=outer_level,
        inner_level=inner_level,
    )
    column_name = legacy_ee_column(args.ao_system, args.legacy_ee_column)
    left = read_best_ee_input(left_path, legacy_column=column_name)
    right = read_best_ee_input(right_path, legacy_column=column_name)
    common_pixs, left_values, right_values = align_inner_pixels(left, right)
    geometry = build_geometry(
        common_pixs,
        outer_pix=outer_pix,
        outer_level=outer_level,
        inner_level=inner_level,
        grid_size=args.grid_size,
    )
    left_image = interpolate_to_grid(geometry, left_values)
    right_image = interpolate_to_grid(geometry, right_values)
    make_plot(
        geometry,
        left_image,
        right_image,
        left_values=left_values,
        right_values=right_values,
        left_label=args.left_label,
        right_label=args.right_label,
        output=args.output,
        ee_low_percentile=args.ee_low_percentile,
        ee_high_percentile=args.ee_high_percentile,
        auto_ee_scale=args.auto_ee_scale,
        ee_vmin=args.ee_vmin,
        ee_vmax=args.ee_vmax,
        cmap=args.cmap,
        diff_percentile=args.diff_percentile,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
    )
    print_summary(
        output=args.output,
        left=left,
        right=right,
        common_pixs=common_pixs,
        left_values=left_values,
        right_values=right_values,
        left_label=args.left_label,
        right_label=args.right_label,
        left_image=left_image,
        right_image=right_image,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
