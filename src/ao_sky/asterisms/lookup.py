"""Read-only lookup of retained winner asterisms from build artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.table import Table

from ..build._constants import ASTERISMS_DTYPE, WORK_STATUS_DONE
from ..build.artifacts import read_outer_products
from ..build.control import load_build_definition, load_state, outer_artifact_filename
from ..spatial import get_pixel_skycoord
from ._exceptions import AsterismError

ASTERISM_LOOKUP_MEAN_FIELDS: tuple[str, ...] = (
    "on_axis_winner_ee",
    "field_averaged_winner_ee",
    "gaia_A0",
)
ASTERISM_LOOKUP_COLUMNS: tuple[str, ...] = (
    *(ASTERISMS_DTYPE.names or ()),
    "global_asterism_id",
    "representative_outer_pix",
    "representative_asterism_id",
    "inner_pixel_count",
    *ASTERISM_LOOKUP_MEAN_FIELDS,
)


@dataclass(slots=True)
class _LookupAccumulator:
    representative: dict[str, object]
    representative_outer_pix: int
    representative_asterism_id: int
    inner_pixel_count: int = 0
    mean_values: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _LookupObservation:
    outer_pix: int
    local_asterism_id: int
    key: tuple[int, ...]
    representative: dict[str, object]
    support: Table


@dataclass(frozen=True, slots=True)
class AsterismLookupFilters:
    """Filters for retained-winner asterism lookup.

    Magnitude filters apply to all real member slots. Empty member slots with
    source id ``-1`` are ignored. Inner-pixel metric filters apply to lookup
    output means over pixels where the returned asterism is the retained winner.

    Attributes:
        min_num_stars: Inclusive lower bound on real member-star count.
        max_num_stars: Inclusive upper bound on real member-star count.
        min_member_mag: Inclusive lower bound applied to every real member
            magnitude.
        max_member_mag: Inclusive upper bound applied to every real member
            magnitude.
        min_inner_pixel_count: Inclusive lower bound on retained-winner support
            pixels.
        max_inner_pixel_count: Inclusive upper bound on retained-winner support
            pixels.
        min_on_axis_winner_ee: Inclusive lower bound on mean on-axis winner EE.
        max_on_axis_winner_ee: Inclusive upper bound on mean on-axis winner EE.
        min_field_averaged_winner_ee: Inclusive lower bound on mean
            field-averaged winner EE.
        max_field_averaged_winner_ee: Inclusive upper bound on mean
            field-averaged winner EE.
        min_gaia_A0: Inclusive lower bound on mean Gaia A0 extinction.
        max_gaia_A0: Inclusive upper bound on mean Gaia A0 extinction.
    """

    min_num_stars: int | None = None
    max_num_stars: int | None = None
    min_member_mag: float | None = None
    max_member_mag: float | None = None
    min_inner_pixel_count: int | None = None
    max_inner_pixel_count: int | None = None
    min_on_axis_winner_ee: float | None = None
    max_on_axis_winner_ee: float | None = None
    min_field_averaged_winner_ee: float | None = None
    max_field_averaged_winner_ee: float | None = None
    min_gaia_A0: float | None = None
    max_gaia_A0: float | None = None

    def __post_init__(self) -> None:
        if self.min_num_stars is not None and int(self.min_num_stars) < 1:
            raise AsterismError("min_num_stars must be at least 1")
        if self.max_num_stars is not None and int(self.max_num_stars) < 1:
            raise AsterismError("max_num_stars must be at least 1")
        if self.min_inner_pixel_count is not None and int(self.min_inner_pixel_count) < 1:
            raise AsterismError("min_inner_pixel_count must be at least 1")
        if self.max_inner_pixel_count is not None and int(self.max_inner_pixel_count) < 1:
            raise AsterismError("max_inner_pixel_count must be at least 1")
        if (
            self.min_num_stars is not None
            and self.max_num_stars is not None
            and int(self.min_num_stars) > int(self.max_num_stars)
        ):
            raise AsterismError("min_num_stars cannot exceed max_num_stars")
        if (
            self.min_inner_pixel_count is not None
            and self.max_inner_pixel_count is not None
            and int(self.min_inner_pixel_count) > int(self.max_inner_pixel_count)
        ):
            raise AsterismError("min_inner_pixel_count cannot exceed max_inner_pixel_count")
        if (
            self.min_member_mag is not None
            and self.max_member_mag is not None
            and float(self.min_member_mag) > float(self.max_member_mag)
        ):
            raise AsterismError("min_member_mag cannot exceed max_member_mag")
        _validate_float_bound_pair(
            self.min_on_axis_winner_ee,
            self.max_on_axis_winner_ee,
            "on_axis_winner_ee",
        )
        _validate_float_bound_pair(
            self.min_field_averaged_winner_ee,
            self.max_field_averaged_winner_ee,
            "field_averaged_winner_ee",
        )
        _validate_float_bound_pair(self.min_gaia_A0, self.max_gaia_A0, "gaia_A0")


def _validate_float_bound_pair(
    minimum: float | None,
    maximum: float | None,
    field_name: str,
) -> None:
    if minimum is not None and maximum is not None and float(minimum) > float(maximum):
        raise AsterismError(f"min_{field_name} cannot exceed max_{field_name}")


def _normalize_outer_pixels(outer_pixels: int | Iterable[int] | None) -> tuple[int, ...]:
    if outer_pixels is None:
        raise AsterismError("find_asterisms requires outer_pixels or moc_file")
    if isinstance(outer_pixels, (str, bytes)):
        raise AsterismError("outer_pixels must be an int or iterable of ints")
    if isinstance(outer_pixels, int):
        values = [outer_pixels]
    else:
        values = [int(pixel) for pixel in outer_pixels]
    if not values:
        raise AsterismError("outer_pixels must not be empty")

    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if int(value) < 0:
            raise AsterismError("outer_pixels must be non-negative")
        if int(value) in seen:
            continue
        seen.add(int(value))
        ordered.append(int(value))
    return tuple(ordered)


def _load_moc_outer_pixels(moc_file: Path, *, outer_level: int) -> tuple[object, tuple[int, ...]]:
    from mocpy import MOC

    moc = MOC.from_fits(str(moc_file))
    level_moc = moc.to_order(int(outer_level))
    outer_pixels = tuple(int(pixel) for pixel in np.asarray(level_moc.flatten(), dtype=np.int64))
    return moc, outer_pixels


def _real_member_source_ids(row: dict[str, object]) -> tuple[int, ...]:
    source_ids = [
        int(row[f"star{index}_source_id"])
        for index in (1, 2, 3)
        if int(row[f"star{index}_source_id"]) >= 0
    ]
    return tuple(sorted(source_ids))


def _row_to_dict(row) -> dict[str, object]:
    return {name: row[name] for name in ASTERISMS_DTYPE.names or ()}


def _real_member_magnitudes(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(row[f"star{index}_mag"])
            for index in (1, 2, 3)
            if int(row[f"star{index}_source_id"]) >= 0
        ],
        dtype=np.float64,
    )


def _passes_asterism_filters(
    row: dict[str, object],
    filters: AsterismLookupFilters | None,
) -> bool:
    if filters is None:
        return True
    num_stars = int(row["num_stars"])
    if filters.min_num_stars is not None and num_stars < int(filters.min_num_stars):
        return False
    if filters.max_num_stars is not None and num_stars > int(filters.max_num_stars):
        return False

    member_mags = _real_member_magnitudes(row)
    if filters.min_member_mag is not None and (
        len(member_mags) == 0
        or np.any(member_mags < float(filters.min_member_mag))
    ):
        return False
    if filters.max_member_mag is not None and (
        len(member_mags) == 0
        or np.any(member_mags > float(filters.max_member_mag))
    ):
        return False
    return True


def _passes_lookup_metric_filters(
    row: dict[str, object],
    filters: AsterismLookupFilters | None,
) -> bool:
    if filters is None:
        return True
    inner_pixel_count = int(row["inner_pixel_count"])
    if (
        filters.min_inner_pixel_count is not None
        and inner_pixel_count < int(filters.min_inner_pixel_count)
    ):
        return False
    if (
        filters.max_inner_pixel_count is not None
        and inner_pixel_count > int(filters.max_inner_pixel_count)
    ):
        return False
    return (
        _passes_float_bounds(
            row,
            "on_axis_winner_ee",
            filters.min_on_axis_winner_ee,
            filters.max_on_axis_winner_ee,
        )
        and _passes_float_bounds(
            row,
            "field_averaged_winner_ee",
            filters.min_field_averaged_winner_ee,
            filters.max_field_averaged_winner_ee,
        )
        and _passes_float_bounds(row, "gaia_A0", filters.min_gaia_A0, filters.max_gaia_A0)
    )


def _passes_float_bounds(
    row: dict[str, object],
    field_name: str,
    minimum: float | None,
    maximum: float | None,
) -> bool:
    if minimum is None and maximum is None:
        return True
    value = float(row[field_name])
    if not np.isfinite(value):
        return False
    if minimum is not None and value < float(minimum):
        return False
    if maximum is not None and value > float(maximum):
        return False
    return True


def _record_support(
    accumulator: _LookupAccumulator,
    *,
    support: Table,
) -> None:
    accumulator.inner_pixel_count += int(len(support))
    for field_name in ASTERISM_LOOKUP_MEAN_FIELDS:
        values = np.asarray(support[field_name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        accumulator.mean_values.setdefault(field_name, []).extend(float(value) for value in finite)


def _lookup_outer_pixel_asterisms(
    build_path: Path,
    outer_pixels: tuple[int, ...],
    *,
    inner_selector=None,
    filters: AsterismLookupFilters | None = None,
) -> list[dict[str, object]]:
    definition = load_build_definition(build_path)
    state = load_state(build_path)
    accumulators: dict[tuple[int, ...], _LookupAccumulator] = {}

    for observation in _iter_lookup_observations(
        build_path,
        outer_pixels,
        inner_selector=inner_selector,
        filters=filters,
        definition=definition,
        state=state,
    ):
        accumulator = accumulators.get(observation.key)
        if accumulator is None:
            accumulator = _LookupAccumulator(
                representative=observation.representative,
                representative_outer_pix=int(observation.outer_pix),
                representative_asterism_id=int(observation.local_asterism_id),
            )
            accumulators[observation.key] = accumulator
        _record_support(accumulator, support=observation.support)

    rows: list[dict[str, object]] = []
    for global_asterism_id, key in enumerate(sorted(accumulators), start=1):
        accumulator = accumulators[key]
        row = _accumulator_to_row(accumulator, global_asterism_id)
        if _passes_lookup_metric_filters(row, filters):
            rows.append(row)
    for global_asterism_id, row in enumerate(rows, start=1):
        row["global_asterism_id"] = int(global_asterism_id)
    return rows


def _iter_lookup_observations(
    build_path: Path,
    outer_pixels: tuple[int, ...],
    *,
    inner_selector=None,
    filters: AsterismLookupFilters | None = None,
    definition=None,
    state: Table | None = None,
):
    if definition is None:
        definition = load_build_definition(build_path)
    if state is None:
        state = load_state(build_path)
    max_outer_pix = len(state) - 1

    for outer_pix in outer_pixels:
        if outer_pix > max_outer_pix:
            raise AsterismError(
                f"outer pixel {outer_pix} is outside build outer level "
                f"{definition.outer_level}"
            )
        state_row = state[int(outer_pix)]
        if int(state_row["traversal_status"]) != WORK_STATUS_DONE:
            raise AsterismError(f"outer pixel {outer_pix} traversal is not done")

        filename = outer_artifact_filename(build_path, definition, outer_pix)
        if not filename.is_file():
            raise AsterismError(f"outer artifact is missing for pixel {outer_pix}: {filename}")

        inner, asterisms = read_outer_products(filename)
        if len(inner) == 0 or len(asterisms) == 0:
            continue
        if inner_selector is not None:
            selected = np.asarray(inner_selector(outer_pix, inner), dtype=np.bool_)
            if selected.shape != (len(inner),):
                raise AsterismError(
                    "inner selector must return one boolean per inner row "
                    f"for outer pixel {outer_pix}"
                )
            inner = inner[selected]
        if len(inner) == 0 or len(asterisms) == 0:
            continue
        winner_ids = np.asarray(inner["winner_asterism_id"], dtype=np.int64)
        referenced_ids = set(int(value) for value in winner_ids if int(value) >= 0)
        if not referenced_ids:
            continue

        for row in asterisms:
            local_id = int(row["asterism_id"])
            if local_id not in referenced_ids:
                continue
            support = inner[winner_ids == local_id]
            if len(support) == 0:
                continue

            representative = _row_to_dict(row)
            if not _passes_asterism_filters(representative, filters):
                continue
            key = _real_member_source_ids(representative)
            if not key:
                continue
            yield _LookupObservation(
                outer_pix=int(outer_pix),
                local_asterism_id=local_id,
                key=key,
                representative=representative,
                support=support,
            )


def _make_moc_inner_selector(moc, *, inner_level: int):
    def select_inner(_outer_pix: int, inner: Table) -> np.ndarray:
        inner_coords = get_pixel_skycoord(
            int(inner_level),
            np.asarray(inner["pix"], dtype=np.int64),
        )
        return np.asarray(moc.contains_skycoords(inner_coords), dtype=np.bool_)

    return select_inner


def _accumulator_to_row(
    accumulator: _LookupAccumulator,
    global_asterism_id: int,
) -> dict[str, object]:
    row = dict(accumulator.representative)
    row["global_asterism_id"] = int(global_asterism_id)
    row["representative_outer_pix"] = int(accumulator.representative_outer_pix)
    row["representative_asterism_id"] = int(accumulator.representative_asterism_id)
    row["inner_pixel_count"] = int(accumulator.inner_pixel_count)
    for field_name in ASTERISM_LOOKUP_MEAN_FIELDS:
        values = np.asarray(accumulator.mean_values.get(field_name, []), dtype=np.float64)
        row[field_name] = np.nan if len(values) == 0 else float(np.mean(values))
    return row


def _empty_lookup_table() -> Table:
    arrays: list[np.ndarray] = []
    for name in ASTERISM_LOOKUP_COLUMNS:
        if name in (ASTERISMS_DTYPE.names or ()):
            arrays.append(np.array([], dtype=ASTERISMS_DTYPE[name]))
        elif name in ASTERISM_LOOKUP_MEAN_FIELDS:
            arrays.append(np.array([], dtype=np.float64))
        else:
            arrays.append(np.array([], dtype=np.int64))
    return Table(arrays, names=ASTERISM_LOOKUP_COLUMNS)


def _rows_to_table(rows: list[dict[str, object]]) -> Table:
    if not rows:
        return _empty_lookup_table()
    columns = [
        np.asarray([row[name] for row in rows])
        for name in ASTERISM_LOOKUP_COLUMNS
    ]
    return Table(columns, names=ASTERISM_LOOKUP_COLUMNS)


def find_asterisms(
    build_path: Path | str,
    *,
    outer_pixels: int | Iterable[int] | None = None,
    moc_file: Path | str | None = None,
    filters: AsterismLookupFilters | None = None,
    max_rows: int | None = None,
) -> Table:
    """Return retained regularized winner asterisms from build artifacts.

    The lookup is strict and read-only: selected outer pixels must be marked
    done in ``build.h5`` and must have a readable ``outer.h5`` artifact. A MOC
    lookup selects support inner pixels by testing inner-pixel centers against
    the supplied MOC.
    """

    if max_rows is not None and int(max_rows) < 0:
        raise AsterismError("max_rows must be non-negative when provided")

    if outer_pixels is not None and moc_file is not None:
        raise AsterismError("find_asterisms accepts only one selector")
    resolved_build_path = Path(build_path)
    if moc_file is None:
        resolved_outer_pixels = _normalize_outer_pixels(outer_pixels)
        rows = _lookup_outer_pixel_asterisms(
            resolved_build_path,
            resolved_outer_pixels,
            filters=filters,
        )
    else:
        if outer_pixels is not None:
            raise AsterismError("outer_pixels cannot be combined with moc_file")
        definition = load_build_definition(resolved_build_path)
        moc, resolved_outer_pixels = _load_moc_outer_pixels(
            Path(moc_file),
            outer_level=definition.outer_level,
        )
        rows = _lookup_outer_pixel_asterisms(
            resolved_build_path,
            resolved_outer_pixels,
            inner_selector=_make_moc_inner_selector(moc, inner_level=definition.inner_level),
            filters=filters,
        )
    if max_rows is not None and len(rows) > int(max_rows):
        raise AsterismError(
            f"find_asterisms materialized {len(rows)} rows, exceeding max_rows={int(max_rows)}"
        )
    return _rows_to_table(rows)
