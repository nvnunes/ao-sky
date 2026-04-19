"""Regional Traversal planning helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..spatial import get_parent_pixel, get_pixel_neighbours, get_pixel_skycoord
from ._constants import WORK_STATUS_FAILED, WORK_STATUS_PENDING
from ._models import TraversalWorkerPlan

REGIONAL_SCHEDULE_SECONDS_COEFFICIENT = 6.8603409602539e-05
REGIONAL_SCHEDULE_SECONDS_STAR_EXPONENT = 1.2184647278722778
REGIONAL_SCHEDULE_SECONDS_DENSE_CAP = 18.58285714285714
REGIONAL_SCHEDULE_THROUGHPUT_SCALE = 0.5
GALACTIC_LATITUDE_LOW_BAND_DEG = 15.0


def build_regional_worker_plans(
    *,
    state: np.ndarray,
    outer_level: int,
    region_level: int,
    workers: int,
    status_field: str,
    star_counts: np.ndarray,
    low_latitude_deg: float = GALACTIC_LATITUDE_LOW_BAND_DEG,
    low_latitude_workers: int | None = None,
) -> tuple[TraversalWorkerPlan, ...]:
    """Return Galactic-latitude-aware Traversal plans for long-lived workers."""

    eligible = [
        int(row["outer_pix"])
        for row in state
        if int(row[status_field]) in (WORK_STATUS_PENDING, WORK_STATUS_FAILED)
    ]
    if not eligible:
        return ()

    worker_count = int(workers)
    if worker_count < 1:
        return ()

    low_latitude_worker_count = (
        max(1, int(np.ceil(worker_count * 2.0 / 3.0)))
        if low_latitude_workers is None
        else int(low_latitude_workers)
    )
    low_latitude_worker_count = min(
        max(low_latitude_worker_count, 1),
        worker_count,
    )
    region_for_outer = get_parent_pixel(
        outer_level,
        np.asarray(eligible, dtype=np.int64),
        region_level,
    )
    region_members: dict[int, list[int]] = defaultdict(list)
    for outer_pix, region_pix in zip(eligible, region_for_outer, strict=True):
        region_members[int(region_pix)].append(int(outer_pix))

    region_costs = {
        int(region_pix): _region_star_count(
            region_members[int(region_pix)],
            star_counts=star_counts,
        )
        for region_pix in region_members
    }
    region_peak_stars = {
        int(region_pix): _region_peak_star_count(
            region_members[int(region_pix)],
            star_counts=star_counts,
        )
        for region_pix in region_members
    }
    region_runtimes = {
        int(region_pix): _region_schedule_seconds(
            region_members[int(region_pix)],
            star_counts=star_counts,
        )
        for region_pix in region_members
    }
    region_abs_lat, region_lon = _region_galactic_centers(
        tuple(region_members),
        region_level=region_level,
    )
    low_lat_regions = [
        int(region_pix)
        for region_pix in region_members
        if region_abs_lat[int(region_pix)] <= float(low_latitude_deg)
    ]
    high_lat_regions = [
        int(region_pix)
        for region_pix in region_members
        if region_abs_lat[int(region_pix)] > float(low_latitude_deg)
    ]
    if low_lat_regions and high_lat_regions:
        low_latitude_worker_ids = list(range(low_latitude_worker_count))
        high_latitude_worker_ids = list(range(low_latitude_worker_count, worker_count))
    elif low_lat_regions:
        low_latitude_worker_ids = list(range(worker_count))
        high_latitude_worker_ids = []
    else:
        low_latitude_worker_ids = []
        high_latitude_worker_ids = list(range(worker_count))

    assignments: list[list[int]] = [[] for _ in range(worker_count)]
    _assign_regions_by_runtime(
        low_lat_regions,
        worker_ids=low_latitude_worker_ids,
        assignments=assignments,
        region_runtimes=region_runtimes,
        region_costs=region_costs,
    )
    _assign_regions_by_runtime(
        high_lat_regions,
        worker_ids=high_latitude_worker_ids or low_latitude_worker_ids,
        assignments=assignments,
        region_runtimes=region_runtimes,
        region_costs=region_costs,
    )

    active_worker_ids = [
        worker_id for worker_id, region_pixs in enumerate(assignments) if region_pixs
    ]
    worker_rank_by_id = _rank_workers_by_peak_region(
        active_worker_ids,
        assignments=assignments,
        region_peak_stars=region_peak_stars,
    )

    plans: list[TraversalWorkerPlan] = []
    for worker_id, region_pixs in enumerate(assignments):
        if not region_pixs:
            continue
        worker_rank = worker_rank_by_id[worker_id]
        ordered_region_pixs = _galactic_longitude_region_order(
            region_pixs,
            region_abs_lat=region_abs_lat,
            region_lon=region_lon,
            worker_rank=worker_rank,
            worker_count=len(active_worker_ids),
        )
        ordered_outer_pixs: list[int] = []
        for region_index, region_pix in enumerate(ordered_region_pixs):
            seed_rank = 0
            if region_index == 0:
                seed_rank = _memory_staggered_seed_rank(
                    len(region_members[region_pix]),
                    worker_rank=worker_rank,
                    worker_count=len(active_worker_ids),
                )
            ordered_outer_pixs.extend(
                order_region_outer_pixs(
                    region_members[region_pix],
                    outer_level=outer_level,
                    star_counts=star_counts,
                    seed_rank=seed_rank,
                )
            )
        plans.append(
            TraversalWorkerPlan(
                worker_id=worker_id,
                region_pixs=tuple(int(pix) for pix in ordered_region_pixs),
                outer_pixs=tuple(ordered_outer_pixs),
                estimated_star_count=sum(
                    int(star_counts[pix]) for pix in ordered_outer_pixs
                ),
            )
        )
    return tuple(plans)


def order_region_outer_pixs(
    outer_pixs: list[int] | tuple[int, ...],
    *,
    outer_level: int,
    star_counts: np.ndarray,
    seed_rank: int = 0,
) -> tuple[int, ...]:
    """Order one region's outer pixels with neighbour-first locality."""

    remaining = set(int(pix) for pix in outer_pixs)
    ordered: list[int] = []
    frontier: list[int] = []
    seeded = False

    while remaining:
        if frontier:
            current = frontier.pop()
            if current not in remaining:
                continue
        elif not seeded:
            current = _ranked_outer_pix(
                remaining,
                star_counts=star_counts,
                rank=seed_rank,
            )
            seeded = True
        else:
            current = max(remaining, key=lambda pix: (int(star_counts[pix]), -pix))

        remaining.remove(current)
        ordered.append(current)
        neighbours = [
            int(pix)
            for pix in get_pixel_neighbours(outer_level, current)
            if int(pix) in remaining
        ]
        neighbours.sort(key=lambda pix: (int(star_counts[pix]), -pix))
        frontier.extend(neighbours)

    return tuple(ordered)


def _region_star_count(
    outer_pixs: list[int] | tuple[int, ...],
    *,
    star_counts: np.ndarray,
) -> int:
    return sum(int(star_counts[int(pix)]) for pix in outer_pixs)


def _region_peak_star_count(
    outer_pixs: list[int] | tuple[int, ...],
    *,
    star_counts: np.ndarray,
) -> int:
    return max((int(star_counts[int(pix)]) for pix in outer_pixs), default=0)


def _region_schedule_seconds(
    outer_pixs: list[int] | tuple[int, ...],
    *,
    star_counts: np.ndarray,
) -> float:
    return sum(
        _outer_pixel_schedule_seconds(int(star_counts[int(pix)]))
        for pix in outer_pixs
    )


def _outer_pixel_schedule_seconds(star_count: int) -> float:
    safe_star_count = max(float(star_count), 1.0)
    fitted_seconds = (
        REGIONAL_SCHEDULE_THROUGHPUT_SCALE
        * REGIONAL_SCHEDULE_SECONDS_COEFFICIENT
        * (safe_star_count ** REGIONAL_SCHEDULE_SECONDS_STAR_EXPONENT)
    )
    return min(float(fitted_seconds), REGIONAL_SCHEDULE_SECONDS_DENSE_CAP)


def _region_galactic_centers(
    region_pixs: tuple[int, ...],
    *,
    region_level: int,
) -> tuple[dict[int, float], dict[int, float]]:
    if not region_pixs:
        return {}, {}
    region_array = np.asarray(region_pixs, dtype=np.int64)
    galactic = get_pixel_skycoord(region_level, region_array).galactic
    abs_latitudes = np.abs(np.asarray(galactic.b.to_value("degree"), dtype=np.float64))
    longitudes = np.asarray(galactic.l.to_value("degree"), dtype=np.float64)
    return (
        {
            int(region_pix): float(abs_lat)
            for region_pix, abs_lat in zip(region_array, abs_latitudes, strict=True)
        },
        {
            int(region_pix): float(lon)
            for region_pix, lon in zip(region_array, longitudes, strict=True)
        },
    )


def _assign_regions_by_runtime(
    region_pixs: list[int],
    *,
    worker_ids: list[int],
    assignments: list[list[int]],
    region_runtimes: dict[int, float],
    region_costs: dict[int, int],
) -> None:
    if not region_pixs or not worker_ids:
        return
    assigned_runtimes = {
        int(worker_id): sum(
            region_runtimes[int(region_pix)]
            for region_pix in assignments[int(worker_id)]
        )
        for worker_id in worker_ids
    }
    ordered_regions = sorted(
        (int(region_pix) for region_pix in region_pixs),
        key=lambda pix: (-region_runtimes[int(pix)], -region_costs[int(pix)], int(pix)),
    )
    for region_pix in ordered_regions:
        worker_id = min(
            worker_ids,
            key=lambda idx: (assigned_runtimes[int(idx)], int(idx)),
        )
        assignments[int(worker_id)].append(int(region_pix))
        assigned_runtimes[int(worker_id)] += region_runtimes[int(region_pix)]


def _galactic_longitude_region_order(
    region_pixs: list[int],
    *,
    region_abs_lat: dict[int, float],
    region_lon: dict[int, float],
    worker_rank: int,
    worker_count: int,
) -> tuple[int, ...]:
    ordered = sorted(
        (int(pix) for pix in region_pixs),
        key=lambda pix: (region_lon[int(pix)], region_abs_lat[int(pix)], int(pix)),
    )
    if len(ordered) <= 1 or worker_count <= 1:
        return tuple(ordered)
    start_index = _spread_rank(
        count=len(ordered),
        worker_rank=worker_rank,
        worker_count=worker_count,
    )
    return tuple(ordered[start_index:] + ordered[:start_index])


def _rank_workers_by_peak_region(
    worker_ids: list[int],
    *,
    assignments: list[list[int]],
    region_peak_stars: dict[int, int],
) -> dict[int, int]:
    ordered_worker_ids = sorted(
        worker_ids,
        key=lambda worker_id: (
            -max(
                region_peak_stars[region_pix]
                for region_pix in assignments[worker_id]
            ),
            worker_id,
        ),
    )
    return {worker_id: rank for rank, worker_id in enumerate(ordered_worker_ids)}


def _memory_staggered_seed_rank(
    outer_pix_count: int,
    *,
    worker_rank: int,
    worker_count: int,
) -> int:
    return _spread_rank(
        count=int(outer_pix_count),
        worker_rank=worker_rank,
        worker_count=worker_count,
    )


def _spread_rank(*, count: int, worker_rank: int, worker_count: int) -> int:
    if count <= 1 or worker_count <= 1:
        return 0
    clamped_worker_rank = min(max(int(worker_rank), 0), int(worker_count) - 1)
    return int(
        round(clamped_worker_rank * (int(count) - 1) / (int(worker_count) - 1))
    )


def _ranked_outer_pix(
    outer_pixs: set[int],
    *,
    star_counts: np.ndarray,
    rank: int,
) -> int:
    ordered = sorted(outer_pixs, key=lambda pix: (-int(star_counts[pix]), pix))
    clamped_rank = min(max(int(rank), 0), len(ordered) - 1)
    return int(ordered[clamped_rank])
