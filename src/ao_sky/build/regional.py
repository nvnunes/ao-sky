"""Regional Traversal planning helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..spatial import get_parent_pixel, get_pixel_neighbours
from ._constants import WORK_STATUS_FAILED, WORK_STATUS_PENDING
from ._models import TraversalWorkerPlan


def build_regional_worker_plans(
    *,
    state: np.ndarray,
    outer_level: int,
    region_level: int,
    workers: int,
    status_field: str,
    star_counts: np.ndarray,
) -> tuple[TraversalWorkerPlan, ...]:
    """Return balanced regional Traversal plans for long-lived workers."""

    eligible = [
        int(row["outer_pix"])
        for row in state
        if int(row[status_field]) in (WORK_STATUS_PENDING, WORK_STATUS_FAILED)
    ]
    if not eligible:
        return ()

    region_for_outer = get_parent_pixel(
        outer_level,
        np.asarray(eligible, dtype=np.int64),
        region_level,
    )
    region_members: dict[int, list[int]] = defaultdict(list)
    for outer_pix, region_pix in zip(eligible, region_for_outer, strict=True):
        region_members[int(region_pix)].append(int(outer_pix))

    ordered_regions = sorted(
        region_members,
        key=lambda region_pix: (
            -sum(int(star_counts[pix]) for pix in region_members[region_pix]),
            region_pix,
        ),
    )
    assignments: list[list[int]] = [[] for _ in range(int(workers))]
    assignment_costs = [0 for _ in range(int(workers))]
    for region_pix in ordered_regions:
        worker_id = min(range(int(workers)), key=lambda idx: (assignment_costs[idx], idx))
        assignments[worker_id].append(region_pix)
        assignment_costs[worker_id] += sum(
            int(star_counts[pix]) for pix in region_members[region_pix]
        )

    plans: list[TraversalWorkerPlan] = []
    for worker_id, region_pixs in enumerate(assignments):
        if not region_pixs:
            continue
        ordered_outer_pixs: list[int] = []
        for region_pix in region_pixs:
            ordered_outer_pixs.extend(
                order_region_outer_pixs(
                    region_members[region_pix],
                    outer_level=outer_level,
                    star_counts=star_counts,
                )
            )
        plans.append(
            TraversalWorkerPlan(
                worker_id=worker_id,
                region_pixs=tuple(int(pix) for pix in region_pixs),
                outer_pixs=tuple(ordered_outer_pixs),
                estimated_star_count=sum(int(star_counts[pix]) for pix in ordered_outer_pixs),
            )
        )
    return tuple(plans)


def order_region_outer_pixs(
    outer_pixs: list[int] | tuple[int, ...],
    *,
    outer_level: int,
    star_counts: np.ndarray,
) -> tuple[int, ...]:
    """Order one region's outer pixels with neighbour-first locality."""

    remaining = set(int(pix) for pix in outer_pixs)
    ordered: list[int] = []
    frontier: list[int] = []

    while remaining:
        if frontier:
            current = frontier.pop()
            if current not in remaining:
                continue
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
