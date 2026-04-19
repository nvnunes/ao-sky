"""Simulate Traversal regional schedule memory high-water.

This is an offline planning tool. It uses the same regional worker planner as
Traversal, then estimates how worker RAM overlaps over the static worker plans.
The model treats each outer pixel as holding its estimated peak worker RAM for
the full estimated runtime of that pixel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
from astropy.table import Table

from ao_sky.build._constants import STATE_DTYPE, WORK_STATUS_DONE, WORK_STATUS_PENDING
from ao_sky.build.config import derive_traversal_region_level
from ao_sky.build.control import load_build_definition, load_build_roots, load_state
from ao_sky.build.regional import build_regional_worker_plans
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.spatial import get_pixel_skycoord

WORKER_RAM_STAR_COEFFICIENT_GIB = 0.000264
WORKER_RAM_STAR_EXPONENT = 0.690

OUTER_PIXEL_SECONDS_COEFFICIENT = 6.8603409602539e-05
OUTER_PIXEL_SECONDS_STAR_EXPONENT = 1.2184647278722778
OUTER_PIXEL_SECONDS_DENSE_CAP = 18.58285714285714
OUTER_PIXEL_SECONDS_REGIONAL_THROUGHPUT_SCALE = 0.5

CPU_THROUGHPUT_SLOPE = -0.00171
CPU_THROUGHPUT_INTERCEPT = 0.04572
GPU_THROUGHPUT_SLOPE = -0.00208
GPU_THROUGHPUT_INTERCEPT = 0.07455

GPU_RAM_OVERHEAD_INTERCEPT_GIB = 9.71928571
GPU_RAM_OVERHEAD_SLOPE_GIB_PER_WORKER = -0.66107143


@dataclass(frozen=True, slots=True)
class PixelEstimate:
    """One scheduled outer-pixel interval."""

    worker_id: int
    outer_pix: int
    star_count: int
    worker_ram_mb: float
    seconds: float
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class WorkerEstimate:
    """Summary of one worker plan."""

    worker_id: int
    outer_pixels: int
    regions: int
    estimated_star_count: int
    first_outer_pix: int
    first_outer_pix_stars: int
    peak_worker_ram_mb: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class ScheduleEstimate:
    """Approximate memory shape of one static regional Traversal schedule."""

    worker_count: int
    outer_pixel_count: int
    total_worker_seconds: float
    elapsed_seconds: float
    worker_ram_peak_mb: float
    worker_ram_sum_peak_mb: float
    gpu_overhead_peak_mb: float
    ram_overhead_mb: float
    total_ram_peak_mb: float
    peak_time_seconds: float
    heaviest_outer_pix: int
    heaviest_outer_pix_stars: int
    heaviest_outer_pix_ram_mb: float
    active_at_peak: tuple[PixelEstimate, ...]
    workers: tuple[WorkerEstimate, ...]
    pixel_rows: tuple[PixelEstimate, ...]


def main() -> None:
    args = _parse_args()
    build_path = Path(args.build_path).expanduser().resolve()
    definition = load_build_definition(build_path)
    roots = load_build_roots(build_path)
    star_counts = _load_star_counts(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
    )
    state = _prepare_state(
        build_path,
        outer_level=definition.outer_level,
        scope=args.scope,
        sample_pixels=args.sample_pixels,
        schedule_full_sky=bool(args.schedule_full_sky),
    )
    workers = int(args.workers)
    if workers < 1:
        raise ValueError(f"--workers must be at least 1, got {workers}")
    region_level = (
        int(args.region_level)
        if args.region_level is not None
        else derive_traversal_region_level(
            outer_level=definition.outer_level,
            workers=workers,
        )
    )
    plans = build_regional_worker_plans(
        state=state,
        outer_level=definition.outer_level,
        region_level=region_level,
        workers=workers,
        status_field="traversal_status",
        star_counts=star_counts,
        low_latitude_deg=float(args.galactic_low_latitude_deg),
        low_latitude_workers=args.galactic_low_latitude_workers,
    )
    estimate = estimate_schedule(
        plans,
        star_counts=star_counts,
        per_worker_gpu_overhead_mb=1024.0 * float(args.per_worker_gpu_overhead_gib),
        runtime_scale=throughput_runtime_scale(
            workers=workers,
            device=args.throughput_device,
        ),
    )
    _print_summary(
        estimate,
        build_path=build_path,
        outer_level=definition.outer_level,
        region_level=region_level,
        scope="full-sky-config" if args.schedule_full_sky else args.scope,
        scheduler="galactic-latitude",
    )
    if args.show_active_peak:
        _print_active_peak(estimate)
    if args.csv_output is not None:
        _write_worker_csv(Path(args.csv_output).expanduser(), estimate)
    if args.plot_output is not None:
        active_low_latitude_workers = _active_low_latitude_worker_count(
            plans,
            region_level=region_level,
            low_latitude_deg=float(args.galactic_low_latitude_deg),
        )
        _write_plot(
            Path(args.plot_output).expanduser(),
            estimate,
            per_worker_gpu_overhead_mb=1024.0 * float(args.per_worker_gpu_overhead_gib),
            plot_title=_plot_title(
                args,
                estimate,
                low_latitude_workers=active_low_latitude_workers,
            ),
        )


def estimate_schedule(
    plans,
    *,
    star_counts: np.ndarray,
    per_worker_gpu_overhead_mb: float,
    runtime_scale: float = 1.0,
) -> ScheduleEstimate:
    active_plans = tuple(plan for plan in plans if plan.outer_pixs)
    if not active_plans:
        return ScheduleEstimate(
            worker_count=0,
            outer_pixel_count=0,
            total_worker_seconds=0.0,
            elapsed_seconds=0.0,
            worker_ram_peak_mb=0.0,
            worker_ram_sum_peak_mb=0.0,
            gpu_overhead_peak_mb=0.0,
            ram_overhead_mb=0.0,
            total_ram_peak_mb=0.0,
            peak_time_seconds=0.0,
            heaviest_outer_pix=-1,
            heaviest_outer_pix_stars=0,
            heaviest_outer_pix_ram_mb=0.0,
            active_at_peak=(),
            workers=(),
            pixel_rows=(),
        )

    rows_by_worker = {
        int(plan.worker_id): tuple(
            _pixel_estimate(
                worker_id=int(plan.worker_id),
                outer_pix=int(outer_pix),
                star_counts=star_counts,
                runtime_scale=runtime_scale,
            )
            for outer_pix in plan.outer_pixs
        )
        for plan in active_plans
    }
    workers = tuple(
        _worker_estimate(
            plan,
            rows_by_worker[int(plan.worker_id)],
            star_counts=star_counts,
        )
        for plan in active_plans
    )
    active: dict[int, PixelEstimate] = {}
    worker_indices = {int(plan.worker_id): 0 for plan in active_plans}
    heap: list[tuple[float, int]] = []
    scheduled_rows: list[PixelEstimate] = []

    for plan in active_plans:
        worker_id = int(plan.worker_id)
        row = _schedule_row(rows_by_worker[worker_id][0], start_seconds=0.0)
        active[worker_id] = row
        scheduled_rows.append(row)
        heapq.heappush(heap, (row.end_seconds, worker_id))

    ram_overhead_mb = estimate_gpu_ram_overhead_mb(len(active_plans))
    peak_time_seconds, worker_ram_sum_peak_mb, gpu_overhead_peak_mb, active_at_peak = (
        _snapshot_peak(
            active,
            per_worker_gpu_overhead_mb,
            worker_count=len(active_plans),
        )
    )
    elapsed_seconds = 0.0

    while heap:
        end_seconds = heap[0][0]
        elapsed_seconds = max(elapsed_seconds, float(end_seconds))
        completed_worker_ids: list[int] = []
        while heap and heap[0][0] == end_seconds:
            _, worker_id = heapq.heappop(heap)
            completed_worker_ids.append(worker_id)

        for worker_id in completed_worker_ids:
            active.pop(worker_id, None)

        for worker_id in completed_worker_ids:
            worker_indices[worker_id] += 1
            next_index = worker_indices[worker_id]
            rows = rows_by_worker[worker_id]
            if next_index >= len(rows):
                continue
            row = _schedule_row(rows[next_index], start_seconds=float(end_seconds))
            active[worker_id] = row
            scheduled_rows.append(row)
            heapq.heappush(heap, (row.end_seconds, worker_id))

        _, worker_sum, gpu_overhead, active_rows = _snapshot_peak(
            active,
            per_worker_gpu_overhead_mb,
            worker_count=len(active_plans),
        )
        if worker_sum + gpu_overhead > worker_ram_sum_peak_mb + gpu_overhead_peak_mb:
            peak_time_seconds = float(end_seconds)
            worker_ram_sum_peak_mb = worker_sum
            gpu_overhead_peak_mb = gpu_overhead
            active_at_peak = active_rows

    all_rows = tuple(scheduled_rows)
    heaviest = max(
        all_rows,
        key=lambda row: (row.worker_ram_mb, row.star_count, -row.outer_pix),
    )
    return ScheduleEstimate(
        worker_count=len(active_plans),
        outer_pixel_count=len(all_rows),
        total_worker_seconds=sum(row.seconds for row in all_rows),
        elapsed_seconds=elapsed_seconds,
        worker_ram_peak_mb=max(row.worker_ram_mb for row in all_rows),
        worker_ram_sum_peak_mb=worker_ram_sum_peak_mb,
        gpu_overhead_peak_mb=gpu_overhead_peak_mb,
        ram_overhead_mb=ram_overhead_mb,
        total_ram_peak_mb=ram_overhead_mb + worker_ram_sum_peak_mb + gpu_overhead_peak_mb,
        peak_time_seconds=peak_time_seconds,
        heaviest_outer_pix=heaviest.outer_pix,
        heaviest_outer_pix_stars=heaviest.star_count,
        heaviest_outer_pix_ram_mb=heaviest.worker_ram_mb,
        active_at_peak=active_at_peak,
        workers=workers,
        pixel_rows=all_rows,
    )


def estimate_worker_ram_mb(star_count: int) -> float:
    safe_star_count = max(float(star_count), 0.0)
    return 1024.0 * (
        WORKER_RAM_STAR_COEFFICIENT_GIB
        * (safe_star_count ** WORKER_RAM_STAR_EXPONENT)
    )


def estimate_gpu_ram_overhead_mb(workers: int) -> float:
    overhead_gib = (
        GPU_RAM_OVERHEAD_INTERCEPT_GIB
        + GPU_RAM_OVERHEAD_SLOPE_GIB_PER_WORKER * float(workers)
    )
    return 1024.0 * max(float(overhead_gib), 0.0)


def estimate_outer_pixel_seconds(star_count: int) -> float:
    safe_star_count = max(float(star_count), 1.0)
    fitted_seconds = (
        OUTER_PIXEL_SECONDS_REGIONAL_THROUGHPUT_SCALE
        * OUTER_PIXEL_SECONDS_COEFFICIENT
        * (safe_star_count ** OUTER_PIXEL_SECONDS_STAR_EXPONENT)
    )
    return min(float(fitted_seconds), OUTER_PIXEL_SECONDS_DENSE_CAP)


def throughput_runtime_scale(*, workers: int, device: str) -> float:
    if device == "none":
        return 1.0
    if device == "cpu":
        slope = CPU_THROUGHPUT_SLOPE
        intercept = CPU_THROUGHPUT_INTERCEPT
    elif device == "gpu":
        slope = GPU_THROUGHPUT_SLOPE
        intercept = GPU_THROUGHPUT_INTERCEPT
    else:
        raise ValueError(f"unknown throughput device {device!r}")
    reference = _throughput_per_worker(workers=1, slope=slope, intercept=intercept)
    current = _throughput_per_worker(
        workers=int(workers),
        slope=slope,
        intercept=intercept,
    )
    return float(reference / current)


def _throughput_per_worker(*, workers: int, slope: float, intercept: float) -> float:
    value = float(slope) * float(workers) + float(intercept)
    if value <= 0.0:
        raise ValueError(
            "throughput fit is non-positive for "
            f"workers={workers}: {value:.6f}"
        )
    return value


def _pixel_estimate(
    *,
    worker_id: int,
    outer_pix: int,
    star_counts: np.ndarray,
    runtime_scale: float,
) -> PixelEstimate:
    star_count = int(star_counts[int(outer_pix)])
    seconds = estimate_outer_pixel_seconds(star_count) * float(runtime_scale)
    return PixelEstimate(
        worker_id=int(worker_id),
        outer_pix=int(outer_pix),
        star_count=star_count,
        worker_ram_mb=estimate_worker_ram_mb(star_count),
        seconds=seconds,
        start_seconds=0.0,
        end_seconds=0.0,
    )


def _schedule_row(row: PixelEstimate, *, start_seconds: float) -> PixelEstimate:
    return PixelEstimate(
        worker_id=row.worker_id,
        outer_pix=row.outer_pix,
        star_count=row.star_count,
        worker_ram_mb=row.worker_ram_mb,
        seconds=row.seconds,
        start_seconds=float(start_seconds),
        end_seconds=float(start_seconds) + float(row.seconds),
    )


def _worker_estimate(plan, rows: tuple[PixelEstimate, ...], *, star_counts: np.ndarray):
    if not rows:
        first_outer_pix = -1
        first_outer_pix_stars = 0
        peak_worker_ram_mb = 0.0
    else:
        first_outer_pix = rows[0].outer_pix
        first_outer_pix_stars = int(star_counts[first_outer_pix])
        peak_worker_ram_mb = max(row.worker_ram_mb for row in rows)
    return WorkerEstimate(
        worker_id=int(plan.worker_id),
        outer_pixels=len(rows),
        regions=len(plan.region_pixs),
        estimated_star_count=int(plan.estimated_star_count),
        first_outer_pix=int(first_outer_pix),
        first_outer_pix_stars=int(first_outer_pix_stars),
        peak_worker_ram_mb=float(peak_worker_ram_mb),
        total_seconds=sum(row.seconds for row in rows),
    )


def _snapshot_peak(
    active: dict[int, PixelEstimate],
    per_worker_gpu_overhead_mb: float,
    worker_count: int,
) -> tuple[float, float, float, tuple[PixelEstimate, ...]]:
    active_rows = tuple(active[worker_id] for worker_id in sorted(active))
    worker_ram_sum_mb = sum(row.worker_ram_mb for row in active_rows)
    gpu_overhead_mb = float(per_worker_gpu_overhead_mb) * int(worker_count)
    peak_time_seconds = min((row.start_seconds for row in active_rows), default=0.0)
    return peak_time_seconds, worker_ram_sum_mb, gpu_overhead_mb, active_rows


def _load_star_counts(*, gaia_root: Path, gaia_release: str, outer_level: int) -> np.ndarray:
    summary = GaiaSummaryStore(
        GaiaStoreConfig(
            root=gaia_root,
            release=gaia_release,
            healpix_level=outer_level,
        )
    ).load_summary()
    expected_rows = 12 * (4 ** int(outer_level))
    if len(summary) != expected_rows:
        raise RuntimeError(
            f"Gaia summary has {len(summary)} rows, expected {expected_rows}"
        )
    loaded = np.asarray(summary["loaded"], dtype=np.bool_)
    if not np.all(loaded):
        missing = np.flatnonzero(~loaded)
        raise RuntimeError(f"Gaia summary is incomplete; first missing={missing[0]}")
    return np.asarray(summary["star_count"], dtype=np.int64)


def _prepare_state(
    build_path: Path,
    *,
    outer_level: int,
    scope: str,
    sample_pixels: Path | None,
    schedule_full_sky: bool,
) -> np.ndarray:
    if schedule_full_sky:
        if scope != "all":
            raise ValueError("--schedule-full-sky cannot be combined with --scope remaining")
        if sample_pixels is not None:
            raise ValueError("--schedule-full-sky cannot be combined with --sample-pixels")
        return _make_full_sky_state(outer_level)

    state = load_state(build_path).copy()
    if scope == "all":
        state["traversal_status"] = WORK_STATUS_PENDING
    elif scope != "remaining":
        raise ValueError(f"unknown scope {scope!r}")

    if sample_pixels is not None:
        pixels = _read_sample_pixels(sample_pixels)
        if np.any(pixels >= len(state)):
            raise ValueError("sample outer_pix values exceed build state size")
        mask = np.zeros(len(state), dtype=np.bool_)
        mask[pixels] = True
        state["traversal_status"] = WORK_STATUS_DONE
        state["traversal_status"][mask] = WORK_STATUS_PENDING
    return state


def _make_full_sky_state(outer_level: int) -> np.ndarray:
    num_pixels = 12 * (4 ** int(outer_level))
    state = np.zeros(num_pixels, dtype=STATE_DTYPE)
    state["outer_pix"] = np.arange(num_pixels, dtype=np.int64)
    state["traversal_status"] = WORK_STATUS_PENDING
    return state


def _read_sample_pixels(filename: Path) -> np.ndarray:
    table = Table.read(Path(filename).expanduser())
    if "outer_pix" not in table.colnames:
        raise ValueError(f"{filename} must contain an outer_pix column")
    pixels = np.asarray(table["outer_pix"], dtype=np.int64)
    if np.any(pixels < 0):
        raise ValueError("outer_pix values must be non-negative")
    return pixels


def _print_summary(
    estimate: ScheduleEstimate,
    *,
    build_path: Path,
    outer_level: int,
    region_level: int,
    scope: str,
    scheduler: str,
) -> None:
    print(f"Build: {build_path}")
    print(f"Scope: {scope}")
    print(f"Scheduler: {scheduler}")
    print(f"Outer level: {outer_level}")
    print(f"Region level: {region_level}")
    print(f"Workers: {estimate.worker_count}")
    print(f"Outer pixels: {estimate.outer_pixel_count}")
    print(f"Estimated elapsed: {estimate.elapsed_seconds:.2f} s")
    print(f"Estimated worker-seconds: {estimate.total_worker_seconds:.2f} s")
    print(f"Worker RAM peak: {_gib(estimate.worker_ram_peak_mb):.2f} GiB")
    print(f"Worker RAM sum high-water: {_gib(estimate.worker_ram_sum_peak_mb):.2f} GiB")
    print(f"GPU overhead high-water: {_gib(estimate.gpu_overhead_peak_mb):.2f} GiB")
    print(f"RAM overhead: {_gib(estimate.ram_overhead_mb):.2f} GiB")
    print(f"Total RAM high-water: {_gib(estimate.total_ram_peak_mb):.2f} GiB")
    print(f"Peak time: {estimate.peak_time_seconds:.2f} s")
    print(
        "Heaviest outer pixel: "
        f"{estimate.heaviest_outer_pix} "
        f"stars={estimate.heaviest_outer_pix_stars} "
        f"ram={_gib(estimate.heaviest_outer_pix_ram_mb):.2f} GiB"
    )
    print()
    print(
        "| Worker | Pixels | Regions | First Pixel | First Stars | "
        "Plan Stars | Peak RAM | Runtime |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for worker in estimate.workers:
        print(
            f"| {worker.worker_id} "
            f"| {worker.outer_pixels} "
            f"| {worker.regions} "
            f"| {worker.first_outer_pix} "
            f"| {worker.first_outer_pix_stars} "
            f"| {worker.estimated_star_count} "
            f"| {_gib(worker.peak_worker_ram_mb):.2f} GiB "
            f"| {worker.total_seconds:.2f} s |"
        )


def _print_active_peak(estimate: ScheduleEstimate) -> None:
    print()
    print("Active outer pixels at simulated high-water:")
    print("| Worker | Outer Pixel | Stars | RAM | Start | End |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in estimate.active_at_peak:
        print(
            f"| {row.worker_id} "
            f"| {row.outer_pix} "
            f"| {row.star_count} "
            f"| {_gib(row.worker_ram_mb):.2f} GiB "
            f"| {row.start_seconds:.2f} s "
            f"| {row.end_seconds:.2f} s |"
        )


def _write_worker_csv(filename: Path, estimate: ScheduleEstimate) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "worker_id",
                "outer_pixels",
                "regions",
                "estimated_star_count",
                "first_outer_pix",
                "first_outer_pix_stars",
                "peak_worker_ram_mb",
                "total_seconds",
            ),
        )
        writer.writeheader()
        for worker in estimate.workers:
            writer.writerow(
                {
                    "worker_id": worker.worker_id,
                    "outer_pixels": worker.outer_pixels,
                    "regions": worker.regions,
                    "estimated_star_count": worker.estimated_star_count,
                    "first_outer_pix": worker.first_outer_pix,
                    "first_outer_pix_stars": worker.first_outer_pix_stars,
                    "peak_worker_ram_mb": f"{worker.peak_worker_ram_mb:.3f}",
                    "total_seconds": f"{worker.total_seconds:.3f}",
                }
            )


def build_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not estimate.pixel_rows:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, zeros, zeros, zeros

    events: list[tuple[float, int, float]] = []
    for row in estimate.pixel_rows:
        events.append((float(row.start_seconds), 1, float(row.worker_ram_mb)))
        events.append((float(row.end_seconds), -1, float(row.worker_ram_mb)))
    events.sort(key=lambda event: (event[0], event[1]))

    times: list[float] = [0.0]
    worker_ram_values: list[float] = [0.0]
    configured_gpu_mb = float(per_worker_gpu_overhead_mb) * int(estimate.worker_count)
    gpu_values: list[float] = [configured_gpu_mb]
    total_values: list[float] = [float(estimate.ram_overhead_mb)]
    active_worker_count = 0
    worker_ram_mb = 0.0
    index = 0

    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and events[index][0] == event_time:
            _, direction, ram_mb = events[index]
            active_worker_count += int(direction)
            worker_ram_mb += float(direction) * float(ram_mb)
            index += 1
        times.append(float(event_time))
        worker_ram_values.append(float(worker_ram_mb))
        gpu_values.append(float(configured_gpu_mb))
        total_values.append(
            float(estimate.ram_overhead_mb + worker_ram_mb + configured_gpu_mb)
        )

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(worker_ram_values, dtype=np.float64),
        np.asarray(gpu_values, dtype=np.float64),
        np.asarray(total_values, dtype=np.float64),
    )


def build_worker_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    rows_by_worker: dict[int, list[PixelEstimate]] = {}
    for row in estimate.pixel_rows:
        rows_by_worker.setdefault(row.worker_id, []).append(row)

    series: list[tuple[int, np.ndarray, np.ndarray]] = []
    for worker_id in sorted(rows_by_worker):
        times = [0.0]
        values = [0.0]
        for row in sorted(rows_by_worker[worker_id], key=lambda item: item.start_seconds):
            ram_mb = float(row.worker_ram_mb) + float(per_worker_gpu_overhead_mb)
            if times[-1] != row.start_seconds:
                times.append(float(row.start_seconds))
                values.append(values[-1])
            values[-1] = ram_mb
            times.append(float(row.end_seconds))
            values.append(ram_mb)
        if values[-1] != 0.0:
            times.append(times[-1])
            values.append(0.0)
        series.append(
            (
                int(worker_id),
                np.asarray(times, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
        )
    return tuple(series)


def build_stacked_worker_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    worker_ids = tuple(worker.worker_id for worker in estimate.workers)
    if not worker_ids:
        return (
            np.asarray([0.0], dtype=np.float64),
            (),
            np.zeros((0, 1), dtype=np.float64),
        )

    worker_index = {worker_id: index for index, worker_id in enumerate(worker_ids)}
    events: list[tuple[float, int, float]] = []
    for row in estimate.pixel_rows:
        ram_mb = float(row.worker_ram_mb) + float(per_worker_gpu_overhead_mb)
        events.append((float(row.start_seconds), int(row.worker_id), ram_mb))
        events.append((float(row.end_seconds), int(row.worker_id), 0.0))
    events.sort(key=lambda event: (event[0], event[2] != 0.0))

    times: list[float] = [0.0]
    worker_values = np.zeros(len(worker_ids), dtype=np.float64)
    rows: list[np.ndarray] = [np.sort(worker_values)]
    index = 0
    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and events[index][0] == event_time:
            _, worker_id, ram_mb = events[index]
            worker_values[worker_index[int(worker_id)]] = float(ram_mb)
            index += 1
        times.append(float(event_time))
        rows.append(np.sort(worker_values))

    return (
        np.asarray(times, dtype=np.float64),
        tuple(range(len(worker_ids))),
        np.asarray(rows, dtype=np.float64).T,
    )


def _write_plot(
    filename: Path,
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
    plot_title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, worker_ids, worker_ram_by_worker_mb = build_stacked_worker_ram_time_series(
        estimate,
        per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    worker_ram_by_worker_gib = worker_ram_by_worker_mb / 1024.0
    overhead_gib = estimate.ram_overhead_mb / 1024.0
    if overhead_gib > 0.0:
        ax.fill_between(
            times,
            np.zeros(len(times), dtype=np.float64),
            np.full(len(times), overhead_gib, dtype=np.float64),
            step="post",
            color="0.80",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
    baseline = np.full(len(times), overhead_gib, dtype=np.float64)
    colors = plt.get_cmap("tab10").colors
    for index, worker_id in enumerate(worker_ids):
        worker_gib = worker_ram_by_worker_gib[index]
        top = baseline + worker_gib
        ax.fill_between(
            times,
            baseline,
            top,
            step="post",
            color=colors[index % len(colors)],
            alpha=0.82,
            linewidth=0,
            zorder=2,
        )
        baseline = top
    ax.axhline(
        estimate.total_ram_peak_mb / 1024.0,
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        zorder=3,
    )
    ax.set_xlabel("Simulated elapsed time [s]")
    ax.set_ylabel("Estimated RAM [GiB]")
    title = plot_title or (
        f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    )
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _active_low_latitude_worker_count(
    plans,
    *,
    region_level: int,
    low_latitude_deg: float,
) -> int:
    count = 0
    for plan in plans:
        if not plan.region_pixs:
            continue
        region_pixs = np.asarray(plan.region_pixs, dtype=np.int64)
        galactic = get_pixel_skycoord(region_level, region_pixs).galactic
        abs_latitude = np.abs(
            np.asarray(galactic.b.to_value("degree"), dtype=np.float64)
        )
        if np.any(abs_latitude <= float(low_latitude_deg)):
            count += 1
    return count


def _plot_title(
    args: argparse.Namespace,
    estimate: ScheduleEstimate,
    *,
    low_latitude_workers: int,
) -> str:
    if args.plot_title is not None:
        return str(args.plot_title)
    title = f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    if low_latitude_workers > 0:
        title = f"{title} [{low_latitude_workers} low-lat workers]"
    return title


def _gib(mb: float) -> float:
    return float(mb) / 1024.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_path", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--region-level", type=int)
    parser.add_argument(
        "--scope",
        choices=("all", "remaining"),
        default="all",
        help="'all' simulates every state row; 'remaining' uses pending/failed rows.",
    )
    parser.add_argument(
        "--sample-pixels",
        type=Path,
        help="Optional ECSV/FITS/etc. table with an outer_pix column.",
    )
    parser.add_argument(
        "--schedule-full-sky",
        action="store_true",
        help=(
            "Use the build config and Gaia roots, but synthesize a full-sky "
            "pending Traversal state for the configured outer level."
        ),
    )
    parser.add_argument(
        "--per-worker-gpu-overhead-gib",
        type=float,
        default=0.0,
        help="Optional GPU high-water overhead to add for each configured worker.",
    )
    parser.add_argument(
        "--throughput-device",
        choices=("gpu", "cpu", "none"),
        default="gpu",
        help=(
            "Per-worker throughput correction to apply to one-worker pixel "
            "runtime estimates. Use 'none' to disable the correction."
        ),
    )
    parser.add_argument(
        "--galactic-low-latitude-deg",
        type=float,
        default=15.0,
        help=(
            "Absolute Galactic latitude cut for the low-latitude worker band "
            "used by the regional scheduler."
        ),
    )
    parser.add_argument(
        "--galactic-low-latitude-workers",
        type=int,
        help=(
            "Number of workers reserved for the low-latitude band. Defaults "
            "to two thirds of the workers."
        ),
    )
    parser.add_argument("--show-active-peak", action="store_true")
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--plot-title")
    return parser.parse_args()


if __name__ == "__main__":
    main()
