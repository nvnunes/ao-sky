"""Simulate Traversal regional schedule memory high-water.

This is an offline planning tool. Static scheduling uses the production regional
worker planner. Dynamic scheduling uses the production dynamic batch planner,
then applies this script's runtime and memory model to simulate execution.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import heapq
from pathlib import Path
import re

import numpy as np
from astropy.table import Table

from ao_sky.build._constants import (
    STATE_DTYPE,
    WORK_STATUS_DONE,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ao_sky.build.config import derive_traversal_region_level
from ao_sky.build.control import load_build_definition, load_build_roots, load_state
from ao_sky.build._models import TraversalWorkBatch
from ao_sky.build.regional import (
    DYNAMIC_RUNTIME_TRANSITION_END_STARS,
    DYNAMIC_RUNTIME_TRANSITION_START_STARS,
    build_regional_worker_plans,
    build_dynamic_work_batches,
    dynamic_outer_pixel_seconds,
    dynamic_worker_ram_mb,
)
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.spatial import get_parent_pixel, get_pixel_skycoord

WORKER_TOTAL_STRESS_INTERCEPT_GIB = 1.46
WORST_DENSE_PIXEL_STARS = 1_116_354

ORIGINAL_WORKER_RAM_STAR_COEFFICIENT_GIB = 0.000264
ORIGINAL_WORKER_RAM_STAR_EXPONENT = 0.690

ORIGINAL_OUTER_PIXEL_SECONDS_COEFFICIENT = 6.8603409602539e-05
ORIGINAL_OUTER_PIXEL_SECONDS_STAR_EXPONENT = 1.2184647278722778
ORIGINAL_OUTER_PIXEL_SECONDS_DENSE_CAP = 18.58285714285714
ORIGINAL_OUTER_PIXEL_SECONDS_THROUGHPUT_SCALE = 0.5
ORIGINAL_RAM_OVERHEAD_INTERCEPT_GIB = 9.71928571
ORIGINAL_RAM_OVERHEAD_SLOPE_GIB_PER_WORKER = -0.66107143

STRESS_PIXEL_RAM_FLOOR_GIB = 1.29
STRESS_PIXEL_RAM_DECAY_TAU_MIN = 2.5
NORMAL_PIXEL_RAM_FLOOR_GIB = 0.93
NORMAL_PIXEL_RAM_DECAY_TAU_MIN = 0.5

CPU_THROUGHPUT_SLOPE = -0.00171
CPU_THROUGHPUT_INTERCEPT = 0.04572
GPU_THROUGHPUT_SLOPE = -0.00208
GPU_THROUGHPUT_INTERCEPT = 0.07455

LINE_TIME_RE = re.compile(r"^(\S+) ")
KEY_VALUE_RE = re.compile(r"(\w+)=([^\s]+)")


@dataclass(frozen=True, slots=True)
class PixelEstimate:
    """One scheduled outer-pixel interval."""

    worker_id: int
    outer_pix: int
    star_count: int
    worker_ram_mb: float
    seconds: float
    planning_worker_ram_mb: float
    planning_seconds: float
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class WorkerTrimEvent:
    """One simulated worker trim event."""

    worker_id: int
    seconds: float
    is_stress: bool


@dataclass(frozen=True, slots=True)
class WorkBatch:
    """One scheduler assignment containing one or more outer pixels."""

    rows: tuple[PixelEstimate, ...]
    region_level: int
    region_pix: int
    is_stress: bool
    sort_star_count: int
    planning_seconds: float
    center_vector: tuple[float, float, float]


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
    trim_events: tuple[WorkerTrimEvent, ...] = ()
    stress_star_threshold: float = float("inf")
    stress_worker_count: int = 0
    worker_ram_includes_gpu: bool = False


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    build_path = Path(args.build_path).expanduser().resolve()
    definition = load_build_definition(build_path)
    roots = load_build_roots(build_path)
    star_counts = _load_star_counts(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
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
    rng = _random_generator(args)
    plans = ()
    measured_throughput_rows: tuple[PixelEstimate, ...] = ()
    measured_throughput_curve: tuple[np.ndarray, np.ndarray] | None = None
    if args.actual_run_window_timing is not None:
        estimate = _estimate_actual_run_window(
            build_path,
            run_window=args.actual_run_window_timing,
            build_log=args.build_log,
            star_counts=star_counts,
            configured_worker_count=workers,
            per_worker_gpu_overhead_mb=1024.0
            * float(args.per_worker_gpu_overhead_gib),
        )
    elif args.actual_run_window_throughput is not None:
        measured_throughput_rows = _actual_run_window_measurement_rows(
            build_path,
            run_window=args.actual_run_window_throughput,
            build_log=args.build_log,
            star_counts=star_counts,
        )
        state = _state_at_run_window_start(
            build_path,
            completed_rows=measured_throughput_rows,
        )
        plans, estimate = _estimate_planned_schedule(
            args,
            state=state,
            outer_level=definition.outer_level,
            region_level=region_level,
            workers=workers,
            star_counts=star_counts,
            rng=rng,
        )
    elif args.throughput_comparison_run_window is not None:
        measured_throughput_curve = _measured_progress_curve(
            build_path,
            run_window=args.throughput_comparison_run_window,
            build_log=args.build_log,
        )
        state = _prepare_state(
            build_path,
            outer_level=definition.outer_level,
            scope=args.scope,
            sample_pixels=args.sample_pixels,
            schedule_full_sky=bool(args.schedule_full_sky),
        )
        plans, estimate = _estimate_planned_schedule(
            args,
            state=state,
            outer_level=definition.outer_level,
            region_level=region_level,
            workers=workers,
            star_counts=star_counts,
            rng=rng,
        )
    else:
        state = _prepare_state(
            build_path,
            outer_level=definition.outer_level,
            scope=args.scope,
            sample_pixels=args.sample_pixels,
            schedule_full_sky=bool(args.schedule_full_sky),
        )
        plans, estimate = _estimate_planned_schedule(
            args,
            state=state,
            outer_level=definition.outer_level,
            region_level=region_level,
            workers=workers,
            star_counts=star_counts,
            rng=rng,
        )
    _print_summary(
        estimate,
        build_path=build_path,
        outer_level=definition.outer_level,
        region_level=region_level,
        scope=_scope_label(args),
        scheduler=_scheduler_label(args),
    )
    if args.show_active_peak:
        _print_active_peak(estimate)
    if args.csv_output is not None:
        _write_worker_csv(Path(args.csv_output).expanduser(), estimate)
    if args.series_output is not None:
        _write_series_output(
            Path(args.series_output).expanduser(),
            estimate,
            scheduler=_scheduler(args),
            model=_model_label(args),
            random_seed=int(args.random_seed),
            per_worker_gpu_overhead_mb=1024.0
            * float(args.per_worker_gpu_overhead_gib),
            low_latitude_workers=_configured_low_latitude_workers(args, workers),
            ram_model=_execution_ram_model(args),
            memory_limit_mb=args.memory_limit_mb,
            trim_fraction=float(args.trim_fraction),
            pause_fraction=float(args.pause_fraction),
        )
    if args.plot_output is not None:
        active_low_latitude_workers = (
            0
            if args.actual_run_window_timing is not None
            else _active_low_latitude_worker_count(
                plans,
                region_level=region_level,
                low_latitude_deg=float(args.galactic_low_latitude_deg),
            )
        )
        _write_plot(
            Path(args.plot_output).expanduser(),
            estimate,
            per_worker_gpu_overhead_mb=1024.0 * float(args.per_worker_gpu_overhead_gib),
            low_latitude_workers=_configured_low_latitude_workers(args, workers),
            ram_model=_execution_ram_model(args),
            memory_limit_mb=args.memory_limit_mb,
            trim_fraction=args.trim_fraction,
            pause_fraction=args.pause_fraction,
            plot_title=_plot_title(
                args,
                estimate,
                low_latitude_workers=active_low_latitude_workers,
            ),
        )
    if args.throughput_plot_output is not None:
        active_low_latitude_workers = (
            0
            if args.actual_run_window_timing is not None
            else _active_low_latitude_worker_count(
                plans,
                region_level=region_level,
                low_latitude_deg=float(args.galactic_low_latitude_deg),
            )
        )
        _write_throughput_plot(
            Path(args.throughput_plot_output).expanduser(),
            estimate,
            plot_title=_plot_title(
                args,
                estimate,
                low_latitude_workers=active_low_latitude_workers,
            ),
        )
    if args.throughput_comparison_plot_output is not None:
        if not measured_throughput_rows and measured_throughput_curve is None:
            raise ValueError(
                "--throughput-comparison-plot-output requires either "
                "--actual-run-window-throughput or "
                "--throughput-comparison-run-window"
            )
        active_low_latitude_workers = (
            int(args.galactic_low_latitude_workers)
            if args.galactic_low_latitude_workers is not None
            else 0
        )
        if measured_throughput_curve is None:
            measured_throughput_curve = _completion_curve(measured_throughput_rows)
        _write_throughput_comparison_plot(
            Path(args.throughput_comparison_plot_output).expanduser(),
            measured_throughput_curve,
            estimate,
            plot_title=_plot_title(
                args,
                estimate,
                low_latitude_workers=active_low_latitude_workers,
            ),
        )
    if args.ram_comparison_plot_output is not None:
        comparison_window = args.actual_run_window_timing or args.ram_comparison_run_window
        if comparison_window is None:
            raise ValueError(
                "--ram-comparison-plot-output requires either "
                "--actual-run-window-timing or --ram-comparison-run-window"
            )
        _write_actual_ram_comparison_plot(
            Path(args.ram_comparison_plot_output).expanduser(),
            estimate,
            build_path=build_path,
            run_window=comparison_window,
            build_log=args.build_log,
            per_worker_gpu_overhead_mb=1024.0
            * float(args.per_worker_gpu_overhead_gib),
            low_latitude_workers=_configured_low_latitude_workers(args, workers),
            ram_model=_execution_ram_model(args),
            memory_limit_mb=args.memory_limit_mb,
            trim_fraction=args.trim_fraction,
            pause_fraction=args.pause_fraction,
            plot_title=_plot_title(args, estimate, low_latitude_workers=0),
        )


def _estimate_planned_schedule(
    args: argparse.Namespace,
    *,
    state: np.ndarray,
    outer_level: int,
    region_level: int,
    workers: int,
    star_counts: np.ndarray,
    rng: np.random.Generator | None,
):
    runtime_scale = _runtime_scale(args, workers=workers)
    per_worker_gpu_overhead_mb = 1024.0 * float(args.per_worker_gpu_overhead_gib)
    low_latitude_workers = _configured_low_latitude_workers(args, workers)
    scheduler_planning_model = _scheduler_planning_model(args)
    runtime_model = _execution_runtime_model(args)
    ram_model = _execution_ram_model(args)
    if _scheduler_mode(args) == "dynamic":
        return (), estimate_dynamic_schedule(
            state=state,
            outer_level=outer_level,
            region_level=region_level,
            workers=workers,
            star_counts=star_counts,
            per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
            low_latitude_workers=low_latitude_workers,
            memory_limit_mb=args.memory_limit_mb,
            trim_fraction=float(args.trim_fraction),
            pause_fraction=float(args.pause_fraction),
            runtime_scale=runtime_scale,
            scheduler_planning_model=scheduler_planning_model,
            runtime_model=runtime_model,
            ram_model=ram_model,
            runtime_jitter_sigma=float(args.runtime_jitter_sigma),
            memory_jitter_sigma=float(args.memory_jitter_sigma),
            random_seed=int(args.random_seed),
            rng=rng,
        )

    plans = build_regional_worker_plans(
        state=state,
        outer_level=outer_level,
        region_level=region_level,
        workers=workers,
        status_field="traversal_status",
        star_counts=star_counts,
        low_latitude_deg=float(args.galactic_low_latitude_deg),
        low_latitude_workers=args.galactic_low_latitude_workers,
    )
    return plans, estimate_schedule(
        plans,
        star_counts=star_counts,
        per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
        low_latitude_workers=low_latitude_workers,
        runtime_scale=runtime_scale,
        scheduler_planning_model=scheduler_planning_model,
        runtime_model=runtime_model,
        ram_model=ram_model,
        runtime_jitter_sigma=float(args.runtime_jitter_sigma),
        memory_jitter_sigma=float(args.memory_jitter_sigma),
        random_seed=int(args.random_seed),
        rng=rng,
    )


def _configured_low_latitude_workers(args: argparse.Namespace, workers: int) -> int:
    if args.galactic_low_latitude_workers is not None:
        return int(args.galactic_low_latitude_workers)
    if _scheduler_mode(args) == "static":
        return max(1, min(int(workers), int(np.ceil(float(workers) * 2.0 / 3.0))))
    return 0


def _random_generator(args: argparse.Namespace) -> np.random.Generator | None:
    if (
        _execution_runtime_model(args) != "stochastic"
        and float(args.runtime_jitter_sigma) <= 0.0
        and float(args.memory_jitter_sigma) <= 0.0
    ):
        return None
    return np.random.default_rng(int(args.random_seed))


def _runtime_scale(args: argparse.Namespace, *, workers: int) -> float:
    """Return the optional worker-throughput correction for this model."""

    if _execution_runtime_model(args) != "original-deterministic":
        return 1.0
    return throughput_runtime_scale(
        workers=workers,
        device=args.throughput_device,
    )


def _outer_pixel_rng(
    *,
    random_seed: int,
    outer_pix: int,
    salt: int,
) -> np.random.Generator:
    """Return a deterministic RNG for one stochastic outer-pixel quantity."""

    seed = np.random.SeedSequence(
        [int(random_seed), int(outer_pix), int(salt)]
    )
    return np.random.default_rng(seed)


def _scheduler_label(args: argparse.Namespace) -> str:
    scheduler = _scheduler(args)
    if scheduler == "dynamic":
        return "dynamic-ram-bin"
    return "galactic-latitude/production"


def _scheduler(args: argparse.Namespace) -> str:
    return str(args.scheduler)


def _scheduler_mode(args: argparse.Namespace) -> str:
    return "dynamic" if _scheduler(args) == "dynamic" else "static"


def _scheduler_planning_model(args: argparse.Namespace) -> str:
    return "deterministic" if _scheduler(args) == "dynamic" else "original"


def _model_label(args: argparse.Namespace) -> str:
    return "stochastic" if bool(args.stochastic) else "deterministic"


def _execution_runtime_model(args: argparse.Namespace) -> str:
    if bool(args.stochastic):
        return "stochastic"
    if _scheduler_mode(args) == "static":
        return "original-deterministic"
    return "deterministic"


def _execution_ram_model(args: argparse.Namespace) -> str:
    if bool(args.stochastic):
        return "current-rss"
    if _scheduler_mode(args) == "static":
        return "original-current"
    return "high-water"


def estimate_schedule(
    plans,
    *,
    star_counts: np.ndarray,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    runtime_scale: float = 1.0,
    scheduler_planning_model: str = "original",
    runtime_model: str = "deterministic",
    ram_model: str = "high-water",
    runtime_jitter_sigma: float = 0.0,
    memory_jitter_sigma: float = 0.0,
    random_seed: int = 1,
    rng: np.random.Generator | None = None,
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
                scheduler_planning_model=scheduler_planning_model,
                runtime_model=runtime_model,
                ram_model=ram_model,
                runtime_jitter_sigma=runtime_jitter_sigma,
                memory_jitter_sigma=memory_jitter_sigma,
                random_seed=random_seed,
                rng=rng,
                worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
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

    all_rows = tuple(scheduled_rows)
    peak_time_seconds, worker_ram_sum_peak_mb, gpu_overhead_peak_mb, active_at_peak = (
        schedule_ram_peak(
            all_rows,
            worker_count=len(active_plans),
            per_worker_gpu_overhead_mb=0.0,
            low_latitude_workers=low_latitude_workers,
            ram_model=ram_model,
            worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
        )
    )
    ram_overhead_mb = estimate_ram_overhead_mb(len(active_plans), ram_model=ram_model)
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
        worker_ram_includes_gpu=True,
    )


def estimate_dynamic_schedule(
    *,
    state: np.ndarray,
    outer_level: int,
    region_level: int,
    workers: int,
    star_counts: np.ndarray,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    memory_limit_mb: float | None = None,
    trim_fraction: float = 0.85,
    pause_fraction: float = 0.95,
    runtime_scale: float = 1.0,
    scheduler_planning_model: str = "original",
    runtime_model: str = "deterministic",
    ram_model: str = "high-water",
    runtime_jitter_sigma: float = 0.0,
    memory_jitter_sigma: float = 0.0,
    random_seed: int = 1,
    rng: np.random.Generator | None = None,
) -> ScheduleEstimate:
    """Estimate a dynamic schedule with RAM-bin stress mixing."""

    production_schedule = build_dynamic_work_batches(
        state=state,
        outer_level=outer_level,
        workers=workers,
        status_field="traversal_status",
        star_counts=star_counts,
        memory_limit_mb=(
            float(memory_limit_mb) if memory_limit_mb is not None else 0.0
        ),
        trim_fraction=float(trim_fraction),
        worker_ram_overhead_mb=float(per_worker_gpu_overhead_mb),
    )
    worker_count = int(workers)
    if worker_count < 1 or not production_schedule.batches:
        return _empty_schedule_estimate(worker_count=0)

    simulated_batches = _estimate_dynamic_batches(
        production_schedule.batches,
        star_counts=star_counts,
        runtime_scale=runtime_scale,
        scheduler_planning_model=scheduler_planning_model,
        runtime_model=runtime_model,
        ram_model=ram_model,
        runtime_jitter_sigma=runtime_jitter_sigma,
        memory_jitter_sigma=memory_jitter_sigma,
        random_seed=random_seed,
        rng=rng,
        worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
    )
    stress_threshold = float(production_schedule.stress_star_threshold)
    stress_worker_count = int(production_schedule.stress_worker_count)
    ram_bins = _split_batch_ram_bins(
        simulated_batches,
        bin_count=worker_count,
    )
    active: dict[int, PixelEstimate] = {}
    active_batch_vectors: dict[int, tuple[float, float, float]] = {}
    heap: list[tuple[float, int]] = []
    idle_worker_ids = list(range(worker_count))
    scheduled_rows: list[PixelEstimate] = []
    rows_by_worker: dict[int, list[PixelEstimate]] = {
        worker_id: [] for worker_id in range(worker_count)
    }
    active_stress_workers: set[int] = set()
    post_stress_bins: list[list[WorkBatch]] | None = None
    post_stress_assignments: np.ndarray | None = None
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]] = {}
    worker_state_mb = np.zeros(worker_count, dtype=np.float64)
    worker_state_is_stress = np.zeros(worker_count, dtype=bool)
    trim_events: list[WorkerTrimEvent] = []
    if ram_model == "current-rss":
        worker_state_mb[:] = [
            _worker_ram_floor_mb(
                is_stress=False,
                worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
            )
            for _ in range(worker_count)
        ]
    worker_state_time = np.zeros(worker_count, dtype=np.float64)

    def current_worker_state(worker_id: int, now: float) -> float:
        if ram_model == "current-rss":
            current = _decay_worker_state_mb(
                worker_state_mb[int(worker_id)],
                max(float(now) - worker_state_time[int(worker_id)], 0.0),
                is_stress=bool(worker_state_is_stress[int(worker_id)]),
                worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
            )
            active_row = active.get(int(worker_id))
            if active_row is not None:
                current = max(current, float(active_row.worker_ram_mb))
            return current
        return worker_state_mb[int(worker_id)]

    def advance_worker_state(worker_id: int, now: float) -> None:
        worker_state_mb[int(worker_id)] = current_worker_state(worker_id, now)
        worker_state_time[int(worker_id)] = float(now)

    def trim_worker_state(worker_id: int, now: float) -> None:
        worker_id = int(worker_id)
        is_stress_state = bool(worker_state_is_stress[worker_id])
        worker_state_mb[worker_id] = _worker_ram_floor_mb(
            is_stress=is_stress_state,
            worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
        )
        worker_state_time[worker_id] = float(now)
        trim_events.append(
            WorkerTrimEvent(
                worker_id=worker_id,
                seconds=float(now),
                is_stress=is_stress_state,
            )
        )

    def projected_total_ram_mb(worker_id: int, batch: WorkBatch, now: float) -> float:
        candidate_ram_mb = max(row.worker_ram_mb for row in batch.rows)
        worker_ram_mb = 0.0
        for state_worker_id in range(worker_count):
            current_mb = current_worker_state(int(state_worker_id), now)
            if int(state_worker_id) == int(worker_id):
                current_mb = max(current_mb, float(candidate_ram_mb))
            worker_ram_mb += current_mb
        return (
            worker_ram_mb
            + estimate_ram_overhead_mb(worker_count, ram_model=ram_model)
        )

    def exceeds_trim_threshold(worker_id: int, batch: WorkBatch, now: float) -> bool:
        if memory_limit_mb is None or float(memory_limit_mb) <= 0.0:
            return False
        if not active:
            return False
        threshold_mb = float(memory_limit_mb) * float(trim_fraction)
        return projected_total_ram_mb(worker_id, batch, now) > threshold_mb

    def can_assign_without_throttle(
        worker_id: int,
        batch: WorkBatch,
        now: float,
        *,
        must_run: bool,
    ) -> bool:
        if must_run:
            return True
        return not exceeds_trim_threshold(int(worker_id), batch, float(now))

    def take_batch_from_bins(
        bins: list[list[WorkBatch]],
        bin_indexes: list[int],
        *,
        worker_id: int,
        now: float,
        preferred_affinity: tuple[int, int] | None,
        require_stress: bool | None,
        must_run_stress: bool,
    ) -> tuple[WorkBatch | None, int, bool, bool]:
        denied_for_memory = False
        for bin_index in bin_indexes:
            if bin_index < 0 or bin_index >= len(bins):
                continue
            queue = bins[bin_index]
            for batch_index in _ordered_batch_indexes(
                queue,
                preferred_affinity=preferred_affinity,
                active_vectors=active_batch_vectors,
                require_stress=require_stress,
            ):
                candidate = queue[batch_index]
                must_run = bool(candidate.is_stress) and bool(must_run_stress)
                if can_assign_without_throttle(
                    int(worker_id),
                    candidate,
                    float(now),
                    must_run=must_run,
                ):
                    return (
                        queue.pop(batch_index),
                        int(bin_index),
                        bool(candidate.is_stress),
                        denied_for_memory,
                    )
                denied_for_memory = True
        return None, -1, False, denied_for_memory

    def assign_next(worker_id: int, now: float) -> bool:
        nonlocal post_stress_bins, post_stress_assignments
        for active_worker_id in range(worker_count):
            advance_worker_state(active_worker_id, now)
        batch: WorkBatch | None = None
        is_stress = False
        denied_for_memory = False
        stress_remaining = _ram_bins_have_stress(ram_bins)
        non_stress_remaining = _ram_bins_have_non_stress(ram_bins)
        stress_limit = int(stress_worker_count)
        stress_affinity_idle_workers = [
            int(idle_worker_id)
            for idle_worker_id in idle_worker_ids
            if _worker_has_stress_affinity(
                last_affinity_by_worker,
                worker_id=int(idle_worker_id),
                region_level=max(0, outer_level - 1),
                stress_star_threshold=float(stress_threshold),
            )
        ]
        if stress_remaining and len(active_stress_workers) < stress_limit:
            if not stress_affinity_idle_workers or int(worker_id) in stress_affinity_idle_workers:
                stress_bin_index = _highest_stress_bin_index(ram_bins)
                unthrottled_stress_slots = max(int(stress_limit) - 1, 0)
                stress_denied_for_memory = False
                batch, _, is_stress, denied_for_memory = take_batch_from_bins(
                    ram_bins,
                    [stress_bin_index],
                    worker_id=int(worker_id),
                    now=float(now),
                    preferred_affinity=_last_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, outer_level - 1),
                    ),
                    require_stress=True,
                    must_run_stress=(
                        len(active_stress_workers) < unthrottled_stress_slots
                    ),
                )
                stress_denied_for_memory = bool(denied_for_memory)
                if batch is None and stress_denied_for_memory:
                    fallback_bin_indexes = list(range(len(ram_bins)))
                    batch, _, is_stress, fallback_denied_for_memory = (
                        take_batch_from_bins(
                            ram_bins,
                            fallback_bin_indexes,
                            worker_id=int(worker_id),
                            now=float(now),
                            preferred_affinity=None,
                            require_stress=None,
                            must_run_stress=False,
                        )
                    )
                    denied_for_memory = bool(
                        stress_denied_for_memory or fallback_denied_for_memory
                    )
            elif not non_stress_remaining:
                return False
        if batch is None:
            stress_phase_active = _ram_bins_have_stress(ram_bins) or bool(
                active_stress_workers
            )
            if stress_phase_active:
                normal_bin_indexes = [
                    bin_index
                    for bin_index, bin_rows in enumerate(ram_bins)
                    if any(not candidate.is_stress for candidate in bin_rows)
                ]
                batch, _, is_stress, denied_for_memory = take_batch_from_bins(
                    ram_bins,
                    normal_bin_indexes,
                    worker_id=int(worker_id),
                    now=float(now),
                    preferred_affinity=_last_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, outer_level - 3),
                    ),
                    require_stress=False,
                    must_run_stress=False,
                )
            else:
                if post_stress_bins is None:
                    remaining = [
                        queued_batch
                        for bin_rows in ram_bins
                        for queued_batch in bin_rows
                    ]
                    post_stress_bins = _split_batch_bins(
                        remaining,
                        bin_count=worker_count,
                    )
                    post_stress_assignments = np.zeros(
                        len(post_stress_bins),
                        dtype=np.int64,
                    )
                    for bin_rows in ram_bins:
                        bin_rows.clear()
                bin_indexes = _ordered_normal_bin_indexes(
                    post_stress_bins,
                    assignments=post_stress_assignments,
                )
                batch, bin_index, is_stress, denied_for_memory = take_batch_from_bins(
                    post_stress_bins,
                    bin_indexes,
                    worker_id=int(worker_id),
                    now=float(now),
                    preferred_affinity=_last_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, outer_level - 3),
                    ),
                    require_stress=None,
                    must_run_stress=False,
                )
                if batch is not None and bin_index >= 0:
                    post_stress_assignments[bin_index] += 1
        if batch is None:
            if denied_for_memory:
                trim_worker_state(int(worker_id), float(now))
            return False
        scheduled_batch = _schedule_batch_for_worker(
            batch,
            worker_id=int(worker_id),
            start_seconds=float(now),
        )
        scheduled_rows.extend(scheduled_batch)
        rows_by_worker[worker_id].extend(scheduled_batch)
        heaviest_active_row = max(
            scheduled_batch,
            key=lambda scheduled: (scheduled.worker_ram_mb, scheduled.star_count),
        )
        worker_state_mb[int(worker_id)] = max(
            worker_state_mb[int(worker_id)],
            heaviest_active_row.worker_ram_mb,
        )
        worker_state_time[int(worker_id)] = float(now)
        active[worker_id] = heaviest_active_row
        active_batch_vectors[int(worker_id)] = batch.center_vector
        last_affinity_by_worker[(int(worker_id), int(batch.region_level))] = (
            _affinity_parent(
                region_level=int(batch.region_level),
                region_pix=int(batch.region_pix),
            ),
            int(batch.sort_star_count),
        )
        if is_stress:
            active_stress_workers.add(int(worker_id))
        worker_state_is_stress[int(worker_id)] = bool(is_stress)
        heapq.heappush(heap, (scheduled_batch[-1].end_seconds, worker_id))
        return True

    _assign_idle_dynamic_workers(idle_worker_ids, assign_next, now=0.0)

    elapsed_seconds = 0.0
    while heap:
        end_seconds, worker_id = heapq.heappop(heap)
        elapsed_seconds = max(elapsed_seconds, float(end_seconds))
        advance_worker_state(worker_id, float(end_seconds))
        active.pop(worker_id, None)
        active_batch_vectors.pop(int(worker_id), None)
        if int(worker_id) in active_stress_workers:
            active_stress_workers.discard(int(worker_id))
        if int(worker_id) not in idle_worker_ids:
            idle_worker_ids.append(int(worker_id))
        _assign_idle_dynamic_workers(
            idle_worker_ids,
            assign_next,
            now=float(end_seconds),
        )

    all_rows = tuple(scheduled_rows)
    peak_time_seconds, worker_ram_sum_peak_mb, gpu_overhead_peak_mb, active_at_peak = (
        schedule_ram_peak(
            all_rows,
            worker_count=worker_count,
            per_worker_gpu_overhead_mb=0.0,
            low_latitude_workers=low_latitude_workers,
            ram_model=ram_model,
            stress_star_threshold=float(stress_threshold),
            trim_events=tuple(trim_events),
            worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
        )
    )
    workers_summary = tuple(
        WorkerEstimate(
            worker_id=worker_id,
            outer_pixels=len(rows),
            regions=0,
            estimated_star_count=sum(row.star_count for row in rows),
            first_outer_pix=rows[0].outer_pix if rows else -1,
            first_outer_pix_stars=rows[0].star_count if rows else 0,
            peak_worker_ram_mb=max((row.worker_ram_mb for row in rows), default=0.0),
            total_seconds=sum(row.seconds for row in rows),
        )
        for worker_id, rows in sorted(rows_by_worker.items())
        if rows
    )
    heaviest = max(
        all_rows,
        key=lambda row: (row.worker_ram_mb, row.star_count, -row.outer_pix),
    )
    return ScheduleEstimate(
        worker_count=len(workers_summary),
        outer_pixel_count=len(all_rows),
        total_worker_seconds=sum(row.seconds for row in all_rows),
        elapsed_seconds=max((row.end_seconds for row in all_rows), default=elapsed_seconds),
        worker_ram_peak_mb=max(row.worker_ram_mb for row in all_rows),
        worker_ram_sum_peak_mb=worker_ram_sum_peak_mb,
        gpu_overhead_peak_mb=gpu_overhead_peak_mb,
        ram_overhead_mb=estimate_ram_overhead_mb(worker_count, ram_model=ram_model),
        total_ram_peak_mb=(
            estimate_ram_overhead_mb(worker_count, ram_model=ram_model)
            + worker_ram_sum_peak_mb
            + gpu_overhead_peak_mb
        ),
        peak_time_seconds=peak_time_seconds,
        heaviest_outer_pix=heaviest.outer_pix,
        heaviest_outer_pix_stars=heaviest.star_count,
        heaviest_outer_pix_ram_mb=heaviest.worker_ram_mb,
        active_at_peak=active_at_peak,
        workers=workers_summary,
        pixel_rows=all_rows,
        trim_events=tuple(trim_events),
        stress_star_threshold=float(stress_threshold),
        stress_worker_count=int(stress_worker_count),
        worker_ram_includes_gpu=True,
    )


def _empty_schedule_estimate(*, worker_count: int) -> ScheduleEstimate:
    return ScheduleEstimate(
        worker_count=int(worker_count),
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


def estimate_worker_ram_mb(star_count: int) -> float:
    """Return the calibrated current-pixel worker RSS demand."""

    return dynamic_worker_ram_mb(star_count, worker_ram_overhead_mb=0.0)


def estimate_worker_total_ram_mb(
    star_count: int,
    *,
    worker_ram_overhead_mb: float,
) -> float:
    """Return worker memory including per-worker runtime overhead."""

    return estimate_worker_ram_mb(star_count) + float(worker_ram_overhead_mb)


def estimate_original_worker_ram_mb(star_count: int) -> float:
    safe_star_count = max(float(star_count), 0.0)
    return 1024.0 * (
        ORIGINAL_WORKER_RAM_STAR_COEFFICIENT_GIB
        * (safe_star_count ** ORIGINAL_WORKER_RAM_STAR_EXPONENT)
    )


def estimate_ram_overhead_mb(workers: int, *, ram_model: str) -> float:
    """Return the additive RAM overhead for the selected execution model."""

    if ram_model == "original-current":
        overhead_gib = ORIGINAL_RAM_OVERHEAD_INTERCEPT_GIB + (
            ORIGINAL_RAM_OVERHEAD_SLOPE_GIB_PER_WORKER * float(workers)
        )
        return 1024.0 * max(float(overhead_gib), 0.0)
    return 0.0


def estimate_outer_pixel_seconds(star_count: int) -> float:
    """Return the deterministic 8/5 piecewise runtime model in seconds."""

    return dynamic_outer_pixel_seconds(star_count)


def estimate_original_outer_pixel_seconds(star_count: int) -> float:
    safe_star_count = max(float(star_count), 1.0)
    fitted_seconds = (
        ORIGINAL_OUTER_PIXEL_SECONDS_THROUGHPUT_SCALE
        * ORIGINAL_OUTER_PIXEL_SECONDS_COEFFICIENT
        * (safe_star_count ** ORIGINAL_OUTER_PIXEL_SECONDS_STAR_EXPONENT)
    )
    return min(float(fitted_seconds), ORIGINAL_OUTER_PIXEL_SECONDS_DENSE_CAP)


def scheduler_planning_seconds(
    star_count: int,
    *,
    scheduler_planning_model: str,
) -> float:
    if scheduler_planning_model == "original":
        return estimate_original_outer_pixel_seconds(star_count)
    if scheduler_planning_model == "deterministic":
        return estimate_outer_pixel_seconds(star_count)
    raise ValueError(f"unknown scheduler planning model {scheduler_planning_model!r}")


def scheduler_planning_worker_ram_mb(
    star_count: int,
    *,
    scheduler_planning_model: str,
) -> float:
    if scheduler_planning_model == "original":
        return estimate_original_worker_ram_mb(star_count)
    if scheduler_planning_model == "deterministic":
        return estimate_worker_ram_mb(star_count)
    raise ValueError(f"unknown scheduler planning model {scheduler_planning_model!r}")


def stochastic_outer_pixel_seconds(
    star_count: int,
    rng: np.random.Generator,
) -> float:
    """Return one stochastic realization around the deterministic runtime model."""

    model = estimate_outer_pixel_seconds(star_count)
    u = max(float(star_count), 0.0) / 1000.0
    transition_start = DYNAMIC_RUNTIME_TRANSITION_START_STARS / 1000.0
    transition_end = DYNAMIC_RUNTIME_TRANSITION_END_STARS / 1000.0
    if u <= transition_start:
        mu = -0.007
        x = u / transition_start
        sigma = np.exp(-5.044 * x * x + 10.876 * x - 4.401)
        cap = 28.0
    elif u <= transition_end:
        mu = -0.250
        sigma = 3.001
        cap = 28.0
    else:
        mu = 0.002
        x = np.log(max(u / transition_end, 1.0e-9))
        sigma = np.exp(0.0914 * x * x - 0.328 * x + 0.209)
        cap = None
    sigma = float(np.clip(sigma, 0.03, 6.0))
    degrees = 7.0
    noise = (
        float(rng.standard_t(degrees))
        * sigma
        * np.sqrt((degrees - 2.0) / degrees)
    )
    realized = max(0.1, model + mu + noise)
    if cap is not None:
        realized = min(realized, cap)
    return float(realized)


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
    scheduler_planning_model: str = "original",
    runtime_model: str = "deterministic",
    ram_model: str = "high-water",
    runtime_jitter_sigma: float = 0.0,
    memory_jitter_sigma: float = 0.0,
    random_seed: int = 1,
    rng: np.random.Generator | None = None,
    worker_ram_overhead_mb: float = 0.0,
) -> PixelEstimate:
    star_count = int(star_counts[int(outer_pix)])
    if runtime_model == "stochastic":
        seconds = stochastic_outer_pixel_seconds(
            star_count,
            _outer_pixel_rng(
                random_seed=random_seed,
                outer_pix=outer_pix,
                salt=0,
            ),
        )
    elif runtime_model == "original-deterministic":
        seconds = estimate_original_outer_pixel_seconds(star_count)
    elif runtime_model == "deterministic":
        seconds = estimate_outer_pixel_seconds(star_count)
    else:
        raise ValueError(f"unknown runtime model {runtime_model!r}")
    seconds *= float(runtime_scale) * _lognormal_unit_multiplier(
        _outer_pixel_rng(
            random_seed=random_seed,
            outer_pix=outer_pix,
            salt=1,
        ),
        runtime_jitter_sigma,
    )
    planning_seconds = (
        scheduler_planning_seconds(
            star_count,
            scheduler_planning_model=scheduler_planning_model,
        )
        * float(runtime_scale)
    )
    if ram_model == "original-current":
        worker_ram_mb = estimate_original_worker_ram_mb(star_count)
    elif ram_model in ("high-water", "current-rss"):
        worker_ram_mb = estimate_worker_ram_mb(star_count)
    else:
        raise ValueError(f"unknown RAM model {ram_model!r}")
    worker_ram_mb += float(worker_ram_overhead_mb)
    planning_worker_ram_mb = (
        scheduler_planning_worker_ram_mb(
            star_count,
            scheduler_planning_model=scheduler_planning_model,
        )
        + float(worker_ram_overhead_mb)
    )
    return PixelEstimate(
        worker_id=int(worker_id),
        outer_pix=int(outer_pix),
        star_count=star_count,
        worker_ram_mb=worker_ram_mb
        * _lognormal_unit_multiplier(
            _outer_pixel_rng(
                random_seed=random_seed,
                outer_pix=outer_pix,
                salt=2,
            ),
            memory_jitter_sigma,
        ),
        seconds=seconds,
        planning_worker_ram_mb=planning_worker_ram_mb,
        planning_seconds=planning_seconds,
        start_seconds=0.0,
        end_seconds=0.0,
    )


def _lognormal_unit_multiplier(
    rng: np.random.Generator | None,
    sigma: float,
) -> float:
    if rng is None or float(sigma) <= 0.0:
        return 1.0
    sigma_value = float(sigma)
    return float(np.exp(rng.normal(-0.5 * sigma_value * sigma_value, sigma_value)))


def _schedule_row(row: PixelEstimate, *, start_seconds: float) -> PixelEstimate:
    return PixelEstimate(
        worker_id=row.worker_id,
        outer_pix=row.outer_pix,
        star_count=row.star_count,
        worker_ram_mb=row.worker_ram_mb,
        seconds=row.seconds,
        planning_worker_ram_mb=row.planning_worker_ram_mb,
        planning_seconds=row.planning_seconds,
        start_seconds=float(start_seconds),
        end_seconds=float(start_seconds) + float(row.seconds),
    )


def _schedule_row_for_worker(
    row: PixelEstimate,
    *,
    worker_id: int,
    start_seconds: float,
) -> PixelEstimate:
    return PixelEstimate(
        worker_id=int(worker_id),
        outer_pix=row.outer_pix,
        star_count=row.star_count,
        worker_ram_mb=row.worker_ram_mb,
        seconds=row.seconds,
        planning_worker_ram_mb=row.planning_worker_ram_mb,
        planning_seconds=row.planning_seconds,
        start_seconds=float(start_seconds),
        end_seconds=float(start_seconds) + float(row.seconds),
    )


def _schedule_batch_for_worker(
    batch: WorkBatch,
    *,
    worker_id: int,
    start_seconds: float,
) -> tuple[PixelEstimate, ...]:
    rows: list[PixelEstimate] = []
    current_start = float(start_seconds)
    for row in batch.rows:
        scheduled = _schedule_row_for_worker(
            row,
            worker_id=int(worker_id),
            start_seconds=current_start,
        )
        rows.append(scheduled)
        current_start = float(scheduled.end_seconds)
    return tuple(rows)


def _estimate_queue(
    outer_pixs: list[int],
    *,
    star_counts: np.ndarray,
    runtime_scale: float,
    scheduler_planning_model: str,
    runtime_model: str,
    ram_model: str,
    runtime_jitter_sigma: float,
    memory_jitter_sigma: float,
    random_seed: int,
    rng: np.random.Generator | None,
    worker_ram_overhead_mb: float,
) -> list[PixelEstimate]:
    return [
        _pixel_estimate(
            worker_id=-1,
            outer_pix=outer_pix,
            star_counts=star_counts,
            runtime_scale=runtime_scale,
            scheduler_planning_model=scheduler_planning_model,
            runtime_model=runtime_model,
            ram_model=ram_model,
            runtime_jitter_sigma=runtime_jitter_sigma,
            memory_jitter_sigma=memory_jitter_sigma,
            random_seed=random_seed,
            rng=rng,
            worker_ram_overhead_mb=worker_ram_overhead_mb,
        )
        for outer_pix in outer_pixs
    ]


def _estimate_dynamic_batches(
    batches: tuple[TraversalWorkBatch, ...],
    *,
    star_counts: np.ndarray,
    runtime_scale: float,
    scheduler_planning_model: str,
    runtime_model: str,
    ram_model: str,
    runtime_jitter_sigma: float,
    memory_jitter_sigma: float,
    random_seed: int,
    rng: np.random.Generator | None,
    worker_ram_overhead_mb: float,
) -> list[WorkBatch]:
    simulated: list[WorkBatch] = []
    for batch in batches:
        rows = tuple(
            _estimate_queue(
                list(batch.outer_pixs),
                star_counts=star_counts,
                runtime_scale=runtime_scale,
                scheduler_planning_model=scheduler_planning_model,
                runtime_model=runtime_model,
                ram_model=ram_model,
                runtime_jitter_sigma=runtime_jitter_sigma,
                memory_jitter_sigma=memory_jitter_sigma,
                random_seed=random_seed,
                rng=rng,
                worker_ram_overhead_mb=worker_ram_overhead_mb,
            )
        )
        if not rows:
            continue
        simulated.append(
            WorkBatch(
                rows=rows,
                region_level=int(batch.region_level),
                region_pix=int(batch.region_pix),
                is_stress=bool(batch.is_stress),
                sort_star_count=int(batch.sort_star_count),
                planning_seconds=sum(row.planning_seconds for row in rows),
                center_vector=batch.center_vector,
            )
        )
    return simulated


def _split_batch_bins(
    batches: list[WorkBatch],
    *,
    bin_count: int,
) -> list[list[WorkBatch]]:
    safe_bin_count = max(1, int(bin_count))
    if not batches:
        return [[] for _ in range(safe_bin_count)]
    ordered = sorted(
        batches,
        key=lambda batch: (int(batch.sort_star_count), int(batch.region_pix)),
    )
    if safe_bin_count == 1:
        return [ordered]

    runtimes = np.asarray(
        [batch.planning_seconds for batch in ordered],
        dtype=np.float64,
    )
    cumulative = np.cumsum(runtimes)
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    bins: list[list[WorkBatch]] = []
    start = 0
    for bin_index in range(safe_bin_count - 1):
        target = total * float(bin_index + 1) / float(safe_bin_count)
        end = int(np.searchsorted(cumulative, target, side="left")) + 1
        min_remaining = safe_bin_count - bin_index - 1
        end = max(end, start + 1)
        end = min(end, len(ordered) - min_remaining)
        bins.append(ordered[start:end])
        start = end
    bins.append(ordered[start:])
    return bins


def _split_batch_ram_bins(
    batches: list[WorkBatch],
    *,
    bin_count: int,
) -> list[list[WorkBatch]]:
    safe_bin_count = max(1, int(bin_count))
    if not batches:
        return [[] for _ in range(safe_bin_count)]
    ordered = sorted(
        batches,
        key=lambda batch: (int(batch.sort_star_count), int(batch.region_pix)),
    )
    if safe_bin_count == 1:
        return [ordered]

    bins: list[list[WorkBatch]] = []
    for split_rows in np.array_split(np.asarray(ordered, dtype=object), safe_bin_count):
        bins.append([batch for batch in split_rows.tolist()])
    return bins


def _ordered_normal_bin_indexes(
    bins: list[list[WorkBatch]],
    *,
    assignments: np.ndarray,
) -> list[int]:
    if not bins:
        return []
    remaining = np.asarray([len(bin_rows) for bin_rows in bins], dtype=np.float64)
    total_remaining = float(np.sum(remaining))
    if total_remaining <= 0.0:
        return []
    total_assignments = float(np.sum(assignments))
    target_assignments = (total_assignments + 1.0) * remaining / total_remaining
    deficits = target_assignments - np.asarray(assignments, dtype=np.float64)
    deficits[remaining <= 0.0] = -np.inf
    return [
        int(index)
        for index in np.argsort(-deficits)
        if np.isfinite(deficits[int(index)])
    ]


def _ram_bins_have_stress(bins: list[list[WorkBatch]]) -> bool:
    return any(batch.is_stress for bin_rows in bins for batch in bin_rows)


def _ram_bins_have_non_stress(bins: list[list[WorkBatch]]) -> bool:
    return any(not batch.is_stress for bin_rows in bins for batch in bin_rows)


def _highest_stress_bin_index(bins: list[list[WorkBatch]]) -> int:
    for bin_index in range(len(bins) - 1, -1, -1):
        if any(batch.is_stress for batch in bins[bin_index]):
            return bin_index
    return -1


def _affinity_parent(*, region_level: int, region_pix: int) -> int:
    if int(region_level) <= 0:
        return int(region_pix)
    return int(
        get_parent_pixel(
            int(region_level),
            np.asarray([int(region_pix)], dtype=np.int64),
            int(region_level) - 1,
        )[0]
    )


def _last_affinity_for_level(
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]],
    *,
    worker_id: int,
    region_level: int,
) -> tuple[int, int] | None:
    return last_affinity_by_worker.get((int(worker_id), int(region_level)))


def _worker_has_stress_affinity(
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]],
    *,
    worker_id: int,
    region_level: int,
    stress_star_threshold: float,
) -> bool:
    affinity = _last_affinity_for_level(
        last_affinity_by_worker,
        worker_id=int(worker_id),
        region_level=int(region_level),
    )
    if affinity is None:
        return False
    _, star_count = affinity
    return float(star_count) > float(stress_star_threshold)


def _choose_batch_index(
    queue: list[WorkBatch],
    *,
    preferred_affinity: tuple[int, int] | None,
    active_vectors: dict[int, tuple[float, float, float]],
    require_stress: bool | None,
) -> int:
    indexes = _ordered_batch_indexes(
        queue,
        preferred_affinity=preferred_affinity,
        active_vectors=active_vectors,
        require_stress=require_stress,
    )
    return indexes[0] if indexes else -1


def _ordered_batch_indexes(
    queue: list[WorkBatch],
    *,
    preferred_affinity: tuple[int, int] | None,
    active_vectors: dict[int, tuple[float, float, float]],
    require_stress: bool | None,
) -> list[int]:
    candidate_indexes = [
        index
        for index, batch in enumerate(queue)
        if require_stress is None or bool(batch.is_stress) == bool(require_stress)
    ]
    if not candidate_indexes:
        return []
    if preferred_affinity is not None:
        preferred_parent, preferred_star_count = preferred_affinity
        affinity_indexes = [
            index
            for index in candidate_indexes
            if _affinity_parent(
                region_level=int(queue[index].region_level),
                region_pix=int(queue[index].region_pix),
            )
            == int(preferred_parent)
        ]
        if affinity_indexes:
            preferred = sorted(
                affinity_indexes,
                key=lambda index: (
                    abs(int(queue[index].sort_star_count) - int(preferred_star_count)),
                    int(queue[index].region_pix),
                ),
            )
            preferred_set = set(preferred)
            return preferred + [
                index for index in candidate_indexes if index not in preferred_set
            ]
    if not active_vectors:
        return candidate_indexes
    active = np.asarray(list(active_vectors.values()), dtype=np.float64)
    ranked: list[tuple[float, int, int]] = []
    for index in candidate_indexes:
        batch = queue[index]
        vector = np.asarray(batch.center_vector, dtype=np.float64)
        distances = 1.0 - np.clip(active @ vector, -1.0, 1.0)
        min_distance = float(np.min(distances))
        ranked.append((-min_distance, int(batch.region_pix), index))
    return [index for _, _, index in sorted(ranked)]


def _assign_idle_dynamic_workers(
    idle_worker_ids: list[int],
    assign_next,
    *,
    now: float,
) -> None:
    attempts = len(idle_worker_ids)
    for _ in range(attempts):
        if not idle_worker_ids:
            break
        worker_id = idle_worker_ids.pop(0)
        if not assign_next(worker_id, float(now)):
            idle_worker_ids.append(worker_id)

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


def _estimate_actual_run_window(
    build_path: Path,
    *,
    run_window: str,
    build_log: Path | None,
    star_counts: np.ndarray,
    configured_worker_count: int,
    per_worker_gpu_overhead_mb: float,
) -> ScheduleEstimate:
    log_path = (
        Path(build_log).expanduser()
        if build_log is not None
        else build_path / "build.log"
    )
    start = _resolve_run_start_timestamp(log_path, run_window)
    end = _next_run_start_after(log_path, start)
    rows = _actual_run_window_rows(
        log_path,
        start=start,
        end=end,
        star_counts=star_counts,
        worker_ram_overhead_mb=per_worker_gpu_overhead_mb,
    )
    if not rows:
        raise ValueError(f"No completed outer pixels found for run window {run_window!r}")
    worker_ids = sorted({row.worker_id for row in rows})
    rows_by_worker = {
        worker_id: tuple(row for row in rows if row.worker_id == worker_id)
        for worker_id in worker_ids
    }
    workers = tuple(
        WorkerEstimate(
            worker_id=worker_id,
            outer_pixels=len(worker_rows),
            regions=0,
            estimated_star_count=sum(row.star_count for row in worker_rows),
            first_outer_pix=worker_rows[0].outer_pix,
            first_outer_pix_stars=worker_rows[0].star_count,
            peak_worker_ram_mb=max(row.worker_ram_mb for row in worker_rows),
            total_seconds=sum(row.seconds for row in worker_rows),
        )
        for worker_id, worker_rows in rows_by_worker.items()
    )
    peak_time, worker_ram_sum_peak, gpu_overhead_peak, active_at_peak = (
        _actual_interval_peak(
            rows,
            per_worker_gpu_overhead_mb=0.0,
            worker_count=configured_worker_count,
        )
    )
    heaviest = max(rows, key=lambda row: (row.worker_ram_mb, row.star_count, -row.outer_pix))
    return ScheduleEstimate(
        worker_count=int(configured_worker_count),
        outer_pixel_count=len(rows),
        total_worker_seconds=sum(row.seconds for row in rows),
        elapsed_seconds=max(row.end_seconds for row in rows),
        worker_ram_peak_mb=max(row.worker_ram_mb for row in rows),
        worker_ram_sum_peak_mb=worker_ram_sum_peak,
        gpu_overhead_peak_mb=gpu_overhead_peak,
        ram_overhead_mb=0.0,
        total_ram_peak_mb=(
            worker_ram_sum_peak + gpu_overhead_peak
        ),
        peak_time_seconds=peak_time,
        heaviest_outer_pix=heaviest.outer_pix,
        heaviest_outer_pix_stars=heaviest.star_count,
        heaviest_outer_pix_ram_mb=heaviest.worker_ram_mb,
        active_at_peak=active_at_peak,
        workers=workers,
        pixel_rows=tuple(rows),
        worker_ram_includes_gpu=True,
    )


def _actual_run_window_measurement_rows(
    build_path: Path,
    *,
    run_window: str,
    build_log: Path | None,
    star_counts: np.ndarray,
) -> tuple[PixelEstimate, ...]:
    log_path = (
        Path(build_log).expanduser()
        if build_log is not None
        else build_path / "build.log"
    )
    start = _resolve_run_start_timestamp(log_path, run_window)
    end = _next_run_start_after(log_path, start)
    rows = _actual_run_window_rows(
        log_path,
        start=start,
        end=end,
        star_counts=star_counts,
        worker_ram_overhead_mb=0.0,
    )
    if not rows:
        raise ValueError(f"No completed outer pixels found for run window {run_window!r}")
    return rows


def _state_at_run_window_start(
    build_path: Path,
    *,
    completed_rows: tuple[PixelEstimate, ...],
) -> np.ndarray:
    state = load_state(build_path).copy()
    state["traversal_status"][state["traversal_status"] == WORK_STATUS_RUNNING] = (
        WORK_STATUS_PENDING
    )
    completed_outer_pixs = np.asarray(
        tuple({int(row.outer_pix) for row in completed_rows}),
        dtype=np.int64,
    )
    state["traversal_status"][completed_outer_pixs] = WORK_STATUS_PENDING
    return state


def _actual_run_window_rows(
    log_path: Path,
    *,
    start: datetime,
    end: datetime | None,
    star_counts: np.ndarray,
    worker_ram_overhead_mb: float,
) -> tuple[PixelEstimate, ...]:
    starts: dict[tuple[int, int], datetime] = {}
    rows: list[PixelEstimate] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        timestamp = _line_timestamp(line)
        if timestamp is None or timestamp < start:
            continue
        if end is not None and timestamp >= end:
            break
        values = dict(KEY_VALUE_RE.findall(line))
        if " outer_pixel_start " in line:
            try:
                starts[(int(values["worker"]), int(values["outer_pix"]))] = timestamp
            except (KeyError, ValueError):
                continue
            continue
        if " outer_pixel_done " not in line:
            continue
        try:
            worker_id = int(values["worker"])
            outer_pix = int(values["outer_pix"])
        except (KeyError, ValueError):
            continue
        fallback_elapsed = _optional_float(values.get("elapsed_s"))
        row_start = starts.pop((worker_id, outer_pix), None)
        if row_start is None:
            row_start_seconds = (
                max((timestamp - start).total_seconds() - fallback_elapsed, 0.0)
                if fallback_elapsed is not None
                else 0.0
            )
        else:
            row_start_seconds = max((row_start - start).total_seconds(), 0.0)
        row_end_seconds = max((timestamp - start).total_seconds(), row_start_seconds)
        star_count = int(star_counts[outer_pix])
        rows.append(
            PixelEstimate(
                worker_id=worker_id,
                outer_pix=outer_pix,
                star_count=star_count,
                worker_ram_mb=estimate_worker_total_ram_mb(
                    star_count,
                    worker_ram_overhead_mb=worker_ram_overhead_mb,
                ),
                seconds=row_end_seconds - row_start_seconds,
                planning_worker_ram_mb=estimate_worker_total_ram_mb(
                    star_count,
                    worker_ram_overhead_mb=worker_ram_overhead_mb,
                ),
                planning_seconds=row_end_seconds - row_start_seconds,
                start_seconds=row_start_seconds,
                end_seconds=row_end_seconds,
            )
        )
    rows.sort(key=lambda row: (row.start_seconds, row.end_seconds, row.worker_id))
    return tuple(rows)


def _actual_run_window_bounds(
    build_path: Path,
    *,
    run_window: str,
    build_log: Path | None,
) -> tuple[Path, datetime, datetime | None]:
    log_path = (
        Path(build_log).expanduser()
        if build_log is not None
        else build_path / "build.log"
    )
    start = _resolve_run_start_timestamp(log_path, run_window)
    return log_path, start, _next_run_start_after(log_path, start)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _actual_interval_peak(
    rows: tuple[PixelEstimate, ...],
    *,
    per_worker_gpu_overhead_mb: float,
    worker_count: int,
) -> tuple[float, float, float, tuple[PixelEstimate, ...]]:
    events: list[tuple[float, int, PixelEstimate]] = []
    for row in rows:
        events.append((row.start_seconds, 1, row))
        events.append((row.end_seconds, -1, row))
    events.sort(key=lambda event: (event[0], event[1]))
    active: dict[tuple[int, int], PixelEstimate] = {}
    peak_time = 0.0
    peak_worker_ram = 0.0
    active_at_peak: tuple[PixelEstimate, ...] = ()
    index = 0
    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and events[index][0] == event_time:
            _, direction, row = events[index]
            key = (row.worker_id, row.outer_pix)
            if direction < 0:
                active.pop(key, None)
            else:
                active[key] = row
            index += 1
        worker_ram = sum(row.worker_ram_mb for row in active.values())
        if worker_ram > peak_worker_ram:
            peak_time = float(event_time)
            peak_worker_ram = worker_ram
            active_at_peak = tuple(
                active[key] for key in sorted(active, key=lambda item: item[0])
            )
    return (
        peak_time,
        peak_worker_ram,
        float(per_worker_gpu_overhead_mb) * int(worker_count),
        active_at_peak,
    )


def _resolve_run_start_timestamp(log_path: Path, run_start: str) -> datetime:
    if run_start == "latest":
        latest = _latest_run_start(log_path)
        if latest is None:
            raise ValueError(f"No run start found in {log_path}")
        return latest
    return datetime.fromisoformat(run_start.replace("Z", "+00:00"))


def _next_run_start_after(log_path: Path, timestamp: datetime) -> datetime | None:
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " run start " not in line:
            continue
        run_start = _line_timestamp(line)
        if run_start is not None and run_start > timestamp:
            return run_start
    return None


def _latest_run_start(log_path: Path) -> datetime | None:
    latest: datetime | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " run start " not in line:
            continue
        timestamp = _line_timestamp(line)
        if timestamp is not None:
            latest = timestamp
    return latest


def _line_timestamp(line: str) -> datetime | None:
    timestamp_match = LINE_TIME_RE.match(line)
    if timestamp_match is None:
        return None
    try:
        return datetime.fromisoformat(timestamp_match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


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
    if estimate.stress_worker_count > 0:
        print(f"Stress workers: {estimate.stress_worker_count}")
        print(f"Stress threshold: {estimate.stress_star_threshold:.0f} stars")
    print(f"Outer pixels: {estimate.outer_pixel_count}")
    print(f"Estimated elapsed: {estimate.elapsed_seconds:.2f} s")
    print(f"Estimated worker-seconds: {estimate.total_worker_seconds:.2f} s")
    worker_label = (
        "Worker RAM peak (incl. GPU)"
        if estimate.worker_ram_includes_gpu
        else "Worker RAM peak"
    )
    worker_sum_label = (
        "Worker RAM sum high-water (incl. GPU)"
        if estimate.worker_ram_includes_gpu
        else "Worker RAM sum high-water"
    )
    print(f"{worker_label}: {_gib(estimate.worker_ram_peak_mb):.2f} GiB")
    print(f"{worker_sum_label}: {_gib(estimate.worker_ram_sum_peak_mb):.2f} GiB")
    if estimate.gpu_overhead_peak_mb > 0.0:
        print(f"Separate GPU reserve high-water: {_gib(estimate.gpu_overhead_peak_mb):.2f} GiB")
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


def _write_series_output(
    filename: Path,
    estimate: ScheduleEstimate,
    *,
    scheduler: str,
    model: str,
    random_seed: int,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    ram_model: str,
    memory_limit_mb: float | None,
    trim_fraction: float,
    pause_fraction: float,
) -> None:
    """Write comparable RAM and throughput time series for scheduler studies."""

    ram_seconds, _, _, total_ram_mb = build_ram_time_series(
        estimate,
        per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
    )
    throughput_seconds, completed = _completion_curve(estimate.pixel_rows)
    active_seconds, active_workers = _active_worker_count_curve(
        estimate.pixel_rows,
        worker_count=estimate.worker_count,
    )
    metadata = {
        "scheduler": str(scheduler),
        "model": str(model),
        "random_seed": int(random_seed),
        "workers": int(estimate.worker_count),
        "low_latitude_workers": int(low_latitude_workers),
        "stress_workers": int(estimate.stress_worker_count),
        "stress_star_threshold": float(estimate.stress_star_threshold),
        "outer_pixels": int(estimate.outer_pixel_count),
        "elapsed_seconds": float(estimate.elapsed_seconds),
        "peak_total_ram_gib": (
            float(np.max(total_ram_mb)) / 1024.0 if len(total_ram_mb) else 0.0
        ),
        "median_total_ram_gib": (
            float(np.median(total_ram_mb)) / 1024.0 if len(total_ram_mb) else 0.0
        ),
        "throughput_pix_per_s": (
            float(estimate.outer_pixel_count) / float(estimate.elapsed_seconds)
            if float(estimate.elapsed_seconds) > 0.0
            else 0.0
        ),
        "memory_limit_mb": (
            None if memory_limit_mb is None else float(memory_limit_mb)
        ),
        "trim_fraction": float(trim_fraction),
        "pause_fraction": float(pause_fraction),
    }
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filename,
        ram_seconds=np.asarray(ram_seconds, dtype=np.float64),
        total_ram_gib=np.asarray(total_ram_mb, dtype=np.float64) / 1024.0,
        throughput_seconds=np.asarray(throughput_seconds, dtype=np.float64),
        completed=np.asarray(completed, dtype=np.float64),
        active_worker_seconds=np.asarray(active_seconds, dtype=np.float64),
        active_workers=np.asarray(active_workers, dtype=np.float64),
        metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
    )


def build_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int = 0,
    ram_model: str = "high-water",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not estimate.pixel_rows:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, zeros, zeros, zeros
    worker_ram_overhead_mb = (
        float(per_worker_gpu_overhead_mb) if estimate.worker_ram_includes_gpu else 0.0
    )
    times, worker_states = build_worker_state_time_series(
        estimate.pixel_rows,
        worker_count=estimate.worker_count,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
        stress_star_threshold=estimate.stress_star_threshold,
        trim_events=estimate.trim_events,
        worker_ram_overhead_mb=worker_ram_overhead_mb,
    )
    worker_ram_values = np.sum(worker_states, axis=0)
    configured_gpu_mb = (
        0.0
        if estimate.worker_ram_includes_gpu
        else float(per_worker_gpu_overhead_mb) * int(estimate.worker_count)
    )
    gpu_values = np.full(len(times), configured_gpu_mb, dtype=np.float64)
    total_values = float(estimate.ram_overhead_mb) + worker_ram_values + gpu_values

    return (
        times,
        worker_ram_values,
        gpu_values,
        total_values,
    )


def build_worker_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int = 0,
    ram_model: str = "high-water",
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    worker_ram_overhead_mb = (
        float(per_worker_gpu_overhead_mb) if estimate.worker_ram_includes_gpu else 0.0
    )
    times, worker_states = build_worker_state_time_series(
        estimate.pixel_rows,
        worker_count=estimate.worker_count,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
        stress_star_threshold=estimate.stress_star_threshold,
        trim_events=estimate.trim_events,
        worker_ram_overhead_mb=worker_ram_overhead_mb,
    )
    per_worker_gpu = 0.0 if estimate.worker_ram_includes_gpu else float(per_worker_gpu_overhead_mb)
    return tuple(
        (
            worker_id,
            times,
            worker_states[worker_id] + per_worker_gpu,
        )
        for worker_id in range(worker_states.shape[0])
    )


def build_stacked_worker_ram_time_series(
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int = 0,
    ram_model: str = "high-water",
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    if not estimate.pixel_rows:
        return (
            np.asarray([0.0], dtype=np.float64),
            (),
            np.zeros((0, 1), dtype=np.float64),
        )
    worker_ram_overhead_mb = (
        float(per_worker_gpu_overhead_mb) if estimate.worker_ram_includes_gpu else 0.0
    )
    times, worker_states = build_worker_state_time_series(
        estimate.pixel_rows,
        worker_count=estimate.worker_count,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
        stress_star_threshold=estimate.stress_star_threshold,
        trim_events=estimate.trim_events,
        worker_ram_overhead_mb=worker_ram_overhead_mb,
    )
    per_worker_gpu = 0.0 if estimate.worker_ram_includes_gpu else float(per_worker_gpu_overhead_mb)
    worker_values = worker_states + per_worker_gpu
    return times, tuple(range(worker_values.shape[0])), np.sort(worker_values, axis=0)


def build_worker_state_time_series(
    rows: tuple[PixelEstimate, ...],
    *,
    worker_count: int,
    low_latitude_workers: int = 0,
    ram_model: str = "high-water",
    stress_star_threshold: float = float("inf"),
    trim_events: tuple[WorkerTrimEvent, ...] = (),
    worker_ram_overhead_mb: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return modeled worker RSS states over one simulated schedule."""

    if worker_count < 1:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, np.zeros((0, 1), dtype=np.float64)
    if ram_model == "original-current":
        return build_current_worker_state_time_series(rows, worker_count=worker_count)
    starts_by_time: dict[float, list[PixelEstimate]] = {}
    end_times: set[float] = set()
    for row in rows:
        starts_by_time.setdefault(float(row.start_seconds), []).append(row)
        end_times.add(float(row.end_seconds))
    trims_by_time: dict[float, list[WorkerTrimEvent]] = {}
    for event in trim_events:
        trims_by_time.setdefault(float(event.seconds), []).append(event)
    event_times = sorted(set(starts_by_time) | end_times | set(trims_by_time) | {0.0})
    states = np.zeros(int(worker_count), dtype=np.float64)
    state_is_stress = np.zeros(int(worker_count), dtype=bool)
    if ram_model == "current-rss":
        states[:] = [
            _worker_ram_floor_mb(
                is_stress=False,
                worker_ram_overhead_mb=worker_ram_overhead_mb,
            )
            for _ in range(int(worker_count))
        ]
    elif ram_model != "high-water":
        raise ValueError(f"unknown RAM model {ram_model!r}")

    times: list[float] = []
    state_rows: list[np.ndarray] = []
    last_time = 0.0
    for event_time in event_times:
        delta_seconds = max(float(event_time) - float(last_time), 0.0)
        if ram_model == "current-rss" and delta_seconds > 0.0:
            for worker_id in range(int(worker_count)):
                states[worker_id] = _decay_worker_state_mb(
                    states[worker_id],
                    delta_seconds,
                    is_stress=bool(state_is_stress[worker_id]),
                    worker_ram_overhead_mb=worker_ram_overhead_mb,
                )
        if ram_model == "current-rss":
            for event in trims_by_time.get(float(event_time), ()):
                worker_id = int(event.worker_id)
                if 0 <= worker_id < int(worker_count):
                    states[worker_id] = _worker_ram_floor_mb(
                        is_stress=bool(event.is_stress),
                        worker_ram_overhead_mb=worker_ram_overhead_mb,
                    )
                    state_is_stress[worker_id] = bool(event.is_stress)
        for row in starts_by_time.get(float(event_time), ()):
            worker_id = int(row.worker_id)
            if 0 <= worker_id < int(worker_count):
                states[worker_id] = max(states[worker_id], float(row.worker_ram_mb))
                state_is_stress[worker_id] = (
                    float(row.star_count) > float(stress_star_threshold)
                )
        times.append(float(event_time))
        state_rows.append(states.copy())
        last_time = float(event_time)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(state_rows, dtype=np.float64).T,
    )


def build_current_worker_state_time_series(
    rows: tuple[PixelEstimate, ...],
    *,
    worker_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active-pixel worker RAM, matching the original simulator plots."""

    events: list[tuple[float, int, int, float]] = []
    for row in rows:
        events.append((float(row.start_seconds), 1, int(row.worker_id), float(row.worker_ram_mb)))
        events.append((float(row.end_seconds), -1, int(row.worker_id), 0.0))
    if not events:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, np.zeros((int(worker_count), 1), dtype=np.float64)
    events.sort(key=lambda event: (event[0], event[1]))
    worker_values = np.zeros(int(worker_count), dtype=np.float64)
    times: list[float] = [0.0]
    state_rows: list[np.ndarray] = [worker_values.copy()]
    index = 0
    while index < len(events):
        event_time = events[index][0]
        while index < len(events) and events[index][0] == event_time:
            _, direction, worker_id, ram_mb = events[index]
            if 0 <= worker_id < int(worker_count):
                worker_values[worker_id] = 0.0 if direction < 0 else float(ram_mb)
            index += 1
        times.append(float(event_time))
        state_rows.append(worker_values.copy())
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(state_rows, dtype=np.float64).T,
    )


def schedule_ram_peak(
    rows: tuple[PixelEstimate, ...],
    *,
    worker_count: int,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    ram_model: str,
    stress_star_threshold: float = float("inf"),
    trim_events: tuple[WorkerTrimEvent, ...] = (),
    worker_ram_overhead_mb: float = 0.0,
) -> tuple[float, float, float, tuple[PixelEstimate, ...]]:
    """Return high-water RAM and active pixels for the selected RAM model."""

    if not rows:
        return 0.0, 0.0, float(per_worker_gpu_overhead_mb) * int(worker_count), ()
    times, worker_states = build_worker_state_time_series(
        rows,
        worker_count=worker_count,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
        stress_star_threshold=stress_star_threshold,
        trim_events=trim_events,
        worker_ram_overhead_mb=worker_ram_overhead_mb,
    )
    worker_totals = np.sum(worker_states, axis=0)
    peak_index = int(np.argmax(worker_totals))
    peak_time = float(times[peak_index])
    active_at_peak = tuple(
        row
        for row in sorted(rows, key=lambda item: (item.worker_id, item.outer_pix))
        if row.start_seconds <= peak_time <= row.end_seconds
    )
    return (
        peak_time,
        float(worker_totals[peak_index]),
        float(per_worker_gpu_overhead_mb) * int(worker_count),
        active_at_peak,
    )


def _worker_ram_floor_mb(
    *,
    is_stress: bool,
    worker_ram_overhead_mb: float = 0.0,
) -> float:
    if bool(is_stress):
        return 1024.0 * STRESS_PIXEL_RAM_FLOOR_GIB + float(worker_ram_overhead_mb)
    return 1024.0 * NORMAL_PIXEL_RAM_FLOOR_GIB + float(worker_ram_overhead_mb)


def _worker_ram_tau_seconds(*, is_stress: bool) -> float:
    if bool(is_stress):
        return 60.0 * STRESS_PIXEL_RAM_DECAY_TAU_MIN
    return 60.0 * NORMAL_PIXEL_RAM_DECAY_TAU_MIN


def _decay_worker_state_mb(
    state_mb: float,
    delta_seconds: float,
    *,
    is_stress: bool,
    worker_ram_overhead_mb: float = 0.0,
) -> float:
    floor_mb = _worker_ram_floor_mb(
        is_stress=bool(is_stress),
        worker_ram_overhead_mb=worker_ram_overhead_mb,
    )
    tau_seconds = _worker_ram_tau_seconds(is_stress=bool(is_stress))
    if tau_seconds <= 0.0:
        return floor_mb
    return floor_mb + (float(state_mb) - floor_mb) * np.exp(
        -float(delta_seconds) / tau_seconds
    )


def _write_plot(
    filename: Path,
    estimate: ScheduleEstimate,
    *,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    ram_model: str,
    memory_limit_mb: float | None,
    trim_fraction: float,
    pause_fraction: float,
    plot_title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, worker_ids, worker_ram_by_worker_mb = build_stacked_worker_ram_time_series(
        estimate,
        per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
    )
    elapsed_hours = times / 3600.0
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    worker_ram_by_worker_gib = worker_ram_by_worker_mb / 1024.0
    overhead_gib = estimate.ram_overhead_mb / 1024.0
    if overhead_gib > 0.0:
        ax.fill_between(
            elapsed_hours,
            np.zeros(len(elapsed_hours), dtype=np.float64),
            np.full(len(elapsed_hours), overhead_gib, dtype=np.float64),
            step="post",
            color="0.80",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )
    baseline = np.full(len(elapsed_hours), overhead_gib, dtype=np.float64)
    colors = plt.get_cmap("tab10").colors
    for index, worker_id in enumerate(worker_ids):
        worker_gib = worker_ram_by_worker_gib[index]
        top = baseline + worker_gib
        ax.fill_between(
            elapsed_hours,
            baseline,
            top,
            step="post",
            color=colors[index % len(colors)],
            alpha=0.82,
            linewidth=0,
            zorder=2,
        )
        baseline = top
    if memory_limit_mb is not None and float(memory_limit_mb) > 0.0:
        memory_limit_gib = float(memory_limit_mb) / 1024.0
        ax.axhline(
            memory_limit_gib,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )
        ax.axhline(
            memory_limit_gib * float(trim_fraction),
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )
        ax.axhline(
            memory_limit_gib * float(pause_fraction),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xlabel("Simulated Elapsed Time [h]")
    ax.set_ylabel("Estimated RAM [GiB]")
    title = plot_title or (
        f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    )
    ax.set_title(title)
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _write_actual_ram_comparison_plot(
    filename: Path,
    estimate: ScheduleEstimate,
    *,
    build_path: Path,
    run_window: str,
    build_log: Path | None,
    per_worker_gpu_overhead_mb: float,
    low_latitude_workers: int,
    ram_model: str,
    memory_limit_mb: float | None,
    trim_fraction: float,
    pause_fraction: float,
    plot_title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path, start, end = _actual_run_window_bounds(
        build_path,
        run_window=run_window,
        build_log=build_log,
    )
    measured_seconds, measured_ram_mb = _measured_total_ram_samples(
        log_path,
        start=start,
        end=end,
    )
    if len(measured_seconds) == 0:
        raise ValueError(f"No memory samples found in {log_path} for {run_window!r}")

    simulated_seconds, _, _, simulated_ram_mb = build_ram_time_series(
        estimate,
        per_worker_gpu_overhead_mb=per_worker_gpu_overhead_mb,
        low_latitude_workers=low_latitude_workers,
        ram_model=ram_model,
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    ax.plot(
        measured_seconds / 3600.0,
        measured_ram_mb / 1024.0,
        color="black",
        linewidth=1.5,
        label="Measured",
    )
    ax.plot(
        simulated_seconds / 3600.0,
        simulated_ram_mb / 1024.0,
        color="tab:blue",
        linewidth=1.3,
        label="Simulated",
    )
    if memory_limit_mb is not None and float(memory_limit_mb) > 0.0:
        memory_limit_gib = float(memory_limit_mb) / 1024.0
        ax.axhline(
            memory_limit_gib,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            memory_limit_gib * float(trim_fraction),
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            memory_limit_gib * float(pause_fraction),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
    measured_peak = float(np.max(measured_ram_mb)) / 1024.0
    simulated_peak = float(np.max(simulated_ram_mb)) / 1024.0
    ax.text(
        0.03,
        0.92,
        f"Peak gap: {measured_peak - simulated_peak:.1f} GiB",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25",
        },
    )
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Total RAM [GiB]")
    title = plot_title or (
        f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    )
    ax.set_title(f"{title} RAM Comparison")
    ax.set_xlim(left=0.0, right=float(np.max(measured_seconds)) / 3600.0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _measured_total_ram_samples(
    log_path: Path,
    *,
    start: datetime,
    end: datetime | None,
) -> tuple[np.ndarray, np.ndarray]:
    seconds: list[float] = []
    total_mb: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        timestamp = _line_timestamp(line)
        if timestamp is None or timestamp < start:
            continue
        if end is not None and timestamp >= end:
            break
        if " memory_pressure " not in line and " memory_snapshot " not in line:
            continue
        values = dict(KEY_VALUE_RE.findall(line))
        try:
            total_value = float(values["total_mb"])
        except (KeyError, ValueError):
            continue
        seconds.append((timestamp - start).total_seconds())
        total_mb.append(total_value)
    return (
        np.asarray(seconds, dtype=np.float64),
        np.asarray(total_mb, dtype=np.float64),
    )


def _measured_progress_curve(
    build_path: Path,
    *,
    run_window: str,
    build_log: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    log_path, start, end = _actual_run_window_bounds(
        build_path,
        run_window=run_window,
        build_log=build_log,
    )
    completed_by_worker: dict[int, int] = {}
    seconds: list[float] = [0.0]
    completed: list[float] = [0.0]
    for line in log_path.read_text(encoding="utf-8").splitlines():
        timestamp = _line_timestamp(line)
        if timestamp is None or timestamp < start:
            continue
        if end is not None and timestamp >= end:
            break
        if " progress " not in line and " profile " not in line:
            continue
        values = dict(KEY_VALUE_RE.findall(line))
        try:
            worker_id = int(values["worker"])
            worker_completed = int(values["completed"])
        except (KeyError, ValueError):
            continue
        completed_by_worker[worker_id] = worker_completed
        total_completed = float(sum(completed_by_worker.values()))
        elapsed_seconds = max((timestamp - start).total_seconds(), 0.0)
        if total_completed < completed[-1]:
            continue
        if total_completed == completed[-1]:
            seconds[-1] = elapsed_seconds
        else:
            seconds.append(elapsed_seconds)
            completed.append(total_completed)
    if len(completed) <= 1:
        raise ValueError(f"No progress samples found in {log_path} for {run_window!r}")
    return (
        np.asarray(seconds, dtype=np.float64),
        np.asarray(completed, dtype=np.float64),
    )


def _write_throughput_plot(
    filename: Path,
    estimate: ScheduleEstimate,
    *,
    plot_title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not estimate.pixel_rows:
        end_seconds = np.asarray([0.0], dtype=np.float64)
        completed = np.asarray([0.0], dtype=np.float64)
    else:
        end_seconds = np.sort(
            np.asarray(
                [row.end_seconds for row in estimate.pixel_rows],
                dtype=np.float64,
            )
        )
        completed = np.arange(1, len(end_seconds) + 1, dtype=np.float64)
        end_seconds = np.concatenate(([0.0], end_seconds))
        completed = np.concatenate(([0.0], completed))

    elapsed_hours = end_seconds / 3600.0
    slope = (
        float(completed[-1]) / float(end_seconds[-1])
        if float(end_seconds[-1]) > 0.0
        else 0.0
    )
    average_line = slope * end_seconds
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    ax.plot(
        elapsed_hours,
        completed,
        color="0.20",
        linewidth=1.6,
    )
    ax.plot(
        elapsed_hours,
        average_line,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax.text(
        0.03,
        0.92,
        f"Average: {slope:.1f} pix/s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25",
        },
    )
    title = plot_title or (
        f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    )
    ax.set_title(f"{title} Throughput")
    ax.set_xlabel("Simulated Elapsed Time [h]")
    ax.set_ylabel("Completed Outer Pixels")
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _write_throughput_comparison_plot(
    filename: Path,
    measured_curve: tuple[np.ndarray, np.ndarray],
    estimate: ScheduleEstimate,
    *,
    plot_title: str | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured_seconds, measured_completed = measured_curve
    simulated_seconds, simulated_completed = _completion_curve(
        estimate.pixel_rows,
        limit=int(measured_completed[-1]),
    )
    measured_rate = _completion_rate(measured_seconds, measured_completed)
    simulated_rate = _completion_rate(simulated_seconds, simulated_completed)
    rate_gap = (
        (simulated_rate / measured_rate - 1.0) * 100.0
        if measured_rate > 0.0
        else 0.0
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    ax.plot(
        measured_seconds / 3600.0,
        measured_completed,
        color="black",
        linewidth=1.6,
        label="Measured",
    )
    ax.plot(
        simulated_seconds / 3600.0,
        simulated_completed,
        color="tab:blue",
        linewidth=1.4,
        label="Simulated",
    )
    if measured_seconds[-1] > 0.0:
        ax.plot(
            measured_seconds / 3600.0,
            measured_rate * measured_seconds,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
    if simulated_seconds[-1] > 0.0:
        ax.plot(
            simulated_seconds / 3600.0,
            simulated_rate * simulated_seconds,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
        )
    ax.text(
        0.03,
        0.92,
        (
            f"Measured: {measured_rate:.2f} pix/s\n"
            f"Simulated: {simulated_rate:.2f} pix/s\n"
            f"Gap: {rate_gap:+.0f}%"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.85,
            "boxstyle": "round,pad=0.25",
        },
    )
    title = plot_title or (
        f"{estimate.outer_pixel_count}-Pixels with {estimate.worker_count} Workers"
    )
    if estimate.outer_pixel_count > int(measured_completed[-1]):
        title = f"First {int(measured_completed[-1])} of {title}"
    ax.set_title(f"{title} Throughput Comparison")
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Completed Outer Pixels")
    ax.set_xlim(
        left=0.0,
        right=max(float(measured_seconds[-1]), float(simulated_seconds[-1])) / 3600.0,
    )
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _completion_curve(
    rows: tuple[PixelEstimate, ...],
    *,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, zeros
    end_seconds = np.sort(
        np.asarray([row.end_seconds for row in rows], dtype=np.float64)
    )
    if limit is not None:
        end_seconds = end_seconds[: int(limit)]
    completed = np.arange(1, len(end_seconds) + 1, dtype=np.float64)
    return (
        np.concatenate(([0.0], end_seconds)),
        np.concatenate(([0.0], completed)),
    )


def _completion_rate(seconds: np.ndarray, completed: np.ndarray) -> float:
    if len(seconds) == 0 or float(seconds[-1]) <= 0.0:
        return 0.0
    return float(completed[-1]) / float(seconds[-1])


def _active_worker_count_curve(
    rows: tuple[PixelEstimate, ...],
    *,
    worker_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        zeros = np.asarray([0.0], dtype=np.float64)
        return zeros, zeros
    events: list[tuple[float, int, int]] = []
    for row in rows:
        events.append((float(row.start_seconds), 1, int(row.worker_id)))
        events.append((float(row.end_seconds), -1, int(row.worker_id)))
    events.sort(key=lambda event: (event[0], event[1]))
    active = np.zeros(int(worker_count), dtype=bool)
    times: list[float] = [0.0]
    counts: list[float] = [0.0]
    index = 0
    while index < len(events):
        event_time = float(events[index][0])
        while index < len(events) and float(events[index][0]) == event_time:
            _, direction, worker_id = events[index]
            if 0 <= worker_id < int(worker_count):
                active[worker_id] = direction > 0
            index += 1
        times.append(event_time)
        counts.append(float(np.count_nonzero(active)))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


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
    if _scheduler_mode(args) == "dynamic":
        return f"{title} [Dynamic RAM Bins]"
    if low_latitude_workers > 0:
        title = f"{title} [{low_latitude_workers} Low-Lat Workers]"
    return title


def _scope_label(args: argparse.Namespace) -> str:
    if args.actual_run_window_timing is not None:
        return f"actual-run-window-timing:{args.actual_run_window_timing}"
    if args.actual_run_window_throughput is not None:
        return f"actual-run-window-throughput:{args.actual_run_window_throughput}"
    if args.throughput_comparison_run_window is not None:
        return f"throughput-comparison-run-window:{args.throughput_comparison_run_window}"
    if args.schedule_full_sky:
        return "full-sky-config"
    return str(args.scope)


def _gib(mb: float) -> float:
    return float(mb) / 1024.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_path", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--region-level", type=int)
    parser.add_argument(
        "--scheduler",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "Scheduler policy: 'static' uses the production regional planner "
            "and 'dynamic' uses the production dynamic batch planner."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Use fitted stochastic runtime residuals and current-RSS memory "
            "decay. Without this flag, static uses the production deterministic "
            "model and dynamic uses the calibrated deterministic model."
        ),
    )
    parser.add_argument(
        "--runtime-jitter-sigma",
        type=float,
        default=0.0,
        help="Optional extra lognormal runtime jitter sigma; deterministic when 0.",
    )
    parser.add_argument(
        "--memory-jitter-sigma",
        type=float,
        default=0.0,
        help="Optional lognormal worker-RAM jitter sigma; deterministic when 0.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Seed for runtime/RAM jitter.",
    )
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
        "--actual-run-window-timing",
        metavar="TIMESTAMP|latest",
        help=(
            "Replay the actual completed outer-pixel intervals from a build-log "
            "run window and apply the RAM model to those intervals. Use this "
            "for same-pixel, same-timing RAM calibration."
        ),
    )
    parser.add_argument(
        "--actual-run-window-throughput",
        metavar="TIMESTAMP|latest",
        help=(
            "Reconstruct the pending Traversal state at a build-log run "
            "window start, simulate the full remaining schedule, and compare "
            "the first N simulated completions with the N completions measured "
            "in that window."
        ),
    )
    parser.add_argument(
        "--build-log",
        type=Path,
        help=(
            "Build log to use with actual run-window modes. Defaults to "
            "<build_path>/build.log."
        ),
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
        help="Optional GPU overhead to include in each worker memory estimate.",
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
    parser.add_argument(
        "--series-output",
        type=Path,
        help=(
            "Optional .npz output containing simulated total-RAM and "
            "throughput series for scheduler comparison plots."
        ),
    )
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--throughput-plot-output", type=Path)
    parser.add_argument(
        "--throughput-comparison-run-window",
        metavar="TIMESTAMP|latest",
        help=(
            "Build-log run window whose progress checkpoints should be "
            "overlaid on the simulated throughput curve. Use this for older "
            "logs that do not have outer_pixel_done events."
        ),
    )
    parser.add_argument(
        "--throughput-comparison-plot-output",
        type=Path,
        help=(
            "Optional measured-versus-simulated throughput plot. Requires "
            "--actual-run-window-throughput or "
            "--throughput-comparison-run-window."
        ),
    )
    parser.add_argument(
        "--ram-comparison-plot-output",
        type=Path,
        help=(
            "Optional measured-versus-simulated total-RAM plot. Requires "
            "--actual-run-window-timing or --ram-comparison-run-window."
        ),
    )
    parser.add_argument(
        "--ram-comparison-run-window",
        metavar="TIMESTAMP|latest",
        help=(
            "Build-log run window to overlay on the simulated RAM curve when "
            "the simulation is not using --actual-run-window-timing."
        ),
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        help="Optional total-RAM ceiling used to draw trim/pause threshold lines.",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.80)
    parser.add_argument("--pause-fraction", type=float, default=0.90)
    parser.add_argument("--plot-title")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.runtime_jitter_sigma < 0.0:
        raise ValueError("--runtime-jitter-sigma must be non-negative")
    if args.memory_jitter_sigma < 0.0:
        raise ValueError("--memory-jitter-sigma must be non-negative")
    if (
        _scheduler_mode(args) == "dynamic"
        and args.galactic_low_latitude_workers is not None
    ):
        raise ValueError(
            "--galactic-low-latitude-workers only applies to static schedulers"
        )
    if (
        args.actual_run_window_throughput is not None
        and args.throughput_comparison_run_window is not None
    ):
        raise ValueError(
            "--actual-run-window-throughput cannot be combined with "
            "--throughput-comparison-run-window"
        )
    if (
        args.throughput_comparison_plot_output is not None
        and args.actual_run_window_throughput is None
        and args.throughput_comparison_run_window is None
    ):
        raise ValueError(
            "--throughput-comparison-plot-output requires either "
            "--actual-run-window-throughput or --throughput-comparison-run-window"
        )
    if (
        args.actual_run_window_timing is not None
        and args.actual_run_window_throughput is not None
    ):
        raise ValueError(
            "--actual-run-window-timing cannot be combined with "
            "--actual-run-window-throughput"
        )
    if args.actual_run_window_timing is not None:
        if args.sample_pixels is not None:
            raise ValueError(
                "--actual-run-window-timing cannot be combined with --sample-pixels"
            )
        if args.schedule_full_sky:
            raise ValueError(
                "--actual-run-window-timing cannot be combined with --schedule-full-sky"
            )
        if args.scope != "all":
            raise ValueError(
                "--actual-run-window-timing cannot be combined with --scope remaining"
            )
        return
    if args.actual_run_window_throughput is not None:
        if args.sample_pixels is not None:
            raise ValueError(
                "--actual-run-window-throughput cannot be combined with --sample-pixels"
            )
        if args.schedule_full_sky:
            raise ValueError(
                "--actual-run-window-throughput cannot be combined with "
                "--schedule-full-sky"
            )
        if args.scope != "all":
            raise ValueError(
                "--actual-run-window-throughput cannot be combined with "
                "--scope remaining"
            )
        return


if __name__ == "__main__":
    main()
