"""Compare exported scheduler simulation RAM and throughput series."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_build_memory_timeline import (
    TimeWindow,
    parse_memory_samples,
    parse_optional_datetime,
    parse_progress_samples,
)


ACTIVE_WORKER_MODE_BIN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    ram_seconds: np.ndarray
    total_ram_gib: np.ndarray
    throughput_seconds: np.ndarray
    completed: np.ndarray
    active_worker_seconds: np.ndarray
    active_workers: np.ndarray
    metadata: dict


@dataclass(frozen=True, slots=True)
class MeasuredRun:
    label: str
    window: TimeWindow


def main() -> None:
    args = _parse_args()
    series = tuple(_load_series(item) for item in args.series)
    measured_run = (
        None if args.measured_run is None else _parse_measured_run(args.measured_run)
    )
    if args.limit_to_measured_run:
        if measured_run is None or measured_run.window.until is None:
            raise ValueError(
                "--limit-to-measured-run requires --measured-run with an UNTIL timestamp"
            )
        max_seconds = (
            measured_run.window.until - measured_run.window.since
        ).total_seconds()
        series = tuple(_truncate_series(item, max_seconds) for item in series)
    build_log = None if args.build_log is None else Path(args.build_log).expanduser()
    if args.ram_output is not None:
        _write_ram_plot(
            Path(args.ram_output).expanduser(),
            series,
            measured_run=measured_run,
            build_log=build_log,
            title_prefix=args.title_prefix,
        )
    if args.throughput_output is not None:
        _write_throughput_plot(
            Path(args.throughput_output).expanduser(),
            series,
            measured_run=measured_run,
            build_log=build_log,
            title_prefix=args.title_prefix,
        )


def _load_series(argument: str) -> Series:
    if "=" not in argument:
        raise ValueError("--series values must be LABEL=PATH")
    label, raw_path = argument.split("=", 1)
    path = Path(raw_path).expanduser()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        active_worker_seconds = (
            np.asarray(data["active_worker_seconds"], dtype=np.float64)
            if "active_worker_seconds" in data.files
            else np.asarray([], dtype=np.float64)
        )
        active_workers = (
            np.asarray(data["active_workers"], dtype=np.float64)
            if "active_workers" in data.files
            else np.asarray([], dtype=np.float64)
        )
        return Series(
            label=label,
            ram_seconds=np.asarray(data["ram_seconds"], dtype=np.float64),
            total_ram_gib=np.asarray(data["total_ram_gib"], dtype=np.float64),
            throughput_seconds=np.asarray(
                data["throughput_seconds"],
                dtype=np.float64,
            ),
            completed=np.asarray(data["completed"], dtype=np.float64),
            active_worker_seconds=active_worker_seconds,
            active_workers=active_workers,
            metadata=metadata,
        )


def _parse_measured_run(argument: str) -> MeasuredRun:
    parts = [part.strip() for part in argument.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(
            "--measured-run must be LABEL,SINCE or LABEL,SINCE,UNTIL, "
            f"got {argument!r}"
        )
    until = parse_optional_datetime(parts[2]) if len(parts) == 3 and parts[2] else None
    return MeasuredRun(
        label=parts[0],
        window=TimeWindow(
            since=parse_optional_datetime(parts[1]),
            until=until,
        ),
    )


def _truncate_series(series: Series, max_seconds: float) -> Series:
    """Return a copy of one series clipped to a fixed elapsed duration."""

    if max_seconds <= 0.0:
        raise ValueError("measured run duration must be positive")

    ram_seconds, total_ram_gib = _truncate_xy(
        series.ram_seconds,
        series.total_ram_gib,
        max_seconds,
    )
    throughput_seconds, completed = _truncate_xy(
        series.throughput_seconds,
        series.completed,
        max_seconds,
    )
    active_worker_seconds, active_workers = _truncate_xy(
        series.active_worker_seconds,
        series.active_workers,
        max_seconds,
        step=True,
    )
    return Series(
        label=series.label,
        ram_seconds=ram_seconds,
        total_ram_gib=total_ram_gib,
        throughput_seconds=throughput_seconds,
        completed=completed,
        active_worker_seconds=active_worker_seconds,
        active_workers=active_workers,
        metadata=series.metadata,
    )


def _truncate_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_x: float,
    *,
    step: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = x <= max_x
    clipped_x = x[mask]
    clipped_y = y[mask]

    if len(clipped_x) == 0:
        if step:
            endpoint_y = y[0]
        else:
            endpoint_y = np.interp(max_x, x, y)
        return np.asarray([0.0, max_x]), np.asarray([endpoint_y, endpoint_y])

    if clipped_x[-1] < max_x:
        if step:
            endpoint_y = clipped_y[-1]
        else:
            endpoint_y = np.interp(max_x, x, y)
        clipped_x = np.append(clipped_x, max_x)
        clipped_y = np.append(clipped_y, endpoint_y)

    return clipped_x, clipped_y


def _write_ram_plot(
    filename: Path,
    series: tuple[Series, ...],
    *,
    measured_run: MeasuredRun | None,
    build_log: Path | None,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    for item in series:
        (line,) = ax.plot(
            item.ram_seconds / 3600.0,
            item.total_ram_gib,
            linewidth=1.5,
            label=f"{item.label}: {np.median(item.total_ram_gib):.1f} GiB",
        )
        ax.axhline(
            float(np.median(item.total_ram_gib)),
            color=line.get_color(),
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
    if measured_run is not None:
        if build_log is None:
            raise ValueError("--measured-run requires --build-log")
        samples = parse_memory_samples(build_log, measured_run.window)
        if not samples:
            raise ValueError(
                f"No memory samples found for measured run {measured_run.label!r}"
            )
        start = samples[0].timestamp
        elapsed_hours = np.asarray(
            [
                (sample.timestamp - start).total_seconds() / 3600.0
                for sample in samples
            ],
            dtype=np.float64,
        )
        total_gib = np.asarray(
            [sample.total_mb / 1024.0 for sample in samples],
            dtype=np.float64,
        )
        ax.plot(
            elapsed_hours,
            total_gib,
            color="black",
            linewidth=1.7,
            label=f"{measured_run.label}: {np.median(total_gib):.1f} GiB",
        )
        ax.axhline(
            float(np.median(total_gib)),
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
    ax.set_title(f"{title_prefix} RAM Comparison")
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Total RAM [GiB]")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
    )
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _write_throughput_plot(
    filename: Path,
    series: tuple[Series, ...],
    *,
    measured_run: MeasuredRun | None,
    build_log: Path | None,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    ax_cores = ax.twinx()
    ax.set_zorder(ax_cores.get_zorder() + 1)
    ax.patch.set_visible(False)
    plotted_cores = False
    max_active_workers = 0
    for item in series:
        rate = _completion_rate(item.throughput_seconds, item.completed)
        (line,) = ax.plot(
            item.throughput_seconds / 3600.0,
            item.completed,
            linewidth=1.5,
            label=f"{item.label}: {rate:.2f} pix/s",
        )
        ax.plot(
            [0.0, float(item.throughput_seconds[-1]) / 3600.0],
            [0.0, float(item.completed[-1])],
            color=line.get_color(),
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        if (
            str(item.metadata.get("scheduler", "")) == "dynamic"
            and len(item.active_worker_seconds)
            and len(item.active_workers)
        ):
            active_worker_seconds, active_workers = _mode_filter_step_series(
                item.active_worker_seconds,
                item.active_workers,
                bin_seconds=ACTIVE_WORKER_MODE_BIN_SECONDS,
            )
            ax_cores.step(
                active_worker_seconds / 3600.0,
                active_workers,
                where="post",
                color=line.get_color(),
                linestyle="-",
                linewidth=0.8,
                alpha=0.65,
                zorder=1,
            )
            plotted_cores = True
            max_active_workers = max(
                max_active_workers,
                int(np.max(active_workers)) if len(active_workers) else 0,
            )
    if measured_run is not None:
        if build_log is None:
            raise ValueError("--measured-run requires --build-log")
        samples = parse_progress_samples(build_log, measured_run.window)
        if not samples:
            raise ValueError(
                f"No progress samples found for measured run {measured_run.label!r}"
            )
        start = samples[0].timestamp
        elapsed_hours = np.asarray(
            [
                (sample.timestamp - start).total_seconds() / 3600.0
                for sample in samples
            ],
            dtype=np.float64,
        )
        completed = np.asarray(
            [sample.completed for sample in samples],
            dtype=np.float64,
        )
        rate = 0.0
        if elapsed_hours[-1] > 0.0:
            rate = float(completed[-1]) / float(elapsed_hours[-1] * 3600.0)
        ax.plot(
            elapsed_hours,
            completed,
            color="black",
            linewidth=1.7,
            label=f"{measured_run.label}: {rate:.2f} pix/s",
        )
        if elapsed_hours[-1] > 0.0:
            ax.plot(
                [0.0, elapsed_hours[-1]],
                [0.0, completed[-1]],
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )
        active_seconds, active_workers = parse_active_worker_samples(
            build_log,
            measured_run.window,
        )
        if len(active_seconds) and len(active_workers):
            active_seconds, active_workers = _mode_filter_step_series(
                active_seconds,
                active_workers,
                bin_seconds=ACTIVE_WORKER_MODE_BIN_SECONDS,
            )
            ax_cores.step(
                active_seconds / 3600.0,
                active_workers,
                where="post",
                color="black",
                linestyle="-",
                linewidth=0.8,
                alpha=0.65,
                zorder=1,
            )
            plotted_cores = True
            max_active_workers = max(max_active_workers, int(np.max(active_workers)))
    ax.set_title(f"{title_prefix} Throughput Comparison")
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Completed Outer Pixels")
    if plotted_cores:
        ax_cores.set_ylim(0.0, max_active_workers + 0.5)
        ax_cores.set_ylabel("Active Workers")
    else:
        ax_cores.set_visible(False)
    ax.grid(True, color="0.88", linewidth=0.8)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def _completion_rate(seconds: np.ndarray, completed: np.ndarray) -> float:
    if len(seconds) == 0 or float(seconds[-1]) <= 0.0:
        return 0.0
    return float(completed[-1]) / float(seconds[-1])


def _mode_filter_step_series(
    seconds: np.ndarray,
    values: np.ndarray,
    *,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer modal values for fixed-width windows of a step series."""

    if len(seconds) <= 2:
        return seconds, values

    seconds = np.asarray(seconds, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    duration = float(seconds[-1])
    if duration <= float(seconds[0]) or bin_seconds <= 0.0:
        return seconds, values

    bin_edges = np.arange(0.0, duration + bin_seconds, bin_seconds)
    if bin_edges[-1] < duration:
        bin_edges = np.append(bin_edges, duration)
    bin_edges[-1] = duration

    modal_seconds: list[float] = []
    modal_values: list[float] = []
    segment_index = 0
    for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:]):
        if bin_end <= bin_start:
            continue

        while (
            segment_index + 1 < len(seconds)
            and seconds[segment_index + 1] <= bin_start
        ):
            segment_index += 1

        cursor = float(bin_start)
        local_index = segment_index
        durations_by_value: dict[float, float] = {}
        while cursor < bin_end:
            next_change = (
                float(seconds[local_index + 1])
                if local_index + 1 < len(seconds)
                else float(bin_end)
            )
            segment_end = min(float(bin_end), next_change)
            if segment_end > cursor:
                value = float(values[local_index])
                durations_by_value[value] = (
                    durations_by_value.get(value, 0.0) + segment_end - cursor
                )
                cursor = segment_end
            if local_index + 1 < len(seconds) and cursor >= seconds[local_index + 1]:
                local_index += 1
            else:
                break

        if not durations_by_value:
            continue
        mode_value = max(durations_by_value.items(), key=lambda item: item[1])[0]
        if modal_values and mode_value == modal_values[-1]:
            continue
        modal_seconds.append(float(bin_start))
        modal_values.append(mode_value)

    if not modal_seconds:
        return seconds, values
    if modal_seconds[0] > 0.0:
        modal_seconds.insert(0, 0.0)
        modal_values.insert(0, modal_values[0])
    if modal_seconds[-1] < duration:
        modal_seconds.append(duration)
        modal_values.append(modal_values[-1])

    return np.asarray(modal_seconds), np.asarray(modal_values)


def parse_active_worker_samples(
    log_path: Path,
    window: TimeWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct active worker count from outer-pixel start/done log events."""

    run_re = re.compile(r"^(\S+) run start")
    start_re = re.compile(r"^(\S+) phase=traversal worker=(\d+) outer_pixel_start ")
    done_re = re.compile(r"^(\S+) phase=traversal worker=(\d+) outer_pixel_done ")

    active_by_worker: dict[int, int] = {}
    run_start = None
    samples: list[tuple[float, int]] = []
    window_start = window.since
    window_end = window.until

    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = run_re.search(line)
        if m:
            timestamp = parse_optional_datetime(m.group(1))
            if timestamp <= window_start:
                run_start = timestamp
                active_by_worker.clear()
                samples.clear()
            continue

        if run_start is None:
            continue

        m = start_re.search(line)
        delta = 1
        if m is None:
            m = done_re.search(line)
            delta = -1
        if m is None:
            continue

        timestamp = parse_optional_datetime(m.group(1))
        if timestamp < run_start:
            continue
        if window_end is not None and timestamp > window_end:
            break

        worker = int(m.group(2))
        if delta > 0:
            active_by_worker[worker] = active_by_worker.get(worker, 0) + 1
        else:
            active_by_worker[worker] = max(0, active_by_worker.get(worker, 0) - 1)

        if timestamp >= window_start:
            elapsed = (timestamp - window_start).total_seconds()
            active = sum(1 for count in active_by_worker.values() if count > 0)
            samples.append((elapsed, active))

    if not samples:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    seconds = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    workers = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    if seconds[0] > 0.0:
        seconds = np.insert(seconds, 0, 0.0)
        workers = np.insert(workers, 0, workers[0])
    return seconds, workers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Series to plot as LABEL=PATH. Repeat for multiple simulations.",
    )
    parser.add_argument(
        "--measured-run",
        help="Optional measured run overlay as LABEL,SINCE or LABEL,SINCE,UNTIL.",
    )
    parser.add_argument(
        "--build-log",
        type=Path,
        help="Build log used with --measured-run.",
    )
    parser.add_argument("--ram-output", type=Path)
    parser.add_argument("--throughput-output", type=Path)
    parser.add_argument(
        "--title-prefix",
        default="Stochastic Scheduler Simulation",
    )
    parser.add_argument(
        "--limit-to-measured-run",
        action="store_true",
        help=(
            "Clip simulation series to the elapsed duration of --measured-run. "
            "Requires --measured-run with an UNTIL timestamp."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
