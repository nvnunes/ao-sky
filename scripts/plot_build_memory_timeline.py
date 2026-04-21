"""Plot measured Traversal memory and throughput from an AO-sky build log."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np


LINE_TIME_RE = re.compile(r"^(\S+) ")
KEY_VALUE_RE = re.compile(r"(\w+)=([^\s]+)")
MEMORY_EVENTS = ("memory_pressure", "memory_snapshot")
TRIM_ENTER_FRACTION = 0.80
PAUSE_ENTER_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class MemorySample:
    """One parent memory sample parsed from build.log."""

    timestamp: datetime
    event: str
    state: str
    total_mb: float
    gpu_reserve_mb: float
    limit_mb: float
    trim_fraction: float
    pause_fraction: float
    worker_rss_mb: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ProgressSample:
    """One cumulative completion sample parsed from build.log."""

    timestamp: datetime
    completed: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Run-level context parsed from build.log for plot titles."""

    workers: int | None
    outer_pixels: int | None
    low_latitude_workers: int | None


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Inclusive/exclusive timestamp filter for one run-log window."""

    since: datetime | None
    until: datetime | None


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log).expanduser()
    window = resolve_time_window(log_path, args)
    summary = parse_run_summary(log_path, window)
    if args.output is not None:
        samples = parse_memory_samples(log_path, window)
        if not samples:
            raise SystemExit(f"No memory pressure samples found in {args.log}")
        write_plot(
            samples,
            Path(args.output).expanduser(),
            title=args.title or default_plot_title(summary, throughput=False),
            max_samples=int(args.max_samples),
        )
        print_summary(samples)
    if args.throughput_output is not None:
        progress = parse_progress_samples(log_path, window)
        if not progress:
            raise SystemExit(f"No progress samples found in {args.log}")
        write_throughput_plot(
            progress,
            Path(args.throughput_output).expanduser(),
            title=args.throughput_title
            or args.title
            or default_plot_title(summary, throughput=True),
        )
        print_progress_summary(progress)


def resolve_time_window(log_path: Path, args: argparse.Namespace) -> TimeWindow:
    """Return the requested log window."""

    if args.latest_run and args.since is not None:
        raise SystemExit("--latest-run and --since cannot be used together")
    since = parse_optional_datetime(args.since)
    until = parse_optional_datetime(args.until)
    if args.latest_run:
        since = latest_run_start(log_path)
        if since is None:
            raise SystemExit(f"No run start found in {log_path}")
    if since is not None and until is not None and until <= since:
        raise SystemExit("--until must be later than --since")
    return TimeWindow(since=since, until=until)


def latest_run_start(log_path: Path) -> datetime | None:
    """Return the timestamp of the final `run start` line in a build log."""

    start: datetime | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " run start " not in line:
            continue
        timestamp = parse_line_timestamp(line)
        if timestamp is not None:
            start = timestamp
    return start


def parse_optional_datetime(value: str | None) -> datetime | None:
    """Parse an optional ISO timestamp accepted by build logs."""

    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_line_timestamp(line: str) -> datetime | None:
    """Return the leading build-log timestamp, if present and valid."""

    timestamp_match = LINE_TIME_RE.match(line)
    if timestamp_match is None:
        return None
    try:
        return datetime.fromisoformat(timestamp_match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


def in_time_window(timestamp: datetime, window: TimeWindow) -> bool:
    """Return whether a timestamp is inside a requested log window."""

    return not (
        (window.since is not None and timestamp < window.since)
        or (window.until is not None and timestamp >= window.until)
    )


def parse_run_summary(log_path: Path, window: TimeWindow) -> RunSummary:
    """Return title context for the latest run recorded in a build log.

    Current logs record worker count on the `run start` line and each worker's
    assigned outer-pixel count on subsequent `started` lines. Newer logs may
    also record `low_latitude_workers`; for older logs this is inferred from the
    worker-plan split when the low-latitude lane has a visibly smaller outer
    pixel count.
    """

    workers: int | None = None
    low_latitude_workers: int | None = None
    outer_pixels_by_worker: dict[int, int] = {}

    for line in log_path.read_text(encoding="utf-8").splitlines():
        timestamp = parse_line_timestamp(line)
        if timestamp is None or not in_time_window(timestamp, window):
            continue
        values = dict(KEY_VALUE_RE.findall(line))
        if " run start " in line:
            workers = parse_optional_int(values.get("workers"))
            low_latitude_workers = parse_optional_int(values.get("low_latitude_workers"))
            outer_pixels_by_worker = {}
            continue
        if " started" not in line or "worker" not in values:
            continue
        worker_id = parse_optional_int(values.get("worker"))
        outer_pixels = parse_optional_int(values.get("outer_pixels"))
        if worker_id is None or outer_pixels is None:
            continue
        outer_pixels_by_worker[worker_id] = outer_pixels

    if workers is None and outer_pixels_by_worker:
        workers = len(outer_pixels_by_worker)
    outer_pixels = (
        sum(outer_pixels_by_worker.values()) if outer_pixels_by_worker else None
    )
    if low_latitude_workers is None:
        low_latitude_workers = infer_low_latitude_worker_count(
            tuple(outer_pixels_by_worker.values())
        )
    return RunSummary(
        workers=workers,
        outer_pixels=outer_pixels,
        low_latitude_workers=low_latitude_workers,
    )


def parse_optional_int(value: str | None) -> int | None:
    """Return an integer from a log value, or None when it is absent/invalid."""

    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def infer_low_latitude_worker_count(outer_pixel_counts: tuple[int, ...]) -> int | None:
    """Infer the low-latitude worker lane from a two-band worker split."""

    if len(outer_pixel_counts) < 2:
        return None
    counts = sorted(count for count in outer_pixel_counts if count > 0)
    if len(counts) < 2:
        return None
    gaps = [
        (counts[index + 1] / counts[index], index + 1)
        for index in range(len(counts) - 1)
        if counts[index] > 0
    ]
    if not gaps:
        return None
    ratio, split_index = max(gaps, key=lambda item: item[0])
    if ratio < 2.0:
        return None
    return split_index


def default_plot_title(summary: RunSummary, *, throughput: bool) -> str:
    """Build a measured plot title from log-derived run context."""

    if summary.outer_pixels is not None and summary.workers is not None:
        title = f"Measured {summary.outer_pixels}-Pixels with {summary.workers} Workers"
    elif summary.workers is not None:
        title = f"Measured Traversal with {summary.workers} Workers"
    else:
        title = "Measured Traversal"
    if summary.low_latitude_workers is not None:
        title = f"{title} [{summary.low_latitude_workers} Low-Lat Workers]"
    if throughput:
        title = f"{title} Throughput"
    return title


def parse_memory_samples(log_path: Path, window: TimeWindow) -> list[MemorySample]:
    """Return memory-pressure and memory-snapshot samples from one build log."""

    samples: list[MemorySample] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        event = next((name for name in MEMORY_EVENTS if f" {name} " in line), None)
        if event is None:
            continue
        timestamp = parse_line_timestamp(line)
        if timestamp is None or not in_time_window(timestamp, window):
            continue
        values = dict(KEY_VALUE_RE.findall(line))
        if "total_mb" not in values:
            continue
        try:
            samples.append(
                MemorySample(
                    timestamp=timestamp,
                    event=event,
                    state=values.get("state", "unknown"),
                    total_mb=float(values["total_mb"]),
                    gpu_reserve_mb=float(values.get("gpu_reserve_mb", "0")),
                    limit_mb=float(values.get("limit_mb", "0")),
                    trim_fraction=float(
                        values.get("trim_fraction", str(TRIM_ENTER_FRACTION))
                    ),
                    pause_fraction=float(
                        values.get("pause_fraction", str(PAUSE_ENTER_FRACTION))
                    ),
                    worker_rss_mb=parse_worker_rss(values.get("worker_rss_mb", "")),
                )
            )
        except ValueError:
            continue
    return samples


def parse_progress_samples(log_path: Path, window: TimeWindow) -> list[ProgressSample]:
    """Return cumulative completion samples from one build log."""

    completed_by_worker: dict[int, int] = {}
    samples: list[ProgressSample] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if (
            " progress " not in line
            and " profile " not in line
            and " outer_pixel_done " not in line
        ):
            continue
        timestamp = parse_line_timestamp(line)
        if timestamp is None or not in_time_window(timestamp, window):
            continue
        values = dict(KEY_VALUE_RE.findall(line))
        try:
            worker_id = int(values["worker"])
        except (KeyError, ValueError):
            continue
        if "outer_pixel_done" in line:
            completed_by_worker[worker_id] = completed_by_worker.get(worker_id, 0) + 1
        elif "completed" in values:
            try:
                completed_by_worker[worker_id] = int(values["completed"])
            except ValueError:
                continue
        else:
            continue
        total_completed = sum(completed_by_worker.values())
        if samples and total_completed < samples[-1].completed:
            continue
        if samples and total_completed == samples[-1].completed:
            samples[-1] = ProgressSample(timestamp=timestamp, completed=total_completed)
        else:
            samples.append(ProgressSample(timestamp=timestamp, completed=total_completed))
    if samples and samples[0].completed != 0:
        samples.insert(0, ProgressSample(timestamp=samples[0].timestamp, completed=0))
    return samples


def parse_worker_rss(text: str) -> tuple[float, ...]:
    """Parse worker_rss_mb=0:123,1:456 into a tuple ordered by worker id."""

    values: dict[int, float] = {}
    for item in text.split(","):
        if ":" not in item:
            continue
        worker_id, rss = item.split(":", 1)
        try:
            values[int(worker_id)] = float(rss)
        except ValueError:
            continue
    return tuple(values[worker_id] for worker_id in sorted(values))


def write_plot(
    samples: list[MemorySample],
    output: Path,
    *,
    title: str | None,
    max_samples: int,
) -> None:
    """Write a stacked measured RAM timeline plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_samples = downsample_samples(samples, max_samples=max_samples)
    start = plot_samples[0].timestamp
    times_h = np.asarray(
        [
            (sample.timestamp - start).total_seconds() / 3600.0
            for sample in plot_samples
        ],
        dtype=np.float64,
    )
    worker_rows, overhead_gib, total_gib, limit_gib = sample_arrays(plot_samples)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    baseline = np.zeros(len(plot_samples), dtype=np.float64)
    ax.fill_between(
        times_h,
        baseline,
        overhead_gib,
        step="post",
        color="0.80",
        alpha=0.35,
        linewidth=0,
    )
    baseline = overhead_gib.copy()
    colors = plt.get_cmap("tab10").colors
    for index, worker_gib in enumerate(worker_rows):
        top = baseline + worker_gib
        ax.fill_between(
            times_h,
            baseline,
            top,
            step="post",
            color=colors[index % len(colors)],
            alpha=0.82,
            linewidth=0,
        )
        baseline = top
    if np.any(limit_gib > 0.0):
        limit_value = float(np.max(limit_gib))
        trim_fraction = float(plot_samples[-1].trim_fraction)
        pause_fraction = float(plot_samples[-1].pause_fraction)
        ax.axhline(
            limit_value * trim_fraction,
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            limit_value * pause_fraction,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            limit_value,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
        )
    ax.plot(times_h, total_gib, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Measured RAM [GiB]")
    ax.set_title(title or "Measured Traversal RAM Timeline")
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_throughput_plot(
    samples: list[ProgressSample],
    output: Path,
    *,
    title: str | None,
) -> None:
    """Write a cumulative completion plot from build-log progress samples."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = samples[0].timestamp
    elapsed_seconds = np.asarray(
        [(sample.timestamp - start).total_seconds() for sample in samples],
        dtype=np.float64,
    )
    elapsed_hours = elapsed_seconds / 3600.0
    completed = np.asarray([sample.completed for sample in samples], dtype=np.float64)
    slope = (
        float(completed[-1]) / float(elapsed_seconds[-1])
        if float(elapsed_seconds[-1]) > 0.0
        else 0.0
    )
    average_line = slope * elapsed_seconds

    output.parent.mkdir(parents=True, exist_ok=True)
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
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Completed Outer Pixels")
    ax.set_title(title or "Measured Traversal Throughput")
    ax.grid(True, color="0.88", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def downsample_samples(
    samples: list[MemorySample],
    *,
    max_samples: int,
) -> list[MemorySample]:
    """Keep plotting responsive for very large logs."""

    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    indexes = np.linspace(0, len(samples) - 1, max_samples, dtype=np.int64)
    return [samples[int(index)] for index in indexes]


def sample_arrays(
    samples: list[MemorySample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return worker bands, overhead, total, and limit arrays in GiB."""

    worker_count = max((len(sample.worker_rss_mb) for sample in samples), default=0)
    worker_rows = np.zeros((worker_count, len(samples)), dtype=np.float64)
    overhead = np.zeros(len(samples), dtype=np.float64)
    total = np.zeros(len(samples), dtype=np.float64)
    limit = np.zeros(len(samples), dtype=np.float64)

    for column, sample in enumerate(samples):
        active_workers = len(sample.worker_rss_mb)
        per_worker_gpu = (
            sample.gpu_reserve_mb / active_workers if active_workers else 0.0
        )
        worker_values = sorted(
            (rss + per_worker_gpu) / 1024.0
            for rss in sample.worker_rss_mb
        )
        for row, value in enumerate(worker_values):
            worker_rows[row, column] = value
        worker_plus_gpu_mb = sum(sample.worker_rss_mb) + sample.gpu_reserve_mb
        overhead[column] = max(sample.total_mb - worker_plus_gpu_mb, 0.0) / 1024.0
        total[column] = sample.total_mb / 1024.0
        limit[column] = sample.limit_mb / 1024.0
    return worker_rows, overhead, total, limit


def print_summary(samples: list[MemorySample]) -> None:
    """Print compact numeric context for the generated plot."""

    start = samples[0].timestamp
    end = samples[-1].timestamp
    elapsed_h = (end - start).total_seconds() / 3600.0
    states: dict[str, int] = {}
    for sample in samples:
        states[sample.state] = states.get(sample.state, 0) + 1
    peak = max(samples, key=lambda sample: sample.total_mb)
    print(f"Samples: {len(samples)}")
    print(f"Elapsed: {elapsed_h:.2f} h")
    print(f"Peak total RAM: {peak.total_mb / 1024.0:.2f} GiB")
    print(f"Peak timestamp: {peak.timestamp.isoformat()}")
    print(f"States: {states}")


def print_progress_summary(samples: list[ProgressSample]) -> None:
    """Print compact numeric context for the generated throughput plot."""

    elapsed_h = (samples[-1].timestamp - samples[0].timestamp).total_seconds() / 3600.0
    elapsed_s = elapsed_h * 3600.0
    throughput = samples[-1].completed / elapsed_s if elapsed_s > 0.0 else 0.0
    print(f"Progress samples: {len(samples)}")
    print(f"Completed: {samples[-1].completed}")
    print(f"Progress elapsed: {elapsed_h:.2f} h")
    print(f"Progress throughput: {throughput:.3f} pix/s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot measured Traversal memory pressure and throughput from build.log.",
    )
    parser.add_argument("log", help="Path to build.log")
    parser.add_argument(
        "--output",
        help="Output image path, usually under docs/assets/benchmarking/<version>.",
    )
    parser.add_argument(
        "--throughput-output",
        help="Optional cumulative throughput output image path.",
    )
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument("--throughput-title", help="Optional throughput plot title.")
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Plot only the final run window recorded in the build log.",
    )
    parser.add_argument(
        "--since",
        help="Plot only samples at or after this ISO timestamp.",
    )
    parser.add_argument(
        "--until",
        help="Plot only samples before this ISO timestamp.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Maximum memory samples to draw; 0 disables downsampling.",
    )
    args = parser.parse_args()
    if args.output is None and args.throughput_output is None:
        parser.error("at least one of --output or --throughput-output is required")
    return args


if __name__ == "__main__":
    main()
