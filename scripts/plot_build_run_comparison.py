"""Compare measured Traversal run windows from one AO-sky build log."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plot_build_memory_timeline import (
    TimeWindow,
    parse_memory_samples,
    parse_optional_datetime,
    parse_progress_samples,
)


@dataclass(frozen=True, slots=True)
class RunWindowSpec:
    """One labeled build-log window to overlay."""

    label: str
    window: TimeWindow


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log).expanduser()
    runs = tuple(_parse_run_spec(value) for value in args.run)
    if args.ram_output is not None:
        _write_ram_comparison(
            log_path,
            runs,
            Path(args.ram_output).expanduser(),
        )
    if args.throughput_output is not None:
        _write_throughput_comparison(
            log_path,
            runs,
            Path(args.throughput_output).expanduser(),
        )


def _parse_run_spec(value: str) -> RunWindowSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError(
            "--run must be LABEL,SINCE or LABEL,SINCE,UNTIL, "
            f"got {value!r}"
        )
    until = parse_optional_datetime(parts[2]) if len(parts) == 3 and parts[2] else None
    return RunWindowSpec(
        label=parts[0],
        window=TimeWindow(
            since=parse_optional_datetime(parts[1]),
            until=until,
        ),
    )


def _write_ram_comparison(
    log_path: Path,
    runs: tuple[RunWindowSpec, ...],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    for run in runs:
        samples = parse_memory_samples(log_path, run.window)
        if not samples:
            raise ValueError(f"No memory samples found for run {run.label!r}")
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
        median_gib = float(np.median(total_gib))
        (line,) = ax.plot(
            elapsed_hours,
            total_gib,
            linewidth=1.5,
            label=f"{run.label}: {median_gib:.1f} GiB",
        )
        ax.axhline(
            median_gib,
            color=line.get_color(),
            linestyle="--",
            linewidth=1.1,
            label="_nolegend_",
        )
    ax.set_title("Measured Full-Sky Run RAM Comparison")
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Total RAM [GiB]")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_throughput_comparison(
    log_path: Path,
    runs: tuple[RunWindowSpec, ...],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=140)
    for run in runs:
        samples = parse_progress_samples(log_path, run.window)
        if not samples:
            raise ValueError(f"No progress samples found for run {run.label!r}")
        start = samples[0].timestamp
        elapsed_hours = np.asarray(
            [
                (sample.timestamp - start).total_seconds() / 3600.0
                for sample in samples
            ],
            dtype=np.float64,
        )
        completed = np.asarray([sample.completed for sample in samples], dtype=np.float64)
        throughput = 0.0
        if elapsed_hours[-1] > 0.0:
            throughput = float(completed[-1]) / float(elapsed_hours[-1] * 3600.0)
        (line,) = ax.plot(
            elapsed_hours,
            completed,
            linewidth=1.5,
            label=f"{run.label}: {throughput:.2f} pix/s",
        )
        if elapsed_hours[-1] > 0.0:
            ax.plot(
                [0.0, elapsed_hours[-1]],
                [0.0, completed[-1]],
                color=line.get_color(),
                linestyle="--",
                linewidth=1.1,
                label="_nolegend_",
            )
    ax.set_title("Measured Full-Sky Run Throughput Comparison")
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Completed Outer Pixels")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run window as LABEL,SINCE or LABEL,SINCE,UNTIL.",
    )
    parser.add_argument("--ram-output", type=Path)
    parser.add_argument("--throughput-output", type=Path)
    args = parser.parse_args()
    if args.ram_output is None and args.throughput_output is None:
        parser.error("at least one of --ram-output or --throughput-output is required")
    return args


if __name__ == "__main__":
    main()
