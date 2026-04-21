"""Plot per-outer-pixel runtime and RAM evidence from an AO-sky build log."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import numpy as np


LINE_TIME_RE = re.compile(r"^(\S+) ")
OUTER_PIXEL_START_RE = re.compile(
    r"^(\S+) phase=traversal worker=(\d+) outer_pixel_start "
    r"outer_pix=(\d+) star_count=(\d+)"
)
OUTER_PIXEL_DONE_RE = re.compile(
    r"^(\S+) phase=traversal worker=(\d+) outer_pixel_done "
    r"outer_pix=(\d+) elapsed_s=([0-9.]+) peak_rss_mb=([0-9.]+)"
)
MEMORY_SNAPSHOT_RE = re.compile(
    r"^(\S+) phase=traversal (?:memory_snapshot|memory_pressure) (.*)"
)
KEY_VALUE_RE = re.compile(r"(\w+)=([^\s]+)")

LOW_LATITUDE_RAM_FLOOR_GIB = 1.29
LOW_LATITUDE_RAM_DECAY_TAU_MIN = 2.5
HIGH_LATITUDE_RAM_FLOOR_GIB = 0.93
HIGH_LATITUDE_RAM_DECAY_TAU_MIN = 0.5


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Inclusive/exclusive timestamp filter for one run-log window."""

    since: datetime | None
    until: datetime | None


@dataclass(frozen=True, slots=True)
class OuterPixelRecord:
    """One completed outer-pixel traversal record."""

    started: datetime
    finished: datetime
    worker: int
    outer_pix: int
    star_count: int
    elapsed_s: float
    peak_rss_mb: float


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One current worker-RSS snapshot from the parent memory guard."""

    timestamp: datetime
    total_gib: float
    gpu_reserve_gib: float
    limit_gib: float
    trim_fraction: float
    pause_fraction: float
    worker_rss_gib: dict[int, float]


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log).expanduser()
    window = TimeWindow(
        since=_parse_optional_datetime(args.since),
        until=_parse_optional_datetime(args.until),
    )
    if window.since is not None and window.until is not None and window.until <= window.since:
        raise SystemExit("--until must be later than --since")
    records = _parse_outer_pixel_records(log_path, window)
    if not records:
        raise SystemExit(f"No per-outer-pixel records found in {log_path}")
    snapshots = _parse_memory_snapshots(log_path, window)

    output_dir = Path(args.output_dir).expanduser()
    prefix = str(args.prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_output = output_dir / f"{prefix}-runtime-model.png"
    stochastic_output = output_dir / f"{prefix}-runtime-stochastic.png"
    ram_output = output_dir / f"{prefix}-peak-rss-vs-stars.png"
    plateau_output = output_dir / f"{prefix}-rss-plateau-vs-max-stars.png"
    decay_output = output_dir / f"{prefix}-worker-rss-decay-model.png"
    total_ram_output = output_dir / f"{prefix}-predicted-total-ram.png"

    _write_runtime_model_plot(records, runtime_output)
    _write_stochastic_runtime_plot(records, stochastic_output, seed=int(args.seed))
    _write_ram_cloud_plot(records, ram_output)
    _write_rss_plateau_plot(records, plateau_output)
    if snapshots:
        _write_worker_rss_decay_model_plot(records, snapshots, decay_output)
        _write_total_ram_model_plot(records, snapshots, total_ram_output)

    print(f"records={len(records)}")
    print(f"memory_snapshots={len(snapshots)}")
    print(f"runtime_model={runtime_output}")
    print(f"runtime_stochastic={stochastic_output}")
    print(f"peak_rss_vs_stars={ram_output}")
    print(f"rss_plateau={plateau_output}")
    if snapshots:
        print(f"worker_rss_decay_model={decay_output}")
        print(f"predicted_total_ram={total_ram_output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="AO-sky build.log path.")
    parser.add_argument("--since", help="Inclusive ISO timestamp for the run window.")
    parser.add_argument("--until", help="Exclusive ISO timestamp for the run window.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="outer-pixel",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for the stochastic runtime realization.",
    )
    return parser.parse_args()


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_line_timestamp(line: str) -> datetime | None:
    match = LINE_TIME_RE.match(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


def _in_window(timestamp: datetime, window: TimeWindow) -> bool:
    return not (
        (window.since is not None and timestamp < window.since)
        or (window.until is not None and timestamp >= window.until)
    )


def _parse_outer_pixel_records(
    log_path: Path,
    window: TimeWindow,
) -> tuple[OuterPixelRecord, ...]:
    starts: dict[tuple[int, int], tuple[datetime, int]] = {}
    records: list[OuterPixelRecord] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = _parse_line_timestamp(line)
        if timestamp is None or not _in_window(timestamp, window):
            continue
        start_match = OUTER_PIXEL_START_RE.match(line)
        if start_match is not None:
            worker = int(start_match.group(2))
            outer_pix = int(start_match.group(3))
            star_count = int(start_match.group(4))
            starts[(worker, outer_pix)] = (timestamp, star_count)
            continue
        done_match = OUTER_PIXEL_DONE_RE.match(line)
        if done_match is None:
            continue
        worker = int(done_match.group(2))
        outer_pix = int(done_match.group(3))
        start = starts.get((worker, outer_pix))
        if start is None:
            continue
        records.append(
            OuterPixelRecord(
                started=start[0],
                finished=timestamp,
                worker=worker,
                outer_pix=outer_pix,
                star_count=start[1],
                elapsed_s=float(done_match.group(4)),
                peak_rss_mb=float(done_match.group(5)),
            )
        )
    return tuple(sorted(records, key=lambda record: record.finished))


def _parse_memory_snapshots(
    log_path: Path,
    window: TimeWindow,
) -> tuple[MemorySnapshot, ...]:
    snapshots: list[MemorySnapshot] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = _parse_line_timestamp(line)
        if timestamp is None or not _in_window(timestamp, window):
            continue
        match = MEMORY_SNAPSHOT_RE.match(line)
        if match is None:
            continue
        values = dict(KEY_VALUE_RE.findall(match.group(2)))
        worker_rss_text = values.get("worker_rss_mb")
        if worker_rss_text is None:
            continue
        worker_rss_gib: dict[int, float] = {}
        for part in worker_rss_text.split(","):
            if ":" not in part:
                continue
            worker_id, value = part.split(":", 1)
            try:
                worker_rss_gib[int(worker_id)] = float(value) / 1024.0
            except ValueError:
                continue
        if not worker_rss_gib:
            continue
        try:
            total_gib = float(values.get("total_mb", "nan")) / 1024.0
            gpu_reserve_gib = float(values.get("gpu_reserve_mb", "0")) / 1024.0
            limit_gib = float(values.get("limit_mb", "0")) / 1024.0
            trim_fraction = float(values.get("trim_fraction", "0.85"))
            pause_fraction = float(values.get("pause_fraction", "0.95"))
        except ValueError:
            continue
        snapshots.append(
            MemorySnapshot(
                timestamp=timestamp,
                total_gib=total_gib,
                gpu_reserve_gib=gpu_reserve_gib,
                limit_gib=limit_gib,
                trim_fraction=trim_fraction,
                pause_fraction=pause_fraction,
                worker_rss_gib=worker_rss_gib,
            )
        )
    return tuple(sorted(snapshots, key=lambda snapshot: snapshot.timestamp))


def _runtime_model(star_count: np.ndarray) -> np.ndarray:
    """Return the retained 8-worker runtime model in seconds."""

    u = np.asarray(star_count, dtype=np.float64) / 1000.0
    result = np.empty_like(u)
    sparse = u <= 20.0
    transition = (u > 20.0) & (u <= 33.0)
    dense = u > 33.0
    result[sparse] = 0.571526 - 0.182845 * u[sparse] + 0.0725123 * u[sparse] ** 2
    v = u[transition] - 20.0
    result[transition] = 25.9196 - 1.45735 * v + 0.0445096 * v**2
    v = u[dense] - 33.0
    result[dense] = 14.4961 + 0.00559679 * v + 0.00000504397 * v**2
    return np.maximum(0.1, result)


def _write_runtime_model_plot(records: tuple[OuterPixelRecord, ...], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stars = _star_counts(records)
    elapsed = _elapsed_s(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150, sharey=True)
    for ax, log_x, title in (
        (axes[0], True, "Full Range"),
        (axes[1], False, "Peak Zoom"),
    ):
        ax.scatter(stars, elapsed, s=9 if log_x else 11, alpha=0.28, linewidths=0, color="#3455a4")
        _add_runtime_model_lines(ax, stars)
        if log_x:
            ax.set_xscale("log")
            ax.grid(True, which="minor", axis="x", alpha=0.12)
        else:
            ax.set_xlim(0.0, 50_000.0)
        ax.set_ylim(0.0, 33.0)
        ax.set_xlabel("Total Gaia Stars In Outer Pixel")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.25)
    axes[0].set_ylabel("Elapsed Time Per Outer Pixel (s)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Outer-Pixel Runtime With 20k-33k Transition", y=1.02)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_stochastic_runtime_plot(
    records: tuple[OuterPixelRecord, ...],
    output: Path,
    *,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stars = _star_counts(records)
    elapsed = _elapsed_s(records)
    model = _runtime_model(stars)
    synthetic = _stochastic_runtime(stars, elapsed, model, seed=seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150, sharex=True, sharey=True)
    for ax, values, title in (
        (axes[0], elapsed, "Measured Runtime"),
        (axes[1], synthetic, "Synthetic Runtime, Cap Only Sparse/Transition"),
    ):
        ax.scatter(stars, values, s=9, alpha=0.28, linewidths=0, color="#3455a4")
        order = np.argsort(stars)
        ax.plot(stars[order], model[order], color="#c43b3b", linewidth=2.0)
        ax.set_xscale("log")
        ax.set_xlabel("Total Gaia Stars In Outer Pixel")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", axis="x", alpha=0.12)
    axes[0].set_ylabel("Elapsed Time Per Outer Pixel (s)")
    axes[0].set_ylim(0.0, 33.0)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_ram_cloud_plot(records: tuple[OuterPixelRecord, ...], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stars = _star_counts(records)
    rss_gib = _peak_rss_gib(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    ax.scatter(stars, rss_gib, s=9, alpha=0.30, linewidths=0, color="#3455a4")
    ax.set_xscale("log")
    ax.set_xlabel("Total Gaia Stars In Outer Pixel")
    ax.set_ylabel("Worker Peak RSS High-Water (GiB)")
    ax.set_title("Worker Peak RSS Versus Star Count")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_rss_plateau_plot(records: tuple[OuterPixelRecord, ...], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _rss_plateau_points(records)
    stars = np.asarray([point[0] for point in points], dtype=np.float64)
    rss_gib = np.asarray([point[1] for point in points], dtype=np.float64)
    linear = np.polyfit(stars, rss_gib, 1)
    xx = np.geomspace(stars.min(), stars.max(), 500)
    y_linear = np.polyval(linear, xx)
    rmse_linear = float(np.sqrt(np.mean((rss_gib - np.polyval(linear, stars)) ** 2)))
    ss_total = float(np.sum((rss_gib - rss_gib.mean()) ** 2))
    r2_linear = 1.0 - float(np.sum((rss_gib - np.polyval(linear, stars)) ** 2)) / ss_total

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    ax.scatter(stars, rss_gib, s=24, alpha=0.72, linewidths=0, color="#3455a4")
    ax.plot(
        xx,
        y_linear,
        color="#999999",
        linewidth=1.8,
        label=f"Linear RMSE {rmse_linear:.2f} GiB",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Maximum Gaia Stars Seen So Far By Worker")
    ax.set_ylabel("Worker Peak RSS High-Water (GiB)")
    ax.set_title("Worker RSS Plateaus With Linear Fit")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.03,
        0.95,
        f"plateaus = {len(points)}\nR2 = {r2_linear:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_worker_rss_decay_model_plot(
    records: tuple[OuterPixelRecord, ...],
    snapshots: tuple[MemorySnapshot, ...],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workers = sorted({record.worker for record in records})
    start_time = min(record.started for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(workers),
        1,
        figsize=(11.0, 1.75 * len(workers)),
        dpi=150,
        sharex=True,
    )
    if len(workers) == 1:
        axes = [axes]
    for ax, worker in zip(axes, workers):
        worker_records = [record for record in records if record.worker == worker]
        event_hours = np.asarray(
            [
                (record.started - start_time).total_seconds() / 3600.0
                for record in worker_records
            ],
            dtype=np.float64,
        )
        star_counts = np.asarray(
            [record.star_count for record in worker_records],
            dtype=np.float64,
        )
        ax.scatter(
            event_hours,
            star_counts / 1000.0,
            s=8,
            alpha=0.22,
            linewidths=0,
            color="#3455a4",
        )
        ax.set_yscale("log")
        ax.set_ylabel(f"W{worker}\nStars (k)", color="#3455a4")
        ax.tick_params(axis="y", labelcolor="#3455a4", labelsize=7)
        ax.grid(True, which="major", alpha=0.18)

        worker_class, floor_gib, tau_min = _worker_ram_class(worker)
        times, measured, modeled = _simulate_worker_rss_decay(
            worker_records,
            snapshots,
            worker,
            start_time,
            floor_gib=floor_gib,
            tau_min=tau_min,
        )
        ax2 = ax.twinx()
        ax2.plot(times, measured, color="#dc2626", linewidth=1.3, label="Measured RSS")
        ax2.plot(
            times,
            modeled,
            color="#111827",
            linewidth=1.5,
            linestyle="--",
            label=f"{worker_class} model",
        )
        ax2.set_ylabel("RSS (GiB)", color="#dc2626")
        ax2.tick_params(axis="y", labelcolor="#dc2626", labelsize=7)
        if len(measured):
            combined = np.concatenate([measured, modeled])
            lower = max(0.0, float(np.percentile(combined, 1.0)) - 0.2)
            upper = float(np.percentile(combined, 99.0)) + 0.2
            if upper - lower < 0.6:
                midpoint = (upper + lower) / 2.0
                lower = midpoint - 0.3
                upper = midpoint + 0.3
            ax2.set_ylim(lower, upper)
        if len(measured):
            ax2.text(
                0.995,
                0.82,
                f"{worker_class}\nfloor {floor_gib:.2f}\ntau {tau_min:.2f}",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#111827",
            )
        if worker == workers[0]:
            ax2.legend(frameon=False, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Elapsed Time Since Window Start (hours)")
    fig.suptitle(
        "Current Worker RSS And Intended-Split Two-Class RAM Model",
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(output)
    plt.close(fig)


def _write_total_ram_model_plot(
    records: tuple[OuterPixelRecord, ...],
    snapshots: tuple[MemorySnapshot, ...],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_time = snapshots[0].timestamp
    modeled_worker_rss = _simulate_all_worker_rss_decay(records, snapshots, start_time)
    times = np.asarray(
        [
            (snapshot.timestamp - start_time).total_seconds() / 3600.0
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    measured_total = np.asarray(
        [snapshot.total_gib for snapshot in snapshots],
        dtype=np.float64,
    )
    gpu_reserve = np.asarray(
        [snapshot.gpu_reserve_gib for snapshot in snapshots],
        dtype=np.float64,
    )
    predicted_total = modeled_worker_rss + gpu_reserve
    limit = np.asarray([snapshot.limit_gib for snapshot in snapshots], dtype=np.float64)
    trim = np.asarray(
        [snapshot.limit_gib * snapshot.trim_fraction for snapshot in snapshots],
        dtype=np.float64,
    )
    pause = np.asarray(
        [snapshot.limit_gib * snapshot.pause_fraction for snapshot in snapshots],
        dtype=np.float64,
    )
    finite = np.isfinite(measured_total) & np.isfinite(predicted_total)
    residual = predicted_total[finite] - measured_total[finite]
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else float("nan")
    bias = float(np.mean(residual)) if len(residual) else float("nan")
    corr = (
        float(np.corrcoef(measured_total[finite], predicted_total[finite])[0, 1])
        if int(finite.sum()) >= 2
        else float("nan")
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    ax.plot(
        times,
        measured_total,
        color="black",
        linewidth=1.6,
        label="Measured total RAM",
    )
    ax.plot(
        times,
        predicted_total,
        color="#2563eb",
        linewidth=1.7,
        linestyle="--",
        label="Predicted total RAM",
    )
    if np.any(limit > 0.0):
        ax.axhline(
            float(np.max(trim)),
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            float(np.max(pause)),
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            float(np.max(limit)),
            color="0.25",
            linestyle="--",
            linewidth=1.0,
        )
    ax.text(
        0.02,
        0.95,
        "Predicted = worker model + GPU reserve\n"
        f"RMSE: {rmse:.2f} GiB   Bias: {bias:+.2f} GiB   Corr: {corr:.2f}\n"
        f"Peak measured/predicted: "
        f"{np.nanmax(measured_total):.1f}/{np.nanmax(predicted_total):.1f} GiB",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "0.7",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.25",
        },
    )
    ax.set_xlabel("Elapsed Time [h]")
    ax.set_ylabel("Total RAM [GiB]")
    ax.set_title("Measured Versus Predicted Total RAM")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _add_runtime_model_lines(ax: object, stars: np.ndarray) -> None:
    segments = (
        (float(stars.min()), 20_000.0, "#c43b3b", "Continuous quadratic fit"),
        (20_000.0, 33_000.0, "#d97706", None),
        (33_000.0, float(stars.max()), "#15803d", None),
    )
    for lower, upper, color, label in segments:
        xx = np.linspace(lower, upper, 500)
        ax.plot(xx, _runtime_model(xx), color=color, linewidth=2.5, label=label)
    ax.axvline(20_000.0, color="#d97706", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(33_000.0, color="#d97706", linestyle="--", linewidth=1.2, alpha=0.8)


def _stochastic_runtime(
    stars: np.ndarray,
    elapsed: np.ndarray,
    model: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    residual = elapsed - model
    u = stars / 1000.0
    sparse = u <= 20.0
    transition = (u > 20.0) & (u <= 33.0)
    dense = u > 33.0
    sparse_sigma = _fit_log_sigma(
        u,
        residual,
        [0.0, 3.0, 6.0, 10.0, 15.0, 20.0],
        lambda values: values / 20.0,
    )
    dense_sigma = _fit_log_sigma(
        u,
        residual,
        [33.0, 50.0, 100.0, 250.0, 500.0, 1200.0],
        lambda values: np.log(values / 33.0),
    )
    mu = np.empty_like(stars, dtype=np.float64)
    mu[sparse] = float(np.median(residual[sparse]))
    mu[transition] = float(np.median(residual[transition]))
    mu[dense] = float(np.median(residual[dense]))
    sigma = np.empty_like(stars, dtype=np.float64)
    sigma[sparse] = np.exp(np.polyval(sparse_sigma, u[sparse] / 20.0))
    sigma[transition] = _robust_sigma(residual[transition])
    sigma[dense] = np.exp(np.polyval(dense_sigma, np.log(u[dense] / 33.0)))
    sigma = np.clip(sigma, 0.03, 6.0)

    rng = np.random.default_rng(seed)
    degrees = 7.0
    noise = rng.standard_t(degrees, size=len(stars)) * sigma * np.sqrt((degrees - 2.0) / degrees)
    realized = np.maximum(0.1, model + noise + mu)
    for mask in (sparse, transition, dense):
        lower, upper = np.percentile(residual[mask], [0.5, 99.5])
        realized[mask] = np.maximum(0.1, model[mask] + np.clip(noise[mask] + mu[mask], lower, upper))
    realized[sparse | transition] = np.minimum(realized[sparse | transition], 28.0)
    return realized


def _fit_log_sigma(
    u: np.ndarray,
    residual: np.ndarray,
    edges: list[float],
    transform: object,
) -> np.ndarray:
    centers: list[float] = []
    sigmas: list[float] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (u >= lower) & (u < upper)
        if int(mask.sum()) < 30:
            continue
        centers.append(float(np.median(u[mask])))
        sigmas.append(_robust_sigma(residual[mask]))
    if len(centers) < 2:
        raise ValueError("not enough residual bins to fit stochastic scatter")
    x = transform(np.asarray(centers, dtype=np.float64))
    degree = min(2, len(centers) - 1)
    return np.polyfit(x, np.log(np.asarray(sigmas, dtype=np.float64)), degree)


def _robust_sigma(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return max(float((q75 - q25) / 1.349), 0.03)


def _rss_plateau_points(records: tuple[OuterPixelRecord, ...]) -> tuple[tuple[int, float], ...]:
    points: list[tuple[int, float]] = []
    for worker in sorted({record.worker for record in records}):
        worker_records = [record for record in records if record.worker == worker]
        max_so_far = 0
        last_rss: float | None = None
        for record in worker_records:
            max_so_far = max(max_so_far, record.star_count)
            rss = round(record.peak_rss_mb, 1)
            if last_rss is None or rss != last_rss:
                points.append((max_so_far, rss / 1024.0))
                last_rss = rss
    return tuple(points)


def _simulate_worker_rss_decay(
    records: list[OuterPixelRecord],
    snapshots: tuple[MemorySnapshot, ...],
    worker: int,
    start_time: datetime,
    *,
    floor_gib: float,
    tau_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = floor_gib
    last_time = start_time
    event_index = 0
    ordered_records = sorted(records, key=lambda record: record.started)
    times: list[float] = []
    measured: list[float] = []
    modeled: list[float] = []
    tau_seconds = tau_min * 60.0
    for snapshot in snapshots:
        if worker not in snapshot.worker_rss_gib:
            continue
        while (
            event_index < len(ordered_records)
            and ordered_records[event_index].started <= snapshot.timestamp
        ):
            event = ordered_records[event_index]
            delta_seconds = (event.started - last_time).total_seconds()
            state = _decay_worker_rss_state(
                state,
                delta_seconds,
                tau_seconds,
                floor_gib=floor_gib,
            )
            state = max(state, _worker_rss_demand_gib(event.star_count))
            last_time = event.started
            event_index += 1
        delta_seconds = (snapshot.timestamp - last_time).total_seconds()
        sample_state = _decay_worker_rss_state(
            state,
            delta_seconds,
            tau_seconds,
            floor_gib=floor_gib,
        )
        times.append((snapshot.timestamp - start_time).total_seconds() / 3600.0)
        measured.append(snapshot.worker_rss_gib[worker])
        modeled.append(sample_state)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(measured, dtype=np.float64),
        np.asarray(modeled, dtype=np.float64),
    )


def _simulate_all_worker_rss_decay(
    records: tuple[OuterPixelRecord, ...],
    snapshots: tuple[MemorySnapshot, ...],
    start_time: datetime,
) -> np.ndarray:
    """Return summed modeled worker RSS at each parent memory snapshot."""

    workers = sorted({record.worker for record in records})
    records_by_worker = {
        worker: sorted(
            (record for record in records if record.worker == worker),
            key=lambda record: record.started,
        )
        for worker in workers
    }
    state: dict[int, float] = {}
    last_time: dict[int, datetime] = {}
    event_index: dict[int, int] = {}
    for worker in workers:
        _, floor_gib, _ = _worker_ram_class(worker)
        state[worker] = floor_gib
        last_time[worker] = start_time
        event_index[worker] = 0

    modeled = np.zeros(len(snapshots), dtype=np.float64)
    for snapshot_index, snapshot in enumerate(snapshots):
        total = 0.0
        for worker in workers:
            worker_records = records_by_worker[worker]
            _, floor_gib, tau_min = _worker_ram_class(worker)
            tau_seconds = tau_min * 60.0
            while (
                event_index[worker] < len(worker_records)
                and worker_records[event_index[worker]].started <= snapshot.timestamp
            ):
                event = worker_records[event_index[worker]]
                state[worker] = _decay_worker_rss_state(
                    state[worker],
                    (event.started - last_time[worker]).total_seconds(),
                    tau_seconds,
                    floor_gib=floor_gib,
                )
                state[worker] = max(
                    state[worker],
                    _worker_rss_demand_gib(event.star_count),
                )
                last_time[worker] = event.started
                event_index[worker] += 1
            total += _decay_worker_rss_state(
                state[worker],
                (snapshot.timestamp - last_time[worker]).total_seconds(),
                tau_seconds,
                floor_gib=floor_gib,
            )
        modeled[snapshot_index] = total
    return modeled


def _decay_worker_rss_state(
    state_gib: float,
    delta_seconds: float,
    tau_seconds: float,
    *,
    floor_gib: float,
) -> float:
    if tau_seconds <= 0.0:
        return floor_gib
    return floor_gib + (state_gib - floor_gib) * np.exp(-delta_seconds / tau_seconds)


def _worker_ram_class(worker: int) -> tuple[str, float, float]:
    """Return the intended 8/5 scheduler RAM class for a worker."""

    if worker <= 4:
        return (
            "low-lat",
            LOW_LATITUDE_RAM_FLOOR_GIB,
            LOW_LATITUDE_RAM_DECAY_TAU_MIN,
        )
    return (
        "high-lat",
        HIGH_LATITUDE_RAM_FLOOR_GIB,
        HIGH_LATITUDE_RAM_DECAY_TAU_MIN,
    )


def _worker_rss_demand_gib(star_count: int | float) -> float:
    return 0.914584 + 0.000004256378 * max(float(star_count), 0.0)


def _star_counts(records: tuple[OuterPixelRecord, ...]) -> np.ndarray:
    return np.asarray([record.star_count for record in records], dtype=np.float64)


def _elapsed_s(records: tuple[OuterPixelRecord, ...]) -> np.ndarray:
    return np.asarray([record.elapsed_s for record in records], dtype=np.float64)


def _peak_rss_gib(records: tuple[OuterPixelRecord, ...]) -> np.ndarray:
    return np.asarray([record.peak_rss_mb / 1024.0 for record in records], dtype=np.float64)


if __name__ == "__main__":
    main()
