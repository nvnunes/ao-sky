"""Benchmark the current Traversal baseline with detailed stage telemetry."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import h5py
import yaml
from astropy.table import Table

os.environ.setdefault("AO_SKY_PREDICT_DEVICE", "auto")
os.environ.setdefault("AO_SKY_AVERAGED_PREDICT_DEVICE", "auto")

from ao_sky.build._constants import WORK_STATUS_DONE, WORK_STATUS_PENDING
from ao_sky.build._models import TraversalExecutionConfig
from ao_sky.build.config import derive_traversal_region_level
from ao_sky.build.runner import _run_traversal_phase, init_build


SOURCE_CONFIG = Path(
    os.environ.get(
        "AO_SKY_BENCH_CONFIG",
        "/Volumes/Data/Galaxy/aosky/gnao-baseline/ao-sky.yaml",
    )
)
GAIA_ROOT = Path(
    os.environ.get("AO_SKY_BENCH_GAIA_ROOT", "/Users/nelsonnunes/ao-sky-cache/gaia")
)
MODEL_ROOT = Path(
    os.environ.get(
        "AO_SKY_BENCH_MODEL_ROOT",
        "/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/survey_tools/data/models",
    )
)
SURVEY_ROOT = Path(
    os.environ.get(
        "AO_SKY_BENCH_SURVEY_ROOT",
        "/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/survey_tools/data/euclid",
    )
)
REFERENCE_SAMPLE = (
    Path("/Volumes/Data/Galaxy/aosky/benchmark-runs")
    / "ao-sky-compression-sweep-bench-20260416-212643"
    / "sample-pixels.ecsv"
)
BENCH_ROOT = (
    Path("/Volumes/Data/Galaxy/aosky/benchmark-runs")
    / f"ao-sky-traversal-baseline-{time.strftime('%Y%m%d-%H%M%S')}"
)
OUTER_LEVEL = 6
WORKERS = 3
WORKER_COUNTS = tuple(
    int(value.strip())
    for value in os.environ.get("AO_SKY_BENCH_WORKERS", str(WORKERS)).split(",")
    if value.strip()
)
CACHE_ENTRIES = 8
CACHE_MB = 128
PARENT_MEMORY_LIMIT_MB = int(os.environ.get("AO_SKY_BENCH_PARENT_MEMORY_LIMIT_MB", "0"))
TELEMETRY = os.environ.get("AO_SKY_BENCH_TELEMETRY", "detailed")
SAMPLE_LIMIT = int(os.environ.get("AO_SKY_BENCH_SAMPLE_LIMIT", "0"))
PREDICT_DEVICE = os.environ.get("AO_SKY_PREDICT_DEVICE", "auto")
AVERAGED_PREDICT_DEVICE = os.environ.get("AO_SKY_AVERAGED_PREDICT_DEVICE", "auto")
PREDICT_DEVICE_POLICY = f"resolved={PREDICT_DEVICE},averaged={AVERAGED_PREDICT_DEVICE}"
PREDICTION_BATCH_SIZE = os.environ.get("AO_SKY_PREDICTION_BATCH_SIZE", "25000")
DEFAULT_BACKEND_BUCKETS = ",".join(str(value) for value in range(1000, 25001, 1000))
RESOLVED_BACKEND_BUCKETS = os.environ.get(
    "AO_SKY_RESOLVED_BACKEND_BUCKETS",
    DEFAULT_BACKEND_BUCKETS,
)
AVERAGED_BACKEND_BUCKETS = os.environ.get(
    "AO_SKY_AVERAGED_BACKEND_BUCKETS",
    DEFAULT_BACKEND_BUCKETS,
)
RESOLVED_CACHE_CLEAR_EVERY = os.environ.get("AO_SKY_RESOLVED_CACHE_CLEAR_EVERY", "never")
RSS_MONITOR_INTERVAL_S = float(
    os.environ.get("AO_SKY_BENCH_RSS_MONITOR_INTERVAL_S", "0.25")
)


PROFILE_FLOAT_FIELDS = (
    "elapsed_s",
    "pixel_s",
    "traversal_other_s",
    "traversal_profiled_s",
    "stage_star_selection_s",
    "stage_candidate_generation_s",
    "stage_filtering_s",
    "stage_bright_star_filter_s",
    "stage_inner_assignment_s",
    "stage_local_selection_s",
    "stage_inner_table_s",
    "stage_context_s",
    "stage_point_prediction_s",
    "stage_point_prediction_eligibility_s",
    "stage_point_prediction_eligibility_intersection_s",
    "stage_point_prediction_eligibility_extract_s",
    "stage_point_prediction_buffer_s",
    "stage_point_prediction_ngs_array_s",
    "stage_point_prediction_model_s",
    "stage_point_prediction_feature_s",
    "stage_point_prediction_backend_s",
    "stage_point_prediction_scatter_s",
    "stage_point_prediction_scatter_filter_s",
    "stage_point_prediction_scatter_merge_s",
    "stage_point_prediction_scatter_sort_s",
    "stage_point_prediction_scatter_write_s",
    "stage_point_prediction_cache_clear_s",
    "stage_field_mean_prediction_s",
    "stage_field_mean_prediction_feature_s",
    "stage_field_mean_prediction_backend_s",
    "stage_coverage_s",
    "stage_dust_s",
    "stage_persisted_asterisms_s",
    "artifact_write_s",
    "artifact_convert_s",
    "artifact_convert_inner_s",
    "artifact_convert_asterisms_s",
    "artifact_hdf5_s",
    "artifact_hdf5_open_s",
    "artifact_hdf5_inner_s",
    "artifact_hdf5_asterisms_s",
    "artifact_hdf5_close_s",
    "artifact_replace_s",
    "artifact_mib",
    "artifact_inner_input_mib",
    "artifact_asterism_input_mib",
    "artifact_inner_structured_mib",
    "artifact_asterism_structured_mib",
    "point_feature_mib_peak",
    "point_mps_current_mib_peak",
    "point_mps_driver_mib_peak",
    "point_mps_recommended_mib",
    "field_mean_feature_mib_peak",
    "field_mean_mps_current_mib_peak",
    "field_mean_mps_driver_mib_peak",
    "field_mean_mps_recommended_mib",
    "peak_rss_mb",
    "cache_mb",
    "cache_peak_mb",
    "gaia_load_s",
    "gaia_raw_load_s",
    "gaia_prepare_s",
)
MPS_MEMORY_SUM_FIELDS = (
    "point_mps_current_mib_peak",
    "point_mps_driver_mib_peak",
    "field_mean_mps_current_mib_peak",
    "field_mean_mps_driver_mib_peak",
)
PROFILE_INT_FIELDS = (
    "completed",
    "failed",
    "artifact_inner_rows",
    "artifact_asterism_rows",
    "search_star_rows",
    "ngs_rows",
    "close_pair_rows",
    "self_pair_rows",
    "raw_asterism_rows",
    "dedupe_key_rows",
    "post_bright_asterism_rows",
    "candidate_graph_rows",
    "local_asterism_rows",
    "context_pair_rows",
    "winner_rows",
    "winner_payload_rows",
    "point_prediction_batches",
    "point_prediction_rows",
    "point_prediction_batch_rows_peak",
    "point_prediction_backend_rows",
    "point_prediction_backend_batch_rows_peak",
    "field_mean_prediction_batches",
    "field_mean_prediction_rows",
    "field_mean_prediction_batch_rows_peak",
    "field_mean_prediction_backend_rows",
    "field_mean_prediction_backend_batch_rows_peak",
    "search_star_rows_peak",
    "ngs_rows_peak",
    "close_pair_rows_peak",
    "context_pair_rows_peak",
    "raw_asterism_rows_peak",
    "local_asterism_rows_peak",
    "winner_payload_rows_peak",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "cache_oversized_skips",
    "cache_entries",
)
PROFILE_BUCKET_FIELDS = (
    "point_prediction_backend_bucket_counts",
    "point_prediction_backend_bucket_rows",
    "field_mean_prediction_backend_bucket_counts",
    "field_mean_prediction_backend_bucket_rows",
)


class ProcessTreeRSSMonitor:
    """Sample current RSS for this process and all descendant workers."""

    def __init__(self, *, interval_s: float = RSS_MONITOR_INTERVAL_S) -> None:
        self.interval_s = max(float(interval_s), 0.05)
        self.root_pid = os.getpid()
        self.peak_mb = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ProcessTreeRSSMonitor":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sample_once()
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        try:
            total_mb = _process_tree_rss_mb(self.root_pid)
        except Exception:
            return
        self.samples += 1
        self.peak_mb = max(self.peak_mb, float(total_mb))


def _process_tree_rss_mb(root_pid: int) -> float:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return 0.0

    children_by_parent: dict[int, list[int]] = {}
    rss_by_pid: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kib = int(parts[2])
        except ValueError:
            continue
        rss_by_pid[pid] = rss_kib
        children_by_parent.setdefault(ppid, []).append(pid)

    total_kib = 0
    stack = [int(root_pid)]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_kib += rss_by_pid.get(pid, 0)
        stack.extend(children_by_parent.get(pid, ()))
    return total_kib / 1024.0


def load_outer_pixs() -> list[int]:
    explicit = os.environ.get("AO_SKY_BENCH_OUTER_PIXS")
    if explicit:
        return [int(value.strip()) for value in explicit.split(",") if value.strip()]

    table = Table.read(REFERENCE_SAMPLE)
    outer_pixs = [int(value) for value in table["outer_pix"]]
    if SAMPLE_LIMIT > 0:
        return outer_pixs[:SAMPLE_LIMIT]
    return outer_pixs


def write_benchmark_config(case_root: Path) -> Path:
    """Write a schema-v2 traversal benchmark config derived from the live config."""

    with SOURCE_CONFIG.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    payload["schema_version"] = 2

    gaia = payload.setdefault("gaia", {})
    if not isinstance(gaia, dict):
        raise TypeError("gaia must be a mapping")
    gaia.pop("min_galactic_latitude_deg", None)
    gaia.pop("max_star_density", None)
    gaia.setdefault("max_bright_star_exclusion_arcsec", None)

    asterism = payload.setdefault("asterism", {})
    if not isinstance(asterism, dict):
        raise TypeError("asterism must be a mapping")
    asterism.pop("max_overlap", None)
    asterism.setdefault("winner_ee_epsilon", 0.01)

    # Traversal does not use survey overlays. Omitting them avoids snapshot I/O
    # from dominating setup time in a traversal-only benchmark.
    payload.pop("survey_overlays", None)

    config_path = case_root / "benchmark-build.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def prepare_build(case_name: str, outer_pixs: list[int]) -> Path:
    case_root = BENCH_ROOT / case_name
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True)
    config_path = write_benchmark_config(case_root)
    build = init_build(
        config_filename=config_path,
        gaia_root=GAIA_ROOT,
        build_root=case_root,
        model_root=MODEL_ROOT,
        survey_root=SURVEY_ROOT,
    )
    with h5py.File(build / "build.h5", "r+") as handle:
        state = handle["state/outer_pixels"][:]
        state["traversal_status"] = WORK_STATUS_DONE
        state["traversal_attempt_count"] = 0
        state["traversal_last_error_message"] = b""
        state["traversal_status"][outer_pixs] = WORK_STATUS_PENDING
        handle["state/outer_pixels"][:] = state
    return build


def parse_profiles(log_text: str) -> dict[str, float]:
    totals: dict[str, float] = {key: 0.0 for key in PROFILE_FLOAT_FIELDS}
    totals.update({f"{key}_worker_sum": 0.0 for key in MPS_MEMORY_SUM_FIELDS})
    totals.update({key: 0 for key in PROFILE_INT_FIELDS})
    bucket_totals: dict[str, dict[int, int]] = {
        key: {} for key in PROFILE_BUCKET_FIELDS
    }
    for line in log_text.splitlines():
        if " phase=traversal " not in line or " profile " not in line:
            continue
        fields = {}
        for part in line.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        for key in PROFILE_FLOAT_FIELDS:
            value = float(fields.get(key, "0"))
            if key in MPS_MEMORY_SUM_FIELDS:
                totals[f"{key}_worker_sum"] = float(
                    totals[f"{key}_worker_sum"]
                ) + value
            if key in {
                "elapsed_s",
                "peak_rss_mb",
                "cache_peak_mb",
                "point_feature_mib_peak",
                "point_mps_current_mib_peak",
                "point_mps_driver_mib_peak",
                "point_mps_recommended_mib",
                "field_mean_feature_mib_peak",
                "field_mean_mps_current_mib_peak",
                "field_mean_mps_driver_mib_peak",
                "field_mean_mps_recommended_mib",
            }:
                totals[key] = max(float(totals[key]), value)
            else:
                totals[key] = float(totals[key]) + value
        for key in PROFILE_INT_FIELDS:
            if key.endswith("_peak"):
                totals[key] = max(int(totals[key]), int(fields.get(key, "0")))
            else:
                totals[key] = int(totals[key]) + int(fields.get(key, "0"))
        for key in PROFILE_BUCKET_FIELDS:
            raw = fields.get(key, "-")
            if raw in {"", "-"}:
                continue
            for part in raw.split(";"):
                if not part:
                    continue
                bucket, value = part.split(":", 1)
                bucket_int = int(bucket)
                bucket_totals[key][bucket_int] = (
                    bucket_totals[key].get(bucket_int, 0) + int(value)
                )

    completed = int(totals["completed"])
    for key, values in bucket_totals.items():
        totals[key] = ";".join(f"{bucket}:{values[bucket]}" for bucket in sorted(values))
    totals["point_prediction_padding_rows"] = int(
        totals["point_prediction_backend_rows"]
    ) - int(totals["point_prediction_rows"])
    totals["field_mean_prediction_padding_rows"] = int(
        totals["field_mean_prediction_backend_rows"]
    ) - int(totals["field_mean_prediction_rows"])
    totals["point_prediction_padding_fraction"] = (
        float(totals["point_prediction_padding_rows"])
        / int(totals["point_prediction_rows"])
        if int(totals["point_prediction_rows"])
        else 0.0
    )
    totals["field_mean_prediction_padding_fraction"] = (
        float(totals["field_mean_prediction_padding_rows"])
        / int(totals["field_mean_prediction_rows"])
        if int(totals["field_mean_prediction_rows"])
        else 0.0
    )
    totals["mps_driver_mib_peak_worker_sum"] = max(
        float(totals["point_mps_driver_mib_peak_worker_sum"]),
        float(totals["field_mean_mps_driver_mib_peak_worker_sum"]),
    )
    accesses = int(totals["cache_hits"]) + int(totals["cache_misses"])
    totals["cache_hit_ratio"] = int(totals["cache_hits"]) / accesses if accesses else 0.0
    totals["avg_artifact_mib"] = float(totals["artifact_mib"]) / completed if completed else 0.0
    totals["avg_artifact_inner_input_mib"] = (
        float(totals["artifact_inner_input_mib"]) / completed if completed else 0.0
    )
    totals["avg_artifact_asterism_input_mib"] = (
        float(totals["artifact_asterism_input_mib"]) / completed if completed else 0.0
    )
    totals["avg_artifact_inner_structured_mib"] = (
        float(totals["artifact_inner_structured_mib"]) / completed if completed else 0.0
    )
    totals["avg_artifact_asterism_structured_mib"] = (
        float(totals["artifact_asterism_structured_mib"]) / completed if completed else 0.0
    )
    totals["pixels_per_s"] = completed / float(totals["elapsed_s"]) if totals["elapsed_s"] else 0.0
    totals["avg_pixel_s"] = float(totals["pixel_s"]) / completed if completed else 0.0
    totals["avg_traversal_other_s"] = (
        float(totals["traversal_other_s"]) / completed if completed else 0.0
    )
    totals["traversal_unprofiled_s"] = (
        float(totals["pixel_s"])
        - float(totals["artifact_write_s"])
        - float(totals["traversal_profiled_s"])
    )
    totals["avg_artifact_write_s"] = (
        float(totals["artifact_write_s"]) / completed if completed else 0.0
    )
    for key in (
        "search_star_rows",
        "ngs_rows",
        "close_pair_rows",
        "self_pair_rows",
        "raw_asterism_rows",
        "dedupe_key_rows",
        "post_bright_asterism_rows",
        "candidate_graph_rows",
        "local_asterism_rows",
        "context_pair_rows",
        "winner_rows",
        "winner_payload_rows",
        "point_prediction_batches",
        "point_prediction_rows",
        "field_mean_prediction_batches",
        "field_mean_prediction_rows",
    ):
        totals[f"avg_{key}"] = int(totals[key]) / completed if completed else 0.0
    totals["avg_point_prediction_rows_per_batch"] = (
        int(totals["point_prediction_rows"]) / int(totals["point_prediction_batches"])
        if int(totals["point_prediction_batches"])
        else 0.0
    )
    totals["avg_point_prediction_backend_rows_per_batch"] = (
        int(totals["point_prediction_backend_rows"])
        / int(totals["point_prediction_batches"])
        if int(totals["point_prediction_batches"])
        else 0.0
    )
    totals["avg_field_mean_prediction_rows_per_batch"] = (
        int(totals["field_mean_prediction_rows"])
        / int(totals["field_mean_prediction_batches"])
        if int(totals["field_mean_prediction_batches"])
        else 0.0
    )
    totals["avg_field_mean_prediction_backend_rows_per_batch"] = (
        int(totals["field_mean_prediction_backend_rows"])
        / int(totals["field_mean_prediction_batches"])
        if int(totals["field_mean_prediction_batches"])
        else 0.0
    )
    for key in (
        "traversal_profiled_s",
        "traversal_unprofiled_s",
        "stage_star_selection_s",
        "stage_candidate_generation_s",
        "stage_filtering_s",
        "stage_bright_star_filter_s",
        "stage_inner_assignment_s",
        "stage_local_selection_s",
        "stage_inner_table_s",
        "stage_context_s",
        "stage_point_prediction_s",
        "stage_point_prediction_eligibility_s",
        "stage_point_prediction_eligibility_intersection_s",
        "stage_point_prediction_eligibility_extract_s",
        "stage_point_prediction_buffer_s",
        "stage_point_prediction_ngs_array_s",
        "stage_point_prediction_model_s",
        "stage_point_prediction_feature_s",
        "stage_point_prediction_backend_s",
        "stage_point_prediction_scatter_s",
        "stage_point_prediction_scatter_filter_s",
        "stage_point_prediction_scatter_merge_s",
        "stage_point_prediction_scatter_sort_s",
        "stage_point_prediction_scatter_write_s",
        "stage_point_prediction_cache_clear_s",
        "stage_field_mean_prediction_s",
        "stage_field_mean_prediction_feature_s",
        "stage_field_mean_prediction_backend_s",
        "stage_coverage_s",
        "stage_dust_s",
        "stage_persisted_asterisms_s",
    ):
        totals[f"avg_{key}"] = float(totals[key]) / completed if completed else 0.0
    return totals


def run_case(case_name: str, outer_pixs: list[int]) -> dict[str, object]:
    build = prepare_build(case_name, outer_pixs)
    workers = int(case_name.rsplit("workers", 1)[-1])
    config = TraversalExecutionConfig(
        workers=workers,
        gaia_cache_entries=CACHE_ENTRIES,
        gaia_cache_mb=CACHE_MB,
        region_level=derive_traversal_region_level(
            outer_level=OUTER_LEVEL,
            workers=workers,
        ),
        parent_memory_limit_mb=PARENT_MEMORY_LIMIT_MB,
        telemetry=TELEMETRY,
    )
    started = time.perf_counter()
    with ProcessTreeRSSMonitor() as rss_monitor:
        complete, counts = _run_traversal_phase(build, execution_config=config)
    wall_s = time.perf_counter() - started
    log_text = (build / "build.log").read_text(encoding="utf-8")
    stats = parse_profiles(log_text)
    stats.update(
        {
            "case": case_name,
            "predict_device": PREDICT_DEVICE,
            "predict_device_policy": PREDICT_DEVICE_POLICY,
            "prediction_batch_size": PREDICTION_BATCH_SIZE,
            "resolved_backend_buckets": RESOLVED_BACKEND_BUCKETS,
            "averaged_backend_buckets": AVERAGED_BACKEND_BUCKETS,
            "resolved_cache_clear_every": RESOLVED_CACHE_CLEAR_EVERY,
            "workers": workers,
            "wall_s": wall_s,
            "process_tree_rss_mb_high_water": rss_monitor.peak_mb,
            "process_tree_rss_samples": rss_monitor.samples,
            "observed_total_ram_mb_high_water": (
                rss_monitor.peak_mb + float(stats["mps_driver_mib_peak_worker_sum"])
            ),
            "complete": complete,
            "pending": counts["pending"],
            "running": counts["running"],
            "done": counts["done"],
            "failed_state": counts["failed"],
            "build_dir": str(build),
        }
    )
    print("SUMMARY", stats, flush=True)
    return stats


def main() -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    outer_pixs = load_outer_pixs()
    Table({"outer_pix": outer_pixs}).write(BENCH_ROOT / "sample-pixels.ecsv", overwrite=True)
    print("BENCH_ROOT", BENCH_ROOT, flush=True)
    print("SOURCE_CONFIG", SOURCE_CONFIG, flush=True)
    print("GAIA_ROOT", GAIA_ROOT, flush=True)
    print("MODEL_ROOT", MODEL_ROOT, flush=True)
    print("SURVEY_ROOT", SURVEY_ROOT, flush=True)
    print("PIXEL_COUNT", len(outer_pixs), flush=True)
    print("SAMPLE_LIMIT", SAMPLE_LIMIT, flush=True)
    print("WORKER_COUNTS", ",".join(str(worker) for worker in WORKER_COUNTS), flush=True)
    print("PARENT_MEMORY_LIMIT_MB", PARENT_MEMORY_LIMIT_MB, flush=True)
    print("PREDICT_DEVICE", PREDICT_DEVICE, flush=True)
    print("PREDICT_DEVICE_POLICY", PREDICT_DEVICE_POLICY, flush=True)
    print("PREDICTION_BATCH_SIZE", PREDICTION_BATCH_SIZE, flush=True)
    print("RESOLVED_BACKEND_BUCKETS", RESOLVED_BACKEND_BUCKETS, flush=True)
    print("AVERAGED_BACKEND_BUCKETS", AVERAGED_BACKEND_BUCKETS, flush=True)
    print("RESOLVED_CACHE_CLEAR_EVERY", RESOLVED_CACHE_CLEAR_EVERY, flush=True)
    results = [
        run_case(f"ssd_gaia_cache_blosc_zstd_direct_hdd_workers{workers}", outer_pixs)
        for workers in WORKER_COUNTS
    ]
    keys = [
        "case",
        "predict_device",
        "predict_device_policy",
        "prediction_batch_size",
        "resolved_backend_buckets",
        "averaged_backend_buckets",
        "resolved_cache_clear_every",
        "workers",
        "wall_s",
        "process_tree_rss_mb_high_water",
        "process_tree_rss_samples",
        "mps_driver_mib_peak_worker_sum",
        "observed_total_ram_mb_high_water",
        "completed",
        "elapsed_s",
        "pixels_per_s",
        "avg_pixel_s",
        "traversal_other_s",
        "avg_traversal_other_s",
        "traversal_profiled_s",
        "avg_traversal_profiled_s",
        "traversal_unprofiled_s",
        "avg_traversal_unprofiled_s",
        "stage_star_selection_s",
        "avg_stage_star_selection_s",
        "stage_candidate_generation_s",
        "avg_stage_candidate_generation_s",
        "stage_filtering_s",
        "avg_stage_filtering_s",
        "stage_bright_star_filter_s",
        "avg_stage_bright_star_filter_s",
        "stage_inner_assignment_s",
        "avg_stage_inner_assignment_s",
        "stage_local_selection_s",
        "avg_stage_local_selection_s",
        "stage_inner_table_s",
        "avg_stage_inner_table_s",
        "stage_context_s",
        "avg_stage_context_s",
        "stage_point_prediction_s",
        "avg_stage_point_prediction_s",
        "stage_point_prediction_eligibility_s",
        "avg_stage_point_prediction_eligibility_s",
        "stage_point_prediction_eligibility_intersection_s",
        "avg_stage_point_prediction_eligibility_intersection_s",
        "stage_point_prediction_eligibility_extract_s",
        "avg_stage_point_prediction_eligibility_extract_s",
        "stage_point_prediction_buffer_s",
        "avg_stage_point_prediction_buffer_s",
        "stage_point_prediction_ngs_array_s",
        "avg_stage_point_prediction_ngs_array_s",
        "stage_point_prediction_model_s",
        "avg_stage_point_prediction_model_s",
        "stage_point_prediction_feature_s",
        "avg_stage_point_prediction_feature_s",
        "stage_point_prediction_backend_s",
        "avg_stage_point_prediction_backend_s",
        "stage_point_prediction_scatter_s",
        "avg_stage_point_prediction_scatter_s",
        "stage_point_prediction_scatter_filter_s",
        "avg_stage_point_prediction_scatter_filter_s",
        "stage_point_prediction_scatter_merge_s",
        "avg_stage_point_prediction_scatter_merge_s",
        "stage_point_prediction_scatter_sort_s",
        "avg_stage_point_prediction_scatter_sort_s",
        "stage_point_prediction_scatter_write_s",
        "avg_stage_point_prediction_scatter_write_s",
        "stage_point_prediction_cache_clear_s",
        "avg_stage_point_prediction_cache_clear_s",
        "stage_field_mean_prediction_s",
        "avg_stage_field_mean_prediction_s",
        "stage_field_mean_prediction_feature_s",
        "avg_stage_field_mean_prediction_feature_s",
        "stage_field_mean_prediction_backend_s",
        "avg_stage_field_mean_prediction_backend_s",
        "stage_coverage_s",
        "avg_stage_coverage_s",
        "stage_dust_s",
        "avg_stage_dust_s",
        "stage_persisted_asterisms_s",
        "avg_stage_persisted_asterisms_s",
        "artifact_write_s",
        "avg_artifact_write_s",
        "artifact_hdf5_s",
        "artifact_mib",
        "avg_artifact_mib",
        "artifact_inner_input_mib",
        "avg_artifact_inner_input_mib",
        "artifact_asterism_input_mib",
        "avg_artifact_asterism_input_mib",
        "artifact_inner_structured_mib",
        "avg_artifact_inner_structured_mib",
        "artifact_asterism_structured_mib",
        "avg_artifact_asterism_structured_mib",
        "search_star_rows",
        "avg_search_star_rows",
        "search_star_rows_peak",
        "ngs_rows",
        "avg_ngs_rows",
        "ngs_rows_peak",
        "close_pair_rows",
        "avg_close_pair_rows",
        "close_pair_rows_peak",
        "self_pair_rows",
        "avg_self_pair_rows",
        "raw_asterism_rows",
        "avg_raw_asterism_rows",
        "raw_asterism_rows_peak",
        "dedupe_key_rows",
        "avg_dedupe_key_rows",
        "post_bright_asterism_rows",
        "avg_post_bright_asterism_rows",
        "candidate_graph_rows",
        "avg_candidate_graph_rows",
        "local_asterism_rows",
        "avg_local_asterism_rows",
        "local_asterism_rows_peak",
        "context_pair_rows",
        "avg_context_pair_rows",
        "context_pair_rows_peak",
        "winner_rows",
        "avg_winner_rows",
        "winner_payload_rows",
        "avg_winner_payload_rows",
        "winner_payload_rows_peak",
        "point_prediction_batches",
        "avg_point_prediction_batches",
        "point_prediction_rows",
        "avg_point_prediction_rows",
        "avg_point_prediction_rows_per_batch",
        "point_prediction_batch_rows_peak",
        "point_prediction_backend_rows",
        "avg_point_prediction_backend_rows_per_batch",
        "point_prediction_backend_batch_rows_peak",
        "point_prediction_padding_rows",
        "point_prediction_padding_fraction",
        "point_prediction_backend_bucket_counts",
        "point_prediction_backend_bucket_rows",
        "point_feature_mib_peak",
        "point_mps_current_mib_peak",
        "point_mps_current_mib_peak_worker_sum",
        "point_mps_driver_mib_peak",
        "point_mps_driver_mib_peak_worker_sum",
        "point_mps_recommended_mib",
        "field_mean_prediction_batches",
        "avg_field_mean_prediction_batches",
        "field_mean_prediction_rows",
        "avg_field_mean_prediction_rows",
        "avg_field_mean_prediction_rows_per_batch",
        "field_mean_prediction_batch_rows_peak",
        "field_mean_prediction_backend_rows",
        "avg_field_mean_prediction_backend_rows_per_batch",
        "field_mean_prediction_backend_batch_rows_peak",
        "field_mean_prediction_padding_rows",
        "field_mean_prediction_padding_fraction",
        "field_mean_prediction_backend_bucket_counts",
        "field_mean_prediction_backend_bucket_rows",
        "field_mean_feature_mib_peak",
        "field_mean_mps_current_mib_peak",
        "field_mean_mps_current_mib_peak_worker_sum",
        "field_mean_mps_driver_mib_peak",
        "field_mean_mps_driver_mib_peak_worker_sum",
        "field_mean_mps_recommended_mib",
        "gaia_load_s",
        "gaia_raw_load_s",
        "gaia_prepare_s",
        "cache_hits",
        "cache_misses",
        "cache_hit_ratio",
        "cache_evictions",
        "cache_peak_mb",
        "peak_rss_mb",
        "pending",
        "running",
        "done",
        "failed_state",
        "build_dir",
    ]
    csv_path = BENCH_ROOT / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in keys})
    print("CSV", csv_path, flush=True)
    print("TABLE")
    print(",".join(keys))
    for result in results:
        print(",".join(str(result.get(key, "")) for key in keys), flush=True)


if __name__ == "__main__":
    main()
