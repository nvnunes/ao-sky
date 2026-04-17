"""Benchmark the current Traversal baseline with detailed stage telemetry."""

from __future__ import annotations

import csv
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

from ao_sky.build._constants import BUILD_STATUS_RUNNING, WORK_STATUS_DONE, WORK_STATUS_PENDING
from ao_sky.build._models import TraversalExecutionConfig
from ao_sky.build.config import derive_traversal_region_level
from ao_sky.build.runner import _run_traversal_phase


SOURCE_BUILD = Path("/Volumes/Data/Galaxy/aosky/gnao-baseline/v1")
GAIA_ROOT = Path("/Users/nelsonnunes/ao-sky-cache/gaia")
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
MEMORY_LIMIT_MB = int(os.environ.get("AO_SKY_BENCH_WORKER_MEMORY_LIMIT_MB", "6144"))
PARENT_MEMORY_LIMIT_MB = int(os.environ.get("AO_SKY_BENCH_PARENT_MEMORY_LIMIT_MB", "0"))
TELEMETRY = os.environ.get("AO_SKY_BENCH_TELEMETRY", "detailed")


PROFILE_FLOAT_FIELDS = (
    "elapsed_s",
    "pixel_s",
    "traversal_other_s",
    "traversal_profiled_s",
    "stage_star_selection_s",
    "stage_candidate_generation_s",
    "stage_filtering_s",
    "stage_bright_star_filter_s",
    "stage_overlap_quality_s",
    "stage_overlap_geometry_s",
    "stage_inner_assignment_s",
    "stage_local_selection_s",
    "stage_inner_table_s",
    "stage_context_s",
    "stage_point_prediction_s",
    "stage_field_mean_prediction_s",
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
    "peak_rss_mb",
    "cache_mb",
    "cache_peak_mb",
    "gaia_load_s",
    "gaia_raw_load_s",
    "gaia_prepare_s",
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
    "post_overlap_asterism_rows",
    "local_asterism_rows",
    "context_pair_rows",
    "winner_rows",
    "winner_payload_rows",
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


def text_dataset(value: str) -> np.bytes_:
    return np.bytes_(value)


def load_outer_pixs() -> list[int]:
    table = Table.read(REFERENCE_SAMPLE)
    return [int(value) for value in table["outer_pix"]]


def copy_or_link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def prepare_build(case_name: str, outer_pixs: list[int]) -> Path:
    build = BENCH_ROOT / case_name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    shutil.copy2(SOURCE_BUILD / "build.h5", build / "build.h5")
    shutil.copy2(SOURCE_BUILD / "build.yaml", build / "build.yaml")
    copy_or_link(SOURCE_BUILD / "dust", build / "dust")
    copy_or_link(SOURCE_BUILD / "models", build / "models")
    copy_or_link(SOURCE_BUILD / "surveys", build / "surveys")
    with h5py.File(build / "build.h5", "r+") as handle:
        config = handle["metadata/config"]
        config["build_status"][()] = text_dataset(BUILD_STATUS_RUNNING)
        config["current_phase"][()] = text_dataset("traversal")
        config["gaia_root"][()] = text_dataset(str(GAIA_ROOT))
        state = handle["state/outer_pixels"][:]
        state["traversal_status"] = WORK_STATUS_DONE
        state["traversal_last_error_message"] = b""
        state["traversal_status"][outer_pixs] = WORK_STATUS_PENDING
        handle["state/outer_pixels"][:] = state
    return build


def parse_profiles(log_text: str) -> dict[str, float]:
    totals: dict[str, float] = {key: 0.0 for key in PROFILE_FLOAT_FIELDS}
    totals.update({key: 0 for key in PROFILE_INT_FIELDS})
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
            if key in {"elapsed_s", "peak_rss_mb", "cache_peak_mb"}:
                totals[key] = max(float(totals[key]), float(fields.get(key, "0")))
            else:
                totals[key] = float(totals[key]) + float(fields.get(key, "0"))
        for key in PROFILE_INT_FIELDS:
            if key.endswith("_peak"):
                totals[key] = max(int(totals[key]), int(fields.get(key, "0")))
            else:
                totals[key] = int(totals[key]) + int(fields.get(key, "0"))

    completed = int(totals["completed"])
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
        "post_overlap_asterism_rows",
        "local_asterism_rows",
        "context_pair_rows",
        "winner_rows",
        "winner_payload_rows",
    ):
        totals[f"avg_{key}"] = int(totals[key]) / completed if completed else 0.0
    for key in (
        "traversal_profiled_s",
        "traversal_unprofiled_s",
        "stage_star_selection_s",
        "stage_candidate_generation_s",
        "stage_filtering_s",
        "stage_bright_star_filter_s",
        "stage_overlap_quality_s",
        "stage_overlap_geometry_s",
        "stage_inner_assignment_s",
        "stage_local_selection_s",
        "stage_inner_table_s",
        "stage_context_s",
        "stage_point_prediction_s",
        "stage_field_mean_prediction_s",
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
        region_level=derive_traversal_region_level(outer_level=OUTER_LEVEL, workers=workers),
        worker_memory_limit_mb=MEMORY_LIMIT_MB,
        parent_memory_limit_mb=PARENT_MEMORY_LIMIT_MB,
        telemetry=TELEMETRY,
    )
    started = time.perf_counter()
    complete, counts = _run_traversal_phase(build, execution_config=config)
    wall_s = time.perf_counter() - started
    log_text = (build / "build.log").read_text(encoding="utf-8")
    stats = parse_profiles(log_text)
    stats.update(
        {
            "case": case_name,
            "workers": workers,
            "wall_s": wall_s,
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
    print("PIXEL_COUNT", len(outer_pixs), flush=True)
    print("WORKER_COUNTS", ",".join(str(worker) for worker in WORKER_COUNTS), flush=True)
    print("WORKER_MEMORY_LIMIT_MB", MEMORY_LIMIT_MB, flush=True)
    print("PARENT_MEMORY_LIMIT_MB", PARENT_MEMORY_LIMIT_MB, flush=True)
    results = [
        run_case(f"ssd_gaia_cache_blosc_zstd_direct_hdd_workers{workers}", outer_pixs)
        for workers in WORKER_COUNTS
    ]
    keys = [
        "case",
        "workers",
        "wall_s",
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
        "stage_overlap_quality_s",
        "avg_stage_overlap_quality_s",
        "stage_overlap_geometry_s",
        "avg_stage_overlap_geometry_s",
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
        "stage_field_mean_prediction_s",
        "avg_stage_field_mean_prediction_s",
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
        "post_overlap_asterism_rows",
        "avg_post_overlap_asterism_rows",
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
