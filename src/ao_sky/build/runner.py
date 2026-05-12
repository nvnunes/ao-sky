"""Build creation, execution, and summary helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass, field
import gc
import multiprocessing
from pathlib import Path
from queue import Empty
import resource
import subprocess
import sys
import time

import numpy as np
from astropy.table import Table

from ..dust import prepare_gaia_tge_a0_cache
from ..gaia import GaiaHealpixStore, GaiaStoreConfig, GaiaSummaryStore
from ..predict import PredictRuntime, backend as predict_backend
from ..predict import configure_inference_threads, warm_model_cache
from ..spatial import get_parent_pixel
from .augmentation import build_survey_extent_layers
from .aggregation import build_maps
from ._constants import (
    BUILD_PHASE_AGGREGATION,
    BUILD_PHASE_AUGMENTATION,
    BUILD_PHASE_TRAVERSAL,
    BUILD_STATUS_COMPLETED,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_RUNNING,
    STATE_ERROR_MAX_BYTES,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ._exceptions import BuildError
from .artifacts import (
    ArtifactMemoryProfile,
    ArtifactWriteProfile,
    write_outer_artifact_profiled,
)
from .config import (
    load_build_definition,
    resolve_build_root_only,
    resolve_build_roots,
    resolve_traversal_execution_config,
)
from .control import (
    append_build_log,
    build_artifact_root,
    create_build_root,
    latest_build_path,
    load_build_definition as load_persisted_build_definition,
    load_build_roots,
    load_current_phase,
    inspect_build,
    load_runtime_config_path,
    load_state,
    outer_artifact_filename,
    phase_state_fields,
    set_current_phase,
    set_dust_root,
    set_build_status,
    update_state_row,
    write_state_rows,
)
from .runtime_config import load_runtime_config, write_runtime_config
from .model_snapshot import fetch_model_data
from .survey_snapshot import fetch_survey_data
from ._models import (
    BuildDefinition,
    TraversalCacheStats,
    TraversalExecutionConfig,
    TraversalMemorySample,
    TraversalStageStats,
    TraversalStructureStats,
    TraversalTaskContext,
    TraversalTaskResult,
    TraversalWorkBatch,
    TraversalWorkerMessage,
    TraversalWorkerMemorySample,
    TraversalWorkerPlan,
    TraversalWorkerStats,
)
from .regional import (
    build_dynamic_work_batches,
    build_regional_worker_plans,
    dynamic_model_name,
)
from .runtime_gaia import RuntimeGaiaHealpixStore
from .traversal import (
    TraversalGeometry,
    TraversalMemoryProfile,
    TraversalStageProfile,
    TraversalStructureProfile,
    build_traversal_products,
)

TRAVERSAL_PROGRESS_LOG_INTERVAL = 100
TRUNCATED_ERROR_SUFFIX = "... [truncated]"
STATE_UPDATE_RETRY_ATTEMPTS = 20
STATE_UPDATE_RETRY_DELAY_SECONDS = 0.05
STATE_UPDATE_FLUSH_INTERVAL = 100
PARENT_MEMORY_CHECK_INTERVAL_SECONDS = 1.0
PARENT_MEMORY_SOFT_FRACTION = 0.85
PARENT_MEMORY_SOFT_RELEASE_FRACTION = 0.80
PARENT_MEMORY_HARD_FRACTION = 0.95
PARENT_MEMORY_HARD_RELEASE_FRACTION = 0.90
PARENT_MEMORY_PRESSURE_NORMAL = "normal"
PARENT_MEMORY_PRESSURE_TRIM = "trim"
PARENT_MEMORY_PRESSURE_PAUSE = "pause"
PARENT_MEMORY_PRESSURE_RESUME = "resume"
PARENT_MEMORY_PRESSURE_COMMAND_INTERVAL_SECONDS = 5.0
PARENT_MEMORY_SNAPSHOT_INTERVAL_SECONDS = 60.0
WORKER_MEMORY_PRESSURE_SLEEP_SECONDS = 0.25
WORKER_MEMORY_PRESSURE_MAX_PAUSE_SECONDS = 30.0


def init_build(
    *,
    config_filename: Path,
    gaia_root: Path | None,
    build_root: Path | None,
    model_root: Path | None = None,
    survey_root: Path | None = None,
    aosky_yaml: Path | None = None,
) -> Path:
    """Create a new build root and seed its initial metadata/state."""

    definition, definition_yaml = load_build_definition(config_filename)
    roots = resolve_build_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        model_root=model_root,
        aosky_yaml=aosky_yaml,
    )
    _require_gaia_summary(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
    )
    runtime = load_runtime_config(
        config_filename,
        model_root=roots.model_root,
    )
    _validate_runtime_matches_build_definition(definition, runtime)
    build_path = create_build_root(
        definition=definition,
        definition_yaml=definition_yaml,
        roots=roots,
        runtime_config_source_path=config_filename,
    )
    write_runtime_config(build_path, runtime)
    try:
        build_dust_root = build_path / "dust"
        prepare_gaia_tge_a0_cache(
            source_dust_root=roots.dust_root,
            destination_dust_root=build_dust_root,
            level=definition.max_data_level,
        )
        set_dust_root(build_path, build_dust_root)
        fetch_model_data(
            build_path,
            model_root=roots.model_root,
            aosky_yaml=aosky_yaml,
        )
        if definition.survey_extent_overlays:
            fetch_survey_data(
                build_path,
                survey_root=survey_root,
                aosky_yaml=aosky_yaml,
            )
    except Exception as exc:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        append_build_log(build_path, f"init snapshot failed: {exc}")
        raise
    return build_path


def _validate_runtime_matches_build_definition(
    definition: BuildDefinition,
    runtime: PredictRuntime,
) -> None:
    if int(runtime.outer_level) != int(definition.outer_level):
        raise BuildError(
            "Runtime config traversal.outer_level must match build definition "
            f"outer_level ({runtime.outer_level} != {definition.outer_level})"
        )
    if int(runtime.inner_level) != int(definition.inner_level):
        raise BuildError(
            "Runtime config traversal.inner_level must match build definition "
            f"inner_level ({runtime.inner_level} != {definition.inner_level})"
        )


def _load_gaia_star_counts(
    *,
    gaia_root: Path,
    gaia_release: str,
    outer_level: int,
) -> np.ndarray:
    summary = GaiaSummaryStore(
        GaiaStoreConfig(
            root=gaia_root,
            release=gaia_release,
            healpix_level=outer_level,
        )
    ).load_summary()
    expected_rows = 12 * (4 ** outer_level)
    if len(summary) != expected_rows:
        raise BuildError(
            "Gaia summary has wrong size; rerun "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}`"
        )
    loaded = np.asarray(summary["loaded"], dtype=np.bool_)
    if not np.all(loaded):
        missing = np.flatnonzero(~loaded)
        raise BuildError(
            "Gaia summary is incomplete; rerun "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}` "
            f"(first missing outer_pix={int(missing[0])})"
        )
    return np.asarray(summary["star_count"], dtype=np.int64)


def _require_gaia_summary(
    *,
    gaia_root: Path,
    gaia_release: str,
    outer_level: int,
) -> None:
    try:
        _load_gaia_star_counts(
            gaia_root=gaia_root,
            gaia_release=gaia_release,
            outer_level=outer_level,
        )
    except Exception as exc:
        if isinstance(exc, BuildError):
            raise
        raise BuildError(
            "Missing Gaia summary for build initialization; run "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}` "
            "first"
        ) from exc


def _load_traversal_task_context(build_path: Path) -> TraversalTaskContext:
    """Load build metadata once in the parent process for worker tasks."""

    return TraversalTaskContext(
        build_path=build_path,
        definition=load_persisted_build_definition(build_path),
        roots=load_build_roots(build_path),
        runtime_config_path=load_runtime_config_path(build_path),
    )


@dataclass(slots=True)
class _StateUpdateTelemetry:
    """Low-volume counters for parent-owned build-state updates."""

    updates: int = 0
    flushes: int = 0
    rows_written: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    lock_retries: int = 0
    lock_wait_seconds: float = 0.0
    dirty_rows_peak: int = 0


@dataclass(slots=True)
class _ArtifactWriteTelemetry:
    """Accumulated worker-owned outer-artifact write timings."""

    total_seconds: float = 0.0
    convert_inner_seconds: float = 0.0
    convert_asterisms_seconds: float = 0.0
    hdf5_open_seconds: float = 0.0
    hdf5_inner_seconds: float = 0.0
    hdf5_asterisms_seconds: float = 0.0
    hdf5_close_seconds: float = 0.0
    replace_seconds: float = 0.0
    output_bytes: int = 0
    inner_input_bytes: int = 0
    asterism_input_bytes: int = 0
    inner_structured_bytes: int = 0
    asterism_structured_bytes: int = 0
    inner_rows: int = 0
    asterism_rows: int = 0

    def add(self, profile: ArtifactWriteProfile) -> None:
        self.total_seconds += profile.total_seconds
        self.convert_inner_seconds += profile.convert_inner_seconds
        self.convert_asterisms_seconds += profile.convert_asterisms_seconds
        self.hdf5_open_seconds += profile.hdf5_open_seconds
        self.hdf5_inner_seconds += profile.hdf5_inner_seconds
        self.hdf5_asterisms_seconds += profile.hdf5_asterisms_seconds
        self.hdf5_close_seconds += profile.hdf5_close_seconds
        self.replace_seconds += profile.replace_seconds
        self.output_bytes += int(profile.output_bytes)
        self.inner_input_bytes += int(profile.inner_input_bytes)
        self.asterism_input_bytes += int(profile.asterism_input_bytes)
        self.inner_structured_bytes += int(profile.inner_structured_bytes)
        self.asterism_structured_bytes += int(profile.asterism_structured_bytes)
        self.inner_rows += int(profile.inner_rows)
        self.asterism_rows += int(profile.asterism_rows)


@dataclass(slots=True)
class _TraversalStageTelemetry:
    """Accumulated worker-owned Traversal pipeline timings."""

    star_selection_seconds: float = 0.0
    candidate_generation_seconds: float = 0.0
    filtering_seconds: float = 0.0
    bright_star_filter_seconds: float = 0.0
    inner_assignment_seconds: float = 0.0
    local_selection_seconds: float = 0.0
    inner_table_seconds: float = 0.0
    context_seconds: float = 0.0
    point_prediction_seconds: float = 0.0
    point_prediction_eligibility_seconds: float = 0.0
    point_prediction_eligibility_intersection_seconds: float = 0.0
    point_prediction_eligibility_extract_seconds: float = 0.0
    point_prediction_buffer_seconds: float = 0.0
    point_prediction_ngs_array_seconds: float = 0.0
    point_prediction_model_seconds: float = 0.0
    point_prediction_feature_seconds: float = 0.0
    point_prediction_backend_seconds: float = 0.0
    point_prediction_scatter_seconds: float = 0.0
    point_prediction_scatter_filter_seconds: float = 0.0
    point_prediction_scatter_merge_seconds: float = 0.0
    point_prediction_scatter_sort_seconds: float = 0.0
    point_prediction_scatter_write_seconds: float = 0.0
    point_prediction_cache_clear_seconds: float = 0.0
    field_mean_prediction_seconds: float = 0.0
    field_mean_prediction_feature_seconds: float = 0.0
    field_mean_prediction_backend_seconds: float = 0.0
    coverage_seconds: float = 0.0
    dust_seconds: float = 0.0
    persisted_asterisms_seconds: float = 0.0

    def add(self, stats: TraversalStageStats) -> None:
        self.star_selection_seconds += stats.star_selection_seconds
        self.candidate_generation_seconds += stats.candidate_generation_seconds
        self.filtering_seconds += stats.filtering_seconds
        self.bright_star_filter_seconds += stats.bright_star_filter_seconds
        self.inner_assignment_seconds += stats.inner_assignment_seconds
        self.local_selection_seconds += stats.local_selection_seconds
        self.inner_table_seconds += stats.inner_table_seconds
        self.context_seconds += stats.context_seconds
        self.point_prediction_seconds += stats.point_prediction_seconds
        self.point_prediction_eligibility_seconds += (
            stats.point_prediction_eligibility_seconds
        )
        self.point_prediction_eligibility_intersection_seconds += (
            stats.point_prediction_eligibility_intersection_seconds
        )
        self.point_prediction_eligibility_extract_seconds += (
            stats.point_prediction_eligibility_extract_seconds
        )
        self.point_prediction_buffer_seconds += stats.point_prediction_buffer_seconds
        self.point_prediction_ngs_array_seconds += (
            stats.point_prediction_ngs_array_seconds
        )
        self.point_prediction_model_seconds += stats.point_prediction_model_seconds
        self.point_prediction_feature_seconds += stats.point_prediction_feature_seconds
        self.point_prediction_backend_seconds += stats.point_prediction_backend_seconds
        self.point_prediction_scatter_seconds += stats.point_prediction_scatter_seconds
        self.point_prediction_scatter_filter_seconds += (
            stats.point_prediction_scatter_filter_seconds
        )
        self.point_prediction_scatter_merge_seconds += (
            stats.point_prediction_scatter_merge_seconds
        )
        self.point_prediction_scatter_sort_seconds += (
            stats.point_prediction_scatter_sort_seconds
        )
        self.point_prediction_scatter_write_seconds += (
            stats.point_prediction_scatter_write_seconds
        )
        self.point_prediction_cache_clear_seconds += (
            stats.point_prediction_cache_clear_seconds
        )
        self.field_mean_prediction_seconds += stats.field_mean_prediction_seconds
        self.field_mean_prediction_feature_seconds += (
            stats.field_mean_prediction_feature_seconds
        )
        self.field_mean_prediction_backend_seconds += (
            stats.field_mean_prediction_backend_seconds
        )
        self.coverage_seconds += stats.coverage_seconds
        self.dust_seconds += stats.dust_seconds
        self.persisted_asterisms_seconds += stats.persisted_asterisms_seconds

    def to_stats(self) -> TraversalStageStats:
        return TraversalStageStats(
            star_selection_seconds=self.star_selection_seconds,
            candidate_generation_seconds=self.candidate_generation_seconds,
            filtering_seconds=self.filtering_seconds,
            bright_star_filter_seconds=self.bright_star_filter_seconds,
            inner_assignment_seconds=self.inner_assignment_seconds,
            local_selection_seconds=self.local_selection_seconds,
            inner_table_seconds=self.inner_table_seconds,
            context_seconds=self.context_seconds,
            point_prediction_seconds=self.point_prediction_seconds,
            point_prediction_eligibility_seconds=(
                self.point_prediction_eligibility_seconds
            ),
            point_prediction_eligibility_intersection_seconds=(
                self.point_prediction_eligibility_intersection_seconds
            ),
            point_prediction_eligibility_extract_seconds=(
                self.point_prediction_eligibility_extract_seconds
            ),
            point_prediction_buffer_seconds=self.point_prediction_buffer_seconds,
            point_prediction_ngs_array_seconds=self.point_prediction_ngs_array_seconds,
            point_prediction_model_seconds=self.point_prediction_model_seconds,
            point_prediction_feature_seconds=self.point_prediction_feature_seconds,
            point_prediction_backend_seconds=self.point_prediction_backend_seconds,
            point_prediction_scatter_seconds=self.point_prediction_scatter_seconds,
            point_prediction_scatter_filter_seconds=(
                self.point_prediction_scatter_filter_seconds
            ),
            point_prediction_scatter_merge_seconds=(
                self.point_prediction_scatter_merge_seconds
            ),
            point_prediction_scatter_sort_seconds=(
                self.point_prediction_scatter_sort_seconds
            ),
            point_prediction_scatter_write_seconds=(
                self.point_prediction_scatter_write_seconds
            ),
            point_prediction_cache_clear_seconds=(
                self.point_prediction_cache_clear_seconds
            ),
            field_mean_prediction_seconds=self.field_mean_prediction_seconds,
            field_mean_prediction_feature_seconds=(
                self.field_mean_prediction_feature_seconds
            ),
            field_mean_prediction_backend_seconds=(
                self.field_mean_prediction_backend_seconds
            ),
            coverage_seconds=self.coverage_seconds,
            dust_seconds=self.dust_seconds,
            persisted_asterisms_seconds=self.persisted_asterisms_seconds,
        )


def _merge_bucket_stats(target: dict[int, int], values: dict[int, int]) -> None:
    for bucket, count in values.items():
        target[int(bucket)] = target.get(int(bucket), 0) + int(count)


def _format_bucket_stats(values: tuple[tuple[int, int], ...]) -> str:
    return ";".join(f"{bucket}:{count}" for bucket, count in values)


@dataclass(slots=True)
class _TraversalStructureTelemetry:
    """Accumulated worker-owned Traversal intermediate cardinality counters."""

    search_star_rows: int = 0
    ngs_rows: int = 0
    close_pair_rows: int = 0
    self_pair_rows: int = 0
    raw_asterism_rows: int = 0
    dedupe_key_rows: int = 0
    post_bright_asterism_rows: int = 0
    candidate_graph_rows: int = 0
    local_asterism_rows: int = 0
    context_pair_rows: int = 0
    winner_rows: int = 0
    winner_payload_rows: int = 0
    point_prediction_batches: int = 0
    point_prediction_rows: int = 0
    recovered_point_prediction_rows: int = 0
    recovered_winner_pixels: int = 0
    point_prediction_batch_rows_peak: int = 0
    point_prediction_backend_rows: int = 0
    point_prediction_backend_batch_rows_peak: int = 0
    point_feature_bytes_peak: int = 0
    point_mps_current_bytes_peak: int = 0
    point_mps_driver_bytes_peak: int = 0
    point_mps_recommended_bytes: int = 0
    point_backend_bucket_counts: dict[int, int] = field(default_factory=dict)
    point_backend_bucket_rows: dict[int, int] = field(default_factory=dict)
    field_mean_prediction_batches: int = 0
    field_mean_prediction_rows: int = 0
    field_mean_prediction_batch_rows_peak: int = 0
    field_mean_prediction_backend_rows: int = 0
    field_mean_prediction_backend_batch_rows_peak: int = 0
    field_mean_feature_bytes_peak: int = 0
    field_mean_mps_current_bytes_peak: int = 0
    field_mean_mps_driver_bytes_peak: int = 0
    field_mean_mps_recommended_bytes: int = 0
    field_mean_backend_bucket_counts: dict[int, int] = field(default_factory=dict)
    field_mean_backend_bucket_rows: dict[int, int] = field(default_factory=dict)
    search_star_rows_peak: int = 0
    ngs_rows_peak: int = 0
    close_pair_rows_peak: int = 0
    context_pair_rows_peak: int = 0
    raw_asterism_rows_peak: int = 0
    local_asterism_rows_peak: int = 0
    winner_payload_rows_peak: int = 0

    def add(self, stats: TraversalStructureStats) -> None:
        self.search_star_rows += int(stats.search_star_rows)
        self.ngs_rows += int(stats.ngs_rows)
        self.close_pair_rows += int(stats.close_pair_rows)
        self.self_pair_rows += int(stats.self_pair_rows)
        self.raw_asterism_rows += int(stats.raw_asterism_rows)
        self.dedupe_key_rows += int(stats.dedupe_key_rows)
        self.post_bright_asterism_rows += int(stats.post_bright_asterism_rows)
        self.candidate_graph_rows += int(stats.candidate_graph_rows)
        self.local_asterism_rows += int(stats.local_asterism_rows)
        self.context_pair_rows += int(stats.context_pair_rows)
        self.winner_rows += int(stats.winner_rows)
        self.winner_payload_rows += int(stats.winner_payload_rows)
        self.point_prediction_batches += int(stats.point_prediction_batches)
        self.point_prediction_rows += int(stats.point_prediction_rows)
        self.recovered_point_prediction_rows += int(
            stats.recovered_point_prediction_rows
        )
        self.recovered_winner_pixels += int(stats.recovered_winner_pixels)
        self.point_prediction_batch_rows_peak = max(
            self.point_prediction_batch_rows_peak,
            int(stats.point_prediction_batch_rows_peak),
        )
        self.point_prediction_backend_rows += int(stats.point_prediction_backend_rows)
        self.point_prediction_backend_batch_rows_peak = max(
            self.point_prediction_backend_batch_rows_peak,
            int(stats.point_prediction_backend_batch_rows_peak),
        )
        _merge_bucket_stats(
            self.point_backend_bucket_counts,
            dict(stats.point_prediction_backend_bucket_counts),
        )
        _merge_bucket_stats(
            self.point_backend_bucket_rows,
            dict(stats.point_prediction_backend_bucket_rows),
        )
        self.point_feature_bytes_peak = max(
            self.point_feature_bytes_peak,
            int(stats.point_feature_bytes_peak),
        )
        self.point_mps_current_bytes_peak = max(
            self.point_mps_current_bytes_peak,
            int(stats.point_mps_current_bytes_peak),
        )
        self.point_mps_driver_bytes_peak = max(
            self.point_mps_driver_bytes_peak,
            int(stats.point_mps_driver_bytes_peak),
        )
        self.point_mps_recommended_bytes = max(
            self.point_mps_recommended_bytes,
            int(stats.point_mps_recommended_bytes),
        )
        self.field_mean_prediction_batches += int(stats.field_mean_prediction_batches)
        self.field_mean_prediction_rows += int(stats.field_mean_prediction_rows)
        self.field_mean_prediction_batch_rows_peak = max(
            self.field_mean_prediction_batch_rows_peak,
            int(stats.field_mean_prediction_batch_rows_peak),
        )
        self.field_mean_prediction_backend_rows += int(
            stats.field_mean_prediction_backend_rows
        )
        self.field_mean_prediction_backend_batch_rows_peak = max(
            self.field_mean_prediction_backend_batch_rows_peak,
            int(stats.field_mean_prediction_backend_batch_rows_peak),
        )
        _merge_bucket_stats(
            self.field_mean_backend_bucket_counts,
            dict(stats.field_mean_prediction_backend_bucket_counts),
        )
        _merge_bucket_stats(
            self.field_mean_backend_bucket_rows,
            dict(stats.field_mean_prediction_backend_bucket_rows),
        )
        self.field_mean_feature_bytes_peak = max(
            self.field_mean_feature_bytes_peak,
            int(stats.field_mean_feature_bytes_peak),
        )
        self.field_mean_mps_current_bytes_peak = max(
            self.field_mean_mps_current_bytes_peak,
            int(stats.field_mean_mps_current_bytes_peak),
        )
        self.field_mean_mps_driver_bytes_peak = max(
            self.field_mean_mps_driver_bytes_peak,
            int(stats.field_mean_mps_driver_bytes_peak),
        )
        self.field_mean_mps_recommended_bytes = max(
            self.field_mean_mps_recommended_bytes,
            int(stats.field_mean_mps_recommended_bytes),
        )
        self.search_star_rows_peak = max(
            self.search_star_rows_peak,
            int(stats.search_star_rows_peak),
        )
        self.ngs_rows_peak = max(self.ngs_rows_peak, int(stats.ngs_rows_peak))
        self.close_pair_rows_peak = max(
            self.close_pair_rows_peak,
            int(stats.close_pair_rows_peak),
        )
        self.context_pair_rows_peak = max(
            self.context_pair_rows_peak,
            int(stats.context_pair_rows_peak),
        )
        self.raw_asterism_rows_peak = max(
            self.raw_asterism_rows_peak,
            int(stats.raw_asterism_rows_peak),
        )
        self.local_asterism_rows_peak = max(
            self.local_asterism_rows_peak,
            int(stats.local_asterism_rows_peak),
        )
        self.winner_payload_rows_peak = max(
            self.winner_payload_rows_peak,
            int(stats.winner_payload_rows_peak),
        )

    def to_stats(self) -> TraversalStructureStats:
        return TraversalStructureStats(
            search_star_rows=self.search_star_rows,
            ngs_rows=self.ngs_rows,
            close_pair_rows=self.close_pair_rows,
            self_pair_rows=self.self_pair_rows,
            raw_asterism_rows=self.raw_asterism_rows,
            dedupe_key_rows=self.dedupe_key_rows,
            post_bright_asterism_rows=self.post_bright_asterism_rows,
            candidate_graph_rows=self.candidate_graph_rows,
            local_asterism_rows=self.local_asterism_rows,
            context_pair_rows=self.context_pair_rows,
            winner_rows=self.winner_rows,
            winner_payload_rows=self.winner_payload_rows,
            point_prediction_batches=self.point_prediction_batches,
            point_prediction_rows=self.point_prediction_rows,
            recovered_point_prediction_rows=self.recovered_point_prediction_rows,
            recovered_winner_pixels=self.recovered_winner_pixels,
            point_prediction_batch_rows_peak=self.point_prediction_batch_rows_peak,
            point_prediction_backend_rows=self.point_prediction_backend_rows,
            point_prediction_backend_batch_rows_peak=(
                self.point_prediction_backend_batch_rows_peak
            ),
            point_prediction_backend_bucket_counts=tuple(
                sorted(self.point_backend_bucket_counts.items())
            ),
            point_prediction_backend_bucket_rows=tuple(
                sorted(self.point_backend_bucket_rows.items())
            ),
            point_feature_bytes_peak=self.point_feature_bytes_peak,
            point_mps_current_bytes_peak=self.point_mps_current_bytes_peak,
            point_mps_driver_bytes_peak=self.point_mps_driver_bytes_peak,
            point_mps_recommended_bytes=self.point_mps_recommended_bytes,
            field_mean_prediction_batches=self.field_mean_prediction_batches,
            field_mean_prediction_rows=self.field_mean_prediction_rows,
            field_mean_prediction_batch_rows_peak=(
                self.field_mean_prediction_batch_rows_peak
            ),
            field_mean_prediction_backend_rows=(
                self.field_mean_prediction_backend_rows
            ),
            field_mean_prediction_backend_batch_rows_peak=(
                self.field_mean_prediction_backend_batch_rows_peak
            ),
            field_mean_prediction_backend_bucket_counts=tuple(
                sorted(self.field_mean_backend_bucket_counts.items())
            ),
            field_mean_prediction_backend_bucket_rows=tuple(
                sorted(self.field_mean_backend_bucket_rows.items())
            ),
            field_mean_feature_bytes_peak=self.field_mean_feature_bytes_peak,
            field_mean_mps_current_bytes_peak=self.field_mean_mps_current_bytes_peak,
            field_mean_mps_driver_bytes_peak=self.field_mean_mps_driver_bytes_peak,
            field_mean_mps_recommended_bytes=self.field_mean_mps_recommended_bytes,
            search_star_rows_peak=self.search_star_rows_peak,
            ngs_rows_peak=self.ngs_rows_peak,
            close_pair_rows_peak=self.close_pair_rows_peak,
            context_pair_rows_peak=self.context_pair_rows_peak,
            raw_asterism_rows_peak=self.raw_asterism_rows_peak,
            local_asterism_rows_peak=self.local_asterism_rows_peak,
            winner_payload_rows_peak=self.winner_payload_rows_peak,
        )


class _TraversalDiagnosticsWriter:
    """Parent-owned CSV writer for opt-in per-pixel Traversal diagnostics."""

    FIELDNAMES = (
        "outer_pix",
        "worker_id",
        "success",
        "pixel_seconds",
        "artifact_write_seconds",
        "artifact_rss_before_convert_inner_mb",
        "artifact_peak_before_convert_inner_mb",
        "artifact_rss_after_convert_inner_mb",
        "artifact_peak_after_convert_inner_mb",
        "artifact_rss_after_convert_asterisms_mb",
        "artifact_peak_after_convert_asterisms_mb",
        "artifact_rss_after_hdf5_open_mb",
        "artifact_peak_after_hdf5_open_mb",
        "artifact_rss_after_hdf5_inner_mb",
        "artifact_peak_after_hdf5_inner_mb",
        "artifact_rss_after_hdf5_asterisms_mb",
        "artifact_peak_after_hdf5_asterisms_mb",
        "artifact_rss_after_hdf5_close_mb",
        "artifact_peak_after_hdf5_close_mb",
        "artifact_rss_after_replace_mb",
        "artifact_peak_after_replace_mb",
        "rss_start_mb",
        "peak_rss_start_mb",
        "rss_after_star_selection_mb",
        "peak_rss_after_star_selection_mb",
        "rss_after_candidate_generation_mb",
        "peak_rss_after_candidate_generation_mb",
        "rss_after_filtering_mb",
        "peak_rss_after_filtering_mb",
        "rss_after_context_mb",
        "peak_rss_after_context_mb",
        "rss_after_point_prediction_mb",
        "peak_rss_after_point_prediction_mb",
        "rss_after_field_mean_mb",
        "peak_rss_after_field_mean_mb",
        "rss_after_coverage_mb",
        "peak_rss_after_coverage_mb",
        "rss_after_dust_mb",
        "peak_rss_after_dust_mb",
        "rss_after_persisted_asterisms_mb",
        "peak_rss_after_persisted_asterisms_mb",
        "rss_after_artifact_write_mb",
        "peak_rss_after_artifact_write_mb",
        "rss_after_gc_mb",
        "peak_rss_after_gc_mb",
        "search_star_rows",
        "ngs_rows",
        "close_pair_rows",
        "raw_asterism_rows",
        "candidate_graph_rows",
        "local_asterism_rows",
        "context_pair_rows",
        "winner_payload_rows",
        "point_prediction_batches",
        "point_prediction_rows",
        "point_prediction_batch_rows_peak",
        "point_prediction_backend_rows",
        "point_prediction_backend_batch_rows_peak",
        "point_prediction_backend_bucket_counts",
        "point_prediction_backend_bucket_rows",
        "point_feature_mib_peak",
        "point_mps_current_mib_peak",
        "point_mps_driver_mib_peak",
        "point_mps_recommended_mib",
        "field_mean_prediction_batches",
        "field_mean_prediction_rows",
        "field_mean_prediction_batch_rows_peak",
        "field_mean_prediction_backend_rows",
        "field_mean_prediction_backend_batch_rows_peak",
        "field_mean_prediction_backend_bucket_counts",
        "field_mean_prediction_backend_bucket_rows",
        "field_mean_feature_mib_peak",
        "field_mean_mps_current_mib_peak",
        "field_mean_mps_driver_mib_peak",
        "field_mean_mps_recommended_mib",
        "artifact_inner_structured_mib",
        "artifact_asterism_structured_mib",
        "error_message",
    )
    WORKER_FIELDNAMES = (
        "worker_id",
        "event",
        "elapsed_seconds",
        "rss_mb",
        "peak_rss_mb",
    )

    def __init__(self, build_path: Path) -> None:
        diagnostics_dir = build_path / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.filename = diagnostics_dir / "traversal-memory.csv"
        self._handle = self.filename.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self.worker_filename = diagnostics_dir / "traversal-worker-memory.csv"
        self._worker_handle = self.worker_filename.open("w", newline="", encoding="utf-8")
        self._worker_writer = csv.DictWriter(
            self._worker_handle,
            fieldnames=self.WORKER_FIELDNAMES,
        )
        self._worker_writer.writeheader()

    def write(self, sample: TraversalMemorySample) -> None:
        self._writer.writerow(
            {
                "outer_pix": sample.outer_pix,
                "worker_id": sample.worker_id,
                "success": int(sample.success),
                "pixel_seconds": f"{sample.pixel_seconds:.6f}",
                "artifact_write_seconds": f"{sample.artifact_write_seconds:.6f}",
                "artifact_rss_before_convert_inner_mb": (
                    f"{sample.artifact_rss_before_convert_inner_mb:.3f}"
                ),
                "artifact_peak_before_convert_inner_mb": (
                    f"{sample.artifact_peak_before_convert_inner_mb:.3f}"
                ),
                "artifact_rss_after_convert_inner_mb": (
                    f"{sample.artifact_rss_after_convert_inner_mb:.3f}"
                ),
                "artifact_peak_after_convert_inner_mb": (
                    f"{sample.artifact_peak_after_convert_inner_mb:.3f}"
                ),
                "artifact_rss_after_convert_asterisms_mb": (
                    f"{sample.artifact_rss_after_convert_asterisms_mb:.3f}"
                ),
                "artifact_peak_after_convert_asterisms_mb": (
                    f"{sample.artifact_peak_after_convert_asterisms_mb:.3f}"
                ),
                "artifact_rss_after_hdf5_open_mb": (
                    f"{sample.artifact_rss_after_hdf5_open_mb:.3f}"
                ),
                "artifact_peak_after_hdf5_open_mb": (
                    f"{sample.artifact_peak_after_hdf5_open_mb:.3f}"
                ),
                "artifact_rss_after_hdf5_inner_mb": (
                    f"{sample.artifact_rss_after_hdf5_inner_mb:.3f}"
                ),
                "artifact_peak_after_hdf5_inner_mb": (
                    f"{sample.artifact_peak_after_hdf5_inner_mb:.3f}"
                ),
                "artifact_rss_after_hdf5_asterisms_mb": (
                    f"{sample.artifact_rss_after_hdf5_asterisms_mb:.3f}"
                ),
                "artifact_peak_after_hdf5_asterisms_mb": (
                    f"{sample.artifact_peak_after_hdf5_asterisms_mb:.3f}"
                ),
                "artifact_rss_after_hdf5_close_mb": (
                    f"{sample.artifact_rss_after_hdf5_close_mb:.3f}"
                ),
                "artifact_peak_after_hdf5_close_mb": (
                    f"{sample.artifact_peak_after_hdf5_close_mb:.3f}"
                ),
                "artifact_rss_after_replace_mb": (
                    f"{sample.artifact_rss_after_replace_mb:.3f}"
                ),
                "artifact_peak_after_replace_mb": (
                    f"{sample.artifact_peak_after_replace_mb:.3f}"
                ),
                "rss_start_mb": f"{sample.rss_start_mb:.3f}",
                "peak_rss_start_mb": f"{sample.peak_rss_start_mb:.3f}",
                "rss_after_star_selection_mb": f"{sample.rss_after_star_selection_mb:.3f}",
                "peak_rss_after_star_selection_mb": (
                    f"{sample.peak_rss_after_star_selection_mb:.3f}"
                ),
                "rss_after_candidate_generation_mb": (
                    f"{sample.rss_after_candidate_generation_mb:.3f}"
                ),
                "peak_rss_after_candidate_generation_mb": (
                    f"{sample.peak_rss_after_candidate_generation_mb:.3f}"
                ),
                "rss_after_filtering_mb": f"{sample.rss_after_filtering_mb:.3f}",
                "peak_rss_after_filtering_mb": f"{sample.peak_rss_after_filtering_mb:.3f}",
                "rss_after_context_mb": f"{sample.rss_after_context_mb:.3f}",
                "peak_rss_after_context_mb": f"{sample.peak_rss_after_context_mb:.3f}",
                "rss_after_point_prediction_mb": f"{sample.rss_after_point_prediction_mb:.3f}",
                "peak_rss_after_point_prediction_mb": (
                    f"{sample.peak_rss_after_point_prediction_mb:.3f}"
                ),
                "rss_after_field_mean_mb": f"{sample.rss_after_field_mean_mb:.3f}",
                "peak_rss_after_field_mean_mb": (
                    f"{sample.peak_rss_after_field_mean_mb:.3f}"
                ),
                "rss_after_coverage_mb": f"{sample.rss_after_coverage_mb:.3f}",
                "peak_rss_after_coverage_mb": f"{sample.peak_rss_after_coverage_mb:.3f}",
                "rss_after_dust_mb": f"{sample.rss_after_dust_mb:.3f}",
                "peak_rss_after_dust_mb": f"{sample.peak_rss_after_dust_mb:.3f}",
                "rss_after_persisted_asterisms_mb": (
                    f"{sample.rss_after_persisted_asterisms_mb:.3f}"
                ),
                "peak_rss_after_persisted_asterisms_mb": (
                    f"{sample.peak_rss_after_persisted_asterisms_mb:.3f}"
                ),
                "rss_after_artifact_write_mb": f"{sample.rss_after_artifact_write_mb:.3f}",
                "peak_rss_after_artifact_write_mb": (
                    f"{sample.peak_rss_after_artifact_write_mb:.3f}"
                ),
                "rss_after_gc_mb": f"{sample.rss_after_gc_mb:.3f}",
                "peak_rss_after_gc_mb": f"{sample.peak_rss_after_gc_mb:.3f}",
                "search_star_rows": sample.search_star_rows,
                "ngs_rows": sample.ngs_rows,
                "close_pair_rows": sample.close_pair_rows,
                "raw_asterism_rows": sample.raw_asterism_rows,
                "candidate_graph_rows": sample.candidate_graph_rows,
                "local_asterism_rows": sample.local_asterism_rows,
                "context_pair_rows": sample.context_pair_rows,
                "winner_payload_rows": sample.winner_payload_rows,
                "point_prediction_batches": sample.point_prediction_batches,
                "point_prediction_rows": sample.point_prediction_rows,
                "point_prediction_batch_rows_peak": (
                    sample.point_prediction_batch_rows_peak
                ),
                "point_prediction_backend_rows": sample.point_prediction_backend_rows,
                "point_prediction_backend_batch_rows_peak": (
                    sample.point_prediction_backend_batch_rows_peak
                ),
                "point_prediction_backend_bucket_counts": (
                    sample.point_prediction_backend_bucket_counts
                ),
                "point_prediction_backend_bucket_rows": (
                    sample.point_prediction_backend_bucket_rows
                ),
                "point_feature_mib_peak": f"{sample.point_feature_mib_peak:.3f}",
                "point_mps_current_mib_peak": (
                    f"{sample.point_mps_current_mib_peak:.3f}"
                ),
                "point_mps_driver_mib_peak": (
                    f"{sample.point_mps_driver_mib_peak:.3f}"
                ),
                "point_mps_recommended_mib": (
                    f"{sample.point_mps_recommended_mib:.3f}"
                ),
                "field_mean_prediction_batches": sample.field_mean_prediction_batches,
                "field_mean_prediction_rows": sample.field_mean_prediction_rows,
                "field_mean_prediction_batch_rows_peak": (
                    sample.field_mean_prediction_batch_rows_peak
                ),
                "field_mean_prediction_backend_rows": (
                    sample.field_mean_prediction_backend_rows
                ),
                "field_mean_prediction_backend_batch_rows_peak": (
                    sample.field_mean_prediction_backend_batch_rows_peak
                ),
                "field_mean_prediction_backend_bucket_counts": (
                    sample.field_mean_prediction_backend_bucket_counts
                ),
                "field_mean_prediction_backend_bucket_rows": (
                    sample.field_mean_prediction_backend_bucket_rows
                ),
                "field_mean_feature_mib_peak": (
                    f"{sample.field_mean_feature_mib_peak:.3f}"
                ),
                "field_mean_mps_current_mib_peak": (
                    f"{sample.field_mean_mps_current_mib_peak:.3f}"
                ),
                "field_mean_mps_driver_mib_peak": (
                    f"{sample.field_mean_mps_driver_mib_peak:.3f}"
                ),
                "field_mean_mps_recommended_mib": (
                    f"{sample.field_mean_mps_recommended_mib:.3f}"
                ),
                "artifact_inner_structured_mib": (
                    f"{sample.artifact_inner_structured_mib:.3f}"
                ),
                "artifact_asterism_structured_mib": (
                    f"{sample.artifact_asterism_structured_mib:.3f}"
                ),
                "error_message": sample.error_message,
            }
        )

    def write_worker(self, sample: TraversalWorkerMemorySample) -> None:
        self._worker_writer.writerow(
            {
                "worker_id": sample.worker_id,
                "event": sample.event,
                "elapsed_seconds": f"{sample.elapsed_seconds:.6f}",
                "rss_mb": f"{sample.rss_mb:.3f}",
                "peak_rss_mb": f"{sample.peak_rss_mb:.3f}",
            }
        )

    def flush(self) -> None:
        self._handle.flush()
        self._worker_handle.flush()

    def close(self) -> None:
        self._handle.close()
        self._worker_handle.close()

    def __enter__(self) -> "_TraversalDiagnosticsWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class _BufferedStateWriter:
    """Batch parent-owned state row writes while keeping in-memory state current."""

    def __init__(
        self,
        *,
        build_path: Path,
        state: np.ndarray,
        telemetry: _StateUpdateTelemetry,
        flush_interval: int = STATE_UPDATE_FLUSH_INTERVAL,
    ) -> None:
        self.build_path = build_path
        self.state = state
        self.telemetry = telemetry
        self.flush_interval = max(1, int(flush_interval))
        self._dirty_rows: set[int] = set()

    @property
    def dirty_count(self) -> int:
        return len(self._dirty_rows)

    def update(self, outer_pix: int, **updates: object) -> None:
        row_index = int(outer_pix)
        for key, value in updates.items():
            self._set_state_value(row_index, key, value)
        self._dirty_rows.add(row_index)
        self.telemetry.updates += 1
        self.telemetry.dirty_rows_peak = max(
            self.telemetry.dirty_rows_peak,
            len(self._dirty_rows),
        )
        if len(self._dirty_rows) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._dirty_rows:
            return

        row_indexes = np.asarray(sorted(self._dirty_rows), dtype=np.int64)
        started = time.perf_counter()
        retries = 0
        waited = 0.0
        while True:
            try:
                write_state_rows(self.build_path, row_indexes, self.state)
                elapsed = time.perf_counter() - started
                self.telemetry.flushes += 1
                self.telemetry.rows_written += int(len(row_indexes))
                self.telemetry.total_seconds += elapsed
                self.telemetry.max_seconds = max(self.telemetry.max_seconds, elapsed)
                self.telemetry.lock_retries += retries
                self.telemetry.lock_wait_seconds += waited
                self._dirty_rows.difference_update(int(row) for row in row_indexes)
                return
            except Exception as exc:
                if not _is_hdf5_lock_error(exc) or retries >= STATE_UPDATE_RETRY_ATTEMPTS:
                    raise
                retries += 1
                wait_seconds = STATE_UPDATE_RETRY_DELAY_SECONDS * retries
                waited += wait_seconds
                time.sleep(wait_seconds)

    def _set_state_value(self, row_index: int, key: str, value: object) -> None:
        if key not in self.state.dtype.names:
            raise BuildError(f"Unknown state field {key!r}")
        field_dtype = self.state.dtype[key]
        if field_dtype.kind == "S":
            self.state[key][row_index] = _encode_state_bytes(
                value,
                max_bytes=field_dtype.itemsize,
                field_name=key,
            )
        else:
            self.state[key][row_index] = value


def _is_hdf5_lock_error(exc: BaseException) -> bool:
    if isinstance(exc, BlockingIOError):
        return True
    message = str(exc).lower()
    return "unable to lock file" in message or "resource temporarily unavailable" in message


def _encode_state_bytes(
    value: object,
    *,
    max_bytes: int,
    field_name: str,
) -> bytes:
    if value is None:
        return b""
    encoded = str(value).encode("utf-8")
    if len(encoded) > max_bytes:
        raise BuildError(
            f"{field_name} exceeds persisted limit of {max_bytes} bytes"
        )
    return encoded


def _update_state_row_with_telemetry(
    build_path: Path,
    outer_pix: int,
    telemetry: _StateUpdateTelemetry | None,
    **updates: object,
) -> None:
    """Update one state row and record retry/timing counters for Traversal telemetry."""

    started = time.perf_counter()
    retries = 0
    waited = 0.0
    while True:
        try:
            update_state_row(build_path, outer_pix, **updates)
            elapsed = time.perf_counter() - started
            if telemetry is not None:
                telemetry.updates += 1
                telemetry.flushes += 1
                telemetry.rows_written += 1
                telemetry.total_seconds += elapsed
                telemetry.max_seconds = max(telemetry.max_seconds, elapsed)
                telemetry.lock_retries += retries
                telemetry.lock_wait_seconds += waited
                telemetry.dirty_rows_peak = max(telemetry.dirty_rows_peak, 1)
            return
        except Exception as exc:
            if not _is_hdf5_lock_error(exc) or retries >= STATE_UPDATE_RETRY_ATTEMPTS:
                raise
            retries += 1
            wait_seconds = STATE_UPDATE_RETRY_DELAY_SECONDS * retries
            waited += wait_seconds
            time.sleep(wait_seconds)


def _append_state_update_telemetry(
    build_path: Path,
    telemetry: _StateUpdateTelemetry,
) -> None:
    append_build_log(
        build_path,
        "phase=traversal "
        "state_update_stats "
        f"updates={telemetry.updates} "
        f"flushes={telemetry.flushes} "
        f"rows_written={telemetry.rows_written} "
        f"total_s={telemetry.total_seconds:.3f} "
        f"max_s={telemetry.max_seconds:.3f} "
        f"lock_retries={telemetry.lock_retries} "
        f"lock_wait_s={telemetry.lock_wait_seconds:.3f} "
        f"dirty_rows_peak={telemetry.dirty_rows_peak}",
    )


def _materialize_outer_pixel_products(
    context: TraversalTaskContext,
    outer_pix: int,
    *,
    execution_config: TraversalExecutionConfig | None = None,
    runtime=None,
    store=None,
    geometry: TraversalGeometry | None = None,
    memory_profile: TraversalMemoryProfile | None = None,
    artifact_memory_profile: ArtifactMemoryProfile | None = None,
    memory_pressure_callback=None,
) -> tuple[ArtifactWriteProfile, TraversalStageStats, TraversalStructureStats]:
    """Run the Traversal pipeline for one outer pixel without mutating build state."""

    execution_config = execution_config or TraversalExecutionConfig()
    filename, inner, asterisms, stage_stats, structure_stats = _build_outer_pixel_products(
        context,
        outer_pix,
        execution_config=execution_config,
        runtime=runtime,
        store=store,
        geometry=geometry,
        memory_profile=memory_profile,
        memory_pressure_callback=memory_pressure_callback,
    )
    write_profile = _write_outer_pixel_products(
        filename,
        inner=inner,
        asterisms=asterisms,
        measure_input_bytes=memory_profile is not None,
        memory_profile=artifact_memory_profile,
    )
    if memory_profile is not None:
        memory_profile.rss_after_artifact_write_mb = _current_rss_mb()
        memory_profile.peak_rss_after_artifact_write_mb = _peak_rss_mb()
    return write_profile, stage_stats, structure_stats


def _build_outer_pixel_products(
    context: TraversalTaskContext,
    outer_pix: int,
    *,
    execution_config: TraversalExecutionConfig | None = None,
    runtime=None,
    store=None,
    geometry: TraversalGeometry | None = None,
    memory_profile: TraversalMemoryProfile | None = None,
    memory_pressure_callback=None,
) -> tuple[Path, Table, Table, TraversalStageStats, TraversalStructureStats]:
    """Build Traversal products for one outer pixel without writing artifacts."""

    execution_config = execution_config or TraversalExecutionConfig()
    if runtime is None:
        runtime = load_runtime_config(
            context.runtime_config_path,
            model_root=context.roots.model_root,
        )
        warm_model_cache(
            runtime,
            prediction_device=execution_config.prediction_device,
            averaged_prediction_device=execution_config.averaged_prediction_device,
        )
    if store is None:
        base_store = GaiaHealpixStore(
            GaiaStoreConfig(
                root=context.roots.gaia_root,
                release=context.definition.gaia_release,
                healpix_level=context.definition.outer_level,
            )
        )
        store = RuntimeGaiaHealpixStore(
            base_store,
            runtime,
            max_entries=0,
            max_bytes=0,
        )
    if geometry is None:
        geometry = TraversalGeometry.from_runtime(runtime)
    profile = TraversalStageProfile()
    structure_profile = TraversalStructureProfile()
    asterisms, inner = build_traversal_products(
        store,
        runtime,
        outer_pix,
        execution_config=execution_config,
        dust_root=context.roots.dust_root,
        max_data_level=context.definition.max_data_level,
        geometry=geometry,
        profile=profile,
        structure_profile=structure_profile,
        memory_profile=memory_profile,
        rss_sampler=_current_rss_mb if memory_profile is not None else None,
        peak_sampler=_peak_rss_mb if memory_profile is not None else None,
        memory_pressure_callback=memory_pressure_callback,
    )
    filename = outer_artifact_filename(context.build_path, context.definition, outer_pix)
    structure_stats = (
        structure_profile.to_stats()
        if structure_profile is not None
        else TraversalStructureStats()
    )
    return filename, inner, asterisms, profile.to_stats(), structure_stats


def _write_outer_pixel_products(
    filename: Path,
    *,
    inner: Table,
    asterisms: Table,
    measure_input_bytes: bool = False,
    memory_profile: ArtifactMemoryProfile | None = None,
) -> ArtifactWriteProfile:
    """Write one outer-pixel artifact and return artifact write profile."""

    return write_outer_artifact_profiled(
        filename,
        inner=inner,
        asterisms=asterisms,
        measure_input_bytes=measure_input_bytes,
        memory_profile=memory_profile,
        rss_sampler=_current_rss_mb if memory_profile is not None else None,
        peak_sampler=_peak_rss_mb if memory_profile is not None else None,
    )


def _run_outer_pixel_traversal_task(
    context: TraversalTaskContext,
    outer_pix: int,
    *,
    execution_config: TraversalExecutionConfig | None = None,
) -> TraversalTaskResult:
    """Run one worker-owned outer-pixel Traversal task and serialize the outcome."""

    try:
        configure_inference_threads(1)
        _materialize_outer_pixel_products(
            context,
            outer_pix,
            execution_config=execution_config,
        )
    except Exception as exc:
        return TraversalTaskResult(
            outer_pix=int(outer_pix),
            success=False,
            error_message=str(exc),
        )
    return TraversalTaskResult(outer_pix=int(outer_pix), success=True)


def _mark_outer_pixel_running(
    build_path: Path,
    state: np.ndarray,
    outer_pix: int,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
) -> None:
    current_state = state[int(outer_pix)]
    updates = dict(
        traversal_status=WORK_STATUS_RUNNING,
        traversal_attempt_count=int(current_state["traversal_attempt_count"]) + 1,
        traversal_last_error_message="",
    )
    if state_writer is not None:
        state_writer.update(outer_pix, **updates)
    else:
        _update_state_row_with_telemetry(
            build_path,
            outer_pix,
            telemetry,
            **updates,
        )
        state["traversal_status"][int(outer_pix)] = WORK_STATUS_RUNNING
        state["traversal_attempt_count"][int(outer_pix)] = (
            int(current_state["traversal_attempt_count"]) + 1
        )
        state["traversal_last_error_message"][int(outer_pix)] = b""


def _record_traversal_result(
    build_path: Path,
    state: np.ndarray,
    result: TraversalTaskResult,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
) -> None:
    if result.success:
        updates = dict(
            traversal_status=WORK_STATUS_DONE,
            traversal_last_error_message="",
        )
        if state_writer is not None:
            state_writer.update(result.outer_pix, **updates)
        else:
            _update_state_row_with_telemetry(
                build_path,
                result.outer_pix,
                telemetry,
                **updates,
            )
            state["traversal_status"][int(result.outer_pix)] = WORK_STATUS_DONE
            state["traversal_last_error_message"][int(result.outer_pix)] = b""
        return

    error_message = _truncate_state_error_message(result.error_message)
    updates = dict(
        traversal_status=WORK_STATUS_FAILED,
        traversal_last_error_message=error_message,
    )
    if state_writer is not None:
        state_writer.update(result.outer_pix, **updates)
    else:
        _update_state_row_with_telemetry(
            build_path,
            result.outer_pix,
            telemetry,
            **updates,
        )
        state["traversal_status"][int(result.outer_pix)] = WORK_STATUS_FAILED
        state["traversal_last_error_message"][int(result.outer_pix)] = error_message.encode(
            "utf-8"
        )


def _truncate_state_error_message(message: object) -> str:
    """Return a UTF-8-safe error message that fits the persisted state field."""

    text = str(message)
    encoded = text.encode("utf-8")
    if len(encoded) <= STATE_ERROR_MAX_BYTES:
        return text

    suffix = TRUNCATED_ERROR_SUFFIX.encode("utf-8")
    limit = max(0, STATE_ERROR_MAX_BYTES - len(suffix))
    truncated = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated + TRUNCATED_ERROR_SUFFIX


def _create_regional_process_context():
    """Return the process context used by cache-aware regional Traversal."""

    return multiprocessing.get_context("spawn")


def _repair_stale_running_rows(
    build_path: Path,
    state: np.ndarray,
    *,
    status_field: str,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
) -> int:
    stale = np.flatnonzero(state[status_field] == WORK_STATUS_RUNNING)
    for outer_pix in stale:
        updates = {status_field: WORK_STATUS_PENDING}
        if state_writer is not None:
            state_writer.update(int(outer_pix), **updates)
        else:
            _update_state_row_with_telemetry(
                build_path,
                int(outer_pix),
                telemetry,
                **updates,
            )
            state[status_field][int(outer_pix)] = WORK_STATUS_PENDING
    return int(len(stale))


def _reset_running_rows(
    build_path: Path,
    state: np.ndarray,
    *,
    status_field: str,
    error_field: str,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
) -> int:
    running = np.flatnonzero(state[status_field] == WORK_STATUS_RUNNING)
    for outer_pix in running:
        updates = {
            status_field: WORK_STATUS_PENDING,
            error_field: "",
        }
        if state_writer is not None:
            state_writer.update(int(outer_pix), **updates)
        else:
            _update_state_row_with_telemetry(
                build_path,
                int(outer_pix),
                telemetry,
                **updates,
            )
            state[status_field][int(outer_pix)] = WORK_STATUS_PENDING
            state[error_field][int(outer_pix)] = b""
    return int(len(running))


def _clear_torch_mps_cache() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        return


def _trim_worker_runtime_memory(store: object) -> None:
    if isinstance(store, RuntimeGaiaHealpixStore):
        store.trim_cache(max_entries=0, max_bytes=0)
    _clear_torch_mps_cache()
    gc.collect()


def _drain_worker_control_command(control_queue) -> str | None:
    if control_queue is None:
        return None
    command = None
    while True:
        try:
            command = control_queue.get_nowait()
        except Empty:
            return command


def _apply_worker_memory_pressure_command(
    *,
    control_queue,
    store: object,
    worker_id: int,
    run_started: float,
    detailed: bool,
    allow_pause: bool,
) -> Iterator[TraversalWorkerMessage]:
    command = _drain_worker_control_command(control_queue)
    if command is None or command == PARENT_MEMORY_PRESSURE_RESUME:
        return
    if command not in (
        PARENT_MEMORY_PRESSURE_TRIM,
        PARENT_MEMORY_PRESSURE_PAUSE,
    ):
        return

    _trim_worker_runtime_memory(store)
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event=f"after_memory_pressure_{command}",
            run_started=run_started,
        )
    if command != PARENT_MEMORY_PRESSURE_PAUSE or not allow_pause:
        return

    pause_started = time.perf_counter()
    while (
        time.perf_counter() - pause_started
        < WORKER_MEMORY_PRESSURE_MAX_PAUSE_SECONDS
    ):
        time.sleep(WORKER_MEMORY_PRESSURE_SLEEP_SECONDS)
        next_command = _drain_worker_control_command(control_queue)
        if next_command is None or next_command == PARENT_MEMORY_PRESSURE_PAUSE:
            continue
        if next_command == PARENT_MEMORY_PRESSURE_RESUME:
            if detailed:
                yield _worker_memory_sample(
                    worker_id=worker_id,
                    event="after_memory_pressure_resume",
                    run_started=run_started,
                )
            return
        if next_command == PARENT_MEMORY_PRESSURE_TRIM:
            _trim_worker_runtime_memory(store)
            if detailed:
                yield _worker_memory_sample(
                    worker_id=worker_id,
                    event="after_memory_pressure_trim_while_paused",
                    run_started=run_started,
                )
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_memory_pressure_pause_timeout",
            run_started=run_started,
        )


def _run_regional_traversal_worker(
    context: TraversalTaskContext,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
    result_queue,
    control_queue=None,
) -> None:
    """Worker entrypoint for one region-owned Traversal plan."""

    for message in _iter_long_lived_traversal_worker_messages(
        context,
        plan,
        execution_config,
        control_queue=control_queue,
    ):
        result_queue.put(message)


def _run_dynamic_traversal_worker(
    context: TraversalTaskContext,
    worker_id: int,
    execution_config: TraversalExecutionConfig,
    result_queue,
    work_queue,
    control_queue=None,
) -> None:
    """Worker entrypoint for parent-dispatched dynamic Traversal batches."""

    for message in _iter_dynamic_traversal_worker_messages(
        context,
        worker_id,
        execution_config,
        work_queue=work_queue,
        control_queue=control_queue,
    ):
        result_queue.put(message)


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return float(usage.ru_maxrss) / (1024.0 * 1024.0)
    return float(usage.ru_maxrss) / 1024.0


def _current_rss_mb() -> float:
    """Return current worker RSS when available; fall back to peak RSS."""

    if sys.platform == "darwin":
        try:
            return _darwin_current_rss_mb()
        except Exception:
            return _peak_rss_mb()
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            rss_pages = int(statm.read_text(encoding="utf-8").split()[1])
            page_size = resource.getpagesize()
            return rss_pages * page_size / (1024.0 * 1024.0)
        except Exception:
            return _peak_rss_mb()
    return _peak_rss_mb()


def _process_current_rss_mb(pid: int) -> float:
    """Return current RSS for one process id when available."""

    if int(pid) <= 0:
        return 0.0
    statm = Path(f"/proc/{int(pid)}/statm")
    if statm.is_file():
        try:
            rss_pages = int(statm.read_text(encoding="utf-8").split()[1])
            page_size = resource.getpagesize()
            return rss_pages * page_size / (1024.0 * 1024.0)
        except Exception:
            return 0.0

    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(int(pid))],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 0.0
    if completed.returncode != 0:
        return 0.0
    text = completed.stdout.strip()
    if not text:
        return 0.0
    try:
        return float(text.splitlines()[0].strip()) / 1024.0
    except ValueError:
        return 0.0


def _parent_total_current_rss_mb(processes: tuple | list = ()) -> float:
    total = _current_rss_mb()
    for process in processes:
        pid = getattr(process, "pid", None)
        if pid is None:
            continue
        total += _process_current_rss_mb(int(pid))
    return total


def _gpu_prediction_enabled(
    execution_config: TraversalExecutionConfig | None = None,
) -> bool:
    if not predict_backend.mps_is_available():
        return False
    execution_config = execution_config or TraversalExecutionConfig()
    return any(
        value == "gpu"
        for value in (
            execution_config.prediction_device,
            execution_config.averaged_prediction_device,
        )
    )


def _parent_gpu_driver_reserve_mb(
    worker_count: int,
    execution_config: TraversalExecutionConfig | None = None,
) -> float:
    if int(worker_count) <= 0 or not _gpu_prediction_enabled(execution_config):
        return 0.0
    execution_config = execution_config or TraversalExecutionConfig()
    return max(
        0.0,
        float(execution_config.parent_gpu_driver_reserve_mb),
    ) * int(worker_count)


def _active_process_count(processes: tuple | list) -> int:
    count = 0
    for process in processes:
        exitcode = getattr(process, "exitcode", None)
        if exitcode is None:
            count += 1
    return count


def _parent_total_current_ram_mb(
    processes: tuple | list = (),
    *,
    worker_count: int | None = None,
    execution_config: TraversalExecutionConfig | None = None,
) -> float:
    reserve_workers = (
        max(1, int(worker_count))
        if worker_count is not None
        else _active_process_count(processes)
    )
    return _parent_total_current_rss_mb(processes) + _parent_gpu_driver_reserve_mb(
        reserve_workers,
        execution_config,
    )


def _raise_if_parent_memory_limit_exceeded(
    execution_config: TraversalExecutionConfig,
    processes: tuple | list = (),
) -> None:
    limit = int(execution_config.parent_memory_limit_mb)
    if limit <= 0:
        return
    total_mb = _parent_total_current_ram_mb(
        processes,
        worker_count=None if processes else 1,
        execution_config=execution_config,
    )
    if total_mb > float(limit):
        raise BuildError(
            f"parent memory limit exceeded: current total RAM {total_mb:.1f} MiB "
            f"> {limit} MiB"
        )


def _classify_parent_memory_pressure(
    *,
    total_mb: float,
    limit_mb: int,
    previous_state: str,
) -> str:
    """Classify total-RAM pressure with hysteresis around soft and hard bands."""

    if int(limit_mb) <= 0:
        return PARENT_MEMORY_PRESSURE_NORMAL
    hard_mb = float(limit_mb) * PARENT_MEMORY_HARD_FRACTION
    soft_mb = float(limit_mb) * PARENT_MEMORY_SOFT_FRACTION
    hard_release_mb = float(limit_mb) * PARENT_MEMORY_HARD_RELEASE_FRACTION
    soft_release_mb = float(limit_mb) * PARENT_MEMORY_SOFT_RELEASE_FRACTION

    if total_mb >= hard_mb:
        return PARENT_MEMORY_PRESSURE_PAUSE
    if (
        previous_state == PARENT_MEMORY_PRESSURE_PAUSE
        and total_mb >= hard_release_mb
    ):
        return PARENT_MEMORY_PRESSURE_PAUSE
    if total_mb >= soft_mb:
        return PARENT_MEMORY_PRESSURE_TRIM
    if (
        previous_state
        in (PARENT_MEMORY_PRESSURE_TRIM, PARENT_MEMORY_PRESSURE_PAUSE)
        and total_mb >= soft_release_mb
    ):
        return PARENT_MEMORY_PRESSURE_TRIM
    return PARENT_MEMORY_PRESSURE_NORMAL


def _select_parent_memory_pressure_targets(
    *,
    pressure_state: str,
    worker_rss_mb: dict[int, float],
    active_worker_ids: set[int],
) -> tuple[int, ...]:
    """Return heaviest active workers to target under memory pressure."""

    if pressure_state == PARENT_MEMORY_PRESSURE_NORMAL or not active_worker_ids:
        return ()
    ranked = sorted(
        (int(worker_id) for worker_id in active_worker_ids),
        key=lambda worker_id: worker_rss_mb.get(worker_id, 0.0),
        reverse=True,
    )
    if pressure_state == PARENT_MEMORY_PRESSURE_TRIM or len(ranked) <= 1:
        return tuple(ranked[:1])
    target_count = max(1, (len(ranked) + 1) // 2)
    return tuple(ranked[:target_count])


def _worker_process_rss_mb(
    process_by_worker_id: dict[int, object],
) -> dict[int, float]:
    rss_by_worker: dict[int, float] = {}
    for worker_id, process in process_by_worker_id.items():
        pid = getattr(process, "pid", None)
        if pid is None:
            rss_by_worker[int(worker_id)] = 0.0
            continue
        rss_by_worker[int(worker_id)] = _process_current_rss_mb(int(pid))
    return rss_by_worker


def _send_parent_memory_pressure_commands(
    *,
    build_path: Path,
    control_queues: dict[int, object],
    active_worker_ids: set[int],
    worker_rss_mb: dict[int, float],
    worker_pressure_states: dict[int, str],
    worker_pressure_command_times: dict[int, float],
    pressure_state: str,
    targets: tuple[int, ...],
    now: float,
) -> None:
    target_set = set(targets)
    for worker_id in sorted(active_worker_ids):
        if worker_id in target_set:
            desired_state = pressure_state
            if (
                desired_state == PARENT_MEMORY_PRESSURE_PAUSE
                and len(active_worker_ids) <= 1
            ):
                desired_state = PARENT_MEMORY_PRESSURE_TRIM
        else:
            desired_state = PARENT_MEMORY_PRESSURE_NORMAL

        current_state = worker_pressure_states.get(
            worker_id,
            PARENT_MEMORY_PRESSURE_NORMAL,
        )
        last_command = worker_pressure_command_times.get(worker_id, 0.0)
        should_send = desired_state != current_state
        if (
            desired_state != PARENT_MEMORY_PRESSURE_NORMAL
            and now - last_command >= PARENT_MEMORY_PRESSURE_COMMAND_INTERVAL_SECONDS
        ):
            should_send = True
        if not should_send:
            continue

        command = (
            PARENT_MEMORY_PRESSURE_RESUME
            if desired_state == PARENT_MEMORY_PRESSURE_NORMAL
            else desired_state
        )
        control_queue = control_queues.get(worker_id)
        if control_queue is None:
            continue
        control_queue.put(command)
        worker_pressure_states[worker_id] = desired_state
        worker_pressure_command_times[worker_id] = now
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={worker_id} "
            f"memory_pressure_command={command} "
            f"worker_rss_mb={worker_rss_mb.get(worker_id, 0.0):.1f}",
        )


def _update_parent_memory_pressure(
    *,
    build_path: Path,
    execution_config: TraversalExecutionConfig,
    process_by_worker_id: dict[int, object],
    control_queues: dict[int, object],
    active_worker_ids: set[int],
    worker_pressure_states: dict[int, str],
    worker_pressure_command_times: dict[int, float],
    previous_state: str,
    previous_targets: tuple[int, ...],
    force_snapshot: bool = False,
) -> tuple[str, tuple[int, ...]]:
    """Monitor total RAM and ask the heaviest workers to back off when needed."""

    limit = int(execution_config.parent_memory_limit_mb)
    if limit <= 0:
        return PARENT_MEMORY_PRESSURE_NORMAL, ()

    worker_rss_mb = _worker_process_rss_mb(process_by_worker_id)
    active_worker_rss_mb = {
        worker_id: worker_rss_mb.get(worker_id, 0.0)
        for worker_id in active_worker_ids
    }
    gpu_reserve_mb = _parent_gpu_driver_reserve_mb(
        len(active_worker_ids),
        execution_config,
    )
    total_mb = _current_rss_mb() + sum(active_worker_rss_mb.values()) + gpu_reserve_mb
    pressure_state = _classify_parent_memory_pressure(
        total_mb=total_mb,
        limit_mb=limit,
        previous_state=previous_state,
    )
    targets = _select_parent_memory_pressure_targets(
        pressure_state=pressure_state,
        worker_rss_mb=active_worker_rss_mb,
        active_worker_ids=active_worker_ids,
    )
    now = time.perf_counter()
    _send_parent_memory_pressure_commands(
        build_path=build_path,
        control_queues=control_queues,
        active_worker_ids=active_worker_ids,
        worker_rss_mb=active_worker_rss_mb,
        worker_pressure_states=worker_pressure_states,
        worker_pressure_command_times=worker_pressure_command_times,
        pressure_state=pressure_state,
        targets=targets,
        now=now,
    )
    pressure_changed = pressure_state != previous_state or targets != previous_targets
    if pressure_changed or force_snapshot:
        target_text = ",".join(str(worker_id) for worker_id in targets) or "none"
        rss_text = ",".join(
            f"{worker_id}:{active_worker_rss_mb[worker_id]:.1f}"
            for worker_id in sorted(active_worker_rss_mb)
        )
        event = "memory_pressure" if pressure_changed else "memory_snapshot"
        append_build_log(
            build_path,
            "phase=traversal "
            f"{event} state={pressure_state} "
            f"total_mb={total_mb:.1f} "
            f"gpu_reserve_mb={gpu_reserve_mb:.1f} "
            f"limit_mb={limit} "
            f"trim_fraction={PARENT_MEMORY_SOFT_FRACTION:.2f} "
            f"trim_release_fraction={PARENT_MEMORY_SOFT_RELEASE_FRACTION:.2f} "
            f"pause_fraction={PARENT_MEMORY_HARD_FRACTION:.2f} "
            f"pause_release_fraction={PARENT_MEMORY_HARD_RELEASE_FRACTION:.2f} "
            f"targets={target_text} "
            f"worker_rss_mb={rss_text}",
        )
    return pressure_state, targets


def _darwin_current_rss_mb() -> float:
    """Return current RSS on macOS using Mach task_info."""

    import ctypes

    class TimeValue(ctypes.Structure):
        _fields_ = [("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32)]

    class MachTaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", TimeValue),
            ("system_time", TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    task = ctypes.c_uint32.in_dll(libc, "mach_task_self_").value
    info = MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    result = libc.task_info(
        ctypes.c_uint32(task),
        ctypes.c_int(20),  # MACH_TASK_BASIC_INFO
        ctypes.byref(info),
        ctypes.byref(count),
    )
    if result != 0:
        raise OSError(f"task_info failed with status {result}")
    return float(info.resident_size) / (1024.0 * 1024.0)


def _traversal_cache_stats(store: RuntimeGaiaHealpixStore) -> TraversalCacheStats:
    stats = store.stats()
    return TraversalCacheStats(
        hits=stats.hits,
        misses=stats.misses,
        evictions=stats.evictions,
        oversized_skips=stats.oversized_skips,
        current_bytes=stats.current_bytes,
        peak_bytes=stats.peak_bytes,
        entries=stats.entries,
        load_seconds=stats.load_seconds,
        raw_load_seconds=stats.raw_load_seconds,
        prepare_seconds=stats.prepare_seconds,
    )


def _traversal_worker_stats_message(
    *,
    worker_id: int,
    completed: int,
    failed: int,
    run_started: float,
    pixel_seconds: float,
    artifact_write_telemetry: _ArtifactWriteTelemetry,
    stage_telemetry: _TraversalStageTelemetry,
    structure_telemetry: _TraversalStructureTelemetry,
    store: RuntimeGaiaHealpixStore,
) -> TraversalWorkerMessage:
    return TraversalWorkerMessage(
        worker_id=worker_id,
        kind="profile",
        worker_stats=TraversalWorkerStats(
            completed=completed,
            failed=failed,
            elapsed_seconds=time.perf_counter() - run_started,
            pixel_seconds=pixel_seconds,
            artifact_write_seconds=artifact_write_telemetry.total_seconds,
            artifact_convert_inner_seconds=artifact_write_telemetry.convert_inner_seconds,
            artifact_convert_asterisms_seconds=(
                artifact_write_telemetry.convert_asterisms_seconds
            ),
            artifact_hdf5_open_seconds=artifact_write_telemetry.hdf5_open_seconds,
            artifact_hdf5_inner_seconds=artifact_write_telemetry.hdf5_inner_seconds,
            artifact_hdf5_asterisms_seconds=(
                artifact_write_telemetry.hdf5_asterisms_seconds
            ),
            artifact_hdf5_close_seconds=artifact_write_telemetry.hdf5_close_seconds,
            artifact_replace_seconds=artifact_write_telemetry.replace_seconds,
            artifact_bytes=artifact_write_telemetry.output_bytes,
            artifact_inner_input_bytes=artifact_write_telemetry.inner_input_bytes,
            artifact_asterism_input_bytes=artifact_write_telemetry.asterism_input_bytes,
            artifact_inner_structured_bytes=artifact_write_telemetry.inner_structured_bytes,
            artifact_asterism_structured_bytes=(
                artifact_write_telemetry.asterism_structured_bytes
            ),
            artifact_inner_rows=artifact_write_telemetry.inner_rows,
            artifact_asterism_rows=artifact_write_telemetry.asterism_rows,
            stage_stats=stage_telemetry.to_stats(),
            structure_stats=structure_telemetry.to_stats(),
            peak_rss_mb=_peak_rss_mb(),
            cache_stats=_traversal_cache_stats(store),
        ),
    )


def _build_memory_sample(
    *,
    outer_pix: int,
    worker_id: int,
    success: bool,
    pixel_seconds: float,
    artifact_write_profile: ArtifactWriteProfile | None,
    artifact_memory_profile: ArtifactMemoryProfile | None,
    structure_stats: TraversalStructureStats | None,
    memory_profile: TraversalMemoryProfile,
    error_message: str = "",
) -> TraversalMemorySample:
    artifact_write_seconds = 0.0
    artifact_inner_structured_mib = 0.0
    artifact_asterism_structured_mib = 0.0
    if artifact_write_profile is not None:
        artifact_write_seconds = artifact_write_profile.total_seconds
        artifact_inner_structured_mib = artifact_write_profile.inner_structured_bytes / (
            1024.0 * 1024.0
        )
        artifact_asterism_structured_mib = (
            artifact_write_profile.asterism_structured_bytes / (1024.0 * 1024.0)
        )
    if structure_stats is None:
        structure_stats = TraversalStructureStats()
    return TraversalMemorySample(
        outer_pix=int(outer_pix),
        worker_id=int(worker_id),
        success=success,
        pixel_seconds=float(pixel_seconds),
        artifact_write_seconds=float(artifact_write_seconds),
        artifact_rss_before_convert_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_before_convert_inner_mb
        ),
        artifact_peak_before_convert_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_before_convert_inner_mb
        ),
        artifact_rss_after_convert_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_convert_inner_mb
        ),
        artifact_peak_after_convert_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_convert_inner_mb
        ),
        artifact_rss_after_convert_asterisms_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_convert_asterisms_mb
        ),
        artifact_peak_after_convert_asterisms_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_convert_asterisms_mb
        ),
        artifact_rss_after_hdf5_open_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_hdf5_open_mb
        ),
        artifact_peak_after_hdf5_open_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_hdf5_open_mb
        ),
        artifact_rss_after_hdf5_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_hdf5_inner_mb
        ),
        artifact_peak_after_hdf5_inner_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_hdf5_inner_mb
        ),
        artifact_rss_after_hdf5_asterisms_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_hdf5_asterisms_mb
        ),
        artifact_peak_after_hdf5_asterisms_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_hdf5_asterisms_mb
        ),
        artifact_rss_after_hdf5_close_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_hdf5_close_mb
        ),
        artifact_peak_after_hdf5_close_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_hdf5_close_mb
        ),
        artifact_rss_after_replace_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.rss_after_replace_mb
        ),
        artifact_peak_after_replace_mb=(
            0.0
            if artifact_memory_profile is None
            else artifact_memory_profile.peak_after_replace_mb
        ),
        rss_start_mb=memory_profile.rss_start_mb,
        peak_rss_start_mb=memory_profile.peak_rss_start_mb,
        rss_after_star_selection_mb=memory_profile.rss_after_star_selection_mb,
        peak_rss_after_star_selection_mb=memory_profile.peak_rss_after_star_selection_mb,
        rss_after_candidate_generation_mb=memory_profile.rss_after_candidate_generation_mb,
        peak_rss_after_candidate_generation_mb=memory_profile.peak_rss_after_candidate_generation_mb,
        rss_after_filtering_mb=memory_profile.rss_after_filtering_mb,
        peak_rss_after_filtering_mb=memory_profile.peak_rss_after_filtering_mb,
        rss_after_context_mb=memory_profile.rss_after_context_mb,
        peak_rss_after_context_mb=memory_profile.peak_rss_after_context_mb,
        rss_after_point_prediction_mb=memory_profile.rss_after_point_prediction_mb,
        peak_rss_after_point_prediction_mb=memory_profile.peak_rss_after_point_prediction_mb,
        rss_after_field_mean_mb=memory_profile.rss_after_field_mean_mb,
        peak_rss_after_field_mean_mb=memory_profile.peak_rss_after_field_mean_mb,
        rss_after_coverage_mb=memory_profile.rss_after_coverage_mb,
        peak_rss_after_coverage_mb=memory_profile.peak_rss_after_coverage_mb,
        rss_after_dust_mb=memory_profile.rss_after_dust_mb,
        peak_rss_after_dust_mb=memory_profile.peak_rss_after_dust_mb,
        rss_after_persisted_asterisms_mb=memory_profile.rss_after_persisted_asterisms_mb,
        peak_rss_after_persisted_asterisms_mb=(
            memory_profile.peak_rss_after_persisted_asterisms_mb
        ),
        rss_after_artifact_write_mb=memory_profile.rss_after_artifact_write_mb,
        peak_rss_after_artifact_write_mb=memory_profile.peak_rss_after_artifact_write_mb,
        rss_after_gc_mb=memory_profile.rss_after_gc_mb,
        peak_rss_after_gc_mb=memory_profile.peak_rss_after_gc_mb,
        search_star_rows=structure_stats.search_star_rows,
        ngs_rows=structure_stats.ngs_rows,
        close_pair_rows=structure_stats.close_pair_rows,
        raw_asterism_rows=structure_stats.raw_asterism_rows,
        candidate_graph_rows=structure_stats.candidate_graph_rows,
        local_asterism_rows=structure_stats.local_asterism_rows,
        context_pair_rows=structure_stats.context_pair_rows,
        winner_payload_rows=structure_stats.winner_payload_rows,
        point_prediction_batches=structure_stats.point_prediction_batches,
        point_prediction_rows=structure_stats.point_prediction_rows,
        point_prediction_batch_rows_peak=(
            structure_stats.point_prediction_batch_rows_peak
        ),
        point_prediction_backend_rows=structure_stats.point_prediction_backend_rows,
        point_prediction_backend_batch_rows_peak=(
            structure_stats.point_prediction_backend_batch_rows_peak
        ),
        point_prediction_backend_bucket_counts=_format_bucket_stats(
            structure_stats.point_prediction_backend_bucket_counts
        ),
        point_prediction_backend_bucket_rows=_format_bucket_stats(
            structure_stats.point_prediction_backend_bucket_rows
        ),
        point_feature_mib_peak=(
            structure_stats.point_feature_bytes_peak / (1024.0 * 1024.0)
        ),
        point_mps_current_mib_peak=(
            structure_stats.point_mps_current_bytes_peak / (1024.0 * 1024.0)
        ),
        point_mps_driver_mib_peak=(
            structure_stats.point_mps_driver_bytes_peak / (1024.0 * 1024.0)
        ),
        point_mps_recommended_mib=(
            structure_stats.point_mps_recommended_bytes / (1024.0 * 1024.0)
        ),
        field_mean_prediction_batches=structure_stats.field_mean_prediction_batches,
        field_mean_prediction_rows=structure_stats.field_mean_prediction_rows,
        field_mean_prediction_batch_rows_peak=(
            structure_stats.field_mean_prediction_batch_rows_peak
        ),
        field_mean_prediction_backend_rows=(
            structure_stats.field_mean_prediction_backend_rows
        ),
        field_mean_prediction_backend_batch_rows_peak=(
            structure_stats.field_mean_prediction_backend_batch_rows_peak
        ),
        field_mean_prediction_backend_bucket_counts=_format_bucket_stats(
            structure_stats.field_mean_prediction_backend_bucket_counts
        ),
        field_mean_prediction_backend_bucket_rows=_format_bucket_stats(
            structure_stats.field_mean_prediction_backend_bucket_rows
        ),
        field_mean_feature_mib_peak=(
            structure_stats.field_mean_feature_bytes_peak / (1024.0 * 1024.0)
        ),
        field_mean_mps_current_mib_peak=(
            structure_stats.field_mean_mps_current_bytes_peak / (1024.0 * 1024.0)
        ),
        field_mean_mps_driver_mib_peak=(
            structure_stats.field_mean_mps_driver_bytes_peak / (1024.0 * 1024.0)
        ),
        field_mean_mps_recommended_mib=(
            structure_stats.field_mean_mps_recommended_bytes / (1024.0 * 1024.0)
        ),
        artifact_inner_structured_mib=float(artifact_inner_structured_mib),
        artifact_asterism_structured_mib=float(artifact_asterism_structured_mib),
        error_message=error_message,
    )


def _worker_memory_sample(
    *,
    worker_id: int,
    event: str,
    run_started: float,
) -> TraversalWorkerMessage:
    return TraversalWorkerMessage(
        worker_id=worker_id,
        kind="worker_memory_sample",
        worker_memory_sample=TraversalWorkerMemorySample(
            worker_id=int(worker_id),
            event=event,
            elapsed_seconds=time.perf_counter() - run_started,
            rss_mb=_current_rss_mb(),
            peak_rss_mb=_peak_rss_mb(),
        ),
    )


def _iter_long_lived_traversal_worker_messages(
    context: TraversalTaskContext,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
    *,
    control_queue=None,
) -> Iterator[TraversalWorkerMessage]:
    """Yield Traversal messages from one reusable worker runtime."""

    run_started = time.perf_counter()
    detailed = execution_config.telemetry == "detailed"
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="worker_start",
            run_started=run_started,
        )
    configure_inference_threads(1)
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_inference_thread_config",
            run_started=run_started,
        )
    runtime = load_runtime_config(
        context.runtime_config_path,
        model_root=context.roots.model_root,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_runtime_config",
            run_started=run_started,
        )
    warm_model_cache(
        runtime,
        prediction_device=execution_config.prediction_device,
        averaged_prediction_device=execution_config.averaged_prediction_device,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_model_warmup",
            run_started=run_started,
        )
    base_store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=context.roots.gaia_root,
            release=context.definition.gaia_release,
            healpix_level=context.definition.outer_level,
        )
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_base_store",
            run_started=run_started,
        )
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=execution_config.gaia_cache_entries,
        max_bytes=execution_config.gaia_cache_mb * 1024 * 1024,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_runtime_gaia_store",
            run_started=run_started,
        )
    geometry = TraversalGeometry.from_runtime(runtime)
    if detailed:
        yield _worker_memory_sample(
            worker_id=plan.worker_id,
            event="after_geometry",
            run_started=run_started,
        )
    yield from _apply_worker_memory_pressure_command(
        control_queue=control_queue,
        store=store,
        worker_id=plan.worker_id,
        run_started=run_started,
        detailed=detailed,
        allow_pause=True,
    )
    completed = 0
    failed = 0
    pixel_seconds = 0.0
    artifact_write_telemetry = _ArtifactWriteTelemetry()
    stage_telemetry = _TraversalStageTelemetry()
    structure_telemetry = _TraversalStructureTelemetry()

    for outer_pix in plan.outer_pixs:
        yield from _apply_worker_memory_pressure_command(
            control_queue=control_queue,
            store=store,
            worker_id=plan.worker_id,
            run_started=run_started,
            detailed=detailed,
            allow_pause=True,
        )
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="started",
            outer_pix=int(outer_pix),
        )
        pixel_started = time.perf_counter()
        memory_profile = None
        artifact_memory_profile = None
        if execution_config.telemetry == "detailed":
            memory_profile = TraversalMemoryProfile(
                rss_start_mb=_current_rss_mb(),
                peak_rss_start_mb=_peak_rss_mb(),
            )
            artifact_memory_profile = ArtifactMemoryProfile()
        pressure_messages: list[TraversalWorkerMessage] = []

        def poll_memory_pressure() -> None:
            pressure_messages.extend(
                _apply_worker_memory_pressure_command(
                    control_queue=control_queue,
                    store=store,
                    worker_id=plan.worker_id,
                    run_started=run_started,
                    detailed=detailed,
                    allow_pause=True,
                )
            )

        try:
            write_profile, stage_stats, structure_stats = _materialize_outer_pixel_products(
                context,
                int(outer_pix),
                execution_config=execution_config,
                runtime=runtime,
                store=store,
                geometry=geometry,
                memory_profile=memory_profile,
                artifact_memory_profile=artifact_memory_profile,
                memory_pressure_callback=poll_memory_pressure,
            )
            artifact_write_telemetry.add(write_profile)
            stage_telemetry.add(stage_stats)
            structure_telemetry.add(structure_stats)
        except Exception as exc:
            elapsed = time.perf_counter() - pixel_started
            pixel_seconds += elapsed
            failed += 1
            gc.collect()
            if memory_profile is not None:
                memory_profile.rss_after_gc_mb = _current_rss_mb()
                memory_profile.peak_rss_after_gc_mb = _peak_rss_mb()
                yield TraversalWorkerMessage(
                    worker_id=plan.worker_id,
                    kind="memory_sample",
                    memory_sample=_build_memory_sample(
                        outer_pix=int(outer_pix),
                        worker_id=plan.worker_id,
                        success=False,
                        pixel_seconds=time.perf_counter() - pixel_started,
                        artifact_write_profile=None,
                        artifact_memory_profile=artifact_memory_profile,
                        structure_stats=None,
                        memory_profile=memory_profile,
                        error_message=str(exc),
                    ),
                )
            yield from pressure_messages
            yield TraversalWorkerMessage(
                worker_id=plan.worker_id,
                kind="failed",
                outer_pix=int(outer_pix),
                pixel_seconds=elapsed,
                peak_rss_mb=_peak_rss_mb(),
                result=TraversalTaskResult(
                    outer_pix=int(outer_pix),
                    success=False,
                    error_message=str(exc),
                ),
            )
            if (completed + failed) % TRAVERSAL_PROGRESS_LOG_INTERVAL == 0:
                yield _traversal_worker_stats_message(
                    worker_id=plan.worker_id,
                    completed=completed,
                    failed=failed,
                    run_started=run_started,
                    pixel_seconds=pixel_seconds,
                    artifact_write_telemetry=artifact_write_telemetry,
                    stage_telemetry=stage_telemetry,
                    structure_telemetry=structure_telemetry,
                    store=store,
                )
            yield from _apply_worker_memory_pressure_command(
                control_queue=control_queue,
                store=store,
                worker_id=plan.worker_id,
                run_started=run_started,
                detailed=detailed,
                allow_pause=True,
            )
            continue
        elapsed = time.perf_counter() - pixel_started
        pixel_seconds += elapsed
        completed += 1
        gc.collect()
        if memory_profile is not None:
            memory_profile.rss_after_gc_mb = _current_rss_mb()
            memory_profile.peak_rss_after_gc_mb = _peak_rss_mb()
            yield TraversalWorkerMessage(
                worker_id=plan.worker_id,
                kind="memory_sample",
                memory_sample=_build_memory_sample(
                    outer_pix=int(outer_pix),
                    worker_id=plan.worker_id,
                    success=True,
                    pixel_seconds=time.perf_counter() - pixel_started,
                    artifact_write_profile=write_profile,
                    artifact_memory_profile=artifact_memory_profile,
                    structure_stats=structure_stats,
                    memory_profile=memory_profile,
                ),
            )
        yield from pressure_messages
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="completed",
            outer_pix=int(outer_pix),
            pixel_seconds=elapsed,
            peak_rss_mb=_peak_rss_mb(),
            result=TraversalTaskResult(
                outer_pix=int(outer_pix),
                success=True,
            ),
        )
        if (completed + failed) % TRAVERSAL_PROGRESS_LOG_INTERVAL == 0:
            yield _traversal_worker_stats_message(
                worker_id=plan.worker_id,
                completed=completed,
                failed=failed,
                run_started=run_started,
                pixel_seconds=pixel_seconds,
                artifact_write_telemetry=artifact_write_telemetry,
                stage_telemetry=stage_telemetry,
                structure_telemetry=structure_telemetry,
                store=store,
            )
        yield from _apply_worker_memory_pressure_command(
            control_queue=control_queue,
            store=store,
            worker_id=plan.worker_id,
            run_started=run_started,
            detailed=detailed,
            allow_pause=True,
        )

    yield _traversal_worker_stats_message(
        worker_id=plan.worker_id,
        completed=completed,
        failed=failed,
        run_started=run_started,
        pixel_seconds=pixel_seconds,
        artifact_write_telemetry=artifact_write_telemetry,
        stage_telemetry=stage_telemetry,
        structure_telemetry=structure_telemetry,
        store=store,
    )

    if isinstance(store, RuntimeGaiaHealpixStore) and store.enabled:
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="cache_stats",
            cache_stats=_traversal_cache_stats(store),
        )
    yield TraversalWorkerMessage(worker_id=plan.worker_id, kind="done")


def _iter_dynamic_traversal_worker_messages(
    context: TraversalTaskContext,
    worker_id: int,
    execution_config: TraversalExecutionConfig,
    *,
    work_queue,
    control_queue=None,
) -> Iterator[TraversalWorkerMessage]:
    """Yield Traversal messages from one dynamically dispatched worker."""

    run_started = time.perf_counter()
    detailed = execution_config.telemetry == "detailed"
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="worker_start",
            run_started=run_started,
        )
    configure_inference_threads(1)
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_inference_thread_config",
            run_started=run_started,
        )
    runtime = load_runtime_config(
        context.runtime_config_path,
        model_root=context.roots.model_root,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_runtime_config",
            run_started=run_started,
        )
    warm_model_cache(
        runtime,
        prediction_device=execution_config.prediction_device,
        averaged_prediction_device=execution_config.averaged_prediction_device,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_model_warmup",
            run_started=run_started,
        )
    base_store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=context.roots.gaia_root,
            release=context.definition.gaia_release,
            healpix_level=context.definition.outer_level,
        )
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_base_store",
            run_started=run_started,
        )
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=execution_config.gaia_cache_entries,
        max_bytes=execution_config.gaia_cache_mb * 1024 * 1024,
    )
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_runtime_gaia_store",
            run_started=run_started,
        )
    geometry = TraversalGeometry.from_runtime(runtime)
    if detailed:
        yield _worker_memory_sample(
            worker_id=worker_id,
            event="after_geometry",
            run_started=run_started,
        )
    yield from _apply_worker_memory_pressure_command(
        control_queue=control_queue,
        store=store,
        worker_id=worker_id,
        run_started=run_started,
        detailed=detailed,
        allow_pause=True,
    )
    completed = 0
    failed = 0
    pixel_seconds = 0.0
    artifact_write_telemetry = _ArtifactWriteTelemetry()
    stage_telemetry = _TraversalStageTelemetry()
    structure_telemetry = _TraversalStructureTelemetry()

    while True:
        yield TraversalWorkerMessage(worker_id=worker_id, kind="ready")
        batch = work_queue.get()
        while batch == PARENT_MEMORY_PRESSURE_TRIM:
            _trim_worker_runtime_memory(store)
            if detailed:
                yield _worker_memory_sample(
                    worker_id=worker_id,
                    event="after_dynamic_idle_trim",
                    run_started=run_started,
                )
            batch = work_queue.get()
        if batch is None:
            break

        for outer_pix in batch.outer_pixs:
            yield from _apply_worker_memory_pressure_command(
                control_queue=control_queue,
                store=store,
                worker_id=worker_id,
                run_started=run_started,
                detailed=detailed,
                allow_pause=True,
            )
            yield TraversalWorkerMessage(
                worker_id=worker_id,
                kind="started",
                outer_pix=int(outer_pix),
            )
            pixel_started = time.perf_counter()
            memory_profile = None
            artifact_memory_profile = None
            if execution_config.telemetry == "detailed":
                memory_profile = TraversalMemoryProfile(
                    rss_start_mb=_current_rss_mb(),
                    peak_rss_start_mb=_peak_rss_mb(),
                )
                artifact_memory_profile = ArtifactMemoryProfile()
            pressure_messages: list[TraversalWorkerMessage] = []

            def poll_memory_pressure() -> None:
                pressure_messages.extend(
                    _apply_worker_memory_pressure_command(
                        control_queue=control_queue,
                        store=store,
                        worker_id=worker_id,
                        run_started=run_started,
                        detailed=detailed,
                        allow_pause=True,
                    )
                )

            try:
                write_profile, stage_stats, structure_stats = _materialize_outer_pixel_products(
                    context,
                    int(outer_pix),
                    execution_config=execution_config,
                    runtime=runtime,
                    store=store,
                    geometry=geometry,
                    memory_profile=memory_profile,
                    artifact_memory_profile=artifact_memory_profile,
                    memory_pressure_callback=poll_memory_pressure,
                )
                artifact_write_telemetry.add(write_profile)
                stage_telemetry.add(stage_stats)
                structure_telemetry.add(structure_stats)
            except Exception as exc:
                elapsed = time.perf_counter() - pixel_started
                pixel_seconds += elapsed
                failed += 1
                gc.collect()
                if memory_profile is not None:
                    memory_profile.rss_after_gc_mb = _current_rss_mb()
                    memory_profile.peak_rss_after_gc_mb = _peak_rss_mb()
                    yield TraversalWorkerMessage(
                        worker_id=worker_id,
                        kind="memory_sample",
                        memory_sample=_build_memory_sample(
                            outer_pix=int(outer_pix),
                            worker_id=worker_id,
                            success=False,
                            pixel_seconds=time.perf_counter() - pixel_started,
                            artifact_write_profile=None,
                            artifact_memory_profile=artifact_memory_profile,
                            structure_stats=None,
                            memory_profile=memory_profile,
                            error_message=str(exc),
                        ),
                    )
                yield from pressure_messages
                yield TraversalWorkerMessage(
                    worker_id=worker_id,
                    kind="failed",
                    outer_pix=int(outer_pix),
                    pixel_seconds=elapsed,
                    peak_rss_mb=_peak_rss_mb(),
                    result=TraversalTaskResult(
                        outer_pix=int(outer_pix),
                        success=False,
                        error_message=str(exc),
                    ),
                )
                if (completed + failed) % TRAVERSAL_PROGRESS_LOG_INTERVAL == 0:
                    yield _traversal_worker_stats_message(
                        worker_id=worker_id,
                        completed=completed,
                        failed=failed,
                        run_started=run_started,
                        pixel_seconds=pixel_seconds,
                        artifact_write_telemetry=artifact_write_telemetry,
                        stage_telemetry=stage_telemetry,
                        structure_telemetry=structure_telemetry,
                        store=store,
                    )
                yield from _apply_worker_memory_pressure_command(
                    control_queue=control_queue,
                    store=store,
                    worker_id=worker_id,
                    run_started=run_started,
                    detailed=detailed,
                    allow_pause=True,
                )
                continue

            elapsed = time.perf_counter() - pixel_started
            pixel_seconds += elapsed
            completed += 1
            gc.collect()
            if memory_profile is not None:
                memory_profile.rss_after_gc_mb = _current_rss_mb()
                memory_profile.peak_rss_after_gc_mb = _peak_rss_mb()
                yield TraversalWorkerMessage(
                    worker_id=worker_id,
                    kind="memory_sample",
                    memory_sample=_build_memory_sample(
                        outer_pix=int(outer_pix),
                        worker_id=worker_id,
                        success=True,
                        pixel_seconds=time.perf_counter() - pixel_started,
                        artifact_write_profile=write_profile,
                        artifact_memory_profile=artifact_memory_profile,
                        structure_stats=structure_stats,
                        memory_profile=memory_profile,
                    ),
                )
            yield from pressure_messages
            yield TraversalWorkerMessage(
                worker_id=worker_id,
                kind="completed",
                outer_pix=int(outer_pix),
                pixel_seconds=elapsed,
                peak_rss_mb=_peak_rss_mb(),
                result=TraversalTaskResult(
                    outer_pix=int(outer_pix),
                    success=True,
                ),
            )
            if (completed + failed) % TRAVERSAL_PROGRESS_LOG_INTERVAL == 0:
                yield _traversal_worker_stats_message(
                    worker_id=worker_id,
                    completed=completed,
                    failed=failed,
                    run_started=run_started,
                    pixel_seconds=pixel_seconds,
                    artifact_write_telemetry=artifact_write_telemetry,
                    stage_telemetry=stage_telemetry,
                    structure_telemetry=structure_telemetry,
                    store=store,
                )
            yield from _apply_worker_memory_pressure_command(
                control_queue=control_queue,
                store=store,
                worker_id=worker_id,
                run_started=run_started,
                detailed=detailed,
                allow_pause=True,
            )

    yield _traversal_worker_stats_message(
        worker_id=worker_id,
        completed=completed,
        failed=failed,
        run_started=run_started,
        pixel_seconds=pixel_seconds,
        artifact_write_telemetry=artifact_write_telemetry,
        stage_telemetry=stage_telemetry,
        structure_telemetry=structure_telemetry,
        store=store,
    )

    if isinstance(store, RuntimeGaiaHealpixStore) and store.enabled:
        yield TraversalWorkerMessage(
            worker_id=worker_id,
            kind="cache_stats",
            cache_stats=_traversal_cache_stats(store),
        )
    yield TraversalWorkerMessage(worker_id=worker_id, kind="done")


def _handle_regional_worker_message(
    *,
    build_path: Path,
    state: np.ndarray,
    message: TraversalWorkerMessage,
    star_counts: np.ndarray | None = None,
    progress: dict[int, dict[str, int]] | None = None,
    progress_interval: int = TRAVERSAL_PROGRESS_LOG_INTERVAL,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
    diagnostics_writer: _TraversalDiagnosticsWriter | None = None,
) -> bool:
    """Apply one regional worker message. Return whether it marks a failure."""

    worker_progress = None
    if progress is not None:
        worker_progress = progress.setdefault(
            int(message.worker_id),
            {"completed": 0, "failed": 0},
        )

    if message.kind == "started":
        if message.outer_pix is None:
            raise BuildError("Regional worker start message is missing outer_pix")
        _mark_outer_pixel_running(
            build_path,
            state,
            int(message.outer_pix),
            telemetry,
            state_writer=state_writer,
        )
        star_count = (
            int(star_counts[int(message.outer_pix)])
            if star_counts is not None
            else -1
        )
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={message.worker_id} "
            "outer_pixel_start "
            f"outer_pix={message.outer_pix} "
            f"star_count={star_count}",
        )
        return False

    if message.kind in ("completed", "failed"):
        if message.result is None:
            raise BuildError(f"Regional worker {message.kind} message is missing result")
        _record_traversal_result(
            build_path,
            state,
            message.result,
            telemetry,
            state_writer=state_writer,
        )
        if message.result.success:
            append_build_log(
                build_path,
                "phase=traversal "
                f"worker={message.worker_id} "
                "outer_pixel_done "
                f"outer_pix={message.result.outer_pix} "
                f"elapsed_s={message.pixel_seconds:.3f} "
                f"peak_rss_mb={message.peak_rss_mb:.1f}",
            )
            if worker_progress is not None:
                worker_progress["completed"] += 1
                if (
                    progress_interval > 0
                    and worker_progress["completed"] % progress_interval == 0
                ):
                    append_build_log(
                        build_path,
                        "phase=traversal "
                        f"worker={message.worker_id} "
                        f"progress completed={worker_progress['completed']} "
                            f"failed={worker_progress['failed']} "
                            f"last_outer_pix={message.result.outer_pix}",
                    )
                    if state_writer is not None:
                        state_writer.flush()
                    if telemetry is not None:
                        _append_state_update_telemetry(build_path, telemetry)
            return False
        if worker_progress is not None:
            worker_progress["failed"] += 1
        if state_writer is not None:
            state_writer.flush()
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={message.worker_id} "
            "outer_pixel_failed "
            f"outer_pix={message.result.outer_pix} "
            f"elapsed_s={message.pixel_seconds:.3f} "
            f"peak_rss_mb={message.peak_rss_mb:.1f} "
            f"error={message.result.error_message}",
        )
        if telemetry is not None:
            _append_state_update_telemetry(build_path, telemetry)
        return True

    if message.kind == "cache_stats":
        stats = message.cache_stats
        if stats is None:
            raise BuildError("Regional worker cache_stats message is missing stats")
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={message.worker_id} "
            "cache_stats "
            f"hits={stats.hits} "
            f"misses={stats.misses} "
            f"evictions={stats.evictions} "
            f"oversized_skips={stats.oversized_skips} "
            f"entries={stats.entries} "
            f"current_bytes={stats.current_bytes} "
            f"peak_bytes={stats.peak_bytes} "
            f"load_s={stats.load_seconds:.3f} "
            f"raw_load_s={stats.raw_load_seconds:.3f} "
            f"prepare_s={stats.prepare_seconds:.3f}",
        )
        return False

    if message.kind == "profile":
        stats = message.worker_stats
        if stats is None:
            raise BuildError("Regional worker profile message is missing stats")
        cache_stats = stats.cache_stats
        processed = stats.completed + stats.failed
        average_pixel_seconds = stats.pixel_seconds / processed if processed else 0.0
        average_write_seconds = stats.artifact_write_seconds / processed if processed else 0.0
        artifact_convert_seconds = (
            stats.artifact_convert_inner_seconds
            + stats.artifact_convert_asterisms_seconds
        )
        artifact_hdf5_seconds = (
            stats.artifact_hdf5_open_seconds
            + stats.artifact_hdf5_inner_seconds
            + stats.artifact_hdf5_asterisms_seconds
            + stats.artifact_hdf5_close_seconds
        )
        stage_stats = stats.stage_stats
        structure_stats = stats.structure_stats
        traversal_profiled_seconds = (
            stage_stats.star_selection_seconds
            + stage_stats.candidate_generation_seconds
            + stage_stats.filtering_seconds
            + stage_stats.local_selection_seconds
            + stage_stats.inner_table_seconds
            + stage_stats.context_seconds
            + stage_stats.point_prediction_seconds
            + stage_stats.field_mean_prediction_seconds
            + stage_stats.coverage_seconds
            + stage_stats.dust_seconds
            + stage_stats.persisted_asterisms_seconds
        )
        traversal_other_seconds = (
            stats.pixel_seconds
            - stats.artifact_write_seconds
            - cache_stats.load_seconds
        )
        traversal_unprofiled_seconds = (
            stats.pixel_seconds
            - stats.artifact_write_seconds
            - traversal_profiled_seconds
        )
        average_other_seconds = traversal_other_seconds / processed if processed else 0.0
        cache_mb = cache_stats.current_bytes / (1024.0 * 1024.0)
        peak_cache_mb = cache_stats.peak_bytes / (1024.0 * 1024.0)
        artifact_mib = stats.artifact_bytes / (1024.0 * 1024.0)
        artifact_inner_input_mib = stats.artifact_inner_input_bytes / (1024.0 * 1024.0)
        artifact_asterism_input_mib = stats.artifact_asterism_input_bytes / (
            1024.0 * 1024.0
        )
        artifact_inner_structured_mib = stats.artifact_inner_structured_bytes / (
            1024.0 * 1024.0
        )
        artifact_asterism_structured_mib = stats.artifact_asterism_structured_bytes / (
            1024.0 * 1024.0
        )
        point_feature_mib_peak = structure_stats.point_feature_bytes_peak / (
            1024.0 * 1024.0
        )
        point_mps_current_mib_peak = structure_stats.point_mps_current_bytes_peak / (
            1024.0 * 1024.0
        )
        point_mps_driver_mib_peak = structure_stats.point_mps_driver_bytes_peak / (
            1024.0 * 1024.0
        )
        point_mps_recommended_mib = structure_stats.point_mps_recommended_bytes / (
            1024.0 * 1024.0
        )
        field_mean_feature_mib_peak = structure_stats.field_mean_feature_bytes_peak / (
            1024.0 * 1024.0
        )
        field_mean_mps_current_mib_peak = (
            structure_stats.field_mean_mps_current_bytes_peak / (1024.0 * 1024.0)
        )
        field_mean_mps_driver_mib_peak = (
            structure_stats.field_mean_mps_driver_bytes_peak / (1024.0 * 1024.0)
        )
        field_mean_mps_recommended_mib = (
            structure_stats.field_mean_mps_recommended_bytes / (1024.0 * 1024.0)
        )
        point_backend_bucket_counts = _format_bucket_stats(
            structure_stats.point_prediction_backend_bucket_counts
        )
        point_backend_bucket_rows = _format_bucket_stats(
            structure_stats.point_prediction_backend_bucket_rows
        )
        field_mean_backend_bucket_counts = _format_bucket_stats(
            structure_stats.field_mean_prediction_backend_bucket_counts
        )
        field_mean_backend_bucket_rows = _format_bucket_stats(
            structure_stats.field_mean_prediction_backend_bucket_rows
        )
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={message.worker_id} "
            "profile "
            f"completed={stats.completed} "
            f"failed={stats.failed} "
            f"elapsed_s={stats.elapsed_seconds:.3f} "
            f"pixel_s={stats.pixel_seconds:.3f} "
            f"avg_pixel_s={average_pixel_seconds:.3f} "
            f"traversal_other_s={traversal_other_seconds:.3f} "
            f"avg_traversal_other_s={average_other_seconds:.3f} "
            f"traversal_profiled_s={traversal_profiled_seconds:.3f} "
            f"traversal_unprofiled_s={traversal_unprofiled_seconds:.3f} "
            f"stage_star_selection_s={stage_stats.star_selection_seconds:.3f} "
            f"stage_candidate_generation_s={stage_stats.candidate_generation_seconds:.3f} "
            f"stage_filtering_s={stage_stats.filtering_seconds:.3f} "
            f"stage_bright_star_filter_s={stage_stats.bright_star_filter_seconds:.3f} "
            f"stage_inner_assignment_s={stage_stats.inner_assignment_seconds:.3f} "
            f"stage_local_selection_s={stage_stats.local_selection_seconds:.3f} "
            f"stage_inner_table_s={stage_stats.inner_table_seconds:.3f} "
            f"stage_context_s={stage_stats.context_seconds:.3f} "
            f"stage_point_prediction_s={stage_stats.point_prediction_seconds:.3f} "
            f"stage_point_prediction_eligibility_s={stage_stats.point_prediction_eligibility_seconds:.3f} "
            f"stage_point_prediction_eligibility_intersection_s={stage_stats.point_prediction_eligibility_intersection_seconds:.3f} "
            f"stage_point_prediction_eligibility_extract_s={stage_stats.point_prediction_eligibility_extract_seconds:.3f} "
            f"stage_point_prediction_buffer_s={stage_stats.point_prediction_buffer_seconds:.3f} "
            f"stage_point_prediction_ngs_array_s={stage_stats.point_prediction_ngs_array_seconds:.3f} "
            f"stage_point_prediction_model_s={stage_stats.point_prediction_model_seconds:.3f} "
            f"stage_point_prediction_feature_s={stage_stats.point_prediction_feature_seconds:.3f} "
            f"stage_point_prediction_backend_s={stage_stats.point_prediction_backend_seconds:.3f} "
            f"stage_point_prediction_scatter_s={stage_stats.point_prediction_scatter_seconds:.3f} "
            f"stage_point_prediction_scatter_filter_s={stage_stats.point_prediction_scatter_filter_seconds:.3f} "
            f"stage_point_prediction_scatter_merge_s={stage_stats.point_prediction_scatter_merge_seconds:.3f} "
            f"stage_point_prediction_scatter_sort_s={stage_stats.point_prediction_scatter_sort_seconds:.3f} "
            f"stage_point_prediction_scatter_write_s={stage_stats.point_prediction_scatter_write_seconds:.3f} "
            f"stage_point_prediction_cache_clear_s={stage_stats.point_prediction_cache_clear_seconds:.3f} "
            f"stage_field_mean_prediction_s={stage_stats.field_mean_prediction_seconds:.3f} "
            f"stage_field_mean_prediction_feature_s={stage_stats.field_mean_prediction_feature_seconds:.3f} "
            f"stage_field_mean_prediction_backend_s={stage_stats.field_mean_prediction_backend_seconds:.3f} "
            f"stage_coverage_s={stage_stats.coverage_seconds:.3f} "
            f"stage_dust_s={stage_stats.dust_seconds:.3f} "
            f"stage_persisted_asterisms_s={stage_stats.persisted_asterisms_seconds:.3f} "
            f"artifact_write_s={stats.artifact_write_seconds:.3f} "
            f"avg_artifact_write_s={average_write_seconds:.3f} "
            f"artifact_convert_s={artifact_convert_seconds:.3f} "
            f"artifact_convert_inner_s={stats.artifact_convert_inner_seconds:.3f} "
            f"artifact_convert_asterisms_s={stats.artifact_convert_asterisms_seconds:.3f} "
            f"artifact_hdf5_s={artifact_hdf5_seconds:.3f} "
            f"artifact_hdf5_open_s={stats.artifact_hdf5_open_seconds:.3f} "
            f"artifact_hdf5_inner_s={stats.artifact_hdf5_inner_seconds:.3f} "
            f"artifact_hdf5_asterisms_s={stats.artifact_hdf5_asterisms_seconds:.3f} "
            f"artifact_hdf5_close_s={stats.artifact_hdf5_close_seconds:.3f} "
            f"artifact_replace_s={stats.artifact_replace_seconds:.3f} "
            f"artifact_mib={artifact_mib:.1f} "
            f"artifact_inner_input_mib={artifact_inner_input_mib:.1f} "
            f"artifact_asterism_input_mib={artifact_asterism_input_mib:.1f} "
            f"artifact_inner_structured_mib={artifact_inner_structured_mib:.1f} "
            f"artifact_asterism_structured_mib={artifact_asterism_structured_mib:.1f} "
            f"artifact_inner_rows={stats.artifact_inner_rows} "
            f"artifact_asterism_rows={stats.artifact_asterism_rows} "
            f"search_star_rows={structure_stats.search_star_rows} "
            f"ngs_rows={structure_stats.ngs_rows} "
            f"close_pair_rows={structure_stats.close_pair_rows} "
            f"self_pair_rows={structure_stats.self_pair_rows} "
            f"raw_asterism_rows={structure_stats.raw_asterism_rows} "
            f"dedupe_key_rows={structure_stats.dedupe_key_rows} "
            f"post_bright_asterism_rows={structure_stats.post_bright_asterism_rows} "
            f"candidate_graph_rows={structure_stats.candidate_graph_rows} "
            f"local_asterism_rows={structure_stats.local_asterism_rows} "
            f"context_pair_rows={structure_stats.context_pair_rows} "
            f"winner_rows={structure_stats.winner_rows} "
            f"winner_payload_rows={structure_stats.winner_payload_rows} "
            f"point_prediction_batches={structure_stats.point_prediction_batches} "
            f"point_prediction_rows={structure_stats.point_prediction_rows} "
            f"recovered_point_prediction_rows={structure_stats.recovered_point_prediction_rows} "
            f"recovered_winner_pixels={structure_stats.recovered_winner_pixels} "
            f"point_prediction_batch_rows_peak={structure_stats.point_prediction_batch_rows_peak} "
            f"point_prediction_backend_rows={structure_stats.point_prediction_backend_rows} "
            f"point_prediction_backend_batch_rows_peak={structure_stats.point_prediction_backend_batch_rows_peak} "
            f"point_prediction_backend_bucket_counts={point_backend_bucket_counts or '-'} "
            f"point_prediction_backend_bucket_rows={point_backend_bucket_rows or '-'} "
            f"point_feature_mib_peak={point_feature_mib_peak:.3f} "
            f"point_mps_current_mib_peak={point_mps_current_mib_peak:.3f} "
            f"point_mps_driver_mib_peak={point_mps_driver_mib_peak:.3f} "
            f"point_mps_recommended_mib={point_mps_recommended_mib:.3f} "
            f"field_mean_prediction_batches={structure_stats.field_mean_prediction_batches} "
            f"field_mean_prediction_rows={structure_stats.field_mean_prediction_rows} "
            f"field_mean_prediction_batch_rows_peak={structure_stats.field_mean_prediction_batch_rows_peak} "
            f"field_mean_prediction_backend_rows={structure_stats.field_mean_prediction_backend_rows} "
            f"field_mean_prediction_backend_batch_rows_peak={structure_stats.field_mean_prediction_backend_batch_rows_peak} "
            f"field_mean_prediction_backend_bucket_counts={field_mean_backend_bucket_counts or '-'} "
            f"field_mean_prediction_backend_bucket_rows={field_mean_backend_bucket_rows or '-'} "
            f"field_mean_feature_mib_peak={field_mean_feature_mib_peak:.3f} "
            f"field_mean_mps_current_mib_peak={field_mean_mps_current_mib_peak:.3f} "
            f"field_mean_mps_driver_mib_peak={field_mean_mps_driver_mib_peak:.3f} "
            f"field_mean_mps_recommended_mib={field_mean_mps_recommended_mib:.3f} "
            f"search_star_rows_peak={structure_stats.search_star_rows_peak} "
            f"ngs_rows_peak={structure_stats.ngs_rows_peak} "
            f"close_pair_rows_peak={structure_stats.close_pair_rows_peak} "
            f"context_pair_rows_peak={structure_stats.context_pair_rows_peak} "
            f"raw_asterism_rows_peak={structure_stats.raw_asterism_rows_peak} "
            f"local_asterism_rows_peak={structure_stats.local_asterism_rows_peak} "
            f"winner_payload_rows_peak={structure_stats.winner_payload_rows_peak} "
            f"peak_rss_mb={stats.peak_rss_mb:.1f} "
            f"cache_hits={cache_stats.hits} "
            f"cache_misses={cache_stats.misses} "
            f"cache_evictions={cache_stats.evictions} "
            f"cache_oversized_skips={cache_stats.oversized_skips} "
            f"cache_entries={cache_stats.entries} "
            f"cache_mb={cache_mb:.1f} "
            f"cache_peak_mb={peak_cache_mb:.1f} "
            f"gaia_load_s={cache_stats.load_seconds:.3f} "
            f"gaia_raw_load_s={cache_stats.raw_load_seconds:.3f} "
            f"gaia_prepare_s={cache_stats.prepare_seconds:.3f}",
        )
        return False

    if message.kind == "memory_sample":
        if message.memory_sample is None:
            raise BuildError("Regional worker memory_sample message is missing sample")
        if diagnostics_writer is not None:
            diagnostics_writer.write(message.memory_sample)
        return False

    if message.kind == "worker_memory_sample":
        if message.worker_memory_sample is None:
            raise BuildError("Regional worker worker_memory_sample message is missing sample")
        if diagnostics_writer is not None:
            diagnostics_writer.write_worker(message.worker_memory_sample)
        return False

    if message.kind == "done":
        return False

    raise BuildError(f"Unknown regional worker message kind {message.kind!r}")


def _run_regional_traversal_workers(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    plans: tuple[TraversalWorkerPlan, ...],
    execution_config: TraversalExecutionConfig,
    star_counts: np.ndarray | None = None,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
    diagnostics_writer: _TraversalDiagnosticsWriter | None = None,
) -> bool:
    """Run long-lived region-owned Traversal workers."""

    if not plans:
        return False

    process_context = _create_regional_process_context()
    result_queue = process_context.Queue()
    control_queues = {
        plan.worker_id: process_context.Queue()
        for plan in plans
    }
    processes = []
    process_by_worker_id: dict[int, object] = {}
    active_worker_ids = {plan.worker_id for plan in plans}
    worker_pressure_states = {
        plan.worker_id: PARENT_MEMORY_PRESSURE_NORMAL
        for plan in plans
    }
    worker_pressure_command_times: dict[int, float] = {}
    memory_pressure_state = PARENT_MEMORY_PRESSURE_NORMAL
    memory_pressure_targets: tuple[int, ...] = ()
    progress: dict[int, dict[str, int]] = {
        plan.worker_id: {"completed": 0, "failed": 0} for plan in plans
    }
    failed = False
    last_parent_memory_check = 0.0
    last_parent_memory_snapshot = 0.0

    for plan in plans:
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={plan.worker_id} "
            f"regions={','.join(str(pix) for pix in plan.region_pixs)} "
            f"outer_pixels={len(plan.outer_pixs)} "
            f"estimated_star_count={plan.estimated_star_count} started",
        )
        process = process_context.Process(
            target=_run_regional_traversal_worker,
            args=(
                context,
                plan,
                execution_config,
                result_queue,
                control_queues[plan.worker_id],
            ),
        )
        process.start()
        processes.append(process)
        process_by_worker_id[plan.worker_id] = process

    try:
        while active_worker_ids:
            now = time.perf_counter()
            if now - last_parent_memory_check >= PARENT_MEMORY_CHECK_INTERVAL_SECONDS:
                force_snapshot = (
                    now - last_parent_memory_snapshot
                    >= PARENT_MEMORY_SNAPSHOT_INTERVAL_SECONDS
                )
                memory_pressure_state, memory_pressure_targets = (
                    _update_parent_memory_pressure(
                        build_path=build_path,
                        execution_config=execution_config,
                        process_by_worker_id=process_by_worker_id,
                        control_queues=control_queues,
                        active_worker_ids=active_worker_ids,
                        worker_pressure_states=worker_pressure_states,
                        worker_pressure_command_times=worker_pressure_command_times,
                        previous_state=memory_pressure_state,
                        previous_targets=memory_pressure_targets,
                        force_snapshot=force_snapshot,
                    )
                )
                if force_snapshot:
                    last_parent_memory_snapshot = now
                _raise_if_parent_memory_limit_exceeded(execution_config, processes)
                last_parent_memory_check = now
            try:
                message = result_queue.get(timeout=0.1)
            except Empty:
                for process, plan in zip(processes, plans, strict=True):
                    if plan.worker_id not in active_worker_ids:
                        continue
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"regional worker {plan.worker_id} exited with code {process.exitcode}"
                        )
                continue

            failed = _handle_regional_worker_message(
                build_path=build_path,
                state=state,
                message=message,
                star_counts=star_counts,
                progress=progress,
                telemetry=telemetry,
                state_writer=state_writer,
                diagnostics_writer=diagnostics_writer,
            ) or failed
            now = time.perf_counter()
            if now - last_parent_memory_check >= PARENT_MEMORY_CHECK_INTERVAL_SECONDS:
                force_snapshot = (
                    now - last_parent_memory_snapshot
                    >= PARENT_MEMORY_SNAPSHOT_INTERVAL_SECONDS
                )
                memory_pressure_state, memory_pressure_targets = (
                    _update_parent_memory_pressure(
                        build_path=build_path,
                        execution_config=execution_config,
                        process_by_worker_id=process_by_worker_id,
                        control_queues=control_queues,
                        active_worker_ids=active_worker_ids,
                        worker_pressure_states=worker_pressure_states,
                        worker_pressure_command_times=worker_pressure_command_times,
                        previous_state=memory_pressure_state,
                        previous_targets=memory_pressure_targets,
                        force_snapshot=force_snapshot,
                    )
                )
                if force_snapshot:
                    last_parent_memory_snapshot = now
                _raise_if_parent_memory_limit_exceeded(execution_config, processes)
                last_parent_memory_check = now
            if message.kind == "done":
                active_worker_ids.discard(message.worker_id)
                worker_pressure_states[message.worker_id] = (
                    PARENT_MEMORY_PRESSURE_NORMAL
                )
                worker_progress = progress[message.worker_id]
                append_build_log(
                    build_path,
                    "phase=traversal "
                    f"worker={message.worker_id} "
                    f"done completed={worker_progress['completed']} "
                    f"failed={worker_progress['failed']}",
                )
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
    return failed


def _run_dynamic_traversal_workers(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    schedule,
    execution_config: TraversalExecutionConfig,
    star_counts: np.ndarray | None = None,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
    diagnostics_writer: _TraversalDiagnosticsWriter | None = None,
) -> bool:
    """Run parent-dispatched dynamic Traversal workers."""

    if not schedule.batches:
        return False

    worker_count = int(execution_config.workers)
    process_context = _create_regional_process_context()
    result_queue = process_context.Queue()
    control_queues = {worker_id: process_context.Queue() for worker_id in range(worker_count)}
    work_queues = {worker_id: process_context.Queue() for worker_id in range(worker_count)}
    processes = []
    process_by_worker_id: dict[int, object] = {}
    active_worker_ids = set(range(worker_count))
    worker_pressure_states = {
        worker_id: PARENT_MEMORY_PRESSURE_NORMAL for worker_id in range(worker_count)
    }
    worker_pressure_command_times: dict[int, float] = {}
    memory_pressure_state = PARENT_MEMORY_PRESSURE_NORMAL
    memory_pressure_targets: tuple[int, ...] = ()
    progress: dict[int, dict[str, int]] = {
        worker_id: {"completed": 0, "failed": 0} for worker_id in range(worker_count)
    }
    ram_bins = _split_dynamic_ram_bins(list(schedule.batches), bin_count=worker_count)
    post_stress_bins: list[list[TraversalWorkBatch]] | None = None
    post_stress_assignments: np.ndarray | None = None
    idle_worker_ids: list[int] = []
    active_batches: dict[int, TraversalWorkBatch] = {}
    active_batch_vectors: dict[int, tuple[float, float, float]] = {}
    active_stress_workers: set[int] = set()
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]] = {}
    failed = False
    last_parent_memory_check = 0.0
    last_parent_memory_snapshot = 0.0

    def projected_total_ram_mb(worker_id: int, batch: TraversalWorkBatch) -> float:
        active_ram = 0.0
        for active_worker_id, active_batch in active_batches.items():
            if int(active_worker_id) == int(worker_id):
                continue
            active_ram += float(active_batch.estimated_ram_mb)
        return active_ram + float(batch.estimated_ram_mb)

    def exceeds_trim_threshold(worker_id: int, batch: TraversalWorkBatch) -> bool:
        limit = int(execution_config.parent_memory_limit_mb)
        if limit <= 0 or not active_batches:
            return False
        return projected_total_ram_mb(worker_id, batch) > (
            float(limit) * PARENT_MEMORY_SOFT_FRACTION
        )

    def can_assign(
        worker_id: int,
        batch: TraversalWorkBatch,
        *,
        must_run: bool,
    ) -> bool:
        return bool(must_run) or not exceeds_trim_threshold(worker_id, batch)

    def take_batch_from_bins(
        bins: list[list[TraversalWorkBatch]],
        bin_indexes: list[int],
        *,
        worker_id: int,
        preferred_affinity: tuple[int, int] | None,
        require_stress: bool | None,
        must_run_stress: bool,
    ) -> tuple[TraversalWorkBatch | None, int, bool, bool]:
        denied_for_memory = False
        for bin_index in bin_indexes:
            if bin_index < 0 or bin_index >= len(bins):
                continue
            queue = bins[bin_index]
            for batch_index in _ordered_dynamic_batch_indexes(
                queue,
                preferred_affinity=preferred_affinity,
                active_vectors=active_batch_vectors,
                require_stress=require_stress,
            ):
                candidate = queue[batch_index]
                must_run = bool(candidate.is_stress) and bool(must_run_stress)
                if can_assign(worker_id, candidate, must_run=must_run):
                    return (
                        queue.pop(batch_index),
                        int(bin_index),
                        bool(candidate.is_stress),
                        denied_for_memory,
                    )
                denied_for_memory = True
        return None, -1, False, denied_for_memory

    def assign_next(worker_id: int) -> bool:
        nonlocal post_stress_bins, post_stress_assignments
        batch: TraversalWorkBatch | None = None
        is_stress = False
        denied_for_memory = False
        stress_remaining = _dynamic_bins_have_stress(ram_bins)
        non_stress_remaining = _dynamic_bins_have_non_stress(ram_bins)
        stress_limit = int(schedule.stress_worker_count)
        stress_affinity_idle_workers = [
            int(idle_worker_id)
            for idle_worker_id in idle_worker_ids
            if _worker_has_dynamic_stress_affinity(
                last_affinity_by_worker,
                worker_id=int(idle_worker_id),
                region_level=max(0, context.definition.outer_level - 1),
                stress_star_threshold=float(schedule.stress_star_threshold),
            )
        ]
        if stress_remaining and len(active_stress_workers) < stress_limit:
            if not stress_affinity_idle_workers or int(worker_id) in stress_affinity_idle_workers:
                stress_bin_index = _highest_dynamic_stress_bin_index(ram_bins)
                unthrottled_stress_slots = max(stress_limit - 1, 0)
                batch, _, is_stress, denied_for_memory = take_batch_from_bins(
                    ram_bins,
                    [stress_bin_index],
                    worker_id=int(worker_id),
                    preferred_affinity=_last_dynamic_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, context.definition.outer_level - 1),
                    ),
                    require_stress=True,
                    must_run_stress=(
                        len(active_stress_workers) < unthrottled_stress_slots
                    ),
                )
                if batch is None and denied_for_memory:
                    batch, _, is_stress, fallback_denied = take_batch_from_bins(
                        ram_bins,
                        list(range(len(ram_bins))),
                        worker_id=int(worker_id),
                        preferred_affinity=None,
                        require_stress=None,
                        must_run_stress=False,
                    )
                    denied_for_memory = bool(denied_for_memory or fallback_denied)
            elif not non_stress_remaining:
                return False

        if batch is None:
            stress_phase_active = _dynamic_bins_have_stress(ram_bins) or bool(
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
                    preferred_affinity=_last_dynamic_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, context.definition.outer_level - 3),
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
                    post_stress_bins = _split_dynamic_runtime_bins(
                        remaining,
                        bin_count=worker_count,
                    )
                    post_stress_assignments = np.zeros(
                        len(post_stress_bins),
                        dtype=np.int64,
                    )
                    for bin_rows in ram_bins:
                        bin_rows.clear()
                bin_indexes = _ordered_dynamic_runtime_bin_indexes(
                    post_stress_bins,
                    assignments=post_stress_assignments,
                )
                batch, bin_index, is_stress, denied_for_memory = take_batch_from_bins(
                    post_stress_bins,
                    bin_indexes,
                    worker_id=int(worker_id),
                    preferred_affinity=_last_dynamic_affinity_for_level(
                        last_affinity_by_worker,
                        worker_id=int(worker_id),
                        region_level=max(0, context.definition.outer_level - 3),
                    ),
                    require_stress=None,
                    must_run_stress=False,
                )
                if batch is not None and bin_index >= 0:
                    post_stress_assignments[bin_index] += 1

        if batch is None:
            if denied_for_memory:
                work_queues[int(worker_id)].put(PARENT_MEMORY_PRESSURE_TRIM)
            return False

        work_queues[int(worker_id)].put(batch)
        active_batches[int(worker_id)] = batch
        active_batch_vectors[int(worker_id)] = batch.center_vector
        last_affinity_by_worker[(int(worker_id), int(batch.region_level))] = (
            _dynamic_affinity_parent(
                region_level=int(batch.region_level),
                region_pix=int(batch.region_pix),
            ),
            int(batch.sort_star_count),
        )
        if bool(is_stress):
            active_stress_workers.add(int(worker_id))
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={worker_id} "
            f"dynamic_batch region_level={batch.region_level} "
            f"region={batch.region_pix} "
            f"outer_pixels={len(batch.outer_pixs)} "
            f"stress={int(batch.is_stress)} "
            f"estimated_ram_mb={batch.estimated_ram_mb:.1f} "
            f"estimated_seconds={batch.estimated_seconds:.2f}",
        )
        return True

    def assign_idle_workers() -> None:
        index = 0
        while index < len(idle_worker_ids):
            worker_id = idle_worker_ids.pop(index)
            if assign_next(worker_id):
                continue
            idle_worker_ids.insert(index, worker_id)
            index += 1

    for worker_id in range(worker_count):
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={worker_id} dynamic started",
        )
        process = process_context.Process(
            target=_run_dynamic_traversal_worker,
            args=(
                context,
                worker_id,
                execution_config,
                result_queue,
                work_queues[worker_id],
                control_queues[worker_id],
            ),
        )
        process.start()
        processes.append(process)
        process_by_worker_id[worker_id] = process

    try:
        while active_worker_ids:
            now = time.perf_counter()
            if now - last_parent_memory_check >= PARENT_MEMORY_CHECK_INTERVAL_SECONDS:
                force_snapshot = (
                    now - last_parent_memory_snapshot
                    >= PARENT_MEMORY_SNAPSHOT_INTERVAL_SECONDS
                )
                memory_pressure_state, memory_pressure_targets = (
                    _update_parent_memory_pressure(
                        build_path=build_path,
                        execution_config=execution_config,
                        process_by_worker_id=process_by_worker_id,
                        control_queues=control_queues,
                        active_worker_ids=active_worker_ids,
                        worker_pressure_states=worker_pressure_states,
                        worker_pressure_command_times=worker_pressure_command_times,
                        previous_state=memory_pressure_state,
                        previous_targets=memory_pressure_targets,
                        force_snapshot=force_snapshot,
                    )
                )
                if force_snapshot:
                    last_parent_memory_snapshot = now
                _raise_if_parent_memory_limit_exceeded(execution_config, processes)
                last_parent_memory_check = now
            try:
                message = result_queue.get(timeout=0.1)
            except Empty:
                for process, worker_id in zip(processes, range(worker_count), strict=True):
                    if worker_id not in active_worker_ids:
                        continue
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"dynamic worker {worker_id} exited with code {process.exitcode}"
                        )
                continue

            if message.kind == "ready":
                previous = active_batches.pop(int(message.worker_id), None)
                active_batch_vectors.pop(int(message.worker_id), None)
                if previous is not None and previous.is_stress:
                    active_stress_workers.discard(int(message.worker_id))
                if int(message.worker_id) not in idle_worker_ids:
                    idle_worker_ids.append(int(message.worker_id))
                assign_idle_workers()
                if not _dynamic_batches_remaining(ram_bins, post_stress_bins):
                    if int(message.worker_id) in idle_worker_ids:
                        idle_worker_ids.remove(int(message.worker_id))
                    work_queues[int(message.worker_id)].put(None)
                continue

            failed = _handle_regional_worker_message(
                build_path=build_path,
                state=state,
                message=message,
                star_counts=star_counts,
                progress=progress,
                telemetry=telemetry,
                state_writer=state_writer,
                diagnostics_writer=diagnostics_writer,
            ) or failed
            now = time.perf_counter()
            if now - last_parent_memory_check >= PARENT_MEMORY_CHECK_INTERVAL_SECONDS:
                force_snapshot = (
                    now - last_parent_memory_snapshot
                    >= PARENT_MEMORY_SNAPSHOT_INTERVAL_SECONDS
                )
                memory_pressure_state, memory_pressure_targets = (
                    _update_parent_memory_pressure(
                        build_path=build_path,
                        execution_config=execution_config,
                        process_by_worker_id=process_by_worker_id,
                        control_queues=control_queues,
                        active_worker_ids=active_worker_ids,
                        worker_pressure_states=worker_pressure_states,
                        worker_pressure_command_times=worker_pressure_command_times,
                        previous_state=memory_pressure_state,
                        previous_targets=memory_pressure_targets,
                        force_snapshot=force_snapshot,
                    )
                )
                if force_snapshot:
                    last_parent_memory_snapshot = now
                _raise_if_parent_memory_limit_exceeded(execution_config, processes)
                last_parent_memory_check = now
            if message.kind == "done":
                active_worker_ids.discard(message.worker_id)
                worker_pressure_states[message.worker_id] = (
                    PARENT_MEMORY_PRESSURE_NORMAL
                )
                worker_progress = progress[message.worker_id]
                append_build_log(
                    build_path,
                    "phase=traversal "
                    f"worker={message.worker_id} "
                    f"done completed={worker_progress['completed']} "
                    f"failed={worker_progress['failed']}",
                )
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for queue in work_queues.values():
            try:
                queue.put(None)
            except Exception:
                pass
        for process in processes:
            process.join()
    return failed


def _dynamic_batches_remaining(
    ram_bins: list[list[TraversalWorkBatch]],
    post_stress_bins: list[list[TraversalWorkBatch]] | None,
) -> bool:
    return any(ram_bins) or (
        post_stress_bins is not None and any(post_stress_bins)
    )


def _split_dynamic_ram_bins(
    batches: list[TraversalWorkBatch],
    *,
    bin_count: int,
) -> list[list[TraversalWorkBatch]]:
    safe_bin_count = max(1, int(bin_count))
    if not batches:
        return [[] for _ in range(safe_bin_count)]
    ordered = sorted(
        batches,
        key=lambda batch: (int(batch.sort_star_count), int(batch.region_pix)),
    )
    return [
        [batch for batch in split_rows.tolist()]
        for split_rows in np.array_split(np.asarray(ordered, dtype=object), safe_bin_count)
    ]


def _split_dynamic_runtime_bins(
    batches: list[TraversalWorkBatch],
    *,
    bin_count: int,
) -> list[list[TraversalWorkBatch]]:
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
        [batch.estimated_seconds for batch in ordered],
        dtype=np.float64,
    )
    cumulative = np.cumsum(runtimes)
    total = float(cumulative[-1]) if len(cumulative) else 0.0
    bins: list[list[TraversalWorkBatch]] = []
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


def _ordered_dynamic_runtime_bin_indexes(
    bins: list[list[TraversalWorkBatch]],
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


def _dynamic_bins_have_stress(bins: list[list[TraversalWorkBatch]]) -> bool:
    return any(batch.is_stress for bin_rows in bins for batch in bin_rows)


def _dynamic_bins_have_non_stress(bins: list[list[TraversalWorkBatch]]) -> bool:
    return any(not batch.is_stress for bin_rows in bins for batch in bin_rows)


def _highest_dynamic_stress_bin_index(bins: list[list[TraversalWorkBatch]]) -> int:
    for bin_index in range(len(bins) - 1, -1, -1):
        if any(batch.is_stress for batch in bins[bin_index]):
            return bin_index
    return -1


def _dynamic_affinity_parent(*, region_level: int, region_pix: int) -> int:
    if int(region_level) <= 0:
        return int(region_pix)
    return int(
        get_parent_pixel(
            int(region_level),
            np.asarray([int(region_pix)], dtype=np.int64),
            int(region_level) - 1,
        )[0]
    )


def _last_dynamic_affinity_for_level(
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]],
    *,
    worker_id: int,
    region_level: int,
) -> tuple[int, int] | None:
    return last_affinity_by_worker.get((int(worker_id), int(region_level)))


def _worker_has_dynamic_stress_affinity(
    last_affinity_by_worker: dict[tuple[int, int], tuple[int, int]],
    *,
    worker_id: int,
    region_level: int,
    stress_star_threshold: float,
) -> bool:
    affinity = _last_dynamic_affinity_for_level(
        last_affinity_by_worker,
        worker_id=int(worker_id),
        region_level=int(region_level),
    )
    if affinity is None:
        return False
    _, star_count = affinity
    return float(star_count) > float(stress_star_threshold)


def _ordered_dynamic_batch_indexes(
    queue: list[TraversalWorkBatch],
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
            if _dynamic_affinity_parent(
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
    ranked: list[tuple[float, int]] = []
    for index in candidate_indexes:
        vector = np.asarray(queue[index].center_vector, dtype=np.float64)
        distances = np.sum((active - vector) ** 2, axis=1)
        ranked.append((float(np.min(distances)), index))
    ranked.sort(key=lambda item: (-item[0], int(queue[item[1]].region_pix)))
    return [index for _, index in ranked]


def _run_in_process_traversal_worker(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
    star_counts: np.ndarray | None = None,
    telemetry: _StateUpdateTelemetry | None = None,
    state_writer: _BufferedStateWriter | None = None,
    diagnostics_writer: _TraversalDiagnosticsWriter | None = None,
) -> bool:
    """Run one long-lived Traversal worker in the parent process."""

    failed = False
    progress: dict[int, dict[str, int]] = {
        plan.worker_id: {"completed": 0, "failed": 0}
    }
    append_build_log(
        build_path,
        "phase=traversal "
        f"worker={plan.worker_id} "
        f"regions={','.join(str(pix) for pix in plan.region_pixs)} "
        f"outer_pixels={len(plan.outer_pixs)} "
        f"estimated_star_count={plan.estimated_star_count} started",
    )
    for message in _iter_long_lived_traversal_worker_messages(
        context,
        plan,
        execution_config,
    ):
        _raise_if_parent_memory_limit_exceeded(execution_config)
        failed = _handle_regional_worker_message(
            build_path=build_path,
            state=state,
            message=message,
            star_counts=star_counts,
            progress=progress,
            telemetry=telemetry,
            state_writer=state_writer,
            diagnostics_writer=diagnostics_writer,
        ) or failed
        if message.kind == "done":
            worker_progress = progress[message.worker_id]
            append_build_log(
                build_path,
                "phase=traversal "
                f"worker={message.worker_id} "
                f"done completed={worker_progress['completed']} "
                f"failed={worker_progress['failed']}",
            )
    return failed


def _run_traversal_phase(
    build_path: Path,
    *,
    execution_config: TraversalExecutionConfig,
) -> tuple[bool, dict[str, int]]:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_TRAVERSAL:
        raise BuildError(f"Expected traversal phase, got {current_phase!r}")

    fields = phase_state_fields(current_phase)
    if fields is None:
        raise BuildError(f"Build phase {current_phase!r} is not outer-pixel-local")
    status_field, _, error_field = fields

    definition = load_persisted_build_definition(build_path)
    build_artifact_root(build_path, definition).mkdir(parents=True, exist_ok=True)
    roots = load_build_roots(build_path)
    context = TraversalTaskContext(
        build_path=build_path,
        definition=definition,
        roots=roots,
        runtime_config_path=load_runtime_config_path(build_path),
    )
    star_counts = _load_gaia_star_counts(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
    )

    state = load_state(build_path)
    state_telemetry = _StateUpdateTelemetry()
    state_writer = _BufferedStateWriter(
        build_path=build_path,
        state=state,
        telemetry=state_telemetry,
    )
    repaired = _repair_stale_running_rows(
        build_path,
        state,
        status_field=status_field,
        telemetry=state_telemetry,
        state_writer=state_writer,
    )
    state_writer.flush()
    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(
        build_path,
        "run start "
        f"phase={current_phase} "
        f"repaired_stale_running={repaired} "
        f"workers={execution_config.workers} "
        f"scheduler={execution_config.scheduler} "
        f"low_latitude_workers={execution_config.low_latitude_workers or 'auto'} "
        f"derived_region_level={execution_config.region_level} "
        f"gaia_cache_entries={execution_config.gaia_cache_entries} "
        f"gaia_cache_mb={execution_config.gaia_cache_mb} "
        f"parent_memory_limit_mb={execution_config.parent_memory_limit_mb} "
        f"telemetry={execution_config.telemetry}",
    )
    failed = False
    diagnostics_writer = (
        _TraversalDiagnosticsWriter(build_path)
        if execution_config.telemetry == "detailed"
        else None
    )

    if execution_config.workers == 1:
        try:
            plans = build_regional_worker_plans(
                state=state,
                outer_level=definition.outer_level,
                region_level=int(execution_config.region_level),
                workers=1,
                status_field=status_field,
                star_counts=star_counts,
                low_latitude_workers=execution_config.low_latitude_workers,
            )
            if plans:
                failed = _run_in_process_traversal_worker(
                    build_path=build_path,
                    context=context,
                    state=state,
                    plan=plans[0],
                    execution_config=execution_config,
                    star_counts=star_counts,
                    telemetry=state_telemetry,
                    state_writer=state_writer,
                    diagnostics_writer=diagnostics_writer,
                ) or failed
        except Exception as exc:
            if diagnostics_writer is not None:
                diagnostics_writer.close()
            reset_running = _reset_running_rows(
                build_path,
                state,
                status_field=status_field,
                error_field=error_field,
                telemetry=state_telemetry,
                state_writer=state_writer,
            )
            state_writer.flush()
            set_build_status(build_path, BUILD_STATUS_FAILED)
            append_build_log(build_path, f"phase=traversal infrastructure failed: {exc}")
            _append_state_update_telemetry(build_path, state_telemetry)
            append_build_log(
                build_path,
                "run complete "
                "phase=traversal "
                "status=failed "
                f"reset_running={reset_running}",
            )
            raise
    else:
        try:
            if execution_config.scheduler == "dynamic":
                schedule = build_dynamic_work_batches(
                    state=state,
                    outer_level=definition.outer_level,
                    workers=execution_config.workers,
                    status_field=status_field,
                    star_counts=star_counts,
                    memory_limit_mb=float(execution_config.parent_memory_limit_mb),
                    trim_fraction=PARENT_MEMORY_SOFT_FRACTION,
                    worker_ram_overhead_mb=_parent_gpu_driver_reserve_mb(
                        1,
                        execution_config,
                    ),
                    recover_no_winner_pixels=execution_config.recover_no_winner_pixels,
                )
                model_name = dynamic_model_name(
                    recover_no_winner_pixels=execution_config.recover_no_winner_pixels,
                )
                append_build_log(
                    build_path,
                    "phase=traversal "
                    "dynamic_schedule "
                    f"batches={len(schedule.batches)} "
                    f"model={model_name} "
                    f"stress_workers={schedule.stress_worker_count} "
                    f"stress_star_threshold={schedule.stress_star_threshold:.0f}",
                )
                failed = _run_dynamic_traversal_workers(
                    build_path=build_path,
                    context=context,
                    state=state,
                    schedule=schedule,
                    execution_config=execution_config,
                    star_counts=star_counts,
                    telemetry=state_telemetry,
                    state_writer=state_writer,
                    diagnostics_writer=diagnostics_writer,
                ) or failed
            else:
                plans = build_regional_worker_plans(
                    state=state,
                    outer_level=definition.outer_level,
                    region_level=int(execution_config.region_level),
                    workers=execution_config.workers,
                    status_field=status_field,
                    star_counts=star_counts,
                    low_latitude_workers=execution_config.low_latitude_workers,
                )
                failed = _run_regional_traversal_workers(
                    build_path=build_path,
                    context=context,
                    state=state,
                    plans=plans,
                    execution_config=execution_config,
                    star_counts=star_counts,
                    telemetry=state_telemetry,
                    state_writer=state_writer,
                    diagnostics_writer=diagnostics_writer,
                ) or failed
        except Exception as exc:
            if diagnostics_writer is not None:
                diagnostics_writer.close()
            reset_running = _reset_running_rows(
                build_path,
                state,
                status_field=status_field,
                error_field=error_field,
                telemetry=state_telemetry,
                state_writer=state_writer,
            )
            state_writer.flush()
            set_build_status(build_path, BUILD_STATUS_FAILED)
            append_build_log(build_path, f"phase=traversal infrastructure failed: {exc}")
            _append_state_update_telemetry(build_path, state_telemetry)
            append_build_log(
                build_path,
                "run complete "
                "phase=traversal "
                "status=failed "
                f"reset_running={reset_running}",
            )
            raise

    state_writer.flush()
    phase_counts = {
        "pending": int(np.count_nonzero(state[status_field] == WORK_STATUS_PENDING)),
        "running": int(np.count_nonzero(state[status_field] == WORK_STATUS_RUNNING)),
        "done": int(np.count_nonzero(state[status_field] == WORK_STATUS_DONE)),
        "failed": int(np.count_nonzero(state[status_field] == WORK_STATUS_FAILED)),
    }
    final_status = BUILD_STATUS_FAILED if failed or phase_counts["failed"] else BUILD_STATUS_RUNNING
    set_build_status(build_path, final_status)
    if diagnostics_writer is not None:
        diagnostics_writer.flush()
        diagnostics_writer.close()
    _append_state_update_telemetry(build_path, state_telemetry)
    append_build_log(
        build_path,
        "run complete "
        f"phase={current_phase} "
        f"status={final_status} "
        f"pending={phase_counts['pending']} "
        f"running={phase_counts['running']} "
        f"done={phase_counts['done']} "
        f"failed={phase_counts['failed']}",
    )
    return final_status != BUILD_STATUS_FAILED and phase_counts["pending"] == 0, phase_counts


def _run_aggregation_phase(build_path: Path) -> None:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_AGGREGATION:
        raise BuildError(f"Expected aggregation phase, got {current_phase!r}")

    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(build_path, "run start phase=aggregation")
    try:
        level_maps = build_maps(build_path)
    except Exception as exc:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        append_build_log(build_path, f"phase=aggregation failed: {exc}")
        append_build_log(build_path, "run complete phase=aggregation status=failed")
        raise
    definition = load_persisted_build_definition(build_path)
    if definition.survey_extent_overlays:
        set_build_status(build_path, BUILD_STATUS_RUNNING)
        status = "running"
    else:
        set_build_status(build_path, BUILD_STATUS_COMPLETED)
        status = "completed"
    append_build_log(
        build_path,
        "run complete "
        "phase=aggregation "
        f"status={status} "
        f"levels={','.join(str(level) for level in sorted(level_maps))}",
    )


def _run_augmentation_phase(build_path: Path) -> None:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_AUGMENTATION:
        raise BuildError(f"Expected augmentation phase, got {current_phase!r}")

    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(build_path, "run start phase=augmentation")
    try:
        level_layers = build_survey_extent_layers(build_path)
    except Exception as exc:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        append_build_log(build_path, f"phase=augmentation failed: {exc}")
        append_build_log(build_path, "run complete phase=augmentation status=failed")
        raise

    set_build_status(build_path, BUILD_STATUS_COMPLETED)
    append_build_log(
        build_path,
        "run complete "
        "phase=augmentation "
        "status=completed "
        f"levels={','.join(str(level) for level in sorted(level_layers))}",
    )


def run_build(
    build_path: Path,
    *,
    workers: int | None = None,
    scheduler: str | None = None,
    low_latitude_workers: int | None = None,
    gaia_cache_entries: int | None = None,
    gaia_cache_mb: int | None = None,
    parent_memory_limit_mb: int | None = None,
    telemetry: str | None = None,
    aosky_yaml: Path | None = None,
) -> Path:
    """Run all unfinished build work through the implemented phases."""

    if workers is not None and int(workers) < 1:
        raise BuildError(f"workers must be at least 1, got {workers}")
    if not build_path.is_dir():
        raise BuildError(f"Build path does not exist: {build_path}")

    current_phase = load_current_phase(build_path)
    if current_phase not in (
        BUILD_PHASE_TRAVERSAL,
        BUILD_PHASE_AGGREGATION,
        BUILD_PHASE_AUGMENTATION,
    ):
        raise BuildError(
            f"Runner implements only {BUILD_PHASE_TRAVERSAL!r} and "
            f"{BUILD_PHASE_AGGREGATION!r}, and {BUILD_PHASE_AUGMENTATION!r}, "
            f"but build phase is {current_phase!r}"
        )

    if current_phase == BUILD_PHASE_TRAVERSAL:
        definition = load_persisted_build_definition(build_path)
        execution_config = resolve_traversal_execution_config(
            outer_level=definition.outer_level,
            workers=workers,
            scheduler=scheduler,
            low_latitude_workers=low_latitude_workers,
            gaia_cache_entries=gaia_cache_entries,
            gaia_cache_mb=gaia_cache_mb,
            parent_memory_limit_mb=parent_memory_limit_mb,
            telemetry=telemetry,
            aosky_yaml=aosky_yaml,
        )
        traversal_complete, _ = _run_traversal_phase(
            build_path,
            execution_config=execution_config,
        )
        if not traversal_complete:
            return build_path
        set_current_phase(build_path, BUILD_PHASE_AGGREGATION)
        current_phase = BUILD_PHASE_AGGREGATION

    if current_phase == BUILD_PHASE_AGGREGATION:
        _run_aggregation_phase(build_path)
        definition = load_persisted_build_definition(build_path)
        if not definition.survey_extent_overlays:
            return build_path
        set_current_phase(build_path, BUILD_PHASE_AUGMENTATION)

    _run_augmentation_phase(build_path)
    return build_path


def run_build_outer_pixels(
    build_path: Path,
    outer_pixels: int | Iterable[int],
    *,
    force: bool = False,
    execution_config: TraversalExecutionConfig | None = None,
) -> Path:
    """Run Traversal for selected outer pixels without running aggregation.

    This helper exists for validation and targeted rebuild workflows that need
    fresh per-outer-pixel `outer.h5` artifacts for a small sky patch. It uses
    the same persisted build configuration, model snapshots, Gaia root, and dust
    root as normal Traversal execution, updates the central `build.h5` state,
    and leaves the build in the `traversal` phase. It does not schedule
    multi-worker execution and it never advances into aggregation.

    Args:
        build_path: Initialized build root whose current phase is `traversal`.
        outer_pixels: One outer-pixel id or an iterable of outer-pixel ids.
        force: Re-run selected pixels even when they are already marked done.
        execution_config: Optional runtime-only Traversal settings for the
            selected-pixel run.

    Returns:
        The input build path.

    Raises:
        BuildError: If the build is missing, is not in the traversal phase, or
            any selected outer pixel is invalid.
    """

    if not build_path.is_dir():
        raise BuildError(f"Build path does not exist: {build_path}")
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_TRAVERSAL:
        raise BuildError(f"Expected traversal phase, got {current_phase!r}")

    selected = _normalize_selected_outer_pixels(outer_pixels)
    state = load_state(build_path)
    _validate_selected_outer_pixels(selected, state)
    definition = load_persisted_build_definition(build_path)
    build_artifact_root(build_path, definition).mkdir(parents=True, exist_ok=True)
    context = TraversalTaskContext(
        build_path=build_path,
        definition=definition,
        roots=load_build_roots(build_path),
        runtime_config_path=load_runtime_config_path(build_path),
    )

    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(
        build_path,
        "run selected_outer_pixels start "
        f"phase={current_phase} "
        f"force={bool(force)} "
        f"outer_pixels={','.join(str(pixel) for pixel in selected)}",
    )
    failed = False
    skipped: list[int] = []
    for outer_pix in selected:
        if not force and int(state["traversal_status"][outer_pix]) == WORK_STATUS_DONE:
            skipped.append(outer_pix)
            continue
        _mark_outer_pixel_running(build_path, state, outer_pix)
        result = _run_outer_pixel_traversal_task(
            context,
            outer_pix,
            execution_config=execution_config,
        )
        _record_traversal_result(build_path, state, result)
        if result.success:
            append_build_log(build_path, f"selected_outer_pixels outer_pixel={outer_pix} done")
        else:
            failed = True
            append_build_log(
                build_path,
                f"selected_outer_pixels outer_pixel={outer_pix} failed: {result.error_message}",
            )

    final_status = BUILD_STATUS_FAILED if failed else BUILD_STATUS_RUNNING
    set_build_status(build_path, final_status)
    append_build_log(
        build_path,
        "run selected_outer_pixels complete "
        f"phase={current_phase} "
        f"status={final_status} "
        f"requested={len(selected)} "
        f"skipped_done={len(skipped)} "
        f"failed={int(failed)}",
    )
    return build_path


def restart_build(
    *,
    lineage_name: str,
    build_root: Path | None,
    aosky_yaml: Path | None = None,
    workers: int | None = None,
    scheduler: str | None = None,
    low_latitude_workers: int | None = None,
    gaia_cache_entries: int | None = None,
    gaia_cache_mb: int | None = None,
    parent_memory_limit_mb: int | None = None,
    telemetry: str | None = None,
) -> Path:
    """Restart the latest build in one lineage."""

    resolved_build_root = resolve_build_root_only(
        build_root=build_root,
        aosky_yaml=aosky_yaml,
    )
    build_path = latest_build_path(
        resolved_build_root,
        lineage_name=lineage_name,
    )
    return run_build(
        build_path,
        workers=workers,
        scheduler=scheduler,
        low_latitude_workers=low_latitude_workers,
        gaia_cache_entries=gaia_cache_entries,
        gaia_cache_mb=gaia_cache_mb,
        parent_memory_limit_mb=parent_memory_limit_mb,
        telemetry=telemetry,
        aosky_yaml=aosky_yaml,
    )


def _normalize_selected_outer_pixels(outer_pixels: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(outer_pixels, int):
        values = [outer_pixels]
    elif isinstance(outer_pixels, (str, bytes)):
        raise BuildError("outer_pixels must be an int or iterable of ints")
    else:
        try:
            values = [int(pixel) for pixel in outer_pixels]
        except TypeError as exc:
            raise BuildError("outer_pixels must be an int or iterable of ints") from exc
    selected = tuple(dict.fromkeys(values))
    if not selected:
        raise BuildError("outer_pixels must not be empty")
    return selected


def _validate_selected_outer_pixels(outer_pixels: tuple[int, ...], state: np.ndarray) -> None:
    max_pixel = len(state) - 1
    invalid = [pixel for pixel in outer_pixels if pixel < 0 or pixel > max_pixel]
    if invalid:
        raise BuildError(
            "outer_pixels must be between "
            f"0 and {max_pixel}, got {invalid}"
        )


def show_build(build_path: Path) -> str:
    """Return a summary string for one build."""

    inspection = inspect_build(build_path)
    lines = [
        f"build: {inspection.build_path}",
        f"status: {inspection.build_status}",
        f"stage: {inspection.current_stage}",
        f"lineage: {inspection.lineage_name} v{inspection.lineage_version}",
        (
            "gaia: "
            f"release={inspection.gaia_release} "
            f"outer_level={inspection.outer_level} "
            f"inner_level={inspection.inner_level} "
            f"max_data_level={inspection.max_data_level}"
        ),
        f"runtime config: {inspection.runtime_config_path}",
        f"gaia root: {inspection.gaia_root}",
        f"dust root: {inspection.dust_root}",
        f"model root: {inspection.model_root}",
        (
            "model manifest: "
            f"{_present_missing(inspection.model_manifest_exists)} "
            f"{inspection.model_manifest_path}"
        ),
    ]
    stage_counts = inspection.stage_counts
    if stage_counts is not None:
        lines.extend(
            [
                "stage work:",
                f"  pending={stage_counts['pending']}",
                f"  running={stage_counts['running']}",
                f"  done={stage_counts['done']}",
                f"  failed={stage_counts['failed']}",
            ]
        )
    if inspection.survey_overlay_names:
        lines.extend(
            [
                "survey overlays: " + ", ".join(inspection.survey_overlay_names),
                (
                    "survey manifest: "
                    f"{_present_missing(inspection.survey_manifest_exists)} "
                    f"{inspection.survey_manifest_path}"
                ),
            ]
        )
    if inspection.map_artifacts:
        lines.append("map artifacts:")
        for level, artifact in inspection.map_artifacts.items():
            lines.append(
                f"  hpx{level}: {_present_missing(bool(artifact['exists']))} "
                f"{artifact['path']}"
            )
    if inspection.problems:
        lines.append("problems:")
        lines.extend(f"  {problem}" for problem in inspection.problems)
    return "\n".join(lines)


def _present_missing(exists: bool) -> str:
    return "present" if exists else "missing"
