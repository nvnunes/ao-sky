"""Typed build-config, filesystem, and execution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..survey import SurveyExtentOverlaySpec

DEFAULT_PREDICTION_BATCH_SIZE = 25000
DEFAULT_BACKEND_BUCKETS = tuple(range(1000, DEFAULT_PREDICTION_BATCH_SIZE + 1, 1000))
DEFAULT_PARENT_GPU_DRIVER_RESERVE_MB = 1075.0


@dataclass(frozen=True, slots=True)
class BuildDefinition:
    """Normalized build identity derived from one merged build config."""

    lineage_name: str
    gaia_release: str
    outer_level: int
    inner_level: int
    max_data_level: int
    survey_extent_overlays: tuple[SurveyExtentOverlaySpec, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildPaths:
    """Resolved filesystem roots used by build commands."""

    gaia_root: Path
    build_root: Path
    dust_root: Path
    model_root: Path


@dataclass(frozen=True, slots=True)
class BuildInspection:
    """Read-only inspection summary for one persisted build root."""

    build_path: Path
    build_status: str
    current_stage: str
    stage_counts: dict[str, int] | None
    lineage_name: str
    lineage_version: int
    gaia_release: str
    outer_level: int
    inner_level: int
    max_data_level: int
    runtime_config_path: Path
    runtime_config_source_path: Path
    gaia_root: Path
    dust_root: Path
    model_root: Path
    model_manifest_path: Path
    model_manifest_exists: bool
    survey_manifest_path: Path
    survey_manifest_exists: bool
    survey_overlay_names: tuple[str, ...]
    map_artifacts: dict[int, dict[str, object]]
    problems: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraversalTaskContext:
    """Pickle-safe worker context for one Traversal run."""

    build_path: Path
    definition: BuildDefinition
    roots: BuildPaths
    runtime_config_path: Path


@dataclass(frozen=True, slots=True)
class TraversalTaskResult:
    """Serialized result for one worker-owned Traversal outer-pixel task."""

    outer_pix: int
    success: bool
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class TraversalExecutionConfig:
    """Runtime-only Traversal execution settings."""

    workers: int = 1
    scheduler: str = "static"
    low_latitude_workers: int | None = None
    gaia_cache_entries: int = 64
    gaia_cache_mb: int = 2048
    region_level: int = 0
    parent_memory_limit_mb: int = 0
    telemetry: str = "basic"
    prediction_device: str = "cpu"
    averaged_prediction_device: str = "cpu"
    prediction_batch_size: int = DEFAULT_PREDICTION_BATCH_SIZE
    resolved_backend_buckets: tuple[int, ...] = DEFAULT_BACKEND_BUCKETS
    averaged_backend_buckets: tuple[int, ...] = DEFAULT_BACKEND_BUCKETS
    resolved_cache_clear_every: int = -1
    parent_gpu_driver_reserve_mb: float = DEFAULT_PARENT_GPU_DRIVER_RESERVE_MB
    recover_no_winner_pixels: bool = True


@dataclass(frozen=True, slots=True)
class TraversalWorkerPlan:
    """Outer-pixel work assigned to one long-lived Traversal worker."""

    worker_id: int
    region_pixs: tuple[int, ...]
    outer_pixs: tuple[int, ...]
    estimated_star_count: int


@dataclass(frozen=True, slots=True)
class TraversalWorkBatch:
    """One dynamic Traversal work assignment."""

    region_level: int
    region_pix: int
    outer_pixs: tuple[int, ...]
    is_stress: bool
    sort_star_count: int
    estimated_seconds: float
    estimated_ram_mb: float
    center_vector: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DynamicTraversalSchedule:
    """Parent-owned dynamic Traversal work queue and policy metadata."""

    batches: tuple[TraversalWorkBatch, ...]
    stress_star_threshold: float
    stress_worker_count: int


@dataclass(frozen=True, slots=True)
class TraversalCacheStats:
    """Serialized worker-local Gaia cache counters."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversized_skips: int = 0
    current_bytes: int = 0
    peak_bytes: int = 0
    entries: int = 0
    load_seconds: float = 0.0
    raw_load_seconds: float = 0.0
    prepare_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TraversalStageStats:
    """Serialized Traversal pipeline timing counters."""

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


@dataclass(frozen=True, slots=True)
class TraversalStructureStats:
    """Serialized Traversal intermediate cardinality and byte counters."""

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
    point_prediction_backend_bucket_counts: tuple[tuple[int, int], ...] = ()
    point_prediction_backend_bucket_rows: tuple[tuple[int, int], ...] = ()
    point_feature_bytes_peak: int = 0
    point_mps_current_bytes_peak: int = 0
    point_mps_driver_bytes_peak: int = 0
    point_mps_recommended_bytes: int = 0
    field_mean_prediction_batches: int = 0
    field_mean_prediction_rows: int = 0
    field_mean_prediction_batch_rows_peak: int = 0
    field_mean_prediction_backend_rows: int = 0
    field_mean_prediction_backend_batch_rows_peak: int = 0
    field_mean_prediction_backend_bucket_counts: tuple[tuple[int, int], ...] = ()
    field_mean_prediction_backend_bucket_rows: tuple[tuple[int, int], ...] = ()
    field_mean_feature_bytes_peak: int = 0
    field_mean_mps_current_bytes_peak: int = 0
    field_mean_mps_driver_bytes_peak: int = 0
    field_mean_mps_recommended_bytes: int = 0
    search_star_rows_peak: int = 0
    ngs_rows_peak: int = 0
    close_pair_rows_peak: int = 0
    context_pair_rows_peak: int = 0
    raw_asterism_rows_peak: int = 0
    local_asterism_rows_peak: int = 0
    winner_payload_rows_peak: int = 0


@dataclass(frozen=True, slots=True)
class TraversalWorkerStats:
    """Serialized low-volume Traversal worker profiling counters."""

    completed: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    pixel_seconds: float = 0.0
    artifact_write_seconds: float = 0.0
    artifact_convert_inner_seconds: float = 0.0
    artifact_convert_asterisms_seconds: float = 0.0
    artifact_hdf5_open_seconds: float = 0.0
    artifact_hdf5_inner_seconds: float = 0.0
    artifact_hdf5_asterisms_seconds: float = 0.0
    artifact_hdf5_close_seconds: float = 0.0
    artifact_replace_seconds: float = 0.0
    artifact_bytes: int = 0
    artifact_inner_input_bytes: int = 0
    artifact_asterism_input_bytes: int = 0
    artifact_inner_structured_bytes: int = 0
    artifact_asterism_structured_bytes: int = 0
    artifact_inner_rows: int = 0
    artifact_asterism_rows: int = 0
    stage_stats: TraversalStageStats = field(default_factory=TraversalStageStats)
    structure_stats: TraversalStructureStats = field(default_factory=TraversalStructureStats)
    peak_rss_mb: float = 0.0
    cache_stats: TraversalCacheStats = field(default_factory=TraversalCacheStats)


@dataclass(frozen=True, slots=True)
class TraversalMemorySample:
    """One optional per-pixel diagnostic memory sample."""

    outer_pix: int
    worker_id: int
    success: bool
    pixel_seconds: float = 0.0
    artifact_write_seconds: float = 0.0
    artifact_rss_before_convert_inner_mb: float = 0.0
    artifact_peak_before_convert_inner_mb: float = 0.0
    artifact_rss_after_convert_inner_mb: float = 0.0
    artifact_peak_after_convert_inner_mb: float = 0.0
    artifact_rss_after_convert_asterisms_mb: float = 0.0
    artifact_peak_after_convert_asterisms_mb: float = 0.0
    artifact_rss_after_hdf5_open_mb: float = 0.0
    artifact_peak_after_hdf5_open_mb: float = 0.0
    artifact_rss_after_hdf5_inner_mb: float = 0.0
    artifact_peak_after_hdf5_inner_mb: float = 0.0
    artifact_rss_after_hdf5_asterisms_mb: float = 0.0
    artifact_peak_after_hdf5_asterisms_mb: float = 0.0
    artifact_rss_after_hdf5_close_mb: float = 0.0
    artifact_peak_after_hdf5_close_mb: float = 0.0
    artifact_rss_after_replace_mb: float = 0.0
    artifact_peak_after_replace_mb: float = 0.0
    rss_start_mb: float = 0.0
    peak_rss_start_mb: float = 0.0
    rss_after_star_selection_mb: float = 0.0
    peak_rss_after_star_selection_mb: float = 0.0
    rss_after_candidate_generation_mb: float = 0.0
    peak_rss_after_candidate_generation_mb: float = 0.0
    rss_after_filtering_mb: float = 0.0
    peak_rss_after_filtering_mb: float = 0.0
    rss_after_context_mb: float = 0.0
    peak_rss_after_context_mb: float = 0.0
    rss_after_point_prediction_mb: float = 0.0
    peak_rss_after_point_prediction_mb: float = 0.0
    rss_after_field_mean_mb: float = 0.0
    peak_rss_after_field_mean_mb: float = 0.0
    rss_after_coverage_mb: float = 0.0
    peak_rss_after_coverage_mb: float = 0.0
    rss_after_dust_mb: float = 0.0
    peak_rss_after_dust_mb: float = 0.0
    rss_after_persisted_asterisms_mb: float = 0.0
    peak_rss_after_persisted_asterisms_mb: float = 0.0
    rss_after_artifact_write_mb: float = 0.0
    peak_rss_after_artifact_write_mb: float = 0.0
    rss_after_gc_mb: float = 0.0
    peak_rss_after_gc_mb: float = 0.0
    search_star_rows: int = 0
    ngs_rows: int = 0
    close_pair_rows: int = 0
    raw_asterism_rows: int = 0
    candidate_graph_rows: int = 0
    local_asterism_rows: int = 0
    context_pair_rows: int = 0
    winner_payload_rows: int = 0
    point_prediction_batches: int = 0
    point_prediction_rows: int = 0
    point_prediction_batch_rows_peak: int = 0
    point_prediction_backend_rows: int = 0
    point_prediction_backend_batch_rows_peak: int = 0
    point_prediction_backend_bucket_counts: str = ""
    point_prediction_backend_bucket_rows: str = ""
    point_feature_mib_peak: float = 0.0
    point_mps_current_mib_peak: float = 0.0
    point_mps_driver_mib_peak: float = 0.0
    point_mps_recommended_mib: float = 0.0
    field_mean_prediction_batches: int = 0
    field_mean_prediction_rows: int = 0
    field_mean_prediction_batch_rows_peak: int = 0
    field_mean_prediction_backend_rows: int = 0
    field_mean_prediction_backend_batch_rows_peak: int = 0
    field_mean_prediction_backend_bucket_counts: str = ""
    field_mean_prediction_backend_bucket_rows: str = ""
    field_mean_feature_mib_peak: float = 0.0
    field_mean_mps_current_mib_peak: float = 0.0
    field_mean_mps_driver_mib_peak: float = 0.0
    field_mean_mps_recommended_mib: float = 0.0
    artifact_inner_structured_mib: float = 0.0
    artifact_asterism_structured_mib: float = 0.0
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class TraversalWorkerMemorySample:
    """One optional worker-lifecycle diagnostic memory sample."""

    worker_id: int
    event: str
    elapsed_seconds: float
    rss_mb: float
    peak_rss_mb: float


@dataclass(frozen=True, slots=True)
class TraversalWorkerMessage:
    """Message emitted by a long-lived Traversal worker."""

    worker_id: int
    kind: str
    outer_pix: int | None = None
    pixel_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    result: TraversalTaskResult | None = None
    cache_stats: TraversalCacheStats | None = None
    worker_stats: TraversalWorkerStats | None = None
    memory_sample: TraversalMemorySample | None = None
    worker_memory_sample: TraversalWorkerMemorySample | None = None
    error_message: str = ""
