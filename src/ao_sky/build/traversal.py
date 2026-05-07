"""Native Traversal pipeline for retained asterisms and rich inner products."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
import astropy_healpix
import numpy as np

from ..asterisms import load_asterism_stars
from ..dust import add_gaia_a0_to_inner
from ..gaia import GAIA_SCHEMA_COLUMNS, GaiaHealpixStore, compute_r_magnitude
from ..predict import (
    clear_backend_cache,
    get_mean_model,
    get_point_model,
    get_seeing_baseline_performance,
    predict_field_mean_arrays,
    predict_point_arrays,
)
from ..predict._models import PredictRuntime, SeeingBaselinePerformance
from ..predict.service import PredictionArrayTelemetry
from ..spatial import (
    get_parent_pixel,
    get_pixel_area,
    get_pixel_from_skycoord,
    get_pixel_resolution,
    get_pixel_skycoord,
    get_subpixels,
)
from ._exceptions import BuildError
from ._models import (
    DEFAULT_BACKEND_BUCKETS as _DEFAULT_BACKEND_BUCKETS,
    DEFAULT_PREDICTION_BATCH_SIZE as _DEFAULT_PREDICTION_BATCH_SIZE,
    TraversalExecutionConfig,
    TraversalStageStats,
    TraversalStructureStats,
)
from .runtime_gaia import RUNTIME_HPX_COLUMN, RUNTIME_HPX_LEVEL

ASTERISM_BOUNDARY_RINGS = 2
WINNER_REGULARIZATION_PASSES = 3
ASTERISM_CENTER_TOLERANCE_ARCSEC = 1e-6
DEFAULT_PREDICTION_BATCH_SIZE = _DEFAULT_PREDICTION_BATCH_SIZE
DEFAULT_BACKEND_BUCKETS = _DEFAULT_BACKEND_BUCKETS


def _stream_batch_size(
    prediction_batch_size: int,
    buckets: tuple[int, ...],
) -> int:
    if not buckets:
        return int(prediction_batch_size)
    return min(int(prediction_batch_size), int(buckets[-1]))


def _backend_row_count(row_count: int, buckets: tuple[int, ...]) -> int | None:
    if not buckets:
        return None
    normalized_row_count = int(row_count)
    for bucket in buckets:
        if normalized_row_count <= int(bucket):
            return int(bucket)
    raise BuildError("prediction row count exceeds the largest backend bucket")


def _backend_buffer_row_count(buckets: tuple[int, ...]) -> int | None:
    if not buckets:
        return None
    return int(buckets[-1])


def _format_bucket_stats(values: dict[int, int]) -> str:
    return ";".join(f"{bucket}:{values[bucket]}" for bucket in sorted(values))


def _merge_bucket_stats(target: dict[int, int], values: dict[int, int]) -> None:
    for bucket, count in values.items():
        target[int(bucket)] = target.get(int(bucket), 0) + int(count)


@dataclass(slots=True)
class _PredictionRowBuffer:
    """Fixed-size NumPy row buffer for homogeneous prediction batches."""

    pixel_idxs: np.ndarray
    candidate_ids: np.ndarray
    pointing_x: np.ndarray | None = None
    pointing_y: np.ndarray | None = None
    size: int = 0

    @classmethod
    def create(
        cls,
        capacity: int,
        *,
        with_pointing: bool = False,
    ) -> "_PredictionRowBuffer":
        resolved_capacity = max(1, int(capacity))
        return cls(
            pixel_idxs=np.empty((resolved_capacity,), dtype=np.int64),
            candidate_ids=np.empty((resolved_capacity,), dtype=np.int64),
            pointing_x=(
                np.empty((resolved_capacity,), dtype=np.float64)
                if with_pointing
                else None
            ),
            pointing_y=(
                np.empty((resolved_capacity,), dtype=np.float64)
                if with_pointing
                else None
            ),
        )

    def append(
        self,
        pixel_idxs: np.ndarray,
        candidate_id: int,
        *,
        pointing_x: np.ndarray | None = None,
        pointing_y: np.ndarray | None = None,
    ) -> None:
        rows = int(len(pixel_idxs))
        if rows == 0:
            return
        self._ensure_capacity(self.size + rows)
        end = self.size + rows
        self.pixel_idxs[self.size : end] = np.asarray(pixel_idxs, dtype=np.int64)
        self.candidate_ids[self.size : end] = int(candidate_id)
        if self.pointing_x is not None:
            if pointing_x is None or pointing_y is None:
                raise BuildError("pointing buffer append requires pointing coordinates")
            self.pointing_x[self.size : end] = np.asarray(pointing_x, dtype=np.float64)
            self.pointing_y[self.size : end] = np.asarray(pointing_y, dtype=np.float64)
        self.size = end

    def arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        return (
            self.pixel_idxs[: self.size],
            self.candidate_ids[: self.size],
            None if self.pointing_x is None else self.pointing_x[: self.size],
            None if self.pointing_y is None else self.pointing_y[: self.size],
        )

    def head(
        self,
        rows: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        resolved_rows = min(max(0, int(rows)), self.size)
        return (
            self.pixel_idxs[:resolved_rows],
            self.candidate_ids[:resolved_rows],
            None if self.pointing_x is None else self.pointing_x[:resolved_rows],
            None if self.pointing_y is None else self.pointing_y[:resolved_rows],
        )

    def discard(self, rows: int) -> None:
        resolved_rows = min(max(0, int(rows)), self.size)
        if resolved_rows == 0:
            return
        remaining = self.size - resolved_rows
        if remaining:
            self.pixel_idxs[:remaining] = self.pixel_idxs[resolved_rows : self.size]
            self.candidate_ids[:remaining] = self.candidate_ids[
                resolved_rows : self.size
            ]
            if self.pointing_x is not None:
                self.pointing_x[:remaining] = self.pointing_x[
                    resolved_rows : self.size
                ]
                self.pointing_y[:remaining] = self.pointing_y[
                    resolved_rows : self.size
                ]
        self.size = remaining

    def clear(self) -> None:
        self.size = 0

    def _ensure_capacity(self, required: int) -> None:
        if required <= len(self.pixel_idxs):
            return
        new_capacity = max(int(required), len(self.pixel_idxs) * 2)
        new_pixels = np.empty((new_capacity,), dtype=np.int64)
        new_candidates = np.empty((new_capacity,), dtype=np.int64)
        new_pixels[: self.size] = self.pixel_idxs[: self.size]
        new_candidates[: self.size] = self.candidate_ids[: self.size]
        self.pixel_idxs = new_pixels
        self.candidate_ids = new_candidates
        if self.pointing_x is not None:
            new_pointing_x = np.empty((new_capacity,), dtype=np.float64)
            new_pointing_y = np.empty((new_capacity,), dtype=np.float64)
            new_pointing_x[: self.size] = self.pointing_x[: self.size]
            new_pointing_y[: self.size] = self.pointing_y[: self.size]
            self.pointing_x = new_pointing_x
            self.pointing_y = new_pointing_y


@dataclass(slots=True)
class TraversalStageProfile:
    """Mutable timing profile for one outer-pixel Traversal pipeline."""

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


@dataclass(slots=True)
class TraversalStructureProfile:
    """Mutable cardinality profile for one outer-pixel Traversal pipeline."""

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
            search_star_rows_peak=self.search_star_rows_peak or self.search_star_rows,
            ngs_rows_peak=self.ngs_rows_peak or self.ngs_rows,
            close_pair_rows_peak=self.close_pair_rows_peak or self.close_pair_rows,
            context_pair_rows_peak=self.context_pair_rows_peak or self.context_pair_rows,
            raw_asterism_rows_peak=self.raw_asterism_rows_peak or self.raw_asterism_rows,
            local_asterism_rows_peak=(
                self.local_asterism_rows_peak or self.local_asterism_rows
            ),
            winner_payload_rows_peak=(
                self.winner_payload_rows_peak or self.winner_payload_rows
            ),
        )


@dataclass(slots=True)
class TraversalMemoryProfile:
    """Mutable per-pixel RSS checkpoints for detailed diagnostics."""

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


@dataclass(frozen=True, slots=True)
class _CandidateGraph:
    """Geometry-constrained candidate graph for one selected NGS table."""

    sorted_ngs: Table
    final_count: int
    candidate_count: int
    edges: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True, slots=True)
class _CandidateSet:
    """Exact candidate identities retained for streaming prediction."""

    members: np.ndarray
    star_counts: np.ndarray
    source_keys: np.ndarray


@dataclass(frozen=True, slots=True)
class _StarForCoverage:
    """Star-to-inner-pixel FOR coverage in compact CSR form."""

    pixel_indices: np.ndarray
    starts: np.ndarray
    full_depth: np.ndarray
    target_depth: np.ndarray


@dataclass(frozen=True, slots=True)
class _MulticoverSelection:
    """Selected NGS subset and resulting FOR depth map."""

    star_indices: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True, slots=True)
class _CandidateCount:
    """Geometry-constrained candidate identity counts for one NGS set."""

    total: int
    singles: int
    pairs: int
    triples: int


@dataclass(frozen=True, slots=True)
class _RegionalCandidateStats:
    """Internal telemetry for regional candidate graph construction."""

    exact_regions: int
    for_optimized_regions: int
    split_regions: int
    max_depth: int
    max_regional_combination_work: int
    exact_combination_work: int
    final_resolved_inferences: int
    incomplete_for_regions: int


@dataclass(frozen=True, slots=True)
class _CandidateCenter:
    """Minimum circular FOV center for one 1-3 star candidate."""

    ra_deg: float
    dec_deg: float
    radius_arcsec: float


def _sample_memory(
    memory_profile: TraversalMemoryProfile | None,
    current_attr: str,
    peak_attr: str,
    *,
    rss_sampler=None,
    peak_sampler=None,
) -> None:
    if memory_profile is None or rss_sampler is None or peak_sampler is None:
        return
    setattr(memory_profile, current_attr, float(rss_sampler()))
    setattr(memory_profile, peak_attr, float(peak_sampler()))


def _new_prediction_telemetry(
    profile: TraversalStageProfile | None,
    structure_profile: TraversalStructureProfile | None,
) -> PredictionArrayTelemetry | None:
    if profile is None and structure_profile is None:
        return None
    return PredictionArrayTelemetry()


def _record_prediction_telemetry(
    telemetry: PredictionArrayTelemetry | None,
    *,
    resolved: bool,
    profile: TraversalStageProfile | None,
    structure_profile: TraversalStructureProfile | None,
) -> None:
    if telemetry is None:
        return
    if profile is not None:
        if resolved:
            profile.point_prediction_feature_seconds += telemetry.feature_seconds
            profile.point_prediction_backend_seconds += telemetry.backend_seconds
        else:
            profile.field_mean_prediction_feature_seconds += telemetry.feature_seconds
            profile.field_mean_prediction_backend_seconds += telemetry.backend_seconds
    if structure_profile is not None:
        if resolved:
            structure_profile.point_prediction_batches += int(telemetry.batches)
            structure_profile.point_prediction_rows += int(telemetry.rows)
            structure_profile.point_prediction_batch_rows_peak = max(
                structure_profile.point_prediction_batch_rows_peak,
                int(telemetry.batch_rows_peak),
            )
            structure_profile.point_prediction_backend_rows += int(
                telemetry.backend_rows
            )
            structure_profile.point_prediction_backend_batch_rows_peak = max(
                structure_profile.point_prediction_backend_batch_rows_peak,
                int(telemetry.backend_batch_rows_peak),
            )
            _merge_bucket_stats(
                structure_profile.point_backend_bucket_counts,
                telemetry.backend_bucket_counts,
            )
            _merge_bucket_stats(
                structure_profile.point_backend_bucket_rows,
                telemetry.backend_bucket_rows,
            )
            structure_profile.point_feature_bytes_peak = max(
                structure_profile.point_feature_bytes_peak,
                int(telemetry.feature_bytes_peak),
            )
            structure_profile.point_mps_current_bytes_peak = max(
                structure_profile.point_mps_current_bytes_peak,
                int(telemetry.mps_current_bytes_peak),
            )
            structure_profile.point_mps_driver_bytes_peak = max(
                structure_profile.point_mps_driver_bytes_peak,
                int(telemetry.mps_driver_bytes_peak),
            )
            structure_profile.point_mps_recommended_bytes = max(
                structure_profile.point_mps_recommended_bytes,
                int(telemetry.mps_recommended_bytes),
            )
        else:
            structure_profile.field_mean_prediction_batches += int(telemetry.batches)
            structure_profile.field_mean_prediction_rows += int(telemetry.rows)
            structure_profile.field_mean_prediction_batch_rows_peak = max(
                structure_profile.field_mean_prediction_batch_rows_peak,
                int(telemetry.batch_rows_peak),
            )
            structure_profile.field_mean_prediction_backend_rows += int(
                telemetry.backend_rows
            )
            structure_profile.field_mean_prediction_backend_batch_rows_peak = max(
                structure_profile.field_mean_prediction_backend_batch_rows_peak,
                int(telemetry.backend_batch_rows_peak),
            )
            _merge_bucket_stats(
                structure_profile.field_mean_backend_bucket_counts,
                telemetry.backend_bucket_counts,
            )
            _merge_bucket_stats(
                structure_profile.field_mean_backend_bucket_rows,
                telemetry.backend_bucket_rows,
            )
            structure_profile.field_mean_feature_bytes_peak = max(
                structure_profile.field_mean_feature_bytes_peak,
                int(telemetry.feature_bytes_peak),
            )
            structure_profile.field_mean_mps_current_bytes_peak = max(
                structure_profile.field_mean_mps_current_bytes_peak,
                int(telemetry.mps_current_bytes_peak),
            )
            structure_profile.field_mean_mps_driver_bytes_peak = max(
                structure_profile.field_mean_mps_driver_bytes_peak,
                int(telemetry.mps_driver_bytes_peak),
            )
            structure_profile.field_mean_mps_recommended_bytes = max(
                structure_profile.field_mean_mps_recommended_bytes,
                int(telemetry.mps_recommended_bytes),
            )


def _clear_backend_cache_with_profile(profile: TraversalStageProfile | None) -> None:
    started = time.perf_counter()
    clear_backend_cache()
    if profile is not None:
        profile.point_prediction_cache_clear_seconds += time.perf_counter() - started


@dataclass(slots=True)
class TraversalGeometry:
    """Worker-local geometry values reused while building outer pixels."""

    outer_level: int
    inner_level: int
    fov_level: int
    fov_level_area_arcmin2: float
    inner_resolution: u.Quantity
    baseline: SeeingBaselinePerformance

    @classmethod
    def from_runtime(cls, runtime: PredictRuntime) -> TraversalGeometry:
        """Build static traversal geometry from the native runtime contract."""

        fov_level = _get_level_with_resolution(runtime.ao_system.fov)
        return cls(
            outer_level=runtime.outer_level,
            inner_level=runtime.inner_level,
            fov_level=fov_level,
            fov_level_area_arcmin2=get_pixel_area(fov_level).to(u.arcmin**2).value,
            inner_resolution=get_pixel_resolution(runtime.inner_level),
            baseline=get_seeing_baseline_performance(runtime),
        )

    def inner_pixs(self, outer_pix: int) -> np.ndarray:
        return get_subpixels(
            self.outer_level,
            int(outer_pix),
            self.inner_level,
        )

    def inner_centres(self, outer_pix: int) -> SkyCoord:
        return get_pixel_skycoord(
            self.inner_level,
            self.inner_pixs(outer_pix),
        )


def _require_runtime_gaia_columns(table: Table) -> None:
    missing = [name for name in ("R", RUNTIME_HPX_COLUMN) if name not in table.colnames]
    if missing:
        raise BuildError(
            "Native Traversal requires runtime Gaia rows with columns: "
            + ", ".join(missing)
        )


def _get_runtime_table_pixels(table: Table, level: int) -> np.ndarray:
    _require_runtime_gaia_columns(table)
    if level > RUNTIME_HPX_LEVEL:
        raise BuildError(
            f"Runtime Gaia {RUNTIME_HPX_COLUMN} cannot derive finer level {level}"
        )
    hpx = np.asarray(table[RUNTIME_HPX_COLUMN], dtype=np.int64)
    pixels = np.full(hpx.shape, -1, dtype=np.int64)
    valid = hpx >= 0
    if np.any(valid):
        pixels[valid] = np.asarray(
            get_parent_pixel(RUNTIME_HPX_LEVEL, hpx[valid], level),
            dtype=np.int64,
        )
    return pixels


def _valid_runtime_gaia_mask(table: Table) -> np.ndarray:
    _require_runtime_gaia_columns(table)
    return (
        np.isfinite(np.asarray(table["ra"], dtype=np.float64))
        & np.isfinite(np.asarray(table["dec"], dtype=np.float64))
        & np.isfinite(np.asarray(table["R"], dtype=np.float64))
        & (np.asarray(table[RUNTIME_HPX_COLUMN], dtype=np.int64) >= 0)
    )


def _get_level_with_resolution(target_resolution: u.Quantity) -> int:
    target = target_resolution.to(u.arcmin)
    for level in range(0, 30):
        if get_pixel_resolution(level).to(u.arcmin) <= target:
            return level
    raise BuildError(f"Could not find HEALPix level for resolution {target_resolution}")


def _regional_selector_max_depth(runtime: PredictRuntime) -> int:
    target = runtime.ao_system.fov.to(u.arcsec)
    floor_level = int(runtime.outer_level)
    for level in range(int(runtime.outer_level), int(runtime.inner_level) + 1):
        if get_pixel_resolution(level).to(u.arcsec) >= target:
            floor_level = level
        else:
            break
    return max(0, int(floor_level) - int(runtime.outer_level))


def _max_regional_combination_work(runtime: PredictRuntime) -> int:
    level_delta = int(runtime.inner_level) - int(runtime.outer_level)
    if level_delta < 0:
        raise BuildError("inner_level must be greater than or equal to outer_level")
    return int(4**level_delta)


def _regional_sparse_graph_probe_star_limit(runtime: PredictRuntime) -> int:
    return max(
        int(runtime.ao_system.max_wfs),
        int(math.sqrt(_max_regional_combination_work(runtime))) * 8,
    )


def _get_inner_count_band_values(table: Table, band: str) -> np.ndarray:
    if band in table.colnames:
        return np.asarray(table[band], dtype=np.float64)
    if band == "R":
        return np.asarray(
            compute_r_magnitude(table[list(GAIA_SCHEMA_COLUMNS)]),
            dtype=np.float64,
        )
    raise BuildError(f"Unsupported build band {band!r}")


def _get_inner_count_pixels(table: Table, level: int) -> np.ndarray:
    if RUNTIME_HPX_COLUMN in table.colnames and level <= RUNTIME_HPX_LEVEL:
        return _get_runtime_table_pixels(table, level)

    pixels = np.full((len(table),), -1, dtype=np.int64)
    valid = np.isfinite(np.asarray(table["ra"], dtype=np.float64)) & np.isfinite(
        np.asarray(table["dec"], dtype=np.float64)
    )
    if np.any(valid):
        pixels[valid] = np.asarray(
            get_pixel_from_skycoord(
                level,
                SkyCoord(
                    ra=table["ra"][valid],
                    dec=table["dec"][valid],
                    unit=(u.degree, u.degree),
                ),
            ),
            dtype=np.int64,
        )
    return pixels


def _load_inner_count_stars(store: GaiaHealpixStore, outer_pix: int) -> Table:
    return store.load_healpix(outer_pix, read_only=True)


def prepare_search_inputs(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
    boundary_rings: int = ASTERISM_BOUNDARY_RINGS,
) -> tuple[Table, Table]:
    """Return the prepared search-star and NGS tables for one outer pixel."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=geometry.fov_level,
        boundary_rings=boundary_rings,
        include_locality=True,
    )
    _require_runtime_gaia_columns(stars)
    stars = stars[_valid_runtime_gaia_mask(stars)]
    stars["pix"] = _get_runtime_table_pixels(stars, geometry.fov_level)

    ngs = stars[
        (stars["R"] >= runtime.ao_system.min_mag) & (stars["R"] < runtime.ao_system.max_mag)
    ]

    return stars, ngs


def _empty_persisted_asterisms() -> Table:
    return Table(
        [
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        ],
        names=(
            "asterism_id",
            "ra",
            "dec",
            "num_stars",
            "pix",
            "star1_source_id",
            "star1_ra",
            "star1_dec",
            "star1_mag",
            "star2_source_id",
            "star2_ra",
            "star2_dec",
            "star2_mag",
            "star3_source_id",
            "star3_ra",
            "star3_dec",
            "star3_mag",
        ),
    )


def _to_persisted_asterisms(asterisms: Table | None) -> Table:
    if asterisms is None or len(asterisms) == 0:
        return _empty_persisted_asterisms()

    result = Table()
    result["asterism_id"] = np.asarray(asterisms["asterism_id"], dtype=np.int64)
    result["ra"] = np.asarray(asterisms["ra"], dtype=np.float64)
    result["dec"] = np.asarray(asterisms["dec"], dtype=np.float64)
    result["num_stars"] = np.asarray(asterisms["num_stars"], dtype=np.int64)
    result["pix"] = np.asarray(asterisms["pix"], dtype=np.int64)
    for index in (1, 2, 3):
        result[f"star{index}_source_id"] = np.asarray(asterisms[f"star{index}_source_id"], dtype=np.int64)
        result[f"star{index}_ra"] = np.asarray(asterisms[f"star{index}_ra"], dtype=np.float64)
        result[f"star{index}_dec"] = np.asarray(asterisms[f"star{index}_dec"], dtype=np.float64)
        result[f"star{index}_mag"] = np.asarray(asterisms[f"star{index}_mag"], dtype=np.float64)
    return result


def build_base_inner_table(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    geometry: TraversalGeometry | None = None,
    asterisms: Table | None = None,
) -> Table:
    """Return the dense base inner table for one outer pixel."""

    geometry = geometry or TraversalGeometry.from_runtime(runtime)
    local_stars = _load_inner_count_stars(store, outer_pix)
    pixs = geometry.inner_pixs(outer_pix)

    gaia_pixs = _get_inner_count_pixels(local_stars, runtime.inner_level)
    valid_gaia_pixs = np.asarray(gaia_pixs, dtype=np.int64) >= 0
    unique_pixs, counts = np.unique(
        np.asarray(gaia_pixs, dtype=np.int64)[valid_gaia_pixs],
        return_counts=True,
    )
    count_map = dict(zip(unique_pixs, counts, strict=False))

    band_values = _get_inner_count_band_values(local_stars, runtime.ao_system.band)
    ngs_mask = valid_gaia_pixs & np.isfinite(band_values)
    ngs_mask &= band_values >= runtime.ao_system.min_mag
    ngs_mask &= band_values < runtime.ao_system.max_mag
    if np.any(ngs_mask):
        ngs_unique, ngs_counts = np.unique(
            np.asarray(gaia_pixs, dtype=np.int64)[ngs_mask],
            return_counts=True,
        )
        ngs_count_map = dict(zip(ngs_unique, ngs_counts, strict=False))
    else:
        ngs_count_map = {}

    baseline = geometry.baseline
    size = len(pixs)
    return Table(
        [
            np.asarray(pixs, dtype=np.int64),
            np.asarray([count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.asarray([ngs_count_map.get(int(pix), 0) for pix in pixs], dtype=np.int64),
            np.full((size,), np.nan if baseline.ee is None else baseline.ee, dtype=np.float64),
            np.full((size,), np.nan if baseline.sr is None else baseline.sr, dtype=np.float64),
            np.full((size,), np.nan if baseline.fwhm is None else baseline.fwhm, dtype=np.float64),
            np.full((size,), -1, dtype=np.int64),
            np.full((size,), np.nan if baseline.ee is None else baseline.ee, dtype=np.float64),
            np.full((size,), np.nan if baseline.ee is None else baseline.ee, dtype=np.float64),
            np.zeros((size,), dtype=np.bool_),
            np.zeros((size,), dtype=np.bool_),
        ],
        names=(
            "pix",
            "star_count",
            "ngs_count",
            "best_ee",
            "best_sr",
            "best_fwhm",
            "winner_asterism_id",
            "winner_ee_resolved",
            "winner_ee_averaged",
            "coverage_resolved",
            "coverage_averaged",
        ),
    )


def _get_plane_offsets_arcsec(reference_coord: SkyCoord, skycoords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    lon_offset, lat_offset = reference_coord.spherical_offsets_to(skycoords)
    return lon_offset.to(u.arcsec).value, lat_offset.to(u.arcsec).value


def _candidate_enclosing_fov_center(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> _CandidateCenter:
    """Return the minimum circular FOV center for 1-3 tangent-plane points."""

    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    dec_deg = np.asarray(dec_deg, dtype=np.float64)
    count = int(len(ra_deg))
    if count < 1 or count > 3:
        raise BuildError("asterism center requires 1-3 member positions")

    ra_rad = np.unwrap(np.deg2rad(ra_deg))
    dec_rad = np.deg2rad(dec_deg)
    ra0 = float(np.mean(ra_rad))
    dec0 = float(np.mean(dec_rad))
    cos_dec0 = float(np.cos(dec0))
    if abs(cos_dec0) < 1e-12:
        raise BuildError("asterism center is undefined at the celestial pole")

    points = np.column_stack(((ra_rad - ra0) * cos_dec0, dec_rad - dec0))

    if count == 1:
        center_xy = points[0]
    elif count == 2:
        center_xy = np.mean(points[:2], axis=0)
    else:
        pairs = ((0, 1), (1, 2), (2, 0))
        side_lengths_sq = np.asarray(
            [
                float(np.sum((points[first] - points[second]) ** 2))
                for first, second in pairs
            ],
            dtype=np.float64,
        )
        longest = int(np.argmax(side_lengths_sq))
        other_lengths_sq = float(np.sum(side_lengths_sq) - side_lengths_sq[longest])
        if side_lengths_sq[longest] >= other_lengths_sq:
            first, second = pairs[longest]
            center_xy = (points[first] + points[second]) / 2.0
        else:
            (ax, ay), (bx, by), (cx, cy) = points
            determinant = 2.0 * (
                ax * (by - cy)
                + bx * (cy - ay)
                + cx * (ay - by)
            )
            if abs(determinant) < 1e-18:
                first, second = pairs[longest]
                center_xy = (points[first] + points[second]) / 2.0
            else:
                ux = (
                    (ax * ax + ay * ay) * (by - cy)
                    + (bx * bx + by * by) * (cy - ay)
                    + (cx * cx + cy * cy) * (ay - by)
                ) / determinant
                uy = (
                    (ax * ax + ay * ay) * (cx - bx)
                    + (bx * bx + by * by) * (ax - cx)
                    + (cx * cx + cy * cy) * (bx - ax)
                ) / determinant
                center_xy = np.asarray((ux, uy), dtype=np.float64)

    radius_rad = float(np.max(np.linalg.norm(points - center_xy, axis=1)))
    center_ra = float(np.rad2deg(center_xy[0] / cos_dec0 + ra0) % 360.0)
    if center_ra >= 360.0 - 1e-12:
        center_ra = 0.0
    center_dec = float(np.rad2deg(center_xy[1] + dec0))
    return _CandidateCenter(
        ra_deg=center_ra,
        dec_deg=center_dec,
        radius_arcsec=float(np.rad2deg(radius_rad) * 3600.0),
    )


def _coerce_recovery_projection_inputs(
    target_x: np.ndarray,
    target_y: np.ndarray,
    disk_x: np.ndarray,
    disk_y: np.ndarray,
    radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, float]:
    target_x = np.asarray(target_x, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)
    disk_x = np.asarray(disk_x, dtype=np.float64)
    disk_y = np.asarray(disk_y, dtype=np.float64)
    if disk_x.ndim != 2 or disk_y.shape != disk_x.shape:
        raise BuildError("recovery disk coordinates must have shape (rows, stars)")
    if disk_x.shape[1] < 1 or disk_x.shape[1] > 3:
        raise BuildError("recovery projection requires 1-3 guide-star disks")
    if len(target_x) != len(target_y) or len(target_x) != disk_x.shape[0]:
        raise BuildError("recovery target and disk row counts do not match")

    row_count, disk_count = disk_x.shape
    radius = float(radius_arcsec)
    allowed_radius_sq = (radius + ASTERISM_CENTER_TOLERANCE_ARCSEC) ** 2
    return (
        target_x,
        target_y,
        disk_x,
        disk_y,
        row_count,
        disk_count,
        radius,
        allowed_radius_sq,
    )


def _nearest_feasible_pointing_one_star(
    target_x: np.ndarray,
    target_y: np.ndarray,
    disk_x: np.ndarray,
    disk_y: np.ndarray,
    radius: float,
    allowed_radius_sq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_x = disk_x[:, 0]
    center_y = disk_y[:, 0]
    dx = target_x - center_x
    dy = target_y - center_y
    distance = np.hypot(dx, dy)
    inside = distance * distance <= allowed_radius_sq
    valid_direction = distance > 0.0
    safe_distance = np.where(valid_direction, distance, 1.0)
    projected_x = center_x + radius * dx / safe_distance
    projected_y = center_y + radius * dy / safe_distance
    pointing_x = np.where(inside, target_x, projected_x)
    pointing_y = np.where(inside, target_y, projected_y)
    return pointing_x, pointing_y, np.ones((len(target_x),), dtype=np.bool_)


def _nearest_feasible_pointing_two_star(
    target_x: np.ndarray,
    target_y: np.ndarray,
    disk_x: np.ndarray,
    disk_y: np.ndarray,
    radius: float,
    allowed_radius_sq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = len(target_x)
    best_x = np.full((row_count,), np.nan, dtype=np.float64)
    best_y = np.full((row_count,), np.nan, dtype=np.float64)
    best_dist_sq = np.full((row_count,), np.inf, dtype=np.float64)

    def consider(qx: np.ndarray, qy: np.ndarray, valid: np.ndarray) -> None:
        nonlocal best_x, best_y, best_dist_sq
        first_dist_sq = (qx - disk_x[:, 0]) ** 2 + (qy - disk_y[:, 0]) ** 2
        second_dist_sq = (qx - disk_x[:, 1]) ** 2 + (qy - disk_y[:, 1]) ** 2
        inside = (
            np.asarray(valid, dtype=np.bool_)
            & (first_dist_sq <= allowed_radius_sq)
            & (second_dist_sq <= allowed_radius_sq)
        )
        dist_to_target_sq = (qx - target_x) ** 2 + (qy - target_y) ** 2
        update = inside & (dist_to_target_sq < best_dist_sq)
        best_x[update] = qx[update]
        best_y[update] = qy[update]
        best_dist_sq[update] = dist_to_target_sq[update]

    consider(target_x, target_y, np.ones((row_count,), dtype=np.bool_))

    for disk_index in range(2):
        dx = target_x - disk_x[:, disk_index]
        dy = target_y - disk_y[:, disk_index]
        distance = np.hypot(dx, dy)
        valid = distance > 0.0
        safe_distance = np.where(valid, distance, 1.0)
        qx = disk_x[:, disk_index] + radius * dx / safe_distance
        qy = disk_y[:, disk_index] + radius * dy / safe_distance
        consider(qx, qy, valid)

    dx = disk_x[:, 1] - disk_x[:, 0]
    dy = disk_y[:, 1] - disk_y[:, 0]
    separation = np.hypot(dx, dy)
    valid = (separation > 0.0) & (separation <= 2.0 * radius)
    safe_separation = np.where(valid, separation, 1.0)
    midpoint_x = (disk_x[:, 0] + disk_x[:, 1]) / 2.0
    midpoint_y = (disk_y[:, 0] + disk_y[:, 1]) / 2.0
    half_chord = np.sqrt(
        np.maximum(0.0, radius * radius - (safe_separation / 2.0) ** 2)
    )
    perp_x = -dy / safe_separation
    perp_y = dx / safe_separation
    consider(
        midpoint_x + half_chord * perp_x,
        midpoint_y + half_chord * perp_y,
        valid,
    )
    consider(
        midpoint_x - half_chord * perp_x,
        midpoint_y - half_chord * perp_y,
        valid,
    )

    return best_x, best_y, np.isfinite(best_dist_sq)


def _nearest_feasible_pointing_generic(
    target_x: np.ndarray,
    target_y: np.ndarray,
    disk_x: np.ndarray,
    disk_y: np.ndarray,
    radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        target_x,
        target_y,
        disk_x,
        disk_y,
        row_count,
        disk_count,
        radius,
        allowed_radius_sq,
    ) = _coerce_recovery_projection_inputs(
        target_x,
        target_y,
        disk_x,
        disk_y,
        radius_arcsec,
    )
    best_x = np.full((row_count,), np.nan, dtype=np.float64)
    best_y = np.full((row_count,), np.nan, dtype=np.float64)
    best_dist_sq = np.full((row_count,), np.inf, dtype=np.float64)

    def consider(qx: np.ndarray, qy: np.ndarray, valid: np.ndarray) -> None:
        nonlocal best_x, best_y, best_dist_sq
        dist_to_stars_sq = (qx[:, None] - disk_x) ** 2 + (qy[:, None] - disk_y) ** 2
        inside = np.asarray(valid, dtype=np.bool_) & np.all(
            dist_to_stars_sq <= allowed_radius_sq,
            axis=1,
        )
        dist_to_target_sq = (qx - target_x) ** 2 + (qy - target_y) ** 2
        update = inside & (dist_to_target_sq < best_dist_sq)
        best_x[update] = qx[update]
        best_y[update] = qy[update]
        best_dist_sq[update] = dist_to_target_sq[update]

    consider(target_x, target_y, np.ones((row_count,), dtype=np.bool_))

    for disk_index in range(disk_count):
        dx = target_x - disk_x[:, disk_index]
        dy = target_y - disk_y[:, disk_index]
        distance = np.hypot(dx, dy)
        valid = distance > 0.0
        safe_distance = np.where(valid, distance, 1.0)
        qx = disk_x[:, disk_index] + radius * dx / safe_distance
        qy = disk_y[:, disk_index] + radius * dy / safe_distance
        consider(qx, qy, valid)

    for first in range(disk_count):
        for second in range(first + 1, disk_count):
            dx = disk_x[:, second] - disk_x[:, first]
            dy = disk_y[:, second] - disk_y[:, first]
            separation = np.hypot(dx, dy)
            valid = (separation > 0.0) & (separation <= 2.0 * radius)
            safe_separation = np.where(valid, separation, 1.0)
            midpoint_x = (disk_x[:, first] + disk_x[:, second]) / 2.0
            midpoint_y = (disk_y[:, first] + disk_y[:, second]) / 2.0
            half_chord = np.sqrt(
                np.maximum(0.0, radius * radius - (safe_separation / 2.0) ** 2)
            )
            perp_x = -dy / safe_separation
            perp_y = dx / safe_separation
            consider(
                midpoint_x + half_chord * perp_x,
                midpoint_y + half_chord * perp_y,
                valid,
            )
            consider(
                midpoint_x - half_chord * perp_x,
                midpoint_y - half_chord * perp_y,
                valid,
            )

    return best_x, best_y, np.isfinite(best_dist_sq)


def _nearest_feasible_pointing(
    target_x: np.ndarray,
    target_y: np.ndarray,
    disk_x: np.ndarray,
    disk_y: np.ndarray,
    radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project target points onto the intersection of 1-3 equal-radius disks."""

    (
        target_x,
        target_y,
        disk_x,
        disk_y,
        _row_count,
        disk_count,
        radius,
        allowed_radius_sq,
    ) = _coerce_recovery_projection_inputs(
        target_x,
        target_y,
        disk_x,
        disk_y,
        radius_arcsec,
    )
    if disk_count == 1:
        return _nearest_feasible_pointing_one_star(
            target_x,
            target_y,
            disk_x,
            disk_y,
            radius,
            allowed_radius_sq,
        )
    if disk_count == 2:
        return _nearest_feasible_pointing_two_star(
            target_x,
            target_y,
            disk_x,
            disk_y,
            radius,
            allowed_radius_sq,
        )
    return _nearest_feasible_pointing_generic(
        target_x,
        target_y,
        disk_x,
        disk_y,
        radius,
    )


def _combination_count(n: int, k: int) -> int:
    if n < k:
        return 0
    return int(math.comb(int(n), int(k)))


def _enabled_candidate_orders(runtime: PredictRuntime) -> tuple[int, ...]:
    return tuple(range(runtime.ao_system.min_wfs, runtime.ao_system.max_wfs + 1))


def _raw_candidate_estimate(n: int, runtime: PredictRuntime) -> int:
    return sum(_combination_count(n, order) for order in _enabled_candidate_orders(runtime))


def _sort_ngs_for_candidates(ngs: Table) -> Table:
    if len(ngs) == 0:
        return ngs.copy(copy_data=True)
    order = np.lexsort(
        (
            np.asarray(ngs["source_id"], dtype=np.int64),
            np.asarray(ngs["R"], dtype=np.float64),
        )
    )
    return ngs[order].copy(copy_data=True)


def _build_close_pair_edges(stars: Table, runtime: PredictRuntime) -> np.ndarray:
    if len(stars) < 2:
        return np.empty((0, 2), dtype=np.int64)
    coords = SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree))
    idx1, idx2, seps, _ = search_around_sky(
        coords,
        coords,
        runtime.ao_system.fov,
    )
    separations = seps.to(u.arcsec).value
    keep = (
        (np.asarray(idx1, dtype=np.int64) < np.asarray(idx2, dtype=np.int64))
        & (separations >= runtime.ao_system.min_sep.to_value(u.arcsec))
    )
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int64)
    edges = np.column_stack(
        (
            np.asarray(idx1[keep], dtype=np.int64),
            np.asarray(idx2[keep], dtype=np.int64),
        )
    )
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return np.asarray(edges[order], dtype=np.int64)


def _build_adjacency(size: int, edges: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((int(size), int(size)), dtype=np.bool_)
    if len(edges) > 0:
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
    return adjacency


def _build_graph_triangles(adjacency: np.ndarray) -> np.ndarray:
    size = adjacency.shape[0]
    triangles: list[tuple[int, int, int]] = []
    for first in range(size):
        seconds = np.flatnonzero(adjacency[first, first + 1 :]) + first + 1
        for second in seconds:
            thirds = np.flatnonzero(adjacency[first] & adjacency[second])
            thirds = thirds[thirds > second]
            for third in thirds:
                triangles.append((first, int(second), int(third)))
    if not triangles:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _filter_fov_valid_triangles(
    stars: Table,
    triangles: np.ndarray,
    runtime: PredictRuntime,
) -> np.ndarray:
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.int64)
    star_ra = np.asarray(stars["ra"], dtype=np.float64)
    star_dec = np.asarray(stars["dec"], dtype=np.float64)
    max_radius_arcsec = (
        runtime.ao_system.fov.to_value(u.arcsec) / 2.0
        + ASTERISM_CENTER_TOLERANCE_ARCSEC
    )
    keep = np.zeros((len(triangles),), dtype=np.bool_)
    for row, triangle in enumerate(np.asarray(triangles, dtype=np.int64)):
        center = _candidate_enclosing_fov_center(
            star_ra[triangle],
            star_dec[triangle],
        )
        keep[row] = center.radius_arcsec <= max_radius_arcsec
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(triangles[keep], dtype=np.int64)


def _build_candidate_graph_from_ngs(
    ngs: Table,
    runtime: PredictRuntime,
) -> _CandidateGraph:
    """Build the geometry-constrained candidate graph for a selected NGS table."""

    sorted_ngs = _sort_ngs_for_candidates(ngs)
    final_count = len(sorted_ngs)
    if final_count < runtime.ao_system.min_wfs:
        return _CandidateGraph(
            sorted_ngs=sorted_ngs[:0],
            final_count=0,
            candidate_count=0,
            edges=np.empty((0, 2), dtype=np.int64),
            triangles=np.empty((0, 3), dtype=np.int64),
        )

    enabled = set(_enabled_candidate_orders(runtime))
    edges = (
        _build_close_pair_edges(sorted_ngs, runtime)
        if (2 in enabled or 3 in enabled)
        else np.empty((0, 2), dtype=np.int64)
    )
    adjacency = (
        _build_adjacency(final_count, edges)
        if 3 in enabled
        else np.zeros((final_count, final_count), dtype=np.bool_)
    )
    triangles = (
        _filter_fov_valid_triangles(
            sorted_ngs,
            _build_graph_triangles(adjacency),
            runtime,
        )
        if 3 in enabled
        else np.empty((0, 3), dtype=np.int64)
    )
    candidate_count = (
        (final_count if 1 in enabled else 0)
        + (len(edges) if 2 in enabled else 0)
        + (len(triangles) if 3 in enabled else 0)
    )
    return _CandidateGraph(
        sorted_ngs=sorted_ngs,
        final_count=final_count,
        candidate_count=int(candidate_count),
        edges=edges,
        triangles=triangles,
    )


def _candidate_source_keys_from_graph(
    graph: _CandidateGraph,
    runtime: PredictRuntime,
) -> list[tuple[int, int, int]]:
    enabled = set(_enabled_candidate_orders(runtime))
    source_ids = np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64)
    keys: list[tuple[int, int, int]] = []
    if 1 in enabled:
        keys.extend((int(source_id), -1, -1) for source_id in source_ids)
    if 2 in enabled:
        keys.extend(
            (int(source_ids[int(first)]), int(source_ids[int(second)]), -1)
            for first, second in graph.edges
        )
    if 3 in enabled:
        keys.extend(
            (
                int(source_ids[int(first)]),
                int(source_ids[int(second)]),
                int(source_ids[int(third)]),
            )
            for first, second, third in graph.triangles
        )
    return keys


def _build_candidate_graph_from_source_keys(
    ngs: Table,
    keys: set[tuple[int, int, int]],
    runtime: PredictRuntime,
) -> _CandidateGraph:
    if not keys:
        return _CandidateGraph(
            sorted_ngs=_sort_ngs_for_candidates(ngs[:0]),
            final_count=0,
            candidate_count=0,
            edges=np.empty((0, 2), dtype=np.int64),
            triangles=np.empty((0, 3), dtype=np.int64),
        )

    selected_source_ids = sorted(
        {
            int(source_id)
            for key in keys
            for source_id in key
            if int(source_id) >= 0
        }
    )
    source_to_ngs_index = {
        int(source_id): index
        for index, source_id in enumerate(np.asarray(ngs["source_id"], dtype=np.int64))
    }
    selected_rows = np.asarray(
        [source_to_ngs_index[source_id] for source_id in selected_source_ids],
        dtype=np.int64,
    )
    sorted_ngs = _sort_ngs_for_candidates(ngs[selected_rows])
    source_to_member = {
        int(source_id): index
        for index, source_id in enumerate(
            np.asarray(sorted_ngs["source_id"], dtype=np.int64),
        )
    }

    edge_set: set[tuple[int, int]] = set()
    triangle_set: set[tuple[int, int, int]] = set()
    for key in keys:
        members = [source_to_member[int(source_id)] for source_id in key if source_id >= 0]
        if len(members) == 2:
            first, second = sorted(members)
            edge_set.add((int(first), int(second)))
        elif len(members) == 3:
            first, second, third = sorted(members)
            triangle_set.add((int(first), int(second), int(third)))

    edges = (
        np.asarray(sorted(edge_set), dtype=np.int64)
        if edge_set
        else np.empty((0, 2), dtype=np.int64)
    )
    triangles = (
        np.asarray(sorted(triangle_set), dtype=np.int64)
        if triangle_set
        else np.empty((0, 3), dtype=np.int64)
    )
    triangles = _filter_fov_valid_triangles(sorted_ngs, triangles, runtime)
    enabled = set(_enabled_candidate_orders(runtime))
    candidate_count = (
        (len(sorted_ngs) if 1 in enabled else 0)
        + (len(edges) if 2 in enabled else 0)
        + (len(triangles) if 3 in enabled else 0)
    )
    return _CandidateGraph(
        sorted_ngs=sorted_ngs,
        final_count=len(sorted_ngs),
        candidate_count=int(candidate_count),
        edges=edges,
        triangles=triangles,
    )


def _candidate_source_key(
    members: np.ndarray,
    source_ids: np.ndarray,
) -> tuple[int, int, int]:
    key = []
    for member in members:
        key.append(-1 if member < 0 else int(source_ids[int(member)]))
    return tuple(key)  # type: ignore[return-value]


def _build_candidate_set(graph: _CandidateGraph, runtime: PredictRuntime) -> _CandidateSet:
    final_count = int(graph.final_count)
    if final_count == 0:
        return _CandidateSet(
            members=np.empty((0, 3), dtype=np.int64),
            star_counts=np.empty((0,), dtype=np.int64),
            source_keys=np.empty((0, 3), dtype=np.int64),
        )

    enabled = set(_enabled_candidate_orders(runtime))
    members: list[tuple[int, int, int]] = []
    if 1 in enabled:
        members.extend((idx, -1, -1) for idx in range(final_count))
    if 2 in enabled and len(graph.edges) > 0:
        members.extend(
            (int(first), int(second), -1)
            for first, second in graph.edges
            if first < final_count and second < final_count
        )
    if 3 in enabled and len(graph.triangles) > 0:
        members.extend(
            (int(first), int(second), int(third))
            for first, second, third in graph.triangles
            if first < final_count and second < final_count and third < final_count
        )
    if not members:
        return _CandidateSet(
            members=np.empty((0, 3), dtype=np.int64),
            star_counts=np.empty((0,), dtype=np.int64),
            source_keys=np.empty((0, 3), dtype=np.int64),
        )

    member_array = np.asarray(members, dtype=np.int64)
    source_ids = np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64)
    keys = np.asarray(
        [_candidate_source_key(row, source_ids) for row in member_array],
        dtype=np.int64,
    )
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    member_array = member_array[order]
    keys = keys[order]
    star_counts = np.count_nonzero(member_array >= 0, axis=1).astype(np.int64)
    return _CandidateSet(members=member_array, star_counts=star_counts, source_keys=keys)


def _count_candidate_identities(stars: Table, runtime: PredictRuntime) -> _CandidateCount:
    """Return geometry-constrained one-, two-, and three-star candidate counts."""

    enabled = set(_enabled_candidate_orders(runtime))
    graph = _build_candidate_graph_from_ngs(stars, runtime)
    singles = graph.final_count if 1 in enabled else 0
    pairs = len(graph.edges) if 2 in enabled else 0
    triples = len(graph.triangles) if 3 in enabled else 0
    return _CandidateCount(
        total=int(singles + pairs + triples),
        singles=int(singles),
        pairs=int(pairs),
        triples=int(triples),
    )


def _pack_pixel_indices(pixel_indices: np.ndarray, inner_count: int) -> np.ndarray:
    words = np.zeros((int(math.ceil(inner_count / 64)),), dtype=np.uint64)
    if len(pixel_indices) == 0:
        return words
    indexes = np.asarray(pixel_indices, dtype=np.int64)
    word_indexes = indexes // 64
    bit_indexes = indexes % 64
    values = np.left_shift(np.uint64(1), bit_indexes.astype(np.uint64))
    np.bitwise_or.at(words, word_indexes, values)
    return words


def _all_allowed_bitset(inner_count: int) -> np.ndarray:
    words = np.full((int(math.ceil(inner_count / 64)),), np.uint64(2**64 - 1), dtype=np.uint64)
    remainder = inner_count % 64
    if remainder:
        words[-1] = np.uint64((1 << remainder) - 1)
    return words


def _bitset_to_indices(words: np.ndarray, inner_count: int) -> np.ndarray:
    word_indexes = np.flatnonzero(words)
    if len(word_indexes) == 0:
        return np.array([], dtype=np.int64)
    selected_words = np.ascontiguousarray(words[word_indexes])
    bits = np.unpackbits(selected_words.view(np.uint8), bitorder="little").reshape(
        -1,
        64,
    )
    local_indexes = np.flatnonzero(bits)
    if len(local_indexes) == 0:
        return np.array([], dtype=np.int64)
    indexes = (
        word_indexes[local_indexes // 64].astype(np.int64) * 64
        + (local_indexes % 64)
    )
    if indexes[-1] >= inner_count:
        indexes = indexes[indexes < inner_count]
    return indexes.astype(np.int64, copy=False)


def _build_star_pixel_bitsets(
    stars: Table,
    inner_centres: SkyCoord,
    runtime: PredictRuntime,
    *,
    radius_arcsec: float | None = None,
) -> np.ndarray:
    inner_count = len(inner_centres)
    words = int(math.ceil(inner_count / 64))
    bitsets = np.zeros((len(stars), words), dtype=np.uint64)
    if len(stars) == 0 or inner_count == 0:
        return bitsets
    star_coords = SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree))
    pixel_idxs, star_idxs, _, _ = search_around_sky(
        inner_centres,
        star_coords,
        (
            runtime.ao_system.fov / 2.0
            if radius_arcsec is None
            else float(radius_arcsec) * u.arcsec
        ),
    )
    if len(pixel_idxs) == 0:
        return bitsets
    order = np.argsort(star_idxs, kind="stable")
    sorted_star_idxs = np.asarray(star_idxs[order], dtype=np.int64)
    sorted_pixel_idxs = np.asarray(pixel_idxs[order], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(sorted_star_idxs[1:] != sorted_star_idxs[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_star_idxs)]
    for start, end in zip(starts, ends, strict=True):
        star_idx = int(sorted_star_idxs[start])
        bitsets[star_idx] = _pack_pixel_indices(sorted_pixel_idxs[start:end], inner_count)
    return bitsets


def _bitset_to_mask(words: np.ndarray, inner_count: int) -> np.ndarray:
    mask = np.zeros((int(inner_count),), dtype=np.bool_)
    mask[_bitset_to_indices(words, inner_count)] = True
    return mask


def _build_star_for_coverage(
    stars: Table,
    inner_centres: SkyCoord,
    runtime: PredictRuntime,
    *,
    required_depth: int,
    allowed_pixels: np.ndarray | None = None,
) -> _StarForCoverage:
    """Build star-level FOR coverage for FOR-optimized NGS selection."""

    inner_count = len(inner_centres)
    star_count = len(stars)
    if required_depth < 1:
        raise BuildError("required_depth must be at least 1")
    if allowed_pixels is None:
        allowed = np.ones((inner_count,), dtype=np.bool_)
    else:
        allowed = np.asarray(allowed_pixels, dtype=np.bool_)
        if allowed.shape != (inner_count,):
            raise BuildError(
                "allowed_pixels must have one entry per inner pixel "
                f"({inner_count}), got {allowed.shape}"
            )

    starts = np.zeros((star_count + 1,), dtype=np.int64)
    full_depth = np.zeros((inner_count,), dtype=np.uint32)
    target_depth = np.zeros((inner_count,), dtype=np.uint16)
    if star_count == 0 or inner_count == 0 or not np.any(allowed):
        return _StarForCoverage(
            pixel_indices=np.empty((0,), dtype=np.int64),
            starts=starts,
            full_depth=full_depth,
            target_depth=target_depth,
        )

    star_coords = SkyCoord(ra=stars["ra"], dec=stars["dec"], unit=(u.degree, u.degree))
    pixel_idxs, star_idxs, _, _ = search_around_sky(
        inner_centres,
        star_coords,
        runtime.ao_system.fov / 2.0,
    )
    if len(pixel_idxs) == 0:
        return _StarForCoverage(
            pixel_indices=np.empty((0,), dtype=np.int64),
            starts=starts,
            full_depth=full_depth,
            target_depth=target_depth,
        )

    pixel_idxs = np.asarray(pixel_idxs, dtype=np.int64)
    star_idxs = np.asarray(star_idxs, dtype=np.int64)
    keep = allowed[pixel_idxs]
    pixel_idxs = pixel_idxs[keep]
    star_idxs = star_idxs[keep]
    if len(pixel_idxs) == 0:
        return _StarForCoverage(
            pixel_indices=np.empty((0,), dtype=np.int64),
            starts=starts,
            full_depth=full_depth,
            target_depth=target_depth,
        )

    full_depth = np.bincount(pixel_idxs, minlength=inner_count).astype(
        np.uint32,
        copy=False,
    )
    target_depth = np.minimum(full_depth, int(required_depth)).astype(
        np.uint16,
        copy=False,
    )
    target_depth[~allowed] = 0

    order = np.argsort(star_idxs, kind="stable")
    sorted_star_idxs = star_idxs[order]
    sorted_pixel_idxs = pixel_idxs[order]
    counts = np.bincount(sorted_star_idxs, minlength=star_count).astype(
        np.int64,
        copy=False,
    )
    starts[1:] = np.cumsum(counts, dtype=np.int64)
    return _StarForCoverage(
        pixel_indices=sorted_pixel_idxs.astype(np.int64, copy=False),
        starts=starts,
        full_depth=full_depth,
        target_depth=target_depth,
    )


def _select_ngs_for_multicover(
    ngs: Table,
    coverage: _StarForCoverage,
    *,
    selected_depth_limit: int | None = None,
) -> _MulticoverSelection:
    """Select a bright NGS subset preserving each pixel's achievable FOR depth."""

    star_count = len(ngs)
    if len(coverage.starts) != star_count + 1:
        raise BuildError("coverage starts must have one entry per NGS plus one")
    if selected_depth_limit is not None and int(selected_depth_limit) < 1:
        raise BuildError("selected_depth_limit must be at least 1 when set")
    depth = np.zeros(coverage.target_depth.shape, dtype=np.uint16)
    selected_depth = np.zeros(coverage.target_depth.shape, dtype=np.uint16)
    undercovered = int(np.count_nonzero(depth < coverage.target_depth))
    if star_count == 0 or undercovered == 0:
        return _MulticoverSelection(
            star_indices=np.empty((0,), dtype=np.int64),
            depth=depth,
        )

    order = np.lexsort(
        (
            np.asarray(ngs["source_id"], dtype=np.int64),
            np.asarray(ngs["R"], dtype=np.float64),
        )
    )
    selected: list[int] = []
    for star_idx in np.asarray(order, dtype=np.int64):
        start = int(coverage.starts[int(star_idx)])
        end = int(coverage.starts[int(star_idx) + 1])
        if start == end:
            continue
        pixels = coverage.pixel_indices[start:end]
        target = coverage.target_depth[pixels]
        previous = depth[pixels]
        needs = previous < target
        if not np.any(needs):
            continue
        if (
            selected_depth_limit is not None
            and np.any(selected_depth[pixels] >= int(selected_depth_limit))
        ):
            continue

        selected.append(int(star_idx))
        updated = np.minimum(previous + 1, target).astype(np.uint16, copy=False)
        depth[pixels] = updated
        selected_depth[pixels] = np.minimum(
            selected_depth[pixels] + 1,
            int(selected_depth_limit)
            if selected_depth_limit is not None
            else np.iinfo(np.uint16).max,
        ).astype(np.uint16, copy=False)
        undercovered -= int(np.count_nonzero((previous < target) & (updated >= target)))
        if undercovered == 0:
            break

    return _MulticoverSelection(
        star_indices=np.asarray(selected, dtype=np.int64),
        depth=depth,
    )


def _build_pixel_star_csr(coverage: _StarForCoverage) -> tuple[np.ndarray, np.ndarray]:
    star_count = len(coverage.starts) - 1
    inner_count = len(coverage.target_depth)
    counts = np.diff(coverage.starts).astype(np.int64, copy=False)
    if star_count == 0 or int(np.sum(counts)) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.zeros((inner_count + 1,), dtype=np.int64),
        )

    star_indices = np.repeat(np.arange(star_count, dtype=np.int64), counts)
    order = np.argsort(coverage.pixel_indices, kind="stable")
    sorted_pixels = np.asarray(coverage.pixel_indices[order], dtype=np.int64)
    sorted_stars = np.asarray(star_indices[order], dtype=np.int64)
    pixel_counts = np.bincount(sorted_pixels, minlength=inner_count).astype(
        np.int64,
        copy=False,
    )
    starts = np.zeros((inner_count + 1,), dtype=np.int64)
    starts[1:] = np.cumsum(pixel_counts, dtype=np.int64)
    return sorted_stars, starts


def _region_star_indices(
    pixel_indices: np.ndarray,
    pixel_star_indices: np.ndarray,
    pixel_star_starts: np.ndarray,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for pixel_idx in np.asarray(pixel_indices, dtype=np.int64):
        start = int(pixel_star_starts[int(pixel_idx)])
        end = int(pixel_star_starts[int(pixel_idx) + 1])
        if start != end:
            parts.append(pixel_star_indices[start:end])
    if not parts:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(parts).astype(np.int64, copy=False))


def _restrict_star_for_coverage(
    coverage: _StarForCoverage,
    star_indices: np.ndarray,
    pixel_indices: np.ndarray,
) -> _StarForCoverage:
    star_indices = np.asarray(star_indices, dtype=np.int64)
    pixel_indices = np.asarray(pixel_indices, dtype=np.int64)
    inner_count = len(coverage.target_depth)
    pixel_mask = np.zeros((inner_count,), dtype=np.bool_)
    pixel_mask[pixel_indices] = True

    starts = np.zeros((len(star_indices) + 1,), dtype=np.int64)
    parts: list[np.ndarray] = []
    for local_idx, star_idx in enumerate(star_indices):
        start = int(coverage.starts[int(star_idx)])
        end = int(coverage.starts[int(star_idx) + 1])
        if start == end:
            starts[local_idx + 1] = starts[local_idx]
            continue
        pixels = coverage.pixel_indices[start:end]
        kept = pixels[pixel_mask[pixels]]
        if len(kept) > 0:
            parts.append(np.asarray(kept, dtype=np.int64))
        starts[local_idx + 1] = starts[local_idx] + len(kept)

    target_depth = np.zeros_like(coverage.target_depth)
    target_depth[pixel_indices] = coverage.target_depth[pixel_indices]
    full_depth = np.zeros_like(coverage.full_depth)
    full_depth[pixel_indices] = coverage.full_depth[pixel_indices]
    pixel_rows = (
        np.concatenate(parts).astype(np.int64, copy=False)
        if parts
        else np.empty((0,), dtype=np.int64)
    )
    return _StarForCoverage(
        pixel_indices=pixel_rows,
        starts=starts,
        full_depth=full_depth,
        target_depth=target_depth,
    )


def _count_bitset_words(words: np.ndarray) -> int:
    packed = np.ascontiguousarray(words, dtype=np.uint64)
    return int(np.unpackbits(packed.view(np.uint8), bitorder="little").sum())


def _build_region_star_bitsets(
    coverage: _StarForCoverage,
    star_indices: np.ndarray,
    pixel_indices: np.ndarray,
) -> np.ndarray:
    inner_count = len(coverage.target_depth)
    pixel_mask = np.zeros((inner_count,), dtype=np.bool_)
    pixel_mask[np.asarray(pixel_indices, dtype=np.int64)] = True
    bitsets = np.zeros(
        (len(star_indices), int(math.ceil(inner_count / 64))),
        dtype=np.uint64,
    )
    for local_idx, star_idx in enumerate(np.asarray(star_indices, dtype=np.int64)):
        start = int(coverage.starts[int(star_idx)])
        end = int(coverage.starts[int(star_idx) + 1])
        if start == end:
            continue
        pixels = coverage.pixel_indices[start:end]
        bitsets[local_idx] = _pack_pixel_indices(pixels[pixel_mask[pixels]], inner_count)
    return bitsets


def _count_final_candidate_graph_resolved_inferences(
    graph: _CandidateGraph,
    coverage: _StarForCoverage,
    ngs: Table,
    pixel_indices: np.ndarray,
    runtime: PredictRuntime,
) -> int:
    if graph.final_count == 0:
        return 0
    bitsets = _build_region_star_bitsets(
        coverage,
        _graph_star_indices(graph, ngs),
        pixel_indices,
    )
    enabled = set(_enabled_candidate_orders(runtime))
    total = 0
    if 1 in enabled:
        for row in bitsets:
            total += _count_bitset_words(row)
    if 2 in enabled:
        for first, second in graph.edges:
            total += _count_bitset_words(bitsets[int(first)] & bitsets[int(second)])
    if 3 in enabled:
        for first, second, third in graph.triangles:
            total += _count_bitset_words(
                bitsets[int(first)] & bitsets[int(second)] & bitsets[int(third)],
            )
    return int(total)


def _graph_star_indices(
    graph: _CandidateGraph,
    ngs: Table,
) -> np.ndarray:
    source_to_star_index = {
        int(source_id): index
        for index, source_id in enumerate(np.asarray(ngs["source_id"], dtype=np.int64))
    }
    return np.asarray(
        [
            source_to_star_index[int(source_id)]
            for source_id in np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64)
        ],
        dtype=np.int64,
    )


def _split_region_pixels(
    pixel_indices: np.ndarray,
    *,
    depth: int,
    outer_level: int,
    inner_level: int,
) -> list[np.ndarray]:
    next_depth = int(depth) + 1
    inner_depth = int(inner_level) - int(outer_level)
    if next_depth > inner_depth:
        return []
    child_size = 4 ** (inner_depth - next_depth)
    child_ids = np.asarray(pixel_indices, dtype=np.int64) // int(child_size)
    order = np.argsort(child_ids, kind="stable")
    sorted_child_ids = child_ids[order]
    sorted_pixels = np.asarray(pixel_indices, dtype=np.int64)[order]
    starts = np.r_[0, np.flatnonzero(sorted_child_ids[1:] != sorted_child_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_pixels)]
    return [
        np.asarray(sorted_pixels[start:end], dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
        if end > start
    ]


def _build_regional_candidate_graph(
    ngs: Table,
    coverage: _StarForCoverage,
    runtime: PredictRuntime,
) -> tuple[_CandidateGraph, _RegionalCandidateStats]:
    """Build complete regional candidates where tractable and FOR-selected elsewhere."""

    target_pixels = np.flatnonzero(coverage.target_depth > 0).astype(np.int64)
    if len(ngs) < runtime.ao_system.min_wfs or len(target_pixels) == 0:
        empty_graph = _build_candidate_graph_from_ngs(ngs[:0], runtime)
        return empty_graph, _RegionalCandidateStats(0, 0, 0, 0, 0, 0, 0, 0)

    pixel_star_indices, pixel_star_starts = _build_pixel_star_csr(coverage)
    max_depth = _regional_selector_max_depth(runtime)
    stack: list[tuple[np.ndarray, int]] = [(target_pixels, 0)]
    exact_candidate_keys: set[tuple[int, int, int]] = set()
    fallback_regions: list[tuple[np.ndarray, np.ndarray]] = []
    exact_regions = 0
    for_optimized_regions = 0
    split_regions = 0
    deepest_region = 0
    max_regional_combination_work = _max_regional_combination_work(runtime)
    sparse_graph_probe_star_limit = _regional_sparse_graph_probe_star_limit(runtime)
    exact_combination_work = 0

    while stack:
        region_pixels, depth = stack.pop()
        deepest_region = max(deepest_region, int(depth))
        region_stars = _region_star_indices(
            region_pixels,
            pixel_star_indices,
            pixel_star_starts,
        )
        if len(region_stars) < runtime.ao_system.min_wfs:
            continue

        raw_candidate_work = _raw_candidate_estimate(len(region_stars), runtime)
        graph: _CandidateGraph | None = None
        if raw_candidate_work <= max_regional_combination_work or (
            int(depth) == 0 and len(region_stars) <= sparse_graph_probe_star_limit
        ):
            candidate_graph = _build_candidate_graph_from_ngs(ngs[region_stars], runtime)
            if int(candidate_graph.candidate_count) <= max_regional_combination_work:
                graph = candidate_graph

        if graph is not None:
            exact_candidate_keys.update(_candidate_source_keys_from_graph(graph, runtime))
            exact_regions += 1
            exact_combination_work += int(graph.candidate_count)
            continue

        children = (
            _split_region_pixels(
                region_pixels,
                depth=depth,
                outer_level=runtime.outer_level,
                inner_level=runtime.inner_level,
            )
            if depth < max_depth and len(region_pixels) > 1
            else []
        )
        if children:
            stack.extend((child, depth + 1) for child in children)
            split_regions += 1
            continue

        fallback_regions.append(
            (
                np.asarray(region_stars, dtype=np.int64),
                np.asarray(region_pixels, dtype=np.int64),
            )
        )
        for_optimized_regions += 1

    def build_for_graph(
    ) -> tuple[_CandidateGraph, int, int]:
        candidate_keys = set(exact_candidate_keys)
        incomplete_regions = 0
        for region_stars, region_pixels in fallback_regions:
            region_coverage = _restrict_star_for_coverage(
                coverage,
                region_stars,
                region_pixels,
            )
            selection = _select_ngs_for_multicover(
                ngs[region_stars],
                region_coverage,
            )
            if np.any(selection.depth[region_pixels] < region_coverage.target_depth[region_pixels]):
                incomplete_regions += 1
            selected_stars = region_stars[selection.star_indices]
            graph = _build_candidate_graph_from_ngs(ngs[selected_stars], runtime)
            candidate_keys.update(_candidate_source_keys_from_graph(graph, runtime))
        graph = _build_candidate_graph_from_source_keys(ngs, candidate_keys, runtime)
        return (
            graph,
            _count_final_candidate_graph_resolved_inferences(
                graph,
                coverage,
                ngs,
                target_pixels,
                runtime,
            ),
            incomplete_regions,
        )

    graph, final_resolved_inferences, incomplete_for_regions = build_for_graph()

    return graph, _RegionalCandidateStats(
        exact_regions=exact_regions,
        for_optimized_regions=for_optimized_regions,
        split_regions=split_regions,
        max_depth=deepest_region,
        max_regional_combination_work=max_regional_combination_work,
        exact_combination_work=exact_combination_work,
        final_resolved_inferences=final_resolved_inferences,
        incomplete_for_regions=incomplete_for_regions,
    )


def _build_bright_star_allowed_bitset(
    stars: Table,
    inner_centres: SkyCoord,
    runtime: PredictRuntime,
) -> np.ndarray:
    inner_count = len(inner_centres)
    allowed = _all_allowed_bitset(inner_count)
    threshold = runtime.max_bright_star_mag
    if threshold is None or len(stars) == 0 or inner_count == 0:
        return allowed
    bright_stars = stars[np.asarray(stars["R"], dtype=np.float64) < float(threshold)]
    if len(bright_stars) == 0:
        return allowed
    bright_coords = SkyCoord(
        ra=bright_stars["ra"],
        dec=bright_stars["dec"],
        unit=(u.degree, u.degree),
    )
    pixel_idxs, _, _, _ = search_around_sky(
        inner_centres,
        bright_coords,
        runtime.max_bright_star_exclusion,
    )
    if len(pixel_idxs) == 0:
        return allowed
    masked = np.unique(np.asarray(pixel_idxs, dtype=np.int64))
    masked_words = _pack_pixel_indices(masked, inner_count)
    return allowed & ~masked_words


def _build_ngs_feature_arrays(
    *,
    pixel_idxs: np.ndarray,
    candidate_ids: np.ndarray,
    num_stars: int,
    candidate_set: _CandidateSet,
    star_x: np.ndarray,
    star_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    star_mags: np.ndarray,
    pointing_x: np.ndarray | None = None,
    pointing_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = candidate_set.members[np.asarray(candidate_ids, dtype=np.int64), : int(num_stars)]
    pixels = np.asarray(pixel_idxs, dtype=np.int64)
    if pointing_x is None:
        center_x = inner_x[pixels]
    else:
        center_x = np.asarray(pointing_x, dtype=np.float64)
    if pointing_y is None:
        center_y = inner_y[pixels]
    else:
        center_y = np.asarray(pointing_y, dtype=np.float64)
    dx = star_x[members] - center_x[:, None]
    dy = star_y[members] - center_y[:, None]
    return (
        np.hypot(dx, dy),
        np.rad2deg(np.arctan2(dx, dy)),
        star_mags[members],
    )


def _candidate_is_better(
    *,
    ee: float,
    candidate_id: int,
    existing_ee: float,
    existing_candidate_id: int,
) -> bool:
    if not np.isfinite(existing_ee):
        return True
    if ee > existing_ee:
        return True
    return bool(ee == existing_ee and candidate_id < existing_candidate_id)


def _insert_top_candidate(
    pixel_idx: int,
    candidate_id: int,
    *,
    sr: float,
    ee: float,
    fwhm: float,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_sr: np.ndarray,
    top_fwhm: np.ndarray,
) -> None:
    if not np.isfinite(ee):
        return
    row_refs = top_refs[pixel_idx]
    row_ee = top_ee[pixel_idx]
    top_k = top_refs.shape[1]
    position = top_k
    for idx in range(top_k):
        if _candidate_is_better(
            ee=ee,
            candidate_id=candidate_id,
            existing_ee=float(row_ee[idx]),
            existing_candidate_id=int(row_refs[idx]),
        ):
            position = idx
            break
    if position == top_k:
        return
    if position + 1 < top_k:
        row_refs[position + 1 :] = row_refs[position:-1]
        row_ee[position + 1 :] = row_ee[position:-1]
        top_sr[pixel_idx, position + 1 :] = top_sr[pixel_idx, position:-1]
        top_fwhm[pixel_idx, position + 1 :] = top_fwhm[pixel_idx, position:-1]
    row_refs[position] = int(candidate_id)
    row_ee[position] = float(ee)
    top_sr[pixel_idx, position] = float(sr)
    top_fwhm[pixel_idx, position] = float(fwhm)


def _scatter_top_candidates(
    pixel_idxs: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    pointing_x: np.ndarray,
    pointing_y: np.ndarray,
    sr: np.ndarray,
    ee: np.ndarray,
    fwhm: np.ndarray,
    best_ee: np.ndarray,
    best_sr: np.ndarray,
    best_fwhm: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_sr: np.ndarray,
    top_fwhm: np.ndarray,
    top_pointing_x: np.ndarray,
    top_pointing_y: np.ndarray,
    profile: TraversalStageProfile | None = None,
) -> None:
    started = time.perf_counter()
    finite = np.isfinite(ee)
    if not np.any(finite):
        if profile is not None:
            profile.point_prediction_scatter_filter_seconds += (
                time.perf_counter() - started
            )
        return

    batch_pixels = np.asarray(pixel_idxs, dtype=np.int64)[finite]
    batch_candidates = np.asarray(candidate_ids, dtype=np.int64)[finite]
    batch_ee = np.asarray(ee, dtype=np.float64)[finite]
    batch_sr = np.asarray(sr, dtype=np.float64)[finite]
    batch_fwhm = np.asarray(fwhm, dtype=np.float64)[finite]
    batch_pointing_x = np.asarray(pointing_x, dtype=np.float64)[finite]
    batch_pointing_y = np.asarray(pointing_y, dtype=np.float64)[finite]
    if profile is not None:
        profile.point_prediction_scatter_filter_seconds += (
            time.perf_counter() - started
        )

    started = time.perf_counter()
    affected_pixels = np.unique(batch_pixels)
    top_k = top_refs.shape[1]
    existing_pixels = np.repeat(affected_pixels, top_k)
    existing_refs = top_refs[affected_pixels].reshape(-1)
    existing_ee = top_ee[affected_pixels].reshape(-1)
    existing_sr = top_sr[affected_pixels].reshape(-1)
    existing_fwhm = top_fwhm[affected_pixels].reshape(-1)
    existing_pointing_x = top_pointing_x[affected_pixels].reshape(-1)
    existing_pointing_y = top_pointing_y[affected_pixels].reshape(-1)
    existing_valid = (existing_refs >= 0) & np.isfinite(existing_ee)

    combined_pixels = np.concatenate((existing_pixels[existing_valid], batch_pixels))
    combined_refs = np.concatenate((existing_refs[existing_valid], batch_candidates))
    combined_ee = np.concatenate((existing_ee[existing_valid], batch_ee))
    combined_sr = np.concatenate((existing_sr[existing_valid], batch_sr))
    combined_fwhm = np.concatenate((existing_fwhm[existing_valid], batch_fwhm))
    combined_pointing_x = np.concatenate(
        (existing_pointing_x[existing_valid], batch_pointing_x)
    )
    combined_pointing_y = np.concatenate(
        (existing_pointing_y[existing_valid], batch_pointing_y)
    )
    if profile is not None:
        profile.point_prediction_scatter_merge_seconds += (
            time.perf_counter() - started
        )

    started = time.perf_counter()
    order = np.lexsort((combined_refs, -combined_ee, combined_pixels))

    sorted_pixels = combined_pixels[order]
    group_start = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    group_start_indexes = np.maximum.accumulate(
        np.where(group_start, np.arange(len(sorted_pixels)), 0)
    )
    ranks = np.arange(len(sorted_pixels)) - group_start_indexes
    keep = ranks < top_k
    if profile is not None:
        profile.point_prediction_scatter_sort_seconds += time.perf_counter() - started

    started = time.perf_counter()
    top_refs[affected_pixels] = -1
    top_ee[affected_pixels] = -np.inf
    top_sr[affected_pixels] = np.nan
    top_fwhm[affected_pixels] = np.nan
    top_pointing_x[affected_pixels] = np.nan
    top_pointing_y[affected_pixels] = np.nan

    kept_order = order[keep]
    kept_pixels = sorted_pixels[keep]
    kept_ranks = ranks[keep]
    top_refs[kept_pixels, kept_ranks] = combined_refs[kept_order]
    top_ee[kept_pixels, kept_ranks] = combined_ee[kept_order]
    top_sr[kept_pixels, kept_ranks] = combined_sr[kept_order]
    top_fwhm[kept_pixels, kept_ranks] = combined_fwhm[kept_order]
    top_pointing_x[kept_pixels, kept_ranks] = combined_pointing_x[kept_order]
    top_pointing_y[kept_pixels, kept_ranks] = combined_pointing_y[kept_order]

    best_candidate_ee = top_ee[affected_pixels, 0]
    improved = np.isfinite(best_candidate_ee) & (
        np.isnan(best_ee[affected_pixels]) | (best_candidate_ee > best_ee[affected_pixels])
    )
    improved_pixels = affected_pixels[improved]
    best_ee[improved_pixels] = top_ee[improved_pixels, 0]
    best_sr[improved_pixels] = top_sr[improved_pixels, 0]
    best_fwhm[improved_pixels] = top_fwhm[improved_pixels, 0]
    if profile is not None:
        profile.point_prediction_scatter_write_seconds += time.perf_counter() - started


def _flush_resolved_prediction_rows(
    runtime: PredictRuntime,
    execution_config: TraversalExecutionConfig,
    *,
    num_stars: int,
    pixel_idxs: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_set: _CandidateSet,
    star_x: np.ndarray,
    star_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    star_mags: np.ndarray,
    best_ee: np.ndarray,
    best_sr: np.ndarray,
    best_fwhm: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_sr: np.ndarray,
    top_fwhm: np.ndarray,
    top_pointing_x: np.ndarray,
    top_pointing_y: np.ndarray,
    pointing_x: np.ndarray | None = None,
    pointing_y: np.ndarray | None = None,
    recovered: bool = False,
    profile: TraversalStageProfile | None = None,
    structure_profile: TraversalStructureProfile | None = None,
) -> None:
    if len(pixel_idxs) == 0:
        return
    pixel_array = np.asarray(pixel_idxs, dtype=np.int64)
    candidate_array = np.asarray(candidate_ids, dtype=np.int64)
    pointing_x_array = (
        inner_x[pixel_array]
        if pointing_x is None
        else np.asarray(pointing_x, dtype=np.float64)
    )
    pointing_y_array = (
        inner_y[pixel_array]
        if pointing_y is None
        else np.asarray(pointing_y, dtype=np.float64)
    )
    started = time.perf_counter()
    ngs_zd, ngs_az, ngs_mag = _build_ngs_feature_arrays(
        pixel_idxs=pixel_array,
        candidate_ids=candidate_array,
        num_stars=num_stars,
        candidate_set=candidate_set,
        star_x=star_x,
        star_y=star_y,
        inner_x=inner_x,
        inner_y=inner_y,
        star_mags=star_mags,
        pointing_x=pointing_x_array,
        pointing_y=pointing_y_array,
    )
    if profile is not None:
        profile.point_prediction_ngs_array_seconds += time.perf_counter() - started
    started = time.perf_counter()
    model = get_point_model(
        runtime,
        num_stars,
        device=execution_config.prediction_device,
    )
    if profile is not None:
        profile.point_prediction_model_seconds += time.perf_counter() - started
    prediction_telemetry = _new_prediction_telemetry(profile, structure_profile)
    metrics = predict_point_arrays(
        runtime,
        num_stars=num_stars,
        model=model,
        ngs_zd=ngs_zd,
        ngs_az_deg=ngs_az,
        ngs_mag=ngs_mag,
        backend_row_count=_backend_row_count(
            len(ngs_zd),
            execution_config.resolved_backend_buckets,
        ),
        feature_buffer_row_count=_backend_buffer_row_count(
            execution_config.resolved_backend_buckets,
        ),
        prediction_telemetry=prediction_telemetry,
    )
    _record_prediction_telemetry(
        prediction_telemetry,
        resolved=True,
        profile=profile,
        structure_profile=structure_profile,
    )
    if recovered and structure_profile is not None and prediction_telemetry is not None:
        structure_profile.recovered_point_prediction_rows += int(
            prediction_telemetry.rows
        )
    started = time.perf_counter()
    _scatter_top_candidates(
        pixel_array,
        candidate_array,
        pointing_x=pointing_x_array,
        pointing_y=pointing_y_array,
        sr=np.asarray(metrics.sr, dtype=np.float64),
        ee=np.asarray(metrics.ee, dtype=np.float64),
        fwhm=np.asarray(metrics.fwhm, dtype=np.float64),
        best_ee=best_ee,
        best_sr=best_sr,
        best_fwhm=best_fwhm,
        top_refs=top_refs,
        top_ee=top_ee,
        top_sr=top_sr,
        top_fwhm=top_fwhm,
        top_pointing_x=top_pointing_x,
        top_pointing_y=top_pointing_y,
        profile=profile,
    )
    if profile is not None:
        profile.point_prediction_scatter_seconds += time.perf_counter() - started


def _stream_resolved_candidate_predictions(
    runtime: PredictRuntime,
    execution_config: TraversalExecutionConfig,
    candidate_set: _CandidateSet,
    star_pixel_bits: np.ndarray,
    bright_allowed_bits: np.ndarray,
    *,
    star_x: np.ndarray,
    star_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    star_mags: np.ndarray,
    best_ee: np.ndarray,
    best_sr: np.ndarray,
    best_fwhm: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_sr: np.ndarray,
    top_fwhm: np.ndarray,
    top_pointing_x: np.ndarray,
    top_pointing_y: np.ndarray,
    profile: TraversalStageProfile | None = None,
    structure_profile: TraversalStructureProfile | None = None,
    memory_pressure_callback: Callable[[], None] | None = None,
) -> int:
    inner_count = len(inner_x)
    stream_batch_size = _stream_batch_size(
        execution_config.prediction_batch_size,
        execution_config.resolved_backend_buckets,
    )
    buffer_capacity = stream_batch_size + inner_count
    buffers: dict[int, _PredictionRowBuffer] = {
        order: _PredictionRowBuffer.create(buffer_capacity)
        for order in _enabled_candidate_orders(runtime)
    }
    evaluated_rows = 0
    empty_footprints = 0
    prediction_flushes = 0

    def maybe_clear_cache() -> None:
        if execution_config.resolved_cache_clear_every < 1:
            return
        if prediction_flushes % execution_config.resolved_cache_clear_every == 0:
            _clear_backend_cache_with_profile(profile)

    for candidate_id, members in enumerate(candidate_set.members):
        valid_members = members[members >= 0]
        if len(valid_members) == 0:
            continue
        started = time.perf_counter()
        eligible_words = bright_allowed_bits.copy()
        for member in valid_members:
            eligible_words &= star_pixel_bits[int(member)]
        intersection_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pixel_idxs = _bitset_to_indices(eligible_words, inner_count)
        extract_seconds = time.perf_counter() - started
        if profile is not None:
            profile.point_prediction_eligibility_intersection_seconds += (
                intersection_seconds
            )
            profile.point_prediction_eligibility_extract_seconds += extract_seconds
            profile.point_prediction_eligibility_seconds += (
                intersection_seconds + extract_seconds
            )
        if len(pixel_idxs) == 0:
            empty_footprints += 1
            continue
        num_stars = int(candidate_set.star_counts[candidate_id])
        buffer = buffers[num_stars]
        started = time.perf_counter()
        buffer.append(pixel_idxs, int(candidate_id))
        if profile is not None:
            profile.point_prediction_buffer_seconds += time.perf_counter() - started
        evaluated_rows += int(len(pixel_idxs))
        while buffer.size >= stream_batch_size:
            batch_pixel_idxs, batch_candidate_ids, _, _ = buffer.head(
                stream_batch_size
            )
            _flush_resolved_prediction_rows(
                runtime,
                execution_config,
                num_stars=num_stars,
                pixel_idxs=batch_pixel_idxs,
                candidate_ids=batch_candidate_ids,
                candidate_set=candidate_set,
                star_x=star_x,
                star_y=star_y,
                inner_x=inner_x,
                inner_y=inner_y,
                star_mags=star_mags,
                best_ee=best_ee,
                best_sr=best_sr,
                best_fwhm=best_fwhm,
                top_refs=top_refs,
                top_ee=top_ee,
                top_sr=top_sr,
                top_fwhm=top_fwhm,
                top_pointing_x=top_pointing_x,
                top_pointing_y=top_pointing_y,
                profile=profile,
                structure_profile=structure_profile,
            )
            prediction_flushes += 1
            buffer.discard(stream_batch_size)
            maybe_clear_cache()
            if memory_pressure_callback is not None:
                memory_pressure_callback()
    for num_stars, buffer in buffers.items():
        batch_pixel_idxs, batch_candidate_ids, _, _ = buffer.arrays()
        _flush_resolved_prediction_rows(
            runtime,
            execution_config,
            num_stars=num_stars,
            pixel_idxs=batch_pixel_idxs,
            candidate_ids=batch_candidate_ids,
            candidate_set=candidate_set,
            star_x=star_x,
            star_y=star_y,
            inner_x=inner_x,
            inner_y=inner_y,
            star_mags=star_mags,
            best_ee=best_ee,
            best_sr=best_sr,
            best_fwhm=best_fwhm,
            top_refs=top_refs,
            top_ee=top_ee,
            top_sr=top_sr,
            top_fwhm=top_fwhm,
            top_pointing_x=top_pointing_x,
            top_pointing_y=top_pointing_y,
            profile=profile,
            structure_profile=structure_profile,
        )
        if len(batch_pixel_idxs):
            prediction_flushes += 1
            maybe_clear_cache()
            if memory_pressure_callback is not None:
                memory_pressure_callback()
    if execution_config.resolved_cache_clear_every > 0:
        _clear_backend_cache_with_profile(profile)
    if structure_profile is not None:
        structure_profile.context_pair_rows = int(evaluated_rows)
        structure_profile.post_bright_asterism_rows = int(empty_footprints)
    return evaluated_rows


def _stream_recovered_candidate_predictions(
    runtime: PredictRuntime,
    execution_config: TraversalExecutionConfig,
    candidate_set: _CandidateSet,
    recovery_pixel_bits: np.ndarray,
    recovery_allowed_bits: np.ndarray,
    *,
    star_x: np.ndarray,
    star_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    star_mags: np.ndarray,
    best_ee: np.ndarray,
    best_sr: np.ndarray,
    best_fwhm: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_sr: np.ndarray,
    top_fwhm: np.ndarray,
    top_pointing_x: np.ndarray,
    top_pointing_y: np.ndarray,
    recovery_pixel_index_map: np.ndarray | None = None,
    profile: TraversalStageProfile | None = None,
    structure_profile: TraversalStructureProfile | None = None,
    memory_pressure_callback: Callable[[], None] | None = None,
) -> int:
    pixel_index_map = (
        None
        if recovery_pixel_index_map is None
        else np.asarray(recovery_pixel_index_map, dtype=np.int64)
    )
    inner_count = len(inner_x) if pixel_index_map is None else len(pixel_index_map)
    if (
        len(candidate_set.members) == 0
        or len(recovery_allowed_bits) == 0
        or not np.any(recovery_allowed_bits)
    ):
        return 0

    radius_arcsec = runtime.ao_system.fov.to_value(u.arcsec) / 2.0
    max_science_distance_sq = (
        radius_arcsec + ASTERISM_CENTER_TOLERANCE_ARCSEC
    ) ** 2
    stream_batch_size = _stream_batch_size(
        execution_config.prediction_batch_size,
        execution_config.resolved_backend_buckets,
    )
    buffer_capacity = stream_batch_size
    buffers: dict[int, _PredictionRowBuffer] = {
        order: _PredictionRowBuffer.create(buffer_capacity, with_pointing=True)
        for order in _enabled_candidate_orders(runtime)
    }
    recovered_rows = 0
    prediction_flushes = 0

    def maybe_clear_cache() -> None:
        if execution_config.resolved_cache_clear_every < 1:
            return
        if prediction_flushes % execution_config.resolved_cache_clear_every == 0:
            _clear_backend_cache_with_profile(profile)

    def flush_buffer(num_stars: int, rows: int | None = None) -> None:
        nonlocal prediction_flushes
        buffer = buffers[int(num_stars)]
        if buffer.size == 0:
            return
        batch_size = buffer.size if rows is None else int(rows)
        batch_pixel_idxs, batch_candidate_ids, batch_pointing_x, batch_pointing_y = (
            buffer.head(batch_size)
        )
        if len(batch_pixel_idxs) == 0:
            return
        _flush_resolved_prediction_rows(
            runtime,
            execution_config,
            num_stars=num_stars,
            pixel_idxs=batch_pixel_idxs,
            candidate_ids=batch_candidate_ids,
            candidate_set=candidate_set,
            star_x=star_x,
            star_y=star_y,
            inner_x=inner_x,
            inner_y=inner_y,
            star_mags=star_mags,
            best_ee=best_ee,
            best_sr=best_sr,
            best_fwhm=best_fwhm,
            top_refs=top_refs,
            top_ee=top_ee,
            top_sr=top_sr,
            top_fwhm=top_fwhm,
            top_pointing_x=top_pointing_x,
            top_pointing_y=top_pointing_y,
            pointing_x=batch_pointing_x,
            pointing_y=batch_pointing_y,
            recovered=True,
            profile=profile,
            structure_profile=structure_profile,
        )
        prediction_flushes += 1
        buffer.discard(len(batch_pixel_idxs))
        maybe_clear_cache()
        if memory_pressure_callback is not None:
            memory_pressure_callback()

    for num_stars in _enabled_candidate_orders(runtime):
        order_candidate_ids = np.flatnonzero(
            np.asarray(candidate_set.star_counts, dtype=np.int64) == int(num_stars)
        )
        for candidate_id in order_candidate_ids:
            members = candidate_set.members[int(candidate_id), : int(num_stars)]
            eligible_words = np.asarray(recovery_allowed_bits, dtype=np.uint64).copy()
            for member in members:
                eligible_words &= recovery_pixel_bits[int(member)]
            recovery_pixel_idxs = _bitset_to_indices(eligible_words, inner_count)
            if len(recovery_pixel_idxs) == 0:
                continue
            pixel_idxs = (
                recovery_pixel_idxs
                if pixel_index_map is None
                else pixel_index_map[recovery_pixel_idxs]
            )
            target_x = inner_x[pixel_idxs]
            target_y = inner_y[pixel_idxs]
            pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
                target_x,
                target_y,
                np.broadcast_to(
                    star_x[members],
                    (len(recovery_pixel_idxs), int(num_stars)),
                ),
                np.broadcast_to(
                    star_y[members],
                    (len(recovery_pixel_idxs), int(num_stars)),
                ),
                radius_arcsec,
            )
            science_distance_sq = (
                (pointing_x - target_x) ** 2 + (pointing_y - target_y) ** 2
            )
            valid = feasible & (science_distance_sq <= max_science_distance_sq)
            valid_positions = np.flatnonzero(valid)
            recovered_rows += int(len(valid_positions))
            if len(valid_positions) == 0:
                continue
            buffer = buffers[int(num_stars)]
            buffer.append(
                pixel_idxs[valid_positions],
                int(candidate_id),
                pointing_x=pointing_x[valid_positions],
                pointing_y=pointing_y[valid_positions],
            )
            while buffer.size >= stream_batch_size:
                flush_buffer(num_stars, stream_batch_size)
    for num_stars in _enabled_candidate_orders(runtime):
        flush_buffer(num_stars)
    if prediction_flushes > 0 and execution_config.resolved_cache_clear_every > 0:
        _clear_backend_cache_with_profile(profile)
    return recovered_rows


def _local_neighbor_index_matrix(inner_pixs: np.ndarray, inner_level: int) -> np.ndarray:
    pixels = np.asarray(inner_pixs, dtype=np.int64)
    if len(pixels) == 0:
        return np.empty((0, 8), dtype=np.int64)
    first_pix = int(pixels[0])
    expected_pixels = np.arange(first_pix, first_pix + len(pixels), dtype=np.int64)
    if not np.array_equal(pixels, expected_pixels):
        raise BuildError(
            "Winner regularization requires a dense contiguous nested inner-pixel block"
        )

    with np.errstate(invalid="ignore"):
        neighbours = astropy_healpix.neighbours(
            pixels,
            2 ** int(inner_level),
            order="nested",
        )
    neighbour_array = np.asarray(neighbours, dtype=np.int64)
    if neighbour_array.shape == (8, len(pixels)):
        neighbour_array = neighbour_array.T

    stop_pix = first_pix + len(pixels)
    local = neighbour_array - first_pix
    local[(neighbour_array < first_pix) | (neighbour_array >= stop_pix)] = -1
    return np.asarray(local, dtype=np.int64)


def _top_position_for_label(top_refs: np.ndarray, pixel_idx: int, label: int) -> int:
    matches = np.flatnonzero(top_refs[pixel_idx] == int(label))
    if len(matches) == 0:
        return -1
    return int(matches[0])


def _regularize_winner_labels(
    runtime: PredictRuntime,
    inner_pixs: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
) -> np.ndarray:
    labels = top_refs[:, 0].copy()
    best = top_ee[:, 0]
    labels[~np.isfinite(best)] = -1
    if len(labels) == 0 or np.count_nonzero(labels >= 0) == 0:
        return labels

    epsilon_ok = (
        (top_refs >= 0)
        & np.isfinite(top_ee)
        & (top_ee >= ((1.0 - float(runtime.winner_ee_epsilon)) * best[:, None]))
    )
    neighbours = _local_neighbor_index_matrix(inner_pixs, runtime.inner_level)
    row_indexes = np.arange(len(labels), dtype=np.int64)
    for _ in range(WINNER_REGULARIZATION_PASSES):
        neighbour_labels = np.full(neighbours.shape, -1, dtype=np.int64)
        valid_neighbours = neighbours >= 0
        neighbour_labels[valid_neighbours] = labels[neighbours[valid_neighbours]]

        candidate_support = np.count_nonzero(
            (top_refs[:, None, :] >= 0)
            & (neighbour_labels[:, :, None] == top_refs[:, None, :]),
            axis=1,
        ).astype(np.int64)
        candidate_support = np.where(epsilon_ok, candidate_support, -1)
        current_support = np.count_nonzero(
            neighbour_labels == labels[:, None],
            axis=1,
        ).astype(np.int64)

        current_matches = top_refs == labels[:, None]
        has_current = np.any(current_matches, axis=1)
        current_positions = np.argmax(current_matches, axis=1)
        best_local_ee = np.full((len(labels),), -np.inf, dtype=np.float64)
        best_local_ee[has_current] = top_ee[
            row_indexes[has_current],
            current_positions[has_current],
        ]
        best_label = labels.copy()
        best_support = current_support.copy()

        valid_pixels = labels >= 0
        for position in range(top_refs.shape[1]):
            candidate_label = top_refs[:, position]
            support = candidate_support[:, position]
            local_ee = top_ee[:, position]
            can_use = valid_pixels & (candidate_label >= 0)
            better_support = can_use & (support > best_support)
            better_tie = (
                can_use
                & (support == best_support)
                & (support > current_support)
                & (
                    (local_ee > best_local_ee)
                    | ((local_ee == best_local_ee) & (candidate_label < best_label))
                )
            )
            update = better_support | better_tie
            best_label[update] = candidate_label[update]
            best_support[update] = support[update]
            best_local_ee[update] = local_ee[update]
        labels = best_label
    return labels


def _fill_regularized_winner_fields(
    inner: Table,
    labels: np.ndarray,
    top_refs: np.ndarray,
    top_ee: np.ndarray,
    top_pointing_x: np.ndarray,
    top_pointing_y: np.ndarray,
    retained_candidate_ids: np.ndarray,
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    candidate_to_asterism_id = {
        int(candidate_id): index + 1
        for index, candidate_id in enumerate(np.asarray(retained_candidate_ids, dtype=np.int64))
    }
    winner_ids = np.full((len(inner),), -1, dtype=np.int64)
    winner_ee_resolved = np.asarray(inner["winner_ee_resolved"], dtype=np.float64).copy()
    winner_pointing_x = np.full((len(inner),), np.nan, dtype=np.float64)
    winner_pointing_y = np.full((len(inner),), np.nan, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    retained = np.asarray(retained_candidate_ids, dtype=np.int64)
    if len(retained) > 0:
        positions = np.searchsorted(retained, label_array)
        valid = (
            (label_array >= 0)
            & (positions < len(retained))
            & (retained[np.minimum(positions, len(retained) - 1)] == label_array)
        )
        winner_ids[valid] = positions[valid] + 1

        matches = top_refs == label_array[:, None]
        has_match = valid & np.any(matches, axis=1)
        match_positions = np.argmax(matches, axis=1)
        row_indexes = np.arange(len(label_array), dtype=np.int64)
        winner_ee_resolved[has_match] = np.fmax(
            winner_ee_resolved[has_match],
            top_ee[
                row_indexes[has_match],
                match_positions[has_match],
            ],
        )
        winner_pointing_x[has_match] = top_pointing_x[
            row_indexes[has_match],
            match_positions[has_match],
        ]
        winner_pointing_y[has_match] = top_pointing_y[
            row_indexes[has_match],
            match_positions[has_match],
        ]
    inner["winner_asterism_id"] = winner_ids
    inner["winner_ee_resolved"] = winner_ee_resolved
    return candidate_to_asterism_id, winner_pointing_x, winner_pointing_y


def _build_retained_asterism_table(
    candidate_set: _CandidateSet,
    retained_candidate_ids: np.ndarray,
    stars: Table,
    runtime: PredictRuntime,
) -> Table:
    if len(retained_candidate_ids) == 0:
        return _empty_persisted_asterisms()

    retained_candidate_ids = np.asarray(retained_candidate_ids, dtype=np.int64)
    rows: dict[str, list[float | int]] = {
        "asterism_id": [],
        "ra": [],
        "dec": [],
        "num_stars": [],
    }
    for index in (1, 2, 3):
        rows[f"star{index}_source_id"] = []
        rows[f"star{index}_ra"] = []
        rows[f"star{index}_dec"] = []
        rows[f"star{index}_mag"] = []

    source_ids = np.asarray(stars["source_id"], dtype=np.int64)
    star_ra = np.asarray(stars["ra"], dtype=np.float64)
    star_dec = np.asarray(stars["dec"], dtype=np.float64)
    star_mag = np.asarray(stars["R"], dtype=np.float64)
    for row_id, candidate_id in enumerate(retained_candidate_ids, start=1):
        members = candidate_set.members[int(candidate_id)]
        members = np.asarray(members[members >= 0], dtype=np.int64)
        member_ra = star_ra[members]
        member_dec = star_dec[members]
        center = _candidate_enclosing_fov_center(member_ra, member_dec)

        rows["asterism_id"].append(row_id)
        rows["ra"].append(center.ra_deg)
        rows["dec"].append(center.dec_deg)
        rows["num_stars"].append(int(len(members)))
        for slot in range(3):
            prefix = f"star{slot + 1}"
            if slot < len(members):
                member = int(members[slot])
                rows[f"{prefix}_source_id"].append(int(source_ids[member]))
                rows[f"{prefix}_ra"].append(float(star_ra[member]))
                rows[f"{prefix}_dec"].append(float(star_dec[member]))
                rows[f"{prefix}_mag"].append(float(star_mag[member]))
            else:
                rows[f"{prefix}_source_id"].append(-1)
                rows[f"{prefix}_ra"].append(-1.0)
                rows[f"{prefix}_dec"].append(-1.0)
                rows[f"{prefix}_mag"].append(-1.0)

    raw = Table(
        [
            np.asarray(rows["asterism_id"], dtype=np.int64),
            np.asarray(rows["ra"], dtype=np.float64),
            np.asarray(rows["dec"], dtype=np.float64),
            np.asarray(rows["num_stars"], dtype=np.int64),
            np.asarray(rows["star1_source_id"], dtype=np.int64),
            np.asarray(rows["star1_ra"], dtype=np.float64),
            np.asarray(rows["star1_dec"], dtype=np.float64),
            np.asarray(rows["star1_mag"], dtype=np.float64),
            np.asarray(rows["star2_source_id"], dtype=np.int64),
            np.asarray(rows["star2_ra"], dtype=np.float64),
            np.asarray(rows["star2_dec"], dtype=np.float64),
            np.asarray(rows["star2_mag"], dtype=np.float64),
            np.asarray(rows["star3_source_id"], dtype=np.int64),
            np.asarray(rows["star3_ra"], dtype=np.float64),
            np.asarray(rows["star3_dec"], dtype=np.float64),
            np.asarray(rows["star3_mag"], dtype=np.float64),
        ],
        names=(
            "asterism_id",
            "ra",
            "dec",
            "num_stars",
            "star1_source_id",
            "star1_ra",
            "star1_dec",
            "star1_mag",
            "star2_source_id",
            "star2_ra",
            "star2_dec",
            "star2_mag",
            "star3_source_id",
            "star3_ra",
            "star3_dec",
            "star3_mag",
        ),
    )
    raw["pix"] = get_pixel_from_skycoord(
        runtime.inner_level,
        SkyCoord(ra=raw["ra"], dec=raw["dec"], unit=(u.degree, u.degree)),
    )
    return _to_persisted_asterisms(raw)


def _update_regularized_winner_averaged_ee(
    runtime: PredictRuntime,
    execution_config: TraversalExecutionConfig,
    inner: Table,
    labels: np.ndarray,
    candidate_set: _CandidateSet,
    *,
    star_x: np.ndarray,
    star_y: np.ndarray,
    inner_x: np.ndarray,
    inner_y: np.ndarray,
    winner_pointing_x: np.ndarray,
    winner_pointing_y: np.ndarray,
    star_mags: np.ndarray,
    profile: TraversalStageProfile | None = None,
    structure_profile: TraversalStructureProfile | None = None,
    memory_pressure_callback: Callable[[], None] | None = None,
) -> None:
    winner_pixel_idxs = np.flatnonzero(np.asarray(labels, dtype=np.int64) >= 0)
    if len(winner_pixel_idxs) == 0:
        return
    buffers: dict[int, dict[str, list[int]]] = {
        order: {"pixel_idxs": [], "candidate_ids": []}
        for order in _enabled_candidate_orders(runtime)
    }
    prediction_flushes = 0
    stream_batch_size = _stream_batch_size(
        execution_config.prediction_batch_size,
        execution_config.averaged_backend_buckets,
    )

    def maybe_clear_cache() -> None:
        if execution_config.resolved_cache_clear_every < 1:
            return
        if prediction_flushes % execution_config.resolved_cache_clear_every == 0:
            _clear_backend_cache_with_profile(profile)

    for pixel_idx in winner_pixel_idxs:
        candidate_id = int(labels[int(pixel_idx)])
        num_stars = int(candidate_set.star_counts[candidate_id])
        buffers[num_stars]["pixel_idxs"].append(int(pixel_idx))
        buffers[num_stars]["candidate_ids"].append(candidate_id)

    for num_stars, buffer in buffers.items():
        pixel_idxs = buffer["pixel_idxs"]
        candidate_ids = buffer["candidate_ids"]
        if not pixel_idxs:
            continue
        model = get_mean_model(
            runtime,
            num_stars,
            device=execution_config.averaged_prediction_device,
        )
        for start_idx in range(0, len(pixel_idxs), stream_batch_size):
            end_idx = min(start_idx + stream_batch_size, len(pixel_idxs))
            batch_pixel_idxs = np.asarray(pixel_idxs[start_idx:end_idx], dtype=np.int64)
            batch_candidate_ids = np.asarray(candidate_ids[start_idx:end_idx], dtype=np.int64)
            ngs_zd, ngs_az, ngs_mag = _build_ngs_feature_arrays(
                pixel_idxs=batch_pixel_idxs,
                candidate_ids=batch_candidate_ids,
                num_stars=num_stars,
                candidate_set=candidate_set,
                star_x=star_x,
                star_y=star_y,
                inner_x=inner_x,
                inner_y=inner_y,
                star_mags=star_mags,
                pointing_x=np.asarray(winner_pointing_x, dtype=np.float64)[
                    batch_pixel_idxs
                ],
                pointing_y=np.asarray(winner_pointing_y, dtype=np.float64)[
                    batch_pixel_idxs
                ],
            )
            prediction_telemetry = _new_prediction_telemetry(profile, structure_profile)
            predicted_ee_averaged = predict_field_mean_arrays(
                runtime,
                num_stars=num_stars,
                model=model,
                ngs_zd=ngs_zd,
                ngs_az_deg=ngs_az,
                ngs_mag=ngs_mag,
                backend_row_count=_backend_row_count(
                    len(ngs_zd),
                    execution_config.averaged_backend_buckets,
                ),
                feature_buffer_row_count=_backend_buffer_row_count(
                    execution_config.averaged_backend_buckets,
                ),
                prediction_telemetry=prediction_telemetry,
            )
            current_ee_averaged = np.asarray(
                inner["winner_ee_averaged"],
                dtype=np.float64,
            )[batch_pixel_idxs]
            inner["winner_ee_averaged"][batch_pixel_idxs] = np.fmax(
                current_ee_averaged,
                predicted_ee_averaged,
            )
            _record_prediction_telemetry(
                prediction_telemetry,
                resolved=False,
                profile=profile,
                structure_profile=structure_profile,
            )
            prediction_flushes += 1
            maybe_clear_cache()
            if memory_pressure_callback is not None:
                memory_pressure_callback()
    if execution_config.resolved_cache_clear_every > 0:
        _clear_backend_cache_with_profile(profile)


def build_traversal_products(
    store: GaiaHealpixStore,
    runtime: PredictRuntime,
    outer_pix: int,
    *,
    execution_config: TraversalExecutionConfig | None = None,
    dust_root: Path,
    max_data_level: int,
    geometry: TraversalGeometry | None = None,
    profile: TraversalStageProfile | None = None,
    structure_profile: TraversalStructureProfile | None = None,
    memory_profile: TraversalMemoryProfile | None = None,
    rss_sampler=None,
    peak_sampler=None,
    memory_pressure_callback: Callable[[], None] | None = None,
) -> tuple[Table, Table]:
    """Return retained asterisms and the rich inner table for one outer pixel."""

    execution_config = execution_config or TraversalExecutionConfig()
    geometry = geometry or TraversalGeometry.from_runtime(runtime)

    started = time.perf_counter()
    stars, ngs = prepare_search_inputs(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
    )
    if structure_profile is not None:
        structure_profile.search_star_rows = int(len(stars))
        structure_profile.ngs_rows = int(len(ngs))
    if profile is not None:
        profile.star_selection_seconds += time.perf_counter() - started
    _sample_memory(
        memory_profile,
        "rss_after_star_selection_mb",
        "peak_rss_after_star_selection_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )

    started = time.perf_counter()
    inner = build_base_inner_table(
        store,
        runtime,
        outer_pix,
        geometry=geometry,
        asterisms=None,
    )
    if profile is not None:
        profile.inner_table_seconds += time.perf_counter() - started

    inner_centres = geometry.inner_centres(outer_pix)
    outer_centre = get_pixel_skycoord(runtime.outer_level, outer_pix)
    inner_x, inner_y = _get_plane_offsets_arcsec(outer_centre, inner_centres)
    retained_asterisms = _empty_persisted_asterisms()

    started = time.perf_counter()
    bright_allowed_bits = _build_bright_star_allowed_bitset(stars, inner_centres, runtime)
    bright_allowed_pixels = _bitset_to_mask(bright_allowed_bits, len(inner))
    if profile is not None:
        profile.bright_star_filter_seconds += time.perf_counter() - started

    started = time.perf_counter()
    coverage = _build_star_for_coverage(
        ngs,
        inner_centres,
        runtime,
        required_depth=int(runtime.ao_system.max_wfs),
        allowed_pixels=bright_allowed_pixels,
    )
    graph, _regional_stats = _build_regional_candidate_graph(
        ngs,
        coverage,
        runtime,
    )
    candidate_set = _build_candidate_set(graph, runtime)
    if profile is not None:
        profile.candidate_generation_seconds += time.perf_counter() - started
    if structure_profile is not None:
        structure_profile.close_pair_rows = int(len(graph.edges))
        structure_profile.raw_asterism_rows = int(len(candidate_set.members))
        structure_profile.dedupe_key_rows = int(graph.candidate_count)
        structure_profile.candidate_graph_rows = int(graph.candidate_count)
        structure_profile.close_pair_rows_peak = int(len(graph.edges))
        structure_profile.raw_asterism_rows_peak = int(len(candidate_set.members))
    _sample_memory(
        memory_profile,
        "rss_after_candidate_generation_mb",
        "peak_rss_after_candidate_generation_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )
    if memory_pressure_callback is not None:
        memory_pressure_callback()

    if len(candidate_set.members) > 0:
        started = time.perf_counter()
        candidate_ngs = graph.sorted_ngs[: graph.final_count]
        candidate_coords = SkyCoord(
            ra=candidate_ngs["ra"],
            dec=candidate_ngs["dec"],
            unit=(u.degree, u.degree),
        )
        star_x, star_y = _get_plane_offsets_arcsec(outer_centre, candidate_coords)
        star_mags = np.asarray(candidate_ngs["R"], dtype=np.float64)
        star_pixel_bits = _build_star_pixel_bitsets(candidate_ngs, inner_centres, runtime)
        if profile is not None:
            profile.context_seconds += time.perf_counter() - started
        _sample_memory(
            memory_profile,
            "rss_after_context_mb",
            "peak_rss_after_context_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )

        best_ee = np.asarray(inner["best_ee"], dtype=np.float64).copy()
        best_sr = np.asarray(inner["best_sr"], dtype=np.float64).copy()
        best_fwhm = np.asarray(inner["best_fwhm"], dtype=np.float64).copy()
        top_shape = (len(inner), int(runtime.winner_top_k))
        top_refs = np.full(top_shape, -1, dtype=np.int64)
        top_ee = np.full(top_shape, -np.inf, dtype=np.float64)
        top_sr = np.full(top_shape, np.nan, dtype=np.float64)
        top_fwhm = np.full(top_shape, np.nan, dtype=np.float64)
        top_pointing_x = np.full(top_shape, np.nan, dtype=np.float64)
        top_pointing_y = np.full(top_shape, np.nan, dtype=np.float64)

        started = time.perf_counter()
        _stream_resolved_candidate_predictions(
            runtime,
            execution_config,
            candidate_set,
            star_pixel_bits,
            bright_allowed_bits,
            star_x=star_x,
            star_y=star_y,
            inner_x=inner_x,
            inner_y=inner_y,
            star_mags=star_mags,
            best_ee=best_ee,
            best_sr=best_sr,
            best_fwhm=best_fwhm,
            top_refs=top_refs,
            top_ee=top_ee,
            top_sr=top_sr,
            top_fwhm=top_fwhm,
            top_pointing_x=top_pointing_x,
            top_pointing_y=top_pointing_y,
            profile=profile,
            structure_profile=structure_profile,
            memory_pressure_callback=memory_pressure_callback,
        )
        no_winner_pixel_idxs = np.flatnonzero(top_refs[:, 0] < 0)
        if len(no_winner_pixel_idxs) > 0:
            recovery_inner_centres = inner_centres[no_winner_pixel_idxs]
            recovery_pixel_bits = _build_star_pixel_bitsets(
                candidate_ngs,
                recovery_inner_centres,
                runtime,
                radius_arcsec=runtime.ao_system.fov.to_value(u.arcsec),
            )
            recovery_allowed_pixel_idxs = np.flatnonzero(
                bright_allowed_pixels[no_winner_pixel_idxs]
            )
            recovery_allowed_bits = _pack_pixel_indices(
                recovery_allowed_pixel_idxs,
                len(recovery_inner_centres),
            )
            _stream_recovered_candidate_predictions(
                runtime,
                execution_config,
                candidate_set,
                recovery_pixel_bits,
                recovery_allowed_bits,
                star_x=star_x,
                star_y=star_y,
                inner_x=inner_x,
                inner_y=inner_y,
                star_mags=star_mags,
                best_ee=best_ee,
                best_sr=best_sr,
                best_fwhm=best_fwhm,
                top_refs=top_refs,
                top_ee=top_ee,
                top_sr=top_sr,
                top_fwhm=top_fwhm,
                top_pointing_x=top_pointing_x,
                top_pointing_y=top_pointing_y,
                recovery_pixel_index_map=no_winner_pixel_idxs,
                profile=profile,
                structure_profile=structure_profile,
                memory_pressure_callback=memory_pressure_callback,
            )
        inner["best_ee"] = best_ee
        inner["best_sr"] = best_sr
        inner["best_fwhm"] = best_fwhm
        if profile is not None:
            profile.point_prediction_seconds += time.perf_counter() - started
        if structure_profile is not None:
            structure_profile.winner_payload_rows = int(np.count_nonzero(top_refs[:, 0] >= 0))
            structure_profile.search_star_rows_peak = int(len(stars))
            structure_profile.ngs_rows_peak = int(len(ngs))
            structure_profile.raw_asterism_rows_peak = int(len(candidate_set.members))
            structure_profile.context_pair_rows_peak = int(structure_profile.context_pair_rows)
            structure_profile.winner_payload_rows_peak = int(structure_profile.winner_payload_rows)
        _sample_memory(
            memory_profile,
            "rss_after_point_prediction_mb",
            "peak_rss_after_point_prediction_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )

        started = time.perf_counter()
        labels = _regularize_winner_labels(
            runtime,
            np.asarray(inner["pix"], dtype=np.int64),
            top_refs,
            top_ee,
        )
        retained_candidate_ids = np.unique(labels[labels >= 0])
        _candidate_to_asterism_id, winner_pointing_x, winner_pointing_y = (
            _fill_regularized_winner_fields(
                inner,
                labels,
                top_refs,
                top_ee,
                top_pointing_x,
                top_pointing_y,
                retained_candidate_ids,
            )
        )
        if structure_profile is not None:
            recovered_winners = (labels >= 0) & np.isfinite(winner_pointing_x)
            recovered_winners &= (
                (np.abs(winner_pointing_x - inner_x) > ASTERISM_CENTER_TOLERANCE_ARCSEC)
                | (np.abs(winner_pointing_y - inner_y) > ASTERISM_CENTER_TOLERANCE_ARCSEC)
            )
            structure_profile.recovered_winner_pixels = int(
                np.count_nonzero(recovered_winners)
            )
        retained_asterisms = _build_retained_asterism_table(
            candidate_set,
            retained_candidate_ids,
            candidate_ngs,
            runtime,
        )
        if profile is not None:
            profile.local_selection_seconds += time.perf_counter() - started
        if structure_profile is not None:
            structure_profile.local_asterism_rows = int(len(retained_asterisms))
            structure_profile.winner_rows = int(np.count_nonzero(labels >= 0))
            structure_profile.local_asterism_rows_peak = int(len(retained_asterisms))

        started = time.perf_counter()
        _update_regularized_winner_averaged_ee(
            runtime,
            execution_config,
            inner,
            labels,
            candidate_set,
            star_x=star_x,
            star_y=star_y,
            inner_x=inner_x,
            inner_y=inner_y,
            winner_pointing_x=winner_pointing_x,
            winner_pointing_y=winner_pointing_y,
            star_mags=star_mags,
            profile=profile,
            structure_profile=structure_profile,
            memory_pressure_callback=memory_pressure_callback,
        )
        if profile is not None:
            profile.field_mean_prediction_seconds += time.perf_counter() - started
        if execution_config.resolved_cache_clear_every == 0:
            _clear_backend_cache_with_profile(profile)
        _sample_memory(
            memory_profile,
            "rss_after_field_mean_mb",
            "peak_rss_after_field_mean_mb",
            rss_sampler=rss_sampler,
            peak_sampler=peak_sampler,
        )

    started = time.perf_counter()
    inner["coverage_resolved"] = (
        np.asarray(inner["winner_ee_resolved"], dtype=np.float64)
        >= runtime.coverage_ee_threshold_resolved
    )
    inner["coverage_averaged"] = (
        np.asarray(inner["winner_ee_averaged"], dtype=np.float64)
        >= runtime.coverage_ee_threshold_averaged
    )
    if profile is not None:
        profile.coverage_seconds += time.perf_counter() - started
    _sample_memory(
        memory_profile,
        "rss_after_coverage_mb",
        "peak_rss_after_coverage_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )

    started = time.perf_counter()
    inner = add_gaia_a0_to_inner(
        inner,
        dust_root=dust_root,
        outer_level=runtime.outer_level,
        outer_pix=outer_pix,
        inner_level=runtime.inner_level,
        max_data_level=max_data_level,
    )
    if profile is not None:
        profile.dust_seconds += time.perf_counter() - started
    _sample_memory(
        memory_profile,
        "rss_after_dust_mb",
        "peak_rss_after_dust_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )

    started = time.perf_counter()
    persisted_asterisms = retained_asterisms
    if profile is not None:
        profile.persisted_asterisms_seconds += time.perf_counter() - started
    _sample_memory(
        memory_profile,
        "rss_after_persisted_asterisms_mb",
        "peak_rss_after_persisted_asterisms_mb",
        rss_sampler=rss_sampler,
        peak_sampler=peak_sampler,
    )
    return persisted_asterisms, inner
