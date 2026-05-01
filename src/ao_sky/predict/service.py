"""Native prediction helpers backed temporarily by girmos-aosims."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import astropy.units as u
import numpy as np

from . import backend
from ._exceptions import PredictError
from ._models import PointPredictionBatch, PredictRuntime, SeeingBaselinePerformance

_ModelCacheKey = tuple[str, str, str, int, bool]
_FeatureTemplateKey = tuple[float, int, bool, tuple[tuple[float, float], ...]]
_POINT_MODEL_CACHE: dict[_ModelCacheKey, object] = {}
_MEAN_MODEL_CACHE: dict[_ModelCacheKey, object] = {}
_FEATURE_TEMPLATE_CACHE: dict[_FeatureTemplateKey, "_FeatureTemplate"] = {}
_FEATURE_BUFFER_CACHE: dict[int, np.ndarray] = {}


@dataclass(slots=True)
class PredictionArrayTelemetry:
    """Mutable timing and feature-memory counters for vectorized prediction."""

    feature_seconds: float = 0.0
    backend_seconds: float = 0.0
    batches: int = 0
    rows: int = 0
    batch_rows_peak: int = 0
    backend_rows: int = 0
    backend_batch_rows_peak: int = 0
    feature_bytes_peak: int = 0
    mps_current_bytes_peak: int = 0
    mps_driver_bytes_peak: int = 0
    mps_recommended_bytes: int = 0
    backend_bucket_counts: dict[int, int] = field(default_factory=dict)
    backend_bucket_rows: dict[int, int] = field(default_factory=dict)

    def record_feature_batch(
        self,
        x: np.ndarray,
        elapsed_seconds: float,
        *,
        row_count: int | None = None,
        record_bucket: bool = False,
    ) -> None:
        logical_rows = int(x.shape[0] if row_count is None else row_count)
        backend_rows = int(x.shape[0])
        self.feature_seconds += float(elapsed_seconds)
        self.batches += 1
        self.rows += logical_rows
        self.batch_rows_peak = max(self.batch_rows_peak, logical_rows)
        self.backend_rows += backend_rows
        self.backend_batch_rows_peak = max(self.backend_batch_rows_peak, backend_rows)
        self.feature_bytes_peak = max(self.feature_bytes_peak, int(x.nbytes))
        if record_bucket:
            self.backend_bucket_counts[backend_rows] = (
                self.backend_bucket_counts.get(backend_rows, 0) + 1
            )
            self.backend_bucket_rows[backend_rows] = (
                self.backend_bucket_rows.get(backend_rows, 0) + logical_rows
            )

    def record_backend_batch(self, elapsed_seconds: float) -> None:
        self.backend_seconds += float(elapsed_seconds)

    def record_mps_snapshot(self, model) -> None:
        try:
            device_type = backend.get_model_device_type(model)
        except (AttributeError, KeyError, TypeError, PredictError):
            return
        if device_type != "mps":
            return
        current_bytes, driver_bytes, recommended_bytes = backend.get_mps_memory_bytes()
        self.mps_current_bytes_peak = max(
            self.mps_current_bytes_peak,
            int(current_bytes),
        )
        self.mps_driver_bytes_peak = max(
            self.mps_driver_bytes_peak,
            int(driver_bytes),
        )
        self.mps_recommended_bytes = max(
            self.mps_recommended_bytes,
            int(recommended_bytes),
        )


@dataclass(frozen=True, slots=True)
class _FeatureTemplate:
    """Static feature row and dynamic NGS column positions."""

    row: np.ndarray
    zd_columns: tuple[int, ...]
    az_columns: tuple[int, ...]
    mag_columns: tuple[int, ...]


def _get_model_cache_key(
    prefix: str,
    runtime: PredictRuntime,
    *,
    model_name: str,
    num_stars: int,
    force_cpu: bool,
) -> _ModelCacheKey:
    return (
        prefix,
        str(Path(runtime.model_root).expanduser().resolve()),
        model_name,
        int(num_stars),
        bool(force_cpu),
    )


def _force_cpu_for_device(device: str, *, field_name: str) -> bool:
    value = str(device).strip().lower()
    if value in {"", "cpu"}:
        return True
    if value == "auto":
        return False
    raise PredictError(f"{field_name} must be 'cpu' or 'auto', got {value!r}")


def clear_backend_cache() -> None:
    """Clear loaded model caches owned by the temporary native backend."""

    cleared_models: set[int] = set()
    for model in tuple(_POINT_MODEL_CACHE.values()) + tuple(_MEAN_MODEL_CACHE.values()):
        model_id = id(model)
        if model_id in cleared_models:
            continue
        backend.clear_cache(model)
        cleared_models.add(model_id)


def configure_inference_threads(num_threads: int) -> None:
    """Pin backend inference libraries to a bounded thread count."""

    backend.configure_inference_threads(num_threads)


def get_point_model(runtime: PredictRuntime, num_stars: int, *, device: str = "cpu"):
    """Return the cached point model for one surviving guide-star count."""

    key = f"{int(num_stars)}star"
    model_name = runtime.resolved_models.get(key)
    if model_name is None:
        raise PredictError(f"Missing resolved model for {key}")

    force_cpu = _force_cpu_for_device(device, field_name="device")
    cache_key = _get_model_cache_key(
        "point",
        runtime,
        model_name=model_name,
        num_stars=num_stars,
        force_cpu=force_cpu,
    )
    if cache_key not in _POINT_MODEL_CACHE:
        _POINT_MODEL_CACHE[cache_key] = backend.load_model(
            runtime.model_root,
            model_name,
            force_cpu=force_cpu,
        )
    return _POINT_MODEL_CACHE[cache_key]


def get_mean_model(runtime: PredictRuntime, num_stars: int, *, device: str = "cpu"):
    """Return the cached mean-field model for one surviving guide-star count."""

    key = f"{int(num_stars)}star"
    model_name = runtime.averaged_models.get(key)
    if model_name is None:
        raise PredictError(f"Missing averaged model for {key}")

    force_cpu = _force_cpu_for_device(device, field_name="device")
    cache_key = _get_model_cache_key(
        "mean",
        runtime,
        model_name=model_name,
        num_stars=num_stars,
        force_cpu=force_cpu,
    )
    if cache_key not in _MEAN_MODEL_CACHE:
        _MEAN_MODEL_CACHE[cache_key] = backend.load_model(
            runtime.model_root,
            model_name,
            force_cpu=force_cpu,
        )
    return _MEAN_MODEL_CACHE[cache_key]


def warm_model_cache(
    runtime: PredictRuntime,
    *,
    prediction_device: str = "cpu",
    averaged_prediction_device: str = "cpu",
) -> None:
    """Load all configured temporary backend models for one AO runtime."""

    for num_stars in range(runtime.ao_system.min_wfs, runtime.ao_system.max_wfs + 1):
        get_point_model(runtime, num_stars, device=prediction_device)
        get_mean_model(runtime, num_stars, device=averaged_prediction_device)


def _get_ao_lgs_xy(lgs: tuple[dict[str, float], ...]) -> list[dict[str, float]]:
    return [
        {
            **star,
            "x": float(star["zd"]) * np.cos(np.deg2rad(float(star["az"]))),
            "y": float(star["zd"]) * np.sin(np.deg2rad(float(star["az"]))),
        }
        for star in lgs
    ]


def _wrap_angle_deg(angle: np.ndarray) -> np.ndarray:
    return (angle + 180.0) % 360.0 - 180.0


def _get_feature_template_key(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    mean_only: bool,
) -> _FeatureTemplateKey:
    wavelength_micron = float(runtime.prediction_wavelength.to(u.micron).value)
    lgs_key = (
        ()
        if mean_only
        else tuple(
            (float(star["zd"]), float(star["az"]))
            for star in runtime.ao_system.lgs
        )
    )
    return (wavelength_micron, int(num_stars), bool(mean_only), lgs_key)


def _get_feature_template(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    mean_only: bool,
) -> _FeatureTemplate:
    cache_key = _get_feature_template_key(
        runtime,
        num_stars=num_stars,
        mean_only=mean_only,
    )
    template = _FEATURE_TEMPLATE_CACHE.get(cache_key)
    if template is not None:
        return template

    wavelength_micron = cache_key[0]
    lgs = () if mean_only else tuple(_get_ao_lgs_xy(runtime.ao_system.lgs))
    column_count = 1 + 3 * int(num_stars)
    if not mean_only:
        column_count += 2 + len(lgs)

    row = np.zeros((column_count,), dtype=np.float64)
    row[0] = wavelength_micron
    zd_columns: list[int] = []
    az_columns: list[int] = []
    mag_columns: list[int] = []
    column = 1
    for _star_idx in range(int(num_stars)):
        zd_columns.append(column)
        column += 1
        az_columns.append(column)
        column += 1
        mag_columns.append(column)
        column += 1

    if not mean_only:
        column += 2  # science target r and theta are zero in Traversal point mode.
        for lgs_star in lgs:
            row[column] = float(np.hypot(float(lgs_star["x"]), float(lgs_star["y"])))
            column += 1

    template = _FeatureTemplate(
        row=row,
        zd_columns=tuple(zd_columns),
        az_columns=tuple(az_columns),
        mag_columns=tuple(mag_columns),
    )
    _FEATURE_TEMPLATE_CACHE[cache_key] = template
    return template


def _get_reusable_feature_buffer(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    mean_only: bool,
    row_count: int,
    buffer_row_count: int,
    template: _FeatureTemplate,
    buffer_key: int,
) -> np.ndarray:
    x = _FEATURE_BUFFER_CACHE.get(int(buffer_key))
    if x is None or x.shape[0] < int(buffer_row_count):
        x = np.empty((int(buffer_row_count), template.row.shape[0]), dtype=np.float64)
        x[:] = template.row
        _FEATURE_BUFFER_CACHE[int(buffer_key)] = x
    if x.shape[0] == int(row_count):
        return x
    return x[: int(row_count)]


def build_model_x_from_ngs_arrays(
    runtime: PredictRuntime,
    *,
    ngs_zd: np.ndarray,
    ngs_az_deg: np.ndarray,
    ngs_mag: np.ndarray,
    mean_only: bool = False,
    backend_row_count: int | None = None,
    feature_buffer_key: int | None = None,
    feature_buffer_row_count: int | None = None,
) -> np.ndarray:
    """Return backend model features from magnitude-ordered NGS arrays."""

    ngs_zd = np.asarray(ngs_zd, dtype=np.float64)
    ngs_az_deg = np.asarray(ngs_az_deg, dtype=np.float64)
    ngs_mag = np.asarray(ngs_mag, dtype=np.float64)
    if ngs_zd.shape != ngs_az_deg.shape or ngs_zd.shape != ngs_mag.shape:
        raise PredictError("NGS zd, az, and magnitude arrays must have identical shapes")
    if ngs_zd.ndim != 2:
        raise PredictError("NGS arrays must have shape (rows, num_stars)")

    row_count, num_stars = ngs_zd.shape
    output_rows = row_count if backend_row_count is None else int(backend_row_count)
    if output_rows < row_count:
        raise PredictError("backend_row_count cannot be smaller than the input row count")
    buffer_rows = (
        output_rows
        if feature_buffer_row_count is None
        else int(feature_buffer_row_count)
    )
    if buffer_rows < output_rows:
        raise PredictError(
            "feature_buffer_row_count cannot be smaller than backend_row_count"
        )
    if output_rows > 0 and row_count == 0:
        raise PredictError("Cannot pad an empty NGS batch")
    template = _get_feature_template(
        runtime,
        num_stars=int(num_stars),
        mean_only=mean_only,
    )

    ordered_az_rad = np.deg2rad(_wrap_angle_deg(ngs_az_deg))

    if backend_row_count is None or feature_buffer_key is None:
        x = np.empty((output_rows, template.row.shape[0]), dtype=np.float64)
        x[:] = template.row
    else:
        x = _get_reusable_feature_buffer(
            runtime,
            num_stars=int(num_stars),
            mean_only=mean_only,
            row_count=output_rows,
            buffer_row_count=buffer_rows,
            template=template,
            buffer_key=feature_buffer_key,
        )
    for star_idx, (zd_column, az_column, mag_column) in enumerate(
        zip(
            template.zd_columns,
            template.az_columns,
            template.mag_columns,
            strict=True,
        )
    ):
        x[:row_count, zd_column] = ngs_zd[:, star_idx]
        x[:row_count, az_column] = ordered_az_rad[:, star_idx]
        x[:row_count, mag_column] = ngs_mag[:, star_idx]
    return x


def _validate_prediction_num_stars(ngs_zd: np.ndarray, num_stars: int) -> None:
    shape = np.asarray(ngs_zd).shape
    if len(shape) != 2:
        return
    actual_num_stars = int(shape[1])
    if actual_num_stars != int(num_stars):
        raise PredictError(
            "num_stars must match the second dimension of the NGS arrays "
            f"({num_stars} != {actual_num_stars})"
        )


def predict_point_arrays(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    model,
    ngs_zd: np.ndarray,
    ngs_az_deg: np.ndarray,
    ngs_mag: np.ndarray,
    backend_row_count: int | None = None,
    feature_buffer_row_count: int | None = None,
    prediction_telemetry: PredictionArrayTelemetry | None = None,
) -> PointPredictionBatch:
    """Return resolved predictions for one magnitude-ordered homogeneous batch."""

    row_count = int(np.asarray(ngs_zd).shape[0])
    _validate_prediction_num_stars(ngs_zd, num_stars)
    started = time.perf_counter()
    x = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=ngs_zd,
        ngs_az_deg=ngs_az_deg,
        ngs_mag=ngs_mag,
        backend_row_count=backend_row_count,
        feature_buffer_key=id(model),
        feature_buffer_row_count=feature_buffer_row_count,
    )
    if prediction_telemetry is not None:
        prediction_telemetry.record_feature_batch(
            x,
            time.perf_counter() - started,
            row_count=row_count,
            record_bucket=backend_row_count is not None,
        )
        prediction_telemetry.record_mps_snapshot(model)
    started = time.perf_counter()
    y_pred = backend.get_prediction(x, model, skip_cache_clear=True)[:row_count]
    if prediction_telemetry is not None:
        prediction_telemetry.record_backend_batch(time.perf_counter() - started)
        prediction_telemetry.record_mps_snapshot(model)
    return PointPredictionBatch(
        sr=y_pred[:, backend.get_sr_index()],
        ee=y_pred[:, backend.get_ee_index()],
        fwhm=y_pred[:, backend.get_fwhm_index()],
        ee_angle=np.zeros((row_count,), dtype=np.float64),
    )


def predict_field_mean_arrays(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    model,
    ngs_zd: np.ndarray,
    ngs_az_deg: np.ndarray,
    ngs_mag: np.ndarray,
    backend_row_count: int | None = None,
    feature_buffer_row_count: int | None = None,
    prediction_telemetry: PredictionArrayTelemetry | None = None,
) -> np.ndarray:
    """Return mean-field EE predictions for one magnitude-ordered batch."""

    row_count = int(np.asarray(ngs_zd).shape[0])
    _validate_prediction_num_stars(ngs_zd, num_stars)
    started = time.perf_counter()
    x = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=ngs_zd,
        ngs_az_deg=ngs_az_deg,
        ngs_mag=ngs_mag,
        mean_only=True,
        backend_row_count=backend_row_count,
        feature_buffer_key=id(model),
        feature_buffer_row_count=feature_buffer_row_count,
    )
    if prediction_telemetry is not None:
        prediction_telemetry.record_feature_batch(
            x,
            time.perf_counter() - started,
            row_count=row_count,
            record_bucket=backend_row_count is not None,
        )
        prediction_telemetry.record_mps_snapshot(model)
    started = time.perf_counter()
    y_pred = backend.get_prediction(x, model, skip_cache_clear=True)[:row_count]
    if prediction_telemetry is not None:
        prediction_telemetry.record_backend_batch(time.perf_counter() - started)
        prediction_telemetry.record_mps_snapshot(model)
    return np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64)


def scale_seeing_performance(
    prediction_wavelength: u.Quantity,
    *,
    sr: float | None,
    ee: float | None,
    fwhm: float | None,
    reference_wavelength: u.Quantity,
) -> SeeingBaselinePerformance:
    """Scale seeing-limited fallback metrics to the prediction wavelength."""

    wavelength_ratio = (
        prediction_wavelength / reference_wavelength
    ).to(u.dimensionless_unscaled).value
    if sr is None:
        scaled_sr = None
    elif sr <= 0:
        scaled_sr = sr
    else:
        scaled_sr = float(np.power(sr, np.power(wavelength_ratio, -2.0)))
    scaled_ee = None if ee is None else float(ee * np.power(wavelength_ratio, 2.0 / 5.0))
    scaled_fwhm = (
        None if fwhm is None else float(fwhm * np.power(wavelength_ratio, -1.0 / 5.0))
    )
    return SeeingBaselinePerformance(
        sr=scaled_sr,
        ee=scaled_ee,
        fwhm=scaled_fwhm,
    )


def get_seeing_baseline_performance(runtime: PredictRuntime) -> SeeingBaselinePerformance:
    """Return the current seeing-limited fallback metrics."""

    return scale_seeing_performance(
        runtime.prediction_wavelength,
        sr=runtime.seeing_reference_sr,
        ee=runtime.seeing_reference_ee,
        fwhm=runtime.seeing_reference_fwhm,
        reference_wavelength=runtime.seeing_reference_wavelength,
    )
