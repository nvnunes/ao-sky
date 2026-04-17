"""Native prediction helpers backed temporarily by girmos-aosims."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np

from . import backend
from ._exceptions import PredictError
from ._models import PointPredictionBatch, PredictRuntime, SeeingBaselinePerformance

_ModelCacheKey = tuple[str, str, str, int]
_POINT_MODEL_CACHE: dict[_ModelCacheKey, object] = {}
_MEAN_MODEL_CACHE: dict[_ModelCacheKey, object] = {}


def _get_model_cache_key(
    prefix: str,
    runtime: PredictRuntime,
    *,
    model_name: str,
    num_stars: int,
) -> _ModelCacheKey:
    return (
        prefix,
        str(Path(runtime.model_root).expanduser().resolve()),
        model_name,
        int(num_stars),
    )


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


def get_point_model(runtime: PredictRuntime, num_stars: int):
    """Return the cached point model for one surviving guide-star count."""

    key = f"{int(num_stars)}star"
    model_name = runtime.resolved_models.get(key)
    if model_name is None:
        raise PredictError(f"Missing resolved model for {key}")

    cache_key = _get_model_cache_key(
        "point",
        runtime,
        model_name=model_name,
        num_stars=num_stars,
    )
    if cache_key not in _POINT_MODEL_CACHE:
        _POINT_MODEL_CACHE[cache_key] = backend.load_model(
            runtime.model_root,
            model_name,
            force_cpu=True,
        )
    return _POINT_MODEL_CACHE[cache_key]


def get_mean_model(runtime: PredictRuntime, num_stars: int):
    """Return the cached mean-field model for one surviving guide-star count."""

    key = f"{int(num_stars)}star"
    model_name = runtime.averaged_models.get(key)
    if model_name is None:
        raise PredictError(f"Missing averaged model for {key}")

    cache_key = _get_model_cache_key(
        "mean",
        runtime,
        model_name=model_name,
        num_stars=num_stars,
    )
    if cache_key not in _MEAN_MODEL_CACHE:
        _MEAN_MODEL_CACHE[cache_key] = backend.load_model(
            runtime.model_root,
            model_name,
            force_cpu=True,
        )
    return _MEAN_MODEL_CACHE[cache_key]


def warm_model_cache(runtime: PredictRuntime) -> None:
    """Load all configured temporary backend models for one AO runtime."""

    for num_stars in range(runtime.ao_system.min_wfs, runtime.ao_system.max_wfs + 1):
        get_point_model(runtime, num_stars)
        get_mean_model(runtime, num_stars)


def _get_ao_lgs_xy(lgs: tuple[dict[str, float], ...]) -> list[dict[str, float]]:
    return [
        {
            **star,
            "x": float(star["zd"]) * np.cos(np.deg2rad(float(star["az"]))),
            "y": float(star["zd"]) * np.sin(np.deg2rad(float(star["az"]))),
        }
        for star in lgs
    ]


def _build_prediction_payload(
    runtime: PredictRuntime,
    *,
    lgs,
    ngs: list[list[dict[str, float]]],
) -> dict[str, object]:
    num_rows = len(ngs)
    return {
        "wavelength": runtime.prediction_wavelength,
        "lgs": lgs,
        "ngs": ngs,
        "r": np.zeros((num_rows,), dtype=np.float64),
        "theta": np.zeros((num_rows,), dtype=np.float64),
        "x": np.zeros((num_rows,), dtype=np.float64),
        "y": np.zeros((num_rows,), dtype=np.float64),
    }


def predict_point_batch(
    runtime: PredictRuntime,
    *,
    num_stars: int,  # noqa: ARG001 - retained for backend-call symmetry.
    model,
    ngs: list[list[dict[str, float]]],
) -> PointPredictionBatch:
    """Return resolved SR/EE/FWHM predictions for one homogeneous batch."""

    num_pairs = len(ngs)
    payload = _build_prediction_payload(
        runtime,
        lgs=_get_ao_lgs_xy(runtime.ao_system.lgs),
        ngs=ngs,
    )
    X = backend.get_model_X(payload)
    y_pred = backend.get_prediction(X, model, skip_cache_clear=True)
    return PointPredictionBatch(
        sr=y_pred[:, backend.get_sr_index()],
        ee=y_pred[:, backend.get_ee_index()],
        fwhm=y_pred[:, backend.get_fwhm_index()],
        ee_angle=np.zeros((num_pairs,), dtype=np.float64),
    )


def predict_field_mean_batch(
    runtime: PredictRuntime,
    *,
    num_stars: int,  # noqa: ARG001 - retained for backend-call symmetry.
    model,
    ngs: list[list[dict[str, float]]],
) -> np.ndarray:
    """Return mean-field EE predictions for one homogeneous winner batch."""

    payload = _build_prediction_payload(
        runtime,
        lgs=list(runtime.ao_system.lgs),
        ngs=ngs,
    )
    X = backend.get_model_X(payload, mean_only=True)

    y_pred = backend.get_prediction(X, model, skip_cache_clear=True)
    return np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64)


def predict_asterism_ee(
    runtime: PredictRuntime,
    *,
    num_stars: int,  # noqa: ARG001 - retained for backend-call symmetry.
    model,
    ngs: list[list[dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean-model EE qualities and best angles for catalog ranking."""

    payload = _build_prediction_payload(
        runtime,
        lgs=list(runtime.ao_system.lgs),
        ngs=ngs,
    )
    X = backend.get_model_X(payload, mean_only=True)
    y_pred = backend.get_prediction(X, model)
    return (
        np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64),
        np.zeros((len(ngs),), dtype=np.float64),
    )


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
