"""Native prediction helpers backed temporarily by girmos-aosims."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import astropy.units as u
import numpy as np

from . import backend
from ._models import PointPredictionBatch, PredictRuntime, SeeingBaselinePerformance

_ModelCacheKey = tuple[str, str, str, str, int]
_POINT_MODEL_CACHE: dict[_ModelCacheKey, object] = {}
_MEAN_MODEL_CACHE: dict[_ModelCacheKey, object] = {}
_ZERO_ROTATION_ANGLES: Final[tuple[float, ...]] = (0.0,)


def _get_model_cache_key(
    prefix: str,
    runtime: PredictRuntime,
    *,
    model_name: str,
    num_stars: int,
) -> _ModelCacheKey:
    return (
        prefix,
        runtime.ao_system.name,
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
    model_name = runtime.ao_system.point_models.get(key)
    if model_name is None:
        return None

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
    model_name = runtime.ao_system.mean_models.get(key)
    if model_name is None:
        return None

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


def get_rotation_angles(runtime: PredictRuntime) -> tuple[float, ...]:
    """Return the sampled instrument rotation angles for one AO runtime."""

    if runtime.ao_system.rot_range is None or runtime.ao_system.rot_step is None:
        return _ZERO_ROTATION_ANGLES
    start, stop = runtime.ao_system.rot_range
    return tuple(
        [0.0]
        + [
            float(angle)
            for angle in np.arange(start, stop, runtime.ao_system.rot_step)
            if angle != 0
        ]
    )


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
    num_stars: int,
    model,
    ngs: list[list[dict[str, float]]],
) -> PointPredictionBatch:
    """Return resolved SR/EE/FWHM predictions for one homogeneous batch."""

    theta_idxs = backend.get_ngs_theta_indexes(num_stars)
    rot_angles = get_rotation_angles(runtime)
    num_pairs = len(ngs)
    payload = _build_prediction_payload(
        runtime,
        lgs=_get_ao_lgs_xy(runtime.ao_system.lgs),
        ngs=ngs,
    )
    X = backend.get_model_X(payload)

    if len(rot_angles) == 1:
        y_pred = backend.get_prediction(X, model, skip_cache_clear=True)
        return PointPredictionBatch(
            sr=y_pred[:, backend.get_sr_index()],
            ee=y_pred[:, backend.get_ee_index()],
            fwhm=y_pred[:, backend.get_fwhm_index()],
            ee_angle=np.zeros((num_pairs,), dtype=np.float64),
        )

    X_rot = np.tile(X, (len(rot_angles), 1))
    for rot_idx, rot_angle in enumerate(rot_angles):
        if rot_angle == 0:
            continue
        start_idx = rot_idx * num_pairs
        end_idx = start_idx + num_pairs
        X_rot[start_idx:end_idx, theta_idxs] += np.deg2rad(rot_angle)
        X_rot[start_idx:end_idx, theta_idxs] = backend.wrap_angle_rad(
            X_rot[start_idx:end_idx, theta_idxs]
        )

    y_pred = backend.get_prediction(X_rot, model, skip_cache_clear=True)
    y_pred = y_pred.reshape(len(rot_angles), num_pairs, y_pred.shape[1])
    ee_values = y_pred[:, :, backend.get_ee_index()]
    best_ee_rot_idxs = np.argmax(ee_values, axis=0)
    return PointPredictionBatch(
        sr=np.max(y_pred[:, :, backend.get_sr_index()], axis=0),
        ee=ee_values[best_ee_rot_idxs, np.arange(num_pairs)],
        fwhm=np.min(y_pred[:, :, backend.get_fwhm_index()], axis=0),
        ee_angle=np.asarray(rot_angles, dtype=np.float64)[best_ee_rot_idxs],
    )


def predict_field_mean_batch(
    runtime: PredictRuntime,
    *,
    num_stars: int,
    model,
    ngs: list[list[dict[str, float]]],
    rot_angles: list[float] | np.ndarray,
) -> np.ndarray:
    """Return mean-field EE predictions for one homogeneous winner batch."""

    payload = _build_prediction_payload(
        runtime,
        lgs=list(runtime.ao_system.lgs),
        ngs=ngs,
    )
    X = backend.get_model_X(payload, mean_only=True)
    theta_idxs = backend.get_ngs_theta_indexes(num_stars)
    if len(theta_idxs) > 0:
        X[:, theta_idxs] += np.deg2rad(
            np.asarray(rot_angles, dtype=np.float64)
        ).reshape(-1, 1)
        X[:, theta_idxs] = backend.wrap_angle_rad(X[:, theta_idxs])

    y_pred = backend.get_prediction(X, model, skip_cache_clear=True)
    return np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64)


def predict_asterism_ee(
    runtime: PredictRuntime,
    *,
    num_stars: int,
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
    rot_angles = get_rotation_angles(runtime)

    if len(rot_angles) == 1:
        y_pred = backend.get_prediction(X, model)
        return (
            np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64),
            np.zeros((len(ngs),), dtype=np.float64),
        )

    theta_idxs = backend.get_ngs_theta_indexes(num_stars)
    qualities = np.zeros((len(ngs),), dtype=np.float64)
    best_angles = np.zeros((len(ngs),), dtype=np.float64)
    last_rot_angle = 0.0
    for rot_angle in rot_angles:
        delta_theta = np.deg2rad(rot_angle - last_rot_angle)
        if delta_theta != 0:
            X[:, theta_idxs] += delta_theta
            X[:, theta_idxs] = backend.wrap_angle_rad(X[:, theta_idxs])
        y_pred = backend.get_prediction(X, model)
        ee = np.asarray(y_pred[:, backend.get_ee_index()], dtype=np.float64)
        best_angles[ee > qualities] = rot_angle
        qualities = np.maximum(qualities, ee)
        last_rot_angle = float(rot_angle)
    return qualities, best_angles


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
