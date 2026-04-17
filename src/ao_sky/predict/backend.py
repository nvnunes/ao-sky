"""Thin temporary adapter over `ao_tools.training` from girmos-aosims."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import sys

from ._exceptions import PredictError

INFERENCE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _get_training_module():
    try:
        return import_module("ao_tools.training")
    except ModuleNotFoundError as exc:
        sibling_repo = Path(__file__).resolve().parents[3].parent / "girmos-aosims"
        if sibling_repo.is_dir():
            sys.path.insert(0, str(sibling_repo))
            try:
                return import_module("ao_tools.training")
            except ModuleNotFoundError:
                pass
        raise PredictError(
            "AO prediction requires 'ao_tools.training', which is provided by "
            "girmos-aosims. Install girmos-aosims or keep a sibling "
            "'../girmos-aosims' checkout available to use the native Traversal "
            "prediction path."
        ) from exc


def load_model(model_root: Path, model_name: str, force_cpu: bool = True):
    """Load one TorchScript AO model from the resolved model root."""

    training = _get_training_module()
    return training.load_model(str(Path(model_root)), model_name, force_cpu=force_cpu)


def get_model_X(*args, **kwargs):
    """Delegate model-feature construction to `ao_tools.training`."""

    training = _get_training_module()
    return training.get_model_X(*args, **kwargs)


def get_prediction(*args, **kwargs):
    """Delegate model inference to `ao_tools.training`."""

    training = _get_training_module()
    return training.get_prediction(*args, **kwargs)


def get_sr_index() -> int:
    """Return the SR output index used by the current backend."""

    training = _get_training_module()
    return int(training.get_sr_index())


def get_ee_index() -> int:
    """Return the EE output index used by the current backend."""

    training = _get_training_module()
    return int(training.get_ee_index())


def get_fwhm_index() -> int:
    """Return the FWHM output index used by the current backend."""

    training = _get_training_module()
    return int(training.get_fwhm_index())


def clear_cache(model) -> None:
    """Release temporary backend cache state for one loaded model."""

    training = _get_training_module()
    training.clear_cache(model)


def configure_inference_threads(num_threads: int) -> None:
    """Pin backend inference libraries to a bounded thread count."""

    resolved = max(1, int(num_threads))
    for env_var in INFERENCE_THREAD_ENV_VARS:
        os.environ[env_var] = str(resolved)

    try:
        import torch

        torch.set_num_threads(resolved)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
