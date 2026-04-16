"""Prediction-backend and runtime exceptions."""

from __future__ import annotations


class PredictError(RuntimeError):
    """Raised when AO prediction runtime setup or execution fails."""

