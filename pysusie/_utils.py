"""Shared numerical helpers for pysusie."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp


EPS = np.finfo(float).eps


def as_float_array(x: Any, *, copy: bool = False) -> np.ndarray:
    """Convert input to a float64 numpy array."""
    return np.array(x, dtype=float, copy=copy)


def softmax_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D log-weights."""
    log_weights = np.asarray(log_weights, dtype=float)
    return np.exp(log_weights - logsumexp(log_weights))


def log1mexp_from_probs(p: np.ndarray) -> np.ndarray:
    """Compute log(1 - p) safely for probabilities in [0, 1]."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0 - 1e-16)
    return np.log1p(-p)


def ensure_2d(x: Any, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return arr


def ensure_1d(x: Any, *, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def is_close_to_zero(x: float, *, atol: float = 1e-12) -> bool:
    return abs(float(x)) <= atol


def rng_from_seed(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def freeze_array(arr: np.ndarray | None) -> None:
    """Mark array as read-only in-place when provided."""
    if arr is not None and isinstance(arr, np.ndarray):
        arr.flags.writeable = False
