"""Wrappers for SuSiE-inf and SuSiE-ash style models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._mrash import fit_mrash
from ._types import MrASHResult, SuSiEResult
from ._utils import ensure_1d, ensure_2d


@dataclass(frozen=True)
class SuSiEInfResult:
    beta: np.ndarray
    sigma2: float
    tau2: float
    precision: np.ndarray | None = None
    eigen_values: np.ndarray | None = None
    eigen_vectors: np.ndarray | None = None


@dataclass(frozen=True)
class SuSiEAshResult:
    theta: np.ndarray
    sparse: np.ndarray
    coef: np.ndarray
    mrash: MrASHResult
    susie: SuSiEResult


def susie_inf_precision(X: np.ndarray, sigma2: float, tau2: float) -> np.ndarray:
    """Compute (sigma^2 I + tau^2 XX')^{-1} via eigendecomposition."""
    X = ensure_2d(X, name="X")
    sigma2 = max(float(sigma2), 1e-12)
    tau2 = max(float(tau2), 0.0)

    XXt = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(XXt)
    eigvals = np.maximum(eigvals, 0.0)
    inv_diag = 1.0 / (sigma2 + tau2 * eigvals)
    return (eigvecs * inv_diag) @ eigvecs.T


def susie_inf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tau2: float = 1e-3,
    sigma2: float | None = None,
    return_precision: bool = False,
) -> SuSiEInfResult:
    """Fit a simple SuSiE-inf style dense-effect model via ridge regression."""
    X = ensure_2d(X, name="X")
    y = ensure_1d(y, name="y")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y have incompatible shapes")

    if sigma2 is None:
        sigma2 = float(np.var(y, ddof=1))
    sigma2 = max(float(sigma2), 1e-12)
    tau2 = max(float(tau2), 1e-12)

    p = X.shape[1]
    beta = np.linalg.solve(X.T @ X + (sigma2 / tau2) * np.eye(p), X.T @ y)

    XXt = X @ X.T
    eigvals, eigvecs = np.linalg.eigh(XXt)
    eigvals = np.maximum(eigvals, 0.0)

    precision = None
    if return_precision:
        inv_diag = 1.0 / (sigma2 + tau2 * eigvals)
        precision = (eigvecs * inv_diag) @ eigvecs.T

    return SuSiEInfResult(
        beta=np.asarray(beta, dtype=float),
        sigma2=sigma2,
        tau2=tau2,
        precision=precision,
        eigen_values=eigvals,
        eigen_vectors=eigvecs,
    )


def susie_ash(
    X: np.ndarray,
    y: np.ndarray,
    sa2: np.ndarray,
    *,
    n_effects: int = 10,
    n_outer_iter: int = 2,
    mrash_kwargs: dict | None = None,
    susie_kwargs: dict | None = None,
) -> SuSiEAshResult:
    """Fit a two-stage SuSiE-ash approximation with sparse + background effects."""
    from .susie import SuSiE

    mrash_kwargs = dict(mrash_kwargs or {})
    susie_kwargs = dict(susie_kwargs or {})

    X_arr = ensure_2d(X, name="X")
    y_arr = ensure_1d(y, name="y")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("X and y have incompatible shapes")

    n_outer_iter = max(int(n_outer_iter), 1)
    sparse = np.zeros(X_arr.shape[1], dtype=float)
    theta = np.zeros(X_arr.shape[1], dtype=float)
    mrash = None
    model = None

    for _ in range(n_outer_iter):
        # Background polygenic fit, conditioning on current sparse effects.
        mrash = fit_mrash(X_arr, y_arr - X_arr @ sparse, sa2, **mrash_kwargs)
        theta = np.asarray(mrash.beta, dtype=float)

        # Sparse SuSiE fit, conditioning on current background.
        model = SuSiE(n_effects=n_effects, **susie_kwargs)
        model.fit(X_arr, y_arr - X_arr @ theta)
        sparse = np.asarray(model.result_.coef, dtype=float)

    assert mrash is not None
    assert model is not None
    coef = theta + sparse

    return SuSiEAshResult(
        theta=theta,
        sparse=sparse,
        coef=coef,
        mrash=mrash,
        susie=model.result_,
    )


# Backward-compatible aliases.
fit_susie_inf = susie_inf
fit_susie_ash = susie_ash


__all__ = [
    "SuSiEInfResult",
    "SuSiEAshResult",
    "susie_inf_precision",
    "susie_inf",
    "susie_ash",
    "fit_susie_inf",
    "fit_susie_ash",
]
