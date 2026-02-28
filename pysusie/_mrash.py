"""Mr.ASH implementation (numpy + optional numba acceleration)."""

from __future__ import annotations

import numpy as np

from ._numba_kernels import _mrash_loop
from ._types import MrASHResult
from ._utils import ensure_1d, ensure_2d


def fit_mrash(
    X: np.ndarray,
    y: np.ndarray,
    sa2: np.ndarray,
    *,
    pi: np.ndarray | None = None,
    beta_init: np.ndarray | None = None,
    sigma2: float | None = None,
    max_iter: int = 100,
    min_iter: int = 5,
    tol: float = 1e-3,
    update_pi: bool = True,
    update_sigma2: bool = True,
) -> MrASHResult:
    """Fit Mr.ASH model via coordinate-ascent variational EM."""
    X = ensure_2d(X, name="X").astype(float, copy=False)
    y = ensure_1d(y, name="y").astype(float, copy=False)
    sa2 = ensure_1d(sa2, name="sa2").astype(float, copy=False)

    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("X and y have incompatible shapes")
    if np.any(sa2 < 0):
        raise ValueError("sa2 must be non-negative")
    if sa2.size == 0:
        raise ValueError("sa2 must contain at least one mixture component")

    if pi is None:
        pi_arr = np.full(sa2.size, 1.0 / sa2.size)
    else:
        pi_arr = ensure_1d(pi, name="pi").astype(float, copy=True)
        if pi_arr.shape[0] != sa2.shape[0]:
            raise ValueError("pi must have shape (K,)")
        if np.any(pi_arr < 0):
            raise ValueError("pi must be non-negative")
        pi_mass = float(np.sum(pi_arr))
        if not np.isfinite(pi_mass) or pi_mass <= 0:
            raise ValueError("pi must contain positive mass")
        pi_arr /= pi_mass

    if beta_init is None:
        beta = np.zeros(p, dtype=float)
    else:
        beta = ensure_1d(beta_init, name="beta_init").astype(float, copy=True)
        if beta.shape[0] != p:
            raise ValueError("beta_init must have shape (p,)")

    if sigma2 is None:
        sigma2 = float(np.var(y, ddof=1))
    sigma2 = max(float(sigma2), 1e-12)

    w = np.sum(X * X, axis=0)
    r = y - X @ beta
    order = np.arange(p, dtype=np.int64)

    beta_out, pi_out, sigma2_out, elbo, converged, n_iter = _mrash_loop(
        X,
        w,
        sa2,
        pi_arr,
        beta,
        r,
        sigma2,
        order,
        int(max_iter),
        int(min_iter),
        float(tol),
        1e-12,
        bool(update_pi),
        bool(update_sigma2),
    )

    return MrASHResult(
        beta=np.asarray(beta_out, dtype=float),
        pi=np.asarray(pi_out, dtype=float),
        sigma2=float(sigma2_out),
        converged=bool(converged),
        n_iter=int(n_iter),
        elbo=np.asarray(elbo, dtype=float),
    )
