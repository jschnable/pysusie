"""Preprocessing and data conversion utilities for SuSiE."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal, MutableMapping

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar

from ._types import _FitData
from ._utils import ensure_1d, ensure_2d

_LD_EIG_CACHE_MAX_ENTRIES = 2
_LD_EIG_CACHE: OrderedDict[
    tuple[int, int, tuple[int, ...], tuple[int, ...], str],
    tuple[np.ndarray, np.ndarray],
] = OrderedDict()


def _matrix_cache_key(mat: np.ndarray) -> tuple[int, int, tuple[int, ...], tuple[int, ...], str]:
    arr = np.asarray(mat)
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    ptr = int(arr.__array_interface__["data"][0])
    return (id(base), ptr, arr.shape, arr.strides, arr.dtype.str)


def clear_ld_eigendecomp_cache() -> None:
    """Clear the module-level LD eigendecomposition cache."""
    _LD_EIG_CACHE.clear()


def _get_ld_eigendecomp(
    R_sym: np.ndarray,
    *,
    ld_eigendecomp_cache: MutableMapping[Any, tuple[np.ndarray, np.ndarray]] | None,
    cache_key_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cache = _LD_EIG_CACHE if ld_eigendecomp_cache is None else ld_eigendecomp_cache
    key_mat = R_sym if cache_key_matrix is None else cache_key_matrix
    key = _matrix_cache_key(key_mat)
    cached = cache.get(key)
    if cached is not None:
        if isinstance(cache, OrderedDict):
            cache.move_to_end(key)
        return cached

    eigvals, eigvecs = np.linalg.eigh(R_sym)
    cache[key] = (eigvals, eigvecs)

    if isinstance(cache, OrderedDict):
        while len(cache) > _LD_EIG_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    return eigvals, eigvecs


def _extract_feature_names(X: Any) -> list[str] | None:
    if hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    return None


def _to_dense_array(X: Any) -> np.ndarray:
    return np.asarray(X, dtype=float)


def _validate_no_nan(x: np.ndarray, name: str) -> None:
    if np.isnan(x).any():
        raise ValueError(f"{name} contains NaN values")


def preprocess_individual_data(
    X,
    y,
    *,
    standardize: bool = True,
    intercept: bool = True,
    null_weight: float = 0.0,
) -> _FitData:
    """Validate and preprocess individual-level data."""
    y_arr = ensure_1d(y, name="y").astype(float, copy=True)
    _validate_no_nan(y_arr, "y")

    if sp.issparse(X):
        X_mat = X.tocsr().astype(float)
        n, p_phys = X_mat.shape
    else:
        X_mat = ensure_2d(X, name="X").astype(float, copy=True)
        n, p_phys = X_mat.shape

    if y_arr.shape[0] != n:
        raise ValueError("X and y have incompatible shapes")
    if n <= 1:
        raise ValueError("X must have at least 2 rows")

    y_mean = float(np.mean(y_arr)) if intercept else 0.0
    y_centered = y_arr - y_mean if intercept else y_arr
    yty = float(np.dot(y_centered, y_centered))

    if sp.issparse(X_mat):
        means = np.asarray(X_mat.mean(axis=0)).ravel() if intercept else np.zeros(p_phys)
        col_sq = np.asarray(X_mat.power(2).sum(axis=0)).ravel()
        centered_ss = np.maximum(col_sq - n * means**2, 0.0)

        if standardize:
            scale = np.sqrt(centered_ss / max(n - 1, 1))
            if np.any(scale <= 0):
                raise ValueError("X contains constant columns")
        else:
            scale = np.ones(p_phys)
            if np.any(centered_ss <= 0):
                raise ValueError("X contains constant columns")

        d = col_sq / (scale**2) - n * (means / scale) ** 2
        Xty = np.asarray((X_mat.T @ y_centered) / scale).ravel()
        Xty -= (means / scale) * float(np.sum(y_centered))

        X_store = X_mat
        X_center = means
        X_scale = scale
        is_sparse = True

    else:
        _validate_no_nan(X_mat, "X")
        means = np.mean(X_mat, axis=0) if intercept else np.zeros(p_phys)
        X_centered = X_mat - means if intercept else X_mat.copy()

        if standardize:
            scale = np.std(X_centered, axis=0, ddof=1)
            if np.any(scale <= 0):
                raise ValueError("X contains constant columns")
        else:
            scale = np.ones(p_phys)
            ss = np.sum(X_centered**2, axis=0)
            if np.any(ss <= 0):
                raise ValueError("X contains constant columns")

        X_trans = X_centered / scale
        d = np.sum(X_trans**2, axis=0)
        Xty = np.asarray(X_trans.T @ y_centered, dtype=float)

        X_store = np.asfortranarray(X_trans)
        X_center = means
        X_scale = scale
        is_sparse = False

    if np.any(d <= 0):
        raise ValueError("X contains constant columns after preprocessing")

    has_null = null_weight > 0
    if has_null:
        d = np.concatenate([d, np.array([0.0])])
        Xty = np.concatenate([Xty, np.array([0.0])])

    p = d.shape[0]
    return _FitData(
        XtX=None,
        Xty=Xty,
        yty=yty,
        d=d,
        n=n,
        p=p,
        X=X_store,
        y=y_centered,
        X_center=X_center,
        X_scale=X_scale,
        y_mean=y_mean,
        is_sparse=is_sparse,
        has_null_column=has_null,
        eigen_values=None,
        eigen_vectors=None,
        regularization=0.0,
    )


def preprocess_sufficient_stats(
    XtX,
    Xty,
    yty,
    n,
    *,
    standardize: bool = True,
    check_psd: bool = False,
    maf: np.ndarray | None = None,
    null_weight: float = 0.0,
) -> _FitData:
    """Validate and preprocess sufficient statistics."""
    XtX = ensure_2d(XtX, name="XtX").astype(float, copy=True)
    Xty = ensure_1d(Xty, name="Xty").astype(float, copy=True)

    if XtX.shape[0] != XtX.shape[1]:
        raise ValueError("XtX must be square")
    if XtX.shape[0] != Xty.shape[0]:
        raise ValueError("XtX and Xty dimension mismatch")
    if n <= 1:
        raise ValueError("n must be > 1")

    XtX = 0.5 * (XtX + XtX.T)

    if maf is not None:
        maf = np.asarray(maf, dtype=float)
        if maf.shape != (Xty.shape[0],):
            raise ValueError("maf must have shape (p,)")
        keep = maf > 0
        XtX = XtX[np.ix_(keep, keep)]
        Xty = Xty[keep]

    p_phys = Xty.shape[0]

    if standardize:
        diag = np.diag(XtX)
        csd = np.sqrt(diag / (n - 1))
        if np.any(csd <= 0):
            raise ValueError("XtX has non-positive diagonal entries")
        XtX = XtX / np.outer(csd, csd)
        Xty = Xty / csd
        d = np.diag(XtX)
        X_scale = csd
    else:
        d = np.diag(XtX)
        if np.any(d <= 0):
            raise ValueError("XtX has non-positive diagonal entries")
        X_scale = np.ones(p_phys)

    if check_psd:
        ev = np.linalg.eigvalsh(XtX)
        if np.min(ev) < -1e-8:
            raise ValueError("XtX is not positive semi-definite")

    has_null = null_weight > 0
    if has_null:
        XtX_aug = np.zeros((p_phys + 1, p_phys + 1), dtype=float)
        XtX_aug[:-1, :-1] = XtX
        XtX = XtX_aug
        Xty = np.concatenate([Xty, np.array([0.0])])
        d = np.concatenate([d, np.array([0.0])])

    return _FitData(
        XtX=XtX,
        Xty=Xty,
        yty=float(yty),
        d=d,
        n=int(n),
        p=d.shape[0],
        X=None,
        y=None,
        X_center=np.zeros(p_phys),
        X_scale=X_scale,
        y_mean=0.0,
        is_sparse=False,
        has_null_column=has_null,
        eigen_values=None,
        eigen_vectors=None,
        regularization=0.0,
    )


def estimate_ld_regularization(
    z,
    R,
    n=None,
    *,
    return_eigendecomp: bool = False,
    ld_eigendecomp_cache: MutableMapping[Any, tuple[np.ndarray, np.ndarray]] | None = None,
):
    """Estimate lambda in R_tilde = (1-lambda)R + lambda*I by ML.

    Uses a single eigendecomposition of R so objective evaluations are O(p).
    """
    z = ensure_1d(z, name="z")
    R = ensure_2d(R, name="R")
    if R.shape[0] != R.shape[1] or R.shape[0] != z.shape[0]:
        raise ValueError("R must be square and match z dimension")

    R_sym = 0.5 * (R + R.T)
    eigvals, eigvecs = _get_ld_eigendecomp(
        R_sym,
        ld_eigendecomp_cache=ld_eigendecomp_cache,
        cache_key_matrix=R,
    )
    z_rot = eigvecs.T @ z
    z2 = z_rot * z_rot

    def neg_loglik(lam: float) -> float:
        denom = (1.0 - lam) * eigvals + lam
        if np.any(denom <= 1e-12):
            return np.inf
        logdet = float(np.sum(np.log(denom)))
        quad = float(np.sum(z2 / denom))
        return 0.5 * (logdet + quad)

    opt = minimize_scalar(neg_loglik, bounds=(0.0, 1.0), method="bounded")
    lam = float(np.clip(opt.x, 0.0, 1.0)) if opt.success else 0.0

    if return_eigendecomp:
        return lam, eigvals, eigvecs
    return lam


def preprocess_summary_stats(
    z,
    R,
    n,
    *,
    bhat=None,
    shat=None,
    var_y: float = 1.0,
    regularize_ld: float | Literal["auto"] = 0.0,
    null_weight: float = 0.0,
    ld_eigendecomp_cache: MutableMapping[Any, tuple[np.ndarray, np.ndarray]] | None = None,
) -> _FitData:
    """Convert summary statistics to sufficient statistics form."""
    R = ensure_2d(R, name="R")
    if R.shape[0] != R.shape[1]:
        raise ValueError("R must be square")
    var_y = float(var_y)
    if (not np.isfinite(var_y)) or var_y <= 0:
        raise ValueError("var_y must be positive and finite")

    if z is None:
        if bhat is None or shat is None:
            raise ValueError("Provide either z or both bhat and shat")
        bhat = ensure_1d(bhat, name="bhat")
        shat = ensure_1d(shat, name="shat")
        if bhat.shape != shat.shape:
            raise ValueError("bhat and shat must have same shape")
        z_arr = bhat / shat
    else:
        z_arr = ensure_1d(z, name="z")

    if z_arr.shape[0] != R.shape[0]:
        raise ValueError("z and R dimensions do not match")

    if n is not None:
        n = int(n)
        if n <= 1:
            raise ValueError("n must be > 1 when provided")
        adj = (n - 1) / (z_arr**2 + n - 2)
        z_adj = np.sqrt(adj) * z_arr
    else:
        z_adj = z_arr.copy()

    eigvals = None
    eigvecs = None
    R_sym = 0.5 * (R + R.T)

    if regularize_ld == "auto":
        if n is None:
            raise ValueError("regularize_ld='auto' requires n")
        lam, eigvals_R, eigvecs_R = estimate_ld_regularization(
            z_adj,
            R,
            n=n,
            return_eigendecomp=True,
            ld_eigendecomp_cache=ld_eigendecomp_cache,
        )
        eigvals = np.maximum((1.0 - lam) * eigvals_R + lam, 1e-10)
        eigvecs = eigvecs_R
    else:
        lam = float(regularize_ld)
        if not (0.0 <= lam <= 1.0):
            raise ValueError("regularize_ld must be in [0, 1]")
        if lam > 0.0:
            eigvals_R, eigvecs = _get_ld_eigendecomp(
                R_sym,
                ld_eigendecomp_cache=ld_eigendecomp_cache,
                cache_key_matrix=R,
            )
            eigvals = np.maximum((1.0 - lam) * eigvals_R + lam, 1e-10)

    R_tilde = (1.0 - lam) * R_sym + lam * np.eye(R.shape[0])

    if n is not None:
        XtX = (n - 1) * R_tilde
        Xty = np.sqrt(n - 1) * z_adj * np.sqrt(var_y)
        yty = float((n - 1) * var_y)
        n_eff = n
    else:
        XtX = R_tilde
        Xty = z_adj * np.sqrt(var_y)
        yty = float(var_y)
        n_eff = 2

    XtX = 0.5 * (XtX + XtX.T)
    d = np.diag(XtX)

    has_null = null_weight > 0
    if has_null:
        XtX_aug = np.zeros((XtX.shape[0] + 1, XtX.shape[1] + 1), dtype=float)
        XtX_aug[:-1, :-1] = XtX
        XtX = XtX_aug
        Xty = np.concatenate([Xty, np.array([0.0])])
        d = np.concatenate([d, np.array([0.0])])

    return _FitData(
        XtX=XtX,
        Xty=Xty,
        yty=yty,
        d=d,
        n=n_eff,
        p=d.shape[0],
        X=None,
        y=None,
        X_center=np.zeros(R.shape[0]),
        X_scale=np.ones(R.shape[0]),
        y_mean=0.0,
        is_sparse=False,
        has_null_column=has_null,
        eigen_values=eigvals,
        eigen_vectors=eigvecs,
        regularization=lam,
    )


def compute_sufficient_stats(X, y, *, standardize: bool = True):
    """Compute sufficient statistics from individual-level data."""
    data = preprocess_individual_data(X, y, standardize=standardize, intercept=True)

    if data.is_sparse:
        X_raw = data.X
        assert X_raw is not None
        gram = (X_raw.T @ X_raw).toarray()
        scale = data.X_scale
        center = data.X_center
        XtX = gram / np.outer(scale, scale)
        XtX -= data.n * np.outer(center / scale, center / scale)
    else:
        XtX = np.asarray(data.X.T @ data.X, dtype=float)

    return {
        "XtX": XtX,
        "Xty": data.Xty,
        "yty": data.yty,
        "n": data.n,
        "X_col_means": data.X_center,
        "y_mean": data.y_mean,
    }


def univariate_regression(X, y, *, center: bool = True, scale: bool = False):
    """Fit marginal univariate regressions for each column of X."""
    X_arr = ensure_2d(X, name="X").astype(float, copy=True)
    y_arr = ensure_1d(y, name="y").astype(float, copy=True)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y have incompatible shapes")

    n = X_arr.shape[0]
    if center:
        X_arr -= np.mean(X_arr, axis=0)
        y_arr -= np.mean(y_arr)

    if scale:
        s = np.std(X_arr, axis=0, ddof=1)
        if np.any(s <= 0):
            raise ValueError("X contains constant columns")
        X_arr /= s

    xtx = np.sum(X_arr**2, axis=0)
    xty = X_arr.T @ y_arr
    betahat = np.zeros_like(xty)
    valid = xtx > 0
    betahat[valid] = xty[valid] / xtx[valid]

    yty = float(np.dot(y_arr, y_arr))
    rss = yty - betahat * xty
    sigma2 = np.maximum(rss / max(n - 2, 1), 0.0)

    se = np.full_like(betahat, np.inf)
    se[valid] = np.sqrt(sigma2[valid] / xtx[valid])
    z_scores = np.zeros_like(betahat)
    nz = se > 0
    z_scores[nz] = betahat[nz] / se[nz]

    return {"betahat": betahat, "sebetahat": se, "z_scores": z_scores}
