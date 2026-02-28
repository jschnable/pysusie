"""Credible set extraction and purity metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._types import PurityMetrics


def _upper_triangle_values(a: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(a.shape[0], k=1)
    return a[iu]


def extract_credible_sets(
    alpha: np.ndarray,
    prior_variance: np.ndarray,
    *,
    coverage: float = 0.95,
    prior_tol: float = 1e-9,
    lbf: np.ndarray | None = None,
) -> list[tuple[np.ndarray, float, int, float]]:
    """Extract and deduplicate credible sets from effect-wise posteriors."""
    alpha = np.asarray(alpha, dtype=float)
    prior_variance = np.asarray(prior_variance, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape (L, p)")
    if prior_variance.shape != (alpha.shape[0],):
        raise ValueError("prior_variance must have shape (L,)")
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")

    if lbf is None:
        lbf = np.zeros(alpha.shape[0])
    lbf = np.asarray(lbf, dtype=float)

    best_by_signature: dict[tuple[int, ...], tuple[np.ndarray, float, int, float]] = {}

    for effect_idx in range(alpha.shape[0]):
        if prior_variance[effect_idx] <= prior_tol:
            continue

        row = alpha[effect_idx]
        order = np.argsort(-row)
        cum = np.cumsum(row[order])
        # Numerical tolerance avoids adding an extra variable when cumulative
        # coverage is equal to target up to floating-point error.
        k = int(np.searchsorted(cum + 1e-12, coverage, side="left")) + 1
        selected = order[:k]
        achieved = float(cum[k - 1])

        signature = tuple(sorted(int(i) for i in selected))
        entry = (
            np.array(signature, dtype=int),
            achieved,
            effect_idx,
            float(lbf[effect_idx]),
        )

        prev = best_by_signature.get(signature)
        if prev is None or entry[3] > prev[3]:
            best_by_signature[signature] = entry

    out = list(best_by_signature.values())
    out.sort(key=lambda t: t[2])
    return out


def _subset_variables(cs_variables: np.ndarray, n_purity: int) -> np.ndarray:
    cs_variables = np.asarray(cs_variables, dtype=int)
    if cs_variables.size <= n_purity:
        return cs_variables
    # Deterministic subsampling avoids stochastic unit tests.
    idx = np.linspace(0, cs_variables.size - 1, n_purity, dtype=int)
    return cs_variables[idx]


def compute_purity(
    cs_variables: np.ndarray,
    X: np.ndarray | None = None,
    R: np.ndarray | None = None,
    n_purity: int = 100,
) -> PurityMetrics:
    """Compute min/mean/median absolute correlations within a credible set."""
    vars_idx = _subset_variables(np.asarray(cs_variables, dtype=int), int(n_purity))
    if vars_idx.size <= 1:
        return PurityMetrics(1.0, 1.0, 1.0)

    if R is None and X is None:
        raise ValueError("Either X or R must be provided to compute purity")

    if R is not None:
        R = np.asarray(R, dtype=float)
        corr_abs = np.abs(R[np.ix_(vars_idx, vars_idx)])
    else:
        X = np.asarray(X, dtype=float)
        sub = X[:, vars_idx]
        corr = np.corrcoef(sub, rowvar=False)
        if np.ndim(corr) == 0:
            corr = np.array([[1.0]], dtype=float)
        corr = np.nan_to_num(corr, nan=0.0)
        corr_abs = np.abs(corr)

    vals = _upper_triangle_values(corr_abs)
    if vals.size == 0:
        return PurityMetrics(1.0, 1.0, 1.0)

    return PurityMetrics(
        min_abs_corr=float(np.min(vals)),
        mean_abs_corr=float(np.mean(vals)),
        median_abs_corr=float(np.median(vals)),
    )
