"""ELBO computation for SuSiE."""

from __future__ import annotations

import numpy as np

from ._types import _FitData, _ModelState


def compute_elbo(data: _FitData, state: _ModelState) -> float:
    """Compute the variational evidence lower bound."""
    sigma2 = float(state.sigma2)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")

    bbar = np.sum(state.alpha * state.mu, axis=0)
    diag_term = float(np.dot(data.d, np.sum(state.alpha * state.mu2, axis=0)))
    cross_correction = float(np.sum(state.Xb_sq_norms))

    if state.Xr is not None and data.y is not None:
        residual = data.y - state.Xr
        rss = float(np.dot(residual, residual) - cross_correction + diag_term)
    else:
        if state.XtXr is None:
            state.XtXr = data.compute_XtXb(bbar)
        rss = float(
            data.yty
            - 2.0 * np.dot(bbar, data.Xty)
            + np.dot(bbar, state.XtXr)
            - cross_correction
            + diag_term
        )

    rss = max(rss, 0.0)
    expected_ll = -0.5 * data.n * np.log(2.0 * np.pi * sigma2) - 0.5 * rss / sigma2
    return float(expected_ll - np.sum(state.KL))
