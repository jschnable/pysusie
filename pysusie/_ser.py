"""Single-effect regression (SER) update for SuSiE."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ._types import SERResult
from ._utils import EPS, softmax_log_weights


def _compute_lbf(betahat: np.ndarray, shat2: np.ndarray, V: float) -> np.ndarray:
    """Compute per-variable log Bayes factors for SER."""
    lbf = np.zeros_like(betahat, dtype=float)
    if V <= 0:
        return lbf

    mask = np.isfinite(shat2) & (shat2 > 0)
    if not np.any(mask):
        return lbf

    s2 = shat2[mask]
    b = betahat[mask]

    denom = s2 * (s2 + V)
    term1 = -0.5 * np.log1p(V / s2)
    term2 = 0.5 * (b * b) * V / denom
    lbf[mask] = term1 + term2
    return lbf


def _optimize_prior_variance(
    betahat: np.ndarray,
    shat2: np.ndarray,
    prior_weights: np.ndarray,
    V_init: float,
    method: str,
) -> float:
    """Empirical-Bayes estimate of SER prior variance."""
    betahat = np.asarray(betahat, dtype=float)
    shat2 = np.asarray(shat2, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)

    if np.any(prior_weights < 0):
        raise ValueError("prior_weights must be non-negative")
    if not np.isclose(np.sum(prior_weights), 1.0):
        prior_weights = prior_weights / np.sum(prior_weights)

    log_pi = np.log(np.clip(prior_weights, 1e-300, None))

    finite_mask = np.isfinite(shat2) & (shat2 > 0)
    if not np.any(finite_mask):
        return 0.0

    def log_marginal(V: float) -> float:
        if V <= 0:
            return 0.0
        lbf = _compute_lbf(betahat, shat2, V)
        return float(logsumexp(log_pi + lbf))

    method = method.lower()
    if method == "simple":
        v0 = max(float(V_init), 0.0)
        return v0 if log_marginal(v0) >= log_marginal(0.0) else 0.0

    if method == "em":
        v0 = max(float(V_init), 1e-12)
        lbf = _compute_lbf(betahat, shat2, v0)
        alpha = softmax_log_weights(log_pi + lbf)

        mu2 = np.zeros_like(alpha)
        mask = np.isfinite(shat2) & (shat2 > 0)
        if np.any(mask):
            post_mean = (v0 / (v0 + shat2[mask])) * betahat[mask]
            post_var = v0 * shat2[mask] / (v0 + shat2[mask])
            mu2[mask] = post_var + post_mean**2

        return float(max(np.sum(alpha * mu2), 0.0))

    if method != "optim":
        raise ValueError("prior_variance_method must be one of {'optim', 'em', 'simple'}")

    def objective(log_v: float) -> float:
        V = float(np.exp(log_v))
        return -log_marginal(V)

    opt = minimize_scalar(
        objective,
        bounds=(-30.0, 15.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not opt.success:
        return max(float(V_init), 0.0)
    return float(np.exp(opt.x))


def fit_ser(
    Xty_residual: np.ndarray,
    d: np.ndarray,
    sigma2: float,
    prior_variance: float,
    prior_weights: np.ndarray,
    estimate_prior_variance: bool,
    prior_variance_method: str,
    check_null_threshold: float = 1e-9,
) -> SERResult:
    """Fit the single-effect regression (SER) update."""
    Xty_residual = np.asarray(Xty_residual, dtype=float)
    d = np.asarray(d, dtype=float)
    prior_weights = np.asarray(prior_weights, dtype=float)

    if Xty_residual.shape != d.shape:
        raise ValueError("Xty_residual and d must have same shape")
    if prior_weights.shape != d.shape:
        raise ValueError("prior_weights must have shape (p,)")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")

    prior_weights = np.clip(prior_weights, 0.0, None)
    if prior_weights.sum() <= 0:
        raise ValueError("prior_weights must contain positive mass")
    prior_weights = prior_weights / prior_weights.sum()

    p = d.size
    betahat = np.zeros(p, dtype=float)
    shat2 = np.full(p, np.inf, dtype=float)

    active = d > 0
    if np.any(active):
        betahat[active] = Xty_residual[active] / d[active]
        shat2[active] = sigma2 / d[active]

    V = float(max(prior_variance, 0.0))
    if estimate_prior_variance:
        V = _optimize_prior_variance(
            betahat=betahat,
            shat2=shat2,
            prior_weights=prior_weights,
            V_init=V,
            method=prior_variance_method,
        )

    if V <= check_null_threshold:
        V = 0.0
        lbf_variable = np.zeros(p, dtype=float)
        alpha = prior_weights.copy()
        mu = np.zeros(p, dtype=float)
        mu2 = np.zeros(p, dtype=float)
        lbf = 0.0
        KL = 0.0
        return SERResult(alpha=alpha, mu=mu, mu2=mu2, lbf=lbf, lbf_variable=lbf_variable, V=V, KL=KL)

    lbf_variable = _compute_lbf(betahat, shat2, V)
    log_prior = np.log(np.clip(prior_weights, 1e-300, None))
    log_alpha = log_prior + lbf_variable
    log_norm = logsumexp(log_alpha)
    log_alpha -= log_norm
    alpha = np.exp(log_alpha)

    mu = np.zeros(p, dtype=float)
    mu2 = np.zeros(p, dtype=float)

    if np.any(active):
        denom = sigma2 + V * d[active]
        mu[active] = V * Xty_residual[active] / denom
        post_var = sigma2 * V / denom
        mu2[active] = post_var + mu[active] ** 2

    lbf = float(log_norm)

    # KL(q||p) for the SER variational factor.
    kl = float(np.sum(alpha * (np.log(np.clip(alpha, 1e-300, None)) - log_prior)))
    if np.any(active):
        s2_post = sigma2 * V / (sigma2 + V * d[active])
        kl_normal = 0.5 * (
            (s2_post + mu[active] ** 2) / V - 1.0 + np.log(np.clip(V / s2_post, 1e-300, None))
        )
        kl += float(np.sum(alpha[active] * kl_normal))

    return SERResult(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        lbf=lbf,
        lbf_variable=lbf_variable,
        V=float(V),
        KL=kl,
    )
