"""IBSS coordinate-ascent loop for SuSiE."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._elbo import compute_elbo
from ._ser import fit_ser
from ._types import _FitData, _ModelState


def _expected_rss(data: _FitData, state: _ModelState) -> float:
    bbar = np.sum(state.alpha * state.mu, axis=0)
    diag_term = float(np.dot(data.d, np.sum(state.alpha * state.mu2, axis=0)))
    correction = float(np.sum(state.Xb_sq_norms))

    if state.Xr is not None and data.y is not None:
        residual = data.y - state.Xr
        rss = float(np.dot(residual, residual) - correction + diag_term)
    else:
        if state.XtXr is None:
            state.XtXr = data.compute_XtXb(bbar)
        rss = float(
            data.yty
            - 2.0 * np.dot(bbar, data.Xty)
            + np.dot(bbar, state.XtXr)
            - correction
            + diag_term
        )
    return max(rss, 0.0)


def _estimate_residual_variance(data: _FitData, state: _ModelState, method: str) -> float:
    rss = _expected_rss(data, state)
    method = method.lower()
    if method == "mom":
        denom = max(data.n - 1, 1)
    elif method == "mle":
        denom = max(data.n, 1)
    else:
        raise ValueError("residual_variance_method must be one of {'mom', 'mle'}")
    return max(rss / denom, 1e-12)


def _initialize_caches(data: _FitData, state: _ModelState) -> None:
    L = state.alpha.shape[0]
    if data.X is not None and data.y is not None:
        state.XtXr = None
        state.XtXb_effects = None
        state.Xr = np.zeros(data.n, dtype=float)
        for l in range(L):
            bl = state.alpha[l] * state.mu[l]
            Xbl = data.compute_Xb(bl)
            state.Xr += Xbl
            state.Xb_sq_norms[l] = float(np.dot(Xbl, Xbl))
    else:
        state.Xr = None
        B = state.alpha * state.mu
        XtXB = data.compute_XtXB(B)
        state.XtXb_effects = XtXB
        state.XtXr = np.sum(XtXB, axis=0)
        state.Xb_sq_norms[:] = np.sum(B * XtXB, axis=1)


def _select_prior_variance_method(it: int, max_iter: int, params: dict[str, Any]) -> str:
    base_method = str(params.get("prior_variance_method", "optim")).lower()
    if base_method != "optim":
        return base_method

    warmup = max(int(params.get("prior_variance_warmup_iters", 1)), 0)
    period = max(int(params.get("prior_variance_optim_period", 5)), 1)
    force_final = bool(params.get("prior_variance_force_final_optim", True))

    run_optim = False
    if it >= warmup and ((it - warmup) % period == 0):
        run_optim = True
    if force_final and it == max_iter - 1:
        run_optim = True
    return "optim" if run_optim else "em"


def _ibss_loop_ss_batched(
    data: _FitData,
    state: _ModelState,
    *,
    prior_weights: np.ndarray,
    max_iter: int,
    tol: float,
    criterion: str,
    verbose: bool,
    estimate_prior_variance: bool,
    params: dict[str, Any],
) -> tuple[_ModelState, list[float], bool]:
    """Run SS/RSS updates with block-wise batched XtX @ B recomputation."""
    L, _ = state.alpha.shape
    batch_size = max(int(params.get("xtx_batch_size", 2)), 1)

    elbo_trace: list[float] = []
    converged = False
    prev_pip = None

    for it in range(max_iter):
        prior_variance_method = _select_prior_variance_method(it, max_iter, params)

        if state.XtXr is None or state.XtXb_effects is None:
            raise ValueError("Internal XtX caches are not initialized for SS/RSS path")
        XtXr_cur = np.asarray(state.XtXr, dtype=float).copy()
        XtXb_cur = np.asarray(state.XtXb_effects, dtype=float).copy()

        for start in range(0, L, batch_size):
            stop = min(start + batch_size, L)
            idx = slice(start, stop)

            for l in range(start, stop):
                Xty_residual = data.Xty - (XtXr_cur - XtXb_cur[l])

                ser = fit_ser(
                    Xty_residual=Xty_residual,
                    d=data.d,
                    sigma2=state.sigma2,
                    prior_variance=state.V[l],
                    prior_weights=prior_weights,
                    estimate_prior_variance=estimate_prior_variance,
                    prior_variance_method=prior_variance_method,
                    check_null_threshold=float(params.get("check_null_threshold", 1e-9)),
                )

                state.alpha[l] = ser.alpha
                state.mu[l] = ser.mu
                state.mu2[l] = ser.mu2
                state.V[l] = ser.V
                state.KL[l] = ser.KL
                state.lbf[l] = ser.lbf
                state.lbf_variable[l] = ser.lbf_variable

            B_new_block = state.alpha[idx] * state.mu[idx]
            XtXb_new_block = data.compute_XtXB(B_new_block)
            XtXr_cur += np.sum(XtXb_new_block - XtXb_cur[idx], axis=0)
            XtXb_cur[idx] = XtXb_new_block
            state.Xb_sq_norms[idx] = np.sum(B_new_block * XtXb_new_block, axis=1)

        state.XtXb_effects = XtXb_cur
        state.XtXr = XtXr_cur

        elbo = compute_elbo(data, state)
        elbo_trace.append(elbo)

        if params.get("estimate_residual_variance", True):
            state.sigma2 = _estimate_residual_variance(
                data,
                state,
                method=str(params.get("residual_variance_method", "mom")),
            )

        if verbose:
            print(f"IBSS iter={it + 1}, ELBO={elbo:.6f}, sigma2={state.sigma2:.6f}")

        if criterion == "elbo":
            if len(elbo_trace) >= 2:
                delta = elbo_trace[-1] - elbo_trace[-2]
                if abs(delta) < tol:
                    converged = True
                    break
        elif criterion == "pip":
            pip = 1.0 - np.prod(1.0 - np.clip(state.alpha, 0.0, 1.0), axis=0)
            if prev_pip is not None and np.max(np.abs(pip - prev_pip)) < tol:
                converged = True
                break
            prev_pip = pip
        else:
            raise ValueError("convergence_criterion must be one of {'elbo', 'pip'}")

    return state, elbo_trace, converged


def ibss_loop(data: _FitData, state: _ModelState, params: dict[str, Any]) -> tuple[_ModelState, list[float], bool]:
    """Run the IBSS coordinate-ascent loop."""
    L, p = state.alpha.shape
    prior_weights = np.asarray(params["prior_weights"], dtype=float)
    if prior_weights.shape != (p,):
        raise ValueError("prior_weights must have shape (p,)")

    max_iter = int(params.get("max_iter", 100))
    tol = float(params.get("tol", 1e-3))
    criterion = params.get("convergence_criterion", "elbo")
    verbose = bool(params.get("verbose", False))
    estimate_prior_variance = bool(params.get("estimate_prior_variance", True))
    xtx_update_mode = str(params.get("xtx_update_mode", "sequential")).lower()
    if xtx_update_mode not in {"sequential", "batched"}:
        raise ValueError("xtx_update_mode must be one of {'sequential', 'batched'}")

    _initialize_caches(data, state)

    if data.X is None and xtx_update_mode == "batched":
        return _ibss_loop_ss_batched(
            data,
            state,
            prior_weights=prior_weights,
            max_iter=max_iter,
            tol=tol,
            criterion=criterion,
            verbose=verbose,
            estimate_prior_variance=estimate_prior_variance,
            params=params,
        )

    elbo_trace: list[float] = []
    converged = False
    prev_pip = None

    for it in range(max_iter):
        prior_variance_method = _select_prior_variance_method(it, max_iter, params)
        for l in range(L):
            b_old = state.alpha[l] * state.mu[l]

            if state.Xr is not None and data.y is not None:
                Xb_old = data.compute_Xb(b_old)
                state.Xr -= Xb_old
                Xty_residual = data.compute_Xty(data.y - state.Xr)
            else:
                if state.XtXb_effects is not None:
                    XtXb_old = state.XtXb_effects[l]
                else:
                    XtXb_old = data.compute_XtXb(b_old)
                state.XtXr -= XtXb_old
                Xty_residual = data.Xty - state.XtXr

            ser = fit_ser(
                Xty_residual=Xty_residual,
                d=data.d,
                sigma2=state.sigma2,
                prior_variance=state.V[l],
                prior_weights=prior_weights,
                estimate_prior_variance=estimate_prior_variance,
                prior_variance_method=prior_variance_method,
                check_null_threshold=float(params.get("check_null_threshold", 1e-9)),
            )

            state.alpha[l] = ser.alpha
            state.mu[l] = ser.mu
            state.mu2[l] = ser.mu2
            state.V[l] = ser.V
            state.KL[l] = ser.KL
            state.lbf[l] = ser.lbf
            state.lbf_variable[l] = ser.lbf_variable

            b_new = state.alpha[l] * state.mu[l]
            if state.Xr is not None and data.y is not None:
                Xb_new = data.compute_Xb(b_new)
                state.Xr += Xb_new
                state.Xb_sq_norms[l] = float(np.dot(Xb_new, Xb_new))
            else:
                XtXb_new = data.compute_XtXb(b_new)
                if state.XtXb_effects is not None:
                    state.XtXb_effects[l] = XtXb_new
                state.XtXr += XtXb_new
                state.Xb_sq_norms[l] = float(np.dot(b_new, XtXb_new))

        elbo = compute_elbo(data, state)
        elbo_trace.append(elbo)

        if params.get("estimate_residual_variance", True):
            state.sigma2 = _estimate_residual_variance(
                data,
                state,
                method=str(params.get("residual_variance_method", "mom")),
            )

        if verbose:
            print(f"IBSS iter={it + 1}, ELBO={elbo:.6f}, sigma2={state.sigma2:.6f}")

        if criterion == "elbo":
            if len(elbo_trace) >= 2:
                delta = elbo_trace[-1] - elbo_trace[-2]
                if abs(delta) < tol:
                    converged = True
                    break
        elif criterion == "pip":
            pip = 1.0 - np.prod(1.0 - np.clip(state.alpha, 0.0, 1.0), axis=0)
            if prev_pip is not None and np.max(np.abs(pip - prev_pip)) < tol:
                converged = True
                break
            prev_pip = pip
        else:
            raise ValueError("convergence_criterion must be one of {'elbo', 'pip'}")

    return state, elbo_trace, converged
