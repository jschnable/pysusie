"""Public SuSiE estimator API."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._credible_sets import extract_credible_sets
from ._ibss import ibss_loop
from ._preprocessing import (
    _extract_feature_names,
    compute_sufficient_stats,
    estimate_ld_regularization,
    preprocess_individual_data,
    preprocess_summary_stats,
    preprocess_sufficient_stats,
    univariate_regression,
)
from ._types import SuSiEResult, _FitData, _ModelState


try:  # pragma: no cover - sklearn optional
    from sklearn.exceptions import NotFittedError
except Exception:  # pragma: no cover

    class NotFittedError(RuntimeError):
        pass


def _ensure_prior_variance(prior_variance: float | np.ndarray, L: int, ref_var: float) -> np.ndarray:
    if np.isscalar(prior_variance):
        V = np.full(L, float(prior_variance), dtype=float)
    else:
        arr = np.asarray(prior_variance, dtype=float)
        if arr.ndim != 1:
            raise ValueError("prior_variance must be scalar or 1D array")
        if arr.size == 1:
            V = np.full(L, float(arr[0]), dtype=float)
        elif arr.size == L:
            V = arr.copy()
        else:
            raise ValueError("prior_variance array must have length 1 or L")
    V = np.clip(V, 0.0, None)
    return V * float(ref_var)


def _validate_count_vector(counts: np.ndarray | None, p: int, name: str) -> np.ndarray | None:
    if counts is None:
        return None
    arr = np.asarray(counts, dtype=float)
    if arr.shape != (p,):
        raise ValueError(f"{name} must have shape ({p},)")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return arr


def _counts_from_genotypes(X, *, carrier_threshold: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(X):
        X_csr = X.tocsr()
        carriers = np.asarray((X_csr > carrier_threshold).sum(axis=0)).ravel().astype(float)
        noncarriers = float(X_csr.shape[0]) - carriers
        return carriers, noncarriers

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2D")
    finite = np.isfinite(X_arr)
    carriers = np.sum((X_arr > carrier_threshold) & finite, axis=0, dtype=float)
    noncarriers = np.sum((X_arr <= carrier_threshold) & finite, axis=0, dtype=float)
    return carriers, noncarriers


class SuSiE:
    """Sum of Single Effects regression estimator."""

    _FIT_OVERRIDE_KEYS = {
        "n_effects",
        "prior_variance",
        "estimate_prior_variance",
        "prior_variance_method",
        "prior_variance_warmup_iters",
        "prior_variance_optim_period",
        "prior_variance_force_final_optim",
        "xtx_update_mode",
        "xtx_batch_size",
        "estimate_residual_variance",
        "residual_variance_method",
        "prior_weights",
        "null_weight",
        "standardize",
        "intercept",
        "max_iter",
        "tol",
        "convergence_criterion",
        "coverage",
        "min_abs_corr",
        "min_carriers",
        "min_noncarriers",
        "refine",
        "refine_max_steps",
        "verbose",
    }

    def __init__(
        self,
        *,
        n_effects: int = 10,
        prior_variance: float | np.ndarray = 0.2,
        estimate_prior_variance: bool = True,
        prior_variance_method: str = "optim",
        prior_variance_warmup_iters: int = 1,
        prior_variance_optim_period: int = 5,
        prior_variance_force_final_optim: bool = True,
        xtx_update_mode: str = "sequential",
        xtx_batch_size: int = 2,
        estimate_residual_variance: bool = True,
        residual_variance_method: str = "mom",
        prior_weights: np.ndarray | None = None,
        null_weight: float = 0.0,
        standardize: bool = True,
        intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-3,
        convergence_criterion: str = "elbo",
        coverage: float = 0.95,
        min_abs_corr: float = 0.5,
        min_carriers: int = 10,
        min_noncarriers: int = 10,
        refine: bool = False,
        refine_max_steps: int = 5,
        verbose: bool = False,
    ):
        self.n_effects = int(n_effects)
        self.prior_variance = prior_variance
        self.estimate_prior_variance = bool(estimate_prior_variance)
        self.prior_variance_method = str(prior_variance_method)
        self.prior_variance_warmup_iters = int(prior_variance_warmup_iters)
        self.prior_variance_optim_period = int(prior_variance_optim_period)
        self.prior_variance_force_final_optim = bool(prior_variance_force_final_optim)
        self.xtx_update_mode = str(xtx_update_mode)
        self.xtx_batch_size = int(xtx_batch_size)
        self.estimate_residual_variance = bool(estimate_residual_variance)
        self.residual_variance_method = str(residual_variance_method)
        self.prior_weights = prior_weights
        self.null_weight = float(null_weight)
        self.standardize = bool(standardize)
        self.intercept = bool(intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.convergence_criterion = str(convergence_criterion)
        self.coverage = float(coverage)
        self.min_abs_corr = float(min_abs_corr)
        self.min_carriers = int(min_carriers)
        self.min_noncarriers = int(min_noncarriers)
        self.refine = bool(refine)
        self.refine_max_steps = int(refine_max_steps)
        self.verbose = bool(verbose)
        self._validate_support_thresholds()

        self._result: SuSiEResult | None = None
        self._can_predict = False

        self._X_col_means: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._X_scale: np.ndarray | None = None
        self._ld_eigendecomp_cache: OrderedDict[Any, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.refine_attempts_: int = 0
        self.refine_history_: list[float] = []

    def _coerce_param_value(self, key: str, value: Any) -> Any:
        if key in {
            "n_effects",
            "max_iter",
            "refine_max_steps",
            "prior_variance_warmup_iters",
            "prior_variance_optim_period",
            "xtx_batch_size",
            "min_carriers",
            "min_noncarriers",
        }:
            return int(value)
        if key in {"null_weight", "tol", "coverage", "min_abs_corr"}:
            return float(value)
        if key in {
            "estimate_prior_variance",
            "prior_variance_force_final_optim",
            "estimate_residual_variance",
            "standardize",
            "intercept",
            "refine",
            "verbose",
        }:
            return bool(value)
        if key in {"prior_variance_method", "residual_variance_method", "convergence_criterion", "xtx_update_mode"}:
            return str(value)
        return value

    def _apply_fit_overrides(self, overrides: dict[str, Any]) -> dict[str, Any]:
        if not overrides:
            return {}

        unknown = sorted(set(overrides) - self._FIT_OVERRIDE_KEYS)
        if unknown:
            raise TypeError(f"Unsupported fit kwargs: {unknown}")

        original: dict[str, Any] = {}
        for key, value in overrides.items():
            original[key] = getattr(self, key)
            setattr(self, key, self._coerce_param_value(key, value))
        self._validate_support_thresholds()
        return original

    def _restore_fit_overrides(self, original: dict[str, Any]) -> None:
        for key, value in original.items():
            setattr(self, key, value)

    def _resolve_model_init(self, model_init: Any) -> SuSiEResult | None:
        if model_init is None:
            return None
        if isinstance(model_init, SuSiE):
            return model_init.result_
        if isinstance(model_init, SuSiEResult):
            return model_init
        raise TypeError("model_init must be a SuSiEResult or fitted SuSiE instance")

    def _prepare_prior_weights(self, p_phys: int) -> np.ndarray:
        if self.prior_weights is None:
            pi = np.full(p_phys, 1.0 / p_phys, dtype=float)
        else:
            pi = np.asarray(self.prior_weights, dtype=float)
            if pi.shape != (p_phys,):
                raise ValueError(f"prior_weights must have shape ({p_phys},)")
            if np.any(pi < 0):
                raise ValueError("prior_weights must be non-negative")
            s = float(np.sum(pi))
            if s <= 0:
                raise ValueError("prior_weights must contain positive mass")
            pi = pi / s

        if not (0.0 <= self.null_weight < 1.0):
            raise ValueError("null_weight must be in [0, 1)")

        if self.null_weight > 0:
            pi = np.concatenate([pi * (1.0 - self.null_weight), np.array([self.null_weight])])

        return pi

    def _validate_support_thresholds(self) -> None:
        if self.min_carriers < 0:
            raise ValueError("min_carriers must be >= 0")
        if self.min_noncarriers < 0:
            raise ValueError("min_noncarriers must be >= 0")

    def _compute_support_mask(
        self,
        p: int,
        *,
        n: int | None,
        maf: np.ndarray | None = None,
        carrier_counts: np.ndarray | None = None,
        noncarrier_counts: np.ndarray | None = None,
        allow_missing_support: bool = False,
    ) -> np.ndarray:
        self._validate_support_thresholds()
        keep = np.ones(p, dtype=bool)
        if self.min_carriers == 0 and self.min_noncarriers == 0 and maf is None:
            return keep

        carriers = _validate_count_vector(carrier_counts, p, "carrier_counts")
        noncarriers = _validate_count_vector(noncarrier_counts, p, "noncarrier_counts")

        maf_arr = None
        if maf is not None:
            maf_arr = np.asarray(maf, dtype=float)
            if maf_arr.shape != (p,):
                raise ValueError(f"maf must have shape ({p},)")
            if np.any(~np.isfinite(maf_arr)) or np.any((maf_arr < 0) | (maf_arr > 1.0)):
                raise ValueError("maf values must be finite and in [0, 1]")
            keep &= maf_arr > 0

        if self.min_carriers == 0 and self.min_noncarriers == 0:
            return keep

        if carriers is None and noncarriers is not None and n is not None:
            carriers = float(n) - noncarriers
        if noncarriers is None and carriers is not None and n is not None:
            noncarriers = float(n) - carriers

        if (carriers is None or noncarriers is None) and maf_arr is not None and n is not None:
            expected_carriers = float(n) * (1.0 - (1.0 - maf_arr) ** 2)
            if carriers is None:
                carriers = expected_carriers
            if noncarriers is None:
                noncarriers = float(n) - expected_carriers

        if carriers is None or noncarriers is None:
            if allow_missing_support:
                return keep
            raise ValueError(
                "min_carriers/min_noncarriers filtering requires carrier_counts/noncarrier_counts "
                "(or maf + n for sufficient-statistics approximation)"
            )

        carriers = np.maximum(carriers, 0.0)
        noncarriers = np.maximum(noncarriers, 0.0)

        if self.min_carriers > 0:
            keep &= carriers >= float(self.min_carriers)
        if self.min_noncarriers > 0:
            keep &= noncarriers >= float(self.min_noncarriers)
        return keep

    @staticmethod
    def _subset_design_columns(X: Any, keep: np.ndarray):
        if sp.issparse(X):
            return X[:, keep]
        if hasattr(X, "iloc"):
            return X.iloc[:, keep]
        X_arr = np.asarray(X)
        return X_arr[:, keep]

    @staticmethod
    def _apply_column_mask(values: Any, keep: np.ndarray) -> Any:
        if values is None:
            return None
        arr = np.asarray(values)
        if arr.shape != (keep.size,):
            raise ValueError(f"Array must have shape ({keep.size},)")
        return arr[keep]

    def _initialize_state(
        self,
        L: int,
        p: int,
        p_phys: int,
        data: _FitData,
        prior_weights: np.ndarray,
        sigma2_init: float,
        V_init: np.ndarray,
        *,
        model_init: SuSiEResult | None = None,
        init_coef: np.ndarray | None = None,
        summary_scale: bool,
    ) -> _ModelState:
        alpha = np.tile(prior_weights, (L, 1))
        mu = np.zeros((L, p), dtype=float)
        mu2 = np.zeros((L, p), dtype=float)
        V = V_init.copy()
        sigma2 = float(max(sigma2_init, 1e-12))

        if model_init is not None:
            if model_init.n_variables != p_phys:
                raise ValueError("model_init has incompatible number of variables")

            l_take = min(L, model_init.alpha.shape[0])
            alpha_init = np.asarray(model_init.alpha[:l_take], dtype=float).copy()
            mu_init = np.asarray(model_init.mu[:l_take], dtype=float).copy()
            mu2_init = np.asarray(model_init.mu2[:l_take], dtype=float).copy()

            if data.has_null_column:
                if model_init.alpha_null is not None and model_init.alpha_null.shape[0] >= l_take:
                    alpha_null = np.asarray(model_init.alpha_null[:l_take], dtype=float).copy()
                else:
                    alpha_null = np.maximum(1.0 - np.sum(alpha_init, axis=1), 0.0)
                alpha_init = np.concatenate([alpha_init, alpha_null[:, None]], axis=1)
                mu_init = np.concatenate([mu_init, np.zeros((l_take, 1), dtype=float)], axis=1)
                mu2_init = np.concatenate([mu2_init, np.zeros((l_take, 1), dtype=float)], axis=1)

            row_sums = np.sum(alpha_init, axis=1, keepdims=True)
            row_sums[row_sums <= 0] = 1.0
            alpha[:l_take] = alpha_init / row_sums
            mu[:l_take] = mu_init
            mu2[:l_take] = np.maximum(mu2_init, mu_init**2)

            v_take = min(L, model_init.prior_variance.shape[0])
            V[:v_take] = np.maximum(np.asarray(model_init.prior_variance[:v_take], dtype=float), 0.0)
            sigma2 = max(float(model_init.residual_variance), 1e-12)

        if init_coef is not None:
            coef = np.asarray(init_coef, dtype=float)
            if coef.shape != (p_phys,):
                raise ValueError(f"init_coef must have shape ({p_phys},)")

            coef_internal = coef.copy()
            if (not summary_scale) and (self._X_scale is not None):
                coef_internal = coef_internal * self._X_scale

            order = np.argsort(-np.abs(coef_internal))
            nz = order[np.abs(coef_internal[order]) > 0]
            n_assign = min(L, nz.size)

            for l, j in enumerate(nz[:n_assign]):
                alpha[l, :] = 0.0
                alpha[l, int(j)] = 1.0
                mu_val = float(coef_internal[int(j)])
                mu[l, int(j)] = mu_val
                d_j = max(float(data.d[int(j)]), 1e-12)
                mu2[l, int(j)] = mu_val * mu_val + sigma2 / d_j
                V[l] = max(V[l], mu_val * mu_val)

            mu2 = np.maximum(mu2, mu**2)

        KL = np.zeros(L, dtype=float)
        lbf = np.zeros(L, dtype=float)
        lbf_variable = np.zeros((L, p), dtype=float)

        return _ModelState(
            alpha=alpha,
            mu=mu,
            mu2=mu2,
            V=V,
            sigma2=sigma2,
            KL=KL,
            lbf=lbf,
            lbf_variable=lbf_variable,
            Xr=None,
            XtXr=None,
            Xb_sq_norms=np.zeros(L, dtype=float),
            Xb_effects=None,
            XtXb_effects=None,
            theta=None,
            tau2=None,
            ash_pi=None,
        )

    def _fit_core(
        self,
        data: _FitData,
        *,
        feature_names: list[str] | None,
        estimate_residual_variance: bool,
        summary_scale: bool,
        model_init: Any = None,
        init_coef: np.ndarray | None = None,
    ) -> "SuSiE":
        p = data.p
        p_phys = p - (1 if data.has_null_column else 0)
        if p_phys <= 0:
            raise ValueError("No variables available after preprocessing")
        L = min(self.n_effects, p_phys)

        prior_weights = self._prepare_prior_weights(p_phys)
        if data.has_null_column and prior_weights.size == p_phys:
            prior_weights = np.concatenate([prior_weights * (1.0 - self.null_weight), np.array([self.null_weight])])

        if prior_weights.shape != (p,):
            raise ValueError("Internal prior weight shape mismatch")

        ref_var = float(data.yty / max(data.n - 1, 1))
        V_init = _ensure_prior_variance(self.prior_variance, L, ref_var)
        sigma2_init = max(ref_var, 1e-12)

        init_result = self._resolve_model_init(model_init)
        state = self._initialize_state(
            L,
            p,
            p_phys,
            data,
            prior_weights,
            sigma2_init,
            V_init,
            model_init=init_result,
            init_coef=init_coef,
            summary_scale=summary_scale,
        )

        params = {
            "prior_weights": prior_weights,
            "estimate_prior_variance": self.estimate_prior_variance,
            "prior_variance_method": self.prior_variance_method,
            "prior_variance_warmup_iters": self.prior_variance_warmup_iters,
            "prior_variance_optim_period": self.prior_variance_optim_period,
            "prior_variance_force_final_optim": self.prior_variance_force_final_optim,
            "xtx_update_mode": self.xtx_update_mode,
            "xtx_batch_size": self.xtx_batch_size,
            "estimate_residual_variance": estimate_residual_variance,
            "residual_variance_method": self.residual_variance_method,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "convergence_criterion": self.convergence_criterion,
            "check_null_threshold": 1e-9,
            "verbose": self.verbose,
        }

        state, elbo_trace, converged = ibss_loop(data, state, params)

        alpha = state.alpha
        mu = state.mu
        mu2 = state.mu2
        lbf_variable = state.lbf_variable
        prior_weights_out = prior_weights
        alpha_null = None

        if data.has_null_column:
            alpha_null = alpha[:, -1].copy()
            alpha = alpha[:, :-1]
            mu = mu[:, :-1]
            mu2 = mu2[:, :-1]
            lbf_variable = lbf_variable[:, :-1]
            prior_weights_out = prior_weights[:-1]

        coef_internal = np.sum(alpha * mu, axis=0)

        if summary_scale:
            coef = coef_internal
            intercept = 0.0
        else:
            if self._X_scale is not None:
                coef = coef_internal / self._X_scale
            else:
                coef = coef_internal.copy()
            if self.intercept:
                center = self._X_col_means if self._X_col_means is not None else np.zeros_like(coef)
                intercept = float(self._y_mean - np.dot(center, coef))
            else:
                intercept = 0.0

        self._result = SuSiEResult(
            alpha=np.asarray(alpha, dtype=float),
            mu=np.asarray(mu, dtype=float),
            mu2=np.asarray(mu2, dtype=float),
            prior_variance=np.asarray(state.V, dtype=float),
            residual_variance=float(state.sigma2),
            prior_weights=np.asarray(prior_weights_out, dtype=float),
            elbo=np.asarray(elbo_trace, dtype=float),
            n_iter=len(elbo_trace),
            converged=bool(converged),
            coef=np.asarray(coef, dtype=float),
            intercept=float(intercept),
            lbf=np.asarray(state.lbf, dtype=float),
            lbf_variable=np.asarray(lbf_variable, dtype=float),
            n_samples=int(data.n),
            n_variables=int(p_phys),
            feature_names=feature_names,
            alpha_null=alpha_null,
        )

        return self

    def _maybe_refine(
        self,
        data: _FitData,
        *,
        feature_names: list[str] | None,
        estimate_residual_variance: bool,
        summary_scale: bool,
        model_init: Any = None,
        init_coef: np.ndarray | None = None,
    ) -> None:
        """Run CS-based refinement by re-fitting with modified prior weights."""
        self.refine_attempts_ = 0
        self.refine_history_ = []
        if not self.refine or self._result is None:
            return

        p_phys = self._result.n_variables
        if p_phys <= 1:
            return

        max_steps = max(int(self.refine_max_steps), 0)
        if max_steps == 0:
            return

        if self.prior_weights is None:
            base_prior = np.full(p_phys, 1.0 / p_phys, dtype=float)
        else:
            base_prior = np.asarray(self.prior_weights, dtype=float).copy()
            if base_prior.shape != (p_phys,):
                return
            mass = float(np.sum(base_prior))
            if mass <= 0:
                return
            base_prior /= mass

        best_result = self._result
        best_elbo = float(self._result.elbo[-1]) if self._result.elbo.size else -np.inf
        self.refine_history_.append(best_elbo)
        original_prior_weights = self.prior_weights

        try:
            for _ in range(max_steps):
                cs_raw = extract_credible_sets(
                    best_result.alpha,
                    best_result.prior_variance,
                    coverage=self.coverage,
                    lbf=best_result.lbf,
                )
                if not cs_raw:
                    break

                improved = False
                round_best = best_result
                round_best_elbo = best_elbo

                for vars_idx, _cs_cov, _effect_idx, _lbf in cs_raw:
                    candidate = base_prior.copy()
                    candidate[np.asarray(vars_idx, dtype=int)] = 0.0
                    mass = float(np.sum(candidate))
                    if mass <= 0:
                        continue
                    candidate /= mass

                    self.refine_attempts_ += 1

                    # Step 1: fit with CS variables downweighted in the prior.
                    self.prior_weights = candidate
                    self._fit_core(
                        data,
                        feature_names=feature_names,
                        estimate_residual_variance=estimate_residual_variance,
                        summary_scale=summary_scale,
                        model_init=None,
                        init_coef=None,
                    )
                    t_k = self._result

                    # Step 2: restart from this solution under original prior.
                    self.prior_weights = base_prior
                    self._fit_core(
                        data,
                        feature_names=feature_names,
                        estimate_residual_variance=estimate_residual_variance,
                        summary_scale=summary_scale,
                        model_init=t_k,
                        init_coef=None,
                    )
                    s_k = self._result
                    if s_k is None or s_k.elbo.size == 0:
                        continue

                    elbo_k = float(s_k.elbo[-1])
                    if elbo_k > round_best_elbo + 1e-8:
                        round_best_elbo = elbo_k
                        round_best = s_k
                        improved = True

                if not improved:
                    break
                best_result = round_best
                best_elbo = round_best_elbo
                self.refine_history_.append(best_elbo)
        finally:
            self.prior_weights = original_prior_weights

        self._result = best_result

    def fit(self, X, y, **kwargs) -> "SuSiE":
        """Fit SuSiE model with individual-level data."""
        model_init = kwargs.pop("model_init", None)
        init_coef = kwargs.pop("init_coef", None)
        originals = self._apply_fit_overrides(kwargs)

        try:
            feature_names = _extract_feature_names(X)
            keep = None
            if not hasattr(X, "shape") or len(X.shape) != 2:
                raise ValueError("X must be 2D")
            n_obs = int(X.shape[0])
            p0 = int(X.shape[1])
            if self.min_carriers > 0 or self.min_noncarriers > 0:
                carriers, noncarriers = _counts_from_genotypes(X, carrier_threshold=0.0)
                keep = self._compute_support_mask(
                    p0,
                    n=n_obs,
                    carrier_counts=carriers,
                    noncarrier_counts=noncarriers,
                )
            if keep is not None and not np.all(keep):
                if not np.any(keep):
                    raise ValueError("No variables available after min_carriers/min_noncarriers filtering")
                X = self._subset_design_columns(X, keep)
                if feature_names is not None:
                    feature_names = [name for name, k in zip(feature_names, keep) if k]

            data = preprocess_individual_data(
                X,
                y,
                standardize=self.standardize,
                intercept=self.intercept,
                null_weight=self.null_weight,
            )

            p_phys = data.p - (1 if data.has_null_column else 0)
            self._X_col_means = np.asarray(data.X_center[:p_phys], dtype=float) if data.X_center is not None else np.zeros(p_phys)
            self._X_scale = np.asarray(data.X_scale[:p_phys], dtype=float) if data.X_scale is not None else np.ones(p_phys)
            self._y_mean = float(data.y_mean)
            self._can_predict = True

            self._fit_core(
                data,
                feature_names=feature_names,
                estimate_residual_variance=self.estimate_residual_variance,
                summary_scale=False,
                model_init=model_init,
                init_coef=init_coef,
            )
            self._maybe_refine(
                data,
                feature_names=feature_names,
                estimate_residual_variance=self.estimate_residual_variance,
                summary_scale=False,
                model_init=model_init,
                init_coef=init_coef,
            )
            return self
        finally:
            self._restore_fit_overrides(originals)

    def fit_from_sufficient_stats(
        self,
        XtX,
        Xty,
        yty,
        n,
        *,
        X_col_means=None,
        y_mean=None,
        maf=None,
        carrier_counts=None,
        noncarrier_counts=None,
        var_y=None,
        model_init=None,
        init_coef=None,
    ) -> "SuSiE":
        """Fit SuSiE model from sufficient statistics."""
        if self.intercept and (X_col_means is None or y_mean is None):
            raise ValueError("X_col_means and y_mean are required when intercept=True")

        XtX_arr = np.asarray(XtX, dtype=float)
        Xty_arr = np.asarray(Xty, dtype=float)
        if XtX_arr.ndim != 2 or XtX_arr.shape[0] != XtX_arr.shape[1]:
            raise ValueError("XtX must be square")
        if Xty_arr.shape != (XtX_arr.shape[0],):
            raise ValueError("XtX and Xty dimension mismatch")

        p0 = Xty_arr.shape[0]
        keep = self._compute_support_mask(
            p0,
            n=int(n) if n is not None else None,
            maf=maf,
            carrier_counts=carrier_counts,
            noncarrier_counts=noncarrier_counts,
            allow_missing_support=True,
        )
        if not np.any(keep):
            raise ValueError("No variables available after min_carriers/min_noncarriers filtering")
        if not np.all(keep):
            XtX_arr = XtX_arr[np.ix_(keep, keep)]
            Xty_arr = Xty_arr[keep]
            X_col_means = self._apply_column_mask(X_col_means, keep)
            maf = self._apply_column_mask(maf, keep)

        data = preprocess_sufficient_stats(
            XtX_arr,
            Xty_arr,
            yty,
            n,
            standardize=self.standardize,
            maf=maf,
            null_weight=self.null_weight,
        )

        p_phys = data.p - (1 if data.has_null_column else 0)
        if X_col_means is not None:
            self._X_col_means = np.asarray(X_col_means, dtype=float)[:p_phys]
        else:
            self._X_col_means = np.zeros(p_phys)
        self._y_mean = float(y_mean) if y_mean is not None else 0.0
        if data.X_scale is not None:
            self._X_scale = np.asarray(data.X_scale, dtype=float)[:p_phys]
        else:
            self._X_scale = np.ones(p_phys)
        self._can_predict = True

        if var_y is not None:
            # override scale reference by rescaling yty to match provided var(y)
            data.yty = float(var_y) * max(int(n) - 1, 1)

        self._fit_core(
            data,
            feature_names=None,
            estimate_residual_variance=self.estimate_residual_variance,
            summary_scale=False,
            model_init=model_init,
            init_coef=init_coef,
        )
        self._maybe_refine(
            data,
            feature_names=None,
            estimate_residual_variance=self.estimate_residual_variance,
            summary_scale=False,
            model_init=model_init,
            init_coef=init_coef,
        )
        return self

    def fit_from_summary_stats(
        self,
        *,
        z=None,
        R=None,
        n=None,
        bhat=None,
        shat=None,
        carrier_counts=None,
        noncarrier_counts=None,
        var_y=1.0,
        regularize_ld=0.0,
        estimate_residual_variance=None,
        model_init=None,
        init_coef=None,
    ) -> "SuSiE":
        """Fit SuSiE model from summary statistics."""
        if estimate_residual_variance is None:
            estimate_residual_variance = False
        if estimate_residual_variance and n is None:
            raise ValueError("estimate_residual_variance=True requires n")

        R_arr = np.asarray(R, dtype=float)
        if R_arr.ndim != 2 or R_arr.shape[0] != R_arr.shape[1]:
            raise ValueError("R must be square")
        p0 = R_arr.shape[0]
        keep = None
        if self.min_carriers > 0 or self.min_noncarriers > 0:
            keep = self._compute_support_mask(
                p0,
                n=int(n) if n is not None else None,
                carrier_counts=carrier_counts,
                noncarrier_counts=noncarrier_counts,
                allow_missing_support=True,
            )
            if not np.any(keep):
                raise ValueError("No variables available after min_carriers/min_noncarriers filtering")
            if not np.all(keep):
                R_arr = R_arr[np.ix_(keep, keep)]
                if z is not None:
                    z = self._apply_column_mask(z, keep)
                if bhat is not None:
                    bhat = self._apply_column_mask(bhat, keep)
                if shat is not None:
                    shat = self._apply_column_mask(shat, keep)

        data = preprocess_summary_stats(
            z,
            R_arr,
            n,
            bhat=bhat,
            shat=shat,
            var_y=float(var_y),
            regularize_ld=regularize_ld,
            null_weight=self.null_weight,
            ld_eigendecomp_cache=self._ld_eigendecomp_cache,
        )

        self._X_col_means = None
        self._X_scale = None
        self._y_mean = 0.0
        self._can_predict = False

        self._fit_core(
            data,
            feature_names=None,
            estimate_residual_variance=bool(estimate_residual_variance),
            summary_scale=True,
            model_init=model_init,
            init_coef=init_coef,
        )
        self._maybe_refine(
            data,
            feature_names=None,
            estimate_residual_variance=bool(estimate_residual_variance),
            summary_scale=True,
            model_init=model_init,
            init_coef=init_coef,
        )
        return self

    def predict(self, X) -> np.ndarray:
        """Predict responses for new data."""
        if self._result is None:
            raise NotFittedError("SuSiE instance is not fitted")
        if not self._can_predict:
            raise ValueError(
                "predict() is unavailable after fit_from_summary_stats(); coefficients are on internal scale"
            )

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        if X_arr.shape[1] != self._result.n_variables:
            raise ValueError("X has incorrect number of columns")

        return X_arr @ self._result.coef + self._result.intercept

    @property
    def result_(self) -> SuSiEResult:
        """The fitted result object."""
        if self._result is None:
            raise NotFittedError("SuSiE instance is not fitted")
        return self._result

    @property
    def coef_(self) -> np.ndarray:
        return self.result_.coef

    @property
    def intercept_(self) -> float:
        return self.result_.intercept

    @property
    def pip_(self) -> np.ndarray:
        return self.result_.pip


def susie_auto(X, y, *, L_init: int = 1, L_max: int = 512, **kwargs) -> SuSiEResult:
    """Fit SuSiE while adaptively increasing the number of effects."""
    L = max(int(L_init), 1)
    L_max = max(int(L_max), L)

    common = dict(kwargs)
    common.pop("n_effects", None)

    while True:
        stage1 = SuSiE(
            n_effects=L,
            estimate_prior_variance=False,
            estimate_residual_variance=False,
            **common,
        )
        stage1.fit(X, y)

        stage2 = SuSiE(
            n_effects=L,
            estimate_prior_variance=False,
            estimate_residual_variance=True,
            **common,
        )
        stage2.fit(X, y)

        stage3 = SuSiE(
            n_effects=L,
            estimate_prior_variance=True,
            estimate_residual_variance=True,
            **common,
        )
        stage3.fit(X, y)

        result = stage3.result_
        active = int(np.sum(result.prior_variance > 1e-9))
        if active < L or L >= L_max:
            return result
        L = min(2 * L, L_max)


__all__ = [
    "SuSiE",
    "SuSiEResult",
    "compute_sufficient_stats",
    "univariate_regression",
    "estimate_ld_regularization",
    "susie_auto",
]
