"""Core types and dataclasses used by pysusie."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

from ._utils import freeze_array, log1mexp_from_probs, rng_from_seed


@dataclass(frozen=True)
class PurityMetrics:
    min_abs_corr: float
    mean_abs_corr: float
    median_abs_corr: float


@dataclass(frozen=True)
class CredibleSet:
    variables: np.ndarray
    coverage: float
    effect_index: int
    log_bayes_factor: float
    purity: PurityMetrics | None = None

    def __post_init__(self) -> None:
        freeze_array(self.variables)


@dataclass
class SERResult:
    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    lbf: float
    lbf_variable: np.ndarray
    V: float
    KL: float


@dataclass
class _FitData:
    """Internal data representation for the IBSS algorithm."""

    XtX: np.ndarray | None
    Xty: np.ndarray
    yty: float
    d: np.ndarray
    n: int
    p: int

    X: np.ndarray | sp.spmatrix | None
    y: np.ndarray | None

    X_center: np.ndarray | None
    X_scale: np.ndarray | None
    y_mean: float
    is_sparse: bool

    has_null_column: bool

    eigen_values: np.ndarray | None = None
    eigen_vectors: np.ndarray | None = None
    regularization: float = 0.0

    def compute_Xb(self, b: np.ndarray) -> np.ndarray:
        """Compute transformed X @ b, skipping the virtual null if present."""
        if self.X is None:
            raise ValueError("compute_Xb requires individual-level X")

        b = np.asarray(b, dtype=float)
        b_phys = b[:-1] if self.has_null_column else b

        if not self.is_sparse:
            return np.asarray(self.X @ b_phys, dtype=float)

        if self.X_scale is None:
            raise ValueError("Sparse path requires X_scale")

        center = 0.0
        c = b_phys / self.X_scale
        if self.X_center is not None:
            center = float(np.dot(self.X_center, c))
        return np.asarray(self.X @ c, dtype=float) - center

    def compute_Xty(self, r: np.ndarray) -> np.ndarray:
        """Compute transformed X' @ r, appending 0 for the virtual null."""
        if self.X is None:
            raise ValueError("compute_Xty requires individual-level X")

        r = np.asarray(r, dtype=float)

        if not self.is_sparse:
            out = np.asarray(self.X.T @ r, dtype=float)
        else:
            if self.X_scale is None:
                raise ValueError("Sparse path requires X_scale")
            out = np.asarray((self.X.T @ r) / self.X_scale, dtype=float)
            if self.X_center is not None:
                out -= (self.X_center / self.X_scale) * float(np.sum(r))

        if self.has_null_column:
            return np.concatenate([out, np.array([0.0])])
        return out

    def compute_XtXb(self, b: np.ndarray) -> np.ndarray:
        """Compute X'X @ b using XtX directly when available."""
        b = np.asarray(b, dtype=float)
        if self.XtX is not None:
            return np.asarray(self.XtX @ b, dtype=float)
        xb = self.compute_Xb(b)
        return self.compute_Xty(xb)

    def compute_XtXB(self, B: np.ndarray) -> np.ndarray:
        """Compute X'X @ B for a stack of coefficient vectors."""
        B_arr = np.asarray(B, dtype=float)
        if B_arr.ndim != 2:
            raise ValueError("B must be a 2D array with shape (k, p)")
        if B_arr.shape[1] != self.p:
            raise ValueError(f"B must have shape (k, {self.p})")

        if self.XtX is not None:
            return np.asarray((self.XtX @ B_arr.T).T, dtype=float)

        out = np.empty_like(B_arr)
        for k in range(B_arr.shape[0]):
            out[k] = self.compute_XtXb(B_arr[k])
        return out


@dataclass
class _ModelState:
    """Mutable state tracked during IBSS iterations."""

    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    V: np.ndarray
    sigma2: float
    KL: np.ndarray
    lbf: np.ndarray
    lbf_variable: np.ndarray

    Xr: np.ndarray | None
    XtXr: np.ndarray | None
    Xb_sq_norms: np.ndarray
    Xb_effects: np.ndarray | None = None
    XtXb_effects: np.ndarray | None = None

    theta: np.ndarray | None = None
    tau2: float | None = None
    ash_pi: np.ndarray | None = None


@dataclass
class MrASHResult:
    beta: np.ndarray
    pi: np.ndarray
    sigma2: float
    converged: bool
    n_iter: int
    elbo: np.ndarray


@dataclass(frozen=True)
class SuSiEResult:
    """Immutable SuSiE fitting result."""

    alpha: np.ndarray
    mu: np.ndarray
    mu2: np.ndarray
    prior_variance: np.ndarray
    residual_variance: float
    prior_weights: np.ndarray

    elbo: np.ndarray
    n_iter: int
    converged: bool

    coef: np.ndarray
    intercept: float

    lbf: np.ndarray
    lbf_variable: np.ndarray

    n_samples: int
    n_variables: int
    feature_names: list[str] | None

    alpha_null: np.ndarray | None = None

    def __post_init__(self) -> None:
        freeze_array(self.alpha)
        freeze_array(self.mu)
        freeze_array(self.mu2)
        freeze_array(self.prior_variance)
        freeze_array(self.prior_weights)
        freeze_array(self.elbo)
        freeze_array(self.coef)
        freeze_array(self.lbf)
        freeze_array(self.lbf_variable)
        freeze_array(self.alpha_null)

    @cached_property
    def pip(self) -> np.ndarray:
        """Posterior inclusion probability for each variable."""
        log1m = np.sum(log1mexp_from_probs(self.alpha), axis=0)
        pip = 1.0 - np.exp(log1m)
        return np.clip(pip, 0.0, 1.0)

    def get_credible_sets(
        self,
        X: Any = None,
        R: Any = None,
        *,
        coverage: float = 0.95,
        min_abs_corr: float = 0.5,
        n_purity: int = 100,
    ) -> list[CredibleSet]:
        """Extract credible sets with optional purity filtering."""
        from ._credible_sets import compute_purity, extract_credible_sets

        raw_sets = extract_credible_sets(
            self.alpha,
            self.prior_variance,
            lbf=self.lbf,
            coverage=coverage,
        )

        out: list[CredibleSet] = []
        for variables, cs_coverage, effect_idx, lbf in raw_sets:
            purity = None
            if X is not None or R is not None:
                purity = compute_purity(
                    variables,
                    X=np.asarray(X) if X is not None else None,
                    R=np.asarray(R) if R is not None else None,
                    n_purity=n_purity,
                )
                if purity.min_abs_corr < min_abs_corr:
                    continue

            out.append(
                CredibleSet(
                    variables=variables,
                    coverage=float(cs_coverage),
                    effect_index=int(effect_idx),
                    log_bayes_factor=float(lbf),
                    purity=purity,
                )
            )
        return out

    def posterior_mean(self, prior_tol: float = 1e-9) -> np.ndarray:
        """Posterior mean E[b]."""
        active = self.prior_variance > prior_tol
        if not np.any(active):
            return np.zeros(self.n_variables)
        return np.sum(self.alpha[active] * self.mu[active], axis=0)

    def posterior_sd(self, prior_tol: float = 1e-9) -> np.ndarray:
        """Posterior standard deviation for each coefficient."""
        active = self.prior_variance > prior_tol
        if not np.any(active):
            return np.zeros(self.n_variables)

        a = self.alpha[active]
        m = self.mu[active]
        m2 = self.mu2[active]

        eb_l = a * m
        var_l = a * m2 - eb_l**2
        var = np.sum(np.maximum(var_l, 0.0), axis=0)
        return np.sqrt(np.maximum(var, 0.0))

    def posterior_samples(
        self,
        n_samples: int = 100,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw samples from the mean-field SuSiE posterior approximation."""
        generator = rng_from_seed(rng)
        p = self.n_variables
        L = self.alpha.shape[0]

        draws = np.zeros((n_samples, p))
        for i in range(n_samples):
            sample = np.zeros(p)
            for l in range(L):
                probs = self.alpha[l]
                has_null = self.alpha_null is not None
                if has_null:
                    probs = np.concatenate([probs, np.array([self.alpha_null[l]])])
                idx = generator.choice(probs.size, p=probs)
                if has_null and idx == p:
                    continue
                cond_var = max(self.mu2[l, idx] - self.mu[l, idx] ** 2, 0.0)
                sample[idx] += generator.normal(self.mu[l, idx], np.sqrt(cond_var))
            draws[i] = sample
        return draws

    def lsfr(self) -> np.ndarray:
        """Approximate local false sign rate via normal approximation."""
        mean = self.posterior_mean()
        sd = self.posterior_sd()

        out = np.ones_like(mean)
        nz = sd > 0
        if np.any(nz):
            p_neg = norm.cdf(0.0, loc=mean[nz], scale=sd[nz])
            p_pos = 1.0 - p_neg
            out[nz] = np.minimum(p_neg, p_pos)

        z = ~nz
        out[z & (mean != 0.0)] = 0.0
        return np.clip(out, 0.0, 1.0)

    def summary(
        self,
        X: Any = None,
        R: Any = None,
        *,
        coverage: float = 0.95,
        min_abs_corr: float = 0.5,
        n_purity: int = 100,
        sort_by: str = "pip",
        ascending: bool = False,
    ):
        """Return a summary dataframe with PIP/effect-size/credible-set metadata."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("summary() requires pandas") from exc

        cs_list = self.get_credible_sets(
            X=X,
            R=R,
            coverage=coverage,
            min_abs_corr=min_abs_corr,
            n_purity=n_purity,
        )

        cs_membership = np.full(self.n_variables, fill_value=-1, dtype=int)
        cs_effect_index = np.full(self.n_variables, fill_value=-1, dtype=int)
        cs_coverage = np.full(self.n_variables, fill_value=np.nan, dtype=float)
        cs_log_bf = np.full(self.n_variables, fill_value=np.nan, dtype=float)
        cs_purity_min = np.full(self.n_variables, fill_value=np.nan, dtype=float)
        cs_purity_mean = np.full(self.n_variables, fill_value=np.nan, dtype=float)
        cs_purity_median = np.full(self.n_variables, fill_value=np.nan, dtype=float)

        for idx, cs in enumerate(cs_list):
            for v in cs.variables:
                v = int(v)
                current = cs_log_bf[v]
                if np.isnan(current) or cs.log_bayes_factor > current:
                    cs_membership[v] = idx
                    cs_effect_index[v] = int(cs.effect_index)
                    cs_coverage[v] = float(cs.coverage)
                    cs_log_bf[v] = float(cs.log_bayes_factor)
                    if cs.purity is not None:
                        cs_purity_min[v] = float(cs.purity.min_abs_corr)
                        cs_purity_mean[v] = float(cs.purity.mean_abs_corr)
                        cs_purity_median[v] = float(cs.purity.median_abs_corr)

        names = self.feature_names if self.feature_names is not None else list(range(self.n_variables))
        df = pd.DataFrame(
            {
                "variable": names,
                "pip": self.pip,
                "coef": self.coef,
                "posterior_sd": self.posterior_sd(),
                "lfsr": self.lsfr(),
                "credible_set": cs_membership,
                "cs_effect_index": cs_effect_index,
                "cs_coverage": cs_coverage,
                "cs_log_bayes_factor": cs_log_bf,
                "cs_min_abs_corr": cs_purity_min,
                "cs_mean_abs_corr": cs_purity_mean,
                "cs_median_abs_corr": cs_purity_median,
            }
        )
        if sort_by not in df.columns:
            raise ValueError(f"sort_by must be one of {list(df.columns)}")
        return df.sort_values(sort_by, ascending=ascending, kind="stable").reset_index(drop=True)

    def plot(self, *, y_type: str = "pip", ax=None, **kwargs):
        """Plot high-level summaries of the fitted result."""
        from ._plotting import plot_diagnostic, plot_pip, plot_z

        if y_type == "pip":
            return plot_pip(self, ax=ax, **kwargs)
        if y_type in {"z", "zscore", "z-score"}:
            return plot_z(self, ax=ax, **kwargs)
        return plot_diagnostic(self, what=y_type, ax=ax, **kwargs)


def make_result_arrays_writeable(result: SuSiEResult, writeable: bool) -> None:
    """Utility for tests only."""
    for field_name in (
        "alpha",
        "mu",
        "mu2",
        "prior_variance",
        "prior_weights",
        "elbo",
        "coef",
        "lbf",
        "lbf_variable",
    ):
        arr = getattr(result, field_name)
        if isinstance(arr, np.ndarray):
            arr.flags.writeable = writeable
