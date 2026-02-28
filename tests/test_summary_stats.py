import numpy as np
import pytest

from pysusie import SuSiE, compute_sufficient_stats, univariate_regression


def test_fit_vs_sufficient_stats_equivalence():
    rng = np.random.default_rng(9)
    n, p = 120, 12
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[1, 8]] = [0.7, -0.5]
    y = X @ beta + rng.normal(size=n)

    m1 = SuSiE(n_effects=4, max_iter=50, tol=1e-4)
    m1.fit(X, y)

    ss = compute_sufficient_stats(X, y, standardize=True)
    m2 = SuSiE(n_effects=4, max_iter=50, tol=1e-4)
    m2.fit_from_sufficient_stats(
        ss["XtX"],
        ss["Xty"],
        ss["yty"],
        ss["n"],
        X_col_means=ss["X_col_means"],
        y_mean=ss["y_mean"],
    )

    assert np.allclose(m1.pip_, m2.pip_, atol=1e-2)


def test_summary_stats_fit_and_predict_not_available():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(200, 15))
    y = rng.normal(size=200)

    stats = univariate_regression(X, y)
    R = np.corrcoef(X.T)

    m = SuSiE(n_effects=3)
    m.fit_from_summary_stats(z=stats["z_scores"], R=R, n=200)

    with pytest.raises(ValueError):
        m.predict(X[:10])


def test_summary_stats_auto_lambda_requires_n():
    rng = np.random.default_rng(11)
    z = rng.normal(size=6)
    A = rng.normal(size=(6, 6))
    R = np.corrcoef(A)

    m = SuSiE(n_effects=2)
    with pytest.raises(ValueError):
        m.fit_from_summary_stats(z=z, R=R, n=None, regularize_ld="auto")


def test_sufficient_stats_supports_external_initialization():
    rng = np.random.default_rng(28)
    n, p = 110, 10
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    base = SuSiE(n_effects=3, max_iter=20)
    base.fit(X, y)

    ss = compute_sufficient_stats(X, y, standardize=True)
    model = SuSiE(n_effects=3, max_iter=20)
    model.fit_from_sufficient_stats(
        ss["XtX"],
        ss["Xty"],
        ss["yty"],
        ss["n"],
        X_col_means=ss["X_col_means"],
        y_mean=ss["y_mean"],
        model_init=base.result_,
        init_coef=base.result_.coef,
    )
    assert model.result_.coef.shape == (p,)


def test_sufficient_stats_batched_xtx_updates_run_and_respect_bounds():
    rng = np.random.default_rng(34)
    n, p = 140, 16
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[3, 11]] = [0.9, -0.6]
    y = X @ beta + rng.normal(size=n)

    ss = compute_sufficient_stats(X, y, standardize=True)

    batched = SuSiE(n_effects=4, max_iter=60, tol=1e-4, xtx_update_mode="batched")
    batched.fit_from_sufficient_stats(
        ss["XtX"],
        ss["Xty"],
        ss["yty"],
        ss["n"],
        X_col_means=ss["X_col_means"],
        y_mean=ss["y_mean"],
    )

    assert batched.result_.n_iter >= 1
    assert np.allclose(batched.result_.alpha.sum(axis=1), 1.0, atol=1e-10)
    assert np.all((batched.pip_ >= -1e-12) & (batched.pip_ <= 1.0 + 1e-12))


def test_summary_stats_reuses_ld_eigendecomposition_across_repeated_fits(monkeypatch):
    rng = np.random.default_rng(35)
    n, p = 220, 9
    X = rng.normal(size=(n, p))
    y1 = rng.normal(size=n)
    y2 = rng.normal(size=n)
    stats1 = univariate_regression(X, y1)
    stats2 = univariate_regression(X, y2)
    R = np.corrcoef(X.T)

    calls = {"count": 0}
    orig_eigh = np.linalg.eigh

    def counting_eigh(x):
        calls["count"] += 1
        return orig_eigh(x)

    monkeypatch.setattr(np.linalg, "eigh", counting_eigh)

    model = SuSiE(n_effects=3, max_iter=10)
    model.fit_from_summary_stats(z=stats1["z_scores"], R=R, n=n, regularize_ld="auto")
    model.fit_from_summary_stats(z=stats2["z_scores"], R=R, n=n, regularize_ld="auto")

    assert calls["count"] == 1
