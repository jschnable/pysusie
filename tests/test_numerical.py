import numpy as np

from pysusie import SuSiE


def test_probability_constraints_and_pip_bounds():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(180, 30))
    y = rng.normal(size=180)

    model = SuSiE(n_effects=5, max_iter=40, tol=1e-4)
    model.fit(X, y)
    result = model.result_

    assert np.allclose(result.alpha.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(result.pip >= -1e-10)
    assert np.all(result.pip <= 1.0 + 1e-10)


def test_numerical_stability_with_high_collinearity():
    rng = np.random.default_rng(18)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 + 1e-3 * rng.normal(size=n)
    X = np.column_stack([x1, x2, rng.normal(size=(n, 8))])
    y = 0.7 * x1 + rng.normal(size=n)

    model = SuSiE(n_effects=3, max_iter=40, tol=1e-4)
    model.fit(X, y)
    result = model.result_

    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.mu))
    assert np.all(np.isfinite(result.mu2))
    assert np.all(np.isfinite(result.elbo))


def test_no_signal_does_not_crash_and_stays_finite():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(120, 20))
    y = rng.normal(size=120)

    model = SuSiE(n_effects=5, max_iter=30)
    model.fit(X, y)

    result = model.result_
    assert np.all(result.prior_variance >= 0)
    assert np.isfinite(result.residual_variance)
