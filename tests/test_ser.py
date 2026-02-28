import numpy as np

from pysusie._ser import _optimize_prior_variance, fit_ser


def test_fit_ser_shapes_and_probabilities():
    rng = np.random.default_rng(0)
    p = 8
    Xty_resid = rng.normal(size=p)
    d = np.full(p, 25.0)
    prior_weights = np.full(p, 1.0 / p)

    res = fit_ser(
        Xty_residual=Xty_resid,
        d=d,
        sigma2=1.0,
        prior_variance=0.2,
        prior_weights=prior_weights,
        estimate_prior_variance=False,
        prior_variance_method="optim",
        check_null_threshold=1e-9,
    )

    assert res.alpha.shape == (p,)
    assert res.mu.shape == (p,)
    assert res.mu2.shape == (p,)
    assert res.lbf_variable.shape == (p,)
    assert np.isclose(res.alpha.sum(), 1.0)
    assert np.all(res.alpha >= 0)
    assert np.all(res.mu2 >= 0)


def test_fit_ser_null_column_guard():
    p = 6
    d = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 0.0])
    Xty_resid = np.array([1.1, -0.4, 0.2, 0.3, -0.7, 0.0])
    prior_weights = np.full(p, 1.0 / p)

    res = fit_ser(
        Xty_residual=Xty_resid,
        d=d,
        sigma2=1.0,
        prior_variance=0.2,
        prior_weights=prior_weights,
        estimate_prior_variance=False,
        prior_variance_method="optim",
        check_null_threshold=1e-9,
    )

    assert res.lbf_variable[-1] == 0.0
    assert res.mu[-1] == 0.0
    assert res.mu2[-1] == 0.0
    assert res.alpha[-1] > 0.0


def test_optimize_prior_variance_methods():
    betahat = np.array([0.1, 0.8, -0.6, 0.2])
    shat2 = np.array([0.1, 0.1, 0.1, 0.1])
    pi = np.full(4, 0.25)

    v_simple = _optimize_prior_variance(betahat, shat2, pi, 0.2, "simple")
    v_em = _optimize_prior_variance(betahat, shat2, pi, 0.2, "em")
    v_opt = _optimize_prior_variance(betahat, shat2, pi, 0.2, "optim")

    assert v_simple >= 0
    assert v_em >= 0
    assert v_opt >= 0
