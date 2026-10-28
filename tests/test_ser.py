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


def test_fit_ser_drops_prior_variance_not_beating_null(monkeypatch):
    # susieR keeps an estimated V only if the SER marginal likelihood at V exceeds that at V = 0.
    # Force the optimiser to return a small positive V with no supporting signal: it must be reset to 0.
    import pysusie._ser as ser

    monkeypatch.setattr(ser, "_optimize_prior_variance", lambda **kw: 1e-4)
    p = 6
    res = fit_ser(
        Xty_residual=np.zeros(p),
        d=np.full(p, 700.0),
        sigma2=1.0,
        prior_variance=0.2,
        prior_weights=np.full(p, 1.0 / p),
        estimate_prior_variance=True,
        prior_variance_method="optim",
    )
    assert res.V == 0.0
    assert np.allclose(res.mu, 0.0)


def test_optimize_prior_variance_keeps_current_v_when_search_is_worse():
    # One variable with z ~ 4.2 among 1000: the SER marginal likelihood is ~0 on a plateau as V -> 0, dips
    # slightly, and peaks (+0.31) near log V = -3.95. The bounded search settles on the plateau; as in susieR,
    # a current V that scores better must be kept rather than dropping the effect.
    from scipy.special import logsumexp
    from pysusie._ser import _compute_lbf

    p = 1000
    d = np.full(p, 700.0)
    shat2 = 1.0 / d
    betahat = np.zeros(p)
    betahat[0] = 0.16
    pw = np.full(p, 1.0 / p)
    V_cur = float(np.exp(-3.95))
    V = _optimize_prior_variance(betahat=betahat, shat2=shat2, prior_weights=pw, V_init=V_cur, method="optim")

    def lm(v):
        return logsumexp(np.log(pw) + _compute_lbf(betahat, shat2, v))

    assert V > 1e-6
    assert lm(V) >= lm(V_cur) - 1e-12


def test_ser_log_marginal_kernel_matches_reference():
    # The fused kernel (numba when available) and its numpy fallback must equal logsumexp(log_pi + lbf).
    from scipy.special import logsumexp
    from pysusie._numba_kernels import _ser_log_marginal, _ser_log_marginal_numpy
    from pysusie._ser import _compute_lbf

    rng = np.random.default_rng(1)
    p = 500
    betahat = rng.normal(0.0, 0.1, p)
    shat2 = rng.uniform(0.5, 2.0, p) / 700.0
    shat2[3] = np.inf
    shat2[7] = 0.0
    log_pi = np.log(rng.dirichlet(np.ones(p)))
    for V in [1e-12, 1e-4, 0.01, 1.0, 1e5]:
        ref = logsumexp(log_pi + _compute_lbf(betahat, shat2, V))
        assert np.isclose(_ser_log_marginal(betahat, shat2, log_pi, V), ref, rtol=1e-12, atol=1e-12)
        assert np.isclose(_ser_log_marginal_numpy(betahat, shat2, log_pi, V), ref, rtol=1e-12, atol=1e-12)
