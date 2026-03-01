import numpy as np

from pysusie._ibss import ibss_loop
from pysusie._preprocessing import preprocess_individual_data, preprocess_sufficient_stats
from pysusie._types import _ModelState


def test_ibss_elbo_monotone_without_sigma2_updates():
    rng = np.random.default_rng(3)
    n, p = 120, 20
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[2, 9]] = [0.8, -0.7]
    y = X @ beta + rng.normal(scale=1.0, size=n)

    data = preprocess_individual_data(X, y, standardize=True, intercept=True)

    L = 4
    alpha = np.full((L, data.p), 1.0 / data.p)
    state = _ModelState(
        alpha=alpha,
        mu=np.zeros((L, data.p)),
        mu2=np.zeros((L, data.p)),
        V=np.full(L, 0.2 * np.var(y, ddof=1)),
        sigma2=np.var(y, ddof=1),
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, data.p)),
        Xr=None,
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    params = {
        "prior_weights": np.full(data.p, 1.0 / data.p),
        "estimate_prior_variance": True,
        "prior_variance_method": "optim",
        "estimate_residual_variance": False,
        "residual_variance_method": "mom",
        "max_iter": 20,
        "tol": 1e-4,
        "convergence_criterion": "elbo",
        "check_null_threshold": 1e-9,
        "verbose": False,
    }

    state_out, elbo, converged = ibss_loop(data, state, params)

    assert len(elbo) > 1
    assert np.all(np.diff(elbo) >= -1e-6)
    assert np.allclose(state_out.alpha.sum(axis=1), 1.0, atol=1e-10)
    assert isinstance(converged, bool)


def test_ibss_probability_constraints_with_null_weight():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 10))
    y = rng.normal(size=80)
    data = preprocess_individual_data(X, y, standardize=True, intercept=True, null_weight=0.2)

    L = 3
    alpha0 = np.full((L, data.p), 1.0 / data.p)
    state = _ModelState(
        alpha=alpha0,
        mu=np.zeros((L, data.p)),
        mu2=np.zeros((L, data.p)),
        V=np.full(L, 0.2),
        sigma2=1.0,
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, data.p)),
        Xr=None,
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    params = {
        "prior_weights": np.concatenate([np.full(data.p - 1, 0.8 / (data.p - 1)), np.array([0.2])]),
        "estimate_prior_variance": False,
        "prior_variance_method": "optim",
        "estimate_residual_variance": False,
        "residual_variance_method": "mom",
        "max_iter": 3,
        "tol": 1e-6,
        "convergence_criterion": "elbo",
        "check_null_threshold": 1e-9,
        "verbose": False,
    }

    state_out, _, _ = ibss_loop(data, state, params)
    row_sums = state_out.alpha.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)
    assert np.all((state_out.alpha >= 0) & (state_out.alpha <= 1))


def test_ibss_ss_path_populates_xtxb_effect_cache():
    rng = np.random.default_rng(30)
    n, p = 90, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    yty = float(yc @ yc)
    data = preprocess_sufficient_stats(XtX, Xty, yty, n=n, standardize=True)

    L = 4
    alpha0 = np.full((L, data.p), 1.0 / data.p)
    state = _ModelState(
        alpha=alpha0,
        mu=np.zeros((L, data.p)),
        mu2=np.zeros((L, data.p)),
        V=np.full(L, 0.2),
        sigma2=1.0,
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, data.p)),
        Xr=None,
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    params = {
        "prior_weights": np.full(data.p, 1.0 / data.p),
        "estimate_prior_variance": True,
        "prior_variance_method": "optim",
        "prior_variance_warmup_iters": 1,
        "prior_variance_optim_period": 4,
        "prior_variance_force_final_optim": True,
        "estimate_residual_variance": False,
        "residual_variance_method": "mom",
        "max_iter": 5,
        "tol": 0.0,
        "convergence_criterion": "elbo",
        "check_null_threshold": 1e-9,
        "verbose": False,
    }

    state_out, _, _ = ibss_loop(data, state, params)
    assert state_out.XtXb_effects is not None
    assert state_out.XtXb_effects.shape == (L, data.p)

    # Cached XtXb per effect should match direct multiplication at the solution.
    for l in range(L):
        bl = state_out.alpha[l] * state_out.mu[l]
        direct = data.compute_XtXb(bl)
        assert np.allclose(state_out.XtXb_effects[l], direct, atol=1e-10)


def test_ibss_individual_path_populates_xb_effect_cache():
    rng = np.random.default_rng(35)
    n, p = 120, 16
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    data = preprocess_individual_data(X, y, standardize=True, intercept=True)

    L = 4
    alpha0 = np.full((L, data.p), 1.0 / data.p)
    state = _ModelState(
        alpha=alpha0,
        mu=np.zeros((L, data.p)),
        mu2=np.zeros((L, data.p)),
        V=np.full(L, 0.2),
        sigma2=1.0,
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, data.p)),
        Xr=None,
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    params = {
        "prior_weights": np.full(data.p, 1.0 / data.p),
        "estimate_prior_variance": True,
        "prior_variance_method": "optim",
        "prior_variance_warmup_iters": 1,
        "prior_variance_optim_period": 4,
        "prior_variance_force_final_optim": True,
        "estimate_residual_variance": False,
        "residual_variance_method": "mom",
        "max_iter": 6,
        "tol": 0.0,
        "convergence_criterion": "elbo",
        "check_null_threshold": 1e-9,
        "verbose": False,
    }

    state_out, _, _ = ibss_loop(data, state, params)
    assert state_out.Xb_effects is not None
    assert state_out.Xb_effects.shape == (L, n)

    # Cached Xb per effect should match direct multiplication at the solution.
    for l in range(L):
        bl = state_out.alpha[l] * state_out.mu[l]
        direct = data.compute_Xb(bl)
        assert np.allclose(state_out.Xb_effects[l], direct, atol=1e-10)


def test_ibss_ss_batched_xtx_updates_match_direct_products():
    rng = np.random.default_rng(33)
    n, p = 100, 14
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    yty = float(yc @ yc)
    data = preprocess_sufficient_stats(XtX, Xty, yty, n=n, standardize=True)

    L = 5
    alpha0 = np.full((L, data.p), 1.0 / data.p)
    state = _ModelState(
        alpha=alpha0,
        mu=np.zeros((L, data.p)),
        mu2=np.zeros((L, data.p)),
        V=np.full(L, 0.2),
        sigma2=1.0,
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, data.p)),
        Xr=None,
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    params = {
        "prior_weights": np.full(data.p, 1.0 / data.p),
        "estimate_prior_variance": True,
        "prior_variance_method": "optim",
        "prior_variance_warmup_iters": 1,
        "prior_variance_optim_period": 4,
        "prior_variance_force_final_optim": True,
        "xtx_update_mode": "batched",
        "estimate_residual_variance": False,
        "residual_variance_method": "mom",
        "max_iter": 6,
        "tol": 0.0,
        "convergence_criterion": "elbo",
        "check_null_threshold": 1e-9,
        "verbose": False,
    }

    state_out, _, _ = ibss_loop(data, state, params)
    assert state_out.XtXb_effects is not None
    assert state_out.XtXb_effects.shape == (L, data.p)

    for l in range(L):
        bl = state_out.alpha[l] * state_out.mu[l]
        direct = data.compute_XtXb(bl)
        assert np.allclose(state_out.XtXb_effects[l], direct, atol=1e-10)
