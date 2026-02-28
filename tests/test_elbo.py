import numpy as np

from pysusie._elbo import compute_elbo
from pysusie._types import _ModelState
from pysusie._preprocessing import preprocess_individual_data


def test_elbo_matches_closed_form_zero_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 5))
    y = rng.normal(size=40)

    data = preprocess_individual_data(X, y, standardize=True, intercept=True)

    L = 3
    p = data.p
    sigma2 = 1.5

    state = _ModelState(
        alpha=np.full((L, p), 1.0 / p),
        mu=np.zeros((L, p)),
        mu2=np.zeros((L, p)),
        V=np.full(L, 0.2),
        sigma2=sigma2,
        KL=np.zeros(L),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, p)),
        Xr=np.zeros(data.n),
        XtXr=None,
        Xb_sq_norms=np.zeros(L),
    )

    elbo = compute_elbo(data, state)
    expected = -0.5 * data.n * np.log(2 * np.pi * sigma2) - 0.5 * data.yty / sigma2
    assert np.isclose(elbo, expected)


def test_elbo_finite_after_simple_state_update():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    data = preprocess_individual_data(X, y, standardize=True, intercept=True)

    L = 2
    p = data.p
    alpha = np.zeros((L, p))
    alpha[:, 0] = 1.0
    mu = np.zeros((L, p))
    mu[:, 0] = 0.3
    mu2 = mu**2 + 0.01

    bbar = np.sum(alpha * mu, axis=0)
    Xr = data.compute_Xb(bbar)
    Xb_sq = np.array([np.dot(data.compute_Xb(alpha[l] * mu[l]), data.compute_Xb(alpha[l] * mu[l])) for l in range(L)])

    state = _ModelState(
        alpha=alpha,
        mu=mu,
        mu2=mu2,
        V=np.array([0.2, 0.2]),
        sigma2=1.0,
        KL=np.array([0.1, 0.1]),
        lbf=np.zeros(L),
        lbf_variable=np.zeros((L, p)),
        Xr=Xr,
        XtXr=None,
        Xb_sq_norms=Xb_sq,
    )

    elbo = compute_elbo(data, state)
    assert np.isfinite(elbo)
