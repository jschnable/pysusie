import numpy as np

from pysusie import SuSiEInfResult, susie_ash, susie_inf


def test_susie_inf_public_api_and_precision():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(90, 14))
    y = rng.normal(size=90)

    res = susie_inf(X, y, tau2=1e-2, return_precision=True)

    assert isinstance(res, SuSiEInfResult)
    assert res.beta.shape == (14,)
    assert res.sigma2 > 0
    assert res.tau2 > 0
    assert res.precision is not None
    assert res.precision.shape == (90, 90)
    assert res.eigen_values is not None
    assert res.eigen_vectors is not None
    assert np.all(res.eigen_values >= -1e-12)

    direct = np.linalg.inv(res.sigma2 * np.eye(90) + res.tau2 * (X @ X.T))
    assert np.allclose(res.precision, direct, atol=1e-6)


def test_susie_ash_public_api_shapes():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(120, 16))
    y = rng.normal(size=120)
    sa2 = np.array([0.0, 0.05, 0.2, 1.0])

    res = susie_ash(
        X,
        y,
        sa2,
        n_effects=4,
        n_outer_iter=3,
        mrash_kwargs={"max_iter": 20, "min_iter": 2},
        susie_kwargs={"max_iter": 20},
    )

    assert res.theta.shape == (16,)
    assert res.sparse.shape == (16,)
    assert res.coef.shape == (16,)
    assert np.allclose(res.coef, res.theta + res.sparse)
    assert res.susie.n_variables == 16
