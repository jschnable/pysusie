import numpy as np

from pysusie._mrash import fit_mrash


def test_fit_mrash_shapes_and_constraints():
    rng = np.random.default_rng(12)
    n, p = 80, 10
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    sa2 = np.array([0.0, 0.1, 1.0])
    res = fit_mrash(X, y, sa2, max_iter=30, min_iter=2, tol=1e-4)

    assert res.beta.shape == (p,)
    assert res.pi.shape == (sa2.size,)
    assert np.isclose(res.pi.sum(), 1.0)
    assert res.sigma2 > 0
    assert res.n_iter >= 1
    assert res.elbo.shape[0] == res.n_iter
