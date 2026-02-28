import numpy as np
import pytest

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


def test_fit_mrash_rejects_zero_mass_pi():
    rng = np.random.default_rng(33)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    sa2 = np.array([0.0, 0.2, 1.0])

    with pytest.raises(ValueError, match="positive mass"):
        fit_mrash(X, y, sa2, pi=np.array([0.0, 0.0, 0.0]))
