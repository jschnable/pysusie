import numpy as np
import pytest

from pysusie._preprocessing import (
    preprocess_individual_data,
    preprocess_summary_stats,
    preprocess_sufficient_stats,
)


def test_preprocess_individual_dense_standardized_diagonal():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 7))
    y = rng.normal(size=60)

    data = preprocess_individual_data(X, y, standardize=True, intercept=True)

    assert data.X.shape == X.shape
    assert data.y.shape == y.shape
    assert np.allclose(data.d, 59.0, atol=1e-8)
    assert data.has_null_column is False


def test_preprocess_individual_adds_virtual_null_column():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50)

    data = preprocess_individual_data(X, y, standardize=True, intercept=True, null_weight=0.1)
    assert data.p == 5
    assert data.has_null_column is True
    assert data.d[-1] == 0.0
    assert data.Xty[-1] == 0.0


def test_preprocess_sufficient_stats_standardize():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 6))
    y = rng.normal(size=80)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    yty = float(yc @ yc)

    data = preprocess_sufficient_stats(XtX, Xty, yty, n=80, standardize=True)
    assert data.XtX.shape == (6, 6)
    assert np.allclose(data.d, 79.0, atol=1e-8)


def test_preprocess_summary_stats_shapes_and_auto_regularization():
    rng = np.random.default_rng(8)
    p = 5
    A = rng.normal(size=(p, p))
    R = np.corrcoef(A)
    z = rng.normal(size=p)

    data = preprocess_summary_stats(z=z, R=R, n=200, regularize_ld="auto")

    assert data.XtX.shape == (p, p)
    assert data.Xty.shape == (p,)
    assert data.eigen_values is not None
    assert 0.0 <= data.regularization <= 1.0


def test_preprocess_summary_stats_reuses_ld_eigendecomposition_cache(monkeypatch):
    rng = np.random.default_rng(31)
    p = 7
    A = rng.normal(size=(p, p))
    R = np.corrcoef(A)
    z1 = rng.normal(size=p)
    z2 = rng.normal(size=p)

    calls = {"count": 0}
    orig_eigh = np.linalg.eigh

    def counting_eigh(x):
        calls["count"] += 1
        return orig_eigh(x)

    monkeypatch.setattr(np.linalg, "eigh", counting_eigh)

    cache = {}
    preprocess_summary_stats(z=z1, R=R, n=180, regularize_ld="auto", ld_eigendecomp_cache=cache)
    preprocess_summary_stats(z=z2, R=R, n=180, regularize_ld="auto", ld_eigendecomp_cache=cache)

    assert calls["count"] == 1


def test_preprocess_summary_stats_numeric_lambda_reuses_r_eigendecomposition(monkeypatch):
    rng = np.random.default_rng(32)
    p = 8
    A = rng.normal(size=(p, p))
    R = np.corrcoef(A)
    z = rng.normal(size=p)

    calls = {"count": 0}
    orig_eigh = np.linalg.eigh

    def counting_eigh(x):
        calls["count"] += 1
        return orig_eigh(x)

    monkeypatch.setattr(np.linalg, "eigh", counting_eigh)

    cache = {}
    data1 = preprocess_summary_stats(z=z, R=R, n=220, regularize_ld=0.2, ld_eigendecomp_cache=cache)
    data2 = preprocess_summary_stats(z=z, R=R, n=220, regularize_ld=0.6, ld_eigendecomp_cache=cache)

    assert calls["count"] == 1
    assert data1.eigen_values is not None
    assert data2.eigen_values is not None


def test_preprocess_individual_data_rejects_single_sample():
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([0.5])

    with pytest.raises(ValueError, match="at least 2 rows"):
        preprocess_individual_data(X, y, standardize=True, intercept=True)


def test_preprocess_summary_stats_rejects_non_positive_var_y():
    z = np.array([1.0, -0.5])
    R = np.eye(2)

    with pytest.raises(ValueError, match="var_y must be positive"):
        preprocess_summary_stats(z=z, R=R, n=100, var_y=0.0)

    with pytest.raises(ValueError, match="var_y must be positive"):
        preprocess_summary_stats(z=z, R=R, n=100, var_y=-1.0)
