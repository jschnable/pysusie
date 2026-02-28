import numpy as np

from pysusie import fit_trendfilter, trend_filter_design


def test_trend_filter_design_shape_and_structure():
    X0 = trend_filter_design(8, order=0)
    X1 = trend_filter_design(8, order=1)

    assert X0.shape == (8, 7)
    assert X1.shape == (8, 7)
    assert np.all(X0[0] == 0)
    assert np.all(np.diff(X0[:, 0]) >= 0)


def test_fit_trendfilter_detects_changepoints():
    rng = np.random.default_rng(29)
    n = 120
    y = np.zeros(n)
    y[40:] += 1.5
    y[85:] -= 2.0
    y += rng.normal(scale=0.2, size=n)

    fit = fit_trendfilter(
        y,
        n_effects=8,
        order=0,
        coef_threshold=5e-2,
        susie_kwargs={"max_iter": 60, "tol": 1e-4},
    )

    cps = fit.changepoints
    assert fit.signal.shape == (n,)
    assert cps.ndim == 1
    assert np.any(np.abs(cps - 40) <= 3)
    assert np.any(np.abs(cps - 85) <= 3)
