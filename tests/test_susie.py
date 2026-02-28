import numpy as np
import pytest

from pysusie import SuSiE, susie_auto
from pysusie.datasets import load_example


def test_susie_fit_predict_and_forwarding_properties():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(150, 25))
    beta = np.zeros(25)
    beta[[4, 10, 18]] = [0.9, -0.8, 0.6]
    y = X @ beta + rng.normal(size=150)

    model = SuSiE(n_effects=6, max_iter=50, tol=1e-4)
    out = model.fit(X, y)

    assert out is model
    assert model.result_.coef.shape == (25,)
    assert model.pip_.shape == (25,)
    yhat = model.predict(X[:5])
    assert yhat.shape == (5,)


def test_result_arrays_are_immutable():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(100, 10))
    y = rng.normal(size=100)

    model = SuSiE(n_effects=3, max_iter=10)
    model.fit(X, y)
    result = model.result_

    with pytest.raises(ValueError):
        result.alpha[0, 0] = 0.0


def test_null_weight_strips_null_column_but_tracks_alpha_null():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(120, 12))
    y = rng.normal(size=120)

    model = SuSiE(n_effects=4, null_weight=0.2, max_iter=30)
    model.fit(X, y)
    result = model.result_

    assert result.alpha.shape[1] == 12
    assert result.alpha_null is not None
    assert np.all(result.alpha_null >= 0)
    assert np.all(result.alpha_null <= 1)
    assert np.isclose(result.prior_weights.sum(), 0.8)
    cs = result.get_credible_sets()
    assert isinstance(cs, list)


def test_posterior_samples_shape():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(80, 9))
    y = rng.normal(size=80)

    model = SuSiE(n_effects=3, max_iter=20)
    model.fit(X, y)

    draws = model.result_.posterior_samples(25, rng=0)
    assert draws.shape == (25, 9)


def test_susie_auto_returns_result():
    ex = load_example("small")
    result = susie_auto(ex["X"], ex["y"], L_init=1, L_max=8, max_iter=30)
    assert result.n_variables == ex["X"].shape[1]
    assert result.n_iter >= 1


def test_fit_accepts_temporary_parameter_overrides():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(100, 14))
    y = rng.normal(size=100)

    model = SuSiE(n_effects=2, max_iter=5, refine=False)
    model.fit(X, y, n_effects=4, max_iter=25, refine=True)

    assert model.result_.alpha.shape[0] == 4
    assert model.n_effects == 2
    assert model.max_iter == 5
    assert model.refine is False
    assert model.refine_attempts_ > 0
    assert len(model.refine_history_) >= 1


def test_fit_unknown_override_raises():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(60, 8))
    y = rng.normal(size=60)
    model = SuSiE(n_effects=2)

    with pytest.raises(TypeError):
        model.fit(X, y, does_not_exist=True)


def test_result_plot_supports_z_type():
    pytest.importorskip("matplotlib")

    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 12))
    y = rng.normal(size=100)
    model = SuSiE(n_effects=3, max_iter=20)
    model.fit(X, y)

    ax = model.result_.plot(y_type="z")
    assert ax.get_ylabel() == "z-score"


def test_result_summary_includes_credible_set_metadata():
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(23)
    X = rng.normal(size=(150, 20))
    beta = np.zeros(20)
    beta[[3, 11]] = [1.0, -0.9]
    y = X @ beta + rng.normal(size=150)

    model = SuSiE(n_effects=4, max_iter=40)
    model.fit(X, y)
    df = model.result_.summary(X=X)

    expected_cols = {
        "variable",
        "pip",
        "coef",
        "posterior_sd",
        "lfsr",
        "credible_set",
        "cs_effect_index",
        "cs_coverage",
        "cs_log_bayes_factor",
        "cs_min_abs_corr",
        "cs_mean_abs_corr",
        "cs_median_abs_corr",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == 20
    assert df["pip"].is_monotonic_decreasing


def test_fit_accepts_external_initialization():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(140, 18))
    beta = np.zeros(18)
    beta[[2, 12]] = [0.9, -0.7]
    y = X @ beta + rng.normal(size=140)

    base = SuSiE(n_effects=4, max_iter=40, tol=1e-4)
    base.fit(X, y)

    # Initialize with previous fitted object.
    init_model = SuSiE(n_effects=4, max_iter=10, tol=1e-4)
    init_model.fit(X, y, model_init=base)
    assert init_model.result_.n_variables == 18

    # Initialize with external coefficients.
    coef_model = SuSiE(n_effects=4, max_iter=10, tol=1e-4)
    coef_model.fit(X, y, init_coef=base.result_.coef)
    assert coef_model.result_.coef.shape == (18,)
    assert np.all(np.isfinite(coef_model.result_.coef))


def test_refine_history_is_non_decreasing():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(160, 24))
    beta = np.zeros(24)
    beta[[5, 14]] = [1.0, -0.9]
    y = X @ beta + rng.normal(size=160)

    model = SuSiE(n_effects=6, refine=True, refine_max_steps=3, max_iter=40, tol=1e-4)
    model.fit(X, y)

    history = np.asarray(model.refine_history_, dtype=float)
    assert history.size >= 1
    assert np.all(np.diff(history) >= -1e-10)


def test_fit_from_sufficient_stats_raises_if_all_variants_filtered():
    XtX = np.eye(2)
    Xty = np.array([1.0, -0.5])
    yty = 3.2
    maf = np.array([0.0, 0.0])
    model = SuSiE(intercept=False)

    with pytest.raises(ValueError, match="No variables available"):
        model.fit_from_sufficient_stats(XtX, Xty, yty, n=50, maf=maf)
