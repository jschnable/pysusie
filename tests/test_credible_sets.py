import numpy as np

from pysusie._credible_sets import compute_purity, extract_credible_sets


def test_extract_credible_sets_and_deduplicate():
    alpha = np.array(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.68, 0.22, 0.1, 0.0],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    V = np.array([0.2, 0.2, 0.2])
    lbf = np.array([1.0, 2.0, 3.0])

    cs = extract_credible_sets(alpha, V, coverage=0.9, lbf=lbf)

    # First two effects produce same set; keep one with larger lbf.
    assert len(cs) == 2
    assert any(np.array_equal(c[0], np.array([0, 1])) for c in cs)


def test_compute_purity_from_R():
    R = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )
    purity = compute_purity(np.array([0, 1]), R=R)

    assert np.isclose(purity.min_abs_corr, 0.9)
    assert np.isclose(purity.mean_abs_corr, 0.9)
    assert np.isclose(purity.median_abs_corr, 0.9)


def test_extract_credible_sets_handles_row_mass_below_coverage():
    alpha = np.array([[0.08, 0.02, 0.0]])
    V = np.array([0.5])

    cs = extract_credible_sets(alpha, V, coverage=0.95)

    assert len(cs) == 1
    vars_idx, achieved, effect_idx, _ = cs[0]
    assert effect_idx == 0
    assert np.array_equal(vars_idx, np.array([0, 1, 2]))
    assert np.isclose(achieved, alpha.sum())
