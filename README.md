# pysusie

Pure-Python implementation of the **Sum of Single Effects (SuSiE)** model for
variable selection and genetic fine-mapping.

## Overview

SuSiE decomposes a sparse regression into a sum of "single-effect" components,
each explaining one causal signal.  It returns **posterior inclusion
probabilities (PIPs)** and **credible sets** — small groups of variables that
together capture each signal with high confidence.

Key features:

- **Three fitting modes** — individual-level data (`fit`), sufficient statistics
  (`fit_from_sufficient_stats`), and GWAS summary statistics
  (`fit_from_summary_stats`).
- **Pure Python + NumPy/SciPy** — no compiled extensions required.
- **Credible sets with purity filtering** — automatically identifies
  independent signals and removes sets driven by LD.
- **Optional Numba JIT** — install the `numba` extra for faster inner loops.

## Installation

```bash
pip install pysusie
```

For development:

```bash
git clone <repo-url>
cd pysusie
pip install -e ".[dev]"
```

## Quick Start — Individual-Level Data

```python
from pysusie import SuSiE, load_example

data = load_example("N3finemapping")  # simulated: 574 samples, 1001 variables
model = SuSiE(n_effects=10)
model.fit(data["X"], data["y"])

result = model.result_
print("PIPs (top 5):", result.pip.argsort()[-5:][::-1])

# Credible sets (purity-filtered using the design matrix)
cs_list = result.get_credible_sets(X=data["X"])
for i, cs in enumerate(cs_list):
    print(f"CS {i}: variables {cs.variables}, coverage {cs.coverage:.2f}")
```

## Quick Start — Summary Statistics

```python
import numpy as np
from pysusie import SuSiE

# z: z-scores, R: LD correlation matrix, n: sample size
model = SuSiE(n_effects=5)
model.fit_from_summary_stats(z=z, R=R, n=n, var_y=1.0, regularize_ld="auto")

result = model.result_
cs_list = result.get_credible_sets(R=R)
for cs in cs_list:
    print(f"CS: {cs.variables}, coverage={cs.coverage:.2f}, "
          f"purity={cs.purity.min_abs_corr:.2f}")
```

## API Reference

### `SuSiE` — Main Estimator

| Parameter | Default | Description |
|---|---|---|
| `n_effects` | `10` | Maximum number of causal signals (L) |
| `prior_variance` | `0.2` | Prior effect-size variance (scaled by var(y)) |
| `estimate_prior_variance` | `True` | Learn effect-size variance from data |
| `estimate_residual_variance` | `True` | Learn residual variance from data |
| `standardize` | `True` | Standardize columns of X before fitting |
| `intercept` | `True` | Include an intercept term |
| `max_iter` | `100` | Maximum IBSS iterations |
| `tol` | `1e-3` | Convergence tolerance |
| `coverage` | `0.95` | Credible set coverage target |
| `min_abs_corr` | `0.5` | Purity threshold for credible sets |
| `refine` | `False` | Run credible-set refinement procedure |
| `null_weight` | `0.0` | Prior mass on the null (no variable selected) |
| `verbose` | `False` | Print convergence diagnostics |

**Methods:**

| Method | Description |
|---|---|
| `fit(X, y)` | Fit from individual-level genotype matrix and phenotype |
| `fit_from_sufficient_stats(XtX, Xty, yty, n, *, X_col_means=, y_mean=, ...)` | Fit from precomputed sufficient statistics (`X_col_means` and `y_mean` required when `intercept=True`) |
| `fit_from_summary_stats(*, z, R, n, var_y, ...)` | Fit from GWAS z-scores and LD matrix |
| `predict(X)` | Predict phenotype for new samples (individual-level fits only) |

**Properties:** `result_`, `coef_`, `intercept_`, `pip_`

### `SuSiEResult` — Fitted Result

| Attribute | Description |
|---|---|
| `pip` | Posterior inclusion probabilities (length p) |
| `coef` | Posterior mean coefficients (length p) |
| `intercept` | Fitted intercept |
| `alpha` | Component-by-variable inclusion matrix (L × p) |
| `prior_variance` | Learned prior variances (length L) |
| `residual_variance` | Learned residual variance |
| `elbo` | ELBO trace across iterations |
| `converged` | Whether IBSS converged |
| `n_iter` | Number of iterations run |

| Method | Description |
|---|---|
| `get_credible_sets(X=, R=)` | Extract credible sets with purity filtering |
| `summary(X=, R=)` | Pandas DataFrame with PIP, coefficients, CS membership |
| `plot(y_type="pip")` | Manhattan-style PIP plot (requires matplotlib) |
| `posterior_mean()` | Posterior mean E[b] |
| `posterior_sd()` | Posterior standard deviation |
| `posterior_samples(n_samples=)` | Draw from the approximate posterior |
| `lsfr()` | Local false sign rate |

### `CredibleSet`

| Field | Description |
|---|---|
| `variables` | Array of variable indices in this set |
| `coverage` | Achieved posterior coverage |
| `effect_index` | Which SuSiE component (0-indexed) |
| `log_bayes_factor` | Log Bayes factor for this component |
| `purity` | `PurityMetrics(min_abs_corr, mean_abs_corr, median_abs_corr)` |

### Preprocessing Functions

| Function | Description |
|---|---|
| `compute_sufficient_stats(X, y)` | Compute XtX, Xty, yty from data |
| `univariate_regression(X, y)` | Per-variable effect estimates and standard errors |
| `estimate_ld_regularization(z, R, n)` | ML estimate of optimal LD shrinkage |
| `preprocess_summary_stats(z, R, n, ...)` | Convert summary stats to internal form |

### Advanced Models

| Function | Description |
|---|---|
| `susie_auto(X, y)` | Adaptive SuSiE that grows L automatically |
| `susie_inf(X, y, ...)` | SuSiE-inf (infinitesimal random effects) |
| `susie_ash(X, y, ...)` | SuSiE-ash (adaptive shrinkage prior) |
| `fit_trendfilter(y, order=)` | SuSiE trend filtering for changepoint detection |
| `fit_mrash(X, y, ...)` | Mr.ASH penalized regression |

## Fine-Mapping from GWAS Data

A common workflow is to fine-map a GWAS locus using summary statistics and a
reference-panel LD matrix:

1. Load GWAS summary statistics (effect sizes and standard errors)
2. Read genotypes from a VCF for the locus of interest
3. Compute an LD correlation matrix from the genotype dosages
4. Run `fit_from_summary_stats` with `regularize_ld="auto"`
5. Extract credible sets and posterior inclusion probabilities

See [`examples/finemapping_from_vcf.py`](examples/finemapping_from_vcf.py) for
a complete, runnable script implementing this workflow.

## Batch Fine-Mapping

When fine-mapping hundreds or thousands of loci (e.g., eQTL fine-mapping
across all genes), the key optimization is loading all genotype data into
memory once and slicing per-locus instead of re-reading from VCF each time.

See [`examples/batch_finemapping.py`](examples/batch_finemapping.py) for a
production-oriented script that demonstrates:

- Preloading genotypes into a single NumPy array for fast per-locus slicing
- Precomputing chromosome indices for O(1) window lookups
- Checkpointing for resumable long runs
- Both individual-level and summary-statistics fitting modes

## Optional Dependencies

| Extra | Package | Used for |
|---|---|---|
| `plot` | matplotlib | `result.plot()` |
| `pandas` | pandas | `result.summary()` |
| `numba` | numba | JIT-compiled inner loops |
| `sklearn` | scikit-learn | `NotFittedError`, sklearn integration |
| `r` | rpy2 | Cross-validation against R susieR |

Install extras with:

```bash
pip install "pysusie[plot,pandas]"
```

## Citations

If you use pysusie, please cite the SuSiE methodology papers:

- **Wang et al. (2020).**
  *A simple new approach to variable selection in regression, with application
  to genetic fine mapping.*
  J. R. Stat. Soc. B, 82(5), 1273–1300.
  [doi:10.1111/rssb.12388](https://doi.org/10.1111/rssb.12388)

- **Zou et al. (2022).**
  *Fine-mapping from summary data with the "Sum of Single Effects" model.*
  PLoS Genetics, 18(7), e1010299.
  [doi:10.1371/journal.pgen.1010299](https://doi.org/10.1371/journal.pgen.1010299)
