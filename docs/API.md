# pysusie API Notes

## Core Estimator

- `SuSiE.fit(X, y, **kwargs)` supports temporary per-fit overrides and initialization:
  - `model_init`: a previous `SuSiEResult` or fitted `SuSiE` instance.
  - `init_coef`: initial coefficient vector.
- `refine=True` enables credible-set based re-fitting; diagnostics are exposed via:
  - `model.refine_attempts_`
  - `model.refine_history_`

## Summary Helpers

- `SuSiEResult.summary(...)` returns variable-level summaries with CS metadata columns:
  - `cs_effect_index`, `cs_coverage`, `cs_log_bayes_factor`
  - `cs_min_abs_corr`, `cs_mean_abs_corr`, `cs_median_abs_corr`

## Advanced Models

- `susie_inf(X, y, ...)` returns `SuSiEInfResult` with eigendecomposition-backed precision support.
- `susie_ash(X, y, sa2, ...)` returns `SuSiEAshResult` for sparse + background decomposition.

## Trend Filtering

- `trend_filter_design(n, order=0)` creates a 1D trend-filter basis.
- `fit_trendfilter(y, ...)` fits SuSiE on the trend-filter basis and returns detected changepoints.

## Benchmarking

- `benchmarks/benchmark_vs_r.py` compares runtime with `susieR` when `rpy2` + R package are available.
- Use `--python-only` to run a local benchmark without R.
