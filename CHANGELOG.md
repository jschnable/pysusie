# Changelog

## 1.0.0 — 2026-10-28

First stable release. Benchmarked against susieR 0.14.2 on 192 regions of real maize genotypes (711 inbred
lines, 1,000–10,000 variables per region; simulated one-, two- and three-causal and allelic-series traits plus
a real flowering-time trait). With the susieR-equivalent settings listed in the README ("Matching susieR"):
maximum |PIP difference| 0.01, ELBO within 0.0015, all fits converge, and identical credible sets in 189/192
regions. The other 3 differ only because credible sets with more than 100 variables have purity estimated on
a 100-variable subsample, which susieR draws at random. Median single-thread time per fit at 1k / 3k / 10k
variables: susieR 0.13 / 0.41 / 1.50 s, pysusie 1.0.0 0.016 / 0.040 / 0.103 s.

### Fixed
- Prior-variance estimation now applies susieR's two guards (`optimize_prior_variance`). The current V is kept
  when the bounded search returns a worse value, and V is set to 0 unless the single-effect marginal
  likelihood at V beats V = 0. Without them, the search could settle on the flat V -> 0 plateau and drop a
  supported effect. That could change credible sets, and in some regions made IBSS cycle without converging.

### Changed
- The standardized dense X used in the IBSS matrix-vector products is stored in single precision by default
  (`x_dtype="float32"`), which makes those products about 3x faster. Means, scales, d and X'y are computed in
  double precision, and results match float64 to ~1e-4 in PIP. Pass `x_dtype="float64"` for a
  double-precision copy. `compute_sufficient_stats` always uses float64.

### Performance
- A fused numba kernel computes the single-effect log marginal likelihood used by the prior-variance search
  (NumPy fallback without numba). It is identical to the previous computation to ~1e-15 and about 2.5x faster.
- Within an IBSS sweep, consecutive effects with all-zero coefficients now reuse one identical update instead
  of recomputing it, and skip X @ b. The output is bit-identical. Fits typically use a few of the L effects,
  so this is about 2.3x faster per fit, and a larger L costs little.
- Leaner dense preprocessing: column sums of squares without a full-size temporary (bit-identical).

### Added
- `SuSiE(x_dtype=...)` option.
- `pysusie.__version__`.
- README: GitHub installation (the PyPI name `pysusie` belongs to an unrelated package) and the
  susieR-equivalent settings.

## 0.1.0

Initial implementation: individual-level, sufficient-statistic and summary-statistic SuSiE; credible sets
with purity filtering; refinement; SuSiE-inf / SuSiE-ash; Mr.ASH; trend filtering.
